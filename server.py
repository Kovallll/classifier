import grpc
from concurrent import futures
import categorizer_pb2
import categorizer_pb2_grpc
import fasttext
import psycopg2
import psycopg2.extras
import os
import re
import logging
import tempfile
import hashlib
import io
from datetime import datetime
from threading import Lock
import time
import json
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация
DB_URL = os.getenv('DATABASE_URL', 'postgres://user:pass@db:5432/mydb')

class DatabaseReader:
    def __init__(self, db_url):
        self.db_url = db_url
        self.last_trained_at = datetime(1970, 1, 1)  # В памяти, не в файле
    
    def _get_conn(self):
        return psycopg2.connect(self.db_url)
    
    def get_all_categories(self):
        """Получение всех категорий с примерами"""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, icon, color 
                    FROM categories 
                    ORDER BY name
                """)
                categories = []
                for row in cur.fetchall():
                    cat = dict(row)
                    cur.execute("""
                        SELECT text, created_at FROM examples 
                        WHERE category_id = %s
                    """, (cat['id'],))
                    rows = cur.fetchall()
                    cat['examples'] = [r['text'] for r in rows]
                    cat['created_ats'] = [r['created_at'] for r in rows]
                    categories.append(cat)
                return categories
        finally:
            conn.close()
    
    def get_examples_count_since(self, since: datetime) -> int:
        """Сколько новых примеров"""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM examples WHERE created_at > %s", (since,))
                return cur.fetchone()[0]
        finally:
            conn.close()


class FastTextCategorizerServicer:
    def __init__(self):
        self.db = DatabaseReader(DB_URL)
        self.model = None
        self.is_training = False
        self.training_lock = Lock()
        self.categories_cache = []
        self.training_data = []  # Храним в памяти вместо файла
        self.last_trained_hash = set()  # Хеши уже обученных примеров
        
        # Параметры
        self.lr = 0.5
        self.word_ngrams = 2
        self.dim = 25
        self.epoch = 25
        self.bucket = 100000
        self.incremental_epoch = 5
        self.thread = 1
        
        self._init_model()
        self._start_watcher()
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста"""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'\d+[\s]*[₽руб$€]?', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())
    
    def _generate_training_lines(self, categories) -> list[str]:
        """Генерация строк обучения в памяти"""
        lines = []
        for cat in categories:
            for example in cat.get('examples', []):
                clean = self._clean_text(example)
                if clean:
                    line = f"__label__{cat['id']} {clean}"
                    lines.append(line)
        return lines
    
    def _train_model_from_lines(self, lines: list[str], epoch: int = None) -> bool:
        """Обучение модели из списка строк в памяти (без файлов)"""
        if not lines:
            logger.error("❌ Нет данных для обучения")
            return False
        
        # Создаём временный файл только для FastText (библиотека требует файл)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write('\n'.join(lines))
            temp_path = f.name
        
        try:
            self.model = fasttext.train_supervised(
                input=temp_path,
                lr=self.lr,
                epoch=epoch or self.epoch,
                wordNgrams=self.word_ngrams,
                bucket=self.bucket,
                thread=self.thread,
                dim=self.dim,
                loss='softmax'
            )
            return True
        finally:
            # Удаляем временный файл
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def _full_train(self) -> bool:
        """Полное обучение на всех данных"""
        with self.training_lock:
            self.is_training = True
            try:
                categories = self.db.get_all_categories()
                if not categories:
                    logger.warning("⚠️ Нет категорий в БД!")
                    return False
                
                lines = self._generate_training_lines(categories)
                if not lines:
                    logger.error("❌ Нет валидных примеров для обучения!")
                    return False
                
                # Обновляем хеши обученных примеров
                self.training_data = lines
                self.last_trained_hash = {hashlib.md5(line.encode()).hexdigest() for line in lines}
                
                # Находим максимальную дату создания примера
                max_created = datetime(1970, 1, 1)
                for cat in categories:
                    for created_at in cat.get('created_ats', []):
                        if created_at and created_at > max_created:
                            max_created = created_at
                
                logger.info(f"📚 Полное обучение: {len(lines)} примеров, {len(categories)} категорий")
                
                success = self._train_model_from_lines(lines, self.epoch)
                if success:
                    self.categories_cache = categories
                    self.db.last_trained_at = max_created if max_created != datetime(1970, 1, 1) else datetime.now()
                    logger.info(f"✅ Полное обучение завершено, last_trained_at: {self.db.last_trained_at}")
                
                return success
                
            except Exception as e:
                logger.error(f"❌ Ошибка при полном обучении: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                self.is_training = False
    
    def _incremental_train(self) -> bool:
        """Инкрементальное обучение только на новых данных"""
        with self.training_lock:
            self.is_training = True
            try:
                categories = self.db.get_all_categories()
                if not categories:
                    logger.warning("⚠️ Нет категорий в БД!")
                    return False
                
                # Получаем ВСЕ примеры, но фильтруем только новые по хешу
                all_lines = self._generate_training_lines(categories)
                new_lines = []
                max_created = self.db.last_trained_at
                
                for cat in categories:
                    for example, created_at in zip(cat.get('examples', []), cat.get('created_ats', [])):
                        clean = self._clean_text(example)
                        if clean:
                            line = f"__label__{cat['id']} {clean}"
                            line_hash = hashlib.md5(line.encode()).hexdigest()
                            if line_hash not in self.last_trained_hash:
                                new_lines.append(line)
                                self.last_trained_hash.add(line_hash)
                                # Отслеживаем максимальную дату
                                if created_at and created_at > max_created:
                                    max_created = created_at
                
                if not new_lines:
                    logger.info("✅ Нет новых данных для обучения")
                    # Обновляем время, чтобы не проверять снова
                    self.db.last_trained_at = datetime.now()
                    return False
                
                # Объединяем старые и новые данные
                combined_lines = self.training_data + new_lines
                self.training_data = combined_lines
                
                logger.info(f"📈 Инкрементальное обучение: {len(new_lines)} новых примеров (всего: {len(combined_lines)})")
                
                success = self._train_model_from_lines(combined_lines, self.incremental_epoch)
                if success:
                    self.categories_cache = categories
                    self.db.last_trained_at = max_created if max_created > self.db.last_trained_at else datetime.now()
                    logger.info(f"✅ Инкрементальное обучение завершено, last_trained_at: {self.db.last_trained_at}")
                
                return success
                
            except Exception as e:
                logger.error(f"❌ Ошибка при инкрементальном обучении: {e}")
                import traceback
                traceback.print_exc()
                return False
            finally:
                self.is_training = False
    
    def _init_model(self):
        """Инициализация модели"""
        # Всегда делаем полное обучение при старте (нет сохраненной модели)
        logger.info("🆕 Инициализация: полное обучение...")
        success = self._full_train()
        if not success:
            logger.error("❌ Не удалось выполнить начальное обучение!")
            # Создаём пустую модель-заглушку, чтобы сервер не упал
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Создаёт минимальную модель, чтобы сервер мог работать"""
        try:
            dummy_lines = ["__label__1 test example"]
            self._train_model_from_lines(dummy_lines, 1)
            logger.warning("⚠️ Создана временная заглушка модели")
        except Exception as e:
            logger.error(f"❌ Не удалось создать заглушку: {e}")
    
    def _start_watcher(self):
        """Проверяет новые данные раз в 30 секунд"""
        def watch():
            # Ждём первичной инициализации
            time.sleep(5)
            while True:
                time.sleep(30)
                try:
                    if self.is_training:
                        continue
                    
                    new_count = self.db.get_examples_count_since(self.db.last_trained_at)
                    if new_count > 0:
                        logger.info(f"🔄 Watcher: {new_count} новых примеров, запуск обучения...")
                        self._incremental_train()
                    else:
                        logger.debug("✅ Нет новых примеров")
                        
                except Exception as e:
                    logger.error(f"Ошибка watcher: {e}")
        
        threading.Thread(target=watch, daemon=True).start()
        logger.info("👁️ Watcher запущен (проверка каждые 30с)")
    
    # ============ gRPC методы ============
    
    def Predict(self, request, context):
        """Предсказание категории"""
        if self.is_training:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Модель обучается, подождите")
            return categorizer_pb2.PredictResponse()
        
        if not self.model:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Модель не загружена")
            return categorizer_pb2.PredictResponse()
        
        clean = self._clean_text(request.text)
        if not clean:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Пустой текст")
            return categorizer_pb2.PredictResponse()
        
        try:
            labels, probs = self.model.predict(clean, k=3)
            
            alternatives = []
            for label, prob in zip(labels, probs):
                cat_id = int(label.replace('__label__', ''))
                
                cat_meta = next(
                    (c for c in self.categories_cache if c['id'] == cat_id),
                    {'name': str(cat_id), 'icon': '❓', 'color': '#CCCCCC'}
                )
                
                alternatives.append(categorizer_pb2.PredictionResult(
                    category_id=str(cat_id),
                    category_name=cat_meta['name'],
                    category_icon=cat_meta['icon'],
                    category_color=cat_meta['color'],
                    confidence=float(prob)
                ))
            
            primary = alternatives[0] if alternatives else None
            
            return categorizer_pb2.PredictResponse(
                primary=primary,
                alternatives=alternatives[1:],
                needs_confirmation=(primary.confidence < 0.7) if primary else True,
                source='fasttext'
            )
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return categorizer_pb2.PredictResponse()
    
    def ForceRetrain(self, request, context):
        """Принудительное обучение"""
        force_full = request.full
        
        if force_full:
            # Сбрасываем хеши для полного переобучения
            self.last_trained_hash = set()
            self.training_data = []
            self.db.last_trained_at = datetime(1970, 1, 1)
            success = self._full_train()
            msg = "Полное переобучение выполнено" if success else "Ошибка полного обучения"
        else:
            success = self._incremental_train()
            msg = "Инкрементальное обучение выполнено" if success else "Нет новых данных"
        
        return categorizer_pb2.StatusResponse(
            success=success,
            message=msg,
            categories_count=len(self.categories_cache),
            is_training=self.is_training
        )
    
    def GetStatus(self, request, context):
        """Статус сервиса"""
        return categorizer_pb2.StatusResponse(
            success=True,
            message="Сервис работает",
            categories_count=len(self.categories_cache),
            is_training=self.is_training
        )
    
    def GetModelInfo(self, request, context):
        """Информация о модели"""
        info = {
            'last_trained_at': self.db.last_trained_at.isoformat(),
            'examples_count': len(self.training_data),
            'categories_count': len(self.categories_cache),
            'unique_hashes': len(self.last_trained_hash),
            'params': {
                'lr': self.lr,
                'epoch': self.epoch,
                'wordNgrams': self.word_ngrams,
                'dim': self.dim
            }
        }
        
        return categorizer_pb2.ModelInfoResponse(
            model_path='in-memory',
            data_hash=str(len(self.last_trained_hash)),
            categories_count=len(self.categories_cache),
            is_training=self.is_training,
            metadata=json.dumps(info)
        )


def serve():
    """Запуск gRPC сервера"""
    port = os.getenv('PORT', '50051')
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    servicer = FastTextCategorizerServicer()
    categorizer_pb2_grpc.add_ExpenseCategorizerServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'0.0.0.0:{port}')
    server.start()
    
    logger.info(f"🚀 gRPC сервер запущен на порту {port}")
    logger.info(f"📊 PostgreSQL: {DB_URL.replace('pass', '***') if 'pass' in DB_URL else '***'}")
    
    server.wait_for_termination()


if __name__ == '__main__':
    serve()