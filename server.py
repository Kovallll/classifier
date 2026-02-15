# ml-service/server.py
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
from datetime import datetime
from threading import Lock
import time
import hashlib
import json
import threading  # ← ДОБАВИТЬ (используется в _start_watcher)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация
DB_URL = os.getenv('DATABASE_URL', 'postgres://user:pass@db:5432/mydb')
MODEL_DIR = '/app/models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.bin')
DATA_PATH = os.path.join(MODEL_DIR, 'train.txt')
META_PATH = os.path.join(MODEL_DIR, 'metadata.json')

os.makedirs(MODEL_DIR, exist_ok=True)


class DatabaseReader:
    def __init__(self, db_url):
        self.db_url = db_url
        self.last_trained_at = self._load_last_trained()
    
    def _load_last_trained(self) -> datetime:
        """Загружает время последнего обучения из файла"""
        meta_path = os.path.join(MODEL_DIR, 'training_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data.get('last_trained_at', '1970-01-01T00:00:00'))
        return datetime(1970, 1, 1)
    
    def save_last_trained(self, dt: datetime):
        """Сохраняет время обучения"""
        meta_path = os.path.join(MODEL_DIR, 'training_meta.json')
        data = {
            'last_trained_at': dt.isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_conn(self):
        """← ДОБАВИТЬ: создание подключения к БД"""
        return psycopg2.connect(self.db_url)
    
    def get_all_categories(self):
        """← ДОБАВИТЬ: получение всех категорий (используется в _full_train)"""
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
                        SELECT text FROM examples 
                        WHERE category_id = %s
                    """, (cat['id'],))
                    cat['examples'] = [r['text'] for r in cur.fetchall()]
                    categories.append(cat)
                return categories
        finally:
            conn.close()
    
    def get_new_examples(self, since: datetime) -> list[dict]:  # ← ИСПРАВИТЬ: list[dict] вместо List[Dict]
        """Получает только новые примеры с момента last_trained_at"""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT e.category_id, e.text, c.name, c.icon, c.color
                    FROM examples e
                    JOIN categories c ON e.category_id = c.id
                    WHERE e.created_at > %s
                    ORDER BY e.created_at
                """, (since,))
                return [dict(row) for row in cur.fetchall()]
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
        
        # Параметры
        self.lr = 0.5  # ← ДОБАВИТЬ: learning rate (используется в обучении)
        self.word_ngrams = 2
        self.dim = 100
        self.epoch = 100
        self.incremental_epoch = 5
        
        self._init_model()
        self._start_watcher()
    
    def _clean_text(self, text):  # ← ДОБАВИТЬ: используется в _incremental_train и Predict
        """Очистка текста"""
        text = text.lower().strip()
        text = re.sub(r'\d+[\s]*[₽руб$€]?', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())
    
    def _generate_training_file(self, categories):  # ← ДОБАВИТЬ: используется в _full_train
        """Генерация файла обучения"""
        lines = []
        for cat in categories:
            for example in cat.get('examples', []):
                clean = self._clean_text(example)
                if clean:
                    lines.append(f"__label__{cat['id']} {clean}")
        
        with open(DATA_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return len(lines)
    
    def _load_model(self):  # ← ДОБАВИТЬ: используется в _init_model
        """Загрузка модели из файла"""
        try:
            self.model = fasttext.load_model(MODEL_PATH)
            self.categories_cache = self.db.get_all_categories()
            logger.info(f"✅ Модель загружена, категорий: {len(self.categories_cache)}")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            raise
    
    def _init_model(self):
        """Инициализация: полное обучение если нет модели, иначе загрузка"""
        if os.path.exists(MODEL_PATH):
            self._load_model()
            new_count = self.db.get_examples_count_since(self.db.last_trained_at)
            if new_count > 0:
                logger.info(f"📬 Обнаружено {new_count} новых примеров с {self.db.last_trained_at}")
                self._incremental_train()
        else:
            logger.info("🆕 Первая инициализация, полное обучение...")
            self._full_train()
    
    def _full_train(self):
        """Полное обучение на всех данных"""
        with self.training_lock:
            self.is_training = True
            try:
                categories = self.db.get_all_categories()
                if not categories:
                    logger.warning("⚠️ Нет категорий в БД!")
                    return False
                
                count = self._generate_training_file(categories)
                logger.info(f"📚 Полное обучение: {count} примеров")
                
                self.model = fasttext.train_supervised(
                    input=DATA_PATH,
                    lr=self.lr,
                    epoch=self.epoch,
                    wordNgrams=self.word_ngrams,
                    dim=self.dim,
                    loss='softmax'
                )
                
                self._save_model_and_meta(categories, count, "full")
                return True
            finally:
                self.is_training = False
    
    def _incremental_train(self):
        """Инкрементальное обучение только на новых данных"""
        with self.training_lock:
            self.is_training = True
            try:
                new_examples = self.db.get_new_examples(self.db.last_trained_at)
                if not new_examples:
                    logger.info("✅ Нет новых данных для обучения")
                    return False
                
                lines = []
                for ex in new_examples:
                    clean = self._clean_text(ex['text'])
                    if clean:
                        lines.append(f"__label__{ex['category_id']} {clean}")
                
                temp_path = os.path.join(MODEL_DIR, 'incremental_train.txt')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                logger.info(f"📈 Инкрементальное обучение: {len(lines)} новых примеров")
                
                # Объединяем старые + новые данные
                combined_path = os.path.join(MODEL_DIR, 'combined_train.txt')
                with open(DATA_PATH, 'r', encoding='utf-8') as f_old, \
                     open(temp_path, 'r', encoding='utf-8') as f_new, \
                     open(combined_path, 'w', encoding='utf-8') as f_out:
                    old_data = f_old.read()
                    new_data = f_new.read()
                    f_out.write(old_data + ('\n' if old_data and new_data else '') + new_data)
                
                self.model = fasttext.train_supervised(
                    input=combined_path,
                    lr=self.lr,
                    epoch=self.incremental_epoch,
                    wordNgrams=self.word_ngrams,
                    dim=self.dim,
                    loss='softmax'
                )
                
                os.replace(combined_path, DATA_PATH)
                
                self._save_model_and_meta(
                    self.db.get_all_categories(), 
                    len(lines), 
                    "incremental"
                )
                
                return True
                
            finally:
                self.is_training = False
    
    def _save_model_and_meta(self, categories, count, train_type):
        """Сохраняет модель и метаданные"""
        self.model.save_model(MODEL_PATH)
        self.categories_cache = categories
        
        now = datetime.now()
        self.db.save_last_trained(now)
        
        with open(META_PATH, 'w') as f:
            json.dump({
                'trained_at': now.isoformat(),
                'train_type': train_type,
                'examples_count': count,
                'categories_count': len(categories),
                'params': {
                    'lr': self.lr,
                    'epoch': self.epoch if train_type == 'full' else self.incremental_epoch,
                    'wordNgrams': self.word_ngrams,
                    'dim': self.dim
                }
            }, f, indent=2)
        
        logger.info(f"✅ Модель сохранена ({train_type})")
    
    def _start_watcher(self):
        """Проверяет новые данные раз в 30 секунд"""
        def watch():
            while True:
                time.sleep(30)
                try:
                    if self.is_training:
                        continue
                    
                    new_count = self.db.get_examples_count_since(self.db.last_trained_at)
                    if new_count > 5:
                        logger.info(f"🔄 Watcher: {new_count} новых примеров, запуск обучения...")
                        self._incremental_train()
                        
                except Exception as e:
                    logger.error(f"Ошибка watcher: {e}")
        
        threading.Thread(target=watch, daemon=True).start()
        logger.info("👁️ Watcher запущен (проверка каждые 30с)")
    
    # ============ gRPC методы ============
    
    def Predict(self, request, context):
        """Предсказание категории"""
        if self.is_training:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Модель обучается, подождите 5 секунд")
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
                cat_id = label.replace('__label__', '')
                
                cat_meta = next(
                    (c for c in self.categories_cache if c['id'] == cat_id),
                    {'name': cat_id, 'icon': '❓', 'color': '#CCCCCC'}
                )
                
                alternatives.append(categorizer_pb2.PredictionResult(
                    category_id=cat_id,
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
            success = self._full_train()
            msg = "Полное переобучение выполнено"
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
        info = {}
        if os.path.exists(META_PATH):
            with open(META_PATH, 'r') as f:
                info = json.load(f)
        
        return categorizer_pb2.ModelInfoResponse(
            model_path=MODEL_PATH,
            data_hash='',  # ← УБРАТЬ: self.last_data_hash не существует
            categories_count=len(self.categories_cache),
            is_training=self.is_training,
            metadata=json.dumps(info)
        )


def serve():
    """Запуск gRPC сервера"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    servicer = FastTextCategorizerServicer()
    categorizer_pb2_grpc.add_ExpenseCategorizerServicer_to_server(servicer, server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
    
    logger.info("🚀 gRPC сервер запущен на порту 50051")
    logger.info(f"📊 PostgreSQL: {DB_URL.replace('pass', '***')}")
    
    server.wait_for_termination()


if __name__ == '__main__':
    time.sleep(3)
    serve()