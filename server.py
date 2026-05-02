import fasttext
import psycopg2
import psycopg2.extras
import os
import re
import logging
import tempfile
import hashlib
from datetime import datetime
from threading import Lock
import time
import json
import threading
import uuid
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация
DB_URL = os.getenv('DATABASE_URL', 'postgres://user:pass@db:5432/mydb')
PORT = int(os.getenv('PORT', '8080'))

app = Flask(__name__)


class DatabaseReader:
    def __init__(self, db_url):
        self.db_url = db_url
        self.last_trained_at = datetime(1970, 1, 1)
    
    def _get_conn(self):
        return psycopg2.connect(self.db_url)
    
    def get_all_categories(self):
        """Получение всех категорий с примерами"""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                      c.id,
                      c.name,
                      c.icon,
                      c.color,
                      COALESCE(
                        json_agg(
                          json_build_object('text', e.text, 'created_at', e.created_at)
                          ORDER BY e.created_at
                        ) FILTER (WHERE e.text IS NOT NULL),
                        '[]'::json
                      ) AS examples_data
                    FROM categories c
                    LEFT JOIN examples e ON e.category_id = c.id
                    GROUP BY c.id, c.name, c.icon, c.color
                    ORDER BY c.name
                """)
                categories = []
                for row in cur.fetchall():
                    cat = dict(row)
                    data = cat.get('examples_data') or []
                    cat['examples'] = [x.get('text') for x in data if x.get('text')]
                    cat['created_ats'] = [x.get('created_at') for x in data if x.get('created_at')]
                    cat.pop('examples_data', None)
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


class CategorizerService:
    def __init__(self):
        self.db = DatabaseReader(DB_URL)
        self.model = None
        self.is_training = False
        self.training_lock = Lock()
        self.categories_cache = []
        self.training_data = []
        self.last_trained_hash = set()
        self.last_retrain_requested_at = 0.0
        self.pending_full_retrain = False
        self.min_retrain_interval_sec = int(os.getenv('RETRAIN_MIN_INTERVAL_SEC', '15'))
        
        # Параметры FastText
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

    def _to_datetime(self, value):
        """Безопасно приводит created_at (str|datetime|None) к datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            # PostgreSQL JSON может отдавать ISO со смещением/Z
            if s.endswith('Z'):
                s = s[:-1] + '+00:00'
            try:
                return datetime.fromisoformat(s)
            except Exception:
                return None
        return None

    def _new_trace_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _preview_text(self, text: str, limit: int = 120) -> str:
        if text is None:
            return ""
        s = str(text).replace('\n', ' ').replace('\r', ' ')
        return s[:limit] + ('…' if len(s) > limit else '')

    def _log_predict_stage(self, trace_id: str, stage: str, **kwargs):
        payload = {'trace_id': trace_id, 'stage': stage, **kwargs}
        try:
            logger.info("classifier.trace %s", json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.info("classifier.trace trace_id=%s stage=%s %s", trace_id, stage, kwargs)
    
    def _generate_training_lines(self, categories) -> list[str]:
        """Генерация строк обучения в памяти.
        Включаем очищенное имя категории — иначе совпадение только по примерам из examples,
        а лексикон на бэкенде не связывает разные языки (напр. «одежда» и «Shopping»).
        """
        lines = []
        for cat in categories:
            cid = cat['id']
            name_clean = self._clean_text(cat.get('name') or '')
            if name_clean:
                lines.append(f"__label__{cid} {name_clean}")
            for example in cat.get('examples', []):
                clean = self._clean_text(example)
                if clean:
                    line = f"__label__{cid} {clean}"
                    lines.append(line)
        return lines

    @staticmethod
    def _label_id_matches_category_id(label_raw: str, cat_id) -> bool:
        """Совпадение id из метки FastText и id категории в БД (UUID — без учёта регистра)."""
        a = str(label_raw).strip()
        b = str(cat_id).strip()
        if len(a) >= 32 and len(b) >= 32 and a.count('-') >= 4 and b.count('-') >= 4:
            return a.lower() == b.lower()
        return a == b

    def _train_model_from_lines(self, lines: list[str], epoch: int = None) -> bool:
        """Обучение модели из списка строк в памяти"""
        if not lines:
            logger.error("❌ Нет данных для обучения")
            return False
        
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
                
                self.training_data = lines
                self.last_trained_hash = {hashlib.md5(line.encode()).hexdigest() for line in lines}
                
                max_created = datetime(1970, 1, 1)
                for cat in categories:
                    for created_at in cat.get('created_ats', []):
                        dt = self._to_datetime(created_at)
                        if dt and dt > max_created:
                            max_created = dt
                
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
                                dt = self._to_datetime(created_at)
                                if dt and dt > max_created:
                                    max_created = dt
                
                if not new_lines:
                    logger.info("✅ Нет новых данных для обучения")
                    self.db.last_trained_at = datetime.now()
                    return False
                
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
        logger.info("🆕 Инициализация: полное обучение...")
        success = self._full_train()
        if not success:
            logger.error("❌ Не удалось выполнить начальное обучение!")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Создаёт минимальную модель, чтобы сервер мог работать"""
        try:
            dummy_lines = ["__label__unknown test example"]
            self._train_model_from_lines(dummy_lines, 1)
            logger.warning("⚠️ Создана временная заглушка модели")
        except Exception as e:
            logger.error(f"❌ Не удалось создать заглушку: {e}")
    
    def _start_watcher(self):
        """Проверяет новые данные раз в 30 секунд"""
        def watch():
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
                        
                except Exception as e:
                    logger.error(f"Ошибка watcher: {e}")
        
        threading.Thread(target=watch, daemon=True).start()
        logger.info("👁️ Watcher запущен (проверка каждые 30с)")
    
    def predict(self, text: str, trace_id: str = None) -> dict:
        """Предсказание категории"""
        trace_id = trace_id or self._new_trace_id()
        self._log_predict_stage(
            trace_id,
            "predict_in",
            raw_text=self._preview_text(text),
            is_training=self.is_training,
            categories_cache_count=len(self.categories_cache),
            training_examples_count=len(self.training_data),
        )

        if self.is_training:
            self._log_predict_stage(trace_id, "blocked_training")
            return {
                'success': False,
                'error': 'Модель обучается, подождите',
                'is_training': True
            }
        
        if not self.model:
            self._log_predict_stage(trace_id, "blocked_no_model")
            return {
                'success': False,
                'error': 'Модель не загружена'
            }

        # Если кеш категорий пустой (например, после неудачного старта),
        # пытаемся подгрузить категории из БД перед предсказанием.
        if not self.categories_cache:
            try:
                self.categories_cache = self.db.get_all_categories()
                self._log_predict_stage(
                    trace_id,
                    "categories_cache_reload",
                    loaded_count=len(self.categories_cache),
                )
            except Exception as e:
                logger.error(f"Не удалось загрузить категории перед predict: {e}")
                self._log_predict_stage(trace_id, "categories_cache_reload_error", error=str(e))

        if not self.categories_cache:
            self._log_predict_stage(trace_id, "blocked_no_categories_cache")
            return {
                'success': False,
                'error': 'Категории не загружены, предсказание недоступно'
            }
        
        clean = self._clean_text(text)
        self._log_predict_stage(
            trace_id,
            "text_cleaned",
            clean_text=self._preview_text(clean),
            clean_len=len(clean or ''),
        )
        if not clean:
            logger.warning("classifier.predict: пустой текст после _clean_text, raw=%r", text[:200] if text else '')
            self._log_predict_stage(trace_id, "blocked_empty_after_clean")
            return {
                'success': False,
                'error': 'Пустой текст'
            }

        cache_n = len(self.categories_cache)
        cache_id_sample = [str(c['id']) for c in self.categories_cache[:8]]
        logger.info(
            "classifier.predict: start clean=%r cache_categories=%s sample_ids=%s",
            clean,
            cache_n,
            cache_id_sample,
        )
        self._log_predict_stage(
            trace_id,
            "model_predict_start",
            clean_text=self._preview_text(clean),
            cache_categories=cache_n,
            sample_ids=cache_id_sample,
        )

        try:
            labels, probs = self.model.predict(clean, k=3)
            # Приведение к спискам Python, чтобы избежать ошибки NumPy
            # "Unable to avoid copy while creating an array as requested" (на части хостингов)
            labels = [str(x) for x in labels]
            probs = [float(x) for x in probs]

            logger.info(
                "classifier.predict: fasttext raw_labels=%s",
                [(labels[i], round(probs[i], 4)) for i in range(len(labels))],
            )
            self._log_predict_stage(
                trace_id,
                "model_predict_raw",
                labels=[str(x) for x in labels],
                probs=[round(float(x), 6) for x in probs],
            )

            alternatives = []
            for rank, (label, prob) in enumerate(zip(labels, probs), start=1):
                # id категории может быть int или UUID (строка) — не приводим к int
                raw_id = str(label).replace('__label__', '').strip()
                cat_meta = next(
                    (c for c in self.categories_cache if self._label_id_matches_category_id(raw_id, c['id'])),
                    None,
                )
                is_unknown = cat_meta is None
                if is_unknown:
                    logger.warning(
                        "classifier.predict: rank=%s label=%r raw_id=%r prob=%.4f -> NO_MATCH in categories_cache "
                        "(проверьте, что id в модели совпадает с id в БД; переобучение)",
                        rank,
                        label,
                        raw_id,
                        prob,
                    )
                    cat_meta = {'name': 'Неизвестно', 'icon': '❓', 'color': '#CCCCCC'}
                    self._log_predict_stage(
                        trace_id,
                        "label_no_match",
                        rank=rank,
                        label=label,
                        raw_id=raw_id,
                        prob=round(float(prob), 6),
                    )
                else:
                    logger.info(
                        "classifier.predict: rank=%s raw_id=%r -> name=%r prob=%.4f",
                        rank,
                        raw_id,
                        cat_meta.get('name'),
                        prob,
                    )
                    self._log_predict_stage(
                        trace_id,
                        "label_match",
                        rank=rank,
                        label=label,
                        raw_id=raw_id,
                        category_name=cat_meta.get('name'),
                        prob=round(float(prob), 6),
                    )

                alternatives.append({
                    'category_id': raw_id if not is_unknown else '',
                    'category_name': cat_meta['name'],
                    'category_icon': cat_meta['icon'],
                    'category_color': cat_meta['color'],
                    'confidence': prob if not is_unknown else min(float(prob), 0.49)
                })
            
            primary = alternatives[0] if alternatives else None
            if primary:
                logger.info(
                    "classifier.predict: primary_out id=%r name=%r conf=%.4f needs_confirm=%s",
                    primary.get('category_id'),
                    primary.get('category_name'),
                    float(primary.get('confidence') or 0),
                    (not primary.get('category_id'))
                    or (float(primary.get('confidence') or 0) < 0.7),
                )
                self._log_predict_stage(
                    trace_id,
                    "primary_selected",
                    category_id=primary.get('category_id'),
                    category_name=primary.get('category_name'),
                    confidence=round(float(primary.get('confidence') or 0), 6),
                    needs_confirmation=(
                        (not primary.get('category_id'))
                        or (float(primary.get('confidence') or 0) < 0.7)
                    ),
                )

            response = {
                'success': True,
                'primary': primary,
                'alternatives': alternatives[1:],
                'needs_confirmation': (
                    (not primary)
                    or (not primary['category_id'])
                    or (primary['confidence'] < 0.7)
                ),
                'source': 'fasttext'
            }
            self._log_predict_stage(
                trace_id,
                "predict_out",
                success=True,
                source=response.get('source'),
                needs_confirmation=response.get('needs_confirmation'),
                primary_id=(primary or {}).get('category_id') if primary else '',
                primary_name=(primary or {}).get('category_name') if primary else '',
                alternatives_count=len(response.get('alternatives') or []),
            )
            return response
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            self._log_predict_stage(trace_id, "predict_error", error=str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def force_retrain(self, full: bool = False) -> dict:
        """Принудительное обучение"""
        now = time.time()
        if now - self.last_retrain_requested_at < self.min_retrain_interval_sec:
            if full:
                self.pending_full_retrain = True
            return {
                'success': True,
                'message': f"Retrain отложен (cooldown {self.min_retrain_interval_sec}s)",
                'categories_count': len(self.categories_cache),
                'is_training': self.is_training,
                'queued_full': self.pending_full_retrain,
            }
        self.last_retrain_requested_at = now

        if full:
            self.last_trained_hash = set()
            self.training_data = []
            self.db.last_trained_at = datetime(1970, 1, 1)
            success = self._full_train()
            msg = "Полное переобучение выполнено" if success else "Ошибка полного обучения"
            self.pending_full_retrain = False
        else:
            if self.pending_full_retrain:
                self.last_trained_hash = set()
                self.training_data = []
                self.db.last_trained_at = datetime(1970, 1, 1)
                success = self._full_train()
                msg = "Выполнено отложенное полное переобучение" if success else "Ошибка полного обучения"
                self.pending_full_retrain = False
            else:
                success = self._incremental_train()
                msg = "Инкрементальное обучение выполнено" if success else "Нет новых данных"
        
        return {
            'success': success,
            'message': msg,
            'categories_count': len(self.categories_cache),
            'is_training': self.is_training,
            'queued_full': self.pending_full_retrain,
        }
    
    def get_status(self) -> dict:
        """Статус сервиса"""
        return {
            'success': True,
            'message': 'Сервис работает',
            'categories_count': len(self.categories_cache),
            'is_training': self.is_training
        }
    
    def get_model_info(self) -> dict:
        """Информация о модели (model_version меняется при переобучении для инвалидации кэша)."""
        version_parts = (
            self.db.last_trained_at.isoformat(),
            str(len(self.training_data)),
            str(len(self.categories_cache)),
        )
        model_version = hashlib.sha256('|'.join(version_parts).encode()).hexdigest()[:16]
        return {
            'success': True,
            'model_version': model_version,
            'last_trained_at': self.db.last_trained_at.isoformat(),
            'examples_count': len(self.training_data),
            'categories_count': len(self.categories_cache),
            'unique_hashes': len(self.last_trained_hash),
            'is_training': self.is_training,
            'params': {
                'lr': self.lr,
                'epoch': self.epoch,
                'wordNgrams': self.word_ngrams,
                'dim': self.dim
            }
        }


# Создаём сервис при старте
service = CategorizerService()


# ============ HTTP Endpoints ============

@app.route('/health', methods=['GET'])
def health():
    """Health check для Render/Fly.io"""
    return jsonify({'status': 'ok', 'is_training': service.is_training})


@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание категории"""
    data = request.get_json()
    trace_id = service._new_trace_id()
    logger.info(
        "classifier.http predict_in trace_id=%s has_text=%s payload_keys=%s",
        trace_id,
        bool(data and 'text' in data),
        sorted(list((data or {}).keys())),
    )
    if not data or 'text' not in data:
        logger.info("classifier.http predict_out trace_id=%s status=400 reason=missing_text", trace_id)
        return jsonify({'success': False, 'error': 'Поле text обязательно'}), 400
    
    result = service.predict(data['text'], trace_id=trace_id)
    status_code = 200 if result.get('success') else (503 if result.get('is_training') else 500)
    logger.info(
        "classifier.http predict_out trace_id=%s status=%s success=%s",
        trace_id,
        status_code,
        bool(result.get('success')),
    )
    return jsonify(result), status_code


@app.route('/retrain', methods=['POST'])
def retrain():
    """Принудительное обучение"""
    data = request.get_json() or {}
    result = service.force_retrain(full=data.get('full', False))
    return jsonify(result)


@app.route('/status', methods=['GET'])
def status():
    """Статус сервиса"""
    return jsonify(service.get_status())


@app.route('/model-info', methods=['GET'])
def model_info():
    """Информация о модели"""
    return jsonify(service.get_model_info())


@app.route('/categories', methods=['GET'])
def get_categories():
    """Получение списка категорий"""
    return jsonify({
        'success': True,
        'categories': [
            {
                'id': c['id'],
                'name': c['name'],
                'icon': c['icon'],
                'color': c['color']
            }
            for c in service.categories_cache
        ]
    })


if __name__ == '__main__':
    logger.info(f"🚀 HTTP сервер запускается на порту {PORT}")
    # Для production используем threaded=True, для Render/Fly.io это важно
    app.run(host='0.0.0.0', port=PORT, threaded=True)