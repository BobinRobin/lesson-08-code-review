
## **Примеры использования**


В этом разделе представлены практические примеры использования проекта для различных сценариев.

## Базовые примеры

### Обучение модели

# Основной сценарий обучения
# scripts/train.py
```python
import logging
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.features.engineering import add_time_features
from src.models.training import train_model
from src.models.evaluation import evaluate_model, save_metrics

# Загрузка конфигурации
config = load_config("config.json")

# Настройка логгирования
setup_logger(config)

# Загрузка данных
df = load_data(config["data"]["raw_path"])

# Предобработка
df = preprocess_data(df)

# Создание признаков
if "pickup_datetime" in df.columns:
    df = add_time_features(df)

# Подготовка данных
features = config["features"]["selected"] + config["features"]["engineered"]
target = config["target"]
X = df[features]
y = df[target]

# Обучение модели
model, X_test, y_test = train_model(
    X, y,
    random_state=config["model"]["random_state"],
    test_size=config["model"]["test_size"]
)

# Оценка модели
metrics = evaluate_model(model, X_test, y_test)

# Сохранение метрик
save_metrics(metrics, "models/metrics.json")
```