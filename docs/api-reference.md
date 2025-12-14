
## **API Reference**


Автоматически сгенерированная документация по модулям проекта.

## Модули

### src.models.training

::: src.models.training

### src.models.evaluation

::: src.models.evaluation

### src.data.ingestion

::: src.data.ingestion

### src.data.preprocessing

::: src.data.preprocessing

### src.features.engineering

::: src.features.engineering

### src.utils.config

::: src.utils.config

### src.utils.logger

::: src.utils.logger

## Конфигурация

### Формат конфигурационного файла

{
  "data": {
    "raw_path": "data/raw/uber.csv",
    "processed_dir": "data/processed"
  },
  "features": {
    "selected": ["passenger_count", "trip_distance"],
    "engineered": ["hour", "day_of_week"]
  },
  "target": "fare_amount",
  "model": {
    "output_path": "models/model.pkl",
    "random_state": 42,
    "test_size": 0.2
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}