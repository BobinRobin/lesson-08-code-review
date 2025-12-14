"""Скрипт для обучения модели."""

import logging
import pickle
import json
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.features.engineering import add_time_features
from src.models.training import train_model
from src.models.evaluation import evaluate_model, save_metrics

logger = logging.getLogger(__name__)


def main() -> None:
    """Основной пайплайн обучения."""
    logger.info("Начало обучения модели")
    
    try:
        # Загрузка конфигурации
        config = load_config()
        
        # Настройка логгера
        setup_logger(config)
        
        # Загрузка данных
        data_path = config["data"]["raw_path"]
        logger.info(f"Загрузка данных из {data_path}")
        df = load_data(data_path)
        
        # Предобработка
        logger.info("Предобработка данных")
        df = preprocess_data(df)
        
        # Feature engineering
        if "pickup_datetime" in df.columns:
            logger.info("Извлечение временных признаков")
            df = add_time_features(df)
        
        # Подготовка признаков и таргета
        features = config["features"]["selected"] + config["features"]["engineered"]
        target = config["target"]
        
        # Проверяем наличие признаков в данных
        available_features = [f for f in features if f in df.columns]
        missing_features = set(features) - set(available_features)
        
        if missing_features:
            logger.warning(f"Отсутствуют признаки: {missing_features}")
        
        if not available_features:
            logger.error("Ни один из указанных признаков не найден в данных")
            raise ValueError("Нет доступных признаков для обучения")
        
        X = df[available_features]
        y = df[target]
        
        logger.info(f"Признаки для обучения: {available_features}")
        logger.info(f"Размер данных: {len(X)} строк")
        
        # Обучение модели
        logger.info("Обучение модели")
        model, X_test, y_test = train_model(
            X,
            y,
            random_state=config["model"]["random_state"],
            test_size=config["model"]["test_size"],
            use_grid_search=False  # Можно включить в конфиге
        )
        
        # Оценка модели
        logger.info("Оценка модели")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Сохранение метрик
        metrics_path = Path(config["model"]["output_path"]).parent / "metrics.json"
        save_metrics(metrics, str(metrics_path))
        
        # Сохранение модели
        model_path = config["model"]["output_path"]
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Модель сохранена в {model_path}")
        logger.info(f"Метрики сохранены в {metrics_path}")
        
        # Сохранение информации о признаках
        feature_info = {
            "features_used": available_features,
            "target": target,
            "data_shape": {
                "train_rows": len(X) - len(X_test),
                "test_rows": len(X_test)
            }
        }
        
        feature_info_path = Path(model_path).parent / "feature_info.json"
        with open(feature_info_path, "w") as f:
            json.dump(feature_info, f, indent=4)
        
        logger.info("Обучение завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()