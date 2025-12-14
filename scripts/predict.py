"""Скрипт для выполнения предсказаний."""

import logging
import pickle
import pandas as pd
from pathlib import Path
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.features.engineering import add_time_features

logger = logging.getLogger(__name__)


def main() -> None:
    """Основной пайплайн предсказаний."""
    logger.info("Начало выполнения предсказаний")
    
    try:
        config = load_config()
        setup_logger(config)
        
        # Загрузка данных
        data_path = config.get("prediction", {}).get("data_path", 
                    config["data"]["raw_path"])
        df = load_data(data_path)
        logger.info(f"Загружено {len(df)} строк данных")
        
        # Предобработка
        df = preprocess_data(df)
        logger.info(f"После предобработки: {len(df)} строк")
        
        # Feature engineering
        if "pickup_datetime" in df.columns:
            df = add_time_features(df)
            logger.info("Временные признаки добавлены")
        
        # Подготовка признаков
        features = config["features"]["selected"] + config["features"]["engineered"]
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            logger.error("Ни один из указанных признаков не найден в данных")
            raise ValueError("Нет доступных признаков для предсказания")
        
        X = df[available_features]
        logger.info(f"Используются признаки: {available_features}")
        
        # Загрузка модели
        model_path = config["model"]["output_path"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Модель загружена из {model_path}")
        
        # Предсказание
        predictions = model.predict(X)
        
        # Сохранение результатов
        results_df = pd.DataFrame({
            'predictions': predictions
        })
        
        output_path = Path("predictions/predictions.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Предсказания сохранены в {output_path}")
        logger.info(f"Первые 5 предсказаний: {predictions[:5].tolist()}")
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()