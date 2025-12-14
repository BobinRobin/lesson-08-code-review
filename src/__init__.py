"""
Пакет для предсказания стоимости поездок в такси.
Основные модули:
- data: загрузка и обработка данных
- features: создание признаков
- models: обучение и оценка моделей
- utils: утилиты и конфигурация
"""

__version__ = "1.0.0"
__author__ = "Taxi Fare Prediction Team"

from .data.ingestion import load_data
from .data.preprocessing import preprocess_data
from .features.engineering import add_time_features
from .models.training import TaxiFareModel, train_model
from .models.evaluation import evaluate_model

__all__ = [
    "load_data",
    "preprocess_data",
    "add_time_features",
    "TaxiFareModel",
    "train_model",
    "evaluate_model",
]
