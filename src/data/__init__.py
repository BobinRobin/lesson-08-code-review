"""
Модуль для работы с данными.
Содержит функции для загрузки и предобработки данных.
"""

from .ingestion import load_data
from .preprocessing import preprocess_data

__all__ = [
    "load_data",
    "preprocess_data",
]
