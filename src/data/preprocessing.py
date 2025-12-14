"""Модуль для предобработки данных."""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger(__name__)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предобработку данных о поездках такси.
    
    Аргументы:
        df (pd.DataFrame): Исходный DataFrame с данными о поездках.
    
    Возвращает:
        pd.DataFrame: Очищенный DataFrame с признаками.
        
    Исключения:
        ValueError: Если отсутствуют необходимые столбцы.
    """
    # Создаем копию, чтобы не изменять оригинал
    df_clean = df.copy()
    
    # 1. Фильтрация по стоимости и количеству пассажиров
    if 'fare_amount' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean[df_clean['fare_amount'] > 0]
        filtered_fare = initial_rows - len(df_clean)
        if filtered_fare > 0:
            logger.info(f"Отфильтровано {filtered_fare} строк с fare_amount <= 0")
    
    if 'passenger_count' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean[
            (df_clean['passenger_count'] > 0) & 
            (df_clean['passenger_count'] <= 6)
        ].copy()
        filtered_passengers = initial_rows - len(df_clean)
        if filtered_passengers > 0:
            logger.info(f"Отфильтровано {filtered_passengers} строк с passenger_count вне диапазона 1-6")
    
    # 2. Удаление строк с пропущенными значениями
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        logger.info(f"Удалено {removed_rows} строк с пропущенными значениями")
    
    # 3. Обработка бесконечных значений
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df_clean[col]).any():
            logger.warning(f"В колонке {col} найдены бесконечные значения")
            # Заменяем бесконечности на NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
    
    # 4. Удаление строк, которые стали NaN после замены бесконечностей
    df_clean = df_clean.dropna()
    
    # 5. Создание признака distance (если есть координаты)
    required_cols = ['pickup_longitude', 'pickup_latitude', 
                    'dropoff_longitude', 'dropoff_latitude']
    
    if all(col in df_clean.columns for col in required_cols):
        # Вычисляем евклидово расстояние
        df_clean['distance'] = np.sqrt(
            (df_clean['dropoff_longitude'] - df_clean['pickup_longitude'])**2 +
            (df_clean['dropoff_latitude'] - df_clean['pickup_latitude'])**2
        )
        logger.info("Признак distance создан из координат")
    
    logger.info(f"После очистки осталось {len(df_clean)} строк")
    return df_clean


def create_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает дополнительные признаки для поездок.
    
    Аргументы:
        df (pd.DataFrame): DataFrame с координатами поездок.
    
    Возвращает:
        pd.DataFrame: DataFrame с добавленными признаками.
    """
    df_features = df.copy()
    
    # Создаем признаки направления
    if all(col in df_features.columns for col in 
           ['pickup_longitude', 'pickup_latitude', 
            'dropoff_longitude', 'dropoff_latitude']):
        
        df_features['delta_longitude'] = (
            df_features['dropoff_longitude'] - df_features['pickup_longitude']
        )
        df_features['delta_latitude'] = (
            df_features['dropoff_latitude'] - df_features['pickup_latitude']
        )
    
    return df_features