"""Тесты для модуля feature engineering."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features.engineering import add_time_features


def test_add_time_features_creates_correct_columns():
    """Тест создания временных признаков."""
    df = pd.DataFrame({
        'pickup_datetime': [
            '2023-01-01 10:30:00',
            '2023-01-02 15:45:00',
            '2023-01-03 20:15:00'
        ],
        'fare_amount': [10.0, 20.0, 30.0]
    })
    
    result = add_time_features(df)
    
    # Проверяем наличие новых колонок
    assert 'hour' in result.columns
    assert 'day_of_week' in result.columns
    
    # Проверяем удаление исходной колонки
    assert 'pickup_datetime' not in result.columns
    
    # Проверяем значения
    assert result['hour'].iloc[0] == 10
    assert result['day_of_week'].iloc[0] == 6  # Воскресенье


def test_add_time_features_with_missing_column():
    """Тест обработки отсутствия колонки pickup_datetime."""
    df = pd.DataFrame({
        'fare_amount': [10.0, 20.0],
        'passenger_count': [1, 2]
    })
    
    with pytest.raises(KeyError):
        add_time_features(df)


def test_add_time_features_preserves_other_columns():
    """Тест сохранения других колонок при создании признаков."""
    df = pd.DataFrame({
        'pickup_datetime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00'],
        'fare_amount': [10.0, 20.0],
        'passenger_count': [1, 2],
        'trip_distance': [1.5, 2.5]
    })
    
    result = add_time_features(df)
    
    # Проверяем сохранение исходных колонок
    assert 'fare_amount' in result.columns
    assert 'passenger_count' in result.columns
    assert 'trip_distance' in result.columns
    
    # Проверяем добавление новых колонок
    assert 'hour' in result.columns
    assert 'day_of_week' in result.columns


@pytest.mark.parametrize(
    "datetime_str, expected_hour, expected_day",
    [
        ("2023-01-01 00:00:00", 0, 6),  # Воскресенье
        ("2023-01-02 12:30:00", 12, 0),  # Понедельник
        ("2023-01-07 23:59:59", 23, 5),  # Суббота
        ("2023-12-25 18:45:00", 18, 0),  # Понедельник
    ]
)
def test_time_features_calculations(datetime_str, expected_hour, expected_day):
    """Параметризованный тест вычислений временных признаков."""
    df = pd.DataFrame({
        'pickup_datetime': [datetime_str],
        'fare_amount': [10.0]
    })
    
    result = add_time_features(df)
    
    assert result['hour'].iloc[0] == expected_hour
    assert result['day_of_week'].iloc[0] == expected_day


def test_add_time_features_with_invalid_datetime():
    """Тест обработки невалидных дат."""
    df = pd.DataFrame({
        'pickup_datetime': ['invalid_datetime', '2023-01-01 10:00:00'],
        'fare_amount': [10.0, 20.0]
    })
    
    with pytest.raises(ValueError):
        add_time_features(df)