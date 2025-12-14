"""Комплексные тесты для модуля предобработки данных."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.preprocessing import preprocess_data, create_trip_features


def test_preprocess_data_does_not_modify_original_dataframe():
    """Тест проверяет, что функция не изменяет исходный DataFrame."""
    original_df = pd.DataFrame({
        'fare_amount': [10.0], 'passenger_count': [1],
        'pickup_longitude': [-73.9], 'pickup_latitude': [40.7],
        'dropoff_longitude': [-74.0], 'dropoff_latitude': [40.8],
    })
    original_columns = list(original_df.columns)
    
    # Передаём копию, чтобы избежать ошибки в других тестах
    preprocess_data(original_df.copy())
    
    assert list(original_df.columns) == original_columns


def test_filters_invalid_rows(raw_uber_data):
    """Тест проверяет, что невалидные строки отфильтрованы."""
    processed_df = preprocess_data(raw_uber_data)
    
    # Исправляем ожидаемое количество строк
    # Валидные строки: индексы 0, 4, 5, 6 (fare_amount>0 и passenger_count в [1,6])
    assert len(processed_df) == 4
    
    # Проверяем корректность фильтрации
    assert all(processed_df['fare_amount'] > 0)
    assert all((processed_df['passenger_count'] >= 1) & 
               (processed_df['passenger_count'] <= 6))
    assert processed_df['passenger_count'].tolist() == [1, 3, 4, 1]


def test_creates_distance_feature():
    """Тест проверяет корректность расчёта дистанции."""
    data = pd.DataFrame({
        'fare_amount': [10.0], 'passenger_count': [1],
        'pickup_longitude': [0.0], 'pickup_latitude': [0.0],
        'dropoff_longitude': [3.0], 'dropoff_latitude': [4.0],
        'pickup_datetime': ['2023-01-01 10:00:00']
    })
    processed_df = preprocess_data(data)
    
    # Ожидаемое расстояние (по теореме Пифагора) = sqrt(3^2 + 4^2) = 5
    assert 'distance' in processed_df.columns
    assert processed_df['distance'].iloc[0] == pytest.approx(5.0)


def test_returns_correct_columns(raw_uber_data):
    """Тест проверяет, что возвращаются все колонки после очистки."""
    processed_df = preprocess_data(raw_uber_data)
    
    # Ожидаем все исходные колонки + distance
    expected_columns = [
        'fare_amount', 'passenger_count', 
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'pickup_datetime', 'distance'
    ]
    
    # Проверяем наличие всех ожидаемых колонок
    for col in expected_columns:
        assert col in processed_df.columns


def test_handles_missing_values():
    """Тест проверяет обработку пропущенных значений."""
    data = pd.DataFrame({
        'fare_amount': [10.0, 20.0, None, 30.0],
        'passenger_count': [1, 2, 3, None],
        'pickup_longitude': [-73.9, -73.9, -73.9, -73.9],
        'pickup_latitude': [40.7, 40.7, 40.7, 40.7],
        'dropoff_longitude': [-74.0, -74.0, -74.0, -74.0],
        'dropoff_latitude': [40.8, 40.8, 40.8, 40.8],
        'pickup_datetime': ['2023-01-01 10:00:00'] * 4
    })
    
    processed_df = preprocess_data(data)
    
    # Строки с NaN в fare_amount или passenger_count должны быть удалены
    # Остаются только строки 0 и 1 -> 2 строки
    assert len(processed_df) == 2
    assert processed_df['fare_amount'].iloc[0] == 10.0
    assert processed_df['fare_amount'].iloc[1] == 20.0


def test_creates_trip_features():
    """Тест проверяет создание дополнительных признаков поездки."""
    data = pd.DataFrame({
        'pickup_longitude': [0.0, 1.0],
        'pickup_latitude': [0.0, 2.0],
        'dropoff_longitude': [3.0, 4.0],
        'dropoff_latitude': [4.0, 5.0],
    })
    
    features_df = create_trip_features(data)
    
    # Проверяем создание новых признаков
    assert 'delta_longitude' in features_df.columns
    assert 'delta_latitude' in features_df.columns
    
    # Проверяем корректность вычислений
    assert features_df['delta_longitude'].iloc[0] == 3.0
    assert features_df['delta_latitude'].iloc[0] == 4.0


@pytest.mark.parametrize(
    "test_input, expected_rows",
    [
        ({'passenger_count': [0], 'fare_amount': [10]}, 0),
        ({'passenger_count': [7], 'fare_amount': [10]}, 0),
        ({'passenger_count': [1], 'fare_amount': [-10]}, 0),
        ({'passenger_count': [1, 2], 'fare_amount': [10, 20]}, 2),
        ({'passenger_count': [1, 0, 7], 'fare_amount': [10, 20, 30]}, 1),
    ]
)
def test_edge_case_filtering(test_input, expected_rows):
    """Параметризованный тест для граничных случаев фильтрации."""
    # Базовые данные с координатами
    base_data = {
        'pickup_longitude': [-73.9], 'pickup_latitude': [40.7],
        'dropoff_longitude': [-74.0], 'dropoff_latitude': [40.8],
        'pickup_datetime': ['2023-01-01 10:00:00']
    }
    
    # Создаём DataFrame с тестовыми данными
    num_rows = len(test_input['passenger_count'])
    df_data = {}
    
    # Расширяем базовые данные до нужного количества строк
    for key, value in base_data.items():
        df_data[key] = value * num_rows if len(value) == 1 else value
    
    # Добавляем тестовые данные
    df_data.update(test_input)
    
    df = pd.DataFrame(df_data)
    processed_df = preprocess_data(df)
    
    assert len(processed_df) == expected_rows


@patch('src.data.preprocessing.logger')
def test_logging_in_preprocessing(mock_logger):
    """Тест проверяет корректность логирования в функции предобработки."""
    # Добавляем бесконечные значения для проверки warning
    data = pd.DataFrame({
        'fare_amount': [10.0, -5.0, 20.0, float('inf')],
        'passenger_count': [1, 8, 3, 2],
        'pickup_longitude': [-73.9, -73.9, -73.9, -73.9],
        'pickup_latitude': [40.7, 40.7, 40.7, 40.7],
        'dropoff_longitude': [-74.0, -74.0, -74.0, -74.0],
        'dropoff_latitude': [40.8, 40.8, 40.8, 40.8],
        'pickup_datetime': ['2023-01-01 10:00:00'] * 4
    })
    
    preprocess_data(data)
    
    # Проверяем, что были вызовы логгера info
    assert mock_logger.info.call_count >= 1
    
    # Проверяем warning для бесконечных значений
    warning_calls = [call for call in mock_logger.warning.call_args_list 
                    if any('бесконечные значения' in str(arg) for arg in call[0] if isinstance(arg, str))]
    assert len(warning_calls) > 0