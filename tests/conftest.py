"""Конфигурация тестов и общие фикстуры."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os


@pytest.fixture
def sample_training_data():
    """Фикстура создаёт предсказуемые данные для обучения."""
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.squeeze() + np.random.normal(0, 1, 100)
    return X, y


@pytest.fixture
def raw_uber_data():
    """Фикстура, создающая сырые данные для тестов предобработки."""
    data = {
        'fare_amount': [10.0, -5.0, 20.0, 15.0, 30.0, 25.0, 40.0],
        'passenger_count': [1, 2, 0, 7, 3, 4, 1],
        'pickup_longitude': [-73.9, -73.9, -73.9, -73.9, -73.9, -73.9, -73.9],
        'pickup_latitude': [40.7, 40.7, 40.7, 40.7, 40.7, 40.7, 40.7],
        'dropoff_longitude': [-74.0, -74.0, -74.0, -74.0, -74.0, -74.0, -74.0],
        'dropoff_latitude': [40.8, 40.8, 40.8, 40.8, 40.8, 40.8, 40.8],
        'pickup_datetime': [
            '2023-01-01 10:00:00', '2023-01-01 11:00:00',
            '2023-01-01 12:00:00', '2023-01-01 13:00:00',
            '2023-01-01 14:00:00', '2023-01-01 15:00:00',
            '2023-01-01 16:00:00'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def taxi_fare_model():
    """Фикстура возвращает инициализированную модель такси."""
    from src.models.training import TaxiFareModel
    return TaxiFareModel(random_state=42)


@pytest.fixture
def temp_model_file():
    """Фикстура создает временный файл для сохранения модели."""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Очистка после теста
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_config():
    """Фикстура возвращает тестовую конфигурацию."""
    return {
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