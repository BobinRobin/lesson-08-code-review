"""Комплексные тесты для модуля моделей."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import pickle
import joblib

from src.models.training import TaxiFareModel, train_model
from src.models.evaluation import evaluate_model, save_metrics


def test_taxi_fare_model_initialization(taxi_fare_model):
    """Тест инициализации модели такси."""
    assert taxi_fare_model.random_state == 42
    assert taxi_fare_model.model is not None
    assert taxi_fare_model.feature_names is None


def test_model_fit_and_predict(taxi_fare_model, sample_training_data):
    """Тест обучения и предсказания модели."""
    X, y = sample_training_data
    
    # Обучение
    taxi_fare_model.fit(X, y)
    
    # Проверяем, что модель обучилась
    assert taxi_fare_model.feature_names is None  # X не имеет атрибута columns
    
    # Предсказание
    predictions = taxi_fare_model.predict(X[:5])
    
    # Проверяем форму предсказаний
    assert predictions.shape == (5,)
    assert all(np.isfinite(predictions))


def test_model_score(taxi_fare_model, sample_training_data):
    """Тест оценки модели."""
    X, y = sample_training_data
    
    # Разделяем данные
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Обучаем на train
    taxi_fare_model.fit(X_train, y_train)
    
    # Оцениваем на test
    score = taxi_fare_model.score(X_test, y_test)
    
    # Проверяем, что score - число
    assert isinstance(score, float)
    assert -1 <= score <= 1  # R² score range


def test_model_fit_calls_sklearn(sample_training_data):
    """Тест проверяет, что fit вызывает sklearn."""
    X, y = sample_training_data
    
    # Создаем модель
    model = TaxiFareModel(random_state=42)
    
    # Мокаем метод fit у внутренней модели
    with patch.object(model.model, 'fit') as mock_fit:
        model.fit(X, y)
        mock_fit.assert_called_once_with(X, y)


def test_train_model_function(sample_training_data):
    """Тест функции train_model."""
    X, y = sample_training_data
    
    model, X_test, y_test = train_model(
        X, y, 
        random_state=42,
        test_size=0.2
    )
    
    # Проверяем возвращаемые значения
    assert isinstance(model, TaxiFareModel)
    assert len(X_test) == 20  # 100 * 0.2 = 20
    assert len(y_test) == 20
    
    # Проверяем, что модель обучилась
    assert hasattr(model, 'model')
    assert hasattr(model.model, 'predict')


@patch('src.models.training.GridSearchCV')
def test_train_model_with_grid_search(mock_grid_search, sample_training_data):
    """Тест train_model с GridSearchCV."""
    X, y = sample_training_data
    
    # Настраиваем mock
    mock_best_estimator = MagicMock()
    mock_best_estimator.predict.return_value = np.array([1, 2, 3])
    mock_grid_search.return_value.best_estimator_ = mock_best_estimator
    mock_grid_search.return_value.best_params_ = {'n_estimators': 100}
    
    with patch('src.models.training.GradientBoostingRegressor'):
        model, _, _ = train_model(
            X, y,
            use_grid_search=True
        )
    
    assert mock_grid_search.called


def test_evaluate_model_metrics(sample_training_data, taxi_fare_model):
    """Тест вычисления метрик модели."""
    X, y = sample_training_data
    
    # Обучаем модель
    taxi_fare_model.fit(X, y)
    
    # Создаем тестовые данные
    X_test = X[:20]
    y_test = y[:20]
    
    # Вычисляем метрики
    metrics = evaluate_model(taxi_fare_model, X_test, y_test)
    
    # Проверяем наличие всех метрик
    expected_metrics = ['r2_score', 'mse', 'rmse', 'mae', 'mape']
    for metric in expected_metrics:
        assert metric in metrics
    
    # Проверяем типы значений
    for value in metrics.values():
        assert isinstance(value, float)
    
    # Проверяем разумные диапазоны
    assert -1 <= metrics['r2_score'] <= 1
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0


@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_save_metrics(mock_json_dump, mock_file):
    """Тест сохранения метрик в файл."""
    metrics = {'r2_score': 0.85, 'mse': 10.5}
    
    save_metrics(metrics, 'test_metrics.json')
    
    # Проверяем, что файл был открыт для записи
    mock_file.assert_called_once_with('test_metrics.json', 'w')
    
    # Проверяем, что json.dump был вызван
    mock_json_dump.assert_called_once()


def test_mean_absolute_percentage_error_edge_cases():
    """Тест MAPE для граничных случаев."""
    from src.models.evaluation import _mean_absolute_percentage_error
    
    # Тест с нулевыми значениями
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 2, 3])
    mape = _mean_absolute_percentage_error(y_true, y_pred)
    assert mape == 0.0
    
    # Тест с корректными значениями
    y_true = np.array([10, 20, 30])
    y_pred = np.array([11, 19, 32])
    mape = _mean_absolute_percentage_error(y_true, y_pred)
    assert isinstance(mape, float)
    assert mape >= 0


@patch('joblib.dump')
def test_model_saving_with_joblib(mock_joblib_dump, taxi_fare_model, sample_training_data):
    """Тест сохранения модели с использованием mock."""
    X, y = sample_training_data
    taxi_fare_model.fit(X, y)
    
    # Сохраняем модель (мок)
    joblib.dump(taxi_fare_model.model, 'test_model.pkl')
    
    # Проверяем, что dump был вызван
    mock_joblib_dump.assert_called_once_with(
        taxi_fare_model.model, 'test_model.pkl'
    )