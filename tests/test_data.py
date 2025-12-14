"""Расширенные тесты для модуля данных."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from src.data.ingestion import load_data


def test_load_data_success():
    """Тест успешной загрузки данных."""
    test_csv_content = """fare_amount,passenger_count,pickup_datetime
10.0,1,2023-01-01 10:00:00
20.0,2,2023-01-01 11:00:00"""
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({
            'fare_amount': [10.0, 20.0],
            'passenger_count': [1, 2],
            'pickup_datetime': ['2023-01-01 10:00:00', '2023-01-01 11:00:00']
        })
        
        df = load_data('test.csv')
        
        # Проверяем вызов read_csv с правильным путем
        mock_read_csv.assert_called_once_with('test.csv')
        
        # Проверяем результат
        assert not df.empty
        assert len(df) == 2
        assert 'fare_amount' in df.columns


@patch('src.data.ingestion.logger')
def test_load_data_file_not_found(mock_logger):
    """Тест загрузки данных при отсутствии файла."""
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent.csv')
        
        # Проверяем логирование ошибки
        assert mock_logger.error.called


@patch('src.data.ingestion.logger')
def test_load_data_general_exception(mock_logger):
    """Тест загрузки данных при общей ошибке."""
    with patch('pandas.read_csv', side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            load_data('corrupt.csv')
    
    # Проверяем, что error был вызван
    assert mock_logger.error.called
    # Проверяем, что первый аргумент содержит правильный текст
    call_args = mock_logger.error.call_args
    assert "Ошибка при загрузке данных" in call_args[0][0]


def test_load_data_returns_dataframe():
    """Тест проверяет, что функция возвращает DataFrame."""
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({'col': [1, 2, 3]})
        
        result = load_data('test.csv')
        
        assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "file_path, expected_error",
    [
        ("", FileNotFoundError),
        (None, TypeError),
        (123, AttributeError),
    ]
)
def test_load_data_invalid_paths(file_path, expected_error):
    """Тест загрузки данных с невалидными путями."""
    with patch('pandas.read_csv', side_effect=expected_error):
        with pytest.raises(expected_error):
            load_data(file_path)