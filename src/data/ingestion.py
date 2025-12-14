"""Модуль для загрузки данных из CSV файлов."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Загружает данные из CSV-файла в pandas DataFrame.

    Args:
        data_path (str): Путь к файлу с данными. Должен быть валидным путем
            к CSV файлу.

    Returns:
        pd.DataFrame: DataFrame с загруженными данными.

    Raises:
        FileNotFoundError: Если файл по указанному пути не существует.
        pd.errors.EmptyDataError: Если файл пуст.
        pd.errors.ParserError: Если файл содержит некорректный CSV.

    Examples:
        >>> df = load_data('data/raw/uber.csv')
        >>> print(f"Загружено {len(df)} строк")
        >>> print(f"Колонки: {df.columns.tolist()}")
    """
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Данные загружены из {data_path}")
        logger.info(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} колонок")
        return df
    except FileNotFoundError:
        logger.error(f"Файл {data_path} не найден")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Файл {data_path} пуст")
        raise
    except Exception as err:
        logger.error(f"Ошибка при загрузке данных из {data_path}: {err}")
        raise