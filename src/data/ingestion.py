import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Загружает данные из CSV-файла.

    Аргументы:
        data_path (str): Путь к файлу с данными.

    Возвращает:
        pd.DataFrame: DataFrame с данными.

    Исключения:
        FileNotFoundError: Если файл не найден.
    """
    try:
        df = pd.read_csv(data_path)
        logger.info("Данные загружены из %s", data_path)
        return df
    except FileNotFoundError:
        logger.error("Файл %s не найден", data_path)
        raise
    except Exception as err:
        logger.error("Ошибка при загрузке данных: %s", err)
        raise
