import logging
import pandas as pd

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает временные признаки из столбца pickup_datetime.

    Аргументы:
        df (pd.DataFrame): DataFrame со столбцом pickup_datetime.

    Возвращает:
        pd.DataFrame: DataFrame с добавленными признаками hour и day_of_week.

    Исключения:
        KeyError: Если столбец pickup_datetime отсутствует.
    """
    if "pickup_datetime" not in df.columns:
        logger.error("Столбец 'pickup_datetime' не найден в данных")
        raise KeyError("Столбец 'pickup_datetime' не найден")

    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    logger.info("Временные признаки добавлены: hour, day_of_week")
    return df.drop("pickup_datetime", axis=1)
