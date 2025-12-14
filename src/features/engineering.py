"""Модуль для создания признаков из временных данных."""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает временные признаки из столбца с датой и временем.

    Создает признаки:
    - hour: Час поездки (0-23)
    - day_of_week: День недели (0-6, где 0 - понедельник)

    Args:
        df (pd.DataFrame): DataFrame со столбцом 'pickup_datetime'.
            Значения должны быть в формате, распознаваемом pandas.to_datetime.

    Returns:
        pd.DataFrame: DataFrame с добавленными временными признаками
            и удаленным исходным столбцом 'pickup_datetime'.

    Raises:
        KeyError: Если столбец 'pickup_datetime' отсутствует в DataFrame.
        ValueError: Если значения в 'pickup_datetime' не могут быть преобразованы
            в формат datetime.

    Examples:
        >>> df = pd.DataFrame({
        ...     'pickup_datetime': ['2023-01-01 10:30:00', '2023-01-02 15:45:00'],
        ...     'fare_amount': [10.0, 20.0]
        ... })
        >>> df_with_time = add_time_features(df)
        >>> print(df_with_time[['hour', 'day_of_week']])
    """
    if 'pickup_datetime' not in df.columns:
        error_msg = "Столбец 'pickup_datetime' не найден в данных"
        logger.error(error_msg)
        raise KeyError(error_msg)
    
    df_processed = df.copy()
    
    try:
        # Преобразование в datetime
        df_processed['pickup_datetime'] = pd.to_datetime(df_processed['pickup_datetime'])
        
        # Извлечение признаков
        df_processed['hour'] = df_processed['pickup_datetime'].dt.hour
        df_processed['day_of_week'] = df_processed['pickup_datetime'].dt.dayofweek
        
        logger.info("Временные признаки добавлены: hour, day_of_week")
        
        # Удаление исходного столбца
        df_processed = df_processed.drop('pickup_datetime', axis=1)
        
        return df_processed
    except (ValueError, TypeError) as e:
        error_msg = f"Ошибка при обработке даты и времени: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)