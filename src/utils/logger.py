"""Модуль для настройки логгирования в проекте."""

import logging
from typing import Dict, Any


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """Настраивает логгер для проекта на основе конфигурации.

    Args:
        config (Dict[str, Any]): Конфигурация проекта, содержащая
            раздел 'logging' с параметрами:
            - level: Уровень логгирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            - format: Формат сообщений лога

    Returns:
        logging.Logger: Настроенный логгер с указанными параметрами.

    Examples:
        >>> config = {
        ...     'logging': {
        ...         'level': 'INFO',
        ...         'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ...     }
        ... }
        >>> logger = setup_logger(config)
        >>> logger.info("Логгирование настроено")
    """
    log_config = config.get("logging", {})
    
    # Уровень логгирования
    level_str = log_config.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # Формат сообщений
    fmt = log_config.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Настройка базовой конфигурации
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Получение логгера для текущего модуля
    logger = logging.getLogger(__name__)
    logger.info(f"Логгирование настроено (уровень: {level_str})")
    
    return logger