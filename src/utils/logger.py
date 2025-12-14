"""Модуль для настройки логгирования."""

import logging
from typing import Dict, Any


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """Настраивает логгер для проекта.

    Аргументы:
        config (Dict[str, Any]): Конфигурация проекта.

    Возвращает:
        logging.Logger: Настроенный логгер.
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger(__name__)
