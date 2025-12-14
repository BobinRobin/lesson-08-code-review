"""Модуль для загрузки конфигурации."""

import json
from typing import Dict


def load_config(path: str = "config.json") -> Dict:
    """Загружает конфигурацию из JSON файла.

    Аргументы:
        path (str): Путь к файлу конфигурации.

    Возвращает:
        Dict: Словарь с конфигурацией.

    Исключения:
        FileNotFoundError: Если файл не найден.
        json.JSONDecodeError: Если файл содержит некорректный JSON.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
