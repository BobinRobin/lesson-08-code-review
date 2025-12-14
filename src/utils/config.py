"""Модуль для загрузки конфигурации проекта из JSON файла."""

import json
from typing import Dict


def load_config(path: str = "config.json") -> Dict:
    """Загружает конфигурацию проекта из JSON файла.

    Args:
        path (str, optional): Путь к файлу конфигурации.
            По умолчанию "config.json".

    Returns:
        Dict: Словарь с конфигурацией проекта.

    Raises:
        FileNotFoundError: Если файл конфигурации не найден.
        json.JSONDecodeError: Если файл содержит некорректный JSON.
        PermissionError: Если нет прав на чтение файла.

    Examples:
        >>> config = load_config()
        >>> print(f"Путь к данным: {config['data']['raw_path']}")
        >>> print(f"Признаки: {config['features']['selected']}")
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        error_msg = f"Файл конфигурации не найден: {path}"
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Некорректный JSON в файле {path}: {e}"
        raise json.JSONDecodeError(error_msg)
    except PermissionError:
        error_msg = f"Нет прав на чтение файла: {path}"
        raise PermissionError(error_msg)