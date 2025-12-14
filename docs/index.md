## **Предсказание стоимости поездок в такси**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/)

Проект машинного обучения для предсказания стоимости поездок в сервисе такси на основе исторических данных. 
Реализует полный ML пайплайн от загрузки данных до выполнения предсказаний.

## Основные возможности

✅ **Полный ML пайплайн:**
- Загрузка и предобработка данных
- Извлечение признаков
- Обучение модели Gradient Boosting
- Оценка качества модели
- Выполнение предсказаний

✅ **Единообразный код:**
- Google Style docstrings
- Статическая типизация
- Полное покрытие тестами
- Соответствие PEP8

✅ **Воспроизводимость:**
- Фиксированный random_state
- Версионирование зависимостей
- Сохранение конфигурации
- Детальное логгирование

✅ **Автоматически генерируемая документация API:**
- Подробные примеры использования
- Описание архитектуры

## Быстрый старт

# Клонирование репозитория
```bash
git clone https://github.com/BobinRobin/lesson-08-code-review.git
cd lesson-08-code-review
```

# Установка зависимостей
```bash
pip install -r requirements.txt
```

# Обучение модели
```bash
python scripts/train.py
```

# Выполнение предсказаний
```bash
python scripts/predict.py
```