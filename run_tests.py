"""Скрипт для запуска тестов с различными опциями."""

import subprocess
import sys


def run_tests():
    """Запускает тесты с различными опциями."""
    print("Запуск тестов проекта")
    
    # 1. Запуск всех тестов
    print("\n 1. Запуск всех тестов:")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
    
    if result.returncode != 0:
        print("\n Некоторые тесты не прошли. Исправьте ошибки и попробуйте снова.")
        return result.returncode
    
    # 2. Запуск с покрытием кода
    print("\n 2. Запуск тестов с покрытием кода:")
    subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--tb=short"
    ])
    
    # 3. Запуск конкретного файла тестов
    print("\n 3. Запуск тестов предобработки:")
    subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_preprocessing.py",
        "-v",
        "--tb=short"
    ])
    
    # 4. Запуск с параметризованными тестами
    print("\n 4. Запуск параметризованных тестов:")
    subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_preprocessing.py::test_edge_case_filtering",
        "-v",
        "--tb=short"
    ])
    
    # 5. Запуск тестов моделей
    print("\n 5. Запуск тестов моделей:")
    subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/test_models.py",
        "-v",
        "--tb=short"
    ])
    
    print("\n Все тесты завершены")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())