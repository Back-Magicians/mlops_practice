#!/bin/bash

# Запуск первого Python-скрипта
echo "Запуск первого скрипта..."
python3 model_preprocessing.py

# Запуск второго Python-скрипта
echo "Запуск второго скрипта..."
python3 model_preparation.py

# Запуск третьего Python-скрипта
echo "Запуск третьего скрипта..."
python3 model_testing.py

# Завершение скрипта
echo "Скрипты выполнены."
