import pandas as pd
import os
from sklearn.model_selection import train_test_split
import data_engineering

if os.path.exists('data'):
    # Чтение данных из файла .csv
    df = pd.read_csv('data/data.csv')

    # Проведение обработки данных, выделение важных признаков
    df.dropna(subset=['Name'], inplace=True)

    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['User_Score'] = df['User_Score'].astype(float)

    df = data_engineering.filling_nan_KNN_method(df)
    df = data_engineering.fill_columns_median_value(df)

    # Проверка на пустые значения в записях
    df_nan_columns = df.columns[df.isnull().sum() != 0]
    if len(df_nan_columns) != 0:
        print('В датасете присутсвуют NaN')

    # Разделение данных на тренировочный и тестовый датасеты
    X = df.drop('User_Score', axis=1)  # Признаки
    y = df['User_Score']  # Целевая переменная

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Сохранение датасетов для тренировки и тестирования модели
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    print('Данные успешно обработаны')
else:
    print('Ошибка обработки данных')

