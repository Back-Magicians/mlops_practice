import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Загрузить данные из файла в DataFrame
data_df = pd.read_csv('путь_к_файлу.csv')

# Предположим, что 'X' содержит признаки, а 'y' - целевую переменную
X = data_df.drop(columns=['Average Temperature'])  # Удалить столбец с целевой переменной и использовать остальные столбцы как признаки
y = data_df['Average Temperature']  # Выделить столбец с целевой переменной

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создать экземпляр модели линейной регрессии
model = LinearRegression()

# Обучение модели на обучающем наборе данных
model.fit(X_train, y_train)

# Предсказание на тестовом наборе данных
predictions = model.predict(X_test)

# Создание DataFrame с предсказаниями
predictions_df = pd.DataFrame(predictions, columns=['Average Temperature'])

# Запись предсказанных значений в CSV файл
predictions_df.to_csv('путь_к_файлу_с_предсказаниями.csv', index=False)