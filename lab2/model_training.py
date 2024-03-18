import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# Преобразование текстовых признаков в числовые
X_train = pd.get_dummies(X_train)

# Обучение Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Сохранение модели в файл pickle
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)
