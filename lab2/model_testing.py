import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv")

with open("model.pickle", "rb") as f:
    model = pickle.load(f)

# X_test = pd.get_dummies(X_test)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Среднеквадратическая ошибка:", mse)
print("R^2:", r2)
