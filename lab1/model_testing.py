import os

import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

preprocessed_dir = 'preprocessed'
model_file = 'linear_regression_model.pkl'

data_df_test = pd.read_csv(os.path.join(preprocessed_dir, 'preprocessed_test.csv'))

model = joblib.load(model_file)

X_test = data_df_test[['AverageTemperatureUncertainty']]
y_test = data_df_test['AverageTemperature']

predictions = model.predict(X_test)

# Рассчитываем среднеквадратичную ошибку (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)