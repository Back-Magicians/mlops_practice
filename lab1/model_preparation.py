import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

preprocessed_dir = 'preprocessed'
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

data_df_train = pd.read_csv(os.path.join(preprocessed_dir, 'preprocessed_train.csv'))

X_train = data_df_train[['AverageTemperatureUncertainty']]
y_train = data_df_train['AverageTemperature']
model = LinearRegression()

model.fit(X_train, y_train)

joblib.dump(model, 'linear_regression_model.pkl')
