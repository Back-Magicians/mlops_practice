import os
import pandas as pd

preprocessed_dir = 'preprocessed'
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Загружаем данные из файлов тренировочных и тестовых
data_df_train = pd.read_csv('train/train1.csv')
data_df_test = pd.read_csv('test/test1.csv')

# Предобработка данных тренировочного набора
data_df_train.dropna(inplace=True)
data_df_train = data_df_train[['AverageTemperature', 'AverageTemperatureUncertainty']]
data_df_train.to_csv(os.path.join(preprocessed_dir, 'preprocessed_train.csv'), index=False)

# Предобработка данных тестового набора
data_df_test.dropna(inplace=True)
data_df_test = data_df_test[['AverageTemperature', 'AverageTemperatureUncertainty']]
data_df_test.to_csv(os.path.join(preprocessed_dir, 'preprocessed_test.csv'), index=False)
