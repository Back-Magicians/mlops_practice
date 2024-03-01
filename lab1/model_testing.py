import pandas as pd
from sklearn.metrics import accuracy_score

# Загрузка данных из внешнего файла в DataFrame
data_df = pd.read_csv('путь_к_файлу.csv')

true_labels = data_df['метки']
predicted_labels = pd.read_csv('путь_к_файлу_с_предсказанными_метками.csv')['предсказанные_метки']

# Рассчитываем accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)