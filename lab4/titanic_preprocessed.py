import pandas as pd

df = pd.read_csv("titanic_train.csv")

mean_age = df['Age'].mean()

df['Age'] = df['Age'].fillna(mean_age)

df.to_csv("dataset/titanic_train_v2.csv", index=False)
