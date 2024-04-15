import pandas as pd

df = pd.read_csv("dataset/titanic_train.csv")

df_encoded = pd.get_dummies(df, columns=['Sex'])


df_encoded['Sex_female'] = df_encoded['Sex_female'].astype(int)
df_encoded['Sex_male'] = df_encoded['Sex_male'].astype(int)

df_encoded.to_csv("dataset/titanic_train.csv", index=False)
