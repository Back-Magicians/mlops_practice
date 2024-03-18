import pandas as pd
from sklearn.impute import KNNImputer
import json

def filling_nan_KNN_method(data):
    ds = data.copy()
    ds = ds.select_dtypes(["float64", "int64"])

    ds_nan_columns = ds.columns[ds.isnull().sum() != 0]

    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    imputer.fit(ds)

    ds = pd.DataFrame(imputer.transform(ds), index=ds.index, columns=ds.columns)
    ds.head()

    for column in ds_nan_columns:
        data[column] = data[column].fillna(ds[column])

    return data


def fill_columns_median_value(data):
    data_nan_columns = data.columns[data.isnull().sum() != 0]

    for column in data_nan_columns:
        most_frequent_value = data[column].mode()[0]
        data.fillna({column: most_frequent_value}, inplace=True)

    return data

def ordinal_coding(data):

    with open('encoding_dict.json', 'r') as json_file:
        encoding_dict = json.load(json_file)

    # Применение словаря кодировок к новым данным
    for column, encoding_mapping in encoding_dict.items():
        data[column] = data[column].map(encoding_mapping)

    return data