import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def read_dataset_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start_index = 0
    for i, line in enumerate(lines):
        if not line.startswith('@'):
            data_start_index = i
            break

    data = pd.read_csv(file_path, skiprows=data_start_index, header=None)

    data.head()
    return data

def get_data_and_labels(df):
    y = df[df.columns[-1]].values
    X = df.drop(df.columns[-1], axis=1).values
    return X, y


def one_hot_encode_dataframe(df):
    """
    Perform one hot encoding on all categorical columns in a dataframe, excluding the last column.

    :param df: DataFrame to encode.
    :return: Encoded DataFrame.
    """
    label_column = df.columns[-1]
    categorical_columns = df.select_dtypes(include=['object']).drop(label_column, axis=1).columns

    encoder = OneHotEncoder(sparse_output=False)
    df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
    df_encoded.index = df.index

    df_encoded = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    return pd.concat([df_encoded.drop(label_column, axis=1), df[label_column].to_frame()], axis=1)



