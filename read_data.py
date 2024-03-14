import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_name):
    df = pd.read_csv(file_name, comment='@', header=None)

    features = df.iloc[:, :-1].values
    # Remove leading and trailing spaces from labels
    labels = df.iloc[:, -1].str.strip().values

    print(np.unique(labels))
    le = LabelEncoder()
    numerical_labels = le.fit_transform(labels)

    return features, numerical_labels
