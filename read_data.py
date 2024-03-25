import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch.utils.data as data
import torch

def load_data(file_name):
    df = pd.read_csv(file_name, comment='@', header=None)

    features = df.iloc[:, :-1].values
    # Remove leading and trailing spaces from labels
    labels = df.iloc[:, -1].str.strip().values

    print(np.unique(labels))
    le = LabelEncoder()
    numerical_labels = le.fit_transform(labels)

    return features, numerical_labels

def prepare_dataloaders(X_train, y_train, X_test, y_test, batch_size, device):
    # Preparing dataloaders
    X_tensor_train = torch.from_numpy(X_train).float().to(device)
    y_tensor_train = torch.from_numpy(y_train).long().to(device)
    X_tensor_valid = torch.from_numpy(X_test).float().to(device)
    y_tensor_valid = torch.from_numpy(y_test).long().to(device)

    data.TensorDataset(X_tensor_train, y_tensor_train)

    train_dataloader = data.DataLoader(data.TensorDataset(X_tensor_train, y_tensor_train), batch_size=batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(data.TensorDataset(X_tensor_valid, y_tensor_valid), batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_dataloader