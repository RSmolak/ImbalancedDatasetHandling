import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

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

def generate_synthetic_dataset(dataset_name):
    _, n_samples, n_features, imbalance_ratio = dataset_name.split('_')
    n_samples, n_features, imbalance_ratio = int(n_samples), int(n_features), float(imbalance_ratio)
    print(n_samples, n_features, imbalance_ratio)

    imbalance_ratio = [imbalance_ratio, 1 - imbalance_ratio]
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2,
                               weights=imbalance_ratio, n_informative=n_features, n_redundant=0, random_state=42)
    
    return X, y


class WeightedDataset(data.Dataset):
    def __init__(self, data, labels, weights):
        self.data = data  # Your data
        self.labels = labels  # Your labels
        self.weights = weights  # Corresponding weights for each data sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label = self.labels[idx]
        weight = self.weights[idx]
        return data_sample, label, weight


def prepare_dataloaders(train_dataset, valid_dataset, batch_size):
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size)
    return train_dataloader, valid_dataloader