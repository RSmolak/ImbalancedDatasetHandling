from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

from kde import get_kde_weights

def handle_imbalanced(model, X, y, imbalance_method, train_dataloader):
    if imbalance_method == "none":
        return X, y, train_dataloader

    elif imbalance_method == "SMOTE":
        X, y = perform_SMOTE(X, y)
        print("Dataset after SMOTE:", X.shape, y.shape)
        return X, y, train_dataloader
    
    elif imbalance_method == "random_undersampling":
        X, y = perform_random_undersampling(X, y)
        print("Dataset after random undersampling:", X.shape, y.shape)
        return X, y, train_dataloader
    
    elif imbalance_method == "batch_balancing":
        dataloader = perform_batch_balancing(X, y, train_dataloader)
        return X, y, dataloader
    
    elif imbalance_method == "KDE-based_oversampling":
        pass
    elif imbalance_method == "KDE-based_loss_weighting":
        pass
    elif imbalance_method == "KDE-based_batch_balancing":
        dataloader = perform_KDE_based_batch_balancing(X, y, train_dataloader)
        return X, y, dataloader

def perform_SMOTE(X, y):
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    return X, y

def perform_random_undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(X, y)
    return X, y

def perform_batch_balancing(X, y, train_dataloader):
    # Calculate weights
    y = torch.tensor(y)
    class_counts = torch.tensor([(y == class_id).sum() for class_id in torch.unique(y, sorted=True)])
    class_weights = 1. / class_counts.float()
    weights = class_weights[y]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(weights))

    new_dataloader = DataLoader(train_dataloader.dataset, batch_size=train_dataloader.batch_size, sampler=sampler)
    return new_dataloader

def perform_KDE_based_oversampling(X, y):
    pass

def perform_KDE_based_loss_weighting(X, y):
    weights = get_kde_weights(X, transform='normalize-expand')
    return weights

def perform_KDE_based_batch_balancing(X, y, train_dataloader):
    # Calculate weights
    weights = get_kde_weights(X, transform='normalize-expand')

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(weights))

    new_dataloader = DataLoader(train_dataloader.dataset, batch_size=train_dataloader.batch_size, sampler=sampler)
    return new_dataloader