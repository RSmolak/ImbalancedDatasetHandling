from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

from read_data import WeightedDataset
from kde import get_kde_weights, generate_kde_samples

def handle_imbalanced(train_dataset, imbalance_method, train_dataloader):
    if imbalance_method == "none":
        return train_dataset, train_dataloader

    elif imbalance_method == "SMOTE":
        X, y = train_dataset.data, train_dataset.labels
        X, y = perform_SMOTE(X, y)

        train_dataset = WeightedDataset(torch.from_numpy(X), torch.from_numpy(y), weights=np.ones(len(y)).tolist())
        print("Dataset after SMOTE:", train_dataset.data.shape, train_dataset.labels.shape)
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
        return train_dataset, train_dataloader
    
    elif imbalance_method == "random_undersampling":
        X, y = train_dataset.data, train_dataset.labels
        X, y = perform_random_undersampling(X, y)

        train_dataset = WeightedDataset(torch.from_numpy(X), torch.from_numpy(y), weights=np.ones(len(y)).tolist())
        print("Dataset after random undersampling:", X.shape, y.shape)
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
        return train_dataset, train_dataloader
    
    elif imbalance_method == "batch_balancing":
        dataloader = perform_batch_balancing(train_dataset, train_dataloader)
        return train_dataset, dataloader
    
    elif imbalance_method == "KDE-based_oversampling":
        X, y = train_dataset.data, train_dataset.labels
        X, y = perform_KDE_based_oversampling(X, y)
        
        train_dataset = WeightedDataset(X, y, weights=np.ones(len(y)).tolist())
        print("Dataset after KDE-based oversampling:", X.shape, y.shape)
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
        return train_dataset, train_dataloader

    elif imbalance_method == "KDE-based_loss_weighting":
        X, y = train_dataset.data, train_dataset.labels
    
        weights = perform_KDE_based_loss_weighting(train_dataset.data, train_dataset.labels)
        train_dataset = WeightedDataset(X, y, weights=weights)
        print("Dataset after KDE-based loss weighting:", train_dataset.data.shape, train_dataset.labels.shape, train_dataset.weights[0:5])
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
        return train_dataset, train_dataloader
    
    elif imbalance_method == "KDE-based_batch_balancing":
        dataloader = perform_KDE_based_batch_balancing(train_dataset, train_dataloader)
        return train_dataset, dataloader

def perform_SMOTE(X, y):
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    return X, y

def perform_random_undersampling(X, y):
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(X, y)
    return X, y

def perform_batch_balancing(train_dataset, train_dataloader):
    # Obliczanie wag dla klas
    y = torch.tensor(train_dataset.labels)
    class_counts = torch.tensor([(y == class_id).sum() for class_id in torch.unique(y, sorted=True)])
    class_weights = 1. / class_counts.float()
    weights = class_weights[y]

    # Tworzenie samplera na obliczonych wagach
    sampler = WeightedRandomSampler(weights, len(weights))
    new_dataloader = DataLoader(train_dataloader.dataset, batch_size=train_dataloader.batch_size, sampler=sampler)
    
    return new_dataloader

def perform_KDE_based_oversampling(X, y):
    # Identify the number of samples for each class
    unique, counts = np.unique(y, return_counts=True)
    class_sample_counts = dict(zip(unique, counts))
    
    # Find the majority and minority classes
    majority_class = max(class_sample_counts, key=class_sample_counts.get)
    minority_class = min(class_sample_counts, key=class_sample_counts.get)
    
    # Calculate the number of samples to generate
    num_samples_to_generate = class_sample_counts[majority_class] - class_sample_counts[minority_class]

    if num_samples_to_generate > 0:
        # Generate new samples using KDE
        new_samples = generate_kde_samples(X[y == minority_class], num_samples_to_generate)

        # Labels for new samples
        new_labels = np.array([minority_class] * num_samples_to_generate)

        # Combine the original data with the new samples
        new_X = np.vstack([X, new_samples])
        new_y = np.concatenate([y, new_labels])

        # Convert to Torch tensors ensuring data type consistency
        new_X = torch.from_numpy(new_X).float()  # Convert to float32
        new_y = torch.from_numpy(new_y).long()   # Convert labels to long for compatibility with CrossEntropyLoss, etc.
        
        return new_X, new_y
    else:
        # No need to generate new samples if the minority class has equal or more samples than the majority
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).long()
        return X, y



def perform_KDE_based_loss_weighting(X, y):
    weights = get_kde_weights(X, transform='normalize-expand')
    return weights
def perform_KDE_based_batch_balancing(train_dataset, train_dataloader):
    weights = get_kde_weights(train_dataset.data, transform='normalize-expand')
    sampler = WeightedRandomSampler(weights, len(weights))
    new_dataloader = DataLoader(train_dataloader.dataset, batch_size=train_dataloader.batch_size, sampler=sampler)
    return new_dataloader


