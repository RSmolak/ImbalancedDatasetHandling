##########################################################
# TODO:
# - 
# - Add time measures
# - Refactor code?
# - Add imbalanced dataset handling (4/6)
##########################################################e
from matplotlib import pyplot as plt
import tqdm
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim

import model 
from read_data import load_data, prepare_dataloaders, WeightedDataset
from imbalanced_handling import handle_imbalanced
from report import plot_experiment_losses



models = [
    #["MLP_10_10_10",[10, 10, 10]],
    ["MLP_20_20_20",[20, 20, 20]],  
]

datasets = [
    #'ecoli1',
    'glass4',
    #'vowel0',
    #'yeast3',
    #'yeast5'
]

imbalance_handling_methods = [
    #"none",
    #"SMOTE",
    #"random_undersampling",
    #"batch_balancing",
    #"KDE-based_oversampling",
    "KDE-based_loss_weighting",
    #"KDE-based_batch_balancing"
]

results = {}

epochs = 200
batch_size = 32
learning_rate = 0.0002

for id_architecture, architecture in enumerate(models): 
    for id_dataset, dataset in enumerate(datasets):

        # Loading dataset
        print(f"Dataset: {dataset}")    
        X, y = load_data(f'DATASETS/{dataset}/{dataset}.dat')
        
        for id_imbalance, imbalance_method in enumerate(imbalance_handling_methods):
            print(f"Imbalance handling method: {imbalance_method}")

            train_losses = []     
            val_losses = []

            # Cross-validation
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            all_ones = []
            all_zeros = []

            for fold_id, (train_index, test_index) in enumerate(kf.split(X,y)):

                # Unique key for each experiment configuration
                key = (architecture[0], dataset, imbalance_method, fold_id)
                if key not in results:
                    results[key] = {'Train': [], 'Valid': []}

                # Splitting data into train and test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Defining device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Preparing model
                current_model = model.prepare_model(architecture, X_train, y_train, dropout=0.0)
                current_model = current_model.to(device)

                # Creating dataset objects
                train_dataset = WeightedDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long(), weights=np.ones(len(y_train)).tolist())
                valid_dataset = WeightedDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long(), weights=np.ones(len(y_test)).tolist())

                # Preparing dataloaders
                train_dataloader, valid_dataloader = prepare_dataloaders(train_dataset, valid_dataset, batch_size)

                # Handling imbalanced dataset
                train_dataset, train_dataloader = handle_imbalanced(train_dataset,
                                                                    imbalance_method, 
                                                                    train_dataloader)

                # Defining loss function and optimizer
                optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
                train_criterion = nn.CrossEntropyLoss(reduction='none')
                val_criterion = nn.CrossEntropyLoss()

                # Counting number of ones and zeros in train labels per fold
                count_ones = 0
                count_zeros = 0

                # Training
                pbar = tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch")
                for epoch in pbar:
                    current_model.train()
                    train_loss = 0
                    correct = 0
                    total = 0

                    train_true_labels = []
                    train_predicted_labels = []
                    val_true_labels = []
                    val_predicted_labels = []

                    for batch_X, batch_y, weights in train_dataloader:
                        batch_X, batch_y, weights = batch_X.to(device), batch_y.to(device), weights.to(device)

                        # Counting number of ones and zeros in batch
                        ones = (batch_y == 1).sum().item()
                        zeros = (batch_y == 0).sum().item()
                        count_ones += ones
                        count_zeros += zeros


                        # Forward pass
                        optimizer.zero_grad()
                        y_pred = current_model(batch_X)
                        loss = train_criterion(y_pred, batch_y)

                        # Calculating weighted loss
                        if weights is not None:
                            loss = torch.mean(loss * weights)
                        else:
                            loss = torch.mean(loss)

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        _, predicted = torch.max(y_pred.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                        train_true_labels.extend(batch_y.cpu().numpy())
                        train_predicted_labels.extend(predicted.cpu().numpy())

                    # Saving train results
                    results[key]['Train'].append({
                        'Loss': train_loss / len(train_dataloader),
                        'TrueLabels': train_true_labels,
                        'PredictedLabels': train_predicted_labels
                    })

                    # Validation
                    val_loss = 0
                    correct = 0
                    total = 0
                    current_model.eval()

                    with torch.no_grad():
                        for batch_X, batch_y, _ in valid_dataloader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            y_pred = current_model(batch_X)
                            loss = val_criterion(y_pred, batch_y)
                            val_loss += loss.item()
                            _, predicted = torch.max(y_pred.data, 1)
                            total += batch_y.size(0)
                            correct += (predicted == batch_y).sum().item()

                            val_true_labels.extend(batch_y.cpu().numpy())
                            val_predicted_labels.extend(predicted.cpu().numpy())

                    results[key]['Valid'].append({
                    'Loss': val_loss / len(valid_dataloader),
                    'TrueLabels': val_true_labels,
                    'PredictedLabels': val_predicted_labels
                    })

                    # Update progress bar or print statement
                    pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(valid_dataloader):.4f}")

                all_ones.append(count_ones)
                all_zeros.append(count_zeros)

            plot_experiment_losses(results, architecture[0], dataset, imbalance_method)
            
            print("All ones:", all_ones)
            print("All zeros:", all_zeros)

        plt.show()


df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)