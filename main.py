##########################################################
# TODO:
# 1. Add time measures
# 2. Refactor code
# 3. Add imbalanced dataset handling
#
##########################################################
from matplotlib import pyplot as plt
import tqdm
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import model 
from read_data import load_data
from imbalanced_handling import handle_imbalanced
from report import plot_experiment_losses


models = [
    #["MLP_10_10_10",[10, 10, 10]],
    ["MLP_20_20_20",[20, 20, 20]],  
]

datasets = [
    'ecoli1',
    #'glass4',
    #'vowel0',
    #'yeast3',
    #'yeast5'
]

imbalance_handling_methods = [
    "none",
    #"SMOTE",
    #"random_undersampling",
    #"batch_balancing"
    #"KDE-based_oversampling",
    #"KDE-based_loss_weighting",
    #"KDE-based_batch_balancing"
]

results = {}


epochs = 200
batch_size = 32
learning_rate = 0.001


for id_architecture, architecture in enumerate(models): 
    for id_dataset, dataset in enumerate(datasets):

        # Loading dataset
        print(f"Dataset: {dataset}")
        X, y = load_data(f'DATASETS/{dataset}/{dataset}.dat')
        
        for id_imbalance, imbalance_method in enumerate(imbalance_handling_methods):

            train_losses = []     
            val_losses = []

            # Cross-validation
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold_id, (train_index, test_index) in enumerate(kf.split(X,y)):

                # Unique key for each experiment configuration
                key = (architecture[0], dataset, imbalance_method, fold_id)
                if key not in results:
                    results[key] = {'Train': [], 'Valid': []}

                # Splitting data into train and test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Preparing model
                current_model = model.prepare_model(architecture, X_train, y_train, dropout=0.0)
                handle_imbalanced(current_model, X_train, y_train, imbalance_method)

                # Defining device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                current_model = current_model.to(device)

                # Preparing dataloaders
                X_tensor_train = torch.from_numpy(X_train).float().to(device)
                y_tensor_train = torch.from_numpy(y_train).long().to(device)
                X_tensor_valid = torch.from_numpy(X_test).float().to(device)
                y_tensor_valid = torch.from_numpy(y_test).long().to(device)
                train_dataloader = data.DataLoader(data.TensorDataset(X_tensor_train, y_tensor_train), batch_size=batch_size, shuffle=True)
                valid_dataloader = data.DataLoader(data.TensorDataset(X_tensor_valid, y_tensor_valid), batch_size=batch_size, shuffle=True)

                # Defining loss function and optimizer
                optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()

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

                    for batch_X, batch_y in train_dataloader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                        # Forward pass
                        optimizer.zero_grad()
                        y_pred = current_model(batch_X)
                        loss = criterion(y_pred, batch_y)
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
                        for batch_X, batch_y in valid_dataloader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            y_pred = current_model(batch_X)
                            loss = criterion(y_pred, batch_y)
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

            plot_experiment_losses(results, architecture[0], dataset, imbalance_method)
            plt.show()


df = pd.DataFrame(results)
df.to_csv("experiment_results.csv", index=False)