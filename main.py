##########################################################
#TODO:
# 1. Add more metrics
# 2. Add confusion matrix
# 3. Add ROC curve (maybe)
# 4. Add tables and saving results
#
##########################################################
import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import model 
from read_data import load_data
from imbalanced_handling import handle_imbalanced
from report import plot_losses, plot_accuracies, plot_metric




models = [
    # model.NeuralNet(input_dim=10, hidden_layers=[10, 10, 10], output_dim=3),
    # model.NeuralNet(input_dim=10, hidden_layers=[20, 20, 20], output_dim=3),
    ["MLP_10_10_10",[10, 10, 10]],
    #["MLP_20_20_20",[20, 20, 20]],  
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

results = {
    'architecture_name': {
        'dataset_name': {
            'imbalance_method_name': {
                'fold_number': {
                    'train': {'labels': [], 'predictions': [], 'losses': []},
                    'validation': {'labels': [], 'predictions': [], 'losses': []}
                }
            }
        }
    }
}



epochs = 200
batch_size = 32
learning_rate = 0.001


for id_architecture, architecture in enumerate(models): 
    for id_dataset, dataset in enumerate(datasets):
        print(f"Dataset: {dataset}")
        X, y = load_data(f'DATASETS/{dataset}/{dataset}.dat')
        
        current_model = model.prepare_model(architecture, X, y)
        print("Using model", current_model.get_name())

        for id_imbalance, imbalance_method in enumerate(imbalance_handling_methods):

            train_losses = []
            train_accuracies = []
            train_f1_scores = []
            train_balanced_accuracies = []
            train_precisions = []
            train_recalls = []
            
            val_losses = []
            val_accuracies = []
            val_f1_scores = []
            val_balanced_accuracies = []
            val_precisions = []
            val_recalls = []

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for fold_id, (train_index, test_index) in enumerate(kf.split(X,y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                current_model = model.prepare_model(architecture, X_train, y_train)
                handle_imbalanced(current_model, X_train, y_train, imbalance_method)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(device)

                X_tensor_train = torch.from_numpy(X_train).float().to(device)
                y_tensor_train = torch.from_numpy(y_train).long().to(device)
                X_tensor_valid = torch.from_numpy(X_test).float().to(device)
                y_tensor_valid = torch.from_numpy(y_test).long().to(device)

                current_model = current_model.to(device)

                train_dataloader = data.DataLoader(data.TensorDataset(X_tensor_train, y_tensor_train), batch_size=batch_size, shuffle=True)
                valid_dataloader = data.DataLoader(data.TensorDataset(X_tensor_valid, y_tensor_valid), batch_size=batch_size, shuffle=True)

                optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()

                fold_train_losses = []
                fold_train_accuracies = []
                fold_train_f1_scores = []
                fold_train_balanced_accuracies = []
                fold_train_precisions = []
                fold_train_recalls = []

                fold_val_losses = []
                fold_val_accuracies = []
                fold_val_f1_scores = []
                fold_val_balanced_accuracies = []
                fold_val_precisions = []
                fold_val_recalls = []

                pbar = tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch")
                for epoch in pbar:
                    current_model.train()
                    train_loss = 0
                    correct = 0
                    total = 0
                    for batch_X, batch_y in train_dataloader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                        optimizer.zero_grad()
                        y_pred = current_model(batch_X)
                        loss = criterion(y_pred, batch_y)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        _, predicted = torch.max(y_pred.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                        true_labels = batch_y.cpu().numpy()
                        predicted_labels = predicted.cpu().numpy()

                        results[architecture[0]][dataset][imbalance_method][fold_id]['train']['labels'].extend(true_labels)
                        results['architecture_name']['dataset_name']['imbalance_method_name']['fold_number']['train']['predictions'].extend(predicted_labels)
                        results['architecture_name']['dataset_name']['imbalance_method_name']['fold_number']['train']['losses'].extend([loss.item()] * len(true_labels))

                    train_accuracy = correct / total
                    fold_train_losses.append(train_loss / len(train_dataloader))
                    fold_train_accuracies.append(train_accuracy)

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

                            true_labels = batch_y.cpu().numpy()
                            predicted_labels = predicted.cpu().numpy()

                            results['architecture_name']['dataset_name']['imbalance_method_name']['fold_number']['validation']['labels'].extend(true_labels)
                            results['architecture_name']['dataset_name']['imbalance_method_name']['fold_number']['validation']['predictions'].extend(predicted_labels)
                            results['architecture_name']['dataset_name']['imbalance_method_name']['fold_number']['validation']['losses'].extend([loss.item()] * len(true_labels))

                    val_accuracy = correct / total
                    fold_val_losses.append(val_loss / len(valid_dataloader))
                    fold_val_accuracies.append(val_accuracy)

                    # Update progress bar or print statement
                    pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_accuracy:.2f}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Acc: {val_accuracy:.2f}, Val Loss: {val_loss / len(valid_dataloader):.4f}")

                train_losses.append(fold_train_losses)
                val_losses.append(fold_val_losses)
                train_accuracies.append(fold_train_accuracies)
                val_accuracies.append(fold_val_accuracies)
            

            plot_losses(train_losses, val_losses)
            plot_accuracies(train_accuracies, val_accuracies)
            plot_metric(results, architecture[0], dataset, imbalance_method, metric='f1')