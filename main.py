import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import model 
from read_data import load_data
from imbalanced_handling import handle_imbalanced



models = [
    # model.NeuralNet(input_dim=10, hidden_layers=[10, 10, 10], output_dim=3),
    # model.NeuralNet(input_dim=10, hidden_layers=[20, 20, 20], output_dim=3),
    ["MLP_10_10_10",[10, 10]],
    ["MLP_20_20_20",[20, 20]],  
]

datasets = [
    'ecoli1',
    'vowel0',
    'yeast3',
]

imbalance_handling_methods = [
    "none",
    #"SMOTE",
    #"random_undersampling",
    #"KDE-based_oversampling",
    #"KDE-based_loss_weighting",
    #"KDE-based_batch_balancing"
]

epochs = 100
batch_size = 16
learning_rate = 0.01


for id_architecture, architecture in enumerate(models): 
    print(f"Architecture: {architecture[0]}")

    for id_dataset, dataset in enumerate(datasets):
        print(f"Dataset: {dataset}")
        X, y = load_data(f'DATASETS/{dataset}/{dataset}.dat')
        
        current_model = model.prepare_model(architecture, X, y)
        print("Using model", architecture[0])
        print(current_model)

        for id_imbalance, imbalance_method in enumerate(imbalance_handling_methods):
            handle_imbalanced(current_model, X, y, imbalance_method)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)

            X_tensor = torch.from_numpy(X).float().to(device)
            y_tensor = torch.from_numpy(y).long().to(device)
            current_model = current_model.to(device)

            dataloader = data.DataLoader(data.TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

            optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            epoch_accuracies = []
            epoch_losses = []
            # Improved TQDM usage for cleaner output
            pbar = tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch")
            for epoch in pbar:
                current_model.train()
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    y_pred = current_model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                current_model.eval()
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for batch_X, batch_y in dataloader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        y_pred = current_model(batch_X)
                        _, predicted = torch.max(y_pred, 1)
                        total_correct += (predicted == batch_y).sum().item()
                        total_samples += batch_y.size(0)

                epoch_accuracy = total_correct / total_samples
                epoch_losses.append(epoch_loss / len(dataloader))
                epoch_accuracies.append(epoch_accuracy)

                # Update progress bar description with accuracy and loss
                pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Acc: {epoch_accuracy:.2f}, Loss: {epoch_loss / len(dataloader):.4f}")

            # Plotting accuracy and loss
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(range(epochs), [x * 100 for x in epoch_accuracies], label="Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title(f"Accuracy: {architecture[0]}, {dataset}, {imbalance_method}")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(range(epochs), epoch_losses, label="Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss: {architecture[0]}, {dataset}, {imbalance_method}")
            plt.legend()

            plt.tight_layout()
            plt.show()