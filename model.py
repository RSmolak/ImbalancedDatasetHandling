import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List



class NeuralNet(nn.Module):
    """
    PyTorch implementation of a multi-layer perceptron (MLP)
    """
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int,
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.2) -> None:
        """
        Initialize the model's layers

        :param input_dim: number of input features
        :param hidden_layers: list of hidden layer sizes
        :param output_dim: number of output features
        :param activation: activation function to use in hidden layers
        :param dropout: dropout rate
        """
        super().__init__()

        if input_dim < 1:
            raise ValueError("Invalid input dimension: input_dim must be greater than 0")
        if not all(hd > 0 for hd in hidden_layers):
            raise ValueError("Invalid hidden layer size: all hidden layer sizes must be greater than 0")
        if output_dim < 1:
            raise ValueError("Invalid output dimension: output_dim must be greater than 0")

        layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            *[
                layer
                for i in range(len(hidden_layers) - 1)
                for layer in [
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                    activation,
                    nn.Dropout(dropout)
                ]
            ],
            nn.Linear(hidden_layers[-1], output_dim)
        )

        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model

        :param x: input tensor
        :return: output tensor
        """
        return self.layers(x)


    def fit(self, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
        """
        Funkcja trenująca model.

        :param model: Model do trenowania.
        :param train_loader: DataLoader dla danych treningowych.
        :param val_loader: DataLoader dla danych walidacyjnych.
        :param criterion: Funkcja straty.
        :param optimizer: Optymalizator.
        :param num_epochs: Liczba epok treningowych.
        :param device: Urządzenie na którym odbywa się trening ('cpu' lub 'cuda').
        """
        self.to(device)
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Total training loss: {total_loss}')

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = self(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Total validation loss: {val_loss}')