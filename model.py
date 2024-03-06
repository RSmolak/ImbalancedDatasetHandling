import torch
import torch.nn as nn
from typing import List
from torch_geometric.nn import GeneralizedMeanPooling


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
