import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import numpy as np

def prepare_model(model_string, X, y, activation=nn.ReLU(), dropout=0.2):
    if model_string[0][0:3] == "MLP":
        return NeuralNet(input_dim=X.shape[1], hidden_layers=model_string[1], output_dim=len(np.unique(y)), activation=activation, dropout=dropout)


class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int,
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.2) -> None:
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_layers[0]), activation]
        
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), activation, nn.Dropout(dropout)]
        
        layers += [nn.Linear(hidden_layers[-1], output_dim)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
