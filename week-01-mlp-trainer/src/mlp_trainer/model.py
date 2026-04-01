# Import necessary libraries
import torch.nn as nn


# Model definition
class MLP(nn.Module):
    def __init__(self, input_dim = 54, hidden_dim = [96, 64], output_dim = 7):
        super().__init__()
        layers = []
        current_input_dim = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(current_input_dim, h))
            layers.append(nn.ReLU())
            current_input_dim = h
        layers.append(nn.Linear(current_input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x

