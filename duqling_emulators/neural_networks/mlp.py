"""
A basic MLP that can be constructed using a variety of layer shapes.
"""

from typing import List

import torch
from torch import nn

class MLP(nn.Module):
    """
    The general architecture of an MLP with customizable layer shapes.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str='ReLU'):
        """
        Args:
            input_dim: Size of input features.
            hidden_dims: List of hidden layer widths.
            activation: Activation function name.
        """
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [1]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(getattr(nn, activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """Forward pass through the network"""
        return self.net(x)

def train_mlp(model, X_train, y_train, X_val, y_val, lr, max_epochs=1000):
    X_train, y_train = X_train.float(), y_train.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss, patience, counter = float('inf'), 50, 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()

            if val_loss < best_loss:
                best_loss, counter = val_loss, 0
            else:
                counter += 10
                if counter >= patience:
                    break

    return best_loss
