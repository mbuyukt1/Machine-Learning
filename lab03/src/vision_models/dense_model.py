import torch
import torch.nn as nn

class StackedDenseClassifier(nn.Module):
    """Deep Feedforward Perceptron with Dropout Regularization"""
    def __init__(self, in_features=784, hidden_dims=(256, 128), num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.block1 = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.head = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)
