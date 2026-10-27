import torch
import torch.nn as nn

class SpatialConvClassifier(nn.Module):
    """Dual-Stage Convolutional Network with Spatial Max-Pooling"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return self.classifier(x)
