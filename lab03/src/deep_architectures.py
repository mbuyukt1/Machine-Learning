import torch
import torch.nn as nn

class LinearPerceptronClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(x)

class SpatialConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stage = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.linear_head = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        return self.linear_head(self.conv_stage(x))

class SelfAttentionTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_mapper = nn.Linear(49, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc_out = nn.Linear(64, 10)
    def forward(self, x):
        B = x.shape[0]
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7).contiguous().view(B, 16, 49)
        tokens = self.patch_mapper(patches)
        h = self.transformer(tokens)
        pooled = h.mean(dim=1)
        return self.fc_out(pooled)
