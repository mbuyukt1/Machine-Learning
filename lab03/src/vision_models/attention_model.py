import torch
import torch.nn as nn

class PatchVisionTransformer(nn.Module):
    """Patch-based Vision Transformer with Multi-Head Self-Attention"""
    def __init__(self, patch_dim=49, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2, num_classes=10):
        super().__init__()
        self.linear_projection = nn.Linear(patch_dim, embed_dim)
        encoder_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_block, num_layers=num_layers)
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Unfold 28x28 into 16 non-overlapping 7x7 patches
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7).contiguous().view(B, 16, 49)
        tokens = self.linear_projection(patches)
        hidden_states = self.transformer_encoder(tokens)
        pooled = hidden_states.mean(dim=1)
        return self.classification_head(pooled)
