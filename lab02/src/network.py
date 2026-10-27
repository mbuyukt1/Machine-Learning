import numpy as np
from layers import LinearAffineBlock, LeakyReLUActivation

class DualHiddenLayerMLP:
    """Vectorized Two-Hidden-Layer Multilayer Perceptron"""
    def __init__(self, in_dim=64, hidden_dim=32, out_dim=10, seed=42):
        self.layer1 = LinearAffineBlock(in_dim, hidden_dim, seed=seed)
        self.act1 = LeakyReLUActivation(alpha=0.01)
        self.layer2 = LinearAffineBlock(hidden_dim, hidden_dim, seed=seed+1)
        self.act2 = LeakyReLUActivation(alpha=0.01)
        self.layer3 = LinearAffineBlock(hidden_dim, out_dim, seed=seed+2)
        self.probs = None
        self.Y_cache = None

    def forward(self, X, Y=None):
        z1 = self.layer1.forward(X)
        a1 = self.act1.forward(z1)
        z2 = self.layer2.forward(a1)
        a2 = self.act2.forward(z2)
        logits = self.layer3.forward(a2)

        # Numerically stable softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(logits - max_logits)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        loss = None
        if Y is not None:
            eps = 1e-12
            loss = -np.mean(np.sum(Y * np.log(probs + eps), axis=1))

        self.probs = probs
        self.Y_cache = Y
        return logits, probs, loss

    def backward(self):
        dZ3 = self.probs - self.Y_cache
        dH2 = self.layer3.backward(dZ3)
        dZ2 = self.act2.backward(dH2)
        dH1 = self.layer2.backward(dZ2)
        dZ1 = self.act1.backward(dH1)
        dX = self.layer1.backward(dZ1)
        return dX

    def step(self, lr):
        self.layer1.apply_update(lr)
        self.layer2.apply_update(lr)
        self.layer3.apply_update(lr)
