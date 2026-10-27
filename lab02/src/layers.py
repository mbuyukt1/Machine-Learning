import numpy as np

class LinearAffineBlock:
    """Vectorized Affine Transformation with Xavier Initialization"""
    def __init__(self, in_features, out_features, seed=42):
        rng = np.random.RandomState(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = rng.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros((1, out_features))
        self.grad_W = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)
        self.inputs = None

    def forward(self, X):
        self.inputs = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dOut):
        N = self.inputs.shape[0]
        self.grad_W = np.dot(self.inputs.T, dOut) / N
        self.grad_b = np.sum(dOut, axis=0, keepdims=True) / N
        return np.dot(dOut, self.weights.T)

    def apply_update(self, lr):
        self.weights -= lr * self.grad_W
        self.bias -= lr * self.grad_b

class LeakyReLUActivation:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self, Z):
        self.mask = (Z > 0)
        return np.where(self.mask, Z, self.alpha * Z)

    def backward(self, dOut):
        return np.where(self.mask, dOut, self.alpha * dOut)
