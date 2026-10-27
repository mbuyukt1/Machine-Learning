import numpy as np

class AffineTransformLayer:
    def __init__(self, in_features, out_features, sigma=0.01):
        self.W = np.random.normal(0.0, sigma, (in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X_in = None

    def forward(self, X):
        self.X_in = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        self.dW = np.dot(self.X_in.T, dZ)
        self.db = np.sum(dZ, axis=0)
        return np.dot(dZ, self.W.T)

    def apply_update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLUSubgradient:
    def __init__(self):
        self.cached_Z = None

    def forward(self, Z):
        self.cached_Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        return dA * (self.cached_Z > 0.0)

class SoftmaxCategoricalCrossEntropy:
    def __init__(self, eps=1e-15):
        self.eps = eps
        self.P = None
        self.Y = None

    def forward(self, Z, Y):
        shift = Z - np.max(Z, axis=-1, keepdims=True)
        exp_z = np.exp(shift)
        self.P = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        self.Y = Y
        N = Z.shape[0]
        loss = -np.sum(Y * np.log(self.P + self.eps)) / N
        return float(loss)

    def backward(self):
        N = self.Y.shape[0]
        return (self.P - self.Y) / N

class TwoLayerPerceptron:
    def __init__(self, in_dim=64, hidden_dim=16, out_dim=10, seed=42):
        if seed is not None:
            np.random.seed(seed)
        self.l1 = AffineTransformLayer(in_dim, hidden_dim)
        self.a1 = ReLUSubgradient()
        self.l2 = AffineTransformLayer(hidden_dim, hidden_dim)
        self.a2 = ReLUSubgradient()
        self.l3 = AffineTransformLayer(hidden_dim, out_dim)
        self.loss_fn = SoftmaxCategoricalCrossEntropy()

    def forward(self, X, Y=None):
        z1 = self.l1.forward(X)
        h1 = self.a1.forward(z1)
        z2 = self.l2.forward(h1)
        h2 = self.a2.forward(z2)
        z3 = self.l3.forward(h2)
        if Y is not None:
            loss = self.loss_fn.forward(z3, Y)
            return z3, self.loss_fn.P, loss
        shift = z3 - np.max(z3, axis=-1, keepdims=True)
        probs = np.exp(shift) / np.sum(np.exp(shift), axis=-1, keepdims=True)
        return z3, probs

    def backward(self):
        dZ3 = self.loss_fn.backward()
        dH2 = self.l3.backward(dZ3)
        dZ2 = self.a2.backward(dH2)
        dH1 = self.l2.backward(dZ2)
        dZ1 = self.a1.backward(dH1)
        dX = self.l1.backward(dZ1)
        return dX

    def step(self, lr):
        self.l1.apply_update(lr)
        self.l2.apply_update(lr)
        self.l3.apply_update(lr)
