import numpy as np

def run_gradient_check(model, X_sample, Y_sample, epsilon=1e-5, num_samples=10, seed=42):
    """Centered Finite Difference Gradient Check"""
    model.forward(X_sample, Y_sample)
    model.backward()
    layers = [model.layer1, model.layer2, model.layer3]
    rng = np.random.RandomState(seed)
    results = []

    for idx, layer in enumerate(layers):
        W = layer.weights
        grad_W = layer.grad_W
        for _ in range(num_samples // 3 + 1):
            r = rng.randint(0, W.shape[0])
            c = rng.randint(0, W.shape[1])
            orig = W[r, c]

            W[r, c] = orig + epsilon
            _, _, loss_pos = model.forward(X_sample, Y_sample)

            W[r, c] = orig - epsilon
            _, _, loss_neg = model.forward(X_sample, Y_sample)

            W[r, c] = orig
            num_grad = (loss_pos - loss_neg) / (2.0 * epsilon)
            ana_grad = grad_W[r, c]

            denom = max(abs(num_grad) + abs(ana_grad), 1e-12)
            rel_error = abs(num_grad - ana_grad) / denom
            results.append({
                "layer": f"layer_{idx+1}",
                "coordinate": f"({r}, {c})",
                "analytic": float(ana_grad),
                "numeric": float(num_grad),
                "relative_error": float(rel_error)
            })
    return results[:num_samples]
