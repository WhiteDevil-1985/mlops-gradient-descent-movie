import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def mse_loss(y_hat, y):
    diff = y_hat - y
    return np.mean(diff * diff)

def compute_gradients(X, y, w, b):

    z = X @ w + b
    y_hat = sigmoid(z)

    n = X.shape[0]
    dL_dyhat = 2.0 * (y_hat - y) / n
    dyhat_dz = y_hat * (1.0 - y_hat)
    dL_dz = dL_dyhat * dyhat_dz

    grad_w = X.T @ dL_dz
    grad_b = np.sum(dL_dz)

    loss = mse_loss(y_hat, y)
    return grad_w, grad_b, loss

def train_vanilla_gd(X, y, lr=0.01, epochs=1000):
    rng = np.random.default_rng(seed=0)
    w = rng.normal(0, 0.1, size=2)
    b = 0.0

    losses = []
    for epoch in range(epochs):
        grad_w, grad_b, loss = compute_gradients(X, y, w, b)
        w = w - lr * grad_w
        b = b - lr * grad_b
        losses.append(loss)
    return w, b, losses

def train_momentum_gd(X, y, lr=0.01, epochs=1000, momentum=0.9):
    rng = np.random.default_rng(seed=0)
    w = rng.normal(0, 0.1, size=2)
    b = 0.0

    v_w = np.zeros_like(w)
    v_b = 0.0
    losses = []

    for epoch in range(epochs):
        grad_w, grad_b, loss = compute_gradients(X, y, w, b)
        v_w = momentum * v_w + lr * grad_w
        v_b = momentum * v_b + lr * grad_b
        w = w - v_w
        b = b - v_b
        losses.append(loss)
    return w, b, losses

def train_nesterov_gd(X, y, lr=0.01, epochs=1000, momentum=0.9):
    rng = np.random.default_rng(seed=0)
    w = rng.normal(0, 0.1, size=2)
    b = 0.0

    v_w = np.zeros_like(w)
    v_b = 0.0
    losses = []

    for epoch in range(epochs):
        w_look = w - momentum * v_w
        b_look = b - momentum * v_b

        grad_w, grad_b, loss = compute_gradients(X, y, w_look, b_look)
        v_w = momentum * v_w + lr * grad_w
        v_b = momentum * v_b + lr * grad_b

        w = w - v_w
        b = b - v_b
        losses.append(loss)
    return w, b, losses
