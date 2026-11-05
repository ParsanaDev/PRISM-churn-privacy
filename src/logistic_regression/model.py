"""Logistic regression trained with mini-batch gradient descent.

Adapted from the course notebook Cross-Fold-Validation-LR.ipynb (posted on Brightspace).
Changes: the weight vector is sized to the number of features (was fixed at 3), and
train_lr uses its loss_function argument (the original always used Hossein's error).
"""
import numpy as np

from logistic_regression.config import SEED


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# Hossein's error E = (1 - y_pred) for positives, y_pred for negatives; returns dE/dz
def loss_function_hossein(y_batch, y_pred):
    return (1 - 2 * y_batch) * y_pred * (1 - y_pred)


# Standard binary cross-entropy; its gradient w.r.t. z simplifies to (y_pred - y)
def loss_function_standard(y_batch, y_pred):
    return y_pred - y_batch


LOSS_FUNCTIONS = {"standard": loss_function_standard, "hossein": loss_function_hossein}


def train_lr(X, y_train, learning_rate, epochs, batch_size, loss_function):
    """Return weights [w_1..w_d, bias] learned by mini-batch gradient descent."""
    n, d = X.shape
    rng = np.random.default_rng(SEED)
    w = rng.uniform(-0.01, 0.01, size=d + 1)  # randomly initialize weights + bias
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            X_aug = np.column_stack((X[idx], np.ones(len(idx))))  # add constant 1 (bias)
            y_pred = sigmoid(X_aug @ w)                            # forward pass
            g = loss_function(y_train[idx], y_pred)
            w -= learning_rate * (X_aug.T @ g / len(idx))          # gradient descent
    return w


def predict_proba(X, w):
    return sigmoid(np.column_stack((X, np.ones(len(X)))) @ w)


if __name__ == "__main__":
    # self-check: separable toy data must be learned almost perfectly
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    w = train_lr(X, y, 0.5, 100, 32, loss_function_standard)
    acc = ((predict_proba(X, w) >= 0.5) == y).mean()
    assert acc > 0.95, acc
    print("self-check passed, accuracy", acc)
