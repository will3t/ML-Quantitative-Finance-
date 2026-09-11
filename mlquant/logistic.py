"""
mlquant.logistic
================
Logistic regression implemented from scratch using batch gradient descent
and binary cross-entropy loss.

The gradient has the same matrix structure as linear regression — only
the prediction step differs (sigmoid replaces the identity function).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation: 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(theta, X):
    """Return predicted class-1 probabilities for feature matrix X."""
    return sigmoid(X @ theta)


def compute_cost(y, p_hat):
    """
    Binary cross-entropy loss.

    L = -(1/N) * sum(y * log(p) + (1-y) * log(1-p))

    A small epsilon prevents log(0).
    """
    eps = 1e-7
    N = len(y)
    return -(1 / N) * np.sum(
        y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps)
    )


def compute_gradient(X, y, p_hat):
    """
    Gradient of cross-entropy w.r.t. theta.

    grad = (1/N) * X^T @ (p_hat - y)

    Identical structure to the linear regression gradient — only p_hat
    differs (sigmoid output vs linear prediction).
    """
    N = len(y)
    return (1 / N) * X.T @ (p_hat - y)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def gradient_descent(X, y, alpha, epochs):
    """
    Batch gradient descent for logistic regression.

    Parameters
    ----------
    X : np.ndarray, shape (N, D+1)
        Feature matrix including bias column.
    y : np.ndarray, shape (N,)
        Binary labels (0 or 1).
    alpha : float
        Learning rate.
    epochs : int
        Number of training epochs.

    Returns
    -------
    theta : np.ndarray, shape (D+1,)
        Trained parameter vector.
    costs : list of float
        Cross-entropy cost recorded at each epoch.
    """
    theta = np.zeros(X.shape[1])
    costs = []

    for _ in range(epochs):
        p_hat = predict_proba(theta, X)
        costs.append(compute_cost(y, p_hat))
        theta -= alpha * compute_gradient(X, y, p_hat)

    return theta, costs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(X, y, theta):
    """
    Compute classification metrics and confusion matrix.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray — true binary labels
    theta : np.ndarray

    Returns
    -------
    metrics : dict
        Keys: accuracy, precision, recall, tp, tn, fp, fn.
    """
    p_hat = predict_proba(theta, X)
    y_pred = (p_hat >= 0.5).astype(int)

    tp = int(np.sum((y_pred == 1) & (y == 1)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))

    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Time-series split (no shuffling — chronological order preserved)
# ---------------------------------------------------------------------------

def time_series_split(X, y, train_frac=0.7, val_frac=0.15):
    """
    Chronological train / validation / test split.

    No shuffling — future data must never appear in the training set.

    Parameters
    ----------
    train_frac : float, default 0.70
    val_frac   : float, default 0.15
        Remaining fraction (1 - train_frac - val_frac) goes to the test set.
    """
    N = len(y)
    train_end = int(N * train_frac)
    val_end = int(N * (train_frac + val_frac))
    return (
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:],
    )


# ---------------------------------------------------------------------------
# Feature normalisation (reused from linear but kept here for independence)
# ---------------------------------------------------------------------------

def normalise(X_train, X_val, X_test):
    """
    Z-score normalise feature columns using training set statistics only.

    Bias column (index 0) is left unchanged.
    """
    mu = X_train[:, 1:].mean(axis=0)
    sigma = X_train[:, 1:].std(axis=0)

    def _std(X):
        X_n = X.copy()
        X_n[:, 1:] = (X[:, 1:] - mu) / sigma
        return X_n

    return _std(X_train), _std(X_val), _std(X_test), mu, sigma
