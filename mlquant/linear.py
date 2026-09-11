
"""
mlquant.linear
==============
Linear regression implemented from scratch using batch gradient descent.
 
Covers:
  - Simple linear regression (D=1): benchmark task on Advertising data.
  - Multi-feature linear regression (D>1): matrix form with normalisation.
  - Ridge regularisation via the normal equation.
 
Indexing convention (benchmark task)
-------------------------------------
results_df.loc[k] stores the model state AFTER k gradient updates.
results_df.loc[0] is the untouched initial state (w=0, b=0).
The benchmark document's "epoch N" therefore corresponds to
results_df.loc[N + 1] in this implementation.
"""
 
import numpy as np
import pandas as pd
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
 
def load_advertising(filepath, feature_col="radio", target_col="sales"):
    """
    Load the Advertising dataset and return a single feature and the target.
 
    Parameters
    ----------
    filepath : str
        Path to Advertising.xlsx.
    feature_col : str
        Column name to use as input feature x. Default is "radio".
    target_col : str
        Column name to use as label y. Default is "sales".
 
    Returns
    -------
    x : np.ndarray, shape (N,)
    y : np.ndarray, shape (N,)
    N : int
    """
    df = pd.read_excel(filepath)
    x = df[feature_col].values
    y = df[target_col].values
    return x, y, len(x)
 
 
def load_advertising_multi(filepath):
    """
    Load all three advertising features (TV, radio, newspaper) and sales.
 
    Returns
    -------
    X : np.ndarray, shape (N, 4)
        Feature matrix with leading bias column of ones.
    y : np.ndarray, shape (N,)
    N : int
    """
    df = pd.read_excel(filepath)
    N = len(df)
    X = np.column_stack([
        np.ones(N),
        df["TV"].values,
        df["radio"].values,
        df["newspaper"].values,
    ])
    y = df["sales"].values
    return X, y, N
 
 
# ---------------------------------------------------------------------------
# Core functions — simple (D=1) regression
# ---------------------------------------------------------------------------
 
def predict_simple(w, b, x):
    """Compute y_hat = w*x + b for a scalar or array x."""
    return w * x + b
 
 
def compute_cost(y, y_hat, N):
    """Mean squared error: (1/N) * sum((y - y_hat)^2)."""
    return (1 / N) * np.sum((y - y_hat) ** 2)
 
 
def compute_gradients_simple(x, y, y_hat, N):
    """
    Gradients of MSE w.r.t. w and b for simple linear regression.
 
    Returns
    -------
    dldw, dldb : float
    """
    residuals = y - y_hat
    dldw = -(1 / N) * np.sum(2 * x * residuals)
    dldb = -(1 / N) * np.sum(2 * residuals)
    return dldw, dldb
 
 
def gradient_descent_simple(x, y, N, alpha, epochs, x_new):
    """
    Batch gradient descent for simple (D=1) linear regression.
 
    Parameters
    ----------
    x, y : np.ndarray
    N : int
    alpha : float
        Learning rate.
    epochs : int
        Training epochs in the benchmark document's sense.
        Runs epochs+2 iterations so results_df.loc[epochs+1] exists.
    x_new : float
        New x value to track prediction convergence.
 
    Returns
    -------
    results_df : pd.DataFrame
        Columns: epoch, w, b, cost, prediction.
        Row k is the state after k gradient updates.
    """
    results = np.zeros((epochs + 2, 5))
    w, b = 0.0, 0.0
 
    for epoch in range(epochs + 2):
        y_hat = predict_simple(w, b, x)
        cost = compute_cost(y, y_hat, N)
        prediction = predict_simple(w, b, x_new)
        results[epoch] = [epoch, w, b, cost, prediction]
        dldw, dldb = compute_gradients_simple(x, y, y_hat, N)
        w -= alpha * dldw
        b -= alpha * dldb
 
    return pd.DataFrame(
        results,
        columns=["epoch", "w", "b", "cost", "prediction"],
    )
 
 
def benchmark_checks(results_df, epochs):
    """
    Return a dict of benchmark values for display or logging.
 
    Keys: epoch_0_cost, epoch_1_cost, epoch_800_cost, final_prediction.
 
    The epoch_1 and epoch_800 anchors are only available when the run is
    long enough to have produced those rows; otherwise they are NaN. The
    last row produced is results_df.loc[epochs + 1].
    """
    last = epochs + 1
    return {
        "epoch_0_cost": results_df.loc[0, "cost"],
        "epoch_1_cost": results_df.loc[1, "cost"] if last >= 1 else np.nan,
        "epoch_800_cost": results_df.loc[801, "cost"] if last >= 801 else np.nan,
        "final_prediction": results_df.loc[last, "prediction"],
        "final_w": results_df.loc[last, "w"],
        "final_b": results_df.loc[last, "b"],
    }
 
 
# ---------------------------------------------------------------------------
# Core functions — multi-feature (D>1) regression
# ---------------------------------------------------------------------------
 
def predict_multi(theta, X):
    """Compute y_hat = X @ theta."""
    return X @ theta
 
 
def compute_gradient_multi(X, y, y_hat, N):
    """
    Gradient of MSE w.r.t. theta for multi-feature regression.
 
    grad = -(2/N) * X^T @ (y - y_hat)
    """
    residuals = y - y_hat
    return -(2 / N) * X.T @ residuals
 
 
def normal_equation(X, y):
    """
    Analytical solution: theta* = (X^T X)^{-1} X^T y.
 
    Uses np.linalg.solve for numerical stability over explicit inversion.
    """
    return np.linalg.solve(X.T @ X, X.T @ y)
 
 
def ridge_normal_equation(X, y, lam):
    """
    Ridge regression analytical solution: (X^T X + lam*N*I_reg)^{-1} X^T y.
 
    The bias term (theta[0]) is not penalised — I_reg has a zero in the
    top-left element.
 
    Parameters
    ----------
    lam : float
        Regularisation strength.
    """
    N = len(y)
    I_reg = np.eye(X.shape[1])
    I_reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * N * I_reg, X.T @ y)
 
 
def gradient_descent_multi(X, y, N, alpha, epochs):
    """
    Batch gradient descent for multi-feature linear regression.
 
    Parameters
    ----------
    X : np.ndarray, shape (N, D+1)
        Feature matrix including bias column.
    y : np.ndarray, shape (N,)
    N : int
    alpha : float
    epochs : int
 
    Returns
    -------
    results_df : pd.DataFrame
        Columns: epoch, cost, b, w_TV, w_radio, w_newspaper.
        Row k is the state after k updates.
    """
    theta = np.zeros(X.shape[1])
    n_rows = epochs + 2
    results = np.zeros((n_rows, 6))
 
    for epoch in range(n_rows):
        y_hat = predict_multi(theta, X)
        cost = compute_cost(y, y_hat, N)
        results[epoch, 0] = epoch
        results[epoch, 1] = cost
        results[epoch, 2:] = theta
        grad = compute_gradient_multi(X, y, y_hat, N)
        theta -= alpha * grad
 
    return pd.DataFrame(
        results,
        columns=["epoch", "cost", "b", "w_TV", "w_radio", "w_newspaper"],
    )
 
 
# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
 
def normalise(X_train, X_val, X_test):
    """
    Z-score normalise feature columns using training set statistics only.
 
    The bias column (index 0) is left unchanged.
 
    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
 
    Returns
    -------
    X_train_n, X_val_n, X_test_n : np.ndarray
        Normalised matrices.
    mu : np.ndarray, shape (D,)
        Feature means from training data.
    sigma : np.ndarray, shape (D,)
        Feature standard deviations from training data.
    """
    mu = X_train[:, 1:].mean(axis=0)
    sigma = X_train[:, 1:].std(axis=0)
 
    def _standardise(X):
        X_n = X.copy()
        X_n[:, 1:] = (X[:, 1:] - mu) / sigma
        return X_n
 
    return _standardise(X_train), _standardise(X_val), _standardise(X_test), mu, sigma
 
 
def normalise_single(X, mu, sigma):
    """Apply stored mu/sigma normalisation to a new matrix."""
    X_n = X.copy()
    X_n[:, 1:] = (X[:, 1:] - mu) / sigma
    return X_n
 
 
# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------
 
def train_val_test_split(X, y, train_frac=0.7, val_frac=0.15, seed=1):
    """
    Randomly shuffle and split into train, validation, and test sets.
 
    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    N = len(y)
    idx = np.random.permutation(N)
    train_end = int(N * train_frac)
    val_end = int(N * (train_frac + val_frac))
    ti, vi, tsi = idx[:train_end], idx[train_end:val_end], idx[val_end:]
    return X[ti], y[ti], X[vi], y[vi], X[tsi], y[tsi]
 
 
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
 
def r_squared(y, y_hat):
    """Coefficient of determination R²."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot
 
 
def rmse(y, y_hat):
    """Root mean squared error."""
    return np.sqrt(np.mean((y - y_hat) ** 2))
 
