"""
mlquant.plotting
================
All plot generation for the ml-quant-finance project.

Every function follows the same contract:
  - Accepts data arrays and a save_path (str or None).
  - If save_path is provided, saves the figure to that path and closes it.
  - If save_path is None, calls plt.show() instead.
  - Returns the matplotlib Figure object.

This keeps plotting logic completely separate from analysis logic and makes
it easy to save all figures to results/ from the GUI or main.py.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Use a non-interactive backend when running headless (e.g. from the GUI
# on a thread). The GUI switches back to TkAgg for its own windows.
_STYLE = {
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Benchmark / linear regression plots
# ---------------------------------------------------------------------------

def plot_regression_fit(x, y, w, b, save_path=None):
    """Scatter plot of data with the fitted regression line."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, s=18, alpha=0.6, label="Data")
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, w * x_line + b, color="tab:red", linewidth=2,
                label=f"Fit: y = {b:.2f} + {w:.2f}x")
        ax.set_xlabel("Radio advertising spend ($k)")
        ax.set_ylabel("Sales ($k units)")
        ax.set_title("Linear Regression — Radio vs Sales")
        ax.legend()
    return _save_or_show(fig, save_path)


def plot_cost_curve(epochs, costs, title="Cost Function During Gradient Descent",
                    save_path=None):
    """Line plot of cost vs epoch number."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, costs, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost (MSE)")
        ax.set_title(title)
    return _save_or_show(fig, save_path)


def plot_prediction_convergence(epochs, predictions, x_new, save_path=None):
    """Track how the predicted value for x_new evolves over epochs."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, predictions, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Predicted sales for x = {x_new}")
        ax.set_title("Prediction Convergence During Training")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Multi-feature regression
# ---------------------------------------------------------------------------

def plot_feature_weights(theta, feature_names, save_path=None):
    """Horizontal bar chart of normalised feature weights."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["tab:green" if t >= 0 else "tab:red" for t in theta]
        ax.barh(feature_names, theta, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Weight (normalised features)")
        ax.set_title("Feature Importance — Multi-Feature Regression")
    return _save_or_show(fig, save_path)


def plot_ridge_lambda_search(lambdas, train_costs, val_costs, test_costs,
                             best_lambda, save_path=None):
    """Line plot of train/val/test cost across lambda values."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(lambdas, train_costs, label="Train", marker="o", markersize=4)
        ax.plot(lambdas, val_costs, label="Validation", marker="s", markersize=4)
        ax.plot(lambdas, test_costs, label="Test", marker="^", markersize=4,
                linestyle="--")
        ax.axvline(best_lambda, color="gray", linestyle=":", linewidth=1,
                   label=f"Best λ = {best_lambda}")
        ax.set_xlabel("λ (regularisation strength)")
        ax.set_ylabel("MSE Cost")
        ax.set_title("Ridge Regularisation — Lambda Search")
        ax.legend()
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Financial data
# ---------------------------------------------------------------------------

def plot_normalised_prices(data_dict, save_path=None):
    """
    Plot price series normalised to 100 at the start date.

    Parameters
    ----------
    data_dict : dict
        Mapping of ticker str -> pd.DataFrame with "Close" column.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for ticker, df in data_dict.items():
            normalised = 100 * df["Close"] / df["Close"].iloc[0]
            ax.plot(df.index, normalised, label=ticker, linewidth=1.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalised price (base = 100)")
        ax.set_title("Price Performance (Normalised)")
        ax.legend()
    return _save_or_show(fig, save_path)


def plot_return_distributions(returns_df, save_path=None):
    """Histogram of log return distributions for each ticker."""
    n = len(returns_df.columns)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, returns_df.columns):
            ax.hist(returns_df[col], bins=60, alpha=0.7, density=True)
            ax.set_title(col)
            ax.set_xlabel("Log return")
            ax.set_ylabel("Density")
        fig.suptitle("Return Distributions", fontsize=13)
        fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_correlation_heatmap(corr_matrix, save_path=None):
    """Heatmap of the return correlation matrix."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        tickers = corr_matrix.columns.tolist()
        im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdYlGn")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(tickers)))
        ax.set_yticks(range(len(tickers)))
        ax.set_xticklabels(tickers)
        ax.set_yticklabels(tickers)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
        ax.set_title("Return Correlation Matrix")
        ax.grid(False)
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------

def plot_capm_scatter(market_returns, stock_returns, alpha, beta, ticker,
                      save_path=None):
    """Scatter plot of stock vs market returns with the CAPM regression line."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(market_returns, stock_returns, alpha=0.3, s=8)
        x_line = np.linspace(market_returns.min(), market_returns.max(), 200)
        ax.plot(x_line, alpha + beta * x_line, color="tab:red", linewidth=2,
                label=f"α = {alpha * 100:.3f}%,  β = {beta:.3f}")
        ax.set_xlabel("SPY log return (market)")
        ax.set_ylabel(f"{ticker} log return")
        ax.set_title(f"CAPM Regression — {ticker}")
        ax.legend()
    return _save_or_show(fig, save_path)


def plot_capm_betas(capm_results, save_path=None):
    """Bar chart comparing beta values across tickers."""
    tickers = list(capm_results.keys())
    betas = [capm_results[t]["beta"] for t in tickers]
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(tickers, betas, color="steelblue", width=0.5)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="β = 1 (market)")
        for bar, val in zip(bars, betas):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Beta (β)")
        ax.set_title("CAPM Beta by Ticker")
        ax.legend()
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def plot_training_loss(costs, save_path=None):
    """Training cross-entropy loss curve."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(costs, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Logistic Regression — Training Loss")
    return _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def plot_equity_curves(backtest_results, label="", save_path=None):
    """
    Equity curve: cumulative log return of strategy vs buy-and-hold.

    Parameters
    ----------
    backtest_results : dict
        Output of backtest.run_backtest.
    label : str
        Period label for the title.
    """
    strat_sharpe = backtest_results["strategy_sharpe"]
    bench_sharpe = backtest_results["benchmark_sharpe"]
    strat_total = backtest_results["strategy_total"] * 100
    bench_total = backtest_results["benchmark_total"] * 100

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(backtest_results["strategy_cumret"], linewidth=1.5,
                label=f"Model strategy  (Sharpe = {strat_sharpe:+.2f},  "
                      f"return = {strat_total:+.1f}%)")
        ax.plot(backtest_results["benchmark_cumret"], linewidth=1.5,
                linestyle="--",
                label=f"Buy & hold  (Sharpe = {bench_sharpe:+.2f},  "
                      f"return = {bench_total:+.1f}%)")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Trading days")
        ax.set_ylabel("Cumulative log return")
        ax.set_title(f"Equity Curve — {label}")
        ax.legend(fontsize=9)
    return _save_or_show(fig, save_path)


def plot_positions(positions, returns, save_path=None):
    """
    Show when the model is invested vs in cash, overlaid on returns.
    """
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(returns, linewidth=0.8, color="steelblue")
        ax1.set_ylabel("Daily log return")
        ax1.set_title("Asset Returns")
        ax2.fill_between(range(len(positions)), positions, alpha=0.5,
                         color="tab:green", label="Long")
        ax2.set_ylabel("Position (1 = long, 0 = cash)")
        ax2.set_title("Model Positions")
        ax2.legend()
        fig.tight_layout()
    return _save_or_show(fig, save_path)
