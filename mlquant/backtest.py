"""
mlquant.backtest
================
Strategy simulation and performance evaluation.

Implements a long/cash strategy: hold the asset when the model predicts
an up day, stay in cash otherwise. No short selling, no leverage.

Strategy return on day t:
    r_strategy_t = position_t * r_asset_t
    position_t   = 1 if p_hat_t >= 0.5 else 0

The Sharpe ratio is the primary risk-adjusted performance metric:

    Sharpe_annual = (mean_daily_return / std_daily_return) * sqrt(252)

where 252 is the number of trading days per year.

Limitations (always reported alongside results)
-----------------------------------------------
- No transaction costs (bid-ask spread, commissions, market impact).
- Single-asset strategy with no position sizing.
- Look-ahead bias has been controlled for via chronological splits and
  lagged features, but cannot be fully eliminated in a backtest.
- Non-stationarity: past performance does not guarantee future results.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def run_backtest(p_hat, returns):
    """
    Simulate a long/cash strategy from predicted probabilities.

    Parameters
    ----------
    p_hat : np.ndarray
        Predicted class-1 probabilities from the logistic classifier.
    returns : np.ndarray
        Actual daily log returns for the asset over the same period.

    Returns
    -------
    results : dict
        Keys:
            positions         : np.ndarray — 1 (long) or 0 (cash) each day.
            strategy_returns  : np.ndarray — daily strategy log returns.
            benchmark_returns : np.ndarray — buy-and-hold daily returns.
            strategy_cumret   : np.ndarray — cumulative log return (strategy).
            benchmark_cumret  : np.ndarray — cumulative log return (B&H).
            strategy_sharpe   : float — annualised Sharpe ratio (strategy).
            benchmark_sharpe  : float — annualised Sharpe ratio (B&H).
            strategy_total    : float — total return over the period.
            benchmark_total   : float — total return (B&H) over the period.
            pct_invested      : float — fraction of days holding the asset.
    """
    positions = (p_hat >= 0.5).astype(float)
    strategy_returns = positions * returns
    benchmark_returns = returns

    strat_cumret = np.cumsum(strategy_returns)
    bench_cumret = np.cumsum(benchmark_returns)

    return {
        "positions": positions,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "strategy_cumret": strat_cumret,
        "benchmark_cumret": bench_cumret,
        "strategy_sharpe": _annualised_sharpe(strategy_returns),
        "benchmark_sharpe": _annualised_sharpe(benchmark_returns),
        "strategy_total": float(np.exp(strat_cumret[-1]) - 1),
        "benchmark_total": float(np.exp(bench_cumret[-1]) - 1),
        "pct_invested": float(positions.mean()),
    }


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

def _annualised_sharpe(daily_returns, trading_days=252):
    """Annualised Sharpe ratio assuming zero risk-free rate."""
    std = daily_returns.std()
    if std == 0:
        return 0.0
    return (daily_returns.mean() / std) * np.sqrt(trading_days)


def sharpe_ratio(daily_returns, trading_days=252):
    """Public wrapper around _annualised_sharpe."""
    return _annualised_sharpe(daily_returns, trading_days)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def backtest_summary(results, label=""):
    """
    Return a formatted summary string from a run_backtest result dict.

    Parameters
    ----------
    results : dict
        Output of run_backtest.
    label : str
        Period label, e.g. "Test set (2023)".

    Returns
    -------
    lines : list of str
    """
    lines = [
        f"  {label}",
        f"    Strategy  — Sharpe: {results['strategy_sharpe']:+.3f}   "
        f"Total return: {results['strategy_total'] * 100:+.1f}%   "
        f"Days invested: {results['pct_invested'] * 100:.0f}%",
        f"    Benchmark — Sharpe: {results['benchmark_sharpe']:+.3f}   "
        f"Total return: {results['benchmark_total'] * 100:+.1f}%",
    ]
    return lines
