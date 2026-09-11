"""
mlquant.data
============
Financial data acquisition and summary statistics.

Uses yfinance to download daily OHLCV data from Yahoo Finance.
All returns are computed as log returns:

    r_t = ln(P_t / P_{t-1})

Log returns are time-additive and approximately symmetric, making them
more suitable for statistical modelling than simple returns.
"""

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf


# ---------------------------------------------------------------------------
# Download and alignment
# ---------------------------------------------------------------------------

def get_stock_data(ticker, start, end):
    """
    Download daily closing prices and compute log returns.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol, e.g. "AAPL", "SPY", "^VIX".
    start, end : str
        Date strings in "YYYY-MM-DD" format.

    Returns
    -------
    df : pd.DataFrame
        Columns: Close, LogReturn. Index is DatetimeIndex.
        First row (which would have NaN log return) is dropped.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = pd.DataFrame()
    df["Close"] = raw["Close"].squeeze()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def align_returns(data_dict):
    """
    Align log return series from multiple tickers to a common date index.

    Parameters
    ----------
    data_dict : dict
        Mapping of ticker str -> pd.DataFrame (from get_stock_data).

    Returns
    -------
    returns : pd.DataFrame
        Columns are ticker names. Only dates present for all tickers are kept.
    """
    series = [df["LogReturn"].rename(ticker) for ticker, df in data_dict.items()]
    returns = pd.concat(series, axis=1, join="inner").dropna()
    return returns


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(returns_df):
    """
    Compute per-ticker summary statistics relevant to quant finance.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Each column is a return series.

    Returns
    -------
    summary : pd.DataFrame
        Columns: mean_pct, daily_vol_pct, ann_vol_pct, skewness, kurtosis.
        Volatility is annualised using sqrt(252) — the number of trading days
        per calendar year.
    """
    rows = []
    for col in returns_df.columns:
        r = returns_df[col]
        rows.append({
            "ticker": col,
            "mean_%": r.mean() * 100,
            "daily_vol_%": r.std() * 100,
            "ann_vol_%": r.std() * np.sqrt(252) * 100,
            "skewness": stats.skew(r),
            "kurtosis": stats.kurtosis(r),
        })
    return pd.DataFrame(rows).set_index("ticker")


def correlation_matrix(returns_df):
    """Return the Pearson correlation matrix of return series."""
    return returns_df.corr()
