"""
mlquant.capm
============
Capital Asset Pricing Model (CAPM) regression and multi-factor feature
engineering.
 
CAPM models a stock's excess return as a linear function of the market's
excess return:
 
    R_i = alpha + beta * R_m + epsilon
 
where:
    alpha  — excess return unexplained by market risk (the "holy grail")
    beta   — sensitivity to market moves
    R_m    — market return proxy (typically SPY)
 
Beta is mathematically identical to the slope from simple linear regression:
 
    beta = Cov(R_i, R_m) / Var(R_m)
 
This module also builds the lagged feature matrix used for direction
classification in mlquant.logistic.
"""
 
import numpy as np
import pandas as pd
from scipy import stats
 
 
# ---------------------------------------------------------------------------
# CAPM regression
# ---------------------------------------------------------------------------
 
def capm_regression(market_returns, stock_returns):
    """
    Fit CAPM via OLS: stock_returns = alpha + beta * market_returns.
 
    Uses scipy.stats.linregress for the core fit, then computes
    standard errors and t-statistics manually so they are available
    for hypothesis testing.
 
    Parameters
    ----------
    market_returns : np.ndarray
        Daily log returns of the market proxy (e.g. SPY).
    stock_returns : np.ndarray
        Daily log returns of the stock being analysed.
 
    Returns
    -------
    results : dict
        Keys: alpha, beta, t_alpha, t_beta, p_alpha, p_beta, r_squared.
    """
    slope, intercept, r, _, _ = stats.linregress(market_returns, stock_returns)
    beta, alpha, r_squared = slope, intercept, r ** 2
 
    N = len(market_returns)
    residuals = stock_returns - (alpha + beta * market_returns)
    s2 = np.sum(residuals ** 2) / (N - 2)
    mkt_var = np.sum((market_returns - market_returns.mean()) ** 2)
 
    se_beta = np.sqrt(s2 / mkt_var)
    se_alpha = np.sqrt(s2 * (1 / N + market_returns.mean() ** 2 / mkt_var))
 
    t_beta = beta / se_beta
    t_alpha = alpha / se_alpha
    p_beta = 2 * stats.t.sf(abs(t_beta), df=N - 2)
    p_alpha = 2 * stats.t.sf(abs(t_alpha), df=N - 2)
 
    return {
        "alpha": alpha,
        "beta": beta,
        "t_alpha": t_alpha,
        "t_beta": t_beta,
        "p_alpha": p_alpha,
        "p_beta": p_beta,
        "r_squared": r_squared,
    }
 
 
def run_capm_multiple(aligned_returns, market_col="SPY"):
    """
    Run CAPM for every non-market column in aligned_returns.
 
    Parameters
    ----------
    aligned_returns : pd.DataFrame
        Columns are tickers; one column must be the market proxy.
    market_col : str
        Name of the market column to regress against.
 
    Returns
    -------
    results : dict
        Mapping of ticker -> capm_regression result dict.
    """
    market = aligned_returns[market_col].values
    results = {}
    for col in aligned_returns.columns:
        if col == market_col:
            continue
        results[col] = capm_regression(market, aligned_returns[col].values)
    return results
 
 
# ---------------------------------------------------------------------------
# Feature engineering for direction classification
# ---------------------------------------------------------------------------
 
def build_features(stock_df, market_df):
    """
    Build a lagged feature matrix for predicting next-day stock direction.
 
    All features are lagged by at least one day to prevent look-ahead bias.
    The target column is today's (unlagged) log return — to be binarised
    by the caller.
 
    Features
    --------
    market_return : SPY log return, lagged 1 day.
    momentum_5    : Stock's 5-day cumulative return, lagged 1 day.
    momentum_21   : Stock's 21-day cumulative return, lagged 1 day.
    volatility_21 : Stock's 21-day rolling std of returns, lagged 1 day.
    return_lag1   : Stock's return from 1 day ago.
    return_lag2   : Stock's return from 2 days ago.
    target        : Today's stock log return (unlagged).
 
    Parameters
    ----------
    stock_df : pd.DataFrame
        From data.get_stock_data. Must contain "LogReturn".
    market_df : pd.DataFrame
        From data.get_stock_data for the market proxy (e.g. SPY).
 
    Returns
    -------
    features : pd.DataFrame
        Rows with any NaN (due to rolling windows) are dropped.
    """
    features = pd.DataFrame(index=stock_df.index)
    features["market_return"] = market_df["LogReturn"].shift(1)
    features["momentum_5"] = stock_df["LogReturn"].rolling(5).sum().shift(1)
    features["momentum_21"] = stock_df["LogReturn"].rolling(21).sum().shift(1)
    features["volatility_21"] = stock_df["LogReturn"].rolling(21).std().shift(1)
    features["return_lag1"] = stock_df["LogReturn"].shift(1)
    features["return_lag2"] = stock_df["LogReturn"].shift(2)
    features["target"] = stock_df["LogReturn"]
    return features.dropna()
 
 
FEATURE_COLS = [
    "market_return",
    "momentum_5",
    "momentum_21",
    "volatility_21",
    "return_lag1",
    "return_lag2",
]
 
