"""
mlquant — Machine Learning for Quantitative Finance
====================================================
A from-scratch implementation of linear regression, logistic regression,
CAPM factor models, and a direction classifier with backtesting.

Modules
-------
linear    : Linear regression with gradient descent and ridge regularisation.
logistic  : Logistic regression with gradient descent and cross-entropy loss.
data      : Financial data download, alignment, and summary statistics.
capm      : CAPM regression, multi-factor feature engineering.
backtest  : Strategy simulation and Sharpe ratio computation.
plotting  : All plot generation, saving figures to results/.
"""

from mlquant import linear, logistic, data, capm, backtest, plotting
