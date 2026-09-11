# ML Quant Finance
William Thompson : Lancaster University

A from-scratch implementation of machine learning methods applied to quantitative finance.

Every algorithm — gradient descent, logistic regression, ridge regularisation, CAPM regression — is implemented from first principles using only NumPy and pandas. No scikit-learn or ML frameworks.

---

## What it does

| Module | Description |
|---|---|
| **Benchmark task** | Simple linear regression (D=1) on radio advertising spend → sales, verified against known benchmarks |
| **Multi-feature regression** | Extends to D=3 features with matrix operations, feature normalisation, and ridge regularisation |
| **CAPM** | Fits the Capital Asset Pricing Model to real stock data (AAPL, MSFT, NVDA), recovering alpha, beta, t-statistics, and R² |
| **Direction classifier** | Logistic regression trained to predict next-day stock direction from lagged features |
| **Backtest** | Simulates a long/cash trading strategy from classifier predictions and computes the Sharpe ratio |

---

## Project structure

```
ml-quant-finance/
├── main.py                 # Run the full pipeline
├── gui.py                  # Interactive tkinter GUI
├── requirements.txt
├── data/
│   └── Advertising.xlsx    # Advertising dataset (Burkov, 2019)
├── results/                # Auto-generated plots saved here
├── mlquant/                # The analysis package
│   ├── linear.py           # Linear regression, gradient descent, ridge
│   ├── logistic.py         # Logistic regression, cross-entropy, evaluation
│   ├── data.py             # Financial data download and summary stats
│   ├── capm.py             # CAPM regression and feature engineering
│   ├── backtest.py         # Strategy simulation and Sharpe ratio
│   └── plotting.py         # All matplotlib figures
└── report/
    └── report.tex          # LaTeX project report
```

---

## Getting started

```bash
git clone https://github.com/will3t/ml-quant-finance
cd ml-quant-finance
pip install -r requirements.txt
```

Place `Advertising.xlsx` in the `data/` folder, then either:

**Run the full pipeline (saves all plots to results/):**
```bash
python main.py
```

**Launch the interactive GUI:**
```bash
python gui.py
```

---

## Key results

| Analysis | Result |
|---|---|
| Benchmark prediction (x=16.5, 5000 epochs, α=0.001) | ŷ ≈ 12.395 |
| Multi-feature R² (Advertising, D=3) | 0.89 |
| Optimal ridge λ | 0.04 |
| NVDA beta (2018–2023) | 1.77 |
| NVDA CAPM alpha (p-value) | 0.209 (not significant) |
| Direction classifier accuracy (test) | ~52% |
| Strategy Sharpe (test, 2023) | 0.90 vs 1.99 buy-and-hold |

---

## Mathematical background

All mathematics is derived from first principles in the accompanying report (`report/report.tex`). Key topics covered:

- Mean squared error and maximum likelihood equivalence under Gaussian noise
- Gradient descent derivation and convergence
- Matrix form of multi-feature regression and the normal equation
- Ridge regularisation geometry and the bias-variance tradeoff
- Logistic regression and binary cross-entropy
- CAPM as a linear regression problem; hypothesis testing on alpha
- Sharpe ratio and equity curve construction

---

## Dependencies

- **numpy** — all numerical computation
- **pandas** — data loading and alignment
- **matplotlib** — visualisation
- **scipy** — t-distribution for hypothesis testing, skewness/kurtosis
- **yfinance** — financial data download
- **openpyxl** — reading the Advertising xlsx file

No ML frameworks (scikit-learn, TensorFlow, PyTorch) are used.

---

## Limitations

The direction classifier produces results consistent with semi-strong market efficiency for large-cap liquid stocks — simple lagged price features are insufficient to reliably predict next-day direction. The backtest demonstrates that even a weak signal can provide downside protection in bear markets (validation Sharpe −0.23 vs buy-and-hold −0.34 during 2022) while underperforming in strong bull markets (test Sharpe 0.90 vs 1.99 during 2023).

All backtests exclude transaction costs and assume perfect execution. Results should not be interpreted as investment advice.
