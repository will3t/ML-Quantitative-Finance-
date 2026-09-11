"""
ml-quant-finance — main.py
==========================
Programmatic entry point for the full analysis pipeline.

Can be run directly:
    python main.py

Or imported and called from the GUI:
    from main import run_benchmark, run_multi_feature, run_capm, run_classifier

Every run_* function accepts a `log` callable (default: print) so the GUI
can redirect output to its text widget, and a `results_dir` string so plots
are saved to the correct folder.
"""

import os
import numpy as np

from mlquant import linear, logistic, data, capm, backtest, plotting

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ADVERTISING_PATH = os.path.join(DATA_DIR, "Advertising.xlsx")


def _results_path(filename, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)


# ---------------------------------------------------------------------------
# 1. Benchmark task — simple linear regression (D=1)
# ---------------------------------------------------------------------------

def run_benchmark(
    alpha=0.001,
    epochs=5000,
    x_new=16.5,
    log=print,
    results_dir=RESULTS_DIR,
):
    """
    Individual benchmark task.

    Fits a simple linear regression (radio spend -> sales) via gradient
    descent and verifies against the known benchmark values.

    Parameters
    ----------
    alpha : float   Learning rate.
    epochs : int    Training epochs.
    x_new : float   New x value to predict after training.
    log : callable  Output function (print or GUI text widget).
    results_dir : str  Directory to save plots.

    Returns
    -------
    checks : dict   Benchmark verification values.
    """
    log("=" * 60)
    log("BENCHMARK TASK — Simple Linear Regression (D=1)")
    log("=" * 60)

    x, y, N = linear.load_advertising(ADVERTISING_PATH)
    log(f"Loaded Advertising.xlsx — {N} data points")

    results_df = linear.gradient_descent_simple(x, y, N, alpha, epochs, x_new)
    checks = linear.benchmark_checks(results_df, epochs)

    log(f"\nInitial state  (epoch 0) — cost = {checks['epoch_0_cost']:.4f}")
    log(f"After 1 update (epoch 1) — cost = {checks['epoch_1_cost']:.4f}  "
        f"(expect ~92.3208)")
    log(f"After 800 upd. (epoch 800) — cost = {checks['epoch_800_cost']:.4f}  "
        f"(expect ~27.9919)")
    log(f"\nFinal prediction for x = {x_new} after {epochs} epochs:")
    log(f"  y_hat = {checks['final_prediction']:.4f}")
    log(f"  w = {checks['final_w']:.4f},  b = {checks['final_b']:.4f}")

    # Plots
    plotting.plot_regression_fit(
        x, y,
        checks["final_w"], checks["final_b"],
        save_path=_results_path("01_benchmark_fit.png", results_dir),
    )
    plotting.plot_cost_curve(
        results_df["epoch"].values,
        results_df["cost"].values,
        title="Benchmark — Cost During Gradient Descent",
        save_path=_results_path("02_benchmark_cost.png", results_dir),
    )
    plotting.plot_prediction_convergence(
        results_df["epoch"].values,
        results_df["prediction"].values,
        x_new,
        save_path=_results_path("03_benchmark_prediction.png", results_dir),
    )

    log("\nPlots saved to results/")
    return checks


# ---------------------------------------------------------------------------
# 2. Multi-feature regression (D=3) on Advertising data
# ---------------------------------------------------------------------------

def run_multi_feature(
    alpha=0.001,
    epochs=200_000,
    lam_search=None,
    log=print,
    results_dir=RESULTS_DIR,
):
    """
    Multi-feature linear regression on the full Advertising dataset (D=3).

    Uses the normal equation for the analytical solution and gradient descent
    for comparison. Includes ridge regularisation lambda search.

    Parameters
    ----------
    lam_search : list or None
        Lambda values to test. Defaults to [0, 0.01, ..., 0.1].
    """
    if lam_search is None:
        lam_search = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
                      0.06, 0.07, 0.08, 0.09, 0.10]

    log("\n" + "=" * 60)
    log("MULTI-FEATURE REGRESSION — Advertising (D=3)")
    log("=" * 60)

    X, y, N = linear.load_advertising_multi(ADVERTISING_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = linear.train_val_test_split(X, y)
    N_train = len(y_train)
    X_train_n, X_val_n, X_test_n, mu, sigma = linear.normalise(X_train, X_val, X_test)

    log(f"Dataset: {N} points  |  Train: {N_train}  Val: {len(y_val)}  "
        f"Test: {len(y_test)}")

    # Analytical solution
    theta_opt = linear.normal_equation(X_train_n, y_train)
    train_opt = linear.compute_cost(y_train, X_train_n @ theta_opt, N_train)
    val_opt = linear.compute_cost(y_val, X_val_n @ theta_opt, len(y_val))
    test_opt = linear.compute_cost(y_test, X_test_n @ theta_opt, len(y_test))
    r2 = linear.r_squared(y_train, X_train_n @ theta_opt)

    log(f"\nNormal equation:")
    log(f"  Train cost = {train_opt:.4f}   Val cost = {val_opt:.4f}   "
        f"Test cost = {test_opt:.4f}")
    log(f"  R² (train) = {r2:.4f}")
    log(f"  Weights: b={theta_opt[0]:.3f}  TV={theta_opt[1]:.3f}  "
        f"Radio={theta_opt[2]:.3f}  Newspaper={theta_opt[3]:.3f}")

    # Ridge lambda search
    log(f"\nRidge lambda search:")
    log(f"  {'Lambda':>8}  {'Train':>8}  {'Val':>8}  {'Test':>8}")
    train_costs, val_costs, test_costs = [], [], []
    for lam in lam_search:
        theta = linear.ridge_normal_equation(X_train_n, y_train, lam)
        tr = linear.compute_cost(y_train, X_train_n @ theta, N_train)
        va = linear.compute_cost(y_val, X_val_n @ theta, len(y_val))
        te = linear.compute_cost(y_test, X_test_n @ theta, len(y_test))
        train_costs.append(tr)
        val_costs.append(va)
        test_costs.append(te)
        log(f"  {lam:>8.3f}  {tr:>8.4f}  {va:>8.4f}  {te:>8.4f}")

    best_lam = lam_search[int(np.argmin(val_costs))]
    log(f"\n  Best lambda (by val cost) = {best_lam}")

    # Plots
    plotting.plot_feature_weights(
        theta_opt[1:],
        ["TV", "Radio", "Newspaper"],
        save_path=_results_path("04_feature_weights.png", results_dir),
    )
    plotting.plot_ridge_lambda_search(
        lam_search, train_costs, val_costs, test_costs, best_lam,
        save_path=_results_path("05_ridge_lambda.png", results_dir),
    )

    log("Plots saved to results/")
    return {"theta_opt": theta_opt, "r2": r2, "best_lambda": best_lam}


# ---------------------------------------------------------------------------
# 3. CAPM and factor models
# ---------------------------------------------------------------------------

def run_capm(
    tickers=("AAPL", "MSFT", "NVDA"),
    market="SPY",
    start="2018-01-01",
    end="2023-12-31",
    log=print,
    results_dir=RESULTS_DIR,
):
    """
    Download stock data, compute summary statistics, and run CAPM regression.

    Parameters
    ----------
    tickers : tuple of str
        Stocks to analyse. SPY (market proxy) is added automatically.
    market : str
        Market proxy ticker.
    start, end : str
        Date range.
    """
    log("\n" + "=" * 60)
    log("CAPM & FACTOR MODELS")
    log("=" * 60)

    all_tickers = [market] + list(tickers)
    log(f"\nDownloading data for {all_tickers} ({start} to {end})...")

    data_dict = {}
    for ticker in all_tickers:
        data_dict[ticker] = data.get_stock_data(ticker, start, end)
        log(f"  {ticker}: {len(data_dict[ticker])} trading days")

    aligned = data.align_returns(data_dict)
    log(f"\nAligned to {len(aligned)} common trading days")

    # Summary statistics
    summary = data.summary_statistics(aligned)
    log("\nSummary statistics:")
    log(f"  {'Ticker':>6}  {'Mean%':>8}  {'AnnVol%':>9}  "
        f"{'Skew':>7}  {'Kurt':>7}")
    for ticker, row in summary.iterrows():
        log(f"  {ticker:>6}  {row['mean_%']:>8.4f}  "
            f"{row['ann_vol_%']:>9.2f}  "
            f"{row['skewness']:>7.3f}  {row['kurtosis']:>7.3f}")

    # CAPM regression
    capm_results = capm.run_capm_multiple(aligned, market_col=market)
    log(f"\nCAPM results (market = {market}):")
    log(f"  {'Ticker':>6}  {'Alpha%':>8}  {'Beta':>6}  "
        f"{'t_alpha':>8}  {'p_alpha':>8}  {'R²':>6}")
    for ticker, res in capm_results.items():
        log(f"  {ticker:>6}  {res['alpha']*100:>8.4f}  "
            f"{res['beta']:>6.4f}  "
            f"{res['t_alpha']:>8.3f}  "
            f"{res['p_alpha']:>8.4f}  "
            f"{res['r_squared']:>6.4f}")

    # Plots
    plotting.plot_normalised_prices(
        {t: data_dict[t] for t in tickers},
        save_path=_results_path("06_normalised_prices.png", results_dir),
    )
    plotting.plot_return_distributions(
        aligned[list(tickers)],
        save_path=_results_path("07_return_distributions.png", results_dir),
    )
    plotting.plot_correlation_heatmap(
        data.correlation_matrix(aligned),
        save_path=_results_path("08_correlation.png", results_dir),
    )
    for ticker, res in capm_results.items():
        plotting.plot_capm_scatter(
            aligned[market].values,
            aligned[ticker].values,
            res["alpha"], res["beta"], ticker,
            save_path=_results_path(f"09_capm_{ticker}.png", results_dir),
        )
    plotting.plot_capm_betas(
        capm_results,
        save_path=_results_path("10_capm_betas.png", results_dir),
    )

    log("Plots saved to results/")
    return {"summary": summary, "capm": capm_results,
            "data_dict": data_dict, "aligned": aligned}


# ---------------------------------------------------------------------------
# 4. Direction classifier and backtest
# ---------------------------------------------------------------------------

def run_classifier(
    ticker="NVDA",
    market="SPY",
    start="2018-01-01",
    end="2023-12-31",
    alpha=0.01,
    epochs=100_000,
    threshold=0.005,
    log=print,
    results_dir=RESULTS_DIR,
):
    """
    Train a logistic regression classifier to predict next-day direction
    and backtest the resulting long/cash strategy.

    Parameters
    ----------
    ticker : str    Stock to classify.
    market : str    Market proxy.
    threshold : float
        Minimum return to label as "up" (default 0.5%). Filters out
        noise from small moves.
    """
    log("\n" + "=" * 60)
    log(f"DIRECTION CLASSIFIER & BACKTEST — {ticker}")
    log("=" * 60)

    log(f"\nDownloading {ticker} and {market} ({start} to {end})...")
    stock_df = data.get_stock_data(ticker, start, end)
    market_df = data.get_stock_data(market, start, end)

    features_df = capm.build_features(stock_df, market_df)
    log(f"Feature matrix: {features_df.shape[0]} rows x "
        f"{len(capm.FEATURE_COLS)} features")

    X_raw = features_df[capm.FEATURE_COLS].values
    y = (features_df["target"].values > threshold).astype(int)
    N = len(y)
    X_raw = np.column_stack([np.ones(N), X_raw])

    log(f"Class balance: {y.mean() * 100:.1f}% up days  "
        f"(threshold = {threshold * 100:.1f}%)")

    # Chronological split
    X_train, y_train, X_val, y_val, X_test, y_test = logistic.time_series_split(
        X_raw, y
    )
    X_train_n, X_val_n, X_test_n, mu, sigma = logistic.normalise(
        X_train, X_val, X_test
    )

    log(f"\nSplit: Train {len(y_train)}  Val {len(y_val)}  Test {len(y_test)}")
    log(f"Training logistic regression (alpha={alpha}, epochs={epochs})...")

    theta, costs = logistic.gradient_descent(X_train_n, y_train, alpha, epochs)

    # Evaluation
    log("\nClassification results:")
    for split_name, Xs, ys in [
        ("Training", X_train_n, y_train),
        ("Validation", X_val_n, y_val),
        ("Test", X_test_n, y_test),
    ]:
        m = logistic.evaluate(Xs, ys, theta)
        log(f"\n  {split_name}:")
        log(f"    Accuracy  = {m['accuracy']*100:.2f}%")
        log(f"    Precision = {m['precision']*100:.2f}%")
        log(f"    Recall    = {m['recall']*100:.2f}%")
        log(f"    Confusion: TN={m['tn']}  FP={m['fp']}  "
            f"FN={m['fn']}  TP={m['tp']}")

    # Backtest — use actual returns (not binary labels)
    returns_arr = features_df["target"].values
    N_total = len(returns_arr)
    train_end = int(N_total * 0.7)
    val_end = int(N_total * 0.85)

    log("\nBacktest results:")
    backtest_outputs = {}
    for label, Xs, ret_slice in [
        (f"{ticker} — Validation", X_val_n, returns_arr[train_end:val_end]),
        (f"{ticker} — Test", X_test_n, returns_arr[val_end:]),
    ]:
        p_hat = logistic.predict_proba(theta, Xs)
        bt = backtest.run_backtest(p_hat, ret_slice)
        backtest_outputs[label] = bt
        for line in backtest.backtest_summary(bt, label):
            log(line)

    # Plots
    plotting.plot_training_loss(
        costs,
        save_path=_results_path(f"11_training_loss_{ticker}.png", results_dir),
    )
    for label, bt in backtest_outputs.items():
        safe_label = label.replace(" ", "_").replace("—", "").strip("_")
        plotting.plot_equity_curves(
            bt, label=label,
            save_path=_results_path(f"12_equity_{safe_label}.png", results_dir),
        )
        plotting.plot_positions(
            bt["positions"], bt["benchmark_returns"],
            save_path=_results_path(f"13_positions_{safe_label}.png", results_dir),
        )

    log("\nPlots saved to results/")
    return {"theta": theta, "costs": costs, "backtest": backtest_outputs}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_all(
    tickers=("AAPL", "MSFT", "NVDA"),
    classifier_ticker="NVDA",
    start="2018-01-01",
    end="2023-12-31",
    log=print,
    results_dir=RESULTS_DIR,
):
    """Run the complete analysis pipeline end to end."""
    run_benchmark(log=log, results_dir=results_dir)
    run_multi_feature(log=log, results_dir=results_dir)
    run_capm(tickers=tickers, start=start, end=end,
             log=log, results_dir=results_dir)
    run_classifier(ticker=classifier_ticker, start=start, end=end,
                   log=log, results_dir=results_dir)
    log("\n" + "=" * 60)
    log("Pipeline complete. All results saved to results/")
    log("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all()
