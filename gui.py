"""
ml-quant-finance — gui.py
=========================
tkinter GUI for the ML Quant Finance analysis suite.

Layout
------
Left  : Configuration panel (dates, tickers, hyperparameters)
Right : Buttons to run each analysis section
Bottom: Scrollable text output showing results as they run
"""

import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import main as pipeline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FONT = ("Helvetica", 11)
FONT_BOLD = ("Helvetica", 11, "bold")
FONT_MONO = ("Courier", 10)


class MLQuantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Quant Finance")
        self.resizable(True, True)
        self.minsize(850, 600)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Title
        tk.Label(self, text="ML Quant Finance — Analysis Suite",
                 font=("Helvetica", 14, "bold")).pack(pady=(10, 2))
        tk.Label(self, text="Machine Learning Applied to Quantitative Finance",
                 font=("Helvetica", 10), fg="gray").pack(pady=(0, 8))

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10)

        # Middle row: config left, buttons right
        mid = tk.Frame(self)
        mid.pack(fill="both", expand=False, padx=10, pady=6)

        self._build_config(mid)
        self._build_buttons(mid)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10)

        # Output box
        out_frame = tk.LabelFrame(self, text="Output", font=FONT_BOLD)
        out_frame.pack(fill="both", expand=True, padx=10, pady=6)

        self._output = scrolledtext.ScrolledText(
            out_frame,
            font=FONT_MONO,
            state="disabled",
            wrap="word",
            height=18,
        )
        self._output.pack(fill="both", expand=True, padx=4, pady=4)

        # Status bar
        self._status_var = tk.StringVar(value="Status: Ready")
        tk.Label(self, textvariable=self._status_var,
                 font=("Helvetica", 10), fg="gray",
                 anchor="w").pack(fill="x", padx=10, pady=(0, 4))

    def _build_config(self, parent):
        frame = tk.LabelFrame(parent, text="Configuration", font=FONT_BOLD)
        frame.pack(side="left", fill="y", padx=(0, 6))

        def row(label, default, r):
            tk.Label(frame, text=label, font=FONT,
                     anchor="w", width=16).grid(
                row=r, column=0, sticky="w", padx=8, pady=4)
            var = tk.StringVar(value=default)
            tk.Entry(frame, textvariable=var, font=FONT_MONO,
                     width=14).grid(
                row=r, column=1, sticky="ew", padx=8, pady=4)
            return var

        tk.Label(frame, text="Date Range", font=FONT_BOLD).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 0))
        self._start_var = row("Start", "2018-01-01", 1)
        self._end_var = row("End", "2023-12-31", 2)

        ttk.Separator(frame, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=4)

        tk.Label(frame, text="Tickers to analyse", font=FONT_BOLD).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=8)

        self._ticker_vars = {}
        for i, ticker in enumerate(["AAPL", "MSFT", "NVDA", "TSLA"]):
            var = tk.BooleanVar(value=ticker in {"AAPL", "MSFT", "NVDA"})
            self._ticker_vars[ticker] = var
            tk.Checkbutton(frame, text=ticker, variable=var,
                           font=FONT).grid(
                row=5 + i // 2, column=i % 2, sticky="w", padx=8)

        ttk.Separator(frame, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=8, pady=4)

        tk.Label(frame, text="Hyperparameters", font=FONT_BOLD).grid(
            row=8, column=0, columnspan=2, sticky="w", padx=8)
        self._class_ticker_var = row("Classify ticker", "NVDA", 9)
        self._alpha_var = row("Learning rate α", "0.001", 10)
        self._epochs_var = row("Epochs", "5000", 11)
        self._threshold_var = row("Up threshold", "0.005", 12)

        ttk.Separator(frame, orient="horizontal").grid(
            row=13, column=0, columnspan=2, sticky="ew", padx=8, pady=4)

        tk.Button(frame, text="Clear Output", font=FONT,
                  command=self._clear_output).grid(
            row=14, column=0, columnspan=2, sticky="ew",
            padx=8, pady=6)

    def _build_buttons(self, parent):
        frame = tk.LabelFrame(parent, text="Run Analysis", font=FONT_BOLD)
        frame.pack(side="left", fill="both", expand=True)

        sections = [
            ("Benchmark Task (D=1 Linear Regression)",
             self._run_benchmark),
            ("Multi-Feature Regression (D=3, Advertising)",
             self._run_multi),
            ("CAPM & Factor Models",
             self._run_capm),
            ("Direction Classifier & Backtest",
             self._run_classifier),
            ("Run Full Pipeline",
             self._run_all),
        ]

        for label, cmd in sections:
            tk.Button(
                frame, text=label,
                font=FONT, width=45,
                pady=6,
                command=cmd,
            ).pack(fill="x", padx=16, pady=4)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _log(self, text):
        self._output.configure(state="normal")
        self._output.insert("end", text + "\n")
        self._output.see("end")
        self._output.configure(state="disabled")
        self.update_idletasks()  # refresh UI so text appears as it runs

    def _clear_output(self):
        self._output.configure(state="normal")
        self._output.delete("1.0", "end")
        self._output.configure(state="disabled")

    def _set_status(self, text):
        self._status_var.set(f"Status: {text}")
        self.update_idletasks()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _selected_tickers(self):
        return tuple(t for t, v in self._ticker_vars.items() if v.get())

    def _get_float(self, var, name):
        try:
            return float(var.get())
        except ValueError:
            messagebox.showerror("Invalid input", f"{name} must be a number.")
            return None

    def _get_int(self, var, name):
        try:
            return int(var.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 f"{name} must be a whole number.")
            return None

    # ------------------------------------------------------------------
    # Button callbacks — each just calls the pipeline function directly
    # ------------------------------------------------------------------

    def _run_benchmark(self):
        alpha = self._get_float(self._alpha_var, "Learning rate")
        epochs = self._get_int(self._epochs_var, "Epochs")
        if None in (alpha, epochs):
            return
        self._set_status("Running benchmark task...")
        try:
            pipeline.run_benchmark(
                alpha=alpha, epochs=epochs,
                log=self._log, results_dir=RESULTS_DIR,
            )
        except Exception as e:
            self._log(f"\nERROR: {e}")
        self._set_status("Done")

    def _run_multi(self):
        self._set_status("Running multi-feature regression...")
        try:
            pipeline.run_multi_feature(
                log=self._log, results_dir=RESULTS_DIR,
            )
        except Exception as e:
            self._log(f"\nERROR: {e}")
        self._set_status("Done")

    def _run_capm(self):
        tickers = self._selected_tickers()
        if not tickers:
            messagebox.showwarning("No tickers",
                                   "Select at least one ticker.")
            return
        self._set_status("Running CAPM...")
        try:
            pipeline.run_capm(
                tickers=tickers,
                start=self._start_var.get(),
                end=self._end_var.get(),
                log=self._log, results_dir=RESULTS_DIR,
            )
        except Exception as e:
            self._log(f"\nERROR: {e}")
        self._set_status("Done")

    def _run_classifier(self):
        ticker = self._class_ticker_var.get().strip().upper()
        alpha = self._get_float(self._alpha_var, "Learning rate")
        epochs = self._get_int(self._epochs_var, "Epochs")
        threshold = self._get_float(self._threshold_var, "Up threshold")
        if None in (alpha, epochs, threshold):
            return
        self._set_status(f"Training classifier on {ticker}...")
        try:
            pipeline.run_classifier(
                ticker=ticker,
                start=self._start_var.get(),
                end=self._end_var.get(),
                alpha=alpha, epochs=epochs, threshold=threshold,
                log=self._log, results_dir=RESULTS_DIR,
            )
        except Exception as e:
            self._log(f"\nERROR: {e}")
        self._set_status("Done")

    def _run_all(self):
        tickers = self._selected_tickers()
        if not tickers:
            messagebox.showwarning("No tickers",
                                   "Select at least one ticker.")
            return
        classifier_ticker = self._class_ticker_var.get().strip().upper()
        self._set_status("Running full pipeline...")
        try:
            pipeline.run_all(
                tickers=tickers,
                classifier_ticker=classifier_ticker,
                start=self._start_var.get(),
                end=self._end_var.get(),
                log=self._log, results_dir=RESULTS_DIR,
            )
        except Exception as e:
            self._log(f"\nERROR: {e}")
        self._set_status("Done")


if __name__ == "__main__":
    app = MLQuantApp()
    app.mainloop()