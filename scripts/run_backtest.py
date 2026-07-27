"""
run_backtest.py
----------------
End-to-end driver: loads prices, runs the baseline momentum strategy
against two benchmarks, runs the robustness grid + in/out-of-sample
check, and writes all figures/tables to results/.

    python scripts/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import metrics as mx
from src import robustness as rb
from src.backtest import BacktestConfig, equal_weight_benchmark, run_backtest, single_asset_benchmark
from src.data_loader import load_prices

FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

BASELINE_CONFIG = BacktestConfig(
    lookback=12, skip=1, n_holdings=6, weighting="equal", cost_bps=10.0
)


def plot_equity_curves(series_dict: dict[str, pd.Series], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, returns in series_dict.items():
        wealth = (1 + returns.dropna()).cumprod()
        ax.plot(wealth.index, wealth.values, label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drawdowns(series_dict: dict[str, pd.Series], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, returns in series_dict.items():
        dd = mx.drawdown_series(returns)
        ax.plot(dd.index, dd.values * 100, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rolling_sharpe(returns: pd.Series, path: Path, window: int = 24):
    fig, ax = plt.subplots(figsize=(9, 4))
    roll = returns.rolling(window).apply(
        lambda x: mx.sharpe_ratio(pd.Series(x)), raw=False
    )
    ax.plot(roll.index, roll.values, color="darkorange", linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio - Momentum Strategy")
    ax.set_ylabel("Sharpe")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sharpe_heatmap(grid_df: pd.DataFrame, cost_bps: float, path: Path):
    subset = grid_df[grid_df["cost_bps"] == cost_bps]
    pivot = subset.pivot(index="weighting", columns="lookback", values="Sharpe")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Lookback (months)")
    ax.set_title(f"Sharpe Ratio Grid - {cost_bps:.0f} bps Transaction Cost")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_turnover_vs_cost(grid_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for weighting, g in grid_df.groupby("weighting"):
        summary = g.groupby("cost_bps")["Sharpe"].mean()
        ax.plot(summary.index, summary.values, marker="o", label=weighting)
    ax.set_xlabel("Transaction Cost Assumption (bps)")
    ax.set_ylabel("Mean Sharpe Across Lookbacks")
    ax.set_title("Sharpe Sensitivity to Transaction Cost Assumptions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    print("Loading price data...")
    prices = load_prices(refresh=False)
    print(f"  {prices.shape[1]} ETFs, {prices.shape[0]} months "
          f"({prices.index.min().date()} to {prices.index.max().date()})")

    # ---- Baseline backtest vs. two benchmarks ----------------------------
    print("Running baseline backtest...")
    result = run_backtest(prices, BASELINE_CONFIG)
    ew_bench = equal_weight_benchmark(prices)
    spy_bench = single_asset_benchmark(prices, "SPY")

    common_start = result.net_returns.dropna().index.min()
    strategy_r = result.net_returns[result.net_returns.index >= common_start]
    ew_r = ew_bench[ew_bench.index >= common_start]
    spy_r = spy_bench[spy_bench.index >= common_start]

    series = {"Momentum (net)": strategy_r, "Equal-Weight Universe": ew_r, "SPY Buy & Hold": spy_r}

    summary = pd.concat(
        [
            mx.summary_table(strategy_r, result.turnover, label="Momentum (net of costs)"),
            mx.summary_table(result.gross_returns[result.gross_returns.index >= common_start],
                              result.turnover, label="Momentum (gross)"),
            mx.summary_table(ew_r, label="Equal-Weight Universe"),
            mx.summary_table(spy_r, label="SPY Buy & Hold"),
        ],
        axis=1,
    )
    summary.to_csv(TBL_DIR / "baseline_summary.csv")
    print("\n=== Baseline Summary ===")
    print(summary.round(3).to_string())

    plot_equity_curves(series, FIG_DIR / "equity_curves.png",
                        "Momentum Strategy vs. Benchmarks - Growth of $1")
    plot_drawdowns(series, FIG_DIR / "drawdowns.png",
                    "Drawdowns - Momentum Strategy vs. Benchmarks")
    plot_rolling_sharpe(strategy_r, FIG_DIR / "rolling_sharpe.png")

    # ---- Robustness grid ---------------------------------------------------
    print("\nRunning robustness grid (lookback x weighting x cost)...")
    grid = rb.run_grid(
        prices,
        lookbacks=(3, 6, 9, 12),
        weightings=("equal", "rank", "inv_vol"),
        cost_bps_list=(0.0, 5.0, 10.0, 20.0),
        n_holdings=6,
    )
    grid.to_csv(TBL_DIR / "robustness_grid.csv", index=False)
    print(f"  {len(grid)} parameter combinations evaluated -> results/tables/robustness_grid.csv")

    plot_sharpe_heatmap(grid, cost_bps=10.0, path=FIG_DIR / "sharpe_heatmap_10bps.png")
    plot_turnover_vs_cost(grid, FIG_DIR / "sharpe_vs_cost.png")

    # ---- In-sample / out-of-sample stability -------------------------------
    print("\nIn-sample vs. out-of-sample split (baseline config)...")
    ios = rb.in_out_sample(prices, BASELINE_CONFIG)
    ios.to_csv(TBL_DIR / "in_out_sample.csv")
    print(ios.round(3).to_string())

    # ---- Predictive regression (optional, requires statsmodels) -----------
    print("\nPredictive regression (pooled, Newey-West SEs)...")
    try:
        reg_results = rb.predictive_regression(prices, lookback=12, skip=1)
        coef = reg_results.params["signal"]
        tstat = reg_results.tvalues["signal"]
        pval = reg_results.pvalues["signal"]
        with open(TBL_DIR / "predictive_regression_summary.txt", "w") as f:
            f.write(str(reg_results.summary()))
        print(f"  signal coefficient={coef:.4f}, t-stat={tstat:.2f}, p-value={pval:.4f}")
        print("  -> results/tables/predictive_regression_summary.txt")
    except ImportError as e:
        print(f"  Skipped: {e}")

    print("\nDone. Figures in results/figures/, tables in results/tables/.")


if __name__ == "__main__":
    main()
