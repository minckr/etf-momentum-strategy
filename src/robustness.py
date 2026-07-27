"""
robustness.py
--------------
Robustness / sensitivity analysis for the momentum signal:

1. `run_grid`        -- Sharpe ratio (and other metrics) across a grid
                         of lookback windows x weighting schemes x
                         transaction-cost assumptions.
2. `in_out_sample`    -- splits history in half and compares Sharpe in
                         each half, as a simple stability check (a
                         "real" walk-forward/CV setup would roll this;
                         this is the lightweight version).
3. `predictive_regression` -- regresses next-month asset return on
                         this month's momentum signal, pooled across
                         assets and time, with Newey-West (HAC)
                         standard errors via statsmodels. This is the
                         standard cross-sectional check for whether
                         the signal has genuine forward-looking
                         predictive power, independent of any specific
                         portfolio construction choice.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, List

import numpy as np
import pandas as pd

from src import metrics as mx
from src import signal as sig
from src.backtest import BacktestConfig, run_backtest


def run_grid(
    prices: pd.DataFrame,
    lookbacks: Iterable[int] = (3, 6, 9, 12),
    weightings: Iterable[str] = ("equal", "rank", "inv_vol"),
    cost_bps_list: Iterable[float] = (0.0, 5.0, 10.0, 20.0),
    n_holdings: int = 6,
) -> pd.DataFrame:
    """Run the backtest across a full parameter grid and return one row
    of summary metrics per combination."""
    rows = []
    for lookback, weighting, cost_bps in product(lookbacks, weightings, cost_bps_list):
        cfg = BacktestConfig(
            lookback=lookback,
            n_holdings=n_holdings,
            weighting=weighting,
            cost_bps=cost_bps,
        )
        result = run_backtest(prices, cfg)
        row = {
            "lookback": lookback,
            "weighting": weighting,
            "cost_bps": cost_bps,
            "CAGR": mx.cagr(result.net_returns),
            "Ann. Vol": mx.annualized_vol(result.net_returns),
            "Sharpe": mx.sharpe_ratio(result.net_returns),
            "Max Drawdown": mx.max_drawdown(result.net_returns),
            "Ann. Turnover": result.turnover.mean() * 12,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def in_out_sample(
    prices: pd.DataFrame,
    config: BacktestConfig,
    split_date: str | None = None,
) -> pd.DataFrame:
    """Compare Sharpe (and other metrics) in the first vs. second half
    of the sample -- a quick check for whether performance is driven
    by one lucky regime rather than being reasonably stable over time.
    """
    result = run_backtest(prices, config)
    returns = result.net_returns.dropna()

    if split_date is None:
        split_idx = len(returns) // 2
        split_date = returns.index[split_idx]

    in_sample = returns[returns.index < split_date]
    out_sample = returns[returns.index >= split_date]

    return pd.DataFrame(
        {
            "In-Sample": mx.summary_table(in_sample, label="In-Sample"),
            "Out-of-Sample": mx.summary_table(out_sample, label="Out-of-Sample"),
        }
    )


def predictive_regression(
    prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    newey_west_lags: int = 3,
):
    """Pooled panel regression: next-month return ~ momentum signal.

    Requires `statsmodels` (optional dependency -- this function is a
    supplementary diagnostic, not on the critical path of the backtest
    itself). Returns the fitted statsmodels results object with HAC
    (Newey-West) standard errors, or raises ImportError with a clear
    message if statsmodels isn't installed.
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError(
            "predictive_regression requires statsmodels: pip install statsmodels"
        ) from e

    asset_returns = prices.pct_change()
    momentum_signal = sig.trailing_return(prices, lookback=lookback, skip=skip)
    fwd_returns = asset_returns.shift(-1)

    # Stack into a long panel: one row per (date, asset)
    panel = pd.DataFrame(
        {
            "signal": momentum_signal.stack(),
            "fwd_return": fwd_returns.stack(),
        }
    ).dropna()

    X = sm.add_constant(panel["signal"])
    y = panel["fwd_return"]
    model = sm.OLS(y, X)
    # Cluster-robust / HAC standard errors account for the fact that
    # observations within the same month are cross-sectionally
    # correlated (a common factor moves everything that month).
    results = model.fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags})
    return results
