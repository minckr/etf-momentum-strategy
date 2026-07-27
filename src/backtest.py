"""
backtest.py
-----------
Ties signal construction, portfolio weighting, and transaction costs
together into a monthly-rebalanced backtest, and builds comparison
benchmarks.

Timing convention (important, and a common source of lookahead bugs):
  - Signal at the end of month t is computed from price data available
    at t (with the 1-month skip already baked into signal.trailing_return).
  - Weights decided at t are HELD OVER month t+1, i.e. weights.shift(1)
    is what actually earns returns.mtm at t+1.
  - Transaction costs are charged on turnover measured at the moment
    the trade happens (the rebalance at t), applied against the return
    realized in t+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import signal as sig
from src import portfolio as pf


@dataclass
class BacktestConfig:
    lookback: int = 12
    skip: int = 1
    n_holdings: int = 6
    weighting: str = "equal"          # "equal" | "rank" | "inv_vol"
    vol_window: int = 12
    cost_bps: float = 10.0            # one-way transaction cost, in bps of turnover


@dataclass
class BacktestResult:
    config: BacktestConfig
    weights: pd.DataFrame
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    holdings_count: pd.Series = field(repr=False, default=None)


def run_backtest(prices: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Run a single momentum backtest given a price panel and config."""
    asset_returns = prices.pct_change()

    momentum_signal = sig.trailing_return(prices, lookback=config.lookback, skip=config.skip)

    trailing_vol = None
    if config.weighting == "inv_vol":
        trailing_vol = sig.volatility(asset_returns, window=config.vol_window)

    weights = pf.build_weights(
        momentum_signal,
        n_holdings=config.n_holdings,
        scheme=config.weighting,
        trailing_vol=trailing_vol,
    )

    turnover = pf.compute_turnover(weights)

    # Weights decided at t are applied to returns realized in t+1.
    applied_weights = weights.shift(1)
    gross_returns = (applied_weights * asset_returns).sum(axis=1)

    # Cost charged in the period the trade occurs (turnover at t hits
    # the return earned in t+1, matching applied_weights.shift above).
    cost_per_period = (turnover.shift(1) * config.cost_bps / 10_000.0).fillna(0.0)
    net_returns = gross_returns - cost_per_period

    holdings_count = (weights > 0).sum(axis=1)

    valid_from = momentum_signal.dropna(how="all").index.min()
    mask = gross_returns.index >= valid_from

    return BacktestResult(
        config=config,
        weights=weights[mask],
        gross_returns=gross_returns[mask],
        net_returns=net_returns[mask],
        turnover=turnover[mask],
        holdings_count=holdings_count[mask],
    )


def equal_weight_benchmark(prices: pd.DataFrame, rebalance: bool = True) -> pd.Series:
    """Equal-weight, monthly-rebalanced benchmark across the full universe
    (buy-and-hold-the-universe, not a momentum tilt)."""
    asset_returns = prices.pct_change()
    n_assets = asset_returns.notna().sum(axis=1).replace(0, np.nan)
    if rebalance:
        weights = asset_returns.notna().div(n_assets, axis=0)
        bench_returns = (weights.shift(1) * asset_returns).sum(axis=1)
    else:
        # Buy-and-hold from inception, weights drift with prices.
        norm = prices.div(prices.bfill().iloc[0])
        weights = norm.div(norm.sum(axis=1), axis=0)
        bench_returns = (weights.shift(1) * asset_returns).sum(axis=1)
    return bench_returns


def single_asset_benchmark(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Simple buy-and-hold return series for a single reference ticker (e.g. SPY)."""
    if ticker not in prices.columns:
        raise KeyError(f"{ticker} not found in price panel")
    return prices[ticker].pct_change()
