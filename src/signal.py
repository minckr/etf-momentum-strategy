"""
signal.py
---------
Cross-sectional momentum signal construction.

Signal definition: trailing total return over a lookback window,
measured with a one-month skip (the "12-1" convention from
Jegadeesh & Titman) to avoid the well-documented short-term reversal
effect contaminating the momentum signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return(
    prices: pd.DataFrame, lookback: int, skip: int = 1
) -> pd.DataFrame:
    """Trailing total return signal, monthly frequency.

    For each date t, computes cumulative return from t-(lookback+skip)
    to t-skip. `skip=1` excludes the most recent month (the standard
    12-1 momentum construction). `skip=0` uses the full window through
    the most recent observation.

    Returns a DataFrame aligned to `prices.index`, with NaN wherever
    there isn't enough history.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if skip < 0:
        raise ValueError("skip must be non-negative")

    shifted = prices.shift(skip)
    signal = shifted / shifted.shift(lookback) - 1.0
    return signal


def volatility(
    returns: pd.DataFrame, window: int = 12, min_periods: int = 6
) -> pd.DataFrame:
    """Trailing realized volatility (annualized), used for vol-scaling."""
    return returns.rolling(window, min_periods=min_periods).std() * np.sqrt(12)


def cross_sectional_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """Rank assets cross-sectionally each period (1 = weakest, N = strongest).

    NaNs are preserved (an asset with no signal that period gets no rank).
    """
    return signal.rank(axis=1, method="average", na_option="keep")


def zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each period -- useful for combining signals
    or for rank-weighted portfolio construction with less lumpiness than
    raw ranks."""
    mu = signal.mean(axis=1)
    sigma = signal.std(axis=1)
    return signal.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)
