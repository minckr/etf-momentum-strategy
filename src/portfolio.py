"""
portfolio.py
------------
Turns a cross-sectional signal into monthly rebalanced portfolio
weights, under a few different weighting rules, and computes turnover.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

WeightingScheme = Literal["equal", "rank", "inv_vol"]


def select_top_n(signal: pd.DataFrame, n_holdings: int) -> pd.DataFrame:
    """Boolean mask of the top-N ranked assets each period (True = held).

    Periods with fewer than n_holdings non-NaN signals select whatever
    is available rather than raising.
    """
    def _top_n_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if valid.empty:
            return pd.Series(False, index=row.index)
        k = min(n_holdings, len(valid))
        top = valid.nlargest(k).index
        return row.index.isin(top)

    mask = signal.apply(_top_n_row, axis=1, result_type="expand")
    mask.columns = signal.columns
    return mask.astype(bool)


def build_weights(
    signal: pd.DataFrame,
    n_holdings: int = 6,
    scheme: WeightingScheme = "equal",
    trailing_vol: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct monthly portfolio weights from a momentum signal.

    Parameters
    ----------
    signal : cross-sectional momentum signal (higher = stronger momentum)
    n_holdings : number of ETFs held each month (top-N by signal)
    scheme :
        "equal"   -- equal weight across the N holdings
        "rank"    -- weight proportional to cross-sectional rank among
                     holdings (strongest signal gets the largest weight)
        "inv_vol" -- equal-weight selection, but sized inversely to
                     trailing realized volatility (risk parity within
                     the selected sleeve); requires `trailing_vol`
    """
    held = select_top_n(signal, n_holdings)
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    if scheme == "equal":
        counts = held.sum(axis=1).replace(0, np.nan)
        weights = held.div(counts, axis=0).fillna(0.0)

    elif scheme == "rank":
        masked_signal = signal.where(held)
        ranks = masked_signal.rank(axis=1, method="average", na_option="keep")
        rank_sum = ranks.sum(axis=1).replace(0, np.nan)
        weights = ranks.div(rank_sum, axis=0).fillna(0.0)

    elif scheme == "inv_vol":
        if trailing_vol is None:
            raise ValueError("trailing_vol is required for the inv_vol scheme")
        inv_vol = 1.0 / trailing_vol.where(held)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        vol_sum = inv_vol.sum(axis=1)
        weights = inv_vol.div(vol_sum, axis=0).fillna(0.0)

    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")

    return weights


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way monthly turnover: sum of |weight changes| / 2 per period.

    Dividing by 2 gives the conventional "one-way" turnover measure
    (fraction of the book traded), rather than double-counting both
    the buy and the sell leg of each rebalance.
    """
    delta = weights.diff().abs().sum(axis=1)
    return (delta / 2.0).fillna(0.0)
