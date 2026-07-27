"""
metrics.py
----------
Standard performance/risk metrics for a monthly return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def cagr(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total_growth = (1 + returns).prod()
    n_years = len(returns) / PERIODS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return total_growth ** (1 / n_years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.dropna().std() * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    std = returns.std()
    # Use a tolerance rather than `== 0`: a constant return series can
    # come back as a tiny nonzero float (e.g. 1e-18) due to floating-point
    # rounding, which would otherwise blow up into a meaningless ratio.
    if np.isnan(std) or std < 1e-9:
        return np.nan
    rf_monthly = rf_annual / PERIODS_PER_YEAR
    excess = returns - rf_monthly
    return (excess.mean() / excess.std()) * np.sqrt(PERIODS_PER_YEAR)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Sortino ratio using the standard downside-deviation formula:
    sqrt(mean(min(excess, 0)^2)) over the FULL sample (not just the
    negative-return subset). Computing it over the full sample avoids
    the degenerate case of a single downside observation, where a
    sample standard deviation (ddof=1) is undefined.
    """
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    rf_monthly = rf_annual / PERIODS_PER_YEAR
    excess = returns - rf_monthly
    downside = excess.clip(upper=0)
    downside_deviation = np.sqrt((downside ** 2).mean())
    if np.isnan(downside_deviation) or downside_deviation < 1e-9:
        return np.nan
    return (excess.mean() / downside_deviation) * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if not mdd or mdd == 0 or np.isnan(mdd):
        return np.nan
    return cagr(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return (returns > 0).mean()


def drawdown_series(returns: pd.Series) -> pd.Series:
    returns = returns.dropna()
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def summary_table(returns: pd.Series, turnover: pd.Series | None = None, label: str = "Strategy") -> pd.Series:
    """One-row summary of headline metrics, convenient for comparing
    multiple strategies/benchmarks side by side (pd.concat along axis=1)."""
    out = {
        "CAGR": cagr(returns),
        "Ann. Vol": annualized_vol(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(returns),
        "Calmar": calmar_ratio(returns),
        "Hit Rate": hit_rate(returns),
    }
    if turnover is not None:
        out["Ann. Turnover"] = turnover.mean() * PERIODS_PER_YEAR
    return pd.Series(out, name=label)
