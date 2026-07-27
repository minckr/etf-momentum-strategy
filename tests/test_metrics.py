import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_vol,
    cagr,
    calmar_ratio,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


def test_cagr_matches_hand_calc():
    # 1% every month for 12 months -> total growth 1.01^12
    returns = pd.Series([0.01] * 12)
    expected = 1.01 ** 12 - 1
    assert cagr(returns) == pytest.approx(expected, rel=1e-6)


def test_cagr_empty_series_is_nan():
    assert np.isnan(cagr(pd.Series(dtype=float)))


def test_annualized_vol_scales_by_sqrt12():
    returns = pd.Series([0.0, 0.02, -0.02, 0.02, -0.02] * 4)
    vol = annualized_vol(returns)
    assert vol == pytest.approx(returns.std() * np.sqrt(12))


def test_sharpe_zero_vol_is_nan():
    returns = pd.Series([0.01] * 12)
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_positive_for_positive_mean_return():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.01, 0.02, 100))
    assert sharpe_ratio(returns) > 0


def test_sortino_ignores_upside_vol():
    # All positive returns except one small negative one: Sortino should
    # be much larger than Sharpe since only downside vol counts.
    returns = pd.Series([0.05, 0.06, 0.04, 0.07, -0.01, 0.05, 0.06])
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    assert sortino > sharpe


def test_max_drawdown_known_path():
    # wealth path: 1 -> 1.1 -> 0.99 -> 1.05  (peak 1.1, trough 0.99)
    returns = pd.Series([0.10, -0.10, 0.06060606])
    mdd = max_drawdown(returns)
    assert mdd == pytest.approx(-0.10, abs=1e-3)


def test_max_drawdown_is_non_positive():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.005, 0.03, 60))
    assert max_drawdown(returns) <= 0


def test_calmar_ratio_relates_cagr_to_drawdown():
    returns = pd.Series([0.02] * 24)  # no drawdown at all -> should be nan (mdd=0)
    assert np.isnan(calmar_ratio(returns))


def test_hit_rate_counts_positive_periods():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
    assert hit_rate(returns) == pytest.approx(3 / 5)
