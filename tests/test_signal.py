import numpy as np
import pandas as pd
import pytest

from src.signal import cross_sectional_rank, trailing_return, volatility, zscore


@pytest.fixture
def toy_prices():
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    # A: steady 2%/mo grower. B: flat. C: steady 1%/mo decliner.
    a = 100 * (1.02 ** np.arange(15))
    b = np.full(15, 100.0)
    c = 100 * (0.99 ** np.arange(15))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


def test_trailing_return_shape_and_nan_prefix(toy_prices):
    tr = trailing_return(toy_prices, lookback=6, skip=1)
    assert tr.shape == toy_prices.shape
    # first (lookback + skip) rows must be NaN -- not enough history
    assert tr.iloc[:7].isna().all().all()
    assert tr.iloc[7:].notna().all().all()


def test_trailing_return_ordering_matches_trend(toy_prices):
    tr = trailing_return(toy_prices, lookback=6, skip=1)
    last = tr.iloc[-1]
    # Grower should rank above flat, flat above decliner.
    assert last["A"] > last["B"] > last["C"]


def test_trailing_return_skip_excludes_most_recent_month():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    # big move only in the very last month
    prices = pd.DataFrame({"X": [100, 101, 102, 150]}, index=dates)
    with_skip = trailing_return(prices, lookback=2, skip=1)
    no_skip = trailing_return(prices, lookback=2, skip=0)
    # with skip=1, the last-row signal should NOT reflect the 150 jump
    assert with_skip["X"].iloc[-1] == pytest.approx(102 / 100 - 1)
    assert no_skip["X"].iloc[-1] == pytest.approx(150 / 101 - 1)


def test_trailing_return_invalid_args(toy_prices):
    with pytest.raises(ValueError):
        trailing_return(toy_prices, lookback=0)
    with pytest.raises(ValueError):
        trailing_return(toy_prices, lookback=6, skip=-1)


def test_cross_sectional_rank_preserves_nan(toy_prices):
    tr = trailing_return(toy_prices, lookback=6, skip=1)
    ranks = cross_sectional_rank(tr)
    assert ranks.isna().equals(tr.isna())
    last_ranks = ranks.iloc[-1]
    assert last_ranks["A"] == 3 and last_ranks["C"] == 1


def test_zscore_mean_near_zero(toy_prices):
    tr = trailing_return(toy_prices, lookback=6, skip=1).dropna(how="all")
    z = zscore(tr)
    row_means = z.mean(axis=1)
    assert (row_means.abs() < 1e-8).all()


def test_volatility_nan_before_min_periods(toy_prices):
    rets = toy_prices.pct_change()
    vol = volatility(rets, window=6, min_periods=6)
    assert vol.iloc[:6].isna().all().all()
    assert vol.iloc[6:].notna().all().all()
