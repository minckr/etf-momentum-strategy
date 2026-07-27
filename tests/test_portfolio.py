import numpy as np
import pandas as pd
import pytest

from src.portfolio import build_weights, compute_turnover, select_top_n


@pytest.fixture
def toy_signal():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame(
        {
            "A": [5, 4, np.nan],
            "B": [4, 3, 3],
            "C": [3, np.nan, 2],
            "D": [2, 1, 1],
            "E": [1, 2, 0],
        },
        index=dates,
    )


def test_select_top_n_basic(toy_signal):
    held = select_top_n(toy_signal, n_holdings=2)
    assert held.loc[toy_signal.index[0]].sum() == 2
    assert held.loc[toy_signal.index[0], ["A", "B"]].all()
    assert not held.loc[toy_signal.index[0], ["C", "D", "E"]].any()


def test_select_top_n_handles_fewer_valid_than_n(toy_signal):
    # row 2 (index 1) has only 3 non-NaN values (B, D, E... wait A is NaN)
    held = select_top_n(toy_signal, n_holdings=10)
    row = toy_signal.index[1]
    n_valid = toy_signal.loc[row].notna().sum()
    assert held.loc[row].sum() == n_valid


def test_equal_weight_sums_to_one(toy_signal):
    w = build_weights(toy_signal, n_holdings=2, scheme="equal")
    sums = w.sum(axis=1)
    assert np.allclose(sums, 1.0)
    # each held name gets exactly 1/2
    assert w.loc[toy_signal.index[0], "A"] == pytest.approx(0.5)


def test_rank_weight_sums_to_one_and_favors_higher_signal(toy_signal):
    w = build_weights(toy_signal, n_holdings=3, scheme="rank")
    sums = w.sum(axis=1)
    assert np.allclose(sums, 1.0)
    row = toy_signal.index[0]
    # A has the highest signal among the top-3 (A, B, C) -> highest weight
    assert w.loc[row, "A"] > w.loc[row, "B"] > w.loc[row, "C"]


def test_inv_vol_requires_trailing_vol(toy_signal):
    with pytest.raises(ValueError):
        build_weights(toy_signal, n_holdings=2, scheme="inv_vol")


def test_inv_vol_weight_favors_lower_vol(toy_signal):
    vol = pd.DataFrame(
        1.0,
        index=toy_signal.index,
        columns=toy_signal.columns,
    )
    vol.loc[toy_signal.index[0], "A"] = 0.5  # A is lower vol than B
    w = build_weights(toy_signal, n_holdings=2, scheme="inv_vol", trailing_vol=vol)
    row = toy_signal.index[0]
    assert w.loc[row, "A"] > w.loc[row, "B"]


def test_unknown_scheme_raises(toy_signal):
    with pytest.raises(ValueError):
        build_weights(toy_signal, n_holdings=2, scheme="bogus")


def test_turnover_zero_when_weights_unchanged():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates)
    turnover = compute_turnover(w)
    assert (turnover.iloc[1:] == 0).all()


def test_turnover_full_flip_is_one():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=dates)
    turnover = compute_turnover(w)
    assert turnover.iloc[1] == pytest.approx(1.0)
