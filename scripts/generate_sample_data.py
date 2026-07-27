"""
generate_sample_data.py
------------------------
Generates a synthetic monthly price history for the ETF universe and
writes it to data/etf_prices.csv, so the rest of the pipeline can be
run and demonstrated fully offline.

*** THIS IS SYNTHETIC DATA, CLEARLY LABELED AS SUCH. ***

It exists for two reasons:
1. To let this repo be cloned and run end-to-end with zero API keys
   and no network dependency (useful for CI, for reviewers, and for
   reproducing the committed results/ outputs exactly).
2. To develop/debug the pipeline without hammering Yahoo Finance.

To backtest on REAL market history instead:
    python -c "from src.data_loader import load_prices; load_prices(refresh=True)"
(requires `pip install yfinance` and an internet connection -- this
overwrites data/etf_prices.csv with live Yahoo Finance data and every
downstream script/notebook will use it automatically.)

The generator uses a simple factor model so the synthetic universe has
realistic-ish momentum-relevant properties: cross-sectional dispersion
in trend/drift, asset-class correlation clusters, fat-tailed shocks,
and regime shifts (calm vs. stressed volatility periods) -- rather
than pure i.i.d. noise, which would make momentum signals meaningless
by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DATA_DIR, DEFAULT_UNIVERSE

SEED = 42

# Rough asset-class grouping -> (annual drift, annual vol, factor loading)
# Loosely calibrated to long-run real-world behavior of each ETF's asset
# class; not fitted to any specific historical sample.
ASSET_PARAMS = {
    "SPY": ("equity_us", 0.09, 0.16),
    "QQQ": ("equity_us", 0.12, 0.21),
    "IWM": ("equity_us", 0.08, 0.20),
    "EFA": ("equity_intl", 0.05, 0.17),
    "EEM": ("equity_intl", 0.05, 0.22),
    "XLK": ("equity_us", 0.13, 0.22),
    "XLF": ("equity_us", 0.08, 0.22),
    "XLE": ("equity_us", 0.04, 0.27),
    "XLV": ("equity_us", 0.09, 0.15),
    "XLY": ("equity_us", 0.10, 0.19),
    "XLP": ("equity_us", 0.07, 0.12),
    "XLI": ("equity_us", 0.08, 0.18),
    "XLU": ("equity_us", 0.07, 0.14),
    "TLT": ("rates", 0.03, 0.13),
    "IEF": ("rates", 0.025, 0.07),
    "LQD": ("credit", 0.035, 0.08),
    "HYG": ("credit", 0.05, 0.09),
    "SHY": ("rates", 0.015, 0.02),
    "GLD": ("commodity", 0.04, 0.15),
    "SLV": ("commodity", 0.03, 0.28),
    "DBC": ("commodity", 0.01, 0.18),
    "VNQ": ("real_estate", 0.07, 0.20),
}

FACTOR_CORR = {
    "equity_us": {"equity_us": 1.0, "equity_intl": 0.8, "rates": -0.2, "credit": 0.5, "commodity": 0.1, "real_estate": 0.6},
    "equity_intl": {"equity_intl": 1.0, "rates": -0.15, "credit": 0.45, "commodity": 0.15, "real_estate": 0.4},
    "rates": {"rates": 1.0, "credit": 0.3, "commodity": -0.1, "real_estate": 0.2},
    "credit": {"credit": 1.0, "commodity": 0.1, "real_estate": 0.3},
    "commodity": {"commodity": 1.0, "real_estate": 0.05},
    "real_estate": {"real_estate": 1.0},
}


def _build_factor_cov():
    factors = list(FACTOR_CORR.keys())
    n = len(factors)
    corr = np.eye(n)
    for i, fi in enumerate(factors):
        for j, fj in enumerate(factors):
            if fj in FACTOR_CORR[fi]:
                corr[i, j] = FACTOR_CORR[fi][fj]
            elif fi in FACTOR_CORR[fj]:
                corr[i, j] = FACTOR_CORR[fj][fi]
    return factors, corr


def generate(
    tickers=None,
    start="2008-01-01",
    end="2024-12-31",
    seed: int = SEED,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_UNIVERSE
    tickers = [t for t in tickers if t in ASSET_PARAMS]
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start, end=end, freq="ME")
    n_months = len(dates)

    factors, factor_corr = _build_factor_cov()
    factor_vol = 0.06  # monthly factor vol
    factor_cov = factor_corr * (factor_vol ** 2)

    # Regime switching: alternate calm/stressed vol multipliers so
    # momentum has to survive multiple volatility regimes, not just one.
    regime_len = 18  # months
    n_regimes = int(np.ceil(n_months / regime_len))
    regime_mult = rng.choice([0.7, 1.0, 1.0, 1.8], size=n_regimes)  # mostly calm, occasional stress
    vol_mult = np.repeat(regime_mult, regime_len)[:n_months]

    factor_returns = rng.multivariate_normal(
        mean=np.zeros(len(factors)), cov=factor_cov, size=n_months
    )
    factor_returns = factor_returns * vol_mult[:, None]
    factor_df = pd.DataFrame(factor_returns, index=dates, columns=factors)

    # Slow-moving idiosyncratic drift shifts per asset, so cross-sectional
    # momentum has genuine signal (assets trend for multi-month stretches,
    # then mean-revert into a new trend) rather than being pure noise.
    returns = {}
    for tkr in tickers:
        cls, drift, vol = ASSET_PARAMS[tkr]
        beta = rng.uniform(0.7, 1.1)
        idio_vol = vol * rng.uniform(0.5, 0.7)

        # AR(1) latent drift process -> creates trending/momentum behavior
        drift_state = drift / 12
        drift_path = np.zeros(n_months)
        phi = 0.93
        shock_scale = 0.006
        for i in range(n_months):
            drift_state = phi * drift_state + (1 - phi) * (drift / 12) + rng.normal(0, shock_scale)
            drift_path[i] = drift_state

        idio_shock = rng.normal(0, idio_vol / np.sqrt(12), size=n_months) * vol_mult
        # fat tails: occasional larger idiosyncratic jumps
        jump_mask = rng.random(n_months) < 0.03
        idio_shock[jump_mask] += rng.normal(0, idio_vol, size=jump_mask.sum())

        r = drift_path + beta * factor_df[cls].values + idio_shock
        returns[tkr] = r

    ret_df = pd.DataFrame(returns, index=dates)
    prices = 100 * (1 + ret_df).cumprod()
    prices.index.name = "date"
    return prices


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices = generate()
    out_path = DATA_DIR / "etf_prices.csv"
    prices.to_csv(out_path)
    print(f"Wrote synthetic price history for {prices.shape[1]} ETFs, "
          f"{prices.shape[0]} months -> {out_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
