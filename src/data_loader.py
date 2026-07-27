"""
data_loader.py
---------------
Fetches monthly adjusted-close price data for the ETF universe.

Primary source: Yahoo Finance via `yfinance`.
Fallback: a local cache (data/etf_prices.csv). If the cache exists and
`refresh=False`, it is used instead of hitting the network -- this keeps
repeated backtests fast and lets the project run in offline / CI
environments (a synthetic sample dataset is checked in for exactly this
reason -- see scripts/generate_sample_data.py).

Usage
-----
    from src.data_loader import load_prices
    prices = load_prices(tickers, start="2007-01-01", end="2024-12-31")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_PATH = DATA_DIR / "etf_prices.csv"

DEFAULT_UNIVERSE = [
    # Broad equity
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    # Sector equity
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU",
    # Fixed income
    "TLT", "IEF", "LQD", "HYG", "SHY",
    # Commodities / real assets
    "GLD", "SLV", "DBC", "VNQ",
]


def fetch_from_yfinance(
    tickers: List[str], start: str, end: str, interval: str = "1mo"
) -> pd.DataFrame:
    """Pull adjusted close prices from Yahoo Finance.

    Requires `pip install yfinance` and an internet connection. Raises
    ImportError if yfinance isn't installed, so callers can fall back
    to the offline cache cleanly.
    """
    import yfinance as yf  # local import: optional dependency

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.concat(
            {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)},
            axis=1,
        )
    else:
        # Single-ticker download shape
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index.name = "date"
    return prices.sort_index()


def load_prices(
    tickers: Optional[List[str]] = None,
    start: str = "2008-01-01",
    end: str = "2024-12-31",
    refresh: bool = False,
    use_cache_fallback: bool = True,
) -> pd.DataFrame:
    """Load monthly price history for `tickers`.

    Tries yfinance first (if `refresh=True` or no cache exists). Falls
    back to the checked-in CSV cache if the network/yfinance is
    unavailable, so the rest of the pipeline never breaks.
    """
    tickers = tickers or DEFAULT_UNIVERSE
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if refresh or not CACHE_PATH.exists():
        try:
            prices = fetch_from_yfinance(tickers, start, end)
            prices.to_csv(CACHE_PATH)
            print(f"[data_loader] Fetched live data for {len(tickers)} tickers, cached to {CACHE_PATH}")
            return prices
        except Exception as e:  # ImportError, network error, rate limit, etc.
            if not use_cache_fallback or not CACHE_PATH.exists():
                raise RuntimeError(
                    "Could not fetch live data via yfinance and no offline cache "
                    "is available. Install yfinance and check your network, or "
                    "run scripts/generate_sample_data.py to create a cache."
                ) from e
            print(f"[data_loader] yfinance unavailable ({e.__class__.__name__}); "
                  f"using cached data at {CACHE_PATH}")

    prices = pd.read_csv(CACHE_PATH, index_col="date", parse_dates=True)
    available = [t for t in tickers if t in prices.columns]
    missing = sorted(set(tickers) - set(available))
    if missing:
        print(f"[data_loader] Warning: {missing} not found in cache, dropping from universe")
    return prices[available]


if __name__ == "__main__":
    px = load_prices()
    print(px.tail())
