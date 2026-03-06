"""Price data loading with local caching.

All downstream code consumes only `load_returns()`. Caching to parquet
means the pipeline is reproducible offline after the first run and reruns
are fast during iteration.
"""
import os
import pandas as pd
import yfinance as yf

from src.universe import TICKERS

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PRICES_PATH = os.path.join(_DATA_DIR, "adj_close.parquet")


def download_prices(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for the full universe and cache to parquet."""
    raw = yf.download(
        TICKERS, start=start, end=end, auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )
    close = pd.DataFrame({t: raw[t]["Close"] for t in TICKERS if t in raw.columns.get_level_values(0)})
    close = close.dropna(axis=1, thresh=int(0.95 * len(close)))  # drop tickers with too much missing history
    close = close.ffill(limit=5).dropna(axis=0, how="any")
    os.makedirs(_DATA_DIR, exist_ok=True)
    close.to_parquet(_PRICES_PATH)
    return close


def load_prices(refresh: bool = False) -> pd.DataFrame:
    if refresh or not os.path.exists(_PRICES_PATH):
        return download_prices()
    return pd.read_parquet(_PRICES_PATH)


def load_returns(refresh: bool = False) -> pd.DataFrame:
    """Daily simple returns, assets as columns, dates as index."""
    prices = load_prices(refresh=refresh)
    return prices.pct_change().dropna(how="all")


if __name__ == "__main__":
    prices = download_prices()
    print(f"Downloaded {prices.shape[1]} tickers, {prices.shape[0]} trading days "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
