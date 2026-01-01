from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache_prices")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = f"{ticker}_{start}_{end}".replace(":", "-").replace("/", "-")
    return os.path.join(CACHE_DIR, safe + ".parquet")

def fetch_prices(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    path = _cache_path(ticker, start, end)
    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data returned for {ticker} {start}..{end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    if "Close" not in df.columns:
        raise ValueError("Expected Close column from yfinance")

    df = df[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].dropna()
    df = df.sort_index()

    if use_cache:
        df.to_parquet(path)

    return df

def align_series(*series):
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return [s.reindex(idx) for s in series]
