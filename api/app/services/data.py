from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = ".cache_prices"

def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = (ticker or "").replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.parquet")

def fetch_close(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker, start, end)

    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise ValueError(f"No data for {ticker} {start}..{end}")
        df.to_parquet(path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" in df.columns:
        s = df["Close"].copy()
    elif "Adj Close" in df.columns:
        s = df["Adj Close"].copy()
    else:
        s = df.iloc[:, -1].copy()

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()

def align(*series: pd.Series) -> list[pd.Series]:
    idx = None
    for s in series:
        idx = s.index if idx is None else idx.intersection(s.index)
    return [s.reindex(idx) for s in series]
