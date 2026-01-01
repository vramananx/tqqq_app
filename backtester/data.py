from __future__ import annotations

import os
import pandas as pd
import yfinance as yf

CACHE_DIR = ".cache_prices"


def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = (ticker or "").replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.parquet")


def _normalize_download_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError("No data returned from yfinance")
    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_prices(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with at least: Close
    Uses auto_adjust=True so Close is adjusted.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker, start, end)

    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
        df = _normalize_download_df(df)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df = _normalize_download_df(df)
        if use_cache:
            df.to_parquet(path)

    # Ensure Close exists
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            # fallback to last column
            df["Close"] = df.iloc[:, -1]

    return df[["Close"]].dropna()


def fetch_adj_close(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    """
    Backwards compatible: returns adjusted close as a Series.
    """
    df = fetch_prices(ticker, start, end, use_cache=use_cache)
    s = df["Close"].copy()
    s.name = (ticker or "").upper().strip()
    return s


def align_series(*series: pd.Series):
    """
    Align multiple Series to their common index intersection.
    """
    if len(series) == 0:
        return []
    idx = series[0].index
    for s in series[1:]:
        idx = idx.intersection(s.index)
    return [s.reindex(idx) for s in series]
