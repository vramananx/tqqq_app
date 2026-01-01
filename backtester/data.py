from __future__ import annotations

import os
import pandas as pd
import yfinance as yf


def fetch_adj_close(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = ".cache_prices",
) -> pd.Series:
    os.makedirs(cache_dir, exist_ok=True)
    safe = ticker.replace("/", "_").replace(":", "_")
    cache_path = os.path.join(cache_dir, f"{safe}_{start}_{end}.parquet")

    df = None
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if df is None or len(df) == 0:
            raise ValueError(f"No data returned for {ticker} in range {start}..{end}")

        df.to_parquet(cache_path)

    # Normalize columns (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # With auto_adjust=True, 'Close' is adjusted close
    if "Close" in df.columns:
        s = df["Close"].copy()
    elif "Adj Close" in df.columns:
        s = df["Adj Close"].copy()
    else:
        # fallback to last column if needed
        s = df.iloc[:, -1].copy()

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()
    s.name = ticker
    return s
