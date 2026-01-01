from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_adj_close(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = ".cache_prices",
) -> pd.Series:
    os.makedirs(cache_dir, exist_ok=True)
    safe = ticker.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe}_{start}_{end}.parquet")

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise ValueError(f"No data returned for {ticker} in range {start}..{end}")
        df.to_parquet(cache_path)

    # auto_adjust=True => "Close" is adjusted
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    s.name = ticker
    return s
