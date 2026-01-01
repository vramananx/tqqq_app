from __future__ import annotations
import math
import numpy as np
import pandas as pd

def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0

def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)

def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * math.sqrt(252))

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = returns - rf_daily
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * math.sqrt(252))

def worst_drawdown_details(equity: pd.Series) -> dict:
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])

    pre = equity.loc[:trough]
    peak_val = float(pre.max())
    peak_dt = pre.idxmax()
    trough_val = float(equity.loc[trough])

    post = equity.loc[trough:]
    rec = post[post >= peak_val]
    rec_dt = rec.index[0] if len(rec) else None

    return {
        "max_drawdown": max_dd,
        "peak_date": peak_dt,
        "trough_date": trough,
        "recovery_date": rec_dt,
        "peak_value": peak_val,
        "trough_value": trough_val,
    }

def compute_metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    return {
        "final": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity),
        "sharpe": sharpe_ratio(rets, 0.0),
        "vol": annualized_vol(rets),
        "max_dd": float(drawdown_series(equity).min()),
    }
