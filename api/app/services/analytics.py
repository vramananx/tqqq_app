from __future__ import annotations
import numpy as np
import pandas as pd

def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return dict(final=np.nan, cagr=np.nan, sharpe=np.nan, vol=np.nan, max_dd=np.nan)

    rets = equity.pct_change().dropna()
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1e-9)

    final = float(equity.iloc[-1])
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)

    vol = float(rets.std(ddof=0) * np.sqrt(252))
    sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(252))

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    return dict(final=final, cagr=cagr, sharpe=sharpe, vol=vol, max_dd=max_dd)

def underwater_vs_contrib(equity: pd.Series, contrib: pd.Series) -> dict:
    equity = equity.dropna()
    contrib = contrib.reindex(equity.index).ffill().fillna(0.0)

    pnl = equity - contrib
    trough_dt = pnl.idxmin()
    trough_val = float(pnl.loc[trough_dt])

    after = pnl.loc[trough_dt:]
    rec = after[after >= 0]
    rec_dt = rec.index[0] if len(rec) else None
    rec_days = int((rec_dt - trough_dt).days) if rec_dt is not None else None

    under = pnl < 0
    streaks = []
    start = None
    for dt, is_under in under.items():
        if is_under and start is None:
            start = dt
        if (not is_under) and start is not None:
            streaks.append((start, dt))
            start = None
    if start is not None:
        streaks.append((start, under.index[-1]))

    longest = None
    if streaks:
        longest = max(streaks, key=lambda ab: (ab[1] - ab[0]).days)
        longest_days = int((longest[1] - longest[0]).days)
    else:
        longest_days = 0

    return dict(
        worst_underwater_dollars=trough_val,
        worst_underwater_date=str(trough_dt.date()),
        breakeven_recovery_date=str(rec_dt.date()) if rec_dt is not None else None,
        breakeven_recovery_days=rec_days,
        longest_underwater_days=longest_days,
        longest_underwater_start=str(longest[0].date()) if longest else None,
        longest_underwater_end=str(longest[1].date()) if longest else None,
    )
