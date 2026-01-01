from __future__ import annotations
import pandas as pd

def underwater_vs_contrib(equity: pd.Series, contrib: pd.Series) -> dict:
    """
    equity: daily equity series
    contrib: daily total invested series (same index)
    Returns:
      - worst underwater P&L (equity - contrib)
      - date of worst underwater
      - breakeven recovery date + recovery days
      - longest continuous underwater spell
    """
    equity = equity.dropna()
    contrib = contrib.reindex(equity.index).ffill().fillna(0.0)

    pnl = equity - contrib
    trough_dt = pnl.idxmin()
    trough_val = float(pnl.loc[trough_dt])

    # recovery to breakeven
    after = pnl.loc[trough_dt:]
    rec = after[after >= 0]
    rec_dt = rec.index[0] if len(rec) else None
    rec_days = int((rec_dt - trough_dt).days) if rec_dt is not None else None

    # longest underwater streak
    under = pnl < 0
    # find streaks
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
        longest = max(streaks, key=lambda ab: (ab[1]-ab[0]).days)
        longest_days = int((longest[1] - longest[0]).days)
    else:
        longest_days = 0

    return {
        "worst_underwater_dollars": trough_val,
        "worst_underwater_date": str(trough_dt.date()),
        "breakeven_recovery_date": str(rec_dt.date()) if rec_dt is not None else None,
        "breakeven_recovery_days": rec_days,
        "longest_underwater_days": longest_days,
        "longest_underwater_start": str(longest[0].date()) if longest else None,
        "longest_underwater_end": str(longest[1].date()) if longest else None,
    }
