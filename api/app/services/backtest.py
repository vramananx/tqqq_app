from __future__ import annotations
import pandas as pd
import numpy as np

from .analytics import compute_metrics, underwater_vs_contrib

def _simple_schedule(idx: pd.DatetimeIndex, freq: str, day: int) -> pd.DatetimeIndex:
    day = max(1, min(int(day), 28))
    if freq == "monthly":
        months = pd.PeriodIndex(idx, freq="M").unique()
        dates = []
        for m in months:
            dt = pd.Timestamp(m.year, m.month, day)
            # shift to next trading day
            cand = idx[idx.get_indexer([dt], method="bfill")[0]] if dt <= idx[-1] else None
            if cand is not None:
                dates.append(cand)
        return pd.DatetimeIndex(sorted(set(dates)))
    else:
        months = pd.PeriodIndex(idx, freq="M").unique()
        dates = []
        for m in months:
            if m.month in (1, 4, 7, 10):
                dt = pd.Timestamp(m.year, m.month, day)
                cand = idx[idx.get_indexer([dt], method="bfill")[0]] if dt <= idx[-1] else None
                if cand is not None:
                    dates.append(cand)
        return pd.DatetimeIndex(sorted(set(dates)))

def run_stub_strategies(price: pd.Series, initial: float, contrib_amount: float, contrib_freq: str, contrib_day: int):
    idx = price.index
    schedule = _simple_schedule(idx, contrib_freq, contrib_day)

    # Buy & Hold
    shares = initial / float(price.iloc[0])
    equity_bh = shares * price.astype(float)
    contrib_bh = pd.Series(initial, index=idx).ffill()

    # DCA (monthly/quarterly deposits then buy at close)
    cash = float(initial)
    sh = 0.0
    contrib_total = float(initial)
    equity = []
    contrib = []
    markers = []
    for dt in idx:
        if dt in schedule:
            cash += contrib_amount
            contrib_total += contrib_amount
            buy = contrib_amount / float(price.loc[dt])
            sh += buy
            cash -= buy * float(price.loc[dt])
            markers.append(dict(t=str(dt.date()), action="BUY", price=float(price.loc[dt]), note=f"+{contrib_amount}"))
        equity.append(cash + sh * float(price.loc[dt]))
        contrib.append(contrib_total)

    equity_dca = pd.Series(equity, index=idx)
    contrib_dca = pd.Series(contrib, index=idx)

    return dict(
        buy_hold=dict(equity=equity_bh, contrib=contrib_bh, markers=[dict(t=str(idx[0].date()), action="BUY", price=float(price.iloc[0]), note="init")]),
        dca=dict(equity=equity_dca, contrib=contrib_dca, markers=markers),
    )

def to_points(s: pd.Series):
    return [dict(t=str(d.date()), v=float(v)) for d, v in s.items()]

def to_markers(markers: list[dict], equity: pd.Series):
    # add equity value to marker
    out = []
    for m in markers:
        dt = pd.Timestamp(m["t"])
        dt = equity.index[equity.index.get_indexer([dt], method="nearest")[0]]
        out.append(dict(
            t=str(dt.date()),
            action=m.get("action",""),
            price=m.get("price"),
            equity=float(equity.loc[dt]),
            note=m.get("note")
        ))
    return out
