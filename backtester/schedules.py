from __future__ import annotations
import numpy as np
import pandas as pd

def invest_dates_monthly(trading_days: pd.DatetimeIndex, day_of_month: int) -> pd.DatetimeIndex:
    day_of_month = int(np.clip(day_of_month, 1, 28))
    td = pd.DatetimeIndex(trading_days)

    out = []
    ym = list(zip(td.year, td.month))
    i = 0
    while i < len(td):
        y, m = ym[i]
        j = i
        while j < len(td) and ym[j] == (y, m):
            j += 1

        month_days = td[i:j]
        target = pd.Timestamp(year=y, month=m, day=day_of_month)
        cand = month_days[month_days >= target]
        out.append(cand[0] if len(cand) else month_days[-1])
        i = j

    return pd.DatetimeIndex(out)

def invest_dates_quarterly(trading_days: pd.DatetimeIndex, day_of_month: int) -> pd.DatetimeIndex:
    monthly = invest_dates_monthly(trading_days, day_of_month)
    return pd.DatetimeIndex(monthly[monthly.month.isin([1, 4, 7, 10])])

def skip_first_period(dates: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    if len(dates) == 0:
        return dates
    d0 = dates[0]
    if mode == "monthly":
        mask = ~((dates.year == d0.year) & (dates.month == d0.month))
        return pd.DatetimeIndex(dates[mask])
    if mode == "quarterly":
        q0 = d0.to_period("Q")
        mask = ~(dates.to_period("Q") == q0)
        return pd.DatetimeIndex(dates[mask])
    return dates
