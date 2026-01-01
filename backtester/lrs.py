from __future__ import annotations
import pandas as pd

def compute_trend_state(price: pd.Series, ma: pd.Series, buffer_pct: float = 0.0, confirm_days: int = 1) -> pd.Series:
    ma = ma.reindex(price.index)
    enter = price > (ma * (1.0 + buffer_pct))
    exit_ = price < (ma * (1.0 - buffer_pct))

    state = False
    enter_count = 0
    exit_count = 0
    out = []

    for dt in price.index:
        if pd.isna(ma.loc[dt]):
            out.append(False)
            continue

        if bool(enter.loc[dt]): enter_count += 1
        else: enter_count = 0

        if bool(exit_.loc[dt]): exit_count += 1
        else: exit_count = 0

        if (not state) and enter_count >= confirm_days:
            state = True
            exit_count = 0
        elif state and exit_count >= confirm_days:
            state = False
            enter_count = 0

        out.append(state)

    return pd.Series(out, index=price.index, name="InMarket")
