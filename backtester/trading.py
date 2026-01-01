from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class RunConfig:
    initial_cash: float
    position_size: float  # 0..1 fraction of portfolio to allocate when "IN"
    start: str
    end: str


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        # accept 1-col dataframe or take Close if present
        if "Close" in x.columns:
            return x["Close"]
        if "Adj Close" in x.columns:
            return x["Adj Close"]
        if x.shape[1] == 1:
            return x.iloc[:, 0]
    raise TypeError(f"Expected pandas Series, got {type(x)}")



def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def summarize_equity(equity: pd.Series, trades: pd.DataFrame, rf_annual: float = 0.0) -> Dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 3:
        return {
            "Final Value": np.nan,
            "Total Return %": np.nan,
            "CAGR %": np.nan,
            "Volatility %": np.nan,
            "Sharpe": np.nan,
            "Max Drawdown %": np.nan,
            "Trades": float(len(trades) if trades is not None else 0),
        }

    rets = equity.pct_change().dropna()
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1e-9)

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0

    vol = rets.std() * np.sqrt(252)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1.0
    sharpe = (rets.mean() - rf_daily) / (rets.std() + 1e-12) * np.sqrt(252)

    mdd = _drawdown(equity).min()

    return {
        "Final Value": float(equity.iloc[-1]),
        "Total Return %": float(total_return * 100),
        "CAGR %": float(cagr * 100),
        "Volatility %": float(vol * 100),
        "Sharpe": float(sharpe),
        "Max Drawdown %": float(mdd * 100),
        "Trades": float(len(trades) if trades is not None else 0),
    }


def worst_drawdown_window(equity: pd.Series) -> Dict[str, object]:
    equity = equity.dropna()
    if len(equity) < 3:
        return {"Peak Date": None, "Trough Date": None, "Recovery Date": None, "Max Drawdown %": np.nan}

    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough_date = dd.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()
    max_dd = float(dd.loc[trough_date] * 100)

    # first date after trough where equity recovers above old peak
    old_peak_val = equity.loc[peak_date]
    after = equity.loc[trough_date:]
    rec = after[after >= old_peak_val]
    recovery_date = rec.index[0] if len(rec) else None

    return {
        "Peak Date": peak_date,
        "Trough Date": trough_date,
        "Recovery Date": recovery_date,
        "Max Drawdown %": max_dd,
    }


def _ledger_rows_to_df(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["date", "action", "price", "shares_delta", "shares", "cash", "assets", "note"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _trades_from_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger is None or len(ledger) == 0:
        return pd.DataFrame(columns=["date", "action", "price", "shares_delta", "note"])
    t = ledger[ledger["action"].isin(["BUY", "SELL", "SWITCH_IN", "SWITCH_OUT", "REBAL"])].copy()
    return t[["date", "action", "price", "shares_delta", "note"]].reset_index(drop=True)


def backtest_all_in_out(
    price: pd.Series,
    signal_in: pd.Series,
    cfg: RunConfig,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Generic: when signal_in True => invest position_size of portfolio into the asset.
             when signal_in False => go 100% cash (no interest).
    Trades happen at close of that date.
    """
    price = _to_series(price).dropna()
    signal_in = signal_in.reindex(price.index).fillna(False).astype(bool)

    cash = float(cfg.initial_cash)
    shares = 0.0

    ledger_rows = []
    equity = []

    last_state: Optional[bool] = None

    for dt, px in price.items():
        state = bool(signal_in.loc[dt])
        # switch?
        if last_state is None:
            last_state = state

        if state != last_state:
            # liquidate to cash
            if shares != 0.0:
                cash += shares * float(px)
                ledger_rows.append({
                    "date": dt, "action": "SWITCH_OUT", "price": float(px),
                    "shares_delta": -shares, "shares": 0.0, "cash": cash,
                    "assets": cash, "note": "Exit to cash"
                })
                shares = 0.0

            # enter position
            if state:
                target = cash * float(cfg.position_size)
                buy_sh = target / float(px) if float(px) > 0 else 0.0
                cash -= buy_sh * float(px)
                shares += buy_sh
                ledger_rows.append({
                    "date": dt, "action": "SWITCH_IN", "price": float(px),
                    "shares_delta": buy_sh, "shares": shares, "cash": cash,
                    "assets": cash + shares * float(px),
                    "note": f"Enter {cfg.position_size:.2f} allocation"
                })

            last_state = state

        assets = cash + shares * float(px)
        equity.append((dt, assets))

    equity_s = pd.Series([v for _, v in equity], index=pd.to_datetime([d for d, _ in equity]), name="equity")
    ledger = _ledger_rows_to_df(ledger_rows)
    trades = _trades_from_ledger(ledger)
    return equity_s, ledger, trades


def sma_signal(price: pd.Series, sma_len: int) -> pd.Series:
    px = price.dropna()
    sma = px.rolling(int(sma_len)).mean()
    return (px > sma).reindex(px.index).fillna(False)


def ema_signal(price: pd.Series, ema_len: int) -> pd.Series:
    px = price.dropna()
    ema = px.ewm(span=int(ema_len), adjust=False).mean()
    return (px > ema).reindex(px.index).fillna(False)


def backtest_dca(
    price: pd.Series,
    start_cash: float,
    contrib_amount: float,
    contrib_freq: str,     # "Monthly" or "Quarterly"
    contrib_day: int,      # day-of-month to invest (1..28 recommended)
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, float]:
    """
    DCA into the asset, never sells. Keeps leftover cash.
    Returns: equity, ledger, trades, total_contributed
    """
    px = price.dropna()
    cash = float(start_cash)
    shares = 0.0
    total_contrib = float(start_cash)

    freq = contrib_freq.lower()
    if freq not in ["monthly", "quarterly"]:
        raise ValueError("contrib_freq must be Monthly or Quarterly")

    ledger_rows = []
    equity = []

    # Determine invest dates based on month boundaries of the available index
    idx = pd.to_datetime(px.index)
    months = pd.PeriodIndex(idx, freq="M").unique()

    invest_dates = []
    for m in months:
        dt = pd.Timestamp(m.start_time.year, m.start_time.month, 1)
        # choose day-of-month
        day = int(contrib_day)
        day = max(1, min(day, 28))
        dt = pd.Timestamp(dt.year, dt.month, day)

        # if chosen day not in price index (weekend/holiday), move to next available trading day
        candidates = px.loc[dt: dt + pd.Timedelta(days=10)]
        if len(candidates) == 0:
            continue
        invest_dt = candidates.index[0]

        if freq == "monthly":
            invest_dates.append(invest_dt)
        else:
            # quarterly: months 1,4,7,10
            if m.month in [1, 4, 7, 10]:
                invest_dates.append(invest_dt)

    invest_dates = sorted(set(pd.to_datetime(invest_dates)))

    for dt, p in px.items():
        p = float(p)

        if dt in invest_dates:
            cash += float(contrib_amount)
            total_contrib += float(contrib_amount)

            buy_sh = (float(contrib_amount) / p) if p > 0 else 0.0
            shares += buy_sh
            cash -= buy_sh * p  # uses exactly contrib_amount (small float rounding ok)

            ledger_rows.append({
                "date": dt, "action": "BUY", "price": p,
                "shares_delta": buy_sh, "shares": shares, "cash": cash,
                "assets": cash + shares * p,
                "note": f"DCA {contrib_freq} +{contrib_amount:,.0f}"
            })

        assets = cash + shares * p
        equity.append((dt, assets))

    equity_s = pd.Series([v for _, v in equity], index=pd.to_datetime([d for d, _ in equity]), name="equity")
    ledger = _ledger_rows_to_df(ledger_rows)
    trades = _trades_from_ledger(ledger)
    return equity_s, ledger, trades, total_contrib


def backtest_buy_hold(price: pd.Series, start_cash: float) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame, float]:
    px = price.dropna()
    first_dt = px.index[0]
    first_px = float(px.iloc[0])

    shares = float(start_cash) / first_px if first_px > 0 else 0.0
    cash = float(start_cash) - shares * first_px
    total_contrib = float(start_cash)

    ledger = _ledger_rows_to_df([{
        "date": first_dt, "action": "BUY", "price": first_px,
        "shares_delta": shares, "shares": shares, "cash": cash,
        "assets": cash + shares * first_px,
        "note": "Initial buy & hold"
    }])

    equity = cash + shares * px.astype(float)
    equity.name = "equity"
    trades = _trades_from_ledger(ledger)
    return equity, ledger, trades, total_contrib
