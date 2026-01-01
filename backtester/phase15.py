from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .lrs import compute_trend_state

def realized_vol_annual(price: pd.Series, lookback: int = 20) -> pd.Series:
    """Annualized realized vol from daily returns."""
    r = price.pct_change()
    v = r.rolling(lookback).std(ddof=0) * np.sqrt(252)
    return v

def backtest_lrs_vol_target(
    price_risk: pd.Series,
    price_def: pd.Series,
    signal_price: pd.Series,
    ma: pd.Series,
    initial_capital: float,
    contrib_amount: float,
    contrib_dates: pd.DatetimeIndex,
    include_contribs: bool,
    out_to_defensive: bool,
    buffer_pct: float,
    confirm_days: int,
    # vol targeting
    vol_lookback: int = 20,
    target_vol: float = 0.20,      # 20% annual
    max_exposure: float = 1.0,     # cap leverage/exposure
    min_exposure: float = 0.0,
) -> pd.DataFrame:
    """
    LRS gate + volatility targeting:
      - IN only if trend is IN
      - exposure scales to target_vol / realized_vol
      - OUT goes defensive (or cash if out_to_defensive=False)
    Daily rebalance.
    Returns daily df: Equity, Contributions, Exposure, InMarket
    """
    idx = price_risk.index.intersection(price_def.index).intersection(signal_price.index).intersection(ma.index)
    risk = price_risk.reindex(idx)
    defs = price_def.reindex(idx)
    sig = signal_price.reindex(idx)
    ma = ma.reindex(idx)

    contrib_dates = pd.DatetimeIndex(contrib_dates).intersection(idx)
    contrib_set = set(contrib_dates)

    in_mkt = compute_trend_state(sig, ma, buffer_pct=buffer_pct, confirm_days=confirm_days)
    rv = realized_vol_annual(sig, lookback=vol_lookback).reindex(idx)

    cash = float(initial_capital)
    risk_sh = 0.0
    def_sh = 0.0
    contrib_total = float(initial_capital)

    eq, contribs, expo = [], [], []

    for dt in idx:
        if include_contribs and dt in contrib_set and contrib_amount > 0:
            cash += float(contrib_amount)
            contrib_total += float(contrib_amount)

        rprice = float(risk.at[dt])
        dprice = float(defs.at[dt])

        risk_val = risk_sh * rprice
        def_val = def_sh * dprice
        total = cash + risk_val + def_val

        state = bool(in_mkt.at[dt])
        # compute exposure when IN
        if state and not np.isnan(rv.at[dt]) and rv.at[dt] > 0:
            e = float(target_vol / rv.at[dt])
            e = float(np.clip(e, min_exposure, max_exposure))
        elif state:
            e = 0.0
        else:
            e = 0.0

        # targets
        target_risk = e * total
        target_def = (total - target_risk) if out_to_defensive else 0.0
        target_cash = 0.0 if out_to_defensive else (total - target_risk)

        # SELL down to targets
        if risk_val > target_risk:
            sell_val = risk_val - target_risk
            sell_sh = sell_val / rprice
            risk_sh -= sell_sh
            cash += sell_val

        def_val = def_sh * dprice
        if out_to_defensive and def_val > target_def:
            sell_val = def_val - target_def
            sell_sh = sell_val / dprice
            def_sh -= sell_sh
            cash += sell_val

        # BUY up to targets
        risk_val = risk_sh * rprice
        def_val = def_sh * dprice
        total = cash + risk_val + def_val

        if risk_val < target_risk:
            buy_val = min(target_risk - risk_val, cash)
            risk_sh += buy_val / rprice
            cash -= buy_val

        if out_to_defensive:
            def_val = def_sh * dprice
            total = cash + risk_sh * rprice + def_val
            # invest remaining cash into defensive to match target_def
            def_target = total - (risk_sh * rprice)
            buy_val = min(def_target - def_val, cash)
            if buy_val > 0:
                def_sh += buy_val / dprice
                cash -= buy_val

        # finalize
        equity = cash + risk_sh * rprice + def_sh * dprice
        eq.append(equity)
        contribs.append(contrib_total)
        expo.append(e)

    return pd.DataFrame(
        {"Equity": eq, "Contributions": contribs, "Exposure": expo, "InMarket": in_mkt.astype(int)},
        index=idx
    )

def backtest_dual_momentum_rotation(
    prices: dict[str, pd.Series],
    defensive_ticker: str,
    contrib_dates: pd.DatetimeIndex,
    initial_capital: float,
    contrib_amount: float,
    include_contribs: bool,
    rebalance_dates: pd.DatetimeIndex,
    lookback_days: int = 126,   # ~6 months
    use_absolute_momentum: bool = True,  # require top momentum > 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dual momentum rotation:
      - On rebalance_dates, compute trailing return over lookback_days for each 'risk' candidate
      - Choose best risk candidate
      - If absolute momentum required and best <= 0 => go defensive
      - Always hold ONE asset at a time
    prices includes defensive_ticker too.
    """
    tickers = list(prices.keys())
    idx = None
    for t in tickers:
        idx = prices[t].index if idx is None else idx.intersection(prices[t].index)
    idx = pd.DatetimeIndex(idx)

    px = {t: prices[t].reindex(idx) for t in tickers}
    rebalance_dates = pd.DatetimeIndex(rebalance_dates).intersection(idx)
    contrib_dates = pd.DatetimeIndex(contrib_dates).intersection(idx)
    contrib_set = set(contrib_dates)
    reb_set = set(rebalance_dates)

    cash = float(initial_capital)
    sh = {t: 0.0 for t in tickers}
    held = defensive_ticker
    contrib_total = float(initial_capital)

    daily_eq, daily_contrib = [], []
    ledger = []

    def portfolio_value(dt):
        total = cash
        for t in tickers:
            total += sh[t] * float(px[t].at[dt])
        return total

    for dt in idx:
        if include_contribs and dt in contrib_set and contrib_amount > 0:
            cash += float(contrib_amount)
            contrib_total += float(contrib_amount)

        if dt in reb_set:
            # compute momentums
            moms = {}
            for t in tickers:
                if t == defensive_ticker:
                    continue
                s = px[t].loc[:dt]
                if len(s) <= lookback_days:
                    continue
                ret = float(s.iloc[-1] / s.iloc[-1 - lookback_days] - 1.0)
                moms[t] = ret

            # select
            choice = defensive_ticker
            if moms:
                best = max(moms.items(), key=lambda kv: kv[1])
                if (not use_absolute_momentum) or (best[1] > 0):
                    choice = best[0]

            # rebalance to 100% chosen
            total = portfolio_value(dt)
            # sell everything
            for t in tickers:
                if sh[t] != 0:
                    cash += sh[t] * float(px[t].at[dt])
                    sh[t] = 0.0
            # buy chosen
            p = float(px[choice].at[dt])
            if p > 0:
                sh[choice] = cash / p
                cash = 0.0

            held = choice
            ledger.append({
                "Date": dt,
                "Held": held,
                "TotalAssets": total,
                "Contributions": contrib_total,
            })

        daily_eq.append(portfolio_value(dt))
        daily_contrib.append(contrib_total)

    daily = pd.DataFrame({"Equity": daily_eq, "Contributions": daily_contrib}, index=idx)
    return daily, pd.DataFrame(ledger)

def walk_forward_optimize_9sig_lrs(
    run_fn,
    param_grid: list[dict],
    dates: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    score_dd_penalty: float = 0.5,
) -> pd.DataFrame:
    """
    Generic walk-forward optimizer.
    - run_fn(params, start_dt, end_dt) => daily df with Equity
    - score = CAGR - penalty*abs(MaxDD)
    Returns rows per test segment with chosen params and realized metrics.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) < 2:
        return pd.DataFrame()

    start_all = dates[0]
    end_all = dates[-1]
    cur = start_all

    rows = []

    while True:
        train_end = cur + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end_all:
            break

        # snap to nearest available trading date
        train_end = dates[dates.get_indexer([train_end], method="nearest")[0]]
        test_end = dates[dates.get_indexer([test_end], method="nearest")[0]]

        best_score = -1e9
        best_params = None

        # train search
        for params in param_grid:
            daily = run_fn(params, cur, train_end)
            m = compute_metrics(daily["Equity"])
            score = m["cagr"] - score_dd_penalty * abs(m["max_dd"])
            if score > best_score:
                best_score = score
                best_params = params

        # test
        test_daily = run_fn(best_params, train_end, test_end)
        tm = compute_metrics(test_daily["Equity"])

        rows.append({
            "train_start": cur.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "chosen_params": best_params,
            "test_cagr": tm["cagr"],
            "test_maxdd": tm["max_dd"],
            "test_final": tm["final"],
        })

        cur = train_end

    return pd.DataFrame(rows)
