from __future__ import annotations
import pandas as pd
from .lrs import compute_trend_state

def _apply_cost(cash: float, notional: float, cost_bps: float) -> float:
    """
    Deduct transaction cost from cash for a trade with 'notional' dollars.
    cost_bps = 10 => 0.10% of notional.
    """
    if cost_bps is None or cost_bps <= 0 or notional <= 0:
        return cash
    fee = notional * (cost_bps / 10_000.0)
    return cash - fee

# -----------------------------
# Buy & Hold (single asset)
# -----------------------------
def backtest_buy_hold(
    price: pd.Series,
    initial_capital: float,
    contrib_amount: float,
    contrib_dates: pd.DatetimeIndex,
    include_contribs: bool,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = price.index
    contrib_dates = pd.DatetimeIndex(contrib_dates).intersection(idx)
    contrib_set = set(contrib_dates)

    cash = float(initial_capital)
    shares = 0.0
    cb = 0.0
    contrib_total = float(initial_capital)

    daily_equity, daily_contrib = [], []
    ledger = []

    def buy_all(dt, action):
        nonlocal cash, shares, cb
        if cash <= 0:
            return 0.0
        p = float(price.at[dt])
        amt = cash
        cash = _apply_cost(cash, amt, cost_bps)
        amt2 = max(0.0, amt if cash >= 0 else amt + cash)  # protect if fee exceeded
        shares += amt2 / p
        cash -= amt2
        cb += amt2
        return amt2

    def record(dt, dollar_added, action):
        p = float(price.at[dt])
        stock_val = shares * p
        total_assets = cash + stock_val
        unreal = stock_val - cb
        pl = total_assets - contrib_total
        ledger.append({
            "Date": dt, "Price": p, "Action": action, "DollarAdded": dollar_added,
            "RiskShares": shares, "RiskValue": stock_val,
            "DefShares": 0.0, "DefValue": 0.0,
            "Cash": cash, "TotalAssets": total_assets,
            "Contributions": contrib_total, "PL": pl, "Unrealized": unreal,
            "InMarket": 1
        })

    # day0 buy
    buy_all(idx[0], "BUY_INIT")
    record(idx[0], 0.0, "BUY_INIT")

    for dt in idx:
        if include_contribs and (dt in contrib_set) and contrib_amount > 0:
            cash += float(contrib_amount)
            contrib_total += float(contrib_amount)
            buy_all(dt, "BUY_CONTRIB")
            record(dt, float(contrib_amount), "BUY_CONTRIB")

        p = float(price.at[dt])
        daily_equity.append(cash + shares * p)
        daily_contrib.append(contrib_total)

    record(idx[-1], 0.0, "END")
    return pd.DataFrame({"Equity": daily_equity, "Contributions": daily_contrib}, index=idx), pd.DataFrame(ledger)

# -----------------------------
# 9-SIG (A/B) with defensive parking
# -----------------------------
def backtest_9sig(
    price_risk: pd.Series,
    price_def: pd.Series,
    initial_capital: float,
    contrib_amount: float,
    rebalance_dates: pd.DatetimeIndex,
    include_contribs: bool,
    target_growth_per_rebalance: float,
    allow_sells: bool,
    out_to_defensive: bool,
    max_buy: float | None = None,
    max_sell: float | None = None,
    cost_bps: float = 0.0,
    name: str = "9-SIG",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = price_risk.index.intersection(price_def.index)
    price_risk = price_risk.reindex(idx)
    price_def = price_def.reindex(idx)

    rebalance_dates = pd.DatetimeIndex(rebalance_dates).intersection(idx)
    reb_set = set(rebalance_dates)

    cash = float(initial_capital)
    risk_sh = 0.0
    def_sh = 0.0
    risk_cb = 0.0
    def_cb = 0.0

    contrib_total = float(initial_capital)
    target = float(initial_capital)

    daily_equity, daily_contrib = [], []
    ledger = []

    def pr(dt): return float(price_risk.at[dt])
    def pdv(dt): return float(price_def.at[dt])

    def pv(dt):
        return cash + risk_sh * pr(dt) + def_sh * pdv(dt)

    def buy_risk(dt, amount):
        nonlocal cash, risk_sh, risk_cb
        amt = min(amount, cash)
        if amt <= 0: return 0.0
        cash = _apply_cost(cash, amt, cost_bps)
        amt2 = min(amt, max(0.0, cash)) if cash < amt else amt
        p = pr(dt)
        risk_sh += amt2 / p
        risk_cb += amt2
        cash -= amt2
        return amt2

    def sell_risk_value(dt, value):
        nonlocal cash, risk_sh, risk_cb
        if value <= 0: return 0.0
        p = pr(dt)
        held = risk_sh * p
        sell_val = min(value, held)
        if sell_val <= 0: return 0.0
        sell_sh = sell_val / p
        cb_removed = risk_cb * (sell_sh / risk_sh) if risk_sh > 0 else 0.0
        risk_sh -= sell_sh
        risk_cb -= cb_removed
        cash += sell_val
        cash = _apply_cost(cash, sell_val, cost_bps)
        return sell_val

    def buy_def(dt, amount):
        nonlocal cash, def_sh, def_cb
        amt = min(amount, cash)
        if amt <= 0: return 0.0
        cash = _apply_cost(cash, amt, cost_bps)
        amt2 = min(amt, max(0.0, cash)) if cash < amt else amt
        p = pdv(dt)
        def_sh += amt2 / p
        def_cb += amt2
        cash -= amt2
        return amt2

    def sell_def_value(dt, value):
        nonlocal cash, def_sh, def_cb
        if value <= 0: return 0.0
        p = pdv(dt)
        held = def_sh * p
        sell_val = min(value, held)
        if sell_val <= 0: return 0.0
        sell_sh = sell_val / p
        cb_removed = def_cb * (sell_sh / def_sh) if def_sh > 0 else 0.0
        def_sh -= sell_sh
        def_cb -= cb_removed
        cash += sell_val
        cash = _apply_cost(cash, sell_val, cost_bps)
        return sell_val

    def park_cash(dt):
        if out_to_defensive:
            buy_def(dt, cash)

    park_cash(idx[0])

    for dt in idx:
        daily_equity.append(pv(dt))
        daily_contrib.append(contrib_total)

        if dt not in reb_set:
            continue

        dollar_added = 0.0
        if include_contribs and contrib_amount > 0:
            cash += float(contrib_amount)
            contrib_total += float(contrib_amount)
            dollar_added = float(contrib_amount)

        target *= (1.0 + float(target_growth_per_rebalance))

        port = pv(dt)
        gap = target - port

        buy_amt = 0.0
        sell_amt = 0.0
        action = "HOLD"

        if gap > 0:
            desired = min(gap, float(max_buy)) if max_buy is not None else gap
            if out_to_defensive and cash < desired:
                sell_def_value(dt, desired - cash)
            buy_amt = buy_risk(dt, desired)
            action = "BUY" if buy_amt > 0 else "HOLD"

        elif gap < 0 and allow_sells:
            desired = min(-gap, float(max_sell)) if max_sell is not None else (-gap)
            sell_amt = sell_risk_value(dt, desired)
            action = "SELL" if sell_amt > 0 else "HOLD"

        park_cash(dt)

        risk_val = risk_sh * pr(dt)
        def_val = def_sh * pdv(dt)
        total_assets = cash + risk_val + def_val
        unreal = (risk_val - risk_cb) + (def_val - def_cb)
        pl = total_assets - contrib_total

        ledger.append({
            "RebalanceDate": dt, "InMarket": 1,
            "RiskPrice": pr(dt), "DefPrice": pdv(dt),
            "Target": target, "PortfolioValue": total_assets, "Gap": gap,
            "Action": action, "DollarAdded": dollar_added,
            "BuyAmt": buy_amt, "SellAmt": sell_amt,
            "RiskShares": risk_sh, "RiskValue": risk_val,
            "DefShares": def_sh, "DefValue": def_val,
            "Cash": cash, "Contributions": contrib_total,
            "PL": pl, "Unrealized": unreal,
        })

    daily = pd.DataFrame({"Equity": daily_equity, "Contributions": daily_contrib}, index=idx)
    daily.attrs["name"] = name
    return daily, pd.DataFrame(ledger)

# -----------------------------
# LRS rotation (signal from separate ticker allowed)
# -----------------------------
def backtest_lrs_rotation(
    price_risk: pd.Series,
    price_def: pd.Series,
    signal_price: pd.Series,
    ma: pd.Series,
    initial_capital: float,
    contrib_amount: float,
    contrib_dates: pd.DatetimeIndex,
    include_contribs: bool,
    position_size: float,
    out_to_defensive: bool = True,
    buffer_pct: float = 0.0,
    confirm_days: int = 1,
    cost_bps: float = 0.0,
    name: str = "LRS Rotation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Align all series
    idx = price_risk.index.intersection(price_def.index).intersection(signal_price.index).intersection(ma.index)
    price_risk = price_risk.reindex(idx)
    price_def = price_def.reindex(idx)
    signal_price = signal_price.reindex(idx)
    ma = ma.reindex(idx)

    contrib_dates = pd.DatetimeIndex(contrib_dates).intersection(idx)
    contrib_set = set(contrib_dates)

    in_market = compute_trend_state(signal_price, ma, buffer_pct=buffer_pct, confirm_days=confirm_days)

    cash = float(initial_capital)
    risk_sh = 0.0
    def_sh = 0.0
    risk_cb = 0.0
    def_cb = 0.0
    contrib_total = float(initial_capital)

    equity_daily, contrib_daily = [], []
    ledger = []
    prev = None

    def pr(dt): return float(price_risk.at[dt])
    def pdv(dt): return float(price_def.at[dt])

    def sell_to_cash(dt, which: str, value: float):
        nonlocal cash, risk_sh, def_sh, risk_cb, def_cb
        if value <= 0: return 0.0
        if which == "risk":
            p = pr(dt); held = risk_sh * p
            sell_val = min(value, held)
            if sell_val <= 0: return 0.0
            sell_sh = sell_val / p
            cb_removed = risk_cb * (sell_sh / risk_sh) if risk_sh > 0 else 0.0
            risk_sh -= sell_sh; risk_cb -= cb_removed
            cash += sell_val
            cash = _apply_cost(cash, sell_val, cost_bps)
            return sell_val
        else:
            p = pdv(dt); held = def_sh * p
            sell_val = min(value, held)
            if sell_val <= 0: return 0.0
            sell_sh = sell_val / p
            cb_removed = def_cb * (sell_sh / def_sh) if def_sh > 0 else 0.0
            def_sh -= sell_sh; def_cb -= cb_removed
            cash += sell_val
            cash = _apply_cost(cash, sell_val, cost_bps)
            return sell_val

    def buy_from_cash(dt, which: str, value: float):
        nonlocal cash, risk_sh, def_sh, risk_cb, def_cb
        amt = min(value, cash)
        if amt <= 0: return 0.0
        cash = _apply_cost(cash, amt, cost_bps)
        amt2 = min(amt, max(0.0, cash)) if cash < amt else amt
        if which == "risk":
            p = pr(dt); risk_sh += amt2 / p; risk_cb += amt2
        else:
            p = pdv(dt); def_sh += amt2 / p; def_cb += amt2
        cash -= amt2
        return amt2

    def record(dt, action: str, dollar_added: float):
        risk_val = risk_sh * pr(dt)
        def_val = def_sh * pdv(dt)
        total_assets = cash + risk_val + def_val
        unreal = (risk_val - risk_cb) + (def_val - def_cb)
        pl = total_assets - contrib_total
        ledger.append({
            "Date": dt, "InMarket": int(in_market.at[dt]), "Action": action, "DollarAdded": dollar_added,
            "RiskShares": risk_sh, "RiskValue": risk_val,
            "DefShares": def_sh, "DefValue": def_val,
            "Cash": cash, "TotalAssets": total_assets,
            "Contributions": contrib_total, "PL": pl, "Unrealized": unreal,
        })

    for dt in idx:
        dollar_added = 0.0
        if include_contribs and dt in contrib_set and contrib_amount > 0:
            cash += float(contrib_amount)
            contrib_total += float(contrib_amount)
            dollar_added = float(contrib_amount)

        state = bool(in_market.at[dt])

        risk_val = risk_sh * pr(dt)
        def_val = def_sh * pdv(dt)
        total = cash + risk_val + def_val

        if state:
            target_risk = position_size * total
            target_def = 0.0
        else:
            target_risk = 0.0
            target_def = (position_size * total) if out_to_defensive else 0.0

        if risk_val > target_risk:
            sell_to_cash(dt, "risk", risk_val - target_risk)
        def_val = def_sh * pdv(dt)
        if def_val > target_def:
            sell_to_cash(dt, "def", def_val - target_def)

        risk_val = risk_sh * pr(dt)
        def_val = def_sh * pdv(dt)
        total = cash + risk_val + def_val

        if risk_val < target_risk:
            buy_from_cash(dt, "risk", target_risk - risk_val)
        if def_val < target_def:
            buy_from_cash(dt, "def", target_def - def_val)

        equity = cash + risk_sh * pr(dt) + def_sh * pdv(dt)
        equity_daily.append(equity)
        contrib_daily.append(contrib_total)

        action = ""
        if prev is None:
            action = "START"
        elif prev != state:
            action = "ENTER_RISK" if state else "EXIT_RISK"
        elif include_contribs and dt in contrib_set:
            action = "CONTRIB"
        elif dt == idx[-1]:
            action = "END"

        if action:
            record(dt, action, dollar_added)

        prev = state

    daily = pd.DataFrame({"Equity": equity_daily, "Contributions": contrib_daily}, index=idx)
    daily.attrs["name"] = name
    return daily, pd.DataFrame(ledger)

# -----------------------------
# 9-SIG + LRS gate (signal from separate ticker)
# -----------------------------
def backtest_9sig_with_lrs_gate(
    price_risk: pd.Series,
    price_def: pd.Series,
    signal_price: pd.Series,
    ma: pd.Series,
    initial_capital: float,
    contrib_amount: float,
    rebalance_dates: pd.DatetimeIndex,
    include_contribs: bool,
    target_growth_per_rebalance: float,
    allow_sells: bool,
    out_to_defensive: bool,
    buffer_pct: float = 0.0,
    confirm_days: int = 1,
    max_buy: float | None = None,
    max_sell: float | None = None,
    cost_bps: float = 0.0,
    name: str = "9SIG + LRS",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = price_risk.index.intersection(price_def.index).intersection(signal_price.index).intersection(ma.index)
    price_risk = price_risk.reindex(idx)
    price_def = price_def.reindex(idx)
    signal_price = signal_price.reindex(idx)
    ma = ma.reindex(idx)

    rebalance_dates = pd.DatetimeIndex(rebalance_dates).intersection(idx)
    reb_set = set(rebalance_dates)

    in_market = compute_trend_state(signal_price, ma, buffer_pct=buffer_pct, confirm_days=confirm_days)

    cash = float(initial_capital)
    risk_sh = 0.0
    def_sh = 0.0
    risk_cb = 0.0
    def_cb = 0.0
    contrib_total = float(initial_capital)
    target = float(initial_capital)

    equity_daily, contrib_daily = [], []
    ledger = []

    def pr(dt): return float(price_risk.at[dt])
    def pdv(dt): return float(price_def.at[dt])

    def pv(dt):
        return cash + risk_sh * pr(dt) + def_sh * pdv(dt)

    def buy_risk(dt, amount):
        nonlocal cash, risk_sh, risk_cb
        amt = min(amount, cash)
        if amt <= 0: return 0.0
        cash = _apply_cost(cash, amt, cost_bps)
        amt2 = min(amt, max(0.0, cash)) if cash < amt else amt
        p = pr(dt)
        risk_sh += amt2 / p
        risk_cb += amt2
        cash -= amt2
        return amt2

    def sell_risk_value(dt, value):
        nonlocal cash, risk_sh, risk_cb
        if value <= 0: return 0.0
        p = pr(dt)
        held = risk_sh * p
        sell_val = min(value, held)
        if sell_val <= 0: return 0.0
        sell_sh = sell_val / p
        cb_removed = risk_cb * (sell_sh / risk_sh) if risk_sh > 0 else 0.0
        risk_sh -= sell_sh
        risk_cb -= cb_removed
        cash += sell_val
        cash = _apply_cost(cash, sell_val, cost_bps)
        return sell_val

    def buy_def(dt, amount):
        nonlocal cash, def_sh, def_cb
        amt = min(amount, cash)
        if amt <= 0: return 0.0
        cash = _apply_cost(cash, amt, cost_bps)
        amt2 = min(amt, max(0.0, cash)) if cash < amt else amt
        p = pdv(dt)
        def_sh += amt2 / p
        def_cb += amt2
        cash -= amt2
        return amt2

    def sell_def_value(dt, value):
        nonlocal cash, def_sh, def_cb
        if value <= 0: return 0.0
        p = pdv(dt)
        held = def_sh * p
        sell_val = min(value, held)
        if sell_val <= 0: return 0.0
        sell_sh = sell_val / p
        cb_removed = def_cb * (sell_sh / def_sh) if def_sh > 0 else 0.0
        def_sh -= sell_sh
        def_cb -= cb_removed
        cash += sell_val
        cash = _apply_cost(cash, sell_val, cost_bps)
        return sell_val

    def park_cash(dt):
        if out_to_defensive:
            buy_def(dt, cash)

    park_cash(idx[0])

    for dt in idx:
        equity_daily.append(pv(dt))
        contrib_daily.append(contrib_total)

        if dt not in reb_set:
            continue

        dollar_added = 0.0
        if include_contribs and contrib_amount > 0:
            cash += float(contrib_amount)
            contrib_total += float(contrib_amount)
            dollar_added = float(contrib_amount)

        target *= (1.0 + float(target_growth_per_rebalance))

        state = bool(in_market.at[dt])
        gap = target - pv(dt)

        buy_amt = 0.0
        sell_amt = 0.0

        if state:
            if gap > 0:
                desired = min(gap, float(max_buy)) if max_buy is not None else gap
                if out_to_defensive and cash < desired:
                    sell_def_value(dt, desired - cash)
                buy_amt = buy_risk(dt, desired)
                action = "BUY" if buy_amt > 0 else "HOLD"
            elif gap < 0 and allow_sells:
                desired = min(-gap, float(max_sell)) if max_sell is not None else (-gap)
                sell_amt = sell_risk_value(dt, desired)
                action = "SELL" if sell_amt > 0 else "HOLD"
            else:
                action = "HOLD"
            park_cash(dt)
        else:
            # OUT: do not buy risk
            if allow_sells and gap < 0:
                desired = min(-gap, float(max_sell)) if max_sell is not None else (-gap)
                sell_amt = sell_risk_value(dt, desired)
                action = "OUT_SELL" if sell_amt > 0 else "OUT_HOLD"
            else:
                action = "OUT_HOLD"
            park_cash(dt)

        risk_val = risk_sh * pr(dt)
        def_val = def_sh * pdv(dt)
        total_assets = cash + risk_val + def_val
        unreal = (risk_val - risk_cb) + (def_val - def_cb)
        pl = total_assets - contrib_total

        ledger.append({
            "RebalanceDate": dt, "InMarket": int(state),
            "RiskPrice": pr(dt), "DefPrice": pdv(dt),
            "Target": target, "PortfolioValue": total_assets, "Gap": gap,
            "Action": action, "DollarAdded": dollar_added,
            "BuyAmt": buy_amt, "SellAmt": sell_amt,
            "RiskShares": risk_sh, "RiskValue": risk_val,
            "DefShares": def_sh, "DefValue": def_val,
            "Cash": cash, "Contributions": contrib_total,
            "PL": pl, "Unrealized": unreal,
        })

    daily = pd.DataFrame({"Equity": equity_daily, "Contributions": contrib_daily}, index=idx)
    daily.attrs["name"] = name
    return daily, pd.DataFrame(ledger)
