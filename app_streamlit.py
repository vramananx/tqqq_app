from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from backtester.data import fetch_prices, align_series
from backtester.schedules import invest_dates_monthly, invest_dates_quarterly, skip_first_period
from backtester.metrics import compute_metrics, worst_drawdown_details, drawdown_series
from backtester.mapping import guess_signal_ticker, supported_main_tickers
from backtester.strategies import (
    backtest_buy_hold,
    backtest_9sig,
    backtest_lrs_rotation,
    backtest_9sig_with_lrs_gate,
)

st.set_page_config(page_title="TQQQ Strategy Lab", layout="wide")
st.title("Strategy Lab — Buy&Hold vs 9-SIG vs LRS Rotation (mobile-friendly toggles)")

def money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    ax = abs(float(x))
    if ax >= 1_000_000:
        return f"${x/1_000_000:,.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:,.1f}K"
    return f"${x:,.0f}"

def pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x*100:.1f}%"

def _hash_params(d: dict) -> str:
    s = repr(sorted(d.items())).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

# ---------------------------
# Sidebar UI
# ---------------------------
with st.sidebar:
    st.header("Tickers")

    main_mode = st.selectbox("Main ticker (dropdown)", ["Mapped list", "Custom"], index=0)
    if main_mode == "Mapped list":
        risk_ticker = st.selectbox("Risk ticker", supported_main_tickers(), index=supported_main_tickers().index("TQQQ"))
    else:
        risk_ticker = st.text_input("Risk ticker (custom)", "TQQQ")

    defensive_ticker = st.selectbox("Defensive ticker", ["BIL", "SGOV", "SHY"], index=0)

    start = st.text_input("Start", "2010-02-11")
    end = st.text_input("End", "2025-12-30")

    st.divider()
    st.header("Signal source for MA crossover")
    sig_mode = st.selectbox("Signal ticker mode", ["auto", "custom"], index=0)
    auto_sig = guess_signal_ticker(risk_ticker)
    signal_ticker = auto_sig if sig_mode == "auto" else st.text_input("Custom signal ticker", auto_sig)
    st.caption(f"Signal ticker used: **{signal_ticker}**")

    st.divider()
    st.header("Schedule")
    reb_freq = st.selectbox("Rebalance frequency", ["quarterly", "monthly"], index=0)
    invest_day = st.slider("Investment day (1–28)", 1, 28, 15, 1)
    skip_first = st.checkbox("Skip first contribution period", value=True)

    st.divider()
    st.header("Capital")
    initial = st.number_input("Initial capital", min_value=0.0, value=10_000.0, step=1000.0)
    include_contribs = st.checkbox("Include contributions", value=True)
    default_contrib = 3000.0 if reb_freq == "quarterly" else 1000.0
    contrib_amount = st.number_input("Contribution amount (on rebalance dates)", min_value=0.0, value=float(default_contrib), step=100.0)

    st.divider()
    st.header("Costs")
    cost_bps = st.number_input("Cost per trade (bps)", value=0.0, step=1.0)

    st.divider()
    st.header("9-SIG")
    target_g = st.number_input("Target growth per rebalance (0.09 = 9%)", value=0.09 if reb_freq == "quarterly" else 0.03, step=0.01)
    out_to_defensive = st.checkbox("OUT goes to defensive (vs cash)", value=True)

    st.divider()
    st.header("LRS (trend filter)")
    lrs_ma_type = st.selectbox("MA type", ["SMA", "EMA"], index=0)
    lrs_ma_len = st.slider("MA length", 50, 300, 200, 10)
    buffer_pct = st.number_input("MA buffer (0.02 = 2%)", value=0.0, step=0.01)
    confirm_days = st.slider("Confirm days", 1, 20, 3, 1)
    position_size = st.slider("Position size (%)", 0, 100, 100, 5) / 100.0

    st.divider()
    st.header("Compute strategies (requires rerun)")
    compute_bh = st.checkbox("Buy & Hold", True)
    compute_9a = st.checkbox("9-SIG A (no sells)", True)
    compute_9b = st.checkbox("9-SIG B (allow sells)", True)
    compute_lrs = st.checkbox("LRS Rotation", True)
    compute_9a_lrs = st.checkbox("9-SIG A + LRS gate", True)
    compute_9b_lrs = st.checkbox("9-SIG B + LRS gate", True)

    st.divider()
    run_btn = st.button("Run / Recompute backtest", type="primary")

# ---------------------------
# Build params + session cache
# ---------------------------
params = dict(
    risk_ticker=risk_ticker, defensive_ticker=defensive_ticker,
    start=start, end=end,
    signal_ticker=signal_ticker,
    reb_freq=reb_freq, invest_day=invest_day, skip_first=skip_first,
    initial=float(initial), include_contribs=bool(include_contribs), contrib_amount=float(contrib_amount),
    cost_bps=float(cost_bps),
    target_g=float(target_g), out_to_defensive=bool(out_to_defensive),
    lrs_ma_type=lrs_ma_type, lrs_ma_len=int(lrs_ma_len), buffer_pct=float(buffer_pct),
    confirm_days=int(confirm_days), position_size=float(position_size),
    compute=dict(bh=compute_bh, a=compute_9a, b=compute_9b, lrs=compute_lrs, a_lrs=compute_9a_lrs, b_lrs=compute_9b_lrs)
)
key = _hash_params(params)

if "results" not in st.session_state:
    st.session_state["results"] = {}
if "last_key" not in st.session_state:
    st.session_state["last_key"] = None

def compute_all():
    risk_df = fetch_prices(risk_ticker, start, end, use_cache=True)
    def_df  = fetch_prices(defensive_ticker, start, end, use_cache=True)
    sig_df  = fetch_prices(signal_ticker, start, end, use_cache=True)

    risk = risk_df["Close"]
    defs = def_df["Close"]
    sigp = sig_df["Close"]
    risk, defs, sigp = align_series(risk, defs, sigp)

    # schedule
    if reb_freq == "monthly":
        dates = invest_dates_monthly(risk.index, invest_day)
    else:
        dates = invest_dates_quarterly(risk.index, invest_day)
    if skip_first:
        dates = skip_first_period(dates, reb_freq)
    dates = pd.DatetimeIndex(dates).intersection(risk.index)

    # MA on signal
    if lrs_ma_type == "SMA":
        ma = sigp.rolling(int(lrs_ma_len)).mean()
    else:
        ma = sigp.ewm(span=int(lrs_ma_len), adjust=False).mean()

    strategies = {}

    if compute_bh:
        bh_daily, bh_led = backtest_buy_hold(risk, initial, contrib_amount, dates, include_contribs, cost_bps=cost_bps)
        strategies["Buy & Hold"] = (bh_daily, bh_led)

    if compute_9a:
        d, led = backtest_9sig(risk, defs, initial, contrib_amount, dates, include_contribs,
                               target_g, allow_sells=False, out_to_defensive=out_to_defensive,
                               max_buy=None, max_sell=None, cost_bps=cost_bps, name="9-SIG A")
        strategies["9-SIG A"] = (d, led)

    if compute_9b:
        d, led = backtest_9sig(risk, defs, initial, contrib_amount, dates, include_contribs,
                               target_g, allow_sells=True, out_to_defensive=out_to_defensive,
                               max_buy=None, max_sell=None, cost_bps=cost_bps, name="9-SIG B")
        strategies["9-SIG B"] = (d, led)

    if compute_lrs:
        d, led = backtest_lrs_rotation(risk, defs, sigp, ma, initial, contrib_amount, dates, include_contribs,
                                       position_size=position_size, out_to_defensive=out_to_defensive,
                                       buffer_pct=buffer_pct, confirm_days=confirm_days,
                                       cost_bps=cost_bps,
                                       name=f"LRS({signal_ticker}:{lrs_ma_type}{lrs_ma_len})")
        strategies[d.attrs.get("name","LRS")] = (d, led)

    if compute_9a_lrs:
        d, led = backtest_9sig_with_lrs_gate(risk, defs, sigp, ma, initial, contrib_amount, dates, include_contribs,
                                             target_g, allow_sells=False, out_to_defensive=out_to_defensive,
                                             buffer_pct=buffer_pct, confirm_days=confirm_days,
                                             max_buy=None, max_sell=None, cost_bps=cost_bps, name="9-SIG A + LRS")
        strategies["9-SIG A + LRS"] = (d, led)

    if compute_9b_lrs:
        d, led = backtest_9sig_with_lrs_gate(risk, defs, sigp, ma, initial, contrib_amount, dates, include_contribs,
                                             target_g, allow_sells=True, out_to_defensive=out_to_defensive,
                                             buffer_pct=buffer_pct, confirm_days=confirm_days,
                                             max_buy=None, max_sell=None, cost_bps=cost_bps, name="9-SIG B + LRS")
        strategies["9-SIG B + LRS"] = (d, led)

    return dict(
        price=risk, defensive=defs, signal=sigp, ma=ma, schedule=dates,
        strategies=strategies,
        meta=dict(signal_ticker_used=signal_ticker, risk_ticker=risk_ticker, defensive_ticker=defensive_ticker)
    )

# Trigger compute when requested or cache miss
if run_btn or (key not in st.session_state["results"]):
    with st.spinner("Computing backtests..."):
        st.session_state["results"][key] = compute_all()
        st.session_state["last_key"] = key

res = st.session_state["results"].get(st.session_state["last_key"] or key)
if res is None:
    st.info("Configure settings and click **Run / Recompute backtest**.")
    st.stop()

price = res["price"]
strategies = res["strategies"]

if len(strategies) == 0:
    st.warning("No strategies computed. Enable at least one in the sidebar.")
    st.stop()

# ---------------------------
# View-only toggles (NO recompute)
# ---------------------------
st.subheader("View controls (does not rerun backtests)")
all_names = list(strategies.keys())
show_strats = st.multiselect("Show strategies", all_names, default=all_names)

chart_cols = st.columns(5)
with chart_cols[0]:
    show_price_chart = st.checkbox("Price+Markers", True)
with chart_cols[1]:
    show_equity_chart = st.checkbox("Equity", True)
with chart_cols[2]:
    show_dd_chart = st.checkbox("Drawdown", True)
with chart_cols[3]:
    show_norm_chart = st.checkbox("Normalized", True)
with chart_cols[4]:
    show_bar_chart = st.checkbox("Skyscraper", True)

# ---------------------------
# Summary table + trade counts
# ---------------------------
rows = []
dd_rows = []
for name, (daily, ledger) in strategies.items():
    m = compute_metrics(daily["Equity"])
    dd = worst_drawdown_details(daily["Equity"])

    # Trade count definition:
    # - 9-SIG ledgers have Action BUY/SELL/OUT_...
    # - LRS ledger has ENTER/EXIT/CONTRIB
    # We'll count any row in ledger where Action is non-empty AND not END.
    trades = 0
    if ledger is not None and len(ledger) > 0:
        action_col = "Action" if "Action" in ledger.columns else None
        if action_col:
            trades = int((ledger[action_col].astype(str) != "").sum())

    rows.append({
        "Strategy": name,
        "Deposited": float(daily["Contributions"].iloc[-1]),
        "Final": m["final"],
        "CAGR": m["cagr"],
        "Sharpe": m["sharpe"],
        "Vol": m["vol"],
        "MaxDD": m["max_dd"],
        "Trades/Adjustments": trades,
    })
    dd_rows.append({
        "Strategy": name,
        "Peak": str(dd["peak_date"].date()),
        "Trough": str(dd["trough_date"].date()),
        "Recovery": str(dd["recovery_date"].date()) if dd["recovery_date"] is not None else None,
        "MaxDD": dd["max_drawdown"],
    })

summary = pd.DataFrame(rows)
disp = summary.copy()
disp["Deposited"] = disp["Deposited"].map(money)
disp["Final"] = disp["Final"].map(money)
disp["CAGR"] = (disp["CAGR"]*100).round(1).map(lambda x: f"{x:.1f}%")
disp["Vol"] = (disp["Vol"]*100).round(1).map(lambda x: f"{x:.1f}%")
disp["MaxDD"] = (disp["MaxDD"]*100).round(1).map(lambda x: f"{x:.1f}%")
disp["Sharpe"] = disp["Sharpe"].round(2)

c1, c2 = st.columns([1.2, 1.0])
with c1:
    st.subheader("Performance summary")
    st.dataframe(disp, use_container_width=True)

with c2:
    st.subheader("Worst drawdown window")
    ddf = pd.DataFrame(dd_rows)
    ddf["MaxDD"] = (ddf["MaxDD"]*100).round(1).map(lambda x: f"{x:.1f}%")
    st.dataframe(ddf, use_container_width=True)

# ---------------------------
# Marker helpers
# ---------------------------
def extract_markers(name: str, daily: pd.DataFrame, ledger: pd.DataFrame):
    """
    Return two marker series: buy_dates, sell_dates mapped to y values for price/equity charts.
    We infer BUY/SELL/ENTER/EXIT from Action text.
    """
    if ledger is None or len(ledger) == 0:
        return [], [], [], []

    # pick date col
    date_col = None
    for c in ["RebalanceDate", "Date"]:
        if c in ledger.columns:
            date_col = c
            break
    if date_col is None:
        return [], [], [], []

    action_col = "Action" if "Action" in ledger.columns else None
    if action_col is None:
        return [], [], [], []

    df = ledger[[date_col, action_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[action_col] = df[action_col].astype(str)

    buy_like = df[df[action_col].str.contains("BUY|ENTER", case=False, regex=True)][date_col].tolist()
    sell_like = df[df[action_col].str.contains("SELL|EXIT|OUT_SELL", case=False, regex=True)][date_col].tolist()

    # y-values for price markers
    buy_y_price = [float(price.reindex([d]).iloc[0]) if d in price.index else np.nan for d in buy_like]
    sell_y_price = [float(price.reindex([d]).iloc[0]) if d in price.index else np.nan for d in sell_like]

    eq = daily["Equity"].reindex(price.index).ffill()
    buy_y_eq = [float(eq.reindex([d]).iloc[0]) if d in eq.index else np.nan for d in buy_like]
    sell_y_eq = [float(eq.reindex([d]).iloc[0]) if d in eq.index else np.nan for d in sell_like]

    return buy_like, buy_y_price, sell_like, sell_y_price, buy_y_eq, sell_y_eq

# ---------------------------
# Price chart + markers
# ---------------------------
if show_price_chart:
    st.subheader("Price + per-strategy markers")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price.index, y=price.values, mode="lines",
        name=f"{res['meta']['risk_ticker']} price",
        hovertemplate="%{x|%Y-%m-%d}<br>Price=$%{y:,.2f}<extra></extra>"
    ))

    for name in show_strats:
        daily, ledger = strategies[name]
        buy_d, buy_y_p, sell_d, sell_y_p, _, _ = extract_markers(name, daily, ledger)

        if len(buy_d):
            fig.add_trace(go.Scatter(
                x=buy_d, y=buy_y_p, mode="markers",
                name=f"{name} BUY/IN",
                marker=dict(symbol="triangle-up", size=10),
                hovertemplate="%{x|%Y-%m-%d}<br>BUY/IN<br>Price=$%{y:,.2f}<extra></extra>"
            ))
        if len(sell_d):
            fig.add_trace(go.Scatter(
                x=sell_d, y=sell_y_p, mode="markers",
                name=f"{name} SELL/OUT",
                marker=dict(symbol="triangle-down", size=10),
                hovertemplate="%{x|%Y-%m-%d}<br>SELL/OUT<br>Price=$%{y:,.2f}<extra></extra>"
            ))

    fig.update_layout(hovermode="x unified", height=520, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Equity chart + markers
# ---------------------------
if show_equity_chart:
    st.subheader("Equity curves + per-strategy markers")
    fig = go.Figure()

    for name in show_strats:
        daily, ledger = strategies[name]
        eq = daily["Equity"].reindex(price.index).ffill()
        contrib = daily["Contributions"].reindex(price.index).ffill()

        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines",
            name=name,
            customdata=np.stack([contrib.values], axis=1),
            hovertemplate="%{x|%Y-%m-%d}<br><b>Value</b>=$%{y:,.0f}<br><b>Deposited</b>=$%{customdata[0]:,.0f}<extra></extra>"
        ))

        buy_d, _, sell_d, _, buy_y_eq, sell_y_eq = extract_markers(name, daily, ledger)

        if len(buy_d):
            fig.add_trace(go.Scatter(
                x=buy_d, y=buy_y_eq, mode="markers",
                name=f"{name} BUY/IN",
                marker=dict(symbol="circle", size=8),
                hovertemplate="%{x|%Y-%m-%d}<br>BUY/IN<br>Equity=$%{y:,.0f}<extra></extra>"
            ))
        if len(sell_d):
            fig.add_trace(go.Scatter(
                x=sell_d, y=sell_y_eq, mode="markers",
                name=f"{name} SELL/OUT",
                marker=dict(symbol="x", size=9),
                hovertemplate="%{x|%Y-%m-%d}<br>SELL/OUT<br>Equity=$%{y:,.0f}<extra></extra>"
            ))

    fig.update_layout(hovermode="x unified", height=520, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Drawdown chart
# ---------------------------
if show_dd_chart:
    st.subheader("Drawdown comparison")
    fig = go.Figure()
    for name in show_strats:
        daily, _ = strategies[name]
        dd = drawdown_series(daily["Equity"].reindex(price.index).ffill())
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=name,
            hovertemplate="%{x|%Y-%m-%d}<br>DD=%{y:.1%}<extra></extra>"
        ))
    fig.update_layout(hovermode="x unified", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Normalized equity chart
# ---------------------------
if show_norm_chart:
    st.subheader("Normalized equity (start = 100)")
    fig = go.Figure()
    for name in show_strats:
        daily, _ = strategies[name]
        eq = daily["Equity"].reindex(price.index).ffill()
        norm = eq / eq.iloc[0] * 100.0
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, mode="lines", name=name,
            hovertemplate="%{x|%Y-%m-%d}<br>Index=%{y:,.1f}<extra></extra>"
        ))
    fig.update_layout(hovermode="x unified", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Skyscraper chart
# ---------------------------
if show_bar_chart:
    st.subheader("Skyscraper comparison (Final equity)")
    bar = summary[summary["Strategy"].isin(show_strats)].copy()
    fig = px.bar(bar, x="Strategy", y="Final", text="Final")
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig.update_layout(height=420, yaxis_title="Final ($)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Ledger viewer
# ---------------------------
st.subheader("Ledger viewer")
pick = st.selectbox("Pick a strategy", show_strats if len(show_strats) else list(strategies.keys()))
daily, ledger = strategies[pick]
if ledger is None or len(ledger) == 0:
    st.info("No ledger rows for this strategy.")
else:
    led = ledger.copy()
    # readability
    for col in led.columns:
        if col.lower().endswith("shares"):
            led[col] = pd.to_numeric(led[col], errors="coerce").round(6)
        if col.lower().endswith("value") or col in ["Cash","TotalAssets","Contributions","PL","Unrealized","Target","PortfolioValue","Gap","BuyAmt","SellAmt","DollarAdded"]:
            led[col] = pd.to_numeric(led[col], errors="coerce").round(0)
    st.dataframe(led, use_container_width=True, height=420)
