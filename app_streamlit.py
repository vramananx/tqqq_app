from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from backtester.data import fetch_prices, align_series
from backtester.schedules import invest_dates_monthly, invest_dates_quarterly, skip_first_period
from backtester.metrics import compute_metrics, worst_drawdown_details, drawdown_series
from backtester.robustness import rolling_window_outperformance, summarize_outperformance
from backtester.mapping import guess_signal_ticker
from backtester.strategies import (
    backtest_buy_hold,
    backtest_9sig,
    backtest_lrs_rotation,
    backtest_9sig_with_lrs_gate
)

from backtester.phase15 import (
    backtest_lrs_vol_target,
    backtest_dual_momentum_rotation,
    walk_forward_optimize_9sig_lrs
)

st.set_page_config(page_title="9-SIG + LRS Lab", layout="wide")
st.title("Research Cockpit — Buy&Hold vs 9-SIG (A/B) vs LRS vs 9-SIG+LRS (with robustness + costs)")

def money(x: float) -> str:
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000:
        return f"${x/1_000_000:,.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:,.1f}K"
    return f"${x:,.0f}"

with st.sidebar:
    st.header("Data")
    risk_ticker = st.text_input("Risk ticker", "TQQQ")
    def_ticker = st.selectbox("Defensive ticker", ["AGG", "BIL", "SGOV", "SHY"], index=0)
    start = st.text_input("Start", "2010-02-11")
    end = st.text_input("End", "2025-12-30")

    st.divider()
    st.header("Signal source for MA crossover")
    sig_mode = st.selectbox("Signal ticker mode", ["auto", "custom"], index=0)
    auto_sig = guess_signal_ticker(risk_ticker)
    if sig_mode == "custom":
        signal_ticker = st.text_input("Custom signal ticker", auto_sig)
    else:
        signal_ticker = auto_sig
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
    contrib_amount = st.number_input(
        "Contribution amount (applies on rebalance dates)",
        min_value=0.0,
        value=3_000.0 if reb_freq == "quarterly" else 1_000.0,
        step=100.0
    )

    st.divider()
    st.header("Transaction costs (simple)")
    cost_bps = st.number_input("Cost per trade (bps). Example 10 = 0.10%", value=0.0, step=1.0)

    st.divider()
    st.header("9-SIG")
    target_g = st.number_input(
        "Target growth per rebalance (0.09 = 9%)",
        value=0.09 if reb_freq == "quarterly" else 0.03,
        step=0.01
    )
    out_to_defensive = st.checkbox("Park idle cash in defensive", value=True)

    st.divider()
    st.header("9-SIG caps (optional)")
    cap_on = st.checkbox("Cap buy/sell per rebalance", value=False)
    max_buy = st.number_input("Max buy", value=50_000.0, step=1000.0) if cap_on else None
    max_sell = st.number_input("Max sell", value=50_000.0, step=1000.0) if cap_on else None

    st.divider()
    st.header("LRS (trend filter)")
    lrs_ma_type = st.selectbox("MA type", ["SMA", "EMA"], index=0)
    lrs_ma_len = st.slider("MA length", 50, 300, 200, 10)
    buffer_pct = st.number_input("MA buffer (0.02=2%)", value=0.0, step=0.01)
    confirm_days = st.slider("Confirm days", 1, 20, 3, 1)
    position_size = st.slider("Position size (%)", 0, 100, 100, 5) / 100.0

    st.divider()
    st.header("Phase 1.5 overlays")
    
    enable_vol_target = st.checkbox("Enable LRS + Volatility Targeting strategy", value=True)
    vol_lookback = st.slider("Vol lookback (days)", 10, 120, 20, 5)
    target_vol = st.number_input("Target vol (annual, e.g. 0.20)", value=0.20, step=0.05)
    max_expo = st.slider("Max exposure cap", 0.0, 2.0, 1.0, 0.1)
    
    st.divider()
    enable_dual_momo = st.checkbox("Enable Dual Momentum Rotation strategy", value=True)
    momo_lookback = st.slider("Momentum lookback (days)", 21, 252, 126, 21)
    abs_momo = st.checkbox("Require absolute momentum > 0 (else defensive)", value=True)
    risk_candidates = st.text_input("Risk candidates (comma)", "QQQ,SPY,IWM").replace(" ", "")
    # Defensive candidate is your def_ticker already.
    
    st.divider()
    enable_walk_forward = st.checkbox("Enable Walk-Forward optimization (9-SIG+LRS)", value=False)
    wf_train_years = st.slider("WF train years", 2, 10, 5, 1)
    wf_test_years = st.slider("WF test years", 1, 3, 1, 1)
    wf_penalty = st.number_input("WF DD penalty (score=CAGR-penalty*|DD|)", value=0.5, step=0.1)

    st.divider()
    st.header("Robustness")
    roll_years = st.selectbox("Rolling window (years)", [1, 3, 5], index=1)

    st.divider()
    st.header("Mini sweep (optional)")
    do_sweep = st.checkbox("Run mini parameter sweep", value=False)
    if do_sweep:
        sweep_g = st.multiselect("9-SIG g values", [0.06, 0.09, 0.12], default=[0.06, 0.09, 0.12])
        sweep_ma = st.multiselect("LRS MA lengths", [100, 150, 200, 250], default=[150, 200, 250])

# Load data
risk_df = fetch_prices(risk_ticker, start, end, use_cache=True)
def_df = fetch_prices(def_ticker, start, end, use_cache=True)
sig_df = fetch_prices(signal_ticker, start, end, use_cache=True)

risk = risk_df["Close"]
defs = def_df["Close"]
sigp = sig_df["Close"]
risk, defs, sigp = align_series(risk, defs, sigp)

# Dual momentum price set
cand_list = [t for t in risk_candidates.split(",") if t]
cand_list = list(dict.fromkeys(cand_list))  # unique keep order

prices_for_momo = {}
if enable_dual_momo:
    # Always include defensive ticker as fallback
    momo_tickers = sorted(set(cand_list + [def_ticker]))
    for t in momo_tickers:
        df = fetch_prices(t, start, end, use_cache=True)
        prices_for_momo[t] = df["Close"]


# Schedule shared for apples-to-apples
if reb_freq == "monthly":
    dates = invest_dates_monthly(risk.index, invest_day)
else:
    dates = invest_dates_quarterly(risk.index, invest_day)
if skip_first:
    dates = skip_first_period(dates, reb_freq)
dates = pd.DatetimeIndex(dates).intersection(risk.index)

# MA on SIGNAL ticker
if lrs_ma_type == "SMA":
    ma = sigp.rolling(int(lrs_ma_len)).mean()
else:
    ma = sigp.ewm(span=int(lrs_ma_len), adjust=False).mean()

# Run strategies
bh_daily, bh_led = backtest_buy_hold(risk, initial, contrib_amount, dates, include_contribs, cost_bps=cost_bps)

sigA_daily, sigA_led = backtest_9sig(
    risk, defs, initial, contrib_amount, dates, include_contribs,
    target_g, allow_sells=False, out_to_defensive=out_to_defensive,
    max_buy=max_buy, max_sell=max_sell, cost_bps=cost_bps, name="9-SIG A"
)

sigB_daily, sigB_led = backtest_9sig(
    risk, defs, initial, contrib_amount, dates, include_contribs,
    target_g, allow_sells=True, out_to_defensive=out_to_defensive,
    max_buy=max_buy, max_sell=max_sell, cost_bps=cost_bps, name="9-SIG B"
)

lrs_daily, lrs_led = backtest_lrs_rotation(
    risk, defs, sigp, ma, initial, contrib_amount, dates, include_contribs,
    position_size=position_size, out_to_defensive=out_to_defensive,
    buffer_pct=buffer_pct, confirm_days=confirm_days, cost_bps=cost_bps,
    name=f"LRS({signal_ticker}:{lrs_ma_type}{lrs_ma_len})"
)

sigA_lrs_daily, sigA_lrs_led = backtest_9sig_with_lrs_gate(
    risk, defs, sigp, ma, initial, contrib_amount, dates, include_contribs,
    target_g, allow_sells=False, out_to_defensive=out_to_defensive,
    buffer_pct=buffer_pct, confirm_days=confirm_days,
    max_buy=max_buy, max_sell=max_sell, cost_bps=cost_bps, name="9-SIG A + LRS"
)

sigB_lrs_daily, sigB_lrs_led = backtest_9sig_with_lrs_gate(
    risk, defs, sigp, ma, initial, contrib_amount, dates, include_contribs,
    target_g, allow_sells=True, out_to_defensive=out_to_defensive,
    buffer_pct=buffer_pct, confirm_days=confirm_days,
    max_buy=max_buy, max_sell=max_sell, cost_bps=cost_bps, name="9-SIG B + LRS"
)

strategies = [
    ("Buy & Hold", bh_daily, bh_led),
    ("9-SIG A (no sells)", sigA_daily, sigA_led),
    ("9-SIG B (allow sells)", sigB_daily, sigB_led),
    (lrs_daily.attrs.get("name","LRS"), lrs_daily, lrs_led),
    ("9-SIG A + LRS", sigA_lrs_daily, sigA_lrs_led),
    ("9-SIG B + LRS", sigB_lrs_daily, sigB_lrs_led),
]

# Phase 1.5 strategy: LRS + Vol targeting
if enable_vol_target:
    vt_daily = backtest_lrs_vol_target(
        price_risk=risk,
        price_def=defs,
        signal_price=sigp,
        ma=ma,
        initial_capital=initial,
        contrib_amount=contrib_amount,
        contrib_dates=dates,
        include_contribs=include_contribs,
        out_to_defensive=out_to_defensive,
        buffer_pct=buffer_pct,
        confirm_days=confirm_days,
        vol_lookback=vol_lookback,
        target_vol=target_vol,
        max_exposure=max_expo,
        min_exposure=0.0,
    )
    vt_daily.attrs["name"] = f"LRS+VolTarget({signal_ticker})"
    strategies.append((vt_daily.attrs["name"], vt_daily[["Equity","Contributions"]], None))

# Phase 1.5 strategy: Dual Momentum Rotation
if enable_dual_momo and prices_for_momo:
    # align all series
    common_idx = None
    for s in prices_for_momo.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    for k in list(prices_for_momo.keys()):
        prices_for_momo[k] = prices_for_momo[k].reindex(common_idx)

    momo_daily, momo_led = backtest_dual_momentum_rotation(
        prices=prices_for_momo,
        defensive_ticker=def_ticker,
        contrib_dates=dates,
        initial_capital=initial,
        contrib_amount=contrib_amount,
        include_contribs=include_contribs,
        rebalance_dates=dates,
        lookback_days=momo_lookback,
        use_absolute_momentum=abs_momo
    )
    momo_daily.attrs["name"] = f"DualMomentum({','.join(cand_list)}→{def_ticker})"
    strategies.append((momo_daily.attrs["name"], momo_daily, momo_led))


# Summary table
rows = []
for name, daily, _ in strategies:
    m = compute_metrics(daily["Equity"])
    dd = worst_drawdown_details(daily["Equity"])
    rows.append({
        "Strategy": name,
        "Total deposited": float(daily["Contributions"].iloc[-1]),
        "Final": m["final"],
        "CAGR": m["cagr"],
        "Sharpe": m["sharpe"],
        "Vol": m["vol"],
        "MaxDD": m["max_dd"],
        "DD trough": str(dd["trough_date"].date()),
    })

summary = pd.DataFrame(rows)
disp = summary.copy()
disp["Total deposited"] = disp["Total deposited"].map(money)
disp["Final"] = disp["Final"].map(money)
disp["CAGR"] = (disp["CAGR"]*100).round(1).map(lambda x: f"{x:.1f}%")
disp["Vol"] = (disp["Vol"]*100).round(1).map(lambda x: f"{x:.1f}%")
disp["MaxDD"] = (disp["MaxDD"]*100).round(1).map(lambda x: f"{x:.1f}%")
disp["Sharpe"] = disp["Sharpe"].round(2)

st.subheader("Performance summary")
st.dataframe(disp, use_container_width=True)

# Equity chart
st.subheader("Equity curves (hover shows value + deposited)")
fig = go.Figure()
for name, daily, _ in strategies:
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["Equity"], mode="lines", name=name,
        customdata=np.stack([daily["Contributions"].values], axis=1),
        hovertemplate="%{x|%Y-%m-%d}<br><b>Value</b>: $%{y:,.0f}<br><b>Deposited</b>: $%{customdata[0]:,.0f}<extra>"+name+"</extra>"
    ))
fig.update_layout(hovermode="x unified", height=520, xaxis_title="Date", yaxis_title="Equity ($)")
st.plotly_chart(fig, use_container_width=True)

# Drawdown chart
st.subheader("Drawdown comparison")
fig = go.Figure()
for name, daily, _ in strategies:
    dd = drawdown_series(daily["Equity"])
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, mode="lines", name=name,
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1%}<extra>"+name+"</extra>"
    ))
fig.update_layout(hovermode="x unified", height=420, xaxis_title="Date", yaxis_title="Drawdown")
st.plotly_chart(fig, use_container_width=True)

# Skyscraper
st.subheader("Skyscraper comparison")
bar = summary.copy()
bar["CAGR_pct"] = bar["CAGR"]*100.0
bar["MaxDD_pct"] = bar["MaxDD"]*100.0

fig = px.bar(bar, x="Strategy", y="Final", text="Final", title="Final equity")
fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
fig.update_layout(height=420, yaxis_title="Final ($)")
st.plotly_chart(fig, use_container_width=True)

# Rolling robustness vs Buy&Hold
st.subheader(f"Robustness — rolling {roll_years}Y windows vs Buy & Hold")
eq_map = {name: daily["Equity"] for name, daily, _ in strategies}
roll = rolling_window_outperformance(eq_map, benchmark_name="Buy & Hold", window_years=int(roll_years))
roll_sum = summarize_outperformance(roll)

if len(roll_sum) > 0:
    rs = roll_sum.copy()
    rs["pct_windows_beating_bench"] = (rs["pct_windows_beating_bench"]*100).round(1).map(lambda x: f"{x:.1f}%")
    rs["median_excess_cagr"] = (rs["median_excess_cagr"]*100).round(2).map(lambda x: f"{x:.2f}%")
    rs["worst_excess_cagr"] = (rs["worst_excess_cagr"]*100).round(2).map(lambda x: f"{x:.2f}%")
    st.dataframe(rs, use_container_width=True)
else:
    st.info("Not enough data for rolling window calculation.")

# Mini sweep (lightweight)
if do_sweep:
    st.subheader("Mini sweep results (quick grid)")
    sweep_rows = []
    for gg in sweep_g:
        for mm in sweep_ma:
            if lrs_ma_type == "SMA":
                ma2 = sigp.rolling(int(mm)).mean()
            else:
                ma2 = sigp.ewm(span=int(mm), adjust=False).mean()

            dA, _ = backtest_9sig_with_lrs_gate(
                risk, defs, sigp, ma2, initial, contrib_amount, dates, include_contribs,
                float(gg), allow_sells=False, out_to_defensive=out_to_defensive,
                buffer_pct=buffer_pct, confirm_days=confirm_days,
                max_buy=max_buy, max_sell=max_sell, cost_bps=cost_bps
            )
            mA = compute_metrics(dA["Equity"])

            dB, _ = backtest_9sig_with_lrs_gate(
                risk, defs, sigp, ma2, initial, contrib_amount, dates, include_contribs,
                float(gg), allow_sells=True, out_to_defensive=out_to_defensive,
                buffer_pct=buffer_pct, confirm_days=confirm_days,
                max_buy=max_buy, max_sell=max_sell, cost_bps=cost_bps
            )
            mB = compute_metrics(dB["Equity"])

            sweep_rows.append({
                "g": gg, "ma_len": mm,
                "A_final": mA["final"], "A_cagr": mA["cagr"], "A_maxdd": mA["max_dd"],
                "B_final": mB["final"], "B_cagr": mB["cagr"], "B_maxdd": mB["max_dd"],
            })

    sdf = pd.DataFrame(sweep_rows)
    sdf["A_final"] = sdf["A_final"].round(0)
    sdf["B_final"] = sdf["B_final"].round(0)
    sdf["A_cagr"] = (sdf["A_cagr"]*100).round(2)
    sdf["B_cagr"] = (sdf["B_cagr"]*100).round(2)
    sdf["A_maxdd"] = (sdf["A_maxdd"]*100).round(2)
    sdf["B_maxdd"] = (sdf["B_maxdd"]*100).round(2)
    st.dataframe(sdf, use_container_width=True)

if enable_walk_forward:
    st.subheader("Walk-Forward optimization (9-SIG + LRS)")

    # Build param grid (keep small to avoid slow runs)
    grid = []
    for gg in [0.06, 0.09, 0.12]:
        for mm in [150, 200, 250]:
            grid.append({"g": gg, "ma_len": mm})

    # Define runner that slices data and runs 9-SIG B + LRS
    def run_fn(params, sdt, edt):
        # slice series
        sl = (risk.index >= pd.Timestamp(sdt)) & (risk.index <= pd.Timestamp(edt))
        r2 = risk.loc[sl]
        d2 = defs.reindex(r2.index)
        sp2 = sigp.reindex(r2.index)

        ma2 = sp2.rolling(int(params["ma_len"])).mean() if lrs_ma_type == "SMA" else sp2.ewm(span=int(params["ma_len"]), adjust=False).mean()

        # slice schedule dates to period
        dt2 = pd.DatetimeIndex(dates).intersection(r2.index)

        # Run: choose B (allow sells) because it’s the “full” 9-SIG
        from backtester.strategies import backtest_9sig_with_lrs_gate
        daily, _ = backtest_9sig_with_lrs_gate(
            r2, d2, sp2, ma2,
            initial_capital=initial,
            contrib_amount=contrib_amount,
            rebalance_dates=dt2,
            include_contribs=include_contribs,
            target_growth_per_rebalance=float(params["g"]),
            allow_sells=True,
            out_to_defensive=out_to_defensive,
            buffer_pct=buffer_pct,
            confirm_days=confirm_days,
            max_buy=max_buy,
            max_sell=max_sell,
            cost_bps=cost_bps,
            name="WF 9SIGB+LRS"
        )
        return daily

    wf = walk_forward_optimize_9sig_lrs(
        run_fn=run_fn,
        param_grid=grid,
        dates=risk.index,
        train_years=wf_train_years,
        test_years=wf_test_years,
        score_dd_penalty=wf_penalty
    )

    if len(wf) == 0:
        st.info("Not enough data for the walk-forward settings.")
    else:
        wf2 = wf.copy()
        wf2["test_cagr"] = (wf2["test_cagr"]*100).round(2).map(lambda x: f"{x:.2f}%")
        wf2["test_maxdd"] = (wf2["test_maxdd"]*100).round(2).map(lambda x: f"{x:.2f}%")
        wf2["test_final"] = wf2["test_final"].round(0).map(lambda x: f"${x:,.0f}")
        st.dataframe(wf2, use_container_width=True)

# Ledger viewer
st.subheader("Signals / Rebalance ledger")
ledger_map = {name: led for name, _, led in strategies}
pick = st.selectbox("Pick a strategy ledger", list(ledger_map.keys()), index=3)
led = ledger_map[pick].copy()

for col in led.columns:
    if col.lower().endswith("shares"):
        led[col] = pd.to_numeric(led[col], errors="coerce").round(5)
    if col.lower().endswith("value") or col in ["Cash","TotalAssets","Contributions","PL","Unrealized","Target","PortfolioValue","Gap","BuyAmt","SellAmt","DollarAdded"]:
        led[col] = pd.to_numeric(led[col], errors="coerce").round(0)

st.dataframe(led, use_container_width=True, height=420)
