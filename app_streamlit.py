from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester.data import fetch_adj_close
from backtester.trading import (
    RunConfig,
    sma_signal,
    ema_signal,
    backtest_all_in_out,
    backtest_dca,
    backtest_buy_hold,
    summarize_equity,
    worst_drawdown_window,
)

st.set_page_config(page_title="TQQQ Backtester", layout="wide")

st.title("TQQQ Backtester — Strategy Comparison")


def fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"${x:,.0f}"


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.1f}%"


def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.1f}"


with st.sidebar:
    st.header("Universe")
    ticker = st.text_input("Risk Ticker", value="TQQQ")
    start = st.date_input("Start", value=pd.to_datetime("2010-01-01"))
    end = st.date_input("End", value=pd.to_datetime("2025-12-30"))

    st.divider()
    st.header("Capital")
    initial = st.number_input("Initial capital", min_value=0.0, value=10000.0, step=1000.0)
    position_size = st.slider("Position size when IN", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    st.divider()
    st.header("Strategy parameters")
    sma_len = st.number_input("SMA length", min_value=5, value=200, step=5)
    ema_len = st.number_input("EMA length (9-sig)", min_value=2, value=9, step=1)

    st.divider()
    st.header("DCA parameters")
    dca_freq = st.selectbox("DCA frequency", ["Monthly", "Quarterly"], index=0)
    dca_amount = st.number_input("DCA amount", min_value=0.0, value=1000.0, step=100.0)
    dca_day = st.number_input("DCA day of month (1..28)", min_value=1, max_value=28, value=11, step=1)

    st.divider()
    st.header("Show / Hide strategies")
    show_sma = st.checkbox("SMA crossover", value=True)
    show_ema = st.checkbox("EMA 9-sig", value=True)
    show_dca = st.checkbox("DCA", value=True)
    show_bh = st.checkbox("Buy & Hold", value=True)

    st.divider()
    run = st.button("Run backtest", type="primary")


@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str, end: str) -> pd.Series:
    return fetch_adj_close(ticker, start=start, end=end)


def add_markers(fig: go.Figure, price: pd.Series, trades: pd.DataFrame, name: str):
    if trades is None or len(trades) == 0:
        return

    px = price.astype(float)
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    trades = trades[trades["date"].isin(px.index)]

    if len(trades) == 0:
        return

    # BUY-ish markers
    buy_like = trades[trades["action"].isin(["BUY", "SWITCH_IN"])]
    sell_like = trades[trades["action"].isin(["SELL", "SWITCH_OUT"])]

    def _scatter(df, symbol, label):
        if len(df) == 0:
            return
        y = px.loc[df["date"]].values
        fig.add_trace(go.Scatter(
            x=df["date"], y=y,
            mode="markers",
            name=f"{name} {label}",
            marker=dict(symbol=symbol, size=10),
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Price=%{y:$,.2f}<extra></extra>",
        ))

    _scatter(buy_like, "triangle-up", "BUY/IN")
    _scatter(sell_like, "triangle-down", "OUT")


def normalized_curve(equity: pd.Series) -> pd.Series:
    e = equity.dropna()
    if len(e) == 0:
        return e
    return e / e.iloc[0] * 100.0


if run:
    with st.spinner("Loading price data..."):
        price = load_prices(ticker, str(start), str(end))

    cfg = RunConfig(
        initial_cash=float(initial),
        position_size=float(position_size),
        start=str(start),
        end=str(end),
    )

    results = {}

    if show_sma:
        sig = sma_signal(price, int(sma_len))
        eq, ledger, trades = backtest_all_in_out(price, sig, cfg)
        results["SMA"] = dict(equity=eq, ledger=ledger, trades=trades, contributed=float(initial))

    if show_ema:
        sig = ema_signal(price, int(ema_len))
        eq, ledger, trades = backtest_all_in_out(price, sig, cfg)
        results["EMA"] = dict(equity=eq, ledger=ledger, trades=trades, contributed=float(initial))

    if show_dca:
        eq, ledger, trades, contrib = backtest_dca(
            price=price,
            start_cash=float(initial),
            contrib_amount=float(dca_amount),
            contrib_freq=str(dca_freq),
            contrib_day=int(dca_day),
        )
        results["DCA"] = dict(equity=eq, ledger=ledger, trades=trades, contributed=float(contrib))

    if show_bh:
        eq, ledger, trades, contrib = backtest_buy_hold(price=price, start_cash=float(initial))
        results["B&H"] = dict(equity=eq, ledger=ledger, trades=trades, contributed=float(contrib))

    if len(results) == 0:
        st.warning("Turn on at least one strategy.")
        st.stop()

    # =========================
    # Summary metrics table
    # =========================
    summary_rows = []
    dd_rows = []
    for k, r in results.items():
        stats = summarize_equity(r["equity"], r["trades"])
        ddw = worst_drawdown_window(r["equity"])

        summary_rows.append({
            "Strategy": k,
            "Final Value": stats["Final Value"],
            "Total Return %": stats["Total Return %"],
            "CAGR %": stats["CAGR %"],
            "Sharpe": stats["Sharpe"],
            "Volatility %": stats["Volatility %"],
            "Max Drawdown %": stats["Max Drawdown %"],
            "Trades/Adjustments": int(stats["Trades"]),
            "Total Invested": r.get("contributed", float(initial)),
        })

        dd_rows.append({
            "Strategy": k,
            "Peak Date": ddw["Peak Date"],
            "Trough Date": ddw["Trough Date"],
            "Recovery Date": ddw["Recovery Date"],
            "Max Drawdown %": ddw["Max Drawdown %"],
        })

    summary_df = pd.DataFrame(summary_rows)
    dd_df = pd.DataFrame(dd_rows)

    # Pretty formatting
    pretty = summary_df.copy()
    pretty["Final Value"] = pretty["Final Value"].map(fmt_money)
    pretty["Total Invested"] = pretty["Total Invested"].map(fmt_money)
    pretty["Total Return %"] = pretty["Total Return %"].map(fmt_pct)
    pretty["CAGR %"] = pretty["CAGR %"].map(fmt_pct)
    pretty["Volatility %"] = pretty["Volatility %"].map(fmt_pct)
    pretty["Max Drawdown %"] = pretty["Max Drawdown %"].map(fmt_pct)
    pretty["Sharpe"] = pretty["Sharpe"].map(fmt_num)

    colA, colB = st.columns([1.2, 1.0])
    with colA:
        st.subheader("Performance Summary")
        st.dataframe(pretty, use_container_width=True, hide_index=True)

    with colB:
        st.subheader("Worst Drawdown Window (by strategy)")
        dd_pretty = dd_df.copy()
        dd_pretty["Max Drawdown %"] = dd_pretty["Max Drawdown %"].map(fmt_pct)
        st.dataframe(dd_pretty, use_container_width=True, hide_index=True)

    # =========================
    # Charts
    # =========================
    st.subheader("Price + Buy/Adjustment Markers")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=price.index, y=price.values,
        mode="lines",
        name=ticker,
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Price=%{y:$,.2f}<extra></extra>",
    ))

    # Add markers per strategy
    for name, r in results.items():
        add_markers(fig_price, price, r["trades"], name)

    fig_price.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Portfolio Value Comparison")
    fig_eq = go.Figure()
    for name, r in results.items():
        eq = r["equity"].reindex(price.index).ffill()
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            mode="lines",
            name=name,
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Value=%{y:$,.0f}<extra></extra>",
        ))
    fig_eq.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Normalized Equity Curves (Start = 100)")
    fig_norm = go.Figure()
    for name, r in results.items():
        eq = r["equity"].reindex(price.index).ffill()
        n = normalized_curve(eq)
        fig_norm.add_trace(go.Scatter(
            x=n.index, y=n.values,
            mode="lines",
            name=name,
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Index=%{y:,.1f}<extra></extra>",
        ))
    fig_norm.update_layout(
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    st.subheader("Skyscraper (Final Value)")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=summary_df["Strategy"],
        y=summary_df["Final Value"],
        hovertemplate="Strategy=%{x}<br>Final=%{y:$,.0f}<extra></extra>",
        name="Final Value",
    ))
    fig_bar.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    # =========================
    # Ledger viewer
    # =========================
    st.subheader("Trades / Adjustments Ledger")
    pick = st.selectbox("Select strategy ledger", list(results.keys()))
    ledger = results[pick]["ledger"].copy()
    if len(ledger) == 0:
        st.info("No trades/adjustments for this strategy in this window.")
    else:
        # human-friendly columns
        out = ledger.copy()
        out["price"] = out["price"].map(lambda x: f"{x:,.2f}")
        out["cash"] = out["cash"].map(fmt_money)
        out["assets"] = out["assets"].map(fmt_money)
        out["shares_delta"] = out["shares_delta"].map(lambda x: f"{x:,.6f}")
        out["shares"] = out["shares"].map(lambda x: f"{x:,.6f}")
        st.dataframe(out, use_container_width=True, hide_index=True)
else:
    st.info("Set parameters on the left and click **Run backtest**.")
