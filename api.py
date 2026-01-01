from __future__ import annotations
from fastapi import FastAPI
import pandas as pd

from backtester.schemas import ComboBacktestRequest, ComboBacktestResponse
from backtester.data import fetch_prices, align_series
from backtester.schedules import invest_dates_monthly, invest_dates_quarterly, skip_first_period
from backtester.metrics import compute_metrics, worst_drawdown_details
from backtester.robustness import rolling_window_outperformance, summarize_outperformance
from backtester.mapping import guess_signal_ticker
from backtester.strategies import (
    backtest_buy_hold,
    backtest_9sig,
    backtest_lrs_rotation,
    backtest_9sig_with_lrs_gate
)

app = FastAPI(title="Backtest API", version="0.3")

@app.post("/backtest/combo", response_model=ComboBacktestResponse)
def run_combo(req: ComboBacktestRequest):
    # choose signal ticker
    auto_sig = guess_signal_ticker(req.risk_ticker)
    signal_ticker = auto_sig if req.signal_ticker_mode == "auto" else (req.signal_ticker_custom or auto_sig)

    risk_df = fetch_prices(req.risk_ticker, req.start, req.end, use_cache=True)
    def_df = fetch_prices(req.defensive_ticker, req.start, req.end, use_cache=True)
    sig_df = fetch_prices(signal_ticker, req.start, req.end, use_cache=True)

    risk = risk_df["Close"]
    defs = def_df["Close"]
    sigp = sig_df["Close"]
    risk, defs, sigp = align_series(risk, defs, sigp)

    # schedule
    if req.rebalance_freq == "monthly":
        dates = invest_dates_monthly(risk.index, req.invest_day)
    else:
        dates = invest_dates_quarterly(risk.index, req.invest_day)
    if req.skip_first_period:
        dates = skip_first_period(dates, req.rebalance_freq)
    dates = pd.DatetimeIndex(dates).intersection(risk.index)

    # MA on SIGNAL ticker
    if req.lrs_ma_type == "SMA":
        ma = sigp.rolling(int(req.lrs_ma_len)).mean()
    else:
        ma = sigp.ewm(span=int(req.lrs_ma_len), adjust=False).mean()

    # Run strategies
    bh_daily, bh_led = backtest_buy_hold(risk, req.initial_capital, req.contrib_amount, dates, req.include_contribs, cost_bps=req.cost_bps)

    sigA_daily, sigA_led = backtest_9sig(
        risk, defs, req.initial_capital, req.contrib_amount, dates, req.include_contribs,
        req.target_growth_per_rebalance, allow_sells=False, out_to_defensive=req.out_to_defensive,
        max_buy=req.max_buy, max_sell=req.max_sell, cost_bps=req.cost_bps, name="9-SIG A"
    )
    sigB_daily, sigB_led = backtest_9sig(
        risk, defs, req.initial_capital, req.contrib_amount, dates, req.include_contribs,
        req.target_growth_per_rebalance, allow_sells=True, out_to_defensive=req.out_to_defensive,
        max_buy=req.max_buy, max_sell=req.max_sell, cost_bps=req.cost_bps, name="9-SIG B"
    )

    lrs_daily, lrs_led = backtest_lrs_rotation(
        risk, defs, sigp, ma, req.initial_capital, req.contrib_amount, dates, req.include_contribs,
        position_size=req.position_size, out_to_defensive=req.out_to_defensive,
        buffer_pct=req.buffer_pct, confirm_days=req.confirm_days,
        cost_bps=req.cost_bps, name=f"LRS({signal_ticker}:{req.lrs_ma_type}{req.lrs_ma_len})"
    )

    sigA_lrs_daily, sigA_lrs_led = backtest_9sig_with_lrs_gate(
        risk, defs, sigp, ma, req.initial_capital, req.contrib_amount, dates, req.include_contribs,
        req.target_growth_per_rebalance, allow_sells=False, out_to_defensive=req.out_to_defensive,
        buffer_pct=req.buffer_pct, confirm_days=req.confirm_days,
        max_buy=req.max_buy, max_sell=req.max_sell, cost_bps=req.cost_bps, name="9-SIG A + LRS"
    )
    sigB_lrs_daily, sigB_lrs_led = backtest_9sig_with_lrs_gate(
        risk, defs, sigp, ma, req.initial_capital, req.contrib_amount, dates, req.include_contribs,
        req.target_growth_per_rebalance, allow_sells=True, out_to_defensive=req.out_to_defensive,
        buffer_pct=req.buffer_pct, confirm_days=req.confirm_days,
        max_buy=req.max_buy, max_sell=req.max_sell, cost_bps=req.cost_bps, name="9-SIG B + LRS"
    )

    # summary helper
    def summarize(daily: pd.DataFrame):
        m = compute_metrics(daily["Equity"])
        dd = worst_drawdown_details(daily["Equity"])
        return {
            "final": m["final"],
            "total_return": m["total_return"],
            "cagr": m["cagr"],
            "sharpe": m["sharpe"],
            "vol": m["vol"],
            "max_dd": m["max_dd"],
            "total_deposited": float(daily["Contributions"].iloc[-1]),
            "worst_dd": {
                "max_drawdown": dd["max_drawdown"],
                "peak_date": str(dd["peak_date"].date()),
                "trough_date": str(dd["trough_date"].date()),
                "recovery_date": str(dd["recovery_date"].date()) if dd["recovery_date"] is not None else None,
            },
        }

    summary = {
        "meta": {"signal_ticker_used": signal_ticker},
        "Buy & Hold": summarize(bh_daily),
        "9-SIG A": summarize(sigA_daily),
        "9-SIG B": summarize(sigB_daily),
        "LRS": summarize(lrs_daily),
        "9-SIG A + LRS": summarize(sigA_lrs_daily),
        "9-SIG B + LRS": summarize(sigB_lrs_daily),
        "params": req.model_dump(),
    }

    # equity rows
    equity_rows = []
    def add_equity(label: str, daily: pd.DataFrame):
        for dt, row in daily.iterrows():
            equity_rows.append({
                "date": dt.date().isoformat(),
                "strategy": label,
                "equity": float(row["Equity"]),
                "contributions": float(row["Contributions"]),
            })

    add_equity("Buy & Hold", bh_daily)
    add_equity("9-SIG A", sigA_daily)
    add_equity("9-SIG B", sigB_daily)
    add_equity("LRS", lrs_daily)
    add_equity("9-SIG A + LRS", sigA_lrs_daily)
    add_equity("9-SIG B + LRS", sigB_lrs_daily)

    ledgers = {
        "Buy & Hold": bh_led.fillna("").to_dict(orient="records"),
        "9-SIG A": sigA_led.fillna("").to_dict(orient="records"),
        "9-SIG B": sigB_led.fillna("").to_dict(orient="records"),
        "LRS": lrs_led.fillna("").to_dict(orient="records"),
        "9-SIG A + LRS": sigA_lrs_led.fillna("").to_dict(orient="records"),
        "9-SIG B + LRS": sigB_lrs_led.fillna("").to_dict(orient="records"),
    }

    # rolling robustness
    eq_map = {
        "Buy & Hold": bh_daily["Equity"],
        "9-SIG A": sigA_daily["Equity"],
        "9-SIG B": sigB_daily["Equity"],
        "LRS": lrs_daily["Equity"],
        "9-SIG A + LRS": sigA_lrs_daily["Equity"],
        "9-SIG B + LRS": sigB_lrs_daily["Equity"],
    }
    roll = rolling_window_outperformance(eq_map, benchmark_name="Buy & Hold", window_years=3)
    roll_sum = summarize_outperformance(roll)

    rolling = {
        "window_years": 3,
        "rows": roll.to_dict(orient="records"),
        "summary": roll_sum.to_dict(orient="records"),
    }

    return ComboBacktestResponse(summary=summary, equity=equity_rows, ledgers=ledgers, rolling=rolling)
