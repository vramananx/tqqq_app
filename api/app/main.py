from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

from .models import BacktestRequest, BacktestResponse, StrategyResult, SeriesPoint, Metrics, UnderwaterStats, TradeMarker
from .services.data import fetch_close
from .services.analytics import compute_metrics, underwater_vs_contrib
from .services.backtest import run_stub_strategies, to_points, to_markers

app = FastAPI(title="Strategy Lab API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    price = fetch_close(req.risk_ticker, req.start, req.end, use_cache=True)
    price = price.dropna()
    if len(price) < 50:
        raise ValueError("Not enough price data.")

    # TODO: replace stub with your real engine:
    runs = run_stub_strategies(price, req.initial, req.contrib_amount, req.contrib_freq, req.contrib_day)

    strategies = []

    if req.toggles.buy_hold and "buy_hold" in runs:
        eq = runs["buy_hold"]["equity"]
        contrib = runs["buy_hold"]["contrib"]
        m = compute_metrics(eq)
        u = underwater_vs_contrib(eq, contrib)
        strategies.append(StrategyResult(
            name="Buy & Hold",
            equity=[SeriesPoint(**p) for p in to_points(eq)],
            contrib=[SeriesPoint(**p) for p in to_points(contrib)],
            metrics=Metrics(**m, trades=len(runs["buy_hold"]["markers"])),
            underwater=UnderwaterStats(**u),
            markers=[TradeMarker(**x) for x in to_markers(runs["buy_hold"]["markers"], eq)],
        ))

    if req.toggles.dca and "dca" in runs:
        eq = runs["dca"]["equity"]
        contrib = runs["dca"]["contrib"]
        m = compute_metrics(eq)
        u = underwater_vs_contrib(eq, contrib)
        strategies.append(StrategyResult(
            name=f"DCA ({req.contrib_freq})",
            equity=[SeriesPoint(**p) for p in to_points(eq)],
            contrib=[SeriesPoint(**p) for p in to_points(contrib)],
            metrics=Metrics(**m, trades=len(runs["dca"]["markers"])),
            underwater=UnderwaterStats(**u),
            markers=[TradeMarker(**x) for x in to_markers(runs["dca"]["markers"], eq)],
        ))

    meta = {
        "risk_ticker": req.risk_ticker,
        "start": req.start,
        "end": req.end,
        "note": "Stub strategies active. Next step: wire your full backtester into api/app/services/backtest.py",
    }
    return BacktestResponse(meta=meta, strategies=strategies)
