from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

Freq = Literal["monthly", "quarterly"]
Theme = Literal["dark", "light"]

class StrategyToggles(BaseModel):
    buy_hold: bool = True
    sma: bool = True
    ema9: bool = True
    dca: bool = True
    lrs: bool = True
    sig9_lrs: bool = True

class BacktestRequest(BaseModel):
    risk_ticker: str = "TQQQ"
    start: str = "2010-01-01"
    end: str = "2025-12-30"

    # mapping / signal reference
    signal_mode: Literal["auto", "custom"] = "auto"
    signal_ticker_custom: Optional[str] = None

    defensive_ticker: str = "BIL"
    out_mode: Literal["cash", "defensive"] = "defensive"

    initial: float = 10000.0
    include_contribs: bool = True
    contrib_amount: float = 1000.0
    contrib_freq: Freq = "monthly"
    contrib_day: int = 11
    skip_first_period: bool = True

    # SMA/EMA params
    sma_len: int = 200
    ema_len: int = 9

    # LRS params
    lrs_ma_type: Literal["SMA", "EMA"] = "SMA"
    lrs_ma_len: int = 200
    buffer_pct: float = 0.0
    confirm_days: int = 3
    position_size: float = 1.0

    # 9-sig param
    target_g: float = 0.09

    # costs
    cost_bps: float = 0.0

    toggles: StrategyToggles = Field(default_factory=StrategyToggles)

class SeriesPoint(BaseModel):
    t: str
    v: float

class TradeMarker(BaseModel):
    t: str
    action: str
    price: Optional[float] = None
    equity: Optional[float] = None
    note: Optional[str] = None

class UnderwaterStats(BaseModel):
    worst_underwater_dollars: float
    worst_underwater_date: str
    breakeven_recovery_date: Optional[str]
    breakeven_recovery_days: Optional[int]
    longest_underwater_days: int
    longest_underwater_start: Optional[str]
    longest_underwater_end: Optional[str]

class Metrics(BaseModel):
    final: float
    cagr: float
    sharpe: float
    vol: float
    max_dd: float
    trades: int

class StrategyResult(BaseModel):
    name: str
    equity: List[SeriesPoint]
    contrib: List[SeriesPoint]
    metrics: Metrics
    underwater: UnderwaterStats
    markers: List[TradeMarker]

class BacktestResponse(BaseModel):
    meta: Dict[str, Any]
    strategies: List[StrategyResult]
