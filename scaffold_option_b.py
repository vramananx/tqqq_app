from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "option_b"

FILES: dict[str, str] = {}

def add(path: str, content: str):
    FILES[path] = content.strip() + "\n"

# ----------------------------
# API
# ----------------------------
add("option_b/api/requirements.txt", """
fastapi
uvicorn[standard]
pydantic
pandas
numpy
yfinance
pyarrow
""")

add("option_b/api/Dockerfile", """
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
""")

add("option_b/api/app/__init__.py", "")

add("option_b/api/app/models.py", r"""
from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field

Freq = Literal["monthly", "quarterly"]

class StrategyToggles(BaseModel):
    buy_hold: bool = True
    dca: bool = True
    sma: bool = True
    ema9: bool = True
    lrs: bool = True
    sig9: bool = True
    sig9_lrs: bool = True

class BacktestRequest(BaseModel):
    risk_ticker: str = "TQQQ"
    start: str = "2010-01-01"
    end: str = "2025-12-30"

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

    sma_len: int = 200
    ema_len: int = 9

    # LRS params (trend gate on signal ticker)
    lrs_ma_type: Literal["SMA", "EMA"] = "SMA"
    lrs_ma_len: int = 200
    buffer_pct: float = 0.0
    confirm_days: int = 3
    position_size: float = 1.0

    # 9-sig target growth per rebalance
    target_g: float = 0.09

    cost_bps: float = 0.0

    toggles: StrategyToggles = Field(default_factory=StrategyToggles)

class SeriesPoint(BaseModel):
    t: str
    v: float

class TradeMarker(BaseModel):
    t: str
    action: str
    qty: float | None = None
    price: float | None = None
    note: str | None = None

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
""")

add("option_b/api/app/services/__init__.py", "")

add("option_b/api/app/services/mapping.py", r"""
from __future__ import annotations

def guess_signal_ticker(risk_ticker: str) -> str:
    t = (risk_ticker or "").upper().strip()
    mapping = {
        "TQQQ": "QQQ", "SQQQ": "QQQ", "QLD": "QQQ",
        "UPRO": "SPY", "SPXL": "SPY", "SPXU": "SPY", "SSO":"SPY", "SDS":"SPY",
        "UDOW":"DIA", "SDOW":"DIA", "DDM":"DIA",
        "TNA":"IWM", "TZA":"IWM", "UWM":"IWM",
        "TMF":"TLT", "TMV":"TLT",
    }
    return mapping.get(t, t)
""")

add("option_b/api/app/services/data.py", r"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

CACHE_DIR = ".cache_prices"

def _cache_path(ticker: str, start: str, end: str) -> str:
    safe = (ticker or "").replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{start}_{end}.parquet")

def fetch_close(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.Series:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker, start, end)

    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            raise ValueError(f"No data for {ticker} {start}..{end}")
        if use_cache:
            df.to_parquet(path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" in df.columns:
        s = df["Close"].copy()
    elif "Adj Close" in df.columns:
        s = df["Adj Close"].copy()
    else:
        s = df.iloc[:, -1].copy()

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()

def align(*series: pd.Series) -> list[pd.Series]:
    idx = None
    for s in series:
        idx = s.index if idx is None else idx.intersection(s.index)
    return [s.reindex(idx) for s in series]
""")

add("option_b/api/app/services/schedule.py", r"""
from __future__ import annotations
import pandas as pd

def schedule_dates(index: pd.DatetimeIndex, freq: str, day: int) -> pd.DatetimeIndex:
    day = max(1, min(int(day), 28))
    idx = pd.DatetimeIndex(index)
    months = pd.PeriodIndex(idx, freq="M").unique()

    dates = []
    for m in months:
        if freq == "quarterly" and m.month not in (1,4,7,10):
            continue
        target = pd.Timestamp(m.year, m.month, day)

        # roll forward to next available trading day
        pos = idx.get_indexer([target], method="bfill")
        if pos[0] == -1:
            continue
        dates.append(idx[pos[0]])

    return pd.DatetimeIndex(sorted(set(dates)))

def maybe_skip_first(dates: pd.DatetimeIndex, skip_first: bool) -> pd.DatetimeIndex:
    if not skip_first or len(dates) == 0:
        return dates
    return pd.DatetimeIndex(dates[1:])
""")

add("option_b/api/app/services/analytics.py", r"""
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return dict(final=np.nan, cagr=np.nan, sharpe=np.nan, vol=np.nan, max_dd=np.nan)

    rets = equity.pct_change().dropna()
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1e-9)

    final = float(equity.iloc[-1])
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)

    vol = float(rets.std(ddof=0) * np.sqrt(252))
    sharpe = float((rets.mean() / (rets.std(ddof=0) + 1e-12)) * np.sqrt(252))

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    return dict(final=final, cagr=cagr, sharpe=sharpe, vol=vol, max_dd=max_dd)

def underwater_vs_contrib(equity: pd.Series, contrib: pd.Series) -> dict:
    equity = equity.dropna()
    contrib = contrib.reindex(equity.index).ffill().fillna(0.0)

    pnl = equity - contrib
    trough_dt = pnl.idxmin()
    trough_val = float(pnl.loc[trough_dt])

    after = pnl.loc[trough_dt:]
    rec = after[after >= 0]
    rec_dt = rec.index[0] if len(rec) else None
    rec_days = int((rec_dt - trough_dt).days) if rec_dt is not None else None

    under = pnl < 0
    streaks = []
    start = None
    for dt, is_under in under.items():
        if is_under and start is None:
            start = dt
        if (not is_under) and start is not None:
            streaks.append((start, dt))
            start = None
    if start is not None:
        streaks.append((start, under.index[-1]))

    longest = None
    if streaks:
        longest = max(streaks, key=lambda ab: (ab[1] - ab[0]).days)
        longest_days = int((longest[1] - longest[0]).days)
    else:
        longest_days = 0

    return dict(
        worst_underwater_dollars=trough_val,
        worst_underwater_date=str(trough_dt.date()),
        breakeven_recovery_date=str(rec_dt.date()) if rec_dt is not None else None,
        breakeven_recovery_days=rec_days,
        longest_underwater_days=longest_days,
        longest_underwater_start=str(longest[0].date()) if longest else None,
        longest_underwater_end=str(longest[1].date()) if longest else None,
    )
""")

# ----------------------------
# Normalized engine: simulator + strategy definitions
# ----------------------------
add("option_b/api/app/engine/__init__.py", "")

add("option_b/api/app/engine/types.py", r"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

Action = Literal["BUY", "SELL", "SWITCH", "CONTRIB", "REBAL"]

@dataclass
class TradeEvent:
    dt: object  # datetime-like
    action: Action
    asset: str
    qty: float
    price: float
    note: str = ""

@dataclass
class PortfolioState:
    cash: float
    risk_shares: float
    def_shares: float
""")

add("option_b/api/app/engine/simulator.py", r"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Literal

from .types import TradeEvent, PortfolioState

@dataclass
class SimConfig:
    initial_cash: float
    out_mode: Literal["cash", "defensive"] = "defensive"
    cost_bps: float = 0.0

class PortfolioSimulator:
    """
    One simulator for all strategies:
    - holdings: cash + risk shares + defensive shares
    - consistent trade logging
    - daily mark-to-market equity + contributions
    """
    def __init__(self, risk_ticker: str, def_ticker: str, risk_px: pd.Series, def_px: pd.Series, cfg: SimConfig):
        self.risk_ticker = risk_ticker
        self.def_ticker = def_ticker
        self.risk_px = risk_px.astype(float)
        self.def_px = def_px.astype(float)
        self.cfg = cfg

        self.state = PortfolioState(cash=float(cfg.initial_cash), risk_shares=0.0, def_shares=0.0)
        self.contrib_total = float(cfg.initial_cash)

        self.events: List[TradeEvent] = []
        self.equity: List[float] = []
        self.contrib: List[float] = []
        self.index: List[pd.Timestamp] = []

    def _apply_cost(self, notional: float) -> float:
        # cost in dollars (bps = 1/100 of 1%)
        return abs(notional) * (self.cfg.cost_bps / 10_000.0)

    def value(self, dt: pd.Timestamp) -> float:
        r = self.state.risk_shares * float(self.risk_px.loc[dt])
        d = self.state.def_shares * float(self.def_px.loc[dt])
        return float(self.state.cash + r + d)

    def contribute(self, dt: pd.Timestamp, amount: float):
        if amount <= 0:
            return
        self.state.cash += float(amount)
        self.contrib_total += float(amount)
        self.events.append(TradeEvent(dt, "CONTRIB", "CASH", 0.0, 1.0, note=f"+{amount:.2f}"))

    def buy_risk(self, dt: pd.Timestamp, dollars: float, note: str = ""):
        px = float(self.risk_px.loc[dt])
        if px <= 0 or dollars <= 0:
            return
        dollars = min(dollars, self.state.cash)
        qty = dollars / px
        notional = qty * px
        cost = self._apply_cost(notional)
        if cost > self.state.cash - notional:
            # ensure cost fits
            notional = max(0.0, self.state.cash / (1 + self.cfg.cost_bps/10_000.0))
            qty = notional / px
            cost = self._apply_cost(notional)
        self.state.cash -= (notional + cost)
        self.state.risk_shares += qty
        self.events.append(TradeEvent(dt, "BUY", self.risk_ticker, qty, px, note=note))

    def sell_risk(self, dt: pd.Timestamp, dollars: float, note: str = ""):
        px = float(self.risk_px.loc[dt])
        if px <= 0 or dollars <= 0:
            return
        cur_val = self.state.risk_shares * px
        dollars = min(dollars, cur_val)
        qty = dollars / px
        notional = qty * px
        cost = self._apply_cost(notional)
        self.state.risk_shares -= qty
        self.state.cash += (notional - cost)
        self.events.append(TradeEvent(dt, "SELL", self.risk_ticker, -qty, px, note=note))

    def buy_def(self, dt: pd.Timestamp, dollars: float, note: str = ""):
        px = float(self.def_px.loc[dt])
        if px <= 0 or dollars <= 0:
            return
        dollars = min(dollars, self.state.cash)
        qty = dollars / px
        notional = qty * px
        cost = self._apply_cost(notional)
        if cost > self.state.cash - notional:
            notional = max(0.0, self.state.cash / (1 + self.cfg.cost_bps/10_000.0))
            qty = notional / px
            cost = self._apply_cost(notional)
        self.state.cash -= (notional + cost)
        self.state.def_shares += qty
        self.events.append(TradeEvent(dt, "BUY", self.def_ticker, qty, px, note=note))

    def sell_def(self, dt: pd.Timestamp, dollars: float, note: str = ""):
        px = float(self.def_px.loc[dt])
        if px <= 0 or dollars <= 0:
            return
        cur_val = self.state.def_shares * px
        dollars = min(dollars, cur_val)
        qty = dollars / px
        notional = qty * px
        cost = self._apply_cost(notional)
        self.state.def_shares -= qty
        self.state.cash += (notional - cost)
        self.events.append(TradeEvent(dt, "SELL", self.def_ticker, -qty, px, note=note))

    def ensure_out_mode(self, dt: pd.Timestamp):
        # If OUT mode = defensive, invest ALL free cash into defensive
        if self.cfg.out_mode == "defensive" and self.state.cash > 0:
            self.buy_def(dt, self.state.cash, note="OUT sweep -> defensive")

    def record_day(self, dt: pd.Timestamp):
        self.index.append(dt)
        self.equity.append(self.value(dt))
        self.contrib.append(self.contrib_total)

    def run(self, index: pd.DatetimeIndex, step_fn):
        for dt in index:
            step_fn(self, dt)
            self.record_day(dt)

    def results(self):
        idx = pd.DatetimeIndex(self.index)
        daily = pd.DataFrame({"Equity": self.equity, "Contributions": self.contrib}, index=idx)
        return daily, self.events
""")

add("option_b/api/app/engine/strategies.py", r"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .simulator import PortfolioSimulator

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(int(n)).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=int(n), adjust=False).mean()

def lrs_in_state(signal_px: pd.Series, ma: pd.Series, buffer_pct: float, confirm_days: int) -> pd.Series:
    """
    Simple trend state:
      IN if signal_px > (1+buffer)*ma for confirm_days
      OUT if signal_px < (1-buffer)*ma for confirm_days
      otherwise keep last state
    """
    idx = signal_px.index.intersection(ma.index)
    sig = signal_px.reindex(idx).astype(float)
    ma = ma.reindex(idx).astype(float)

    up = sig > (1.0 + buffer_pct) * ma
    dn = sig < (1.0 - buffer_pct) * ma

    state = []
    cur = False
    up_count = 0
    dn_count = 0
    for dt in idx:
        if bool(up.loc[dt]):
            up_count += 1
            dn_count = 0
        elif bool(dn.loc[dt]):
            dn_count += 1
            up_count = 0
        else:
            up_count = 0
            dn_count = 0

        if up_count >= confirm_days:
            cur = True
        elif dn_count >= confirm_days:
            cur = False

        state.append(cur)
    return pd.Series(state, index=idx)

def strategy_buy_hold(sim: PortfolioSimulator, dt, did_init: dict):
    if not did_init.get("done"):
        # invest all cash into risk
        sim.buy_risk(dt, sim.state.cash, note="Buy&Hold init")
        did_init["done"] = True

def strategy_dca(sim: PortfolioSimulator, dt, schedule: set[pd.Timestamp], amount: float):
    if dt in schedule:
        sim.contribute(dt, amount)
        sim.buy_risk(dt, amount, note="DCA buy")

def strategy_sma_crossover(sim: PortfolioSimulator, dt, sig: pd.Series, ma: pd.Series, position_size: float):
    # position_size fraction in risk when IN, else OUT
    if dt not in sig.index:
        return
    in_mkt = bool(sig.loc[dt])
    total = sim.value(dt)
    risk_val = sim.state.risk_shares * float(sim.risk_px.loc[dt])
    def_val = sim.state.def_shares * float(sim.def_px.loc[dt])

    target_risk = (position_size * total) if in_mkt else 0.0

    # move to target risk (simple: sell all then buy)
    if risk_val > target_risk + 1e-9:
        sim.sell_risk(dt, risk_val - target_risk, note="SMA rebalance")

    # if OUT, optionally move everything to defensive
    if not in_mkt:
        if sim.cfg.out_mode == "defensive":
            # liquidate risk first, then sweep cash to defensive
            sim.ensure_out_mode(dt)
        return

    # if IN, ensure defensive is not held (optional simplification)
    if def_val > 1e-9:
        sim.sell_def(dt, def_val, note="IN -> sell defensive")

    # buy up to target
    risk_val = sim.state.risk_shares * float(sim.risk_px.loc[dt])
    if risk_val < target_risk - 1e-9:
        sim.buy_risk(dt, target_risk - risk_val, note="SMA buy")

def strategy_lrs_rotation(sim: PortfolioSimulator, dt, in_state: pd.Series, position_size: float):
    if dt not in in_state.index:
        return
    in_mkt = bool(in_state.loc[dt])
    total = sim.value(dt)
    risk_val = sim.state.risk_shares * float(sim.risk_px.loc[dt])
    def_val = sim.state.def_shares * float(sim.def_px.loc[dt])
    target_risk = (position_size * total) if in_mkt else 0.0

    if risk_val > target_risk + 1e-9:
        sim.sell_risk(dt, risk_val - target_risk, note="LRS rebalance")
    if not in_mkt:
        if sim.cfg.out_mode == "defensive":
            sim.ensure_out_mode(dt)
        return

    if def_val > 1e-9:
        sim.sell_def(dt, def_val, note="IN -> sell defensive")

    risk_val = sim.state.risk_shares * float(sim.risk_px.loc[dt])
    if risk_val < target_risk - 1e-9:
        sim.buy_risk(dt, target_risk - risk_val, note="LRS buy")
""")

# ----------------------------
# API main
# ----------------------------
add("option_b/api/app/main.py", r"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from .models import BacktestRequest, BacktestResponse, StrategyResult, SeriesPoint, Metrics, UnderwaterStats, TradeMarker
from .services.data import fetch_close, align
from .services.mapping import guess_signal_ticker
from .services.schedule import schedule_dates, maybe_skip_first
from .services.analytics import compute_metrics, underwater_vs_contrib

from .engine.simulator import PortfolioSimulator, SimConfig
from .engine.strategies import (
    sma, ema, lrs_in_state,
    strategy_buy_hold,
    strategy_dca,
    strategy_sma_crossover,
    strategy_lrs_rotation,
)

app = FastAPI(title="Strategy Lab API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def to_points(s: pd.Series):
    return [SeriesPoint(t=str(d.date()), v=float(v)) for d, v in s.items()]

def to_markers(events, equity: pd.Series):
    out = []
    for e in events:
        dt = pd.Timestamp(e.dt)
        if dt not in equity.index:
            # snap to nearest
            dt = equity.index[equity.index.get_indexer([dt], method="nearest")[0]]
        out.append(TradeMarker(
            t=str(dt.date()),
            action=e.action,
            qty=float(e.qty) if e.qty is not None else None,
            price=float(e.price) if e.price is not None else None,
            note=e.note
        ))
    return out

@app.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest):
    sig_ticker = guess_signal_ticker(req.risk_ticker) if req.signal_mode == "auto" else (req.signal_ticker_custom or req.risk_ticker)

    risk = fetch_close(req.risk_ticker, req.start, req.end, use_cache=True)
    defs = fetch_close(req.defensive_ticker, req.start, req.end, use_cache=True)
    sigp = fetch_close(sig_ticker, req.start, req.end, use_cache=True)

    risk, defs, sigp = align(risk, defs, sigp)
    idx = risk.index
    if len(idx) < 120:
        raise ValueError("Not enough data. Increase date range.")

    dates = schedule_dates(idx, req.contrib_freq, req.contrib_day)
    dates = maybe_skip_first(dates, req.skip_first_period)
    date_set = set(pd.DatetimeIndex(dates).intersection(idx))

    # trend MA on signal ticker
    if req.lrs_ma_type == "SMA":
        ma = sma(sigp, req.lrs_ma_len)
    else:
        ma = ema(sigp, req.lrs_ma_len)

    in_state = lrs_in_state(sigp, ma, req.buffer_pct, req.confirm_days)

    results = []
    meta = {
        "risk_ticker": req.risk_ticker,
        "signal_ticker": sig_ticker,
        "defensive_ticker": req.defensive_ticker,
        "start": req.start,
        "end": req.end,
        "out_mode": req.out_mode,
        "note": "Normalized simulator v0 (core strategies wired: Buy&Hold, DCA, SMA crossover, LRS rotation). Next: wire 9-SIG variants.",
    }

    def build_strategy(name: str, step_fn):
        sim = PortfolioSimulator(
            req.risk_ticker, req.defensive_ticker, risk, defs,
            SimConfig(initial_cash=req.initial, out_mode=req.out_mode, cost_bps=req.cost_bps)
        )

        did_init = {"done": False}

        def step(simulator: PortfolioSimulator, dt):
            # contributions (deposit on schedule)
            if req.include_contribs and dt in date_set:
                simulator.contribute(dt, req.contrib_amount)

            # strategy step
            step_fn(simulator, dt, did_init)

        sim.run(idx, step)
        daily, events = sim.results()

        m = compute_metrics(daily["Equity"])
        u = underwater_vs_contrib(daily["Equity"], daily["Contributions"])

        results.append(StrategyResult(
            name=name,
            equity=to_points(daily["Equity"]),
            contrib=to_points(daily["Contributions"]),
            metrics=Metrics(**m, trades=len([e for e in events if e.action in ("BUY","SELL","REBAL","SWITCH")])),
            underwater=UnderwaterStats(**u),
            markers=to_markers(events, daily["Equity"])
        ))

    if req.toggles.buy_hold:
        def step_bh(sim, dt, did_init):
            if not did_init["done"]:
                strategy_buy_hold(sim, dt, did_init)
        build_strategy("Buy & Hold", step_bh)

    if req.toggles.dca:
        def step_dca(sim, dt, did_init):
            # only buys at deposit dates
            if req.include_contribs and dt in date_set:
                strategy_dca(sim, dt, date_set, req.contrib_amount)
        build_strategy(f"DCA ({req.contrib_freq})", step_dca)

    if req.toggles.sma:
        sma_ma = sma(sigp, req.sma_len)
        sma_sig = (sigp > sma_ma).reindex(idx).fillna(False)
        def step_sma(sim, dt, did_init):
            strategy_sma_crossover(sim, dt, sma_sig, sma_ma, req.position_size)
        build_strategy(f"SMA({req.sma_len}) on {sig_ticker}", step_sma)

    if req.toggles.lrs:
        def step_lrs(sim, dt, did_init):
            strategy_lrs_rotation(sim, dt, in_state, req.position_size)
        build_strategy(f"LRS({req.lrs_ma_type}{req.lrs_ma_len}) on {sig_ticker}", step_lrs)

    return BacktestResponse(meta=meta, strategies=results)
""")

# ----------------------------
# WEB (Next.js)
# ----------------------------
add("option_b/web/package.json", """
{
  "name": "strategy-lab-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start -p 3000"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "plotly.js": "2.35.2",
    "react-plotly.js": "2.6.0"
  },
  "devDependencies": {
    "autoprefixer": "10.4.19",
    "postcss": "8.4.39",
    "tailwindcss": "3.4.10",
    "typescript": "5.5.4"
  }
}
""")

add("option_b/web/next.config.mjs", """
/** @type {import('next').NextConfig} */
const nextConfig = { reactStrictMode: true };
export default nextConfig;
""")

add("option_b/web/postcss.config.mjs", """
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} }
};
""")

add("option_b/web/tailwind.config.ts", """
import type { Config } from "tailwindcss";
export default {
  content: ["./app/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: []
} satisfies Config;
""")

add("option_b/web/tsconfig.json", """
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "jsx": "preserve",
    "types": ["node"]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"]
}
""")

add("option_b/web/app/globals.css", """
@tailwind base;
@tailwind components;
@tailwind utilities;

:root { color-scheme: dark; }
html, body { height: 100%; }
""")

add("option_b/web/app/types.ts", """
export type SeriesPoint = { t: string; v: number };
export type TradeMarker = { t: string; action: string; qty?: number | null; price?: number | null; note?: string | null; };

export type UnderwaterStats = {
  worst_underwater_dollars: number;
  worst_underwater_date: string;
  breakeven_recovery_date: string | null;
  breakeven_recovery_days: number | null;
  longest_underwater_days: number;
  longest_underwater_start: string | null;
  longest_underwater_end: string | null;
};

export type Metrics = { final: number; cagr: number; sharpe: number; vol: number; max_dd: number; trades: number; };

export type StrategyResult = {
  name: string;
  equity: SeriesPoint[];
  contrib: SeriesPoint[];
  metrics: Metrics;
  underwater: UnderwaterStats;
  markers: TradeMarker[];
};

export type BacktestResponse = { meta: Record<string, any>; strategies: StrategyResult[]; };
""")

add("option_b/web/app/api.ts", """
import { BacktestResponse } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function runBacktest(payload: any): Promise<BacktestResponse> {
  const res = await fetch(`${API_BASE}/backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
""")

add("option_b/web/app/components/ThemeToggle.tsx", """
"use client";
import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const saved = (localStorage.getItem("theme") as any) || "dark";
    setTheme(saved);
  }, []);

  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  return (
    <button
      className="px-3 py-2 rounded-xl border border-white/15 bg-white/5 hover:bg-white/10"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      Theme: {theme === "dark" ? "Black" : "Light"}
    </button>
  );
}
""")

add("option_b/web/app/components/Controls.tsx", """
"use client";
type Props = { params: any; setParams: (p:any)=>void; onRun: ()=>void; running: boolean; };

export default function Controls({ params, setParams, onRun, running }: Props) {
  const set = (k: string, v: any) => setParams({ ...params, [k]: v });
  const setT = (k: string, v: any) => setParams({ ...params, toggles: { ...params.toggles, [k]: v } });

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <label className="space-y-1">
          <div className="text-sm opacity-80">Risk ticker</div>
          <input className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.risk_ticker} onChange={e => set("risk_ticker", e.target.value)} />
        </label>
        <label className="space-y-1">
          <div className="text-sm opacity-80">Defensive ticker</div>
          <input className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.defensive_ticker} onChange={e => set("defensive_ticker", e.target.value)} />
        </label>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="space-y-1">
          <div className="text-sm opacity-80">Start</div>
          <input className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.start} onChange={e => set("start", e.target.value)} />
        </label>
        <label className="space-y-1">
          <div className="text-sm opacity-80">End</div>
          <input className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.end} onChange={e => set("end", e.target.value)} />
        </label>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <label className="space-y-1">
          <div className="text-sm opacity-80">Initial</div>
          <input type="number" className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.initial} onChange={e => set("initial", Number(e.target.value))} />
        </label>
        <label className="space-y-1">
          <div className="text-sm opacity-80">Contrib</div>
          <input type="number" className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.contrib_amount} onChange={e => set("contrib_amount", Number(e.target.value))} />
        </label>
        <label className="space-y-1">
          <div className="text-sm opacity-80">Freq</div>
          <select className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.contrib_freq} onChange={e => set("contrib_freq", e.target.value)}>
            <option value="monthly">monthly</option>
            <option value="quarterly">quarterly</option>
          </select>
        </label>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="space-y-1">
          <div className="text-sm opacity-80">Contrib day (1-28)</div>
          <input type="number" className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.contrib_day} onChange={e => set("contrib_day", Number(e.target.value))} />
        </label>
        <label className="space-y-1">
          <div className="text-sm opacity-80">OUT mode</div>
          <select className="w-full rounded-xl bg-black/40 border border-white/10 p-2"
            value={params.out_mode} onChange={e => set("out_mode", e.target.value)}>
            <option value="defensive">defensive</option>
            <option value="cash">cash</option>
          </select>
        </label>
      </div>

      <div className="rounded-xl border border-white/10 p-3 space-y-2">
        <div className="text-sm font-semibold">Strategies</div>
        <label className="flex items-center justify-between text-sm">
          <span>Buy & Hold</span>
          <input type="checkbox" checked={params.toggles.buy_hold} onChange={e => setT("buy_hold", e.target.checked)} />
        </label>
        <label className="flex items-center justify-between text-sm">
          <span>DCA</span>
          <input type="checkbox" checked={params.toggles.dca} onChange={e => setT("dca", e.target.checked)} />
        </label>
        <label className="flex items-center justify-between text-sm">
          <span>SMA crossover</span>
          <input type="checkbox" checked={params.toggles.sma} onChange={e => setT("sma", e.target.checked)} />
        </label>
        <label className="flex items-center justify-between text-sm">
          <span>LRS rotation</span>
          <input type="checkbox" checked={params.toggles.lrs} onChange={e => setT("lrs", e.target.checked)} />
        </label>
      </div>

      <button
        className="w-full py-3 rounded-2xl bg-white text-black font-semibold hover:opacity-90 disabled:opacity-50"
        onClick={onRun}
        disabled={running}
      >
        {running ? "Running..." : "Run backtest"}
      </button>
    </div>
  );
}
""")

add("option_b/web/app/components/Charts.tsx", """
"use client";
import dynamic from "next/dynamic";
import { StrategyResult } from "../types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function Charts({ strategies }: { strategies: StrategyResult[] }) {
  const lineTraces = strategies.map(s => ({
    x: s.equity.map(p => p.t),
    y: s.equity.map(p => p.v),
    type: "scatter",
    mode: "lines",
    name: s.name,
    hovertemplate: "%{x}<br>Value=$%{y:,.0f}<extra></extra>"
  }));

  const markerTraces = strategies.flatMap(s => {
    const buys = s.markers.filter(m => /BUY|IN/i.test(m.action));
    const sells = s.markers.filter(m => /SELL|OUT|EXIT/i.test(m.action));
    const out: any[] = [];

    if (buys.length) out.push({
      x: buys.map(m => m.t),
      y: buys.map(m => null),   // equity not included in v0; will add in next iteration
      type: "scatter",
      mode: "markers",
      name: `${s.name} BUY`,
      marker: { symbol: "circle", size: 7 },
      hovertemplate: "%{x}<br>BUY<extra></extra>"
    });

    if (sells.length) out.push({
      x: sells.map(m => m.t),
      y: sells.map(m => null),
      type: "scatter",
      mode: "markers",
      name: `${s.name} SELL`,
      marker: { symbol: "x", size: 8 },
      hovertemplate: "%{x}<br>SELL<extra></extra>"
    });

    return out;
  });

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
      <div className="text-sm opacity-80 px-2 py-1">Equity (black UI)</div>
      <Plot
        data={[...lineTraces, ...markerTraces]}
        layout={{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "white" },
          height: 520,
          hovermode: "x unified",
          margin: { l: 40, r: 20, t: 30, b: 40 },
          legend: { orientation: "h" }
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
""")

add("option_b/web/app/components/MetricsCards.tsx", """
"use client";
import { StrategyResult } from "../types";

function money(x: number) {
  const ax = Math.abs(x);
  if (ax >= 1_000_000) return `$${(x/1_000_000).toFixed(2)}M`;
  if (ax >= 1_000) return `$${(x/1_000).toFixed(1)}K`;
  return `$${x.toFixed(0)}`;
}
function pct(x: number) { return `${(x*100).toFixed(1)}%`; }

export default function MetricsCards({ strategies }: { strategies: StrategyResult[] }) {
  return (
    <div className="grid md:grid-cols-3 gap-3">
      {strategies.map(s => (
        <div key={s.name} className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="text-lg font-semibold">{s.name}</div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
            <div className="opacity-80">Final</div><div className="text-right">{money(s.metrics.final)}</div>
            <div className="opacity-80">CAGR</div><div className="text-right">{pct(s.metrics.cagr)}</div>
            <div className="opacity-80">Max DD</div><div className="text-right">{pct(s.metrics.max_dd)}</div>
            <div className="opacity-80">Trades</div><div className="text-right">{s.metrics.trades}</div>
          </div>
          <div className="mt-3 text-xs opacity-80">
            Worst underwater: {money(s.underwater.worst_underwater_dollars)} on {s.underwater.worst_underwater_date}
            {s.underwater.breakeven_recovery_days != null
              ? ` • Recovered in ${s.underwater.breakeven_recovery_days} days`
              : " • Not yet recovered"}
          </div>
        </div>
      ))}
    </div>
  );
}
""")

add("option_b/web/app/layout.tsx", """
import "./globals.css";
export const metadata = { title: "Strategy Lab" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-black text-white">
        {children}
      </body>
    </html>
  );
}
""")

add("option_b/web/app/page.tsx", """
"use client";
import { useState } from "react";
import ThemeToggle from "./components/ThemeToggle";
import Controls from "./components/Controls";
import Charts from "./components/Charts";
import MetricsCards from "./components/MetricsCards";
import { runBacktest } from "./api";
import { BacktestResponse } from "./types";

export default function Page() {
  const [params, setParams] = useState<any>({
    risk_ticker: "TQQQ",
    defensive_ticker: "BIL",
    start: "2010-01-01",
    end: "2025-12-30",
    out_mode: "defensive",
    initial: 10000,
    include_contribs: true,
    contrib_amount: 1000,
    contrib_freq: "monthly",
    contrib_day: 11,
    skip_first_period: true,
    cost_bps: 0.0,
    toggles: { buy_hold: true, dca: true, sma: true, lrs: true }
  });

  const [running, setRunning] = useState(false);
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function onRun() {
    setRunning(true);
    setErr(null);
    try {
      const res = await runBacktest(params);
      setData(res);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="max-w-6xl mx-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-2xl font-semibold">Strategy Lab</div>
          <div className="text-sm opacity-80">Option B skeleton (Next.js + FastAPI) • Black UI + theme toggle</div>
        </div>
        <ThemeToggle />
      </div>

      <div className="grid md:grid-cols-[380px_1fr] gap-4">
        <Controls params={params} setParams={setParams} onRun={onRun} running={running} />
        <div className="space-y-4">
          {err && <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-3 text-sm">{err}</div>}
          {data?.strategies?.length ? (
            <>
              <MetricsCards strategies={data.strategies} />
              <Charts strategies={data.strategies} />
            </>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 opacity-80">
              Run a backtest to see results.
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
""")

add("option_b/web/Dockerfile", """
FROM node:20-alpine
WORKDIR /app
COPY package.json /app/package.json
RUN npm install
COPY . /app
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start"]
""")

# ----------------------------
# Render notes
# ----------------------------
add("option_b/render/README_RENDER.md", """
## Deploy Option B on Render

Deploy TWO services:
1) API (FastAPI) from option_b/api (port 8000)
2) WEB (Next.js) from option_b/web (port 3000)

### API service
- Environment: Docker
- Root directory: option_b/api
- Port: 8000

### WEB service
- Environment: Docker
- Root directory: option_b/web
- Port: 3000
- Env var:
  NEXT_PUBLIC_API_BASE = https://<your-api-service>.onrender.com
""")

def main():
    for rel, content in FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"Created {len(FILES)} files under {BASE}")

if __name__ == "__main__":
    main()

