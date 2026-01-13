export type SeriesPoint = { t: string; v: number };

export type TradeMarker = {
  t: string;
  action: string;
  price?: number | null;
  equity?: number | null;
  note?: string | null;
};

export type UnderwaterStats = {
  worst_underwater_dollars: number;
  worst_underwater_date: string;
  breakeven_recovery_date: string | null;
  breakeven_recovery_days: number | null;
  longest_underwater_days: number;
  longest_underwater_start: string | null;
  longest_underwater_end: string | null;
};

export type Metrics = {
  final: number;
  cagr: number;
  sharpe: number;
  vol: number;
  max_dd: number;
  trades: number;
};

export type StrategyResult = {
  name: string;
  equity: SeriesPoint[];
  contrib: SeriesPoint[];
  metrics: Metrics;
  underwater: UnderwaterStats;
  markers: TradeMarker[];
};

export type BacktestResponse = {
  meta: Record<string, any>;
  strategies: StrategyResult[];
};
