from __future__ import annotations
import pandas as pd

def rolling_window_outperformance(
    equity_by_strategy: dict[str, pd.Series],
    benchmark_name: str,
    window_years: int = 3,
) -> pd.DataFrame:
    """
    Compute rolling-window CAGR per strategy and compare to benchmark.
    Output columns:
      start, end, strat, cagr, bench_cagr, excess_cagr
    """
    bench = equity_by_strategy[benchmark_name].dropna()
    idx = bench.index
    window_days = int(window_years * 365.25)

    rows = []
    for start_dt in idx:
        end_dt = start_dt + pd.Timedelta(days=window_days)
        if end_dt > idx[-1]:
            break
        # align end to nearest available date
        end_dt = idx[idx.get_indexer([end_dt], method="nearest")[0]]

        bench_seg = bench.loc[start_dt:end_dt]
        if len(bench_seg) < 2:
            continue

        b_cagr = (bench_seg.iloc[-1] / bench_seg.iloc[0]) ** (1 / window_years) - 1

        for name, eq in equity_by_strategy.items():
            seg = eq.reindex(bench_seg.index).dropna()
            if len(seg) < 2:
                continue
            c = (seg.iloc[-1] / seg.iloc[0]) ** (1 / window_years) - 1
            rows.append({
                "start": start_dt.date(),
                "end": end_dt.date(),
                "strategy": name,
                "cagr": float(c),
                "bench_cagr": float(b_cagr),
                "excess_cagr": float(c - b_cagr),
            })

    return pd.DataFrame(rows)

def summarize_outperformance(roll_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize by strategy:
      - pct_windows_beating_bench
      - median_excess_cagr
      - worst_excess_cagr
    """
    if len(roll_df) == 0:
        return pd.DataFrame()

    g = roll_df.groupby("strategy")["excess_cagr"]
    out = pd.DataFrame({
        "pct_windows_beating_bench": g.apply(lambda s: float((s > 0).mean())),
        "median_excess_cagr": g.median(),
        "worst_excess_cagr": g.min(),
    }).reset_index()
    return out
