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
