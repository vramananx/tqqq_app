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
    initial: 10000,
    contrib_amount: 1000,
    contrib_freq: "monthly",
    contrib_day: 11,
    toggles: { buy_hold: true, dca: true }
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
          <div className="text-sm opacity-80">Black UI, mobile-friendly. Next: wire full strategies + optimizer.</div>
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
