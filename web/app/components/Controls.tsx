"use client";

type Props = {
  params: any;
  setParams: (p: any) => void;
  onRun: () => void;
  running: boolean;
};

export default function Controls({ params, setParams, onRun, running }: Props) {
  const set = (k: string, v: any) => setParams({ ...params, [k]: v });

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
          <div className="text-sm opacity-80">Contrib amount</div>
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
