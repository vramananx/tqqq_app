"use client";

import dynamic from "next/dynamic";
import { StrategyResult } from "../types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function Charts({ strategies }: { strategies: StrategyResult[] }) {
  const traces = strategies.map(s => ({
    x: s.equity.map(p => p.t),
    y: s.equity.map(p => p.v),
    type: "scatter",
    mode: "lines",
    name: s.name,
    hovertemplate: "%{x}<br>Value=$%{y:,.0f}<extra></extra>"
  }));

  // marker traces per strategy
  const markerTraces = strategies.flatMap(s => {
    const buys = s.markers.filter(m => /BUY|IN/i.test(m.action));
    const sells = s.markers.filter(m => /SELL|OUT|EXIT/i.test(m.action));
    const out: any[] = [];

    if (buys.length) {
      out.push({
        x: buys.map(m => m.t),
        y: buys.map(m => m.equity ?? null),
        type: "scatter",
        mode: "markers",
        name: `${s.name} BUY/IN`,
        marker: { symbol: "circle", size: 7 },
        hovertemplate: "%{x}<br>BUY/IN<br>Equity=$%{y:,.0f}<extra></extra>"
      });
    }
    if (sells.length) {
      out.push({
        x: sells.map(m => m.t),
        y: sells.map(m => m.equity ?? null),
        type: "scatter",
        mode: "markers",
        name: `${s.name} SELL/OUT`,
        marker: { symbol: "x", size: 8 },
        hovertemplate: "%{x}<br>SELL/OUT<br>Equity=$%{y:,.0f}<extra></extra>"
      });
    }
    return out;
  });

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
      <div className="text-sm opacity-80 px-2 py-1">Equity + Markers</div>
      <Plot
        data={[...traces, ...markerTraces]}
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
