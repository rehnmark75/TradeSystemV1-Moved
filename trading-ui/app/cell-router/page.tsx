"use client";

import { useEffect, useMemo, useState } from "react";

type EdgeMapRow = {
  scanner_name: string;
  trend_state: string | null;
  vol_regime: string | null;
  n: number | null;
  pf: number | null;
  win_rate: number | null;
  avg_pnl_pct: number | null;
  calendar_days: number | null;
  verdict: string | null;
  computed_at: string | null;
};

type VerdictRow = {
  verdict: string;
  n: number;
  wins: number;
  win_rate: number | null;
  pf: number | "inf" | null;
  avg_pnl_pct: number | null;
};

type WouldDropSignal = {
  ticker: string;
  scanner_name: string;
  cell_key: string;
  cell_pf: number | null;
  cell_n: number | null;
  signal_timestamp: string | null;
  status: string | null;
  realized_pnl_pct: number | null;
};

type WouldDropByScanner = {
  scanner_name: string;
  count: number;
  closed: number;
  sum_realized_pnl_pct: number;
  avg_realized_pnl_pct: number | null;
};

type Payload = {
  meta: {
    window_days: number;
    monitor_only_scanners: string[];
    shadow_mode: boolean;
    generated_at: string;
  };
  edgeMap: EdgeMapRow[];
  verdictValidation: VerdictRow[];
  wouldDropImpact: {
    total: number;
    byScanner: WouldDropByScanner[];
    signals: WouldDropSignal[];
  };
  error?: string;
};

const VERDICT_ORDER = ["trade", "monitor", "block", "insufficient"];

const VERDICT_COLOR: Record<string, string> = {
  trade: "#16a34a",
  monitor: "#d97706",
  block: "#dc2626",
  insufficient: "#64748b",
  unmapped: "#475569",
};

const TREND_STATES = ["range", "mid", "trend"];
const VOL_REGIMES = ["low", "normal", "high"];

function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

function fmtPf(v: number | "inf" | null | undefined): string {
  if (v === "inf") return "∞";
  if (v == null || Number.isNaN(v as number)) return "—";
  return (v as number).toFixed(2);
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtSignedPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(2)}%`;
}

function fmtDateTime(v: string | null): string {
  if (!v) return "—";
  const d = new Date(v);
  if (Number.isNaN(d.valueOf())) return v;
  return d.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const cellStyle: React.CSSProperties = {
  padding: "6px 10px",
  borderBottom: "1px solid rgba(148,163,184,0.15)",
  textAlign: "right",
  fontVariantNumeric: "tabular-nums",
  whiteSpace: "nowrap",
};
const headStyle: React.CSSProperties = {
  ...cellStyle,
  textAlign: "right",
  fontWeight: 600,
  color: "#94a3b8",
  borderBottom: "1px solid rgba(148,163,184,0.3)",
};
const leftCell: React.CSSProperties = { ...cellStyle, textAlign: "left" };
const leftHead: React.CSSProperties = { ...headStyle, textAlign: "left" };

function VerdictChip({ verdict }: { verdict: string | null }) {
  const v = verdict ?? "unmapped";
  const color = VERDICT_COLOR[v] ?? VERDICT_COLOR.unmapped;
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 8px",
        borderRadius: 6,
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 0.3,
        textTransform: "uppercase",
        color: "#fff",
        background: color,
      }}
    >
      {v}
    </span>
  );
}

export default function CellRouterPage() {
  const [windowDays, setWindowDays] = useState(120);
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setErr(null);
      try {
        const res = await fetch(`/trading/api/cell-router?windowDays=${windowDays}`, {
          cache: "no-store",
        });
        const json: Payload = await res.json();
        if (cancelled) return;
        if (json.error) {
          setErr(json.error);
          setData(null);
        } else {
          setData(json);
        }
      } catch {
        if (!cancelled) setErr("Failed to load cell-router data");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [windowDays]);

  const verdictByName = useMemo(() => {
    const m = new Map<string, VerdictRow>();
    (data?.verdictValidation ?? []).forEach((r) => m.set(r.verdict, r));
    return m;
  }, [data]);

  // Is the router predictive? block PF should be below trade PF.
  const tradePf = verdictByName.get("trade")?.pf;
  const blockPf = verdictByName.get("block")?.pf;
  const routerPredictive =
    typeof tradePf === "number" && typeof blockPf === "number" ? blockPf < tradePf : null;

  // Edge map keyed for grid rendering
  const edgeByKey = useMemo(() => {
    const m = new Map<string, EdgeMapRow>();
    (data?.edgeMap ?? []).forEach((r) => {
      m.set(`${r.scanner_name}|${r.trend_state}|${r.vol_regime}`, r);
    });
    return m;
  }, [data]);

  const scanners = useMemo(() => {
    const set = new Set<string>();
    (data?.edgeMap ?? []).forEach((r) => set.add(r.scanner_name));
    return [...set].sort();
  }, [data]);

  return (
    <div className="page">
      <section className="ops-panel" style={{ marginBottom: 16 }}>
        <div className="ops-panel-head">
          <div>
            <div className="ops-panel-kicker">Shadow-Mode Analytics</div>
            <h2 style={{ margin: "4px 0" }}>Cell-Router Monitor</h2>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label style={{ fontSize: 13, color: "#94a3b8" }}>Window</label>
            <select
              value={windowDays}
              onChange={(e) => setWindowDays(Number(e.target.value))}
              style={{
                background: "rgba(15,23,42,0.6)",
                color: "#e2e8f0",
                border: "1px solid rgba(148,163,184,0.3)",
                borderRadius: 6,
                padding: "4px 8px",
              }}
            >
              {[30, 60, 90, 120, 180, 365].map((d) => (
                <option key={d} value={d}>
                  {d} days
                </option>
              ))}
            </select>
          </div>
        </div>
        <p style={{ color: "#94a3b8", fontSize: 13, lineHeight: 1.6, maxWidth: 900 }}>
          The per-stock regime router runs in <strong>SHADOW MODE</strong> — it logs the routing
          decision it <em>would</em> make for each signal but nothing is actually dropped or held
          yet. Every signal is tagged with a market-character cell (<code>trend_state</code> ×{" "}
          <code>vol_regime</code>) and mapped to a learned verdict from{" "}
          <code>scanner_cell_edge</code>:{" "}
          <strong style={{ color: VERDICT_COLOR.trade }}>trade</strong> = cell has shown a positive
          edge,{" "}
          <strong style={{ color: VERDICT_COLOR.monitor }}>monitor</strong> = marginal / watch,{" "}
          <strong style={{ color: VERDICT_COLOR.block }}>block</strong> = cell realized negative edge
          (would-drop if enforced),{" "}
          <strong style={{ color: VERDICT_COLOR.insufficient }}>insufficient</strong> = too few
          samples to judge. This page checks whether those verdicts actually predict realized
          outcomes before we flip enforcement on.
        </p>
      </section>

      {loading && <div className="ops-loading">Loading cell-router data…</div>}
      {err && (
        <div className="ops-panel" style={{ marginBottom: 16, color: "#fca5a5" }}>
          Error: {err}
        </div>
      )}

      {data && (
        <>
          {/* ---------- VERDICT VALIDATION (headline) ---------- */}
          <section className="ops-panel" style={{ marginBottom: 16 }}>
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Headline · Is the router predictive?</div>
                <h2 style={{ margin: "4px 0" }}>Verdict Validation</h2>
              </div>
              {routerPredictive != null && (
                <span
                  style={{
                    padding: "4px 12px",
                    borderRadius: 8,
                    fontWeight: 700,
                    fontSize: 13,
                    color: "#fff",
                    background: routerPredictive ? VERDICT_COLOR.trade : VERDICT_COLOR.block,
                  }}
                >
                  {routerPredictive
                    ? "PREDICTIVE ✓ block PF < trade PF"
                    : "NOT PREDICTIVE ✗ block PF ≥ trade PF"}
                </span>
              )}
            </div>
            <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 12px" }}>
              Realized outcomes of closed signals over the last {data.meta.window_days} days, grouped
              by the verdict of the cell they fell in. If the router works, <strong>trade</strong>{" "}
              cells realize the best PF and <strong>block</strong> cells the worst.
            </p>
            {data.verdictValidation.length === 0 ? (
              <div className="ops-empty">No closed signals with a mapped cell in this window.</div>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={leftHead}>Verdict</th>
                    <th style={headStyle}>n</th>
                    <th style={headStyle}>Wins</th>
                    <th style={headStyle}>Win Rate</th>
                    <th style={headStyle}>PF</th>
                    <th style={headStyle}>Avg PnL</th>
                  </tr>
                </thead>
                <tbody>
                  {[...data.verdictValidation]
                    .sort(
                      (a, b) =>
                        (VERDICT_ORDER.indexOf(a.verdict) + 100 * (VERDICT_ORDER.indexOf(a.verdict) < 0 ? 1 : 0)) -
                        (VERDICT_ORDER.indexOf(b.verdict) + 100 * (VERDICT_ORDER.indexOf(b.verdict) < 0 ? 1 : 0))
                    )
                    .map((r) => {
                      const pfNum = typeof r.pf === "number" ? r.pf : r.pf === "inf" ? Infinity : null;
                      return (
                        <tr key={r.verdict}>
                          <td style={leftCell}>
                            <VerdictChip verdict={r.verdict} />
                          </td>
                          <td style={cellStyle}>{r.n}</td>
                          <td style={cellStyle}>{r.wins}</td>
                          <td style={cellStyle}>{fmtPct(r.win_rate)}</td>
                          <td
                            style={{
                              ...cellStyle,
                              fontWeight: 700,
                              color:
                                pfNum == null
                                  ? "#e2e8f0"
                                  : pfNum >= 1
                                    ? VERDICT_COLOR.trade
                                    : VERDICT_COLOR.block,
                            }}
                          >
                            {fmtPf(r.pf)}
                          </td>
                          <td
                            style={{
                              ...cellStyle,
                              color:
                                r.avg_pnl_pct == null
                                  ? "#e2e8f0"
                                  : r.avg_pnl_pct >= 0
                                    ? VERDICT_COLOR.trade
                                    : VERDICT_COLOR.block,
                            }}
                          >
                            {fmtSignedPct(r.avg_pnl_pct)}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            )}
          </section>

          {/* ---------- EDGE MAP GRID ---------- */}
          <section className="ops-panel" style={{ marginBottom: 16 }}>
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Learned Edge Map · 2-axis grid</div>
                <h2 style={{ margin: "4px 0" }}>Scanner × Cell Verdicts</h2>
              </div>
            </div>
            <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 12px" }}>
              Each cell shows verdict + PF/n for the (scanner × trend_state × vol_regime)
              combination. Columns are trend_state (range / mid / trend) sub-divided by vol_regime
              (low / normal / high).
            </p>
            {scanners.length === 0 ? (
              <div className="ops-empty">No 2-axis edge-map rows found.</div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th style={{ ...leftHead, position: "sticky", left: 0 }}>Scanner</th>
                      {TREND_STATES.map((ts) => (
                        <th
                          key={ts}
                          colSpan={VOL_REGIMES.length}
                          style={{ ...headStyle, textAlign: "center", textTransform: "uppercase" }}
                        >
                          {ts}
                        </th>
                      ))}
                    </tr>
                    <tr>
                      <th style={{ ...leftHead, position: "sticky", left: 0 }} />
                      {TREND_STATES.flatMap((ts) =>
                        VOL_REGIMES.map((vr) => (
                          <th key={`${ts}-${vr}`} style={{ ...headStyle, textAlign: "center", fontSize: 10 }}>
                            {vr}
                          </th>
                        ))
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {scanners.map((sc) => (
                      <tr key={sc}>
                        <td style={{ ...leftCell, fontWeight: 600, position: "sticky", left: 0, background: "rgba(15,23,42,0.85)" }}>
                          {sc}
                        </td>
                        {TREND_STATES.flatMap((ts) =>
                          VOL_REGIMES.map((vr) => {
                            const cell = edgeByKey.get(`${sc}|${ts}|${vr}`);
                            const color = cell?.verdict
                              ? VERDICT_COLOR[cell.verdict] ?? VERDICT_COLOR.unmapped
                              : "transparent";
                            return (
                              <td
                                key={`${sc}-${ts}-${vr}`}
                                title={
                                  cell
                                    ? `${cell.verdict ?? "?"} · PF ${fmtPf(cell.pf)} · n ${cell.n ?? 0} · WR ${fmtPct(cell.win_rate)}`
                                    : "no data"
                                }
                                style={{
                                  ...cellStyle,
                                  textAlign: "center",
                                  padding: "4px 6px",
                                  background: cell ? `${color}22` : "transparent",
                                  borderLeft: `3px solid ${cell ? color : "transparent"}`,
                                }}
                              >
                                {cell ? (
                                  <div style={{ lineHeight: 1.3 }}>
                                    <div style={{ fontWeight: 700, color, textTransform: "uppercase", fontSize: 9 }}>
                                      {cell.verdict}
                                    </div>
                                    <div style={{ color: "#e2e8f0" }}>{fmtPf(cell.pf)}</div>
                                    <div style={{ color: "#64748b", fontSize: 10 }}>n{cell.n ?? 0}</div>
                                  </div>
                                ) : (
                                  <span style={{ color: "#334155" }}>·</span>
                                )}
                              </td>
                            );
                          })
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* ---------- WOULD-DROP IMPACT ---------- */}
          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Enforcement Impact</div>
                <h2 style={{ margin: "4px 0" }}>Would-Drop Impact ({data.wouldDropImpact.total})</h2>
              </div>
            </div>
            <p style={{ color: "#94a3b8", fontSize: 12, margin: "0 0 12px" }}>
              Signals from currently-tradable scanners (monitor-only excluded:{" "}
              {data.meta.monitor_only_scanners.join(", ")}) that landed in a <strong>block</strong>{" "}
              cell — these would be dropped if the router were enforced. Their realized PnL (where
              closed) tells you whether dropping them would have helped.
            </p>

            {data.wouldDropImpact.byScanner.length === 0 ? (
              <div className="ops-empty">No would-drop signals in this window.</div>
            ) : (
              <>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, marginBottom: 20 }}>
                  <thead>
                    <tr>
                      <th style={leftHead}>Scanner</th>
                      <th style={headStyle}>Would-drop</th>
                      <th style={headStyle}>Closed</th>
                      <th style={headStyle}>Avg Realized PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.wouldDropImpact.byScanner.map((s) => (
                      <tr key={s.scanner_name}>
                        <td style={leftCell}>{s.scanner_name}</td>
                        <td style={cellStyle}>{s.count}</td>
                        <td style={cellStyle}>{s.closed}</td>
                        <td
                          style={{
                            ...cellStyle,
                            color:
                              s.avg_realized_pnl_pct == null
                                ? "#e2e8f0"
                                : s.avg_realized_pnl_pct >= 0
                                  ? VERDICT_COLOR.trade
                                  : VERDICT_COLOR.block,
                          }}
                        >
                          {fmtSignedPct(s.avg_realized_pnl_pct)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>
                  Individual signals (most recent first, up to 1000)
                </div>
                <div style={{ maxHeight: 480, overflowY: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr>
                        <th style={leftHead}>Time</th>
                        <th style={leftHead}>Ticker</th>
                        <th style={leftHead}>Scanner</th>
                        <th style={leftHead}>Cell</th>
                        <th style={headStyle}>Cell PF</th>
                        <th style={headStyle}>Cell n</th>
                        <th style={leftHead}>Status</th>
                        <th style={headStyle}>Realized PnL</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.wouldDropImpact.signals.map((sig, i) => (
                        <tr key={`${sig.ticker}-${sig.signal_timestamp}-${i}`}>
                          <td style={leftCell}>{fmtDateTime(sig.signal_timestamp)}</td>
                          <td style={{ ...leftCell, fontWeight: 600 }}>{sig.ticker}</td>
                          <td style={leftCell}>{sig.scanner_name}</td>
                          <td style={{ ...leftCell, color: "#64748b" }}>
                            {sig.cell_key.split("|").slice(1).join("·")}
                          </td>
                          <td style={cellStyle}>{fmtPf(sig.cell_pf)}</td>
                          <td style={cellStyle}>{sig.cell_n ?? "—"}</td>
                          <td style={leftCell}>{sig.status ?? "—"}</td>
                          <td
                            style={{
                              ...cellStyle,
                              color:
                                sig.realized_pnl_pct == null
                                  ? "#64748b"
                                  : sig.realized_pnl_pct >= 0
                                    ? VERDICT_COLOR.trade
                                    : VERDICT_COLOR.block,
                            }}
                          >
                            {sig.realized_pnl_pct == null ? "open" : fmtSignedPct(sig.realized_pnl_pct)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        </>
      )}
    </div>
  );
}
