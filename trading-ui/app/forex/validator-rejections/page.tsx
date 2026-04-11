/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

// ─── Types ───────────────────────────────────────────────────────────────────

type Options = { steps: string[]; pairs: string[]; table_exists: boolean };

type Stats = {
  total: number;
  unique_pairs: number;
  top_step: string;
  top_pair: string;
  total_lpf: number;
  avg_lpf_penalty: string | null;
  max_lpf_penalty: string | null;
  by_step: Record<string, number>;
  by_pair: Array<{ pair: string; count: number }>;
  by_direction: Record<string, number>;
};

type Row = {
  id: number;
  created_at: string;
  epic: string;
  pair: string | null;
  signal_type: string | null;
  strategy: string | null;
  confidence_score: number | null;
  step: string;
  rejection_reason: string;
  entry_price: number | null;
  risk_pips: number | null;
  reward_pips: number | null;
  rr_ratio: number | null;
  market_regime: string | null;
  market_session: string | null;
  lpf_penalty: number | null;
  lpf_would_block: boolean | null;
  lpf_triggered_rules: string[] | null;
};

type LpfRule = {
  rule_name: string;
  times_triggered: number;
  pairs_affected: number;
  avg_total_penalty: number;
};

type LpfPayload = {
  rule_breakdown: LpfRule[];
  by_pair: Array<{ pair: string; total_lpf_blocks: number; avg_penalty: number; max_penalty: number }>;
  hourly: Array<{ hour: number; count: number }>;
};

// ─── Constants ───────────────────────────────────────────────────────────────

const DAY_OPTIONS = [1, 3, 7, 14, 30, 60, 90];
const SIGNAL_TYPES = ["All", "BULL", "BEAR"];

const STEP_COLORS: Record<string, string> = {
  STRUCTURE:            "#868e96",
  MARKET_HOURS:         "#adb5bd",
  EPIC:                 "#ced4da",
  CONFIDENCE:           "#f08c00",
  RISK:                 "#e03131",
  SR_LEVEL:             "#1098ad",
  NEWS:                 "#7950f2",
  MARKET_INTELLIGENCE:  "#2f9e44",
  TRADING_SUITABILITY:  "#fab005",
  LPF:                  "#d9480f",
  CLAUDE:               "#ae3ec9",
  UNKNOWN:              "#495057"
};

const STEP_LABELS: Record<string, string> = {
  STRUCTURE:            "Structure",
  MARKET_HOURS:         "Market Hours",
  EPIC:                 "Epic Blocked",
  CONFIDENCE:           "Confidence",
  RISK:                 "Risk / R:R",
  SR_LEVEL:             "S/R Level",
  NEWS:                 "News Filter",
  MARKET_INTELLIGENCE:  "Market Intelligence",
  TRADING_SUITABILITY:  "Trading Suitability",
  LPF:                  "Loss Prevention",
  CLAUDE:               "Claude AI",
  UNKNOWN:              "Unknown"
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

const formatDt = (value: string) => {
  const d = new Date(value);
  if (Number.isNaN(d.valueOf())) return value;
  return d.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
};

const pct = (v: number, total: number) => (!total ? "0.0" : ((v / total) * 100).toFixed(1));
const stepColor = (step: string) => STEP_COLORS[step] ?? STEP_COLORS.UNKNOWN;
const stepLabel = (step: string) => STEP_LABELS[step] ?? step;

// ─── Sub-components ──────────────────────────────────────────────────────────

function HourBar({ hourly, color = "#1971c2" }: { hourly: Array<{ hour: number; count: number }>; color?: string }) {
  const max = Math.max(...hourly.map((x) => x.count), 1);
  return (
    <div style={{ display: "flex", gap: 3, alignItems: "flex-end", height: 100, marginTop: 8 }}>
      {hourly.map(({ hour, count }) => (
        <div key={hour} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
          <div
            title={`${String(hour).padStart(2, "0")}:00 — ${count} rejections`}
            style={{
              width: "100%",
              height: count ? `${Math.max((count / max) * 88, 2)}px` : "2px",
              background: count > max * 0.7 ? "#e03131" : count > max * 0.4 ? "#f08c00" : color,
              borderRadius: "2px 2px 0 0",
              minHeight: 2,
              transition: "height 0.2s"
            }}
          />
          <span style={{ fontSize: 9, color: "#adb5bd" }}>{String(hour).padStart(2, "0")}</span>
        </div>
      ))}
    </div>
  );
}

function StepBadge({ step }: { step: string }) {
  return (
    <span style={{
      background: stepColor(step), color: "#fff",
      borderRadius: 4, padding: "2px 7px", fontSize: 11, fontWeight: 600,
      whiteSpace: "nowrap"
    }}>
      {stepLabel(step)}
    </span>
  );
}

function DirBadge({ dir }: { dir: string | null }) {
  const bg = dir === "BULL" ? "#2f9e44" : dir === "BEAR" ? "#e03131" : "#868e96";
  return (
    <span style={{ background: bg, color: "#fff", borderRadius: 4, padding: "2px 6px", fontSize: 11, fontWeight: 600 }}>
      {dir ?? "—"}
    </span>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ValidatorRejectionsPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(7);
  const [step, setStep] = useState("All");
  const [pair, setPair] = useState("All");
  const [signalType, setSignalType] = useState("All");
  const [activeTab, setActiveTab] = useState("steps");

  const [options, setOptions] = useState<Options | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [lpf, setLpf] = useState<LpfPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/trading/api/forex/validator-rejections/options/?env=${environment}`)
      .then((r) => r.json())
      .then(setOptions)
      .catch(() => setOptions(null));
  }, [environment]);

  const loadData = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ days: String(days), step, pair, signal_type: signalType });

    Promise.all([
      fetch(`/trading/api/forex/validator-rejections/stats/?days=${days}&env=${environment}`).then((r) => r.json()),
      fetch(`/trading/api/forex/validator-rejections/list/?${params}&env=${environment}`).then((r) => r.json()),
      fetch(`/trading/api/forex/validator-rejections/lpf/?days=${days}&env=${environment}`).then((r) => r.json())
    ])
      .then(([s, l, lpfData]) => {
        setStats(s);
        setRows(l.rows ?? []);
        setLpf(lpfData);
      })
      .catch(() => setError("Failed to load validator rejection data."))
      .finally(() => setLoading(false));
  };

  useEffect(() => { loadData(); }, []);

  // ── Derived ───────────────────────────────────────────────────────────────

  const filteredRows = useMemo(() => {
    let r = rows;
    if (step !== "All") r = r.filter((x) => x.step === step);
    if (pair !== "All") r = r.filter((x) => x.pair === pair);
    if (signalType !== "All") r = r.filter((x) => x.signal_type === signalType);
    return r;
  }, [rows, step, pair, signalType]);

  const stepCounts = useMemo(() => {
    if (stats?.by_step && Object.keys(stats.by_step).length > 0) {
      return Object.entries(stats.by_step)
        .map(([label, value]) => ({ label, value: Number(value) }))
        .sort((a, b) => b.value - a.value);
    }
    const counts: Record<string, number> = {};
    rows.forEach((r) => { counts[r.step] = (counts[r.step] || 0) + 1; });
    return Object.entries(counts).map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);
  }, [rows, stats?.by_step]);

  const stepTotal = stepCounts.reduce((s, r) => s + r.value, 0) || 1;

  const hourCounts = useMemo(() =>
    Array.from({ length: 24 }, (_, h) => {
      const count = filteredRows.filter((r) => new Date(r.created_at).getUTCHours() === h).length;
      return { hour: h, count };
    }),
    [filteredRows]
  );

  const regimeCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    filteredRows.forEach((r) => { const k = r.market_regime ?? "unknown"; counts[k] = (counts[k] || 0) + 1; });
    return Object.entries(counts).sort(([, a], [, b]) => b - a);
  }, [filteredRows]);

  const sessionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    filteredRows.forEach((r) => { const k = r.market_session ?? "unknown"; counts[k] = (counts[k] || 0) + 1; });
    return Object.entries(counts).sort(([, a], [, b]) => b - a);
  }, [filteredRows]);

  const pairStepBreakdown = useMemo(() => {
    const acc: Record<string, Record<string, number>> = {};
    filteredRows.forEach((r) => {
      const p = r.pair ?? "Unknown";
      if (!acc[p]) acc[p] = {};
      acc[p][r.step] = (acc[p][r.step] || 0) + 1;
    });
    return Object.entries(acc)
      .flatMap(([p, steps]) => Object.entries(steps).map(([s, c]) => ({ pair: p, step: s, count: c })))
      .sort((a, b) => b.count - a.count)
      .slice(0, 30);
  }, [filteredRows]);

  // ── Early return if table missing ─────────────────────────────────────────

  if (options && !options.table_exists) {
    return (
      <div className="page">
        <div className="topbar">
          <Link href="/" className="brand">Trading Hub</Link>
        </div>
        <div className="panel">
          <div className="error">
            validator_rejections table not found. Run the migration: create_validator_rejections_table.sql
          </div>
        </div>
      </div>
    );
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">Trading Hub</Link>
        <EnvironmentToggle />
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Validator Rejection Log</h1>
          <p>All TradeValidator rejections across all 11 filter steps — confidence, R:R, LPF, Claude AI and more.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/validator-rejections" />

      <div className="panel">
        {/* ── Controls ── */}
        <div className="forex-controls">
          <div>
            <label>Time Period</label>
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {DAY_OPTIONS.map((d) => <option key={d} value={d}>{d} days</option>)}
            </select>
          </div>
          <div>
            <label>Step</label>
            <select value={step} onChange={(e) => setStep(e.target.value)}>
              <option value="All">All Steps</option>
              {(options?.steps ?? []).map((s) => <option key={s} value={s}>{stepLabel(s)}</option>)}
            </select>
          </div>
          <div>
            <label>Pair</label>
            <select value={pair} onChange={(e) => setPair(e.target.value)}>
              <option value="All">All Pairs</option>
              {(options?.pairs ?? []).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div>
            <label>Direction</label>
            <select value={signalType} onChange={(e) => setSignalType(e.target.value)}>
              {SIGNAL_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <button className="section-tab active" onClick={loadData} disabled={loading}>
            {loading ? "Loading…" : "↻ Refresh"}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {/* ── Summary cards ── */}
        {loading ? (
          <div className="chart-placeholder">Loading rejection analytics…</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Total Rejections
                <strong>{(stats?.total ?? 0).toLocaleString()}</strong>
              </div>
              <div className="summary-card">
                Unique Pairs
                <strong>{stats?.unique_pairs ?? 0}</strong>
              </div>
              <div className="summary-card">
                Top Step
                <strong>{stepLabel(stats?.top_step ?? "N/A")}</strong>
              </div>
              <div className="summary-card">
                Most Rejected Pair
                <strong>{stats?.top_pair ?? "N/A"}</strong>
              </div>
              <div className="summary-card">
                LPF Blocks
                <strong>{stats?.total_lpf ?? 0}</strong>
              </div>
              {stats?.avg_lpf_penalty && (
                <div className="summary-card">
                  Avg LPF Penalty
                  <strong>{stats.avg_lpf_penalty}</strong>
                </div>
              )}
            </div>

            {/* ── Direction split ── */}
            {stats?.by_direction && Object.keys(stats.by_direction).length > 0 && (() => {
              const bull = Number(stats.by_direction.BULL ?? 0);
              const bear = Number(stats.by_direction.BEAR ?? 0);
              const total = bull + bear || 1;
              return (
                <div className="direction-analysis">
                  <h4>Direction Split</h4>
                  <div className="direction-comparison">
                    <div className="direction-stat bull">
                      <span className="label">BULL Rejections</span>
                      <span className="value">{bull}</span>
                      <span className="sub">{((bull / total) * 100).toFixed(0)}% of total</span>
                    </div>
                    <div className="direction-stat bear">
                      <span className="label">BEAR Rejections</span>
                      <span className="value">{bear}</span>
                      <span className="sub">{((bear / total) * 100).toFixed(0)}% of total</span>
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* ── Tabs ── */}
            <div className="section-tabs">
              {[
                { key: "steps", label: "By Step" },
                { key: "pairs", label: "By Pair" },
                { key: "time",  label: "Time Analysis" },
                { key: "lpf",   label: "LPF Detail" },
                { key: "list",  label: "Recent List" }
              ].map(({ key, label }) => (
                <button
                  key={key}
                  className={`section-tab ${activeTab === key ? "active" : ""}`}
                  onClick={() => setActiveTab(key)}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* ════════ BY STEP ════════ */}
            {activeTab === "steps" && (
              <>
                <h3>Rejections by Validator Step</h3>
                <p className="muted">Which filter step is blocking the most signals.</p>
                {stepCounts.length === 0 ? (
                  <div className="chart-placeholder">No rejections logged yet — data appears as signals are processed.</div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Step</th>
                        <th style={{ textAlign: "right" }}>Count</th>
                        <th style={{ textAlign: "right" }}>Share</th>
                        <th>Distribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stepCounts.map(({ label, value }) => (
                        <tr key={label}>
                          <td><StepBadge step={label} /></td>
                          <td style={{ textAlign: "right", fontWeight: 600 }}>{value}</td>
                          <td style={{ textAlign: "right", color: "#868e96" }}>{pct(value, stepTotal)}%</td>
                          <td style={{ width: 200 }}>
                            <div style={{ background: "#f1f3f5", borderRadius: 4, height: 8 }}>
                              <div style={{
                                width: `${pct(value, stepTotal)}%`, height: 8,
                                borderRadius: 4, background: stepColor(label)
                              }} />
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </>
            )}

            {/* ════════ BY PAIR ════════ */}
            {activeTab === "pairs" && (
              <>
                <h3>Rejections by Pair</h3>
                {(stats?.by_pair ?? []).length === 0 ? (
                  <div className="chart-placeholder">No data for this period.</div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Pair</th>
                        <th style={{ textAlign: "right" }}>Rejections</th>
                        <th style={{ textAlign: "right" }}>Share</th>
                        <th>Distribution</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(stats?.by_pair ?? []).map(({ pair: p, count }) => {
                        const tot = (stats?.by_pair ?? []).reduce((s, x) => s + x.count, 0) || 1;
                        return (
                          <tr key={p}>
                            <td style={{ fontWeight: 600 }}>{p}</td>
                            <td style={{ textAlign: "right" }}>{count}</td>
                            <td style={{ textAlign: "right", color: "#868e96" }}>{pct(count, tot)}%</td>
                            <td style={{ width: 200 }}>
                              <div style={{ background: "#f1f3f5", borderRadius: 4, height: 8 }}>
                                <div style={{ width: `${pct(count, tot)}%`, height: 8, borderRadius: 4, background: "#1971c2" }} />
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}

                {pairStepBreakdown.length > 0 && (
                  <>
                    <h4 style={{ marginTop: 24 }}>Pair × Step Breakdown</h4>
                    <table>
                      <thead>
                        <tr>
                          <th>Pair</th>
                          <th>Step</th>
                          <th style={{ textAlign: "right" }}>Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pairStepBreakdown.map(({ pair: p, step: s, count }) => (
                          <tr key={`${p}-${s}`}>
                            <td>{p}</td>
                            <td><StepBadge step={s} /></td>
                            <td style={{ textAlign: "right", fontWeight: 600 }}>{count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}
              </>
            )}

            {/* ════════ TIME ANALYSIS ════════ */}
            {activeTab === "time" && (
              <>
                <h3>Rejections by Hour (UTC)</h3>
                <p className="muted">When are signals getting rejected most frequently?</p>
                <HourBar hourly={hourCounts} />

                {regimeCounts.length > 0 && (
                  <>
                    <h4 style={{ marginTop: 24 }}>By Market Regime</h4>
                    <table>
                      <thead><tr><th>Regime</th><th style={{ textAlign: "right" }}>Count</th></tr></thead>
                      <tbody>
                        {regimeCounts.map(([regime, count]) => (
                          <tr key={regime}>
                            <td style={{ textTransform: "capitalize" }}>{regime}</td>
                            <td style={{ textAlign: "right" }}>{count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}

                {sessionCounts.length > 0 && (
                  <>
                    <h4 style={{ marginTop: 24 }}>By Session</h4>
                    <table>
                      <thead><tr><th>Session</th><th style={{ textAlign: "right" }}>Count</th></tr></thead>
                      <tbody>
                        {sessionCounts.map(([session, count]) => (
                          <tr key={session}>
                            <td style={{ textTransform: "capitalize" }}>{session}</td>
                            <td style={{ textAlign: "right" }}>{count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}

                {filteredRows.length === 0 && (
                  <div className="chart-placeholder">No data for the current filters.</div>
                )}
              </>
            )}

            {/* ════════ LPF DETAIL ════════ */}
            {activeTab === "lpf" && (
              <>
                <h3>Loss Prevention Filter — Rule Breakdown</h3>
                <p className="muted">Which LPF rules are triggering most often across hard-blocked signals.</p>

                {!lpf || lpf.rule_breakdown.length === 0 ? (
                  <div className="chart-placeholder">No LPF blocks logged yet.</div>
                ) : (
                  <>
                    <table>
                      <thead>
                        <tr>
                          <th>Rule</th>
                          <th style={{ textAlign: "right" }}>Triggered</th>
                          <th style={{ textAlign: "right" }}>Pairs Affected</th>
                          <th style={{ textAlign: "right" }}>Avg Total Penalty</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lpf.rule_breakdown.map((r) => (
                          <tr key={r.rule_name}>
                            <td><code style={{ fontSize: 12 }}>{r.rule_name}</code></td>
                            <td style={{ textAlign: "right", fontWeight: 600 }}>{r.times_triggered}</td>
                            <td style={{ textAlign: "right" }}>{r.pairs_affected}</td>
                            <td style={{ textAlign: "right", color: "#f08c00" }}>{Number(r.avg_total_penalty).toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    <h4 style={{ marginTop: 24 }}>LPF Blocks by Pair</h4>
                    <table>
                      <thead>
                        <tr>
                          <th>Pair</th>
                          <th style={{ textAlign: "right" }}>Blocks</th>
                          <th style={{ textAlign: "right" }}>Avg Penalty</th>
                          <th style={{ textAlign: "right" }}>Max Penalty</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lpf.by_pair.map((r) => (
                          <tr key={r.pair}>
                            <td style={{ fontWeight: 600 }}>{r.pair}</td>
                            <td style={{ textAlign: "right" }}>{r.total_lpf_blocks}</td>
                            <td style={{ textAlign: "right" }}>{Number(r.avg_penalty).toFixed(2)}</td>
                            <td style={{ textAlign: "right", color: "#e03131" }}>{Number(r.max_penalty).toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    {lpf.hourly.length > 0 && (
                      <>
                        <h4 style={{ marginTop: 24 }}>LPF Blocks by Hour (UTC)</h4>
                        <HourBar
                          hourly={Array.from({ length: 24 }, (_, h) => ({
                            hour: h,
                            count: Number(lpf.hourly.find((x) => Number(x.hour) === h)?.count ?? 0)
                          }))}
                          color="#d9480f"
                        />
                      </>
                    )}
                  </>
                )}
              </>
            )}

            {/* ════════ RECENT LIST ════════ */}
            {activeTab === "list" && (
              <>
                <h3>
                  Recent Rejections
                  <span className="muted" style={{ fontSize: 13, fontWeight: 400, marginLeft: 8 }}>
                    ({filteredRows.length} records)
                  </span>
                </h3>

                {filteredRows.length === 0 ? (
                  <div className="chart-placeholder">No rejections match the current filters.</div>
                ) : (
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ minWidth: 900 }}>
                      <thead>
                        <tr>
                          <th>Time</th>
                          <th>Pair</th>
                          <th>Dir</th>
                          <th>Step</th>
                          <th style={{ textAlign: "right" }}>Conf</th>
                          <th style={{ textAlign: "right" }}>R:R</th>
                          <th>Regime</th>
                          <th>Reason</th>
                          <th>LPF Rules</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredRows.slice(0, 200).map((r) => (
                          <tr key={r.id}>
                            <td style={{ whiteSpace: "nowrap", color: "#868e96" }}>{formatDt(r.created_at)}</td>
                            <td style={{ fontWeight: 600 }}>{r.pair ?? r.epic}</td>
                            <td><DirBadge dir={r.signal_type} /></td>
                            <td><StepBadge step={r.step} /></td>
                            <td style={{ textAlign: "right", color: "#868e96" }}>
                              {r.confidence_score != null ? `${(r.confidence_score * 100).toFixed(0)}%` : "—"}
                            </td>
                            <td style={{ textAlign: "right", color: "#868e96" }}>
                              {r.rr_ratio != null ? Number(r.rr_ratio).toFixed(2) : "—"}
                            </td>
                            <td style={{ textTransform: "capitalize", color: "#868e96" }}>
                              {r.market_regime ?? "—"}
                            </td>
                            <td style={{ maxWidth: 260, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", color: "#868e96", fontSize: 12 }}>
                              {r.rejection_reason}
                            </td>
                            <td style={{ fontSize: 11, color: "#f08c00" }}>
                              {r.lpf_triggered_rules?.length ? r.lpf_triggered_rules.join(", ") : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
