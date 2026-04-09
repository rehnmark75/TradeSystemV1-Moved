/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

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

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ValidatorRejectionsPage() {
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

  // Load filter options once
  useEffect(() => {
    fetch("/trading/api/forex/validator-rejections/options/")
      .then((r) => r.json())
      .then(setOptions)
      .catch(() => setOptions(null));
  }, []);

  const loadData = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({ days: String(days), step, pair, signal_type: signalType });

    Promise.all([
      fetch(`/trading/api/forex/validator-rejections/stats/?days=${days}`).then((r) => r.json()),
      fetch(`/trading/api/forex/validator-rejections/list/?${params}`).then((r) => r.json()),
      fetch(`/trading/api/forex/validator-rejections/lpf/?days=${days}`).then((r) => r.json())
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

  // ── Derived ──────────────────────────────────────────────────────────────

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

  const hourCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    filteredRows.forEach((r) => {
      const h = new Date(r.created_at).getUTCHours();
      counts[h] = (counts[h] || 0) + 1;
    });
    return Array.from({ length: 24 }, (_, h) => ({ hour: h, count: counts[h] ?? 0 }));
  }, [filteredRows]);

  const maxHourCount = Math.max(...hourCounts.map((x) => x.count), 1);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="page-content">
      {/* ── Breadcrumb nav ── */}
      <div className="forex-nav">
        <Link href="/forex/strategy" className="forex-pill">Strategy Performance</Link>
        <Link href="/forex/trade-performance" className="forex-pill">Trade Performance</Link>
        <Link href="/forex/entry-timing" className="forex-pill">Entry Timing</Link>
        <Link href="/forex/mae-analysis" className="forex-pill">MAE Analysis</Link>
        <Link href="/forex/alert-history" className="forex-pill">Alert History</Link>
        <Link href="/forex/trade-analysis" className="forex-pill">Trade Analysis</Link>
        <Link href="/forex/performance-snapshot" className="forex-pill">Performance Snapshot</Link>
        <Link href="/forex/market-intelligence" className="forex-pill">Market Intelligence</Link>
        <Link href="/forex/smc-rejections" className="forex-pill">SMC Rejections</Link>
        <Link href="/forex/validator-rejections" className="forex-pill" style={{ background: "#1971c2", color: "#fff" }}>
          Validator Rejections
        </Link>
        <Link href="/forex/filter-effectiveness" className="forex-pill">Filter Audit</Link>
      </div>

      {/* ── Header ── */}
      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
          <div>
            <h2 style={{ margin: 0 }}>Validator Rejection Log</h2>
            <p style={{ margin: "4px 0 0", color: "#868e96", fontSize: 13 }}>
              All TradeValidator rejections across all 11 filter steps — confidence, R:R, LPF, Claude AI and more.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {DAY_OPTIONS.map((d) => <option key={d} value={d}>{d}d</option>)}
            </select>
            <select value={step} onChange={(e) => setStep(e.target.value)}>
              <option value="All">All Steps</option>
              {(options?.steps ?? []).map((s) => <option key={s} value={s}>{stepLabel(s)}</option>)}
            </select>
            <select value={pair} onChange={(e) => setPair(e.target.value)}>
              <option value="All">All Pairs</option>
              {(options?.pairs ?? []).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            <select value={signalType} onChange={(e) => setSignalType(e.target.value)}>
              {SIGNAL_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            <button onClick={loadData} disabled={loading}>
              {loading ? "Loading…" : "Refresh"}
            </button>
          </div>
        </div>

        {error && <p style={{ color: "#e03131", marginTop: 12 }}>{error}</p>}

        {/* ── Summary cards ── */}
        {stats && (
          <div className="metric-grid" style={{ marginTop: 16 }}>
            <div className="metric-card">
              <div className="metric-label">Total Rejections</div>
              <div className="metric-value">{stats.total.toLocaleString()}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Unique Pairs</div>
              <div className="metric-value">{stats.unique_pairs}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Top Step</div>
              <div className="metric-value" style={{ fontSize: 16 }}>{stepLabel(stats.top_step)}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Top Pair</div>
              <div className="metric-value" style={{ fontSize: 16 }}>{stats.top_pair}</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">LPF Blocks</div>
              <div className="metric-value">{stats.total_lpf}</div>
            </div>
            {stats.avg_lpf_penalty && (
              <div className="metric-card">
                <div className="metric-label">Avg LPF Penalty</div>
                <div className="metric-value">{stats.avg_lpf_penalty}</div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Tabs ── */}
      <div className="panel">
        <div className="section-tabs">
          {[
            { key: "steps",   label: "By Step" },
            { key: "pairs",   label: "By Pair" },
            { key: "time",    label: "Time Analysis" },
            { key: "lpf",     label: "LPF Detail" },
            { key: "list",    label: "Recent List" }
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

        {/* ── BY STEP tab ── */}
        {activeTab === "steps" && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginTop: 0 }}>Rejections by Validator Step</h3>
            <p style={{ color: "#868e96", fontSize: 13 }}>
              Which filter step is blocking the most signals.
            </p>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #373a40" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px" }}>Step</th>
                  <th style={{ textAlign: "right", padding: "6px 8px" }}>Count</th>
                  <th style={{ textAlign: "right", padding: "6px 8px" }}>Share</th>
                  <th style={{ padding: "6px 8px" }}>Bar</th>
                </tr>
              </thead>
              <tbody>
                {stepCounts.map(({ label, value }) => (
                  <tr key={label} style={{ borderBottom: "1px solid #2c2e33" }}>
                    <td style={{ padding: "6px 8px" }}>
                      <span style={{
                        display: "inline-block", width: 10, height: 10, borderRadius: 2,
                        background: stepColor(label), marginRight: 8
                      }} />
                      {stepLabel(label)}
                    </td>
                    <td style={{ textAlign: "right", padding: "6px 8px", fontWeight: 600 }}>{value}</td>
                    <td style={{ textAlign: "right", padding: "6px 8px", color: "#adb5bd" }}>
                      {pct(value, stepTotal)}%
                    </td>
                    <td style={{ padding: "6px 8px", width: 200 }}>
                      <div style={{ background: "#2c2e33", borderRadius: 4, height: 8 }}>
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

            {/* Direction split */}
            {stats?.by_direction && Object.keys(stats.by_direction).length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h4>Direction Split</h4>
                <div style={{ display: "flex", gap: 16 }}>
                  {Object.entries(stats.by_direction).map(([dir, count]) => (
                    <div key={dir} className="metric-card" style={{ flex: 1 }}>
                      <div className="metric-label">{dir}</div>
                      <div className="metric-value">{count}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── BY PAIR tab ── */}
        {activeTab === "pairs" && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginTop: 0 }}>Rejections by Pair</h3>
            {(stats?.by_pair ?? []).length === 0 ? (
              <p style={{ color: "#868e96" }}>No data for this period.</p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid #373a40" }}>
                    <th style={{ textAlign: "left", padding: "6px 8px" }}>Pair</th>
                    <th style={{ textAlign: "right", padding: "6px 8px" }}>Rejections</th>
                    <th style={{ textAlign: "right", padding: "6px 8px" }}>Share</th>
                    <th style={{ padding: "6px 8px" }}>Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {(stats?.by_pair ?? []).map(({ pair: p, count }) => {
                    const total = (stats?.by_pair ?? []).reduce((s, x) => s + x.count, 0) || 1;
                    return (
                      <tr key={p} style={{ borderBottom: "1px solid #2c2e33" }}>
                        <td style={{ padding: "6px 8px", fontWeight: 600 }}>{p}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px" }}>{count}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px", color: "#adb5bd" }}>
                          {pct(count, total)}%
                        </td>
                        <td style={{ padding: "6px 8px", width: 200 }}>
                          <div style={{ background: "#2c2e33", borderRadius: 4, height: 8 }}>
                            <div style={{
                              width: `${pct(count, total)}%`, height: 8,
                              borderRadius: 4, background: "#228be6"
                            }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}

            {/* Pair × Step breakdown from filtered rows */}
            {filteredRows.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h4>Step breakdown for filtered rows</h4>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #373a40" }}>
                      <th style={{ textAlign: "left", padding: "4px 8px" }}>Pair</th>
                      <th style={{ textAlign: "left", padding: "4px 8px" }}>Step</th>
                      <th style={{ textAlign: "right", padding: "4px 8px" }}>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      filteredRows.reduce<Record<string, Record<string, number>>>((acc, r) => {
                        const p = r.pair ?? "Unknown";
                        if (!acc[p]) acc[p] = {};
                        acc[p][r.step] = (acc[p][r.step] || 0) + 1;
                        return acc;
                      }, {})
                    )
                      .flatMap(([p, steps]) =>
                        Object.entries(steps).map(([s, c]) => ({ pair: p, step: s, count: c }))
                      )
                      .sort((a, b) => b.count - a.count)
                      .slice(0, 40)
                      .map(({ pair: p, step: s, count }) => (
                        <tr key={`${p}-${s}`} style={{ borderBottom: "1px solid #2c2e33" }}>
                          <td style={{ padding: "4px 8px" }}>{p}</td>
                          <td style={{ padding: "4px 8px" }}>
                            <span style={{
                              background: stepColor(s), color: "#fff",
                              borderRadius: 4, padding: "1px 6px", fontSize: 11
                            }}>
                              {stepLabel(s)}
                            </span>
                          </td>
                          <td style={{ textAlign: "right", padding: "4px 8px", fontWeight: 600 }}>{count}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── TIME ANALYSIS tab ── */}
        {activeTab === "time" && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginTop: 0 }}>Rejections by Hour (UTC)</h3>
            <p style={{ color: "#868e96", fontSize: 13 }}>
              When are signals getting rejected most frequently?
            </p>
            <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 120, marginTop: 16 }}>
              {hourCounts.map(({ hour, count }) => (
                <div key={hour} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                  <div style={{
                    width: "100%",
                    height: count ? `${(count / maxHourCount) * 100}px` : "2px",
                    background: count > maxHourCount * 0.7 ? "#e03131" : count > maxHourCount * 0.4 ? "#f08c00" : "#228be6",
                    borderRadius: "2px 2px 0 0",
                    minHeight: 2
                  }} />
                  <span style={{ fontSize: 9, color: "#868e96" }}>{String(hour).padStart(2, "0")}</span>
                </div>
              ))}
            </div>

            {/* Market regime breakdown */}
            {filteredRows.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h4>By Market Regime</h4>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #373a40" }}>
                      <th style={{ textAlign: "left", padding: "4px 8px" }}>Regime</th>
                      <th style={{ textAlign: "right", padding: "4px 8px" }}>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      filteredRows.reduce<Record<string, number>>((acc, r) => {
                        const reg = r.market_regime ?? "unknown";
                        acc[reg] = (acc[reg] || 0) + 1;
                        return acc;
                      }, {})
                    )
                      .sort(([, a], [, b]) => b - a)
                      .map(([regime, count]) => (
                        <tr key={regime} style={{ borderBottom: "1px solid #2c2e33" }}>
                          <td style={{ padding: "4px 8px", textTransform: "capitalize" }}>{regime}</td>
                          <td style={{ textAlign: "right", padding: "4px 8px" }}>{count}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Session breakdown */}
            {filteredRows.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <h4>By Session</h4>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #373a40" }}>
                      <th style={{ textAlign: "left", padding: "4px 8px" }}>Session</th>
                      <th style={{ textAlign: "right", padding: "4px 8px" }}>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(
                      filteredRows.reduce<Record<string, number>>((acc, r) => {
                        const ses = r.market_session ?? "unknown";
                        acc[ses] = (acc[ses] || 0) + 1;
                        return acc;
                      }, {})
                    )
                      .sort(([, a], [, b]) => b - a)
                      .map(([session, count]) => (
                        <tr key={session} style={{ borderBottom: "1px solid #2c2e33" }}>
                          <td style={{ padding: "4px 8px", textTransform: "capitalize" }}>{session}</td>
                          <td style={{ textAlign: "right", padding: "4px 8px" }}>{count}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── LPF DETAIL tab ── */}
        {activeTab === "lpf" && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginTop: 0 }}>Loss Prevention Filter — Rule Breakdown</h3>
            <p style={{ color: "#868e96", fontSize: 13 }}>
              Which LPF rules are triggering most often across hard-blocked signals.
            </p>

            {!lpf || lpf.rule_breakdown.length === 0 ? (
              <p style={{ color: "#868e96" }}>No LPF blocks in this period.</p>
            ) : (
              <>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14, marginBottom: 24 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #373a40" }}>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Rule</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Triggered</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Pairs Affected</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Avg Penalty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lpf.rule_breakdown.map((r) => (
                      <tr key={r.rule_name} style={{ borderBottom: "1px solid #2c2e33" }}>
                        <td style={{ padding: "6px 8px", fontFamily: "monospace", fontSize: 12 }}>{r.rule_name}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px", fontWeight: 600 }}>{r.times_triggered}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px" }}>{r.pairs_affected}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px", color: "#f08c00" }}>
                          {Number(r.avg_total_penalty).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h4>LPF Blocks by Pair</h4>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #373a40" }}>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Pair</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Blocks</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Avg Penalty</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Max Penalty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lpf.by_pair.map((r) => (
                      <tr key={r.pair} style={{ borderBottom: "1px solid #2c2e33" }}>
                        <td style={{ padding: "6px 8px", fontWeight: 600 }}>{r.pair}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px" }}>{r.total_lpf_blocks}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px" }}>{Number(r.avg_penalty).toFixed(2)}</td>
                        <td style={{ textAlign: "right", padding: "6px 8px", color: "#e03131" }}>
                          {Number(r.max_penalty).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {lpf.hourly.length > 0 && (
                  <>
                    <h4 style={{ marginTop: 24 }}>LPF Blocks by Hour (UTC)</h4>
                    <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 80 }}>
                      {Array.from({ length: 24 }, (_, h) => {
                        const found = lpf.hourly.find((x) => Number(x.hour) === h);
                        const count = found ? Number(found.count) : 0;
                        const maxH = Math.max(...lpf.hourly.map((x) => Number(x.count)), 1);
                        return (
                          <div key={h} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                            <div style={{
                              width: "100%", height: count ? `${(count / maxH) * 70}px` : "2px",
                              background: "#d9480f", borderRadius: "2px 2px 0 0", minHeight: 2
                            }} />
                            <span style={{ fontSize: 9, color: "#868e96" }}>{String(h).padStart(2, "0")}</span>
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        )}

        {/* ── RECENT LIST tab ── */}
        {activeTab === "list" && (
          <div style={{ marginTop: 16 }}>
            <h3 style={{ marginTop: 0 }}>
              Recent Rejections
              <span style={{ fontSize: 13, fontWeight: 400, color: "#868e96", marginLeft: 8 }}>
                ({filteredRows.length} shown)
              </span>
            </h3>

            {filteredRows.length === 0 ? (
              <p style={{ color: "#868e96" }}>No rejections match the current filters.</p>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, minWidth: 800 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #373a40" }}>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Time</th>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Pair</th>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Dir</th>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Step</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>Conf</th>
                      <th style={{ textAlign: "right", padding: "6px 8px" }}>R:R</th>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Regime</th>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>Reason</th>
                      <th style={{ textAlign: "left", padding: "6px 8px" }}>LPF Rules</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.slice(0, 200).map((r) => (
                      <tr key={r.id} style={{ borderBottom: "1px solid #2c2e33" }}>
                        <td style={{ padding: "5px 8px", whiteSpace: "nowrap", color: "#adb5bd" }}>
                          {formatDt(r.created_at)}
                        </td>
                        <td style={{ padding: "5px 8px", fontWeight: 600 }}>{r.pair ?? r.epic}</td>
                        <td style={{ padding: "5px 8px" }}>
                          <span style={{
                            background: r.signal_type === "BULL" ? "#2f9e44" : r.signal_type === "BEAR" ? "#e03131" : "#495057",
                            color: "#fff", borderRadius: 4, padding: "1px 5px", fontSize: 11
                          }}>
                            {r.signal_type ?? "—"}
                          </span>
                        </td>
                        <td style={{ padding: "5px 8px" }}>
                          <span style={{
                            background: stepColor(r.step), color: "#fff",
                            borderRadius: 4, padding: "1px 6px", fontSize: 11
                          }}>
                            {stepLabel(r.step)}
                          </span>
                        </td>
                        <td style={{ textAlign: "right", padding: "5px 8px", color: "#adb5bd" }}>
                          {r.confidence_score != null ? `${(r.confidence_score * 100).toFixed(0)}%` : "—"}
                        </td>
                        <td style={{ textAlign: "right", padding: "5px 8px", color: "#adb5bd" }}>
                          {r.rr_ratio != null ? `${Number(r.rr_ratio).toFixed(2)}` : "—"}
                        </td>
                        <td style={{ padding: "5px 8px", color: "#adb5bd", textTransform: "capitalize" }}>
                          {r.market_regime ?? "—"}
                        </td>
                        <td style={{ padding: "5px 8px", color: "#868e96", fontSize: 12, maxWidth: 260, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {r.rejection_reason}
                        </td>
                        <td style={{ padding: "5px 8px", fontSize: 11, color: "#f08c00" }}>
                          {r.lpf_triggered_rules && r.lpf_triggered_rules.length > 0
                            ? r.lpf_triggered_rules.join(", ")
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
