/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type Scores = {
  trending: number;
  ranging: number;
  breakout: number;
  reversal: number;
  high_volatility: number;
  low_volatility: number;
};

type NowPayload = {
  scan_timestamp: string;
  regime: string;
  confidence: number | null;
  session: string;
  volatility_level: string;
  market_bias: string | null;
  risk_sentiment: string | null;
  recommended_strategy: string | null;
  avg_trend_strength: number | null;
  avg_volatility: number | null;
  scores: Scores;
};

type PairRow = {
  epic: string;
  price: number | null;
  regime: string;
  confidence: number | null;
  combined_regime: string | null;
  scores: Scores;
  alerts: number;
  closed: number;
  wins: number;
  losses: number;
  pips: number;
  pf: number | null;
  pf_infinite: boolean;
  last_alert: string | null;
};

type EffectivenessRow = {
  strategy: string;
  regime: string;
  session: string;
  alerts: number;
  closed: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  pf: number | null;
  pips: number;
};

type TimelinePoint = {
  scan_timestamp: string;
  regime: string;
  confidence: number | null;
  session: string;
  volatility: string;
};

type SessionRegimeMatrix = Array<{
  session: string;
  cells: Array<{ regime: string; count: number; avg_confidence: number }>;
}>;

type IntelligencePayload = {
  summary: { total: number; avg_epics: number; unique_regimes: number; avg_confidence: number };
  regimes: Record<string, number>;
  sessions: Record<string, number>;
  volatility: Record<string, number>;
  intelligence_sources: Record<string, number>;
  session_regime_matrix: SessionRegimeMatrix;
  now: NowPayload | null;
  timeline: TimelinePoint[];
  by_pair: PairRow[];
  effectiveness: EffectivenessRow[];
  comprehensive: Array<Record<string, any>>;
  signals: Array<Record<string, any>>;
};

const toDateInput = (value: Date) => value.toISOString().slice(0, 10);

const mapToRows = (data: Record<string, number>) =>
  Object.entries(data || {})
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value);

const fmtConf = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "-" : v.toFixed(2);

const fmtPct = (v: number | null | undefined) =>
  v == null || !Number.isFinite(v) ? "-" : `${(v * 100).toFixed(0)}%`;

const fmtPf = (v: number | null | undefined, infinite = false) => {
  if (infinite) return "∞";
  if (v == null || !Number.isFinite(v)) return "-";
  return v.toFixed(2);
};

const ageLabel = (iso: string | null | undefined) => {
  if (!iso) return "-";
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 0 || !Number.isFinite(ms)) return "-";
  const s = Math.floor(ms / 1000);
  if (s < 90) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 90) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
};

const REGIME_COLORS: Record<string, string> = {
  trending: "#2f9e44",
  breakout: "#f08c00",
  ranging: "#228be6",
  reversal: "#e03131",
  high_volatility: "#d9480f",
  low_volatility: "#2b8a3e",
  medium_volatility_ranging: "#228be6",
  high_volatility_trending: "#2f9e44",
  unknown: "#adb5bd"
};

const VOLATILITY_COLORS: Record<string, string> = {
  high: "#e03131",
  medium: "#f08c00",
  low: "#2f9e44"
};

const SESSION_ORDER = ["asian", "london", "new_york", "overlap", "quiet", "off_hours"];

function regimeColor(r: string | null | undefined) {
  if (!r) return REGIME_COLORS.unknown;
  return REGIME_COLORS[r] ?? REGIME_COLORS.unknown;
}

function recommend(now: NowPayload | null): { text: string; tone: "good" | "warn" | "bad" | "info" } {
  if (!now) return { text: "No scan data yet for this window.", tone: "info" };
  const regime = (now.regime || "").toLowerCase();
  const vol = (now.volatility_level || "").toLowerCase();
  const conf = now.confidence ?? 0;
  if (conf < 0.4) {
    return { text: `Contested market (confidence ${fmtConf(conf)}). Sit out or size down.`, tone: "warn" };
  }
  if (regime === "ranging" && (vol === "low" || vol === "medium")) {
    return { text: "Ranging + low/med vol → MEAN_REVERSION favored; SMC trend entries gated.", tone: "good" };
  }
  if (regime === "trending" && (vol === "medium" || vol === "high")) {
    return { text: "Trending + vol → SMC_SIMPLE / trend-continuation favored.", tone: "good" };
  }
  if (regime === "breakout") {
    return { text: "Breakout regime → volatility expansion; watch for news-driven spikes.", tone: "warn" };
  }
  if (regime === "reversal") {
    return { text: "Reversal regime → mixed signal; confirm with HTF bias before entry.", tone: "warn" };
  }
  if (vol === "high") {
    return { text: "High volatility → widen stops, reduce size.", tone: "warn" };
  }
  if (now.recommended_strategy) {
    return { text: `Scanner recommends: ${now.recommended_strategy}.`, tone: "info" };
  }
  return { text: `Current regime: ${regime || "unknown"}.`, tone: "info" };
}

const TONE_BG: Record<string, string> = {
  good: "#0f3a1f",
  warn: "#3a2a0a",
  bad: "#3a1515",
  info: "#12263a"
};

const TONE_BORDER: Record<string, string> = {
  good: "#2f9e44",
  warn: "#f08c00",
  bad: "#e03131",
  info: "#5bc0be"
};

export default function ForexMarketIntelligencePage() {
  const { environment } = useEnvironment();
  const [start, setStart] = useState(toDateInput(new Date(Date.now() - 7 * 24 * 3600 * 1000)));
  const [end, setEnd] = useState(toDateInput(new Date()));
  const [source, setSource] = useState("comprehensive");
  const [payload, setPayload] = useState<IntelligencePayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pairSort, setPairSort] = useState<"alerts" | "pf" | "confidence">("alerts");
  const [regimeFilter, setRegimeFilter] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const loadData = () => {
    setLoading(true);
    setError(null);
    fetch(
      `/trading/api/forex/market-intelligence/?start=${start}&end=${end}&source=${source}&env=${environment}`
    )
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load market intelligence."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, [environment]);

  const now = payload?.now ?? null;
  const rec = useMemo(() => recommend(now), [now]);

  const timeline = payload?.timeline ?? [];
  const byPair = payload?.by_pair ?? [];
  const effectiveness = payload?.effectiveness ?? [];
  const matrix = payload?.session_regime_matrix ?? [];

  // Timeline geometry: span from start→end, one segment per scan.
  const rangeMs = useMemo(() => {
    const s = new Date(start).getTime();
    const e = new Date(end).getTime() + 24 * 3600 * 1000; // inclusive end of day
    return { s, e, total: Math.max(e - s, 1) };
  }, [start, end]);

  const timelineSegments = useMemo(() => {
    if (!timeline.length) return [];
    const segs: Array<{ regime: string; left: number; width: number; t: string; conf: number | null }> = [];
    for (let i = 0; i < timeline.length; i++) {
      const cur = timeline[i];
      const nxt = timeline[i + 1];
      const curMs = new Date(cur.scan_timestamp).getTime();
      const nxtMs = nxt ? new Date(nxt.scan_timestamp).getTime() : rangeMs.e;
      const left = ((curMs - rangeMs.s) / rangeMs.total) * 100;
      const width = ((nxtMs - curMs) / rangeMs.total) * 100;
      segs.push({
        regime: cur.regime,
        left,
        width: Math.max(width, 0.05),
        t: cur.scan_timestamp,
        conf: cur.confidence
      });
    }
    return segs;
  }, [timeline, rangeMs]);

  // Per-pair filtering + sorting.
  const pairRows = useMemo(() => {
    let rows = byPair.slice();
    if (regimeFilter) rows = rows.filter((r) => r.regime === regimeFilter);
    rows.sort((a, b) => {
      if (pairSort === "pf") {
        const ap = a.pf_infinite ? Infinity : a.pf ?? -1;
        const bp = b.pf_infinite ? Infinity : b.pf ?? -1;
        return bp - ap;
      }
      if (pairSort === "confidence") return (b.confidence ?? 0) - (a.confidence ?? 0);
      return b.alerts - a.alerts;
    });
    return rows;
  }, [byPair, pairSort, regimeFilter]);

  const pairRegimes = useMemo(() => {
    const s = new Set<string>();
    byPair.forEach((r) => s.add(r.regime));
    return Array.from(s);
  }, [byPair]);

  // Secondary aggregations for the Advanced Details accordion.
  const regimeRows = useMemo(() => mapToRows(payload?.regimes ?? {}), [payload]);
  const sessionRows = useMemo(() => mapToRows(payload?.sessions ?? {}), [payload]);
  const volatilityRows = useMemo(() => mapToRows(payload?.volatility ?? {}), [payload]);
  const sourceRows = useMemo(() => mapToRows(payload?.intelligence_sources ?? {}), [payload]);
  const maxSession = sessionRows.length ? Math.max(...sessionRows.map((r) => r.value)) : 1;
  const maxVolatility = volatilityRows.length ? Math.max(...volatilityRows.map((r) => r.value)) : 1;
  const maxSource = sourceRows.length ? Math.max(...sourceRows.map((r) => r.value)) : 1;
  const recentItems = (source === "signal" ? payload?.signals : payload?.comprehensive) ?? [];

  // Heatmap geometry.
  const heatmap = useMemo(() => {
    const regimes = Array.from(
      new Set(matrix.flatMap((row) => row.cells.map((c) => c.regime)))
    );
    const sessions = matrix.map((r) => r.session).sort((a, b) => {
      const ai = SESSION_ORDER.indexOf(a);
      const bi = SESSION_ORDER.indexOf(b);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });
    const maxCount = Math.max(
      1,
      ...matrix.flatMap((row) => row.cells.map((c) => c.count))
    );
    return { regimes, sessions, maxCount };
  }, [matrix]);

  const getCell = (session: string, regime: string) => {
    const row = matrix.find((r) => r.session === session);
    return row?.cells.find((c) => c.regime === regime);
  };

  const envQ = `env=${environment}`;

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          K.L.I.R.R
        </Link>
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
          <h1>Market Intelligence</h1>
          <p>Current regime, per-pair state, and strategy effectiveness.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/market-intelligence" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Start</label>
            <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
          </div>
          <div>
            <label>End</label>
            <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
          </div>
          <div>
            <label>Source</label>
            <select value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="comprehensive">Comprehensive Scans</option>
              <option value="signal">Signal-Based</option>
              <option value="both">Both</option>
            </select>
          </div>
          <button className="section-tab active" onClick={loadData}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}
        {loading ? (
          <div className="chart-placeholder">Loading intelligence...</div>
        ) : (
          <>
            {/* 1. NOW BANNER */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1.2fr 1fr 1fr 1fr 1fr",
                gap: 12,
                alignItems: "stretch",
                padding: 16,
                background: TONE_BG[rec.tone],
                border: `1px solid ${TONE_BORDER[rec.tone]}`,
                borderRadius: 8,
                marginBottom: 16
              }}
            >
              <div>
                <div style={{ fontSize: 11, opacity: 0.7, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Now
                </div>
                <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>
                  <span style={{ color: regimeColor(now?.regime) }}>●</span> {now?.regime ?? "—"}
                </div>
                <div style={{ fontSize: 12, opacity: 0.7 }}>{ageLabel(now?.scan_timestamp)}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, opacity: 0.7, textTransform: "uppercase" }}>Confidence</div>
                <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{fmtConf(now?.confidence)}</div>
                <div style={{ fontSize: 12, opacity: 0.7 }}>
                  bias: {now?.market_bias ?? "—"}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, opacity: 0.7, textTransform: "uppercase" }}>Session</div>
                <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4 }}>{now?.session ?? "—"}</div>
                <div style={{ fontSize: 12, opacity: 0.7 }}>risk: {now?.risk_sentiment ?? "—"}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, opacity: 0.7, textTransform: "uppercase" }}>Volatility</div>
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 700,
                    marginTop: 4,
                    color: VOLATILITY_COLORS[(now?.volatility_level ?? "").toLowerCase()] ?? "inherit"
                  }}
                >
                  {now?.volatility_level ?? "—"}
                </div>
                <div style={{ fontSize: 12, opacity: 0.7 }}>
                  trend: {now?.avg_trend_strength != null ? now.avg_trend_strength.toFixed(2) : "—"}
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
                <div style={{ fontSize: 11, opacity: 0.7, textTransform: "uppercase" }}>Recommendation</div>
                <div style={{ fontSize: 14, marginTop: 4, lineHeight: 1.3 }}>{rec.text}</div>
                {now?.recommended_strategy ? (
                  <div style={{ fontSize: 12, opacity: 0.7, marginTop: 4 }}>
                    scanner → <strong>{now.recommended_strategy}</strong>
                  </div>
                ) : null}
              </div>
            </div>

            {/* 1b. Regime score breakdown bar for "Now" */}
            {now ? (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
                  Regime score distribution (full profile, not just argmax)
                </div>
                <div style={{ display: "flex", height: 20, borderRadius: 4, overflow: "hidden" }}>
                  {(Object.keys(now.scores) as Array<keyof Scores>)
                    .filter((k) => now.scores[k] > 0)
                    .sort((a, b) => now.scores[b] - now.scores[a])
                    .map((k) => {
                      const total = Object.values(now.scores).reduce((s, v) => s + v, 0) || 1;
                      const pct = (now.scores[k] / total) * 100;
                      return (
                        <div
                          key={k}
                          title={`${k}: ${now.scores[k].toFixed(3)}`}
                          style={{
                            width: `${pct}%`,
                            background: regimeColor(k),
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 10,
                            color: "#fff",
                            overflow: "hidden",
                            whiteSpace: "nowrap"
                          }}
                        >
                          {pct > 8 ? k : ""}
                        </div>
                      );
                    })}
                </div>
              </div>
            ) : null}

            {/* 2. REGIME TIMELINE */}
            <div className="panel chart-panel" style={{ marginBottom: 16 }}>
              <div className="chart-title">Regime Timeline</div>
              <p className="chart-caption">
                One stripe per scan, colored by dominant regime. Spot regime shifts at a glance.
              </p>
              <div
                style={{
                  position: "relative",
                  height: 40,
                  background: "#1a1a1a",
                  borderRadius: 4,
                  overflow: "hidden",
                  marginTop: 8
                }}
              >
                {timelineSegments.map((seg, idx) => (
                  <div
                    key={idx}
                    title={`${new Date(seg.t).toLocaleString()} — ${seg.regime} (${fmtConf(seg.conf)})`}
                    style={{
                      position: "absolute",
                      left: `${seg.left}%`,
                      width: `${seg.width}%`,
                      top: 0,
                      bottom: 0,
                      background: regimeColor(seg.regime),
                      opacity: Math.min(1, 0.4 + (seg.conf ?? 0) * 0.6)
                    }}
                  />
                ))}
                {!timelineSegments.length ? (
                  <div style={{ color: "#888", padding: 10, fontSize: 12 }}>No scans in this range.</div>
                ) : null}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, opacity: 0.6, marginTop: 4 }}>
                <span>{start}</span>
                <span>{timeline.length} scans</span>
                <span>{end}</span>
              </div>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 8, fontSize: 11 }}>
                {Object.entries(REGIME_COLORS)
                  .filter(([k]) => !k.includes("_"))
                  .map(([k, c]) => (
                    <span key={k} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <span style={{ width: 10, height: 10, background: c, borderRadius: 2 }} />
                      {k}
                    </span>
                  ))}
              </div>
            </div>

            {/* 3. PER-PAIR MATRIX */}
            <div className="panel table-panel" style={{ marginBottom: 16 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 8,
                  flexWrap: "wrap"
                }}
              >
                <div>
                  <div className="chart-title">Per-Pair State (latest scan)</div>
                  <p className="chart-caption">
                    Current regime + score breakdown per epic, with alerts & W/L over the window.
                  </p>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12 }}>
                  <label>Sort</label>
                  <select value={pairSort} onChange={(e) => setPairSort(e.target.value as any)}>
                    <option value="alerts">Most alerts</option>
                    <option value="pf">Best PF</option>
                    <option value="confidence">Highest confidence</option>
                  </select>
                  <label>Regime</label>
                  <select
                    value={regimeFilter ?? ""}
                    onChange={(e) => setRegimeFilter(e.target.value || null)}
                  >
                    <option value="">All</option>
                    {pairRegimes.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Epic</th>
                    <th>Regime</th>
                    <th>Conf</th>
                    <th style={{ minWidth: 140 }}>Scores</th>
                    <th>Alerts</th>
                    <th>Closed</th>
                    <th>W/L</th>
                    <th>Win%</th>
                    <th>PF</th>
                    <th>Pips</th>
                    <th>Last</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {pairRows.map((row) => {
                    const total =
                      Object.values(row.scores).reduce((s, v) => s + (v || 0), 0) || 1;
                    const wr = row.closed > 0 ? row.wins / row.closed : null;
                    return (
                      <tr key={row.epic}>
                        <td style={{ fontFamily: "monospace" }}>{row.epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", "")}</td>
                        <td>
                          <span
                            style={{
                              display: "inline-block",
                              width: 8,
                              height: 8,
                              background: regimeColor(row.regime),
                              borderRadius: 2,
                              marginRight: 6
                            }}
                          />
                          {row.regime}
                        </td>
                        <td>{fmtConf(row.confidence)}</td>
                        <td>
                          <div style={{ display: "flex", height: 12, borderRadius: 2, overflow: "hidden" }}>
                            {(Object.keys(row.scores) as Array<keyof Scores>)
                              .filter((k) => row.scores[k] > 0)
                              .map((k) => (
                                <div
                                  key={k}
                                  title={`${k}: ${row.scores[k].toFixed(2)}`}
                                  style={{
                                    width: `${(row.scores[k] / total) * 100}%`,
                                    background: regimeColor(k)
                                  }}
                                />
                              ))}
                          </div>
                        </td>
                        <td>{row.alerts}</td>
                        <td>{row.closed}</td>
                        <td>
                          <span style={{ color: "#2f9e44" }}>{row.wins}</span>/
                          <span style={{ color: "#e03131" }}>{row.losses}</span>
                        </td>
                        <td>{fmtPct(wr)}</td>
                        <td style={{ color: row.pf && row.pf >= 1 ? "#2f9e44" : row.pf != null ? "#e03131" : "inherit" }}>
                          {fmtPf(row.pf, row.pf_infinite)}
                        </td>
                        <td style={{ color: row.pips > 0 ? "#2f9e44" : row.pips < 0 ? "#e03131" : "inherit" }}>
                          {row.pips?.toFixed(1) ?? "-"}
                        </td>
                        <td style={{ fontSize: 11, opacity: 0.7 }}>{ageLabel(row.last_alert)}</td>
                        <td>
                          <Link
                            href={`/forex/chart?epic=${encodeURIComponent(row.epic)}`}
                            style={{ fontSize: 11, color: "#5bc0be" }}
                          >
                            chart →
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                  {!pairRows.length ? (
                    <tr>
                      <td colSpan={12} style={{ opacity: 0.5, textAlign: "center", padding: 20 }}>
                        No per-pair data in latest scan.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>

            {/* 4. STRATEGY EFFECTIVENESS */}
            <div className="panel table-panel" style={{ marginBottom: 16 }}>
              <div className="chart-title">Strategy Effectiveness (live outcomes)</div>
              <p className="chart-caption">
                Joined to <code>trade_log</code>. PF &amp; win% reflect actual closed trades in the window, not scanner opinions.
              </p>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Regime</th>
                    <th>Session</th>
                    <th>Alerts</th>
                    <th>Closed</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Win%</th>
                    <th>PF</th>
                    <th>Pips</th>
                  </tr>
                </thead>
                <tbody>
                  {effectiveness.slice(0, 40).map((r, i) => (
                    <tr key={i}>
                      <td>{r.strategy}</td>
                      <td>
                        <span
                          style={{
                            display: "inline-block",
                            width: 8,
                            height: 8,
                            background: regimeColor(r.regime),
                            borderRadius: 2,
                            marginRight: 6
                          }}
                        />
                        <Link href={`/forex/alert-history?regime=${encodeURIComponent(r.regime)}&${envQ}`}>
                          {r.regime}
                        </Link>
                      </td>
                      <td>{r.session}</td>
                      <td>{r.alerts}</td>
                      <td>{r.closed}</td>
                      <td style={{ color: "#2f9e44" }}>{r.wins}</td>
                      <td style={{ color: "#e03131" }}>{r.losses}</td>
                      <td>{fmtPct(r.win_rate)}</td>
                      <td
                        style={{
                          color:
                            r.pf == null
                              ? "inherit"
                              : r.pf >= 1.2
                              ? "#2f9e44"
                              : r.pf >= 1
                              ? "#f08c00"
                              : "#e03131"
                        }}
                      >
                        {fmtPf(r.pf)}
                      </td>
                      <td style={{ color: r.pips > 0 ? "#2f9e44" : r.pips < 0 ? "#e03131" : "inherit" }}>
                        {r.pips?.toFixed(1) ?? "-"}
                      </td>
                    </tr>
                  ))}
                  {!effectiveness.length ? (
                    <tr>
                      <td colSpan={10} style={{ opacity: 0.5, textAlign: "center", padding: 20 }}>
                        No alerts joined to trades in this window.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>

            {/* 5. SESSION × REGIME HEATMAP */}
            <div className="panel chart-panel" style={{ marginBottom: 16 }}>
              <div className="chart-title">Session × Regime Heatmap</div>
              <p className="chart-caption">
                Cell = scan count; color intensity ∝ count; label = avg confidence.
              </p>
              <div style={{ overflowX: "auto", marginTop: 8 }}>
                <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", padding: 6, opacity: 0.7 }}>session \ regime</th>
                      {heatmap.regimes.map((r) => (
                        <th
                          key={r}
                          style={{
                            textAlign: "center",
                            padding: 6,
                            borderBottom: `2px solid ${regimeColor(r)}`,
                            minWidth: 80
                          }}
                        >
                          {r}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {heatmap.sessions.map((s) => (
                      <tr key={s}>
                        <td style={{ padding: 6, fontWeight: 600 }}>{s}</td>
                        {heatmap.regimes.map((r) => {
                          const cell = getCell(s, r);
                          const intensity = cell ? cell.count / heatmap.maxCount : 0;
                          const base = regimeColor(r);
                          return (
                            <td
                              key={r}
                              title={
                                cell
                                  ? `${cell.count} scans, avg conf ${fmtConf(cell.avg_confidence)}`
                                  : "no data"
                              }
                              style={{
                                padding: 6,
                                textAlign: "center",
                                background: cell
                                  ? `${base}${Math.floor(intensity * 200 + 40)
                                      .toString(16)
                                      .padStart(2, "0")}`
                                  : "transparent",
                                border: "1px solid #222",
                                color: intensity > 0.5 ? "#fff" : "#ddd"
                              }}
                            >
                              {cell ? (
                                <>
                                  <div style={{ fontWeight: 700 }}>{cell.count}</div>
                                  <div style={{ fontSize: 10, opacity: 0.8 }}>
                                    {fmtConf(cell.avg_confidence)}
                                  </div>
                                </>
                              ) : (
                                "–"
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 6. SUMMARY STRIP */}
            <div className="metrics-grid">
              <div className="summary-card">
                Total Scans
                <strong>{payload?.summary?.total ?? 0}</strong>
              </div>
              <div className="summary-card">
                Avg Epics / Scan
                <strong>{payload?.summary?.avg_epics?.toFixed(1) ?? "-"}</strong>
              </div>
              <div className="summary-card">
                Regimes Seen
                <strong>{payload?.summary?.unique_regimes ?? 0}</strong>
              </div>
              <div className="summary-card">
                Avg Confidence
                <strong>{fmtConf(payload?.summary?.avg_confidence ?? 0)}</strong>
              </div>
              <div className="summary-card">
                Pairs Tracked
                <strong>{byPair.length}</strong>
              </div>
            </div>

            {/* 7. ADVANCED DETAILS (collapsed) */}
            <div style={{ marginTop: 16 }}>
              <button
                className="section-tab"
                onClick={() => setShowAdvanced((v) => !v)}
                style={{ marginBottom: 8 }}
              >
                {showAdvanced ? "▾ Hide" : "▸ Show"} advanced details
              </button>

              {showAdvanced ? (
                <>
                  <div className="forex-grid">
                    <div className="panel chart-panel">
                      <div className="chart-title">Trading Session Volume</div>
                      <p className="chart-caption">Signal/scan volume by trading session.</p>
                      <div className="vertical-bars">
                        {sessionRows.map((row) => (
                          <div key={row.label} className="vertical-bar-item">
                            <div className="vertical-bar-track">
                              <div
                                className="vertical-bar-fill"
                                style={{ height: `${(row.value / maxSession) * 100}%` }}
                              />
                            </div>
                            <span className="vertical-bar-label">{row.label}</span>
                            <strong className="vertical-bar-value">{row.value}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="panel chart-panel">
                      <div className="chart-title">Volatility Level Distribution</div>
                      <p className="chart-caption">How often each volatility regime appears.</p>
                      <div className="vertical-bars">
                        {volatilityRows.map((row) => (
                          <div key={row.label} className="vertical-bar-item">
                            <div className="vertical-bar-track">
                              <div
                                className="vertical-bar-fill"
                                style={{
                                  height: `${(row.value / maxVolatility) * 100}%`,
                                  background: VOLATILITY_COLORS[row.label] ?? "#adb5bd"
                                }}
                              />
                            </div>
                            <span className="vertical-bar-label">{row.label}</span>
                            <strong className="vertical-bar-value">{row.value}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="panel chart-panel">
                    <div className="chart-title">Intelligence Sources</div>
                    <p className="chart-caption">Which engine produced the insight record.</p>
                    <div className="bar-stack">
                      {sourceRows.map((row) => (
                        <div key={row.label} className="bar-row">
                          <span>{row.label}</span>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{ width: `${(row.value / maxSource) * 100}%` }}
                            />
                          </div>
                          <strong>{row.value}</strong>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="panel table-panel">
                    <div className="chart-title">Recent Intelligence Records</div>
                    <table className="forex-table">
                      <thead>
                        <tr>
                          <th>Timestamp</th>
                          <th>Regime</th>
                          <th>Session</th>
                          <th>Volatility</th>
                          <th>Confidence</th>
                          <th>Strategy</th>
                        </tr>
                      </thead>
                      <tbody>
                        {recentItems.slice(0, 20).map((row, idx) => (
                          <tr key={row.id ?? idx}>
                            <td>
                              {new Date(
                                row.scan_timestamp || row.alert_timestamp || Date.now()
                              ).toLocaleString("en-GB", {
                                day: "2-digit",
                                month: "short",
                                hour: "2-digit",
                                minute: "2-digit"
                              })}
                            </td>
                            <td>{row.regime ?? "-"}</td>
                            <td>{row.session ?? "-"}</td>
                            <td>{row.volatility_level ?? "-"}</td>
                            <td>
                              {row.regime_confidence != null
                                ? Number(row.regime_confidence).toFixed(2)
                                : "-"}
                            </td>
                            <td>{row.strategy ?? row.recommended_strategy ?? "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : null}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
