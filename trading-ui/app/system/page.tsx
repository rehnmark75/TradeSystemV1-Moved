/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import KpiTile from "../../components/ops/KpiTile";
import LiveBadge from "../../components/ops/LiveBadge";
import SectionTabs from "../../components/ops/SectionTabs";
import StatusPill, { type HealthState } from "../../components/ops/StatusPill";
import HealthCheckGrid, { type HealthCheckItem } from "../../components/ops/HealthCheckGrid";

const BASE = "/trading";
const REFRESH_MS = 10_000;

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "health", label: "Health" },
  { id: "streams", label: "Streams" },
  { id: "logs", label: "Logs" },
  { id: "analytics", label: "Analytics" },
];

interface OpsEvent {
  id?: string;
  message: string;
  severity?: string;
  at?: string;
  created_at?: string;
  service?: string;
  kind?: string;
  [key: string]: unknown;
}

interface BackfillStatus {
  in_flight?: number;
  queued?: number;
  completed_last_24h?: number;
  failed_last_24h?: number;
  last_run?: string;
  status?: string;
  running?: boolean;
  stats?: Record<string, unknown>;
  [key: string]: unknown;
}

interface StreamSummary {
  totals?: { streams?: number; active?: number; stalled?: number; epics?: number };
  stream_running?: boolean;
  active_epics?: number;
  subscriptions?: number;
  status?: string;
  [key: string]: unknown;
}

interface EpicCandle {
  epic?: string;
  open?: number; high?: number; low?: number; close?: number; volume?: number;
  timestamp?: string; start_time?: string;
  age_seconds?: number; stale?: boolean;
}

interface LogLine {
  at?: string; timestamp?: string; level: string;
  service?: string; source?: string; message: string;
}

interface StreamHealth {
  status?: string;
  components?: Record<string, unknown>;
  [key: string]: unknown;
}

function useAutoRefresh<T>(url: string, intervalMs = REFRESH_MS) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetch_ = useCallback(async () => {
    if (document.hidden) return;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const d = await res.json();
      setData(d);
      setError(null);
      setLastUpdated(new Date());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    fetch_();
    const id = setInterval(fetch_, intervalMs);
    return () => clearInterval(id);
  }, [fetch_, intervalMs]);

  return { data, loading, error, lastUpdated, refetch: fetch_ };
}

function fmtTime(ts?: string) {
  if (!ts) return "—";
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

function fmtPrice(n?: number) {
  if (n == null) return "—";
  return n.toFixed(5);
}

function severityColor(s?: string) {
  if (!s) return "var(--muted)";
  if (/crit/i.test(s)) return "#8b1e2b";
  if (/error|err/i.test(s)) return "var(--bad)";
  if (/warn/i.test(s)) return "var(--warn)";
  if (/info/i.test(s)) return "#1971c2";
  return "var(--muted)";
}

function toHealthState(s?: string): HealthState {
  if (s == null) return "unknown";
  const v = String(s).toLowerCase();
  if (/healthy|ok|running|up/.test(v) || v === "true") return "healthy";
  // market_closed = intentional idle, not a fault
  if (/degraded|warn|market_closed|closed|idle|issues/.test(v)) return "degraded";
  if (/down|error|err|stopped|unavailable/.test(v) || v === "false") return "down";
  return "unknown";
}

// ─── Sub-page: Overview ──────────────────────────────────────────────────────

function OverviewTab() {
  const { data: summary, loading: sl } = useAutoRefresh<StreamSummary>(`${BASE}/api/stream/summary`);
  const { data: backfill, loading: bl } = useAutoRefresh<BackfillStatus>(`${BASE}/api/stream/backfill/status`);
  const { data: gaps } = useAutoRefresh<unknown[]>(`${BASE}/api/stream/backfill/gaps`, 30_000);
  const { data: ops, loading: ol } = useAutoRefresh<OpsEvent[]>(`${BASE}/api/stream/operations/recent?limit=60`);
  const { data: recentAlerts } = useAutoRefresh<OpsEvent[]>(`${BASE}/api/stream/alerts/recent?limit=20`);

  const active = summary?.totals?.active ?? summary?.active_epics ?? (summary?.stream_running ? 1 : 0);
  const stalled = summary?.totals?.stalled ?? 0;
  const epics = summary?.totals?.epics ?? summary?.subscriptions ?? 0;

  return (
    <div>
      {/* KPI Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(160px,1fr))", gap: "10px", marginBottom: "20px" }}>
        <KpiTile label="Active Streams" value={sl ? "…" : active} accent="var(--good)" />
        <KpiTile label="Stalled Streams" value={sl ? "…" : stalled} accent={stalled ? "var(--warn)" : "var(--border)"} />
        <KpiTile label="Tracked Epics" value={sl ? "…" : epics} />
        <KpiTile label="Backfill In-Flight" value={bl ? "…" : backfill?.in_flight ?? 0} accent={backfill?.in_flight ? "var(--accent)" : "var(--border)"} />
        <KpiTile label="Gaps Detected" value={(gaps ?? []).length} accent={(gaps ?? []).length > 0 ? "var(--warn)" : "var(--border)"} />
        <KpiTile label="Completed 24h" value={bl ? "…" : backfill?.completed_last_24h ?? 0} />
      </div>

      {/* Backfill + Stream Summary */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "20px" }}>
        <div className="panel">
          <h3 style={{ margin: "0 0 12px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Stream Status</h3>
          {sl ? <p style={{ color: "var(--muted)" }}>Loading…</p> : (
            <div style={{ display: "grid", gap: "8px" }}>
              {[
                ["Status", summary?.status ?? (summary?.stream_running ? "running" : "—")],
                ["Active Epics", String(active)],
                ["Subscriptions", String(summary?.subscriptions ?? epics)],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ color: "var(--muted)" }}>{k}</span>
                  <span style={{ fontWeight: 600 }}>{v as string}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="panel">
          <h3 style={{ margin: "0 0 12px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Backfill</h3>
          {bl ? <p style={{ color: "var(--muted)" }}>Loading…</p> : (
            <div style={{ display: "grid", gap: "8px" }}>
              {[
                ["In-flight", String(backfill?.in_flight ?? 0)],
                ["Queued", String(backfill?.queued ?? 0)],
                ["Completed 24h", String(backfill?.completed_last_24h ?? 0)],
                ["Failed 24h", String(backfill?.failed_last_24h ?? 0)],
                ["Last run", fmtTime(backfill?.last_run)],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", padding: "5px 0", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ color: "var(--muted)" }}>{k}</span>
                  <span style={{ fontWeight: 600, color: k === "Failed 24h" && v !== "0" ? "var(--bad)" : undefined }}>{v}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Recent Alerts */}
      {(recentAlerts ?? []).length > 0 && (
        <div style={{ marginBottom: "20px" }}>
          <h3 style={{ margin: "0 0 10px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Recent Alerts</h3>
          {(recentAlerts ?? []).slice(0, 5).map((a, i) => (
            <div key={i} style={{
              display: "flex", gap: "10px", alignItems: "flex-start",
              padding: "8px 12px", borderRadius: "7px",
              background: `${severityColor(a.severity)}08`, border: `1px solid ${severityColor(a.severity)}33`,
              marginBottom: "6px", fontSize: "0.83rem",
            }}>
              <span style={{ color: severityColor(a.severity), fontWeight: 700, minWidth: "46px", fontSize: "0.72rem", paddingTop: "2px", textTransform: "uppercase" }}>
                {a.severity ?? "info"}
              </span>
              <span style={{ flex: 1 }}>{a.message}</span>
              <span style={{ color: "var(--muted)", fontSize: "0.75rem", whiteSpace: "nowrap" }}>{fmtTime(a.at ?? a.created_at)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Operations Feed */}
      <div>
        <h3 style={{ margin: "0 0 10px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>
          Operations Feed
          {ol && <span style={{ marginLeft: "8px", fontSize: "0.75rem", color: "var(--muted)", fontWeight: 400 }}>loading…</span>}
        </h3>
        {!ol && (ops ?? []).length === 0 && <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>No recent operations.</p>}
        <div style={{ display: "flex", flexDirection: "column", gap: "4px", maxHeight: "400px", overflowY: "auto" }}>
          {(ops ?? []).map((op, i) => (
            <div key={i} style={{
              display: "flex", gap: "10px", alignItems: "flex-start",
              padding: "7px 12px", background: i % 2 ? "#fafaf8" : "transparent",
              borderRadius: "6px", fontSize: "0.82rem",
            }}>
              <span style={{ color: severityColor(op.severity), fontSize: "0.65rem", paddingTop: "3px", minWidth: "8px" }}>●</span>
              <span style={{ flex: 1, wordBreak: "break-word" }}>
                {op.service && <span style={{ color: "var(--muted)", marginRight: "6px" }}>[{op.service}]</span>}
                {op.message}
              </span>
              <span style={{ color: "var(--muted)", fontSize: "0.72rem", whiteSpace: "nowrap" }}>{fmtTime(op.at ?? op.created_at)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Sub-page: Health ─────────────────────────────────────────────────────────

function HealthTab() {
  const { data: health, loading } = useAutoRefresh<StreamHealth>(`${BASE}/api/stream/health`);
  const { data: streamStatus } = useAutoRefresh<Record<string, unknown>>(`${BASE}/api/stream/status`);

  const components = health?.components ?? {};
  const checks: HealthCheckItem[] = Object.entries(components).map(([k, v]) => ({
    service: k,
    status: toHealthState(typeof v === "object" && v !== null ? String((v as Record<string, unknown>).status ?? (v as Record<string, unknown>).healthy ?? v) : String(v)),
    latency_ms: typeof v === "object" && v !== null ? Number((v as Record<string, unknown>).latency_ms ?? 0) || undefined : undefined,
  }));

  // Prefer structured stream_health from the health endpoint; fall back to raw running flag
  const streamHealthStatus =
    (health as Record<string, unknown> | null)?.stream_health as Record<string, unknown> | undefined;
  const streamServiceState = streamHealthStatus?.status
    ? toHealthState(String(streamHealthStatus.status))
    : streamStatus != null
      ? toHealthState(streamStatus.running === true ? "running" : streamStatus.running === false ? "market_closed" : "unknown")
      : "unknown";
  checks.push({
    service: "stream-service",
    status: streamServiceState,
    error: streamHealthStatus?.details as string | undefined,
  });

  return (
    <div>
      <div style={{ marginBottom: "16px", display: "flex", gap: "10px", alignItems: "center" }}>
        <StatusPill state={toHealthState(health?.status as string)} label={health?.status as string ?? "unknown"} />
        {loading && <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>Refreshing…</span>}
      </div>
      <div className="panel">
        <HealthCheckGrid checks={checks} />
      </div>
      {health && (
        <details style={{ marginTop: "16px" }}>
          <summary style={{ cursor: "pointer", fontSize: "0.8rem", color: "var(--muted)" }}>Raw health data</summary>
          <pre style={{ fontSize: "0.72rem", background: "#fafaf8", padding: "12px", borderRadius: "8px", overflow: "auto", maxHeight: "300px" }}>
            {JSON.stringify(health, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

// ─── Sub-page: Streams ────────────────────────────────────────────────────────

const TRACKED_EPICS = [
  "CS.D.EURUSD.CEEM.IP", "CS.D.GBPUSD.MINI.IP", "CS.D.USDJPY.MINI.IP",
  "CS.D.AUDUSD.MINI.IP", "CS.D.USDCAD.MINI.IP", "CS.D.USDCHF.MINI.IP",
  "CS.D.EURJPY.MINI.IP", "CS.D.AUDJPY.MINI.IP", "CS.D.GBPJPY.MINI.IP",
];

function StreamsTab() {
  const [selectedEpic, setSelectedEpic] = useState(TRACKED_EPICS[0]);
  const { data: gaps } = useAutoRefresh<BackfillGapItem[]>(`${BASE}/api/stream/backfill/gaps`);
  const { data: candles, loading } = useAutoRefresh<EpicCandle[]>(
    `${BASE}/api/stream/candles/${encodeURIComponent(selectedEpic)}?limit=20`
  );
  const { data: latest } = useAutoRefresh<EpicCandle>(
    `${BASE}/api/stream/candle/latest/${encodeURIComponent(selectedEpic)}`,
    5000
  );

  const epicGaps = (gaps ?? []).filter(g => g.epic === selectedEpic);
  const ageS = latest?.age_seconds;
  const isStale = ageS !== undefined ? ageS > 600 : false;

  return (
    <div>
      <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "16px", flexWrap: "wrap" }}>
        <label style={{ fontSize: "0.85rem", color: "var(--muted)", fontWeight: 500 }}>Epic</label>
        <select value={selectedEpic} onChange={e => setSelectedEpic(e.target.value)} style={{ maxWidth: "280px" }}>
          {TRACKED_EPICS.map(e => <option key={e} value={e}>{e}</option>)}
        </select>
        {latest && (
          <StatusPill
            state={isStale ? "degraded" : "healthy"}
            label={isStale ? `Stale (${Math.round((ageS ?? 0) / 60)}m ago)` : "Fresh"}
          />
        )}
      </div>

      {/* Latest candle */}
      {latest && (
        <div className="panel" style={{ marginBottom: "16px" }}>
          <h3 style={{ margin: "0 0 12px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Latest Candle</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(140px,1fr))", gap: "10px" }}>
            {[
              ["Open", fmtPrice(latest.open)],
              ["High", fmtPrice(latest.high)],
              ["Low", fmtPrice(latest.low)],
              ["Close", fmtPrice(latest.close)],
              ["Volume", String(latest.volume ?? "—")],
              ["Time", fmtTime(latest.timestamp ?? latest.start_time)],
            ].map(([k, v]) => (
              <div key={k} style={{ background: "#fafaf8", border: "1px solid var(--border)", borderRadius: "8px", padding: "8px 12px" }}>
                <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginBottom: "2px" }}>{k}</div>
                <div style={{ fontFamily: "monospace", fontWeight: 600, fontSize: "0.9rem" }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gap alerts */}
      {epicGaps.length > 0 && (
        <div style={{ background: "#fff9e6", border: "1px solid #f0c040", borderRadius: "8px", padding: "10px 14px", marginBottom: "16px", fontSize: "0.83rem" }}>
          ⚠️ {epicGaps.length} data gap{epicGaps.length > 1 ? "s" : ""} detected for this epic
        </div>
      )}

      {/* Candles table */}
      <div className="panel">
        <h3 style={{ margin: "0 0 12px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Recent Candles</h3>
        {loading && <p style={{ color: "var(--muted)" }}>Loading…</p>}
        {!loading && (!candles || candles.length === 0) && <p style={{ color: "var(--muted)" }}>No candle data available.</p>}
        {(candles ?? []).length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem", fontFamily: "monospace" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid var(--border)" }}>
                  {["Time", "Open", "High", "Low", "Close", "Volume"].map(h => (
                    <th key={h} style={{ padding: "6px 10px", textAlign: "right", color: "var(--muted)", fontWeight: 600, fontSize: "0.75rem" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(candles ?? []).map((c, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border)", background: i % 2 ? "#fafaf8" : "transparent" }}>
                    <td style={{ padding: "5px 10px", color: "var(--muted)" }}>{fmtTime(c.timestamp ?? c.start_time)}</td>
                    <td style={{ padding: "5px 10px", textAlign: "right" }}>{fmtPrice(c.open)}</td>
                    <td style={{ padding: "5px 10px", textAlign: "right", color: "var(--good)" }}>{fmtPrice(c.high)}</td>
                    <td style={{ padding: "5px 10px", textAlign: "right", color: "var(--bad)" }}>{fmtPrice(c.low)}</td>
                    <td style={{ padding: "5px 10px", textAlign: "right", fontWeight: 600 }}>{fmtPrice(c.close)}</td>
                    <td style={{ padding: "5px 10px", textAlign: "right" }}>{c.volume ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

interface BackfillGapItem {
  epic?: string;
  timeframe?: string;
  gap_start?: string;
  gap_end?: string;
  missing_bars?: number;
}

// ─── Sub-page: Logs ───────────────────────────────────────────────────────────

function LogsTab() {
  const [query, setQuery] = useState("");
  const [level, setLevel] = useState("");
  const [service, setService] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<LogLine[] | null>(null);
  const [searchErr, setSearchErr] = useState<string | null>(null);

  const { data: recent, loading: rl } = useAutoRefresh<LogLine[]>(
    `${BASE}/api/stream/logs/recent?limit=100${level ? `&level=${level}` : ""}${service ? `&service=${service}` : ""}`
  );

  const doSearch = async () => {
    if (!query && !level && !service) return;
    setSearching(true);
    setSearchErr(null);
    try {
      const qs = new URLSearchParams();
      if (query) qs.set("q", query);
      if (level) qs.set("level", level);
      if (service) qs.set("service", service);
      qs.set("limit", "300");
      const res = await fetch(`${BASE}/api/stream/logs/search/?${qs}`);
      if (!res.ok) throw new Error(`${res.status}`);
      const d = await res.json();
      setResults(d.lines ?? d);
    } catch (e) {
      setSearchErr(String(e));
    } finally {
      setSearching(false);
    }
  };

  const displayLines = results ?? recent ?? [];
  const levelColor = (l: string) => {
    if (/error|crit/i.test(l)) return "#f85149";
    if (/warn/i.test(l)) return "#e3b341";
    if (/info/i.test(l)) return "#58a6ff";
    return "#8b949e";
  };

  return (
    <div>
      {/* Search controls */}
      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "flex-end", marginBottom: "16px" }}>
        <div style={{ flex: "1 1 200px" }}>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--muted)", marginBottom: "4px" }}>Search (substring or re:pattern)</label>
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && doSearch()}
            placeholder="e.g. ERROR or re:signal.*EURUSD"
            style={{ width: "100%", fontFamily: "monospace", fontSize: "0.85rem" }}
          />
        </div>
        <div>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--muted)", marginBottom: "4px" }}>Level</label>
          <select value={level} onChange={e => setLevel(e.target.value)} style={{ minWidth: "100px" }}>
            <option value="">All</option>
            <option value="debug">Debug</option>
            <option value="info">Info</option>
            <option value="warn">Warn</option>
            <option value="error">Error</option>
          </select>
        </div>
        <div>
          <label style={{ display: "block", fontSize: "0.75rem", color: "var(--muted)", marginBottom: "4px" }}>Service</label>
          <select value={service} onChange={e => setService(e.target.value)} style={{ minWidth: "110px" }}>
            <option value="">stream (default)</option>
            <option value="worker">task-worker</option>
            <option value="worker-live">task-worker-live</option>
            <option value="dev">fastapi-dev</option>
            <option value="live">fastapi-live</option>
            <option value="scanner">stock-scanner</option>
          </select>
        </div>
        <button
          onClick={doSearch}
          disabled={searching}
          style={{ padding: "8px 18px", background: "var(--accent)", color: "#fff", border: "none", borderRadius: "8px", cursor: "pointer", fontWeight: 600, height: "38px" }}
        >
          {searching ? "…" : "Search"}
        </button>
        {results && (
          <button onClick={() => setResults(null)} style={{ padding: "8px 14px", border: "1px solid var(--border)", borderRadius: "8px", background: "transparent", cursor: "pointer", height: "38px" }}>
            Clear
          </button>
        )}
      </div>

      {searchErr && <p style={{ color: "var(--bad)", fontSize: "0.85rem" }}>Error: {searchErr}</p>}

      {/* Log viewer */}
      <div style={{
        background: "#0d1117", borderRadius: "10px", padding: "12px 16px",
        fontFamily: "monospace", fontSize: "0.76rem", lineHeight: 1.6,
        maxHeight: "520px", overflowY: "auto",
        border: "1px solid #30363d",
      }}>
        {rl && !results && <div style={{ color: "#8b949e" }}>Loading recent logs…</div>}
        {!rl && displayLines.length === 0 && <div style={{ color: "#8b949e" }}>No log lines available.</div>}
        {displayLines.map((line, i) => {
          const ts = line.at ?? line.timestamp ?? "";
          const lv = (line.level ?? "").toUpperCase();
          const msg = line.message;
          const svc = line.service ?? line.source ?? "";
          return (
            <div key={i} style={{ display: "flex", gap: "8px", borderBottom: "1px solid #161b22", padding: "2px 0" }}>
              <span style={{ color: "#484f58", minWidth: "80px", flexShrink: 0 }}>
                {ts ? new Date(ts).toLocaleTimeString() : ""}
              </span>
              <span style={{ color: levelColor(lv), minWidth: "40px", flexShrink: 0, fontWeight: 700 }}>{lv}</span>
              {svc && <span style={{ color: "#58a6ff", flexShrink: 0 }}>[{svc}]</span>}
              <span style={{ color: "#c9d1d9", wordBreak: "break-all" }}>{msg}</span>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: "0.72rem", color: "var(--muted)", marginTop: "6px" }}>
        {results ? `${results.length} search results` : `${displayLines.length} recent lines (live)`}
        {" · "}Log search requires <code>fastapi-stream /logs/search</code> endpoint (Phase 6 — see follow-ups if not yet deployed)
      </div>
    </div>
  );
}

// ─── Sub-page: Analytics ─────────────────────────────────────────────────────

function AnalyticsTab() {
  const { data: ops } = useAutoRefresh<OpsEvent[]>(`${BASE}/api/stream/operations/recent?limit=200`, 30_000);
  const { data: alerts } = useAutoRefresh<OpsEvent[]>(`${BASE}/api/stream/alerts/recent?limit=200`, 30_000);

  const byHour = Array.from({ length: 24 }, (_, h) => ({
    hour: h,
    ops: (ops ?? []).filter(o => new Date(o.at ?? o.created_at ?? "").getUTCHours() === h).length,
    alerts: (alerts ?? []).filter(a => new Date(a.at ?? a.created_at ?? "").getUTCHours() === h).length,
  }));

  const maxCount = Math.max(1, ...byHour.map(h => h.ops + h.alerts));

  const bySeverity = ["info", "warning", "error", "critical"].map(s => ({
    label: s,
    count: (alerts ?? []).filter(a => (a.severity ?? "info") === s).length,
  }));

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", alignItems: "start" }}>
        {/* Hourly ops histogram */}
        <div className="panel">
          <h3 style={{ margin: "0 0 14px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Hourly Activity (UTC, last 200 events)</h3>
          <div style={{ display: "flex", alignItems: "flex-end", gap: "3px", height: "80px" }}>
            {byHour.map(({ hour, ops: o, alerts: a }) => (
              <div key={hour} title={`${hour}:00 — ops:${o} alerts:${a}`}
                style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: "0", height: "100%" }}>
                <div style={{ flex: 1, width: "100%", display: "flex", flexDirection: "column", justifyContent: "flex-end", gap: "1px" }}>
                  {a > 0 && <div style={{ width: "100%", height: `${Math.round((a / maxCount) * 60)}px`, background: "var(--warn)", opacity: 0.7, borderRadius: "2px 2px 0 0" }} />}
                  {o > 0 && <div style={{ width: "100%", height: `${Math.round((o / maxCount) * 60)}px`, background: "var(--accent)", opacity: 0.6, borderRadius: a > 0 ? "0" : "2px 2px 0 0" }} />}
                </div>
                {hour % 6 === 0 && <span style={{ fontSize: "0.6rem", color: "var(--muted)", marginTop: "2px" }}>{hour}h</span>}
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: "12px", marginTop: "8px", fontSize: "0.72rem", color: "var(--muted)" }}>
            <span><span style={{ color: "var(--accent)" }}>■</span> Ops</span>
            <span><span style={{ color: "var(--warn)" }}>■</span> Alerts</span>
          </div>
        </div>

        {/* Alert severity breakdown */}
        <div className="panel">
          <h3 style={{ margin: "0 0 14px", fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem" }}>Alert Severity Breakdown</h3>
          <div style={{ display: "grid", gap: "8px" }}>
            {bySeverity.map(({ label, count }) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ minWidth: "60px", fontSize: "0.78rem", fontWeight: 600, color: severityColor(label), textTransform: "capitalize" }}>{label}</span>
                <div style={{ flex: 1, height: "16px", background: "#e4ddd2", borderRadius: "8px", overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${Math.round((count / Math.max(1, (alerts ?? []).length)) * 100)}%`, background: severityColor(label), borderRadius: "8px", transition: "width 0.4s ease" }} />
                </div>
                <span style={{ minWidth: "28px", textAlign: "right", fontSize: "0.82rem", fontWeight: 700 }}>{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main page ────────────────────────────────────────────────────────────────

function SystemStatusInner() {
  const [activeTab, setActiveTab] = useState("overview");

  const { data: streamStatus, lastUpdated, refetch, loading: sl } =
    useAutoRefresh<Record<string, unknown>>(`${BASE}/api/stream/status`);
  const { data: backfill } = useAutoRefresh<BackfillStatus>(`${BASE}/api/stream/backfill/status`);
  const { data: health } = useAutoRefresh<StreamHealth>(`${BASE}/api/stream/health`, 15_000);

  const streamHealthStatus = (health as Record<string, unknown> | null)?.stream_health as Record<string, unknown> | undefined;
  const streamState = streamHealthStatus?.status
    ? toHealthState(String(streamHealthStatus.status))
    : toHealthState(streamStatus?.running === true ? "running" : streamStatus?.running === false ? "market_closed" : "unknown");
  const backfillState = toHealthState(backfill?.status ?? (backfill?.running ? "running" : "unknown"));
  const healthState = toHealthState(health?.status as string);

  return (
    <div className="page">
      {/* Topbar */}
      <div className="topbar">
        <Link href="/" className="brand">Trading Hub</Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/infrastructure">Infrastructure</Link>
          <Link href="/system" style={{ fontWeight: 700, color: "var(--accent)" }}>System Status</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      {/* Header */}
      <div className="header" style={{ marginBottom: "20px" }}>
        <div>
          <h1 style={{ margin: 0 }}>System Status</h1>
          <p style={{ margin: "6px 0 0", opacity: 0.85, fontSize: "0.9rem" }}>
            Stream health, candle data, operations, and log intelligence
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <LiveBadge lastUpdated={lastUpdated} />
          <button onClick={refetch} className="refresh-btn" title="Refresh">↻</button>
        </div>
      </div>

      {/* Service health strip */}
      <div style={{
        display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center",
        background: "var(--panel)", border: "1px solid var(--border)",
        borderRadius: "10px", padding: "10px 16px", marginBottom: "20px",
        fontSize: "0.82rem",
      }}>
        <span style={{ color: "var(--muted)", fontWeight: 600, marginRight: "4px" }}>Services:</span>
        <StatusPill state={streamState} label={`Stream ${streamStatus?.running ? "running" : "stopped"}`} size="sm" />
        <StatusPill state={backfillState} label={`Backfill ${backfill?.running ? "running" : (backfill?.status ?? "unknown")}`} size="sm" />
        <StatusPill state={healthState} label={`Health ${health?.status ?? "unknown"}`} size="sm" />
      </div>

      {/* Tabs */}
      <SectionTabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

      {/* Tab content */}
      {activeTab === "overview" && <OverviewTab />}
      {activeTab === "health" && <HealthTab />}
      {activeTab === "streams" && <StreamsTab />}
      {activeTab === "logs" && <LogsTab />}
      {activeTab === "analytics" && <AnalyticsTab />}
    </div>
  );
}

export default function SystemStatusPage() {
  return <SystemStatusInner />;
}
