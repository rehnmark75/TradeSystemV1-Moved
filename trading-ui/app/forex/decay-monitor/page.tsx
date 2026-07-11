/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type SparkDay = { day: string; n: number; wins: number; net_pips: number | null };

type Cell = {
  strategy: string;
  epic: string;
  config_set: string;
  trip_source: string;
  auto_resume: boolean;
  baseline_pf: number | null;
  baseline_shadow_pf: number | null;
  baseline_shadow_wr: number | null;
  baseline_shadow_n: number | null;
  baseline_source: string | null;
  notes: string | null;
  pause_state: string;
  paused_at: string | null;
  pause_reason: string | null;
  resume_proposed_at: string | null;
  resume_proposal_count: number;
  rolling: {
    n: number;
    pf: number | null;
    win_rate: number | null;
    expectancy: number | null;
    pf_headroom: number | null;
    wr_headroom: number | null;
    would_trip: boolean;
  };
  spark: SparkDay[];
};

type EventRow = {
  id: number | null;
  event_type: string;
  strategy: string;
  epic: string;
  config_set: string;
  reason: string | null;
  metrics: Record<string, unknown> | null;
  created_at: string;
  notified: boolean;
};

type Params = {
  shadow_window: number;
  shadow_min_outcomes: number;
  shadow_trip_pf: number;
  shadow_trip_wr_drop: number;
  shadow_max_consecutive_losses: number;
};

type Payload = { meta: { env: string; params: Params }; cells: Cell[]; events: EventRow[] };

const fmt = (v: number | null, digits = 2) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
const pct = (v: number | null) =>
  v == null || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(0)}%`;
const fmtTime = (iso: string | null) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 16).replace("T", " ");
};
const shortEpic = (epic: string) => {
  const m = epic.match(/CS\.D\.([A-Z]+)\./);
  return m ? m[1] : epic;
};

const EVENT_STYLE: Record<string, { emoji: string; label: string }> = {
  trip: { emoji: "⏸️", label: "Trip" },
  pause: { emoji: "⏸️", label: "Paused" },
  dry_run_trip: { emoji: "🟡", label: "Dry-run trip" },
  resume_proposed: { emoji: "🔔", label: "Resume proposed" },
  resumed: { emoji: "✅", label: "Resumed" },
  flip_noop_error: { emoji: "🚨", label: "Flip failed" },
};

// Tiny inline sparkline: one bar per day, green up / red down, height by |net|.
function Spark({ days }: { days: SparkDay[] }) {
  if (!days.length) return <span style={{ opacity: 0.4 }}>—</span>;
  const maxAbs = Math.max(1, ...days.map((d) => Math.abs(d.net_pips ?? 0)));
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 22 }}>
      {days.map((d) => {
        const v = d.net_pips ?? 0;
        const h = Math.max(2, (Math.abs(v) / maxAbs) * 20);
        return (
          <div
            key={d.day}
            title={`${d.day}: ${d.n} signals, ${d.wins} wins, ${v > 0 ? "+" : ""}${v.toFixed(0)} pips`}
            style={{
              width: 4,
              height: h,
              background: v >= 0 ? "rgba(34,197,94,0.8)" : "rgba(239,68,68,0.8)",
              borderRadius: 1,
            }}
          />
        );
      })}
    </div>
  );
}

export default function ForexDecayMonitorPage() {
  const { environment } = useEnvironment();
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/decay-monitor/?env=${environment}`)
      .then((res) => res.json())
      .then((data) => {
        if (data?.error) throw new Error(data.error);
        setPayload(data);
      })
      .catch(() => setError("Failed to load decay monitor."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, [environment]);

  const cells = payload?.cells ?? [];
  const events = payload?.events ?? [];
  const params = payload?.meta?.params;
  // Dry-run state isn't queryable from the worker env here; infer from events:
  // if the newest trip-ish event is a dry_run_trip, the layer is observing only.
  const lastTrip = events.find((e) =>
    ["trip", "pause", "dry_run_trip"].includes(e.event_type)
  );
  const dryRunLikely = lastTrip?.event_type === "dry_run_trip";

  const stateBadge = (c: Cell) => {
    if (c.pause_state === "paused")
      return <span className="negative" style={{ fontWeight: 700 }}>⏸ PAUSED</span>;
    if (c.rolling.would_trip)
      return <span className="negative" style={{ fontWeight: 700 }}>⚠ TRIP CONDITION MET</span>;
    if (c.pause_state === "resumed")
      return <span className="positive">✓ resumed</span>;
    return <span className="positive">● active</span>;
  };

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
          <h1>Strategy Decay Monitor</h1>
          <p>
            Continuous invalidation layer: each enrolled (strategy · pair) cell&apos;s rolling{" "}
            <strong>ref-grid shadow performance</strong> (last {params?.shadow_window ?? 50} resolved
            outcomes from <code>monitor_only_outcomes</code>) is compared against its{" "}
            <strong>frozen enrollment baseline</strong>. Trip rule: PF &lt;{" "}
            {params?.shadow_trip_pf ?? 0.8} <strong>and</strong> win-rate more than{" "}
            {((params?.shadow_trip_wr_drop ?? 0.12) * 100).toFixed(0)}pp below baseline → the cell is
            auto-flipped to monitor-only (or logged, in dry-run). Ref-grid PF is a{" "}
            <strong>decay proxy, not live P&amp;L</strong> — breakeven WR is 40% on the 10/15-pip grid.
          </p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/decay-monitor" />

      <div className="panel">
        <div className="forex-controls">
          <button className="section-tab active" onClick={load}>
            Refresh
          </button>
          <div className="forex-badge">{cells.length} enrolled cells</div>
          {dryRunLikely ? (
            <div
              className="forex-badge"
              style={{ background: "rgba(234,179,8,0.25)", fontWeight: 600 }}
            >
              🟡 DRY-RUN — trips logged, nothing paused
            </div>
          ) : null}
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading decay monitor...</div>
        ) : (
          <>
            <h3 style={{ marginTop: 8 }}>Enrolled cells</h3>
            {cells.length === 0 ? (
              <div style={{ opacity: 0.7, padding: "12px 0" }}>
                No cells enrolled for <strong>{environment}</strong>. Enroll via{" "}
                <code>scripts/auto_pause_enroll.py</code> in task-worker.
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Strategy</th>
                      <th>Pair</th>
                      <th>Source</th>
                      <th>State</th>
                      <th style={{ textAlign: "right" }}>Base PF</th>
                      <th style={{ textAlign: "right" }}>Base WR</th>
                      <th style={{ textAlign: "right" }}>Now PF</th>
                      <th style={{ textAlign: "right" }}>Now WR</th>
                      <th style={{ textAlign: "right" }}>n</th>
                      <th style={{ textAlign: "right" }}>PF headroom</th>
                      <th style={{ textAlign: "right" }}>WR headroom</th>
                      <th>30d shadow</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cells.map((c) => {
                      const key = `${c.strategy}__${c.epic}`;
                      const thin =
                        params != null && c.rolling.n < params.shadow_min_outcomes;
                      return (
                        <tr key={key} style={c.pause_state === "paused" ? { opacity: 0.85 } : undefined}>
                          <td>{c.strategy}</td>
                          <td title={c.epic}>{shortEpic(c.epic)}</td>
                          <td>
                            {c.trip_source}
                            {c.auto_resume ? " · auto-resume" : ""}
                          </td>
                          <td title={c.pause_reason ?? undefined}>
                            {stateBadge(c)}
                            {c.pause_state === "paused" && c.paused_at ? (
                              <div style={{ fontSize: 11, opacity: 0.7 }}>
                                since {fmtTime(c.paused_at)}
                              </div>
                            ) : null}
                            {c.resume_proposed_at ? (
                              <div style={{ fontSize: 11, opacity: 0.7 }}>
                                🔔 resume proposed {fmtTime(c.resume_proposed_at)}
                              </div>
                            ) : null}
                          </td>
                          <td style={{ textAlign: "right" }} title={c.baseline_source ?? undefined}>
                            {fmt(c.baseline_shadow_pf ?? c.baseline_pf)}
                          </td>
                          <td style={{ textAlign: "right" }}>{pct(c.baseline_shadow_wr)}</td>
                          <td
                            style={{ textAlign: "right" }}
                            className={
                              c.rolling.pf != null && params != null
                                ? c.rolling.pf < params.shadow_trip_pf
                                  ? "negative"
                                  : "positive"
                                : ""
                            }
                          >
                            {fmt(c.rolling.pf)}
                          </td>
                          <td style={{ textAlign: "right" }}>{pct(c.rolling.win_rate)}</td>
                          <td style={{ textAlign: "right", opacity: thin ? 0.6 : 1 }}>
                            {c.rolling.n}
                            {thin ? " (thin)" : ""}
                          </td>
                          <td
                            style={{ textAlign: "right" }}
                            className={
                              c.rolling.pf_headroom != null && c.rolling.pf_headroom < 0
                                ? "negative"
                                : ""
                            }
                            title="rolling PF minus the 0.8 trip floor"
                          >
                            {fmt(c.rolling.pf_headroom)}
                          </td>
                          <td
                            style={{ textAlign: "right" }}
                            className={
                              c.rolling.wr_headroom != null && c.rolling.wr_headroom < 0
                                ? "negative"
                                : ""
                            }
                            title="rolling WR minus (baseline WR − 12pp)"
                          >
                            {c.rolling.wr_headroom == null
                              ? "—"
                              : `${c.rolling.wr_headroom >= 0 ? "+" : ""}${(c.rolling.wr_headroom * 100).toFixed(0)}pp`}
                          </td>
                          <td>
                            <Spark days={c.spark} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            <h3 style={{ marginTop: 24 }}>Recent events</h3>
            {events.length === 0 ? (
              <div style={{ opacity: 0.7, padding: "12px 0" }}>
                No auto-pause events yet for <strong>{environment}</strong>.
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table className="forex-table" style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>Time (UTC)</th>
                      <th>Event</th>
                      <th>Strategy</th>
                      <th>Pair</th>
                      <th>Reason</th>
                      <th>Metrics</th>
                      <th>Notified</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map((e) => {
                      const style = EVENT_STYLE[e.event_type] ?? { emoji: "ℹ️", label: e.event_type };
                      const m = e.metrics ?? {};
                      const pfV = typeof m.pf === "number" ? m.pf : null;
                      const wrV = typeof m.win_rate === "number" ? m.win_rate : null;
                      const nV = typeof m.n === "number" ? m.n : null;
                      return (
                        <tr key={e.id ?? `${e.created_at}-${e.strategy}`}>
                          <td>{fmtTime(e.created_at)}</td>
                          <td>
                            {style.emoji} {style.label}
                          </td>
                          <td>{e.strategy}</td>
                          <td title={e.epic}>{shortEpic(e.epic)}</td>
                          <td>{e.reason ?? "—"}</td>
                          <td>
                            {pfV != null ? `PF ${pfV.toFixed(2)}` : ""}
                            {wrV != null ? ` · WR ${(wrV * 100).toFixed(0)}%` : ""}
                            {nV != null ? ` · n=${nV}` : ""}
                          </td>
                          <td>{e.notified ? "✓" : "…"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
