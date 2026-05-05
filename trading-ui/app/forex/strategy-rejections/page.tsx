/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { colorForStage, describeStage, UNKNOWN_STAGE_COLOR } from "../../../lib/rejectionStyles";

// ─── Types ───────────────────────────────────────────────────────────────────

type Options = {
  strategies: string[];
  epics: Array<{ epic: string; pair: string }>;
  stages: string[];
  hours: number[];
};

type Stats = {
  total: number;
  unique_pairs: number;
  most_rejected_pair: string;
  by_strategy: Record<string, number>;
  by_stage: Record<string, number>;
  by_hour: Record<string, number>;
  by_session: Record<string, number>;
  by_direction: Record<string, number>;
};

type TopStageRow = {
  strategy: string;
  stage: string;
  total: string;
  pairs_affected: string;
  pct_of_strategy: string;
};

type RejectionRow = {
  strategy: string;
  epic: string;
  pair: string | null;
  scan_timestamp: string;
  stage: string;
  reason: string | null;
  direction: string | null;
  hour_utc: number | null;
  session: string | null;
  details: Record<string, unknown> | null;
};

// ─── Constants ───────────────────────────────────────────────────────────────

const DAY_OPTIONS = [1, 3, 7, 14, 30, 60, 90];
const TABS = ["Overview", "Top Stages", "Time Analysis", "Per Pair", "Raw Log"] as const;
type Tab = (typeof TABS)[number];

const STRATEGY_COLORS: Record<string, string> = {
  MEAN_REVERSION: "#60a5fa",
  IMPULSE_FADE: "#f59e0b",
  XAU_GOLD: "#fbbf24",
};

const SESSION_ORDER = ["london", "overlap", "new_york", "asian"];

// ─── Helpers ─────────────────────────────────────────────────────────────────

const formatTs = (v: string) => {
  const d = new Date(v);
  if (Number.isNaN(d.valueOf())) return v;
  return d.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
};

function Bar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div
        style={{
          flex: 1,
          height: 10,
          background: "#1e293b",
          borderRadius: 5,
          overflow: "hidden",
        }}
      >
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 5 }} />
      </div>
      <span style={{ color: "#94a3b8", fontSize: 12, minWidth: 36, textAlign: "right" }}>{value}</span>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        background: "#1e293b",
        borderRadius: 8,
        padding: "12px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      <span style={{ color: "#64748b", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>
      <span style={{ color: "#f1f5f9", fontSize: 22, fontWeight: 700 }}>{value}</span>
    </div>
  );
}

function StagePill({ stage }: { stage: string }) {
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 7px",
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        background: colorForStage(stage) + "22",
        color: colorForStage(stage),
        border: `1px solid ${colorForStage(stage)}44`,
      }}
    >
      {stage}
    </span>
  );
}

function JsonExpand({ data }: { data: Record<string, unknown> | null }) {
  const [open, setOpen] = useState(false);
  if (!data || Object.keys(data).length === 0) return <span style={{ color: "#475569" }}>—</span>;
  return (
    <div>
      <button
        onClick={() => setOpen((x) => !x)}
        style={{
          background: "none",
          border: "none",
          color: "#60a5fa",
          cursor: "pointer",
          fontSize: 11,
          padding: 0,
        }}
      >
        {open ? "▲ hide" : "▼ details"}
      </button>
      {open && (
        <pre
          style={{
            marginTop: 4,
            padding: 8,
            background: "#0f172a",
            borderRadius: 4,
            fontSize: 11,
            color: "#94a3b8",
            overflowX: "auto",
          }}
        >
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────────────────────

export default function StrategyRejectionsPage() {
  const [days, setDays] = useState(7);
  const [strategy, setStrategy] = useState("ALL");
  const [epic, setEpic] = useState("ALL");
  const [stage, setStage] = useState("ALL");
  const [direction, setDirection] = useState("ALL");
  const [activeTab, setActiveTab] = useState<Tab>("Overview");

  const [options, setOptions] = useState<Options | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [topStages, setTopStages] = useState<TopStageRow[]>([]);
  const [rows, setRows] = useState<RejectionRow[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);

  const [loadingStats, setLoadingStats] = useState(false);
  const [loadingRows, setLoadingRows] = useState(false);

  const LIMIT = 50;

  // ── Fetch options once ──────────────────────────────────────────────────
  useEffect(() => {
    fetch("/api/forex/strategy-rejections/options")
      .then((r) => r.json())
      .then(setOptions)
      .catch(console.error);
  }, []);

  // ── Fetch stats whenever filters change ─────────────────────────────────
  useEffect(() => {
    setLoadingStats(true);
    const params = new URLSearchParams({ days: String(days), strategy, epic });
    Promise.all([
      fetch(`/api/forex/strategy-rejections/stats?${params}`).then((r) => r.json()),
      fetch(`/api/forex/strategy-rejections/top-stages?${params}`).then((r) => r.json()),
    ])
      .then(([s, ts]) => {
        setStats(s);
        setTopStages(ts.rows ?? []);
      })
      .catch(console.error)
      .finally(() => setLoadingStats(false));
  }, [days, strategy, epic]);

  // ── Fetch raw list for Raw Log & Per Pair tabs ───────────────────────────
  useEffect(() => {
    if (activeTab !== "Raw Log" && activeTab !== "Per Pair") return;
    setLoadingRows(true);
    const params = new URLSearchParams({
      days: String(days),
      strategy,
      epic,
      stage,
      direction,
      limit: String(LIMIT),
      offset: String(offset),
    });
    fetch(`/api/forex/strategy-rejections/list?${params}`)
      .then((r) => r.json())
      .then((d) => {
        setRows(d.rows ?? []);
        setTotal(d.total ?? 0);
      })
      .catch(console.error)
      .finally(() => setLoadingRows(false));
  }, [activeTab, days, strategy, epic, stage, direction, offset]);

  // Reset offset when filters change
  useEffect(() => {
    setOffset(0);
  }, [days, strategy, epic, stage, direction]);

  // ── Derived ─────────────────────────────────────────────────────────────
  const stageEntries = useMemo(() => {
    if (!stats?.by_stage) return [];
    return Object.entries(stats.by_stage).sort((a, b) => b[1] - a[1]);
  }, [stats]);

  const maxStageCount = stageEntries[0]?.[1] ?? 1;

  const hourEntries = useMemo(() => {
    if (!stats?.by_hour) return [];
    return Object.entries(stats.by_hour)
      .map(([h, c]) => [Number(h), c] as [number, number])
      .sort((a, b) => a[0] - b[0]);
  }, [stats]);

  const maxHourCount = Math.max(...hourEntries.map(([, c]) => c), 1);

  const strategyEntries = useMemo(() => {
    if (!stats?.by_strategy) return [];
    return Object.entries(stats.by_strategy).sort((a, b) => b[1] - a[1]);
  }, [stats]);

  const perPairEntries = useMemo(() => {
    const map: Record<string, Record<string, number>> = {};
    for (const row of rows) {
      if (!map[row.epic]) map[row.epic] = {};
      map[row.epic][row.stage] = (map[row.epic][row.stage] ?? 0) + 1;
    }
    return Object.entries(map).sort((a, b) => {
      const sumA = Object.values(a[1]).reduce((s, v) => s + v, 0);
      const sumB = Object.values(b[1]).reduce((s, v) => s + v, 0);
      return sumB - sumA;
    });
  }, [rows]);

  const topStagesByStrategy = useMemo(() => {
    const map: Record<string, TopStageRow[]> = {};
    for (const r of topStages) {
      if (!map[r.strategy]) map[r.strategy] = [];
      map[r.strategy].push(r);
    }
    return map;
  }, [topStages]);

  // ── Filter bar ──────────────────────────────────────────────────────────
  const filterBar = (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
      {/* Days */}
      <div style={{ display: "flex", gap: 4 }}>
        {DAY_OPTIONS.map((d) => (
          <button
            key={d}
            onClick={() => setDays(d)}
            style={{
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid",
              borderColor: days === d ? "#60a5fa" : "#334155",
              background: days === d ? "#1e3a5f" : "#1e293b",
              color: days === d ? "#93c5fd" : "#64748b",
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            {d}d
          </button>
        ))}
      </div>

      {/* Strategy */}
      <select
        value={strategy}
        onChange={(e) => setStrategy(e.target.value)}
        style={{
          background: "#1e293b",
          color: "#cbd5e1",
          border: "1px solid #334155",
          borderRadius: 6,
          padding: "4px 8px",
          fontSize: 13,
        }}
      >
        <option value="ALL">All strategies</option>
        {(options?.strategies ?? []).map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>

      {/* Epic */}
      <select
        value={epic}
        onChange={(e) => setEpic(e.target.value)}
        style={{
          background: "#1e293b",
          color: "#cbd5e1",
          border: "1px solid #334155",
          borderRadius: 6,
          padding: "4px 8px",
          fontSize: 13,
        }}
      >
        <option value="ALL">All pairs</option>
        {(options?.epics ?? []).map((e) => (
          <option key={e.epic} value={e.epic}>
            {e.pair ?? e.epic}
          </option>
        ))}
      </select>

      {/* Stage (Raw Log only) */}
      {activeTab === "Raw Log" && (
        <select
          value={stage}
          onChange={(e) => setStage(e.target.value)}
          style={{
            background: "#1e293b",
            color: "#cbd5e1",
            border: "1px solid #334155",
            borderRadius: 6,
            padding: "4px 8px",
            fontSize: 13,
          }}
        >
          <option value="ALL">All stages</option>
          {(options?.stages ?? []).map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      )}

      {/* Direction (Raw Log only) */}
      {activeTab === "Raw Log" && (
        <select
          value={direction}
          onChange={(e) => setDirection(e.target.value)}
          style={{
            background: "#1e293b",
            color: "#cbd5e1",
            border: "1px solid #334155",
            borderRadius: 6,
            padding: "4px 8px",
            fontSize: 13,
          }}
        >
          <option value="ALL">All directions</option>
          <option value="BUY">BUY</option>
          <option value="SELL">SELL</option>
        </select>
      )}
    </div>
  );

  // ── Sub-tab: Overview ────────────────────────────────────────────────────
  const overviewTab = (
    <div>
      {loadingStats ? (
        <p style={{ color: "#475569" }}>Loading…</p>
      ) : !stats ? null : (
        <>
          {/* KPI cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 12, marginBottom: 24 }}>
            <StatCard label="Total rejections" value={stats.total.toLocaleString()} />
            <StatCard label="Unique pairs" value={stats.unique_pairs} />
            <StatCard label="Most rejected" value={stats.most_rejected_pair} />
          </div>

          {/* Per-strategy breakdown */}
          {strategyEntries.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                By Strategy
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {strategyEntries.map(([strat, cnt]) => (
                  <div key={strat}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ color: STRATEGY_COLORS[strat] ?? "#94a3b8", fontWeight: 600, fontSize: 13 }}>{strat}</span>
                      <span style={{ color: "#64748b", fontSize: 12 }}>
                        {((cnt / stats.total) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Bar value={cnt} max={stats.total} color={STRATEGY_COLORS[strat] ?? "#94a3b8"} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top stages */}
          {stageEntries.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                Top Rejection Stages
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {stageEntries.slice(0, 12).map(([s, cnt]) => (
                  <div key={s}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                      <StagePill stage={s} />
                      <span style={{ color: "#64748b", fontSize: 12 }}>
                        {((cnt / stats.total) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Bar value={cnt} max={maxStageCount} color={colorForStage(s)} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* By session */}
          {stats.by_session && Object.keys(stats.by_session).length > 0 && (
            <div>
              <h3 style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 10 }}>
                By Session
              </h3>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {SESSION_ORDER.filter((s) => stats.by_session[s]).map((s) => (
                  <div
                    key={s}
                    style={{
                      background: "#1e293b",
                      borderRadius: 8,
                      padding: "8px 14px",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: 2,
                      minWidth: 80,
                    }}
                  >
                    <span style={{ color: "#64748b", fontSize: 11, textTransform: "capitalize" }}>{s.replace("_", " ")}</span>
                    <span style={{ color: "#f1f5f9", fontSize: 18, fontWeight: 700 }}>{stats.by_session[s]}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );

  // ── Sub-tab: Top Stages (per strategy) ───────────────────────────────────
  const topStagesTab = (
    <div>
      {topStages.length === 0 ? (
        <p style={{ color: "#475569" }}>No data for selected filters.</p>
      ) : (
        Object.keys(topStagesByStrategy).map((strat) => (
          <div key={strat} style={{ marginBottom: 32 }}>
            <h3
              style={{
                color: STRATEGY_COLORS[strat] ?? "#94a3b8",
                fontSize: 14,
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                marginBottom: 12,
                borderBottom: `2px solid ${STRATEGY_COLORS[strat] ?? "#334155"}`,
                paddingBottom: 6,
              }}
            >
              {strat}
            </h3>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
                  <th style={{ textAlign: "left", padding: "6px 0" }}>Stage</th>
                  <th style={{ textAlign: "left", padding: "6px 0" }}>Description</th>
                  <th style={{ textAlign: "right", padding: "6px 0" }}>Count</th>
                  <th style={{ textAlign: "right", padding: "6px 0" }}>Pairs</th>
                  <th style={{ textAlign: "right", padding: "6px 0" }}>% of strategy</th>
                </tr>
              </thead>
              <tbody>
                {topStagesByStrategy[strat].map((r) => (
                  <tr key={r.stage} style={{ borderBottom: "1px solid #0f172a" }}>
                    <td style={{ padding: "6px 0" }}>
                      <StagePill stage={r.stage} />
                    </td>
                    <td style={{ padding: "6px 8px", color: "#64748b", fontSize: 12 }}>
                      {describeStage(r.stage)}
                    </td>
                    <td style={{ textAlign: "right", padding: "6px 0", color: "#cbd5e1" }}>{r.total}</td>
                    <td style={{ textAlign: "right", padding: "6px 0", color: "#94a3b8" }}>{r.pairs_affected}</td>
                    <td style={{ textAlign: "right", padding: "6px 0", color: "#f59e0b" }}>{r.pct_of_strategy}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))
      )}
    </div>
  );

  // ── Sub-tab: Time Analysis ────────────────────────────────────────────────
  const timeTab = (
    <div>
      {hourEntries.length === 0 ? (
        <p style={{ color: "#475569" }}>No data for selected filters.</p>
      ) : (
        <>
          <p style={{ color: "#64748b", fontSize: 13, marginBottom: 16 }}>
            Rejection count by UTC hour. Tall bars during session window confirm the gate fires correctly;
            bars outside the window reveal unfiltered rejections.
          </p>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 120, paddingBottom: 20, position: "relative" }}>
            {Array.from({ length: 24 }, (_, h) => {
              const cnt = hourEntries.find(([hh]) => hh === h)?.[1] ?? 0;
              const pct = maxHourCount > 0 ? (cnt / maxHourCount) * 100 : 0;
              const isSessionHour = strategy === "IMPULSE_FADE" ? h >= 18 && h <= 22 : false;
              return (
                <div
                  key={h}
                  title={`${h}:00 UTC — ${cnt} rejections`}
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    height: "100%",
                    justifyContent: "flex-end",
                    gap: 2,
                  }}
                >
                  <div
                    style={{
                      width: "100%",
                      height: `${pct}%`,
                      minHeight: cnt > 0 ? 2 : 0,
                      background: isSessionHour ? "#f59e0b" : "#60a5fa",
                      borderRadius: 2,
                      opacity: cnt > 0 ? 1 : 0.15,
                    }}
                  />
                  <span style={{ color: "#475569", fontSize: 9 }}>{h}</span>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: 16, marginTop: 8, flexWrap: "wrap" }}>
            {hourEntries.slice(0, 8).map(([h, cnt]) => (
              <div key={h} style={{ color: "#64748b", fontSize: 12 }}>
                <span style={{ color: "#94a3b8", fontWeight: 600 }}>{String(h).padStart(2, "0")}:00</span> — {cnt}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );

  // ── Sub-tab: Per Pair ────────────────────────────────────────────────────
  const perPairTab = (
    <div>
      {loadingRows ? (
        <p style={{ color: "#475569" }}>Loading…</p>
      ) : perPairEntries.length === 0 ? (
        <p style={{ color: "#475569" }}>No data for selected filters.</p>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
              <th style={{ textAlign: "left", padding: "6px 0" }}>Pair / Epic</th>
              <th style={{ textAlign: "right", padding: "6px 0" }}>Total</th>
              <th style={{ textAlign: "left", padding: "6px 16px" }}>Top stages</th>
            </tr>
          </thead>
          <tbody>
            {perPairEntries.map(([ep, stageCounts]) => {
              const total = Object.values(stageCounts).reduce((s, v) => s + v, 0);
              const sorted = Object.entries(stageCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
              return (
                <tr key={ep} style={{ borderBottom: "1px solid #0f172a" }}>
                  <td style={{ padding: "8px 0", color: "#cbd5e1", fontFamily: "monospace", fontSize: 12 }}>{ep}</td>
                  <td style={{ textAlign: "right", padding: "8px 0", color: "#f1f5f9", fontWeight: 600 }}>{total}</td>
                  <td style={{ padding: "8px 16px" }}>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {sorted.map(([s, c]) => (
                        <span key={s} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <StagePill stage={s} />
                          <span style={{ color: "#64748b", fontSize: 11 }}>×{c}</span>
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );

  // ── Sub-tab: Raw Log ─────────────────────────────────────────────────────
  const rawLogTab = (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ color: "#64748b", fontSize: 13 }}>
          {total.toLocaleString()} rows
        </span>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - LIMIT))}
            style={{
              padding: "4px 12px",
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              color: offset === 0 ? "#334155" : "#94a3b8",
              cursor: offset === 0 ? "default" : "pointer",
              fontSize: 13,
            }}
          >
            ← Prev
          </button>
          <span style={{ color: "#64748b", fontSize: 13, alignSelf: "center" }}>
            {offset + 1}–{Math.min(offset + LIMIT, total)}
          </span>
          <button
            disabled={offset + LIMIT >= total}
            onClick={() => setOffset(offset + LIMIT)}
            style={{
              padding: "4px 12px",
              background: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 6,
              color: offset + LIMIT >= total ? "#334155" : "#94a3b8",
              cursor: offset + LIMIT >= total ? "default" : "pointer",
              fontSize: 13,
            }}
          >
            Next →
          </button>
        </div>
      </div>

      {loadingRows ? (
        <p style={{ color: "#475569" }}>Loading…</p>
      ) : rows.length === 0 ? (
        <p style={{ color: "#475569" }}>No rows for selected filters.</p>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ color: "#64748b", borderBottom: "1px solid #1e293b" }}>
                <th style={{ textAlign: "left", padding: "6px 0" }}>Time</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Strategy</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Pair</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Stage</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Dir</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Hour</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Session</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Reason</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>Details</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #0f172a" }}>
                  <td style={{ padding: "6px 0", color: "#64748b", whiteSpace: "nowrap" }}>{formatTs(row.scan_timestamp)}</td>
                  <td style={{ padding: "6px 8px", color: STRATEGY_COLORS[row.strategy] ?? "#94a3b8", fontWeight: 600, whiteSpace: "nowrap" }}>
                    {row.strategy}
                  </td>
                  <td style={{ padding: "6px 8px", color: "#cbd5e1", fontFamily: "monospace" }}>{row.pair ?? row.epic}</td>
                  <td style={{ padding: "6px 8px" }}>
                    <StagePill stage={row.stage} />
                  </td>
                  <td style={{ padding: "6px 8px", color: row.direction === "BUY" ? "#34d399" : row.direction === "SELL" ? "#f87171" : "#475569" }}>
                    {row.direction ?? "—"}
                  </td>
                  <td style={{ padding: "6px 8px", color: "#64748b" }}>{row.hour_utc ?? "—"}</td>
                  <td style={{ padding: "6px 8px", color: "#64748b", textTransform: "capitalize" }}>{row.session ?? "—"}</td>
                  <td style={{ padding: "6px 8px", color: "#94a3b8", maxWidth: 280 }}>{row.reason ?? "—"}</td>
                  <td style={{ padding: "6px 8px" }}>
                    <JsonExpand data={row.details} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="page-container">
      <ForexNav activeHref="/forex/strategy-rejections" />

      <div style={{ padding: "24px 0" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, flexWrap: "wrap", gap: 8 }}>
          <div>
            <h1 style={{ color: "#f1f5f9", fontSize: 22, fontWeight: 700, margin: 0 }}>Strategy Rejections</h1>
            <p style={{ color: "#475569", fontSize: 13, margin: "4px 0 0" }}>
              MEAN_REVERSION · IMPULSE_FADE · XAU_GOLD &nbsp;|&nbsp; For SMC_SIMPLE rejections use the{" "}
              <a href="/forex/smc-rejections" style={{ color: "#60a5fa", textDecoration: "underline" }}>SMC Rejections</a> tab.
            </p>
          </div>
        </div>

        {filterBar}

        {/* Sub-tab nav */}
        <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid #1e293b", paddingBottom: 0 }}>
          {TABS.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "8px 16px",
                background: "none",
                border: "none",
                borderBottom: activeTab === tab ? "2px solid #60a5fa" : "2px solid transparent",
                color: activeTab === tab ? "#93c5fd" : "#64748b",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: activeTab === tab ? 600 : 400,
                marginBottom: -1,
              }}
            >
              {tab}
            </button>
          ))}
        </div>

        {activeTab === "Overview" && overviewTab}
        {activeTab === "Top Stages" && topStagesTab}
        {activeTab === "Time Analysis" && timeTab}
        {activeTab === "Per Pair" && perPairTab}
        {activeTab === "Raw Log" && rawLogTab}
      </div>
    </div>
  );
}
