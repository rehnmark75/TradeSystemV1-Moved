/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type FilterOptions = {
  stages: string[];
  pairs: string[];
  sessions: string[];
  table_exists: boolean;
};

type StatsPayload = {
  total: number;
  unique_pairs: number;
  near_misses: number;
  smc_conflicts: number;
  most_rejected_pair: string;
  by_stage: Record<string, number>;
};

type RejectionRow = {
  id: number;
  scan_timestamp: string;
  epic: string;
  pair: string | null;
  rejection_stage: string;
  rejection_reason: string | null;
  rejection_details: any;
  attempted_direction: string | null;
  current_price: number | null;
  market_hour: number | null;
  market_session: string | null;
  ema_distance_pips: number | null;
  price_position_vs_ema: string | null;
  atr_percentile: number | null;
  volume_ratio: number | null;
  confidence_score: number | null;
  sr_blocking_type: string | null;
  sr_blocking_distance_pips: number | null;
  sr_path_blocked_pct: number | null;
};

type ConflictPayload = {
  stats: { total: number; unique_pairs: number; sessions_affected: number };
  by_pair: Array<{ pair: string; count: number }>;
  by_session: Array<{ market_session: string; count: number }>;
  top_reasons: Array<{ rejection_reason: string; count: number }>;
  details: Array<RejectionRow>;
};

type OutcomePayload = {
  summary: Record<string, any>;
  win_rate_by_stage: Array<Record<string, any>>;
  suggestions: Record<string, any>;
};

const DAY_OPTIONS = [1, 3, 7, 14, 30, 60, 90];

const formatDateTime = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit"
  });
};

const parseDetails = (details: any) => {
  if (!details) return {};
  if (typeof details === "string") {
    try {
      return JSON.parse(details);
    } catch {
      return {};
    }
  }
  return typeof details === "object" ? details : {};
};

const stageColors: Record<string, string> = {
  CONFIDENCE: "#f08c00",
  TIER1_EMA: "#2f9e44",
  TIER2_SWING: "#228be6",
  SCALP_ENTRY_FILTER: "#d9480f",
  SMC_CONFLICT: "#e03131"
};

export default function SMCRejectionsPage() {
  const [days, setDays] = useState(7);
  const [stage, setStage] = useState("All");
  const [pair, setPair] = useState("All");
  const [session, setSession] = useState("All");
  const [activeTab, setActiveTab] = useState("stage");
  const [selectedPairDetail, setSelectedPairDetail] = useState<string>("All");

  const [options, setOptions] = useState<FilterOptions | null>(null);
  const [stats, setStats] = useState<StatsPayload | null>(null);
  const [rows, setRows] = useState<RejectionRow[]>([]);
  const [conflicts, setConflicts] = useState<ConflictPayload | null>(null);
  const [outcomes, setOutcomes] = useState<OutcomePayload | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/stock-scanner/api/forex/smc-rejections/options/")
      .then((res) => res.json())
      .then((data) => setOptions(data))
      .catch(() => setOptions(null));
  }, []);

  const loadData = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      days: String(days),
      stage,
      pair,
      session
    });

    Promise.all([
      fetch(`/stock-scanner/api/forex/smc-rejections/stats/?days=${days}`).then((res) => res.json()),
      fetch(`/stock-scanner/api/forex/smc-rejections/list/?${params.toString()}`).then((res) =>
        res.json()
      ),
      fetch(`/stock-scanner/api/forex/smc-rejections/conflicts/?days=${days}`).then((res) =>
        res.json()
      ),
      fetch(`/stock-scanner/api/forex/smc-rejections/outcomes/?days=${days}`).then((res) =>
        res.json()
      )
    ])
      .then(([statsPayload, listPayload, conflictPayload, outcomePayload]) => {
        setStats(statsPayload);
        setRows(listPayload.rows ?? []);
        setConflicts(conflictPayload);
        setOutcomes(outcomePayload);
      })
      .catch(() => setError("Failed to load rejection analytics."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  const stageCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    rows.forEach((row) => {
      counts[row.rejection_stage] = (counts[row.rejection_stage] || 0) + 1;
    });
    const fromRows = Object.entries(counts).map(([label, value]) => ({ label, value }));
    if (!fromRows.length && stats?.by_stage) {
      return Object.entries(stats.by_stage).map(([label, value]) => ({
        label,
        value: Number(value ?? 0)
      }));
    }
    return fromRows.sort((a, b) => b.value - a.value);
  }, [rows, stats?.by_stage]);

  const stageTotal = stageCounts.reduce((sum, row) => sum + row.value, 0) || 1;
  const stageGradient = stageCounts.length
    ? stageCounts
        .map((row, idx) => {
          const start = stageCounts.slice(0, idx).reduce((sum, r) => sum + r.value, 0);
          const end = start + row.value;
          return `${stageColors[row.label] ?? "#adb5bd"} ${(start / stageTotal) * 360}deg ${
            (end / stageTotal) * 360
          }deg`;
        })
        .join(", ")
    : "#f1e5d3 0deg 360deg";

  const topReasons = useMemo(() => {
    const counts: Record<string, number> = {};
    rows.forEach((row) => {
      const stageLabel = row.rejection_stage ?? "UNKNOWN";
      const reasonLabel = row.rejection_reason ?? "Unknown";
      const key = `${stageLabel}|||${reasonLabel}`;
      counts[key] = (counts[key] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([key, value]) => {
        const [stageLabel, reasonLabel] = key.split("|||");
        return { stage: stageLabel, reason: reasonLabel, value };
      })
      .sort((a, b) => b.value - a.value)
      .slice(0, 20);
  }, [rows]);

  const scalpFilterBreakdown = useMemo(() => {
    const counts: Record<string, number> = {};
    rows
      .filter((row) => row.rejection_stage === "SCALP_ENTRY_FILTER")
      .forEach((row) => {
        const details = parseDetails(row.rejection_details);
        const filterType = details.filter || details.filter_type || "Other";
        counts[filterType] = (counts[filterType] || 0) + 1;
      });
    return Object.entries(counts).map(([label, value]) => ({ label, value }));
  }, [rows]);

  const tier2Breakdown = useMemo(() => {
    const counts: Record<string, number> = {};
    rows
      .filter((row) => row.rejection_stage === "TIER2_SWING")
      .forEach((row) => {
        const reason = row.rejection_reason ?? "";
        let category = "Other";
        if (reason.includes("No recent swing")) category = "No Swing Levels";
        else if (reason.includes("Weak breakout body")) category = "Weak Breakout Body";
        else if (reason.includes("Weak breakout candle")) category = "Weak Breakout Candle";
        else if (reason.includes("Price did not break swing")) category = "Insufficient Breakout";
        counts[category] = (counts[category] || 0) + 1;
      });
    return Object.entries(counts).map(([label, value]) => ({ label, value }));
  }, [rows]);

  const sessionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    rows.forEach((row) => {
      if (!row.market_session) return;
      counts[row.market_session] = (counts[row.market_session] || 0) + 1;
    });
    return Object.entries(counts).map(([label, value]) => ({ label, value }));
  }, [rows]);

  const hourCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    rows.forEach((row) => {
      if (row.market_hour == null) return;
      const key = String(row.market_hour).padStart(2, "0");
      counts[key] = (counts[key] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([label, value]) => ({ label, value }))
      .sort((a, b) => Number(a.label) - Number(b.label));
  }, [rows]);

  const nearMisses = useMemo(() => {
    return rows.filter(
      (row) => row.rejection_stage === "CONFIDENCE" && (row.confidence_score ?? 0) >= 0.45
    );
  }, [rows]);

  const srBlocking = useMemo(() => {
    const counts: Record<string, number> = {};
    rows.forEach((row) => {
      if (!row.sr_blocking_type) return;
      counts[row.sr_blocking_type] = (counts[row.sr_blocking_type] || 0) + 1;
    });
    return Object.entries(counts).map(([label, value]) => ({ label, value }));
  }, [rows]);

  useEffect(() => {
    if (selectedPairDetail === "All" && options?.pairs?.length) {
      const firstPair = options.pairs.find((item) => item !== "All");
      if (firstPair) setSelectedPairDetail(firstPair);
    }
  }, [options?.pairs, selectedPairDetail]);

  const emaStats = useMemo(() => {
    const values = rows
      .map((row) => Number(row.ema_distance_pips ?? 0))
      .filter((value) => Number.isFinite(value) && value > 0);
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : 0;
    const median = sorted.length
      ? sorted[Math.floor(sorted.length / 2)]
      : 0;
    return { mean, median, count: values.length };
  }, [rows]);

  const atrStats = useMemo(() => {
    const values = rows
      .map((row) => Number(row.atr_percentile ?? 0))
      .filter((value) => Number.isFinite(value));
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : 0;
    const median = sorted.length
      ? sorted[Math.floor(sorted.length / 2)]
      : 0;
    return { mean, median, count: values.length };
  }, [rows]);

  const emaBuckets = useMemo(() => {
    const buckets = [
      { label: "0-10", min: 0, max: 10 },
      { label: "10-25", min: 10, max: 25 },
      { label: "25-50", min: 25, max: 50 },
      { label: "50-100", min: 50, max: 100 },
      { label: "100+", min: 100, max: Infinity }
    ];
    const counts = buckets.map((bucket) => ({ ...bucket, value: 0 }));
    rows.forEach((row) => {
      const value = Number(row.ema_distance_pips ?? 0);
      if (!Number.isFinite(value)) return;
      const bucket = counts.find((item) => value >= item.min && value < item.max);
      if (bucket) bucket.value += 1;
    });
    return counts;
  }, [rows]);

  const atrBuckets = useMemo(() => {
    const buckets = [
      { label: "0-20", min: 0, max: 20 },
      { label: "20-40", min: 20, max: 40 },
      { label: "40-60", min: 40, max: 60 },
      { label: "60-80", min: 60, max: 80 },
      { label: "80-100", min: 80, max: Infinity }
    ];
    const counts = buckets.map((bucket) => ({ ...bucket, value: 0 }));
    rows.forEach((row) => {
      const value = Number(row.atr_percentile ?? 0);
      if (!Number.isFinite(value)) return;
      const bucket = counts.find((item) => value >= item.min && value < item.max);
      if (bucket) bucket.value += 1;
    });
    return counts;
  }, [rows]);

  const maxEmaBucket = emaBuckets.length
    ? Math.max(...emaBuckets.map((row) => row.value))
    : 1;
  const maxAtrBucket = atrBuckets.length
    ? Math.max(...atrBuckets.map((row) => row.value))
    : 1;

  const pairContext = useMemo(() => {
    const buckets: Record<
      string,
      { count: number; emaTotal: number; atrTotal: number }
    > = {};
    rows.forEach((row) => {
      const key = row.pair ?? row.epic ?? "Unknown";
      if (!buckets[key]) {
        buckets[key] = { count: 0, emaTotal: 0, atrTotal: 0 };
      }
      const ema = Number(row.ema_distance_pips ?? 0);
      const atr = Number(row.atr_percentile ?? 0);
      if (Number.isFinite(ema)) buckets[key].emaTotal += ema;
      if (Number.isFinite(atr)) buckets[key].atrTotal += atr;
      buckets[key].count += 1;
    });
    return Object.entries(buckets)
      .map(([pair, data]) => ({
        pair,
        count: data.count,
        avg_ema: data.count ? data.emaTotal / data.count : 0,
        avg_atr: data.count ? data.atrTotal / data.count : 0
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 20);
  }, [rows]);

  const maxStageCount = stageCounts.length ? Math.max(...stageCounts.map((row) => row.value)) : 1;
  const maxSessionCount = sessionCounts.length
    ? Math.max(...sessionCounts.map((row) => row.value))
    : 1;
  const maxHourCount = hourCounts.length ? Math.max(...hourCounts.map((row) => row.value)) : 1;
  const maxConflictPair = conflicts?.by_pair?.length
    ? Math.max(...conflicts.by_pair.map((row) => row.count))
    : 1;
  const maxConflictSession = conflicts?.by_session?.length
    ? Math.max(...conflicts.by_session.map((row) => row.count))
    : 1;
  const maxSrBlocking = srBlocking.length ? Math.max(...srBlocking.map((row) => row.value)) : 1;

  if (options && !options.table_exists) {
    return (
      <div className="page">
        <div className="topbar">
          <Link href="/" className="brand">
            Trading Hub
          </Link>
        </div>
        <div className="panel">
          <div className="error">
            SMC Rejections table not yet created. Run the migration to enable this view.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Trading Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>SMC Simple Rejection Analysis</h1>
          <p>Why SMC signals are filtered out and what would have happened.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <div className="forex-nav">
        <Link href="/forex" className="forex-pill">
          Overview
        </Link>
        <Link href="/forex/strategy" className="forex-pill">
          Strategy Performance
        </Link>
        <Link href="/forex/trade-performance" className="forex-pill">
          Trade Performance
        </Link>
        <Link href="/forex/trade-analysis" className="forex-pill">
          Trade Analysis
        </Link>
        <Link href="/forex/performance-snapshot" className="forex-pill">
          Performance Snapshot
        </Link>
        <Link href="/forex/market-intelligence" className="forex-pill">
          Market Intelligence
        </Link>
        <Link href="/forex/smc-rejections" className="forex-pill">
          SMC Rejections
        </Link>
      </div>

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Time Period (days)</label>
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {DAY_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value} days
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Rejection Stage</label>
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              {(options?.stages ?? ["All"]).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Pair</label>
            <select value={pair} onChange={(e) => setPair(e.target.value)}>
              {(options?.pairs ?? ["All"]).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Session</label>
            <select value={session} onChange={(e) => setSession(e.target.value)}>
              {(options?.sessions ?? ["All"]).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={loadData}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}
        {loading ? (
          <div className="chart-placeholder">Loading rejection analytics...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Total Rejections
                <strong>{stats?.total ?? 0}</strong>
              </div>
              <div className="summary-card">
                Unique Pairs
                <strong>{stats?.unique_pairs ?? 0}</strong>
              </div>
              <div className="summary-card">
                Top Stage
                <strong>
                  {stageCounts.length ? stageCounts[0].label : "N/A"}
                </strong>
              </div>
              <div className="summary-card">
                Near-Misses
                <strong>{stats?.near_misses ?? 0}</strong>
              </div>
              <div className="summary-card">
                SMC Conflicts
                <strong>{stats?.smc_conflicts ?? 0}</strong>
              </div>
            </div>

            <div className="section-tabs">
              <button
                className={`section-tab ${activeTab === "stage" ? "active" : ""}`}
                onClick={() => setActiveTab("stage")}
              >
                Stage Breakdown
              </button>
              <button
                className={`section-tab ${activeTab === "conflicts" ? "active" : ""}`}
                onClick={() => setActiveTab("conflicts")}
              >
                SMC Conflicts
              </button>
              <button
                className={`section-tab ${activeTab === "outcomes" ? "active" : ""}`}
                onClick={() => setActiveTab("outcomes")}
              >
                Outcome Analysis
              </button>
              <button
                className={`section-tab ${activeTab === "sr" ? "active" : ""}`}
                onClick={() => setActiveTab("sr")}
              >
                S/R Path Blocking
              </button>
              <button
                className={`section-tab ${activeTab === "time" ? "active" : ""}`}
                onClick={() => setActiveTab("time")}
              >
                Time Analysis
              </button>
              <button
                className={`section-tab ${activeTab === "context" ? "active" : ""}`}
                onClick={() => setActiveTab("context")}
              >
                Market Context
              </button>
              <button
                className={`section-tab ${activeTab === "near" ? "active" : ""}`}
                onClick={() => setActiveTab("near")}
              >
                Near-Misses
              </button>
              <button
                className={`section-tab ${activeTab === "eff" ? "active" : ""}`}
                onClick={() => setActiveTab("eff")}
              >
                Scanner Efficiency
              </button>
            </div>

            {activeTab === "stage" ? (
              <>
                <div className="forex-grid">
                  <div className="panel chart-panel">
                    <div className="chart-title">Rejection Distribution by Stage</div>
                    <div className="donut-wrap">
                      <div
                        className="donut-chart"
                        style={{ background: `conic-gradient(${stageGradient})` }}
                      />
                      <div className="donut-legend">
                        {stageCounts.map((row) => (
                          <div key={row.label} className="legend-item">
                            <span
                              className="legend-dot"
                              style={{ background: stageColors[row.label] ?? "#adb5bd" }}
                            />
                            <span>{row.label}</span>
                            <strong>{row.value}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="panel chart-panel">
                    <div className="chart-title">Rejection Counts by Stage</div>
                    <div className="bar-stack">
                      {stageCounts.map((row) => (
                        <div key={row.label} className="bar-row">
                          <span>{row.label}</span>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{
                                width: `${(row.value / stageTotal) * 100}%`,
                                background: stageColors[row.label] ?? "#5bc0be"
                              }}
                            />
                          </div>
                          <strong>{row.value}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="panel table-panel pair-detail-panel">
                  <div className="chart-title">Top Rejection Reasons</div>
                  {topReasons.length ? (
                    <table className="forex-table">
                      <thead>
                        <tr>
                          <th>Stage</th>
                          <th>Reason</th>
                          <th>Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topReasons.map((row) => (
                          <tr key={`${row.stage}-${row.reason}`}>
                            <td>{row.stage}</td>
                            <td>{row.reason}</td>
                            <td>{row.value}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="muted">No rejection reasons found for these filters.</div>
                  )}
                </div>

                <div className="panel table-panel market-context-wide">
                  <div className="chart-title">Pair Rejection Detail</div>
                  <div className="forex-controls">
                    <div>
                      <label>Pair</label>
                      <select
                        value={selectedPairDetail}
                        onChange={(e) => setSelectedPairDetail(e.target.value)}
                      >
                        {(options?.pairs ?? ["All"]).map((value) => (
                          <option key={value} value={value}>
                            {value}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table className="forex-table pair-detail-table">
                    <thead>
                      <tr>
                        <th>Time</th>
                        <th>Stage</th>
                        <th>Reason</th>
                        <th>Direction</th>
                        <th>Session</th>
                        <th>EMA Dist</th>
                        <th>ATR%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows
                        .filter((row) =>
                          selectedPairDetail === "All"
                            ? true
                            : (row.pair ?? row.epic) === selectedPairDetail
                        )
                        .slice(0, 50)
                        .map((row) => (
                          <tr key={row.id}>
                            <td className="cell-time">
                              {formatDateTime(row.scan_timestamp).replace(", ", "\n")}
                            </td>
                            <td className="cell-nowrap">{row.rejection_stage}</td>
                            <td className="cell-reason">{row.rejection_reason ?? "-"}</td>
                            <td className="cell-nowrap">{row.attempted_direction ?? "-"}</td>
                            <td className="cell-nowrap">{row.market_session ?? "-"}</td>
                            <td className="cell-nowrap">{row.ema_distance_pips ?? "-"}</td>
                            <td className="cell-nowrap">{row.atr_percentile ?? "-"}</td>
                          </tr>
                        ))}
                    </tbody>
                    </table>
                  </div>
                </div>

                {scalpFilterBreakdown.length ? (
                  <div className="panel chart-panel">
                    <div className="chart-title">Scalp Entry Filter Breakdown</div>
                    <div className="bar-stack">
                      {scalpFilterBreakdown.map((row) => (
                        <div key={row.label} className="bar-row">
                          <span>{row.label}</span>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{ width: `${(row.value / stageTotal) * 100}%` }}
                            />
                          </div>
                          <strong>{row.value}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {tier2Breakdown.length ? (
                  <div className="panel chart-panel">
                    <div className="chart-title">Tier2 Swing Rejections</div>
                    <div className="bar-stack">
                      {tier2Breakdown.map((row) => (
                        <div key={row.label} className="bar-row">
                          <span>{row.label}</span>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{ width: `${(row.value / stageTotal) * 100}%` }}
                            />
                          </div>
                          <strong>{row.value}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}

            {activeTab === "conflicts" ? (
              <>
                <div className="metrics-grid">
                  <div className="summary-card">
                    Total Conflicts
                    <strong>{conflicts?.stats.total ?? 0}</strong>
                  </div>
                  <div className="summary-card">
                    Unique Pairs
                    <strong>{conflicts?.stats.unique_pairs ?? 0}</strong>
                  </div>
                  <div className="summary-card">
                    Sessions Affected
                    <strong>{conflicts?.stats.sessions_affected ?? 0}</strong>
                  </div>
                </div>
                <div className="forex-grid">
                  <div className="panel chart-panel">
                    <div className="chart-title">Conflicts by Pair</div>
                    <div className="bar-stack">
                      {(conflicts?.by_pair ?? []).map((row) => (
                        <div key={row.pair} className="bar-row">
                          <span>{row.pair ?? "-"}</span>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{ width: `${(row.count / maxConflictPair) * 100}%` }}
                            />
                          </div>
                          <strong>{row.count}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="panel chart-panel">
                    <div className="chart-title">Conflicts by Session</div>
                    <div className="bar-stack">
                      {(conflicts?.by_session ?? []).map((row) => (
                        <div key={row.market_session} className="bar-row">
                          <span>{row.market_session ?? "-"}</span>
                          <div className="bar-track">
                            <div
                              className="bar-fill"
                              style={{ width: `${(row.count / maxConflictSession) * 100}%` }}
                            />
                          </div>
                          <strong>{row.count}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="panel table-panel">
                  <div className="chart-title">Top Conflict Reasons</div>
                  <table className="forex-table">
                    <thead>
                      <tr>
                        <th>Reason</th>
                        <th>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(conflicts?.top_reasons ?? []).map((row) => (
                        <tr key={row.rejection_reason ?? "unknown"}>
                          <td>{row.rejection_reason}</td>
                          <td>{row.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}

            {activeTab === "outcomes" ? (
              <>
                <div className="metrics-grid">
                  <div className="summary-card">
                    Total Analyzed
                    <strong>{outcomes?.summary?.total_analyzed ?? 0}</strong>
                  </div>
                  <div className="summary-card">
                    Would-Be Win Rate
                    <strong>{outcomes?.summary?.would_be_win_rate?.toFixed(1) ?? 0}%</strong>
                  </div>
                  <div className="summary-card">
                    Winners
                    <strong>{outcomes?.summary?.winners ?? 0}</strong>
                  </div>
                  <div className="summary-card">
                    Missed Profit
                    <strong>{outcomes?.summary?.total_missed_pips ?? 0} pips</strong>
                  </div>
                  <div className="summary-card">
                    Avoided Loss
                    <strong>{outcomes?.summary?.avoided_loss_pips ?? 0} pips</strong>
                  </div>
                </div>

                <div className="panel chart-panel">
                  <div className="chart-title">Would-Be Win Rate by Stage</div>
                  <div className="bar-stack">
                    {(outcomes?.win_rate_by_stage ?? []).map((row: any) => (
                      <div key={row.rejection_stage} className="bar-row">
                        <span>{row.rejection_stage}</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${row.would_be_win_rate ?? 0}%` }}
                          />
                        </div>
                        <strong>{row.would_be_win_rate?.toFixed(1) ?? 0}%</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : null}

            {activeTab === "sr" ? (
              <div className="panel chart-panel">
                <div className="chart-title">S/R Path Blocking</div>
                <div className="bar-stack">
                  {srBlocking.map((row) => (
                    <div key={row.label} className="bar-row">
                      <span>{row.label}</span>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{ width: `${(row.value / maxSrBlocking) * 100}%` }}
                        />
                      </div>
                      <strong>{row.value}</strong>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {activeTab === "time" ? (
              <div className="forex-grid">
                <div className="panel chart-panel">
                  <div className="chart-title">Rejections by Session</div>
                  <div className="bar-stack">
                    {sessionCounts.map((row) => (
                      <div key={row.label} className="bar-row">
                        <span>{row.label}</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${(row.value / maxSessionCount) * 100}%` }}
                          />
                        </div>
                        <strong>{row.value}</strong>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel chart-panel">
                  <div className="chart-title">Rejections by Hour (UTC)</div>
                  <div className="bar-stack">
                    {hourCounts.map((row) => (
                      <div key={row.label} className="bar-row">
                        <span>{row.label}:00</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${(row.value / maxHourCount) * 100}%` }}
                          />
                        </div>
                        <strong>{row.value}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}

            {activeTab === "context" ? (
              <>
              <div className="market-context-grid">
                <div className="panel chart-panel">
                  <div className="chart-title">EMA Distance Distribution</div>
                  <p className="chart-caption">How far price deviates from the 4H EMA at rejection time.</p>
                  <div className="metrics-grid">
                    <div className="summary-card">
                      Mean
                      <strong>{emaStats.mean.toFixed(1)} pips</strong>
                    </div>
                    <div className="summary-card">
                      Median
                      <strong>{emaStats.median.toFixed(1)} pips</strong>
                    </div>
                    <div className="summary-card">
                      Samples
                      <strong>{emaStats.count}</strong>
                    </div>
                  </div>
                  <div className="bar-stack">
                    {emaBuckets.map((row) => (
                      <div key={row.label} className="bar-row">
                        <span>{row.label}</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${(row.value / maxEmaBucket) * 100}%` }}
                          />
                        </div>
                        <strong>{row.value}</strong>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel chart-panel">
                  <div className="chart-title">ATR Percentile Distribution</div>
                  <p className="chart-caption">Volatility regime when SMC signals are rejected.</p>
                  <div className="metrics-grid">
                    <div className="summary-card">
                      Mean
                      <strong>{atrStats.mean.toFixed(0)}%</strong>
                    </div>
                    <div className="summary-card">
                      Median
                      <strong>{atrStats.median.toFixed(0)}%</strong>
                    </div>
                    <div className="summary-card">
                      Samples
                      <strong>{atrStats.count}</strong>
                    </div>
                  </div>
                  <div className="bar-stack">
                    {atrBuckets.map((row) => (
                      <div key={row.label} className="bar-row">
                        <span>{row.label}</span>
                        <div className="bar-track">
                          <div
                            className="bar-fill"
                            style={{ width: `${(row.value / maxAtrBucket) * 100}%` }}
                          />
                        </div>
                        <strong>{row.value}</strong>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel table-panel">
                  <div className="chart-title">Per-Pair Context</div>
                  <p className="chart-caption">Top pairs by rejection count with average EMA distance and ATR percentile.</p>
                  <table className="forex-table">
                    <thead>
                      <tr>
                        <th>Pair</th>
                        <th>Rejections</th>
                        <th>Avg EMA Dist</th>
                        <th>Avg ATR%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pairContext.map((row) => (
                        <tr key={row.pair}>
                          <td>{row.pair}</td>
                          <td>{row.count}</td>
                          <td>{row.avg_ema.toFixed(1)}</td>
                          <td>{row.avg_atr.toFixed(0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              </>
            ) : null}

            {activeTab === "near" ? (
              <div className="panel table-panel">
                <div className="chart-title">Near-Misses (Confidence ≥ 0.45)</div>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Pair</th>
                      <th>Confidence</th>
                      <th>Stage</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {nearMisses.slice(0, 30).map((row) => (
                      <tr key={row.id}>
                        <td>{formatDateTime(row.scan_timestamp)}</td>
                        <td>{row.pair ?? row.epic}</td>
                        <td>{row.confidence_score?.toFixed(2)}</td>
                        <td>{row.rejection_stage}</td>
                        <td>{row.rejection_reason ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}

            {activeTab === "eff" ? (
              <div className="panel table-panel">
                <div className="chart-title">Scanner Efficiency</div>
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Stage</th>
                      <th>Count</th>
                      <th>Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stageCounts.map((row) => (
                      <tr key={row.label}>
                        <td>{row.label}</td>
                        <td>{row.value}</td>
                        <td>{((row.value / stageTotal) * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
