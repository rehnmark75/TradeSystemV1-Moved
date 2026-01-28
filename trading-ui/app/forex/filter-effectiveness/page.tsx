/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type FilterMetric = {
  filter_name: string;
  filter_value: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
};

type FilterGroup = {
  name: string;
  description: string;
  metrics: FilterMetric[];
  recommendation: string;
  is_predictive: boolean;
};

type Baseline = {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
};

type Payload = {
  baseline: Baseline;
  filterGroups: FilterGroup[];
  days: number;
  generatedAt: string;
};

const formatNumber = (value: unknown, fractionDigits = 2) => {
  const parsed = typeof value === "number" ? value : Number(value);
  return parsed != null && Number.isFinite(parsed)
    ? parsed.toLocaleString("en-US", {
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits
      })
    : "0.00";
};

const formatPercent = (value: unknown, fractionDigits = 1) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${parsed.toFixed(fractionDigits)}%` : "0%";
};

export default function FilterEffectivenessPage() {
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/stock-scanner/api/forex/filter-effectiveness/?days=${days}`, {
      signal: controller.signal
    })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load filter effectiveness analysis."))
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [days]);

  const predictiveFilters = payload?.filterGroups.filter((g) => g.is_predictive) || [];
  const nonPredictiveFilters = payload?.filterGroups.filter((g) => !g.is_predictive) || [];

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
          <h1>Filter Effectiveness Audit</h1>
          <p>
            Analyze which signal filters actually predict trade outcomes vs those with no predictive value.
          </p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <div className="forex-nav">
        <Link href="/forex" className="forex-pill">Overview</Link>
        <Link href="/forex/strategy" className="forex-pill">Strategy</Link>
        <Link href="/forex/trade-performance" className="forex-pill">Trade Performance</Link>
        <Link href="/forex/entry-timing" className="forex-pill">Entry Timing</Link>
        <Link href="/forex/mae-analysis" className="forex-pill">MAE Analysis</Link>
        <Link href="/forex/alert-history" className="forex-pill">Alert History</Link>
        <Link href="/forex/smc-rejections" className="forex-pill">SMC Rejections</Link>
        <Link href="/forex/filter-effectiveness" className="forex-pill active">Filter Audit</Link>
      </div>

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Analysis Period</label>
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={60}>60 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        {loading ? (
          <div className="chart-placeholder">Loading filter analysis...</div>
        ) : payload ? (
          <>
            {/* Baseline Stats */}
            <div className="metrics-grid" style={{ marginBottom: "2rem" }}>
              <div className="summary-card">
                Total Trades
                <strong>{payload.baseline.total_trades}</strong>
              </div>
              <div className="summary-card">
                Baseline Win Rate
                <strong>{formatPercent(payload.baseline.win_rate)}</strong>
              </div>
              <div className="summary-card">
                Total P&L
                <strong className={payload.baseline.total_pnl < 0 ? "bad" : "good"}>
                  {formatNumber(payload.baseline.total_pnl)} SEK
                </strong>
              </div>
              <div className="summary-card">
                Avg P&L / Trade
                <strong className={payload.baseline.avg_pnl < 0 ? "bad" : "good"}>
                  {formatNumber(payload.baseline.avg_pnl)} SEK
                </strong>
              </div>
            </div>

            {/* Predictive Filters */}
            {predictiveFilters.length > 0 && (
              <div style={{ marginBottom: "2rem" }}>
                <h2 style={{ color: "var(--good)", marginBottom: "1rem" }}>
                  Predictive Filters ({predictiveFilters.length})
                </h2>
                <p style={{ color: "var(--muted)", marginBottom: "1rem" }}>
                  These filters show significant performance differences between groups (10%+ win rate spread).
                </p>
                {predictiveFilters.map((group) => (
                  <FilterGroupCard key={group.name} group={group} baseline={payload.baseline} />
                ))}
              </div>
            )}

            {/* Non-Predictive Filters */}
            {nonPredictiveFilters.length > 0 && (
              <div>
                <h2 style={{ color: "var(--warn)", marginBottom: "1rem" }}>
                  Non-Predictive Filters ({nonPredictiveFilters.length})
                </h2>
                <p style={{ color: "var(--muted)", marginBottom: "1rem" }}>
                  These filters show no significant performance difference - consider disabling them to reduce complexity.
                </p>
                {nonPredictiveFilters.map((group) => (
                  <FilterGroupCard key={group.name} group={group} baseline={payload.baseline} />
                ))}
              </div>
            )}

            <div style={{ marginTop: "2rem", color: "var(--muted)", fontSize: "0.85rem" }}>
              Generated: {new Date(payload.generatedAt).toLocaleString()}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

function FilterGroupCard({ group, baseline }: { group: FilterGroup; baseline: Baseline }) {
  const maxWinRate = Math.max(...group.metrics.map((m) => m.win_rate || 0));
  const minWinRate = Math.min(...group.metrics.filter((m) => m.trades >= 3).map((m) => m.win_rate || 0));

  return (
    <div
      className="panel"
      style={{
        marginBottom: "1.5rem",
        borderLeft: group.is_predictive ? "4px solid var(--good)" : "4px solid var(--warn)"
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h3 style={{ margin: 0 }}>{group.name}</h3>
          <p style={{ color: "var(--muted)", margin: "0.25rem 0" }}>{group.description}</p>
        </div>
        <span
          className={group.is_predictive ? "good" : "warn"}
          style={{
            padding: "0.25rem 0.75rem",
            borderRadius: "4px",
            fontSize: "0.8rem",
            fontWeight: 600,
            background: group.is_predictive ? "rgba(30, 126, 52, 0.1)" : "rgba(253, 126, 20, 0.1)"
          }}
        >
          {group.is_predictive ? "PREDICTIVE" : "NOT PREDICTIVE"}
        </span>
      </div>

      <table className="forex-table" style={{ marginTop: "1rem" }}>
        <thead>
          <tr>
            <th>Value</th>
            <th style={{ textAlign: "right" }}>Trades</th>
            <th style={{ textAlign: "right" }}>Wins</th>
            <th style={{ textAlign: "right" }}>Win Rate</th>
            <th style={{ textAlign: "right" }}>vs Baseline</th>
            <th style={{ textAlign: "right" }}>Total P&L</th>
            <th style={{ textAlign: "right" }}>Avg P&L</th>
          </tr>
        </thead>
        <tbody>
          {group.metrics.map((metric) => {
            const vsBaseline = (metric.win_rate || 0) - (baseline.win_rate || 0);
            const isBest = metric.win_rate === maxWinRate && group.metrics.length > 1;
            const isWorst = metric.win_rate === minWinRate && group.metrics.length > 1 && metric.trades >= 3;

            return (
              <tr
                key={metric.filter_value}
                style={{
                  background: isBest
                    ? "rgba(30, 126, 52, 0.08)"
                    : isWorst
                    ? "rgba(220, 53, 69, 0.08)"
                    : undefined
                }}
              >
                <td>
                  <span style={{ fontWeight: isBest || isWorst ? 600 : 400 }}>
                    {metric.filter_value}
                  </span>
                  {isBest && (
                    <span
                      style={{
                        marginLeft: "0.5rem",
                        fontSize: "0.7rem",
                        color: "var(--good)",
                        fontWeight: 600
                      }}
                    >
                      BEST
                    </span>
                  )}
                  {isWorst && (
                    <span
                      style={{
                        marginLeft: "0.5rem",
                        fontSize: "0.7rem",
                        color: "var(--bad)",
                        fontWeight: 600
                      }}
                    >
                      WORST
                    </span>
                  )}
                </td>
                <td style={{ textAlign: "right" }}>{metric.trades}</td>
                <td style={{ textAlign: "right" }}>{metric.wins}</td>
                <td style={{ textAlign: "right" }}>
                  <strong>{formatPercent(metric.win_rate)}</strong>
                </td>
                <td
                  style={{ textAlign: "right" }}
                  className={vsBaseline > 0 ? "good" : vsBaseline < 0 ? "bad" : ""}
                >
                  {vsBaseline > 0 ? "+" : ""}
                  {formatPercent(vsBaseline)}
                </td>
                <td
                  style={{ textAlign: "right" }}
                  className={metric.total_pnl < 0 ? "bad" : metric.total_pnl > 0 ? "good" : ""}
                >
                  {formatNumber(metric.total_pnl)}
                </td>
                <td
                  style={{ textAlign: "right" }}
                  className={metric.avg_pnl < 0 ? "bad" : metric.avg_pnl > 0 ? "good" : ""}
                >
                  {formatNumber(metric.avg_pnl)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {group.recommendation && (
        <div
          style={{
            marginTop: "1rem",
            padding: "0.75rem",
            background: "rgba(15, 76, 92, 0.05)",
            borderRadius: "4px",
            fontSize: "0.9rem"
          }}
        >
          <strong>Recommendation:</strong> {group.recommendation}
        </div>
      )}
    </div>
  );
}
