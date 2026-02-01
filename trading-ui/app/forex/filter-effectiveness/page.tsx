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
    fetch(`/trading/api/forex/filter-effectiveness/?days=${days}`, {
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
          <Link href="/settings">Settings</Link>
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
            {/* How to use */}
            <div
              style={{
                marginBottom: "1.5rem",
                padding: "1rem",
                background: "rgba(15, 76, 92, 0.04)",
                borderRadius: "6px",
                fontSize: "0.85rem",
                lineHeight: 1.6,
                color: "var(--muted)"
              }}
            >
              <strong style={{ color: "var(--ink)" }}>How to use this page</strong>
              <br />
              Each section groups trades by a filter dimension and compares win rates across groups.
              A filter is <strong style={{ color: "var(--good)" }}>PREDICTIVE</strong> if the
              spread between best and worst group is 10%+ win rate.
              The <strong>vs Baseline</strong> column shows how each group compares to overall
              performance. Green BEST/red WORST labels highlight actionable rows.
              If the worst group has many trades and poor results, consider blocking it.
              If a filter is <strong style={{ color: "var(--warn)" }}>NOT PREDICTIVE</strong>,
              it adds complexity without improving outcomes and may be worth disabling.
            </div>

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

const FILTER_GUIDES: Record<string, string> = {
  "Entry Quality Score":
    "Measures how close the entry is to the optimal Fibonacci retracement zone (38.2%-50%) combined with entry candle momentum. Higher scores mean price pulled back to a better level with stronger candle body. If 'Very Low' performs worst, set a minimum threshold to block poor entries.",
  "Direction vs Structure Alignment":
    "Compares trade direction (BUY/SELL) against the detected market structure bias (BULLISH/BEARISH/RANGING). 'ALIGNED' means trading with structure, 'COUNTER' means against it. If RANGING outperforms ALIGNED, the structure detector may be lagging price action.",
  "Market Structure Bias":
    "The raw market structure classification from Smart Money analysis. Shows whether trades taken during BULLISH, BEARISH, RANGING, or NEUTRAL structure perform differently. Large win rate gaps suggest the strategy works better in certain structures.",
  "Order Flow Alignment":
    "Whether trade direction matches the order flow bias from order block and fair value gap analysis. 'ALIGNED' means order flow supports the trade direction. If CONFLICTING trades perform equally, the order flow analysis isn't adding edge.",
  "Market Regime":
    "The detected market regime (trending, ranging, breakout, high_volatility) from performance metrics. If all regimes show similar win rates, the regime detector isn't useful for filtering. If one regime consistently underperforms, consider blocking trades during that regime.",
  "Volatility State":
    "Market volatility classification (low, normal, high, extreme) at signal time. Helps identify whether the strategy performs better in calm or volatile conditions. Wide spreads between groups suggest volatility-based filtering could help.",
  "MTF Alignment":
    "Whether all analyzed timeframes (1m, 5m, 15m) agree on direction. 'All TFs Aligned' means every timeframe confirms the signal. If aligned and non-aligned trades perform similarly, requiring full alignment just reduces signal count without improving quality.",
  "Efficiency Ratio":
    "Measures price movement efficiency (0-1). High efficiency means price is trending cleanly; low means choppy/ranging. Useful for identifying whether the strategy needs clean trends to work or handles chop well.",
  "MTF Confluence Score":
    "A composite score (0-1) measuring how many timeframes support the signal direction and how strong that support is. Higher scores mean more timeframes agree. Check if higher confluence actually translates to better outcomes.",
};

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

      {FILTER_GUIDES[group.name] && (
        <p
          style={{
            margin: "0.75rem 0 0",
            padding: "0.6rem 0.75rem",
            background: "rgba(15, 76, 92, 0.04)",
            borderRadius: "4px",
            fontSize: "0.82rem",
            lineHeight: 1.5,
            color: "var(--muted)"
          }}
        >
          <strong style={{ color: "var(--ink)" }}>How to read: </strong>
          {FILTER_GUIDES[group.name]}
        </p>
      )}

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
