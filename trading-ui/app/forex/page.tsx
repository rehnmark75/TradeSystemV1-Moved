/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type ForexStats = {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  pending_trades: number;
  total_profit_loss: number;
  win_rate: number;
  avg_profit: number;
  avg_loss: number;
  profit_factor: number;
  largest_win: number;
  largest_loss: number;
  best_pair: string;
  worst_pair: string;
  active_pairs: string[];
};

type DailyPnl = {
  date: string;
  daily_pnl: number;
  trade_count: number;
};

type RecentTrade = {
  id: number;
  symbol: string;
  direction: string;
  timestamp: string;
  status: string;
  profit_loss: number | null;
  pnl_currency: string | null;
  strategy: string | null;
};

type StrategyRow = {
  strategy: string;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_confidence: number;
  best_trade: number;
  worst_trade: number;
  pairs_traded: number;
};

type PairRow = {
  symbol: string;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  best_trade: number;
  worst_trade: number;
};

type OverviewPayload = {
  stats: ForexStats;
  daily_pnl: DailyPnl[];
  recent_trades: RecentTrade[];
};

type AnalysisPayload = {
  strategies: StrategyRow[];
  pairs: PairRow[];
};

const OVERVIEW_PRESETS = [
  { label: "24h", value: 1 },
  { label: "7d", value: 7 },
  { label: "30d", value: 30 },
  { label: "90d", value: 90 }
];

const ANALYSIS_PRESETS = [
  { label: "7d", value: 7 },
  { label: "30d", value: 30 },
  { label: "90d", value: 90 }
];

const toNumber = (value: unknown) => {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const formatNumber = (value: unknown, fractionDigits = 2) => {
  const parsed = toNumber(value);
  return parsed != null
    ? parsed.toLocaleString("en-US", {
        minimumFractionDigits: fractionDigits,
        maximumFractionDigits: fractionDigits
      })
    : "0.00";
};

const formatPercent = (value: unknown, fractionDigits = 1) => {
  const parsed = toNumber(value);
  return parsed != null ? `${parsed.toFixed(fractionDigits)}%` : "0%";
};

const formatDateTime = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
};

const buildSparklinePoints = (values: number[]) => {
  if (!values.length) return "";
  const width = 400;
  const height = 140;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1 || 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
};

export default function ForexAnalyticsPage() {
  const [section, setSection] = useState<"overview" | "analysis">("overview");
  const [overviewDays, setOverviewDays] = useState(1);
  const [overviewRefreshKey, setOverviewRefreshKey] = useState(0);
  const [analysisDays, setAnalysisDays] = useState(30);
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [loadingOverview, setLoadingOverview] = useState(false);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoadingOverview(true);
    setOverviewError(null);
    fetch(`/trading/api/forex/overview/?days=${overviewDays}`, {
      signal: controller.signal
    })
      .then((res) => res.json())
      .then((data) => setOverview(data))
      .catch((error) => {
        if (error.name !== "AbortError") {
          setOverviewError("Failed to load overview.");
        }
      })
      .finally(() => setLoadingOverview(false));
    return () => controller.abort();
  }, [overviewDays, overviewRefreshKey]);

  useEffect(() => {
    const controller = new AbortController();
    setLoadingAnalysis(true);
    setAnalysisError(null);
    fetch(`/trading/api/forex/analysis/?days=${analysisDays}`, {
      signal: controller.signal
    })
      .then((res) => res.json())
      .then((data) => setAnalysis(data))
      .catch((error) => {
        if (error.name !== "AbortError") {
          setAnalysisError("Failed to load analysis.");
        }
      })
      .finally(() => setLoadingAnalysis(false));
    return () => controller.abort();
  }, [analysisDays]);

  const pnlSeries = useMemo(() => {
    if (!overview?.daily_pnl?.length) return [];
    let cumulative = 0;
    return overview.daily_pnl.map((row) => {
      cumulative += Number(row.daily_pnl ?? 0);
      return cumulative;
    });
  }, [overview?.daily_pnl]);

  const pnlPoints = useMemo(() => buildSparklinePoints(pnlSeries), [pnlSeries]);
  const winLossTotal =
    (overview?.stats.winning_trades ?? 0) +
    (overview?.stats.losing_trades ?? 0) +
    (overview?.stats.pending_trades ?? 0);
  const winPct = winLossTotal
    ? (overview?.stats.winning_trades ?? 0) / winLossTotal
    : 0;
  const lossPct = winLossTotal
    ? (overview?.stats.losing_trades ?? 0) / winLossTotal
    : 0;
  const pendingPct = winLossTotal
    ? (overview?.stats.pending_trades ?? 0) / winLossTotal
    : 0;

  const strategyMax = useMemo(() => {
    if (!analysis?.strategies?.length) return 1;
    return Math.max(...analysis.strategies.map((row) => Math.abs(row.total_pnl)), 1);
  }, [analysis?.strategies]);

  const pairMax = useMemo(() => {
    if (!analysis?.pairs?.length) return 1;
    return Math.max(...analysis.pairs.map((row) => Math.abs(row.total_pnl)), 1);
  }, [analysis?.pairs]);

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
          <h1>Forex Unified Analytics</h1>
          <p>Performance overview and strategy diagnostics for the live book.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <div className="section-tabs">
        <button
          className={`section-tab ${section === "overview" ? "active" : ""}`}
          onClick={() => setSection("overview")}
        >
          Overview
        </button>
        <button
          className={`section-tab ${section === "analysis" ? "active" : ""}`}
          onClick={() => setSection("analysis")}
        >
          Analysis
        </button>
      </div>

      <div className="forex-nav">
        <Link href="/forex/strategy" className="forex-pill">
          Strategy Performance
        </Link>
        <Link href="/forex/trade-performance" className="forex-pill">
          Trade Performance
        </Link>
        <Link href="/forex/entry-timing" className="forex-pill">
          Entry Timing
        </Link>
        <Link href="/forex/mae-analysis" className="forex-pill">
          MAE Analysis
        </Link>
        <Link href="/forex/alert-history" className="forex-pill">
          Alert History
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
        <Link href="/forex/filter-effectiveness" className="forex-pill">
          Filter Audit
        </Link>
      </div>

      {section === "overview" ? (
        <div className="panel">
          <div className="forex-controls">
            <div>
              <label>Timeframe</label>
              <select
                value={overviewDays}
                onChange={(event) => setOverviewDays(Number(event.target.value))}
              >
                {OVERVIEW_PRESETS.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.label}
                  </option>
                ))}
              </select>
              <button
                className="refresh-btn"
                onClick={() => setOverviewRefreshKey((k) => k + 1)}
                disabled={loadingOverview}
                title="Refresh data"
              >
                ↻
              </button>
            </div>
            <div className="forex-badge">
              {overview?.stats.best_pair ? `Best: ${overview.stats.best_pair}` : "Best: -"}
            </div>
            <div className="forex-badge">
              {overview?.stats.worst_pair ? `Worst: ${overview.stats.worst_pair}` : "Worst: -"}
            </div>
          </div>

          {overviewError ? <div className="error">{overviewError}</div> : null}

          <div className="metrics-grid">
            <div className="summary-card">
              Total P&L
              <strong>
                {overview
                  ? `${formatNumber(overview.stats.total_profit_loss)} SEK`
                  : "-"}
              </strong>
            </div>
            <div className="summary-card">
              Win Rate
              <strong>{overview ? formatPercent(overview.stats.win_rate) : "-"}</strong>
            </div>
            <div className="summary-card">
              Profit Factor
              <strong>
                {overview
                  ? overview.stats.profit_factor === Number.POSITIVE_INFINITY
                    ? "∞"
                    : formatNumber(overview.stats.profit_factor, 2)
                  : "-"}
              </strong>
            </div>
            <div className="summary-card">
              Best Trade
              <strong>
                {overview ? `+${formatNumber(overview.stats.largest_win)} SEK` : "-"}
              </strong>
            </div>
            <div className="summary-card">
              Worst Trade
              <strong>
                {overview ? formatNumber(overview.stats.largest_loss) : "-"} SEK
              </strong>
            </div>
          </div>

          <div className="forex-grid">
            <div className="panel chart-panel">
              <div className="chart-title">Cumulative P&L</div>
              {loadingOverview ? (
                <div className="chart-placeholder">Loading chart...</div>
              ) : pnlSeries.length ? (
                <svg viewBox="0 0 400 140" className="pnl-chart">
                  <polyline points={pnlPoints} className="pnl-line" />
                </svg>
              ) : (
                <div className="chart-placeholder">No P&L data yet.</div>
              )}
            </div>
            <div className="panel chart-panel">
              <div className="chart-title">Trade Outcomes</div>
              <div
                className="donut"
                style={{
                  background: `conic-gradient(var(--good) ${winPct * 360}deg, var(--bad) ${winPct * 360}deg ${
                    (winPct + lossPct) * 360
                  }deg, #9aa5b1 ${(winPct + lossPct) * 360}deg 360deg)`
                }}
              />
              <div className="donut-legend">
                <span>Wins: {overview?.stats.winning_trades ?? 0}</span>
                <span>Losses: {overview?.stats.losing_trades ?? 0}</span>
                <span>Pending: {overview?.stats.pending_trades ?? 0}</span>
              </div>
            </div>
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Recent Trades</div>
            {loadingOverview ? (
              <div className="chart-placeholder">Loading trades...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Strategy</th>
                    <th>Status</th>
                    <th>P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview?.recent_trades ?? []).map((trade) => {
                    const pnl = trade.profit_loss;
                    const pnlValue =
                      pnl == null
                        ? trade.status
                        : `${pnl >= 0 ? "+" : ""}${formatNumber(pnl)} ${
                            trade.pnl_currency ?? ""
                          }`;
                    const pnlClass = pnl == null ? "" : pnl < 0 ? "bad" : "good";
                    return (
                      <tr key={trade.id}>
                        <td>{formatDateTime(trade.timestamp)}</td>
                        <td>{trade.symbol}</td>
                        <td>{trade.direction}</td>
                        <td>{trade.strategy ?? "-"}</td>
                        <td>{trade.status}</td>
                        <td className={pnlClass}>{pnlValue}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      ) : (
        <div className="panel">
          <div className="forex-controls">
            <div>
              <label>Analysis Window</label>
              <select
                value={analysisDays}
                onChange={(event) => setAnalysisDays(Number(event.target.value))}
              >
                {ANALYSIS_PRESETS.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="forex-badge">Strategy + Pair breakdown</div>
          </div>

          {analysisError ? <div className="error">{analysisError}</div> : null}

          <div className="forex-grid">
            <div className="panel chart-panel">
              <div className="chart-title">P&L by Strategy</div>
              {loadingAnalysis ? (
                <div className="chart-placeholder">Loading analysis...</div>
              ) : (
                <div className="bar-stack">
                  {(analysis?.strategies ?? []).map((row) => (
                    <div key={row.strategy} className="bar-row">
                      <span>{row.strategy}</span>
                      <div className="bar-track">
                        <div
                          className={`bar-fill ${row.total_pnl >= 0 ? "good" : "bad"}`}
                          style={{
                            width: `${(Math.abs(row.total_pnl) / strategyMax) * 100}%`
                          }}
                        />
                      </div>
                      <strong>{formatNumber(row.total_pnl)} SEK</strong>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="panel chart-panel">
              <div className="chart-title">Win Rate by Strategy</div>
              {loadingAnalysis ? (
                <div className="chart-placeholder">Loading analysis...</div>
              ) : (
                <div className="bar-stack">
                  {(analysis?.strategies ?? []).map((row) => (
                    <div key={`${row.strategy}-win`} className="bar-row">
                      <span>{row.strategy}</span>
                      <div className="bar-track">
                        <div
                          className="bar-fill good"
                          style={{ width: `${row.win_rate}%` }}
                        />
                      </div>
                      <strong>{formatPercent(row.win_rate)}</strong>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Strategy Performance</div>
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Trades</th>
                  <th>Win Rate</th>
                  <th>Total P&L</th>
                  <th>Avg P&L</th>
                  <th>Pairs</th>
                </tr>
              </thead>
              <tbody>
                {(analysis?.strategies ?? []).map((row) => (
                  <tr key={`${row.strategy}-table`}>
                    <td>{row.strategy}</td>
                    <td>{row.total_trades}</td>
                    <td>{formatPercent(row.win_rate)}</td>
                    <td className={row.total_pnl < 0 ? "bad" : "good"}>
                      {formatNumber(row.total_pnl)}
                    </td>
                    <td>{formatNumber(row.avg_pnl)}</td>
                    <td>{row.pairs_traded}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Top Pairs by P&L</div>
            <div className="bar-stack">
              {(analysis?.pairs ?? []).slice(0, 10).map((row) => (
                <div key={row.symbol} className="bar-row">
                  <span>{row.symbol}</span>
                  <div className="bar-track">
                    <div
                      className={`bar-fill ${row.total_pnl >= 0 ? "good" : "bad"}`}
                      style={{
                        width: `${(Math.abs(row.total_pnl) / pairMax) * 100}%`
                      }}
                    />
                  </div>
                  <strong>{formatNumber(row.total_pnl)} SEK</strong>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
