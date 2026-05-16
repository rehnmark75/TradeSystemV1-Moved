/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "./_components/ForexNav";
import { useEnvironment } from "../../lib/environment";
import EnvironmentToggle from "../../components/EnvironmentToggle";

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

const formatYLabel = (v: number) => {
  const abs = Math.abs(v);
  const s = abs >= 10000 ? `${(v / 1000).toFixed(1)}k` : abs >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0);
  return v > 0 ? `+${s}` : s;
};

const formatXDate = (d: string) => {
  if (!d) return "";
  const dt = new Date(d);
  return dt.toLocaleDateString("en-GB", { month: "short", day: "numeric" });
};

export default function ForexAnalyticsPage() {
  const { environment } = useEnvironment();
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
  const [tradeSort, setTradeSort] = useState<{ key: keyof RecentTrade; dir: "asc" | "desc" }>({
    key: "timestamp",
    dir: "desc"
  });
  const [strategyFilter, setStrategyFilter] = useState("All");
  const [pairFilter, setPairFilter] = useState("All");

  useEffect(() => {
    const controller = new AbortController();
    setLoadingOverview(true);
    setOverviewError(null);
    fetch(`/trading/api/forex/overview/?days=${overviewDays}&env=${environment}`, {
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
  }, [overviewDays, overviewRefreshKey, environment]);

  useEffect(() => {
    const controller = new AbortController();
    setLoadingAnalysis(true);
    setAnalysisError(null);
    fetch(`/trading/api/forex/analysis/?days=${analysisDays}&env=${environment}`, {
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
  }, [analysisDays, environment]);

  const pnlSeries = useMemo(() => {
    if (!overview?.daily_pnl?.length) return [];
    let cumulative = 0;
    return overview.daily_pnl.map((row) => {
      cumulative += Number(row.daily_pnl ?? 0);
      return cumulative;
    });
  }, [overview?.daily_pnl]);

  const chartGeometry = useMemo(() => {
    if (!pnlSeries.length) return null;
    const ML = 54, MR = 6, MT = 8, MB = 22;
    const VW = 480, VH = 170;
    const PW = VW - ML - MR;
    const PH = VH - MT - MB;

    const rawMin = Math.min(...pnlSeries);
    const rawMax = Math.max(...pnlSeries);
    const min = Math.min(rawMin, 0);
    const max = Math.max(rawMax, 0);
    const range = max - min || 1;

    const toX = (i: number) => ML + (i / Math.max(pnlSeries.length - 1, 1)) * PW;
    const toY = (v: number) => MT + PH - ((v - min) / range) * PH;

    const pts = pnlSeries.map((v, i) => [toX(i), toY(v)] as [number, number]);
    const linePath = "M " + pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L ");
    const bottomY = MT + PH;
    const areaPath =
      `M ${pts[0][0].toFixed(1)},${bottomY.toFixed(1)} ` +
      pts.map(([x, y]) => `L ${x.toFixed(1)},${y.toFixed(1)}`).join(" ") +
      ` L ${pts[pts.length - 1][0].toFixed(1)},${bottomY.toFixed(1)} Z`;

    const zeroY = toY(0);
    const final = pnlSeries[pnlSeries.length - 1];
    const isPositive = final >= 0;

    const yLabelCandidates = [
      { value: max, y: toY(max) },
      { value: 0, y: zeroY },
      { value: min, y: toY(min) }
    ];
    const yLabels: { value: number; y: number }[] = [];
    for (const c of yLabelCandidates) {
      if (!yLabels.length || Math.abs(c.y - yLabels[yLabels.length - 1].y) >= 18) {
        yLabels.push(c);
      }
    }

    const dailyPnl = overview?.daily_pnl ?? [];
    const firstDate = dailyPnl[0]?.date ?? "";
    const lastDate = dailyPnl[dailyPnl.length - 1]?.date ?? "";
    const firstX = pts[0]?.[0] ?? ML;
    const lastX = pts[pts.length - 1]?.[0] ?? ML + PW;

    return { VW, VH, linePath, areaPath, zeroY, yLabels, final, isPositive, bottomY, firstDate, lastDate, firstX, lastX };
  }, [pnlSeries, overview?.daily_pnl]);
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

  const sortedTrades = useMemo(() => {
    const trades = overview?.recent_trades ?? [];
    const { key, dir } = tradeSort;
    return [...trades].sort((a, b) => {
      const av = a[key] ?? "";
      const bv = b[key] ?? "";
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return dir === "asc" ? cmp : -cmp;
    });
  }, [overview?.recent_trades, tradeSort]);

  const toggleSort = (key: keyof RecentTrade) => {
    setTradeSort((prev) =>
      prev.key === key ? { key, dir: prev.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" }
    );
  };

  const sortIndicator = (key: keyof RecentTrade) => {
    if (tradeSort.key !== key) return " ↕";
    return tradeSort.dir === "asc" ? " ↑" : " ↓";
  };

  const availableStrategies = useMemo(() => {
    const strats = new Set<string>();
    for (const t of overview?.recent_trades ?? []) {
      if (t.strategy) strats.add(t.strategy);
    }
    return ["All", ...Array.from(strats).sort()];
  }, [overview?.recent_trades]);

  const availablePairs = useMemo(() => {
    const pairs = new Set<string>();
    for (const t of overview?.recent_trades ?? []) {
      if (t.symbol) pairs.add(t.symbol);
    }
    return ["All", ...Array.from(pairs).sort()];
  }, [overview?.recent_trades]);

  const filteredTrades = useMemo(() => {
    return sortedTrades.filter((t) => {
      if (strategyFilter !== "All" && t.strategy !== strategyFilter) return false;
      if (pairFilter !== "All" && t.symbol !== pairFilter) return false;
      return true;
    });
  }, [sortedTrades, strategyFilter, pairFilter]);

  const filteredStats = useMemo(() => {
    const pnlOf = (t: RecentTrade) => (t.profit_loss != null ? Number(t.profit_loss) : null);
    const closed = filteredTrades.filter((t) => pnlOf(t) != null);
    const wins = closed.filter((t) => (pnlOf(t) ?? 0) > 0).length;
    const pnl = closed.reduce((sum, t) => sum + (pnlOf(t) ?? 0), 0);
    const wr = closed.length > 0 ? (wins / closed.length) * 100 : 0;
    return { total: filteredTrades.length, closed: closed.length, wins, pnl, wr };
  }, [filteredTrades]);

  const directionStats = useMemo(() => {
    const pnlOf = (t: RecentTrade) => (t.profit_loss != null ? Number(t.profit_loss) : 0);
    const trades = overview?.recent_trades ?? [];
    const buy = trades.filter((t) => t.direction === "BUY");
    const sell = trades.filter((t) => t.direction === "SELL");
    const buyWins = buy.filter((t) => pnlOf(t) > 0).length;
    const sellWins = sell.filter((t) => pnlOf(t) > 0).length;
    return {
      buyCount: buy.length,
      buyWR: buy.length ? (buyWins / buy.length) * 100 : 0,
      buyPnL: buy.reduce((s, t) => s + pnlOf(t), 0),
      sellCount: sell.length,
      sellWR: sell.length ? (sellWins / sell.length) * 100 : 0,
      sellPnL: sell.reduce((s, t) => s + pnlOf(t), 0)
    };
  }, [overview?.recent_trades]);

  const expectancy = useMemo(() => {
    if (!overview) return null;
    const { avg_profit, avg_loss, win_rate } = overview.stats;
    if (!avg_profit && !avg_loss) return null;
    const wr = win_rate / 100;
    return avg_profit * wr + avg_loss * (1 - wr);
  }, [overview]);

  const avgRR = useMemo(() => {
    if (!overview) return null;
    const { avg_profit, avg_loss } = overview.stats;
    if (!avg_profit || !avg_loss || avg_loss >= 0) return null;
    return avg_profit / Math.abs(avg_loss);
  }, [overview]);


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

      <div className="desk-intro">
        <div>
          <div className="mission-kicker">FX Command Desk</div>
          <h2>Scanner performance, pair behavior, and execution review in one operator-grade overview.</h2>
          <p>
            This surface is the anchor for the forex workflow: assess the book, inspect quality drift,
            and decide whether to drill into strategy, chart, or rejection analysis.
          </p>
        </div>
        <div className="desk-intro-meta">
          <div className="desk-intro-stat">
            <span>Environment</span>
            <strong>{environment.toUpperCase()}</strong>
          </div>
          <div className="desk-intro-stat">
            <span>Scope</span>
            <strong>Performance, diagnostics, and live operational review</strong>
          </div>
        </div>
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

      <ForexNav activeHref="/forex" />

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
            <div className="summary-card">
              Expectancy
              <strong className={expectancy == null ? "" : expectancy >= 0 ? "good" : "bad"}>
                {expectancy != null
                  ? `${expectancy >= 0 ? "+" : ""}${formatNumber(expectancy)} SEK`
                  : "-"}
              </strong>
            </div>
            <div className="summary-card">
              Avg R:R
              <strong>
                {avgRR != null ? `1 : ${formatNumber(avgRR, 2)}` : "-"}
              </strong>
            </div>
          </div>

          <div className="forex-grid">
            <div className="panel chart-panel">
              <div className="chart-title-row">
                <span className="chart-title">Cumulative P&L</span>
                {chartGeometry && (
                  <span className={`chart-final-value ${chartGeometry.isPositive ? "good" : "bad"}`}>
                    {chartGeometry.final >= 0 ? "+" : ""}{formatNumber(chartGeometry.final)} SEK
                  </span>
                )}
              </div>
              {loadingOverview ? (
                <div className="chart-placeholder">Loading chart...</div>
              ) : chartGeometry ? (
                <svg
                  viewBox={`0 0 ${chartGeometry.VW} ${chartGeometry.VH}`}
                  className="pnl-chart"
                  style={{ overflow: "visible" }}
                >
                  <defs>
                    <linearGradient id="pnlAreaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={chartGeometry.isPositive ? "var(--good)" : "var(--bad)"} stopOpacity="0.18" />
                      <stop offset="100%" stopColor={chartGeometry.isPositive ? "var(--good)" : "var(--bad)"} stopOpacity="0.02" />
                    </linearGradient>
                  </defs>

                  {/* area fill */}
                  <path d={chartGeometry.areaPath} fill="url(#pnlAreaGrad)" />

                  {/* zero reference line */}
                  <line
                    x1="54" y1={chartGeometry.zeroY} x2={480 - 6} y2={chartGeometry.zeroY}
                    stroke="#9aa5b1" strokeWidth="1" strokeDasharray="4 3" opacity="0.55"
                  />

                  {/* Y-axis labels */}
                  {chartGeometry.yLabels.map(({ value, y }) => (
                    <text
                      key={value}
                      x="50"
                      y={y}
                      textAnchor="end"
                      dominantBaseline="middle"
                      className="chart-axis-label"
                    >
                      {formatYLabel(value)}
                    </text>
                  ))}

                  {/* X-axis date labels */}
                  <text x={chartGeometry.firstX} y={chartGeometry.VH - 4} textAnchor="start" className="chart-axis-label">
                    {formatXDate(chartGeometry.firstDate)}
                  </text>
                  <text x={chartGeometry.lastX} y={chartGeometry.VH - 4} textAnchor="end" className="chart-axis-label">
                    {formatXDate(chartGeometry.lastDate)}
                  </text>

                  {/* data line */}
                  <path d={chartGeometry.linePath} fill="none" className={`pnl-line-path ${chartGeometry.isPositive ? "pnl-positive" : "pnl-negative"}`} />
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
              <div className="direction-breakdown">
                <div>
                  <span className="dir-label">BUY</span>
                  {directionStats.buyCount} trades ·{" "}
                  <span className={directionStats.buyWR >= 50 ? "good" : "bad"}>
                    {formatPercent(directionStats.buyWR)} WR
                  </span>{" "}
                  · <span className={directionStats.buyPnL >= 0 ? "good" : "bad"}>
                    {directionStats.buyPnL >= 0 ? "+" : ""}{formatNumber(directionStats.buyPnL)} SEK
                  </span>
                </div>
                <div>
                  <span className="dir-label">SELL</span>
                  {directionStats.sellCount} trades ·{" "}
                  <span className={directionStats.sellWR >= 50 ? "good" : "bad"}>
                    {formatPercent(directionStats.sellWR)} WR
                  </span>{" "}
                  · <span className={directionStats.sellPnL >= 0 ? "good" : "bad"}>
                    {directionStats.sellPnL >= 0 ? "+" : ""}{formatNumber(directionStats.sellPnL)} SEK
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="panel table-panel">
            <div className="chart-title">
              Trades in Period
              {overview?.recent_trades?.length
                ? ` (${overview.recent_trades.length})`
                : ""}
            </div>
            <div className="trade-filters">
              <div className="trade-filter-group">
                <label>Strategy</label>
                <select value={strategyFilter} onChange={(e) => setStrategyFilter(e.target.value)}>
                  {availableStrategies.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div className="trade-filter-group">
                <label>Pair</label>
                <select value={pairFilter} onChange={(e) => setPairFilter(e.target.value)}>
                  {availablePairs.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              {(strategyFilter !== "All" || pairFilter !== "All") && (
                <div className="filter-stats">
                  <span>{filteredStats.total} trades</span>
                  <span className={filteredStats.pnl >= 0 ? "good" : "bad"}>
                    {filteredStats.pnl >= 0 ? "+" : ""}{formatNumber(filteredStats.pnl)} SEK
                  </span>
                  <span>{formatPercent(filteredStats.wr)} WR</span>
                  <button className="clear-filter-btn" onClick={() => { setStrategyFilter("All"); setPairFilter("All"); }}>✕ Clear</button>
                </div>
              )}
            </div>
            {loadingOverview ? (
              <div className="chart-placeholder">Loading trades...</div>
            ) : (
              <table className="forex-table mobile-card-table">
                <thead>
                  <tr>
                    <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort("timestamp")}>Time{sortIndicator("timestamp")}</th>
                    <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort("symbol")}>Symbol{sortIndicator("symbol")}</th>
                    <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort("direction")}>Direction{sortIndicator("direction")}</th>
                    <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort("strategy")}>Strategy{sortIndicator("strategy")}</th>
                    <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort("status")}>Status{sortIndicator("status")}</th>
                    <th style={{ cursor: "pointer", userSelect: "none" }} onClick={() => toggleSort("profit_loss")}>P&amp;L{sortIndicator("profit_loss")}</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades.map((trade) => {
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
                        <td data-label="Time">{formatDateTime(trade.timestamp)}</td>
                        <td data-label="Symbol">{trade.symbol}</td>
                        <td data-label="Direction">{trade.direction}</td>
                        <td data-label="Strategy">{trade.strategy ?? "-"}</td>
                        <td data-label="Status">{trade.status}</td>
                        <td data-label="P&L" className={pnlClass}>{pnlValue}</td>
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
            <table className="forex-table mobile-card-table">
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
                    <td data-label="Strategy">{row.strategy}</td>
                    <td data-label="Trades">{row.total_trades}</td>
                    <td data-label="Win Rate">{formatPercent(row.win_rate)}</td>
                    <td data-label="Total P&L" className={row.total_pnl < 0 ? "bad" : "good"}>
                      {formatNumber(row.total_pnl)}
                    </td>
                    <td data-label="Avg P&L">{formatNumber(row.avg_pnl)}</td>
                    <td data-label="Pairs">{row.pairs_traded}</td>
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
