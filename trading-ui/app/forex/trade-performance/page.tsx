/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type TradeRow = {
  id: number;
  symbol: string;
  entry_price: number | null;
  direction: string;
  timestamp: string;
  status: string;
  profit_loss: number | null;
  pnl_currency: string | null;
  strategy: string | null;
  trade_result: string;
  profit_loss_formatted: string;
};

type TradesPayload = {
  trades: TradeRow[];
};

const PRESETS = [
  { label: "7d", value: 7 },
  { label: "30d", value: 30 },
  { label: "90d", value: 90 }
];

const resultOrder = [
  "WIN",
  "LOSS",
  "BREAKEVEN",
  "OPEN",
  "PENDING",
  "EXPIRED",
  "REJECTED",
  "CANCELLED"
];

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

const formatNumber = (value: number, digits = 2) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      })
    : "0.00";

function calculateMetrics(rows: TradeRow[]) {
  const tradesWithPnl = rows.filter((row) => row.profit_loss != null);
  const pendingTrades = rows.filter((row) => ["pending", "pending_limit"].includes(row.status));
  const openTrades = rows.filter((row) => row.status === "tracking");
  const expiredTrades = rows.filter((row) => row.status === "limit_not_filled");
  const rejectedTrades = rows.filter((row) =>
    ["limit_rejected", "limit_cancelled"].includes(row.status)
  );
  const winningTrades = tradesWithPnl.filter((row) => (row.profit_loss ?? 0) > 0);
  const losingTrades = tradesWithPnl.filter((row) => (row.profit_loss ?? 0) < 0);
  const totalPnl = tradesWithPnl.reduce((acc, row) => acc + (row.profit_loss ?? 0), 0);
  const winRate = tradesWithPnl.length
    ? (winningTrades.length / tradesWithPnl.length) * 100
    : 0;

  return {
    total_trades: rows.length,
    completed_trades: tradesWithPnl.length,
    pending_trades: pendingTrades.length,
    open_trades: openTrades.length,
    expired_trades: expiredTrades.length,
    rejected_trades: rejectedTrades.length,
    total_pnl: totalPnl,
    win_rate: winRate
  };
}

export default function ForexTradePerformancePage() {
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<TradesPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [resultFilter, setResultFilter] = useState<string[]>([]);
  const [directionFilter, setDirectionFilter] = useState<string[]>([]);
  const [symbolFilter, setSymbolFilter] = useState<string[]>([]);
  const [strategyFilter, setStrategyFilter] = useState<string[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/stock-scanner/api/forex/trades/?days=${days}`, { signal: controller.signal })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load trade performance.");
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [days]);

  const trades = payload?.trades ?? [];
  const availableResults = useMemo(() => {
    const unique = Array.from(new Set(trades.map((row) => row.trade_result)));
    const ordered = resultOrder.filter((result) => unique.includes(result));
    return [...ordered, ...unique.filter((item) => !ordered.includes(item))];
  }, [trades]);

  const availableDirections = useMemo(
    () => Array.from(new Set(trades.map((row) => row.direction))).filter(Boolean),
    [trades]
  );
  const availableSymbols = useMemo(
    () => Array.from(new Set(trades.map((row) => row.symbol))).filter(Boolean),
    [trades]
  );
  const availableStrategies = useMemo(
    () => Array.from(new Set(trades.map((row) => row.strategy ?? "Unknown"))),
    [trades]
  );

  useEffect(() => {
    if (!resultFilter.length && availableResults.length) setResultFilter(availableResults);
  }, [availableResults, resultFilter.length]);
  useEffect(() => {
    if (!directionFilter.length && availableDirections.length) setDirectionFilter(availableDirections);
  }, [availableDirections, directionFilter.length]);
  useEffect(() => {
    if (!symbolFilter.length && availableSymbols.length) setSymbolFilter(availableSymbols);
  }, [availableSymbols, symbolFilter.length]);
  useEffect(() => {
    if (!strategyFilter.length && availableStrategies.length) setStrategyFilter(availableStrategies);
  }, [availableStrategies, strategyFilter.length]);

  const filteredTrades = trades.filter((row) => {
    if (resultFilter.length && !resultFilter.includes(row.trade_result)) return false;
    if (directionFilter.length && !directionFilter.includes(row.direction)) return false;
    if (symbolFilter.length && !symbolFilter.includes(row.symbol)) return false;
    const strategyValue = row.strategy ?? "Unknown";
    if (strategyFilter.length && !strategyFilter.includes(strategyValue)) return false;
    return true;
  });

  const metrics = useMemo(() => calculateMetrics(filteredTrades), [filteredTrades]);

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
          <h1>Trade Performance</h1>
          <p>Filter trades by result, direction, and strategy.</p>
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
            <label>Date Filter</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {PRESETS.map((preset) => (
                <option key={preset.value} value={preset.value}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
          <div className="forex-badge">{filteredTrades.length} trades</div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        <div className="metrics-grid">
          <div className="summary-card">
            Total P&L
            <strong className={metrics.total_pnl < 0 ? "bad" : "good"}>
              {metrics.total_pnl >= 0 ? "+" : ""}
              {formatNumber(metrics.total_pnl)} SEK
            </strong>
          </div>
          <div className="summary-card">
            Win Rate
            <strong>{metrics.win_rate.toFixed(1)}%</strong>
          </div>
          <div className="summary-card">
            Completed Trades
            <strong>{metrics.completed_trades}</strong>
          </div>
          <div className="summary-card">
            Pending/Open
            <strong>
              {metrics.pending_trades}/{metrics.open_trades}
            </strong>
          </div>
          <div className="summary-card">
            Expired/Rejected
            <strong>
              {metrics.expired_trades}/{metrics.rejected_trades}
            </strong>
          </div>
        </div>

        <div className="forex-filters">
          <div>
            <label>Result</label>
            <select
              multiple
              value={resultFilter}
              onChange={(event) =>
                setResultFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
              }
            >
              {availableResults.map((result) => (
                <option key={result} value={result}>
                  {result}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Direction</label>
            <select
              multiple
              value={directionFilter}
              onChange={(event) =>
                setDirectionFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
              }
            >
              {availableDirections.map((direction) => (
                <option key={direction} value={direction}>
                  {direction}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Symbol</label>
            <select
              multiple
              value={symbolFilter}
              onChange={(event) =>
                setSymbolFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
              }
            >
              {availableSymbols.map((symbol) => (
                <option key={symbol} value={symbol}>
                  {symbol}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Strategy</label>
            <select
              multiple
              value={strategyFilter}
              onChange={(event) =>
                setStrategyFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
              }
            >
              {availableStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Trade Details</div>
          {loading ? (
            <div className="chart-placeholder">Loading trades...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Symbol</th>
                  <th>Strategy</th>
                  <th>Direction</th>
                  <th>Entry</th>
                  <th>P&L</th>
                  <th>Result</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredTrades.slice(0, 50).map((row) => (
                  <tr key={row.id}>
                    <td>{formatDateTime(row.timestamp)}</td>
                    <td>{row.symbol}</td>
                    <td>{row.strategy ?? "Unknown"}</td>
                    <td>{row.direction}</td>
                    <td>{row.entry_price != null ? row.entry_price.toFixed(5) : "-"}</td>
                    <td className={row.profit_loss != null && row.profit_loss < 0 ? "bad" : "good"}>
                      {row.profit_loss_formatted}
                    </td>
                    <td>{row.trade_result}</td>
                    <td>{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
