/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

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
};

type AnalysisPayload = {
  strategies: StrategyRow[];
  pairs: PairRow[];
};

const PRESETS = [
  { label: "7d", value: 7 },
  { label: "30d", value: 30 },
  { label: "90d", value: 90 }
];

const formatNumber = (value: number, digits = 2) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      })
    : "0.00";

const formatPercent = (value: number) =>
  Number.isFinite(value) ? `${value.toFixed(1)}%` : "0%";

export default function ForexStrategyPage() {
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<AnalysisPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/stock-scanner/api/forex/analysis/?days=${days}`, {
      signal: controller.signal
    })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load strategy performance.");
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [days]);

  const maxPnl = useMemo(() => {
    if (!payload?.strategies?.length) return 1;
    return Math.max(
      1,
      ...payload.strategies.map((row) => Math.abs(row.total_pnl))
    );
  }, [payload?.strategies]);

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
          <h1>Strategy Performance</h1>
          <p>Compare strategies by P&L, win rate, and confidence.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <div className="forex-nav">
        <Link href="/forex" className="forex-pill">
          Overview
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
            <label>Analysis Window</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {PRESETS.map((preset) => (
                <option key={preset.value} value={preset.value}>
                  {preset.label}
                </option>
              ))}
            </select>
          </div>
          <div className="forex-badge">
            {payload?.strategies?.length ?? 0} strategies
          </div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        <div className="forex-grid">
          <div className="panel chart-panel">
            <div className="chart-title">Total P&L by Strategy</div>
            {loading ? (
              <div className="chart-placeholder">Loading strategies...</div>
            ) : (
              <div className="bar-stack">
                {(payload?.strategies ?? []).map((row) => (
                  <div key={row.strategy} className="bar-row">
                    <span>{row.strategy}</span>
                    <div className="bar-track">
                      <div
                        className={`bar-fill ${row.total_pnl >= 0 ? "good" : "bad"}`}
                        style={{ width: `${(Math.abs(row.total_pnl) / maxPnl) * 100}%` }}
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
            {loading ? (
              <div className="chart-placeholder">Loading strategies...</div>
            ) : (
              <div className="bar-stack">
                {(payload?.strategies ?? []).map((row) => (
                  <div key={`${row.strategy}-win`} className="bar-row">
                    <span>{row.strategy}</span>
                    <div className="bar-track">
                      <div className="bar-fill good" style={{ width: `${row.win_rate}%` }} />
                    </div>
                    <strong>{formatPercent(row.win_rate)}</strong>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Strategy Summary</div>
          <table className="forex-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Total P&L</th>
                <th>Avg P&L</th>
                <th>Avg Conf</th>
                <th>Pairs</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.strategies ?? []).map((row) => (
                <tr key={`${row.strategy}-table`}>
                  <td>{row.strategy}</td>
                  <td>{row.total_trades}</td>
                  <td>{formatPercent(row.win_rate)}</td>
                  <td className={row.total_pnl < 0 ? "bad" : "good"}>
                    {formatNumber(row.total_pnl)}
                  </td>
                  <td>{formatNumber(row.avg_pnl)}</td>
                  <td>{formatPercent(row.avg_confidence * 100)}</td>
                  <td>{row.pairs_traded}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Top Pairs</div>
          <table className="forex-table">
            <thead>
              <tr>
                <th>Pair</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Total P&L</th>
                <th>Avg P&L</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.pairs ?? []).slice(0, 12).map((row) => (
                <tr key={row.symbol}>
                  <td>{row.symbol}</td>
                  <td>{row.total_trades}</td>
                  <td>{formatPercent(row.win_rate)}</td>
                  <td className={row.total_pnl < 0 ? "bad" : "good"}>
                    {formatNumber(row.total_pnl)}
                  </td>
                  <td>{formatNumber(row.avg_pnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
