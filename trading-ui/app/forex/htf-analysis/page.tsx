"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type AlignmentRow = {
  alignment: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  breakeven: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
};

type PatternRow = {
  pattern: string;
  signal_type: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
};

type PairRow = {
  pair: string | null;
  alignment: string;
  total_signals: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
};

type DistributionRow = {
  direction: string;
  signal_type: string;
  count: number;
};

type HtfPayload = {
  days: number;
  alignment: AlignmentRow[];
  patterns: PatternRow[];
  pairs: PairRow[];
  distribution: DistributionRow[];
};

const DAY_OPTIONS = [7, 14, 30, 60, 90];

const formatNumber = (value: number, digits = 1) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "0.0";

const formatPercent = (value: number) => (Number.isFinite(value) ? `${value.toFixed(1)}%` : "0%");

export default function ForexHtfAnalysisPage() {
  const { environment } = useEnvironment();
  const [days, setDays] = useState(30);
  const [payload, setPayload] = useState<HtfPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/htf-analysis/?days=${days}&env=${environment}`, {
      signal: controller.signal,
    })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load HTF analysis.");
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [days, environment]);

  const summary = useMemo(() => {
    const aligned = (payload?.alignment ?? []).find((row) => row.alignment === "ALIGNED");
    const counter = (payload?.alignment ?? []).find((row) => row.alignment === "COUNTER");
    return {
      alignedSignals: aligned?.total_signals ?? 0,
      counterSignals: counter?.total_signals ?? 0,
      alignedWinRate: aligned?.win_rate ?? 0,
      counterWinRate: counter?.win_rate ?? 0,
      pnlDiff: (aligned?.total_pnl ?? 0) - (counter?.total_pnl ?? 0),
    };
  }, [payload?.alignment]);

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
          <h1>HTF Analysis</h1>
          <p>Measure whether 4H candle alignment at signal time changes trade quality, pair behavior, and pattern edge.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/htf-analysis" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Analysis Window</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {DAY_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}d
                </option>
              ))}
            </select>
          </div>
          <div className="forex-badge">
            Environment
            <strong>{environment.toUpperCase()}</strong>
          </div>
          <div className="forex-badge">
            Pattern Rows
            <strong>{payload?.patterns?.length ?? 0}</strong>
          </div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        <div className="metrics-grid">
          <div className="summary-card">
            Aligned Signals
            <strong>{summary.alignedSignals}</strong>
          </div>
          <div className="summary-card">
            Counter Signals
            <strong>{summary.counterSignals}</strong>
          </div>
          <div className="summary-card">
            Aligned Win Rate
            <strong>{formatPercent(summary.alignedWinRate)}</strong>
          </div>
          <div className="summary-card">
            Counter Win Rate
            <strong>{formatPercent(summary.counterWinRate)}</strong>
          </div>
          <div className="summary-card">
            P&L Difference
            <strong>{formatNumber(summary.pnlDiff, 2)}</strong>
          </div>
        </div>

        <div className="forex-grid">
          <div className="panel table-panel">
            <div className="chart-title">Alignment Summary</div>
            {loading ? (
              <div className="chart-placeholder">Loading HTF alignment...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Alignment</th>
                    <th>Signals</th>
                    <th>Trades</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>BE</th>
                    <th>Win Rate</th>
                    <th>Total P&amp;L</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.alignment ?? []).map((row) => (
                    <tr key={row.alignment}>
                      <td>{row.alignment}</td>
                      <td>{row.total_signals}</td>
                      <td>{row.total_trades}</td>
                      <td>{row.wins}</td>
                      <td>{row.losses}</td>
                      <td>{row.breakeven}</td>
                      <td>{formatPercent(row.win_rate)}</td>
                      <td>{formatNumber(row.total_pnl, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="panel table-panel">
            <div className="chart-title">Direction Distribution</div>
            {loading ? (
              <div className="chart-placeholder">Loading distribution...</div>
            ) : (
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>4H Direction</th>
                    <th>Signal Type</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.distribution ?? []).map((row, index) => (
                    <tr key={`${row.direction}-${row.signal_type}-${index}`}>
                      <td>{row.direction}</td>
                      <td>{row.signal_type}</td>
                      <td>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Two-Candle Patterns</div>
          {loading ? (
            <div className="chart-placeholder">Loading pattern analysis...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Pattern</th>
                  <th>Signal</th>
                  <th>Signals</th>
                  <th>Trades</th>
                  <th>Wins</th>
                  <th>Losses</th>
                  <th>Win Rate</th>
                  <th>Total P&amp;L</th>
                  <th>Avg P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {(payload?.patterns ?? []).map((row) => (
                  <tr key={`${row.pattern}-${row.signal_type}`}>
                    <td>{row.pattern}</td>
                    <td>{row.signal_type}</td>
                    <td>{row.total_signals}</td>
                    <td>{row.total_trades}</td>
                    <td>{row.wins}</td>
                    <td>{row.losses}</td>
                    <td>{formatPercent(row.win_rate)}</td>
                    <td>{formatNumber(row.total_pnl, 2)}</td>
                    <td>{formatNumber(row.avg_pnl, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel table-panel">
          <div className="chart-title">By Pair</div>
          {loading ? (
            <div className="chart-placeholder">Loading pair breakdown...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Pair</th>
                  <th>Alignment</th>
                  <th>Signals</th>
                  <th>Trades</th>
                  <th>Wins</th>
                  <th>Losses</th>
                  <th>Win Rate</th>
                  <th>Total P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {(payload?.pairs ?? []).map((row, index) => (
                  <tr key={`${row.pair}-${row.alignment}-${index}`}>
                    <td>{row.pair ?? "N/A"}</td>
                    <td>{row.alignment}</td>
                    <td>{row.total_signals}</td>
                    <td>{row.total_trades}</td>
                    <td>{row.wins}</td>
                    <td>{row.losses}</td>
                    <td>{formatPercent(row.win_rate)}</td>
                    <td>{formatNumber(row.total_pnl, 2)}</td>
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
