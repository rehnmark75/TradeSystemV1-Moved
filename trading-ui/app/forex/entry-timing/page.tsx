/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type EntryTimingTrade = {
  id: number;
  symbol: string;
  symbol_short: string;
  direction: string;
  entry_price: number | null;
  trade_timestamp: string;
  status: string;
  profit_loss: number | null;
  mfe_pips: number | null;
  mae_pips: number | null;
  mae_timestamp: string | null;
  virtual_sl_pips: number | null;
  vsl_stage: string | null;
  closed_at: string | null;
  result: string;
  zero_mfe: boolean;
  slippage_pips: number | null;
  time_to_mae_seconds: number | null;
  signal_price: number | null;
  signal_trigger: string | null;
  entry_type: string | null;
  pullback_depth: number | null;
  confidence_score: number | null;
  market_session: string | null;
};

type EntryTimingSummary = {
  entry_type: string;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_pnl: number;
  total_pnl: number;
  avg_mae_pips: number | null;
  avg_mfe_pips: number | null;
  zero_mfe_pct: number;
  avg_confidence: number | null;
  avg_pullback_depth: number | null;
};

type EntryTimingTriggerSummary = {
  signal_trigger: string;
  entry_type: string;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_pnl: number;
  total_pnl: number;
  avg_mae_pips: number | null;
  avg_mfe_pips: number | null;
  zero_mfe_pct: number;
  avg_confidence: number | null;
};

type EntryTimingPayload = {
  trades: EntryTimingTrade[];
  summary: EntryTimingSummary[];
  by_trigger: EntryTimingTriggerSummary[];
};

const DAY_OPTIONS = [1, 3, 7, 14, 30];

const formatNumber = (value: number, digits = 2) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      })
    : "0.00";

const formatPercent = (value: number) =>
  Number.isFinite(value) ? `${value.toFixed(1)}%` : "0%";

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

export default function ForexEntryTimingPage() {
  const [days, setDays] = useState(7);
  const [payload, setPayload] = useState<EntryTimingPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTiming = () => {
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/entry-timing/?days=${days}`)
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load entry timing analysis."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadTiming();
  }, [days]);

  const trades = payload?.trades ?? [];
  const summary = payload?.summary ?? [];
  const byTrigger = payload?.by_trigger ?? [];

  const closedTrades = useMemo(
    () => trades.filter((trade) => ["WIN", "LOSS"].includes(trade.result)),
    [trades]
  );

  const zeroMfeTrades = useMemo(
    () => closedTrades.filter((trade) => trade.zero_mfe),
    [closedTrades]
  );

  const zeroMfePct = closedTrades.length
    ? (zeroMfeTrades.length / closedTrades.length) * 100
    : 0;

  const slippageTrades = useMemo(
    () => trades.filter((trade) => trade.slippage_pips != null),
    [trades]
  );

  const avgSlippage = slippageTrades.length
    ? slippageTrades.reduce((acc, row) => acc + (row.slippage_pips ?? 0), 0) / slippageTrades.length
    : 0;

  const worstSlippage = useMemo(() => {
    return [...slippageTrades]
      .sort((a, b) => Math.abs(b.slippage_pips ?? 0) - Math.abs(a.slippage_pips ?? 0))
      .slice(0, 8);
  }, [slippageTrades]);

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
          <h1>Entry Timing Analysis</h1>
          <p>Diagnose timing quality, zero-MFE trades, and entry slippage.</p>
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
      </div>

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Time Period</label>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {DAY_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}d
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={loadTiming}>
            Refresh
          </button>
          <div className="forex-badge">{trades.length} trades</div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading entry timing data...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Closed Trades
                <strong>{closedTrades.length}</strong>
              </div>
              <div className="summary-card">
                Zero MFE Trades
                <strong>{zeroMfeTrades.length}</strong>
              </div>
              <div className="summary-card">
                Zero MFE %
                <strong className={zeroMfePct > 50 ? "bad" : "good"}>
                  {formatPercent(zeroMfePct)}
                </strong>
              </div>
              <div className="summary-card">
                Avg Slippage (pips)
                <strong>{formatNumber(avgSlippage, 1)}</strong>
              </div>
              <div className="summary-card">
                Zero MFE Wins
                <strong>
                  {zeroMfeTrades.filter((trade) => trade.result === "WIN").length}
                </strong>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Performance by Entry Type</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Entry Type</th>
                    <th>Trades</th>
                    <th>Win %</th>
                    <th>Avg P&amp;L</th>
                    <th>Total P&amp;L</th>
                    <th>Avg MAE</th>
                    <th>Avg MFE</th>
                    <th>Zero MFE %</th>
                    <th>Avg Conf</th>
                    <th>Avg Pullback</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => (
                    <tr key={row.entry_type}>
                      <td>{row.entry_type}</td>
                      <td>{row.total_trades}</td>
                      <td>{formatPercent(row.win_rate)}</td>
                      <td>{formatNumber(row.avg_pnl)}</td>
                      <td>{formatNumber(row.total_pnl)}</td>
                      <td>{formatNumber(row.avg_mae_pips ?? 0, 1)}</td>
                      <td>{formatNumber(row.avg_mfe_pips ?? 0, 1)}</td>
                      <td>{formatPercent(row.zero_mfe_pct)}</td>
                      <td>{formatNumber(row.avg_confidence ?? 0, 2)}</td>
                      <td>{formatNumber(row.avg_pullback_depth ?? 0, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Performance by Signal Trigger</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Trigger</th>
                    <th>Entry Type</th>
                    <th>Trades</th>
                    <th>Win %</th>
                    <th>Avg P&amp;L</th>
                    <th>Total P&amp;L</th>
                    <th>Avg MAE</th>
                    <th>Avg MFE</th>
                    <th>Zero MFE %</th>
                    <th>Avg Conf</th>
                  </tr>
                </thead>
                <tbody>
                  {byTrigger.map((row, index) => (
                    <tr key={`${row.signal_trigger}-${row.entry_type}-${index}`}>
                      <td>{row.signal_trigger}</td>
                      <td>{row.entry_type}</td>
                      <td>{row.total_trades}</td>
                      <td>{formatPercent(row.win_rate)}</td>
                      <td>{formatNumber(row.avg_pnl)}</td>
                      <td>{formatNumber(row.total_pnl)}</td>
                      <td>{formatNumber(row.avg_mae_pips ?? 0, 1)}</td>
                      <td>{formatNumber(row.avg_mfe_pips ?? 0, 1)}</td>
                      <td>{formatPercent(row.zero_mfe_pct)}</td>
                      <td>{formatNumber(row.avg_confidence ?? 0, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Worst Slippage Samples</div>
              {worstSlippage.length ? (
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Pair</th>
                      <th>Direction</th>
                      <th>Slippage (pips)</th>
                      <th>Entry</th>
                      <th>Signal</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {worstSlippage.map((row) => (
                      <tr key={row.id}>
                        <td>{formatDateTime(row.trade_timestamp)}</td>
                        <td>{row.symbol_short}</td>
                        <td>{row.direction}</td>
                        <td>{formatNumber(row.slippage_pips ?? 0, 1)}</td>
                        <td>{row.entry_price != null ? row.entry_price.toFixed(5) : "-"}</td>
                        <td>{row.signal_price != null ? row.signal_price.toFixed(5) : "-"}</td>
                        <td>{row.result}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="chart-placeholder">No slippage data available.</div>
              )}
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Trade Details</div>
              {trades.length ? (
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Pair</th>
                      <th>Entry Type</th>
                      <th>Trigger</th>
                      <th>Result</th>
                      <th>MAE</th>
                      <th>MFE</th>
                      <th>Slippage</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.slice(0, 50).map((row) => (
                      <tr key={row.id}>
                        <td>{formatDateTime(row.trade_timestamp)}</td>
                        <td>{row.symbol_short}</td>
                        <td>{row.entry_type ?? "-"}</td>
                        <td>{row.signal_trigger ?? "-"}</td>
                        <td>{row.result}</td>
                        <td>{formatNumber(row.mae_pips ?? 0, 1)}</td>
                        <td>{formatNumber(row.mfe_pips ?? 0, 1)}</td>
                        <td>{formatNumber(row.slippage_pips ?? 0, 1)}</td>
                        <td>{formatNumber(row.confidence_score ?? 0, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="chart-placeholder">No entry timing data yet.</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
