/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type MaeTrade = {
  id: number;
  symbol: string;
  symbol_short: string;
  direction: string;
  entry_price: number | null;
  timestamp: string;
  status: string;
  profit_loss: number | null;
  mfe_pips: number | null;
  mae_pips: number | null;
  mae_price: number | null;
  mae_time: string | null;
  virtual_sl_pips: number | null;
  vsl_stage: string | null;
  result: string;
  mae_pct_of_vsl: number | null;
};

type MaeSummary = {
  symbol: string;
  symbol_short: string;
  total_trades: number;
  win_rate: number;
  avg_mae_pips: number;
  median_mae_pips: number;
  p75_mae_pips: number;
  p90_mae_pips: number;
  max_mae_pips: number;
  avg_mfe_pips: number;
  avg_vsl_setting: number;
};

type MaePayload = {
  trades: MaeTrade[];
  summary: MaeSummary[];
};

const DAY_OPTIONS = [1, 3, 7, 14, 30];

const formatNumber = (value: number, digits = 1) =>
  Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
      })
    : "0.0";

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

export default function ForexMaeAnalysisPage() {
  const [days, setDays] = useState(7);
  const [payload, setPayload] = useState<MaePayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [pairFilter, setPairFilter] = useState<string[]>([]);
  const [resultFilter, setResultFilter] = useState<string[]>([]);

  const loadMae = () => {
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/mae-analysis/?days=${days}`)
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load MAE analysis."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadMae();
  }, [days]);

  const trades = payload?.trades ?? [];
  const summary = payload?.summary ?? [];

  const availablePairs = useMemo(
    () => Array.from(new Set(trades.map((trade) => trade.symbol_short))).filter(Boolean),
    [trades]
  );
  const availableResults = useMemo(
    () => Array.from(new Set(trades.map((trade) => trade.result))).filter(Boolean),
    [trades]
  );

  useEffect(() => {
    if (!pairFilter.length && availablePairs.length) setPairFilter(availablePairs);
  }, [availablePairs, pairFilter.length]);
  useEffect(() => {
    if (!resultFilter.length && availableResults.length) setResultFilter(availableResults);
  }, [availableResults, resultFilter.length]);

  const filteredTrades = trades.filter((trade) => {
    if (pairFilter.length && !pairFilter.includes(trade.symbol_short)) return false;
    if (resultFilter.length && !resultFilter.includes(trade.result)) return false;
    return true;
  });

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
          <h1>MAE Analysis (Scalp Trades)</h1>
          <p>Track maximum adverse excursion to refine virtual stop settings.</p>
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
          <button className="section-tab active" onClick={loadMae}>
            Refresh
          </button>
          <div className="forex-badge">{trades.length} trades</div>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading MAE analysis...</div>
        ) : (
          <>
            <div className="panel table-panel">
              <div className="chart-title">MAE Summary by Pair</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Pair</th>
                    <th>Trades</th>
                    <th>Win %</th>
                    <th>Avg MAE</th>
                    <th>Median MAE</th>
                    <th>75th %</th>
                    <th>90th %</th>
                    <th>Max MAE</th>
                    <th>Avg MFE</th>
                    <th>Avg VSL</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.map((row) => (
                    <tr key={row.symbol}>
                      <td>{row.symbol_short}</td>
                      <td>{row.total_trades}</td>
                      <td>{row.win_rate.toFixed(1)}%</td>
                      <td>{formatNumber(row.avg_mae_pips)}</td>
                      <td>{formatNumber(row.median_mae_pips)}</td>
                      <td>{formatNumber(row.p75_mae_pips)}</td>
                      <td>{formatNumber(row.p90_mae_pips)}</td>
                      <td>{formatNumber(row.max_mae_pips)}</td>
                      <td>{formatNumber(row.avg_mfe_pips)}</td>
                      <td>{formatNumber(row.avg_vsl_setting)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="panel">
              <div className="chart-title">Trade Filters</div>
              <div className="forex-filters">
                <div>
                  <label>Pair</label>
                  <select
                    multiple
                    value={pairFilter}
                    onChange={(event) =>
                      setPairFilter(Array.from(event.target.selectedOptions).map((opt) => opt.value))
                    }
                  >
                    {availablePairs.map((pair) => (
                      <option key={pair} value={pair}>
                        {pair}
                      </option>
                    ))}
                  </select>
                </div>
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
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Scalp Trade Details</div>
              {filteredTrades.length ? (
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Pair</th>
                      <th>Direction</th>
                      <th>Result</th>
                      <th>MAE</th>
                      <th>MFE</th>
                      <th>MAE % VSL</th>
                      <th>VSL Stage</th>
                      <th>P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTrades.slice(0, 50).map((row) => (
                      <tr key={row.id}>
                        <td>{formatDateTime(row.timestamp)}</td>
                        <td>{row.symbol_short}</td>
                        <td>{row.direction}</td>
                        <td>{row.result}</td>
                        <td>{formatNumber(row.mae_pips ?? 0)}</td>
                        <td>{formatNumber(row.mfe_pips ?? 0)}</td>
                        <td>{formatNumber(row.mae_pct_of_vsl ?? 0)}</td>
                        <td>{row.vsl_stage ?? "-"}</td>
                        <td className={row.profit_loss != null && row.profit_loss < 0 ? "bad" : "good"}>
                          {row.profit_loss != null ? formatNumber(row.profit_loss, 2) : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="chart-placeholder">No trades match the selected filters.</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
