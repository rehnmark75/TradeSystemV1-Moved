/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type BrokerOverview = {
  error?: string;
  balance: {
    total_value: string;
    invested: string;
    available: string;
    recorded_at: string;
  } | null;
  trend: {
    change: number;
    change_pct: number;
    trend: "up" | "down" | "neutral";
    data_points: number;
  };
  last_sync: string | null;
  stats: {
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_profit: number;
    total_loss: number;
    net_profit: number;
    avg_profit: number;
    avg_loss: number;
    avg_profit_pct: number;
    avg_loss_pct: number;
    largest_win: number;
    largest_loss: number;
    profit_factor: number;
    expectancy: number;
    max_drawdown: number;
    max_drawdown_pct: number;
    max_consecutive_wins: number;
    max_consecutive_losses: number;
    avg_trade_duration_hours: number;
    long_trades: number;
    short_trades: number;
    long_win_rate: number;
    short_win_rate: number;
    long_profit: number;
    short_profit: number;
  };
  open_positions: Array<{
    deal_id: string;
    ticker: string;
    side: string;
    quantity: string;
    entry_price: number;
    current_price: number;
    unrealized_pnl: number;
    profit_pct: number;
    open_time: string;
  }>;
  closed_trades: Array<{
    deal_id: string;
    ticker: string;
    side: string;
    quantity: string;
    open_price: string;
    close_price: string;
    profit: string;
    profit_pct: string;
    duration_hours: string;
    open_time: string;
    close_time: string;
  }>;
  by_day: Array<{ date: string; pnl: number; count: number }>;
  by_ticker: Array<{ ticker: string; trades: number; win_rate: number; pnl: number }>;
  equity_curve: Array<{ recorded_at: string; total_value: string }>;
};

const formatMoney = (value: number) => `$${value.toFixed(2)}`;
const formatPct = (value: number) => `${value.toFixed(1)}%`;

export default function BrokerPage() {
  const [days, setDays] = useState(30);
  const [overview, setOverview] = useState<BrokerOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiPath = (path: string) => `../api/${path}`;

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      const res = await fetch(`${apiPath("broker/overview")}?days=${days}`);
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error || "Broker data not available.");
        setOverview(null);
      } else {
        setOverview(data);
      }
      setLoading(false);
    };
    load();
  }, [days]);

  const equityPoints = useMemo(() => {
    if (!overview?.equity_curve?.length) return "";
    const values = overview.equity_curve.map((p) => Number(p.total_value));
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (max === min) {
      return values.map((_, idx) => `${idx * 10},50`).join(" ");
    }
    return values
      .map((v, idx) => {
        const x = (idx / (values.length - 1)) * 300;
        const y = 80 - ((v - min) / (max - min)) * 60;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }, [overview]);

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
          <h1>Broker Trading Statistics</h1>
          <p>Performance data synced from RoboMarkets.</p>
        </div>
      </div>

      <div className="panel">
        <div className="controls">
          <div>
            <label>Analysis Period</label>
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {[7, 14, 30, 60, 90].map((value) => (
                <option key={value} value={value}>{value} days</option>
              ))}
            </select>
          </div>
        </div>

        {loading ? (
          <div className="footer-note">Loading broker stats...</div>
        ) : error ? (
          <div className="footer-note">
            {error} Sync data with: `docker exec task-worker python -m stock_scanner.main broker-sync`.
          </div>
        ) : overview ? (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Total Account Value
                <strong>{formatMoney(Number(overview.balance?.total_value || 0))}</strong>
                <span className={`trend ${overview.trend.trend}`}>
                  {formatMoney(overview.trend.change)} ({formatPct(overview.trend.change_pct)})
                </span>
              </div>
              <div className="summary-card">
                Invested
                <strong>{formatMoney(Number(overview.balance?.invested || 0))}</strong>
              </div>
              <div className="summary-card">
                Available Cash
                <strong>{formatMoney(Number(overview.balance?.available || 0))}</strong>
              </div>
              <div className="summary-card">
                Last Sync
                <strong>{overview.last_sync ? new Date(overview.last_sync).toLocaleString() : "N/A"}</strong>
              </div>
            </div>

            <div className="metrics-grid">
              <div className="summary-card">Net Profit<strong>{formatMoney(overview.stats.net_profit)}</strong></div>
              <div className="summary-card">Win Rate<strong>{formatPct(overview.stats.win_rate)}</strong></div>
              <div className="summary-card">Profit Factor<strong>{overview.stats.profit_factor.toFixed(2)}</strong></div>
              <div className="summary-card">Total Trades<strong>{overview.stats.total_trades}</strong></div>
              <div className="summary-card">Avg Win<strong>{formatMoney(overview.stats.avg_profit)}</strong></div>
              <div className="summary-card">Avg Loss<strong>{formatMoney(overview.stats.avg_loss)}</strong></div>
              <div className="summary-card">Largest Win<strong>{formatMoney(overview.stats.largest_win)}</strong></div>
              <div className="summary-card">Largest Loss<strong>{formatMoney(overview.stats.largest_loss)}</strong></div>
              <div className="summary-card">Expectancy<strong>{formatMoney(overview.stats.expectancy)}</strong></div>
              <div className="summary-card">Max Drawdown<strong>{formatMoney(overview.stats.max_drawdown)}</strong></div>
              <div className="summary-card">Win Streak<strong>{overview.stats.max_consecutive_wins}</strong></div>
              <div className="summary-card">Loss Streak<strong>{overview.stats.max_consecutive_losses}</strong></div>
            </div>

            <div className="broker-grid">
              <div className="broker-card">
                <h3>Open Positions</h3>
                {overview.open_positions.length ? (
                  <table>
                    <thead>
                      <tr>
                        <th>Ticker</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overview.open_positions.map((pos) => (
                        <tr key={pos.deal_id}>
                          <td>{pos.ticker}</td>
                          <td>{pos.side}</td>
                          <td>{Number(pos.quantity).toFixed(2)}</td>
                          <td>{formatMoney(pos.entry_price)}</td>
                          <td>{formatMoney(pos.current_price)}</td>
                          <td className={pos.unrealized_pnl >= 0 ? "positive" : "negative"}>{formatMoney(pos.unrealized_pnl)}</td>
                          <td className={pos.profit_pct >= 0 ? "positive" : "negative"}>{formatPct(pos.profit_pct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p>No open positions.</p>
                )}
              </div>
              <div className="broker-card">
                <h3>Performance by Side</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Long</th>
                      <th>Short</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Trades</td>
                      <td>{overview.stats.long_trades}</td>
                      <td>{overview.stats.short_trades}</td>
                    </tr>
                    <tr>
                      <td>Win Rate</td>
                      <td>{formatPct(overview.stats.long_win_rate)}</td>
                      <td>{formatPct(overview.stats.short_win_rate)}</td>
                    </tr>
                    <tr>
                      <td>Profit</td>
                      <td>{formatMoney(overview.stats.long_profit)}</td>
                      <td>{formatMoney(overview.stats.short_profit)}</td>
                    </tr>
                  </tbody>
                </table>
                <div className="footer-note">Avg Hold Time: {overview.stats.avg_trade_duration_hours.toFixed(1)} hours</div>
              </div>
            </div>

            <div className="broker-grid">
              <div className="broker-card">
                <h3>Equity Curve</h3>
                {equityPoints ? (
                  <svg width="100%" height="100" viewBox="0 0 300 100">
                    <polyline fill="none" stroke="#0f4c5c" strokeWidth="2" points={equityPoints} />
                  </svg>
                ) : (
                  <p>No equity data.</p>
                )}
              </div>
              <div className="broker-card">
                <h3>Daily P&amp;L</h3>
                <div className="pnl-grid">
                  {overview.by_day.slice(0, 10).map((day) => (
                    <div key={day.date} className={`pnl-row ${day.pnl >= 0 ? "positive" : "negative"}`}>
                      <span>{day.date}</span>
                      <span>{formatMoney(day.pnl)}</span>
                      <span>{day.count} trades</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="broker-grid">
              <div className="broker-card">
                <h3>Performance by Ticker</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Trades</th>
                      <th>Win Rate</th>
                      <th>Net P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overview.by_ticker.slice(0, 15).map((row) => (
                      <tr key={row.ticker}>
                        <td>{row.ticker}</td>
                        <td>{row.trades}</td>
                        <td>{formatPct(row.win_rate)}</td>
                        <td className={row.pnl >= 0 ? "positive" : "negative"}>{formatMoney(row.pnl)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="broker-card">
                <h3>Recent Closed Trades</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Side</th>
                      <th>Profit</th>
                      <th>Close</th>
                    </tr>
                  </thead>
                  <tbody>
                    {overview.closed_trades.slice(0, 10).map((trade) => (
                      <tr key={trade.deal_id}>
                        <td>{trade.ticker}</td>
                        <td>{trade.side}</td>
                        <td className={Number(trade.profit) >= 0 ? "positive" : "negative"}>{formatMoney(Number(trade.profit))}</td>
                        <td>{trade.close_time ? new Date(trade.close_time).toLocaleDateString() : "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : (
          <div className="footer-note">No broker data available.</div>
        )}
      </div>
    </div>
  );
}
