/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type AlertRow = {
  id: number;
  alert_timestamp: string;
  epic: string | null;
  pair: string | null;
  signal_type: string | null;
  strategy: string | null;
  price: number | null;
  market_session: string | null;
  claude_score: number | null;
  claude_decision: string | null;
  claude_approved: boolean | null;
  claude_reason: string | null;
  claude_mode: string | null;
  claude_raw_response: string | null;
  vision_chart_url: string | null;
  status: string | null;
  alert_level: string | null;
  htf_candle_direction: string | null;
  htf_candle_direction_prev: string | null;
};

type AlertHistoryPayload = {
  filters: {
    strategies: string[];
    pairs: string[];
  };
  stats: {
    total_alerts: number;
    approved: number;
    rejected: number;
    avg_score: number;
    approval_rate: number;
  };
  alerts: AlertRow[];
  page: number;
  total_pages: number;
  total_alerts: number;
};

const DAYS = [1, 3, 7, 14, 30];
const STATUSES = ["All", "Approved", "Rejected"];

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

const resolvePair = (row: AlertRow) => {
  if (row.pair) return row.pair;
  if (!row.epic) return "N/A";
  const parts = row.epic.split(".");
  if (parts.length >= 3) return parts[2].slice(0, 6);
  return row.epic;
};

export default function ForexAlertHistoryPage() {
  const [days, setDays] = useState(1);
  const [status, setStatus] = useState("All");
  const [strategy, setStrategy] = useState("All");
  const [pair, setPair] = useState("All");
  const [page, setPage] = useState(1);
  const [payload, setPayload] = useState<AlertHistoryPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAlerts = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      days: String(days),
      status,
      strategy,
      pair,
      page: String(page),
      limit: "25"
    });
    fetch(`/trading/api/forex/alert-history/?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch(() => setError("Failed to load alert history."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAlerts();
  }, [days, status, strategy, pair, page]);

  const stats = payload?.stats;
  const alerts = payload?.alerts ?? [];
  const totalPages = payload?.total_pages ?? 1;

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
          <h1>Alert History</h1>
          <p>Claude-reviewed signals with detailed diagnostics and chart snapshots.</p>
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
            <select
              value={days}
              onChange={(event) => {
                setDays(Number(event.target.value));
                setPage(1);
              }}
            >
              {DAYS.map((option) => (
                <option key={option} value={option}>
                  {option}d
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Claude Status</label>
            <select
              value={status}
              onChange={(event) => {
                setStatus(event.target.value);
                setPage(1);
              }}
            >
              {STATUSES.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Strategy</label>
            <select
              value={strategy}
              onChange={(event) => {
                setStrategy(event.target.value);
                setPage(1);
              }}
            >
              {(payload?.filters?.strategies ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Pair</label>
            <select
              value={pair}
              onChange={(event) => {
                setPair(event.target.value);
                setPage(1);
              }}
            >
              {(payload?.filters?.pairs ?? ["All"]).map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={loadAlerts}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading alert history...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Total Alerts
                <strong>{stats?.total_alerts ?? 0}</strong>
              </div>
              <div className="summary-card">
                Approved
                <strong>{stats?.approved ?? 0}</strong>
              </div>
              <div className="summary-card">
                Rejected
                <strong>{stats?.rejected ?? 0}</strong>
              </div>
              <div className="summary-card">
                Approval Rate
                <strong>{stats ? `${stats.approval_rate.toFixed(1)}%` : "0%"}</strong>
              </div>
              <div className="summary-card">
                Avg Score
                <strong>{stats?.avg_score ? stats.avg_score.toFixed(1) : "N/A"}</strong>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Alerts</div>
              {alerts.length ? (
                alerts.map((row) => {
                  const approved =
                    row.claude_approved === true || row.claude_decision === "APPROVE";
                  const rejected =
                    row.claude_approved === false ||
                    row.claude_decision === "REJECT" ||
                    row.alert_level === "REJECTED";
                  const statusIcon = approved ? "✅" : rejected ? "❌" : "⚪";
                  const statusText = approved ? "APPROVED" : rejected ? "REJECTED" : "PENDING";
                  const pairLabel = resolvePair(row);
                  const score = row.claude_score != null ? `${row.claude_score}/10` : "N/A";
                  const title = `${statusIcon} ${formatDateTime(row.alert_timestamp)} | ${pairLabel} | ${
                    row.strategy ?? "N/A"
                  } | ${row.signal_type ?? "N/A"} | ${score}`;

                  return (
                    <details key={row.id} className="alert-row">
                      <summary>{title}</summary>
                      <div className="alert-grid">
                        <div>
                          <p>
                            <strong>Status:</strong> {statusText}
                          </p>
                          <p>
                            <strong>Pair:</strong> {pairLabel}
                          </p>
                          <p>
                            <strong>Strategy:</strong> {row.strategy ?? "N/A"}
                          </p>
                          <p>
                            <strong>Signal:</strong> {row.signal_type ?? "N/A"}
                          </p>
                          <p>
                            <strong>Price:</strong>{" "}
                            {row.price != null ? row.price.toFixed(5) : "N/A"}
                          </p>
                          <p>
                            <strong>Session:</strong> {row.market_session ?? "N/A"}
                          </p>
                          <p>
                            <strong>Claude Score:</strong> {score}
                          </p>
                          <p>
                            <strong>Claude Mode:</strong> {row.claude_mode ?? "N/A"}
                          </p>
                          {row.claude_reason ? (
                            <div className="alert-reason">
                              <strong>Claude Reason:</strong>
                              <p>{row.claude_reason}</p>
                            </div>
                          ) : null}
                        </div>
                        <div>
                          {row.vision_chart_url ? (
                            <img
                              src={row.vision_chart_url}
                              alt="Alert chart"
                              className="alert-chart"
                            />
                          ) : (
                            <div className="chart-placeholder">No chart image available.</div>
                          )}
                        </div>
                      </div>
                    </details>
                  );
                })
              ) : (
                <div className="chart-placeholder">No alerts found for the filters.</div>
              )}
            </div>

            <div className="panel">
              <div className="forex-controls">
                <button
                  className="section-tab"
                  disabled={page <= 1}
                  onClick={() => setPage((value) => Math.max(1, value - 1))}
                >
                  Previous
                </button>
                <div className="forex-badge">
                  Page {page} of {totalPages}
                </div>
                <button
                  className="section-tab"
                  disabled={page >= totalPages}
                  onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
