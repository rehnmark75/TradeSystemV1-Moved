"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import ForexNav from "../../_components/ForexNav";
import EnvironmentToggle from "../../../../components/EnvironmentToggle";

type ExecutionSignal = {
  id: number;
  signal_timestamp: string;
  epic: string | null;
  signal_type: string | null;
  entry_price: number | null;
  exit_price: number | null;
  pips_gained: number | null;
  trade_result: string | null;
  confidence_score: number | null;
};

type ExecutionPayload = {
  execution: {
    id: number;
    strategy_name: string | null;
    status: string;
    start_time: string;
    end_time: string | null;
    epics_tested: string[] | null;
    execution_duration_seconds: number | null;
    chart_url: string | null;
    config_snapshot: Record<string, unknown> | null;
    signal_count: number;
    win_count: number;
    loss_count: number;
    total_pips: number;
    avg_win: number;
    avg_loss: number;
    win_rate: number;
  };
  signals: ExecutionSignal[];
};

const formatDateTime = (value: string | null | undefined) => {
  if (!value) return "N/A";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const pairFromEpic = (epic: string | null | undefined) => {
  if (!epic) return "N/A";
  const parts = epic.split(".");
  if (parts.length >= 3) return parts[2].slice(0, 6);
  return epic;
};

export default function ForexBacktestExecutionPage() {
  const params = useParams<{ id: string }>();
  const executionId = params?.id ?? "";
  const [payload, setPayload] = useState<ExecutionPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!executionId) return;
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/backtests/executions/${executionId}/`)
      .then((res) => res.json())
      .then((data) => {
        if (data.error) throw new Error(data.error);
        setPayload(data);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load execution"))
      .finally(() => setLoading(false));
  }, [executionId]);

  const execution = payload?.execution;

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
          <h1>Backtest Execution #{executionId || "..."}</h1>
          <p>Detailed execution metrics, chart access, and signal-level review.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/backtests" />

      <div className="panel">
        <div style={{ marginBottom: 16 }}>
          <Link href="/forex/backtests">← Back to Backtests</Link>
        </div>

        {loading ? <div className="chart-placeholder">Loading execution...</div> : null}
        {error ? <div className="error">{error}</div> : null}

        {!loading && !error && execution ? (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Pair
                <strong>{pairFromEpic(execution.epics_tested?.[0])}</strong>
              </div>
              <div className="summary-card">
                Strategy
                <strong>{execution.strategy_name ?? "N/A"}</strong>
              </div>
              <div className="summary-card">
                Signals
                <strong>{execution.signal_count}</strong>
              </div>
              <div className="summary-card">
                Win Rate
                <strong>{execution.win_rate.toFixed(1)}%</strong>
              </div>
              <div className="summary-card">
                Total Pips
                <strong>{execution.total_pips.toFixed(1)}</strong>
              </div>
              <div className="summary-card">
                Status
                <strong>{execution.status}</strong>
              </div>
            </div>

            <div className="forex-grid">
              <div className="panel table-panel">
                <div className="chart-title">Execution Summary</div>
                <table className="forex-table">
                  <tbody>
                    <tr>
                      <td>Started</td>
                      <td>{formatDateTime(execution.start_time)}</td>
                    </tr>
                    <tr>
                      <td>Ended</td>
                      <td>{formatDateTime(execution.end_time)}</td>
                    </tr>
                    <tr>
                      <td>Duration</td>
                      <td>{execution.execution_duration_seconds ?? 0}s</td>
                    </tr>
                    <tr>
                      <td>Wins / Losses</td>
                      <td>
                        {execution.win_count} / {execution.loss_count}
                      </td>
                    </tr>
                    <tr>
                      <td>Average Win</td>
                      <td>{execution.avg_win.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td>Average Loss</td>
                      <td>{execution.avg_loss.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td>Chart</td>
                      <td>
                        {execution.chart_url ? (
                          <a
                            href={`/trading/api/forex/chart-image/?url=${encodeURIComponent(execution.chart_url)}`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Open Chart
                          </a>
                        ) : (
                          "No chart"
                        )}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="panel table-panel">
                <div className="chart-title">Config Snapshot</div>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                  {JSON.stringify(execution.config_snapshot ?? {}, null, 2)}
                </pre>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">Signals</div>
              <table className="forex-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Pair</th>
                    <th>Type</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Pips</th>
                    <th>Result</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(payload?.signals ?? []).map((signal) => (
                    <tr key={signal.id}>
                      <td>{formatDateTime(signal.signal_timestamp)}</td>
                      <td>{pairFromEpic(signal.epic)}</td>
                      <td>{signal.signal_type ?? "N/A"}</td>
                      <td>{signal.entry_price ?? "-"}</td>
                      <td>{signal.exit_price ?? "-"}</td>
                      <td>{signal.pips_gained ?? "-"}</td>
                      <td>{signal.trade_result ?? "-"}</td>
                      <td>{signal.confidence_score ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
