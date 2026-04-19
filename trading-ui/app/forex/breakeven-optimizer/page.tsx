"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import EnvironmentToggle from "../../../components/EnvironmentToggle";
import ForexNav from "../_components/ForexNav";

type CacheRow = {
  epic: string;
  epic_display: string;
  direction: string;
  trade_count: number;
  win_rate: number;
  avg_mfe: number;
  median_mfe: number;
  percentile_25_mfe: number;
  percentile_75_mfe: number;
  avg_mae: number;
  median_mae: number;
  percentile_75_mae: number;
  percentile_95_mae: number;
  max_mae: number;
  optimal_be_trigger: number;
  conservative_be_trigger: number;
  current_be_trigger: number;
  optimal_stop_loss: number;
  current_stop_loss: number;
  configured_stop_loss: number;
  sl_recommendation: string;
  sl_priority: string;
  recommendation: string;
  priority: string;
  confidence: string;
  be_reach_rate: number;
  be_protection_rate: number;
  be_profit_rate: number;
  analysis_notes: string | null;
  analyzed_at: string;
  trades_analyzed: number[] | null;
  be_diff: number;
  sl_diff: number;
  sl_mismatch: boolean;
};

type SummaryPayload = {
  epicDirectionPairs: number;
  totalTradesAnalyzed: number;
  highPriorityBe: number;
  highPrioritySl: number;
  avgWinRate: number;
  analyzedAt: string | null;
};

type Payload = {
  cacheExists: boolean;
  summary: SummaryPayload | null;
  rows: CacheRow[];
};

const formatNumber = (value: number | null | undefined, digits = 1) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "N/A";

const formatWhole = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(0) : "N/A";

const formatPercent = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}%` : "N/A";

const formatSigned = (value: number | null | undefined) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return value > 0 ? `+${value.toFixed(0)}` : value.toFixed(0);
};

const formatDate = (value: string | null | undefined) => {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("sv-SE", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const priorityTone = (value: string | null | undefined) => {
  if (value === "high") return "warn";
  if (value === "medium") return "off";
  return "on";
};

export default function ForexBreakevenOptimizerPage() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch("/trading/api/forex/breakeven-optimizer/", { signal: controller.signal })
      .then((res) => res.json())
      .then((data) => setPayload(data))
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError("Failed to load breakeven optimizer cache.");
        }
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, []);

  const topBeRows = useMemo(
    () => [...(payload?.rows ?? [])].sort((a, b) => Math.abs(b.be_diff) - Math.abs(a.be_diff)).slice(0, 5),
    [payload?.rows]
  );

  const topSlRows = useMemo(
    () => [...(payload?.rows ?? [])].sort((a, b) => Math.abs(b.sl_diff) - Math.abs(a.sl_diff)).slice(0, 5),
    [payload?.rows]
  );

  const summary = payload?.summary;

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
          <h1>Breakeven Optimizer</h1>
          <p>Review cached MFE/MAE analysis to see where break-even triggers and initial stop-loss settings look misaligned with realized trade behavior.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/breakeven-optimizer" />

      {error ? <div className="error">{error}</div> : null}

      {!loading && payload && !payload.cacheExists ? (
        <div className="panel">
          <div className="chart-title">Unavailable</div>
          <div className="chart-placeholder">
            The `breakeven_analysis_cache` table is not present in the forex database yet.
          </div>
        </div>
      ) : null}

      <div className="metrics-grid">
        <div className="summary-card">
          Epic/Direction Pairs
          <strong>{summary?.epicDirectionPairs ?? 0}</strong>
        </div>
        <div className="summary-card">
          Trades Analyzed
          <strong>{summary?.totalTradesAnalyzed ?? 0}</strong>
        </div>
        <div className="summary-card">
          High Priority BE
          <strong>{summary?.highPriorityBe ?? 0}</strong>
        </div>
        <div className="summary-card">
          High Priority SL
          <strong>{summary?.highPrioritySl ?? 0}</strong>
        </div>
        <div className="summary-card">
          Avg Win Rate
          <strong>{formatPercent(summary?.avgWinRate)}</strong>
        </div>
      </div>

      <div className="panel">
        <div className="chart-title">Snapshot Status</div>
        <div className="stack-list">
          <div className="analysis-card">
            <strong>Cached analysis</strong>
            <p>This page currently reads the latest cached optimizer results from the old Streamlit workflow rather than rerunning the Python MFE/MAE analysis inside `trading-ui`.</p>
          </div>
          <div className="analysis-card">
            <strong>Last analyzed</strong>
            <p>{formatDate(summary?.analyzedAt)}</p>
          </div>
        </div>
      </div>

      <div className="forex-grid">
        <div className="panel table-panel">
          <div className="chart-title">Largest BE Gaps</div>
          {loading ? (
            <div className="chart-placeholder">Loading optimizer summary...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Epic</th>
                  <th>Dir</th>
                  <th>Current</th>
                  <th>Optimal</th>
                  <th>Diff</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {topBeRows.map((row) => (
                  <tr key={`${row.epic}-${row.direction}-be`}>
                    <td>{row.epic_display}</td>
                    <td>{row.direction}</td>
                    <td>{formatWhole(row.current_be_trigger)}</td>
                    <td>{formatWhole(row.optimal_be_trigger)}</td>
                    <td>{formatSigned(row.be_diff)}</td>
                    <td>{row.recommendation}</td>
                  </tr>
                ))}
                {!topBeRows.length ? (
                  <tr>
                    <td colSpan={6}>No cached break-even recommendations available.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel table-panel">
          <div className="chart-title">Largest SL Gaps</div>
          {loading ? (
            <div className="chart-placeholder">Loading stop-loss summary...</div>
          ) : (
            <table className="forex-table">
              <thead>
                <tr>
                  <th>Epic</th>
                  <th>Dir</th>
                  <th>Config</th>
                  <th>Optimal</th>
                  <th>Diff</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {topSlRows.map((row) => (
                  <tr key={`${row.epic}-${row.direction}-sl`}>
                    <td>{row.epic_display}</td>
                    <td>{row.direction}</td>
                    <td>{formatWhole(row.configured_stop_loss)}</td>
                    <td>{formatWhole(row.optimal_stop_loss)}</td>
                    <td>{formatSigned(row.sl_diff)}</td>
                    <td>{row.sl_recommendation}</td>
                  </tr>
                ))}
                {!topSlRows.length ? (
                  <tr>
                    <td colSpan={6}>No cached stop-loss recommendations available.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="panel table-panel">
        <div className="chart-title">Breakeven Recommendations</div>
        {loading ? (
          <div className="chart-placeholder">Loading break-even table...</div>
        ) : (
          <table className="forex-table">
            <thead>
              <tr>
                <th>Epic</th>
                <th>Dir</th>
                <th>Trades</th>
                <th>Win%</th>
                <th>Avg MFE</th>
                <th>Med MFE</th>
                <th>Avg MAE</th>
                <th>Optimal BE</th>
                <th>Current BE</th>
                <th>Diff</th>
                <th>Action</th>
                <th>Priority</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.rows ?? []).map((row) => (
                <tr key={`${row.epic}-${row.direction}-be-table`}>
                  <td>{row.epic_display}</td>
                  <td>{row.direction}</td>
                  <td>{row.trade_count}</td>
                  <td>{formatPercent(row.win_rate)}</td>
                  <td>{formatWhole(row.avg_mfe)}</td>
                  <td>{formatWhole(row.median_mfe)}</td>
                  <td>{formatWhole(row.avg_mae)}</td>
                  <td>{formatWhole(row.optimal_be_trigger)}</td>
                  <td>{formatWhole(row.current_be_trigger)}</td>
                  <td>{formatSigned(row.be_diff)}</td>
                  <td>{row.recommendation}</td>
                  <td>
                    <span className={`status-pill ${priorityTone(row.priority)}`}>{row.priority.toUpperCase()}</span>
                  </td>
                  <td>{row.confidence.toUpperCase()}</td>
                </tr>
              ))}
              {!payload?.rows?.length ? (
                <tr>
                  <td colSpan={13}>No cached optimizer rows found.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel table-panel">
        <div className="chart-title">Stop-Loss Recommendations</div>
        {loading ? (
          <div className="chart-placeholder">Loading stop-loss table...</div>
        ) : (
          <table className="forex-table">
            <thead>
              <tr>
                <th>Epic</th>
                <th>Dir</th>
                <th>Trades</th>
                <th>Avg MAE</th>
                <th>P95 MAE</th>
                <th>Optimal SL</th>
                <th>Config SL</th>
                <th>Actual SL</th>
                <th>Diff</th>
                <th>Action</th>
                <th>Priority</th>
              </tr>
            </thead>
            <tbody>
              {(payload?.rows ?? []).map((row) => (
                <tr key={`${row.epic}-${row.direction}-sl-table`}>
                  <td>{row.epic_display}</td>
                  <td>{row.direction}</td>
                  <td>{row.trade_count}</td>
                  <td>{formatWhole(row.avg_mae)}</td>
                  <td>{formatWhole(row.percentile_95_mae)}</td>
                  <td>{formatWhole(row.optimal_stop_loss)}</td>
                  <td>{formatWhole(row.configured_stop_loss)}</td>
                  <td>{`${formatWhole(row.current_stop_loss)}${row.sl_mismatch ? " !" : ""}`}</td>
                  <td>{formatSigned(row.sl_diff)}</td>
                  <td>{row.sl_recommendation}</td>
                  <td>
                    <span className={`status-pill ${priorityTone(row.sl_priority)}`}>{row.sl_priority.toUpperCase()}</span>
                  </td>
                </tr>
              ))}
              {!payload?.rows?.length ? (
                <tr>
                  <td colSpan={11}>No cached stop-loss rows found.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <div className="chart-title">Detailed Analysis</div>
        <div className="stack-list">
          {(payload?.rows ?? []).map((row) => (
            <div key={`${row.epic}-${row.direction}-detail`} className="analysis-card">
              <strong>
                {row.epic_display} {row.direction} ({row.trade_count} trades)
              </strong>
              <p>
                BE: current {formatNumber(row.current_be_trigger, 0)} vs optimal {formatNumber(row.optimal_be_trigger, 1)}.
                Suggested action: {row.recommendation}.
              </p>
              <p>
                SL: configured {formatNumber(row.configured_stop_loss, 0)}, actual {formatNumber(row.current_stop_loss, 0)},
                optimal {formatNumber(row.optimal_stop_loss, 1)}. Action: {row.sl_recommendation}.
              </p>
              <p>
                Efficiency: reach {formatPercent(row.be_reach_rate)}, protection {formatPercent(row.be_protection_rate)},
                profit continuation {formatPercent(row.be_profit_rate)}.
              </p>
              <p>
                Priority <span className={`status-pill ${priorityTone(row.priority)}`}>{row.priority.toUpperCase()}</span>{" "}
                and SL priority <span className={`status-pill ${priorityTone(row.sl_priority)}`}>{row.sl_priority.toUpperCase()}</span>.
              </p>
              {row.analysis_notes ? <p>{row.analysis_notes}</p> : null}
            </div>
          ))}
          {!loading && !payload?.rows?.length ? (
            <div className="analysis-card">
              <strong>No cached analysis available</strong>
              <p>The Streamlit optimizer cache is empty, so there is nothing useful to display yet.</p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
