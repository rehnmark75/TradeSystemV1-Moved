"use client";

import Link from "next/link";
import { epicToPair, formatDateTime, getStrategyLabel } from "../../../../lib/backtests";
import type { JobRow } from "../_lib/types";

type Props = {
  selectedJob: JobRow | null;
  loading: boolean;
  actionLoading: string | null;
  onAction: (jobId: string, action: "cancel" | "retry") => void;
};

export default function BacktestJobPanel({ selectedJob, loading, actionLoading, onAction }: Props) {
  return (
    <div className="panel table-panel">
      <div className="chart-title">Selected Job</div>
      {loading ? <div className="chart-placeholder">Loading job...</div> : null}
      {!loading && !selectedJob ? (
        <div className="chart-placeholder">Select a queued job to inspect status.</div>
      ) : null}
      {!loading && selectedJob ? (
        <div style={{ display: "grid", gap: 12 }}>
          <div className="forex-badge">
            {selectedJob.status.toUpperCase()}
            <strong>{epicToPair(selectedJob.epic)}</strong>
          </div>
          <div className="metrics-grid">
            <div className="summary-card">
              Job ID
              <strong>{selectedJob.job_id}</strong>
            </div>
            <div className="summary-card">
              Timeframe
              <strong>{selectedJob.timeframe}</strong>
            </div>
            <div className="summary-card">
              Submitted
              <strong>{formatDateTime(selectedJob.submitted_at)}</strong>
            </div>
            <div className="summary-card">
              Finished
              <strong>{formatDateTime(selectedJob.completed_at)}</strong>
            </div>
          </div>
          <div className="forex-badge">
            Config
            <strong>
              {selectedJob.days}d {getStrategyLabel(selectedJob.strategy)}{" "}
              {selectedJob.parallel ? "Parallel" : "Single"}
            </strong>
          </div>
          {selectedJob.snapshot_name ? (
            <div className="forex-badge">
              Snapshot
              <strong>{selectedJob.snapshot_name}</strong>
            </div>
          ) : null}
          {selectedJob.use_historical_intelligence ? (
            <div className="forex-badge">
              Historical Intelligence
              <strong>Enabled</strong>
            </div>
          ) : null}
          {selectedJob.variation_config?.enabled ? (
            <div className="forex-badge">
              Variation Run
              <strong>{Object.keys(selectedJob.variation_config.param_grid ?? {}).length} parameters</strong>
            </div>
          ) : null}
          {selectedJob.progress ? (
            <div className="panel table-panel">
              <div className="chart-title">Live Progress</div>
              <table className="forex-table">
                <tbody>
                  <tr>
                    <td>Phase</td>
                    <td>{selectedJob.progress.phase ?? "running"}</td>
                  </tr>
                  <tr>
                    <td>Elapsed</td>
                    <td>
                      {selectedJob.progress.elapsed_seconds != null
                        ? `${Math.round(selectedJob.progress.elapsed_seconds)}s`
                        : "N/A"}
                    </td>
                  </tr>
                  <tr>
                    <td>Activity</td>
                    <td>{selectedJob.progress.last_activity ?? "N/A"}</td>
                  </tr>
                  {selectedJob.progress.current && selectedJob.progress.total ? (
                    <tr>
                      <td>Variation Progress</td>
                      <td>
                        {selectedJob.progress.current}/{selectedJob.progress.total}
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
              {selectedJob.recent_output?.length ? (
                <pre style={{ whiteSpace: "pre-wrap", marginTop: 12 }}>
                  {selectedJob.recent_output.slice(-10).join("\n")}
                </pre>
              ) : null}
            </div>
          ) : null}
          {selectedJob.execution_id ? (
            <Link
              href={`/forex/backtests/${selectedJob.execution_id}`}
              className="alert-history-button alert-history-button-active"
            >
              Open Execution #{selectedJob.execution_id}
            </Link>
          ) : null}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {selectedJob.status === "pending" || selectedJob.status === "running" ? (
              <button
                className="alert-history-button"
                disabled={
                  actionLoading === `cancel:${selectedJob.job_id}` ||
                  Boolean(selectedJob.cancel_requested_at)
                }
                onClick={() => onAction(selectedJob.job_id, "cancel")}
              >
                {actionLoading === `cancel:${selectedJob.job_id}` || selectedJob.cancel_requested_at
                  ? "Cancelling..."
                  : selectedJob.status === "running"
                    ? "Cancel Running Job"
                    : "Cancel Pending Job"}
              </button>
            ) : null}
            {selectedJob.status === "failed" ||
            selectedJob.status === "completed" ||
            selectedJob.status === "cancelled" ? (
              <button
                className="alert-history-button"
                disabled={actionLoading === `retry:${selectedJob.job_id}`}
                onClick={() => onAction(selectedJob.job_id, "retry")}
              >
                {actionLoading === `retry:${selectedJob.job_id}` ? "Retrying..." : "Retry Job"}
              </button>
            ) : null}
          </div>
          {selectedJob.error_message ? <div className="error">{selectedJob.error_message}</div> : null}
        </div>
      ) : null}
    </div>
  );
}
