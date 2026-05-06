"use client";

import { epicToPair, formatDateTime, getStrategyLabel } from "../../../../lib/backtests";
import type { JobRow } from "../_lib/types";

type Props = {
  jobs: JobRow[];
  loading: boolean;
  onSelectJob: (jobId: string) => void;
};

export default function BacktestQueuedJobsTable({ jobs, loading, onSelectJob }: Props) {
  return (
    <div className="panel table-panel">
      <div className="chart-title">Queued Jobs</div>
      {loading ? (
        <div className="chart-placeholder">Loading jobs...</div>
      ) : (
        <table className="forex-table">
          <thead>
            <tr>
              <th>Job</th>
              <th>Pair</th>
              <th>Status</th>
              <th>Strategy</th>
              <th>Submitted</th>
              <th>Execution</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.job_id} onClick={() => onSelectJob(job.job_id)} style={{ cursor: "pointer" }}>
                <td>{job.job_id}</td>
                <td>{epicToPair(job.epic)}</td>
                <td>{job.status}</td>
                <td>
                  {getStrategyLabel(job.strategy)} · {job.timeframe}
                </td>
                <td>{formatDateTime(job.submitted_at)}</td>
                <td>{job.execution_id ? `#${job.execution_id}` : "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
