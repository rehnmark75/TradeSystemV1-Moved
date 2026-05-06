"use client";

type Props = {
  queueStats: { queued: number; running: number; failed: number };
  executionStats: { executions: number; totalSignals: number; avgWinRate: number };
};

export default function BacktestSummaryCards({ queueStats, executionStats }: Props) {
  return (
    <div className="metrics-grid">
      <div className="summary-card">
        Queued Jobs
        <strong>{queueStats.queued}</strong>
      </div>
      <div className="summary-card">
        Running Jobs
        <strong>{queueStats.running}</strong>
      </div>
      <div className="summary-card">
        Recent Executions
        <strong>{executionStats.executions}</strong>
      </div>
      <div className="summary-card">
        Avg Win Rate
        <strong>{executionStats.avgWinRate.toFixed(1)}%</strong>
      </div>
    </div>
  );
}
