"use client";

import Link from "next/link";
import {
  epicToPair,
  formatDateTime,
  formatDuration,
  getStrategyBadgeStyle,
  getStrategyLabel,
} from "../../../../lib/backtests";
import type { ExecutionRow } from "../_lib/types";

type Props = {
  executions: ExecutionRow[];
  loading: boolean;
};

export default function BacktestExecutionsTable({ executions, loading }: Props) {
  return (
    <div className="panel table-panel">
      <div className="chart-title">Recent Executions</div>
      {loading ? (
        <div className="chart-placeholder">Loading executions...</div>
      ) : (
        <table className="forex-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Started</th>
              <th>Pair</th>
              <th>Strategy</th>
              <th>Signals</th>
              <th>Win Rate</th>
              <th>Total Pips</th>
              <th>Duration</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {executions.map((execution) => (
              <tr key={execution.id}>
                <td>{execution.id}</td>
                <td>{formatDateTime(execution.start_time)}</td>
                <td>{epicToPair(execution.epics_tested?.[0])}</td>
                <td>
                  <span
                    style={{
                      ...getStrategyBadgeStyle(execution.strategy_name),
                      border: "1px solid",
                      borderRadius: 999,
                      display: "inline-flex",
                      fontSize: 12,
                      fontWeight: 700,
                      lineHeight: 1,
                      padding: "5px 8px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {getStrategyLabel(execution.strategy_name)}
                  </span>
                </td>
                <td>{execution.signal_count}</td>
                <td>{execution.win_rate.toFixed(1)}%</td>
                <td>{execution.total_pips.toFixed(1)}</td>
                <td>{formatDuration(execution.execution_duration_seconds)}</td>
                <td>
                  <Link href={`/forex/backtests/${execution.id}`}>Open</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
