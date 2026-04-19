import { NextResponse } from "next/server";
import { forexPool } from "../../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const executionId = Number(id);

  if (!Number.isFinite(executionId) || executionId <= 0) {
    return NextResponse.json({ error: "Invalid execution ID" }, { status: 400 });
  }

  try {
    const [executionResult, signalsResult] = await Promise.all([
      forexPool.query(
        `
        SELECT
          be.id,
          be.execution_name,
          be.strategy_name,
          be.start_time,
          be.end_time,
          be.data_start_date,
          be.data_end_date,
          be.epics_tested,
          be.timeframes,
          be.status,
          be.total_candles_processed,
          be.execution_duration_seconds,
          be.chart_url,
          be.chart_object_name,
          be.config_snapshot,
          COALESCE(bs.signal_count, 0) AS signal_count,
          COALESCE(bs.win_count, 0) AS win_count,
          COALESCE(bs.loss_count, 0) AS loss_count,
          COALESCE(bs.total_pips, 0) AS total_pips,
          COALESCE(bs.avg_win, 0) AS avg_win,
          COALESCE(bs.avg_loss, 0) AS avg_loss
        FROM backtest_executions be
        LEFT JOIN (
          SELECT
            execution_id,
            COUNT(*) AS signal_count,
            SUM(CASE WHEN pips_gained > 0 THEN 1 ELSE 0 END) AS win_count,
            SUM(CASE WHEN pips_gained <= 0 THEN 1 ELSE 0 END) AS loss_count,
            COALESCE(SUM(pips_gained), 0) AS total_pips,
            COALESCE(AVG(CASE WHEN pips_gained > 0 THEN pips_gained END), 0) AS avg_win,
            COALESCE(AVG(CASE WHEN pips_gained <= 0 THEN pips_gained END), 0) AS avg_loss
          FROM backtest_signals
          GROUP BY execution_id
        ) bs ON bs.execution_id = be.id
        WHERE be.id = $1
        LIMIT 1
        `,
        [executionId]
      ),
      forexPool.query(
        `
        SELECT
          id,
          signal_timestamp,
          epic,
          signal_type,
          strategy_name,
          entry_price,
          exit_price,
          stop_loss_price,
          take_profit_price,
          pips_gained,
          trade_result,
          confidence_score,
          market_intelligence
        FROM backtest_signals
        WHERE execution_id = $1
        ORDER BY signal_timestamp ASC
        LIMIT 500
        `,
        [executionId]
      ),
    ]);

    const execution = executionResult.rows[0];
    if (!execution) {
      return NextResponse.json({ error: "Backtest execution not found" }, { status: 404 });
    }

    const signalCount = Number(execution.signal_count ?? 0);
    const winCount = Number(execution.win_count ?? 0);

    return NextResponse.json({
      execution: {
        ...execution,
        signal_count: signalCount,
        win_count: winCount,
        loss_count: Number(execution.loss_count ?? 0),
        total_pips: Number(execution.total_pips ?? 0),
        avg_win: Number(execution.avg_win ?? 0),
        avg_loss: Number(execution.avg_loss ?? 0),
        win_rate: signalCount > 0 ? (winCount / signalCount) * 100 : 0,
      },
      signals: signalsResult.rows,
    });
  } catch (error) {
    console.error("Failed to load backtest execution", error);
    return NextResponse.json({ error: "Failed to load backtest execution" }, { status: 500 });
  }
}
