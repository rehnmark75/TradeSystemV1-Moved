import { NextResponse } from "next/server";
import { forexPool } from "../../../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function POST(
  _request: Request,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;

  try {
    const sourceResult = await forexPool.query(
      `
      SELECT
        epic,
        days,
        strategy,
        timeframe,
        parallel,
        workers,
        chunk_days,
        generate_chart,
        pipeline_mode,
        parameter_overrides,
        snapshot_name,
        use_historical_intelligence,
        variation_config,
        start_date,
        end_date,
        submitted_by
      FROM backtest_job_queue
      WHERE job_id = $1
      LIMIT 1
      `,
      [jobId]
    );

    const source = sourceResult.rows[0];
    if (!source) {
      return NextResponse.json({ error: "Backtest job not found" }, { status: 404 });
    }

    const insertResult = await forexPool.query(
      `
      INSERT INTO backtest_job_queue (
        job_id,
        status,
        epic,
        days,
        strategy,
        timeframe,
        parallel,
        workers,
        chunk_days,
        generate_chart,
        pipeline_mode,
        parameter_overrides,
        snapshot_name,
        use_historical_intelligence,
        variation_config,
        start_date,
        end_date,
        submitted_by
      )
      SELECT
        CONCAT('tui_', TO_CHAR(NOW(), 'YYYYMMDD_HH24MISS_MS'), '_', SUBSTRING(MD5(RANDOM()::text) FROM 1 FOR 8)),
        'pending',
        epic,
        days,
        strategy,
        timeframe,
        parallel,
        workers,
        chunk_days,
        generate_chart,
        pipeline_mode,
        parameter_overrides,
        snapshot_name,
        use_historical_intelligence,
        variation_config,
        start_date,
        end_date,
        COALESCE(submitted_by, 'trading-ui')
      FROM backtest_job_queue
      WHERE job_id = $1
      RETURNING *
      `,
      [jobId]
    );

    return NextResponse.json({ job: insertResult.rows[0] }, { status: 201 });
  } catch (error) {
    console.error("Failed to retry backtest job", error);
    return NextResponse.json({ error: "Failed to retry backtest job" }, { status: 500 });
  }
}
