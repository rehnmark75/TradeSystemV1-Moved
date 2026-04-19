import { NextResponse } from "next/server";
import { forexPool } from "../../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;

  try {
    const result = await forexPool.query(
      `
      SELECT
        id,
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
        progress,
        recent_output,
        cancel_requested_at,
        start_date,
        end_date,
        submitted_at,
        started_at,
        completed_at,
        execution_id,
        error_message,
        submitted_by
      FROM backtest_job_queue
      WHERE job_id = $1
      LIMIT 1
      `,
      [jobId]
    );

    if (!result.rows[0]) {
      return NextResponse.json({ error: "Backtest job not found" }, { status: 404 });
    }

    return NextResponse.json({ job: result.rows[0] });
  } catch (error) {
    console.error("Failed to load backtest job", error);
    return NextResponse.json({ error: "Failed to load backtest job" }, { status: 500 });
  }
}
