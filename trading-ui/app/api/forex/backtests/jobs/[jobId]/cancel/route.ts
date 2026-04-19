import { NextResponse } from "next/server";
import { forexPool } from "../../../../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function POST(
  _request: Request,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;

  try {
    const result = await forexPool.query(
      `
      UPDATE backtest_job_queue
      SET
        status = CASE WHEN status = 'pending' THEN 'cancelled' ELSE status END,
        completed_at = CASE WHEN status = 'pending' THEN NOW() ELSE completed_at END,
        cancel_requested_at = NOW(),
        progress = jsonb_build_object(
          'phase',
          CASE WHEN status = 'pending' THEN 'cancelled' ELSE 'cancelling' END,
          'last_activity',
          CASE WHEN status = 'pending' THEN 'Cancelled from trading-ui' ELSE 'Cancellation requested from trading-ui' END
        )
      WHERE job_id = $1
        AND status IN ('pending', 'running')
      RETURNING job_id, status, cancel_requested_at
      `,
      [jobId]
    );

    if (!result.rows[0]) {
      return NextResponse.json(
        { error: "Only pending or running jobs can be cancelled from trading-ui" },
        { status: 409 }
      );
    }

    return NextResponse.json({ job: result.rows[0] });
  } catch (error) {
    console.error("Failed to cancel backtest job", error);
    return NextResponse.json({ error: "Failed to cancel backtest job" }, { status: 500 });
  }
}
