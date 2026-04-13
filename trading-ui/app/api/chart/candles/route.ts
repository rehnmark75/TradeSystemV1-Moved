import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const epic = searchParams.get("epic");
  const timeframe = parseInt(searchParams.get("timeframe") ?? "5", 10);
  const limit = parseInt(searchParams.get("limit") ?? "1000", 10);

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }

  try {
    // Synthesize candles from 1m base data (live streaming table)
    // timeframe=5 → group 1m candles into 5m buckets, etc.
    const result = await forexPool.query(
      `
      SELECT
        date_trunc('minute', start_time) -
          (EXTRACT(minute FROM start_time)::int % $2) * INTERVAL '1 minute' AS bucket,
        (array_agg(open ORDER BY start_time ASC))[1]  AS open,
        MAX(high)                                      AS high,
        MIN(low)                                       AS low,
        (array_agg(close ORDER BY start_time DESC))[1] AS close
      FROM ig_candles
      WHERE epic = $1 AND timeframe = 1
      GROUP BY bucket
      ORDER BY bucket DESC
      LIMIT $3
      `,
      [epic, timeframe, limit]
    );

    const candles = result.rows.reverse().map((r) => ({
      start_time: r.bucket,
      open: Number(r.open),
      high: Number(r.high),
      low: Number(r.low),
      close: Number(r.close),
    }));

    return NextResponse.json(candles);
  } catch (error) {
    console.error("Failed to load candles", error);
    return NextResponse.json({ error: "Failed to load candles" }, { status: 500 });
  }
}
