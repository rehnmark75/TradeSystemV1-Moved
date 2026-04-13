import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const epic = searchParams.get("epic");
  const timeframe = parseInt(searchParams.get("timeframe") ?? "5", 10);
  const limit = parseInt(searchParams.get("limit") ?? "5000", 10);
  const from = searchParams.get("from");
  const to = searchParams.get("to");

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }

  try {
    const params: (string | number)[] = [epic, timeframe];
    let dateFilter = "";

    if (from) {
      params.push(from);
      dateFilter += ` AND start_time >= $${params.length}`;
    }
    if (to) {
      params.push(to);
      dateFilter += ` AND start_time <= $${params.length}`;
    }

    // Synthesize candles from 1m base data (live streaming table)
    params.push(limit);
    const limitClause = `LIMIT $${params.length}`;

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
        ${dateFilter}
      GROUP BY bucket
      ORDER BY bucket DESC
      ${limitClause}
      `,
      params
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
