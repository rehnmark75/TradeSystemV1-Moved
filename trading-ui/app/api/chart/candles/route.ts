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
    const result = await forexPool.query(
      `
      SELECT start_time, open, high, low, close
      FROM ig_candles_backtest
      WHERE epic = $1 AND timeframe = $2
      ORDER BY start_time DESC
      LIMIT $3
      `,
      [epic, timeframe, limit]
    );

    const candles = result.rows.reverse().map((r) => ({
      start_time: r.start_time,
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
