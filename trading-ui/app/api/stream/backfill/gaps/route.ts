import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export async function GET() {
  try {
    const res = await fetch(`${BASE_URL}/backfill/gaps`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });

    // Upstream returns { report: string, statistics: { gaps_by_epic: { epic: { tf: count } } } }
    // Convert to BackfillGapItem[] so the page can filter by epic
    const gapsByEpic = data?.statistics?.gaps_by_epic ?? {};
    const gaps = Object.entries(gapsByEpic).flatMap(([epicKey, tfs]) =>
      Object.entries(tfs as Record<string, unknown>)
        .filter(([k]) => k !== "missing_candles" && !isNaN(Number(k)))
        .map(([tf, count]) => ({
          epic: epicKey,
          timeframe: `${tf}m`,
          missing_bars: count as number,
        }))
        .filter(g => (g.missing_bars as number) > 0)
    );
    return NextResponse.json(gaps);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
