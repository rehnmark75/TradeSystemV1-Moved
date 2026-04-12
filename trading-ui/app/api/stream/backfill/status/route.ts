import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export async function GET() {
  try {
    const res = await fetch(`${BASE_URL}/backfill/status`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    // Normalize: map is_running→running, map stats fields to page interface
    const stats = data.stats ?? {};
    const statistics = data.statistics ?? {};
    return NextResponse.json({
      ...data,
      running: data.is_running ?? data.running,
      status: data.is_running ? "running" : "idle",
      in_flight: stats.gaps_detected ?? 0,
      queued: 0,
      completed_last_24h: statistics.total_gaps_filled ?? stats.gaps_filled ?? 0,
      failed_last_24h: statistics.total_failures ?? stats.failures ?? 0,
      last_run: data.last_run,
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
