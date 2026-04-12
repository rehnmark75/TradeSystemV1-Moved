import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get("limit") ?? "50";
  try {
    const res = await fetch(`${BASE_URL}/stream/alerts/recent?limit=${limit}`, {
      cache: "no-store", signal: AbortSignal.timeout(5000)
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    // Upstream returns { alerts: [...], total_count, timestamp } — extract the array
    return NextResponse.json(data.alerts ?? data);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
