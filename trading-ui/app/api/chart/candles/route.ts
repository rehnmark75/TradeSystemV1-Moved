import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";

const BASE_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const epic = searchParams.get("epic");
  const timeframe = searchParams.get("timeframe") ?? "5";
  const limit = searchParams.get("limit") ?? "1000";

  if (!epic) {
    return NextResponse.json({ error: "epic is required" }, { status: 400 });
  }

  try {
    const res = await fetch(
      `${BASE_URL}/stream/candles/${encodeURIComponent(epic)}?timeframe=${timeframe}&limit=${limit}`,
      { cache: "no-store", signal: AbortSignal.timeout(10000) }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
