import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export async function GET(_req: Request, { params }: { params: { epic: string } }) {
  try {
    const res = await fetch(
      `${BASE_URL}/stream/candle/latest/${encodeURIComponent(params.epic)}?timeframe=5`,
      { cache: "no-store", signal: AbortSignal.timeout(5000) }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
