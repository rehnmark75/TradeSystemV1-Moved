import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get("limit") ?? "200";
  const level = searchParams.get("level") ?? "";
  try {
    const res = await fetch(
      `${BASE_URL}/stream/logs/recent?limit=${limit}${level ? `&level=${level}` : ""}`,
      { cache: "no-store", signal: AbortSignal.timeout(5000) }
    );
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
