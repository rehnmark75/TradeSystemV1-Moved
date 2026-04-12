import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const limit = searchParams.get("limit") ?? "50";
  const active_only = searchParams.get("active_only") ?? "false";
  try {
    const res = await fetch(`${BASE_URL}/api/v1/alerts?limit=${limit}&active_only=${active_only}`, {
      cache: "no-store", signal: AbortSignal.timeout(5000)
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
