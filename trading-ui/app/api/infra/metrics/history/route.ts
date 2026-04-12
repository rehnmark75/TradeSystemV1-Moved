import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const hours = searchParams.get("hours") ?? "24";
  const container = searchParams.get("container") ?? "";
  const qs = `hours=${hours}${container ? `&container_name=${container}` : ""}`;
  try {
    const res = await fetch(`${BASE_URL}/api/v1/metrics/history?${qs}`, {
      cache: "no-store", signal: AbortSignal.timeout(5000)
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
