import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET(req: Request, { params }: { params: { name: string } }) {
  const { searchParams } = new URL(req.url);
  const lines = searchParams.get("lines") ?? "100";
  try {
    // system-monitor uses ?tail= not ?lines=
    const res = await fetch(`${BASE_URL}/api/v1/containers/${encodeURIComponent(params.name)}/logs?tail=${lines}`, {
      cache: "no-store", signal: AbortSignal.timeout(10000)
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    // Upstream returns { logs: "<raw string>", ... } — split into array for the drawer
    const raw: string = data.logs ?? "";
    const log_lines = raw.split("\n").filter(Boolean);
    return NextResponse.json({ log_lines });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
