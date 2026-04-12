import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET() {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/alerts/active`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    const alerts = (data.alerts ?? data ?? []).map((a: Record<string, unknown>) => ({
      id: a.id,
      severity: a.severity,
      source: a.container_name ?? a.source,
      message: a.message ?? a.title,
      created_at: a.created_at,
      status: a.resolved_at ? "resolved" : a.acknowledged_at ? "acknowledged" : "active",
    }));
    return NextResponse.json(alerts);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
