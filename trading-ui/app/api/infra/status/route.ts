import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET() {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/status`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });

    // Normalize field names to match the SystemStatus interface used by the page
    return NextResponse.json({
      health_score: data.health_score,
      containers_running: data.running_containers ?? data.containers_running,
      containers_stopped: data.stopped_containers ?? data.containers_stopped,
      containers_unhealthy: data.unhealthy_containers ?? data.containers_unhealthy,
      active_alerts: data.active_alerts,
      ...data,
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
