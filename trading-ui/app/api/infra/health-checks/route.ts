import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET() {
  try {
    const res = await fetch(`${BASE_URL}/api/v1/health-checks`, { cache: "no-store", signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });

    // Normalize: upstream returns { services: { name: { status, response_time_ms, error, consecutive_failures, last_check } }, timestamp }
    const services = data.services ?? {};
    const checks = Object.entries(services).map(([name, s]) => {
      const svc = s as Record<string, unknown>;
      return {
        service: name,
        status: svc.status,
        latency_ms: svc.response_time_ms,
        consecutive_failures: svc.consecutive_failures,
        error: svc.error,
        last_run: svc.last_check,
      };
    });
    return NextResponse.json(checks);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
