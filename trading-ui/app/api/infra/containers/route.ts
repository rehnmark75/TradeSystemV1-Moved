import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";
const BASE_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const include_stopped = searchParams.get("include_stopped") ?? "true";
  try {
    const res = await fetch(`${BASE_URL}/api/v1/containers?include_stopped=${include_stopped}`, {
      cache: "no-store", signal: AbortSignal.timeout(5000)
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });

    // Normalize: upstream returns { containers: [...], count, timestamp }
    // Map snake_case API fields → ContainerInfo camelCase interface
    const containers = (data.containers ?? data ?? []).map((c: Record<string, unknown>) => {
      const metrics = (c.metrics ?? {}) as Record<string, unknown>;
      return {
        name: c.name,
        image: c.image,
        // state = docker state (running/exited); status = health_status
        state: c.status,
        status: c.health_status,
        uptimeSeconds: c.uptime_seconds,
        restartCount: c.restart_count,
        cpuPercent: metrics.cpu_percent,
        memUsageMb: metrics.mem_usage_mb,
        memLimitMb: metrics.mem_limit_mb,
        is_critical: c.is_critical,
        warnings: c.warnings,
        errors: c.errors,
      };
    });
    return NextResponse.json(containers);
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
