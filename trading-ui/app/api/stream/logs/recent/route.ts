import { NextResponse } from "next/server";
export const dynamic = "force-dynamic";

const STREAM_URL  = process.env.FASTAPI_STREAM_URL  ?? "http://fastapi-stream:8003";
const MONITOR_URL = process.env.SYSTEM_MONITOR_URL  ?? "http://system-monitor:8095";

// Map UI service names → Docker container names
const SERVICE_TO_CONTAINER: Record<string, string> = {
  stream:        "fastapi-stream",
  worker:        "task-worker",
  "worker-live": "task-worker-live",
  dev:           "fastapi-dev",
  live:          "fastapi-live",
  scanner:       "stock-scanner",
};

// Parse a raw Docker log line into a structured object.
// Format: "{dockerTs}Z {date} {time} {tz} - {LEVEL} - {message}"
// Some lines (separators, etc.) won't match — those are kept as INFO.
function parseDockerLine(raw: string, service: string) {
  // Strip the Docker nanosecond timestamp prefix (everything up to the first space)
  const withoutDockerTs = raw.replace(/^\S+Z\s*/, "");

  // Try to extract level from " - LEVEL - " pattern
  const levelMatch = withoutDockerTs.match(/\s-\s(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s-\s(.+)$/i);
  if (levelMatch) {
    // The timestamp is the human-readable part before " - LEVEL -"
    const tsRaw = withoutDockerTs.slice(0, withoutDockerTs.indexOf(` - ${levelMatch[1]}`)).trim();
    // tsRaw is like "2026-04-12 17:10:05 CEST"
    const ts = (() => {
      try { return new Date(tsRaw.replace(" CEST", "+02:00").replace(" CET", "+01:00")).toISOString(); }
      catch { return new Date().toISOString(); }
    })();
    return {
      timestamp: ts,
      level: levelMatch[1].replace("WARNING", "WARN"),
      service,
      message: levelMatch[2].trim(),
    };
  }

  // Fallback: no level pattern — extract just the Docker timestamp
  const dockerTs = raw.match(/^(\S+Z)/)?.[1];
  return {
    timestamp: dockerTs ?? new Date().toISOString(),
    level: "INFO",
    service,
    message: withoutDockerTs.trim() || raw.trim(),
  };
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const limit   = parseInt(searchParams.get("limit")   ?? "100", 10);
  const level   = searchParams.get("level")   ?? "";
  const service = searchParams.get("service") ?? "";

  try {
    // If a specific container service is requested, fetch from system-monitor
    if (service && service !== "stream") {
      const container = SERVICE_TO_CONTAINER[service] ?? service;
      const res = await fetch(
        `${MONITOR_URL}/api/v1/containers/${encodeURIComponent(container)}/logs?tail=${limit}`,
        { cache: "no-store", signal: AbortSignal.timeout(8000) }
      );
      const data = await res.json();
      if (!res.ok) return NextResponse.json({ error: data.detail ?? "upstream error" }, { status: res.status });

      const rawLines: string[] = (data.logs ?? "").split("\n").filter(Boolean);
      let lines = rawLines.map(l => parseDockerLine(l, service));

      // Apply level filter if requested
      if (level) {
        const lv = level.toUpperCase().replace("WARNING", "WARN");
        lines = lines.filter(l => l.level.toUpperCase() === lv);
      }

      // Return most-recent first, capped at limit
      lines.reverse();
      return NextResponse.json(lines.slice(0, limit));
    }

    // Default: fastapi-stream structured log endpoint
    const res = await fetch(
      `${STREAM_URL}/stream/logs/recent?max_entries=${limit}`,
      { cache: "no-store", signal: AbortSignal.timeout(5000) }
    );
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });

    let lines = (data.logs ?? data ?? []) as Array<Record<string, unknown>>;

    // Apply level filter (upstream doesn't support it)
    if (level) {
      const lv = level.toUpperCase().replace("WARNING", "WARN");
      lines = lines.filter(l => String(l.level ?? "").toUpperCase().replace("WARNING", "WARN") === lv);
    }

    // Normalise field names for LogLine interface
    return NextResponse.json(lines.map(l => ({
      timestamp: l.timestamp,
      level:   String(l.level ?? "INFO").replace("WARNING", "WARN"),
      service: "stream",
      message: l.message,
    })));
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 502 });
  }
}
