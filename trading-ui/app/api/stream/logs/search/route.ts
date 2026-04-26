import { NextResponse } from "next/server";
import { readdir, readFile, stat } from "fs/promises";
import path from "path";

export const dynamic = "force-dynamic";

const STREAM_URL = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8000";
const MONITOR_URL = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";
const LOG_ROOT = process.env.TRADING_UI_LOG_ROOT ?? "/app/system-logs";

const SOURCE_CONFIG: Record<string, { label: string; dir: string; patterns: RegExp[] }> = {
  forex_scanner: {
    label: "Forex Scanner",
    dir: "worker",
    patterns: [/^forex_scanner(?:\..*)?\.log$/, /^trading-signals\.log$/],
  },
  stream_service: {
    label: "Stream Service",
    dir: "stream",
    patterns: [/^fastapi-stream(?:\..*)?\.log$/],
  },
  trade_monitor: {
    label: "Trade Monitor",
    dir: "dev",
    patterns: [/^trade_monitor(?:\..*)?\.log$/],
  },
  fastapi_dev: {
    label: "FastAPI Dev",
    dir: "dev",
    patterns: [/^fastapi-dev(?:\..*)?\.log$/],
  },
  dev_trade: {
    label: "Dev Trade",
    dir: "dev",
    patterns: [/^dev-trade(?:\..*)?\.log$/],
  },
  trade_sync: {
    label: "Trade Sync",
    dir: "dev",
    patterns: [/^trade_sync(?:\..*)?\.log$/],
  },
};

const CONTAINER_TO_SERVICE: Record<string, string> = {
  "task-worker": "forex_scanner",
  "task-worker-live": "forex_scanner",
  "fastapi-dev": "fastapi_dev",
  "fastapi-live": "fastapi_dev",
  "fastapi-stream": "stream_service",
  "stock-scanner": "stock_scanner",
};

type LogMatch = {
  timestamp: string | null;
  level: string;
  source: string;
  sourceLabel: string;
  file: string;
  path: string;
  lineNumber: number | null;
  logType: string;
  message: string;
  container?: string;
};

type SourceDiagnostic = {
  key: string;
  label: string;
  filesFound: number;
  filesSearched: number;
  totalBytes: number;
  files: Array<{ path: string; size: number; modified: string }>;
};

function parseCsv(value: string | null): string[] {
  return (value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseBool(value: string | null, fallback = false): boolean {
  if (value == null) return fallback;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

function safeLimit(value: string | null, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function normalizeLevel(level: string): string {
  return level.toUpperCase().replace("WARNING", "WARN");
}

function parseTimestamp(raw: string): string | null {
  const docker = raw.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)/)?.[1];
  if (docker) return docker;

  const standard = raw.match(/(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:\s+(CEST|CET|UTC))?/);
  if (!standard) return null;

  const [, date, time, zone] = standard;
  const suffix = zone === "CEST" ? "+02:00" : zone === "CET" ? "+01:00" : zone === "UTC" ? "Z" : "";
  const parsed = new Date(`${date}T${time}${suffix}`);
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString();
}

function inferLevel(line: string): string {
  const match = line.match(/\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b/i);
  return normalizeLevel(match?.[1] ?? "INFO");
}

function inferLogType(line: string, level: string): string {
  if (/ERROR|CRITICAL/i.test(level)) return "error";
  if (/WARN/i.test(level)) return "warning";
  if (/\[PROFIT\]/i.test(line)) return "trade_monitoring";
  if (/Trade logged|Place-Order|placing order/i.test(line)) return "trade_opened";
  if (/ADJUST-STOP|Stop level|Limit level/i.test(line)) return "trade_adjustment";
  if (/REJECTED|Filtered out/i.test(line)) return "rejected";
  if (/signal|CS\.D\.[A-Z0-9._-]+\.IP/i.test(line)) return "signal";
  if (/trade|position|deal/i.test(line)) return "trade";
  return "info";
}

function inDateRange(timestamp: string | null, start: string | null, end: string | null): boolean {
  if (!start && !end) return true;
  if (!timestamp) return false;
  const day = timestamp.slice(0, 10);
  if (start && day < start) return false;
  if (end && day > end) return false;
  return true;
}

function buildMatcher(query: string, regexMode: boolean, caseSensitive: boolean): (line: string) => boolean {
  if (!query) return () => true;
  const regexPrefix = query.startsWith("re:");
  if (regexMode || regexPrefix) {
    const pattern = regexPrefix ? query.slice(3) : query;
    const regex = new RegExp(pattern, caseSensitive ? "" : "i");
    return (line) => regex.test(line);
  }
  const needle = caseSensitive ? query : query.toLowerCase();
  return (line) => (caseSensitive ? line : line.toLowerCase()).includes(needle);
}

async function discoverSourceFiles(sourceKeys: string[]): Promise<{
  files: Array<{ source: string; filePath: string; size: number; modified: string }>;
  diagnostics: SourceDiagnostic[];
}> {
  const files: Array<{ source: string; filePath: string; size: number; modified: string }> = [];
  const diagnostics: SourceDiagnostic[] = [];

  for (const source of sourceKeys) {
    const config = SOURCE_CONFIG[source];
    if (!config) continue;

    const dir = path.join(LOG_ROOT, config.dir);
    const diagnostic: SourceDiagnostic = {
      key: source,
      label: config.label,
      filesFound: 0,
      filesSearched: 0,
      totalBytes: 0,
      files: [],
    };

    try {
      const names = await readdir(dir);
      for (const name of names) {
        if (!config.patterns.some((pattern) => pattern.test(name))) continue;
        const filePath = path.join(dir, name);
        const info = await stat(filePath);
        if (!info.isFile()) continue;
        const modified = info.mtime.toISOString();
        files.push({ source, filePath, size: info.size, modified });
        diagnostic.filesFound += 1;
        diagnostic.totalBytes += info.size;
        diagnostic.files.push({ path: filePath.replace(`${LOG_ROOT}/`, ""), size: info.size, modified });
      }
    } catch {
      // Missing source directories are surfaced as zero-count diagnostics.
    }

    diagnostic.files.sort((a, b) => b.modified.localeCompare(a.modified));
    diagnostics.push(diagnostic);
  }

  files.sort((a, b) => b.modified.localeCompare(a.modified));
  return { files, diagnostics };
}

function parseDockerLine(raw: string, container: string): LogMatch {
  const withoutDockerTs = raw.replace(/^\S+Z\s*/, "");
  const level = inferLevel(withoutDockerTs);
  const timestamp = parseTimestamp(raw) ?? parseTimestamp(withoutDockerTs);
  const source = CONTAINER_TO_SERVICE[container] ?? container;
  return {
    timestamp,
    level,
    source,
    sourceLabel: SOURCE_CONFIG[source]?.label ?? container,
    file: container,
    path: `container:${container}`,
    lineNumber: null,
    logType: inferLogType(withoutDockerTs, level),
    message: withoutDockerTs.trim() || raw.trim(),
    container,
  };
}

async function searchContainerLogs(
  containers: string[],
  matcher: (line: string) => boolean,
  startDate: string | null,
  endDate: string | null,
  levelFilter: string,
  tail: number,
): Promise<LogMatch[]> {
  const batches = await Promise.all(containers.map(async (container) => {
    try {
      const res = await fetch(
        `${MONITOR_URL}/api/v1/containers/${encodeURIComponent(container)}/logs?tail=${tail}`,
        { cache: "no-store", signal: AbortSignal.timeout(8000) },
      );
      if (!res.ok) return [];
      const data = await res.json();
      const rawLines = String(data.logs ?? "").split("\n").filter(Boolean);
      return rawLines
        .map((line) => parseDockerLine(line, container))
        .filter((line) => matcher(line.message))
        .filter((line) => !levelFilter || normalizeLevel(line.level) === levelFilter)
        .filter((line) => inDateRange(line.timestamp, startDate, endDate));
    } catch {
      return [];
    }
  }));
  return batches.flat();
}

async function proxyLegacySearch(searchParams: URLSearchParams) {
  const qs = searchParams.toString();
  const res = await fetch(`${STREAM_URL}/logs/search?${qs}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(15000),
  });
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);

  if (parseBool(searchParams.get("proxy"), false)) {
    try {
      return await proxyLegacySearch(searchParams);
    } catch (e) {
      return NextResponse.json({ error: String(e) }, { status: 502 });
    }
  }

  const query = searchParams.get("q") ?? searchParams.get("query") ?? "";
  const sourceKeys = parseCsv(searchParams.get("sources"));
  const selectedSources = sourceKeys.length ? sourceKeys : ["forex_scanner", "fastapi_dev"];
  const containers = parseCsv(searchParams.get("containers"));
  const includeContainers = parseBool(searchParams.get("include_containers"), containers.length > 0);
  const regexMode = parseBool(searchParams.get("regex"), query.startsWith("re:"));
  const caseSensitive = parseBool(searchParams.get("case_sensitive"), false);
  const startDate = searchParams.get("start_date");
  const endDate = searchParams.get("end_date");
  const limit = safeLimit(searchParams.get("limit"), 100, 1, 1000);
  const containerTail = safeLimit(searchParams.get("container_tail"), 1000, 50, 5000);
  const levelFilter = searchParams.get("level") ? normalizeLevel(searchParams.get("level") as string) : "";
  const logTypeFilter = searchParams.get("log_type") ?? "";

  try {
    const matcher = buildMatcher(query, regexMode, caseSensitive);
    const { files, diagnostics } = await discoverSourceFiles(selectedSources);
    const results: LogMatch[] = [];
    const stats = {
      files_searched: files.length,
      files_found: files.length,
      files_missing: diagnostics.filter((item) => item.filesFound === 0).length,
      lines_scanned: 0,
      matches_found: 0,
      container_lines_searched: 0,
    };

    for (const file of files) {
      const diagnostic = diagnostics.find((item) => item.key === file.source);
      if (diagnostic) diagnostic.filesSearched += 1;

      const content = await readFile(file.filePath, "utf8");
      const lines = content.split(/\r?\n/);
      for (let index = 0; index < lines.length; index += 1) {
        const line = lines[index];
        if (!line) continue;
        stats.lines_scanned += 1;

        const timestamp = parseTimestamp(line);
        if (!inDateRange(timestamp, startDate, endDate)) continue;
        if (!matcher(line)) continue;

        const level = inferLevel(line);
        if (levelFilter && normalizeLevel(level) !== levelFilter) continue;
        const logType = inferLogType(line, level);
        if (logTypeFilter && logType !== logTypeFilter) continue;

        stats.matches_found += 1;
        results.push({
          timestamp,
          level,
          source: file.source,
          sourceLabel: SOURCE_CONFIG[file.source]?.label ?? file.source,
          file: path.basename(file.filePath),
          path: file.filePath.replace(`${LOG_ROOT}/`, ""),
          lineNumber: index + 1,
          logType,
          message: line.trim(),
        });
      }
    }

    if (includeContainers && containers.length) {
      const containerMatches = await searchContainerLogs(containers, matcher, startDate, endDate, levelFilter, containerTail);
      const filteredContainerMatches = logTypeFilter
        ? containerMatches.filter((line) => line.logType === logTypeFilter)
        : containerMatches;
      stats.container_lines_searched = containers.length * containerTail;
      stats.matches_found += filteredContainerMatches.length;
      results.push(...filteredContainerMatches);
    }

    results.sort((a, b) => {
      const at = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const bt = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return bt - at;
    });

    return NextResponse.json({
      query,
      sources: selectedSources,
      containers: includeContainers ? containers : [],
      stats,
      diagnostics,
      lines: results.slice(0, limit),
      truncated: results.length > limit,
      total_matches: results.length,
    });
  } catch (e) {
    const status = e instanceof SyntaxError ? 400 : 500;
    return NextResponse.json({ error: String(e) }, { status });
  }
}
