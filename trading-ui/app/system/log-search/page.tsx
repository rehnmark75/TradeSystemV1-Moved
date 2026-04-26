"use client";

import Link from "next/link";
import { FormEvent, useMemo, useState } from "react";
import styles from "./page.module.css";

const BASE = "/trading";

const SOURCES = [
  { key: "forex_scanner", label: "Forex Scanner", hint: "worker/forex_scanner*.log" },
  { key: "fastapi_dev", label: "FastAPI Dev", hint: "dev/fastapi-dev*.log" },
  { key: "trade_monitor", label: "Trade Monitor", hint: "dev/trade_monitor*.log" },
  { key: "dev_trade", label: "Dev Trade", hint: "dev/dev-trade*.log" },
  { key: "stream_service", label: "Stream Service", hint: "stream/fastapi-stream*.log" },
  { key: "trade_sync", label: "Trade Sync", hint: "dev/trade_sync*.log" },
];

const CONTAINERS = [
  "task-worker",
  "task-worker-live",
  "fastapi-dev",
  "fastapi-live",
  "fastapi-stream",
  "stock-scanner",
];

const QUICK_SEARCHES = [
  { label: "Signals", query: "re:CS\\.D\\.[A-Z]{6}\\.(MINI|CEEM)\\.IP", regex: true, logType: "signal" },
  { label: "Errors", query: "ERROR", regex: false, level: "ERROR" },
  { label: "Warnings", query: "WARNING", regex: false, level: "WARN" },
  { label: "Rejected", query: "REJECTED", regex: false, logType: "rejected" },
  { label: "High Confidence", query: "re:\\(9[0-9]\\.[0-9]%\\)", regex: true },
  { label: "Trade Opened", query: "Trade logged", regex: false, logType: "trade_opened" },
  { label: "Profit Monitor", query: "re:\\[PROFIT\\] Trade", regex: true, logType: "trade_monitoring" },
  { label: "Stop Adjust", query: "ADJUST-STOP", regex: false, logType: "trade_adjustment" },
];

type LogLine = {
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

type Diagnostic = {
  key: string;
  label: string;
  filesFound: number;
  filesSearched: number;
  totalBytes: number;
  files: Array<{ path: string; size: number; modified: string }>;
};

type SearchPayload = {
  stats?: {
    files_searched: number;
    files_found: number;
    files_missing: number;
    lines_scanned: number;
    matches_found: number;
    container_lines_searched: number;
  };
  diagnostics?: Diagnostic[];
  lines?: LogLine[];
  total_matches?: number;
  truncated?: boolean;
  error?: string;
};

function today(offsetDays = 0) {
  const date = new Date();
  date.setDate(date.getDate() + offsetDays);
  return date.toISOString().slice(0, 10);
}

function formatDateTime(value: string | null) {
  if (!value) return "No timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}

function levelColor(level: string) {
  if (/CRIT|ERROR/i.test(level)) return "#ff7b7b";
  if (/WARN/i.test(level)) return "#f4c96a";
  if (/DEBUG/i.test(level)) return "#9aa8ba";
  return "#75c7ff";
}

function typeColor(type: string) {
  if (type === "error") return "#ff7b7b";
  if (type === "warning") return "#f4c96a";
  if (type.includes("trade")) return "#92e6a7";
  if (type === "signal") return "#75c7ff";
  if (type === "rejected") return "#ff9c75";
  return "#b8c7d9";
}

function toggleValue(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function highlightedParts(text: string, query: string, regex: boolean, caseSensitive: boolean) {
  if (!query) return [{ text, hit: false }];
  try {
    const pattern = query.startsWith("re:") ? query.slice(3) : query;
    const matcher = regex || query.startsWith("re:")
      ? new RegExp(`(${pattern})`, caseSensitive ? "g" : "gi")
      : new RegExp(`(${pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, caseSensitive ? "g" : "gi");
    const exactMatcher = regex || query.startsWith("re:")
      ? new RegExp(`^(?:${pattern})$`, caseSensitive ? "" : "i")
      : null;
    return text
      .split(matcher)
      .filter((part) => part !== "")
      .map((part) => ({
        text: part,
        hit: exactMatcher ? exactMatcher.test(part) : caseSensitive ? part === pattern : part.toLowerCase() === pattern.toLowerCase(),
      }));
  } catch {
    return [{ text, hit: false }];
  }
}

export default function LogSearchPage() {
  const [query, setQuery] = useState("");
  const [selectedSources, setSelectedSources] = useState(["forex_scanner", "fastapi_dev"]);
  const [selectedContainers, setSelectedContainers] = useState<string[]>([]);
  const [includeContainers, setIncludeContainers] = useState(false);
  const [regex, setRegex] = useState(false);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [startDate, setStartDate] = useState(today(-1));
  const [endDate, setEndDate] = useState(today());
  const [limit, setLimit] = useState(100);
  const [level, setLevel] = useState("");
  const [logType, setLogType] = useState("");
  const [payload, setPayload] = useState<SearchPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const lines = payload?.lines ?? [];
  const stats = payload?.stats;
  const sourceSummary = useMemo(() => {
    const found = payload?.diagnostics?.reduce((sum, item) => sum + item.filesFound, 0) ?? 0;
    const bytes = payload?.diagnostics?.reduce((sum, item) => sum + item.totalBytes, 0) ?? 0;
    return { found, bytes };
  }, [payload]);

  const runSearch = async (event?: FormEvent) => {
    event?.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      params.set("q", query);
      params.set("sources", selectedSources.join(","));
      params.set("limit", String(limit));
      params.set("start_date", startDate);
      params.set("end_date", endDate);
      if (regex) params.set("regex", "true");
      if (caseSensitive) params.set("case_sensitive", "true");
      if (level) params.set("level", level);
      if (logType) params.set("log_type", logType);
      if (includeContainers) params.set("include_containers", "true");
      if (selectedContainers.length) params.set("containers", selectedContainers.join(","));

      const res = await fetch(`${BASE}/api/stream/logs/search?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `${res.status} ${res.statusText}`);
      setPayload(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const applyQuickSearch = (item: typeof QUICK_SEARCHES[number]) => {
    setQuery(item.query);
    setRegex(item.regex);
    setLevel(item.level ?? "");
    setLogType(item.logType ?? "");
  };

  return (
    <div className={`page ${styles.logSearchPage}`}>
      <div className={styles.logSearchHero}>
        <div>
          <div className="ops-kicker">Operations / Log Search</div>
          <h2>Search the scanner trail without leaving trading-ui.</h2>
          <p>
            File-backed scanner, dev, trade, and stream logs sit beside optional recent container output so incidents can be traced from one screen.
          </p>
        </div>
        <div className={styles.logSearchStatus}>
          <span>Mounted files</span>
          <strong>{payload ? sourceSummary.found : "-"}</strong>
          <em>{payload ? formatBytes(sourceSummary.bytes) : "Run a search"}</em>
        </div>
      </div>

      <form className={styles.logSearchConsole} onSubmit={runSearch}>
        <section className={styles.logSearchQuery}>
          <label>
            Search term
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="ERROR, REJECTED, EURUSD, or re:signal.*BULL"
            />
          </label>
          <button className={styles.logSearchPrimary} type="submit" disabled={loading}>
            {loading ? "Searching" : "Search"}
          </button>
          <button
            type="button"
            className={styles.logSearchSecondary}
            onClick={() => {
              setQuery("");
              setPayload(null);
              setError(null);
            }}
          >
            Clear
          </button>
        </section>

        <section className={styles.logSearchFilters}>
          <label>
            Start date
            <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
          </label>
          <label>
            End date
            <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
          </label>
          <label>
            Level
            <select value={level} onChange={(event) => setLevel(event.target.value)}>
              <option value="">All levels</option>
              <option value="DEBUG">Debug</option>
              <option value="INFO">Info</option>
              <option value="WARN">Warn</option>
              <option value="ERROR">Error</option>
              <option value="CRITICAL">Critical</option>
            </select>
          </label>
          <label>
            Type
            <select value={logType} onChange={(event) => setLogType(event.target.value)}>
              <option value="">All types</option>
              <option value="signal">Signal</option>
              <option value="rejected">Rejected</option>
              <option value="trade">Trade</option>
              <option value="trade_opened">Trade opened</option>
              <option value="trade_monitoring">Trade monitoring</option>
              <option value="trade_adjustment">Trade adjustment</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
              <option value="info">Info</option>
            </select>
          </label>
          <label>
            Max results
            <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
              {[50, 100, 200, 300, 500, 1000].map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>
        </section>

        <section className={styles.logSearchSwitches} aria-label="Search modes">
          <label>
            <input type="checkbox" checked={regex} onChange={(event) => setRegex(event.target.checked)} />
            Regex mode
          </label>
          <label>
            <input type="checkbox" checked={caseSensitive} onChange={(event) => setCaseSensitive(event.target.checked)} />
            Case sensitive
          </label>
          <label>
            <input type="checkbox" checked={includeContainers} onChange={(event) => setIncludeContainers(event.target.checked)} />
            Include container logs
          </label>
        </section>

        <section className={styles.logSearchGroups}>
          <div>
            <div className={styles.logSearchGroupTitle}>Log sources</div>
            <div className={styles.logSearchChipGrid}>
              {SOURCES.map((source) => (
                <button
                  type="button"
                  key={source.key}
                  className={`${styles.logSourceChip}${selectedSources.includes(source.key) ? ` ${styles.logSourceChipActive}` : ""}`}
                  onClick={() => setSelectedSources(toggleValue(selectedSources, source.key))}
                  title={source.hint}
                >
                  <span>{source.label}</span>
                  <em>{source.hint}</em>
                </button>
              ))}
            </div>
          </div>
          <div>
            <div className={styles.logSearchGroupTitle}>Containers</div>
            <div className={styles.logContainerGrid}>
              {CONTAINERS.map((container) => (
                <button
                  type="button"
                  key={container}
                  className={`${styles.logContainerChip}${selectedContainers.includes(container) ? ` ${styles.logContainerChipActive}` : ""}`}
                  onClick={() => {
                    setSelectedContainers(toggleValue(selectedContainers, container));
                    setIncludeContainers(true);
                  }}
                >
                  {container}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className={styles.logQuickRow} aria-label="Quick searches">
          {QUICK_SEARCHES.map((item) => (
            <button type="button" key={item.label} onClick={() => applyQuickSearch(item)}>
              {item.label}
            </button>
          ))}
        </section>
      </form>

      {error && <div className="ops-banner">Search failed: {error}</div>}

      {stats && (
        <div className={styles.logSearchMetrics}>
          <div><span>Files searched</span><strong>{stats.files_searched}</strong></div>
          <div><span>Files found</span><strong>{stats.files_found}</strong></div>
          <div><span>Missing groups</span><strong>{stats.files_missing}</strong></div>
          <div><span>Lines scanned</span><strong>{stats.lines_scanned.toLocaleString()}</strong></div>
          <div><span>Matches</span><strong>{payload?.total_matches ?? stats.matches_found}</strong></div>
        </div>
      )}

      <div className={styles.logResultsLayout}>
        <section className={styles.logResultsPanel}>
          <div className={styles.logResultsHead}>
            <div>
              <div className="ops-kicker">Results</div>
              <h3>{payload ? `${lines.length} shown${payload.truncated ? " - capped by limit" : ""}` : "Ready to search"}</h3>
            </div>
            <Link href="/system" className={styles.logSearchSecondaryLink}>Back to System Pulse</Link>
          </div>

          <div className={styles.logResultsList}>
            {loading && <div className={styles.logEmpty}>Searching mounted files and selected containers...</div>}
            {!loading && payload && lines.length === 0 && <div className={styles.logEmpty}>No matching lines for the selected filters.</div>}
            {!loading && !payload && <div className={styles.logEmpty}>Choose sources, enter a term, or run a quick search.</div>}
            {lines.map((line, index) => {
              const parts = highlightedParts(line.message, query, regex, caseSensitive);
              return (
                <article key={`${line.path}-${line.lineNumber ?? index}-${index}`} className={styles.logResultRow}>
                  <div className={styles.logResultMeta}>
                    <span>{formatDateTime(line.timestamp)}</span>
                    <strong style={{ color: levelColor(line.level) }}>{line.level}</strong>
                    <em style={{ color: typeColor(line.logType) }}>{line.logType}</em>
                    <code>{line.container ?? line.file}</code>
                    {line.lineNumber && <span>line {line.lineNumber}</span>}
                  </div>
                  <pre>
                    {parts.map((part, partIndex) => {
                      return part.hit ? <mark key={partIndex}>{part.text}</mark> : <span key={partIndex}>{part.text}</span>;
                    })}
                  </pre>
                </article>
              );
            })}
          </div>
        </section>

        <aside className={styles.logDiagnosticsPanel}>
          <div className="ops-kicker">Diagnostics</div>
          <h3>Source coverage</h3>
          {(payload?.diagnostics ?? []).map((source) => (
            <div key={source.key} className={styles.logDiagnosticSource}>
              <div>
                <strong>{source.label}</strong>
                <span>{source.filesFound} files, {formatBytes(source.totalBytes)}</span>
              </div>
              {source.files.slice(0, 4).map((file) => (
                <code key={file.path}>{file.path} ({formatBytes(file.size)})</code>
              ))}
              {source.filesFound === 0 && <em>No mounted files found for this source.</em>}
            </div>
          ))}
          {!payload && <p>Diagnostics appear after the first search.</p>}
        </aside>
      </div>
    </div>
  );
}
