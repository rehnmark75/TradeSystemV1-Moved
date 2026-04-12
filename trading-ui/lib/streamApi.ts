import { fetchJson } from "./http";

const BASE = process.env.FASTAPI_STREAM_URL ?? "http://fastapi-stream:8003";

export interface StreamSummary {
  totals?: { streams: number; active: number; stalled: number; epics: number };
  [key: string]: unknown;
}

export interface BackfillStatus {
  in_flight?: number;
  queued?: number;
  completed_last_24h?: number;
  failed_last_24h?: number;
  last_run?: string;
  [key: string]: unknown;
}

export interface BackfillGap {
  epic: string;
  timeframe: string;
  gap_start: string;
  gap_end: string;
  missing_bars: number;
}

export interface EpicCandle {
  epic?: string;
  timeframe?: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  timestamp?: string;
  start_time?: string;
  age_seconds?: number;
  stale?: boolean;
}

export interface OpsEvent {
  id?: string;
  kind?: string;
  service?: string;
  message: string;
  at?: string;
  created_at?: string;
  severity?: string;
  [key: string]: unknown;
}

export interface LogLine {
  at?: string;
  timestamp?: string;
  level: string;
  service?: string;
  source?: string;
  message: string;
  meta?: Record<string, unknown>;
}

export interface LogSearchResponse {
  lines: LogLine[];
  total: number;
  truncated: boolean;
  nextCursor?: string;
}

export interface StreamHealth {
  status: string;
  components?: Record<string, unknown>;
  [key: string]: unknown;
}

const get = <T>(path: string) => fetchJson<T>(`${BASE}${path}`);

export const streamApi = {
  backfillStatus: () => get<BackfillStatus>("/backfill/status"),
  backfillGaps: () => get<BackfillGap[]>("/backfill/gaps"),
  streamStatus: () => get<Record<string, unknown>>("/stream/status"),
  candles: (epic: string, limit = 15) =>
    get<EpicCandle[]>(`/stream/candles/${encodeURIComponent(epic)}?timeframe=5&limit=${limit}`),
  latestCandle: (epic: string) =>
    get<EpicCandle>(`/stream/candle/latest/${encodeURIComponent(epic)}?timeframe=5`),
  summary: () => get<StreamSummary>("/stream/system/summary"),
  recentAlerts: (limit = 50) => get<OpsEvent[]>(`/stream/alerts/recent?limit=${limit}`),
  recentOps: (limit = 50) => get<OpsEvent[]>(`/stream/operations/recent?limit=${limit}`),
  recentLogs: (limit = 200, level?: string) =>
    get<LogLine[]>(`/stream/logs/recent?limit=${limit}${level ? `&level=${level}` : ""}`),
  searchLogs: (params: {
    q?: string;
    level?: string;
    service?: string;
    from?: string;
    to?: string;
    limit?: number;
    cursor?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && qs.set(k, String(v)));
    return get<LogSearchResponse>(`/logs/search?${qs}`);
  },
  health: () => get<StreamHealth>("/stream/health"),
};
