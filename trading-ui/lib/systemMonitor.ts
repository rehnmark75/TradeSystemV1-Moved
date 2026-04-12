import { fetchJson } from "./http";

const BASE = process.env.SYSTEM_MONITOR_URL ?? "http://system-monitor:8095";

export type HealthState = "healthy" | "degraded" | "down" | "unknown";

export interface SystemStatus {
  health_score: number;
  containers_running: number;
  containers_stopped: number;
  containers_unhealthy: number;
  active_alerts: number;
  checked_at: string;
}

export interface ContainerHealth {
  name: string;
  image: string;
  state: "running" | "exited" | "restarting" | "paused" | "created";
  status: HealthState;
  startedAt?: string;
  uptimeSeconds?: number;
  restartCount: number;
  cpuPercent?: number;
  memUsageMb?: number;
  memLimitMb?: number;
  ports?: string[];
  is_critical?: boolean;
  warnings?: string[];
  errors?: string[];
}

export interface Alert {
  id: string;
  severity: "info" | "warning" | "error" | "critical";
  source: string;
  message: string;
  created_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  status: "active" | "acknowledged" | "resolved";
  metadata?: Record<string, unknown>;
}

export interface HealthCheckResult {
  service: string;
  probe?: string;
  status: HealthState;
  latency_ms?: number;
  last_run?: string;
  consecutive_failures?: number;
  error?: string;
}

export interface MetricsSnapshot {
  cpu_percent: number;
  mem_percent: number;
  disk_percent: number;
  load_avg?: [number, number, number];
  timestamp: string;
  container_name?: string;
}

export interface MetricsHistory {
  points: MetricsSnapshot[];
  container_name?: string;
}

export interface MonitorConfig {
  thresholds: Record<string, number>;
  critical_containers: string[];
  notification_channels: string[];
}

const get = <T>(path: string) => fetchJson<T>(`${BASE}${path}`);
const post = <T>(path: string, body?: unknown) =>
  fetchJson<T>(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });

export const systemMonitor = {
  status: () => get<SystemStatus>("/api/v1/status"),
  containers: (include_stopped = true) =>
    get<ContainerHealth[]>(`/api/v1/containers?include_stopped=${include_stopped}`),
  container: (name: string) => get<ContainerHealth>(`/api/v1/containers/${name}`),
  containerLogs: (name: string, lines = 100) =>
    get<{ logs: string[]; container: string }>(`/api/v1/containers/${name}/logs?lines=${lines}`),
  restartContainer: (name: string) =>
    post<{ ok: boolean; message: string }>(`/api/v1/containers/${name}/restart`),
  metrics: () => get<MetricsSnapshot[]>("/api/v1/metrics"),
  metricsHistory: (hours = 24, container?: string) =>
    get<MetricsHistory>(
      `/api/v1/metrics/history?hours=${hours}${container ? `&container_name=${container}` : ""}`
    ),
  alerts: (active_only = false, limit = 50) =>
    get<Alert[]>(`/api/v1/alerts?limit=${limit}&active_only=${active_only}`),
  activeAlerts: () => get<Alert[]>("/api/v1/alerts/active"),
  acknowledgeAlert: (id: string) => post<{ ok: boolean }>(`/api/v1/alerts/${id}/acknowledge`),
  resolveAlert: (id: string) => post<{ ok: boolean }>(`/api/v1/alerts/${id}/resolve`),
  healthChecks: () => get<HealthCheckResult[]>("/api/v1/health-checks"),
  testNotification: () => post<{ ok: boolean; message: string }>("/api/v1/test-notification"),
  config: () => get<MonitorConfig>("/api/v1/config"),
};
