"use client";

import { useEffect, useMemo, useState } from "react";
import { useEnvironment } from "../lib/environment";

type MarketRegimeRow = {
  calculation_date?: string | null;
  market_regime?: string | null;
  spy_trend?: string | null;
  volatility_regime?: string | null;
  pct_above_sma200?: number | null;
  pct_above_sma50?: number | null;
  ad_ratio?: number | null;
  advancing_count?: number | null;
  declining_count?: number | null;
  avg_atr_pct?: number | null;
};

type MarketRegimePayload = {
  row?: MarketRegimeRow | null;
};

type InfraStatus = {
  health_score?: number;
  containers_running?: number;
  containers_stopped?: number;
  containers_unhealthy?: number;
  active_alerts?: number;
};

type ContainerInfo = {
  name?: string;
  state?: string;
  status?: string;
};

type ActiveAlert = {
  id?: string | number;
  severity?: string;
  source?: string;
  message?: string;
  created_at?: string;
  status?: string;
};

type StreamSummary = {
  status?: string;
  stream_running?: boolean;
  last_event_at?: string;
  last_message_at?: string;
  latest_timestamp?: string;
  lag_seconds?: number;
};

type OperationRow = {
  id?: string | number;
  operation?: string;
  action?: string;
  status?: string;
  source?: string;
  created_at?: string;
  timestamp?: string;
  message?: string;
};

type ForexAlertRow = {
  id: number;
  alert_timestamp: string;
  pair: string | null;
  epic: string | null;
  strategy: string | null;
  signal_type: string | null;
  claude_score: number | null;
  claude_decision: string | null;
  claude_approved: boolean | null;
  alert_level: string | null;
};

type ForexAlertPayload = {
  stats?: {
    total_alerts?: number;
    approved?: number;
    rejected?: number;
    approval_rate?: number;
  };
  alerts?: ForexAlertRow[];
};

type ForexOverviewPayload = {
  stats?: {
    total_trades?: number;
    winning_trades?: number;
    losing_trades?: number;
    pending_trades?: number;
    total_profit_loss?: number;
    win_rate?: number;
    best_pair?: string;
    worst_pair?: string;
    active_pairs?: string[];
  };
};

type ForexMarketIntelligencePayload = {
  summary?: {
    total?: number;
    avg_confidence?: number;
  };
  regimes?: Record<string, number>;
  sessions?: Record<string, number>;
  volatility?: Record<string, number>;
};

type BrokerOverview = {
  balance?: {
    total_value?: number | null;
    invested?: number | null;
    available?: number | null;
    recorded_at?: string | null;
  } | null;
  trend?: {
    change?: number | null;
    change_pct?: number | null;
    trend?: string | null;
  } | null;
  last_sync?: string | null;
  stats?: {
    total_trades?: number | null;
    win_rate?: number | null;
    net_profit?: number | null;
  } | null;
  open_positions?: Array<unknown>;
};

const numberFormatter = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 1,
});

const compactNumberFormatter = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const ACTIVE_ALERT_WINDOW_HOURS = 24;

function formatDateTime(value?: string | null) {
  if (!value) return "No timestamp";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatRelative(value?: string | null) {
  if (!value) return "No recent update";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  const diffMs = Date.now() - parsed.getTime();
  const diffMin = Math.round(diffMs / 60000);
  if (diffMin < 1) return "Updated just now";
  if (diffMin < 60) return `Updated ${diffMin}m ago`;
  const diffHours = Math.round(diffMin / 60);
  if (diffHours < 24) return `Updated ${diffHours}h ago`;
  return `Updated ${Math.round(diffHours / 24)}d ago`;
}

function formatPercent(value?: number | null) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${numberFormatter.format(value)}%`;
}

function formatCurrency(value?: number | null) {
  if (value == null || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function getSessionLabel() {
  const now = new Date();
  const utcHour = now.getUTCHours() + now.getUTCMinutes() / 60;
  if (utcHour >= 7 && utcHour < 12) return "London Session";
  if (utcHour >= 12 && utcHour < 16.5) return "London / New York Overlap";
  if (utcHour >= 16.5 && utcHour < 21) return "New York Session";
  if (utcHour >= 21 || utcHour < 1) return "Post-Close Monitoring";
  return "Asia / Prep Session";
}

function toSeverityTone(severity?: string | null) {
  const value = (severity ?? "").toLowerCase();
  if (value.includes("critical")) return "critical";
  if (value.includes("high") || value.includes("error")) return "warn";
  return "info";
}

function resolveStreamState(summary?: StreamSummary | null) {
  const status = String(summary?.status ?? "").toLowerCase();
  if (status) {
    if (/running|healthy|ok|active/.test(status)) return "running";
    if (/degraded|warn|market_closed|closed|idle|issues/.test(status)) return "degraded";
    if (/down|error|err|stopped|unavailable/.test(status)) return "down";
  }
  if (summary?.stream_running === true) return "running";
  if ((summary?.lag_seconds ?? 0) > 0) return "degraded";
  return "unknown";
}

function isRecentAlert(value?: string) {
  if (!value) return false;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return false;
  const ageMs = Date.now() - parsed.getTime();
  return ageMs >= 0 && ageMs <= ACTIVE_ALERT_WINDOW_HOURS * 60 * 60 * 1000;
}

function resolvePair(row: ForexAlertRow) {
  if (row.pair) return row.pair;
  if (!row.epic) return "N/A";
  const parts = row.epic.split(".");
  if (parts.length >= 3) return parts[2].slice(0, 6);
  return row.epic;
}

function resolveContainerTone(container?: ContainerInfo) {
  const state = String(container?.state ?? "").toLowerCase();
  const status = String(container?.status ?? "").toLowerCase();
  const normalizedStatus = status === "none" ? "" : status;
  if (normalizedStatus === "healthy" || (state === "running" && !normalizedStatus)) return "good";
  if (normalizedStatus === "degraded" || normalizedStatus === "starting") return "warn";
  if (normalizedStatus === "unhealthy" || state === "exited" || state === "dead") return "bad";
  return "neutral";
}

function resolveContainerLabel(container?: ContainerInfo) {
  const status = String(container?.status ?? "").toLowerCase();
  const state = String(container?.state ?? "").toLowerCase();
  const normalizedStatus = status === "none" ? "" : status;
  return normalizedStatus || state || "unknown";
}

export default function Page() {
  const { environment } = useEnvironment();
  const [market, setMarket] = useState<MarketRegimePayload | null>(null);
  const [infra, setInfra] = useState<InfraStatus | null>(null);
  const [alerts, setAlerts] = useState<ActiveAlert[]>([]);
  const [containers, setContainers] = useState<ContainerInfo[]>([]);
  const [stream, setStream] = useState<StreamSummary | null>(null);
  const [operations, setOperations] = useState<OperationRow[]>([]);
  const [forexAlerts, setForexAlerts] = useState<ForexAlertPayload | null>(null);
  const [forexOverview, setForexOverview] = useState<ForexOverviewPayload | null>(null);
  const [forexIntelligence, setForexIntelligence] = useState<ForexMarketIntelligencePayload | null>(null);
  const [broker, setBroker] = useState<BrokerOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastLoadedAt, setLastLoadedAt] = useState<Date | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function loadOverview() {
      setLoading(true);
      try {
        const responses = await Promise.all([
          fetch("/trading/api/market/regime", { cache: "no-store" }).then((r) => r.json()),
          fetch("/trading/api/infra/status", { cache: "no-store" }).then((r) => r.json()),
          fetch("/trading/api/infra/alerts/active", { cache: "no-store" }).then((r) => r.json()),
          fetch("/trading/api/infra/containers", { cache: "no-store" }).then((r) => r.json()),
          fetch("/trading/api/stream/summary", { cache: "no-store" }).then((r) => r.json()),
          fetch("/trading/api/stream/operations/recent?limit=5", { cache: "no-store" }).then((r) => r.json()),
          fetch(`/trading/api/forex/alert-history/?days=1&page=1&limit=5&status=All&strategy=All&pair=All&env=${environment}`, {
            cache: "no-store",
          }).then((r) => r.json()),
          fetch(`/trading/api/forex/overview/?days=7&env=${environment}`, { cache: "no-store" }).then((r) => r.json()),
          fetch(`/trading/api/forex/market-intelligence/?env=${environment}`, { cache: "no-store" }).then((r) => r.json()),
          fetch("/trading/api/broker/overview?days=7", { cache: "no-store" }).then((r) => r.json()),
        ]);

        if (cancelled) return;
        setMarket(responses[0]);
        setInfra(responses[1]);
        setAlerts(Array.isArray(responses[2]) ? responses[2] : []);
        setContainers(Array.isArray(responses[3]) ? responses[3] : []);
        setStream(responses[4]);
        setOperations(Array.isArray(responses[5]) ? responses[5] : []);
        setForexAlerts(responses[6]);
        setForexOverview(responses[7]);
        setForexIntelligence(responses[8]);
        setBroker(responses[9]);
        setLastLoadedAt(new Date());
      } catch {
        if (cancelled) return;
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadOverview();
    const interval = window.setInterval(loadOverview, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [environment, refreshTick]);

  const regime = market?.row;
  const systemState = useMemo(() => {
    const unhealthy = infra?.containers_unhealthy ?? 0;
    const stopped = infra?.containers_stopped ?? 0;
    const activeAlerts = alerts.filter((alert) => isRecentAlert(alert.created_at)).length;
    if (unhealthy > 0 || stopped > 0 || activeAlerts > 0) return "Attention Required";
    return "Nominal";
  }, [infra, alerts]);

  const recentAlerts = useMemo(
    () => alerts.filter((alert) => isRecentAlert(alert.created_at)).slice(0, 5),
    [alerts]
  );

  const dominantFxRegime =
    Object.entries(forexIntelligence?.regimes ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "Unavailable";
  const dominantFxSession =
    Object.entries(forexIntelligence?.sessions ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "Unavailable";
  const dominantFxVolatility =
    Object.entries(forexIntelligence?.volatility ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "Unavailable";
  const streamState = resolveStreamState(stream);
  const streamTimestamp =
    stream?.latest_timestamp ?? stream?.last_event_at ?? stream?.last_message_at ?? null;
  const trackedContainerNames =
    environment === "live"
      ? ["task-worker-live", "fastapi-live"]
      : ["task-worker", "fastapi-dev"];
  const trackedContainers = trackedContainerNames.map((name) => ({
    name,
    info: containers.find((container) => container.name === name),
  }));

  return (
    <div className="page operations-overview">
      <section className="ops-overview-shell">
        <div className="ops-status-strip ops-status-strip-top">
          <div className="ops-status-card">
            <span>Session</span>
            <strong>{getSessionLabel()}</strong>
          </div>
          <div className="ops-status-card">
            <span>Environment</span>
            <strong>{environment.toUpperCase()}</strong>
          </div>
          <div className="ops-status-card">
            <span>System</span>
            <strong>{systemState}</strong>
          </div>
          <button
            type="button"
            className="ops-status-card ops-status-card-button"
            onClick={() => setRefreshTick((value) => value + 1)}
            title="Click to refresh the overview now"
          >
            <span>Refresh</span>
            <strong>{loading ? "Refreshing…" : "Refresh Now"}</strong>
            <em>{lastLoadedAt ? `Last refresh ${formatRelative(lastLoadedAt.toISOString()).replace(/^Updated /, "")}` : "Click to refresh now"}</em>
          </button>
        </div>

        <div className="ops-grid-primary">
          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Market Pulse</div>
                <h2>Context Snapshot</h2>
              </div>
            </div>
            <div className="ops-kpi-grid">
              <div className="ops-kpi-tile">
                <span>Regime</span>
                <strong>{regime?.market_regime?.replaceAll("_", " ") ?? "Unavailable"}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Trend</span>
                <strong>{regime?.spy_trend ?? "—"}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Volatility</span>
                <strong>{regime?.volatility_regime ?? "—"}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Above 200 SMA</span>
                <strong>{formatPercent(regime?.pct_above_sma200)}</strong>
              </div>
            </div>
            <div className="ops-mini-grid">
              <div className="ops-mini-stat">
                <span>Above 50 SMA</span>
                <strong>{formatPercent(regime?.pct_above_sma50)}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>A/D Ratio</span>
                <strong>{regime?.ad_ratio != null ? numberFormatter.format(regime.ad_ratio) : "—"}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Breadth</span>
                <strong>
                  {(regime?.advancing_count ?? 0)}/{(regime?.declining_count ?? 0)}
                </strong>
              </div>
              <div className="ops-mini-stat">
                <span>Avg ATR</span>
                <strong>{formatPercent(regime?.avg_atr_pct)}</strong>
              </div>
            </div>
          </section>

          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">System Health</div>
                <h2>Platform Status</h2>
              </div>
            </div>
            <div className="ops-kpi-grid">
              <div className="ops-kpi-tile">
                <span>Health Score</span>
                <strong>{infra?.health_score != null ? `${Math.round(infra.health_score)}%` : "—"}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Running</span>
                <strong>{infra?.containers_running ?? "—"}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Stopped</span>
                <strong>{infra?.containers_stopped ?? "—"}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Unhealthy</span>
                <strong>{infra?.containers_unhealthy ?? "—"}</strong>
              </div>
            </div>
            <div className="ops-inline-flags">
              <span className={`ops-flag ${streamState === "running" ? "good" : streamState === "down" ? "bad" : "warn"}`}>
                Stream {streamState === "running" ? "Running" : streamState === "down" ? "Down" : streamState === "degraded" ? "Degraded" : "Unknown"}
              </span>
              <span className={`ops-flag ${recentAlerts.length > 0 ? "warn" : "good"}`}>
                {recentAlerts.length > 0 ? `${recentAlerts.length} Fresh Alerts` : "No Fresh Alerts"}
              </span>
              <span className="ops-flag neutral">{formatRelative(streamTimestamp)}</span>
            </div>
            <div className="ops-service-strip">
              {trackedContainers.map(({ name, info }) => (
                <div className="ops-service-chip" key={name}>
                  <span>{name}</span>
                  <strong className={`ops-service-state ${resolveContainerTone(info)}`}>
                    {resolveContainerLabel(info)}
                  </strong>
                </div>
              ))}
            </div>
          </section>
        </div>

        <div className="ops-grid-secondary">
          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">FX Snapshot</div>
                <h2>Forex Context</h2>
              </div>
            </div>
            <div className="ops-kpi-grid ops-kpi-grid-compact">
              <div className="ops-kpi-tile">
                <span>Dominant Regime</span>
                <strong>{dominantFxRegime.replaceAll("_", " ")}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Dominant Session</span>
                <strong>{dominantFxSession.replaceAll("_", " ")}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Volatility</span>
                <strong>{dominantFxVolatility.replaceAll("_", " ")}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Confidence</span>
                <strong>{formatPercent(forexIntelligence?.summary?.avg_confidence)}</strong>
              </div>
            </div>
            <div className="ops-mini-grid">
              <div className="ops-mini-stat">
                <span>7D Trades</span>
                <strong>{forexOverview?.stats?.total_trades ?? "—"}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Win Rate</span>
                <strong>{formatPercent(forexOverview?.stats?.win_rate)}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Best Pair</span>
                <strong>{forexOverview?.stats?.best_pair ?? "—"}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Net PnL</span>
                <strong>{formatCurrency(forexOverview?.stats?.total_profit_loss)}</strong>
              </div>
            </div>
          </section>

          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Alerts</div>
                <h2>Active Issues</h2>
              </div>
            </div>
            <div className="ops-list">
              {recentAlerts.length ? (
                recentAlerts.map((alert) => (
                  <div className="ops-list-row" key={`${alert.id ?? alert.message}`}>
                    <span className={`ops-dot ${toSeverityTone(alert.severity)}`} />
                    <div className="ops-list-copy">
                      <strong>{alert.message ?? "Unnamed alert"}</strong>
                      <span>
                        {(alert.source ?? "system").toString()} · {formatDateTime(alert.created_at)}
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="ops-empty">No fresh system alerts in the last 24 hours.</div>
              )}
            </div>
          </section>
        </div>

        <div className="ops-grid-secondary">
          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Recent Activity</div>
                <h2>System Operations</h2>
              </div>
            </div>
            <div className="ops-list">
              {operations.length ? (
                operations.map((row, index) => (
                  <div className="ops-list-row" key={`${row.id ?? row.timestamp ?? index}`}>
                    <span className={`ops-dot ${row.status === "failed" ? "critical" : "info"}`} />
                    <div className="ops-list-copy">
                      <strong>{row.operation ?? row.action ?? row.message ?? "Recent operation"}</strong>
                      <span>
                        {(row.source ?? row.status ?? "stream").toString()} ·{" "}
                        {formatDateTime(row.created_at ?? row.timestamp)}
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="ops-empty">No recent operations recorded.</div>
              )}
            </div>
          </section>
        </div>

        <div className="ops-grid-secondary">
          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">FX Queue</div>
                <h2>Recent Signal Decisions</h2>
              </div>
            </div>
            <div className="ops-list">
              {(forexAlerts?.alerts ?? []).length ? (
                forexAlerts!.alerts!.map((row) => {
                  const approved = row.claude_approved === true || row.claude_decision === "APPROVE";
                  const rejected =
                    row.claude_approved === false ||
                    row.claude_decision === "REJECT" ||
                    row.alert_level === "REJECTED";
                  return (
                    <div className="ops-list-row" key={row.id}>
                      <span className={`ops-dot ${approved ? "good" : rejected ? "critical" : "warn"}`} />
                      <div className="ops-list-copy">
                        <strong>
                          {resolvePair(row)} · {row.strategy ?? "No strategy"} · {row.signal_type ?? "Signal"}
                        </strong>
                        <span>
                          {approved ? "Approved" : rejected ? "Rejected" : "Pending"} · Score{" "}
                          {row.claude_score ?? "—"} · {formatDateTime(row.alert_timestamp)}
                        </span>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="ops-empty">No recent FX decisions found for the selected environment.</div>
              )}
            </div>
          </section>

          <section className="ops-panel">
            <div className="ops-panel-head">
              <div>
                <div className="ops-panel-kicker">Broker Snapshot</div>
                <h2>Capital and Exposure</h2>
              </div>
            </div>
            <div className="ops-kpi-grid">
              <div className="ops-kpi-tile">
                <span>Equity</span>
                <strong>{formatCurrency(broker?.balance?.total_value)}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Available</span>
                <strong>{formatCurrency(broker?.balance?.available)}</strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Trend</span>
                <strong>
                  {broker?.trend?.change_pct != null ? formatPercent(broker.trend.change_pct) : "—"}
                </strong>
              </div>
              <div className="ops-kpi-tile">
                <span>Open Positions</span>
                <strong>{broker?.open_positions?.length ?? "—"}</strong>
              </div>
            </div>
            <div className="ops-mini-grid">
              <div className="ops-mini-stat">
                <span>Net Profit</span>
                <strong>{formatCurrency(broker?.stats?.net_profit)}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Closed Trades</span>
                <strong>{broker?.stats?.total_trades ?? "—"}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Win Rate</span>
                <strong>{formatPercent(broker?.stats?.win_rate)}</strong>
              </div>
              <div className="ops-mini-stat">
                <span>Last Sync</span>
                <strong>{broker?.last_sync ? formatDateTime(broker.last_sync) : "—"}</strong>
              </div>
            </div>
          </section>
        </div>

        <section className="ops-panel">
          <div className="ops-panel-head">
            <div>
              <div className="ops-panel-kicker">Today’s Queue</div>
              <h2>Suggested Next Checks</h2>
            </div>
          </div>
          <div className="ops-action-grid">
            <div className="ops-action-card">
              <span>Priority 01</span>
              <strong>{(infra?.active_alerts ?? 0) > 0 ? "Review active infrastructure alerts" : "Confirm system remains nominal"}</strong>
            </div>
            <div className="ops-action-card">
              <span>Priority 02</span>
              <strong>Review the latest {compactNumberFormatter.format(forexAlerts?.stats?.total_alerts ?? 0)} FX alerts for {environment}</strong>
            </div>
            <div className="ops-action-card">
              <span>Priority 03</span>
              <strong>Check market regime drift before opening discretionary workflows</strong>
            </div>
            <div className="ops-action-card">
              <span>Priority 04</span>
              <strong>Verify broker equity, exposure, and sync freshness</strong>
            </div>
          </div>
        </section>

        {loading ? <div className="ops-loading">Refreshing overview…</div> : null}
      </section>
    </div>
  );
}
