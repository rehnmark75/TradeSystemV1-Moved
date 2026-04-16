/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import AlertRow, { type AlertItem } from "../../components/ops/AlertRow";
import ContainerCard, { type ContainerInfo } from "../../components/ops/ContainerCard";
import ContainerLogsDrawer from "../../components/ops/ContainerLogsDrawer";

import HealthCheckGrid, { type HealthCheckItem } from "../../components/ops/HealthCheckGrid";
import KpiTile from "../../components/ops/KpiTile";
import LiveBadge from "../../components/ops/LiveBadge";
import SectionTabs from "../../components/ops/SectionTabs";
import StatusPill from "../../components/ops/StatusPill";

const BASE = "/trading";
const REFRESH_MS = 10_000;

const TABS = [
  { id: "containers", label: "Containers" },
  { id: "alerts", label: "Alerts" },
  { id: "health", label: "Health Checks" },
  { id: "config", label: "Config" },
];

interface SystemStatus {
  health_score?: number;
  containers_running?: number;
  containers_stopped?: number;
  containers_unhealthy?: number;
  active_alerts?: number;
  [key: string]: unknown;
}

interface MonitorConfig {
  thresholds?: Record<string, number>;
  critical_containers?: string[];
  notification_channels?: string[];
  [key: string]: unknown;
}

function useAutoRefresh<T>(url: string, intervalMs = REFRESH_MS) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetch_ = useCallback(async () => {
    if (document.hidden) return;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const d = await res.json();
      setData(d);
      setError(null);
      setLastUpdated(new Date());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    fetch_();
    const id = setInterval(fetch_, intervalMs);
    return () => clearInterval(id);
  }, [fetch_, intervalMs]);

  return { data, loading, error, lastUpdated, refetch: fetch_ };
}

function InfraPageInner() {
  const [activeTab, setActiveTab] = useState("containers");
  const [logsFor, setLogsFor] = useState<string | null>(null);
  const [testingNotif, setTestingNotif] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [activeOnly, setActiveOnly] = useState(false);

  const { data: status, loading: statusLoading, lastUpdated, refetch } =
    useAutoRefresh<SystemStatus>(`${BASE}/api/infra/status`);
  const { data: containers, loading: containersLoading, refetch: refetchContainers } =
    useAutoRefresh<ContainerInfo[]>(`${BASE}/api/infra/containers`);
  const { data: alerts, loading: alertsLoading, refetch: refetchAlerts } =
    useAutoRefresh<AlertItem[]>(`${BASE}/api/infra/alerts?limit=100&active_only=${activeOnly}`);
  const { data: healthChecks, loading: hcLoading } =
    useAutoRefresh<HealthCheckItem[]>(`${BASE}/api/infra/health-checks`);
  const { data: config } =
    useAutoRefresh<MonitorConfig>(`${BASE}/api/infra/config`, 60_000);

  const healthScore = status?.health_score ?? 0;
  const healthColor =
    healthScore >= 90 ? "var(--good)" : healthScore >= 70 ? "var(--warn)" : "var(--bad)";

  const sortedContainers = [...(containers ?? [])].sort((a, b) => {
    const scoreA = (a.is_critical ? 10 : 0) + (a.state !== "running" ? 5 : 0);
    const scoreB = (b.is_critical ? 10 : 0) + (b.state !== "running" ? 5 : 0);
    return scoreB - scoreA;
  });

  const freshAlertCount = (alerts ?? []).filter((alert) => {
    if (!alert.created_at || alert.status === "resolved") return false;
    const timestamp = new Date(alert.created_at).getTime();
    if (Number.isNaN(timestamp)) return false;
    return Date.now() - timestamp <= 15 * 60 * 1000;
  }).length;

  const handleRestart = async (name: string) => {
    await fetch(`${BASE}/api/infra/containers/${encodeURIComponent(name)}/restart`, {
      method: "POST",
    });
    setTimeout(refetchContainers, 3000);
  };

  const handleAck = async (id: string) => {
    await fetch(`${BASE}/api/infra/alerts/${id}/acknowledge`, { method: "POST" });
    refetchAlerts();
  };

  const handleResolve = async (id: string) => {
    await fetch(`${BASE}/api/infra/alerts/${id}/resolve`, { method: "POST" });
    refetchAlerts();
  };

  const testNotification = async () => {
    setTestingNotif(true);
    setTestResult(null);
    try {
      const res = await fetch(`${BASE}/api/infra/test-notification`, { method: "POST" });
      const d = await res.json();
      setTestResult(d.message ?? (res.ok ? "Sent!" : "Failed"));
    } catch (e) {
      setTestResult(String(e));
    } finally {
      setTestingNotif(false);
    }
  };

  return (
    <div className="page">
      {/* Topbar */}
      <div className="topbar">
        <Link href="/" className="brand">K.L.I.R.R</Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/infrastructure" style={{ fontWeight: 700, color: "var(--accent)" }}>Infrastructure</Link>
          <Link href="/system">System Status</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="desk-intro">
        <div>
          <div className="mission-kicker">Platform Operations</div>
          <h2>Infrastructure oversight for the trading stack, with intervention controls and live system context.</h2>
          <p>
            Monitor the full service estate, identify degradation quickly, and act from the same surface used to
            supervise the rest of the trading platform.
          </p>
        </div>
        <div className="desk-intro-meta">
          <div className="desk-intro-stat">
            <span>Health score</span>
            <strong>{Math.round(healthScore)}%</strong>
          </div>
          <div className="desk-intro-stat">
            <span>Status</span>
            <strong>
              <StatusPill
                state={healthScore >= 90 ? "healthy" : healthScore >= 70 ? "degraded" : "down"}
                label={`${Math.round(healthScore)}% health`}
              />
            </strong>
          </div>
        </div>
      </div>

      <div className="desk-toolbar">
        <div className="header-chip">Internal operator console</div>
        <div className="desk-toolbar-actions">
          <LiveBadge lastUpdated={lastUpdated} />
          <button onClick={refetch} className="refresh-btn" title="Refresh">↻</button>
        </div>
      </div>

      <div className="ops-banner">
        ⚠️ Operator console — actions like restart and notification test are internal-network only with no auth layer. Handle with care.
      </div>

      <div className="kpi-grid-5">
        <KpiTile label="Running" value={statusLoading ? "…" : status?.containers_running ?? 0} accent="var(--good)" />
        <KpiTile label="Stopped" value={statusLoading ? "…" : status?.containers_stopped ?? 0} accent={status?.containers_stopped ? "var(--warn)" : "var(--border)"} />
        <KpiTile label="Unhealthy" value={statusLoading ? "…" : status?.containers_unhealthy ?? 0} accent={status?.containers_unhealthy ? "var(--bad)" : "var(--border)"} />
        <KpiTile label="Active Alerts" value={statusLoading ? "…" : status?.active_alerts ?? 0} accent={status?.active_alerts ? "var(--warn)" : "var(--border)"} />
        <div style={{
          background: "var(--panel)", border: "1px solid var(--border)", borderRadius: "12px",
          padding: "14px 16px", display: "flex", flexDirection: "column", gap: "6px",
        }}>
          <span style={{ fontSize: "0.75rem", color: "var(--muted)", fontWeight: 500 }}>Health Score</span>
          <span style={{ fontSize: "1.5rem", fontFamily: "'Space Grotesk',sans-serif", fontWeight: 700, color: healthColor }}>
            {statusLoading ? "…" : `${Math.round(healthScore)}%`}
          </span>
          <div style={{ height: "6px", background: "#e4ddd2", borderRadius: "999px", overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${healthScore}%`, background: healthColor, borderRadius: "999px", transition: "width 0.5s ease" }} />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <SectionTabs tabs={TABS} active={activeTab} onChange={setActiveTab} />

      {/* Containers Tab */}
      {activeTab === "containers" && (
        <div>
          {containersLoading && <p style={{ color: "var(--muted)" }}>Loading containers…</p>}
          {!containersLoading && sortedContainers.length === 0 && (
            <p style={{ color: "var(--muted)" }}>No containers found — is system-monitor reachable?</p>
          )}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(260px,1fr))", gap: "12px" }}>
            {sortedContainers.map(c => (
              <ContainerCard
                key={c.name}
                container={c}
                onViewLogs={setLogsFor}
                onRestart={handleRestart}
              />
            ))}
          </div>
        </div>
      )}

      {/* Alerts Tab */}
      {activeTab === "alerts" && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "14px" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.85rem", cursor: "pointer" }}>
              <input type="checkbox" checked={activeOnly} onChange={e => setActiveOnly(e.target.checked)} />
              Active only
            </label>
            <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>{(alerts ?? []).length} alert{(alerts?.length ?? 0) !== 1 ? "s" : ""}</span>
            <span
              style={{
                fontSize: "0.76rem",
                color: freshAlertCount ? "#8fd5ff" : "var(--muted)",
                border: `1px solid ${freshAlertCount ? "rgba(143, 213, 255, 0.28)" : "var(--border)"}`,
                background: freshAlertCount ? "rgba(79, 171, 255, 0.08)" : "transparent",
                borderRadius: "999px",
                padding: "4px 10px",
                fontWeight: 600,
                letterSpacing: "0.04em",
              }}
            >
              {freshAlertCount} fresh
            </span>
          </div>
          {alertsLoading && <p style={{ color: "var(--muted)" }}>Loading alerts…</p>}
          {!alertsLoading && (alerts ?? []).length === 0 && (
            <p style={{ color: "var(--good)", fontWeight: 500 }}>✓ No alerts — system is quiet.</p>
          )}
          {(alerts ?? []).map(a => (
            <AlertRow key={a.id} alert={a} onAck={handleAck} onResolve={handleResolve} />
          ))}
        </div>
      )}

      {/* Health Checks Tab */}
      {activeTab === "health" && (
        <div className="panel">
          {hcLoading && <p style={{ color: "var(--muted)" }}>Loading health checks…</p>}
          {!hcLoading && <HealthCheckGrid checks={healthChecks ?? []} />}
        </div>
      )}

      {/* Config Tab */}
      {activeTab === "config" && (
        <div>
          <div className="panel" style={{ marginBottom: "16px" }}>
            <h3 style={{ margin: "0 0 12px", fontFamily: "'Space Grotesk',sans-serif" }}>Monitor Configuration</h3>
            {config ? (
              <div style={{ display: "grid", gap: "16px" }}>
                {config.critical_containers && (
                  <div>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "6px", fontWeight: 600 }}>CRITICAL CONTAINERS</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      {config.critical_containers.map(c => (
                        <span
                          key={c}
                          style={{
                            background: "rgba(255, 124, 124, 0.08)",
                            border: "1px solid rgba(255, 124, 124, 0.22)",
                            color: "rgba(255, 205, 205, 0.9)",
                            borderRadius: "6px",
                            padding: "2px 10px",
                            fontSize: "0.82rem",
                            fontFamily: "monospace",
                          }}
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                {config.thresholds && (
                  <div>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "6px", fontWeight: 600 }}>THRESHOLDS</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: "8px" }}>
                      {Object.entries(config.thresholds).map(([k, v]) => (
                        <div
                          key={k}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            fontSize: "0.82rem",
                            background: "rgba(255, 255, 255, 0.03)",
                            border: "1px solid rgba(125, 162, 214, 0.16)",
                            borderRadius: "6px",
                            padding: "6px 10px",
                          }}
                        >
                          <span style={{ color: "var(--muted)" }}>{k}</span>
                          <span style={{ fontWeight: 600 }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {config.notification_channels && (
                  <div>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "6px", fontWeight: 600 }}>NOTIFICATION CHANNELS</div>
                    <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
                      {config.notification_channels.map(c => (
                        <span
                          key={c}
                          style={{
                            background: "rgba(79, 171, 255, 0.1)",
                            border: "1px solid rgba(143, 213, 255, 0.22)",
                            color: "#c9e8ff",
                            borderRadius: "6px",
                            padding: "3px 10px",
                            fontSize: "0.82rem",
                          }}
                        >
                          {c}
                        </span>
                      ))}
                      <button
                        onClick={testNotification}
                        disabled={testingNotif}
                        style={{ padding: "5px 14px", border: "1px solid var(--accent)", borderRadius: "6px", background: "transparent", color: "var(--accent)", cursor: "pointer", fontSize: "0.82rem", fontWeight: 600 }}
                      >
                        {testingNotif ? "Sending…" : "Test Notification"}
                      </button>
                      {testResult && <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>{testResult}</span>}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p style={{ color: "var(--muted)" }}>Loading config…</p>
            )}
          </div>
          <div
            style={{
              background: "linear-gradient(180deg, rgba(17, 28, 46, 0.96), rgba(9, 18, 33, 0.98))",
              border: "1px solid rgba(125, 162, 214, 0.18)",
              borderRadius: "8px",
              padding: "12px 16px",
              boxShadow: "inset 0 1px 0 rgba(255,255,255,0.03)",
            }}
          >
            <pre style={{ margin: 0, fontSize: "0.75rem", color: "#b8c7d9", overflow: "auto", maxHeight: "300px" }}>
              {JSON.stringify(config, null, 2)}
            </pre>
          </div>
        </div>
      )}

      {/* Logs Drawer */}
      <ContainerLogsDrawer containerName={logsFor} onClose={() => setLogsFor(null)} />
    </div>
  );
}

export default function InfrastructurePage() {
  return <InfraPageInner />;
}
