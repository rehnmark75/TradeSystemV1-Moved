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
        <Link href="/" className="brand">Trading Hub</Link>
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

      {/* Header */}
      <div className="header" style={{ marginBottom: "20px" }}>
        <div>
          <h1 style={{ margin: 0, display: "flex", alignItems: "center", gap: "12px" }}>
            Infrastructure
            <StatusPill
              state={healthScore >= 90 ? "healthy" : healthScore >= 70 ? "degraded" : "down"}
              label={`${Math.round(healthScore)}% health`}
            />
          </h1>
          <p style={{ margin: "6px 0 0", opacity: 0.85, fontSize: "0.9rem" }}>
            Container health, alerts, and system monitoring — internal network only
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <LiveBadge lastUpdated={lastUpdated} />
          <button onClick={refetch} className="refresh-btn" title="Refresh">↻</button>
        </div>
      </div>

      {/* Disclaimer */}
      <div style={{
        background: "#fff9e6", border: "1px solid #f0c040", borderRadius: "8px",
        padding: "8px 14px", marginBottom: "16px", fontSize: "0.8rem", color: "#6b5000",
      }}>
        ⚠️ Operator console — actions like restart and notification test are internal-network only with no auth layer. Handle with care.
      </div>

      {/* KPI Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: "10px", marginBottom: "20px" }}>
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
                        <span key={c} style={{ background: "#fff0f0", border: "1px solid #fcc", borderRadius: "6px", padding: "2px 10px", fontSize: "0.82rem", fontFamily: "monospace" }}>{c}</span>
                      ))}
                    </div>
                  </div>
                )}
                {config.thresholds && (
                  <div>
                    <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginBottom: "6px", fontWeight: 600 }}>THRESHOLDS</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(200px,1fr))", gap: "8px" }}>
                      {Object.entries(config.thresholds).map(([k, v]) => (
                        <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", background: "#fafaf8", border: "1px solid var(--border)", borderRadius: "6px", padding: "6px 10px" }}>
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
                        <span key={c} style={{ background: "#f0f8ff", border: "1px solid #cde", borderRadius: "6px", padding: "3px 10px", fontSize: "0.82rem" }}>{c}</span>
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
          <div style={{ background: "#fafaf8", border: "1px solid var(--border)", borderRadius: "8px", padding: "12px 16px" }}>
            <pre style={{ margin: 0, fontSize: "0.75rem", color: "var(--muted)", overflow: "auto", maxHeight: "300px" }}>
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
