"use client";

import { useEffect, useState } from "react";
import HealthIndicator from "../../components/settings/HealthIndicator";
import { logTelemetry } from "../../lib/settings/telemetry";
import { apiUrl } from "../../lib/settings/api";
import { useEnvironment } from "../../lib/environment";

interface AuditEntry {
  id: number;
  changed_by: string;
  changed_at: string;
  change_reason?: string;
  category?: string;
}

export default function SettingsDashboard() {
  const { environment } = useEnvironment();
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [overrideCount, setOverrideCount] = useState<number | null>(null);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings" });
    const load = async () => {
      try {
        const cs = encodeURIComponent(environment);
        const [auditResponse, overridesResponse] = await Promise.all([
          fetch(apiUrl(`/api/settings/scanner/audit?limit=10&config_set=${cs}`)),
          fetch(apiUrl(`/api/settings/strategy/smc/pairs?config_set=${cs}`))
        ]);
        const auditPayload = await auditResponse.json();
        const overridesPayload = await overridesResponse.json();
        if (auditResponse.ok) {
          setAudit(auditPayload ?? []);
        }
        if (overridesResponse.ok) {
          setOverrideCount(overridesPayload?.overrides?.length ?? 0);
        }
      } catch {
        setAudit([]);
        setOverrideCount(null);
      }
    };
    load();
  }, [environment]);

  return (
    <div className="settings-panel">
      <HealthIndicator />
      <div className="settings-hero">
        <div className="mission-kicker">Configuration Governance</div>
        <h1>Settings Dashboard</h1>
        <p>Scanner, strategy, and audit controls with one commercial-grade governance surface.</p>
      </div>
      <div className="settings-dashboard-grid">
        <div className="settings-card">
          <h3>Recent Scanner Changes</h3>
          <ul>
            {audit.map((entry) => (
              <li key={entry.id}>
                <strong>{entry.changed_by}</strong> · {entry.change_reason ?? "Update"} ·{" "}
                {new Date(entry.changed_at).toLocaleString()}
              </li>
            ))}
          </ul>
        </div>
        <div className="settings-card">
          <h3>Pair Overrides</h3>
          <p>
            Active overrides:{" "}
            {overrideCount === null ? "Loading..." : overrideCount}
          </p>
          <p>Use the sidebar to manage scanner + strategy settings.</p>
        </div>
      </div>
    </div>
  );
}
