"use client";

import Link from "next/link";
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
        <div className="mission-kicker">Governance Desk</div>
        <h1>Operational Settings</h1>
        <p>
          Control scanner behavior, strategy logic, trailing policies, and audit visibility from one environment-aware
          governance surface.
        </p>
      </div>

      <div className="settings-overview-strip">
        <div className="settings-overview-stat">
          <span>Environment</span>
          <strong>{environment.toUpperCase()}</strong>
        </div>
        <div className="settings-overview-stat">
          <span>Pair Overrides</span>
          <strong>{overrideCount === null ? "..." : overrideCount}</strong>
        </div>
        <div className="settings-overview-stat">
          <span>Recent Changes</span>
          <strong>{audit.length}</strong>
        </div>
      </div>

      <div className="settings-dashboard-grid settings-dashboard-grid-featured">
        <div className="settings-card settings-card-feature">
          <div className="settings-card-kicker">Discovery</div>
          <h3>Scanner Control</h3>
          <p>
            Tune validation thresholds, cooldown behavior, and scan filters from a structured control surface built for
            operational edits.
          </p>
          <Link href="/settings/scanner" className="settings-card-link">Open Scanner Settings</Link>
        </div>

        <div className="settings-card settings-card-feature">
          <div className="settings-card-kicker">Execution Logic</div>
          <h3>Strategy Control</h3>
          <p>
            Manage global behavior and pair overrides with clearer separation between inherited defaults and explicit
            changes.
          </p>
          <Link href="/settings/strategy" className="settings-card-link">Open Strategy Settings</Link>
        </div>

        <div className="settings-card settings-card-feature">
          <div className="settings-card-kicker">Trade Handling</div>
          <h3>Trailing Control</h3>
          <p>
            Review trailing-stop stages and ratio ladders without leaving the main governance surface.
          </p>
          <div className="settings-inline-links">
            <Link href="/settings/trailing" className="settings-card-link">Trailing Stops</Link>
            <Link href="/settings/trailing-ratios" className="settings-card-link">Trailing Ratios</Link>
          </div>
        </div>
      </div>

      <div className="settings-dashboard-grid">
        <div className="settings-card">
          <div className="settings-card-kicker">Recent Activity</div>
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
          <div className="settings-card-kicker">Governance Notes</div>
          <h3>Control Focus</h3>
          <p>
            Use scanner settings for discovery behavior, strategy settings for execution logic, and trailing settings
            for live trade management. Audit remains the verification layer for every change.
          </p>
          <p>
            Active overrides: {overrideCount === null ? "Loading..." : overrideCount}
          </p>
        </div>
      </div>
    </div>
  );
}
