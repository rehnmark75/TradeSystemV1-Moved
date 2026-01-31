"use client";

import { useEffect, useState } from "react";
import { logTelemetry } from "../../../lib/settings/telemetry";
import { apiUrl } from "../../../lib/settings/api";

interface AuditEntry {
  id: number;
  changed_by: string;
  changed_at: string;
  change_reason?: string;
  category?: string | null;
  change_type?: string;
  source?: "scanner" | "smc";
}

export default function SettingsAuditPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/audit" });
    const load = async () => {
      try {
        const [scannerResponse, smcResponse] = await Promise.all([
          fetch(apiUrl("/api/settings/scanner/audit?limit=50")),
          fetch(apiUrl("/api/settings/strategy/smc/audit?limit=50"))
        ]);
        const scannerPayload = await scannerResponse.json();
        const smcPayload = await smcResponse.json();
        const merged = [
          ...(scannerResponse.ok
            ? (scannerPayload ?? []).map((entry: AuditEntry) => ({
                ...entry,
                source: "scanner" as const
              }))
            : []),
          ...(smcResponse.ok
            ? (smcPayload ?? []).map((entry: AuditEntry) => ({
                ...entry,
                source: "smc" as const
              }))
            : [])
        ].sort(
          (a, b) =>
            new Date(b.changed_at).getTime() - new Date(a.changed_at).getTime()
        );
        setEntries(merged);
      } catch {
        setEntries([]);
      }
    };
    load();
  }, []);

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <h1>Settings Audit Trail</h1>
        <p>Unified history of scanner + strategy changes.</p>
      </div>
      <div className="audit-table">
        <div className="audit-row audit-header">
          <span>User</span>
          <span>Reason</span>
          <span>Category</span>
          <span>When</span>
        </div>
        {entries.map((entry) => (
          <div key={`${entry.source ?? "scanner"}-${entry.id}`} className="audit-row">
            <span>{entry.changed_by}</span>
            <span>
              {entry.change_reason ?? "Update"}{" "}
              <em>({entry.source ?? "scanner"})</em>
            </span>
            <span>{entry.category ?? entry.change_type ?? "-"}</span>
            <span>{new Date(entry.changed_at).toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
