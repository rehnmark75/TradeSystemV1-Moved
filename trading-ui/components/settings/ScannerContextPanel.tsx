"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

interface AuditEntry {
  id: number;
  changed_by: string;
  changed_at: string;
  change_reason: string;
  previous_values: Record<string, unknown> | null;
  new_values: Record<string, unknown> | null;
}

interface ScannerContextPanelProps {
  paramName: string | null;
  currentValue: unknown;
  defaultValue: unknown;
  onReset?: () => void;
}

function formatVal(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function toLabel(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function ScannerContextPanel({
  paramName,
  currentValue,
  defaultValue,
  onReset,
}: ScannerContextPanelProps) {
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);

  useEffect(() => {
    if (!paramName) {
      setAuditEntries([]);
      return;
    }
    setAuditLoading(true);
    fetch(apiUrl(`/api/settings/scanner/audit?limit=30`))
      .then((r) => r.json())
      .then((data) => {
        const entries = Array.isArray(data) ? data : [];
        const relevant = entries.filter((e: AuditEntry) => {
          const keys = [
            ...Object.keys(e.new_values ?? {}),
            ...Object.keys(e.previous_values ?? {}),
          ];
          return keys.includes(paramName);
        });
        setAuditEntries(relevant.slice(0, 5));
      })
      .catch(() => setAuditEntries([]))
      .finally(() => setAuditLoading(false));
  }, [paramName]);

  if (!paramName) {
    return (
      <div className="context-panel context-panel--empty">
        <div className="context-panel-hint">Click a parameter to see details</div>
      </div>
    );
  }

  const hasDefault = defaultValue !== null && defaultValue !== undefined;
  const isDefault = hasDefault && String(currentValue) === String(defaultValue);

  return (
    <div className="context-panel">
      <div className="context-panel-header">
        <div className="context-panel-name">{toLabel(paramName)}</div>
        <div className="context-panel-param-id">{paramName}</div>
      </div>

      <div className="context-panel-meta">
        {hasDefault ? (
          <div className="context-panel-row">
            <span>Default</span>
            <strong>{formatVal(defaultValue)}</strong>
            {!isDefault && onReset ? (
              <button
                type="button"
                className="context-panel-reset"
                onClick={onReset}
                title="Reset to default"
              >
                Reset
              </button>
            ) : null}
          </div>
        ) : null}
        <div className="context-panel-row">
          <span>Current</span>
          <strong>{formatVal(currentValue)}</strong>
        </div>
      </div>

      <div className="context-panel-section">
        <div className="context-panel-section-title">Recent changes</div>
        {auditLoading ? (
          <div className="context-panel-loading">Loading…</div>
        ) : auditEntries.length === 0 ? (
          <div className="context-panel-empty-state">No recent changes</div>
        ) : (
          <div className="context-panel-audit">
            {auditEntries.map((entry) => (
              <div key={entry.id} className="context-audit-entry">
                <div className="context-audit-who">{entry.changed_by}</div>
                <div className="context-audit-when">{timeAgo(entry.changed_at)}</div>
                <div className="context-audit-reason">{entry.change_reason}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
