"use client";

import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";
import { epicToDisplayName } from "../../lib/settings/epicDisplay";
import type { SmcParameterMetadata, AuditEntry } from "../../types/settings";

interface ParameterContextPanelProps {
  paramName: string | null;
  metadata?: SmcParameterMetadata;
  currentValue: unknown;
  allPairEpics?: string[];
  allPairEffectiveValues?: Map<string, unknown>;
  onReset?: () => void;
}

function formatVal(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export default function ParameterContextPanel({
  paramName,
  metadata,
  currentValue,
  allPairEpics = [],
  allPairEffectiveValues = new Map(),
  onReset,
}: ParameterContextPanelProps) {
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);

  useEffect(() => {
    if (!paramName) {
      setAuditEntries([]);
      return;
    }
    setAuditLoading(true);
    fetch(apiUrl(`/api/settings/strategy/smc/audit?limit=20`))
      .then((r) => r.json())
      .then((data) => {
        const entries = Array.isArray(data) ? data : (data.entries ?? []);
        // Filter to entries that changed this specific parameter
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

  if (!paramName || !metadata) {
    return (
      <div className="context-panel context-panel--empty">
        <div className="context-panel-hint">
          Click a parameter to see details
        </div>
      </div>
    );
  }

  const defaultValue = metadata.default_value;
  const minValue = metadata.min_value;
  const maxValue = metadata.max_value;
  const hasDefault = defaultValue !== null && defaultValue !== undefined;
  const isCurrent = hasDefault && String(currentValue) === String(defaultValue);

  return (
    <div className="context-panel">
      <div className="context-panel-header">
        <div className="context-panel-name">{metadata.display_name}</div>
        <div className="context-panel-param-id">{paramName}</div>
      </div>

      {metadata.help_text || metadata.description ? (
        <div className="context-panel-description">
          {metadata.help_text ?? metadata.description}
        </div>
      ) : null}

      <div className="context-panel-meta">
        {hasDefault ? (
          <div className="context-panel-row">
            <span>Default</span>
            <strong>{formatVal(defaultValue)}</strong>
            {!isCurrent && onReset ? (
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
        {minValue !== null && minValue !== undefined ? (
          <div className="context-panel-row">
            <span>Range</span>
            <strong>
              {formatVal(minValue)} – {formatVal(maxValue)}
            </strong>
          </div>
        ) : null}
        {metadata.unit ? (
          <div className="context-panel-row">
            <span>Unit</span>
            <strong>{metadata.unit}</strong>
          </div>
        ) : null}
        {metadata.requires_restart ? (
          <div className="context-panel-row">
            <span>Note</span>
            <strong className="restart-note">Requires restart</strong>
          </div>
        ) : null}
      </div>

      {allPairEpics.length > 0 ? (
        <div className="context-panel-section">
          <div className="context-panel-section-title">Per-pair effective values</div>
          <div className="context-panel-pairs">
            {allPairEpics.map((epic) => {
              const val = allPairEffectiveValues.get(epic);
              const isOverride = val !== undefined && formatVal(val) !== formatVal(currentValue);
              return (
                <div key={epic} className={`context-pair-row ${isOverride ? "is-override" : ""}`}>
                  <span className="context-pair-name">{epicToDisplayName(epic)}</span>
                  <span className="context-pair-value">{formatVal(val ?? currentValue)}</span>
                  {isOverride ? <span className="context-pair-tag">override</span> : null}
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

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
