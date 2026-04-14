"use client";

import { useState } from "react";
import { getParamRiskLevel } from "../../lib/settings/riskClassification";

interface SaveModalProps {
  changes: Record<string, unknown>;
  originalValues: Record<string, unknown>;
  onConfirm: (changeReason: string) => Promise<void> | void;
  onCancel: () => void;
  saving?: boolean;
  error?: string | null;
}

function formatVal(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function toLabel(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function SaveModal({
  changes,
  originalValues,
  onConfirm,
  onCancel,
  saving = false,
  error = null,
}: SaveModalProps) {
  const [reason, setReason] = useState("");

  const changedFields = Object.keys(changes);
  const criticalFields = changedFields.filter(
    (f) => getParamRiskLevel(f) === "critical"
  );
  const highFields = changedFields.filter(
    (f) => getParamRiskLevel(f) === "high"
  );
  const hasCritical = criticalFields.length > 0;

  const handleConfirm = async () => {
    if (!reason.trim() || saving) return;
    await onConfirm(reason.trim());
  };

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="save-modal">
        <div className="save-modal-header">
          <h2>Confirm changes</h2>
          <button type="button" className="modal-close" onClick={onCancel}>
            ×
          </button>
        </div>

        {hasCritical ? (
          <div className="save-modal-alert save-modal-alert--critical">
            ⚠ {criticalFields.length} critical parameter{criticalFields.length > 1 ? "s" : ""} modified —
            this may directly affect live trading.
          </div>
        ) : highFields.length > 0 ? (
          <div className="save-modal-alert save-modal-alert--warn">
            {highFields.length} high-impact parameter{highFields.length > 1 ? "s" : ""} modified.
          </div>
        ) : null}

        {error ? (
          <div className="save-modal-alert save-modal-alert--critical">
            {error}
          </div>
        ) : null}

        <div className="save-modal-diff">
          <div className="save-modal-diff-header">
            <span>Parameter</span>
            <span>Before</span>
            <span>After</span>
          </div>
          <div className="save-modal-diff-rows">
            {changedFields.map((field) => {
              const risk = getParamRiskLevel(field);
              return (
                <div key={field} className={`save-modal-diff-row risk-row--${risk}`}>
                  <span className="diff-field-name">{toLabel(field)}</span>
                  <span className="diff-before">{formatVal(originalValues[field])}</span>
                  <span className="diff-after">{formatVal(changes[field])}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="save-modal-reason">
          <label htmlFor="change-reason">
            Change reason <span className="required">*</span>
          </label>
          <textarea
            id="change-reason"
            placeholder="Describe why you are making these changes…"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            rows={3}
            autoFocus
          />
        </div>

        <div className="save-modal-actions">
          <button type="button" className="btn-ghost" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className={`btn-primary ${!reason.trim() || saving ? "disabled" : hasCritical ? "danger" : ""}`}
            onClick={handleConfirm}
            disabled={!reason.trim() || saving}
          >
            {saving ? "Saving..." : hasCritical ? "⚠ Confirm & Save" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
