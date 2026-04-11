"use client";

import type { SnapshotDiff } from "../../types/settings";

interface SnapshotDiffViewProps {
  snapshotName: string;
  snapshotDate: string;
  diff: SnapshotDiff[];
  changedCount: number;
  onRestore: () => void;
  onClose: () => void;
}

const CATEGORY_ORDER = [
  "Tier 1: 4H Directional Bias",
  "Tier 2: 15m Entry Trigger",
  "Tier 3: 5m Execution",
  "Risk Management",
  "Session Filter",
  "Confidence Scoring",
  "Other",
];

function formatVal(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function toLabel(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function getDiffClass(sv: unknown, cv: unknown): string {
  if (typeof sv === "number" && typeof cv === "number") {
    if (sv > cv) return "diff-higher";
    if (sv < cv) return "diff-lower";
  }
  return "diff-changed";
}

export default function SnapshotDiffView({
  snapshotName,
  snapshotDate,
  diff,
  changedCount,
  onRestore,
  onClose,
}: SnapshotDiffViewProps) {
  const changedDiff = diff.filter((d) => d.changed);

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="snapshot-diff-modal">
        <div className="snapshot-diff-header">
          <div>
            <h2>Compare: {snapshotName}</h2>
            <p className="snapshot-diff-meta">
              Snapshot from {new Date(snapshotDate).toLocaleDateString()} ·{" "}
              {changedCount} field{changedCount !== 1 ? "s" : ""} differ
            </p>
          </div>
          <div className="snapshot-diff-actions">
            <button type="button" className="btn-primary" onClick={onRestore}>
              Restore this snapshot
            </button>
            <button type="button" className="modal-close" onClick={onClose}>
              ×
            </button>
          </div>
        </div>

        {changedCount === 0 ? (
          <div className="snapshot-diff-empty">
            Current config matches this snapshot exactly.
          </div>
        ) : (
          <div className="snapshot-diff-table">
            <div className="snapshot-diff-cols">
              <span>Parameter</span>
              <span>Snapshot</span>
              <span>Current</span>
            </div>
            <div className="snapshot-diff-rows">
              {changedDiff.map((row) => (
                <div
                  key={row.field}
                  className={`snapshot-diff-row ${getDiffClass(row.snapshot_value, row.current_value)}`}
                >
                  <span className="diff-field">{toLabel(row.field)}</span>
                  <span className="diff-snapshot">{formatVal(row.snapshot_value)}</span>
                  <span className="diff-current">{formatVal(row.current_value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
