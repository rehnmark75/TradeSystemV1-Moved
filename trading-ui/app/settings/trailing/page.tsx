"use client";

import { useMemo, useState } from "react";
import { useEnvironment } from "../../../lib/environment";
import {
  useTrailingConfig,
  type TrailingConfigRow,
} from "../../../hooks/settings/useTrailingConfig";

type NumericField =
  | "early_breakeven_trigger_points"
  | "early_breakeven_buffer_points"
  | "stage1_trigger_points"
  | "stage1_lock_points"
  | "stage2_trigger_points"
  | "stage2_lock_points"
  | "stage3_trigger_points"
  | "stage3_atr_multiplier"
  | "stage3_min_distance"
  | "min_trail_distance"
  | "break_even_trigger_points"
  | "partial_close_trigger_points"
  | "partial_close_size";

type BoolField = "enable_partial_close";

type AnyField = NumericField | BoolField;

const NUMERIC_FIELDS: NumericField[] = [
  "early_breakeven_trigger_points",
  "early_breakeven_buffer_points",
  "stage1_trigger_points",
  "stage1_lock_points",
  "stage2_trigger_points",
  "stage2_lock_points",
  "stage3_trigger_points",
  "stage3_atr_multiplier",
  "stage3_min_distance",
  "min_trail_distance",
  "break_even_trigger_points",
  "partial_close_trigger_points",
  "partial_close_size",
];

const BOOL_FIELDS: BoolField[] = ["enable_partial_close"];

const FIELD_LABELS: Record<AnyField, string> = {
  early_breakeven_trigger_points: "Early BE trigger (pips)",
  early_breakeven_buffer_points: "Early BE buffer (pips)",
  stage1_trigger_points: "Stage 1 trigger (pips)",
  stage1_lock_points: "Stage 1 lock (pips)",
  stage2_trigger_points: "Stage 2 trigger (pips)",
  stage2_lock_points: "Stage 2 lock (pips)",
  stage3_trigger_points: "Stage 3 trigger (pips)",
  stage3_atr_multiplier: "Stage 3 ATR multiplier",
  stage3_min_distance: "Stage 3 min distance (pips)",
  min_trail_distance: "Min trail distance (pips)",
  break_even_trigger_points: "BE trigger (pips)",
  enable_partial_close: "Partial close enabled",
  partial_close_trigger_points: "Partial close trigger (pips)",
  partial_close_size: "Partial close size (0.0-1.0)",
};

export default function TrailingSettingsPage() {
  const { environment } = useEnvironment();
  const [isScalp, setIsScalp] = useState(false);
  const { rows, loading, error, saveRow, reload } = useTrailingConfig(environment, isScalp);

  const [drafts, setDrafts] = useState<Record<string, Record<string, unknown>>>({});
  const [saveState, setSaveState] = useState<Record<string, "idle" | "saving" | "saved" | "error">>({});
  const [saveError, setSaveError] = useState<Record<string, string>>({});

  const rowKey = (r: TrailingConfigRow) => `${r.epic}::${r.is_scalp}`;

  const onEdit = (row: TrailingConfigRow, field: AnyField, raw: string | boolean) => {
    const key = rowKey(row);
    let parsed: unknown = raw;
    if (BOOL_FIELDS.includes(field as BoolField)) {
      parsed = Boolean(raw);
    } else {
      // Numeric — allow "" to mean null
      if (raw === "" || raw === null) parsed = null;
      else parsed = Number(raw);
    }
    setDrafts((prev) => ({
      ...prev,
      [key]: { ...(prev[key] ?? {}), [field]: parsed },
    }));
  };

  const onSave = async (row: TrailingConfigRow) => {
    const key = rowKey(row);
    const updates = drafts[key];
    if (!updates || Object.keys(updates).length === 0) return;

    setSaveState((prev) => ({ ...prev, [key]: "saving" }));
    setSaveError((prev) => ({ ...prev, [key]: "" }));

    try {
      await saveRow(row.epic, updates, {
        updatedBy: "admin",
        changeReason: `Trailing update via UI (${environment})`,
        updatedAt: row.updated_at,
      });
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      setSaveState((prev) => ({ ...prev, [key]: "saved" }));
      setTimeout(() => setSaveState((p) => ({ ...p, [key]: "idle" })), 1500);
    } catch (err) {
      setSaveState((prev) => ({ ...prev, [key]: "error" }));
      setSaveError((prev) => ({
        ...prev,
        [key]: err instanceof Error ? err.message : "Save failed",
      }));
    }
  };

  const hasChanges = (row: TrailingConfigRow) => {
    const d = drafts[rowKey(row)];
    return Boolean(d && Object.keys(d).length > 0);
  };

  const getCell = (row: TrailingConfigRow, field: AnyField) => {
    const d = drafts[rowKey(row)];
    if (d && field in d) return d[field];
    return (row as unknown as Record<string, unknown>)[field];
  };

  const sortedRows = useMemo(() => rows, [rows]);

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <div className="mission-kicker">Risk Mechanics</div>
        <h1>Trailing Stop Settings</h1>
        <p>
          Per-pair trailing stop stages. Demo and live are independent — changes
          here apply only to the <strong>{environment.toUpperCase()}</strong> environment.
        </p>
      </div>

      <div className="settings-form-actions" style={{ gap: 12, alignItems: "center" }}>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={isScalp}
            onChange={(e) => setIsScalp(e.target.checked)}
          />
          Scalp configs (is_scalp=true)
        </label>
        <button onClick={() => reload()}>Reload</button>
      </div>

      {loading ? (
        <div className="settings-placeholder">Loading...</div>
      ) : error ? (
        <div className="settings-placeholder">Error: {error}</div>
      ) : (
        <div className="settings-card" style={{ overflowX: "auto" }}>
          <table className="trailing-config-table" style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.85rem" }}>
            <thead>
              <tr>
                <th
                  style={{
                    textAlign: "left",
                    position: "sticky",
                    left: 0,
                    background: "linear-gradient(180deg, rgba(17, 28, 46, 0.98), rgba(12, 21, 36, 0.98))",
                  }}
                >
                  Epic
                </th>
                {NUMERIC_FIELDS.map((f) => (
                  <th key={f} title={FIELD_LABELS[f]} style={{ textAlign: "right", padding: "6px 8px", whiteSpace: "nowrap" }}>
                    {FIELD_LABELS[f]}
                  </th>
                ))}
                {BOOL_FIELDS.map((f) => (
                  <th key={f} title={FIELD_LABELS[f]} style={{ padding: "6px 8px" }}>
                    {FIELD_LABELS[f]}
                  </th>
                ))}
                <th style={{ padding: "6px 8px" }}>Save</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row) => {
                const key = rowKey(row);
                const state = saveState[key] ?? "idle";
                return (
                  <tr
                    key={key}
                    style={{
                      background: row.epic === "DEFAULT" ? "rgba(255, 255, 255, 0.03)" : undefined,
                      borderBottom: "1px solid var(--border)",
                    }}
                  >
                    <td style={{ padding: "4px 8px", fontWeight: 600, position: "sticky", left: 0, background: row.epic === "DEFAULT" ? "#f7f7fa" : "#fff" }}>
                      {row.epic}
                    </td>
                    {NUMERIC_FIELDS.map((f) => {
                      const val = getCell(row, f);
                      return (
                        <td key={f} style={{ padding: "2px 4px" }}>
                          <input
                            type="number"
                            step="any"
                            value={val === null || val === undefined ? "" : String(val)}
                            onChange={(e) => onEdit(row, f, e.target.value)}
                            style={{ width: 80, padding: "2px 4px", textAlign: "right" }}
                          />
                        </td>
                      );
                    })}
                    {BOOL_FIELDS.map((f) => {
                      const val = getCell(row, f);
                      return (
                        <td key={f} style={{ padding: "2px 4px", textAlign: "center" }}>
                          <input
                            type="checkbox"
                            checked={Boolean(val)}
                            onChange={(e) => onEdit(row, f, e.target.checked)}
                          />
                        </td>
                      );
                    })}
                    <td style={{ padding: "2px 4px", textAlign: "center" }}>
                      <button
                        disabled={!hasChanges(row) || state === "saving"}
                        onClick={() => onSave(row)}
                      >
                        {state === "saving" ? "..." : state === "saved" ? "✓" : "Save"}
                      </button>
                      {state === "error" && saveError[key] ? (
                        <div style={{ color: "#b91c1c", fontSize: "0.7rem" }}>{saveError[key]}</div>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
