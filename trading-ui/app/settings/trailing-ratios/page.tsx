"use client";

import { useMemo, useState } from "react";
import { useEnvironment } from "../../../lib/environment";
import {
  useTrailingRatios,
  type TrailingRatioRow,
} from "../../../hooks/settings/useTrailingRatios";

type NumericField =
  | "early_be_trigger_ratio"
  | "stage1_trigger_ratio"
  | "stage2_trigger_ratio"
  | "stage3_trigger_ratio"
  | "break_even_trigger_ratio"
  | "partial_close_trigger_ratio"
  | "stage1_lock_ratio"
  | "stage2_lock_ratio"
  | "early_be_buffer_points"
  | "stage3_atr_multiplier"
  | "stage3_min_distance_ratio"
  | "min_trail_distance_ratio"
  | "min_early_be_trigger"
  | "min_stage1_trigger"
  | "min_stage1_lock"
  | "min_stage2_trigger"
  | "min_stage2_lock"
  | "min_stage3_trigger"
  | "min_break_even_trigger"
  | "min_trail_distance";

const RATIO_FIELDS: NumericField[] = [
  "early_be_trigger_ratio",
  "stage1_trigger_ratio",
  "stage2_trigger_ratio",
  "stage3_trigger_ratio",
  "break_even_trigger_ratio",
  "partial_close_trigger_ratio",
  "stage1_lock_ratio",
  "stage2_lock_ratio",
];

const FIXED_FIELDS: NumericField[] = [
  "early_be_buffer_points",
  "stage3_atr_multiplier",
  "stage3_min_distance_ratio",
  "min_trail_distance_ratio",
];

const MIN_FIELDS: NumericField[] = [
  "min_early_be_trigger",
  "min_stage1_trigger",
  "min_stage1_lock",
  "min_stage2_trigger",
  "min_stage2_lock",
  "min_stage3_trigger",
  "min_break_even_trigger",
  "min_trail_distance",
];

const ALL_FIELDS: NumericField[] = [...RATIO_FIELDS, ...FIXED_FIELDS, ...MIN_FIELDS];

const FIELD_LABELS: Record<NumericField, string> = {
  early_be_trigger_ratio: "Early BE × TP",
  stage1_trigger_ratio: "S1 trig × TP",
  stage2_trigger_ratio: "S2 trig × TP",
  stage3_trigger_ratio: "S3 trig × TP",
  break_even_trigger_ratio: "BE × TP",
  partial_close_trigger_ratio: "Partial close × TP",
  stage1_lock_ratio: "S1 lock × TP",
  stage2_lock_ratio: "S2 lock × TP",
  early_be_buffer_points: "Early BE buffer (pips)",
  stage3_atr_multiplier: "S3 ATR mult",
  stage3_min_distance_ratio: "S3 min dist × TP",
  min_trail_distance_ratio: "Min trail × TP",
  min_early_be_trigger: "Min early BE",
  min_stage1_trigger: "Min S1 trig",
  min_stage1_lock: "Min S1 lock",
  min_stage2_trigger: "Min S2 trig",
  min_stage2_lock: "Min S2 lock",
  min_stage3_trigger: "Min S3 trig",
  min_break_even_trigger: "Min BE",
  min_trail_distance: "Min trail (pips)",
};

export default function TrailingRatiosPage() {
  const { environment } = useEnvironment();
  const [isScalp, setIsScalp] = useState(false);
  const { rows, loading, error, saveRow, createRow, deleteRow, reload } =
    useTrailingRatios(environment, isScalp);

  const [drafts, setDrafts] = useState<Record<string, Record<string, unknown>>>({});
  const [saveState, setSaveState] = useState<Record<string, "idle" | "saving" | "saved" | "error">>({});
  const [saveError, setSaveError] = useState<Record<string, string>>({});
  const [newEpic, setNewEpic] = useState("");

  const [previewTp, setPreviewTp] = useState(30);

  const rowKey = (r: TrailingRatioRow) => `${r.epic}::${r.is_scalp}`;

  const onEdit = (row: TrailingRatioRow, field: NumericField, raw: string) => {
    const key = rowKey(row);
    const parsed = raw === "" ? null : Number(raw);
    setDrafts((prev) => ({
      ...prev,
      [key]: { ...(prev[key] ?? {}), [field]: parsed },
    }));
  };

  const getCell = (row: TrailingRatioRow, field: NumericField) => {
    const d = drafts[rowKey(row)];
    if (d && field in d) return d[field];
    return (row as unknown as Record<string, unknown>)[field];
  };

  const hasChanges = (row: TrailingRatioRow) => {
    const d = drafts[rowKey(row)];
    return Boolean(d && Object.keys(d).length > 0);
  };

  const onSave = async (row: TrailingRatioRow) => {
    const key = rowKey(row);
    const updates = drafts[key];
    if (!updates || Object.keys(updates).length === 0) return;

    setSaveState((prev) => ({ ...prev, [key]: "saving" }));
    setSaveError((prev) => ({ ...prev, [key]: "" }));
    try {
      await saveRow(row.epic, updates, {
        updatedBy: "admin",
        changeReason: `Ratio update via UI (${environment})`,
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

  const onAddOverride = async () => {
    if (!newEpic.trim()) return;
    try {
      await createRow(newEpic.trim(), {}, {
        updatedBy: "admin",
        changeReason: `Add pair override (${environment}, scalp=${isScalp})`,
      });
      setNewEpic("");
    } catch (err) {
      alert(err instanceof Error ? err.message : "Create failed");
    }
  };

  const onDelete = async (row: TrailingRatioRow) => {
    if (row.epic === "DEFAULT") return;
    if (!confirm(`Delete ${row.epic} override (scalp=${row.is_scalp})?`)) return;
    try {
      await deleteRow(row.epic, {
        updatedBy: "admin",
        changeReason: `Delete pair override (${environment})`,
      });
    } catch (err) {
      alert(err instanceof Error ? err.message : "Delete failed");
    }
  };

  // Preview: given a TP_pips, show the effective earlyBE / BE for DEFAULT row
  const defaultRow = rows.find((r) => r.epic === "DEFAULT");
  const preview = useMemo(() => {
    if (!defaultRow) return null;
    const r = (f: NumericField) => {
      const d = drafts[rowKey(defaultRow)];
      if (d && f in d) return d[f] as number | null;
      return defaultRow[f] as number | null;
    };
    const mk = (ratio: NumericField, min: NumericField) => {
      const rv = r(ratio);
      const mn = r(min);
      if (rv == null || mn == null) return null;
      return Math.max(mn, Math.round(previewTp * rv));
    };
    return {
      earlyBE: mk("early_be_trigger_ratio", "min_early_be_trigger"),
      be: mk("break_even_trigger_ratio", "min_break_even_trigger"),
      s1: mk("stage1_trigger_ratio", "min_stage1_trigger"),
      s2: mk("stage2_trigger_ratio", "min_stage2_trigger"),
      s3: mk("stage3_trigger_ratio", "min_stage3_trigger"),
    };
  }, [defaultRow, drafts, previewTp]);

  const sortedRows = useMemo(() => rows, [rows]);

  const fieldGroup = (title: string, fields: NumericField[]) => (
    <>
      <tr style={{ background: "#eef" }}>
        <td colSpan={fields.length + 2} style={{ padding: "4px 8px", fontWeight: 700, fontSize: "0.8rem" }}>
          {title}
        </td>
      </tr>
      {/* header rendered inline inside each section's table below */}
    </>
  );

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <h1>Trailing Ratios</h1>
        <p>
          Controls how <code>compute_sltp_trailing_config()</code> computes trigger/lock points
          from each trade&apos;s TP distance. DEFAULT row is the baseline; add pair rows to
          override specific pairs. Environment:{" "}
          <strong>{environment.toUpperCase()}</strong>. Cache TTL 120s.
        </p>
      </div>

      <div className="settings-form-actions" style={{ gap: 12, alignItems: "center" }}>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={isScalp}
            onChange={(e) => setIsScalp(e.target.checked)}
          />
          Scalp profile (is_scalp=true)
        </label>
        <button onClick={() => reload()}>Reload</button>

        <span style={{ marginLeft: 24 }}>
          Preview for TP =
          <input
            type="number"
            value={previewTp}
            onChange={(e) => setPreviewTp(Number(e.target.value) || 0)}
            style={{ width: 60, margin: "0 4px" }}
          />
          pips →
          {preview ? (
            <span style={{ marginLeft: 8 }}>
              earlyBE=<strong>{preview.earlyBE}</strong>, BE=<strong>{preview.be}</strong>,
              S1=<strong>{preview.s1}</strong>, S2=<strong>{preview.s2}</strong>,
              S3=<strong>{preview.s3}</strong> pts
            </span>
          ) : null}
        </span>
      </div>

      <div style={{ margin: "12px 0", display: "flex", gap: 8, alignItems: "center" }}>
        <input
          placeholder="Add pair override (e.g. CS.D.AUDJPY.MINI.IP)"
          value={newEpic}
          onChange={(e) => setNewEpic(e.target.value)}
          style={{ width: 300 }}
        />
        <button onClick={onAddOverride} disabled={!newEpic.trim()}>
          Add override
        </button>
      </div>

      {loading ? (
        <div className="settings-placeholder">Loading...</div>
      ) : error ? (
        <div className="settings-placeholder">Error: {error}</div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table
            className="trailing-ratios-table"
            style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.8rem" }}
          >
            <thead>
              <tr>
                <th
                  style={{
                    textAlign: "left",
                    position: "sticky",
                    left: 0,
                    background: "linear-gradient(180deg, rgba(17, 28, 46, 0.98), rgba(12, 21, 36, 0.98))",
                    padding: "6px 8px",
                  }}
                >
                  Epic
                </th>
                {ALL_FIELDS.map((f) => (
                  <th
                    key={f}
                    title={FIELD_LABELS[f]}
                    style={{ textAlign: "right", padding: "6px 6px", whiteSpace: "nowrap" }}
                  >
                    {FIELD_LABELS[f]}
                  </th>
                ))}
                <th style={{ padding: "6px 8px" }}>Save</th>
                <th style={{ padding: "6px 8px" }}>Del</th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row) => {
                const key = rowKey(row);
                const state = saveState[key] ?? "idle";
                const isDefault = row.epic === "DEFAULT";
                return (
                  <tr
                    key={key}
                    style={{
                      background: isDefault ? "#f7f7fa" : undefined,
                      borderBottom: "1px solid #eee",
                    }}
                  >
                    <td
                      style={{
                        padding: "4px 8px",
                        fontWeight: 600,
                        position: "sticky",
                        left: 0,
                        background: isDefault ? "#f7f7fa" : "#fff",
                      }}
                    >
                      {row.epic}
                    </td>
                    {ALL_FIELDS.map((f) => {
                      const val = getCell(row, f);
                      return (
                        <td key={f} style={{ padding: "2px 4px" }}>
                          <input
                            type="number"
                            step="any"
                            value={val === null || val === undefined ? "" : String(val)}
                            onChange={(e) => onEdit(row, f, e.target.value)}
                            style={{ width: 70, padding: "2px 4px", textAlign: "right" }}
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
                        <div style={{ color: "#b91c1c", fontSize: "0.7rem" }}>
                          {saveError[key]}
                        </div>
                      ) : null}
                    </td>
                    <td style={{ padding: "2px 4px", textAlign: "center" }}>
                      {!isDefault ? (
                        <button onClick={() => onDelete(row)} style={{ color: "#b91c1c" }}>
                          ✕
                        </button>
                      ) : (
                        <span style={{ color: "#999" }}>—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div style={{ marginTop: 8, fontSize: "0.75rem", color: "#666" }}>
            Empty cells on pair rows inherit from the DEFAULT row at runtime.
            Grouping: ratios × TP (0-1 scale), then fixed values, then minimum floors (pips).
          </div>
        </div>
      )}
    </div>
  );
}
