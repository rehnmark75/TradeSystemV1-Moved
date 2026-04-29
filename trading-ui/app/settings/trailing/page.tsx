"use client";

import { useMemo, useState } from "react";
import { useEnvironment } from "../../../lib/environment";
import {
  useTrailingConfig,
  TRAILING_STRATEGIES,
  type TrailingConfigRow,
  type TrailingStrategy,
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

type StageFilter = "all" | "early" | "stage1" | "stage2" | "stage3" | "partials";

const FIELD_GROUPS: Array<{
  id: StageFilter;
  label: string;
  fields: AnyField[];
}> = [
  {
    id: "early",
    label: "Breakeven",
    fields: ["early_breakeven_trigger_points", "early_breakeven_buffer_points"],
  },
  {
    id: "stage1",
    label: "Stage 1",
    fields: ["stage1_trigger_points", "stage1_lock_points"],
  },
  {
    id: "stage2",
    label: "Stage 2",
    fields: ["stage2_trigger_points", "stage2_lock_points"],
  },
  {
    id: "stage3",
    label: "Stage 3",
    fields: ["stage3_trigger_points", "stage3_atr_multiplier", "stage3_min_distance", "min_trail_distance"],
  },
  {
    id: "partials",
    label: "Partials",
    fields: ["break_even_trigger_points", "enable_partial_close", "partial_close_trigger_points", "partial_close_size"],
  },
];

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

const compactEpic = (epic: string) =>
  epic
    .replace(/^CS\.D\./, "")
    .replace(/\.MINI\.IP$/, "")
    .replace(/\.CEEM\.IP$/, "")
    .replace(/\.IP$/, "");

const formatNumber = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : String(value);

export default function TrailingSettingsPage() {
  const { environment } = useEnvironment();
  const [isScalp, setIsScalp] = useState(false);
  const [strategy, setStrategy] = useState<TrailingStrategy>("DEFAULT");
  const { rows, loading, error, saveRow, resetOverride, reload } = useTrailingConfig(
    environment,
    isScalp,
    strategy
  );

  const [drafts, setDrafts] = useState<Record<string, Record<string, unknown>>>({});
  const [saveState, setSaveState] = useState<Record<string, "idle" | "saving" | "saved" | "error">>({});
  const [resetState, setResetState] = useState<Record<string, "idle" | "resetting" | "reset" | "error">>({});
  const [saveError, setSaveError] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [stageFilter, setStageFilter] = useState<StageFilter>("all");
  const [showChangedOnly, setShowChangedOnly] = useState(false);

  const rowKey = (r: TrailingConfigRow) => `${r.strategy}::${r.epic}::${r.is_scalp}`;

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

  const onResetOverride = async (row: TrailingConfigRow) => {
    if (strategy === "DEFAULT" || row.inherited || row.id === null) return;
    const key = rowKey(row);
    const ok = window.confirm(`Reset ${compactEpic(row.epic)} ${strategy} trailing override?`);
    if (!ok) return;

    setResetState((prev) => ({ ...prev, [key]: "resetting" }));
    setSaveError((prev) => ({ ...prev, [key]: "" }));

    try {
      await resetOverride(row.epic, {
        updatedBy: "admin",
        changeReason: `Reset trailing override via UI (${environment})`,
      });
      setDrafts((prev) => {
        const next = { ...prev };
        delete next[key];
        return next;
      });
      setResetState((prev) => ({ ...prev, [key]: "reset" }));
      setTimeout(() => setResetState((p) => ({ ...p, [key]: "idle" })), 1500);
    } catch (err) {
      setResetState((prev) => ({ ...prev, [key]: "error" }));
      setSaveError((prev) => ({
        ...prev,
        [key]: err instanceof Error ? err.message : "Reset failed",
      }));
    }
  };

  const hasChanges = (row: TrailingConfigRow) => {
    const d = drafts[rowKey(row)];
    return Boolean(d && Object.keys(d).length > 0);
  };

  const changedCount = useMemo(() => Object.keys(drafts).length, [drafts]);

  const getCell = (row: TrailingConfigRow, field: AnyField) => {
    const d = drafts[rowKey(row)];
    if (d && field in d) return d[field];
    return (row as unknown as Record<string, unknown>)[field];
  };

  const visibleGroups = useMemo(
    () =>
      stageFilter === "all"
        ? FIELD_GROUPS
        : FIELD_GROUPS.filter((group) => group.id === stageFilter),
    [stageFilter]
  );

  const visibleFields = useMemo(
    () => visibleGroups.flatMap((group) => group.fields),
    [visibleGroups]
  );

  const numericVisibleFields = useMemo(
    () => visibleFields.filter((field): field is NumericField => NUMERIC_FIELDS.includes(field as NumericField)),
    [visibleFields]
  );

  const boolVisibleFields = useMemo(
    () => visibleFields.filter((field): field is BoolField => BOOL_FIELDS.includes(field as BoolField)),
    [visibleFields]
  );

  const filteredRows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return rows.filter((row) => {
      if (showChangedOnly && !hasChanges(row)) return false;
      if (!needle) return true;
      return row.epic.toLowerCase().includes(needle) || compactEpic(row.epic).toLowerCase().includes(needle);
    });
  }, [rows, query, showChangedOnly, drafts]);

  const profileLabel = isScalp ? "Scalp" : "Standard";
  const defaultRow = rows.find((row) => row.epic === "DEFAULT");
  const partialsEnabled = rows.filter((row) => row.enable_partial_close).length;
  const inheritedCount = rows.filter((row) => row.inherited).length;
  const overrideCount = rows.filter((row) => !row.inherited && row.strategy !== "DEFAULT").length;

  return (
    <div className="settings-panel trailing-page">
      <div className="settings-hero trailing-hero">
        <div>
          <div className="mission-kicker">Risk Mechanics</div>
          <h1>Trailing Stop Settings</h1>
          <p>
            Per-pair trailing stop stages. Demo and live are independent; this view is editing{" "}
            <strong>{environment.toUpperCase()}</strong>. Strategy rows inherit DEFAULT values until overridden.
          </p>
        </div>
        <div className="trailing-hero-stats" aria-label="Trailing config summary">
          <div>
            <span>Profile</span>
            <strong>{profileLabel}</strong>
          </div>
          <div>
            <span>Overrides</span>
            <strong>{strategy === "DEFAULT" ? "Base" : overrideCount}</strong>
          </div>
          <div>
            <span>Drafts</span>
            <strong>{changedCount}</strong>
          </div>
        </div>
      </div>

      <div className="trailing-toolbar">
        <div className="trailing-profile-switch" aria-label="Trailing profile">
          <button
            type="button"
            className={!isScalp ? "active" : ""}
            onClick={() => setIsScalp(false)}
          >
            Standard
          </button>
          <button
            type="button"
            className={isScalp ? "active" : ""}
            onClick={() => setIsScalp(true)}
          >
            Scalp
          </button>
        </div>

        <label className="trailing-search" style={{ minWidth: 200 }}>
          <span>Strategy</span>
          <select
            value={strategy}
            onChange={(event) => setStrategy(event.target.value as TrailingStrategy)}
          >
            {TRAILING_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="trailing-search">
          <span>Search epic</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="AUDUSD, DEFAULT..."
          />
        </label>

        <label className="trailing-filter-toggle">
          <input
            type="checkbox"
            checked={showChangedOnly}
            onChange={(event) => setShowChangedOnly(event.target.checked)}
          />
          Changed only
        </label>

        <button className="trailing-reload" type="button" onClick={() => reload()}>
          Reload
        </button>
      </div>

      <div className="trailing-stage-tabs" aria-label="Trailing stage columns">
        {(["all", "early", "stage1", "stage2", "stage3", "partials"] as StageFilter[]).map((stage) => (
          <button
            key={stage}
            type="button"
            className={stageFilter === stage ? "active" : ""}
            onClick={() => setStageFilter(stage)}
          >
            {stage === "all" ? "All columns" : FIELD_GROUPS.find((group) => group.id === stage)?.label}
          </button>
        ))}
      </div>

      {!loading && !error ? (
        <div className="trailing-insight-strip">
          <div>
            <span>Default early BE</span>
            <strong>
              {formatNumber(defaultRow?.early_breakeven_trigger_points)} / {formatNumber(defaultRow?.early_breakeven_buffer_points)}
            </strong>
          </div>
          <div>
            <span>Default stage locks</span>
            <strong>
              {formatNumber(defaultRow?.stage1_lock_points)} / {formatNumber(defaultRow?.stage2_lock_points)}
            </strong>
          </div>
          <div>
            <span>Partial close enabled</span>
            <strong>{partialsEnabled}</strong>
          </div>
          <div>
            <span>Visible pairs</span>
            <strong>{filteredRows.length}</strong>
          </div>
          <div>
            <span>Inherited rows</span>
            <strong>{strategy === "DEFAULT" ? "—" : inheritedCount}</strong>
          </div>
        </div>
      ) : null}

      {loading ? (
        <div className="settings-placeholder">Loading...</div>
      ) : error ? (
        <div className="settings-placeholder">Error: {error}</div>
      ) : (
        <div className="trailing-table-shell">
          <table className="trailing-config-table">
            <thead>
              <tr>
                <th className="trailing-epic-head" rowSpan={2}>
                  Epic
                </th>
                {visibleGroups.map((group) => (
                  <th key={group.id} className="trailing-group-head" colSpan={group.fields.length}>
                    {group.label}
                  </th>
                ))}
                <th className="trailing-action-head" rowSpan={2}>State</th>
              </tr>
              <tr>
                {numericVisibleFields.map((f) => (
                  <th key={f} title={FIELD_LABELS[f]} className="trailing-field-head">
                    {FIELD_LABELS[f]}
                  </th>
                ))}
                {boolVisibleFields.map((f) => (
                  <th key={f} title={FIELD_LABELS[f]} className="trailing-field-head">
                    {FIELD_LABELS[f]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => {
                const key = rowKey(row);
                const state = saveState[key] ?? "idle";
                const reset = resetState[key] ?? "idle";
                const rowHasChanges = hasChanges(row);
                const isInherited = Boolean(row.inherited);
                const canReset = strategy !== "DEFAULT" && !isInherited && row.id !== null;
                return (
                  <tr
                    key={key}
                    className={`${row.epic === "DEFAULT" ? "default-row" : ""} ${isInherited ? "inherited-row" : ""} ${canReset ? "override-row" : ""} ${rowHasChanges ? "changed-row" : ""}`}
                  >
                    <td className="trailing-epic-cell">
                      <strong>{compactEpic(row.epic)}</strong>
                      <span>{row.epic}</span>
                      {strategy !== "DEFAULT" ? (
                        <em className={isInherited ? "trailing-state-badge inherited" : "trailing-state-badge override"}>
                          {isInherited ? "Inherited" : `${row.override_field_count ?? 0} overrides`}
                        </em>
                      ) : null}
                    </td>
                    {numericVisibleFields.map((f) => {
                      const val = getCell(row, f);
                      const isChanged = drafts[key] && f in drafts[key];
                      return (
                        <td key={f} className={isChanged ? "changed-cell" : ""}>
                          <input
                            type="number"
                            step="any"
                            value={val === null || val === undefined ? "" : String(val)}
                            onChange={(e) => onEdit(row, f, e.target.value)}
                            aria-label={`${compactEpic(row.epic)} ${FIELD_LABELS[f]}`}
                          />
                        </td>
                      );
                    })}
                    {boolVisibleFields.map((f) => {
                      const val = getCell(row, f);
                      return (
                        <td key={f} className="trailing-bool-cell">
                          <label className="trailing-mini-toggle">
                            <input
                              type="checkbox"
                              checked={Boolean(val)}
                              onChange={(e) => onEdit(row, f, e.target.checked)}
                              aria-label={`${compactEpic(row.epic)} ${FIELD_LABELS[f]}`}
                            />
                            <span />
                          </label>
                        </td>
                      );
                    })}
                    <td className="trailing-save-cell">
                      <button
                        type="button"
                        disabled={!rowHasChanges || state === "saving"}
                        onClick={() => onSave(row)}
                      >
                        {state === "saving" ? "Saving" : state === "saved" ? "Saved" : rowHasChanges ? "Save" : "Clean"}
                      </button>
                      {strategy !== "DEFAULT" ? (
                        <button
                          className="trailing-reset-button"
                          type="button"
                          disabled={!canReset || reset === "resetting"}
                          onClick={() => onResetOverride(row)}
                        >
                          {!canReset ? "Inherited" : reset === "resetting" ? "Resetting" : "Inherit"}
                        </button>
                      ) : null}
                      {state === "error" && saveError[key] ? (
                        <div className="trailing-save-error">{saveError[key]}</div>
                      ) : null}
                      {reset === "error" && saveError[key] ? (
                        <div className="trailing-save-error">{saveError[key]}</div>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
              {filteredRows.length === 0 ? (
                <tr>
                  <td className="trailing-empty" colSpan={visibleFields.length + 2}>
                    No trailing configs match the current filters.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
