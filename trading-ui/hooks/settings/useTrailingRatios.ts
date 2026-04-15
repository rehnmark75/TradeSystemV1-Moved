import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

export interface TrailingRatioRow {
  id: number;
  config_set: string;
  epic: string;
  is_scalp: boolean;
  is_active: boolean;

  early_be_trigger_ratio: number | null;
  stage1_trigger_ratio: number | null;
  stage2_trigger_ratio: number | null;
  stage3_trigger_ratio: number | null;
  break_even_trigger_ratio: number | null;
  partial_close_trigger_ratio: number | null;
  stage1_lock_ratio: number | null;
  stage2_lock_ratio: number | null;
  early_be_buffer_points: number | null;
  stage3_atr_multiplier: number | null;
  stage3_min_distance_ratio: number | null;
  min_trail_distance_ratio: number | null;
  min_early_be_trigger: number | null;
  min_stage1_trigger: number | null;
  min_stage1_lock: number | null;
  min_stage2_trigger: number | null;
  min_stage2_lock: number | null;
  min_stage3_trigger: number | null;
  min_break_even_trigger: number | null;
  min_trail_distance: number | null;

  updated_by: string | null;
  change_reason: string | null;
  updated_at: string;
}

export function useTrailingRatios(configSet: string, isScalp: boolean) {
  const [rows, setRows] = useState<TrailingRatioRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = apiUrl(
        `/api/settings/trailing-ratios?config_set=${encodeURIComponent(configSet)}&is_scalp=${isScalp}`
      );
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Failed to load trailing ratios");
      setRows(payload.rows ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load trailing ratios");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [configSet, isScalp]);

  const saveRow = async (
    epic: string,
    updates: Record<string, unknown>,
    meta: { updatedBy: string; changeReason: string; updatedAt: string }
  ) => {
    const response = await fetch(
      apiUrl(`/api/settings/trailing-ratios/${encodeURIComponent(epic)}`),
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          updates,
          updated_by: meta.updatedBy,
          change_reason: meta.changeReason,
          updated_at: meta.updatedAt,
          config_set: configSet,
          is_scalp: isScalp,
        }),
      }
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? payload.message ?? "Failed to save");
    }
    await reload();
    return payload;
  };

  const createRow = async (
    epic: string,
    updates: Record<string, unknown>,
    meta: { updatedBy: string; changeReason: string }
  ) => {
    const response = await fetch(apiUrl("/api/settings/trailing-ratios"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epic,
        is_scalp: isScalp,
        updates,
        updated_by: meta.updatedBy,
        change_reason: meta.changeReason,
        config_set: configSet,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? "Failed to create");
    await reload();
    return payload;
  };

  const deleteRow = async (
    epic: string,
    meta: { updatedBy: string; changeReason: string }
  ) => {
    const response = await fetch(
      apiUrl(`/api/settings/trailing-ratios/${encodeURIComponent(epic)}`),
      {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          updated_by: meta.updatedBy,
          change_reason: meta.changeReason,
          config_set: configSet,
          is_scalp: isScalp,
        }),
      }
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error ?? "Failed to delete");
    await reload();
    return payload;
  };

  return { rows, loading, error, reload, saveRow, createRow, deleteRow };
}
