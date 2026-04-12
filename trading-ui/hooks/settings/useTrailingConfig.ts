import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

export interface TrailingConfigRow {
  id: number;
  config_set: string;
  epic: string;
  is_scalp: boolean;
  is_active: boolean;
  early_breakeven_trigger_points: number | null;
  early_breakeven_buffer_points: number | null;
  stage1_trigger_points: number | null;
  stage1_lock_points: number | null;
  stage2_trigger_points: number | null;
  stage2_lock_points: number | null;
  stage3_trigger_points: number | null;
  stage3_atr_multiplier: number | null;
  stage3_min_distance: number | null;
  min_trail_distance: number | null;
  break_even_trigger_points: number | null;
  enable_partial_close: boolean | null;
  partial_close_trigger_points: number | null;
  partial_close_size: number | null;
  updated_by: string | null;
  change_reason: string | null;
  updated_at: string;
}

export function useTrailingConfig(configSet: string, isScalp: boolean) {
  const [rows, setRows] = useState<TrailingConfigRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = apiUrl(
        `/api/settings/trailing?config_set=${encodeURIComponent(configSet)}&is_scalp=${isScalp}`
      );
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? "Failed to load trailing configs");
      setRows(payload.rows ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load trailing configs");
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
      apiUrl(`/api/settings/trailing/${encodeURIComponent(epic)}`),
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
    setRows((prev) => prev.map((r) => (r.epic === epic && r.is_scalp === isScalp ? payload : r)));
    return payload as TrailingConfigRow;
  };

  return { rows, loading, error, reload, saveRow };
}
