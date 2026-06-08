import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

export const TRAILING_STRATEGIES = [
  "DEFAULT",
  "SMC_SIMPLE",
  "SMC_SIMPLE_V2",
  "XAU_GOLD",
  "RANGE_FADE",
  "MEAN_REVERSION",
] as const;

export type TrailingStrategy = (typeof TRAILING_STRATEGIES)[number];

export interface TrailingConfigRow {
  id: number | null;
  strategy: TrailingStrategy;
  config_set: string;
  epic: string;
  is_scalp: boolean;
  is_active: boolean;
  inherited?: boolean;
  override_field_count?: number;
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
  early_failure_stop_enabled: boolean | null;
  early_failure_check_bars: number | null;
  early_failure_min_mfe_pips: number | null;
  early_failure_stop_pips: number | null;
  enable_partial_close: boolean | null;
  partial_close_trigger_points: number | null;
  partial_close_size: number | null;
  updated_by: string | null;
  change_reason: string | null;
  updated_at: string;
}

export function useTrailingConfig(
  configSet: string,
  isScalp: boolean,
  strategy: TrailingStrategy = "DEFAULT"
) {
  const [rows, setRows] = useState<TrailingConfigRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = apiUrl(
        `/api/settings/trailing?config_set=${encodeURIComponent(configSet)}` +
          `&is_scalp=${isScalp}&strategy=${encodeURIComponent(strategy)}`
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
  }, [configSet, isScalp, strategy]);

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
          strategy,
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
    return payload as TrailingConfigRow;
  };

  const resetOverride = async (
    epic: string,
    meta: { updatedBy: string; changeReason: string }
  ) => {
    const response = await fetch(
      apiUrl(`/api/settings/trailing/${encodeURIComponent(epic)}`),
      {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy,
          updated_by: meta.updatedBy,
          change_reason: meta.changeReason,
          config_set: configSet,
          is_scalp: isScalp,
        }),
      }
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? payload.message ?? "Failed to reset override");
    }
    await reload();
    return payload;
  };

  return { rows, loading, error, reload, saveRow, resetOverride };
}
