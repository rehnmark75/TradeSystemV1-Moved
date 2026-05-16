import { useCallback, useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

export interface DailyPnlGateConfig {
  environment: string;
  is_enabled: boolean;
  profit_limit_sek: number;
  loss_limit_sek: number;
  updated_at: string;
}

export interface DailyPnlGateBlock {
  id: number;
  blocked_at: string;
  environment: string;
  limit_hit: "profit" | "loss";
  daily_pnl_sek: number;
  profit_limit_sek: number;
  loss_limit_sek: number;
  epic: string | null;
  direction: string | null;
  alert_id: number | null;
  trigger_source: string | null;
}

export function useDailyPnlGate(environment: string) {
  const [config, setConfig] = useState<DailyPnlGateConfig | null>(null);
  const [blocks, setBlocks] = useState<DailyPnlGateBlock[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [cfgRes, blkRes] = await Promise.all([
        fetch(apiUrl(`/api/settings/daily-pnl-gate?environment=${environment}`)),
        fetch(apiUrl(`/api/settings/daily-pnl-gate/blocks?environment=${environment}&days=30`)),
      ]);
      if (!cfgRes.ok) throw new Error("Failed to load config");
      const cfg = await cfgRes.json();
      setConfig(cfg);
      if (blkRes.ok) {
        const blk = await blkRes.json();
        setBlocks(blk.rows ?? []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [environment]);

  useEffect(() => { load(); }, [load]);

  const save = useCallback(async (updates: Omit<DailyPnlGateConfig, "updated_at">) => {
    setSaving(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/settings/daily-pnl-gate"), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? "Save failed");
      }
      const updated = await res.json();
      setConfig(updated);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      return false;
    } finally {
      setSaving(false);
    }
  }, []);

  return { config, blocks, loading, saving, error, reload: load, save };
}
