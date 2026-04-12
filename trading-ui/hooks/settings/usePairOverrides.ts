import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

interface PairOverride {
  id: number;
  epic: string;
  updated_at: string;
  parameter_overrides?: Record<string, unknown>;
  [key: string]: unknown;
}

export function usePairOverrides(configSet?: string) {
  const effectiveConfigSet = configSet ?? "demo";
  const [overrides, setOverrides] = useState<PairOverride[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadOverrides = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        apiUrl(`/api/settings/strategy/smc/pairs?config_set=${encodeURIComponent(effectiveConfigSet)}`)
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error ?? "Failed to load overrides");
      }
      setOverrides(payload.overrides ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load overrides");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOverrides();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveConfigSet]);

  const saveOverride = async (epic: string, updates: Record<string, unknown>, meta: {
    updatedBy: string;
    changeReason: string;
    updatedAt: string;
  }) => {
    const response = await fetch(
      apiUrl(`/api/settings/strategy/smc/pairs/${epic}?config_set=${encodeURIComponent(effectiveConfigSet)}`),
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          updates,
          updated_by: meta.updatedBy,
          change_reason: meta.changeReason,
          updated_at: meta.updatedAt,
          config_set: effectiveConfigSet
        })
      }
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? "Failed to update override");
    }
    setOverrides((prev) =>
      prev.map((item) => (item.epic === epic ? payload : item))
    );
    return payload;
  };

  const createOverride = async (
    epic: string,
    updates: Record<string, unknown>,
    meta: {
      updatedBy: string;
      changeReason: string;
    }
  ) => {
    const response = await fetch(apiUrl("/api/settings/strategy/smc/pairs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epic,
        updates,
        updated_by: meta.updatedBy,
        change_reason: meta.changeReason,
        config_set: effectiveConfigSet
      })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? "Failed to create override");
    }
    setOverrides((prev) => [...prev.filter((item) => item.epic !== epic), payload]);
    return payload;
  };

  const deleteOverride = async (epic: string, meta: { updatedBy: string; changeReason: string }) => {
    const response = await fetch(
      apiUrl(`/api/settings/strategy/smc/pairs/${epic}`),
      {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        updated_by: meta.updatedBy,
        change_reason: meta.changeReason,
        config_set: effectiveConfigSet
      })
      }
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? "Failed to delete override");
    }
    setOverrides((prev) => prev.filter((item) => item.epic !== epic));
    return payload;
  };

  const bulkAction = async (action: string, epics: string[], meta: {
    updatedBy: string;
    changeReason: string;
    sourceEpic?: string;
  }) => {
    const response = await fetch(apiUrl("/api/settings/strategy/smc/bulk"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action,
        epics,
        source_epic: meta.sourceEpic,
        updated_by: meta.updatedBy,
        change_reason: meta.changeReason,
        config_set: effectiveConfigSet
      })
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? "Failed to apply bulk action");
    }
    await loadOverrides();
    return payload;
  };

  return {
    overrides,
    loading,
    error,
    reload: loadOverrides,
    saveOverride,
    createOverride,
    deleteOverride,
    bulkAction
  };
}
