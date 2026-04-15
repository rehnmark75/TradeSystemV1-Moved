"use client";

import { useState, useCallback } from "react";
import { apiUrl } from "../../lib/settings/api";
import type { ConfigSnapshot, SnapshotDiff } from "../../types/settings";

interface SnapshotCompareResult {
  snapshot_name: string;
  snapshot_created_at: string;
  snapshot_values: Record<string, unknown>;
  current_values: Record<string, unknown>;
  diff: SnapshotDiff[];
  changed_count: number;
}

export function useSnapshots(configSet: string) {
  const [snapshots, setSnapshots] = useState<ConfigSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        apiUrl(`/api/settings/strategy/smc/snapshots?config_set=${encodeURIComponent(configSet)}`)
      );
      if (!res.ok) throw new Error("Failed to load snapshots");
      const data = await res.json();
      setSnapshots(data.snapshots ?? []);
    } catch (err: any) {
      setError(err.message ?? "Failed to load snapshots");
    } finally {
      setLoading(false);
    }
  }, [configSet]);

  const createSnapshot = useCallback(
    async (name: string, description?: string, tags?: string[]) => {
      const res = await fetch(apiUrl("/api/settings/strategy/smc/snapshots"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, tags, created_by: "admin", config_set: configSet }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.error ?? "Failed to create snapshot");
      }
      const data = await res.json();
      await load();
      return data.snapshot as ConfigSnapshot;
    },
    [configSet, load]
  );

  const deleteSnapshot = useCallback(
    async (id: number) => {
      const res = await fetch(apiUrl(`/api/settings/strategy/smc/snapshots/${id}`), {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete snapshot");
      await load();
    },
    [load]
  );

  const restoreSnapshot = useCallback(async (id: number) => {
    const res = await fetch(
      apiUrl(`/api/settings/strategy/smc/snapshots/${id}/restore`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ restored_by: "admin" }),
      }
    );
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.error ?? "Failed to restore snapshot");
    }
    return res.json();
  }, []);

  const compareSnapshot = useCallback(
    async (id: number): Promise<SnapshotCompareResult> => {
      const res = await fetch(
        apiUrl(`/api/settings/strategy/smc/snapshots/${id}/compare?config_set=${encodeURIComponent(configSet)}`)
      );
      if (!res.ok) throw new Error("Failed to compare snapshot");
      return res.json();
    },
    [configSet]
  );

  return {
    snapshots,
    loading,
    error,
    load,
    createSnapshot,
    deleteSnapshot,
    restoreSnapshot,
    compareSnapshot,
  };
}
