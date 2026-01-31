import { useEffect, useMemo, useState } from "react";
import { apiUrl } from "../../lib/settings/api";

interface UseSettingsOptions<T> {
  endpoint: string;
  defaultsEndpoint?: string;
  draftKey: string;
  onConflict?: (current: Record<string, unknown>) => void;
  parseDefaults?: (raw: Record<string, string | null>) => Record<string, unknown>;
}

export function useSettings<T>({
  endpoint,
  defaultsEndpoint,
  draftKey,
  onConflict,
  parseDefaults
}: UseSettingsOptions<T>) {
  const [data, setData] = useState<T | null>(null);
  const [defaults, setDefaults] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);
  const [changes, setChanges] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(draftKey);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setChanges(parsed.changes ?? {});
      } catch {
        setChanges({});
      }
    }
  }, [draftKey]);

  useEffect(() => {
    if (!Object.keys(changes).length) return;
    localStorage.setItem(draftKey, JSON.stringify({ changes }));
  }, [changes, draftKey]);

  useEffect(() => {
    let active = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(apiUrl(endpoint));
        const payload = await response.json();
        if (!active) return;
        if (!response.ok) throw new Error(payload.error ?? "Failed to load settings");
        setData(payload);
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Failed to load settings");
        }
      } finally {
        if (active) setLoading(false);
      }
    };
    load();
    return () => {
      active = false;
    };
  }, [endpoint]);

  useEffect(() => {
    if (!defaultsEndpoint) return;
    let active = true;
    const loadDefaults = async () => {
      try {
        const response = await fetch(apiUrl(defaultsEndpoint));
        const payload = await response.json();
        if (!active) return;
        if (!response.ok) return;
        setDefaults(parseDefaults ? parseDefaults(payload) : payload);
      } catch {
        if (active) setDefaults({});
      }
    };
    loadDefaults();
    return () => {
      active = false;
    };
  }, [defaultsEndpoint, parseDefaults]);

  const effectiveData = useMemo(() => {
    if (!data) return null;
    return { ...data, ...changes } as T;
  }, [data, changes]);

  const updateField = (key: string, value: unknown) => {
    setChanges((prev) => ({ ...prev, [key]: value }));
  };

  const resetChanges = () => {
    setChanges({});
    localStorage.removeItem(draftKey);
  };

  const saveChanges = async (
    payload: { updatedBy: string; changeReason: string },
    overrideChanges?: Record<string, unknown>,
    overrideUpdatedAt?: string
  ) => {
    if (!data) return { success: false };
    if (!payload.updatedBy || !payload.changeReason) {
      setError("Updated by and change reason are required.");
      return { success: false };
    }
    const changesToSend = overrideChanges ?? changes;
    if (!Object.keys(changesToSend).length) {
      setError("No changes to save.");
      return { success: false };
    }

    try {
      const response = await fetch(apiUrl(endpoint), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          updates: changesToSend,
          updated_by: payload.updatedBy,
          change_reason: payload.changeReason,
          updated_at: overrideUpdatedAt ?? (data as any).updated_at
        })
      });
      const payloadResponse = await response.json();
      if (response.status === 409 && onConflict) {
        onConflict(payloadResponse.current_config ?? payloadResponse.current_override ?? null);
      }
      if (!response.ok) {
        throw new Error(payloadResponse.error ?? "Failed to save settings");
      }
      setData(payloadResponse);
      resetChanges();
      return { success: true };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
      return { success: false };
    }
  };

  return {
    data,
    defaults,
    effectiveData,
    loading,
    error,
    changes,
    updateField,
    resetChanges,
    saveChanges,
    setData,
    setChanges
  };
}
