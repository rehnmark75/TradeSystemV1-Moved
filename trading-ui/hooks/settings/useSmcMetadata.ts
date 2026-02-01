import { useEffect, useState } from "react";
import { apiUrl } from "../../lib/settings/api";
import type { SmcParameterMetadata } from "../../types/settings";

export function useSmcMetadata() {
  const [metadata, setMetadata] = useState<SmcParameterMetadata[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetch(apiUrl("/api/settings/strategy/smc/metadata"))
      .then(async (res) => {
        if (!res.ok) {
          const payload = await res.json().catch(() => null);
          throw new Error(payload?.error || "Failed to load metadata");
        }
        return res.json();
      })
      .then((payload) => {
        if (!mounted) return;
        setMetadata(Array.isArray(payload) ? payload : []);
        setError(null);
      })
      .catch((err) => {
        if (!mounted) return;
        setMetadata([]);
        setError(err?.message ?? "Failed to load metadata");
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  return { metadata, loading, error };
}
