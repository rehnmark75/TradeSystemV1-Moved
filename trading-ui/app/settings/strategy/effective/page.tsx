"use client";

import { useEffect, useState } from "react";
import { logTelemetry } from "../../../../lib/settings/telemetry";
import { apiUrl } from "../../../../lib/settings/api";

interface EffectivePayload {
  epic: string;
  global: Record<string, unknown>;
  override: Record<string, unknown> | null;
  effective: Record<string, unknown>;
}

export default function EffectiveConfigLanding() {
  const [epic, setEpic] = useState("");
  const [data, setData] = useState<EffectivePayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    logTelemetry({ event_type: "page_view", page: "/settings/strategy/effective" });
  }, []);

  useEffect(() => {
    if (!epic) return;
    const load = async () => {
      try {
        const response = await fetch(
          apiUrl(`/api/settings/strategy/smc/effective/${epic}`)
        );
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error ?? "Failed to load effective config");
        }
        setData(payload);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load effective config");
        setData(null);
      }
    };
    load();
  }, [epic]);

  return (
    <div className="settings-panel">
      <div className="settings-hero">
        <h1>Effective Config</h1>
        <p>Select a pair to compare global vs overrides.</p>
      </div>
      <div className="settings-form-actions">
        <input
          placeholder="Epic (e.g. CS.D.EURUSD.MINI.IP)"
          value={epic}
          onChange={(event) => setEpic(event.target.value)}
        />
      </div>
      {error ? <div className="settings-placeholder">Error: {error}</div> : null}
      {data ? (
        <div className="effective-config-table">
          <div className="effective-row effective-header">
            <span>Field</span>
            <span>Global</span>
            <span>Override</span>
            <span>Effective</span>
          </div>
          {Object.keys(data.effective).map((key) => (
            <div key={key} className="effective-row">
              <span>{key}</span>
              <span>{JSON.stringify((data.global as any)[key])}</span>
              <span>{JSON.stringify((data.override as any)?.[key] ?? null)}</span>
              <span>{JSON.stringify((data.effective as any)[key])}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
