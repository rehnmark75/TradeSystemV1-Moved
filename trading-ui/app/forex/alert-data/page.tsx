"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";

type AlertDataResponse = {
  hours: number;
  total: number;
  alerts: Record<string, unknown>[];
  error?: string;
};

const TIME_SLOTS = [
  { label: "Last 1 hour", value: 1 },
  { label: "Last 3 hours", value: 3 },
  { label: "Last 8 hours", value: 8 },
  { label: "Last 24 hours", value: 24 },
  { label: "Last 3 days", value: 72 },
  { label: "Last 7 days", value: 168 }
];

function getAlertSummary(alert: Record<string, unknown>) {
  const id = alert.id ?? "?";
  const timestamp = alert.alert_timestamp ?? "";
  const pair = alert.pair ?? alert.epic ?? "";
  const signalType = alert.signal_type ?? "";
  const strategy = alert.strategy ?? "";

  return `#${id} ${pair} ${signalType} ${strategy} ${timestamp}`.trim();
}

export default function AlertDataPage() {
  const [hours, setHours] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [payload, setPayload] = useState<AlertDataResponse | null>(null);
  const [refreshTick, setRefreshTick] = useState(0);

  const query = useMemo(() => new URLSearchParams({ hours: String(hours) }).toString(), [hours]);

  useEffect(() => {
    let isActive = true;

    async function load() {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/trading/api/forex/alert-data?${query}`);
        const data = (await response.json()) as AlertDataResponse;

        if (!response.ok) {
          throw new Error(data.error ?? "Failed to load alert data");
        }

        if (isActive) {
          setPayload(data);
        }
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : "Failed to load alert data");
        }
      } finally {
        if (isActive) {
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      isActive = false;
    };
  }, [query, refreshTick]);

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          Trading Hub
        </Link>
        <div className="nav-links">
          <Link href="/watchlists">Watchlists</Link>
          <Link href="/signals">Signals</Link>
          <Link href="/broker">Broker</Link>
          <Link href="/market">Market</Link>
          <Link href="/forex">Forex Analytics</Link>
          <Link href="/settings">Settings</Link>
        </div>
      </div>

      <div className="header">
        <div>
          <h1>Alert Data</h1>
          <p>Raw alert_history rows for a selected time slot.</p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/alert-data" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Time Slot</label>
            <select value={hours} onChange={(event) => setHours(Number(event.target.value))}>
              {TIME_SLOTS.map((slot) => (
                <option key={slot.value} value={slot.value}>
                  {slot.label}
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={() => setRefreshTick((t) => t + 1)}>
            Refresh
          </button>
          <div className="forex-badge">{payload?.total ?? 0} alerts</div>
        </div>

        {loading ? <div className="chart-placeholder">Loading alert data...</div> : null}
        {error ? <div className="error">{error}</div> : null}

        {!loading && !error && payload?.alerts?.length ? (
          <div>
            {payload.alerts.map((alert, index) => (
              <details key={String(alert.id ?? index)} className="panel">
                <summary>{getAlertSummary(alert)}</summary>
                <pre>{JSON.stringify(alert, null, 2)}</pre>
              </details>
            ))}
          </div>
        ) : null}

        {!loading && !error && payload && payload.alerts.length === 0 ? (
          <div className="chart-placeholder">No alerts found for this time slot.</div>
        ) : null}
      </div>
    </div>
  );
}
