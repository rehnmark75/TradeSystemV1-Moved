"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ContainerLogsDrawer from "../../../components/ops/ContainerLogsDrawer";

const BASE = "/trading";

type SettingValue = boolean | number;

interface AutoTraderSetting {
  key: string;
  value: SettingValue;
  raw_value: string;
  value_type: "bool" | "int" | "float";
  label: string;
  description: string;
  updated_at?: string;
  min?: number;
  max?: number;
}

interface AutoTradeRun {
  id: number;
  trade_date: string;
  status: string;
  enabled: boolean;
  dry_run: boolean;
  started_at?: string;
  validated_at?: string;
  traded_at?: string;
  completed_at?: string;
  updated_at?: string;
}

interface AutoTradeCandidate {
  id: number;
  rank?: number;
  ticker: string;
  status: string;
  candidate_score?: number;
  order_bias?: string;
  pm_status?: string;
  pm_direction?: string;
  broker_ask?: number;
  broker_spread_pct?: number;
  broker_quote_age_minutes?: number;
  relative_volume?: number;
  intraday_relative_volume?: number;
  planned_entry?: number;
  planned_stop_loss?: number;
  planned_take_profit?: number;
  planned_quantity?: number;
  robomarkets_order_id?: string;
  stock_order_id?: number;
  reason?: string;
  updated_at?: string;
}

interface ContainerStatus {
  name: string;
  state?: string;
  status?: string;
  health?: string;
}

interface Payload {
  settings: AutoTraderSetting[];
  latest_run: AutoTradeRun | null;
  candidates: AutoTradeCandidate[];
  active_count: number;
  containers: ContainerStatus[];
}

function fmtDate(value?: string) {
  if (!value) return "Never";
  return new Date(value).toLocaleString("sv-SE");
}

function fmtNumber(value?: number, digits = 2) {
  if (value === undefined || value === null || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function statusColor(status?: string) {
  const normalized = String(status ?? "").toLowerCase();
  if (["order_submitted", "monitoring", "running", "healthy", "active", "watching"].includes(normalized)) return "var(--good)";
  if (["rejected", "failed", "error", "unhealthy", "exited"].includes(normalized)) return "var(--bad)";
  if (["skipped", "dry_run", "window_closed", "completed"].includes(normalized)) return "var(--warn)";
  return "var(--muted)";
}

export default function AutoTraderSettingsPage() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [values, setValues] = useState<Record<string, SettingValue>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logsFor, setLogsFor] = useState<string | null>(null);
  const [restarting, setRestarting] = useState<string | null>(null);

  async function load() {
    try {
      const res = await fetch(`${BASE}/api/settings/auto-trader`, { cache: "no-store" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Failed to load auto trader settings");
      setPayload(data);
      setValues(Object.fromEntries((data.settings ?? []).map((setting: AutoTraderSetting) => [setting.key, setting.value])));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const timer = setInterval(load, 15000);
    return () => clearInterval(timer);
  }, []);

  const dirty = useMemo(() => {
    if (!payload) return false;
    return payload.settings.some((setting) => values[setting.key] !== setting.value);
  }, [payload, values]);

  async function save() {
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const res = await fetch(`${BASE}/api/settings/auto-trader`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings: values }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Failed to save settings");
      setPayload(data);
      setValues(Object.fromEntries((data.settings ?? []).map((setting: AutoTraderSetting) => [setting.key, setting.value])));
      setMessage("Saved. The worker will use these values on the next cycle.");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function restartContainer(name: string) {
    setRestarting(name);
    setMessage(null);
    try {
      const res = await fetch(`${BASE}/api/infra/containers/${encodeURIComponent(name)}/restart`, { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `Failed to restart ${name}`);
      setMessage(`${name} restart requested.`);
      setTimeout(load, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRestarting(null);
    }
  }

  const enabled = values.AUTO_TRADING_ENABLED === true;
  const dryRun = values.AUTO_TRADING_DRY_RUN === true;
  const latestRun = payload?.latest_run;
  const candidates = payload?.candidates ?? [];

  return (
    <div className="settings-page">
      <div className="settings-page-header">
        <div className="mission-kicker">Execution Automation</div>
        <h1>Stock Auto Trader</h1>
        <p>
          Control the open-window day-trade automation, review the latest run, and inspect service logs from one
          operator surface.
        </p>
      </div>

      <div className="settings-toolbar" style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 18 }}>
        <Link href="/settings" className="settings-card-link">Back to Settings</Link>
        <button className="refresh-btn" onClick={load} title="Refresh">Refresh</button>
        <button className="settings-save-btn" onClick={save} disabled={!dirty || saving}>
          {saving ? "Saving..." : "Save Changes"}
        </button>
        {message && <span className="settings-meta">{message}</span>}
        {error && <span className="settings-error" style={{ margin: 0 }}>{error}</span>}
      </div>

      {loading && <div className="settings-loading">Loading...</div>}

      {!loading && payload && (
        <>
          <div className="settings-overview-strip">
            <div className="settings-overview-stat">
              <span>Automation</span>
              <strong style={{ color: enabled ? "var(--good)" : "var(--bad)" }}>{enabled ? "Enabled" : "Disabled"}</strong>
            </div>
            <div className="settings-overview-stat">
              <span>Mode</span>
              <strong style={{ color: dryRun ? "var(--warn)" : "var(--good)" }}>{dryRun ? "Dry Run" : "Live"}</strong>
            </div>
            <div className="settings-overview-stat">
              <span>Active Orders</span>
              <strong>{payload.active_count}</strong>
            </div>
            <div className="settings-overview-stat">
              <span>Latest Run</span>
              <strong>{latestRun?.status ?? "No run"}</strong>
            </div>
          </div>

          <section className="settings-section">
            <div className="settings-section-header">
              <h2>Runtime Settings</h2>
              <span className="settings-meta">Applies on the next auto-trader cycle</span>
            </div>

            <div className="settings-field-group">
              {payload.settings.map((setting) => (
                <div className="settings-field" key={setting.key}>
                  <label htmlFor={setting.key}>
                    <span className="settings-field-label">{setting.label}</span>
                    <span className="settings-field-desc">{setting.description}</span>
                  </label>
                  {setting.value_type === "bool" ? (
                    <button
                      type="button"
                      className={`toggle-btn${values[setting.key] ? " active" : ""}`}
                      onClick={() => setValues((current) => ({ ...current, [setting.key]: !current[setting.key] }))}
                    >
                      {values[setting.key] ? "ON" : "OFF"}
                    </button>
                  ) : (
                    <input
                      id={setting.key}
                      className="settings-input"
                      type="number"
                      min={setting.min}
                      max={setting.max}
                      step={setting.value_type === "int" ? 1 : 0.1}
                      value={String(values[setting.key] ?? "")}
                      onChange={(event) => {
                        const raw = event.target.value;
                        const next = setting.value_type === "int" ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
                        setValues((current) => ({ ...current, [setting.key]: Number.isFinite(next) ? next : 0 }));
                      }}
                    />
                  )}
                </div>
              ))}
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-header">
              <h2>Services</h2>
              <span className="settings-meta">Logs open in a side drawer</span>
            </div>
            <div className="settings-dashboard-grid">
              {payload.containers.map((container) => (
                <div className="settings-card" key={container.name}>
                  <div className="settings-card-kicker">{container.name}</div>
                  <h3 style={{ color: statusColor(container.state) }}>{container.state ?? "unknown"}</h3>
                  <p>{container.status ?? container.health ?? "Status unavailable"}</p>
                  <div className="settings-inline-links">
                    <button className="settings-card-link" onClick={() => setLogsFor(container.name)}>View Logs</button>
                    <button
                      className="settings-card-link"
                      onClick={() => restartContainer(container.name)}
                      disabled={restarting === container.name}
                    >
                      {restarting === container.name ? "Restarting..." : "Restart"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-header">
              <h2>Latest Run</h2>
              <span className="settings-meta">Updated {fmtDate(latestRun?.updated_at)}</span>
            </div>
            <div className="settings-dashboard-grid">
              <div className="settings-card">
                <div className="settings-card-kicker">Run State</div>
                <h3>{latestRun?.trade_date ? new Date(latestRun.trade_date).toLocaleDateString("sv-SE") : "No run yet"}</h3>
                <p>Status: {latestRun?.status ?? "-"}</p>
                <p>Validated: {fmtDate(latestRun?.validated_at)}</p>
                <p>Traded: {fmtDate(latestRun?.traded_at)}</p>
              </div>
              <div className="settings-card">
                <div className="settings-card-kicker">Candidate Counts</div>
                <h3>{candidates.length}</h3>
                <p>Watching: {candidates.filter((row) => row.status === "watching").length}</p>
                <p>Submitted: {candidates.filter((row) => row.status === "order_submitted").length}</p>
                <p>Rejected or failed: {candidates.filter((row) => ["rejected", "failed"].includes(row.status)).length}</p>
              </div>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-header">
              <h2>Latest Candidates</h2>
              <span className="settings-meta">{candidates.length} rows</span>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="settings-table" style={{ minWidth: 1080 }}>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Ticker</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>PM</th>
                    <th>Bias</th>
                    <th>Spread</th>
                    <th>Quote Age</th>
                    <th>Live RVOL</th>
                    <th>Prior RVOL</th>
                    <th>Entry</th>
                    <th>SL</th>
                    <th>TP</th>
                    <th>Qty</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.map((row) => (
                    <tr key={row.id}>
                      <td>{row.rank ?? "-"}</td>
                      <td><strong>{row.ticker}</strong></td>
                      <td style={{ color: statusColor(row.status), fontWeight: 700 }}>{row.status}</td>
                      <td>{fmtNumber(row.candidate_score, 1)}</td>
                      <td>{row.pm_status ?? row.pm_direction ?? "-"}</td>
                      <td>{row.order_bias ?? "-"}</td>
                      <td>{fmtNumber(row.broker_spread_pct, 3)}%</td>
                      <td>{row.broker_quote_age_minutes != null ? `${row.broker_quote_age_minutes}m` : "-"}</td>
                      <td>{fmtNumber(row.intraday_relative_volume, 2)}</td>
                      <td>{fmtNumber(row.relative_volume, 2)}</td>
                      <td>{fmtNumber(row.planned_entry, 2)}</td>
                      <td>{fmtNumber(row.planned_stop_loss, 2)}</td>
                      <td>{fmtNumber(row.planned_take_profit, 2)}</td>
                      <td>{row.planned_quantity ?? "-"}</td>
                      <td style={{ maxWidth: 280, whiteSpace: "normal" }}>{row.reason ?? "-"}</td>
                    </tr>
                  ))}
                  {candidates.length === 0 && (
                    <tr>
                      <td colSpan={15} style={{ color: "var(--muted)", padding: 18 }}>No auto-trader candidates have been recorded yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      <ContainerLogsDrawer containerName={logsFor} onClose={() => setLogsFor(null)} />
    </div>
  );
}
