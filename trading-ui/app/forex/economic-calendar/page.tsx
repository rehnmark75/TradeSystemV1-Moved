/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type CalendarEvent = {
  id: number | string;
  event_name: string;
  currency: string;
  event_date: string;
  impact_level: "high" | "medium" | "low";
  forecast_value?: string | null;
  previous_value?: string | null;
  actual_value?: string | null;
  market_moving?: boolean | null;
  display_time: string;
  is_critical?: boolean;
};

type PairRisk = {
  pair: string;
  currencies: string[];
  state: "blocked" | "caution" | "monitor" | "clear";
  headline: string;
  events_count: number;
  highest_impact: "high" | "medium" | "low";
  risk_score: number;
  time_to_nearest_event: number | null;
  time_to_nearest_high_impact: number | null;
};

type Payload = {
  generated_at: string;
  source: string;
  query: {
    pair: string;
    hours: number;
    impact: "high" | "medium" | "all";
  };
  summary: {
    total_events: number;
    high_impact_events: number;
    medium_impact_events: number;
    market_moving_events: number;
    currencies_affected: number;
    next_event_at: string | null;
  };
  selected_pair: {
    pair: string;
    currencies: string[];
    summary: {
      events_count: number;
      highest_impact: "high" | "medium" | "low";
      risk_score: number;
      nearest_event_at: string | null;
      nearest_event_name: string | null;
      time_to_nearest_event: number | null;
      time_to_nearest_high_impact: number | null;
      critical_events: number;
      affected_currencies: string[];
    };
    events: CalendarEvent[];
  };
  pair_risks: PairRisk[];
  currencies: Array<{ currency: string; count: number }>;
  events: CalendarEvent[];
};

const PAIRS = [
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "USDCHF",
  "AUDUSD",
  "NZDUSD",
  "USDCAD",
  "EURGBP",
  "EURJPY",
  "GBPJPY",
  "AUDJPY",
  "XAUUSD"
];

const HOUR_OPTIONS = [6, 12, 24, 48, 72, 168];
const IMPACT_OPTIONS = [
  { value: "high", label: "High only" },
  { value: "medium", label: "Medium + High" },
  { value: "all", label: "All impacts" }
] as const;

function formatMinutes(value: number | null) {
  if (value == null) return "—";
  if (value < 60) return `${value}m`;
  const hours = Math.floor(value / 60);
  const minutes = value % 60;
  if (!minutes) return `${hours}h`;
  return `${hours}h ${minutes}m`;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

function stateLabel(state: PairRisk["state"]) {
  if (state === "blocked") return "Blocked";
  if (state === "caution") return "Caution";
  if (state === "monitor") return "Monitor";
  return "Clear";
}

function stateClass(state: PairRisk["state"]) {
  if (state === "blocked") return "bad";
  if (state === "caution") return "warn";
  if (state === "monitor") return "neutral";
  return "good";
}

function impactClass(level: CalendarEvent["impact_level"], critical?: boolean) {
  if (critical || level === "high") return "bad";
  if (level === "medium") return "warn";
  return "good";
}

export default function EconomicCalendarPage() {
  const { environment } = useEnvironment();
  const [pair, setPair] = useState("EURUSD");
  const [hours, setHours] = useState(24);
  const [impact, setImpact] = useState<"high" | "medium" | "all">("medium");
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      pair,
      hours: String(hours),
      impact,
      env: environment
    });

    fetch(`/trading/api/forex/economic-calendar?${params.toString()}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.error) throw new Error(data.error);
        setPayload(data);
      })
      .catch((err) => setError(err.message || "Failed to load economic calendar."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, [pair, hours, impact, environment]);

  const topRiskPairs = useMemo(
    () => (payload?.pair_risks ?? []).slice(0, 6),
    [payload?.pair_risks]
  );
  const activeEvents = useMemo(
    () => (payload?.events ?? []).slice(0, 12),
    [payload?.events]
  );
  const selectedSummary = payload?.selected_pair.summary;

  return (
    <div className="page">
      <div className="topbar">
        <Link href="/" className="brand">
          K.L.I.R.R
        </Link>
        <EnvironmentToggle />
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
          <h1>Economic Calendar</h1>
          <p>
            Live macro event feed translated into pair-level trading risk for the forex dashboard.
          </p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/economic-calendar" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Pair</label>
            <select value={pair} onChange={(e) => setPair(e.target.value)}>
              {PAIRS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Window</label>
            <select value={hours} onChange={(e) => setHours(Number(e.target.value))}>
              {HOUR_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  Next {option}h
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Impact</label>
            <select
              value={impact}
              onChange={(e) => setImpact(e.target.value as "high" | "medium" | "all")}
            >
              {IMPACT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <button className="section-tab active" onClick={loadData}>
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Loading economic calendar...</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Upcoming Events
                <strong>{payload?.summary.total_events ?? 0}</strong>
              </div>
              <div className="summary-card">
                High Impact
                <strong>{payload?.summary.high_impact_events ?? 0}</strong>
              </div>
              <div className="summary-card">
                Market Moving
                <strong>{payload?.summary.market_moving_events ?? 0}</strong>
              </div>
              <div className="summary-card">
                Selected Pair Risk
                <strong>{formatPercent(selectedSummary?.risk_score ?? 0)}</strong>
              </div>
              <div className="summary-card">
                Next High Impact
                <strong>{formatMinutes(selectedSummary?.time_to_nearest_high_impact ?? null)}</strong>
              </div>
            </div>

            <div
              className="panel"
              style={{
                marginTop: "1rem",
                border: "1px solid rgba(255,255,255,0.06)",
                background: "rgba(255,255,255,0.02)"
              }}
            >
              <div className="chart-title">Selected Pair Decision Context</div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
                  gap: "0.85rem"
                }}
              >
                <div className="summary-card">
                  Pair
                  <strong>{payload?.selected_pair.pair ?? pair}</strong>
                </div>
                <div className="summary-card">
                  Relevant Events
                  <strong>{selectedSummary?.events_count ?? 0}</strong>
                </div>
                <div className="summary-card">
                  Highest Impact
                  <strong>{selectedSummary?.highest_impact ?? "low"}</strong>
                </div>
                <div className="summary-card">
                  Nearest Event
                  <strong>{formatMinutes(selectedSummary?.time_to_nearest_event ?? null)}</strong>
                </div>
                <div className="summary-card">
                  Critical Events
                  <strong>{selectedSummary?.critical_events ?? 0}</strong>
                </div>
              </div>

              <div
                style={{
                  marginTop: "1rem",
                  padding: "0.9rem 1rem",
                  borderRadius: 8,
                  background: "rgba(15, 23, 42, 0.45)",
                  border: "1px solid rgba(255,255,255,0.05)"
                }}
              >
                <div className="chart-title" style={{ marginBottom: "0.35rem" }}>
                  Trading Read
                </div>
                <div style={{ color: "var(--ink)", fontWeight: 600 }}>
                  {selectedSummary?.nearest_event_name
                    ? `${payload?.selected_pair.pair}: ${selectedSummary.nearest_event_name}`
                    : `${payload?.selected_pair.pair ?? pair}: no upcoming relevant event in the selected window`}
                </div>
                <div style={{ color: "var(--muted)", marginTop: "0.3rem", lineHeight: 1.55 }}>
                  Use this as a decision surface, not just a feed. If the nearest high-impact event is inside the scanner
                  buffer, treat the pair as execution-hostile; otherwise use the risk score to decide whether to size down or wait.
                </div>
              </div>
            </div>

            <div className="panel table-panel" style={{ marginTop: "1rem" }}>
              <div className="chart-title">Pair Risk Radar</div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
                  gap: "0.9rem"
                }}
              >
                {topRiskPairs.map((row) => (
                  <div
                    key={row.pair}
                    style={{
                      borderRadius: 10,
                      padding: "0.95rem",
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid rgba(255,255,255,0.06)"
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: "0.8rem"
                      }}
                    >
                      <strong>{row.pair}</strong>
                      <span className={stateClass(row.state)}>{stateLabel(row.state)}</span>
                    </div>
                    <div style={{ color: "var(--muted)", marginTop: "0.45rem", minHeight: 42 }}>
                      {row.headline}
                    </div>
                    <div
                      style={{
                        marginTop: "0.75rem",
                        display: "grid",
                        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                        gap: "0.5rem"
                      }}
                    >
                      <div className="summary-card">
                        Risk
                        <strong>{formatPercent(row.risk_score)}</strong>
                      </div>
                      <div className="summary-card">
                        Events
                        <strong>{row.events_count}</strong>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1.25fr 0.75fr",
                gap: "1rem",
                marginTop: "1rem"
              }}
            >
              <div className="panel table-panel">
                <div className="chart-title">Live Event Feed</div>
                {activeEvents.length ? (
                  activeEvents.map((event) => (
                    <div
                      key={`${event.id}-${event.event_date}`}
                      style={{
                        padding: "0.85rem 0",
                        borderBottom: "1px solid rgba(255,255,255,0.06)"
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          gap: "1rem"
                        }}
                      >
                        <div>
                          <div style={{ fontWeight: 600, color: "var(--ink)" }}>
                            {event.currency} {event.event_name}
                          </div>
                          <div style={{ color: "var(--muted)", marginTop: "0.2rem" }}>
                            {event.display_time}
                          </div>
                        </div>
                        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                          <span className={impactClass(event.impact_level, event.is_critical)}>
                            {event.is_critical ? "critical" : event.impact_level}
                          </span>
                          {event.market_moving ? <span className="warn">moving</span> : null}
                        </div>
                      </div>
                      {(event.previous_value || event.forecast_value || event.actual_value) && (
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                            gap: "0.6rem",
                            marginTop: "0.7rem",
                            color: "var(--muted)",
                            fontSize: "0.88rem"
                          }}
                        >
                          <div>Prev: {event.previous_value || "—"}</div>
                          <div>Fcst: {event.forecast_value || "—"}</div>
                          <div>Act: {event.actual_value || "—"}</div>
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="chart-placeholder">No events found for the selected filters.</div>
                )}
              </div>

              <div className="panel table-panel">
                <div className="chart-title">Currency Pressure</div>
                {(payload?.currencies ?? []).length ? (
                  payload?.currencies.map((row) => (
                    <div
                      key={row.currency}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "0.7rem 0",
                        borderBottom: "1px solid rgba(255,255,255,0.06)"
                      }}
                    >
                      <span style={{ fontWeight: 600 }}>{row.currency}</span>
                      <span className="neutral">{row.count} events</span>
                    </div>
                  ))
                ) : (
                  <div className="chart-placeholder">No currency exposure in the current window.</div>
                )}

                <div
                  style={{
                    marginTop: "1rem",
                    paddingTop: "1rem",
                    borderTop: "1px solid rgba(255,255,255,0.06)",
                    color: "var(--muted)",
                    lineHeight: 1.6
                  }}
                >
                  This page is most useful when read alongside Alert History and Validator Rejections.
                  It turns the raw calendar into pair risk so the desk can see which symbols are becoming hostile before the scanner rejects them.
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
