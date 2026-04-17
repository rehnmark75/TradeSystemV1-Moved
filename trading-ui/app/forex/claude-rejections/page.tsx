/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type StopPathPoint = {
  minutes: number;
  stop_pips_from_entry: number;
  label: string;
};

type RejectionRow = {
  id: number;
  alert_timestamp: string;
  epic: string;
  pair: string | null;
  signal_type: string;
  price: number;
  confidence_score: number | null;
  claude_score: number | null;
  claude_reason: string | null;
  market_session: string | null;
  market_regime: string | null;
  environment: string | null;
  sl_pips: number;
  tp_pips: number;
  trailing_used: boolean;
  candle_resolution: "1m" | "5m" | "none";
  outcome: "WIN" | "LOSS" | "BREAKEVEN" | "TRAILED" | "TRAILED_FORCED" | "STILL_OPEN" | "NO_DATA";
  pips: number;
  minutes_to_resolve: number | null;
  max_favorable_pips: number;
  max_adverse_pips: number;
  exit_reason: string;
  stop_path: StopPathPoint[];
  locked_pips: number;
  alert_age_hours: number;
};

type PairAggregate = {
  epic: string;
  n: number;
  wins: number;
  losses: number;
  trailed: number;
  breakevens: number;
  net_pips: number;
  win_rate: number;
  verdict: string;
};

type Payload = {
  params: {
    window_hours: number;
    env: string;
    epic: string;
    trailing: boolean;
    timeout_hours: number;
  };
  filters: {
    epics: string[];
    trailing_available: string[];
  };
  stats: {
    total: number;
    resolved: number;
    wins: number;
    losses: number;
    trailed: number;
    breakevens: number;
    still_open: number;
    no_data: number;
    net_pips: number;
    gross_win: number;
    gross_loss: number;
    profit_factor: number | null;
    win_rate: number;
    verdict: "CLAUDE_RIGHT" | "CLAUDE_WRONG" | "NEUTRAL" | "INSUFFICIENT_DATA";
  };
  by_pair: PairAggregate[];
  rejections: RejectionRow[];
};

const WINDOW_PRESETS = [
  { label: "6h", hours: 6 },
  { label: "24h", hours: 24 },
  { label: "3d", hours: 72 },
  { label: "7d", hours: 168 },
  { label: "30d", hours: 720 },
  { label: "90d", hours: 2160 },
];

const formatDateTime = (value: string) => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return value;
  return parsed.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const shortEpic = (epic: string) => {
  const parts = epic.split(".");
  return parts.length >= 3 ? parts[2] : epic;
};

const signed = (n: number, digits = 1) => `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;

const outcomeClass = (outcome: RejectionRow["outcome"]) => {
  switch (outcome) {
    case "WIN":
      return "rej-outcome rej-outcome-bad";
    case "LOSS":
      return "rej-outcome rej-outcome-good";
    case "TRAILED":
    case "TRAILED_FORCED":
      return "rej-outcome rej-outcome-mixed";
    case "BREAKEVEN":
      return "rej-outcome rej-outcome-neutral";
    case "STILL_OPEN":
      return "rej-outcome rej-outcome-open";
    default:
      return "rej-outcome";
  }
};

// All outcomes are COUNTERFACTUAL — these signals were rejected, no trade existed.
// Labels describe what would have happened if we'd taken the trade anyway.
const outcomeLabel = (outcome: RejectionRow["outcome"]) => {
  switch (outcome) {
    case "WIN":
      return "WOULD HAVE WON";
    case "LOSS":
      return "WOULD HAVE LOST";
    case "TRAILED":
      return "WOULD HAVE TRAILED";
    case "TRAILED_FORCED":
      return "WOULD HAVE TRAILED (FORCED)";
    case "BREAKEVEN":
      return "WOULD HAVE BE'D";
    case "STILL_OPEN":
      return "UNRESOLVED";
    case "NO_DATA":
      return "NO DATA";
    default:
      return outcome;
  }
};

const verdictLabel = (verdict: Payload["stats"]["verdict"]) => {
  switch (verdict) {
    case "CLAUDE_RIGHT":
      return "✅ Rejecting saved pips (counterfactual)";
    case "CLAUDE_WRONG":
      return "❌ Rejecting cost pips (counterfactual)";
    case "NEUTRAL":
      return "⚖️ Rejections were net neutral";
    default:
      return "— insufficient data";
  }
};

const verdictTone = (verdict: Payload["stats"]["verdict"]) => {
  if (verdict === "CLAUDE_RIGHT") return "good";
  if (verdict === "CLAUDE_WRONG") return "bad";
  return "warn";
};

export default function ClaudeRejectionsPage() {
  const { environment } = useEnvironment();
  const [windowHours, setWindowHours] = useState(24);
  const [epic, setEpic] = useState("All");
  const [trailing, setTrailing] = useState(true);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  const load = () => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      window: String(windowHours),
      env: environment,
      trailing: String(trailing),
    });
    if (epic !== "All") params.set("epic", epic);
    fetch(`/trading/api/forex/claude-rejections/?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
      })
      .then((data) => {
        setPayload(data);
        setExpanded({});
      })
      .catch(() => setError("Failed to load Claude rejection analysis."))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, [windowHours, epic, trailing, environment]);

  const stats = payload?.stats;
  const rejections = payload?.rejections ?? [];
  const byPair = payload?.by_pair ?? [];
  const epicOptions = payload?.filters?.epics ?? ["All"];
  const trailingPairs = useMemo(
    () => new Set(payload?.filters?.trailing_available ?? []),
    [payload]
  );

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
          <h1>Claude Rejection Validation</h1>
          <p>
            <strong>Counterfactual analysis only.</strong> These signals were rejected — no
            trade was ever placed. We replay each rejected alert through the candle stream
            using the per-pair SMC fixed SL/TP and (optionally) the progressive trailing
            stop stages to answer: <em>if we had taken the trade anyway, what would have
            happened?</em> &ldquo;UNRESOLVED&rdquo; means a hypothetical trade would still
            be running if it had been placed; it is not a live position.
          </p>
        </div>
        <div className="header-chip">Forex</div>
      </div>

      <ForexNav activeHref="/forex/claude-rejections" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Window</label>
            <select
              value={windowHours}
              onChange={(e) => setWindowHours(Number(e.target.value))}
            >
              {WINDOW_PRESETS.map((opt) => (
                <option key={opt.hours} value={opt.hours}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Epic</label>
            <select value={epic} onChange={(e) => setEpic(e.target.value)}>
              {epicOptions.map((opt) => (
                <option key={opt} value={opt}>
                  {opt === "All" ? "All pairs" : shortEpic(opt)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label>Trailing Stops</label>
            <select value={String(trailing)} onChange={(e) => setTrailing(e.target.value === "true")}>
              <option value="true">Applied (progressive)</option>
              <option value="false">Fixed SL/TP only</option>
            </select>
          </div>
          <button
            className="alert-history-button alert-history-button-active"
            onClick={load}
          >
            Refresh
          </button>
        </div>

        {error ? <div className="error">{error}</div> : null}

        {loading ? (
          <div className="chart-placeholder">Running candle replay...</div>
        ) : !stats ? (
          <div className="chart-placeholder">No data.</div>
        ) : (
          <>
            <div className="metrics-grid">
              <div className="summary-card">
                Rejections
                <strong>{stats.total}</strong>
              </div>
              <div className="summary-card">
                Resolved
                <strong>{stats.resolved}</strong>
              </div>
              <div className="summary-card">
                Win Rate
                <strong>{stats.win_rate.toFixed(1)}%</strong>
              </div>
              <div className="summary-card">
                Net Pips
                <strong className={stats.net_pips < 0 ? "rej-good" : stats.net_pips > 0 ? "rej-bad" : ""}>
                  {signed(stats.net_pips)}
                </strong>
              </div>
              <div className="summary-card">
                Profit Factor
                <strong>
                  {stats.profit_factor === null ? "—" : stats.profit_factor.toFixed(2)}
                </strong>
              </div>
              <div className="summary-card">
                Wins / Losses / Trailed / BE
                <strong>
                  {stats.wins} / {stats.losses} / {stats.trailed} / {stats.breakevens}
                </strong>
              </div>
              <div className="summary-card">
                Unresolved (<code>&lt;48h</code>)
                <strong>{stats.still_open}</strong>
              </div>
              <div className={`summary-card rej-verdict rej-verdict-${verdictTone(stats.verdict)}`}>
                Verdict
                <strong>{verdictLabel(stats.verdict)}</strong>
              </div>
            </div>

            <div className="panel table-panel">
              <div className="chart-title">By Pair (counterfactual — no trades were placed)</div>
              {byPair.length ? (
                <div className="rej-pair-table">
                  <div className="rej-pair-header">
                    <span>Pair</span>
                    <span>N</span>
                    <span>WR%</span>
                    <span>W/L/Tr/BE</span>
                    <span>Net pips</span>
                    <span>Verdict</span>
                  </div>
                  {byPair.map((p) => (
                    <div className="rej-pair-row" key={p.epic}>
                      <span>{shortEpic(p.epic)}</span>
                      <span>{p.n}</span>
                      <span>{p.win_rate.toFixed(1)}</span>
                      <span>
                        {p.wins}/{p.losses}/{p.trailed}/{p.breakevens}
                      </span>
                      <span className={p.net_pips < 0 ? "rej-good" : p.net_pips > 0 ? "rej-bad" : ""}>
                        {signed(p.net_pips)}
                      </span>
                      <span>
                        {p.verdict === "CLAUDE_RIGHT"
                          ? "✅ right"
                          : p.verdict === "CLAUDE_WRONG"
                          ? "❌ wrong"
                          : "—"}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="chart-placeholder">No resolved rejections in this window.</div>
              )}
            </div>

            <div className="panel table-panel">
              <div className="chart-title">
                Rejected Alerts ({rejections.length}) — candle-validated counterfactuals (no trades placed)
              </div>
              {rejections.length ? (
                <div className="rej-table">
                  <div className="rej-table-header">
                    <span></span>
                    <span>Time</span>
                    <span>Pair</span>
                    <span>Dir</span>
                    <span>Score</span>
                    <span>Conf</span>
                    <span>SL/TP</span>
                    <span>MFE</span>
                    <span>MAE</span>
                    <span>Trail</span>
                    <span>Outcome</span>
                    <span>Pips</span>
                    <span>Mins</span>
                  </div>
                  {rejections.map((row) => {
                    const isOpen = expanded[row.id];
                    const trailingApplied = row.trailing_used && trailingPairs.has(row.epic);
                    return (
                      <div key={row.id}>
                        <div className={`rej-row ${isOpen ? "rej-row-open" : ""}`}>
                          <button
                            className="expand-btn"
                            onClick={() => setExpanded((p) => ({ ...p, [row.id]: !p[row.id] }))}
                          >
                            {isOpen ? "▾" : "▸"}
                          </button>
                          <span>{formatDateTime(row.alert_timestamp)}</span>
                          <span>{shortEpic(row.epic)}</span>
                          <span>{row.signal_type}</span>
                          <span>{row.claude_score ?? "-"}</span>
                          <span>
                            {row.confidence_score === null
                              ? "-"
                              : row.confidence_score.toFixed(2)}
                          </span>
                          <span>
                            {row.sl_pips}/{row.tp_pips}
                          </span>
                          <span className="rej-good-text">
                            {signed(row.max_favorable_pips)}
                          </span>
                          <span className="rej-bad-text">
                            {signed(row.max_adverse_pips)}
                          </span>
                          <span>{trailingApplied ? "yes" : "—"}</span>
                          <span className={outcomeClass(row.outcome)}>
                            {outcomeLabel(row.outcome)}
                          </span>
                          <span
                            className={
                              row.pips > 0 ? "rej-bad-text" : row.pips < 0 ? "rej-good-text" : ""
                            }
                          >
                            {signed(row.pips)}
                          </span>
                          <span>{row.minutes_to_resolve ?? "-"}</span>
                        </div>
                        {isOpen ? (
                          <div className="rej-detail">
                            <div className="rej-detail-grid">
                              <div>
                                <h5>Trade Setup</h5>
                                <div className="rej-detail-item">
                                  <span>Entry:</span>
                                  <strong>{row.price.toFixed(5)}</strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>SL / TP:</span>
                                  <strong>
                                    {row.sl_pips} / {row.tp_pips} pips
                                  </strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Candle resolution:</span>
                                  <strong>{row.candle_resolution}</strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Session:</span>
                                  <strong>{row.market_session ?? "—"}</strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Regime:</span>
                                  <strong>{row.market_regime ?? "—"}</strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Trailing applied:</span>
                                  <strong>{trailingApplied ? "yes" : "no"}</strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Locked at exit:</span>
                                  <strong>{signed(row.locked_pips)} pips</strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Alert age:</span>
                                  <strong>
                                    {row.alert_age_hours < 72
                                      ? `${row.alert_age_hours.toFixed(1)}h`
                                      : `${(row.alert_age_hours / 24).toFixed(1)}d`}
                                  </strong>
                                </div>
                                <div className="rej-detail-item">
                                  <span>Exit reason:</span>
                                  <strong>{row.exit_reason}</strong>
                                </div>
                              </div>
                              <div>
                                <h5>Claude Rationale</h5>
                                <p className="rej-reason">
                                  {row.claude_reason ?? "No rationale recorded."}
                                </p>
                              </div>
                            </div>
                            {row.stop_path.length > 1 ? (
                              <div className="rej-stop-path">
                                <h5>Stop Trail Path</h5>
                                <div className="rej-stop-path-rows">
                                  {row.stop_path.map((pt, idx) => (
                                    <div className="rej-stop-path-row" key={idx}>
                                      <span>t+{pt.minutes}m</span>
                                      <span>{pt.label}</span>
                                      <span>
                                        stop @ {signed(pt.stop_pips_from_entry)} pips
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="chart-placeholder">No Claude rejections in this window.</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
