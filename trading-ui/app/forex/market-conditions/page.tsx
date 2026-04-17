/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";
import { useEnvironment } from "../../../lib/environment";
import EnvironmentToggle from "../../../components/EnvironmentToggle";

type PeriodStats = {
  signals: number;
  approved: number;
  rejected: number;
  approval_rate: number | null;
  bull: number;
  bear: number;
  avg_conf: number | null;
  avg_adx: number | null;
  avg_rsi: number | null;
  avg_atr: number | null;
  avg_spread: number | null;
  dominant_regime: string | null;
  avg_daily_range_pips: number | null;
  avg_abs_daily_move_pips: number | null;
  cumulative_move_pips: number | null;
  trend_days: number;
  trading_days: number;
  period_range_pips: number | null;
};

type DailyRow = {
  epic: string;
  d: string;
  dow: string;
  open: number;
  close: number;
  low: number;
  high: number;
  net_pips: number;
  range_pips: number;
};

type PairBlock = {
  pair: string;
  epic: string;
  pip_size: number;
  is_gold: boolean;
  a: PeriodStats;
  b: PeriodStats;
  daily_a: DailyRow[];
  daily_b: DailyRow[];
  regime_shift:
    | "TRENDING→RANGING"
    | "RANGING→TRENDING"
    | "STABLE"
    | "MORE_VOLATILE"
    | "LESS_VOLATILE"
    | "NO_DATA";
  adx_delta: number | null;
};

type Payload = {
  generated_at: string;
  env: string;
  period_a: { start: string; end: string; label: string };
  period_b: { start: string; end: string; label: string };
  pairs: PairBlock[];
};

type PresetKey = "this_vs_last_week" | "today_vs_yesterday" | "7d_vs_prior" | "custom";

const PRESETS: Array<{ key: PresetKey; label: string }> = [
  { key: "this_vs_last_week", label: "This week vs last week" },
  { key: "today_vs_yesterday", label: "Today vs yesterday" },
  { key: "7d_vs_prior", label: "7d vs prior 7d" },
  { key: "custom", label: "Custom" },
];

function addDaysUTC(d: Date, n: number): Date {
  const c = new Date(d);
  c.setUTCDate(c.getUTCDate() + n);
  return c;
}

function startOfIsoWeek(d: Date): Date {
  const c = new Date(d);
  c.setUTCHours(0, 0, 0, 0);
  const dow = c.getUTCDay();
  const diff = (dow + 6) % 7;
  c.setUTCDate(c.getUTCDate() - diff);
  return c;
}

function startOfDayUTC(d: Date): Date {
  const c = new Date(d);
  c.setUTCHours(0, 0, 0, 0);
  return c;
}

function toIso(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function computePresetRanges(key: PresetKey): {
  aStart: string;
  aEnd: string;
  bStart: string;
  bEnd: string;
} {
  const now = new Date();
  if (key === "this_vs_last_week") {
    const curWk = startOfIsoWeek(now);
    const lastWk = addDaysUTC(curWk, -7);
    return {
      aStart: toIso(curWk),
      aEnd: toIso(addDaysUTC(curWk, 7)),
      bStart: toIso(lastWk),
      bEnd: toIso(curWk),
    };
  }
  if (key === "today_vs_yesterday") {
    const today = startOfDayUTC(now);
    return {
      aStart: toIso(today),
      aEnd: toIso(addDaysUTC(today, 1)),
      bStart: toIso(addDaysUTC(today, -1)),
      bEnd: toIso(today),
    };
  }
  if (key === "7d_vs_prior") {
    const end = startOfDayUTC(addDaysUTC(now, 1));
    const aStart = addDaysUTC(end, -7);
    const bEnd = aStart;
    const bStart = addDaysUTC(bEnd, -7);
    return {
      aStart: toIso(aStart),
      aEnd: toIso(end),
      bStart: toIso(bStart),
      bEnd: toIso(bEnd),
    };
  }
  // custom: default to this vs last week on first open
  return computePresetRanges("this_vs_last_week");
}

const fmt = (v: number | null, d = 1) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(d);

const fmtPct = (v: number | null) =>
  v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(0)}%`;

const fmtSigned = (v: number | null, d = 1) => {
  if (v == null || !Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(d)}`;
};

const deltaCell = (
  b: number | null,
  a: number | null,
  opts: { dp?: number; inverseGood?: boolean; epsilon?: number } = {}
) => {
  const { dp = 1, inverseGood = false, epsilon = 0.05 } = opts;
  if (a == null || b == null) return { text: "—", cls: "" };
  const delta = a - b;
  const sign = delta > 0 ? "+" : "";
  const text = `${sign}${delta.toFixed(dp)}`;
  if (Math.abs(delta) < epsilon) return { text, cls: "delta-neutral" };
  const better = inverseGood ? delta < 0 : delta > 0;
  return { text, cls: better ? "delta-good" : "delta-bad" };
};

const regimeBadgeClass = (shift: PairBlock["regime_shift"]) => {
  switch (shift) {
    case "TRENDING→RANGING":
    case "LESS_VOLATILE":
      return "bad";
    case "RANGING→TRENDING":
    case "MORE_VOLATILE":
      return "good";
    case "STABLE":
      return "neutral";
    default:
      return "nodata";
  }
};

export default function MarketConditionsPage() {
  const { environment } = useEnvironment();
  const [preset, setPreset] = useState<PresetKey>("this_vs_last_week");
  const initialRanges = useMemo(() => computePresetRanges("this_vs_last_week"), []);
  const [aStart, setAStart] = useState(initialRanges.aStart);
  const [aEnd, setAEnd] = useState(initialRanges.aEnd);
  const [bStart, setBStart] = useState(initialRanges.bStart);
  const [bEnd, setBEnd] = useState(initialRanges.bEnd);
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  // When preset changes (unless custom), recompute the date inputs
  useEffect(() => {
    if (preset === "custom") return;
    const r = computePresetRanges(preset);
    setAStart(r.aStart);
    setAEnd(r.aEnd);
    setBStart(r.bStart);
    setBEnd(r.bEnd);
  }, [preset]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    const params = new URLSearchParams({
      env: environment,
      aStart,
      aEnd,
      bStart,
      bEnd,
    });
    fetch(`/trading/api/forex/market-conditions?${params.toString()}`, {
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then((json) => {
        if (json.error) throw new Error(json.error);
        setData(json);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message || "Failed to load");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [environment, aStart, aEnd, bStart, bEnd]);

  const fx = (data?.pairs ?? []).filter((p) => !p.is_gold);
  const gold = (data?.pairs ?? []).filter((p) => p.is_gold);

  const renderRow = (p: PairBlock) => {
    const isOpen = expanded === p.epic;
    const regimeCls = regimeBadgeClass(p.regime_shift);
    const signalsDelta = deltaCell(p.b.signals, p.a.signals, { dp: 0 });
    const approvalDelta = deltaCell(p.b.approval_rate, p.a.approval_rate);
    const adxDelta = deltaCell(p.b.avg_adx, p.a.avg_adx);
    const rangeDelta = deltaCell(p.b.avg_daily_range_pips, p.a.avg_daily_range_pips);
    const moveDelta = deltaCell(
      p.b.avg_abs_daily_move_pips,
      p.a.avg_abs_daily_move_pips
    );

    return (
      <div key={p.epic} className="mc-pair-card">
        <div
          className="mc-pair-head"
          onClick={() => setExpanded(isOpen ? null : p.epic)}
          role="button"
        >
          <div className="mc-pair-title">
            <span className="mc-pair-name">{p.pair}</span>
            <span className={`mc-regime-badge mc-regime-${regimeCls}`}>
              {p.regime_shift.replace("_", " ")}
            </span>
            <span className="mc-epic">{p.epic}</span>
          </div>
          <div className="mc-toggle">{isOpen ? "▾" : "▸"}</div>
        </div>

        <div className="mc-grid">
          <div className="mc-cell">
            <div className="mc-cell-label">Signals</div>
            <div className="mc-cell-values">
              <span className="mc-prior">{p.b.signals}</span>
              <span className="mc-arrow">→</span>
              <span className="mc-current">{p.a.signals}</span>
              <span className={`mc-delta ${signalsDelta.cls}`}>
                {signalsDelta.text}
              </span>
            </div>
            <div className="mc-cell-sub">
              {p.b.bull}B/{p.b.bear}S → {p.a.bull}B/{p.a.bear}S
            </div>
          </div>

          <div className="mc-cell">
            <div className="mc-cell-label">Claude approval</div>
            <div className="mc-cell-values">
              <span className="mc-prior">{fmtPct(p.b.approval_rate)}</span>
              <span className="mc-arrow">→</span>
              <span className="mc-current">{fmtPct(p.a.approval_rate)}</span>
              <span className={`mc-delta ${approvalDelta.cls}`}>
                {approvalDelta.text}
              </span>
            </div>
            <div className="mc-cell-sub">
              {p.b.approved}/{p.b.approved + p.b.rejected} → {p.a.approved}/
              {p.a.approved + p.a.rejected}
            </div>
          </div>

          <div className="mc-cell">
            <div className="mc-cell-label">Avg ADX</div>
            <div className="mc-cell-values">
              <span className="mc-prior">{fmt(p.b.avg_adx)}</span>
              <span className="mc-arrow">→</span>
              <span className="mc-current">{fmt(p.a.avg_adx)}</span>
              <span className={`mc-delta ${adxDelta.cls}`}>{adxDelta.text}</span>
            </div>
            <div className="mc-cell-sub">
              RSI {fmt(p.b.avg_rsi, 0)} → {fmt(p.a.avg_rsi, 0)}
            </div>
          </div>

          <div className="mc-cell">
            <div className="mc-cell-label">Avg daily range (pips)</div>
            <div className="mc-cell-values">
              <span className="mc-prior">{fmt(p.b.avg_daily_range_pips)}</span>
              <span className="mc-arrow">→</span>
              <span className="mc-current">{fmt(p.a.avg_daily_range_pips)}</span>
              <span className={`mc-delta ${rangeDelta.cls}`}>{rangeDelta.text}</span>
            </div>
            <div className="mc-cell-sub">
              {p.b.trading_days}d → {p.a.trading_days}d trading days
            </div>
          </div>

          <div className="mc-cell">
            <div className="mc-cell-label">Avg |daily move|</div>
            <div className="mc-cell-values">
              <span className="mc-prior">{fmt(p.b.avg_abs_daily_move_pips)}</span>
              <span className="mc-arrow">→</span>
              <span className="mc-current">{fmt(p.a.avg_abs_daily_move_pips)}</span>
              <span className={`mc-delta ${moveDelta.cls}`}>{moveDelta.text}</span>
            </div>
            <div className="mc-cell-sub">
              Trend days {p.b.trend_days} → {p.a.trend_days}
            </div>
          </div>

          <div className="mc-cell">
            <div className="mc-cell-label">Cumulative move (pips)</div>
            <div className="mc-cell-values">
              <span className="mc-prior">{fmtSigned(p.b.cumulative_move_pips)}</span>
              <span className="mc-arrow">→</span>
              <span className="mc-current">{fmtSigned(p.a.cumulative_move_pips)}</span>
            </div>
            <div className="mc-cell-sub">
              Period range {fmt(p.b.period_range_pips)} → {fmt(p.a.period_range_pips)}
            </div>
          </div>
        </div>

        {isOpen ? (
          <div className="mc-drilldown">
            <div className="mc-drill-col">
              <div className="mc-drill-title">
                Period B — {data?.period_b.label}
              </div>
              {p.daily_b.length ? (
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Dow</th>
                      <th>Open</th>
                      <th>Close</th>
                      <th>Net</th>
                      <th>Range</th>
                      <th>Dir</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.daily_b.map((d) => (
                      <tr key={`b-${d.d}`}>
                        <td>{d.d}</td>
                        <td>{d.dow}</td>
                        <td>{d.open.toFixed(p.is_gold ? 2 : 3)}</td>
                        <td>{d.close.toFixed(p.is_gold ? 2 : 3)}</td>
                        <td
                          className={d.net_pips >= 0 ? "delta-good" : "delta-bad"}
                        >
                          {fmtSigned(d.net_pips)}
                        </td>
                        <td>{fmt(d.range_pips)}</td>
                        <td>
                          {d.net_pips > 0 ? "UP" : d.net_pips < 0 ? "DN" : "="}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="mc-empty">No candle data.</div>
              )}
            </div>
            <div className="mc-drill-col">
              <div className="mc-drill-title">
                Period A — {data?.period_a.label}
              </div>
              {p.daily_a.length ? (
                <table className="forex-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Dow</th>
                      <th>Open</th>
                      <th>Close</th>
                      <th>Net</th>
                      <th>Range</th>
                      <th>Dir</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.daily_a.map((d) => (
                      <tr key={`a-${d.d}`}>
                        <td>{d.d}</td>
                        <td>{d.dow}</td>
                        <td>{d.open.toFixed(p.is_gold ? 2 : 3)}</td>
                        <td>{d.close.toFixed(p.is_gold ? 2 : 3)}</td>
                        <td
                          className={d.net_pips >= 0 ? "delta-good" : "delta-bad"}
                        >
                          {fmtSigned(d.net_pips)}
                        </td>
                        <td>{fmt(d.range_pips)}</td>
                        <td>
                          {d.net_pips > 0 ? "UP" : d.net_pips < 0 ? "DN" : "="}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="mc-empty">No candle data.</div>
              )}
            </div>
          </div>
        ) : null}
      </div>
    );
  };

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

      <div className="desk-intro">
        <div>
          <div className="mission-kicker">FX Command Desk</div>
          <h2>Market Conditions — is it the market or the system?</h2>
          <p>
            Compare signal volume, Claude approvals, and price-action regime across two
            periods. Use this when signal flow feels off: the table below tells you
            whether the pair went quiet, ranged out, or shifted volatility.
          </p>
        </div>
        <div className="desk-intro-meta">
          <div className="desk-intro-stat">
            <span>Environment</span>
            <strong>{environment.toUpperCase()}</strong>
          </div>
          <div className="desk-intro-stat">
            <span>Sorted by</span>
            <strong>Biggest ADX shift</strong>
          </div>
        </div>
      </div>

      <ForexNav activeHref="/forex/market-conditions" />

      <div className="panel">
        <div className="forex-controls">
          <div>
            <label>Preset</label>
            <select value={preset} onChange={(e) => setPreset(e.target.value as PresetKey)}>
              {PRESETS.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label>Period B start</label>
            <input
              type="date"
              value={bStart}
              disabled={preset !== "custom"}
              onChange={(e) => {
                setPreset("custom");
                setBStart(e.target.value);
              }}
            />
          </div>
          <div>
            <label>Period B end</label>
            <input
              type="date"
              value={bEnd}
              disabled={preset !== "custom"}
              onChange={(e) => {
                setPreset("custom");
                setBEnd(e.target.value);
              }}
            />
          </div>
          <div>
            <label>Period A start</label>
            <input
              type="date"
              value={aStart}
              disabled={preset !== "custom"}
              onChange={(e) => {
                setPreset("custom");
                setAStart(e.target.value);
              }}
            />
          </div>
          <div>
            <label>Period A end</label>
            <input
              type="date"
              value={aEnd}
              disabled={preset !== "custom"}
              onChange={(e) => {
                setPreset("custom");
                setAEnd(e.target.value);
              }}
            />
          </div>
          {data ? (
            <div className="forex-badge">
              B: {data.period_b.label} &nbsp;|&nbsp; A: {data.period_a.label}
            </div>
          ) : null}
        </div>

        {error ? <div className="error">{error}</div> : null}
        {loading ? <div className="chart-placeholder">Loading…</div> : null}

        {!loading && data ? (
          <>
            <div className="mc-section-title">FX Pairs ({fx.length})</div>
            {fx.length ? fx.map(renderRow) : <div className="mc-empty">No pair data.</div>}

            {gold.length ? (
              <>
                <div className="mc-section-title">Metals</div>
                {gold.map(renderRow)}
              </>
            ) : null}
          </>
        ) : null}
      </div>

      <style jsx>{`
        .mc-section-title {
          margin: 20px 0 8px 4px;
          font-size: 0.85rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: var(--ink-soft, #64748b);
        }
        .mc-pair-card {
          border: 1px solid var(--line, rgba(255, 255, 255, 0.08));
          border-radius: 10px;
          padding: 12px 16px;
          margin-bottom: 10px;
          background: var(--panel-soft, rgba(255, 255, 255, 0.02));
        }
        .mc-pair-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          cursor: pointer;
          margin-bottom: 10px;
        }
        .mc-pair-title {
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
        }
        .mc-pair-name {
          font-weight: 700;
          font-size: 1rem;
          letter-spacing: 0.04em;
        }
        .mc-epic {
          font-size: 0.7rem;
          color: var(--ink-soft, #94a3b8);
          font-family: var(--font-mono, monospace);
        }
        .mc-regime-badge {
          padding: 2px 8px;
          border-radius: 4px;
          font-size: 0.7rem;
          letter-spacing: 0.05em;
          font-weight: 600;
        }
        .mc-regime-bad {
          background: rgba(239, 68, 68, 0.15);
          color: #f87171;
        }
        .mc-regime-good {
          background: rgba(34, 197, 94, 0.15);
          color: #4ade80;
        }
        .mc-regime-neutral {
          background: rgba(148, 163, 184, 0.15);
          color: #cbd5e1;
        }
        .mc-regime-nodata {
          background: rgba(148, 163, 184, 0.1);
          color: #64748b;
        }
        .mc-toggle {
          font-size: 1.1rem;
          color: var(--ink-soft, #94a3b8);
        }
        .mc-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 10px;
        }
        .mc-cell {
          padding: 8px 10px;
          border-radius: 6px;
          background: var(--panel-inner, rgba(255, 255, 255, 0.03));
        }
        .mc-cell-label {
          font-size: 0.7rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--ink-soft, #94a3b8);
          margin-bottom: 4px;
        }
        .mc-cell-values {
          display: flex;
          gap: 6px;
          align-items: baseline;
          flex-wrap: wrap;
        }
        .mc-prior {
          color: var(--ink-soft, #94a3b8);
          font-size: 0.9rem;
        }
        .mc-arrow {
          color: var(--ink-soft, #64748b);
          font-size: 0.8rem;
        }
        .mc-current {
          font-weight: 700;
          font-size: 1rem;
        }
        .mc-delta {
          margin-left: auto;
          font-weight: 600;
          font-size: 0.85rem;
        }
        .mc-cell-sub {
          margin-top: 4px;
          font-size: 0.7rem;
          color: var(--ink-soft, #94a3b8);
        }
        .mc-drilldown {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 14px;
          padding-top: 14px;
          border-top: 1px dashed var(--line, rgba(255, 255, 255, 0.08));
        }
        .mc-drill-title {
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-soft, #94a3b8);
          margin-bottom: 6px;
        }
        .mc-empty {
          padding: 18px;
          text-align: center;
          color: var(--ink-soft, #94a3b8);
          font-size: 0.9rem;
        }
        @media (max-width: 900px) {
          .mc-drilldown {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
