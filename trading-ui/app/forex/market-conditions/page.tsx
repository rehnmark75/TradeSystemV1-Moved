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

const regimeCardClass = (shift: PairBlock["regime_shift"]) => {
  switch (shift) {
    case "RANGING→TRENDING":
    case "MORE_VOLATILE":
      return "mc-card-positive";
    case "TRENDING→RANGING":
    case "LESS_VOLATILE":
      return "mc-card-negative";
    case "STABLE":
      return "mc-card-stable";
    default:
      return "mc-card-nodata";
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
  const allPairs = data?.pairs ?? [];

  const overview = useMemo(() => {
    if (!allPairs.length) return null;

    const trendingImproved = allPairs.filter((p) => p.regime_shift === "RANGING→TRENDING").length;
    const trendingFaded = allPairs.filter((p) => p.regime_shift === "TRENDING→RANGING").length;
    const moreVolatile = allPairs.filter((p) => p.regime_shift === "MORE_VOLATILE").length;
    const lessVolatile = allPairs.filter((p) => p.regime_shift === "LESS_VOLATILE").length;
    const avgAdxShift =
      allPairs.reduce((sum, p) => sum + (p.adx_delta ?? 0), 0) / allPairs.length;

    const strongestShift = [...allPairs]
      .filter((p) => p.adx_delta != null)
      .sort((a, b) => Math.abs(b.adx_delta ?? 0) - Math.abs(a.adx_delta ?? 0))[0];

    const strongestApproval = [...allPairs]
      .filter((p) => p.a.approval_rate != null && p.b.approval_rate != null)
      .sort(
        (a, b) =>
          (b.a.approval_rate ?? 0) - (b.b.approval_rate ?? 0) - ((a.a.approval_rate ?? 0) - (a.b.approval_rate ?? 0))
      )[0];

    return {
      total: allPairs.length,
      trendingImproved,
      trendingFaded,
      moreVolatile,
      lessVolatile,
      avgAdxShift,
      strongestShift,
      strongestApproval,
    };
  }, [allPairs]);

  const renderRow = (p: PairBlock) => {
    const isOpen = expanded === p.epic;
    const regimeCls = regimeBadgeClass(p.regime_shift);
    const cardCls = regimeCardClass(p.regime_shift);
    const signalsDelta = deltaCell(p.b.signals, p.a.signals, { dp: 0 });
    const approvalDelta = deltaCell(p.b.approval_rate, p.a.approval_rate);
    const adxDelta = deltaCell(p.b.avg_adx, p.a.avg_adx);
    const rangeDelta = deltaCell(p.b.avg_daily_range_pips, p.a.avg_daily_range_pips);
    const moveDelta = deltaCell(
      p.b.avg_abs_daily_move_pips,
      p.a.avg_abs_daily_move_pips
    );
    const adxDirection =
      p.adx_delta == null ? "No ADX change" : p.adx_delta >= 0 ? "ADX strengthening" : "ADX fading";

    return (
      <div key={p.epic} className={`mc-pair-card ${cardCls}`}>
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
          <div className="mc-head-metrics">
            <div className="mc-head-chip">
              <span>ADX Δ</span>
              <strong className={adxDelta.cls}>{adxDelta.text}</strong>
            </div>
            <div className="mc-head-chip">
              <span>Approval</span>
              <strong className={approvalDelta.cls}>{approvalDelta.text}</strong>
            </div>
            <div className="mc-toggle">{isOpen ? "▾" : "▸"}</div>
          </div>
        </div>

        <div className="mc-hero-strip">
          <div className="mc-hero-metric">
            <span className="mc-hero-label">Current period read</span>
            <strong>{adxDirection}</strong>
            <small>
              ADX {fmt(p.a.avg_adx)} • range {fmt(p.a.avg_daily_range_pips)} pips • move{" "}
              {fmt(p.a.avg_abs_daily_move_pips)} pips
            </small>
          </div>
          <div className="mc-hero-metric">
            <span className="mc-hero-label">Signal flow</span>
            <strong>
              {p.a.signals} signals / {fmtPct(p.a.approval_rate)} approval
            </strong>
            <small>
              {p.a.bull} bull • {p.a.bear} bear • {p.a.approved} approved
            </small>
          </div>
          <div className="mc-hero-metric">
            <span className="mc-hero-label">Benchmark</span>
            <strong>
              {data?.period_b.label}: {p.b.signals} / {fmtPct(p.b.approval_rate)}
            </strong>
            <small>
              ADX {fmt(p.b.avg_adx)} • trend days {p.b.trend_days}
            </small>
          </div>
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
        <div className="mc-intro-copy">
          <div className="mission-kicker">FX Command Desk</div>
          <h2>Market Conditions — is it the market or the system?</h2>
          <p>
            Compare signal volume, Claude approvals, and price-action regime across two
            periods. Use this when signal flow feels off: the table below tells you
            whether the pair went quiet, ranged out, or shifted volatility.
          </p>
          <div className="mc-intro-pills">
            <span>Regime drift</span>
            <span>ADX context</span>
            <span>Approval pressure</span>
            <span>Volatility shift</span>
          </div>
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

        {overview ? (
          <div className="mc-overview-grid">
            <div className="mc-overview-card">
              <span className="mc-overview-label">Trend Rotation</span>
              <strong>
                {overview.trendingImproved} improving / {overview.trendingFaded} fading
              </strong>
              <small>Pairs moving from range into trend versus losing trend structure.</small>
            </div>
            <div className="mc-overview-card">
              <span className="mc-overview-label">Volatility Tilt</span>
              <strong>
                {overview.moreVolatile} louder / {overview.lessVolatile} quieter
              </strong>
              <small>Fast read on whether expansion or compression is dominating.</small>
            </div>
            <div className="mc-overview-card">
              <span className="mc-overview-label">Desk Average</span>
              <strong className={overview.avgAdxShift >= 0 ? "delta-good" : "delta-bad"}>
                {fmtSigned(overview.avgAdxShift)}
              </strong>
              <small>Average ADX change across all tracked pairs.</small>
            </div>
            <div className="mc-overview-card">
              <span className="mc-overview-label">Largest ADX Move</span>
              <strong>{overview.strongestShift?.pair ?? "—"}</strong>
              <small>
                {overview.strongestShift
                  ? `${fmtSigned(overview.strongestShift.adx_delta)} vs prior period`
                  : "No ADX comparison available."}
              </small>
            </div>
            <div className="mc-overview-card mc-overview-card-wide">
              <span className="mc-overview-label">Approval Leader</span>
              <strong>{overview.strongestApproval?.pair ?? "—"}</strong>
              <small>
                {overview.strongestApproval
                  ? `${fmtPct(overview.strongestApproval.b.approval_rate)} → ${fmtPct(
                      overview.strongestApproval.a.approval_rate
                    )} with ${overview.strongestApproval.a.signals} signals in period A`
                  : "No approval-rate comparison available."}
              </small>
            </div>
          </div>
        ) : null}

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

      <style jsx global>{`
        .mc-intro-copy {
          max-width: 760px;
        }
        .mc-intro-pills {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 18px;
        }
        .mc-intro-pills span {
          border: 1px solid rgba(141, 167, 196, 0.24);
          background:
            linear-gradient(135deg, rgba(103, 232, 249, 0.09), rgba(244, 114, 182, 0.08)),
            rgba(10, 18, 34, 0.55);
          color: #dbe8ff;
          padding: 8px 12px;
          border-radius: 999px;
          font-size: 0.72rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }
        .mc-overview-grid {
          display: grid;
          grid-template-columns: repeat(12, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 18px;
        }
        .mc-overview-card {
          grid-column: span 3;
          position: relative;
          overflow: hidden;
          border: 1px solid rgba(125, 147, 178, 0.14);
          border-radius: 18px;
          padding: 16px 18px;
          background:
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 40%),
            linear-gradient(180deg, rgba(18, 29, 52, 0.96), rgba(8, 14, 28, 0.96));
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }
        .mc-overview-card-wide {
          grid-column: span 12;
        }
        .mc-overview-label {
          display: block;
          margin-bottom: 10px;
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #8ea5c7;
        }
        .mc-overview-card strong {
          display: block;
          font-size: 1.15rem;
          letter-spacing: 0.02em;
          color: #f7fbff;
        }
        .mc-overview-card small {
          display: block;
          margin-top: 8px;
          color: #8fa1bb;
          line-height: 1.45;
        }
        .mc-section-title {
          margin: 26px 0 12px 4px;
          font-size: 0.78rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: #90a6c5;
        }
        .mc-pair-card {
          position: relative;
          overflow: hidden;
          border: 1px solid rgba(123, 147, 179, 0.14);
          border-radius: 20px;
          padding: 16px 18px 18px;
          margin-bottom: 14px;
          background:
            radial-gradient(circle at top right, rgba(45, 212, 191, 0.1), transparent 28%),
            linear-gradient(180deg, rgba(10, 18, 33, 0.98), rgba(8, 13, 24, 0.96));
          box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.04),
            0 14px 28px rgba(0, 0, 0, 0.18);
        }
        .mc-pair-card::before {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
          background:
            linear-gradient(90deg, var(--mc-accent-edge) 0%, transparent 20%, transparent 80%, var(--mc-accent-edge) 100%),
            radial-gradient(circle at top left, var(--mc-accent-glow), transparent 34%),
            radial-gradient(circle at 85% 20%, var(--mc-accent-glow-soft), transparent 28%);
          opacity: 0.92;
        }
        .mc-pair-card::after {
          content: "";
          position: absolute;
          left: 18px;
          right: 18px;
          top: 0;
          height: 3px;
          border-radius: 0 0 999px 999px;
          background: linear-gradient(
            90deg,
            transparent 0%,
            var(--mc-accent-line) 18%,
            var(--mc-accent-line-strong) 50%,
            var(--mc-accent-line) 82%,
            transparent 100%
          );
          opacity: 0.95;
        }
        .mc-card-positive {
          --mc-accent-glow: rgba(34, 197, 94, 0.18);
          --mc-accent-glow-soft: rgba(45, 212, 191, 0.12);
          --mc-accent-edge: rgba(34, 197, 94, 0.08);
          --mc-accent-line: rgba(74, 222, 128, 0.34);
          --mc-accent-line-strong: rgba(110, 231, 183, 0.74);
        }
        .mc-card-negative {
          --mc-accent-glow: rgba(244, 63, 94, 0.18);
          --mc-accent-glow-soft: rgba(251, 146, 60, 0.1);
          --mc-accent-edge: rgba(244, 63, 94, 0.08);
          --mc-accent-line: rgba(251, 113, 133, 0.32);
          --mc-accent-line-strong: rgba(253, 164, 175, 0.72);
        }
        .mc-card-stable {
          --mc-accent-glow: rgba(96, 165, 250, 0.16);
          --mc-accent-glow-soft: rgba(148, 163, 184, 0.1);
          --mc-accent-edge: rgba(96, 165, 250, 0.07);
          --mc-accent-line: rgba(125, 211, 252, 0.28);
          --mc-accent-line-strong: rgba(191, 219, 254, 0.64);
        }
        .mc-card-nodata {
          --mc-accent-glow: rgba(148, 163, 184, 0.14);
          --mc-accent-glow-soft: rgba(100, 116, 139, 0.1);
          --mc-accent-edge: rgba(148, 163, 184, 0.06);
          --mc-accent-line: rgba(148, 163, 184, 0.22);
          --mc-accent-line-strong: rgba(203, 213, 225, 0.46);
        }
        .mc-pair-head {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          cursor: pointer;
          gap: 16px;
          margin-bottom: 14px;
          position: relative;
          z-index: 1;
        }
        .mc-pair-title {
          display: flex;
          gap: 12px;
          align-items: center;
          flex-wrap: wrap;
        }
        .mc-pair-name {
          font-weight: 700;
          font-size: 1.08rem;
          letter-spacing: 0.08em;
        }
        .mc-epic {
          font-size: 0.7rem;
          color: #7f93af;
          font-family: var(--font-mono, monospace);
        }
        .mc-regime-badge {
          padding: 5px 10px;
          border-radius: 999px;
          font-size: 0.7rem;
          letter-spacing: 0.08em;
          font-weight: 700;
          text-transform: uppercase;
          border: 1px solid transparent;
        }
        .mc-regime-bad {
          background: rgba(239, 68, 68, 0.13);
          color: #ff8c8c;
          border-color: rgba(248, 113, 113, 0.18);
        }
        .mc-regime-good {
          background: rgba(34, 197, 94, 0.12);
          color: #7ef0a5;
          border-color: rgba(74, 222, 128, 0.16);
        }
        .mc-regime-neutral {
          background: rgba(148, 163, 184, 0.12);
          color: #d7dfeb;
          border-color: rgba(203, 213, 225, 0.14);
        }
        .mc-regime-nodata {
          background: rgba(148, 163, 184, 0.1);
          color: #7c8ea8;
          border-color: rgba(148, 163, 184, 0.12);
        }
        .mc-head-metrics {
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }
        .mc-head-chip {
          min-width: 108px;
          padding: 8px 10px;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(143, 164, 194, 0.12);
          text-align: right;
        }
        .mc-head-chip span {
          display: block;
          font-size: 0.62rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #8096b5;
          margin-bottom: 3px;
        }
        .mc-head-chip strong {
          font-size: 0.95rem;
          color: #edf4ff;
        }
        .mc-toggle {
          display: grid;
          place-items: center;
          width: 36px;
          height: 36px;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(143, 164, 194, 0.12);
          font-size: 1.05rem;
          color: #a7b9d1;
        }
        .mc-hero-strip {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin-bottom: 14px;
          position: relative;
          z-index: 1;
        }
        .mc-hero-metric {
          padding: 14px 14px 12px;
          border-radius: 14px;
          background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.015)),
            rgba(10, 18, 34, 0.62);
          border: 1px solid rgba(143, 164, 194, 0.12);
        }
        .mc-hero-label {
          display: block;
          margin-bottom: 8px;
          font-size: 0.66rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #89a0c0;
        }
        .mc-hero-metric strong {
          display: block;
          font-size: 1rem;
          color: #f3f8ff;
          margin-bottom: 6px;
        }
        .mc-hero-metric small {
          display: block;
          color: #8ea2bd;
          line-height: 1.45;
        }
        .mc-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
          gap: 12px;
          position: relative;
          z-index: 1;
        }
        .mc-cell {
          padding: 12px 12px 10px;
          border-radius: 14px;
          background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.025), rgba(255, 255, 255, 0.01)),
            rgba(8, 15, 28, 0.72);
          border: 1px solid rgba(143, 164, 194, 0.08);
        }
        .mc-cell-label {
          font-size: 0.65rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #88a0c0;
          margin-bottom: 8px;
        }
        .mc-cell-values {
          display: flex;
          gap: 6px;
          align-items: baseline;
          flex-wrap: wrap;
        }
        .mc-prior {
          color: #91a5bf;
          font-size: 0.9rem;
        }
        .mc-arrow {
          color: #5f748f;
          font-size: 0.8rem;
        }
        .mc-current {
          font-weight: 700;
          font-size: 1.02rem;
        }
        .mc-delta {
          margin-left: auto;
          font-weight: 700;
          font-size: 0.8rem;
        }
        .mc-cell-sub {
          margin-top: 8px;
          font-size: 0.7rem;
          color: #8599b4;
        }
        .mc-drilldown {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 18px;
          padding-top: 18px;
          border-top: 1px dashed rgba(143, 164, 194, 0.18);
          position: relative;
          z-index: 1;
        }
        .mc-drill-title {
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: #8ea4c2;
          margin-bottom: 10px;
        }
        .mc-empty {
          padding: 18px;
          text-align: center;
          color: #8ea2bd;
          font-size: 0.9rem;
        }
        @media (max-width: 1180px) {
          .mc-overview-card {
            grid-column: span 6;
          }
          .mc-overview-card-wide {
            grid-column: span 12;
          }
          .mc-hero-strip {
            grid-template-columns: 1fr;
          }
        }
        @media (max-width: 900px) {
          .mc-overview-grid {
            grid-template-columns: 1fr;
          }
          .mc-overview-card,
          .mc-overview-card-wide {
            grid-column: auto;
          }
          .mc-pair-head {
            flex-direction: column;
          }
          .mc-head-metrics {
            width: 100%;
            justify-content: flex-start;
          }
          .mc-head-chip {
            text-align: left;
          }
          .mc-drilldown {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
