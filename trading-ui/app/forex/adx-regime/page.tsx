/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ForexNav from "../_components/ForexNav";

type RangeStats = {
  bars: number;
  mean: number | null;
  median: number | null;
  pct_below_threshold: number | null;
  pct_below_20: number | null;
};

type PairResult = {
  epic: string;
  short: string;
  threshold: number | null;
  threshold_source: "column" | "jsonb" | "default";
  series: Array<{ t: string; adx: number | null }>;
  this_week: RangeStats;
  last_week: RangeStats;
  coverage: { first_bar: string | null; last_bar: string | null };
};

type Payload = {
  generated_at: string;
  this_week: { start: string; end: string };
  last_week: { start: string; end: string };
  pairs: PairResult[];
};

const fmtNum = (v: number | null, d = 1) =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(d);

const fmtPct = (v: number | null, d = 1) =>
  v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(d)}%`;

const fmtDelta = (last: number | null, cur: number | null, d = 1) => {
  if (last == null || cur == null) return "—";
  const delta = cur - last;
  const sign = delta >= 0 ? "+" : "";
  return `${sign}${delta.toFixed(d)}`;
};

const deltaClass = (last: number | null, cur: number | null, inverse = false) => {
  if (last == null || cur == null) return "";
  const delta = cur - last;
  if (Math.abs(delta) < 0.3) return "delta-neutral";
  const bad = inverse ? delta < 0 : delta > 0;
  return bad ? "delta-bad" : "delta-good";
};

type Severity = {
  level: "nodata" | "ok" | "watch" | "low" | "critical";
  label: string;
  hint?: string;
};

function classifyAdxSeverity(p: PairResult): Severity {
  const mean = p.this_week.mean;
  const pctBelowThresh = p.this_week.pct_below_threshold;
  const pctBelow20 = p.this_week.pct_below_20;

  if (mean == null) return { level: "nodata", label: "NO DATA" };

  // Live (latest) ADX from the series
  const latest = [...p.series].reverse().find((s) => s.adx != null)?.adx ?? null;
  const thresh = p.threshold;

  if (thresh != null && latest != null && latest < thresh) {
    return {
      level: "critical",
      label: "BELOW THRESHOLD NOW",
      hint: `Current ADX ${latest.toFixed(1)} < threshold ${thresh} — scalp filter rejecting signals.`
    };
  }
  if (thresh != null && mean < thresh) {
    return {
      level: "critical",
      label: "WEEK MEAN < THRESHOLD",
      hint: `Mean ADX ${mean.toFixed(1)} < threshold ${thresh} — most bars failing the filter.`
    };
  }
  if (thresh != null && pctBelowThresh != null && pctBelowThresh >= 60) {
    return {
      level: "low",
      label: "MOSTLY BELOW THRESH",
      hint: `${pctBelowThresh.toFixed(0)}% of 5m bars this week below ${thresh}.`
    };
  }
  if (mean < 20 || (pctBelow20 != null && pctBelow20 >= 50)) {
    return {
      level: "low",
      label: "LOW ADX",
      hint:
        pctBelow20 != null
          ? `${pctBelow20.toFixed(0)}% of bars below 20 — choppy/ranging regime.`
          : `Mean ADX ${mean.toFixed(1)} — choppy regime.`
    };
  }
  if (mean < 22 || (pctBelow20 != null && pctBelow20 >= 30)) {
    return { level: "watch", label: "WATCH" };
  }
  return { level: "ok", label: "OK" };
}

function Sparkline({
  pair,
  width = 420,
  height = 90
}: {
  pair: PairResult;
  width?: number;
  height?: number;
}) {
  const series = pair.series.filter((p) => p.adx !== null) as Array<{ t: string; adx: number }>;
  if (series.length < 2) {
    return <div className="spark-empty">No data</div>;
  }

  const padL = 30;
  const padR = 8;
  const padT = 8;
  const padB = 18;
  const w = width - padL - padR;
  const h = height - padT - padB;

  const values = series.map((p) => p.adx);
  const minV = Math.max(0, Math.min(...values) - 3);
  const maxV = Math.max(...values, pair.threshold ?? 0) + 3;
  const rangeV = maxV - minV || 1;

  const t0 = new Date(series[0].t).getTime();
  const t1 = new Date(series[series.length - 1].t).getTime();
  const rangeT = t1 - t0 || 1;

  const xAt = (t: string) => padL + ((new Date(t).getTime() - t0) / rangeT) * w;
  const yAt = (v: number) => padT + (1 - (v - minV) / rangeV) * h;

  const path = series
    .map((p, i) => `${i === 0 ? "M" : "L"}${xAt(p.t).toFixed(1)},${yAt(p.adx).toFixed(1)}`)
    .join(" ");

  // Monday split line
  const thisWeekT = new Date();
  const mondayUtc = new Date(
    Date.UTC(thisWeekT.getUTCFullYear(), thisWeekT.getUTCMonth(), thisWeekT.getUTCDate())
  );
  mondayUtc.setUTCDate(mondayUtc.getUTCDate() - ((mondayUtc.getUTCDay() + 6) % 7));
  const splitX = padL + ((mondayUtc.getTime() - t0) / rangeT) * w;

  const thresholdY = pair.threshold != null ? yAt(pair.threshold) : null;

  const ticks = [minV, (minV + maxV) / 2, maxV];

  return (
    <svg className="adx-spark" width={width} height={height}>
      {/* y-axis ticks */}
      {ticks.map((v, i) => (
        <g key={i}>
          <line
            x1={padL}
            x2={width - padR}
            y1={yAt(v)}
            y2={yAt(v)}
            stroke="#eadfca"
            strokeDasharray="2 4"
          />
          <text x={padL - 4} y={yAt(v) + 3} textAnchor="end" fontSize="9" fill="#8a6f48">
            {v.toFixed(0)}
          </text>
        </g>
      ))}

      {/* Monday split */}
      {splitX >= padL && splitX <= width - padR && (
        <line
          x1={splitX}
          x2={splitX}
          y1={padT}
          y2={height - padB}
          stroke="#b88a55"
          strokeDasharray="3 3"
          opacity={0.8}
        />
      )}

      {/* threshold */}
      {thresholdY !== null && (
        <g>
          <line
            x1={padL}
            x2={width - padR}
            y1={thresholdY}
            y2={thresholdY}
            stroke="#c43d3d"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
          <text
            x={width - padR - 2}
            y={thresholdY - 3}
            textAnchor="end"
            fontSize="9"
            fill="#c43d3d"
          >
            {pair.threshold?.toFixed(0)}
          </text>
        </g>
      )}

      {/* ADX line */}
      <path d={path} fill="none" stroke="#2b6cb0" strokeWidth={1.5} />

      {/* x labels */}
      <text x={padL} y={height - 4} fontSize="9" fill="#8a6f48">
        {new Date(series[0].t).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
      </text>
      <text x={width - padR} y={height - 4} fontSize="9" fill="#8a6f48" textAnchor="end">
        {new Date(series[series.length - 1].t).toLocaleDateString(undefined, {
          month: "short",
          day: "numeric"
        })}
      </text>
    </svg>
  );
}

export default function AdxRegimePage() {
  const [data, setData] = useState<Payload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(21);

  const loadData = () => {
    setLoading(true);
    setError(null);
    fetch(`/trading/api/forex/adx-regime/?days=${days}`)
      .then((r) => r.json())
      .then((json) => {
        if (json.error) throw new Error(json.error);
        setData(json);
      })
      .catch((e) => setError(e.message || "Failed to load"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, [days]);

  const summary = useMemo(() => {
    if (!data) return null;
    let compressed = 0;
    let expanded = 0;
    let avgDelta = 0;
    let count = 0;
    for (const p of data.pairs) {
      const lw = p.last_week.mean;
      const tw = p.this_week.mean;
      if (lw == null || tw == null) continue;
      const d = tw - lw;
      avgDelta += d;
      count++;
      if (d < -0.5) compressed++;
      else if (d > 0.5) expanded++;
    }
    return {
      compressed,
      expanded,
      neutral: count - compressed - expanded,
      avgDelta: count ? avgDelta / count : 0,
      total: count
    };
  }, [data]);

  const overview = useMemo(() => {
    if (!summary) return null;
    if (summary.total === 0) return "Awaiting ADX data…";
    if (summary.compressed >= summary.total - 1)
      return "Broad ADX compression — market is going quiet across the board.";
    if (summary.expanded >= summary.total - 1)
      return "Broad ADX expansion — trending conditions developing across pairs.";
    if (summary.compressed > summary.expanded)
      return `${summary.compressed} of ${summary.total} pairs compressing, ${summary.expanded} expanding. Mixed regime skewed quiet.`;
    if (summary.expanded > summary.compressed)
      return `${summary.expanded} of ${summary.total} pairs expanding, ${summary.compressed} compressing. Mixed regime skewed trending.`;
    return `${summary.total} pairs in mixed regime — no dominant direction.`;
  }, [summary]);

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
          <h1>ADX Regime — Week vs Week</h1>
          <p>
            Rolling ADX(14) on 5m candles per enabled pair, overlaid with the per-pair{" "}
            <code>scalp_min_adx</code> threshold. Vertical dashed line = start of current week.
          </p>
        </div>
      </div>

      <ForexNav activeHref="/forex/adx-regime" />

      <div className="panel">
        <div className="adx-controls">
          <label>
            Lookback:{" "}
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              <option value={14}>14 days</option>
              <option value={21}>21 days</option>
              <option value={30}>30 days</option>
              <option value={45}>45 days</option>
            </select>
          </label>
          <button onClick={loadData} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
          {data && (
            <span className="muted">
              Generated {new Date(data.generated_at).toLocaleString()} • This wk{" "}
              {new Date(data.this_week.start).toLocaleDateString()} • Last wk{" "}
              {new Date(data.last_week.start).toLocaleDateString()}
            </span>
          )}
        </div>

        {error && <div className="error-banner">Error: {error}</div>}

        {overview && <div className="adx-overview">{overview}</div>}

        {data && (
          <div className="adx-summary-table">
            <div className="adx-row adx-header">
              <span>Pair</span>
              <span title="scalp_min_adx threshold for this pair">Thresh</span>
              <span>Last mean</span>
              <span>This mean</span>
              <span>Δ mean</span>
              <span title="% of 5m bars below scalp_min_adx threshold">% &lt;thresh last</span>
              <span>% &lt;thresh this</span>
              <span>% &lt;20 this</span>
              <span>Coverage</span>
            </div>
            {data.pairs.map((p) => {
              const coverageEnd = p.coverage.last_bar
                ? new Date(p.coverage.last_bar).toLocaleDateString()
                : "—";
              return (
                <div className="adx-row" key={p.epic}>
                  <span>
                    <strong>{p.short}</strong>
                    <em className={`thresh-src thresh-src-${p.threshold_source}`}>
                      {p.threshold_source}
                    </em>
                  </span>
                  <span>{fmtNum(p.threshold, 0)}</span>
                  <span>{fmtNum(p.last_week.mean)}</span>
                  <span>{fmtNum(p.this_week.mean)}</span>
                  <span className={deltaClass(p.last_week.mean, p.this_week.mean, true)}>
                    {fmtDelta(p.last_week.mean, p.this_week.mean)}
                  </span>
                  <span>{fmtPct(p.last_week.pct_below_threshold)}</span>
                  <span className={deltaClass(p.last_week.pct_below_threshold, p.this_week.pct_below_threshold)}>
                    {fmtPct(p.this_week.pct_below_threshold)}
                  </span>
                  <span>{fmtPct(p.this_week.pct_below_20)}</span>
                  <span className="muted" style={{ fontSize: "0.8rem" }}>
                    → {coverageEnd}
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {data && (
          <div className="adx-spark-grid">
            {data.pairs.map((p) => {
              const severity = classifyAdxSeverity(p);
              return (
                <div className={`adx-spark-card adx-card-${severity.level}`} key={p.epic}>
                  <div className="adx-spark-head">
                    <div className="adx-spark-title">
                      <strong>{p.short}</strong>
                      <span className={`adx-badge adx-badge-${severity.level}`}>
                        {severity.label}
                      </span>
                    </div>
                    <span className="muted">
                      thresh {fmtNum(p.threshold, 0)} • this {fmtNum(p.this_week.mean)} (last{" "}
                      {fmtNum(p.last_week.mean)})
                    </span>
                  </div>
                  {severity.hint && <div className="adx-hint">{severity.hint}</div>}
                  <Sparkline pair={p} />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
