import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_PAIRS = [
  "CS.D.EURUSD.CEEM.IP",
  "CS.D.USDJPY.MINI.IP",
  "CS.D.EURJPY.MINI.IP",
  "CS.D.GBPUSD.MINI.IP",
  "CS.D.AUDJPY.MINI.IP",
  "CS.D.AUDUSD.MINI.IP",
  "CS.D.NZDUSD.MINI.IP",
  "CS.D.USDCAD.MINI.IP"
];

const ADX_PERIOD = 14;
const BARS_PER_PAIR_IN_RESPONSE = 240;

type Candle = { t: Date; h: number; l: number; c: number };

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

function mondayOfWeek(d: Date): Date {
  const utc = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const day = utc.getUTCDay(); // 0=Sun, 1=Mon, ...
  const diff = (day + 6) % 7; // days since Monday
  utc.setUTCDate(utc.getUTCDate() - diff);
  return utc;
}

function rma(src: number[], n: number): number[] {
  const out = new Array<number>(src.length).fill(NaN);
  if (src.length < n) return out;
  // seed: mean of first n non-NaN values we have
  let seed = 0;
  let count = 0;
  let seedIdx = -1;
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (!Number.isFinite(v)) continue;
    seed += v;
    count++;
    if (count === n) {
      seedIdx = i;
      break;
    }
  }
  if (seedIdx < 0) return out;
  out[seedIdx] = seed / n;
  const alpha = 1 / n;
  for (let i = seedIdx + 1; i < src.length; i++) {
    const prev = out[i - 1];
    const v = src[i];
    if (!Number.isFinite(v) || !Number.isFinite(prev)) {
      out[i] = prev;
    } else {
      out[i] = prev + alpha * (v - prev);
    }
  }
  return out;
}

function computeAdx(candles: Candle[], period = ADX_PERIOD): number[] {
  const n = candles.length;
  const out = new Array<number>(n).fill(NaN);
  if (n < period * 3) return out;

  const tr = new Array<number>(n).fill(NaN);
  const plusDm = new Array<number>(n).fill(NaN);
  const minusDm = new Array<number>(n).fill(NaN);

  for (let i = 1; i < n; i++) {
    const cur = candles[i];
    const prev = candles[i - 1];
    const hlRange = cur.h - cur.l;
    const hcRange = Math.abs(cur.h - prev.c);
    const lcRange = Math.abs(cur.l - prev.c);
    tr[i] = Math.max(hlRange, hcRange, lcRange);
    const up = cur.h - prev.h;
    const dn = prev.l - cur.l;
    plusDm[i] = up > dn && up > 0 ? up : 0;
    minusDm[i] = dn > up && dn > 0 ? dn : 0;
  }

  const atr = rma(tr, period);
  const plusDmS = rma(plusDm, period);
  const minusDmS = rma(minusDm, period);

  const dx = new Array<number>(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const a = atr[i];
    if (!Number.isFinite(a) || a === 0) continue;
    const pdi = (100 * plusDmS[i]) / a;
    const mdi = (100 * minusDmS[i]) / a;
    const sum = pdi + mdi;
    if (!Number.isFinite(pdi) || !Number.isFinite(mdi) || sum === 0) continue;
    dx[i] = (100 * Math.abs(pdi - mdi)) / sum;
  }
  return rma(dx, period);
}

function computeRangeStats(
  series: Array<{ t: Date; adx: number }>,
  start: Date,
  end: Date,
  threshold: number | null
): RangeStats {
  const vals: number[] = [];
  for (const p of series) {
    if (p.t >= start && p.t < end && Number.isFinite(p.adx)) vals.push(p.adx);
  }
  if (!vals.length) {
    return { bars: 0, mean: null, median: null, pct_below_threshold: null, pct_below_20: null };
  }
  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const belowThresh =
    threshold != null ? (vals.filter((v) => v < threshold).length / vals.length) * 100 : null;
  const below20 = (vals.filter((v) => v < 20).length / vals.length) * 100;
  return {
    bars: vals.length,
    mean: Number(mean.toFixed(2)),
    median: Number(median.toFixed(2)),
    pct_below_threshold: belowThresh != null ? Number(belowThresh.toFixed(1)) : null,
    pct_below_20: Number(below20.toFixed(1))
  };
}

function downsample<T>(arr: T[], maxPoints: number): T[] {
  if (arr.length <= maxPoints) return arr;
  const stride = arr.length / maxPoints;
  const out: T[] = [];
  for (let i = 0; i < maxPoints; i++) {
    out.push(arr[Math.floor(i * stride)]);
  }
  return out;
}

async function getThresholds(epics: string[]): Promise<
  Map<string, { value: number | null; source: PairResult["threshold_source"] }>
> {
  const map = new Map<string, { value: number | null; source: PairResult["threshold_source"] }>();
  if (!epics.length) return map;

  // smc_simple_global_config has no scalp_min_adx column — per-pair only. NULL = no filter.
  const globalDefault: number | null = null;

  const res = await strategyConfigPool.query(
    `
    SELECT epic, scalp_min_adx AS col_val, parameter_overrides->>'scalp_min_adx' AS jsonb_val
    FROM smc_simple_pair_overrides
    WHERE epic = ANY($1) AND is_enabled = TRUE
    `,
    [epics]
  );

  for (const epic of epics) {
    const rows = res.rows.filter((r) => r.epic === epic);
    if (rows.length === 0) {
      map.set(epic, { value: globalDefault, source: "default" });
      continue;
    }
    // Use first row (duplicates exist in EURJPY — prefer non-null values)
    const preferred =
      rows.find((r) => r.col_val != null) ||
      rows.find((r) => r.jsonb_val != null) ||
      rows[0];
    if (preferred.col_val != null) {
      map.set(epic, { value: Number(preferred.col_val), source: "column" });
    } else if (preferred.jsonb_val != null) {
      map.set(epic, { value: Number(preferred.jsonb_val), source: "jsonb" });
    } else {
      map.set(epic, { value: globalDefault, source: "default" });
    }
  }
  return map;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const daysBack = Math.max(7, Math.min(60, Number(searchParams.get("days") || 21)));
  const epicsParam = searchParams.get("epics");
  const epics = epicsParam ? epicsParam.split(",").filter(Boolean) : DEFAULT_PAIRS;

  const now = new Date();
  const thisMonday = mondayOfWeek(now);
  const lastMonday = new Date(thisMonday);
  lastMonday.setUTCDate(thisMonday.getUTCDate() - 7);
  const queryFrom = new Date(lastMonday);
  queryFrom.setUTCDate(queryFrom.getUTCDate() - daysBack);

  try {
    const thresholds = await getThresholds(epics);

    const pairs: PairResult[] = [];

    for (const epic of epics) {
      // Resample 1m → 5m from ig_candles (continuously streamed).
      // ig_candles_backtest is only lazy-populated by backtests, so it has per-pair staleness.
      const res = await forexPool.query(
        `
        SELECT
          date_trunc('hour', start_time)
            + (floor(EXTRACT(MINUTE FROM start_time)::int / 5) * INTERVAL '5 minutes') AS start_time,
          MAX(high)  AS high,
          MIN(low)   AS low,
          (ARRAY_AGG(close ORDER BY start_time DESC))[1] AS close
        FROM ig_candles
        WHERE epic = $1 AND timeframe = 1 AND start_time >= $2
        GROUP BY 1
        ORDER BY 1
        `,
        [epic, queryFrom]
      );

      if (!res.rows.length) {
        pairs.push({
          epic,
          short: epic.split(".")[2] || epic,
          threshold: thresholds.get(epic)?.value ?? null,
          threshold_source: thresholds.get(epic)?.source ?? "default",
          series: [],
          this_week: { bars: 0, mean: null, median: null, pct_below_threshold: null, pct_below_20: null },
          last_week: { bars: 0, mean: null, median: null, pct_below_threshold: null, pct_below_20: null },
          coverage: { first_bar: null, last_bar: null }
        });
        continue;
      }

      const candles: Candle[] = res.rows.map((r) => ({
        t: new Date(r.start_time),
        h: Number(r.high),
        l: Number(r.low),
        c: Number(r.close)
      }));

      const adx = computeAdx(candles);
      const fullSeries = candles.map((c, i) => ({ t: c.t, adx: adx[i] }));
      const tInfo = thresholds.get(epic) ?? { value: null, source: "default" as const };

      const lastWeekStats = computeRangeStats(fullSeries, lastMonday, thisMonday, tInfo.value);
      const thisWeekEnd = new Date(thisMonday);
      thisWeekEnd.setUTCDate(thisMonday.getUTCDate() + 7);
      const thisWeekStats = computeRangeStats(fullSeries, thisMonday, thisWeekEnd, tInfo.value);

      const sparkSeries = fullSeries.filter((p) => p.t >= lastMonday);
      const downsampled = downsample(sparkSeries, BARS_PER_PAIR_IN_RESPONSE).map((p) => ({
        t: p.t.toISOString(),
        adx: Number.isFinite(p.adx) ? Number(p.adx.toFixed(2)) : null
      }));

      pairs.push({
        epic,
        short: epic.split(".")[2] || epic,
        threshold: tInfo.value,
        threshold_source: tInfo.source,
        series: downsampled,
        this_week: thisWeekStats,
        last_week: lastWeekStats,
        coverage: {
          first_bar: candles[0].t.toISOString(),
          last_bar: candles[candles.length - 1].t.toISOString()
        }
      });
    }

    return NextResponse.json({
      generated_at: new Date().toISOString(),
      this_week: {
        start: thisMonday.toISOString(),
        end: new Date(thisMonday.getTime() + 7 * 86400 * 1000).toISOString()
      },
      last_week: { start: lastMonday.toISOString(), end: thisMonday.toISOString() },
      pairs
    });
  } catch (error) {
    console.error("adx-regime error", error);
    return NextResponse.json({ error: "Failed to compute ADX regime" }, { status: 500 });
  }
}
