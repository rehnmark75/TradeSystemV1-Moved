import { NextRequest, NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

type PeriodKey = "A" | "B";

type SignalAgg = {
  pair: string;
  period: PeriodKey;
  signals: number;
  approved: number;
  rejected: number;
  bull: number;
  bear: number;
  avg_conf: number | null;
  avg_adx: number | null;
  avg_rsi: number | null;
  avg_atr: number | null;
  avg_spread: number | null;
  trending_count: number;
  ranging_count: number;
  high_vol_count: number;
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

type PairBlock = {
  pair: string;
  epic: string;
  pip_size: number;
  is_gold: boolean;
  a: PeriodStats;
  b: PeriodStats;
  daily_a: DailyRow[];
  daily_b: DailyRow[];
  regime_shift: "TRENDING→RANGING" | "RANGING→TRENDING" | "STABLE" | "MORE_VOLATILE" | "LESS_VOLATILE" | "NO_DATA";
  adx_delta: number | null;
};

function epicToPair(epic: string): string {
  const m = epic.match(/CS\.D\.([A-Z]+)\./);
  if (!m) return epic;
  const name = m[1];
  if (name === "CFEGOLD") return "XAUUSD";
  return name;
}

function pipSize(epic: string): number {
  if (epic.includes("CFEGOLD")) return 0.1;
  if (epic.includes("JPY")) return 0.01;
  return 0.0001;
}

function parseDate(raw: string | null, fallback: Date): Date {
  if (!raw) return fallback;
  const d = new Date(raw);
  return Number.isNaN(d.valueOf()) ? fallback : d;
}

function startOfIsoWeek(d: Date): Date {
  const copy = new Date(d);
  copy.setUTCHours(0, 0, 0, 0);
  const dow = copy.getUTCDay(); // 0=Sun..6=Sat
  const diff = (dow + 6) % 7; // Mon-origin offset
  copy.setUTCDate(copy.getUTCDate() - diff);
  return copy;
}

function addDays(d: Date, n: number): Date {
  const copy = new Date(d);
  copy.setUTCDate(copy.getUTCDate() + n);
  return copy;
}

function fmtIsoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
}

function classifyRegimeShift(
  a: PeriodStats,
  b: PeriodStats
): PairBlock["regime_shift"] {
  if (a.avg_adx == null || b.avg_adx == null) return "NO_DATA";
  const aTrending = a.avg_adx >= 20;
  const bTrending = b.avg_adx >= 20;
  if (aTrending && !bTrending) return "TRENDING→RANGING";
  if (!aTrending && bTrending) return "RANGING→TRENDING";
  // Both on same side — check volatility via daily range
  const aRange = a.avg_daily_range_pips ?? 0;
  const bRange = b.avg_daily_range_pips ?? 0;
  if (aRange > 0 && bRange > 0) {
    const ratio = bRange / aRange;
    if (ratio >= 1.3) return "MORE_VOLATILE";
    if (ratio <= 0.7) return "LESS_VOLATILE";
  }
  return "STABLE";
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const env = searchParams.get("env") || "demo";

  const now = new Date();
  const currentWeekStart = startOfIsoWeek(now);
  const prevWeekStart = addDays(currentWeekStart, -7);

  const periodAStart = parseDate(searchParams.get("aStart"), currentWeekStart);
  const periodAEnd = parseDate(
    searchParams.get("aEnd"),
    addDays(currentWeekStart, 7)
  );
  const periodBStart = parseDate(searchParams.get("bStart"), prevWeekStart);
  const periodBEnd = parseDate(searchParams.get("bEnd"), currentWeekStart);

  const earliest = new Date(
    Math.min(periodAStart.getTime(), periodBStart.getTime())
  );
  const latest = new Date(Math.max(periodAEnd.getTime(), periodBEnd.getTime()));

  const client = await forexPool.connect();

  try {
    // Signal aggregates per pair per period
    const signalQuery = `
      SELECT
        epic,
        pair,
        CASE
          WHEN alert_timestamp >= $1 AND alert_timestamp < $2 THEN 'A'
          WHEN alert_timestamp >= $3 AND alert_timestamp < $4 THEN 'B'
        END AS period,
        COUNT(*)::int AS signals,
        COUNT(*) FILTER (WHERE claude_approved = true)::int AS approved,
        COUNT(*) FILTER (WHERE claude_approved = false)::int AS rejected,
        COUNT(*) FILTER (WHERE signal_type IN ('BULL','BUY','LONG'))::int AS bull,
        COUNT(*) FILTER (WHERE signal_type IN ('BEAR','SELL','SHORT'))::int AS bear,
        AVG(confidence_score)::float AS avg_conf,
        AVG(adx)::float AS avg_adx,
        AVG(rsi)::float AS avg_rsi,
        AVG(atr)::float AS avg_atr,
        AVG(spread_pips)::float AS avg_spread,
        COUNT(*) FILTER (WHERE market_regime = 'trending')::int AS trending_count,
        COUNT(*) FILTER (WHERE market_regime = 'ranging')::int AS ranging_count,
        COUNT(*) FILTER (WHERE market_regime IN ('high_volatility','breakout'))::int AS high_vol_count
      FROM alert_history
      WHERE environment = $5
        AND (
          (alert_timestamp >= $1 AND alert_timestamp < $2) OR
          (alert_timestamp >= $3 AND alert_timestamp < $4)
        )
      GROUP BY epic, pair, period
    `;

    const signalRes = await client.query(signalQuery, [
      periodAStart.toISOString(),
      periodAEnd.toISOString(),
      periodBStart.toISOString(),
      periodBEnd.toISOString(),
      env,
    ]);

    // Daily candle stats per epic over both periods
    const candleQuery = `
      WITH hourly AS (
        SELECT
          epic,
          date_trunc('hour', start_time) AS hr,
          MIN(low) AS lo,
          MAX(high) AS hi,
          (array_agg(open ORDER BY start_time))[1] AS o,
          (array_agg(close ORDER BY start_time DESC))[1] AS c
        FROM ig_candles
        WHERE timeframe = 1
          AND start_time >= $1
          AND start_time < $2
        GROUP BY epic, date_trunc('hour', start_time)
      )
      SELECT
        epic,
        hr::date AS d,
        (array_agg(o ORDER BY hr))[1]::float AS open,
        (array_agg(c ORDER BY hr DESC))[1]::float AS close,
        MIN(lo)::float AS low,
        MAX(hi)::float AS high
      FROM hourly
      GROUP BY epic, hr::date
      ORDER BY epic, d
    `;

    const candleRes = await client.query(candleQuery, [
      earliest.toISOString(),
      latest.toISOString(),
    ]);

    // Gather all epics observed (union of signals + candles)
    const epicSet = new Set<string>();
    signalRes.rows.forEach((r) => epicSet.add(r.epic));
    candleRes.rows.forEach((r) => epicSet.add(r.epic));

    // Group daily rows by epic, classify into A/B period, and compute move/range pips
    const dailyByEpic = new Map<string, DailyRow[]>();
    for (const row of candleRes.rows) {
      const epic: string = row.epic;
      const pip = pipSize(epic);
      const dStr = new Date(row.d).toISOString().slice(0, 10);
      const dObj = new Date(dStr + "T00:00:00Z");
      const dow = dObj.toLocaleDateString("en-US", {
        weekday: "short",
        timeZone: "UTC",
      });
      const netPips = ((row.close - row.open) / pip);
      const rangePips = ((row.high - row.low) / pip);
      const daily: DailyRow = {
        epic,
        d: dStr,
        dow,
        open: row.open,
        close: row.close,
        low: row.low,
        high: row.high,
        net_pips: Math.round(netPips * 10) / 10,
        range_pips: Math.round(rangePips * 10) / 10,
      };
      if (!dailyByEpic.has(epic)) dailyByEpic.set(epic, []);
      dailyByEpic.get(epic)!.push(daily);
    }

    // Helper: is a date string inside a period?
    const dateInPeriod = (dStr: string, start: Date, end: Date): boolean => {
      const t = new Date(dStr + "T12:00:00Z").getTime();
      return t >= start.getTime() && t < end.getTime();
    };

    // Index signal rows by (epic, period)
    const signalMap = new Map<string, SignalAgg>();
    for (const row of signalRes.rows) {
      if (!row.period) continue;
      const key = `${row.epic}|${row.period}`;
      signalMap.set(key, row as SignalAgg);
    }

    const pairs: PairBlock[] = [];
    for (const epic of epicSet) {
      const dailyRows = (dailyByEpic.get(epic) || []).sort((x, y) =>
        x.d.localeCompare(y.d)
      );
      const dailyA = dailyRows.filter((d) =>
        dateInPeriod(d.d, periodAStart, periodAEnd)
      );
      const dailyB = dailyRows.filter((d) =>
        dateInPeriod(d.d, periodBStart, periodBEnd)
      );

      const buildStats = (
        sig: SignalAgg | undefined,
        daily: DailyRow[]
      ): PeriodStats => {
        const signals = sig?.signals ?? 0;
        const approved = sig?.approved ?? 0;
        const rejected = sig?.rejected ?? 0;
        const decided = approved + rejected;
        const avgRange = daily.length
          ? daily.reduce((s, r) => s + r.range_pips, 0) / daily.length
          : null;
        const avgAbsMove = daily.length
          ? daily.reduce((s, r) => s + Math.abs(r.net_pips), 0) / daily.length
          : null;
        const cumulative = daily.length
          ? daily.reduce((s, r) => s + r.net_pips, 0)
          : null;
        const periodLow = daily.length ? Math.min(...daily.map((r) => r.low)) : null;
        const periodHigh = daily.length ? Math.max(...daily.map((r) => r.high)) : null;
        const pip = pipSize(epic);
        const periodRangePips =
          periodLow != null && periodHigh != null
            ? Math.round(((periodHigh - periodLow) / pip) * 10) / 10
            : null;
        const trendDays = daily.filter(
          (r) => Math.abs(r.net_pips) >= 40 && epicToPair(epic) !== "XAUUSD"
        ).length + daily.filter(
          (r) => epicToPair(epic) === "XAUUSD" && Math.abs(r.net_pips) >= 100
        ).length;

        let dominantRegime: string | null = null;
        if (sig) {
          const counts: Array<[string, number]> = [
            ["trending", sig.trending_count],
            ["ranging", sig.ranging_count],
            ["high_volatility", sig.high_vol_count],
          ];
          counts.sort((a, b) => b[1] - a[1]);
          if (counts[0][1] > 0) dominantRegime = counts[0][0];
        }

        return {
          signals,
          approved,
          rejected,
          approval_rate: decided > 0 ? (approved / decided) * 100 : null,
          bull: sig?.bull ?? 0,
          bear: sig?.bear ?? 0,
          avg_conf: sig?.avg_conf ?? null,
          avg_adx: sig?.avg_adx ?? null,
          avg_rsi: sig?.avg_rsi ?? null,
          avg_atr: sig?.avg_atr ?? null,
          avg_spread: sig?.avg_spread ?? null,
          dominant_regime: dominantRegime,
          avg_daily_range_pips: avgRange != null ? Math.round(avgRange * 10) / 10 : null,
          avg_abs_daily_move_pips:
            avgAbsMove != null ? Math.round(avgAbsMove * 10) / 10 : null,
          cumulative_move_pips:
            cumulative != null ? Math.round(cumulative * 10) / 10 : null,
          trend_days: trendDays,
          trading_days: daily.length,
          period_range_pips: periodRangePips,
        };
      };

      const sigA = signalMap.get(`${epic}|A`);
      const sigB = signalMap.get(`${epic}|B`);
      const statsA = buildStats(sigA, dailyA);
      const statsB = buildStats(sigB, dailyB);

      const regimeShift = classifyRegimeShift(statsB, statsA); // "B→A" direction
      const adxDelta =
        statsA.avg_adx != null && statsB.avg_adx != null
          ? Math.round((statsA.avg_adx - statsB.avg_adx) * 10) / 10
          : null;

      pairs.push({
        pair: epicToPair(epic),
        epic,
        pip_size: pipSize(epic),
        is_gold: epic.includes("CFEGOLD"),
        a: statsA,
        b: statsB,
        daily_a: dailyA,
        daily_b: dailyB,
        regime_shift: regimeShift,
        adx_delta: adxDelta,
      });
    }

    // Sort by biggest absolute ADX shift first (most interesting to inspect)
    pairs.sort((x, y) => {
      const xd = Math.abs(x.adx_delta ?? 0);
      const yd = Math.abs(y.adx_delta ?? 0);
      return yd - xd;
    });

    return NextResponse.json({
      generated_at: new Date().toISOString(),
      env,
      period_a: {
        start: periodAStart.toISOString(),
        end: periodAEnd.toISOString(),
        label: fmtIsoDate(periodAStart) + " → " + fmtIsoDate(addDays(periodAEnd, -1)),
      },
      period_b: {
        start: periodBStart.toISOString(),
        end: periodBEnd.toISOString(),
        label: fmtIsoDate(periodBStart) + " → " + fmtIsoDate(addDays(periodBEnd, -1)),
      },
      pairs,
    });
  } catch (err) {
    console.error("[market-conditions] error", err);
    return NextResponse.json(
      { error: "Failed to load market conditions", detail: String(err) },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
