import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";
import { strategyConfigPool } from "../../../../lib/strategyConfigDb";

export const dynamic = "force-dynamic";

const DEFAULT_WINDOW_HOURS = 24;
const TIMEOUT_HOURS = 14 * 24; // 14 days — force-close lingering trailed trades at their locked stop
const DEFAULT_SL_PIPS = 10;
const DEFAULT_TP_PIPS = 15;

type TrailingConfig = {
  break_even_trigger_points: number | null;
  early_breakeven_trigger_points: number | null;
  early_breakeven_buffer_points: number | null;
  stage1_trigger_points: number | null;
  stage1_lock_points: number | null;
  stage2_trigger_points: number | null;
  stage2_lock_points: number | null;
  stage3_trigger_points: number | null;
  stage3_atr_multiplier: number | null;
  stage3_min_distance: number | null;
  min_trail_distance: number | null;
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
};

function pipSize(epic: string): number {
  if (epic.includes("CFEGOLD")) return 0.1;
  if (epic.includes("JPY")) return 0.01;
  return 0.0001;
}

function parseWindow(value: string | null): number {
  if (!value) return DEFAULT_WINDOW_HOURS;
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_WINDOW_HOURS;
  return Math.min(n, 24 * 90);
}

type SimResult = {
  outcome: "WIN" | "LOSS" | "BREAKEVEN" | "TRAILED" | "TRAILED_FORCED" | "STILL_OPEN" | "NO_DATA";
  pips: number;
  minutes_to_resolve: number | null;
  max_favorable_pips: number;
  max_adverse_pips: number;
  exit_reason: string;
  stop_path: Array<{ minutes: number; stop_pips_from_entry: number; label: string }>;
  locked_pips: number;
  alert_age_hours: number;
};

function simulateTrade(
  row: RejectionRow,
  slPips: number,
  tpPips: number,
  trailing: TrailingConfig | null,
  candles: Array<{ start_time: Date; high: number; low: number }>,
  alertTime: Date
): SimResult {
  const pip = pipSize(row.epic);
  const entry = row.price;
  const isLong = ["BULL", "BUY", "LONG"].includes(row.signal_type.toUpperCase());
  const tp = isLong ? entry + tpPips * pip : entry - tpPips * pip;

  // Stop is expressed as "pips from entry" in our favor; initial = -slPips (i.e. we lose slPips if hit)
  let stopLockPips = -slPips;
  let stopLabel = "initial";
  let peakMfe = 0;
  let peakAdv = 0;
  let stopPath: Array<{ minutes: number; stop_pips_from_entry: number; label: string }> = [
    { minutes: 0, stop_pips_from_entry: stopLockPips, label: stopLabel },
  ];

  const hasTrailing = !!trailing;
  const beTrigger = trailing?.break_even_trigger_points ?? null;
  const earlyBeTrigger = trailing?.early_breakeven_trigger_points ?? null;
  const earlyBeBuffer = trailing?.early_breakeven_buffer_points ?? 0;
  const stage1Trigger = trailing?.stage1_trigger_points ?? null;
  const stage1Lock = trailing?.stage1_lock_points ?? 0;
  const stage2Trigger = trailing?.stage2_trigger_points ?? null;
  const stage2Lock = trailing?.stage2_lock_points ?? 0;
  const stage3Trigger = trailing?.stage3_trigger_points ?? null;
  const minTrailDistance = trailing?.min_trail_distance ?? 5;

  const nowMs = Date.now();
  const alertAgeHours = (nowMs - alertTime.getTime()) / 3600000;

  if (!candles.length) {
    return {
      outcome: "NO_DATA",
      pips: 0,
      minutes_to_resolve: null,
      max_favorable_pips: 0,
      max_adverse_pips: 0,
      exit_reason: "no candle data",
      stop_path: stopPath,
      locked_pips: stopLockPips,
      alert_age_hours: Math.round(alertAgeHours * 10) / 10,
    };
  }

  for (const c of candles) {
    const minutes = Math.round((c.start_time.getTime() - alertTime.getTime()) / 60000);
    const favorableThisBar = isLong ? (c.high - entry) / pip : (entry - c.low) / pip;
    const adverseThisBar = isLong ? (c.low - entry) / pip : (entry - c.high) / pip;

    // Check stop hit FIRST (conservative — assume adverse move came before favorable)
    const stopHitPrice = isLong ? entry + stopLockPips * pip : entry - stopLockPips * pip;
    const hitStop = isLong ? c.low <= stopHitPrice : c.high >= stopHitPrice;
    const hitTp = isLong ? c.high >= tp : c.low <= tp;

    if (hitStop && hitTp) {
      const exitReason =
        stopLockPips >= 0
          ? `stop locked at +${stopLockPips} pips (trailed, bar ambiguous)`
          : `initial SL at -${slPips} pips (bar ambiguous)`;
      return {
        outcome: stopLockPips > 0 ? "TRAILED" : stopLockPips === 0 ? "BREAKEVEN" : "LOSS",
        pips: stopLockPips,
        minutes_to_resolve: minutes,
        max_favorable_pips: Math.round(Math.max(peakMfe, favorableThisBar) * 10) / 10,
        max_adverse_pips: Math.round(Math.min(peakAdv, adverseThisBar) * 10) / 10,
        exit_reason: exitReason,
        stop_path: stopPath,
        locked_pips: stopLockPips,
        alert_age_hours: Math.round(alertAgeHours * 10) / 10,
      };
    }

    if (hitStop) {
      return {
        outcome: stopLockPips > 0 ? "TRAILED" : stopLockPips === 0 ? "BREAKEVEN" : "LOSS",
        pips: stopLockPips,
        minutes_to_resolve: minutes,
        max_favorable_pips: Math.round(Math.max(peakMfe, favorableThisBar) * 10) / 10,
        max_adverse_pips: Math.round(Math.min(peakAdv, adverseThisBar) * 10) / 10,
        exit_reason: `stop hit @ ${stopLabel} (${stopLockPips >= 0 ? "+" : ""}${stopLockPips} pips)`,
        stop_path: stopPath,
        locked_pips: stopLockPips,
        alert_age_hours: Math.round(alertAgeHours * 10) / 10,
      };
    }

    if (hitTp) {
      return {
        outcome: "WIN",
        pips: tpPips,
        minutes_to_resolve: minutes,
        max_favorable_pips: Math.round(Math.max(peakMfe, tpPips) * 10) / 10,
        max_adverse_pips: Math.round(Math.min(peakAdv, adverseThisBar) * 10) / 10,
        exit_reason: `TP hit (+${tpPips} pips)`,
        stop_path: stopPath,
        locked_pips: stopLockPips,
        alert_age_hours: Math.round(alertAgeHours * 10) / 10,
      };
    }

    // No exit — update peaks and potentially trail the stop
    peakMfe = Math.max(peakMfe, favorableThisBar);
    peakAdv = Math.min(peakAdv, adverseThisBar);

    if (hasTrailing) {
      let newLock = stopLockPips;
      let newLabel = stopLabel;

      if (earlyBeTrigger !== null && peakMfe >= earlyBeTrigger && newLock < earlyBeBuffer) {
        newLock = earlyBeBuffer;
        newLabel = "early_be";
      }
      if (beTrigger !== null && peakMfe >= beTrigger && newLock < 0) {
        newLock = 0;
        newLabel = "breakeven";
      }
      if (stage1Trigger !== null && peakMfe >= stage1Trigger && newLock < stage1Lock) {
        newLock = stage1Lock;
        newLabel = "stage1";
      }
      if (stage2Trigger !== null && peakMfe >= stage2Trigger && newLock < stage2Lock) {
        newLock = stage2Lock;
        newLabel = "stage2";
      }
      if (stage3Trigger !== null && peakMfe >= stage3Trigger) {
        const trailCandidate = peakMfe - minTrailDistance;
        if (trailCandidate > newLock) {
          newLock = trailCandidate;
          newLabel = "stage3_trail";
        }
      }

      if (newLock !== stopLockPips || newLabel !== stopLabel) {
        stopLockPips = newLock;
        stopLabel = newLabel;
        stopPath.push({
          minutes,
          stop_pips_from_entry: Math.round(stopLockPips * 10) / 10,
          label: stopLabel,
        });
      }
    }
  }

  // End of walk. For alerts older than 48h, a realistic assumption is the trade resolved long ago
  // (broker or operator closed the position). Force-close at the current locked stop — the guaranteed
  // outcome given the stop trailed to that level. For alerts <48h old, treat as genuinely live.
  const candleSpanHours =
    candles.length > 0
      ? (candles[candles.length - 1].start_time.getTime() - alertTime.getTime()) / 3600000
      : 0;
  const OLD_ALERT_THRESHOLD_HOURS = 48;
  const isOldAlert = alertAgeHours >= OLD_ALERT_THRESHOLD_HOURS;

  if (isOldAlert) {
    if (stopLockPips > 0) {
      return {
        outcome: "TRAILED_FORCED",
        pips: stopLockPips,
        minutes_to_resolve: Math.round(candleSpanHours * 60),
        max_favorable_pips: Math.round(peakMfe * 10) / 10,
        max_adverse_pips: Math.round(peakAdv * 10) / 10,
        exit_reason: `trailed +${stopLockPips} pips over ${Math.round(candleSpanHours / 24)}d — forced close at locked stop (alert is ${Math.round(alertAgeHours / 24)}d old)`,
        stop_path: stopPath,
        locked_pips: stopLockPips,
        alert_age_hours: Math.round(alertAgeHours * 10) / 10,
      };
    }
    if (stopLockPips === 0) {
      return {
        outcome: "BREAKEVEN",
        pips: 0,
        minutes_to_resolve: Math.round(candleSpanHours * 60),
        max_favorable_pips: Math.round(peakMfe * 10) / 10,
        max_adverse_pips: Math.round(peakAdv * 10) / 10,
        exit_reason: `stop advanced to breakeven, TP never hit — forced flat close (alert is ${Math.round(alertAgeHours / 24)}d old)`,
        stop_path: stopPath,
        locked_pips: stopLockPips,
        alert_age_hours: Math.round(alertAgeHours * 10) / 10,
      };
    }
    // Stop never advanced in an old alert — drifted around without resolving. Treat as expired (no outcome).
    return {
      outcome: "STILL_OPEN",
      pips: 0,
      minutes_to_resolve: null,
      max_favorable_pips: Math.round(peakMfe * 10) / 10,
      max_adverse_pips: Math.round(peakAdv * 10) / 10,
      exit_reason: `no resolve in ${Math.round(candleSpanHours / 24)}d — stop never advanced, TP never hit`,
      stop_path: stopPath,
      locked_pips: stopLockPips,
      alert_age_hours: Math.round(alertAgeHours * 10) / 10,
    };
  }

  // Recent alert (<48h old) — trade may genuinely still be live
  return {
    outcome: "STILL_OPEN",
    pips: 0,
    minutes_to_resolve: null,
    max_favorable_pips: Math.round(peakMfe * 10) / 10,
    max_adverse_pips: Math.round(peakAdv * 10) / 10,
    exit_reason: `trade still live (${Math.round(alertAgeHours)}h old, ${stopLabel} stop @ ${stopLockPips >= 0 ? "+" : ""}${Math.round(stopLockPips * 10) / 10} pips)`,
    stop_path: stopPath,
    locked_pips: stopLockPips,
    alert_age_hours: Math.round(alertAgeHours * 10) / 10,
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const windowHours = parseWindow(searchParams.get("window"));
  const env = searchParams.get("env") || "demo";
  const epic = searchParams.get("epic");
  const useTrailing = searchParams.get("trailing") !== "false";

  try {
    const rejectionParams: Array<string | number> = [windowHours, env];
    let rejectionWhere =
      "claude_decision = 'REJECT' AND alert_timestamp > NOW() - ($1::int || ' hours')::interval AND environment = $2";
    if (epic && epic !== "All") {
      rejectionParams.push(epic);
      rejectionWhere += ` AND epic = $${rejectionParams.length}`;
    }

    const [rejectionsResult, pairConfigResult, trailingConfigResult, pairsResult] =
      await Promise.all([
        forexPool.query<RejectionRow>(
          `
          SELECT id, alert_timestamp, epic, pair, signal_type, price,
                 confidence_score, claude_score, claude_reason,
                 market_session, market_regime, environment
          FROM alert_history
          WHERE ${rejectionWhere}
          ORDER BY alert_timestamp DESC
          `,
          rejectionParams
        ),
        strategyConfigPool.query(
          `
          SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips
          FROM smc_simple_pair_overrides
          WHERE fixed_stop_loss_pips IS NOT NULL AND fixed_take_profit_pips IS NOT NULL
          `
        ),
        strategyConfigPool.query(
          `
          SELECT epic, is_scalp, break_even_trigger_points, early_breakeven_trigger_points,
                 early_breakeven_buffer_points, stage1_trigger_points, stage1_lock_points,
                 stage2_trigger_points, stage2_lock_points, stage3_trigger_points,
                 stage3_atr_multiplier, stage3_min_distance, min_trail_distance
          FROM trailing_pair_config
          WHERE is_active = TRUE AND config_set = $1
          `,
          [env]
        ),
        forexPool.query(
          `
          SELECT DISTINCT epic
          FROM alert_history
          WHERE claude_decision = 'REJECT' AND environment = $1
          ORDER BY epic
          `,
          [env]
        ),
      ]);

    const pairConfigs = new Map<string, { sl: number; tp: number }>();
    for (const row of pairConfigResult.rows) {
      const key = row.epic;
      if (!pairConfigs.has(key)) {
        pairConfigs.set(key, {
          sl: Number(row.fixed_stop_loss_pips),
          tp: Number(row.fixed_take_profit_pips),
        });
      }
    }

    const trailingConfigs = new Map<string, TrailingConfig>();
    for (const row of trailingConfigResult.rows) {
      if (row.is_scalp) continue;
      const cfg: TrailingConfig = {
        break_even_trigger_points: row.break_even_trigger_points,
        early_breakeven_trigger_points: row.early_breakeven_trigger_points,
        early_breakeven_buffer_points: row.early_breakeven_buffer_points,
        stage1_trigger_points: row.stage1_trigger_points,
        stage1_lock_points: row.stage1_lock_points,
        stage2_trigger_points: row.stage2_trigger_points,
        stage2_lock_points: row.stage2_lock_points,
        stage3_trigger_points: row.stage3_trigger_points,
        stage3_atr_multiplier: row.stage3_atr_multiplier ? Number(row.stage3_atr_multiplier) : null,
        stage3_min_distance: row.stage3_min_distance,
        min_trail_distance: row.min_trail_distance,
      };
      trailingConfigs.set(row.epic, cfg);
    }

    const rejections = rejectionsResult.rows;

    const results = await Promise.all(
      rejections.map(async (row) => {
        const alertTime = new Date(row.alert_timestamp);
        const endTime = new Date(alertTime.getTime() + TIMEOUT_HOURS * 3600 * 1000);
        const pairCfg = pairConfigs.get(row.epic) || { sl: DEFAULT_SL_PIPS, tp: DEFAULT_TP_PIPS };
        const trailing = useTrailing ? trailingConfigs.get(row.epic) || null : null;

        // Prefer 1m; fall back to 5m when 1m unavailable
        const candles1m = await forexPool.query(
          `
          SELECT start_time, high, low FROM ig_candles
          WHERE epic = $1 AND timeframe = 1
            AND start_time >= $2 AND start_time <= $3
          ORDER BY start_time ASC
          `,
          [row.epic, alertTime, endTime]
        );
        let candles = candles1m.rows;
        let resolution: "1m" | "5m" | "none" = "1m";
        if (!candles.length) {
          const candles5m = await forexPool.query(
            `
            SELECT start_time, high, low FROM ig_candles
            WHERE epic = $1 AND timeframe = 5
              AND start_time >= $2 AND start_time <= $3
            ORDER BY start_time ASC
            `,
            [row.epic, alertTime, endTime]
          );
          candles = candles5m.rows;
          resolution = candles.length ? "5m" : "none";
        }

        const sim = simulateTrade(
          row,
          pairCfg.sl,
          pairCfg.tp,
          trailing,
          candles.map((c: any) => ({
            start_time: new Date(c.start_time),
            high: Number(c.high),
            low: Number(c.low),
          })),
          alertTime
        );

        return {
          ...row,
          price: Number(row.price),
          confidence_score: row.confidence_score == null ? null : Number(row.confidence_score),
          sl_pips: pairCfg.sl,
          tp_pips: pairCfg.tp,
          trailing_used: !!trailing,
          candle_resolution: resolution,
          ...sim,
        };
      })
    );

    // Aggregate stats — TRAILED_FORCED counts as a resolved trailed profit
    const resolved = results.filter((r) =>
      ["WIN", "LOSS", "BREAKEVEN", "TRAILED", "TRAILED_FORCED"].includes(r.outcome)
    );
    const wins = resolved.filter((r) => r.outcome === "WIN");
    const trailed = resolved.filter(
      (r) => r.outcome === "TRAILED" || r.outcome === "TRAILED_FORCED"
    );
    const breakevens = resolved.filter((r) => r.outcome === "BREAKEVEN");
    const losses = resolved.filter((r) => r.outcome === "LOSS");
    const stillOpen = results.filter((r) => r.outcome === "STILL_OPEN");
    const noData = results.filter((r) => r.outcome === "NO_DATA");

    const netPips = resolved.reduce((s, r) => s + r.pips, 0);
    const grossWin = resolved.filter((r) => r.pips > 0).reduce((s, r) => s + r.pips, 0);
    const grossLoss = Math.abs(
      resolved.filter((r) => r.pips < 0).reduce((s, r) => s + r.pips, 0)
    );
    const profitFactor = grossLoss > 0 ? grossWin / grossLoss : null;
    const winRate = resolved.length > 0 ? (wins.length / resolved.length) * 100 : 0;
    const verdict =
      !resolved.length ? "INSUFFICIENT_DATA" : netPips < 0 ? "CLAUDE_RIGHT" : netPips > 0 ? "CLAUDE_WRONG" : "NEUTRAL";

    // By pair
    const pairAgg = new Map<
      string,
      { n: number; wins: number; losses: number; trailed: number; breakevens: number; pips: number }
    >();
    for (const r of resolved) {
      const cur = pairAgg.get(r.epic) || { n: 0, wins: 0, losses: 0, trailed: 0, breakevens: 0, pips: 0 };
      cur.n += 1;
      cur.pips += r.pips;
      if (r.outcome === "WIN") cur.wins += 1;
      else if (r.outcome === "LOSS") cur.losses += 1;
      else if (r.outcome === "TRAILED" || r.outcome === "TRAILED_FORCED") cur.trailed += 1;
      else if (r.outcome === "BREAKEVEN") cur.breakevens += 1;
      pairAgg.set(r.epic, cur);
    }
    const byPair = Array.from(pairAgg.entries())
      .map(([epicKey, v]) => ({
        epic: epicKey,
        n: v.n,
        wins: v.wins,
        losses: v.losses,
        trailed: v.trailed,
        breakevens: v.breakevens,
        net_pips: Math.round(v.pips * 10) / 10,
        win_rate: v.n > 0 ? (v.wins / v.n) * 100 : 0,
        verdict: v.pips < 0 ? "CLAUDE_RIGHT" : v.pips > 0 ? "CLAUDE_WRONG" : "NEUTRAL",
      }))
      .sort((a, b) => b.n - a.n);

    return NextResponse.json({
      params: {
        window_hours: windowHours,
        env,
        epic: epic || "All",
        trailing: useTrailing,
        timeout_hours: TIMEOUT_HOURS,
      },
      filters: {
        epics: ["All", ...pairsResult.rows.map((r) => r.epic)],
        trailing_available: useTrailing
          ? Array.from(trailingConfigs.keys()).sort()
          : [],
      },
      stats: {
        total: results.length,
        resolved: resolved.length,
        wins: wins.length,
        losses: losses.length,
        trailed: trailed.length,
        breakevens: breakevens.length,
        still_open: stillOpen.length,
        no_data: noData.length,
        net_pips: Math.round(netPips * 10) / 10,
        gross_win: Math.round(grossWin * 10) / 10,
        gross_loss: Math.round(grossLoss * 10) / 10,
        profit_factor: profitFactor === null ? null : Math.round(profitFactor * 100) / 100,
        win_rate: Math.round(winRate * 10) / 10,
        verdict,
      },
      by_pair: byPair,
      rejections: results,
    });
  } catch (error) {
    console.error("Failed to load Claude rejection analysis", error);
    return NextResponse.json(
      { error: "Failed to load Claude rejection analysis" },
      { status: 500 }
    );
  }
}
