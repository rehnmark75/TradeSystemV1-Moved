import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 7;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

function normalizeSymbol(symbol: string | null) {
  if (!symbol) return "";
  return symbol
    .replace("CS.D.", "")
    .replace(".MINI.IP", "")
    .replace(".CEEM.IP", "");
}

function deriveResult(profitLoss: number | null, status: string | null) {
  if (profitLoss != null) {
    if (profitLoss > 0) return "WIN";
    if (profitLoss < 0) return "LOSS";
    return "BREAKEVEN";
  }
  if (status === "tracking") return "OPEN";
  return "PENDING";
}

function parseNumeric(value: unknown) {
  if (value == null) return null;
  if (value === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return null;
  return parsed;
}

function deriveSettingsContext(row: any) {
  const strategy = String(row.strategy || "UNKNOWN").toUpperCase();
  const items =
    strategy === "SMC_SIMPLE"
      ? [
          ["Entry TF", row.entry_timeframe],
          ["Order", row.order_type],
          ["Pullback", parseNumeric(row.pullback_depth)],
          ["Fib Zone", row.fib_zone],
          ["Limit Offset", parseNumeric(row.limit_offset_pips)],
          ["RR", parseNumeric(row.risk_reward_ratio)]
        ]
      : strategy === "RANGE_FADE"
      ? [
          ["HTF Bias", row.range_htf_bias],
          ["RSI", parseNumeric(row.range_rsi)],
          ["Band Width", parseNumeric(row.range_band_width_pips)],
          ["Dist Low", parseNumeric(row.range_distance_to_low_pips)],
          ["Dist High", parseNumeric(row.range_distance_to_high_pips)]
        ]
      : [
          ["Trigger Type", row.trigger_type],
          ["Pattern", row.pattern_type],
          ["RSI", parseNumeric(row.rsi)],
          ["ADX", parseNumeric(row.adx)]
        ];

  return items
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value]) => ({ label, value }));
}

function average(values: unknown[]) {
  const finite = values
    .map((value) => parseNumeric(value))
    .filter((value): value is number => value !== null);
  if (!finite.length) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function summarizeTrades(
  trades: any[],
  keyFn: (trade: any) => Record<string, string>
) {
  const groups = new Map<string, any[]>();
  const labels = new Map<string, Record<string, string>>();

  trades
    .filter((trade) => ["WIN", "LOSS"].includes(trade.result))
    .forEach((trade) => {
      const label = keyFn(trade);
      const key = JSON.stringify(label);
      labels.set(key, label);
      groups.set(key, [...(groups.get(key) ?? []), trade]);
    });

  return [...groups.entries()]
    .map(([key, rows]) => {
      const wins = rows.filter((trade) => trade.result === "WIN").length;
      const losses = rows.filter((trade) => trade.result === "LOSS").length;
      const totalPnl = rows.reduce((sum, trade) => sum + (trade.profit_loss ?? 0), 0);
      const zeroMfeCount = rows.filter((trade) => trade.zero_mfe).length;
      return {
        ...(labels.get(key) ?? {}),
        total_trades: rows.length,
        wins,
        losses,
        win_rate: rows.length ? (wins / rows.length) * 100 : 0,
        avg_pnl: rows.length ? totalPnl / rows.length : 0,
        total_pnl: totalPnl,
        avg_mae_pips: average(rows.map((trade) => trade.mae_pips)),
        avg_mfe_pips: average(rows.map((trade) => trade.mfe_pips)),
        zero_mfe_count: zeroMfeCount,
        zero_mfe_pct: rows.length ? (zeroMfeCount / rows.length) * 100 : 0,
        avg_confidence: average(rows.map((trade) => trade.confidence_score)),
        avg_pullback_depth: average(rows.map((trade) => trade.pullback_depth)),
        avg_rr: average(rows.map((trade) => trade.risk_reward_ratio)),
        avg_spread_pips: average(rows.map((trade) => trade.spread_pips))
      };
    })
    .sort((a, b) => b.total_trades - a.total_trades || String((a as any).strategy ?? "").localeCompare(String((b as any).strategy ?? "")));
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const env = searchParams.get("env") || "demo";
  const strategyFilter = searchParams.get("strategy");
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    const strategyOptionsResult = await forexPool.query(
      `
      SELECT DISTINCT UPPER(COALESCE(a.strategy, 'UNKNOWN')) as strategy
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'tracking', 'expired')
      AND t.environment = $2
      ORDER BY 1
      `,
      [since, env]
    );

    const tradesResult = await forexPool.query(
      `
      SELECT
        t.id,
        t.symbol,
        t.direction,
        t.entry_price,
        t.timestamp as trade_timestamp,
        t.status,
        t.profit_loss,
        t.vsl_peak_profit_pips as mfe_pips,
        t.vsl_mae_pips as mae_pips,
        t.vsl_mae_timestamp as mae_timestamp,
        t.virtual_sl_pips,
        t.vsl_stage,
        t.closed_at,
        t.is_scalp_trade,
        a.id as alert_id,
        COALESCE(a.strategy, 'UNKNOWN') as strategy,
        a.alert_timestamp as signal_timestamp,
        a.price as signal_price,
        a.confidence_score,
        a.signal_trigger,
        a.trigger_type,
        CASE
          WHEN UPPER(COALESCE(a.strategy, '')) = 'SMC_SIMPLE'
            THEN COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', NULLIF(a.signal_trigger, ''), 'SMC_ENTRY')
          WHEN UPPER(COALESCE(a.strategy, '')) = 'RANGE_FADE'
            THEN CASE WHEN UPPER(a.signal_type) = 'BUY' THEN 'LOW_FADE'
                      WHEN UPPER(a.signal_type) = 'SELL' THEN 'HIGH_FADE'
                      ELSE 'RANGE_FADE' END
          ELSE COALESCE(NULLIF(a.signal_trigger, ''), NULLIF(a.trigger_type, ''), NULLIF(a.pattern_type, ''), 'UNCLASSIFIED')
        END as entry_type,
        CASE
          WHEN UPPER(COALESCE(a.strategy, '')) = 'SMC_SIMPLE' THEN 'SMC Tier 3'
          WHEN UPPER(COALESCE(a.strategy, '')) = 'RANGE_FADE' THEN 'Range Fade'
          WHEN a.strategy IS NULL THEN 'Unlinked Alert'
          ELSE 'Generic Strategy'
        END as setup_family,
        a.strategy_indicators->'tier3_entry'->>'order_type' as order_type,
        a.strategy_indicators->'tier3_entry'->>'timeframe' as entry_timeframe,
        a.strategy_indicators->'tier3_entry'->>'fib_zone' as fib_zone,
        a.strategy_indicators->'tier3_entry'->>'limit_offset_pips' as limit_offset_pips,
        a.strategy_indicators->'tier3_entry'->>'pullback_depth' as pullback_depth,
        a.strategy_indicators->'tier3_entry'->>'in_optimal_zone' as in_optimal_zone,
        a.strategy_indicators->>'rsi' as range_rsi,
        a.strategy_indicators->>'htf_bias' as range_htf_bias,
        a.strategy_indicators->>'band_width_pips' as range_band_width_pips,
        a.strategy_indicators->>'distance_to_low_pips' as range_distance_to_low_pips,
        a.strategy_indicators->>'distance_to_high_pips' as range_distance_to_high_pips,
        a.strategy_indicators->>'sweep_side' as rs_sweep_side,
        a.strategy_indicators->>'wick_ratio' as rs_wick_ratio,
        a.strategy_indicators->>'range_top' as rs_range_top,
        a.strategy_indicators->>'range_bottom' as rs_range_bottom,
        a.strategy_indicators->>'ob_count' as rs_ob_count,
        a.pattern_type,
        a.pattern_strength,
        a.rsi,
        a.adx,
        a.risk_reward_ratio,
        a.spread_pips,
        a.timeframe,
        a.rsi_divergence_detected,
        a.rsi_divergence,
        a.htf_candle_direction,
        a.market_session
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'tracking', 'expired')
      AND t.environment = $2
      AND ($3::text IS NULL OR UPPER(COALESCE(a.strategy, 'UNKNOWN')) = UPPER($3::text))
      ORDER BY t.timestamp DESC
      `,
      [since, env, strategyFilter && strategyFilter !== "ALL" ? strategyFilter : null]
    );

    const trades = (tradesResult.rows ?? []).map((row) => {
      const profitLoss = row.profit_loss == null ? null : Number(row.profit_loss);
      const entryPrice = row.entry_price == null ? null : Number(row.entry_price);
      const signalPrice = row.signal_price == null ? null : Number(row.signal_price);
      const mfePips = row.mfe_pips == null ? null : Number(row.mfe_pips);
      const maePips = row.mae_pips == null ? null : Number(row.mae_pips);
      const symbol = row.symbol ?? "";
      const direction = row.direction ?? "";
      const status = row.status ?? "";

      let slippagePips: number | null = null;
      if (entryPrice != null && signalPrice != null) {
        const pipDivisor = symbol.includes("JPY") ? 0.01 : 0.0001;
        let diff = (entryPrice - signalPrice) / pipDivisor;
        if (direction === "SELL") diff = -diff;
        slippagePips = Number.isFinite(diff) ? diff : null;
      }

      let timeToMaeSeconds: number | null = null;
      if (row.mae_timestamp && row.trade_timestamp) {
        const maeTime = new Date(row.mae_timestamp).getTime();
        const tradeTime = new Date(row.trade_timestamp).getTime();
        if (Number.isFinite(maeTime) && Number.isFinite(tradeTime)) {
          timeToMaeSeconds = (maeTime - tradeTime) / 1000;
        }
      }

      return {
        ...row,
        entry_price: entryPrice,
        profit_loss: profitLoss,
        signal_price: signalPrice,
        mfe_pips: mfePips,
        mae_pips: maePips,
        symbol_short: normalizeSymbol(symbol),
        result: deriveResult(profitLoss, status),
        zero_mfe: (mfePips ?? 0) < 0.5,
        slippage_pips: slippagePips,
        time_to_mae_seconds: timeToMaeSeconds,
        limit_offset_pips: parseNumeric(row.limit_offset_pips),
        pullback_depth: parseNumeric(row.pullback_depth),
        confidence_score: parseNumeric(row.confidence_score),
        risk_reward_ratio: parseNumeric(row.risk_reward_ratio),
        spread_pips: parseNumeric(row.spread_pips),
        settings_context: deriveSettingsContext(row)
      };
    });

    const summary = summarizeTrades(trades, (trade) => ({
      strategy: trade.strategy,
      setup_family: trade.setup_family,
      entry_type: trade.entry_type
    }));
    const by_trigger = summarizeTrades(trades, (trade) => ({
      strategy: trade.strategy,
      signal_trigger: trade.signal_trigger || trade.trigger_type || "STANDARD",
      entry_type: trade.entry_type
    }));
    const strategy_summary = summarizeTrades(trades, (trade) => ({
      strategy: trade.strategy
    }));
    const strategy_options = (strategyOptionsResult.rows ?? [])
      .map((row) => row.strategy)
      .filter((strategy): strategy is string => typeof strategy === "string" && strategy.length > 0);

    return NextResponse.json({ trades, summary, by_trigger, strategy_summary, strategy_options });
  } catch (error) {
    console.error("Failed to load entry timing analysis", error);
    return NextResponse.json(
      { error: "Failed to load entry timing analysis" },
      { status: 500 }
    );
  }
}
