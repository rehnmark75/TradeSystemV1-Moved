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

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
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
        a.alert_timestamp as signal_timestamp,
        a.price as signal_price,
        a.confidence_score,
        a.signal_trigger,
        a.trigger_type,
        a.strategy_indicators->'tier3_entry'->>'entry_type' as entry_type,
        a.strategy_indicators->'tier3_entry'->>'order_type' as order_type,
        a.strategy_indicators->'tier3_entry'->>'limit_offset_pips' as limit_offset_pips,
        a.strategy_indicators->'tier3_entry'->>'pullback_depth' as pullback_depth,
        a.strategy_indicators->'tier3_entry'->>'in_optimal_zone' as in_optimal_zone,
        a.pattern_type,
        a.pattern_strength,
        a.rsi_divergence_detected,
        a.rsi_divergence,
        a.htf_candle_direction,
        a.market_session
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'tracking', 'expired')
      ORDER BY t.timestamp DESC
      `,
      [since]
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
        confidence_score: parseNumeric(row.confidence_score)
      };
    });

    const summaryResult = await forexPool.query(
      `
      SELECT
        COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', 'UNKNOWN') as entry_type,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
        COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
        COALESCE(SUM(t.profit_loss), 0) as total_pnl,
        AVG(t.vsl_mae_pips) as avg_mae_pips,
        AVG(t.vsl_peak_profit_pips) as avg_mfe_pips,
        COUNT(CASE WHEN COALESCE(t.vsl_peak_profit_pips, 0) < 0.5 THEN 1 END) as zero_mfe_count,
        AVG(a.confidence_score) as avg_confidence,
        AVG(CAST(a.strategy_indicators->'tier3_entry'->>'pullback_depth' AS FLOAT)) as avg_pullback_depth
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'expired')
      AND t.profit_loss IS NOT NULL
      GROUP BY a.strategy_indicators->'tier3_entry'->>'entry_type'
      ORDER BY total_trades DESC
      `,
      [since]
    );

    const summary = (summaryResult.rows ?? []).map((row) => {
      const totalTrades = Number(row.total_trades ?? 0);
      const wins = Number(row.wins ?? 0);
      const zeroMfeCount = Number(row.zero_mfe_count ?? 0);
      return {
        ...row,
        total_trades: totalTrades,
        wins,
        losses: Number(row.losses ?? 0),
        avg_pnl: Number(row.avg_pnl ?? 0),
        total_pnl: Number(row.total_pnl ?? 0),
        avg_mae_pips: Number(row.avg_mae_pips ?? 0),
        avg_mfe_pips: Number(row.avg_mfe_pips ?? 0),
        zero_mfe_count: zeroMfeCount,
        avg_confidence: Number(row.avg_confidence ?? 0),
        avg_pullback_depth: Number(row.avg_pullback_depth ?? 0),
        win_rate: totalTrades ? (wins / totalTrades) * 100 : 0,
        zero_mfe_pct: totalTrades ? (zeroMfeCount / totalTrades) * 100 : 0
      };
    });

    const triggerResult = await forexPool.query(
      `
      SELECT
        COALESCE(NULLIF(a.signal_trigger, ''), 'STANDARD') as signal_trigger,
        COALESCE(a.strategy_indicators->'tier3_entry'->>'entry_type', 'UNKNOWN') as entry_type,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
        COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
        COALESCE(SUM(t.profit_loss), 0) as total_pnl,
        AVG(t.vsl_mae_pips) as avg_mae_pips,
        AVG(t.vsl_peak_profit_pips) as avg_mfe_pips,
        COUNT(CASE WHEN COALESCE(t.vsl_peak_profit_pips, 0) < 0.5 THEN 1 END) as zero_mfe_count,
        AVG(a.confidence_score) as avg_confidence
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      AND t.status IN ('closed', 'expired')
      AND t.profit_loss IS NOT NULL
      GROUP BY a.signal_trigger, a.strategy_indicators->'tier3_entry'->>'entry_type'
      ORDER BY total_trades DESC
      `,
      [since]
    );

    const by_trigger = (triggerResult.rows ?? []).map((row) => {
      const totalTrades = Number(row.total_trades ?? 0);
      const wins = Number(row.wins ?? 0);
      const zeroMfeCount = Number(row.zero_mfe_count ?? 0);
      return {
        ...row,
        total_trades: totalTrades,
        wins,
        losses: Number(row.losses ?? 0),
        avg_pnl: Number(row.avg_pnl ?? 0),
        total_pnl: Number(row.total_pnl ?? 0),
        avg_mae_pips: Number(row.avg_mae_pips ?? 0),
        avg_mfe_pips: Number(row.avg_mfe_pips ?? 0),
        zero_mfe_count: zeroMfeCount,
        avg_confidence: Number(row.avg_confidence ?? 0),
        win_rate: totalTrades ? (wins / totalTrades) * 100 : 0,
        zero_mfe_pct: totalTrades ? (zeroMfeCount / totalTrades) * 100 : 0
      };
    });

    return NextResponse.json({ trades, summary, by_trigger });
  } catch (error) {
    console.error("Failed to load entry timing analysis", error);
    return NextResponse.json(
      { error: "Failed to load entry timing analysis" },
      { status: 500 }
    );
  }
}
