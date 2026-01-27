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

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    const tradesResult = await forexPool.query(
      `
      SELECT
        id,
        symbol,
        direction,
        entry_price,
        timestamp,
        status,
        profit_loss,
        vsl_peak_profit_pips as mfe_pips,
        vsl_mae_pips as mae_pips,
        vsl_mae_price as mae_price,
        vsl_mae_timestamp as mae_time,
        virtual_sl_pips,
        vsl_stage,
        vsl_breakeven_triggered as hit_breakeven,
        vsl_stage1_triggered as hit_stage1,
        vsl_stage2_triggered as hit_stage2
      FROM trade_log
      WHERE is_scalp_trade = true
      AND timestamp >= $1
      ORDER BY timestamp DESC
      `,
      [since]
    );

    const trades = (tradesResult.rows ?? []).map((row) => {
      const profitLoss = row.profit_loss == null ? null : Number(row.profit_loss);
      const maePips = row.mae_pips == null ? null : Number(row.mae_pips);
      const virtualSl = row.virtual_sl_pips == null ? null : Number(row.virtual_sl_pips);
      const maePct =
        maePips != null && virtualSl != null && virtualSl !== 0
          ? (maePips / virtualSl) * 100
          : null;

      return {
        ...row,
        entry_price: row.entry_price == null ? null : Number(row.entry_price),
        profit_loss: profitLoss,
        mfe_pips: row.mfe_pips == null ? null : Number(row.mfe_pips),
        mae_pips: maePips,
        virtual_sl_pips: virtualSl,
        mae_pct_of_vsl: maePct == null ? null : Number(maePct.toFixed(1)),
        symbol_short: normalizeSymbol(row.symbol),
        result: deriveResult(profitLoss, row.status)
      };
    });

    const summaryResult = await forexPool.query(
      `
      SELECT
        symbol,
        COUNT(*) as total_trades,
        AVG(vsl_mae_pips) as avg_mae_pips,
        MAX(vsl_mae_pips) as max_mae_pips,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY vsl_mae_pips) as median_mae_pips,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY vsl_mae_pips) as p75_mae_pips,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY vsl_mae_pips) as p90_mae_pips,
        AVG(vsl_peak_profit_pips) as avg_mfe_pips,
        MAX(vsl_peak_profit_pips) as max_mfe_pips,
        AVG(virtual_sl_pips) as avg_vsl_setting,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses
      FROM trade_log
      WHERE is_scalp_trade = true
      AND timestamp >= $1
      AND vsl_mae_pips IS NOT NULL
      GROUP BY symbol
      ORDER BY total_trades DESC
      `,
      [since]
    );

    const summary = (summaryResult.rows ?? []).map((row) => {
      const totalTrades = Number(row.total_trades ?? 0);
      const wins = Number(row.wins ?? 0);
      return {
        ...row,
        symbol_short: normalizeSymbol(row.symbol),
        total_trades: totalTrades,
        wins,
        losses: Number(row.losses ?? 0),
        win_rate: totalTrades ? (wins / totalTrades) * 100 : 0,
        avg_mae_pips: Number(row.avg_mae_pips ?? 0),
        max_mae_pips: Number(row.max_mae_pips ?? 0),
        median_mae_pips: Number(row.median_mae_pips ?? 0),
        p75_mae_pips: Number(row.p75_mae_pips ?? 0),
        p90_mae_pips: Number(row.p90_mae_pips ?? 0),
        avg_mfe_pips: Number(row.avg_mfe_pips ?? 0),
        max_mfe_pips: Number(row.max_mfe_pips ?? 0),
        avg_vsl_setting: Number(row.avg_vsl_setting ?? 0)
      };
    });

    return NextResponse.json({ trades, summary });
  } catch (error) {
    console.error("Failed to load MAE analysis", error);
    return NextResponse.json(
      { error: "Failed to load MAE analysis" },
      { status: 500 }
    );
  }
}
