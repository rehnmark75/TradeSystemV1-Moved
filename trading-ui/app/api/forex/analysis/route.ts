import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    const strategyResult = await forexPool.query(
      `
      SELECT
        a.strategy,
        COUNT(t.*) as total_trades,
        COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN t.profit_loss < 0 THEN 1 END) as losses,
        COALESCE(SUM(t.profit_loss), 0) as total_pnl,
        COALESCE(AVG(t.profit_loss), 0) as avg_pnl,
        COALESCE(AVG(a.confidence_score), 0) as avg_confidence,
        COALESCE(MAX(t.profit_loss), 0) as best_trade,
        COALESCE(MIN(t.profit_loss), 0) as worst_trade,
        COUNT(DISTINCT t.symbol) as pairs_traded
      FROM trade_log t
      INNER JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      GROUP BY a.strategy
      ORDER BY total_pnl DESC
      `,
      [since]
    );

    const strategies = (strategyResult.rows ?? []).map((row) => {
      const totalTrades = Number(row.total_trades ?? 0);
      const wins = Number(row.wins ?? 0);
      return {
        ...row,
        total_trades: totalTrades,
        wins,
        losses: Number(row.losses ?? 0),
        total_pnl: Number(row.total_pnl ?? 0),
        avg_pnl: Number(row.avg_pnl ?? 0),
        avg_confidence: Number(row.avg_confidence ?? 0),
        best_trade: Number(row.best_trade ?? 0),
        worst_trade: Number(row.worst_trade ?? 0),
        pairs_traded: Number(row.pairs_traded ?? 0),
        win_rate: totalTrades > 0 ? (wins / totalTrades) * 100 : 0
      };
    });

    const pairResult = await forexPool.query(
      `
      SELECT
        symbol,
        COUNT(*) as total_trades,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
        COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losses,
        COALESCE(SUM(profit_loss), 0) as total_pnl,
        COALESCE(AVG(profit_loss), 0) as avg_pnl,
        COALESCE(MAX(profit_loss), 0) as best_trade,
        COALESCE(MIN(profit_loss), 0) as worst_trade
      FROM trade_log
      WHERE timestamp >= $1
      GROUP BY symbol
      ORDER BY total_pnl DESC
      `,
      [since]
    );

    const pairs = (pairResult.rows ?? []).map((row) => {
      const totalTrades = Number(row.total_trades ?? 0);
      const wins = Number(row.wins ?? 0);
      return {
        ...row,
        total_trades: totalTrades,
        wins,
        losses: Number(row.losses ?? 0),
        total_pnl: Number(row.total_pnl ?? 0),
        avg_pnl: Number(row.avg_pnl ?? 0),
        best_trade: Number(row.best_trade ?? 0),
        worst_trade: Number(row.worst_trade ?? 0),
        win_rate: totalTrades > 0 ? (wins / totalTrades) * 100 : 0
      };
    });

    return NextResponse.json({ strategies, pairs });
  } catch (error) {
    console.error("Failed to load forex analysis", error);
    return NextResponse.json(
      { error: "Failed to load forex analysis" },
      { status: 500 }
    );
  }
}
