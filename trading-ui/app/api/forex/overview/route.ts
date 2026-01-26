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

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    const statsResult = await forexPool.query(
      `
      SELECT
        COUNT(*) as total_trades,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winning_trades,
        COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losing_trades,
        COUNT(CASE WHEN status IN ('pending', 'pending_limit') THEN 1 END) as pending_trades,
        COALESCE(SUM(profit_loss), 0) as total_profit_loss,
        COALESCE(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 0) as avg_profit,
        COALESCE(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 0) as avg_loss,
        COALESCE(MAX(profit_loss), 0) as largest_win,
        COALESCE(MIN(profit_loss), 0) as largest_loss
      FROM trade_log
      WHERE timestamp >= $1
      `,
      [since]
    );

    const statsRow = statsResult.rows[0] ?? {};
    const totalTrades = Number(statsRow.total_trades ?? 0);
    const winningTrades = Number(statsRow.winning_trades ?? 0);
    const losingTrades = Number(statsRow.losing_trades ?? 0);
    const pendingTrades = Number(statsRow.pending_trades ?? 0);
    const totalProfitLoss = Number(statsRow.total_profit_loss ?? 0);
    const avgProfit = Number(statsRow.avg_profit ?? 0);
    const avgLoss = Number(statsRow.avg_loss ?? 0);
    const largestWin = Number(statsRow.largest_win ?? 0);
    const largestLoss = Number(statsRow.largest_loss ?? 0);
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
    const profitFactor =
      losingTrades > 0 && avgLoss < 0
        ? (avgProfit * winningTrades) / Math.abs(avgLoss * losingTrades)
        : Number.POSITIVE_INFINITY;

    const pairStatsResult = await forexPool.query(
      `
      SELECT
        symbol,
        COUNT(*) as trades,
        COALESCE(SUM(profit_loss), 0) as total_pnl,
        COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins
      FROM trade_log
      WHERE timestamp >= $1
      GROUP BY symbol
      ORDER BY total_pnl DESC
      `,
      [since]
    );

    const pairs = pairStatsResult.rows ?? [];
    const bestPair = pairs[0]?.symbol ?? "None";
    const worstPair = pairs.length ? pairs[pairs.length - 1].symbol : "None";

    const dailyPnlResult = await forexPool.query(
      `
      SELECT
        DATE(timestamp) as date,
        SUM(profit_loss) as daily_pnl,
        COUNT(*) as trade_count
      FROM trade_log
      WHERE timestamp >= $1
        AND profit_loss IS NOT NULL
      GROUP BY DATE(timestamp)
      ORDER BY date ASC
      `,
      [since]
    );

    const recentTradesResult = await forexPool.query(
      `
      SELECT
        t.id,
        t.symbol,
        t.direction,
        t.timestamp,
        t.status,
        t.profit_loss,
        t.pnl_currency,
        a.strategy
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= $1
      ORDER BY t.timestamp DESC
      LIMIT 12
      `,
      [since]
    );

    return NextResponse.json({
      stats: {
        total_trades: totalTrades,
        winning_trades: winningTrades,
        losing_trades: losingTrades,
        pending_trades: pendingTrades,
        total_profit_loss: totalProfitLoss,
        win_rate: winRate,
        avg_profit: avgProfit,
        avg_loss: avgLoss,
        profit_factor: profitFactor,
        largest_win: largestWin,
        largest_loss: largestLoss,
        best_pair: bestPair,
        worst_pair: worstPair,
        active_pairs: pairs.map((row) => row.symbol)
      },
      daily_pnl: dailyPnlResult.rows ?? [],
      recent_trades: recentTradesResult.rows ?? []
    });
  } catch (error) {
    console.error("Failed to load forex overview", error);
    return NextResponse.json(
      { error: "Failed to load forex overview" },
      { status: 500 }
    );
  }
}
