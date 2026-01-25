import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const toNumber = (value: unknown) => {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const toDateKey = (value: string | null) => {
  if (!value) return "unknown";
  const date = new Date(value);
  return date.toISOString().slice(0, 10);
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = Number(searchParams.get("days") || 30);
  const trendDays = Number(searchParams.get("trendDays") || 7);

  const client = await pool.connect();
  try {
    const balanceQuery = `
      SELECT total_value, invested, available, recorded_at
      FROM broker_account_balance
      ORDER BY recorded_at DESC
      LIMIT 1
    `;
    const balanceResult = await client.query(balanceQuery);
    const balance = balanceResult.rows[0] || null;

    const trendQuery = `
      SELECT recorded_at, total_value
      FROM broker_account_balance
      WHERE recorded_at >= NOW() - INTERVAL '${trendDays} days'
      ORDER BY recorded_at ASC
    `;
    const trendResult = await client.query(trendQuery);
    const trendRows = trendResult.rows || [];
    const trendFirst = trendRows[0];
    const trendLast = trendRows[trendRows.length - 1];
    const trendChange = trendFirst && trendLast
      ? toNumber(trendLast.total_value)! - toNumber(trendFirst.total_value)!
      : 0;
    const trendChangePct = trendFirst && toNumber(trendFirst.total_value)
      ? (trendChange / toNumber(trendFirst.total_value)!) * 100
      : 0;
    const trendDirection = trendChange > 0 ? "up" : trendChange < 0 ? "down" : "neutral";

    const lastSyncQuery = `
      SELECT completed_at
      FROM broker_sync_log
      WHERE status = 'completed'
      ORDER BY completed_at DESC
      LIMIT 1
    `;
    const lastSyncResult = await client.query(lastSyncQuery);
    const lastSync = lastSyncResult.rows[0]?.completed_at || null;

    const closedTradesQuery = `
      SELECT
        deal_id,
        ticker,
        side,
        quantity,
        open_price,
        close_price,
        profit,
        profit_pct,
        duration_hours,
        open_time,
        close_time
      FROM broker_trades
      WHERE status = 'closed'
        AND close_time >= NOW() - INTERVAL '${days} days'
      ORDER BY close_time DESC
    `;
    const closedTradesResult = await client.query(closedTradesQuery);
    const closedTrades = closedTradesResult.rows || [];

    const openPositionsQuery = `
      SELECT
        deal_id,
        ticker,
        side,
        quantity,
        open_price,
        current_price,
        profit,
        stop_loss,
        take_profit,
        open_time
      FROM broker_trades
      WHERE status = 'open'
      ORDER BY open_time DESC
    `;
    const openPositionsResult = await client.query(openPositionsQuery);
    const openPositions = openPositionsResult.rows || [];

    let totalProfit = 0;
    let totalLoss = 0;
    let netProfit = 0;
    let winningTrades = 0;
    let losingTrades = 0;
    let largestWin = 0;
    let largestLoss = 0;
    let totalDuration = 0;
    let durationCount = 0;

    let longTrades = 0;
    let shortTrades = 0;
    let longWins = 0;
    let shortWins = 0;
    let longProfit = 0;
    let shortProfit = 0;

    const profitPcts: number[] = [];
    const lossPcts: number[] = [];
    const wins: number[] = [];
    const losses: number[] = [];

    const orderedByClose = [...closedTrades].sort((a, b) => new Date(a.close_time).getTime() - new Date(b.close_time).getTime());
    let currentWinStreak = 0;
    let currentLossStreak = 0;
    let maxWinStreak = 0;
    let maxLossStreak = 0;

    const byDay: Record<string, { pnl: number; count: number }> = {};
    const byTicker: Record<string, { trades: number; wins: number; pnl: number }> = {};

    closedTrades.forEach((trade) => {
      const profit = toNumber(trade.profit) || 0;
      const profitPct = toNumber(trade.profit_pct);
      const duration = toNumber(trade.duration_hours);
      const side = trade.side;
      const dateKey = toDateKey(trade.close_time);

      netProfit += profit;
      if (profit >= 0) {
        totalProfit += profit;
        winningTrades += 1;
        wins.push(profit);
        if (profitPct !== null) profitPcts.push(profitPct);
        if (profit > largestWin) largestWin = profit;
      } else {
        totalLoss += profit;
        losingTrades += 1;
        losses.push(profit);
        if (profitPct !== null) lossPcts.push(profitPct);
        if (profit < largestLoss) largestLoss = profit;
      }

      if (duration !== null) {
        totalDuration += duration;
        durationCount += 1;
      }

      if (side === "long") {
        longTrades += 1;
        longProfit += profit;
        if (profit > 0) longWins += 1;
      }
      if (side === "short") {
        shortTrades += 1;
        shortProfit += profit;
        if (profit > 0) shortWins += 1;
      }

      if (!byDay[dateKey]) {
        byDay[dateKey] = { pnl: 0, count: 0 };
      }
      byDay[dateKey].pnl += profit;
      byDay[dateKey].count += 1;

      if (!byTicker[trade.ticker]) {
        byTicker[trade.ticker] = { trades: 0, wins: 0, pnl: 0 };
      }
      byTicker[trade.ticker].trades += 1;
      byTicker[trade.ticker].pnl += profit;
      if (profit > 0) byTicker[trade.ticker].wins += 1;
    });

    orderedByClose.forEach((trade) => {
      const profit = toNumber(trade.profit) || 0;
      if (profit >= 0) {
        currentWinStreak += 1;
        currentLossStreak = 0;
      } else {
        currentLossStreak += 1;
        currentWinStreak = 0;
      }
      if (currentWinStreak > maxWinStreak) maxWinStreak = currentWinStreak;
      if (currentLossStreak > maxLossStreak) maxLossStreak = currentLossStreak;
    });

    const totalTrades = closedTrades.length;
    const winRate = totalTrades ? (winningTrades / totalTrades) * 100 : 0;
    const avgProfit = wins.length ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
    const avgLoss = losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : 0;
    const avgProfitPct = profitPcts.length ? profitPcts.reduce((a, b) => a + b, 0) / profitPcts.length : 0;
    const avgLossPct = lossPcts.length ? lossPcts.reduce((a, b) => a + b, 0) / lossPcts.length : 0;
    const profitFactor = totalLoss !== 0 ? totalProfit / Math.abs(totalLoss) : 0;
    const expectancy = totalTrades ? netProfit / totalTrades : 0;
    const avgTradeDurationHours = durationCount ? totalDuration / durationCount : 0;
    const longWinRate = longTrades ? (longWins / longTrades) * 100 : 0;
    const shortWinRate = shortTrades ? (shortWins / shortTrades) * 100 : 0;

    const equityQuery = `
      SELECT recorded_at, total_value
      FROM broker_account_balance
      WHERE recorded_at >= NOW() - INTERVAL '${days} days'
      ORDER BY recorded_at ASC
    `;
    const equityResult = await client.query(equityQuery);
    const equityCurve = equityResult.rows || [];
    let peak = 0;
    let maxDrawdown = 0;
    equityCurve.forEach((point) => {
      const value = toNumber(point.total_value) || 0;
      if (value > peak) peak = value;
      const drawdown = peak ? peak - value : 0;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    });
    const maxDrawdownPct = peak ? (maxDrawdown / peak) * 100 : 0;

    return NextResponse.json({
      balance,
      trend: {
        change: trendChange,
        change_pct: trendChangePct,
        trend: trendDirection,
        data_points: trendRows.length
      },
      last_sync: lastSync,
      stats: {
        total_trades: totalTrades,
        winning_trades: winningTrades,
        losing_trades: losingTrades,
        win_rate: winRate,
        total_profit: totalProfit,
        total_loss: totalLoss,
        net_profit: netProfit,
        avg_profit: avgProfit,
        avg_loss: avgLoss,
        avg_profit_pct: avgProfitPct,
        avg_loss_pct: avgLossPct,
        largest_win: largestWin,
        largest_loss: largestLoss,
        profit_factor: profitFactor,
        expectancy,
        max_drawdown: maxDrawdown,
        max_drawdown_pct: maxDrawdownPct,
        max_consecutive_wins: maxWinStreak,
        max_consecutive_losses: maxLossStreak,
        avg_trade_duration_hours: avgTradeDurationHours,
        long_trades: longTrades,
        short_trades: shortTrades,
        long_win_rate: longWinRate,
        short_win_rate: shortWinRate,
        long_profit: longProfit,
        short_profit: shortProfit
      },
      open_positions: openPositions.map((pos) => {
        const entry = toNumber(pos.open_price) || 0;
        const current = toNumber(pos.current_price) || 0;
        let profitPct = 0;
        if (entry && current) {
          profitPct = pos.side === "long" ? ((current - entry) / entry) * 100 : ((entry - current) / entry) * 100;
        }
        return {
          ...pos,
          entry_price: entry,
          current_price: current,
          unrealized_pnl: toNumber(pos.profit) || 0,
          profit_pct: profitPct
        };
      }),
      closed_trades: closedTrades.slice(0, 50),
      by_day: Object.entries(byDay).map(([date, data]) => ({
        date,
        pnl: data.pnl,
        count: data.count
      })),
      by_ticker: Object.entries(byTicker).map(([ticker, data]) => ({
        ticker,
        trades: data.trades,
        win_rate: data.trades ? (data.wins / data.trades) * 100 : 0,
        pnl: data.pnl
      })),
      equity_curve: equityCurve
    });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load broker stats" }, { status: 500 });
  } finally {
    client.release();
  }
}
