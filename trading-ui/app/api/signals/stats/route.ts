import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  const client = await pool.connect();
  try {
    const totalsQuery = `
      SELECT
        COUNT(*) as total_signals,
        COUNT(*) FILTER (WHERE status = 'active') as active_signals,
        COUNT(*) FILTER (WHERE quality_tier IN ('A+', 'A')) as high_quality,
        COUNT(*) FILTER (WHERE DATE(signal_timestamp) = CURRENT_DATE) as today_signals,
        COUNT(*) FILTER (WHERE claude_analyzed_at IS NOT NULL) as claude_analyzed,
        COUNT(*) FILTER (WHERE claude_grade IN ('A+', 'A')) as claude_high_grade,
        COUNT(*) FILTER (WHERE claude_action = 'STRONG BUY') as claude_strong_buys,
        COUNT(*) FILTER (WHERE claude_action = 'BUY') as claude_buys,
        COUNT(*) FILTER (WHERE claude_analyzed_at IS NULL AND status = 'active') as awaiting_analysis
      FROM stock_scanner_signals
    `;
    const totalsResult = await client.query(totalsQuery);

    const byScannerQuery = `
      SELECT
        r.scanner_name,
        COALESCE(s.signal_count, 0) as signal_count,
        COALESCE(s.avg_score, 0) as avg_score,
        COALESCE(s.active_count, 0) as active_count,
        COALESCE(t.broker_trade_count, 0) as broker_trade_count,
        COALESCE(t.closed_trade_count, 0) as closed_trade_count,
        COALESCE(t.winning_trades, 0) as winning_trades,
        COALESCE(t.losing_trades, 0) as losing_trades,
        COALESCE(t.win_rate, 0) as win_rate,
        COALESCE(t.net_profit, 0) as net_profit,
        COALESCE(t.gross_profit, 0) as gross_profit,
        COALESCE(t.gross_loss, 0) as gross_loss,
        t.profit_factor
      FROM stock_signal_scanners r
      LEFT JOIN (
        SELECT
          scanner_name,
          COUNT(*) as signal_count,
          ROUND(AVG(composite_score)::numeric, 1) as avg_score,
          COUNT(*) FILTER (WHERE status = 'active') as active_count
        FROM stock_scanner_signals
        GROUP BY scanner_name
      ) s ON r.scanner_name = s.scanner_name
      LEFT JOIN (
        SELECT
          s.scanner_name,
          COUNT(bt.id)::int as broker_trade_count,
          COUNT(bt.id) FILTER (WHERE bt.status = 'closed')::int as closed_trade_count,
          COUNT(bt.id) FILTER (WHERE bt.status = 'closed' AND bt.profit > 0)::int as winning_trades,
          COUNT(bt.id) FILTER (WHERE bt.status = 'closed' AND bt.profit < 0)::int as losing_trades,
          ROUND(
            100.0 * COUNT(bt.id) FILTER (WHERE bt.status = 'closed' AND bt.profit > 0)
            / NULLIF(COUNT(bt.id) FILTER (WHERE bt.status = 'closed'), 0),
            1
          ) as win_rate,
          COALESCE(SUM(bt.profit) FILTER (WHERE bt.status = 'closed'), 0) as net_profit,
          COALESCE(SUM(bt.profit) FILTER (WHERE bt.status = 'closed' AND bt.profit > 0), 0) as gross_profit,
          COALESCE(ABS(SUM(bt.profit) FILTER (WHERE bt.status = 'closed' AND bt.profit < 0)), 0) as gross_loss,
          COALESCE(SUM(bt.profit) FILTER (WHERE bt.status = 'closed' AND bt.profit > 0), 0)
            / NULLIF(ABS(SUM(bt.profit) FILTER (WHERE bt.status = 'closed' AND bt.profit < 0)), 0) as profit_factor
        FROM stock_scanner_signals s
        JOIN broker_trades bt ON bt.signal_id = s.id
        GROUP BY s.scanner_name
      ) t ON r.scanner_name = t.scanner_name
      WHERE r.is_active = true
      ORDER BY COALESCE(t.net_profit, 0) DESC, COALESCE(s.signal_count, 0) DESC, r.scanner_name
    `;
    const byScannerResult = await client.query(byScannerQuery);

    const byTierQuery = `
      SELECT
        quality_tier,
        COUNT(*) as count
      FROM stock_scanner_signals
      WHERE status = 'active'
      GROUP BY quality_tier
      ORDER BY
        CASE quality_tier
          WHEN 'A+' THEN 1
          WHEN 'A' THEN 2
          WHEN 'B' THEN 3
          WHEN 'C' THEN 4
          WHEN 'D' THEN 5
        END
    `;
    const byTierResult = await client.query(byTierQuery);

    return NextResponse.json({
      ...totalsResult.rows[0],
      by_scanner: byScannerResult.rows,
      by_tier: Object.fromEntries(byTierResult.rows.map((row) => [row.quality_tier, Number(row.count)]))
    });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load signal stats" }, { status: 500 });
  } finally {
    client.release();
  }
}
