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
        COALESCE(s.active_count, 0) as active_count
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
      WHERE r.is_active = true
      ORDER BY COALESCE(s.signal_count, 0) DESC, r.scanner_name
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
