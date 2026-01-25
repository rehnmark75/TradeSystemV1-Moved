import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const minRs = Number(searchParams.get("minRs") || 80);
  const limit = Number(searchParams.get("limit") || 30);

  const client = await pool.connect();
  try {
    const query = `
      SELECT
        m.ticker,
        i.name,
        i.sector,
        m.current_price,
        m.rs_vs_spy,
        m.rs_percentile,
        m.rs_trend,
        m.price_change_20d,
        m.trend_strength,
        m.ma_alignment,
        m.atr_percent,
        m.rsi_14,
        m.pct_from_52w_high
      FROM stock_screening_metrics m
      JOIN stock_instruments i ON m.ticker = i.ticker
      WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        AND m.rs_percentile >= $1
        AND m.rs_percentile IS NOT NULL
      ORDER BY m.rs_percentile DESC
      LIMIT $2
    `;
    const result = await client.query(query, [minRs, limit]);
    return NextResponse.json({ rows: result.rows || [] });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load RS leaders" }, { status: 500 });
  } finally {
    client.release();
  }
}
