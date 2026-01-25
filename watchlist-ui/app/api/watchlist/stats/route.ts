import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const CROSSOVER = ["ema_50_crossover", "ema_20_crossover", "macd_bullish_cross"];
const EVENT = ["gap_up_continuation", "rsi_oversold_bounce"];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const scanDate = searchParams.get("date");

  const client = await pool.connect();
  try {
    const crossoverQuery = `
      SELECT watchlist_name, COUNT(*) as stock_count, MAX(scan_date) as last_scan
      FROM stock_watchlist_results
      WHERE watchlist_name = ANY($1)
        AND status = 'active'
      GROUP BY watchlist_name
    `;
    const crossoverResult = await client.query(crossoverQuery, [CROSSOVER]);

    let eventDate = scanDate;
    if (!eventDate) {
      const maxDateQuery = `
        SELECT MAX(scan_date) as max_date
        FROM stock_watchlist_results
        WHERE watchlist_name = ANY($1)
      `;
      const maxDateResult = await client.query(maxDateQuery, [EVENT]);
      eventDate = maxDateResult.rows[0]?.max_date || null;
    }

    let eventResultRows = [];
    if (eventDate) {
      const eventQuery = `
        SELECT watchlist_name, COUNT(*) as stock_count, MAX(scan_date) as last_scan
        FROM stock_watchlist_results
        WHERE watchlist_name = ANY($1)
          AND scan_date = $2
        GROUP BY watchlist_name
      `;
      const eventResult = await client.query(eventQuery, [EVENT, eventDate]);
      eventResultRows = eventResult.rows;
    }

    const totalQuery = `
      SELECT COUNT(*) as total
      FROM stock_instruments
      WHERE is_active = true
    `;
    const totalResult = await client.query(totalQuery);

    const allRows = [...crossoverResult.rows, ...eventResultRows];
    const counts: Record<string, number> = {};
    let lastScan: string | null = null;
    allRows.forEach((row) => {
      counts[row.watchlist_name] = Number(row.stock_count || 0);
      if (row.last_scan && (!lastScan || row.last_scan > lastScan)) {
        lastScan = row.last_scan;
      }
    });

    return NextResponse.json({
      counts,
      last_scan: lastScan,
      total_stocks_scanned: Number(totalResult.rows[0]?.total || 0),
      event_date: eventDate
    });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load stats" }, { status: 500 });
  } finally {
    client.release();
  }
}
