import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const CROSSOVER = ["ema_50_crossover", "ema_20_crossover", "macd_bullish_cross"];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const watchlist = searchParams.get("watchlist");
  const scanDate = searchParams.get("date");
  const limit = Number(searchParams.get("limit") || 100);

  if (!watchlist) {
    return NextResponse.json({ error: "watchlist is required" }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    if (CROSSOVER.includes(watchlist)) {
      const query = `
        SELECT
          w.ticker,
          w.price,
          w.volume,
          w.avg_volume,
          w.ema_20,
          w.ema_50,
          w.ema_200,
          w.rsi_14,
          w.macd,
          w.gap_pct,
          w.price_change_1d,
          w.scan_date,
          w.crossover_date,
          (CURRENT_DATE - w.crossover_date) + 1 as days_on_list,
          w.avg_daily_change_5d,
          w.daq_score,
          w.daq_grade,
          w.daq_earnings_risk,
          w.daq_high_short_interest,
          w.rs_percentile,
          w.rs_trend,
          m.tv_overall_score,
          m.tv_overall_signal,
          m.perf_1w,
          m.perf_1m,
          m.perf_3m,
          ar.period as reco_period,
          ar.strong_buy as reco_strong_buy,
          ar.buy as reco_buy,
          ar.hold as reco_hold,
          ar.sell as reco_sell,
          ar.strong_sell as reco_strong_sell,
          COALESCE(i.exchange, 'NASDAQ') as exchange
        FROM stock_watchlist_results w
        LEFT JOIN stock_instruments i ON w.ticker = i.ticker
        LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
          AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        LEFT JOIN LATERAL (
          SELECT period, strong_buy, buy, hold, sell, strong_sell
          FROM stock_analyst_recommendations
          WHERE ticker = w.ticker
          ORDER BY period DESC
          LIMIT 1
        ) ar ON TRUE
        WHERE w.watchlist_name = $1
          AND w.status = 'active'
        ORDER BY w.crossover_date DESC NULLS LAST, w.volume DESC
        LIMIT $2
      `;
      const result = await client.query(query, [watchlist, limit]);
      return NextResponse.json({ rows: result.rows });
    }

    const query = `
      SELECT
        w.ticker,
        w.price,
        w.volume,
        w.avg_volume,
        w.ema_20,
        w.ema_50,
        w.ema_200,
        w.rsi_14,
        w.macd,
        w.gap_pct,
        w.price_change_1d,
        w.scan_date,
        w.crossover_date,
        1 as days_on_list,
        w.avg_daily_change_5d,
        w.daq_score,
        w.daq_grade,
        w.daq_earnings_risk,
        w.daq_high_short_interest,
        w.rs_percentile,
        w.rs_trend,
        m.tv_overall_score,
        m.tv_overall_signal,
        m.perf_1w,
        m.perf_1m,
        m.perf_3m,
        ar.period as reco_period,
        ar.strong_buy as reco_strong_buy,
        ar.buy as reco_buy,
        ar.hold as reco_hold,
        ar.sell as reco_sell,
        ar.strong_sell as reco_strong_sell,
        COALESCE(i.exchange, 'NASDAQ') as exchange
      FROM stock_watchlist_results w
      LEFT JOIN stock_instruments i ON w.ticker = i.ticker
      LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
        AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
      LEFT JOIN LATERAL (
        SELECT period, strong_buy, buy, hold, sell, strong_sell
        FROM stock_analyst_recommendations
        WHERE ticker = w.ticker
        ORDER BY period DESC
        LIMIT 1
      ) ar ON TRUE
      WHERE w.watchlist_name = $1
        AND w.scan_date = COALESCE($2::date, (
          SELECT MAX(scan_date)
          FROM stock_watchlist_results
          WHERE watchlist_name = $1
        ))
      ORDER BY w.scan_date DESC, w.volume DESC
      LIMIT $3
    `;
    const result = await client.query(query, [watchlist, scanDate, limit]);
    return NextResponse.json({ rows: result.rows });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load results" }, { status: 500 });
  } finally {
    client.release();
  }
}
