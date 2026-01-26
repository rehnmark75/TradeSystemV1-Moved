import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const sectorAliases: Record<string, string[]> = {
  Technology: ["Technology"],
  "Health Care": ["Health Care", "Healthcare"],
  Financials: ["Financials", "Financial Services"],
  "Consumer Discretionary": ["Consumer Discretionary", "Consumer Cyclical"],
  "Consumer Staples": ["Consumer Staples", "Consumer Defensive"],
  "Communication Services": ["Communication Services"],
  Industrials: ["Industrials"],
  Energy: ["Energy"],
  Utilities: ["Utilities"],
  "Real Estate": ["Real Estate"],
  Materials: ["Materials", "Basic Materials"]
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const tickerRaw = searchParams.get("ticker") || "";
  const ticker = tickerRaw.trim().toUpperCase();

  if (!ticker) {
    return NextResponse.json({ error: "ticker is required" }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    const instrumentResult = await client.query(
      `
      SELECT
        ticker,
        name,
        exchange,
        sector,
        industry,
        market_cap,
        avg_volume,
        currency,
        earnings_date,
        dividend_yield,
        trailing_pe,
        forward_pe,
        profit_margin,
        revenue_growth,
        earnings_growth,
        debt_to_equity,
        current_ratio,
        quick_ratio,
        analyst_rating,
        target_price,
        target_high,
        target_low,
        number_of_analysts,
        fifty_two_week_high,
        fifty_two_week_low,
        fifty_two_week_change,
        fifty_day_average,
        two_hundred_day_average
      FROM stock_instruments
      WHERE ticker = $1
      LIMIT 1
      `,
      [ticker]
    );

    const metricsResult = await client.query(
      `
      SELECT *
      FROM stock_screening_metrics
      WHERE ticker = $1
      ORDER BY calculation_date DESC
      LIMIT 1
      `,
      [ticker]
    );

    const watchlistResult = await client.query(
      `
      SELECT *
      FROM stock_watchlist_results
      WHERE ticker = $1
      ORDER BY COALESCE(scan_date, crossover_date) DESC NULLS LAST, crossover_date DESC NULLS LAST
      LIMIT 1
      `,
      [ticker]
    );

    const latestSignalResult = await client.query(
      `
      SELECT
        s.*,
        d.daq_score,
        d.daq_grade,
        d.mtf_score,
        d.volume_score as daq_volume_score,
        d.smc_score as daq_smc_score,
        d.quality_score as daq_quality_score,
        d.catalyst_score as daq_catalyst_score,
        d.news_score as daq_news_score,
        d.regime_score as daq_regime_score,
        d.sector_score as daq_sector_score
      FROM stock_scanner_signals s
      LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
      WHERE s.ticker = $1
      ORDER BY s.signal_timestamp DESC
      LIMIT 1
      `,
      [ticker]
    );

    const signalHistoryResult = await client.query(
      `
      SELECT
        id,
        signal_timestamp,
        scanner_name,
        signal_type,
        composite_score,
        quality_tier,
        status,
        claude_action,
        claude_grade,
        news_sentiment_level,
        entry_price,
        risk_reward_ratio
      FROM stock_scanner_signals
      WHERE ticker = $1
      ORDER BY signal_timestamp DESC
      LIMIT 20
      `,
      [ticker]
    );

    const analystResult = await client.query(
      `
      SELECT period, strong_buy, buy, hold, sell, strong_sell
      FROM stock_analyst_recommendations
      WHERE ticker = $1
      ORDER BY period DESC
      LIMIT 1
      `,
      [ticker]
    );

    const newsResult = await client.query(
      `
      SELECT headline, summary, source, url, published_at, sentiment_score
      FROM stock_news_cache
      WHERE ticker = $1
      ORDER BY published_at DESC
      LIMIT 8
      `,
      [ticker]
    );

    const sector = instrumentResult.rows[0]?.sector || null;
    let sectorContext = null;
    if (sector) {
      const sectorResult = await client.query(
        `
        SELECT
          sector,
          sector_return_1d,
          sector_return_5d,
          sector_return_20d,
          rs_vs_spy,
          rs_percentile,
          rs_trend,
          stocks_in_sector,
          pct_above_sma50,
          pct_bullish_trend,
          sector_stage
        FROM sector_analysis
        WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
          AND sector = $1
        LIMIT 1
        `,
        [sector]
      );
      sectorContext = sectorResult.rows[0] || null;
      if (!sectorContext) {
        const aliases = sectorAliases[sector] || [];
        if (aliases.length) {
          const aliasResult = await client.query(
            `
            SELECT
              sector,
              sector_return_1d,
              sector_return_5d,
              sector_return_20d,
              rs_vs_spy,
              rs_percentile,
              rs_trend,
              stocks_in_sector,
              pct_above_sma50,
              pct_bullish_trend,
              sector_stage
            FROM sector_analysis
            WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
              AND sector = ANY($1::text[])
            LIMIT 1
            `,
            [aliases]
          );
          sectorContext = aliasResult.rows[0] || null;
        }
      }
    }

    const marketRegimeResult = await client.query(
      `
      SELECT *
      FROM v_current_market_regime
      LIMIT 1
      `
    );

    return NextResponse.json({
      instrument: instrumentResult.rows[0] || null,
      metrics: metricsResult.rows[0] || null,
      watchlist: watchlistResult.rows[0] || null,
      signal: latestSignalResult.rows[0] || null,
      signal_history: signalHistoryResult.rows || [],
      analyst: analystResult.rows[0] || null,
      news: newsResult.rows || [],
      sector_context: sectorContext,
      market_regime: marketRegimeResult.rows[0] || null
    });
  } catch (error) {
    console.error("stock detail error", error);
    return NextResponse.json({ error: "Failed to load stock detail", detail: String(error) }, { status: 500 });
  } finally {
    client.release();
  }
}
