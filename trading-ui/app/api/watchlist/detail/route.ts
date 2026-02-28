import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const CROSSOVER = ["ema_50_crossover", "ema_20_crossover", "macd_bullish_cross"];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const watchlist = searchParams.get("watchlist");
  const ticker = searchParams.get("ticker");
  const scanDate = searchParams.get("date");

  if (!watchlist || !ticker) {
    return NextResponse.json({ error: "watchlist and ticker are required" }, { status: 400 });
  }

  const client = await pool.connect();
  try {
    if (CROSSOVER.includes(watchlist)) {
      const query = `
        SELECT
          w.ticker,
          i.name,
          COALESCE(i.exchange, 'NASDAQ') as exchange,
          w.price,
          w.volume,
          w.avg_volume,
          COALESCE(w.ema_20, m.ema_20) as ema_20,
          COALESCE(w.ema_50, m.ema_50) as ema_50,
          COALESCE(w.ema_200, m.ema_200) as ema_200,
          COALESCE(w.rsi_14, m.rsi_14) as rsi_14,
          COALESCE(w.macd, m.macd) as macd,
          m.macd_signal,
          w.macd_histogram,
          w.gap_pct,
          w.price_change_1d,
          w.scan_date,
          w.crossover_date,
          (CURRENT_DATE - w.crossover_date) + 1 as days_on_list,
          w.avg_daily_change_5d,
          w.daq_score,
          w.daq_grade,
          w.daq_mtf_score,
          w.daq_volume_score,
          w.daq_smc_score,
          w.daq_quality_score,
          w.daq_catalyst_score,
          w.daq_news_score,
          w.daq_regime_score,
          w.daq_sector_score,
          w.daq_earnings_risk,
          w.daq_high_short_interest,
          w.daq_sector_underperforming,
          w.atr_14,
          w.atr_percent,
          w.swing_high,
          w.swing_low,
          w.swing_high_date,
          w.swing_low_date,
          w.nearest_ob_price,
          w.nearest_ob_type,
          w.nearest_ob_distance,
          w.suggested_entry_low,
          w.suggested_entry_high,
          w.suggested_stop_loss,
          w.suggested_target_1,
          w.suggested_target_2,
          w.risk_reward_ratio,
          w.risk_percent,
          w.volume_trend,
          w.relative_volume,
          w.rs_percentile,
          w.rs_trend,
          w.trade_ready,
          w.trade_ready_score,
          w.trade_ready_reasons,
          w.structure_stop_loss,
          w.structure_target_1,
          w.structure_target_2,
          w.structure_rr_ratio,
          i.earnings_date,
          CASE
            WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
            THEN (i.earnings_date - CURRENT_DATE)
            ELSE NULL
          END as days_to_earnings,
          -- TradingView summary counts
          m.tv_osc_buy,
          m.tv_osc_sell,
          m.tv_osc_neutral,
          m.tv_ma_buy,
          m.tv_ma_sell,
          m.tv_ma_neutral,
          m.tv_overall_signal,
          m.tv_overall_score,
          -- Oscillators and indicators
          m.stoch_k,
          m.stoch_d,
          m.cci_20,
          m.adx_14,
          m.plus_di,
          m.minus_di,
          m.ao_value,
          m.momentum_10,
          m.stoch_rsi_k,
          m.stoch_rsi_d,
          m.williams_r,
          m.bull_power,
          m.bear_power,
          m.ultimate_osc,
          -- Moving averages
          m.ema_10,
          m.ema_20,
          m.ema_30,
          m.ema_50,
          m.ema_100,
          m.ema_200,
          m.sma_10,
          m.sma_20,
          m.sma_30,
          m.sma_50,
          m.sma_100,
          m.sma_200,
          m.ichimoku_base,
          m.vwma_20,
          m.perf_1w,
          m.perf_1m,
          m.perf_3m,
          i.analyst_rating,
          i.target_price,
          i.number_of_analysts,
          ar.period as reco_period,
          ar.strong_buy as reco_strong_buy,
          ar.buy as reco_buy,
          ar.hold as reco_hold,
          ar.sell as reco_sell,
          ar.strong_sell as reco_strong_sell,
          ls.signal_id,
          COALESCE(cs.claude_grade, cl.claude_grade) as claude_grade,
          COALESCE(cs.claude_score, cl.claude_score) as claude_score,
          COALESCE(cs.claude_action, cl.claude_action) as claude_action,
          COALESCE(cs.claude_thesis, cl.claude_thesis) as claude_thesis,
          COALESCE(cs.claude_conviction, cl.claude_conviction) as claude_conviction,
          COALESCE(cs.claude_key_strengths, cl.claude_key_strengths) as claude_key_strengths,
          COALESCE(cs.claude_key_risks, cl.claude_key_risks) as claude_key_risks,
          COALESCE(cs.claude_position_rec, cl.claude_position_rec) as claude_position_rec,
          COALESCE(cs.claude_stop_adjustment, cl.claude_stop_adjustment) as claude_stop_adjustment,
          COALESCE(cs.claude_time_horizon, cl.claude_time_horizon) as claude_time_horizon,
          COALESCE(cs.claude_analyzed_at, cl.claude_analyzed_at) as claude_analyzed_at
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
        LEFT JOIN LATERAL (
          SELECT id as signal_id
          FROM stock_scanner_signals
          WHERE ticker = w.ticker
          ORDER BY signal_timestamp DESC
          LIMIT 1
        ) ls ON TRUE
        LEFT JOIN LATERAL (
          SELECT
            claude_grade,
            claude_score,
            claude_action,
            claude_thesis,
            claude_conviction,
            claude_key_strengths,
            claude_key_risks,
            claude_position_rec,
            claude_stop_adjustment,
            claude_time_horizon,
            claude_analyzed_at
          FROM stock_scanner_signals
          WHERE ticker = w.ticker
            AND claude_analyzed_at IS NOT NULL
          ORDER BY claude_analyzed_at DESC
          LIMIT 1
        ) cs ON TRUE
        LEFT JOIN LATERAL (
          SELECT
            claude_grade,
            claude_score,
            claude_action,
            claude_thesis,
            claude_conviction,
            claude_key_strengths,
            claude_key_risks,
            claude_position_rec,
            claude_stop_adjustment,
            claude_time_horizon,
            claude_analyzed_at
          FROM stock_watchlist_claude_analysis
          WHERE watchlist_name = w.watchlist_name
            AND ticker = w.ticker
          ORDER BY claude_analyzed_at DESC
          LIMIT 1
        ) cl ON TRUE
        WHERE w.watchlist_name = $1
          AND w.status = 'active'
          AND w.ticker = $2
        LIMIT 1
      `;
      const result = await client.query(query, [watchlist, ticker]);
      return NextResponse.json({ row: result.rows[0] || null });
    }

    const query = `
      SELECT
        w.ticker,
        i.name,
        COALESCE(i.exchange, 'NASDAQ') as exchange,
        w.price,
        w.volume,
        w.avg_volume,
        COALESCE(w.ema_20, m.ema_20) as ema_20,
        COALESCE(w.ema_50, m.ema_50) as ema_50,
        COALESCE(w.ema_200, m.ema_200) as ema_200,
        COALESCE(w.rsi_14, m.rsi_14) as rsi_14,
        COALESCE(w.macd, m.macd) as macd,
        m.macd_signal,
        w.macd_histogram,
        w.gap_pct,
        w.price_change_1d,
        w.scan_date,
        w.crossover_date,
        1 as days_on_list,
        w.avg_daily_change_5d,
        w.daq_score,
        w.daq_grade,
        w.daq_mtf_score,
        w.daq_volume_score,
        w.daq_smc_score,
        w.daq_quality_score,
        w.daq_catalyst_score,
        w.daq_news_score,
        w.daq_regime_score,
        w.daq_sector_score,
        w.daq_earnings_risk,
        w.daq_high_short_interest,
        w.daq_sector_underperforming,
        w.atr_14,
        w.atr_percent,
        w.swing_high,
        w.swing_low,
        w.swing_high_date,
        w.swing_low_date,
        w.nearest_ob_price,
        w.nearest_ob_type,
        w.nearest_ob_distance,
        w.suggested_entry_low,
        w.suggested_entry_high,
        w.suggested_stop_loss,
        w.suggested_target_1,
        w.suggested_target_2,
        w.risk_reward_ratio,
        w.risk_percent,
        w.volume_trend,
        w.relative_volume,
        w.rs_percentile,
        w.rs_trend,
        i.earnings_date,
        CASE
          WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
          THEN (i.earnings_date - CURRENT_DATE)
          ELSE NULL
        END as days_to_earnings,
        -- TradingView summary counts
        m.tv_osc_buy,
        m.tv_osc_sell,
        m.tv_osc_neutral,
        m.tv_ma_buy,
        m.tv_ma_sell,
        m.tv_ma_neutral,
        m.tv_overall_signal,
        m.tv_overall_score,
        -- Oscillators and indicators
        m.stoch_k,
        m.stoch_d,
        m.cci_20,
        m.adx_14,
        m.plus_di,
        m.minus_di,
        m.ao_value,
        m.momentum_10,
        m.stoch_rsi_k,
        m.stoch_rsi_d,
        m.williams_r,
        m.bull_power,
        m.bear_power,
        m.ultimate_osc,
        -- Moving averages
        m.ema_10,
        m.ema_20,
        m.ema_30,
        m.ema_50,
        m.ema_100,
        m.ema_200,
        m.sma_10,
        m.sma_20,
        m.sma_30,
        m.sma_50,
        m.sma_100,
        m.sma_200,
        m.ichimoku_base,
        m.vwma_20,
        m.perf_1w,
        m.perf_1m,
        m.perf_3m,
        i.analyst_rating,
        i.target_price,
        i.number_of_analysts,
        ar.period as reco_period,
        ar.strong_buy as reco_strong_buy,
        ar.buy as reco_buy,
        ar.hold as reco_hold,
        ar.sell as reco_sell,
        ar.strong_sell as reco_strong_sell,
        ls.signal_id,
        COALESCE(cs.claude_grade, cl.claude_grade) as claude_grade,
        COALESCE(cs.claude_score, cl.claude_score) as claude_score,
        COALESCE(cs.claude_action, cl.claude_action) as claude_action,
        COALESCE(cs.claude_thesis, cl.claude_thesis) as claude_thesis,
        COALESCE(cs.claude_conviction, cl.claude_conviction) as claude_conviction,
        COALESCE(cs.claude_key_strengths, cl.claude_key_strengths) as claude_key_strengths,
        COALESCE(cs.claude_key_risks, cl.claude_key_risks) as claude_key_risks,
        COALESCE(cs.claude_position_rec, cl.claude_position_rec) as claude_position_rec,
        COALESCE(cs.claude_stop_adjustment, cl.claude_stop_adjustment) as claude_stop_adjustment,
        COALESCE(cs.claude_time_horizon, cl.claude_time_horizon) as claude_time_horizon,
        COALESCE(cs.claude_analyzed_at, cl.claude_analyzed_at) as claude_analyzed_at
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
      LEFT JOIN LATERAL (
        SELECT id as signal_id
        FROM stock_scanner_signals
        WHERE ticker = w.ticker
        ORDER BY signal_timestamp DESC
        LIMIT 1
      ) ls ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          claude_grade,
          claude_score,
          claude_action,
          claude_thesis,
          claude_conviction,
          claude_key_strengths,
          claude_key_risks,
          claude_position_rec,
          claude_stop_adjustment,
          claude_time_horizon,
          claude_analyzed_at
        FROM stock_scanner_signals
        WHERE ticker = w.ticker
          AND claude_analyzed_at IS NOT NULL
        ORDER BY claude_analyzed_at DESC
        LIMIT 1
      ) cs ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          claude_grade,
          claude_score,
          claude_action,
          claude_thesis,
          claude_conviction,
          claude_key_strengths,
          claude_key_risks,
          claude_position_rec,
          claude_stop_adjustment,
          claude_time_horizon,
          claude_analyzed_at
        FROM stock_watchlist_claude_analysis
        WHERE watchlist_name = w.watchlist_name
          AND ticker = w.ticker
        ORDER BY claude_analyzed_at DESC
        LIMIT 1
      ) cl ON TRUE
      WHERE w.watchlist_name = $1
        AND w.scan_date = COALESCE($2::date, (
          SELECT MAX(scan_date)
          FROM stock_watchlist_results
          WHERE watchlist_name = $1
        ))
        AND w.ticker = $3
      LIMIT 1
    `;
    const result = await client.query(query, [watchlist, scanDate, ticker]);
    return NextResponse.json({ row: result.rows[0] || null });
  } catch (error) {
    console.error("watchlist detail error", error);
    return NextResponse.json({ error: "Failed to load detail", detail: String(error) }, { status: 500 });
  } finally {
    client.release();
  }
}
