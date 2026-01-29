import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

const gradeOrder = ["D", "C", "B", "A", "A+"];

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const scanner = searchParams.get("scanner");
  const status = searchParams.get("status");
  const minScore = searchParams.get("minScore");
  const minClaudeGrade = searchParams.get("minClaudeGrade");
  const claudeOnly = searchParams.get("claudeOnly") === "true";
  const claudeAction = searchParams.get("claudeAction");
  const dateFrom = searchParams.get("dateFrom");
  const dateTo = searchParams.get("dateTo");
  const minRs = searchParams.get("minRs");
  const maxRs = searchParams.get("maxRs");
  const rsTrend = searchParams.get("rsTrend");
  const limit = Number(searchParams.get("limit") || 100);
  const orderBy = searchParams.get("orderBy") || "date_desc";

  const conditions: string[] = [];
  const params: Array<string | number | string[]> = [];

  if (scanner) {
    params.push(scanner);
    conditions.push(`scanner_name = $${params.length}`);
  }

  if (status) {
    params.push(status);
    conditions.push(`status = $${params.length}`);
  }

  if (minScore) {
    params.push(Number(minScore));
    conditions.push(`composite_score >= $${params.length}`);
  }

  if (claudeOnly) {
    conditions.push(`claude_analyzed_at IS NOT NULL`);
  }

  if (claudeAction) {
    params.push(claudeAction);
    conditions.push(`claude_action = $${params.length}`);
  }

  if (minClaudeGrade) {
    const minIndex = gradeOrder.indexOf(minClaudeGrade);
    const validGrades = gradeOrder.filter((_, idx) => idx >= minIndex);
    conditions.push(`claude_grade = ANY($${params.length + 1})`);
    params.push(validGrades);
  }

  if (dateFrom) {
    params.push(dateFrom);
    conditions.push(`DATE(signal_timestamp) >= $${params.length}`);
  }

  if (dateTo) {
    params.push(dateTo);
    conditions.push(`DATE(signal_timestamp) <= $${params.length}`);
  }

  const whereClause = conditions.length ? conditions.join(" AND ") : "1=1";

  const rsConditions: string[] = [];
  if (minRs) {
    rsConditions.push(`m.rs_percentile >= ${Number(minRs)}`);
  }
  if (maxRs) {
    rsConditions.push(`m.rs_percentile <= ${Number(maxRs)}`);
  }
  if (rsTrend) {
    rsConditions.push(`m.rs_trend = '${rsTrend}'`);
  }

  const client = await pool.connect();
  try {
    let query = `
      WITH latest_signals AS (
        SELECT DISTINCT ON (ticker, scanner_name)
          *
        FROM stock_scanner_signals
        WHERE ${whereClause}
        ORDER BY ticker, scanner_name, signal_timestamp DESC
      )
      SELECT
        s.id,
        s.signal_timestamp,
        s.scanner_name,
        s.ticker,
        s.signal_type,
        s.entry_price,
        s.composite_score,
        s.quality_tier,
        s.status,
        s.trend_score,
        s.momentum_score,
        s.volume_score,
        s.pattern_score,
        s.risk_percent,
        s.risk_reward_ratio,
        s.setup_description,
        s.confluence_factors,
        s.timeframe,
        s.market_regime,
        s.claude_grade,
        s.claude_score,
        s.claude_action,
        s.claude_thesis,
        s.claude_key_strengths,
        s.claude_key_risks,
        s.claude_analyzed_at,
        s.news_sentiment_score,
        s.news_sentiment_level,
        s.news_headlines_count,
        i.name as company_name,
        i.sector,
        COALESCE(i.exchange, 'NASDAQ') as exchange,
        i.analyst_rating,
        i.target_price,
        i.number_of_analysts,
        -- RS and trade plan context
        m.rs_percentile,
        m.rs_trend,
        m.atr_14,
        m.atr_percent,
        m.swing_high,
        m.swing_low,
        m.swing_high_date,
        m.swing_low_date,
        m.relative_volume,
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
        m.rsi_14,
        m.stoch_k,
        m.stoch_d,
        m.cci_20,
        m.adx_14,
        m.plus_di,
        m.minus_di,
        m.ao_value,
        m.momentum_10,
        m.macd,
        m.macd_signal,
        m.stoch_rsi_k,
        m.stoch_rsi_d,
        m.williams_r,
        m.bull_power,
        m.bear_power,
        m.ultimate_osc,
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
        -- DAQ
        d.daq_score,
        d.daq_grade,
        d.mtf_score,
        d.volume_score as daq_volume_score,
        d.smc_score as daq_smc_score,
        d.quality_score as daq_quality_score,
        d.catalyst_score as daq_catalyst_score,
        d.news_score as daq_news_score,
        d.regime_score as daq_regime_score,
        d.sector_score as daq_sector_score,
        d.earnings_within_7d,
        d.high_short_interest,
        d.sector_underperforming,
        -- Earnings
        i.earnings_date,
        CASE
          WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
          THEN (i.earnings_date - CURRENT_DATE)
          ELSE NULL
        END as days_to_earnings,
        bt_summary.trade_count,
        bt_summary.open_trade_count,
        bt_summary.latest_open_time,
        bt_last.last_trade_status,
        bt_last.last_trade_open_time,
        bt_last.last_trade_close_time,
        bt_last.last_trade_profit,
        bt_last.last_trade_profit_pct,
        bt_last.last_trade_side,
        bt_closed.last_closed_time,
        bt_closed.last_closed_profit,
        bt_closed.last_closed_profit_pct,
        bt_closed.last_closed_side
      FROM latest_signals s
      LEFT JOIN stock_instruments i ON s.ticker = i.ticker
      LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
        AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
      LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
      LEFT JOIN LATERAL (
        SELECT
          COUNT(*)::int AS trade_count,
          COUNT(*) FILTER (WHERE status = 'open')::int AS open_trade_count,
          MAX(open_time) FILTER (WHERE status = 'open') AS latest_open_time
        FROM broker_trades bt
        WHERE bt.ticker = s.ticker
           OR split_part(bt.ticker, '.', 1) = s.ticker
      ) bt_summary ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          bt.status AS last_trade_status,
          bt.open_time AS last_trade_open_time,
          bt.close_time AS last_trade_close_time,
          bt.profit AS last_trade_profit,
          bt.profit_pct AS last_trade_profit_pct,
          bt.side AS last_trade_side
        FROM broker_trades bt
        WHERE bt.ticker = s.ticker
           OR split_part(bt.ticker, '.', 1) = s.ticker
        ORDER BY bt.open_time DESC NULLS LAST
        LIMIT 1
      ) bt_last ON TRUE
      LEFT JOIN LATERAL (
        SELECT
          bt.close_time AS last_closed_time,
          bt.profit AS last_closed_profit,
          bt.profit_pct AS last_closed_profit_pct,
          bt.side AS last_closed_side
        FROM broker_trades bt
        WHERE (bt.ticker = s.ticker OR split_part(bt.ticker, '.', 1) = s.ticker)
          AND bt.status = 'closed'
        ORDER BY bt.close_time DESC NULLS LAST
        LIMIT 1
      ) bt_closed ON TRUE
    `;

    if (rsConditions.length) {
      query += ` WHERE ${rsConditions.join(" AND ")}`;
    }

    if (orderBy === "timestamp" || orderBy === "date_desc") {
      query += " ORDER BY s.signal_timestamp DESC";
    } else if (orderBy === "date_asc") {
      query += " ORDER BY s.signal_timestamp ASC";
    } else {
      query += `
        ORDER BY
          CASE WHEN s.claude_analyzed_at IS NOT NULL THEN 0 ELSE 1 END,
          COALESCE(s.claude_score, 0) DESC,
          s.composite_score DESC,
          s.signal_timestamp DESC
      `;
    }

    query += ` LIMIT ${limit}`;

    const result = await client.query(query, params);
    return NextResponse.json({ rows: result.rows });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load signals" }, { status: 500 });
  } finally {
    client.release();
  }
}
