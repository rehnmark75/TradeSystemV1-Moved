import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

// --- Scanner edge floor -------------------------------------------------
// scanner_pf here is intentionally the UPWARD-biased profit factor: open
// positions showing paper profit count as "wins", while losses require a
// fully-closed trade. We use it ONE-SIDED: if a scanner scores below the
// floor even under this optimistic accounting, over a real closed sample,
// it has no demonstrated edge and is dropped. Window/threshold are
// strictness knobs — surfaced in the UI so marginal scanners stay visible
// rather than silently excluded.
const EDGE_WINDOW_DAYS = 60; // trailing window for scanner edge (larger = more stable)
const EDGE_PF_FLOOR = 1.0; // drop a scanner below this PF...
const EDGE_MIN_CLOSED = 10; // ...only once it has >= this many closed trades

// Candidate score (validated against the live candidate pool):
//   0.55 * rs_percentile
// + 0.45 * rescaled tv_overall_score (-100..100 -> 0..100)
// + rs_trend: improving +5 / stable 0 / deteriorating -15   (user prefers RISING RS)
// - risk penalties: earnings<=7d (-15), high short interest (-10), RSI>80 overbought (-10)
//
// The row projection MIRRORS /api/signals/results so the signals-page detail
// panel can render an expanded Top-10 row with full field parity.

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Math.min(Math.max(1, Number(searchParams.get("limit") || 10)), 50);
  const maxPerScanner = Math.min(
    limit,
    Math.max(1, Number(searchParams.get("maxPerScanner") || limit))
  );

  const client = await pool.connect();
  try {
    const query = `
      WITH scanner_edge AS (
        SELECT
          scanner_name,
          count(*) FILTER (WHERE status = 'closed') AS closed_n,
          (
            (count(*) FILTER (WHERE realized_pnl_pct > 0)::numeric
              * NULLIF(avg(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0), 0))
            / NULLIF(
                count(*) FILTER (WHERE realized_pnl_pct <= 0 AND status = 'closed')
                * abs(avg(realized_pnl_pct) FILTER (WHERE realized_pnl_pct <= 0 AND status = 'closed')),
                0)
          ) AS pf
        FROM stock_scanner_signals
        WHERE signal_timestamp >= NOW() - ($1 * INTERVAL '1 day')
        GROUP BY scanner_name
      ),
      latest_batch AS (
        SELECT DISTINCT ON (ticker, scanner_name) *
        FROM stock_scanner_signals
        WHERE signal_date = (SELECT max(signal_date) FROM stock_scanner_signals)
        ORDER BY ticker, scanner_name, signal_timestamp DESC
      ),
      enriched AS (
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
          i.name AS company_name,
          i.sector,
          COALESCE(i.exchange, 'NASDAQ') AS exchange,
          i.analyst_rating,
          i.target_price,
          i.number_of_analysts,
          m.rs_percentile,
          m.rs_trend,
          m.atr_14,
          m.atr_percent,
          m.swing_high,
          m.swing_low,
          m.swing_high_date,
          m.swing_low_date,
          m.relative_volume,
          m.tv_osc_buy,
          m.tv_osc_sell,
          m.tv_osc_neutral,
          m.tv_ma_buy,
          m.tv_ma_sell,
          m.tv_ma_neutral,
          m.tv_overall_signal,
          m.tv_overall_score,
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
          d.daq_score,
          d.daq_grade,
          d.mtf_score,
          d.volume_score AS daq_volume_score,
          d.smc_score AS daq_smc_score,
          d.quality_score AS daq_quality_score,
          d.catalyst_score AS daq_catalyst_score,
          d.news_score AS daq_news_score,
          d.regime_score AS daq_regime_score,
          d.sector_score AS daq_sector_score,
          d.earnings_within_7d,
          d.high_short_interest,
          d.sector_underperforming,
          i.earnings_date,
          CASE
            WHEN i.earnings_date IS NOT NULL AND i.earnings_date >= CURRENT_DATE
            THEN (i.earnings_date - CURRENT_DATE)
            ELSE NULL
          END AS days_to_earnings,
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
          bt_closed.last_closed_side,
          e.pf AS scanner_pf,
          e.closed_n AS scanner_closed_n
        FROM latest_batch s
        LEFT JOIN stock_instruments i ON s.ticker = i.ticker
        LEFT JOIN stock_screening_metrics m ON s.ticker = m.ticker
          AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
        LEFT JOIN scanner_edge e ON s.scanner_name = e.scanner_name
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
      ),
      scored AS (
        SELECT *,
          round(
              0.55 * COALESCE(rs_percentile, 0)
            + 0.45 * ((COALESCE(tv_overall_score, 0) + 100) / 2.0)
            + (CASE rs_trend WHEN 'improving' THEN 5 WHEN 'deteriorating' THEN -15 ELSE 0 END)
            - (CASE WHEN earnings_within_7d THEN 15 ELSE 0 END)
            - (CASE WHEN high_short_interest THEN 10 ELSE 0 END)
            - (CASE WHEN rsi_14 > 80 THEN 10 ELSE 0 END)
          , 1) AS candidate_score
        FROM enriched
      ),
      eligible AS (
        -- one-sided floor: keep unless the OPTIMISTIC PF is below floor over a real sample
        SELECT * FROM scored
        WHERE NOT (COALESCE(scanner_pf, 999) < $2 AND COALESCE(scanner_closed_n, 0) >= $3)
      ),
      capped AS (
        SELECT *,
          row_number() OVER (
            PARTITION BY scanner_name
            ORDER BY candidate_score DESC, rs_percentile DESC NULLS LAST, ticker
          ) AS scanner_rank
        FROM eligible
      )
      SELECT *,
        row_number() OVER (ORDER BY candidate_score DESC, rs_percentile DESC NULLS LAST, ticker) AS rank
      FROM capped
      WHERE scanner_rank <= $4
      ORDER BY candidate_score DESC, rs_percentile DESC NULLS LAST, ticker
      LIMIT $5
    `;
    const result = await client.query(query, [
      EDGE_WINDOW_DAYS,
      EDGE_PF_FLOOR,
      EDGE_MIN_CLOSED,
      maxPerScanner,
      limit,
    ]);
    return NextResponse.json({
      rows: result.rows,
      meta: {
        batch_date: result.rows[0]?.signal_timestamp ?? null,
        edge_window_days: EDGE_WINDOW_DAYS,
        edge_pf_floor: EDGE_PF_FLOOR,
        edge_min_closed: EDGE_MIN_CLOSED,
        max_per_scanner: maxPerScanner,
      },
    });
  } catch (error) {
    console.error("top-candidates query failed", error);
    return NextResponse.json({ error: "Failed to load top candidates" }, { status: 500 });
  } finally {
    client.release();
  }
}
