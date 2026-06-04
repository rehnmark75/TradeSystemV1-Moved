import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

// --- Scanner edge floor -------------------------------------------------
// scanner_pf is a closed-trade profit factor. Winners and losers must use
// the same status='closed' population, otherwise open/expired/partial rows can
// inflate the edge metric. A scanner with closed wins and zero closed losses is
// treated as capped-positive PF; a scanner with no closed outcome history stays
// null and is not floor-filtered until it reaches EDGE_MIN_CLOSED.
const EDGE_WINDOW_DAYS = 60; // trailing window for scanner edge (larger = more stable)
const EDGE_PF_FLOOR = 1.0; // drop a scanner below this PF...
const EDGE_MIN_CLOSED = 10; // ...only once it has >= this many closed trades
const EDGE_NO_LOSS_PF_CAP = 3.0; // score cap for closed wins with zero closed losses
const ROBOMARKETS_API_URL = process.env.ROBOMARKETS_API_URL || "https://api.stockstrader.com/api/v1";
const ROBOMARKETS_API_KEY = process.env.ROBOMARKETS_API_KEY || "";
const ROBOMARKETS_ACCOUNT_ID = process.env.ROBOMARKETS_ACCOUNT_ID || "";

// Candidate score (validated against the live candidate pool):
//   0.55 * rs_percentile
// + 0.45 * rescaled tv_overall_score (-100..100 -> 0..100)
// + rs_trend: improving +5 / stable 0 / deteriorating -15   (user prefers RISING RS)
// - risk penalties: earnings<=7d (-15), high short interest (-10), RSI>80 overbought (-10)
//
// Day-trade score:
//   emphasizes tradability today: relative volume, catalyst/news/DAQ context,
//   sector/regime alignment, setup quality, execution quality, and extension risk.
//
// The row projection MIRRORS /api/signals/results so the signals-page detail
// panel can render an expanded Top-10 row with full field parity.

type QuoteSnapshot = {
  broker_bid: number | null;
  broker_ask: number | null;
  broker_last: number | null;
  broker_volume: number | null;
  broker_spread: number | null;
  broker_spread_pct: number | null;
  broker_quote_time: string | null;
};

const firstFiniteNumber = (...values: unknown[]) => {
  for (const value of values) {
    if (value == null) continue;
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const fetchBrokerQuotes = async (tickers: string[]): Promise<Record<string, QuoteSnapshot>> => {
  if (!ROBOMARKETS_API_KEY || !ROBOMARKETS_ACCOUNT_ID || tickers.length === 0) return {};

  const entries = await Promise.all(
    tickers.map(async (ticker) => {
      try {
        const res = await fetch(
          `${ROBOMARKETS_API_URL}/accounts/${ROBOMARKETS_ACCOUNT_ID}/instruments/${encodeURIComponent(ticker)}/quote`,
          {
            headers: {
              Authorization: `Bearer ${ROBOMARKETS_API_KEY}`,
              Accept: "application/json",
            },
            cache: "no-store",
          }
        );
        if (!res.ok) return [ticker, null] as const;
        const payload = await res.json();
        const data = payload?.data ?? {};
        const bid = data.bid_price == null ? null : Number(data.bid_price);
        const ask = data.ask_price == null ? null : Number(data.ask_price);
        const last = data.last_price == null ? null : Number(data.last_price);
        const volume = firstFiniteNumber(
          data.volume,
          data.day_volume,
          data.daily_volume,
          data.traded_volume,
          data.turnover_volume
        );
        const spread = bid != null && ask != null ? ask - bid : null;
        const mid = bid != null && ask != null ? (bid + ask) / 2 : last;
        const quoteTime =
          data.ask_bid_price_time != null
            ? new Date(Number(data.ask_bid_price_time) * 1000).toISOString()
            : data.last_price_time != null
              ? new Date(Number(data.last_price_time) * 1000).toISOString()
              : null;
        return [
          ticker,
          {
            broker_bid: bid,
            broker_ask: ask,
            broker_last: last,
            broker_volume: volume,
            broker_spread: spread,
            broker_spread_pct: spread != null && mid ? Number(((spread / mid) * 100).toFixed(3)) : null,
            broker_quote_time: quoteTime,
          },
        ] as const;
      } catch {
        return [ticker, null] as const;
      }
    })
  );

  return Object.fromEntries(entries.filter((entry): entry is readonly [string, QuoteSnapshot] => entry[1] !== null));
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const limit = Math.min(Math.max(1, Number(searchParams.get("limit") || 10)), 50);
  const mode = searchParams.get("mode") === "daytrades" ? "daytrades" : "candidates";
  const maxPerScanner = Math.min(
    limit,
    Math.max(1, Number(searchParams.get("maxPerScanner") || limit))
  );
  const queryLimit = mode === "daytrades" ? Math.min(50, Math.max(limit, limit * 3)) : limit;

  const client = await pool.connect();
  try {
    const scoreExpression =
      mode === "daytrades"
        ? `
              0.20 * COALESCE(rs_percentile, 0)
            + 0.20 * LEAST(COALESCE(relative_volume, 0) * 25, 100)
            + 0.20 * (
                0.55 * COALESCE(daq_catalyst_score, 0)
              + 0.45 * COALESCE(daq_news_score, 0)
              )
            + 0.10 * (
                CASE
                  WHEN pm_generated_at IS NULL THEN 0
                  WHEN pm_is_current_session THEN COALESCE(pm_confidence, 0) * 100
                  ELSE 0
                END
              )
            + 0.15 * (
                0.50 * COALESCE(daq_sector_score, 50)
              + 0.50 * COALESCE(daq_regime_score, 50)
              )
            + 0.10 * (
                0.45 * ((COALESCE(tv_overall_score, 0) + 100) / 2.0)
              + 0.35 * COALESCE(daq_quality_score, 50)
              + 0.20 * COALESCE(mtf_score, 50)
              )
            + 0.05 * COALESCE(daq_volume_score, 50)
            + (CASE rs_trend WHEN 'improving' THEN 5 WHEN 'deteriorating' THEN -12 ELSE 0 END)
            + (CASE
                WHEN pm_is_current_session AND pm_direction = 'BUY' AND pm_gap_percent BETWEEN 1 AND 8 THEN 6
                WHEN pm_is_current_session AND pm_direction = 'BUY' AND pm_gap_percent > 8 THEN 2
                WHEN pm_is_current_session AND pm_direction = 'SELL' THEN -10
                ELSE 0
              END)
            - (CASE WHEN COALESCE(relative_volume, 0) < 1 THEN 10 ELSE 0 END)
            - (CASE WHEN rsi_14 > 80 THEN 12 WHEN rsi_14 > 75 THEN 6 ELSE 0 END)
            - (CASE WHEN high_short_interest THEN 8 ELSE 0 END)
            - (CASE WHEN sector_underperforming THEN 8 ELSE 0 END)
            - (CASE
                WHEN earnings_within_7d
                  AND COALESCE(daq_catalyst_score, 0) < 50
                  AND COALESCE(daq_news_score, 0) < 50
                THEN 12
                ELSE 0
              END)
        `
        : `
              0.55 * COALESCE(rs_percentile, 0)
            + 0.45 * ((COALESCE(tv_overall_score, 0) + 100) / 2.0)
            + (CASE rs_trend WHEN 'improving' THEN 5 WHEN 'deteriorating' THEN -15 ELSE 0 END)
            - (CASE WHEN earnings_within_7d THEN 15 ELSE 0 END)
            - (CASE WHEN high_short_interest THEN 10 ELSE 0 END)
            - (CASE WHEN rsi_14 > 80 THEN 10 ELSE 0 END)
        `;

    const query = `
      WITH scanner_edge AS (
        SELECT
          scanner_name,
          closed_n,
          CASE
            WHEN gross_loss > 0 THEN gross_profit / gross_loss
            WHEN gross_profit > 0 THEN $6::numeric
            ELSE NULL
          END AS pf
        FROM (
          SELECT
            scanner_name,
            count(*) AS closed_n,
            COALESCE(sum(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0), 0)::numeric AS gross_profit,
            ABS(COALESCE(sum(realized_pnl_pct) FILTER (WHERE realized_pnl_pct <= 0), 0))::numeric AS gross_loss
          FROM stock_scanner_signals
          WHERE signal_timestamp >= NOW() - ($1 * INTERVAL '1 day')
            AND status = 'closed'
            AND realized_pnl_pct IS NOT NULL
          GROUP BY scanner_name
        ) closed_edge
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
          m.avg_volume_20,
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
          e.closed_n AS scanner_closed_n,
          pm.signal_type AS pm_signal_type,
          pm.direction AS pm_direction,
          pm.strength AS pm_strength,
          pm.confidence AS pm_confidence,
          pm.gap_percent AS pm_gap_percent,
          pm.gap_type AS pm_gap_type,
          pm.current_price AS pm_current_price,
          pm.previous_close AS pm_previous_close,
          pm.news_count AS pm_news_count,
          pm.news_sentiment_score AS pm_news_sentiment_score,
          pm.news_sentiment_level AS pm_news_sentiment_level,
          pm.suggested_entry AS pm_suggested_entry,
          pm.suggested_stop AS pm_suggested_stop,
          pm.suggested_target AS pm_suggested_target,
          pm.risk_reward AS pm_risk_reward,
          pm.generated_at AS pm_generated_at,
          EXTRACT(EPOCH FROM (NOW() - pm.generated_at)) / 3600.0 AS pm_age_hours,
          (
            (pm.generated_at AT TIME ZONE 'America/New_York')::date =
            (NOW() AT TIME ZONE 'America/New_York')::date
          ) AS pm_is_current_session
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
        LEFT JOIN LATERAL (
          SELECT
            p.signal_type,
            p.direction,
            p.strength,
            p.confidence,
            p.gap_percent,
            p.gap_type,
            p.current_price,
            p.previous_close,
            p.news_count,
            p.news_sentiment_score,
            p.news_sentiment_level,
            p.suggested_entry,
            p.suggested_stop,
            p.suggested_target,
            p.risk_reward,
            p.generated_at
          FROM stock_premarket_signals p
          WHERE p.symbol = s.ticker
          ORDER BY p.generated_at DESC
          LIMIT 1
        ) pm ON TRUE
      ),
      scored AS (
        SELECT *,
          round(${scoreExpression}, 1) AS candidate_score,
          (
            COALESCE(scanner_pf, 999) < $2
            AND COALESCE(scanner_closed_n, 0) >= $3
          ) AS edge_floor_blocked,
          CASE
            WHEN pm_generated_at IS NULL THEN 'No PM data'
            WHEN NOT pm_is_current_session THEN 'Stale PM'
            WHEN pm_direction <> 'BUY' THEN 'PM against'
            WHEN pm_gap_percent >= 1 AND COALESCE(pm_confidence, 0) >= 0.65 THEN 'PM confirmed'
            WHEN pm_gap_percent > 0 THEN 'PM watch'
            WHEN pm_gap_percent < -1 THEN 'PM fading'
            ELSE 'PM neutral'
          END AS pm_status,
          CASE
            WHEN pm_generated_at IS NULL THEN 'Wait'
            WHEN NOT pm_is_current_session THEN 'Refresh'
            WHEN pm_direction <> 'BUY' THEN 'Avoid'
            WHEN high_short_interest OR sector_underperforming THEN 'Small only'
            WHEN pm_suggested_entry IS NOT NULL AND COALESCE(pm_confidence, 0) >= 0.65 THEN 'Use PM levels'
            WHEN pm_gap_percent BETWEEN 1 AND 8 AND COALESCE(relative_volume, 0) >= 1.2 THEN 'Pullback'
            WHEN pm_gap_percent > 8 OR rsi_14 > 80 THEN 'Wait pullback'
            ELSE 'Watch'
          END AS order_bias
        FROM enriched
      ),
      eligible AS (
        -- Closed-only edge floor: enforce when possible, but never blank the
        -- whole current list if today's batch only contains below-floor scanners.
        SELECT * FROM scored
        WHERE NOT edge_floor_blocked
          OR NOT EXISTS (SELECT 1 FROM scored WHERE NOT edge_floor_blocked)
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
      queryLimit,
      EDGE_NO_LOSS_PF_CAP,
    ]);
    const rows =
      mode === "daytrades"
        ? await (async () => {
            const quotes = await fetchBrokerQuotes(result.rows.map((row) => row.ticker));
            const enrichedRows = result.rows.map((row) => {
              const quote = quotes[row.ticker] ?? {};
              const spreadPct = quote.broker_spread_pct ?? null;
              const quoteTime = quote.broker_quote_time ? new Date(quote.broker_quote_time) : null;
              const quoteAgeMinutes =
                quoteTime && !Number.isNaN(quoteTime.getTime())
                  ? Math.round((Date.now() - quoteTime.getTime()) / 60000)
                  : null;
              const avgVolume20 = Number(row.avg_volume_20 ?? 0);
              const intradayRelativeVolume =
                quote.broker_volume != null && avgVolume20 > 0
                  ? Number((quote.broker_volume / avgVolume20).toFixed(2))
                  : null;
              const quotePrice = Number(quote.broker_last ?? quote.broker_ask ?? 0);
              const setupEntry = Number(row.entry_price ?? 0);
              const openConfirmed =
                row.pm_status === "No PM data" &&
                quoteAgeMinutes != null &&
                quoteAgeMinutes <= 3 &&
                quotePrice > 0 &&
                setupEntry > 0 &&
                quotePrice >= setupEntry * 0.995;
              const priorRvol = Number(row.relative_volume ?? 0);
              const oldRvolTerm =
                0.20 * Math.min(priorRvol * 25, 100) - (priorRvol < 1 ? 10 : 0);
              const liveRvolTerm =
                intradayRelativeVolume == null
                  ? oldRvolTerm
                  : 0.20 * Math.min(intradayRelativeVolume * 25, 100) - (intradayRelativeVolume < 1 ? 10 : 0);
              const spreadScore =
                spreadPct == null ? 0 :
                spreadPct <= 0.25 ? 4 :
                spreadPct <= 0.4 ? 2 :
                0;
              const quoteFreshnessScore =
                quoteAgeMinutes == null ? 0 :
                quoteAgeMinutes <= 1 ? 2 :
                quoteAgeMinutes <= 3 ? 1 :
                0;
              const pmSuggestedEntry = Number(row.pm_suggested_entry ?? 0);
              const ask = Number(quote.broker_ask ?? 0);
              const entryDisciplineScore =
                pmSuggestedEntry > 0 && ask > 0 && ask <= pmSuggestedEntry ? 2 :
                pmSuggestedEntry > 0 && ask > 0 && ask <= pmSuggestedEntry * 1.01 ? 1 :
                0;
              const quantityUnderCap = ask > 0 ? Math.floor(500 / ask) : 0;
              const sizeFitScore = quantityUnderCap >= 2 ? 2 : quantityUnderCap >= 1 ? 1 : 0;
              const executionQualityScore =
                spreadScore + quoteFreshnessScore + entryDisciplineScore + sizeFitScore;
              let orderBias =
                spreadPct != null && spreadPct > 1
                  ? "Avoid spread"
                  : spreadPct != null && spreadPct > 0.4 && row.order_bias !== "Avoid"
                    ? "Small only"
                    : row.order_bias;
              if (openConfirmed && orderBias === "Wait") {
                orderBias = row.high_short_interest || row.sector_underperforming ? "Small only" : "Watch";
              }
              if (orderBias === "Pullback" && intradayRelativeVolume != null && intradayRelativeVolume < 1.2) {
                orderBias = "Watch";
              }
              return {
                ...row,
                ...quote,
                pm_status: openConfirmed ? "Open confirmed" : row.pm_status,
                broker_quote_age_minutes: quoteAgeMinutes,
                intraday_relative_volume: intradayRelativeVolume,
                relative_volume_source: "prior_session_daily",
                intraday_relative_volume_source: intradayRelativeVolume == null ? null : "broker_quote_volume_vs_avg_20d",
                execution_quality_score: executionQualityScore,
                execution_quality_parts: {
                  spread: spreadScore,
                  quote_age: quoteFreshnessScore,
                  entry: entryDisciplineScore,
                  size: sizeFitScore,
                },
                candidate_score: Number((Number(row.candidate_score ?? 0) - oldRvolTerm + liveRvolTerm + executionQualityScore).toFixed(1)),
                order_bias: orderBias,
              };
            });
            return enrichedRows
              .sort((a, b) => {
                const tradableStatus = (row: typeof enrichedRows[number]) => {
                  const pmStatus = String(row.pm_status ?? "");
                  const orderBias = String(row.order_bias ?? "");
                  const hardPmReject = ["Stale PM", "No PM data", "PM against", "PM fading"].includes(pmStatus);
                  const hardBiasReject = ["Avoid", "Avoid spread", "Refresh", "Wait", "Wait pullback"].includes(orderBias);
                  return hardPmReject || hardBiasReject ? 0 : 1;
                };
                const tradableDelta = tradableStatus(b) - tradableStatus(a);
                if (tradableDelta !== 0) return tradableDelta;
                const scoreDelta = Number(b.candidate_score ?? 0) - Number(a.candidate_score ?? 0);
                if (scoreDelta !== 0) return scoreDelta;
                return Number(b.rs_percentile ?? 0) - Number(a.rs_percentile ?? 0);
              })
              .slice(0, limit)
              .map((row, index) => ({ ...row, rank: index + 1 }));
          })()
        : result.rows;

    return NextResponse.json({
      rows,
      meta: {
        scoring_mode: mode,
        batch_date: result.rows[0]?.signal_timestamp ?? null,
        edge_window_days: EDGE_WINDOW_DAYS,
        edge_pf_floor: EDGE_PF_FLOOR,
        edge_min_closed: EDGE_MIN_CLOSED,
        edge_no_loss_pf_cap: EDGE_NO_LOSS_PF_CAP,
        max_per_scanner: maxPerScanner,
        daytrade_rvol_source: mode === "daytrades" ? "broker_quote_volume_vs_avg_20d_when_available" : null,
        daytrade_execution_quality: mode === "daytrades" ? "0-10 live score: spread 4, quote age 2, entry discipline 2, size fit 2" : null,
      },
    });
  } catch (error) {
    console.error("top-candidates query failed", error);
    return NextResponse.json({ error: "Failed to load top candidates" }, { status: 500 });
  } finally {
    client.release();
  }
}
