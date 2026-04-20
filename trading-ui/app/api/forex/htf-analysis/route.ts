import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";
import { parsePositiveInt } from "../../../../lib/backtests";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parsePositiveInt(searchParams.get("days"), DEFAULT_DAYS);
  const env = searchParams.get("env") || "demo";

  try {
    const [alignmentResult, patternsResult, pairsResult, distributionResult] = await Promise.all([
      forexPool.query(
        `
        WITH alert_outcomes AS (
          SELECT
            ah.id AS alert_id,
            ah.signal_type,
            ah.htf_candle_direction,
            ah.htf_candle_direction_prev,
            CASE
              WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BULLISH')
                OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BEARISH')
              THEN 'ALIGNED'
              WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BEARISH')
                OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BULLISH')
              THEN 'COUNTER'
              ELSE 'NEUTRAL'
            END AS alignment,
            tl.profit_loss,
            CASE
              WHEN tl.profit_loss > 0 THEN 'WIN'
              WHEN tl.profit_loss < 0 THEN 'LOSS'
              WHEN tl.profit_loss = 0 THEN 'BREAKEVEN'
              ELSE NULL
            END AS outcome
          FROM alert_history ah
          LEFT JOIN trade_log tl
            ON tl.alert_id = ah.id
            AND tl.environment = $2
          WHERE ah.alert_timestamp >= NOW() - ($1::text || ' days')::interval
            AND ah.htf_candle_direction IS NOT NULL
            AND ah.environment = $2
        )
        SELECT
          alignment,
          COUNT(*) AS total_signals,
          COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) AS total_trades,
          COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) AS wins,
          COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) AS losses,
          COUNT(CASE WHEN outcome = 'BREAKEVEN' THEN 1 END) AS breakeven,
          ROUND(
            CASE
              WHEN COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) > 0
              THEN COUNT(CASE WHEN outcome = 'WIN' THEN 1 END)::numeric
                 / COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) * 100
              ELSE 0
            END,
            1
          ) AS win_rate,
          COALESCE(SUM(profit_loss), 0) AS total_pnl,
          ROUND(COALESCE(AVG(profit_loss), 0)::numeric, 2) AS avg_pnl
        FROM alert_outcomes
        GROUP BY alignment
        ORDER BY alignment
        `,
        [days, env]
      ),
      forexPool.query(
        `
        WITH alert_outcomes AS (
          SELECT
            ah.signal_type,
            ah.htf_candle_direction || '_' || COALESCE(ah.htf_candle_direction_prev, 'UNKNOWN') AS pattern,
            tl.profit_loss,
            CASE
              WHEN tl.profit_loss > 0 THEN 'WIN'
              WHEN tl.profit_loss < 0 THEN 'LOSS'
              WHEN tl.profit_loss = 0 THEN 'BREAKEVEN'
              ELSE NULL
            END AS outcome
          FROM alert_history ah
          LEFT JOIN trade_log tl
            ON tl.alert_id = ah.id
            AND tl.environment = $2
          WHERE ah.alert_timestamp >= NOW() - ($1::text || ' days')::interval
            AND ah.htf_candle_direction IS NOT NULL
            AND ah.environment = $2
        )
        SELECT
          pattern,
          signal_type,
          COUNT(*) AS total_signals,
          COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) AS total_trades,
          COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) AS wins,
          COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) AS losses,
          ROUND(
            CASE
              WHEN COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) > 0
              THEN COUNT(CASE WHEN outcome = 'WIN' THEN 1 END)::numeric
                 / COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) * 100
              ELSE 0
            END,
            1
          ) AS win_rate,
          COALESCE(SUM(profit_loss), 0) AS total_pnl,
          ROUND(COALESCE(AVG(profit_loss), 0)::numeric, 2) AS avg_pnl
        FROM alert_outcomes
        GROUP BY pattern, signal_type
        HAVING COUNT(*) >= 2
        ORDER BY total_trades DESC, win_rate DESC
        LIMIT 20
        `,
        [days, env]
      ),
      forexPool.query(
        `
        WITH alert_outcomes AS (
          SELECT
            ah.pair,
            ah.signal_type,
            ah.htf_candle_direction,
            CASE
              WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BULLISH')
                OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BEARISH')
              THEN 'ALIGNED'
              ELSE 'COUNTER'
            END AS alignment,
            tl.profit_loss,
            CASE
              WHEN tl.profit_loss > 0 THEN 'WIN'
              WHEN tl.profit_loss < 0 THEN 'LOSS'
              ELSE NULL
            END AS outcome
          FROM alert_history ah
          LEFT JOIN trade_log tl
            ON tl.alert_id = ah.id
            AND tl.environment = $2
          WHERE ah.alert_timestamp >= NOW() - ($1::text || ' days')::interval
            AND ah.htf_candle_direction IS NOT NULL
            AND ah.htf_candle_direction != 'NEUTRAL'
            AND ah.environment = $2
        )
        SELECT
          pair,
          alignment,
          COUNT(*) AS total_signals,
          COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) AS total_trades,
          COUNT(CASE WHEN outcome = 'WIN' THEN 1 END) AS wins,
          COUNT(CASE WHEN outcome = 'LOSS' THEN 1 END) AS losses,
          ROUND(
            CASE
              WHEN COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) > 0
              THEN COUNT(CASE WHEN outcome = 'WIN' THEN 1 END)::numeric
                 / COUNT(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 END) * 100
              ELSE 0
            END,
            1
          ) AS win_rate,
          COALESCE(SUM(profit_loss), 0) AS total_pnl
        FROM alert_outcomes
        GROUP BY pair, alignment
        HAVING COUNT(*) >= 2
        ORDER BY pair, alignment
        `,
        [days, env]
      ),
      forexPool.query(
        `
        SELECT
          htf_candle_direction AS direction,
          signal_type,
          COUNT(*) AS count
        FROM alert_history
        WHERE alert_timestamp >= NOW() - ($1::text || ' days')::interval
          AND htf_candle_direction IS NOT NULL
          AND environment = $2
        GROUP BY htf_candle_direction, signal_type
        ORDER BY htf_candle_direction, signal_type
        `,
        [days, env]
      ),
    ]);

    return NextResponse.json({
      days,
      alignment: alignmentResult.rows,
      patterns: patternsResult.rows,
      pairs: pairsResult.rows,
      distribution: distributionResult.rows,
    });
  } catch (error) {
    console.error("Failed to load HTF analysis", error);
    return NextResponse.json({ error: "Failed to load HTF analysis" }, { status: 500 });
  }
}
