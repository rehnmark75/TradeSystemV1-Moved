import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";
import { parsePositiveInt } from "../../../../lib/backtests";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;
const MIN_RELIABLE_TRADES = 10;
const HIGH_CONFIDENCE_TRADES = 30;

type StatRow = Record<string, unknown> & {
  alignment?: string;
  total_signals?: unknown;
  total_trades?: unknown;
  wins?: unknown;
  losses?: unknown;
  breakeven?: unknown;
  win_rate?: unknown;
  total_pnl?: unknown;
  avg_pnl?: unknown;
  avg_win?: unknown;
  avg_loss?: unknown;
  profit_factor?: unknown;
};

const BASE_CTE = `
  WITH alert_outcomes AS (
    SELECT
      ah.id AS alert_id,
      ah.pair,
      ah.signal_type,
      ah.htf_candle_direction,
      ah.htf_candle_direction_prev,
      COALESCE(ah.market_regime_detected, ah.market_regime, 'UNKNOWN') AS regime,
      CASE
        WHEN ah.claude_approved IS TRUE THEN 'APPROVED'
        WHEN ah.claude_approved IS FALSE THEN 'REJECTED'
        ELSE 'NOT_ANALYZED'
      END AS claude_bucket,
      CASE
        WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BULLISH')
          OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BEARISH')
        THEN 'ALIGNED'
        WHEN (ah.signal_type = 'BULL' AND ah.htf_candle_direction = 'BEARISH')
          OR (ah.signal_type = 'BEAR' AND ah.htf_candle_direction = 'BULLISH')
        THEN 'COUNTER'
        ELSE 'NEUTRAL'
      END AS alignment,
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
`;

const METRIC_SELECT = `
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
  ROUND(COALESCE(AVG(profit_loss), 0)::numeric, 2) AS avg_pnl,
  ROUND(COALESCE(AVG(profit_loss) FILTER (WHERE profit_loss > 0), 0)::numeric, 2) AS avg_win,
  ROUND(COALESCE(AVG(profit_loss) FILTER (WHERE profit_loss < 0), 0)::numeric, 2) AS avg_loss,
  ROUND(
    CASE
      WHEN ABS(COALESCE(SUM(profit_loss) FILTER (WHERE profit_loss < 0), 0)) > 0
      THEN COALESCE(SUM(profit_loss) FILTER (WHERE profit_loss > 0), 0)
        / ABS(COALESCE(SUM(profit_loss) FILTER (WHERE profit_loss < 0), 0))
      WHEN COALESCE(SUM(profit_loss) FILTER (WHERE profit_loss > 0), 0) > 0 THEN 99
      ELSE 0
    END::numeric,
    2
  ) AS profit_factor
`;

function toNumber(value: unknown): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function sampleConfidence(totalTrades: number): "HIGH" | "MEDIUM" | "LOW" {
  if (totalTrades >= HIGH_CONFIDENCE_TRADES) return "HIGH";
  if (totalTrades >= MIN_RELIABLE_TRADES) return "MEDIUM";
  return "LOW";
}

function recommendationFor(row: StatRow): string {
  const totalTrades = toNumber(row.total_trades);
  const expectancy = toNumber(row.avg_pnl);
  const profitFactor = toNumber(row.profit_factor);

  if (totalTrades < MIN_RELIABLE_TRADES) return "INSUFFICIENT DATA";
  if (expectancy > 0 && profitFactor >= 1.2) return "FAVORABLE";
  if (expectancy < 0 || (profitFactor > 0 && profitFactor < 0.9)) return "AVOID";
  return "WATCH";
}

function normalizeStats<T extends StatRow>(row: T) {
  const totalSignals = toNumber(row.total_signals);
  const totalTrades = toNumber(row.total_trades);

  return {
    ...row,
    total_signals: totalSignals,
    total_trades: totalTrades,
    wins: toNumber(row.wins),
    losses: toNumber(row.losses),
    breakeven: toNumber(row.breakeven),
    win_rate: toNumber(row.win_rate),
    total_pnl: toNumber(row.total_pnl),
    avg_pnl: toNumber(row.avg_pnl),
    avg_win: toNumber(row.avg_win),
    avg_loss: toNumber(row.avg_loss),
    profit_factor: toNumber(row.profit_factor),
    trade_rate: totalSignals > 0 ? Number(((totalTrades / totalSignals) * 100).toFixed(1)) : 0,
    sample_confidence: sampleConfidence(totalTrades),
    recommendation: recommendationFor(row),
  };
}

function buildDecision(alignmentRows: ReturnType<typeof normalizeStats>[]) {
  const aligned = alignmentRows.find((row) => row.alignment === "ALIGNED");
  const counter = alignmentRows.find((row) => row.alignment === "COUNTER");
  const alignedExpectancy = aligned?.avg_pnl ?? 0;
  const counterExpectancy = counter?.avg_pnl ?? 0;
  const counterTrades = counter?.total_trades ?? 0;
  const counterPnl = counter?.total_pnl ?? 0;

  let verdict = "INSUFFICIENT DATA";
  let action = "Keep monitoring until counter-HTF has at least 10 executed trades.";

  if (counterTrades >= MIN_RELIABLE_TRADES) {
    if (counterExpectancy < 0 || counterPnl < 0) {
      verdict = "BLOCK COUNTER";
      action = "Counter-HTF is damaging realized P&L in this window.";
    } else if (aligned && aligned.total_trades >= MIN_RELIABLE_TRADES && alignedExpectancy > counterExpectancy * 1.25) {
      verdict = "PREFER ALIGNED";
      action = "Aligned HTF has a stronger expectancy; use counter trades selectively.";
    } else {
      verdict = "ALLOW COUNTER";
      action = "Counter-HTF is not currently worse than aligned on realized expectancy.";
    }
  }

  return {
    verdict,
    action,
    aligned_expectancy: alignedExpectancy,
    counter_expectancy: counterExpectancy,
    expectancy_delta: Number((alignedExpectancy - counterExpectancy).toFixed(2)),
    win_rate_delta: Number(((aligned?.win_rate ?? 0) - (counter?.win_rate ?? 0)).toFixed(1)),
    pnl_impact_if_counter_blocked: Number((-counterPnl).toFixed(2)),
    counter_trades: counterTrades,
    missed_winners: counter?.wins ?? 0,
    avoided_losers: counter?.losses ?? 0,
    sample_confidence: sampleConfidence(Math.min(aligned?.total_trades ?? 0, counterTrades)),
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parsePositiveInt(searchParams.get("days"), DEFAULT_DAYS);
  const env = searchParams.get("env") || "demo";

  try {
    const [
      alignmentResult,
      patternsResult,
      pairsResult,
      regimesResult,
      claudeResult,
      distributionResult,
    ] = await Promise.all([
      forexPool.query(
        `
        ${BASE_CTE}
        SELECT
          alignment,
          ${METRIC_SELECT}
        FROM alert_outcomes
        GROUP BY alignment
        ORDER BY alignment
        `,
        [days, env]
      ),
      forexPool.query(
        `
        ${BASE_CTE}
        SELECT
          pattern,
          signal_type,
          ${METRIC_SELECT}
        FROM alert_outcomes
        GROUP BY pattern, signal_type
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) DESC, COALESCE(AVG(profit_loss), 0) DESC
        LIMIT 20
        `,
        [days, env]
      ),
      forexPool.query(
        `
        ${BASE_CTE}
        SELECT
          pair,
          alignment,
          ${METRIC_SELECT}
        FROM alert_outcomes
        WHERE htf_candle_direction != 'NEUTRAL'
        GROUP BY pair, alignment
        HAVING COUNT(*) >= 2
        ORDER BY pair, alignment
        `,
        [days, env]
      ),
      forexPool.query(
        `
        ${BASE_CTE}
        SELECT
          regime,
          alignment,
          ${METRIC_SELECT}
        FROM alert_outcomes
        GROUP BY regime, alignment
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(CASE WHEN outcome IS NOT NULL THEN 1 END) DESC, COALESCE(SUM(profit_loss), 0) DESC
        LIMIT 16
        `,
        [days, env]
      ),
      forexPool.query(
        `
        ${BASE_CTE}
        SELECT
          claude_bucket,
          alignment,
          ${METRIC_SELECT}
        FROM alert_outcomes
        GROUP BY claude_bucket, alignment
        ORDER BY claude_bucket, alignment
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

    const alignment = alignmentResult.rows.map(normalizeStats);
    const patterns = patternsResult.rows.map(normalizeStats);
    const pairs = pairsResult.rows.map(normalizeStats);
    const regimes = regimesResult.rows.map(normalizeStats);
    const claude = claudeResult.rows.map(normalizeStats);

    return NextResponse.json({
      days,
      decision: buildDecision(alignment),
      alignment,
      patterns,
      pairs,
      regimes,
      claude,
      distribution: distributionResult.rows,
    });
  } catch (error) {
    console.error("Failed to load HTF analysis", error);
    return NextResponse.json({ error: "Failed to load HTF analysis" }, { status: 500 });
  }
}
