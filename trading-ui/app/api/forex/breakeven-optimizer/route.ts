import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 60;
const CANDIDATE_TRIGGERS = [4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35];

type TradeRow = {
  id: number;
  strategy: string;
  epic: string;
  epic_display: string;
  direction: string;
  profit_loss: number;
  mfe: number;
  mae: number;
  moved_to_breakeven: boolean;
  early_be_executed: boolean;
};

function parseDays(value: string | null) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_DAYS;
  return Math.min(parsed, 365);
}

function cleanEpic(epic: string) {
  return epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", "");
}

function toNumber(value: unknown) {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function average(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function percentile(values: number[], pct: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * pct;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function priorityFor(sampleSize: number, edge: number, cutWinnerRisk: number) {
  if (sampleSize < 8) return "low";
  if (edge >= 20 && cutWinnerRisk <= 35) return "high";
  if (edge >= 8 && cutWinnerRisk <= 45) return "medium";
  return "low";
}

function evaluateGroup(key: string, trades: TradeRow[]) {
  const [strategy, epic, direction] = key.split("|");
  const closed = trades.filter((trade) => trade.profit_loss !== 0 || trade.mfe > 0 || trade.mae > 0);
  const winners = closed.filter((trade) => trade.profit_loss > 0);
  const losers = closed.filter((trade) => trade.profit_loss < 0);
  const avgWinnerPnl = average(winners.map((trade) => trade.profit_loss));
  const avgLoserAbsPnl = Math.abs(average(losers.map((trade) => trade.profit_loss)));
  const medianWinnerMfe = percentile(winners.map((trade) => trade.mfe), 0.5);
  const p75WinnerMfe = percentile(winners.map((trade) => trade.mfe), 0.75);
  const p50LoserMfe = percentile(losers.map((trade) => trade.mfe), 0.5);

  const candidates = CANDIDATE_TRIGGERS.map((trigger) => {
    const reached = closed.filter((trade) => trade.mfe >= trigger);
    const savedLosers = losers.filter((trade) => trade.mfe >= trigger);
    const winnersReached = winners.filter((trade) => trade.mfe >= trigger);
    const cutWinnerRisk = winners.length ? ((winners.length - winnersReached.length) / winners.length) * 100 : 0;
    const reachRate = closed.length ? (reached.length / closed.length) * 100 : 0;
    const saveRate = losers.length ? (savedLosers.length / losers.length) * 100 : 0;
    const opportunityScore = (saveRate / 100) * avgLoserAbsPnl;
    const cutRiskScore = (cutWinnerRisk / 100) * avgWinnerPnl * 0.35;
    const score = opportunityScore - cutRiskScore;
    return { trigger, reachRate, saveRate, cutWinnerRisk, score };
  });

  const viable = candidates.filter((candidate) => candidate.reachRate >= 20 && candidate.cutWinnerRisk <= 55);
  const best = [...(viable.length ? viable : candidates)].sort((a, b) => b.score - a.score)[0];
  const currentBeRate = closed.length
    ? (closed.filter((trade) => trade.moved_to_breakeven || trade.early_be_executed).length / closed.length) * 100
    : 0;
  const edge = best ? best.score : 0;
  const priority = priorityFor(closed.length, edge, best?.cutWinnerRisk ?? 100);
  const recommendation =
    closed.length < 8
      ? "COLLECT_MORE_DATA"
      : !best || best.score <= 0
      ? "DO_NOT_TIGHTEN_BE"
      : best.trigger <= Math.max(4, p50LoserMfe)
      ? "TEST_EARLIER_BE"
      : "TEST_SELECTIVE_BE";

  return {
    strategy,
    epic,
    epic_display: cleanEpic(epic),
    direction,
    trades: closed.length,
    wins: winners.length,
    losses: losers.length,
    win_rate: closed.length ? (winners.length / closed.length) * 100 : 0,
    avg_mfe: average(closed.map((trade) => trade.mfe)),
    median_winner_mfe: medianWinnerMfe,
    p75_winner_mfe: p75WinnerMfe,
    avg_mae: average(closed.map((trade) => trade.mae)),
    p75_mae: percentile(closed.map((trade) => trade.mae), 0.75),
    current_be_rate: currentBeRate,
    recommended_trigger: best?.trigger ?? null,
    reach_rate: best?.reachRate ?? 0,
    save_rate: best?.saveRate ?? 0,
    cut_winner_risk: best?.cutWinnerRisk ?? 0,
    policy_score: edge,
    recommendation,
    priority,
    rationale:
      recommendation === "COLLECT_MORE_DATA"
        ? "Too few closed trades for a reliable BE policy."
        : recommendation === "DO_NOT_TIGHTEN_BE"
        ? "Candidate BE triggers do not show enough loser protection after accounting for winner interruption risk."
        : `Best candidate is ${best?.trigger ?? "N/A"} pips: reach ${best?.reachRate.toFixed(0)}%, loser save ${best?.saveRate.toFixed(0)}%, cut-winner risk ${best?.cutWinnerRisk.toFixed(0)}%.`,
    candidates,
  };
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const env = searchParams.get("env") || "demo";
  const strategyFilter = searchParams.get("strategy");

  try {
    const result = await forexPool.query(
      `
      SELECT
        t.id,
        COALESCE(a.strategy, 'UNKNOWN') as strategy,
        t.symbol as epic,
        t.direction,
        t.profit_loss,
        t.vsl_peak_profit_pips as mfe,
        t.vsl_mae_pips as mae,
        t.moved_to_breakeven,
        t.early_be_executed
      FROM trade_log t
      LEFT JOIN alert_history a ON t.alert_id = a.id
      WHERE t.timestamp >= NOW() - ($1::int * INTERVAL '1 day')
      AND t.environment = $2
      AND t.status IN ('closed', 'expired')
      AND t.profit_loss IS NOT NULL
      AND ($3::text IS NULL OR UPPER(COALESCE(a.strategy, 'UNKNOWN')) = UPPER($3::text))
      ORDER BY t.timestamp DESC
      `,
      [days, env, strategyFilter && strategyFilter !== "ALL" ? strategyFilter : null]
    );

    const trades: TradeRow[] = result.rows.map((row) => ({
      id: Number(row.id),
      strategy: String(row.strategy ?? "UNKNOWN").toUpperCase(),
      epic: String(row.epic ?? ""),
      epic_display: cleanEpic(String(row.epic ?? "")),
      direction: String(row.direction ?? ""),
      profit_loss: toNumber(row.profit_loss),
      mfe: toNumber(row.mfe),
      mae: toNumber(row.mae),
      moved_to_breakeven: Boolean(row.moved_to_breakeven),
      early_be_executed: Boolean(row.early_be_executed),
    }));

    const groups = new Map<string, TradeRow[]>();
    trades.forEach((trade) => {
      const key = `${trade.strategy}|${trade.epic}|${trade.direction}`;
      groups.set(key, [...(groups.get(key) ?? []), trade]);
    });

    const rows = [...groups.entries()]
      .map(([key, groupTrades]) => evaluateGroup(key, groupTrades))
      .sort((a, b) => {
        const order = { high: 0, medium: 1, low: 2 } as Record<string, number>;
        return order[a.priority] - order[b.priority] || b.policy_score - a.policy_score || b.trades - a.trades;
      });

    const strategyOptions = [...new Set(trades.map((trade) => trade.strategy))].sort();
    const actionable = rows.filter((row) => row.recommendation === "TEST_EARLIER_BE" || row.recommendation === "TEST_SELECTIVE_BE");
    const closedTrades = trades.length;
    const winners = trades.filter((trade) => trade.profit_loss > 0).length;
    return NextResponse.json({
      source: "live_trade_log",
      days,
      env,
      strategy_options: strategyOptions,
      summary: {
        groups: rows.length,
        trades: closedTrades,
        win_rate: closedTrades ? (winners / closedTrades) * 100 : 0,
        actionable: actionable.length,
        high_priority: rows.filter((row) => row.priority === "high").length,
        avg_current_be_rate: average(rows.map((row) => row.current_be_rate)),
      },
      rows,
    });
  } catch (error) {
    console.error("Failed to load breakeven policy evaluator", error);
    return NextResponse.json({ error: "Failed to load breakeven policy evaluator" }, { status: 500 });
  }
}
