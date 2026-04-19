import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

type CacheRow = {
  epic: string;
  direction: string;
  trade_count: number;
  win_rate: number;
  avg_mfe: number;
  median_mfe: number;
  percentile_25_mfe: number;
  percentile_75_mfe: number;
  avg_mae: number;
  median_mae: number;
  percentile_75_mae: number;
  percentile_95_mae: number;
  max_mae: number;
  optimal_be_trigger: number;
  conservative_be_trigger: number;
  current_be_trigger: number;
  optimal_stop_loss: number;
  current_stop_loss: number;
  configured_stop_loss: number;
  sl_recommendation: string;
  sl_priority: string;
  recommendation: string;
  priority: string;
  confidence: string;
  be_reach_rate: number;
  be_protection_rate: number;
  be_profit_rate: number;
  analysis_notes: string | null;
  analyzed_at: string;
  trades_analyzed: number[] | null;
};

type SummaryPayload = {
  epicDirectionPairs: number;
  totalTradesAnalyzed: number;
  highPriorityBe: number;
  highPrioritySl: number;
  avgWinRate: number;
  analyzedAt: string | null;
};

const toNumberOrNull = (value: unknown) => {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const toNumber = (value: unknown) => Number(value ?? 0);

const cleanEpic = (epic: string) => epic.replace("CS.D.", "").replace(".MINI.IP", "").replace(".CEEM.IP", "");

function normalizeRow(row: Record<string, unknown>): CacheRow {
  return {
    epic: String(row.epic ?? ""),
    direction: String(row.direction ?? ""),
    trade_count: toNumber(row.trade_count),
    win_rate: toNumber(row.win_rate),
    avg_mfe: toNumber(row.avg_mfe),
    median_mfe: toNumber(row.median_mfe),
    percentile_25_mfe: toNumber(row.percentile_25_mfe),
    percentile_75_mfe: toNumber(row.percentile_75_mfe),
    avg_mae: toNumber(row.avg_mae),
    median_mae: toNumber(row.median_mae),
    percentile_75_mae: toNumber(row.percentile_75_mae),
    percentile_95_mae: toNumber(row.percentile_95_mae),
    max_mae: toNumber(row.max_mae),
    optimal_be_trigger: toNumber(row.optimal_be_trigger),
    conservative_be_trigger: toNumber(row.conservative_be_trigger),
    current_be_trigger: toNumber(row.current_be_trigger),
    optimal_stop_loss: toNumber(row.optimal_stop_loss),
    current_stop_loss: toNumber(row.current_stop_loss),
    configured_stop_loss: toNumber(row.configured_stop_loss),
    sl_recommendation: String(row.sl_recommendation ?? ""),
    sl_priority: String(row.sl_priority ?? ""),
    recommendation: String(row.recommendation ?? ""),
    priority: String(row.priority ?? ""),
    confidence: String(row.confidence ?? ""),
    be_reach_rate: toNumber(row.be_reach_rate),
    be_protection_rate: toNumber(row.be_protection_rate),
    be_profit_rate: toNumber(row.be_profit_rate),
    analysis_notes: row.analysis_notes ? String(row.analysis_notes) : null,
    analyzed_at: String(row.analyzed_at ?? ""),
    trades_analyzed: Array.isArray(row.trades_analyzed)
      ? row.trades_analyzed.map((item) => Number(item)).filter((item) => Number.isFinite(item))
      : null,
  };
}

function buildSummary(rows: CacheRow[]): SummaryPayload {
  const totalTradesAnalyzed = rows.reduce((sum, row) => sum + row.trade_count, 0);
  const avgWinRate =
    rows.length > 0 ? rows.reduce((sum, row) => sum + row.win_rate, 0) / rows.length : 0;

  return {
    epicDirectionPairs: rows.length,
    totalTradesAnalyzed,
    highPriorityBe: rows.filter((row) => row.priority === "high").length,
    highPrioritySl: rows.filter((row) => row.sl_priority === "high").length,
    avgWinRate,
    analyzedAt: rows[0]?.analyzed_at ?? null,
  };
}

export async function GET() {
  try {
    const tableResult = await forexPool.query<{ exists: boolean }>(
      `
      SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'breakeven_analysis_cache'
      ) AS exists
      `
    );

    const cacheExists = tableResult.rows[0]?.exists ?? false;
    if (!cacheExists) {
      return NextResponse.json({
        cacheExists: false,
        summary: null,
        rows: [],
      });
    }

    const result = await forexPool.query(
      `
      SELECT
        epic,
        direction,
        trade_count,
        win_rate,
        avg_mfe,
        median_mfe,
        percentile_25_mfe,
        percentile_75_mfe,
        avg_mae,
        median_mae,
        percentile_75_mae,
        percentile_95_mae,
        max_mae,
        optimal_be_trigger,
        conservative_be_trigger,
        current_be_trigger,
        optimal_stop_loss,
        current_stop_loss,
        configured_stop_loss,
        sl_recommendation,
        sl_priority,
        recommendation,
        priority,
        confidence,
        be_reach_rate,
        be_protection_rate,
        be_profit_rate,
        analysis_notes,
        analyzed_at,
        trades_analyzed
      FROM breakeven_analysis_cache
      ORDER BY
        CASE priority
          WHEN 'high' THEN 1
          WHEN 'medium' THEN 2
          ELSE 3
        END,
        CASE sl_priority
          WHEN 'high' THEN 1
          WHEN 'medium' THEN 2
          ELSE 3
        END,
        epic,
        direction
      `
    );

    const rows = result.rows.map((row) => normalizeRow(row as Record<string, unknown>));
    const summary = buildSummary(rows);

    return NextResponse.json({
      cacheExists: true,
      summary,
      rows: rows.map((row) => ({
        ...row,
        epic_display: cleanEpic(row.epic),
        be_diff: Number((row.optimal_be_trigger - row.current_be_trigger).toFixed(1)),
        sl_diff: Number((row.optimal_stop_loss - row.configured_stop_loss).toFixed(1)),
        sl_mismatch: Math.abs(row.current_stop_loss - row.configured_stop_loss) > 5,
      })),
    });
  } catch (error) {
    console.error("Failed to load breakeven optimizer cache", error);
    return NextResponse.json({ error: "Failed to load breakeven optimizer cache" }, { status: 500 });
  }
}
