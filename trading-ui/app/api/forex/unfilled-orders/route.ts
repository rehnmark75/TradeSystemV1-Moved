import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

type SummaryRow = {
  total_unfilled: number;
  would_fill_4h: number;
  would_fill_24h: number;
  good_signals: number;
  bad_signals: number;
  inconclusive_signals: number;
  win_rate_pct: number | null;
};

type DetailRow = {
  id: number;
  symbol: string;
  direction: string;
  order_time: string;
  expiry_time: string | null;
  entry_level: number | null;
  stop_loss: number | null;
  take_profit: number | null;
  price_at_expiry: number | null;
  gap_to_entry_pips: number | null;
  would_fill_4h: boolean | null;
  outcome_4h: string | null;
  would_fill_24h: boolean | null;
  outcome_24h: string | null;
  signal_quality: string | null;
  max_favorable_pips: number | null;
  max_adverse_pips: number | null;
  alert_id: number | null;
};

type EpicBreakdownRow = {
  symbol: string;
  total_unfilled: number;
  good: number;
  bad: number;
  inconclusive: number;
  avg_gap_pips: number | null;
  avg_favorable: number | null;
  avg_adverse: number | null;
};

type RecommendationRow = EpicBreakdownRow & {
  issues: string[];
  recommendations: string[];
};

const toNumberOrNull = (value: unknown) => {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const toNumber = (value: unknown) => Number(value ?? 0);

function normalizeSummary(row: Record<string, unknown> | null): SummaryRow | null {
  if (!row) return null;

  return {
    total_unfilled: toNumber(row.total_unfilled),
    would_fill_4h: toNumber(row.would_fill_4h),
    would_fill_24h: toNumber(row.would_fill_24h),
    good_signals: toNumber(row.good_signals),
    bad_signals: toNumber(row.bad_signals),
    inconclusive_signals: toNumber(row.inconclusive_signals),
    win_rate_pct: toNumberOrNull(row.win_rate_pct),
  };
}

function normalizeDetail(row: Record<string, unknown>): DetailRow {
  return {
    id: toNumber(row.id),
    symbol: String(row.symbol ?? ""),
    direction: String(row.direction ?? ""),
    order_time: String(row.order_time ?? ""),
    expiry_time: row.expiry_time ? String(row.expiry_time) : null,
    entry_level: toNumberOrNull(row.entry_level),
    stop_loss: toNumberOrNull(row.stop_loss),
    take_profit: toNumberOrNull(row.take_profit),
    price_at_expiry: toNumberOrNull(row.price_at_expiry),
    gap_to_entry_pips: toNumberOrNull(row.gap_to_entry_pips),
    would_fill_4h: typeof row.would_fill_4h === "boolean" ? row.would_fill_4h : null,
    outcome_4h: row.outcome_4h ? String(row.outcome_4h) : null,
    would_fill_24h: typeof row.would_fill_24h === "boolean" ? row.would_fill_24h : null,
    outcome_24h: row.outcome_24h ? String(row.outcome_24h) : null,
    signal_quality: row.signal_quality ? String(row.signal_quality) : null,
    max_favorable_pips: toNumberOrNull(row.max_favorable_pips),
    max_adverse_pips: toNumberOrNull(row.max_adverse_pips),
    alert_id: toNumberOrNull(row.alert_id),
  };
}

function normalizeEpicBreakdown(row: Record<string, unknown>): EpicBreakdownRow {
  return {
    symbol: String(row.symbol ?? ""),
    total_unfilled: toNumber(row.total_unfilled),
    good: toNumber(row.good),
    bad: toNumber(row.bad),
    inconclusive: toNumber(row.inconclusive),
    avg_gap_pips: toNumberOrNull(row.avg_gap_pips),
    avg_favorable: toNumberOrNull(row.avg_favorable),
    avg_adverse: toNumberOrNull(row.avg_adverse),
  };
}

function buildRecommendations(rows: EpicBreakdownRow[]): RecommendationRow[] {
  return rows
    .filter((row) => row.total_unfilled >= 2)
    .map((row) => {
      const issues: string[] = [];
      const recommendations: string[] = [];

      if ((row.avg_gap_pips ?? 0) > 5) {
        issues.push(`High avg gap to entry: ${row.avg_gap_pips} pips`);
        recommendations.push("Consider reducing stop-entry offset or extending expiry.");
      }

      if (row.bad > row.good && row.good + row.bad > 0) {
        issues.push(`More bad signals (${row.bad}) than good (${row.good}).`);
        recommendations.push("Review entry direction logic. Unfilled orders often resolved against the intended setup.");
      }

      if (row.good > 0 && row.bad === 0) {
        issues.push(`All decisive signals were good (${row.good}).`);
        recommendations.push("Consider extending expiry time. Good setups may be expiring before they retrace into entry.");
      }

      if ((row.avg_gap_pips ?? Number.POSITIVE_INFINITY) < 3) {
        issues.push(`Entries are already close to market (${row.avg_gap_pips} pips average gap).`);
      }

      return {
        ...row,
        issues,
        recommendations,
      };
    });
}

export async function GET() {
  try {
    const viewExistsResult = await forexPool.query<{ exists: boolean }>(
      `
      SELECT EXISTS (
        SELECT 1
        FROM information_schema.views
        WHERE table_schema = 'public'
          AND table_name = 'v_unfilled_order_analysis'
      ) AS exists
      `
    );

    const viewExists = viewExistsResult.rows[0]?.exists ?? false;
    if (!viewExists) {
      return NextResponse.json({
        viewExists: false,
        summary: null,
        detail: [],
        epicBreakdown: [],
        recommendations: [],
      });
    }

    const [summaryResult, detailResult, epicBreakdownResult] = await Promise.all([
      forexPool.query<SummaryRow>(
        `
        SELECT
          total_unfilled,
          would_fill_4h,
          would_fill_24h,
          good_signals,
          bad_signals,
          inconclusive AS inconclusive_signals,
          win_rate_pct
        FROM v_unfilled_order_summary
        LIMIT 1
        `
      ),
      forexPool.query<DetailRow>(
        `
        SELECT
          id,
          symbol,
          direction,
          order_time,
          expiry_time,
          entry_level,
          stop_loss,
          take_profit,
          price_at_expiry,
          gap_to_entry_pips,
          would_fill_4h,
          outcome_4h,
          would_fill_24h,
          outcome_24h,
          signal_quality,
          max_favorable_pips,
          max_adverse_pips,
          alert_id
        FROM v_unfilled_order_analysis
        WHERE symbol NOT LIKE '%CEEM%'
        ORDER BY order_time DESC
        `
      ),
      forexPool.query<EpicBreakdownRow>(
        `
        SELECT
          symbol,
          COUNT(*) AS total_unfilled,
          SUM(CASE WHEN signal_quality = 'GOOD_SIGNAL' THEN 1 ELSE 0 END) AS good,
          SUM(CASE WHEN signal_quality = 'BAD_SIGNAL' THEN 1 ELSE 0 END) AS bad,
          SUM(CASE WHEN signal_quality = 'INCONCLUSIVE' THEN 1 ELSE 0 END) AS inconclusive,
          ROUND(AVG(gap_to_entry_pips)::numeric, 1) AS avg_gap_pips,
          ROUND(AVG(max_favorable_pips)::numeric, 1) AS avg_favorable,
          ROUND(AVG(max_adverse_pips)::numeric, 1) AS avg_adverse
        FROM v_unfilled_order_analysis
        WHERE symbol NOT LIKE '%CEEM%'
        GROUP BY symbol
        ORDER BY total_unfilled DESC, symbol
        `
      ),
    ]);

    const summary = normalizeSummary(summaryResult.rows[0] as Record<string, unknown> | null);
    const detail = detailResult.rows.map((row) => normalizeDetail(row as Record<string, unknown>));
    const epicBreakdown = epicBreakdownResult.rows.map((row) =>
      normalizeEpicBreakdown(row as Record<string, unknown>)
    );

    return NextResponse.json({
      viewExists: true,
      summary,
      detail,
      epicBreakdown,
      recommendations: buildRecommendations(epicBreakdown),
    });
  } catch (error) {
    console.error("Failed to load unfilled orders analysis", error);
    return NextResponse.json({ error: "Failed to load unfilled orders analysis" }, { status: 500 });
  }
}
