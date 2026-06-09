import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

function normalizeSymbol(symbol: string | null) {
  if (!symbol) return "";
  return symbol
    .replace("CS.D.", "")
    .replace(".MINI.IP", "")
    .replace(".CEEM.IP", "")
    .replace(".CEE.IP", "");
}

const num = (v: unknown) => (v == null ? null : Number(v));

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const env = searchParams.get("env") || "demo";
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    // Per-strategy summary (resolved rows only — OPEN rows are still maturing).
    const summaryResult = await forexPool.query(
      `
      SELECT
        strategy,
        COUNT(*)                                            AS n,
        AVG(mfe_pips)                                       AS avg_mfe,
        AVG(mae_pips)                                       AS avg_mae,
        AVG(early_mae_pips)                                 AS avg_early_mae,
        AVG(pnl_1440m_pips)                                 AS avg_pnl_1440m,
        COUNT(*) FILTER (WHERE ref_outcome = 'HIT_TP')      AS tp,
        COUNT(*) FILTER (WHERE ref_outcome = 'HIT_SL')      AS sl,
        AVG(ref_pnl_pips)                                   AS avg_ref_pnl
      FROM monitor_only_outcomes
      WHERE signal_timestamp >= $1
        AND environment = $2
        AND status = 'RESOLVED'
      GROUP BY strategy
      ORDER BY n DESC
      `,
      [since, env]
    );

    const summary = (summaryResult.rows ?? []).map((row) => {
      const tp = Number(row.tp ?? 0);
      const sl = Number(row.sl ?? 0);
      return {
        strategy: row.strategy,
        n: Number(row.n ?? 0),
        avg_mfe: num(row.avg_mfe),
        avg_mae: num(row.avg_mae),
        avg_early_mae: num(row.avg_early_mae),
        avg_pnl_1440m: num(row.avg_pnl_1440m),
        tp,
        sl,
        ref_wr: tp + sl ? (tp / (tp + sl)) * 100 : null,
        avg_ref_pnl: num(row.avg_ref_pnl)
      };
    });

    // Detail rows (most recent first, capped).
    const rowsResult = await forexPool.query(
      `
      SELECT
        alert_id, strategy, epic, pair, direction, signal_timestamp, status,
        entry_price, mfe_pips, mae_pips, early_mae_pips,
        pnl_60m_pips, pnl_240m_pips, pnl_1440m_pips,
        ref_sl_pips, ref_tp_pips, ref_outcome, ref_pnl_pips,
        time_to_mfe_minutes, candles_evaluated
      FROM monitor_only_outcomes
      WHERE signal_timestamp >= $1
        AND environment = $2
      ORDER BY signal_timestamp DESC
      LIMIT 1000
      `,
      [since, env]
    );

    const rows = (rowsResult.rows ?? []).map((row) => ({
      alert_id: Number(row.alert_id),
      strategy: row.strategy,
      epic: row.epic,
      pair: normalizeSymbol(row.epic) || row.pair,
      direction: row.direction,
      signal_timestamp: row.signal_timestamp,
      status: row.status,
      entry_price: num(row.entry_price),
      mfe_pips: num(row.mfe_pips),
      mae_pips: num(row.mae_pips),
      early_mae_pips: num(row.early_mae_pips),
      pnl_60m_pips: num(row.pnl_60m_pips),
      pnl_240m_pips: num(row.pnl_240m_pips),
      pnl_1440m_pips: num(row.pnl_1440m_pips),
      ref_sl_pips: num(row.ref_sl_pips),
      ref_tp_pips: num(row.ref_tp_pips),
      ref_outcome: row.ref_outcome,
      ref_pnl_pips: num(row.ref_pnl_pips),
      time_to_mfe_minutes: num(row.time_to_mfe_minutes),
      candles_evaluated: num(row.candles_evaluated)
    }));

    return NextResponse.json({ summary, rows });
  } catch (error) {
    console.error("Failed to load monitor outcomes", error);
    return NextResponse.json({ error: "Failed to load monitor outcomes" }, { status: 500 });
  }
}
