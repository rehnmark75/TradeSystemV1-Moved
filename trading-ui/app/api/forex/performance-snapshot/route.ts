import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 1;

function parseDate(value: string | null) {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? null : parsed;
}

function parseDays(value: string | null) {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_DAYS;
  if (parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const startParam = parseDate(searchParams.get("start"));
  const endParam = parseDate(searchParams.get("end"));
  const days = parseDays(searchParams.get("days"));

  const end = endParam ?? new Date();
  const start =
    startParam ??
    new Date(new Date(end).setDate(end.getDate() - days));

  try {
    const summaryResult = await forexPool.query(
      `
      SELECT
        COUNT(*) as total_scans,
        COUNT(DISTINCT scan_cycle_id) as scan_cycles,
        COUNT(DISTINCT epic) as unique_epics,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals_generated,
        SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
        AVG(raw_confidence) as avg_raw_confidence,
        AVG(final_confidence) as avg_final_confidence,
        AVG(CASE WHEN signal_generated THEN final_confidence END) as avg_signal_confidence,
        COUNT(DISTINCT rejection_reason) as rejection_types,
        SUM(CASE WHEN rejection_reason = 'confidence' THEN 1 ELSE 0 END) as confidence_rejections,
        SUM(CASE WHEN rejection_reason = 'dedup' THEN 1 ELSE 0 END) as dedup_rejections,
        MIN(scan_timestamp) as first_scan,
        MAX(scan_timestamp) as last_scan
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      `,
      [start, end]
    );

    const summaryRow = summaryResult.rows[0] ?? {};
    const totalScans = Number(summaryRow.total_scans ?? 0);
    const signalsGenerated = Number(summaryRow.signals_generated ?? 0);
    const summary = {
      total_scans: totalScans,
      scan_cycles: Number(summaryRow.scan_cycles ?? 0),
      unique_epics: Number(summaryRow.unique_epics ?? 0),
      signals_generated: signalsGenerated,
      buy_signals: Number(summaryRow.buy_signals ?? 0),
      sell_signals: Number(summaryRow.sell_signals ?? 0),
      avg_raw_confidence: Number(summaryRow.avg_raw_confidence ?? 0),
      avg_final_confidence: Number(summaryRow.avg_final_confidence ?? 0),
      avg_signal_confidence: Number(summaryRow.avg_signal_confidence ?? 0),
      rejection_types: Number(summaryRow.rejection_types ?? 0),
      confidence_rejections: Number(summaryRow.confidence_rejections ?? 0),
      dedup_rejections: Number(summaryRow.dedup_rejections ?? 0),
      first_scan: summaryRow.first_scan,
      last_scan: summaryRow.last_scan,
      signal_rate: totalScans ? signalsGenerated / totalScans : 0
    };

    const timelineResult = await forexPool.query(
      `
      SELECT
        DATE_TRUNC('hour', scan_timestamp) as hour,
        COUNT(*) as total_scans,
        COUNT(DISTINCT scan_cycle_id) as scan_cycles,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        AVG(raw_confidence) as avg_confidence,
        AVG(atr_pips) as avg_atr
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      GROUP BY DATE_TRUNC('hour', scan_timestamp)
      ORDER BY hour
      `,
      [start, end]
    );

    const regimeResult = await forexPool.query(
      `
      SELECT
        market_regime,
        COUNT(*) as count,
        AVG(regime_confidence) as avg_confidence,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        AVG(CASE WHEN signal_generated THEN final_confidence END) as avg_signal_confidence
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
        AND market_regime IS NOT NULL
      GROUP BY market_regime
      ORDER BY count DESC
      `,
      [start, end]
    );

    const sessionResult = await forexPool.query(
      `
      SELECT
        session,
        session_volatility,
        COUNT(*) as count,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        AVG(raw_confidence) as avg_confidence,
        AVG(atr_pips) as avg_atr_pips,
        AVG(spread_pips) as avg_spread
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
        AND session IS NOT NULL
      GROUP BY session, session_volatility
      ORDER BY count DESC
      `,
      [start, end]
    );

    const rejectionResult = await forexPool.query(
      `
      SELECT
        rejection_reason,
        COUNT(*) as count,
        AVG(raw_confidence) as avg_raw_confidence,
        AVG(final_confidence) as avg_final_confidence,
        AVG(confidence_threshold) as avg_threshold,
        COUNT(DISTINCT epic) as affected_epics
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
        AND rejection_reason IS NOT NULL
      GROUP BY rejection_reason
      ORDER BY count DESC
      `,
      [start, end]
    );

    const epicResult = await forexPool.query(
      `
      SELECT
        epic,
        pair_name,
        COUNT(*) as total_scans,
        SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals,
        SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
        SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
        AVG(raw_confidence) as avg_raw_confidence,
        AVG(final_confidence) as avg_final_confidence,
        AVG(rsi_14) as avg_rsi,
        AVG(adx) as avg_adx,
        AVG(atr_pips) as avg_atr_pips,
        AVG(spread_pips) as avg_spread,
        MODE() WITHIN GROUP (ORDER BY market_regime) as dominant_regime,
        MODE() WITHIN GROUP (ORDER BY volatility_state) as dominant_volatility,
        SUM(CASE WHEN rejection_reason = 'confidence' THEN 1 ELSE 0 END) as confidence_rejections,
        SUM(CASE WHEN rejection_reason = 'dedup' THEN 1 ELSE 0 END) as dedup_rejections
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      GROUP BY epic, pair_name
      ORDER BY signals DESC, total_scans DESC
      `,
      [start, end]
    );

    const indicatorResult = await forexPool.query(
      `
      SELECT
        signal_generated,
        AVG(rsi_14) as avg_rsi,
        AVG(adx) as avg_adx,
        AVG(efficiency_ratio) as avg_er,
        AVG(atr_pips) as avg_atr,
        AVG(bb_width_percentile) as avg_bb_percentile,
        AVG(smart_money_score) as avg_smc_score,
        AVG(mtf_confluence_score) as avg_mtf_score,
        AVG(entry_quality_score) as avg_entry_quality,
        COUNT(*) as count
      FROM scan_performance_snapshot
      WHERE scan_timestamp >= $1
        AND scan_timestamp <= $2
      GROUP BY signal_generated
      `,
      [start, end]
    );

    const indicators = { signals: null, non_signals: null } as Record<
      string,
      Record<string, number> | null
    >;

    for (const row of indicatorResult.rows ?? []) {
      const key = row.signal_generated ? "signals" : "non_signals";
      indicators[key] = {
        avg_rsi: Number(row.avg_rsi ?? 0),
        avg_adx: Number(row.avg_adx ?? 0),
        avg_er: Number(row.avg_er ?? 0),
        avg_atr: Number(row.avg_atr ?? 0),
        avg_bb_percentile: Number(row.avg_bb_percentile ?? 0),
        avg_smc_score: Number(row.avg_smc_score ?? 0),
        avg_mtf_score: Number(row.avg_mtf_score ?? 0),
        avg_entry_quality: Number(row.avg_entry_quality ?? 0),
        count: Number(row.count ?? 0)
      };
    }

    return NextResponse.json({
      range: { start, end },
      summary,
      timeline: timelineResult.rows ?? [],
      regimes: regimeResult.rows ?? [],
      sessions: sessionResult.rows ?? [],
      rejections: rejectionResult.rows ?? [],
      epics: epicResult.rows ?? [],
      indicators
    });
  } catch (error) {
    console.error("Failed to load performance snapshot", error);
    return NextResponse.json(
      { error: "Failed to load performance snapshot" },
      { status: 500 }
    );
  }
}
