import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

function parseDate(value: string | null) {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? null : parsed;
}

function resolveRange(startParam: Date | null, endParam: Date | null) {
  const end = endParam ?? new Date();
  const start = startParam ?? new Date(new Date(end).setDate(end.getDate() - 7));
  return { start, end };
}

async function getMarketIntelligenceColumns() {
  const result = await forexPool.query(
    `
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'market_intelligence_history'
    `
  );
  return (result.rows ?? []).map((row) => row.column_name);
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const source = (searchParams.get("source") || "comprehensive").toLowerCase();
  const startParam = parseDate(searchParams.get("start"));
  const endParam = parseDate(searchParams.get("end"));
  const { start, end } = resolveRange(startParam, endParam);

  try {
    let comprehensiveData: Record<string, unknown>[] = [];
    let signalData: Record<string, unknown>[] = [];

    if (source === "comprehensive" || source === "both") {
      const columns = await getMarketIntelligenceColumns();
      if (columns.length) {
        const baseColumns = [
          "mih.id",
          "mih.scan_timestamp",
          "mih.scan_cycle_id",
          "mih.epic_list",
          "mih.epic_count",
          "mih.dominant_regime as regime",
          "mih.regime_confidence",
          "mih.current_session as session",
          "mih.session_volatility as volatility_level",
          "mih.market_bias",
          "mih.average_trend_strength",
          "mih.average_volatility",
          "mih.risk_sentiment",
          "mih.recommended_strategy",
          "mih.confidence_threshold",
          "mih.intelligence_source",
          "mih.regime_trending_score",
          "mih.regime_ranging_score",
          "mih.regime_breakout_score",
          "mih.regime_reversal_score",
          "mih.regime_high_vol_score",
          "mih.regime_low_vol_score"
        ];

        const optionalColumns: string[] = [];
        if (columns.includes("individual_epic_regimes")) {
          optionalColumns.push("mih.individual_epic_regimes");
        }
        if (columns.includes("pair_analyses")) {
          optionalColumns.push("mih.pair_analyses");
        }

        const query = `
          SELECT
            ${[...baseColumns, ...optionalColumns].join(", ")}
          FROM market_intelligence_history mih
          WHERE mih.scan_timestamp >= $1
            AND mih.scan_timestamp <= $2
          ORDER BY mih.scan_timestamp DESC
        `;

        const result = await forexPool.query(query, [start, end]);
        comprehensiveData = result.rows ?? [];
      }
    }

    if (source === "signal" || source === "both") {
      const result = await forexPool.query(
        `
        SELECT
          a.id,
          a.alert_timestamp,
          a.epic,
          a.strategy,
          a.signal_type,
          a.confidence_score,
          a.strategy_metadata,
          (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'dominant_regime') as regime,
          (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'confidence')::float as regime_confidence,
          (a.strategy_metadata::json->'market_intelligence'->'session_analysis'->>'current_session') as session,
          (a.strategy_metadata::json->'market_intelligence'->>'volatility_level') as volatility_level,
          (a.strategy_metadata::json->'market_intelligence'->>'intelligence_source') as intelligence_source
        FROM alert_history a
        WHERE a.alert_timestamp >= $1
          AND a.alert_timestamp <= $2
          AND a.strategy_metadata IS NOT NULL
          AND (a.strategy_metadata::json->'market_intelligence') IS NOT NULL
        ORDER BY a.alert_timestamp DESC
        `,
        [start, end]
      );
      signalData = result.rows ?? [];
    }

    const primaryData =
      source === "signal" ? signalData : comprehensiveData.length ? comprehensiveData : signalData;

    const summary = {
      total: primaryData.length,
      avg_epics: primaryData.reduce((acc, row) => {
        const value = Number(row.epic_count ?? 0);
        return acc + (Number.isFinite(value) ? value : 0);
      }, 0),
      unique_regimes: new Set(primaryData.map((row) => row.regime).filter(Boolean))
        .size,
      avg_confidence: primaryData.length
        ? primaryData.reduce((acc, row) => {
            const value = Number(row.regime_confidence ?? 0);
            return acc + (Number.isFinite(value) ? value : 0);
          }, 0) / primaryData.length
        : 0
    };
    summary.avg_epics = primaryData.length ? summary.avg_epics / primaryData.length : 0;

    const regimeCounts: Record<string, number> = {};
    const sessionCounts: Record<string, number> = {};
    const volatilityCounts: Record<string, number> = {};
    const sourceCounts: Record<string, number> = {};

    for (const row of primaryData) {
      if (row.regime) {
        regimeCounts[String(row.regime)] = (regimeCounts[String(row.regime)] || 0) + 1;
      }
      if (row.session) {
        sessionCounts[String(row.session)] =
          (sessionCounts[String(row.session)] || 0) + 1;
      }
      if (row.volatility_level) {
        volatilityCounts[String(row.volatility_level)] =
          (volatilityCounts[String(row.volatility_level)] || 0) + 1;
      }
      if (row.intelligence_source) {
        sourceCounts[String(row.intelligence_source)] =
          (sourceCounts[String(row.intelligence_source)] || 0) + 1;
      }
    }

    return NextResponse.json({
      range: { start, end },
      source,
      summary,
      regimes: regimeCounts,
      sessions: sessionCounts,
      volatility: volatilityCounts,
      intelligence_sources: sourceCounts,
      comprehensive: comprehensiveData,
      signals: signalData
    });
  } catch (error) {
    console.error("Failed to load market intelligence", error);
    return NextResponse.json(
      { error: "Failed to load market intelligence" },
      { status: 500 }
    );
  }
}
