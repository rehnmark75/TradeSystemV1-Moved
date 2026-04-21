import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

type Row = Record<string, any>;

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

async function getMihColumns() {
  const result = await forexPool.query(
    `SELECT column_name FROM information_schema.columns WHERE table_name='market_intelligence_history'`
  );
  return (result.rows ?? []).map((r) => r.column_name as string);
}

function num(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function parsePairAnalyses(raw: unknown): Record<string, any> {
  if (!raw) return {};
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw);
    } catch {
      return {};
    }
  }
  if (typeof raw === "object") return raw as Record<string, any>;
  return {};
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const env = searchParams.get("env") || "demo";
  const source = (searchParams.get("source") || "comprehensive").toLowerCase();
  const startParam = parseDate(searchParams.get("start"));
  const endParam = parseDate(searchParams.get("end"));
  const { start, end } = resolveRange(startParam, endParam);

  try {
    let comprehensiveData: Row[] = [];
    let signalData: Row[] = [];
    let timeline: Row[] = [];
    let latestScan: Row | null = null;
    let byPair: Row[] = [];
    let effectiveness: Row[] = [];

    if (source === "comprehensive" || source === "both") {
      const cols = await getMihColumns();
      if (cols.length) {
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
        const optional: string[] = [];
        if (cols.includes("individual_epic_regimes")) optional.push("mih.individual_epic_regimes");
        if (cols.includes("pair_analyses")) optional.push("mih.pair_analyses");

        const query = `
          SELECT ${[...baseColumns, ...optional].join(", ")}
          FROM market_intelligence_history mih
          WHERE mih.scan_timestamp >= $1 AND mih.scan_timestamp <= $2
          ORDER BY mih.scan_timestamp DESC
        `;
        const result = await forexPool.query(query, [start, end]);
        comprehensiveData = result.rows ?? [];

        // Latest scan (include pair_analyses).
        if (comprehensiveData.length) {
          latestScan = comprehensiveData[0];

          const analyses = parsePairAnalyses(latestScan.pair_analyses);
          const epics = Object.keys(analyses);

          // Per-pair alert + outcome aggregation over the window.
          let alertRows: Row[] = [];
          if (epics.length) {
            const { rows } = await forexPool.query(
              `
              SELECT
                a.epic,
                COUNT(*)::int AS alerts,
                COUNT(t.id) FILTER (WHERE t.status = 'closed')::int AS closed,
                COUNT(*) FILTER (WHERE t.profit_loss > 0)::int AS wins,
                COUNT(*) FILTER (WHERE t.profit_loss < 0)::int AS losses,
                COALESCE(SUM(CASE WHEN t.profit_loss > 0 THEN t.profit_loss ELSE 0 END), 0) AS gross_profit,
                COALESCE(SUM(CASE WHEN t.profit_loss < 0 THEN -t.profit_loss ELSE 0 END), 0) AS gross_loss,
                COALESCE(SUM(t.pips_gained), 0) AS pips,
                MAX(a.alert_timestamp) AS last_alert
              FROM alert_history a
              LEFT JOIN trade_log t ON t.alert_id = a.id
              WHERE a.alert_timestamp >= $1 AND a.alert_timestamp <= $2
                AND a.environment = $3
                AND a.epic = ANY($4::text[])
              GROUP BY a.epic
              `,
              [start, end, env, epics]
            );
            alertRows = rows;
          }
          const alertMap = new Map<string, Row>();
          alertRows.forEach((r) => alertMap.set(String(r.epic), r));

          byPair = epics
            .map((epic) => {
              const info = analyses[epic] ?? {};
              const rs = info.regime_scores ?? {};
              const enhanced = info?.enhanced_regime ?? {};
              const legacy = enhanced?.legacy ?? {};
              const a = alertMap.get(epic);
              const gp = num(a?.gross_profit) ?? 0;
              const gl = num(a?.gross_loss) ?? 0;
              const pf = gl > 0 ? gp / gl : gp > 0 ? Infinity : null;
              return {
                epic,
                price: num(info.current_price),
                regime: String(legacy.dominant_regime || info.dominant_regime || "unknown"),
                confidence: num(legacy.confidence) ?? num(info.confidence),
                combined_regime: enhanced?.enhanced?.combined_regime ?? null,
                scores: {
                  trending: num(rs.trending) ?? 0,
                  ranging: num(rs.ranging) ?? 0,
                  breakout: num(rs.breakout) ?? 0,
                  reversal: num(rs.reversal) ?? 0,
                  high_volatility: num(rs.high_volatility) ?? 0,
                  low_volatility: num(rs.low_volatility) ?? 0
                },
                alerts: num(a?.alerts) ?? 0,
                closed: num(a?.closed) ?? 0,
                wins: num(a?.wins) ?? 0,
                losses: num(a?.losses) ?? 0,
                pips: num(a?.pips) ?? 0,
                pf: pf === Infinity ? null : pf,
                pf_infinite: pf === Infinity,
                last_alert: a?.last_alert ?? null
              };
            })
            .sort((x, y) => y.alerts - x.alerts);
        }

        // Lightweight timeline (last 500 scans within range).
        const tl = await forexPool.query(
          `
          SELECT scan_timestamp, dominant_regime AS regime, regime_confidence AS confidence,
                 current_session AS session, session_volatility AS volatility
          FROM market_intelligence_history
          WHERE scan_timestamp >= $1 AND scan_timestamp <= $2
          ORDER BY scan_timestamp ASC
          `,
          [start, end]
        );
        timeline = tl.rows ?? [];
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
          (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'dominant_regime') as regime,
          (a.strategy_metadata::json->'market_intelligence'->'regime_analysis'->>'confidence')::float as regime_confidence,
          (a.strategy_metadata::json->'market_intelligence'->'session_analysis'->>'current_session') as session,
          (a.strategy_metadata::json->'market_intelligence'->>'volatility_level') as volatility_level,
          (a.strategy_metadata::json->'market_intelligence'->>'intelligence_source') as intelligence_source
        FROM alert_history a
        WHERE a.alert_timestamp >= $1 AND a.alert_timestamp <= $2
          AND a.strategy_metadata IS NOT NULL
          AND (a.strategy_metadata::json->'market_intelligence') IS NOT NULL
          AND a.environment = $3
        ORDER BY a.alert_timestamp DESC
        `,
        [start, end, env]
      );
      signalData = result.rows ?? [];
    }

    // Strategy effectiveness (joined to trade_log).
    {
      const { rows } = await forexPool.query(
        `
        SELECT
          a.strategy,
          COALESCE(NULLIF(a.market_regime, ''), 'unknown') AS regime,
          COALESCE(NULLIF(a.market_session, ''), 'unknown') AS session,
          COUNT(*)::int AS alerts,
          COUNT(t.id) FILTER (WHERE t.status = 'closed')::int AS closed,
          COUNT(*) FILTER (WHERE t.profit_loss > 0)::int AS wins,
          COUNT(*) FILTER (WHERE t.profit_loss < 0)::int AS losses,
          COALESCE(SUM(CASE WHEN t.profit_loss > 0 THEN t.profit_loss ELSE 0 END), 0) AS gross_profit,
          COALESCE(SUM(CASE WHEN t.profit_loss < 0 THEN -t.profit_loss ELSE 0 END), 0) AS gross_loss,
          COALESCE(SUM(t.pips_gained), 0) AS pips
        FROM alert_history a
        LEFT JOIN trade_log t ON t.alert_id = a.id
        WHERE a.alert_timestamp >= $1 AND a.alert_timestamp <= $2
          AND a.environment = $3
        GROUP BY a.strategy, regime, session
        HAVING COUNT(*) > 0
        ORDER BY alerts DESC
        `,
        [start, end, env]
      );
      effectiveness = rows.map((r) => {
        const gp = Number(r.gross_profit);
        const gl = Number(r.gross_loss);
        const closed = Number(r.closed);
        const wins = Number(r.wins);
        const pf = gl > 0 ? gp / gl : gp > 0 ? null : null;
        return {
          strategy: r.strategy,
          regime: r.regime,
          session: r.session,
          alerts: Number(r.alerts),
          closed,
          wins,
          losses: Number(r.losses),
          win_rate: closed > 0 ? wins / closed : null,
          pf,
          pips: Number(r.pips)
        };
      });
    }

    const primaryData =
      source === "signal" ? signalData : comprehensiveData.length ? comprehensiveData : signalData;

    const summary = {
      total: primaryData.length,
      avg_epics: primaryData.reduce((acc, row) => {
        const v = Number(row.epic_count ?? 0);
        return acc + (Number.isFinite(v) ? v : 0);
      }, 0),
      unique_regimes: new Set(primaryData.map((r) => r.regime).filter(Boolean)).size,
      avg_confidence: primaryData.length
        ? primaryData.reduce((acc, row) => {
            const v = Number(row.regime_confidence ?? 0);
            return acc + (Number.isFinite(v) ? v : 0);
          }, 0) / primaryData.length
        : 0
    };
    summary.avg_epics = primaryData.length ? summary.avg_epics / primaryData.length : 0;

    const regimeCounts: Record<string, number> = {};
    const sessionCounts: Record<string, number> = {};
    const volatilityCounts: Record<string, number> = {};
    const sourceCounts: Record<string, number> = {};
    // Session × regime matrix: session -> regime -> {count, conf_sum}
    const sxr: Record<string, Record<string, { count: number; sum: number }>> = {};

    for (const row of primaryData) {
      if (row.regime) regimeCounts[String(row.regime)] = (regimeCounts[String(row.regime)] || 0) + 1;
      if (row.session) sessionCounts[String(row.session)] = (sessionCounts[String(row.session)] || 0) + 1;
      if (row.volatility_level)
        volatilityCounts[String(row.volatility_level)] = (volatilityCounts[String(row.volatility_level)] || 0) + 1;
      if (row.intelligence_source)
        sourceCounts[String(row.intelligence_source)] = (sourceCounts[String(row.intelligence_source)] || 0) + 1;
      if (row.session && row.regime) {
        const s = String(row.session);
        const r = String(row.regime);
        const c = Number(row.regime_confidence ?? 0);
        if (!sxr[s]) sxr[s] = {};
        if (!sxr[s][r]) sxr[s][r] = { count: 0, sum: 0 };
        sxr[s][r].count += 1;
        sxr[s][r].sum += Number.isFinite(c) ? c : 0;
      }
    }
    const session_regime_matrix = Object.entries(sxr).map(([session, regimes]) => ({
      session,
      cells: Object.entries(regimes).map(([regime, v]) => ({
        regime,
        count: v.count,
        avg_confidence: v.count ? v.sum / v.count : 0
      }))
    }));

    // "Now" payload from latest scan.
    const now = latestScan
      ? {
          scan_timestamp: latestScan.scan_timestamp,
          regime: latestScan.regime,
          confidence: num(latestScan.regime_confidence),
          session: latestScan.session,
          volatility_level: latestScan.volatility_level,
          market_bias: latestScan.market_bias,
          risk_sentiment: latestScan.risk_sentiment,
          recommended_strategy: latestScan.recommended_strategy,
          avg_trend_strength: num(latestScan.average_trend_strength),
          avg_volatility: num(latestScan.average_volatility),
          scores: {
            trending: num(latestScan.regime_trending_score) ?? 0,
            ranging: num(latestScan.regime_ranging_score) ?? 0,
            breakout: num(latestScan.regime_breakout_score) ?? 0,
            reversal: num(latestScan.regime_reversal_score) ?? 0,
            high_volatility: num(latestScan.regime_high_vol_score) ?? 0,
            low_volatility: num(latestScan.regime_low_vol_score) ?? 0
          }
        }
      : null;

    return NextResponse.json({
      range: { start, end },
      source,
      summary,
      regimes: regimeCounts,
      sessions: sessionCounts,
      volatility: volatilityCounts,
      intelligence_sources: sourceCounts,
      session_regime_matrix,
      now,
      timeline,
      by_pair: byPair,
      effectiveness,
      comprehensive: comprehensiveData.map((r) => {
        // Strip pair_analyses to keep payload small in the big list.
        const { pair_analyses, individual_epic_regimes, ...rest } = r;
        return rest;
      }),
      signals: signalData
    });
  } catch (error) {
    console.error("Failed to load market intelligence", error);
    return NextResponse.json({ error: "Failed to load market intelligence" }, { status: 500 });
  }
}
