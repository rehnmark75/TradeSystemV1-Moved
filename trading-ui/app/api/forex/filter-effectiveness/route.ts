import { NextResponse } from "next/server";
import { forexPool } from "../../../../lib/forexDb";

export const dynamic = "force-dynamic";

const DEFAULT_DAYS = 30;

function parseDays(value: string | null): number {
  if (!value) return DEFAULT_DAYS;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_DAYS;
  return parsed;
}

type FilterMetric = {
  filter_name: string;
  filter_value: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
};

type FilterGroup = {
  name: string;
  description: string;
  metrics: FilterMetric[];
  recommendation: string;
  is_predictive: boolean;
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const days = parseDays(searchParams.get("days"));
  const since = new Date();
  since.setDate(since.getDate() - days);

  try {
    // 1. Market Regime effectiveness (from performance_metrics)
    const marketRegimeResult = await forexPool.query(
      `SELECT
        'market_regime' as filter_name,
        COALESCE(a.performance_metrics->>'market_regime', 'Unknown') as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY COALESCE(a.performance_metrics->>'market_regime', 'Unknown')
      ORDER BY trades DESC`,
      [since]
    );

    // 2. Volatility State effectiveness (from performance_metrics)
    const volatilityResult = await forexPool.query(
      `SELECT
        'volatility_state' as filter_name,
        COALESCE(a.performance_metrics->>'volatility_state', 'Unknown') as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY COALESCE(a.performance_metrics->>'volatility_state', 'Unknown')
      ORDER BY trades DESC`,
      [since]
    );

    // 3. Market Structure Bias effectiveness
    const structureBiasResult = await forexPool.query(
      `SELECT
        'structure_bias' as filter_name,
        COALESCE(a.market_structure_analysis->>'current_bias', 'Unknown') as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY COALESCE(a.market_structure_analysis->>'current_bias', 'Unknown')
      ORDER BY trades DESC`,
      [since]
    );

    // 4. Structure Alignment (direction vs bias) effectiveness
    const alignmentResult = await forexPool.query(
      `SELECT
        'direction_alignment' as filter_name,
        CASE
          WHEN a.market_structure_analysis->>'current_bias' = 'RANGING' THEN 'RANGING'
          WHEN (t.direction = 'BUY' AND a.market_structure_analysis->>'current_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.market_structure_analysis->>'current_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.market_structure_analysis->>'current_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'COUNTER'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN a.market_structure_analysis->>'current_bias' = 'RANGING' THEN 'RANGING'
          WHEN (t.direction = 'BUY' AND a.market_structure_analysis->>'current_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.market_structure_analysis->>'current_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.market_structure_analysis->>'current_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'COUNTER'
        END
      ORDER BY win_rate DESC NULLS LAST`,
      [since]
    );

    // 5. Order Flow Bias alignment (from order_flow_analysis)
    const orderFlowResult = await forexPool.query(
      `SELECT
        'order_flow_alignment' as filter_name,
        CASE
          WHEN (t.direction = 'BUY' AND a.order_flow_analysis->>'order_flow_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.order_flow_analysis->>'order_flow_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.order_flow_analysis->>'order_flow_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'CONFLICTING'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN (t.direction = 'BUY' AND a.order_flow_analysis->>'order_flow_bias' = 'BULLISH')
            OR (t.direction = 'SELL' AND a.order_flow_analysis->>'order_flow_bias' = 'BEARISH')
          THEN 'ALIGNED'
          WHEN a.order_flow_analysis->>'order_flow_bias' = 'NEUTRAL' THEN 'NEUTRAL'
          ELSE 'CONFLICTING'
        END
      ORDER BY trades DESC`,
      [since]
    );

    // 6. Entry Quality Score buckets (from performance_metrics)
    const entryQualityResult = await forexPool.query(
      `SELECT
        'entry_quality' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.3 THEN 'Low (0.3-0.5)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float < 0.3 THEN 'Very Low (<0.3)'
          ELSE 'Unknown'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.3 THEN 'Low (0.3-0.5)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float < 0.3 THEN 'Very Low (<0.3)'
          ELSE 'Unknown'
        END
      ORDER BY
        CASE CASE
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float >= 0.3 THEN 'Low (0.3-0.5)'
          WHEN (a.performance_metrics->>'entry_quality_score')::float < 0.3 THEN 'Very Low (<0.3)'
          ELSE 'Unknown'
        END
          WHEN 'High (>=0.7)' THEN 1
          WHEN 'Medium (0.5-0.7)' THEN 2
          WHEN 'Low (0.3-0.5)' THEN 3
          WHEN 'Very Low (<0.3)' THEN 4
          ELSE 5
        END`,
      [since]
    );

    // 7. Efficiency Ratio buckets (from performance_metrics)
    const efficiencyResult = await forexPool.query(
      `SELECT
        'efficiency_ratio' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.5 THEN 'High (>=0.5)'
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.3 THEN 'Medium (0.3-0.5)'
          ELSE 'Low (<0.3)'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
        AND a.performance_metrics->>'efficiency_ratio' IS NOT NULL
      GROUP BY CASE
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.5 THEN 'High (>=0.5)'
          WHEN (a.performance_metrics->>'efficiency_ratio')::float >= 0.3 THEN 'Medium (0.3-0.5)'
          ELSE 'Low (<0.3)'
        END
      ORDER BY trades DESC`,
      [since]
    );

    // 8. MTF Alignment (from performance_metrics)
    const mtfResult = await forexPool.query(
      `SELECT
        'mtf_alignment' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'all_timeframes_aligned')::boolean = true THEN 'All TFs Aligned'
          ELSE 'Not Aligned'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
      GROUP BY CASE
          WHEN (a.performance_metrics->>'all_timeframes_aligned')::boolean = true THEN 'All TFs Aligned'
          ELSE 'Not Aligned'
        END
      ORDER BY trades DESC`,
      [since]
    );

    // 9. MTF Confluence Score buckets (from performance_metrics)
    const confluenceResult = await forexPool.query(
      `SELECT
        'mtf_confluence' as filter_name,
        CASE
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          ELSE 'Low (<0.5)'
        END as filter_value,
        COUNT(*)::int as trades,
        SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN t.profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(t.profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(t.profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log t
      LEFT JOIN alert_history a ON t.symbol = a.epic
        AND a.alert_timestamp BETWEEN t.timestamp - interval '2 minutes' AND t.timestamp + interval '2 minutes'
      WHERE t.timestamp >= $1 AND LOWER(t.status) = 'closed'
        AND a.performance_metrics->>'mtf_confluence_score' IS NOT NULL
      GROUP BY CASE
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.7 THEN 'High (>=0.7)'
          WHEN (a.performance_metrics->>'mtf_confluence_score')::float >= 0.5 THEN 'Medium (0.5-0.7)'
          ELSE 'Low (<0.5)'
        END
      ORDER BY trades DESC`,
      [since]
    );

    // 10. Overall baseline stats
    const baselineResult = await forexPool.query(
      `SELECT
        COUNT(*)::int as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::int as wins,
        SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END)::int as losses,
        ROUND(100.0 * SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) as win_rate,
        ROUND(SUM(profit_loss)::numeric, 2) as total_pnl,
        ROUND(AVG(profit_loss)::numeric, 2) as avg_pnl
      FROM trade_log
      WHERE timestamp >= $1 AND LOWER(status) = 'closed'`,
      [since]
    );

    const baseline = baselineResult.rows[0] || {
      total_trades: 0,
      wins: 0,
      losses: 0,
      win_rate: 0,
      total_pnl: 0,
      avg_pnl: 0
    };

    // Helper to determine if filter is predictive
    const analyzeFilter = (
      name: string,
      description: string,
      metrics: FilterMetric[]
    ): FilterGroup => {
      const validMetrics = metrics.filter(m => m.trades >= 5);
      if (validMetrics.length < 2) {
        return {
          name,
          description,
          metrics,
          recommendation: "Insufficient data for analysis",
          is_predictive: false
        };
      }

      const winRates = validMetrics.map(m => m.win_rate || 0);
      const maxWinRate = Math.max(...winRates);
      const minWinRate = Math.min(...winRates);
      const spread = maxWinRate - minWinRate;

      const bestBucket = validMetrics.find(m => m.win_rate === maxWinRate);
      const worstBucket = validMetrics.find(m => m.win_rate === minWinRate);

      const isPredictive = spread >= 10; // At least 10% spread to be useful

      let recommendation = "";
      if (!isPredictive) {
        recommendation = "Not predictive - no significant performance difference between groups";
      } else if (bestBucket && worstBucket) {
        recommendation = `Consider blocking "${worstBucket.filter_value}" (${worstBucket.win_rate}% WR) and favoring "${bestBucket.filter_value}" (${bestBucket.win_rate}% WR)`;
      }

      return {
        name,
        description,
        metrics,
        recommendation,
        is_predictive: isPredictive
      };
    };

    const filterGroups: FilterGroup[] = [
      analyzeFilter(
        "Entry Quality Score",
        "Signal entry quality based on Fib zone and candle momentum",
        entryQualityResult.rows
      ),
      analyzeFilter(
        "Direction vs Structure Alignment",
        "Whether trade direction matches market structure bias",
        alignmentResult.rows
      ),
      analyzeFilter(
        "Market Structure Bias",
        "Current market structure from Smart Money analysis",
        structureBiasResult.rows
      ),
      analyzeFilter(
        "Order Flow Alignment",
        "Trade direction vs order flow bias",
        orderFlowResult.rows
      ),
      analyzeFilter(
        "Market Regime",
        "Detected market regime at signal time",
        marketRegimeResult.rows
      ),
      analyzeFilter(
        "Volatility State",
        "Market volatility state at signal time",
        volatilityResult.rows
      ),
      analyzeFilter(
        "MTF Alignment",
        "Multi-timeframe directional alignment",
        mtfResult.rows
      ),
      analyzeFilter(
        "Efficiency Ratio",
        "Price movement efficiency (trend strength)",
        efficiencyResult.rows
      ),
      analyzeFilter(
        "MTF Confluence Score",
        "Overall multi-timeframe confluence score",
        confluenceResult.rows
      )
    ];

    return NextResponse.json({
      baseline,
      filterGroups,
      days,
      generatedAt: new Date().toISOString()
    });
  } catch (error) {
    console.error("Failed to load filter effectiveness", error);
    return NextResponse.json(
      { error: "Failed to load filter effectiveness analysis" },
      { status: 500 }
    );
  }
}
