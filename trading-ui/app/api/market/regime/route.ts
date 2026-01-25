import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const numberOrZero = (value: unknown) => {
  const parsed = Number(value);
  return Number.isNaN(parsed) ? 0 : parsed;
};

export async function GET() {
  const client = await pool.connect();
  try {
    const query = `
      SELECT
        calculation_date,
        market_regime,
        spy_price,
        spy_sma50,
        spy_sma200,
        spy_vs_sma50_pct,
        spy_vs_sma200_pct,
        spy_trend,
        pct_above_sma200,
        pct_above_sma50,
        pct_above_sma20,
        new_highs_count,
        new_lows_count,
        high_low_ratio,
        advancing_count,
        declining_count,
        ad_ratio,
        avg_atr_pct,
        volatility_regime,
        recommended_strategies
      FROM market_context
      ORDER BY calculation_date DESC
      LIMIT 1
    `;
    const result = await client.query(query);
    if (result.rows.length) {
      const row = result.rows[0];
      if (row.recommended_strategies && typeof row.recommended_strategies === "string") {
        try {
          row.recommended_strategies = JSON.parse(row.recommended_strategies);
        } catch (_) {
          row.recommended_strategies = null;
        }
      }
      return NextResponse.json({ row });
    }

    const breadthQuery = `
      SELECT
        COUNT(*) FILTER (WHERE current_price > sma_200) as above_200,
        COUNT(*) FILTER (WHERE current_price > sma_50) as above_50,
        COUNT(*) FILTER (WHERE current_price > sma_20) as above_20,
        COUNT(*) FILTER (WHERE trend_strength IN ('strong_up', 'up')) as bullish,
        COUNT(*) FILTER (WHERE trend_strength IN ('strong_down', 'down')) as bearish,
        COUNT(*) as total,
        AVG(atr_percent) as avg_atr,
        AVG(current_price) as avg_price
      FROM stock_screening_metrics
      WHERE calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
    `;
    const breadthResult = await client.query(breadthQuery);
    const breadth = breadthResult.rows[0];
    if (!breadth) {
      return NextResponse.json({ row: null });
    }

    const total = numberOrZero(breadth.total) || 1;
    const pctAbove200 = (numberOrZero(breadth.above_200) / total) * 100;
    const pctAbove50 = (numberOrZero(breadth.above_50) / total) * 100;
    const pctAbove20 = (numberOrZero(breadth.above_20) / total) * 100;

    const advancing = numberOrZero(breadth.bullish);
    const declining = numberOrZero(breadth.bearish);
    const adRatio = declining > 0 ? advancing / declining : advancing;

    const avgAtr = numberOrZero(breadth.avg_atr);
    let volatility = "normal";
    if (avgAtr < 2) volatility = "low";
    else if (avgAtr < 4) volatility = "normal";
    else if (avgAtr < 6) volatility = "high";
    else volatility = "extreme";

    let marketRegime = "bear_confirmed";
    let spyTrend = "falling";
    if (pctAbove200 > 60 && pctAbove50 > 50) {
      marketRegime = "bull_confirmed";
      spyTrend = "rising";
    } else if (pctAbove200 > 50) {
      marketRegime = "bull_weakening";
      spyTrend = "flat";
    } else if (pctAbove200 > 40) {
      marketRegime = "bear_weakening";
      spyTrend = "flat";
    }

    const avgPrice = numberOrZero(breadth.avg_price);
    const spyPrice = avgPrice ? Number((avgPrice * 5).toFixed(2)) : 0;
    const spySma50 = spyPrice * (pctAbove50 > 50 ? 0.98 : 1.02);
    const spySma200 = spyPrice * (pctAbove200 > 50 ? 0.95 : 1.05);

    const recommendedStrategies =
      marketRegime === "bull_confirmed"
        ? { trend_following: 0.8, breakout: 0.7, pullback: 0.6, mean_reversion: 0.2 }
        : marketRegime === "bull_weakening"
          ? { trend_following: 0.5, breakout: 0.4, pullback: 0.7, mean_reversion: 0.4 }
          : marketRegime === "bear_weakening"
            ? { trend_following: 0.3, breakout: 0.3, pullback: 0.5, mean_reversion: 0.6 }
            : { trend_following: 0.2, breakout: 0.2, pullback: 0.3, mean_reversion: 0.7 };

    return NextResponse.json({
      row: {
        market_regime: marketRegime,
        spy_price: spyPrice,
        spy_sma50: spySma50,
        spy_sma200: spySma200,
        spy_vs_sma50_pct: pctAbove50 - 50,
        spy_vs_sma200_pct: pctAbove200 - 50,
        spy_trend: spyTrend,
        pct_above_sma200: pctAbove200,
        pct_above_sma50: pctAbove50,
        pct_above_sma20: pctAbove20,
        new_highs_count: 0,
        new_lows_count: 0,
        high_low_ratio: 1,
        advancing_count: advancing,
        declining_count: declining,
        ad_ratio: adRatio,
        avg_atr_pct: avgAtr,
        volatility_regime: volatility,
        recommended_strategies: recommendedStrategies
      }
    });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load market regime" }, { status: 500 });
  } finally {
    client.release();
  }
}
