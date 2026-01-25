import { NextResponse } from "next/server";
import { pool } from "../../../../lib/db";

export const dynamic = "force-dynamic";

const sectorEtfs: Record<string, string> = {
  Technology: "XLK",
  "Health Care": "XLV",
  Financials: "XLF",
  "Consumer Discretionary": "XLY",
  "Communication Services": "XLC",
  Industrials: "XLI",
  "Consumer Staples": "XLP",
  Energy: "XLE",
  Utilities: "XLU",
  "Real Estate": "XLRE",
  Materials: "XLB"
};

const sectorAliases: Record<string, string[]> = {
  Technology: ["Technology"],
  "Health Care": ["Health Care", "Healthcare"],
  Financials: ["Financials", "Financial Services"],
  "Consumer Discretionary": ["Consumer Discretionary", "Consumer Cyclical"],
  "Consumer Staples": ["Consumer Staples", "Consumer Defensive"],
  "Communication Services": ["Communication Services"],
  Industrials: ["Industrials"],
  Energy: ["Energy"],
  Utilities: ["Utilities"],
  "Real Estate": ["Real Estate"],
  Materials: ["Materials", "Basic Materials"]
};

export async function GET() {
  const client = await pool.connect();
  try {
    const query = `
      SELECT
        sector,
        sector_etf,
        sector_return_1d,
        sector_return_5d,
        sector_return_20d,
        rs_vs_spy,
        rs_percentile,
        rs_trend,
        stocks_in_sector,
        pct_above_sma50,
        pct_bullish_trend,
        top_stocks,
        sector_stage
      FROM sector_analysis
      WHERE calculation_date = (SELECT MAX(calculation_date) FROM sector_analysis)
      ORDER BY rs_vs_spy DESC
    `;
    const result = await client.query(query);
    if (result.rows.length) {
      const rows = result.rows.map((row) => {
        if (row.top_stocks && typeof row.top_stocks === "string") {
          try {
            row.top_stocks = JSON.parse(row.top_stocks);
          } catch (_) {
            row.top_stocks = [];
          }
        }
        if (!Array.isArray(row.top_stocks)) {
          row.top_stocks = [];
        }
        return row;
      });

      const sectorsNeedingTop = rows
        .filter((row) => !row.top_stocks || row.top_stocks.length === 0)
        .map((row) => row.sector)
        .filter(Boolean);

      if (sectorsNeedingTop.length) {
        const aliasToSector: Record<string, string> = {};
        const aliasList = sectorsNeedingTop.flatMap((sector) => {
          const aliases = sectorAliases[sector] || [sector];
          aliases.forEach((alias) => {
            aliasToSector[alias] = sector;
          });
          return aliases;
        });
        const topStocksQuery = `
          SELECT
            i.sector,
            m.ticker,
            m.rs_percentile,
            m.rs_trend,
            m.current_price,
            ROW_NUMBER() OVER (PARTITION BY i.sector ORDER BY m.rs_percentile DESC) as rn
          FROM stock_screening_metrics m
          JOIN stock_instruments i ON m.ticker = i.ticker
          WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            AND m.rs_percentile IS NOT NULL
            AND i.sector = ANY($1::text[])
        `;
        const topStocksResult = await client.query(topStocksQuery, [aliasList]);
        const topBySector: Record<string, Array<{ ticker: string; rs_percentile: number; rs_trend: string | null; price: number | null }>> = {};
        topStocksResult.rows.forEach((row) => {
          if (row.rn > 5) return;
          const sectorKey = aliasToSector[row.sector] || row.sector;
          if (!topBySector[sectorKey]) {
            topBySector[sectorKey] = [];
          }
          topBySector[sectorKey].push({
            ticker: row.ticker,
            rs_percentile: row.rs_percentile,
            rs_trend: row.rs_trend,
            price: row.current_price
          });
        });

        rows.forEach((row) => {
          if (!row.top_stocks || row.top_stocks.length === 0) {
            row.top_stocks = topBySector[row.sector] || [];
          }
        });
      }

      return NextResponse.json({ rows });
    }

    const fallbackQuery = `
      SELECT
        i.sector,
        COUNT(*) as stocks_in_sector,
        AVG(m.rs_vs_spy) as avg_rs,
        AVG(m.rs_percentile) as avg_rs_percentile,
        COUNT(*) FILTER (WHERE m.current_price > m.sma_50) * 100.0 / NULLIF(COUNT(*), 0) as pct_above_sma50,
        COUNT(*) FILTER (WHERE m.trend_strength IN ('strong_up', 'up')) * 100.0 / NULLIF(COUNT(*), 0) as pct_bullish_trend,
        AVG(m.price_change_1d) as sector_return_1d,
        AVG(m.price_change_5d) as sector_return_5d,
        AVG(m.price_change_20d) as sector_return_20d
      FROM stock_instruments i
      JOIN stock_screening_metrics m ON i.ticker = m.ticker
      WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        AND i.sector IS NOT NULL
        AND i.sector <> ''
      GROUP BY i.sector
      ORDER BY avg_rs DESC NULLS LAST
    `;
    const fallbackResult = await client.query(fallbackQuery);
    if (!fallbackResult.rows.length) {
      return NextResponse.json({ rows: [] });
    }

    const topStocksQuery = `
      SELECT
        i.sector,
        m.ticker,
        m.rs_percentile,
        m.rs_trend,
        m.current_price,
        ROW_NUMBER() OVER (PARTITION BY i.sector ORDER BY m.rs_percentile DESC) as rn
      FROM stock_screening_metrics m
      JOIN stock_instruments i ON m.ticker = i.ticker
      WHERE m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
        AND m.rs_percentile IS NOT NULL
    `;
    const topStocksResult = await client.query(topStocksQuery);
    const topBySector: Record<string, Array<{ ticker: string; rs_percentile: number; rs_trend: string | null; price: number | null }>> = {};
    topStocksResult.rows.forEach((row) => {
      if (row.rn > 5) return;
      if (!topBySector[row.sector]) {
        topBySector[row.sector] = [];
      }
      topBySector[row.sector].push({
        ticker: row.ticker,
        rs_percentile: row.rs_percentile,
        rs_trend: row.rs_trend,
        price: row.current_price
      });
    });

    const rows = fallbackResult.rows.map((row) => {
      const avgRs = row.avg_rs;
      const pctBullish = row.pct_bullish_trend || 0;
      let stage = "lagging";
      if (avgRs && avgRs > 1.0) {
        stage = pctBullish > 50 ? "leading" : "weakening";
      } else {
        stage = pctBullish > 40 ? "improving" : "lagging";
      }

      return {
        sector: row.sector,
        sector_etf: sectorEtfs[row.sector] || "",
        rs_vs_spy: row.avg_rs,
        rs_percentile: row.avg_rs_percentile ? Math.round(row.avg_rs_percentile) : null,
        rs_trend: "stable",
        sector_return_1d: row.sector_return_1d,
        sector_return_5d: row.sector_return_5d,
        sector_return_20d: row.sector_return_20d,
        stocks_in_sector: row.stocks_in_sector,
        pct_above_sma50: row.pct_above_sma50,
        pct_bullish_trend: row.pct_bullish_trend,
        sector_stage: stage,
        top_stocks: topBySector[row.sector] || []
      };
    });

    return NextResponse.json({ rows });
  } catch (error) {
    return NextResponse.json({ error: "Failed to load sector analysis" }, { status: 500 });
  } finally {
    client.release();
  }
}
