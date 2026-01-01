"""
Sector Analyzer

Calculates sector-level relative strength and rotation metrics:
- Sector RS vs SPY (sector ETF performance vs market)
- Sector stage classification (leading, weakening, lagging, improving)
- Sector RS trend (improving, stable, deteriorating)
- Top stocks per sector by RS percentile

Sector ETF Mapping:
- Technology (XLK)
- Health Care (XLV)
- Financials (XLF)
- Consumer Discretionary (XLY)
- Communication Services (XLC)
- Industrials (XLI)
- Consumer Staples (XLP)
- Energy (XLE)
- Utilities (XLU)
- Real Estate (XLRE)
- Materials (XLB)
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Sector to ETF mapping
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB',
}

# Reverse mapping
ETF_SECTOR_MAP = {v: k for k, v in SECTOR_ETF_MAP.items()}


class SectorAnalyzer:
    """
    Analyzes sector rotation and relative strength.

    Calculates:
    - RS vs SPY for each sector
    - Sector stage (leading/weakening/lagging/improving)
    - Top stocks per sector
    """

    def __init__(self, db_connection):
        """
        Initialize with database connection.

        Args:
            db_connection: PostgreSQL connection or connection string
        """
        self.conn = db_connection

    def calculate_sector_rs(self, lookback_days: int = 20) -> pd.DataFrame:
        """
        Calculate relative strength for each sector vs SPY.

        Args:
            lookback_days: Period for RS calculation (default 20 days)

        Returns:
            DataFrame with sector RS metrics
        """
        try:
            # Get SPY return for the period
            spy_return = self._get_ticker_return('SPY', lookback_days)
            if spy_return is None:
                logger.warning("Could not calculate SPY return")
                return pd.DataFrame()

            sectors = []

            for sector, etf in SECTOR_ETF_MAP.items():
                # Get sector ETF return
                etf_return = self._get_ticker_return(etf, lookback_days)

                if etf_return is None:
                    continue

                # Calculate RS vs SPY
                rs_vs_spy = (1 + etf_return) / (1 + spy_return) if spy_return != -1 else None

                # Get previous RS for trend calculation
                prev_rs = self._get_previous_sector_rs(sector, days_ago=5)

                # Determine RS trend
                rs_trend = self._calculate_rs_trend(rs_vs_spy, prev_rs)

                # Determine sector stage
                sector_stage = self._classify_sector_stage(rs_vs_spy, rs_trend)

                # Get RS percentile (rank among sectors)
                # Will be calculated after all sectors are processed

                # Get stock count in sector
                stock_count = self._get_sector_stock_count(sector)

                # Get top stocks
                top_stocks = self._get_top_sector_stocks(sector, limit=5)

                sectors.append({
                    'sector': sector,
                    'sector_etf': etf,
                    'rs_vs_spy': rs_vs_spy,
                    'sector_return_20d': etf_return * 100 if etf_return else None,
                    'rs_trend': rs_trend,
                    'sector_stage': sector_stage,
                    'stocks_in_sector': stock_count,
                    'top_stocks': top_stocks,
                })

            df = pd.DataFrame(sectors)

            # Calculate RS percentile (rank among sectors)
            if not df.empty and 'rs_vs_spy' in df.columns:
                df['rs_percentile'] = df['rs_vs_spy'].rank(pct=True) * 100
                df['rs_percentile'] = df['rs_percentile'].round(0).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error calculating sector RS: {e}")
            return pd.DataFrame()

    def _get_ticker_return(self, ticker: str, days: int) -> Optional[float]:
        """Get return for a ticker over specified days."""
        try:
            query = """
                SELECT
                    (LAST_VALUE(close) OVER w - FIRST_VALUE(close) OVER w) /
                    NULLIF(FIRST_VALUE(close) OVER w, 0) as return_pct
                FROM stock_candles_daily
                WHERE ticker = %s
                AND date >= CURRENT_DATE - %s
                WINDOW w AS (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
                ORDER BY date DESC
                LIMIT 1
            """

            cursor = self.conn.cursor()
            cursor.execute(query, (ticker, days))
            result = cursor.fetchone()
            cursor.close()

            return float(result[0]) if result and result[0] else None

        except Exception as e:
            logger.error(f"Error getting return for {ticker}: {e}")
            return None

    def _get_previous_sector_rs(self, sector: str, days_ago: int = 5) -> Optional[float]:
        """Get previous RS value for trend calculation."""
        try:
            query = """
                SELECT rs_vs_spy
                FROM sector_analysis
                WHERE sector = %s
                AND calculation_date <= CURRENT_DATE - %s
                ORDER BY calculation_date DESC
                LIMIT 1
            """

            cursor = self.conn.cursor()
            cursor.execute(query, (sector, days_ago))
            result = cursor.fetchone()
            cursor.close()

            return float(result[0]) if result and result[0] else None

        except Exception as e:
            # Table might not exist yet
            return None

    def _calculate_rs_trend(self, current_rs: Optional[float], prev_rs: Optional[float]) -> str:
        """
        Determine RS trend based on current vs previous RS.

        Returns: 'improving', 'stable', or 'deteriorating'
        """
        if current_rs is None or prev_rs is None:
            return 'stable'

        change_pct = ((current_rs - prev_rs) / prev_rs) * 100 if prev_rs != 0 else 0

        if change_pct > 2:
            return 'improving'
        elif change_pct < -2:
            return 'deteriorating'
        else:
            return 'stable'

    def _classify_sector_stage(self, rs_vs_spy: Optional[float], rs_trend: str) -> str:
        """
        Classify sector into rotation stage.

        Stages:
        - leading: RS > 1 and improving
        - weakening: RS > 1 but deteriorating
        - lagging: RS < 1 and deteriorating
        - improving: RS < 1 but improving
        """
        if rs_vs_spy is None:
            return 'unknown'

        if rs_vs_spy > 1.0:
            if rs_trend == 'deteriorating':
                return 'weakening'
            else:
                return 'leading'
        else:
            if rs_trend == 'improving':
                return 'improving'
            else:
                return 'lagging'

    def _get_sector_stock_count(self, sector: str) -> int:
        """Get count of stocks in sector."""
        try:
            query = """
                SELECT COUNT(DISTINCT ticker)
                FROM stock_fundamentals
                WHERE sector = %s
            """

            cursor = self.conn.cursor()
            cursor.execute(query, (sector,))
            result = cursor.fetchone()
            cursor.close()

            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Error getting stock count for {sector}: {e}")
            return 0

    def _get_top_sector_stocks(self, sector: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top stocks in sector by RS percentile."""
        try:
            query = """
                SELECT
                    m.ticker,
                    m.rs_percentile,
                    m.rs_trend,
                    m.current_price
                FROM stock_screening_metrics m
                JOIN stock_fundamentals f ON m.ticker = f.ticker
                WHERE f.sector = %s
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
                AND m.rs_percentile IS NOT NULL
                ORDER BY m.rs_percentile DESC
                LIMIT %s
            """

            cursor = self.conn.cursor()
            cursor.execute(query, (sector, limit))
            results = cursor.fetchall()
            cursor.close()

            return [
                {
                    'ticker': row[0],
                    'rs_percentile': row[1],
                    'rs_trend': row[2],
                    'price': float(row[3]) if row[3] else None
                }
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error getting top stocks for {sector}: {e}")
            return []

    def save_sector_analysis(self, df: pd.DataFrame) -> bool:
        """
        Save sector analysis results to database.

        Args:
            df: DataFrame with sector analysis data

        Returns:
            True if saved successfully
        """
        if df.empty:
            return False

        try:
            cursor = self.conn.cursor()

            for _, row in df.iterrows():
                # Convert top_stocks to JSON
                import json
                top_stocks_json = json.dumps(row.get('top_stocks', []))

                cursor.execute("""
                    INSERT INTO sector_analysis (
                        calculation_date, sector, sector_etf,
                        rs_vs_spy, rs_percentile, rs_trend,
                        sector_return_20d, sector_stage,
                        stocks_in_sector, top_stocks
                    ) VALUES (
                        CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (calculation_date, sector) DO UPDATE SET
                        rs_vs_spy = EXCLUDED.rs_vs_spy,
                        rs_percentile = EXCLUDED.rs_percentile,
                        rs_trend = EXCLUDED.rs_trend,
                        sector_return_20d = EXCLUDED.sector_return_20d,
                        sector_stage = EXCLUDED.sector_stage,
                        stocks_in_sector = EXCLUDED.stocks_in_sector,
                        top_stocks = EXCLUDED.top_stocks,
                        updated_at = NOW()
                """, (
                    row['sector'],
                    row['sector_etf'],
                    row.get('rs_vs_spy'),
                    row.get('rs_percentile'),
                    row.get('rs_trend'),
                    row.get('sector_return_20d'),
                    row.get('sector_stage'),
                    row.get('stocks_in_sector', 0),
                    top_stocks_json
                ))

            self.conn.commit()
            cursor.close()

            logger.info(f"Saved sector analysis for {len(df)} sectors")
            return True

        except Exception as e:
            logger.error(f"Error saving sector analysis: {e}")
            self.conn.rollback()
            return False


def run_sector_analysis(connection_string: str = None) -> pd.DataFrame:
    """
    Run sector analysis and save results.

    Entry point for scheduled execution.
    """
    import psycopg2

    conn_str = connection_string or "postgresql://postgres:postgres@postgres:5432/stocks"

    try:
        conn = psycopg2.connect(conn_str)
        analyzer = SectorAnalyzer(conn)

        # Calculate sector RS
        df = analyzer.calculate_sector_rs(lookback_days=20)

        if not df.empty:
            # Save to database
            analyzer.save_sector_analysis(df)
            logger.info(f"Sector analysis complete: {len(df)} sectors analyzed")

        conn.close()
        return df

    except Exception as e:
        logger.error(f"Sector analysis failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_sector_analysis()
    print(result)
