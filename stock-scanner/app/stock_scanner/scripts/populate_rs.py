#!/usr/bin/env python3
"""
Populate Relative Strength (RS) Metrics

Calculates and populates RS vs SPY data for all stocks in stock_screening_metrics.
Uses yfinance to get SPY 20-day return as the benchmark.

RS Calculation:
- RS vs SPY = Stock 20-day return / SPY 20-day return
- RS > 1.0 means stock outperforms SPY
- RS < 1.0 means stock underperforms SPY

RS Percentile:
- Ranks all stocks by RS from 1-100
- RS 90+ = Elite (top 10%)
- RS 70+ = Strong (top 30%)
- RS 40+ = Average
- RS < 40 = Weak

RS Trend:
- Compares current RS to 5-day previous RS
- improving: RS increased by >5%
- stable: RS changed by <5%
- deteriorating: RS decreased by >5%

Usage:
    docker exec task-worker python -m stock_scanner.scripts.populate_rs
    docker exec task-worker python -m stock_scanner.scripts.populate_rs --date 2025-12-31
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List

import yfinance as yf
import numpy as np

sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class RSPopulator:
    """Populates Relative Strength metrics using yfinance for SPY benchmark."""

    def __init__(self, db: AsyncDatabaseManager):
        self.db = db
        self.spy_return_20d: Optional[float] = None
        self.spy_return_5d: Optional[float] = None

    async def get_spy_returns(self) -> bool:
        """Fetch SPY returns using yfinance."""
        try:
            logger.info("Fetching SPY data from yfinance...")
            spy = yf.Ticker("SPY")
            hist = spy.history(period="1mo")

            if hist.empty or len(hist) < 5:
                logger.error(f"Insufficient SPY data: {len(hist)} days (need at least 5)")
                return False

            # Calculate returns using available data
            # Use all available days up to 20
            closes = hist['Close'].values
            current_price = closes[-1]

            # Use oldest available price for "20-day" return (or whatever we have)
            available_days = len(closes)
            price_20d_ago = closes[0]  # Use oldest available
            price_5d_ago = closes[-5] if len(closes) >= 5 else closes[0]

            self.spy_return_20d = ((current_price / price_20d_ago) - 1) * 100
            self.spy_return_5d = ((current_price / price_5d_ago) - 1) * 100

            logger.info(f"SPY {available_days}-day return: {self.spy_return_20d:+.2f}%")
            logger.info(f"SPY 5-day return: {self.spy_return_5d:+.2f}%")
            logger.info(f"SPY current price: ${current_price:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to fetch SPY data: {e}")
            return False

    async def get_stocks_with_metrics(self, calc_date: date = None) -> List[Dict]:
        """Get all stocks that have metrics for the given date."""
        if calc_date is None:
            calc_date = datetime.now().date()

        query = """
            SELECT
                m.ticker,
                COALESCE(m.price_change_20d, m.perf_1m) as stock_return,
                m.price_change_5d
            FROM stock_screening_metrics m
            WHERE m.calculation_date = $1
              AND (m.price_change_20d IS NOT NULL OR m.perf_1m IS NOT NULL)
        """
        rows = await self.db.fetch(query, calc_date)
        return [dict(r) for r in rows]

    async def get_previous_rs_values(self, calc_date: date) -> Dict[str, float]:
        """Get RS values from ~5 days ago for trend calculation."""
        past_date = calc_date - timedelta(days=7)  # Account for weekends

        query = """
            SELECT ticker, rs_vs_spy
            FROM stock_screening_metrics
            WHERE calculation_date = (
                SELECT MAX(calculation_date)
                FROM stock_screening_metrics
                WHERE calculation_date <= $1 AND calculation_date < $2
                  AND rs_vs_spy IS NOT NULL
            )
              AND rs_vs_spy IS NOT NULL
        """
        try:
            rows = await self.db.fetch(query, past_date, calc_date)
            return {r['ticker']: float(r['rs_vs_spy']) for r in rows}
        except Exception:
            return {}

    def calculate_rs_trend(self, current_rs: float, prev_rs: Optional[float]) -> str:
        """Calculate RS trend based on change from previous period."""
        if prev_rs is None or prev_rs == 0:
            return 'stable'

        change_pct = ((current_rs / prev_rs) - 1) * 100

        if change_pct > 5:
            return 'improving'
        elif change_pct < -5:
            return 'deteriorating'
        else:
            return 'stable'

    async def populate_rs(self, calc_date: date = None) -> Dict:
        """
        Main entry point: Calculate and populate RS metrics for all stocks.

        Returns statistics about the operation.
        """
        if calc_date is None:
            calc_date = datetime.now().date()

        logger.info("=" * 60)
        logger.info("RELATIVE STRENGTH POPULATION")
        logger.info(f"Calculation date: {calc_date}")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Step 1: Get SPY benchmark returns
        if not await self.get_spy_returns():
            return {'error': 'Failed to get SPY data'}

        # Step 2: Get all stocks with metrics
        stocks = await self.get_stocks_with_metrics(calc_date)
        logger.info(f"Found {len(stocks)} stocks with metrics for {calc_date}")

        if not stocks:
            return {'error': 'No stocks found with metrics', 'date': str(calc_date)}

        # Step 3: Get previous RS values for trend calculation
        prev_rs_values = await self.get_previous_rs_values(calc_date)
        logger.info(f"Found {len(prev_rs_values)} previous RS values for trend calculation")

        # Step 4: Calculate RS for each stock
        rs_data = []
        for stock in stocks:
            ticker = stock['ticker']
            stock_return = stock.get('stock_return')

            # Convert Decimal to float if needed
            if stock_return is not None:
                stock_return = float(stock_return)

            if stock_return is not None:
                # RS calculation - handle edge cases when SPY return is near zero
                # When SPY is flat, use stock return directly to rank stocks
                if abs(self.spy_return_20d) < 0.5:  # SPY return less than 0.5%
                    # Use stock return as the RS indicator directly
                    # Normalize to center around 1.0
                    # +10% stock = 2.0 RS, -10% stock = 0.5 RS
                    rs_vs_spy = 1.0 + (stock_return / 20.0)  # 20% swing = 1.0 RS change
                elif self.spy_return_20d > 0:
                    # Normal case: SPY is up
                    # RS ratio: stock return / SPY return
                    rs_vs_spy = stock_return / self.spy_return_20d
                else:
                    # SPY is down - invert so "less bad" = higher RS
                    # Stock down 5%, SPY down 10% = 0.5x = good (outperforming)
                    # Normalize: stock_return / spy_return, but flip sign
                    rs_vs_spy = stock_return / self.spy_return_20d
                    # Flip to make positive returns = higher RS
                    # Down 5% vs down 10% = 0.5 ratio = outperforming

                # Prevent zero/negative values only (no upper cap)
                rs_vs_spy = max(0.01, rs_vs_spy)

                # Calculate trend
                prev_rs = prev_rs_values.get(ticker)
                rs_trend = self.calculate_rs_trend(rs_vs_spy, prev_rs)

                rs_data.append({
                    'ticker': ticker,
                    'rs_vs_spy': round(rs_vs_spy, 4),
                    'stock_return': stock_return,
                    'rs_trend': rs_trend
                })

        logger.info(f"Calculated RS for {len(rs_data)} stocks")

        # Step 5: Calculate percentiles
        if rs_data:
            rs_values = [d['rs_vs_spy'] for d in rs_data]
            for data in rs_data:
                # Percentile: what % of stocks have lower RS
                count_below = sum(1 for r in rs_values if r < data['rs_vs_spy'])
                data['rs_percentile'] = int((count_below / len(rs_values)) * 100)

        # Step 6: Update database
        updated = await self._update_rs_metrics(rs_data, calc_date)

        # Summary statistics
        elapsed = (datetime.now() - start_time).total_seconds()

        # Calculate distribution
        elite = sum(1 for d in rs_data if d['rs_percentile'] >= 90)
        strong = sum(1 for d in rs_data if 70 <= d['rs_percentile'] < 90)
        average = sum(1 for d in rs_data if 40 <= d['rs_percentile'] < 70)
        weak = sum(1 for d in rs_data if d['rs_percentile'] < 40)

        improving = sum(1 for d in rs_data if d['rs_trend'] == 'improving')
        stable = sum(1 for d in rs_data if d['rs_trend'] == 'stable')
        deteriorating = sum(1 for d in rs_data if d['rs_trend'] == 'deteriorating')

        stats = {
            'calculation_date': str(calc_date),
            'spy_return_20d': round(self.spy_return_20d, 2),
            'total_stocks': len(stocks),
            'processed': len(rs_data),
            'updated': updated,
            'duration_seconds': round(elapsed, 2),
            'distribution': {
                'elite_90+': elite,
                'strong_70-89': strong,
                'average_40-69': average,
                'weak_0-39': weak
            },
            'trends': {
                'improving': improving,
                'stable': stable,
                'deteriorating': deteriorating
            }
        }

        logger.info("\n" + "=" * 60)
        logger.info("RS POPULATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {len(rs_data)} stocks")
        logger.info(f"Updated: {updated}")
        logger.info(f"Duration: {elapsed:.1f}s")
        logger.info(f"\nDistribution:")
        logger.info(f"  Elite (90+):    {elite:4d} ({elite/len(rs_data)*100:.1f}%)")
        logger.info(f"  Strong (70-89): {strong:4d} ({strong/len(rs_data)*100:.1f}%)")
        logger.info(f"  Average (40-69):{average:4d} ({average/len(rs_data)*100:.1f}%)")
        logger.info(f"  Weak (0-39):    {weak:4d} ({weak/len(rs_data)*100:.1f}%)")
        logger.info(f"\nTrends:")
        logger.info(f"  Improving:     {improving:4d}")
        logger.info(f"  Stable:        {stable:4d}")
        logger.info(f"  Deteriorating: {deteriorating:4d}")

        return stats

    async def _update_rs_metrics(self, rs_data: List[Dict], calc_date: date) -> int:
        """Update RS columns in stock_screening_metrics."""
        updated = 0
        batch_size = 100

        for i in range(0, len(rs_data), batch_size):
            batch = rs_data[i:i + batch_size]

            for data in batch:
                try:
                    query = """
                        UPDATE stock_screening_metrics
                        SET rs_vs_spy = $1,
                            rs_percentile = $2,
                            rs_trend = $3
                        WHERE ticker = $4 AND calculation_date = $5
                    """
                    await self.db.execute(
                        query,
                        data['rs_vs_spy'],
                        data['rs_percentile'],
                        data['rs_trend'],
                        data['ticker'],
                        calc_date
                    )
                    updated += 1
                except Exception as e:
                    logger.warning(f"Failed to update RS for {data['ticker']}: {e}")

            if (i + batch_size) % 500 == 0:
                logger.info(f"  Updated {i + batch_size}/{len(rs_data)} stocks...")

        return updated


async def main():
    parser = argparse.ArgumentParser(description='Populate RS metrics')
    parser.add_argument('--date', type=str, help='Calculation date (YYYY-MM-DD), defaults to today')
    args = parser.parse_args()

    # Parse date
    calc_date = None
    if args.date:
        calc_date = datetime.strptime(args.date, '%Y-%m-%d').date()

    # Connect to database
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    try:
        populator = RSPopulator(db)
        stats = await populator.populate_rs(calc_date)

        if 'error' in stats:
            logger.error(f"RS population failed: {stats['error']}")
            sys.exit(1)

        print(f"\nRS population complete: {stats['updated']} stocks updated")

    finally:
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
