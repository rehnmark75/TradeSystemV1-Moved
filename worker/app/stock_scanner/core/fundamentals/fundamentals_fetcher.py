"""
Fundamentals Data Fetcher

Fetches fundamental data from yfinance and stores in PostgreSQL.

Data includes:
- Earnings date (critical for avoiding earnings surprises)
- Beta (useful for risk/position sizing)
- Short interest (squeeze potential)
- Institutional ownership
- Valuation metrics (P/E, P/B, etc.)

This data is refreshed weekly as fundamentals don't change frequently.
"""

import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf

logger = logging.getLogger(__name__)


class FundamentalsFetcher:
    """
    Fetches and manages fundamental stock data from yfinance.

    Data sources:
    - yfinance: Earnings dates, beta, short interest, valuation metrics

    Usage:
        fetcher = FundamentalsFetcher(db_manager)
        await fetcher.fetch_all_fundamentals(concurrency=10)
    """

    def __init__(self, db_manager, thread_pool_size: int = 4):
        """
        Initialize fundamentals fetcher.

        Args:
            db_manager: Async database manager instance
            thread_pool_size: Size of thread pool for yfinance calls
        """
        self.db = db_manager
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)

    def _fetch_fundamentals_sync(self, ticker: str) -> Optional[Dict]:
        """
        Synchronous yfinance fundamentals fetch (runs in thread pool).

        Args:
            ticker: Stock ticker

        Returns:
            Dict with fundamental data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            # Extract earnings date
            earnings_date = None
            earnings_estimated = True

            # Try calendar first (upcoming earnings)
            try:
                calendar = stock.calendar
                if calendar:
                    # calendar can be a dict or DataFrame
                    if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                        earnings_dates = calendar['Earnings Date']
                        if earnings_dates and len(earnings_dates) > 0:
                            ed = earnings_dates[0]
                            if isinstance(ed, date):
                                earnings_date = ed
                            elif isinstance(ed, datetime):
                                earnings_date = ed.date()
                    elif hasattr(calendar, 'columns') and 'Earnings Date' in calendar.columns:
                        # DataFrame format
                        earnings_dates = calendar['Earnings Date']
                        if len(earnings_dates) > 0 and earnings_dates[0]:
                            ed = earnings_dates[0]
                            if isinstance(ed, datetime):
                                earnings_date = ed.date()
                            elif hasattr(ed, 'date'):
                                earnings_date = ed.date()
            except Exception:
                pass

            # Fallback to earningsTimestamp from info
            if not earnings_date:
                try:
                    ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
                    if ts:
                        earnings_date = datetime.fromtimestamp(ts).date()
                    # Check if estimated
                    if 'isEarningsDateEstimate' in info:
                        earnings_estimated = bool(info['isEarningsDateEstimate'])
                except Exception:
                    pass

            # Extract fundamentals
            fundamentals = {
                'ticker': ticker,
                'earnings_date': earnings_date,
                'earnings_date_estimated': earnings_estimated,
                'beta': self._safe_get(info, 'beta'),
                'short_ratio': self._safe_get(info, 'shortRatio'),
                'short_percent_float': self._safe_get(info, 'shortPercentOfFloat', multiply=100),
                'institutional_percent': self._safe_get(info, 'heldPercentInstitutions', multiply=100),
                'forward_pe': self._safe_get(info, 'forwardPE'),
                'trailing_pe': self._safe_get(info, 'trailingPE'),
                'price_to_book': self._safe_get(info, 'priceToBook'),
                'dividend_yield': self._safe_get(info, 'dividendYield', multiply=100),
                'analyst_rating': self._safe_get_str(info, 'recommendationKey'),
                'target_price': self._safe_get(info, 'targetMeanPrice'),
            }

            return fundamentals

        except Exception as e:
            logger.debug(f"Error fetching fundamentals for {ticker}: {e}")
            return None

    def _safe_get(self, info: Dict, key: str, multiply: float = None, max_val: float = 99999) -> Optional[float]:
        """Safely extract a numeric value from info dict."""
        try:
            val = info.get(key)
            if val is None or val == 'Infinity' or val == '-Infinity':
                return None
            val = float(val)
            if multiply:
                val = val * multiply
            # Cap extreme values to avoid database overflow
            if val > max_val:
                val = max_val
            elif val < -max_val:
                val = -max_val
            return round(val, 3)
        except (ValueError, TypeError):
            return None

    def _safe_get_str(self, info: Dict, key: str) -> Optional[str]:
        """Safely extract a string value from info dict."""
        try:
            val = info.get(key)
            return str(val) if val else None
        except Exception:
            return None

    async def fetch_fundamentals(self, ticker: str) -> Optional[Dict]:
        """
        Fetch fundamentals for a single ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with fundamental data or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_fundamentals_sync,
            ticker
        )

    async def save_fundamentals(self, fundamentals: Dict) -> bool:
        """
        Save fundamentals to database.

        Args:
            fundamentals: Dict with fundamental data

        Returns:
            True if successful
        """
        query = """
            UPDATE stock_instruments SET
                earnings_date = $2,
                earnings_date_estimated = $3,
                beta = $4,
                short_ratio = $5,
                short_percent_float = $6,
                institutional_percent = $7,
                forward_pe = $8,
                trailing_pe = $9,
                price_to_book = $10,
                dividend_yield = $11,
                analyst_rating = $12,
                target_price = $13,
                fundamentals_updated_at = NOW(),
                updated_at = NOW()
            WHERE ticker = $1
        """

        try:
            await self.db.execute(
                query,
                fundamentals['ticker'],
                fundamentals['earnings_date'],
                fundamentals['earnings_date_estimated'],
                fundamentals['beta'],
                fundamentals['short_ratio'],
                fundamentals['short_percent_float'],
                fundamentals['institutional_percent'],
                fundamentals['forward_pe'],
                fundamentals['trailing_pe'],
                fundamentals['price_to_book'],
                fundamentals['dividend_yield'],
                fundamentals['analyst_rating'],
                fundamentals['target_price'],
            )
            return True
        except Exception as e:
            logger.error(f"Error saving fundamentals for {fundamentals['ticker']}: {e}")
            return False

    async def get_tickers_needing_update(self, days_stale: int = 7) -> List[str]:
        """
        Get tickers that need fundamentals update.

        Args:
            days_stale: Number of days before data is considered stale

        Returns:
            List of tickers needing update
        """
        query = """
            SELECT ticker FROM stock_instruments
            WHERE is_active = TRUE AND is_tradeable = TRUE
            AND (fundamentals_updated_at IS NULL
                 OR fundamentals_updated_at < NOW() - INTERVAL '%s days')
            ORDER BY ticker
        """

        rows = await self.db.fetch(query % days_stale)
        return [row['ticker'] for row in rows]

    async def fetch_all_fundamentals(
        self,
        concurrency: int = 10,
        force: bool = False,
        tickers: List[str] = None
    ) -> Dict:
        """
        Fetch fundamentals for all stocks.

        Args:
            concurrency: Max concurrent fetches
            force: If True, update all stocks regardless of age
            tickers: Optional list of specific tickers to update

        Returns:
            Stats dict
        """
        if tickers:
            tickers_to_update = tickers
        elif force:
            query = """
                SELECT ticker FROM stock_instruments
                WHERE is_active = TRUE AND is_tradeable = TRUE
                ORDER BY ticker
            """
            rows = await self.db.fetch(query)
            tickers_to_update = [row['ticker'] for row in rows]
        else:
            tickers_to_update = await self.get_tickers_needing_update()

        if not tickers_to_update:
            logger.info("All fundamentals are up to date")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'with_earnings': 0,
                'with_beta': 0,
                'with_short_interest': 0,
            }

        logger.info(f"Fetching fundamentals for {len(tickers_to_update)} stocks...")

        semaphore = asyncio.Semaphore(concurrency)
        successful = 0
        failed = 0
        with_earnings = 0
        with_beta = 0
        with_short = 0

        async def fetch_and_save(ticker: str) -> Tuple[str, bool, Optional[Dict]]:
            async with semaphore:
                try:
                    fundamentals = await self.fetch_fundamentals(ticker)
                    if fundamentals:
                        success = await self.save_fundamentals(fundamentals)
                        return (ticker, success, fundamentals)
                    return (ticker, False, None)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    return (ticker, False, None)

        tasks = [fetch_and_save(t) for t in tickers_to_update]
        results = await asyncio.gather(*tasks)

        for ticker, success, fundamentals in results:
            if success and fundamentals:
                successful += 1
                if fundamentals.get('earnings_date'):
                    with_earnings += 1
                if fundamentals.get('beta'):
                    with_beta += 1
                if fundamentals.get('short_ratio') or fundamentals.get('short_percent_float'):
                    with_short += 1
            else:
                failed += 1

        stats = {
            'total': len(tickers_to_update),
            'successful': successful,
            'failed': failed,
            'with_earnings': with_earnings,
            'with_beta': with_beta,
            'with_short_interest': with_short,
        }

        logger.info(
            f"Fundamentals sync: {successful}/{len(tickers_to_update)} successful, "
            f"earnings: {with_earnings}, beta: {with_beta}, short: {with_short}"
        )

        return stats

    async def run_fundamentals_pipeline(self) -> Dict:
        """
        Run the fundamentals pipeline (for weekly sync).

        Returns:
            Stats dict
        """
        logger.info("Starting fundamentals pipeline...")
        return await self.fetch_all_fundamentals(concurrency=10)

    async def get_upcoming_earnings(self, days_ahead: int = 14) -> List[Dict]:
        """
        Get stocks with earnings in the next N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of stocks with upcoming earnings
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.earnings_date,
                i.earnings_date_estimated,
                m.close as current_price,
                m.trend
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
            WHERE i.earnings_date IS NOT NULL
            AND i.earnings_date >= CURRENT_DATE
            AND i.earnings_date <= CURRENT_DATE + INTERVAL '%s days'
            ORDER BY i.earnings_date
        """

        rows = await self.db.fetch(query % days_ahead)
        return [dict(row) for row in rows]

    async def get_high_short_interest(self, min_short_percent: float = 15.0) -> List[Dict]:
        """
        Get stocks with high short interest (squeeze potential).

        Args:
            min_short_percent: Minimum short percent of float

        Returns:
            List of stocks with high short interest
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.short_ratio,
                i.short_percent_float,
                m.close as current_price,
                m.relative_volume
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
            WHERE i.short_percent_float >= $1
            ORDER BY i.short_percent_float DESC
        """

        rows = await self.db.fetch(query, min_short_percent)
        return [dict(row) for row in rows]

    async def close(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
