"""
Daily Candle Synthesizer

Aggregates 1H candles into daily candles with market hours awareness.
US Market hours: 9:30 AM - 4:00 PM ET (Eastern Time)
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional
import pytz

logger = logging.getLogger(__name__)


class DailyCandleSynthesizer:
    """
    Synthesizes daily candles from 1H data with proper market hours handling.

    Key features:
    - Filters for regular market hours (9:30 AM - 4:00 PM ET)
    - Handles timezone conversions correctly
    - Supports incremental updates
    - Validates data quality
    """

    # US Eastern timezone
    ET = pytz.timezone('America/New_York')
    UTC = pytz.UTC

    # Market hours (ET)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    # Minimum candles for valid daily bar
    MIN_CANDLES_PER_DAY = 4  # At least 4 hours of data

    def __init__(self, db_manager):
        self.db = db_manager

    async def synthesize_all_daily(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        incremental: bool = True,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Synthesize daily candles for all active tickers.

        Args:
            start_date: Start date for synthesis (None = auto-detect)
            end_date: End date for synthesis (None = yesterday)
            incremental: If True, only process new data
            batch_size: Number of tickers to process in parallel

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("DAILY CANDLE SYNTHESIS")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Determine date range
        if end_date is None:
            # Yesterday (don't include incomplete today)
            end_date = datetime.now(self.ET).date() - timedelta(days=1)

        if incremental and start_date is None:
            start_date = await self._get_last_synthesis_date()
            if start_date:
                logger.info(f"Incremental mode: starting from {start_date}")

        # Get active tickers
        tickers = await self._get_active_tickers()
        logger.info(f"Processing {len(tickers)} tickers")
        logger.info(f"Date range: {start_date or 'all'} to {end_date}")

        # Process in batches
        total_candles = 0
        successful = 0
        failed = 0

        semaphore = asyncio.Semaphore(batch_size)

        async def process_ticker(ticker: str):
            async with semaphore:
                try:
                    count = await self.synthesize_ticker_daily(
                        ticker, start_date, end_date
                    )
                    return (ticker, count, None)
                except Exception as e:
                    logger.error(f"Failed to synthesize {ticker}: {e}")
                    return (ticker, 0, str(e))

        tasks = [process_ticker(t) for t in tickers]
        results = await asyncio.gather(*tasks)

        for ticker, count, error in results:
            if error:
                failed += 1
            else:
                successful += 1
                total_candles += count

        # Refresh materialized view
        try:
            await self.db.execute("SELECT refresh_daily_candles_view()")
            logger.info("Refreshed daily candles materialized view")
        except Exception as e:
            logger.warning(f"Could not refresh materialized view: {e}")

        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            'tickers_processed': len(tickers),
            'successful': successful,
            'failed': failed,
            'total_daily_candles': total_candles,
            'start_date': str(start_date) if start_date else 'all',
            'end_date': str(end_date),
            'duration_seconds': round(elapsed, 2)
        }

        logger.info(f"\nSynthesis complete:")
        logger.info(f"  Tickers: {successful}/{len(tickers)}")
        logger.info(f"  Daily candles: {total_candles:,}")
        logger.info(f"  Duration: {int(elapsed//60)}m {int(elapsed%60)}s")

        return stats

    async def synthesize_ticker_daily(
        self,
        ticker: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> int:
        """
        Synthesize daily candles for a single ticker.

        Aggregation logic:
        - Open: First candle's open (9:30 AM or first available)
        - High: Maximum high across all candles
        - Low: Minimum low across all candles
        - Close: Last candle's close (4:00 PM or last available)
        - Volume: Sum of all candle volumes
        """

        # Build date filter
        date_filter = ""
        params = [ticker]
        param_idx = 2

        if start_date:
            date_filter += f" AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') >= ${param_idx}"
            params.append(start_date)
            param_idx += 1

        if end_date:
            date_filter += f" AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') <= ${param_idx}"
            params.append(end_date)
            param_idx += 1

        # Aggregate 1H candles to daily with market hours filter
        query = f"""
            WITH hourly_data AS (
                SELECT
                    ticker,
                    timestamp,
                    timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York' as et_time,
                    open, high, low, close, volume
                FROM stock_candles
                WHERE ticker = $1
                  AND timeframe = '1h'
                  {date_filter}
            ),
            market_hours AS (
                SELECT *
                FROM hourly_data
                WHERE EXTRACT(HOUR FROM et_time) >= 9
                  AND (EXTRACT(HOUR FROM et_time) < 16
                       OR (EXTRACT(HOUR FROM et_time) = 16 AND EXTRACT(MINUTE FROM et_time) = 0))
                  -- Exclude weekends
                  AND EXTRACT(DOW FROM et_time) BETWEEN 1 AND 5
            ),
            daily_agg AS (
                SELECT
                    ticker,
                    DATE(et_time) as trading_date,
                    (ARRAY_AGG(open ORDER BY timestamp ASC))[1] as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (ARRAY_AGG(close ORDER BY timestamp DESC))[1] as close,
                    SUM(volume) as volume,
                    COUNT(*) as candle_count
                FROM market_hours
                GROUP BY ticker, DATE(et_time)
                HAVING COUNT(*) >= {self.MIN_CANDLES_PER_DAY}
            )
            INSERT INTO stock_candles_synthesized (
                ticker, timeframe, timestamp, open, high, low, close, volume,
                source_timeframe, candles_used
            )
            SELECT
                ticker,
                '1d',
                (trading_date || ' 16:00:00')::TIMESTAMP AT TIME ZONE 'America/New_York' AT TIME ZONE 'UTC',
                open, high, low, close, volume,
                '1h',
                candle_count
            FROM daily_agg
            ON CONFLICT (ticker, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                candles_used = EXCLUDED.candles_used,
                created_at = NOW()
            RETURNING 1
        """

        result = await self.db.fetch(query, *params)
        return len(result)

    async def validate_synthesis(
        self,
        ticker: str,
        date: datetime
    ) -> Dict[str, Any]:
        """
        Validate a synthesized daily candle against source 1H data.

        Checks:
        - Candle count (should be 6-7 for full day)
        - High is max of all highs
        - Low is min of all lows
        - Volume sums correctly
        """
        query = """
            WITH daily AS (
                SELECT * FROM stock_candles_synthesized
                WHERE ticker = $1 AND timeframe = '1d'
                  AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') = $2
            ),
            hourly AS (
                SELECT
                    COUNT(*) as count,
                    MAX(high) as max_high,
                    MIN(low) as min_low,
                    SUM(volume) as total_volume
                FROM stock_candles
                WHERE ticker = $1 AND timeframe = '1h'
                  AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') = $2
            )
            SELECT
                d.candles_used,
                h.count as hourly_count,
                d.high = h.max_high as high_valid,
                d.low = h.min_low as low_valid,
                d.volume = h.total_volume as volume_valid
            FROM daily d, hourly h
        """

        result = await self.db.fetch_one(query, ticker, date)

        if not result:
            return {'valid': False, 'error': 'No data found'}

        return {
            'valid': result['high_valid'] and result['low_valid'] and result['volume_valid'],
            'candles_used': result['candles_used'],
            'hourly_count': result['hourly_count'],
            'high_valid': result['high_valid'],
            'low_valid': result['low_valid'],
            'volume_valid': result['volume_valid']
        }

    async def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about synthesized daily candles."""
        query = """
            SELECT
                COUNT(DISTINCT ticker) as tickers_with_data,
                COUNT(*) as total_daily_candles,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date,
                AVG(candles_used) as avg_candles_per_day
            FROM stock_candles_synthesized
            WHERE timeframe = '1d'
        """

        result = await self.db.fetch_one(query)
        return dict(result) if result else {}

    async def _get_last_synthesis_date(self) -> Optional[datetime]:
        """Get the last date we synthesized daily candles for."""
        query = """
            SELECT MAX(DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York'))
            FROM stock_candles_synthesized
            WHERE timeframe = '1d'
        """

        result = await self.db.fetchval(query)

        if result:
            # Start from the day after last synthesis
            return result + timedelta(days=1)

        return None

    async def _get_active_tickers(self) -> List[str]:
        """Get list of active tradeable tickers with data."""
        query = """
            SELECT DISTINCT i.ticker
            FROM stock_instruments i
            JOIN stock_candles c ON i.ticker = c.ticker
            WHERE i.is_active = TRUE
              AND i.is_tradeable = TRUE
              AND c.timeframe = '1h'
            ORDER BY i.ticker
        """

        rows = await self.db.fetch(query)
        return [r['ticker'] for r in rows]

    # =========================================================================
    # WEEKLY CANDLE SYNTHESIS
    # =========================================================================

    async def synthesize_all_weekly(
        self,
        start_date: datetime = None,
        end_date: datetime = None,
        incremental: bool = True,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Synthesize weekly candles from daily data for all active tickers.

        Weekly candles run Monday-Friday (trading week).

        Args:
            start_date: Start date for synthesis (None = auto-detect)
            end_date: End date for synthesis (None = last complete week)
            incremental: If True, only process new data
            batch_size: Number of tickers to process in parallel

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("WEEKLY CANDLE SYNTHESIS")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Determine date range - end at last complete week (last Friday)
        if end_date is None:
            today = datetime.now(self.ET).date()
            # Find last Friday
            days_since_friday = (today.weekday() - 4) % 7
            if days_since_friday == 0 and datetime.now(self.ET).hour < 16:
                days_since_friday = 7  # Current Friday not complete yet
            end_date = today - timedelta(days=days_since_friday)

        if incremental and start_date is None:
            start_date = await self._get_last_weekly_synthesis_date()
            if start_date:
                logger.info(f"Incremental mode: starting from {start_date}")

        # Get tickers with daily data
        tickers = await self._get_tickers_with_daily_data()
        logger.info(f"Processing {len(tickers)} tickers for weekly synthesis")
        logger.info(f"Date range: {start_date or 'all'} to {end_date}")

        # Process in batches
        total_candles = 0
        successful = 0
        failed = 0

        semaphore = asyncio.Semaphore(batch_size)

        async def process_ticker(ticker: str):
            async with semaphore:
                try:
                    count = await self.synthesize_ticker_weekly(
                        ticker, start_date, end_date
                    )
                    return (ticker, count, None)
                except Exception as e:
                    logger.error(f"Failed to synthesize weekly {ticker}: {e}")
                    return (ticker, 0, str(e))

        tasks = [process_ticker(t) for t in tickers]
        results = await asyncio.gather(*tasks)

        for ticker, count, error in results:
            if error:
                failed += 1
            else:
                successful += 1
                total_candles += count

        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            'tickers_processed': len(tickers),
            'successful': successful,
            'failed': failed,
            'total_weekly_candles': total_candles,
            'start_date': str(start_date) if start_date else 'all',
            'end_date': str(end_date),
            'duration_seconds': round(elapsed, 2)
        }

        logger.info(f"\nWeekly synthesis complete:")
        logger.info(f"  Tickers: {successful}/{len(tickers)}")
        logger.info(f"  Weekly candles: {total_candles:,}")
        logger.info(f"  Duration: {int(elapsed//60)}m {int(elapsed%60)}s")

        return stats

    async def synthesize_ticker_weekly(
        self,
        ticker: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> int:
        """
        Synthesize weekly candles for a single ticker from daily data.

        Aggregation logic:
        - Open: Monday's open (or first trading day of week)
        - High: Maximum high across the week
        - Low: Minimum low across the week
        - Close: Friday's close (or last trading day of week)
        - Volume: Sum of daily volumes

        Week boundaries: Monday 00:00 to Friday 23:59 ET
        """
        # Build date filter
        date_filter = ""
        params = [ticker]
        param_idx = 2

        if start_date:
            date_filter += f" AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') >= ${param_idx}"
            params.append(start_date)
            param_idx += 1

        if end_date:
            date_filter += f" AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') <= ${param_idx}"
            params.append(end_date)
            param_idx += 1

        # Aggregate daily candles to weekly
        # Use DATE_TRUNC to get Monday of each week
        query = f"""
            WITH daily_data AS (
                SELECT
                    ticker,
                    timestamp,
                    DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') as trading_date,
                    open, high, low, close, volume
                FROM stock_candles_synthesized
                WHERE ticker = $1
                  AND timeframe = '1d'
                  {date_filter}
            ),
            weekly_agg AS (
                SELECT
                    ticker,
                    DATE_TRUNC('week', trading_date)::DATE as week_start,
                    (ARRAY_AGG(open ORDER BY trading_date ASC))[1] as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    (ARRAY_AGG(close ORDER BY trading_date DESC))[1] as close,
                    SUM(volume) as volume,
                    COUNT(*) as candle_count,
                    MIN(trading_date) as first_day,
                    MAX(trading_date) as last_day
                FROM daily_data
                GROUP BY ticker, DATE_TRUNC('week', trading_date)::DATE
                HAVING COUNT(*) >= 3  -- At least 3 trading days for valid week
            )
            INSERT INTO stock_candles_synthesized (
                ticker, timeframe, timestamp, open, high, low, close, volume,
                source_timeframe, candles_used
            )
            SELECT
                ticker,
                '1w',
                -- Timestamp is Friday 4PM ET (end of trading week)
                (week_start + INTERVAL '4 days' + INTERVAL '16 hours') AT TIME ZONE 'America/New_York' AT TIME ZONE 'UTC',
                open, high, low, close, volume,
                '1d',
                candle_count
            FROM weekly_agg
            ON CONFLICT (ticker, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                candles_used = EXCLUDED.candles_used,
                created_at = NOW()
            RETURNING 1
        """

        result = await self.db.fetch(query, *params)
        return len(result)

    async def _get_last_weekly_synthesis_date(self) -> Optional[datetime]:
        """Get the last week we synthesized weekly candles for."""
        query = """
            SELECT MAX(DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York'))
            FROM stock_candles_synthesized
            WHERE timeframe = '1w'
        """

        result = await self.db.fetchval(query)

        if result:
            # Start from Monday of the next week
            days_until_monday = (7 - result.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return result + timedelta(days=days_until_monday)

        return None

    async def _get_tickers_with_daily_data(self) -> List[str]:
        """Get list of tickers that have synthesized daily data."""
        query = """
            SELECT DISTINCT ticker
            FROM stock_candles_synthesized
            WHERE timeframe = '1d'
            ORDER BY ticker
        """

        rows = await self.db.fetch(query)
        return [r['ticker'] for r in rows]

    async def get_weekly_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about synthesized weekly candles."""
        query = """
            SELECT
                COUNT(DISTINCT ticker) as tickers_with_data,
                COUNT(*) as total_weekly_candles,
                MIN(timestamp) as earliest_week,
                MAX(timestamp) as latest_week,
                AVG(candles_used) as avg_days_per_week
            FROM stock_candles_synthesized
            WHERE timeframe = '1w'
        """

        result = await self.db.fetch_one(query)
        return dict(result) if result else {}
