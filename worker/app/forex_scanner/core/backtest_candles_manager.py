# core/backtest_candles_manager.py
"""
Backtest Candles Manager - Validates and updates the ig_candles_backtest table.

This module ensures the backtest candles table has up-to-date pre-computed
candles (5m, 15m, 4h) before backtests run. Integrates with BacktestScanner
for lazy validation - first run populates data, subsequent runs are fast.

Usage:
    manager = BacktestCandlesManager(db_manager)
    manager.ensure_data_current(
        epics=['CS.D.EURUSD.CEEM.IP'],
        start_date=start,
        end_date=end
    )
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import pandas as pd

try:
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner.core.database import DatabaseManager


logger = logging.getLogger(__name__)


# Default timeframes to pre-compute (in minutes)
DEFAULT_BACKTEST_TIMEFRAMES = [5, 15, 60, 240]  # 5m, 15m, 1h, 4h


def resample_ohlcv(df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    """
    Resample OHLCV data to a larger timeframe.

    Args:
        df: DataFrame with OHLCV columns and datetime index
        target_minutes: Target timeframe in minutes (5, 15, 240)

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'start_time' in df.columns:
            df = df.set_index('start_time')
        else:
            raise ValueError("DataFrame must have datetime index or start_time column")

    # Sort by time
    df = df.sort_index()

    # Resample rule
    rule = f'{target_minutes}min'

    # OHLCV aggregation
    resampled = df.resample(rule, closed='left', label='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ltv': 'sum'
    }).dropna()

    return resampled


class BacktestCandlesManager:
    """
    Manages the ig_candles_backtest table - validates freshness and updates as needed.
    """

    def __init__(self, db_manager: DatabaseManager, timeframes: List[int] = None):
        """
        Initialize the manager.

        Args:
            db_manager: Database manager instance
            timeframes: List of timeframes to manage (default: [5, 15, 240])
        """
        self.db = db_manager
        self.timeframes = timeframes or DEFAULT_BACKTEST_TIMEFRAMES
        self.logger = logging.getLogger(__name__)

    def table_exists(self) -> bool:
        """Check if the backtest candles table exists."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'ig_candles_backtest'
            )
        """
        result = self.db.execute_query(query)
        return result.iloc[0, 0] if not result.empty else False

    def create_table_if_needed(self) -> bool:
        """Create the backtest candles table if it doesn't exist."""
        if self.table_exists():
            return False  # Already exists

        self.logger.info("📦 Creating ig_candles_backtest table...")

        create_query = """
        CREATE TABLE IF NOT EXISTS ig_candles_backtest (
            start_time       TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            epic             VARCHAR NOT NULL,
            timeframe        INTEGER NOT NULL,
            open             DOUBLE PRECISION NOT NULL,
            high             DOUBLE PRECISION NOT NULL,
            low              DOUBLE PRECISION NOT NULL,
            close            DOUBLE PRECISION NOT NULL,
            volume           INTEGER NOT NULL,
            ltv              INTEGER,
            resampled_from   INTEGER DEFAULT 1,
            created_at       TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (start_time, epic, timeframe)
        );

        CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic
        ON ig_candles_backtest (epic);

        CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic_tf_time
        ON ig_candles_backtest (epic, timeframe, start_time DESC);
        """

        try:
            self.db.execute_query(create_query)
            self.logger.info("✅ Created ig_candles_backtest table")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create table: {e}")
            raise

    def get_data_range(self, epic: str, timeframe: int) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the min/max timestamps for an epic+timeframe in the backtest table.

        Returns:
            Tuple of (min_time, max_time) or (None, None) if no data
        """
        query = """
            SELECT MIN(start_time) as min_time, MAX(start_time) as max_time
            FROM ig_candles_backtest
            WHERE epic = :epic AND timeframe = :timeframe
        """
        result = self.db.execute_query(query, {'epic': epic, 'timeframe': timeframe})

        if result.empty or result.iloc[0]['min_time'] is None:
            return None, None

        return result.iloc[0]['min_time'], result.iloc[0]['max_time']

    def get_source_data_range(self, epic: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the min/max timestamps for 1m candles in the source table.

        Returns:
            Tuple of (min_time, max_time) or (None, None) if no data
        """
        query = """
            SELECT MIN(start_time) as min_time, MAX(start_time) as max_time
            FROM ig_candles
            WHERE epic = :epic AND timeframe = 1
        """
        result = self.db.execute_query(query, {'epic': epic})

        if result.empty or result.iloc[0]['min_time'] is None:
            return None, None

        return result.iloc[0]['min_time'], result.iloc[0]['max_time']

    def check_freshness(
        self,
        epic: str,
        start_date: datetime,
        end_date: datetime,
        max_staleness_hours: int = 1
    ) -> Dict:
        """
        Check if backtest data is fresh enough for the requested period.

        Args:
            epic: Epic to check
            start_date: Required start of backtest period
            end_date: Required end of backtest period
            max_staleness_hours: How many hours behind source data is acceptable

        Returns:
            Dict with:
                - is_fresh: bool - True if no update needed
                - missing_start: datetime or None - Data needed from this time
                - missing_end: datetime or None - Data needed until this time
                - timeframes_needing_update: List of timeframes that need data
        """
        result = {
            'is_fresh': True,
            'missing_start': None,
            'missing_end': None,
            'timeframes_needing_update': []
        }

        # Check source data availability
        source_min, source_max = self.get_source_data_range(epic)
        if source_min is None:
            self.logger.warning(f"⚠️ No 1m source data for {epic}")
            result['is_fresh'] = False
            return result

        # Check each timeframe
        for tf in self.timeframes:
            bt_min, bt_max = self.get_data_range(epic, tf)

            needs_update = False
            tf_missing_start = None
            tf_missing_end = None

            if bt_min is None:
                # No backtest data at all
                needs_update = True
                tf_missing_start = start_date
                tf_missing_end = end_date
            else:
                # Check if backtest data covers the requested period
                # Add buffer for lookback (indicator warmup)
                lookback_buffer = timedelta(days=7)
                required_start = start_date - lookback_buffer

                if bt_min > required_start:
                    needs_update = True
                    tf_missing_start = required_start
                    tf_missing_end = bt_min

                if bt_max < end_date:
                    needs_update = True
                    if tf_missing_start is None:
                        tf_missing_start = bt_max
                    tf_missing_end = end_date

                # Also check staleness (is backtest data up-to-date with source?)
                staleness = source_max - bt_max if bt_max else timedelta(hours=999)
                if staleness > timedelta(hours=max_staleness_hours):
                    needs_update = True
                    if tf_missing_start is None:
                        tf_missing_start = bt_max
                    tf_missing_end = source_max

            if needs_update:
                result['is_fresh'] = False
                result['timeframes_needing_update'].append(tf)

                # Track overall missing range
                if tf_missing_start:
                    if result['missing_start'] is None or tf_missing_start < result['missing_start']:
                        result['missing_start'] = tf_missing_start
                if tf_missing_end:
                    if result['missing_end'] is None or tf_missing_end > result['missing_end']:
                        result['missing_end'] = tf_missing_end

        return result

    def resample_epic(
        self,
        epic: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        batch_size: int = 50000
    ) -> Dict[int, int]:
        """
        Resample 1m candles for a single epic into target timeframes.

        Args:
            epic: Epic to resample
            since: Only process data since this time
            until: Only process data until this time
            batch_size: Rows per insert batch

        Returns:
            Dict with counts of inserted rows per timeframe
        """
        results = {tf: 0 for tf in self.timeframes}

        # Build query for 1m data
        query = """
            SELECT start_time, open, high, low, close, volume, COALESCE(ltv, 0) as ltv
            FROM ig_candles
            WHERE epic = :epic AND timeframe = 1
        """
        params = {'epic': epic}

        if since:
            query += " AND start_time >= :since"
            params['since'] = since
        if until:
            query += " AND start_time <= :until"
            params['until'] = until

        query += " ORDER BY start_time"

        self.logger.info(f"  📊 Fetching 1m data for {epic}...")
        df_1m = self.db.execute_query(query, params)

        if df_1m.empty:
            self.logger.warning(f"  ⚠️ No 1m data found for {epic}")
            return results

        self.logger.info(f"  Found {len(df_1m):,} 1m candles")

        # Set index for resampling
        df_1m['start_time'] = pd.to_datetime(df_1m['start_time'])
        df_1m = df_1m.set_index('start_time')

        # Resample to each target timeframe
        for tf_minutes in self.timeframes:
            self.logger.info(f"  Resampling to {tf_minutes}m...")

            df_resampled = resample_ohlcv(df_1m, tf_minutes)

            if df_resampled.empty:
                self.logger.warning(f"  No data after resampling to {tf_minutes}m")
                continue

            # Prepare for insert
            df_resampled = df_resampled.reset_index()
            df_resampled['epic'] = epic
            df_resampled['timeframe'] = tf_minutes
            df_resampled['resampled_from'] = 1

            # Delete existing data for this epic/timeframe in the time range (upsert)
            if since or until:
                delete_query = "DELETE FROM ig_candles_backtest WHERE epic = :epic AND timeframe = :tf"
                delete_params = {'epic': epic, 'tf': tf_minutes}
                if since:
                    delete_query += " AND start_time >= :since"
                    delete_params['since'] = since
                if until:
                    delete_query += " AND start_time <= :until"
                    delete_params['until'] = until
                self.db.execute_query(delete_query, delete_params)
            else:
                delete_query = "DELETE FROM ig_candles_backtest WHERE epic = :epic AND timeframe = :tf"
                self.db.execute_query(delete_query, {'epic': epic, 'tf': tf_minutes})

            # Insert in batches
            inserted = 0
            for i in range(0, len(df_resampled), batch_size):
                batch = df_resampled.iloc[i:i + batch_size]

                # Build insert values
                values = []
                for _, row in batch.iterrows():
                    values.append(f"""(
                        '{row['start_time']}',
                        '{epic}',
                        {tf_minutes},
                        {row['open']},
                        {row['high']},
                        {row['low']},
                        {row['close']},
                        {int(row['volume'])},
                        {int(row['ltv'])},
                        1
                    )""")

                if values:
                    insert_query = f"""
                        INSERT INTO ig_candles_backtest
                        (start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from)
                        VALUES {', '.join(values)}
                        ON CONFLICT (start_time, epic, timeframe) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            ltv = EXCLUDED.ltv
                    """
                    self.db.execute_query(insert_query)
                    inserted += len(batch)

            results[tf_minutes] = inserted
            self.logger.info(f"  ✅ Inserted {inserted:,} {tf_minutes}m candles")

        return results

    def ensure_data_current(
        self,
        epics: List[str],
        start_date: datetime,
        end_date: datetime,
        max_staleness_hours: int = 1
    ) -> Dict:
        """
        Ensure backtest data is current for the given epics and period.

        This is the main entry point - call this before running a backtest.
        First run may take a while to populate data, subsequent runs are fast.

        Args:
            epics: List of epics to validate
            start_date: Backtest start date
            end_date: Backtest end date
            max_staleness_hours: Max hours behind source data before update

        Returns:
            Dict with:
                - updated: bool - True if any data was updated
                - epics_updated: List of epics that were updated
                - total_rows_inserted: Total rows added
                - time_taken_seconds: Time to complete
        """
        import time
        start_time = time.time()

        result = {
            'updated': False,
            'epics_updated': [],
            'total_rows_inserted': 0,
            'time_taken_seconds': 0
        }

        # Ensure table exists
        self.create_table_if_needed()

        # Check and update each epic
        for epic in epics:
            freshness = self.check_freshness(epic, start_date, end_date, max_staleness_hours)

            if freshness['is_fresh']:
                self.logger.debug(f"✅ {epic}: Backtest data is current")
                continue

            # Need to update
            self.logger.info(f"🔄 {epic}: Updating backtest candles...")
            self.logger.info(f"   Missing range: {freshness['missing_start']} to {freshness['missing_end']}")
            self.logger.info(f"   Timeframes: {freshness['timeframes_needing_update']}")

            # Resample with some buffer before the missing range
            buffer = timedelta(days=1)
            since = freshness['missing_start'] - buffer if freshness['missing_start'] else None
            until = freshness['missing_end']

            counts = self.resample_epic(epic, since=since, until=until)

            total_inserted = sum(counts.values())
            if total_inserted > 0:
                result['updated'] = True
                result['epics_updated'].append(epic)
                result['total_rows_inserted'] += total_inserted

        result['time_taken_seconds'] = time.time() - start_time

        if result['updated']:
            self.logger.info(f"✅ Backtest candles updated: {result['total_rows_inserted']:,} rows "
                           f"for {len(result['epics_updated'])} epics in {result['time_taken_seconds']:.1f}s")
        else:
            self.logger.info("✅ All backtest candles are current - no update needed")

        return result
