# core/backtest_data_fetcher.py
"""
BacktestDataFetcher - Specialized DataFetcher for historical data
Extends DataFetcher with backtest-specific optimizations
"""

import logging
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json

try:
    import config
    from core.data_fetcher import DataFetcher
    from core.database import DatabaseManager
    from core.memory_cache import get_forex_cache, initialize_cache
    from utils.timezone_utils import add_timezone_columns
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.data_fetcher import DataFetcher
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.memory_cache import get_forex_cache, initialize_cache
    from forex_scanner.utils.timezone_utils import add_timezone_columns


class BacktestDataValidator:
    """Data validation for backtest integrity"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

        # Market schedule for gap validation
        self.market_schedule = {
            'forex': {
                'sunday_open': 21,  # UTC hour
                'friday_close': 21,  # UTC hour
                'daily_maintenance': [(21, 22)]  # UTC hours
            }
        }

    def validate_historical_data(
        self,
        epic: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Comprehensive historical data validation"""

        validation_results = {
            'epic': epic,
            'timeframe': timeframe,
            'period': f"{start_date} to {end_date}",
            'completeness_score': 0.0,
            'quality_score': 0.0,
            'gaps_detected': [],
            'price_anomalies': [],
            'volume_anomalies': [],
            'validation_passed': False
        }

        try:
            # 1. Data Completeness Check
            completeness = self._check_data_completeness(
                epic, timeframe, start_date, end_date
            )
            validation_results.update(completeness)

            # 2. Price Data Integrity
            price_validation = self._validate_price_data(
                epic, timeframe, start_date, end_date
            )
            validation_results['price_anomalies'] = price_validation['anomalies']

            # 3. Volume Data Validation
            volume_validation = self._validate_volume_data(
                epic, timeframe, start_date, end_date
            )
            validation_results['volume_anomalies'] = volume_validation['anomalies']

            # Calculate overall quality score
            validation_results['quality_score'] = self._calculate_quality_score(
                validation_results
            )

            # Determine if validation passed
            validation_results['validation_passed'] = (
                validation_results['completeness_score'] >= 0.90 and
                validation_results['quality_score'] >= 0.85 and
                len(validation_results['price_anomalies']) <= 5
            )

            return validation_results

        except Exception as e:
            self.logger.error(f"Error validating historical data: {e}")
            validation_results['validation_passed'] = False
            validation_results['error'] = str(e)
            return validation_results

    def _check_data_completeness(
        self, epic: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check for missing candles in the dataset"""

        timeframe_minutes = self._timeframe_to_minutes(timeframe)

        # Count expected vs actual candles
        query = """
        WITH expected_periods AS (
            SELECT generate_series(
                date_trunc('minute', :start_date_1),
                date_trunc('minute', :end_date_1),
                interval :timeframe_minutes_1 || ' minutes'
            ) AS expected_time
        ),
        actual_candles AS (
            SELECT start_time, quality_score
            FROM ig_candles
            WHERE epic = :epic
              AND timeframe = :timeframe_minutes_2
              AND start_time >= :start_date_2
              AND start_time <= :end_date_2
        )
        SELECT
            COUNT(e.expected_time) as expected_candles,
            COUNT(a.start_time) as actual_candles,
            COUNT(e.expected_time) - COUNT(a.start_time) as missing_candles,
            COALESCE(AVG(a.quality_score), 0) as avg_quality_score
        FROM expected_periods e
        LEFT JOIN actual_candles a ON e.expected_time = a.start_time
        """

        params = {
            'start_date_1': start_date,
            'end_date_1': end_date,
            'timeframe_minutes_1': timeframe_minutes,
            'epic': epic,
            'timeframe_minutes_2': timeframe_minutes,
            'start_date_2': start_date,
            'end_date_2': end_date
        }

        result = self.db_manager.execute_query(query, params).fetchone()

        if result:
            expected = result['expected_candles']
            actual = result['actual_candles']
            completeness_score = actual / max(expected, 1)

            return {
                'expected_candles': expected,
                'actual_candles': actual,
                'missing_candles': result['missing_candles'],
                'completeness_score': completeness_score,
                'avg_quality_score': float(result['avg_quality_score'])
            }
        else:
            return {
                'expected_candles': 0,
                'actual_candles': 0,
                'missing_candles': 0,
                'completeness_score': 0.0,
                'avg_quality_score': 0.0
            }

    def _validate_price_data(
        self, epic: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Validate price data for anomalies"""

        query = """
        SELECT start_time, open_price_mid as open, high_price_mid as high,
               low_price_mid as low, close_price_mid as close,
               -- Price relationship checks
               CASE
                   WHEN high_price_mid < low_price_mid THEN 'high_less_than_low'
                   WHEN high_price_mid < open_price_mid OR high_price_mid < close_price_mid THEN 'high_invalid'
                   WHEN low_price_mid > open_price_mid OR low_price_mid > close_price_mid THEN 'low_invalid'
                   ELSE 'valid'
               END as price_consistency,

               -- Large gap detection (> 3% change)
               CASE
                   WHEN ABS((close_price_mid - LAG(close_price_mid) OVER (ORDER BY start_time))
                           / NULLIF(LAG(close_price_mid) OVER (ORDER BY start_time), 0)) > 0.03
                   THEN 'large_gap'
                   ELSE 'normal'
               END as gap_check,

               -- Extremely narrow spreads (potential data error)
               CASE
                   WHEN (high_price_mid - low_price_mid) / NULLIF(close_price_mid, 0) < 0.0001
                   THEN 'narrow_range'
                   ELSE 'normal'
               END as range_check

        FROM ig_candles
        WHERE epic = :epic
          AND timeframe = :timeframe_minutes
          AND start_time >= :start_date
          AND start_time <= :end_date
        ORDER BY start_time
        """

        params = {
            'epic': epic,
            'timeframe_minutes': self._timeframe_to_minutes(timeframe),
            'start_date': start_date,
            'end_date': end_date
        }

        results = self.db_manager.execute_query(query, params).fetchall()

        anomalies = []
        for row in results:
            if row['price_consistency'] != 'valid':
                anomalies.append({
                    'timestamp': row['start_time'],
                    'type': 'price_inconsistency',
                    'details': row['price_consistency'],
                    'ohlc': [row['open'], row['high'], row['low'], row['close']]
                })

            if row['gap_check'] == 'large_gap':
                anomalies.append({
                    'timestamp': row['start_time'],
                    'type': 'large_price_gap',
                    'details': 'Price gap >3%'
                })

            if row['range_check'] == 'narrow_range':
                anomalies.append({
                    'timestamp': row['start_time'],
                    'type': 'narrow_range',
                    'details': 'Unusually narrow high-low range'
                })

        return {'anomalies': anomalies}

    def _validate_volume_data(
        self, epic: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Validate volume data for anomalies"""

        query = """
        SELECT start_time, volume,
               CASE
                   WHEN volume = 0 THEN 'zero_volume'
                   WHEN volume > (
                       SELECT AVG(volume) * 10
                       FROM ig_candles sub
                       WHERE sub.epic = ig_candles.epic
                         AND sub.timeframe = ig_candles.timeframe
                         AND sub.start_time >= :sub_start_date
                         AND sub.start_time <= :sub_end_date
                   ) THEN 'extremely_high_volume'
                   ELSE 'normal'
               END as volume_check
        FROM ig_candles
        WHERE epic = :epic
          AND timeframe = :timeframe_minutes
          AND start_time >= :start_date
          AND start_time <= :end_date
        ORDER BY start_time
        """

        params = {
            'sub_start_date': start_date,
            'sub_end_date': end_date,
            'epic': epic,
            'timeframe_minutes': self._timeframe_to_minutes(timeframe),
            'start_date': start_date,
            'end_date': end_date
        }

        results = self.db_manager.execute_query(query, params).fetchall()

        anomalies = []
        for row in results:
            if row['volume_check'] != 'normal':
                anomalies.append({
                    'timestamp': row['start_time'],
                    'type': 'volume_anomaly',
                    'details': row['volume_check'],
                    'volume': row['volume']
                })

        return {'anomalies': anomalies}

    def _calculate_quality_score(self, validation_results: Dict) -> float:
        """Calculate overall data quality score"""

        # Base score from completeness
        quality_score = validation_results['completeness_score'] * 0.6

        # Penalty for anomalies
        price_anomaly_penalty = min(len(validation_results['price_anomalies']) * 0.05, 0.3)
        volume_anomaly_penalty = min(len(validation_results['volume_anomalies']) * 0.02, 0.1)

        quality_score = max(0, quality_score - price_anomaly_penalty - volume_anomaly_penalty)

        return quality_score

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            return 15  # Default


class BacktestDataFetcher(DataFetcher):
    """Enhanced DataFetcher optimized for backtest scenarios"""

    def __init__(
        self,
        db_manager: DatabaseManager,
        user_timezone: str = 'UTC',
        start_date: datetime = None,
        end_date: datetime = None,
        epics: List[str] = None
    ):
        """
        Initialize BacktestDataFetcher with optional period-based cache loading.

        Args:
            db_manager: Database manager instance
            user_timezone: User's timezone string
            start_date: Optional backtest start date (for optimized cache loading)
            end_date: Optional backtest end date (for optimized cache loading)
            epics: Optional list of epics to load (for even faster loading)

        If start_date and end_date are provided, only data for that period
        (plus lookback for indicators) will be loaded into cache - MUCH faster
        than loading all historical data.
        """
        super().__init__(db_manager, user_timezone)

        self.backtest_mode = True
        self.batch_size = 50000  # Larger batches for backtest
        self.current_backtest_time = None
        # CRITICAL FIX (Jan 2026): Store backtest date range for proper cache population
        # The _resampled_cache must contain data for the ENTIRE backtest period
        self.backtest_start_date = start_date
        self.backtest_end_date = end_date
        self.data_validator = BacktestDataValidator(db_manager)

        # CRITICAL FIX: Disable reduced_lookback for backtesting
        # In backtest mode, we need full historical data access, not recent data only
        self.reduced_lookback = False
        self.logger.info("✅ BacktestDataFetcher: reduced_lookback DISABLED for full historical access")

        # Enhanced caching for backtest
        self._backtest_cache = {}
        self._validation_cache = {}

        # PERFORMANCE FIX: Cache resampled timeframes to avoid re-resampling on every iteration
        # Key format: f"{epic}_{timeframe}" e.g., "CS.D.EURUSD.CEEM.IP_1h"
        self._resampled_cache = {}
        self._resampled_cache_hits = 0
        self._resampled_cache_misses = 0

        # Initialize in-memory cache for ultra-fast data access
        self.memory_cache = get_forex_cache(db_manager)
        if self.memory_cache is None:
            self.memory_cache = initialize_cache(db_manager, auto_load=False)  # Don't auto-load

        # PERFORMANCE OPTIMIZATION: Load only data for backtest period if dates provided
        if start_date and end_date:
            self.logger.info(f"⚡ Loading cache for backtest period only: {start_date.date()} to {end_date.date()}")
            self.memory_cache.load_data_for_period(
                start_date=start_date,
                end_date=end_date,
                epics=epics,
                lookback_hours=168,  # 7 days for indicator warmup
                force_reload=True
            )
        else:
            # Fallback to loading all data (slower, but works for unknown periods)
            if not self.memory_cache.is_loaded:
                self.memory_cache.load_all_data()

        # Check cache status
        cache_stats = self.memory_cache.get_cache_stats()
        if cache_stats['loaded']:
            self.logger.info("🧪 BacktestDataFetcher initialized with enhanced historical data support")
            self.logger.info(f"🚀 In-memory cache loaded: {cache_stats['memory_usage_mb']:.1f}MB, "
                           f"{cache_stats['total_rows']:,} rows, "
                           f"{cache_stats['epics_cached']} epics")
        else:
            self.logger.warning("⚠️ In-memory cache failed to load, using database fallback")

    def set_backtest_time(self, current_time: datetime, end_date: datetime = None, start_date: datetime = None):
        """Set the current backtest time for data filtering

        Args:
            current_time: The current timestamp in the backtest simulation
            end_date: The END of the entire backtest period (for cache population)
                     This is CRITICAL - cache must contain data up to end_date,
                     not just current_time, so future iterations can filter correctly
            start_date: The START of the entire backtest period (for cache population)
                       This ensures we fetch data covering the entire simulation range
        """
        self.current_backtest_time = current_time

        # CRITICAL FIX (Jan 2026): Store backtest date range for proper cache population
        # Cache must contain data for the ENTIRE backtest period, from start_date to end_date
        if start_date is not None:
            self.backtest_start_date = start_date
        elif self.backtest_start_date is None:
            # First call without start_date - use current_time but log warning
            self.backtest_start_date = current_time
            self.logger.warning("⚠️ set_backtest_time called without start_date - cache may not cover full period")

        if end_date is not None:
            self.backtest_end_date = end_date
        elif self.backtest_end_date is None:
            # First call without end_date - use current_time but log warning
            self.backtest_end_date = current_time
            self.logger.warning("⚠️ set_backtest_time called without end_date - cache may not contain future data")

        # CRITICAL FIX: Override timezone_manager's get_lookback_time_utc to use backtest time
        # This ensures data is fetched relative to the backtest timestamp, not real current time
        # Only set up the override once to avoid nested closures
        if not hasattr(self, '_backtest_lookback_override_set'):
            original_get_lookback = self.timezone_manager.get_lookback_time_utc

            def backtest_aware_get_lookback(hours_back: int) -> datetime:
                """Calculate lookback from backtest time instead of real current time"""
                if self.current_backtest_time is not None:
                    # Use backtest time as reference point
                    from pytz import UTC
                    backtest_utc = self.current_backtest_time
                    if backtest_utc.tzinfo is None:
                        backtest_utc = UTC.localize(backtest_utc)
                    return backtest_utc - timedelta(hours=hours_back)
                else:
                    # Fallback to original behavior
                    return original_get_lookback(hours_back)

            self.timezone_manager.get_lookback_time_utc = backtest_aware_get_lookback
            self._backtest_lookback_override_set = True

    def get_resampled_cache_stats(self) -> dict:
        """Get statistics about resampled data cache performance"""
        total_requests = self._resampled_cache_hits + self._resampled_cache_misses
        hit_rate = (self._resampled_cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self._resampled_cache_hits,
            'cache_misses': self._resampled_cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cached_datasets': len(self._resampled_cache),
            'cache_keys': list(self._resampled_cache.keys())
        }

    async def get_historical_data_batch(
        self,
        epics: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        validate_quality: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Optimized batch retrieval for multiple epics with prefetching"""

        cache_key = f"batch_{hash((tuple(epics), timeframe, start_date, end_date))}"

        if cache_key in self._backtest_cache:
            self.logger.debug(f"📦 Cache hit for batch data: {len(epics)} epics")
            return self._backtest_cache[cache_key]

        try:
            # Construct optimized query with named parameters for multiple epics
            epic_clauses = [f"epic = :epic_{i}" for i in range(len(epics))]
            epic_filter = " OR ".join(epic_clauses)

            query = f"""
            SELECT epic, start_time,
                   open_price_mid as open, high_price_mid as high,
                   low_price_mid as low, close_price_mid as close,
                   volume, ltv, quality_score
            FROM ig_candles
            WHERE ({epic_filter})
              AND timeframe = :timeframe
              AND start_time >= :start_date
              AND start_time <= :end_date
            ORDER BY epic, start_time ASC
            """

            # Build parameters dictionary: epics + timeframe + dates
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            params = {
                'timeframe': timeframe_minutes,
                'start_date': start_date,
                'end_date': end_date
            }

            # Add epic parameters
            for i, epic in enumerate(epics):
                params[f'epic_{i}'] = epic

            # Execute query
            result = self.db_manager.execute_query(query, params).fetchall()

            # Split results by epic
            epic_data = {}
            for epic in epics:
                epic_rows = [
                    {
                        'start_time': row['start_time'],
                        'open': float(row['open']) if row['open'] else 0,
                        'high': float(row['high']) if row['high'] else 0,
                        'low': float(row['low']) if row['low'] else 0,
                        'close': float(row['close']) if row['close'] else 0,
                        'volume': int(row['volume']) if row['volume'] else 0,
                        'ltv': float(row['ltv']) if row['ltv'] else 0,
                        'quality_score': float(row['quality_score']) if row['quality_score'] else 1.0
                    }
                    for row in result if row['epic'] == epic
                ]

                if epic_rows:
                    df = pd.DataFrame(epic_rows)
                    df = df.set_index('start_time')
                    df.index = pd.to_datetime(df.index, utc=True)

                    # Add timezone columns
                    df = add_timezone_columns(df, self.user_timezone)

                    # Add technical indicators
                    df = self._add_technical_indicators(df)

                    epic_data[epic] = df
                else:
                    epic_data[epic] = pd.DataFrame()
                    self.logger.warning(f"⚠️ No data found for {epic} in period {start_date} to {end_date}")

            # Cache the results
            self._backtest_cache[cache_key] = epic_data

            # Validate data quality if requested
            if validate_quality:
                await self._validate_batch_quality(epics, timeframe, start_date, end_date)

            self.logger.info(f"📊 Loaded batch data: {len(epics)} epics, "
                           f"{sum(len(df) for df in epic_data.values())} total candles")

            return epic_data

        except Exception as e:
            self.logger.error(f"Error loading batch historical data: {e}")
            # Return empty DataFrames for all epics
            return {epic: pd.DataFrame() for epic in epics}

    def get_enhanced_data(self, epic: str, pair: str, timeframe: str = '5m', **kwargs) -> pd.DataFrame:
        """
        Override parent method to provide time-aware data for backtesting

        In backtest mode, this method respects the current_backtest_time to provide
        only data up to the current simulation timestamp, ensuring realistic backtesting.

        CRITICAL FIX (Jan 2026): The _resampled_cache now stores data for the ENTIRE
        backtest period (up to backtest_end_date), then filters to current_backtest_time.
        Previously, cache stored only data up to the first call's current_backtest_time,
        causing subsequent iterations to see no new data.
        """
        try:
            # If we're in backtest mode and have a current timestamp, filter data accordingly
            if hasattr(self, 'current_backtest_time') and self.current_backtest_time:
                # PERFORMANCE FIX: Check resampled cache first to avoid expensive re-resampling
                cache_key = f"{epic}_{timeframe}"

                if cache_key in self._resampled_cache:
                    # Cache HIT - reuse previously resampled data
                    full_df = self._resampled_cache[cache_key]
                    self._resampled_cache_hits += 1
                    self.logger.debug(f"⚡ CACHE HIT: Reused {timeframe} data for {epic} (hits: {self._resampled_cache_hits})")
                else:
                    # Cache MISS - need to fetch data for ENTIRE backtest period
                    # CRITICAL FIX (Jan 2026): Calculate lookback to cover from backtest_start_date to backtest_end_date
                    # The lookback must be calculated from backtest_end_date back to backtest_start_date (plus extra for indicators)

                    # Save current backtest time
                    original_backtest_time = self.current_backtest_time

                    # Calculate extended lookback to cover entire backtest period
                    extended_lookback_hours = kwargs.get('lookback_hours')
                    if hasattr(self, 'backtest_start_date') and self.backtest_start_date and \
                       hasattr(self, 'backtest_end_date') and self.backtest_end_date:
                        # Calculate hours from backtest_end_date back to backtest_start_date
                        backtest_duration_hours = (self.backtest_end_date - self.backtest_start_date).total_seconds() / 3600
                        # Add the original lookback_hours on top (for indicator warmup at backtest_start_date)
                        original_lookback = kwargs.get('lookback_hours', 24)
                        extended_lookback_hours = int(backtest_duration_hours + original_lookback + 24)  # +24h buffer

                        self.logger.debug(f"📊 Cache population: extended lookback from {original_lookback}h to {extended_lookback_hours}h "
                                         f"(backtest duration: {backtest_duration_hours:.1f}h)")

                        # Temporarily set current_backtest_time to end_date for proper end point
                        self.current_backtest_time = self.backtest_end_date

                    try:
                        # Pass extended lookback if calculated
                        cache_kwargs = kwargs.copy()
                        if extended_lookback_hours:
                            cache_kwargs['lookback_hours'] = extended_lookback_hours
                        full_df = super().get_enhanced_data(epic, pair, timeframe, **cache_kwargs)
                    finally:
                        # Restore original backtest time
                        self.current_backtest_time = original_backtest_time

                    # Store in cache for future iterations (only if valid data)
                    if full_df is not None and len(full_df) > 0:
                        self._resampled_cache[cache_key] = full_df.copy()
                        self._resampled_cache_misses += 1
                        max_time = full_df['start_time'].max() if 'start_time' in full_df.columns else 'unknown'
                        self.logger.debug(f"💾 CACHE MISS: Stored {timeframe} data for {epic}, {len(full_df)} rows up to {max_time}")
                    else:
                        full_df = None

                if full_df is None or len(full_df) == 0:
                    return full_df

                # Filter data up to current backtest time
                # Ensure timestamp column exists and is properly formatted
                if 'start_time' in full_df.columns:
                    # Convert current_backtest_time to timestamp with proper timezone handling
                    cutoff_time = pd.Timestamp(self.current_backtest_time)

                    # Handle timezone mismatch: ensure both timestamps are comparable
                    start_time_col = full_df['start_time']
                    if hasattr(start_time_col.dtype, 'tz') and start_time_col.dtype.tz is not None:
                        # DataFrame has timezone-aware timestamps, make cutoff_time timezone-aware
                        if cutoff_time.tz is None:
                            # Assume cutoff_time is in UTC if no timezone specified
                            cutoff_time = cutoff_time.tz_localize('UTC')
                        else:
                            # Convert to same timezone as DataFrame
                            cutoff_time = cutoff_time.tz_convert(start_time_col.dtype.tz)
                    else:
                        # DataFrame has timezone-naive timestamps, make cutoff_time timezone-naive
                        if cutoff_time.tz is not None:
                            cutoff_time = cutoff_time.tz_localize(None)

                    filtered_df = full_df[full_df['start_time'] <= cutoff_time].copy()

                    if len(filtered_df) < len(full_df):
                        self.logger.debug(f"🕒 BACKTEST: Filtered {epic} data from {len(full_df)} to {len(filtered_df)} rows (cutoff: {cutoff_time})")

                    return filtered_df
                else:
                    self.logger.warning(f"⚠️ No 'start_time' column found in {epic} data, returning unfiltered data")
                    return full_df
            else:
                # No current timestamp set, return full data (fallback behavior)
                return super().get_enhanced_data(epic, pair, timeframe, **kwargs)

        except Exception as e:
            self.logger.error(f"Error in backtest get_enhanced_data for {epic}: {e}")
            # Fallback to parent method
            return super().get_enhanced_data(epic, pair, timeframe, **kwargs)

    def get_enhanced_data_for_backtest(
        self,
        epic: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        current_time: datetime = None,
        strategy_config: Dict[str, Any] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get enhanced data specifically optimized for backtesting"""

        # Use current_time to limit data (simulating real-time progression)
        effective_end_date = min(end_date, current_time) if current_time else end_date

        try:
            # Validate data quality first if not cached
            validation_key = f"validation_{epic}_{timeframe}_{start_date}_{end_date}"

            if validation_key in self._validation_cache:
                validation_report = self._validation_cache[validation_key]
            else:
                validation_report = self.data_validator.validate_historical_data(
                    epic, timeframe, start_date, effective_end_date
                )
                self._validation_cache[validation_key] = validation_report

            if not validation_report['validation_passed']:
                self.logger.warning(
                    f"⚠️ Data quality issues for {epic} {timeframe}: "
                    f"Completeness {validation_report['completeness_score']:.2%}, "
                    f"Quality {validation_report['quality_score']:.2%}"
                )

            # Get historical data
            df = self._get_historical_data_optimized(
                epic, timeframe, start_date, effective_end_date
            )

            if df.empty:
                return df, validation_report

            # Add strategy-specific indicators if provided
            if strategy_config:
                df = self._add_strategy_specific_indicators(df, strategy_config)

            # Add backtest metadata
            df['data_quality_score'] = validation_report['quality_score']
            df['backtest_mode'] = True

            return df, validation_report

        except Exception as e:
            self.logger.error(f"Error getting enhanced backtest data: {e}")
            return pd.DataFrame(), {'validation_passed': False, 'error': str(e)}

    def _get_historical_data_optimized(
        self, epic: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data with ultra-fast in-memory cache first, database fallback"""

        cache_key = f"hist_{epic}_{timeframe}_{start_date}_{end_date}"
        if cache_key in self._backtest_cache:
            return self._backtest_cache[cache_key]

        try:
            # 🚀 FAST PATH: Try in-memory cache first
            timeframe_minutes = self._timeframe_to_minutes(timeframe)

            if self.memory_cache and self.memory_cache.is_loaded:
                df = self.memory_cache.get_historical_data(
                    epic, timeframe_minutes, start_date, end_date
                )

                if df is not None and not df.empty:
                    # Add timezone columns
                    df = add_timezone_columns(df, self.user_timezone)

                    # Add technical indicators
                    df = self._add_technical_indicators(df)

                    # Cache the result for future requests
                    self._backtest_cache[cache_key] = df

                    self.logger.debug(f"⚡ Memory cache hit: {epic} {timeframe}, {len(df)} rows")
                    return df

            # 🐌 SLOW PATH: Database fallback
            self.logger.debug(f"💾 Database fallback for {epic} {timeframe}")

            query = """
            SELECT start_time,
                   open_price_mid as open, high_price_mid as high,
                   low_price_mid as low, close_price_mid as close,
                   volume, ltv, quality_score
            FROM ig_candles
            WHERE epic = :epic
              AND timeframe = :timeframe_minutes
              AND start_time >= :start_date
              AND start_time <= :end_date
            ORDER BY start_time ASC
            """

            params = {
                'epic': epic,
                'timeframe_minutes': timeframe_minutes,
                'start_date': start_date,
                'end_date': end_date
            }
            result = self.db_manager.execute_query(query, params)

            if result.empty:
                return pd.DataFrame()

            # Process database result
            df = result.copy()

            # Rename columns to match expected format
            column_mapping = {
                'open_price_mid': 'open',
                'high_price_mid': 'high',
                'low_price_mid': 'low',
                'close_price_mid': 'close'
            }
            df = df.rename(columns=column_mapping)

            # Set index and ensure datetime
            df = df.set_index('start_time')
            df.index = pd.to_datetime(df.index, utc=True)

            # Add timezone columns
            df = add_timezone_columns(df, self.user_timezone)

            # Add technical indicators
            df = self._add_technical_indicators(df)

            # Cache the result
            self._backtest_cache[cache_key] = df

            return df

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def _fetch_candle_data_optimized(
        self,
        epic: str,
        timeframe: str,
        lookback_hours: int,
        tz_manager
    ) -> Optional[pd.DataFrame]:
        """
        Override parent method to use in-memory cache for backtest mode.

        This is CRITICAL for date range backtesting - the parent class queries
        the database using lookback_hours from current time, but for backtests
        we need data from the backtest time period (which is in the memory cache).
        """
        try:
            # Calculate the lookback time using the (potentially overridden) tz_manager method
            since_utc = tz_manager.get_lookback_time_utc(lookback_hours)

            # Determine end time: use backtest time if set, otherwise current time
            if hasattr(self, 'current_backtest_time') and self.current_backtest_time:
                from pytz import UTC
                end_utc = self.current_backtest_time
                if end_utc.tzinfo is None:
                    end_utc = UTC.localize(end_utc)
            else:
                end_utc = tz_manager.get_current_utc_time()

            self.logger.debug(f"📊 Backtest fetch: {epic} {timeframe}, {since_utc} to {end_utc}")

            # Convert timeframe to minutes
            timeframe_map = {
                '5m': 5,
                '15m': 15,
                '1h': 60,
                '4h': 240
            }
            tf_minutes = timeframe_map.get(timeframe, 5)

            # Determine source timeframe for resampling
            # OPTIMIZATION: Use highest available timeframe from cache to minimize resampling
            # Check if target timeframe exists directly in cache first
            source_tf = tf_minutes  # Default: use target timeframe directly if available
            needs_resampling = False

            if self.memory_cache and self.memory_cache.is_loaded:
                # Check what timeframes are available in cache for this epic
                available_tfs = []
                if epic in self.memory_cache.cache:
                    available_tfs = list(self.memory_cache.cache[epic].keys())

                if tf_minutes in available_tfs:
                    # Target timeframe exists directly - no resampling needed!
                    source_tf = tf_minutes
                    needs_resampling = False
                    self.logger.debug(f"⚡ Using {timeframe} directly from cache (no resampling)")
                elif timeframe == '4h' and 60 in available_tfs:
                    # 4h not available but 1h is - resample from 1h (4x faster than from 1m)
                    source_tf = 60
                    needs_resampling = True
                    self.logger.debug(f"📊 Using 1h base for {timeframe} (4x resample)")
                elif timeframe in ('15m', '1h', '4h') and 5 in available_tfs:
                    # Use 5m as base (faster than 1m)
                    source_tf = 5
                    needs_resampling = True
                    self.logger.debug(f"📊 Using 5m base for {timeframe}")
                elif 1 in available_tfs:
                    # Fallback to 1m
                    source_tf = 1
                    needs_resampling = True
                    self.logger.debug(f"📊 Using 1m base for {timeframe} (slowest)")
                else:
                    self.logger.warning(f"⚠️ No suitable source timeframe found for {timeframe}")

            # Adjust lookback if resampling is needed
            if needs_resampling:
                adjusted_lookback = int(lookback_hours * 1.2)
                since_utc = tz_manager.get_lookback_time_utc(adjusted_lookback)

            # 🚀 FAST PATH: Try in-memory cache first
            df = None
            if self.memory_cache and self.memory_cache.is_loaded:
                # CRITICAL: Memory cache uses timezone-naive datetimes
                # Convert our UTC timestamps to naive for comparison
                since_naive = since_utc.replace(tzinfo=None) if since_utc.tzinfo else since_utc
                end_naive = end_utc.replace(tzinfo=None) if end_utc.tzinfo else end_utc

                df = self.memory_cache.get_historical_data(
                    epic, source_tf, since_naive, end_naive
                )

                if df is not None and not df.empty:
                    self.logger.debug(f"⚡ Memory cache hit: {epic} {timeframe}, {len(df)} source rows")
                else:
                    self.logger.debug(f"⚠️ Memory cache miss for {epic} {source_tf}m, {since_naive} to {end_naive}")

            # 🐌 SLOW PATH: Database fallback
            if df is None or df.empty:
                self.logger.debug(f"💾 Database fallback for {epic} {timeframe}")

                query = """
                SELECT start_time,
                       open, high, low, close,
                       volume, ltv
                FROM ig_candles
                WHERE epic = :epic
                  AND timeframe = :source_tf
                  AND start_time >= :since_utc
                  AND start_time <= :end_utc
                ORDER BY start_time ASC
                """

                params = {
                    'epic': epic,
                    'source_tf': source_tf,
                    'since_utc': since_utc,
                    'end_utc': end_utc
                }

                result = self.db_manager.execute_query(query, params)

                if result is None or (hasattr(result, 'empty') and result.empty):
                    self.logger.warning(f"⚠️ No data from database for {epic} {timeframe}")
                    return None

                # Convert to DataFrame if needed
                if hasattr(result, 'fetchall'):
                    rows = result.fetchall()
                    if not rows:
                        return None
                    df = pd.DataFrame(rows)
                else:
                    df = result.copy()

            if df is None or len(df) == 0:
                self.logger.warning(f"⚠️ No data available for {epic} {timeframe}")
                return None

            # Ensure proper column format
            if 'start_time' not in df.columns and df.index.name == 'start_time':
                df = df.reset_index()

            # Convert timestamp
            df['start_time'] = pd.to_datetime(df['start_time'], utc=True)

            # Add timezone columns
            df = tz_manager.add_timezone_columns_to_df(df)

            # Resample from 1m base candles to target timeframe
            if source_tf == 1:
                if timeframe == '5m':
                    self.logger.debug(f"🔄 Resampling 1m→5m for {epic}")
                    df = self._resample_to_5m_from_1m(df)
                elif timeframe == '15m':
                    self.logger.debug(f"🔄 Resampling 1m→15m for {epic}")
                    df = self._resample_to_15m_from_1m(df)
                elif timeframe == '1h':
                    self.logger.debug(f"🔄 Resampling 1m→1h for {epic}")
                    df = self._resample_to_60m_from_1m(df)
                elif timeframe == '4h':
                    self.logger.debug(f"🔄 Resampling 1m→4h for {epic}")
                    df = self._resample_to_4h_from_1m(df)

                if df is None or len(df) == 0:
                    self.logger.error(f"❌ Resampling failed for {epic} {timeframe}")
                    return None

            self.logger.debug(f"✅ Fetched {len(df)} bars for {epic} {timeframe}")
            return df.reset_index(drop=True)

        except Exception as e:
            self.logger.error(f"❌ Error in backtest _fetch_candle_data_optimized: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _add_strategy_specific_indicators(self, df: pd.DataFrame, strategy_config: Dict[str, Any]) -> pd.DataFrame:
        """Add strategy-specific technical indicators"""

        try:
            # Add indicators based on strategy requirements
            if 'ema_periods' in strategy_config:
                for period in strategy_config['ema_periods']:
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            if 'macd_config' in strategy_config:
                macd_config = strategy_config['macd_config']
                fast = macd_config.get('fast', 12)
                slow = macd_config.get('slow', 26)
                signal = macd_config.get('signal', 9)

                ema_fast = df['close'].ewm(span=fast).mean()
                ema_slow = df['close'].ewm(span=slow).mean()
                df['macd_line'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd_line'].ewm(span=signal).mean()
                df['macd_histogram'] = df['macd_line'] - df['macd_signal']

            if 'rsi_period' in strategy_config:
                period = strategy_config['rsi_period']
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))

        except Exception as e:
            self.logger.error(f"Error adding strategy-specific indicators: {e}")

        return df

    async def _validate_batch_quality(self, epics: List[str], timeframe: str,
                                    start_date: datetime, end_date: datetime):
        """Validate data quality for batch of epics"""

        validation_tasks = []
        for epic in epics:
            validation_key = f"validation_{epic}_{timeframe}_{start_date}_{end_date}"
            if validation_key not in self._validation_cache:
                # Run validation asynchronously
                task = asyncio.create_task(
                    self._async_validate_data(epic, timeframe, start_date, end_date, validation_key)
                )
                validation_tasks.append(task)

        if validation_tasks:
            await asyncio.gather(*validation_tasks, return_exceptions=True)

    async def _async_validate_data(self, epic: str, timeframe: str,
                                 start_date: datetime, end_date: datetime, cache_key: str):
        """Async data validation"""
        try:
            validation_report = self.data_validator.validate_historical_data(
                epic, timeframe, start_date, end_date
            )
            self._validation_cache[cache_key] = validation_report
        except Exception as e:
            self.logger.error(f"Error in async data validation for {epic}: {e}")

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            return 15  # Default

    def clear_backtest_cache(self):
        """Clear backtest-specific caches"""
        self._backtest_cache.clear()
        self._validation_cache.clear()
        self.logger.info("🧹 Backtest cache cleared")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get backtest cache statistics"""
        return {
            'backtest_cache_size': len(self._backtest_cache),
            'validation_cache_size': len(self._validation_cache),
            'cache_memory_estimate_mb': (
                len(str(self._backtest_cache)) + len(str(self._validation_cache))
            ) / 1024 / 1024
        }