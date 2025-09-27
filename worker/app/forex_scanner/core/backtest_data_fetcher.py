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

    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'UTC'):
        super().__init__(db_manager, user_timezone)

        self.backtest_mode = True
        self.batch_size = 50000  # Larger batches for backtest
        self.current_backtest_time = None
        self.data_validator = BacktestDataValidator(db_manager)

        # Enhanced caching for backtest
        self._backtest_cache = {}
        self._validation_cache = {}

        # Initialize in-memory cache for ultra-fast data access
        self.memory_cache = get_forex_cache(db_manager)
        if self.memory_cache is None:
            self.memory_cache = initialize_cache(db_manager, auto_load=True)
        else:
            # Ensure cache is loaded
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

    def set_backtest_time(self, current_time: datetime):
        """Set the current backtest time for data filtering"""
        self.current_backtest_time = current_time

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