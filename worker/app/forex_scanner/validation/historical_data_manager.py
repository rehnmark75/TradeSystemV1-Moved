# validation/historical_data_manager.py
"""
Historical Data Manager for Signal Validation

This module handles the retrieval and management of historical market data
for signal replay and validation. It provides methods to fetch candle data,
recreate market conditions, and prepare data for strategy analysis.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Any
from sqlalchemy import text
from functools import lru_cache

from forex_scanner.core.database import DatabaseManager
from forex_scanner.core.data_fetcher import DataFetcher
from forex_scanner.analysis.technical import TechnicalAnalyzer
from forex_scanner.utils.timezone_utils import TimezoneManager
from .replay_config import ReplayConfig, TIMEFRAME_MINUTES


class HistoricalDataManager:
    """
    Manages historical data retrieval and preparation for signal validation
    
    This class provides methods to:
    - Fetch historical candle data around specific timestamps
    - Recreate market conditions as they existed at validation time
    - Calculate technical indicators using historical data
    - Prepare data in the exact format expected by strategies
    """
    
    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'Europe/Stockholm'):
        """
        Initialize the historical data manager
        
        Args:
            db_manager: Database manager for data access
            user_timezone: User's timezone for timestamp handling
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.timezone_manager = TimezoneManager(user_timezone)
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize data fetcher for technical analysis
        self.data_fetcher = DataFetcher(db_manager, user_timezone)
        
        # Cache for recently fetched data
        self._data_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        self.logger.info(f"🕒 HistoricalDataManager initialized")
        self.logger.info(f"   Timezone: {user_timezone}")
        self.logger.info(f"   Cache enabled: {ReplayConfig.is_feature_enabled('enable_data_caching')}")
    
    def get_historical_candles(
        self, 
        epic: str, 
        target_timestamp: datetime, 
        timeframe: str = '15m',
        lookback_bars: int = None,
        strategy: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data for the specified epic and timestamp
        
        Args:
            epic: Epic code (e.g., 'CS.D.EURUSD.MINI.IP')
            target_timestamp: Target timestamp for validation
            timeframe: Timeframe ('5m', '15m', '1h')
            lookback_bars: Number of bars to fetch before target (auto-calculated if None)
            strategy: Strategy name for optimization (optional)
            
        Returns:
            DataFrame with historical candle data or None if error
        """
        try:
            # Ensure target timestamp has timezone info
            if target_timestamp.tzinfo is None:
                target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
            
            # Calculate lookback requirements
            if lookback_bars is None:
                lookback_bars = ReplayConfig.get_lookback_bars(strategy, timeframe)
            
            self.logger.info(f"📊 Fetching historical data for {epic}")
            self.logger.info(f"   Target: {target_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Lookback bars: {lookback_bars}")
            
            # Check cache first
            cache_key = self._get_cache_key(epic, target_timestamp, timeframe, lookback_bars)
            if ReplayConfig.is_feature_enabled('enable_data_caching'):
                cached_data = self._get_cached_data(cache_key)
                if cached_data is not None:
                    self.logger.debug(f"📦 Using cached data for {epic}")
                    return cached_data
            
            # Calculate time window
            timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
            lookback_duration = timedelta(minutes=lookback_bars * timeframe_minutes * 1.5)  # 50% buffer
            start_time = target_timestamp - lookback_duration
            end_time = target_timestamp + timedelta(minutes=timeframe_minutes)  # Include target bar
            
            # Fetch data from database
            df = self._fetch_candles_from_db(epic, start_time, end_time, timeframe_minutes)
            
            if df is None or len(df) == 0:
                self.logger.warning(f"⚠️ No historical data found for {epic}")
                return None
            
            # Filter to exact timestamp requirement
            df = df[df['start_time'] <= target_timestamp].copy()
            
            if len(df) < ReplayConfig.get_strategy_config(strategy).get('min_bars_required', 100):
                self.logger.warning(f"⚠️ Insufficient data: {len(df)} bars (need at least {ReplayConfig.get_strategy_config(strategy).get('min_bars_required', 100)})")
                return None
            
            # Sort by timestamp to ensure proper order
            df = df.sort_values('start_time').reset_index(drop=True)
            
            # Add timezone columns
            df = self._add_timezone_columns(df)
            
            # Cache the result
            if ReplayConfig.is_feature_enabled('enable_data_caching'):
                self._cache_data(cache_key, df)
            
            self.logger.info(f"✅ Historical data fetched: {len(df)} bars")
            self.logger.info(f"   Date range: {df.iloc[0]['start_time'].strftime('%Y-%m-%d %H:%M')} → {df.iloc[-1]['start_time'].strftime('%Y-%m-%d %H:%M')}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching historical data for {epic}: {e}")
            return None
    
    def get_enhanced_historical_data(
        self,
        epic: str,
        target_timestamp: datetime, 
        timeframe: str = '15m',
        strategy: str = None,
        indicators_needed: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data enhanced with technical indicators
        
        Args:
            epic: Epic code
            target_timestamp: Target timestamp for validation
            timeframe: Timeframe for analysis
            strategy: Strategy name for configuration
            indicators_needed: List of specific indicators to calculate
            
        Returns:
            Enhanced DataFrame with technical indicators
        """
        try:
            # Get base historical data
            df = self.get_historical_candles(epic, target_timestamp, timeframe, strategy=strategy)
            
            if df is None:
                return None
            
            # Get pair information
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            self.logger.info(f"🔧 Enhancing data with technical indicators for {epic}")
            
            # Calculate technical indicators based on strategy requirements
            if strategy:
                strategy_config = ReplayConfig.get_strategy_config(strategy)
                required_indicators = indicators_needed or strategy_config.get('indicators_needed', [])
            else:
                required_indicators = indicators_needed or []
            
            # Always calculate basic indicators
            df = self._calculate_base_indicators(df)
            
            # ARCHITECTURAL FIX: Use the live DataFetcher's enhancement methods
            # This ensures we use exactly the same calculation methods as the live system
            self.logger.info("📊 Using live DataFetcher methods to enhance historical data")
            
            # Use the live data fetcher to enhance the data with the same methods
            df = self._enhance_with_live_methods(df, epic, strategy or 'ema')
            
            if (strategy and 'momentum_bias' in strategy.lower()) or (not strategy and 'momentum_bias' in required_indicators):
                df = self._calculate_momentum_bias_indicators(df)
            
            # Calculate multi-timeframe indicators if needed
            if strategy and ReplayConfig.get_strategy_config(strategy).get('mtf_analysis', False):
                df = self._add_mtf_analysis(df, epic, target_timestamp, timeframe)
            
            # Validate data integrity
            if not self._validate_data_integrity(df, required_indicators):
                self.logger.error(f"❌ Data integrity validation failed for {epic}")
                return None
            
            self.logger.info(f"✅ Enhanced data ready with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error enhancing historical data for {epic}: {e}")
            return None
    
    def get_market_state_at_timestamp(
        self,
        epic: str,
        timestamp: datetime,
        timeframe: str = '15m'
    ) -> Optional[Dict[str, Any]]:
        """
        Get complete market state information at specific timestamp
        
        Args:
            epic: Epic code
            timestamp: Exact timestamp to analyze
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with market state information
        """
        try:
            # Get enhanced data
            df = self.get_enhanced_historical_data(epic, timestamp, timeframe)
            
            if df is None or len(df) == 0:
                return None
            
            # Find the candle at or before the target timestamp
            target_candles = df[df['start_time'] <= timestamp]
            if len(target_candles) == 0:
                return None
            
            current_candle = target_candles.iloc[-1]
            
            # Build market state
            try:
                
                market_state = {
                    'timestamp': timestamp,
                    'candle_timestamp': current_candle['start_time'].to_pydatetime() if hasattr(current_candle['start_time'], 'to_pydatetime') else current_candle['start_time'],
                    'epic': epic,
                    'timeframe': timeframe,
                    
                    # OHLC data with safe conversion
                    'price': {
                        'open': float(current_candle['open']) if pd.notna(current_candle['open']) else 0.0,
                        'high': float(current_candle['high']) if pd.notna(current_candle['high']) else 0.0,
                        'low': float(current_candle['low']) if pd.notna(current_candle['low']) else 0.0,
                        'close': float(current_candle['close']) if pd.notna(current_candle['close']) else 0.0,
                        'volume': int(current_candle['volume']) if 'volume' in current_candle and pd.notna(current_candle['volume']) else 0
                    },
                    
                    # Technical indicators
                    'indicators': {},
                    
                    # Market conditions
                    'conditions': {},
                    
                    # Trend analysis
                    'trend': {}
                }
            except Exception as e:
                self.logger.error(f"❌ Error converting OHLC data: {e}")
                self.logger.error(f"   Candle columns: {list(current_candle.index)}")
                self.logger.error(f"   Candle values: {current_candle.to_dict()}")
                raise
            
            # Add technical indicators to state
            for col in df.columns:
                if col not in ['start_time', 'epic', 'timeframe', 'open', 'high', 'low', 'close', 'volume', 'user_time']:
                    if pd.notna(current_candle[col]):
                        try:
                            market_state['indicators'][col] = float(current_candle[col])
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"⚠️ Skipping indicator {col} due to conversion error: {e}")
                            continue
            
            # Add trend analysis
            if 'ema_21' in current_candle and 'ema_50' in current_candle and 'ema_200' in current_candle:
                price = current_candle['close']
                ema_21 = current_candle['ema_21']
                ema_50 = current_candle['ema_50']
                ema_200 = current_candle['ema_200']
                
                # Determine trend direction
                if ema_21 > ema_50 > ema_200 and price > ema_21:
                    trend_direction = 'BULLISH'
                elif ema_21 < ema_50 < ema_200 and price < ema_21:
                    trend_direction = 'BEARISH'
                else:
                    trend_direction = 'SIDEWAYS'
                
                market_state['trend'] = {
                    'direction': trend_direction,
                    'ema_alignment': ema_21 > ema_50 > ema_200 if trend_direction == 'BULLISH' else ema_21 < ema_50 < ema_200,
                    'price_above_ema21': price > ema_21,
                    'strength': 'STRONG' if abs(ema_21 - ema_50) / ema_50 > 0.001 else 'WEAK'
                }
            
            # Add market conditions
            if 'atr' in current_candle:
                volatility = current_candle['atr'] / current_candle['close'] * 100
                market_state['conditions']['volatility'] = {
                    'atr': float(current_candle['atr']),
                    'volatility_pct': float(volatility),
                    'regime': 'HIGH' if volatility > 1.0 else 'MEDIUM' if volatility > 0.5 else 'LOW'
                }
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market state for {epic}: {e}")
            return None
    
    def _fetch_candles_from_db(
        self, 
        epic: str, 
        start_time: datetime, 
        end_time: datetime, 
        timeframe_minutes: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from database with 5m->15m synthesis support
        
        Since we store 5-minute candles but analyze on 15-minute timeframes,
        this method handles the synthesis for the main data fetching pipeline.
        """
        try:
            self.logger.info(f"📊 Fetching {timeframe_minutes}m candles for {epic}")
            self.logger.info(f"   Time range: {start_time} to {end_time}")
            
            if timeframe_minutes == 5:
                # Direct fetch for 5-minute data
                query = text("""
                    SELECT start_time, epic, open, high, low, close, volume, timeframe
                    FROM ig_candles 
                    WHERE epic = :epic 
                    AND timeframe = 5
                    AND start_time >= :start_time 
                    AND start_time <= :end_time
                    ORDER BY start_time ASC
                """)
                
                with self.db_manager.get_engine().connect() as conn:
                    df = pd.read_sql(
                        query,
                        conn,
                        params={
                            'epic': epic,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                    )
                
            elif timeframe_minutes == 15:
                # Synthesize 15-minute candles from 5-minute data
                self.logger.info("🔄 Synthesizing 15m candles from 5m data for historical analysis")
                
                # Get 5-minute data for the range
                query = text("""
                    SELECT start_time, epic, open, high, low, close, volume
                    FROM ig_candles 
                    WHERE epic = :epic 
                    AND timeframe = 5
                    AND start_time >= :start_time 
                    AND start_time <= :end_time
                    ORDER BY start_time ASC
                """)
                
                with self.db_manager.get_engine().connect() as conn:
                    df_5m = pd.read_sql(
                        query,
                        conn,
                        params={
                            'epic': epic,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                    )
                
                if len(df_5m) > 0:
                    df_5m['start_time'] = pd.to_datetime(df_5m['start_time'], utc=True)
                    # Synthesize 15m candles from the 5m data
                    df = self._synthesize_15m_range_from_5m(df_5m, start_time, end_time)
                    
                    if df is None or len(df) == 0:
                        self.logger.warning(f"⚠️ Failed to synthesize 15m candles from {len(df_5m)} 5m candles")
                        return None
                    else:
                        self.logger.info(f"✅ Synthesized {len(df)} x 15m candles from {len(df_5m)} x 5m candles")
                else:
                    self.logger.warning(f"⚠️ No 5m data found for synthesis")
                    return None
                    
            elif timeframe_minutes == 60:
                # Synthesize hourly candles from 5-minute data
                self.logger.info("🔄 Synthesizing 1h candles from 5m data for historical analysis")
                
                query = text("""
                    SELECT start_time, epic, open, high, low, close, volume
                    FROM ig_candles 
                    WHERE epic = :epic 
                    AND timeframe = 5
                    AND start_time >= :start_time 
                    AND start_time <= :end_time
                    ORDER BY start_time ASC
                """)
                
                with self.db_manager.get_engine().connect() as conn:
                    df_5m = pd.read_sql(
                        query,
                        conn,
                        params={
                            'epic': epic,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                    )
                
                if len(df_5m) > 0:
                    df_5m['start_time'] = pd.to_datetime(df_5m['start_time'], utc=True)
                    df = self._synthesize_hourly_range_from_5m(df_5m, start_time, end_time)
                    
                    if df is None or len(df) == 0:
                        self.logger.warning(f"⚠️ Failed to synthesize hourly candles")
                        return None
                else:
                    return None
                    
            else:
                # Fallback - try to get 5-minute data
                self.logger.warning(f"⚠️ Unsupported timeframe {timeframe_minutes}m, falling back to 5m data")
                return self._fetch_candles_from_db(epic, start_time, end_time, 5)
            
            if len(df) == 0:
                return None
                
            # Convert timestamp column if not already converted
            if 'start_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['start_time']):
                df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Database query failed: {e}")
            return None
    
    def _calculate_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        try:
            # RSI
            df = self.technical_analyzer.calculate_rsi(df, period=14)
            
            # ATR for volatility
            df = self.technical_analyzer.calculate_atr(df, period=14)
            
            # Bollinger Bands
            df = self.technical_analyzer.calculate_bollinger_bands(df, period=20, std_dev=2.0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating base indicators: {e}")
            return df
    
    def _calculate_ema_indicators(self, df: pd.DataFrame, epic: str, strategy: str = None) -> pd.DataFrame:
        """Calculate EMA indicators with proper configuration"""
        try:
            # Get EMA configuration for this epic
            # This would use the same logic as the actual scanner
            ema_periods = [21, 50, 200]  # Default periods
            
            # Try to get configuration from the strategy system
            try:
                from configdata import config as cfg
                if hasattr(cfg, 'get_ema_config_for_epic'):
                    ema_config = cfg.get_ema_config_for_epic(epic)
                    ema_periods = [ema_config.get('short', 21), ema_config.get('long', 50), ema_config.get('trend', 200)]
            except:
                pass  # Use defaults
            
            # Calculate EMAs with adjust=False to match live scanner (EMAIndicatorCalculator)
            for period in ema_periods:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Add semantic columns for backward compatibility
            if len(ema_periods) >= 3:
                df['ema_short'] = df[f'ema_{ema_periods[0]}']
                df['ema_long'] = df[f'ema_{ema_periods[1]}'] 
                df['ema_trend'] = df[f'ema_{ema_periods[2]}']
            
            # Calculate Two-Pole Oscillator if needed
            try:
                df = self.technical_analyzer.calculate_two_pole_oscillator(df)
            except:
                pass  # Not critical
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating EMA indicators: {e}")
            return df
    
    def _calculate_macd_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        try:
            df = self.technical_analyzer.calculate_macd(df, fast=12, slow=26, signal=9)
            return df
        except Exception as e:
            self.logger.error(f"❌ Error calculating MACD indicators: {e}")
            return df
    
    def _calculate_kama_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate KAMA indicators"""
        try:
            df = self.technical_analyzer.calculate_kama(df, period=14)
            return df
        except Exception as e:
            self.logger.error(f"❌ Error calculating KAMA indicators: {e}")
            return df
    
    def _enhance_with_live_methods(self, df: pd.DataFrame, epic: str, strategy: str) -> pd.DataFrame:
        """
        Use the live DataFetcher's enhancement methods to add indicators
        
        This ensures we use exactly the same calculation methods as the live system,
        eliminating discrepancies between validation and live results.
        """
        try:
            self.logger.info(f"🔧 Enhancing data using live DataFetcher methods")
            
            # Create a temporary EMA strategy to get the configuration
            # This mimics what the live system does
            from forex_scanner.core.strategies.ema_strategy import EMAStrategy
            temp_ema_strategy = EMAStrategy(backtest_mode=True)
            
            # Get EMA periods the same way the live system does
            ema_periods = self.data_fetcher._get_required_ema_periods(epic, temp_ema_strategy)
            self.logger.info(f"📊 Using EMA periods from live system: {ema_periods}")
            
            # Use the live DataFetcher's TechnicalAnalyzer to add EMAs
            # This ensures we use the exact same calculation method (adjust=False)
            df_enhanced = self.data_fetcher.technical_analyzer.add_ema_indicators(df, ema_periods)
            
            self.logger.info("✅ Enhanced data using live system methods")
            
            # Log the calculated EMA values for verification
            if len(df_enhanced) > 0:
                last_row = df_enhanced.iloc[-1]
                ema_values = {}
                for period in ema_periods:
                    col_name = f'ema_{period}'
                    if col_name in df_enhanced.columns:
                        ema_values[col_name] = last_row[col_name]
                
                self.logger.info(f"📈 Latest EMA values: {ema_values}")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error enhancing with live methods: {e}")
            # Fallback to original data if enhancement fails
            return df
    
    def _calculate_momentum_bias_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Momentum Bias indicators"""
        try:
            # This would implement the Momentum Bias Index calculation
            # Placeholder for now
            df['momentum_bias_index'] = 0.5  # Default neutral value
            return df
        except Exception as e:
            self.logger.error(f"❌ Error calculating Momentum Bias indicators: {e}")
            return df
    
    def _add_mtf_analysis(self, df: pd.DataFrame, epic: str, timestamp: datetime, timeframe: str) -> pd.DataFrame:
        """Add multi-timeframe analysis"""
        try:
            # This would add higher timeframe analysis
            # For now, add placeholder columns
            df['mtf_trend'] = 'BULLISH'  # Placeholder
            df['mtf_momentum'] = 0.6     # Placeholder
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding MTF analysis: {e}")
            return df
    
    def _add_timezone_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add timezone-specific columns"""
        try:
            # Add user timezone column
            df['user_time'] = df['start_time'].dt.tz_convert(self.timezone_manager.user_timezone)
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding timezone columns: {e}")
            return df
    
    def _validate_data_integrity(self, df: pd.DataFrame, required_indicators: List[str]) -> bool:
        """Validate that the data has all required indicators"""
        try:
            # Check for basic OHLC data
            required_columns = ['open', 'high', 'low', 'close', 'start_time']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"❌ Missing required column: {col}")
                    return False
            
            # Check for required indicators
            for indicator in required_indicators:
                if indicator not in df.columns:
                    self.logger.warning(f"⚠️ Missing indicator: {indicator}")
            
            # Check data quality
            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                self.logger.error("❌ OHLC data contains null values")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating data integrity: {e}")
            return False
    
    def _get_cache_key(self, epic: str, timestamp: datetime, timeframe: str, lookback_bars: int) -> str:
        """Generate cache key for data caching"""
        return f"{epic}_{timestamp.isoformat()}_{timeframe}_{lookback_bars}"
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached data if valid"""
        if cache_key not in self._data_cache:
            return None
        
        cached_item = self._data_cache[cache_key]
        if datetime.now() - cached_item['timestamp'] > timedelta(seconds=self._cache_timeout):
            del self._data_cache[cache_key]
            return None
        
        return cached_item['data']
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp"""
        self._data_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Cleanup old cache entries
        if len(self._data_cache) > 100:  # Limit cache size
            oldest_key = min(self._data_cache.keys(), 
                           key=lambda k: self._data_cache[k]['timestamp'])
            del self._data_cache[oldest_key]
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._data_cache.clear()
        self.logger.info("🧹 Historical data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_items': len(self._data_cache),
            'cache_timeout': self._cache_timeout,
            'memory_usage_mb': sum(
                df['data'].memory_usage(deep=True).sum() 
                for df in self._data_cache.values()
            ) / (1024 * 1024)
        }
    
    # ========== ENHANCED METHODS FOR REAL TRADE ANALYSIS ==========
    
    def get_market_data_at_timestamp(
        self,
        epic: str,
        timestamp: datetime,
        timeframe: str = '15m',
        exact_match: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get market data at exact timestamp for real trade analysis
        
        This method provides second-level precision for analyzing real trades
        by finding the exact candle that was active when the trade was placed.
        
        Args:
            epic: Epic code 
            timestamp: Exact timestamp (second precision)
            timeframe: Timeframe for analysis
            exact_match: If True, find exact candle containing timestamp
            
        Returns:
            DataFrame with the exact candle data or None if not found
        """
        try:
            self.logger.info(f"🎯 Getting exact market data at {timestamp}")
            
            # Ensure timezone
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            # Calculate time window for exact candle search
            timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
            timeframe_delta = timedelta(minutes=timeframe_minutes)
            
            # For exact match, we need to find the candle that contains this timestamp
            if exact_match:
                self.logger.info(f"🎯 Real trade timestamp analysis: {timestamp}")
                
                # Since we store 5-minute candles and synthesize 15-minute ones,
                # and trades can happen every 2 minutes (not aligned to candle boundaries),
                # we need to find the appropriate 5-minute candles to synthesize the timeframe
                
                if timeframe_minutes == 15:
                    # For 15m analysis, we need to synthesize from three 5m candles
                    # Find which 15m period this timestamp falls into
                    minute = timestamp.minute
                    candle_start_minute = (minute // 15) * 15
                    candle_start = timestamp.replace(minute=candle_start_minute, second=0, microsecond=0)
                    
                    # Get the three 5-minute candles that make up this 15-minute period
                    candle_times = [
                        candle_start,
                        candle_start + timedelta(minutes=5),
                        candle_start + timedelta(minutes=10)
                    ]
                    
                    self.logger.info(f"📊 Synthesizing 15m candle from 5m candles:")
                    self.logger.info(f"   Period: {candle_start} to {candle_start + timedelta(minutes=15)}")
                    self.logger.info(f"   5m candles: {[t.strftime('%H:%M') for t in candle_times]}")
                    
                    # Query for all three 5-minute candles
                    query = text("""
                        SELECT * FROM ig_candles 
                        WHERE epic = :epic 
                        AND timeframe = 5
                        AND start_time IN :candle_times
                        ORDER BY start_time ASC
                    """)
                    
                    with self.db_manager.get_engine().connect() as conn:
                        df_5m = pd.read_sql(
                            query,
                            conn,
                            params={
                                'epic': epic,
                                'candle_times': tuple(candle_times)
                            }
                        )
                    
                    if len(df_5m) > 0:
                        df_5m['start_time'] = pd.to_datetime(df_5m['start_time'], utc=True)
                        
                        # Synthesize 15-minute candle from 5-minute data
                        df = self._synthesize_15m_from_5m(df_5m, candle_start)
                        
                        if df is not None and len(df) > 0:
                            self.logger.info(f"✅ Synthesized 15m candle from {len(df_5m)} 5m candles")
                            return df
                        else:
                            self.logger.warning(f"⚠️ Failed to synthesize 15m candle")
                            return None
                    else:
                        self.logger.warning(f"⚠️ No 5m candles found for synthesis")
                        return None
                        
                else:
                    # For other timeframes, find the containing candle period
                    if timeframe_minutes == 5:
                        # Find the 5-minute candle that contains this timestamp
                        candle_start_minute = (timestamp.minute // 5) * 5
                        candle_start = timestamp.replace(minute=candle_start_minute, second=0, microsecond=0)
                    elif timeframe_minutes == 60:
                        # Find the hourly candle - synthesize from 5-minute candles
                        candle_start = timestamp.replace(minute=0, second=0, microsecond=0)
                        return self._synthesize_hourly_from_5m(epic, candle_start)
                    else:
                        candle_start_minute = 0
                        candle_start = timestamp.replace(minute=candle_start_minute, second=0, microsecond=0)
                    
                    self.logger.info(f"📊 Searching for {timeframe_minutes}m candle containing {timestamp}")
                    self.logger.info(f"   Candle period: {candle_start} to {candle_start + timeframe_delta}")
                    
                    # Query for the specific 5-minute candle
                    query = text("""
                        SELECT * FROM ig_candles 
                        WHERE epic = :epic 
                        AND timeframe = 5
                        AND start_time = :candle_start
                        ORDER BY start_time ASC
                    """)
                    
                    with self.db_manager.get_engine().connect() as conn:
                        df = pd.read_sql(
                            query, 
                            conn,
                            params={
                                'epic': epic,
                                'candle_start': candle_start
                            }
                        )
                
                if len(df) > 0:
                    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
                    self.logger.info(f"✅ Found exact candle at {df.iloc[0]['start_time']}")
                    return df
                else:
                    self.logger.warning(f"⚠️ No exact candle found for {candle_start}")
                    return None
            else:
                # Fallback to nearest candle before timestamp
                return self.get_historical_candles(epic, timestamp, timeframe, lookback_bars=1)
                
        except Exception as e:
            self.logger.error(f"❌ Error getting market data at timestamp: {e}")
            return None
    
    def get_market_data_range(
        self,
        epic: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        timeframe: str = '15m'
    ) -> Optional[pd.DataFrame]:
        """
        Get market data for a specific time range with second-level precision
        
        This method fetches all candles between start and end timestamps,
        perfect for analyzing price movement after a trade signal.
        
        Args:
            epic: Epic code
            start_timestamp: Range start (second precision)
            end_timestamp: Range end (second precision)  
            timeframe: Timeframe for analysis
            
        Returns:
            DataFrame with all candles in the range or None if error
        """
        try:
            self.logger.info(f"📈 Getting market data range: {start_timestamp} to {end_timestamp}")
            
            # Ensure timezones
            if start_timestamp.tzinfo is None:
                start_timestamp = start_timestamp.replace(tzinfo=timezone.utc)
            if end_timestamp.tzinfo is None:
                end_timestamp = end_timestamp.replace(tzinfo=timezone.utc)
            
            # Since we store 5-minute candles, we need different logic for different timeframes
            timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
            
            if timeframe_minutes == 5:
                # Direct query for 5-minute data
                query = text("""
                    SELECT * FROM ig_candles 
                    WHERE epic = :epic 
                    AND timeframe = 5
                    AND start_time >= :start_time
                    AND start_time <= :end_time
                    ORDER BY start_time ASC
                """)
                
                with self.db_manager.get_engine().connect() as conn:
                    df = pd.read_sql(
                        query,
                        conn,
                        params={
                            'epic': epic,
                            'start_time': start_timestamp,
                            'end_time': end_timestamp
                        }
                    )
            else:
                # For 15m, 1h etc., we need to synthesize from 5m data
                self.logger.info(f"📊 Synthesizing {timeframe_minutes}m candles from 5m data for range query")
                
                # Get all 5-minute candles in the range
                query = text("""
                    SELECT * FROM ig_candles 
                    WHERE epic = :epic 
                    AND timeframe = 5
                    AND start_time >= :start_time
                    AND start_time <= :end_time
                    ORDER BY start_time ASC
                """)
                
                with self.db_manager.get_engine().connect() as conn:
                    df_5m = pd.read_sql(
                        query,
                        conn,
                        params={
                            'epic': epic,
                            'start_time': start_timestamp,
                            'end_time': end_timestamp
                        }
                    )
                
                if len(df_5m) > 0:
                    df_5m['start_time'] = pd.to_datetime(df_5m['start_time'], utc=True)
                    
                    # Synthesize higher timeframe candles
                    if timeframe_minutes == 15:
                        df = self._synthesize_15m_range_from_5m(df_5m, start_timestamp, end_timestamp)
                    elif timeframe_minutes == 60:
                        df = self._synthesize_hourly_range_from_5m(df_5m, start_timestamp, end_timestamp)
                    else:
                        # Fallback to 5m data
                        df = df_5m
                else:
                    df = pd.DataFrame()
            
            if len(df) > 0:
                df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
                self.logger.info(f"✅ Found {len(df)} candles in range")
                self.logger.info(f"   Range: {df.iloc[0]['start_time']} to {df.iloc[-1]['start_time']}")
                return df
            else:
                self.logger.warning(f"⚠️ No candles found in range {start_timestamp} to {end_timestamp}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting market data range: {e}")
            return None
    
    def get_precise_signal_context(
        self,
        epic: str,
        signal_timestamp: datetime,
        timeframe: str = '15m',
        context_bars_before: int = 50,
        context_bars_after: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive market context around a real trade signal
        
        This method provides detailed context for real trade analysis including
        the exact market conditions before and after the signal timestamp.
        
        Args:
            epic: Epic code
            signal_timestamp: Exact signal timestamp (second precision)
            timeframe: Timeframe for analysis
            context_bars_before: Number of bars to include before signal
            context_bars_after: Number of bars to include after signal
            
        Returns:
            Dictionary with comprehensive signal context
        """
        try:
            self.logger.info(f"🔍 Getting precise signal context for {epic} @ {signal_timestamp}")
            
            # Ensure timezone
            if signal_timestamp.tzinfo is None:
                signal_timestamp = signal_timestamp.replace(tzinfo=timezone.utc)
            
            # Calculate time ranges
            timeframe_minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
            bar_duration = timedelta(minutes=timeframe_minutes)
            
            context_start = signal_timestamp - (bar_duration * context_bars_before)
            context_end = signal_timestamp + (bar_duration * context_bars_after)
            
            # Get full context data
            context_df = self.get_market_data_range(epic, context_start, context_end, timeframe)
            
            if context_df is None or len(context_df) == 0:
                return None
            
            # Find the exact signal candle
            signal_candles = context_df[context_df['start_time'] <= signal_timestamp]
            if len(signal_candles) == 0:
                signal_candle_idx = 0
                signal_candle = context_df.iloc[0]
            else:
                signal_candle_idx = len(signal_candles) - 1
                signal_candle = signal_candles.iloc[-1]
            
            # Build comprehensive context
            context = {
                'signal_timestamp': signal_timestamp,
                'signal_candle_timestamp': signal_candle['start_time'],
                'signal_candle_index': signal_candle_idx,
                'total_candles': len(context_df),
                'bars_before_signal': signal_candle_idx,
                'bars_after_signal': len(context_df) - signal_candle_idx - 1,
                
                # Signal candle data
                'signal_candle': {
                    'open': float(signal_candle['open']),
                    'high': float(signal_candle['high']),
                    'low': float(signal_candle['low']),
                    'close': float(signal_candle['close']),
                    'volume': int(signal_candle.get('volume', 0))
                },
                
                # Context data
                'context_data': context_df,
                
                # Quick stats
                'context_stats': {
                    'highest_high': float(context_df['high'].max()),
                    'lowest_low': float(context_df['low'].min()),
                    'price_range_pips': (context_df['high'].max() - context_df['low'].min()) * 10000,
                    'average_volume': int(context_df.get('volume', pd.Series([0])).mean())
                }
            }
            
            self.logger.info(f"✅ Signal context built:")
            self.logger.info(f"   Signal candle: {signal_candle['start_time']}")
            self.logger.info(f"   Context range: {context_df.iloc[0]['start_time']} to {context_df.iloc[-1]['start_time']}")
            self.logger.info(f"   Total candles: {len(context_df)} ({signal_candle_idx} before, {len(context_df)-signal_candle_idx-1} after)")
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Error getting precise signal context: {e}")
            return None
    
    def _synthesize_15m_from_5m(self, df_5m: pd.DataFrame, candle_start: datetime) -> Optional[pd.DataFrame]:
        """
        Synthesize a 15-minute candle from three 5-minute candles
        
        This is crucial for real trade analysis since the system stores 5m candles
        but analyzes on 15m timeframes.
        
        Args:
            df_5m: DataFrame with 5-minute candles (should have 3 rows)
            candle_start: Start time for the synthesized 15m candle
            
        Returns:
            DataFrame with single 15m candle or None if synthesis fails
        """
        try:
            if df_5m.empty:
                return None
            
            # Sort by time to ensure correct order
            df_5m = df_5m.sort_values('start_time').reset_index(drop=True)
            
            # Synthesize 15m OHLC from 5m data
            open_price = float(df_5m.iloc[0]['open'])  # First 5m open
            high_price = float(df_5m['high'].max())    # Highest high
            low_price = float(df_5m['low'].min())      # Lowest low
            close_price = float(df_5m.iloc[-1]['close']) # Last 5m close
            total_volume = int(df_5m['volume'].sum() if 'volume' in df_5m.columns else 0)
            
            # Create synthesized 15m candle
            synthesized_candle = pd.DataFrame({
                'epic': [df_5m.iloc[0]['epic']],
                'start_time': [candle_start],
                'timeframe': [15],
                'open': [open_price],
                'high': [high_price],
                'low': [low_price],
                'close': [close_price],
                'volume': [total_volume]
            })
            
            self.logger.info(f"📊 15m candle synthesized:")
            self.logger.info(f"   OHLC: {open_price:.5f} | {high_price:.5f} | {low_price:.5f} | {close_price:.5f}")
            self.logger.info(f"   Volume: {total_volume}")
            self.logger.info(f"   From {len(df_5m)} x 5m candles")
            
            return synthesized_candle
            
        except Exception as e:
            self.logger.error(f"❌ Error synthesizing 15m candle: {e}")
            return None
    
    def _synthesize_hourly_from_5m(self, epic: str, hour_start: datetime) -> Optional[pd.DataFrame]:
        """
        Synthesize an hourly candle from twelve 5-minute candles
        
        Args:
            epic: Epic code
            hour_start: Start of the hour
            
        Returns:
            DataFrame with single hourly candle or None if synthesis fails
        """
        try:
            # Get all 5-minute candles for this hour
            hour_end = hour_start + timedelta(hours=1)
            
            query = text("""
                SELECT * FROM ig_candles 
                WHERE epic = :epic 
                AND timeframe = 5
                AND start_time >= :hour_start
                AND start_time < :hour_end
                ORDER BY start_time ASC
            """)
            
            with self.db_manager.get_engine().connect() as conn:
                df_5m = pd.read_sql(
                    query,
                    conn,
                    params={
                        'epic': epic,
                        'hour_start': hour_start,
                        'hour_end': hour_end
                    }
                )
            
            if len(df_5m) == 0:
                return None
                
            df_5m['start_time'] = pd.to_datetime(df_5m['start_time'], utc=True)
            
            # Synthesize hourly OHLC
            open_price = float(df_5m.iloc[0]['open'])
            high_price = float(df_5m['high'].max())
            low_price = float(df_5m['low'].min())
            close_price = float(df_5m.iloc[-1]['close'])
            total_volume = int(df_5m['volume'].sum() if 'volume' in df_5m.columns else 0)
            
            synthesized_candle = pd.DataFrame({
                'epic': [epic],
                'start_time': [hour_start],
                'timeframe': [60],
                'open': [open_price],
                'high': [high_price],
                'low': [low_price],
                'close': [close_price],
                'volume': [total_volume]
            })
            
            self.logger.info(f"📊 Hourly candle synthesized from {len(df_5m)} x 5m candles")
            return synthesized_candle
            
        except Exception as e:
            self.logger.error(f"❌ Error synthesizing hourly candle: {e}")
            return None
    
    def _synthesize_15m_range_from_5m(self, df_5m: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Synthesize a range of 15-minute candles from 5-minute candle data
        
        Args:
            df_5m: DataFrame with 5-minute candles
            start_time: Range start
            end_time: Range end
            
        Returns:
            DataFrame with synthesized 15-minute candles
        """
        try:
            if df_5m.empty:
                return pd.DataFrame()
                
            synthesized_candles = []
            
            # Group 5m candles into 15m periods
            # Round start_time down to nearest 15m boundary
            current_15m_start = start_time.replace(minute=(start_time.minute // 15) * 15, second=0, microsecond=0)
            
            while current_15m_start < end_time:
                current_15m_end = current_15m_start + timedelta(minutes=15)
                
                # Get 5m candles in this 15m period
                period_candles = df_5m[
                    (df_5m['start_time'] >= current_15m_start) & 
                    (df_5m['start_time'] < current_15m_end)
                ].copy()
                
                if len(period_candles) > 0:
                    # Synthesize 15m candle from this group
                    synthesized_15m = self._synthesize_15m_from_5m(period_candles, current_15m_start)
                    if synthesized_15m is not None and len(synthesized_15m) > 0:
                        synthesized_candles.append(synthesized_15m)
                
                # Move to next 15m period
                current_15m_start = current_15m_end
            
            if synthesized_candles:
                # Combine all synthesized candles
                result_df = pd.concat(synthesized_candles, ignore_index=True)
                self.logger.info(f"📊 Synthesized {len(result_df)} x 15m candles from {len(df_5m)} x 5m candles")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error synthesizing 15m range: {e}")
            return pd.DataFrame()
    
    def _synthesize_hourly_range_from_5m(self, df_5m: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Synthesize a range of hourly candles from 5-minute candle data
        
        Args:
            df_5m: DataFrame with 5-minute candles
            start_time: Range start
            end_time: Range end
            
        Returns:
            DataFrame with synthesized hourly candles
        """
        try:
            if df_5m.empty:
                return pd.DataFrame()
                
            synthesized_candles = []
            
            # Group 5m candles into hourly periods
            current_hour_start = start_time.replace(minute=0, second=0, microsecond=0)
            
            while current_hour_start < end_time:
                current_hour_end = current_hour_start + timedelta(hours=1)
                
                # Get 5m candles in this hour
                hour_candles = df_5m[
                    (df_5m['start_time'] >= current_hour_start) & 
                    (df_5m['start_time'] < current_hour_end)
                ].copy()
                
                if len(hour_candles) > 0:
                    # Synthesize hourly candle
                    open_price = float(hour_candles.iloc[0]['open'])
                    high_price = float(hour_candles['high'].max())
                    low_price = float(hour_candles['low'].min())
                    close_price = float(hour_candles.iloc[-1]['close'])
                    total_volume = int(hour_candles['volume'].sum() if 'volume' in hour_candles.columns else 0)
                    
                    hourly_candle = pd.DataFrame({
                        'epic': [hour_candles.iloc[0]['epic']],
                        'start_time': [current_hour_start],
                        'timeframe': [60],
                        'open': [open_price],
                        'high': [high_price],
                        'low': [low_price],
                        'close': [close_price],
                        'volume': [total_volume]
                    })
                    
                    synthesized_candles.append(hourly_candle)
                
                # Move to next hour
                current_hour_start = current_hour_end
            
            if synthesized_candles:
                # Combine all synthesized candles
                result_df = pd.concat(synthesized_candles, ignore_index=True)
                self.logger.info(f"📊 Synthesized {len(result_df)} x 1h candles from {len(df_5m)} x 5m candles")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error synthesizing hourly range: {e}")
            return pd.DataFrame()