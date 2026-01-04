# core/data_fetcher.py
"""
Optimized Data Fetching Module with KAMA Support
Handles fetching and enhancing candle data from database with performance optimizations
Now supports dynamic EMA configuration and KAMA indicators
FIXED: Properly respects disabled strategies in config.py
FIXED: Uses EMA strategy's configuration methods instead of hardcoded dynamic manager
ENHANCED: 15m resampling with trading confidence scoring and completeness tracking
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import hashlib
import pickle
from functools import lru_cache

try:
    from .database import DatabaseManager
    from analysis.technical import TechnicalAnalyzer
    from analysis.volume import VolumeAnalyzer
    from analysis.behavior import BehaviorAnalyzer
    from utils.timezone_utils import TimezoneManager, add_timezone_columns
    from configdata import config
    import config as system_config
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.analysis.technical import TechnicalAnalyzer
    from forex_scanner.analysis.volume import VolumeAnalyzer
    from forex_scanner.analysis.behavior import BehaviorAnalyzer
    from forex_scanner.utils.timezone_utils import TimezoneManager, add_timezone_columns
    from forex_scanner.configdata import config
    from forex_scanner import config as system_config

# Import scanner config service for database-driven settings
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    SCANNER_CONFIG_AVAILABLE = True
except ImportError:
    SCANNER_CONFIG_AVAILABLE = False
    get_scanner_config = None


class DataFetcher:
    """Enhanced data fetcher with KAMA support and performance optimizations"""
    
    def __init__(self, db_manager: DatabaseManager, user_timezone: str = 'Europe/Stockholm'):
        self.db_manager = db_manager
        self.technical_analyzer = TechnicalAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.timezone_manager = TimezoneManager(user_timezone)
        self.logger = logging.getLogger(__name__)

        # Performance optimizations - try database first, fallback to config.py
        self._scanner_config = None  # Store reference for strategy checks
        if SCANNER_CONFIG_AVAILABLE:
            try:
                self._scanner_config = get_scanner_config()
                self.cache_enabled = self._scanner_config.enable_data_cache
                self.reduced_lookback = self._scanner_config.reduced_lookback_hours
                self.lazy_indicators = self._scanner_config.lazy_indicator_loading
                self.batch_size = self._scanner_config.data_batch_size
                self.enable_support_resistance = self._scanner_config.enable_support_resistance
                self.enable_volume_analysis = self._scanner_config.enable_volume_analysis
                self.enable_behavior_analysis = self._scanner_config.enable_behavior_analysis
                logging.getLogger(__name__).debug("[CONFIG:DB] DataFetcher settings loaded from database")
            except Exception as e:
                logging.getLogger(__name__).warning(f"[CONFIG:DB] Failed to load DataFetcher settings: {e}")
                self._load_fallback_config()
        else:
            self._load_fallback_config()

        # Cache for recently fetched data
        self._data_cache = {}
        self._cache_timeout = 150  # 5 minutes
        self._indicator_cache = {}

    def _load_fallback_config(self):
        """Load hardcoded fallback defaults when database is unavailable"""
        self.cache_enabled = False
        self.reduced_lookback = True
        self.lazy_indicators = True
        self.batch_size = 10000
        self.enable_support_resistance = True
        self.enable_volume_analysis = True
        self.enable_behavior_analysis = False

    
    def get_enhanced_data(
        self, 
        epic: str, 
        pair: str, 
        timeframe: str = None,
        lookback_hours: int = None,
        user_timezone: str = None,
        required_indicators: list = None,
        ema_strategy=None
    ) -> Optional[pd.DataFrame]:
        """
        FIXED: Get enhanced candle data with proper EMA strategy integration
        
        This version properly integrates with the EMA strategy's configuration methods
        to ensure the selected EMA periods are actually used in calculations.
        
        Args:
            epic: Epic code
            pair: Currency pair
            timeframe: Timeframe ('5m', '15m', '1h')
            lookback_hours: Hours of historical data (auto-optimized if None)
            user_timezone: User's timezone (default: use fetcher's timezone)
            required_indicators: List of required indicators (for lazy loading)
            ema_strategy: EMA strategy instance for configuration
            
        Returns:
            Enhanced DataFrame or None if error
        """
        # Use default timeframe from database if not specified - NO FALLBACK
        if timeframe is None:
            if self._scanner_config:
                timeframe = self._scanner_config.default_timeframe
            else:
                # Use hardcoded default if no config available (should not happen in normal operation)
                timeframe = '15m'
            
        try:
            # Optimize lookback hours based on usage
            if lookback_hours is None:
                lookback_hours = self._get_optimal_lookback_hours(epic, timeframe, ema_strategy)
            
            # Use provided timezone or default
            if user_timezone:
                tz_manager = TimezoneManager(user_timezone)
            else:
                tz_manager = self.timezone_manager
            
            # FIXED: Get EMA periods from strategy or config
            ema_periods = self._get_required_ema_periods(epic, ema_strategy)
            
            # Determine configuration source for logging
            config_source = self._get_ema_config_source(epic, ema_strategy)
            
            # Cached EMA configuration logging to prevent spam
            if not hasattr(self, '_logged_ema_configs'):
                self._logged_ema_configs = set()
            
            # Create unique config key for this epic and EMA configuration
            config_key = f"{epic}_{tuple(ema_periods)}_{config_source}"
            
            if config_key not in self._logged_ema_configs:
                # First time seeing this configuration - log at DEBUG level
                self.logger.debug(f"🎯 EMA configuration for {epic}:")
                self.logger.debug(f"   Periods: {ema_periods}")
                self.logger.debug(f"   Source: {config_source}")
                
                # Mark this config as logged
                self._logged_ema_configs.add(config_key)
                
                # Optional: Clean up old entries to prevent memory growth
                if len(self._logged_ema_configs) > 100:  # Keep last 100 unique configs
                    # Remove oldest entries (simple approach)
                    old_entries = list(self._logged_ema_configs)[:50]
                    for old_entry in old_entries:
                        self._logged_ema_configs.discard(old_entry)
                    self.logger.debug(f"🧹 Cleaned up old EMA config log cache (kept {len(self._logged_ema_configs)} entries)")
            else:
                # Configuration already logged - use debug level for subsequent calls
                self.logger.debug(f"🎯 EMA config for {epic}: {ema_periods} ({config_source}) [cached]")
            
            # Check cache first (include EMA periods and KAMA config in cache key)
            cache_key = self._get_cache_key(epic, timeframe, lookback_hours, ema_periods)
            if self.cache_enabled and self._is_cache_valid(cache_key):
                self.logger.debug(f"📦 Using cached data for {epic} with EMA config {ema_periods}")
                cached_data = self._data_cache[cache_key]['data']
                
                # Apply lazy indicator loading to cached data
                if self.lazy_indicators and required_indicators:
                    return self._apply_lazy_indicators(cached_data, pair, required_indicators, ema_strategy)
                return cached_data
            
            # Fetch raw candle data
            df = self._fetch_candle_data_optimized(epic, timeframe, lookback_hours, tz_manager)
            
            if df is None or len(df) == 0:
                self.logger.warning(f"⚠️ No data fetched for {epic}")
                return None
            
            # FIXED: Enhanced analysis with proper EMA configuration
            df_enhanced = self._enhance_with_analysis_optimized(df, pair, ema_strategy, ema_periods)
            
            # ENHANCED: Validate that the correct EMAs were calculated
            self._validate_ema_calculation(df_enhanced, epic, ema_periods, config_source)
            
            # Cache the enhanced data
            if self.cache_enabled:
                self._cache_data(cache_key, df_enhanced)
            
            self.logger.debug(f"✅ Enhanced data for {epic}: {len(df_enhanced)} bars")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error getting enhanced data for {epic}: {e}")
            return None

    def clear_ema_config_log_cache(self):
        """Clear the EMA configuration logging cache"""
        if hasattr(self, '_logged_ema_configs'):
            self._logged_ema_configs.clear()
            self.logger.info("🧹 EMA configuration logging cache cleared")

    def get_ema_config_log_stats(self):
        """Get statistics about the EMA configuration logging cache"""
        if hasattr(self, '_logged_ema_configs'):
            return {
                'cached_configs': len(self._logged_ema_configs),
                'memory_usage_estimate': len(self._logged_ema_configs) * 50,  # rough estimate in bytes
                'sample_entries': list(self._logged_ema_configs)[:5]  # first 5 entries as sample
            }
        else:
            return {'cached_configs': 0, 'memory_usage_estimate': 0, 'sample_entries': []}

    def _get_ema_config_source(self, epic: str, ema_strategy=None) -> str:
        """
        Determine which EMA configuration source is being used.
        Note: EMA strategy is archived - only SMC_SIMPLE is active.
        EMA periods (21/50/200) are now used for indicator calculations only.
        """
        try:
            # Check if we have an EMA strategy instance with dynamic config
            if ema_strategy and hasattr(ema_strategy, 'enable_dynamic_config'):
                if ema_strategy.enable_dynamic_config:
                    return "dynamic (strategy-managed)"
                else:
                    config_name = getattr(ema_strategy, 'ema_config_name', 'default')
                    return f"static (strategy config: {config_name})"

            # Default EMA periods used for indicator calculations
            return "standard (21/50/200)"

        except Exception:
            return "unknown"
    
    def _enhance_with_analysis_optimized(self, df: pd.DataFrame, pair: str, ema_strategy=None, ema_periods=None) -> pd.DataFrame:
        """
        Enhanced analysis with proper market_timestamp handling and ALL indicators.
        Always adds all indicators for SMC_SIMPLE strategy and performance collection.
        """
        try:
            df_enhanced = df.copy()

            # Ensure EMA 200 is always present
            df_enhanced = self._ensure_ema200_always_present(df_enhanced)

            # Get EMA periods
            if ema_periods is None:
                epic_guess = f"CS.D.{pair}.MINI.IP" if pair and '.' not in pair else pair
                ema_periods = self._get_required_ema_periods(epic_guess, ema_strategy)

            self.logger.debug(f"🔄 Adding EMA indicators: {ema_periods}")

            # Add EMA indicators
            df_enhanced = self.technical_analyzer.add_ema_indicators(df_enhanced, ema_periods)

            # Add semantic column mapping for dynamic periods
            df_enhanced = self._add_semantic_ema_columns(df_enhanced, ema_periods)

            # Always add MACD indicators (needed for SMC_SIMPLE MACD filter)
            self.logger.debug("🔄 Adding MACD indicators")
            # Get MACD periods from database config
            macd_fast = self._scanner_config.macd_fast_period if self._scanner_config else 12
            macd_slow = self._scanner_config.macd_slow_period if self._scanner_config else 26
            macd_signal = self._scanner_config.macd_signal_period if self._scanner_config else 9
            df_enhanced = self.technical_analyzer.add_macd_indicators(
                df_enhanced,
                macd_fast,
                macd_slow,
                macd_signal
            )

            # Ensure EMA 50, 100, 200 for trend filtering
            if 'ema_50' not in df_enhanced.columns:
                df_enhanced['ema_50'] = df_enhanced['close'].ewm(span=50).mean()
            if 'ema_100' not in df_enhanced.columns:
                df_enhanced['ema_100'] = df_enhanced['close'].ewm(span=100).mean()
            if 'ema_200' not in df_enhanced.columns:
                df_enhanced['ema_200'] = df_enhanced['close'].ewm(span=200).mean()

            # Always add KAMA indicators (for performance collection)
            self.logger.debug("🔄 Adding KAMA indicators")
            df_enhanced = self._add_kama_indicators(df_enhanced)

            # Always add BB+Supertrend indicators (for performance collection)
            self.logger.debug("🔄 Adding BB+Supertrend indicators")
            df_enhanced = self._add_bb_supertrend_indicators(df_enhanced)

            # Always add Zero Lag EMA indicators (for performance collection)
            self.logger.debug("🔄 Adding Zero Lag EMA indicators")
            df_enhanced = self._add_zero_lag_indicators(df_enhanced)

            # Always add Two-Pole Oscillator indicators (for performance collection)
            self.logger.debug("🔄 Adding Two-Pole Oscillator indicators")
            df_enhanced = self._add_two_pole_oscillator_indicators(df_enhanced)
            
            
            # Add other indicators conditionally using instance settings (from database)
            if self.enable_support_resistance:
                try:
                    df_enhanced = self.technical_analyzer.add_support_resistance_to_df(df_enhanced, pair)
                except AttributeError:
                    self.logger.debug("Support/resistance method not available, skipping")

            if self.enable_volume_analysis:
                try:
                    df_enhanced = self.volume_analyzer.add_volume_analysis(df_enhanced)
                except AttributeError:
                    self.logger.debug("Volume analysis method not available, skipping")

            if self.enable_behavior_analysis:
                try:
                    df_enhanced = self.behavior_analyzer.add_behavior_patterns(df_enhanced)
                except AttributeError:
                    self.logger.debug("Behavior analysis method not available, skipping")
            
            
            # CRITICAL FIX: Set proper market_timestamp from latest candle
            if len(df_enhanced) > 0:
                # Get the latest timestamp from the DataFrame index
                if hasattr(df_enhanced.index, '__getitem__') and len(df_enhanced.index) > 0:
                    latest_timestamp = df_enhanced.index[-1]
                elif 'start_time' in df_enhanced.columns:
                    latest_timestamp = df_enhanced.iloc[-1]['start_time']
                else:
                    latest_timestamp = datetime.now()
                
                # Convert to proper datetime
                if hasattr(latest_timestamp, 'to_pydatetime'):
                    market_timestamp = latest_timestamp.to_pydatetime()
                elif hasattr(latest_timestamp, 'timestamp'):
                    market_timestamp = latest_timestamp
                else:
                    # Use datetime parsing instead of pd.to_datetime
                    try:
                        if isinstance(latest_timestamp, str):
                            if 'T' in latest_timestamp:
                                market_timestamp = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                            else:
                                market_timestamp = datetime.strptime(latest_timestamp, '%Y-%m-%d %H:%M:%S')
                        else:
                            market_timestamp = datetime.now()
                    except (ValueError, TypeError):
                        market_timestamp = datetime.now()
                
                # Add market_timestamp column to DataFrame for signal processing
                df_enhanced['market_timestamp'] = market_timestamp
                
                self.logger.debug(f"✅ Set market_timestamp to latest candle: {market_timestamp}")
            else:
                self.logger.warning("⚠️ No data available to set market_timestamp")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced analysis: {e}")
            # Return original data with basic timestamp fix
            if len(df) > 0:
                try:
                    if hasattr(df.index, '__getitem__') and len(df.index) > 0:
                        latest_candle_time = df.index[-1]
                    elif 'start_time' in df.columns:
                        latest_candle_time = df.iloc[-1]['start_time']
                    else:
                        latest_candle_time = datetime.now()
                    
                    df['market_timestamp'] = latest_candle_time
                    self.logger.info(f"🔧 Applied emergency timestamp fix: {latest_candle_time}")
                except Exception as inner_e:
                    df['market_timestamp'] = datetime.now()
                    self.logger.warning(f"⚠️ Applied fallback current timestamp due to: {inner_e}")
            return df

    def _add_kama_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add KAMA indicators to dataframe using database config"""
        try:
            # Get KAMA configuration from database
            kama_period = self._scanner_config.kama_period if self._scanner_config else 10
            fast_sc = self._scanner_config.kama_fast if self._scanner_config else 2
            slow_sc = self._scanner_config.kama_slow if self._scanner_config else 30

            df_with_kama = self._calculate_kama_indicators(df, kama_period, fast_sc, slow_sc)

            self.logger.debug(f"✅ KAMA indicators added: KAMA_{kama_period}, ER, trend, signal")
            return df_with_kama

        except Exception as e:
            self.logger.error(f"❌ Error adding KAMA indicators: {e}")
            return df
    
    def _calculate_kama_indicators(self, df: pd.DataFrame, period: int, fast: int, slow: int) -> pd.DataFrame:
        """
        Calculate KAMA indicators for a specific period
        🔥 FIXED: Efficiency ratio calculation to prevent 0.000 values
        """
        try:
            df_kama = df.copy()
            
            if len(df_kama) < period + 5:  # Need extra buffer for calculations
                self.logger.warning(f"⚠️ Insufficient data for KAMA_{period} calculation: {len(df_kama)} < {period + 5}")
                return df_kama
            
            # 🔥 FIX 1: Improved Efficiency Ratio calculation
            close_series = df_kama['close']
            
            # Calculate directional change over the full period
            price_change = close_series.diff(period).abs()
            
            # Calculate total volatility (sum of absolute changes) over the period
            daily_changes = close_series.diff().abs()
            volatility = daily_changes.rolling(window=period, min_periods=max(1, period//2)).sum()
            
            # 🔥 FIX 2: Proper division by zero handling
            # Instead of replacing with NaN then filling with 0, handle it properly
            er = pd.Series(index=df_kama.index, dtype=float)
            
            for i in range(len(df_kama)):
                vol_val = volatility.iloc[i] if i < len(volatility) else 0
                change_val = price_change.iloc[i] if i < len(price_change) else 0
                
                if pd.isna(vol_val) or pd.isna(change_val) or vol_val == 0:
                    # 🔥 CRITICAL FIX: Use reasonable default instead of 0.000
                    if i < period:
                        er.iloc[i] = 0.1  # Default for insufficient data
                    else:
                        # Use previous value or reasonable default
                        er.iloc[i] = er.iloc[i-1] if i > 0 and not pd.isna(er.iloc[i-1]) else 0.1
                else:
                    calculated_er = change_val / vol_val
                    # 🔥 FIX 3: Clamp efficiency ratio to reasonable bounds
                    er.iloc[i] = max(0.01, min(1.0, calculated_er))  # Never below 0.01, never above 1.0
            
            # 🔥 FIX 4: Forward fill any remaining NaN values with minimum efficiency
            er = er.ffill().fillna(0.1)
            
            # 🔥 FIX 5: Validate efficiency ratios
            if (er == 0).any():
                self.logger.warning(f"⚠️ Found zero efficiency ratios, replacing with minimum 0.01")
                er = er.replace(0, 0.01)
            
            # Log efficiency ratio stats for debugging
            if len(er.dropna()) > 0:
                er_stats = er.dropna()
                self.logger.debug(f"📊 KAMA_{period} Efficiency Ratio - Min: {er_stats.min():.3f}, "
                                f"Max: {er_stats.max():.3f}, Mean: {er_stats.mean():.3f}")
                
                # Warn if most efficiency ratios are suspiciously low
                low_efficiency_pct = (er_stats < 0.05).sum() / len(er_stats)
                if low_efficiency_pct > 0.8:
                    self.logger.warning(f"⚠️ {low_efficiency_pct:.1%} of efficiency ratios are very low (<0.05) - "
                                    f"possible data quality issue")
            
            # Store the fixed efficiency ratio
            df_kama[f'kama_{period}_er'] = er
            
            # Calculate Smoothing Constants using fixed ER
            fast_sc = 2.0 / (fast + 1)
            slow_sc = 2.0 / (slow + 1)
            sc = ((er * (fast_sc - slow_sc)) + slow_sc) ** 2
            
            # Calculate KAMA using improved efficiency ratio
            kama = pd.Series(index=df_kama.index, dtype=float)
            kama.iloc[0] = close_series.iloc[0]  # Start with first close price
            
            for i in range(1, len(df_kama)):
                if pd.isna(sc.iloc[i]) or sc.iloc[i] == 0:
                    kama.iloc[i] = kama.iloc[i-1]
                else:
                    kama.iloc[i] = kama.iloc[i-1] + (sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1]))
            
            # Add to dataframe
            df_kama[f'kama_{period}'] = kama
            
            # Calculate trend direction (-1, 0, 1)
            kama_diff = kama.diff()
            df_kama[f'kama_{period}_trend'] = np.where(
                kama_diff > 0, 1,
                np.where(kama_diff < 0, -1, 0)
            )
            
            # Calculate signal strength (trend changes)
            trend_changes = df_kama[f'kama_{period}_trend'].diff()
            df_kama[f'kama_{period}_signal'] = np.where(
                trend_changes == 2, 2,    # Bullish trend change (-1 to 1)
                np.where(trend_changes == -2, -2, 0)  # Bearish trend change (1 to -1)
            )
            
            # 🔥 FIX 6: Final validation
            final_er_min = df_kama[f'kama_{period}_er'].min()
            if final_er_min <= 0:
                self.logger.error(f"❌ CRITICAL: Still found efficiency ratio ≤ 0 after fixes: {final_er_min}")
                # Emergency fix
                df_kama[f'kama_{period}_er'] = df_kama[f'kama_{period}_er'].clip(lower=0.01)
            
            return df_kama
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating KAMA indicators: {e}")
            # Return original dataframe to prevent crashes
            return df
    
    def _add_bb_supertrend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands and Supertrend indicators for performance collection"""
        try:
            self.logger.debug("🔄 Calculating BB+Supertrend indicators...")

            # Get configuration from database
            bb_period = self._scanner_config.bb_period if self._scanner_config else 20
            bb_std_dev = self._scanner_config.bb_std_dev if self._scanner_config else 2.0
            supertrend_period = self._scanner_config.supertrend_period if self._scanner_config else 10
            supertrend_multiplier = self._scanner_config.supertrend_multiplier if self._scanner_config else 3.0
            
            # Calculate Bollinger Bands
            if 'bb_upper' not in df.columns:
                rolling_mean = df['close'].rolling(window=bb_period).mean()
                rolling_std = df['close'].rolling(window=bb_period).std()
                
                df['bb_middle'] = rolling_mean
                df['bb_upper'] = rolling_mean + (rolling_std * bb_std_dev)
                df['bb_lower'] = rolling_mean - (rolling_std * bb_std_dev)
            
            # Calculate ATR for Supertrend and adaptive volatility
            if 'atr' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())

                true_range = np.maximum(high_low, np.maximum(high_close, low_close))

                # Standard ATR (14-period, used by supertrend_period which is typically 14)
                df['atr'] = true_range.rolling(window=supertrend_period).mean()

                # 20-period ATR for percentile calculation (adaptive volatility)
                df['atr_20'] = true_range.rolling(window=20).mean()

                # ATR percentile: current ATR vs 20-period baseline
                # Used for regime detection (high volatility, breakout conditions)
                df['atr_percentile'] = (df['atr'] / df['atr_20'] * 100).fillna(50.0)

            # Calculate Bollinger Band width percentile for ranging market detection
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_width = df['bb_upper'] - df['bb_lower']
                # Percentile rank over 50-period window
                df['bb_width_percentile'] = bb_width.rolling(window=50).rank(pct=True) * 100
                df['bb_width_percentile'] = df['bb_width_percentile'].fillna(50.0)
            
            if 'ltv' in df.columns and 'volume' not in df.columns:
                df['volume'] = df['ltv']
                self.logger.debug("📊 Mapped 'ltv' column to 'volume' for strategy compatibility")
            
            # Calculate Supertrend
            if 'supertrend' not in df.columns:
                hl2 = (df['high'] + df['low']) / 2
                atr = df['atr']
                
                # Initialize arrays
                supertrend = np.zeros(len(df))
                direction = np.ones(len(df))  # 1 for bullish, -1 for bearish
                
                # Set initial values
                if len(df) > 0:
                    supertrend[0] = hl2.iloc[0]
                    direction[0] = 1
                
                # Calculate upper and lower bands
                upper_band = hl2 + (supertrend_multiplier * atr)
                lower_band = hl2 - (supertrend_multiplier * atr)
                
                # Calculate supertrend
                for i in range(1, len(df)):
                    prev_close = df['close'].iloc[i-1]
                    curr_close = df['close'].iloc[i]
                    
                    # Upper band calculation
                    if upper_band.iloc[i] < upper_band.iloc[i-1] or prev_close > upper_band.iloc[i-1]:
                        final_upper = upper_band.iloc[i]
                    else:
                        final_upper = upper_band.iloc[i-1]
                    
                    # Lower band calculation  
                    if lower_band.iloc[i] > lower_band.iloc[i-1] or prev_close < lower_band.iloc[i-1]:
                        final_lower = lower_band.iloc[i]
                    else:
                        final_lower = lower_band.iloc[i-1]
                    
                    # Determine supertrend and direction
                    if supertrend[i-1] <= final_lower:
                        # Was in uptrend
                        if curr_close <= final_lower:
                            supertrend[i] = final_upper
                            direction[i] = -1
                        else:
                            supertrend[i] = final_lower
                            direction[i] = 1
                    else:
                        # Was in downtrend
                        if curr_close >= final_upper:
                            supertrend[i] = final_lower
                            direction[i] = 1
                        else:
                            supertrend[i] = final_upper
                            direction[i] = -1
                
                df['supertrend'] = supertrend
                df['supertrend_direction'] = direction
            
            self.logger.debug("✅ BB+Supertrend indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating BB+Supertrend indicators: {e}")
            return df


    def _add_zero_lag_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Zero Lag EMA indicators to DataFrame
        This mirrors the calculation in ZeroLagStrategy
        """
        try:
            # Get parameters from database config
            length = self._scanner_config.zero_lag_length if self._scanner_config else 21
            band_multiplier = self._scanner_config.zero_lag_band_mult if self._scanner_config else 1.5
            
            # Calculate Zero Lag EMA
            src = df['close']
            lag = int(np.floor((length - 1) / 2))
            
            # Create lagged price series
            src_lagged = src.shift(lag)
            
            # Calculate momentum adjustment
            momentum_adj = src - src_lagged
            
            # Apply Zero Lag formula
            zlema_input = src + momentum_adj
            
            # Calculate EMA on the adjusted input
            df['zlema'] = zlema_input.ewm(span=length).mean()
            
            # Calculate ATR for volatility bands (if not already calculated)
            if 'atr' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())

                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=length).mean()

                df['atr'] = atr

                # Add 20-period ATR and percentile for adaptive volatility
                df['atr_20'] = true_range.rolling(window=20).mean()
                df['atr_percentile'] = (df['atr'] / df['atr_20'] * 100).fillna(50.0)
            else:
                atr = df['atr']

            # Calculate volatility using highest ATR over 3x length period
            volatility_period = min(length * 3, len(df))
            volatility = atr.rolling(window=volatility_period).max() * band_multiplier

            df['volatility'] = volatility
            df['upper_band'] = df['zlema'] + volatility
            df['lower_band'] = df['zlema'] - volatility
            
            # Calculate trend state
            df['trend'] = 0
            trend = np.zeros(len(df))
            
            for i in range(1, len(df)):
                # Carry forward previous trend
                trend[i] = trend[i-1]
                
                # Check for bullish trend change
                if (df['close'].iloc[i] > df['upper_band'].iloc[i] and 
                    df['close'].iloc[i-1] <= df['upper_band'].iloc[i-1]):
                    trend[i] = 1
                
                # Check for bearish trend change
                elif (df['close'].iloc[i] < df['lower_band'].iloc[i] and 
                    df['close'].iloc[i-1] >= df['lower_band'].iloc[i-1]):
                    trend[i] = -1
            
            df['trend'] = trend
            
            # Fill NaN values
            df['zlema'] = df['zlema'].bfill()
            
            self.logger.debug(f"✅ Zero Lag EMA indicators added successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding Zero Lag EMA indicators: {e}")
            return df

    
    def _add_two_pole_oscillator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Two-Pole Oscillator indicators to DataFrame
        Based on BigBeluga's Two-Pole Oscillator for Pine Script
        """
        try:
            # Get configuration parameters from database
            filter_length = self._scanner_config.two_pole_filter_length if self._scanner_config else 20
            sma_length = self._scanner_config.two_pole_sma_length if self._scanner_config else 25
            signal_delay = self._scanner_config.two_pole_signal_delay if self._scanner_config else 4
            
            self.logger.debug(f"🔄 Calculating Two-Pole Oscillator indicators (length={filter_length})")
            
            if len(df) < max(sma_length * 2, filter_length * 2):
                self.logger.debug(f"⚠️ Insufficient data for Two-Pole Oscillator: {len(df)} bars")
                return df
            
            df_enhanced = df.copy()
            
            # Step 1: Calculate SMA(25) and normalize price deviation
            sma1 = df_enhanced['close'].rolling(window=sma_length).mean()
            price_deviation = df_enhanced['close'] - sma1
            
            # Calculate rolling mean and std of the deviation
            deviation_mean = price_deviation.rolling(window=sma_length).mean()
            deviation_std = price_deviation.rolling(window=sma_length).std()
            
            # Normalized deviation (sma_n1 in Pine Script)
            sma_n1 = (price_deviation - deviation_mean) / deviation_std
            sma_n1 = sma_n1.fillna(0)  # Handle NaN values
            
            # Step 2: Apply Two-Pole Smoothing Filter  
            alpha = 2.0 / (filter_length + 1)
            
            # Initialize arrays for the two-pole filter
            smooth1 = np.full(len(df_enhanced), np.nan)
            smooth2 = np.full(len(df_enhanced), np.nan)
            
            # Apply the two-pole filter iteratively
            for i in range(len(df_enhanced)):
                if i == 0:
                    smooth1[i] = sma_n1.iloc[i] if not pd.isna(sma_n1.iloc[i]) else 0
                    smooth2[i] = smooth1[i]
                else:
                    if not pd.isna(sma_n1.iloc[i]):
                        smooth1[i] = (1 - alpha) * smooth1[i-1] + alpha * sma_n1.iloc[i]
                    else:
                        smooth1[i] = smooth1[i-1]
                    
                    smooth2[i] = (1 - alpha) * smooth2[i-1] + alpha * smooth1[i]
            
            # Step 3: Create oscillator values
            df_enhanced['two_pole_osc'] = smooth2
            df_enhanced['two_pole_osc_delayed'] = df_enhanced['two_pole_osc'].shift(signal_delay)
            
            # Step 4: Generate oscillator color/direction
            # Color logic: Green when rising (two_p > two_pp), Purple when falling (two_p <= two_pp)
            df_enhanced['two_pole_is_green'] = df_enhanced['two_pole_osc'] > df_enhanced['two_pole_osc_delayed']
            df_enhanced['two_pole_is_purple'] = df_enhanced['two_pole_osc'] <= df_enhanced['two_pole_osc_delayed']
            
            # Step 5: Generate crossover/crossunder signals
            df_enhanced['two_pole_crossover'] = (
                (df_enhanced['two_pole_osc'] > df_enhanced['two_pole_osc_delayed']) & 
                (df_enhanced['two_pole_osc'].shift(1) <= df_enhanced['two_pole_osc_delayed'].shift(1))
            )
            
            df_enhanced['two_pole_crossunder'] = (
                (df_enhanced['two_pole_osc'] < df_enhanced['two_pole_osc_delayed']) & 
                (df_enhanced['two_pole_osc'].shift(1) >= df_enhanced['two_pole_osc_delayed'].shift(1))
            )
            
            # Step 6: Generate buy/sell signals with CORRECT color-based filtering
            # Buy signals: crossover + in oversold zone + GREEN oscillator (rising)
            df_enhanced['two_pole_buy_signal'] = (
                df_enhanced['two_pole_crossover'] & 
                (df_enhanced['two_pole_osc'] < 0) &  # Oversold zone
                df_enhanced['two_pole_is_green']     # Green (rising) oscillator
            )
            
            # Sell signals: crossunder + in overbought zone + PURPLE oscillator (falling) 
            df_enhanced['two_pole_sell_signal'] = (
                df_enhanced['two_pole_crossunder'] & 
                (df_enhanced['two_pole_osc'] > 0) &  # Overbought zone
                df_enhanced['two_pole_is_purple']    # Purple (falling) oscillator
            )
            
            # Add oscillator strength (absolute value for confidence calculation)
            df_enhanced['two_pole_strength'] = np.abs(df_enhanced['two_pole_osc'])
            
            # Add zone classification
            df_enhanced['two_pole_zone'] = np.where(
                df_enhanced['two_pole_osc'] > 0.5, 'overbought',
                np.where(df_enhanced['two_pole_osc'] < -0.5, 'oversold', 'neutral')
            )
            
            self.logger.debug(f"✅ Two-Pole Oscillator indicators added successfully")
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error adding Two-Pole Oscillator indicators: {e}")
            return df

    def _apply_lazy_indicators(self, df: pd.DataFrame, pair: str, required_indicators: List[str], ema_strategy=None) -> pd.DataFrame:
        """Apply indicators on-demand based on strategy requirements including KAMA"""
        try:
            df_enhanced = df.copy()
            df_enhanced = self._ensure_ema200_always_present(df_enhanced)

            ema_periods = self._get_required_ema_periods(pair, ema_strategy)
            
            # Apply indicators based on requirements
            if 'ema' in required_indicators:
                self.logger.debug(f"🔄 Adding EMA indicators: {ema_periods} (EMA strategy enabled)")
                df_enhanced = self.technical_analyzer.add_ema_indicators(df_enhanced, ema_periods)
            
            if 'macd' in required_indicators:
                self.logger.debug("🔄 Adding MACD indicators")
                macd_fast = self._scanner_config.macd_fast_period if self._scanner_config else 12
                macd_slow = self._scanner_config.macd_slow_period if self._scanner_config else 26
                macd_signal = self._scanner_config.macd_signal_period if self._scanner_config else 9
                df_enhanced = self.technical_analyzer.add_macd_indicators(
                    df_enhanced,
                    macd_fast,
                    macd_slow,
                    macd_signal
                )

                # Ensure EMA 50, 100, 200 for trend filtering
                if 'ema_50' not in df_enhanced.columns:
                    df_enhanced['ema_50'] = df_enhanced['close'].ewm(span=50).mean()
                if 'ema_100' not in df_enhanced.columns:
                    df_enhanced['ema_100'] = df_enhanced['close'].ewm(span=100).mean()
                if 'ema_200' not in df_enhanced.columns:
                    df_enhanced['ema_200'] = df_enhanced['close'].ewm(span=200).mean()

            if 'kama' in required_indicators:
                self.logger.debug("🔄 Adding KAMA indicators")
                df_enhanced = self._add_kama_indicators(df_enhanced)

            if 'support_resistance' in required_indicators:
                self.logger.debug("🔄 Adding S/R levels")
                df_enhanced = self.technical_analyzer.add_support_resistance_to_df(df_enhanced, pair)

            if 'volume' in required_indicators:
                self.logger.debug("🔄 Adding volume analysis")
                df_enhanced = self.volume_analyzer.add_volume_analysis(df_enhanced)

            if 'behavior' in required_indicators:
                self.logger.debug("🔄 Adding behavior analysis")
                df_enhanced = self.behavior_analyzer.add_behavior_analysis(df_enhanced, pair)

            if 'zero_lag_ema' in required_indicators:
                self.logger.debug("🔄 Adding Zero Lag EMA indicators")
                df_enhanced = self._add_zero_lag_indicators(df_enhanced)

            if 'bb_supertrend' in required_indicators:
                self.logger.debug("🔄 Adding BB+Supertrend indicators")
                df_enhanced = self._add_bb_supertrend_indicators(df_enhanced)

            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"❌ Error in lazy indicator loading: {e}")
            return df
    
    def _get_optimal_lookback_hours(self, epic: str, timeframe: str, ema_strategy=None) -> int:
        """Get optimal lookback hours including consideration for KAMA requirements"""
        try:
            # Base lookback hours by timeframe
            base_lookback = {
                '1m': 24,    # 1 day for 1m (1440 bars)
                '5m': 48,    # 2 days for 5m (576 bars)
                '15m': 168,  # 1 week for 15m (672 bars)
                '1h': 720,   # 1 month for 1h (720 bars)
                '1d': 8760   # 1 year for 1d (365 bars)
            }.get(timeframe, 48)
            
            # Adjust for dynamic EMA requirements
            if ema_strategy:
                ema_periods = self._get_required_ema_periods(epic, ema_strategy)
                max_ema = max(ema_periods) if ema_periods else 200
                
                # Ensure we have enough data for the largest EMA (at least 3x the period)
                min_bars_needed = max_ema * 3
                timeframe_minutes = self._timeframe_to_minutes(timeframe)
                min_hours_needed = (min_bars_needed * timeframe_minutes) / 60
                
                base_lookback = max(base_lookback, int(min_hours_needed))
            
            # Always adjust for KAMA requirements (for performance collection)
            # KAMA period is now in database (no config.py fallback)
            kama_period = self._scanner_config.kama_period

            # KAMA needs more data for stable calculation (5x the period minimum)
            min_kama_bars = kama_period * 5
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            min_kama_hours = (min_kama_bars * timeframe_minutes) / 60

            base_lookback = max(base_lookback, int(min_kama_hours))
            self.logger.debug(f"📊 KAMA lookback adjustment: {int(min_kama_hours)} hours for period {kama_period}")
            
            # Apply reduction factor if enabled (database-only - no fallback)
            if self.reduced_lookback:
                reduction_factor = self._scanner_config.lookback_reduction_factor
                base_lookback = int(base_lookback * reduction_factor)
            
            self.logger.debug(f"📈 Optimal lookback for {epic} ({timeframe}): {base_lookback} hours")
            return base_lookback
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal lookback: {e}")
            return 48  # Safe default
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        return timeframe_map.get(timeframe, 5)
    
    def _get_required_ema_periods(self, epic: str, ema_strategy=None) -> List[int]:
        """
        CRITICAL FIX: Get required EMA periods from the EMA strategy with optimal parameter support

        This method now properly detects and uses optimal parameters to ensure
        BacktestDataFetcher and EMA strategy use the same EMA configuration.
        """
        try:
            # PRIORITY 1: Check for optimal parameters first (backtest consistency)
            if epic and ema_strategy and getattr(ema_strategy, 'use_optimal_parameters', False):
                try:
                    from optimization.optimal_parameter_service import get_epic_ema_config
                    optimal_config = get_epic_ema_config(epic)
                    if optimal_config:
                        ema_periods = [
                            optimal_config['short'],
                            optimal_config['long'],
                            optimal_config['trend']
                        ]
                        self.logger.debug(f"🎯 Using OPTIMAL EMA periods for {epic}: {ema_periods} (backtest consistency)")
                        return ema_periods
                except Exception as e:
                    self.logger.warning(f"⚠️ Optimal EMA config failed for {epic}: {e}, falling back to strategy config")

            # PRIORITY 2: Use EMA strategy's own _get_ema_periods method
            if ema_strategy and hasattr(ema_strategy, '_get_ema_periods'):
                try:
                    # Get configuration from the strategy itself
                    ema_config = ema_strategy._get_ema_periods(epic)

                    if ema_config:
                        ema_periods = [
                            ema_config.get('short', 21),
                            ema_config.get('long', 50),
                            ema_config.get('trend', 200)
                        ]

                        # Determine if this is optimal or static
                        config_mode = 'optimal' if getattr(ema_strategy, 'use_optimal_parameters', False) else 'static'
                        self.logger.debug(f"📊 Using {config_mode} EMA config from strategy for {epic}: {ema_periods}")
                        return ema_periods

                except Exception as e:
                    self.logger.warning(f"⚠️ Strategy _get_ema_periods failed for {epic}: {e}")
                    # Fall through to configdata

            # PRIORITY 3: Use strategy's ema_config attribute if available
            if ema_strategy and hasattr(ema_strategy, 'ema_config'):
                try:
                    ema_config = ema_strategy.ema_config
                    if ema_config and isinstance(ema_config, dict):
                        ema_periods = [
                            ema_config.get('short', 21),
                            ema_config.get('long', 50),
                            ema_config.get('trend', 200)
                        ]
                        self.logger.debug(f"📊 Using strategy ema_config attribute for {epic}: {ema_periods}")
                        return ema_periods
                except Exception as e:
                    self.logger.warning(f"⚠️ Strategy ema_config attribute failed: {e}")

            # FINAL: Standard EMA periods for trend filtering
            # These are used by all strategies for trend direction (EMA disabled, SMC uses these)
            default_periods = [21, 50, 200]
            self.logger.debug(f"📊 Using standard EMA periods for {epic}: {default_periods}")
            return default_periods

        except Exception as e:
            self.logger.error(f"❌ Error getting EMA periods for {epic}: {e}")
            # Ultimate fallback
            return [21, 50, 200]

    def _ensure_ema200_always_present(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALWAYS ensure EMA 50, 100, 200 are present for trend filtering

        This separates the concern of trend filtering from dynamic EMA periods
        used for signal generation (ema_short, ema_long, ema_trend)

        Args:
            df: DataFrame to enhance

        Returns:
            DataFrame with guaranteed ema_50, ema_100, ema_200 columns
        """
        try:
            df_enhanced = df.copy()

            # ALWAYS ensure EMA 50, 100, 200 exist for trend filtering
            if 'ema_50' not in df_enhanced.columns:
                self.logger.debug("🔄 Adding EMA 50 for trend filtering (always present)")
                df_enhanced['ema_50'] = df_enhanced['close'].ewm(span=50).mean()
            else:
                self.logger.debug("✅ EMA 50 already present")

            if 'ema_100' not in df_enhanced.columns:
                self.logger.debug("🔄 Adding EMA 100 for trend filtering (always present)")
                df_enhanced['ema_100'] = df_enhanced['close'].ewm(span=100).mean()
            else:
                self.logger.debug("✅ EMA 100 already present")

            if 'ema_200' not in df_enhanced.columns:
                self.logger.debug("🔄 Adding EMA 200 for trend filtering (always present)")
                df_enhanced['ema_200'] = df_enhanced['close'].ewm(span=200).mean()
            else:
                self.logger.debug("✅ EMA 200 already present")

            return df_enhanced

        except Exception as e:
            self.logger.error(f"❌ Error ensuring EMAs: {e}")
            return df

    
    def _resample_to_15m_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced 15m resampling optimized for real-time trading
        
        Key improvements:
        - Detects incomplete periods for current candle awareness
        - Validates data completeness for trading accuracy  
        - Adds confidence scores for each 15m candle
        - Handles real-time vs historical data differently
        """
        try:
            # Validate input data
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for resampling")
                return df
                
            if len(df) < 3:
                self.logger.warning("⚠️ Insufficient 5m data for 15m resampling (need at least 3 bars)")
                return df
            
            # Ensure start_time is datetime and timezone-aware
            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df
                
            # Sort by time to ensure proper order
            df = df.sort_values('start_time')
            
            # Create indexed dataframe
            df_indexed = df.set_index('start_time')
            
            # Standard resampling
            df_15m = df_indexed.resample(
                '15min', 
                label='left',      
                closed='left',     
                origin='epoch'     
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'ltv': 'sum'
            })
            
            # Add data completeness analysis
            def calculate_completeness(group):
                """Calculate how complete each 15m period is"""
                expected_candles = 3  # 3 x 5m candles per 15m period
                actual_candles = len(group)
                return {
                    'actual_5m_candles': actual_candles,
                    'expected_5m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles == expected_candles,
                    'missing_minutes': (expected_candles - actual_candles) * 5
                }
            
            # Calculate completeness for each 15m period
            completeness_data = []
            for timestamp in df_15m.index:
                # Get 5m candles that belong to this 15m period
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(minutes=15)
                
                period_candles = df_indexed[
                    (df_indexed.index >= period_start) & 
                    (df_indexed.index < period_end)
                ]
                
                completeness = calculate_completeness(period_candles)
                completeness['period_start'] = timestamp
                completeness_data.append(completeness)
            
            # Convert to DataFrame and merge
            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_15m = df_15m.join(completeness_df)
            
            # Add trading confidence score
            def calculate_trading_confidence(row):
                """
                Calculate confidence score for trading decisions
                100% = complete data, perfect for trading
                90%+ = minor gaps, good for trading  
                80%+ = some gaps, caution advised
                <80% = significant gaps, avoid trading
                """
                base_score = row['completeness_ratio'] * 100
                
                # Penalize if missing recent data (affects current market state)
                try:
                    # Get current time in same timezone as the data
                    current_time = pd.Timestamp.now()
                    if row.name.tz is not None:
                        # If data is timezone-aware, make current_time timezone-aware too
                        if current_time.tz is None:
                            current_time = current_time.tz_localize('UTC')
                        current_time = current_time.tz_convert(row.name.tz)
                    elif current_time.tz is not None:
                        # If data is timezone-naive but current_time is aware, make it naive
                        current_time = current_time.tz_localize(None)
                    
                    if current_time - row.name < pd.Timedelta(minutes=20):
                        # Current or very recent candle - penalize incomplete data more
                        if not row['is_complete']:
                            base_score *= 0.8  # 20% penalty for incomplete current data
                except Exception as e:
                    # If timezone handling fails, skip the recent data penalty
                    pass
                
                # Boost score if we have the close (most important for signals)
                if pd.notna(row['close']):
                    base_score = min(100, base_score * 1.05)  # 5% bonus for having close
                
                return round(base_score, 1)
            
            df_15m['trading_confidence'] = df_15m.apply(calculate_trading_confidence, axis=1)
            
            # Add trading suitability flags
            df_15m['suitable_for_entry'] = df_15m['trading_confidence'] >= 90.0
            df_15m['suitable_for_analysis'] = df_15m['trading_confidence'] >= 80.0
            df_15m['data_warning'] = df_15m['trading_confidence'] < 85.0
            
            # Only drop rows where ALL OHLC values are missing
            df_15m = df_15m.dropna(subset=['open', 'high', 'low', 'close'], how='all')
            
            # Reset index to get start_time back as column
            df_15m_reset = df_15m.reset_index()
            
            # Log quality metrics
            total_candles = len(df_15m_reset)
            complete_candles = len(df_15m_reset[df_15m_reset['is_complete']])
            high_confidence = len(df_15m_reset[df_15m_reset['trading_confidence'] >= 90])
            
            self.logger.debug(f"✅ 15m synthesis quality: {complete_candles}/{total_candles} complete ({complete_candles/total_candles*100:.1f}%), {high_confidence} high-confidence")
            
            # Warning for recent incomplete data
            try:
                current_time_for_filter = pd.Timestamp.now()
                if len(df_15m_reset) > 0 and df_15m_reset['start_time'].dt.tz is not None:
                    current_time_for_filter = current_time_for_filter.tz_localize('UTC').tz_convert(df_15m_reset['start_time'].dt.tz)
                elif len(df_15m_reset) > 0 and current_time_for_filter.tz is not None:
                    current_time_for_filter = current_time_for_filter.tz_localize(None)
                
                recent_candles = df_15m_reset[
                    df_15m_reset['start_time'] >= current_time_for_filter - pd.Timedelta(hours=2)
                ]
                incomplete_recent = recent_candles[~recent_candles['is_complete']]
                
                if len(incomplete_recent) > 0:
                    self.logger.debug(f"⚠️ {len(incomplete_recent)} incomplete 15m candles in last 2 hours")
            except Exception as e:
                self.logger.debug(f"Could not check recent incomplete candles: {e}")
            
            # Add timezone columns if they were in original data
            timezone_columns = ['local_time', 'market_session', 'user_time']
            for col in timezone_columns:
                if col in df.columns:
                    try:
                        df_with_tz = df.set_index('start_time')
                        tz_resampled = df_with_tz[col].resample(
                            '15min', label='left', closed='left', origin='epoch'
                        ).first()
                        df_15m_reset[col] = df_15m_reset['start_time'].map(tz_resampled).ffill()
                    except Exception as tz_error:
                        self.logger.debug(f"⚠️ Could not resample timezone column {col}: {tz_error}")
            
            return df_15m_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 15m resampling: {e}")
            return df

    def _resample_to_60m_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced 60m resampling optimized for real-time trading

        Key improvements:
        - Detects incomplete periods for current candle awareness
        - Validates data completeness for trading accuracy
        - Adds confidence scores for each 60m candle
        - Handles real-time vs historical data differently
        """
        try:
            # Validate input data
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for resampling")
                return df

            if len(df) < 12:
                self.logger.warning("⚠️ Insufficient 5m data for 60m resampling (need at least 12 bars)")
                return df

            # Ensure start_time is datetime and timezone-aware
            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df

            # Sort by time to ensure proper order
            df = df.sort_values('start_time')

            # Create indexed dataframe
            df_indexed = df.set_index('start_time')

            # Standard resampling
            df_60m = df_indexed.resample(
                '60min',
                label='left',
                closed='left',
                origin='epoch'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'ltv': 'sum'
            })

            # Add data completeness analysis
            def calculate_completeness(group):
                """Calculate how complete each 60m period is"""
                expected_candles = 12  # 12 x 5m candles per 60m period
                actual_candles = len(group)
                return {
                    'actual_5m_candles': actual_candles,
                    'expected_5m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles == expected_candles,
                    'missing_minutes': (expected_candles - actual_candles) * 5
                }

            # Calculate completeness for each 60m period
            completeness_data = []
            for timestamp in df_60m.index:
                # Get 5m candles that belong to this 60m period
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(minutes=60)

                period_candles = df_indexed[
                    (df_indexed.index >= period_start) &
                    (df_indexed.index < period_end)
                ]

                completeness = calculate_completeness(period_candles)
                completeness['period_start'] = timestamp
                completeness_data.append(completeness)

            # Convert to DataFrame and merge
            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_60m = df_60m.join(completeness_df)

            # Add trading confidence score
            def calculate_trading_confidence(row):
                """
                Calculate confidence score for trading decisions
                100% = complete data, perfect for trading
                90%+ = minor gaps, good for trading
                80%+ = some gaps, caution advised
                <80% = significant gaps, avoid trading
                """
                base_score = row['completeness_ratio'] * 100

                # Penalize if missing recent data (affects current market state)
                try:
                    # Get current time in same timezone as the data
                    current_time = pd.Timestamp.now()
                    if row.name.tz is not None:
                        # If data is timezone-aware, make current_time timezone-aware too
                        if current_time.tz is None:
                            current_time = current_time.tz_localize('UTC')
                        current_time = current_time.tz_convert(row.name.tz)
                    elif current_time.tz is not None:
                        # If data is timezone-naive but current_time is aware, make it naive
                        current_time = current_time.tz_localize(None)

                    if current_time - row.name < pd.Timedelta(hours=2):
                        # Current or very recent candle - penalize incomplete data more
                        if not row['is_complete']:
                            base_score *= 0.8  # 20% penalty for incomplete current data
                except Exception as e:
                    # If timezone handling fails, skip the recent data penalty
                    pass

                # Boost score if we have the close (most important for signals)
                if pd.notna(row['close']):
                    base_score = min(100, base_score * 1.05)  # 5% bonus for having close

                return round(base_score, 1)

            df_60m['trading_confidence'] = df_60m.apply(calculate_trading_confidence, axis=1)

            # Add trading suitability flags
            df_60m['suitable_for_entry'] = df_60m['trading_confidence'] >= 90.0
            df_60m['suitable_for_analysis'] = df_60m['trading_confidence'] >= 80.0
            df_60m['data_warning'] = df_60m['trading_confidence'] < 85.0

            # Only drop rows where ALL OHLC values are missing
            df_60m = df_60m.dropna(subset=['open', 'high', 'low', 'close'], how='all')

            # Reset index to get start_time back as column
            df_60m_reset = df_60m.reset_index()

            # Log quality metrics
            total_candles = len(df_60m_reset)
            complete_candles = len(df_60m_reset[df_60m_reset['is_complete']])
            high_confidence = len(df_60m_reset[df_60m_reset['trading_confidence'] >= 90])

            self.logger.debug(f"✅ 60m synthesis quality: {complete_candles}/{total_candles} complete ({complete_candles/total_candles*100:.1f}%), {high_confidence} high-confidence")

            # Warning for recent incomplete data
            try:
                current_time_for_filter = pd.Timestamp.now()
                if len(df_60m_reset) > 0 and df_60m_reset['start_time'].dt.tz is not None:
                    current_time_for_filter = current_time_for_filter.tz_localize('UTC').tz_convert(df_60m_reset['start_time'].dt.tz)
                elif len(df_60m_reset) > 0 and current_time_for_filter.tz is not None:
                    current_time_for_filter = current_time_for_filter.tz_localize(None)

                recent_candles = df_60m_reset[
                    df_60m_reset['start_time'] >= current_time_for_filter - pd.Timedelta(hours=4)
                ]
                incomplete_recent = recent_candles[~recent_candles['is_complete']]

                if len(incomplete_recent) > 0:
                    self.logger.debug(f"⚠️ {len(incomplete_recent)} incomplete 60m candles in last 4 hours")
            except Exception as e:
                self.logger.debug(f"Could not check recent incomplete candles: {e}")

            # Add timezone columns if they were in original data
            timezone_columns = ['local_time', 'market_session', 'user_time']
            for col in timezone_columns:
                if col in df.columns:
                    try:
                        df_with_tz = df.set_index('start_time')
                        tz_resampled = df_with_tz[col].resample(
                            '60min', label='left', closed='left', origin='epoch'
                        ).first()
                        df_60m_reset[col] = df_60m_reset['start_time'].map(tz_resampled).ffill()
                    except Exception as tz_error:
                        self.logger.debug(f"⚠️ Could not resample timezone column {col}: {tz_error}")

            return df_60m_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 60m resampling: {e}")
            return df

    def _resample_to_4h_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced 4H resampling optimized for multi-timeframe analysis

        Resamples 5m data to 4-hour candles for higher timeframe trend analysis.
        """
        try:
            # Validate input data
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for 4H resampling")
                return df

            if len(df) < 48:
                self.logger.warning("⚠️ Insufficient 5m data for 4H resampling (need at least 48 bars)")
                return df

            # Ensure start_time is datetime
            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df

            # Sort by time to ensure proper order
            df = df.sort_values('start_time')

            # Create indexed dataframe
            df_indexed = df.set_index('start_time')

            # Resample to 4H
            df_4h = df_indexed.resample(
                '4h',
                label='left',
                closed='left',
                origin='epoch'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'ltv': 'sum'
            })

            # Add data completeness analysis
            def calculate_completeness(group):
                """Calculate how complete each 4H period is"""
                expected_candles = 48  # 48 x 5m candles per 4H period
                actual_candles = len(group)
                return {
                    'actual_5m_candles': actual_candles,
                    'expected_5m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles >= 45,  # Allow 95% complete
                    'missing_minutes': (expected_candles - actual_candles) * 5
                }

            # Calculate completeness for each 4H period
            completeness_data = []
            for timestamp in df_4h.index:
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(hours=4)

                period_candles = df_indexed[
                    (df_indexed.index >= period_start) &
                    (df_indexed.index < period_end)
                ]

                completeness = calculate_completeness(period_candles)
                completeness['period_start'] = timestamp
                completeness_data.append(completeness)

            # Convert to DataFrame and merge
            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_4h = df_4h.join(completeness_df)

            # Add trading confidence score
            df_4h['trading_confidence'] = (df_4h['completeness_ratio'] * 100).round(1)
            df_4h['suitable_for_analysis'] = df_4h['trading_confidence'] >= 80.0

            # Drop rows where ALL OHLC values are missing
            df_4h = df_4h.dropna(subset=['open', 'high', 'low', 'close'], how='all')

            # Reset index to get start_time back as column
            df_4h_reset = df_4h.reset_index()

            # Log quality metrics
            total_candles = len(df_4h_reset)
            complete_candles = len(df_4h_reset[df_4h_reset['is_complete']])

            self.logger.debug(f"✅ 4h synthesis quality: {complete_candles}/{total_candles} complete ({complete_candles/total_candles*100:.1f}%)")

            return df_4h_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 4H resampling: {e}")
            return df

    # =========================================================================
    # 1M-BASED RESAMPLING FUNCTIONS
    # =========================================================================
    # These functions synthesize higher timeframes from 1-minute base candles.
    # Benefits over 5m base:
    # - Better gap resilience (1 missing = 6.7% vs 33% data loss)
    # - Fresher data (max 1m stale vs 5m)
    # - Precise boundary detection (1m granularity)
    # Enable via config.USE_1M_BASE_SYNTHESIS = True
    # =========================================================================

    def _resample_to_15m_from_1m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize 15m candles from 1m base data (15 x 1m = 1 x 15m)

        This provides better gap resilience than 5m-based synthesis:
        - 5m base: 1 missing candle = 33% data loss
        - 1m base: 1 missing candle = 6.7% data loss
        """
        try:
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for 1m→15m resampling")
                return df

            if len(df) < 15:
                self.logger.warning("⚠️ Insufficient 1m data for 15m resampling (need at least 15 bars)")
                return df

            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df

            df = df.sort_values('start_time')
            df_indexed = df.set_index('start_time')

            # Resample 1m → 15m
            df_15m = df_indexed.resample(
                '15min',
                label='left',
                closed='left',
                origin='epoch'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'ltv': 'sum'
            })

            # Calculate completeness for each 15m period
            expected_candles = 15  # 15 x 1m candles per 15m period
            completeness_data = []

            for timestamp in df_15m.index:
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(minutes=15)

                period_candles = df_indexed[
                    (df_indexed.index >= period_start) &
                    (df_indexed.index < period_end)
                ]

                actual_candles = len(period_candles)
                completeness_data.append({
                    'period_start': timestamp,
                    'actual_1m_candles': actual_candles,
                    'expected_1m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles == expected_candles,
                    'missing_minutes': expected_candles - actual_candles
                })

            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_15m = df_15m.join(completeness_df)

            # Add trading confidence score
            def calculate_trading_confidence(row):
                base_score = row['completeness_ratio'] * 100
                try:
                    current_time = pd.Timestamp.now()
                    if row.name.tz is not None:
                        if current_time.tz is None:
                            current_time = current_time.tz_localize('UTC')
                        current_time = current_time.tz_convert(row.name.tz)
                    elif current_time.tz is not None:
                        current_time = current_time.tz_localize(None)

                    if current_time - row.name < pd.Timedelta(minutes=20):
                        if not row['is_complete']:
                            base_score *= 0.8
                except Exception:
                    pass

                if pd.notna(row['close']):
                    base_score = min(100, base_score * 1.05)

                return round(base_score, 1)

            df_15m['trading_confidence'] = df_15m.apply(calculate_trading_confidence, axis=1)
            df_15m['suitable_for_entry'] = df_15m['trading_confidence'] >= 90.0
            df_15m['suitable_for_analysis'] = df_15m['trading_confidence'] >= 80.0
            df_15m['data_warning'] = df_15m['trading_confidence'] < 85.0

            df_15m = df_15m.dropna(subset=['open', 'high', 'low', 'close'], how='all')
            df_15m_reset = df_15m.reset_index()

            # Log quality metrics
            total_candles = len(df_15m_reset)
            if total_candles > 0:
                complete_candles = len(df_15m_reset[df_15m_reset['is_complete']])
                high_confidence = len(df_15m_reset[df_15m_reset['trading_confidence'] >= 90])
                self.logger.debug(f"✅ 15m from 1m synthesis: {complete_candles}/{total_candles} complete ({complete_candles/total_candles*100:.1f}%), {high_confidence} high-confidence")

            return df_15m_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 1m→15m resampling: {e}")
            return df

    def _resample_to_60m_from_1m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize 1h candles from 1m base data (60 x 1m = 1 x 1h)
        """
        try:
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for 1m→60m resampling")
                return df

            if len(df) < 60:
                self.logger.warning("⚠️ Insufficient 1m data for 60m resampling (need at least 60 bars)")
                return df

            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df

            df = df.sort_values('start_time')
            df_indexed = df.set_index('start_time')

            # Resample 1m → 60m
            df_60m = df_indexed.resample(
                '60min',
                label='left',
                closed='left',
                origin='epoch'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'ltv': 'sum'
            })

            # Calculate completeness
            expected_candles = 60
            completeness_data = []

            for timestamp in df_60m.index:
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(hours=1)

                period_candles = df_indexed[
                    (df_indexed.index >= period_start) &
                    (df_indexed.index < period_end)
                ]

                actual_candles = len(period_candles)
                completeness_data.append({
                    'period_start': timestamp,
                    'actual_1m_candles': actual_candles,
                    'expected_1m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles >= 57,  # Allow 95% complete
                    'missing_minutes': expected_candles - actual_candles
                })

            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_60m = df_60m.join(completeness_df)

            df_60m['trading_confidence'] = (df_60m['completeness_ratio'] * 100).round(1)
            df_60m['suitable_for_analysis'] = df_60m['trading_confidence'] >= 80.0

            df_60m = df_60m.dropna(subset=['open', 'high', 'low', 'close'], how='all')
            df_60m_reset = df_60m.reset_index()

            total_candles = len(df_60m_reset)
            if total_candles > 0:
                complete_candles = len(df_60m_reset[df_60m_reset['is_complete']])
                self.logger.debug(f"✅ 60m from 1m synthesis: {complete_candles}/{total_candles} complete ({complete_candles/total_candles*100:.1f}%)")

            return df_60m_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 1m→60m resampling: {e}")
            return df

    def _resample_to_4h_from_1m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize 4h candles from 1m base data (240 x 1m = 1 x 4h)
        """
        try:
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for 1m→4h resampling")
                return df

            if len(df) < 240:
                self.logger.warning("⚠️ Insufficient 1m data for 4h resampling (need at least 240 bars)")
                return df

            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df

            df = df.sort_values('start_time')
            df_indexed = df.set_index('start_time')

            # Resample 1m → 4h
            df_4h = df_indexed.resample(
                '4h',
                label='left',
                closed='left',
                origin='epoch'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'ltv': 'sum'
            })

            # Calculate completeness
            expected_candles = 240
            completeness_data = []

            for timestamp in df_4h.index:
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(hours=4)

                period_candles = df_indexed[
                    (df_indexed.index >= period_start) &
                    (df_indexed.index < period_end)
                ]

                actual_candles = len(period_candles)
                completeness_data.append({
                    'period_start': timestamp,
                    'actual_1m_candles': actual_candles,
                    'expected_1m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles >= 228,  # Allow 95% complete
                    'missing_minutes': expected_candles - actual_candles
                })

            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_4h = df_4h.join(completeness_df)

            df_4h['trading_confidence'] = (df_4h['completeness_ratio'] * 100).round(1)
            df_4h['suitable_for_analysis'] = df_4h['trading_confidence'] >= 80.0

            df_4h = df_4h.dropna(subset=['open', 'high', 'low', 'close'], how='all')
            df_4h_reset = df_4h.reset_index()

            total_candles = len(df_4h_reset)
            if total_candles > 0:
                complete_candles = len(df_4h_reset[df_4h_reset['is_complete']])
                self.logger.debug(f"✅ 4h from 1m synthesis: {complete_candles}/{total_candles} complete ({complete_candles/total_candles*100:.1f}%)")

            return df_4h_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 1m→4h resampling: {e}")
            return df

    def _resample_to_5m_from_1m(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Synthesize 5m candles from 1m base data (5 x 1m = 1 x 5m)

        This allows complete synthesis chain: 1m → 5m → (validation against streamed 5m)
        """
        try:
            if df is None or len(df) == 0:
                self.logger.warning("⚠️ Empty dataframe provided for 1m→5m resampling")
                return df

            if len(df) < 5:
                self.logger.warning("⚠️ Insufficient 1m data for 5m resampling (need at least 5 bars)")
                return df

            if 'start_time' not in df.columns:
                self.logger.error("❌ Missing 'start_time' column for resampling")
                return df

            df = df.sort_values('start_time')
            df_indexed = df.set_index('start_time')

            # Resample 1m → 5m
            df_5m = df_indexed.resample(
                '5min',
                label='left',
                closed='left',
                origin='epoch'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'ltv': 'sum'
            })

            # Calculate completeness
            expected_candles = 5
            completeness_data = []

            for timestamp in df_5m.index:
                period_start = timestamp
                period_end = timestamp + pd.Timedelta(minutes=5)

                period_candles = df_indexed[
                    (df_indexed.index >= period_start) &
                    (df_indexed.index < period_end)
                ]

                actual_candles = len(period_candles)
                completeness_data.append({
                    'period_start': timestamp,
                    'actual_1m_candles': actual_candles,
                    'expected_1m_candles': expected_candles,
                    'completeness_ratio': actual_candles / expected_candles,
                    'is_complete': actual_candles == expected_candles,
                    'missing_minutes': expected_candles - actual_candles
                })

            completeness_df = pd.DataFrame(completeness_data).set_index('period_start')
            df_5m = df_5m.join(completeness_df)

            df_5m['trading_confidence'] = (df_5m['completeness_ratio'] * 100).round(1)
            df_5m['suitable_for_analysis'] = df_5m['trading_confidence'] >= 80.0

            df_5m = df_5m.dropna(subset=['open', 'high', 'low', 'close'], how='all')
            df_5m_reset = df_5m.reset_index()

            return df_5m_reset

        except Exception as e:
            self.logger.error(f"❌ Error in 1m→5m resampling: {e}")
            return df

    def get_candle_boundary_info(self, timeframe: str = '15m') -> dict:
        """
        Get information about candle boundary timing for scan alignment.

        15m boundaries: :00, :15, :30, :45 minutes past hour
        1h boundaries: :00 minutes past hour
        4h boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC

        Returns:
            dict: {
                'is_at_boundary': bool,      # Within offset of boundary
                'minutes_since_close': int,  # Minutes since last boundary
                'seconds_until_next': int,   # Seconds until next boundary
                'last_boundary': datetime,   # Last candle close time
                'next_boundary': datetime    # Next candle close time
            }
        """
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        current_minute = now.minute
        current_hour = now.hour

        if timeframe == '15m':
            # 15m boundaries at :00, :15, :30, :45
            boundary_minute = (current_minute // 15) * 15
            last_boundary = now.replace(minute=boundary_minute, second=0, microsecond=0)
            next_boundary = last_boundary + timedelta(minutes=15)

        elif timeframe == '1h' or timeframe == '60m':
            # 1h boundaries at :00
            last_boundary = now.replace(minute=0, second=0, microsecond=0)
            next_boundary = last_boundary + timedelta(hours=1)

        elif timeframe == '4h':
            # 4h boundaries at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
            boundary_hour = (current_hour // 4) * 4
            last_boundary = now.replace(hour=boundary_hour, minute=0, second=0, microsecond=0)
            next_boundary = last_boundary + timedelta(hours=4)

        else:
            # Default to 15m
            boundary_minute = (current_minute // 15) * 15
            last_boundary = now.replace(minute=boundary_minute, second=0, microsecond=0)
            next_boundary = last_boundary + timedelta(minutes=15)

        minutes_since_close = (now - last_boundary).total_seconds() / 60
        seconds_until_next = (next_boundary - now).total_seconds()

        # Consider "at boundary" if within configured offset (from database - NO FALLBACK)
        if self._scanner_config:
            offset = self._scanner_config.scan_boundary_offset_seconds
        else:
            # Hardcoded default if no config available (should not happen in normal operation)
            offset = 60
        is_at_boundary = minutes_since_close * 60 <= offset

        return {
            'is_at_boundary': is_at_boundary,
            'minutes_since_close': int(minutes_since_close),
            'seconds_until_next': int(seconds_until_next),
            'last_boundary': last_boundary,
            'next_boundary': next_boundary
        }

    def should_trade_on_candle(self, candle_row) -> Dict[str, any]:
        """
        Determine if a 15m candle is suitable for trading decisions
        
        Returns:
            dict: {
                'can_trade': bool,
                'confidence': float,
                'reason': str,
                'warnings': List[str]
            }
        """
        warnings = []
        
        # Check data completeness
        if candle_row.get('trading_confidence', 0) < 90:
            if candle_row.get('trading_confidence', 0) < 80:
                return {
                    'can_trade': False,
                    'confidence': candle_row.get('trading_confidence', 0),
                    'reason': 'Insufficient data quality for trading',
                    'warnings': [f"Only {candle_row.get('actual_5m_candles', 0)}/3 source candles available"]
                }
            else:
                warnings.append("Lower confidence due to missing 5m data")
        
        # Check if current incomplete candle - with timezone handling
        try:
            candle_time = pd.Timestamp(candle_row['start_time'])
            current_time = pd.Timestamp.now()
            
            # Handle timezone compatibility
            if candle_time.tz is not None and current_time.tz is None:
                current_time = current_time.tz_localize('UTC').tz_convert(candle_time.tz)
            elif candle_time.tz is None and current_time.tz is not None:
                current_time = current_time.tz_localize(None)
                
            time_since = current_time - candle_time
            
            if time_since < pd.Timedelta(minutes=15) and not candle_row.get('is_complete', False):
                warnings.append("Current candle incomplete - price may change")
        except Exception as e:
            # If timezone handling fails, skip this check
            pass
        
        # Check OHLC validity
        ohlc_valid = (
            candle_row['low'] <= candle_row['open'] <= candle_row['high'] and
            candle_row['low'] <= candle_row['close'] <= candle_row['high']
        )
        
        if not ohlc_valid:
            return {
                'can_trade': False,
                'confidence': 0,
                'reason': 'Invalid OHLC relationship detected',
                'warnings': ['Data integrity issue - do not trade']
            }
        
        return {
            'can_trade': True,
            'confidence': candle_row.get('trading_confidence', 100),
            'reason': 'Data quality sufficient for trading',
            'warnings': warnings
        }
    
    def _get_cache_key(self, epic: str, timeframe: str, lookback_hours: int, ema_periods: List[int]) -> str:
        """
        ENHANCED: Generate cache key that includes EMA periods for proper cache isolation
        
        This ensures that different EMA configurations don't share cached data.
        """
        try:
            # Include EMA periods in cache key to ensure proper isolation
            ema_key = "_".join(map(str, ema_periods))
            base_key = f"{epic}_{timeframe}_{lookback_hours}_{ema_key}"
            
            # Add timestamp for cache expiration (5 minutes)
            import time
            cache_window = int(time.time() // 300)  # 5-minute windows
            
            return f"{base_key}_{cache_window}"
            
        except Exception as e:
            self.logger.error(f"❌ Error generating cache key: {e}")
            # Fallback cache key
            return f"{epic}_{timeframe}_{lookback_hours}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._data_cache:
            return False
        
        cache_entry = self._data_cache[cache_key]
        cache_age = (datetime.now() - cache_entry['timestamp']).total_seconds()
        
        return cache_age < self._cache_timeout
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache enhanced data"""
        try:
            self._data_cache[cache_key] = {
                'data': data.copy(),
                'timestamp': datetime.now()
            }
            
            # Clean old cache entries
            current_time = datetime.now()
            keys_to_remove = [
                key for key, entry in self._data_cache.items()
                if (current_time - entry['timestamp']).total_seconds() > self._cache_timeout
            ]
            
            for key in keys_to_remove:
                del self._data_cache[key]
                
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
    
    def _fetch_candle_data_optimized(
        self, 
        epic: str, 
        timeframe: str, 
        lookback_hours: int, 
        tz_manager: TimezoneManager
    ) -> Optional[pd.DataFrame]:
        """
        FIXED: Optimized candle data fetching with proper 15m resampling completion
        """
        
        # Convert timeframe to database format
        timeframe_map = {
            '5m': 5,
            '15m': 15,
            '1h': 60
        }
        
        tf_minutes = timeframe_map.get(timeframe, 5)
        
        # Calculate lookback time in UTC (database time)
        since_utc = tz_manager.get_lookback_time_utc(lookback_hours)
        
        # Determine source timeframe based on synthesis mode (from database - NO FALLBACK)
        # USE_1M_BASE_SYNTHESIS: Use 1m candles as base (better gap resilience)
        # Default (False): Use 5m candles as base (current behavior)
        if self._scanner_config:
            use_1m_base = self._scanner_config.use_1m_base_synthesis
        else:
            # Hardcoded default if no config available (should not happen in normal operation)
            use_1m_base = False

        if use_1m_base:
            # 1M BASE SYNTHESIS MODE
            # Benefits: 1 gap = 6.7% loss vs 33% with 5m base, fresher data
            if timeframe in ('15m', '1h', '4h', '5m'):
                source_tf = 1  # Fetch 1m data for all resampling
                # Calculate lookback multiplier based on target timeframe
                # 15m needs 15x more 1m bars, 1h needs 60x, 4h needs 240x, 5m needs 5x
                multipliers = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
                base_multiplier = multipliers.get(timeframe, 15)
                # Add 20% buffer for resampling edge cases
                adjusted_lookback = lookback_hours * 1.2
                since_utc = tz_manager.get_lookback_time_utc(adjusted_lookback)
                self.logger.debug(f"📊 Using 1m base synthesis for {timeframe} (source_tf=1)")
            else:
                source_tf = tf_minutes
        else:
            # 5M BASE SYNTHESIS MODE (default - current behavior)
            if timeframe == '15m':
                source_tf = 5  # Always fetch 5m data for 15m resampling
                # Increase lookback to ensure we have enough 5m data
                # 15m needs 3x more 5m bars, so increase lookback accordingly
                adjusted_lookback = lookback_hours * 1.2  # 20% buffer for resampling
                since_utc = tz_manager.get_lookback_time_utc(adjusted_lookback)
            elif timeframe == '1h':
                source_tf = 5  # Always fetch 5m data for 60m resampling
                # Increase lookback to ensure we have enough 5m data
                # 60m needs 12x more 5m bars, so increase lookback accordingly
                adjusted_lookback = lookback_hours * 1.2  # 20% buffer for resampling
                since_utc = tz_manager.get_lookback_time_utc(adjusted_lookback)
            elif timeframe == '4h':
                source_tf = 5  # Always fetch 5m data for 4h resampling
                # 4h needs 48x more 5m bars (48 x 5m = 4h), increase lookback accordingly
                adjusted_lookback = lookback_hours * 1.2  # 20% buffer for resampling
                since_utc = tz_manager.get_lookback_time_utc(adjusted_lookback)
            else:
                source_tf = tf_minutes
        
        # Use validated preferred prices for trading safety
        # Optimize data fetching: for large requests (backtesting), get most recent data
        # For normal requests (live trading), use simple chronological order
        use_recent_data_optimization = self.batch_size > 3000  # Likely backtesting
        
        if use_recent_data_optimization:
            # Backtesting mode: fetch most recent data within lookback period
            query = """
                WITH recent_data AS (
                    SELECT 
                        pfp.start_time, 
                        ic.open, 
                        ic.high, 
                        ic.low, 
                        pfp.preferred_price as close,
                        ic.ltv,
                        pfp.preferred_source as data_source,
                        pfp.quality_score,
                        -- Add safety validation flag
                        is_price_safe_for_trading(pfp.epic, pfp.timeframe, pfp.start_time, 10.0) as is_safe_for_trading
                    FROM preferred_forex_prices pfp
                    JOIN ig_candles ic ON (
                        pfp.epic = ic.epic 
                        AND pfp.timeframe = ic.timeframe 
                        AND pfp.start_time = ic.start_time
                        AND pfp.preferred_source = ic.data_source
                    )
                    WHERE pfp.epic = :epic
                    AND pfp.timeframe = :timeframe
                    AND pfp.start_time >= :since
                    ORDER BY pfp.start_time DESC
                    LIMIT :limit
                )
                SELECT * FROM recent_data ORDER BY start_time ASC
            """
        else:
            # Live trading mode: simple chronological order (original behavior)
            query = """
                SELECT 
                    pfp.start_time, 
                    ic.open, 
                    ic.high, 
                    ic.low, 
                    pfp.preferred_price as close,
                    ic.ltv,
                    pfp.preferred_source as data_source,
                    pfp.quality_score,
                    -- Add safety validation flag
                    is_price_safe_for_trading(pfp.epic, pfp.timeframe, pfp.start_time, 10.0) as is_safe_for_trading
                FROM preferred_forex_prices pfp
                JOIN ig_candles ic ON (
                    pfp.epic = ic.epic 
                    AND pfp.timeframe = ic.timeframe 
                    AND pfp.start_time = ic.start_time
                    AND pfp.preferred_source = ic.data_source
                )
                WHERE pfp.epic = :epic
                AND pfp.timeframe = :timeframe
                AND pfp.start_time >= :since
                ORDER BY pfp.start_time ASC
                LIMIT :limit
            """
        
        try:
            self.logger.debug(f"🔍 Fetching {epic} data since {tz_manager.format_time_for_display(since_utc)}")
            
            df = self.db_manager.execute_query(query, {
                "epic": epic,
                "timeframe": source_tf,
                "since": since_utc,
                "limit": self.batch_size
            })
            
            if df.empty:
                return None

            # Data quality filtering for trading safety (database-only - no fallback)
            if self._scanner_config.enable_data_quality_filtering:
                if 'is_safe_for_trading' in df.columns:
                    unsafe_count = len(df[df['is_safe_for_trading'] == False])
                    if unsafe_count > 0:
                        self.logger.warning(f"⚠️ {unsafe_count} unsafe price points detected for {epic} - filtering out")

                        if self._scanner_config.block_trading_on_data_issues:
                            df = df[df['is_safe_for_trading'] == True]

                            if df.empty:
                                self.logger.error(f"❌ All data filtered out as unsafe for {epic} - cannot proceed with trading")
                                return None

                    # Log data quality summary
                    if 'quality_score' in df.columns:
                        avg_quality = df['quality_score'].mean()
                        min_quality = df['quality_score'].min()
                        min_quality_threshold = self._scanner_config.min_quality_score_for_trading

                        self.logger.debug(f"📊 Data quality for {epic}: avg={avg_quality:.3f}, min={min_quality:.3f}")

                        if min_quality < min_quality_threshold:
                            self.logger.warning(f"⚠️ Low quality data detected for {epic} (min: {min_quality:.3f} < threshold: {min_quality_threshold})")

                            if self._scanner_config.block_trading_on_data_issues:
                                # Filter out low quality data
                                df = df[df['quality_score'] >= min_quality_threshold]
                                if df.empty:
                                    self.logger.error(f"❌ All data below quality threshold for {epic}")
                                    return None
            else:
                self.logger.debug(f"📊 Data quality filtering disabled for {epic}")

            # Convert timestamp efficiently
            df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
            
            # Add timezone columns using the correct method
            df = tz_manager.add_timezone_columns_to_df(df)
            
            # Resampling implementation - supports both 5m and 1m base modes
            if source_tf == 1:
                # 1M BASE SYNTHESIS MODE
                if timeframe == '5m':
                    self.logger.debug(f"🔄 Resampling 1m data to 5m for {epic}")
                    df = self._resample_to_5m_from_1m(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 1m→5m resampling failed for {epic}")
                        return None
                elif timeframe == '15m':
                    self.logger.debug(f"🔄 Resampling 1m data to 15m for {epic}")
                    df = self._resample_to_15m_from_1m(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 1m→15m resampling failed for {epic}")
                        return None
                elif timeframe == '1h':
                    self.logger.debug(f"🔄 Resampling 1m data to 60m for {epic}")
                    df = self._resample_to_60m_from_1m(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 1m→60m resampling failed for {epic}")
                        return None
                elif timeframe == '4h':
                    self.logger.debug(f"🔄 Resampling 1m data to 4h for {epic}")
                    df = self._resample_to_4h_from_1m(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 1m→4h resampling failed for {epic}")
                        return None
            elif source_tf == 5:
                # 5M BASE SYNTHESIS MODE (default)
                if timeframe == '15m':
                    self.logger.debug(f"🔄 Resampling 5m data to 15m for {epic}")
                    df = self._resample_to_15m_optimized(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 15m resampling failed for {epic}")
                        return None
                elif timeframe == '1h':
                    self.logger.debug(f"🔄 Resampling 5m data to 60m for {epic}")
                    df = self._resample_to_60m_optimized(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 60m resampling failed for {epic}")
                        return None
                elif timeframe == '4h':
                    self.logger.debug(f"🔄 Resampling 5m data to 4h for {epic}")
                    df = self._resample_to_4h_optimized(df)
                    if df is None or len(df) == 0:
                        self.logger.error(f"❌ 4h resampling failed for {epic}")
                        return None

            return df.reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"❌ Database query failed: {e}")
            return None
    
    def get_kama_data_for_debug(self, epic: str, pair: str, timeframe: str = '5m') -> Optional[Dict]:
        """Get KAMA-specific data for debugging purposes"""
        try:
            df = self.get_enhanced_data(epic, pair, timeframe, lookback_hours=168)
            
            if df is None or len(df) == 0:
                return None
            
            # Get KAMA configuration from database (no config.py fallback)
            kama_config = {
                'period': self._scanner_config.kama_period,
                'fast': self._scanner_config.kama_fast,
                'slow': self._scanner_config.kama_slow
            }
            kama_period = kama_config['period']
            
            # Extract KAMA data
            latest = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else latest
            
            kama_data = {
                'epic': epic,
                'timeframe': timeframe,
                'kama_config': kama_config,
                'latest_timestamp': latest['start_time'],
                'current_price': latest['close'],
                'data_points': len(df)
            }
            
            # Add KAMA indicators if they exist
            kama_fields = [
                f'kama_{kama_period}', f'kama_{kama_period}_er', 
                f'kama_{kama_period}_trend', f'kama_{kama_period}_signal'
            ]
            
            for field in kama_fields:
                if field in latest:
                    kama_data[f'{field}_current'] = latest[field]
                    kama_data[f'{field}_previous'] = previous[field] if field in previous else None
                else:
                    kama_data[f'{field}_current'] = None
                    kama_data[f'{field}_previous'] = None
            
            # Calculate some basic statistics
            if f'kama_{kama_period}_er' in df.columns:
                er_values = df[f'kama_{kama_period}_er'].dropna()
                if len(er_values) > 0:
                    kama_data['er_stats'] = {
                        'mean': er_values.mean(),
                        'std': er_values.std(),
                        'min': er_values.min(),
                        'max': er_values.max(),
                        'current': latest.get(f'kama_{kama_period}_er', 0)
                    }
            
            return kama_data
            
        except Exception as e:
            self.logger.error(f"❌ Error getting KAMA debug data: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
        self._indicator_cache.clear()
        self.logger.info("🧹 Data cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'data_cache_size': len(self._data_cache),
            'indicator_cache_size': len(self._indicator_cache),
            'cache_enabled': self.cache_enabled,
            'cache_timeout_seconds': self._cache_timeout
        }

    def _add_semantic_ema_columns(self, df: pd.DataFrame, ema_periods: List[int]) -> pd.DataFrame:
        """
        ENHANCED: Add semantic EMA column names that match the periods actually calculated
        
        This ensures that ema_short, ema_long, ema_trend map to the correct 
        dynamically selected EMA periods instead of hardcoded defaults.
        """
        try:
            df_semantic = df.copy()
            
            # Map semantic names to actual calculated EMAs
            short_period, long_period, trend_period = ema_periods
            
            # Add semantic mappings for the actual periods calculated
            if f'ema_{short_period}' in df.columns:
                df_semantic['ema_short'] = df[f'ema_{short_period}']
                self.logger.debug(f"✅ Mapped ema_short to ema_{short_period}")
            else:
                # Create the EMA if it's missing
                df_semantic[f'ema_{short_period}'] = df_semantic['close'].ewm(span=short_period).mean()
                df_semantic['ema_short'] = df_semantic[f'ema_{short_period}']
                self.logger.warning(f"⚠️ Created missing ema_{short_period} and mapped to ema_short")
            
            if f'ema_{long_period}' in df.columns:
                df_semantic['ema_long'] = df[f'ema_{long_period}']
                self.logger.debug(f"✅ Mapped ema_long to ema_{long_period}")
            else:
                # Create the EMA if it's missing
                df_semantic[f'ema_{long_period}'] = df_semantic['close'].ewm(span=long_period).mean()
                df_semantic['ema_long'] = df_semantic[f'ema_{long_period}']
                self.logger.warning(f"⚠️ Created missing ema_{long_period} and mapped to ema_long")
                
            if f'ema_{trend_period}' in df.columns:
                df_semantic['ema_trend'] = df[f'ema_{trend_period}']
                self.logger.debug(f"✅ Mapped ema_trend to ema_{trend_period}")
            else:
                # Create the EMA if it's missing
                df_semantic[f'ema_{trend_period}'] = df_semantic['close'].ewm(span=trend_period).mean()
                df_semantic['ema_trend'] = df_semantic[f'ema_{trend_period}']
                self.logger.warning(f"⚠️ Created missing ema_{trend_period} and mapped to ema_trend")
            
            # ENHANCED: Ensure backward compatibility with legacy column names
            # If the actual periods don't match legacy names, create aliases
            
            # Handle ema_9 compatibility
            if 'ema_9' not in df_semantic.columns:
                if short_period == 9 and f'ema_{short_period}' in df.columns:
                    df_semantic['ema_9'] = df[f'ema_{short_period}']
                elif f'ema_9' in df.columns:
                    pass  # Already exists
                else:
                    # Create ema_9 if needed for legacy compatibility
                    df_semantic['ema_9'] = df_semantic['close'].ewm(span=9).mean()
                    self.logger.debug(f"🔧 Created ema_9 for backward compatibility")
            
            # Handle ema_21 compatibility
            if 'ema_21' not in df_semantic.columns:
                if long_period == 21 and f'ema_{long_period}' in df.columns:
                    df_semantic['ema_21'] = df[f'ema_{long_period}']
                elif f'ema_21' in df.columns:
                    pass  # Already exists
                else:
                    # Create ema_21 if needed for legacy compatibility
                    df_semantic['ema_21'] = df_semantic['close'].ewm(span=21).mean()
                    self.logger.debug(f"🔧 Created ema_21 for backward compatibility")
            
            # Handle ema_50, ema_100, ema_200 compatibility (always ensure these exist)
            if 'ema_50' not in df_semantic.columns:
                if f'ema_50' in df.columns:
                    pass  # Already exists
                else:
                    df_semantic['ema_50'] = df_semantic['close'].ewm(span=50).mean()
                    self.logger.debug(f"🔧 Created ema_50 for trend filtering")

            if 'ema_100' not in df_semantic.columns:
                if f'ema_100' in df.columns:
                    pass  # Already exists
                else:
                    df_semantic['ema_100'] = df_semantic['close'].ewm(span=100).mean()
                    self.logger.debug(f"🔧 Created ema_100 for trend filtering")

            if 'ema_200' not in df_semantic.columns:
                if trend_period == 200 and f'ema_{trend_period}' in df.columns:
                    df_semantic['ema_200'] = df[f'ema_{trend_period}']
                elif f'ema_200' in df.columns:
                    pass  # Already exists
                else:
                    df_semantic['ema_200'] = df_semantic['close'].ewm(span=200).mean()
                    self.logger.debug(f"🔧 Created ema_200 for trend filtering")

            return df_semantic
            
        except Exception as e:
            self.logger.error(f"❌ Error adding semantic EMA columns: {e}")
            return df

    def _validate_ema_calculation(self, df: pd.DataFrame, epic: str, ema_periods: List[int], config_source: str):
        """
        ENHANCED: Validate that the correct EMAs were calculated and log verification
        """
        try:
            short_period, long_period, trend_period = ema_periods
            
            # Check if the expected EMA columns exist
            expected_columns = [f'ema_{short_period}', f'ema_{long_period}', f'ema_{trend_period}']
            semantic_columns = ['ema_short', 'ema_long', 'ema_trend']
            
            # Validate specific period columns
            missing_periods = []
            for period in ema_periods:
                if f'ema_{period}' not in df.columns:
                    missing_periods.append(period)
            
            # Validate semantic columns
            missing_semantic = []
            for col in semantic_columns:
                if col not in df.columns:
                    missing_semantic.append(col)
            
            # Log validation results
            if not missing_periods and not missing_semantic:
                self.logger.debug(f"✅ EMA validation passed for {epic} (semantic mapping: ✅)")
                
                # Log actual EMA values for verification
                if len(df) > 0:
                    latest = df.iloc[-1]
                    self.logger.debug(f"   Latest EMA values:")
                    self.logger.debug(f"     ema_short ({short_period}): {latest.get('ema_short', 'N/A'):.5f}")
                    self.logger.debug(f"     ema_long ({long_period}): {latest.get('ema_long', 'N/A'):.5f}")
                    self.logger.debug(f"     ema_trend ({trend_period}): {latest.get('ema_trend', 'N/A'):.5f}")
            else:
                self.logger.warning(f"⚠️ EMA validation issues for {epic}")
                if missing_periods:
                    self.logger.warning(f"   Missing period columns: {missing_periods}")
                if missing_semantic:
                    self.logger.warning(f"   Missing semantic columns: {missing_semantic}")
            
        except Exception as e:
            self.logger.error(f"❌ Error validating EMA calculation for {epic}: {e}")

# Maintain backward compatibility
OptimizedDataFetcher = DataFetcher  # Alias for backward compatibility