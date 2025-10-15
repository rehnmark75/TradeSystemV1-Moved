# core/strategies/helpers/ema_indicator_calculator.py
"""
Multi-Indicator Calculator Module (EMA/Supertrend)
Handles both EMA and Supertrend calculation depending on strategy mode.

When USE_SUPERTREND_MODE = True: Uses Supertrend indicators
When USE_SUPERTREND_MODE = False: Uses legacy EMA indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

try:
    from configdata.strategies import config_ema_strategy
except ImportError:
    from forex_scanner.configdata.strategies import config_ema_strategy

from .supertrend_calculator import SupertrendCalculator


class EMAIndicatorCalculator:
    """Calculates indicators (EMA or Supertrend) and detects signals"""

    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8, use_supertrend: Optional[bool] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps  # Epsilon for stability

        # Determine which mode to use
        if use_supertrend is None:
            self.use_supertrend = getattr(config_ema_strategy, 'USE_SUPERTREND_MODE', False)
        else:
            self.use_supertrend = use_supertrend

        # Initialize Supertrend calculator if needed
        if self.use_supertrend:
            self.supertrend_calc = SupertrendCalculator(logger=self.logger)
            self.logger.info("📊 Indicator Calculator initialized in SUPERTREND mode")
        else:
            self.supertrend_calc = None
            self.logger.info("📊 Indicator Calculator initialized in EMA mode (legacy)")

    def ensure_supertrends(self, df: pd.DataFrame, supertrend_config: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate multi-Supertrend indicators for trend confluence

        Args:
            df: DataFrame with OHLC data
            supertrend_config: Dictionary with Supertrend parameters

        Returns:
            DataFrame with Supertrend columns added
        """
        try:
            # Extract parameters
            fast_period = int(supertrend_config.get('fast_period', 7))
            fast_multiplier = float(supertrend_config.get('fast_multiplier', 1.5))
            medium_period = int(supertrend_config.get('medium_period', 14))
            medium_multiplier = float(supertrend_config.get('medium_multiplier', 2.0))
            slow_period = int(supertrend_config.get('slow_period', 21))
            slow_multiplier = float(supertrend_config.get('slow_multiplier', 3.0))
            atr_period = int(supertrend_config.get('atr_period', 14))

            self.logger.debug(
                f"Calculating Supertrends: Fast({fast_period},{fast_multiplier}), "
                f"Medium({medium_period},{medium_multiplier}), Slow({slow_period},{slow_multiplier})"
            )

            # Calculate multi-Supertrend
            supertrends = self.supertrend_calc.calculate_multi_supertrend(
                df,
                fast_period=fast_period,
                fast_multiplier=fast_multiplier,
                medium_period=medium_period,
                medium_multiplier=medium_multiplier,
                slow_period=slow_period,
                slow_multiplier=slow_multiplier,
                atr_period=atr_period
            )

            # Add to DataFrame with clear column names
            df['st_fast'] = supertrends['fast']['supertrend']
            df['st_fast_trend'] = supertrends['fast']['trend']
            df['st_medium'] = supertrends['medium']['supertrend']
            df['st_medium_trend'] = supertrends['medium']['trend']
            df['st_slow'] = supertrends['slow']['supertrend']
            df['st_slow_trend'] = supertrends['slow']['trend']
            df['atr'] = supertrends['fast']['atr']  # Use ATR from any Supertrend

            # Log for verification
            if len(df) > 0:
                fast_trend = int(df['st_fast_trend'].iloc[-1])
                medium_trend = int(df['st_medium_trend'].iloc[-1])
                slow_trend = int(df['st_slow_trend'].iloc[-1])
                self.logger.debug(
                    f"Supertrends calculated: Fast={fast_trend}, Medium={medium_trend}, Slow={slow_trend}"
                )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating Supertrends: {e}", exc_info=True)
            return df

    def ensure_emas(self, df: pd.DataFrame, ema_config: Dict[str, int]) -> pd.DataFrame:
        """
        Calculate EMAs using specified periods, regardless of what's in the data
        This ensures consistency with the original detect_ema_alerts function
        
        Args:
            df: DataFrame with price data
            ema_config: Dictionary with 'short', 'long', 'trend' periods
            
        Returns:
            DataFrame with EMA columns added
        """
        try:
            # ALWAYS calculate our own EMAs to ensure correct periods
            ema_short = ema_config.get('short', 12)
            ema_long = ema_config.get('long', 50)
            ema_trend = ema_config.get('trend', 200)
            
            self.logger.debug(f"Calculating EMAs with periods {ema_short}/{ema_long}/{ema_trend}")
            
            # Calculate EMAs with specific periods
            df[f'ema_{ema_short}'] = df['close'].ewm(span=ema_short, adjust=False).mean()
            df[f'ema_{ema_long}'] = df['close'].ewm(span=ema_long, adjust=False).mean()
            df[f'ema_{ema_trend}'] = df['close'].ewm(span=ema_trend, adjust=False).mean()
            
            # Add generic column names for the detection logic
            df['ema_short'] = df[f'ema_{ema_short}']
            df['ema_long'] = df[f'ema_{ema_long}']
            df['ema_trend'] = df[f'ema_{ema_trend}']
            
            # Log for verification
            if len(df) > 0:
                last_short = df['ema_short'].iloc[-1]
                last_long = df['ema_long'].iloc[-1]
                self.logger.debug(f"EMAs calculated: short={last_short:.5f}, long={last_long:.5f}")

            # Add overextension oscillators if enabled
            df = self.ensure_overextension_indicators(df)

            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating EMAs: {e}")
            return df
    
    def detect_ema_alerts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CORE DETECTION LOGIC: Based on the provided detect_ema_alerts function
        
        This implements the exact logic from the user-provided function:
        - Price crossover detection with previous values
        - Bull/Bear condition validation with EMA alignment  
        - Alert generation based on crossover + condition
        
        Args:
            df: DataFrame with EMA columns
            
        Returns:
            DataFrame with alert columns added
        """
        try:
            # Sort by time and reset index
            df = df.sort_values('start_time').reset_index(drop=True)
            
            # Get EMA values (use the generic names we set in ensure_emas)
            ema_short = df['ema_short']
            ema_long = df['ema_long'] 
            ema_trend = df['ema_trend']
            
            # Get previous values for crossover detection
            df['prev_close'] = df['close'].shift(1)
            df['prev_ema_short'] = ema_short.shift(1)
            
            # BULL DETECTION - CORRECTED ALIGNMENT RULES
            # Bull cross: price was below EMA_short, now above
            df['bull_cross'] = (
                (df['prev_close'] < df['prev_ema_short'] - self.eps) & 
                (df['close'] > ema_short + self.eps)
            )
            
            # Bull condition: CORRECT ALIGNMENT - Short > Long > Trend (all EMAs in uptrend alignment)
            df['bull_condition'] = (
                (ema_short > ema_long + self.eps) &     # Short EMA above Long EMA
                (ema_long > ema_trend + self.eps) &     # Long EMA above Trend EMA  
                (ema_short > ema_trend + self.eps)      # Short EMA above Trend EMA (redundant but explicit)
            )
            
            # Bull alert when both cross and condition are true
            df['bull_alert'] = df['bull_cross'] & df['bull_condition']

            # DEBUG: Check for any crossovers or conditions
            bull_crosses = df['bull_cross'].sum()
            bull_conditions = df['bull_condition'].sum()
            bull_alerts = df['bull_alert'].sum()
            self.logger.info(f"🔍 EMA Detection Debug: Bull crosses: {bull_crosses}, Bull conditions: {bull_conditions}, Bull alerts: {bull_alerts}")

            # BEAR DETECTION - CORRECTED ALIGNMENT RULES
            # Bear cross: price was above EMA_short, now below
            df['bear_cross'] = (
                (df['prev_close'] > df['prev_ema_short'] + self.eps) & 
                (df['close'] < ema_short - self.eps)
            )
            
            # Bear condition: CORRECT ALIGNMENT - Short < Long < Trend (all EMAs in downtrend alignment)
            df['bear_condition'] = (
                (ema_short < ema_long - self.eps) &     # Short EMA below Long EMA
                (ema_long < ema_trend - self.eps) &     # Long EMA below Trend EMA
                (ema_short < ema_trend - self.eps)      # Short EMA below Trend EMA (redundant but explicit)
            )
            
            # Bear alert when both cross and condition are true
            df['bear_alert'] = df['bear_cross'] & df['bear_condition']

            # DEBUG: Check for any bear signals too
            bear_crosses = df['bear_cross'].sum()
            bear_conditions = df['bear_condition'].sum()
            bear_alerts = df['bear_alert'].sum()
            self.logger.info(f"🔍 EMA Detection Debug: Bear crosses: {bear_crosses}, Bear conditions: {bear_conditions}, Bear alerts: {bear_alerts}")

            # CRITICAL DEBUG: Show which specific rows have alerts
            if bull_alerts > 0:
                bull_alert_rows = df[df['bull_alert'] == True].index.tolist()
                bull_timestamps = df[df['bull_alert'] == True]['start_time'].tolist()
                self.logger.debug(f"🎯 BULL ALERTS at rows: {bull_alert_rows}, timestamps: {bull_timestamps}")

            if bear_alerts > 0:
                bear_alert_rows = df[df['bear_alert'] == True].index.tolist()
                bear_timestamps = df[df['bear_alert'] == True]['start_time'].tolist()
                self.logger.debug(f"🎯 BEAR ALERTS at rows: {bear_alert_rows}, timestamps: {bear_timestamps}")

            return df

        except Exception as e:
            self.logger.error(f"Error in EMA alert detection: {e}")
            return df

    def detect_ema_bounce_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EMA BOUNCE DETECTION: TradingView professional methodology

        Based on "EMA Bounce Scalping" strategy (180+ likes)
        Entry on pullback to EMA with rejection candle confirmation

        Logic:
        1. Price near EMA21 (within 0.1% = ~10 pips)
        2. Rejection candle (wick shows bounce)
        3. EMA alignment intact (trend not broken)

        Returns:
            DataFrame with bounce signal columns added
        """
        try:
            from configdata.strategies.config_ema_strategy import (
                EMA_BOUNCE_DISTANCE_PCT,
                EMA_BOUNCE_WICK_RATIO,
                EMA_BOUNCE_MIN_CANDLE_BODY
            )

            # Calculate distance from EMA21
            df['ema21_distance_pct'] = abs((df['close'] - df['ema_21']) / (df['ema_21'] + self.eps) * 100)

            # Price near EMA condition
            df['price_near_ema21'] = df['ema21_distance_pct'] < EMA_BOUNCE_DISTANCE_PCT

            # Candle body and wick calculations
            df['candle_body'] = abs(df['close'] - df['open'])
            df['candle_range'] = df['high'] - df['low']
            df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']

            # Candle body ratio (avoid doji/small candles)
            df['body_ratio'] = df['candle_body'] / (df['candle_range'] + self.eps)

            # BULLISH BOUNCE DETECTION
            df['bullish_bounce'] = (
                (df['close'] < df['ema_21']) &                    # Below EMA (pullback)
                (df['close'] > df['low']) &                       # Closed off lows (rejection)
                (df['close'] > df['open']) &                      # Green candle
                (df['lower_wick'] > df['upper_wick'] * EMA_BOUNCE_WICK_RATIO) &  # Strong lower wick
                (df['body_ratio'] > EMA_BOUNCE_MIN_CANDLE_BODY)   # Decent body size
            )

            # BEARISH BOUNCE DETECTION
            df['bearish_bounce'] = (
                (df['close'] > df['ema_21']) &                    # Above EMA (pullback)
                (df['close'] < df['high']) &                      # Closed off highs (rejection)
                (df['close'] < df['open']) &                      # Red candle
                (df['upper_wick'] > df['lower_wick'] * EMA_BOUNCE_WICK_RATIO) &  # Strong upper wick
                (df['body_ratio'] > EMA_BOUNCE_MIN_CANDLE_BODY)   # Decent body size
            )

            # Get EMA values
            ema_short = df['ema_short']
            ema_long = df['ema_long']
            ema_trend = df['ema_trend']

            # Generate bounce signals (with EMA alignment check)
            df['bull_bounce_signal'] = (
                df['price_near_ema21'] &
                df['bullish_bounce'] &
                (ema_short > ema_long + self.eps) &       # Trend still intact
                (ema_long > ema_trend + self.eps)
            )

            df['bear_bounce_signal'] = (
                df['price_near_ema21'] &
                df['bearish_bounce'] &
                (ema_short < ema_long - self.eps) &       # Trend still intact
                (ema_long < ema_trend - self.eps)
            )

            # Debug logging with timestamps
            bounce_bulls = df['bull_bounce_signal'].sum()
            bounce_bears = df['bear_bounce_signal'].sum()
            self.logger.info(f"🎯 EMA Bounce Detection: Bulls: {bounce_bulls}, Bears: {bounce_bears}")

            # Log WHERE the bounces occurred
            if bounce_bulls > 0 or bounce_bears > 0:
                bull_timestamps = df[df['bull_bounce_signal']]['start_time'].tolist() if 'start_time' in df.columns else []
                bear_timestamps = df[df['bear_bounce_signal']]['start_time'].tolist() if 'start_time' in df.columns else []
                if bull_timestamps:
                    self.logger.info(f"   📍 BULL bounces at: {bull_timestamps[:3]}")  # Show first 3
                if bear_timestamps:
                    self.logger.info(f"   📍 BEAR bounces at: {bear_timestamps[:3]}")  # Show first 3

            return df

        except Exception as e:
            self.logger.error(f"Error in bounce detection: {e}")
            return df

    def detect_ema_signals(self, df: pd.DataFrame, trigger_mode: str = 'bounce') -> pd.DataFrame:
        """
        UNIFIED SIGNAL DETECTION: Route to appropriate detection method

        Args:
            df: DataFrame with EMA columns
            trigger_mode: 'bounce', 'crossover', or 'hybrid'

        Returns:
            DataFrame with signal columns (always includes bull_alert and bear_alert)
        """
        if trigger_mode == 'bounce':
            df = self.detect_ema_bounce_signals(df)
            # Map bounce signals to standard alert columns
            df['bull_alert'] = df.get('bull_bounce_signal', False)
            df['bear_alert'] = df.get('bear_bounce_signal', False)
            return df
        elif trigger_mode == 'crossover':
            return self.detect_ema_alerts(df)  # Existing crossover logic
        elif trigger_mode == 'hybrid':
            # Both methods, merge results
            df = self.detect_ema_alerts(df)
            df = self.detect_ema_bounce_signals(df)
            # Combine: bounce OR crossover
            df['bull_alert'] = df.get('bull_alert', False) | df.get('bull_bounce_signal', False)
            df['bear_alert'] = df.get('bear_alert', False) | df.get('bear_bounce_signal', False)
            return df
        else:
            self.logger.warning(f"Unknown trigger mode: {trigger_mode}, defaulting to bounce")
            df = self.detect_ema_bounce_signals(df)
            df['bull_alert'] = df.get('bull_bounce_signal', False)
            df['bear_alert'] = df.get('bear_bounce_signal', False)
            return df

    def get_required_indicators(self, ema_config: Dict[str, int]) -> List[str]:
        """
        Get list of required indicators for EMA strategy (UPDATED: No Two-Pole)

        Args:
            ema_config: Dictionary with EMA periods

        Returns:
            List of required indicator column names
        """
        base_indicators = [
            f'ema_{ema_config.get("short", 12)}',
            f'ema_{ema_config.get("long", 50)}',
            f'ema_{ema_config.get("trend", 200)}',
            'close',
            'open',
            'high',
            'low',
            'start_time',
            'atr'  # Always needed for stops
        ]

        # Add RSI (replaces Two-Pole)
        try:
            from configdata import config
        except ImportError:
            from forex_scanner.configdata import config

        from configdata.strategies.config_ema_strategy import (
            EMA_USE_RSI_FILTER,
            STOCHASTIC_OVEREXTENSION_ENABLED,
            WILLIAMS_R_OVEREXTENSION_ENABLED,
            RSI_EXTREME_OVEREXTENSION_ENABLED
        )

        if EMA_USE_RSI_FILTER or RSI_EXTREME_OVEREXTENSION_ENABLED:
            base_indicators.append('rsi')

        # Add MACD if enabled
        if getattr(config, 'MACD_MOMENTUM_FILTER_ENABLED', True):
            base_indicators.extend(['macd_line', 'macd_signal', 'macd_histogram'])

        # Add ADX (always needed for trend strength)
        base_indicators.append('adx')

        # Add overextension indicators if enabled
        if STOCHASTIC_OVEREXTENSION_ENABLED:
            base_indicators.extend(['stoch_k', 'stoch_d'])

        if WILLIAMS_R_OVEREXTENSION_ENABLED:
            base_indicators.append('williams_r')

        return base_indicators

    def ensure_overextension_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overextension oscillator indicators if enabled and not present

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with oscillator indicators added
        """
        try:
            from configdata.strategies.config_ema_strategy import (
                STOCHASTIC_OVEREXTENSION_ENABLED,
                WILLIAMS_R_OVEREXTENSION_ENABLED,
                RSI_EXTREME_OVEREXTENSION_ENABLED
            )

            # Calculate Stochastic if enabled
            if STOCHASTIC_OVEREXTENSION_ENABLED:
                if 'stoch_k' not in df.columns or 'stoch_d' not in df.columns:
                    self.logger.debug("Calculating Stochastic indicators")
                    # Calculate Stochastic %K and %D
                    period = 14  # Standard Stochastic period
                    k_period = 3  # %K smoothing
                    d_period = 3  # %D smoothing

                    # Calculate %K
                    low_min = df['low'].rolling(window=period).min()
                    high_max = df['high'].rolling(window=period).max()
                    k_percent = 100 * (df['close'] - low_min) / (high_max - low_min + self.eps)
                    df['stoch_k'] = k_percent.rolling(window=k_period).mean()

                    # Calculate %D
                    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

            # Calculate Williams %R if enabled
            if WILLIAMS_R_OVEREXTENSION_ENABLED:
                if 'williams_r' not in df.columns:
                    self.logger.debug("Calculating Williams %R indicator")
                    period = 14  # Standard Williams %R period
                    high_max = df['high'].rolling(window=period).max()
                    low_min = df['low'].rolling(window=period).min()
                    df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min + self.eps)

            # Calculate RSI if enabled
            if RSI_EXTREME_OVEREXTENSION_ENABLED:
                if 'rsi' not in df.columns:
                    self.logger.debug("Calculating RSI indicator")
                    period = 14  # Standard RSI period
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / (loss + self.eps)
                    df['rsi'] = 100 - (100 / (1 + rs))

            return df

        except Exception as e:
            self.logger.error(f"Error calculating overextension indicators: {e}")
            return df

    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int) -> bool:
        """
        Validate that DataFrame meets minimum requirements
        
        Args:
            df: DataFrame to validate
            min_bars: Minimum number of bars required
            
        Returns:
            True if requirements met, False otherwise
        """
        if df is None or df.empty:
            self.logger.debug("DataFrame is None or empty")
            return False
        
        if len(df) < min_bars:
            self.logger.info(f"Insufficient data: {len(df)} < {min_bars}")
            return False
        
        # Check for required columns
        required_cols = ['close', 'open', 'high', 'low']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
        
        return True