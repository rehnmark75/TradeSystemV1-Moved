# core/strategies/helpers/ema_indicator_calculator.py
"""
EMA Indicator Calculator Module
Handles EMA calculation and signal detection logic
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List


class EMAIndicatorCalculator:
    """Calculates EMA indicators and detects crossover signals"""
    
    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps  # Epsilon for stability
    
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

            return df
            
        except Exception as e:
            self.logger.error(f"Error in EMA alert detection: {e}")
            return df
    
    def get_required_indicators(self, ema_config: Dict[str, int]) -> List[str]:
        """
        Get list of required indicators for EMA strategy
        
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
            'start_time'
        ]
        
        # Add Two-Pole Oscillator indicators if enabled
        try:
            from configdata import config
        except ImportError:
            from forex_scanner.configdata import config
        if getattr(config, 'TWO_POLE_OSCILLATOR_ENABLED', False):
            base_indicators.extend([
                'two_pole_osc',
                'two_pole_osc_delayed',
                'two_pole_is_green',
                'two_pole_is_purple',
                'two_pole_buy_signal', 
                'two_pole_sell_signal',
                'two_pole_strength',
                'two_pole_zone'
            ])
        

        # Add overextension indicators if enabled
        from configdata.strategies.config_ema_strategy import (
            STOCHASTIC_OVEREXTENSION_ENABLED,
            WILLIAMS_R_OVEREXTENSION_ENABLED,
            RSI_EXTREME_OVEREXTENSION_ENABLED
        )

        if STOCHASTIC_OVEREXTENSION_ENABLED:
            base_indicators.extend(['stoch_k', 'stoch_d'])

        if WILLIAMS_R_OVEREXTENSION_ENABLED:
            base_indicators.extend(['williams_r'])

        if RSI_EXTREME_OVEREXTENSION_ENABLED:
            base_indicators.extend(['rsi'])

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