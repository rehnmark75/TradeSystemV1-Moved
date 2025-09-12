# analysis/technical.py
"""
Technical Analysis Module
Handles EMA calculations, support/resistance detection, and technical indicators
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import List, Dict, Tuple
import logging
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class TechnicalAnalyzer:
    """Technical analysis calculations and indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands to DataFrame
        
        Args:
            df: DataFrame with OHLC data
            period: Period for moving average (typically 20)
            std_dev: Standard deviation multiplier (typically 2.0)
            
        Returns:
            DataFrame with BB columns added
        """
        df = df.copy()
        
        # Calculate Simple Moving Average (BB Middle Line)
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        
        # Calculate Standard Deviation
        df['bb_std'] = df['close'].rolling(window=period).std()
        
        # Calculate Upper and Lower Bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate BB Width (useful for volatility analysis)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Calculate BB Position (where current price sits within bands)
        # 0.5 = middle, 1.0 = upper band, 0.0 = lower band
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df

    def add_ema_indicators(self, df: pd.DataFrame, ema_periods: List[int] = None) -> pd.DataFrame:
        """
        Add EMA indicators to DataFrame with configurable periods
        
        Args:
            df: DataFrame with OHLC data
            ema_periods: List of EMA periods to calculate (if None, uses config)
            
        Returns:
            DataFrame with EMA indicators added
        """
        df_emas = df.copy()
        
        # Use provided periods or get from config
        if ema_periods is None:
            if hasattr(config, 'get_ema_periods'):
                ema_periods = config.get_ema_periods()
            else:
                # Fallback to legacy config
                ema_periods = getattr(config, 'EMA_PERIODS', [9, 21, 200])
        
        # Log what periods we're calculating
        self.logger.debug(f"Adding EMA indicators for periods: {ema_periods}")
        
        # Calculate EMAs for each period with adjust=False to match live scanner
        for period in ema_periods:
            col_name = f'ema_{period}'
            if col_name not in df_emas.columns:
                df_emas[col_name] = df_emas['close'].ewm(span=period, adjust=False).mean()
                self.logger.debug(f"Created {col_name} column")
            else:
                self.logger.debug(f"{col_name} already exists")
        
        # Add Bollinger Bands (using middle EMA or default 20)
        if not any(col.startswith('bb_') for col in df_emas.columns):
            bb_period = 20  # Standard BB period
            df_emas = self.add_bollinger_bands(df_emas, period=bb_period)
        
        # Log columns after adding
        ema_cols = [col for col in df_emas.columns if col.startswith('ema_')]
        self.logger.debug(f"EMA columns after adding: {ema_cols}")
        
        return df_emas
    
    def add_macd_indicators(self, df: pd.DataFrame, fast_period: int = 12, 
                           slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Add MACD indicators to DataFrame
        
        Args:
            df: DataFrame with OHLC data
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with MACD indicators added
        """
        df_macd = df.copy()
        
        # Calculate EMAs
        ema_fast = df_macd['close'].ewm(span=fast_period).mean()
        ema_slow = df_macd['close'].ewm(span=slow_period).mean()
        
        # Calculate MACD line
        df_macd['macd_line'] = ema_fast - ema_slow
        
        # Calculate signal line
        df_macd['macd_signal'] = df_macd['macd_line'].ewm(span=signal_period).mean()
        
        # Calculate histogram
        df_macd['macd_histogram'] = df_macd['macd_line'] - df_macd['macd_signal']
        
        # Add histogram color (green/red) for easier analysis
        df_macd['macd_color'] = df_macd['macd_histogram'].apply(lambda x: 'green' if x > 0 else 'red')
        df_macd['macd_color_prev'] = df_macd['macd_color'].shift(1)
        
        # Detect histogram color switches
        df_macd['macd_red_to_green'] = (
            (df_macd['macd_color_prev'] == 'red') & 
            (df_macd['macd_color'] == 'green')
        )
        df_macd['macd_green_to_red'] = (
            (df_macd['macd_color_prev'] == 'green') & 
            (df_macd['macd_color'] == 'red')
        )
        
        return df_macd
    
    def find_support_resistance_levels(
        self, 
        df: pd.DataFrame, 
        window: int = 20, 
        min_touches: int = 2
    ) -> Dict[str, List[float]]:
        """
        Find support and resistance levels using local extrema
        
        Args:
            df: DataFrame with OHLC data
            window: Window for finding local extrema
            min_touches: Minimum touches to confirm a level
            
        Returns:
            Dictionary with support and resistance levels
        """
        if len(df) < window * 2:
            return {'support_levels': [], 'resistance_levels': []}
        
        highs = df['high'].values
        lows = df['low'].values
        
        # Find local extrema
        resistance_indices = argrelextrema(highs, np.greater, order=window)[0]
        support_indices = argrelextrema(lows, np.less, order=window)[0]
        
        resistance_levels = highs[resistance_indices]
        support_levels = lows[support_indices]
        
        # Cluster nearby levels
        resistance_clusters = self._cluster_levels(resistance_levels)
        support_clusters = self._cluster_levels(support_levels)
        
        return {
            'resistance_levels': sorted(resistance_clusters, reverse=True),
            'support_levels': sorted(support_clusters, reverse=True)
        }
    
    def _cluster_levels(self, levels: np.ndarray, tolerance_pct: float = 0.001) -> List[float]:
        """
        Cluster nearby price levels
        
        Args:
            levels: Array of price levels
            tolerance_pct: Tolerance as percentage of price
            
        Returns:
            List of clustered levels
        """
        if len(levels) == 0:
            return []
        
        levels_sorted = np.sort(levels)
        clusters = []
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance_pct:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def add_support_resistance_to_df(
        self, 
        df: pd.DataFrame, 
        pair: str = 'EURUSD'
    ) -> pd.DataFrame:
        """
        Add support/resistance levels and distances to DataFrame
        
        Args:
            df: DataFrame with OHLC data
            pair: Currency pair for pip calculation
            
        Returns:
            Enhanced DataFrame with S/R data
        """
        if df is None or len(df) == 0:
            return df
        
        df_enhanced = df.copy()
        
        # Determine pip multiplier
        pip_multiplier = 100 if 'JPY' in pair.upper() else 10000
        
        # Find support/resistance levels
        sr_levels = self.find_support_resistance_levels(df_enhanced)
        
        # Calculate distances for each row
        nearest_resistance = []
        nearest_support = []
        distance_to_resistance_pips = []
        distance_to_support_pips = []
        risk_reward_ratio = []
        
        for _, row in df_enhanced.iterrows():
            current_price = row['close']
            
            # Find nearest resistance (above current price)
            resistance_above = [r for r in sr_levels['resistance_levels'] if r > current_price]
            nearest_res = min(resistance_above) if resistance_above else None
            
            # Find nearest support (below current price)
            support_below = [s for s in sr_levels['support_levels'] if s < current_price]
            nearest_sup = max(support_below) if support_below else None
            
            # Calculate distances in pips
            if nearest_res:
                dist_to_res = (nearest_res - current_price) * pip_multiplier
            else:
                dist_to_res = np.nan
                
            if nearest_sup:
                dist_to_sup = (current_price - nearest_sup) * pip_multiplier
            else:
                dist_to_sup = np.nan
            
            # Calculate risk/reward ratio
            if not np.isnan(dist_to_res) and not np.isnan(dist_to_sup) and dist_to_sup != 0:
                rr_ratio = dist_to_res / dist_to_sup
            else:
                rr_ratio = np.nan
            
            nearest_resistance.append(nearest_res)
            nearest_support.append(nearest_sup)
            distance_to_resistance_pips.append(dist_to_res)
            distance_to_support_pips.append(dist_to_sup)
            risk_reward_ratio.append(rr_ratio)
        
        # Add columns to dataframe
        df_enhanced['nearest_resistance'] = nearest_resistance
        df_enhanced['nearest_support'] = nearest_support
        df_enhanced['distance_to_resistance_pips'] = distance_to_resistance_pips
        df_enhanced['distance_to_support_pips'] = distance_to_support_pips
        df_enhanced['risk_reward_ratio'] = risk_reward_ratio
        
        return df_enhanced
    
    def calculate_moving_averages(
        self, 
        df: pd.DataFrame, 
        periods: List[int] = [20, 50, 100]
    ) -> pd.DataFrame:
        """
        Add simple moving averages to DataFrame
        
        Args:
            df: DataFrame with price data
            periods: List of MA periods
            
        Returns:
            DataFrame with MA columns
        """
        df_ma = df.copy()
        
        for period in periods:
            df_ma[f'sma_{period}'] = df_ma['close'].rolling(window=period).mean()
        
        return df_ma
    
    def calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with price data
            period: Period for moving average
            std_dev: Standard deviations for bands
            
        Returns:
            DataFrame with Bollinger Band columns
        """
        df_bb = df.copy()
        
        # Calculate middle band (SMA)
        df_bb['bb_middle'] = df_bb['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = df_bb['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df_bb['bb_upper'] = df_bb['bb_middle'] + (rolling_std * std_dev)
        df_bb['bb_lower'] = df_bb['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate position within bands
        df_bb['bb_position'] = (df_bb['close'] - df_bb['bb_lower']) / (df_bb['bb_upper'] - df_bb['bb_lower'])
        
        return df_bb
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with price data
            period: RSI period
            
        Returns:
            DataFrame with RSI column
        """
        df_rsi = df.copy()
        
        # Calculate price changes
        delta = df_rsi['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        df_rsi['rsi'] = 100 - (100 / (1 + rs))
        
        return df_rsi
    
    def calculate_macd(
        self, 
        df: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with MACD columns
        """
        df_macd = df.copy()
        
        # Calculate EMAs
        ema_fast = df_macd['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_macd['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df_macd['macd'] = ema_fast - ema_slow
        
        # Calculate signal line
        df_macd['macd_signal'] = df_macd['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df_macd['macd_histogram'] = df_macd['macd'] - df_macd['macd_signal']
        
        return df_macd
    
    def detect_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect basic price patterns (doji, hammer, etc.)
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern columns
        """
        df_patterns = df.copy()
        
        # Calculate candle properties
        df_patterns['body_size'] = abs(df_patterns['close'] - df_patterns['open'])
        df_patterns['upper_wick'] = df_patterns['high'] - df_patterns[['open', 'close']].max(axis=1)
        df_patterns['lower_wick'] = df_patterns[['open', 'close']].min(axis=1) - df_patterns['low']
        df_patterns['total_range'] = df_patterns['high'] - df_patterns['low']
        
        # Identify patterns
        # Doji: small body relative to range
        df_patterns['is_doji'] = (
            df_patterns['body_size'] < df_patterns['total_range'] * 0.1
        ) & (df_patterns['total_range'] > 0)
        
        # Hammer: small body, long lower wick, small upper wick
        df_patterns['is_hammer'] = (
            (df_patterns['lower_wick'] > df_patterns['body_size'] * 2) &
            (df_patterns['upper_wick'] < df_patterns['body_size'] * 0.5) &
            (df_patterns['body_size'] > 0)
        )
        
        # Shooting star: small body, long upper wick, small lower wick
        df_patterns['is_shooting_star'] = (
            (df_patterns['upper_wick'] > df_patterns['body_size'] * 2) &
            (df_patterns['lower_wick'] < df_patterns['body_size'] * 0.5) &
            (df_patterns['body_size'] > 0)
        )
        
        # Engulfing patterns
        df_patterns['is_bullish_engulfing'] = (
            (df_patterns['close'] > df_patterns['open']) &  # Current candle is green
            (df_patterns['close'].shift(1) < df_patterns['open'].shift(1)) &  # Previous candle is red
            (df_patterns['close'] > df_patterns['open'].shift(1)) &  # Current close > previous open
            (df_patterns['open'] < df_patterns['close'].shift(1))    # Current open < previous close
        )
        
        df_patterns['is_bearish_engulfing'] = (
            (df_patterns['close'] < df_patterns['open']) &  # Current candle is red
            (df_patterns['close'].shift(1) > df_patterns['open'].shift(1)) &  # Previous candle is green
            (df_patterns['close'] < df_patterns['open'].shift(1)) &  # Current close < previous open
            (df_patterns['open'] > df_patterns['close'].shift(1))    # Current open > previous close
        )
        
        return df_patterns
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily pivot points
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pivot point columns
        """
        df_pivot = df.copy()
        
        # Get previous day's high, low, close (simplified for intraday data)
        prev_high = df_pivot['high'].rolling(window=288).max().shift(1)  # ~24h of 5min bars
        prev_low = df_pivot['low'].rolling(window=288).min().shift(1)
        prev_close = df_pivot['close'].shift(288)
        
        # Calculate pivot point
        df_pivot['pivot'] = (prev_high + prev_low + prev_close) / 3
        
        # Calculate support and resistance levels
        df_pivot['r1'] = 2 * df_pivot['pivot'] - prev_low
        df_pivot['s1'] = 2 * df_pivot['pivot'] - prev_high
        df_pivot['r2'] = df_pivot['pivot'] + (prev_high - prev_low)
        df_pivot['s2'] = df_pivot['pivot'] - (prev_high - prev_low)
        df_pivot['r3'] = prev_high + 2 * (df_pivot['pivot'] - prev_low)
        df_pivot['s3'] = prev_low - 2 * (prev_high - df_pivot['pivot'])
        
        return df_pivot
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR)
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
            
        Returns:
            DataFrame with ATR column
        """
        df_atr = df.copy()
        
        # Calculate True Range components
        tr1 = df_atr['high'] - df_atr['low']
        tr2 = abs(df_atr['high'] - df_atr['close'].shift(1))
        tr3 = abs(df_atr['low'] - df_atr['close'].shift(1))
        
        # True Range is the maximum of the three
        df_atr['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the EMA of True Range
        df_atr['atr'] = df_atr['true_range'].ewm(span=period).mean()
        
        return df_atr