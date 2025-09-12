# analysis/behavior.py
"""
Market Behavior Analysis Module
Analyzes price action patterns and market behavior
"""

import pandas as pd
import numpy as np
import logging


class BehaviorAnalyzer:
    """Market behavior and price action analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_behavior_analysis(self, df: pd.DataFrame, pair: str = 'EURUSD') -> pd.DataFrame:
        """
        Add market behavior analysis to DataFrame
        
        Args:
            df: DataFrame with OHLC data
            pair: Currency pair for pip calculations
            
        Returns:
            DataFrame with behavior indicators
        """
        if df is None or len(df) == 0:
            return df
        
        df_behavior = df.copy()
        
        # Determine pip multiplier
        pip_multiplier = 100 if 'JPY' in pair.upper() else 10000
        
        # Price changes in pips
        df_behavior['price_change_1_bar_pips'] = (
            df_behavior['close'] - df_behavior['close'].shift(1)
        ) * pip_multiplier
        
        df_behavior['price_change_4_bars_pips'] = (
            df_behavior['close'] - df_behavior['close'].shift(4)
        ) * pip_multiplier
        
        df_behavior['price_change_12_bars_pips'] = (
            df_behavior['close'] - df_behavior['close'].shift(12)
        ) * pip_multiplier
        
        # Candle types
        df_behavior['is_green'] = df_behavior['close'] > df_behavior['open']
        df_behavior['is_red'] = df_behavior['close'] < df_behavior['open']
        df_behavior['is_doji'] = abs(df_behavior['close'] - df_behavior['open']) < (
            df_behavior['high'] - df_behavior['low']
        ) * 0.1
        
        # Consecutive candles
        df_behavior['consecutive_green_candles'] = self._calculate_consecutive_candles(
            df_behavior['is_green']
        )
        df_behavior['consecutive_red_candles'] = self._calculate_consecutive_candles(
            df_behavior['is_red']
        )
        
        # Rejection wicks
        df_behavior = self._add_rejection_analysis(df_behavior)
        
        # Consolidation and breakout analysis
        df_behavior = self._add_consolidation_analysis(df_behavior, pip_multiplier)
        
        return df_behavior
    
    def _calculate_consecutive_candles(self, series: pd.Series) -> list:
        """Calculate consecutive candle counts"""
        consecutive_counts = []
        current_count = 0
        current_type = None
        
        for value in series:
            if value == current_type:
                current_count += 1
            else:
                current_count = 1 if value else 0
                current_type = value
            consecutive_counts.append(current_count if value else 0)
        
        return consecutive_counts
    
    def _add_rejection_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rejection wick analysis"""
        
        # Calculate candle components
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Rejection wicks (avoid division by zero)
        mask_valid_range = df['total_range'] > 0
        df['upper_rejection'] = False
        df['lower_rejection'] = False
        
        if mask_valid_range.any():
            df.loc[mask_valid_range, 'upper_rejection'] = (
                (df.loc[mask_valid_range, 'upper_wick'] > 
                 df.loc[mask_valid_range, 'total_range'] * 0.5) & 
                (df.loc[mask_valid_range, 'upper_wick'] > 
                 df.loc[mask_valid_range, 'body_size'] * 2)
            )
            
            df.loc[mask_valid_range, 'lower_rejection'] = (
                (df.loc[mask_valid_range, 'lower_wick'] > 
                 df.loc[mask_valid_range, 'total_range'] * 0.5) & 
                (df.loc[mask_valid_range, 'lower_wick'] > 
                 df.loc[mask_valid_range, 'body_size'] * 2)
            )
        
        # Count rejection wicks in recent periods
        df['rejection_wicks_count'] = self._rolling_rejection_count(df)
        
        return df
    
    def _rolling_rejection_count(self, df: pd.DataFrame, window: int = 10) -> list:
        """Count rejection wicks in rolling window"""
        rejection_counts = []
        
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            upper_rejections = df['upper_rejection'].iloc[start_idx:i+1].sum()
            lower_rejections = df['lower_rejection'].iloc[start_idx:i+1].sum()
            rejection_counts.append(upper_rejections + lower_rejections)
        
        return rejection_counts
    
    def _add_consolidation_analysis(self, df: pd.DataFrame, pip_multiplier: int) -> pd.DataFrame:
        """Add consolidation and breakout analysis"""
        
        # Consolidation range
        df['consolidation_range_pips'] = self._calculate_consolidation_range(
            df, pip_multiplier
        )
        
        # Bars since breakout
        df['bars_since_breakout'] = self._detect_breakouts(df)
        
        # Bars at current level
        df['bars_at_current_level'] = self._bars_at_level(df, pip_multiplier)
        
        return df
    
    def _calculate_consolidation_range(
        self, 
        df: pd.DataFrame, 
        pip_multiplier: int, 
        window: int = 20
    ) -> list:
        """Calculate consolidation range in pips"""
        ranges = []
        
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            period_high = df['high'].iloc[start_idx:i+1].max()
            period_low = df['low'].iloc[start_idx:i+1].min()
            range_pips = (period_high - period_low) * pip_multiplier
            ranges.append(range_pips)
        
        return ranges
    
    def _detect_breakouts(self, df: pd.DataFrame, window: int = 20) -> list:
        """Detect breakouts and count bars since"""
        breakout_bars = []
        
        for i in range(len(df)):
            if i < window:
                breakout_bars.append(0)
                continue
            
            # Look back for recent range
            start_idx = max(0, i - window)
            recent_high = df['high'].iloc[start_idx:i].max()
            recent_low = df['low'].iloc[start_idx:i].min()
            
            # Check if current bar breaks out
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            if current_high > recent_high or current_low < recent_low:
                bars_since = 1  # Breakout bar
            else:
                bars_since = breakout_bars[-1] + 1 if breakout_bars else 1
            
            breakout_bars.append(bars_since)
        
        return breakout_bars
    
    def _bars_at_level(
        self, 
        df: pd.DataFrame, 
        pip_multiplier: int, 
        tolerance_pips: int = 5
    ) -> list:
        """Count bars at current price level"""
        level_bars = []
        
        for i in range(len(df)):
            if i == 0:
                level_bars.append(1)
                continue
            
            current_price = df['close'].iloc[i]
            bars_at_current = 1
            
            # Look backwards
            for j in range(i-1, -1, -1):
                past_price = df['close'].iloc[j]
                if abs(current_price - past_price) * pip_multiplier <= tolerance_pips:
                    bars_at_current += 1
                else:
                    break
            
            level_bars.append(bars_at_current)
        
        return level_bars