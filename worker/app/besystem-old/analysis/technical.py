# ================================
# 4. analysis/technical.py
# ================================

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, List

class TechnicalAnalysis:
    """Complete technical analysis engine"""
    
    @staticmethod
    def add_ema_indicators(df: pd.DataFrame, periods: List[int] = [9, 21, 200]) -> pd.DataFrame:
        """Add EMA indicators"""
        df_ema = df.copy()
        for period in periods:
            df_ema[f'ema_{period}'] = df_ema['close'].ewm(span=period, adjust=False).mean()
        return df_ema
    
    @staticmethod
    def find_support_resistance_levels(df: pd.DataFrame, window: int = 20) -> Dict:
        """Find support and resistance levels using local extrema"""
        highs = df['high'].values
        lows = df['low'].values
        
        # Find local extrema
        resistance_indices = argrelextrema(highs, np.greater, order=window)[0]
        support_indices = argrelextrema(lows, np.less, order=window)[0]
        
        resistance_levels = highs[resistance_indices]
        support_levels = lows[support_indices]
        
        # Cluster nearby levels
        def cluster_levels(levels, tolerance_pct=0.001):
            if len(levels) == 0:
                return []
            
            levels = np.sort(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance_pct:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        return {
            'resistance_levels': sorted(cluster_levels(resistance_levels), reverse=True),
            'support_levels': sorted(cluster_levels(support_levels), reverse=True)
        }
    
    @staticmethod
    def add_volume_analysis(df: pd.DataFrame, periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """Add volume analysis indicators"""
        if 'ltv' not in df.columns:
            return df
        
        df_vol = df.copy()
        df_vol['ltv'] = df_vol['ltv'].fillna(0)
        
        # Volume moving averages and ratios
        for period in periods:
            df_vol[f'volume_avg_{period}'] = df_vol['ltv'].rolling(window=period, min_periods=1).mean()
            df_vol[f'volume_ratio_{period}'] = df_vol['ltv'] / df_vol[f'volume_avg_{period}'].replace(0, np.nan)
        
        # Volume percentile ranking
        df_vol['volume_percentile_50'] = df_vol['ltv'].rolling(window=50, min_periods=1).rank(pct=True) * 100
        
        return df_vol
    
    def enhance_dataframe(self, df: pd.DataFrame, epic_pair: str = 'EURUSD') -> pd.DataFrame:
        """Complete dataframe enhancement"""
        # Add EMAs
        df_enhanced = self.add_ema_indicators(df)
        
        # Add volume analysis
        df_enhanced = self.add_volume_analysis(df_enhanced)
        
        # Add support/resistance
        df_enhanced = self._add_support_resistance_to_df(df_enhanced, epic_pair)
        
        # Add market behavior
        df_enhanced = self._add_market_behavior(df_enhanced, epic_pair)
        
        return df_enhanced
    
    def _add_support_resistance_to_df(self, df: pd.DataFrame, epic_pair: str) -> pd.DataFrame:
        """Add support/resistance levels to dataframe"""
        df_enhanced = df.copy()
        
        # Determine pip multiplier
        pip_multiplier = 100 if 'JPY' in epic_pair.upper() else 10000
        
        # Find support/resistance levels
        sr_levels = self.find_support_resistance_levels(df_enhanced)
        
        # Calculate distances for each row
        nearest_resistance = []
        nearest_support = []
        distance_to_resistance_pips = []
        distance_to_support_pips = []
        
        for _, row in df_enhanced.iterrows():
            current_price = row['close']
            
            # Find nearest resistance (above current price)
            resistance_above = [r for r in sr_levels['resistance_levels'] if r > current_price]
            nearest_res = min(resistance_above) if resistance_above else None
            
            # Find nearest support (below current price)
            support_below = [s for s in sr_levels['support_levels'] if s < current_price]
            nearest_sup = max(support_below) if support_below else None
            
            # Calculate distances in pips
            dist_to_res = (nearest_res - current_price) * pip_multiplier if nearest_res else np.nan
            dist_to_sup = (current_price - nearest_sup) * pip_multiplier if nearest_sup else np.nan
            
            nearest_resistance.append(nearest_res)
            nearest_support.append(nearest_sup)
            distance_to_resistance_pips.append(dist_to_res)
            distance_to_support_pips.append(dist_to_sup)
        
        df_enhanced['nearest_resistance'] = nearest_resistance
        df_enhanced['nearest_support'] = nearest_support
        df_enhanced['distance_to_resistance_pips'] = distance_to_resistance_pips
        df_enhanced['distance_to_support_pips'] = distance_to_support_pips
        
        return df_enhanced
    
    def _add_market_behavior(self, df: pd.DataFrame, epic_pair: str) -> pd.DataFrame:
        """Add market behavior indicators"""
        df_behavior = df.copy()
        
        pip_multiplier = 100 if 'JPY' in epic_pair.upper() else 10000
        
        # Price changes in pips
        df_behavior['price_change_1_bar_pips'] = (df_behavior['close'] - df_behavior['close'].shift(1)) * pip_multiplier
        df_behavior['price_change_4_bars_pips'] = (df_behavior['close'] - df_behavior['close'].shift(4)) * pip_multiplier
        
        # Candle types
        df_behavior['is_green'] = df_behavior['close'] > df_behavior['open']
        df_behavior['is_red'] = df_behavior['close'] < df_behavior['open']
        
        # Trend analysis
        df_behavior['trend_short'] = self._calculate_trend(df_behavior, 10)
        df_behavior['trend_medium'] = self._calculate_trend(df_behavior, 20)
        df_behavior['trend_long'] = self._calculate_trend(df_behavior, 50)
        
        return df_behavior
    
    def _calculate_trend(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate trend direction"""
        trend = []
        for i in range(len(df)):
            if i < period:
                trend.append('ranging')
                continue
            
            recent_data = df.iloc[max(0, i-period):i+1]
            slope = np.polyfit(range(len(recent_data)), recent_data['close'], 1)[0]
            
            threshold = recent_data['close'].iloc[-1] * 0.0001
            if slope > threshold:
                trend.append('bullish')
            elif slope < -threshold:
                trend.append('bearish')
            else:
                trend.append('ranging')
        
        return pd.Series(trend, index=df.index)
