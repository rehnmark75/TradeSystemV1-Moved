#!/usr/bin/env python3
"""
ADX (Average Directional Index) Calculator for Trend Strength Analysis

The ADX is a technical indicator used to measure the strength of a trend,
regardless of its direction. It's particularly useful for filtering out
signals during ranging/sideways markets.

ADX Values:
- ADX > 25: Strong trending market (good for trend-following strategies)
- ADX 20-25: Moderate trend
- ADX < 20: Weak trend or ranging market (avoid trend-following signals)

Author: Claude Code Assistant
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict


class ADXCalculator:
    """
    Calculate ADX (Average Directional Index) for trend strength analysis
    """
    
    def __init__(self, period: int = 14, logger: Optional[logging.Logger] = None):
        """
        Initialize ADX calculator
        
        Args:
            period: ADX calculation period (default 14)
            logger: Optional logger instance
        """
        self.period = period
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX, +DI, and -DI for the given DataFrame
        
        Args:
            df: DataFrame with OHLC data (columns: high, low, close)
            
        Returns:
            DataFrame with additional columns: ADX, DI_plus, DI_minus, TR, DM_plus, DM_minus
        """
        try:
            if len(df) < self.period + 1:
                self.logger.warning(f"Insufficient data for ADX calculation: {len(df)} < {self.period + 1}")
                return df
            
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                # Try alternative column names
                alt_mapping = {
                    'high': ['mid_h', 'High', 'HIGH'],
                    'low': ['mid_l', 'Low', 'LOW'], 
                    'close': ['mid_c', 'Close', 'CLOSE']
                }
                
                for missing in missing_cols:
                    found = False
                    for alt in alt_mapping.get(missing, []):
                        if alt in df.columns:
                            df[missing] = df[alt]
                            found = True
                            break
                    
                    if not found:
                        self.logger.error(f"Missing required column for ADX: {missing}")
                        return df
            
            # Calculate True Range (TR)
            df['TR'] = self._calculate_true_range(df)
            
            # Calculate Directional Movement (+DM and -DM)
            df['DM_plus'], df['DM_minus'] = self._calculate_directional_movement(df)
            
            # Calculate smoothed versions using Wilder's smoothing
            df['TR_smooth'] = self._wilders_smoothing(df['TR'], self.period)
            df['DM_plus_smooth'] = self._wilders_smoothing(df['DM_plus'], self.period)
            df['DM_minus_smooth'] = self._wilders_smoothing(df['DM_minus'], self.period)
            
            # Calculate Directional Indicators (+DI and -DI)
            df['DI_plus'] = 100 * (df['DM_plus_smooth'] / df['TR_smooth'])
            df['DI_minus'] = 100 * (df['DM_minus_smooth'] / df['TR_smooth'])
            
            # Calculate DX (Directional Index)
            di_sum = df['DI_plus'] + df['DI_minus']
            di_diff = abs(df['DI_plus'] - df['DI_minus'])
            df['DX'] = 100 * (di_diff / di_sum)
            
            # Calculate ADX using Wilder's smoothing on DX
            df['ADX'] = self._wilders_smoothing(df['DX'], self.period)
            
            # Clean up intermediate columns
            df.drop(['TR_smooth', 'DM_plus_smooth', 'DM_minus_smooth', 'DX'], axis=1, inplace=True, errors='ignore')
            
            self.logger.debug(f"ADX calculation completed for {len(df)} bars")
            return df
            
        except Exception as e:
            self.logger.error(f"ADX calculation failed: {e}")
            return df
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range (TR)"""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _calculate_directional_movement(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate +DM and -DM"""
        high = df['high']
        low = df['low']
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        dm_plus = pd.Series(0.0, index=df.index)
        dm_minus = pd.Series(0.0, index=df.index)
        
        # +DM when up_move > down_move and up_move > 0
        mask_plus = (up_move > down_move) & (up_move > 0)
        dm_plus[mask_plus] = up_move[mask_plus]
        
        # -DM when down_move > up_move and down_move > 0
        mask_minus = (down_move > up_move) & (down_move > 0)
        dm_minus[mask_minus] = down_move[mask_minus]
        
        return dm_plus, dm_minus
    
    def _wilders_smoothing(self, series: pd.Series, period: int) -> pd.Series:
        """
        Apply Wilder's smoothing (modified exponential moving average)
        
        Formula: 
        - First value: Simple average of first 'period' values
        - Subsequent values: (Previous * (period-1) + Current) / period
        """
        result = pd.Series(np.nan, index=series.index)
        
        # Need at least 'period' values to start
        if len(series) < period:
            return result
        
        # First smoothed value is simple average
        first_avg = series.iloc[:period].mean()
        result.iloc[period-1] = first_avg
        
        # Apply Wilder's smoothing formula
        for i in range(period, len(series)):
            if pd.notna(series.iloc[i]) and pd.notna(result.iloc[i-1]):
                result.iloc[i] = (result.iloc[i-1] * (period - 1) + series.iloc[i]) / period
        
        return result
    
    def get_trend_strength(self, adx_value: float, pair_multiplier: float = 1.0) -> Dict[str, any]:
        """
        Classify trend strength based on ADX value
        
        Args:
            adx_value: Current ADX value
            pair_multiplier: Pair-specific adjustment multiplier
            
        Returns:
            Dict with trend classification and metadata
        """
        # Adjust thresholds based on pair characteristics
        import config
        
        thresholds = getattr(config, 'ADX_THRESHOLDS', {
            'STRONG_TREND': 25.0,
            'MODERATE_TREND': 20.0,
            'WEAK_TREND': 15.0,
            'VERY_WEAK': 10.0
        })
        
        # Apply pair-specific multiplier to thresholds
        adjusted_thresholds = {
            key: value * pair_multiplier for key, value in thresholds.items()
        }
        
        if pd.isna(adx_value) or adx_value < 0:
            return {
                'strength': 'INVALID',
                'allow_signals': False,
                'confidence': 0.0,
                'description': 'Invalid ADX value',
                'adx_value': adx_value,
                'thresholds_used': adjusted_thresholds
            }
        
        if adx_value >= adjusted_thresholds['STRONG_TREND']:
            return {
                'strength': 'STRONG',
                'allow_signals': True,
                'confidence': 0.9,
                'description': f'Strong trending market (ADX: {adx_value:.1f})',
                'adx_value': adx_value,
                'thresholds_used': adjusted_thresholds
            }
        elif adx_value >= adjusted_thresholds['MODERATE_TREND']:
            return {
                'strength': 'MODERATE',
                'allow_signals': True,
                'confidence': 0.7,
                'description': f'Moderate trending market (ADX: {adx_value:.1f})',
                'adx_value': adx_value,
                'thresholds_used': adjusted_thresholds
            }
        elif adx_value >= adjusted_thresholds['WEAK_TREND']:
            return {
                'strength': 'WEAK',
                'allow_signals': False,
                'confidence': 0.4,
                'description': f'Weak trend, ranging market (ADX: {adx_value:.1f})',
                'adx_value': adx_value,
                'thresholds_used': adjusted_thresholds
            }
        else:
            return {
                'strength': 'VERY_WEAK',
                'allow_signals': False,
                'confidence': 0.1,
                'description': f'Very weak trend, ranging market (ADX: {adx_value:.1f})',
                'adx_value': adx_value,
                'thresholds_used': adjusted_thresholds
            }
    
    def validate_adx_signal(self, adx_value: float, epic: str, filter_mode: str = 'moderate') -> Dict[str, any]:
        """
        Validate if a signal should be allowed based on ADX trend strength
        
        Args:
            adx_value: Current ADX value
            epic: Trading pair identifier
            filter_mode: ADX filter mode ('strict', 'moderate', 'permissive', 'disabled')
            
        Returns:
            Dict with validation result and detailed reasoning
        """
        try:
            # Get pair-specific multiplier
            import config
            pair_multipliers = getattr(config, 'ADX_PAIR_MULTIPLIERS', {})
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.IP', '') if epic else 'DEFAULT'
            multiplier = pair_multipliers.get(pair_name, pair_multipliers.get('DEFAULT', 1.0))
            
            # Get trend strength classification
            trend_info = self.get_trend_strength(adx_value, multiplier)
            
            # Determine if signal should be allowed based on filter mode
            if filter_mode == 'disabled':
                is_valid = True
                reason = "ADX filter disabled"
            elif filter_mode == 'strict':
                is_valid = trend_info['strength'] == 'STRONG'
                reason = f"Strict mode: requires strong trend (ADX > {trend_info['thresholds_used']['STRONG_TREND']:.1f})"
            elif filter_mode == 'moderate':
                is_valid = trend_info['strength'] in ['STRONG', 'MODERATE']
                reason = f"Moderate mode: requires moderate+ trend (ADX > {trend_info['thresholds_used']['MODERATE_TREND']:.1f})"
            elif filter_mode == 'permissive':
                is_valid = trend_info['strength'] in ['STRONG', 'MODERATE', 'WEAK']
                reason = f"Permissive mode: requires weak+ trend (ADX > {trend_info['thresholds_used']['WEAK_TREND']:.1f})"
            else:
                is_valid = False
                reason = f"Unknown filter mode: {filter_mode}"
            
            return {
                'is_valid': is_valid,
                'trend_strength': trend_info['strength'],
                'adx_value': adx_value,
                'filter_mode': filter_mode,
                'pair_multiplier': multiplier,
                'reason': reason,
                'confidence': trend_info['confidence'],
                'description': trend_info['description'],
                'thresholds_used': trend_info['thresholds_used']
            }
            
        except Exception as e:
            self.logger.error(f"ADX signal validation failed: {e}")
            return {
                'is_valid': False,
                'trend_strength': 'ERROR',
                'adx_value': adx_value,
                'filter_mode': filter_mode,
                'reason': f"Validation error: {str(e)}",
                'confidence': 0.0
            }