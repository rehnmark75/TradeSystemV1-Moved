# analysis/volume.py
"""
Volume Analysis Module
Handles volume-based indicators and analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import List


class VolumeAnalyzer:
    """Volume analysis and indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_volume_analysis(
        self, 
        df: pd.DataFrame, 
        periods: List[int] = [10, 20, 50]
    ) -> pd.DataFrame:
        """
        Add comprehensive volume analysis to DataFrame
        
        Args:
            df: DataFrame with OHLC and volume data
            periods: Periods for volume moving averages
            
        Returns:
            DataFrame with volume indicators
        """
        if 'ltv' not in df.columns:
            self.logger.warning("No volume data (ltv column) found")
            return df
        
        df_vol = df.copy()
        
        # Handle NaN values in volume
        df_vol['ltv'] = df_vol['ltv'].fillna(0)
        
        # Volume moving averages
        for period in periods:
            df_vol[f'volume_avg_{period}'] = df_vol['ltv'].rolling(
                window=period, min_periods=1
            ).mean()
        
        # Volume ratios (current vs averages)
        for period in periods:
            avg_col = f'volume_avg_{period}'
            ratio_col = f'volume_ratio_{period}'
            df_vol[ratio_col] = df_vol['ltv'] / df_vol[avg_col].replace(0, np.nan)
        
        # Volume percentile ranking
        df_vol['volume_percentile_50'] = df_vol['ltv'].rolling(
            window=50, min_periods=1
        ).rank(pct=True) * 100
        
        # Volume-Price Trend (VPT)
        df_vol['price_change_pct'] = df_vol['close'].pct_change()
        df_vol['vpt'] = (df_vol['ltv'] * df_vol['price_change_pct']).cumsum()
        
        # Volume Rate of Change
        df_vol['volume_roc_10'] = df_vol['ltv'].pct_change(periods=10) * 100
        
        # Volume classifications
        df_vol['volume_high'] = df_vol['ltv'] > (df_vol['volume_avg_20'] * 1.5)
        df_vol['volume_low'] = df_vol['ltv'] < (df_vol['volume_avg_20'] * 0.5)
        df_vol['volume_spike'] = df_vol['ltv'] > (df_vol['volume_avg_20'] * 2.0)
        
        # Accumulation/Distribution approximation
        df_vol = self._add_accumulation_distribution(df_vol)

        # Money Flow Index (volume-weighted RSI)
        df_vol = self._add_money_flow_index(df_vol)

        return df_vol
    
    def _add_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Accumulation/Distribution indicator"""
        
        # Money Flow Multiplier
        df['money_flow_multiplier'] = (
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / 
            (df['high'] - df['low'])
        )
        
        # Handle division by zero
        df['money_flow_multiplier'] = df['money_flow_multiplier'].fillna(0)
        
        # Money Flow Volume
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['ltv']
        
        # Accumulation/Distribution Line
        df['accumulation_distribution'] = df['money_flow_volume'].cumsum()

        return df

    def _add_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Money Flow Index (volume-weighted RSI, 0-100 scale)

        Uses tick volume (ltv) as a proxy for real volume, which is standard
        for forex where true volume is unavailable.
        """
        if 'ltv' not in df.columns:
            return df

        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # Raw money flow = typical price * volume
        raw_mf = tp * df['ltv']

        # Separate into positive (rising TP) and negative (falling TP)
        tp_change = tp.diff()
        positive_mf = raw_mf.where(tp_change > 0, 0.0)
        negative_mf = raw_mf.where(tp_change < 0, 0.0)

        # Rolling sums over period
        pos_sum = positive_mf.rolling(window=period, min_periods=1).sum()
        neg_sum = negative_mf.rolling(window=period, min_periods=1).sum()

        # MFI = 100 - (100 / (1 + money_flow_ratio))
        mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
        df['mfi'] = 100 - (100 / (1 + mf_ratio))
        df['mfi'] = df['mfi'].fillna(50.0)  # Neutral default when undefined

        # MFI slope over 3 bars for direction detection
        df['mfi_slope'] = df['mfi'].diff(3)

        return df