# core/strategies/helpers/macd_indicator_calculator.py
"""
MACD Indicator Calculator Module
Handles MACD calculations, crossover detection and data validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional


class MACDIndicatorCalculator:
    """Calculates MACD indicators and detects crossovers - lightweight and focused"""
    
    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps
        
        # Default MACD parameters (can be overridden)
        self.default_config = {
            'fast_ema': 12,
            'slow_ema': 26, 
            'signal_ema': 9
        }
    
    def get_required_indicators(self, macd_config: Dict = None) -> List[str]:
        """Get list of required indicators for MACD strategy"""
        config = macd_config or self.default_config
        return [
            f'ema_{config["fast_ema"]}',
            f'ema_{config["slow_ema"]}',
            'macd_line',
            'macd_signal', 
            'macd_histogram',
            'ema_200'  # For trend filter
        ]
    
    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int = 50) -> bool:
        """Validate that we have enough data for MACD calculations"""
        if len(df) < min_bars:
            self.logger.debug(f"Insufficient data: {len(df)} bars (need {min_bars})")
            return False
            
        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
            
        return True
    
    def ensure_macd_indicators(self, df: pd.DataFrame, macd_config: Dict = None) -> pd.DataFrame:
        """
        Calculate MACD indicators if not present
        
        Args:
            df: DataFrame with price data
            macd_config: MACD configuration (fast_ema, slow_ema, signal_ema)
            
        Returns:
            DataFrame with MACD indicators added
        """
        config = macd_config or self.default_config
        df_copy = df.copy()
        
        try:
            # Calculate EMAs needed for MACD
            fast_ema_col = f'ema_{config["fast_ema"]}'
            slow_ema_col = f'ema_{config["slow_ema"]}'
            
            if fast_ema_col not in df_copy.columns:
                df_copy[fast_ema_col] = df_copy['close'].ewm(span=config['fast_ema']).mean()
                
            if slow_ema_col not in df_copy.columns:
                df_copy[slow_ema_col] = df_copy['close'].ewm(span=config['slow_ema']).mean()
            
            # Calculate MACD line (fast EMA - slow EMA)
            if 'macd_line' not in df_copy.columns:
                df_copy['macd_line'] = df_copy[fast_ema_col] - df_copy[slow_ema_col]
            
            # Calculate MACD signal line (EMA of MACD line)
            if 'macd_signal' not in df_copy.columns:
                df_copy['macd_signal'] = df_copy['macd_line'].ewm(span=config['signal_ema']).mean()
                
            # Calculate MACD histogram (MACD line - signal line)
            if 'macd_histogram' not in df_copy.columns:
                df_copy['macd_histogram'] = df_copy['macd_line'] - df_copy['macd_signal']
            
            # Add EMA 200 for trend filter if not present
            if 'ema_200' not in df_copy.columns:
                df_copy['ema_200'] = df_copy['close'].ewm(span=200).mean()
            
            self.logger.debug("MACD indicators calculated successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD indicators: {e}")
            return df_copy
    
    def detect_macd_crossovers(self, df: pd.DataFrame, epic: str = '') -> pd.DataFrame:
        """
        Detect MACD histogram crossovers with strength filtering (main signal generation)
        
        Args:
            df: DataFrame with MACD indicators
            epic: Trading pair epic for strength threshold determination
            
        Returns:
            DataFrame with crossover signals added
        """
        df_copy = df.copy()
        
        try:
            # Ensure we have MACD data
            required_cols = ['macd_line', 'macd_signal', 'macd_histogram']
            if not all(col in df_copy.columns for col in required_cols):
                self.logger.error("Missing MACD indicators for crossover detection")
                return df_copy
            
            # Initialize signal columns
            df_copy['bull_crossover'] = False
            df_copy['bear_crossover'] = False
            df_copy['bull_alert'] = False
            df_copy['bear_alert'] = False
            
            # Detect histogram crossovers (zero line crosses)
            df_copy['histogram_prev'] = df_copy['macd_histogram'].shift(1)
            
            # Determine strength threshold based on currency pair
            strength_threshold = self.get_histogram_strength_threshold(epic)
            
            # Bull signal: histogram crosses above zero AND meets strength threshold
            bull_cross = (
                (df_copy['macd_histogram'] > 0) & 
                (df_copy['histogram_prev'] <= 0) &
                (df_copy['macd_histogram'] >= strength_threshold)  # STRENGTH FILTER RESTORED
            )
            
            # Bear signal: histogram crosses below zero AND meets strength threshold  
            bear_cross = (
                (df_copy['macd_histogram'] < 0) & 
                (df_copy['histogram_prev'] >= 0) &
                (df_copy['macd_histogram'] <= -strength_threshold)  # STRENGTH FILTER RESTORED
            )
            
            df_copy['bull_crossover'] = bull_cross
            df_copy['bear_crossover'] = bear_cross
            
            # For compatibility, also set alert flags
            df_copy['bull_alert'] = bull_cross
            df_copy['bear_alert'] = bear_cross
            
            # Log crossover detection with strength filtering info
            bull_count = bull_cross.sum()
            bear_count = bear_cross.sum()
            if bull_count > 0 or bear_count > 0:
                self.logger.debug(f"MACD crossovers detected - Bull: {bull_count}, Bear: {bear_count} (threshold: {strength_threshold})")
            
            # Log filtered signals if any crossovers were filtered out
            raw_bull = ((df_copy['macd_histogram'] > 0) & (df_copy['histogram_prev'] <= 0)).sum()
            raw_bear = ((df_copy['macd_histogram'] < 0) & (df_copy['histogram_prev'] >= 0)).sum()
            if raw_bull > bull_count or raw_bear > bear_count:
                self.logger.debug(f"Strength filter removed {raw_bull - bull_count} bull + {raw_bear - bear_count} bear signals")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD crossovers: {e}")
            return df_copy
    
    def get_histogram_strength_threshold(self, epic: str) -> float:
        """
        Get MACD histogram strength threshold based on currency pair
        
        JPY pairs: 0.010 minimum histogram strength (calibrated from actual data)
        Non-JPY pairs: 0.00010 minimum histogram strength (calibrated from actual data)
        
        Args:
            epic: Trading pair epic (e.g., 'CS.D.USDJPY.MINI.IP')
            
        Returns:
            Minimum histogram strength threshold
        """
        try:
            # Check if this is a JPY pair
            if 'JPY' in epic.upper():
                threshold = 0.010  # JPY pairs - calibrated from actual market data
                self.logger.debug(f"Using JPY histogram threshold: {threshold}")
            else:
                threshold = 0.00010  # Non-JPY pairs - calibrated from actual market data
                self.logger.debug(f"Using non-JPY histogram threshold: {threshold}")
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Error determining histogram threshold for {epic}: {e}")
            return 0.00010  # Default to non-JPY threshold
    
    def validate_macd_strength(self, row: pd.Series, signal_type: str, threshold: float = 0.0001) -> bool:
        """
        Validate MACD signal strength
        
        Args:
            row: DataFrame row with MACD data
            signal_type: 'BULL' or 'BEAR'
            threshold: Minimum histogram value for valid signal
            
        Returns:
            True if signal is strong enough
        """
        try:
            histogram = row.get('macd_histogram', 0)
            
            if signal_type == 'BULL':
                # Bull signals need positive histogram above threshold
                return histogram > threshold
            elif signal_type == 'BEAR':
                # Bear signals need negative histogram below negative threshold  
                return histogram < -threshold
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating MACD strength: {e}")
            return False
    
    def get_macd_momentum_score(self, row: pd.Series) -> float:
        """
        Calculate MACD momentum score (0.0 to 1.0)
        
        Args:
            row: DataFrame row with MACD data
            
        Returns:
            Momentum score between 0.0 and 1.0
        """
        try:
            histogram = row.get('macd_histogram', 0)
            macd_line = row.get('macd_line', 0)
            macd_signal = row.get('macd_signal', 0)
            
            # Base score from histogram strength
            hist_score = min(1.0, abs(histogram) / 0.001) if histogram != 0 else 0.0
            
            # Bonus for MACD line and signal alignment
            line_signal_aligned = (
                (macd_line > macd_signal and histogram > 0) or
                (macd_line < macd_signal and histogram < 0)
            )
            alignment_bonus = 0.2 if line_signal_aligned else 0.0
            
            return min(1.0, hist_score + alignment_bonus)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.5  # Neutral score on error