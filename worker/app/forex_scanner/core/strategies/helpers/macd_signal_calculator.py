# core/strategies/helpers/macd_signal_calculator.py
"""
MACD Signal Calculator Module
Handles confidence calculation and signal strength assessment for MACD strategy
"""

import pandas as pd
import logging
from typing import Optional, Dict
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class MACDSignalCalculator:
    """Calculates confidence scores and signal strength for MACD signals"""
    
    def __init__(self, logger: logging.Logger = None, trend_validator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.30)  # TEMP: Lower for debugging
    
    def calculate_simple_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        SIMPLE CONFIDENCE CALCULATION: Based on MACD histogram strength and trend alignment
        
        Factors considered:
        - MACD histogram strength (most important)
        - MACD line vs signal line alignment
        - EMA 200 trend alignment
        - Price position relative to EMA 200
        
        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Confidence score between 0.0 and 0.95
        """
        try:
            base_confidence = 0.5  # Start with 50%
            
            # Get MACD values
            macd_histogram = latest_row.get('macd_histogram', 0)
            macd_line = latest_row.get('macd_line', 0)
            macd_signal = latest_row.get('macd_signal', 0)
            ema_200 = latest_row.get('ema_200', 0)
            close = latest_row.get('close', 0)
            
            if macd_histogram == 0:
                return 0.3  # Low confidence if MACD missing
            
            # 1. MACD HISTOGRAM STRENGTH (50% weight)
            # Strong histogram values indicate strong momentum
            histogram_abs = abs(macd_histogram)
            
            if signal_type == 'BULL' and macd_histogram > 0:
                # Bullish histogram strength
                if histogram_abs > 0.001:
                    histogram_boost = 0.25  # Very strong
                elif histogram_abs > 0.0005:
                    histogram_boost = 0.15  # Strong
                elif histogram_abs > 0.0001:
                    histogram_boost = 0.10  # Moderate
                else:
                    histogram_boost = 0.05  # Weak
            elif signal_type == 'BEAR' and macd_histogram < 0:
                # Bearish histogram strength
                if histogram_abs > 0.001:
                    histogram_boost = 0.25  # Very strong
                elif histogram_abs > 0.0005:
                    histogram_boost = 0.15  # Strong
                elif histogram_abs > 0.0001:
                    histogram_boost = 0.10  # Moderate
                else:
                    histogram_boost = 0.05  # Weak
            else:
                # Wrong direction histogram
                return 0.2  # Very low confidence
            
            base_confidence += histogram_boost
            
            # 2. MACD LINE vs SIGNAL LINE ALIGNMENT (20% weight)
            if signal_type == 'BULL' and macd_line > macd_signal:
                base_confidence += 0.10  # MACD above signal is bullish
            elif signal_type == 'BEAR' and macd_line < macd_signal:
                base_confidence += 0.10  # MACD below signal is bearish
            else:
                base_confidence -= 0.05  # Misalignment penalty
            
            # 3. EMA 200 TREND ALIGNMENT (20% weight)
            if ema_200 > 0 and close > 0:
                if signal_type == 'BULL' and close > ema_200:
                    base_confidence += 0.10  # Price above EMA 200 supports bullish
                elif signal_type == 'BEAR' and close < ema_200:
                    base_confidence += 0.10  # Price below EMA 200 supports bearish
                else:
                    base_confidence -= 0.05  # Against major trend penalty
            
            # 4. MACD MOMENTUM DIRECTION (10% weight)
            # Check if MACD line is moving in signal direction
            macd_prev = latest_row.get('macd_line_prev', macd_line)
            if signal_type == 'BULL' and macd_line > macd_prev:
                base_confidence += 0.05  # MACD rising supports bullish
            elif signal_type == 'BEAR' and macd_line < macd_prev:
                base_confidence += 0.05  # MACD falling supports bearish
            
            # Ensure confidence stays within bounds
            confidence = max(0.1, min(0.95, base_confidence))
            
            self.logger.debug(f"MACD confidence calculated: {confidence:.3f} for {signal_type}")
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD confidence: {e}")
            return 0.4  # Conservative fallback
    
    def validate_confidence_threshold(self, confidence: float) -> bool:
        """
        Validate that confidence meets minimum threshold
        
        Args:
            confidence: Calculated confidence score
            
        Returns:
            True if confidence meets threshold
        """
        passes = confidence >= self.min_confidence
        if not passes:
            self.logger.debug(f"Signal rejected: confidence {confidence:.3f} < threshold {self.min_confidence}")
        return passes
    
    def calculate_macd_strength_factor(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate MACD-specific strength factor based on histogram and momentum
        
        Args:
            latest_row: DataFrame row with MACD data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Strength factor between 0.0 and 2.0 (multiplier)
        """
        try:
            histogram = latest_row.get('macd_histogram', 0)
            macd_line = latest_row.get('macd_line', 0)
            macd_signal_line = latest_row.get('macd_signal', 0)
            
            # Base strength from histogram magnitude
            histogram_strength = min(2.0, abs(histogram) / 0.0005) if histogram != 0 else 0.5
            
            # Direction alignment check
            if signal_type == 'BULL':
                direction_correct = histogram > 0 and macd_line > macd_signal_line
            else:
                direction_correct = histogram < 0 and macd_line < macd_signal_line
            
            # Apply direction penalty if misaligned
            if not direction_correct:
                histogram_strength *= 0.5
            
            return max(0.1, min(2.0, histogram_strength))
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD strength factor: {e}")
            return 1.0  # Neutral strength on error
    
    def get_signal_quality_score(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """
        Get detailed signal quality breakdown for debugging/analysis
        
        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Dictionary with quality score components
        """
        try:
            histogram = latest_row.get('macd_histogram', 0)
            macd_line = latest_row.get('macd_line', 0)
            macd_signal = latest_row.get('macd_signal', 0)
            close = latest_row.get('close', 0)
            ema_200 = latest_row.get('ema_200', 0)
            
            return {
                'histogram_strength': abs(histogram),
                'histogram_direction_ok': (
                    (signal_type == 'BULL' and histogram > 0) or
                    (signal_type == 'BEAR' and histogram < 0)
                ),
                'macd_signal_aligned': (
                    (signal_type == 'BULL' and macd_line > macd_signal) or
                    (signal_type == 'BEAR' and macd_line < macd_signal)
                ),
                'ema200_trend_aligned': (
                    (signal_type == 'BULL' and close > ema_200) or
                    (signal_type == 'BEAR' and close < ema_200)
                ) if ema_200 > 0 and close > 0 else None,
                'overall_confidence': self.calculate_simple_confidence(latest_row, signal_type)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting signal quality score: {e}")
            return {'error': str(e)}