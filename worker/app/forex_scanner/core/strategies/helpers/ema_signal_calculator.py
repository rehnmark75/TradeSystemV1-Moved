# core/strategies/helpers/ema_signal_calculator.py
"""
EMA Signal Calculator Module
Handles confidence calculation and signal strength assessment for EMA strategy
"""

import pandas as pd
import logging
from typing import Optional
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class EMASignalCalculator:
    """Calculates confidence scores and signal strength for EMA signals"""
    
    def __init__(self, logger: logging.Logger = None, trend_validator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)
    
    def calculate_simple_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        SIMPLE CONFIDENCE CALCULATION: Based on EMA alignment and crossover strength
        
        Factors considered:
        - EMA trend alignment (most important)
        - Price position relative to EMAs
        - MACD histogram alignment (if available)
        - Crossover strength (how far from EMA)
        
        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Confidence score between 0.0 and 0.95
        """
        try:
            base_confidence = 0.5  # Start with 50%
            
            # Get EMA values
            ema_short = latest_row.get('ema_short', 0)
            ema_long = latest_row.get('ema_long', 0)
            ema_trend = latest_row.get('ema_trend', 0)
            close = latest_row.get('close', 0)
            
            if ema_short == 0 or ema_long == 0 or ema_trend == 0:
                return 0.3  # Low confidence if EMAs missing
            
            # 1. EMA TREND ALIGNMENT (40% weight)
            if signal_type == 'BULL':
                # Bull signal: check if ema_short > ema_long > ema_trend
                if ema_short > ema_long and ema_long > ema_trend:
                    base_confidence += 0.3  # Strong alignment
                elif ema_short > ema_long:
                    base_confidence += 0.1  # Partial alignment
            else:  # BEAR
                # Bear signal: check if ema_short < ema_long < ema_trend  
                if ema_short < ema_long and ema_long < ema_trend:
                    base_confidence += 0.3  # Strong alignment
                elif ema_short < ema_long:
                    base_confidence += 0.1  # Partial alignment
            
            # 2. PRICE POSITION (20% weight)
            if signal_type == 'BULL':
                if close > ema_short and close > ema_long:
                    base_confidence += 0.15  # Price above EMAs
                elif close > ema_short:
                    base_confidence += 0.05  # Price above short EMA
            else:  # BEAR
                if close < ema_short and close < ema_long:
                    base_confidence += 0.15  # Price below EMAs
                elif close < ema_short:
                    base_confidence += 0.05  # Price below short EMA
            
            # 3. MACD CONFIRMATION (15% weight)
            macd_histogram = latest_row.get('macd_histogram', 0)
            if macd_histogram != 0:
                if signal_type == 'BULL' and macd_histogram > 0:
                    base_confidence += 0.1  # MACD supports bull signal
                elif signal_type == 'BEAR' and macd_histogram < 0:
                    base_confidence += 0.1  # MACD supports bear signal
            
            # 4. CROSSOVER STRENGTH (10% weight)
            # Check how strong the crossover was
            bull_cross = latest_row.get('bull_cross', False)
            bear_cross = latest_row.get('bear_cross', False)
            
            if (signal_type == 'BULL' and bull_cross) or (signal_type == 'BEAR' and bear_cross):
                base_confidence += 0.05  # Confirmed crossover
            
            # 5. EMA SEPARATION (15% weight) - EMAs not too close
            ema_separation = abs(ema_short - ema_long)
            if ema_separation > 0.0001:  # Reasonable separation for forex
                base_confidence += 0.1
            
            # 6. TWO-POLE OSCILLATOR VALIDATION (Optional - 15% weight)
            if getattr(config, 'TWO_POLE_OSCILLATOR_ENABLED', False) and self.trend_validator:
                two_pole_boost = self.trend_validator.validate_two_pole_oscillator(latest_row, signal_type)
                base_confidence += two_pole_boost
                if two_pole_boost > 0:
                    self.logger.debug(f"Two-Pole Oscillator boost: +{two_pole_boost:.1%}")
            
            # Cap confidence at 95%
            final_confidence = min(0.95, base_confidence)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating simple confidence: {e}")
            return 0.4  # Safe fallback
    
    def calculate_crossover_strength(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate the strength of the EMA crossover
        
        Args:
            latest_row: DataFrame row with price and EMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Strength value between 0.0 and 1.0
        """
        try:
            close = latest_row.get('close', 0)
            ema_short = latest_row.get('ema_short', 0)
            
            if close == 0 or ema_short == 0:
                return 0.0
            
            # Calculate distance from EMA as percentage
            distance = abs(close - ema_short) / close
            
            # Convert to strength (closer = weaker, farther = stronger)
            # But cap at reasonable values for forex (typically < 1% moves)
            max_distance = 0.01  # 1% maximum expected distance
            strength = min(1.0, distance / max_distance)
            
            # Validate direction
            if signal_type == 'BULL':
                # For bull signal, price should be above EMA
                if close <= ema_short:
                    return 0.0
            else:  # BEAR
                # For bear signal, price should be below EMA
                if close >= ema_short:
                    return 0.0
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating crossover strength: {e}")
            return 0.0
    
    def calculate_trend_strength(self, latest_row: pd.Series) -> float:
        """
        Calculate overall trend strength based on EMA alignment
        
        Returns:
            Trend strength value between -1.0 (strong bear) and 1.0 (strong bull)
        """
        try:
            ema_short = latest_row.get('ema_short', 0)
            ema_long = latest_row.get('ema_long', 0)
            ema_trend = latest_row.get('ema_trend', 0)
            
            if ema_short == 0 or ema_long == 0 or ema_trend == 0:
                return 0.0
            
            # Calculate separations
            short_long_sep = (ema_short - ema_long) / ema_long
            long_trend_sep = (ema_long - ema_trend) / ema_trend
            
            # Average separation indicates trend strength
            avg_separation = (short_long_sep + long_trend_sep) / 2
            
            # Cap at reasonable values
            return max(-1.0, min(1.0, avg_separation * 100))
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def validate_confidence_threshold(self, confidence: float) -> bool:
        """
        Check if confidence meets minimum threshold
        
        Args:
            confidence: Calculated confidence score
            
        Returns:
            True if confidence meets threshold, False otherwise
        """
        if confidence < self.min_confidence:
            self.logger.debug(f"Signal confidence {confidence:.1%} below threshold {self.min_confidence:.1%}")
            return False
        return True