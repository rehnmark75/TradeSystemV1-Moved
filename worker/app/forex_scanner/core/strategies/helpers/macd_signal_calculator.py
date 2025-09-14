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
        ENHANCED CONFIDENCE CALCULATION: Multi-factor analysis with restrictive weighting
        
        New weighting system (research-based 2025):
        - ADX strength (30%): Strong trending markets preferred
        - MACD histogram strength (25%): Core momentum signal
        - RSI confluence (20%): Overbought/oversold alignment  
        - EMA 200 trend alignment (15%): Major trend direction
        - MACD line vs signal alignment (10%): Additional confirmation
        
        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Confidence score between 0.0 and 0.95 (higher threshold required)
        """
        try:
            base_confidence = 0.65  # Start with 65% (raised from 50% for quality)
            
            # Get enhanced indicator values
            macd_histogram = latest_row.get('macd_histogram', 0)
            macd_line = latest_row.get('macd_line', 0)
            macd_signal = latest_row.get('macd_signal', 0)
            ema_200 = latest_row.get('ema_200', 0)
            close = latest_row.get('close', 0)
            adx = latest_row.get('adx', 0)
            rsi = latest_row.get('rsi', 50)
            
            # Divergence signals (premium quality boost)
            bullish_divergence = latest_row.get('bullish_divergence', False)
            bearish_divergence = latest_row.get('bearish_divergence', False)
            divergence_strength = latest_row.get('divergence_strength', 0)
            
            if macd_histogram == 0:
                return 0.3  # Low confidence if MACD missing
            
            # 1. ADX TREND STRENGTH ANALYSIS (30% weight) - NEW PRIMARY FACTOR
            adx_boost = 0.0
            if adx >= 40:
                adx_boost = 0.30  # Very strong trend
            elif adx >= 30:
                adx_boost = 0.20  # Strong trend
            elif adx >= 25:
                adx_boost = 0.10  # Moderate trend
            else:
                # Weak trend - penalize significantly for restrictive strategy
                return 0.25  # Too choppy for high-confidence signals
            
            base_confidence += adx_boost
            
            # 2. MACD HISTOGRAM STRENGTH (25% weight) - CORE MOMENTUM
            histogram_abs = abs(macd_histogram)
            histogram_boost = 0.0
            
            if signal_type == 'BULL' and macd_histogram > 0:
                if histogram_abs > 0.002:
                    histogram_boost = 0.25  # Very strong bullish momentum
                elif histogram_abs > 0.001:
                    histogram_boost = 0.20  # Strong
                elif histogram_abs > 0.0005:
                    histogram_boost = 0.15  # Moderate
                else:
                    histogram_boost = 0.05  # Weak
            elif signal_type == 'BEAR' and macd_histogram < 0:
                if histogram_abs > 0.002:
                    histogram_boost = 0.25  # Very strong bearish momentum
                elif histogram_abs > 0.001:
                    histogram_boost = 0.20  # Strong
                elif histogram_abs > 0.0005:
                    histogram_boost = 0.15  # Moderate
                else:
                    histogram_boost = 0.05  # Weak
            else:
                # Wrong direction histogram - critical failure
                return 0.20
            
            base_confidence += histogram_boost
            
            # 3. RSI CONFLUENCE ANALYSIS (20% weight) - OVERBOUGHT/OVERSOLD ALIGNMENT
            rsi_boost = 0.0
            if signal_type == 'BULL':
                if rsi < 30:
                    rsi_boost = 0.20  # Oversold - excellent for bull signals
                elif rsi < 50:
                    rsi_boost = 0.15  # Below midline - good
                elif rsi < 60:
                    rsi_boost = 0.10  # Neutral zone
                elif rsi < 70:
                    rsi_boost = 0.05  # Getting overbought
                else:
                    rsi_boost = -0.10  # Overbought - penalty
            else:  # BEAR
                if rsi > 70:
                    rsi_boost = 0.20  # Overbought - excellent for bear signals
                elif rsi > 50:
                    rsi_boost = 0.15  # Above midline - good
                elif rsi > 40:
                    rsi_boost = 0.10  # Neutral zone
                elif rsi > 30:
                    rsi_boost = 0.05  # Getting oversold
                else:
                    rsi_boost = -0.10  # Oversold - penalty
            
            base_confidence += rsi_boost
            
            # 4. EMA 200 TREND ALIGNMENT (15% weight) - MAJOR TREND DIRECTION
            ema_boost = 0.0
            if ema_200 > 0 and close > 0:
                if signal_type == 'BULL' and close > ema_200:
                    distance_ratio = (close - ema_200) / ema_200
                    if distance_ratio > 0.02:  # More than 2% above EMA200
                        ema_boost = 0.15  # Strong uptrend confirmation
                    else:
                        ema_boost = 0.10  # Basic uptrend confirmation
                elif signal_type == 'BEAR' and close < ema_200:
                    distance_ratio = (ema_200 - close) / ema_200
                    if distance_ratio > 0.02:  # More than 2% below EMA200
                        ema_boost = 0.15  # Strong downtrend confirmation
                    else:
                        ema_boost = 0.10  # Basic downtrend confirmation
                else:
                    ema_boost = -0.10  # Against major trend - significant penalty
            
            base_confidence += ema_boost
            
            # 5. MACD LINE vs SIGNAL ALIGNMENT (10% weight) - ADDITIONAL CONFIRMATION
            alignment_boost = 0.0
            if signal_type == 'BULL' and macd_line > macd_signal:
                alignment_boost = 0.10  # MACD above signal supports bull
            elif signal_type == 'BEAR' and macd_line < macd_signal:
                alignment_boost = 0.10  # MACD below signal supports bear
            else:
                alignment_boost = -0.05  # Misalignment penalty
            
            base_confidence += alignment_boost
            
            # 6. DIVERGENCE PREMIUM BONUS - HIGHEST QUALITY SIGNALS
            divergence_boost = 0.0
            if signal_type == 'BULL' and bullish_divergence:
                divergence_boost = 0.15 + (divergence_strength * 0.10)  # Up to 25% bonus
                self.logger.debug(f"🎯 BULLISH DIVERGENCE DETECTED! Confidence boost: +{divergence_boost:.3f}")
            elif signal_type == 'BEAR' and bearish_divergence:
                divergence_boost = 0.15 + (divergence_strength * 0.10)  # Up to 25% bonus
                self.logger.debug(f"🎯 BEARISH DIVERGENCE DETECTED! Confidence boost: +{divergence_boost:.3f}")
            
            base_confidence += divergence_boost
            
            # Final confidence bounds and quality control
            # Ensure we have a high-quality signal (65%+ minimum after all boosts)
            final_confidence = max(0.20, min(0.95, base_confidence))
            
            # Additional restrictive quality gate
            if final_confidence < 0.65:
                self.logger.debug(f"Signal rejected: confidence {final_confidence:.3f} below 65% threshold")
                return 0.30  # Below trading threshold
            
            self.logger.debug(f"Enhanced MACD confidence: {final_confidence:.3f} for {signal_type} "
                            f"(ADX: {adx:.1f}, RSI: {rsi:.1f}, Histogram: {macd_histogram:.6f})")
            return final_confidence
            
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