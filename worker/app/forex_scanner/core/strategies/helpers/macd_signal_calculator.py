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

    def __init__(self, logger: logging.Logger = None, trend_validator=None, swing_validator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.swing_validator = swing_validator  # NEW: Swing proximity validator
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.65)  # QUALITY: Raised to 65% for high-quality signals only

        # PAIR-SPECIFIC CALIBRATION - Different pairs have different volatility characteristics
        self.pair_configs = {
            # JPY pairs: High volatility, wide swings, prone to false breakouts
            'JPY': {
                'adx_minimum': 35,  # Require very strong trend
                'rsi_bull_max': 45,  # Only boost if RSI < 45 (stricter than 50)
                'rsi_bear_min': 55,  # Only boost if RSI > 55 (stricter than 50)
                'multi_factor_required': 4,  # Need 4 of 5 factors
                'base_adjustment': -0.05,  # Lower base confidence for JPY
            },
            # GBP pairs: MOST erratic, whipsaws, needs strictest filtering
            'GBP': {
                'adx_minimum': 40,  # Require extremely strong trend
                'rsi_bull_max': 40,  # Very strict: only < 40
                'rsi_bear_min': 60,  # Very strict: only > 60
                'multi_factor_required': 5,  # Need ALL 5 factors!
                'base_adjustment': -0.10,  # Significant penalty for GBP
            },
            # Major pairs: EUR, USD crosses (lower volatility, cleaner)
            'MAJOR': {
                'adx_minimum': 30,  # Moderate requirement
                'rsi_bull_max': 50,  # Standard midline
                'rsi_bear_min': 50,  # Standard midline
                'multi_factor_required': 4,  # Need 4 of 5
                'base_adjustment': 0.0,  # No adjustment
            },
            # Commodity pairs: AUD, NZD (medium volatility)
            'COMMODITY': {
                'adx_minimum': 32,  # Slightly higher
                'rsi_bull_max': 48,  # Slightly stricter
                'rsi_bear_min': 52,  # Slightly stricter
                'multi_factor_required': 4,  # Need 4 of 5
                'base_adjustment': -0.03,  # Small penalty
            },
        }

    def _get_pair_config(self, epic: str = None) -> dict:
        """Get pair-specific configuration based on epic"""
        if not epic:
            return self.pair_configs['MAJOR']  # Default to major pairs

        epic_upper = epic.upper()

        # GBP pairs (highest priority - most volatile)
        if 'GBP' in epic_upper:
            return self.pair_configs['GBP']

        # JPY pairs (high volatility)
        if 'JPY' in epic_upper:
            return self.pair_configs['JPY']

        # Commodity pairs
        if any(x in epic_upper for x in ['AUD', 'NZD']):
            return self.pair_configs['COMMODITY']

        # Default: Major pairs (EUR, USD, CHF, CAD)
        return self.pair_configs['MAJOR']
    
    def calculate_simple_confidence(self, latest_row: pd.Series, signal_type: str, epic: str = None) -> float:
        """
        ENHANCED CONFIDENCE CALCULATION: Multi-factor analysis with pair-specific calibration

        Pair-specific filtering:
        - JPY pairs: Stricter ADX (35+), RSI zones (< 45 / > 55)
        - GBP pairs: STRICTEST - ADX 40+, RSI extremes (< 40 / > 60), need ALL 5 factors
        - Major pairs: Standard filtering (ADX 30+, RSI < 50 / > 50)
        - Commodity pairs: Medium strictness

        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            epic: Pair symbol for pair-specific calibration

        Returns:
            Confidence score between 0.0 and 0.95 (higher threshold required)
        """
        try:
            # Get pair-specific configuration
            pair_config = self._get_pair_config(epic)

            base_confidence = 0.15 + pair_config['base_adjustment']  # Pair-specific base adjustment
            
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
            
            # 1. ADX TREND STRENGTH ANALYSIS - STRICT PENALTIES for weak/ranging markets
            adx_boost = 0.0
            if adx >= 40:
                adx_boost = 0.12  # Very strong trend
            elif adx >= 30:
                adx_boost = 0.09  # Strong trend
            elif adx >= 25:
                adx_boost = 0.06  # Moderate trend
            elif adx >= 20:
                adx_boost = -0.05  # Weak trend - PENALTY
            else:
                adx_boost = -0.15  # STRONG PENALTY: Ranging market - MACD unreliable
            
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
            
            # 3. RSI CONFLUENCE ANALYSIS - PAIR-SPECIFIC thresholds
            rsi_boost = 0.0
            rsi_bull_max = pair_config['rsi_bull_max']  # Pair-specific threshold
            rsi_bear_min = pair_config['rsi_bear_min']  # Pair-specific threshold

            if signal_type == 'BULL':
                # BULL signals: Only boost if RSI < pair-specific threshold
                if rsi < 30:
                    rsi_boost = 0.20  # Oversold - excellent for bull signals
                elif rsi < 40:
                    rsi_boost = 0.15  # Below midline - good
                elif rsi < rsi_bull_max:
                    rsi_boost = 0.10  # Approaching threshold - acceptable
                else:
                    # RSI above threshold = no boost
                    rsi_boost = 0.0  # No bonus for buying in neutral/overbought zone
            else:  # BEAR
                # BEAR signals: Only boost if RSI > pair-specific threshold
                if rsi > 70:
                    rsi_boost = 0.20  # Overbought - excellent for bear signals
                elif rsi > 60:
                    rsi_boost = 0.15  # Above midline - good
                elif rsi > rsi_bear_min:
                    rsi_boost = 0.10  # Approaching threshold - acceptable
                else:
                    # RSI below threshold = no boost
                    rsi_boost = 0.0  # No bonus for selling in neutral/oversold zone
            
            base_confidence += rsi_boost
            
            # 4. EMA 200 TREND ALIGNMENT - REMOVED (MACD is momentum, not trend-following)
            # EMA200 filter disabled - MACD should catch early momentum shifts
            # Counter-trend trades are VALID for momentum strategies (reversals!)
            ema_boost = 0.0  # No boost, no penalty - EMA200 not relevant for MACD

            # Commented out - EMA200 logic removed
            # if ema_200 > 0 and close > 0:
            #     if signal_type == 'BULL' and close > ema_200:
            #         ema_boost = 0.15
            #     elif signal_type == 'BEAR' and close < ema_200:
            #         ema_boost = 0.15
            #     else:
            #         ema_boost = -0.10  # Counter-trend penalty NOT APPROPRIATE for momentum

            base_confidence += ema_boost  # Always 0.0
            
            # 5. MACD LINE vs SIGNAL ALIGNMENT - REQUIRED (not optional)
            alignment_boost = 0.0
            if signal_type == 'BULL' and macd_line > macd_signal:
                alignment_boost = 0.10  # MACD above signal supports bull
            elif signal_type == 'BEAR' and macd_line < macd_signal:
                alignment_boost = 0.10  # MACD below signal supports bear
            else:
                alignment_boost = -0.15  # STRONG PENALTY: Misalignment indicates weak signal
            
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

            # 7. SWING PROXIMITY VALIDATION - PREVENT POOR ENTRY TIMING (NEW)
            swing_proximity_penalty = 0.0
            if self.swing_validator:
                # Note: We need df and epic for full validation, but we only have latest_row here
                # This is a simplified check - full integration happens at strategy level
                # For now, we just log that swing validation is available
                self.logger.debug("Swing proximity validator available (full check at strategy level)")

            # 8. DIVERGENCE CAP - Limit high confidence to divergence signals only
            # Without divergence, cap at 70% to reserve 70%+ for premium signals
            if divergence_boost == 0.0:  # No divergence detected
                base_confidence = min(0.70, base_confidence)  # Cap non-divergence signals at 70%

            # 9. MULTI-FACTOR GATE - PAIR-SPECIFIC requirements
            # Factors: ADX ≥ threshold, RSI in correct zone, Alignment correct, Divergence present, Swing OK
            positive_factors = 0
            adx_minimum = pair_config['adx_minimum']  # Pair-specific ADX requirement
            required_factors = pair_config['multi_factor_required']  # Pair-specific: 4 or 5

            # Factor 1: Strong ADX (pair-specific threshold)
            if adx >= adx_minimum:
                positive_factors += 1

            # Factor 2: RSI in correct zone (pair-specific thresholds)
            if signal_type == 'BULL' and rsi < rsi_bull_max:
                positive_factors += 1
            elif signal_type == 'BEAR' and rsi > rsi_bear_min:
                positive_factors += 1

            # Factor 3: MACD/Signal alignment correct
            if (signal_type == 'BULL' and macd_line > macd_signal) or \
               (signal_type == 'BEAR' and macd_line < macd_signal):
                positive_factors += 1

            # Factor 4: Divergence present
            if divergence_boost > 0.0:
                positive_factors += 1

            # Factor 5: Swing proximity OK (assume OK if validator not available)
            if swing_proximity_penalty == 0.0:
                positive_factors += 1

            # Apply multi-factor gate penalty if less than required factors (PAIR-SPECIFIC)
            multi_factor_penalty = 0.0
            if positive_factors < required_factors:
                multi_factor_penalty = -0.15  # STRONG PENALTY for weak multi-factor confirmation
                self.logger.debug(f"Multi-factor gate: Only {positive_factors}/5 factors positive (need {required_factors}) - applying -0.15 penalty")

            base_confidence -= multi_factor_penalty

            # Final confidence calculation (capped at 95%)
            final_confidence = min(0.95, base_confidence - swing_proximity_penalty)

            # Log confidence breakdown for debugging (only if debug enabled)
            if self.logger.isEnabledFor(logging.DEBUG):
                pair_type = 'GBP' if epic and 'GBP' in epic.upper() else \
                           'JPY' if epic and 'JPY' in epic.upper() else \
                           'COMMODITY' if epic and any(x in epic.upper() for x in ['AUD', 'NZD']) else 'MAJOR'
                self.logger.debug(f"CONFIDENCE BREAKDOWN for {signal_type} ({epic or 'unknown'} - {pair_type}):")
                self.logger.debug(f"   Base: {0.15 + pair_config['base_adjustment']:.3f}, ADX: {adx_boost:.3f} (min: {adx_minimum})")
                self.logger.debug(f"   Histogram: {histogram_boost:.3f}, RSI: {rsi_boost:.3f} (thresholds: <{rsi_bull_max} / >{rsi_bear_min})")
                self.logger.debug(f"   EMA: {ema_boost:.3f}, Alignment: {alignment_boost:.3f}")
                self.logger.debug(f"   Divergence: {divergence_boost:.3f}")
                self.logger.debug(f"   Multi-factor gate: {positive_factors}/5 factors (need {required_factors}), penalty: {multi_factor_penalty:.3f}")
                self.logger.debug(f"   Swing penalty: {swing_proximity_penalty:.3f}")
                self.logger.debug(f"   FINAL: {final_confidence:.3f} (min threshold: {self.min_confidence:.3f})")

            self.logger.debug(f"MACD confidence: {final_confidence:.3f} for {signal_type}")
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
        # BALANCED MODE: Normal confidence validation
        passes = confidence >= self.min_confidence
        if not passes:
            self.logger.debug(f"Signal rejected: confidence {confidence:.3f} < threshold {self.min_confidence:.3f}")
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

    def validate_swing_proximity(
        self,
        df: pd.DataFrame,
        current_price: float,
        signal_type: str,
        epic: str = None
    ) -> Dict[str, any]:
        """
        Validate swing proximity for signal entry timing (NEW)

        Args:
            df: DataFrame with price data and swing analysis
            current_price: Current market price
            signal_type: 'BULL' or 'BEAR'
            epic: Epic/symbol for pip calculation

        Returns:
            Dictionary with validation result and confidence adjustment
        """
        try:
            if not self.swing_validator:
                return {
                    'valid': True,
                    'confidence_penalty': 0.0,
                    'reason': 'Swing validator not configured'
                }

            # Call swing proximity validator
            result = self.swing_validator.validate_entry_proximity(
                df=df,
                current_price=current_price,
                direction=signal_type,
                epic=epic
            )

            if not result['valid']:
                self.logger.warning(
                    f"⚠️ Swing proximity violation: {result.get('rejection_reason', 'Unknown')}"
                )

            return {
                'valid': result['valid'],
                'confidence_penalty': result.get('confidence_penalty', 0.0),
                'reason': result.get('rejection_reason'),
                'swing_distance': result.get('distance_to_swing'),
                'swing_price': result.get('nearest_swing_price'),
                'swing_type': result.get('swing_type')
            }

        except Exception as e:
            self.logger.error(f"Swing proximity validation error: {e}")
            return {
                'valid': True,
                'confidence_penalty': 0.0,
                'reason': f'Validation error: {str(e)}'
            }