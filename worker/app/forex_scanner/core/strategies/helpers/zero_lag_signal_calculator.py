# core/strategies/helpers/zero_lag_signal_calculator.py
"""
Zero Lag Signal Calculator Module
Handles confidence calculation and signal strength assessment for Zero Lag strategy
"""

import pandas as pd
import logging
from typing import Optional, Dict
try:
    import config
except ImportError:
    from forex_scanner import config


class ZeroLagSignalCalculator:
    """Calculates confidence scores and signal strength for Zero Lag signals"""
    
    def __init__(self, logger: logging.Logger = None, trend_validator=None, squeeze_analyzer=None, enhanced_validation: bool = True):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.squeeze_analyzer = squeeze_analyzer
        self.enhanced_validation = enhanced_validation
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)
    
    def calculate_signal_confidence(self, latest_row: pd.Series, signal_type: str, 
                                    signal_data: Dict = None) -> float:
        """
        Calculate comprehensive confidence score for Zero Lag signals

        Factors considered:
        1. Zero Lag trend alignment (25% weight)
        2. Price position relative to ZLEMA (15% weight)
        3. EMA 200 trend filter (15% weight)
        4. Squeeze Momentum confirmation (20% weight)
        5. RSI validation (15% weight)
        6. Volatility conditions (5% weight)
        7. Signal strength (5% weight)
        
        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            signal_data: Additional signal context
            
        Returns:
            Confidence score between 0.0 and 0.95
        """
        try:
            base_confidence = 0.5  # Start with 50%
            
            # Get core values
            close = latest_row.get('close', 0)
            zlema = latest_row.get('zlema', 0)
            ema_200 = latest_row.get('ema_200', 0)
            trend_state = latest_row.get('trend', 0)
            volatility = latest_row.get('volatility', 0.001)
            
            if close == 0 or zlema == 0:
                return 0.3  # Low confidence if key data missing
            
            # 1. ZERO LAG TREND ALIGNMENT (25% weight)
            trend_factor = self._calculate_trend_alignment_factor(
                close, zlema, trend_state, signal_type, volatility
            )

            # 2. PRICE POSITION RELATIVE TO ZLEMA (15% weight)
            position_factor = self._calculate_position_factor(
                close, zlema, signal_type, volatility
            )

            # 3. EMA 200 TREND FILTER (15% weight)
            ema200_factor = self._calculate_ema200_factor(
                close, ema_200, signal_type
            )

            # 4. SQUEEZE MOMENTUM CONFIRMATION (20% weight)
            squeeze_factor = 0.0
            if self.squeeze_analyzer:
                squeeze_factor = self._calculate_squeeze_factor(
                    latest_row, signal_type
                )

            # 5. RSI VALIDATION (15% weight)
            rsi_factor = self._calculate_rsi_factor(
                latest_row, signal_type
            )

            # 6. VOLATILITY CONDITIONS (5% weight)
            volatility_factor = self._calculate_volatility_factor(
                close, volatility
            )

            # 7. SIGNAL STRENGTH (5% weight)
            strength_factor = self._calculate_strength_factor(
                latest_row, signal_data
            )
            
            # Calculate total confidence
            total_confidence = (
                base_confidence +
                trend_factor +
                position_factor +
                ema200_factor +
                squeeze_factor +
                rsi_factor +
                volatility_factor +
                strength_factor
            )
            
            # Cap confidence at 95%
            final_confidence = min(0.95, max(0.1, total_confidence))
            
            # Log confidence breakdown
            self._log_confidence_breakdown({
                'base': base_confidence,
                'trend_alignment': trend_factor,
                'position': position_factor,
                'ema_200': ema200_factor,
                'squeeze_momentum': squeeze_factor,
                'rsi': rsi_factor,
                'volatility': volatility_factor,
                'strength': strength_factor,
                'final': final_confidence
            })
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating Zero Lag confidence: {e}")
            return 0.4  # Safe fallback
    
    def _calculate_trend_alignment_factor(self, close: float, zlema: float, 
                                          trend_state: int, signal_type: str, 
                                          volatility: float) -> float:
        """Calculate confidence factor based on trend alignment"""
        try:
            factor = 0.0
            
            # Price-ZLEMA relationship
            price_above_zlema = close > zlema
            
            if signal_type == 'BULL':
                if trend_state == 1 and price_above_zlema:
                    factor += 0.20  # Perfect alignment (reduced from 25% to 20%)
                    self.logger.debug("Trend alignment: Perfect bull alignment (+20%)")
                elif price_above_zlema:
                    factor += 0.12  # Good price position (reduced from 15% to 12%)
                    self.logger.debug("Trend alignment: Price above ZLEMA (+12%)")
                elif trend_state == 1:
                    factor += 0.06  # Trend is up but price below ZLEMA (reduced from 8% to 6%)
                    self.logger.debug("Trend alignment: Uptrend but price below ZLEMA (+6%)")
                else:
                    factor -= 0.08  # Poor alignment (reduced from -10% to -8%)
                    self.logger.debug("Trend alignment: Poor bull alignment (-8%)")
            
            else:  # BEAR
                if trend_state == -1 and not price_above_zlema:
                    factor += 0.20  # Perfect alignment (reduced from 25% to 20%)
                    self.logger.debug("Trend alignment: Perfect bear alignment (+20%)")
                elif not price_above_zlema:
                    factor += 0.12  # Good price position (reduced from 15% to 12%)
                    self.logger.debug("Trend alignment: Price below ZLEMA (+12%)")
                elif trend_state == -1:
                    factor += 0.06  # Trend is down but price above ZLEMA (reduced from 8% to 6%)
                    self.logger.debug("Trend alignment: Downtrend but price above ZLEMA (+6%)")
                else:
                    factor -= 0.08  # Poor alignment (reduced from -10% to -8%)
                    self.logger.debug("Trend alignment: Poor bear alignment (-8%)")
            
            return factor
            
        except Exception as e:
            self.logger.debug(f"Trend alignment factor calculation failed: {e}")
            return 0.0
    
    def _calculate_position_factor(self, close: float, zlema: float, 
                                   signal_type: str, volatility: float) -> float:
        """Calculate confidence factor based on price position"""
        try:
            if volatility == 0:
                return 0.0
            
            # Calculate distance from ZLEMA in volatility units
            distance = abs(close - zlema) / volatility
            
            if signal_type == 'BULL':
                if close > zlema:
                    # Price above ZLEMA is good for bull signals (reduced from 15% to 12%)
                    position_factor = min(0.12, distance * 0.04)
                    self.logger.debug(f"Position: Bull price above ZLEMA (+{position_factor:.1%})")
                else:
                    # Price below ZLEMA reduces confidence (reduced from -8% to -6%)
                    position_factor = -min(0.06, distance * 0.02)
                    self.logger.debug(f"Position: Bull price below ZLEMA ({position_factor:.1%})")
            else:  # BEAR
                if close < zlema:
                    # Price below ZLEMA is good for bear signals (reduced from 15% to 12%)
                    position_factor = min(0.12, distance * 0.04)
                    self.logger.debug(f"Position: Bear price below ZLEMA (+{position_factor:.1%})")
                else:
                    # Price above ZLEMA reduces confidence (reduced from -8% to -6%)
                    position_factor = -min(0.06, distance * 0.02)
                    self.logger.debug(f"Position: Bear price above ZLEMA ({position_factor:.1%})")
            
            return position_factor
            
        except Exception as e:
            self.logger.debug(f"Position factor calculation failed: {e}")
            return 0.0
    
    def _calculate_ema200_factor(self, close: float, ema_200: float, 
                                 signal_type: str) -> float:
        """Calculate confidence factor based on EMA 200 trend filter"""
        try:
            if ema_200 == 0 or close == 0:
                return 0.0
            
            # Determine pip multiplier
            pip_multiplier = 100 if close > 50 else 10000
            
            if signal_type == 'BULL':
                if close > ema_200:
                    distance_above = (close - ema_200) * pip_multiplier
                    ema200_factor = min(0.12, distance_above / 100 * 0.05)  # Scale by distance
                    self.logger.debug(f"EMA 200: Bull {distance_above:.1f} pips above (+{ema200_factor:.1%})")
                else:
                    distance_below = (ema_200 - close) * pip_multiplier
                    ema200_factor = -min(0.15, distance_below / 100 * 0.08)  # Penalty
                    self.logger.debug(f"EMA 200: Bull {distance_below:.1f} pips below ({ema200_factor:.1%})")
            
            else:  # BEAR
                if close < ema_200:
                    distance_below = (ema_200 - close) * pip_multiplier
                    ema200_factor = min(0.12, distance_below / 100 * 0.05)  # Scale by distance
                    self.logger.debug(f"EMA 200: Bear {distance_below:.1f} pips below (+{ema200_factor:.1%})")
                else:
                    distance_above = (close - ema_200) * pip_multiplier
                    ema200_factor = -min(0.15, distance_above / 100 * 0.08)  # Penalty
                    self.logger.debug(f"EMA 200: Bear {distance_above:.1f} pips above ({ema200_factor:.1%})")
            
            return ema200_factor
            
        except Exception as e:
            self.logger.debug(f"EMA 200 factor calculation failed: {e}")
            return 0.0
    
    def _calculate_squeeze_factor(self, latest_row: pd.Series, signal_type: str) -> float:
        """Calculate confidence factor based on Squeeze Momentum"""
        try:
            if not self.squeeze_analyzer:
                return 0.0
            
            # Get squeeze momentum boost
            squeeze_boost = self.squeeze_analyzer.get_squeeze_confidence_boost(
                latest_row, signal_type
            )
            
            self.logger.debug(f"Squeeze factor: {squeeze_boost:+.1%}")
            return squeeze_boost
            
        except Exception as e:
            self.logger.debug(f"Squeeze factor calculation failed: {e}")
            return 0.0
    
    def _calculate_volatility_factor(self, close: float, volatility: float) -> float:
        """Calculate confidence factor based on volatility conditions"""
        try:
            if close == 0:
                return 0.0
            
            volatility_ratio = volatility / close
            
            if 0.003 <= volatility_ratio <= 0.025:  # Good volatility range (reduced from 8% to 4%)
                volatility_factor = 0.04
                self.logger.debug(f"Volatility: Optimal ({volatility_ratio:.4f}) (+4%)")
            elif volatility_ratio < 0.001:  # Too low volatility (reduced from -5% to -3%)
                volatility_factor = -0.03
                self.logger.debug(f"Volatility: Too low ({volatility_ratio:.4f}) (-3%)")
            elif volatility_ratio > 0.04:  # Too high volatility (reduced from -3% to -2%)
                volatility_factor = -0.02
                self.logger.debug(f"Volatility: Too high ({volatility_ratio:.4f}) (-2%)")
            else:
                volatility_factor = 0.02  # Acceptable (reduced from 3% to 2%)
                self.logger.debug(f"Volatility: Acceptable ({volatility_ratio:.4f}) (+2%)")
            
            return volatility_factor
            
        except Exception as e:
            self.logger.debug(f"Volatility factor calculation failed: {e}")
            return 0.0
    
    def _calculate_strength_factor(self, latest_row: pd.Series, 
                                   signal_data: Dict = None) -> float:
        """Calculate confidence factor based on signal strength"""
        try:
            # Use ZLEMA slope as strength indicator
            zlema_slope = latest_row.get('zlema_slope', 0)
            volatility = latest_row.get('volatility', 0.001)
            
            if volatility == 0:
                return 0.0
            
            # Normalize slope by volatility
            normalized_slope = abs(zlema_slope) / volatility
            
            # Strong slope gets boost
            strength_factor = min(0.04, normalized_slope * 0.02)
            
            self.logger.debug(f"Signal strength: {normalized_slope:.3f} (+{strength_factor:.1%})")
            return strength_factor
            
        except Exception as e:
            self.logger.debug(f"Strength factor calculation failed: {e}")
            return 0.0
    
    def _log_confidence_breakdown(self, factors: Dict):
        """Log detailed confidence calculation breakdown"""
        try:
            self.logger.debug("🔥 Zero Lag Confidence Breakdown:")
            self.logger.debug(f"   Base Confidence: {factors['base']:.1%}")
            self.logger.debug(f"   Trend Alignment: {factors['trend_alignment']:+.1%}")
            self.logger.debug(f"   Price Position: {factors['position']:+.1%}")
            self.logger.debug(f"   EMA 200 Filter: {factors['ema_200']:+.1%}")
            self.logger.debug(f"   Squeeze Momentum: {factors['squeeze_momentum']:+.1%}")
            self.logger.debug(f"   RSI Validation: {factors['rsi']:+.1%}")
            self.logger.debug(f"   Volatility: {factors['volatility']:+.1%}")
            self.logger.debug(f"   Signal Strength: {factors['strength']:+.1%}")
            self.logger.debug(f"   === FINAL: {factors['final']:.1%} ===")
            
        except Exception as e:
            self.logger.debug(f"Confidence breakdown logging failed: {e}")
    
    def calculate_crossover_strength(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate the strength of the Zero Lag crossover
        
        Args:
            latest_row: DataFrame row with price and ZLEMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Strength value between 0.0 and 1.0
        """
        try:
            close = latest_row.get('close', 0)
            zlema = latest_row.get('zlema', 0)
            volatility = latest_row.get('volatility', 0.001)
            
            if close == 0 or zlema == 0:
                return 0.0
            
            # Calculate distance from ZLEMA
            distance = abs(close - zlema) / close
            
            # Convert to strength relative to volatility
            volatility_ratio = volatility / close
            strength = min(1.0, distance / (volatility_ratio * 2))
            
            # Validate direction
            if signal_type == 'BULL':
                if close <= zlema:
                    return 0.0
            else:  # BEAR
                if close >= zlema:
                    return 0.0
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating crossover strength: {e}")
            return 0.0
    
    def calculate_trend_strength(self, latest_row: pd.Series) -> float:
        """
        Calculate overall trend strength based on Zero Lag indicators
        
        Returns:
            Trend strength value between -1.0 (strong bear) and 1.0 (strong bull)
        """
        try:
            close = latest_row.get('close', 0)
            zlema = latest_row.get('zlema', 0)
            zlema_slope = latest_row.get('zlema_slope', 0)
            trend_state = latest_row.get('trend', 0)
            
            if close == 0 or zlema == 0:
                return 0.0
            
            # Base trend strength from trend state
            base_strength = trend_state * 0.5  # -0.5 to 0.5
            
            # Add slope contribution
            slope_contribution = max(-0.3, min(0.3, zlema_slope * 1000))
            
            # Add price position contribution
            price_contribution = (close - zlema) / zlema * 0.2
            price_contribution = max(-0.2, min(0.2, price_contribution))
            
            # Combine factors
            total_strength = base_strength + slope_contribution + price_contribution
            
            # Cap at -1.0 to 1.0
            return max(-1.0, min(1.0, total_strength))
            
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

    def validate_rsi_conditions(self, latest_row: pd.Series, signal_type: str) -> bool:
        """
        Validate RSI conditions for signal filtering

        RSI Rules:
        - BULL signals: RSI must be under 70 (not overbought)
        - BEAR signals: RSI must be over 30 (not oversold)

        Args:
            latest_row: DataFrame row with RSI data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if RSI conditions are met, False otherwise
        """
        try:
            rsi = latest_row.get('rsi', 50.0)

            if signal_type == 'BULL':
                if rsi >= 70:
                    self.logger.debug(f"❌ BULL signal blocked: RSI {rsi:.1f} >= 70 (overbought)")
                    return False
                else:
                    self.logger.debug(f"✅ RSI validation passed for BULL: RSI {rsi:.1f} < 70")
                    return True

            elif signal_type == 'BEAR':
                if rsi <= 30:
                    self.logger.debug(f"❌ BEAR signal blocked: RSI {rsi:.1f} <= 30 (oversold)")
                    return False
                else:
                    self.logger.debug(f"✅ RSI validation passed for BEAR: RSI {rsi:.1f} > 30")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error validating RSI conditions: {e}")
            return True  # Allow signal on error to avoid blocking strategy

    def _calculate_rsi_factor(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate confidence factor based on RSI position

        RSI Confidence Boost Logic:
        - BULL signals: Higher confidence when RSI is 40-65 (good buying zone)
        - BEAR signals: Higher confidence when RSI is 35-60 (good selling zone)

        Args:
            latest_row: DataFrame row with RSI data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            RSI confidence factor between -0.1 and +0.1
        """
        try:
            rsi = latest_row.get('rsi', 50.0)

            if signal_type == 'BULL':
                if 40 <= rsi <= 65:
                    # Optimal RSI range for bull signals
                    factor = 0.08
                    self.logger.debug(f"RSI factor: Optimal bull range RSI={rsi:.1f} (+8%)")
                elif 30 <= rsi < 40:
                    # Good oversold recovery
                    factor = 0.05
                    self.logger.debug(f"RSI factor: Oversold recovery RSI={rsi:.1f} (+5%)")
                elif 65 < rsi < 70:
                    # Getting expensive but still acceptable
                    factor = 0.02
                    self.logger.debug(f"RSI factor: Getting expensive RSI={rsi:.1f} (+2%)")
                else:
                    # Neutral or poor RSI
                    factor = 0.0
                    self.logger.debug(f"RSI factor: Neutral/poor RSI={rsi:.1f} (0%)")

            else:  # BEAR
                if 35 <= rsi <= 60:
                    # Optimal RSI range for bear signals
                    factor = 0.08
                    self.logger.debug(f"RSI factor: Optimal bear range RSI={rsi:.1f} (+8%)")
                elif 60 < rsi <= 70:
                    # Good overbought reversal
                    factor = 0.05
                    self.logger.debug(f"RSI factor: Overbought reversal RSI={rsi:.1f} (+5%)")
                elif 30 < rsi < 35:
                    # Getting cheap but still acceptable
                    factor = 0.02
                    self.logger.debug(f"RSI factor: Getting cheap RSI={rsi:.1f} (+2%)")
                else:
                    # Neutral or poor RSI
                    factor = 0.0
                    self.logger.debug(f"RSI factor: Neutral/poor RSI={rsi:.1f} (0%)")

            return factor

        except Exception as e:
            self.logger.debug(f"RSI factor calculation failed: {e}")
            return 0.0