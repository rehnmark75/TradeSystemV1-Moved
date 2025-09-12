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
    
    def __init__(self, logger: logging.Logger = None, trend_validator=None, squeeze_analyzer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.squeeze_analyzer = squeeze_analyzer
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.45)
    
    def calculate_signal_confidence(self, latest_row: pd.Series, signal_type: str, 
                                    signal_data: Dict = None) -> float:
        """
        Calculate comprehensive confidence score for Zero Lag signals
        
        Factors considered:
        1. Zero Lag trend alignment (30% weight)
        2. Price position relative to ZLEMA (20% weight)  
        3. EMA 200 trend filter (15% weight)
        4. Squeeze Momentum confirmation (20% weight)
        5. Volatility conditions (10% weight)
        6. Signal strength (5% weight)
        
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
            
            # 1. ZERO LAG TREND ALIGNMENT (30% weight)
            trend_factor = self._calculate_trend_alignment_factor(
                close, zlema, trend_state, signal_type, volatility
            )
            
            # 2. PRICE POSITION RELATIVE TO ZLEMA (20% weight)  
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
            
            # 5. VOLATILITY CONDITIONS (10% weight)
            volatility_factor = self._calculate_volatility_factor(
                close, volatility
            )
            
            # 6. SIGNAL STRENGTH (5% weight)
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
                    factor += 0.25  # Perfect alignment
                    self.logger.debug("Trend alignment: Perfect bull alignment (+25%)")
                elif price_above_zlema:
                    factor += 0.15  # Good price position
                    self.logger.debug("Trend alignment: Price above ZLEMA (+15%)")
                elif trend_state == 1:
                    factor += 0.08  # Trend is up but price below ZLEMA
                    self.logger.debug("Trend alignment: Uptrend but price below ZLEMA (+8%)")
                else:
                    factor -= 0.1  # Poor alignment
                    self.logger.debug("Trend alignment: Poor bull alignment (-10%)")
            
            else:  # BEAR
                if trend_state == -1 and not price_above_zlema:
                    factor += 0.25  # Perfect alignment
                    self.logger.debug("Trend alignment: Perfect bear alignment (+25%)")
                elif not price_above_zlema:
                    factor += 0.15  # Good price position
                    self.logger.debug("Trend alignment: Price below ZLEMA (+15%)")
                elif trend_state == -1:
                    factor += 0.08  # Trend is down but price above ZLEMA
                    self.logger.debug("Trend alignment: Downtrend but price above ZLEMA (+8%)")
                else:
                    factor -= 0.1  # Poor alignment
                    self.logger.debug("Trend alignment: Poor bear alignment (-10%)")
            
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
                    # Price above ZLEMA is good for bull signals
                    position_factor = min(0.15, distance * 0.05)
                    self.logger.debug(f"Position: Bull price above ZLEMA (+{position_factor:.1%})")
                else:
                    # Price below ZLEMA reduces confidence
                    position_factor = -min(0.08, distance * 0.03)
                    self.logger.debug(f"Position: Bull price below ZLEMA ({position_factor:.1%})")
            else:  # BEAR
                if close < zlema:
                    # Price below ZLEMA is good for bear signals
                    position_factor = min(0.15, distance * 0.05)
                    self.logger.debug(f"Position: Bear price below ZLEMA (+{position_factor:.1%})")
                else:
                    # Price above ZLEMA reduces confidence
                    position_factor = -min(0.08, distance * 0.03)
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
            
            if 0.003 <= volatility_ratio <= 0.025:  # Good volatility range
                volatility_factor = 0.08
                self.logger.debug(f"Volatility: Optimal ({volatility_ratio:.4f}) (+8%)")
            elif volatility_ratio < 0.001:  # Too low volatility
                volatility_factor = -0.05
                self.logger.debug(f"Volatility: Too low ({volatility_ratio:.4f}) (-5%)")
            elif volatility_ratio > 0.04:  # Too high volatility
                volatility_factor = -0.03
                self.logger.debug(f"Volatility: Too high ({volatility_ratio:.4f}) (-3%)")
            else:
                volatility_factor = 0.03  # Acceptable
                self.logger.debug(f"Volatility: Acceptable ({volatility_ratio:.4f}) (+3%)")
            
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