# core/strategies/helpers/ichimoku_signal_calculator.py
"""
Ichimoku Signal Calculator Module
Handles confidence calculation and signal strength assessment for Ichimoku Cloud strategy

Confidence calculation based on:
- TK line alignment and cross strength (30%)
- Cloud position and thickness (25%)
- Chikou span confirmation (20%)
- Price position relative to cloud (15%)
- Market context and momentum (10%)
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class IchimokuSignalCalculator:
    """Calculates confidence scores and signal strength for Ichimoku Cloud signals"""

    def __init__(self, logger: logging.Logger = None, trend_validator=None):
        self.logger = logger or logging.getLogger(__name__)
        self.trend_validator = trend_validator
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.55)  # Ichimoku needs higher confidence

    def calculate_ichimoku_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        ICHIMOKU CONFIDENCE CALCULATION: Multi-factor cloud-based analysis

        Weighting system for Ichimoku signals:
        - TK line alignment and cross strength (30%): Core directional signal
        - Cloud position and thickness (25%): Support/resistance strength
        - Chikou span confirmation (20%): Momentum confirmation
        - Price position relative to cloud (15%): Breakout quality
        - Market context and volatility (10%): Additional confirmation

        Args:
            latest_row: DataFrame row with Ichimoku indicator data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Confidence score between 0.0 and 0.95
        """
        try:
            base_confidence = 0.50  # Start with 50%

            # Get Ichimoku component values
            tenkan_sen = latest_row.get('tenkan_sen', 0)
            kijun_sen = latest_row.get('kijun_sen', 0)
            cloud_top = latest_row.get('cloud_top', 0)
            cloud_bottom = latest_row.get('cloud_bottom', 0)
            chikou_span = latest_row.get('chikou_span', 0)
            close = latest_row.get('close', 0)

            # Signal trigger indicators
            tk_bull_cross = latest_row.get('tk_bull_cross', False)
            tk_bear_cross = latest_row.get('tk_bear_cross', False)
            cloud_bull_breakout = latest_row.get('cloud_bull_breakout', False)
            cloud_bear_breakout = latest_row.get('cloud_bear_breakout', False)

            if tenkan_sen == 0 or kijun_sen == 0 or cloud_top == 0:
                return 0.3  # Low confidence if Ichimoku components missing

            # 1. TK LINE ALIGNMENT AND CROSS STRENGTH (30% weight)
            tk_confidence = self._calculate_tk_confidence(
                latest_row, signal_type, tenkan_sen, kijun_sen, close
            )
            base_confidence += tk_confidence * 0.30

            # 2. CLOUD POSITION AND THICKNESS (25% weight)
            cloud_confidence = self._calculate_cloud_confidence(
                latest_row, signal_type, cloud_top, cloud_bottom, close
            )
            base_confidence += cloud_confidence * 0.25

            # 3. CHIKOU SPAN CONFIRMATION (20% weight)
            chikou_confidence = self._calculate_chikou_confidence(
                latest_row, signal_type, chikou_span
            )
            base_confidence += chikou_confidence * 0.20

            # 4. PRICE POSITION RELATIVE TO CLOUD (15% weight)
            price_confidence = self._calculate_price_position_confidence(
                latest_row, signal_type, cloud_top, cloud_bottom, close
            )
            base_confidence += price_confidence * 0.15

            # 5. MARKET CONTEXT AND MOMENTUM (10% weight)
            context_confidence = self._calculate_market_context_confidence(
                latest_row, signal_type
            )
            base_confidence += context_confidence * 0.10

            # BONUS: Premium signal quality boosts
            base_confidence = self._apply_premium_bonuses(
                latest_row, signal_type, base_confidence,
                tk_bull_cross, tk_bear_cross, cloud_bull_breakout, cloud_bear_breakout
            )

            # Cap at 95% confidence (never 100% certain in markets)
            final_confidence = min(0.95, max(0.0, base_confidence))

            self.logger.debug(f"Ichimoku {signal_type} confidence: {final_confidence:.1%} "
                            f"(TK:{tk_confidence:.2f}, Cloud:{cloud_confidence:.2f}, "
                            f"Chikou:{chikou_confidence:.2f}, Price:{price_confidence:.2f})")

            return final_confidence

        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku confidence: {e}")
            return 0.35  # Conservative fallback

    def _calculate_tk_confidence(self, latest_row: pd.Series, signal_type: str,
                                tenkan: float, kijun: float, close: float) -> float:
        """Calculate confidence from TK line alignment and cross strength"""
        try:
            tk_confidence = 0.0

            # TK line separation (stronger when lines are further apart)
            tk_separation = abs(tenkan - kijun) / (close if close > 0 else 1)
            separation_score = min(1.0, tk_separation / 0.001)  # Normalize

            if signal_type == 'BULL':
                # Bull: Tenkan should be above Kijun
                if tenkan > kijun:
                    tk_confidence += 0.6  # Strong alignment
                    tk_confidence += separation_score * 0.4  # Cross strength
                elif tenkan == kijun:
                    tk_confidence += 0.3  # Neutral
            else:  # BEAR
                # Bear: Tenkan should be below Kijun
                if tenkan < kijun:
                    tk_confidence += 0.6  # Strong alignment
                    tk_confidence += separation_score * 0.4  # Cross strength
                elif tenkan == kijun:
                    tk_confidence += 0.3  # Neutral

            # Check for actual crossover (higher quality signal)
            if signal_type == 'BULL' and latest_row.get('tk_bull_cross', False):
                tk_confidence += 0.2  # Crossover bonus
            elif signal_type == 'BEAR' and latest_row.get('tk_bear_cross', False):
                tk_confidence += 0.2  # Crossover bonus

            return min(1.0, tk_confidence)

        except Exception as e:
            self.logger.error(f"Error calculating TK confidence: {e}")
            return 0.0

    def _calculate_cloud_confidence(self, latest_row: pd.Series, signal_type: str,
                                   cloud_top: float, cloud_bottom: float, close: float) -> float:
        """Calculate confidence from cloud position and thickness"""
        try:
            cloud_confidence = 0.0

            # Cloud thickness (thicker cloud = stronger support/resistance)
            cloud_thickness = cloud_top - cloud_bottom
            thickness_ratio = cloud_thickness / (close if close > 0 else 1)
            thickness_score = min(1.0, thickness_ratio / 0.002)  # Normalize

            # Cloud color preference
            senkou_a = latest_row.get('senkou_span_a', 0)
            senkou_b = latest_row.get('senkou_span_b', 0)
            cloud_is_green = senkou_a > senkou_b

            if signal_type == 'BULL':
                # Bull signals prefer price above cloud
                if close > cloud_top:
                    cloud_confidence += 0.7  # Above cloud
                elif close > cloud_bottom:
                    cloud_confidence += 0.3  # In cloud
                else:
                    cloud_confidence += 0.1  # Below cloud

                # Green cloud is bullish
                if cloud_is_green:
                    cloud_confidence += 0.2
            else:  # BEAR
                # Bear signals prefer price below cloud
                if close < cloud_bottom:
                    cloud_confidence += 0.7  # Below cloud
                elif close < cloud_top:
                    cloud_confidence += 0.3  # In cloud
                else:
                    cloud_confidence += 0.1  # Above cloud

                # Red cloud is bearish
                if not cloud_is_green:
                    cloud_confidence += 0.2

            # Thickness bonus (thicker clouds are more reliable)
            cloud_confidence += thickness_score * 0.1

            return min(1.0, cloud_confidence)

        except Exception as e:
            self.logger.error(f"Error calculating cloud confidence: {e}")
            return 0.0

    def _calculate_chikou_confidence(self, latest_row: pd.Series, signal_type: str,
                                    chikou_span: float) -> float:
        """Calculate confidence from Chikou span position"""
        try:
            chikou_confidence = 0.0

            # Get historical price context for Chikou validation
            # Note: This is a simplified version - full validation done in trend_validator
            high = latest_row.get('high', 0)
            low = latest_row.get('low', 0)
            close = latest_row.get('close', 0)

            if chikou_span == 0:
                return 0.0

            if signal_type == 'BULL':
                # Bull: Chikou should be above historical price action
                if chikou_span > high:
                    chikou_confidence = 1.0  # Clear above
                elif chikou_span > close:
                    chikou_confidence = 0.7  # Above close
                elif chikou_span > low:
                    chikou_confidence = 0.4  # Above low
                else:
                    chikou_confidence = 0.1  # Below all
            else:  # BEAR
                # Bear: Chikou should be below historical price action
                if chikou_span < low:
                    chikou_confidence = 1.0  # Clear below
                elif chikou_span < close:
                    chikou_confidence = 0.7  # Below close
                elif chikou_span < high:
                    chikou_confidence = 0.4  # Below high
                else:
                    chikou_confidence = 0.1  # Above all

            return chikou_confidence

        except Exception as e:
            self.logger.error(f"Error calculating Chikou confidence: {e}")
            return 0.0

    def _calculate_price_position_confidence(self, latest_row: pd.Series, signal_type: str,
                                           cloud_top: float, cloud_bottom: float, close: float) -> float:
        """Calculate confidence from price position relative to cloud"""
        try:
            position_confidence = 0.0

            if signal_type == 'BULL':
                if close > cloud_top:
                    # Distance above cloud (stronger when further above)
                    distance_ratio = (close - cloud_top) / (close if close > 0 else 1)
                    position_confidence = min(1.0, 0.7 + distance_ratio * 10)  # Scale distance
                elif close >= cloud_bottom:
                    position_confidence = 0.4  # In cloud
                else:
                    position_confidence = 0.1  # Below cloud
            else:  # BEAR
                if close < cloud_bottom:
                    # Distance below cloud (stronger when further below)
                    distance_ratio = (cloud_bottom - close) / (close if close > 0 else 1)
                    position_confidence = min(1.0, 0.7 + distance_ratio * 10)  # Scale distance
                elif close <= cloud_top:
                    position_confidence = 0.4  # In cloud
                else:
                    position_confidence = 0.1  # Above cloud

            return position_confidence

        except Exception as e:
            self.logger.error(f"Error calculating price position confidence: {e}")
            return 0.0

    def _calculate_market_context_confidence(self, latest_row: pd.Series, signal_type: str) -> float:
        """Calculate confidence from market context and momentum"""
        try:
            context_confidence = 0.0

            # Volume context if available
            volume = latest_row.get('ltv', 0)
            volume_avg = latest_row.get('volume_avg_20', volume)
            if volume > volume_avg * 1.2:  # Above average volume
                context_confidence += 0.3

            # ATR context for volatility
            atr = latest_row.get('atr', 0)
            if atr > 0:
                context_confidence += 0.2  # Has volatility data

            # RSI context if available
            rsi = latest_row.get('rsi', 50)
            if signal_type == 'BULL' and 40 <= rsi <= 70:
                context_confidence += 0.3  # Good RSI range for bulls
            elif signal_type == 'BEAR' and 30 <= rsi <= 60:
                context_confidence += 0.3  # Good RSI range for bears

            # Trend strength from additional indicators
            macd_histogram = latest_row.get('macd_histogram', 0)
            if signal_type == 'BULL' and macd_histogram > 0:
                context_confidence += 0.2
            elif signal_type == 'BEAR' and macd_histogram < 0:
                context_confidence += 0.2

            return min(1.0, context_confidence)

        except Exception as e:
            self.logger.error(f"Error calculating market context confidence: {e}")
            return 0.0

    def _apply_premium_bonuses(self, latest_row: pd.Series, signal_type: str, base_confidence: float,
                              tk_bull_cross: bool, tk_bear_cross: bool,
                              cloud_bull_breakout: bool, cloud_bear_breakout: bool) -> float:
        """Apply premium bonuses for high-quality signal combinations"""
        try:
            # Perfect Ichimoku setup bonus: TK cross + cloud breakout
            if signal_type == 'BULL' and tk_bull_cross and cloud_bull_breakout:
                base_confidence += 0.1  # Premium combination
                self.logger.debug("Premium bull bonus: TK cross + cloud breakout")
            elif signal_type == 'BEAR' and tk_bear_cross and cloud_bear_breakout:
                base_confidence += 0.1  # Premium combination
                self.logger.debug("Premium bear bonus: TK cross + cloud breakout")

            # Cloud thickness bonus for thick, reliable clouds
            cloud_thickness = latest_row.get('cloud_thickness', 0)
            close = latest_row.get('close', 1)
            thickness_ratio = cloud_thickness / close if close > 0 else 0
            if thickness_ratio > 0.003:  # Very thick cloud
                base_confidence += 0.05
                self.logger.debug("Thick cloud bonus applied")

            # Momentum alignment bonus
            tenkan = latest_row.get('tenkan_sen', 0)
            kijun = latest_row.get('kijun_sen', 0)
            cloud_mid = (latest_row.get('cloud_top', 0) + latest_row.get('cloud_bottom', 0)) / 2

            if signal_type == 'BULL' and close > tenkan > kijun > cloud_mid:
                base_confidence += 0.05  # Perfect bull alignment
            elif signal_type == 'BEAR' and close < tenkan < kijun < cloud_mid:
                base_confidence += 0.05  # Perfect bear alignment

            return base_confidence

        except Exception as e:
            self.logger.error(f"Error applying premium bonuses: {e}")
            return base_confidence

    def validate_confidence_threshold(self, confidence: float, min_threshold: Optional[float] = None) -> bool:
        """
        Validate that confidence meets minimum threshold

        Args:
            confidence: Calculated confidence score
            min_threshold: Minimum required confidence (uses instance default if None)

        Returns:
            True if confidence meets threshold
        """
        threshold = min_threshold or self.min_confidence
        meets_threshold = confidence >= threshold

        if not meets_threshold:
            self.logger.debug(f"Confidence {confidence:.1%} below threshold {threshold:.1%}")

        return meets_threshold

    def calculate_signal_strength(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """
        Calculate detailed signal strength breakdown

        Args:
            latest_row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Dictionary with strength components
        """
        try:
            strength = {
                'tk_strength': latest_row.get('tk_cross_strength', 0),
                'cloud_breakout_strength': (
                    latest_row.get('cloud_breakout_strength_bull', 0) if signal_type == 'BULL'
                    else latest_row.get('cloud_breakout_strength_bear', 0)
                ),
                'overall_strength': latest_row.get(f'{signal_type.lower()}_signal_strength', 0),
                'cloud_thickness_ratio': self._get_cloud_thickness_ratio(latest_row)
            }

            return strength

        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return {'tk_strength': 0, 'cloud_breakout_strength': 0, 'overall_strength': 0}

    def _get_cloud_thickness_ratio(self, latest_row: pd.Series) -> float:
        """Get cloud thickness as ratio of current price"""
        try:
            cloud_thickness = latest_row.get('cloud_thickness', 0)
            close = latest_row.get('close', 1)
            return cloud_thickness / close if close > 0 else 0
        except:
            return 0