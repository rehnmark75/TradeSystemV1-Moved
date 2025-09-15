# core/strategies/helpers/ichimoku_trend_validator.py
"""
Ichimoku Trend Validator Module
Validates Ichimoku Cloud signals against trend components and momentum filters

Key Validations:
- Cloud position validation (price vs Kumo for trend direction)
- Chikou span validation (lagging span clear of historical price action)
- TK line trend validation (Tenkan vs Kijun alignment)
- Multi-component Ichimoku alignment validation
- Cloud thickness validation for signal reliability
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class IchimokuTrendValidator:
    """Handles all trend and momentum validation for Ichimoku Cloud signals"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = 1e-8  # Epsilon for stability

    def validate_cloud_position(self, row: pd.Series, signal_type: str) -> bool:
        """
        CLOUD POSITION FILTER: Ensure signals align with cloud trend direction

        Critical trend filter:
        - BULL signals: Price should be ABOVE cloud (uptrend)
        - BEAR signals: Price should be BELOW cloud (downtrend)

        Args:
            row: DataFrame row with Ichimoku data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if cloud position is correct, False if against trend
        """
        try:
            if not getattr(config, 'ICHIMOKU_CLOUD_FILTER_ENABLED', True):
                return True

            # Get cloud boundaries and price
            cloud_top = row.get('cloud_top', 0)
            cloud_bottom = row.get('cloud_bottom', 0)
            close_price = row.get('close', 0)

            if cloud_top == 0 or cloud_bottom == 0 or close_price == 0:
                self.logger.debug("Cloud or price data not available for validation")
                return True  # Allow signal if data not available

            # Calculate buffer for edge cases
            buffer_pips = getattr(config, 'ICHIMOKU_CLOUD_BUFFER_PIPS', 2.0)
            pip_value = self._get_pip_value(row, close_price)
            buffer = buffer_pips * pip_value

            if signal_type == 'BULL':
                # Bull signals prefer price above cloud
                trend_valid = close_price > (cloud_bottom - buffer)  # Allow some tolerance

                if trend_valid:
                    if close_price > cloud_top:
                        self.logger.debug(f"Cloud position STRONG for BULL: price above cloud")
                    else:
                        self.logger.debug(f"Cloud position OK for BULL: price in/near cloud")
                else:
                    distance_below = (cloud_bottom - close_price) / pip_value
                    self.logger.debug(f"Cloud position WEAK for BULL: price {distance_below:.1f} pips below cloud")

                return trend_valid

            elif signal_type == 'BEAR':
                # Bear signals prefer price below cloud
                trend_valid = close_price < (cloud_top + buffer)  # Allow some tolerance

                if trend_valid:
                    if close_price < cloud_bottom:
                        self.logger.debug(f"Cloud position STRONG for BEAR: price below cloud")
                    else:
                        self.logger.debug(f"Cloud position OK for BEAR: price in/near cloud")
                else:
                    distance_above = (close_price - cloud_top) / pip_value
                    self.logger.debug(f"Cloud position WEAK for BEAR: price {distance_above:.1f} pips above cloud")

                return trend_valid

            return False

        except Exception as e:
            self.logger.error(f"Error validating cloud position: {e}")
            return True  # Allow signal on error

    def validate_chikou_span(self, df: pd.DataFrame, signal_type: str) -> bool:
        """
        CHIKOU SPAN VALIDATION: Ensure Chikou span is clear of historical price action

        The Chikou span (lagging span) is the current close price plotted 26 periods behind.
        For valid signals, it should be clear of historical price action in the direction of the signal.

        Args:
            df: DataFrame with full historical data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if Chikou span confirms the signal direction
        """
        try:
            if not getattr(config, 'ICHIMOKU_CHIKOU_FILTER_ENABLED', True):
                return True

            if len(df) < 50:  # Need enough historical data
                self.logger.debug("Insufficient data for Chikou span validation")
                return True

            latest_row = df.iloc[-1]
            chikou_value = latest_row.get('chikou_span', 0)

            if chikou_value == 0:
                self.logger.debug("Chikou span data not available")
                return True

            # Get historical price data where Chikou is plotted (26 periods ago)
            chikou_periods = getattr(config, 'ICHIMOKU_CHIKOU_PERIODS', 26)
            lookback_periods = 5  # Check 5 periods around the Chikou point

            historical_start = max(0, len(df) - chikou_periods - lookback_periods)
            historical_end = min(len(df), len(df) - chikou_periods + lookback_periods)

            if historical_start >= historical_end:
                self.logger.debug("Invalid historical range for Chikou validation")
                return True

            historical_data = df.iloc[historical_start:historical_end]

            if len(historical_data) == 0:
                return True

            # Get price action in the historical zone
            historical_high = historical_data['high'].max()
            historical_low = historical_data['low'].min()
            historical_close_avg = historical_data['close'].mean()

            # Calculate buffer for edge cases
            buffer_pips = getattr(config, 'ICHIMOKU_CHIKOU_BUFFER_PIPS', 1.0)
            pip_value = self._get_pip_value(latest_row, chikou_value)
            buffer = buffer_pips * pip_value

            if signal_type == 'BULL':
                # For bull signals: Chikou should be ABOVE historical price action
                if chikou_value > historical_high + buffer:
                    self.logger.debug(f"Chikou STRONG for BULL: above historical high")
                    return True
                elif chikou_value > historical_close_avg + buffer:
                    self.logger.debug(f"Chikou OK for BULL: above historical average")
                    return True
                else:
                    self.logger.debug(f"Chikou WEAK for BULL: in/below historical range")
                    return False

            elif signal_type == 'BEAR':
                # For bear signals: Chikou should be BELOW historical price action
                if chikou_value < historical_low - buffer:
                    self.logger.debug(f"Chikou STRONG for BEAR: below historical low")
                    return True
                elif chikou_value < historical_close_avg - buffer:
                    self.logger.debug(f"Chikou OK for BEAR: below historical average")
                    return True
                else:
                    self.logger.debug(f"Chikou WEAK for BEAR: in/above historical range")
                    return False

            return False

        except Exception as e:
            self.logger.error(f"Error validating Chikou span: {e}")
            return True

    def validate_tk_alignment(self, row: pd.Series, signal_type: str) -> bool:
        """
        TK LINE ALIGNMENT VALIDATION: Check Tenkan-Kijun relationship

        Args:
            row: DataFrame row with TK line data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if TK alignment supports the signal
        """
        try:
            if not getattr(config, 'ICHIMOKU_TK_FILTER_ENABLED', True):
                return True

            tenkan = row.get('tenkan_sen', 0)
            kijun = row.get('kijun_sen', 0)

            if tenkan == 0 or kijun == 0:
                self.logger.debug("TK line data not available for validation")
                return True

            # Calculate TK separation strength
            close = row.get('close', 1)
            tk_separation = abs(tenkan - kijun) / close if close > 0 else 0
            min_separation = getattr(config, 'ICHIMOKU_MIN_TK_SEPARATION', 0.0005)

            if signal_type == 'BULL':
                # Bull signals: Tenkan should be above or crossing above Kijun
                tk_bullish = tenkan >= kijun - self.eps
                if tk_bullish and tk_separation >= min_separation:
                    self.logger.debug(f"TK alignment STRONG for BULL: good separation")
                    return True
                elif tk_bullish:
                    self.logger.debug(f"TK alignment OK for BULL: lines close")
                    return True
                else:
                    self.logger.debug(f"TK alignment INVALID for BULL: Tenkan below Kijun")
                    return False

            elif signal_type == 'BEAR':
                # Bear signals: Tenkan should be below or crossing below Kijun
                tk_bearish = tenkan <= kijun + self.eps
                if tk_bearish and tk_separation >= min_separation:
                    self.logger.debug(f"TK alignment STRONG for BEAR: good separation")
                    return True
                elif tk_bearish:
                    self.logger.debug(f"TK alignment OK for BEAR: lines close")
                    return True
                else:
                    self.logger.debug(f"TK alignment INVALID for BEAR: Tenkan above Kijun")
                    return False

            return False

        except Exception as e:
            self.logger.error(f"Error validating TK alignment: {e}")
            return True

    def validate_cloud_thickness(self, row: pd.Series) -> bool:
        """
        CLOUD THICKNESS VALIDATION: Ensure cloud is thick enough for reliable signals

        Thin clouds are less reliable as support/resistance.

        Args:
            row: DataFrame row with cloud data

        Returns:
            True if cloud is thick enough for reliable signals
        """
        try:
            if not getattr(config, 'ICHIMOKU_CLOUD_THICKNESS_FILTER_ENABLED', True):
                return True

            cloud_top = row.get('cloud_top', 0)
            cloud_bottom = row.get('cloud_bottom', 0)
            close = row.get('close', 1)

            if cloud_top == 0 or cloud_bottom == 0:
                return True

            # Calculate thickness as percentage of price
            cloud_thickness = cloud_top - cloud_bottom
            thickness_ratio = cloud_thickness / close if close > 0 else 0

            min_thickness_ratio = getattr(config, 'ICHIMOKU_MIN_CLOUD_THICKNESS_RATIO', 0.001)

            if thickness_ratio >= min_thickness_ratio:
                self.logger.debug(f"Cloud thickness OK: {thickness_ratio:.4f} ratio")
                return True
            else:
                self.logger.debug(f"Cloud too thin: {thickness_ratio:.4f} < {min_thickness_ratio}")
                return False

        except Exception as e:
            self.logger.error(f"Error validating cloud thickness: {e}")
            return True

    def validate_ichimoku_full_alignment(self, row: pd.Series, signal_type: str) -> Tuple[bool, str]:
        """
        FULL ICHIMOKU ALIGNMENT: Check all components for perfect setup

        Perfect Ichimoku setups have all components aligned in the same direction.

        Args:
            row: DataFrame row with all Ichimoku data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Tuple of (is_aligned, alignment_description)
        """
        try:
            tenkan = row.get('tenkan_sen', 0)
            kijun = row.get('kijun_sen', 0)
            cloud_top = row.get('cloud_top', 0)
            cloud_bottom = row.get('cloud_bottom', 0)
            close = row.get('close', 0)

            if any(val == 0 for val in [tenkan, kijun, cloud_top, cloud_bottom, close]):
                return True, "Partial data available"

            cloud_mid = (cloud_top + cloud_bottom) / 2

            if signal_type == 'BULL':
                # Perfect bull alignment: Price > Tenkan > Kijun > Cloud
                perfect = close > tenkan > kijun > cloud_mid
                good = close > tenkan and tenkan > kijun and close > cloud_bottom
                acceptable = close > tenkan and close > cloud_bottom

                if perfect:
                    return True, "Perfect bull alignment"
                elif good:
                    return True, "Good bull alignment"
                elif acceptable:
                    return True, "Acceptable bull alignment"
                else:
                    return False, "Poor bull alignment"

            elif signal_type == 'BEAR':
                # Perfect bear alignment: Price < Tenkan < Kijun < Cloud
                perfect = close < tenkan < kijun < cloud_mid
                good = close < tenkan and tenkan < kijun and close < cloud_top
                acceptable = close < tenkan and close < cloud_top

                if perfect:
                    return True, "Perfect bear alignment"
                elif good:
                    return True, "Good bear alignment"
                elif acceptable:
                    return True, "Acceptable bear alignment"
                else:
                    return False, "Poor bear alignment"

            return False, "Unknown signal type"

        except Exception as e:
            self.logger.error(f"Error validating full alignment: {e}")
            return True, "Error in validation"

    def validate_momentum_confluence(self, row: pd.Series, signal_type: str) -> bool:
        """
        MOMENTUM CONFLUENCE: Check additional momentum indicators

        Validates Ichimoku signals against other momentum indicators for confluence.

        Args:
            row: DataFrame row with momentum data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if momentum indicators confirm the signal
        """
        try:
            if not getattr(config, 'ICHIMOKU_MOMENTUM_FILTER_ENABLED', False):
                return True

            confluence_count = 0
            total_indicators = 0

            # MACD confirmation
            macd_histogram = row.get('macd_histogram', None)
            if macd_histogram is not None:
                total_indicators += 1
                if (signal_type == 'BULL' and macd_histogram > 0) or \
                   (signal_type == 'BEAR' and macd_histogram < 0):
                    confluence_count += 1

            # RSI confirmation
            rsi = row.get('rsi', None)
            if rsi is not None:
                total_indicators += 1
                if (signal_type == 'BULL' and 40 <= rsi <= 70) or \
                   (signal_type == 'BEAR' and 30 <= rsi <= 60):
                    confluence_count += 1

            # Volume confirmation
            volume = row.get('ltv', None)
            volume_avg = row.get('volume_avg_20', None)
            if volume is not None and volume_avg is not None:
                total_indicators += 1
                if volume > volume_avg * 1.1:  # Above average volume
                    confluence_count += 1

            # Require at least 50% confluence if indicators available
            if total_indicators > 0:
                confluence_ratio = confluence_count / total_indicators
                required_confluence = getattr(config, 'ICHIMOKU_MIN_CONFLUENCE_RATIO', 0.5)

                if confluence_ratio >= required_confluence:
                    self.logger.debug(f"Momentum confluence OK: {confluence_count}/{total_indicators}")
                    return True
                else:
                    self.logger.debug(f"Momentum confluence WEAK: {confluence_count}/{total_indicators}")
                    return False

            return True  # No additional indicators, allow signal

        except Exception as e:
            self.logger.error(f"Error validating momentum confluence: {e}")
            return True

    def _get_pip_value(self, row: pd.Series, price: float) -> float:
        """Get pip value based on epic and price level"""
        try:
            epic = row.get('epic', '')

            # JPY pairs have different pip values
            if 'JPY' in epic or price > 50:
                return 0.01  # JPY pairs: 1 pip = 0.01
            else:
                return 0.0001  # Standard pairs: 1 pip = 0.0001

        except Exception:
            return 0.0001  # Safe default

    def get_trend_strength(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate overall trend strength from Ichimoku components

        Args:
            row: DataFrame row with Ichimoku data

        Returns:
            Dictionary with trend strength metrics
        """
        try:
            tenkan = row.get('tenkan_sen', 0)
            kijun = row.get('kijun_sen', 0)
            cloud_top = row.get('cloud_top', 0)
            cloud_bottom = row.get('cloud_bottom', 0)
            close = row.get('close', 0)

            if any(val == 0 for val in [tenkan, kijun, cloud_top, cloud_bottom, close]):
                return {'bull_strength': 0.0, 'bear_strength': 0.0, 'trend_clarity': 0.0}

            # Calculate component strengths
            cloud_mid = (cloud_top + cloud_bottom) / 2

            # Bull strength: how well aligned for uptrend
            bull_strength = 0.0
            if close > tenkan: bull_strength += 0.25
            if tenkan > kijun: bull_strength += 0.25
            if close > cloud_top: bull_strength += 0.3
            if kijun > cloud_mid: bull_strength += 0.2

            # Bear strength: how well aligned for downtrend
            bear_strength = 0.0
            if close < tenkan: bear_strength += 0.25
            if tenkan < kijun: bear_strength += 0.25
            if close < cloud_bottom: bear_strength += 0.3
            if kijun < cloud_mid: bear_strength += 0.2

            # Trend clarity: how clear the direction is
            trend_clarity = abs(bull_strength - bear_strength)

            return {
                'bull_strength': bull_strength,
                'bear_strength': bear_strength,
                'trend_clarity': trend_clarity
            }

        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return {'bull_strength': 0.0, 'bear_strength': 0.0, 'trend_clarity': 0.0}