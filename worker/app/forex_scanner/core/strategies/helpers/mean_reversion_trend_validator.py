# core/strategies/helpers/mean_reversion_trend_validator.py
"""
Mean Reversion Trend Validator Module
Validates mean reversion signals against trend strength and mean reversion zone analysis
Ensures signals occur in appropriate market conditions for mean reversion strategies
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

try:
    from forex_scanner.configdata.strategies import config_mean_reversion_strategy as mr_config
except ImportError:
    import configdata.strategies.config_mean_reversion_strategy as mr_config


class MeanReversionTrendValidator:
    """Handles trend analysis and mean reversion zone validation for mean reversion signals"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = 1e-8  # Epsilon for stability

        # Load configuration
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict:
        """Load mean reversion validation configuration"""
        return {
            # Mean reversion zone settings
            'zone_validation_enabled': mr_config.MEAN_REVERSION_ZONE_ENABLED,
            'zone_lookback_periods': mr_config.MEAN_REVERSION_LOOKBACK_PERIODS,
            'zone_multiplier': mr_config.MEAN_REVERSION_ZONE_MULTIPLIER,
            'require_zone_touch': mr_config.MEAN_REVERSION_REQUIRE_ZONE_TOUCH,
            'min_zone_distance': mr_config.MEAN_REVERSION_MIN_ZONE_DISTANCE,
            'max_zone_age': mr_config.MEAN_REVERSION_MAX_ZONE_AGE,

            # Market regime settings
            'regime_detection_enabled': mr_config.MARKET_REGIME_DETECTION_ENABLED,
            'disable_in_strong_trend': mr_config.MARKET_REGIME_DISABLE_IN_STRONG_TREND,
            'boost_in_ranging': mr_config.MARKET_REGIME_BOOST_IN_RANGING,
            'trend_strength_threshold': mr_config.MARKET_REGIME_TREND_STRENGTH_THRESHOLD,
            'volatility_period': mr_config.MARKET_REGIME_VOLATILITY_PERIOD,
            'trend_period': mr_config.MARKET_REGIME_TREND_PERIOD,
            'ranging_threshold': mr_config.MARKET_REGIME_RANGING_THRESHOLD,

            # Multi-timeframe settings
            'mtf_analysis_enabled': mr_config.MTF_ANALYSIS_ENABLED,
            'mtf_timeframes': mr_config.MTF_TIMEFRAMES,
            'mtf_min_alignment': mr_config.MTF_MIN_ALIGNMENT_SCORE,
            'require_higher_tf_confluence': mr_config.MTF_REQUIRE_HIGHER_TF_CONFLUENCE,

            # Supporting filters
            'ema_200_filter_enabled': getattr(mr_config, 'EMA_200_TREND_FILTER_ENABLED', True),
            'vwap_deviation_threshold': 2.0,  # Maximum VWAP deviation percentage
            'adx_ranging_threshold': 25,  # ADX below this = ranging market
            'adx_strong_trend_threshold': 50  # ADX above this = very strong trend
        }

    def validate_mean_reversion_zone(self, df: pd.DataFrame, signal_idx: int, signal_type: str, epic: str) -> bool:
        """
        Validate that price is in a mean reversion zone

        Mean reversion zones are areas where price has historically reverted from.
        These are calculated using statistical analysis of price movements.

        Args:
            df: DataFrame with price data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            True if price is in a valid mean reversion zone
        """
        try:
            if not self.config['zone_validation_enabled']:
                return True

            # Get current price and calculate mean reversion zones
            current_price = df.iloc[signal_idx]['close']
            zones = self._calculate_mean_reversion_zones(df, signal_idx, epic)

            if not zones:
                self.logger.debug("No mean reversion zones calculated")
                return True  # Allow signal if zones can't be calculated

            # Check if current price is near a mean reversion zone
            zone_touched = False
            min_distance = self.config['min_zone_distance']

            # Convert min_distance from pips to price units
            if 'JPY' in epic:
                pip_value = 0.01
            else:
                pip_value = 0.0001
            min_distance_price = min_distance * pip_value

            for zone in zones:
                zone_price = zone['price']
                zone_type = zone['type']  # 'support' or 'resistance'
                zone_age = signal_idx - zone['bar_idx']

                # Check zone age
                if zone_age > self.config['max_zone_age']:
                    continue

                # Check distance to zone
                distance = abs(current_price - zone_price)
                if distance <= min_distance_price:
                    # Price is touching a zone
                    if signal_type == 'BULL' and zone_type == 'support':
                        zone_touched = True
                        self.logger.debug(f"BULL signal: Price {current_price:.5f} touching support zone {zone_price:.5f}")
                        break
                    elif signal_type == 'BEAR' and zone_type == 'resistance':
                        zone_touched = True
                        self.logger.debug(f"BEAR signal: Price {current_price:.5f} touching resistance zone {zone_price:.5f}")
                        break

            if self.config['require_zone_touch'] and not zone_touched:
                self.logger.debug(f"{signal_type} signal rejected: Price not touching mean reversion zone")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating mean reversion zone: {e}")
            return True  # Allow on error

    def _calculate_mean_reversion_zones(self, df: pd.DataFrame, current_idx: int, epic: str) -> List[Dict]:
        """
        Calculate mean reversion zones using statistical analysis

        Zones are identified as areas where price has historically shown strong rejection
        (rapid reversal after touching the level).
        """
        try:
            zones = []
            lookback = self.config['zone_lookback_periods']
            multiplier = self.config['zone_multiplier']

            if current_idx < lookback:
                return zones

            # Get historical data window
            start_idx = max(0, current_idx - lookback)
            window_data = df.iloc[start_idx:current_idx]

            # Calculate statistical levels
            mean_price = window_data['close'].mean()
            std_price = window_data['close'].std()

            # Define potential zones based on standard deviations
            resistance_level = mean_price + (std_price * multiplier)
            support_level = mean_price - (std_price * multiplier)

            # Find recent touches and rejections
            recent_highs = window_data['high'].rolling(window=5, center=True).max()
            recent_lows = window_data['low'].rolling(window=5, center=True).min()

            # Check for resistance zone validation
            resistance_touches = 0
            for i in range(len(window_data)):
                if abs(recent_highs.iloc[i] - resistance_level) < (std_price * 0.2):
                    # Price touched near resistance level
                    resistance_touches += 1

            # Check for support zone validation
            support_touches = 0
            for i in range(len(window_data)):
                if abs(recent_lows.iloc[i] - support_level) < (std_price * 0.2):
                    # Price touched near support level
                    support_touches += 1

            # Add validated zones
            if resistance_touches >= 2:  # At least 2 touches for valid zone
                zones.append({
                    'price': resistance_level,
                    'type': 'resistance',
                    'strength': resistance_touches,
                    'bar_idx': current_idx - 1
                })

            if support_touches >= 2:  # At least 2 touches for valid zone
                zones.append({
                    'price': support_level,
                    'type': 'support',
                    'strength': support_touches,
                    'bar_idx': current_idx - 1
                })

            return zones

        except Exception as e:
            self.logger.error(f"Error calculating mean reversion zones: {e}")
            return []

    def validate_market_regime(self, df: pd.DataFrame, signal_idx: int, signal_type: str) -> bool:
        """
        Validate that market regime is suitable for mean reversion trading

        Mean reversion works best in:
        - Ranging markets (low ADX)
        - Markets with normal volatility
        - Markets without strong directional bias

        Args:
            df: DataFrame with indicator data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if market regime supports mean reversion
        """
        try:
            if not self.config['regime_detection_enabled']:
                return True

            row = df.iloc[signal_idx]

            # 1. Trend Strength Analysis (ADX)
            adx = row.get('adx', 0)

            if self.config['disable_in_strong_trend']:
                # Strong trend = poor environment for mean reversion
                if adx > self.config['adx_strong_trend_threshold']:
                    self.logger.debug(f"{signal_type} signal rejected: Very strong trend (ADX: {adx:.1f})")
                    return False

            # 2. Volatility Analysis
            atr = row.get('atr', 0)
            if atr > 0 and len(df) >= self.config['volatility_period']:
                # Get recent volatility context
                recent_atr = df['atr'].iloc[max(0, signal_idx - self.config['volatility_period']):signal_idx].mean()

                # Extreme volatility is bad for mean reversion
                if atr > (recent_atr * 2.0):
                    self.logger.debug(f"{signal_type} signal rejected: Extreme volatility (ATR: {atr:.6f} vs avg: {recent_atr:.6f})")
                    return False

            # 3. Price Movement Efficiency
            if len(df) >= 20:
                # Calculate movement efficiency over last 20 bars
                price_change = abs(row['close'] - df.iloc[signal_idx - 19]['close'])
                high_range = df['high'].iloc[signal_idx - 19:signal_idx + 1].max()
                low_range = df['low'].iloc[signal_idx - 19:signal_idx + 1].min()
                total_range = high_range - low_range

                if total_range > 0:
                    efficiency = price_change / total_range

                    # Low efficiency = choppy/ranging market = good for mean reversion
                    if efficiency > 0.7:  # High efficiency = trending
                        self.logger.debug(f"{signal_type} signal rejected: High price efficiency (trending): {efficiency:.2f}")
                        return False

            # 4. VWAP Deviation Analysis
            vwap = row.get('vwap', 0)
            vwap_deviation = row.get('vwap_deviation', 0)

            if abs(vwap_deviation) > self.config['vwap_deviation_threshold']:
                self.logger.debug(f"{signal_type} signal rejected: Extreme VWAP deviation: {vwap_deviation:.2f}%")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating market regime: {e}")
            return True

    def validate_multi_timeframe_confluence(self, df: pd.DataFrame, signal_idx: int, signal_type: str) -> bool:
        """
        Validate multi-timeframe confluence for mean reversion signals

        Checks that higher timeframes are not showing strong directional bias
        that would work against the mean reversion signal.

        Args:
            df: DataFrame with MTF indicator data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if MTF confluence supports the signal
        """
        try:
            if not self.config['mtf_analysis_enabled']:
                return True

            row = df.iloc[signal_idx]

            # Check MTF RSI alignment
            mtf_bull_alignment = row.get('mtf_bull_alignment', 0)
            mtf_bear_alignment = row.get('mtf_bear_alignment', 0)

            if signal_type == 'BULL':
                # For bull signals, we want some bullish alignment but not extreme overbought
                if mtf_bull_alignment < 0.3:  # Too bearish across timeframes
                    self.logger.debug(f"BULL signal rejected: Poor MTF alignment ({mtf_bull_alignment:.2f})")
                    return False
                if mtf_bull_alignment > 0.9:  # Too overbought across timeframes
                    self.logger.debug(f"BULL signal rejected: Extreme MTF overbought ({mtf_bull_alignment:.2f})")
                    return False

            else:  # BEAR
                # For bear signals, we want some bearish alignment but not extreme oversold
                if mtf_bear_alignment < 0.3:  # Too bullish across timeframes
                    self.logger.debug(f"BEAR signal rejected: Poor MTF alignment ({mtf_bear_alignment:.2f})")
                    return False
                if mtf_bear_alignment > 0.9:  # Too oversold across timeframes
                    self.logger.debug(f"BEAR signal rejected: Extreme MTF oversold ({mtf_bear_alignment:.2f})")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating MTF confluence: {e}")
            return True

    def validate_ema_200_trend_filter(self, df: pd.DataFrame, signal_idx: int, signal_type: str) -> bool:
        """
        Validate EMA 200 trend filter for mean reversion signals

        For mean reversion, we want to be more permissive than trending strategies.
        We allow counter-trend signals but with reduced confidence.

        Args:
            df: DataFrame with EMA data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if EMA 200 filter passes (more lenient for mean reversion)
        """
        try:
            if not self.config['ema_200_filter_enabled']:
                return True

            row = df.iloc[signal_idx]
            ema_200 = row.get('ema_200', 0)
            close_price = row.get('close', 0)

            if ema_200 == 0 or close_price == 0:
                return True  # Allow if data not available

            # Calculate distance from EMA 200 as percentage
            distance_pct = abs(close_price - ema_200) / ema_200 * 100

            # For mean reversion, we're more interested in extreme deviations
            # rather than strict trend alignment
            if signal_type == 'BULL':
                # Bull signals: Allow if price is significantly below EMA 200
                # (oversold condition good for mean reversion)
                if close_price < ema_200:
                    # Price below EMA 200 - good for bull mean reversion
                    return True
                else:
                    # Price above EMA 200 - still allow but check distance
                    if distance_pct > 3.0:  # More than 3% above EMA 200
                        self.logger.debug(f"BULL signal caution: Price {distance_pct:.2f}% above EMA 200")
                        return False
                    return True

            else:  # BEAR
                # Bear signals: Allow if price is significantly above EMA 200
                # (overbought condition good for mean reversion)
                if close_price > ema_200:
                    # Price above EMA 200 - good for bear mean reversion
                    return True
                else:
                    # Price below EMA 200 - still allow but check distance
                    if distance_pct > 3.0:  # More than 3% below EMA 200
                        self.logger.debug(f"BEAR signal caution: Price {distance_pct:.2f}% below EMA 200")
                        return False
                    return True

        except Exception as e:
            self.logger.error(f"Error validating EMA 200 trend filter: {e}")
            return True

    def validate_squeeze_momentum_timing(self, df: pd.DataFrame, signal_idx: int, signal_type: str) -> bool:
        """
        Validate squeeze momentum timing for mean reversion entries

        Best mean reversion signals occur when:
        - Squeeze has been active (low volatility/consolidation)
        - Squeeze is releasing or just released
        - Momentum is building in signal direction

        Args:
            df: DataFrame with squeeze indicator data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if squeeze timing supports the signal
        """
        try:
            row = df.iloc[signal_idx]
            squeeze_on = row.get('squeeze_on', False)
            squeeze_momentum = row.get('squeeze_momentum', 0)

            # Check if squeeze was recently active (good setup for mean reversion)
            squeeze_history = []
            lookback = min(10, signal_idx)  # Look back up to 10 bars

            for i in range(lookback):
                hist_idx = signal_idx - i
                if hist_idx >= 0:
                    hist_squeeze = df.iloc[hist_idx].get('squeeze_on', False)
                    squeeze_history.append(hist_squeeze)

            recent_squeeze_count = sum(squeeze_history)

            # Good mean reversion setup if squeeze was recently active
            if recent_squeeze_count >= 3:  # At least 3 out of last 10 bars in squeeze
                self.logger.debug(f"{signal_type} signal: Good squeeze setup ({recent_squeeze_count}/10 recent bars in squeeze)")

                # Check momentum direction if squeeze is releasing
                if not squeeze_on:  # Squeeze releasing
                    if signal_type == 'BULL' and squeeze_momentum > 0:
                        return True
                    elif signal_type == 'BEAR' and squeeze_momentum < 0:
                        return True
                    else:
                        # Momentum in wrong direction - still allow but note
                        self.logger.debug(f"{signal_type} signal: Squeeze releasing but momentum opposite")
                        return True
                else:
                    # Still in squeeze - allow signal (breakout may be coming)
                    return True
            else:
                # No recent squeeze - less ideal but still allow
                return True

        except Exception as e:
            self.logger.error(f"Error validating squeeze momentum timing: {e}")
            return True

    def validate_all_trend_filters(self, df: pd.DataFrame, signal_idx: int, signal_type: str, epic: str) -> Dict:
        """
        Run all trend validation filters and return detailed results

        Args:
            df: DataFrame with all indicator data
            signal_idx: Index of the signal bar
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Dictionary with validation results for each filter
        """
        try:
            results = {
                'mean_reversion_zone': self.validate_mean_reversion_zone(df, signal_idx, signal_type, epic),
                'market_regime': self.validate_market_regime(df, signal_idx, signal_type),
                'mtf_confluence': self.validate_multi_timeframe_confluence(df, signal_idx, signal_type),
                'ema_200_filter': self.validate_ema_200_trend_filter(df, signal_idx, signal_type),
                'squeeze_timing': self.validate_squeeze_momentum_timing(df, signal_idx, signal_type)
            }

            # Calculate overall pass rate
            passed = sum(1 for result in results.values() if result)
            total = len(results)
            results['overall_pass_rate'] = passed / total if total > 0 else 0.0
            results['all_passed'] = all(results.values())

            # Log summary for failed validations
            if not results['all_passed']:
                failed_filters = [name for name, result in results.items()
                                if not result and name not in ['overall_pass_rate', 'all_passed']]
                self.logger.debug(f"{signal_type} signal failed trend filters: {failed_filters}")

            return results

        except Exception as e:
            self.logger.error(f"Error running trend validation filters: {e}")
            return {'error': str(e), 'all_passed': False, 'overall_pass_rate': 0.0}

    def apply_mean_reversion_validation(self, validation_results: Dict, min_pass_rate: float = 0.6) -> bool:
        """
        Apply mean reversion specific validation logic

        For mean reversion strategies, we're more lenient than trending strategies
        but still require core mean reversion conditions to be met.

        Args:
            validation_results: Results from validate_all_trend_filters
            min_pass_rate: Minimum pass rate required (0.6 = 60%)

        Returns:
            True if signal should be allowed under mean reversion rules
        """
        try:
            pass_rate = validation_results.get('overall_pass_rate', 0.0)

            # Critical filters for mean reversion
            # Market regime is most important (don't trade mean reversion in strong trends)
            market_regime_ok = validation_results.get('market_regime', False)

            # Mean reversion zone validation (if enabled)
            zone_validation_ok = validation_results.get('mean_reversion_zone', True)  # Default True if disabled

            # At minimum, require market regime and zone validation
            critical_pass = market_regime_ok and zone_validation_ok

            # Overall validation
            mean_reversion_valid = critical_pass and (pass_rate >= min_pass_rate)

            if mean_reversion_valid and not validation_results.get('all_passed', False):
                self.logger.debug(f"Signal allowed under mean reversion validation: {pass_rate:.1%} pass rate")

            return mean_reversion_valid

        except Exception as e:
            self.logger.error(f"Error applying mean reversion validation: {e}")
            return False

    def get_trend_analysis_summary(self, df: pd.DataFrame, signal_idx: int) -> Dict:
        """
        Get comprehensive trend analysis summary for debugging/analysis

        Args:
            df: DataFrame with indicator data
            signal_idx: Index of the signal bar

        Returns:
            Dictionary with trend analysis components
        """
        try:
            row = df.iloc[signal_idx]

            return {
                'adx': row.get('adx', 0),
                'atr': row.get('atr', 0),
                'ema_200': row.get('ema_200', 0),
                'close_price': row.get('close', 0),
                'price_vs_ema200': ((row.get('close', 0) - row.get('ema_200', 0)) / row.get('ema_200', 1)) * 100 if row.get('ema_200', 0) > 0 else 0,
                'vwap_deviation': row.get('vwap_deviation', 0),
                'squeeze_on': row.get('squeeze_on', False),
                'squeeze_momentum': row.get('squeeze_momentum', 0),
                'mtf_bull_alignment': row.get('mtf_bull_alignment', 0),
                'mtf_bear_alignment': row.get('mtf_bear_alignment', 0),
                'market_regime': 'strong_trend' if row.get('adx', 0) > 50 else 'moderate_trend' if row.get('adx', 0) > 25 else 'ranging',
                'volatility_state': 'high' if (row.get('atr', 0) > df['atr'].iloc[max(0, signal_idx-20):signal_idx].mean() * 1.5) else 'normal'
            }

        except Exception as e:
            self.logger.error(f"Error getting trend analysis summary: {e}")
            return {'error': str(e)}