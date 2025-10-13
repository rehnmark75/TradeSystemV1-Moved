# core/strategies/volume_profile_strategy.py
"""
Volume Profile Strategy Implementation
Institutional-grade Volume-by-Price analysis for forex markets

Features:
- High Volume Node (HVN) support/resistance detection
- Low Volume Node (LVN) breakout identification
- Point of Control (POC) mean reversion signals
- Value Area (VAH/VAL) boundary trading
- Compatible with backtest system and live trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

try:
    from .helpers.volume_profile_calculator import VolumeProfileCalculator
    from .helpers.volume_profile_analyzer import VolumeProfileAnalyzer
    from analysis.volume_profile import VolumeProfile, VolumeNode
except ImportError:
    from forex_scanner.core.strategies.helpers.volume_profile_calculator import VolumeProfileCalculator
    from forex_scanner.core.strategies.helpers.volume_profile_analyzer import VolumeProfileAnalyzer
    from forex_scanner.analysis.volume_profile import VolumeProfile, VolumeNode

try:
    from configdata import config
    from configdata.strategies import config_volume_profile_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_volume_profile_strategy
    except ImportError:
        config_volume_profile_strategy = None


class VolumeProfileStrategy(BaseStrategy):
    """
    Volume Profile Strategy - Institutional Level Analysis

    Trading Logic:
    1. HVN Bounce: Buy at HVN support, Sell at HVN resistance
    2. POC Reversion: Price gravitates back to Point of Control
    3. Value Area Breakout: Price breaks VAH (bullish) or VAL (bearish)
    4. LVN Breakout: Price breaks through weak volume zones

    Works in ranging AND trending markets (adaptive strategy)
    """

    def __init__(self,
                 data_fetcher=None,
                 backtest_mode: bool = False,
                 epic: str = None,
                 timeframe: str = '15m',
                 vp_config_name: str = None,
                 **kwargs):
        # Initialize
        self.name = 'volume_profile'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(logging.INFO)

        # Basic config
        self.backtest_mode = backtest_mode
        self.epic = epic
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.price_adjuster = PriceAdjuster()

        # Load configuration preset
        self.vp_config_name = vp_config_name or getattr(
            config_volume_profile_strategy, 'ACTIVE_VP_CONFIG', 'default'
        ) if config_volume_profile_strategy else 'default'

        self.vp_config = self._load_vp_config(epic, self.vp_config_name)

        # Core Volume Profile settings
        self.lookback_periods = self.vp_config.get('lookback_periods', 50)
        self.min_confidence = self.vp_config.get('min_confidence', 0.60)

        # Signal type configuration
        self.enable_hvn_bounce = self.vp_config.get('enable_mean_reversion', True)
        self.enable_poc_reversion = self.vp_config.get('enable_poc_bounce', True)
        self.enable_breakout_signals = self.vp_config.get('enable_breakout_signals', True)
        self.enable_lvn_breakout = self.vp_config.get('enable_lvn_breakout', False)

        # Proximity thresholds (in pips)
        self.hvn_proximity_threshold = self.vp_config.get('hvn_proximity_threshold_pips', 5.0)
        self.lvn_proximity_threshold = self.vp_config.get('lvn_proximity_threshold_pips', 5.0)
        self.poc_proximity_threshold = self.vp_config.get('poc_proximity_threshold_pips', 3.0)

        # Get pip value for this pair
        self.pip_value = self._get_pip_value(epic)

        # Initialize Volume Profile calculator and analyzer
        self.vp_calculator = VolumeProfileCalculator(
            pip_value=self.pip_value
        )

        self.vp_analyzer = VolumeProfileAnalyzer(pip_value=self.pip_value)

        # Risk management settings
        self.stop_loss_config = self._load_stop_loss_config(epic)
        self.take_profit_config = self._load_take_profit_config(epic)

        # Caching for performance
        self.cached_profile: Optional[VolumeProfile] = None
        self.cache_timestamp = None
        self.cache_ttl_bars = getattr(
            config_volume_profile_strategy, 'VP_PERFORMANCE', {}
        ).get('cache_ttl_bars', 5) if config_volume_profile_strategy else 5

        # Signal cooldown to prevent spam (track last signal bar)
        self.last_signal_bar = None
        self.min_bars_between_signals = 5  # Minimum 5 bars between signals

        # Logging
        epic_display = f"{epic} " if epic else ""
        self.logger.info(f"🎯 {epic_display}Volume Profile Strategy initialized")
        self.logger.info(f"📊 Config: {self.vp_config_name} - Lookback: {self.lookback_periods} bars")
        self.logger.info(f"📈 Signal Types: HVN Bounce={self.enable_hvn_bounce}, "
                        f"POC Reversion={self.enable_poc_reversion}, "
                        f"Breakouts={self.enable_breakout_signals}")
        self.logger.info(f"🎚️ Proximity Thresholds: HVN={self.hvn_proximity_threshold} pips, "
                        f"POC={self.poc_proximity_threshold} pips")
        self.logger.info(f"💰 Min Confidence: {self.min_confidence:.0%}")

        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Cache disabled for accuracy")

    def _load_vp_config(self, epic: str, preset: str) -> Dict:
        """Load Volume Profile configuration for epic and preset"""
        if config_volume_profile_strategy and hasattr(config_volume_profile_strategy, 'get_vp_config_for_epic'):
            return config_volume_profile_strategy.get_vp_config_for_epic(epic, preset)

        # Fallback to default config
        default_config = {
            'lookback_periods': 50,
            'min_confidence': 0.60,
            'hvn_proximity_threshold_pips': 5.0,
            'lvn_proximity_threshold_pips': 5.0,
            'poc_proximity_threshold_pips': 3.0,
            'enable_mean_reversion': True,
            'enable_breakout_signals': True,
            'enable_poc_bounce': True,
        }
        return default_config

    def _load_stop_loss_config(self, epic: str) -> Dict:
        """Load stop loss configuration"""
        if config_volume_profile_strategy and hasattr(config_volume_profile_strategy, 'get_vp_stop_loss_config'):
            return config_volume_profile_strategy.get_vp_stop_loss_config(epic)

        return {
            'method': 'hvn_based',
            'hvn_buffer_pips': 2.0,
            'atr_multiplier': 1.5,
            'min_stop_pips': 10.0,
            'max_stop_pips': 40.0,
        }

    def _load_take_profit_config(self, epic: str) -> Dict:
        """Load take profit configuration"""
        if config_volume_profile_strategy and hasattr(config_volume_profile_strategy, 'get_vp_take_profit_config'):
            return config_volume_profile_strategy.get_vp_take_profit_config(epic)

        return {
            'method': 'hvn_based',
            'target_next_hvn': True,
            'target_poc': True,
            'atr_multiplier': 3.0,
            'min_reward_to_risk': 1.5,
        }

    def _get_pip_value(self, epic: str) -> float:
        """Get pip value for currency pair"""
        if not epic:
            return 0.0001

        # JPY pairs use different pip value
        if 'JPY' in epic.upper():
            return 0.01
        else:
            return 0.0001

    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> Optional[Dict]:
        """
        Detect Volume Profile signals

        Args:
            df: DataFrame with OHLCV data
            epic: Currency pair epic
            spread_pips: Current spread in pips
            timeframe: Trading timeframe

        Returns:
            Signal dictionary or None
        """
        try:
            # Minimum data validation
            if len(df) < self.lookback_periods + 10:
                self.logger.debug(f"Insufficient data: {len(df)} bars (need {self.lookback_periods + 10})")
                return None

            # Signal cooldown check - prevent generating signals on every bar
            current_bar_index = len(df) - 1
            if self.last_signal_bar is not None:
                bars_since_last_signal = current_bar_index - self.last_signal_bar
                if bars_since_last_signal < self.min_bars_between_signals:
                    return None  # Too soon since last signal

            # Calculate Volume Profile
            profile = self._calculate_volume_profile(df)
            if profile is None:
                return None

            # Get current price
            current_price = float(df['close'].iloc[-1])

            # Analyze price position relative to volume profile
            position_analysis = self.vp_analyzer.analyze_price_position(current_price, profile)

            # Try to generate signals in priority order
            signal = None

            # Priority 1: HVN Bounce (highest confidence)
            if self.enable_hvn_bounce and not signal:
                signal = self._check_hvn_bounce(df, current_price, profile, position_analysis,
                                               epic, spread_pips, timeframe)

            # Priority 2: POC Reversion
            if self.enable_poc_reversion and not signal:
                signal = self._check_poc_reversion(df, current_price, profile, position_analysis,
                                                   epic, spread_pips, timeframe)

            # Priority 3: Value Area Breakout
            if self.enable_breakout_signals and not signal:
                signal = self._check_value_area_breakout(df, current_price, profile, position_analysis,
                                                        epic, spread_pips, timeframe)

            # Priority 4: LVN Breakout
            if self.enable_lvn_breakout and not signal:
                signal = self._check_lvn_breakout(df, current_price, profile, position_analysis,
                                                 epic, spread_pips, timeframe)

            # Update last signal bar if we generated a signal
            if signal:
                self.last_signal_bar = current_bar_index

            return signal

        except Exception as e:
            self.logger.error(f"Error detecting Volume Profile signal: {e}", exc_info=True)
            return None

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Optional[VolumeProfile]:
        """
        Calculate volume profile with caching

        Args:
            df: DataFrame with OHLCV data

        Returns:
            VolumeProfile or None
        """
        try:
            # Check cache (disabled in backtest mode for accuracy)
            current_bar_time = df['time'].iloc[-1] if 'time' in df.columns else None

            if not self.backtest_mode and self.cached_profile and current_bar_time:
                bars_since_cache = len(df) - self.cache_timestamp if self.cache_timestamp else 999
                if bars_since_cache < self.cache_ttl_bars:
                    return self.cached_profile

            # Calculate new profile
            profile = self.vp_calculator.calculate_profile(df, lookback_periods=self.lookback_periods)

            # Update cache
            if not self.backtest_mode:
                self.cached_profile = profile
                self.cache_timestamp = len(df)

            # Log profile summary
            if profile:
                self.logger.debug(f"📊 Volume Profile: POC={profile.poc:.5f}, "
                                 f"VAH={profile.vah:.5f}, VAL={profile.val:.5f}, "
                                 f"HVNs={len(profile.hvn_zones)}, LVNs={len(profile.lvn_zones)}")

            return profile

        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}", exc_info=True)
            return None

    def _check_hvn_bounce(self, df: pd.DataFrame, current_price: float, profile: VolumeProfile,
                          position_analysis: Dict, epic: str, spread_pips: float,
                          timeframe: str) -> Optional[Dict]:
        """Check for HVN bounce signals (support/resistance)"""

        # Check if price is at HVN
        is_at_hvn, nearest_hvn = self.vp_analyzer.is_at_hvn(
            current_price, profile, self.hvn_proximity_threshold
        )

        if not is_at_hvn or not nearest_hvn:
            return None

        # Determine signal direction based on price position
        # If price is below POC and at HVN = potential bounce UP (BUY)
        # If price is above POC and at HVN = potential bounce DOWN (SELL)

        if position_analysis['distance_to_poc_pips'] < 0:
            # Price below POC - potential BUY at HVN support
            signal_type = 'BUY'
        else:
            # Price above POC - potential SELL at HVN resistance
            signal_type = 'SELL'

        # Calculate confidence
        confidence = self.vp_analyzer.get_signal_confidence(current_price, profile, signal_type)

        # Check minimum confidence
        if confidence < self.min_confidence:
            return None

        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(signal_type, current_price, profile, df)
        take_profit = self._calculate_take_profit(signal_type, current_price, profile, df)

        # Validate R:R ratio
        if not self._validate_risk_reward(current_price, stop_loss, take_profit, signal_type):
            return None

        return {
            'type': signal_type,
            'strategy': 'volume_profile',
            'signal_source': 'hvn_bounce',
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': df['time'].iloc[-1] if 'time' in df.columns else datetime.utcnow(),
            'metadata': {
                'hvn_price': nearest_hvn.price_center,
                'hvn_strength': nearest_hvn.strength,
                'poc': profile.poc,
                'vah': profile.vah,
                'val': profile.val,
                'distance_to_poc_pips': position_analysis['distance_to_poc_pips'],
                'position': position_analysis['position'],
            }
        }

    def _check_poc_reversion(self, df: pd.DataFrame, current_price: float, profile: VolumeProfile,
                            position_analysis: Dict, epic: str, spread_pips: float,
                            timeframe: str) -> Optional[Dict]:
        """Check for POC mean reversion signals"""

        # Price must be away from POC (not already at it)
        distance_to_poc = abs(position_analysis['distance_to_poc_pips'])

        # Much stricter POC distance requirements
        if distance_to_poc < 15.0:  # Too close to POC (increased from 10)
            return None

        if distance_to_poc > 40.0:  # Too far from POC (decreased from 50)
            return None

        # Check that we're actually moving TOWARD POC (fresh signal)
        if len(df) < 2:
            return None

        prev_price = float(df['close'].iloc[-2])
        prev_distance = abs(prev_price - profile.poc) / self.pip_value

        # Require that distance to POC is decreasing (moving toward POC)
        if distance_to_poc >= prev_distance:
            return None  # Not moving toward POC

        # Signal direction: move TOWARD POC
        if position_analysis['distance_to_poc_pips'] > 0:
            # Price above POC - SELL to revert down to POC
            signal_type = 'SELL'
        else:
            # Price below POC - BUY to revert up to POC
            signal_type = 'BUY'

        # Calculate confidence
        confidence = self.vp_analyzer.get_signal_confidence(current_price, profile, signal_type)

        # POC reversion slightly lower confidence than HVN bounce
        confidence = confidence * 0.95

        if confidence < self.min_confidence:
            return None

        # Calculate stop loss and take profit (target = POC)
        stop_loss = self._calculate_stop_loss(signal_type, current_price, profile, df)
        take_profit = profile.poc  # Target the POC

        # Validate R:R ratio
        if not self._validate_risk_reward(current_price, stop_loss, take_profit, signal_type):
            return None

        return {
            'type': signal_type,
            'strategy': 'volume_profile',
            'signal_source': 'poc_reversion',
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': df['time'].iloc[-1] if 'time' in df.columns else datetime.utcnow(),
            'metadata': {
                'poc': profile.poc,
                'distance_to_poc_pips': distance_to_poc,
                'position': position_analysis['position'],
            }
        }

    def _check_value_area_breakout(self, df: pd.DataFrame, current_price: float, profile: VolumeProfile,
                                   position_analysis: Dict, epic: str, spread_pips: float,
                                   timeframe: str) -> Optional[Dict]:
        """Check for Value Area breakout signals"""

        # Must be outside value area
        if position_analysis['within_value_area']:
            return None

        # Check if breakout is recent (price just crossed VAH/VAL)
        prev_price = float(df['close'].iloc[-2])

        # Bullish breakout: price breaks above VAH
        if current_price > profile.vah and prev_price <= profile.vah:
            signal_type = 'BUY'
        # Bearish breakout: price breaks below VAL
        elif current_price < profile.val and prev_price >= profile.val:
            signal_type = 'SELL'
        else:
            return None

        # Calculate confidence (breakouts are slightly higher risk)
        confidence = self.vp_analyzer.get_signal_confidence(current_price, profile, signal_type)
        confidence = confidence * 0.90  # Reduce confidence for breakouts

        if confidence < self.min_confidence:
            return None

        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(signal_type, current_price, profile, df)
        take_profit = self._calculate_take_profit(signal_type, current_price, profile, df)

        # Validate R:R ratio
        if not self._validate_risk_reward(current_price, stop_loss, take_profit, signal_type):
            return None

        return {
            'type': signal_type,
            'strategy': 'volume_profile',
            'signal_source': 'value_area_breakout',
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': df['time'].iloc[-1] if 'time' in df.columns else datetime.utcnow(),
            'metadata': {
                'vah': profile.vah,
                'val': profile.val,
                'poc': profile.poc,
                'breakout_direction': 'upward' if signal_type == 'BUY' else 'downward',
            }
        }

    def _check_lvn_breakout(self, df: pd.DataFrame, current_price: float, profile: VolumeProfile,
                           position_analysis: Dict, epic: str, spread_pips: float,
                           timeframe: str) -> Optional[Dict]:
        """Check for LVN breakout signals (break through weak zones)"""

        # Check if price is at LVN
        is_at_lvn, nearest_lvn = self.vp_analyzer.is_at_lvn(
            current_price, profile, self.lvn_proximity_threshold
        )

        if not is_at_lvn or not nearest_lvn:
            return None

        # Determine breakout direction based on momentum
        # LVN = weak zone, price tends to break through quickly
        prev_price = float(df['close'].iloc[-2])
        price_momentum = current_price - prev_price

        if price_momentum > 0:
            signal_type = 'BUY'  # Breaking up through LVN
        else:
            signal_type = 'SELL'  # Breaking down through LVN

        # Calculate confidence
        confidence = self.vp_analyzer.get_signal_confidence(current_price, profile, signal_type)
        confidence = confidence * 0.85  # Lower confidence for LVN breakouts

        if confidence < self.min_confidence:
            return None

        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(signal_type, current_price, profile, df)
        take_profit = self._calculate_take_profit(signal_type, current_price, profile, df)

        # Validate R:R ratio
        if not self._validate_risk_reward(current_price, stop_loss, take_profit, signal_type):
            return None

        return {
            'type': signal_type,
            'strategy': 'volume_profile',
            'signal_source': 'lvn_breakout',
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': df['time'].iloc[-1] if 'time' in df.columns else datetime.utcnow(),
            'metadata': {
                'lvn_price': nearest_lvn.price_center,
                'lvn_strength': nearest_lvn.strength,
                'momentum': 'upward' if price_momentum > 0 else 'downward',
            }
        }

    def _calculate_stop_loss(self, signal_type: str, entry_price: float,
                            profile: VolumeProfile, df: pd.DataFrame) -> float:
        """Calculate stop loss based on HVN levels or ATR"""

        method = self.stop_loss_config['method']

        if method == 'hvn_based':
            # Place stop beyond nearest HVN in opposite direction
            hvn_above, hvn_below = self.vp_analyzer.get_hvn_above_below(entry_price, profile)
            buffer = self.stop_loss_config['hvn_buffer_pips'] * self.pip_value

            if signal_type == 'BUY':
                # Stop below nearest HVN support
                if hvn_below:
                    stop = hvn_below.price_low - buffer
                else:
                    # Fallback to VAL
                    stop = profile.val - buffer
            else:  # SELL
                # Stop above nearest HVN resistance
                if hvn_above:
                    stop = hvn_above.price_high + buffer
                else:
                    # Fallback to VAH
                    stop = profile.vah + buffer

        else:  # atr_based or hybrid
            # Calculate ATR
            atr = self._calculate_atr(df, period=14)
            atr_multiplier = self.stop_loss_config['atr_multiplier']

            if signal_type == 'BUY':
                stop = entry_price - (atr * atr_multiplier)
            else:
                stop = entry_price + (atr * atr_multiplier)

        # Apply min/max constraints
        stop_distance = abs(entry_price - stop)
        min_stop = self.stop_loss_config['min_stop_pips'] * self.pip_value
        max_stop = self.stop_loss_config['max_stop_pips'] * self.pip_value

        if stop_distance < min_stop:
            if signal_type == 'BUY':
                stop = entry_price - min_stop
            else:
                stop = entry_price + min_stop

        if stop_distance > max_stop:
            if signal_type == 'BUY':
                stop = entry_price - max_stop
            else:
                stop = entry_price + max_stop

        return stop

    def _calculate_take_profit(self, signal_type: str, entry_price: float,
                              profile: VolumeProfile, df: pd.DataFrame) -> float:
        """Calculate take profit based on HVN levels or ATR"""

        method = self.take_profit_config['method']

        if method == 'hvn_based' or method == 'poc_based':
            # Target next HVN or POC
            hvn_above, hvn_below = self.vp_analyzer.get_hvn_above_below(entry_price, profile)

            if signal_type == 'BUY':
                # Target HVN above or POC
                if self.take_profit_config['target_next_hvn'] and hvn_above:
                    target = hvn_above.price_center
                elif self.take_profit_config['target_poc'] and profile.poc > entry_price:
                    target = profile.poc
                elif self.take_profit_config['target_vah_val']:
                    target = profile.vah
                else:
                    # Fallback to ATR
                    atr = self._calculate_atr(df, period=14)
                    target = entry_price + (atr * self.take_profit_config['atr_multiplier'])
            else:  # SELL
                # Target HVN below or POC
                if self.take_profit_config['target_next_hvn'] and hvn_below:
                    target = hvn_below.price_center
                elif self.take_profit_config['target_poc'] and profile.poc < entry_price:
                    target = profile.poc
                elif self.take_profit_config['target_vah_val']:
                    target = profile.val
                else:
                    # Fallback to ATR
                    atr = self._calculate_atr(df, period=14)
                    target = entry_price - (atr * self.take_profit_config['atr_multiplier'])

        else:  # atr_based
            atr = self._calculate_atr(df, period=14)
            atr_multiplier = self.take_profit_config['atr_multiplier']

            if signal_type == 'BUY':
                target = entry_price + (atr * atr_multiplier)
            else:
                target = entry_price - (atr * atr_multiplier)

        return target

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]

        return float(atr) if not pd.isna(atr) else 0.0001

    def _validate_risk_reward(self, entry_price: float, stop_loss: float,
                             take_profit: float, signal_type: str) -> bool:
        """Validate risk:reward ratio meets minimum requirements"""

        if signal_type == 'BUY':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        if risk <= 0:
            return False

        rr_ratio = reward / risk
        min_rr = self.take_profit_config.get('min_reward_to_risk', 1.5)

        return rr_ratio >= min_rr

    def get_required_columns(self) -> List[str]:
        """Get required DataFrame columns"""
        return ['time', 'open', 'high', 'low', 'close', 'volume']

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators for Volume Profile strategy"""
        # Volume Profile only needs OHLCV data, no additional indicators required
        # It calculates its own volume profile metrics
        return []

    def get_min_bars(self) -> int:
        """Get minimum number of bars required"""
        return self.lookback_periods + 20
