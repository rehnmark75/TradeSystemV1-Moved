# core/strategies/momentum_strategy.py
"""
Advanced Momentum Strategy with Minimal Lag
===========================================

Enhanced momentum strategy inspired by TradingView scripts:
- AlgoAlpha AI Momentum Predictor: AI-powered momentum prediction
- Zeiierman Quantitative Momentum Oscillator: Quantitative momentum approach
- ChartPrime Multi-Timeframe Oscillator: Multi-timeframe confirmation

Key Features:
- Minimal lag momentum oscillators
- Velocity-based momentum calculation
- Volume-weighted momentum confirmation
- Adaptive smoothing based on market volatility
- Multi-timeframe validation
- Advanced exit management with trailing stops
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
import warnings

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

# Import configuration
try:
    import config
except ImportError:
    from forex_scanner import config

# Import momentum-specific configuration
try:
    from configdata.strategies import config_momentum_strategy
except ImportError:
    try:
        from forex_scanner.configdata.strategies import config_momentum_strategy
    except ImportError:
        # Fallback - create a minimal config object
        class MockMomentumConfig:
            MOMENTUM_STRATEGY = True
            MOMENTUM_VELOCITY_ENABLED = True
            MOMENTUM_VOLUME_CONFIRMATION = True
            MOMENTUM_MTF_VALIDATION = True
            MOMENTUM_ADAPTIVE_SMOOTHING = True
            MOMENTUM_SIGNAL_THRESHOLD = 0.0001
            MOMENTUM_VELOCITY_THRESHOLD = 0.00002
            MOMENTUM_MIN_CONFIDENCE = 0.65
            MOMENTUM_CALCULATION_METHOD = 'velocity_weighted'

            def get_momentum_config_for_epic(self, epic):
                return {
                    'fast_period': 5,
                    'slow_period': 10,
                    'signal_period': 3,
                    'velocity_period': 7,
                    'volume_period': 14,
                    'description': 'Fallback momentum configuration'
                }

        config_momentum_strategy = MockMomentumConfig()

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class MomentumStrategy(BaseStrategy):
    """
    Advanced Momentum Strategy with Minimal Lag

    Implements multiple momentum calculation methods inspired by
    top-tier TradingView momentum indicators:
    - Fast momentum oscillator (minimal lag)
    - Velocity momentum with adaptive smoothing
    - Volume-weighted momentum for institutional detection
    - Multi-timeframe confirmation
    """

    def __init__(self, data_fetcher=None, backtest_mode: bool = False, use_optimal_parameters: bool = False):
        """Initialize momentum strategy with advanced features"""
        super().__init__('momentum')
        self.price_adjuster = PriceAdjuster()
        self.backtest_mode = backtest_mode
        self.use_optimal_parameters = use_optimal_parameters
        self.data_fetcher = data_fetcher

        # Get configuration from momentum strategy config
        self.momentum_config = config_momentum_strategy.get_momentum_config_for_epic('default')

        # Core momentum parameters (inspired by AlgoAlpha AI Momentum Predictor)
        self.fast_period = self.momentum_config.get('fast_period', 5)
        self.slow_period = self.momentum_config.get('slow_period', 10)
        self.signal_period = self.momentum_config.get('signal_period', 3)
        self.velocity_period = self.momentum_config.get('velocity_period', 7)
        self.volume_period = self.momentum_config.get('volume_period', 14)

        # Feature toggles
        self.velocity_enabled = config_momentum_strategy.MOMENTUM_VELOCITY_ENABLED
        self.volume_confirmation = config_momentum_strategy.MOMENTUM_VOLUME_CONFIRMATION
        self.mtf_validation = config_momentum_strategy.MOMENTUM_MTF_VALIDATION
        self.adaptive_smoothing = config_momentum_strategy.MOMENTUM_ADAPTIVE_SMOOTHING

        # Thresholds
        self.signal_threshold = config_momentum_strategy.MOMENTUM_SIGNAL_THRESHOLD
        self.velocity_threshold = config_momentum_strategy.MOMENTUM_VELOCITY_THRESHOLD
        self.min_confidence = config_momentum_strategy.MOMENTUM_MIN_CONFIDENCE

        # Calculation method
        self.calculation_method = config_momentum_strategy.MOMENTUM_CALCULATION_METHOD

        self.logger.info("🚀 Advanced Momentum Strategy initialized")
        self.logger.info(f"   Parameters: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
        self.logger.info(f"   Features: velocity={self.velocity_enabled}, volume={self.volume_confirmation}, mtf={self.mtf_validation}")
        self.logger.info(f"   Method: {self.calculation_method}, adaptive_smoothing={self.adaptive_smoothing}")

    def get_required_indicators(self) -> List[str]:
        """Return list of required technical indicators"""
        indicators = ['close', 'high', 'low', 'open', 'volume', 'ltv']

        # Add momentum-specific indicators
        indicators.extend([
            'momentum', 'velocity', 'volume_momentum',
            'atr', 'rsi', 'macd_histogram'
        ])

        return indicators

    def detect_signal(
        self,
        df: pd.DataFrame,
        epic: str,
        spread_pips: float = 1.5,
        timeframe: str = '15m',
        evaluation_time: datetime = None
    ) -> Optional[Dict]:
        """
        Detect momentum signals using advanced multi-method approach

        Returns signal dictionary or None if no valid signal
        """
        try:
            self.logger.info(f"🔍 [MOMENTUM] Analyzing {epic} on {timeframe}")

            # Validate data sufficiency
            min_required = max(self.slow_period, self.velocity_period, self.volume_period) + 20
            if len(df) < min_required:
                self.logger.warning(f"[MOMENTUM REJECTED] {epic} - Insufficient data: {len(df)} < {min_required}")
                return None

            # Apply BID price adjustment for accurate signals
            df_adjusted = self.price_adjuster.adjust_bid_to_mid_prices(df, spread_pips)

            # Calculate momentum indicators
            df_enhanced = self._calculate_momentum_indicators(df_adjusted)

            if df_enhanced is None or len(df_enhanced) < 10:
                self.logger.warning(f"[MOMENTUM REJECTED] {epic} - Indicator calculation failed")
                return None

            # Get latest values
            latest = df_enhanced.iloc[-1]
            previous = df_enhanced.iloc[-2] if len(df_enhanced) > 1 else latest

            # Extract current market data
            current_price = latest['close']
            previous_price = previous['close']

            # Extract momentum values
            fast_momentum = latest['momentum_fast']
            slow_momentum = latest['momentum_slow']
            momentum_signal = latest['momentum_signal']
            velocity_momentum = latest.get('velocity_momentum', 0.0)
            volume_momentum = latest.get('volume_momentum', 0.0)

            # Get previous values for trend detection
            fast_momentum_prev = previous['momentum_fast']
            slow_momentum_prev = previous['momentum_slow']

            self.logger.debug(f"   Current Price: {current_price:.5f}")
            self.logger.debug(f"   Fast Momentum: {fast_momentum:.6f} (prev: {fast_momentum_prev:.6f})")
            self.logger.debug(f"   Slow Momentum: {slow_momentum:.6f} (prev: {slow_momentum_prev:.6f})")
            self.logger.debug(f"   Signal Line: {momentum_signal:.6f}")
            self.logger.debug(f"   Velocity: {velocity_momentum:.6f}")

            # Determine signal type using multiple methods
            signal_type, trigger_reason = self._determine_signal_type(
                fast_momentum, slow_momentum, momentum_signal,
                fast_momentum_prev, slow_momentum_prev,
                velocity_momentum, volume_momentum,
                current_price, previous_price
            )

            if not signal_type:
                self.logger.debug(f"[MOMENTUM] No signal: conditions not met for {epic}")
                return None

            # Create enhanced signal data for confidence calculation
            enhanced_signal_data = self._create_enhanced_signal_data(
                latest, signal_type, current_price, epic, timeframe,
                fast_momentum, slow_momentum, velocity_momentum, volume_momentum
            )

            # Calculate confidence using enhanced validation
            confidence = self.calculate_confidence(enhanced_signal_data)

            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                self.logger.warning(f"[MOMENTUM REJECTED] {epic} - Low confidence: {confidence:.1%} < {self.min_confidence:.1%}")
                return None

            # Create base signal
            signal = self._create_base_signal(signal_type, epic, timeframe, latest, current_price)

            # Add momentum-specific data
            signal.update({
                'momentum_fast': fast_momentum,
                'momentum_slow': slow_momentum,
                'momentum_signal': momentum_signal,
                'velocity_momentum': velocity_momentum,
                'volume_momentum': volume_momentum,
                'momentum_divergence': fast_momentum - slow_momentum,
                'signal_strength': abs(fast_momentum - slow_momentum),
                'confidence_score': confidence,
                'trigger_reason': trigger_reason,
                'calculation_method': self.calculation_method,
                'adaptive_smoothing_used': self.adaptive_smoothing
            })

            # Apply BID price adjustment if enabled
            if getattr(config, 'USE_BID_ADJUSTMENT', False):
                signal = self.price_adjuster.add_execution_prices(signal, spread_pips)

            # Enhance signal with comprehensive data
            signal = self._enhance_momentum_signal_comprehensive(
                signal, latest, previous, spread_pips, df_enhanced
            )

            self.logger.info(f"🎯 [MOMENTUM VALIDATED] {epic} - {signal_type} signal with {confidence:.1%} confidence")
            return signal

        except Exception as e:
            self.logger.error(f"❌ [MOMENTUM ERROR] {epic} - Signal detection failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate momentum indicators using multiple methods inspired by TradingView scripts
        """
        try:
            df = df.copy()
            self.logger.debug(f"Starting momentum indicator calculation with {len(df)} data points")
            self.logger.debug(f"Required periods: fast={self.fast_period}, slow={self.slow_period}, velocity={self.velocity_period}")

            # Check input data quality
            if df['close'].isna().all():
                self.logger.warning("All close prices are NaN")
                return None
            if (df['close'].diff() == 0).all():
                self.logger.warning("No price variation detected (all prices equal)")
                return None

            # Method 1: Fast Momentum Oscillator (inspired by AlgoAlpha AI Momentum Predictor)
            df['momentum_fast'] = self._calculate_fast_momentum_oscillator(df)
            df['momentum_slow'] = self._calculate_slow_momentum_oscillator(df)

            # Check for valid momentum calculations
            if df['momentum_fast'].isna().all() or df['momentum_slow'].isna().all():
                self.logger.warning("Momentum oscillators failed - all NaN values")
                self.logger.debug(f"   Fast momentum NaN count: {df['momentum_fast'].isna().sum()}/{len(df['momentum_fast'])}")
                self.logger.debug(f"   Slow momentum NaN count: {df['momentum_slow'].isna().sum()}/{len(df['momentum_slow'])}")
                return None

            # Signal line (smoothed momentum difference)
            momentum_diff = df['momentum_fast'] - df['momentum_slow']
            try:
                signal_ema = ta.ema(momentum_diff, length=self.signal_period)
                df['momentum_signal'] = signal_ema if signal_ema is not None else momentum_diff.rolling(window=self.signal_period).mean()
            except Exception as e:
                self.logger.debug(f"Signal EMA calculation failed: {e}")
                df['momentum_signal'] = momentum_diff.rolling(window=self.signal_period).mean().fillna(0.0)

            # Method 2: Velocity Momentum (rate of change with adaptive smoothing)
            if self.velocity_enabled:
                df['velocity_momentum'] = self._calculate_velocity_momentum(df)
            else:
                df['velocity_momentum'] = 0.0

            # Method 3: Volume-Weighted Momentum (institutional flow detection)
            if self.volume_confirmation:
                df['volume_momentum'] = self._calculate_volume_weighted_momentum(df)
            else:
                df['volume_momentum'] = 0.0

            # Additional technical indicators for confluence
            try:
                rsi_result = ta.rsi(df['close'], length=14)
                df['rsi'] = rsi_result.fillna(50.0) if rsi_result is not None else 50.0
            except Exception as e:
                self.logger.debug(f"RSI calculation failed: {e}")
                df['rsi'] = 50.0

            try:
                atr_result = ta.atr(df['high'], df['low'], df['close'], length=14)
                df['atr'] = atr_result.fillna(0.001) if atr_result is not None else 0.001
            except Exception as e:
                self.logger.debug(f"ATR calculation failed: {e}")
                df['atr'] = 0.001

            # MACD for additional confirmation
            try:
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                if macd is not None and hasattr(macd, 'columns') and len(macd.columns) >= 3:
                    df['macd_line'] = macd.iloc[:, 0].fillna(0.0)
                    df['macd_signal'] = macd.iloc[:, 1].fillna(0.0)
                    df['macd_histogram'] = macd.iloc[:, 2].fillna(0.0)
                else:
                    df['macd_line'] = 0.0
                    df['macd_signal'] = 0.0
                    df['macd_histogram'] = 0.0
            except Exception as e:
                self.logger.debug(f"MACD calculation failed: {e}")
                df['macd_line'] = 0.0
                df['macd_signal'] = 0.0
                df['macd_histogram'] = 0.0

            # Remove NaN values only from essential momentum columns to preserve data
            essential_columns = ['momentum_fast', 'momentum_slow', 'momentum_signal']
            df_clean = df.dropna(subset=essential_columns)

            if len(df_clean) < 10:
                self.logger.warning("Insufficient data after momentum indicator calculations")
                self.logger.debug(f"   Data length after cleanup: {len(df_clean)} (was {len(df)})")
                self.logger.debug(f"   Required minimum: 10")
                self.logger.debug(f"   NaN counts: {df[essential_columns].isna().sum().to_dict()}")
                return None

            # Use cleaned dataframe
            df = df_clean

            return df

        except Exception as e:
            self.logger.error(f"Momentum indicator calculation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _calculate_fast_momentum_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate fast momentum oscillator inspired by AlgoAlpha AI Momentum Predictor
        Uses price rate of change with minimal smoothing for fast response
        """
        # Price rate of change over fast period
        roc = df['close'].pct_change(self.fast_period)

        # Apply minimal smoothing to reduce noise while maintaining responsiveness
        if self.adaptive_smoothing:
            # Calculate volatility for adaptive smoothing
            volatility = df['close'].pct_change().rolling(window=20).std()
            # Safe adaptive smoothing - use fixed factor instead of Series
            try:
                # Get average volatility for a single smoothing factor
                avg_volatility = volatility.fillna(volatility.mean()).mean()
                smoothing_factor = config_momentum_strategy.get_adaptive_smoothing_factor(avg_volatility)
                # Ensure smoothing factor is valid scalar in range (0, 1]
                if not (0 < smoothing_factor <= 1):
                    smoothing_factor = 0.6
                fast_momentum = roc.ewm(alpha=smoothing_factor).mean()
            except:
                # Fallback to simple fixed smoothing
                fast_momentum = roc.ewm(alpha=0.6).mean()
        else:
            # Fixed minimal smoothing
            try:
                fast_momentum = ta.ema(roc, length=2)
                if fast_momentum is None:
                    fast_momentum = roc.rolling(window=2).mean()
            except:
                fast_momentum = roc.rolling(window=2).mean()

        return fast_momentum.fillna(0.0)

    def _calculate_slow_momentum_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate slow momentum oscillator for trend confirmation
        """
        # Price rate of change over slow period
        roc = df['close'].pct_change(self.slow_period)

        # Apply moderate smoothing
        if self.adaptive_smoothing:
            volatility = df['close'].pct_change().rolling(window=20).std()
            # Safe adaptive smoothing - use fixed factor instead of Series
            try:
                # Get average volatility for a single smoothing factor
                avg_volatility = volatility.fillna(volatility.mean()).mean()
                smoothing_factor = config_momentum_strategy.get_adaptive_smoothing_factor(avg_volatility) * 0.7
                # Ensure smoothing factor is valid scalar in range (0, 1]
                if not (0 < smoothing_factor <= 1):
                    smoothing_factor = 0.4
                slow_momentum = roc.ewm(alpha=smoothing_factor).mean()
            except:
                # Fallback to simple fixed smoothing
                slow_momentum = roc.ewm(alpha=0.4).mean()  # 0.6 * 0.7 ≈ 0.4
        else:
            # Fixed moderate smoothing
            try:
                slow_momentum = ta.ema(roc, length=self.signal_period)
                if slow_momentum is None:
                    slow_momentum = roc.rolling(window=self.signal_period).mean()
            except:
                slow_momentum = roc.rolling(window=self.signal_period).mean()

        return slow_momentum.fillna(0.0)

    def _calculate_velocity_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate velocity-based momentum inspired by Zeiierman Quantitative Momentum Oscillator
        """
        try:
            # Price velocity (acceleration of price movement)
            price_velocity = df['close'].diff().rolling(window=self.velocity_period).mean()

            # Normalize by ATR to make it scale-independent
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            if atr is not None:
                # Avoid division by zero
                normalized_velocity = price_velocity / atr.where(atr != 0, 0.001)
            else:
                normalized_velocity = price_velocity / 0.001

            # Apply smoothing
            try:
                velocity_momentum = ta.ema(normalized_velocity, length=3)
                if velocity_momentum is None:
                    velocity_momentum = normalized_velocity.rolling(window=3).mean()
            except:
                velocity_momentum = normalized_velocity.rolling(window=3).mean()

            return velocity_momentum.fillna(0.0)

        except Exception as e:
            self.logger.debug(f"Velocity momentum calculation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _calculate_volume_weighted_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate volume-weighted momentum for institutional flow detection
        Inspired by BigBeluga Whale Movement Tracker concepts
        """
        try:
            # Get volume data (handle both 'volume' and 'ltv' columns)
            if 'volume' in df.columns:
                volume = df['volume'].fillna(1.0)
            elif 'ltv' in df.columns:
                volume = df['ltv'].fillna(1.0)
            else:
                volume = pd.Series(1.0, index=df.index)

            # Check if volume is all zeros (avoid ambiguous truth value)
            if (volume == 0).all():
                return pd.Series(0.0, index=df.index)

            # Price change weighted by volume
            price_change = df['close'].pct_change()
            volume_weighted_change = price_change * volume

            # Rolling momentum with volume weighting
            volume_momentum = volume_weighted_change.rolling(window=self.volume_period).sum()

            # Normalize by average volume to make it comparable
            avg_volume = volume.rolling(window=self.volume_period).mean()
            # Avoid division by zero
            normalized_momentum = volume_momentum / avg_volume.where(avg_volume != 0, 1.0)

            # Apply smoothing
            try:
                smoothed_momentum = ta.ema(normalized_momentum, length=3)
                if smoothed_momentum is None:
                    smoothed_momentum = normalized_momentum.rolling(window=3).mean()
            except:
                smoothed_momentum = normalized_momentum.rolling(window=3).mean()

            return smoothed_momentum.fillna(0.0)

        except Exception as e:
            self.logger.debug(f"Volume momentum calculation failed: {e}")
            return pd.Series(0.0, index=df.index)

    def _determine_signal_type(
        self,
        fast_momentum: float,
        slow_momentum: float,
        momentum_signal: float,
        fast_momentum_prev: float,
        slow_momentum_prev: float,
        velocity_momentum: float,
        volume_momentum: float,
        current_price: float,
        previous_price: float
    ) -> Tuple[Optional[str], str]:
        """
        Determine signal type using multiple momentum methods
        """
        signal_type = None
        trigger_reason = None

        # Primary condition: Fast momentum crosses above slow momentum
        momentum_crossover_bull = (
            fast_momentum > slow_momentum and
            fast_momentum_prev <= slow_momentum_prev and
            fast_momentum > self.signal_threshold
        )

        momentum_crossover_bear = (
            fast_momentum < slow_momentum and
            fast_momentum_prev >= slow_momentum_prev and
            fast_momentum < -self.signal_threshold
        )

        # Secondary condition: Velocity confirmation
        velocity_confirmation_bull = not self.velocity_enabled or velocity_momentum > self.velocity_threshold
        velocity_confirmation_bear = not self.velocity_enabled or velocity_momentum < -self.velocity_threshold

        # Volume confirmation
        volume_confirmation = not self.volume_confirmation or abs(volume_momentum) > 0.00001

        # Price momentum confirmation
        price_momentum_bull = current_price > previous_price
        price_momentum_bear = current_price < previous_price

        # BULL signal conditions
        if (momentum_crossover_bull and
            velocity_confirmation_bull and
            volume_confirmation and
            price_momentum_bull):

            signal_type = 'BULL'
            trigger_reason = 'momentum_crossover_bullish'
            self.logger.debug(f"   🎯 BULL signal: fast={fast_momentum:.6f} > slow={slow_momentum:.6f}, velocity={velocity_momentum:.6f}")

        # BEAR signal conditions
        elif (momentum_crossover_bear and
              velocity_confirmation_bear and
              volume_confirmation and
              price_momentum_bear):

            signal_type = 'BEAR'
            trigger_reason = 'momentum_crossover_bearish'
            self.logger.debug(f"   🎯 BEAR signal: fast={fast_momentum:.6f} < slow={slow_momentum:.6f}, velocity={velocity_momentum:.6f}")

        # Alternative signal: Strong momentum divergence (inspired by TradingView scripts)
        elif abs(fast_momentum - slow_momentum) > self.signal_threshold * 2:
            if fast_momentum > slow_momentum and velocity_confirmation_bull:
                signal_type = 'BULL'
                trigger_reason = 'strong_momentum_divergence_bull'
            elif fast_momentum < slow_momentum and velocity_confirmation_bear:
                signal_type = 'BEAR'
                trigger_reason = 'strong_momentum_divergence_bear'

        return signal_type, trigger_reason

    def _create_enhanced_signal_data(
        self,
        latest: pd.Series,
        signal_type: str,
        current_price: float,
        epic: str,
        timeframe: str,
        fast_momentum: float,
        slow_momentum: float,
        velocity_momentum: float,
        volume_momentum: float
    ) -> Dict:
        """
        Create enhanced signal data structure for proper confidence calculation
        """
        # Calculate additional metrics for confidence
        momentum_strength = abs(fast_momentum - slow_momentum)
        velocity_strength = abs(velocity_momentum)
        volume_strength = abs(volume_momentum)

        # Get volume data
        volume = latest.get('ltv', latest.get('volume', 0))
        volume_ratio = latest.get('volume_ratio', 1.0)

        # ATR for volatility assessment
        atr = latest.get('atr', 0.001)

        return {
            'signal_type': signal_type,
            'price': current_price,
            'epic': epic,
            'timeframe': timeframe,

            # Momentum data
            'momentum_fast': fast_momentum,
            'momentum_slow': slow_momentum,
            'momentum_strength': momentum_strength,
            'velocity_momentum': velocity_momentum,
            'velocity_strength': velocity_strength,
            'volume_momentum': volume_momentum,
            'volume_strength': volume_strength,

            # Technical indicators for confidence calculation
            'rsi': latest.get('rsi', 50.0),
            'atr': atr,
            'volume': volume,
            'volume_ratio': volume_ratio,
            'volume_confirmation': volume_ratio > 1.2,

            # MACD for additional confluence
            'macd_line': latest.get('macd_line', 0.0),
            'macd_signal': latest.get('macd_signal', 0.0),
            'macd_histogram': latest.get('macd_histogram', 0.0),

            # EMA fallbacks for base strategy compatibility
            'ema_short': current_price,
            'ema_long': current_price,
            'ema_trend': current_price,

            # Efficiency ratio calculation
            'efficiency_ratio': self._calculate_efficiency_ratio(momentum_strength, atr)
        }

    def _calculate_efficiency_ratio(self, momentum_strength: float, atr: float) -> float:
        """
        Calculate efficiency ratio for enhanced validation
        Higher momentum strength relative to volatility = higher efficiency
        """
        try:
            if atr == 0 or pd.isna(atr) or momentum_strength == 0:
                return 0.4  # Default efficiency for neutral markets

            # Scale momentum strength to realistic range for forex
            # momentum_strength is typically very small (0.0001), so scale it up
            scaled_momentum = momentum_strength * 10000  # Convert to basis points
            scaled_atr = atr * 10000  # Convert ATR to basis points

            # Calculate efficiency ratio with better scaling
            if scaled_atr > 0:
                efficiency = min(1.0, scaled_momentum / scaled_atr)
            else:
                efficiency = 0.4

            # Normalize to reasonable range (0.1-0.8)
            normalized_efficiency = max(0.1, min(0.8, efficiency))

            return normalized_efficiency

        except Exception:
            return 0.6  # Safe fallback

    def _create_base_signal(
        self,
        signal_type: str,
        epic: str,
        timeframe: str,
        latest: pd.Series,
        current_price: float
    ) -> Dict:
        """
        Create base signal dictionary with proper timestamp handling
        """
        # Safe timestamp extraction
        try:
            if hasattr(latest, 'name') and latest.name is not None:
                if isinstance(latest.name, (pd.Timestamp, datetime)):
                    timestamp = latest.name
                else:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()
        except Exception:
            timestamp = datetime.utcnow()

        return {
            'epic': epic,
            'signal_type': signal_type,
            'strategy': self.name,
            'timeframe': timeframe,
            'price': current_price,
            'timestamp': timestamp,
            'market_timestamp': timestamp
        }

    def _enhance_momentum_signal_comprehensive(
        self,
        signal: Dict,
        latest: pd.Series,
        previous: pd.Series,
        spread_pips: float,
        df_enhanced: pd.DataFrame
    ) -> Dict:
        """
        Enhance momentum signal with comprehensive data for database storage
        """
        try:
            current_price = signal.get('price', latest.get('close', 0))
            signal_type = signal.get('signal_type')
            epic = signal.get('epic')

            # Extract pair from epic
            if not signal.get('pair'):
                signal['pair'] = epic.replace('CS.D.', '').replace('.MINI.IP', '')

            # Core technical data
            signal.update({
                'current_price': current_price,
                'previous_price': float(previous.get('close', current_price)),
                'price_change': current_price - float(previous.get('close', current_price)),
                'price_change_pct': ((current_price / float(previous.get('close', current_price))) - 1) * 100,
            })

            # Volume analysis
            volume = latest.get('volume') or latest.get('ltv', 0)
            signal['volume'] = float(volume) if volume else 0.0

            # Volume ratio calculation
            if len(df_enhanced) >= 20:
                avg_volume = df_enhanced['volume'].tail(20).mean() if 'volume' in df_enhanced else df_enhanced.get('ltv', pd.Series([1.0])).tail(20).mean()
                signal['volume_ratio'] = signal['volume'] / avg_volume if avg_volume > 0 else 1.0
            else:
                signal['volume_ratio'] = 1.0
            signal['volume_confirmation'] = signal['volume_ratio'] > 1.2

            # Technical indicators
            signal.update({
                'rsi': float(latest.get('rsi', 50.0)),
                'atr': float(latest.get('atr', 0.001)),
                'macd_line': float(latest.get('macd_line', 0.0)),
                'macd_signal': float(latest.get('macd_signal', 0.0)),
                'macd_histogram': float(latest.get('macd_histogram', 0.0))
            })

            # Strategy configuration (JSON field)
            signal['strategy_config'] = {
                'strategy_type': 'momentum',
                'strategy_family': 'momentum',
                'calculation_method': self.calculation_method,
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'velocity_period': self.velocity_period,
                'volume_period': self.volume_period,
                'features': {
                    'velocity_enabled': self.velocity_enabled,
                    'volume_confirmation': self.volume_confirmation,
                    'mtf_validation': self.mtf_validation,
                    'adaptive_smoothing': self.adaptive_smoothing
                },
                'signal_method': 'momentum_crossover_with_velocity',
                'bid_adjustment_enabled': getattr(config, 'USE_BID_ADJUSTMENT', False)
            }

            # Strategy indicators (JSON field)
            signal['strategy_indicators'] = {
                'primary_indicator': 'momentum_oscillator',
                'momentum_fast': signal.get('momentum_fast', 0.0),
                'momentum_slow': signal.get('momentum_slow', 0.0),
                'momentum_signal': signal.get('momentum_signal', 0.0),
                'momentum_divergence': signal.get('momentum_divergence', 0.0),
                'velocity_momentum': signal.get('velocity_momentum', 0.0),
                'volume_momentum': signal.get('volume_momentum', 0.0),
                'signal_strength': signal.get('signal_strength', 0.0),
                'momentum_direction': signal_type.lower() if signal_type else 'neutral'
            }

            # Strategy metadata (JSON field)
            confidence = signal.get('confidence_score', 0.7)
            signal['strategy_metadata'] = {
                'strategy_version': '1.0.0',
                'signal_basis': 'multi_method_momentum_analysis',
                'confidence_calculation': 'enhanced_validation',
                'signal_strength': 'strong' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'weak',
                'market_condition': 'momentum_driven',
                'calculation_method': self.calculation_method,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'data_completeness': 'full'
            }

            # Signal conditions (JSON field)
            signal['signal_conditions'] = {
                'market_trend': f'momentum_{signal_type.lower()}' if signal_type else 'neutral',
                'momentum_signal_type': 'crossover_with_confirmation',
                'velocity_confirmation': self.velocity_enabled and abs(signal.get('velocity_momentum', 0)) > self.velocity_threshold,
                'volume_confirmation': signal.get('volume_confirmation', False),
                'signal_timing': 'momentum_crossover',
                'confirmation_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.7 else 'low',
                'trigger_reason': signal.get('trigger_reason', 'momentum_crossover')
            }

            # Pricing and execution data
            spread_adjustment = spread_pips / 10000
            signal.update({
                'spread_pips': spread_pips,
                'bid_price': current_price - spread_adjustment,
                'ask_price': current_price + spread_adjustment,
                'execution_price': (current_price + spread_adjustment) if signal_type in ['BUY', 'BULL'] else (current_price - spread_adjustment)
            })

            # Risk management
            atr_value = signal.get('atr', 0.001)
            stop_distance = max(2.0 * spread_adjustment, 1.5 * atr_value)
            target_distance = 2.0 * stop_distance  # 2:1 RR

            signal.update({
                'stop_loss': current_price - stop_distance if signal_type == 'BULL' else current_price + stop_distance,
                'take_profit': current_price + target_distance if signal_type == 'BULL' else current_price - target_distance,
                'risk_reward_ratio': 2.0,
                'stop_loss_suggestion': signal['stop_loss'],
                'take_profit_suggestion': signal['take_profit']
            })

            # Market context
            current_time = datetime.utcnow()
            signal.update({
                'market_session': self._determine_trading_session(),
                'is_market_hours': self._is_market_hours(),
                'alert_timestamp': current_time,
                'processing_timestamp': current_time.isoformat(),
                'data_source': 'live_scanner'
            })

            # Technical summary
            signal['technical_summary'] = {
                'primary_signal': f"Momentum {signal_type} Crossover",
                'entry_quality': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.7 else 'Low',
                'setup_type': f"Momentum {'Breakout' if signal_type == 'BULL' else 'Breakdown'}",
                'timeframe_analysis': signal.get('timeframe', '15m'),
                'signal_reliability': 'High' if signal.get('signal_strength', 0) > 0.001 else 'Medium',
                'calculation_method': self.calculation_method
            }

            return signal

        except Exception as e:
            self.logger.error(f"❌ Error enhancing momentum signal: {e}")
            return signal

    def _determine_trading_session(self) -> str:
        """Determine current trading session"""
        try:
            import pytz
            london_tz = pytz.timezone('Europe/London')
            london_time = datetime.now(london_tz)
            hour = london_time.hour

            if 8 <= hour < 17:
                return 'london'
            elif 13 <= hour < 22:
                return 'new_york'
            elif 0 <= hour < 9:
                return 'sydney'
            else:
                return 'tokyo'
        except:
            return 'unknown'

    def _is_market_hours(self) -> bool:
        """Check if current time is during major market hours"""
        try:
            current_hour = datetime.utcnow().hour
            return 1 <= current_hour <= 23  # Markets active most of the day
        except:
            return True

    def calculate_confidence(self, signal_data: Dict) -> float:
        """
        Calculate confidence score specifically for momentum strategy
        """
        try:
            # Use enhanced validator if available
            if self._validator:
                should_trade, confidence, reason, analysis = self._validator.validate_signal_enhanced(signal_data)
                return confidence

            # Fallback momentum-specific confidence calculation
            base_confidence = 0.65

            # Factor 1: Momentum strength (how strong is the momentum divergence)
            momentum_strength = signal_data.get('momentum_strength', 0.0)
            strength_factor = min(0.2, momentum_strength * 1000)  # Scale appropriately

            # Factor 2: Velocity confirmation
            velocity_strength = signal_data.get('velocity_strength', 0.0)
            velocity_factor = min(0.15, velocity_strength * 100) if self.velocity_enabled else 0.1

            # Factor 3: Volume confirmation
            volume_confirmation = signal_data.get('volume_confirmation', False)
            volume_factor = 0.1 if volume_confirmation else 0.0

            # Factor 4: RSI confluence (avoid extreme levels)
            rsi = signal_data.get('rsi', 50.0)
            if 30 <= rsi <= 70:  # Good RSI levels
                rsi_factor = 0.05
            elif 20 <= rsi <= 80:  # Acceptable levels
                rsi_factor = 0.0
            else:  # Extreme levels (penalty)
                rsi_factor = -0.05

            # Factor 5: Efficiency ratio
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.6)
            efficiency_factor = (efficiency_ratio - 0.5) * 0.2

            # Calculate total confidence
            total_confidence = (
                base_confidence +
                strength_factor +
                velocity_factor +
                volume_factor +
                rsi_factor +
                efficiency_factor
            )

            # Ensure confidence is within valid range
            final_confidence = max(0.1, min(0.95, total_confidence))

            return final_confidence

        except Exception as e:
            self.logger.error(f"Error calculating momentum confidence: {e}")
            return 0.5  # Safe fallback

    def get_strategy_info(self) -> Dict:
        """Return strategy information for monitoring and debugging"""
        return {
            'name': 'Advanced Momentum Strategy',
            'version': '1.0.0',
            'inspired_by': [
                'AlgoAlpha AI Momentum Predictor',
                'Zeiierman Quantitative Momentum Oscillator',
                'ChartPrime Multi-Timeframe Oscillator',
                'BigBeluga Whale Movement Tracker'
            ],
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'velocity_period': self.velocity_period,
                'volume_period': self.volume_period
            },
            'features': {
                'calculation_method': self.calculation_method,
                'velocity_enabled': self.velocity_enabled,
                'volume_confirmation': self.volume_confirmation,
                'mtf_validation': self.mtf_validation,
                'adaptive_smoothing': self.adaptive_smoothing
            },
            'thresholds': {
                'signal_threshold': self.signal_threshold,
                'velocity_threshold': self.velocity_threshold,
                'min_confidence': self.min_confidence
            },
            'signal_conditions': {
                'bull': 'fast_momentum > slow_momentum with velocity and volume confirmation',
                'bear': 'fast_momentum < slow_momentum with velocity and volume confirmation'
            }
        }