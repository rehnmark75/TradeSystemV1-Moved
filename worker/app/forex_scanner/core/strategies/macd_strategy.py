# core/strategies/macd_strategy.py
"""
CLEAN MACD STRATEGY - Rebuilt from scratch
Simple, reliable MACD crossover detection with swing proximity validation
Works identically in both live and backtest modes
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.swing_proximity_validator import SwingProximityValidator
from .helpers.smc_market_structure import SMCMarketStructure

try:
    from configdata import config
    from configdata.strategies import config_macd_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_macd_strategy
    except ImportError:
        config_macd_strategy = None


class MACDStrategy(BaseStrategy):
    """
    Clean MACD Strategy - Simple and Reliable

    Features:
    - Standard MACD (12, 26, 9) crossover detection
    - Swing proximity validation (don't buy near swing highs, don't sell near swing lows)
    - ADX filter (only trade in trending markets)
    - ONE signal per crossover event
    - Works identically in live and backtest modes
    """

    def __init__(self, data_fetcher=None, backtest_mode: bool = False, epic: str = None,
                 timeframe: str = '15m', **kwargs):
        # Initialize
        self.name = 'macd'
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.logger.setLevel(logging.INFO)

        # Basic config
        self.backtest_mode = backtest_mode
        self.epic = epic
        self.timeframe = timeframe
        self.data_fetcher = data_fetcher
        self.price_adjuster = PriceAdjuster()

        # MACD parameters (standard settings)
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9

        # Validation thresholds
        self.min_adx = 25  # Require trending markets (lowered from 30)
        self.min_confidence = 0.65  # 65% minimum confidence

        # Initialize swing validator
        swing_config = getattr(config_macd_strategy, 'MACD_SWING_VALIDATION', {}) if config_macd_strategy else {}
        if swing_config.get('enabled', True):
            smc_analyzer = SMCMarketStructure(logger=self.logger)
            self.swing_validator = SwingProximityValidator(
                smc_analyzer=smc_analyzer,
                config=swing_config,
                logger=self.logger
            )
        else:
            self.swing_validator = None

        self.logger.info(f"✅ Clean MACD Strategy initialized - ADX >= {self.min_adx}, Swing validation: {self.swing_validator is not None}")

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        df = df.copy()

        # Calculate EMAs
        fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()

        # MACD line and signal line
        df['macd_line'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']

        return df

    def detect_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect MACD histogram crossovers (zero line crosses)
        Returns ONE signal per crossover event
        """
        df = df.copy()

        # Shift histogram to get previous value
        df['histogram_prev'] = df['macd_histogram'].shift(1)

        # Detect crossovers: current > 0 and previous <= 0 (or vice versa)
        df['bull_crossover'] = (df['macd_histogram'] > 0) & (df['histogram_prev'] <= 0)
        df['bear_crossover'] = (df['macd_histogram'] < 0) & (df['histogram_prev'] >= 0)

        return df

    def validate_signal(self, row: pd.Series, signal_type: str) -> bool:
        """
        Validate signal meets quality requirements

        Checks:
        1. ADX >= 30 (strong trend)
        2. Histogram in correct direction
        3. RSI in reasonable zone
        4. Dynamic EMA trend filter (ADX-based volatility adaptation)
        """
        # Check ADX
        adx = row.get('adx', 0)
        if adx < self.min_adx:
            self.logger.debug(f"❌ {signal_type} rejected: ADX {adx:.1f} < {self.min_adx}")
            return False

        # Check histogram direction
        histogram = row.get('macd_histogram', 0)
        if signal_type == 'BULL' and histogram <= 0:
            self.logger.debug(f"❌ BULL rejected: negative histogram {histogram:.6f}")
            return False
        if signal_type == 'BEAR' and histogram >= 0:
            self.logger.debug(f"❌ BEAR rejected: positive histogram {histogram:.6f}")
            return False

        # Check RSI (optional - just avoid extreme zones)
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi > 70:
            self.logger.debug(f"❌ BULL rejected: RSI {rsi:.1f} overbought")
            return False
        if signal_type == 'BEAR' and rsi < 30:
            self.logger.debug(f"❌ BEAR rejected: RSI {rsi:.1f} oversold")
            return False

        # DYNAMIC EMA FILTER: ADX-based volatility adaptation
        # Very strong trend (ADX >= 40) → EMA 50 (fast, catch explosive moves)
        # Strong trend (ADX 30-40) → EMA 100 (balanced, good confirmation)
        # Moderate trend (ADX 25-30) → EMA 200 (conservative, need clear direction)
        price = row.get('close', 0)
        ema_50 = row.get('ema_50', 0)
        ema_100 = row.get('ema_100', 0)
        ema_200 = row.get('ema_200', 0)

        # Select EMA based on trend strength (ADX)
        if adx >= 40:
            ema_filter = ema_50
            ema_name = "EMA50"
        elif adx >= 30:
            ema_filter = ema_100
            ema_name = "EMA100"
        else:  # ADX 25-30
            ema_filter = ema_200
            ema_name = "EMA200"

        # Apply dynamic EMA filter
        if ema_filter > 0 and price > 0:
            if signal_type == 'BULL' and price < ema_filter:
                self.logger.debug(f"❌ BULL rejected: price {price:.5f} below {ema_name} {ema_filter:.5f} (ADX={adx:.1f})")
                return False
            if signal_type == 'BEAR' and price > ema_filter:
                self.logger.debug(f"❌ BEAR rejected: price {price:.5f} above {ema_name} {ema_filter:.5f} (ADX={adx:.1f})")
                return False

            # Log successful filter pass
            self.logger.debug(f"✅ {signal_type} passed {ema_name} filter: price={price:.5f}, {ema_name}={ema_filter:.5f} (ADX={adx:.1f})")

        return True

    def calculate_confidence(self, row: pd.Series, signal_type: str) -> float:
        """
        Calculate signal confidence (simple, transparent)

        Base: 50%
        + ADX strength: up to +20%
        + Histogram strength: up to +15%
        + RSI confluence: up to +15%
        """
        confidence = 0.50  # Base

        # ADX boost
        adx = row.get('adx', 0)
        if adx >= 40:
            confidence += 0.20
        elif adx >= 35:
            confidence += 0.15
        elif adx >= 30:
            confidence += 0.10

        # Histogram strength boost
        histogram = abs(row.get('macd_histogram', 0))
        if histogram > 0.002:
            confidence += 0.15
        elif histogram > 0.001:
            confidence += 0.10
        elif histogram > 0.0005:
            confidence += 0.05

        # RSI confluence boost
        rsi = row.get('rsi', 50)
        if signal_type == 'BULL' and rsi < 40:
            confidence += 0.15
        elif signal_type == 'BULL' and rsi < 50:
            confidence += 0.10
        elif signal_type == 'BEAR' and rsi > 60:
            confidence += 0.15
        elif signal_type == 'BEAR' and rsi > 50:
            confidence += 0.10

        return min(0.95, confidence)

    def create_signal(self, row: pd.Series, signal_type: str, epic: str, timeframe: str) -> Optional[Dict]:
        """Create signal dictionary"""
        try:
            # Calculate confidence
            confidence = self.calculate_confidence(row, signal_type)

            if confidence < self.min_confidence:
                self.logger.debug(f"❌ {signal_type} rejected: confidence {confidence:.1%} < {self.min_confidence:.1%}")
                return None

            # Determine which EMA filter was used (for logging)
            adx = row.get('adx', 0)
            if adx >= 40:
                ema_filter_used = 'EMA50'
            elif adx >= 30:
                ema_filter_used = 'EMA100'
            else:
                ema_filter_used = 'EMA200'

            # Create signal
            signal = {
                'epic': epic,
                'direction': signal_type,
                'signal_type': signal_type,  # For validator compatibility
                'strategy': self.name,
                'timeframe': timeframe,
                'price': row.get('close', 0),
                'confidence': confidence,
                'confidence_score': confidence,  # For validator compatibility
                'timestamp': row.name if hasattr(row, 'name') else datetime.now(),

                # MACD data
                'macd_line': row.get('macd_line', 0),
                'macd_signal': row.get('macd_signal', 0),
                'macd_histogram': row.get('macd_histogram', 0),

                # Indicators
                'adx': row.get('adx', 0),
                'rsi': row.get('rsi', 50),

                # EMA trend data (for validator and logging)
                'ema_50': row.get('ema_50', 0),
                'ema_100': row.get('ema_100', 0),
                'ema_200': row.get('ema_200', 0),
                'ema_filter_used': ema_filter_used,  # Which EMA was used for filtering

                # Stop loss / Take profit (will be set later)
                'stop_distance': 20,
                'limit_distance': 30,
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None

    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float = 1.5,
                     timeframe: str = '15m', **kwargs) -> Optional[Dict]:
        """
        Main signal detection method
        Works identically in both live and backtest modes
        """
        try:
            if len(df) < self.slow_period + self.signal_period:
                self.logger.warning(f"Insufficient data: {len(df)} bars")
                return None

            # Calculate MACD
            df = self.calculate_macd(df)

            # Calculate other indicators if not present
            if 'adx' not in df.columns:
                df = self._calculate_adx(df)

            if 'rsi' not in df.columns:
                df = self._calculate_rsi(df)

            # Detect crossovers
            df = self.detect_crossover(df)

            # CRITICAL FIX: Only check the LATEST bar for crossover
            # The backtest engine calls us once per bar with growing data
            # The last bar is always the NEW bar to check
            # This prevents re-detecting the same crossover multiple times
            latest_bar = df.iloc[-1]

            # Check for BULL crossover on THIS bar only
            if latest_bar.get('bull_crossover', False):
                if self.validate_signal(latest_bar, 'BULL'):
                    signal = self.create_signal(latest_bar, 'BULL', epic, timeframe)
                    if signal:
                        # Swing proximity validation (uses full DataFrame for context)
                        if self.swing_validator:
                            validation = self.swing_validator.validate_entry_proximity(
                                df, latest_bar['close'], 'BULL', epic
                            )
                            if not validation['valid']:
                                self.logger.info(f"❌ BULL rejected: {validation.get('rejection_reason', 'swing proximity')}")
                                return None
                            # Apply confidence penalty if needed
                            signal['confidence'] -= validation.get('confidence_penalty', 0)

                        self.logger.info(f"✅ BULL signal: {latest_bar.name}, ADX={latest_bar['adx']:.1f}, confidence={signal['confidence']:.1%}")
                        return signal

            # Check for BEAR crossover on THIS bar only
            if latest_bar.get('bear_crossover', False):
                if self.validate_signal(latest_bar, 'BEAR'):
                    signal = self.create_signal(latest_bar, 'BEAR', epic, timeframe)
                    if signal:
                        # Swing proximity validation (uses full DataFrame for context)
                        if self.swing_validator:
                            validation = self.swing_validator.validate_entry_proximity(
                                df, latest_bar['close'], 'BEAR', epic
                            )
                            if not validation['valid']:
                                self.logger.info(f"❌ BEAR rejected: {validation.get('rejection_reason', 'swing proximity')}")
                                return None
                            # Apply confidence penalty if needed
                            signal['confidence'] -= validation.get('confidence_penalty', 0)

                        self.logger.info(f"✅ BEAR signal: {latest_bar.name}, ADX={latest_bar['adx']:.1f}, confidence={signal['confidence']:.1%}")
                        return signal

            return None

        except Exception as e:
            self.logger.error(f"Error detecting signal: {e}", exc_info=True)
            return None

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX indicator"""
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    # Required abstract methods from BaseStrategy
    def get_required_indicators(self):
        """Return list of required indicators"""
        return ['macd_line', 'macd_signal', 'macd_histogram', 'adx', 'rsi']

    # Compatibility methods for backtest system
    def detect_signal_auto(self, df: pd.DataFrame, epic: str = None, spread_pips: float = 0,
                          timeframe: str = '15m', **kwargs) -> Optional[Dict]:
        """Auto detection wrapper"""
        return self.detect_signal(df, epic, spread_pips, timeframe, **kwargs)

    def enable_forex_integration(self, epic):
        """Compatibility method"""
        pass
