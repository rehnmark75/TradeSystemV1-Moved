#!/usr/bin/env python3
"""
Ranging Market Strategy - Multi-Oscillator Confluence for Range-Bound Markets

VERSION: 3.0.0
DATE: 2026-02-01
STATUS: Active (backtest-only by default)

Strategy Architecture:
    - ADX Filter: Only trades when ADX < 20 (ranging conditions)
    - Multi-Oscillator Confluence: Squeeze Momentum + Wave Trend + RSI + RVI
    - Support/Resistance: Bounces from dynamic S/R zones
    - Mean Reversion: Targets opposite band

Target Performance:
    - Win Rate: 55%+
    - Profit Factor: 1.8+
    - Average R:R: 1.5:1

Market Regime:
    - Primary: Ranging (ADX < 20)
    - Avoid: Trending (ADX > 25)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import os

import psycopg2
import psycopg2.extras

# Strategy Registry - auto-registers on import
from .strategy_registry import register_strategy, StrategyInterface


# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class RangingMarketConfig:
    """
    Configuration for Ranging Market strategy.

    Loaded from database (strategy_config.ranging_market_global_config).
    Falls back to defaults if database unavailable.
    """

    # Strategy identification
    strategy_name: str = "RANGING_MARKET"
    version: str = "3.0.0"

    # ADX Filter (core condition)
    adx_max_threshold: float = 20.0
    adx_period: int = 14

    # Squeeze Momentum Settings
    squeeze_bb_length: int = 20
    squeeze_bb_mult: float = 2.0
    squeeze_kc_length: int = 20
    squeeze_kc_mult: float = 1.5
    squeeze_momentum_length: int = 12
    squeeze_signal_weight: float = 0.30

    # Wave Trend Settings
    wavetrend_channel_length: int = 9
    wavetrend_avg_length: int = 12
    wavetrend_ob_level1: int = 53
    wavetrend_ob_level2: int = 60
    wavetrend_os_level1: int = -53
    wavetrend_os_level2: int = -60
    wavetrend_signal_weight: float = 0.25

    # RSI Settings
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    rsi_signal_weight: float = 0.25

    # RVI Settings
    rvi_period: int = 10
    rvi_signal_period: int = 4
    rvi_signal_weight: float = 0.20

    # Signal Generation
    min_oscillator_agreement: int = 2
    min_combined_score: float = 0.55

    # Support/Resistance Bounce
    sr_bounce_required: bool = True
    sr_proximity_pips: float = 5.0

    # Timeframes
    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"

    # Risk Management
    fixed_stop_loss_pips: float = 12.0
    fixed_take_profit_pips: float = 18.0
    min_confidence: float = 0.50
    max_confidence: float = 0.85
    sl_buffer_pips: float = 2.0

    # Cooldown
    signal_cooldown_minutes: int = 45

    # Enabled pairs (empty = all pairs)
    enabled_pairs: List[str] = field(default_factory=list)

    @classmethod
    def from_database(cls, database_url: str = None) -> 'RangingMarketConfig':
        """Load configuration from database"""
        config = cls()

        if database_url is None:
            database_url = os.getenv(
                'STRATEGY_CONFIG_DATABASE_URL',
                'postgresql://postgres:postgres@postgres:5432/strategy_config'
            )

        try:
            conn = psycopg2.connect(database_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute("""
                SELECT * FROM ranging_market_global_config
                WHERE is_active = TRUE
                ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()

            if row:
                # Map database columns to config attributes
                field_mapping = {
                    'adx_max_threshold': 'adx_max_threshold',
                    'adx_period': 'adx_period',
                    'squeeze_bb_length': 'squeeze_bb_length',
                    'squeeze_bb_mult': 'squeeze_bb_mult',
                    'squeeze_kc_length': 'squeeze_kc_length',
                    'squeeze_kc_mult': 'squeeze_kc_mult',
                    'squeeze_momentum_length': 'squeeze_momentum_length',
                    'squeeze_signal_weight': 'squeeze_signal_weight',
                    'wavetrend_channel_length': 'wavetrend_channel_length',
                    'wavetrend_avg_length': 'wavetrend_avg_length',
                    'rsi_period': 'rsi_period',
                    'rsi_overbought': 'rsi_overbought',
                    'rsi_oversold': 'rsi_oversold',
                    'rsi_signal_weight': 'rsi_signal_weight',
                    'min_oscillator_agreement': 'min_oscillator_agreement',
                    'min_combined_score': 'min_combined_score',
                    'fixed_stop_loss_pips': 'fixed_stop_loss_pips',
                    'fixed_take_profit_pips': 'fixed_take_profit_pips',
                    'min_confidence': 'min_confidence',
                    'max_confidence': 'max_confidence',
                }

                for db_col, attr_name in field_mapping.items():
                    if db_col in row.keys() and row[db_col] is not None:
                        setattr(config, attr_name, row[db_col])

            cur.close()
            conn.close()

        except Exception as e:
            logging.warning(f"Could not load Ranging Market config from database: {e}")

        return config


# ==============================================================================
# CONFIG SERVICE (Singleton)
# ==============================================================================

class RangingMarketConfigService:
    """Singleton service for loading and caching Ranging Market configuration."""

    _instance: Optional['RangingMarketConfigService'] = None
    _config: Optional[RangingMarketConfig] = None
    _last_refresh: Optional[datetime] = None
    _cache_ttl_seconds: int = 300

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.logger = logging.getLogger(__name__)

    @classmethod
    def get_instance(cls) -> 'RangingMarketConfigService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> RangingMarketConfig:
        """Get configuration with caching"""
        now = datetime.now()

        if (self._config is None or
            self._last_refresh is None or
            (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = RangingMarketConfig.from_database()
            self._last_refresh = now
            self.logger.debug("Refreshed Ranging Market config from database")

        return self._config

    def refresh(self) -> RangingMarketConfig:
        """Force refresh configuration"""
        self._config = None
        return self.get_config()


def get_ranging_market_config() -> RangingMarketConfig:
    """Convenience function to get Ranging Market configuration"""
    return RangingMarketConfigService.get_instance().get_config()


# ==============================================================================
# STRATEGY CLASS
# ==============================================================================

@register_strategy('RANGING_MARKET')
class RangingMarketStrategy(StrategyInterface):
    """
    Ranging Market Strategy Implementation

    Multi-oscillator confluence strategy optimized for range-bound markets.
    Uses ADX filter to only trade when market is ranging (ADX < 20).
    """

    def __init__(self, config=None, logger=None, db_manager=None):
        """
        Initialize Ranging Market Strategy

        Args:
            config: Optional legacy config module (ignored)
            logger: Logger instance
            db_manager: Database manager (not used, we connect directly)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager

        # Load configuration from database
        self.config = get_ranging_market_config()

        # Cooldown tracking (per-pair)
        self._cooldowns: Dict[str, datetime] = {}

        # Rejection logging
        self._pending_rejections: List[Dict] = []

        self.logger.info(f"Ranging Market Strategy v{self.config.version} initialized")

    @property
    def strategy_name(self) -> str:
        """Unique name for this strategy"""
        return "RANGING_MARKET"

    def get_required_timeframes(self) -> List[str]:
        """Get list of timeframes this strategy requires."""
        return [
            self.config.confirmation_timeframe,
            self.config.primary_timeframe
        ]

    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        df_entry: pd.DataFrame = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Detect trading signal using Ranging Market strategy logic.

        Args:
            df_trigger: Primary timeframe data (15m)
            df_4h: Higher timeframe data (1H or 4H)
            epic: IG epic identifier
            pair: Currency pair name
            df_entry: Entry timeframe data (optional)
            **kwargs: Additional parameters

        Returns:
            Signal dict if detected, None otherwise
        """
        # Use df_trigger as primary data
        df = df_trigger if df_trigger is not None else df_entry

        if df is None or len(df) < 50:
            self.logger.debug(f"[RANGING_MARKET] Insufficient data for {epic}")
            return None

        # Check enabled pairs filter
        if self.config.enabled_pairs and epic not in self.config.enabled_pairs:
            return None

        # Check cooldown
        if not self._check_cooldown(epic):
            return None

        # Step 1: Check ADX - must be ranging (ADX < threshold)
        adx_value = self._get_adx(df)
        if adx_value is None or adx_value > self.config.adx_max_threshold:
            self._log_rejection(epic, "ADX_TOO_HIGH", adx_value)
            self.logger.debug(
                f"[RANGING_MARKET] ADX {adx_value:.1f} > {self.config.adx_max_threshold} - market not ranging"
            )
            return None

        # Step 2: Calculate oscillator signals
        oscillator_signals = self._calculate_oscillator_signals(df)

        # Step 3: Check for confluence
        confluence = self._check_confluence(oscillator_signals)
        if confluence is None:
            self._log_rejection(epic, "NO_CONFLUENCE", oscillator_signals)
            return None

        direction = confluence['direction']
        confidence = confluence['confidence']

        # Step 4: Validate S/R proximity (optional)
        if self.config.sr_bounce_required:
            sr_valid = self._validate_sr_bounce(df, direction)
            if not sr_valid:
                self._log_rejection(epic, "SR_VALIDATION_FAILED")
                return None

        # Step 5: Build signal
        latest = df.iloc[-1]
        entry_price = float(latest.get('close', 0))

        signal = self._build_signal(
            epic=epic,
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            sl_pips=self.config.fixed_stop_loss_pips,
            tp_pips=self.config.fixed_take_profit_pips,
            confidence=confidence,
            indicators={
                'adx': adx_value,
                'oscillators': oscillator_signals,
                'regime': 'ranging'
            }
        )

        self._set_cooldown(epic)

        self.logger.info(
            f"[RANGING_MARKET] ✅ Signal: {direction} {epic} "
            f"(ADX={adx_value:.1f}, conf={confidence:.2f})"
        )

        return signal

    # =========================================================================
    # INDICATOR CALCULATIONS
    # =========================================================================

    def _get_adx(self, df: pd.DataFrame) -> Optional[float]:
        """Get ADX value from dataframe or calculate it"""
        if 'adx' in df.columns:
            return float(df['adx'].iloc[-1])

        # Calculate ADX if not present
        try:
            period = self.config.adx_period
            high = df['high']
            low = df['low']
            close = df['close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            dm_plus = (high - high.shift(1)).clip(lower=0)
            dm_minus = (low.shift(1) - low).clip(lower=0)

            # Average True Range
            atr = tr.rolling(window=period).mean()

            # Directional Indicators
            di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

            # ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 0.0001)
            adx = dx.rolling(window=period).mean()

            return float(adx.iloc[-1])
        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {e}")
            return None

    def _calculate_oscillator_signals(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate signals from all oscillators"""
        signals = {}

        # Squeeze Momentum
        squeeze = self._calculate_squeeze_momentum(df)
        if squeeze:
            signals['squeeze'] = squeeze

        # RSI
        rsi = self._calculate_rsi_signal(df)
        if rsi:
            signals['rsi'] = rsi

        # Stochastic (simplified RVI alternative)
        stoch = self._calculate_stochastic_signal(df)
        if stoch:
            signals['stochastic'] = stoch

        return signals

    def _calculate_squeeze_momentum(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Squeeze Momentum indicator signal"""
        try:
            length = self.config.squeeze_bb_length
            bb_mult = self.config.squeeze_bb_mult
            kc_mult = self.config.squeeze_kc_mult

            close = df['close']
            high = df['high']
            low = df['low']

            # Bollinger Bands
            bb_basis = close.rolling(window=length).mean()
            bb_dev = close.rolling(window=length).std()
            bb_upper = bb_basis + (bb_mult * bb_dev)
            bb_lower = bb_basis - (bb_mult * bb_dev)

            # Keltner Channels
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=length).mean()
            kc_upper = bb_basis + (kc_mult * atr)
            kc_lower = bb_basis - (kc_mult * atr)

            # Squeeze detection (BB inside KC)
            squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

            # Momentum (Linear Regression)
            momentum_length = self.config.squeeze_momentum_length
            x = np.arange(momentum_length)
            momentum = close.rolling(window=momentum_length).apply(
                lambda y: np.polyfit(x, y, 1)[0] if len(y) == momentum_length else 0,
                raw=False
            )

            latest_momentum = float(momentum.iloc[-1])
            in_squeeze = bool(squeeze_on.iloc[-1])

            # Signal logic
            if in_squeeze:
                return None  # Wait for squeeze release

            direction = 'BUY' if latest_momentum > 0 else 'SELL'
            strength = min(abs(latest_momentum) / 0.001, 1.0)  # Normalize

            return {
                'direction': direction,
                'strength': strength,
                'momentum': latest_momentum,
                'in_squeeze': in_squeeze
            }
        except Exception as e:
            self.logger.debug(f"Error calculating squeeze: {e}")
            return None

    def _calculate_rsi_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate RSI-based signal"""
        try:
            if 'rsi_14' in df.columns:
                rsi = float(df['rsi_14'].iloc[-1])
            elif 'rsi' in df.columns:
                rsi = float(df['rsi'].iloc[-1])
            else:
                # Calculate RSI
                period = self.config.rsi_period
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / (loss + 0.0001)
                rsi = float((100 - (100 / (1 + rs))).iloc[-1])

            # Signal logic
            if rsi <= self.config.rsi_oversold:
                return {
                    'direction': 'BUY',
                    'strength': (self.config.rsi_oversold - rsi) / self.config.rsi_oversold,
                    'value': rsi
                }
            elif rsi >= self.config.rsi_overbought:
                return {
                    'direction': 'SELL',
                    'strength': (rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought),
                    'value': rsi
                }

            return None
        except Exception as e:
            self.logger.debug(f"Error calculating RSI: {e}")
            return None

    def _calculate_stochastic_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Stochastic-based signal"""
        try:
            period = 14
            high = df['high'].rolling(window=period).max()
            low = df['low'].rolling(window=period).min()
            close = df['close']

            k = 100 * (close - low) / (high - low + 0.0001)
            d = k.rolling(window=3).mean()

            latest_k = float(k.iloc[-1])
            latest_d = float(d.iloc[-1])

            # Signal logic (oversold/overbought with crossover)
            if latest_k < 20 and latest_k > latest_d:
                return {
                    'direction': 'BUY',
                    'strength': (20 - latest_k) / 20,
                    'k': latest_k,
                    'd': latest_d
                }
            elif latest_k > 80 and latest_k < latest_d:
                return {
                    'direction': 'SELL',
                    'strength': (latest_k - 80) / 20,
                    'k': latest_k,
                    'd': latest_d
                }

            return None
        except Exception as e:
            self.logger.debug(f"Error calculating Stochastic: {e}")
            return None

    def _check_confluence(self, oscillator_signals: Dict) -> Optional[Dict]:
        """Check for oscillator confluence"""
        if not oscillator_signals:
            return None

        buy_signals = []
        sell_signals = []
        total_strength = 0

        weights = {
            'squeeze': self.config.squeeze_signal_weight,
            'rsi': self.config.rsi_signal_weight,
            'stochastic': self.config.rvi_signal_weight,  # Using stoch in place of RVI
        }

        for name, signal in oscillator_signals.items():
            weight = weights.get(name, 0.20)
            strength = signal.get('strength', 0.5) * weight

            if signal['direction'] == 'BUY':
                buy_signals.append((name, strength))
                total_strength += strength
            else:
                sell_signals.append((name, strength))
                total_strength += strength

        min_agreement = self.config.min_oscillator_agreement

        if len(buy_signals) >= min_agreement:
            confidence = sum(s[1] for s in buy_signals)
            if confidence >= self.config.min_combined_score:
                # Cap confidence
                confidence = min(confidence, self.config.max_confidence)
                return {
                    'direction': 'BUY',
                    'confidence': confidence,
                    'signals': [s[0] for s in buy_signals]
                }

        if len(sell_signals) >= min_agreement:
            confidence = sum(s[1] for s in sell_signals)
            if confidence >= self.config.min_combined_score:
                confidence = min(confidence, self.config.max_confidence)
                return {
                    'direction': 'SELL',
                    'confidence': confidence,
                    'signals': [s[0] for s in sell_signals]
                }

        return None

    def _validate_sr_bounce(self, df: pd.DataFrame, direction: str) -> bool:
        """Validate that price is bouncing from support/resistance"""
        try:
            # Simple S/R using recent high/low
            lookback = 20
            recent = df.tail(lookback)

            recent_high = float(recent['high'].max())
            recent_low = float(recent['low'].min())
            current_close = float(df['close'].iloc[-1])

            # Calculate range
            range_size = recent_high - recent_low
            proximity = self.config.sr_proximity_pips * 0.0001  # Convert to price

            if direction == 'BUY':
                # Should be near support (recent low)
                return current_close <= (recent_low + proximity * 2)
            else:
                # Should be near resistance (recent high)
                return current_close >= (recent_high - proximity * 2)

        except Exception as e:
            self.logger.debug(f"Error validating S/R: {e}")
            return True  # Allow if can't validate

    # =========================================================================
    # SIGNAL BUILDING
    # =========================================================================

    def _build_signal(
        self,
        epic: str,
        pair: str,
        direction: str,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        confidence: float,
        **kwargs
    ) -> Dict:
        """Build standardized signal dictionary"""
        now = datetime.now(timezone.utc)

        return {
            'signal': direction,
            'signal_type': direction.lower(),
            'strategy': self.strategy_name,
            'epic': epic,
            'pair': pair,
            'entry_price': entry_price,
            'stop_loss_pips': sl_pips,
            'take_profit_pips': tp_pips,
            'confidence_score': confidence,
            'confidence': confidence,
            'signal_timestamp': now.isoformat(),
            'timestamp': now,
            'version': self.config.version,
            'regime': 'ranging',
            'strategy_indicators': kwargs.get('indicators', {})
        }

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def _check_cooldown(self, epic: str) -> bool:
        """Check if epic is in cooldown period"""
        if epic not in self._cooldowns:
            return True

        now = datetime.now(timezone.utc)
        cooldown_end = self._cooldowns[epic]

        if now >= cooldown_end:
            del self._cooldowns[epic]
            return True

        return False

    def _set_cooldown(self, epic: str) -> None:
        """Set cooldown for epic after signal"""
        cooldown_minutes = self.config.signal_cooldown_minutes
        self._cooldowns[epic] = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)

    def reset_cooldowns(self) -> None:
        """Reset all cooldowns (for backtesting)"""
        self._cooldowns.clear()

    # =========================================================================
    # REJECTION LOGGING
    # =========================================================================

    def _log_rejection(self, epic: str, reason: str, value: Any = None) -> None:
        """Log signal rejection for analysis"""
        self._pending_rejections.append({
            'epic': epic,
            'strategy': self.strategy_name,
            'reason': reason,
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def flush_rejections(self) -> None:
        """Flush pending rejections"""
        self._pending_rejections.clear()


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_ranging_market_strategy(
    config=None,
    db_manager=None,
    logger=None
) -> RangingMarketStrategy:
    """Factory function to create Ranging Market strategy instance."""
    return RangingMarketStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager
    )
