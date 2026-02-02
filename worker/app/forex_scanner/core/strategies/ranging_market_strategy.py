#!/usr/bin/env python3
"""
Ranging Market Strategy v4.0 - Regime-Aware with Signal Qualification

VERSION: 4.0.0
DATE: 2026-02-02
STATUS: Active (backtest-only by default)

Architecture:
    1. DETECTION: Multi-oscillator confluence (Squeeze, RSI, Stochastic)
    2. QUALIFICATION: Score signal quality (0-100) based on multiple factors
    3. DECISION: Execute only if quality score meets threshold

Key Features:
    - Regime-Aware: Trusts multi-strategy routing (no redundant ADX check)
    - Per-Pair Tuning: All parameters configurable per epic
    - Signal Qualification: Quality scoring system for trade selection
    - Isolated Testing: Can run standalone with --strategy RANGING_MARKET

Qualification Factors (configurable weights):
    - Oscillator Agreement: How many oscillators agree (0-30 points)
    - Oscillator Strength: Average signal strength (0-20 points)
    - S/R Proximity: Distance to support/resistance (0-20 points)
    - ADX Condition: Lower ADX = better for ranging (0-15 points)
    - Session Bonus: Asian session edge (0-15 points)
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

from .strategy_registry import register_strategy, StrategyInterface


# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class RangingMarketConfig:
    """
    Configuration for Ranging Market strategy v4.0.

    All parameters support per-pair overrides via:
    1. Direct columns in ranging_market_pair_overrides table
    2. JSONB parameter_overrides column for extended parameters
    """

    # Strategy identification
    strategy_name: str = "RANGING_MARKET"
    version: str = "4.0.0"

    # ==========================================================================
    # REGIME INTEGRATION (v4.0)
    # ==========================================================================

    # When True, trusts multi-strategy routing and skips ADX filter
    trust_regime_routing: bool = True

    # Standalone ADX filter (used when trust_regime_routing=False or testing)
    use_adx_filter: bool = True
    adx_max_threshold: float = 25.0  # More permissive default
    adx_period: int = 14

    # ==========================================================================
    # OSCILLATOR SETTINGS
    # ==========================================================================

    # Squeeze Momentum
    squeeze_bb_length: int = 20
    squeeze_bb_mult: float = 2.0
    squeeze_kc_length: int = 20
    squeeze_kc_mult: float = 1.5
    squeeze_momentum_length: int = 12

    # RSI
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30

    # Stochastic
    stoch_period: int = 14
    stoch_smooth_k: int = 3
    stoch_overbought: int = 80
    stoch_oversold: int = 20

    # ==========================================================================
    # SIGNAL QUALIFICATION WEIGHTS (v4.0)
    # ==========================================================================

    # Quality score weights (must sum to 100)
    weight_oscillator_agreement: int = 30   # How many oscillators agree
    weight_oscillator_strength: int = 20    # Average signal strength
    weight_sr_proximity: int = 20           # Near support/resistance
    weight_adx_condition: int = 15          # Lower ADX = better
    weight_session_bonus: int = 15          # Asian session edge

    # Minimum quality score to generate signal (0-100)
    min_quality_score: int = 50

    # High quality threshold for boosted confidence
    high_quality_threshold: int = 75

    # ==========================================================================
    # OSCILLATOR CONFLUENCE
    # ==========================================================================

    # Minimum oscillators that must agree for a signal
    min_oscillator_agreement: int = 2

    # Individual oscillator enable flags
    use_squeeze_momentum: bool = True
    use_rsi: bool = True
    use_stochastic: bool = True

    # ==========================================================================
    # SUPPORT/RESISTANCE
    # ==========================================================================

    sr_bounce_required: bool = False  # Made optional for flexibility
    sr_lookback_bars: int = 20
    sr_proximity_pips: float = 10.0

    # ==========================================================================
    # SESSION CONFIGURATION
    # ==========================================================================

    # Session definitions (UTC hours)
    asian_session_start: int = 0
    asian_session_end: int = 8
    london_session_start: int = 7
    london_session_end: int = 16
    ny_session_start: int = 12
    ny_session_end: int = 21

    # Session quality bonuses (0-15 scale)
    asian_session_bonus: int = 15    # Best for ranging
    london_session_bonus: int = 5    # Moderate
    ny_session_bonus: int = 5        # Moderate
    overlap_session_bonus: int = 0   # Worst for ranging

    # ==========================================================================
    # RISK MANAGEMENT
    # ==========================================================================

    fixed_stop_loss_pips: float = 12.0
    fixed_take_profit_pips: float = 18.0
    min_confidence: float = 0.40     # Lowered - quality score handles filtering
    max_confidence: float = 0.85
    sl_buffer_pips: float = 2.0

    # ==========================================================================
    # TIMEFRAMES & COOLDOWN
    # ==========================================================================

    primary_timeframe: str = "15m"
    confirmation_timeframe: str = "1h"
    signal_cooldown_minutes: int = 30  # Reduced for more opportunities

    # ==========================================================================
    # PAIR CONFIGURATION
    # ==========================================================================

    enabled_pairs: List[str] = field(default_factory=list)
    _pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ==========================================================================
    # PER-PAIR GETTER METHODS
    # ==========================================================================

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        """Generic per-pair parameter getter with JSONB fallback."""
        if epic in self._pair_overrides:
            override_data = self._pair_overrides[epic]

            # Check direct field override
            if param_name in override_data and override_data[param_name] is not None:
                return override_data[param_name]

            # Check parameter_overrides JSONB
            param_overrides = override_data.get('parameter_overrides', {})
            if param_overrides and param_name in param_overrides:
                return param_overrides[param_name]

        # Fall back to global value
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def get_pair_adx_max_threshold(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'adx_max_threshold', self.adx_max_threshold))

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'fixed_stop_loss_pips', self.fixed_stop_loss_pips))

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'fixed_take_profit_pips', self.fixed_take_profit_pips))

    def get_pair_min_confidence(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'min_confidence', self.min_confidence))

    def get_pair_max_confidence(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'max_confidence', self.max_confidence))

    def get_pair_min_quality_score(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'min_quality_score', self.min_quality_score))

    def get_pair_rsi_overbought(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'rsi_overbought', self.rsi_overbought))

    def get_pair_rsi_oversold(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'rsi_oversold', self.rsi_oversold))

    def get_pair_stoch_overbought(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'stoch_overbought', self.stoch_overbought))

    def get_pair_stoch_oversold(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'stoch_oversold', self.stoch_oversold))

    def get_pair_min_oscillator_agreement(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'min_oscillator_agreement', self.min_oscillator_agreement))

    def get_pair_sr_proximity_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'sr_proximity_pips', self.sr_proximity_pips))

    def is_pair_enabled(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return self._pair_overrides[epic].get('is_enabled', True)
        if self.enabled_pairs:
            return epic in self.enabled_pairs
        return True

    @classmethod
    def from_database(cls, database_url: str = None) -> 'RangingMarketConfig':
        """Load configuration from database including per-pair overrides."""
        config = cls()

        if database_url is None:
            database_url = os.getenv(
                'STRATEGY_CONFIG_DATABASE_URL',
                'postgresql://postgres:postgres@postgres:5432/strategy_config'
            )

        try:
            conn = psycopg2.connect(database_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Load global config
            cur.execute("""
                SELECT * FROM ranging_market_global_config
                WHERE is_active = TRUE
                ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()

            if row:
                # Map all database columns to config attributes
                for key in row.keys():
                    if hasattr(config, key) and row[key] is not None:
                        value = row[key]
                        if hasattr(value, '__float__'):
                            # Check default type for int vs float
                            default_val = getattr(cls, key, None)
                            if isinstance(default_val, int):
                                value = int(value)
                            else:
                                value = float(value)
                        elif isinstance(value, bool):
                            value = bool(value)
                        setattr(config, key, value)

            # Load per-pair overrides
            cur.execute("""
                SELECT * FROM ranging_market_pair_overrides
                WHERE is_enabled = TRUE
            """)
            override_rows = cur.fetchall()

            config._pair_overrides = {}
            for override_row in override_rows:
                epic = override_row['epic']
                config._pair_overrides[epic] = dict(override_row)

            if override_rows:
                logging.info(
                    f"[RANGING_MARKET] Loaded {len(config._pair_overrides)} pair overrides"
                )

            cur.close()
            conn.close()

        except Exception as e:
            logging.warning(f"[RANGING_MARKET] Could not load from database: {e}")

        return config


# ==============================================================================
# SIGNAL QUALIFICATION RESULT
# ==============================================================================

@dataclass
class SignalQualification:
    """Result of signal qualification assessment."""

    is_qualified: bool           # Meets minimum quality threshold
    quality_score: int           # Overall quality (0-100)
    direction: str               # BUY or SELL

    # Component scores
    oscillator_agreement_score: int = 0
    oscillator_strength_score: int = 0
    sr_proximity_score: int = 0
    adx_condition_score: int = 0
    session_bonus_score: int = 0

    # Details for logging
    oscillators_agreeing: List[str] = field(default_factory=list)
    oscillators_total: int = 0
    adx_value: Optional[float] = None
    session: str = ""
    sr_distance_pips: Optional[float] = None

    # Rejection reason if not qualified
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'is_qualified': self.is_qualified,
            'quality_score': self.quality_score,
            'direction': self.direction,
            'scores': {
                'oscillator_agreement': self.oscillator_agreement_score,
                'oscillator_strength': self.oscillator_strength_score,
                'sr_proximity': self.sr_proximity_score,
                'adx_condition': self.adx_condition_score,
                'session_bonus': self.session_bonus_score
            },
            'details': {
                'oscillators_agreeing': self.oscillators_agreeing,
                'oscillators_total': self.oscillators_total,
                'adx_value': self.adx_value,
                'session': self.session,
                'sr_distance_pips': self.sr_distance_pips
            },
            'rejection_reason': self.rejection_reason
        }


# ==============================================================================
# CONFIG SERVICE (Singleton)
# ==============================================================================

class RangingMarketConfigService:
    """Singleton service for loading and caching configuration."""

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
        now = datetime.now()
        if (self._config is None or
            self._last_refresh is None or
            (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = RangingMarketConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> RangingMarketConfig:
        self._config = None
        return self.get_config()


def get_ranging_market_config() -> RangingMarketConfig:
    return RangingMarketConfigService.get_instance().get_config()


# ==============================================================================
# STRATEGY CLASS
# ==============================================================================

@register_strategy('RANGING_MARKET')
class RangingMarketStrategy(StrategyInterface):
    """
    Ranging Market Strategy v4.0 with Signal Qualification

    Three-phase approach:
    1. DETECT: Find oscillator confluence signals
    2. QUALIFY: Score signal quality (0-100)
    3. DECIDE: Generate signal only if quality meets threshold
    """

    def __init__(self, config=None, logger=None, db_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config = get_ranging_market_config()

        self._cooldowns: Dict[str, datetime] = {}
        self._rejection_log: List[Dict] = []

        self.logger.info(f"[RANGING_MARKET] Strategy v{self.config.version} initialized")
        self.logger.info(f"[RANGING_MARKET] Trust regime routing: {self.config.trust_regime_routing}")
        self.logger.info(f"[RANGING_MARKET] Min quality score: {self.config.min_quality_score}")

    @property
    def strategy_name(self) -> str:
        return "RANGING_MARKET"

    def get_required_timeframes(self) -> List[str]:
        return [self.config.confirmation_timeframe, self.config.primary_timeframe]

    # ==========================================================================
    # MAIN ENTRY POINT
    # ==========================================================================

    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        df_entry: pd.DataFrame = None,
        current_timestamp: datetime = None,
        routing_context: Dict = None,  # NEW: Context from multi-strategy router
        **kwargs
    ) -> Optional[Dict]:
        """
        Detect and qualify trading signal.

        Args:
            df_trigger: Primary timeframe data (15m)
            df_4h: Higher timeframe data (1H)
            epic: IG epic identifier
            pair: Currency pair name
            current_timestamp: For backtest mode
            routing_context: From multi-strategy router (regime, session, etc.)

        Returns:
            Qualified signal dict or None
        """
        self._current_timestamp = current_timestamp
        df = df_trigger if df_trigger is not None else df_entry

        if df is None or len(df) < 50:
            return None

        if not self.config.is_pair_enabled(epic):
            return None

        if not self._check_cooldown(epic):
            return None

        # Phase 1: Check regime/ADX filter
        adx_value = self._get_adx(df)
        regime_ok = self._check_regime_filter(epic, adx_value, routing_context)
        if not regime_ok:
            return None

        # Phase 2: Detect oscillator signals
        oscillator_results = self._detect_oscillator_confluence(df, epic)
        if not oscillator_results:
            self._log_rejection(epic, "NO_OSCILLATOR_SIGNALS")
            return None

        # Phase 3: Qualify the signal
        qualification = self._qualify_signal(
            epic=epic,
            df=df,
            oscillator_results=oscillator_results,
            adx_value=adx_value,
            current_timestamp=current_timestamp
        )

        if not qualification.is_qualified:
            self._log_rejection(
                epic,
                qualification.rejection_reason or "QUALITY_TOO_LOW",
                {'quality_score': qualification.quality_score}
            )
            return None

        # Phase 4: Build qualified signal
        latest = df.iloc[-1]
        entry_price = float(latest.get('close', 0))

        # Map quality score to confidence
        confidence = self._quality_to_confidence(qualification.quality_score, epic)

        signal = self._build_signal(
            epic=epic,
            pair=pair,
            direction=qualification.direction,
            entry_price=entry_price,
            sl_pips=self.config.get_pair_fixed_stop_loss(epic),
            tp_pips=self.config.get_pair_fixed_take_profit(epic),
            confidence=confidence,
            qualification=qualification,
            adx_value=adx_value
        )

        self._set_cooldown(epic)

        self.logger.info(
            f"[RANGING_MARKET] ✅ QUALIFIED SIGNAL: {qualification.direction} {epic} "
            f"(quality={qualification.quality_score}, conf={confidence:.1%}, "
            f"oscillators={qualification.oscillators_agreeing})"
        )

        return signal

    # ==========================================================================
    # PHASE 1: REGIME/ADX FILTER
    # ==========================================================================

    def _check_regime_filter(
        self,
        epic: str,
        adx_value: Optional[float],
        routing_context: Dict = None
    ) -> bool:
        """
        Check if market conditions are suitable for ranging strategy.

        When trust_regime_routing=True and routing provides regime='ranging',
        skip the ADX filter (router already validated this).
        """
        # If routing says regime is 'ranging' or 'low_volatility', trust it
        if self.config.trust_regime_routing and routing_context:
            regime = routing_context.get('regime', '')
            if regime in ('ranging', 'low_volatility'):
                self.logger.debug(
                    f"[RANGING_MARKET] Trusting regime routing: {regime} for {epic}"
                )
                return True

        # Standalone mode or no routing - use ADX filter
        if self.config.use_adx_filter and adx_value is not None:
            threshold = self.config.get_pair_adx_max_threshold(epic)
            if adx_value > threshold:
                self._log_rejection(
                    epic,
                    "ADX_TOO_HIGH",
                    {'adx': adx_value, 'threshold': threshold}
                )
                self.logger.debug(
                    f"[RANGING_MARKET] ADX {adx_value:.1f} > {threshold} for {epic}"
                )
                return False

        return True

    # ==========================================================================
    # PHASE 2: OSCILLATOR DETECTION
    # ==========================================================================

    def _detect_oscillator_confluence(
        self,
        df: pd.DataFrame,
        epic: str
    ) -> Optional[Dict]:
        """Detect signals from all enabled oscillators."""
        signals = {}

        if self.config.use_squeeze_momentum:
            squeeze = self._calc_squeeze_momentum(df)
            if squeeze:
                signals['squeeze'] = squeeze

        if self.config.use_rsi:
            rsi = self._calc_rsi_signal(df, epic)
            if rsi:
                signals['rsi'] = rsi

        if self.config.use_stochastic:
            stoch = self._calc_stochastic_signal(df, epic)
            if stoch:
                signals['stochastic'] = stoch

        if not signals:
            return None

        # Check minimum agreement
        min_agreement = self.config.get_pair_min_oscillator_agreement(epic)

        buy_signals = [k for k, v in signals.items() if v['direction'] == 'BUY']
        sell_signals = [k for k, v in signals.items() if v['direction'] == 'SELL']

        if len(buy_signals) >= min_agreement:
            return {
                'direction': 'BUY',
                'signals': signals,
                'agreeing': buy_signals,
                'total': len(signals)
            }
        elif len(sell_signals) >= min_agreement:
            return {
                'direction': 'SELL',
                'signals': signals,
                'agreeing': sell_signals,
                'total': len(signals)
            }

        return None

    def _calc_squeeze_momentum(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Squeeze Momentum signal."""
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

            # Squeeze detection
            squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

            # Momentum
            mom_length = self.config.squeeze_momentum_length
            x = np.arange(mom_length)
            momentum = close.rolling(window=mom_length).apply(
                lambda y: np.polyfit(x, y, 1)[0] if len(y) == mom_length else 0,
                raw=False
            )

            latest_momentum = float(momentum.iloc[-1])
            in_squeeze = bool(squeeze_on.iloc[-1])

            if in_squeeze:
                return None  # Wait for squeeze release

            direction = 'BUY' if latest_momentum > 0 else 'SELL'
            strength = min(abs(latest_momentum) / 0.0005, 1.0)

            return {
                'direction': direction,
                'strength': strength,
                'momentum': latest_momentum,
                'in_squeeze': in_squeeze
            }
        except Exception as e:
            self.logger.debug(f"[RANGING_MARKET] Squeeze calc error: {e}")
            return None

    def _calc_rsi_signal(self, df: pd.DataFrame, epic: str) -> Optional[Dict]:
        """Calculate RSI-based signal with per-pair thresholds."""
        try:
            # Check for pre-calculated RSI
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

            # Per-pair thresholds
            oversold = self.config.get_pair_rsi_oversold(epic)
            overbought = self.config.get_pair_rsi_overbought(epic)

            if rsi <= oversold:
                strength = (oversold - rsi) / oversold
                return {'direction': 'BUY', 'strength': min(strength, 1.0), 'value': rsi}
            elif rsi >= overbought:
                strength = (rsi - overbought) / (100 - overbought)
                return {'direction': 'SELL', 'strength': min(strength, 1.0), 'value': rsi}

            return None
        except Exception as e:
            self.logger.debug(f"[RANGING_MARKET] RSI calc error: {e}")
            return None

    def _calc_stochastic_signal(self, df: pd.DataFrame, epic: str) -> Optional[Dict]:
        """Calculate Stochastic signal with per-pair thresholds."""
        try:
            period = self.config.stoch_period
            high = df['high'].rolling(window=period).max()
            low = df['low'].rolling(window=period).min()
            close = df['close']

            k = 100 * (close - low) / (high - low + 0.0001)
            d = k.rolling(window=self.config.stoch_smooth_k).mean()

            latest_k = float(k.iloc[-1])
            latest_d = float(d.iloc[-1])

            # Per-pair thresholds
            oversold = self.config.get_pair_stoch_oversold(epic)
            overbought = self.config.get_pair_stoch_overbought(epic)

            # Oversold with bullish crossover
            if latest_k < oversold and latest_k > latest_d:
                strength = (oversold - latest_k) / oversold
                return {'direction': 'BUY', 'strength': min(strength, 1.0), 'k': latest_k, 'd': latest_d}

            # Overbought with bearish crossover
            elif latest_k > overbought and latest_k < latest_d:
                strength = (latest_k - overbought) / (100 - overbought)
                return {'direction': 'SELL', 'strength': min(strength, 1.0), 'k': latest_k, 'd': latest_d}

            return None
        except Exception as e:
            self.logger.debug(f"[RANGING_MARKET] Stochastic calc error: {e}")
            return None

    # ==========================================================================
    # PHASE 3: SIGNAL QUALIFICATION
    # ==========================================================================

    def _qualify_signal(
        self,
        epic: str,
        df: pd.DataFrame,
        oscillator_results: Dict,
        adx_value: Optional[float],
        current_timestamp: Optional[datetime] = None
    ) -> SignalQualification:
        """
        Score signal quality on multiple factors.

        Returns SignalQualification with is_qualified=True if meets threshold.
        """
        direction = oscillator_results['direction']
        agreeing = oscillator_results['agreeing']
        signals = oscillator_results['signals']
        total_oscillators = oscillator_results['total']

        # 1. Oscillator Agreement Score (0-30)
        agreement_ratio = len(agreeing) / max(total_oscillators, 1)
        osc_agreement_score = int(agreement_ratio * self.config.weight_oscillator_agreement)

        # 2. Oscillator Strength Score (0-20)
        avg_strength = np.mean([signals[osc]['strength'] for osc in agreeing])
        osc_strength_score = int(avg_strength * self.config.weight_oscillator_strength)

        # 3. S/R Proximity Score (0-20)
        sr_distance, sr_score = self._calc_sr_proximity_score(df, direction, epic)

        # 4. ADX Condition Score (0-15) - Lower ADX = better for ranging
        adx_score = self._calc_adx_score(adx_value, epic)

        # 5. Session Bonus Score (0-15)
        session, session_score = self._calc_session_score(current_timestamp)

        # Total quality score
        quality_score = (
            osc_agreement_score +
            osc_strength_score +
            sr_score +
            adx_score +
            session_score
        )

        # Check against threshold
        min_quality = self.config.get_pair_min_quality_score(epic)
        is_qualified = quality_score >= min_quality

        rejection_reason = None
        if not is_qualified:
            rejection_reason = f"Quality {quality_score} < min {min_quality}"

        return SignalQualification(
            is_qualified=is_qualified,
            quality_score=quality_score,
            direction=direction,
            oscillator_agreement_score=osc_agreement_score,
            oscillator_strength_score=osc_strength_score,
            sr_proximity_score=sr_score,
            adx_condition_score=adx_score,
            session_bonus_score=session_score,
            oscillators_agreeing=agreeing,
            oscillators_total=total_oscillators,
            adx_value=adx_value,
            session=session,
            sr_distance_pips=sr_distance,
            rejection_reason=rejection_reason
        )

    def _calc_sr_proximity_score(
        self,
        df: pd.DataFrame,
        direction: str,
        epic: str
    ) -> Tuple[Optional[float], int]:
        """Calculate S/R proximity score. Closer to S/R = higher score."""
        try:
            lookback = self.config.sr_lookback_bars
            recent = df.tail(lookback)

            recent_high = float(recent['high'].max())
            recent_low = float(recent['low'].min())
            current_price = float(df['close'].iloc[-1])

            pip_value = 0.01 if 'JPY' in epic else 0.0001
            max_proximity = self.config.get_pair_sr_proximity_pips(epic)

            if direction == 'BUY':
                distance_pips = (current_price - recent_low) / pip_value
            else:
                distance_pips = (recent_high - current_price) / pip_value

            # Score: closer = higher (max when at S/R, 0 when > max_proximity away)
            if distance_pips <= max_proximity:
                ratio = 1.0 - (distance_pips / max_proximity)
                score = int(ratio * self.config.weight_sr_proximity)
            else:
                score = 0

            return distance_pips, score

        except Exception as e:
            self.logger.debug(f"[RANGING_MARKET] S/R calc error: {e}")
            return None, 0

    def _calc_adx_score(self, adx_value: Optional[float], epic: str) -> int:
        """Calculate ADX condition score. Lower ADX = higher score for ranging."""
        if adx_value is None:
            return self.config.weight_adx_condition // 2  # Neutral score

        threshold = self.config.get_pair_adx_max_threshold(epic)

        if adx_value <= 15:
            return self.config.weight_adx_condition  # Full score - ideal ranging
        elif adx_value <= threshold:
            ratio = 1.0 - ((adx_value - 15) / (threshold - 15))
            return int(ratio * self.config.weight_adx_condition)
        else:
            return 0  # ADX too high

    def _calc_session_score(self, current_timestamp: Optional[datetime] = None) -> Tuple[str, int]:
        """Calculate session bonus score. Asian session best for ranging."""
        now = current_timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        hour = now.hour

        if self.config.asian_session_start <= hour < self.config.asian_session_end:
            return 'asian', self.config.asian_session_bonus
        elif self.config.london_session_start <= hour < self.config.ny_session_start:
            return 'london', self.config.london_session_bonus
        elif self.config.ny_session_start <= hour < self.config.london_session_end:
            return 'overlap', self.config.overlap_session_bonus
        elif self.config.ny_session_start <= hour < self.config.ny_session_end:
            return 'new_york', self.config.ny_session_bonus
        else:
            return 'off_hours', 0

    # ==========================================================================
    # PHASE 4: SIGNAL BUILDING
    # ==========================================================================

    def _quality_to_confidence(self, quality_score: int, epic: str) -> float:
        """Map quality score (0-100) to confidence (min_confidence to max_confidence)."""
        min_conf = self.config.get_pair_min_confidence(epic)
        max_conf = self.config.get_pair_max_confidence(epic)

        # Linear mapping: quality 50 → min_conf, quality 100 → max_conf
        min_quality = self.config.get_pair_min_quality_score(epic)
        ratio = (quality_score - min_quality) / (100 - min_quality)
        ratio = max(0, min(1, ratio))  # Clamp to [0, 1]

        confidence = min_conf + (ratio * (max_conf - min_conf))
        return round(confidence, 3)

    def _build_signal(
        self,
        epic: str,
        pair: str,
        direction: str,
        entry_price: float,
        sl_pips: float,
        tp_pips: float,
        confidence: float,
        qualification: SignalQualification,
        adx_value: Optional[float]
    ) -> Dict:
        """Build standardized signal dictionary with qualification details."""
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

            # Qualification details
            'quality_score': qualification.quality_score,
            'qualification': qualification.to_dict(),

            'strategy_indicators': {
                'adx': adx_value,
                'oscillators_agreeing': qualification.oscillators_agreeing,
                'session': qualification.session,
                'sr_distance_pips': qualification.sr_distance_pips
            }
        }

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def _get_adx(self, df: pd.DataFrame) -> Optional[float]:
        """Get or calculate ADX value."""
        if 'adx' in df.columns:
            return float(df['adx'].iloc[-1])

        try:
            period = self.config.adx_period
            high, low, close = df['high'], df['low'], df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            dm_plus = (high - high.shift(1)).clip(lower=0)
            dm_minus = (low.shift(1) - low).clip(lower=0)

            atr = tr.rolling(window=period).mean()
            di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)

            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 0.0001)
            adx = dx.rolling(window=period).mean()

            return float(adx.iloc[-1])
        except Exception as e:
            self.logger.debug(f"[RANGING_MARKET] ADX calc error: {e}")
            return None

    def _check_cooldown(self, epic: str) -> bool:
        if epic not in self._cooldowns:
            return True
        now = getattr(self, '_current_timestamp', None) or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if now >= self._cooldowns[epic]:
            del self._cooldowns[epic]
            return True
        return False

    def _set_cooldown(self, epic: str) -> None:
        now = getattr(self, '_current_timestamp', None) or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._cooldowns[epic] = now + timedelta(minutes=self.config.signal_cooldown_minutes)

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

    def _log_rejection(self, epic: str, reason: str, details: Any = None) -> None:
        self._rejection_log.append({
            'epic': epic,
            'strategy': self.strategy_name,
            'reason': reason,
            'details': details,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    def get_rejection_log(self) -> List[Dict]:
        return self._rejection_log.copy()

    def clear_rejection_log(self) -> None:
        self._rejection_log.clear()


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_ranging_market_strategy(
    config=None,
    db_manager=None,
    logger=None
) -> RangingMarketStrategy:
    """Factory function to create strategy instance."""
    return RangingMarketStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager
    )
