#!/usr/bin/env python3
"""
Volume Profile Strategy v4.0 - Regime-Aware with Signal Qualification

VERSION: 4.0.0
DATE: 2026-02-02
STATUS: Active (backtest-only by default)

Architecture:
    1. DETECTION: Volume profile signals (HVN bounce, POC reversion, VA breakout)
    2. QUALIFICATION: Score signal quality (0-100) based on multiple factors
    3. DECISION: Execute only if quality score meets threshold

Key Features:
    - Regime-Aware: Trusts multi-strategy routing for high_volatility/breakout
    - Per-Pair Tuning: All parameters configurable per epic
    - Signal Qualification: Quality scoring system for trade selection
    - Session Edge: 66.7% Asian session win rate bonus

Qualification Factors (configurable weights):
    - Signal Type Strength: HVN bounce vs POC vs VA breakout (0-25 points)
    - Level Proximity: How close to key level (0-25 points)
    - Volume Confirmation: Node strength (0-20 points)
    - Session Bonus: Asian session edge (0-20 points)
    - Trend Alignment: Direction with recent trend (0-10 points)
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
class VolumeProfileConfig:
    """
    Configuration for Volume Profile strategy v4.0.

    All parameters support per-pair overrides via database.
    """

    # Strategy identification
    strategy_name: str = "VOLUME_PROFILE"
    version: str = "4.0.0"

    # ==========================================================================
    # REGIME INTEGRATION (v4.0)
    # ==========================================================================

    # When True, trusts multi-strategy routing
    trust_regime_routing: bool = True

    # ==========================================================================
    # VOLUME PROFILE SETTINGS
    # ==========================================================================

    vp_lookback_bars: int = 100
    vp_value_area_percent: float = 70.0
    hvn_threshold_percentile: int = 80
    lvn_threshold_percentile: int = 20

    # POC Settings
    poc_proximity_pips: float = 5.0
    poc_reversion_enabled: bool = True

    # Value Area Settings
    va_high_proximity_pips: float = 8.0
    va_low_proximity_pips: float = 8.0
    va_breakout_enabled: bool = True

    # HVN Bounce Settings
    hvn_bounce_enabled: bool = True
    hvn_bounce_proximity_pips: float = 5.0

    # ==========================================================================
    # SIGNAL QUALIFICATION WEIGHTS (v4.0)
    # ==========================================================================

    # Quality score weights (must sum to 100)
    weight_signal_type: int = 25        # HVN bounce strongest, then POC, then VA
    weight_level_proximity: int = 25    # Closer to level = higher score
    weight_volume_strength: int = 20    # HVN volume strength
    weight_session_bonus: int = 20      # Asian session edge
    weight_trend_alignment: int = 10    # Direction with recent trend

    # Minimum quality score to generate signal (0-100)
    min_quality_score: int = 50

    # High quality threshold for boosted confidence
    high_quality_threshold: int = 75

    # Signal type base scores (out of weight_signal_type max)
    hvn_bounce_base_score: int = 25     # Best signal type
    poc_reversion_base_score: int = 20  # Good signal type
    va_breakout_base_score: int = 15    # Weaker signal type

    # ==========================================================================
    # SESSION CONFIGURATION
    # ==========================================================================

    asian_session_start: int = 0
    asian_session_end: int = 8
    london_session_start: int = 7
    london_session_end: int = 16
    ny_session_start: int = 12
    ny_session_end: int = 21

    # Session quality bonuses (0-20 scale)
    asian_session_bonus: int = 20       # Best for volume profile (66.7% WR)
    london_session_bonus: int = 10      # Moderate
    ny_session_bonus: int = 10          # Moderate
    overlap_session_bonus: int = 5      # Worst for this strategy

    # Legacy boost multiplier (for backwards compatibility)
    asian_session_boost: float = 1.15

    # ==========================================================================
    # RISK MANAGEMENT
    # ==========================================================================

    fixed_stop_loss_pips: float = 15.0
    fixed_take_profit_pips: float = 22.0
    min_confidence: float = 0.40
    max_confidence: float = 0.85
    sl_buffer_pips: float = 2.0

    # ==========================================================================
    # TIMEFRAMES & COOLDOWN
    # ==========================================================================

    primary_timeframe: str = "15m"
    profile_timeframe: str = "1h"
    signal_cooldown_minutes: int = 45

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
            if param_name in override_data and override_data[param_name] is not None:
                return override_data[param_name]
            param_overrides = override_data.get('parameter_overrides', {})
            if param_overrides and param_name in param_overrides:
                return param_overrides[param_name]
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

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

    def get_pair_asian_session_bonus(self, epic: str) -> int:
        return int(self.get_for_pair(epic, 'asian_session_bonus', self.asian_session_bonus))

    def get_pair_poc_proximity_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'poc_proximity_pips', self.poc_proximity_pips))

    def get_pair_hvn_proximity_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'hvn_bounce_proximity_pips', self.hvn_bounce_proximity_pips))

    def get_pair_va_proximity_pips(self, epic: str) -> float:
        return float(self.get_for_pair(epic, 'va_high_proximity_pips', self.va_high_proximity_pips))

    def is_pair_enabled(self, epic: str) -> bool:
        if epic in self._pair_overrides:
            return self._pair_overrides[epic].get('is_enabled', True)
        if self.enabled_pairs:
            return epic in self.enabled_pairs
        return True

    @classmethod
    def from_database(cls, database_url: str = None) -> 'VolumeProfileConfig':
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
                SELECT * FROM volume_profile_global_config
                WHERE is_active = TRUE
                ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()

            if row:
                for key in row.keys():
                    if hasattr(config, key) and row[key] is not None:
                        value = row[key]
                        if hasattr(value, '__float__'):
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
                SELECT * FROM volume_profile_pair_overrides
                WHERE is_enabled = TRUE
            """)
            override_rows = cur.fetchall()

            config._pair_overrides = {}
            for override_row in override_rows:
                epic = override_row['epic']
                config._pair_overrides[epic] = dict(override_row)

            if override_rows:
                logging.info(f"[VOLUME_PROFILE] Loaded {len(config._pair_overrides)} pair overrides")

            cur.close()
            conn.close()

        except Exception as e:
            logging.warning(f"[VOLUME_PROFILE] Could not load from database: {e}")

        return config


# ==============================================================================
# SIGNAL QUALIFICATION RESULT
# ==============================================================================

@dataclass
class SignalQualification:
    """Result of signal qualification assessment."""

    is_qualified: bool
    quality_score: int
    direction: str
    signal_type: str  # HVN_BOUNCE, POC_REVERSION, VA_BREAKOUT

    # Component scores
    signal_type_score: int = 0
    level_proximity_score: int = 0
    volume_strength_score: int = 0
    session_bonus_score: int = 0
    trend_alignment_score: int = 0

    # Details
    session: str = ""
    level_distance_pips: Optional[float] = None
    volume_node_strength: Optional[float] = None
    key_level: Optional[float] = None

    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'is_qualified': self.is_qualified,
            'quality_score': self.quality_score,
            'direction': self.direction,
            'signal_type': self.signal_type,
            'scores': {
                'signal_type': self.signal_type_score,
                'level_proximity': self.level_proximity_score,
                'volume_strength': self.volume_strength_score,
                'session_bonus': self.session_bonus_score,
                'trend_alignment': self.trend_alignment_score
            },
            'details': {
                'session': self.session,
                'level_distance_pips': self.level_distance_pips,
                'volume_node_strength': self.volume_node_strength,
                'key_level': self.key_level
            },
            'rejection_reason': self.rejection_reason
        }


# ==============================================================================
# CONFIG SERVICE (Singleton)
# ==============================================================================

class VolumeProfileConfigService:
    """Singleton service for loading and caching configuration."""

    _instance: Optional['VolumeProfileConfigService'] = None
    _config: Optional[VolumeProfileConfig] = None
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
    def get_instance(cls) -> 'VolumeProfileConfigService':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> VolumeProfileConfig:
        now = datetime.now()
        if (self._config is None or
            self._last_refresh is None or
            (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = VolumeProfileConfig.from_database()
            self._last_refresh = now
        return self._config

    def refresh(self) -> VolumeProfileConfig:
        self._config = None
        return self.get_config()


def get_volume_profile_config() -> VolumeProfileConfig:
    return VolumeProfileConfigService.get_instance().get_config()


# ==============================================================================
# STRATEGY CLASS
# ==============================================================================

@register_strategy('VOLUME_PROFILE')
class VolumeProfileStrategy(StrategyInterface):
    """
    Volume Profile Strategy v4.0 with Signal Qualification

    Three-phase approach:
    1. DETECT: Find volume profile signals (HVN, POC, VA)
    2. QUALIFY: Score signal quality (0-100)
    3. DECIDE: Generate signal only if quality meets threshold
    """

    def __init__(self, config=None, logger=None, db_manager=None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager
        self.config = get_volume_profile_config()

        self._cooldowns: Dict[str, datetime] = {}
        self._rejection_log: List[Dict] = []
        self._profile_cache: Dict[str, Dict] = {}

        self.logger.info(f"[VOLUME_PROFILE] Strategy v{self.config.version} initialized")
        self.logger.info(f"[VOLUME_PROFILE] Min quality score: {self.config.min_quality_score}")

    @property
    def strategy_name(self) -> str:
        return "VOLUME_PROFILE"

    def get_required_timeframes(self) -> List[str]:
        return [self.config.profile_timeframe, self.config.primary_timeframe]

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
        current_timestamp: Optional[datetime] = None,
        routing_context: Optional[Dict] = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Detect and qualify trading signal.
        """
        self._current_timestamp = current_timestamp
        df = df_trigger if df_trigger is not None else df_entry

        if df is None or len(df) < self.config.vp_lookback_bars:
            return None

        if not self.config.is_pair_enabled(epic):
            return None

        if not self._check_cooldown(epic):
            return None

        # Phase 1: Calculate volume profile
        profile = self._calculate_volume_profile(df, epic)
        if profile is None:
            return None

        current_price = float(df['close'].iloc[-1])

        # Phase 2: Detect potential signals
        signal_candidates = self._detect_all_signals(profile, current_price, df, epic)
        if not signal_candidates:
            self._log_rejection(epic, "NO_VP_SIGNALS")
            return None

        # Phase 3: Qualify each signal and pick best
        best_qualification = None
        best_signal_data = None

        for signal_data in signal_candidates:
            qualification = self._qualify_signal(
                epic=epic,
                df=df,
                signal_data=signal_data,
                profile=profile,
                current_timestamp=current_timestamp
            )

            if qualification.is_qualified:
                if best_qualification is None or qualification.quality_score > best_qualification.quality_score:
                    best_qualification = qualification
                    best_signal_data = signal_data

        if best_qualification is None:
            self._log_rejection(epic, "ALL_SIGNALS_LOW_QUALITY")
            return None

        # Phase 4: Build qualified signal
        confidence = self._quality_to_confidence(best_qualification.quality_score, epic)

        signal = self._build_signal(
            epic=epic,
            pair=pair,
            direction=best_qualification.direction,
            entry_price=current_price,
            sl_pips=self.config.get_pair_fixed_stop_loss(epic),
            tp_pips=self.config.get_pair_fixed_take_profit(epic),
            confidence=confidence,
            qualification=best_qualification,
            profile=profile
        )

        self._set_cooldown(epic)

        self.logger.info(
            f"[VOLUME_PROFILE] ✅ QUALIFIED SIGNAL: {best_qualification.direction} {epic} "
            f"(type={best_qualification.signal_type}, quality={best_qualification.quality_score}, "
            f"session={best_qualification.session})"
        )

        return signal

    # ==========================================================================
    # PHASE 1: VOLUME PROFILE CALCULATION
    # ==========================================================================

    def _calculate_volume_profile(self, df: pd.DataFrame, epic: str) -> Optional[Dict]:
        """Calculate Volume Profile including POC, VAH, VAL, and HVNs."""
        try:
            lookback = self.config.vp_lookback_bars
            data = df.tail(lookback).copy()

            if len(data) < 20:
                return None

            high = float(data['high'].max())
            low = float(data['low'].min())
            price_range = high - low

            if price_range <= 0:
                return None

            n_bins = 50
            data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3

            if 'volume' in data.columns and data['volume'].sum() > 0:
                volume = data['volume']
            else:
                volume = (data['high'] - data['low']).abs()

            price_bins = np.linspace(low, high, n_bins + 1)
            volume_by_price = np.zeros(n_bins)

            for i, row in data.iterrows():
                tp = row['typical_price']
                vol = volume.loc[i] if hasattr(volume, 'loc') else 1.0
                bin_idx = int((tp - low) / price_range * (n_bins - 1))
                bin_idx = max(0, min(n_bins - 1, bin_idx))
                volume_by_price[bin_idx] += vol

            # POC
            poc_idx = np.argmax(volume_by_price)
            poc = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            poc_volume = volume_by_price[poc_idx]

            # Value Area
            total_volume = volume_by_price.sum()
            target_volume = total_volume * (self.config.vp_value_area_percent / 100)

            captured_volume = volume_by_price[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx

            while captured_volume < target_volume:
                low_add = volume_by_price[low_idx - 1] if low_idx > 0 else 0
                high_add = volume_by_price[high_idx + 1] if high_idx < n_bins - 1 else 0

                if low_add >= high_add and low_idx > 0:
                    low_idx -= 1
                    captured_volume += low_add
                elif high_idx < n_bins - 1:
                    high_idx += 1
                    captured_volume += high_add
                else:
                    break

            vah = (price_bins[high_idx] + price_bins[high_idx + 1]) / 2
            val = (price_bins[low_idx] + price_bins[low_idx + 1]) / 2

            # HVNs
            hvn_threshold = np.percentile(volume_by_price, self.config.hvn_threshold_percentile)
            max_volume = volume_by_price.max()
            hvns = []
            for i in range(n_bins):
                if volume_by_price[i] >= hvn_threshold:
                    price = (price_bins[i] + price_bins[i + 1]) / 2
                    strength = volume_by_price[i] / max_volume if max_volume > 0 else 0
                    hvns.append({'price': price, 'volume': volume_by_price[i], 'strength': strength})

            return {
                'poc': poc,
                'poc_volume': poc_volume,
                'vah': vah,
                'val': val,
                'hvns': hvns,
                'high': high,
                'low': low,
                'total_volume': total_volume,
                'max_volume': max_volume
            }

        except Exception as e:
            self.logger.warning(f"[VOLUME_PROFILE] Profile calc error: {e}")
            return None

    # ==========================================================================
    # PHASE 2: SIGNAL DETECTION
    # ==========================================================================

    def _detect_all_signals(
        self,
        profile: Dict,
        current_price: float,
        df: pd.DataFrame,
        epic: str
    ) -> List[Dict]:
        """Detect all potential signals from volume profile."""
        signals = []
        pip_value = 0.01 if 'JPY' in epic else 0.0001

        # Check HVN Bounce
        if self.config.hvn_bounce_enabled:
            hvn_signal = self._check_hvn_bounce(profile, current_price, epic, pip_value)
            if hvn_signal:
                signals.append(hvn_signal)

        # Check POC Reversion
        if self.config.poc_reversion_enabled:
            poc_signal = self._check_poc_reversion(profile, current_price, epic, pip_value)
            if poc_signal:
                signals.append(poc_signal)

        # Check VA Breakout
        if self.config.va_breakout_enabled:
            va_signal = self._check_va_breakout(profile, current_price, df, epic, pip_value)
            if va_signal:
                signals.append(va_signal)

        return signals

    def _check_hvn_bounce(
        self,
        profile: Dict,
        current_price: float,
        epic: str,
        pip_value: float
    ) -> Optional[Dict]:
        """Check for HVN bounce signal."""
        proximity = self.config.get_pair_hvn_proximity_pips(epic) * pip_value

        for hvn in profile['hvns']:
            hvn_price = hvn['price']
            distance = abs(current_price - hvn_price)

            if distance <= proximity:
                if current_price > hvn_price:
                    return {
                        'type': 'HVN_BOUNCE',
                        'direction': 'BUY',
                        'level': hvn_price,
                        'distance_pips': distance / pip_value,
                        'volume_strength': hvn['strength']
                    }
                else:
                    return {
                        'type': 'HVN_BOUNCE',
                        'direction': 'SELL',
                        'level': hvn_price,
                        'distance_pips': distance / pip_value,
                        'volume_strength': hvn['strength']
                    }

        return None

    def _check_poc_reversion(
        self,
        profile: Dict,
        current_price: float,
        epic: str,
        pip_value: float
    ) -> Optional[Dict]:
        """Check for POC mean reversion signal."""
        poc = profile['poc']
        vah = profile['vah']
        val = profile['val']
        proximity = self.config.get_pair_poc_proximity_pips(epic) * pip_value

        # Price at VAH, expect reversion to POC
        if current_price >= vah - proximity and current_price > poc:
            return {
                'type': 'POC_REVERSION',
                'direction': 'SELL',
                'level': vah,
                'target': poc,
                'distance_pips': abs(current_price - vah) / pip_value,
                'volume_strength': profile['poc_volume'] / profile['max_volume'] if profile['max_volume'] > 0 else 0.5
            }

        # Price at VAL, expect reversion to POC
        elif current_price <= val + proximity and current_price < poc:
            return {
                'type': 'POC_REVERSION',
                'direction': 'BUY',
                'level': val,
                'target': poc,
                'distance_pips': abs(current_price - val) / pip_value,
                'volume_strength': profile['poc_volume'] / profile['max_volume'] if profile['max_volume'] > 0 else 0.5
            }

        return None

    def _check_va_breakout(
        self,
        profile: Dict,
        current_price: float,
        df: pd.DataFrame,
        epic: str,
        pip_value: float
    ) -> Optional[Dict]:
        """Check for Value Area breakout signal."""
        proximity = self.config.get_pair_va_proximity_pips(epic) * pip_value
        vah = profile['vah']
        val = profile['val']

        recent_closes = df['close'].tail(3)
        prev_close = float(recent_closes.iloc[-2])

        # Breakout above VAH
        if current_price > vah + proximity and prev_close <= vah:
            return {
                'type': 'VA_BREAKOUT',
                'direction': 'BUY',
                'level': vah,
                'distance_pips': (current_price - vah) / pip_value,
                'volume_strength': 0.5  # Breakouts use fixed strength
            }

        # Breakdown below VAL
        if current_price < val - proximity and prev_close >= val:
            return {
                'type': 'VA_BREAKOUT',
                'direction': 'SELL',
                'level': val,
                'distance_pips': (val - current_price) / pip_value,
                'volume_strength': 0.5
            }

        return None

    # ==========================================================================
    # PHASE 3: SIGNAL QUALIFICATION
    # ==========================================================================

    def _qualify_signal(
        self,
        epic: str,
        df: pd.DataFrame,
        signal_data: Dict,
        profile: Dict,
        current_timestamp: Optional[datetime] = None
    ) -> SignalQualification:
        """Score signal quality on multiple factors."""

        signal_type = signal_data['type']
        direction = signal_data['direction']

        # 1. Signal Type Score (0-25)
        if signal_type == 'HVN_BOUNCE':
            type_score = self.config.hvn_bounce_base_score
        elif signal_type == 'POC_REVERSION':
            type_score = self.config.poc_reversion_base_score
        else:  # VA_BREAKOUT
            type_score = self.config.va_breakout_base_score

        # 2. Level Proximity Score (0-25)
        distance_pips = signal_data.get('distance_pips', 10)
        max_distance = 15.0  # Max distance for any score
        if distance_pips <= max_distance:
            proximity_ratio = 1.0 - (distance_pips / max_distance)
            proximity_score = int(proximity_ratio * self.config.weight_level_proximity)
        else:
            proximity_score = 0

        # 3. Volume Strength Score (0-20)
        volume_strength = signal_data.get('volume_strength', 0.5)
        volume_score = int(volume_strength * self.config.weight_volume_strength)

        # 4. Session Bonus Score (0-20)
        session, session_score = self._calc_session_score(current_timestamp, epic)

        # 5. Trend Alignment Score (0-10)
        trend_score = self._calc_trend_alignment_score(df, direction)

        # Total quality score
        quality_score = type_score + proximity_score + volume_score + session_score + trend_score

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
            signal_type=signal_type,
            signal_type_score=type_score,
            level_proximity_score=proximity_score,
            volume_strength_score=volume_score,
            session_bonus_score=session_score,
            trend_alignment_score=trend_score,
            session=session,
            level_distance_pips=distance_pips,
            volume_node_strength=volume_strength,
            key_level=signal_data.get('level'),
            rejection_reason=rejection_reason
        )

    def _calc_session_score(self, current_timestamp: Optional[datetime], epic: str) -> Tuple[str, int]:
        """Calculate session bonus score."""
        now = current_timestamp or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        hour = now.hour

        if self.config.asian_session_start <= hour < self.config.asian_session_end:
            return 'asian', self.config.get_pair_asian_session_bonus(epic)
        elif self.config.london_session_start <= hour < self.config.ny_session_start:
            return 'london', self.config.london_session_bonus
        elif self.config.ny_session_start <= hour < self.config.london_session_end:
            return 'overlap', self.config.overlap_session_bonus
        elif self.config.ny_session_start <= hour < self.config.ny_session_end:
            return 'new_york', self.config.ny_session_bonus
        else:
            return 'off_hours', 0

    def _calc_trend_alignment_score(self, df: pd.DataFrame, direction: str) -> int:
        """Calculate trend alignment score based on recent price action."""
        try:
            # Simple trend: compare last close to 20-bar SMA
            sma = df['close'].tail(20).mean()
            current = float(df['close'].iloc[-1])

            if direction == 'BUY' and current > sma:
                return self.config.weight_trend_alignment
            elif direction == 'SELL' and current < sma:
                return self.config.weight_trend_alignment
            else:
                return self.config.weight_trend_alignment // 2  # Half points for counter-trend

        except Exception:
            return 5  # Neutral

    # ==========================================================================
    # PHASE 4: SIGNAL BUILDING
    # ==========================================================================

    def _quality_to_confidence(self, quality_score: int, epic: str) -> float:
        """Map quality score (0-100) to confidence."""
        min_conf = self.config.get_pair_min_confidence(epic)
        max_conf = self.config.get_pair_max_confidence(epic)
        min_quality = self.config.get_pair_min_quality_score(epic)

        ratio = (quality_score - min_quality) / (100 - min_quality)
        ratio = max(0, min(1, ratio))

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
        profile: Dict
    ) -> Dict:
        """Build standardized signal dictionary."""
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

            'quality_score': qualification.quality_score,
            'qualification': qualification.to_dict(),

            'strategy_indicators': {
                'signal_type': qualification.signal_type,
                'session': qualification.session,
                'poc': profile['poc'],
                'vah': profile['vah'],
                'val': profile['val'],
                'level_distance_pips': qualification.level_distance_pips
            }
        }

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

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

def create_volume_profile_strategy(
    config=None,
    db_manager=None,
    logger=None
) -> VolumeProfileStrategy:
    """Factory function to create strategy instance."""
    return VolumeProfileStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager
    )
