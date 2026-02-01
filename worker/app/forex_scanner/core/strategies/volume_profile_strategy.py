#!/usr/bin/env python3
"""
Volume Profile Strategy - HVN/POC-Based Trading with Session Edge

VERSION: 3.0.0
DATE: 2026-02-01
STATUS: Active (backtest-only by default)

Strategy Architecture:
    - High Volume Nodes (HVN): Support/resistance at high volume areas
    - Point of Control (POC): Mean reversion to highest volume price
    - Value Area: Trade boundaries at 70% volume coverage
    - Session Filter: 66.7% Asian session edge discovered

Target Performance:
    - Win Rate: 55%+ (66.7% in Asian session)
    - Profit Factor: 1.8+
    - Average R:R: 1.5:1

Market Regime:
    - Primary: High volatility, breakout conditions
    - Edge: Asian session (low volatility consolidation)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta, timezone, time
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
    Configuration for Volume Profile strategy.

    Loaded from database (strategy_config.volume_profile_global_config).
    """

    # Strategy identification
    strategy_name: str = "VOLUME_PROFILE"
    version: str = "3.0.0"

    # Volume Profile Settings
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

    # Session Filters (Asian session edge)
    asian_session_boost: float = 1.15  # 15% confidence boost
    london_session_boost: float = 1.0
    ny_session_boost: float = 1.0
    asian_session_start: int = 0   # 00:00 UTC
    asian_session_end: int = 8     # 08:00 UTC
    london_session_start: int = 7  # 07:00 UTC
    london_session_end: int = 16   # 16:00 UTC
    ny_session_start: int = 12     # 12:00 UTC
    ny_session_end: int = 21       # 21:00 UTC

    # Timeframes
    primary_timeframe: str = "15m"
    profile_timeframe: str = "1h"

    # Risk Management
    fixed_stop_loss_pips: float = 15.0
    fixed_take_profit_pips: float = 22.0
    min_confidence: float = 0.50
    max_confidence: float = 0.85
    sl_buffer_pips: float = 2.0

    # Cooldown
    signal_cooldown_minutes: int = 60

    # Enabled pairs
    enabled_pairs: List[str] = field(default_factory=list)

    @classmethod
    def from_database(cls, database_url: str = None) -> 'VolumeProfileConfig':
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
                SELECT * FROM volume_profile_global_config
                WHERE is_active = TRUE
                ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()

            if row:
                field_mapping = {
                    'vp_lookback_bars': 'vp_lookback_bars',
                    'vp_value_area_percent': 'vp_value_area_percent',
                    'poc_proximity_pips': 'poc_proximity_pips',
                    'poc_reversion_enabled': 'poc_reversion_enabled',
                    'hvn_bounce_enabled': 'hvn_bounce_enabled',
                    'hvn_bounce_proximity_pips': 'hvn_bounce_proximity_pips',
                    'asian_session_boost': 'asian_session_boost',
                    'london_session_boost': 'london_session_boost',
                    'ny_session_boost': 'ny_session_boost',
                    'fixed_stop_loss_pips': 'fixed_stop_loss_pips',
                    'fixed_take_profit_pips': 'fixed_take_profit_pips',
                    'min_confidence': 'min_confidence',
                    'max_confidence': 'max_confidence',
                }

                for db_col, attr_name in field_mapping.items():
                    if db_col in row.keys() and row[db_col] is not None:
                        value = row[db_col]
                        # Convert Decimal to appropriate Python type
                        if hasattr(value, '__float__'):
                            # Check if it should be an int based on the default type
                            default_val = getattr(cls, attr_name, None)
                            if isinstance(default_val, int):
                                value = int(value)
                            else:
                                value = float(value)
                        setattr(config, attr_name, value)

            cur.close()
            conn.close()

        except Exception as e:
            logging.warning(f"Could not load Volume Profile config from database: {e}")

        return config


# ==============================================================================
# CONFIG SERVICE (Singleton)
# ==============================================================================

class VolumeProfileConfigService:
    """Singleton service for loading and caching Volume Profile configuration."""

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
        """Get configuration with caching"""
        now = datetime.now()

        if (self._config is None or
            self._last_refresh is None or
            (now - self._last_refresh).seconds > self._cache_ttl_seconds):
            self._config = VolumeProfileConfig.from_database()
            self._last_refresh = now

        return self._config


def get_volume_profile_config() -> VolumeProfileConfig:
    """Convenience function to get Volume Profile configuration"""
    return VolumeProfileConfigService.get_instance().get_config()


# ==============================================================================
# STRATEGY CLASS
# ==============================================================================

@register_strategy('VOLUME_PROFILE')
class VolumeProfileStrategy(StrategyInterface):
    """
    Volume Profile Strategy Implementation

    Uses Volume Profile analysis for entry points:
    - HVN Bounce: Trade bounces from high volume nodes
    - POC Reversion: Mean reversion to Point of Control
    - Value Area: Trade at 70% volume boundaries

    Special edge: 66.7% win rate discovered in Asian session.
    """

    def __init__(self, config=None, logger=None, db_manager=None):
        """Initialize Volume Profile Strategy"""
        self.logger = logger or logging.getLogger(__name__)
        self.db_manager = db_manager

        # Load configuration
        self.config = get_volume_profile_config()

        # Cooldown tracking
        self._cooldowns: Dict[str, datetime] = {}
        self._pending_rejections: List[Dict] = []

        # Volume profile cache
        self._profile_cache: Dict[str, Dict] = {}
        self._cache_expiry: Dict[str, datetime] = {}

        self.logger.info(f"Volume Profile Strategy v{self.config.version} initialized")

    @property
    def strategy_name(self) -> str:
        return "VOLUME_PROFILE"

    def get_required_timeframes(self) -> List[str]:
        return [
            self.config.profile_timeframe,
            self.config.primary_timeframe
        ]

    def detect_signal(
        self,
        df_trigger: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        epic: str = "",
        pair: str = "",
        df_entry: pd.DataFrame = None,
        current_timestamp: datetime = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        Detect trading signal using Volume Profile analysis.

        Args:
            current_timestamp: For backtest mode - use this instead of datetime.now()
        """
        df = df_trigger if df_trigger is not None else df_entry

        if df is None or len(df) < self.config.vp_lookback_bars:
            self.logger.debug(f"[VOLUME_PROFILE] Insufficient data for {epic}")
            return None

        # Check enabled pairs
        if self.config.enabled_pairs and epic not in self.config.enabled_pairs:
            return None

        # Check cooldown (use backtest timestamp if provided)
        if not self._check_cooldown(epic, current_timestamp):
            return None

        # Get current session for boost (use backtest timestamp if provided)
        session = self._get_current_session(current_timestamp)
        session_boost = self._get_session_boost(session)

        # Calculate Volume Profile
        profile = self._calculate_volume_profile(df)
        if profile is None:
            return None

        # Get current price
        latest = df.iloc[-1]
        current_price = float(latest['close'])

        # Check for trading signals
        signal = None

        # 1. Check HVN Bounce
        if self.config.hvn_bounce_enabled:
            signal = self._check_hvn_bounce(profile, current_price, epic)

        # 2. Check POC Reversion
        if signal is None and self.config.poc_reversion_enabled:
            signal = self._check_poc_reversion(profile, current_price, epic)

        # 3. Check Value Area Breakout
        if signal is None and self.config.va_breakout_enabled:
            signal = self._check_va_breakout(profile, current_price, df, epic)

        if signal is None:
            return None

        # Apply session boost
        base_confidence = signal['confidence']
        boosted_confidence = min(base_confidence * session_boost, self.config.max_confidence)

        # Build final signal
        result = self._build_signal(
            epic=epic,
            pair=pair,
            direction=signal['direction'],
            entry_price=current_price,
            sl_pips=self.config.fixed_stop_loss_pips,
            tp_pips=self.config.fixed_take_profit_pips,
            confidence=boosted_confidence,
            indicators={
                'signal_type': signal['type'],
                'session': session,
                'session_boost': session_boost,
                'poc': profile['poc'],
                'vah': profile['vah'],
                'val': profile['val']
            }
        )

        self._set_cooldown(epic, current_timestamp)

        self.logger.info(
            f"[VOLUME_PROFILE] ✅ {signal['type']}: {signal['direction']} {epic} "
            f"(session={session}, boost={session_boost:.2f}, conf={boosted_confidence:.2f})"
        )

        return result

    # =========================================================================
    # VOLUME PROFILE CALCULATIONS
    # =========================================================================

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate Volume Profile including POC, VAH, VAL, and HVNs"""
        try:
            lookback = self.config.vp_lookback_bars
            data = df.tail(lookback).copy()

            if len(data) < 20:
                return None

            # Get price range
            high = float(data['high'].max())
            low = float(data['low'].min())
            price_range = high - low

            if price_range <= 0:
                return None

            # Create price bins (use typical price)
            n_bins = 50
            data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3

            # Use tick volume if available, otherwise simulate
            if 'volume' in data.columns and data['volume'].sum() > 0:
                volume = data['volume']
            else:
                # Simulate volume using range as proxy
                volume = (data['high'] - data['low']).abs()

            # Bin volumes by price level
            price_bins = np.linspace(low, high, n_bins + 1)
            volume_by_price = np.zeros(n_bins)

            for i, row in data.iterrows():
                tp = row['typical_price']
                vol = volume.loc[i] if hasattr(volume, 'loc') else 1.0

                bin_idx = int((tp - low) / price_range * (n_bins - 1))
                bin_idx = max(0, min(n_bins - 1, bin_idx))
                volume_by_price[bin_idx] += vol

            # Find POC (Point of Control) - highest volume price
            poc_idx = np.argmax(volume_by_price)
            poc = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

            # Calculate Value Area (70% of volume)
            total_volume = volume_by_price.sum()
            target_volume = total_volume * (self.config.vp_value_area_percent / 100)

            # Expand from POC to capture 70%
            captured_volume = volume_by_price[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx

            while captured_volume < target_volume:
                # Check which side to expand
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

            vah = (price_bins[high_idx] + price_bins[high_idx + 1]) / 2  # Value Area High
            val = (price_bins[low_idx] + price_bins[low_idx + 1]) / 2    # Value Area Low

            # Find High Volume Nodes
            hvn_threshold = np.percentile(volume_by_price, self.config.hvn_threshold_percentile)
            hvns = []
            for i in range(n_bins):
                if volume_by_price[i] >= hvn_threshold:
                    price = (price_bins[i] + price_bins[i + 1]) / 2
                    hvns.append({'price': price, 'volume': volume_by_price[i]})

            return {
                'poc': poc,
                'vah': vah,
                'val': val,
                'hvns': hvns,
                'high': high,
                'low': low
            }

        except Exception as e:
            self.logger.warning(f"Error calculating volume profile: {e}")
            return None

    def _check_hvn_bounce(
        self,
        profile: Dict,
        current_price: float,
        epic: str
    ) -> Optional[Dict]:
        """Check for HVN bounce signal"""
        pip_value = 0.01 if 'JPY' in epic else 0.0001
        proximity_price = self.config.hvn_bounce_proximity_pips * pip_value

        for hvn in profile['hvns']:
            hvn_price = hvn['price']
            distance = abs(current_price - hvn_price)

            if distance <= proximity_price:
                # Near HVN - check if bounce
                if current_price > hvn_price:
                    # Price above HVN, potential support bounce
                    return {
                        'direction': 'BUY',
                        'type': 'HVN_BOUNCE',
                        'confidence': 0.60,
                        'hvn_price': hvn_price
                    }
                else:
                    # Price below HVN, potential resistance bounce
                    return {
                        'direction': 'SELL',
                        'type': 'HVN_BOUNCE',
                        'confidence': 0.60,
                        'hvn_price': hvn_price
                    }

        return None

    def _check_poc_reversion(
        self,
        profile: Dict,
        current_price: float,
        epic: str
    ) -> Optional[Dict]:
        """Check for POC mean reversion signal"""
        pip_value = 0.01 if 'JPY' in epic else 0.0001
        poc = profile['poc']
        vah = profile['vah']
        val = profile['val']

        # Check if price is outside value area and near boundary
        proximity = self.config.poc_proximity_pips * pip_value

        if current_price >= vah - proximity and current_price > poc:
            # Price at VAH, expect reversion to POC
            return {
                'direction': 'SELL',
                'type': 'POC_REVERSION',
                'confidence': 0.55,
                'target': poc
            }
        elif current_price <= val + proximity and current_price < poc:
            # Price at VAL, expect reversion to POC
            return {
                'direction': 'BUY',
                'type': 'POC_REVERSION',
                'confidence': 0.55,
                'target': poc
            }

        return None

    def _check_va_breakout(
        self,
        profile: Dict,
        current_price: float,
        df: pd.DataFrame,
        epic: str
    ) -> Optional[Dict]:
        """Check for Value Area breakout signal"""
        pip_value = 0.01 if 'JPY' in epic else 0.0001
        proximity = self.config.va_high_proximity_pips * pip_value

        vah = profile['vah']
        val = profile['val']

        # Check recent candles for breakout confirmation
        recent_closes = df['close'].tail(3)
        prev_close = float(recent_closes.iloc[-2])

        # Breakout above VAH
        if current_price > vah + proximity and prev_close <= vah:
            return {
                'direction': 'BUY',
                'type': 'VA_BREAKOUT',
                'confidence': 0.58,
                'breakout_level': vah
            }

        # Breakdown below VAL
        if current_price < val - proximity and prev_close >= val:
            return {
                'direction': 'SELL',
                'type': 'VA_BREAKOUT',
                'confidence': 0.58,
                'breakout_level': val
            }

        return None

    # =========================================================================
    # SESSION HANDLING
    # =========================================================================

    def _get_current_session(self, current_timestamp: datetime = None) -> str:
        """Get current trading session based on UTC time"""
        now = current_timestamp if current_timestamp else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        hour = now.hour

        # Check sessions (can overlap)
        if self.config.asian_session_start <= hour < self.config.asian_session_end:
            return 'asian'
        elif self.config.london_session_start <= hour < self.config.london_session_end:
            if self.config.ny_session_start <= hour < self.config.london_session_end:
                return 'overlap'
            return 'london'
        elif self.config.ny_session_start <= hour < self.config.ny_session_end:
            return 'new_york'

        return 'off_hours'

    def _get_session_boost(self, session: str) -> float:
        """Get confidence boost for current session"""
        if session == 'asian':
            return self.config.asian_session_boost
        elif session == 'london':
            return self.config.london_session_boost
        elif session == 'new_york' or session == 'overlap':
            return self.config.ny_session_boost
        return 1.0

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
            'strategy_indicators': kwargs.get('indicators', {})
        }

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def _check_cooldown(self, epic: str, current_timestamp: datetime = None) -> bool:
        """Check if epic is on cooldown. Uses current_timestamp for backtest mode."""
        if epic not in self._cooldowns:
            return True
        # Use backtest timestamp if provided, otherwise real time
        now = current_timestamp if current_timestamp else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if now >= self._cooldowns[epic]:
            del self._cooldowns[epic]
            return True
        return False

    def _set_cooldown(self, epic: str, current_timestamp: datetime = None) -> None:
        """Set cooldown for epic. Uses current_timestamp for backtest mode."""
        # Use backtest timestamp if provided, otherwise real time
        now = current_timestamp if current_timestamp else datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._cooldowns[epic] = now + timedelta(
            minutes=self.config.signal_cooldown_minutes
        )

    def reset_cooldowns(self) -> None:
        self._cooldowns.clear()

    def flush_rejections(self) -> None:
        self._pending_rejections.clear()


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_volume_profile_strategy(
    config=None,
    db_manager=None,
    logger=None
) -> VolumeProfileStrategy:
    """Factory function to create Volume Profile strategy instance."""
    return VolumeProfileStrategy(
        config=config,
        logger=logger,
        db_manager=db_manager
    )
