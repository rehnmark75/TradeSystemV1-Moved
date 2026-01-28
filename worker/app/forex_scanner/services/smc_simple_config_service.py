"""
SMC Simple Strategy Configuration Service

Provides database-driven configuration loading with:
- In-memory caching with configurable TTL
- Last-known-good fallback when DB unavailable
- Per-pair parameter resolution (global + overrides)
- Hot reload support via configurable refresh intervals
"""

import os
import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, time as dt_time
from threading import RLock
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


@dataclass
class SMCSimpleConfig:
    """Complete configuration object for SMC Simple strategy"""

    # STRATEGY METADATA
    version: str = "2.11.0"
    strategy_name: str = "SMC_SIMPLE"
    strategy_status: str = ""

    # TIER 1: 4H DIRECTIONAL BIAS
    htf_timeframe: str = "4h"
    ema_period: int = 50
    ema_buffer_pips: float = 2.5
    require_close_beyond_ema: bool = True
    min_distance_from_ema_pips: float = 3.0

    # v2.16.0: EMA SLOPE VALIDATION (prevents counter-trend trades)
    ema_slope_validation_enabled: bool = True
    ema_slope_lookback_bars: int = 5
    ema_slope_min_atr_multiplier: float = 0.5

    # TIER 2: 15M ENTRY TRIGGER
    trigger_timeframe: str = "15m"
    swing_lookback_bars: int = 20
    swing_strength_bars: int = 2
    require_body_close_break: bool = False
    wick_tolerance_pips: float = 3.0
    volume_confirmation_enabled: bool = True
    volume_sma_period: int = 20
    volume_spike_multiplier: float = 1.2

    # DYNAMIC SWING LOOKBACK
    use_dynamic_swing_lookback: bool = True
    swing_lookback_atr_low: int = 8
    swing_lookback_atr_high: int = 15
    swing_lookback_min: int = 15
    swing_lookback_max: int = 30

    # TIER 3: 5M EXECUTION
    entry_timeframe: str = "5m"
    pullback_enabled: bool = True
    fib_pullback_min: float = 0.236
    fib_pullback_max: float = 0.700
    fib_optimal_zone_min: float = 0.382
    fib_optimal_zone_max: float = 0.618
    max_pullback_wait_bars: int = 12
    pullback_confirmation_bars: int = 2

    # MOMENTUM MODE
    momentum_mode_enabled: bool = True
    momentum_min_depth: float = -0.50
    momentum_max_depth: float = 0.0
    momentum_confidence_penalty: float = 0.05

    # ATR-BASED SWING VALIDATION
    use_atr_swing_validation: bool = True
    atr_period: int = 14
    min_swing_atr_multiplier: float = 0.25
    fallback_min_swing_pips: float = 5.0

    # MOMENTUM QUALITY FILTER
    momentum_quality_enabled: bool = True
    min_breakout_atr_ratio: float = 0.5
    min_body_percentage: float = 0.20

    # v2.18.0: ATR-BASED EXTENSION FILTER
    # Prevents chasing extended moves beyond swing break
    max_extension_atr: float = 0.50  # Max extension in ATR units (0.5 ATR = ~10-15 pips)
    max_extension_atr_enabled: bool = True  # Enable/disable ATR-based extension filter

    # v2.18.0: MOMENTUM STALENESS FILTER
    # Rejects momentum entries if swing break is too old
    momentum_staleness_enabled: bool = True
    max_momentum_staleness_bars: int = 8  # Max bars since swing break (~2 hours on 15m)

    # LIMIT ORDER CONFIGURATION
    limit_order_enabled: bool = True
    limit_expiry_minutes: int = 45
    pullback_offset_atr_factor: float = 0.2
    pullback_offset_min_pips: float = 2.0
    pullback_offset_max_pips: float = 3.0
    momentum_offset_pips: float = 3.0
    min_risk_after_offset_pips: float = 5.0
    max_sl_atr_multiplier: float = 3.0
    max_sl_absolute_pips: float = 30.0
    max_risk_after_offset_pips: float = 55.0

    # v2.17.0: CONFIDENCE-BASED ORDER ROUTING
    # High confidence + strong trend = market order (immediate entry)
    # Lower confidence = stop order (momentum confirmation)
    market_order_min_confidence: float = 0.65  # Min confidence for market order
    market_order_min_ema_slope: float = 1.0    # Min EMA slope (ATR multiple) for market order
    low_confidence_extra_offset: float = 2.0   # Extra pips for low confidence (0.50-0.54)

    # RISK MANAGEMENT
    min_rr_ratio: float = 1.5
    optimal_rr_ratio: float = 2.5
    max_rr_ratio: float = 5.0
    sl_buffer_pips: int = 6
    sl_atr_multiplier: float = 1.0
    use_atr_stop: bool = True
    min_tp_pips: int = 8
    use_swing_target: bool = True
    tp_structure_lookback: int = 50
    risk_per_trade_pct: float = 1.0

    # FIXED SL/TP OVERRIDE (per-pair configurable)
    fixed_sl_tp_override_enabled: bool = True
    fixed_stop_loss_pips: float = 9.0
    fixed_take_profit_pips: float = 15.0

    # SESSION FILTER
    session_filter_enabled: bool = True
    london_session_start: dt_time = field(default_factory=lambda: dt_time(7, 0))
    london_session_end: dt_time = field(default_factory=lambda: dt_time(16, 0))
    ny_session_start: dt_time = field(default_factory=lambda: dt_time(12, 0))
    ny_session_end: dt_time = field(default_factory=lambda: dt_time(21, 0))
    allowed_sessions: List[str] = field(default_factory=lambda: ['london', 'new_york', 'overlap'])
    block_asian_session: bool = True

    # SIGNAL LIMITS
    max_concurrent_signals: int = 3
    signal_cooldown_hours: int = 3

    # ADAPTIVE COOLDOWN
    adaptive_cooldown_enabled: bool = True
    base_cooldown_hours: float = 2.0
    cooldown_after_win_multiplier: float = 0.5
    cooldown_after_loss_multiplier: float = 1.5
    consecutive_loss_penalty_hours: float = 1.0
    max_consecutive_losses_before_block: int = 3
    consecutive_loss_block_hours: float = 8.0
    win_rate_lookback_trades: int = 20
    high_win_rate_threshold: float = 0.60
    low_win_rate_threshold: float = 0.40
    critical_win_rate_threshold: float = 0.30
    high_win_rate_cooldown_reduction: float = 0.25
    low_win_rate_cooldown_increase: float = 0.50
    high_volatility_atr_multiplier: float = 1.5
    volatility_cooldown_adjustment: float = 0.30
    strong_trend_cooldown_reduction: float = 0.30
    session_change_reset_cooldown: bool = True
    min_cooldown_hours: float = 1.0
    max_cooldown_hours: float = 12.0

    # CONFIDENCE SCORING
    min_confidence_threshold: float = 0.48
    max_confidence_threshold: float = 0.75
    high_confidence_threshold: float = 0.75
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        "ema_alignment": 0.20,
        "swing_break_quality": 0.20,
        "volume_strength": 0.20,
        "pullback_quality": 0.20,
        "rr_ratio": 0.20,
    })

    # VOLUME FILTER
    min_volume_ratio: float = 0.50
    volume_filter_enabled: bool = True
    allow_no_volume_data: bool = True

    # DYNAMIC CONFIDENCE THRESHOLDS
    volume_adjusted_confidence_enabled: bool = True
    high_volume_threshold: float = 0.70
    atr_adjusted_confidence_enabled: bool = True
    low_atr_threshold: float = 0.0004
    high_atr_threshold: float = 0.0008
    ema_distance_adjusted_confidence_enabled: bool = True
    near_ema_threshold_pips: float = 20.0
    far_ema_threshold_pips: float = 30.0

    # MACD ALIGNMENT FILTER
    macd_alignment_filter_enabled: bool = True
    macd_alignment_mode: str = 'momentum'
    macd_min_strength: float = 0.0

    # SWING PROXIMITY VALIDATION (v2.15.0)
    # Prevents entries too close to opposing swing levels (exhaustion zones)
    # Based on trade log analysis: 65% of losing trades were at wrong swing levels
    swing_proximity_enabled: bool = True
    swing_proximity_min_distance_pips: int = 12  # Based on trade analysis: 10-15 pips optimal
    swing_proximity_strict_mode: bool = True  # Reject (True) vs confidence penalty (False)
    swing_proximity_resistance_buffer: float = 1.0  # Multiplier for resistance distance
    swing_proximity_support_buffer: float = 1.0  # Multiplier for support distance
    swing_proximity_lookback_swings: int = 5  # Number of recent swings to check

    # LOGGING & DEBUG
    enable_debug_logging: bool = True
    log_rejected_signals: bool = True
    log_swing_detection: bool = False
    log_ema_checks: bool = False

    # REJECTION TRACKING
    rejection_tracking_enabled: bool = True
    rejection_batch_size: int = 50
    rejection_log_to_console: bool = False
    rejection_retention_days: int = 90

    # BACKTEST SETTINGS
    backtest_spread_pips: float = 1.5
    backtest_slippage_pips: float = 0.5

    # SCALP MODE CONFIGURATION
    # High-frequency trading mode with 5 pip TP targets
    scalp_mode_enabled: bool = False
    scalp_tp_pips: float = 5.0
    scalp_sl_pips: float = 5.0
    scalp_max_spread_pips: float = 1.0
    scalp_htf_timeframe: str = "15m"
    scalp_trigger_timeframe: str = "5m"
    scalp_entry_timeframe: str = "1m"
    scalp_ema_period: int = 20
    scalp_min_confidence: float = 0.30
    scalp_disable_ema_slope_validation: bool = True
    scalp_disable_swing_proximity: bool = True
    scalp_disable_volume_filter: bool = True
    scalp_disable_macd_filter: bool = True
    scalp_use_momentum_only: bool = True
    scalp_momentum_min_depth: float = -0.30
    scalp_fib_pullback_min: float = 0.0
    scalp_fib_pullback_max: float = 1.0
    scalp_micro_pullback_lookback: int = 10  # v2.19.0: bars to lookback for micro-pullback on 1m
    scalp_cooldown_minutes: int = 15
    scalp_require_tight_spread: bool = True
    scalp_swing_lookback_bars: int = 5
    scalp_range_position_threshold: float = 0.80
    scalp_enable_claude_ai: bool = True
    scalp_use_market_orders: bool = True
    scalp_use_limit_orders: bool = False  # v3.3.0: Use LIMIT orders (better price) instead of STOP orders (momentum)

    # SCALP REVERSAL OVERRIDE (counter-trend)
    scalp_reversal_enabled: bool = True
    scalp_reversal_min_runway_pips: float = 15.0
    scalp_reversal_min_entry_momentum: float = 0.60
    scalp_reversal_block_regimes: List[str] = field(default_factory=lambda: ['breakout'])
    scalp_reversal_block_volatility_states: List[str] = field(default_factory=lambda: ['high'])
    scalp_reversal_allow_rsi_extremes: bool = True

    # SCALP SIGNAL QUALIFICATION (v2.21.0)
    # Momentum confirmation filters to improve scalp win rate
    scalp_qualification_enabled: bool = False  # Master toggle
    scalp_qualification_mode: str = "MONITORING"  # MONITORING or ACTIVE
    scalp_min_qualification_score: float = 0.50  # 0.0-1.0, require 50% of filters to pass
    scalp_rsi_filter_enabled: bool = True  # RSI momentum filter
    scalp_two_pole_filter_enabled: bool = True  # Two-Pole oscillator filter
    scalp_macd_filter_enabled: bool = True  # MACD direction filter
    scalp_rsi_bull_min: int = 40  # Min RSI for BULL signals
    scalp_rsi_bull_max: int = 75  # Max RSI for BULL signals
    scalp_rsi_bear_min: int = 25  # Min RSI for BEAR signals
    scalp_rsi_bear_max: int = 60  # Max RSI for BEAR signals
    scalp_two_pole_bull_threshold: float = -0.30  # Two-Pole oversold threshold for BULL
    scalp_two_pole_bear_threshold: float = 0.30  # Two-Pole overbought threshold for BEAR

    # SCALP ENTRY FILTERS (v2.22.0)
    # Based on Jan 2026 trade analysis: pullback entries had 0% win rate
    # Only momentum + HTF aligned trades showed positive results
    scalp_momentum_only_filter: bool = False  # Block pullback/micro-pullback entries
    scalp_require_htf_alignment: bool = False  # Require trade direction matches htf_candle_direction
    scalp_entry_rsi_buy_max: float = 100.0  # Max RSI for BUY entries (100 = disabled)
    scalp_entry_rsi_sell_min: float = 0.0  # Min RSI for SELL entries (0 = disabled)
    scalp_min_ema_distance_pips: float = 0.0  # Min distance from EMA in pips (0 = disabled)

    # SCALP REJECTION CANDLE CONFIRMATION (v2.25.0)
    # Require entry-TF rejection candle (pin bar, engulfing, hammer) before scalp entry
    # Based on Jan 2026 analysis: MAE=0 on most losing trades means no reversal confirmation
    scalp_require_rejection_candle: bool = False  # Require rejection candle for scalp entries
    scalp_rejection_min_strength: float = 0.70  # Minimum pattern strength for rejection candle (0-1)
    scalp_use_market_on_rejection: bool = True  # Use market order when rejection candle confirmed

    # SCALP ENTRY CANDLE ALIGNMENT (v2.25.1)
    # Simpler alternative: require entry candle color matches direction (green=BUY, red=SELL)
    # This ensures immediate momentum has shifted in trade direction before entry
    scalp_require_entry_candle_alignment: bool = False  # Require entry candle aligned with direction
    scalp_use_market_on_entry_alignment: bool = True  # Use market order when entry candle aligned

    # PATTERN CONFIRMATION (v2.23.0) - Alternative triggers
    # Price action patterns to boost confidence or enable marginal entries
    pattern_confirmation_enabled: bool = False  # Master toggle
    pattern_confirmation_mode: str = "MONITORING"  # MONITORING or ACTIVE
    pattern_min_strength: float = 0.70  # Min pattern strength (0.0-1.0)
    pattern_pin_bar_enabled: bool = True  # Enable pin bar detection
    pattern_engulfing_enabled: bool = True  # Enable engulfing pattern detection
    pattern_inside_bar_enabled: bool = True  # Enable inside bar detection
    pattern_hammer_shooter_enabled: bool = True  # Enable hammer/shooting star detection
    pattern_confidence_boost: float = 0.05  # Confidence boost when pattern detected
    # v2.24.0: Pattern as alternative TIER 3 entry
    pattern_as_entry_enabled: bool = False  # Allow pattern as standalone entry
    pattern_entry_min_strength: float = 0.80  # Min strength for pattern-only entry

    # RSI DIVERGENCE (v2.23.0)
    # Detects momentum divergence for reversal confirmation
    rsi_divergence_enabled: bool = False  # Master toggle
    rsi_divergence_mode: str = "MONITORING"  # MONITORING or ACTIVE
    rsi_divergence_lookback: int = 20  # Bars to look back for divergence
    rsi_divergence_min_strength: float = 0.30  # Min divergence strength
    rsi_divergence_confidence_boost: float = 0.08  # Confidence boost when divergence detected
    # v2.24.0: Divergence as alternative TIER 3 entry
    divergence_as_entry_enabled: bool = False  # Allow divergence as standalone entry
    divergence_entry_min_strength: float = 0.50  # Min strength for divergence-only entry

    # MACD ALIGNMENT ENHANCEMENT (v2.23.0)
    # Additional MACD check for TIER 1 validation
    macd_alignment_enabled: bool = False  # Enable MACD alignment check
    macd_alignment_required: bool = False  # Reject signals if MACD not aligned
    macd_alignment_confidence_boost: float = 0.05  # Confidence boost when MACD aligned

    # ENABLED PAIRS
    enabled_pairs: List[str] = field(default_factory=lambda: [
        'CS.D.EURUSD.CEEM.IP',
        'CS.D.GBPUSD.MINI.IP',
        'CS.D.USDJPY.MINI.IP',
        'CS.D.AUDUSD.MINI.IP',
        'CS.D.USDCAD.MINI.IP',
        'CS.D.NZDUSD.MINI.IP',
        'CS.D.EURJPY.MINI.IP',
        'CS.D.AUDJPY.MINI.IP',
    ])

    # PAIR PIP VALUES
    pair_pip_values: Dict[str, float] = field(default_factory=lambda: {
        'CS.D.EURUSD.CEEM.IP': 1.0,
        'CS.D.GBPUSD.MINI.IP': 0.0001,
        'CS.D.USDJPY.MINI.IP': 0.01,
        'CS.D.USDCHF.MINI.IP': 0.0001,
        'CS.D.AUDUSD.MINI.IP': 0.0001,
        'CS.D.USDCAD.MINI.IP': 0.0001,
        'CS.D.NZDUSD.MINI.IP': 0.0001,
        'CS.D.EURJPY.MINI.IP': 0.01,
        'CS.D.GBPJPY.MINI.IP': 0.01,
        'CS.D.AUDJPY.MINI.IP': 0.01,
    })

    # Per-pair overrides cache (populated from database)
    _pair_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    loaded_at: datetime = field(default_factory=datetime.now)
    source: str = "database"  # 'database', 'cache', 'default'
    config_id: int = 0

    def get_for_pair(self, epic: str, param_name: str, default: Any = None) -> Any:
        """Get parameter value with pair-specific override if exists"""
        # Check pair overrides first
        if epic in self._pair_overrides:
            override_data = self._pair_overrides[epic]

            # Check direct field override
            if param_name in override_data and override_data[param_name] is not None:
                return override_data[param_name]

            # Check parameter_overrides JSONB
            param_overrides = override_data.get('parameter_overrides', {})
            if param_name in param_overrides:
                return param_overrides[param_name]

        # Fall back to global value
        if hasattr(self, param_name):
            return getattr(self, param_name)
        return default

    def get_pip_value(self, epic: str) -> float:
        """Get pip value for a given epic"""
        return self.pair_pip_values.get(epic, 0.0001)

    def is_pair_enabled(self, epic: str) -> bool:
        """Check if pair is enabled"""
        return epic in self.enabled_pairs

    def get_optimal_pullback_zone(self) -> tuple:
        """Get optimal Fibonacci pullback zone"""
        return (self.fib_optimal_zone_min, self.fib_optimal_zone_max)

    def is_session_allowed(self, hour_utc: int) -> bool:
        """Check if current hour is in allowed trading session"""
        if not self.session_filter_enabled:
            return True

        if self.block_asian_session and (hour_utc >= 21 or hour_utc < 7):
            return False

        return 7 <= hour_utc <= 21

    def is_asian_session_allowed(self, epic: str) -> bool:
        """Check if Asian session trading is allowed for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('allow_asian_session') is not None:
                return override['allow_asian_session']
        return not self.block_asian_session

    def is_macd_filter_enabled(self, epic: str) -> bool:
        """Check if MACD filter is enabled for a specific pair

        Per-pair overrides take precedence over global setting.
        This allows enabling MACD filter for specific pairs even when globally disabled.
        """
        # Check per-pair override first
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('macd_filter_enabled') is not None:
                return bool(override['macd_filter_enabled'])

        # Fall back to global setting
        return self.macd_alignment_filter_enabled

    def get_pair_sl_buffer(self, epic: str) -> int:
        """Get SL buffer for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('sl_buffer_pips') is not None:
                return override['sl_buffer_pips']
        return self.sl_buffer_pips

    def get_pair_min_confidence(self, epic: str) -> float:
        """Get minimum confidence threshold for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('min_confidence') is not None:
                return override['min_confidence']
        return self.min_confidence_threshold

    def get_pair_max_confidence(self, epic: str) -> float:
        """Get maximum confidence cap for a specific pair (confidence paradox filter)"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('max_confidence') is not None:
                return override['max_confidence']
        return self.max_confidence_threshold

    def get_pair_min_volume_ratio(self, epic: str) -> float:
        """Get minimum volume ratio for a specific pair"""
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('min_volume_ratio') is not None:
                return override['min_volume_ratio']
        return self.min_volume_ratio

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        """Get fixed stop loss in pips for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        Returns None if fixed_sl_tp_override_enabled is False.
        """
        if not self.fixed_sl_tp_override_enabled:
            return None
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('fixed_stop_loss_pips') is not None:
                return float(override['fixed_stop_loss_pips'])
        return self.fixed_stop_loss_pips

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        """Get fixed take profit in pips for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        Returns None if fixed_sl_tp_override_enabled is False.
        """
        if not self.fixed_sl_tp_override_enabled:
            return None
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('fixed_take_profit_pips') is not None:
                return float(override['fixed_take_profit_pips'])
        return self.fixed_take_profit_pips

    # =========================================================================
    # SWING PROXIMITY GETTERS (v2.15.1)
    # Per-pair swing proximity validation settings
    # =========================================================================

    def is_swing_proximity_enabled(self, epic: str) -> bool:
        """Check if swing proximity validation is enabled for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('swing_proximity_enabled') is not None:
                return override['swing_proximity_enabled']
        return self.swing_proximity_enabled

    def get_pair_swing_proximity_min_distance(self, epic: str) -> int:
        """Get swing proximity minimum distance in pips for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('swing_proximity_min_distance_pips') is not None:
                return int(override['swing_proximity_min_distance_pips'])
        return self.swing_proximity_min_distance_pips

    def is_swing_proximity_strict_mode(self, epic: str) -> bool:
        """Check if swing proximity strict mode is enabled for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        When True, signals are rejected. When False, confidence penalty is applied.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('swing_proximity_strict_mode') is not None:
                return override['swing_proximity_strict_mode']
        return self.swing_proximity_strict_mode

    # =========================================================================
    # EMA OVERRIDES GETTERS (v2.16.0)
    # Per-pair EMA period and slope validation settings
    # =========================================================================

    def get_pair_ema_period(self, epic: str) -> int:
        """Get EMA period for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('ema_period') is not None:
                return int(override['ema_period'])
        return self.ema_period

    def is_ema_slope_validation_enabled(self, epic: str) -> bool:
        """Check if EMA slope validation is enabled for a specific pair.

        Returns pair-specific value if set, otherwise global default.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('ema_slope_validation_enabled') is not None:
                return override['ema_slope_validation_enabled']
        return self.ema_slope_validation_enabled

    # =========================================================================
    # STOP OFFSET GETTERS (v2.17.0)
    # Per-pair stop entry offset for momentum confirmation
    # =========================================================================

    def get_pair_stop_offset(self, epic: str, entry_type: str = 'MOMENTUM') -> float:
        """Get stop entry offset in pips for a specific pair.

        Returns pair-specific value if set, otherwise global default based on entry type.
        For PULLBACK entries, returns pullback_offset_max_pips.
        For MOMENTUM entries, returns momentum_offset_pips.

        Args:
            epic: Trading pair epic
            entry_type: 'PULLBACK' or 'MOMENTUM'

        Returns:
            Stop offset in pips
        """
        # Check pair-specific override first
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('stop_offset_pips') is not None:
                return float(override['stop_offset_pips'])

        # Fall back to global default based on entry type
        if entry_type == 'PULLBACK':
            return self.pullback_offset_max_pips
        return self.momentum_offset_pips

    # =========================================================================
    # DIRECTION-AWARE GETTERS (v2.12.0)
    # These methods return direction-specific values when enabled, otherwise
    # fall back to non-directional pair overrides, then global defaults.
    # =========================================================================

    def is_direction_overrides_enabled(self, epic: str) -> bool:
        """Check if direction-specific overrides are enabled for a pair"""
        if epic in self._pair_overrides:
            return self._pair_overrides[epic].get('direction_overrides_enabled', False)
        return False

    def get_fib_pullback_min(self, epic: str, direction: str) -> float:
        """
        Get minimum Fib pullback threshold for a pair and direction.

        Priority: direction-specific -> pair parameter_overrides -> global
        Direction should be 'BULL' or 'BEAR'.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]

            # Check direction-specific if enabled
            if override.get('direction_overrides_enabled', False):
                dir_key = f'fib_pullback_min_{direction.lower()}'
                if override.get(dir_key) is not None:
                    return float(override[dir_key])

            # Check parameter_overrides JSONB
            param_overrides = override.get('parameter_overrides', {})
            if 'FIB_PULLBACK_MIN' in param_overrides:
                return float(param_overrides['FIB_PULLBACK_MIN'])

        # Fall back to global
        return self.fib_pullback_min

    def get_fib_pullback_max(self, epic: str, direction: str) -> float:
        """
        Get maximum Fib pullback threshold for a pair and direction.

        Priority: direction-specific -> pair parameter_overrides -> global
        Direction should be 'BULL' or 'BEAR'.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]

            # Check direction-specific if enabled
            if override.get('direction_overrides_enabled', False):
                dir_key = f'fib_pullback_max_{direction.lower()}'
                if override.get(dir_key) is not None:
                    return float(override[dir_key])

            # Check parameter_overrides JSONB
            param_overrides = override.get('parameter_overrides', {})
            if 'FIB_PULLBACK_MAX' in param_overrides:
                return float(param_overrides['FIB_PULLBACK_MAX'])

        # Fall back to global
        return self.fib_pullback_max

    def get_momentum_min_depth(self, epic: str, direction: str) -> float:
        """
        Get momentum minimum depth threshold for a pair and direction.

        Priority: direction-specific -> pair parameter_overrides -> global
        Direction should be 'BULL' or 'BEAR'.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]

            # Check direction-specific if enabled
            if override.get('direction_overrides_enabled', False):
                dir_key = f'momentum_min_depth_{direction.lower()}'
                if override.get(dir_key) is not None:
                    return float(override[dir_key])

            # Check parameter_overrides JSONB
            param_overrides = override.get('parameter_overrides', {})
            if 'MOMENTUM_MIN_DEPTH' in param_overrides:
                return float(param_overrides['MOMENTUM_MIN_DEPTH'])

        # Fall back to global
        return self.momentum_min_depth

    def get_min_volume_ratio_directional(self, epic: str, direction: str) -> float:
        """
        Get minimum volume ratio for a pair and direction.

        Priority: direction-specific -> pair min_volume_ratio -> global
        Direction should be 'BULL' or 'BEAR'.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]

            # Check direction-specific if enabled
            if override.get('direction_overrides_enabled', False):
                dir_key = f'min_volume_ratio_{direction.lower()}'
                if override.get(dir_key) is not None:
                    return float(override[dir_key])

            # Fall back to non-directional pair override
            if override.get('min_volume_ratio') is not None:
                return float(override['min_volume_ratio'])

        # Fall back to global
        return self.min_volume_ratio

    def get_min_confidence_directional(self, epic: str, direction: str) -> float:
        """
        Get minimum confidence threshold for a pair and direction.

        Priority: direction-specific -> pair min_confidence -> global
        Direction should be 'BULL' or 'BEAR'.
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]

            # Check direction-specific if enabled
            if override.get('direction_overrides_enabled', False):
                dir_key = f'min_confidence_{direction.lower()}'
                if override.get(dir_key) is not None:
                    return float(override[dir_key])

            # Fall back to non-directional pair override
            if override.get('min_confidence') is not None:
                return float(override['min_confidence'])

        # Fall back to global
        return self.min_confidence_threshold

    def get_dynamic_confidence(
        self,
        epic: str,
        volume_ratio: Optional[float] = None,
        atr_value: Optional[float] = None,
        ema_distance_pips: Optional[float] = None
    ) -> float:
        """
        Get dynamic confidence threshold based on market conditions.
        Returns the effective confidence threshold after applying all adjustments.
        """
        base_confidence = self.get_pair_min_confidence(epic)
        override = self._pair_overrides.get(epic, {})

        # Volume adjustment
        if (self.volume_adjusted_confidence_enabled and
                volume_ratio is not None and
                volume_ratio >= self.high_volume_threshold):
            high_vol_conf = override.get('high_volume_confidence')
            if high_vol_conf is not None:
                base_confidence = min(base_confidence, high_vol_conf)

        # ATR adjustment (low ATR = calmer market = lower threshold)
        if self.atr_adjusted_confidence_enabled and atr_value is not None:
            if atr_value < self.low_atr_threshold:
                low_atr_conf = override.get('low_atr_confidence')
                if low_atr_conf is not None:
                    base_confidence = min(base_confidence, low_atr_conf)
            elif atr_value >= self.high_atr_threshold:
                high_atr_conf = override.get('high_atr_confidence')
                if high_atr_conf is not None:
                    base_confidence = max(base_confidence, high_atr_conf)

        # EMA distance adjustment
        if self.ema_distance_adjusted_confidence_enabled and ema_distance_pips is not None:
            if ema_distance_pips < self.near_ema_threshold_pips:
                near_ema_conf = override.get('near_ema_confidence')
                if near_ema_conf is not None:
                    base_confidence = min(base_confidence, near_ema_conf)
            elif ema_distance_pips >= self.far_ema_threshold_pips:
                far_ema_conf = override.get('far_ema_confidence')
                if far_ema_conf is not None:
                    base_confidence = max(base_confidence, far_ema_conf)

        return base_confidence

    def should_block_signal(self, epic: str, signal_data: dict) -> tuple:
        """
        Check if a signal should be blocked based on pair-specific conditions.

        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        if epic not in self._pair_overrides:
            return False, ""

        override = self._pair_overrides[epic]
        blocking_config = override.get('blocking_conditions')
        if not blocking_config or not blocking_config.get('enabled', False):
            return False, ""

        conditions = blocking_config.get('conditions', {})
        blocking_logic = blocking_config.get('blocking_logic', 'any')
        block_reasons = []

        # Check EMA distance
        max_ema = conditions.get('max_ema_distance_pips')
        if max_ema is not None:
            ema_distance = signal_data.get('ema_distance_pips', 0)
            if ema_distance > max_ema:
                block_reasons.append(f"EMA distance {ema_distance:.1f} > {max_ema} pips")

        # Check volume confirmation
        if conditions.get('require_volume_confirmation', False):
            volume_confirmed = signal_data.get('volume_confirmed', False)
            if not volume_confirmed:
                block_reasons.append("No volume confirmation (required for this pair)")

        # Check momentum without volume
        if conditions.get('block_momentum_without_volume', False):
            pullback_depth = signal_data.get('pullback_depth', 0)
            volume_confirmed = signal_data.get('volume_confirmed', False)
            if pullback_depth < 0 and not volume_confirmed:
                block_reasons.append(f"Momentum entry without volume")

        # Check confidence override
        min_conf = conditions.get('min_confidence_override')
        if min_conf is not None:
            confidence = signal_data.get('confidence_score', 0)
            if confidence < min_conf:
                block_reasons.append(f"Confidence {confidence:.1%} < {min_conf:.1%}")

        # Determine if signal should be blocked
        if blocking_logic == 'any' and block_reasons:
            return True, f"Blocked: {'; '.join(block_reasons)}"
        elif blocking_logic == 'all' and len(block_reasons) == len([c for c in conditions.values() if c]):
            return True, f"Blocked (all conditions): {'; '.join(block_reasons)}"

        return False, ""

    # =========================================================================
    # SCALP TIER GETTERS (v2.21.0)
    # Per-pair scalp mode tier settings for optimized scalping parameters
    # =========================================================================

    def get_pair_scalp_ema_period(self, epic: str) -> Optional[int]:
        """
        Get per-pair scalp EMA period override.

        Returns:
            Per-pair scalp EMA period if set, None otherwise (use global scalp_ema_period)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_ema_period') is not None:
                return int(override['scalp_ema_period'])
        return None

    def get_pair_scalp_swing_lookback(self, epic: str) -> Optional[int]:
        """
        Get per-pair scalp swing lookback bars override.

        Returns:
            Per-pair scalp swing lookback if set, None otherwise (use global scalp_swing_lookback_bars)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_swing_lookback_bars') is not None:
                return int(override['scalp_swing_lookback_bars'])
        return None

    def get_pair_scalp_limit_offset(self, epic: str) -> Optional[float]:
        """
        Get per-pair scalp limit order offset override.

        Returns:
            Per-pair scalp limit offset in pips if set, None otherwise (use global)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_limit_offset_pips') is not None:
                return float(override['scalp_limit_offset_pips'])
        return None

    def get_pair_scalp_htf_timeframe(self, epic: str) -> Optional[str]:
        """
        Get per-pair scalp HTF timeframe override.

        Returns:
            Per-pair scalp HTF timeframe (e.g., '15m', '30m') if set, None otherwise
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_htf_timeframe'):
                return str(override['scalp_htf_timeframe'])
        return None

    def get_pair_scalp_trigger_timeframe(self, epic: str) -> Optional[str]:
        """
        Get per-pair scalp trigger timeframe override.

        Returns:
            Per-pair scalp trigger timeframe (e.g., '5m', '15m') if set, None otherwise
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_trigger_timeframe'):
                return str(override['scalp_trigger_timeframe'])
        return None

    def get_pair_scalp_entry_timeframe(self, epic: str) -> Optional[str]:
        """
        Get per-pair scalp entry timeframe override.

        Returns:
            Per-pair scalp entry timeframe (e.g., '1m', '5m') if set, None otherwise
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_entry_timeframe'):
                return str(override['scalp_entry_timeframe'])
        return None

    def get_pair_scalp_min_confidence(self, epic: str) -> Optional[float]:
        """
        Get per-pair scalp minimum confidence threshold override.

        Returns:
            Per-pair scalp min confidence if set, None otherwise (use global scalp_min_confidence)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_min_confidence') is not None:
                return float(override['scalp_min_confidence'])
        return None

    def get_pair_scalp_cooldown_minutes(self, epic: str) -> Optional[int]:
        """
        Get per-pair scalp cooldown minutes override.

        Returns:
            Per-pair scalp cooldown in minutes if set, None otherwise (use global)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_cooldown_minutes') is not None:
                return int(override['scalp_cooldown_minutes'])
        return None

    def get_pair_scalp_swing_break_tolerance(self, epic: str) -> Optional[float]:
        """
        Get per-pair scalp swing break tolerance override.

        Returns:
            Per-pair scalp swing break tolerance in pips if set, None otherwise
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_swing_break_tolerance_pips') is not None:
                return float(override['scalp_swing_break_tolerance_pips'])
        return None

    def get_pair_scalp_qualification_mode(self, epic: str) -> Optional[str]:
        """
        Get per-pair scalp qualification mode override.

        Returns:
            Per-pair scalp qualification mode ('ACTIVE' or 'MONITORING') if set,
            None otherwise (use global scalp_qualification_mode)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_qualification_mode'):
                return str(override['scalp_qualification_mode']).upper()
        return None

    def get_pair_scalp_fib_pullback_min(self, epic: str) -> Optional[float]:
        """
        Get per-pair scalp micro-pullback minimum threshold override.

        Returns:
            Per-pair scalp fib pullback min (e.g., 0.15 for 15%) if set,
            None otherwise (use global scalp_fib_pullback_min)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_fib_pullback_min') is not None:
                return float(override['scalp_fib_pullback_min'])
        return None

    def get_pair_scalp_require_rejection_candle(self, epic: str) -> Optional[bool]:
        """
        Get per-pair scalp rejection candle requirement override.

        v2.25.0: Require entry-TF rejection candle before scalp entry.
        Based on Jan 2026 analysis - helps USDJPY, EURUSD, AUDJPY but hurts GBPUSD.

        Returns:
            Per-pair scalp_require_rejection_candle if set,
            None otherwise (use global scalp_require_rejection_candle)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_require_rejection_candle') is not None:
                return bool(override['scalp_require_rejection_candle'])
        return None

    def get_pair_scalp_require_entry_candle_alignment(self, epic: str) -> Optional[bool]:
        """
        Get per-pair entry candle alignment requirement override.

        v2.25.1: Simpler alternative to rejection candle - requires entry candle
        color to match trade direction (green for BUY, red for SELL).

        Returns:
            Per-pair scalp_require_entry_candle_alignment if set,
            None otherwise (use global scalp_require_entry_candle_alignment)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_require_entry_candle_alignment') is not None:
                return bool(override['scalp_require_entry_candle_alignment'])
        return None

    def get_pair_scalp_sl(self, epic: str) -> Optional[float]:
        """
        Get per-pair scalp stop loss override (ATR-optimized).

        v2.29.0: Per-pair scalp SL values calculated from ATR analysis.
        Each pair has an optimal SL based on its historical volatility.

        Returns:
            Per-pair scalp_sl_pips if set, None otherwise (use global scalp_sl_pips)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_sl_pips') is not None:
                return float(override['scalp_sl_pips'])
        return None

    def get_pair_scalp_tp(self, epic: str) -> Optional[float]:
        """
        Get per-pair scalp take profit override (ATR-optimized).

        v2.29.0: Per-pair scalp TP values calculated from ATR analysis.
        Each pair has an optimal TP based on its historical volatility (2.5:1 R:R).

        Returns:
            Per-pair scalp_tp_pips if set, None otherwise (use global scalp_tp_pips)
        """
        if epic in self._pair_overrides:
            override = self._pair_overrides[epic]
            if override.get('scalp_tp_pips') is not None:
                return float(override['scalp_tp_pips'])
        return None

    def get_effective_scalp_config(self, epic: str) -> dict:
        """
        Get effective scalp configuration for a pair, merging per-pair overrides with globals.

        Returns:
            Dict with effective scalp settings (per-pair if set, otherwise global)
        """
        return {
            'ema_period': self.get_pair_scalp_ema_period(epic) or self.scalp_ema_period,
            'swing_lookback_bars': self.get_pair_scalp_swing_lookback(epic) or self.scalp_swing_lookback_bars,
            'limit_offset_pips': self.get_pair_scalp_limit_offset(epic) or getattr(self, 'scalp_limit_offset_pips', 1.0),
            'htf_timeframe': self.get_pair_scalp_htf_timeframe(epic) or self.scalp_htf_timeframe,
            'trigger_timeframe': self.get_pair_scalp_trigger_timeframe(epic) or self.scalp_trigger_timeframe,
            'entry_timeframe': self.get_pair_scalp_entry_timeframe(epic) or self.scalp_entry_timeframe,
            'min_confidence': self.get_pair_scalp_min_confidence(epic) or self.scalp_min_confidence,
            'cooldown_minutes': self.get_pair_scalp_cooldown_minutes(epic) or self.scalp_cooldown_minutes,
            'swing_break_tolerance_pips': self.get_pair_scalp_swing_break_tolerance(epic) or getattr(self, 'scalp_swing_break_tolerance_pips', 0.5),
            'qualification_mode': self.get_pair_scalp_qualification_mode(epic) or self.scalp_qualification_mode,
            'fib_pullback_min': self.get_pair_scalp_fib_pullback_min(epic) or self.scalp_fib_pullback_min,
            # v2.25.0: Rejection candle confirmation
            'require_rejection_candle': self.get_pair_scalp_require_rejection_candle(epic) if self.get_pair_scalp_require_rejection_candle(epic) is not None else self.scalp_require_rejection_candle,
            # v2.25.1: Entry candle alignment (simpler alternative)
            'require_entry_candle_alignment': self.get_pair_scalp_require_entry_candle_alignment(epic) if self.get_pair_scalp_require_entry_candle_alignment(epic) is not None else self.scalp_require_entry_candle_alignment,
        }

    # =========================================================================
    # SCALP PAIR FILTER GETTERS (v2.31.0)
    # Per-pair scalp trade filters based on Jan 2026 NZDUSD trade analysis.
    # These filters are stored in parameter_overrides JSONB and can be
    # configured independently per pair.
    # =========================================================================

    def get_pair_scalp_min_efficiency_ratio(self, epic: str) -> Optional[float]:
        """
        Get per-pair minimum efficiency ratio for scalp trades.

        Filters out choppy/ranging markets where tight stops get whipsawed.
        Recommended: 0.12 based on NZDUSD analysis (winner had 0.1475, losers < 0.10).

        Returns:
            Minimum efficiency ratio threshold if set, None otherwise (filter disabled)
        """
        value = self.get_for_pair(epic, 'scalp_min_efficiency_ratio')
        return float(value) if value is not None else None

    def get_pair_scalp_require_trending_regime(self, epic: str) -> bool:
        """
        Get per-pair trending regime requirement for scalp trades.

        Only allows scalp entries when market_regime_detected == 'trending'.
        Based on NZDUSD analysis: winner was in trending, 2/3 losers in ranging.

        Returns:
            True if trending regime required, False otherwise (default)
        """
        value = self.get_for_pair(epic, 'scalp_require_trending_regime')
        return bool(value) if value is not None else False

    def get_pair_scalp_session_start_hour(self, epic: str) -> Optional[int]:
        """
        Get per-pair scalp session start hour (UTC).

        Filters out low-liquidity hours where whipsaw risk is higher.
        Recommended: 14 (2 PM UTC = London afternoon / NY open).

        Returns:
            Session start hour (0-23 UTC) if set, None otherwise (filter disabled)
        """
        value = self.get_for_pair(epic, 'scalp_session_start_hour')
        return int(value) if value is not None else None

    def get_pair_scalp_session_end_hour(self, epic: str) -> Optional[int]:
        """
        Get per-pair scalp session end hour (UTC).

        Filters out low-liquidity hours where whipsaw risk is higher.
        Recommended: 22 (10 PM UTC = NY close).

        Returns:
            Session end hour (0-23 UTC) if set, None otherwise (filter disabled)
        """
        value = self.get_for_pair(epic, 'scalp_session_end_hour')
        return int(value) if value is not None else None

    def get_pair_scalp_require_macd_alignment(self, epic: str) -> bool:
        """
        Get per-pair MACD alignment requirement for scalp trades.

        Requires MACD histogram to align with trade direction:
        - BUY: histogram > -0.0001
        - SELL: histogram < +0.0001

        Returns:
            True if MACD alignment required, False otherwise (default)
        """
        value = self.get_for_pair(epic, 'scalp_require_macd_alignment')
        return bool(value) if value is not None else False

    def get_pair_scalp_require_ema_stack_alignment(self, epic: str) -> bool:
        """
        Get per-pair EMA stack alignment requirement for scalp trades.

        Requires EMA stack order to match trade direction:
        - BUY: ema_stack_order == 'bullish'
        - SELL: ema_stack_order == 'bearish'

        Returns:
            True if EMA stack alignment required, False otherwise (default)
        """
        value = self.get_for_pair(epic, 'scalp_require_ema_stack_alignment')
        return bool(value) if value is not None else False

    def get_pair_scalp_ema_buffer_pips(self, epic: str) -> Optional[float]:
        """
        Get per-pair EMA buffer for scalp trades.

        Some pairs like NZDUSD tend to hover close to the EMA, so they need
        a smaller buffer to generate signals. Default scalp buffer is 1.0 pip.

        Returns:
            Custom EMA buffer in pips if configured, None otherwise
        """
        value = self.get_for_pair(epic, 'scalp_ema_buffer_pips')
        return float(value) if value is not None else None

    def check_macd_alignment(
        self,
        signal_type: str,
        macd_line: float,
        macd_signal: float,
        macd_histogram: float = None
    ) -> tuple:
        """
        Check if MACD momentum aligns with trade direction.

        Returns:
            Tuple of (is_aligned: bool, reason: str)
        """
        if self.macd_alignment_mode == 'histogram':
            if macd_histogram is None:
                return True, "No histogram data"
            if signal_type == 'BULL':
                aligned = macd_histogram > 0
                direction = 'bullish' if macd_histogram > 0 else 'bearish'
            else:
                aligned = macd_histogram < 0
                direction = 'bearish' if macd_histogram < 0 else 'bullish'
            reason = f"MACD histogram {direction} ({macd_histogram:.6f})"
        else:
            # Momentum mode (default)
            if macd_line is None or macd_signal is None:
                return True, "No MACD data"

            macd_diff = macd_line - macd_signal
            min_strength = self.macd_min_strength

            if signal_type == 'BULL':
                aligned = macd_diff > min_strength
                direction = "bullish" if macd_diff > 0 else "bearish"
            else:
                aligned = macd_diff < -min_strength
                direction = "bearish" if macd_diff < 0 else "bullish"

            reason = f"MACD momentum {direction} (diff={macd_diff:.6f})"

        return aligned, reason


class SMCSimpleConfigService:
    """
    Database-driven configuration service with in-memory caching.

    Features:
    - Loads config from strategy_config database
    - Caches in memory with configurable TTL
    - Falls back to last-known-good config if DB unavailable
    - Thread-safe operations
    - Hot reload before each scan cycle
    """

    def __init__(
        self,
        database_url: str = None,
        cache_ttl_seconds: int = 120,
        enable_hot_reload: bool = True
    ):
        self.database_url = database_url or self._get_default_database_url()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.enable_hot_reload = enable_hot_reload

        # Thread-safe cache
        self._lock = RLock()
        self._cached_config: Optional[SMCSimpleConfig] = None
        self._cache_timestamp: Optional[datetime] = None
        self._last_known_good: Optional[SMCSimpleConfig] = None

        # Load initial configuration
        self._load_initial_config()

    def _get_default_database_url(self) -> str:
        """Get database URL from environment"""
        return os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        finally:
            if conn:
                conn.close()

    def _load_initial_config(self):
        """Load initial configuration on service startup"""
        try:
            self._load_from_database()
            logger.info("SMC Simple config service initialized from database")
        except Exception as e:
            logger.warning(f"Failed to load initial config from database: {e}")
            # Create default config
            self._cached_config = SMCSimpleConfig()
            self._cached_config.source = 'default'
            self._cache_timestamp = datetime.now()
            logger.info("Using default SMC Simple config")

    def get_config(self, force_refresh: bool = False) -> SMCSimpleConfig:
        """
        Get current configuration, refreshing from DB if needed.

        Args:
            force_refresh: Force reload from database

        Returns:
            SMCSimpleConfig object with all parameters
        """
        with self._lock:
            if force_refresh or self._should_refresh():
                try:
                    self._load_from_database()
                except Exception as e:
                    logger.error(f"Failed to load config from database: {e}")
                    # Fall back to last-known-good
                    if self._last_known_good is not None:
                        logger.warning("Using last-known-good configuration")
                        self._cached_config = copy.deepcopy(self._last_known_good)
                        self._cached_config.source = 'cache'
                        self._cache_timestamp = datetime.now()

            if self._cached_config is None:
                raise RuntimeError("No configuration available - database required")

            return self._cached_config

    def _should_refresh(self) -> bool:
        """Check if cache has expired"""
        if not self.enable_hot_reload:
            return self._cached_config is None

        if self._cache_timestamp is None:
            return True

        return datetime.now() - self._cache_timestamp > self.cache_ttl

    def _load_from_database(self):
        """Load configuration from database"""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Load global config
                cur.execute("""
                    SELECT * FROM smc_simple_global_config
                    WHERE is_active = TRUE
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                global_row = cur.fetchone()

                if global_row is None:
                    raise ValueError("No active configuration found in database")

                # Load pair overrides
                cur.execute("""
                    SELECT * FROM smc_simple_pair_overrides
                    WHERE config_id = %s AND is_enabled = TRUE
                """, (global_row['id'],))
                override_rows = cur.fetchall()

            # Build config object
            config = self._build_config_from_rows(global_row, override_rows)
            config.source = 'database'
            config.loaded_at = datetime.now()
            config.config_id = global_row['id']

            # Update cache
            self._cached_config = config
            self._cache_timestamp = datetime.now()

            # Update last-known-good
            self._last_known_good = copy.deepcopy(config)

            logger.info(f"Loaded SMC Simple config v{config.version} from database")

    def _build_config_from_rows(
        self,
        global_row: Dict,
        override_rows: List[Dict]
    ) -> SMCSimpleConfig:
        """Build SMCSimpleConfig from database rows"""
        config = SMCSimpleConfig()

        # Map database columns to config attributes (snake_case matches)
        direct_mappings = [
            'version', 'strategy_name', 'strategy_status',
            'htf_timeframe', 'ema_period', 'ema_buffer_pips',
            'require_close_beyond_ema', 'min_distance_from_ema_pips',
            'ema_slope_validation_enabled', 'ema_slope_lookback_bars', 'ema_slope_min_atr_multiplier',
            'trigger_timeframe', 'swing_lookback_bars', 'swing_strength_bars',
            'require_body_close_break', 'wick_tolerance_pips',
            'volume_confirmation_enabled', 'volume_sma_period', 'volume_spike_multiplier',
            'use_dynamic_swing_lookback', 'swing_lookback_atr_low', 'swing_lookback_atr_high',
            'swing_lookback_min', 'swing_lookback_max',
            'entry_timeframe', 'pullback_enabled',
            'fib_pullback_min', 'fib_pullback_max',
            'fib_optimal_zone_min', 'fib_optimal_zone_max',
            'max_pullback_wait_bars', 'pullback_confirmation_bars',
            'momentum_mode_enabled', 'momentum_min_depth', 'momentum_max_depth',
            'momentum_confidence_penalty',
            'use_atr_swing_validation', 'atr_period', 'min_swing_atr_multiplier',
            'fallback_min_swing_pips',
            'momentum_quality_enabled', 'min_breakout_atr_ratio', 'min_body_percentage',
            'max_extension_atr', 'max_extension_atr_enabled',
            'momentum_staleness_enabled', 'max_momentum_staleness_bars',
            'limit_order_enabled', 'limit_expiry_minutes',
            'pullback_offset_atr_factor', 'pullback_offset_min_pips', 'pullback_offset_max_pips',
            'momentum_offset_pips', 'min_risk_after_offset_pips',
            'max_sl_atr_multiplier', 'max_sl_absolute_pips', 'max_risk_after_offset_pips',
            'market_order_min_confidence', 'market_order_min_ema_slope', 'low_confidence_extra_offset',
            'min_rr_ratio', 'optimal_rr_ratio', 'max_rr_ratio',
            'sl_buffer_pips', 'sl_atr_multiplier', 'use_atr_stop',
            'min_tp_pips', 'use_swing_target', 'tp_structure_lookback', 'risk_per_trade_pct',
            'fixed_sl_tp_override_enabled', 'fixed_stop_loss_pips', 'fixed_take_profit_pips',
            'session_filter_enabled', 'block_asian_session',
            'max_concurrent_signals', 'signal_cooldown_hours',
            'min_confidence_threshold', 'max_confidence_threshold', 'high_confidence_threshold',
            'min_volume_ratio', 'volume_filter_enabled', 'allow_no_volume_data',
            'volume_adjusted_confidence_enabled', 'high_volume_threshold',
            'atr_adjusted_confidence_enabled', 'low_atr_threshold', 'high_atr_threshold',
            'ema_distance_adjusted_confidence_enabled',
            'near_ema_threshold_pips', 'far_ema_threshold_pips',
            'macd_alignment_filter_enabled', 'macd_alignment_mode', 'macd_min_strength',
            'swing_proximity_enabled', 'swing_proximity_min_distance_pips',
            'swing_proximity_strict_mode', 'swing_proximity_resistance_buffer',
            'swing_proximity_support_buffer', 'swing_proximity_lookback_swings',
            'enable_debug_logging', 'log_rejected_signals', 'log_swing_detection', 'log_ema_checks',
            'rejection_tracking_enabled', 'rejection_batch_size',
            'rejection_log_to_console', 'rejection_retention_days',
            'backtest_spread_pips', 'backtest_slippage_pips',
            # SCALP MODE FIELDS
            'scalp_mode_enabled', 'scalp_tp_pips', 'scalp_sl_pips', 'scalp_max_spread_pips',
            'scalp_htf_timeframe', 'scalp_trigger_timeframe', 'scalp_entry_timeframe',
            'scalp_ema_period', 'scalp_min_confidence',
            'scalp_disable_ema_slope_validation', 'scalp_disable_swing_proximity',
            'scalp_disable_volume_filter', 'scalp_disable_macd_filter',
            'scalp_use_momentum_only', 'scalp_momentum_min_depth',
            'scalp_fib_pullback_min', 'scalp_fib_pullback_max',
            'scalp_micro_pullback_lookback',
            'scalp_cooldown_minutes', 'scalp_require_tight_spread',
            'scalp_swing_lookback_bars', 'scalp_range_position_threshold',
            'scalp_enable_claude_ai', 'scalp_use_market_orders', 'scalp_use_limit_orders',
            'scalp_reversal_enabled', 'scalp_reversal_min_runway_pips',
            'scalp_reversal_min_entry_momentum', 'scalp_reversal_block_regimes',
            'scalp_reversal_block_volatility_states', 'scalp_reversal_allow_rsi_extremes',
            # SCALP QUALIFICATION FIELDS (v2.21.0)
            'scalp_qualification_enabled', 'scalp_qualification_mode',
            'scalp_min_qualification_score',
            'scalp_rsi_filter_enabled', 'scalp_two_pole_filter_enabled', 'scalp_macd_filter_enabled',
            'scalp_rsi_bull_min', 'scalp_rsi_bull_max',
            'scalp_rsi_bear_min', 'scalp_rsi_bear_max',
            'scalp_two_pole_bull_threshold', 'scalp_two_pole_bear_threshold',
            # SCALP ENTRY FILTERS (v2.22.0) - Jan 2026 trade analysis
            'scalp_require_htf_alignment', 'scalp_momentum_only_filter',
            'scalp_entry_rsi_buy_max', 'scalp_entry_rsi_sell_min',
            'scalp_min_ema_distance_pips',
            # SCALP REJECTION CANDLE CONFIRMATION (v2.25.0)
            'scalp_require_rejection_candle', 'scalp_rejection_min_strength',
            'scalp_use_market_on_rejection',
            # SCALP ENTRY CANDLE ALIGNMENT (v2.25.1)
            'scalp_require_entry_candle_alignment', 'scalp_use_market_on_entry_alignment',
            # PATTERN CONFIRMATION (v2.23.0) - Alternative triggers
            'pattern_confirmation_enabled', 'pattern_confirmation_mode',
            'pattern_min_strength', 'pattern_pin_bar_enabled',
            'pattern_engulfing_enabled', 'pattern_inside_bar_enabled',
            'pattern_hammer_shooter_enabled', 'pattern_confidence_boost',
            # v2.24.0: Pattern as alternative entry
            'pattern_as_entry_enabled', 'pattern_entry_min_strength',
            # RSI DIVERGENCE (v2.23.0)
            'rsi_divergence_enabled', 'rsi_divergence_mode',
            'rsi_divergence_lookback', 'rsi_divergence_min_strength',
            'rsi_divergence_confidence_boost',
            # v2.24.0: Divergence as alternative entry
            'divergence_as_entry_enabled', 'divergence_entry_min_strength',
            # MACD ALIGNMENT ENHANCEMENT (v2.23.0)
            'macd_alignment_enabled', 'macd_alignment_required',
            'macd_alignment_confidence_boost',
        ]

        # Fields that must be integers (used for DataFrame slicing, loop counts, etc.)
        int_fields = {
            'ema_period', 'swing_lookback_bars', 'swing_strength_bars',
            'volume_sma_period', 'swing_lookback_atr_low', 'swing_lookback_atr_high',
            'swing_lookback_min', 'swing_lookback_max',
            'max_pullback_wait_bars', 'pullback_confirmation_bars',
            'atr_period', 'limit_expiry_minutes',
            'sl_buffer_pips', 'min_tp_pips', 'tp_structure_lookback',
            'max_concurrent_signals', 'signal_cooldown_hours',
            'max_consecutive_losses_before_block', 'win_rate_lookback_trades',
            'rejection_batch_size', 'rejection_retention_days',
            'swing_proximity_min_distance_pips', 'swing_proximity_lookback_swings',
            'max_momentum_staleness_bars',
            # Scalp mode int fields
            'scalp_ema_period', 'scalp_cooldown_minutes', 'scalp_swing_lookback_bars',
            'scalp_micro_pullback_lookback',
            # Scalp qualification int fields
            'scalp_rsi_bull_min', 'scalp_rsi_bull_max', 'scalp_rsi_bear_min', 'scalp_rsi_bear_max',
            # Pattern confirmation int fields
            'rsi_divergence_lookback',
        }

        for attr_name in direct_mappings:
            if attr_name in global_row and global_row[attr_name] is not None:
                value = global_row[attr_name]
                # Convert to appropriate type
                if attr_name in int_fields:
                    value = int(value)
                elif isinstance(value, bool):
                    # Keep booleans as-is (must check before Decimal since bool has as_integer_ratio)
                    pass
                elif hasattr(value, 'as_integer_ratio'):  # Decimal to float
                    value = float(value)
                setattr(config, attr_name, value)

        # Handle time fields
        for time_field in ['london_session_start', 'london_session_end',
                          'ny_session_start', 'ny_session_end']:
            if time_field in global_row and global_row[time_field] is not None:
                value = global_row[time_field]
                if isinstance(value, str):
                    h, m = map(int, value.split(':')[:2])
                    value = dt_time(h, m)
                setattr(config, time_field, value)

        # Handle array fields
        if global_row.get('allowed_sessions'):
            config.allowed_sessions = list(global_row['allowed_sessions'])
        if global_row.get('enabled_pairs'):
            config.enabled_pairs = list(global_row['enabled_pairs'])

        # Handle JSONB fields
        if global_row.get('adaptive_cooldown_config'):
            cooldown = global_row['adaptive_cooldown_config']
            if isinstance(cooldown, str):
                cooldown = json.loads(cooldown)
            # Map to individual attributes
            config.adaptive_cooldown_enabled = cooldown.get('enabled', True)
            config.base_cooldown_hours = cooldown.get('base_cooldown_hours', 2.0)
            config.cooldown_after_win_multiplier = cooldown.get('cooldown_after_win_multiplier', 0.5)
            config.cooldown_after_loss_multiplier = cooldown.get('cooldown_after_loss_multiplier', 1.5)
            config.consecutive_loss_penalty_hours = cooldown.get('consecutive_loss_penalty_hours', 1.0)
            config.max_consecutive_losses_before_block = cooldown.get('max_consecutive_losses_before_block', 3)
            config.consecutive_loss_block_hours = cooldown.get('consecutive_loss_block_hours', 8.0)
            config.win_rate_lookback_trades = cooldown.get('win_rate_lookback_trades', 20)
            config.high_win_rate_threshold = cooldown.get('high_win_rate_threshold', 0.60)
            config.low_win_rate_threshold = cooldown.get('low_win_rate_threshold', 0.40)
            config.critical_win_rate_threshold = cooldown.get('critical_win_rate_threshold', 0.30)
            config.high_win_rate_cooldown_reduction = cooldown.get('high_win_rate_cooldown_reduction', 0.25)
            config.low_win_rate_cooldown_increase = cooldown.get('low_win_rate_cooldown_increase', 0.50)
            config.high_volatility_atr_multiplier = cooldown.get('high_volatility_atr_multiplier', 1.5)
            config.volatility_cooldown_adjustment = cooldown.get('volatility_cooldown_adjustment', 0.30)
            config.strong_trend_cooldown_reduction = cooldown.get('strong_trend_cooldown_reduction', 0.30)
            config.session_change_reset_cooldown = cooldown.get('session_change_reset_cooldown', True)
            config.min_cooldown_hours = cooldown.get('min_cooldown_hours', 1.0)
            config.max_cooldown_hours = cooldown.get('max_cooldown_hours', 12.0)

        if global_row.get('confidence_weights'):
            weights = global_row['confidence_weights']
            if isinstance(weights, str):
                weights = json.loads(weights)
            config.confidence_weights = weights

        if global_row.get('pair_pip_values'):
            pip_values = global_row['pair_pip_values']
            if isinstance(pip_values, str):
                pip_values = json.loads(pip_values)
            config.pair_pip_values = pip_values

        # Build pair overrides dict
        config._pair_overrides = {}
        for row in override_rows:
            epic = row['epic']
            config._pair_overrides[epic] = {
                'is_enabled': row.get('is_enabled', True),
                'description': row.get('description'),
                'parameter_overrides': row.get('parameter_overrides') or {},
                'allow_asian_session': row.get('allow_asian_session'),
                'sl_buffer_pips': row.get('sl_buffer_pips'),
                'min_confidence': row.get('min_confidence'),
                'max_confidence': row.get('max_confidence'),
                'min_volume_ratio': row.get('min_volume_ratio'),
                'high_volume_confidence': row.get('high_volume_confidence'),
                'low_atr_confidence': row.get('low_atr_confidence'),
                'high_atr_confidence': row.get('high_atr_confidence'),
                'near_ema_confidence': row.get('near_ema_confidence'),
                'far_ema_confidence': row.get('far_ema_confidence'),
                'macd_filter_enabled': row.get('macd_filter_enabled'),
                'blocking_conditions': row.get('blocking_conditions'),
                'fixed_stop_loss_pips': row.get('fixed_stop_loss_pips'),
                'fixed_take_profit_pips': row.get('fixed_take_profit_pips'),
                # Direction-aware overrides (v2.12.0)
                'direction_overrides_enabled': row.get('direction_overrides_enabled', False),
                'fib_pullback_min_bull': row.get('fib_pullback_min_bull'),
                'fib_pullback_min_bear': row.get('fib_pullback_min_bear'),
                'fib_pullback_max_bull': row.get('fib_pullback_max_bull'),
                'fib_pullback_max_bear': row.get('fib_pullback_max_bear'),
                'momentum_min_depth_bull': row.get('momentum_min_depth_bull'),
                'momentum_min_depth_bear': row.get('momentum_min_depth_bear'),
                'min_volume_ratio_bull': row.get('min_volume_ratio_bull'),
                'min_volume_ratio_bear': row.get('min_volume_ratio_bear'),
                'min_confidence_bull': row.get('min_confidence_bull'),
                'min_confidence_bear': row.get('min_confidence_bear'),
                # Swing proximity overrides (v2.15.1)
                'swing_proximity_enabled': row.get('swing_proximity_enabled'),
                'swing_proximity_min_distance_pips': row.get('swing_proximity_min_distance_pips'),
                'swing_proximity_strict_mode': row.get('swing_proximity_strict_mode'),
                # EMA overrides (v2.16.0)
                'ema_period': row.get('ema_period'),
                'ema_slope_validation_enabled': row.get('ema_slope_validation_enabled'),
                # Stop offset override (v2.17.0)
                'stop_offset_pips': row.get('stop_offset_pips'),
                # Scalp tier overrides (v2.21.0)
                'scalp_ema_period': row.get('scalp_ema_period'),
                'scalp_swing_lookback_bars': row.get('scalp_swing_lookback_bars'),
                'scalp_limit_offset_pips': row.get('scalp_limit_offset_pips'),
                'scalp_htf_timeframe': row.get('scalp_htf_timeframe'),
                'scalp_trigger_timeframe': row.get('scalp_trigger_timeframe'),
                'scalp_entry_timeframe': row.get('scalp_entry_timeframe'),
                'scalp_min_confidence': row.get('scalp_min_confidence'),
                'scalp_cooldown_minutes': row.get('scalp_cooldown_minutes'),
                'scalp_swing_break_tolerance_pips': row.get('scalp_swing_break_tolerance_pips'),
                # Scalp qualification mode override (v2.24.1) - per-pair ACTIVE/MONITORING
                'scalp_qualification_mode': row.get('scalp_qualification_mode'),
                # Scalp pullback threshold override (v2.24.1) - per-pair optimal threshold
                'scalp_fib_pullback_min': row.get('scalp_fib_pullback_min'),
                # Scalp SL/TP overrides (v2.30.0) - ATR-optimized per-pair values
                'scalp_sl_pips': row.get('scalp_sl_pips'),
                'scalp_tp_pips': row.get('scalp_tp_pips'),
            }

        return config

    def get_effective_config_for_pair(self, epic: str) -> Dict[str, Any]:
        """
        Get the effective configuration for a specific pair.
        Merges global config with pair-specific overrides.
        """
        config = self.get_config()

        # Start with global values
        effective = {}
        for attr_name in dir(config):
            if not attr_name.startswith('_') and not callable(getattr(config, attr_name)):
                value = getattr(config, attr_name)
                # Skip non-serializable types
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    effective[attr_name] = value
                elif isinstance(value, dt_time):
                    effective[attr_name] = value.strftime('%H:%M')

        # Apply pair overrides
        if epic in config._pair_overrides:
            overrides = config._pair_overrides[epic]
            if overrides.get('parameter_overrides'):
                effective.update(overrides['parameter_overrides'])

            # Apply specific override fields
            for field_name in ['sl_buffer_pips', 'min_confidence', 'max_confidence', 'allow_asian_session',
                              'min_volume_ratio', 'macd_filter_enabled']:
                if overrides.get(field_name) is not None:
                    effective[field_name] = overrides[field_name]

        return effective

    def invalidate_cache(self):
        """Force cache invalidation"""
        with self._lock:
            self._cache_timestamp = None
            logger.info("SMC Simple config cache invalidated")

    # =========================================================================
    # CONFIG ACCESSOR METHODS
    # These delegate to the cached SMCSimpleConfig object for per-pair lookups.
    # The strategy uses the service as its config accessor, so these methods
    # provide a clean interface without exposing the internal config object.
    # =========================================================================

    def is_asian_session_allowed(self, epic: str) -> bool:
        """Check if Asian session trading is allowed for a specific pair"""
        return self.get_config().is_asian_session_allowed(epic)

    def is_macd_filter_enabled(self, epic: str) -> bool:
        """Check if MACD filter is enabled for a specific pair"""
        return self.get_config().is_macd_filter_enabled(epic)

    def get_pair_sl_buffer(self, epic: str) -> int:
        """Get SL buffer for a specific pair"""
        return self.get_config().get_pair_sl_buffer(epic)

    def get_pair_min_confidence(self, epic: str) -> float:
        """Get minimum confidence threshold for a specific pair"""
        return self.get_config().get_pair_min_confidence(epic)

    def get_pair_max_confidence(self, epic: str) -> float:
        """Get maximum confidence cap for a specific pair"""
        return self.get_config().get_pair_max_confidence(epic)

    def get_pair_fixed_stop_loss(self, epic: str) -> float:
        """Get fixed stop loss for a specific pair"""
        return self.get_config().get_pair_fixed_stop_loss(epic)

    def get_pair_fixed_take_profit(self, epic: str) -> float:
        """Get fixed take profit for a specific pair"""
        return self.get_config().get_pair_fixed_take_profit(epic)

    def get_pair_min_volume_ratio(self, epic: str) -> float:
        """Get minimum volume ratio for a specific pair"""
        return self.get_config().get_pair_min_volume_ratio(epic)

    def is_session_allowed(self, hour_utc: int) -> bool:
        """Check if current hour is in allowed trading session"""
        return self.get_config().is_session_allowed(hour_utc)

    # =========================================================================
    # SWING PROXIMITY SERVICE ACCESSORS (v2.15.1)
    # =========================================================================

    def is_swing_proximity_enabled(self, epic: str) -> bool:
        """Check if swing proximity validation is enabled for a specific pair"""
        return self.get_config().is_swing_proximity_enabled(epic)

    def get_pair_swing_proximity_min_distance(self, epic: str) -> int:
        """Get swing proximity min distance in pips for a specific pair"""
        return self.get_config().get_pair_swing_proximity_min_distance(epic)

    def is_swing_proximity_strict_mode(self, epic: str) -> bool:
        """Check if swing proximity strict mode is enabled for a specific pair"""
        return self.get_config().is_swing_proximity_strict_mode(epic)

    # =========================================================================
    # EMA OVERRIDES SERVICE ACCESSORS (v2.16.0)
    # =========================================================================

    def get_pair_ema_period(self, epic: str) -> int:
        """Get EMA period for a specific pair"""
        return self.get_config().get_pair_ema_period(epic)

    def is_ema_slope_validation_enabled(self, epic: str) -> bool:
        """Check if EMA slope validation is enabled for a specific pair"""
        return self.get_config().is_ema_slope_validation_enabled(epic)

    # =========================================================================
    # STOP OFFSET SERVICE ACCESSORS (v2.17.0)
    # =========================================================================

    def get_pair_stop_offset(self, epic: str, entry_type: str = 'MOMENTUM') -> float:
        """Get stop entry offset in pips for a specific pair"""
        return self.get_config().get_pair_stop_offset(epic, entry_type)

    # =========================================================================
    # SCALP EMA BUFFER SERVICE ACCESSORS (v2.31.2)
    # =========================================================================

    def get_pair_scalp_ema_buffer_pips(self, epic: str) -> Optional[float]:
        """Get per-pair EMA buffer for scalp trades.

        Some pairs like NZDUSD tend to hover close to the EMA, so they need
        a smaller buffer to generate signals. Default scalp buffer is 1.0 pip.
        """
        return self.get_config().get_pair_scalp_ema_buffer_pips(epic)


# Global singleton instance
_service_instance: Optional[SMCSimpleConfigService] = None
_service_lock = RLock()


def get_smc_simple_config_service(
    database_url: str = None,
    cache_ttl_seconds: int = 120,
    enable_hot_reload: bool = True
) -> SMCSimpleConfigService:
    """Get singleton instance of config service"""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = SMCSimpleConfigService(
                database_url=database_url,
                cache_ttl_seconds=cache_ttl_seconds,
                enable_hot_reload=enable_hot_reload
            )
        return _service_instance


def get_smc_simple_config() -> SMCSimpleConfig:
    """Convenience function to get current config"""
    return get_smc_simple_config_service().get_config()
