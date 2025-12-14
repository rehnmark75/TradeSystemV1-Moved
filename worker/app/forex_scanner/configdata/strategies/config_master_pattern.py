# ============================================================================
# MASTER PATTERN (ICT POWER OF 3 / AMD) STRATEGY CONFIGURATION
# ============================================================================
# Version: 1.0.0
# Description: ICT Power of 3 (Accumulation, Manipulation, Distribution) strategy
# Architecture:
#   PHASE 1: Accumulation - Asian session consolidation (00:00-08:00 UTC)
#   PHASE 2: Manipulation - Judas swing/liquidity sweep (08:00-10:00 UTC)
#   PHASE 3: Distribution - Real move + entry on FVG pullback
#
# Entry Logic:
#   - Bullish: Accumulation → sweep lows → BOS/ChoCH up → enter on FVG
#   - Bearish: Accumulation → sweep highs → BOS/ChoCH down → enter on FVG
#
# Expected Performance (validated by trading analyst):
#   - Win Rate: 42-48%
#   - Profit Factor: 1.6-1.9
#   - Average R:R: 2.3-2.8
#   - Trades/Month: 12-18 (London focus only)
# ============================================================================

from datetime import time
from typing import Dict, Tuple

# ============================================================================
# STRATEGY METADATA
# ============================================================================
STRATEGY_NAME = "MASTER_PATTERN"
STRATEGY_VERSION = "1.0.0"
STRATEGY_DATE = "2025-12-13"
STRATEGY_STATUS = "Initial Release - ICT Power of 3 / AMD Pattern"

# ============================================================================
# SESSION DEFINITIONS (UTC)
# ============================================================================
# ICT Power of 3 is session-based - timing is critical

# Asian Session (Accumulation Phase)
ASIAN_SESSION_START = time(0, 0)        # 00:00 UTC
ASIAN_SESSION_END = time(8, 0)          # 08:00 UTC

# London Open (Manipulation Phase)
LONDON_OPEN_START = time(8, 0)          # 08:00 UTC
LONDON_OPEN_END = time(10, 0)           # 10:00 UTC

# Distribution Window (Entry Phase)
DISTRIBUTION_START = time(8, 0)         # 08:00 UTC (can enter after sweep)
DISTRIBUTION_END = time(16, 0)          # 16:00 UTC (extended for pullback)

# Hard cutoff for entries
ENTRY_CUTOFF_TIME = time(16, 0)         # 16:00 UTC - extended to allow pullback entries

# ============================================================================
# PHASE 1: ACCUMULATION DETECTION
# ============================================================================
# Looking for tight consolidation during Asian session

# Minimum duration for valid accumulation (tightened for quality)
MIN_ACCUMULATION_CANDLES_5M = 15        # 75 minutes minimum on 5m (was 10)
MIN_ACCUMULATION_CANDLES_15M = 6        # ~1.5 hours on 15m (was 4)

# Maximum range for consolidation
MAX_ACCUMULATION_RANGE_PIPS = 30        # Default max (overridden by ATR) - tightened for quality setups

# ATR-based range validation (adaptive to pair volatility)
USE_ATR_ACCUMULATION_VALIDATION = True
ATR_ACCUMULATION_MULTIPLIER = 0.5       # Range must be < 50% of ATR-20

# ATR compression threshold (realistic for forex markets)
# Note: Ratio = current_atr / baseline_atr. Values > 1.0 mean HIGHER volatility.
# Asian session may not always be quieter, so use a realistic threshold.
USE_ATR_COMPRESSION_CHECK = False       # DISABLED - ATR ratios in real data are too high (1.5-2.5x)
ATR_COMPRESSION_THRESHOLD = 0.90        # ATR must be < 90% of baseline (if enabled)
ATR_BASELINE_PERIOD = 20                # Use 20-period ATR for baseline comparison

# Volume profile during accumulation (should be declining)
VALIDATE_ACCUMULATION_VOLUME = True
ACCUMULATION_VOLUME_DECLINE_RATIO = 1.15  # Second half volume < first half * 1.15

# Daily open position (for directional bias)
USE_DAILY_OPEN_BIAS = True              # Accumulation below open = bullish setup

# ============================================================================
# PHASE 2: MANIPULATION DETECTION (JUDAS SWING)
# ============================================================================
# Looking for false breakout that sweeps liquidity

# Sweep extension limits
MIN_SWEEP_EXTENSION_PIPS = 3            # Minimum pips beyond range
MAX_SWEEP_EXTENSION_PIPS = 15           # Maximum pips beyond range

# Volume spike requirement (key filter from trading analyst)
SWEEP_VOLUME_MULTIPLIER = 1.5           # 1.5x average (was 1.3x)
SWEEP_VOLUME_LOOKBACK = 20              # Compare to 20-candle average

# Rejection wick requirement (key filter from trading analyst)
MIN_REJECTION_WICK_RATIO = 0.65         # 65%+ wick of candle range (was 60%)

# Manipulation timing
MAX_SWEEP_WAIT_MINUTES = 60             # Max time to wait for sweep after Asian close

# ============================================================================
# PHASE 3: STRUCTURE SHIFT VALIDATION
# ============================================================================
# Require BOS or ChoCH after manipulation to confirm direction

REQUIRE_STRUCTURE_SHIFT = True          # Must have BOS/ChoCH to confirm

# Structure shift parameters
STRUCTURE_SHIFT_LOOKBACK_BARS = 10      # Candles to look for shift after sweep
STRUCTURE_SHIFT_TIMEFRAME = "15m"       # Timeframe for structure analysis

# Structure shift quality requirements
MIN_BOS_CANDLE_BODY_RATIO = 0.50        # BOS candle body > 50% of range (relaxed from 60%)
MIN_BOS_VOLUME_MULTIPLIER = 1.2         # BOS volume > 1.2x average (relaxed from 1.4x)

# ChoCH vs BOS scoring
BOS_CONFIDENCE_BONUS = 0.10             # BOS gets 10% confidence boost over ChoCH

# ============================================================================
# ENTRY ZONE DETECTION (FVG / MITIGATION)
# ============================================================================
# Find optimal entry after structure shift

# Primary entry: Fair Value Gap
USE_FVG_ENTRY = True
MAX_FVG_DISTANCE_PIPS = 20              # Maximum distance to FVG for entry
MIN_FVG_SIZE_PIPS = 2                   # Minimum FVG size

# Fallback entry: Mitigation zone (50% of manipulation candle)
USE_MITIGATION_ENTRY = True
MITIGATION_ZONE_WIDTH_PIPS = 5          # Width of mitigation zone

# Entry confirmation
REQUIRE_ENTRY_CANDLE_PATTERN = True     # Require bullish/bearish candle pattern
ALLOWED_ENTRY_PATTERNS = ['engulfing', 'pin_bar', 'inside_bar_break']

# Entry timing
MAX_ENTRY_WAIT_BARS = 15                # Max bars to wait for entry after shift

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Stop Loss
SL_BUFFER_PIPS = 3                      # Buffer beyond manipulation wick
USE_ATR_STOP = True
SL_ATR_MULTIPLIER = 0.5                 # SL = max(buffer, 0.5 * ATR)
MIN_STOP_DISTANCE_PIPS = 15             # Minimum stop distance to avoid noise stops

# Take Profit (R:R based)
MIN_RR_RATIO = 2.0                      # Minimum risk-reward
TARGET_RR_RATIO = 2.5                   # Target R:R for confidence scoring
MAX_RR_RATIO = 4.0                      # Cap R:R for realistic targets

# Liquidity targets (session highs/lows)
USE_SESSION_TARGETS = True              # Target session high/low as TP
SESSION_TARGET_BUFFER_PIPS = 2          # Buffer before session high/low

# Partial close
PARTIAL_CLOSE_ENABLED = True
PARTIAL_CLOSE_RR = 1.0                  # Close 50% at 1:1 R:R
PARTIAL_CLOSE_PERCENT = 50              # Percentage to close

# Position sizing
RISK_PER_TRADE_PCT = 1.0                # 1% risk per trade

# ============================================================================
# HTF (HIGHER TIMEFRAME) TREND ALIGNMENT - CRITICAL FILTER
# ============================================================================
# This is the MOST IMPORTANT filter for preventing counter-trend entries
# which are the primary cause of poor win rates in the Master Pattern strategy.

REQUIRE_HTF_ALIGNMENT = True            # Enable 4H trend direction check (CRITICAL!)
HTF_TIMEFRAME = '4h'                    # Primary HTF timeframe for validation
HTF_LOOKBACK_BARS = 50                  # Bars for HH/HL/LH/LL pattern detection
MIN_HTF_STRENGTH = 0.40                 # Minimum trend strength to filter (0.0-1.0)
HTF_STRONG_ALIGNMENT_BONUS = 0.10       # +10% confidence if HTF > 60% strength

# HTF trend validation:
# - BULL signals require 4H showing HH/HL (bullish structure)
# - BEAR signals require 4H showing LH/LL (bearish structure)
# - Signals against strong 4H trend (>40% strength) are REJECTED

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

# Minimum thresholds
MIN_CONFIDENCE_THRESHOLD = 0.55         # Minimum to take trade (lowered to allow signals while other filters validate)
HIGH_CONFIDENCE_THRESHOLD = 0.70        # High-confidence setup

# Scoring weights (must sum to 1.0)
CONFIDENCE_WEIGHTS = {
    'accumulation_quality': 0.20,       # ATR compression, duration, cleanness
    'manipulation_clarity': 0.25,       # Volume spike, rejection wick
    'structure_shift_strength': 0.25,   # BOS vs ChoCH, volume on shift
    'entry_zone_quality': 0.15,         # FVG (better) vs mitigation
    'rr_quality': 0.15,                 # R:R ratio (2.0-4.0 range)
}

# Confidence scoring details
ACCUMULATION_SCORING = {
    'atr_compression_weight': 0.60,     # Lower ATR ratio = better
    'duration_weight': 0.40,            # Longer consolidation = better (up to limit)
    'optimal_duration_candles': 16,     # ~80 min on 5m is optimal
}

MANIPULATION_SCORING = {
    'volume_spike_weight': 0.50,        # Higher volume spike = better
    'rejection_wick_weight': 0.50,      # Bigger rejection wick = better
    'max_volume_multiplier': 2.0,       # Cap at 2x for scoring
    'max_wick_ratio': 0.80,             # Cap at 80% wick
}

STRUCTURE_SCORING = {
    'bos_bonus': 0.10,                  # BOS gets bonus over ChoCH
    'volume_weight': 0.40,              # Volume on shift candle
    'body_close_weight': 0.60,          # Clean body close beyond structure
}

ENTRY_ZONE_SCORING = {
    'fvg_score': 1.0,                   # FVG = full score
    'mitigation_score': 0.6,            # Mitigation = 60% score
}

RR_SCORING = {
    'min_rr': 2.0,                      # 0% score below this
    'max_rr': 4.0,                      # 100% score at or above this
}

# ============================================================================
# PHASE TIMEOUTS
# ============================================================================

# Maximum time in each phase before reset
ACCUMULATION_TIMEOUT_HOURS = 10         # Max hours in accumulation (Asian + buffer)
MANIPULATION_TIMEOUT_HOURS = 2          # Max hours waiting for manipulation
STRUCTURE_TIMEOUT_CANDLES = 15          # Max candles for structure shift (15m)
ENTRY_TIMEOUT_CANDLES = 20              # Max candles to wait for entry (5m)

# ============================================================================
# SIGNAL LIMITS
# ============================================================================

MAX_SIGNALS_PER_PAIR_PER_DAY = 1        # One AMD setup per pair per day
SIGNAL_COOLDOWN_HOURS = 12              # Cooldown after signal

# ============================================================================
# PAIR CONFIGURATION
# ============================================================================

# Enabled pairs (major forex pairs with good liquidity)
ENABLED_PAIRS = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    'CS.D.USDCHF.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.GBPJPY.MINI.IP',
]

# Pair-specific pip values (correct for IG Markets pricing)
PAIR_PIP_VALUES = {
    'CS.D.EURUSD.CEEM.IP': 0.0001,      # EURUSD = 0.0001 (4th decimal is pip)
    'CS.D.GBPUSD.MINI.IP': 0.0001,
    'CS.D.USDJPY.MINI.IP': 0.01,
    'CS.D.USDCHF.MINI.IP': 0.0001,
    'CS.D.AUDUSD.MINI.IP': 0.0001,
    'CS.D.USDCAD.MINI.IP': 0.0001,
    'CS.D.NZDUSD.MINI.IP': 0.0001,
    'CS.D.EURJPY.MINI.IP': 0.01,
    'CS.D.GBPJPY.MINI.IP': 0.01,
    'CS.D.AUDJPY.MINI.IP': 0.01,
}

# Pair-specific calibration (realistic thresholds based on ATR ratio behavior)
# Note: ATR ratio = current/baseline. Values must be < threshold (below baseline volatility)
PAIR_CALIBRATION = {
    'EURUSD': {
        'atr_compression': 0.85,        # Current ATR must be < 85% of baseline
        'min_sweep_pips': 3,            # Accept small sweeps (3+ pips)
        'max_sweep_pips': 20,           # Allow larger sweeps
        'max_range_pips': 25,           # Tight accumulation range
    },
    'GBPUSD': {
        'atr_compression': 0.90,        # GBP more volatile, looser threshold
        'min_sweep_pips': 4,            # Accept small sweeps (4+ pips)
        'max_sweep_pips': 25,
        'max_range_pips': 30,
    },
    'USDJPY': {
        'atr_compression': 0.90,
        'min_sweep_pips': 5,            # JPY pairs have larger pip values
        'max_sweep_pips': 30,
        'max_range_pips': 35,
    },
    # Default for others uses global settings
}

# Pair-specific confidence floors
PAIR_MIN_CONFIDENCE = {
    # JPY crosses need higher confidence (more volatile)
    'EURJPY': 0.70,
    'CS.D.EURJPY.MINI.IP': 0.70,
    'GBPJPY': 0.70,
    'CS.D.GBPJPY.MINI.IP': 0.70,
    # Default for others = MIN_CONFIDENCE_THRESHOLD (0.65)
}

# ============================================================================
# NEWS FILTER (Risk Mitigation)
# ============================================================================

NEWS_FILTER_ENABLED = True

# High-impact news to avoid
HIGH_IMPACT_NEWS = ['NFP', 'CPI', 'Interest Rate', 'FOMC', 'GDP', 'ECB', 'BOE', 'BOJ']
MEDIUM_IMPACT_NEWS = ['Retail Sales', 'Unemployment', 'PMI', 'Trade Balance']

# Buffer around news events
HIGH_IMPACT_BUFFER_HOURS = 2            # Block trades 2h before/after
MEDIUM_IMPACT_BUFFER_HOURS = 1          # Block trades 1h before/after
MEDIUM_IMPACT_CONFIDENCE_PENALTY = 0.10  # -10% confidence if within buffer

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

ENABLE_DEBUG_LOGGING = True
LOG_PHASE_TRANSITIONS = True            # Log accumulation → manipulation → distribution
LOG_REJECTED_SETUPS = True              # Log why setups were rejected
LOG_CONFIDENCE_BREAKDOWN = True         # Log confidence scoring details

# ============================================================================
# BACKTEST SETTINGS
# ============================================================================

BACKTEST_SPREAD_PIPS = 1.5
BACKTEST_SLIPPAGE_PIPS = 0.5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pip_value(epic: str) -> float:
    """Get pip value for a given epic."""
    return PAIR_PIP_VALUES.get(epic, 0.0001)


def get_pair_calibration(pair: str) -> Dict:
    """Get pair-specific calibration or defaults."""
    # Extract pair name from epic
    pair_name = pair.upper()
    for key in PAIR_CALIBRATION:
        if key in pair_name:
            return PAIR_CALIBRATION[key]

    # Return defaults
    return {
        'atr_compression': ATR_COMPRESSION_THRESHOLD,
        'min_sweep_pips': MIN_SWEEP_EXTENSION_PIPS,
        'max_sweep_pips': MAX_SWEEP_EXTENSION_PIPS,
        'max_range_pips': MAX_ACCUMULATION_RANGE_PIPS,
    }


def get_pair_min_confidence(epic: str) -> float:
    """Get minimum confidence for a pair."""
    return PAIR_MIN_CONFIDENCE.get(epic, MIN_CONFIDENCE_THRESHOLD)


def is_asian_session(hour_utc: int, minute_utc: int = 0) -> bool:
    """Check if time is in Asian session (accumulation phase)."""
    current = time(hour_utc, minute_utc)
    return ASIAN_SESSION_START <= current < ASIAN_SESSION_END


def is_london_open(hour_utc: int, minute_utc: int = 0) -> bool:
    """Check if time is in London open window (manipulation phase)."""
    current = time(hour_utc, minute_utc)
    return LONDON_OPEN_START <= current < LONDON_OPEN_END


def is_distribution_window(hour_utc: int, minute_utc: int = 0) -> bool:
    """Check if time is in distribution/entry window."""
    current = time(hour_utc, minute_utc)
    return DISTRIBUTION_START <= current < DISTRIBUTION_END


def is_entry_allowed(hour_utc: int, minute_utc: int = 0) -> bool:
    """Check if entry is still allowed (before cutoff)."""
    current = time(hour_utc, minute_utc)
    return current < ENTRY_CUTOFF_TIME


def get_session_windows() -> Dict[str, Tuple[time, time]]:
    """Get all session windows as dictionary."""
    return {
        'asian': (ASIAN_SESSION_START, ASIAN_SESSION_END),
        'london_open': (LONDON_OPEN_START, LONDON_OPEN_END),
        'distribution': (DISTRIBUTION_START, DISTRIBUTION_END),
    }
