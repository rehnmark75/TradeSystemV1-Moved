# ============================================================================
# SMC SIMPLE STRATEGY CONFIGURATION
# ============================================================================
# Version: 1.4.0 (Quality Optimization Phase 1)
# Description: Simplified 3-tier SMC strategy for intraday forex trading
# Architecture:
#   TIER 1: 4H 50 EMA for directional bias
#   TIER 2: 15m swing break with body-close confirmation (was 1H)
#   TIER 3: 5m pullback entry with Fibonacci zones (was 15m)
#
# v1.4.0 OPTIMIZATION (Phase 1):
#   - Baseline: 1068 signals, 25.7% WR, 0.93 PF, -0.3 pip expectancy
#   - Target: 300-400 signals, 35%+ WR, 1.2+ PF, +2 pip expectancy
#   - Changes: Session filter ON, confidence 65%, tighter Fib zones,
#              higher EMA distance, increased R:R requirement
# ============================================================================

from datetime import time

# ============================================================================
# STRATEGY METADATA
# ============================================================================
STRATEGY_NAME = "SMC_SIMPLE"
STRATEGY_VERSION = "1.6.0"
STRATEGY_DATE = "2025-12-02"
STRATEGY_STATUS = "Pullback Calculation Fix - Timeframe Alignment"

# ============================================================================
# TIER 1: 4H DIRECTIONAL BIAS (Higher Timeframe)
# ============================================================================
# The 50 EMA is used by institutions for trend direction
# Price above EMA = bullish bias (look for longs only)
# Price below EMA = bearish bias (look for shorts only)

HTF_TIMEFRAME = "4h"                    # Higher timeframe for bias
EMA_PERIOD = 50                          # 50-period EMA (institutional standard)
EMA_BUFFER_PIPS = 3                      # Buffer zone around EMA

# Price position requirements
REQUIRE_CLOSE_BEYOND_EMA = True          # Candle must CLOSE beyond EMA (not just wick)
MIN_DISTANCE_FROM_EMA_PIPS = 10          # v1.4.0: INCREASED from 3 - avoid noise near EMA

# ============================================================================
# TIER 2: 15M ENTRY TRIGGER (Intermediate Timeframe)
# ============================================================================
# Looking for swing break on 15m as entry confirmation (more frequent than 1H)
# A swing high/low break with BODY CLOSE confirms momentum
# 15m gives 4x more candles than 1H = 4x more break opportunities

TRIGGER_TIMEFRAME = "15m"                # Trigger timeframe (was 1h - too slow)
SWING_LOOKBACK_BARS = 20                 # Bars to look back for swing detection
SWING_STRENGTH_BARS = 2                  # Bars on each side to confirm swing

# Body-close confirmation
# NOTE: On 15m timeframe, body-close is too strict - price rarely closes beyond swings
# For 15m trigger, we use wick-based breaks instead (high/low must exceed swing)
REQUIRE_BODY_CLOSE_BREAK = False         # Disabled for 15m - use wick breaks instead
WICK_TOLERANCE_PIPS = 3                  # Tolerance for wick touches

# Volume confirmation
VOLUME_CONFIRMATION_ENABLED = False      # Disabled - was blocking too many signals
VOLUME_SMA_PERIOD = 20                   # Period for volume moving average
VOLUME_SPIKE_MULTIPLIER = 1.3            # Volume must be 1.3x average

# ============================================================================
# TIER 3: 5M EXECUTION (Entry Timeframe)
# ============================================================================
# After 15m break, wait for 5m pullback to Fibonacci zones
# Enter on pullback to get better R:R - 5m gives more precise entries

ENTRY_TIMEFRAME = "5m"                   # Entry/execution timeframe (was 15m)
PULLBACK_ENABLED = True                  # Wait for pullback before entry

# Fibonacci pullback zones (measured from swing to break point)
# v1.5.0: TIGHTENED to golden zone only (38%-62%) - highest probability entries
# v1.4.0: Was 30%-65% (35% range) - still accepting marginal entries
# Analysis: Strict golden zone improves entry timing and reduces false breakouts
FIB_PULLBACK_MIN = 0.38                  # v1.5.0: TIGHTENED from 0.30 - golden zone start (38.2%)
FIB_PULLBACK_MAX = 0.62                  # v1.5.0: TIGHTENED from 0.65 - golden zone end (61.8%)
FIB_OPTIMAL_ZONE = (0.382, 0.618)        # Optimal entry zone (golden zone) - now strictly enforced

# Pullback timing
MAX_PULLBACK_WAIT_BARS = 12              # Maximum bars to wait for pullback
PULLBACK_CONFIRMATION_BARS = 2           # Bars to confirm pullback

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# R:R Requirements
# v1.5.2: INCREASED R:R requirements for better selectivity
# Higher R:R = fewer but better setups with larger potential gains
MIN_RR_RATIO = 2.5                       # v1.5.2: INCREASED from 2.0 - only trades with good reward potential
OPTIMAL_RR_RATIO = 3.5                   # v1.5.2: INCREASED from 3.0 - target premium setups
MAX_RR_RATIO = 6.0                       # v1.5.2: INCREASED from 5.0 - allow larger targets

# Stop Loss
# v1.5.2: TIGHT SL buffer - quality over quantity
# Analysis: v1.5.0 with 6 pips was profitable (PF 1.18), v1.5.1 with 8 pips regressed
SL_BUFFER_PIPS = 6                       # v1.5.2: Back to 6 (v1.5.0 was profitable with this)
SL_ATR_MULTIPLIER = 1.0                  # v1.5.0: REDUCED from 1.2 - tighter ATR multiplier
USE_ATR_STOP = True                      # v1.5.0: ENABLED - adaptive stops for volatility

# Take Profit
# v1.5.0: INCREASED minimum TP for better risk-reward with tighter stops
MIN_TP_PIPS = 15                         # v1.5.0: INCREASED from 12 - larger targets for better R:R
USE_SWING_TARGET = True                  # Target next swing high/low
TP_STRUCTURE_LOOKBACK = 50               # Bars to look for structure targets

# Position sizing (for reference, actual sizing in trade executor)
RISK_PER_TRADE_PCT = 1.0                 # 1% risk per trade

# ============================================================================
# SESSION FILTER
# ============================================================================
# Only trade during high-volume sessions

SESSION_FILTER_ENABLED = True             # v1.4.0: ENABLED - trade only London/NY sessions

# Session definitions (UTC times)
LONDON_SESSION_START = time(7, 0)        # 07:00 UTC
LONDON_SESSION_END = time(16, 0)         # 16:00 UTC
NY_SESSION_START = time(12, 0)           # 12:00 UTC
NY_SESSION_END = time(21, 0)             # 21:00 UTC

# Allowed sessions
ALLOWED_SESSIONS = ['london', 'new_york', 'overlap']
BLOCK_ASIAN_SESSION = True               # Block 21:00-07:00 UTC

# ============================================================================
# SIGNAL LIMITS
# ============================================================================

MAX_SIGNALS_PER_PAIR_PER_DAY = 1         # Maximum 1 signal per pair per day
MAX_CONCURRENT_SIGNALS = 3               # Maximum concurrent open signals
SIGNAL_COOLDOWN_HOURS = 4                # Hours between signals on same pair

# ============================================================================
# PAIR CONFIGURATION
# ============================================================================

# Enabled pairs (major forex pairs with good liquidity)
ENABLED_PAIRS = [
    'CS.D.EURUSD.MINI.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    'CS.D.USDCHF.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.GBPJPY.MINI.IP',
]

# Pair-specific pip values (from existing config)
PAIR_PIP_VALUES = {
    'CS.D.EURUSD.MINI.IP': 0.0001,
    'CS.D.GBPUSD.MINI.IP': 0.0001,
    'CS.D.USDJPY.MINI.IP': 0.01,
    'CS.D.USDCHF.MINI.IP': 0.0001,
    'CS.D.AUDUSD.MINI.IP': 0.0001,
    'CS.D.USDCAD.MINI.IP': 0.0001,
    'CS.D.NZDUSD.MINI.IP': 0.0001,
    'CS.D.EURJPY.MINI.IP': 0.01,
    'CS.D.GBPJPY.MINI.IP': 0.01,
}

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================
# Simple confidence calculation based on setup quality

# Confidence thresholds
# v1.5.3: MAXIMUM quality filter - only the best setups
# v1.5.2 (75%): 264 signals, 41.3% WR, PF 1.23 ✅ BEST SO FAR
# v1.5.0 (70%): 201 signals, 39.8% WR, PF 1.18 ✅ PROFITABLE
# v1.5.1 (65%): 319 signals, 37.6% WR, PF 0.94 ❌ UNPROFITABLE
MIN_CONFIDENCE_THRESHOLD = 0.80          # v1.5.3: INCREASED from 0.75 - elite setups only
HIGH_CONFIDENCE_THRESHOLD = 0.90         # v1.5.3: INCREASED from 0.85 - premium tier

# Scoring weights (must sum to 1.0)
CONFIDENCE_WEIGHTS = {
    'ema_alignment': 0.25,               # 4H EMA alignment strength
    'swing_break_quality': 0.30,         # How clean was the 1H break
    'pullback_depth': 0.20,              # Pullback to optimal Fib zone
    'volume_confirmation': 0.15,         # Volume spike on break
    'rr_ratio': 0.10,                    # Risk-reward quality
}

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

ENABLE_DEBUG_LOGGING = True              # Detailed logging for development
LOG_REJECTED_SIGNALS = True              # Log why signals were rejected
LOG_SWING_DETECTION = False              # Log swing point detection (verbose)
LOG_EMA_CHECKS = False                   # Log EMA alignment checks (verbose)

# ============================================================================
# BACKTEST SETTINGS
# ============================================================================

BACKTEST_SPREAD_PIPS = 1.5               # Assumed spread for backtesting
BACKTEST_SLIPPAGE_PIPS = 0.5             # Assumed slippage for backtesting

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pip_value(epic: str) -> float:
    """Get pip value for a given epic."""
    return PAIR_PIP_VALUES.get(epic, 0.0001)

def is_session_allowed(hour_utc: int) -> bool:
    """Check if current hour is in allowed trading session."""
    if not SESSION_FILTER_ENABLED:
        return True

    # London: 07:00-16:00 UTC
    # NY: 12:00-21:00 UTC
    # Overlap: 12:00-16:00 UTC

    if BLOCK_ASIAN_SESSION and (hour_utc >= 21 or hour_utc < 7):
        return False

    return 7 <= hour_utc <= 21  # London open to NY close

def get_optimal_pullback_zone() -> tuple:
    """Get optimal Fibonacci pullback zone."""
    return FIB_OPTIMAL_ZONE
