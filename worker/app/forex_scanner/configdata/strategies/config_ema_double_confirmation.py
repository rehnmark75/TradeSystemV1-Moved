# ============================================================================
# EMA DOUBLE CONFIRMATION STRATEGY CONFIGURATION
# ============================================================================
# Version: 2.1.0
# Description: EMA crossover strategy with optional prior confirmation requirement
#
# Strategy Logic:
#   1. Detect EMA 9/21 crossovers on 15-minute timeframe
#   2. Validate crossover "success" = price stays on favorable side of EMA 9
#      for SUCCESS_CANDLES consecutive candles (default: 3 = 45 min)
#   3. Optionally require prior successful crossovers before entry
#   4. Apply HTF trend filter, ADX filter for trending markets
#
# v2.1.0 CHANGES (Signal Generation Fixes):
#   - FIX: Reduced MIN_SUCCESSFUL_CROSSOVERS 1→0 (immediate signals, no state dependency)
#   - FIX: Increased MAX_RISK_AFTER_OFFSET_PIPS 20→35 (allow volatile pairs)
#   - FIX: Disabled FVG filter (was blocking 40% of valid signals)
#   - FIX: Reduced ADX threshold 20→15 (allow medium-strength trends)
#   - Analysis: Strategy was generating 0 signals due to state reset + tight filters
#
# v2.2.0 CHANGES (Database Persistence):
#   - NEW: USE_DATABASE_STATE = True (persists crossover history to database)
#   - REVERTED: MIN_SUCCESSFUL_CROSSOVERS back to 1 (now works with DB persistence)
#   - FIXED: "Chicken and egg" problem where state was lost on restart
#   - State now survives scanner restarts, enabling original strategy design
#
# v2.0.0 CHANGES (Limit Orders):
#   - NEW: Limit order support with ATR-based price offsets
#   - BUY orders placed BELOW current price (buy cheaper)
#   - SELL orders placed ABOVE current price (sell higher)
#   - 6-minute auto-expiry for unfilled orders
# ============================================================================

from datetime import time

# ============================================================================
# STRATEGY METADATA
# ============================================================================
STRATEGY_NAME = "EMA_DOUBLE_CONFIRMATION"
STRATEGY_VERSION = "2.2.0"
STRATEGY_DATE = "2025-12-17"
STRATEGY_STATUS = "Database Persistence - Original Strategy Design Restored"

# ============================================================================
# CORE PARAMETERS
# ============================================================================

# EMA Periods (optimized for 15m algo trading)
EMA_FAST_PERIOD = 9           # Fast EMA for crossover detection
EMA_SLOW_PERIOD = 21          # Slow EMA for crossover detection
EMA_TREND_PERIOD = 200        # Trend EMA for additional confirmation

# Success Validation
# A crossover is "successful" if price stays on the favorable side of fast EMA
# for this many consecutive candles after the crossover
SUCCESS_CANDLES = 3           # 3 candles on 15m = 45 min (adjusted for faster EMA)

# Lookback Window
# How far back to look for prior successful crossovers
LOOKBACK_HOURS = 48           # 48 hours = 192 candles on 15m timeframe

# Signal Requirements
# v2.2.0: RESTORED to 1 - Database persistence fixes the "chicken and egg" problem
# Prior issue: State was in-memory only, reset on every restart
# Solution: State now persisted to database (survives restarts)
# Strategy logic: Wait for 1 successful crossover before taking entry signal
MIN_SUCCESSFUL_CROSSOVERS = 1  # v2.2.0: RESTORED from 0 - now works with DB persistence

# ============================================================================
# HIGHER TIMEFRAME TREND FILTER
# ============================================================================

# 4H EMA 21 Trend Filter
# Bullish signals require price ABOVE 4H EMA 21
# Bearish signals require price BELOW 4H EMA 21
HTF_TREND_FILTER_ENABLED = True   # Enable/disable 4H trend filter
HTF_TIMEFRAME = '4h'              # Higher timeframe to check
HTF_EMA_PERIOD = 21               # EMA period on higher timeframe (21 = faster response)
HTF_MIN_BARS = 30                 # Minimum bars needed on HTF

# ============================================================================
# FVG CONFIRMATION FILTER
# ============================================================================
# Require a Fair Value Gap in the signal direction to confirm institutional momentum
# v2.1.0: DISABLED - Was blocking 40% of valid signals in ranging markets

FVG_CONFIRMATION_ENABLED = False  # v2.1.0: DISABLED - too restrictive for intraday
FVG_LOOKBACK_CANDLES = 10         # Look for FVG within last N candles
FVG_MIN_SIZE_PIPS = 2             # Minimum FVG size in pips (lower = more FVGs detected)

# ============================================================================
# ADX TREND STRENGTH FILTER
# ============================================================================
# Only take signals when market is trending (ADX > threshold)
# EMA crossovers perform poorly in ranging/choppy markets

ADX_FILTER_ENABLED = True         # Enable ADX trend strength filter
ADX_MIN_VALUE = 15                # v2.1.0: REDUCED from 20 - allow medium-strength trends
ADX_PERIOD = 14                   # ADX calculation period

# ============================================================================
# CROSSOVER DETECTION
# ============================================================================

# Crossover Detection Settings
CROSSOVER_EPSILON = 1e-8      # Numerical stability threshold
MIN_SEPARATION_PIPS = 1       # Minimum EMA separation to confirm cross

# Crossover direction detection:
# Bullish: prev_ema_21 < prev_ema_50 AND curr_ema_21 > curr_ema_50
# Bearish: prev_ema_21 > prev_ema_50 AND curr_ema_21 < curr_ema_50

# ============================================================================
# SUCCESS VALIDATION
# ============================================================================

# Maximum candles to wait before declaring crossover failed
# If price crosses back within this window, crossover is marked as failed
MAX_VALIDATION_CANDLES = 6    # 1.5 hours on 15m (give some buffer beyond SUCCESS_CANDLES)

# Allow small wick violations during validation?
# If True, only check close prices. If False, also check high/low.
ALLOW_WICK_VIOLATIONS = True  # Only check close prices for simplicity

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

# Minimum confidence to generate signal
MIN_CONFIDENCE = 0.50         # 50% minimum (lowered for initial testing)
HIGH_CONFIDENCE = 0.75        # 75% = high quality signal

# Base confidence (starting point)
BASE_CONFIDENCE = 0.55

# Scoring weights (must sum to 1.0)
CONFIDENCE_WEIGHTS = {
    'crossover_quality': 0.30,    # EMA separation at crossover
    'prior_success_rate': 0.25,   # How convincing were the 2 prior crossovers
    'trend_alignment': 0.25,      # Alignment with EMA 200 trend
    'market_conditions': 0.20,    # RSI zone, ATR levels
}

# Bonus for extended success (crossover held longer than required)
# If prior crossovers held for 6+ candles instead of just 4, add bonus
EXTENDED_SUCCESS_BONUS = 0.05   # 5% confidence boost per extended crossover
EXTENDED_SUCCESS_CANDLES = 6    # Candles required for extended bonus

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# ATR-based stop loss and take profit
STOP_ATR_MULTIPLIER = 2.0       # SL = 2x ATR (provides breathing room)
TARGET_ATR_MULTIPLIER = 4.0     # TP = 4x ATR (2:1 R:R minimum)

# ATR calculation period
ATR_PERIOD = 14

# Minimum/Maximum stops (pips) - safety bounds
MIN_STOP_PIPS = 15              # Never less than 15 pips SL
MAX_STOP_PIPS = 50              # Never more than 50 pips SL
MIN_TARGET_PIPS = 30            # Never less than 30 pips TP

# Risk-Reward requirements
MIN_RR_RATIO = 1.5              # Minimum 1.5:1 R:R
OPTIMAL_RR_RATIO = 2.0          # Optimal R:R for full confidence

# ============================================================================
# v2.0.0: LIMIT ORDER CONFIGURATION
# ============================================================================
# Use limit orders (pending orders) instead of market orders for better entries
# Place orders at OFFSET from current price to get better fill prices
# Auto-expire unfilled orders after configured time

LIMIT_ORDER_ENABLED = True               # v2.0.0: Enable limit orders
LIMIT_EXPIRY_MINUTES = 6                 # v2.0.0: Auto-cancel after 6 min (3 scanner cycles)

# Entry offset configuration (place orders at better prices)
# EMA crossover strategy uses ATR-based offset (adapts to volatility)
LIMIT_OFFSET_ATR_FACTOR = 0.25           # Offset = 25% of ATR (crossovers need tighter entry)
LIMIT_OFFSET_MIN_PIPS = 2.0              # Minimum offset: 2 pips
LIMIT_OFFSET_MAX_PIPS = 6.0              # Maximum offset: 6 pips

# Risk sanity checks after offset
MIN_RISK_AFTER_OFFSET_PIPS = 5.0         # Reject if SL too close after offset
MAX_RISK_AFTER_OFFSET_PIPS = 35.0        # v2.1.0: INCREASED from 20 - allow volatile pairs (GBP, JPY crosses)

# ============================================================================
# SESSION FILTER
# ============================================================================

SESSION_FILTER_ENABLED = False  # Disabled for backtesting - enable for live trading

# Session definitions (UTC times)
LONDON_SESSION_START = time(7, 0)   # 07:00 UTC
LONDON_SESSION_END = time(16, 0)    # 16:00 UTC
NY_SESSION_START = time(12, 0)      # 12:00 UTC
NY_SESSION_END = time(21, 0)        # 21:00 UTC

# Allowed sessions
ALLOWED_SESSIONS = ['london', 'new_york', 'overlap']
BLOCK_ASIAN_SESSION = True          # Block 21:00-07:00 UTC

# ============================================================================
# SIGNAL LIMITS
# ============================================================================

MAX_SIGNALS_PER_PAIR_PER_DAY = 2    # Allow 2 signals per pair per day
SIGNAL_COOLDOWN_HOURS = 2           # Hours between signals on same pair

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

# Pair-specific pip values
PAIR_PIP_VALUES = {
    'CS.D.EURUSD.CEEM.IP': 1.0,     # CEEM uses scaled pricing
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
# STATE MANAGEMENT
# ============================================================================

# v2.2.0: Database persistence for crossover state
# Enables the original strategy design with MIN_SUCCESSFUL_CROSSOVERS >= 1
# State survives scanner restarts and container recreations
USE_DATABASE_STATE = True  # v2.2.0: ENABLED - persistent crossover state

# In-memory cache settings
CACHE_MAX_CROSSOVERS_PER_PAIR = 20  # Max crossovers to keep per pair
CACHE_CLEANUP_INTERVAL_HOURS = 24   # Clean old entries every 24 hours

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

ENABLE_DEBUG_LOGGING = True         # Detailed logging for development
LOG_CROSSOVER_DETECTION = True      # Log when crossovers are detected
LOG_SUCCESS_VALIDATION = True       # Log crossover validation results
LOG_SIGNAL_GENERATION = True        # Log signal generation details
LOG_STATE_CHANGES = True            # Log state tracker changes

# ============================================================================
# BACKTEST SETTINGS
# ============================================================================

BACKTEST_SPREAD_PIPS = 1.5          # Assumed spread for backtesting
BACKTEST_SLIPPAGE_PIPS = 0.5        # Assumed slippage for backtesting

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pip_value(epic: str) -> float:
    """Get pip value for a given epic."""
    return PAIR_PIP_VALUES.get(epic, 0.0001)


def get_lookback_candles(timeframe: str = '15m') -> int:
    """Calculate lookback candles based on timeframe."""
    tf_minutes = {
        '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240
    }
    minutes = tf_minutes.get(timeframe, 15)
    return int((LOOKBACK_HOURS * 60) / minutes)


def get_success_duration_hours() -> float:
    """Calculate hours required for success validation."""
    return (SUCCESS_CANDLES * 15) / 60  # 15m timeframe


def is_session_allowed(hour_utc: int) -> bool:
    """Check if current hour is in allowed trading session."""
    if not SESSION_FILTER_ENABLED:
        return True

    # Block Asian session: 21:00-07:00 UTC
    if BLOCK_ASIAN_SESSION and (hour_utc >= 21 or hour_utc < 7):
        return False

    # Allow London open to NY close: 07:00-21:00 UTC
    return 7 <= hour_utc <= 21
