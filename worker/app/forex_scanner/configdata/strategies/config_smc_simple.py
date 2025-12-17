# ============================================================================
# SMC SIMPLE STRATEGY CONFIGURATION
# ============================================================================
# Version: 2.1.0 (R:R Root Cause Fixes)
# Description: Simplified 3-tier SMC strategy for intraday forex trading
# Architecture:
#   TIER 1: 4H 50 EMA for directional bias
#   TIER 2: 15m swing break with body-close confirmation (was 1H)
#   TIER 3: 5m pullback OR momentum continuation entry
#
# v2.1.0 R:R ROOT CAUSE FIXES:
#   - FIX: Reduced SL_ATR_MULTIPLIER 1.2→1.0 (tighter stops = better R:R)
#   - FIX: Reduced SL_BUFFER_PIPS 8→6 (less buffer = better R:R)
#   - FIX: Reduced pair-specific SL buffers proportionally
#   - FIX: Increased R:R weight in confidence scoring 10%→15%
#   - NEW: Dynamic swing lookback based on ATR volatility
#   - Analysis: 451 R:R rejections were due to inflated SL, not bad setups
#
# v2.0.0 PHASE 3 LIMIT ORDERS:
#   - NEW: Limit order support with intelligent price offsets
#   - PULLBACK entries: ATR-based offset (3-8 pips, adapts to volatility)
#   - MOMENTUM entries: Fixed offset (4 pips)
#   - 6-minute auto-expiry for unfilled orders
#   - Expected improvement: Win rate 39%→52-55%, Profit Factor 1.24→1.6-1.8
#
# v1.8.0 PHASE 2 LOGIC ENHANCEMENTS:
#   - NEW: Momentum continuation entry mode (price beyond break = valid)
#   - NEW: ATR-based swing validation (adapts to pair volatility)
#   - NEW: Momentum quality filter (strong breakout candles only)
#   - Target: 45%+ WR, 1.4+ PF (improved from Phase 1's 1.24 PF)
#
# v1.7.0: Phase 1 Quick Fixes - Relaxed parameters (1.24 PF, 39.4% WR)
# v1.6.0: Pullback calculation fix - Timeframe alignment
# v1.5.x: Over-optimized (too restrictive, 80% confidence, 38-62% Fib)
# ============================================================================

from datetime import time

# ============================================================================
# STRATEGY METADATA
# ============================================================================
STRATEGY_NAME = "SMC_SIMPLE"
STRATEGY_VERSION = "2.1.0"
STRATEGY_DATE = "2025-12-17"
STRATEGY_STATUS = "R:R Root Cause Fixes - Tighter SL for Better R:R"

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
MIN_DISTANCE_FROM_EMA_PIPS = 6           # v2.0.1: REDUCED from 10 - allow trading closer to EMA

# ============================================================================
# TIER 2: 15M ENTRY TRIGGER (Intermediate Timeframe)
# ============================================================================
# Looking for swing break on 15m as entry confirmation (more frequent than 1H)
# A swing high/low break with BODY CLOSE confirms momentum
# 15m gives 4x more candles than 1H = 4x more break opportunities

TRIGGER_TIMEFRAME = "15m"                # Trigger timeframe (was 1h - too slow)
SWING_LOOKBACK_BARS = 20                 # Base bars to look back for swing detection
SWING_STRENGTH_BARS = 2                  # Bars on each side to confirm swing

# v2.1.0: Dynamic swing lookback based on volatility
# In quiet markets, swings are closer together = use shorter lookback
# In volatile markets, swings are further apart = use longer lookback
# This improves R:R by finding better-spaced swing structures
USE_DYNAMIC_SWING_LOOKBACK = True        # v2.1.0: NEW - adapt lookback to volatility
SWING_LOOKBACK_ATR_LOW = 8               # ATR threshold for low volatility (pips)
SWING_LOOKBACK_ATR_HIGH = 15             # ATR threshold for high volatility (pips)
SWING_LOOKBACK_MIN = 15                  # Minimum lookback bars (quiet market)
SWING_LOOKBACK_MAX = 30                  # Maximum lookback bars (volatile market)

# Body-close confirmation
# NOTE: On 15m timeframe, body-close is too strict - price rarely closes beyond swings
# For 15m trigger, we use wick-based breaks instead (high/low must exceed swing)
REQUIRE_BODY_CLOSE_BREAK = False         # Disabled for 15m - use wick breaks instead
WICK_TOLERANCE_PIPS = 3                  # Tolerance for wick touches

# Volume confirmation
# v1.7.0: RE-ENABLED - improves signal quality without blocking too many
VOLUME_CONFIRMATION_ENABLED = True       # v1.7.0: ENABLED - quality filter
VOLUME_SMA_PERIOD = 20                   # Period for volume moving average
VOLUME_SPIKE_MULTIPLIER = 1.2            # v1.7.0: REDUCED from 1.3 - less strict

# ============================================================================
# TIER 3: 5M EXECUTION (Entry Timeframe)
# ============================================================================
# After 15m break, wait for 5m pullback to Fibonacci zones
# Enter on pullback to get better R:R - 5m gives more precise entries

ENTRY_TIMEFRAME = "5m"                   # Entry/execution timeframe (was 15m)
PULLBACK_ENABLED = True                  # Wait for pullback before entry

# Fibonacci pullback zones (measured from swing to break point)
# v1.7.0: WIDENED - over-tight zones were rejecting valid setups
# v1.5.x: 38%-62% was too restrictive - only 24% band
# Analysis: In trending markets, price often doesn't pull back to golden zone
#           Wider zone catches more valid entries while maintaining quality
FIB_PULLBACK_MIN = 0.236                 # v1.7.0: WIDENED from 0.38 - 23.6% Fib (shallow pullbacks OK)
FIB_PULLBACK_MAX = 0.70                  # v1.7.0: WIDENED from 0.62 - allow deeper pullbacks
FIB_OPTIMAL_ZONE = (0.382, 0.618)        # Golden zone for confidence scoring (not strict filter)

# Pullback timing
MAX_PULLBACK_WAIT_BARS = 12              # Maximum bars to wait for pullback
PULLBACK_CONFIRMATION_BARS = 2           # Bars to confirm pullback

# ============================================================================
# v1.8.0 PHASE 2: MOMENTUM CONTINUATION MODE
# ============================================================================
# Allow entries when price continues beyond break without pullback
# In strong trends, price often doesn't pull back - momentum continuation is valid

MOMENTUM_MODE_ENABLED = True             # v1.8.0: NEW - allow momentum entries
MOMENTUM_MIN_DEPTH = -0.20               # Allow up to 20% beyond break point
MOMENTUM_MAX_DEPTH = 0.0                 # 0% = at break point (no pullback yet)
MOMENTUM_CONFIDENCE_PENALTY = 0.05       # Reduce confidence by 5% for momentum entries

# ============================================================================
# v1.8.0 PHASE 2: ATR-BASED SWING VALIDATION
# ============================================================================
# Replace fixed 10-pip minimum swing range with ATR-based threshold
# Adapts to pair volatility (USDCHF has smaller swings than GBPJPY)

USE_ATR_SWING_VALIDATION = True          # v1.8.0: NEW - dynamic swing validation
ATR_PERIOD = 14                          # ATR calculation period
MIN_SWING_ATR_MULTIPLIER = 0.25          # Minimum swing range = 25% of ATR
FALLBACK_MIN_SWING_PIPS = 5              # Absolute minimum if ATR unavailable

# ============================================================================
# v1.8.0 PHASE 2: MOMENTUM QUALITY FILTER
# ============================================================================
# Filter weak breakouts - only enter on strong momentum candles

MOMENTUM_QUALITY_ENABLED = True          # v1.8.0: NEW - filter weak breakouts
MIN_BREAKOUT_ATR_RATIO = 0.5             # Breakout candle range > 50% of ATR
MIN_BODY_PERCENTAGE = 0.45               # v2.0.1: REDUCED from 0.60 - allow more breakout candles

# ============================================================================
# v2.0.0: LIMIT ORDER CONFIGURATION
# ============================================================================
# Use limit orders (pending orders) instead of market orders for better entries
# Place orders at OFFSET from current price to get better fill prices
# Auto-expire unfilled orders after configured time

LIMIT_ORDER_ENABLED = True               # v2.0.0: Enable limit orders
LIMIT_EXPIRY_MINUTES = 6                 # v2.0.0: Auto-cancel after 6 min (3 scanner cycles)

# Entry offset configuration (place orders at better prices)
# PULLBACK entries: ATR-based offset adapts to volatility
PULLBACK_OFFSET_ATR_FACTOR = 0.3         # Offset = 30% of ATR
PULLBACK_OFFSET_MIN_PIPS = 3.0           # Minimum offset: 3 pips
PULLBACK_OFFSET_MAX_PIPS = 8.0           # Maximum offset: 8 pips

# MOMENTUM entries: Fixed offset (trend is strong, don't wait too long)
MOMENTUM_OFFSET_PIPS = 4.0               # Fixed 4 pip offset for momentum entries

# Risk sanity checks after offset
MIN_RISK_AFTER_OFFSET_PIPS = 5.0         # Reject if SL too close after offset
MAX_RISK_AFTER_OFFSET_PIPS = 20.0        # v2.0.1: REDUCED from 40 - max 20 pip SL for better R:R

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# R:R Requirements
# v1.7.0: REDUCED R:R requirements - 2.5 was unrealistic for intraday
# Analysis: Most intraday setups achieve 1.5-2.0 R:R, 2.5+ is too restrictive
MIN_RR_RATIO = 1.5                       # v1.7.0: REDUCED from 2.5 - realistic for intraday
OPTIMAL_RR_RATIO = 2.5                   # v1.7.0: REDUCED from 3.5 - achievable premium setups
MAX_RR_RATIO = 5.0                       # v1.7.0: REDUCED from 6.0 - reasonable cap

# Stop Loss
# v2.1.0: REDUCED buffer and ATR multiplier to improve R:R ratios
# Analysis: 451 R:R rejections (0.01-0.56) were caused by inflated SL distances
# Root cause: SL_BUFFER + ATR_MULTIPLIER combined created 25-30 pip stops
# With tight swings (20 pips), this destroyed R:R (10 reward / 30 risk = 0.33)
# Solution: Tighter stops allow more signals while maintaining edge
SL_BUFFER_PIPS = 6                       # v2.1.0: REDUCED from 8 - tighter stops for better R:R
SL_ATR_MULTIPLIER = 1.0                  # v2.1.0: REDUCED from 1.2 - less ATR inflation
USE_ATR_STOP = True                      # Keep ATR-based adaptive stops

# Take Profit
# v1.7.0: REDUCED minimum TP - 15 pips was rejecting valid tight-structure setups
# Analysis: Rely more on R:R ratio, allow smaller TPs with good R:R
MIN_TP_PIPS = 8                          # v1.7.0: REDUCED from 15 - allow tighter structures
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
    'CS.D.EURUSD.CEEM.IP',  # CEEM uses scaled pricing (11646 instead of 1.1646)
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
    'CS.D.EURUSD.CEEM.IP': 1.0,  # CEEM uses scaled pricing, 1 pip = 1 point
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

# ============================================================================
# v2.1.0: PAIR-SPECIFIC SL BUFFERS (REDUCED)
# ============================================================================
# Override default SL_BUFFER_PIPS for specific pairs based on volatility
# v2.1.0: REDUCED all buffers by ~25% to improve R:R ratios
# Analysis: Large buffers were main cause of R:R rejections (0.01-0.56 values)

PAIR_SL_BUFFERS = {
    # Low volatility / low liquidity pairs
    'USDCHF': 9,            # v2.1.0: REDUCED from 12 (still accounts for spreads)
    'CS.D.USDCHF.MINI.IP': 9,

    # JPY crosses are volatile - but tighter stops improve R:R
    'AUDJPY': 12,           # v2.1.0: REDUCED from 15
    'CS.D.AUDJPY.MINI.IP': 12,
    'EURJPY': 14,           # v2.1.0: REDUCED from 18
    'CS.D.EURJPY.MINI.IP': 14,
    'GBPJPY': 14,           # v2.1.0: REDUCED from 18
    'CS.D.GBPJPY.MINI.IP': 14,
    'USDJPY': 9,            # v2.1.0: REDUCED from 12
    'CS.D.USDJPY.MINI.IP': 9,

    # Commodity currencies
    'USDCAD': 8,            # v2.1.0: REDUCED from 10
    'CS.D.USDCAD.MINI.IP': 8,

    # Cable
    'GBPUSD': 8,            # v2.1.0: REDUCED from 10
    'CS.D.GBPUSD.MINI.IP': 8,

    # Default for majors (EURUSD, AUDUSD, NZDUSD) = SL_BUFFER_PIPS (6)
}

# ============================================================================
# v1.9.0: PAIR-SPECIFIC CONFIDENCE FLOORS
# ============================================================================
# Higher confidence required for volatile / difficult pairs

PAIR_MIN_CONFIDENCE = {
    # Volatile JPY crosses need higher confidence
    'EURJPY': 0.72,
    'CS.D.EURJPY.MINI.IP': 0.72,
    'GBPJPY': 0.72,
    'CS.D.GBPJPY.MINI.IP': 0.72,
    'AUDJPY': 0.70,
    'CS.D.AUDJPY.MINI.IP': 0.70,

    # Low liquidity pairs
    'USDCHF': 0.70,
    'CS.D.USDCHF.MINI.IP': 0.70,

    # Default for others = MIN_CONFIDENCE_THRESHOLD (0.60)
}

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================
# Simple confidence calculation based on setup quality

# Confidence thresholds
# v1.7.0: REDUCED confidence threshold - 80% was too restrictive
# Analysis: With wider Fib zones and better R:R, lower confidence is acceptable
# The tight 80% threshold combined with tight Fib zones left almost no signals
MIN_CONFIDENCE_THRESHOLD = 0.60          # v1.7.0: REDUCED from 0.80 - allow more signals
HIGH_CONFIDENCE_THRESHOLD = 0.75         # v1.7.0: REDUCED from 0.90 - achievable premium tier

# Scoring weights (must sum to 1.0)
# v2.1.0: Increased R:R weight since it directly impacts profitability
CONFIDENCE_WEIGHTS = {
    'ema_alignment': 0.25,               # 4H EMA alignment strength
    'swing_break_quality': 0.25,         # v2.1.0: REDUCED from 0.30 - How clean was the break
    'pullback_depth': 0.20,              # Pullback to optimal Fib zone
    'volume_confirmation': 0.15,         # Volume spike on break
    'rr_ratio': 0.15,                    # v2.1.0: INCREASED from 0.10 - R:R directly affects profit
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
