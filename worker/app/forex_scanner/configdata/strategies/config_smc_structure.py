#!/usr/bin/env python3
"""
SMC Structure Strategy Configuration
Pure structure-based trading using Smart Money Concepts

Strategy Overview:
1. Identify HTF trend (4H structure - HH/HL or LH/LL)
2. Wait for pullback to S/R level
3. Confirm rejection pattern (pin bar, engulfing, etc.)
4. Enter with structure (not indicators)
5. Stop beyond structure (invalidation point)
6. Target next structure level (supply/demand zone)

Design Philosophy:
- No lagging indicators (MACD, RSI, etc.)
- Pure price action and structure
- Institutional logic (where smart money operates)
- Clear structure-based invalidation
- Bot-friendly deterministic rules
"""

# ============================================================================
# STRATEGY IDENTIFICATION
# ============================================================================

STRATEGY_NAME = "SMC_STRUCTURE"
STRATEGY_DESCRIPTION = "Pure structure-based strategy using Smart Money Concepts (price action only)"
STRATEGY_VERSION = "1.0.0"

# ============================================================================
# HIGHER TIMEFRAME (HTF) TREND ANALYSIS
# ============================================================================

# Higher timeframe for trend identification
# The strategy analyzes structure on this timeframe to determine trend
# '4h' provides good balance between noise filtering and responsiveness
SMC_HTF_TIMEFRAME = '4h'

# Lookback period for HTF analysis (in bars)
# How many HTF bars to analyze for trend structure
# 100 bars = ~16 days on 4H = good medium-term trend context
SMC_HTF_LOOKBACK = 100

# Minimum trend strength required (0-1)
# Higher = more conservative (only trade strongest trends)
# 0.50 = moderate requirement (50% trend confidence)
SMC_MIN_TREND_STRENGTH = 0.50

# Swing detection strength for trend analysis
# How many bars on each side to confirm a swing high/low
# 5 = swing must be highest/lowest of 11 bars (5 left + 1 center + 5 right)
SMC_SWING_STRENGTH = 5

# Minimum swing significance (price movement)
# Prevents detecting noise as swings
# 0.0020 = 20 pips for most pairs, 2 pips for JPY pairs
SMC_MIN_SWING_SIGNIFICANCE = 0.0020

# ============================================================================
# SUPPORT/RESISTANCE DETECTION
# ============================================================================

# Lookback period for S/R detection (in bars)
# How far back to look for support/resistance levels
# 100 bars on entry timeframe provides good level history
SMC_SR_LOOKBACK = 100

# Level clustering tolerance (pips)
# Swing points within this range cluster into one level
# 10 pips allows for some spread/noise while keeping levels distinct
SMC_LEVEL_CLUSTER_PIPS = 10

# Minimum touches required for valid level
# Level must be tested this many times to be considered valid
# 2 = minimum (initial formation + one retest)
SMC_MIN_LEVEL_TOUCHES = 2

# Maximum level age (bars)
# Levels older than this are considered stale
# 200 bars keeps levels relevant to current market structure
SMC_MAX_LEVEL_AGE = 200

# Proximity requirement (pips)
# How close price must be to a level to trigger consideration
# 30 pips = accommodates volatile pairs while maintaining relevance
# OPTIMIZED: Increased from 20 to capture valid setups near structure
SMC_SR_PROXIMITY_PIPS = 30

# Minimum level strength for supply/demand zones (0-1)
# Only levels with this strength qualify as strong zones
# 0.70 = strong levels with recent touches and good age
SMC_SUPPLY_DEMAND_MIN_STRENGTH = 0.70

# ============================================================================
# CANDLESTICK PATTERN DETECTION
# ============================================================================

# Minimum pattern strength required (0-1)
# How strong the rejection pattern must be to trigger entry
# 0.50 = moderate patterns (50%+ quality score)
# Lowered from 0.60 to allow more signals
SMC_MIN_PATTERN_STRENGTH = 0.50

# Pattern lookback (bars)
# How many recent bars to check for rejection patterns
# 50 bars = checks last 50 bars for patterns (captures more opportunities)
# Agent analysis: 5 bars too narrow, missing valid pullback patterns
SMC_PATTERN_LOOKBACK_BARS = 50

# Pin bar requirements
SMC_PIN_BAR_MIN_WICK_RATIO = 0.60  # Wick must be 60%+ of total range
SMC_PIN_BAR_MAX_BODY_RATIO = 0.25  # Body must be 25% or less of range

# Engulfing requirements
SMC_ENGULFING_MIN_RATIO = 1.0  # Current candle must fully engulf previous

# Hammer/Shooting star requirements
SMC_HAMMER_MIN_WICK_RATIO = 0.60  # Long wick 60%+
SMC_HAMMER_MAX_BODY_RATIO = 0.30  # Small body 30% or less
SMC_HAMMER_MAX_OPPOSITE_WICK = 0.15  # Opposite wick 15% or less

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Stop loss buffer (pips)
# Additional pips beyond structure invalidation point
# 8 pips = tighter stop for better R:R (avg bar range analysis)
# OPTIMIZED: Reduced from 15 to minimize losses
SMC_SL_BUFFER_PIPS = 8

# Minimum Risk:Reward ratio
# Trade must offer at least this R:R to be valid
# 1.2 = minimum 1.2:1 reward:risk (very relaxed for more signals)
# REDUCED: 1.5 → 1.2 to allow more trades
SMC_MIN_RR_RATIO = 1.2

# Maximum risk per trade (pips)
# Reject trades with stop loss larger than this
# None = no maximum (let structure determine risk)
# Set to value like 50 to cap maximum risk
SMC_MAX_RISK_PIPS = None

# ============================================================================
# PROFIT TAKING
# ============================================================================

# Partial profit enabled
# Close portion of position at intermediate target
SMC_PARTIAL_PROFIT_ENABLED = True

# Partial profit percentage
# What % of position to close at partial target
# 50 = close half, let half run to full TP
SMC_PARTIAL_PROFIT_PERCENT = 50

# Partial profit R:R ratio
# Close partial at this R:R
# 1.0 = take partial profit at 1.0R (breakeven), let rest run to full TP (1.2R+)
# REDUCED: Must be less than SMC_MIN_RR_RATIO (1.2)
SMC_PARTIAL_PROFIT_RR = 1.0

# Move stop to breakeven after partial?
SMC_MOVE_SL_TO_BE_AFTER_PARTIAL = True

# ============================================================================
# TRAILING STOP SYSTEM (Progressive 3-Stage)
# ============================================================================

# Enable trailing stop
# False = use fixed TP only (safer for structure-based strategy)
# True = enable R:R-based progressive trailing stop
SMC_TRAILING_ENABLED = False

# Trailing stop mode (if enabled)
# 'structure_priority' = Only trail on exceptional moves (>3R), preserve structure targets
# 'aggressive' = Trail all profitable trades
# 'conservative' = Trail only after 2R achieved
SMC_TRAILING_MODE = 'structure_priority'

# Minimum R:R before trailing starts (for structure_priority mode)
# 3.0 = only enable trailing if trade reaches 3R (exceptional move beyond structure target)
# This preserves structure-based TP for most trades
SMC_TRAILING_MIN_RR = 3.0

# Progressive trailing stages (R:R thresholds)
# Stage 1: Lock in small profit
SMC_TRAILING_STAGE1_RR = 1.5  # At 1.5R, lock in 4 pips profit
SMC_TRAILING_STAGE1_LOCK_PIPS = 4

# Stage 2: Lock in meaningful profit
SMC_TRAILING_STAGE2_RR = 2.5  # At 2.5R, lock in 12 pips profit
SMC_TRAILING_STAGE2_LOCK_PIPS = 12

# Stage 3: Dynamic trailing (after structure target exceeded)
SMC_TRAILING_STAGE3_RR = 3.0  # At 3R, switch to dynamic trailing
SMC_TRAILING_STAGE3_DISTANCE_PIPS = 20  # Trail 20 pips behind peak

# ============================================================================
# SIGNAL COOLDOWN AND CLUSTERING PREVENTION
# ============================================================================

# Enable cooldown system
# True = prevent signal clustering (recommended for live trading)
# False = disable cooldown (useful for backtesting to see all potential signals)
SMC_COOLDOWN_ENABLED = False  # Disabled for backtesting to see all signals

# Signal cooldown per pair (hours)
# Prevents multiple signals on same pair in short timeframe
# 4 hours = prevents clustering while allowing valid re-entries
SMC_SIGNAL_COOLDOWN_HOURS = 4

# Global signal cooldown across all pairs (minutes)
# Prevents simultaneous signals on multiple pairs
# 30 minutes = staggers entries for better risk distribution
SMC_GLOBAL_COOLDOWN_MINUTES = 30

# Maximum concurrent signals allowed
# Caps total exposure across all pairs
# 3 = maximum 3 open positions at once (diversification + risk control)
SMC_MAX_CONCURRENT_SIGNALS = 3

# Cooldown enforcement level
# 'strict' = hard block on signals within cooldown
# 'warning' = log warning but allow signal (for analysis)
SMC_COOLDOWN_ENFORCEMENT = 'strict'

# ============================================================================
# FILTERS AND CONFLUENCE
# ============================================================================

# Require pullback for entry?
# True = only enter on pullbacks within trend
# False = allow entries on breakouts too
SMC_REQUIRE_PULLBACK = True

# Pullback minimum depth (Fibonacci ratio)
# How much price must retrace to qualify as pullback
# 0.382 = 38.2% Fibonacci level (standard pullback in trends)
# REVERTED: 0.236-0.786 too wide (agent analysis), back to sweet spot
SMC_MIN_PULLBACK_DEPTH = 0.382

# Maximum pullback depth
# Reject pullbacks deeper than this (may indicate reversal)
# 0.618 = 61.8% Fibonacci (deep pullback before reversal risk)
# REVERTED: Back to classic Fibonacci sweet spot (38.2-61.8%)
SMC_MAX_PULLBACK_DEPTH = 0.618

# ============================================================================
# POSITION SIZING (Optional - can be overridden by portfolio manager)
# ============================================================================

# Position sizing method
# 'FIXED_RISK' = risk fixed % of account per trade
# 'FIXED_LOTS' = always trade same lot size
SMC_POSITION_SIZING_METHOD = 'FIXED_RISK'

# Risk percentage per trade (for FIXED_RISK method)
# What % of account to risk per trade
# 1.0 = risk 1% of account per trade
SMC_RISK_PERCENT_PER_TRADE = 1.0

# Fixed lot size (for FIXED_LOTS method)
# Always trade this many lots
SMC_FIXED_LOT_SIZE = 0.1

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

# Verbose logging
# True = detailed step-by-step logs for debugging
# False = only key decisions logged
SMC_VERBOSE_LOGGING = True

# Log rejected signals
# True = log why signals were rejected (useful for debugging)
# False = only log valid signals
SMC_LOG_REJECTED_SIGNALS = True

# ============================================================================
# BACKTESTING SETTINGS
# ============================================================================

# Enable backtesting mode
# When True, uses historical bar data for all calculations
# When False, uses real-time data
SMC_BACKTEST_MODE = False

# Slippage (pips)
# Assumed slippage for backtest entries
# 1 pip = reasonable for major pairs
SMC_BACKTEST_SLIPPAGE_PIPS = 1

# Commission (USD per lot per side)
# Typical broker commission
# 7 = $7 per lot per side (round turn = $14)
SMC_BACKTEST_COMMISSION_PER_LOT = 7.0

# ============================================================================
# TIMEFRAME SETTINGS
# ============================================================================

# Entry timeframe
# The timeframe used for signal generation and pattern detection
# '1h' provides good balance for structure-based entries
SMC_ENTRY_TIMEFRAME = '1h'

# Lookback for entry timeframe analysis (hours)
# How much entry TF data to fetch
# 200 hours = ~8 days of 1H data
SMC_ENTRY_LOOKBACK_HOURS = 200

# ============================================================================
# BOS/CHoCH RE-ENTRY STRATEGY (New Approach)
# ============================================================================

# Enable BOS/CHoCH re-entry strategy
# True = detect BOS/CHoCH on 15m, validate HTF, wait for pullback
# False = use legacy pattern-based entry
SMC_BOS_CHOCH_REENTRY_ENABLED = False  # Disabled - market structure module needs more work

# Detection timeframe for BOS/CHoCH
# '15m' = detect structure breaks on 15-minute timeframe
SMC_BOS_CHOCH_TIMEFRAME = '15m'

# Require 1H timeframe alignment
# True = 1H structure must align with BOS/CHoCH direction
SMC_REQUIRE_1H_ALIGNMENT = True

# Require 4H timeframe alignment
# True = 4H structure must align with BOS/CHoCH direction
SMC_REQUIRE_4H_ALIGNMENT = True

# Lookback for HTF structure validation (bars)
# How many bars to analyze for HTF structure alignment
SMC_HTF_ALIGNMENT_LOOKBACK = 50

# Re-entry zone tolerance (pips)
# ± pips from BOS/CHoCH level to trigger re-entry
# 10 pips = reasonable zone for re-entry
SMC_REENTRY_ZONE_PIPS = 10

# Maximum wait time for pullback (bars)
# Maximum bars to wait for price to return to BOS/CHoCH level
# 20 bars = 5 hours on 15m timeframe
SMC_MAX_WAIT_BARS = 20

# Minimum BOS/CHoCH significance (0-1)
# How significant the structure break must be
# 0.6 = moderate significance requirement
SMC_MIN_BOS_SIGNIFICANCE = 0.6

# Stop loss distance from structure level (pips)
# For BOS/CHoCH re-entry, stop goes beyond structure level
# 10 pips = tight stop at logical invalidation
SMC_BOS_STOP_PIPS = 10

# Make rejection patterns optional (for BOS/CHoCH mode)
# True = patterns boost confidence but don't block signals
# False = patterns required (legacy behavior)
SMC_PATTERNS_OPTIONAL = True

# ============================================================================
# ZERO LAG LIQUIDITY ENTRY TRIGGER
# ============================================================================

# Enable Zero Lag Liquidity as entry trigger
# True = use liquidity breaks/rejections for precise entry timing
# False = use immediate entry at structure level
SMC_USE_ZERO_LAG_ENTRY = False  # Disabled - too restrictive, blocking most signals

# Wick threshold for liquidity detection (0-1)
# Minimum wick-to-body ratio to detect liquidity event
# 0.6 = wick must be 60%+ of total candle range
SMC_ZERO_LAG_WICK_THRESHOLD = 0.6

# Lookback bars for liquidity tracking
# How many bars to track liquidity levels
# 20 bars = good balance for 1H timeframe
SMC_ZERO_LAG_LOOKBACK = 20

# ============================================================================
# EPIC-SPECIFIC OVERRIDES (Optional)
# ============================================================================

# You can override any setting per currency pair
# Example:
# SMC_OVERRIDES = {
#     'USDJPY': {
#         'SMC_MIN_SWING_SIGNIFICANCE': 0.20,  # 20 pips for JPY
#         'SMC_SR_PROXIMITY_PIPS': 15,
#     },
#     'GBPUSD': {
#         'SMC_MIN_RR_RATIO': 2.5,  # More conservative on GBP
#     }
# }

SMC_OVERRIDES = {}

# ============================================================================
# STRATEGY VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    errors = []

    # Validate ranges
    if not (0 <= SMC_MIN_TREND_STRENGTH <= 1):
        errors.append("SMC_MIN_TREND_STRENGTH must be between 0 and 1")

    if not (0 <= SMC_MIN_PATTERN_STRENGTH <= 1):
        errors.append("SMC_MIN_PATTERN_STRENGTH must be between 0 and 1")

    if SMC_MIN_RR_RATIO < 1.0:
        errors.append("SMC_MIN_RR_RATIO must be at least 1.0")

    if SMC_PARTIAL_PROFIT_ENABLED:
        if not (0 < SMC_PARTIAL_PROFIT_PERCENT < 100):
            errors.append("SMC_PARTIAL_PROFIT_PERCENT must be between 0 and 100")

        if SMC_PARTIAL_PROFIT_RR >= SMC_MIN_RR_RATIO:
            errors.append("SMC_PARTIAL_PROFIT_RR must be less than SMC_MIN_RR_RATIO")

    if not (0 <= SMC_MIN_PULLBACK_DEPTH <= 1):
        errors.append("SMC_MIN_PULLBACK_DEPTH must be between 0 and 1")

    if not (0 <= SMC_MAX_PULLBACK_DEPTH <= 1):
        errors.append("SMC_MAX_PULLBACK_DEPTH must be between 0 and 1")

    if SMC_MIN_PULLBACK_DEPTH >= SMC_MAX_PULLBACK_DEPTH:
        errors.append("SMC_MIN_PULLBACK_DEPTH must be less than SMC_MAX_PULLBACK_DEPTH")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    return True


# Validate on import
validate_config()
