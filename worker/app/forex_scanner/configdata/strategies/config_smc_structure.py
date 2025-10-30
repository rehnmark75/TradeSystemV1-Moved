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
# 20 pips = reasonable proximity without being overly restrictive
SMC_SR_PROXIMITY_PIPS = 20

# Minimum level strength for supply/demand zones (0-1)
# Only levels with this strength qualify as strong zones
# 0.70 = strong levels with recent touches and good age
SMC_SUPPLY_DEMAND_MIN_STRENGTH = 0.70

# ============================================================================
# CANDLESTICK PATTERN DETECTION
# ============================================================================

# Minimum pattern strength required (0-1)
# How strong the rejection pattern must be to trigger entry
# 0.70 = strong patterns only (70%+ quality score)
SMC_MIN_PATTERN_STRENGTH = 0.70

# Pattern lookback (bars)
# How many recent bars to check for rejection patterns
# 5 bars = checks last 5 bars for patterns (reasonable recency)
SMC_PATTERN_LOOKBACK_BARS = 5

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
# 5 pips provides cushion for spread and minor wicks
SMC_SL_BUFFER_PIPS = 5

# Minimum Risk:Reward ratio
# Trade must offer at least this R:R to be valid
# 2.0 = minimum 2:1 reward:risk (conservative)
SMC_MIN_RR_RATIO = 2.0

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
# 1.5 = take partial profit at 1.5R, let rest run to full TP (2R+)
SMC_PARTIAL_PROFIT_RR = 1.5

# Move stop to breakeven after partial?
SMC_MOVE_SL_TO_BE_AFTER_PARTIAL = True

# ============================================================================
# FILTERS AND CONFLUENCE
# ============================================================================

# Require pullback for entry?
# True = only enter on pullbacks within trend
# False = allow entries on breakouts too
SMC_REQUIRE_PULLBACK = True

# Pullback minimum depth (Fibonacci ratio)
# How much price must retrace to qualify as pullback
# 0.382 = 38.2% Fibonacci level (common institutional pullback)
SMC_MIN_PULLBACK_DEPTH = 0.382

# Maximum pullback depth
# Reject pullbacks deeper than this (may indicate reversal)
# 0.618 = 61.8% Fibonacci (deeper suggests trend weakness)
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
