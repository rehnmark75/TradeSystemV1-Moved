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
# 15 pips = reasonable proximity for 15m timeframe (REVERTED: 10 was too tight, missed valid touches)
SMC_SR_PROXIMITY_PIPS = 15

# Minimum level strength for supply/demand zones (0-1)
# Only levels with this strength qualify as strong zones
# 0.70 = strong levels with recent touches and good age
SMC_SUPPLY_DEMAND_MIN_STRENGTH = 0.70

# ============================================================================
# CANDLESTICK PATTERN DETECTION
# ============================================================================

# Minimum pattern strength required (0-1)
# How strong the rejection pattern must be to trigger entry
# 0.55 = moderate quality patterns (REVERTED: 0.70 was too strict, reduced wins more than losses)
SMC_MIN_PATTERN_STRENGTH = 0.55

# Pattern lookback (bars)
# How many recent bars to check for rejection patterns
# 10 bars = checks last 10 hours for recent patterns (balanced)
# OPTIMIZED: Reduced from 50 (too stale) to capture fresh pullback patterns
SMC_PATTERN_LOOKBACK_BARS = 10

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
# 15 pips = allows for 15m timeframe noise (IMPROVED from 10)
# 15m timeframe has natural ±10 pip volatility, 15-pip buffer reduces whipsaw losses
SMC_SL_BUFFER_PIPS = 15

# Minimum Risk:Reward ratio
# Trade must offer at least this R:R to be valid
# 2.0 = minimum 2:1 reward:risk (industry standard for profitable trading)
# OPTIMIZED: Increased to 2.0 to filter low-quality trades
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
# 0.8 = take partial profit at 0.8R, let rest run to full TP (1.2R+)
# OPTIMIZED: Set to 0.8 to work with min R:R of 1.2
SMC_PARTIAL_PROFIT_RR = 0.8

# Move stop to breakeven after partial?
SMC_MOVE_SL_TO_BE_AFTER_PARTIAL = True

# ============================================================================
# TRAILING STOP SYSTEM (Progressive 3-Stage)
# ============================================================================

# Enable trailing stop
# False = use fixed TP only (safer for structure-based strategy)
# True = enable R:R-based progressive trailing stop
# ANALYSIS: 44% of losses were TRAILING_STOP exits - keeping DISABLED for consistent targets
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
# 0.382 = 38.2% Fibonacci level (REVERTED: 0.50 was too deep, missed good entries)
SMC_MIN_PULLBACK_DEPTH = 0.382

# Maximum pullback depth
# Reject pullbacks deeper than this (may indicate reversal)
# 0.618 = 61.8% Fibonacci (deep pullback before reversal risk)
# KEPT: Classic Fibonacci sweet spot for maximum pullback depth
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
# '15m' provides more precise entries with tighter stops
# OPTIMIZED: Changed from 1h to 15m for better entry timing
SMC_ENTRY_TIMEFRAME = '15m'

# Lookback for entry timeframe analysis (hours)
# How much entry TF data to fetch
# 100 hours = ~400 bars of 15m data (~17 days)
SMC_ENTRY_LOOKBACK_HOURS = 100

# ============================================================================
# BOS/CHoCH RE-ENTRY STRATEGY (New Approach)
# ============================================================================

# Enable BOS/CHoCH re-entry strategy
# True = detect BOS/CHoCH on 15m, validate HTF, wait for pullback
# False = use legacy pattern-based entry
# TEMP: Disabled - causing 0 signals, needs debugging
SMC_BOS_CHOCH_REENTRY_ENABLED = False

# Detection timeframe for BOS/CHoCH
# '15m' = detect structure breaks on 15-minute timeframe
SMC_BOS_CHOCH_TIMEFRAME = '15m'

# Require 1H timeframe alignment
# True = 1H structure must align with BOS/CHoCH direction
# OPTIMIZED: Disabled to reduce over-filtering (agent recommendation)
SMC_REQUIRE_1H_ALIGNMENT = False

# Require 4H timeframe alignment
# True = 4H structure must align with BOS/CHoCH direction
SMC_REQUIRE_4H_ALIGNMENT = True

# Lookback for HTF structure validation (bars)
# How many bars to analyze for HTF structure alignment
SMC_HTF_ALIGNMENT_LOOKBACK = 50

# Re-entry zone tolerance (pips)
# ± pips from BOS/CHoCH level to trigger re-entry
# 20 pips = widened zone for more re-entry opportunities
# OPTIMIZED: Increased from 10 to 20 pips (balanced profile)
SMC_REENTRY_ZONE_PIPS = 20

# Maximum wait time for pullback (bars)
# Maximum bars to wait for price to return to BOS/CHoCH level
# 20 bars = 5 hours on 15m timeframe
SMC_MAX_WAIT_BARS = 20

# Minimum BOS/CHoCH significance (0-1)
# How significant the structure break must be
# 0.60 = balanced threshold (halfway between 0.50 baseline and 0.70 strict)
SMC_MIN_BOS_SIGNIFICANCE = 0.60

# Stop loss distance from structure level (pips)
# For BOS/CHoCH re-entry, stop goes beyond structure level
# 10 pips = tight stop at logical invalidation
SMC_BOS_STOP_PIPS = 10

# Make rejection patterns optional (for BOS/CHoCH mode)
# True = patterns boost confidence but don't block signals
# False = patterns required (legacy behavior)
SMC_PATTERNS_OPTIONAL = True

# ============================================================================
# FAIR VALUE GAP (FVG) PRIORITY ENTRY
# ============================================================================

# Enable Fair Value Gap as PRIMARY entry trigger
# FVGs have 70-80% fill rate (highest statistical edge in SMC)
# True = prioritize FVG entries over pattern-based entries
SMC_FVG_ENTRY_ENABLED = True

# FVG entry priority level (1 = highest, 4 = lowest)
# Tier 1: FVG + Liquidity Sweep (80%+ edge)
# Tier 2: FVG + Rejection Pattern (70%+ edge)
# Tier 3: Order Block + Pattern (60%+ edge)
# Tier 4: S/R + Pattern (50%+ edge)
SMC_FVG_PRIORITY = 1

# FVG entry zone percentage
# Enter at what % fill of the FVG
# 0.50 = enter at 50% fill of gap (sweet spot per research)
SMC_FVG_ENTRY_ZONE_PERCENT = 0.50

# Maximum FVG age (bars)
# Reject FVGs older than this
# 50 bars on 4H = ~8 days (recent institutional activity)
SMC_FVG_MAX_AGE_BARS = 50

# Minimum FVG size (pips)
# Gap must be at least this many pips to be valid
# 5 pips = filters noise while capturing real imbalances
SMC_FVG_MIN_SIZE_PIPS = 5

# ============================================================================
# TIERED ENTRY LOGIC SYSTEM
# ============================================================================

# Enable tiered entry system
# True = use priority-based entry logic (FVG > OB > S/R)
# False = use legacy single-method entry
SMC_USE_TIERED_ENTRY_LOGIC = True

# Minimum tier level for entry acceptance
# 1 = only Tier 1 entries (FVG + Liquidity)
# 2 = Tier 1-2 entries (FVG-based)
# 3 = Tier 1-3 entries (includes Order Blocks)
# 4 = All tiers (includes S/R patterns)
SMC_MIN_TIER_FOR_ENTRY = 2

# Confidence boost per tier
# Tier 1 signals get higher base confidence
SMC_TIER1_CONFIDENCE_BOOST = 0.25  # +25% for FVG + Liquidity
SMC_TIER2_CONFIDENCE_BOOST = 0.15  # +15% for FVG + Pattern
SMC_TIER3_CONFIDENCE_BOOST = 0.10  # +10% for OB + Pattern
SMC_TIER4_CONFIDENCE_BOOST = 0.00  # No boost for S/R only

# ============================================================================
# MULTI-TIMEFRAME ENTRY REFINEMENT
# ============================================================================

# Enable 15M timeframe for entry refinement
# True = use 15M for precise liquidity sweep detection
# False = use only 1H for entry timing
SMC_USE_15M_ENTRY_REFINEMENT = True

# Entry refinement timeframe
# Timeframe for precise entry trigger detection
SMC_ENTRY_REFINEMENT_TIMEFRAME = '15m'

# Require 15M liquidity sweep for Tier 1 entries
# True = Tier 1 requires 15M sweep confirmation
# False = Tier 1 can use 1H sweeps
SMC_REQUIRE_15M_SWEEP_TIER1 = False  # Optional for more signals

# ============================================================================
# ORDER BLOCK VALIDATION (For Tier 3 Entries)
# ============================================================================

# Minimum order block timeframe
# Only use order blocks from this timeframe or higher
# '4h' = only 4H+ order blocks (more reliable)
SMC_OB_MIN_TIMEFRAME = '4h'

# Maximum order block age (bars)
# Only trade recent order blocks
# 30 bars on 4H = ~5 days (fresh institutional interest)
SMC_OB_MAX_AGE_BARS = 30

# Minimum order block retest count
# How many times OB must be tested
# 0 = fresh untested OBs okay (first retest)
# 1 = require at least one previous retest
SMC_OB_MIN_RETEST_COUNT = 0

# ============================================================================
# ZERO LAG LIQUIDITY ENTRY TRIGGER
# ============================================================================

# Enable Zero Lag Liquidity as entry trigger
# True = use liquidity breaks/rejections for precise entry timing
# False = use immediate entry at structure level
# OPTIMIZED: Enabled for better entry timing on 15m timeframe
SMC_USE_ZERO_LAG_ENTRY = True

# Wick threshold for liquidity detection (0-1)
# Minimum wick-to-body ratio to detect liquidity event
# 0.6 = wick must be 60%+ of total candle range (baseline)
SMC_ZERO_LAG_WICK_THRESHOLD = 0.6

# Lookback bars for liquidity tracking
# How many bars to track liquidity levels
# ENHANCED: Changed from 20 to 5 bars for liquidity reaction detection (lookback mechanism)
# Note: Internal liquidity level tracking still uses 20 bars, but reaction detection uses 5
SMC_ZERO_LAG_LOOKBACK = 5

# Tolerance for liquidity detection at structure level
# How many pips away from structure to still detect interaction
# ENHANCED: Widened from 10 to 15 pips for better detection (agent recommendation)
SMC_ZERO_LAG_TOLERANCE_PIPS = 15

# Make Zero Lag Liquidity optional (not required blocker)
# True = Zero Lag is bonus confirmation (+confidence), not mandatory
# False = Zero Lag required (blocks signals if not present)
# KEEP TRUE: Allows structure-only fallback when Zero Lag not detected
SMC_ZERO_LAG_OPTIONAL = True

# Enable fallback to structure-only entry when Zero Lag not detected
# True = Use lookback-based structure entry as fallback
# False = Strict Zero Lag requirement (may miss valid entries)
# NEW PARAMETER: Recommended TRUE for automated systems
SMC_ZERO_LAG_FALLBACK_ENABLED = True

# Confidence boost when Zero Lag liquidity detected
# Bonus confidence added to signal when liquidity sweep confirmed
# 0.20 = +20% confidence bonus (agent research recommendation)
SMC_ZERO_LAG_CONFIDENCE_BOOST = 0.20

# ============================================================================
# LOOKBACK-BASED ENTRY PARAMETERS (Phase 1 Enhancement)
# ============================================================================

# Lookback bars for structure entry detection
# How many recent bars to search for structure interaction
# 5 bars = 75 minutes on 15m timeframe (captures entries between scans)
# NEW PARAMETER: Core fix for entry timing issues
SMC_ENTRY_LOOKBACK_BARS = 5

# Maximum age for entry validity (bars)
# Entries older than this are rejected as "stale"
# 8 bars = 2 hours on 15m timeframe
# Prevents using outdated entries that may no longer be valid
# NEW PARAMETER: Entry freshness validation
SMC_ENTRY_MAX_AGE_BARS = 8

# Tolerance for structure interaction (pips)
# Distance from structure level to consider as "touched"
# 15 pips = reasonable tolerance for 15m timeframe
# NEW PARAMETER: Structure touch detection
SMC_ENTRY_STRUCTURE_TOLERANCE_PIPS = 15

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
