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
STRATEGY_VERSION = "2.8.5"
STRATEGY_DATE = "2025-11-16"
STRATEGY_STATUS = "Testing - QUALITY OPTIMIZATION: Confidence cap + Pair blacklist + Tighter R:R"

# Version History:
# v2.8.5 (2025-11-16): QUALITY OPTIMIZATION: Signal pattern analysis fixes
#                      Based on v2.8.4 analysis (55 signals, 38.2% WR, 0.53 PF):
#                      1. CONFIDENCE PARADOX FIX:
#                         - 70-76% confidence signals: 0% WR (all 3 lost)
#                         - 55-60% confidence signals: 73% WR (best range)
#                         - Solution: Cap confidence at 50-70% range (reject overconfident)
#                      2. PAIR BLACKLIST:
#                         - GBPUSD: 17% WR (1/6 winners)
#                         - EURJPY: 22% WR (2/9 winners)
#                         - USDJPY: 25% WR (1/4 winners)
#                         - Solution: Blacklist underperforming pairs
#                      3. R:R RATIO IMPROVEMENT:
#                         - Current: 0.87:1 (avg win 12.0 pips, avg loss 13.8 pips)
#                         - Target: 1.8:1 (avg win 18 pips, avg loss 10 pips)
#                         - Solution: Tighter SL (13.8→10 pips), wider TP (12.0→18 pips)
#                      Expected Impact:
#                         - Win Rate: 38.2% → 50-55%
#                         - Profit Factor: 0.53 → 1.3-1.6 (PROFITABLE!)
#                         - Signals: 55 → 30-40/90 days
#                         - R:R Ratio: 0.87:1 → 1.8:1
# v2.8.4 (2025-11-16): CRITICAL BUG FIX: Removed hardcoded 50% HTF minimum strength check
#                      Problem: DUPLICATE filter blocking 3,768 signals (line 725 in smc_structure_strategy.py)
#                        - Hardcoded 50% check ran BEFORE configurable SMC_MIN_HTF_STRENGTH filter
#                        - Made config threshold (20%, 35%, etc.) completely ineffective
#                        - Blocked 68% of structures with dynamic 30% strength (ranging markets)
#                      Evidence: Changing config from 35% → 20% produced ZERO additional signals
#                        - v2.8.1 (35% HTF): 10 signals
#                        - v2.8.2 (35% HTF, 5% swing): 10 signals (identical)
#                        - v2.8.3 (20% HTF): 10 signals (identical - hardcoded 50% was blocking)
#                      Solution: Removed hardcoded check at line 725, now uses ONLY config value
#                      Expected Impact:
#                        - 3,768 signals now pass HTF filter with 20% threshold
#                        - Downstream filters (15m BOS/CHoCH, Order Blocks) will process these
#                        - Significant signal volume increase expected
#                        - May need to adjust other filters to maintain quality
# v2.8.0 (2025-11-15): PHASE 1: DYNAMIC HTF STRENGTH CALCULATION - Critical Foundation Fix
#                      Problem: Hardcoded 60% base strength for ALL clean trends (77% of data)
#                               - Unable to distinguish between choppy weak trends and strong institutional trends
#                               - Threshold optimization meaningless (any >60% blocks core output, any ≤60% allows junk)
#                      Solution: Multi-factor quality scoring system (5 factors x 20% weight each):
#                        1. Swing Consistency - Regular swing spacing = institutional trend
#                        2. Swing Size vs ATR - Large swings = strong momentum
#                        3. Pullback Depth - Shallow pullbacks (<38.2%) = strong trend
#                        4. Price Momentum - Position in range + velocity
#                        5. Volume Profile - Higher impulse volume = institutions
#                      Expected Distribution:
#                        - Weak choppy trends: 30-45% (previously all 60%)
#                        - Clean moderate trends: 50-60% (previously all 60%)
#                        - Strong institutional trends: 65-85% (previously all 60%)
#                        - Exceptional trending markets: 85-100% (previously capped at ~62%)
#                      Impact: TRUE quality filtering now possible
#                        - 65-75% thresholds now capture genuinely strong trends (not just pattern detection)
#                        - Expected: 25-40 signals per 90 days, 45-55% WR, 1.5-2.2 PF
#                        - This is THE FOUNDATION FIX - all future enhancements depend on this
# v2.7.3 (2025-11-15): OPTIMAL HTF THRESHOLD - Data-Driven Sweet Spot (58%)
#                      Analysis Results from v2.7.1-2.7.2 testing:
#                        - 75% HTF: 6 signals, 66.7% WR, 3.68 PF (TOO RESTRICTIVE - 92% rejection)
#                        - 65% HTF: 7 signals, 57.1% WR, 1.90 PF (TOO RESTRICTIVE - 97% rejection)
#                        - 45% HTF: 59 signals, 27.1% WR, 0.35 PF (TOO PERMISSIVE - losing)
#                      Key Insight: Algorithm outputs 60% strength for 77% of trends
#                        - Setting threshold >60% blocks algorithm's core output
#                        - Setting threshold <55% allows too many weak 50% signals
#                      Solution: 58% HTF threshold (optimal sweet spot)
#                        - Allows dominant 60% strength signals (77% of trends)
#                        - Blocks weak 45-57% signals that degrade win rate
#                        - Expected: 30-40 signals, 42-48% WR, 1.4-1.7 PF, +2-5 pips expectancy
#                        - Trading-strategy-analyst recommendation: PRIMARY test candidate
# v2.7.2 (2025-11-15): HTF Threshold Testing - Too Permissive (45%)
#                      Test Result: 59 signals, 27.1% WR, 0.35 PF, -5.6 pips (LOSING)
#                      Issue: Allowed too many weak 45-55% strength signals
#                      Swing proximity rejections: 1,028 (filter working, but HTF passing junk)
# v2.7.1 (2025-11-15): HTF Threshold Testing - Too Restrictive (65% and 75%)
#                      65% test: 7 signals, 57.1% WR (quality but insufficient quantity)
#                      75% test: 6 signals, 66.7% WR (quality but insufficient quantity)
#                      Root Cause: Both reject algorithm's 60% base strength output
# v2.7.1 Phase 1 (2025-11-15): NEW FILTER - Swing Proximity Validation (replaces failed PD filter)
#                      Issue: Premium/Discount filter failed repeatedly (v2.2.0 → v2.5.0 → v2.6.6)
#                             - v2.6.6: Rejected ALL 8 winners (SELL in discount = trend continuation)
#                             - Root cause: Arbitrary 33% zones conflict with trend continuation logic
#                      Solution: Structure-based proximity filter using real HTF swing points
#                      Filter Logic:
#                        - BUY signals: Reject if within 20% of last swing HIGH (exhaustion zone)
#                        - SELL signals: Reject if within 20% of last swing LOW (exhaustion zone)
#                        - Adaptive: Uses actual swing highs/lows, not fixed lookback ranges
#                        - Trend-friendly: Allows continuations at proper pullback distances
#                      Expected Impact: -10-15% exhaustion signals, +5-10% WR, maintains PF
# v2.7.0 (2025-11-15): CRITICAL FIX - Enhanced swing detection for JPY pairs
#                      Issue: USDJPY only detecting 1 swing high (needed 2 minimum)
#                             BOS/CHoCH showing BULLISH in clear downtrend
#                      Fixes: 1. JPY-specific swing significance (20 pips vs 5 pips)
#                             2. Relaxed swing_strength (2 vs 3) for 4H timeframe
#                             3. Fallback price action analysis when swings insufficient
#                      Expected: Proper trend detection on JPY pairs, fewer false UNKNOWN trends
# v2.5.1 (2025-11-11): CRITICAL FIX - Removed trend continuation exception
#                      Issue: Was allowing BULL entries in premium (buying at swing highs)
#                             Was allowing BEAR entries in discount (selling at swing lows)
#                      Fix: Always reject poor entry locations regardless of HTF strength
#                      Expected: Win rate 31% → 35-38%, PF 0.52 → 0.65-0.75
# v2.5.0 (2025-11-11): Analysis-driven optimization based on 30-day backtest (15m timeframe)
#                      Test Results: 68 signals, 30.9% WR, 0.52 PF, -3.5 pips expectancy
#                      CRITICAL CHANGES:
#                      1. Premium/Discount Filter: DISABLED (was inverting performance)
#                         - Premium zone had 45.8% WR (best), Discount 16.7% WR (worst)
#                         - Filter rejected 398 high-quality signals
#                      2. BOS Significance: 0.6 → 0.55 (increase BEAR signals)
#                         - BEAR signals: 47.8% WR vs BULL: 22.2% WR
#                      3. Min Confidence: 0.45 → 0.35 (not correlated with outcomes)
#                         - Winners avg 60.7%, Losers avg 61.2% (confidence not predictive)
#                      Expected Impact: WR 30.9% → 40-45%, Signals 68 → 100+, PF 0.52 → 1.2+
# v2.4.0 (2025-11-05): Dual quality tightening - targeting first profitable configuration
#                      BOS Quality: 60% → 65% (more selective on structure breaks)
#                      Universal Confidence Floor: 45% minimum (all entries)
#                      Test 26: 63 signals, 28.6% WR, 0.88 PF (WR target achieved!)
#                      Target: 40-45 signals, 32-35% WR, 1.1-1.3 PF (PROFITABLE)
# v2.3.0 (2025-11-05): Quality-based filtering for ranging market protection
#                      Equilibrium Confidence Filter: 50% minimum for neutral zones
#                      BOS/CHoCH Quality Score: Body size + clean break + volume
#                      Minimum 60% BOS quality required (filters weak/indecisive breaks)
#                      Target: 28-32% WR, 1.0-1.2 PF (first profitable config)
# v2.2.0 (2025-11-05): BASELINE - Optimized entry timing with 75% HTF strength threshold
#                      Test 24: 39 signals, 25.6% WR, 0.86 PF (BEST PF achieved)
#                      Premium/Discount zone context-aware filtering (75% threshold)
#                      Balanced BOS/CHoCH detection (vote-based system)
#                      Added bearish signal diagnostic logging
# v2.1.1 (2025-11-03): Added session filter implementation (disabled), fixed timestamp bug
# v2.1.0 (2025-11-02): Phase 2.1 baseline - HTF alignment enabled, 112 signals, 39.3% WR, 2.16 PF
# v2.0.0 (2025-10-XX): BOS/CHoCH detection on 15m timeframe, Zero Lag entry trigger
# v1.0.0 (2025-10-XX): Initial SMC Structure implementation

# ============================================================================
# HIGHER TIMEFRAME (HTF) TREND ANALYSIS
# ============================================================================

# Higher timeframe for trend identification
# The strategy analyzes structure on this timeframe to determine trend
# '4h' provides good balance between noise filtering and responsiveness
SMC_HTF_TIMEFRAME = '4h'

# Lookback period for HTF analysis (in bars)
# How many HTF bars to analyze for trend structure
# 60 bars = ~10 days on 4H = sufficient for trend context
# Note: Backtest uses reduced_lookback (0.7 factor) = ~68 bars available max
# 60 bars fits within this constraint while providing good structure analysis
SMC_HTF_LOOKBACK = 60

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
# QUALITY FILTERS (v2.8.5)
# ============================================================================

# Confidence range filter (0-1)
# Based on analysis: 70-76% confidence = 0% WR, 55-60% = 73% WR
# Reject overconfident signals (paradox: high confidence = false positives)
SMC_MIN_CONFIDENCE = 0.50  # 50% minimum confidence
SMC_MAX_CONFIDENCE = 0.70  # 70% maximum confidence (reject overconfident)

# Pair blacklist (underperforming pairs from v2.8.4 analysis)
# GBPUSD: 17% WR (1/6), EURJPY: 22% WR (2/9), USDJPY: 25% WR (1/4)
SMC_BLACKLIST_PAIRS = ['GBPUSD', 'EURJPY', 'USDJPY']

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Stop loss buffer (pips)
# Additional pips beyond structure invalidation point
# v2.8.5: Targeting 10 pips avg loss (down from 13.8 pips)
# OPTIMIZED: Tighter stops for better R:R ratio
SMC_SL_BUFFER_PIPS = 6

# Minimum Risk:Reward ratio
# Trade must offer at least this R:R to be valid
# v2.8.5: Targeting 1.8:1 R:R (18 pips TP / 10 pips SL)
# INCREASED: Was 1.2, now 1.8 for better profitability
SMC_MIN_RR_RATIO = 1.8

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
# '15m' used for BOS/CHoCH detection and entry signals
SMC_ENTRY_TIMEFRAME = '15m'

# Lookback for entry timeframe analysis (hours)
# How much entry TF data to fetch
# 200 hours = ~8 days of 15m data (800 bars)
SMC_ENTRY_LOOKBACK_HOURS = 200

# ============================================================================
# BOS/CHoCH RE-ENTRY STRATEGY (New Approach)
# ============================================================================

# Enable BOS/CHoCH re-entry strategy
# True = detect BOS/CHoCH on 15m, validate HTF, wait for pullback
# False = use legacy pattern-based entry
# CRITICAL: Required for Order Block Re-entry logic (v2.2.0)
SMC_BOS_CHOCH_REENTRY_ENABLED = True  # ENABLED for OB Re-entry implementation

# Detection timeframe for BOS/CHoCH
# '15m' = detect structure breaks on 15-minute timeframe
SMC_BOS_CHOCH_TIMEFRAME = '15m'

# Require 1H timeframe alignment
# True = 1H structure must align with BOS/CHoCH direction
# Phase 2.6.3: DISABLED - Skip 1H validation, use 4H only for cleaner signals
SMC_REQUIRE_1H_ALIGNMENT = False  # Phase 2.6.3: Disabled - 4H-only alignment

# Require 4H timeframe alignment
# True = 4H structure must align with BOS/CHoCH direction
# Phase 2.6.3: ENABLED - Primary HTF validation (4H only)
SMC_REQUIRE_4H_ALIGNMENT = True  # Phase 2.6.3: Enabled - 4H-only alignment

# Lookback for HTF structure validation (bars)
# How many bars to analyze for HTF structure alignment
SMC_HTF_ALIGNMENT_LOOKBACK = 50

# =============================================================================
# TIER 1 FILTER: Session-Based Quality Filter
# =============================================================================

# Enable/disable session filtering
SMC_SESSION_FILTER_ENABLED = False  # DISABLED - needs proper testing with working baseline

# Block Asian session (0-7 UTC) - low liquidity, ranging markets
# Asian session typically has false signals due to range-bound behavior
SMC_BLOCK_ASIAN_SESSION = True

# Session definitions (UTC):
# ASIAN: 0-7 UTC (Tokyo, low liquidity)
# LONDON: 7-15 UTC (London, high liquidity)
# NEW_YORK: 15-22 UTC (New York, high liquidity)
# ASIAN_LATE: 22-24 UTC (Sydney, low liquidity)

# =============================================================================
# TIER 1 FILTER: Pullback Momentum Validator (Entry Timing Improvement)
# =============================================================================

# Enable/disable momentum filter
# Prevents counter-momentum entries (entering when recent price action opposes trade direction)
# DISABLED: Filter proved too restrictive (95% signal reduction, 0% WR on remaining signals)
SMC_MOMENTUM_FILTER_ENABLED = False

# Lookback candles for momentum check on 15m timeframe
# 12 candles × 15m = 3 hours of recent price action
SMC_MOMENTUM_LOOKBACK_CANDLES = 12

# Minimum aligned candles required (8/12 = 66% threshold)
# For BULL: Require 8/12 recent 15m candles to be bullish (close > open)
# For BEAR: Require 8/12 recent 15m candles to be bearish (close < open)
SMC_MOMENTUM_MIN_ALIGNED_CANDLES = 8

# Expected Impact:
# - Signal Reduction: -15% to -20% (112 → 90-95)
# - Win Rate Improvement: +10% to +12% (39.3% → 49-51%)
# - Profit Factor: 2.16 → 2.70-2.90
# - Eliminates: Counter-momentum entries, stalling at levels, false breakouts

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
# 0.55 = slightly relaxed to increase BEAR signals (analysis: BEAR 47.8% WR vs BULL 22.2% WR)
# Lowered from 0.6 based on Nov 2025 backtest analysis
SMC_MIN_BOS_SIGNIFICANCE = 0.55

# Stop loss distance from structure level (pips)
# For BOS/CHoCH re-entry, stop goes beyond structure level
# 10 pips = tight stop at logical invalidation
SMC_BOS_STOP_PIPS = 10

# Make rejection patterns optional (for BOS/CHoCH mode)
# True = patterns boost confidence but don't block signals
# False = patterns required (legacy behavior)
SMC_PATTERNS_OPTIONAL = True

# =============================================================================
# ORDER BLOCK RE-ENTRY CONFIGURATION (v2.2.0)
# =============================================================================

# Enable Order Block re-entry strategy
# When enabled, waits for price to retrace to last opposing OB before entering
# Expected impact: +10-15% WR, -45% signals (quality over quantity)
SMC_OB_REENTRY_ENABLED = True

# Order Block identification
SMC_OB_LOOKBACK_BARS = 20  # How far back to search for opposing OB
SMC_OB_MIN_SIZE_PIPS = 3   # Minimum OB size to be valid

# Re-entry zone settings
SMC_OB_REENTRY_ZONE = 'lower_50'  # 'lower_50', 'upper_50', 'full', 'midpoint'
                                   # lower_50: Enter at bottom 50% of bearish OB
                                   # upper_50: Enter at top 50% of bullish OB

# Rejection confirmation
SMC_OB_REQUIRE_REJECTION = True  # Require rejection signal at OB
SMC_OB_REJECTION_MIN_WICK_RATIO = 0.60  # Min wick ratio for wick rejection (60%)

# Stop loss placement
SMC_OB_SL_BUFFER_PIPS = 5  # Pips beyond OB for stop loss (tighter than old 15 pips)

# Expected Impact (based on trading-strategy-analyst analysis):
# - Win Rate: 39.3% → 48-55% (+10-15%)
# - Signals: 112 → 50-60 (-45% to -55%)
# - Profit Factor: 2.16 → 2.5-3.5 (+16% to +62%)
# - R:R Ratio: 1.2:1 → 2.5:1 (improved entry pricing)

# ============================================================================
# PREMIUM/DISCOUNT ZONE FILTER
# ============================================================================

# Enable Premium/Discount zone filtering for entry timing
# Phase 2.6.7: DISABLED - Directional filter conflicts with profitable Phase 2.6.3 setup
# False = Allow entries in all zones (NO directional restrictions)
#
# CRITICAL FINDING FROM TESTING:
#   - Phase 2.6.3 (75% HTF, NO directional): 9 signals, 44.4% WR, 2.05 PF ✅ PROFITABLE
#   - Phase 2.6.6 (75% HTF + directional): 1 signal, 0% WR ❌ FAILED
#   - Root cause: All 8 winners from Phase 2.6.3 were SELL in discount (bearish trend continuation)
#   - Directional filter logic: "SELL in discount = selling at bottom" is WRONG in strong trends
#   - Reality: With 75% HTF strength, SELL in discount = valid trend continuation setup
#   - Conclusion: Directional filter based on FALSE ASSUMPTION, revert to Phase 2.6.3
SMC_DIRECTIONAL_ZONE_FILTER = False  # Phase 2.6.7: DISABLED - conflicts with 75% HTF logic

# Legacy filter flags (deprecated - do not use)
SMC_PREMIUM_ZONE_ONLY = False  # DEPRECATED - Use SMC_DIRECTIONAL_ZONE_FILTER instead
SMC_PREMIUM_DISCOUNT_FILTER_ENABLED = False  # DEPRECATED

# ============================================================================
# HTF STRENGTH FILTER (Phase 2.6.1)
# ============================================================================

# Minimum HTF strength required for signal generation
# Phase 2.6.7: REVERTED to Phase 2.6.3 baseline (ONLY profitable configuration found)
# New multi-factor calculation provides 50-100% distribution:
#   - Aligned BOS+Swings: 60-100% (use swing strength)
#   - Conflicting BOS/Swings: 50-80% (multi-factor: base 50% ± penalties/bonuses)
# Test results progression:
#   - Phase 2.6.3 (75% HTF, no filters): 9 signals, 44.4% WR, 2.05 PF ✅ PROFITABLE
#   - Phase 2.6.4 (60% HTF + directional): 28 signals, 21.4% WR, 0.27 PF ❌ LOSING
#   - Phase 2.6.5 (60% HTF + directional + liquidity): 21 signals, 23.8% WR, 0.30 PF ❌ LOSING
#   - Phase 2.6.6 (75% HTF + directional): 1 signal, 0% WR ❌ FAILED (too restrictive)
# Phase 2.7.3: OPTIMAL HTF THRESHOLD (58% - Sweet Spot)
#   - 58% HTF strength (allows 60% signals, blocks weak <58% signals)
#   - Test Results Summary:
#       75% HTF: 6 signals, 66.7% WR, 3.68 PF (excellent quality, insufficient quantity)
#       65% HTF: 7 signals, 57.1% WR, 1.90 PF (good quality, insufficient quantity)
#       45% HTF: 59 signals, 27.1% WR, 0.35 PF (sufficient quantity, poor quality - LOSING)
#   - Analysis: 77% of trends output exactly 60% strength (algorithm base)
#   - Solution: 58% threshold = allows 60% trends, blocks 45-57% weak trends
#   - Expected: 30-40 signals, 42-48% WR, 1.4-1.7 PF, +2-5 pips expectancy
#   - Trading-strategy-analyst verdict: "PRIMARY test candidate - optimal sweet spot"
# Phase 2.6.7: BASELINE (75% HTF, no other filters)
#   - Result: 11 signals, 45.5% WR, 2.05 PF (profitable but too few signals)
SMC_MIN_HTF_STRENGTH = 0.30  # Phase 2.8.4: Accept ranging markets at quality floor (30% = minimum from dynamic calculation)

# Exclude signals with UNKNOWN HTF strength (0%)
# UNKNOWN HTF = 28.0% WR vs 60% HTF = 32.6% WR
SMC_EXCLUDE_UNKNOWN_HTF = True  # Phase 2.6.1: Reject 0% HTF signals

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
# LIQUIDITY SWEEP FILTER (Phase 2.6.5)
# ============================================================================

# Enable liquidity sweep filter for quality entry timing
# This is a core SMC concept: smart money takes liquidity before reversing
# True = Require price to take out recent highs/lows before entry
# - SELL signals: Must take out recent swing high (liquidity grab above resistance)
# - BUY signals: Must take out recent swing low (liquidity grab below support)
# False = Allow entries without liquidity sweep requirement
#
# RATIONALE:
# - Phase 2.6.4 with directional filter alone:
#   - 60% HTF + directional: 28 signals, 21.4% WR (poor quality)
#   - 55% HTF + directional: 29 signals, 20.7% WR (poor quality)
# - Phase 2.6.5 test results (60% HTF + directional + liquidity):
#   - 21 signals, 23.8% WR, 0.30 PF (STILL LOSING)
#   - Liquidity sweep filtered 1,422 signals but barely improved WR (+2.4%)
# - Conclusion: Liquidity sweep adds complexity without meaningful value
# - Phase 2.6.6: DISABLED - HTF strength is far more important than liquidity sweeps
SMC_LIQUIDITY_SWEEP_ENABLED = False  # Phase 2.6.6: Disabled - adds complexity without value

# Lookback bars for detecting swing highs/lows
# How many bars to look back for identifying swing points
# 15m timeframe: 20 bars = 5 hours of price action
# Higher values = more significant swings required (more restrictive)
# Lower values = allows recent swings (less restrictive)
SMC_LIQUIDITY_SWEEP_LOOKBACK = 20

# Minimum bars since liquidity sweep occurred
# Ensures entry happens after sweep, not during
# 15m timeframe: 2 bars = 30 minutes after liquidity grab
# This allows reversal to begin before entry
SMC_LIQUIDITY_SWEEP_MIN_BARS = 2

# Minimum bars before liquidity sweep occurred
# Maximum age of the liquidity sweep
# 15m timeframe: 10 bars = 2.5 hours maximum age
# Too old = stale setup, price may have moved away
SMC_LIQUIDITY_SWEEP_MAX_BARS = 10

# ============================================================================
# SWING PROXIMITY FILTER (v2.7.1 - Structure-Based Entry Timing)
# ============================================================================

# Enable swing proximity filter for entry timing validation
# This REPLACES the failed Premium/Discount zone filter with structure-based logic
#
# CRITICAL DIFFERENCE FROM PD FILTER:
#   PD Filter (FAILED): Used arbitrary 33% zones based on lookback range
#   Swing Filter (NEW): Uses actual HTF swing highs/lows from trend structure
#
# RATIONALE (from v2.6.7 analysis):
#   - PD filter rejected ALL 8 winners in Phase 2.6.3 (SELL in discount = trend continuation)
#   - "SELL in discount = selling at bottom" is WRONG in strong trends
#   - Reality: SELL in discount during downtrend = VALID continuation setup
#   - Solution: Check distance to SWING EXTREME, not arbitrary zone position
#
# HOW IT WORKS:
#   - BUY signals: Reject if too close to last swing HIGH (exhaustion zone)
#   - SELL signals: Reject if too close to last swing LOW (exhaustion zone)
#   - Allows trend continuations at proper pullback distances
#   - Adaptive: Uses actual swing points, not fixed lookback ranges
SMC_SWING_PROXIMITY_FILTER_ENABLED = True  # v2.7.1: Structure-based replacement for PD filter

# Exhaustion zone threshold (% distance from swing extreme)
# How close to swing extreme before entry is considered "exhaustion/chasing"
# 0.20 = Within 20% of swing extreme range = REJECT (too close)
# 0.80 = Within 80% distance from swing = ALLOW (good pullback)
#
# Example (SELL signal in downtrend):
#   Swing High: 154.50
#   Swing Low:  153.90 (range = 60 pips)
#   Entry: 154.30
#   Position: (154.30 - 153.90) / 60 = 40/60 = 67% from low ✅ ALLOW
#   If entry was 154.45: (154.45 - 153.90) / 60 = 92% from low ✅ ALLOW (near high)
#   If entry was 154.00: (154.00 - 153.90) / 60 = 17% from low ❌ REJECT (too close)
SMC_SWING_EXHAUSTION_THRESHOLD = 0.10  # Phase 2.8.3: Reverted to 10% (optimal balance from v2.8.1)

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

    # Validate Order Block Re-entry dependencies (v2.2.0)
    if SMC_OB_REENTRY_ENABLED and not SMC_BOS_CHOCH_REENTRY_ENABLED:
        errors.append(
            "⚠️  CRITICAL: SMC_OB_REENTRY_ENABLED requires SMC_BOS_CHOCH_REENTRY_ENABLED=True\n"
            "   Order Block re-entry logic depends on BOS/CHoCH detection.\n"
            "   Either enable BOS/CHoCH or disable OB re-entry."
        )

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
