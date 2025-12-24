# ============================================================================
# SMC SIMPLE STRATEGY CONFIGURATION
# ============================================================================
# Version: 2.3.1 (Capped Structural Stops)
# Description: Simplified 3-tier SMC strategy for intraday forex trading
# Architecture:
#   TIER 1: 4H 50 EMA for directional bias
#   TIER 2: 15m swing break with body-close confirmation (was 1H)
#   TIER 3: 5m pullback OR momentum continuation entry
#
# v2.4.0 ATR-BASED SL CAP:
#   - PROBLEM: Fixed 55 pip cap was still 7x ATR on low-volatility pairs
#   - ANALYSIS: Trade 1594 (GBPUSD) had 48 pip SL with 7.4 pip ATR = massive risk
#   - ROOT CAUSE: Structural stops + fixed cap ignored current volatility
#   - SOLUTION: Dynamic cap = min(3x ATR, 30 pips absolute)
#   - BENEFIT: SL now proportional to market conditions (22 pips for GBPUSD)
#   - IMPACT: Win rate needed drops from 70%+ to ~50% for breakeven
#
# v2.3.1 CAPPED STRUCTURAL STOPS:
#   - FIX: Structural stops were creating 100+ pip risk on 15m timeframe
#   - ANALYSIS: 89 risk rejections/week (USDJPY avg 111.5 pips, GBPUSD 76.0 pips)
#   - ROOT CAUSE: On 15m, swing structures span wider than on higher TFs
#   - SOLUTION: Cap structural stop at max_risk_after_offset_pips (55 pips)
#   - LOGIC: Use max(structural_stop, max_risk_stop) for BULL, min() for BEAR
#   - TRADE-OFF: Some SLs may be inside swing structure (acceptable for 15m algo)
#   - BENEFIT: Recovers ~90 valid signals/week that were rejected for excess risk
#
# v2.2.0 CONFIDENCE SCORING REDESIGN:
#   - FIX: swing_break_quality was in config but NOT implemented - NOW IMPLEMENTED
#   - FIX: pullback was over-weighted at 40% (25% + 15% fib) - NOW 20%
#   - FIX: volume was binary (15%/5%) - NOW gradient based on spike magnitude
#   - FIX: EMA used fixed 30 pips - NOW ATR-normalized for cross-pair fairness
#   - NEW: Balanced 5-component scoring (each 20% weight)
#   - Expected: +2-4% win rate improvement, better tier alignment
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
#   - v2.2.0: CHANGED to stop-entry style (momentum confirmation)
#   - BUY orders placed ABOVE current price (enter when price breaks up)
#   - SELL orders placed BELOW current price (enter when price breaks down)
#   - Max offset reduced to 3 pips, 15-minute auto-expiry
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
STRATEGY_VERSION = "2.5.0"
STRATEGY_DATE = "2025-12-23"
STRATEGY_STATUS = "Pair-Specific Blocking - USDCHF filtered on weak setups"

# ============================================================================
# TIER 1: 4H DIRECTIONAL BIAS (Higher Timeframe)
# ============================================================================
# The 50 EMA is used by institutions for trend direction
# Price above EMA = bullish bias (look for longs only)
# Price below EMA = bearish bias (look for shorts only)

HTF_TIMEFRAME = "4h"                    # Higher timeframe for bias
EMA_PERIOD = 50                          # 50-period EMA (institutional standard)
EMA_BUFFER_PIPS = 2.5                    # v2.3.0: REDUCED from 3 - rejection analysis shows 38 signals lost at 2.0-2.9 pips

# Price position requirements
REQUIRE_CLOSE_BEYOND_EMA = True          # Candle must CLOSE beyond EMA (not just wick)
MIN_DISTANCE_FROM_EMA_PIPS = 3           # v2.1.2: REDUCED from 6 - allow entries 3.5-4.5 pips from EMA

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
MOMENTUM_MIN_DEPTH = -0.50               # v2.3.0: RELAXED from -0.35 - rejection analysis shows many at -40% to -60%
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
MIN_BODY_PERCENTAGE = 0.35               # v2.3.0: REDUCED from 0.45 - rejection analysis: 234 signals lost at 35-44%

# ============================================================================
# v2.0.0: LIMIT ORDER CONFIGURATION
# ============================================================================
# Use limit orders (pending orders) instead of market orders for better entries
# Place orders at OFFSET from current price to get better fill prices
# Auto-expire unfilled orders after configured time

LIMIT_ORDER_ENABLED = True               # v2.0.0: Enable limit orders
LIMIT_EXPIRY_MINUTES = 15                # v2.3.0: Auto-cancel after 15 min (1 candle on 15m TF)

# Entry offset configuration (stop-entry style: confirm direction continuation)
# BUY limit ABOVE price, SELL limit BELOW price (momentum confirmation)
# PULLBACK entries: ATR-based offset adapts to volatility
PULLBACK_OFFSET_ATR_FACTOR = 0.2         # Offset = 20% of ATR (reduced for tighter entry)
PULLBACK_OFFSET_MIN_PIPS = 2.0           # Minimum offset: 2 pips
PULLBACK_OFFSET_MAX_PIPS = 3.0           # Maximum offset: 3 pips (user request: reduced from 8)

# MOMENTUM entries: Fixed offset (trend is strong, don't wait too long)
MOMENTUM_OFFSET_PIPS = 3.0               # Fixed 3 pip offset for momentum entries (reduced from 4)

# Risk sanity checks after offset
MIN_RISK_AFTER_OFFSET_PIPS = 5.0         # Reject if SL too close after offset

# v2.4.0: ATR-BASED SL CAP
# Problem: Fixed 55 pip cap was still too large (7x ATR on some pairs)
# Analysis: Trade 1594 had 48 pip SL with only 7.4 pip ATR - massive risk
# Solution: Cap SL at ATR multiple, with hard absolute cap
# Expected: Better risk/reward, ~50% win rate needed vs 70%+ before
MAX_SL_ATR_MULTIPLIER = 3.0              # Maximum SL = 3x ATR (dynamic per pair)
MAX_SL_ABSOLUTE_PIPS = 30.0              # Hard cap regardless of ATR
# Legacy parameter kept for backwards compatibility
MAX_RISK_AFTER_OFFSET_PIPS = 55.0        # Fallback if ATR unavailable

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

# DEPRECATED in v3.0.0: MAX_SIGNALS_PER_PAIR_PER_DAY was never enforced in code
# Signal frequency is now managed by the adaptive cooldown system below

MAX_CONCURRENT_SIGNALS = 3               # Maximum concurrent open signals
SIGNAL_COOLDOWN_HOURS = 3                # Fallback when ADAPTIVE_COOLDOWN_ENABLED=False

# ============================================================================
# v3.0.0: ADAPTIVE COOLDOWN SYSTEM
# ============================================================================
# Dynamic cooldown based on trade outcomes, win rates, and market context.
# Replaces static cooldown with intelligent, context-aware timing.

ADAPTIVE_COOLDOWN_ENABLED = True         # Enable adaptive cooldown (False = use static SIGNAL_COOLDOWN_HOURS)

# Base cooldown (starting point before adjustments)
BASE_COOLDOWN_HOURS = 2.0                # Base cooldown before adjustments

# Trade outcome multipliers
COOLDOWN_AFTER_WIN_MULTIPLIER = 0.5      # Reduce cooldown by 50% after a winning trade
COOLDOWN_AFTER_LOSS_MULTIPLIER = 1.5     # Increase cooldown by 50% after a losing trade

# Consecutive loss handling
CONSECUTIVE_LOSS_PENALTY_HOURS = 1.0     # Add 1 hour per consecutive loss on same pair
MAX_CONSECUTIVE_LOSSES_BEFORE_BLOCK = 3  # Block pair entirely after 3 consecutive losses
CONSECUTIVE_LOSS_BLOCK_HOURS = 8.0       # Block duration after max consecutive losses

# Win rate-based adjustments (rolling window of recent trades)
WIN_RATE_LOOKBACK_TRADES = 20            # Number of trades to calculate rolling win rate
HIGH_WIN_RATE_THRESHOLD = 0.60           # 60%+ win rate = high performer
LOW_WIN_RATE_THRESHOLD = 0.40            # Below 40% = poor performer
CRITICAL_WIN_RATE_THRESHOLD = 0.30       # Below 30% = consider blocking

HIGH_WIN_RATE_COOLDOWN_REDUCTION = 0.25  # Reduce cooldown by 25% for high win rate pairs
LOW_WIN_RATE_COOLDOWN_INCREASE = 0.50    # Increase cooldown by 50% for low win rate pairs

# Market context adjustments
HIGH_VOLATILITY_ATR_MULTIPLIER = 1.5     # ATR > 1.5x 20-period average = high volatility
VOLATILITY_COOLDOWN_ADJUSTMENT = 0.30    # +30% cooldown during high volatility
STRONG_TREND_COOLDOWN_REDUCTION = 0.30   # -30% cooldown when strong EMA trend alignment

# Session-based adjustments
SESSION_CHANGE_RESET_COOLDOWN = True     # Reset cooldown on session change (london→new_york)

# Absolute bounds (safety limits)
MIN_COOLDOWN_HOURS = 1.0                 # Never less than 1 hour
MAX_COOLDOWN_HOURS = 12.0                # Never more than 12 hours

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
# v2.1.2: REDUCED for forward testing - use strategy's MIN_CONFIDENCE_THRESHOLD (50%)

PAIR_MIN_CONFIDENCE = {
    # v2.1.2: REDUCED from 70-72% to 49% for forward testing
    # Using 49% to allow signals at exactly 50% to pass (< comparison)
    # Re-raise after collecting live performance data

    # Volatile JPY crosses - reduced for forward testing
    'EURJPY': 0.49,
    'CS.D.EURJPY.MINI.IP': 0.49,
    'GBPJPY': 0.49,
    'CS.D.GBPJPY.MINI.IP': 0.49,
    'AUDJPY': 0.49,
    'CS.D.AUDJPY.MINI.IP': 0.49,

    # Low liquidity pairs - reduced for forward testing
    'USDCHF': 0.49,
    'CS.D.USDCHF.MINI.IP': 0.49,

    # Default for others = MIN_CONFIDENCE_THRESHOLD (0.50)
}

# ============================================================================
# v2.5.0: PAIR-SPECIFIC BLOCKING CONDITIONS
# ============================================================================
# Block signals for specific pairs when certain unfavorable conditions combine.
# Based on analysis of live trading outcomes showing consistent losses.
#
# USDCHF Analysis (Trade 1620, Alert 6562):
#   - 139 closed trades, 26.62% win rate, -$4,917.77 total P&L (worst pair)
#   - Avg loss: -$54.04 per trade
#   - High EMA distance (>60 pips): 0% win rate, -$515 loss
#   - No volume confirmation: 0% win rate, -$1,031 loss
#   - Not in optimal zone: 25% win rate, -$480 loss
#   - Combined bad conditions: 0% win rate
#
# These blocking conditions use tier2_swing.volume_confirmed from strategy_indicators
# (NOT the top-level volume_confirmation column which is always false for SMC_SIMPLE)

PAIR_BLOCKING_CONDITIONS = {
    # USDCHF: Block when multiple weak factors combine
    # This pair has the worst performance (-$4,917) and needs strict filtering
    'CS.D.USDCHF.MINI.IP': {
        'enabled': True,
        'description': 'Block USDCHF on weak setups - worst performing pair',
        'conditions': {
            # Block when EMA distance is too extended (price has moved too far)
            'max_ema_distance_pips': 60.0,  # >60 pips from 4H EMA = 0% win rate

            # Require volume confirmation for this pair (normally optional)
            'require_volume_confirmation': True,  # No volume = 0% win rate

            # Block momentum entries (negative pullback) without volume
            'block_momentum_without_volume': True,

            # Require optimal zone entry for this difficult pair
            'require_optimal_zone': False,  # Set True to be more strict

            # Minimum confidence override for this pair
            'min_confidence_override': 0.60,  # Higher than default 0.50
        },
        'blocking_logic': 'any',  # 'any' = block if ANY condition fails, 'all' = block only if ALL fail
    },

    # USDCHF alternative epic format
    'USDCHF': {
        'enabled': True,
        'description': 'Block USDCHF on weak setups - worst performing pair',
        'conditions': {
            'max_ema_distance_pips': 60.0,
            'require_volume_confirmation': True,
            'block_momentum_without_volume': True,
            'require_optimal_zone': False,
            'min_confidence_override': 0.60,
        },
        'blocking_logic': 'any',
    },

    # Template for other pairs (disabled by default)
    # Copy and modify for pairs showing consistent losses
    # 'CS.D.EURJPY.MINI.IP': {
    #     'enabled': False,
    #     'description': 'Template - not active',
    #     'conditions': {
    #         'max_ema_distance_pips': 80.0,
    #         'require_volume_confirmation': False,
    #         'block_momentum_without_volume': False,
    #         'require_optimal_zone': False,
    #         'min_confidence_override': None,
    #     },
    #     'blocking_logic': 'any',
    # },
}

def should_block_signal(epic: str, signal_data: dict) -> tuple[bool, str]:
    """
    Check if a signal should be blocked based on pair-specific conditions.

    Args:
        epic: The trading pair epic (e.g., 'CS.D.USDCHF.MINI.IP')
        signal_data: Dict containing signal details with keys:
            - ema_distance_pips: float - Distance from 4H EMA in pips
            - volume_confirmed: bool - Whether volume spike was confirmed
            - in_optimal_zone: bool - Whether entry is in Fib optimal zone
            - pullback_depth: float - Pullback percentage (negative = momentum)
            - confidence_score: float - Overall confidence score

    Returns:
        Tuple of (should_block: bool, reason: str)
    """
    # Check if pair has blocking conditions configured
    config = PAIR_BLOCKING_CONDITIONS.get(epic)
    if not config or not config.get('enabled', False):
        return False, ""

    conditions = config.get('conditions', {})
    blocking_logic = config.get('blocking_logic', 'any')

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
            block_reasons.append(f"Momentum entry (pullback {pullback_depth:.1%}) without volume")

    # Check optimal zone requirement
    if conditions.get('require_optimal_zone', False):
        in_optimal = signal_data.get('in_optimal_zone', False)
        if not in_optimal:
            block_reasons.append("Not in optimal Fib zone (required for this pair)")

    # Check confidence override
    min_conf = conditions.get('min_confidence_override')
    if min_conf is not None:
        confidence = signal_data.get('confidence_score', 0)
        if confidence < min_conf:
            block_reasons.append(f"Confidence {confidence:.1%} < {min_conf:.1%} minimum for this pair")

    # Determine if signal should be blocked
    if blocking_logic == 'any' and block_reasons:
        return True, f"Blocked: {'; '.join(block_reasons)}"
    elif blocking_logic == 'all' and len(block_reasons) == len([c for c in conditions.values() if c]):
        return True, f"Blocked (all conditions failed): {'; '.join(block_reasons)}"

    return False, ""

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================
# v2.2.0: Redesigned confidence scoring with proper tier alignment
# Fixed issues: swing_break_quality was in config but NOT implemented,
#               pullback was over-weighted at 40% (25% + 15% fib_accuracy)

# Confidence thresholds
# v1.7.0: REDUCED confidence threshold - 80% was too restrictive
# Analysis: With wider Fib zones and better R:R, lower confidence is acceptable
# The tight 80% threshold combined with tight Fib zones left almost no signals
MIN_CONFIDENCE_THRESHOLD = 0.50         # v1.7.0: REDUCED from 0.80 - allow more signals
HIGH_CONFIDENCE_THRESHOLD = 0.75         # v1.7.0: REDUCED from 0.90 - achievable premium tier

# Scoring weights (must sum to 1.0)
# v2.2.0: Balanced 5-component scoring (each 20%)
# - EMA alignment: ATR-normalized (3 ATR from EMA = max)
# - Swing break quality: Body %, break strength, recency (NOW IMPLEMENTED)
# - Volume strength: Gradient scoring based on spike magnitude (was binary)
# - Pullback quality: Combined zone + Fib accuracy (was over-weighted at 40%)
# - R:R ratio: Scales toward 3:1
CONFIDENCE_WEIGHTS = {
    'ema_alignment': 0.20,               # v2.2.0: ATR-normalized EMA distance
    'swing_break_quality': 0.20,         # v2.2.0: NOW IMPLEMENTED - body %, strength, recency
    'volume_strength': 0.20,             # v2.2.0: Gradient scoring (was binary 15%/5%)
    'pullback_quality': 0.20,            # v2.2.0: Combined zone + Fib (was 40% split)
    'rr_ratio': 0.20,                    # v2.2.0: R:R quality toward 3:1
}

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

ENABLE_DEBUG_LOGGING = True              # Detailed logging for development
LOG_REJECTED_SIGNALS = True              # Log why signals were rejected
LOG_SWING_DETECTION = False              # Log swing point detection (verbose)
LOG_EMA_CHECKS = False                   # Log EMA alignment checks (verbose)

# ============================================================================
# REJECTION TRACKING (for strategy improvement analysis)
# ============================================================================
# Store rejection data to database for later analysis in Streamlit dashboard
# Captures market state at each rejection point to identify improvement areas
# Storage: ~1-2 MB/day (~500-900 rejections/day across 9 pairs)

REJECTION_TRACKING_ENABLED = True        # Toggle to enable/disable DB storage
REJECTION_BATCH_SIZE = 50                # Batch inserts for performance
REJECTION_LOG_TO_CONSOLE = False         # Also log rejections to console (verbose)
REJECTION_RETENTION_DAYS = 90            # Auto-cleanup data older than this

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
