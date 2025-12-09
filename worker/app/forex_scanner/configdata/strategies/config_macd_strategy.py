# configdata/strategies/config_macd_strategy.py
"""
MACD Strategy Configuration
Configuration module for MACD strategy settings
"""

# =============================================================================
# STRATEGY MODE SELECTION (NEW - Confluence vs Legacy)
# =============================================================================

# Choose strategy mode
MACD_USE_CONFLUENCE_MODE = True  # True = NEW Confluence strategy, False = Legacy MACD crossover

# =============================================================================
# CONFLUENCE STRATEGY CONFIGURATION (NEW)
# =============================================================================
# Multi-timeframe confluence system:
# - H4 MACD: Trend direction filter
# - H1: Fibonacci + swing level calculation
# - 15M: Price action pattern entry triggers

# Fibonacci Settings
MACD_CONFLUENCE_FIB_LEVELS = [0.382, 0.5, 0.618, 0.786]  # Retracement levels to calculate
MACD_CONFLUENCE_FIB_LOOKBACK = 50  # H1 bars to look back for swing detection
MACD_CONFLUENCE_FIB_SWING_STRENGTH = 5  # Bars left/right for valid swing (5 = 11-bar swing)
MACD_CONFLUENCE_MIN_SWING_PIPS = 15.0  # Minimum swing size to consider (filters noise)
MACD_CONFLUENCE_FIB_TOLERANCE_PIPS = 5.0  # How close price must be to Fib level

# Confluence Zone Settings
MACD_CONFLUENCE_MODE = 'moderate'  # 'strict', 'moderate', or 'loose'
# - strict: Requires 3+ factors (Fib + swing + 1 other)
# - moderate: Requires 2+ factors (Fib + swing OR Fib + EMA/round number)
# - loose: Fib level only

MACD_CONFLUENCE_MIN_SCORE = 2.0  # Minimum confluence score to enter trade
MACD_CONFLUENCE_PROXIMITY_PIPS = 5.0  # How close factors must be to count as aligned

# Entry Pattern Settings (15M candlestick patterns)
MACD_CONFLUENCE_ENTRY_PATTERNS = ['engulfing', 'pin_bar', 'inside_bar']
MACD_CONFLUENCE_MIN_PATTERN_QUALITY = 60  # Minimum pattern quality score (0-100)
MACD_CONFLUENCE_REQUIRE_PATTERN = True  # Require candlestick pattern at confluence zone

# Pattern detector parameters
MACD_PATTERN_MIN_BODY_RATIO = 0.6  # Engulfing: min body size as ratio of range
MACD_PATTERN_MIN_ENGULF_RATIO = 1.1  # Engulfing: current body must be 1.1x previous
MACD_PATTERN_MAX_PIN_BODY_RATIO = 0.3  # Pin bar: max body size (30% of range)
MACD_PATTERN_MIN_PIN_WICK_RATIO = 2.0  # Pin bar: wick must be 2x body size

# Multi-Timeframe H4 Filter
MACD_CONFLUENCE_H4_FILTER_ENABLED = True  # Require H4 MACD trend alignment
MACD_CONFLUENCE_H4_REQUIRE_EXPANSION = True  # Require H4 histogram expanding
MACD_CONFLUENCE_H4_MIN_HISTOGRAM = 0.00001  # Minimum H4 histogram magnitude
MACD_CONFLUENCE_H4_ALLOW_NEUTRAL = False  # Allow entries when H4 is neutral

# Stop Loss & Take Profit (Confluence Mode)
MACD_CONFLUENCE_USE_15M_STOPS = True  # Use tighter 15M swing-based stops
MACD_CONFLUENCE_STOP_ATR_MULTIPLIER = 1.5  # ATR multiplier for stop loss (tighter)
MACD_CONFLUENCE_TP_ATR_MULTIPLIER = 3.0  # ATR multiplier for take profit
MACD_CONFLUENCE_MIN_STOP_PIPS = 10.0  # Minimum stop distance (spread + slippage)
MACD_CONFLUENCE_MAX_STOP_PIPS = 30.0  # Maximum stop distance (risk control)
MACD_CONFLUENCE_MIN_RR_RATIO = 2.0  # Minimum risk:reward ratio
MACD_CONFLUENCE_USE_STRUCTURE_TARGETS = True  # Target next swing level instead of fixed ATR

# Pair-Specific Fibonacci Sensitivity (optional overrides)
MACD_CONFLUENCE_PAIR_SETTINGS = {
    'EURUSD': {
        'fib_lookback': 50,
        'min_swing_pips': 15.0,
        'confluence_mode': 'moderate'
    },
    'GBPUSD': {
        'fib_lookback': 45,
        'min_swing_pips': 20.0,  # GBP more volatile
        'confluence_mode': 'moderate'
    },
    'USDJPY': {
        'fib_lookback': 50,
        'min_swing_pips': 15.0,
        'confluence_mode': 'moderate'
    },
    # JPY pairs
    'EURJPY': {
        'fib_lookback': 50,
        'min_swing_pips': 20.0,
        'confluence_mode': 'moderate'
    },
    'AUDJPY': {
        'fib_lookback': 50,
        'min_swing_pips': 18.0,
        'confluence_mode': 'moderate'
    }
}

# Debug/Logging Settings
MACD_CONFLUENCE_DEBUG_LOGGING = True  # Enable detailed confluence analysis logs
MACD_CONFLUENCE_LOG_FIB_LEVELS = True  # Log all Fibonacci levels calculated
MACD_CONFLUENCE_LOG_PATTERNS = True  # Log candlestick pattern detection
MACD_CONFLUENCE_LOG_H4_VALIDATION = True  # Log H4 trend validation

# =============================================================================
# LEGACY MACD STRATEGY CONFIGURATION SETTINGS (Used when MACD_USE_CONFLUENCE_MODE = False)
# =============================================================================

# Core Strategy Settings
MACD_EMA_STRATEGY = False     # True = enable MACD + EMA 200 strategy
STRATEGY_WEIGHT_MACD = 0.5   # MACD strategy weight  

# MACD Periods Configuration  
MACD_PERIODS = {
    'fast_ema': 12,
    'slow_ema': 26, 
    'signal_ema': 9
}

USE_DYNAMIC_PARAMS = False
# MACD Threshold Settings
MACD_THRESHOLD_BUFFER_MULTIPLIER = 1.1  # Require MACD histogram to exceed threshold by this multiplier

# =============================================================================
# HISTOGRAM EXPANSION CONFIRMATION (Post-Crossover Validation)
# =============================================================================

# Enable wait-for-expansion logic (recommended for quality filtering)
MACD_EXPANSION_ENABLED = True          # Enable expansion confirmation
MACD_EXPANSION_WINDOW_BARS = 6         # Wait up to 3 bars for expansion (15-45 min on 15m chart)
MACD_EXPANSION_ALLOW_IMMEDIATE = True  # Trigger immediately if histogram already > threshold at crossover

# ADX Trend Validation (require ADX to be rising = strengthening trend)
MACD_REQUIRE_ADX_RISING = False        # Require ADX to be increasing (filters weakening trends)
MACD_ADX_RISING_LOOKBACK = 2           # Check ADX trend over N bars (2 = compare current vs 2 bars ago)
MACD_ADX_MIN_INCREASE = 0.5            # Minimum ADX increase required (0.5 = half a point increase)

# Expansion confirmation logging
MACD_EXPANSION_DEBUG_LOGGING = True    # Enable detailed expansion tracking logs

# Minimum Histogram Thresholds (Post-Crossover Expansion)
# After zero-line crossover, histogram must expand to this size within N bars
# This filters weak/choppy crossovers while allowing strong trends to trigger quickly
#
# Format: Can be either:
#   - Simple float: histogram threshold only (uses global MACD_MIN_ADX)
#   - Dict with 'histogram' and 'min_adx': pair-specific thresholds
MACD_MIN_HISTOGRAM_THRESHOLDS = {
    # Profit-optimized thresholds based on 79 real trades with P&L outcomes
    # Strategy: Target median histogram of WINNING trades, not just any signals
    # Key insight: Early entries with moderate expansion often outperform large expansions

    'default': {'histogram': 0.00005, 'min_adx': 15},  # Default for unlisted pairs

    # Major pairs - Based on median histogram of WINNING trades
    'GBPUSD': {'histogram': 0.000055, 'min_adx': 15},  # Median of 11 winners, 73.3% WR, +£810
    'EURUSD': {'histogram': 0.000045, 'min_adx': 15},  # Below median of 2 winners, 66.7% WR, +£79
    'AUDUSD': {'histogram': 0.000052, 'min_adx': 15},  # Median of 7 winners, 100% WR, +£521 (optimal!)
    'USDCHF': {'histogram': 0.000035, 'min_adx': 15},  # Median of 6 winners, 66.7% WR, +£275
    'USDCAD': {'histogram': 0.000040, 'min_adx': 15},  # Median of 4 winners, 80% WR, +£107
    'NZDUSD': {'histogram': 0.000050, 'min_adx': 15},  # Median of 3 winners, 42.9% WR, -£323 (needs work)

    # JPY pairs - Based on median/avg of WINNING trades
    'USDJPY': {'histogram': 0.012, 'min_adx': 15},     # Below avg of 8 winners, 57.1% WR, +¥487
    'EURJPY': {'histogram': 0.020, 'min_adx': 15},     # Below median of 3 winners, 37.5% WR, -£200
    'AUDJPY': {'histogram': 0.015, 'min_adx': 15},     # Below median of 5 winners, 45.5% WR, -£359
    'GBPJPY': {'histogram': 0.020, 'min_adx': 15},     # Conservative (insufficient trade data)
    'NZDJPY': {'histogram': 0.010, 'min_adx': 15},     # Conservative (insufficient trade data)
    'CADJPY': {'histogram': 0.012, 'min_adx': 15},     # Conservative (insufficient trade data)
    'CHFJPY': {'histogram': 0.015, 'min_adx': 15},     # Conservative (insufficient trade data)

    # Crosses (higher volatility) - Keep conservative
    'GBPAUD': {'histogram': 0.00010, 'min_adx': 15},   # High volatility cross
    'GBPNZD': {'histogram': 0.00010, 'min_adx': 15},   # High volatility cross
}

# Technical Validation Thresholds (Enhanced MACD Detection)
CLAUDE_MACD_CRITICAL_THRESHOLD = -0.0001      # Critical MACD histogram threshold for BULL signals
CLAUDE_MACD_BEAR_CRITICAL_THRESHOLD = 0.0001  # Critical MACD histogram threshold for BEAR signals

# EMA Strategy MACD Confirmation Settings
EMA_REQUIRE_MACD_CONFIRMATION = True    # Enable/disable MACD histogram check
EMA_ALLOW_WITHOUT_MACD = False          # Fallback if MACD data unavailable

# Smart Money MACD Settings
SMART_MACD_ORDER_FLOW_VALIDATION = True  # Enable order flow validation for MACD
SMART_MACD_REQUIRE_OB_CONFLUENCE = False  # Require order block confluence (strict)
SMART_MACD_FVG_PROXIMITY_PIPS = 15  # Max distance to FVG for confluence
SMART_MACD_ORDER_FLOW_BOOST = 1.2  # Confidence boost for strong order flow alignment
SMART_MACD_ORDER_FLOW_PENALTY = 0.8  # Confidence penalty for poor order flow alignment
USE_SMART_MONEY_MACD = False

# =============================================================================
# LEGACY MTF & FILTER SETTINGS (DEPRECATED - Not used by current MACD strategy)
# =============================================================================
# These settings were part of an older MTF (Multi-Timeframe) implementation.
# The current strategy (rebuilt Oct 2024) uses a simpler, more reliable approach:
#   - Histogram expansion confirmation (MACD_EXPANSION_ENABLED)
#   - ADX catch-up window (MACD_ADX_CATCHUP_ENABLED)
#   - Swing proximity validation (MACD_SWING_VALIDATION)
#
# The MTF system is disconnected:
#   - macd_strategy.py doesn't have detect_signal_with_mtf() method
#   - macd_mtf_analyzer.py exists but is never imported/used
#   - signal_detector.py checks for MTF but condition always fails
#
# If you need MTF analysis, consider implementing it in the current strategy
# rather than relying on these legacy settings.
# =============================================================================

# MACD Momentum Filter (DEPRECATED - Not used)
# ENABLE_MACD_CONTRADICTION_FILTER = True
# MACD_STRONG_THRESHOLD = 0.0001  # Threshold for "strong" MACD momentum

# MACD Enhanced Filter Configuration (DEPRECATED - Not used)
# MACD_ENHANCED_FILTERS_ENABLED = True       # Required for MTF
# MACD_DETECTION_MODE = 'permissive'         # Start permissive
# MACD_REQUIRE_EMA200_ALIGNMENT = False
MACD_DISABLE_EMA200_FILTER = True             # Keep enabled - used by macd_trend_validator.py (orphaned file)
# MACD_EMA200_FILTER_MODE = 'permissive'
MIN_BARS_FOR_MACD = 50                        # Keep enabled - used by signal_detector.py:363

# MTF (Multi-Timeframe) Settings (DEPRECATED - Not used)
# MACD_MTF_DEBUG = True
# MACD_LOG_MTF_DECISIONS = True
# MACD_MTF_ENABLED = False  # Currently disabled for debugging
# MACD_MTF_TIMEFRAMES = ['15m', '1h']
# MACD_MTF_MIN_ALIGNMENT = 0.5

# MACD_MTF_LOGGING = {
#     'enabled': True,
#     'level': 'INFO',
#     'show_decisions': True,
#     'show_calculations': True
# }

# MACD_FILTER_CONFIG = {
#     'strict_mode': False,
#     'histogram_threshold': 0.0001,
#     'require_momentum_alignment': True,
#     'enable_contradiction_filter': True
# }

# MACD_MTF_CONFIG = {
#     'timeframes': ['15m', '1h'],
#     'min_alignment_score': 0.6,
#     'macd_confidence_boost': 0.15,
#     'require_higher_tf_confirmation': True
# }

# MACD Momentum Confirmation Settings (DEPRECATED - Not used by current MACD strategy)
# These settings were part of an older implementation that was replaced in October 2024
# The current strategy uses histogram expansion + ADX catch-up window instead
# See macd_strategy.py lines 124-138 for the active confirmation system (MACD_EXPANSION_ENABLED, MACD_ADX_CATCHUP_ENABLED)
# MACD_MOMENTUM_CONFIRMATION_ENABLED = True      # Enable delayed confirmation system
# MACD_CONFIRMATION_WINDOW = 3                   # Bars to wait for momentum after crossover
# MACD_MOMENTUM_MULTIPLIER = 1.5                 # Histogram must grow 1.5x from crossover
# MACD_ALLOW_DELAYED_SIGNALS = False             # PHASE 1: Disable delayed signals for immediate entry
# MACD_CONTINUATION_ENABLED = True               # Enable strong momentum without crossover
# MACD_CONTINUATION_MULTIPLIER = 2.0             # Threshold multiplier for continuation (2x normal)
# MACD_TRACK_WEAK_CROSSOVERS = True             # Track crossovers that don't meet threshold
# MACD_CONFIRMATION_LOOKBACK = 1                  # PHASE 1: Only check latest bar (immediate signals only)

# =============================================================================
# LEGACY RSI FILTER SETTINGS (DEPRECATED - RSI logic is hardcoded)
# =============================================================================
# The current strategy has RSI filtering hardcoded with these values:
#   - Reject BULL if RSI > 70 (overbought)
#   - Reject BEAR if RSI < 30 (oversold)
#   - Confidence boost for RSI < 40 (BULL) or > 60 (BEAR)
#
# These config settings are not consulted by the code.
# To change RSI behavior, modify macd_strategy.py lines 553-558, 654-659, 717-724
# =============================================================================
# MACD_RSI_FILTER_ENABLED = True                     # Always enabled, no toggle check
# MACD_RSI_PERIOD = 14                              # RSI calculated in indicator layer
# MACD_RSI_MIDDLE = 50                              # Hardcoded as default value
# MACD_RSI_QUALITY_THRESHOLD_BULL = 55              # Not used (code uses 40/50)
# MACD_RSI_QUALITY_THRESHOLD_BEAR = 45              # Not used (code uses 50/60)
# MACD_RSI_OVERBOUGHT_THRESHOLD = 70                # Hardcoded in strategy
# MACD_RSI_OVERSOLD_THRESHOLD = 30                  # Hardcoded in strategy
# MACD_RSI_REQUIRE_RISING = True                    # Not implemented
# MACD_RSI_REQUIRE_QUALITY_THRESHOLDS = False      # Not implemented
# =============================================================================

# =============================================================================
# PHASE 1+2+3 ENHANCEMENTS (Ported from Momentum Strategy + Adaptive Volatility)
# MACD = MOMENTUM-FOCUSED STRATEGY
# =============================================================================

# PHASE 3: Adaptive Volatility-Based SL/TP (NEW)
# Runtime regime-aware calculation - No hardcoded values!
# =============================================================================
# LEGACY TREND & REGIME SETTINGS (DEPRECATED - Not used by current strategy)
# =============================================================================
# These settings were part of an older trend alignment and regime detection system
# The current strategy uses simpler, more reliable approaches
# USE_ADAPTIVE_SL_TP = False              # Used by EMA/Ranging strategies, not MACD
# MACD_REQUIRE_TREND_ALIGNMENT = False    # Old trend alignment filter (not used)
# MACD_TREND_EMA_PERIOD = 50              # Old trend EMA period (not used)
# MACD_TREND_ALIGNMENT_BOOST = 0.10       # Old confidence boost (not used)
# MACD_ENABLE_REGIME_FILTER = True        # Old regime filter system (not used)
# MACD_MIN_ATR_RATIO = 0.8                # Old ATR ratio filter (not used)
# MACD_MIN_EMA_SEPARATION = 0.2           # Old EMA separation filter (not used)
# =============================================================================

# Market Regime Filter - ADX threshold for trend strength
MACD_MIN_ADX = 21                        # Global fallback ADX threshold (pair-specific values in MACD_MIN_HISTOGRAM_THRESHOLDS)

# ADX Catch-Up Window (Similar to MACD Expansion Window)
# Allows ADX to reach threshold within N bars AFTER MACD threshold is met
# IMPORTANT: MACD must meet threshold first - ADX sliding window does NOT override this requirement
MACD_ADX_CATCHUP_ENABLED = True          # Allow ADX to reach threshold within N bars of MACD crossover
MACD_ADX_CATCHUP_WINDOW_BARS = 3         # Check up to 3 bars for ADX >= 25 (same window as MACD expansion)
MACD_ADX_ALLOW_IMMEDIATE = True          # Trigger immediately if ADX already >= 25 when MACD meets threshold

# Risk Management - ATR-based stops for momentum
MACD_STOP_LOSS_ATR_MULTIPLIER = 1.8      # PHASE 1: Tighter stops (immediate entries allow closer stops)
MACD_TAKE_PROFIT_ATR_MULTIPLIER = 4.0    # PHASE 1: Wider targets for better R:R (1.8:4.0 = 1:2.22 R:R)

# Structure-Based Stop Placement
MACD_USE_STRUCTURE_STOPS = True          # Place stops beyond recent swing points
MACD_MIN_STOP_DISTANCE_PIPS = 12.0       # PHASE 1: Reduced minimum (immediate entries don't need wide stops)
MACD_MAX_STOP_DISTANCE_PIPS = 30.0       # PHASE 1: Reduced maximum (tighter risk control)

# MACD Zero Line Filter (Mean-Reversion Focus)
MACD_ZERO_LINE_FILTER_ENABLED = False                 # Enable zero line validation filter (used in get_macd_config_summary)

# =============================================================================
# MULTI-TIMEFRAME (MTF) MACD HISTOGRAM ALIGNMENT FILTER (NEW)
# =============================================================================
# Ensures signals align with higher timeframe MACD histogram direction
# This prevents trading against the larger trend
#
# Example: 15m BULL signal requires:
#   - 1H MACD histogram > 0 (positive/bullish)
#   - 4H MACD histogram > 0 (positive/bullish)
#
# If 15m shows BULL but 1H/4H show BEAR histogram → signal REJECTED
# =============================================================================

MACD_MTF_HISTOGRAM_FILTER_ENABLED = True             # Enable MTF histogram alignment check

# Timeframes to validate (in addition to signal timeframe)
# These must be HIGHER timeframes than the signal timeframe
# Format: list of timeframe strings ('1h', '4h', '1d')
MACD_MTF_HISTOGRAM_TIMEFRAMES = ['1h', '4h']         # Check 1H and 4H alignment

# Alignment requirements
MACD_MTF_REQUIRE_ALL_ALIGNED = True                  # True = ALL timeframes must align, False = majority (2/3)
MACD_MTF_MINIMUM_HISTOGRAM_MAGNITUDE = 0.000001      # Ignore very tiny histograms (noise threshold)

# Confidence penalties for misalignment
MACD_MTF_PARTIAL_MISALIGNMENT_PENALTY = -0.10        # -10% if 1 timeframe misaligned (when require_all=False)
MACD_MTF_FULL_REJECTION = True                       # True = reject misaligned signals, False = apply penalty only

# Logging configuration
MACD_MTF_LOG_ALIGNMENT_CHECKS = True                 # Log detailed MTF alignment analysis

# Notes:
# - This filter is applied BEFORE signal validation
# - Prevents counter-trend signals (reduces false signals in ranging/choppy conditions)
# - Higher timeframe histogram direction = trend direction
# - Histogram color: Positive (green) = bullish, Negative (red) = bearish
# - Example use case: 15m BULL signal aligned with 1H + 4H bullish histograms = high confidence
# - Example rejection: 15m BULL signal but 4H histogram negative = rejected (trend conflict)

# =============================================================================
# H4 MARKET STRUCTURE ALIGNMENT (BOS/CHOCH) - DISABLED
# =============================================================================
# This feature uses SMC (Smart Money Concepts) to detect Break of Structure (BOS)
# and Change of Character (CHOCH) on H4 timeframe to filter signals.
#
# DISABLED: Structure detection was causing issues and reducing signal quality
# =============================================================================

MACD_H4_STRUCTURE_ALIGNMENT_ENABLED = False  # Disable H4 structure (BOS/CHOCH) validation
MACD_H4_REQUIRE_STRUCTURE_ALIGNMENT = False  # Don't block signals based on structure

# =============================================================================
# RSI Configuration
# =============================================================================
# RSI is used for confidence adjustment, not signal rejection
#
# Instead of rejecting signals in extreme zones, we adjust confidence:
# - BULL signals: Higher confidence when RSI < 50 (not overbought)
# - BEAR signals: Higher confidence when RSI > 50 (not oversold)
#
# Confidence boosts (applied in calculate_confidence method):
# - BULL with RSI < 40: +10% confidence
# - BULL with RSI < 50: +5% confidence
# - BEAR with RSI > 60: +10% confidence
# - BEAR with RSI > 50: +5% confidence
# =============================================================================

MACD_RSI_FILTER_ENABLED = False  # Disable rejection - use RSI for confidence only
MACD_RSI_OVERBOUGHT_THRESHOLD = 999  # Effectively disabled (set very high)
MACD_RSI_OVERSOLD_THRESHOLD = 0      # Effectively disabled (set very low)

# Strategy Integration Settings
MACD_STRATEGY_WEIGHT = 0.15         # Weight in combined strategy mode
MACD_ALLOW_COMBINED = True          # Allow in combined strategies
MACD_PRIORITY_LEVEL = 2             # Priority level (1=highest, 5=lowest)

# Performance Settings
MACD_ENABLE_BACKTESTING = True      # Enable strategy in backtests
MACD_MIN_DATA_PERIODS = 60         # Minimum data periods required (26 + 9 + buffer)
MACD_ENABLE_PERFORMANCE_TRACKING = True  # Track strategy performance

# Debug Settings
MACD_DEBUG_LOGGING = True           # Enable detailed debug logging

# Helper function to get MACD threshold for specific epic (moved from main config)
def get_macd_threshold_for_epic(epic: str) -> float:
    """Get MACD threshold for enhanced Claude validation"""
    # Check if this is a JPY pair
    if 'JPY' in epic.upper():
        return 0.003  # JPY pairs - calibrated from actual market data
    else:
        return 0.00003  # Non-JPY pairs - calibrated from actual market data

def get_macd_min_adx(epic: str) -> float:
    """
    Get minimum ADX value for a specific currency pair.

    Reads from MACD_MIN_HISTOGRAM_THRESHOLDS which contains pair-specific settings.

    Args:
        epic: Currency pair epic code (e.g., 'EURUSD', 'GBPJPY', 'CS.D.EURUSD.CEEM.IP')

    Returns:
        Minimum ADX threshold for the pair (defaults to MACD_MIN_ADX if not specified)

    Usage:
        min_adx = get_macd_min_adx('EURUSD')  # Returns 20
        min_adx = get_macd_min_adx('NZDJPY')  # Returns 21
        min_adx = get_macd_min_adx('CS.D.GBPUSD.MINI.IP')  # Returns 22
    """
    if not epic:
        # Get default from MACD_MIN_HISTOGRAM_THRESHOLDS['default'] or fallback to MACD_MIN_ADX
        default_config = MACD_MIN_HISTOGRAM_THRESHOLDS.get('default', {})
        if isinstance(default_config, dict):
            return default_config.get('min_adx', MACD_MIN_ADX)
        return MACD_MIN_ADX

    epic_upper = epic.upper()

    # Try to find pair in MACD_MIN_HISTOGRAM_THRESHOLDS
    # Check exact match first
    for pair_name, config in MACD_MIN_HISTOGRAM_THRESHOLDS.items():
        if pair_name == 'default':
            continue
        if pair_name in epic_upper:
            # Handle both old format (float) and new format (dict)
            if isinstance(config, dict):
                return config.get('min_adx', MACD_MIN_ADX)
            # Old format - use global default
            return MACD_MIN_ADX

    # No pair-specific config found, use default
    default_config = MACD_MIN_HISTOGRAM_THRESHOLDS.get('default', {})
    if isinstance(default_config, dict):
        return default_config.get('min_adx', MACD_MIN_ADX)
    return MACD_MIN_ADX

# Configuration summary function
def get_macd_config_summary() -> dict:
    """Get a summary of MACD configuration settings"""
    return {
        'strategy_enabled': MACD_EMA_STRATEGY,
        'smart_money_enabled': USE_SMART_MONEY_MACD,
        'zero_line_filter_enabled': MACD_ZERO_LINE_FILTER_ENABLED,
        'macd_periods': MACD_PERIODS,
        'min_data_periods': MACD_MIN_DATA_PERIODS,
        'debug_logging': MACD_DEBUG_LOGGING,
        'expansion_enabled': MACD_EXPANSION_ENABLED,
        'adx_catchup_enabled': MACD_ADX_CATCHUP_ENABLED,
        'swing_validation_enabled': MACD_SWING_VALIDATION.get('enabled', False),
        'mtf_histogram_filter_enabled': MACD_MTF_HISTOGRAM_FILTER_ENABLED,
        'mtf_timeframes': MACD_MTF_HISTOGRAM_TIMEFRAMES,
        'mtf_require_all_aligned': MACD_MTF_REQUIRE_ALL_ALIGNED
    }


# =============================================================================
# SWING PROXIMITY VALIDATION CONFIGURATION (NEW)
# =============================================================================

# Swing Proximity Validation - Prevents poor entry timing near swing points
MACD_SWING_VALIDATION = {
    'enabled': True,  # Enable swing proximity validation
    'min_distance_pips': 8,  # Minimum distance from swing high/low (8 pips = 80 IG points)
    'lookback_swings': 5,  # Number of recent swings to check
    'swing_length': 5,  # Bars for swing detection (matches SMC default)
    'strict_mode': False,  # False = reduce confidence, True = reject signal entirely
    'resistance_buffer': 1.0,  # No multiplier - use 8 pips as-is
    'support_buffer': 1.0,  # No multiplier - use 8 pips as-is
}

# =============================================================================
# EMA TREND FILTER CONFIGURATION
# =============================================================================

# EMA Trend Filter - Filters signals based on EMA trend alignment
# When enabled, BULL signals require price >= EMA, BEAR signals require price <= EMA
MACD_EMA_FILTER = {
    'enabled': False,  # Enable/disable EMA trend filter (default: disabled for more signals)
    'ema_period': 50,  # EMA period to use (50, 100, or 200)
    # Options: 50 = faster/more signals, 100 = balanced, 200 = slower/fewer signals
}

# Notes:
# - Prevents BUY signals when price is too close to recent swing highs (resistance)
# - Prevents SELL signals when price is too close to recent swing lows (support)
# - Works in conjunction with existing S/R validation (different timeframes)
# - Swing points: Recent pivots (5-50 bars) on 5m/15m charts
# - S/R levels: Long-term zones (100-500 bars)
# - 8 pips = 80 IG points (practical for intraday swings on 5m/15m)
# - For JPY pairs: 8 pips = 0.08 price movement (80 IG points)
# - Example: EUR/USD at 1.0850, swing high at 1.0857 → 7 pips away → WARNING
# - Example: USD/JPY at 150.50, swing high at 150.57 → 7 pips away → WARNING

# =============================================================================
# ADX CROSSOVER TRIGGER CONFIGURATION (NEW FEATURE)
# =============================================================================
# Additional signal trigger: ADX crosses above threshold while MACD histogram is aligned
# This catches trend acceleration earlier than MACD histogram crossover

MACD_ADX_CROSSOVER_ENABLED = True          # Enable ADX crossover trigger (in addition to MACD crossover)
MACD_ADX_CROSSOVER_THRESHOLD = 21          # ADX level that triggers signal (25 = strong trend, 18 = weak trend)
MACD_ADX_CROSSOVER_LOOKBACK = 3            # Bars to confirm ADX has been rising (prevents whipsaws)
MACD_ADX_MIN_HISTOGRAM = 0.0002            # Minimum MACD histogram magnitude (increased from 0.0001 to reduce noise)
MACD_ADX_REQUIRE_EXPANSION = True          # Require MACD histogram to be expanding (not shrinking)
MACD_ADX_MIN_CONFIDENCE = 0.60             # Minimum confidence for ADX crossover signals (matches MACD crossover 0.60)

# ADX Crossover Signal Priority
MACD_ADX_SIGNAL_PRIORITY = 2               # Priority: 1=MACD histogram crossover (stronger), 2=ADX crossover (earlier entry)

# Notes:
# - ADX crossover signals are "early entry" signals (lower confidence, better R:R)
# - MACD histogram crossover signals remain the primary trigger (higher confidence)
# - Lookback period prevents false breakouts (ADX crosses 25 but falls back immediately)
# - Histogram expansion check ensures momentum is building, not fading
# - Useful for catching trend acceleration from weak (ADX 20-24) to strong (ADX 25+)

# =============================================================================
# KAMA EFFICIENCY VALIDATION (NEW - Non-Blocking Quality Filter)
# =============================================================================

# Enable KAMA efficiency-based confidence adjustments for MACD
# KAMA efficiency measures market directional clarity (0-1 scale)
# Higher efficiency = clearer trend, lower efficiency = choppy/ranging
MACD_USE_KAMA_EFFICIENCY = True

# KAMA Efficiency Thresholds (non-blocking, for confidence adjustments only)
# Based on analysis of real trades:
#   Alert 5561 (7.0/10 quality): 0.593 efficiency (EXCELLENT)
#   Alert 5576 (6.5/10 quality): 0.302 efficiency (GOOD)
#   Alert 5575 (4.5/10 quality): 0.036 efficiency (VERY POOR)
MACD_KAMA_EFFICIENCY_THRESHOLDS = {
    'excellent': 0.50,    # >= 0.50: Strong directional movement with minimal noise
    'good': 0.30,         # >= 0.30: Moderate trend clarity
    'acceptable': 0.15,   # >= 0.15: Minimal acceptable efficiency
    'poor': 0.10,         # >= 0.10: Choppy market conditions
    'very_poor': 0.05     # < 0.05: Extremely noisy, whipsaw conditions
}

# KAMA Efficiency Confidence Modifiers (added to base signal confidence)
# Conservative adjustments to avoid over-filtering while rewarding quality
MACD_KAMA_CONFIDENCE_ADJUSTMENTS = {
    'excellent_boost': 0.10,      # +10% confidence for efficiency >= 0.50
    'good_boost': 0.05,           # +5% confidence for efficiency >= 0.30
    'acceptable_neutral': 0.0,    # No change for efficiency 0.15-0.30
    'poor_penalty': -0.05,        # -5% confidence for efficiency 0.10-0.15
    'very_poor_penalty': -0.10    # -10% confidence for efficiency < 0.10
}

# KAMA Trend Alignment Check (optional, disabled by default)
# When enabled, checks if KAMA trend direction matches signal direction
# Example: BULL signal should have KAMA trend > 0 (bullish)
MACD_REQUIRE_KAMA_TREND_ALIGNMENT = False  # Start disabled (too strict)
MACD_KAMA_TREND_CONFLICT_PENALTY = -0.08  # Additional -8% penalty if trends conflict

# Logging Configuration
MACD_LOG_KAMA_EFFICIENCY = True  # Log KAMA efficiency analysis for transparency

# Performance Notes:
# - Non-blocking: Never rejects signals, only adjusts confidence
# - Gradual: Smooth confidence curve based on market clarity
# - Validated: Based on 16x quality difference in real trade analysis
# - Safe: Conservative ±5-10% adjustments prevent over-fitting
# - Transparent: All adjustments logged for analysis and tuning

# =============================================================================
# RANGING MARKET PENALTY (NEW - Additional Quality Filter for MACD)
# =============================================================================

# MACD performs poorly in ranging markets (generates false signals)
# This adds an additional confidence penalty when ranging score is high
MACD_RANGING_PENALTY_ENABLED = True

# Ranging score thresholds for penalties
MACD_RANGING_THRESHOLDS = {
    'high_ranging': 0.55,      # >= 55% ranging (e.g., Alert 5575: 60.7%)
    'moderate_ranging': 0.45,  # >= 45% ranging
}

# Confidence penalties for ranging markets
MACD_RANGING_PENALTIES = {
    'high_ranging_penalty': -0.15,      # -15% for ranging >= 55%
    'moderate_ranging_penalty': -0.08,  # -8% for ranging >= 45%
}

# Logging
MACD_LOG_RANGING_ANALYSIS = True

# Combined Effect Example (Alert 5575):
# Base confidence: 70%
# KAMA penalty (0.036 efficiency): -10%
# Ranging penalty (60.7% ranging): -15%
# Final confidence: 45% → Would be BLOCKED by 60% minimum threshold

# =============================================================================
# MARKET BIAS CONFLICT DETECTION (Directional Alignment with Broader Market)
# =============================================================================
# Penalizes signals that contradict the overall market direction/bias
# Example: BULL signal when market intelligence shows bearish bias
#
# Use Case (Alert 5593):
# - USDJPY BULL signal @ 152.26750
# - Market intelligence: bearish bias, high_volatility regime
# - Result: Signal conflicts with bearish market → Apply penalty
#
# Impact: Reduces confidence for counter-trend signals, improving win rate

# Enable/disable market bias conflict detection
MACD_ENABLE_MARKET_BIAS_FILTER = True

# Confidence penalty for normal bias conflicts
# Example: BULL signal in bearish market (or vice versa)
MACD_MARKET_BIAS_CONFLICT_PENALTY = -0.10  # -10% confidence penalty

# Strong consensus threshold (directional_consensus score)
# If market has strong directional consensus (>80%), apply heavier penalty
MACD_STRONG_CONSENSUS_THRESHOLD = 0.8  # 80% consensus

# Confidence penalty for strong consensus conflicts
# Example: BULL signal in strongly bearish market (consensus >80%)
MACD_STRONG_CONSENSUS_PENALTY = -0.15  # -15% confidence penalty

# Combined Effect Example (Alert 5593 - USDJPY BULL):
# Base confidence: 62%
# Market bias conflict (bearish market): -10%
# Swing proximity (1.45 pips from resistance): -15% (from swing validator)
# Final confidence: 37% → Would be REJECTED (below 60% minimum)

# =============================================================================
# PRICE EXTREME FILTER - Prevent Buying Tops / Selling Bottoms
# =============================================================================
# Rejects signals that trigger at extreme price levels (local tops/bottoms)
# Uses percentile ranking to detect if price is at an extreme relative to recent history
#
# How it works:
# - BULL signals: Reject if current price is in the top X% of recent prices (buying at tops)
# - BEAR signals: Reject if current price is in the bottom X% of recent prices (selling at bottoms)
#
# Example: If price_extreme_threshold = 90 and lookback = 200 bars
#   - BULL signal at price higher than 90% of last 200 bars → REJECTED (buying at top)
#   - BEAR signal at price lower than 10% of last 200 bars → REJECTED (selling at bottom)
#
# This prevents late entries after moves are exhausted

# Enable/disable extreme price filter
MACD_PRICE_EXTREME_FILTER_ENABLED = False  # DISABLED - not effective

# =============================================================================
# PRICE STRUCTURE VALIDATION - Hybrid Approach
# =============================================================================
# Validates that price structure confirms the MACD signal direction
# MACD identifies potential setups, structure confirms entry validity
#
# For BULL signals:
#   - Requires higher lows (uptrend structure)
#   - Most recent swing low > previous swing low
#   - Confirms we're in a pullback, not a reversal
#
# For BEAR signals:
#   - Requires lower highs (downtrend structure)
#   - Most recent swing high < previous swing high
#   - Confirms we're in a pullback, not a reversal
#
# This ensures we trade WITH structure, not against it

# Enable/disable structure validation
MACD_PRICE_STRUCTURE_VALIDATION_ENABLED = True

# Lookback period for structure analysis (bars)
MACD_STRUCTURE_LOOKBACK = 30  # Last 30 bars (~1.25 days on 1H)

# Swing detection parameters
MACD_STRUCTURE_SWING_STRENGTH = 3  # Bars on each side to confirm swing

# Require minimum number of swings for validation
MACD_STRUCTURE_MIN_SWINGS = 2  # Need at least 2 swings to compare

# =============================================================================
# 34 EMA TREND FILTER (1H Timeframe) - DISABLED
# =============================================================================
# Requires price to be on correct side of 34 EMA for signal direction
# - BULL signals: Price must be ABOVE 34 EMA
# - BEAR signals: Price must be BELOW 34 EMA
#
# DISABLED: EMA filter was causing late entries - by the time price crosses back
# above/below EMA after a MACD crossover, the move is often exhausted

# Enable/disable EMA filter
MACD_EMA_FILTER_ENABLED = False  # DISABLED - causes late entries

# EMA period
MACD_EMA_FILTER_PERIOD = 34

# Require alignment (if False, only logs warning but doesn't block)
MACD_EMA_REQUIRE_ALIGNMENT = False  # DISABLED