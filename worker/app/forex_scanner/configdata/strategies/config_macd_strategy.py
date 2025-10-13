# configdata/strategies/config_macd_strategy.py
"""
MACD Strategy Configuration
Configuration module for MACD strategy settings
"""

# =============================================================================
# MACD STRATEGY CONFIGURATION SETTINGS  
# =============================================================================

# Core Strategy Settings
MACD_EMA_STRATEGY = True     # True = enable MACD + EMA 200 strategy
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
MACD_EXPANSION_WINDOW_BARS = 3         # Wait up to 3 bars for expansion (15-45 min on 15m chart)
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

    'default': {'histogram': 0.00005, 'min_adx': 18},  # Default for unlisted pairs

    # Major pairs - Based on median histogram of WINNING trades
    'GBPUSD': {'histogram': 0.000050, 'min_adx': 18},  # Median of 11 winners, 73.3% WR, +£810
    'EURUSD': {'histogram': 0.000055, 'min_adx': 18},  # Below median of 2 winners, 66.7% WR, +£79
    'AUDUSD': {'histogram': 0.000052, 'min_adx': 18},  # Median of 7 winners, 100% WR, +£521 (optimal!)
    'USDCHF': {'histogram': 0.000035, 'min_adx': 18},  # Median of 6 winners, 66.7% WR, +£275
    'USDCAD': {'histogram': 0.000040, 'min_adx': 18},  # Median of 4 winners, 80% WR, +£107
    'NZDUSD': {'histogram': 0.000050, 'min_adx': 18},  # Median of 3 winners, 42.9% WR, -£323 (needs work)

    # JPY pairs - Based on median/avg of WINNING trades
    'USDJPY': {'histogram': 0.012, 'min_adx': 18},     # Below avg of 8 winners, 57.1% WR, +¥487
    'EURJPY': {'histogram': 0.020, 'min_adx': 18},     # Below median of 3 winners, 37.5% WR, -£200
    'AUDJPY': {'histogram': 0.015, 'min_adx': 18},     # Below median of 5 winners, 45.5% WR, -£359
    'GBPJPY': {'histogram': 0.020, 'min_adx': 18},     # Conservative (insufficient trade data)
    'NZDJPY': {'histogram': 0.010, 'min_adx': 18},     # Conservative (insufficient trade data)
    'CADJPY': {'histogram': 0.012, 'min_adx': 18},     # Conservative (insufficient trade data)
    'CHFJPY': {'histogram': 0.015, 'min_adx': 18},     # Conservative (insufficient trade data)

    # Crosses (higher volatility) - Keep conservative
    'GBPAUD': {'histogram': 0.00010, 'min_adx': 20},   # High volatility cross
    'GBPNZD': {'histogram': 0.00010, 'min_adx': 20},   # High volatility cross
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

# MACD Momentum Filter (Prevents momentum contradictions)
ENABLE_MACD_CONTRADICTION_FILTER = True  
MACD_STRONG_THRESHOLD = 0.0001  # Threshold for "strong" MACD momentum

# MACD Enhanced Filter Configuration
MACD_ENHANCED_FILTERS_ENABLED = True       # Required for MTF
MACD_DETECTION_MODE = 'permissive'         # Start permissive
MACD_REQUIRE_EMA200_ALIGNMENT = False
MACD_DISABLE_EMA200_FILTER = True
MACD_EMA200_FILTER_MODE = 'permissive' 
MIN_BARS_FOR_MACD = 50

# MTF (Multi-Timeframe) Settings
MACD_MTF_DEBUG = True
MACD_LOG_MTF_DECISIONS = True
MACD_MTF_ENABLED = False  # Currently disabled for debugging
MACD_MTF_TIMEFRAMES = ['15m', '1h']
MACD_MTF_MIN_ALIGNMENT = 0.5

MACD_MTF_LOGGING = {
    'enabled': True,
    'level': 'INFO',
    'show_decisions': True,
    'show_calculations': True
}

MACD_FILTER_CONFIG = {
    'strict_mode': False,
    'histogram_threshold': 0.0001,
    'require_momentum_alignment': True,
    'enable_contradiction_filter': True
}

MACD_MTF_CONFIG = {
    'timeframes': ['15m', '1h'],
    'min_alignment_score': 0.6,
    'macd_confidence_boost': 0.15,
    'require_higher_tf_confirmation': True
}

# MACD Momentum Confirmation Settings (Professional Trading Approach)
MACD_MOMENTUM_CONFIRMATION_ENABLED = True      # Enable delayed confirmation system
MACD_CONFIRMATION_WINDOW = 3                   # Bars to wait for momentum after crossover
MACD_MOMENTUM_MULTIPLIER = 1.5                 # Histogram must grow 1.5x from crossover
MACD_ALLOW_DELAYED_SIGNALS = False             # PHASE 1: Disable delayed signals for immediate entry
MACD_CONTINUATION_ENABLED = True               # Enable strong momentum without crossover
MACD_CONTINUATION_MULTIPLIER = 2.0             # Threshold multiplier for continuation (2x normal)
MACD_TRACK_WEAK_CROSSOVERS = True             # Track crossovers that don't meet threshold
MACD_CONFIRMATION_LOOKBACK = 1                  # PHASE 1: Only check latest bar (immediate signals only)

# RSI Momentum Confirmation for MACD
MACD_RSI_FILTER_ENABLED = True                     # Enable RSI momentum confirmation for MACD signals
MACD_RSI_PERIOD = 14                              # RSI calculation period
MACD_RSI_MIDDLE = 50                              # RSI neutral level (above = bullish, below = bearish)
MACD_RSI_QUALITY_THRESHOLD_BULL = 55              # Require RSI > 55 for high-quality long signals (optional)
MACD_RSI_QUALITY_THRESHOLD_BEAR = 45              # Require RSI < 45 for high-quality short signals (optional)
MACD_RSI_OVERBOUGHT_THRESHOLD = 70                # Skip long signals if RSI > 70 (overbought)
MACD_RSI_OVERSOLD_THRESHOLD = 30                  # Skip short signals if RSI < 30 (oversold)
MACD_RSI_REQUIRE_RISING = True                    # Require RSI rising for long, falling for short
MACD_RSI_REQUIRE_QUALITY_THRESHOLDS = False      # Enable optional quality thresholds (55/45)

# =============================================================================
# PHASE 1+2+3 ENHANCEMENTS (Ported from Momentum Strategy + Adaptive Volatility)
# MACD = MOMENTUM-FOCUSED STRATEGY
# =============================================================================

# PHASE 3: Adaptive Volatility-Based SL/TP (NEW)
# Runtime regime-aware calculation - No hardcoded values!
USE_ADAPTIVE_SL_TP = False               # 🧠 Enable adaptive volatility calculator (default: False for gradual rollout)
                                         # When True: Uses runtime regime detection (trending, ranging, breakout, high volatility)
                                         # When False: Falls back to ATR multipliers below

# Trend Alignment Filter - MACD as momentum strategy can work counter-trend with confirmation
MACD_REQUIRE_TREND_ALIGNMENT = False      # MACD can trade counter-trend (momentum reversals)
MACD_TREND_EMA_PERIOD = 50               # EMA period for trend context (not mandatory)
MACD_TREND_ALIGNMENT_BOOST = 0.10        # Confidence boost if aligned with trend

# Market Regime Filter - Critical for momentum strategies
MACD_ENABLE_REGIME_FILTER = True         # Filter out unfavorable market conditions
MACD_MIN_ADX = 21                        # PHASE 2: Increase to 25 (stronger trend requirement)
MACD_MIN_ATR_RATIO = 0.8                 # PHASE 2: Increase to 0.8 (require higher volatility)
MACD_MIN_EMA_SEPARATION = 0.2            # Price distance from EMA (in ATR units)

# ADX Catch-Up Window (Similar to MACD Expansion Window)
# Allows ADX to reach threshold within N bars AFTER MACD threshold is met
# IMPORTANT: MACD must meet threshold first - ADX sliding window does NOT override this requirement
MACD_ADX_CATCHUP_ENABLED = True          # Allow ADX to reach threshold within N bars of MACD crossover
MACD_ADX_CATCHUP_WINDOW_BARS = 3         # Check up to 3 bars for ADX >= 25 (same window as MACD expansion)
MACD_ADX_ALLOW_IMMEDIATE = True          # Trigger immediately if ADX already >= 25 when MACD meets threshold

# Confirmation Requirements - Strengthen signal quality
MACD_MIN_CONFIRMATIONS = 2               # Require 2 confirmations (MACD + RSI minimum)
MACD_CONFIRMATION_TYPES = ['macd_histogram', 'rsi_momentum', 'volume', 'price_action']

# Risk Management - ATR-based stops for momentum
MACD_STOP_LOSS_ATR_MULTIPLIER = 1.8      # PHASE 1: Tighter stops (immediate entries allow closer stops)
MACD_TAKE_PROFIT_ATR_MULTIPLIER = 4.0    # PHASE 1: Wider targets for better R:R (1.8:4.0 = 1:2.22 R:R)

# Structure-Based Stop Placement
MACD_USE_STRUCTURE_STOPS = True          # Place stops beyond recent swing points
MACD_STRUCTURE_LOOKBACK_BARS = 20        # Look back 20 bars for swing highs/lows
MACD_MIN_STOP_DISTANCE_PIPS = 12.0       # PHASE 1: Reduced minimum (immediate entries don't need wide stops)
MACD_MAX_STOP_DISTANCE_PIPS = 30.0       # PHASE 1: Reduced maximum (tighter risk control)
MACD_STRUCTURE_BUFFER_PIPS = 2.0         # Buffer beyond swing point

# Enhanced Confidence Calculation Factors
MACD_CONFIDENCE_BASE = 0.50              # Start conservative
MACD_CONFIDENCE_MACD_STRENGTH = 0.25     # Factor for MACD histogram strength
MACD_CONFIDENCE_TREND_ALIGNMENT = 0.10   # Bonus if trend-aligned (optional)
MACD_CONFIDENCE_REGIME_FAVORABLE = 0.10  # Bonus for favorable regime
MACD_CONFIDENCE_RSI_MOMENTUM = 0.15      # Factor for RSI momentum
MACD_CONFIDENCE_VOLUME = 0.10            # Factor for volume confirmation
MACD_RSI_SKIP_EXTREME_ZONES = True               # Skip signals in overbought/oversold zones

# MACD Zero Line Filter (Mean-Reversion Focus)
# Require MACD and Signal lines to be on specific side of zero line for quality entries
MACD_ZERO_LINE_FILTER_ENABLED = False                 # Enable zero line validation filter
MACD_ZERO_LINE_REQUIRE_BOTH_LINES = True           # False = either line can meet criteria, True = both lines required
MACD_ZERO_LINE_BULL_BELOW_ZERO = True               # Bull signals: crossover below zero line
MACD_ZERO_LINE_BEAR_ABOVE_ZERO = True               # Bear signals: crossover above zero line
MACD_ZERO_LINE_STRICT_MODE = True                  # False = allow if either line meets criteria, True = require both

# OPTIMIZED FILTER PRESETS - Post-optimization balanced configurations
MACD_FILTER_PRESETS = {
    'conservative': {
        'name': 'Conservative - High Quality',
        'description': 'Higher quality signals, fewer in quantity',
        'histogram_threshold_jpy': 0.00005,  # Higher thresholds
        'histogram_threshold_major': 0.000025,
        'min_confidence': 0.50,  # Higher confidence requirement
        'max_daily_signals': 4,
        'min_spacing_minutes': 60,  # 1 hour spacing
        'adx_weight': 0.15,  # Higher ADX influence
        'enable_adaptive_scoring': True,
        'enable_volatility_filter': True,
        'enable_divergence_detection': True,
        'enable_vwap': True
    },
    'balanced': {
        'name': 'Balanced - Medium Quality & Quantity',
        'description': 'Balanced between signal quality and quantity (CURRENT OPTIMIZED)',
        'histogram_threshold_jpy': 0.00003,  # Current optimized values
        'histogram_threshold_major': 0.00001,
        'min_confidence': 0.35,  # Current optimized confidence
        'max_daily_signals': 8,
        'min_spacing_minutes': 30,  # 30 minute spacing
        'adx_weight': 0.10,  # Current optimized ADX weight
        'enable_adaptive_scoring': False,  # Currently disabled
        'enable_volatility_filter': False,  # Currently disabled
        'enable_divergence_detection': False,  # Currently disabled
        'enable_vwap': False  # Currently disabled
    },
    'aggressive': {
        'name': 'Aggressive - Maximum Signals',
        'description': 'Maximum signal generation, lower quality thresholds',
        'histogram_threshold_jpy': 0.00001,  # Very low thresholds
        'histogram_threshold_major': 0.000005,
        'min_confidence': 0.25,  # Lower confidence requirement
        'max_daily_signals': 15,
        'min_spacing_minutes': 15,  # 15 minute spacing
        'adx_weight': 0.05,  # Minimal ADX influence
        'enable_adaptive_scoring': False,
        'enable_volatility_filter': False,
        'enable_divergence_detection': False,
        'enable_vwap': False
    },
    'maximum': {
        'name': 'Maximum - All Signals',
        'description': 'Catch every possible signal, minimal filtering',
        'histogram_threshold_jpy': 0.000005,  # Extremely low thresholds
        'histogram_threshold_major': 0.000001,
        'min_confidence': 0.20,  # Very low confidence requirement
        'max_daily_signals': 50,  # No practical limit
        'min_spacing_minutes': 0,  # No spacing requirement
        'adx_weight': 0.02,  # Minimal ADX influence
        'enable_adaptive_scoring': False,
        'enable_volatility_filter': False,
        'enable_divergence_detection': False,
        'enable_vwap': False
    }
}

# Current active preset (can be changed by user)
ACTIVE_MACD_PRESET = 'conservative'  # PHASE 2: Switch to high-quality signals

# Legacy Safety Preset Configurations (kept for compatibility)
MACD_SAFETY_PRESETS = {
    'conservative': {
        'ENABLE_MACD_CONTRADICTION_FILTER': True,
        'MACD_STRONG_THRESHOLD': 0.00005,  # Very sensitive
        'CLAUDE_MACD_CRITICAL_THRESHOLD': -0.00005,  # Very strict
    },
    'balanced': {
        'ENABLE_MACD_CONTRADICTION_FILTER': True,
        'MACD_STRONG_THRESHOLD': 0.0001,   # Standard sensitivity
        'CLAUDE_MACD_CRITICAL_THRESHOLD': -0.0001,   # Standard
    },
    'aggressive': {
        'ENABLE_MACD_CONTRADICTION_FILTER': False,  # Disabled
        'MACD_STRONG_THRESHOLD': 0.0002,   # Less sensitive
        'CLAUDE_MACD_CRITICAL_THRESHOLD': -0.0002,   # More lenient
    },
    'very_aggressive': {
        'ENABLE_MACD_CONTRADICTION_FILTER': False,
        'MACD_STRONG_THRESHOLD': 0.0005,   # Much less sensitive
        'CLAUDE_MACD_CRITICAL_THRESHOLD': -0.0005,   # Very lenient
    }
}

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
        epic: Currency pair epic code (e.g., 'EURUSD', 'GBPJPY', 'CS.D.EURUSD.MINI.IP')

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
def get_active_macd_preset() -> dict:
    """Get the currently active MACD preset configuration"""
    try:
        return MACD_FILTER_PRESETS.get(ACTIVE_MACD_PRESET, MACD_FILTER_PRESETS['balanced'])
    except Exception:
        return MACD_FILTER_PRESETS['balanced']  # Fallback to balanced

def set_macd_preset(preset_name: str) -> bool:
    """Set the active MACD preset"""
    global ACTIVE_MACD_PRESET
    if preset_name in MACD_FILTER_PRESETS:
        ACTIVE_MACD_PRESET = preset_name
        return True
    return False

def get_available_macd_presets() -> dict:
    """Get all available MACD presets with descriptions"""
    return {
        name: {
            'name': config['name'],
            'description': config['description']
        }
        for name, config in MACD_FILTER_PRESETS.items()
    }

def get_macd_config_summary() -> dict:
    """Get a summary of MACD configuration settings"""
    preset = get_active_macd_preset()
    return {
        'strategy_enabled': MACD_EMA_STRATEGY,
        'active_preset': ACTIVE_MACD_PRESET,
        'preset_name': preset.get('name', 'Unknown'),
        'preset_description': preset.get('description', 'No description'),
        'mtf_enabled': MACD_MTF_ENABLED,
        'smart_money_enabled': USE_SMART_MONEY_MACD,
        'rsi_filter_enabled': MACD_RSI_FILTER_ENABLED,
        'momentum_confirmation_enabled': MACD_MOMENTUM_CONFIRMATION_ENABLED,
        'zero_line_filter_enabled': MACD_ZERO_LINE_FILTER_ENABLED,
        'contradiction_filter_enabled': ENABLE_MACD_CONTRADICTION_FILTER,
        'macd_periods': MACD_PERIODS,
        'min_data_periods': MACD_MIN_DATA_PERIODS,
        'debug_logging': MACD_DEBUG_LOGGING,
        'optimization_settings': {
            'histogram_threshold_jpy': preset.get('histogram_threshold_jpy'),
            'histogram_threshold_major': preset.get('histogram_threshold_major'),
            'min_confidence': preset.get('min_confidence'),
            'max_daily_signals': preset.get('max_daily_signals'),
            'adx_weight': preset.get('adx_weight')
        }
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
MACD_ADX_CROSSOVER_THRESHOLD = 20          # ADX level that triggers signal (default: 25 = strong trend)
MACD_ADX_CROSSOVER_LOOKBACK = 3            # Bars to confirm ADX has been rising (prevents whipsaws)
MACD_ADX_MIN_HISTOGRAM = 0.0001            # Minimum MACD histogram magnitude (prevents tiny movements)
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