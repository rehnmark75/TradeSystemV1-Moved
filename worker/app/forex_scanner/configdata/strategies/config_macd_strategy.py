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

# MACD Threshold Settings
MACD_THRESHOLD_BUFFER_MULTIPLIER = 1.1  # Require MACD histogram to exceed threshold by this multiplier

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
MACD_ALLOW_DELAYED_SIGNALS = True              # Allow signals after crossover
MACD_CONTINUATION_ENABLED = True               # Enable strong momentum without crossover
MACD_CONTINUATION_MULTIPLIER = 2.0             # Threshold multiplier for continuation (2x normal)
MACD_TRACK_WEAK_CROSSOVERS = True             # Track crossovers that don't meet threshold
MACD_CONFIRMATION_LOOKBACK = 5                 # Bars to prevent duplicate confirmations

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
ACTIVE_MACD_PRESET = 'balanced'  # Default to optimized balanced preset

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