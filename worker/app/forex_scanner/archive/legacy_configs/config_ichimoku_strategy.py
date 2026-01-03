# configdata/strategies/config_ichimoku_strategy.py
"""
Ichimoku Cloud Strategy Configuration
Configuration module for Ichimoku Kinko Hyo (One Glance Equilibrium Chart) strategy settings
"""

# =============================================================================
# ICHIMOKU STRATEGY CORE CONFIGURATION SETTINGS
# =============================================================================

# Core Strategy Settings
ICHIMOKU_CLOUD_STRATEGY = False         # True = enable Ichimoku Cloud strategy
STRATEGY_WEIGHT_ICHIMOKU = 0.6          # Ichimoku strategy weight (slightly higher due to comprehensive nature)

# Epic Filtering Configuration - ICHIMOKU STRATEGY ONLY
# These exclusions ONLY apply to the Ichimoku strategy
# Other strategies (EMA, MACD, SMC, etc.) can still trade these epics
ICHIMOKU_EXCLUDED_EPICS = [
    'CS.D.GBPUSD.MINI.IP'    # Exclude GBPUSD due to extreme volatility causing excessive signals
]

ICHIMOKU_EPIC_EXCLUSION_REASON = {
    'CS.D.GBPUSD.MINI.IP': 'Extreme volatility generates excessive signals (200+ per 14 days) even with 75% confidence threshold'
}

# =============================================================================
# ICHIMOKU PERIODS CONFIGURATION (Traditional: 9-26-52-26)
# =============================================================================

# Traditional Ichimoku Settings
ICHIMOKU_PERIODS = {
    'tenkan_period': 9,      # Conversion Line (Tenkan-sen) - high/low midpoint over 9 periods
    'kijun_period': 26,      # Base Line (Kijun-sen) - high/low midpoint over 26 periods
    'senkou_b_period': 52,   # Leading Span B - high/low midpoint over 52 periods
    'chikou_shift': 26,      # Lagging Span displacement (backward shift)
    'cloud_shift': 26        # Cloud forward displacement (future projection)
}

# =============================================================================
# DYNAMIC ICHIMOKU CONFIGURATION SYSTEM
# =============================================================================

# Enable/disable dynamic Ichimoku configuration
ENABLE_DYNAMIC_ICHIMOKU_CONFIG = True   # Enable optimization-based parameters

# Enhanced Dynamic Ichimoku Strategy Configurations with market conditions
ICHIMOKU_STRATEGY_CONFIG = {
    'traditional': {
        'tenkan_period': 9,
        'kijun_period': 26,
        'senkou_b_period': 52,
        'chikou_shift': 26,
        'cloud_shift': 26,
        'description': 'Traditional Ichimoku settings (9-26-52-26) - balanced for all market conditions',
        'best_for': ['trending', 'medium_volatility'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.USDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP'],
        'min_pip_volatility': 10.0,
        'max_pip_volatility': 80.0
    },
    'fast_scalping': {
        'tenkan_period': 7,
        'kijun_period': 22,
        'senkou_b_period': 44,
        'chikou_shift': 22,
        'cloud_shift': 22,
        'description': 'Faster Ichimoku for scalping and short-term trades',
        'best_for': ['high_volatility', 'scalping', 'breakouts'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'breakout',
        'best_session': ['london', 'new_york_overlap'],
        'preferred_pairs': ['CS.D.GBPUSD.MINI.IP', 'CS.D.GBPJPY.MINI.IP'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 100.0
    },
    'conservative': {
        'tenkan_period': 12,
        'kijun_period': 30,
        'senkou_b_period': 60,
        'chikou_shift': 30,
        'cloud_shift': 30,
        'description': 'Conservative Ichimoku for stable trends and lower volatility',
        'best_for': ['stable_trends', 'low_volatility'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 40.0
    }
}

# Active configuration (will be overridden by optimization system if enabled)
ACTIVE_ICHIMOKU_CONFIG = 'traditional'

# =============================================================================
# ICHIMOKU VALIDATION FILTERS
# =============================================================================

# Cloud Filter Settings (OPTIMAL: Cloud filters cause signal conflicts)
ICHIMOKU_CLOUD_FILTER_ENABLED = False                # DISABLED - conflicts with signal generation logic
ICHIMOKU_CLOUD_BUFFER_PIPS = 5.0                     # Not used when disabled
ICHIMOKU_CLOUD_THICKNESS_FILTER_ENABLED = False      # DISABLED - not effective for this strategy
ICHIMOKU_MIN_CLOUD_THICKNESS_RATIO = 0.0003          # Not used when disabled

# TK Line Filter Settings
ICHIMOKU_TK_FILTER_ENABLED = True                     # Validate Tenkan-Kijun alignment
ICHIMOKU_MIN_TK_SEPARATION = 0.0005                  # Minimum TK separation for strong signals
ICHIMOKU_TK_CROSS_CONFIRMATION_BARS = 1              # Bars to confirm TK crossover

# Chikou Span Filter Settings (MOST EFFECTIVE FILTER - RE-ENABLED WITH LOOSER BUFFER)
ICHIMOKU_CHIKOU_FILTER_ENABLED = True                # Re-enabled for signal quality
ICHIMOKU_CHIKOU_BUFFER_PIPS = 0.5                    # Tighter buffer for more signals (was 1.5)
ICHIMOKU_CHIKOU_PERIODS = 26                         # Chiimoku lookback period

# =============================================================================
# ICHIMOKU SIGNAL VALIDATION THRESHOLDS
# =============================================================================

# Signal Strength Thresholds (More permissive for signal generation)
ICHIMOKU_CLOUD_THICKNESS_THRESHOLD = 0.0001          # Lower minimum cloud thickness for more signals
ICHIMOKU_TK_CROSS_STRENGTH_THRESHOLD = 0.2           # Lower minimum TK cross strength for more signals
ICHIMOKU_CHIKOU_CLEAR_THRESHOLD = 0.0002             # Lower minimum Chikou clearance

# =============================================================================
# MULTI-TIMEFRAME (MTF) SETTINGS
# =============================================================================

# MTF Configuration
ICHIMOKU_MTF_ENABLED = True                          # Enable multi-timeframe validation
ICHIMOKU_MTF_TIMEFRAMES = ['15m', '1h', '4h']        # Timeframes to validate
ICHIMOKU_MTF_MIN_ALIGNMENT = 0.6                     # Minimum MTF alignment ratio
ICHIMOKU_MTF_CLOUD_WEIGHT = 0.4                      # Weight for cloud alignment
ICHIMOKU_MTF_TK_WEIGHT = 0.3                         # Weight for TK alignment
ICHIMOKU_MTF_CHIKOU_WEIGHT = 0.3                     # Weight for Chikou alignment

# MTF Debug Settings
ICHIMOKU_MTF_DEBUG = True                            # Enable MTF debug logging
ICHIMOKU_LOG_MTF_DECISIONS = True                    # Log MTF decision details

ICHIMOKU_MTF_LOGGING = {
    'log_alignment_details': True,
    'log_cloud_analysis': True,
    'log_tk_analysis': True,
    'log_chikou_analysis': True,
    'log_timeframe_weights': True
}

# =============================================================================
# ICHIMOKU MOMENTUM AND CONFLUENCE SETTINGS
# =============================================================================

# Momentum Confluence
ICHIMOKU_MOMENTUM_CONFLUENCE_ENABLED = False         # Enable additional momentum indicators
ICHIMOKU_MIN_CONFLUENCE_RATIO = 0.5                  # Minimum confluence for signal validation

# Enhanced validation with other indicators
ICHIMOKU_EMA_200_TREND_FILTER = True                 # Use EMA 200 as additional trend filter
ICHIMOKU_RSI_CONFLUENCE_ENABLED = False              # Use RSI for confluence
ICHIMOKU_MACD_CONFLUENCE_ENABLED = True              # Use MACD histogram filter
ICHIMOKU_MACD_HISTOGRAM_FILTER = True                # Require MACD histogram alignment
ICHIMOKU_MACD_HISTOGRAM_MIN_VALUE = 0.0001           # Minimum histogram value for signal validation

# =============================================================================
# SMART MONEY INTEGRATION SETTINGS
# =============================================================================

# Smart Money Ichimoku Settings
SMART_ICHIMOKU_ORDER_FLOW_VALIDATION = True         # Enable order flow validation
SMART_ICHIMOKU_REQUIRE_OB_CONFLUENCE = False        # Require order block confluence (strict)
SMART_ICHIMOKU_FVG_PROXIMITY_PIPS = 20              # Max distance to FVG for confluence
SMART_ICHIMOKU_ORDER_FLOW_BOOST = 1.2               # Confidence boost for strong order flow
SMART_ICHIMOKU_ORDER_FLOW_PENALTY = 0.8             # Confidence penalty for poor order flow
USE_SMART_MONEY_ICHIMOKU = False                     # Enable smart money integration

# =============================================================================
# ICHIMOKU SIGNAL QUALITY SETTINGS
# =============================================================================

# Contradiction Filters
ENABLE_ICHIMOKU_CONTRADICTION_FILTER = True         # Prevent contradictory signals
ICHIMOKU_SIGNAL_SPACING_MINUTES = 60                # Minimum time between signals (minutes)

# Signal Quality Thresholds (OPTIMIZED)
ICHIMOKU_MIN_SIGNAL_CONFIDENCE = 0.50               # Raised to 50% for better quality
ICHIMOKU_PERFECT_ALIGNMENT_BONUS = 0.1              # Bonus for perfect Ichimoku alignment

# Detection Mode
ICHIMOKU_DETECTION_MODE = 'balanced'                 # Balanced signal detection mode
ICHIMOKU_REQUIRE_PERFECT_ALIGNMENT = False           # Don't require all components aligned

# =============================================================================
# ADX TREND STRENGTH FILTER
# =============================================================================

# ADX Filter Settings - Filters out ranging/weak trend markets
ICHIMOKU_ADX_FILTER_ENABLED = True                  # Enable ADX trend strength filter
ICHIMOKU_ADX_MIN_THRESHOLD = 20                     # OPTIMAL: Minimum ADX for signal generation (20 = moderate trend)
ICHIMOKU_ADX_STRONG_THRESHOLD = 25                  # ADX above this = strong trend (confidence bonus)
ICHIMOKU_ADX_PERIOD = 14                            # ADX calculation period (standard)

# ADX Filter Modes
ICHIMOKU_ADX_STRICT_MODE = False                    # OPTIMAL: Non-strict mode - apply confidence penalty instead of rejection
ICHIMOKU_ADX_CONFIDENCE_PENALTY = 0.10              # Confidence penalty if ADX < threshold (10% penalty for weak trends)

# Minimum bars for stable Ichimoku calculation
MIN_BARS_FOR_ICHIMOKU = 80                          # Need 52 + 26 + buffer bars

# =============================================================================
# ICHIMOKU RISK MANAGEMENT SETTINGS
# =============================================================================

# Default Risk Parameters (will be overridden by optimization if available)
ICHIMOKU_DEFAULT_STOP_LOSS_PIPS = 15.0              # Default stop loss (wider for cloud analysis)
ICHIMOKU_DEFAULT_TAKE_PROFIT_PIPS = 30.0            # Default take profit (2:1 RR)
ICHIMOKU_DEFAULT_RISK_REWARD_RATIO = 2.0            # Default risk:reward ratio

# Cloud-based risk management
ICHIMOKU_USE_CLOUD_STOPS = True                     # Use cloud boundaries for stops
ICHIMOKU_CLOUD_STOP_BUFFER_PIPS = 3.0               # Buffer beyond cloud for stops

# =============================================================================
# ICHIMOKU PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================

# Optimization Integration
ICHIMOKU_USE_OPTIMIZATION_SYSTEM = True             # Use database optimization if available
ICHIMOKU_OPTIMIZATION_FALLBACK = 'traditional'      # Fallback config if optimization unavailable

# Performance Tracking
ICHIMOKU_TRACK_SIGNAL_QUALITY = True                # Track signal quality metrics
ICHIMOKU_TRACK_CLOUD_PERFORMANCE = True             # Track cloud-specific performance
ICHIMOKU_TRACK_TK_PERFORMANCE = True                # Track TK cross performance
ICHIMOKU_TRACK_CHIKOU_PERFORMANCE = True            # Track Chikou confirmation performance

# =============================================================================
# ICHIMOKU ADVANCED FEATURES
# =============================================================================

# Kumo (Cloud) Analysis
ICHIMOKU_ADVANCED_CLOUD_ANALYSIS = True             # Enable advanced cloud analysis
ICHIMOKU_CLOUD_TWIST_DETECTION = True               # Detect cloud twists (trend changes)
ICHIMOKU_CLOUD_COLOR_SIGNIFICANCE = True            # Consider cloud color (Senkou A vs B)

# Market Structure Integration
ICHIMOKU_MARKET_STRUCTURE_INTEGRATION = False       # Integrate with market structure analysis
ICHIMOKU_SUPPORT_RESISTANCE_LEVELS = True           # Consider TK lines as support/resistance

# Time-based Filters
ICHIMOKU_SESSION_FILTERS = {
    'asian_session': {'enabled': True, 'confidence_modifier': 0.9},
    'london_session': {'enabled': True, 'confidence_modifier': 1.1},
    'new_york_session': {'enabled': True, 'confidence_modifier': 1.0},
    'overlap_sessions': {'enabled': True, 'confidence_modifier': 1.2}
}

# =============================================================================
# ICHIMOKU DEBUG AND LOGGING SETTINGS
# =============================================================================

# Debug Configuration
ICHIMOKU_DEBUG_MODE = False                          # Enable detailed debug logging
ICHIMOKU_LOG_COMPONENT_VALUES = True                # Log all Ichimoku component values
ICHIMOKU_LOG_SIGNAL_BREAKDOWN = True                # Log signal decision breakdown
ICHIMOKU_LOG_CLOUD_ANALYSIS = True                  # Log cloud analysis details
ICHIMOKU_LOG_TK_ANALYSIS = True                     # Log TK line analysis
ICHIMOKU_LOG_CHIKOU_ANALYSIS = True                 # Log Chikou span analysis

# Performance Logging
ICHIMOKU_LOG_PERFORMANCE_METRICS = True             # Log performance tracking
ICHIMOKU_LOG_OPTIMIZATION_USAGE = True              # Log when optimization parameters used

# =============================================================================
# ICHIMOKU STRATEGY COMPATIBILITY SETTINGS
# =============================================================================

# Integration with other strategies
ICHIMOKU_ALLOW_STRATEGY_COMBINATION = True          # Allow combination with other strategies
ICHIMOKU_EMA_COMBINATION_WEIGHT = 0.3               # Weight when combined with EMA
ICHIMOKU_MACD_COMBINATION_WEIGHT = 0.3              # Weight when combined with MACD
ICHIMOKU_STANDALONE_WEIGHT = 1.0                    # Weight when used standalone

# Backtest Compatibility
ICHIMOKU_BACKTEST_MODE_ADJUSTMENTS = {
    'disable_mtf': True,                             # Disable MTF in backtest for performance
    'simplified_validation': False,                  # Use simplified validation in backtest
    'enhanced_logging': True                         # Enhanced logging in backtest mode
}

# Scanner Integration
ICHIMOKU_SCANNER_INTEGRATION = {
    'enabled': True,                                 # Enable in scanner
    'priority': 'medium',                           # Scanner priority (low, medium, high)
    'max_signals_per_scan': 10,                     # Max signals per scanner run
    'cooldown_minutes': 30                          # Cooldown between signals for same epic
}
# =============================================================================
# SWING PROXIMITY VALIDATION SETTINGS
# =============================================================================

# Swing Proximity Validation - Prevents poor entry timing near support/resistance
ICHIMOKU_SWING_VALIDATION = {
    'enabled': True,                    # Test 1: Swing validation only
    'min_distance_pips': 5,             # Reduced from 8 to 5 pips for testing
    'lookback_swings': 5,               # Number of recent swings to check
    'strict_mode': False,               # False = apply confidence penalty, True = reject signal entirely
    'resistance_buffer': 1.0,           # Multiplier for resistance proximity (BUY signals)
    'support_buffer': 1.0,              # Multiplier for support proximity (SELL signals)
}

# Swing Detection Parameters
ICHIMOKU_SWING_LENGTH = 5               # Swing detection length in bars (matches SMC default)

# Notes:
# - Prevents BUY signals when price is too close to recent swing highs (resistance)
# - Prevents SELL signals when price is too close to recent swing lows (support)
# - Works in conjunction with S/R validation (different timeframes)
# - Swing points: Recent pivots (5-50 bars) on current timeframe
# - S/R levels: Long-term zones (100-500 bars)
# - 8 pips = practical minimum for 15m timeframe intraday trading
# - For JPY pairs: 8 pips = 0.08 price movement
# - Example: EUR/USD at 1.0850, swing high at 1.0857 → 7 pips away → WARNING
# - Example: USD/JPY at 150.50, swing high at 150.57 → 7 pips away → WARNING
