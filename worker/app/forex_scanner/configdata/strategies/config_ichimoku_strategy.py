# configdata/strategies/config_ichimoku_strategy.py
"""
Ichimoku Cloud Strategy Configuration
Configuration module for Ichimoku Kinko Hyo (One Glance Equilibrium Chart) strategy settings
"""

# =============================================================================
# ICHIMOKU STRATEGY CORE CONFIGURATION SETTINGS
# =============================================================================

# Core Strategy Settings
ICHIMOKU_CLOUD_STRATEGY = True          # True = enable Ichimoku Cloud strategy
STRATEGY_WEIGHT_ICHIMOKU = 0.6          # Ichimoku strategy weight (slightly higher due to comprehensive nature)

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

# Cloud Filter Settings (Made very permissive)
ICHIMOKU_CLOUD_FILTER_ENABLED = False                # Disable strict cloud position validation
ICHIMOKU_CLOUD_BUFFER_PIPS = 20.0                    # Very large buffer for cloud position validation
ICHIMOKU_CLOUD_THICKNESS_FILTER_ENABLED = False      # Disable cloud thickness filtering
ICHIMOKU_MIN_CLOUD_THICKNESS_RATIO = 0.00001         # Very low minimum cloud thickness requirement

# TK Line Filter Settings
ICHIMOKU_TK_FILTER_ENABLED = True                     # Validate Tenkan-Kijun alignment
ICHIMOKU_MIN_TK_SEPARATION = 0.0005                  # Minimum TK separation for strong signals
ICHIMOKU_TK_CROSS_CONFIRMATION_BARS = 1              # Bars to confirm TK crossover

# Chikou Span Filter Settings
ICHIMOKU_CHIKOU_FILTER_ENABLED = False               # Disable overly strict Chikou validation
ICHIMOKU_CHIKOU_BUFFER_PIPS = 5.0                    # Larger buffer for Chikou validation
ICHIMOKU_CHIKOU_PERIODS = 26                         # Chikou lookback period

# =============================================================================
# ICHIMOKU SIGNAL VALIDATION THRESHOLDS
# =============================================================================

# Signal Strength Thresholds (Made more permissive)
ICHIMOKU_CLOUD_THICKNESS_THRESHOLD = 0.00001         # Lower minimum cloud thickness
ICHIMOKU_TK_CROSS_STRENGTH_THRESHOLD = 0.1           # Lower minimum TK cross strength
ICHIMOKU_CHIKOU_CLEAR_THRESHOLD = 0.00001            # Lower minimum Chikou clearance

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
ICHIMOKU_EMA_200_TREND_FILTER = False                # Use EMA 200 as additional trend filter
ICHIMOKU_RSI_CONFLUENCE_ENABLED = False              # Use RSI for confluence
ICHIMOKU_MACD_CONFLUENCE_ENABLED = False             # Use MACD for confluence

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

# Signal Quality Thresholds (Made more permissive)
ICHIMOKU_MIN_SIGNAL_CONFIDENCE = 0.35               # Lower minimum confidence for signal generation
ICHIMOKU_PERFECT_ALIGNMENT_BONUS = 0.1              # Bonus for perfect Ichimoku alignment

# Detection Mode
ICHIMOKU_DETECTION_MODE = 'aggressive'               # More permissive signal detection
ICHIMOKU_REQUIRE_PERFECT_ALIGNMENT = False           # Don't require all components aligned

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