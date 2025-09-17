# configdata/strategies/config_mean_reversion_strategy.py
"""
Mean Reversion Strategy Configuration
Configuration module for multi-oscillator mean reversion strategy settings
Based on RAG analysis findings: LuxAlgo Premium Oscillator, ChrisMoody RSI, Divergence Detection, Squeeze Momentum
"""

# =============================================================================
# MEAN REVERSION STRATEGY CONFIGURATION SETTINGS
# =============================================================================

# Core Strategy Settings
MEAN_REVERSION_STRATEGY = True     # True = enable mean reversion strategy
STRATEGY_WEIGHT_MEAN_REVERSION = 0.4   # Mean reversion strategy weight

# =============================================================================
# LUXALGO PREMIUM OSCILLATOR CONFIGURATION
# =============================================================================

# LuxAlgo Oscillator Core Settings (Primary Mean Reversion Engine)
LUXALGO_OSCILLATOR_ENABLED = True
LUXALGO_LENGTH = 14                    # Oscillator calculation period
LUXALGO_SOURCE = 'close'               # Price source for calculations
LUXALGO_SMOOTHING = 3                  # Smoothing factor for oscillator
LUXALGO_OVERBOUGHT_THRESHOLD = 80      # Overbought level for mean reversion
LUXALGO_OVERSOLD_THRESHOLD = 20        # Oversold level for mean reversion
LUXALGO_EXTREME_OB_THRESHOLD = 90      # Extreme overbought (high probability reversal)
LUXALGO_EXTREME_OS_THRESHOLD = 10      # Extreme oversold (high probability reversal)

# LuxAlgo Signal Generation
LUXALGO_REQUIRE_DIVERGENCE = True      # Require price/oscillator divergence
LUXALGO_MIN_DIVERGENCE_BARS = 3        # Minimum bars for divergence pattern
LUXALGO_DIVERGENCE_LOOKBACK = 20       # Lookback period for divergence detection

# =============================================================================
# MULTI-TIMEFRAME RSI CONFIGURATION (ChrisMoody Style)
# =============================================================================

# Multi-Timeframe RSI Settings
MTF_RSI_ENABLED = True
MTF_RSI_PERIOD = 14                    # RSI calculation period
MTF_RSI_TIMEFRAMES = ['15m', '1h', '4h']  # Timeframes for confluence
MTF_RSI_MIN_ALIGNMENT = 0.6            # Minimum alignment score for signal
MTF_RSI_OVERBOUGHT = 70                # RSI overbought threshold
MTF_RSI_OVERSOLD = 30                  # RSI oversold threshold
MTF_RSI_EXTREME_OB = 80                # Extreme overbought threshold
MTF_RSI_EXTREME_OS = 20                # Extreme oversold threshold

# RSI Confluence Requirements
MTF_RSI_REQUIRE_HIGHER_TF_CONFIRMATION = True  # Require higher timeframe confirmation
MTF_RSI_CONFIDENCE_BOOST_ALIGNMENT = 0.15      # Confidence boost for full alignment
MTF_RSI_WEIGHT_BY_TIMEFRAME = True             # Weight higher timeframes more

# =============================================================================
# RSI-EMA DIVERGENCE DETECTION (ChrisMoody Style)
# =============================================================================

# RSI-EMA Divergence Settings
RSI_EMA_DIVERGENCE_ENABLED = True
RSI_EMA_PERIOD = 21                    # EMA period for divergence detection
RSI_EMA_RSI_PERIOD = 14                # RSI period for divergence
RSI_EMA_DIVERGENCE_SENSITIVITY = 0.6   # Sensitivity for divergence detection (0.1-1.0)
RSI_EMA_MIN_DIVERGENCE_STRENGTH = 0.7  # Minimum divergence strength for signal

# Divergence Pattern Requirements
RSI_EMA_REQUIRE_HIGHER_HIGHS = True    # For bearish divergence: price higher highs
RSI_EMA_REQUIRE_LOWER_LOWS = True      # For bullish divergence: price lower lows
RSI_EMA_MIN_PATTERN_BARS = 5           # Minimum bars for divergence pattern
RSI_EMA_MAX_PATTERN_BARS = 25          # Maximum bars for divergence pattern
RSI_EMA_DIVERGENCE_CONFIRMATION_BARS = 2  # Bars to confirm divergence signal

# =============================================================================
# SQUEEZE MOMENTUM INDICATOR (LazyBear Style)
# =============================================================================

# Squeeze Momentum Settings
SQUEEZE_MOMENTUM_ENABLED = True
SQUEEZE_BB_LENGTH = 20                 # Bollinger Bands length
SQUEEZE_BB_MULT = 2.0                  # Bollinger Bands multiplier
SQUEEZE_KC_LENGTH = 20                 # Keltner Channel length
SQUEEZE_KC_MULT = 1.5                  # Keltner Channel multiplier
SQUEEZE_MOMENTUM_LENGTH = 12           # Momentum calculation length

# Squeeze Signal Generation
SQUEEZE_REQUIRE_SQUEEZE_RELEASE = True  # Only signal on squeeze release
SQUEEZE_MIN_SQUEEZE_BARS = 6           # Minimum bars in squeeze for valid signal
SQUEEZE_MOMENTUM_THRESHOLD = 0.1       # Minimum momentum for directional signal
SQUEEZE_CONFIRMATION_BARS = 2          # Bars to confirm momentum direction

# =============================================================================
# OSCILLATOR CONFLUENCE REQUIREMENTS
# =============================================================================

# Multi-Oscillator Confluence Settings
OSCILLATOR_CONFLUENCE_ENABLED = True
OSCILLATOR_MIN_CONFIRMATIONS = 2       # Minimum oscillators agreeing for signal
OSCILLATOR_WEIGHTS = {                 # Weight for each oscillator in confluence
    'luxalgo': 0.4,                    # Primary mean reversion engine
    'mtf_rsi': 0.3,                    # Multi-timeframe confirmation
    'divergence': 0.2,                 # Reversal pattern detection
    'squeeze': 0.1                     # Volatility/momentum timing
}

# Confluence Thresholds
OSCILLATOR_BULL_CONFLUENCE_THRESHOLD = 0.65  # Minimum weighted score for bullish signal
OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD = 0.65  # Minimum weighted score for bearish signal
OSCILLATOR_EXTREME_CONFLUENCE_BOOST = 0.15   # Boost for extreme readings

# =============================================================================
# MEAN REVERSION ZONE VALIDATION
# =============================================================================

# Mean Reversion Zone Settings
MEAN_REVERSION_ZONE_ENABLED = True
MEAN_REVERSION_LOOKBACK_PERIODS = 50   # Periods to calculate mean reversion zones
MEAN_REVERSION_ZONE_MULTIPLIER = 1.5   # Standard deviation multiplier for zones
MEAN_REVERSION_REQUIRE_ZONE_TOUCH = True  # Require price to touch reversion zone

# Zone Validation Criteria
MEAN_REVERSION_MIN_ZONE_DISTANCE = 10  # Minimum pips from mean for valid zone
MEAN_REVERSION_MAX_ZONE_AGE = 100      # Maximum bars since zone establishment
MEAN_REVERSION_ZONE_CONFIDENCE_BOOST = 0.1  # Confidence boost for zone signals

# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

# Market Regime Settings
MARKET_REGIME_DETECTION_ENABLED = True
MARKET_REGIME_VOLATILITY_PERIOD = 20   # Period for volatility calculation
MARKET_REGIME_TREND_PERIOD = 50        # Period for trend strength calculation
MARKET_REGIME_RANGING_THRESHOLD = 0.3  # Threshold for ranging market detection

# Regime-Based Signal Filtering
MARKET_REGIME_DISABLE_IN_STRONG_TREND = True  # Disable mean reversion in strong trends
MARKET_REGIME_BOOST_IN_RANGING = True         # Boost confidence in ranging markets
MARKET_REGIME_TREND_STRENGTH_THRESHOLD = 0.7  # Threshold for strong trend detection

# =============================================================================
# MULTI-TIMEFRAME ANALYSIS
# =============================================================================

# MTF Analysis Settings
MTF_ANALYSIS_ENABLED = True
MTF_TIMEFRAMES = ['15m', '1h', '4h']   # Timeframes for analysis
MTF_MIN_ALIGNMENT_SCORE = 0.6          # Minimum alignment for signal
MTF_HIGHER_TF_WEIGHT = 1.5             # Weight multiplier for higher timeframes

# MTF Signal Requirements
MTF_REQUIRE_HIGHER_TF_CONFLUENCE = True  # Require higher timeframe agreement
MTF_ALLOW_LOWER_TF_ENTRY = True         # Allow lower timeframe entry timing
MTF_CONFIDENCE_BOOST_FULL_ALIGNMENT = 0.2  # Boost for full MTF alignment

# =============================================================================
# SIGNAL QUALITY AND FILTERING
# =============================================================================

# Signal Quality Requirements
SIGNAL_QUALITY_MIN_CONFIDENCE = 0.6    # Minimum confidence for signal generation
SIGNAL_QUALITY_REQUIRE_VOLUME_CONFIRMATION = False  # Volume confirmation (if available)
SIGNAL_QUALITY_MIN_RISK_REWARD = 1.5   # Minimum risk-reward ratio

# Signal Filtering
SIGNAL_FILTER_MAX_SIGNALS_PER_DAY = 5   # Maximum signals per epic per day
SIGNAL_FILTER_MIN_SIGNAL_SPACING = 4    # Minimum hours between signals
SIGNAL_FILTER_AVOID_NEWS_EVENTS = True  # Avoid signals during major news

# =============================================================================
# RISK MANAGEMENT SETTINGS
# =============================================================================

# Position Sizing
MEAN_REVERSION_POSITION_SIZE_MULTIPLIER = 0.8  # Conservative position sizing
MEAN_REVERSION_MAX_DRAWDOWN_THRESHOLD = 0.05   # Stop strategy at 5% drawdown

# Stop Loss and Take Profit
MEAN_REVERSION_DEFAULT_SL_PIPS = 25     # Default stop loss in pips
MEAN_REVERSION_DEFAULT_TP_PIPS = 40     # Default take profit in pips
MEAN_REVERSION_DYNAMIC_SL_TP = True     # Use dynamic SL/TP based on volatility
MEAN_REVERSION_TRAIL_STOP_ENABLED = True  # Enable trailing stop

# =============================================================================
# SAFETY PRESET CONFIGURATIONS
# =============================================================================

MEAN_REVERSION_SAFETY_PRESETS = {
    'conservative': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.75,
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.75,
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.75,
        'LUXALGO_EXTREME_OB_THRESHOLD': 85,
        'LUXALGO_EXTREME_OS_THRESHOLD': 15,
        'MTF_RSI_MIN_ALIGNMENT': 0.7,
        'MEAN_REVERSION_REQUIRE_ZONE_TOUCH': True,
        'MARKET_REGIME_DISABLE_IN_STRONG_TREND': True
    },
    'balanced': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.65,
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.65,
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.6,
        'LUXALGO_EXTREME_OB_THRESHOLD': 80,
        'LUXALGO_EXTREME_OS_THRESHOLD': 20,
        'MTF_RSI_MIN_ALIGNMENT': 0.6,
        'MEAN_REVERSION_REQUIRE_ZONE_TOUCH': True,
        'MARKET_REGIME_DISABLE_IN_STRONG_TREND': True
    },
    'aggressive': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.55,
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.55,
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.5,
        'LUXALGO_EXTREME_OB_THRESHOLD': 75,
        'LUXALGO_EXTREME_OS_THRESHOLD': 25,
        'MTF_RSI_MIN_ALIGNMENT': 0.5,
        'MEAN_REVERSION_REQUIRE_ZONE_TOUCH': False,
        'MARKET_REGIME_DISABLE_IN_STRONG_TREND': False
    }
}

# =============================================================================
# EPIC-SPECIFIC CONFIGURATION
# =============================================================================

def get_mean_reversion_threshold_for_epic(epic: str) -> dict:
    """Get mean reversion thresholds for specific epic"""
    # JPY pairs typically have different volatility characteristics
    if 'JPY' in epic.upper():
        return {
            'luxalgo_overbought': 85,
            'luxalgo_oversold': 15,
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'squeeze_momentum_threshold': 0.15,
            'min_zone_distance': 15  # Higher pip requirement for JPY
        }
    # High volatility pairs (GBP, AUD, NZD)
    elif any(curr in epic.upper() for curr in ['GBP', 'AUD', 'NZD']):
        return {
            'luxalgo_overbought': 80,
            'luxalgo_oversold': 20,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'squeeze_momentum_threshold': 0.12,
            'min_zone_distance': 12
        }
    # Standard pairs (EUR, USD, CHF, CAD)
    else:
        return {
            'luxalgo_overbought': 80,
            'luxalgo_oversold': 20,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'squeeze_momentum_threshold': 0.1,
            'min_zone_distance': 10
        }

# =============================================================================
# STRATEGY INTEGRATION SETTINGS
# =============================================================================

# Strategy Integration
MEAN_REVERSION_STRATEGY_WEIGHT = 0.25    # Weight in combined strategy mode
MEAN_REVERSION_ALLOW_COMBINED = True     # Allow in combined strategies
MEAN_REVERSION_PRIORITY_LEVEL = 2        # Priority level (1=highest, 5=lowest)

# Performance Settings
MEAN_REVERSION_ENABLE_BACKTESTING = True      # Enable strategy in backtests
MEAN_REVERSION_MIN_DATA_PERIODS = 100         # Minimum data periods required
MEAN_REVERSION_ENABLE_PERFORMANCE_TRACKING = True  # Track strategy performance

# Debug Settings
MEAN_REVERSION_DEBUG_LOGGING = True           # Enable detailed debug logging
MEAN_REVERSION_LOG_OSCILLATOR_VALUES = True  # Log oscillator values for debugging
MEAN_REVERSION_LOG_CONFLUENCE_SCORES = True  # Log confluence calculations

# =============================================================================
# CONFIGURATION SUMMARY AND VALIDATION
# =============================================================================

def get_mean_reversion_config_summary() -> dict:
    """Get a summary of mean reversion configuration settings"""
    return {
        'strategy_enabled': MEAN_REVERSION_STRATEGY,
        'luxalgo_oscillator_enabled': LUXALGO_OSCILLATOR_ENABLED,
        'mtf_rsi_enabled': MTF_RSI_ENABLED,
        'divergence_detection_enabled': RSI_EMA_DIVERGENCE_ENABLED,
        'squeeze_momentum_enabled': SQUEEZE_MOMENTUM_ENABLED,
        'oscillator_confluence_enabled': OSCILLATOR_CONFLUENCE_ENABLED,
        'market_regime_detection_enabled': MARKET_REGIME_DETECTION_ENABLED,
        'mtf_analysis_enabled': MTF_ANALYSIS_ENABLED,
        'min_confidence_threshold': SIGNAL_QUALITY_MIN_CONFIDENCE,
        'confluence_threshold': OSCILLATOR_BULL_CONFLUENCE_THRESHOLD,
        'timeframes': MTF_TIMEFRAMES,
        'min_data_periods': MEAN_REVERSION_MIN_DATA_PERIODS,
        'debug_logging': MEAN_REVERSION_DEBUG_LOGGING
    }

def validate_mean_reversion_config() -> bool:
    """Validate mean reversion configuration for consistency"""
    validation_errors = []

    # Check threshold consistency
    if LUXALGO_OVERBOUGHT_THRESHOLD <= LUXALGO_OVERSOLD_THRESHOLD:
        validation_errors.append("LuxAlgo overbought threshold must be > oversold threshold")

    if MTF_RSI_OVERBOUGHT <= MTF_RSI_OVERSOLD:
        validation_errors.append("MTF RSI overbought threshold must be > oversold threshold")

    # Check confluence weights
    total_weight = sum(OSCILLATOR_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        validation_errors.append(f"Oscillator weights must sum to 1.0, got {total_weight}")

    # Check timeframe consistency
    if MTF_ANALYSIS_ENABLED and not MTF_TIMEFRAMES:
        validation_errors.append("MTF analysis enabled but no timeframes specified")

    if validation_errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")

    return True

# Validate configuration on import
try:
    validate_mean_reversion_config()
except ValueError as e:
    import logging
    logging.getLogger(__name__).error(f"Mean reversion configuration validation failed: {e}")