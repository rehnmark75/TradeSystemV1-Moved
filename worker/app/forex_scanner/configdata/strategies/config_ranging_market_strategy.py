# configdata/strategies/config_ranging_market_strategy.py
"""
Ranging Market Strategy Configuration
Configuration module for ranging market optimization strategy settings
Based on TradingView analysis: LazyBear Squeeze Momentum, Wave Trend Oscillator, Bollinger Bands/Keltner Channels
"""

# =============================================================================
# RANGING MARKET STRATEGY CONFIGURATION SETTINGS
# =============================================================================

# Core Strategy Settings
RANGING_MARKET_STRATEGY = True     # True = enable ranging market strategy
STRATEGY_WEIGHT_RANGING_MARKET = 0.35   # Ranging market strategy weight

# =============================================================================
# SQUEEZE MOMENTUM INDICATOR CONFIGURATION (LazyBear)
# =============================================================================

# Squeeze Momentum Core Settings (Primary Ranging Engine)
SQUEEZE_MOMENTUM_ENABLED = True
SQUEEZE_BB_LENGTH = 20                 # Bollinger Bands length
SQUEEZE_BB_MULT = 2.0                  # Bollinger Bands multiplier
SQUEEZE_KC_LENGTH = 20                 # Keltner Channel length
SQUEEZE_KC_MULT = 1.5                  # Keltner Channel multiplier
SQUEEZE_MOMENTUM_LENGTH = 12           # Momentum calculation length
SQUEEZE_USE_TRUE_RANGE = True          # Use True Range for Keltner Channels

# Squeeze Detection Settings
SQUEEZE_MIN_BARS_IN_SQUEEZE = 6        # Minimum consecutive bars in squeeze
SQUEEZE_MAX_BARS_IN_SQUEEZE = 50       # Maximum bars in squeeze (avoid dead ranges)
SQUEEZE_RELEASE_CONFIRMATION_BARS = 2   # Bars to confirm squeeze release
SQUEEZE_MOMENTUM_THRESHOLD = 0.1       # Minimum momentum for directional signal

# Squeeze Signal Quality
SQUEEZE_REQUIRE_MOMENTUM_ALIGNMENT = True  # Require momentum direction alignment
SQUEEZE_MIN_MOMENTUM_CHANGE = 0.05         # Minimum momentum change for valid signal
SQUEEZE_FILTER_FALSE_BREAKOUTS = True      # Filter false squeeze releases

# =============================================================================
# WAVE TREND OSCILLATOR CONFIGURATION (LazyBear WTO)
# =============================================================================

# Wave Trend Oscillator Settings
WAVE_TREND_ENABLED = True
WTO_CHANNEL_LENGTH = 10               # Channel length for WTO calculation
WTO_AVERAGE_LENGTH = 21               # Average length for smoothing
WTO_SIGNAL_LENGTH = 4                 # Signal line length
WTO_OVERBOUGHT_LEVEL = 60            # Overbought threshold
WTO_OVERSOLD_LEVEL = -60             # Oversold threshold

# WTO Signal Generation
WTO_REQUIRE_SIGNAL_CROSSOVER = True   # Require signal line crossover
WTO_MIN_OSCILLATOR_SEPARATION = 20    # Minimum separation for valid signals
WTO_DIVERGENCE_DETECTION = True       # Enable divergence detection
WTO_DIVERGENCE_LOOKBACK = 15          # Lookback period for divergence

# WTO Trend Detection
WTO_TREND_THRESHOLD = 30              # Threshold for trend vs ranging detection
WTO_RANGING_ZONE_UPPER = 40           # Upper bound of ranging zone
WTO_RANGING_ZONE_LOWER = -40          # Lower bound of ranging zone

# =============================================================================
# BOLLINGER BANDS & KELTNER CHANNEL CONFIGURATION
# =============================================================================

# Bollinger Bands Settings
BOLLINGER_BANDS_ENABLED = True
BB_LENGTH = 20                        # Bollinger Bands period
BB_MULTIPLIER = 2.0                   # Standard deviation multiplier
BB_USE_CLOSE_PRICE = True            # Use closing price for calculation
BB_PERCENT_B_ENABLED = True          # Calculate %B indicator

# Keltner Channels Settings
KELTNER_CHANNELS_ENABLED = True
KC_LENGTH = 20                        # Keltner Channel period
KC_MULTIPLIER = 1.5                   # ATR multiplier
KC_USE_EXPONENTIAL_MA = True         # Use EMA instead of SMA

# Band/Channel Analysis
BB_KC_SQUEEZE_RATIO_THRESHOLD = 0.95  # Ratio threshold for squeeze detection
BB_KC_EXPANSION_RATIO_THRESHOLD = 1.1 # Ratio threshold for expansion
BB_TOUCH_CONFIRMATION_BARS = 2        # Bars to confirm band touch
BB_REJECTION_CONFIRMATION_BARS = 3    # Bars to confirm rejection from bands

# =============================================================================
# RSI WITH DIVERGENCE CONFIGURATION
# =============================================================================

# RSI Settings
RSI_ENABLED = True
RSI_PERIOD = 14                       # RSI calculation period
RSI_OVERBOUGHT = 70                  # RSI overbought level
RSI_OVERSOLD = 30                    # RSI oversold level
RSI_EXTREME_OB = 80                  # Extreme overbought level
RSI_EXTREME_OS = 20                  # Extreme oversold level

# RSI Divergence Detection
RSI_DIVERGENCE_ENABLED = True
RSI_DIVERGENCE_LOOKBACK = 20         # Lookback period for divergence
RSI_MIN_DIVERGENCE_BARS = 5          # Minimum bars for divergence pattern
RSI_DIVERGENCE_STRENGTH_THRESHOLD = 0.6  # Minimum divergence strength
RSI_REQUIRE_DOUBLE_DIVERGENCE = False # Require double divergence patterns

# RSI Mean Reversion
RSI_MEAN_REVERSION_ENABLED = True
RSI_MEAN_LEVEL = 50                  # RSI mean level
RSI_MEAN_REVERSION_DISTANCE = 25     # Distance from mean for reversion signals

# =============================================================================
# RELATIVE VIGOR INDEX (RVI) CONFIGURATION
# =============================================================================

# RVI Settings
RVI_ENABLED = True
RVI_PERIOD = 10                      # RVI calculation period
RVI_SIGNAL_PERIOD = 4                # RVI signal line smoothing period
RVI_OVERBOUGHT = 0.5                 # RVI overbought threshold
RVI_OVERSOLD = -0.5                  # RVI oversold threshold

# RVI Signal Generation
RVI_REQUIRE_SIGNAL_CROSSOVER = True   # Require signal line crossover
RVI_MOMENTUM_CONFIRMATION = True      # Use RVI for momentum confirmation
RVI_DIVERGENCE_DETECTION = True       # Enable RVI divergence detection
RVI_MIN_SWING_SIZE = 0.3             # Minimum swing size for valid signals

# =============================================================================
# OSCILLATOR CONFLUENCE REQUIREMENTS
# =============================================================================

# Multi-Oscillator Confluence Settings
OSCILLATOR_CONFLUENCE_ENABLED = True
OSCILLATOR_MIN_CONFIRMATIONS = 2       # Minimum oscillators agreeing for signal (balanced quality)
OSCILLATOR_WEIGHTS = {                 # Weight for each oscillator in confluence
    'squeeze_momentum': 0.35,          # Primary ranging market engine
    'wave_trend': 0.25,                # Trend/momentum hybrid
    'rsi': 0.20,                       # Traditional mean reversion
    'rvi': 0.20                        # Additional momentum confirmation
}

# Confluence Thresholds
OSCILLATOR_BULL_CONFLUENCE_THRESHOLD = 0.12  # Minimum weighted score for bullish signal (balanced production)
OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD = 0.12  # Minimum weighted score for bearish signal (balanced production)
OSCILLATOR_EXTREME_CONFLUENCE_BOOST = 0.15   # Boost for extreme readings

# Signal Quality Requirements
CONFLUENCE_REQUIRE_PRIMARY_AGREEMENT = False  # Squeeze momentum (primary engine) must agree for signal
CONFLUENCE_ALLOW_MIXED_SIGNALS = False        # Don't allow mixed oscillator signals
CONFLUENCE_MIN_SIGNAL_STRENGTH = 0.6          # Minimum individual signal strength

# =============================================================================
# DYNAMIC SUPPORT/RESISTANCE ZONES
# =============================================================================

# Support/Resistance Zone Settings
DYNAMIC_ZONES_ENABLED = False          # Disabled for production - relying on oscillator confluence only
ZONE_CALCULATION_PERIOD = 30          # Period for zone calculations (shorter for more frequent updates)
ZONE_STRENGTH_THRESHOLD = 1           # Minimum touches for strong zone (more lenient)
ZONE_PROXIMITY_PIPS = 35              # Pip proximity for zone interaction (extremely generous for production)
ZONE_AGE_LIMIT_BARS = 300            # Maximum age of zones in bars (longer to keep more zones)

# Zone Validation
ZONE_REQUIRE_MULTIPLE_TOUCHES = False  # More lenient zone validation for production
ZONE_MIN_HOLDING_TIME = 1             # Minimum bars price must hold at zone (very lenient)
ZONE_BREAK_CONFIRMATION_BARS = 3      # Bars to confirm zone break
ZONE_RETEST_VALIDATION = True         # Validate zones on retest

# Zone Types
ZONE_INCLUDE_PIVOT_POINTS = True      # Include pivot point zones
ZONE_INCLUDE_FIBONACCI_LEVELS = True  # Include fibonacci retracement levels
ZONE_INCLUDE_VOLUME_PROFILE = False   # Include volume profile zones (if available)

# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

# Market Regime Settings
MARKET_REGIME_DETECTION_ENABLED = True
REGIME_VOLATILITY_PERIOD = 20         # Period for volatility calculation
REGIME_TREND_PERIOD = 50              # Period for trend strength calculation
REGIME_ADX_THRESHOLD = 25             # ADX threshold for trending vs ranging

# Ranging Market Detection
RANGING_MARKET_ADX_MAX = 75           # Maximum ADX for ranging market (very permissive)
RANGING_MARKET_ATR_STABILITY = 0.8    # ATR stability coefficient
RANGING_MARKET_MIN_DURATION = 10      # Minimum bars for ranging regime
RANGING_MARKET_PRICE_OSCILLATION = 0.75  # Price oscillation within range coefficient

# Regime-Based Signal Filtering
REGIME_DISABLE_IN_STRONG_TREND = True  # Disable strategy in strong trends
REGIME_BOOST_IN_RANGING = True         # Boost confidence in ranging markets
REGIME_TREND_STRENGTH_THRESHOLD = 0.7  # Threshold for strong trend detection
REGIME_VOLATILITY_FILTER = True       # Filter based on volatility regime

# =============================================================================
# MULTI-TIMEFRAME ANALYSIS
# =============================================================================

# MTF Analysis Settings
MTF_ANALYSIS_ENABLED = True
MTF_TIMEFRAMES = ['5m', '15m', '1h']   # Timeframes for analysis
MTF_PRIMARY_TIMEFRAME = '15m'          # Primary trading timeframe
MTF_MIN_ALIGNMENT_SCORE = 0.65         # Minimum alignment for signal

# MTF Confluence Requirements
MTF_REQUIRE_HIGHER_TF_RANGING = True   # Require higher TF to be ranging
MTF_ALLOW_LOWER_TF_PRECISION = True    # Use lower TF for precise entry
MTF_CONFIDENCE_BOOST_FULL_ALIGNMENT = 0.2  # Boost for full MTF alignment

# MTF Oscillator Analysis
MTF_OSCILLATOR_ALIGNMENT = True        # Require oscillator alignment across TFs
MTF_SQUEEZE_ALIGNMENT = True           # Require squeeze state alignment
MTF_ZONE_CONFLUENCE = True             # Require S/R zone confluence

# =============================================================================
# SIGNAL QUALITY AND FILTERING
# =============================================================================

# Signal Quality Requirements
SIGNAL_QUALITY_MIN_CONFIDENCE = 0.45   # Minimum confidence for signal generation (balanced quality)
SIGNAL_QUALITY_REQUIRE_VOLUME_CONFIRMATION = False  # Volume confirmation (if available)
SIGNAL_QUALITY_MIN_RISK_REWARD = 1.8   # Minimum risk-reward ratio (high quality)
SIGNAL_QUALITY_MAX_SPREAD_IMPACT = 0.3  # Maximum spread impact on signal quality

# Signal Filtering
SIGNAL_FILTER_MAX_SIGNALS_PER_DAY = 1   # Maximum signals per epic per day (quality over quantity)
SIGNAL_FILTER_MIN_SIGNAL_SPACING = 24   # Minimum hours between signals (daily spacing for proper distribution)
SIGNAL_FILTER_AVOID_NEWS_EVENTS = True  # Avoid signals during major news
SIGNAL_FILTER_SESSION_PREFERENCE = 'london_ny'  # Preferred trading sessions

# Signal Timing Optimization
SIGNAL_ENTRY_TIMING_PRECISION = True   # Use precise entry timing
SIGNAL_WAIT_FOR_PULLBACK = True        # Wait for pullback entry
SIGNAL_MAX_ENTRY_DELAY_BARS = 5        # Maximum bars to wait for entry

# =============================================================================
# RISK MANAGEMENT SETTINGS
# =============================================================================

# Position Sizing
RANGING_POSITION_SIZE_MULTIPLIER = 0.9  # Slightly larger position for ranging
RANGING_MAX_DRAWDOWN_THRESHOLD = 0.04   # Stop strategy at 4% drawdown
RANGING_MAX_DAILY_LOSS = 0.02           # Maximum daily loss threshold

# Stop Loss and Take Profit
RANGING_DEFAULT_SL_PIPS = 22            # Default stop loss in pips
RANGING_DEFAULT_TP_PIPS = 38            # Default take profit in pips
RANGING_DYNAMIC_SL_TP = True            # Use dynamic SL/TP based on range size
RANGING_TRAIL_STOP_ENABLED = True       # Enable trailing stop

# Range-Based Risk Management
RANGING_SL_BASED_ON_RANGE = True        # Base SL on range size
RANGING_TP_BASED_ON_RANGE = True        # Base TP on range size
RANGING_RANGE_SL_MULTIPLIER = 0.4       # SL as % of range size
RANGING_RANGE_TP_MULTIPLIER = 0.8       # TP as % of range size

# =============================================================================
# PHASE 3: Adaptive Volatility-Based SL/TP (NEW)
# =============================================================================

# Runtime regime-aware calculation - No hardcoded values!
USE_ADAPTIVE_SL_TP = False               # 🧠 Enable adaptive volatility calculator (default: False for gradual rollout)
                                         # When True: Uses runtime regime detection (trending, ranging, breakout, high volatility)
                                         # When False: Falls back to ATR multipliers below

# Volatility-Based Adjustments (FALLBACK when adaptive disabled)
RANGING_VOLATILITY_SL_ADJUSTMENT = True  # Adjust SL based on volatility
RANGING_STOP_LOSS_ATR_MULTIPLIER = 1.5   # Tighter stops for ranging/mean reversion
RANGING_TAKE_PROFIT_ATR_MULTIPLIER = 2.5 # Moderate targets for ranging markets
# Legacy aliases for compatibility
RANGING_ATR_SL_MULTIPLIER = 1.5          # ATR multiplier for SL (legacy)
RANGING_ATR_TP_MULTIPLIER = 2.5          # ATR multiplier for TP (legacy)

# =============================================================================
# SAFETY PRESET CONFIGURATIONS
# =============================================================================

RANGING_MARKET_SAFETY_PRESETS = {
    'conservative': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.80,
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.80,
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.75,
        'OSCILLATOR_MIN_CONFIRMATIONS': 4,
        'SQUEEZE_MIN_BARS_IN_SQUEEZE': 8,
        'MTF_MIN_ALIGNMENT_SCORE': 0.75,
        'REGIME_DISABLE_IN_STRONG_TREND': True,
        'SIGNAL_FILTER_MAX_SIGNALS_PER_DAY': 3
    },
    'balanced': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.70,
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.70,
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.65,
        'OSCILLATOR_MIN_CONFIRMATIONS': 3,
        'SQUEEZE_MIN_BARS_IN_SQUEEZE': 6,
        'MTF_MIN_ALIGNMENT_SCORE': 0.65,
        'REGIME_DISABLE_IN_STRONG_TREND': True,
        'SIGNAL_FILTER_MAX_SIGNALS_PER_DAY': 4
    },
    'aggressive': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.60,
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.60,
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.55,
        'OSCILLATOR_MIN_CONFIRMATIONS': 2,
        'SQUEEZE_MIN_BARS_IN_SQUEEZE': 4,
        'MTF_MIN_ALIGNMENT_SCORE': 0.55,
        'REGIME_DISABLE_IN_STRONG_TREND': False,
        'SIGNAL_FILTER_MAX_SIGNALS_PER_DAY': 6
    },
    'production': {
        'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD': 0.15,  # Very low threshold for production
        'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD': 0.15,  # Very low threshold for production
        'SIGNAL_QUALITY_MIN_CONFIDENCE': 0.35,
        'OSCILLATOR_MIN_CONFIRMATIONS': 1,
        'SQUEEZE_MIN_BARS_IN_SQUEEZE': 2,
        'MTF_MIN_ALIGNMENT_SCORE': 0.45,
        'REGIME_DISABLE_IN_STRONG_TREND': False,
        'SIGNAL_FILTER_MAX_SIGNALS_PER_DAY': 8
    }
}

# =============================================================================
# EPIC-SPECIFIC CONFIGURATION
# =============================================================================

def get_ranging_market_threshold_for_epic(epic: str) -> dict:
    """Get ranging market thresholds for specific epic"""
    # JPY pairs typically have different volatility characteristics
    if 'JPY' in epic.upper():
        return {
            'squeeze_momentum_threshold': 0.15,
            'wto_overbought': 70,
            'wto_oversold': -70,
            'rsi_overbought': 75,
            'rsi_oversold': 25,
            'default_sl_pips': 28,
            'default_tp_pips': 45,
            'zone_proximity_pips': 12
        }
    # High volatility pairs (GBP, AUD, NZD)
    elif any(curr in epic.upper() for curr in ['GBP', 'AUD', 'NZD']):
        return {
            'squeeze_momentum_threshold': 0.12,
            'wto_overbought': 65,
            'wto_oversold': -65,
            'rsi_overbought': 72,
            'rsi_oversold': 28,
            'default_sl_pips': 25,
            'default_tp_pips': 42,
            'zone_proximity_pips': 10
        }
    # Standard pairs (EUR, USD, CHF, CAD)
    else:
        return {
            'squeeze_momentum_threshold': 0.1,
            'wto_overbought': 60,
            'wto_oversold': -60,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'default_sl_pips': 22,
            'default_tp_pips': 38,
            'zone_proximity_pips': 8
        }

# =============================================================================
# STRATEGY INTEGRATION SETTINGS
# =============================================================================

# Strategy Integration
RANGING_MARKET_STRATEGY_WEIGHT = 0.35    # Weight in combined strategy mode
RANGING_MARKET_ALLOW_COMBINED = True     # Allow in combined strategies
RANGING_MARKET_PRIORITY_LEVEL = 2        # Priority level (1=highest, 5=lowest)

# Performance Settings
RANGING_MARKET_ENABLE_BACKTESTING = True      # Enable strategy in backtests
RANGING_MARKET_MIN_DATA_PERIODS = 100         # Minimum data periods required
RANGING_MARKET_ENABLE_PERFORMANCE_TRACKING = True  # Track strategy performance

# Debug Settings
RANGING_MARKET_DEBUG_LOGGING = True           # Enable detailed debug logging
RANGING_MARKET_LOG_OSCILLATOR_VALUES = True  # Log oscillator values for debugging
RANGING_MARKET_LOG_CONFLUENCE_SCORES = True  # Log confluence calculations
RANGING_MARKET_LOG_ZONE_INTERACTIONS = True  # Log zone interaction details

# =============================================================================
# CONFIGURATION SUMMARY AND VALIDATION
# =============================================================================

def get_ranging_market_config_summary() -> dict:
    """Get a summary of ranging market configuration settings"""
    return {
        'strategy_enabled': RANGING_MARKET_STRATEGY,
        'squeeze_momentum_enabled': SQUEEZE_MOMENTUM_ENABLED,
        'wave_trend_enabled': WAVE_TREND_ENABLED,
        'bollinger_keltner_enabled': BOLLINGER_BANDS_ENABLED and KELTNER_CHANNELS_ENABLED,
        'rsi_divergence_enabled': RSI_ENABLED and RSI_DIVERGENCE_ENABLED,
        'rvi_enabled': RVI_ENABLED,
        'oscillator_confluence_enabled': OSCILLATOR_CONFLUENCE_ENABLED,
        'dynamic_zones_enabled': DYNAMIC_ZONES_ENABLED,
        'market_regime_detection_enabled': MARKET_REGIME_DETECTION_ENABLED,
        'mtf_analysis_enabled': MTF_ANALYSIS_ENABLED,
        'min_confidence_threshold': SIGNAL_QUALITY_MIN_CONFIDENCE,
        'confluence_threshold': OSCILLATOR_BULL_CONFLUENCE_THRESHOLD,
        'timeframes': MTF_TIMEFRAMES,
        'min_data_periods': RANGING_MARKET_MIN_DATA_PERIODS,
        'debug_logging': RANGING_MARKET_DEBUG_LOGGING
    }

def validate_ranging_market_config() -> bool:
    """Validate ranging market configuration for consistency"""
    validation_errors = []

    # Check oscillator threshold consistency
    if WTO_OVERBOUGHT_LEVEL <= WTO_OVERSOLD_LEVEL:
        validation_errors.append("WTO overbought threshold must be > oversold threshold")

    if RSI_OVERBOUGHT <= RSI_OVERSOLD:
        validation_errors.append("RSI overbought threshold must be > oversold threshold")

    # Check confluence weights
    total_weight = sum(OSCILLATOR_WEIGHTS.values())
    if abs(total_weight - 1.0) > 0.01:
        validation_errors.append(f"Oscillator weights must sum to 1.0, got {total_weight}")

    # Check Bollinger/Keltner settings
    if BB_KC_SQUEEZE_RATIO_THRESHOLD >= BB_KC_EXPANSION_RATIO_THRESHOLD:
        validation_errors.append("BB/KC squeeze ratio must be < expansion ratio")

    # Check timeframe consistency
    if MTF_ANALYSIS_ENABLED and not MTF_TIMEFRAMES:
        validation_errors.append("MTF analysis enabled but no timeframes specified")

    # Check squeeze settings
    if SQUEEZE_MIN_BARS_IN_SQUEEZE >= SQUEEZE_MAX_BARS_IN_SQUEEZE:
        validation_errors.append("Min squeeze bars must be < max squeeze bars")

    if validation_errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")

    return True

# Validate configuration on import
try:
    validate_ranging_market_config()
except ValueError as e:
    import logging
    logging.getLogger(__name__).error(f"Ranging market configuration validation failed: {e}")