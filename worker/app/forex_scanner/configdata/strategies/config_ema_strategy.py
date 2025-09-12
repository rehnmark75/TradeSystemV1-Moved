# configdata/strategies/config_ema_strategy.py
"""
EMA Strategy Configuration
Configuration module for EMA strategy settings including dynamic configurations,
distance validation, and integrated momentum indicators
"""

# =============================================================================
# EMA STRATEGY CORE CONFIGURATION SETTINGS  
# =============================================================================

# Core Strategy Settings
SIMPLE_EMA_STRATEGY = True                # True = basic EMA rules only, False = full filters
STRICT_EMA_ALIGNMENT = True               # Not used in simple mode
REQUIRE_EMA_SEPARATION = False            # Disabled in simple mode
STRATEGY_WEIGHT_EMA = 0.5                 # EMA strategy weight
STRATEGY_WEIGHT_ZERO_LAG = 0.0            # Zero Lag EMA weight (if enabled)

# =============================================================================
# DYNAMIC EMA CONFIGURATION SYSTEM
# =============================================================================

# Enable/disable dynamic EMA configuration
ENABLE_DYNAMIC_EMA_CONFIG = False         # Disabled - use stable 9/21/200 EMAs

# Enhanced Dynamic EMA Strategy Configurations with market conditions
EMA_STRATEGY_CONFIG = {
    'default': {
        'short': 21, 'long': 50, 'trend': 200,
        'description': 'Less choppy configuration for cleaner signals - 21/50/200 EMAs',
        'best_for': ['trending', 'medium_volatility'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'short': 20, 'long': 50, 'trend': 200,
        'description': 'Conservative approach for low volatility trending markets',
        'best_for': ['strong_trends', 'low_volatility'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 25.0
    },
    'aggressive': {
        'short': 5, 'long': 13, 'trend': 50,
        'description': 'Fast-reacting configuration for high volatility breakouts',
        'best_for': ['breakouts', 'high_volatility'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'weak',
        'best_market_regime': 'breakout',
        'best_session': ['overlap', 'new_york'],
        'preferred_pairs': ['CS.D.GBPUSD.MINI.IP', 'CS.D.EURUSD.MINI.IP'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 100.0
    },
    'scalping': {
        'short': 3, 'long': 8, 'trend': 21,
        'description': 'Ultra-fast configuration for scalping strategies',
        'best_for': ['ranging_markets', 'high_frequency'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'weak',
        'best_market_regime': 'ranging',
        'best_session': ['london', 'new_york', 'overlap'],
        'preferred_pairs': ['CS.D.EURUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 80.0
    },
    'swing': {
        'short': 25, 'long': 55, 'trend': 200,
        'description': 'Slow and steady for swing trading',
        'best_for': ['strong_trends', 'position_trading'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 30.0
    },
    'news_safe': {
        'short': 15, 'long': 30, 'trend': 200,
        'description': 'Safer configuration during news events',
        'best_for': ['news_events', 'high_volatility'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'breakout',
        'best_session': ['asian', 'overlap'],
        'preferred_pairs': ['CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 100.0
    },
    'crypto': {
        'short': 7, 'long': 25, 'trend': 99,
        'description': 'Adapted for crypto-like high volatility markets',
        'best_for': ['high_volatility', 'breakouts'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'medium',
        'best_market_regime': 'breakout',
        'best_session': ['overlap', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.MINI.IP'],  # Most liquid
        'min_pip_volatility': 25.0,
        'max_pip_volatility': 150.0
    }
}

# Active configuration when not using dynamic mode
ACTIVE_EMA_CONFIG = 'default'

# Scanner and backtest dynamic EMA usage
SCANNER_USE_DYNAMIC_EMA = True            # Use dynamic EMA in scanner
BACKTEST_USE_DYNAMIC_EMA = False          # Use dynamic EMA in backtesting (experimental)

# =============================================================================
# EMA200 DISTANCE VALIDATION
# =============================================================================

# Enable minimum distance from EMA200 requirement
EMA200_MIN_DISTANCE_ENABLED = True

# Minimum distance from EMA200 in pips for each pair
# These values prevent false signals when price hovers near EMA200
EMA200_MIN_DISTANCE_PIPS = {
    'EURUSD': 4.0,    # EUR/USD - most liquid, 5 pips minimum
    'GBPUSD': 7.0,    # GBP/USD - more volatile, 8 pips minimum
    'AUDUSD': 6.0,    # AUD/USD - commodity currency, 6 pips
    'NZDUSD': 7.0,    # NZD/USD - higher volatility, 7 pips
    'USDCAD': 6.0,    # USD/CAD - oil correlation, 6 pips
    'USDCHF': 5.0,    # USD/CHF - safe haven, 5 pips
    'USDJPY': 5.0,    # USD/JPY - different scale, 50 pips (0.50)
    'EURJPY': 5.0,    # EUR/JPY - cross pair, 60 pips
    'GBPJPY': 5.0,    # GBP/JPY - very volatile, 80 pips
    'AUDJPY': 5.0,    # AUD/JPY - carry trade pair, 70 pips
    'EURGBP': 5.0,    # EUR/GBP - relatively stable, 5 pips
    'EURAUD': 8.0,    # EUR/AUD - commodity impact, 8 pips
    'GBPAUD': 10.0,   # GBP/AUD - volatile cross, 10 pips
    'DEFAULT': 5.0    # Default for unknown pairs
}

# Grace period: Allow signals slightly closer than minimum during strong momentum
# Set to 0.8 means 80% of minimum distance is acceptable with strong momentum
EMA200_DISTANCE_MOMENTUM_MULTIPLIER = 0.8

# =============================================================================
# TWO-POLE OSCILLATOR (EMA INTEGRATION)
# =============================================================================

# Two-Pole Oscillator Configuration (momentum confirmation for EMA signals)
TWO_POLE_OSCILLATOR_ENABLED = True       # Enable Two-Pole Oscillator validation
TWO_POLE_FILTER_LENGTH = 20              # Filter length (default: 20)
TWO_POLE_SMA_LENGTH = 25                 # SMA length for normalization (default: 25)  
TWO_POLE_SIGNAL_DELAY = 4                # Signal delay in bars (default: 4)
TWO_POLE_VALIDATION_ENABLED = True       # Use oscillator for signal validation
TWO_POLE_MIN_STRENGTH = 0.1              # Minimum oscillator strength for signals
TWO_POLE_ZONE_FILTER_ENABLED = True      # Require proper zone for signals (oversold/overbought)

# Two-Pole Oscillator Confidence Scoring
TWO_POLE_CONFIDENCE_WEIGHT = 0.2         # Weight in overall confidence calculation
TWO_POLE_ZONE_BONUS = 0.1                # Confidence bonus for proper zone alignment
TWO_POLE_STRENGTH_MULTIPLIER = 0.5       # Multiplier for oscillator strength

# Two-Pole Multi-Timeframe Validation
TWO_POLE_MTF_VALIDATION = True           # Enable 1H timeframe Two-Pole validation
TWO_POLE_MTF_TIMEFRAME = '1h'            # Higher timeframe for confirmation (1h, 4h, etc.)

# =============================================================================
# MOMENTUM BIAS INDEX (EMA INTEGRATION)
# =============================================================================

# Momentum Bias Index Configuration (momentum confirmation for EMA signals)
MOMENTUM_BIAS_ENABLED = False                   # Enable Momentum Bias Index validation
MOMENTUM_BIAS_LENGTH = 10                      # Momentum calculation length
MOMENTUM_BIAS_BIAS_LENGTH = 5                  # Bias calculation length
MOMENTUM_BIAS_SMOOTH_LENGTH = 10               # Smoothing length
MOMENTUM_BIAS_BOUNDARY_LENGTH = 30             # Impulse boundary length
MOMENTUM_BIAS_STD_MULTIPLIER = 3.0             # Standard deviation multiplier for boundary
MOMENTUM_BIAS_SMOOTH = True                    # Enable smoothing (HMA approximation)

# Momentum Bias Validation Settings
MOMENTUM_BIAS_REQUIRE_ABOVE_BOUNDARY = False   # Require bias above boundary for signals
MOMENTUM_BIAS_VALIDATION_ENABLED = True        # Use bias for signal validation
MOMENTUM_BIAS_MIN_STRENGTH = 0.1               # Minimum bias strength for signals

# Momentum Bias Confidence Scoring
MOMENTUM_BIAS_CONFIDENCE_WEIGHT = 0.15         # Weight in overall confidence calculation
MOMENTUM_BIAS_BOUNDARY_BONUS = 0.1             # Confidence bonus for above boundary signals
MOMENTUM_BIAS_STRENGTH_MULTIPLIER = 0.3        # Multiplier for bias strength

# =============================================================================
# MACD MOMENTUM FILTER (EMA INTEGRATION)
# =============================================================================

# MACD Momentum Filter Configuration (momentum confirmation for EMA signals)
MACD_MOMENTUM_FILTER_ENABLED = True            # Enable MACD histogram momentum validation
MACD_HISTOGRAM_LOOKBACK = 1                   # Number of periods to analyze for histogram trend
MACD_MIN_SLOPE_THRESHOLD = 0.00001              # Minimum slope threshold to avoid noise

# MACD Momentum Filter Rules:
# - BULL signals rejected if MACD histogram is descending (momentum against signal)  
# - BEAR signals rejected if MACD histogram is rising (momentum against signal)
# - NEUTRAL trends allow all signals

# MACD Momentum Filter Settings - RELAXED FOR BETTER SIGNAL GENERATION
MACD_MOMENTUM_VALIDATION_ENABLED = False        # Use MACD for signal validation
MACD_TREND_SENSITIVITY = 'ultra_sensitive'     # Sensitivity: 'strict', 'normal', 'permissive', 'neutral_bias'
MACD_VALIDATION_MODE = 'strict_blocking'       # Mode: 'strict_blocking', 'slope_aware', 'neutral_friendly'

# MACD trend sensitivity mappings (affects lookback and threshold)
MACD_SENSITIVITY_SETTINGS = {
    'strict': {
        'lookback': 5,                          # More periods for stable trend detection
        'min_slope': 0.0002,                    # Higher threshold, less sensitive
        'description': 'Conservative - requires clear momentum direction'
    },
    'normal': {
        'lookback': 3,                          # Standard periods for trend detection  
        'min_slope': 0.0001,                    # Standard threshold
        'description': 'Balanced - moderate sensitivity to momentum changes'
    },
    'permissive': {
        'lookback': 2,                          # Fewer periods, more reactive
        'min_slope': 0.00005,                   # Lower threshold, more sensitive
        'description': 'Permissive - allows signals with weak MACD trends'
    },
    'neutral_bias': {
        'lookback': 3,                          # Standard periods
        'min_slope': 0.00002,                   # Very low threshold - most trends become neutral
        'description': 'Neutral bias - most MACD states allow signals'
    },
    'ultra_sensitive': {
        'lookback': 2,                          # Very reactive
        'min_slope': 0.000005,                  # Ultra-low threshold - detect tiny movements  
        'description': 'Ultra sensitive - blocks even tiny opposing MACD movements'
    }
}

# MACD validation modes determine how strictly MACD momentum is enforced
MACD_VALIDATION_MODES = {
    'strict_blocking': {
        'description': 'Block signals when MACD momentum opposes signal direction',
        'allow_neutral': True,                  # NEUTRAL MACD allows signals
        'block_opposite': True                  # Block when MACD opposes signal
    },
    'slope_aware': {
        'description': 'Consider MACD slope but allow weak opposing momentum',
        'allow_neutral': True,                  # NEUTRAL MACD allows signals  
        'block_opposite': 'strong_only',        # Only block strong opposing momentum
        'weak_threshold': 0.0002                # Threshold for "strong" opposing momentum
    },
    'neutral_friendly': {
        'description': 'Only block very strong opposing MACD momentum',
        'allow_neutral': True,                  # NEUTRAL MACD allows signals
        'block_opposite': 'very_strong_only',   # Only block very strong opposing momentum  
        'strong_threshold': 0.0005              # Threshold for "very strong" opposing momentum
    }
}

# MACD Filter Debug and Logging
MACD_MOMENTUM_DEBUG_LOGGING = True             # Enable detailed MACD momentum logging

# =============================================================================
# EMA SAFETY FILTER SETTINGS
# =============================================================================

# EMA-specific safety filter settings (used in SAFETY_FILTER_PRESETS)
ENABLE_EMA200_CONTRADICTION_FILTER = True     # Enable EMA200 trend contradiction filter
ENABLE_EMA_STACK_CONTRADICTION_FILTER = True  # Enable EMA stack contradiction filter
EMA200_MINIMUM_MARGIN = 0.002                 # Minimum margin from EMA200 (0.2%)

# =============================================================================
# ADDITIONAL EMA STRATEGY SETTINGS
# =============================================================================

# Standard EMA periods for strategy
EMA_PERIODS = [21, 50, 200]                   # Standard EMA periods used in strategy

# EMA validation settings
REQUIRE_PRICE_EMA200_CONFIRMATION = True      # Require price above/below EMA 200
STRICT_EMA_ALIGNMENT = False                  # Allow more flexible EMA requirements  
REQUIRE_EMA_SEPARATION = False                # Enable/disable EMA separation checks

# Minimum EMA separation in pips (if REQUIRE_EMA_SEPARATION is enabled)
MIN_EMA_SEPARATION_PIPS = {
    'EURUSD': 3.0,    # EUR/USD minimum separation
    'GBPUSD': 5.0,    # GBP/USD minimum separation
    'AUDUSD': 4.0,    # AUD/USD minimum separation  
    'NZDUSD': 5.0,    # NZD/USD minimum separation
    'USDCAD': 4.0,    # USD/CAD minimum separation
    'USDCHF': 3.0,    # USD/CHF minimum separation
    'USDJPY': 3.0,    # USD/JPY minimum separation (30 pips)
    'EURJPY': 4.0,    # EUR/JPY minimum separation (40 pips)
    'GBPJPY': 6.0,    # GBP/JPY minimum separation (60 pips)
    'AUDJPY': 5.0,    # AUD/JPY minimum separation (50 pips)
    'EURGBP': 3.0,    # EUR/GBP minimum separation
    'DEFAULT': 4.0    # Default for unknown pairs
}

# =============================================================================
# EMA STRATEGY INTEGRATION SETTINGS
# =============================================================================

# Strategy Integration Settings
EMA_STRATEGY_WEIGHT = 0.15                    # Weight in combined strategy mode
EMA_ALLOW_COMBINED = True                     # Allow in combined strategies
EMA_PRIORITY_LEVEL = 1                        # Priority level (1=highest, 5=lowest)

# Performance Settings
EMA_ENABLE_BACKTESTING = True                 # Enable strategy in backtests
EMA_MIN_DATA_PERIODS = 250                    # Minimum data periods required (200 + buffer)
EMA_ENABLE_PERFORMANCE_TRACKING = True        # Track strategy performance

# Debug Settings
EMA_DEBUG_LOGGING = True                      # Enable detailed debug logging

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ema_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """
    Get EMA configuration for specific epic and market condition
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
        market_condition: Market condition ('default', 'conservative', 'aggressive', etc.)
        
    Returns:
        Dictionary with EMA configuration
    """
    # Use dynamic config if enabled, otherwise use active config
    if ENABLE_DYNAMIC_EMA_CONFIG and market_condition in EMA_STRATEGY_CONFIG:
        config_name = market_condition
    else:
        config_name = ACTIVE_EMA_CONFIG
        
    return EMA_STRATEGY_CONFIG.get(config_name, EMA_STRATEGY_CONFIG['default'])

def get_ema_distance_for_epic(epic: str) -> float:
    """
    Get minimum EMA200 distance for specific epic
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
        
    Returns:
        Minimum distance in pips
    """
    # Extract currency pair from epic (e.g., 'EURUSD' from 'CS.D.EURUSD.MINI.IP')
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic
            
        return EMA200_MIN_DISTANCE_PIPS.get(pair, EMA200_MIN_DISTANCE_PIPS['DEFAULT'])
    except Exception:
        return EMA200_MIN_DISTANCE_PIPS['DEFAULT']

def get_ema_separation_for_epic(epic: str) -> float:
    """
    Get minimum EMA separation distance for specific epic
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
        
    Returns:
        Minimum separation in pips
    """
    # Extract currency pair from epic
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic
            
        return MIN_EMA_SEPARATION_PIPS.get(pair, MIN_EMA_SEPARATION_PIPS['DEFAULT'])
    except Exception:
        return MIN_EMA_SEPARATION_PIPS['DEFAULT']

# Configuration summary function
def get_ema_config_summary() -> dict:
    """Get a summary of EMA configuration settings"""
    return {
        'strategy_enabled': SIMPLE_EMA_STRATEGY,
        'dynamic_config_enabled': ENABLE_DYNAMIC_EMA_CONFIG,
        'active_config': ACTIVE_EMA_CONFIG,
        'ema200_distance_enabled': EMA200_MIN_DISTANCE_ENABLED,
        'two_pole_enabled': TWO_POLE_OSCILLATOR_ENABLED,
        'momentum_bias_enabled': MOMENTUM_BIAS_ENABLED,
        'macd_momentum_filter_enabled': MACD_MOMENTUM_FILTER_ENABLED,
        'macd_trend_sensitivity': MACD_TREND_SENSITIVITY,
        'macd_validation_mode': MACD_VALIDATION_MODE,
        'ema_periods': EMA_PERIODS,
        'min_data_periods': EMA_MIN_DATA_PERIODS,
        'debug_logging': EMA_DEBUG_LOGGING,
        'total_configurations': len(EMA_STRATEGY_CONFIG)
    }

# Quick configuration presets for different signal generation frequencies
def set_macd_filter_preset(preset: str = 'balanced'):
    """
    Quick presets for MACD filter configuration
    
    Args:
        preset: 'strict', 'balanced', 'permissive', 'minimal'
    """
    presets = {
        'strict': {
            'sensitivity': 'strict',
            'mode': 'strict_blocking',
            'description': 'Block all opposing momentum - fewest signals, highest quality'
        },
        'balanced': {
            'sensitivity': 'normal', 
            'mode': 'slope_aware',
            'description': 'Block strong opposing momentum - balanced signal frequency'
        },
        'permissive': {
            'sensitivity': 'permissive',
            'mode': 'slope_aware', 
            'description': 'Allow weak opposing momentum - more signals'
        },
        'minimal': {
            'sensitivity': 'neutral_bias',
            'mode': 'neutral_friendly',
            'description': 'Only block very strong opposing momentum - maximum signals'
        }
    }
    
    return presets.get(preset, presets['balanced'])