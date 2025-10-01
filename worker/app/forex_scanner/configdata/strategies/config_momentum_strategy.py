# configdata/strategies/config_momentum_strategy.py
"""
Momentum Strategy Configuration - TradingView RAG Enhanced
=========================================================

Advanced momentum strategy inspired by TradingView scripts:
- AlgoAlpha AI Momentum Predictor
- Zeiierman Quantitative Momentum Oscillator
- ChartPrime Multi-Timeframe Oscillator

Designed for minimal lag with fast momentum detection and adaptive smoothing.
"""

# Strategy enable/disable
MOMENTUM_STRATEGY = True

# Main configuration dictionary with multiple presets optimized for different market conditions
MOMENTUM_STRATEGY_CONFIG = {
    'default': {
        'fast_period': 5,           # Keep original - Phase 2 testing showed 10 was too slow
        'slow_period': 10,          # Keep original - Phase 2 testing showed 24 was too slow
        'signal_period': 3,         # Keep original - faster response needed
        'velocity_period': 7,       # Keep original - 12 was too slow
        'volume_period': 14,        # Keep original - adequate baseline
        'description': 'Phase 1+2 hybrid - original periods with enhanced filters and pair-specific params',
        'best_for': ['trending', 'medium_volatility', 'breakouts'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium_to_strong',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'fast_period': 7,
        'slow_period': 14,
        'signal_period': 5,
        'velocity_period': 10,
        'volume_period': 20,
        'description': 'Conservative momentum with more smoothing',
        'best_for': ['strong_trends', 'low_volatility', 'ranging_markets'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 30.0
    },
    'aggressive': {
        'fast_period': 3,
        'slow_period': 7,
        'signal_period': 2,
        'velocity_period': 5,
        'volume_period': 10,
        'description': 'Ultra-fast momentum for scalping and quick moves',
        'best_for': ['breakouts', 'high_volatility', 'news_events'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 100.0
    },
    'scalping': {
        'fast_period': 2,
        'slow_period': 5,
        'signal_period': 2,
        'velocity_period': 3,
        'volume_period': 8,
        'description': 'Minimal lag scalping configuration',
        'best_for': ['ranging_markets', 'high_frequency', 'tight_spreads']
    },
    'swing': {
        'fast_period': 10,
        'slow_period': 21,
        'signal_period': 7,
        'velocity_period': 14,
        'volume_period': 30,
        'description': 'Swing trading with trend confirmation',
        'best_for': ['strong_trends', 'position_trading', 'daily_timeframes']
    },
    'news_safe': {
        'fast_period': 8,
        'slow_period': 16,
        'signal_period': 4,
        'velocity_period': 9,
        'volume_period': 18,
        'description': 'Safer configuration during high impact news',
        'best_for': ['news_events', 'high_volatility', 'uncertainty']
    },
    'crypto': {
        'fast_period': 4,
        'slow_period': 9,
        'signal_period': 3,
        'velocity_period': 6,
        'volume_period': 12,
        'description': 'Optimized for crypto-like high volatility markets',
        'best_for': ['high_volatility', 'breakouts', '24_7_markets']
    }
}

# Active configuration selector
ACTIVE_MOMENTUM_CONFIG = 'default'

# Individual feature toggles
MOMENTUM_FEATURE_ENABLED = True
MOMENTUM_VELOCITY_ENABLED = True
MOMENTUM_VOLUME_CONFIRMATION = True
MOMENTUM_MTF_VALIDATION = True

# Strategy weight (for combined strategy integration)
STRATEGY_WEIGHT_MOMENTUM = 0.3

# Dynamic configuration management
ENABLE_DYNAMIC_MOMENTUM_CONFIG = True
MOMENTUM_ADAPTIVE_SMOOTHING = True

# Momentum calculation method
MOMENTUM_CALCULATION_METHOD = 'velocity_weighted'  # Options: 'simple', 'velocity_weighted', 'volume_weighted'

# Threshold settings - Phase 1 Optimization
# Increased from 0.0002 to reduce false signals and improve quality
MOMENTUM_SIGNAL_THRESHOLD = 0.0003   # 3 basis points - higher quality threshold
MOMENTUM_DIVERGENCE_THRESHOLD = 0.0002  # 2 basis points - minimum momentum strength
MOMENTUM_VELOCITY_THRESHOLD = 0.008    # Reduced from 0.020 - was too restrictive

# Volume confirmation settings
MOMENTUM_VOLUME_MULTIPLIER = 1.5  # Increased from 1.2 for stronger confirmation
MOMENTUM_VOLUME_CONFIRMATION_ENABLED = True

# Multi-timeframe settings
MOMENTUM_MTF_TIMEFRAMES = ['5m', '15m', '1h']
MOMENTUM_MTF_MIN_AGREEMENT = 0.6  # 60% agreement required

# Risk management
MOMENTUM_MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
MOMENTUM_STOP_LOSS_METHOD = 'adaptive'  # Options: 'fixed', 'atr', 'adaptive'
MOMENTUM_TAKE_PROFIT_RATIO = 2.0  # 2:1 reward:risk ratio

# Signal validation
# Phase 1 Optimization: Reduced from 93% to allow more statistically valid signals
# 75% provides optimal balance between quality and quantity
MOMENTUM_MIN_CONFIDENCE = 0.75  # 75% confidence - statistically validated threshold
MOMENTUM_SIGNAL_COOLDOWN = 900   # 15 minutes cooldown between signals (TODO: implement)

# Confirmation requirements - Phase 1: Strengthened from 2-of-4 to 3-of-4
MOMENTUM_MIN_CONFIRMATIONS = 3  # Require 3 out of 4 confirmations for higher quality

# Risk Management - Phase 1: ATR-based stop loss and profit targets
MOMENTUM_STOP_LOSS_ATR_MULTIPLIER = 2.5   # Wider stops to avoid premature stop-outs
MOMENTUM_TAKE_PROFIT_ATR_MULTIPLIER = 2.0  # More realistic targets (conservative approach)
# Alternative aggressive setting: MOMENTUM_TAKE_PROFIT_ATR_MULTIPLIER = 6.0

# TESTING: Static stop loss override (set to 40 pips for testing)
MOMENTUM_USE_STATIC_STOPS = True          # Override ATR-based stops
MOMENTUM_STATIC_STOP_PIPS = 40.0          # Fixed 40 pip stop loss
MOMENTUM_STATIC_TARGET_PIPS = 80.0        # Fixed 80 pip target (2:1 R:R)

# Trend Alignment - Phase 1: NEW
MOMENTUM_REQUIRE_TREND_ALIGNMENT = True   # Only trade in direction of trend
MOMENTUM_TREND_EMA_PERIOD = 50            # EMA period for trend determination
MOMENTUM_ALLOW_COUNTER_TREND = False      # Disable counter-trend trades

# Market Regime Filter - Phase 1: NEW
MOMENTUM_ENABLE_REGIME_FILTER = True      # Filter out unfavorable market conditions
MOMENTUM_MIN_ADX = 22                     # Minimum trend strength (ADX)
MOMENTUM_MIN_ATR_RATIO = 0.8              # Current ATR must be > 0.8x 20-bar average
MOMENTUM_MIN_EMA_SEPARATION = 0.3         # Price distance from EMA (in ATR units)

# Phase 2: Pair-Specific Parameters
MOMENTUM_PAIR_SPECIFIC_PARAMS = {
    'EURUSD': {
        'signal_threshold': 0.0003,
        'velocity_threshold': 0.008,
        'stop_atr_multiplier': 2.5,
        'target_atr_multiplier': 2.0,
        'description': 'EUR/USD optimized - most liquid pair'
    },
    'GBPUSD': {
        'signal_threshold': 0.0003,
        'velocity_threshold': 0.010,  # Higher volatility
        'stop_atr_multiplier': 2.8,   # Wider stops for GBP volatility
        'target_atr_multiplier': 2.2,
        'description': 'GBP/USD optimized - high volatility pair'
    },
    'USDJPY': {
        'signal_threshold': 0.0015,   # Different scale (100+ vs 1.1)
        'velocity_threshold': 0.040,  # Adjusted for JPY scale
        'stop_atr_multiplier': 2.5,
        'target_atr_multiplier': 2.0,
        'description': 'USD/JPY optimized - different price scale'
    },
    'AUDUSD': {
        'signal_threshold': 0.0003,
        'velocity_threshold': 0.008,
        'stop_atr_multiplier': 2.6,
        'target_atr_multiplier': 2.0,
        'description': 'AUD/USD optimized'
    },
    'NZDUSD': {
        'signal_threshold': 0.0003,
        'velocity_threshold': 0.008,
        'stop_atr_multiplier': 2.6,
        'target_atr_multiplier': 2.0,
        'description': 'NZD/USD optimized'
    },
    'EURJPY': {
        'signal_threshold': 0.0015,   # JPY scale adjustment
        'velocity_threshold': 0.040,
        'stop_atr_multiplier': 2.7,
        'target_atr_multiplier': 2.1,
        'description': 'EUR/JPY optimized - cross pair'
    },
    'AUDJPY': {
        'signal_threshold': 0.0015,   # JPY scale adjustment
        'velocity_threshold': 0.040,
        'stop_atr_multiplier': 2.7,
        'target_atr_multiplier': 2.1,
        'description': 'AUD/JPY optimized - cross pair'
    }
}

# Phase 2: Structure-Based Stop Placement
MOMENTUM_USE_STRUCTURE_STOPS = True       # Place stops beyond recent swing points
MOMENTUM_STRUCTURE_LOOKBACK_BARS = 20     # Look back 20 bars for swing highs/lows
MOMENTUM_MIN_STOP_DISTANCE_PIPS = 8.0     # Minimum stop distance
MOMENTUM_MAX_STOP_DISTANCE_PIPS = 25.0    # Maximum stop distance (cap)
MOMENTUM_STRUCTURE_BUFFER_PIPS = 2.0      # Buffer beyond swing point

def get_momentum_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get momentum configuration for specific epic with fallbacks"""
    if ENABLE_DYNAMIC_MOMENTUM_CONFIG and market_condition in MOMENTUM_STRATEGY_CONFIG:
        config_name = market_condition
    else:
        config_name = ACTIVE_MOMENTUM_CONFIG

    return MOMENTUM_STRATEGY_CONFIG.get(config_name, MOMENTUM_STRATEGY_CONFIG['default'])

def get_pair_name_from_epic(epic: str) -> str:
    """Extract clean pair name from epic"""
    if 'CS.D.' in epic:
        pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
    else:
        pair = epic
    return pair

def get_momentum_pair_config(epic: str) -> dict:
    """Phase 2: Get pair-specific configuration parameters"""
    pair = get_pair_name_from_epic(epic)

    # Return pair-specific config if available
    if pair in MOMENTUM_PAIR_SPECIFIC_PARAMS:
        return MOMENTUM_PAIR_SPECIFIC_PARAMS[pair]

    # Fallback to defaults
    return {
        'signal_threshold': MOMENTUM_SIGNAL_THRESHOLD,
        'velocity_threshold': MOMENTUM_VELOCITY_THRESHOLD,
        'stop_atr_multiplier': MOMENTUM_STOP_LOSS_ATR_MULTIPLIER,
        'target_atr_multiplier': MOMENTUM_TAKE_PROFIT_ATR_MULTIPLIER,
        'description': 'Default parameters'
    }

def get_momentum_threshold_for_epic(epic: str) -> float:
    """Get momentum-specific thresholds based on currency pair (Phase 2: uses pair config)"""
    pair_config = get_momentum_pair_config(epic)
    return pair_config.get('signal_threshold', MOMENTUM_SIGNAL_THRESHOLD)

def get_momentum_velocity_threshold_for_epic(epic: str) -> float:
    """Get velocity threshold based on pair characteristics (Phase 2: uses pair config)"""
    pair_config = get_momentum_pair_config(epic)
    return pair_config.get('velocity_threshold', MOMENTUM_VELOCITY_THRESHOLD)

def get_adaptive_smoothing_factor(volatility: float) -> float:
    """Calculate adaptive smoothing factor based on market volatility"""
    try:
        # Higher volatility = less smoothing (more responsive)
        # Lower volatility = more smoothing (less noise)
        if volatility > 0.02:  # High volatility (2%+)
            return 0.8  # Minimal smoothing
        elif volatility > 0.01:  # Medium volatility (1-2%)
            return 0.6  # Moderate smoothing
        else:  # Low volatility (<1%)
            return 0.4  # More smoothing
    except Exception:
        return 0.6  # Default moderate smoothing

# Validation function (MANDATORY)
def validate_momentum_config() -> dict:
    """Validate momentum strategy configuration completeness"""
    try:
        required_keys = ['MOMENTUM_STRATEGY', 'MOMENTUM_STRATEGY_CONFIG']
        for key in required_keys:
            if not globals().get(key):
                return {'valid': False, 'error': f'Missing {key}'}

        # Validate configuration structure
        for config_name, config in MOMENTUM_STRATEGY_CONFIG.items():
            if not isinstance(config, dict):
                return {'valid': False, 'error': f'Config {config_name} must be dict'}
            if 'description' not in config:
                return {'valid': False, 'error': f'Config {config_name} missing description'}

            # Validate momentum-specific parameters
            required_params = ['fast_period', 'slow_period', 'signal_period']
            for param in required_params:
                if param not in config:
                    return {'valid': False, 'error': f'Config {config_name} missing {param}'}
                if not isinstance(config[param], int) or config[param] <= 0:
                    return {'valid': False, 'error': f'Config {config_name} {param} must be positive integer'}

            # Validate period order
            if not (config['fast_period'] < config['slow_period']):
                return {'valid': False, 'error': f'Config {config_name}: fast_period must be < slow_period'}

        return {
            'valid': True,
            'config_count': len(MOMENTUM_STRATEGY_CONFIG),
            'presets': list(MOMENTUM_STRATEGY_CONFIG.keys()),
            'active_config': ACTIVE_MOMENTUM_CONFIG,
            'features': {
                'velocity_enabled': MOMENTUM_VELOCITY_ENABLED,
                'volume_confirmation': MOMENTUM_VOLUME_CONFIRMATION,
                'mtf_validation': MOMENTUM_MTF_VALIDATION,
                'adaptive_smoothing': MOMENTUM_ADAPTIVE_SMOOTHING
            }
        }

    except Exception as e:
        return {'valid': False, 'error': f'Validation exception: {str(e)}'}

# Momentum calculation methods
def calculate_velocity_momentum(prices, period: int = 7):
    """Calculate velocity-based momentum (rate of change with smoothing)"""
    import pandas as pd
    import numpy as np

    # Price velocity (rate of change)
    velocity = prices.pct_change(period)

    # Apply exponential smoothing to reduce noise
    smoothed_velocity = velocity.ewm(span=3).mean()

    return smoothed_velocity

def calculate_volume_weighted_momentum(prices, volumes, period: int = 14):
    """Calculate volume-weighted momentum for institutional flow detection"""
    import pandas as pd
    import numpy as np

    # Price change weighted by volume
    price_change = prices.pct_change()
    volume_weighted_change = price_change * volumes

    # Rolling sum for momentum
    momentum = volume_weighted_change.rolling(window=period).sum()

    return momentum

def calculate_adaptive_momentum(prices, volatility, fast_period: int = 5, slow_period: int = 10):
    """Calculate adaptive momentum based on market volatility"""
    import pandas as pd
    import numpy as np

    # Get adaptive smoothing factor
    smoothing_factor = volatility.apply(get_adaptive_smoothing_factor)

    # Calculate fast and slow momentum
    fast_momentum = prices.pct_change(fast_period)
    slow_momentum = prices.pct_change(slow_period)

    # Apply adaptive smoothing
    adaptive_fast = fast_momentum.ewm(alpha=smoothing_factor).mean()
    adaptive_slow = slow_momentum.ewm(alpha=smoothing_factor * 0.7).mean()

    return adaptive_fast - adaptive_slow