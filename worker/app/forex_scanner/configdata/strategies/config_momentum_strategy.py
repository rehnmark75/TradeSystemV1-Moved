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
        'fast_period': 5,           # Fast momentum period (minimal lag)
        'slow_period': 10,          # Slow momentum period
        'signal_period': 3,         # Signal line smoothing
        'velocity_period': 7,       # Price velocity calculation
        'volume_period': 14,        # Volume momentum period
        'description': 'Balanced momentum configuration with minimal lag',
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

# Threshold settings - Optimized for 20-50 signals per week
# Final calibration: 0.0002 captures typical momentum while maintaining quality
MOMENTUM_SIGNAL_THRESHOLD = 0.0002   # 2 basis points - balanced threshold
MOMENTUM_DIVERGENCE_THRESHOLD = 0.0001  # 1 basis point divergence
MOMENTUM_VELOCITY_THRESHOLD = 0.020    # ATR-normalized velocity threshold

# Volume confirmation settings
MOMENTUM_VOLUME_MULTIPLIER = 1.2  # Volume must be 1.2x average
MOMENTUM_VOLUME_CONFIRMATION_ENABLED = True

# Multi-timeframe settings
MOMENTUM_MTF_TIMEFRAMES = ['5m', '15m', '1h']
MOMENTUM_MTF_MIN_AGREEMENT = 0.6  # 60% agreement required

# Risk management
MOMENTUM_MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
MOMENTUM_STOP_LOSS_METHOD = 'adaptive'  # Options: 'fixed', 'atr', 'adaptive'
MOMENTUM_TAKE_PROFIT_RATIO = 2.0  # 2:1 reward:risk ratio

# Signal validation
# Using momentum-specific confidence calculation (validator disabled)
# Raised to 93% to reduce signal frequency (especially for trending pairs like USDCAD)
MOMENTUM_MIN_CONFIDENCE = 0.93  # 93% confidence - very high quality signals only
MOMENTUM_SIGNAL_COOLDOWN = 900   # 15 minutes cooldown between signals (TODO: implement)

def get_momentum_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get momentum configuration for specific epic with fallbacks"""
    if ENABLE_DYNAMIC_MOMENTUM_CONFIG and market_condition in MOMENTUM_STRATEGY_CONFIG:
        config_name = market_condition
    else:
        config_name = ACTIVE_MOMENTUM_CONFIG

    return MOMENTUM_STRATEGY_CONFIG.get(config_name, MOMENTUM_STRATEGY_CONFIG['default'])

def get_momentum_threshold_for_epic(epic: str) -> float:
    """Get momentum-specific thresholds based on currency pair"""
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic

        # JPY pairs typically require different thresholds due to different decimal places
        if 'JPY' in pair:
            return 0.001  # Higher threshold for JPY pairs (different scale)
        elif any(minor in pair for minor in ['EUR', 'GBP', 'AUD', 'NZD', 'CAD']):
            return MOMENTUM_SIGNAL_THRESHOLD  # Standard threshold for majors
        else:
            return MOMENTUM_SIGNAL_THRESHOLD * 1.5  # Higher threshold for exotics
    except Exception:
        return MOMENTUM_SIGNAL_THRESHOLD  # Default fallback

def get_momentum_velocity_threshold_for_epic(epic: str) -> float:
    """Get velocity threshold based on pair characteristics"""
    try:
        if 'JPY' in epic:
            return MOMENTUM_VELOCITY_THRESHOLD * 50  # Adjust for JPY scale
        else:
            return MOMENTUM_VELOCITY_THRESHOLD
    except Exception:
        return MOMENTUM_VELOCITY_THRESHOLD

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