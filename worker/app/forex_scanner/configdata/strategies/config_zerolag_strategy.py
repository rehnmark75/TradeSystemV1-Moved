# configdata/strategies/config_zerolag_strategy.py
"""
Zero-Lag Strategy Configuration
Modular configuration following the new architectural pattern
"""

# Strategy enable/disable
ZERO_LAG_STRATEGY = True

# Main configuration dictionary with multiple presets
ZERO_LAG_STRATEGY_CONFIG = {
    'default': {
        'zl_length': 50,
        'band_multiplier': 1.5, 
        'min_confidence': 0.65,
        'bb_length': 20,
        'bb_mult': 2.0,
        'kc_length': 20,
        'kc_mult': 1.5,
        'description': 'Balanced zero-lag configuration for most market conditions',
        'best_for': ['trending', 'medium_volatility'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'zl_length': 89,
        'band_multiplier': 2.5,
        'min_confidence': 0.70,
        'bb_length': 25,
        'bb_mult': 2.5,
        'kc_length': 25,
        'kc_mult': 2.0,
        'description': 'Conservative setup with longer periods and higher confidence',
        'best_for': ['ranging', 'high_volatility', 'news_events'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'weak',
        'best_market_regime': 'ranging',
        'best_session': ['asian', 'london_open'],
        'preferred_pairs': ['CS.D.EURJPY.MINI.IP', 'CS.D.GBPJPY.MINI.IP', 'CS.D.EURGBP.MINI.IP'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 100.0
    },
    'aggressive': {
        'zl_length': 21,
        'band_multiplier': 1.0,
        'min_confidence': 0.55,
        'bb_length': 15,
        'bb_mult': 1.5,
        'kc_length': 15,
        'kc_mult': 1.0,
        'description': 'Fast-reacting setup for strong trending markets',
        'best_for': ['strong_trending', 'breakouts', 'momentum'],
        'best_volatility_regime': 'low_to_medium',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london_ny_overlap'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 30.0
    },
    'scalping': {
        'zl_length': 12,
        'band_multiplier': 0.8,
        'min_confidence': 0.50,
        'bb_length': 10,
        'bb_mult': 1.5,
        'kc_length': 10,
        'kc_mult': 1.0,
        'description': 'Ultra-fast scalping setup for 5m timeframes',
        'best_for': ['scalping', 'high_frequency', 'tight_spreads'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'any',
        'best_market_regime': 'any',
        'best_session': ['london_ny_overlap'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 15.0
    },
    'swing': {
        'zl_length': 144,
        'band_multiplier': 3.0,
        'min_confidence': 0.75,
        'bb_length': 30,
        'bb_mult': 3.0,
        'kc_length': 30,
        'kc_mult': 2.5,
        'description': 'Long-term swing trading setup for 1H+ timeframes',
        'best_for': ['swing_trading', 'position_holding', 'major_moves'],
        'best_volatility_regime': 'medium_to_high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['any'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.NZDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 200.0
    },
    'news_safe': {
        'zl_length': 70,
        'band_multiplier': 2.0,
        'min_confidence': 0.72,
        'bb_length': 20,
        'bb_mult': 2.0,
        'kc_length': 20,
        'kc_mult': 1.5,
        'description': 'News-event resistant configuration with stable parameters',
        'best_for': ['news_trading', 'volatile_periods', 'economic_events'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'medium',
        'best_market_regime': 'breakout',
        'best_session': ['london_open', 'ny_open'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 10.0,
        'max_pip_volatility': 80.0
    },
    'crypto': {
        'zl_length': 34,
        'band_multiplier': 2.5,
        'min_confidence': 0.60,
        'bb_length': 20,
        'bb_mult': 2.5,
        'kc_length': 20,
        'kc_mult': 2.0,
        'description': 'Optimized for cryptocurrency-like high volatility assets',
        'best_for': ['high_volatility', 'crypto_style', 'wide_ranges'],
        'best_volatility_regime': 'very_high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['24h_trading'],
        'preferred_pairs': ['high_volatility_pairs'],
        'min_pip_volatility': 30.0,
        'max_pip_volatility': 300.0
    }
}

# Active configuration selector
ACTIVE_ZERO_LAG_CONFIG = 'default'

# Individual feature toggles
ZERO_LAG_SQUEEZE_MOMENTUM_ENABLED = True
ZERO_LAG_MTF_VALIDATION_ENABLED = True
ZERO_LAG_SMART_MONEY_ENABLED = False

# =============================================================================
# MULTI-TIMEFRAME VALIDATION SETTINGS
# =============================================================================

# Enable/disable higher timeframe validation for improved signal quality
HIGHER_TIMEFRAME_VALIDATION = True                 # Master switch for multi-timeframe validation

# Zero Lag Strategy Multi-Timeframe Settings
ZERO_LAG_MTF_ENABLED = True                       # Enable MTF validation for Zero Lag strategy
ZERO_LAG_MTF_REQUIRE_1H = True                   # Require 1H timeframe confirmation
ZERO_LAG_MTF_REQUIRE_4H = True                   # Require 4H timeframe confirmation
ZERO_LAG_MTF_STRICT_MODE = True                  # True = both 1H and 4H must pass, False = either one passes

# Multi-timeframe data lookback settings
MTF_1H_LOOKBACK_HOURS = 200                      # Hours of 1H data to fetch for validation
MTF_4H_LOOKBACK_HOURS = 800                      # Hours of 4H data to fetch for validation

# Multi-timeframe validation criteria
# For BULL signals: higher timeframe must have close > EMA200 AND close > ZLEMA AND trend >= 0
# For BEAR signals: higher timeframe must have close < EMA200 AND close < ZLEMA AND trend <= 0

# MTF confidence adjustments
ZERO_LAG_MTF_CONFIDENCE_BOOST = 0.15             # Confidence boost when MTF validates signal
ZERO_LAG_MTF_CONFIDENCE_PENALTY = 0.25           # Confidence penalty when MTF invalidates signal

# MTF timeframe weights for confidence calculation
ZERO_LAG_MTF_TIMEFRAME_WEIGHTS = {
    '1h': 0.6,      # 60% weight for 1H timeframe validation
    '4h': 0.4       # 40% weight for 4H timeframe validation
}

# MTF validation thresholds
ZERO_LAG_MTF_TREND_THRESHOLD = 0.0001            # Minimum trend strength for MTF validation
ZERO_LAG_MTF_EMA200_BUFFER_PIPS = 2              # Buffer around EMA200 for MTF validation

# Backward compatibility with original settings
ZERO_LAG_LENGTH = ZERO_LAG_STRATEGY_CONFIG['default']['zl_length']
ZERO_LAG_BAND_MULT = ZERO_LAG_STRATEGY_CONFIG['default']['band_multiplier']  
ZERO_LAG_MIN_CONFIDENCE = ZERO_LAG_STRATEGY_CONFIG['default']['min_confidence']

# Legacy compatibility settings
ZERO_LAG_BASE_CONFIDENCE = 0.70
ZERO_LAG_MAX_CONFIDENCE = 0.95
ZERO_LAG_TREND_WEIGHT = 0.20
ZERO_LAG_VOLATILITY_WEIGHT = 0.15
ZERO_LAG_MOMENTUM_WEIGHT = 0.10
ZERO_LAG_STRATEGY_WEIGHT = 0.15
ZERO_LAG_ALLOW_COMBINED = True
ZERO_LAG_PRIORITY_LEVEL = 3
ZERO_LAG_ENABLE_BACKTESTING = True
ZERO_LAG_MIN_DATA_PERIODS = 150
ZERO_LAG_ENABLE_PERFORMANCE_TRACKING = True
ZERO_LAG_DEBUG_LOGGING = True


# Helper functions
def get_zerolag_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get zero-lag configuration for specific epic with fallbacks"""
    # Try to get specific market condition config
    if market_condition in ZERO_LAG_STRATEGY_CONFIG:
        config = ZERO_LAG_STRATEGY_CONFIG[market_condition]
    else:
        config = ZERO_LAG_STRATEGY_CONFIG['default']
    
    # Epic-specific adjustments
    epic_upper = epic.upper()
    config = config.copy()  # Don't modify original
    
    if 'JPY' in epic_upper:
        # JPY pairs need wider bands for higher volatility
        config['band_multiplier'] *= 1.2
        config['min_confidence'] = min(0.75, config['min_confidence'] + 0.05)
    elif 'GBP' in epic_upper:
        # GBP pairs are volatile
        config['band_multiplier'] *= 1.1
    elif epic_upper in ['CS.D.EURUSD.CEEM.IP', 'CS.D.USDCAD.MINI.IP']:
        # Major pairs - slight tightening
        config['band_multiplier'] *= 0.9
    
    return config


def get_zerolag_threshold_for_epic(epic: str) -> float:
    """Get zero-lag specific thresholds based on currency pair"""
    epic_upper = epic.upper()
    
    if 'JPY' in epic_upper:
        # JPY pairs have different pip values (0.01 vs 0.0001)
        return 0.003  # Wider threshold for JPY volatility
    elif 'GBP' in epic_upper:
        # GBP pairs are naturally more volatile
        return 0.00005
    else:
        # Standard threshold for major pairs
        return 0.00003


def get_zerolag_squeeze_config_for_epic(epic: str) -> dict:
    """Get squeeze momentum configuration for specific epic"""
    base_config = ZERO_LAG_STRATEGY_CONFIG['default']
    epic_upper = epic.upper()
    
    config = {
        'bb_length': base_config['bb_length'],
        'bb_mult': base_config['bb_mult'],
        'kc_length': base_config['kc_length'],
        'kc_mult': base_config['kc_mult']
    }
    
    # Epic-specific squeeze adjustments
    if 'JPY' in epic_upper:
        # JPY pairs need wider squeeze bands
        config['bb_mult'] *= 1.3
        config['kc_mult'] *= 1.2
    elif epic_upper in ['CS.D.GBPJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP']:
        # Cross JPY pairs are especially volatile
        config['bb_mult'] *= 1.5
        config['kc_mult'] *= 1.4
    
    return config


# Validation function  
def validate_zerolag_config() -> dict:
    """Validate zero-lag strategy configuration completeness"""
    try:
        required_keys = ['ZERO_LAG_STRATEGY', 'ZERO_LAG_STRATEGY_CONFIG']
        missing_keys = []
        
        for key in required_keys:
            if not globals().get(key):
                missing_keys.append(key)
        
        if missing_keys:
            return {
                'valid': False, 
                'error': f'Missing required keys: {missing_keys}',
                'config_count': 0
            }
        
        # Validate all config presets have required fields
        required_config_fields = ['zl_length', 'band_multiplier', 'min_confidence', 
                                'bb_length', 'bb_mult', 'kc_length', 'kc_mult']
        
        invalid_configs = []
        for config_name, config_data in ZERO_LAG_STRATEGY_CONFIG.items():
            missing_fields = [field for field in required_config_fields if field not in config_data]
            if missing_fields:
                invalid_configs.append(f"{config_name}: missing {missing_fields}")
        
        if invalid_configs:
            return {
                'valid': False,
                'error': f'Invalid configurations: {invalid_configs}',
                'config_count': len(ZERO_LAG_STRATEGY_CONFIG)
            }
        
        # All validations passed
        return {
            'valid': True, 
            'config_count': len(ZERO_LAG_STRATEGY_CONFIG),
            'active_config': ACTIVE_ZERO_LAG_CONFIG,
            'presets': list(ZERO_LAG_STRATEGY_CONFIG.keys())
        }
        
    except Exception as e:
        return {
            'valid': False, 
            'error': str(e),
            'config_count': 0
        }