# configdata/strategies/config_bb_supertrend_strategy.py
"""
Bollinger Bands + SuperTrend Strategy Configuration
==================================================

Combined strategy using Bollinger Bands for volatility analysis and SuperTrend
for trend direction. This hybrid approach aims to capture high-quality signals
by combining mean reversion (BB) with trend following (SuperTrend) concepts.

Key Features:
- Multiple configuration presets optimized for different market conditions
- Enhanced signal filtering and quality controls
- Multi-timeframe validation capabilities
- Advanced market condition adaptation
- Comprehensive risk management settings
"""

# Strategy enable/disable
BOLLINGER_SUPERTREND_STRATEGY = False  # Currently disabled - enable when ready for testing

# Core BB SuperTrend Configuration
BB_PERIOD = 24                          # Bollinger Bands period - more conservative base setting
BB_STD_DEV = 2.8                        # Standard deviation multiplier - less noisy
SUPERTREND_PERIOD = 16                  # SuperTrend period - more stable
SUPERTREND_MULTIPLIER = 4.0             # SuperTrend ATR multiplier - higher noise filtering
BB_SUPERTREND_BASE_CONFIDENCE = 0.65    # Higher base confidence
BB_SUPERTREND_MAX_CONFIDENCE = 0.85     # Higher max confidence
DEFAULT_BB_SUPERTREND_CONFIG = 'conservative'  # Conservative default

# BB SuperTrend Strategy Configurations with Much Better Parameters
BB_SUPERTREND_CONFIGS = {
    'conservative': {
        'bb_period': 24,                 # Longer period for maximum stability
        'bb_std_dev': 2.8,               # Very wide bands for highest quality signals
        'supertrend_period': 16,         # Long trend confirmation period
        'supertrend_multiplier': 4.0,    # High multiplier to eliminate noise
        'base_confidence': 0.75,         # High confidence threshold
        'min_bb_width_pct': 0.0018,     # Require significant volatility (0.18%)
        'max_signals_per_session': 2,   # Very selective signal generation
        'require_mtf_confluence': True, # Mandatory multi-timeframe confirmation
        'min_mtf_confluence': 0.65,     # High MTF agreement requirement
        'enable_signal_filtering': True,
        'min_bb_position_score': 0.8,   # Very strict position requirements
        'description': 'Maximum quality signals with minimal noise - 5-10 signals per week',
        'best_for': ['strong_trends', 'low_noise', 'high_quality'],
        'best_volatility_regime': 'low_to_medium',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 25.0
    },
    'balanced': {
        'bb_period': 20,                 # Good balance of stability and responsiveness
        'bb_std_dev': 2.2,               # Moderately wide bands
        'supertrend_period': 12,         # Balanced trend detection
        'supertrend_multiplier': 3.2,    # Good noise filtering
        'base_confidence': 0.65,         # Solid confidence threshold
        'min_bb_width_pct': 0.0015,     # Reasonable volatility requirement (0.15%)
        'max_signals_per_session': 4,   # Moderate signal frequency
        'require_mtf_confluence': True, # MTF confirmation recommended
        'min_mtf_confluence': 0.5,      # 50% MTF agreement minimum
        'enable_signal_filtering': True,
        'min_bb_position_score': 0.7,   # Good position requirements
        'description': 'Balanced quality and frequency - 15-25 signals per week',
        'best_for': ['trending', 'medium_volatility', 'balanced_approach'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium_to_strong',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 40.0
    },
    'default': {
        'bb_period': 20,                 # More stable than original 14
        'bb_std_dev': 2.2,               # Less noisy than original 1.8
        'supertrend_period': 10,         # More stable than original 8
        'supertrend_multiplier': 3.0,    # Less noise than original 2.5
        'base_confidence': 0.6,          # Higher quality than original 0.5
        'min_bb_width_pct': 0.0012,     # Minimum volatility requirement (0.12%)
        'max_signals_per_session': 5,   # Prevent over-trading
        'require_mtf_confluence': False, # Optional MTF for default
        'min_mtf_confluence': 0.4,      # 40% MTF requirement
        'enable_signal_filtering': True,
        'min_bb_position_score': 0.6,   # Moderate position requirements
        'description': 'Standard configuration with improved parameters - 20-30 signals per week',
        'best_for': ['general_trading', 'standard_conditions', 'learning'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'mixed',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.USDJPY.MINI.IP']
    },
    'aggressive': {
        'bb_period': 16,                 # Slightly more stable than original 14
        'bb_std_dev': 2.0,               # Less tight than original 1.8
        'supertrend_period': 8,          # Keep responsive
        'supertrend_multiplier': 2.8,    # Less noise than original 2.5
        'base_confidence': 0.55,         # Higher than original 0.5
        'min_bb_width_pct': 0.001,      # Lower volatility requirement (0.1%)
        'max_signals_per_session': 7,   # Allow more signals
        'require_mtf_confluence': False, # No MTF requirement for aggressive
        'min_mtf_confluence': 0.3,      # Low MTF requirement
        'enable_signal_filtering': True,
        'min_bb_position_score': 0.5,   # Looser position requirements
        'description': 'Higher frequency with controlled quality - 30-45 signals per week',
        'best_for': ['breakouts', 'high_volatility', 'active_trading'],
        'best_volatility_regime': 'medium_to_high',
        'best_trend_strength': 'medium',
        'best_market_regime': 'breakout',
        'preferred_pairs': ['CS.D.GBPUSD.MINI.IP', 'CS.D.EURJPY.MINI.IP']
    },
    'experimental_quality': {
        'bb_period': 25,                 # Very long for maximum stability
        'bb_std_dev': 2.5,               # Wide bands for quality
        'supertrend_period': 18,         # Very long trend confirmation
        'supertrend_multiplier': 4.5,    # Very strong noise filtering
        'base_confidence': 0.8,          # Very high confidence
        'min_bb_width_pct': 0.002,      # High volatility requirement (0.2%)
        'max_signals_per_session': 1,   # Extremely selective
        'require_mtf_confluence': True, # Mandatory MTF
        'min_mtf_confluence': 0.7,      # Very high MTF requirement
        'enable_signal_filtering': True,
        'min_bb_position_score': 0.85,  # Very strict position
        'require_trend_alignment': True, # Additional filter
        'min_atr_multiple': 1.8,        # High volatility filter
        'description': 'Experimental ultra-high quality mode - 2-5 signals per week',
        'best_for': ['research', 'ultra_quality', 'minimal_trades']
    }
}

# Active configuration selector
ACTIVE_BB_SUPERTREND_CONFIG = DEFAULT_BB_SUPERTREND_CONFIG
ENABLE_DYNAMIC_BB_SUPERTREND_CONFIG = True  # Enable dynamic configuration selection

# Critical Signal Quality Improvements
BB_SUPERTREND_ENHANCEMENTS = {
    # Signal Filtering - ENABLED BY DEFAULT
    'enable_signal_filtering': True,
    'min_bb_separation_pips': 10,       # Increased minimum BB width (1.0 pip)
    'max_signals_per_hour': 1,          # Much more restrictive
    'max_signals_per_day': 8,           # Daily limit
    'require_volume_confirmation': False, # Volume confirmation optional
    'min_candle_body_ratio': 0.4,       # Stronger candle requirement

    # Enhanced Quality Filters
    'require_bb_extremes': True,        # Price must be near BB extremes
    'bb_extreme_threshold': 0.8,        # 80% towards BB edge
    'require_supertrend_flip': True,    # SuperTrend must have flipped recently
    'max_supertrend_age': 5,            # SuperTrend flip within 5 bars

    # Market Condition Filters
    'avoid_low_volatility': True,       # Skip low volatility periods
    'min_atr_threshold': 0.0008,        # Minimum ATR for signals (0.08%)
    'avoid_ranging_markets': True,       # Skip sideways markets
    'trend_strength_threshold': 0.6,    # Minimum trend strength (60%)

    # Risk Management Enhancement
    'dynamic_position_sizing': True,
    'max_risk_per_trade': 0.015,        # 1.5% max risk (reduced from 2%)
    'profit_target_multiplier': 3.0,    # 3x risk for profit (increased from 2x)
    'stop_loss_atr_multiple': 2.0,      # 2x ATR for stop loss

    # Session and Time Filters
    'trading_sessions': ['london', 'ny_overlap'], # Preferred sessions only
    'avoid_news_times': True,           # Avoid major news releases
    'session_volatility_filter': True,  # Only trade in volatile sessions
    'avoid_rollover_times': True,       # Avoid daily rollover periods

    # Multi-Timeframe Enhancement
    'mtf_timeframes': ['15m', '1h', '4h'], # Multiple timeframe analysis
    'require_htf_trend_alignment': True,   # Higher timeframe trend alignment
    'htf_lookback_periods': 50,           # Look back 50 periods on HTF
    'mtf_confidence_boost': 0.15,         # Confidence boost for MTF alignment

    # Advanced Signal Validation
    'require_momentum_confirmation': True, # Require momentum indicator confirmation
    'min_momentum_strength': 0.3,        # Minimum momentum strength (30%)
    'require_volatility_expansion': True, # Require volatility to be expanding
    'min_volatility_expansion_ratio': 1.2, # Minimum expansion ratio (20%)

    # Smart Money Integration
    'enable_smart_money_filters': False,  # Smart money structure analysis (disabled by default)
    'require_order_block_confluence': False, # Order block confluence (disabled)
    'require_fvg_alignment': False,       # Fair value gap alignment (disabled)

    # Performance Optimization
    'enable_signal_caching': True,       # Cache signal calculations
    'cache_duration_minutes': 5,         # Signal cache duration
    'enable_parallel_processing': False,  # Parallel processing (experimental)

    # Adaptive Features
    'enable_adaptive_parameters': True,   # Adapt parameters to market conditions
    'volatility_adaptation_enabled': True, # Adapt to volatility changes
    'trend_adaptation_enabled': True,    # Adapt to trend strength changes
    'session_adaptation_enabled': True   # Adapt to trading session characteristics
}

# Debugging and Monitoring Configuration
BB_SUPERTREND_DEBUG = {
    'log_signal_details': True,         # Log detailed signal information
    'log_rejection_reasons': True,      # Log why signals were rejected
    'track_mtf_performance': True,      # Track multi-timeframe performance
    'enable_cache_monitoring': True,    # Monitor caching performance
    'performance_tracking': True,       # Track overall strategy performance
    'log_config_loading': True,         # Log when config is loaded
    'validate_config_changes': True,    # Validate config changes take effect

    # Detailed Component Logging
    'log_bb_calculations': True,        # Log Bollinger Bands calculations
    'log_supertrend_calculations': True, # Log SuperTrend calculations
    'log_confluence_scores': True,      # Log confluence scoring
    'log_filter_results': True,         # Log individual filter results
    'log_risk_calculations': True,      # Log risk management calculations

    # Performance Monitoring
    'track_signal_accuracy': True,      # Track signal accuracy over time
    'track_filter_effectiveness': True, # Track effectiveness of each filter
    'calculate_win_loss_ratios': True,  # Calculate win/loss ratios by config
    'monitor_execution_times': True,    # Monitor strategy execution times

    # Alert System
    'enable_quality_alerts': True,      # Alert on quality issues
    'enable_performance_alerts': True,  # Alert on performance degradation
    'alert_threshold_accuracy': 0.4,    # Alert if accuracy drops below 40%
    'alert_threshold_signals_per_day': 20 # Alert if signals exceed 20 per day
}

# Strategy Integration Settings
STRATEGY_WEIGHT_BB_SUPERTREND = 0.0    # Weight in combined strategies (disabled by default)
BB_SUPERTREND_ALLOW_COMBINED = False   # Don't allow in combined strategies initially
BB_SUPERTREND_PRIORITY_LEVEL = 3       # Priority level (1=highest, 5=lowest)

# Performance Settings
BB_SUPERTREND_ENABLE_BACKTESTING = True # Enable strategy in backtests
BB_SUPERTREND_MIN_DATA_PERIODS = 200   # Minimum data periods required
BB_SUPERTREND_ENABLE_PERFORMANCE_TRACKING = True # Track strategy performance

# Signal Quality Thresholds
BB_SUPERTREND_MIN_CONFIDENCE = 0.60    # Minimum confidence for signals
BB_SUPERTREND_SIGNAL_COOLDOWN = 900    # 15-minute cooldown between signals (seconds)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_bb_supertrend_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """
    Get BB SuperTrend configuration for specific epic and market condition

    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        market_condition: Market condition ('conservative', 'balanced', 'default', 'aggressive', etc.)

    Returns:
        Dictionary with BB SuperTrend configuration
    """
    # Use dynamic config if enabled, otherwise use active config
    if ENABLE_DYNAMIC_BB_SUPERTREND_CONFIG and market_condition in BB_SUPERTREND_CONFIGS:
        config_name = market_condition
    else:
        config_name = ACTIVE_BB_SUPERTREND_CONFIG

    return BB_SUPERTREND_CONFIGS.get(config_name, BB_SUPERTREND_CONFIGS['default'])

def get_bb_supertrend_confidence_for_epic(epic: str, market_condition: str = 'default') -> float:
    """
    Get BB SuperTrend base confidence for specific epic

    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        market_condition: Market condition

    Returns:
        Base confidence value for the configuration
    """
    config = get_bb_supertrend_config_for_epic(epic, market_condition)
    return config.get('base_confidence', BB_SUPERTREND_BASE_CONFIDENCE)

def calculate_bb_position_score(price: float, bb_upper: float, bb_lower: float, bb_middle: float) -> float:
    """
    Calculate Bollinger Bands position score (0-1, where 1 = at extreme)

    Args:
        price: Current price
        bb_upper: Bollinger Band upper line
        bb_lower: Bollinger Band lower line
        bb_middle: Bollinger Band middle line (SMA)

    Returns:
        Position score (0-1)
    """
    try:
        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return 0.0

        if price >= bb_middle:
            # Price above middle - calculate position relative to upper band
            distance_from_middle = price - bb_middle
            max_distance = bb_upper - bb_middle
            return min(1.0, distance_from_middle / max_distance)
        else:
            # Price below middle - calculate position relative to lower band
            distance_from_middle = bb_middle - price
            max_distance = bb_middle - bb_lower
            return min(1.0, distance_from_middle / max_distance)

    except Exception:
        return 0.0

def is_bb_supertrend_signal_valid(bb_position_score: float, supertrend_direction: str,
                                 signal_direction: str, config: dict) -> dict:
    """
    Validate BB SuperTrend signal based on configuration requirements

    Args:
        bb_position_score: Bollinger Bands position score (0-1)
        supertrend_direction: SuperTrend direction ('up' or 'down')
        signal_direction: Signal direction ('BULL' or 'BEAR')
        config: Strategy configuration dictionary

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'rejection_reasons': [],
        'confidence_adjustments': 0.0
    }

    # Check minimum BB position score requirement
    min_bb_score = config.get('min_bb_position_score', 0.6)
    if bb_position_score < min_bb_score:
        validation_result['is_valid'] = False
        validation_result['rejection_reasons'].append(f'BB position score {bb_position_score:.3f} < {min_bb_score}')

    # Check SuperTrend alignment
    expected_supertrend = 'up' if signal_direction == 'BULL' else 'down'
    if supertrend_direction != expected_supertrend:
        validation_result['is_valid'] = False
        validation_result['rejection_reasons'].append(f'SuperTrend {supertrend_direction} conflicts with {signal_direction} signal')

    # Apply confidence boosts for strong signals
    if bb_position_score > 0.9:
        validation_result['confidence_adjustments'] += 0.1  # 10% boost for extreme BB position
    elif bb_position_score > 0.8:
        validation_result['confidence_adjustments'] += 0.05  # 5% boost for strong BB position

    return validation_result

def get_bb_supertrend_risk_parameters(epic: str, config_name: str = None) -> dict:
    """
    Get BB SuperTrend risk management parameters for specific epic

    Args:
        epic: Trading pair epic
        config_name: Optional specific configuration name

    Returns:
        Dictionary with risk management parameters
    """
    config = get_bb_supertrend_config_for_epic(epic, config_name or ACTIVE_BB_SUPERTREND_CONFIG)
    enhancements = BB_SUPERTREND_ENHANCEMENTS

    return {
        'max_risk_per_trade': enhancements['max_risk_per_trade'],
        'profit_target_multiplier': enhancements['profit_target_multiplier'],
        'stop_loss_atr_multiple': enhancements['stop_loss_atr_multiple'],
        'dynamic_position_sizing': enhancements['dynamic_position_sizing'],
        'base_confidence': config.get('base_confidence', BB_SUPERTREND_BASE_CONFIDENCE),
        'max_signals_per_day': enhancements['max_signals_per_day'],
        'max_signals_per_session': config.get('max_signals_per_session', 5)
    }

def validate_bb_supertrend_config() -> dict:
    """
    Validate BB SuperTrend strategy configuration completeness

    Returns:
        Dictionary with validation results
    """
    try:
        # Check required settings
        required_settings = [
            'BOLLINGER_SUPERTREND_STRATEGY', 'BB_SUPERTREND_CONFIGS', 'ACTIVE_BB_SUPERTREND_CONFIG',
            'BB_SUPERTREND_ENHANCEMENTS', 'BB_SUPERTREND_DEBUG'
        ]

        for setting in required_settings:
            if setting not in globals():
                return {'valid': False, 'error': f'Missing required setting: {setting}'}

        # Validate configuration structure
        for config_name, config in BB_SUPERTREND_CONFIGS.items():
            if not isinstance(config, dict):
                return {'valid': False, 'error': f'Config {config_name} must be dict'}

            required_keys = ['bb_period', 'bb_std_dev', 'supertrend_period', 'supertrend_multiplier', 'base_confidence', 'description']
            for key in required_keys:
                if key not in config:
                    return {'valid': False, 'error': f'Config {config_name} missing {key}'}

            # Validate parameter ranges
            if not (5 <= config['bb_period'] <= 50):
                return {'valid': False, 'error': f'Config {config_name}: bb_period must be 5-50'}

            if not (1.0 <= config['bb_std_dev'] <= 5.0):
                return {'valid': False, 'error': f'Config {config_name}: bb_std_dev must be 1.0-5.0'}

            if not (5 <= config['supertrend_period'] <= 50):
                return {'valid': False, 'error': f'Config {config_name}: supertrend_period must be 5-50'}

            if not (1.0 <= config['supertrend_multiplier'] <= 10.0):
                return {'valid': False, 'error': f'Config {config_name}: supertrend_multiplier must be 1.0-10.0'}

            if not (0.0 <= config['base_confidence'] <= 1.0):
                return {'valid': False, 'error': f'Config {config_name}: base_confidence must be 0.0-1.0'}

        # Validate active config exists
        if ACTIVE_BB_SUPERTREND_CONFIG not in BB_SUPERTREND_CONFIGS:
            return {'valid': False, 'error': f'Active config {ACTIVE_BB_SUPERTREND_CONFIG} not found'}

        # Validate enhancement settings
        enhancements = BB_SUPERTREND_ENHANCEMENTS
        if not (0.001 <= enhancements['max_risk_per_trade'] <= 0.1):
            return {'valid': False, 'error': 'max_risk_per_trade must be between 0.1% and 10%'}

        if not (1.0 <= enhancements['profit_target_multiplier'] <= 10.0):
            return {'valid': False, 'error': 'profit_target_multiplier must be 1.0-10.0'}

        return {
            'valid': True,
            'strategy_enabled': BOLLINGER_SUPERTREND_STRATEGY,
            'config_count': len(BB_SUPERTREND_CONFIGS),
            'active_config': ACTIVE_BB_SUPERTREND_CONFIG,
            'dynamic_config_enabled': ENABLE_DYNAMIC_BB_SUPERTREND_CONFIG,
            'signal_filtering_enabled': BB_SUPERTREND_ENHANCEMENTS['enable_signal_filtering'],
            'mtf_enabled': any(config.get('require_mtf_confluence', False) for config in BB_SUPERTREND_CONFIGS.values()),
            'debug_logging_enabled': BB_SUPERTREND_DEBUG['log_signal_details']
        }

    except Exception as e:
        return {'valid': False, 'error': f'Validation exception: {str(e)}'}

def get_bb_supertrend_config_summary() -> dict:
    """Get a summary of BB SuperTrend configuration settings"""
    return {
        'strategy_enabled': BOLLINGER_SUPERTREND_STRATEGY,
        'active_config': ACTIVE_BB_SUPERTREND_CONFIG,
        'dynamic_config_enabled': ENABLE_DYNAMIC_BB_SUPERTREND_CONFIG,
        'total_configurations': len(BB_SUPERTREND_CONFIGS),
        'available_configs': list(BB_SUPERTREND_CONFIGS.keys()),
        'signal_filtering_enabled': BB_SUPERTREND_ENHANCEMENTS['enable_signal_filtering'],
        'max_signals_per_day': BB_SUPERTREND_ENHANCEMENTS['max_signals_per_day'],
        'max_signals_per_hour': BB_SUPERTREND_ENHANCEMENTS['max_signals_per_hour'],
        'min_confidence': BB_SUPERTREND_MIN_CONFIDENCE,
        'signal_cooldown_seconds': BB_SUPERTREND_SIGNAL_COOLDOWN,
        'strategy_weight': STRATEGY_WEIGHT_BB_SUPERTREND,
        'allow_combined': BB_SUPERTREND_ALLOW_COMBINED,
        'debug_logging': BB_SUPERTREND_DEBUG['log_signal_details'],
        'performance_tracking': BB_SUPERTREND_DEBUG['performance_tracking'],
        'current_config_details': get_bb_supertrend_config_for_epic('default')
    }

# BB SuperTrend frequency presets
BB_SUPERTREND_FREQUENCY_PRESETS = {
    'ultra_conservative': {
        'active_config': 'experimental_quality',
        'max_signals_per_day': 3,
        'min_confidence': 0.8,
        'description': 'Ultra-high quality, very few signals - 2-5 per week'
    },
    'conservative': {
        'active_config': 'conservative',
        'max_signals_per_day': 6,
        'min_confidence': 0.75,
        'description': 'High quality signals - 5-10 per week'
    },
    'balanced': {
        'active_config': 'balanced',
        'max_signals_per_day': 8,
        'min_confidence': 0.65,
        'description': 'Balanced quality and frequency - 15-25 per week'
    },
    'active': {
        'active_config': 'default',
        'max_signals_per_day': 12,
        'min_confidence': 0.6,
        'description': 'More active trading - 20-30 per week'
    },
    'aggressive': {
        'active_config': 'aggressive',
        'max_signals_per_day': 15,
        'min_confidence': 0.55,
        'description': 'Higher frequency trading - 30-45 per week'
    }
}

def set_bb_supertrend_frequency_preset(preset: str = 'balanced'):
    """
    Apply a BB SuperTrend frequency preset

    Args:
        preset: 'ultra_conservative', 'conservative', 'balanced', 'active', or 'aggressive'
    """
    if preset in BB_SUPERTREND_FREQUENCY_PRESETS:
        preset_config = BB_SUPERTREND_FREQUENCY_PRESETS[preset]
        return {
            'preset_applied': preset,
            'settings': preset_config,
            'description': preset_config['description']
        }
    else:
        return {
            'error': f'Unknown preset: {preset}',
            'available_presets': list(BB_SUPERTREND_FREQUENCY_PRESETS.keys())
        }

print("✅ BB SuperTrend Strategy configuration loaded successfully")