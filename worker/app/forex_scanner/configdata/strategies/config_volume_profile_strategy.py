# configdata/strategies/config_volume_profile_strategy.py
"""
Volume Profile Strategy Configuration
Configuration module for Volume Profile strategy settings including
institutional levels, HVN/LVN zones, and mean reversion signals
"""

# =============================================================================
# VOLUME PROFILE STRATEGY CORE CONFIGURATION
# =============================================================================

# Core Strategy Settings
VOLUME_PROFILE_STRATEGY = True        # Enable/disable Volume Profile strategy
STRATEGY_WEIGHT_VOLUME_PROFILE = 0.5  # Strategy weight in combined mode

# =============================================================================
# VOLUME PROFILE CALCULATION SETTINGS
# =============================================================================

# Lookback Period Configuration
VP_LOOKBACK_PERIODS = {
    'default': 50,                    # Default lookback bars for volume profile
    'short_term': 30,                 # Short-term profile (intraday)
    'medium_term': 50,                # Medium-term profile (swing trading)
    'long_term': 100,                 # Long-term profile (position trading)
}

# Active lookback setting
ACTIVE_VP_LOOKBACK = 'medium_term'

# Volume Profile Binning Configuration
VP_NUM_BINS = 50                      # Number of price bins for volume distribution (more bins = finer granularity)
VP_MIN_BINS = 30                      # Minimum number of bins
VP_MAX_BINS = 100                     # Maximum number of bins

# Volume Column Selection
VP_VOLUME_COLUMN = 'volume'           # Column to use for volume analysis ('volume', 'tick_volume')
VP_USE_TICK_VOLUME = True             # Use tick volume (price changes) if true volume unavailable

# =============================================================================
# HVN/LVN ZONE DETECTION SETTINGS
# =============================================================================

# High Volume Node (HVN) Detection
VP_HVN_DETECTION = {
    'enabled': True,                  # Enable HVN detection
    'peak_prominence': 0.2,           # Minimum prominence for peak detection (0-1, higher = stronger peaks)
    'peak_distance': 5,               # Minimum distance between peaks (in bins)
    'min_strength': 0.5,              # Minimum normalized strength (0-1)
    'zone_width_pips': 3.0,           # Width of HVN zone in pips (half above, half below center)
}

# Low Volume Node (LVN) Detection
VP_LVN_DETECTION = {
    'enabled': True,                  # Enable LVN detection
    'valley_prominence': 0.15,        # Minimum prominence for valley detection (0-1)
    'valley_distance': 5,             # Minimum distance between valleys (in bins)
    'max_strength': 0.3,              # Maximum normalized strength (0-1, lower = weaker nodes)
    'zone_width_pips': 2.0,           # Width of LVN zone in pips
}

# =============================================================================
# VALUE AREA CONFIGURATION
# =============================================================================

# Value Area Calculation (70% volume around POC)
VP_VALUE_AREA_PERCENT = 0.70          # Percentage of volume contained in value area (default 70%)
VP_VALUE_AREA_STRICT = True           # Strictly enforce 70% (vs. approximately)

# Value Area Boundary Behavior
VP_VAH_VAL_AS_SUPPORT_RESISTANCE = True  # Use VAH/VAL as support/resistance levels
VP_VALUE_AREA_BREAKOUT_THRESHOLD_PIPS = 5.0  # Pips beyond VAH/VAL for breakout confirmation

# =============================================================================
# SIGNAL GENERATION STRATEGY PRESETS
# =============================================================================

# Volume Profile Strategy Configurations with different trading styles
VP_STRATEGY_CONFIG = {
    'default': {
        'description': 'Balanced mean reversion + breakout strategy',
        'lookback_periods': 50,
        'min_confidence': 0.60,
        'hvn_proximity_threshold_pips': 5.0,
        'lvn_proximity_threshold_pips': 5.0,
        'poc_proximity_threshold_pips': 3.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': True,
        'enable_poc_bounce': True,
        'best_for': ['trending', 'balanced'],
        'best_session': ['london', 'new_york'],
    },
    'conservative': {
        'description': 'Conservative mean reversion at strong HVN levels',
        'lookback_periods': 100,             # Longer lookback for stronger levels
        'min_confidence': 0.70,              # Higher confidence requirement
        'hvn_proximity_threshold_pips': 3.0, # Tighter proximity (more precise entries)
        'lvn_proximity_threshold_pips': 8.0,
        'poc_proximity_threshold_pips': 2.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': False,    # Disable breakout signals (higher risk)
        'enable_poc_bounce': True,
        'hvn_min_strength': 0.7,             # Only strongest HVN zones
        'require_value_area_proximity': True,
        'best_for': ['ranging', 'low_volatility'],
        'best_session': ['london', 'asian'],
    },
    'aggressive': {
        'description': 'Aggressive breakouts + mean reversion',
        'lookback_periods': 30,              # Shorter lookback for recent action
        'min_confidence': 0.50,              # Lower confidence for more signals
        'hvn_proximity_threshold_pips': 8.0,
        'lvn_proximity_threshold_pips': 5.0,
        'poc_proximity_threshold_pips': 5.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': True,
        'enable_poc_bounce': True,
        'hvn_min_strength': 0.4,             # Accept weaker HVN zones
        'enable_lvn_breakout': True,
        'best_for': ['breakout', 'high_volatility'],
        'best_session': ['overlap', 'new_york'],
    },
    'scalping': {
        'description': 'Fast scalping at POC and VAH/VAL boundaries',
        'lookback_periods': 20,              # Very short lookback
        'min_confidence': 0.55,
        'hvn_proximity_threshold_pips': 3.0,
        'lvn_proximity_threshold_pips': 3.0,
        'poc_proximity_threshold_pips': 2.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': False,    # Disable breakouts (too slow)
        'enable_poc_bounce': True,
        'enable_vah_val_bounce': True,       # Scalp at VA boundaries
        'tight_stops': True,
        'best_for': ['ranging', 'high_frequency'],
        'best_session': ['london', 'overlap', 'new_york'],
    },
    'swing': {
        'description': 'Swing trading at major institutional levels',
        'lookback_periods': 100,             # Long lookback for major levels
        'min_confidence': 0.65,
        'hvn_proximity_threshold_pips': 8.0,
        'lvn_proximity_threshold_pips': 10.0,
        'poc_proximity_threshold_pips': 5.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': True,
        'enable_poc_bounce': True,
        'hvn_min_strength': 0.6,
        'require_trend_alignment': True,     # Swing with trend only
        'best_for': ['trending', 'position_trading'],
        'best_session': ['london', 'new_york'],
    },
    'news_safe': {
        'description': 'Conservative during high volatility / news events',
        'lookback_periods': 50,
        'min_confidence': 0.75,              # Very high confidence
        'hvn_proximity_threshold_pips': 2.0, # Very tight entries
        'lvn_proximity_threshold_pips': 10.0,
        'poc_proximity_threshold_pips': 2.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': False,    # No breakouts during news
        'enable_poc_bounce': True,
        'hvn_min_strength': 0.8,             # Only strongest levels
        'disable_during_news': True,
        'best_for': ['news_events', 'high_volatility'],
        'best_session': ['asian'],
    },
    'crypto': {
        'description': 'High volatility markets with wide zones',
        'lookback_periods': 40,
        'min_confidence': 0.60,
        'hvn_proximity_threshold_pips': 15.0, # Wider zones for high volatility
        'lvn_proximity_threshold_pips': 10.0,
        'poc_proximity_threshold_pips': 8.0,
        'enable_mean_reversion': True,
        'enable_breakout_signals': True,
        'enable_poc_bounce': True,
        'hvn_min_strength': 0.5,
        'enable_lvn_breakout': True,
        'best_for': ['high_volatility', 'breakout'],
        'best_session': ['overlap', 'new_york'],
    },
}

# Active configuration
ACTIVE_VP_CONFIG = 'default'

# =============================================================================
# SIGNAL TYPES AND PRIORITIES
# =============================================================================

# Signal Types Configuration
VP_SIGNAL_TYPES = {
    'hvn_bounce': {
        'enabled': True,
        'description': 'Price bounces off High Volume Node (support/resistance)',
        'priority': 1,                       # Highest priority (institutional level)
        'min_confidence': 0.60,
    },
    'poc_reversion': {
        'enabled': True,
        'description': 'Price reverts to Point of Control (POC magnet effect)',
        'priority': 2,
        'min_confidence': 0.55,
    },
    'value_area_breakout': {
        'enabled': True,
        'description': 'Price breaks out of Value Area (trend acceleration)',
        'priority': 3,
        'min_confidence': 0.65,
    },
    'lvn_breakout': {
        'enabled': True,
        'description': 'Price breaks through Low Volume Node (weak resistance)',
        'priority': 4,
        'min_confidence': 0.60,
    },
    'value_area_bounce': {
        'enabled': True,
        'description': 'Price bounces at VAH/VAL boundaries',
        'priority': 5,
        'min_confidence': 0.55,
    },
}

# =============================================================================
# CONFIDENCE SCORING CONFIGURATION
# =============================================================================

# Base Confidence Settings
VP_CONFIDENCE_BASE = 0.50                 # Base confidence score

# Confidence Boost Factors
VP_CONFIDENCE_FACTORS = {
    'hvn_strength': 0.20,                 # HVN strength multiplier (max +20%)
    'at_poc': 0.15,                       # Bonus for POC proximity (+15%)
    'value_area_position': 0.10,          # Bonus for favorable VA position (+10%)
    'volume_skewness_alignment': 0.10,    # Bonus if volume skewness supports signal (+10%)
    'multiple_hvn_confluence': 0.10,      # Bonus if multiple HVN zones nearby (+10%)
    'trend_alignment': 0.10,              # Bonus if aligned with trend (+10%)
}

# Confidence Penalty Factors
VP_CONFIDENCE_PENALTIES = {
    'at_lvn': -0.15,                      # Penalty for LVN proximity (-15%)
    'extreme_value_area_position': -0.10, # Penalty if too far from VA (-10%)
    'counter_trend': -0.15,               # Penalty if counter-trend (-15%)
    'low_volume_period': -0.10,           # Penalty for low volume (-10%)
}

# =============================================================================
# RISK MANAGEMENT SETTINGS
# =============================================================================

# Stop Loss Configuration
VP_STOP_LOSS_CONFIG = {
    'method': 'hvn_based',                # 'hvn_based', 'atr_based', 'hybrid'
    'hvn_buffer_pips': 2.0,               # Buffer beyond HVN zone boundary
    'atr_multiplier': 1.5,                # ATR multiplier for ATR-based stops
    'min_stop_pips': 10.0,                # Minimum stop loss distance
    'max_stop_pips': 40.0,                # Maximum stop loss distance
    'use_next_hvn': True,                 # Place stop beyond next HVN level
}

# Take Profit Configuration
VP_TAKE_PROFIT_CONFIG = {
    'method': 'hvn_based',                # 'hvn_based', 'poc_based', 'atr_based', 'hybrid'
    'target_next_hvn': True,              # Target next HVN level
    'target_poc': True,                   # Target POC if closer than HVN
    'target_vah_val': True,               # Target VAH/VAL boundaries
    'atr_multiplier': 3.0,                # ATR multiplier for ATR-based targets
    'min_reward_to_risk': 1.5,            # Minimum R:R ratio
    'max_target_pips': 60.0,              # Maximum take profit distance
}

# Position Sizing
VP_POSITION_SIZING = {
    'risk_per_trade_percent': 1.0,        # Risk 1% of account per trade
    'max_risk_pips': 40.0,                # Maximum risk in pips
    'scale_by_confidence': True,          # Scale position by confidence score
}

# =============================================================================
# SIGNAL FILTERING CONFIGURATION
# =============================================================================

# General Signal Filters
VP_SIGNAL_FILTERS = {
    'min_confidence': 0.60,               # Minimum confidence score (0-1)
    'max_daily_signals': 10,              # Maximum signals per day per pair
    'min_spacing_minutes': 30,            # Minimum time between signals (same pair)
    'require_fresh_profile': True,        # Require recent volume profile calculation
    'max_profile_age_bars': 10,           # Maximum age of profile data
}

# Proximity Filters (Default - overridden by preset)
VP_PROXIMITY_FILTERS = {
    'hvn_threshold_pips': 5.0,            # Distance to HVN for signal
    'lvn_threshold_pips': 5.0,            # Distance to LVN for signal
    'poc_threshold_pips': 3.0,            # Distance to POC for signal
    'vah_val_threshold_pips': 5.0,        # Distance to VAH/VAL for signal
}

# Volume Analysis Filters
VP_VOLUME_FILTERS = {
    'min_volume_percentile': 20,          # Minimum volume percentile for signals
    'require_volume_confirmation': False, # Require volume spike on breakouts
    'volume_spike_threshold': 1.5,        # Volume multiplier for "spike"
}

# Trend Alignment Filter
VP_TREND_FILTER = {
    'enabled': False,                     # Enable trend alignment requirement (default off)
    'ema_period': 200,                    # EMA period for trend determination
    'require_alignment': False,           # Require signal direction matches trend
    'penalty_if_counter_trend': True,     # Apply confidence penalty if counter-trend
}

# =============================================================================
# MARKET REGIME CONFIGURATION
# =============================================================================

# ADX Trend Strength Filter
VP_ADX_FILTER = {
    'enabled': False,                     # Volume Profile works in all conditions (default off)
    'min_adx': 20,                        # Minimum ADX for trending markets
    'max_adx': 60,                        # Maximum ADX (avoid extreme conditions)
    'adx_weight': 0.05,                   # Weight in confidence calculation
}

# ATR Volatility Filter
VP_ATR_FILTER = {
    'enabled': False,                     # Volume Profile adaptive to volatility (default off)
    'min_atr_ratio': 0.7,                 # Minimum ATR vs baseline
    'max_atr_ratio': 3.0,                 # Maximum ATR vs baseline
    'adjust_zones_by_atr': True,          # Scale zone widths by volatility
}

# Market Session Filter
VP_SESSION_FILTER = {
    'enabled': False,                     # Volume Profile works 24/5 (default off)
    'preferred_sessions': ['london', 'new_york', 'overlap'],
    'session_confidence_boost': 0.05,     # Boost during preferred sessions
}

# =============================================================================
# PAIR-SPECIFIC CONFIGURATION
# =============================================================================

# Pair-Specific Overrides
VP_PAIR_CONFIG = {
    'EURUSD': {
        'lookback_periods': 50,
        'hvn_proximity_threshold_pips': 5.0,
        'min_stop_pips': 10.0,
        'max_stop_pips': 35.0,
    },
    'GBPUSD': {
        'lookback_periods': 50,
        'hvn_proximity_threshold_pips': 7.0,  # More volatile
        'min_stop_pips': 12.0,
        'max_stop_pips': 45.0,
    },
    'USDJPY': {
        'lookback_periods': 50,
        'hvn_proximity_threshold_pips': 5.0,  # JPY pairs use different pip scale
        'min_stop_pips': 10.0,
        'max_stop_pips': 40.0,
    },
    'AUDUSD': {
        'lookback_periods': 50,
        'hvn_proximity_threshold_pips': 6.0,
        'min_stop_pips': 10.0,
        'max_stop_pips': 40.0,
    },
    'GBPJPY': {
        'lookback_periods': 50,
        'hvn_proximity_threshold_pips': 8.0,  # Very volatile
        'min_stop_pips': 15.0,
        'max_stop_pips': 50.0,
    },
}

# =============================================================================
# PERFORMANCE AND DEBUG SETTINGS
# =============================================================================

# Performance Settings
VP_PERFORMANCE = {
    'enable_caching': True,               # Cache volume profile calculations
    'cache_ttl_bars': 5,                  # Cache validity in bars
    'max_calculation_time_ms': 100,       # Maximum calculation time (warn if exceeded)
    'use_vectorization': True,            # Use numpy vectorization (faster)
}

# Debug and Logging
VP_DEBUG = {
    'enabled': True,                      # Enable debug logging
    'log_calculations': False,            # Log detailed calculations (verbose)
    'log_signal_generation': True,        # Log signal generation process
    'log_confidence_breakdown': True,     # Log confidence score breakdown
    'log_filtered_signals': True,         # Log signals that were filtered out
    'show_profile_summary': True,         # Show volume profile summary
}

# Strategy Integration Settings
VP_STRATEGY_WEIGHT = 0.15                 # Weight in combined strategy mode
VP_ALLOW_COMBINED = True                  # Allow in combined strategies
VP_PRIORITY_LEVEL = 2                     # Priority level (1=highest, 5=lowest)

# Backtesting Settings
VP_ENABLE_BACKTESTING = True              # Enable strategy in backtests
VP_MIN_DATA_PERIODS = 100                 # Minimum data periods required
VP_ENABLE_PERFORMANCE_TRACKING = True     # Track strategy performance

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_vp_config_for_epic(epic: str, preset: str = None) -> dict:
    """
    Get Volume Profile configuration for specific epic and preset.

    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.MINI.IP')
        preset: Configuration preset name (default: ACTIVE_VP_CONFIG)

    Returns:
        Dictionary with merged configuration (preset + pair-specific overrides)
    """
    # Use active config if preset not specified
    config_name = preset if preset else ACTIVE_VP_CONFIG
    config = VP_STRATEGY_CONFIG.get(config_name, VP_STRATEGY_CONFIG['default']).copy()

    # Extract currency pair from epic
    pair = extract_pair_from_epic(epic)

    # Apply pair-specific overrides
    if pair in VP_PAIR_CONFIG:
        pair_config = VP_PAIR_CONFIG[pair]
        config.update(pair_config)

    return config

def extract_pair_from_epic(epic: str) -> str:
    """
    Extract currency pair from epic code.

    Args:
        epic: Epic code (e.g., 'CS.D.EURUSD.MINI.IP')

    Returns:
        Currency pair (e.g., 'EURUSD')
    """
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            return epic.replace('CS.D.', '').replace('.MINI.IP', '')
        elif 'CS.D.' in epic and '.CEEM.IP' in epic:
            return epic.replace('CS.D.', '').replace('.CEEM.IP', '')
        else:
            return epic
    except Exception:
        return epic

def get_vp_lookback_periods(config_name: str = None) -> int:
    """
    Get lookback periods for volume profile calculation.

    Args:
        config_name: Lookback configuration name (default: ACTIVE_VP_LOOKBACK)

    Returns:
        Number of bars for lookback
    """
    lookback_name = config_name if config_name else ACTIVE_VP_LOOKBACK
    return VP_LOOKBACK_PERIODS.get(lookback_name, VP_LOOKBACK_PERIODS['default'])

def get_vp_proximity_threshold(epic: str, zone_type: str, preset: str = None) -> float:
    """
    Get proximity threshold for specific epic and zone type.

    Args:
        epic: Trading pair epic
        zone_type: 'hvn', 'lvn', 'poc', or 'vah_val'
        preset: Configuration preset name (default: ACTIVE_VP_CONFIG)

    Returns:
        Proximity threshold in pips
    """
    config = get_vp_config_for_epic(epic, preset)
    threshold_key = f'{zone_type}_proximity_threshold_pips'

    # Fallback to default proximity filters if not in config
    if threshold_key not in config:
        return VP_PROXIMITY_FILTERS.get(f'{zone_type}_threshold_pips', 5.0)

    return config.get(threshold_key, 5.0)

def get_vp_stop_loss_config(epic: str, preset: str = None) -> dict:
    """
    Get stop loss configuration for specific epic.

    Args:
        epic: Trading pair epic
        preset: Configuration preset name

    Returns:
        Stop loss configuration dictionary
    """
    pair_config = get_vp_config_for_epic(epic, preset)

    # Merge with default stop loss config
    stop_config = VP_STOP_LOSS_CONFIG.copy()

    # Override with pair-specific settings if available
    if 'min_stop_pips' in pair_config:
        stop_config['min_stop_pips'] = pair_config['min_stop_pips']
    if 'max_stop_pips' in pair_config:
        stop_config['max_stop_pips'] = pair_config['max_stop_pips']

    return stop_config

def get_vp_take_profit_config(epic: str, preset: str = None) -> dict:
    """
    Get take profit configuration for specific epic.

    Args:
        epic: Trading pair epic
        preset: Configuration preset name

    Returns:
        Take profit configuration dictionary
    """
    pair_config = get_vp_config_for_epic(epic, preset)

    # Merge with default take profit config
    tp_config = VP_TAKE_PROFIT_CONFIG.copy()

    # Override with pair-specific settings if available
    if 'max_target_pips' in pair_config:
        tp_config['max_target_pips'] = pair_config['max_target_pips']

    return tp_config

def set_vp_preset(preset_name: str) -> bool:
    """
    Set the active Volume Profile preset.

    Args:
        preset_name: Name of preset to activate

    Returns:
        True if preset exists and was set, False otherwise
    """
    global ACTIVE_VP_CONFIG
    if preset_name in VP_STRATEGY_CONFIG:
        ACTIVE_VP_CONFIG = preset_name
        return True
    return False

def get_available_vp_presets() -> dict:
    """
    Get all available Volume Profile presets with descriptions.

    Returns:
        Dictionary of preset names to descriptions
    """
    return {
        name: {
            'description': config['description'],
            'best_for': config.get('best_for', []),
            'best_session': config.get('best_session', []),
        }
        for name, config in VP_STRATEGY_CONFIG.items()
    }

def get_vp_config_summary() -> dict:
    """
    Get a summary of Volume Profile configuration settings.

    Returns:
        Dictionary with configuration summary
    """
    active_config = VP_STRATEGY_CONFIG.get(ACTIVE_VP_CONFIG, VP_STRATEGY_CONFIG['default'])

    return {
        'strategy_enabled': VOLUME_PROFILE_STRATEGY,
        'active_preset': ACTIVE_VP_CONFIG,
        'preset_description': active_config.get('description', 'No description'),
        'lookback_periods': get_vp_lookback_periods(),
        'min_confidence': active_config.get('min_confidence', 0.60),
        'hvn_detection_enabled': VP_HVN_DETECTION['enabled'],
        'lvn_detection_enabled': VP_LVN_DETECTION['enabled'],
        'value_area_percent': VP_VALUE_AREA_PERCENT,
        'signal_types_enabled': {
            name: config['enabled']
            for name, config in VP_SIGNAL_TYPES.items()
        },
        'debug_logging': VP_DEBUG['enabled'],
        'performance_tracking': VP_ENABLE_PERFORMANCE_TRACKING,
        'backtesting_enabled': VP_ENABLE_BACKTESTING,
        'total_presets': len(VP_STRATEGY_CONFIG),
    }
