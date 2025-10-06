# configdata/strategies/config_kama_strategy.py
"""
KAMA (Kaufman's Adaptive Moving Average) Strategy Configuration
===============================================================

KAMA is a trend-following indicator that automatically adapts to market volatility.
The indicator uses an Efficiency Ratio (ER) to modify the smoothing constant,
which ranges between a fast and slow exponential moving average.

Key Features:
- Adaptive smoothing based on market efficiency
- Reduced noise during sideways markets
- Responsive during trending markets
- Multiple configuration presets for different market conditions
"""

# Strategy enable/disable
KAMA_STRATEGY = True  # ENABLED - Phase 1 optimization complete, ready for testing

# Main KAMA Strategy Configuration Dictionary
KAMA_STRATEGY_CONFIG = {
    'default': {
        'period': 10,           # Efficiency Ratio period
        'fast': 2,              # Fast smoothing constant
        'slow': 30,             # Slow smoothing constant
        'description': 'Balanced KAMA configuration for most market conditions',
        'best_for': ['trending', 'medium_volatility', 'swing_trading'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium_to_strong',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'period': 14,           # Longer period for stability
        'fast': 2,              # Standard fast smoothing
        'slow': 50,             # Slower smoothing for less noise
        'description': 'Conservative KAMA for stable trending markets',
        'best_for': ['strong_trends', 'low_volatility', 'position_trading'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 25.0
    },
    'aggressive': {
        'period': 6,            # Shorter period for responsiveness
        'fast': 2,              # Fast smoothing constant
        'slow': 20,             # Faster slow smoothing
        'description': 'Aggressive KAMA for volatile breakout markets',
        'best_for': ['breakouts', 'high_volatility', 'scalping'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'medium',
        'best_market_regime': 'breakout',
        'best_session': ['overlap', 'new_york'],
        'preferred_pairs': ['CS.D.GBPUSD.MINI.IP', 'CS.D.EURJPY.MINI.IP'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 100.0
    },
    'scalping': {
        'period': 5,            # Very short period
        'fast': 2,              # Fast smoothing
        'slow': 15,             # Moderate slow smoothing
        'description': 'Ultra-responsive KAMA for scalping strategies',
        'best_for': ['ranging_markets', 'high_frequency', 'tight_spreads'],
        'best_volatility_regime': 'medium_to_high',
        'best_trend_strength': 'weak_to_medium',
        'best_market_regime': 'ranging',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP']
    },
    'swing': {
        'period': 20,           # Longer period for swing trading
        'fast': 2,              # Standard fast smoothing
        'slow': 60,             # Very slow smoothing
        'description': 'Smooth KAMA for swing trading and position holding',
        'best_for': ['strong_trends', 'position_trading', 'daily_timeframes'],
        'best_volatility_regime': 'low_to_medium',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP']
    },
    'news_safe': {
        'period': 12,           # Moderate period
        'fast': 2,              # Standard fast smoothing
        'slow': 40,             # Slower smoothing for stability
        'description': 'Stable KAMA configuration during news events',
        'best_for': ['news_events', 'high_volatility', 'uncertainty'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'breakout'
    }
}

# Active configuration selector (when not using dynamic mode)
ACTIVE_KAMA_CONFIG = 'default'
DEFAULT_KAMA_CONFIG = 'default'  # Fallback configuration name

# Dynamic configuration management
ENABLE_DYNAMIC_KAMA_CONFIG = True   # Enable dynamic KAMA configuration selection
KAMA_ADAPTIVE_SMOOTHING = True      # Enable adaptive smoothing based on market conditions

# Additional KAMA periods to calculate (optional)
ADDITIONAL_KAMA_PERIODS = [6, 14]   # Will calculate KAMA_6 and KAMA_14 in addition to default

# ============================================================================
# PHASE 1 OPTIMIZATION SETTINGS (2025-10-05)
# ============================================================================
# These values were updated based on comprehensive analysis to improve signal quality

# KAMA Parameters
KAMA_ER_PERIOD = 14                 # Efficiency Ratio calculation period
KAMA_FAST_SC = 2                    # Fast smoothing constant
KAMA_SLOW_SC = 30                   # Slow smoothing constant

# KAMA Signal Generation Thresholds (PHASE 1: TIGHTENED)
KAMA_MIN_EFFICIENCY = 0.20          # Increased from 0.10 - Higher quality bar (Phase 1)
KAMA_HIGH_EFFICIENCY_THRESHOLD = 0.6 # High trending market threshold (60% efficiency)
KAMA_MIN_EFFICIENCY_RATIO = 0.20    # Alias for backward compatibility (TIGHTENED from 0.1)
KAMA_TREND_THRESHOLD = 0.05         # Minimum trend change (5%) for signals

# KAMA Confidence Component Weights (PHASE 1: REBALANCED)
# OLD: ER=0.45, Trend=0.25, Alignment=0.15, Strength=0.10, Context=0.05
KAMA_CONFIDENCE_WEIGHTS = {
    'efficiency_ratio': 0.30,       # Reduced from 0.45 - Less ER dominance
    'trend_strength': 0.25,         # Kept at 0.25 - KAMA trend momentum
    'price_alignment': 0.20,        # Increased from 0.15 - Better entry timing
    'signal_strength': 0.15,        # Increased from 0.10 - Raw quality matters
    'market_context': 0.10          # Increased from 0.05 - Session/regime important
}

# Legacy weight settings (for backward compatibility)
KAMA_TREND_CHANGE_WEIGHT = 0.3      # Weight for trend change signals
KAMA_CROSSOVER_WEIGHT = 0.4         # Weight for price crossover signals
KAMA_ER_WEIGHT = 0.3                # Weight for efficiency ratio

# KAMA Signal Confidence Settings (PHASE 1.5: BOOSTED)
KAMA_MIN_CONFIDENCE = 0.10          # Minimum confidence threshold (10%) - lowered for filtering
KAMA_BASE_CONFIDENCE = 0.80         # Base confidence level for KAMA signals (INCREASED from 0.75)
KAMA_MAX_CONFIDENCE = 0.95          # Maximum confidence cap

# KAMA Validation Settings (PHASE 1: ENHANCED)
KAMA_ENHANCED_VALIDATION = True     # Enable enhanced KAMA signal validation
KAMA_MIN_BARS = 50                  # Minimum bars needed for KAMA calculation
KAMA_REQUIRE_TREND_CONFIRMATION = True # Require trend confirmation

# ADX Validation Settings (PHASE 1: NEW - Critical Addition)
KAMA_ADX_VALIDATION_ENABLED = True  # Enable ADX trend strength validation
KAMA_MIN_ADX = 20                   # Minimum ADX for signal (moderate trend required)
KAMA_STRONG_ADX = 25                # Strong trend confirmation threshold
KAMA_WEAK_ADX_WARNING = 18          # Warning threshold for weak trends

# MACD Validation Settings (PHASE 1: STRENGTHENED)
KAMA_MACD_VALIDATION_ENABLED = True # Enable MACD validation
KAMA_MIN_MACD_THRESHOLD = 0.0003    # Increased from 0.0001 (3x stricter)
KAMA_STRONG_MACD_THRESHOLD = 0.0005 # Strong MACD confirmation threshold

# Strategy Integration Settings
STRATEGY_WEIGHT_KAMA = 0.2          # Weight for KAMA in combined strategy (20%)
INCLUDE_KAMA_IN_COMBINED = True     # Include KAMA in combined strategy logic
KAMA_ALLOW_COMBINED = True          # Allow KAMA in combined strategies
KAMA_PRIORITY_LEVEL = 2             # Priority level (1=highest, 5=lowest)

# Performance Settings
KAMA_ENABLE_BACKTESTING = True      # Enable strategy in backtests
KAMA_MIN_DATA_PERIODS = 100         # Minimum data periods required
KAMA_ENABLE_PERFORMANCE_TRACKING = True # Track strategy performance

# Debug Settings
KAMA_DEBUG_LOGGING = True           # Enable detailed debug logging
KAMA_LOG_EFFICIENCY_RATIO = True    # Log efficiency ratio calculations
KAMA_LOG_SIGNAL_BREAKDOWN = True    # Log detailed signal breakdown

# Multi-Timeframe KAMA Settings
KAMA_MTF_ENABLED = True             # Enable multi-timeframe KAMA validation
KAMA_MTF_TIMEFRAMES = ['5m', '15m', '1h'] # Timeframes for MTF analysis

# ============================================================================
# PHASE 2 ENHANCEMENT SETTINGS (2025-10-05)
# ============================================================================
# Advanced signal quality improvements: Volume, RSI, Dynamic SL/TP, S/R awareness

# Volume Confirmation (Enhancement 1)
KAMA_VOLUME_VALIDATION_ENABLED = True
KAMA_MIN_VOLUME_MULTIPLIER = 1.5      # 50% above 20-period average required
KAMA_STRONG_VOLUME_MULTIPLIER = 2.0   # 100% above average = strong confirmation
KAMA_VOLUME_LOOKBACK_PERIOD = 20      # Volume average calculation period

# RSI Momentum Filter (Enhancement 2)
KAMA_RSI_VALIDATION_ENABLED = True
KAMA_RSI_PERIOD = 14                  # RSI calculation period
KAMA_RSI_OVERBOUGHT = 70              # Reject BULL signals if RSI > 70
KAMA_RSI_OVERSOLD = 30                # Reject BEAR signals if RSI < 30
KAMA_RSI_OPTIMAL_BULL_MIN = 50        # Optimal BULL entry zone start
KAMA_RSI_OPTIMAL_BULL_MAX = 65        # Optimal BULL entry zone end
KAMA_RSI_OPTIMAL_BEAR_MIN = 35        # Optimal BEAR entry zone start
KAMA_RSI_OPTIMAL_BEAR_MAX = 50        # Optimal BEAR entry zone end

# Dynamic SL/TP (Enhancement 3)
KAMA_DYNAMIC_SL_TP_ENABLED = True
KAMA_ATR_PERIOD = 14                  # ATR calculation period
KAMA_BASE_SL_ATR_MULTIPLIER = 2.0     # Base SL: 2x ATR
KAMA_BASE_TP_ATR_MULTIPLIER = 3.0     # Base TP: 3x ATR
KAMA_MIN_RR_RATIO = 1.5               # Minimum risk:reward ratio

# Support/Resistance Awareness (Enhancement 4)
KAMA_SR_VALIDATION_ENABLED = True
KAMA_SR_PROXIMITY_PIPS = 20           # Check within 20 pips of S/R levels
KAMA_MTF_MIN_AGREEMENT = 0.6        # Minimum agreement required (60%)

# KAMA Risk Management
KAMA_MAX_RISK_PER_TRADE = 0.02      # 2% risk per trade
KAMA_STOP_LOSS_METHOD = 'adaptive'  # Options: 'fixed', 'atr', 'adaptive'
KAMA_TAKE_PROFIT_RATIO = 2.5        # 2.5:1 reward:risk ratio
KAMA_DEFAULT_STOP_LOSS_PIPS = 20    # Default stop loss in pips
KAMA_DEFAULT_TAKE_PROFIT_PIPS = 50  # Default take profit in pips

# KAMA Efficiency Ratio Thresholds by Market Condition
KAMA_ER_THRESHOLDS = {
    'trending': 0.4,        # Strong trending markets (40%+ efficiency)
    'moderate': 0.2,        # Moderate trending markets (20-40% efficiency)
    'ranging': 0.1,         # Ranging/choppy markets (10-20% efficiency)
    'noise': 0.05          # Very noisy markets (<10% efficiency)
}

# Pair-specific KAMA adjustments
KAMA_PAIR_ADJUSTMENTS = {
    'EURUSD': {'er_multiplier': 1.0, 'confidence_boost': 0.0},
    'GBPUSD': {'er_multiplier': 0.9, 'confidence_boost': 0.05},  # More volatile, slight boost
    'USDJPY': {'er_multiplier': 1.1, 'confidence_boost': 0.0},   # Less volatile, higher ER needed
    'EURJPY': {'er_multiplier': 0.85, 'confidence_boost': 0.1},  # Cross pair, boost confidence
    'GBPJPY': {'er_multiplier': 0.8, 'confidence_boost': 0.15},  # Very volatile cross
    'USDCHF': {'er_multiplier': 1.2, 'confidence_boost': 0.0},   # Safe haven, stricter
    'DEFAULT': {'er_multiplier': 1.0, 'confidence_boost': 0.0}   # Default for other pairs
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_kama_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """
    Get KAMA configuration for specific epic and market condition

    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        market_condition: Market condition ('default', 'conservative', 'aggressive', etc.)

    Returns:
        Dictionary with KAMA configuration
    """
    # Use dynamic config if enabled, otherwise use active config
    if ENABLE_DYNAMIC_KAMA_CONFIG and market_condition in KAMA_STRATEGY_CONFIG:
        config_name = market_condition
    else:
        config_name = ACTIVE_KAMA_CONFIG

    return KAMA_STRATEGY_CONFIG.get(config_name, KAMA_STRATEGY_CONFIG['default'])

def get_kama_threshold_for_epic(epic: str) -> float:
    """
    Get KAMA-specific efficiency ratio threshold based on currency pair

    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        Efficiency ratio threshold for the pair
    """
    try:
        # Extract currency pair from epic
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic

        # Get pair-specific adjustment
        adjustment = KAMA_PAIR_ADJUSTMENTS.get(pair, KAMA_PAIR_ADJUSTMENTS['DEFAULT'])
        base_threshold = KAMA_MIN_EFFICIENCY_RATIO

        return base_threshold * adjustment['er_multiplier']

    except Exception:
        return KAMA_MIN_EFFICIENCY_RATIO

def get_kama_confidence_adjustment_for_epic(epic: str) -> float:
    """
    Get KAMA confidence adjustment for specific epic

    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        Confidence adjustment value (to be added to base confidence)
    """
    try:
        # Extract currency pair from epic
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic

        # Get pair-specific adjustment
        adjustment = KAMA_PAIR_ADJUSTMENTS.get(pair, KAMA_PAIR_ADJUSTMENTS['DEFAULT'])
        return adjustment['confidence_boost']

    except Exception:
        return 0.0

def calculate_kama_confidence(efficiency_ratio: float, trend_strength: float, crossover_strength: float) -> float:
    """
    Calculate KAMA signal confidence based on multiple factors

    Args:
        efficiency_ratio: KAMA efficiency ratio (0-1)
        trend_strength: Trend strength indicator (0-1)
        crossover_strength: Price/KAMA crossover strength (0-1)

    Returns:
        Confidence score (0-1)
    """
    try:
        # Weighted confidence calculation
        confidence = (
            efficiency_ratio * KAMA_ER_WEIGHT +
            trend_strength * KAMA_TREND_CHANGE_WEIGHT +
            crossover_strength * KAMA_CROSSOVER_WEIGHT
        )

        # Apply bounds
        confidence = max(0.0, min(1.0, confidence))

        # Scale to configured range
        scaled_confidence = KAMA_BASE_CONFIDENCE + (confidence * (KAMA_MAX_CONFIDENCE - KAMA_BASE_CONFIDENCE))

        return min(KAMA_MAX_CONFIDENCE, max(KAMA_MIN_CONFIDENCE, scaled_confidence))

    except Exception:
        return KAMA_BASE_CONFIDENCE

def get_kama_market_regime(efficiency_ratio: float) -> str:
    """
    Determine market regime based on KAMA efficiency ratio

    Args:
        efficiency_ratio: KAMA efficiency ratio (0-1)

    Returns:
        Market regime string ('trending', 'moderate', 'ranging', 'noise')
    """
    if efficiency_ratio >= KAMA_ER_THRESHOLDS['trending']:
        return 'trending'
    elif efficiency_ratio >= KAMA_ER_THRESHOLDS['moderate']:
        return 'moderate'
    elif efficiency_ratio >= KAMA_ER_THRESHOLDS['ranging']:
        return 'ranging'
    else:
        return 'noise'

def validate_kama_config() -> dict:
    """
    Validate KAMA strategy configuration completeness

    Returns:
        Dictionary with validation results
    """
    try:
        # Check required settings
        required_settings = [
            'KAMA_STRATEGY', 'KAMA_STRATEGY_CONFIG', 'ACTIVE_KAMA_CONFIG',
            'KAMA_MIN_EFFICIENCY_RATIO', 'KAMA_BASE_CONFIDENCE'
        ]

        for setting in required_settings:
            if setting not in globals():
                return {'valid': False, 'error': f'Missing required setting: {setting}'}

        # Validate configuration structure
        for config_name, config in KAMA_STRATEGY_CONFIG.items():
            if not isinstance(config, dict):
                return {'valid': False, 'error': f'Config {config_name} must be dict'}

            required_keys = ['period', 'fast', 'slow', 'description']
            for key in required_keys:
                if key not in config:
                    return {'valid': False, 'error': f'Config {config_name} missing {key}'}

            # Validate parameter ranges
            if not (1 <= config['period'] <= 100):
                return {'valid': False, 'error': f'Config {config_name}: period must be 1-100'}

            if not (1 <= config['fast'] <= config['slow']):
                return {'valid': False, 'error': f'Config {config_name}: fast must be <= slow'}

        # Validate active config exists
        if ACTIVE_KAMA_CONFIG not in KAMA_STRATEGY_CONFIG:
            return {'valid': False, 'error': f'Active config {ACTIVE_KAMA_CONFIG} not found'}

        # Validate threshold ranges
        if not (0 <= KAMA_MIN_EFFICIENCY_RATIO <= 1):
            return {'valid': False, 'error': 'KAMA_MIN_EFFICIENCY_RATIO must be 0-1'}

        if not (0 <= KAMA_BASE_CONFIDENCE <= 1):
            return {'valid': False, 'error': 'KAMA_BASE_CONFIDENCE must be 0-1'}

        return {
            'valid': True,
            'strategy_enabled': KAMA_STRATEGY,
            'config_count': len(KAMA_STRATEGY_CONFIG),
            'active_config': ACTIVE_KAMA_CONFIG,
            'dynamic_config_enabled': ENABLE_DYNAMIC_KAMA_CONFIG,
            'mtf_enabled': KAMA_MTF_ENABLED,
            'enhanced_validation': KAMA_ENHANCED_VALIDATION
        }

    except Exception as e:
        return {'valid': False, 'error': f'Validation exception: {str(e)}'}

def get_kama_config_summary() -> dict:
    """Get a summary of KAMA configuration settings"""
    return {
        'strategy_enabled': KAMA_STRATEGY,
        'active_config': ACTIVE_KAMA_CONFIG,
        'dynamic_config_enabled': ENABLE_DYNAMIC_KAMA_CONFIG,
        'min_efficiency_ratio': KAMA_MIN_EFFICIENCY_RATIO,
        'base_confidence': KAMA_BASE_CONFIDENCE,
        'min_confidence': KAMA_MIN_CONFIDENCE,
        'mtf_enabled': KAMA_MTF_ENABLED,
        'mtf_timeframes': KAMA_MTF_TIMEFRAMES if KAMA_MTF_ENABLED else [],
        'enhanced_validation': KAMA_ENHANCED_VALIDATION,
        'total_configurations': len(KAMA_STRATEGY_CONFIG),
        'available_configs': list(KAMA_STRATEGY_CONFIG.keys()),
        'strategy_weight': STRATEGY_WEIGHT_KAMA,
        'include_in_combined': INCLUDE_KAMA_IN_COMBINED,
        'debug_logging': KAMA_DEBUG_LOGGING
    }

# Configuration presets for different signal generation frequencies
KAMA_SIGNAL_FREQUENCY_PRESETS = {
    'conservative': {
        'min_efficiency_ratio': 0.2,
        'min_confidence': 0.7,
        'active_config': 'conservative',
        'description': 'Fewer, higher quality signals - 10-20 per week'
    },
    'balanced': {
        'min_efficiency_ratio': 0.15,
        'min_confidence': 0.6,
        'active_config': 'default',
        'description': 'Balanced signal frequency - 20-35 per week'
    },
    'aggressive': {
        'min_efficiency_ratio': 0.1,
        'min_confidence': 0.5,
        'active_config': 'aggressive',
        'description': 'More frequent signals - 35-50 per week'
    }
}

def set_kama_frequency_preset(preset: str = 'balanced'):
    """
    Apply a KAMA signal frequency preset

    Args:
        preset: 'conservative', 'balanced', or 'aggressive'
    """
    if preset in KAMA_SIGNAL_FREQUENCY_PRESETS:
        preset_config = KAMA_SIGNAL_FREQUENCY_PRESETS[preset]

        # Apply preset settings (these would be applied at runtime)
        return {
            'preset_applied': preset,
            'settings': preset_config,
            'description': preset_config['description']
        }
    else:
        return {
            'error': f'Unknown preset: {preset}',
            'available_presets': list(KAMA_SIGNAL_FREQUENCY_PRESETS.keys())
        }

print("✅ KAMA Strategy configuration loaded successfully")