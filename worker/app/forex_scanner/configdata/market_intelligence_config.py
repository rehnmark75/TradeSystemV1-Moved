# configdata/market_intelligence_config.py
"""
Market Intelligence Configuration
Centralized configuration for all market intelligence features, analysis, and filtering.
"""

# =============================================================================
# MARKET INTELLIGENCE CORE CONFIGURATION
# =============================================================================

# IMMEDIATE FIX: Start with minimal intelligence to get signals flowing
INTELLIGENCE_PRESET = 'minimal'  # Change from default restrictive settings

# Available presets (order from most to least permissive):
# - 'disabled': No intelligence filtering (maximum signals)
# - 'minimal': Volume + confidence only (many signals)
# - 'balanced': Moderate intelligence (some signals)
# - 'conservative': Full intelligence (few signals)
# - 'testing': Consistent with backtesting

# Intelligence mode selection
INTELLIGENCE_MODE = 'live_only'  # Options: 'disabled', 'backtest_consistent', 'live_only', 'enhanced', balanced
ENABLE_MARKET_INTELLIGENCE = True

# =============================================================================
# INTELLIGENCE PRESETS CONFIGURATION
# =============================================================================

INTELLIGENCE_PRESETS = {
    'disabled': {
        'threshold': 0.0,
        'use_intelligence_engine': False,
        'components_enabled': {
            'market_regime_filter': False,
            'volatility_filter': False,
            'volume_filter': False,
            'time_filter': False,
            'confidence_filter': True,  # Keep minimal confidence check
        },
        'description': 'No intelligence filtering - strategy signals only'
    },

    'minimal': {
        'threshold': 0.3,  # Low threshold for more signals
        'use_intelligence_engine': False,
        'components_enabled': {
            'market_regime_filter': False,
            'volatility_filter': False,
            'volume_filter': True,      # Basic volume check
            'time_filter': False,
            'confidence_filter': True,   # Basic confidence check
        },
        'description': 'Minimal filtering - volume + confidence only'
    },

    'balanced': {
        'threshold': 0.5,  # Medium threshold
        'use_intelligence_engine': True,
        'components_enabled': {
            'market_regime_filter': True,
            'volatility_filter': True,
            'volume_filter': True,
            'time_filter': False,        # Often too restrictive
            'confidence_filter': True,
        },
        'description': 'Balanced filtering with market intelligence'
    },

    'conservative': {
        'threshold': 0.7,  # High threshold for fewer, higher quality signals
        'use_intelligence_engine': True,
        'components_enabled': {
            'market_regime_filter': True,
            'volatility_filter': True,
            'volume_filter': True,
            'time_filter': True,
            'confidence_filter': True,
        },
        'description': 'Conservative filtering - fewer but higher quality signals'
    },

    'testing': {
        'threshold': 0.4,  # Consistent with backtesting
        'use_intelligence_engine': False,
        'components_enabled': {
            'market_regime_filter': False,
            'volatility_filter': True,
            'volume_filter': True,
            'time_filter': False,
            'confidence_filter': True,
        },
        'description': 'Consistent with backtesting environment'
    }
}

# =============================================================================
# PROBABILISTIC CONFIDENCE MODIFIER SYSTEM
# =============================================================================

# NEW: Regime-Strategy Confidence Modifiers (replaces binary blocking)
# Values between 0.2-1.0 where:
# 1.0 = Perfect compatibility (no adjustment)
# 0.8-0.9 = Good compatibility (slight reduction)
# 0.6-0.7 = Moderate compatibility (medium reduction)
# 0.4-0.5 = Poor compatibility (significant reduction)
# 0.2-0.3 = Very poor compatibility (heavy reduction but not blocked)

REGIME_STRATEGY_CONFIDENCE_MODIFIERS = {
    'trending': {
        'macd': 1.0,           # Perfect regime for MACD
        'ema': 1.0,
        'ichimoku': 1.0,
        'kama': 1.0,
        'momentum': 0.9,
        'zero_lag': 0.9,
        'mean_reversion': 0.4,  # Poor fit for trending
        'bollinger': 0.6,
        'ranging_market': 0.3
    },
    'ranging': {
        'macd': 0.8,           # GOOD - especially with divergence analysis
        'mean_reversion': 1.0,  # Perfect for ranging
        'bollinger': 1.0,
        'stochastic': 1.0,
        'ranging_market': 1.0,
        'smc': 1.0,
        'ema': 0.6,            # Moderate - can work in ranges
        'ichimoku': 0.5,       # Poor fit for ranging
        'momentum': 0.4
    },
    'breakout': {
        'macd': 0.9,           # EXCELLENT - confirms momentum expansion
        'bollinger': 1.0,      # Perfect for breakouts
        'kama': 1.0,
        'momentum': 1.0,
        'momentum_bias': 1.0,
        'bb_supertrend': 1.0,
        'zero_lag': 0.9,
        'ema': 0.8,
        'mean_reversion': 0.3,  # Very poor for breakouts
        'ranging_market': 0.2
    },
    'consolidation': {
        'macd': 0.75,          # MODERATE - can detect subtle momentum shifts
        'mean_reversion': 1.0,  # Perfect for consolidation
        'stochastic': 1.0,
        'ranging_market': 1.0,
        'smc': 1.0,
        'bollinger': 0.8,
        'ema': 0.6,
        'momentum': 0.4,
        'zero_lag': 0.5
    },
    'high_volatility': {
        'macd': 1.0,           # Perfect - thrives in volatile conditions
        'zero_lag_squeeze': 1.0,
        'zero_lag': 1.0,
        'momentum': 1.0,
        'kama': 1.0,
        'ema': 1.0,
        'momentum_bias': 1.0,
        'bb_supertrend': 1.0,
        'ichimoku': 0.8,
        'mean_reversion': 0.4,  # Poor in high volatility
        'ranging_market': 0.3
    },
    'low_volatility': {
        'macd': 0.85,          # GOOD - works with tighter thresholds
        'mean_reversion': 1.0,  # Perfect for low vol
        'bollinger': 1.0,
        'stochastic': 1.0,
        'ema': 1.0,
        'ranging_market': 1.0,
        'smc': 1.0,
        'ichimoku': 0.8,
        'momentum': 0.5,        # Poor in low volatility
        'zero_lag': 0.6
    },
    'medium_volatility': {
        'macd': 1.0,           # Perfect - ideal conditions for MACD
        'ichimoku': 1.0,
        'ema': 1.0,
        'kama': 1.0,
        'zero_lag_squeeze': 1.0,
        'zero_lag': 1.0,
        'smart_money_ema': 1.0,
        'smart_money_macd': 1.0,
        'momentum': 0.9,
        'bollinger': 0.8
    },
    'scalping': {
        'macd': 0.6,           # Moderate - can work with very tight parameters
        'scalping': 1.0,        # Perfect for scalping strategies
        'zero_lag': 1.0,
        'momentum_bias': 1.0,
        'ema': 0.8,
        'mean_reversion': 0.4,
        'ichimoku': 0.3
    }
}

# Enable/disable the probabilistic confidence modifier system
ENABLE_PROBABILISTIC_CONFIDENCE_MODIFIERS = True

# Minimum confidence modifier threshold (signals below this are still blocked)
MIN_CONFIDENCE_MODIFIER = 0.2

# =============================================================================
# INTELLIGENCE THRESHOLDS AND WEIGHTS
# =============================================================================

# Override specific thresholds if needed
INTELLIGENCE_THRESHOLDS = {
    'disabled': 0.0,           # Always pass
    'live_only': 0.2,          # Low threshold (was 0.7 - too high!)
    'balanced': 0.3,          # Low threshold (was 0.7 - too high!)
    'enhanced': 0.5,           # Medium threshold (was 0.8 - way too high!)
    'backtest_consistent': 0.4  # Consistent with backtesting
}

# Component enable/disable (start conservative)
INTELLIGENCE_COMPONENTS_ENABLED = {
    'market_regime_filter': False,  # Disable initially - often too restrictive
    'volatility_filter': True,     # Keep - helps filter bad volatility
    'volume_filter': True,         # Keep - volume confirmation useful
    'time_filter': False,          # Disable initially - often too restrictive
    'confidence_filter': True,     # Keep - basic signal quality
    'spread_filter': True,         # Keep - prevents bad spreads
    'recent_signals_filter': True  # Keep - prevents spam
}

# Weights for enabled components
INTELLIGENCE_WEIGHTS = {
    'market_regime': 0.25,
    'volatility': 0.25,
    'volume': 0.25,
    'time': 0.0,      # Disabled
    'confidence': 0.25
}

# =============================================================================
# MARKET INTELLIGENCE STORAGE AND PROCESSING
# =============================================================================

# Market Intelligence Storage Configuration
ENABLE_MARKET_INTELLIGENCE_STORAGE = True  # Store market intelligence for every scan cycle
MARKET_INTELLIGENCE_CLEANUP_DAYS = 30      # Auto-cleanup old records after N days

# Market Intelligence Trade Filtering Configuration
ENABLE_MARKET_INTELLIGENCE_FILTERING = True   # 🔧 RE-ENABLED: Test evening signal filtering with market intelligence
MARKET_INTELLIGENCE_MIN_CONFIDENCE = 0.55    # Minimum confidence threshold for market regime analysis
MARKET_INTELLIGENCE_BLOCK_UNSUITABLE_REGIMES = True   # 🔧 RE-ENABLED: Test regime-strategy compatibility blocking

# =============================================================================
# INTELLIGENCE ANALYSIS CONFIGURATION
# =============================================================================

# Force analysis even when market is closed
FORCE_INTELLIGENCE_ANALYSIS = True
INTELLIGENCE_OVERRIDE_MARKET_HOURS = True
USE_HISTORICAL_DATA_FOR_INTELLIGENCE = True

# Make intelligence less aggressive (for testing)
INTELLIGENCE_CONFIDENCE_THRESHOLD = 0.3  # Lower threshold
INTELLIGENCE_VOLUME_THRESHOLD = 0.2      # Lower volume requirement
INTELLIGENCE_VOLATILITY_MIN = 0.1        # Allow low volatility

# Allow scanning in different market conditions
INTELLIGENCE_ALLOW_LOW_VOLATILITY = True
INTELLIGENCE_ALLOW_RANGING_MARKETS = True
INTELLIGENCE_MODE_OVERRIDE = 'analyze_only'  # Analyze but don't filter

# Market conditions override for testing
FORCE_MARKET_CONDITIONS = {
    'volatility_regime': 'medium',
    'trend_strength': 'medium',
    'market_regime': 'trending'
}

# =============================================================================
# LIVE SCANNER AND DATA SETTINGS
# =============================================================================

# Live scanner data settings (to use recent historical data)
LIVE_SCANNER_LOOKBACK_HOURS = 24
ENABLE_RECENT_HISTORICAL_SCAN = True
USE_BACKTEST_DATA_LOGIC_FOR_LIVE = True
FORCE_MARKET_OPEN = True
ENABLE_WEEKEND_SCANNING = True
MAX_DATA_AGE_MINUTES = 60
MINIMUM_CANDLES_FOR_LIVE_SCAN = 50

# =============================================================================
# DEBUGGING AND MONITORING
# =============================================================================

# Debugging and monitoring
INTELLIGENCE_DEBUG_MODE = True    # Log detailed intelligence decisions
INTELLIGENCE_LOG_REJECTIONS = True  # Log why signals were rejected

# =============================================================================
# CLAUDE INTELLIGENCE INTEGRATION
# =============================================================================

# Integration with Claude AI analysis
CLAUDE_INTEGRATE_WITH_INTELLIGENCE = True    # Integrate with market intelligence
CLAUDE_INTELLIGENCE_INTEGRATION = {
    'use_intelligence_in_claude': True,      # Include intelligence data in Claude prompts
    'claude_override_intelligence': True,    # Allow Claude to override intelligence filtering
    'intelligence_claude_weight': 0.3        # Weight of Claude vs intelligence
}

# Order execution intelligence mode
DEFAULT_INTELLIGENCE_MODE = 'disabled'  # Options: 'disabled', 'backtest_consistent'

# =============================================================================
# INTELLIGENCE PRESET SWITCHING FUNCTION
# =============================================================================

def set_intelligence_preset(preset_name: str):
    """
    Quick function to change intelligence settings
    Call this from your scanner or commands
    """
    global INTELLIGENCE_PRESET, INTELLIGENCE_MODE, INTELLIGENCE_THRESHOLDS

    presets = {
        'disabled': {
            'mode': 'disabled',
            'threshold_override': 0.0,
            'components': {comp: False for comp in INTELLIGENCE_COMPONENTS_ENABLED}
        },
        'minimal': {
            'mode': 'live_only',
            'threshold_override': 0.3,
            'components': {
                'market_regime_filter': False,
                'volatility_filter': False,
                'volume_filter': True,
                'time_filter': False,
                'confidence_filter': True,
                'spread_filter': True,
                'recent_signals_filter': True
            }
        },
        'balanced': {
            'mode': 'live_only',
            'threshold_override': 0.5,
            'components': {
                'market_regime_filter': True,
                'volatility_filter': True,
                'volume_filter': True,
                'time_filter': False,  # Still disable time filter
                'confidence_filter': True,
                'spread_filter': True,
                'recent_signals_filter': True
            }
        }
    }

    if preset_name in presets:
        preset = presets[preset_name]
        INTELLIGENCE_PRESET = preset_name
        INTELLIGENCE_MODE = preset['mode']
        INTELLIGENCE_THRESHOLDS[preset['mode']] = preset['threshold_override']
        INTELLIGENCE_COMPONENTS_ENABLED.update(preset['components'])

        print(f"✅ Intelligence preset changed to: {preset_name}")
        print(f"   Mode: {INTELLIGENCE_MODE}")
        print(f"   Threshold: {preset['threshold_override']:.1%}")
        enabled = [k.replace('_filter', '') for k, v in preset['components'].items() if v]
        print(f"   Enabled: {', '.join(enabled)}")
    else:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"   Available: {', '.join(presets.keys())}")

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_market_intelligence_config() -> dict:
    """
    Validate market intelligence configuration completeness and correctness
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    try:
        # Check required configurations exist
        required_configs = [
            'INTELLIGENCE_PRESET',
            'INTELLIGENCE_MODE',
            'ENABLE_MARKET_INTELLIGENCE',
            'INTELLIGENCE_PRESETS',
            'INTELLIGENCE_THRESHOLDS',
            'INTELLIGENCE_COMPONENTS_ENABLED',
            'INTELLIGENCE_WEIGHTS'
        ]

        for config_name in required_configs:
            if config_name not in globals():
                validation_result['errors'].append(f"Missing required config: {config_name}")
                validation_result['valid'] = False

        # Validate preset exists
        if INTELLIGENCE_PRESET not in INTELLIGENCE_PRESETS:
            validation_result['errors'].append(f"Invalid preset: {INTELLIGENCE_PRESET}")
            validation_result['valid'] = False

        # Validate thresholds are between 0 and 1
        for mode, threshold in INTELLIGENCE_THRESHOLDS.items():
            if not (0 <= threshold <= 1):
                validation_result['errors'].append(f"Invalid threshold for {mode}: {threshold}")
                validation_result['valid'] = False

        # Validate weights sum appropriately
        total_weight = sum(INTELLIGENCE_WEIGHTS.values())
        if total_weight == 0:
            validation_result['warnings'].append("All intelligence weights are zero")

        # Check confidence thresholds are reasonable
        if INTELLIGENCE_CONFIDENCE_THRESHOLD < 0 or INTELLIGENCE_CONFIDENCE_THRESHOLD > 1:
            validation_result['errors'].append("Invalid confidence threshold")
            validation_result['valid'] = False

    except Exception as e:
        validation_result['errors'].append(f"Validation error: {str(e)}")
        validation_result['valid'] = False

    return validation_result

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def get_market_intelligence_config_summary() -> dict:
    """
    Get a summary of current market intelligence configuration
    """
    enabled_components = [
        comp.replace('_filter', '')
        for comp, enabled in INTELLIGENCE_COMPONENTS_ENABLED.items()
        if enabled
    ]

    return {
        'preset': INTELLIGENCE_PRESET,
        'mode': INTELLIGENCE_MODE,
        'enabled': ENABLE_MARKET_INTELLIGENCE,
        'threshold': INTELLIGENCE_THRESHOLDS.get(INTELLIGENCE_MODE, 0.5),
        'enabled_components': enabled_components,
        'component_count': len(enabled_components),
        'storage_enabled': ENABLE_MARKET_INTELLIGENCE_STORAGE,
        'filtering_enabled': ENABLE_MARKET_INTELLIGENCE_FILTERING,
        'claude_integration': CLAUDE_INTEGRATE_WITH_INTELLIGENCE,
        'debug_mode': INTELLIGENCE_DEBUG_MODE
    }

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
"""
# Import market intelligence settings
from configdata.market_intelligence_config import *

# Or override specific settings:
INTELLIGENCE_PRESET = 'balanced'
ENABLE_MARKET_INTELLIGENCE = True
INTELLIGENCE_DEBUG_MODE = False

# Change preset programmatically:
set_intelligence_preset('minimal')

# Get configuration summary:
summary = get_market_intelligence_config_summary()
print(f"Intelligence Mode: {summary['mode']}")
print(f"Enabled Components: {', '.join(summary['enabled_components'])}")
"""

# Validate configuration on import
if __name__ != "__main__":
    validation = validate_market_intelligence_config()
    if not validation['valid']:
        print("⚠️ Market Intelligence Configuration Issues:")
        for error in validation['errors']:
            print(f"   ❌ {error}")
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"   ⚠️ {warning}")