# configdata/market_intelligence_config.py
"""
LEGACY - Market Intelligence Configuration

!! DEPRECATED AS OF 2026-01-03 !!

Intelligence settings have been MIGRATED to the database:
- Database: strategy_config
- Tables: intelligence_global_config, intelligence_presets, intelligence_regime_modifiers

To configure intelligence settings:
1. Use Streamlit UI: Settings -> Intelligence Config tab
2. Or directly in database:
   docker exec postgres psql -U postgres -d strategy_config -c "
   SELECT parameter_name, parameter_value FROM intelligence_global_config ORDER BY category, display_order;"

This file is kept ONLY as a fallback when database is unavailable.
The IntelligenceConfigService reads from database first, then falls back to this file.

For the current active preset and settings, check the database:
   docker exec postgres psql -U postgres -d strategy_config -c "
   SELECT parameter_value FROM intelligence_global_config WHERE parameter_name = 'intelligence_preset';"
"""

# =============================================================================
# LEGACY FALLBACK CONFIGURATION (only used when database is unavailable)
# =============================================================================

# Default preset (database overrides this)
INTELLIGENCE_PRESET = 'collect_only'  # collect_only = full engine, no filtering

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
    },

    'collect_only': {
        'threshold': 0.0,  # No filtering - collect data only
        'use_intelligence_engine': True,  # Run engine for data collection
        'components_enabled': {
            'market_regime_filter': False,
            'volatility_filter': False,
            'volume_filter': False,
            'time_filter': False,
            'confidence_filter': False,
        },
        'description': 'Full engine running for data collection, no signal filtering'
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
        # PRIORITY: EMA - foundational trend strategy
        'ema': 1.0,             # PERFECT - EMA is THE trend indicator
        'smart_money_ema': 1.0, # PERFECT - EMA with smart money
        'ema_double': 1.0,      # PERFECT - EMA with crossover confirmation + FVG + ADX

        # Trend-following strategies (excellent in trending markets)
        'macd': 1.0,            # Perfect - MACD follows trends excellently
        'ichimoku': 1.0,        # Perfect - cloud trend analysis
        'kama': 1.0,            # Perfect - adaptive moving average
        'momentum': 0.95,       # Excellent - momentum follows trends
        'zero_lag': 0.95,       # Excellent - fast trend response
        'bb_supertrend': 0.9,   # Very good - combines trend + volatility
        'smart_money_macd': 0.95, # Excellent - MACD with smart money

        # Mean-reverting strategies (poor in trending markets)
        'mean_reversion': 0.3,  # Poor - fights the trend
        'ranging_market': 0.25, # Very poor - designed for ranges
        'bollinger': 0.5,       # Moderate - can work with trend breakouts
        'smc': 0.7,             # Good - smart money can detect trends
        'scalping': 0.4         # Poor - trends too strong for scalping
    },
    'ranging': {
        # PRIORITY: EMA - still important for range boundaries
        'ema': 0.8,             # GOOD - defines range boundaries effectively
        'smart_money_ema': 0.85, # Good - EMA with smart money context
        'ema_double': 0.6,      # MODERATE - ADX filter helps avoid ranging markets

        # Range-specific strategies (excellent in ranging markets)
        'mean_reversion': 1.0,  # PERFECT - designed for ranges
        'ranging_market': 1.0,  # PERFECT - specifically for ranges
        'bollinger': 1.0,       # PERFECT - mean reversion in bands
        'smc': 1.0,             # PERFECT - smart money excels in ranges
        'scalping': 0.9,        # Excellent - ranges perfect for scalping

        # Trend-following in ranges (moderate to poor)
        'macd': 0.4,            # POOR - momentum crossovers generate false signals in ranging markets
        'smart_money_macd': 0.45, # POOR - slightly better with smart money but still unsuitable
        'ichimoku': 0.6,        # Moderate - cloud can define ranges
        'kama': 0.7,            # Good - adapts to ranging conditions
        'momentum': 0.5,        # Poor - little momentum in ranges
        'zero_lag': 0.6,        # Moderate - can catch range breaks
        'bb_supertrend': 0.7    # Good - trend component helps
    },
    'breakout': {
        # PRIORITY: EMA - excellent for confirming breakouts
        'ema': 0.95,            # EXCELLENT - EMA crossovers confirm breakouts
        'smart_money_ema': 0.95, # Excellent - EMA with smart money confirmation
        'ema_double': 0.95,     # EXCELLENT - FVG confirms breakout momentum

        # Breakout-specific strategies (excellent)
        'bb_supertrend': 1.0,   # PERFECT - designed for breakouts
        'momentum': 1.0,        # PERFECT - momentum drives breakouts
        'zero_lag': 0.95,       # Excellent - fast breakout detection
        'kama': 0.9,            # Excellent - adapts to breakout volatility
        'bollinger': 0.9,       # Excellent - band breakouts

        # Moderate performers in breakouts
        'macd': 0.9,            # Very good - momentum confirmation
        'smart_money_macd': 0.9, # Very good - MACD with smart money
        'ichimoku': 0.8,        # Good - cloud breakouts
        'smc': 0.85,            # Very good - smart money breakout detection
        'scalping': 0.6,        # Moderate - volatility can help

        # Poor performers in breakouts
        'mean_reversion': 0.2,  # Very poor - fights breakout momentum
        'ranging_market': 0.15  # Very poor - opposite of breakouts
    },
    'consolidation': {
        # PRIORITY: EMA - moderate performance in consolidation
        'ema': 0.7,             # GOOD - can define consolidation boundaries
        'smart_money_ema': 0.75, # Good - EMA with smart money insight
        'ema_double': 0.5,      # POOR - ADX filters out but some false signals possible

        # Consolidation specialists (excellent)
        'mean_reversion': 1.0,  # PERFECT - consolidation = mean reversion
        'ranging_market': 1.0,  # PERFECT - similar to ranging
        'smc': 1.0,             # PERFECT - smart money excels here
        'scalping': 0.85,       # Very good - tight ranges for scalping
        'bollinger': 0.8,       # Good - mean reversion in consolidation

        # Moderate performers
        'macd': 0.45,           # POOR - consolidation lacks momentum for reliable MACD signals
        'smart_money_macd': 0.5, # MODERATE - slightly better with smart money
        'ichimoku': 0.7,        # Good - cloud consolidation patterns
        'kama': 0.7,            # Good - adapts to low volatility

        # Poor performers
        'momentum': 0.4,        # Poor - little momentum in consolidation
        'zero_lag': 0.5,        # Poor - needs more movement
        'bb_supertrend': 0.6    # Moderate - trend component struggles
    },
    'high_volatility': {
        # PRIORITY: EMA - excellent in high volatility
        'ema': 1.0,             # PERFECT - EMA handles volatility well
        'smart_money_ema': 1.0, # PERFECT - EMA with smart money
        'ema_double': 0.9,      # EXCELLENT - confirmation filters help in volatile markets

        # High volatility specialists (excellent)
        'zero_lag': 1.0,        # PERFECT - designed for fast markets
        'momentum': 1.0,        # PERFECT - volatility creates momentum
        'macd': 0.6,            # MODERATE - high vol alone doesn't guarantee trending (can be choppy ranging)
        'smart_money_macd': 0.65, # MODERATE - slightly better with smart money context
        'kama': 1.0,            # PERFECT - adapts to high volatility
        'bb_supertrend': 1.0,   # PERFECT - volatility breakouts
        'scalping': 0.9,        # Excellent - volatility creates opportunities

        # Moderate performers
        'ichimoku': 0.8,        # Good - cloud analysis works
        'smc': 0.85,            # Very good - smart money in volatility
        'bollinger': 0.8,       # Good - bands expand with volatility

        # Poor performers
        'mean_reversion': 0.3,  # Poor - volatility fights mean reversion
        'ranging_market': 0.2   # Very poor - opposite of ranging
    },
    'low_volatility': {
        # PRIORITY: EMA - excellent in low volatility
        'ema': 1.0,             # PERFECT - smooth trends in low vol
        'smart_money_ema': 1.0, # PERFECT - EMA with smart money
        'ema_double': 0.85,     # VERY GOOD - works well in smooth low-vol trends

        # Low volatility specialists (excellent)
        'mean_reversion': 1.0,  # PERFECT - low vol enables mean reversion
        'ranging_market': 1.0,  # PERFECT - low vol creates ranges
        'bollinger': 1.0,       # PERFECT - tight bands in low vol
        'smc': 1.0,             # PERFECT - smart money in quiet markets
        'scalping': 0.8,        # Good - tight spreads in low vol

        # Moderate performers
        'macd': 0.85,           # Good - works with tighter thresholds
        'smart_money_macd': 0.85, # Good - MACD with smart money
        'ichimoku': 0.8,        # Good - cloud analysis still works
        'kama': 0.8,            # Good - adapts to low volatility

        # Poor performers
        'momentum': 0.4,        # Poor - little momentum in low vol
        'zero_lag': 0.5,        # Poor - needs more movement
        'bb_supertrend': 0.6    # Moderate - trend component struggles
    },
    'medium_volatility': {
        # PRIORITY: EMA - perfect in medium volatility
        'ema': 1.0,             # PERFECT - ideal conditions for EMA
        'smart_money_ema': 1.0, # PERFECT - EMA with smart money
        'ema_double': 1.0,      # PERFECT - ideal conditions for confirmed crossovers

        # All strategies perform well in medium volatility (Goldilocks zone)
        'macd': 1.0,            # PERFECT - ideal MACD conditions
        'smart_money_macd': 1.0, # PERFECT - MACD with smart money
        'ichimoku': 1.0,        # PERFECT - ideal cloud conditions
        'kama': 1.0,            # PERFECT - balanced adaptive conditions
        'zero_lag': 0.95,       # Excellent - good response time
        'bb_supertrend': 0.9,   # Excellent - balanced trend/volatility
        'momentum': 0.9,        # Excellent - sufficient momentum
        'smc': 0.9,             # Excellent - smart money analysis
        'bollinger': 0.85,      # Very good - moderate band expansion
        'mean_reversion': 0.8,  # Good - still some mean reversion
        'ranging_market': 0.7,  # Good - some ranging behavior
        'scalping': 0.75        # Good - balanced conditions
    },
    'unknown': {
        # PRIORITY: EMA - conservative defaults when regime unclear
        'ema': 0.8,             # GOOD - safe default for foundational strategy
        'smart_money_ema': 0.8, # Good - EMA with smart money
        'ema_double': 0.75,     # GOOD - multiple confirmations help in uncertainty

        # Conservative defaults for all strategies when regime is uncertain
        'macd': 0.7,            # Good - proven versatile strategy
        'smart_money_macd': 0.7, # Good - MACD with smart money
        'ichimoku': 0.7,        # Good - comprehensive analysis
        'momentum': 0.6,        # Moderate - momentum can be risky
        'kama': 0.7,            # Good - adaptive nature helps
        'zero_lag': 0.6,        # Moderate - fast response can be risky
        'mean_reversion': 0.6,  # Moderate - safer in uncertainty
        'bollinger': 0.6,       # Moderate - bands provide guidance
        'smc': 0.7,             # Good - smart money provides insight
        'ranging_market': 0.5,  # Moderate - specific strategy
        'bb_supertrend': 0.6,   # Moderate - trend component helps
        'scalping': 0.4         # Poor - uncertainty bad for scalping
    }
}

# Enable/disable the probabilistic confidence modifier system
ENABLE_PROBABILISTIC_CONFIDENCE_MODIFIERS = True

# Minimum confidence modifier threshold (signals below this are still blocked)
# Raised from 0.2 to 0.5 to filter out poor regime-strategy combinations
MIN_CONFIDENCE_MODIFIER = 0.5

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
ENABLE_MARKET_INTELLIGENCE_FILTERING = True   # RE-ENABLED with lower threshold
MARKET_INTELLIGENCE_MIN_CONFIDENCE = 0.45    # Lowered from 0.55 to 0.45 (regime was 52.1%)
MARKET_INTELLIGENCE_BLOCK_UNSUITABLE_REGIMES = True   # RE-ENABLED for quality filtering

# v2.3.2: Market Bias Filter - Block counter-trend trades when directional consensus is high
# Trade 1586 analysis: BUY in bearish market (1.0 consensus) hit SL immediately
# This filter BLOCKS trades that go against strong market momentum
#
# v2.3.3 (2025-12-28): DISABLED - Market Intelligence Analysis shows:
#   - Counter-trend trades: 51.5% win rate, -$1,478 P&L
#   - Aligned trades: 47.6% win rate, -$4,297 P&L
#   - Counter-trend OUTPERFORMS aligned by 3.9% win rate and $2,819 P&L
#   - The market_bias detection is inverted or lagging
#   - Keeping MI data collection active for further analysis
#   - Will re-enable after fixing regime detection algorithms
MARKET_BIAS_FILTER_ENABLED = False            # DISABLED - counter-trend outperforms aligned
MARKET_BIAS_MIN_CONSENSUS = 0.70              # Block when directional_consensus >= 70%
                                               # Lower = more aggressive filtering
                                               # Higher = only block very strong trends

# =============================================================================
# ENHANCED REGIME DETECTION v2.0 (2025-12-28)
# =============================================================================
# Problem Analysis:
#   - High volatility dominates 83.5% of the time (ATR-based)
#   - Trending detection has 27.3% win rate (lagging indicator)
#   - Counter-trend trades outperform aligned (51.5% vs 47.6%)
#
# Solution: Separate volatility from directionality, add ADX-based trending
#
# DISABLED BY DEFAULT - Only collecting data for analysis

# Enable enhanced regime detection (replaces legacy logic)
ENHANCED_REGIME_DETECTION_ENABLED = False   # DISABLED - data collection only

# ADX-based trending detection thresholds
ADX_TRENDING_THRESHOLD = 25          # ADX > 25 = trending market
ADX_STRONG_TREND_THRESHOLD = 40      # ADX > 40 = strong trend
ADX_WEAK_TREND_THRESHOLD = 20        # ADX < 20 = ranging/consolidating

# EMA alignment weight in trending score
EMA_ALIGNMENT_WEIGHT = 0.4           # 40% weight for EMA alignment
ADX_WEIGHT = 0.4                     # 40% weight for ADX value
MOMENTUM_WEIGHT = 0.2                # 20% weight for momentum

# Volatility-Directionality separation
SEPARATE_VOLATILITY_FROM_STRUCTURE = True   # Treat volatility as orthogonal
VOLATILITY_AS_MODIFIER = True               # Use volatility to modify, not classify

# Structure regime thresholds (when enhanced detection is enabled)
TRENDING_SCORE_THRESHOLD = 0.55      # Score > 0.55 = trending
RANGING_SCORE_THRESHOLD = 0.55       # Inverse trending > 0.55 = ranging
BREAKOUT_SCORE_THRESHOLD = 0.60      # Score > 0.60 = breakout potential

# Data collection for analysis (always enabled)
COLLECT_ENHANCED_REGIME_DATA = True  # Collect new regime scores for comparison
LOG_REGIME_COMPARISON = True         # Log old vs new regime detection

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
FORCE_MARKET_OPEN = False  # Respect actual market hours
ENABLE_WEEKEND_SCANNING = False  # Prevent API calls during weekend (market closed)
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