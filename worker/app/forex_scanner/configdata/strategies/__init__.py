# configdata/strategies/__init__.py
"""
Strategy-specific configuration modules
"""

# Import all strategy configuration modules
from .config_zerolag_strategy import *
from .config_macd_strategy import *
from .config_ema_strategy import *
from .config_smc_strategy import *
from .config_ichimoku_strategy import *
from .config_mean_reversion_strategy import *
from .config_ranging_market_strategy import *
from .config_momentum_strategy import *
from .config_kama_strategy import *
from .config_scalping_strategy import *
from .config_bb_supertrend_strategy import *
from .config_volume_profile_strategy import *

# Import TradingView integration (optional)
try:
    from .tradingview_integration import *
    TRADINGVIEW_INTEGRATION_AVAILABLE = True
except ImportError as e:
    TRADINGVIEW_INTEGRATION_AVAILABLE = False
    # TradingView integration is optional - suppress warning to reduce noise
    # Uncomment the line below if you want to see the import warning:
    # print(f"⚠️ TradingView integration not available: {e}")

# Define what gets exported when using "from strategies import *"
__all__ = [
    # ZeroLag Strategy Settings (Enhanced Modular Configuration)
    'ZERO_LAG_STRATEGY',
    'ZERO_LAG_STRATEGY_CONFIG',
    'ACTIVE_ZERO_LAG_CONFIG',
    'ZERO_LAG_SQUEEZE_MOMENTUM_ENABLED',
    'ZERO_LAG_MTF_VALIDATION_ENABLED',
    'ZERO_LAG_SMART_MONEY_ENABLED',
    
    # Backward Compatibility
    'ZERO_LAG_LENGTH', 
    'ZERO_LAG_BAND_MULT',
    'ZERO_LAG_MIN_CONFIDENCE',
    'ZERO_LAG_BASE_CONFIDENCE',
    'ZERO_LAG_MAX_CONFIDENCE',
    'ZERO_LAG_TREND_WEIGHT',
    'ZERO_LAG_VOLATILITY_WEIGHT',
    'ZERO_LAG_MOMENTUM_WEIGHT',
    'ZERO_LAG_STRATEGY_WEIGHT',
    'ZERO_LAG_ALLOW_COMBINED',
    'ZERO_LAG_PRIORITY_LEVEL',
    'ZERO_LAG_ENABLE_BACKTESTING',
    'ZERO_LAG_MIN_DATA_PERIODS',
    'ZERO_LAG_ENABLE_PERFORMANCE_TRACKING',
    'ZERO_LAG_DEBUG_LOGGING',
    
    # Helper Functions
    'get_zerolag_config_for_epic',
    'get_zerolag_threshold_for_epic',
    'get_zerolag_squeeze_config_for_epic',
    'validate_zerolag_config',
    
    # MACD Strategy Core Settings
    'MACD_EMA_STRATEGY',
    'STRATEGY_WEIGHT_MACD', 
    'MACD_PERIODS',
    'MACD_THRESHOLD_BUFFER_MULTIPLIER',
    'CLAUDE_MACD_CRITICAL_THRESHOLD',
    'CLAUDE_MACD_BEAR_CRITICAL_THRESHOLD',
    
    # EMA-MACD Integration
    'EMA_REQUIRE_MACD_CONFIRMATION',
    'EMA_ALLOW_WITHOUT_MACD',
    
    # Smart Money MACD Settings
    'SMART_MACD_ORDER_FLOW_VALIDATION',
    'SMART_MACD_REQUIRE_OB_CONFLUENCE',
    'SMART_MACD_FVG_PROXIMITY_PIPS',
    'SMART_MACD_ORDER_FLOW_BOOST',
    'SMART_MACD_ORDER_FLOW_PENALTY',
    'USE_SMART_MONEY_MACD',

    # Zero Line Filter Settings
    'MACD_ZERO_LINE_FILTER_ENABLED',

    # MTF Histogram Alignment Filter Settings
    'MACD_MTF_HISTOGRAM_FILTER_ENABLED',
    'MACD_MTF_HISTOGRAM_TIMEFRAMES',
    'MACD_MTF_REQUIRE_ALL_ALIGNED',
    'MACD_MTF_MINIMUM_HISTOGRAM_MAGNITUDE',
    'MACD_MTF_PARTIAL_MISALIGNMENT_PENALTY',
    'MACD_MTF_FULL_REJECTION',
    'MACD_MTF_LOG_ALIGNMENT_CHECKS',

    # RSI Filter Settings
    'MACD_RSI_FILTER_ENABLED',
    'MACD_RSI_OVERBOUGHT_THRESHOLD',
    'MACD_RSI_OVERSOLD_THRESHOLD',

    # Strategy Integration
    'MACD_STRATEGY_WEIGHT',
    'MACD_ALLOW_COMBINED',
    'MACD_PRIORITY_LEVEL',
    
    # Performance and Debug Settings
    'MACD_ENABLE_BACKTESTING',
    'MACD_MIN_DATA_PERIODS',
    'MACD_ENABLE_PERFORMANCE_TRACKING',
    'MACD_DEBUG_LOGGING',
    
    # Helper Functions
    'get_macd_threshold_for_epic',
    'get_macd_config_summary',
    
    # EMA Strategy Core Settings
    'SIMPLE_EMA_STRATEGY',
    'STRICT_EMA_ALIGNMENT',
    'REQUIRE_EMA_SEPARATION',
    'STRATEGY_WEIGHT_EMA',
    'STRATEGY_WEIGHT_ZERO_LAG',
    
    # Dynamic EMA Configuration
    'ENABLE_DYNAMIC_EMA_CONFIG',
    'EMA_STRATEGY_CONFIG',
    'ACTIVE_EMA_CONFIG',
    'SCANNER_USE_DYNAMIC_EMA',
    'BACKTEST_USE_DYNAMIC_EMA',
    
    # EMA200 Distance Validation
    'EMA200_MIN_DISTANCE_ENABLED',
    'EMA200_MIN_DISTANCE_PIPS',
    'EMA200_DISTANCE_MOMENTUM_MULTIPLIER',
    
    # Two-Pole Oscillator (EMA Integration)
    'TWO_POLE_OSCILLATOR_ENABLED',
    'TWO_POLE_FILTER_LENGTH',
    'TWO_POLE_SMA_LENGTH',
    'TWO_POLE_SIGNAL_DELAY',
    'TWO_POLE_VALIDATION_ENABLED',
    'TWO_POLE_MIN_STRENGTH',
    'TWO_POLE_ZONE_FILTER_ENABLED',
    'TWO_POLE_CONFIDENCE_WEIGHT',
    'TWO_POLE_ZONE_BONUS',
    'TWO_POLE_STRENGTH_MULTIPLIER',
    'TWO_POLE_MTF_VALIDATION',
    'TWO_POLE_MTF_TIMEFRAME',
    
    
    # EMA Safety Filter Settings
    'ENABLE_EMA200_CONTRADICTION_FILTER',
    'ENABLE_EMA_STACK_CONTRADICTION_FILTER',
    'EMA200_MINIMUM_MARGIN',
    
    # Additional EMA Strategy Settings
    'EMA_PERIODS',
    'REQUIRE_PRICE_EMA200_CONFIRMATION',
    'MIN_EMA_SEPARATION_PIPS',
    
    # EMA Strategy Integration Settings
    'EMA_STRATEGY_WEIGHT',
    'EMA_ALLOW_COMBINED',
    'EMA_PRIORITY_LEVEL',
    'EMA_ENABLE_BACKTESTING',
    'EMA_MIN_DATA_PERIODS',
    'EMA_ENABLE_PERFORMANCE_TRACKING',
    'EMA_DEBUG_LOGGING',
    
    # EMA Helper Functions
    'get_ema_config_for_epic',
    'get_ema_distance_for_epic',
    'get_ema_separation_for_epic',
    'get_ema_config_summary',
    
    # SMC Strategy Core Settings
    'SMC_STRATEGY',
    'SMC_STRATEGY_CONFIG',
    'ACTIVE_SMC_CONFIG',
    
    # SMC Feature Toggles
    'SMC_MARKET_STRUCTURE_ENABLED',
    'SMC_ORDER_BLOCKS_ENABLED', 
    'SMC_FAIR_VALUE_GAPS_ENABLED',
    'SMC_SUPPLY_DEMAND_ZONES_ENABLED',
    'SMC_MULTI_TIMEFRAME_ENABLED',
    'SMC_LIQUIDITY_ANALYSIS_ENABLED',
    
    # SMC Risk Management
    'SMC_RISK_PROFILES',
    'ACTIVE_SMC_RISK_PROFILE',
    
    # SMC Helper Functions
    'get_smc_config_for_epic',
    'get_smc_threshold_for_epic',
    'get_smc_confluence_factors',
    'validate_smc_config',

    # Ichimoku Strategy Core Settings
    'ICHIMOKU_CLOUD_STRATEGY',
    'STRATEGY_WEIGHT_ICHIMOKU',
    'ICHIMOKU_PERIODS',

    # Ichimoku Configuration System
    'ENABLE_DYNAMIC_ICHIMOKU_CONFIG',
    'ICHIMOKU_STRATEGY_CONFIG',
    'ACTIVE_ICHIMOKU_CONFIG',

    # Ichimoku Validation Filters
    'ICHIMOKU_CLOUD_FILTER_ENABLED',
    'ICHIMOKU_CLOUD_BUFFER_PIPS',
    'ICHIMOKU_CLOUD_THICKNESS_FILTER_ENABLED',
    'ICHIMOKU_MIN_CLOUD_THICKNESS_RATIO',
    'ICHIMOKU_TK_FILTER_ENABLED',
    'ICHIMOKU_MIN_TK_SEPARATION',
    'ICHIMOKU_TK_CROSS_CONFIRMATION_BARS',
    'ICHIMOKU_CHIKOU_FILTER_ENABLED',
    'ICHIMOKU_CHIKOU_BUFFER_PIPS',
    'ICHIMOKU_CHIKOU_PERIODS',

    # Ichimoku Signal Thresholds
    'ICHIMOKU_CLOUD_THICKNESS_THRESHOLD',
    'ICHIMOKU_TK_CROSS_STRENGTH_THRESHOLD',
    'ICHIMOKU_CHIKOU_CLEAR_THRESHOLD',

    # Ichimoku MTF Settings
    'ICHIMOKU_MTF_ENABLED',
    'ICHIMOKU_MTF_TIMEFRAMES',
    'ICHIMOKU_MTF_MIN_ALIGNMENT',
    'ICHIMOKU_MTF_CLOUD_WEIGHT',
    'ICHIMOKU_MTF_TK_WEIGHT',
    'ICHIMOKU_MTF_CHIKOU_WEIGHT',
    'ICHIMOKU_MTF_DEBUG',
    'ICHIMOKU_LOG_MTF_DECISIONS',
    'ICHIMOKU_MTF_LOGGING',

    # Ichimoku Confluence Settings
    'ICHIMOKU_MOMENTUM_CONFLUENCE_ENABLED',
    'ICHIMOKU_MIN_CONFLUENCE_RATIO',
    'ICHIMOKU_EMA_200_TREND_FILTER',
    'ICHIMOKU_RSI_CONFLUENCE_ENABLED',
    'ICHIMOKU_MACD_CONFLUENCE_ENABLED',

    # Smart Money Ichimoku Settings
    'SMART_ICHIMOKU_ORDER_FLOW_VALIDATION',
    'SMART_ICHIMOKU_REQUIRE_OB_CONFLUENCE',
    'SMART_ICHIMOKU_FVG_PROXIMITY_PIPS',
    'SMART_ICHIMOKU_ORDER_FLOW_BOOST',
    'SMART_ICHIMOKU_ORDER_FLOW_PENALTY',
    'USE_SMART_MONEY_ICHIMOKU',

    # Ichimoku Signal Quality
    'ENABLE_ICHIMOKU_CONTRADICTION_FILTER',
    'ICHIMOKU_SIGNAL_SPACING_MINUTES',
    'ICHIMOKU_MIN_SIGNAL_CONFIDENCE',
    'ICHIMOKU_PERFECT_ALIGNMENT_BONUS',
    'ICHIMOKU_DETECTION_MODE',
    'ICHIMOKU_REQUIRE_PERFECT_ALIGNMENT',
    'MIN_BARS_FOR_ICHIMOKU',

    # Ichimoku Risk Management
    'ICHIMOKU_DEFAULT_STOP_LOSS_PIPS',
    'ICHIMOKU_DEFAULT_TAKE_PROFIT_PIPS',
    'ICHIMOKU_DEFAULT_RISK_REWARD_RATIO',
    'ICHIMOKU_USE_CLOUD_STOPS',
    'ICHIMOKU_CLOUD_STOP_BUFFER_PIPS',

    # Ichimoku Optimization
    'ICHIMOKU_USE_OPTIMIZATION_SYSTEM',
    'ICHIMOKU_OPTIMIZATION_FALLBACK',
    'ICHIMOKU_TRACK_SIGNAL_QUALITY',
    'ICHIMOKU_TRACK_CLOUD_PERFORMANCE',
    'ICHIMOKU_TRACK_TK_PERFORMANCE',
    'ICHIMOKU_TRACK_CHIKOU_PERFORMANCE',

    # Ichimoku Advanced Features
    'ICHIMOKU_ADVANCED_CLOUD_ANALYSIS',
    'ICHIMOKU_CLOUD_TWIST_DETECTION',
    'ICHIMOKU_CLOUD_COLOR_SIGNIFICANCE',
    'ICHIMOKU_MARKET_STRUCTURE_INTEGRATION',
    'ICHIMOKU_SUPPORT_RESISTANCE_LEVELS',
    'ICHIMOKU_SESSION_FILTERS',

    # Ichimoku Debug Settings
    'ICHIMOKU_DEBUG_MODE',
    'ICHIMOKU_LOG_COMPONENT_VALUES',
    'ICHIMOKU_LOG_SIGNAL_BREAKDOWN',
    'ICHIMOKU_LOG_CLOUD_ANALYSIS',
    'ICHIMOKU_LOG_TK_ANALYSIS',
    'ICHIMOKU_LOG_CHIKOU_ANALYSIS',
    'ICHIMOKU_LOG_PERFORMANCE_METRICS',
    'ICHIMOKU_LOG_OPTIMIZATION_USAGE',

    # Ichimoku Compatibility Settings
    'ICHIMOKU_ALLOW_STRATEGY_COMBINATION',
    'ICHIMOKU_EMA_COMBINATION_WEIGHT',
    'ICHIMOKU_MACD_COMBINATION_WEIGHT',
    'ICHIMOKU_STANDALONE_WEIGHT',
    'ICHIMOKU_BACKTEST_MODE_ADJUSTMENTS',
    'ICHIMOKU_SCANNER_INTEGRATION',

    # Mean Reversion Strategy Core Settings
    'MEAN_REVERSION_STRATEGY',
    'STRATEGY_WEIGHT_MEAN_REVERSION',
    'MEAN_REVERSION_STRATEGY_WEIGHT',
    'MEAN_REVERSION_ALLOW_COMBINED',
    'MEAN_REVERSION_PRIORITY_LEVEL',

    # LuxAlgo Oscillator Settings
    'LUXALGO_OSCILLATOR_ENABLED',
    'LUXALGO_LENGTH',
    'LUXALGO_SOURCE',
    'LUXALGO_SMOOTHING',
    'LUXALGO_OVERBOUGHT_THRESHOLD',
    'LUXALGO_OVERSOLD_THRESHOLD',
    'LUXALGO_EXTREME_OB_THRESHOLD',
    'LUXALGO_EXTREME_OS_THRESHOLD',
    'LUXALGO_REQUIRE_DIVERGENCE',
    'LUXALGO_MIN_DIVERGENCE_BARS',
    'LUXALGO_DIVERGENCE_LOOKBACK',

    # Multi-Timeframe RSI Settings
    'MTF_RSI_ENABLED',
    'MTF_RSI_PERIOD',
    'MTF_RSI_TIMEFRAMES',
    'MTF_RSI_MIN_ALIGNMENT',
    'MTF_RSI_OVERBOUGHT',
    'MTF_RSI_OVERSOLD',
    'MTF_RSI_EXTREME_OB',
    'MTF_RSI_EXTREME_OS',
    'MTF_RSI_REQUIRE_HIGHER_TF_CONFIRMATION',
    'MTF_RSI_CONFIDENCE_BOOST_ALIGNMENT',
    'MTF_RSI_WEIGHT_BY_TIMEFRAME',

    # RSI-EMA Divergence Settings
    'RSI_EMA_DIVERGENCE_ENABLED',
    'RSI_EMA_PERIOD',
    'RSI_EMA_RSI_PERIOD',
    'RSI_EMA_DIVERGENCE_SENSITIVITY',
    'RSI_EMA_MIN_DIVERGENCE_STRENGTH',
    'RSI_EMA_REQUIRE_HIGHER_HIGHS',
    'RSI_EMA_REQUIRE_LOWER_LOWS',
    'RSI_EMA_MIN_PATTERN_BARS',
    'RSI_EMA_MAX_PATTERN_BARS',
    'RSI_EMA_DIVERGENCE_CONFIRMATION_BARS',

    # Squeeze Momentum Settings
    'SQUEEZE_MOMENTUM_ENABLED',
    'SQUEEZE_BB_LENGTH',
    'SQUEEZE_BB_MULT',
    'SQUEEZE_KC_LENGTH',
    'SQUEEZE_KC_MULT',
    'SQUEEZE_MOMENTUM_LENGTH',
    'SQUEEZE_REQUIRE_SQUEEZE_RELEASE',
    'SQUEEZE_MIN_SQUEEZE_BARS',
    'SQUEEZE_MOMENTUM_THRESHOLD',
    'SQUEEZE_CONFIRMATION_BARS',

    # Oscillator Confluence Settings
    'OSCILLATOR_CONFLUENCE_ENABLED',
    'OSCILLATOR_MIN_CONFIRMATIONS',
    'OSCILLATOR_WEIGHTS',
    'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD',
    'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD',
    'OSCILLATOR_EXTREME_CONFLUENCE_BOOST',

    # Mean Reversion Zone Settings
    'MEAN_REVERSION_ZONE_ENABLED',
    'MEAN_REVERSION_LOOKBACK_PERIODS',
    'MEAN_REVERSION_ZONE_MULTIPLIER',
    'MEAN_REVERSION_REQUIRE_ZONE_TOUCH',
    'MEAN_REVERSION_MIN_ZONE_DISTANCE',
    'MEAN_REVERSION_MAX_ZONE_AGE',
    'MEAN_REVERSION_ZONE_CONFIDENCE_BOOST',

    # Market Regime Detection Settings
    'MARKET_REGIME_DETECTION_ENABLED',
    'MARKET_REGIME_VOLATILITY_PERIOD',
    'MARKET_REGIME_TREND_PERIOD',
    'MARKET_REGIME_RANGING_THRESHOLD',
    'MARKET_REGIME_DISABLE_IN_STRONG_TREND',
    'MARKET_REGIME_BOOST_IN_RANGING',
    'MARKET_REGIME_TREND_STRENGTH_THRESHOLD',

    # Multi-Timeframe Analysis Settings
    'MTF_ANALYSIS_ENABLED',
    'MTF_TIMEFRAMES',
    'MTF_MIN_ALIGNMENT_SCORE',
    'MTF_HIGHER_TF_WEIGHT',
    'MTF_REQUIRE_HIGHER_TF_CONFLUENCE',
    'MTF_ALLOW_LOWER_TF_ENTRY',
    'MTF_CONFIDENCE_BOOST_FULL_ALIGNMENT',

    # Signal Quality and Filtering
    'SIGNAL_QUALITY_MIN_CONFIDENCE',
    'SIGNAL_QUALITY_REQUIRE_VOLUME_CONFIRMATION',
    'SIGNAL_QUALITY_MIN_RISK_REWARD',
    'SIGNAL_FILTER_MAX_SIGNALS_PER_DAY',
    'SIGNAL_FILTER_MIN_SIGNAL_SPACING',
    'SIGNAL_FILTER_AVOID_NEWS_EVENTS',

    # Risk Management
    'MEAN_REVERSION_POSITION_SIZE_MULTIPLIER',
    'MEAN_REVERSION_MAX_DRAWDOWN_THRESHOLD',
    'MEAN_REVERSION_DEFAULT_SL_PIPS',
    'MEAN_REVERSION_DEFAULT_TP_PIPS',
    'MEAN_REVERSION_DYNAMIC_SL_TP',
    'MEAN_REVERSION_TRAIL_STOP_ENABLED',
    'MEAN_REVERSION_SAFETY_PRESETS',

    # Performance and Debug Settings
    'MEAN_REVERSION_ENABLE_BACKTESTING',
    'MEAN_REVERSION_MIN_DATA_PERIODS',
    'MEAN_REVERSION_ENABLE_PERFORMANCE_TRACKING',
    'MEAN_REVERSION_DEBUG_LOGGING',
    'MEAN_REVERSION_LOG_OSCILLATOR_VALUES',
    'MEAN_REVERSION_LOG_CONFLUENCE_SCORES',

    # Ranging Market Strategy Core Settings
    'RANGING_MARKET_STRATEGY',
    'STRATEGY_WEIGHT_RANGING_MARKET',
    'RANGING_MARKET_STRATEGY_WEIGHT',
    'RANGING_MARKET_ALLOW_COMBINED',
    'RANGING_MARKET_PRIORITY_LEVEL',

    # Squeeze Momentum Settings
    'SQUEEZE_MOMENTUM_ENABLED',
    'SQUEEZE_BB_LENGTH',
    'SQUEEZE_BB_MULT',
    'SQUEEZE_KC_LENGTH',
    'SQUEEZE_KC_MULT',
    'SQUEEZE_MOMENTUM_LENGTH',
    'SQUEEZE_USE_TRUE_RANGE',
    'SQUEEZE_MIN_BARS_IN_SQUEEZE',
    'SQUEEZE_MAX_BARS_IN_SQUEEZE',
    'SQUEEZE_RELEASE_CONFIRMATION_BARS',
    'SQUEEZE_MOMENTUM_THRESHOLD',
    'SQUEEZE_REQUIRE_MOMENTUM_ALIGNMENT',
    'SQUEEZE_MIN_MOMENTUM_CHANGE',
    'SQUEEZE_FILTER_FALSE_BREAKOUTS',

    # Wave Trend Oscillator Settings
    'WAVE_TREND_ENABLED',
    'WTO_CHANNEL_LENGTH',
    'WTO_AVERAGE_LENGTH',
    'WTO_SIGNAL_LENGTH',
    'WTO_OVERBOUGHT_LEVEL',
    'WTO_OVERSOLD_LEVEL',
    'WTO_REQUIRE_SIGNAL_CROSSOVER',
    'WTO_MIN_OSCILLATOR_SEPARATION',
    'WTO_DIVERGENCE_DETECTION',
    'WTO_DIVERGENCE_LOOKBACK',
    'WTO_TREND_THRESHOLD',
    'WTO_RANGING_ZONE_UPPER',
    'WTO_RANGING_ZONE_LOWER',

    # Bollinger Bands & Keltner Channel Settings
    'BOLLINGER_BANDS_ENABLED',
    'BB_LENGTH',
    'BB_MULTIPLIER',
    'BB_USE_CLOSE_PRICE',
    'BB_PERCENT_B_ENABLED',
    'KELTNER_CHANNELS_ENABLED',
    'KC_LENGTH',
    'KC_MULTIPLIER',
    'KC_USE_EXPONENTIAL_MA',
    'BB_KC_SQUEEZE_RATIO_THRESHOLD',
    'BB_KC_EXPANSION_RATIO_THRESHOLD',
    'BB_TOUCH_CONFIRMATION_BARS',
    'BB_REJECTION_CONFIRMATION_BARS',

    # RSI Settings for Ranging Markets
    'RSI_ENABLED',
    'RSI_PERIOD',
    'RSI_OVERBOUGHT',
    'RSI_OVERSOLD',
    'RSI_EXTREME_OB',
    'RSI_EXTREME_OS',
    'RSI_DIVERGENCE_ENABLED',
    'RSI_DIVERGENCE_LOOKBACK',
    'RSI_MIN_DIVERGENCE_BARS',
    'RSI_DIVERGENCE_STRENGTH_THRESHOLD',
    'RSI_REQUIRE_DOUBLE_DIVERGENCE',
    'RSI_MEAN_REVERSION_ENABLED',
    'RSI_MEAN_LEVEL',
    'RSI_MEAN_REVERSION_DISTANCE',

    # RVI Settings
    'RVI_ENABLED',
    'RVI_PERIOD',
    'RVI_SIGNAL_PERIOD',
    'RVI_OVERBOUGHT',
    'RVI_OVERSOLD',
    'RVI_REQUIRE_SIGNAL_CROSSOVER',
    'RVI_MOMENTUM_CONFIRMATION',
    'RVI_DIVERGENCE_DETECTION',
    'RVI_MIN_SWING_SIZE',

    # Ranging Market Oscillator Confluence
    'OSCILLATOR_CONFLUENCE_ENABLED',
    'OSCILLATOR_MIN_CONFIRMATIONS',
    'OSCILLATOR_WEIGHTS',
    'OSCILLATOR_BULL_CONFLUENCE_THRESHOLD',
    'OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD',
    'OSCILLATOR_EXTREME_CONFLUENCE_BOOST',
    'CONFLUENCE_REQUIRE_PRIMARY_AGREEMENT',
    'CONFLUENCE_ALLOW_MIXED_SIGNALS',
    'CONFLUENCE_MIN_SIGNAL_STRENGTH',

    # Market Regime Detection
    'MARKET_REGIME_DETECTION_ENABLED',
    'REGIME_VOLATILITY_PERIOD',
    'REGIME_TREND_PERIOD',
    'REGIME_ADX_THRESHOLD',
    'RANGING_MARKET_ADX_MAX',
    'RANGING_MARKET_ATR_STABILITY',
    'RANGING_MARKET_MIN_DURATION',
    'RANGING_MARKET_PRICE_OSCILLATION',
    'REGIME_DISABLE_IN_STRONG_TREND',
    'REGIME_BOOST_IN_RANGING',
    'REGIME_TREND_STRENGTH_THRESHOLD',
    'REGIME_VOLATILITY_FILTER',

    # Ranging Market Risk Management
    'RANGING_POSITION_SIZE_MULTIPLIER',
    'RANGING_MAX_DRAWDOWN_THRESHOLD',
    'RANGING_MAX_DAILY_LOSS',
    'RANGING_DEFAULT_SL_PIPS',
    'RANGING_DEFAULT_TP_PIPS',
    'RANGING_DYNAMIC_SL_TP',
    'RANGING_TRAIL_STOP_ENABLED',
    'RANGING_SL_BASED_ON_RANGE',
    'RANGING_TP_BASED_ON_RANGE',
    'RANGING_RANGE_SL_MULTIPLIER',
    'RANGING_RANGE_TP_MULTIPLIER',
    'RANGING_VOLATILITY_SL_ADJUSTMENT',
    'RANGING_ATR_SL_MULTIPLIER',
    'RANGING_ATR_TP_MULTIPLIER',
    'RANGING_MARKET_SAFETY_PRESETS',

    # Performance and Debug Settings
    'RANGING_MARKET_ENABLE_BACKTESTING',
    'RANGING_MARKET_MIN_DATA_PERIODS',
    'RANGING_MARKET_ENABLE_PERFORMANCE_TRACKING',
    'RANGING_MARKET_DEBUG_LOGGING',
    'RANGING_MARKET_LOG_OSCILLATOR_VALUES',
    'RANGING_MARKET_LOG_CONFLUENCE_SCORES',
    'RANGING_MARKET_LOG_ZONE_INTERACTIONS',

    # Ranging Market Helper Functions
    'get_ranging_market_config_summary',

    # Momentum Strategy Core Settings
    'MOMENTUM_STRATEGY',
    'MOMENTUM_STRATEGY_CONFIG',
    'ACTIVE_MOMENTUM_CONFIG',
    'STRATEGY_WEIGHT_MOMENTUM',

    # Momentum Feature Toggles
    'MOMENTUM_FEATURE_ENABLED',
    'MOMENTUM_VELOCITY_ENABLED',
    'MOMENTUM_VOLUME_CONFIRMATION',
    'MOMENTUM_MTF_VALIDATION',
    'ENABLE_DYNAMIC_MOMENTUM_CONFIG',
    'MOMENTUM_ADAPTIVE_SMOOTHING',

    # Momentum Calculation Settings
    'MOMENTUM_CALCULATION_METHOD',
    'MOMENTUM_SIGNAL_THRESHOLD',
    'MOMENTUM_DIVERGENCE_THRESHOLD',
    'MOMENTUM_VELOCITY_THRESHOLD',

    # Volume Momentum Settings
    'MOMENTUM_VOLUME_MULTIPLIER',
    'MOMENTUM_VOLUME_CONFIRMATION_ENABLED',

    # Multi-Timeframe Momentum Settings
    'MOMENTUM_MTF_TIMEFRAMES',
    'MOMENTUM_MTF_MIN_AGREEMENT',

    # Momentum Risk Management
    'MOMENTUM_MAX_RISK_PER_TRADE',
    'MOMENTUM_STOP_LOSS_METHOD',
    'MOMENTUM_TAKE_PROFIT_RATIO',

    # Signal Validation
    'MOMENTUM_MIN_CONFIDENCE',
    'MOMENTUM_SIGNAL_COOLDOWN',

    # Momentum Helper Functions
    'get_momentum_config_for_epic',
    'get_momentum_threshold_for_epic',
    'get_momentum_velocity_threshold_for_epic',
    'get_adaptive_smoothing_factor',
    'validate_momentum_config',
    'calculate_velocity_momentum',
    'calculate_volume_weighted_momentum',
    'calculate_adaptive_momentum',

    # KAMA Strategy Core Settings
    'KAMA_STRATEGY',
    'KAMA_STRATEGY_CONFIG',
    'ACTIVE_KAMA_CONFIG',
    'DEFAULT_KAMA_CONFIG',
    'ENABLE_DYNAMIC_KAMA_CONFIG',
    'KAMA_ADAPTIVE_SMOOTHING',
    'ADDITIONAL_KAMA_PERIODS',
    'KAMA_MIN_EFFICIENCY_RATIO',
    'KAMA_HIGH_EFFICIENCY_THRESHOLD',
    'KAMA_TREND_CHANGE_WEIGHT',
    'KAMA_CROSSOVER_WEIGHT',
    'KAMA_ER_WEIGHT',
    'KAMA_MIN_CONFIDENCE',
    'KAMA_BASE_CONFIDENCE',
    'KAMA_MAX_CONFIDENCE',
    'KAMA_ENHANCED_VALIDATION',
    'KAMA_MIN_BARS',
    'KAMA_TREND_THRESHOLD',
    'KAMA_REQUIRE_TREND_CONFIRMATION',
    'STRATEGY_WEIGHT_KAMA',
    'INCLUDE_KAMA_IN_COMBINED',
    'KAMA_ALLOW_COMBINED',
    'KAMA_PRIORITY_LEVEL',
    'KAMA_ENABLE_BACKTESTING',
    'KAMA_MIN_DATA_PERIODS',
    'KAMA_ENABLE_PERFORMANCE_TRACKING',
    'KAMA_DEBUG_LOGGING',
    'KAMA_MTF_ENABLED',
    'KAMA_MTF_TIMEFRAMES',
    'KAMA_MTF_MIN_AGREEMENT',
    'KAMA_ER_THRESHOLDS',
    'KAMA_PAIR_ADJUSTMENTS',
    'KAMA_SIGNAL_FREQUENCY_PRESETS',

    # KAMA Helper Functions
    'get_kama_config_for_epic',
    'get_kama_threshold_for_epic',
    'get_kama_confidence_adjustment_for_epic',
    'calculate_kama_confidence',
    'get_kama_market_regime',
    'validate_kama_config',
    'get_kama_config_summary',
    'set_kama_frequency_preset',

    # Scalping Strategy Core Settings
    'SCALPING_STRATEGY_ENABLED',
    'SCALPING_MODE',
    'SCALPING_STRATEGY_CONFIG',
    'ACTIVE_SCALPING_CONFIG',
    'SCALPING_TIMEFRAME',
    'ENABLE_DYNAMIC_SCALPING_CONFIG',
    'SCALPING_ADAPTIVE_SIZING',
    'SCALPING_RISK_MANAGEMENT',
    'SCALPING_MARKET_CONDITIONS',
    'SCALPING_SIGNAL_QUALITY',
    'STRATEGY_WEIGHT_SCALPING',
    'SCALPING_ALLOW_COMBINED',
    'SCALPING_PRIORITY_LEVEL',
    'SCALPING_ENABLE_BACKTESTING',
    'SCALPING_MIN_DATA_PERIODS',
    'SCALPING_ENABLE_PERFORMANCE_TRACKING',
    'SCALPING_DEBUG_LOGGING',
    'SCALPING_FREQUENCY_PRESETS',

    # Scalping Helper Functions
    'get_scalping_config',
    'get_scalping_timeframes',
    'is_scalping_session',
    'get_scalping_position_size',
    'is_valid_scalping_pair',
    'get_scalping_risk_limits',
    'validate_scalping_config',
    'get_scalping_config_summary',
    'set_scalping_frequency_preset',

    # BB SuperTrend Strategy Core Settings
    'BOLLINGER_SUPERTREND_STRATEGY',
    'BB_PERIOD',
    'BB_STD_DEV',
    'SUPERTREND_PERIOD',
    'SUPERTREND_MULTIPLIER',
    'BB_SUPERTREND_BASE_CONFIDENCE',
    'BB_SUPERTREND_MAX_CONFIDENCE',
    'DEFAULT_BB_SUPERTREND_CONFIG',
    'BB_SUPERTREND_CONFIGS',
    'ACTIVE_BB_SUPERTREND_CONFIG',
    'ENABLE_DYNAMIC_BB_SUPERTREND_CONFIG',
    'BB_SUPERTREND_ENHANCEMENTS',
    'BB_SUPERTREND_DEBUG',
    'STRATEGY_WEIGHT_BB_SUPERTREND',
    'BB_SUPERTREND_ALLOW_COMBINED',
    'BB_SUPERTREND_PRIORITY_LEVEL',
    'BB_SUPERTREND_ENABLE_BACKTESTING',
    'BB_SUPERTREND_MIN_DATA_PERIODS',
    'BB_SUPERTREND_ENABLE_PERFORMANCE_TRACKING',
    'BB_SUPERTREND_MIN_CONFIDENCE',
    'BB_SUPERTREND_SIGNAL_COOLDOWN',
    'BB_SUPERTREND_FREQUENCY_PRESETS',

    # BB SuperTrend Helper Functions
    'get_bb_supertrend_config_for_epic',
    'get_bb_supertrend_confidence_for_epic',
    'calculate_bb_position_score',
    'is_bb_supertrend_signal_valid',
    'get_bb_supertrend_risk_parameters',
    'validate_bb_supertrend_config',
    'get_bb_supertrend_config_summary',
    'set_bb_supertrend_frequency_preset',

    # Volume Profile Strategy Core Settings
    'VOLUME_PROFILE_STRATEGY',
    'STRATEGY_WEIGHT_VOLUME_PROFILE',
    'VP_LOOKBACK_PERIODS',
    'ACTIVE_VP_LOOKBACK',
    'VP_NUM_BINS',
    'VP_VOLUME_COLUMN',
    'VP_USE_TICK_VOLUME',

    # Volume Profile Configuration Presets
    'VP_STRATEGY_CONFIG',
    'ACTIVE_VP_CONFIG',

    # HVN/LVN Detection Settings
    'VP_HVN_DETECTION',
    'VP_LVN_DETECTION',

    # Value Area Configuration
    'VP_VALUE_AREA_PERCENT',
    'VP_VALUE_AREA_STRICT',
    'VP_VAH_VAL_AS_SUPPORT_RESISTANCE',

    # Signal Types Configuration
    'VP_SIGNAL_TYPES',

    # Confidence Scoring
    'VP_CONFIDENCE_BASE',
    'VP_CONFIDENCE_FACTORS',
    'VP_CONFIDENCE_PENALTIES',

    # Risk Management
    'VP_STOP_LOSS_CONFIG',
    'VP_TAKE_PROFIT_CONFIG',
    'VP_POSITION_SIZING',

    # Signal Filtering
    'VP_SIGNAL_FILTERS',
    'VP_PROXIMITY_FILTERS',
    'VP_VOLUME_FILTERS',
    'VP_TREND_FILTER',

    # Market Regime Configuration
    'VP_ADX_FILTER',
    'VP_ATR_FILTER',
    'VP_SESSION_FILTER',

    # Pair-Specific Configuration
    'VP_PAIR_CONFIG',

    # Performance Settings
    'VP_PERFORMANCE',
    'VP_DEBUG',

    # Strategy Integration
    'VP_STRATEGY_WEIGHT',
    'VP_ALLOW_COMBINED',
    'VP_PRIORITY_LEVEL',
    'VP_ENABLE_BACKTESTING',
    'VP_MIN_DATA_PERIODS',
    'VP_ENABLE_PERFORMANCE_TRACKING',

    # Volume Profile Helper Functions
    'get_vp_config_for_epic',
    'extract_pair_from_epic',
    'get_vp_lookback_periods',
    'get_vp_proximity_threshold',
    'get_vp_stop_loss_config',
    'get_vp_take_profit_config',
    'set_vp_preset',
    'get_available_vp_presets',
    'get_vp_config_summary'
]

# Add TradingView integration functions if available
if TRADINGVIEW_INTEGRATION_AVAILABLE:
    __all__.extend([
        # TradingView Integration Functions
        'search_tradingview_strategies',
        'import_tradingview_strategy',
        'list_tradingview_imports',
        'get_tradingview_import_status',
        'validate_tradingview_integration',
        'TradingViewStrategyIntegrator',
        'get_integrator',
        'TRADINGVIEW_INTEGRATION_AVAILABLE'
    ])

# Module metadata
__version__ = "1.0.0"
__description__ = "Strategy-specific configuration modules for ZeroLag, MACD, EMA, and SMC strategies"

def get_strategies_summary() -> dict:
    """Get a summary of all loaded strategy configurations"""
    return {
        'loaded_strategies': ['zerolag', 'macd', 'ema', 'smc', 'ichimoku', 'mean_reversion', 'ranging_market', 'momentum', 'kama', 'scalping', 'bb_supertrend'],
        'zerolag_enabled': globals().get('ZERO_LAG_STRATEGY', False),
        'macd_enabled': globals().get('MACD_EMA_STRATEGY', False),
        'ema_enabled': globals().get('SIMPLE_EMA_STRATEGY', False),
        'smc_enabled': globals().get('SMC_STRATEGY', False),
        'ichimoku_enabled': globals().get('ICHIMOKU_CLOUD_STRATEGY', False),
        'mean_reversion_enabled': globals().get('MEAN_REVERSION_STRATEGY', False),
        'ranging_market_enabled': globals().get('RANGING_MARKET_STRATEGY', False),
        'momentum_enabled': globals().get('MOMENTUM_STRATEGY', False),
        'kama_enabled': globals().get('KAMA_STRATEGY', False),
        'scalping_enabled': globals().get('SCALPING_STRATEGY_ENABLED', False),
        'bb_supertrend_enabled': globals().get('BOLLINGER_SUPERTREND_STRATEGY', False),
        'total_settings': len(__all__)
    }

def validate_strategy_configs() -> dict:
    """Validate all strategy configuration completeness"""
    validation_results = {
        'zerolag': validate_zerolag_config(),
        'macd': _validate_macd_config(),
        'ema': _validate_ema_config(),
        'smc': _validate_smc_config(),
        'ichimoku': _validate_ichimoku_config(),
        'mean_reversion': _validate_mean_reversion_config(),
        'ranging_market': _validate_ranging_market_config(),
        'momentum': validate_momentum_config(),
        'kama': validate_kama_config(),
        'scalping': validate_scalping_config(),
        'bb_supertrend': validate_bb_supertrend_config()
    }
    
    # Add TradingView integration validation if available
    if TRADINGVIEW_INTEGRATION_AVAILABLE:
        try:
            validation_results['tradingview_integration'] = validate_tradingview_integration()
        except Exception as e:
            validation_results['tradingview_integration'] = {'valid': False, 'error': str(e)}
    
    validation_results['overall_valid'] = all(result.get('valid', False) for result in validation_results.values() if isinstance(result, dict)) and all(result for result in validation_results.values() if isinstance(result, bool))
    return validation_results

def _validate_smc_config() -> dict:
    """Validate SMC strategy configuration"""
    try:
        # Use the validation function from the SMC config module
        if 'validate_smc_config' in globals():
            return globals()['validate_smc_config']()
        else:
            print("❌ SMC validation function not available")
            return {'valid': False, 'error': 'SMC validation function not found'}
            
    except Exception as e:
        print(f"❌ SMC config validation failed: {e}")
        return {'valid': False, 'error': str(e)}

def _validate_zerolag_config() -> bool:
    """Validate ZeroLag strategy configuration"""
    try:
        required_zerolag_configs = [
            'ZERO_LAG_STRATEGY', 'ZERO_LAG_LENGTH', 'ZERO_LAG_BAND_MULT', 
            'ZERO_LAG_MIN_CONFIDENCE'
        ]
        
        for config_name in required_zerolag_configs:
            if config_name not in globals():
                print(f"❌ Missing ZeroLag config: {config_name}")
                return False
        
        # Validate confidence values are reasonable
        min_conf = globals().get('ZERO_LAG_MIN_CONFIDENCE', 0)
        if not (0 <= min_conf <= 1):
            print("❌ ZeroLag confidence values must be between 0 and 1")
            return False
        
        print("✅ ZeroLag configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ ZeroLag config validation failed: {e}")
        return False

def _validate_macd_config() -> bool:
    """Validate MACD strategy configuration"""
    try:
        required_macd_configs = [
            'MACD_EMA_STRATEGY', 'MACD_PERIODS', 'MACD_THRESHOLD_BUFFER_MULTIPLIER',
            'get_macd_threshold_for_epic'
        ]
        
        for config_name in required_macd_configs:
            if config_name not in globals():
                print(f"❌ Missing MACD config: {config_name}")
                return False
        
        # Validate MACD periods
        periods = globals().get('MACD_PERIODS', {})
        if not isinstance(periods, dict) or not all(k in periods for k in ['fast_ema', 'slow_ema', 'signal_ema']):
            print("❌ MACD_PERIODS must contain fast_ema, slow_ema, and signal_ema")
            return False
        
        # Validate threshold values are reasonable
        threshold = globals().get('MACD_THRESHOLD_BUFFER_MULTIPLIER', 0)
        if threshold < 1.0:
            print("❌ MACD_THRESHOLD_BUFFER_MULTIPLIER should be >= 1.0")
            return False
        
        print("✅ MACD configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ MACD config validation failed: {e}")
        return False

def _validate_ema_config() -> bool:
    """Validate EMA strategy configuration"""
    try:
        required_ema_configs = [
            'SIMPLE_EMA_STRATEGY', 'EMA_STRATEGY_CONFIG', 'ACTIVE_EMA_CONFIG',
            'EMA_PERIODS', 'get_ema_config_for_epic'
        ]
        
        for config_name in required_ema_configs:
            if config_name not in globals():
                print(f"❌ Missing EMA config: {config_name}")
                return False
        
        # Validate EMA strategy config dictionary
        strategy_config = globals().get('EMA_STRATEGY_CONFIG', {})
        if not isinstance(strategy_config, dict) or len(strategy_config) == 0:
            print("❌ EMA_STRATEGY_CONFIG must be a non-empty dictionary")
            return False
        
        # Validate active config exists in strategy config
        active_config = globals().get('ACTIVE_EMA_CONFIG', '')
        if active_config not in strategy_config:
            print(f"❌ ACTIVE_EMA_CONFIG '{active_config}' not found in EMA_STRATEGY_CONFIG")
            return False
        
        # Validate EMA periods
        periods = globals().get('EMA_PERIODS', [])
        if not isinstance(periods, list) or len(periods) == 0:
            print("❌ EMA_PERIODS must be a non-empty list")
            return False
        
        # Validate distance settings if enabled
        distance_enabled = globals().get('EMA200_MIN_DISTANCE_ENABLED', False)
        if distance_enabled:
            distance_pips = globals().get('EMA200_MIN_DISTANCE_PIPS', {})
            if not isinstance(distance_pips, dict) or 'DEFAULT' not in distance_pips:
                print("❌ EMA200_MIN_DISTANCE_PIPS must contain 'DEFAULT' key")
                return False
        
        print("✅ EMA configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ EMA config validation failed: {e}")
        return False

def _validate_ichimoku_config() -> bool:
    """Validate Ichimoku strategy configuration"""
    try:
        required_ichimoku_configs = [
            'ICHIMOKU_CLOUD_STRATEGY', 'ICHIMOKU_PERIODS', 'ICHIMOKU_STRATEGY_CONFIG',
            'ACTIVE_ICHIMOKU_CONFIG', 'ICHIMOKU_MTF_ENABLED'
        ]

        for config_name in required_ichimoku_configs:
            if config_name not in globals():
                print(f"❌ Missing Ichimoku config: {config_name}")
                return False

        # Validate Ichimoku periods
        periods = globals().get('ICHIMOKU_PERIODS', {})
        required_periods = ['tenkan_period', 'kijun_period', 'senkou_b_period', 'chikou_shift', 'cloud_shift']
        if not isinstance(periods, dict) or not all(k in periods for k in required_periods):
            print("❌ ICHIMOKU_PERIODS must contain all required periods")
            return False

        # Validate Ichimoku strategy config dictionary
        strategy_config = globals().get('ICHIMOKU_STRATEGY_CONFIG', {})
        if not isinstance(strategy_config, dict) or len(strategy_config) == 0:
            print("❌ ICHIMOKU_STRATEGY_CONFIG must be a non-empty dictionary")
            return False

        # Validate active config exists in strategy config
        active_config = globals().get('ACTIVE_ICHIMOKU_CONFIG', '')
        if active_config not in strategy_config:
            print(f"❌ ACTIVE_ICHIMOKU_CONFIG '{active_config}' not found in ICHIMOKU_STRATEGY_CONFIG")
            return False

        # Validate threshold values are reasonable
        cloud_threshold = globals().get('ICHIMOKU_CLOUD_THICKNESS_THRESHOLD', 0)
        if cloud_threshold < 0:
            print("❌ ICHIMOKU_CLOUD_THICKNESS_THRESHOLD should be >= 0")
            return False

        # Validate confidence settings
        min_confidence = globals().get('ICHIMOKU_MIN_SIGNAL_CONFIDENCE', 0)
        if not (0 <= min_confidence <= 1):
            print("❌ ICHIMOKU_MIN_SIGNAL_CONFIDENCE must be between 0 and 1")
            return False

        # Validate MTF settings if enabled
        mtf_enabled = globals().get('ICHIMOKU_MTF_ENABLED', False)
        if mtf_enabled:
            mtf_timeframes = globals().get('ICHIMOKU_MTF_TIMEFRAMES', [])
            if not isinstance(mtf_timeframes, list) or len(mtf_timeframes) == 0:
                print("❌ ICHIMOKU_MTF_TIMEFRAMES must be a non-empty list when MTF is enabled")
                return False

            mtf_alignment = globals().get('ICHIMOKU_MTF_MIN_ALIGNMENT', 0)
            if not (0 <= mtf_alignment <= 1):
                print("❌ ICHIMOKU_MTF_MIN_ALIGNMENT must be between 0 and 1")
                return False

        print("✅ Ichimoku configuration validation passed")
        return True

    except Exception as e:
        print(f"❌ Ichimoku config validation failed: {e}")
        return False

def _validate_mean_reversion_config() -> bool:
    """Validate Mean Reversion strategy configuration"""
    try:
        required_mean_reversion_configs = [
            'MEAN_REVERSION_STRATEGY', 'STRATEGY_WEIGHT_MEAN_REVERSION',
            'LUXALGO_OSCILLATOR_ENABLED', 'MTF_RSI_ENABLED',
            'OSCILLATOR_CONFLUENCE_ENABLED', 'SIGNAL_QUALITY_MIN_CONFIDENCE'
        ]

        for config_name in required_mean_reversion_configs:
            if config_name not in globals():
                print(f"❌ Missing Mean Reversion config: {config_name}")
                return False

        # Validate weight values are reasonable
        strategy_weight = globals().get('STRATEGY_WEIGHT_MEAN_REVERSION', 0)
        if not (0 <= strategy_weight <= 1):
            print("❌ STRATEGY_WEIGHT_MEAN_REVERSION must be between 0 and 1")
            return False

        # Validate confidence settings
        min_confidence = globals().get('SIGNAL_QUALITY_MIN_CONFIDENCE', 0)
        if not (0 <= min_confidence <= 1):
            print("❌ SIGNAL_QUALITY_MIN_CONFIDENCE must be between 0 and 1")
            return False

        # Validate oscillator thresholds
        ob_threshold = globals().get('LUXALGO_OVERBOUGHT_THRESHOLD', 0)
        os_threshold = globals().get('LUXALGO_OVERSOLD_THRESHOLD', 0)
        if not (0 <= os_threshold < ob_threshold <= 100):
            print("❌ LuxAlgo thresholds must be: 0 <= oversold < overbought <= 100")
            return False

        # Validate MTF settings if enabled
        mtf_enabled = globals().get('MTF_RSI_ENABLED', False)
        if mtf_enabled:
            mtf_timeframes = globals().get('MTF_RSI_TIMEFRAMES', [])
            if not isinstance(mtf_timeframes, list) or len(mtf_timeframes) == 0:
                print("❌ MTF_RSI_TIMEFRAMES must be a non-empty list when MTF is enabled")
                return False

        print("✅ Mean Reversion configuration validation passed")
        return True

    except Exception as e:
        print(f"❌ Mean Reversion config validation failed: {e}")
        return False

def _validate_ranging_market_config() -> bool:
    """Validate Ranging Market strategy configuration"""
    try:
        required_ranging_market_configs = [
            'RANGING_MARKET_STRATEGY', 'STRATEGY_WEIGHT_RANGING_MARKET',
            'SQUEEZE_MOMENTUM_ENABLED', 'WAVE_TREND_ENABLED',
            'OSCILLATOR_CONFLUENCE_ENABLED', 'SIGNAL_QUALITY_MIN_CONFIDENCE'
        ]

        for config_name in required_ranging_market_configs:
            if config_name not in globals():
                print(f"❌ Missing Ranging Market config: {config_name}")
                return False

        # Validate weight values are reasonable
        strategy_weight = globals().get('STRATEGY_WEIGHT_RANGING_MARKET', 0)
        if not (0 <= strategy_weight <= 1):
            print("❌ STRATEGY_WEIGHT_RANGING_MARKET must be between 0 and 1")
            return False

        # Validate confidence settings
        min_confidence = globals().get('SIGNAL_QUALITY_MIN_CONFIDENCE', 0)
        if not (0 <= min_confidence <= 1):
            print("❌ SIGNAL_QUALITY_MIN_CONFIDENCE must be between 0 and 1")
            return False

        # Validate oscillator confluence settings
        min_confirmations = globals().get('OSCILLATOR_MIN_CONFIRMATIONS', 0)
        if min_confirmations < 1:
            print("❌ OSCILLATOR_MIN_CONFIRMATIONS must be >= 1")
            return False

        bull_threshold = globals().get('OSCILLATOR_BULL_CONFLUENCE_THRESHOLD', 0)
        bear_threshold = globals().get('OSCILLATOR_BEAR_CONFLUENCE_THRESHOLD', 0)
        if not (0 <= bull_threshold <= 1) or not (0 <= bear_threshold <= 1):
            print("❌ Oscillator confluence thresholds must be between 0 and 1")
            return False

        print("✅ Ranging Market configuration validation passed")
        return True

    except Exception as e:
        print(f"❌ Ranging Market config validation failed: {e}")
        return False

# Validate configurations on import (can be disabled in production)
try:
    validation_result = validate_strategy_configs()
    if not validation_result['overall_valid']:
        print("⚠️ Some strategy configuration validations failed. Check settings.")
    else:
        print(f"✅ Strategy configurations loaded successfully")
except Exception as e:
    print(f"⚠️ Strategy configuration validation error: {e}")

print(f"📊 Strategy configs loaded - ZeroLag: {globals().get('ZERO_LAG_STRATEGY', False)}, MACD: {globals().get('MACD_EMA_STRATEGY', False)}, EMA: {globals().get('SIMPLE_EMA_STRATEGY', False)}, SMC: {globals().get('SMC_STRATEGY', False)}, Ichimoku: {globals().get('ICHIMOKU_CLOUD_STRATEGY', False)}, Mean Reversion: {globals().get('MEAN_REVERSION_STRATEGY', False)}, Ranging Market: {globals().get('RANGING_MARKET_STRATEGY', False)}, Momentum: {globals().get('MOMENTUM_STRATEGY', False)}, KAMA: {globals().get('KAMA_STRATEGY', False)}, Scalping: {globals().get('SCALPING_STRATEGY_ENABLED', False)}, BB SuperTrend: {globals().get('BOLLINGER_SUPERTREND_STRATEGY', False)}")