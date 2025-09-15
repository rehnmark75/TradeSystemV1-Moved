# configdata/strategies/__init__.py
"""
Strategy-specific configuration modules
"""

# Import all strategy configuration modules
from .config_zerolag_strategy import *
from .config_macd_strategy import *
from .config_ema_strategy import *
from .config_smc_strategy import *

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
    
    # MACD Filter Settings
    'ENABLE_MACD_CONTRADICTION_FILTER',
    'MACD_STRONG_THRESHOLD',
    'MACD_ENHANCED_FILTERS_ENABLED',
    'MACD_DETECTION_MODE',
    'MACD_REQUIRE_EMA200_ALIGNMENT',
    'MACD_DISABLE_EMA200_FILTER',
    'MACD_EMA200_FILTER_MODE',
    'MIN_BARS_FOR_MACD',
    
    # MTF (Multi-Timeframe) Settings
    'MACD_MTF_DEBUG',
    'MACD_LOG_MTF_DECISIONS',
    'MACD_MTF_ENABLED',
    'MACD_MTF_TIMEFRAMES',
    'MACD_MTF_MIN_ALIGNMENT',
    'MACD_MTF_LOGGING',
    'MACD_FILTER_CONFIG',
    'MACD_MTF_CONFIG',
    
    # Momentum Confirmation Settings
    'MACD_MOMENTUM_CONFIRMATION_ENABLED',
    'MACD_CONFIRMATION_WINDOW',
    'MACD_MOMENTUM_MULTIPLIER',
    'MACD_ALLOW_DELAYED_SIGNALS',
    'MACD_CONTINUATION_ENABLED',
    'MACD_CONTINUATION_MULTIPLIER',
    'MACD_TRACK_WEAK_CROSSOVERS',
    'MACD_CONFIRMATION_LOOKBACK',
    
    # RSI Filter Settings
    'MACD_RSI_FILTER_ENABLED',
    'MACD_RSI_PERIOD',
    'MACD_RSI_MIDDLE',
    'MACD_RSI_QUALITY_THRESHOLD_BULL',
    'MACD_RSI_QUALITY_THRESHOLD_BEAR',
    'MACD_RSI_OVERBOUGHT_THRESHOLD',
    'MACD_RSI_OVERSOLD_THRESHOLD',
    'MACD_RSI_REQUIRE_RISING',
    'MACD_RSI_REQUIRE_QUALITY_THRESHOLDS',
    'MACD_RSI_SKIP_EXTREME_ZONES',
    
    # Zero Line Filter Settings
    'MACD_ZERO_LINE_FILTER_ENABLED',
    'MACD_ZERO_LINE_REQUIRE_BOTH_LINES',
    'MACD_ZERO_LINE_BULL_BELOW_ZERO',
    'MACD_ZERO_LINE_BEAR_ABOVE_ZERO',
    'MACD_ZERO_LINE_STRICT_MODE',
    
    # Safety and Strategy Integration
    'MACD_SAFETY_PRESETS',
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
    
    # Momentum Bias Index (EMA Integration)
    'MOMENTUM_BIAS_ENABLED',
    'MOMENTUM_BIAS_LENGTH',
    'MOMENTUM_BIAS_BIAS_LENGTH',
    'MOMENTUM_BIAS_SMOOTH_LENGTH',
    'MOMENTUM_BIAS_BOUNDARY_LENGTH',
    'MOMENTUM_BIAS_STD_MULTIPLIER',
    'MOMENTUM_BIAS_SMOOTH',
    'MOMENTUM_BIAS_REQUIRE_ABOVE_BOUNDARY',
    'MOMENTUM_BIAS_VALIDATION_ENABLED',
    'MOMENTUM_BIAS_MIN_STRENGTH',
    'MOMENTUM_BIAS_CONFIDENCE_WEIGHT',
    'MOMENTUM_BIAS_BOUNDARY_BONUS',
    'MOMENTUM_BIAS_STRENGTH_MULTIPLIER',
    
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
    'validate_smc_config'
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
        'loaded_strategies': ['zerolag', 'macd', 'ema', 'smc'],
        'zerolag_enabled': globals().get('ZERO_LAG_STRATEGY', False),
        'macd_enabled': globals().get('MACD_EMA_STRATEGY', False),
        'ema_enabled': globals().get('SIMPLE_EMA_STRATEGY', False),
        'smc_enabled': globals().get('SMC_STRATEGY', False),
        'total_settings': len(__all__)
    }

def validate_strategy_configs() -> dict:
    """Validate all strategy configuration completeness"""
    validation_results = {
        'zerolag': validate_zerolag_config(),
        'macd': _validate_macd_config(),
        'ema': _validate_ema_config(),
        'smc': _validate_smc_config()
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

# Validate configurations on import (can be disabled in production)
try:
    validation_result = validate_strategy_configs()
    if not validation_result['overall_valid']:
        print("⚠️ Some strategy configuration validations failed. Check settings.")
    else:
        print(f"✅ Strategy configurations loaded successfully")
except Exception as e:
    print(f"⚠️ Strategy configuration validation error: {e}")

print(f"📊 Strategy configs loaded - ZeroLag: {globals().get('ZERO_LAG_STRATEGY', False)}, MACD: {globals().get('MACD_EMA_STRATEGY', False)}, EMA: {globals().get('SIMPLE_EMA_STRATEGY', False)}, SMC: {globals().get('SMC_STRATEGY', False)}")