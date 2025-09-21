# validation/replay_config.py
"""
Replay Configuration Settings

Configuration settings specific to the signal replay and validation system.
These settings control how historical data is fetched, processed, and validated.
"""

from datetime import timedelta
from typing import Dict, List, Optional

# Data fetching configuration
DEFAULT_LOOKBACK_BARS = 500  # Number of historical bars to fetch before target timestamp
MIN_LOOKBACK_BARS = 100     # Minimum bars required for proper indicator calculation
MAX_LOOKBACK_BARS = 2000    # Maximum bars to prevent memory issues

# Historical data window settings
DEFAULT_LOOKBACK_HOURS = 168  # 7 days in hours for sufficient data
INDICATOR_WARMUP_BARS = 200   # Extra bars needed for indicator calculation warmup

# Timeframe mappings for historical data
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

# Strategy replay configuration
STRATEGY_REPLAY_CONFIG = {
    'ema': {
        'min_bars_required': 200,
        'indicators_needed': ['ema_21', 'ema_50', 'ema_200', 'two_pole_oscillator'],
        'mtf_analysis': True,
        'momentum_bias': True
    },
    'macd': {
        'min_bars_required': 100,
        'indicators_needed': ['macd', 'macd_signal', 'macd_histogram', 'ema_200'],
        'mtf_analysis': True,
        'momentum_bias': False
    },
    'kama': {
        'min_bars_required': 150,
        'indicators_needed': ['kama', 'efficiency_ratio'],
        'mtf_analysis': False,
        'momentum_bias': False
    },
    'zero_lag': {
        'min_bars_required': 100,
        'indicators_needed': ['zero_lag_ema', 'momentum'],
        'mtf_analysis': False,
        'momentum_bias': False
    },
    'momentum_bias': {
        'min_bars_required': 250,
        'indicators_needed': ['momentum_bias_index', 'ema_21', 'ema_50'],
        'mtf_analysis': False,
        'momentum_bias': True
    }
}

# Validation comparison settings
VALIDATION_TOLERANCE = {
    'price_precision': 5,           # Decimal places for price comparison
    'confidence_tolerance': 0.001,  # Tolerance for confidence score comparison
    'timestamp_tolerance_seconds': 30,  # Tolerance for timestamp comparison
    'indicator_tolerance': 0.0001   # Tolerance for indicator value comparison
}

# Output formatting configuration
OUTPUT_CONFIG = {
    'max_line_length': 80,
    'decimal_places': {
        'price': 5,
        'confidence': 3,
        'indicator': 4,
        'percentage': 1
    },
    'colors': {
        'success': '\033[92m',
        'warning': '\033[93m', 
        'error': '\033[91m',
        'info': '\033[94m',
        'reset': '\033[0m'
    },
    'symbols': {
        'check': '✅',
        'cross': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'bullet': '•',
        'arrow': '→'
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    'enable_parallel_processing': True,
    'max_concurrent_epics': 5,
    'enable_data_caching': True,
    'cache_timeout_seconds': 300,
    'enable_progress_bars': True,
    'log_performance_metrics': True
}

# Database query optimization
DB_CONFIG = {
    'batch_size': 1000,
    'connection_pool_size': 3,
    'query_timeout_seconds': 30,
    'enable_query_caching': True,
    'fetch_chunk_size': 500
}

# Error handling configuration  
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay_seconds': 1,
    'continue_on_error': True,
    'detailed_error_logging': True,
    'save_failed_validations': True
}

# Feature flags for experimental features
FEATURE_FLAGS = {
    'enable_smart_money_validation': True,
    'enable_claude_analysis_replay': True,
    'enable_deduplication_replay': True,
    'enable_multi_timeframe_validation': True,
    'enable_configuration_time_travel': True,
    'enable_performance_profiling': False
}

# Default epic list for batch operations
DEFAULT_EPIC_LIST = [
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.GBPUSD.MINI.IP", 
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.EURJPY.MINI.IP",
    "CS.D.AUDJPY.MINI.IP",
    "CS.D.NZDUSD.MINI.IP",
    "CS.D.USDCHF.MINI.IP"
]

class ReplayConfig:
    """Configuration container for replay operations"""
    
    @staticmethod
    def get_strategy_config(strategy: str) -> Dict:
        """Get configuration for specific strategy"""
        if strategy is None:
            return {
                'min_bars_required': 200,
                'indicators_needed': [],
                'mtf_analysis': False,
                'momentum_bias': False
            }
        return STRATEGY_REPLAY_CONFIG.get(strategy.lower(), {
            'min_bars_required': 200,
            'indicators_needed': [],
            'mtf_analysis': False,
            'momentum_bias': False
        })
    
    @staticmethod
    def get_lookback_bars(strategy: str = None, timeframe: str = '15m') -> int:
        """Get appropriate lookback bars for strategy and timeframe"""
        if strategy:
            strategy_config = ReplayConfig.get_strategy_config(strategy)
            min_required = strategy_config.get('min_bars_required', DEFAULT_LOOKBACK_BARS)
        else:
            min_required = DEFAULT_LOOKBACK_BARS
            
        # Add extra bars for higher timeframes
        timeframe_multiplier = {
            '1m': 3.0,
            '5m': 2.0, 
            '15m': 1.0,
            '30m': 0.8,
            '1h': 0.6,
            '4h': 0.4
        }.get(timeframe, 1.0)
        
        return int(min_required * timeframe_multiplier)
    
    @staticmethod 
    def is_feature_enabled(feature: str) -> bool:
        """Check if a feature flag is enabled"""
        return FEATURE_FLAGS.get(feature, False)
    
    @staticmethod
    def get_output_symbol(symbol_name: str) -> str:
        """Get output symbol for display"""
        return OUTPUT_CONFIG['symbols'].get(symbol_name, symbol_name)
    
    @staticmethod
    def get_color_code(color_name: str) -> str:
        """Get ANSI color code"""
        return OUTPUT_CONFIG['colors'].get(color_name, OUTPUT_CONFIG['colors']['reset'])