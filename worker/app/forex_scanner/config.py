# config.py
"""
Configuration settings for the Forex Scanner

NOTE: Most strategy settings are in database (strategy_config schema).
Only SMC_SIMPLE strategy is active. See docs/adding_new_strategy.md for new strategies.

Environment variables take precedence where applicable.
Database settings are loaded via scanner_config_service.py at runtime.
"""
import os
from typing import List

# =============================================================================
# ENVIRONMENT VARIABLES (Infrastructure)
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/forex")
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', None)

# MinIO Object Storage (for Claude vision charts)
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'claude-charts')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() == 'true'
MINIO_CHART_RETENTION_DAYS = int(os.getenv('MINIO_CHART_RETENTION_DAYS', '30'))
MINIO_ENABLED = os.getenv('MINIO_ENABLED', 'true').lower() == 'true'
MINIO_PUBLIC_URL = os.getenv('MINIO_PUBLIC_URL', f"http://{MINIO_ENDPOINT}")

# Order Execution API
ORDER_API_URL = os.getenv('ORDER_API_URL', "http://fastapi-dev:8000/orders/place-order")
API_SUBSCRIPTION_KEY = os.getenv('API_SUBSCRIPTION_KEY', "436abe054a074894a0517e5172f0e5b6")

# =============================================================================
# PAIR METADATA (Used by pip calculations and order mapping)
# =============================================================================

PAIR_INFO = {
    # Major USD pairs
    'CS.D.EURUSD.CEEM.IP': {'pair': 'EURUSD', 'pip_multiplier': 10000},
    'CS.D.GBPUSD.MINI.IP': {'pair': 'GBPUSD', 'pip_multiplier': 10000},
    'CS.D.AUDUSD.MINI.IP': {'pair': 'AUDUSD', 'pip_multiplier': 10000},
    'CS.D.NZDUSD.MINI.IP': {'pair': 'NZDUSD', 'pip_multiplier': 10000},
    'CS.D.USDCHF.MINI.IP': {'pair': 'USDCHF', 'pip_multiplier': 10000},
    'CS.D.USDCAD.MINI.IP': {'pair': 'USDCAD', 'pip_multiplier': 10000},
    # JPY pairs (100 pip multiplier)
    'CS.D.USDJPY.MINI.IP': {'pair': 'USDJPY', 'pip_multiplier': 100},
    'CS.D.EURJPY.MINI.IP': {'pair': 'EURJPY', 'pip_multiplier': 100},
    'CS.D.GBPJPY.MINI.IP': {'pair': 'GBPJPY', 'pip_multiplier': 100},
    'CS.D.AUDJPY.MINI.IP': {'pair': 'AUDJPY', 'pip_multiplier': 100},
    'CS.D.CADJPY.MINI.IP': {'pair': 'CADJPY', 'pip_multiplier': 100},
    'CS.D.CHFJPY.MINI.IP': {'pair': 'CHFJPY', 'pip_multiplier': 100},
    'CS.D.NZDJPY.MINI.IP': {'pair': 'NZDJPY', 'pip_multiplier': 100},
    # Cross pairs
    'CS.D.EURGBP.MINI.IP': {'pair': 'EURGBP', 'pip_multiplier': 10000},
    'CS.D.EURAUD.MINI.IP': {'pair': 'EURAUD', 'pip_multiplier': 10000},
    'CS.D.GBPAUD.MINI.IP': {'pair': 'GBPAUD', 'pip_multiplier': 10000}
}

EPIC_LIST: List[str] = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.USDCHF.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.AUDJPY.MINI.IP'
]

# Epic mapping (scanner epic -> trading API epic)
EPIC_MAP = {
    "CS.D.EURUSD.CEEM.IP": "EURUSD.1.MINI",
    "CS.D.GBPUSD.MINI.IP": "GBPUSD.1.MINI",
    "CS.D.USDJPY.MINI.IP": "USDJPY.100.MINI",
    "CS.D.AUDUSD.MINI.IP": "AUDUSD.1.MINI",
    "CS.D.USDCAD.MINI.IP": "USDCAD.1.MINI",
    "CS.D.EURJPY.MINI.IP": "EURJPY.100.MINI",
    "CS.D.AUDJPY.MINI.IP": "AUDJPY.100.MINI",
    "CS.D.NZDUSD.MINI.IP": "NZDUSD.1.MINI",
    "CS.D.USDCHF.MINI.IP": "USDCHF.1.MINI"
}

REVERSE_EPIC_MAP = {v: k for k, v in EPIC_MAP.items()}

# Trading blacklist (scan but don't trade)
TRADING_BLACKLIST = {}

# =============================================================================
# STRATEGY ENABLED FLAGS
# =============================================================================
# Only SMC_SIMPLE is active. Others are disabled/archived.

# Active strategy
SMC_SIMPLE_STRATEGY = True

# Disabled strategies (archived in forex_scanner/archive/disabled_strategies/)
EMA_STRATEGY_ENABLED = False
MACD_STRATEGY_ENABLED = False
SCALPING_STRATEGY_ENABLED = False
KAMA_STRATEGY = False
SMC_STRATEGY = False
SMC_STRUCTURE_STRATEGY = False
ICHIMOKU_CLOUD_STRATEGY = False
MEAN_REVERSION_STRATEGY = False
RANGING_MARKET_STRATEGY = False
VOLUME_PROFILE_STRATEGY = False
EMA_DOUBLE_CONFIRMATION_STRATEGY = False
SILVER_BULLET_STRATEGY = False
BOLLINGER_SUPERTREND_STRATEGY = False
ZERO_LAG_STRATEGY = False
ZERO_LAG_STRATEGY_ENABLED = False
USE_ZERO_LAG_STRATEGY = False
MOMENTUM_STRATEGY = False
MOMENTUM_STRATEGY_ENABLED = False
USE_MOMENTUM_STRATEGY = False


def get_enabled_strategy_flags():
    """Get enabled strategies from database, fallback to config flags."""
    try:
        from forex_scanner.services.scanner_config_service import get_scanner_config
        config = get_scanner_config()
        if config.enabled_strategies:
            return config.enabled_strategies
    except Exception:
        pass
    return ['SMC_SIMPLE']


def is_strategy_enabled(strategy_name: str) -> bool:
    """Check if a strategy is enabled. Currently only SMC_SIMPLE is active."""
    if not strategy_name:
        return False
    normalized = strategy_name.upper().replace('_STRATEGY', '').replace('_ENABLED', '').strip()
    return 'SMC_SIMPLE' in normalized or normalized in 'SMC_SIMPLE'


def get_enabled_strategies():
    """Get list of enabled strategy names."""
    return get_enabled_strategy_flags()


# =============================================================================
# PAIR-SPECIFIC SETTINGS (USDCHF optimization)
# =============================================================================

USDCHF_BLOCKED_HOURS_UTC = [7, 8, 9, 10, 11, 17]
USDCHF_PREFERRED_HOURS_UTC = [1, 4, 13, 14, 15, 22]
USDCHF_MIN_STOP_LOSS_PIPS = 18
USDCHF_SL_BUFFER_PIPS = 5
ENABLE_USDCHF_HOUR_FILTER = True
ENABLE_USDCHF_MIN_SL_FILTER = True

PAIR_SPECIFIC_SETTINGS = {
    'USDCHF': {
        'blocked_hours_utc': USDCHF_BLOCKED_HOURS_UTC,
        'preferred_hours_utc': USDCHF_PREFERRED_HOURS_UTC,
        'min_stop_loss_pips': USDCHF_MIN_STOP_LOSS_PIPS,
        'sl_buffer_pips': USDCHF_SL_BUFFER_PIPS,
        'enable_hour_filter': ENABLE_USDCHF_HOUR_FILTER,
        'enable_min_sl_filter': ENABLE_USDCHF_MIN_SL_FILTER,
    },
}


def get_pair_settings(pair: str) -> dict:
    """Get pair-specific settings, returns empty dict if none defined."""
    pair_clean = pair.upper()
    for known_pair in PAIR_SPECIFIC_SETTINGS:
        if known_pair in pair_clean:
            return PAIR_SPECIFIC_SETTINGS[known_pair]
    return {}


def is_pair_hour_blocked(pair: str, hour_utc: int) -> bool:
    """Check if trading is blocked for a pair at given UTC hour."""
    settings = get_pair_settings(pair)
    if not settings.get('enable_hour_filter', False):
        return False
    return hour_utc in settings.get('blocked_hours_utc', [])


def get_pair_min_stop_loss(pair: str) -> float:
    """Get minimum stop loss for a pair, returns 0 if no minimum defined."""
    settings = get_pair_settings(pair)
    if not settings.get('enable_min_sl_filter', False):
        return 0.0
    return settings.get('min_stop_loss_pips', 0.0)


# =============================================================================
# SCANNER CORE SETTINGS (Infrastructure - not strategy-specific)
# =============================================================================

SPREAD_PIPS = 1.5
USE_BID_ADJUSTMENT = False
USE_SIGNAL_PROCESSOR = True

# Strategy weights (legacy - only SMC_SIMPLE is active)
STRATEGY_WEIGHT_KAMA = 0.0
STRATEGY_WEIGHT_ZERO_LAG = 0.0

# =============================================================================
# TRADING SETTINGS
# =============================================================================

AUTO_TRADING_ENABLED = True
ENABLE_ORDER_EXECUTION = True

# Position sizing (fixed at 1 mini lot)
DEFAULT_POSITION_SIZE = 1.0
FIXED_POSITION_SIZE = 1.0
BASE_POSITION_SIZE = 1.0
MIN_POSITION_SIZE = 0.01
MAX_POSITION_SIZE = 1.0

# Risk management
ACCOUNT_BALANCE = 10000
RISK_PER_TRADE = 0.02
RISK_PER_TRADE_PERCENT = 0.02
MAX_RISK_PER_TRADE = 30
MAX_DAILY_TRADES = 10
MAX_CONCURRENT_POSITIONS = 3
MAX_SIGNALS_PER_HOUR = 10
MAX_SIGNALS_PER_DAY = 20
SIGNAL_COOLDOWN_MINUTES = 15

# Order parameters
DEFAULT_STOP_DISTANCE = 20
DEFAULT_RISK_REWARD = 2.0
PIP_VALUE = 1.0
ORDER_LABEL_PREFIX = "ForexScanner"
DYNAMIC_STOPS = True
CONFIDENCE_BASED_SIZING = False

# Order retry/circuit breaker
ORDER_MAX_RETRIES = 3
ORDER_RETRY_BASE_DELAY = 2.0
ORDER_CONNECT_TIMEOUT = 10.0
ORDER_READ_TIMEOUT = 45.0
ORDER_CIRCUIT_BREAKER_THRESHOLD = 5
ORDER_CIRCUIT_BREAKER_RECOVERY = 300.0

# Claude rate limiting
CLAUDE_MAX_REQUESTS_PER_MINUTE = 50
CLAUDE_MAX_REQUESTS_PER_DAY = 1000
CLAUDE_MIN_CALL_INTERVAL = 1.2

# =============================================================================
# TIMEZONE AND SCHEDULING
# =============================================================================

USER_TIMEZONE = 'Europe/Stockholm'
DATABASE_TIMEZONE = 'UTC'
MARKET_OPEN_HOUR_LOCAL = 8
MARKET_CLOSE_HOUR_LOCAL = 22
TRADING_CUTOFF_TIME_UTC = 20
ENABLE_TRADING_TIME_CONTROLS = True

SCHEDULED_SCAN_INTERVAL_MINUTES = 1
HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_DB_CHECK = True

# Legacy trading hours (kept for backward compatibility)
TRADING_HOURS_LEGACY = {
    'start_hour': 0,
    'end_hour': 23,
    'enabled_days': [0, 1, 2, 3, 4, 6],
    'enable_24_5': True
}

# =============================================================================
# SIGNAL QUALITY SETTINGS
# =============================================================================

MIN_BARS_FOR_SIGNAL = 50
ALLOW_TRANSITION_SIGNALS = True
REQUIRE_VOLUME_CONFIRMATION = False
MIN_VOLUME_RATIO = 0.5
REQUIRE_NEW_CROSSOVER = False
ENABLE_BB_FILTER = False
ENABLE_BB_EXTREMES_FILTER = False
BB_DISTANCE_THRESHOLD_PCT = 0.01
ENABLE_CANDLE_COLOR_FILTER = False

# EMA settings
USE_ENHANCED_EMA_LOGIC = True
EMA_EPSILON = 1e-4
REQUIRE_PRICE_VS_EMA21 = False
ALWAYS_INCLUDE_EMA200 = True
ENABLE_EMA200_TREND_FILTER = True

# Performance thresholds
VOLUME_SPIKE_THRESHOLD = 1.0
CONSOLIDATION_THRESHOLD_PIPS = 5
REJECTION_WICK_THRESHOLD = 0.1

# Data fetching lookbacks
LOOKBACK_HOURS_5M = 1000
LOOKBACK_HOURS_15M = 1000
LOOKBACK_HOURS_1H = 200

# =============================================================================
# DATA QUALITY AND VALIDATION
# =============================================================================

ENABLE_DATA_QUALITY_FILTERING = False
MAX_PRICE_DISCREPANCY_PIPS = 10.0
MIN_QUALITY_SCORE_FOR_TRADING = 0.5
DATA_QUALITY_MONITORING_ENABLED = True
DATA_QUALITY_LOG_LEVEL = 'WARNING'
TRADING_SAFETY_CHECKS_ENABLED = True
BLOCK_TRADING_ON_DATA_ISSUES = True

# Stream validation
ENABLE_STREAM_API_VALIDATION = True
STREAM_VALIDATION_DELAY_SECONDS = 45
STREAM_VALIDATION_FREQUENCY = 3
ENABLE_AUTOMATIC_PRICE_CORRECTION = True
STREAM_VALIDATION_THRESHOLDS = {
    'MINOR': 1.0,
    'MODERATE': 3.0,
    'MAJOR': 10.0,
    'CRITICAL': 25.0
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FILE = 'forex_scanner.log'
LOG_DIR = 'logs'
LOG_RETENTION_DAYS = 30
LOG_TIMEZONE = 'Europe/Stockholm'
CONSOLE_LOG_LEVEL = 'INFO'
DETAILED_LOGGING = True
COMPRESS_OLD_LOGS = True

MAX_LOG_FILE_SIZE = 50 * 1024 * 1024
MAX_DEBUG_FILES = 3

ENABLE_SIGNAL_LOGGING = True
ENABLE_ERROR_LOGGING = True
ENABLE_PERFORMANCE_LOGGING = True
ENABLE_DEBUG_LOGGING = True

# Dynamic config logging
DYNAMIC_CONFIG_LOG_LEVEL = 'INFO'
LOG_CONFIG_SELECTIONS = True
LOG_PERFORMANCE_UPDATES = True
LOG_MARKET_CONDITIONS = True

# Log filters
SIGNAL_LOG_KEYWORDS = [
    'signal', 'BEAR', 'BUY', 'SELL', 'Final Signal', 'Final Confidence',
    'Strategy signals', 'STRATEGY BREAKDOWN', 'Final Strategy',
    'confidence', 'crossover', 'EMA', 'KAMA', 'Combined'
]

PERFORMANCE_LOG_KEYWORDS = [
    'processing time', 'scan complete', 'database', 'connection',
    'query time', 'analysis time', 'total time', 'memory usage',
    'signals found', 'pairs processed', 'milliseconds', 'seconds',
    'initialized', 'startup', 'shutdown'
]

# Rejection logging
LOG_REJECTION_DETAILS = True
EMERGENCY_BYPASS_SAFETY_FILTERS = False
TRACK_SAFETY_FILTER_STATS = True

# =============================================================================
# SUPPORT/RESISTANCE SETTINGS
# =============================================================================

ENABLE_SR_VALIDATION = True
SR_LEFT_BARS = 15
SR_RIGHT_BARS = 15
SR_VOLUME_THRESHOLD = 20.0
SR_LEVEL_TOLERANCE_PIPS = 3.0
SR_MIN_LEVEL_DISTANCE_PIPS = 20.0
SR_ANALYSIS_TIMEFRAME = '15m'
SR_LOOKBACK_HOURS = 72
MIN_BARS_FOR_SR_ANALYSIS = 100
SR_CACHE_DURATION_MINUTES = 10
ENABLE_ENHANCED_SR_VALIDATION = True
SR_RECENT_FLIP_BARS = 50
SR_MIN_FLIP_STRENGTH = 0.6

# =============================================================================
# LARGE CANDLE FILTER
# =============================================================================

ENABLE_LARGE_CANDLE_FILTER = True
LARGE_CANDLE_ATR_MULTIPLIER = 2.5
CONSECUTIVE_LARGE_CANDLES_THRESHOLD = 2
MOVEMENT_LOOKBACK_PERIODS = 3
EXCESSIVE_MOVEMENT_THRESHOLD_PIPS = 15
LARGE_CANDLE_FILTER_COOLDOWN = 3
PARABOLIC_ACCELERATION_THRESHOLD = 1.5
LARGE_CANDLE_FILTER_PRESET = 'balanced'

LARGE_CANDLE_FILTER_PRESETS = {
    'strict': {
        'LARGE_CANDLE_ATR_MULTIPLIER': 2.0,
        'CONSECUTIVE_LARGE_CANDLES_THRESHOLD': 1,
        'EXCESSIVE_MOVEMENT_THRESHOLD_PIPS': 10,
        'LARGE_CANDLE_FILTER_COOLDOWN': 5
    },
    'balanced': {
        'LARGE_CANDLE_ATR_MULTIPLIER': 2.5,
        'CONSECUTIVE_LARGE_CANDLES_THRESHOLD': 2,
        'EXCESSIVE_MOVEMENT_THRESHOLD_PIPS': 15,
        'LARGE_CANDLE_FILTER_COOLDOWN': 3
    },
    'permissive': {
        'LARGE_CANDLE_ATR_MULTIPLIER': 3.0,
        'CONSECUTIVE_LARGE_CANDLES_THRESHOLD': 3,
        'EXCESSIVE_MOVEMENT_THRESHOLD_PIPS': 20,
        'LARGE_CANDLE_FILTER_COOLDOWN': 2
    }
}

# =============================================================================
# MULTI-TIMEFRAME ANALYSIS
# =============================================================================

ENABLE_MTF_ANALYSIS = True

MTF_CONFIG = {
    'require_alignment': True,
    'min_aligned_timeframes': 2,
    'check_timeframes': ['5m', '15m', '1h'],
    'alignment_threshold': 0.6,
    'confidence_boost_max': 0.15
}

TIMEFRAME_HIERARCHY = {
    '1m': ['5m', '15m'],
    '5m': ['15m', '1h'],
    '15m': ['1h', '4h'],
    '30m': ['1h', '4h'],
    '1h': ['4h', '1d']
}

STRATEGY_TIMEFRAME_MAP = {
    '1m': ['scalping', 'aggressive'],
    '5m': ['scalping', 'default', 'aggressive'],
    '15m': ['default', 'conservative'],
    '1h': ['swing', 'conservative'],
    '4h': ['swing', 'crypto'],
    '1d': ['swing', 'crypto']
}

# =============================================================================
# ADVANCED FILTERING
# =============================================================================

ADVANCED_FILTERING = {
    'bollinger_bands': {
        'enabled': True,
        'period': 14,
        'std_dev': 1.8,
        'extremes_only': False,
        'middle_band_filter': True
    },
    'atr_filter': {
        'enabled': True,
        'period': 8,
        'min_atr_multiplier': 0.5,
        'max_atr_multiplier': 3.0
    },
    'rsi_confirmation': {
        'enabled': False,
        'period': 14,
        'overbought': 70,
        'oversold': 30,
        'require_divergence': False
    }
}

# Adaptive detection
ADAPTIVE_HIGH_VOLATILITY_THRESHOLD = 1.5
ADAPTIVE_LOW_VOLATILITY_THRESHOLD = 0.5
ADAPTIVE_TREND_STRENGTH_THRESHOLD = 0.7
MIN_MARKET_EFFICIENCY = 0.02
EMA_STRICT_ALIGNMENT = False
EMERGENCY_DEBUG_MODE = True
BACKTEST_MODE_RELAXED = True

# =============================================================================
# NOTIFICATIONS
# =============================================================================

NOTIFICATIONS = {
    'console': True,
    'file': True,
    'email': False,
    'webhook': False
}

# =============================================================================
# BACKTESTING
# =============================================================================

DEFAULT_BACKTEST_DAYS = 30
BACKTEST_LOOKBACK_BARS = 1000

# Integration settings
SCANNER_USE_DYNAMIC_EMA = True
SCANNER_CONFIG_REFRESH_ON_SCAN = False
BACKTEST_USE_DYNAMIC_EMA = False
BACKTEST_COMPARE_CONFIGS = True
WEB_INTERFACE_SHOW_DYNAMIC_STATUS = True
WEB_INTERFACE_ALLOW_CONFIG_OVERRIDE = True

# =============================================================================
# MARKET HOURS AND CLOSURE
# =============================================================================

MARKET_CLOSURE_SETTINGS = {
    'save_signals_when_closed': True,
    'execute_signals_when_closed': False,
    'log_market_status': True,
    'queue_signals_for_open': True,
}

TIMESTAMP_VALIDATION_SETTINGS = {
    'fix_epoch_timestamps': True,
    'log_timestamp_issues': True,
    'allow_none_timestamps': True,
    'validate_against_current_time': True,
    'reject_future_timestamps': False,
}

FOREX_MARKET_HOURS = {
    'open_day': 0,
    'open_hour': 22,
    'close_day': 4,
    'close_hour': 22,
    'timezone': 'UTC',
}

ALERT_PROCESSING = {
    'process_during_closure': True,
    'mark_closure_status': True,
    'enhanced_logging': True,
    'separate_closure_queue': False,
}

SAVE_ALERTS_WHEN_MARKET_CLOSED = MARKET_CLOSURE_SETTINGS['save_signals_when_closed']
LOG_TIMESTAMP_CONVERSIONS = TIMESTAMP_VALIDATION_SETTINGS['log_timestamp_issues']
VALIDATE_MARKET_HOURS = True


def is_market_open_now() -> bool:
    """Check if forex market is currently open based on UTC time."""
    from datetime import datetime, timezone

    current_utc = datetime.now(timezone.utc)
    weekday = current_utc.weekday()
    hour = current_utc.hour

    if weekday == 5:  # Saturday
        return False
    elif weekday == 6:  # Sunday
        return hour >= 22
    elif weekday == 4:  # Friday
        return hour < 22

    return True


def get_market_status_info() -> dict:
    """Get detailed market status information."""
    from datetime import datetime, timezone

    current_utc = datetime.now(timezone.utc)
    is_open = is_market_open_now()
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    return {
        'is_open': is_open,
        'current_time_utc': current_utc.strftime('%Y-%m-%d %H:%M:%S UTC'),
        'weekday': weekday_names[current_utc.weekday()],
        'hour_utc': current_utc.hour,
        'status': 'OPEN' if is_open else 'CLOSED',
        'settings': MARKET_CLOSURE_SETTINGS
    }


# =============================================================================
# DYNAMIC CLI COMMANDS
# =============================================================================

DYNAMIC_CONFIG_COMMANDS = {
    'show_configs': {'enabled': True, 'description': 'Show current dynamic configurations'},
    'config_performance': {'enabled': True, 'description': 'Show configuration performance statistics'},
    'optimize_configs': {'enabled': True, 'description': 'Optimize configurations for specific epics'},
    'market_analysis': {'enabled': True, 'description': 'Show market condition analysis'}
}

# =============================================================================
# STRATEGY INDICATOR MAPPING (Legacy - for disabled strategies)
# =============================================================================

STRATEGY_INDICATOR_MAP = {
    'ema': ['ema'],
    'macd': ['macd'],
    'kama': ['kama'],
    'bb_supertrend': ['bb_supertrend'],
    'momentum': ['momentum'],
    'zero_lag': ['zero_lag_ema'],
    'volume': ['volume'],
    'support_resistance': ['support_resistance'],
    'behavior': ['behavior']
}

REQUIRED_INDICATORS_BY_STRATEGY = {
    'ema': ['ema', 'close', 'high', 'low'],
    'macd': ['macd', 'ema', 'close'],
    'kama': [],
    'momentum': [],
    'zero_lag_ema': [],
    'bb_supertrend': [],
    'combined': ['ema', 'macd']
}

STRATEGY_CONFIG_MODULES = {
    'momentum': 'configdata.strategies.config_momentum_strategy',
    'zero_lag_ema': 'configdata.config_zerolag_strategy'
}

# =============================================================================
# RISK MANAGEMENT PRESETS (Legacy)
# =============================================================================

RISK_MANAGEMENT_CONFIG = {
    'scalping': {
        'max_risk_per_trade': 0.5,
        'stop_loss_pips': 8,
        'take_profit_pips': 12,
        'max_trades_per_hour': 3
    },
    'swing': {
        'max_risk_per_trade': 2.0,
        'stop_loss_pips': 30,
        'take_profit_pips': 60,
        'max_trades_per_day': 2
    },
    'conservative': {
        'max_risk_per_trade': 1.0,
        'stop_loss_pips': 20,
        'take_profit_pips': 40,
        'max_trades_per_day': 3
    }
}

MARKET_CONDITION_CONFIG = {
    'trending': {'ema_config': 'aggressive', 'confidence_boost': 0.1, 'volume_multiplier': 1.2},
    'ranging': {'ema_config': 'conservative', 'confidence_penalty': 0.05, 'require_strong_breakout': True},
    'high_volatility': {'ema_config': 'conservative', 'min_confidence': 0.8, 'reduce_position_size': 0.5},
    'low_volatility': {'ema_config': 'aggressive', 'min_confidence': 0.6, 'increase_position_size': 1.2}
}

TIME_BASED_CONFIG = {
    'london_session': {'preferred_pairs': ['GBPUSD', 'EURGBP', 'GBPJPY'], 'strategy_boost': 'aggressive', 'volume_threshold_multiplier': 1.5},
    'ny_session': {'preferred_pairs': ['EURUSD', 'USDCAD', 'USDJPY'], 'strategy_boost': 'default', 'volume_threshold_multiplier': 1.3},
    'asian_session': {'preferred_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'], 'strategy_boost': 'conservative', 'volume_threshold_multiplier': 0.8},
    'overlap_sessions': {'strategy_boost': 'aggressive', 'volume_threshold_multiplier': 2.0, 'confidence_boost': 0.15}
}

PERFORMANCE_CONFIG = {
    'enable_parallel_processing': True,
    'max_worker_threads': 4,
    'cache_indicators': True,
    'cache_duration_minutes': 5,
    'batch_process_signals': True,
    'lazy_load_historical_data': True
}

# =============================================================================
# SAFETY FILTER STATS (Runtime state)
# =============================================================================

SAFETY_FILTER_STATS = {
    'total_signals_processed': 0,
    'total_rejections': 0,
    'rejections_by_filter': {
        'ema200_trend': 0,
        'macd_momentum': 0,
        'ema_stack': 0,
        'consensus': 0,
        'circuit_breaker': 0
    }
}


def get_safety_filter_stats():
    """Get current safety filter statistics."""
    return SAFETY_FILTER_STATS.copy()


def reset_safety_filter_stats():
    """Reset safety filter statistics."""
    global SAFETY_FILTER_STATS
    SAFETY_FILTER_STATS = {
        'total_signals_processed': 0,
        'total_rejections': 0,
        'rejections_by_filter': {
            'ema200_trend': 0,
            'macd_momentum': 0,
            'ema_stack': 0,
            'consensus': 0,
            'circuit_breaker': 0
        }
    }
    print("Safety filter statistics reset")


# =============================================================================
# MARKET INTELLIGENCE - DATABASE DRIVEN
# =============================================================================

# Intelligence config is now read from database via IntelligenceConfigService
# These are minimal fallback defaults only - actual values come from:
#   Database: strategy_config.intelligence_global_config
#   Service: forex_scanner.services.intelligence_config_service

INTELLIGENCE_PRESET = 'collect_only'
INTELLIGENCE_MODE = 'live_only'
ENABLE_MARKET_INTELLIGENCE = True

# =============================================================================
# END OF CONFIG
# =============================================================================
