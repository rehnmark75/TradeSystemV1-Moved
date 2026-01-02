# config.py
"""
Configuration settings for the Forex Scanner
"""
import os
from typing import List

# Import EMA strategy-specific configurations for MACD momentum filter
try:
    from configdata.strategies.config_ema_strategy import (
        MACD_MOMENTUM_FILTER_ENABLED,
        MACD_VALIDATION_MODE,
        MACD_TREND_SENSITIVITY,
        MACD_HISTOGRAM_LOOKBACK,
        MACD_MIN_SLOPE_THRESHOLD,
        MACD_SENSITIVITY_SETTINGS,
        MACD_VALIDATION_MODES,
        MACD_MOMENTUM_DEBUG_LOGGING
    )
except ImportError as e:
    # Fallback configurations if import fails
    print(f"Warning: Could not import EMA strategy MACD config: {e}")
    MACD_MOMENTUM_FILTER_ENABLED = True
    MACD_VALIDATION_MODE = 'strict_blocking'  # Changed from 'slope_aware' to be more strict
    MACD_TREND_SENSITIVITY = 'normal'
    MACD_HISTOGRAM_LOOKBACK = 3  # Changed from 1 to 3 for better trend detection
    MACD_MIN_SLOPE_THRESHOLD = 0.00005  # More sensitive threshold
    MACD_MOMENTUM_DEBUG_LOGGING = True
    MACD_SENSITIVITY_SETTINGS = {
        'normal': {
            'lookback': 3,
            'min_slope': 0.00005,
            'description': 'Balanced - moderate sensitivity to momentum changes'
        }
    }
    MACD_VALIDATION_MODES = {
        'strict_blocking': {
            'description': 'Block signals when MACD momentum opposes signal direction',
            'allow_neutral': True,
            'block_opposite': True
        }
    }

# ================== FIXED SL/TP OVERRIDE ==================
# DEPRECATED: These settings are now managed via the database (strategy_config.smc_simple_global_config)
# Use Streamlit UI -> SMC Config -> Global Settings -> Risk Management to configure
# Or update directly in database: UPDATE smc_simple_global_config SET fixed_stop_loss_pips = X WHERE is_active = TRUE
# Keeping these for backwards compatibility - will be removed in future version
# ---
# FIXED_SL_TP_OVERRIDE_ENABLED = True
# FIXED_STOP_LOSS_PIPS = 9             # Fixed SL in pips (when override enabled)
# FIXED_TAKE_PROFIT_PIPS = 15          # Fixed TP in pips (when override enabled)

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/forex")

# API Keys
# CLAUDE_API_KEY is provided via environment variable or Azure Key Vault
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', None)  # None if not available

# ================== MinIO Configuration ==================
# Object storage for Claude vision analysis charts (30-day retention)
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'claude-charts')
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').lower() == 'true'
MINIO_CHART_RETENTION_DAYS = int(os.getenv('MINIO_CHART_RETENTION_DAYS', '30'))
MINIO_ENABLED = os.getenv('MINIO_ENABLED', 'true').lower() == 'true'  # Set False to use disk storage
MINIO_PUBLIC_URL = os.getenv('MINIO_PUBLIC_URL', f"http://{MINIO_ENDPOINT}")

# Extended PAIR_INFO in config.py to include JPY pairs

# Pair Information for Pip Calculation
PAIR_INFO = {
    # Major USD pairs
    'CS.D.EURUSD.CEEM.IP': {'pair': 'EURUSD', 'pip_multiplier': 10000},
    'CS.D.GBPUSD.MINI.IP': {'pair': 'GBPUSD', 'pip_multiplier': 10000},
    'CS.D.AUDUSD.MINI.IP': {'pair': 'AUDUSD', 'pip_multiplier': 10000},
    'CS.D.NZDUSD.MINI.IP': {'pair': 'NZDUSD', 'pip_multiplier': 10000},
    'CS.D.USDCHF.MINI.IP': {'pair': 'USDCHF', 'pip_multiplier': 10000},
    'CS.D.USDCAD.MINI.IP': {'pair': 'USDCAD', 'pip_multiplier': 10000},
    
    # JPY pairs - all use 100 pip multiplier
    'CS.D.USDJPY.MINI.IP': {'pair': 'USDJPY', 'pip_multiplier': 100},
    'CS.D.EURJPY.MINI.IP': {'pair': 'EURJPY', 'pip_multiplier': 100},
    'CS.D.GBPJPY.MINI.IP': {'pair': 'GBPJPY', 'pip_multiplier': 100},
    'CS.D.AUDJPY.MINI.IP': {'pair': 'AUDJPY', 'pip_multiplier': 100},
    'CS.D.CADJPY.MINI.IP': {'pair': 'CADJPY', 'pip_multiplier': 100},
    'CS.D.CHFJPY.MINI.IP': {'pair': 'CHFJPY', 'pip_multiplier': 100},
    'CS.D.NZDJPY.MINI.IP': {'pair': 'NZDJPY', 'pip_multiplier': 100},
    
    # Cross pairs (if you add them later)
    'CS.D.EURGBP.MINI.IP': {'pair': 'EURGBP', 'pip_multiplier': 10000},
    'CS.D.EURAUD.MINI.IP': {'pair': 'EURAUD', 'pip_multiplier': 10000},
    'CS.D.GBPAUD.MINI.IP': {'pair': 'GBPAUD', 'pip_multiplier': 10000}
}

# Updated EPIC_LIST to include JPY pairs (if you want to trade them)
EPIC_LIST: List[str] = [
    # Current pairs
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP', 
    'CS.D.USDJPY.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.USDCHF.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    
    # Additional JPY pairs
    'CS.D.EURJPY.MINI.IP',
    'CS.D.AUDJPY.MINI.IP'
]

# =============================================================================
# DEPRECATED: SCANNER CORE SETTINGS - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config
#
# To access settings programmatically:
#   from forex_scanner.services.scanner_config_service import get_scanner_config
#   config = get_scanner_config()
#   print(config.scan_interval)
#
# The following values are kept for reference only and are NOT used by the scanner.
# -----------------------------------------------------------------------------
# SCAN_INTERVAL = 120  # MIGRATED to database
# MIN_CONFIDENCE = 0.40  # MIGRATED to database
# DEFAULT_TIMEFRAME = '15m'  # MIGRATED to database
# USE_1M_BASE_SYNTHESIS = True  # MIGRATED to database
# SCAN_ALIGN_TO_BOUNDARIES = True  # MIGRATED to database
# SCAN_BOUNDARY_OFFSET_SECONDS = 60  # MIGRATED to database
# =============================================================================

# Non-migrated scanner settings (still used from config.py)
SPREAD_PIPS = 1.5   # default spread for BID/ASK adjustment
USE_BID_ADJUSTMENT = False  # whether to adjust BID prices to MID
#=============================================================================
# DATA FETCHER OPTIMIZATION CONFIGURATION
# =============================================================================

# Enable optimizations
ENABLE_DATA_CACHE = False          # 5-minute data caching (DISABLED for fresh data)
REDUCED_LOOKBACK_HOURS = True      # Use smart lookback times
LAZY_INDICATOR_LOADING = True      # Load indicators on demand
DATA_BATCH_SIZE = 10000           # Batch size for query results (increased for 1H resampling from 5m data)

# Conditional indicators (disable unused ones)
ENABLE_SUPPORT_RESISTANCE = True
ENABLE_VOLUME_ANALYSIS = True
ENABLE_BEHAVIOR_ANALYSIS = False   # Disable if not used

# Signal Strategy Mode
REQUIRE_VOLUME_CONFIRMATION = False # Disabled in simple mode
REQUIRE_NEW_CROSSOVER = False  # Disabled in simple mode

# Combined Strategy Settings - REMOVED: Strategy was disabled and unused, cleaned up codebase
  
STRATEGY_WEIGHT_KAMA = 0.0             # KAMA strategy weight (if enabled)
#STRATEGY_WEIGHT_BB_SUPERTREND = 0.20    # BB+SuperTrend weight (if enabled)
#STRATEGY_WEIGHT_MOMENTUM_BIAS = 0.05    # Momentum Bias weight (if enabled)
STRATEGY_WEIGHT_ZERO_LAG = 0.0         # Zero Lag EMA weight (if enabled)

# Advanced Combination Rules - REMOVED: Combined strategy removed, these are no longer used

# Strategy Selection for Enhanced Signal Processing Pipeline, name must match strategy name for it to be included in the

# SignalProcessor Configuration
USE_SIGNAL_PROCESSOR = True  # Enable SignalProcessor for Smart Money

# =============================================================================
# DEPRECATED: SMC CONFLICT FILTER SETTINGS - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > SMC Conflict
#
# To access settings programmatically:
#   from forex_scanner.services.scanner_config_service import get_scanner_config
#   config = get_scanner_config()
#   print(config.smc_conflict_filter_enabled)
# -----------------------------------------------------------------------------
# SMART_MONEY_READONLY_ENABLED = True  # MIGRATED to database
# SMART_MONEY_ANALYSIS_TIMEOUT = 5.0   # MIGRATED to database
# SMC_CONFLICT_FILTER_ENABLED = True   # MIGRATED to database
# SMC_MIN_DIRECTIONAL_CONSENSUS = 0.3  # MIGRATED to database
# SMC_REJECT_ORDER_FLOW_CONFLICT = True  # MIGRATED to database
# SMC_REJECT_RANGING_STRUCTURE = True    # MIGRATED to database
# SMC_MIN_STRUCTURE_SCORE = 0.5          # MIGRATED to database
# =============================================================================

# DEPRECATED: SELECTED_STRATEGIES hardcoded list replaced with dynamic detection
# 
# OLD PROBLEM: Required manual maintenance of strategy name variations:
#   SELECTED_STRATEGIES = ['Combined', 'combined', 'COMBINED', 'EMA', 'ema', ...]
#   - Broke when strategy names changed
#   - Required updating for every new strategy
#   - Case-sensitive exact matching only
#
# NEW SOLUTION: Intelligent pattern matching against config flags
#   - Automatically detects enabled strategies from *_STRATEGY = True flags  
#   - Handles case variations, abbreviations, and common naming patterns
#   - Self-maintaining - no manual updates needed
#   - Uses fuzzy matching: 'EMA', 'ema', 'Simple_EMA' all work if SIMPLE_EMA_STRATEGY = True

def get_enabled_strategy_flags():
    """
    Get all enabled strategy flags from config.
    Returns a list of strategy flag names that are set to True.
    """
    import sys
    current_module = sys.modules[__name__]
    enabled_flags = []
    
    # Find all variables ending in _STRATEGY that are set to True
    for name in dir(current_module):
        if name.endswith('_STRATEGY') or name.endswith('_STRATEGY_ENABLED'):
            try:
                value = getattr(current_module, name, False)
                if value is True:
                    enabled_flags.append(name)
            except:
                continue
    
    return enabled_flags

def is_strategy_enabled(strategy_name: str) -> bool:
    """
    Check if a strategy is enabled using fuzzy matching against config flags.
    No hardcoded mappings needed - uses intelligent pattern matching.
    """
    if not strategy_name:
        return False
    
    # Get enabled strategy flags
    enabled_flags = get_enabled_strategy_flags()
    if not enabled_flags:
        return True  # If no flags found, allow all (safe default)
    
    # Normalize strategy name for comparison
    strategy_normalized = strategy_name.upper().strip()
    
    # Remove common variations and prefixes/suffixes
    strategy_clean = strategy_normalized
    for prefix in ['CS.D.', 'SIMPLE_', 'STANDARD_']:
        strategy_clean = strategy_clean.replace(prefix, '')
    for suffix in ['.MINI.IP', '_STRATEGY', '_ENABLED']:
        strategy_clean = strategy_clean.replace(suffix, '')
    
    # Try multiple matching approaches
    for flag in enabled_flags:
        flag_clean = flag.upper().replace('_STRATEGY', '').replace('_ENABLED', '')
        
        # Approach 1: Exact match after cleanup
        if strategy_clean == flag_clean:
            return True
        
        # Approach 2: Strategy name contained in flag
        if strategy_clean in flag_clean or flag_clean in strategy_clean:
            return True
        
        # Approach 3: Handle common abbreviations and variations
        strategy_variations = {
            'EMA': ['SIMPLE_EMA', 'MOVING_AVERAGE'],
            'MACD': ['MACD_EMA', 'STANDARD_MACD'],
            'ZERO_LAG': ['ZEROLAG', 'ZERO_LAG_SQUEEZE'],
            'BB': ['BOLLINGER', 'BOLLINGER_SUPERTREND'],
            'SUPERTREND': ['BOLLINGER_SUPERTREND'],
            'SMC': ['SMART_MONEY'],
        }
        
        for key, variations in strategy_variations.items():
            if strategy_clean == key or strategy_clean in variations:
                if key in flag_clean or any(var in flag_clean for var in variations):
                    return True
    
    return False

def get_enabled_strategies():
    """
    Get a list of enabled strategies for backward compatibility.
    This is mainly for logging and debugging purposes.
    """
    flags = get_enabled_strategy_flags()
    # Extract readable strategy names from flags
    strategy_names = []
    for flag in flags:
        name = flag.replace('_STRATEGY', '').replace('_ENABLED', '').replace('_', ' ').title()
        strategy_names.append(name)
    return strategy_names

# =============================================================================

# =================================================================
# DEPRECATED: DUPLICATE DETECTION CONFIGURATION - NOW IN DATABASE
# =================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > Duplicate Detection
# -----------------------------------------------------------------------------
# ENABLE_DUPLICATE_CHECK = True  # MIGRATED to database
# DUPLICATE_SENSITIVITY = 'smart'  # MIGRATED to database
# SIGNAL_COOLDOWN_MINUTES = 15  # MIGRATED to database
# =================================================================

# =============================================================================
# EMA200 DISTANCE VALIDATION
# =============================================================================



# =============================================================================
# DEPRECATED: ADX TREND STRENGTH FILTER - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > ADX Filter
# -----------------------------------------------------------------------------
# ADX_FILTER_ENABLED = False  # MIGRATED to database
# ADX_FILTER_MODE = 'moderate'  # MIGRATED to database
# ADX_PERIOD = 14  # MIGRATED to database
# ADX_THRESHOLDS = {...}  # MIGRATED to database as JSONB
# ADX_PAIR_MULTIPLIERS = {...}  # MIGRATED to database as JSONB
# ADX_GRACE_PERIOD_BARS = 2  # MIGRATED to database
# =============================================================================

# LEGACY: Kept for backward compatibility during transition - DO NOT USE
ADX_THRESHOLDS_LEGACY = {
    'STRONG_TREND': 25.0,      # ADX > 25 = Strong trend (allow signals)
    'MODERATE_TREND': 22.0,    # ADX 20-25 = Moderate trend (conditional)
    'WEAK_TREND': 15.0,        # ADX < 20 = Weak/ranging market (filter out)
    'VERY_WEAK': 10.0          # ADX < 15 = Very weak trend (definitely filter)
}

# LEGACY: These are kept only for backward compatibility during transition
# All ADX settings should now be read from database via get_scanner_config()
ADX_FILTER_MODE_LEGACY = 'moderate'
ADX_PERIOD_LEGACY = 14
ADX_PAIR_MULTIPLIERS_LEGACY = {
    'EURUSD': 1.0, 'GBPUSD': 0.9, 'USDJPY': 1.1,
    'EURJPY': 0.85, 'GBPJPY': 0.8, 'USDCHF': 1.0, 'DEFAULT': 1.0
}
ADX_GRACE_PERIOD_BARS_LEGACY = 2

# =============================================================================
# USDCHF PAIR-SPECIFIC OPTIMIZATION (v2.6.0 - 2025-12-23)
# =============================================================================
# Analysis Results:
# - Win rate: 40.7% (37 wins / 52 losses)
# - Total P&L: -4,917 SEK
# - Root cause: Stops too tight (9-13 pips), wrong trading hours, poor R:R
# - 100% loss rate during 8-11 UTC (Asian close/London open transition)
# - 83% loss rate at 17 UTC
# - Best performance: 1, 4, 13-15 UTC (US session)

# USDCHF Blocked Trading Hours (UTC) - 100% loss rate during these hours
USDCHF_BLOCKED_HOURS_UTC = [7, 8, 9, 10, 11, 17]  # Asian close / London open overlap

# USDCHF Preferred Trading Hours (UTC) - Best historical performance
USDCHF_PREFERRED_HOURS_UTC = [1, 4, 13, 14, 15, 22]  # US session focus

# USDCHF Minimum Stop Loss (pips) - Prevent stops that are too tight
# Analysis showed 57% of losses from 8-13 pip stops (noise stopouts)
USDCHF_MIN_STOP_LOSS_PIPS = 18  # Minimum 18 pips for USDCHF

# USDCHF Stop Loss Buffer (additional pips added to calculated SL)
USDCHF_SL_BUFFER_PIPS = 5  # Extra buffer for CHF volatility

# Enable/disable USDCHF-specific filters
ENABLE_USDCHF_HOUR_FILTER = True  # Block trading during bad hours
ENABLE_USDCHF_MIN_SL_FILTER = True  # Enforce minimum stop loss

# Pair-specific settings dictionary (extensible for other pairs)
PAIR_SPECIFIC_SETTINGS = {
    'USDCHF': {
        'blocked_hours_utc': USDCHF_BLOCKED_HOURS_UTC,
        'preferred_hours_utc': USDCHF_PREFERRED_HOURS_UTC,
        'min_stop_loss_pips': USDCHF_MIN_STOP_LOSS_PIPS,
        'sl_buffer_pips': USDCHF_SL_BUFFER_PIPS,
        'enable_hour_filter': ENABLE_USDCHF_HOUR_FILTER,
        'enable_min_sl_filter': ENABLE_USDCHF_MIN_SL_FILTER,
    },
    # Add other pair-specific settings here as needed
}

def get_pair_settings(pair: str) -> dict:
    """Get pair-specific settings, returns empty dict if none defined."""
    # Normalize pair name (handle both 'USDCHF' and 'CS.D.USDCHF.MINI.IP' formats)
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
    blocked_hours = settings.get('blocked_hours_utc', [])
    return hour_utc in blocked_hours

def get_pair_min_stop_loss(pair: str) -> float:
    """Get minimum stop loss for a pair, returns 0 if no minimum defined."""
    settings = get_pair_settings(pair)
    if not settings.get('enable_min_sl_filter', False):
        return 0.0
    return settings.get('min_stop_loss_pips', 0.0)

# ==============================================
# DATA QUALITY AND INTEGRITY SETTINGS
# ==============================================

# Enable data quality filtering (recommended for live trading)
ENABLE_DATA_QUALITY_FILTERING = False

# Maximum allowed price discrepancy for trading (in pips)
MAX_PRICE_DISCREPANCY_PIPS = 10.0

# Minimum quality score for trading (0.0 to 1.0)
MIN_QUALITY_SCORE_FOR_TRADING = 0.5

# Data quality monitoring settings
DATA_QUALITY_MONITORING_ENABLED = True
DATA_QUALITY_LOG_LEVEL = 'WARNING'  # INFO, WARNING, ERROR

# Trading safety settings
TRADING_SAFETY_CHECKS_ENABLED = True
BLOCK_TRADING_ON_DATA_ISSUES = True  # Block trades when data quality is poor

# Time-based trading controls (Weekend protection)
ENABLE_TRADING_TIME_CONTROLS = True    # Enable/disable time-based trading controls
TRADING_CUTOFF_TIME_UTC = 20           # No new trades after this hour UTC (20:00 UTC)
# Note: Position closure happens in fastapi-dev container at 20:30 UTC on Fridays

# ==============================================
# STREAM VS API VALIDATION SETTINGS
# ==============================================

# Enable real-time stream validation against IG REST API
ENABLE_STREAM_API_VALIDATION = True

# Delay before validating completed candles (seconds)
STREAM_VALIDATION_DELAY_SECONDS = 45

# Validation frequency (validate every Nth candle to respect rate limits)  
STREAM_VALIDATION_FREQUENCY = 3

# Automatic price correction for critical discrepancies
ENABLE_AUTOMATIC_PRICE_CORRECTION = True

# Thresholds for validation discrepancy classification (in pips)
STREAM_VALIDATION_THRESHOLDS = {
    'MINOR': 1.0,      # 1 pip - acceptable variance
    'MODERATE': 3.0,   # 3 pips - worth noting
    'MAJOR': 10.0,     # 10 pips - significant issue  
    'CRITICAL': 25.0   # 25+ pips - critical data integrity problem
}

# =============================================================================
# CLI COMMAND CONFIGURATIONS
# =============================================================================

# Dynamic configuration commands
DYNAMIC_CONFIG_COMMANDS = {
    'show_configs': {
        'enabled': True,
        'description': 'Show current dynamic configurations'
    },
    'config_performance': {
        'enabled': True,
        'description': 'Show configuration performance statistics'
    },
    'optimize_configs': {
        'enabled': True,
        'description': 'Optimize configurations for specific epics'
    },
    'market_analysis': {
        'enabled': True,
        'description': 'Show market condition analysis'
    }
}

# Enable EMA Strategy
EMA_STRATEGY = False  # Core EMA strategy

# Enable MACD Strategy
MACD_STRATEGY = False  # Core MACD strategy

# Enable Scalping Strategy (Linda Raschke MACD 3-10-16)
SCALPING_STRATEGY_ENABLED = False  # 🔥 Linda Raschke MACD 3-10-16 adaptive scalping

# Enable KAMA Strategy
KAMA_STRATEGY = False  # ENABLED - Phase 1 optimization complete, ready for testing

# Enable SMC Strategies
# NOTE: There are 3 SMC strategies:
#   - SMC_STRATEGY (old): smc_strategy_fast.py - LEGACY, deprecated
#   - SMC_STRUCTURE_STRATEGY: smc_structure_strategy.py - Complex 17+ filter approach
#   - SMC_SIMPLE_STRATEGY (new): smc_simple_strategy.py - v1.0.0 Simple 3-tier EMA approach
SMC_STRATEGY = False  # OLD SMC strategy (deprecated - use SMC_STRUCTURE_STRATEGY instead)
SMC_STRUCTURE_STRATEGY = False  # Complex SMC Structure strategy (disabled - testing SMC_SIMPLE)
SMC_SIMPLE_STRATEGY = True  # NEW SMC Simple strategy v1.0.0 (3-tier: 4H EMA → 1H swing → 15m entry)

# Enable Ichimoku Cloud Strategy
ICHIMOKU_CLOUD_STRATEGY = False  # Ichimoku Kinko Hyo strategy

# Mean Reversion Strategy
MEAN_REVERSION_STRATEGY = False  # Multi-oscillator mean reversion strategy

# Strategy Configurations - Additional strategies
RANGING_MARKET_STRATEGY = False  # Multi-oscillator ranging market strategy - Re-enabled, ADX filter removed from strategy

# Volume Profile Strategy
VOLUME_PROFILE_STRATEGY = False  # Institutional Volume-by-Price analysis strategy

# EMA Double Confirmation Strategy
EMA_DOUBLE_CONFIRMATION_STRATEGY = False  # DISABLED: Claude consistently rejects these signals

# ICT Silver Bullet Strategy
# Time-based SMC strategy trading during specific windows (3-4AM, 10-11AM, 2-3PM NY)
# Looks for liquidity sweeps + FVG entries
SILVER_BULLET_STRATEGY = False  # ICT Silver Bullet - disabled by default for testing

# KAMA Strategy Configuration moved to configdata/strategies/config_kama_strategy.py


# =============================================================================
# LOGGING CONFIGURATIONS
# =============================================================================

# Dynamic configuration logging
DYNAMIC_CONFIG_LOG_LEVEL = 'INFO'        # Log level for dynamic config messages
LOG_CONFIG_SELECTIONS = True             # Log configuration selections
LOG_PERFORMANCE_UPDATES = True           # Log performance updates
LOG_MARKET_CONDITIONS = True             # Log market condition analysis

# =============================================================================
# INTEGRATION SETTINGS
# =============================================================================

# Scanner integration
SCANNER_USE_DYNAMIC_EMA = True           # Use dynamic EMA in scanner
SCANNER_CONFIG_REFRESH_ON_SCAN = False   # Refresh config on each scan (expensive)

# Backtesting integration
BACKTEST_USE_DYNAMIC_EMA = False         # Use dynamic EMA in backtesting (experimental)
BACKTEST_COMPARE_CONFIGS = True          # Enable configuration comparison in backtests

# Web interface integration
WEB_INTERFACE_SHOW_DYNAMIC_STATUS = True # Show dynamic config status in web interface
WEB_INTERFACE_ALLOW_CONFIG_OVERRIDE = True # Allow manual config override in web interface

# Scalping Strategy Configuration moved to configdata/strategies/config_scalping_strategy.py

# Multi-Timeframe Strategy Mapping
STRATEGY_TIMEFRAME_MAP = {
    '1m': ['scalping', 'aggressive'],
    '5m': ['scalping', 'default', 'aggressive'],
    '15m': ['default', 'conservative'],
    '1h': ['swing', 'conservative'],
    '4h': ['swing', 'crypto'],
    '1d': ['swing', 'crypto']
}

# Dynamic Risk Management
RISK_MANAGEMENT_CONFIG = {
    'scalping': {
        'max_risk_per_trade': 0.5,  # 0.5% per trade
        'stop_loss_pips': 8,
        'take_profit_pips': 12,
        'max_trades_per_hour': 3
    },
    'swing': {
        'max_risk_per_trade': 2.0,  # 2% per trade
        'stop_loss_pips': 30,
        'take_profit_pips': 60,
        'max_trades_per_day': 2
    },
    'conservative': {
        'max_risk_per_trade': 1.0,  # 1% per trade
        'stop_loss_pips': 20,
        'take_profit_pips': 40,
        'max_trades_per_day': 3
    }
}

# Market Condition Adaptive Settings
MARKET_CONDITION_CONFIG = {
    'trending': {
        'ema_config': 'aggressive',
        'confidence_boost': 0.1,
        'volume_multiplier': 1.2
    },
    'ranging': {
        'ema_config': 'conservative',
        'confidence_penalty': 0.05,
        'require_strong_breakout': True
    },
    'high_volatility': {
        'ema_config': 'conservative',
        'min_confidence': 0.8,
        'reduce_position_size': 0.5
    },
    'low_volatility': {
        'ema_config': 'aggressive',
        'min_confidence': 0.6,
        'increase_position_size': 1.2
    }
}

# Time-based Strategy Selection
TIME_BASED_CONFIG = {
    'london_session': {
        'preferred_pairs': ['GBPUSD', 'EURGBP', 'GBPJPY'],
        'strategy_boost': 'aggressive',
        'volume_threshold_multiplier': 1.5
    },
    'ny_session': {
        'preferred_pairs': ['EURUSD', 'USDCAD', 'USDJPY'],
        'strategy_boost': 'default',
        'volume_threshold_multiplier': 1.3
    },
    'asian_session': {
        'preferred_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'],
        'strategy_boost': 'conservative',
        'volume_threshold_multiplier': 0.8
    },
    'overlap_sessions': {
        'strategy_boost': 'aggressive',
        'volume_threshold_multiplier': 2.0,
        'confidence_boost': 0.15
    }
}

# Performance Optimization Settings
PERFORMANCE_CONFIG = {
    'enable_parallel_processing': True,
    'max_worker_threads': 4,
    'cache_indicators': True,
    'cache_duration_minutes': 5,
    'batch_process_signals': True,
    'lazy_load_historical_data': True
}

# Advanced Filtering
ADVANCED_FILTERING = {
    'bollinger_bands': {
        'enabled': True,
        'period': 14,                    # ✅ UPDATED: was 20
        'std_dev': 1.8,                  # ✅ UPDATED: was 2
        'extremes_only': False,
        'middle_band_filter': True
    },
    'atr_filter': {
        'enabled': True,
        'period': 8,                     # ✅ UPDATED: was 14 (to match Supertrend period)
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

MIN_BARS_FOR_SIGNAL = 50  # Minimum bars needed for signal generation

# Signal Quality Settings
ALLOW_TRANSITION_SIGNALS = True # Allow signals during trend transitions
REQUIRE_VOLUME_CONFIRMATION = False  # Enable/disable volume filter
MIN_VOLUME_RATIO = 0.5         # Minimum volume (110% of 20-period average)
REQUIRE_NEW_CROSSOVER = False
ENABLE_BB_FILTER = False
ENABLE_BB_EXTREMES_FILTER = False          # Enable BB extremes filter
BB_DISTANCE_THRESHOLD_PCT = 0.01           # Distance threshold (0.1% = very close)
ENABLE_CANDLE_COLOR_FILTER = False

# Enhanced EMA Detection Settings
USE_ENHANCED_EMA_LOGIC = True     # Use previous proven logic
EMA_EPSILON = 1e-4                # Noise filtering buffer (0.0001)
REQUIRE_PRICE_VS_EMA21 = False     # Require price relationship with EMA 21

# Data Fetching
LOOKBACK_HOURS_5M = 1000
LOOKBACK_HOURS_15M = 1000 
LOOKBACK_HOURS_1H = 200

# =============================================================================
# DEPRECATED: LEGACY CLAUDE ANALYSIS SWITCHES - REMOVED
# =============================================================================
# These redundant switches have been consolidated into the database.
# See: Streamlit > Settings > Scanner Config > Claude AI
# Use: require_claude_approval, claude_vision_enabled, etc.
# -----------------------------------------------------------------------------
# ENABLE_CLAUDE_ANALYSIS = False  # REMOVED - use require_claude_approval
# CLAUDE_ANALYSIS_ENABLED = False  # REMOVED - duplicate
# USE_CLAUDE_ANALYSIS = False  # REMOVED - duplicate
# CLAUDE_ANALYSIS_MODE = "disabled"  # REMOVED - use database settings
# =============================================================================

ENABLE_ORDER_EXECUTION = True  # Set to True when ready for live trading
MAX_SIGNALS_PER_HOUR = 10  # Rate limiting

# =============================================================================
# DEPRECATED: RISK MANAGEMENT - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > Risk Management
# -----------------------------------------------------------------------------
# POSITION_SIZE_PERCENT = 1.0  # MIGRATED to database
# STOP_LOSS_PIPS = 5  # MIGRATED to database
# TAKE_PROFIT_PIPS = 15  # MIGRATED to database
# MAX_OPEN_POSITIONS = 3  # MIGRATED to database
# =============================================================================

# Timezone Settings (NOT migrated - infrastructure setting)
USER_TIMEZONE = 'Europe/Stockholm'  # Your local timezone
DATABASE_TIMEZONE = 'UTC'           # Database timezone (IG data is in UTC)
MARKET_OPEN_HOUR_LOCAL = 8          # Local time
MARKET_CLOSE_HOUR_LOCAL = 22        # Local time

# =============================================================================
# DEPRECATED: TRADING HOURS - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > Trading Hours
# -----------------------------------------------------------------------------
# TRADING_HOURS = {...}  # MIGRATED: trading_start_hour, trading_end_hour
# RESPECT_MARKET_HOURS = False  # MIGRATED to database
# WEEKEND_SCANNING = False  # MIGRATED to database
# =============================================================================

# LEGACY: Kept for backward compatibility during transition
TRADING_HOURS_LEGACY = {
    'start_hour': 0,
    'end_hour': 23,
    'enabled_days': [0, 1, 2, 3, 4, 6],
    'enable_24_5': True
}

# Notification Settings
NOTIFICATIONS = {
    'console': True,
    'file': True,
    'email': False,
    'webhook': False
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'forex_scanner.log'

# Backtesting
DEFAULT_BACKTEST_DAYS = 30
BACKTEST_LOOKBACK_BARS = 1000

# Performance Thresholds
VOLUME_SPIKE_THRESHOLD = 1.0  # 2x average volume
CONSOLIDATION_THRESHOLD_PIPS = 5 # 20 original
REJECTION_WICK_THRESHOLD = 0.1  # 50% of candle range

# =============================================================================
# MARKET INTELLIGENCE CONFIGURATION
# =============================================================================

# Import all market intelligence configurations from dedicated config module
from configdata.market_intelligence_config import *

# =============================================================================
# ORDER EXECUTION CONFIGURATION
# =============================================================================

# Enable/disable automatic order execution
AUTO_TRADING_ENABLED = True  # Set to True to enable live trading #
SIGNAL_COOLDOWN_MINUTES = 15  
# Position sizing - ALWAYS use 1.0 for live trading
DEFAULT_POSITION_SIZE = 1.0  # Fixed position size: 1 mini lot
ACCOUNT_BALANCE = 10000      # Account balance for risk calculation
RISK_PER_TRADE = 0.02        # 2% risk per trade
MAX_POSITION_SIZE = 1.0      # Maximum position size

# Order parameters
DEFAULT_STOP_DISTANCE = 20   # Stop loss distance in pips
DEFAULT_RISK_REWARD = 2.0    # Risk:reward ratio for take profit

# Your existing API configuration
ORDER_API_URL = "http://fastapi-dev:8000/orders/place-order"  # Update if hosted elsewhere
API_SUBSCRIPTION_KEY = "436abe054a074894a0517e5172f0e5b6"

# Epic mapping (internal scanner epic -> external API epic) ← FIXED COMMENT
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

# Trading blacklist - epics to scan but NOT trade
TRADING_BLACKLIST = {
    #"CS.D.EURUSD.CEEM.IP": "No trading permissions for FX_NOR exchange",
    # Add other blocked epics here as needed
}

# FIXED: Create correct reverse mapping for order executor
REVERSE_EPIC_MAP = {}
for scanner_epic, api_epic in EPIC_MAP.items():  # ← FIXED variable names
    REVERSE_EPIC_MAP[api_epic] = scanner_epic



# Trading parameters
DEFAULT_RISK_REWARD = 2.0          # 2:1 risk/reward ratio
DEFAULT_STOP_DISTANCE = 20         # Default stop loss in pips
DEFAULT_POSITION_SIZE = 1.0       # None = use broker default, or set specific size
DYNAMIC_STOPS = True               # Adjust stops based on signal confidence

# Risk management
MAX_DAILY_TRADES = 10              # Maximum trades per day
MAX_CONCURRENT_POSITIONS = 3       # Maximum open positions

# Order labeling
ORDER_LABEL_PREFIX = "ForexScanner"  # Prefix for order labels

# =============================================================================
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================

# POSITION SIZING METHOD (choose one)
# ------------------------------------

# Method 1: Fixed Position Size (simplest) - ALWAYS use 1.0
FIXED_POSITION_SIZE = 1.0   # Fixed: 1 mini lot
# Note: DEFAULT_POSITION_SIZE is defined above at line 649 as 1.0

# Method 2: Risk-Based Position Sizing (RECOMMENDED)
RISK_PER_TRADE_PERCENT = 0.02  # Risk 2% of account per trade
ACCOUNT_BALANCE = 10000        # Your account balance in account currency
PIP_VALUE = 1.0               # Value per pip (depends on broker/pair)

# Method 3: Confidence-Based Position Sizing
CONFIDENCE_BASED_SIZING = False  # DISABLED - always use fixed 1.0
BASE_POSITION_SIZE = 1.0         # Base size: 1 mini lot

# STOP LOSS MANAGEMENT
# --------------------
DYNAMIC_STOPS = True           # Adjust stops based on signal confidence
DEFAULT_STOP_DISTANCE = 20     # Default stop in pips

# POSITION LIMITS
# ---------------
MIN_POSITION_SIZE = 0.01       # Minimum position size
MAX_POSITION_SIZE = 1.0        # Maximum position size
MAX_RISK_PER_TRADE = 30       # Maximum $ risk per trade
MAX_DAILY_TRADES = 10          # Maximum trades per day
MAX_CONCURRENT_POSITIONS = 3   # Maximum open positions

# SIGNAL REQUIREMENTS
# -------------------

# =============================================================================
# DIRECTION MAPPING (automatic)
# =============================================================================
# Scanner Signal → Trading Direction
# 'BULL' signal → 'BUY' order
# 'BEAR' signal → 'SELL' order

# Docker and Scheduling Configuration
SCHEDULED_SCAN_INTERVAL_MINUTES = 1    # How often to scan (minutes)
HEARTBEAT_INTERVAL_SECONDS = 30        # Heartbeat frequency (seconds)

# Alternative database check for heartbeat
HEARTBEAT_DB_CHECK = True              # Enable DB health check in heartbeat

# Enhanced Trading Configuration
# MIN_CONFIDENCE_FOR_TRADING removed - use MIN_CONFIDENCE_FOR_ORDERS instead
# NOTE: Claude settings moved to database - see Scanner Config > Claude AI
MAX_SIGNALS_PER_DAY = 20           # Daily signal limit
MIN_VOLUME_RATIO = 1.2             # Minimum volume confirmation

# DEPRECATED: These Claude settings are now in database
# CLAUDE_MIN_CONFIDENCE_THRESHOLD = 0.8  # MIGRATED - use min_claude_quality_score
# CLAUDE_ANALYSIS_MODE = 'strategic_minimal'  # REMOVED
# CLAUDE_STRATEGIC_FOCUS = 'learning'  # REMOVED

# Trading Hours (DEPRECATED - now in database)
# RESPECT_TRADING_HOURS = False  # MIGRATED to database
# TRADING_START_HOUR = 0  # MIGRATED to database
# TRADING_END_HOUR = 23  # MIGRATED to database

# Claude Analysis directory (kept for backward compatibility)
CLAUDE_ANALYSIS_DIR = 'claude_analysis'  # Directory for saved analyses

# Multi-Timeframe Analysis Settings
ENABLE_MULTI_TIMEFRAME_ANALYSIS = False
MIN_CONFLUENCE_SCORE = 0.3  # Minimum confluence score to accept signals
CONFLUENCE_TIMEFRAMES = ['5m', '15m', '1h']  # Timeframes to analyze
CONFLUENCE_WEIGHT_IN_CONFIDENCE = 0.2  # How much confluence affects final confidence

# Individual timeframe weights for confluence calculation
TIMEFRAME_WEIGHTS = {
    '5m': 0.2,   # Lower weight for noise-prone shorter timeframe
    '15m': 0.4,  # Medium weight for your current default
    '1h': 0.4    # Higher weight for more stable longer timeframe
}

# Alert Deduplication Configuration
ALERT_COOLDOWN_MINUTES = 5       # Minutes between same epic+signal alerts (reduced from 15 - hash check disabled)
STRATEGY_COOLDOWN_MINUTES = 3    # Strategy-specific cooldown (reduced from 10)
GLOBAL_COOLDOWN_SECONDS = 30            # Global cooldown between any alerts
MAX_ALERTS_PER_HOUR = 50        # Global hourly alert limit
MAX_ALERTS_PER_EPIC_HOUR = 6    # Per-epic hourly alert limit
ENABLE_ALERT_DEDUPLICATION = True  # Master switch for deduplication

# Trade Cooldown Configuration (prevents signals for epics with recent trades)
# This check queries trade_log table to prevent wasting Claude API calls
TRADE_COOLDOWN_ENABLED = True    # Enable trade cooldown check before signal processing
TRADE_COOLDOWN_MINUTES = 30      # Cooldown after trade open/close (must match dev-app)

# Rate limiting settings
MAX_ALERTS_PER_HOUR = 50                # Global hourly alert limit
MAX_ALERTS_PER_EPIC_HOUR = 6            # Per-epic hourly alert limit

# Similarity detection settings
PRICE_SIMILARITY_THRESHOLD = 0.0002     # Price similarity threshold (2 pips for most pairs)
CONFIDENCE_SIMILARITY_THRESHOLD = 0.05  # 5% confidence similarity threshold

# Cache and performance settings
SIGNAL_HASH_CACHE_SIZE = 1000           # How many signal hashes to keep in memory
SIGNAL_HASH_CACHE_EXPIRY_MINUTES = 15   # Minutes before cache entries expire (matches database check)
MAX_SIGNAL_HASH_CACHE_SIZE = 1000       # Max cache size before forced cleanup
ENABLE_SIGNAL_HASH_CHECK = False        # Master switch - Completely disable hash duplicate check (cooldown layer sufficient)
ENABLE_TIME_BASED_HASH_COMPONENTS = False # Disabled - Hash check too strict, blocking valid signals. Rely on cooldown layer instead.
DEDUPLICATION_DEBUG_MODE = False        # Enable verbose deduplication logging
DEDUPLICATION_CLEANUP_INTERVAL = 100    # Clean cache every N checks

# Enhanced deduplication features
ENABLE_PRICE_SIMILARITY_CHECK = True    # Check for similar price signals
ENABLE_STRATEGY_COOLDOWNS = True        # Enable strategy-specific cooldowns
ENABLE_ENHANCED_HASH_DETECTION = True   # Use enhanced hash generation

# Advanced deduplication settings
DEDUPLICATION_LOOKBACK_HOURS = 2        # How far back to check for duplicates
DEDUPLICATION_CACHE_CLEANUP_HOURS = 4   # When to clean up old cache entries

# Database deduplication settings
USE_DATABASE_DEDUP_CHECK = True         # Check database for recent duplicates
DATABASE_DEDUP_WINDOW_MINUTES = 15      # Database check window

# Signal hash configuration
INCLUDE_INDICATORS_IN_HASH = True       # Include technical indicators in hash
PRICE_DECIMAL_PRECISION = 5             # Decimal places for price hashing
CONFIDENCE_DECIMAL_PRECISION = 4        # Decimal places for confidence hashing

# ========================================================================
# DEDUPLICATION PRESETS
# ========================================================================
# Pre-configured deduplication settings for different use cases

# Strict deduplication - fewer signals, less noise
DEDUPLICATION_STRICT = {
    'ALERT_COOLDOWN_MINUTES': 10,
    'STRATEGY_COOLDOWN_MINUTES': 5,
    'MAX_ALERTS_PER_HOUR': 30,
    'MAX_ALERTS_PER_EPIC_HOUR': 3,
    'PRICE_SIMILARITY_THRESHOLD': 0.0005,  # 5 pips
    'CONFIDENCE_SIMILARITY_THRESHOLD': 0.02  # 2%
}

# Standard deduplication - balanced filtering (DEFAULT)
DEDUPLICATION_STANDARD = {
    'ALERT_COOLDOWN_MINUTES': 5,
    'STRATEGY_COOLDOWN_MINUTES': 3,
    'MAX_ALERTS_PER_HOUR': 50,
    'MAX_ALERTS_PER_EPIC_HOUR': 6,
    'PRICE_SIMILARITY_THRESHOLD': 0.0002,  # 2 pips
    'CONFIDENCE_SIMILARITY_THRESHOLD': 0.05  # 5%
}

# Relaxed deduplication - more signals, some duplicates allowed
DEDUPLICATION_RELAXED = {
    'ALERT_COOLDOWN_MINUTES': 2,
    'STRATEGY_COOLDOWN_MINUTES': 1,
    'MAX_ALERTS_PER_HOUR': 100,
    'MAX_ALERTS_PER_EPIC_HOUR': 12,
    'PRICE_SIMILARITY_THRESHOLD': 0.0001,  # 1 pip
    'CONFIDENCE_SIMILARITY_THRESHOLD': 0.1  # 10%
}

# Set the active deduplication preset
DEDUPLICATION_PRESET = 'standard'  # Options: 'strict', 'standard', 'relaxed'

# Claude Analysis Settings (legacy - see CLAUDE TRADE VALIDATION section for main settings)
CLAUDE_MIN_SCORE_THRESHOLD = 6
# MIN_CLAUDE_QUALITY_SCORE moved to CLAUDE TRADE VALIDATION section

# Add these configuration settings to your config.py file

# ===== ENHANCED STRATEGY SYSTEM CONFIGURATION =====

# Enhanced Signal Detection Configuration
#SIGNAL_DETECTION_MODE = 'ensamble'  # 'single_best', 'combined', 'ensemble', 'adaptive'

# Ensemble Strategy Configuration
#ENSEMBLE_AGGREGATION_METHOD = 'weighted_mean'  # 'weighted_mean', 'majority_vote', 'confidence_weighted', 'adaptive'
#ENSEMBLE_MIN_STRATEGIES = 2
#ENSEMBLE_CONSENSUS_THRESHOLD = 0.6
#ENSEMBLE_MIN_CONFIDENCE = 0.65

# Ensemble Strategy Weights (all strategies included)
#ENSEMBLE_WEIGHT_EMA = 0.25
#ENSEMBLE_WEIGHT_SCALPING = 0.2
#ENSEMBLE_WEIGHT_KAMA = 0.15
#ENSEMBLE_WEIGHT_BB_SUPERTREND = 0.15

# KAMA Strategy Configuration moved to configdata/strategies/config_kama_strategy.py








# ===== BB SUPERTREND STRATEGY CONFIGURATION =====
BOLLINGER_SUPERTREND_STRATEGY = False

# BB SuperTrend Strategy Configuration moved to configdata/strategies/config_bb_supertrend_strategy.py



# Market Strategy Selector Configuration
MARKET_STRATEGY_HISTORICAL_WEIGHT = 0.25
MARKET_STRATEGY_PERFORMANCE_DECAY = 0.1

# Adaptive Detection Thresholds
ADAPTIVE_HIGH_VOLATILITY_THRESHOLD = 1.5
ADAPTIVE_LOW_VOLATILITY_THRESHOLD = 0.5
ADAPTIVE_TREND_STRENGTH_THRESHOLD = 0.7

MIN_MARKET_EFFICIENCY = 0.02  # Was 0.1 - much more permissive  

EMA_STRICT_ALIGNMENT = False  # Don't require perfect alignment
EMERGENCY_DEBUG_MODE = True  # Enable emergency signal generation
BACKTEST_MODE_RELAXED = True  # Boost signals in backtesting

# =============================================================================
# EMA 200 TREND FILTER SETTINGS
# =============================================================================

# Always ensure EMA 200 is calculated for trend filtering (regardless of dynamic config)
ALWAYS_INCLUDE_EMA200 = True

# Enable EMA 200 trend filter in TradeValidator
ENABLE_EMA200_TREND_FILTER = True

# Explanation:
# - ema_trend: Dynamic period based on strategy config (50, 200, 21, etc.)
# - ema_200: Always EMA 200 period for consistent trend filtering
# - This separation allows strategies to use optimal dynamic periods
#   while maintaining consistent trend-based filtering


# ADD these settings to config.py

# Market Closure and Timestamp Validation Settings
# These settings control how the system handles market closure and timestamp issues

# Market Closure Behavior
MARKET_CLOSURE_SETTINGS = {
    'save_signals_when_closed': True,     # Save signals even when market is closed
    'execute_signals_when_closed': False, # Don't execute trades when market is closed
    'log_market_status': True,           # Log market open/close status
    'queue_signals_for_open': True,      # Queue signals for execution when market opens
}

# Timestamp Validation Settings
TIMESTAMP_VALIDATION_SETTINGS = {
    'fix_epoch_timestamps': True,        # Convert 1970 timestamps to None
    'log_timestamp_issues': True,        # Log but don't reject on timestamp issues
    'allow_none_timestamps': True,       # Allow signals with None timestamps
    'validate_against_current_time': True, # Use current time for market validation
    'reject_future_timestamps': False,   # Allow future timestamps (clock skew)
}

# Forex Market Hours (UTC)
FOREX_MARKET_HOURS = {
    'open_day': 0,        # Monday (0=Monday, 6=Sunday)
    'open_hour': 22,      # 22:00 UTC Sunday (actually Monday in some timezones)
    'close_day': 4,       # Friday
    'close_hour': 22,     # 22:00 UTC Friday
    'timezone': 'UTC',    # Reference timezone
}

# Alert Processing Configuration
ALERT_PROCESSING = {
    'process_during_closure': True,       # Process and save alerts during market closure
    'mark_closure_status': True,         # Mark alerts with market status
    'enhanced_logging': True,            # Enhanced logging for debugging
    'separate_closure_queue': False,     # Don't use separate queue (save to main table)
}

# Backward Compatibility
# These ensure existing code continues to work
SAVE_ALERTS_WHEN_MARKET_CLOSED = MARKET_CLOSURE_SETTINGS['save_signals_when_closed']
LOG_TIMESTAMP_CONVERSIONS = TIMESTAMP_VALIDATION_SETTINGS['log_timestamp_issues']
VALIDATE_MARKET_HOURS = True  # Keep existing behavior but improve implementation

# Add this function to config.py for easy access
def is_market_open_now() -> bool:
    """
    Check if forex market is currently open based on UTC time
    
    Returns:
        bool: True if market is open, False if closed
    """
    from datetime import datetime, timezone
    
    current_utc = datetime.now(timezone.utc)
    weekday = current_utc.weekday()  # 0=Monday, 6=Sunday
    hour = current_utc.hour
    
    # Market is closed from Friday 22:00 UTC to Sunday 22:00 UTC
    if weekday == 5:  # Saturday
        return False
    elif weekday == 6:  # Sunday
        return hour >= 22  # Open after 22:00 UTC
    elif weekday == 4:  # Friday
        return hour < 22   # Closed after 22:00 UTC
    
    # Market is open Monday-Thursday and Friday before 22:00, Sunday after 22:00
    return True

def get_market_status_info() -> dict:
    """
    Get detailed market status information
    
    Returns:
        dict: Market status details
    """
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


# Support/Resistance Validation
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

# Enhanced S/R Validation with Level Flip Detection
ENABLE_ENHANCED_SR_VALIDATION = True  # Use enhanced validator with SMC integration
SR_RECENT_FLIP_BARS = 50             # Consider flips within last 50 bars as "recent"
SR_MIN_FLIP_STRENGTH = 0.6           # Minimum strength to consider a level flip significant




# =============================================================================
# DEPRECATED: LEGACY ENHANCED CLAUDE CONFIGURATION - REMOVED
# =============================================================================
# This entire section (~240 lines) has been removed as it was experimental/unused.
# Claude trade validation is now controlled via database settings.
# See: Streamlit > Settings > Scanner Config > Claude AI
#
# Key active settings now in database:
#   - require_claude_approval: Master switch for Claude validation
#   - claude_fail_secure: Fail-secure mode (block on errors)
#   - claude_model: Model selection (haiku/sonnet/opus)
#   - min_claude_quality_score: Minimum approval score (1-10)
#   - claude_vision_enabled: Enable chart-based analysis
#   - claude_chart_timeframes: Timeframes for chart generation
#   - claude_vision_strategies: Strategies that use vision
#
# To access programmatically:
#   from forex_scanner.services.scanner_config_service import get_scanner_config
#   config = get_scanner_config()
#   if config.require_claude_approval:
#       # Use Claude validation
# =============================================================================

# =============================================================================
# ENHANCED LOGGING CONFIGURATION
# =============================================================================

# Logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'INFO'

# Log directory (relative to project root)
LOG_DIR = 'logs'

# Log file retention (days)
LOG_RETENTION_DAYS = 30

# Enable separate log files for different components
ENABLE_SIGNAL_LOGGING = True      # Separate file for signal-related logs
ENABLE_ERROR_LOGGING = True       # Separate file for errors only
ENABLE_PERFORMANCE_LOGGING = True # Separate file for performance metrics
ENABLE_DEBUG_LOGGING = True       # Detailed debug file

# Log file size limits
MAX_LOG_FILE_SIZE = 50 * 1024 * 1024  # 50MB for debug logs
MAX_DEBUG_FILES = 3                    # Keep 3 rotating debug files

# Console logging (what appears in Docker logs)
CONSOLE_LOG_LEVEL = 'INFO'

# Detailed logging format (includes function names and line numbers)
DETAILED_LOGGING = True

# Log filters - keywords that will be captured in specific log files
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

# Timezone for log timestamps (useful for international deployment)
LOG_TIMEZONE = 'Europe/Stockholm'

# Enable log file compression for old files (saves disk space)
COMPRESS_OLD_LOGS = True

# Smart Money Configuration moved to configdata/strategies/config_smc_strategy.py

ZERO_LAG_STRATEGY = False
# MOMENTUM_BIAS_STRATEGY removed - legacy strategy replaced by MOMENTUM_STRATEGY
MOMENTUM_STRATEGY = False

ZERO_LAG_STRATEGY_ENABLED = False
# MOMENTUM_BIAS_STRATEGY_ENABLED removed - legacy strategy replaced by MOMENTUM_STRATEGY
MOMENTUM_STRATEGY_ENABLED = False
USE_ZERO_LAG_STRATEGY = False
# USE_MOMENTUM_BIAS_STRATEGY removed - legacy strategy replaced by MOMENTUM_STRATEGY
USE_MOMENTUM_STRATEGY = False


# Strategy indicator requirements mapping
STRATEGY_INDICATOR_MAP = {
    'ema': ['ema'],
    'macd': ['macd'],
    'kama': ['kama'],
    'bb_supertrend': ['bb_supertrend'],
    'momentum_bias': ['momentum_bias'],      # NEW
    'momentum': ['momentum'],               # NEW
    'zero_lag': ['zero_lag_ema'],           # NEW
    'volume': ['volume'],
    'support_resistance': ['support_resistance'],
    'behavior': ['behavior']
}

# Required Indicators for Data Fetcher (new section)
#REQUIRED_INDICATORS_BY_STRATEGY = {
#    'ema': ['ema', 'close', 'high', 'low'],
#    'macd': ['macd', 'ema', 'close'],
#    'kama': ['kama', 'close', 'high', 'low'],
#    'momentum_bias': ['momentum_bias', 'close', 'high', 'low'],
#    'zero_lag_ema': ['zero_lag', 'close', 'high', 'low'],
#    'bb_supertrend': ['bb_supertrend', 'close', 'high', 'low', 'volume'],
#    'combined': ['ema', 'macd', 'kama', 'momentum_bias', 'zero_lag', 'bb_supertrend']
#}

REQUIRED_INDICATORS_BY_STRATEGY = {
    'ema': ['ema', 'close', 'high', 'low'],
    'macd': ['macd', 'ema', 'close'],
    'kama': ['kama', 'close', 'high', 'low'] if KAMA_STRATEGY else [],
    # 'momentum_bias' removed - legacy strategy replaced by momentum
    'momentum': ['momentum', 'close', 'high', 'low', 'volume'] if MOMENTUM_STRATEGY else [],
    'zero_lag_ema': ['zero_lag', 'close', 'high', 'low'] if ZERO_LAG_STRATEGY else [],
    'bb_supertrend': ['bb_supertrend', 'close', 'high', 'low', 'volume'] if BOLLINGER_SUPERTREND_STRATEGY else [],
    # CRITICAL FIX: Only include enabled strategies in combined
    'combined': (['ema', 'macd'] +
                (['kama'] if KAMA_STRATEGY else []) +
                # momentum_bias removed - legacy strategy replaced by momentum
                (['momentum'] if MOMENTUM_STRATEGY else []) +
                (['zero_lag'] if ZERO_LAG_STRATEGY else []) +
                (['bb_supertrend'] if BOLLINGER_SUPERTREND_STRATEGY else []))
}

# Strategy Configuration Files (new section)
STRATEGY_CONFIG_MODULES = {
    'momentum_bias': 'configdata.config_momentum_bias',
    'momentum': 'configdata.strategies.config_momentum_strategy',
    'zero_lag_ema': 'configdata.config_zerolag_strategy'
}

# =============================================================================
# DEPRECATED: CRITICAL SAFETY FILTERS - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > Safety Filters
# -----------------------------------------------------------------------------
# ENABLE_CRITICAL_SAFETY_FILTERS = True  # MIGRATED to database
# ENABLE_EMA200_CONTRADICTION_FILTER = True  # MIGRATED to database
# ENABLE_EMA_STACK_CONTRADICTION_FILTER = True  # MIGRATED to database
# REQUIRE_INDICATOR_CONSENSUS = True  # MIGRATED to database
# MIN_CONFIRMING_INDICATORS = 1  # MIGRATED to database
# ENABLE_EMERGENCY_CIRCUIT_BREAKER = True  # MIGRATED to database
# MAX_CONTRADICTIONS_ALLOWED = 5  # MIGRATED to database
# EMA200_MINIMUM_MARGIN = 0.002  # MIGRATED to database
# SAFETY_FILTER_LOG_LEVEL = 'ERROR'  # MIGRATED to database
# SAFETY_FILTER_PRESETS = {...}  # MIGRATED to database as JSONB
# =============================================================================

# Safety Filter Logging (NOT migrated - logging infrastructure)
LOG_REJECTION_DETAILS = True  # Log detailed rejection information

# Override for emergency situations (set to False in emergencies)
EMERGENCY_BYPASS_SAFETY_FILTERS = False

# Statistics tracking (runtime state - not config)
TRACK_SAFETY_FILTER_STATS = True
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

# =============================================================================
# LEGACY: SAFETY FILTER PRESETS - NOW IN DATABASE
# =============================================================================
# These presets have been migrated to strategy_config.scanner_global_config
# as the safety_filter_presets JSONB column. DO NOT modify here.
# =============================================================================

# LEGACY: Kept for backward compatibility during transition - DO NOT USE
SAFETY_FILTER_PRESETS_LEGACY = {
    'strict': {
        'ENABLE_EMA200_CONTRADICTION_FILTER': True,
        'ENABLE_EMA_STACK_CONTRADICTION_FILTER': True,
        'REQUIRE_INDICATOR_CONSENSUS': True,
        'MAX_CONTRADICTIONS_ALLOWED': 0,  # No contradictions allowed
        'EMA200_MINIMUM_MARGIN': 0.001,   # 0.1% - Very strict
    },
    
    'balanced': {
        'ENABLE_EMA200_CONTRADICTION_FILTER': True,
        'ENABLE_EMA_STACK_CONTRADICTION_FILTER': True,
        'REQUIRE_INDICATOR_CONSENSUS': True,
        'MAX_CONTRADICTIONS_ALLOWED': 5,  # 1 contradiction allowed
        'EMA200_MINIMUM_MARGIN': 0.002,   # 0.2% - Balanced
    },
    
    'permissive': {
        'ENABLE_EMA200_CONTRADICTION_FILTER': True,
        'ENABLE_MACD_CONTRADICTION_FILTER': False,  # Disabled
        'ENABLE_EMA_STACK_CONTRADICTION_FILTER': False,  # Disabled
        'REQUIRE_INDICATOR_CONSENSUS': False,
        'MAX_CONTRADICTIONS_ALLOWED': 2,  # 2 contradictions allowed
        'EMA200_MINIMUM_MARGIN': 0.005,   # 0.5% - Very permissive
    },
    
    'emergency': {
        # Only the most critical filters enabled
        'ENABLE_EMA200_CONTRADICTION_FILTER': True,
        'ENABLE_EMA_STACK_CONTRADICTION_FILTER': False, 
        'REQUIRE_INDICATOR_CONSENSUS': False,
        'MAX_CONTRADICTIONS_ALLOWED': 3,  # Very permissive
        'EMA200_MINIMUM_MARGIN': 0.01,    # 1% - Emergency only
    }
}

# Active preset (change this to switch safety levels)
ACTIVE_SAFETY_PRESET = 'balanced'  # Options: 'strict', 'balanced', 'permissive', 'emergency'

# Helper function to apply safety preset
def apply_safety_preset(preset_name: str):
    """Apply a safety filter preset configuration"""
    if preset_name in SAFETY_FILTER_PRESETS:
        preset = SAFETY_FILTER_PRESETS[preset_name]
        globals().update(preset)
        print(f"✅ Applied safety preset: {preset_name}")
        
        # Log the configuration
        print(f"   EMA200 Filter: {'ON' if preset.get('ENABLE_EMA200_CONTRADICTION_FILTER') else 'OFF'}")
        print(f"   EMA Stack Filter: {'ON' if preset.get('ENABLE_EMA_STACK_CONTRADICTION_FILTER') else 'OFF'}")
        print(f"   Max Contradictions: {preset.get('MAX_CONTRADICTIONS_ALLOWED', 'N/A')}")
        
        return True
    else:
        print(f"❌ Unknown safety preset: {preset_name}")
        print(f"   Available presets: {', '.join(SAFETY_FILTER_PRESETS.keys())}")
        return False

# Apply the active preset on module load
try:
    apply_safety_preset(ACTIVE_SAFETY_PRESET)
except Exception as e:
    print(f"Warning: Failed to apply safety preset {ACTIVE_SAFETY_PRESET}: {e}")

# =============================================================================
# DEBUGGING AND MONITORING FUNCTIONS
# =============================================================================

def get_safety_filter_stats():
    """Get current safety filter statistics"""
    return SAFETY_FILTER_STATS.copy()

def reset_safety_filter_stats():
    """Reset safety filter statistics"""
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
    print("✅ Safety filter statistics reset")

# =============================================================================
# LARGE CANDLE MOVEMENT FILTER SETTINGS
# =============================================================================

# Enable/disable large candle filtering
ENABLE_LARGE_CANDLE_FILTER = True

# Large candle detection - candles larger than this multiple of ATR are considered "large"
LARGE_CANDLE_ATR_MULTIPLIER = 2.5  # 2.5x ATR = large candle

# Consecutive large candles threshold  
CONSECUTIVE_LARGE_CANDLES_THRESHOLD = 2  # Block if 2+ large candles recently

# Movement analysis periods
MOVEMENT_LOOKBACK_PERIODS = 3  # Check last 5 candles

# Excessive movement threshold in pips
EXCESSIVE_MOVEMENT_THRESHOLD_PIPS = 15  # 15+ pips in lookback period

# Cooldown after large candle (periods to wait)
LARGE_CANDLE_FILTER_COOLDOWN = 3  # Wait 3 periods after large candle

# Parabolic movement sensitivity
PARABOLIC_ACCELERATION_THRESHOLD = 1.5  # 50% acceleration in movement

# Filter strictness presets
LARGE_CANDLE_FILTER_PRESET = 'balanced'  # 'strict', 'balanced', 'permissive'

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
# DEPRECATED: CLAUDE TRADE VALIDATION - NOW IN DATABASE
# =============================================================================
# These settings have been migrated to strategy_config.scanner_global_config
# See: Streamlit > Settings > Scanner Config > Claude AI
#
# To access settings programmatically:
#   from forex_scanner.services.scanner_config_service import get_scanner_config
#   config = get_scanner_config()
#   if config.require_claude_approval:
#       ...
# -----------------------------------------------------------------------------
# REQUIRE_CLAUDE_APPROVAL = True  # MIGRATED to database
# CLAUDE_FAIL_SECURE = True  # MIGRATED to database
# CLAUDE_MODEL = 'sonnet'  # MIGRATED to database
# MIN_CLAUDE_QUALITY_SCORE = 6  # MIGRATED to database
# CLAUDE_INCLUDE_CHART = True  # MIGRATED to database
# CLAUDE_CHART_TIMEFRAMES = ['4h', '1h', '15m']  # MIGRATED to database (JSONB)
# CLAUDE_VISION_ENABLED = True  # MIGRATED to database
# CLAUDE_VISION_STRATEGIES = ['EMA_DOUBLE', 'SMC', 'SMC_STRUCTURE']  # MIGRATED (JSONB)
# CLAUDE_SAVE_VISION_ARTIFACTS = True  # MIGRATED to database
# CLAUDE_VISION_SAVE_DIRECTORY = 'claude_analysis_enhanced/vision_analysis'  # MIGRATED
# SAVE_CLAUDE_REJECTIONS = True  # MIGRATED to database
# CLAUDE_VALIDATE_IN_BACKTEST = False  # MIGRATED to database
# =============================================================================

# Rate limiting (kept as constants - not tunable)
CLAUDE_MAX_REQUESTS_PER_MINUTE = 50
CLAUDE_MAX_REQUESTS_PER_DAY = 1000
CLAUDE_MIN_CALL_INTERVAL = 1.2          # Seconds between API calls

# Multi-Timeframe Analysis Configuration
ENABLE_MTF_ANALYSIS = True  # Enable/disable MTF analysis

MTF_CONFIG = {
    'require_alignment': True,      # Require timeframe alignment
    'min_aligned_timeframes': 2,    # Minimum aligned TFs
    'check_timeframes': ['5m', '15m', '1h'],  # TFs to analyze
    'alignment_threshold': 0.6,     # Minimum alignment score
    'confidence_boost_max': 0.15    # Max confidence boost for alignment
}

# Timeframe hierarchy for trend validation
TIMEFRAME_HIERARCHY = {
    '1m': ['5m', '15m'],    # 1m signals check 5m and 15m
    '5m': ['15m', '1h'],    # 5m signals check 15m and 1h
    '15m': ['1h', '4h'],    # 15m signals check 1h and 4h
    '30m': ['1h', '4h'],    # 30m signals check 1h and 4h
    '1h': ['4h', '1d']      # 1h signals check 4h and daily
}

# Retry settings
ORDER_MAX_RETRIES = 3
ORDER_RETRY_BASE_DELAY = 2.0
ORDER_CONNECT_TIMEOUT = 10.0
ORDER_READ_TIMEOUT = 45.0

# Circuit breaker
ORDER_CIRCUIT_BREAKER_THRESHOLD = 5
ORDER_CIRCUIT_BREAKER_RECOVERY = 300.0


# =============================================================================
# ADDITIONAL CONFIGURATION SETTINGS  
# =============================================================================

import logging

# Zero Lag MTF Validation Settings moved to configdata/strategies/config_zerolag_strategy.py

