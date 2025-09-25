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

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/forex")

# API Keys
# CLAUDE_API_KEY is provided via environment variable or Azure Key Vault
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', None)  # None if not available

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

# Scanner Settings
SCAN_INTERVAL = 120  # seconds between scans
SPREAD_PIPS = 1.5   # default spread for BID/ASK adjustment

USE_BID_ADJUSTMENT = False  # whether to adjust BID prices to MID
DEFAULT_TIMEFRAME = '15m'  # Default timeframe for signals ('5m', '15m', '1h')
# Signal confidence threshold
MIN_CONFIDENCE = 0.45  # 45% confidence (relaxed for more signals)
#=============================================================================
# DATA FETCHER OPTIMIZATION CONFIGURATION
# =============================================================================

# Enable optimizations
ENABLE_DATA_CACHE = False          # 5-minute data caching (DISABLED for fresh data)
REDUCED_LOOKBACK_HOURS = True      # Use smart lookback times
LAZY_INDICATOR_LOADING = True      # Load indicators on demand
DATA_BATCH_SIZE = 2000            # Limit query results

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
# DUPLICATE DETECTION CONFIGURATION
# =================================================================

# Enable/disable duplicate detection entirely
ENABLE_DUPLICATE_CHECK = True  # Set to False to disable duplicate checking

# Duplicate detection sensitivity
# 'strict' - Very precise matching (price + confidence + epic + signal_type + strategy)
# 'smart' - Balanced matching (epic + signal_type + strategy + rounded confidence) 
# 'loose' - Basic matching (epic + signal_type only)
DUPLICATE_SENSITIVITY = 'smart'

# Signal cooldown period in minutes
SIGNAL_COOLDOWN_MINUTES = 15

# Minimum confidence for orders (separate from signal detection)
MIN_CONFIDENCE_FOR_ORDERS = MIN_CONFIDENCE

# =============================================================================
# EMA200 DISTANCE VALIDATION
# =============================================================================



# =============================================================================
# ADX TREND STRENGTH FILTER (PHASE 2)
# =============================================================================

# Enable/disable ADX trend strength filtering
ADX_FILTER_ENABLED = False

# ADX threshold levels for trend strength classification
ADX_THRESHOLDS = {
    'STRONG_TREND': 25.0,      # ADX > 25 = Strong trend (allow signals)
    'MODERATE_TREND': 22.0,    # ADX 20-25 = Moderate trend (conditional)
    'WEAK_TREND': 15.0,        # ADX < 20 = Weak/ranging market (filter out)
    'VERY_WEAK': 10.0          # ADX < 15 = Very weak trend (definitely filter)
}

# ADX filter behavior modes (maps to STRONG_TREND, MODERATE_TREND, WEAK_TREND thresholds above)
# 'strict' - Only allow signals during STRONG trends (ADX > STRONG_TREND = 25)
# 'moderate' - Allow signals during MODERATE+ trends (ADX > MODERATE_TREND = 20) 
# 'permissive' - Allow signals during WEAK+ trends (ADX > WEAK_TREND = 15)
# 'disabled' - ADX calculated but not used for filtering
ADX_FILTER_MODE = 'moderate'

# ADX calculation period (standard is 14)
ADX_PERIOD = 14

# Pair-specific ADX adjustments (some pairs trend differently)
ADX_PAIR_MULTIPLIERS = {
    'EURUSD': 1.0,     # EUR/USD - standard ADX behavior
    'GBPUSD': 0.9,     # GBP/USD - slightly more volatile, lower threshold
    'USDJPY': 1.1,     # USD/JPY - requires stronger trends
    'EURJPY': 0.85,    # EUR/JPY - cross pairs trend more aggressively
    'GBPJPY': 0.8,     # GBP/JPY - very volatile cross
    'USDCHF': 1.2,     # USD/CHF - safe haven, requires clearer trends
    'DEFAULT': 1.0     # Default multiplier for other pairs
}

# Grace period: Allow signals during temporary ADX dips if recent trend was strong
ADX_GRACE_PERIOD_BARS = 2  # Allow 2 bars of weak ADX if previous trend was strong

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
EMA_STRATEGY = True  # Core EMA strategy

# Enable MACD Strategy
MACD_STRATEGY = True  # Core MACD strategy

# Enable KAMA Strategy
KAMA_STRATEGY = False  # This is the key setting that's missing

# Enable SMC Strategy
SMC_STRATEGY = True  # Smart Money Concepts strategy

# Enable Ichimoku Cloud Strategy
ICHIMOKU_CLOUD_STRATEGY = True  # Ichimoku Kinko Hyo strategy

# Mean Reversion Strategy
MEAN_REVERSION_STRATEGY = True  # Multi-oscillator mean reversion strategy

# Strategy Configurations - Additional strategies
RANGING_MARKET_STRATEGY = True  # Multi-oscillator ranging market strategy

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

# Alert Settings
ENABLE_CLAUDE_ANALYSIS = False
CLAUDE_ANALYSIS_ENABLED = False
USE_CLAUDE_ANALYSIS = False
CLAUDE_ANALYSIS_MODE = "disabled"  # or "full", 'disabled'

ENABLE_ORDER_EXECUTION = True  # Set to True when ready for live trading
MAX_SIGNALS_PER_HOUR = 10  # Rate limiting

# Risk Management
POSITION_SIZE_PERCENT = 1.0  # % of account per trade
STOP_LOSS_PIPS = 5
TAKE_PROFIT_PIPS = 15
MAX_OPEN_POSITIONS = 3

# Timezone Settings
USER_TIMEZONE = 'Europe/Stockholm'  # Your local timezone
DATABASE_TIMEZONE = 'UTC'           # Database timezone (IG data is in UTC)
MARKET_OPEN_HOUR_LOCAL = 8          # Local time
MARKET_CLOSE_HOUR_LOCAL = 22        # Local time

# Trading Schedule (24/5 forex hours)
TRADING_HOURS = {
    'start_hour': 0,    # 24/7 (midnight)
    'end_hour': 23,     # 24/7 (11 PM) 
    'enabled_days': [0, 1, 2, 3, 4, 6],  # Monday-Friday + Sunday evening
    'enable_24_5': True  # True = 24/5 forex hours, False = use start/end hours
}

# Market Hours Settings
RESPECT_MARKET_HOURS = False  # Set to False for 24/5 scanning
WEEKEND_SCANNING = False      # Set to True to scan weekends too

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
# Position sizing
DEFAULT_POSITION_SIZE = 0.1  # Fixed position size (optional)
ACCOUNT_BALANCE = 10000      # Account balance for risk calculation
RISK_PER_TRADE = 0.02        # 2% risk per trade
MAX_POSITION_SIZE = 1.0      # Maximum position size

# Order parameters
DEFAULT_STOP_DISTANCE = 20   # Stop loss distance in pips
DEFAULT_RISK_REWARD = 2.0    # Risk:reward ratio for take profit
MIN_CONFIDENCE_FOR_ORDERS = 0.70  # Minimum confidence to execute orders

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
    "CS.D.EURUSD.CEEM.IP": "No trading permissions for FX_NOR exchange",
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
MIN_CONFIDENCE_FOR_ORDERS = 0.70  # Only trade signals above 75% confidence
MAX_DAILY_TRADES = 10              # Maximum trades per day
MAX_CONCURRENT_POSITIONS = 3       # Maximum open positions

# Order labeling
ORDER_LABEL_PREFIX = "ForexScanner"  # Prefix for order labels

# =============================================================================
# POSITION SIZING & RISK MANAGEMENT
# =============================================================================

# POSITION SIZING METHOD (choose one)
# ------------------------------------

# Method 1: Fixed Position Size (simplest)
FIXED_POSITION_SIZE = None  # Set to 0.1 for fixed 0.1 lots, or None to disable
DEFAULT_POSITION_SIZE = 0.1  # Fallback if other methods fail

# Method 2: Risk-Based Position Sizing (RECOMMENDED)
RISK_PER_TRADE_PERCENT = 0.02  # Risk 2% of account per trade
ACCOUNT_BALANCE = 10000        # Your account balance in account currency
PIP_VALUE = 1.0               # Value per pip (depends on broker/pair)

# Method 3: Confidence-Based Position Sizing
CONFIDENCE_BASED_SIZING = True  # Enable confidence-based sizing
BASE_POSITION_SIZE = 0.1       # Base size, adjusted by confidence

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
MIN_CONFIDENCE_FOR_ORDERS = 0.70  # Only trade signals above 75% confidence

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
MIN_CONFIDENCE_FOR_TRADING = MIN_CONFIDENCE  # Higher threshold for actual trades
REQUIRE_CLAUDE_APPROVAL = False     # Require Claude to approve trades
MIN_CLAUDE_QUALITY_SCORE = 6       # Minimum Claude score (1-10)
CLAUDE_MIN_CONFIDENCE_THRESHOLD = 0.8  # 80%+ confidence only
MAX_SIGNALS_PER_DAY = 20           # Daily signal limit
SIGNAL_COOLDOWN_MINUTES = 15       # Cooldown between signals per pair
MIN_VOLUME_RATIO = 1.2             # Minimum volume confirmation
CLAUDE_ANALYSIS_MODE = 'strategic_minimal'
CLAUDE_STRATEGIC_FOCUS = 'learning'

# Trading Hours
RESPECT_TRADING_HOURS = False
TRADING_START_HOUR = 0             # Local time
TRADING_END_HOUR = 23              # Local time


# Claude Analysis
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
ALERT_COOLDOWN_MINUTES = 15      # Minutes between same epic+signal alerts
STRATEGY_COOLDOWN_MINUTES = 10
GLOBAL_COOLDOWN_SECONDS = 30            # Global cooldown between any alerts 
MAX_ALERTS_PER_HOUR = 50        # Global hourly alert limit
MAX_ALERTS_PER_EPIC_HOUR = 6    # Per-epic hourly alert limit
ENABLE_ALERT_DEDUPLICATION = True  # Master switch for deduplication

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
ENABLE_TIME_BASED_HASH_COMPONENTS = True # Include time buckets in hash generation
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

# Claude Analysis Settings
CLAUDE_MIN_SCORE_THRESHOLD = 6
MIN_CLAUDE_QUALITY_SCORE = 5

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
# ENHANCED CLAUDE API CONFIGURATION ADDITIONS
# Add these to your existing config.py file
# =============================================================================

# Enhanced Claude Analysis Configuration
CLAUDE_ENHANCED_MODE = True                    # Enable enhanced parsing and validation
CLAUDE_TECHNICAL_VALIDATION = True             # Enable pre-validation before Claude calls
CLAUDE_COMPLETE_DATAFRAME_ANALYSIS = True     # Use complete DataFrame indicators

# Enhanced Response Parsing Settings
CLAUDE_PARSE_FALLBACK_ENABLED = True          # Enable fallback parsing for varied responses
CLAUDE_NATURAL_LANGUAGE_PARSING = True        # Parse natural language responses
CLAUDE_STRUCTURED_FORMAT_PRIORITY = True      # Try structured format first

# Technical Validation Thresholds (Enhanced Detection)
CLAUDE_EMA_VALIDATION_ENABLED = True          # Enable EMA alignment validation
CLAUDE_KAMA_EFFICIENCY_MIN = 0.3               # Minimum KAMA efficiency ratio

# Signal Classification Verification
CLAUDE_SIGNAL_CLASSIFICATION_CHECK = True     # Allow Claude to override system classification
CLAUDE_CONFLICT_LOGGING = True                # Log classification conflicts
CLAUDE_AGREEMENT_THRESHOLD = 0.8              # Threshold for system-Claude agreement

# Complete DataFrame Analysis Settings
CLAUDE_USE_ALL_INDICATORS = True              # Include all available technical indicators
CLAUDE_CROSS_VALIDATION_ENABLED = True       # Enable cross-indicator validation
CLAUDE_INDICATOR_WEIGHTS = {                  # Weights for different indicator types
    'EMA': 0.3,                               # Medium weight for EMA (trend direction)
    'KAMA': 0.2,                              # Lower weight for KAMA (efficiency)
    'OTHER': 0.1                              # Minimal weight for other indicators
}

# Enhanced Prompt Configuration
CLAUDE_DETAILED_PROMPTS = True                # Use detailed technical prompts
CLAUDE_MARKET_CONTEXT_ANALYSIS = True        # Include market context in analysis
CLAUDE_INCLUDE_FUTURE_ANALYSIS = False       # Include forward-looking analysis (for backtesting)

# Validation Rules Configuration
CLAUDE_VALIDATION_RULES = {
    'require_supporting_indicators': True,     # Require at least one supporting indicator
    'reject_critical_contradictions': True,   # Reject signals with critical contradictions
    'allow_minor_conflicts': True,            # Allow signals with minor indicator conflicts
    'minimum_validation_points': 1           # Minimum supporting technical factors required
}

# Enhanced File Saving Settings
CLAUDE_ENHANCED_SAVING = True                 # Enable enhanced analysis file saving
CLAUDE_SAVE_CLASSIFICATION_CONFLICTS = True  # Save files when Claude disagrees with system
CLAUDE_SAVE_VALIDATION_FAILURES = True      # Save files when technical validation fails
CLAUDE_SAVE_DIRECTORY = "claude_analysis_enhanced"  # Enhanced analysis directory

# Performance and Error Handling
CLAUDE_ROBUST_ERROR_HANDLING = True          # Enable robust error handling
CLAUDE_MULTIPLE_PARSE_ATTEMPTS = True        # Try multiple parsing methods
CLAUDE_GRACEFUL_DEGRADATION = True           # Gracefully handle partial failures

# Integration with Existing Settings
CLAUDE_RESPECT_EXISTING_THRESHOLDS = True    # Respect existing MIN_CONFIDENCE settings
CLAUDE_OVERRIDE_LOW_QUALITY = True           # Override system for low-quality signals

# Backtesting and Analysis Settings
CLAUDE_TIMESTAMP_ANALYSIS_ENABLED = True     # Enable timestamp-based analysis
CLAUDE_FUTURE_PERFORMANCE_TRACKING = True    # Track forward-looking performance
CLAUDE_PIP_CALCULATION_ENABLED = True        # Calculate pip movements for analysis

# Debug and Logging Settings
CLAUDE_DEBUG_TECHNICAL_VALIDATION = True     # Debug technical validation steps
CLAUDE_LOG_INDICATOR_ANALYSIS = True         # Log individual indicator analysis
CLAUDE_LOG_PARSING_STEPS = True              # Log response parsing steps
CLAUDE_VERBOSE_CONFLICT_LOGGING = False      # Detailed conflict logging (can be noisy)

# Strategy-Specific Claude Settings
CLAUDE_STRATEGY_SPECIFIC_ANALYSIS = {
    'EMA': {
        'require_alignment_check': True,
        'allow_transition_signals': True,
        'minimum_separation_pips': 2.0
    },
    'KAMA': {
        'minimum_efficiency_ratio': 0.3,
        'require_trend_confirmation': True,
        'allow_low_efficiency_signals': False
    },
    'COMBINED': {
        'require_majority_agreement': True,
        'minimum_supporting_strategies': 2,
        'conflict_resolution_method': 'claude_override'
    }
}

# Adaptive Analysis Settings
CLAUDE_ADAPTIVE_ANALYSIS = True              # Enable adaptive analysis based on market conditions
CLAUDE_MARKET_CONDITION_WEIGHTS = {
    'trending': 1.2,                         # Boost confidence in trending markets
    'ranging': 0.8,                          # Reduce confidence in ranging markets
    'high_volatility': 0.9,                  # Slight reduction in high volatility
    'low_volatility': 1.1                    # Slight boost in low volatility
}

# Quality Assurance Settings
CLAUDE_QUALITY_CHECKS = {
    'minimum_indicators_for_analysis': 3,    # Minimum indicators required for analysis
    'require_price_data': True,              # Require valid price data
    'validate_epic_format': True,            # Validate epic format
    'check_timestamp_validity': True,        # Check timestamp validity
    'require_strategy_identification': True  # Require strategy identification
}

# API Optimization Settings
CLAUDE_API_OPTIMIZATIONS = {
    'cache_similar_analyses': False,         # Cache similar analyses (experimental)
    'batch_similar_signals': False,         # Batch similar signals (experimental)
    'parallel_processing': False,           # Parallel processing (experimental)
    'rate_limit_handling': True,            # Handle rate limits gracefully
    'retry_on_failure': True,               # Retry failed API calls
    'max_retries': 2                        # Maximum retry attempts
}

# Compatibility Settings
CLAUDE_BACKWARD_COMPATIBILITY = True        # Maintain backward compatibility
CLAUDE_LEGACY_METHOD_SUPPORT = True        # Support legacy analysis methods
CLAUDE_MIGRATION_MODE = False              # Migration mode for existing systems

# Helper Functions for Enhanced Claude Configuration

def get_claude_validation_config(strategy: str) -> dict:
    """Get validation configuration for specific strategy"""
    return CLAUDE_STRATEGY_SPECIFIC_ANALYSIS.get(strategy, {})

def is_claude_enhanced_mode() -> bool:
    """Check if Claude enhanced mode is enabled"""
    return CLAUDE_ENHANCED_MODE and CLAUDE_TECHNICAL_VALIDATION

def get_claude_indicator_requirements(strategy: str) -> list:
    """Get required indicators for Claude analysis based on strategy"""
    requirements = {
        'EMA': ['ema_21', 'ema_50', 'ema_200', 'price'],
        'KAMA': ['kama_value', 'efficiency_ratio', 'ema_200'],
        'COMBINED': ['ema_21', 'ema_50', 'ema_200', 'price']
    }
    return requirements.get(strategy, ['price'])



# =============================================================================
# ENHANCED CLAUDE PRESETS
# =============================================================================

CLAUDE_ANALYSIS_PRESETS = {
    'strict': {
        'CLAUDE_TECHNICAL_VALIDATION': True,
        'CLAUDE_AGREEMENT_THRESHOLD': 0.9,
        'require_supporting_indicators': True,
        'minimum_validation_points': 2
    },
    'balanced': {
        'CLAUDE_TECHNICAL_VALIDATION': True,
        'CLAUDE_AGREEMENT_THRESHOLD': 0.8,
        'require_supporting_indicators': True,
        'minimum_validation_points': 1
    },
    'permissive': {
        'CLAUDE_TECHNICAL_VALIDATION': True,
        'CLAUDE_AGREEMENT_THRESHOLD': 0.7,
        'require_supporting_indicators': False,
        'minimum_validation_points': 1
    },
    'learning': {
        'CLAUDE_TECHNICAL_VALIDATION': False,        # No pre-validation
        'CLAUDE_SAVE_ALL_ANALYSES': True,           # Save everything for learning
        'CLAUDE_DETAILED_LOGGING': True,           # Maximum logging
        'CLAUDE_CONFLICT_LOGGING': True            # Log all conflicts
    }
}

# Set active preset
CLAUDE_ACTIVE_PRESET = 'balanced'  # Options: 'strict', 'balanced', 'permissive', 'learning'

def apply_claude_preset(preset_name: str):
    """Apply a Claude analysis preset"""
    if preset_name in CLAUDE_ANALYSIS_PRESETS:
        preset = CLAUDE_ANALYSIS_PRESETS[preset_name]
        globals().update(preset)
        print(f"✅ Applied Claude preset: {preset_name}")
        return True
    else:
        print(f"❌ Unknown Claude preset: {preset_name}")
        return False

# =============================================================================
# CLAUDE INTEGRATION WITH EXISTING SYSTEMS
# =============================================================================

# Integration with existing confidence systems
CLAUDE_CONFIDENCE_INTEGRATION = {
    'boost_high_claude_scores': True,        # Boost confidence for high Claude scores
    'penalize_low_claude_scores': True,      # Penalize confidence for low Claude scores
    'claude_score_weight': 0.2,              # Weight of Claude score in final confidence
    'min_claude_score_for_boost': 7,        # Minimum Claude score for confidence boost
    'max_claude_score_penalty': 0.1         # Maximum confidence penalty
}


# Integration with deduplication
CLAUDE_DEDUPLICATION_INTEGRATION = {
    'include_claude_score_in_hash': True,    # Include Claude score in deduplication hash
    'claude_decision_affects_dedup': True,   # Claude decision affects deduplication
    'separate_claude_approved_signals': True # Separate tracking for Claude-approved signals
}

# =============================================================================
# MONITORING AND STATISTICS
# =============================================================================

# Claude Performance Monitoring
CLAUDE_PERFORMANCE_TRACKING = {
    'track_response_times': True,            # Track Claude API response times
    'track_parsing_success_rate': True,     # Track parsing success rates
    'track_agreement_rates': True,          # Track system-Claude agreement rates
    'track_validation_failures': True,      # Track technical validation failures
    'save_performance_stats': True,         # Save performance statistics
    'performance_report_interval': 100      # Generate reports every N analyses
}

# Statistical Analysis Settings
CLAUDE_STATISTICS = {
    'calculate_prediction_accuracy': True,   # Calculate prediction accuracy over time
    'track_strategy_specific_performance': True,  # Track per-strategy performance
    'monitor_classification_conflicts': True,     # Monitor classification conflicts
    'analyze_market_condition_performance': True  # Analyze performance by market conditions
}

print("✅ Enhanced Claude API configuration loaded")
print(f"🔧 Technical validation: {'enabled' if CLAUDE_TECHNICAL_VALIDATION else 'disabled'}")
print(f"📊 Complete DataFrame analysis: {'enabled' if CLAUDE_COMPLETE_DATAFRAME_ANALYSIS else 'disabled'}")
print(f"🎯 Active preset: {CLAUDE_ACTIVE_PRESET}")
print(f"📁 Analysis directory: {CLAUDE_SAVE_DIRECTORY}")

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

ZERO_LAG_STRATEGY = True
# MOMENTUM_BIAS_STRATEGY removed - legacy strategy replaced by MOMENTUM_STRATEGY
MOMENTUM_STRATEGY = True

ZERO_LAG_STRATEGY_ENABLED = True
# MOMENTUM_BIAS_STRATEGY_ENABLED removed - legacy strategy replaced by MOMENTUM_STRATEGY
MOMENTUM_STRATEGY_ENABLED = True
USE_ZERO_LAG_STRATEGY = True
# USE_MOMENTUM_BIAS_STRATEGY removed - legacy strategy replaced by MOMENTUM_STRATEGY
USE_MOMENTUM_STRATEGY = True


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
# 🚨 CRITICAL SAFETY FILTERS CONFIGURATION
# Add this section to your config.py to prevent invalid trades
# =============================================================================

# Master switch for all critical safety filters
ENABLE_CRITICAL_SAFETY_FILTERS = True

# EMA 200 Trend Filter (Prevents contra-trend signals)
ENABLE_EMA200_CONTRADICTION_FILTER = True
EMA200_MINIMUM_MARGIN = 0.002  # 0.2% - Minimum margin required for contra-trend signals


# EMA Stack Alignment Filter (Prevents perfect stack contradictions)
ENABLE_EMA_STACK_CONTRADICTION_FILTER = True

# Multi-Indicator Consensus Requirement
REQUIRE_INDICATOR_CONSENSUS = True
MIN_CONFIRMING_INDICATORS = 1  # At least 1 indicator must confirm


# Emergency Circuit Breaker
ENABLE_EMERGENCY_CIRCUIT_BREAKER = True
MAX_CONTRADICTIONS_ALLOWED = 5  # Reject signals with more than this many critical contradictions

# Safety Filter Logging
SAFETY_FILTER_LOG_LEVEL = 'ERROR'  # Log all rejections as errors for visibility
LOG_REJECTION_DETAILS = True  # Log detailed rejection information

# Override for emergency situations (set to False in emergencies)
EMERGENCY_BYPASS_SAFETY_FILTERS = False

# Statistics tracking
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
# SAFETY FILTER PRESETS
# =============================================================================

# Preset configurations for different risk levels
SAFETY_FILTER_PRESETS = {
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

# Advanced Claude Analysis Configuration
USE_ADVANCED_CLAUDE_PROMPTS = True  # Enable institutional-grade analysis
CLAUDE_ANALYSIS_LEVEL = 'prop_trader'  # or 'hedge_fund', 'prop_trader', 'risk_manager'

CLAUDE_FAIL_SECURE = False         # True: reject on Claude errors, False: allow on errors
SAVE_CLAUDE_REJECTIONS = True      # Save rejected signals for analysis

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

