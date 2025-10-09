# ================================================
# DEV-APP CONFIGURATION
# ================================================

# ================== EPIC MAPPINGS ==================
# Ticker maps of IG specific ticker names
EPIC_MAP = {
    "GBPUSD.1.MINI": "CS.D.GBPUSD.MINI.IP",        # ← REVERSED from scanner
    "EURUSD.1.MINI": "CS.D.EURUSD.CEEM.IP",
    "USDJPY.100.MINI": "CS.D.USDJPY.MINI.IP",
    "AUDUSD.1.MINI": "CS.D.AUDUSD.MINI.IP",
    "USDCAD.1.MINI": "CS.D.USDCAD.MINI.IP",
    "EURJPY.100.MINI": "CS.D.EURJPY.MINI.IP",
    "AUDJPY.100.MINI": "CS.D.AUDJPY.MINI.IP",
    "NZDUSD.1.MINI": "CS.D.NZDUSD.MINI.IP",
    "USDCHF.1.MINI": "CS.D.USDCHF.MINI.IP"
}

# Trading blacklist - prevent trading for specific epics (scan-only mode)
TRADING_BLACKLIST = [
    # "EURUSD.1.MINI",  # Removed - trading now enabled
    # Add other blocked epics here as needed
]

# Additional epic mappings from broker transaction analyzer
BROKER_EPIC_MAP = {
    'USD/CAD': 'CS.D.USDCAD.MINI.IP',
    'USD/CHF': 'CS.D.USDCHF.MINI.IP', 
    'USD/JPY': 'CS.D.USDJPY.MINI.IP',
    'EUR/JPY': 'CS.D.EURJPY.MINI.IP',
    'EUR/USD': 'CS.D.EURUSD.CEEM.IP',
    'GBP/USD': 'CS.D.GBPUSD.MINI.IP',
    'AUD/USD': 'CS.D.AUDUSD.MINI.IP',
    'NZD/USD': 'CS.D.NZDUSD.MINI.IP',
    'GBP/JPY': 'CS.D.GBPJPY.MINI.IP',
    'EUR/GBP': 'CS.D.EURGBP.MINI.IP'
}

# Default epics for testing and examples
DEFAULT_EPICS = {
    'EURUSD': 'CS.D.EURUSD.MINI.IP',
    'USDJPY': 'CS.D.USDJPY.MINI.IP',
    'GBPUSD': 'CS.D.GBPUSD.MINI.IP'
}

# ================== API ENDPOINTS ==================
# IG Markets API
API_BASE_URL = "https://demo-api.ig.com/gateway/deal"  # or your demo/live base URL

# Internal service URLs
FASTAPI_DEV_URL = "http://fastapi-dev:8000"
FASTAPI_STREAM_URL = "http://fastapi-stream:8000"

# Specific endpoints
ADJUST_STOP_URL = f"{FASTAPI_DEV_URL}/orders/adjust-stop"

# IG Lightstreamer (for streaming)
LIGHTSTREAMER_URL = "https://demo-apd.marketdatasystems.com"

# ================== AUTHENTICATION ==================
# Note: IG_API_KEY and IG_PWD values are now mapped to environment variables
# by the keyvault.py service. These names are kept for backward compatibility.
IG_USERNAME = "rehnmarkhdemo"
IG_API_KEY = "demoapikey"
IG_PWD = "demopwd"

# ================== TRADE COOLDOWN CONTROLS ==================
TRADE_COOLDOWN_ENABLED = True
TRADE_COOLDOWN_MINUTES = 30  # Default cooldown period after closing a trade
EPIC_SPECIFIC_COOLDOWNS = {
    'CS.D.EURUSD.MINI.IP': 45,  # Major pairs get longer cooldown
    'CS.D.GBPUSD.MINI.IP': 45,
    'CS.D.USDJPY.MINI.IP': 30,
    'CS.D.AUDUSD.MINI.IP': 30,
    'CS.D.USDCAD.MINI.IP': 30,
    # Others use DEFAULT_TRADE_COOLDOWN_MINUTES
}

# ================== RISK MANAGEMENT SETTINGS ==================
# Default risk-reward ratio (take profit will be this multiple of stop loss)
DEFAULT_RISK_REWARD_RATIO = 2.0

# Epic-specific risk-reward ratios (for fine-tuning based on volatility)
EPIC_RISK_REWARD_RATIOS = {
    'CS.D.EURUSD.MINI.IP': 2.5,  # More volatile pairs get higher RR
    'CS.D.GBPUSD.MINI.IP': 2.5,
    'CS.D.USDJPY.MINI.IP': 2.0,  # Stable pairs use default
    'CS.D.AUDUSD.MINI.IP': 2.0,
    'CS.D.USDCAD.MINI.IP': 2.0,
}

# ATR calculation settings
ATR_PERIODS = 14  # Number of periods for ATR calculation
ATR_STOP_MULTIPLIER = 1.5  # ATR multiplier for stop loss distance

# ================== PROGRESSIVE TRAILING SETTINGS ==================
# 3-Stage Progressive Trailing System (based on trade data analysis)

# Stage 1: Balanced Break-Even Protection (OPTIMIZED FOR PROFIT CAPTURE)
# NOTE: Dynamic trigger uses IG minimum distance + offset:
#       - JPY pairs: IG min + 8 points (e.g., 2 + 8 = 10 points for USDJPY)
#       - Other pairs: IG min + 4 points (e.g., 2 + 4 = 6 points for EURUSD)
#       This accommodates higher volatility in JPY pairs (0.01 point value vs 0.0001)
STAGE1_TRIGGER_POINTS = 7    # Fallback when IG minimum not available
STAGE1_LOCK_POINTS = 2       # Fallback: +2 point minimum profit (ENHANCED: uses IG min distance when available)

# Stage 2: Profit Lock-In (OPTIMIZED FOR TREND FOLLOWING)
STAGE2_TRIGGER_POINTS = 16   # Lock in meaningful profit after +16 points (was 12)
STAGE2_LOCK_POINTS = 10      # Guarantee +10 points profit (was 6)

# Stage 3: Dynamic Percentage Trailing (STANDARDIZED FOR ALL PAIRS)
STAGE3_TRIGGER_POINTS = 17   # Start percentage trailing after +17 points (was 15)
STAGE3_ATR_MULTIPLIER = 0.8  # MUCH TIGHTER: 0.8x ATR (was 1.5x)
STAGE3_MIN_DISTANCE = 2      # Small minimum trailing distance from current price (was 3)
STAGE3_MIN_ADJUSTMENT = 5    # Minimum points to move stop (prevents too-frequent tiny adjustments)

# ================== PAIR-SPECIFIC TRAILING CONFIGURATIONS ==================
# Per-pair trailing stop configurations (overrides default values above)
# Note: IG's min_stop_distance_points from trade_log ALWAYS takes priority when available
# These configs act as fallback or can set HIGHER values for tighter protection

PAIR_TRAILING_CONFIGS = {
    # ========== MAJOR PAIRS - Standard Volatility ==========

    'CS.D.EURUSD.MINI.IP': {
        'stage1_trigger_points': 16,      # Break-even trigger
        'stage1_lock_points': 4,          # Minimum profit lock
        'stage2_trigger_points': 22,      # Profit lock trigger
        'stage2_lock_points': 10,         # Profit guarantee
        'stage3_trigger_points': 30,      # Start percentage trailing
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 12,   # Move to BE after 6 pts
    },

    'CS.D.AUDUSD.MINI.IP': {
        'stage1_trigger_points': 16,      # Break-even trigger
        'stage1_lock_points': 4,          # Minimum profit lock
        'stage2_trigger_points': 22,      # Profit lock trigger
        'stage2_lock_points': 10,         # Profit guarantee
        'stage3_trigger_points': 30,      # Start percentage trailing
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 12, 
    },

    'CS.D.NZDUSD.MINI.IP': {
        'stage1_trigger_points': 16,      # Break-even trigger
        'stage1_lock_points': 4,          # Minimum profit lock
        'stage2_trigger_points': 22,      # Profit lock trigger
        'stage2_lock_points': 10,         # Profit guarantee
        'stage3_trigger_points': 30,      # Start percentage trailing
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 12, 
    },

    'CS.D.USDCAD.MINI.IP': {
        'stage1_trigger_points': 16,      # Break-even trigger
        'stage1_lock_points': 4,          # Minimum profit lock
        'stage2_trigger_points': 22,      # Profit lock trigger
        'stage2_lock_points': 10,         # Profit guarantee
        'stage3_trigger_points': 30,      # Start percentage trailing
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 12, 
    },

    'CS.D.USDCHF.MINI.IP': {
        'stage1_trigger_points': 16,      # Break-even trigger
        'stage1_lock_points': 4,          # Minimum profit lock
        'stage2_trigger_points': 22,      # Profit lock trigger
        'stage2_lock_points': 10,         # Profit guarantee
        'stage3_trigger_points': 30,      # Start percentage trailing
        'stage3_atr_multiplier': 0.8,     # ATR trailing multiplier
        'stage3_min_distance': 2,         # Minimum trail distance
        'min_trail_distance': 15,         # Overall minimum distance
        'break_even_trigger_points': 12, 
    },

    # ========== GBP PAIRS - High Volatility ==========

    'CS.D.GBPUSD.MINI.IP': {
        'stage1_trigger_points': 20,      # Wider activation
        'stage1_lock_points': 4,          # More profit lock
        'stage2_trigger_points': 25,      # Higher trigger
        'stage2_lock_points': 12,         # More protection
        'stage3_trigger_points': 35,      # Later trailing start
        'stage3_atr_multiplier': 1.0,     # Wider trailing
        'stage3_min_distance': 3,         # More distance
        'min_trail_distance': 18,         # Higher minimum
        'break_even_trigger_points': 12,   # Later BE
    },

    'CS.D.GBPJPY.MINI.IP': {
        'stage1_trigger_points': 15,      # Wide for high volatility
        'stage1_lock_points': 3,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 22,
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 8,
    },

    'CS.D.GBPAUD.MINI.IP': {
        'stage1_trigger_points': 15,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 22,
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 8,
    },

    'CS.D.GBPNZD.MINI.IP': {
        'stage1_trigger_points': 15,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 22,
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 8,
    },

    # ========== JPY PAIRS - Different Pip Scale ==========

    'CS.D.USDJPY.MINI.IP': {
        'stage1_trigger_points': 15,      # Tighter for JPY scale
        'stage1_lock_points': 2,
        'stage2_trigger_points': 20,      # Adjusted for JPY
        'stage2_lock_points': 8,
        'stage3_trigger_points': 25,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 2,
        'min_trail_distance': 12,
        'break_even_trigger_points': 12,
    },

    'CS.D.EURJPY.MINI.IP': {
        'stage1_trigger_points': 20,      # Slightly wider (more volatile)
        'stage1_lock_points': 2,
        'stage2_trigger_points': 30,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 35,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 12,
    },

    'CS.D.AUDJPY.MINI.IP': {
        'stage1_trigger_points': 18,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 22,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 26,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 12,
        # Profit Protection Rule
        'enable_profit_protection': False,
    },

    'CS.D.CADJPY.MINI.IP': {
        'stage1_trigger_points': 12,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 16,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 18,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    'CS.D.CHFJPY.MINI.IP': {
        'stage1_trigger_points': 12,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 16,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 18,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    'CS.D.NZDJPY.MINI.IP': {
        'stage1_trigger_points': 12,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 16,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 18,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    # ========== CROSS PAIRS - Medium-High Volatility ==========

    'CS.D.EURAUD.MINI.IP': {
        'stage1_trigger_points': 14,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 18,
        'stage2_lock_points': 11,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 16,
        'break_even_trigger_points': 7,
    },

    'CS.D.EURNZD.MINI.IP': {
        'stage1_trigger_points': 14,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 18,
        'stage2_lock_points': 11,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.9,
        'stage3_min_distance': 2,
        'min_trail_distance': 16,
        'break_even_trigger_points': 7,
    },

    'CS.D.AUDNZD.MINI.IP': {
        'stage1_trigger_points': 13,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 17,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 19,
        'stage3_atr_multiplier': 0.85,
        'stage3_min_distance': 2,
        'min_trail_distance': 15,
        'break_even_trigger_points': 7,
    },

    'CS.D.EURGBP.MINI.IP': {
        'stage1_trigger_points': 11,      # Tighter (less volatile)
        'stage1_lock_points': 2,
        'stage2_trigger_points': 14,
        'stage2_lock_points': 9,
        'stage3_trigger_points': 16,
        'stage3_atr_multiplier': 0.75,
        'stage3_min_distance': 2,
        'min_trail_distance': 14,
        'break_even_trigger_points': 12,
    },
}

# Default configuration for pairs not explicitly configured above
DEFAULT_TRAILING_CONFIG = {
    'stage1_trigger_points': 12,
    'stage1_lock_points': 2,
    'stage2_trigger_points': 16,
    'stage2_lock_points': 10,
    'stage3_trigger_points': 17,
    'stage3_atr_multiplier': 0.8,
    'stage3_min_distance': 2,
    'min_trail_distance': 15,
    'break_even_trigger_points': 12,
}


def get_trailing_config_for_epic(epic: str) -> dict:
    """
    Get trailing stop configuration for specific epic/pair.

    Priority:
    1. Pair-specific config from PAIR_TRAILING_CONFIGS
    2. DEFAULT_TRAILING_CONFIG fallback

    Note: IG's min_stop_distance_points from trade_log ALWAYS takes priority
    when available. These configs are fallback or can set HIGHER values.

    Args:
        epic: Trading symbol (e.g., 'CS.D.EURUSD.MINI.IP')

    Returns:
        Dictionary with trailing configuration values
    """
    config = PAIR_TRAILING_CONFIGS.get(epic, DEFAULT_TRAILING_CONFIG.copy())
    return config

# ================== DEFAULT VALUES ==================
DEFAULT_TEST_EPIC = "CS.D.USDJPY.MINI.IP"

