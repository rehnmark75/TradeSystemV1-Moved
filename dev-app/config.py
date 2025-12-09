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
    'EURUSD': 'CS.D.EURUSD.CEEM.IP',
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
    'CS.D.EURUSD.CEEM.IP': 45,  # Major pairs get longer cooldown
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
    'CS.D.EURUSD.CEEM.IP': 2.5,  # More volatile pairs get higher RR
    'CS.D.GBPUSD.MINI.IP': 2.5,
    'CS.D.USDJPY.MINI.IP': 2.0,  # Stable pairs use default
    'CS.D.AUDUSD.MINI.IP': 2.0,
    'CS.D.USDCAD.MINI.IP': 2.0,
}

# ATR calculation settings
ATR_PERIODS = 14  # Number of periods for ATR calculation
ATR_STOP_MULTIPLIER = 1.5  # ATR multiplier for stop loss distance

# ================== PROGRESSIVE TRAILING SETTINGS ==================
# 4-Stage Progressive Trailing System (based on MAE analysis Dec 2025)
#
# MAE ANALYSIS FINDINGS (14 days of trade data):
# - Winners: Avg MAE 3.1 pips, Median 2.7 pips, 75th percentile 3.5 pips
# - Losers: Avg MAE 15.0 pips, Median 13.2 pips
# - Conclusion: Good trades barely dip, bad trades dip significantly
#
# SMALL ACCOUNT PRIORITY: Protect capital early, accept smaller winners

# Stage 0: EARLY BREAKEVEN (NEW - Small Account Protection)
# Triggers early to protect capital when trade shows initial profit
# Based on MAE analysis: winners only dip 3 pips, so +6 pips is safe
EARLY_BREAKEVEN_TRIGGER_POINTS = 6   # Move to breakeven after +6 points
EARLY_BREAKEVEN_BUFFER_POINTS = 1    # SL moves to entry + 1 point (covers spread)

# Stage 1: Profit Lock (formerly Break-Even)
# Lock small guaranteed profit
STAGE1_TRIGGER_POINTS = 10   # Lock profit after +10 points (was 7)
STAGE1_LOCK_POINTS = 5       # Guarantee +5 points profit (was 2)

# Stage 2: Profit Lock-In (OPTIMIZED FOR TREND FOLLOWING)
STAGE2_TRIGGER_POINTS = 15   # Lock in meaningful profit after +15 points (was 16)
STAGE2_LOCK_POINTS = 10      # Guarantee +10 points profit

# Stage 3: Dynamic Percentage Trailing (STANDARDIZED FOR ALL PAIRS)
STAGE3_TRIGGER_POINTS = 20   # Start percentage trailing after +20 points (was 17)
STAGE3_ATR_MULTIPLIER = 0.8  # MUCH TIGHTER: 0.8x ATR (was 1.5x)
STAGE3_MIN_DISTANCE = 2      # Small minimum trailing distance from current price (was 3)
STAGE3_MIN_ADJUSTMENT = 5    # Minimum points to move stop (prevents too-frequent tiny adjustments)

# ================== PARTIAL CLOSE CONFIGURATION ==================
# Partial close feature: Close part of position at break-even trigger instead of moving stop
ENABLE_PARTIAL_CLOSE_AT_BREAKEVEN = True  # Master toggle for partial close feature
PARTIAL_CLOSE_SIZE = 0.5  # Size to close (0.5 = 50% of position)

# ================== PAIR-SPECIFIC TRAILING CONFIGURATIONS ==================
# Per-pair trailing stop configurations (overrides default values above)
# Note: IG's min_stop_distance_points from trade_log ALWAYS takes priority when available
# These configs act as fallback or can set HIGHER values for tighter protection

PAIR_TRAILING_CONFIGS = {
    # ========== MAJOR PAIRS - Standard Volatility ==========

    # ========== MAJOR PAIRS - Early Profit Protection (Small Account Mode) ==========
    # Based on MAE analysis: Winners dip only 3 pips avg, so early BE at +6 is safe

    # CEEM epic uses scaled pricing (11646 instead of 1.1646)
    'CS.D.EURUSD.CEEM.IP': {
        'early_breakeven_trigger_points': 6,  # Move to BE after +6 pts
        'early_breakeven_buffer_points': 1,   # SL at entry + 1 pt
        'stage1_trigger_points': 10,          # Lock profit after +10 pts
        'stage1_lock_points': 5,              # Guarantee +5 pts profit
        'stage2_trigger_points': 15,          # Profit lock trigger
        'stage2_lock_points': 10,             # Profit guarantee
        'stage3_trigger_points': 20,          # Start percentage trailing
        'stage3_atr_multiplier': 0.8,         # ATR trailing multiplier
        'stage3_min_distance': 4,             # Minimum trail distance
        'min_trail_distance': 4,              # Overall minimum distance
        'break_even_trigger_points': 6,       # Legacy field (uses early_breakeven now)
        'enable_partial_close': True,         # Enable partial close
        'partial_close_trigger_points': 13,   # Partial close at +13 pips
        'partial_close_size': 0.5,            # Close 50% of position
    },

    'CS.D.AUDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    'CS.D.NZDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    'CS.D.USDCAD.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    'CS.D.USDCHF.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    # ========== GBP PAIRS - High Volatility (slightly wider early BE) ==========

    'CS.D.GBPUSD.MINI.IP': {
        'early_breakeven_trigger_points': 8,  # Slightly wider for GBP volatility
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 12,
        'stage1_lock_points': 6,
        'stage2_trigger_points': 18,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 25,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 8,
        'enable_partial_close': True,
        'partial_close_trigger_points': 15,   # Higher for GBP volatility
        'partial_close_size': 0.5,
    },

    'CS.D.GBPJPY.MINI.IP': {
        'stage1_trigger_points': 15,
        'stage1_lock_points': 3,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 22,
        'stage3_atr_multiplier': 1.0,
        'stage3_min_distance': 3,
        'min_trail_distance': 18,
        'break_even_trigger_points': 8,
        'enable_partial_close': True,
        'partial_close_trigger_points': 18,   # Higher for GBPJPY volatility
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 18,   # Higher for cross pair volatility
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 18,   # Higher for cross pair volatility
        'partial_close_size': 0.5,
    },

    # ========== JPY PAIRS - Different Pip Scale (Early Protection) ==========

    'CS.D.USDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    'CS.D.EURJPY.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    'CS.D.AUDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 6,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,
        'stage2_lock_points': 10,
        'stage3_trigger_points': 20,
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': 4,
        'break_even_trigger_points': 6,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
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
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },

    'CS.D.EURGBP.MINI.IP': {
        'stage1_trigger_points': 11,
        'stage1_lock_points': 2,
        'stage2_trigger_points': 14,
        'stage2_lock_points': 9,
        'stage3_trigger_points': 16,
        'stage3_atr_multiplier': 0.75,
        'stage3_min_distance': 2,
        'min_trail_distance': 14,
        'break_even_trigger_points': 12,
        'enable_partial_close': True,
        'partial_close_trigger_points': 13,
        'partial_close_size': 0.5,
    },
}

# Default configuration for pairs not explicitly configured above
# Uses early profit protection settings for small account safety
DEFAULT_TRAILING_CONFIG = {
    'early_breakeven_trigger_points': 6,  # Move to BE after +6 pts
    'early_breakeven_buffer_points': 1,   # SL at entry + 1 pt
    'stage1_trigger_points': 10,          # Lock profit after +10 pts
    'stage1_lock_points': 5,              # Guarantee +5 pts profit
    'stage2_trigger_points': 15,          # Profit lock trigger
    'stage2_lock_points': 10,             # Profit guarantee
    'stage3_trigger_points': 20,          # Start percentage trailing
    'stage3_atr_multiplier': 0.8,         # ATR trailing multiplier
    'stage3_min_distance': 4,             # Minimum trail distance
    'min_trail_distance': 4,              # Overall minimum distance
    'break_even_trigger_points': 6,       # Legacy field
    'enable_partial_close': True,         # Enable partial close
    'partial_close_trigger_points': 13,   # Partial close at +13 pips (separate from BE)
    'partial_close_size': 0.5,            # Close 50% of position
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
        epic: Trading symbol (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        Dictionary with trailing configuration values
    """
    config = PAIR_TRAILING_CONFIGS.get(epic, DEFAULT_TRAILING_CONFIG.copy())
    return config

# ================== DEFAULT VALUES ==================
DEFAULT_TEST_EPIC = "CS.D.USDJPY.MINI.IP"

