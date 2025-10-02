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
    "EURUSD.1.MINI",  # No IG trading permissions for FX_NOR exchange
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

# REMOVED: Epic-specific progressive settings - now using standard 3-stage configuration for all pairs

# ================== DEFAULT VALUES ==================
DEFAULT_TEST_EPIC = "CS.D.USDJPY.MINI.IP"

