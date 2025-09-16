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

# Stage 1: Quick Break-Even Protection
STAGE1_TRIGGER_POINTS = 3    # Move to break-even after +3 points profit
STAGE1_LOCK_POINTS = 1       # Guarantee +1 point minimum profit

# Stage 2: Profit Lock-In
STAGE2_TRIGGER_POINTS = 5    # Lock in meaningful profit after +5 points
STAGE2_LOCK_POINTS = 3       # Guarantee +3 points profit

# Stage 3: Dynamic ATR Trailing
STAGE3_TRIGGER_POINTS = 8    # Start ATR trailing after +8 points
STAGE3_ATR_MULTIPLIER = 1.5  # ATR multiplier for trailing distance
STAGE3_MIN_DISTANCE = 2      # Minimum trailing distance in points

# Epic-specific progressive settings (for high-performance pairs)
PROGRESSIVE_EPIC_SETTINGS = {
    'CS.D.EURUSD.MINI.IP': {
        'stage1_trigger': 2,  # More aggressive for major pairs
        'stage2_trigger': 4,
        'stage3_trigger': 7
    },
    'CS.D.GBPUSD.MINI.IP': {
        'stage1_trigger': 2,
        'stage2_trigger': 4,
        'stage3_trigger': 7
    },
    'CS.D.USDJPY.MINI.IP': {
        'stage1_trigger': 40,  # JPY calibrated: 40 points = 4 real pips
        'stage2_trigger': 60,  # JPY calibrated: 60 points = 6 real pips
        'stage3_trigger': 100  # JPY calibrated: 100 points = 10 real pips
    },
    'CS.D.EURJPY.MINI.IP': {
        'stage1_trigger': 40,  # JPY calibrated: 40 points = 4 real pips
        'stage2_trigger': 60,  # JPY calibrated: 60 points = 6 real pips
        'stage3_trigger': 100  # JPY calibrated: 100 points = 10 real pips
    }
}

# ================== DEFAULT VALUES ==================
DEFAULT_TEST_EPIC = "CS.D.USDJPY.MINI.IP"

