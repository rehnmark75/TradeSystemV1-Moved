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
STAGE1_TRIGGER_POINTS = 7    # Move to break-even after +7 points profit (was 3)
STAGE1_LOCK_POINTS = 2       # Guarantee +2 point minimum profit (was 1)

# Stage 2: Profit Lock-In (OPTIMIZED FOR TREND FOLLOWING)
STAGE2_TRIGGER_POINTS = 12   # Lock in meaningful profit after +12 points (was 5)
STAGE2_LOCK_POINTS = 6       # Guarantee +6 points profit (was 3)

# Stage 3: Dynamic ATR Trailing (OPTIMIZED FOR LARGER MOVES)
STAGE3_TRIGGER_POINTS = 20   # Start ATR trailing after +20 points (was 8)
STAGE3_ATR_MULTIPLIER = 1.5  # ATR multiplier for trailing distance
STAGE3_MIN_DISTANCE = 3      # Minimum trailing distance in points (was 2)

# Epic-specific progressive settings (for high-performance pairs)
PROGRESSIVE_EPIC_SETTINGS = {
    'CS.D.EURUSD.MINI.IP': {
        'stage1_trigger': 6,  # Balanced for major pairs (was 2)
        'stage2_trigger': 10,  # Allow trends to develop (was 4)
        'stage3_trigger': 18   # ATR trailing for big moves (was 7)
    },
    'CS.D.GBPUSD.MINI.IP': {
        'stage1_trigger': 6,  # Balanced for major pairs (was 2)
        'stage2_trigger': 10,  # Allow trends to develop (was 4)
        'stage3_trigger': 18   # ATR trailing for big moves (was 7)
    },
    'CS.D.USDJPY.MINI.IP': {
        'stage1_trigger': 6,   # Changed from 40 to 6 for more practical trading
        'stage2_trigger': 10,  # Changed from 60 to 10 for more practical trading
        'stage3_trigger': 18   # Changed from 100 to 18 for more practical trading
    },
    'CS.D.EURJPY.MINI.IP': {
        'stage1_trigger': 6,   # Changed from 40 to 6 for more practical trading
        'stage2_trigger': 10,  # Changed from 60 to 10 for more practical trading
        'stage3_trigger': 18   # Changed from 100 to 18 for more practical trading
    },
    # Additional major and minor pairs with balanced settings
    'CS.D.USDCHF.MINI.IP': {
        'stage1_trigger': 6,   # Balanced for major pairs
        'stage2_trigger': 10,  # Allow trends to develop
        'stage3_trigger': 18   # ATR trailing for big moves
    },
    'CS.D.AUDUSD.MINI.IP': {
        'stage1_trigger': 6,   # Balanced for major pairs
        'stage2_trigger': 10,  # Allow trends to develop
        'stage3_trigger': 18   # ATR trailing for big moves
    },
    'CS.D.USDCAD.MINI.IP': {
        'stage1_trigger': 6,   # Balanced for major pairs
        'stage2_trigger': 10,  # Allow trends to develop
        'stage3_trigger': 18   # ATR trailing for big moves
    },
    'CS.D.NZDUSD.MINI.IP': {
        'stage1_trigger': 7,   # Slightly more conservative for minor pairs
        'stage2_trigger': 12,  # Allow trends to develop
        'stage3_trigger': 20   # ATR trailing for big moves
    },
    'CS.D.AUDJPY.MINI.IP': {
        'stage1_trigger': 6,   # Changed from 40 to 6 for more practical trading
        'stage2_trigger': 10,  # Changed from 60 to 10 for more practical trading
        'stage3_trigger': 18   # Changed from 100 to 18 for more practical trading
    },
    'CS.D.GBPJPY.MINI.IP': {
        'stage1_trigger': 6,   # Changed from 40 to 6 for more practical trading
        'stage2_trigger': 10,  # Changed from 60 to 10 for more practical trading
        'stage3_trigger': 18   # Changed from 100 to 18 for more practical trading
    }
}

# ================== DEFAULT VALUES ==================
DEFAULT_TEST_EPIC = "CS.D.USDJPY.MINI.IP"

