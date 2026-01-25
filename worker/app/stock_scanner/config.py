"""
Stock Scanner Configuration

Configuration for the stock scanner module.
Separate from forex_scanner to allow independent configuration.
"""

import os
from typing import List, Dict, Any

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Stocks database (separate from forex)
STOCKS_DATABASE_URL = os.getenv(
    "STOCKS_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/stocks"
)

# For local development
LOCAL_DATABASE_URL = os.getenv(
    "LOCAL_STOCKS_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/stocks"
)

# =============================================================================
# ROBOMARKETS API CONFIGURATION
# =============================================================================

ROBOMARKETS_API_KEY = os.getenv(
    "ROBOMARKETS_API_KEY",
    "28e2c0dab20ac9f6fd54917100be3c9b8d1c650cde5205b8b79c6c795b6fc676"
)

ROBOMARKETS_ACCOUNT_ID = os.getenv("ROBOMARKETS_ACCOUNT_ID", "")

ROBOMARKETS_API_URL = os.getenv(
    "ROBOMARKETS_API_URL",
    "https://api.stockstrader.com/api/v1"
)

# API settings
ROBOMARKETS_TIMEOUT = 30  # Request timeout in seconds
ROBOMARKETS_MAX_RETRIES = 3

# =============================================================================
# FINNHUB API CONFIGURATION (News Data)
# =============================================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

# Rate limiting: 30 req/min (50% of free tier's 60/min limit)
FINNHUB_MAX_REQUESTS_PER_MINUTE = 30
FINNHUB_TIMEOUT = 30
FINNHUB_MAX_RETRIES = 3

# News settings
NEWS_LOOKBACK_DAYS = 7  # How far back to fetch news
NEWS_CACHE_TTL = 3600   # Cache news for 1 hour
NEWS_MIN_ARTICLES = 2   # Minimum articles for confident sentiment

# Analyst recommendations (Finnhub)
FINNHUB_RECO_CACHE_TTL_HOURS = 24
FINNHUB_RECO_MAX_PER_SCAN = 20

# =============================================================================
# CLAUDE AI ANALYSIS CONFIGURATION
# =============================================================================

# Set to False to disable Claude AI analysis in the daily pipeline
CLAUDE_ANALYSIS_ENABLED = os.getenv("CLAUDE_ANALYSIS_ENABLED", "false").lower() == "true"

# =============================================================================
# DATA FETCHING CONFIGURATION
# =============================================================================

# Primary timeframe for data storage
PRIMARY_TIMEFRAME = "1h"  # 1-hour candles (730 days history available)

# Synthesized timeframes (created from 1h data)
SYNTHESIZED_TIMEFRAMES = ["4h", "1d"]

# yfinance limits by interval
YFINANCE_LIMITS = {
    "1m": 7,      # 7 days max
    "5m": 60,     # 60 days max
    "15m": 60,    # 60 days max
    "1h": 730,    # ~2 years
    "1d": None,   # Unlimited
}

# Default fetch settings
DEFAULT_HISTORY_DAYS = 60  # Default days to fetch
MAX_CONCURRENT_FETCHES = 5  # Max parallel yfinance requests

# Data cache settings
DATA_CACHE_TTL = 300  # 5 minutes

# =============================================================================
# WATCHLIST CONFIGURATION
# =============================================================================

# Default watchlist - US stocks from NYSE and NASDAQ only (no ETFs)
# This will be synced with actual RoboMarkets instruments on startup
DEFAULT_WATCHLIST: List[str] = [
    # Tech Giants (NASDAQ)
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta Platforms
    "TSLA",   # Tesla

    # Financial (NYSE)
    "JPM",    # JPMorgan
    "BAC",    # Bank of America
    "GS",     # Goldman Sachs
    "V",      # Visa (NYSE)
    "MA",     # Mastercard (NYSE)

    # Healthcare (NYSE)
    "JNJ",    # Johnson & Johnson
    "UNH",    # UnitedHealth
    "PFE",    # Pfizer

    # Consumer (NYSE)
    "WMT",    # Walmart
    "KO",     # Coca-Cola
    "PG",     # Procter & Gamble
    "MCD",    # McDonald's

    # Energy (NYSE)
    "XOM",    # Exxon Mobil
    "CVX",    # Chevron

    # Industrial (NYSE)
    "BA",     # Boeing
    "CAT",    # Caterpillar
]

# Sector mapping for stocks
SECTOR_MAPPING: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer Cyclical", "NVDA": "Technology", "META": "Technology",
    "TSLA": "Consumer Cyclical", "JPM": "Financial", "BAC": "Financial",
    "GS": "Financial", "V": "Financial", "MA": "Financial",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "WMT": "Consumer Defensive", "KO": "Consumer Defensive",
    "PG": "Consumer Defensive", "MCD": "Consumer Cyclical",
    "XOM": "Energy", "CVX": "Energy", "BA": "Industrial", "CAT": "Industrial",
}

# =============================================================================
# MARKET HOURS CONFIGURATION
# =============================================================================

# US Market Hours (Eastern Time, converted to UTC)
US_MARKET_HOURS = {
    "pre_market_open": {"hour": 8, "minute": 0},   # 4:00 AM ET
    "market_open": {"hour": 14, "minute": 30},     # 9:30 AM ET
    "market_close": {"hour": 21, "minute": 0},     # 4:00 PM ET
    "post_market_close": {"hour": 24, "minute": 0}, # 8:00 PM ET
    "trading_days": [0, 1, 2, 3, 4],  # Monday-Friday
    "timezone": "America/New_York"
}

# EU Market Hours (for LSE, etc.)
EU_MARKET_HOURS = {
    "market_open": {"hour": 8, "minute": 0},   # 8:00 AM GMT
    "market_close": {"hour": 16, "minute": 30}, # 4:30 PM GMT
    "trading_days": [0, 1, 2, 3, 4],
    "timezone": "Europe/London"
}

# =============================================================================
# SCANNING CONFIGURATION
# =============================================================================

# Scan interval in seconds
SCAN_INTERVAL = 300  # 5 minutes (stocks are slower than forex)

# Minimum confidence for signals
MIN_CONFIDENCE = 0.60

# Signal settings
MIN_BARS_FOR_SIGNAL = 50  # Minimum candles needed for signal generation

# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

# Enable/disable strategies
EMA_STRATEGY = True
MACD_STRATEGY = True
VOLUME_BREAKOUT_STRATEGY = True

# Strategy-specific settings
STRATEGY_CONFIG = {
    "ema": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "trend_ema": 200,
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    },
    "volume_breakout": {
        "volume_multiplier": 2.0,  # Volume must be 2x average
        "breakout_period": 20,     # Period for high/low breakout
        "avg_volume_period": 20,   # Period for average volume
    }
}

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Position sizing
POSITION_SIZE_PERCENT = 2.0  # Percent of account per trade
MAX_POSITION_SIZE = 100      # Maximum shares per position

# Stop loss / Take profit (percentage-based for stocks)
DEFAULT_STOP_LOSS_PERCENT = 2.0   # 2% stop loss
DEFAULT_TAKE_PROFIT_PERCENT = 4.0  # 4% take profit (2:1 R:R)

# Maximum positions
MAX_OPEN_POSITIONS = 5

# Daily limits
MAX_DAILY_TRADES = 10
MAX_DAILY_LOSS_PERCENT = 5.0  # Stop trading after 5% daily loss

# =============================================================================
# ALERT & NOTIFICATION SETTINGS
# =============================================================================

# Alert settings
ENABLE_ALERTS = True
ALERT_COOLDOWN_MINUTES = 30  # Longer cooldown for stocks

# Telegram (reuse from forex_scanner if available)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# =============================================================================
# ORDER EXECUTION
# =============================================================================

# Enable live trading
AUTO_TRADING_ENABLED = False  # Start disabled for safety

# Order settings
DEFAULT_ORDER_TYPE = "market"  # 'market' or 'limit'
LIMIT_ORDER_OFFSET_PERCENT = 0.1  # For limit orders, offset from current price

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("STOCK_SCANNER_LOG_LEVEL", "INFO")
LOG_DIR = "logs"
LOG_FILE = "stock_scanner.log"

# Performance logging
LOG_SLOW_QUERIES = True
SLOW_QUERY_THRESHOLD = 1.0  # Log queries slower than 1 second

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

# EMA periods to calculate
EMA_PERIODS = [12, 21, 50, 200]

# RSI settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2.0

# ATR
ATR_PERIOD = 14

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

DEFAULT_BACKTEST_DAYS = 90
BACKTEST_COMMISSION = 0.001  # 0.1% commission per trade
BACKTEST_SLIPPAGE = 0.001    # 0.1% slippage

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_database_url(use_local: bool = False) -> str:
    """Get appropriate database URL"""
    if use_local:
        return LOCAL_DATABASE_URL
    return STOCKS_DATABASE_URL


def is_market_open() -> bool:
    """
    Check if US stock market is currently open

    Returns:
        True if market is open
    """
    from datetime import datetime
    import pytz

    et = pytz.timezone("America/New_York")
    now = datetime.now(et)

    # Check if weekday
    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    # Check market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now <= market_close


def get_market_status() -> Dict[str, Any]:
    """
    Get detailed market status

    Returns:
        Dict with market status info
    """
    from datetime import datetime
    import pytz

    et = pytz.timezone("America/New_York")
    now = datetime.now(et)

    is_open = is_market_open()

    status = {
        "is_open": is_open,
        "current_time_et": now.strftime("%Y-%m-%d %H:%M:%S ET"),
        "weekday": now.strftime("%A"),
        "status": "OPEN" if is_open else "CLOSED",
    }

    if not is_open:
        # Calculate time until open
        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            next_open = next_open + timedelta(days=days_until_monday)
        elif now.hour < 9 or (now.hour == 9 and now.minute < 30):
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            next_open = next_open + timedelta(days=1)
            if next_open.weekday() >= 5:
                days_until_monday = 7 - next_open.weekday()
                next_open = next_open + timedelta(days=days_until_monday)

        status["next_open"] = next_open.strftime("%Y-%m-%d %H:%M:%S ET")

    return status


# Import timedelta for get_market_status
from datetime import timedelta

# VERSION INFO
# =============================================================================

STOCK_SCANNER_VERSION = "0.1.0"
STOCK_SCANNER_DATE = "2024-12-07"

# Print config on load
print(f"[OK] Stock Scanner config loaded v{STOCK_SCANNER_VERSION}")
print(f"[DB] Database: {STOCKS_DATABASE_URL.split('@')[-1] if '@' in STOCKS_DATABASE_URL else 'configured'}")
print(f"[API] RoboMarkets API: {'configured' if ROBOMARKETS_API_KEY else 'not configured'}")
print(f"[LIST] Watchlist: {len(DEFAULT_WATCHLIST)} stocks")
