# configdata/news_filter_config.py
"""
Economic News Filter Configuration
Add these settings to your main config.py or environment variables
"""

# ===== ECONOMIC NEWS FILTERING CONFIGURATION =====

# Enable/disable news filtering
ENABLE_NEWS_FILTERING = True

# Economic calendar service connection
ECONOMIC_CALENDAR_URL = "http://economic-calendar:8091"
NEWS_SERVICE_TIMEOUT_SECONDS = 5

# Risk assessment timing
NEWS_HIGH_IMPACT_BUFFER_MINUTES = 30    # Block trades 30min before high impact news
NEWS_MEDIUM_IMPACT_BUFFER_MINUTES = 15  # Block trades 15min before medium impact news
NEWS_LOOKAHEAD_HOURS = 4                 # Check news up to 4 hours ahead

# Risk tolerance levels
BLOCK_TRADES_BEFORE_HIGH_IMPACT_NEWS = True    # Block trades before high impact news
BLOCK_TRADES_BEFORE_MEDIUM_IMPACT_NEWS = False # Allow trades before medium impact news
REDUCE_CONFIDENCE_NEAR_NEWS = True             # Reduce confidence when news is near

# Performance and caching
NEWS_CACHE_DURATION_MINUTES = 5  # Cache news data for 5 minutes
NEWS_FILTER_FAIL_SAFE = True     # Allow trades if news service fails

# Critical economic events (always high risk regardless of impact level)
CRITICAL_ECONOMIC_EVENTS = [
    "Non-Farm Employment Change",
    "NFP",
    "FOMC",
    "Federal Funds Rate",
    "ECB Press Conference",
    "Interest Rate Decision",
    "CPI",
    "Core CPI",
    "GDP",
    "Employment",
    "Unemployment Rate",
    "ECB Monetary Policy Meeting",
    "BOE Interest Rate Decision",
    "BOJ Policy Rate",
    "RBA Interest Rate Decision"
]

# Fail mode configuration
NEWS_FILTER_FAIL_SECURE = False  # If True, block trades when news service fails

# ===== EXAMPLE USAGE IN MAIN CONFIG.PY =====
"""
# Add to your main config.py file:

# Import news filter settings
from configdata.news_filter_config import *

# Or override specific settings:
ENABLE_NEWS_FILTERING = True
NEWS_HIGH_IMPACT_BUFFER_MINUTES = 45  # More conservative
BLOCK_TRADES_BEFORE_MEDIUM_IMPACT_NEWS = True  # Block medium impact too
"""

# ===== ENVIRONMENT VARIABLE EXAMPLES =====
"""
# Set these environment variables if you prefer:

export ENABLE_NEWS_FILTERING=true
export ECONOMIC_CALENDAR_URL=http://economic-calendar:8091
export NEWS_HIGH_IMPACT_BUFFER_MINUTES=30
export NEWS_MEDIUM_IMPACT_BUFFER_MINUTES=15
export BLOCK_TRADES_BEFORE_HIGH_IMPACT_NEWS=true
export REDUCE_CONFIDENCE_NEAR_NEWS=true
"""