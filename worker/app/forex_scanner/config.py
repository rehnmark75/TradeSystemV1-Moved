# config.py
"""
Infrastructure configuration for the Forex Scanner.

This file contains ONLY:
- Environment variables (secrets, API keys, database URLs)
- Static mappings (epic maps, pair info)
- Helper functions for the above

All behavioral/strategy settings are in the database (strategy_config schema).
Use scanner_config_service.py to access runtime configuration.
"""
import os
from typing import List

# =============================================================================
# ENVIRONMENT VARIABLES (Secrets & Infrastructure)
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
MINIO_PUBLIC_URL = os.getenv('MINIO_PUBLIC_URL', f"http://{MINIO_ENDPOINT}")

# Order Execution API
ORDER_API_URL = os.getenv('ORDER_API_URL', "http://fastapi-dev:8000/orders/place-order")
API_SUBSCRIPTION_KEY = os.getenv('API_SUBSCRIPTION_KEY', "436abe054a074894a0517e5172f0e5b6")

# =============================================================================
# STATIC PAIR MAPPINGS (Used by pip calculations and order mapping)
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
# HELPER FUNCTIONS
# =============================================================================


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
    }


# =============================================================================
# END OF CONFIG - All other settings are in the database
# =============================================================================
