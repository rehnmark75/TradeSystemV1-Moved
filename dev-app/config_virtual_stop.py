"""
Virtual Stop Loss Configuration for Scalping Mode

This module configures the streaming-based virtual stop loss system that allows
tighter stop losses than IG's broker minimum by monitoring prices in real-time
and closing positions programmatically.

The broker's actual SL remains set at the minimum allowed (~10 pips) as a
safety net in case streaming fails.
"""

# =============================================================================
# MASTER TOGGLE
# =============================================================================

VIRTUAL_STOP_LOSS_ENABLED = True

# =============================================================================
# GLOBAL DEFAULTS
# =============================================================================

# Default virtual stop loss distance in pips
# Trade will be closed when price moves this many pips against entry
DEFAULT_VIRTUAL_SL_PIPS = 4.0

# Broker safety net SL (set on IG as actual stop loss)
# This catches the trade if streaming fails - should be IG's minimum
BROKER_SAFETY_SL_PIPS = 10

# How often to sync positions from database (seconds)
POSITION_SYNC_INTERVAL_SECONDS = 30

# Minimum time between close attempts for same position (seconds)
# Prevents rapid-fire close attempts if first one fails
CLOSE_ATTEMPT_COOLDOWN_SECONDS = 2

# =============================================================================
# LIGHTSTREAMER CONFIGURATION
# =============================================================================

# Lightstreamer endpoint (production)
LIGHTSTREAMER_PROD_URL = "https://apd.marketdatasystems.com"

# Lightstreamer endpoint (demo)
LIGHTSTREAMER_DEMO_URL = "https://demo-apd.marketdatasystems.com"

# Use demo or prod (should match your IG account type)
USE_DEMO_ENDPOINT = False

# Connection timeout (seconds)
LIGHTSTREAMER_CONNECT_TIMEOUT = 30

# Reconnection settings
LIGHTSTREAMER_AUTO_RECONNECT = True
LIGHTSTREAMER_RECONNECT_DELAY_SECONDS = 5
LIGHTSTREAMER_MAX_RECONNECT_ATTEMPTS = 10

# =============================================================================
# PER-PAIR CONFIGURATION
# =============================================================================

# Virtual SL settings per currency pair
# Pairs with higher volatility may need wider virtual stops
PAIR_VIRTUAL_SL_CONFIGS = {
    # Major pairs - tighter stops
    'CS.D.EURUSD.CEEM.IP': {
        'virtual_sl_pips': 4.0,
        'enabled': True,
    },
    'CS.D.EURUSD.MINI.IP': {
        'virtual_sl_pips': 4.0,
        'enabled': True,
    },
    'CS.D.GBPUSD.MINI.IP': {
        'virtual_sl_pips': 5.0,  # Slightly wider for GBP volatility
        'enabled': True,
    },
    'CS.D.GBPUSD.CEEM.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.USDCHF.MINI.IP': {
        'virtual_sl_pips': 4.0,
        'enabled': True,
    },
    'CS.D.USDCAD.MINI.IP': {
        'virtual_sl_pips': 4.0,
        'enabled': True,
    },
    'CS.D.AUDUSD.MINI.IP': {
        'virtual_sl_pips': 4.0,
        'enabled': True,
    },
    'CS.D.NZDUSD.MINI.IP': {
        'virtual_sl_pips': 4.0,
        'enabled': True,
    },

    # JPY pairs - different pip value, slightly wider
    'CS.D.USDJPY.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.EURJPY.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.GBPJPY.MINI.IP': {
        'virtual_sl_pips': 6.0,  # GBP/JPY is volatile
        'enabled': True,
    },
    'CS.D.AUDJPY.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_virtual_sl_config(epic: str) -> dict:
    """
    Get virtual stop loss configuration for an epic.

    Args:
        epic: Market epic (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        dict with 'virtual_sl_pips' and 'enabled' keys
    """
    return PAIR_VIRTUAL_SL_CONFIGS.get(epic, {
        'virtual_sl_pips': DEFAULT_VIRTUAL_SL_PIPS,
        'enabled': True,  # Default to enabled for unknown pairs
    })


def get_virtual_sl_pips(epic: str) -> float:
    """
    Get virtual stop loss distance in pips for an epic.

    Args:
        epic: Market epic

    Returns:
        Virtual SL distance in pips
    """
    config = get_virtual_sl_config(epic)
    return config.get('virtual_sl_pips', DEFAULT_VIRTUAL_SL_PIPS)


def is_virtual_sl_enabled(epic: str) -> bool:
    """
    Check if virtual stop loss is enabled for an epic.

    Args:
        epic: Market epic

    Returns:
        True if VSL is enabled for this epic
    """
    if not VIRTUAL_STOP_LOSS_ENABLED:
        return False
    config = get_virtual_sl_config(epic)
    return config.get('enabled', True)


def get_lightstreamer_url() -> str:
    """Get the appropriate Lightstreamer URL based on config."""
    if USE_DEMO_ENDPOINT:
        return LIGHTSTREAMER_DEMO_URL
    return LIGHTSTREAMER_PROD_URL
