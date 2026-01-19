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
# Using 5 pips as default - analysis showed 3 pips was too tight,
# causing correct signals to be stopped out during normal retracements
DEFAULT_VIRTUAL_SL_PIPS = 5.0

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
# Analysis from Jan 15-16, 2026 showed 3 pips was too tight for majors:
# - 93.9% VSL hit rate with only 6.1% win rate
# - 67% of signals were directionally correct but stopped out during retracements
# - MAE ranged from 3.0-8.4 pips on losing trades
# Widened to 5 pips for majors, 6 pips for JPY (higher volatility)
PAIR_VIRTUAL_SL_CONFIGS = {
    # Major pairs - 5 pips VSL (widened from 3 pips)
    'CS.D.EURUSD.CEEM.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.EURUSD.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.GBPUSD.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.GBPUSD.CEEM.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.USDCHF.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.USDCAD.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.AUDUSD.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },
    'CS.D.NZDUSD.MINI.IP': {
        'virtual_sl_pips': 5.0,
        'enabled': True,
    },

    # JPY pairs - 6 pips VSL (higher volatility - increased from 4 pips)
    # Analysis showed 4 pips was too tight - normal MAE is 6-14 pips
    # Trades were in correct direction but hit VSL during retracements
    'CS.D.USDJPY.MINI.IP': {
        'virtual_sl_pips': 8.0,
        'enabled': True,
    },
    'CS.D.EURJPY.MINI.IP': {
        'virtual_sl_pips': 6.0,
        'enabled': True,
    },
    'CS.D.GBPJPY.MINI.IP': {
        'virtual_sl_pips': 6.0,
        'enabled': True,
    },
    'CS.D.AUDJPY.MINI.IP': {
        'virtual_sl_pips': 6.0,
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


# =============================================================================
# DYNAMIC VSL TRAILING CONFIGURATION
# =============================================================================

# Master toggle for dynamic VSL (moves SL to breakeven when in profit)
DYNAMIC_VSL_ENABLED = True

# Spread-aware threshold adjustment
# When spread widens, require more profit before triggering breakeven
SPREAD_AWARE_TRIGGERS_ENABLED = True
BASELINE_SPREAD_PIPS = 1.0  # Normal spread assumption
MAX_SPREAD_PENALTY_PIPS = 2.0  # Cap adjustment at +2 pips max

# Default dynamic VSL configuration for majors
# Updated Jan 16, 2026: Aligned with widened initial VSL (3->5 pips)
# Stages adjusted to give trades more room to develop
DEFAULT_DYNAMIC_VSL_CONFIG = {
    'initial_vsl_pips': 5.0,        # Starting VSL (widened from 3.0)
    'breakeven_trigger_pips': 3.0,  # Move to BE when +3 pips profit (widened from 2.0)
    'breakeven_lock_pips': 0.5,     # Lock +0.5 pip at breakeven
    'stage1_trigger_pips': 6.0,     # Move to stage1 when +6 pips profit (from 5.0)
    'stage1_lock_pips': 2.5,        # Lock +2.5 pips at stage1 (from 2.0)
    'stage2_trigger_pips': 8.0,     # Move to stage2 when +8 pips profit (from 6.5)
    'stage2_lock_pips': 6.0,        # Lock +6 pips at stage2
    'target_pips': 10.0,            # TP target (extended from 8.0)
}

# Default dynamic VSL configuration for JPY pairs (higher volatility)
# Initial VSL increased from 4 to 6 pips based on analysis showing
# normal MAE is 6-14 pips - 4 pips was too tight causing premature exits
DEFAULT_JPY_DYNAMIC_VSL_CONFIG = {
    'initial_vsl_pips': 6.0,        # Starting VSL (increased from 4.0)
    'breakeven_trigger_pips': 2.5,  # Move to BE when +2.5 pips profit (backtested)
    'breakeven_lock_pips': 0.5,     # Lock +0.5 pip at breakeven
    'stage1_trigger_pips': 6.0,     # Move to stage1 when +6 pips profit
    'stage1_lock_pips': 2.5,        # Lock +2.5 pips at stage1
    'stage2_trigger_pips': 8.0,     # Move to stage2 when +8 pips profit
    'stage2_lock_pips': 5.0,        # Lock +5 pips at stage2
    'target_pips': 10.0,            # TP target (extended from 6.0)
}

# Per-pair dynamic VSL configuration
PAIR_DYNAMIC_VSL_CONFIGS = {
    # Major pairs - use default major config
    'CS.D.EURUSD.CEEM.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.EURUSD.MINI.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.GBPUSD.MINI.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.GBPUSD.CEEM.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.USDCHF.MINI.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.USDCAD.MINI.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.AUDUSD.MINI.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.NZDUSD.MINI.IP': DEFAULT_DYNAMIC_VSL_CONFIG.copy(),

    # JPY pairs - use JPY config (higher volatility)
    # USDJPY: Custom config with 8 pip initial VSL (widened from 6)
    'CS.D.USDJPY.MINI.IP': {
        'initial_vsl_pips': 8.0,        # Widened from 6.0
        'breakeven_trigger_pips': 3.0,  # Move to BE when +3 pips profit
        'breakeven_lock_pips': 0.5,     # Lock +0.5 pip at breakeven
        'stage1_trigger_pips': 7.0,     # Move to stage1 when +7 pips profit
        'stage1_lock_pips': 3.0,        # Lock +3 pips at stage1
        'stage2_trigger_pips': 10.0,    # Move to stage2 when +10 pips profit
        'stage2_lock_pips': 6.0,        # Lock +6 pips at stage2
        'target_pips': 12.0,            # TP target
    },
    'CS.D.EURJPY.MINI.IP': DEFAULT_JPY_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.GBPJPY.MINI.IP': DEFAULT_JPY_DYNAMIC_VSL_CONFIG.copy(),
    'CS.D.AUDJPY.MINI.IP': DEFAULT_JPY_DYNAMIC_VSL_CONFIG.copy(),
}


def get_dynamic_vsl_config(epic: str) -> dict:
    """
    Get dynamic VSL configuration for an epic.

    Args:
        epic: Market epic (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        dict with dynamic VSL stage configuration
    """
    if epic in PAIR_DYNAMIC_VSL_CONFIGS:
        return PAIR_DYNAMIC_VSL_CONFIGS[epic]

    # Check if it's a JPY pair by name
    if 'JPY' in epic:
        return DEFAULT_JPY_DYNAMIC_VSL_CONFIG.copy()

    return DEFAULT_DYNAMIC_VSL_CONFIG.copy()


def is_dynamic_vsl_enabled() -> bool:
    """Check if dynamic VSL trailing is enabled globally."""
    return DYNAMIC_VSL_ENABLED and VIRTUAL_STOP_LOSS_ENABLED
