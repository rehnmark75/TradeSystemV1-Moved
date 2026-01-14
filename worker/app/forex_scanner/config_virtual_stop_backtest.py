"""
Virtual Stop Loss Configuration for Scalping Backtest Mode

Mirrors dev-app/config_virtual_stop.py for backtesting purposes.
These values determine the VSL levels used during trade simulation.

Key difference from live VSL:
- Live: Uses Lightstreamer to monitor real-time prices
- Backtest: Simulates VSL checks bar-by-bar using OHLC data

VSL allows tighter stop losses than IG's broker minimum (~10 pips)
by monitoring prices in real-time and closing programmatically.
"""

# =============================================================================
# PER-PAIR VSL CONFIGURATION
# =============================================================================

# Virtual SL settings per currency pair (matches live trading config)
# Majors: 3 pips VSL
# JPY pairs: 4 pips VSL (higher volatility)
PAIR_VSL_CONFIGS = {
    # Major pairs - 3 pips VSL
    'CS.D.EURUSD.CEEM.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.EURUSD.MINI.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.GBPUSD.MINI.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.GBPUSD.CEEM.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.USDCHF.MINI.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.USDCAD.MINI.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.AUDUSD.MINI.IP': {'vsl_pips': 3.0, 'enabled': True},
    'CS.D.NZDUSD.MINI.IP': {'vsl_pips': 3.0, 'enabled': True},

    # JPY pairs - 4 pips VSL (higher volatility)
    'CS.D.USDJPY.MINI.IP': {'vsl_pips': 4.0, 'enabled': True},
    'CS.D.EURJPY.MINI.IP': {'vsl_pips': 4.0, 'enabled': True},
    'CS.D.GBPJPY.MINI.IP': {'vsl_pips': 4.0, 'enabled': True},
    'CS.D.AUDJPY.MINI.IP': {'vsl_pips': 4.0, 'enabled': True},
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

# Default VSL for unknown pairs
DEFAULT_VSL_PIPS = 3.0

# Default scalp take profit
DEFAULT_SCALP_TP_PIPS = 5.0

# Default limit order offset for scalping (momentum confirmation)
# Live system uses per-pair overrides from database, this is the fallback
DEFAULT_SCALP_LIMIT_OFFSET_PIPS = 3.0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_vsl_pips(epic: str) -> float:
    """
    Get Virtual Stop Loss distance in pips for an epic.

    Args:
        epic: Market epic (e.g., 'CS.D.EURUSD.CEEM.IP')

    Returns:
        VSL distance in pips (3.0 for majors, 4.0 for JPY pairs)
    """
    config = PAIR_VSL_CONFIGS.get(epic, {})
    return config.get('vsl_pips', DEFAULT_VSL_PIPS)


def get_vsl_config(epic: str) -> dict:
    """
    Get full VSL configuration for an epic.

    Args:
        epic: Market epic

    Returns:
        dict with 'vsl_pips' and 'enabled' keys
    """
    return PAIR_VSL_CONFIGS.get(epic, {
        'vsl_pips': DEFAULT_VSL_PIPS,
        'enabled': True,
    })


def is_vsl_enabled(epic: str) -> bool:
    """
    Check if VSL is enabled for an epic.

    Args:
        epic: Market epic

    Returns:
        True if VSL is enabled for this epic
    """
    config = get_vsl_config(epic)
    return config.get('enabled', True)


def is_jpy_pair(epic: str) -> bool:
    """
    Check if epic is a JPY pair (needs different pip calculation).

    Args:
        epic: Market epic

    Returns:
        True if this is a JPY pair
    """
    return 'JPY' in epic


def get_pip_multiplier(epic: str) -> float:
    """
    Get pip multiplier for price calculations.

    Args:
        epic: Market epic

    Returns:
        100.0 for JPY pairs, 10000.0 for standard pairs
    """
    return 100.0 if is_jpy_pair(epic) else 10000.0


def get_all_vsl_configs() -> dict:
    """
    Get all VSL configurations.

    Returns:
        Dictionary mapping epic to VSL config
    """
    return PAIR_VSL_CONFIGS.copy()
