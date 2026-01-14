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


# =============================================================================
# DYNAMIC VSL TRAILING CONFIGURATION (mirrors dev-app/config_virtual_stop.py)
# =============================================================================

# Master toggle for dynamic VSL (moves SL to breakeven when in profit)
DYNAMIC_VSL_ENABLED = True

# Spread-aware threshold adjustment
# When spread widens, require more profit before triggering breakeven
SPREAD_AWARE_TRIGGERS_ENABLED = True
BASELINE_SPREAD_PIPS = 1.0  # Normal spread assumption
MAX_SPREAD_PENALTY_PIPS = 2.0  # Cap adjustment at +2 pips max

# Default dynamic VSL configuration for majors
DEFAULT_DYNAMIC_VSL_CONFIG = {
    'initial_vsl_pips': 3.0,        # Starting VSL (same as current fixed)
    'breakeven_trigger_pips': 3.0,  # Move to BE when +3 pips profit
    'breakeven_lock_pips': 0.5,     # Lock +0.5 pip at breakeven
    'stage1_trigger_pips': 4.5,     # Move to stage1 when +4.5 pips
    'stage1_lock_pips': 2.0,        # Lock +2 pips at stage1
    'target_pips': 5.0,             # TP target (close at this level)
}

# Default dynamic VSL configuration for JPY pairs (higher volatility)
DEFAULT_JPY_DYNAMIC_VSL_CONFIG = {
    'initial_vsl_pips': 4.0,        # Starting VSL (same as current fixed)
    'breakeven_trigger_pips': 3.5,  # Move to BE when +3.5 pips profit
    'breakeven_lock_pips': 0.5,     # Lock +0.5 pip at breakeven
    'stage1_trigger_pips': 5.0,     # Move to stage1 when +5 pips
    'stage1_lock_pips': 2.0,        # Lock +2 pips at stage1
    'target_pips': 6.0,             # TP target (higher for JPY)
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
    'CS.D.USDJPY.MINI.IP': DEFAULT_JPY_DYNAMIC_VSL_CONFIG.copy(),
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
    return DYNAMIC_VSL_ENABLED
