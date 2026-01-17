"""
Virtual Stop Loss Configuration for Scalping Backtest Mode

Mirrors dev-app/config_virtual_stop.py for backtesting purposes.
These values determine the VSL levels used during trade simulation.

Key difference from live VSL:
- Live: Uses Lightstreamer to monitor real-time prices
- Backtest: Simulates VSL checks bar-by-bar using OHLC data

VSL allows tighter stop losses than IG's broker minimum (~10 pips)
by monitoring prices in real-time and closing programmatically.

IMPORTANT: This file must stay in sync with dev-app/config_virtual_stop.py
Last synced: 2026-01-17 (aligned with Jan 15-16 analysis results)
"""

# =============================================================================
# PER-PAIR VSL CONFIGURATION
# =============================================================================

# Virtual SL settings per currency pair
# Analysis from Jan 15-16, 2026 showed 3 pips was too tight for majors:
# - 93.9% VSL hit rate with only 6.1% win rate
# - 67% of signals were directionally correct but stopped out during retracements
# - MAE ranged from 3.0-8.4 pips on losing trades
# Widened to 5 pips for majors, 6 pips for JPY (higher volatility)
PAIR_VSL_CONFIGS = {
    # Major pairs - 5 pips VSL (widened from 3 pips based on Jan 2026 analysis)
    'CS.D.EURUSD.CEEM.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.EURUSD.MINI.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.GBPUSD.MINI.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.GBPUSD.CEEM.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.USDCHF.MINI.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.USDCAD.MINI.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.AUDUSD.MINI.IP': {'vsl_pips': 5.0, 'enabled': True},
    'CS.D.NZDUSD.MINI.IP': {'vsl_pips': 5.0, 'enabled': True},

    # JPY pairs - 6 pips VSL (higher volatility - increased from 4 pips)
    # Analysis showed 4 pips was too tight - normal MAE is 6-14 pips
    # Trades were in correct direction but hit VSL during retracements
    'CS.D.USDJPY.MINI.IP': {'vsl_pips': 6.0, 'enabled': True},
    'CS.D.EURJPY.MINI.IP': {'vsl_pips': 6.0, 'enabled': True},
    'CS.D.GBPJPY.MINI.IP': {'vsl_pips': 6.0, 'enabled': True},
    'CS.D.AUDJPY.MINI.IP': {'vsl_pips': 6.0, 'enabled': True},
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

# Default VSL for unknown pairs (matches live: 5 pips)
DEFAULT_VSL_PIPS = 5.0

# Default scalp take profit (matches live dynamic VSL target)
DEFAULT_SCALP_TP_PIPS = 10.0

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
        VSL distance in pips (5.0 for majors, 6.0 for JPY pairs)
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
# DYNAMIC VSL TRAILING CONFIGURATION
# Synced with dev-app/config_virtual_stop.py (Jan 17, 2026)
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
