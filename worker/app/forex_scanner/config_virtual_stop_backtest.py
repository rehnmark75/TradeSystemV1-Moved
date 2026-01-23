"""
Scalp Mode Configuration for Backtesting

UPDATED: January 23, 2026 - Replaced VSL system with scalp-specific trailing configs
Mirrors live scalp trailing system from dev-app/config.py SCALP_TRAILING_CONFIGS

Key Changes from VSL System:
- VSL DEPRECATED: 5-6 pip virtual stops caused 67% premature exits
- NEW SYSTEM: Uses IG native stops with progressive trailing (6-8 pip initial SL)
- Analysis showed optimal stops: USDCAD 8 pips, Majors 7 pips, some 6 pips
- TP reduced from 15 to 10 pips (more realistic based on actual win patterns)

IMPORTANT: This file must stay in sync with live scalp trailing configs
Last synced: 2026-01-23 (aligned with Jan 20-23 SL/TP optimization)
"""

# =============================================================================
# PER-PAIR SCALP SL/TP CONFIGURATION (Jan 23, 2026 Optimization)
# =============================================================================

# Optimal stop loss and take profit per pair based on 3-day analysis
# Analysis showed average losses at 5-7 pips, wins peaking at 5-7 pips
# New strategy: Tighter SL (6-8 pips), realistic TP (10 pips)
PAIR_VSL_CONFIGS = {
    # Tight stops (6 pips) - Lowest observed loss patterns
    'CS.D.NZDUSD.MINI.IP': {'vsl_pips': 6.0, 'enabled': True},  # Avg loss 5.1 pips
    'CS.D.USDCHF.MINI.IP': {'vsl_pips': 6.0, 'enabled': True},  # Avg loss 5.0 pips, best R:R

    # Standard stops (7 pips) - Majority of pairs
    'CS.D.GBPUSD.MINI.IP': {'vsl_pips': 7.0, 'enabled': True},  # Avg loss 5.9 pips
    'CS.D.GBPUSD.CEEM.IP': {'vsl_pips': 7.0, 'enabled': True},
    'CS.D.EURJPY.MINI.IP': {'vsl_pips': 7.0, 'enabled': True},  # Avg loss 6.4 pips
    'CS.D.AUDJPY.MINI.IP': {'vsl_pips': 7.0, 'enabled': True},  # Avg loss 6.1 pips
    'CS.D.EURUSD.CEEM.IP': {'vsl_pips': 7.0, 'enabled': True},  # Worst performer, needs monitoring
    'CS.D.EURUSD.MINI.IP': {'vsl_pips': 7.0, 'enabled': True},

    # Wider stops (8 pips) - Higher volatility or buffer needed
    'CS.D.AUDUSD.MINI.IP': {'vsl_pips': 8.0, 'enabled': True},  # Best performer, keep wider
    'CS.D.USDJPY.MINI.IP': {'vsl_pips': 8.0, 'enabled': True},  # Highest MAE (8.0 pips)
    'CS.D.USDCAD.MINI.IP': {'vsl_pips': 8.0, 'enabled': True},  # Avg loss 6.5 pips + outlier buffer
    'CS.D.GBPJPY.MINI.IP': {'vsl_pips': 8.0, 'enabled': True},
}

# =============================================================================
# DEFAULT VALUES
# =============================================================================

# Default SL for unknown pairs (matches global optimized value)
DEFAULT_VSL_PIPS = 7.0

# Default scalp take profit (optimized from 15 to 10 pips based on actual win patterns)
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
