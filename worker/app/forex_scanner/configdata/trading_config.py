# configdata/trading_config.py
"""
Trading Configuration - Stop Loss / Take Profit Settings
Centralized SL/TP management for all strategies
"""

# ============================================================
# STOP LOSS / TAKE PROFIT CONFIGURATION
# ============================================================

# Database Parameter Usage
# Set to False during testing to use only ATR/static calculations
# Set to True in production to use optimized database parameters
USE_OPTIMIZED_DATABASE_PARAMS = False  # TESTING MODE - No DB params

# Database Parameter Freshness (days)
# How old can DB parameters be before falling back to ATR
DB_PARAM_FRESHNESS_DAYS = 30  # 30 days freshness window

# ATR Multipliers (default for all strategies)
DEFAULT_STOP_ATR_MULTIPLIER = 2.0  # 2x ATR for stop loss
DEFAULT_TARGET_ATR_MULTIPLIER = 4.0  # 4x ATR for take profit (2:1 RR)

# ============================================================
# PAIR-SPECIFIC MINIMUM STOP LOSS (in pips)
# These are ABSOLUTE MINIMUMS - never go below these values
# ============================================================

PAIR_MINIMUM_STOPS = {
    # Major Pairs (tight spreads, moderate volatility)
    'EURUSD': 15,   # Most liquid pair
    'USDJPY': 20,   # JPY pair, different pip scale

    # GBP Pairs (higher volatility)
    'GBPUSD': 20,   # Cable - volatile
    'EURGBP': 18,   # EUR/GBP cross
    'GBPJPY': 30,   # JPY cross, very volatile
    'GBPAUD': 25,   # Commodity cross
    'GBPNZD': 25,   # Commodity cross
    'GBPCAD': 22,   # Commodity cross
    'GBPCHF': 22,   # Safe haven cross

    # Commodity Pairs (moderate to high volatility)
    'AUDUSD': 18,   # Aussie dollar
    'NZDUSD': 18,   # Kiwi dollar
    'USDCAD': 18,   # Loonie

    # JPY Crosses (higher volatility)
    'EURJPY': 25,   # EUR/JPY cross
    'AUDJPY': 25,   # AUD/JPY cross
    'NZDJPY': 25,   # NZD/JPY cross
    'CADJPY': 25,   # CAD/JPY cross
    'CHFJPY': 28,   # CHF/JPY cross

    # EUR Crosses
    'EURAUD': 20,
    'EURNZD': 20,
    'EURCAD': 18,
    'EURCHF': 16,

    # Safe Haven / Exotic
    'USDCHF': 18,   # Swissie
    'AUDCAD': 20,   # Commodity cross
    'AUDNZD': 15,   # Close correlation pair
    'NZDCAD': 20,   # Commodity cross

    # Default fallback
    'DEFAULT': 20   # Safe default for unknown pairs
}

# ============================================================
# MAXIMUM STOP LOSS (in pips) - prevent excessive risk
# ============================================================

PAIR_MAXIMUM_STOPS = {
    'JPY_PAIRS': 60,    # All JPY pairs
    'GBP_PAIRS': 50,    # All GBP pairs
    'DEFAULT': 40       # Others
}

# ============================================================
# STATIC FALLBACK VALUES (when ATR unavailable)
# ============================================================

STATIC_FALLBACK_SL_TP = {
    # Major Pairs
    'EURUSD': {'sl': 18, 'tp': 36},
    'GBPUSD': {'sl': 25, 'tp': 50},
    'USDJPY': {'sl': 22, 'tp': 44},
    'USDCHF': {'sl': 20, 'tp': 40},

    # Commodity Pairs
    'AUDUSD': {'sl': 20, 'tp': 40},
    'NZDUSD': {'sl': 20, 'tp': 40},
    'USDCAD': {'sl': 20, 'tp': 40},

    # GBP Crosses
    'EURGBP': {'sl': 22, 'tp': 44},
    'GBPJPY': {'sl': 35, 'tp': 70},
    'GBPAUD': {'sl': 28, 'tp': 56},
    'GBPNZD': {'sl': 28, 'tp': 56},
    'GBPCAD': {'sl': 25, 'tp': 50},

    # JPY Crosses
    'EURJPY': {'sl': 30, 'tp': 60},
    'AUDJPY': {'sl': 28, 'tp': 56},
    'NZDJPY': {'sl': 28, 'tp': 56},
    'CADJPY': {'sl': 28, 'tp': 56},
    'CHFJPY': {'sl': 32, 'tp': 64},

    # EUR Crosses
    'EURAUD': {'sl': 22, 'tp': 44},
    'EURNZD': {'sl': 22, 'tp': 44},
    'EURCAD': {'sl': 20, 'tp': 40},
    'EURCHF': {'sl': 18, 'tp': 36},

    # Commodity Crosses
    'AUDCAD': {'sl': 22, 'tp': 44},
    'AUDNZD': {'sl': 18, 'tp': 36},
    'NZDCAD': {'sl': 22, 'tp': 44},

    # Default fallback
    'DEFAULT': {'sl': 20, 'tp': 40}  # Safe default
}

# ============================================================
# RISK/REWARD RATIOS
# ============================================================

# Minimum risk/reward ratio (TP must be at least this multiple of SL)
MINIMUM_RISK_REWARD_RATIO = 1.5

# Target risk/reward ratio when using ATR calculation
TARGET_RISK_REWARD_RATIO = 2.0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_minimum_stop_for_pair(epic: str) -> int:
    """
    Get minimum stop loss for a trading pair

    Args:
        epic: Epic code (e.g., 'CS.D.GBPUSD.MINI.IP')

    Returns:
        Minimum stop loss in pips
    """
    pair = _extract_pair_from_epic(epic)
    return PAIR_MINIMUM_STOPS.get(pair, PAIR_MINIMUM_STOPS['DEFAULT'])

def get_maximum_stop_for_pair(epic: str) -> int:
    """
    Get maximum stop loss for a trading pair

    Args:
        epic: Epic code

    Returns:
        Maximum stop loss in pips
    """
    pair = _extract_pair_from_epic(epic)

    if 'JPY' in pair:
        return PAIR_MAXIMUM_STOPS['JPY_PAIRS']
    elif 'GBP' in pair:
        return PAIR_MAXIMUM_STOPS['GBP_PAIRS']
    else:
        return PAIR_MAXIMUM_STOPS['DEFAULT']

def get_static_fallback_for_pair(epic: str) -> dict:
    """
    Get static fallback SL/TP for a trading pair

    Args:
        epic: Epic code

    Returns:
        Dict with 'sl' and 'tp' keys (in pips)
    """
    pair = _extract_pair_from_epic(epic)
    return STATIC_FALLBACK_SL_TP.get(pair, STATIC_FALLBACK_SL_TP['DEFAULT'])

def _extract_pair_from_epic(epic: str) -> str:
    """
    Extract currency pair from epic code

    Args:
        epic: Epic code (e.g., 'CS.D.GBPUSD.MINI.IP')

    Returns:
        Pair name (e.g., 'GBPUSD') or 'DEFAULT'
    """
    epic_upper = epic.upper()

    # Check known pairs
    for pair in PAIR_MINIMUM_STOPS.keys():
        if pair != 'DEFAULT' and pair in epic_upper:
            return pair

    # Try to extract from dot-separated epic
    parts = epic_upper.split('.')
    for part in parts:
        if len(part) in [6, 7] and any(curr in part for curr in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']):
            return part

    return 'DEFAULT'

# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_trading_config() -> dict:
    """
    Validate trading configuration

    Returns:
        Dict with validation results
    """
    errors = []
    warnings = []

    # Check ATR multipliers are reasonable
    if DEFAULT_STOP_ATR_MULTIPLIER < 1.0 or DEFAULT_STOP_ATR_MULTIPLIER > 5.0:
        warnings.append(f"Stop ATR multiplier {DEFAULT_STOP_ATR_MULTIPLIER} outside typical range (1.0-5.0)")

    if DEFAULT_TARGET_ATR_MULTIPLIER < 2.0 or DEFAULT_TARGET_ATR_MULTIPLIER > 10.0:
        warnings.append(f"Target ATR multiplier {DEFAULT_TARGET_ATR_MULTIPLIER} outside typical range (2.0-10.0)")

    # Check risk/reward ratio
    actual_rr = DEFAULT_TARGET_ATR_MULTIPLIER / DEFAULT_STOP_ATR_MULTIPLIER
    if actual_rr < MINIMUM_RISK_REWARD_RATIO:
        errors.append(f"Risk/reward ratio {actual_rr:.2f} below minimum {MINIMUM_RISK_REWARD_RATIO}")

    # Check all static fallbacks have valid values
    for pair, values in STATIC_FALLBACK_SL_TP.items():
        if 'sl' not in values or 'tp' not in values:
            errors.append(f"Static fallback for {pair} missing 'sl' or 'tp'")
        elif values['sl'] <= 0 or values['tp'] <= 0:
            errors.append(f"Static fallback for {pair} has invalid values: {values}")
        elif values['tp'] < values['sl'] * MINIMUM_RISK_REWARD_RATIO:
            warnings.append(f"Static fallback for {pair} has low RR: {values['tp']/values['sl']:.2f}")

    # Check minimum stops are reasonable
    for pair, min_stop in PAIR_MINIMUM_STOPS.items():
        if pair != 'DEFAULT' and min_stop < 10:
            warnings.append(f"Minimum stop for {pair} is very tight: {min_stop} pips")
        elif min_stop > 50:
            warnings.append(f"Minimum stop for {pair} is very wide: {min_stop} pips")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'summary': {
            'total_pairs_configured': len(PAIR_MINIMUM_STOPS) - 1,  # Exclude DEFAULT
            'db_params_enabled': USE_OPTIMIZED_DATABASE_PARAMS,
            'stop_atr_multiplier': DEFAULT_STOP_ATR_MULTIPLIER,
            'target_atr_multiplier': DEFAULT_TARGET_ATR_MULTIPLIER,
            'actual_risk_reward': DEFAULT_TARGET_ATR_MULTIPLIER / DEFAULT_STOP_ATR_MULTIPLIER
        }
    }

# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def get_trading_config_summary() -> dict:
    """
    Get summary of trading configuration

    Returns:
        Dict with configuration summary
    """
    validation = validate_trading_config()

    return {
        'mode': 'TESTING (DB disabled)' if not USE_OPTIMIZED_DATABASE_PARAMS else 'PRODUCTION (DB enabled)',
        'db_params_enabled': USE_OPTIMIZED_DATABASE_PARAMS,
        'db_freshness_days': DB_PARAM_FRESHNESS_DAYS,
        'atr_multipliers': {
            'stop': DEFAULT_STOP_ATR_MULTIPLIER,
            'target': DEFAULT_TARGET_ATR_MULTIPLIER,
            'risk_reward': DEFAULT_TARGET_ATR_MULTIPLIER / DEFAULT_STOP_ATR_MULTIPLIER
        },
        'minimum_stops_range': {
            'min': min([v for k, v in PAIR_MINIMUM_STOPS.items() if k != 'DEFAULT']),
            'max': max([v for k, v in PAIR_MINIMUM_STOPS.items() if k != 'DEFAULT']),
            'default': PAIR_MINIMUM_STOPS['DEFAULT']
        },
        'pairs_configured': len(PAIR_MINIMUM_STOPS) - 1,  # Exclude DEFAULT
        'validation': validation
    }

# Run validation on import
_validation_result = validate_trading_config()
if not _validation_result['valid']:
    print("⚠️ Trading configuration has errors:")
    for error in _validation_result['errors']:
        print(f"   ❌ {error}")
else:
    print(f"✅ Trading configuration valid - {len(PAIR_MINIMUM_STOPS)-1} pairs configured, DB params: {'DISABLED' if not USE_OPTIMIZED_DATABASE_PARAMS else 'ENABLED'}")

if _validation_result['warnings']:
    print("⚠️ Trading configuration warnings:")
    for warning in _validation_result['warnings']:
        print(f"   ⚠️ {warning}")
