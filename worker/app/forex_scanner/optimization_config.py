# optimization_config.py
"""
Parameter Configuration for Multi-Epic Optimization

Defines tiered parameter grids for different optimization depths:
- fast: Core SL/TP/Confidence only (~15-20 combinations after R:R filter)
- medium: + Buffer and filter toggles (~50-80 combinations)
- extended: Full parameter sweep with all filters (~150-200 combinations)

R:R filtering (TP/SL >= 1.5 and <= 4.0) is applied to reduce invalid combinations.
"""

from typing import Dict, List, Any


# ============================================
# PARAMETER TIERS
# ============================================

PARAMETER_TIERS: Dict[str, Dict[str, List[Any]]] = {
    'fast': {
        # Core risk management parameters only
        # Optimized for ~15 combinations after R:R filter
        'fixed_stop_loss_pips': [8, 10, 12],
        'fixed_take_profit_pips': [16, 20, 25],
        'min_confidence': [0.48, 0.52],
    },

    'medium': {
        # Core + Key filter toggles
        # Target: ~50-80 combinations per epic
        'fixed_stop_loss_pips': [8, 10, 12],
        'fixed_take_profit_pips': [16, 20, 25],
        'min_confidence': [0.45, 0.50, 0.55],
        'sl_buffer_pips': [4, 6],
        # Key filters that block signals
        'volume_filter_enabled': [True, False],
        'macd_filter_enabled': [True, False],
    },

    'extended': {
        # Full parameter sweep with all major filters
        # Target: ~150-200 combinations per epic
        'fixed_stop_loss_pips': [7, 8, 9, 10, 12],
        'fixed_take_profit_pips': [14, 16, 20, 25],
        'min_confidence': [0.45, 0.50, 0.55],
        'sl_buffer_pips': [4, 6],
        # Key filters that block signals
        'volume_filter_enabled': [True, False],
        'macd_filter_enabled': [True, False],
        'session_filter_enabled': [True, False],
    },
}


# ============================================
# R:R FILTER CONSTRAINTS
# ============================================

RR_FILTER = {
    'min_rr_ratio': 1.5,   # Minimum acceptable Risk:Reward ratio
    'max_rr_ratio': 4.0,   # Maximum (avoid unrealistic targets)
}


# ============================================
# EPIC MAPPINGS
# ============================================

# Shortcut names to full epic IDs
EPIC_SHORTCUTS = {
    'EURUSD': 'CS.D.EURUSD.CEEM.IP',
    'GBPUSD': 'CS.D.GBPUSD.MINI.IP',
    'USDJPY': 'CS.D.USDJPY.MINI.IP',
    'AUDUSD': 'CS.D.AUDUSD.MINI.IP',
    'USDCHF': 'CS.D.USDCHF.MINI.IP',
    'USDCAD': 'CS.D.USDCAD.MINI.IP',
    'NZDUSD': 'CS.D.NZDUSD.MINI.IP',
    'EURJPY': 'CS.D.EURJPY.MINI.IP',
    'AUDJPY': 'CS.D.AUDJPY.MINI.IP',
}

# Reverse mapping for display
EPIC_TO_SHORTCUT = {v: k for k, v in EPIC_SHORTCUTS.items()}


# ============================================
# COMPOSITE SCORE WEIGHTS
# ============================================

SCORE_WEIGHTS = {
    'win_rate': 0.30,           # 30% weight on win rate
    'profit_factor': 0.30,      # 30% weight on profit factor (normalized)
    'total_pips': 0.20,         # 20% weight on total pips (normalized)
    'signal_count': 0.20,       # 20% weight on signal count (prefer more signals)
}

# Normalization factors for composite score
SCORE_NORMALIZATION = {
    'profit_factor_cap': 3.0,   # PF above 3.0 treated as 3.0 for scoring
    'pips_divisor': 100.0,      # Divide pips by this for normalization
    'signal_count_cap': 50,     # Signal count above 50 treated as 50
}


# ============================================
# DEFAULTS
# ============================================

DEFAULT_MODE = 'fast'
DEFAULT_DAYS = 30
DEFAULT_OUTPUT_DIR = '/app/forex_scanner/optimization_results'


def get_tier_info(mode: str) -> Dict[str, Any]:
    """Get information about a parameter tier"""
    if mode not in PARAMETER_TIERS:
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(PARAMETER_TIERS.keys())}")

    params = PARAMETER_TIERS[mode]

    # Calculate raw combinations
    raw_count = 1
    for values in params.values():
        raw_count *= len(values)

    return {
        'mode': mode,
        'parameters': list(params.keys()),
        'raw_combinations': raw_count,
        'estimated_filtered': int(raw_count * 0.4),  # ~40% pass R:R filter
        'parameter_counts': {k: len(v) for k, v in params.items()},
    }


def resolve_epic(epic_input: str) -> str:
    """Resolve epic shortcut to full epic ID"""
    upper = epic_input.upper()
    if upper in EPIC_SHORTCUTS:
        return EPIC_SHORTCUTS[upper]
    return epic_input  # Assume it's already a full epic ID
