# configdata/strategies/__init__.py
"""
Strategy-specific configuration modules

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategy configs have been archived to forex_scanner/archive/legacy_configs/
"""

# Import only active strategy configuration
from .config_smc_simple import *

# Import main config flags for strategy enablement
try:
    from forex_scanner.config import SMC_SIMPLE_STRATEGY
except ImportError:
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import SMC_SIMPLE_STRATEGY
    except ImportError:
        SMC_SIMPLE_STRATEGY = True
        print("⚠️ Could not import SMC_SIMPLE_STRATEGY from main config, defaulting to True")

# Define what gets exported when using "from strategies import *"
__all__ = [
    # SMC Simple Strategy - the only active strategy
    'SMC_SIMPLE_STRATEGY',
]

# Module metadata
__version__ = "2.0.0"
__description__ = "Strategy configuration - SMC Simple only (legacy strategies archived)"

def get_strategies_summary() -> dict:
    """Get a summary of all loaded strategy configurations"""
    return {
        'loaded_strategies': ['smc_simple'],
        'smc_simple_enabled': globals().get('SMC_SIMPLE_STRATEGY', True),
        'note': 'Legacy strategies archived - see forex_scanner/archive/'
    }

print(f"📊 Strategy configs loaded - SMC Simple: {globals().get('SMC_SIMPLE_STRATEGY', True)} (legacy strategies archived)")
