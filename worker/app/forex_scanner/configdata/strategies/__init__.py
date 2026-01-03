# configdata/strategies/__init__.py
"""
Strategy-specific configuration modules

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategy configs have been archived to forex_scanner/archive/legacy_configs/
"""

# Import only active strategy configuration
from .config_smc_simple import *

# Import main config flags for strategy enablement
# Use importlib to handle circular import scenarios
import importlib
import sys
import os

SMC_SIMPLE_STRATEGY = True  # Default

# Try multiple import paths to handle different execution contexts
_config_loaded = False

# Method 1: Direct importlib (most reliable)
if not _config_loaded:
    try:
        _config_module = importlib.import_module('forex_scanner.config')
        SMC_SIMPLE_STRATEGY = getattr(_config_module, 'SMC_SIMPLE_STRATEGY', True)
        _config_loaded = True
    except (ImportError, ModuleNotFoundError):
        pass

# Method 2: Relative path import
if not _config_loaded:
    try:
        # Add parent directory to path if not present
        _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent_dir not in sys.path:
            sys.path.insert(0, _parent_dir)
        _config_module = importlib.import_module('config')
        SMC_SIMPLE_STRATEGY = getattr(_config_module, 'SMC_SIMPLE_STRATEGY', True)
        _config_loaded = True
    except (ImportError, ModuleNotFoundError):
        pass

# Clean up temporary variables
del _config_loaded, importlib

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
