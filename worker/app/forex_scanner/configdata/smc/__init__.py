# configdata/smc/__init__.py
"""
Smart Money Concepts (SMC) Configuration Module
First implementation of the modular config system
"""

# Import all SMC configurations
from .smc_configdata import *

# Module metadata
__version__ = "2.0.0"
__description__ = "Smart Money Concepts configuration for Forex Scanner"
__author__ = "Forex Scanner Team"

# Export key functions and constants
__all__ = [
    # Core SMC Settings
    'SMART_MONEY_ENABLED',
    'SMART_MONEY_READONLY_ENABLED',
    'SMART_MONEY_VERSION',
    
    # Structure Analysis Settings
    'STRUCTURE_SWING_LOOKBACK',
    'STRUCTURE_MIN_SWING_STRENGTH', 
    'STRUCTURE_BOS_CONFIRMATION_PIPS',
    'STRUCTURE_REQUIRE_CLOSE',
    
    # Order Flow Analysis Settings
    'ORDER_FLOW_MIN_OB_SIZE_PIPS',
    'ORDER_FLOW_MIN_FVG_SIZE_PIPS',
    'ORDER_FLOW_DISPLACEMENT_FACTOR',
    
    # Performance Settings
    'SMART_MONEY_MIN_DATA_POINTS',
    'SMART_MONEY_ANALYSIS_TIMEOUT',
    'ORDER_FLOW_MAX_LOOKBACK_BARS',
    
    # Weights and Scoring
    'SMART_MONEY_STRUCTURE_WEIGHT',
    'SMART_MONEY_ORDER_FLOW_WEIGHT',
    'SMART_MONEY_MIN_CONFIDENCE_BOOST',
    'SMART_MONEY_MAX_CONFIDENCE_BOOST',
    
    # Instrument Settings
    'PIP_SIZES',
    
    # Helper Functions
    'get_pip_size_for_epic',
    'get_smart_money_config_summary',
    
    # Strategy Integration
    'SMART_EMA_STRUCTURE_VALIDATION',
    'SMART_MACD_ORDER_FLOW_VALIDATION',
    'COMBINED_SMART_MONEY_ENABLED'
]

# Quick access functions
def is_smc_enabled() -> bool:
    """Check if Smart Money Concepts are enabled"""
    return SMART_MONEY_ENABLED

def get_smc_version() -> str:
    """Get SMC configuration version"""
    return SMART_MONEY_VERSION

def get_performance_settings() -> dict:
    """Get performance-related settings"""
    return {
        'min_data_points': SMART_MONEY_MIN_DATA_POINTS,
        'analysis_timeout': SMART_MONEY_ANALYSIS_TIMEOUT,
        'max_lookback_bars': ORDER_FLOW_MAX_LOOKBACK_BARS,
        'structure_lookback': STRUCTURE_SWING_LOOKBACK
    }

def get_validation_settings() -> dict:
    """Get validation-related settings"""
    return {
        'structure_validation': SMART_MONEY_STRUCTURE_VALIDATION,
        'order_flow_validation': SMART_MONEY_ORDER_FLOW_VALIDATION,
        'structure_weight': SMART_MONEY_STRUCTURE_WEIGHT,
        'order_flow_weight': SMART_MONEY_ORDER_FLOW_WEIGHT,
        'min_score': SMART_MONEY_MIN_SCORE
    }

# Configuration validation
def validate_smc_configuration() -> dict:
    """
    Validate SMC configuration for completeness and correctness
    """
    errors = []
    warnings = []
    
    # Check required boolean settings
    boolean_settings = [
        'SMART_MONEY_ENABLED',
        'SMART_MONEY_READONLY_ENABLED', 
        'STRUCTURE_REQUIRE_CLOSE',
        'SMART_MONEY_STRUCTURE_VALIDATION',
        'SMART_MONEY_ORDER_FLOW_VALIDATION'
    ]
    
    for setting in boolean_settings:
        if setting in globals():
            value = globals()[setting]
            if not isinstance(value, bool):
                warnings.append(f"{setting} should be boolean, got {type(value)}")
    
    # Check numeric ranges
    numeric_ranges = {
        'STRUCTURE_MIN_SWING_STRENGTH': (0.0, 1.0),
        'SMART_MONEY_STRUCTURE_WEIGHT': (0.0, 1.0),
        'SMART_MONEY_ORDER_FLOW_WEIGHT': (0.0, 1.0),
        'SMART_MONEY_MIN_CONFIDENCE_BOOST': (0.0, 1.0),
        'SMART_MONEY_MAX_CONFIDENCE_BOOST': (0.0, 1.0),
        'ORDER_FLOW_DISPLACEMENT_FACTOR': (1.0, 5.0),
        'SMART_MONEY_ANALYSIS_TIMEOUT': (1.0, 30.0)
    }
    
    for setting, (min_val, max_val) in numeric_ranges.items():
        if setting in globals():
            value = globals()[setting]
            if not isinstance(value, (int, float)):
                errors.append(f"{setting} must be numeric")
            elif not (min_val <= value <= max_val):
                warnings.append(f"{setting} = {value} outside recommended range [{min_val}, {max_val}]")
    
    # Check pip sizes dictionary
    if 'PIP_SIZES' in globals():
        pip_sizes = globals()['PIP_SIZES']
        if not isinstance(pip_sizes, dict):
            errors.append("PIP_SIZES must be a dictionary")
        elif len(pip_sizes) == 0:
            warnings.append("PIP_SIZES dictionary is empty")
        else:
            # Check for essential instruments
            essential_instruments = ['EURUSD', 'USDJPY', 'XAUUSD']
            missing_instruments = [inst for inst in essential_instruments if inst not in pip_sizes]
            if missing_instruments:
                warnings.append(f"Missing pip sizes for essential instruments: {missing_instruments}")
    else:
        errors.append("PIP_SIZES dictionary is required")
    
    # Check helper function exists
    if 'get_pip_size_for_epic' not in globals():
        errors.append("get_pip_size_for_epic function is required")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'total_configs': len([k for k in globals() if k.isupper() and not k.startswith('_')])
    }

# Run validation on import (development mode)
if __name__ != "__main__":
    try:
        validation = validate_smc_configuration()
        if not validation['valid']:
            print("❌ SMC Configuration validation failed:")
            for error in validation['errors']:
                print(f"   • {error}")
        if validation['warnings']:
            print("⚠️ SMC Configuration warnings:")
            for warning in validation['warnings']:
                print(f"   • {warning}")
    except Exception as e:
        print(f"⚠️ SMC Configuration validation error: {e}")

print(f"📊 SMC Configuration Module loaded - {len(__all__)} settings available")