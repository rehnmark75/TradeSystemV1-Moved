# configdata/config_zerolag_strategy.py
"""
Zero Lag EMA Strategy Configuration
Configuration module for the Zero Lag EMA strategy settings
"""

# =============================================================================
# ZERO LAG EMA STRATEGY CONFIGURATION SETTINGS  
# =============================================================================

# Enable/disable the zero lag ema strategy
ZERO_LAG_STRATEGY = True

# Core Zero Lag EMA Parameters (REQUIRED by ZeroLagStrategy.__init__)
ZERO_LAG_LENGTH = 70                    # Zero lag EMA length (Pine Script default for exact match)
ZERO_LAG_BAND_MULT = 1.2               # Band multiplier for volatility bands (Pine Script default)
ZERO_LAG_MIN_CONFIDENCE = 0.75         # Minimum confidence threshold (CRITICAL) - Higher for selectivity
ZERO_LAG_MIN_COMPONENT_SCORE = 2       # Minimum component validation score (out of 7 max)

# Additional Configuration (for completeness)
ZERO_LAG_BASE_CONFIDENCE = 0.70        # Base confidence score
ZERO_LAG_MAX_CONFIDENCE = 0.95         # Maximum confidence cap
ZERO_LAG_TREND_WEIGHT = 0.20           # Weight for trend consistency factor
ZERO_LAG_VOLATILITY_WEIGHT = 0.15      # Weight for volatility factor
ZERO_LAG_MOMENTUM_WEIGHT = 0.10        # Weight for momentum factor

# Strategy Integration Settings
ZERO_LAG_STRATEGY_WEIGHT = 0.15         # Weight in combined strategy mode
ZERO_LAG_ALLOW_COMBINED = True          # Allow in combined strategies
ZERO_LAG_PRIORITY_LEVEL = 3             # Priority level (1=highest, 5=lowest)

# Performance Settings
ZERO_LAG_ENABLE_BACKTESTING = True      # Enable strategy in backtests
ZERO_LAG_MIN_DATA_PERIODS = 150         # Minimum data periods required
ZERO_LAG_ENABLE_PERFORMANCE_TRACKING = True  # Track strategy performance

# Debug Settings
ZERO_LAG_DEBUG_LOGGING = True           # Enable detailed debug logging