# configdata/config_momentum_bias.py
"""
Momentum Bias Strategy Configuration
Simple configuration module for the Momentum Bias strategy settings
"""

# =============================================================================
# MOMENTUM BIAS STRATEGY CONFIGURATION SETTINGS
# =============================================================================

# Enable/disable the momentum bias strategy
MOMENTUM_BIAS_STRATEGY = True

# Core Momentum Bias Parameters (from Pine Script)
MOMENTUM_BIAS_MOMENTUM_LENGTH = 10        # Momentum calculation period
MOMENTUM_BIAS_BIAS_LENGTH = 5             # Bias sum period
MOMENTUM_BIAS_SMOOTH_LENGTH = 10          # Smoothing period for HMA
MOMENTUM_BIAS_IMPULSE_BOUNDARY_LENGTH = 30  # Boundary calculation period
MOMENTUM_BIAS_STD_DEV_MULTIPLIER = 3.0    # Standard deviation multiplier for boundary
MOMENTUM_BIAS_SMOOTH_INDICATOR = True     # Enable smoothing (uses HMA approximation)

# Confidence Scoring Configuration
MOMENTUM_BIAS_BASE_CONFIDENCE = 0.65      # Base confidence score (65%)
MOMENTUM_BIAS_BOUNDARY_WEIGHT = 0.25      # Weight for boundary strength factor
MOMENTUM_BIAS_SEPARATION_WEIGHT = 0.15    # Weight for bias separation factor

# Risk Management Settings
MOMENTUM_BIAS_DEFAULT_RISK_REWARD = 2.0   # Default risk:reward ratio (2:1)
MOMENTUM_BIAS_STOP_LOSS_MULTIPLIER = 2.0  # Stop loss = 2x spread
MOMENTUM_BIAS_TAKE_PROFIT_MULTIPLIER = 4.0 # Take profit = 4x spread (2:1 RR)

# Signal Quality Filters
MOMENTUM_BIAS_MIN_BOUNDARY_STRENGTH = 1.1  # Minimum boundary strength ratio
MOMENTUM_BIAS_MIN_BIAS_SEPARATION = 0.05   # Minimum bias separation required
MOMENTUM_BIAS_MIN_DOMINANCE_RATIO = 1.2    # Minimum dominance ratio

# Advanced Configuration
MOMENTUM_BIAS_ENABLE_VOLUME_FILTER = False  # Enable volume confirmation
MOMENTUM_BIAS_VOLUME_THRESHOLD = 1.2        # Volume must be 20% above average
MOMENTUM_BIAS_ENABLE_TREND_FILTER = False   # Enable additional trend filtering

# Strategy Integration Settings
MOMENTUM_BIAS_STRATEGY_WEIGHT = 0.15        # Weight in combined strategy mode
MOMENTUM_BIAS_ALLOW_COMBINED = True         # Allow in combined strategies
MOMENTUM_BIAS_PRIORITY_LEVEL = 3            # Priority level (1=highest, 5=lowest)

# Backtesting and Performance
MOMENTUM_BIAS_ENABLE_BACKTESTING = True     # Enable strategy in backtests
MOMENTUM_BIAS_MIN_DATA_PERIODS = 100        # Minimum data periods required
MOMENTUM_BIAS_ENABLE_PERFORMANCE_TRACKING = True  # Track strategy performance

# Logging and Debug Settings
MOMENTUM_BIAS_DEBUG_LOGGING = True          # Enable detailed debug logging
MOMENTUM_BIAS_SAVE_INDICATOR_DATA = False   # Save indicator calculations to files
MOMENTUM_BIAS_LOG_SIGNAL_CONDITIONS = True  # Log detailed signal conditions

