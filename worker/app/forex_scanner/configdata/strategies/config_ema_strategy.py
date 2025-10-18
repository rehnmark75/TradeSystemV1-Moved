# configdata/strategies/config_ema_strategy.py
"""
Multi-Supertrend Strategy Configuration (EMA Replacement)
Configuration module for Multi-Supertrend trend-following strategy.
Uses 3 Supertrend indicators with different parameters for confluence-based signals.

NOTE: File kept as 'config_ema_strategy.py' for backward compatibility with existing references.
Internal logic completely replaced with Supertrend-based trend following.
"""

# =============================================================================
# MULTI-SUPERTREND STRATEGY CORE CONFIGURATION
# =============================================================================

# Strategy Mode Selection
USE_SUPERTREND_MODE = True                 # True = Multi-Supertrend, False = Legacy EMA (for A/B testing)
SIMPLE_EMA_STRATEGY = True                 # Kept for compatibility - ignored when USE_SUPERTREND_MODE = True
STRICT_EMA_ALIGNMENT = True                # Legacy - not used
REQUIRE_EMA_SEPARATION = False             # Legacy - not used
STRATEGY_WEIGHT_EMA = 0.5                  # Strategy weight in combined mode
STRATEGY_WEIGHT_ZERO_LAG = 0.0             # Legacy - not used

# =============================================================================
# MULTI-SUPERTREND CONFIGURATION SYSTEM
# =============================================================================

# Enable/disable dynamic Supertrend configuration (replaces ENABLE_DYNAMIC_EMA_CONFIG)
ENABLE_DYNAMIC_SUPERTREND_CONFIG = False   # Use fixed 'default' config for now

# Multi-Supertrend Strategy Configurations
# Each config has 3 Supertrend indicators: Fast, Medium, Slow
SUPERTREND_STRATEGY_CONFIG = {
    'default': {
        # Fast Supertrend - Quick trend detection, early signals
        'fast_period': 7,
        'fast_multiplier': 1.5,
        # Medium Supertrend - Balanced confirmation
        'medium_period': 14,
        'medium_multiplier': 2.0,
        # Slow Supertrend - Major trend filter
        'slow_period': 21,
        'slow_multiplier': 3.0,
        # ATR settings
        'atr_period': 14,
        # Description
        'description': 'Balanced multi-Supertrend for trending markets',
        'best_for': ['trending', 'medium_volatility'],
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'fast_period': 10,
        'fast_multiplier': 2.0,
        'medium_period': 14,
        'medium_multiplier': 2.5,
        'slow_period': 21,
        'slow_multiplier': 3.5,
        'atr_period': 14,
        'description': 'Conservative - wider Supertrends, fewer false signals',
        'best_for': ['strong_trends', 'low_volatility'],
        'best_market_regime': 'trending',
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 25.0
    },
    'aggressive': {
        'fast_period': 5,
        'fast_multiplier': 1.0,
        'medium_period': 10,
        'medium_multiplier': 1.5,
        'slow_period': 14,
        'slow_multiplier': 2.0,
        'atr_period': 10,
        'description': 'Aggressive - tight Supertrends, more signals',
        'best_for': ['breakouts', 'high_volatility'],
        'best_market_regime': 'breakout',
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 100.0
    },
    'scalping': {
        'fast_period': 3,
        'fast_multiplier': 0.8,
        'medium_period': 7,
        'medium_multiplier': 1.2,
        'slow_period': 10,
        'slow_multiplier': 1.5,
        'atr_period': 7,
        'description': 'Ultra-fast Supertrends for scalping',
        'best_for': ['ranging_markets', 'high_frequency'],
        'best_market_regime': 'ranging',
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 80.0
    },
    'swing': {
        'fast_period': 14,
        'fast_multiplier': 2.0,
        'medium_period': 21,
        'medium_multiplier': 2.5,
        'slow_period': 28,
        'slow_multiplier': 3.5,
        'atr_period': 21,
        'description': 'Slow Supertrends for swing trading',
        'best_for': ['strong_trends', 'position_trading'],
        'best_market_regime': 'trending',
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 30.0
    },
    'youtube_triple_st': {
        # Based on YouTube strategy: https://www.youtube.com/watch?v=hy9Nce__k78
        # SuperTrend 1: ATR 12, Multiplier 3 (Slow - long-term trend)
        'slow_period': 12,
        'slow_multiplier': 3.0,
        # SuperTrend 2: ATR 10, Multiplier 1 (Fast - quick signals)
        'fast_period': 10,
        'fast_multiplier': 1.0,
        # SuperTrend 3: ATR 11, Multiplier 2 (Medium - balanced)
        'medium_period': 11,
        'medium_multiplier': 2.0,
        # ATR settings
        'atr_period': 12,  # Use longest ATR period for stability
        # Description
        'description': 'YouTube Triple SuperTrend + EMA 200 Strategy',
        'best_for': ['trending', 'all_volatility'],
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],  # Avoid first 30-60min after market open
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 100.0,
        # Strategy notes
        'entry_rules': 'LONG: Price > EMA200 + All 3 SuperTrends green | SHORT: Price < EMA200 + All 3 SuperTrends red',
        'stop_loss': 'Closest SuperTrend dot',
        'take_profit': '1:1 to 1.5:1 risk/reward',
        'timeframe': '15m',
        'avoid_first_30min': True  # No trading first 30-60min after market open
    }
}

# Active Supertrend configuration when not using dynamic mode
# Testing different configs to find optimal performance
ACTIVE_SUPERTREND_CONFIG = 'youtube_triple_st'  # Using YouTube triple SuperTrend strategy

# Supertrend Signal Requirements
# ✅ YOUTUBE STRATEGY: Requires ALL 3 SuperTrends to agree (100% confluence)
# YouTube video strategy requires full alignment of all 3 SuperTrends + EMA 200
# This is more conservative but provides higher quality signals
# 2 = At least 2/3 must agree (66% confluence - previous optimal)
# 3 = All Supertrends must agree (100% confluence - YouTube strategy requirement)
SUPERTREND_MIN_CONFLUENCE = 3              # Require ALL 3 SuperTrends to agree (YouTube strategy)
SUPERTREND_ALLOW_PARTIAL = False           # If True, allow 2/3 confluence with reduced confidence

# =============================================================================
# LUXALGO-INSPIRED ENHANCEMENTS (Signal Reduction & Quality Improvement)
# =============================================================================

# ✅ OPTION 2: Enhanced Slow SuperTrend Stability Filter
# Increases requirement from 5 bars to 12+ bars of stability
# Prevents signals in markets that are starting to chop
SUPERTREND_STABILITY_BARS = 8              # Minimum bars of slow ST stability (BALANCED: 8 bars)
                                           # Higher = fewer signals in choppy markets
                                           # Recommended: 8-10 for balanced, 12-15 for conservative

# ✅ OPTION 3: Performance-Based Filter (LuxAlgo inspired)
# Only signals when recent SuperTrend performance is positive
# Performance = Did price move with SuperTrend direction?
SUPERTREND_PERFORMANCE_FILTER = True       # Enable performance-based filtering
SUPERTREND_PERFORMANCE_THRESHOLD = -0.00005  # Minimum performance (RELAXED: allow slight negative)
SUPERTREND_PERFORMANCE_ALPHA = 0.15        # Smoothing factor (INCREASED: more reactive)
                                           # Higher alpha = more reactive to recent performance
                                           # Lower alpha = more stable, considers longer history

# ✅ OPTION 5: Trend Strength Filter
# Filters out signals when SuperTrends are too close (choppy market)
# Wide separation = strong trend, Narrow separation = ranging/choppy
SUPERTREND_TREND_STRENGTH_FILTER = True    # Enable trend strength filtering
SUPERTREND_MIN_TREND_STRENGTH = 0.15       # Minimum separation as % of price (RELAXED: 0.15%)
                                           # EUR/USD @ 1.1000: 0.15% = 16.5 pips separation
                                           # Recommended: 0.15-0.25% for balanced signals

# ✅ OPTION 4: Adaptive Factor Optimization (Advanced - Future Enhancement)
# K-means clustering to find optimal ATR multipliers (LuxAlgo full implementation)
# Currently disabled - requires significant computational resources
SUPERTREND_ADAPTIVE_CLUSTERING = False     # Enable adaptive factor optimization (experimental)
SUPERTREND_CLUSTER_MIN_FACTOR = 1.0        # Minimum ATR multiplier to test
SUPERTREND_CLUSTER_MAX_FACTOR = 5.0        # Maximum ATR multiplier to test
SUPERTREND_CLUSTER_STEP = 0.5              # Step size between factors
SUPERTREND_CLUSTER_LOOKBACK = 500          # Bars for performance evaluation
SUPERTREND_CLUSTER_CHOICE = 'best'         # Which cluster to use (best/average/worst)

# Multi-Timeframe Supertrend Confirmation (4H Filter)
# ⚠️ IMPORTANT: Re-enabled as trend filter for higher quality signals
# RSI filter remains disabled to allow counter-trend signals
SUPERTREND_4H_FILTER_ENABLED = True        # ✅ ENABLED: 4H trend alignment check
SUPERTREND_4H_USE_MEDIUM = True            # Use Medium Supertrend on 4H (more reliable than fast)
SUPERTREND_4H_CONFLUENCE_BONUS = 0.10      # Bonus when 4H trend aligns with 15m
SUPERTREND_4H_PENALTY = 0.15               # Penalty when 4H trend opposes (advisory, not blocking)

# Confidence Scoring for Supertrend Signals
SUPERTREND_CONFIDENCE_BASE = 0.60          # Base confidence when all 3 agree
SUPERTREND_CONFIDENCE_FULL_CONFLUENCE = 0.35  # Bonus for 100% confluence (3/3)
SUPERTREND_CONFIDENCE_PARTIAL_CONFLUENCE = 0.15  # Bonus for partial (2/3) if allowed
SUPERTREND_CONFIDENCE_VOLUME = 0.05        # Bonus for volume confirmation
SUPERTREND_CONFIDENCE_4H_ALIGNMENT = 0.10  # Bonus for 4H trend alignment

# ✅ YOUTUBE STRATEGY: Risk/Reward Settings
# YouTube video recommends 1:1 to 1.5:1 risk/reward ratio
# Stop loss: Closest SuperTrend dot
# Take profit: 1x to 1.5x the risk distance
YOUTUBE_STRATEGY_MIN_RR = 1.0              # Minimum risk/reward ratio (1:1)
YOUTUBE_STRATEGY_MAX_RR = 1.5              # Maximum risk/reward ratio (1:1.5)
YOUTUBE_STRATEGY_DEFAULT_RR = 1.5          # Default to 1.5:1 for optimal results

# =============================================================================
# DYNAMIC EMA CONFIGURATION SYSTEM (LEGACY - KEPT FOR COMPATIBILITY)
# =============================================================================

# Enable/disable dynamic EMA configuration (legacy)
ENABLE_DYNAMIC_EMA_CONFIG = False         # Disabled - using Supertrend mode

# Enhanced Dynamic EMA Strategy Configurations with market conditions
EMA_STRATEGY_CONFIG = {
    'default': {
        'short': 21, 'long': 50, 'trend': 200,
        'description': 'Less choppy configuration for cleaner signals - 21/50/200 EMAs',
        'best_for': ['trending', 'medium_volatility'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'short': 20, 'long': 50, 'trend': 200,
        'description': 'Conservative approach for low volatility trending markets',
        'best_for': ['strong_trends', 'low_volatility'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 25.0
    },
    'aggressive': {
        'short': 5, 'long': 13, 'trend': 50,
        'description': 'Fast-reacting configuration for high volatility breakouts',
        'best_for': ['breakouts', 'high_volatility'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'weak',
        'best_market_regime': 'breakout',
        'best_session': ['overlap', 'new_york'],
        'preferred_pairs': ['CS.D.GBPUSD.MINI.IP', 'CS.D.EURUSD.CEEM.IP'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 100.0
    },
    'scalping': {
        'short': 3, 'long': 8, 'trend': 21,
        'description': 'Ultra-fast configuration for scalping strategies',
        'best_for': ['ranging_markets', 'high_frequency'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'weak',
        'best_market_regime': 'ranging',
        'best_session': ['london', 'new_york', 'overlap'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 80.0
    },
    'swing': {
        'short': 25, 'long': 55, 'trend': 200,
        'description': 'Slow and steady for swing trading',
        'best_for': ['strong_trends', 'position_trading'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 30.0
    },
    'news_safe': {
        'short': 15, 'long': 30, 'trend': 200,
        'description': 'Safer configuration during news events',
        'best_for': ['news_events', 'high_volatility'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'breakout',
        'best_session': ['asian', 'overlap'],
        'preferred_pairs': ['CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 100.0
    },
    'crypto': {
        'short': 7, 'long': 25, 'trend': 99,
        'description': 'Adapted for crypto-like high volatility markets',
        'best_for': ['high_volatility', 'breakouts'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'medium',
        'best_market_regime': 'breakout',
        'best_session': ['overlap', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP'],  # Most liquid
        'min_pip_volatility': 25.0,
        'max_pip_volatility': 150.0
    }
}

# Active configuration when not using dynamic mode
ACTIVE_EMA_CONFIG = 'default'

# Scanner and backtest dynamic EMA usage
SCANNER_USE_DYNAMIC_EMA = True            # Use dynamic EMA in scanner
BACKTEST_USE_DYNAMIC_EMA = False          # Use dynamic EMA in backtesting (experimental)

# =============================================================================
# EMA200 DISTANCE VALIDATION
# =============================================================================

# Enable minimum distance from EMA200 requirement
EMA200_MIN_DISTANCE_ENABLED = True

# Minimum distance from EMA200 in pips for each pair
# These values prevent false signals when price hovers near EMA200
EMA200_MIN_DISTANCE_PIPS = {
    'EURUSD': 4.0,    # EUR/USD - most liquid, 5 pips minimum
    'GBPUSD': 7.0,    # GBP/USD - more volatile, 8 pips minimum
    'AUDUSD': 6.0,    # AUD/USD - commodity currency, 6 pips
    'NZDUSD': 7.0,    # NZD/USD - higher volatility, 7 pips
    'USDCAD': 6.0,    # USD/CAD - oil correlation, 6 pips
    'USDCHF': 5.0,    # USD/CHF - safe haven, 5 pips
    'USDJPY': 5.0,    # USD/JPY - different scale, 50 pips (0.50)
    'EURJPY': 5.0,    # EUR/JPY - cross pair, 60 pips
    'GBPJPY': 5.0,    # GBP/JPY - very volatile, 80 pips
    'AUDJPY': 5.0,    # AUD/JPY - carry trade pair, 70 pips
    'EURGBP': 5.0,    # EUR/GBP - relatively stable, 5 pips
    'EURAUD': 8.0,    # EUR/AUD - commodity impact, 8 pips
    'GBPAUD': 10.0,   # GBP/AUD - volatile cross, 10 pips
    'DEFAULT': 5.0    # Default for unknown pairs
}

# Grace period: Allow signals slightly closer than minimum during strong momentum
# Set to 0.8 means 80% of minimum distance is acceptable with strong momentum
EMA200_DISTANCE_MOMENTUM_MULTIPLIER = 0.8

# =============================================================================
# TWO-POLE OSCILLATOR - DISABLED (REPLACED WITH RSI)
# =============================================================================

# Two-Pole Oscillator Configuration - DISABLED in favor of standard RSI
TWO_POLE_OSCILLATOR_ENABLED = False      # ❌ DISABLED: Replaced with RSI filters
TWO_POLE_FILTER_LENGTH = 20              # (Legacy - not used)
TWO_POLE_SMA_LENGTH = 25                 # (Legacy - not used)
TWO_POLE_SIGNAL_DELAY = 4                # (Legacy - not used)
TWO_POLE_VALIDATION_ENABLED = False      # ❌ DISABLED: Using RSI validation instead
TWO_POLE_MIN_STRENGTH = 0.1              # (Legacy - not used)
TWO_POLE_ZONE_FILTER_ENABLED = False     # ❌ DISABLED

# Two-Pole Oscillator Confidence Scoring - LEGACY
TWO_POLE_CONFIDENCE_WEIGHT = 0.0         # Set to 0 (replaced by EMA_RSI_CONFIDENCE_WEIGHT)
TWO_POLE_ZONE_BONUS = 0.0                # (Legacy - not used)
TWO_POLE_STRENGTH_MULTIPLIER = 0.0       # (Legacy - not used)

# Two-Pole Multi-Timeframe Validation - DISABLED
TWO_POLE_MTF_VALIDATION = False          # ❌ DISABLED: Using 4H EMA + RSI validation instead
TWO_POLE_MTF_TIMEFRAME = '1h'            # (Legacy - not used)

# =============================================================================
# EMA STRATEGY TRIGGER MODE (TRADINGVIEW PROFESSIONAL)
# =============================================================================

# Primary trigger selection
EMA_TRIGGER_MODE = 'crossover'           # Options: 'bounce', 'crossover', 'hybrid'

# Bounce strategy settings (from TradingView "EMA Bounce Scalping" - 180+ likes)
EMA_BOUNCE_ENABLED = True
EMA_BOUNCE_DISTANCE_PCT = 0.1            # Within 0.1% of EMA (~10 pips EURUSD)
EMA_BOUNCE_REQUIRE_REJECTION = True      # Require rejection candle pattern
EMA_BOUNCE_WICK_RATIO = 1.5              # Rejection wick must be 1.5x opposite wick
EMA_BOUNCE_MIN_CANDLE_BODY = 0.3         # Minimum body ratio (avoid dojis)
EMA_BOUNCE_MIN_ADX = 15                  # Lower ADX for bounces (pullbacks in medium trends)

# Crossover strategy (kept as fallback option)
EMA_CROSSOVER_ENABLED = True             # Keep for comparison/hybrid mode
EMA_CROSSOVER_REQUIRE_STRONG_TREND = True

# =============================================================================
# RSI MOMENTUM FILTER (REPLACES TWO-POLE)
# =============================================================================

# Enable RSI-based momentum filtering (TradingView standard)
EMA_USE_RSI_FILTER = True

# 15m RSI settings (CORRECTED: Trend confirmation with buffer zone)
EMA_RSI_LENGTH = 14                      # Standard RSI period
EMA_RSI_BULL_MIN = 40                    # RSI > 40 confirms bullish momentum (10-point buffer)
EMA_RSI_BULL_MAX = 100                   # No upper limit (not using as overbought)
EMA_RSI_BEAR_MIN = 0                     # No lower limit (not using as oversold)
EMA_RSI_BEAR_MAX = 60                    # RSI < 60 confirms bearish momentum (10-point buffer)

# 4H Multi-Timeframe Filters (replaces 1H Two-Pole)
# ⚠️ BALANCED APPROACH: Trend filter enabled, RSI filter disabled
# Trend filter helps with direction, RSI filter was too restrictive
EMA_4H_TREND_FILTER_ENABLED = True       # ✅ ENABLED: 4H EMA trend alignment check
EMA_4H_RSI_FILTER_ENABLED = False        # ❌ DISABLED: Too restrictive, blocks valid counter-trend signals
EMA_4H_RSI_BULL_MIN = 50                 # 4H RSI > 50 for bullish bias (not used when filter disabled)
EMA_4H_RSI_BEAR_MAX = 50                 # 4H RSI < 50 for bearish bias (not used when filter disabled)

# RSI confidence scoring (replaces Two-Pole confidence weight)
EMA_RSI_CONFIDENCE_WEIGHT = 0.15         # Weight in confidence calculation
EMA_4H_RSI_BONUS = 0.10                  # Bonus when 4H RSI + trend aligned

# Confidence calculation adjustments
EMA_BOUNCE_CONFIDENCE_BASE = 0.60        # Higher base for quality bounce signals
EMA_CROSSOVER_CONFIDENCE_BASE = 0.50     # Lower base for crossover signals


# =============================================================================
# MACD MOMENTUM FILTER (EMA INTEGRATION)
# =============================================================================

# MACD Momentum Filter Configuration (momentum confirmation for EMA signals)
MACD_MOMENTUM_FILTER_ENABLED = True            # Enable MACD histogram momentum validation
MACD_HISTOGRAM_LOOKBACK = 3                   # Number of periods to analyze for histogram trend (INCREASED FOR BETTER TREND DETECTION)
MACD_MIN_SLOPE_THRESHOLD = 0.00005              # Minimum slope threshold to avoid noise (MORE SENSITIVE)

# MACD Momentum Filter Rules:
# - BULL signals rejected if MACD histogram is descending (momentum against signal)  
# - BEAR signals rejected if MACD histogram is rising (momentum against signal)
# - NEUTRAL trends allow all signals

# MACD Momentum Filter Settings - STRICT FOR RELIABLE MOMENTUM VALIDATION
MACD_MOMENTUM_VALIDATION_ENABLED = True        # Use MACD for signal validation
MACD_TREND_SENSITIVITY = 'normal'          # Sensitivity: 'strict', 'normal', 'permissive', 'neutral_bias'
MACD_VALIDATION_MODE = 'strict_blocking'      # Mode: 'strict_blocking', 'slope_aware', 'neutral_friendly' (CHANGED TO STRICT)

# MACD trend sensitivity mappings (affects lookback and threshold)
MACD_SENSITIVITY_SETTINGS = {
    'strict': {
        'lookback': 5,                          # More periods for stable trend detection
        'min_slope': 0.0002,                    # Higher threshold, less sensitive
        'description': 'Conservative - requires clear momentum direction'
    },
    'normal': {
        'lookback': 3,                          # Standard periods for trend detection
        'min_slope': 0.00005,                    # More sensitive threshold (IMPROVED)
        'description': 'Balanced - moderate sensitivity to momentum changes'
    },
    'permissive': {
        'lookback': 2,                          # Fewer periods, more reactive
        'min_slope': 0.00005,                   # Lower threshold, more sensitive
        'description': 'Permissive - allows signals with weak MACD trends'
    },
    'neutral_bias': {
        'lookback': 3,                          # Standard periods
        'min_slope': 0.00002,                   # Very low threshold - most trends become neutral
        'description': 'Neutral bias - most MACD states allow signals'
    },
    'ultra_sensitive': {
        'lookback': 2,                          # Very reactive
        'min_slope': 0.000005,                  # Ultra-low threshold - detect tiny movements  
        'description': 'Ultra sensitive - blocks even tiny opposing MACD movements'
    }
}

# MACD validation modes determine how strictly MACD momentum is enforced
MACD_VALIDATION_MODES = {
    'strict_blocking': {
        'description': 'Block signals when MACD momentum opposes signal direction',
        'allow_neutral': True,                  # NEUTRAL MACD allows signals
        'block_opposite': True                  # Block when MACD opposes signal
    },
    'slope_aware': {
        'description': 'Consider MACD slope but allow weak opposing momentum',
        'allow_neutral': True,                  # NEUTRAL MACD allows signals  
        'block_opposite': 'strong_only',        # Only block strong opposing momentum
        'weak_threshold': 0.0002                # Threshold for "strong" opposing momentum
    },
    'neutral_friendly': {
        'description': 'Only block very strong opposing MACD momentum',
        'allow_neutral': True,                  # NEUTRAL MACD allows signals
        'block_opposite': 'very_strong_only',   # Only block very strong opposing momentum  
        'strong_threshold': 0.0005              # Threshold for "very strong" opposing momentum
    }
}

# MACD Filter Debug and Logging
MACD_MOMENTUM_DEBUG_LOGGING = True             # Enable detailed MACD momentum logging

# =============================================================================
# OPTIONAL OVEREXTENSION FILTERS (PRESERVE SIGNAL FREQUENCY)
# =============================================================================

# Enable/disable individual overextension filters
STOCHASTIC_OVEREXTENSION_ENABLED = True      # Optional Stochastic extreme overextension filter
WILLIAMS_R_OVEREXTENSION_ENABLED = True      # Optional Williams %R extreme overextension filter
RSI_EXTREME_OVEREXTENSION_ENABLED = True     # Optional RSI extreme overextension filter
COMPOSITE_OVEREXTENSION_ENABLED = True       # Optional composite overextension scoring

# PRODUCTION: Balanced thresholds to catch overextension while preserving signals
STOCHASTIC_EXTREME_OVERBOUGHT = 80            # Standard overbought threshold
STOCHASTIC_EXTREME_OVERSOLD = 20              # Standard oversold threshold
WILLIAMS_R_EXTREME_OVERBOUGHT = -20           # Standard overbought threshold
WILLIAMS_R_EXTREME_OVERSOLD = -80             # Standard oversold threshold
RSI_EXTREME_OVERBOUGHT = 80                   # Slightly more permissive than typical 70
RSI_EXTREME_OVERSOLD = 20                     # Slightly more permissive than typical 30

# Overextension filter behavior settings - PRODUCTION
OVEREXTENSION_MODE = 'confidence_penalty'     # Use penalty mode to preserve signal frequency
OVEREXTENSION_CONFIDENCE_PENALTY = 0.1        # 10% confidence reduction for overextended signals
OVEREXTENSION_COMPOSITE_THRESHOLD = 2         # Require 2+ indicators for consensus

# Overextension filter weights for composite scoring
OVEREXTENSION_WEIGHTS = {
    'stochastic': 0.35,                       # Weight for Stochastic in composite score
    'williams_r': 0.35,                       # Weight for Williams %R in composite score
    'rsi': 0.30                               # Weight for RSI in composite score
}

# Debug and monitoring settings
OVEREXTENSION_DEBUG_LOGGING = True            # Enable detailed overextension filter logging
OVEREXTENSION_TRACK_FILTERED_SIGNALS = True   # Track signals that would be filtered

# =============================================================================
# EMA SAFETY FILTER SETTINGS
# =============================================================================

# EMA-specific safety filter settings (used in SAFETY_FILTER_PRESETS)
ENABLE_EMA200_CONTRADICTION_FILTER = True     # Enable EMA200 trend contradiction filter
ENABLE_EMA_STACK_CONTRADICTION_FILTER = True  # Enable EMA stack contradiction filter
EMA200_MINIMUM_MARGIN = 0.002                 # Minimum margin from EMA200 (0.2%)

# =============================================================================
# ADDITIONAL EMA STRATEGY SETTINGS
# =============================================================================

# Standard EMA periods for strategy
EMA_PERIODS = [21, 50, 200]                   # Standard EMA periods used in strategy

# EMA validation settings
REQUIRE_PRICE_EMA200_CONFIRMATION = True      # Require price above/below EMA 200
STRICT_EMA_ALIGNMENT = False                  # Allow more flexible EMA requirements  
REQUIRE_EMA_SEPARATION = False                # Enable/disable EMA separation checks

# Minimum EMA separation in pips (if REQUIRE_EMA_SEPARATION is enabled)
MIN_EMA_SEPARATION_PIPS = {
    'EURUSD': 3.0,    # EUR/USD minimum separation
    'GBPUSD': 5.0,    # GBP/USD minimum separation
    'AUDUSD': 4.0,    # AUD/USD minimum separation  
    'NZDUSD': 5.0,    # NZD/USD minimum separation
    'USDCAD': 4.0,    # USD/CAD minimum separation
    'USDCHF': 3.0,    # USD/CHF minimum separation
    'USDJPY': 3.0,    # USD/JPY minimum separation (30 pips)
    'EURJPY': 4.0,    # EUR/JPY minimum separation (40 pips)
    'GBPJPY': 6.0,    # GBP/JPY minimum separation (60 pips)
    'AUDJPY': 5.0,    # AUD/JPY minimum separation (50 pips)
    'EURGBP': 3.0,    # EUR/GBP minimum separation
    'DEFAULT': 4.0    # Default for unknown pairs
}

# =============================================================================
# EMA STRATEGY INTEGRATION SETTINGS
# =============================================================================

# Strategy Integration Settings
EMA_STRATEGY_WEIGHT = 0.15                    # Weight in combined strategy mode
EMA_ALLOW_COMBINED = True                     # Allow in combined strategies
EMA_PRIORITY_LEVEL = 1                        # Priority level (1=highest, 5=lowest)

# Performance Settings
EMA_ENABLE_BACKTESTING = True                 # Enable strategy in backtests
EMA_MIN_DATA_PERIODS = 250                    # Minimum data periods required (200 + buffer)
EMA_ENABLE_PERFORMANCE_TRACKING = True        # Track strategy performance

# Debug Settings
EMA_DEBUG_LOGGING = True                      # Enable detailed debug logging

# =============================================================================
# ENHANCED EMA BREAKOUT VALIDATION SYSTEM (FALSE BREAKOUT REDUCTION)
# =============================================================================

# Main switch for enhanced validation system
EMA_ENHANCED_VALIDATION = True               # Enable/disable enhanced breakout validation (FOREX OPTIMIZED)

# Multi-candle confirmation settings (MORE PERMISSIVE FOR FOREX)
EMA_CONFIRMATION_CANDLES = 1                 # Number of candles required to confirm breakout (1 for forex speed)
EMA_REQUIRE_PROGRESSIVE_MOVEMENT = False     # Don't require progressive movement (too restrictive)

# Volume-based validation settings (FOREX FRIENDLY)
EMA_VOLUME_SPIKE_THRESHOLD = 1.2             # Volume spike multiplier (1.2 = 20% above average - more realistic)
EMA_VOLUME_LOW_THRESHOLD = 0.6               # Low volume threshold (0.6 = 60% of average)

# Pullback and retest settings (DISABLED FOR FOREX SPEED)
EMA_REQUIRE_PULLBACK = False                 # Don't require pullback (too slow for forex)
EMA_PULLBACK_TOLERANCE_PIPS = 0.5            # Tolerance for pullback detection

# Market condition filtering (MORE PERMISSIVE)
EMA_STRONG_TREND_THRESHOLD = 0.002           # EMA separation threshold for strong trending (0.2% - higher)
EMA_MODERATE_TREND_THRESHOLD = 0.0008        # EMA separation threshold for moderate trending (0.08%)
EMA_RANGING_THRESHOLD = 0.0003               # EMA separation threshold for ranging market (0.03%)

# Volatility-based filtering (MORE PERMISSIVE)
EMA_HIGH_VOLATILITY_THRESHOLD = 3.0          # High volatility ratio threshold (higher)
EMA_LOW_VOLATILITY_THRESHOLD = 0.3           # Low volatility ratio threshold (lower)

# Price action validation settings (MORE PERMISSIVE)
EMA_MIN_CANDLE_BODY_RATIO = 0.2              # Minimum candle body to total range ratio (lower)
EMA_REQUIRE_STRONG_CLOSE = False             # Don't require close near high/low (too restrictive)

# ADX trend strength validation (MORE PERMISSIVE FOR FOREX)
EMA_ADX_STRONG_TREND_THRESHOLD = 20.0        # ADX value indicating strong trend (lowered from 25)
EMA_ADX_WEAK_TREND_THRESHOLD = 15.0          # ADX value indicating weak trend (lowered from 20)
EMA_ADX_SCALING_FACTOR = 40.0                # ADX scaling factor for confidence calculation

# BALANCED HIGH-QUALITY validation confidence scoring
EMA_ENHANCED_MIN_CONFIDENCE = 0.70           # ✅ RESTORED: BALANCED MODE: High quality with good frequency
EMA_VALIDATION_WEIGHTS = {
    "multi_candle": 0.25,                    # Weight for multi-candle confirmation
    "volume": 0.15,                          # Weight for volume analysis
    "support_resistance": 0.20,              # Weight for support/resistance levels
    "market_conditions": 0.15,               # Weight for market regime analysis
    "trend_strength": 0.10,                  # Weight for ADX trend strength
    "price_action": 0.10,                    # Weight for price action patterns
    "pullback": 0.05                         # Weight for pullback/retest behavior
}

# Strategy comparison and A/B testing
EMA_ENABLE_VALIDATION_COMPARISON = True      # Enable comparison between old and new validation
EMA_LOG_REJECTED_SIGNALS = True             # Log signals that would have been accepted without enhanced validation

# Performance monitoring
EMA_TRACK_FALSE_POSITIVE_REDUCTION = True   # Track reduction in false positives
EMA_VALIDATION_PERFORMANCE_WINDOW = 100     # Number of signals to track for performance analysis

# =============================================================================
# OVEREXTENSION VALIDATION HELPER FUNCTIONS
# =============================================================================

def check_stochastic_overextension(stoch_k: float, stoch_d: float, signal_direction: str) -> dict:
    """
    Check if Stochastic indicates extreme overextension

    Args:
        stoch_k: Stochastic %K value
        stoch_d: Stochastic %D value
        signal_direction: 'long' or 'short'

    Returns:
        Dictionary with overextension status and penalty
    """
    if not STOCHASTIC_OVEREXTENSION_ENABLED:
        return {'overextended': False, 'penalty': 0.0, 'value': stoch_k}

    is_overextended = False
    penalty = 0.0

    if signal_direction == 'long' and stoch_k > STOCHASTIC_EXTREME_OVERBOUGHT:
        is_overextended = True
        penalty = OVEREXTENSION_CONFIDENCE_PENALTY
    elif signal_direction == 'short' and stoch_k < STOCHASTIC_EXTREME_OVERSOLD:
        is_overextended = True
        penalty = OVEREXTENSION_CONFIDENCE_PENALTY

    return {
        'overextended': is_overextended,
        'penalty': penalty,
        'value': stoch_k,
        'threshold_used': STOCHASTIC_EXTREME_OVERBOUGHT if signal_direction == 'long' else STOCHASTIC_EXTREME_OVERSOLD
    }

def check_williams_r_overextension(williams_r: float, signal_direction: str) -> dict:
    """
    Check if Williams %R indicates extreme overextension

    Args:
        williams_r: Williams %R value (-100 to 0)
        signal_direction: 'long' or 'short'

    Returns:
        Dictionary with overextension status and penalty
    """
    if not WILLIAMS_R_OVEREXTENSION_ENABLED:
        return {'overextended': False, 'penalty': 0.0, 'value': williams_r}

    is_overextended = False
    penalty = 0.0

    if signal_direction == 'long' and williams_r > WILLIAMS_R_EXTREME_OVERBOUGHT:
        is_overextended = True
        penalty = OVEREXTENSION_CONFIDENCE_PENALTY
    elif signal_direction == 'short' and williams_r < WILLIAMS_R_EXTREME_OVERSOLD:
        is_overextended = True
        penalty = OVEREXTENSION_CONFIDENCE_PENALTY

    return {
        'overextended': is_overextended,
        'penalty': penalty,
        'value': williams_r,
        'threshold_used': WILLIAMS_R_EXTREME_OVERBOUGHT if signal_direction == 'long' else WILLIAMS_R_EXTREME_OVERSOLD
    }

def check_rsi_extreme_overextension(rsi: float, signal_direction: str) -> dict:
    """
    Check if RSI indicates extreme overextension (more permissive than typical RSI)

    Args:
        rsi: RSI value (0-100)
        signal_direction: 'long' or 'short'

    Returns:
        Dictionary with overextension status and penalty
    """
    if not RSI_EXTREME_OVEREXTENSION_ENABLED:
        return {'overextended': False, 'penalty': 0.0, 'value': rsi}

    is_overextended = False
    penalty = 0.0

    if signal_direction == 'long' and rsi > RSI_EXTREME_OVERBOUGHT:
        is_overextended = True
        penalty = OVEREXTENSION_CONFIDENCE_PENALTY
    elif signal_direction == 'short' and rsi < RSI_EXTREME_OVERSOLD:
        is_overextended = True
        penalty = OVEREXTENSION_CONFIDENCE_PENALTY

    return {
        'overextended': is_overextended,
        'penalty': penalty,
        'value': rsi,
        'threshold_used': RSI_EXTREME_OVERBOUGHT if signal_direction == 'long' else RSI_EXTREME_OVERSOLD
    }

def calculate_composite_overextension_score(stoch_k: float, stoch_d: float,
                                          williams_r: float, rsi: float,
                                          signal_direction: str) -> dict:
    """
    Calculate composite overextension score from multiple oscillators

    Args:
        stoch_k: Stochastic %K value
        stoch_d: Stochastic %D value
        williams_r: Williams %R value
        rsi: RSI value
        signal_direction: 'long' or 'short'

    Returns:
        Dictionary with composite score and recommendation
    """
    if not COMPOSITE_OVEREXTENSION_ENABLED:
        return {'composite_overextended': False, 'total_penalty': 0.0, 'indicators_triggered': 0}

    # Check each indicator
    stoch_result = check_stochastic_overextension(stoch_k, stoch_d, signal_direction)
    williams_result = check_williams_r_overextension(williams_r, signal_direction)
    rsi_result = check_rsi_extreme_overextension(rsi, signal_direction)

    # Count triggered indicators
    triggered_count = sum([
        stoch_result['overextended'],
        williams_result['overextended'],
        rsi_result['overextended']
    ])

    # Calculate weighted penalty
    total_penalty = 0.0
    if triggered_count >= OVEREXTENSION_COMPOSITE_THRESHOLD:
        # Apply weighted penalties
        total_penalty = (
            stoch_result['penalty'] * OVEREXTENSION_WEIGHTS['stochastic'] +
            williams_result['penalty'] * OVEREXTENSION_WEIGHTS['williams_r'] +
            rsi_result['penalty'] * OVEREXTENSION_WEIGHTS['rsi']
        )

    return {
        'composite_overextended': triggered_count >= OVEREXTENSION_COMPOSITE_THRESHOLD,
        'total_penalty': total_penalty,
        'indicators_triggered': triggered_count,
        'individual_results': {
            'stochastic': stoch_result,
            'williams_r': williams_result,
            'rsi': rsi_result
        },
        'recommendation': 'block' if OVEREXTENSION_MODE == 'hard_block' and triggered_count >= OVEREXTENSION_COMPOSITE_THRESHOLD else 'penalty'
    }

def get_overextension_filter_status() -> dict:
    """Get current overextension filter configuration status"""
    return {
        'stochastic_enabled': STOCHASTIC_OVEREXTENSION_ENABLED,
        'williams_r_enabled': WILLIAMS_R_OVEREXTENSION_ENABLED,
        'rsi_extreme_enabled': RSI_EXTREME_OVEREXTENSION_ENABLED,
        'composite_enabled': COMPOSITE_OVEREXTENSION_ENABLED,
        'mode': OVEREXTENSION_MODE,
        'confidence_penalty': OVEREXTENSION_CONFIDENCE_PENALTY,
        'composite_threshold': OVEREXTENSION_COMPOSITE_THRESHOLD,
        'thresholds': {
            'stochastic_overbought': STOCHASTIC_EXTREME_OVERBOUGHT,
            'stochastic_oversold': STOCHASTIC_EXTREME_OVERSOLD,
            'williams_r_overbought': WILLIAMS_R_EXTREME_OVERBOUGHT,
            'williams_r_oversold': WILLIAMS_R_EXTREME_OVERSOLD,
            'rsi_overbought': RSI_EXTREME_OVERBOUGHT,
            'rsi_oversold': RSI_EXTREME_OVERSOLD
        }
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ema_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """
    Get EMA configuration for specific epic and market condition
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        market_condition: Market condition ('default', 'conservative', 'aggressive', etc.)
        
    Returns:
        Dictionary with EMA configuration
    """
    # Use dynamic config if enabled, otherwise use active config
    if ENABLE_DYNAMIC_EMA_CONFIG and market_condition in EMA_STRATEGY_CONFIG:
        config_name = market_condition
    else:
        config_name = ACTIVE_EMA_CONFIG
        
    return EMA_STRATEGY_CONFIG.get(config_name, EMA_STRATEGY_CONFIG['default'])

def get_ema_distance_for_epic(epic: str) -> float:
    """
    Get minimum EMA200 distance for specific epic
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        
    Returns:
        Minimum distance in pips
    """
    # Extract currency pair from epic (e.g., 'EURUSD' from 'CS.D.EURUSD.CEEM.IP')
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic
            
        return EMA200_MIN_DISTANCE_PIPS.get(pair, EMA200_MIN_DISTANCE_PIPS['DEFAULT'])
    except Exception:
        return EMA200_MIN_DISTANCE_PIPS['DEFAULT']

def get_ema_separation_for_epic(epic: str) -> float:
    """
    Get minimum EMA separation distance for specific epic
    
    Args:
        epic: Trading pair epic (e.g., 'CS.D.EURUSD.CEEM.IP')
        
    Returns:
        Minimum separation in pips
    """
    # Extract currency pair from epic
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic
            
        return MIN_EMA_SEPARATION_PIPS.get(pair, MIN_EMA_SEPARATION_PIPS['DEFAULT'])
    except Exception:
        return MIN_EMA_SEPARATION_PIPS['DEFAULT']

# Configuration summary function
def get_ema_config_summary() -> dict:
    """Get a summary of EMA configuration settings"""
    return {
        'strategy_enabled': SIMPLE_EMA_STRATEGY,
        'dynamic_config_enabled': ENABLE_DYNAMIC_EMA_CONFIG,
        'active_config': ACTIVE_EMA_CONFIG,
        'ema200_distance_enabled': EMA200_MIN_DISTANCE_ENABLED,
        'two_pole_enabled': TWO_POLE_OSCILLATOR_ENABLED,
        'macd_momentum_filter_enabled': MACD_MOMENTUM_FILTER_ENABLED,
        'macd_trend_sensitivity': MACD_TREND_SENSITIVITY,
        'macd_validation_mode': MACD_VALIDATION_MODE,
        'ema_periods': EMA_PERIODS,
        'min_data_periods': EMA_MIN_DATA_PERIODS,
        'debug_logging': EMA_DEBUG_LOGGING,
        'total_configurations': len(EMA_STRATEGY_CONFIG),
        # Enhanced validation settings
        'enhanced_validation_enabled': EMA_ENHANCED_VALIDATION,
        'confirmation_candles': EMA_CONFIRMATION_CANDLES,
        'volume_spike_threshold': EMA_VOLUME_SPIKE_THRESHOLD,
        'require_pullback': EMA_REQUIRE_PULLBACK,
        'enhanced_min_confidence': EMA_ENHANCED_MIN_CONFIDENCE,
        'validation_comparison_enabled': EMA_ENABLE_VALIDATION_COMPARISON,
        'false_positive_tracking': EMA_TRACK_FALSE_POSITIVE_REDUCTION,
        # Overextension filter settings
        'overextension_filters_enabled': any([
            STOCHASTIC_OVEREXTENSION_ENABLED,
            WILLIAMS_R_OVEREXTENSION_ENABLED,
            RSI_EXTREME_OVEREXTENSION_ENABLED,
            COMPOSITE_OVEREXTENSION_ENABLED
        ]),
        'stochastic_overextension': STOCHASTIC_OVEREXTENSION_ENABLED,
        'williams_r_overextension': WILLIAMS_R_OVEREXTENSION_ENABLED,
        'rsi_extreme_overextension': RSI_EXTREME_OVEREXTENSION_ENABLED,
        'composite_overextension': COMPOSITE_OVEREXTENSION_ENABLED,
        'overextension_mode': OVEREXTENSION_MODE,
        'overextension_penalty': OVEREXTENSION_CONFIDENCE_PENALTY
    }

# Quick configuration presets for different signal generation frequencies
def set_macd_filter_preset(preset: str = 'balanced'):
    """
    Quick presets for MACD filter configuration
    
    Args:
        preset: 'strict', 'balanced', 'permissive', 'minimal'
    """
    presets = {
        'strict': {
            'sensitivity': 'strict',
            'mode': 'strict_blocking',
            'description': 'Block all opposing momentum - fewest signals, highest quality'
        },
        'balanced': {
            'sensitivity': 'normal', 
            'mode': 'slope_aware',
            'description': 'Block strong opposing momentum - balanced signal frequency'
        },
        'permissive': {
            'sensitivity': 'permissive',
            'mode': 'slope_aware', 
            'description': 'Allow weak opposing momentum - more signals'
        },
        'minimal': {
            'sensitivity': 'neutral_bias',
            'mode': 'neutral_friendly',
            'description': 'Only block very strong opposing momentum - maximum signals'
        }
    }
    
    return presets.get(preset, presets['balanced'])

def set_enhanced_validation_preset(preset: str = 'balanced'):
    """
    Quick presets for enhanced validation configuration
    
    Args:
        preset: 'conservative', 'balanced', 'aggressive', 'minimal'
    """
    presets = {
        'conservative': {
            'confirmation_candles': 3,
            'volume_spike_threshold': 2.0,
            'enhanced_min_confidence': 0.7,
            'require_pullback': True,
            'strong_trend_threshold': 0.002,
            'adx_strong_threshold': 30.0,
            'description': 'Strictest validation - fewest false breakouts, fewer total signals'
        },
        'balanced': {
            'confirmation_candles': 2,
            'volume_spike_threshold': 1.5,
            'enhanced_min_confidence': 0.6,
            'require_pullback': True,
            'strong_trend_threshold': 0.001,
            'adx_strong_threshold': 25.0,
            'description': 'Balanced validation - good false breakout reduction with reasonable signal frequency'
        },
        'aggressive': {
            'confirmation_candles': 2,
            'volume_spike_threshold': 1.3,
            'enhanced_min_confidence': 0.5,
            'require_pullback': False,
            'strong_trend_threshold': 0.0005,
            'adx_strong_threshold': 20.0,
            'description': 'Moderate validation - some false breakout reduction with more signals'
        },
        'minimal': {
            'confirmation_candles': 1,
            'volume_spike_threshold': 1.2,
            'enhanced_min_confidence': 0.4,
            'require_pullback': False,
            'strong_trend_threshold': 0.0002,
            'adx_strong_threshold': 15.0,
            'description': 'Minimal validation - light filtering for maximum signal generation'
        }
    }
    
    return presets.get(preset, presets['balanced'])

def get_enhanced_validation_status():
    """Get current enhanced validation configuration status"""
    return {
        'enabled': EMA_ENHANCED_VALIDATION,
        'confirmation_candles': EMA_CONFIRMATION_CANDLES,
        'volume_threshold': EMA_VOLUME_SPIKE_THRESHOLD,
        'min_confidence': EMA_ENHANCED_MIN_CONFIDENCE,
        'pullback_required': EMA_REQUIRE_PULLBACK,
        'comparison_mode': EMA_ENABLE_VALIDATION_COMPARISON,
        'description': f"Enhanced validation {'ENABLED' if EMA_ENHANCED_VALIDATION else 'DISABLED'} with {EMA_CONFIRMATION_CANDLES} candle confirmation"
    }

def set_overextension_filter_preset(preset: str = 'conservative'):
    """
    Quick presets for overextension filter configuration

    Args:
        preset: 'disabled', 'conservative', 'balanced', 'comprehensive'

    Returns:
        Dictionary with recommended settings
    """
    presets = {
        'disabled': {
            'stochastic_enabled': False,
            'williams_r_enabled': False,
            'rsi_extreme_enabled': False,
            'composite_enabled': False,
            'description': 'All overextension filters disabled - maximum signal frequency'
        },
        'conservative': {
            'stochastic_enabled': True,
            'williams_r_enabled': False,
            'rsi_extreme_enabled': False,
            'composite_enabled': False,
            'mode': 'confidence_penalty',
            'stochastic_overbought': 90,  # Very extreme levels only
            'stochastic_oversold': 10,
            'description': 'Single Stochastic filter at very extreme levels - minimal impact'
        },
        'balanced': {
            'stochastic_enabled': True,
            'williams_r_enabled': True,
            'rsi_extreme_enabled': False,
            'composite_enabled': True,
            'mode': 'confidence_penalty',
            'composite_threshold': 2,  # Both must agree
            'description': 'Stochastic + Williams %R with composite scoring - moderate filtering'
        },
        'comprehensive': {
            'stochastic_enabled': True,
            'williams_r_enabled': True,
            'rsi_extreme_enabled': True,
            'composite_enabled': True,
            'mode': 'confidence_penalty',
            'composite_threshold': 2,  # Any 2 of 3 must agree
            'description': 'All oscillators enabled with composite scoring - strongest filtering'
        }
    }

    return presets.get(preset, presets['conservative'])

# =============================================================================
# PHASE 1+2 ENHANCEMENTS (Ported from Momentum Strategy)
# EMA = TREND-FOLLOWING STRATEGY
# =============================================================================

# Trend Alignment Filter - MANDATORY for trend-following
EMA_REQUIRE_TREND_ALIGNMENT = True       # EMA MUST trade with trend (core principle)
EMA_TREND_EMA_PERIOD = 200               # Use EMA 200 as primary trend filter
EMA_ALLOW_COUNTER_TREND = False          # Never trade counter-trend

# Market Regime Filter - Critical for trend-following strategies
EMA_ENABLE_REGIME_FILTER = True          # Filter out unfavorable market conditions
EMA_MIN_ADX = 25                         # Minimum trend strength (higher than momentum)
EMA_MIN_ATR_RATIO = 0.8                  # Current ATR > 0.8x baseline
EMA_MIN_EMA_SEPARATION = 0.3             # Price distance from EMA (in ATR units)

# Confirmation Requirements - EMA crossover + supporting indicators
EMA_MIN_CONFIRMATIONS = 2                # Require 2 confirmations (EMA + momentum/volume)
EMA_CONFIRMATION_TYPES = ['ema_crossover', 'macd_alignment', 'volume', 'price_momentum']

# =============================================================================
# PHASE 3: Adaptive Volatility-Based SL/TP (NEW)
# =============================================================================

# Runtime regime-aware calculation - No hardcoded values!
USE_ADAPTIVE_SL_TP = False               # 🧠 Enable adaptive volatility calculator (default: False for gradual rollout)
                                         # When True: Uses runtime regime detection (trending, ranging, breakout, high volatility)
                                         # When False: Falls back to ATR multipliers below

# Risk Management - ATR-based stops for trend-following (FALLBACK when adaptive disabled)
EMA_STOP_LOSS_ATR_MULTIPLIER = 2.0       # Tighter stops for trend-following
EMA_TAKE_PROFIT_ATR_MULTIPLIER = 4.0     # Larger targets (ride trends) (2.0:4.0 = 1:2 R:R)

# SUPERTREND-SPECIFIC: Wider stops for trend-following (overrides EMA settings when in Supertrend mode)
SUPERTREND_STOP_LOSS_ATR_MULTIPLIER = 2.5    # Wider stops - let trends breathe
SUPERTREND_TAKE_PROFIT_ATR_MULTIPLIER = 5.0  # Larger targets - ride trends (2.5:5.0 = 1:2 R:R)

# Pair-Specific Parameters
EMA_PAIR_SPECIFIC_PARAMS = {
    'EURUSD': {
        'short_ema': 21,
        'long_ema': 50,
        'trend_ema': 200,
        'stop_atr_multiplier': 2.0,
        'target_atr_multiplier': 4.0,
        'min_adx': 25
    },
    'GBPUSD': {
        'short_ema': 21,
        'long_ema': 50,
        'trend_ema': 200,
        'stop_atr_multiplier': 2.2,      # Wider for GBP volatility
        'target_atr_multiplier': 4.5,
        'min_adx': 27
    },
    'USDJPY': {
        'short_ema': 21,
        'long_ema': 50,
        'trend_ema': 200,
        'stop_atr_multiplier': 2.0,
        'target_atr_multiplier': 4.0,
        'min_adx': 25
    },
    'AUDUSD': {
        'short_ema': 21,
        'long_ema': 50,
        'trend_ema': 200,
        'stop_atr_multiplier': 2.1,
        'target_atr_multiplier': 4.0,
        'min_adx': 25
    }
}

# Structure-Based Stop Placement
EMA_USE_STRUCTURE_STOPS = True           # Place stops beyond recent swing points
EMA_STRUCTURE_LOOKBACK_BARS = 30         # Longer lookback for trend-following
EMA_MIN_STOP_DISTANCE_PIPS = 12.0        # Minimum stop distance (EMA mode)
EMA_MAX_STOP_DISTANCE_PIPS = 45.0        # Maximum stop distance (EMA mode)
EMA_STRUCTURE_BUFFER_PIPS = 2.0          # Buffer beyond swing point

# SUPERTREND-SPECIFIC: Wider minimums for trend-following
SUPERTREND_MIN_STOP_DISTANCE_PIPS = 20.0    # Wider minimum - let trends develop
SUPERTREND_MAX_STOP_DISTANCE_PIPS = 80.0    # Higher maximum - catch big moves
SUPERTREND_MIN_LIMIT_DISTANCE_PIPS = 40.0   # Minimum take profit - aim for 2:1

# Enhanced Confidence Calculation Factors
EMA_CONFIDENCE_BASE = 0.55               # Start slightly higher for trend clarity
EMA_CONFIDENCE_EMA_SEPARATION = 0.20     # Factor for EMA separation (trend strength)
EMA_CONFIDENCE_TREND_ALIGNMENT = 0.15    # MANDATORY alignment bonus
EMA_CONFIDENCE_REGIME_FAVORABLE = 0.10   # Bonus for favorable regime
EMA_CONFIDENCE_MACD_ALIGNMENT = 0.10     # Factor for MACD momentum alignment
EMA_CONFIDENCE_VOLUME = 0.05             # Factor for volume confirmation

# Trend Strength Requirements
EMA_MIN_EMA_SLOPE = 0.00005              # Minimum slope for trend validity
EMA_REQUIRE_ALIGNED_EMAS = True          # EMAs must be in proper order (9<21<200 for uptrend)
EMA_MAX_EMA_COMPRESSION = 0.002          # Max compression between EMAs (avoid ranging)

# =============================================================================
# SWING PROXIMITY VALIDATION CONFIGURATION (NEW)
# =============================================================================

# Swing Proximity Validation - Prevents poor entry timing near swing points
EMA_SWING_VALIDATION = {
    'enabled': True,  # Enable swing proximity validation
    'min_distance_pips': 12,  # RAISED from 8 - Minimum distance from swing high/low (Fix #3)
    'lookback_swings': 5,  # Number of recent swings to check
    'swing_length': 5,  # Bars for swing detection (matches SMC default)
    'strict_mode': True,  # CHANGED from False - REJECT signals near swings (Fix #2)
    'resistance_buffer': 1.0,  # Multiplier for resistance proximity (more cautious on buys)
    'support_buffer': 1.0,  # Multiplier for support proximity (more cautious on sells)
}

# Notes:
# - Prevents BUY signals when price is too close to recent swing highs (resistance)
# - Prevents SELL signals when price is too close to recent swing lows (support)
# - Works in conjunction with existing S/R validation (different timeframes)
# - Swing points: Recent pivots (5-50 bars)
# - S/R levels: Long-term zones (100-500 bars)
