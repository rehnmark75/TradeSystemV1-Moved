# ================================================
# DEV-APP CONFIGURATION
# ================================================

# ================== EPIC MAPPINGS ==================
# Ticker maps of IG specific ticker names
EPIC_MAP = {
    "GBPUSD.1.MINI": "CS.D.GBPUSD.MINI.IP",        # ← REVERSED from scanner
    "EURUSD.1.MINI": "CS.D.EURUSD.CEEM.IP",
    "USDJPY.100.MINI": "CS.D.USDJPY.MINI.IP",
    "AUDUSD.1.MINI": "CS.D.AUDUSD.MINI.IP",
    "USDCAD.1.MINI": "CS.D.USDCAD.MINI.IP",
    "EURJPY.100.MINI": "CS.D.EURJPY.MINI.IP",
    "AUDJPY.100.MINI": "CS.D.AUDJPY.MINI.IP",
    "NZDUSD.1.MINI": "CS.D.NZDUSD.MINI.IP",
    "USDCHF.1.MINI": "CS.D.USDCHF.MINI.IP"
}

# Trading blacklist - prevent trading for specific epics (scan-only mode)
TRADING_BLACKLIST = [
    # "EURUSD.1.MINI",  # Removed - trading now enabled
    # Add other blocked epics here as needed
]

# ================== TRAILING STOP MASTER CONTROL ==================
# Set to False to completely disable trailing stop monitoring
# Set to True to re-enable trailing stops
# ENABLED (Jan 2026): Re-enabled with scalp-specific configs to replace VSL system
TRAILING_STOPS_ENABLED = True

# ================== GUARANTEED PROFIT LOCK ==================
# CRITICAL PROTECTION: If trade reaches this profit, never go into loss
# This applies to BOTH fixed and dynamic trailing systems
# Feature added: Jan 2026 - Ensures profitable trades stay profitable
ENABLE_GUARANTEED_PROFIT_LOCK = True
GUARANTEED_PROFIT_LOCK_TRIGGER = 10  # Pips profit to activate protection
GUARANTEED_PROFIT_LOCK_MINIMUM = 1   # Minimum pips profit to lock (SL = entry + 1)

# Additional epic mappings from broker transaction analyzer
BROKER_EPIC_MAP = {
    'USD/CAD': 'CS.D.USDCAD.MINI.IP',
    'USD/CHF': 'CS.D.USDCHF.MINI.IP', 
    'USD/JPY': 'CS.D.USDJPY.MINI.IP',
    'EUR/JPY': 'CS.D.EURJPY.MINI.IP',
    'EUR/USD': 'CS.D.EURUSD.CEEM.IP',
    'GBP/USD': 'CS.D.GBPUSD.MINI.IP',
    'AUD/USD': 'CS.D.AUDUSD.MINI.IP',
    'NZD/USD': 'CS.D.NZDUSD.MINI.IP',
    'GBP/JPY': 'CS.D.GBPJPY.MINI.IP',
    'EUR/GBP': 'CS.D.EURGBP.MINI.IP'
}

# Default epics for testing and examples
DEFAULT_EPICS = {
    'EURUSD': 'CS.D.EURUSD.CEEM.IP',
    'USDJPY': 'CS.D.USDJPY.MINI.IP',
    'GBPUSD': 'CS.D.GBPUSD.MINI.IP'
}

# ================== API ENDPOINTS ==================
# IG Markets API
API_BASE_URL = "https://demo-api.ig.com/gateway/deal"  # or your demo/live base URL

# Internal service URLs
FASTAPI_DEV_URL = "http://fastapi-dev:8000"
FASTAPI_STREAM_URL = "http://fastapi-stream:8000"

# Specific endpoints
ADJUST_STOP_URL = f"{FASTAPI_DEV_URL}/orders/adjust-stop"

# IG Lightstreamer (for streaming)
LIGHTSTREAMER_URL = "https://demo-apd.marketdatasystems.com"

# ================== AUTHENTICATION ==================
# Note: IG_API_KEY and IG_PWD values are now mapped to environment variables
# by the keyvault.py service. These names are kept for backward compatibility.
IG_USERNAME = "rehnmarkhdemo"
IG_API_KEY = "demoapikey"
IG_PWD = "demopwd"

# ================== TRADE COOLDOWN CONTROLS ==================
TRADE_COOLDOWN_ENABLED = True
TRADE_COOLDOWN_MINUTES = 30  # Default cooldown period after closing a trade
EPIC_SPECIFIC_COOLDOWNS = {
    'CS.D.EURUSD.CEEM.IP': 45,  # Major pairs get longer cooldown
    'CS.D.GBPUSD.MINI.IP': 45,
    'CS.D.USDJPY.MINI.IP': 30,
    'CS.D.AUDUSD.MINI.IP': 30,
    'CS.D.USDCAD.MINI.IP': 30,
    # Others use DEFAULT_TRADE_COOLDOWN_MINUTES
}

# ================== RISK MANAGEMENT SETTINGS ==================
# Default risk-reward ratio (take profit will be this multiple of stop loss)
DEFAULT_RISK_REWARD_RATIO = 2.0

# Epic-specific risk-reward ratios (for fine-tuning based on volatility)
EPIC_RISK_REWARD_RATIOS = {
    'CS.D.EURUSD.CEEM.IP': 2.5,  # More volatile pairs get higher RR
    'CS.D.GBPUSD.MINI.IP': 2.5,
    'CS.D.USDJPY.MINI.IP': 2.0,  # Stable pairs use default
    'CS.D.AUDUSD.MINI.IP': 2.0,
    'CS.D.USDCAD.MINI.IP': 2.0,
}

# ATR calculation settings
ATR_PERIODS = 14  # Number of periods for ATR calculation
ATR_STOP_MULTIPLIER = 1.5  # ATR multiplier for stop loss distance

# ================== PROGRESSIVE TRAILING SETTINGS ==================
# 4-Stage Progressive Trailing System (based on MAE analysis Dec 2025)
#
# MAE ANALYSIS FINDINGS (14 days of trade data):
# - Winners: Avg MAE 3.1 pips, Median 2.7 pips, 75th percentile 3.5 pips
# - Losers: Avg MAE 15.0 pips, Median 13.2 pips
# - Conclusion: Good trades barely dip, bad trades dip significantly
#
# SMALL ACCOUNT PRIORITY: Protect capital early, accept smaller winners

# Stage 0: EARLY BREAKEVEN (NEW - Small Account Protection)
# Triggers early to protect capital when trade shows initial profit
# Based on MAE analysis: winners only dip 3 pips, so +6 pips is safe
EARLY_BREAKEVEN_TRIGGER_POINTS = 6   # Move to breakeven after +6 points
EARLY_BREAKEVEN_BUFFER_POINTS = 1    # SL moves to entry + 1 point (covers spread)

# Stage 1: Profit Lock (formerly Break-Even)
# Lock small guaranteed profit
STAGE1_TRIGGER_POINTS = 10   # Lock profit after +10 points (was 7)
STAGE1_LOCK_POINTS = 5       # Guarantee +5 points profit (was 2)

# Stage 2: Profit Lock-In (OPTIMIZED FOR TREND FOLLOWING)
STAGE2_TRIGGER_POINTS = 15   # Lock in meaningful profit after +15 points (was 16)
STAGE2_LOCK_POINTS = 10      # Guarantee +10 points profit

# Stage 3: Dynamic Percentage Trailing (STANDARDIZED FOR ALL PAIRS)
STAGE3_TRIGGER_POINTS = 20   # Start percentage trailing after +20 points (was 17)
STAGE3_ATR_MULTIPLIER = 0.8  # MUCH TIGHTER: 0.8x ATR (was 1.5x)
STAGE3_MIN_DISTANCE = 2      # Small minimum trailing distance from current price (was 3)
STAGE3_MIN_ADJUSTMENT = 2    # Minimum points to move stop (lowered for scalp trades)

# ================== PARTIAL CLOSE CONFIGURATION ==================
# Partial close feature: Close part of position at break-even trigger instead of moving stop
ENABLE_PARTIAL_CLOSE_AT_BREAKEVEN = False  # Master toggle for partial close feature (DISABLED)
PARTIAL_CLOSE_SIZE = 0.5  # Size to close (0.5 = 50% of position)

# v3.1.0: Post-Partial-Close Trailing Enhancements
# After partial close, give the remaining position room to breathe before aggressive trailing
POST_PARTIAL_CLOSE_CALM_MINUTES = 5  # Wait 5 minutes before trailing resumes after partial close
POST_PARTIAL_CLOSE_MIN_MOVE_PIPS = 5  # OR require price to move +5 pips beyond partial close level
PARTIAL_POSITION_TRAIL_MULTIPLIER = 1.5  # Use 1.5x wider trailing distance for partial positions

# ================== PAIR-SPECIFIC TRAILING CONFIGURATIONS ==================
# Per-pair trailing stop configurations (overrides default values above)
# Note: IG's min_stop_distance_points from trade_log ALWAYS takes priority when available
# These configs act as fallback or can set HIGHER values for tighter protection
#
# v2.7.0 OPTIMIZATION (2025-12-23) - MFE/MAE Analysis Based:
# - BE triggers optimized per-pair based on actual MFE/MAE data analysis
# - Used weighted average of BUY/SELL optimal values (SELL often needs higher)
# - Key insight: SELL positions typically need higher BE triggers than BUY
#
# BUY vs SELL Analysis Summary (optimal_be_trigger):
#   EURJPY: BUY=27, SELL=55 -> Use 40 (SELL-weighted, high discrepancy)
#   GBPUSD: BUY=25, SELL=46 -> Use 35 (average, SELL needs more room)
#   USDJPY: BUY=33, SELL=47 -> Use 40 (average)
#   EURUSD: BUY=24, SELL=12 -> Use 20 (BUY-weighted, SELL is lower)
#   NZDUSD: BUY=12, SELL=17 -> Use 15 (both low, use average)
#   AUDUSD: BUY=13, SELL=28 -> Use 20 (SELL-weighted)
#   USDCAD: BUY=20, SELL=18 -> Use 20 (keep similar)
#   USDCHF: BUY=26, SELL=25 -> Use 26 (already optimized v2.6.0)
#   AUDJPY: BUY=33, SELL=25 -> Use 28 (BUY-weighted)

# =============================================================================
# v3.0.0 TRAILING STOP REDESIGN - Wider triggers + 2.0x ATR trailing
# =============================================================================
# Key changes from v2.9.0:
# - Stage 3 ATR multiplier: 0.8x → 2.0x (industry standard for trend capture)
# - Stage 3 min distance: 4 → 8 pips (breathing room)
# - Early BE trigger: 10 → 15 pips for majors, 20 pips for JPY crosses
# - Early BE buffer: 1 → 2 pips (lock more profit)
# - Stage triggers: 20/30/40 → 25/38/50 (Fibonacci-inspired gaps)
# - Stage lock points: 8/15 → 12/20 (lock more profit at each stage)
# - Partial close: 13 pips @ 50% → 20 pips @ 40% (let winners run)
# - Time protection: 60 mins @ 10 pips → 90 mins @ 15 pips
#
# Expected impact: +56% expectancy, +83% avg winner size
# =============================================================================

# Time-based protection settings (ENHANCED in v3.0.0)
TIME_BASED_PROTECTION = {
    'enabled': True,
    'min_profit_pips': 15,              # v3.0.0: Raised 10→15 pips
    'time_threshold_minutes': 90,       # v3.0.0: Raised 60→90 mins (6 candles on 15m TF)
    'protection_buffer_pips': 5,        # v3.0.0: Raised 2→5 pips (meaningful protection)
}

PAIR_TRAILING_CONFIGS = {
    # ==========================================================================
    # v3.0.0: BALANCED CONFIG - Wider triggers for better profit capture
    # Rationale: v2.9.0 was cutting winners too early. Stage 3 ATR was 0.8x
    # (2.5x tighter than industry standard 2.0-3.0x).
    # Now using Fibonacci-inspired gaps: 15→25→38→50 with 2.0x ATR trailing.
    # JPY crosses get even wider triggers (20 pips) due to higher volatility.
    # ==========================================================================

    # ========== MAJOR PAIRS (15 pip early BE) ==========
    'CS.D.EURUSD.CEEM.IP': {
        'early_breakeven_trigger_points': 15,   # v3.0.0: 10→15
        'early_breakeven_buffer_points': 2,     # v3.0.0: 1→2
        'stage1_trigger_points': 25,            # v3.0.0: 20→25
        'stage1_lock_points': 12,               # v3.0.0: 8→12
        'stage2_trigger_points': 38,            # v3.0.0: 30→38 (Fibonacci)
        'stage2_lock_points': 20,               # v3.0.0: 15→20
        'stage3_trigger_points': 50,            # v3.0.0: 40→50
        'stage3_atr_multiplier': 2.0,           # v3.0.0: 0.8→2.0 (CRITICAL FIX)
        'stage3_min_distance': 8,               # v3.0.0: 4→8
        'min_trail_distance': 10,               # v3.0.0: 8→10
        'break_even_trigger_points': 18,        # v3.0.0: 15→18
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,     # v3.0.0: 13→20
        'partial_close_size': 0.4,              # v3.0.0: 0.5→0.4 (keep 60% running)
    },

    'CS.D.AUDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.NZDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.USDCAD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.USDCHF.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    # ========== GBP PAIRS (15 pip early BE for GBPUSD, 20 for JPY crosses) ==========
    'CS.D.GBPUSD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    # GBPJPY - High volatility JPY cross (20 pip early BE)
    'CS.D.GBPJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,   # v3.0.0: Wider for JPY volatility
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,            # v3.0.0: Wider for JPY crosses
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    'CS.D.GBPAUD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.GBPNZD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    # ========== JPY PAIRS (20 pip early BE - higher volatility) ==========
    # JPY crosses have higher volatility, need wider triggers to avoid premature stops
    'CS.D.USDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,   # v3.0.0: 10→20 for JPY
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,            # v3.0.0: 20→30 for JPY
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    'CS.D.EURJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    'CS.D.AUDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    'CS.D.CADJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    'CS.D.CHFJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    'CS.D.NZDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 20,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 30,
        'stage1_lock_points': 15,
        'stage2_trigger_points': 45,
        'stage2_lock_points': 25,
        'stage3_trigger_points': 60,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 10,
        'min_trail_distance': 12,
        'break_even_trigger_points': 22,
        'enable_partial_close': False,
        'partial_close_trigger_points': 25,
        'partial_close_size': 0.4,
    },

    # ========== CROSS PAIRS (15 pip early BE) ==========
    'CS.D.EURAUD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.EURNZD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.AUDNZD.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },

    'CS.D.EURGBP.MINI.IP': {
        'early_breakeven_trigger_points': 15,
        'early_breakeven_buffer_points': 2,
        'stage1_trigger_points': 25,
        'stage1_lock_points': 12,
        'stage2_trigger_points': 38,
        'stage2_lock_points': 20,
        'stage3_trigger_points': 50,
        'stage3_atr_multiplier': 2.0,
        'stage3_min_distance': 8,
        'min_trail_distance': 10,
        'break_even_trigger_points': 18,
        'enable_partial_close': False,
        'partial_close_trigger_points': 20,
        'partial_close_size': 0.4,
    },
}

# =============================================================================
# SCALP MODE TRAILING CONFIGURATIONS (Jan 2026)
# =============================================================================
# Scalp-specific trailing stop configs based on VSL analysis (Jan 20-22, 2026)
#
# Key findings from 2-day analysis:
# - VSL at 5-6 pips was too tight (67% premature stops, $2,506 loss)
# - Profitable trades captured only 46% of potential (83% locked at BE)
# - Optimal stops: 12-20 pips (ABOVE IG minimum, so VSL not needed)
# - USDCAD: 100% success rate with 12 pips
# - Majors: 15 pips optimal
# - JPY pairs: 20 pips due to higher volatility
#
# Strategy: Use IG native stops with progressive trailing
# =============================================================================

SCALP_TRAILING_CONFIGS = {
    # ========== USDCAD - HIGHEST PRIORITY (100% success rate) ==========
    'CS.D.USDCAD.MINI.IP': {
        'early_breakeven_trigger_points': 6,    # Quick BE protection
        'early_breakeven_buffer_points': 1,     # Lock +1 pip (from data: improve from 0.5)
        'stage1_trigger_points': 10,            # Stage1 at +10 pips
        'stage1_lock_points': 5,                # Lock +5 pips
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,           # Tighter ATR for scalping
        'stage3_min_distance': 5,
        'min_trail_distance': 6,                # 12 pip initial stop (optimal from data)
        'break_even_trigger_points': 6,
        'enable_partial_close': False,
        'partial_close_trigger_points': 10,     # Close 50% at +10 pips
        'partial_close_size': 0.5,
    },

    # ========== MAJOR PAIRS (15 pip optimal stop) ==========
    'CS.D.EURUSD.CEEM.IP': {
        'early_breakeven_trigger_points': 8,    # From data: avoid premature BE
        'early_breakeven_buffer_points': 1,     # Lock +1 pip (improved from 0.5)
        'stage1_trigger_points': 10,            # Stage1 at +10 pips
        'stage1_lock_points': 5,                # Lock +5 pips
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop (optimal from data)
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,     # Close 50% at +12 pips
        'partial_close_size': 0.5,
    },

    'CS.D.GBPUSD.MINI.IP': {
        'early_breakeven_trigger_points': 8,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    'CS.D.AUDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 6,    # AUDUSD: 14.3% capture rate, needs adjustment
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,            # Realistic MFE for AUDUSD (3-3.5 pips typical)
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 4,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 6,
        'enable_partial_close': False,
        'partial_close_trigger_points': 10,
        'partial_close_size': 0.5,
    },

    'CS.D.NZDUSD.MINI.IP': {
        'early_breakeven_trigger_points': 8,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    'CS.D.USDCHF.MINI.IP': {
        'early_breakeven_trigger_points': 8,
        'early_breakeven_buffer_points': 1,
        'stage1_trigger_points': 10,
        'stage1_lock_points': 5,
        'stage2_trigger_points': 15,            # Stage2 at +15 pips - lock solid profit
        'stage2_lock_points': 10,               # Lock +10 pips
        'stage3_trigger_points': 17,            # Stage3 dynamic trailing starts at +17
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 5,
        'min_trail_distance': 8,                # 15 pip initial stop
        'break_even_trigger_points': 8,
        'enable_partial_close': False,
        'partial_close_trigger_points': 12,
        'partial_close_size': 0.5,
    },

    # ========== JPY PAIRS (20 pip optimal stop - higher volatility) ==========
    'CS.D.USDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,   # JPY pairs need more room
        'early_breakeven_buffer_points': 1.5,   # Lock +1.5 pips (higher volatility)
        'stage1_trigger_points': 15,            # Stage1 at +15 pips
        'stage1_lock_points': 8,                # Lock +8 pips
        'stage2_trigger_points': 20,            # Stage2 at +20 pips - lock solid profit
        'stage2_lock_points': 15,               # Lock +15 pips
        'stage3_trigger_points': 22,            # Stage3 dynamic trailing starts at +22
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop (optimal from data)
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,     # Close 50% at +15 pips
        'partial_close_size': 0.5,
    },

    'CS.D.EURJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,   # EURJPY: 31.3% capture, needs loosening
        'early_breakeven_buffer_points': 1.5,
        'stage1_trigger_points': 15,
        'stage1_lock_points': 8,
        'stage2_trigger_points': 20,            # Stage2 at +20 pips - lock solid profit
        'stage2_lock_points': 15,               # Lock +15 pips
        'stage3_trigger_points': 22,            # Stage3 dynamic trailing starts at +22
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
    },

    'CS.D.AUDJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,
        'early_breakeven_buffer_points': 1.5,
        'stage1_trigger_points': 15,
        'stage1_lock_points': 8,
        'stage2_trigger_points': 20,            # Stage2 at +20 pips - lock solid profit
        'stage2_lock_points': 15,               # Lock +15 pips
        'stage3_trigger_points': 22,            # Stage3 dynamic trailing starts at +22
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
    },

    'CS.D.GBPJPY.MINI.IP': {
        'early_breakeven_trigger_points': 10,
        'early_breakeven_buffer_points': 1.5,
        'stage1_trigger_points': 15,
        'stage1_lock_points': 8,
        'stage2_trigger_points': 20,
        'stage2_lock_points': 12,
        'stage3_trigger_points': 25,
        'stage3_atr_multiplier': 1.5,
        'stage3_min_distance': 6,
        'min_trail_distance': 10,               # 20 pip initial stop
        'break_even_trigger_points': 10,
        'enable_partial_close': False,
        'partial_close_trigger_points': 15,
        'partial_close_size': 0.5,
    },
}

# Default scalp configuration for unlisted pairs
DEFAULT_SCALP_TRAILING_CONFIG = {
    'early_breakeven_trigger_points': 8,
    'early_breakeven_buffer_points': 1,
    'stage1_trigger_points': 12,
    'stage1_lock_points': 6,
    'stage2_trigger_points': 15,
    'stage2_lock_points': 10,
    'stage3_trigger_points': 20,
    'stage3_atr_multiplier': 1.5,
    'stage3_min_distance': 5,
    'min_trail_distance': 8,                    # 15 pip initial stop
    'break_even_trigger_points': 8,
    'enable_partial_close': False,
    'partial_close_trigger_points': 12,
    'partial_close_size': 0.5,
}

# Default configuration - v3.0.0 balanced settings for unlisted pairs
DEFAULT_TRAILING_CONFIG = {
    'early_breakeven_trigger_points': 15,   # v3.0.0: 10→15
    'early_breakeven_buffer_points': 2,     # v3.0.0: 1→2
    'stage1_trigger_points': 25,            # v3.0.0: 20→25
    'stage1_lock_points': 12,               # v3.0.0: 8→12
    'stage2_trigger_points': 38,            # v3.0.0: 30→38
    'stage2_lock_points': 20,               # v3.0.0: 15→20
    'stage3_trigger_points': 50,            # v3.0.0: 40→50
    'stage3_atr_multiplier': 2.0,           # v3.0.0: 0.8→2.0 (CRITICAL)
    'stage3_min_distance': 8,               # v3.0.0: 4→8
    'min_trail_distance': 10,               # v3.0.0: 8→10
    'break_even_trigger_points': 18,        # v3.0.0: 15→18
    'enable_partial_close': False,
    'partial_close_trigger_points': 20,     # v3.0.0: 13→20
    'partial_close_size': 0.4,              # v3.0.0: 0.5→0.4
}


def get_trailing_config_for_epic(epic: str, is_scalp_trade: bool = False) -> dict:
    """
    Get trailing stop configuration for specific epic/pair.

    Priority for scalp trades:
    1. Pair-specific config from SCALP_TRAILING_CONFIGS
    2. DEFAULT_SCALP_TRAILING_CONFIG fallback

    Priority for regular trades:
    1. Pair-specific config from PAIR_TRAILING_CONFIGS
    2. DEFAULT_TRAILING_CONFIG fallback

    Note: IG's min_stop_distance_points from trade_log ALWAYS takes priority
    when available. These configs are fallback or can set HIGHER values.

    Args:
        epic: Trading symbol (e.g., 'CS.D.EURUSD.CEEM.IP')
        is_scalp_trade: Whether this is a scalp trade (default: False)

    Returns:
        Dictionary with trailing configuration values
    """
    if is_scalp_trade:
        config = SCALP_TRAILING_CONFIGS.get(epic, DEFAULT_SCALP_TRAILING_CONFIG.copy())
    else:
        config = PAIR_TRAILING_CONFIGS.get(epic, DEFAULT_TRAILING_CONFIG.copy())
    return config


def get_scalp_trailing_config_for_epic(epic: str) -> dict:
    """
    Convenience function to get scalp trailing configuration.

    Args:
        epic: Trading symbol

    Returns:
        Dictionary with scalp trailing configuration values
    """
    return get_trailing_config_for_epic(epic, is_scalp_trade=True)

# ================== DEFAULT VALUES ==================
DEFAULT_TEST_EPIC = "CS.D.USDJPY.MINI.IP"

