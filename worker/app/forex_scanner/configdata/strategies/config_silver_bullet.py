"""
ICT Silver Bullet Strategy Configuration

VERSION: 1.0.0
DATE: 2025-12-15

The Silver Bullet is a time-based SMC strategy that trades during specific
one-hour windows, looking for liquidity sweeps followed by FVG entries.

Time Windows (New York Time):
- London Open: 03:00 - 04:00 AM
- NY AM Session: 10:00 - 11:00 AM (BEST)
- NY PM Session: 02:00 - 03:00 PM
"""

# =============================================================================
# STRATEGY METADATA
# =============================================================================
STRATEGY_NAME = "SILVER_BULLET"
STRATEGY_VERSION = "1.0.0"
STRATEGY_DATE = "2025-12-15"
STRATEGY_STATUS = "Development"

# =============================================================================
# TIME WINDOWS (UTC Hours)
# =============================================================================
# NY Time to UTC conversion:
# - EST (Nov-Mar): NY + 5 hours = UTC
# - EDT (Mar-Nov): NY + 4 hours = UTC
# We use wider windows to cover both EST and EDT

# London Open: 03:00-04:00 NY = 07:00-09:00 UTC (covers both)
LONDON_OPEN_WINDOW_START = 7
LONDON_OPEN_WINDOW_END = 9

# NY AM Session: 10:00-11:00 NY = 14:00-16:00 UTC (covers both)
NY_AM_WINDOW_START = 14
NY_AM_WINDOW_END = 16

# NY PM Session: 14:00-15:00 NY = 18:00-20:00 UTC (covers both)
NY_PM_WINDOW_START = 18
NY_PM_WINDOW_END = 20

# Session configuration
ENABLED_SESSIONS = ['LONDON_OPEN', 'NY_AM', 'NY_PM']

# Session quality multipliers (for confidence scoring)
SESSION_QUALITY = {
    'NY_AM': 1.00,      # Best session - London/NY overlap
    'NY_PM': 0.90,      # Good session - US institutional activity
    'LONDON_OPEN': 0.85  # Good for EUR/GBP pairs
}

# =============================================================================
# LIQUIDITY DETECTION
# =============================================================================
# Lookback for finding liquidity levels (swing highs/lows)
LIQUIDITY_LOOKBACK_BARS = 20

# Minimum swing strength (bars on each side to confirm swing)
SWING_STRENGTH_BARS = 2  # Reduced from 3 for more liquidity levels

# Minimum sweep beyond liquidity level (pips)
LIQUIDITY_SWEEP_MIN_PIPS = 1.5  # Reduced from 3 for more sweeps in testing

# Maximum sweep - beyond this it's a breakout, not a sweep (pips)
LIQUIDITY_SWEEP_MAX_PIPS = 20  # Increased from 15 for more flexibility

# Require price to return after sweep (confirms it's a sweep not breakout)
REQUIRE_SWEEP_REJECTION = False  # Relaxed for initial testing

# Maximum bars after sweep to still consider it valid
SWEEP_MAX_AGE_BARS = 15  # Increased from 10 for more opportunities

# =============================================================================
# MARKET STRUCTURE SHIFT (MSS)
# =============================================================================
# Look for MSS within N bars after liquidity sweep
MSS_LOOKBACK_BARS = 15

# Minimum break beyond swing to confirm MSS (pips)
MSS_MIN_BREAK_PIPS = 2

# Require body close beyond swing (not just wick)
MSS_REQUIRE_BODY_CLOSE = True

# =============================================================================
# FAIR VALUE GAP (FVG) PARAMETERS
# =============================================================================
# Minimum FVG size to be tradeable (pips)
FVG_MIN_SIZE_PIPS = 0.5  # Reduced from 2 for initial testing

# Maximum age of FVG to consider for entry (bars)
FVG_MAX_AGE_BARS = 10

# FVG entry zone (0.0 = far edge, 1.0 = near edge)
# For bullish FVG: 0.0 = top of gap, 1.0 = bottom
# We enter in the upper 50% for better R:R
FVG_ENTRY_ZONE_MIN = 0.0
FVG_ENTRY_ZONE_MAX = 0.5

# Only use unfilled FVGs (partially filled OK)
FVG_MAX_FILL_PERCENTAGE = 0.5

# =============================================================================
# RISK MANAGEMENT
# =============================================================================
# Minimum Risk:Reward ratio
MIN_RR_RATIO = 2.0

# Optimal R:R for fallback TP calculation
OPTIMAL_RR_RATIO = 2.5

# Maximum R:R (cap unrealistic targets)
MAX_RR_RATIO = 4.0

# Minimum take profit (pips) - Silver Bullet standard
MIN_TP_PIPS = 15

# Maximum stop loss (pips)
MAX_SL_PIPS = 20

# Stop loss buffer beyond FVG/swing (pips)
SL_BUFFER_PIPS = 2

# Use ATR for dynamic SL calculation
USE_ATR_STOP = True
SL_ATR_MULTIPLIER = 1.5

# ATR period for calculations
ATR_PERIOD = 14

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================
# Minimum confidence to generate signal
MIN_CONFIDENCE_THRESHOLD = 0.55

# Confidence weights
CONFIDENCE_WEIGHTS = {
    'session_quality': 0.20,    # Which session we're in
    'sweep_quality': 0.25,      # Clean sweep with rejection
    'fvg_quality': 0.20,        # FVG size and freshness
    'mss_strength': 0.20,       # Market structure shift strength
    'htf_alignment': 0.15       # Higher timeframe alignment
}

# =============================================================================
# TIMEFRAME CONFIGURATION
# =============================================================================
# Higher timeframe for bias (trend direction)
HTF_BIAS_TIMEFRAME = '1h'

# Trigger timeframe for liquidity and MSS detection
TRIGGER_TIMEFRAME = '15m'

# Entry timeframe for FVG and precise entry
ENTRY_TIMEFRAME = '5m'

# =============================================================================
# PAIR-SPECIFIC SETTINGS
# =============================================================================
# Session preferences by pair
PAIR_SESSION_PREFERENCES = {
    'EURUSD': ['NY_AM', 'LONDON_OPEN', 'NY_PM'],
    'GBPUSD': ['NY_AM', 'LONDON_OPEN', 'NY_PM'],
    'USDJPY': ['NY_AM', 'NY_PM', 'LONDON_OPEN'],
    'USDCAD': ['NY_AM', 'NY_PM', 'LONDON_OPEN'],
    'AUDUSD': ['NY_AM', 'LONDON_OPEN', 'NY_PM'],
    'NZDUSD': ['NY_AM', 'LONDON_OPEN', 'NY_PM'],
    'USDCHF': ['NY_AM', 'NY_PM', 'LONDON_OPEN'],
    'EURJPY': ['NY_AM', 'LONDON_OPEN', 'NY_PM'],
    'GBPJPY': ['NY_AM', 'LONDON_OPEN', 'NY_PM'],
}

# Pair-specific minimum TP (some pairs move more)
PAIR_MIN_TP_PIPS = {
    'GBPUSD': 18,
    'GBPJPY': 25,
    'EURJPY': 20,
    'DEFAULT': 15
}

# Pair-specific max SL
PAIR_MAX_SL_PIPS = {
    'GBPUSD': 25,
    'GBPJPY': 30,
    'EURJPY': 25,
    'DEFAULT': 20
}

# =============================================================================
# SIGNAL LIMITS & COOLDOWN
# =============================================================================
# Maximum signals per pair per day
MAX_SIGNALS_PER_PAIR_PER_DAY = 2

# Cooldown after signal (hours)
SIGNAL_COOLDOWN_HOURS = 2

# Maximum signals per session per pair
MAX_SIGNALS_PER_SESSION = 1

# =============================================================================
# DEBUG & LOGGING
# =============================================================================
ENABLE_DEBUG_LOGGING = True
LOG_LIQUIDITY_LEVELS = True
LOG_SWEEP_DETECTION = True
LOG_MSS_DETECTION = True
LOG_FVG_ANALYSIS = True
