# Add these configurations to your existing config.py file
# Enhanced Smart Money Configurations - Production Ready

# =============================================================================
# SMART MONEY CONCEPTS (SMC) CONFIGURATION
# =============================================================================

# Core Smart Money Settings
SMART_MONEY_ENABLED = True
SMART_MONEY_READONLY_ENABLED = True  # Non-disruptive analysis mode
SMART_MONEY_VERSION = "2.0.0_production"

# Market Structure Analyzer Settings
STRUCTURE_SWING_LOOKBACK = 5                    # Bars to look back for swing detection
STRUCTURE_MIN_SWING_STRENGTH = 0.3             # Minimum swing strength (0-1)
STRUCTURE_BOS_CONFIRMATION_PIPS = 5             # Pips required for BOS confirmation
STRUCTURE_CHOCH_LOOKBACK = 20                   # Bars to analyze for ChoCh detection
STRUCTURE_REQUIRE_CLOSE = True                  # Require close beyond level for confirmation
STRUCTURE_MAX_SWING_POINTS = 30                 # Maximum swing points to track
STRUCTURE_MAX_EVENTS = 20                       # Maximum structure events to store

# Order Flow Analyzer Settings
ORDER_FLOW_MIN_OB_SIZE_PIPS = 8                 # Minimum order block size in pips
ORDER_FLOW_MIN_FVG_SIZE_PIPS = 5                # Minimum fair value gap size in pips
ORDER_FLOW_MAX_LOOKBACK_BARS = 50               # Maximum bars to analyze for performance
ORDER_FLOW_MAX_ORDER_BLOCKS = 10                # Maximum order blocks to track
ORDER_FLOW_MAX_FVGS = 10                        # Maximum FVGs to track
ORDER_FLOW_DISPLACEMENT_FACTOR = 1.5            # ATR multiplier for displacement detection
ORDER_FLOW_SKIP_SUPPLY_DEMAND = True            # Skip expensive S/D zone calculation
ORDER_FLOW_PROXIMITY_PIPS = 20                  # Pips for proximity calculations

# Smart Money Read-Only Analyzer Settings
SMART_MONEY_MIN_DATA_POINTS = 50                # Minimum data points for analysis
SMART_MONEY_ANALYSIS_TIMEOUT = 10.0             # Maximum analysis time in seconds
SMART_MONEY_STRUCTURE_WEIGHT = 0.4              # Weight for structure analysis in scoring
SMART_MONEY_ORDER_FLOW_WEIGHT = 0.3             # Weight for order flow analysis in scoring
SMART_MONEY_MIN_CONFIDENCE_BOOST = 0.1          # Minimum confidence boost for aligned signals
SMART_MONEY_MAX_CONFIDENCE_BOOST = 0.3          # Maximum confidence boost for aligned signals

# Smart Money Strategy Integrations
SMART_MONEY_STRUCTURE_VALIDATION = True         # Enable structure validation for strategies
SMART_MONEY_ORDER_FLOW_VALIDATION = True        # Enable order flow validation for strategies
SMART_MONEY_MIN_SCORE = 0.4                     # Minimum smart money score for signal acceptance

# Enhanced EMA Strategy with Smart Money
SMART_EMA_STRUCTURE_VALIDATION = True           # Enable structure validation for EMA signals
SMART_EMA_ORDER_FLOW_VALIDATION = True          # Enable order flow validation for EMA signals
SMART_EMA_STRUCTURE_WEIGHT = 0.3                # Weight for structure in EMA strategy
SMART_EMA_ORDER_FLOW_WEIGHT = 0.2               # Weight for order flow in EMA strategy

# Enhanced MACD Strategy with Smart Money
SMART_MACD_ORDER_FLOW_VALIDATION = True         # Enable order flow validation for MACD signals
SMART_MACD_REQUIRE_OB_CONFLUENCE = False        # Require order block confluence for MACD
SMART_MACD_FVG_PROXIMITY_PIPS = 15              # FVG proximity requirement for MACD signals
SMART_MACD_ORDER_FLOW_BOOST = 1.2               # Confidence boost for order flow alignment
SMART_MACD_ORDER_FLOW_PENALTY = 0.8             # Confidence penalty for order flow misalignment

# =============================================================================
# INSTRUMENT-SPECIFIC PIP SIZING (CRITICAL FOR ACCURACY)
# =============================================================================

# Dynamic pip sizes for accurate calculations across all instruments
# These are used by the smart money analyzers for proper BOS/ChoCh detection
PIP_SIZES = {
    # Major FX Pairs (4-decimal quotes)
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'USDCAD': 0.0001, 'USDCHF': 0.0001,
    
    # JPY Pairs (2-decimal quotes)  
    'USDJPY': 0.01, 'EURJPY': 0.01, 'GBPJPY': 0.01, 'AUDJPY': 0.01,
    'NZDJPY': 0.01, 'CADJPY': 0.01, 'CHFJPY': 0.01,
    
    # Precious Metals
    'XAUUSD': 0.1,    # Gold
    'XAGUSD': 0.01,   # Silver
    
    # Major Indices
    'US500': 1.0,     # S&P 500
    'NAS100': 1.0,    # NASDAQ 100
    'UK100': 1.0,     # FTSE 100
    'GER40': 1.0,     # DAX 40
    'FRA40': 1.0,     # CAC 40
    'AUS200': 1.0,    # ASX 200
    
    # Energy
    'USOIL': 0.01,    # WTI Oil
    'UKOIL': 0.01,    # Brent Oil
    'NGAS': 0.001,    # Natural Gas
    
    # Crypto (if supported)
    'BTCUSD': 1.0,    # Bitcoin
    'ETHUSD': 0.1,    # Ethereum
}

# =============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# =============================================================================

# Smart Money Performance Limits
SMART_MONEY_MAX_CACHE_SIZE = 100                # Maximum cached analyses
SMART_MONEY_CACHE_TTL_SECONDS = 300             # Cache time-to-live (5 minutes)
SMART_MONEY_CONCURRENT_ANALYSIS_LIMIT = 5       # Maximum concurrent analyses
SMART_MONEY_MEMORY_LIMIT_MB = 100               # Memory limit for smart money analysis

# Analysis Frequency Controls
SMART_MONEY_ANALYSIS_FREQUENCY = 'on_signal'    # 'on_signal', 'continuous', 'scheduled'
SMART_MONEY_BATCH_SIZE = 10                     # Batch size for bulk analysis
SMART_MONEY_RETRY_ATTEMPTS = 3                  # Retry attempts for failed analysis

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Smart Money Logging Settings
SMART_MONEY_LOG_LEVEL = 'INFO'                  # DEBUG, INFO, WARNING, ERROR
SMART_MONEY_LOG_ANALYSIS_DETAILS = False        # Log detailed analysis steps
SMART_MONEY_LOG_PERFORMANCE_METRICS = True      # Log performance metrics
SMART_MONEY_LOG_CACHE_OPERATIONS = False        # Log cache hits/misses

# Debug Settings
SMART_MONEY_DEBUG_MODE = False                  # Enable debug mode
SMART_MONEY_SAVE_DEBUG_DATA = False             # Save debug data to files
SMART_MONEY_DEBUG_OUTPUT_DIR = "debug/smart_money"  # Debug output directory

# =============================================================================
# INTEGRATION SETTINGS
# =============================================================================

# Database Integration
SMART_MONEY_STORE_ANALYSIS = True               # Store analysis results in database
SMART_MONEY_STORE_RAW_DATA = False              # Store raw analysis data
SMART_MONEY_CLEANUP_OLD_DATA_DAYS = 30          # Days to keep old analysis data

# Alert System Integration
SMART_MONEY_INCLUDE_IN_ALERTS = True            # Include smart money data in alerts
SMART_MONEY_ALERT_THRESHOLD = 0.7               # Threshold for smart money alerts
SMART_MONEY_PRIORITY_BOOST = True               # Boost alert priority for strong SMC signals

# Claude AI Integration
SMART_MONEY_CLAUDE_ANALYSIS = True              # Send smart money context to Claude
SMART_MONEY_CLAUDE_WEIGHT = 0.2                 # Weight for smart money in Claude scoring

# =============================================================================
# STRATEGY-SPECIFIC SMART MONEY SETTINGS
# =============================================================================

# Combined Strategy Smart Money Integration
COMBINED_SMART_MONEY_ENABLED = True             # Enable smart money in combined strategy
COMBINED_SMART_MONEY_WEIGHT = 0.25              # Weight for smart money in combined scoring
COMBINED_SMART_MONEY_MINIMUM_SCORE = 0.5        # Minimum smart money score for combined signals

# Scalping Strategy Smart Money (if needed)
SCALPING_SMART_MONEY_ENABLED = False            # Disable for high-frequency scalping
SCALPING_SMART_MONEY_TIMEOUT = 2.0              # Faster timeout for scalping

# KAMA Strategy Smart Money Integration
KAMA_SMART_MONEY_ENABLED = True                 # Enable smart money for KAMA
KAMA_SMART_MONEY_ADAPTIVE_WEIGHT = True         # Adapt weight based on market conditions

# =============================================================================
# MARKET CONDITION ADAPTATIONS
# =============================================================================

# Session-Based Smart Money Settings
SMART_MONEY_SESSION_WEIGHTS = {
    'london': 1.0,      # Full weight during London session
    'new_york': 1.0,    # Full weight during New York session
    'asian': 0.7,       # Reduced weight during Asian session
    'overlap': 1.2      # Increased weight during session overlaps
}

# Volatility-Based Adjustments
SMART_MONEY_HIGH_VOLATILITY_THRESHOLD = 2.0     # ATR threshold for high volatility
SMART_MONEY_LOW_VOLATILITY_THRESHOLD = 0.5      # ATR threshold for low volatility
SMART_MONEY_VOLATILITY_WEIGHT_ADJUSTMENT = 0.1  # Weight adjustment for volatility

# News Event Handling
SMART_MONEY_NEWS_EVENT_PAUSE = True             # Pause analysis during major news
SMART_MONEY_NEWS_EVENT_TIMEOUT = 300            # Seconds to pause after news events

# =============================================================================
# FALLBACK AND ERROR HANDLING
# =============================================================================

# Error Handling Settings
SMART_MONEY_GRACEFUL_DEGRADATION = True         # Gracefully handle errors
SMART_MONEY_FALLBACK_TO_BASIC = True            # Fall back to basic analysis on errors
SMART_MONEY_ERROR_RETRY_DELAY = 60              # Seconds to wait before retry after error

# Fallback Values
SMART_MONEY_DEFAULT_SCORE = 0.5                 # Default score when analysis fails
SMART_MONEY_DEFAULT_BIAS = 'NEUTRAL'            # Default bias when analysis fails
SMART_MONEY_DEFAULT_CONFIDENCE_BOOST = 0.0      # Default confidence boost on errors

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

# Validation Settings
SMART_MONEY_VALIDATE_INPUTS = True              # Validate all inputs before analysis
SMART_MONEY_VALIDATE_OUTPUTS = True             # Validate all outputs after analysis
SMART_MONEY_STRICT_MODE = False                 # Enable strict validation mode

# Testing and Monitoring
SMART_MONEY_ENABLE_METRICS = True               # Enable performance metrics collection
SMART_MONEY_METRICS_INTERVAL = 300              # Metrics collection interval (seconds)
SMART_MONEY_HEALTH_CHECK_INTERVAL = 60          # Health check interval (seconds)

# A/B Testing Support
SMART_MONEY_AB_TESTING = False                  # Enable A/B testing
SMART_MONEY_AB_TEST_RATIO = 0.5                 # Ratio for A/B testing (0-1)
SMART_MONEY_AB_TEST_VARIANT = 'A'               # Current test variant ('A' or 'B')

# =============================================================================
# HELPER FUNCTIONS FOR CONFIGURATION
# =============================================================================

def get_pip_size_for_epic(epic: str) -> float:
    """
    Get the pip size for a given epic/symbol
    Used by smart money analyzers for accurate calculations
    """
    if not epic:
        return 0.0001
    
    # Clean up epic name (remove common prefixes/suffixes)
    clean_epic = epic.upper().replace('CS.D.', '').replace('.MINI.IP', '').replace('.DAILY.IP', '')
    
    # Direct lookup
    if clean_epic in PIP_SIZES:
        return PIP_SIZES[clean_epic]
    
    # Pattern matching for instruments not in direct lookup
    if any(jpy in clean_epic for jpy in ['JPY']):
        return 0.01
    elif any(metal in clean_epic for metal in ['XAU', 'GOLD']):
        return 0.1
    elif any(metal in clean_epic for metal in ['XAG', 'SILVER', 'SILV']):
        return 0.01
    elif any(index in clean_epic for index in ['US500', 'SPX', 'NAS', 'UK100', 'GER40', 'DAX']):
        return 1.0
    elif any(energy in clean_epic for energy in ['OIL', 'BRENT', 'WTI', 'NGAS']):
        return 0.01
    else:
        return 0.0001  # Default for major FX pairs

def get_smart_money_config_summary() -> dict:
    """
    Get a summary of smart money configuration for debugging
    """
    return {
        'enabled': SMART_MONEY_ENABLED,
        'readonly_mode': SMART_MONEY_READONLY_ENABLED,
        'version': SMART_MONEY_VERSION,
        'structure_settings': {
            'swing_lookback': STRUCTURE_SWING_LOOKBACK,
            'min_strength': STRUCTURE_MIN_SWING_STRENGTH,
            'bos_confirmation_pips': STRUCTURE_BOS_CONFIRMATION_PIPS,
            'require_close': STRUCTURE_REQUIRE_CLOSE
        },
        'order_flow_settings': {
            'min_ob_size': ORDER_FLOW_MIN_OB_SIZE_PIPS,
            'min_fvg_size': ORDER_FLOW_MIN_FVG_SIZE_PIPS,
            'displacement_factor': ORDER_FLOW_DISPLACEMENT_FACTOR
        },
        'performance_limits': {
            'min_data_points': SMART_MONEY_MIN_DATA_POINTS,
            'analysis_timeout': SMART_MONEY_ANALYSIS_TIMEOUT,
            'max_lookback': ORDER_FLOW_MAX_LOOKBACK_BARS
        },
        'weights': {
            'structure_weight': SMART_MONEY_STRUCTURE_WEIGHT,
            'order_flow_weight': SMART_MONEY_ORDER_FLOW_WEIGHT
        }
    }

# =============================================================================
# MIGRATION SETTINGS (for upgrading from previous versions)
# =============================================================================

# Migration Control
SMART_MONEY_MIGRATION_MODE = False              # Enable migration mode for upgrades
SMART_MONEY_BACKUP_OLD_DATA = True              # Backup data before migration
SMART_MONEY_MIGRATION_BATCH_SIZE = 1000         # Batch size for data migration

# Backwards Compatibility
SMART_MONEY_LEGACY_SUPPORT = True               # Support legacy configurations
SMART_MONEY_LEGACY_FALLBACK = True              # Fall back to legacy mode if needed