"""
Progressive Trailing Configuration System

This module provides optimized progressive trailing configurations based on:
1. Trade data analysis showing 100% win rate for heavily trailed trades
2. Pair-specific performance characteristics
3. Market volatility considerations
"""
from trailing_class import TrailingConfig, TrailingMethod
from config import (
    STAGE1_TRIGGER_POINTS, STAGE1_LOCK_POINTS,
    STAGE2_TRIGGER_POINTS, STAGE2_LOCK_POINTS,
    STAGE3_TRIGGER_POINTS, STAGE3_ATR_MULTIPLIER, STAGE3_MIN_DISTANCE,
    PROGRESSIVE_EPIC_SETTINGS
)

# Default progressive configuration (optimized for overall best performance)
DEFAULT_PROGRESSIVE_CONFIG = TrailingConfig(
    method=TrailingMethod.PROGRESSIVE_3_STAGE,
    break_even_trigger_points=STAGE1_TRIGGER_POINTS,
    min_trail_distance=2,
    max_trail_distance=50,
    monitor_interval_seconds=30,  # More frequent monitoring for progressive system

    # Progressive stage settings
    stage1_trigger_points=STAGE1_TRIGGER_POINTS,
    stage1_lock_points=STAGE1_LOCK_POINTS,
    stage2_trigger_points=STAGE2_TRIGGER_POINTS,
    stage2_lock_points=STAGE2_LOCK_POINTS,
    stage3_trigger_points=STAGE3_TRIGGER_POINTS,
    stage3_atr_multiplier=STAGE3_ATR_MULTIPLIER,
    stage3_min_distance=STAGE3_MIN_DISTANCE
)

# Balanced configuration (for major pairs like EUR/USD, GBP/USD) - OPTIMIZED FOR PROFIT CAPTURE
BALANCED_PROGRESSIVE_CONFIG = TrailingConfig(
    method=TrailingMethod.PROGRESSIVE_3_STAGE,
    break_even_trigger_points=6,  # Balanced approach (was 2)
    min_trail_distance=2,
    max_trail_distance=40,
    monitor_interval_seconds=25,  # Slightly less frequent monitoring

    # Balanced progressive settings - allow trends to develop
    stage1_trigger_points=6,  # Break-even at +6 points (was 2)
    stage1_lock_points=2,     # Better profit guarantee (was 1)
    stage2_trigger_points=10, # Profit lock at +10 points (was 4)
    stage2_lock_points=5,     # Better profit guarantee (was 2)
    stage3_trigger_points=18, # ATR trailing at +18 points (was 7)
    stage3_atr_multiplier=1.3,  # Slightly wider ATR for major pairs
    stage3_min_distance=2     # Minimum distance (was 1)
)

# Conservative configuration (for JPY pairs and volatile instruments)
CONSERVATIVE_PROGRESSIVE_CONFIG = TrailingConfig(
    method=TrailingMethod.PROGRESSIVE_3_STAGE,
    break_even_trigger_points=40,  # JPY: 40 points = ~4 real pips
    min_trail_distance=30,
    max_trail_distance=600,
    monitor_interval_seconds=45,

    # Conservative progressive settings - JPY calibrated
    stage1_trigger_points=40,  # Break-even at +40 JPY points (4 real pips)
    stage1_lock_points=10,     # Lock 10 JPY points (1 real pip)
    stage2_trigger_points=60,  # Profit lock at +60 JPY points (6 real pips)
    stage2_lock_points=30,     # Lock 30 JPY points (3 real pips)
    stage3_trigger_points=100, # ATR trailing at +100 JPY points (10 real pips)
    stage3_atr_multiplier=2.0, # Wider ATR for volatile pairs
    stage3_min_distance=30     # 30 JPY points (3 real pips)
)

# Configuration mapping based on epic performance data
PROGRESSIVE_CONFIG_MAP = {
    # Major pairs - use balanced settings for better profit capture
    'CS.D.EURUSD.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.GBPUSD.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.AUDUSD.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,
    'CS.D.USDCAD.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,

    # JPY pairs - use balanced settings (more practical trailing levels)
    'CS.D.USDJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.EURJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.GBPJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.AUDJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.NZDJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.CADJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,
    'CS.D.CHFJPY.MINI.IP': BALANCED_PROGRESSIVE_CONFIG,

    # Others - use default
    'CS.D.USDCHF.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,
    'CS.D.NZDUSD.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,
}

def get_progressive_config_for_epic(epic: str, candles=None, current_price: float = None, enable_adaptive: bool = True) -> TrailingConfig:
    """
    Get optimized progressive trailing configuration for a specific epic with adaptive market analysis.

    Args:
        epic: IG epic code (e.g., "CS.D.EURUSD.MINI.IP")
        candles: Recent candle data for market analysis (optional)
        current_price: Current market price (optional)
        enable_adaptive: Whether to use adaptive configuration (default: True)
    Returns:
        TrailingConfig: Optimized configuration for the epic and current market conditions
    """
    # Get base configuration
    base_config = PROGRESSIVE_CONFIG_MAP.get(epic, DEFAULT_PROGRESSIVE_CONFIG)

    # Apply epic-specific overrides from config.py first
    if epic in PROGRESSIVE_EPIC_SETTINGS:
        epic_overrides = PROGRESSIVE_EPIC_SETTINGS[epic]
        config_dict = base_config.__dict__.copy()

        if 'stage1_trigger' in epic_overrides:
            config_dict['stage1_trigger_points'] = epic_overrides['stage1_trigger']
            config_dict['break_even_trigger_points'] = epic_overrides['stage1_trigger']
        if 'stage2_trigger' in epic_overrides:
            config_dict['stage2_trigger_points'] = epic_overrides['stage2_trigger']
        if 'stage3_trigger' in epic_overrides:
            config_dict['stage3_trigger_points'] = epic_overrides['stage3_trigger']

        base_config = TrailingConfig(**config_dict)

    # If adaptive mode is disabled or no market data available, return base config
    if not enable_adaptive or not candles or current_price is None:
        # Store reason for skipping adaptive analysis
        # ✅ FIX: Defensive check to prevent Session object length error
        try:
            candles_count = len(candles) if candles else 0
        except TypeError:
            # If len() fails (e.g., Session object), treat as no candles
            candles_count = 0
            candles = None  # Reset to None to prevent further issues

        base_config._adaptive_skip_reason = f"adaptive={enable_adaptive}, candles={candles_count}, price={current_price is not None}"
        return base_config

    # Perform market analysis and get adaptive configuration
    try:
        # ✅ FIX: Defensive check to prevent Session object length error
        try:
            candles_count = len(candles)
        except TypeError:
            # If len() fails (e.g., Session object), treat as no candles
            print(f"[ADAPTIVE CONFIG ERROR] {epic}: Invalid candles parameter (not iterable), using base config")
            return base_config

        print(f"[ADAPTIVE DEBUG] {epic}: Starting market analysis with {candles_count} candles")
        market_context = analyze_market_regime(candles, current_price, epic)
        print(f"[ADAPTIVE DEBUG] {epic}: Market analysis complete: {market_context}")
        adaptive_config = get_adaptive_config_for_regime(base_config, market_context, epic)
        print(f"[ADAPTIVE DEBUG] {epic}: Adaptive config generated")

        # Store market context for logging
        adaptive_config._market_context = market_context

        return adaptive_config
    except Exception as e:
        # Fallback to base config if adaptive analysis fails
        print(f"[ADAPTIVE CONFIG ERROR] {epic}: {e}, using base config")
        import traceback
        print(f"[ADAPTIVE ERROR TRACE] {epic}: {traceback.format_exc()}")
        return base_config

def get_epic_performance_category(epic: str) -> str:
    """
    Categorize epic based on performance characteristics and optimal configuration.

    Returns:
        str: 'balanced', 'conservative', or 'default'
    """
    if epic in ['CS.D.EURUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP']:
        return 'balanced'  # Major pairs with balanced profit capture approach
    elif 'JPY' in epic:
        return 'conservative'  # JPY pairs need wider stops due to decimal structure
    else:
        return 'default'

# ================== ADAPTIVE MARKET CONDITION SYSTEM ==================

def analyze_market_regime(candles, current_price: float, epic: str) -> dict:
    """
    Comprehensive market regime analysis for adaptive trailing configuration.

    Returns:
        dict: Market context with volatility, trend, session, and regime classification
    """
    from datetime import datetime, timezone
    import statistics

    if not candles or len(candles) < 10:
        return {'regime': 'unknown', 'volatility': 'unknown', 'trend': 'unknown', 'session': 'unknown'}

    # Sort and get recent candles
    recent_candles = sorted(candles, key=lambda x: x.start_time)[-14:]  # Last 14 candles for ATR
    trend_candles = recent_candles[-10:]  # Last 10 for trend analysis

    # Calculate ATR (Average True Range)
    atr = calculate_atr(recent_candles)
    atr_ratio = (atr / current_price) * 100 if atr and current_price > 0 else 0

    # Classify volatility based on ATR
    volatility = classify_volatility(atr_ratio, epic)

    # Calculate trend strength and direction
    trend_analysis = analyze_trend_strength(trend_candles, current_price)

    # Determine trading session
    session = get_trading_session()

    # Classify overall market regime
    regime = classify_market_regime(volatility, trend_analysis['direction'], trend_analysis['strength'], session)

    return {
        'regime': regime,
        'volatility': volatility,
        'trend': trend_analysis['direction'],
        'trend_strength': trend_analysis['strength'],
        'session': session,
        'atr_ratio': atr_ratio,
        'atr_absolute': atr,
        'candle_count': len(recent_candles)
    }

def calculate_atr(candles, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(candles) < 2:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i-1]

        tr1 = current.high - current.low
        tr2 = abs(current.high - previous.close)
        tr3 = abs(current.low - previous.close)

        true_ranges.append(max(tr1, tr2, tr3))

    # Take last 'period' true ranges or all if less available
    recent_trs = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
    return sum(recent_trs) / len(recent_trs) if recent_trs else 0.0

def classify_volatility(atr_ratio: float, epic: str) -> str:
    """Classify market volatility based on ATR ratio"""
    # JPY pairs have different volatility characteristics
    if 'JPY' in epic:
        if atr_ratio > 0.8:
            return 'high'
        elif atr_ratio > 0.4:
            return 'medium'
        else:
            return 'low'
    else:
        # Non-JPY pairs
        if atr_ratio > 1.2:
            return 'high'
        elif atr_ratio > 0.6:
            return 'medium'
        else:
            return 'low'

def analyze_trend_strength(candles, current_price: float) -> dict:
    """Analyze trend direction and strength"""
    if len(candles) < 5:
        return {'direction': 'unknown', 'strength': 0.0}

    closes = [c.close for c in candles]

    # Linear regression slope for trend direction
    x_values = list(range(len(closes)))
    n = len(closes)

    sum_x = sum(x_values)
    sum_y = sum(closes)
    sum_xy = sum(x * y for x, y in zip(x_values, closes))
    sum_x2 = sum(x * x for x in x_values)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if n * sum_x2 - sum_x * sum_x != 0 else 0

    # Normalize slope to get strength (0-1)
    price_range = max(closes) - min(closes)
    normalized_slope = abs(slope) / (price_range / len(closes)) if price_range > 0 else 0
    strength = min(normalized_slope, 1.0)

    # Determine direction
    if slope > 0.0001:
        direction = 'up_trend'
    elif slope < -0.0001:
        direction = 'down_trend'
    else:
        direction = 'ranging'

    # Refine direction based on strength
    if strength > 0.7:
        direction = f"strong_{direction}" if direction != 'ranging' else 'ranging'
    elif strength > 0.3:
        direction = f"weak_{direction}" if direction != 'ranging' else 'ranging'
    else:
        direction = 'ranging'

    return {'direction': direction, 'strength': strength}

def get_trading_session() -> str:
    """Determine current trading session based on UTC time"""
    from datetime import datetime, timezone

    utc_time = datetime.now(timezone.utc)
    hour = utc_time.hour

    # Trading session classification (UTC times)
    if 0 <= hour < 6:
        return 'asian'
    elif 6 <= hour < 14:
        return 'london'
    elif 14 <= hour < 22:
        return 'ny'
    else:
        return 'asian'

def classify_market_regime(volatility: str, trend: str, trend_strength: float, session: str) -> str:
    """Classify overall market regime for configuration selection"""

    # High volatility regimes
    if volatility == 'high':
        if 'strong' in trend:
            return 'volatile_trending'
        else:
            return 'volatile_choppy'

    # Trending regimes
    elif 'strong' in trend and trend_strength > 0.6:
        return 'strong_trending'
    elif 'weak' in trend or ('up_trend' in trend or 'down_trend' in trend):
        return 'weak_trending'

    # Ranging/choppy regimes
    elif trend == 'ranging' or volatility == 'low':
        return 'ranging'

    # Default
    else:
        return 'neutral'

def get_adaptive_config_for_regime(base_config: TrailingConfig, market_context: dict, epic: str) -> TrailingConfig:
    """
    Adapt the trailing configuration based on market regime analysis.

    Returns:
        TrailingConfig: Dynamically adjusted configuration
    """
    regime = market_context.get('regime', 'neutral')
    volatility = market_context.get('volatility', 'medium')
    trend = market_context.get('trend', 'ranging')

    # Start with base configuration
    config_dict = base_config.__dict__.copy()

    # Regime-specific adaptations
    if regime == 'strong_trending':
        # Allow trends to develop - wider stops, higher targets
        config_dict['stage1_trigger_points'] = int(config_dict['stage1_trigger_points'] * 1.5)
        config_dict['stage2_trigger_points'] = int(config_dict['stage2_trigger_points'] * 1.4)
        config_dict['stage3_trigger_points'] = int(config_dict['stage3_trigger_points'] * 1.3)
        config_dict['stage3_atr_multiplier'] = config_dict['stage3_atr_multiplier'] * 1.2
        adaptation = "TREND_FOLLOWING"

    elif regime == 'volatile_trending':
        # Volatile but trending - wider ATR-based stops
        config_dict['stage3_atr_multiplier'] = config_dict['stage3_atr_multiplier'] * 1.5
        config_dict['min_trail_distance'] = int(config_dict['min_trail_distance'] * 1.3)
        adaptation = "VOLATILE_TREND"

    elif regime == 'ranging' or regime == 'volatile_choppy':
        # Choppy market - tighter stops, quicker profits
        config_dict['stage1_trigger_points'] = max(int(config_dict['stage1_trigger_points'] * 0.8), 3)
        config_dict['stage2_trigger_points'] = max(int(config_dict['stage2_trigger_points'] * 0.8), 5)
        config_dict['stage3_trigger_points'] = max(int(config_dict['stage3_trigger_points'] * 0.8), 8)
        config_dict['monitor_interval_seconds'] = max(int(config_dict['monitor_interval_seconds'] * 0.8), 15)
        adaptation = "SCALPING"

    elif regime == 'weak_trending':
        # Weak trend - slightly more conservative
        config_dict['stage1_trigger_points'] = int(config_dict['stage1_trigger_points'] * 1.1)
        config_dict['stage2_trigger_points'] = int(config_dict['stage2_trigger_points'] * 1.1)
        adaptation = "CONSERVATIVE"

    else:
        # Neutral/unknown - use base config
        adaptation = "BASE"

    # Volatility-specific adjustments
    if volatility == 'high':
        config_dict['min_trail_distance'] = int(config_dict['min_trail_distance'] * 1.2)
        config_dict['stage3_atr_multiplier'] = config_dict['stage3_atr_multiplier'] * 1.1
    elif volatility == 'low':
        config_dict['min_trail_distance'] = max(int(config_dict['min_trail_distance'] * 0.9), 2)

    # Store adaptation info for logging (temporarily)
    adaptation_info = adaptation
    regime_info = regime

    # ✅ FIX: Filter out metadata parameters before creating TrailingConfig
    # Remove any custom parameters that TrailingConfig constructor doesn't accept
    metadata_params = {'_adaptation', '_regime', '_market_context', '_adaptive_skip_reason'}
    filtered_config_dict = {k: v for k, v in config_dict.items() if k not in metadata_params}

    # Create the TrailingConfig with only valid parameters
    adaptive_config = TrailingConfig(**filtered_config_dict)

    # Add metadata as attributes after creation
    adaptive_config._adaptation = adaptation_info
    adaptive_config._regime = regime_info

    return adaptive_config

def log_progressive_config(config: TrailingConfig, epic: str, logger, market_context: dict = None) -> None:
    """Log the progressive configuration being used for transparency"""
    category = get_epic_performance_category(epic)

    # Check if adaptive analysis was skipped and why
    skip_reason = getattr(config, '_adaptive_skip_reason', None)
    if skip_reason:
        logger.info(f"⚠️ [ADAPTIVE SKIPPED] {epic}: {skip_reason}")

    # Enhanced logging with market context and adaptive decisions
    if market_context:
        logger.info(f"🌍 [MARKET ANALYSIS] {epic}:")
        logger.info(f"   🔄 Volatility: {market_context.get('volatility', 'unknown')} (ATR: {market_context.get('atr_ratio', 'N/A'):.4f}%)")
        logger.info(f"   📈 Trend: {market_context.get('trend', 'unknown')} (Strength: {market_context.get('trend_strength', 'N/A'):.3f})")
        logger.info(f"   🕐 Session: {market_context.get('session', 'unknown')}")
        logger.info(f"   🎯 Regime: {market_context.get('regime', 'unknown')}")

        # Show adaptation details if available
        adaptation = getattr(config, '_adaptation', None)
        regime = getattr(config, '_regime', None)
        if adaptation and regime:
            logger.info(f"🔧 [ADAPTIVE CONFIG] {epic}: {regime} → {adaptation} strategy")
            logger.info(f"   💡 Adaptation: Parameters adjusted for {regime} market conditions")

    config_type = getattr(config, '_adaptation', category.upper())
    logger.info(f"📊 [TRAILING CONFIG] {epic} ({config_type}):")
    logger.info(f"   • Stage 1: Break-even at +{config.stage1_trigger_points}pts → +{config.stage1_lock_points}pt profit")
    logger.info(f"   • Stage 2: Profit lock at +{config.stage2_trigger_points}pts → +{config.stage2_lock_points}pts profit")
    logger.info(f"   • Stage 3: ATR trailing at +{config.stage3_trigger_points}pts (ATR×{config.stage3_atr_multiplier:.1f})")
    logger.info(f"   • Monitor interval: {config.monitor_interval_seconds}s")