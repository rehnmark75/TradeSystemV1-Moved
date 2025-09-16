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

# High-performance configuration (for major pairs like EUR/USD, GBP/USD)
AGGRESSIVE_PROGRESSIVE_CONFIG = TrailingConfig(
    method=TrailingMethod.PROGRESSIVE_3_STAGE,
    break_even_trigger_points=2,
    min_trail_distance=1,
    max_trail_distance=40,
    monitor_interval_seconds=20,  # Even more frequent for aggressive pairs

    # More aggressive progressive settings
    stage1_trigger_points=2,  # Break-even at +2 points
    stage1_lock_points=1,
    stage2_trigger_points=4,  # Profit lock at +4 points
    stage2_lock_points=2,
    stage3_trigger_points=7,  # ATR trailing at +7 points
    stage3_atr_multiplier=1.2,  # Tighter ATR for major pairs
    stage3_min_distance=1
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
    # High performers - use aggressive settings
    'CS.D.EURUSD.MINI.IP': AGGRESSIVE_PROGRESSIVE_CONFIG,
    'CS.D.GBPUSD.MINI.IP': AGGRESSIVE_PROGRESSIVE_CONFIG,
    'CS.D.AUDUSD.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,
    'CS.D.USDCAD.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,

    # JPY pairs - use conservative settings (calibrated for 2-decimal pip structure)
    'CS.D.USDJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,
    'CS.D.EURJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,
    'CS.D.GBPJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,
    'CS.D.AUDJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,
    'CS.D.NZDJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,
    'CS.D.CADJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,
    'CS.D.CHFJPY.MINI.IP': CONSERVATIVE_PROGRESSIVE_CONFIG,

    # Others - use default
    'CS.D.USDCHF.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,
    'CS.D.NZDUSD.MINI.IP': DEFAULT_PROGRESSIVE_CONFIG,
}

def get_progressive_config_for_epic(epic: str) -> TrailingConfig:
    """
    Get optimized progressive trailing configuration for a specific epic.

    Args:
        epic: IG epic code (e.g., "CS.D.EURUSD.MINI.IP")

    Returns:
        TrailingConfig: Optimized configuration for the epic
    """
    config = PROGRESSIVE_CONFIG_MAP.get(epic, DEFAULT_PROGRESSIVE_CONFIG)

    # Apply any epic-specific overrides from config.py
    if epic in PROGRESSIVE_EPIC_SETTINGS:
        epic_overrides = PROGRESSIVE_EPIC_SETTINGS[epic]

        # Create a copy of the config with overrides
        config_dict = config.__dict__.copy()

        # Apply overrides
        if 'stage1_trigger' in epic_overrides:
            config_dict['stage1_trigger_points'] = epic_overrides['stage1_trigger']
            config_dict['break_even_trigger_points'] = epic_overrides['stage1_trigger']

        if 'stage2_trigger' in epic_overrides:
            config_dict['stage2_trigger_points'] = epic_overrides['stage2_trigger']

        if 'stage3_trigger' in epic_overrides:
            config_dict['stage3_trigger_points'] = epic_overrides['stage3_trigger']

        # Create new config with overrides
        return TrailingConfig(**config_dict)

    return config

def get_epic_performance_category(epic: str) -> str:
    """
    Categorize epic based on historical performance data.

    Returns:
        str: 'aggressive', 'conservative', or 'default'
    """
    if epic in ['CS.D.EURUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP']:
        return 'aggressive'  # High performance pairs
    elif 'JPY' in epic:
        return 'conservative'  # JPY pairs need wider stops
    else:
        return 'default'

def log_progressive_config(config: TrailingConfig, epic: str, logger) -> None:
    """Log the progressive configuration being used for transparency"""
    category = get_epic_performance_category(epic)

    logger.info(f"📊 [PROGRESSIVE CONFIG] {epic} ({category.upper()}):")
    logger.info(f"   • Stage 1: Break-even at +{config.stage1_trigger_points}pts → +{config.stage1_lock_points}pt profit")
    logger.info(f"   • Stage 2: Profit lock at +{config.stage2_trigger_points}pts → +{config.stage2_lock_points}pts profit")
    logger.info(f"   • Stage 3: ATR trailing at +{config.stage3_trigger_points}pts (ATR×{config.stage3_atr_multiplier})")
    logger.info(f"   • Monitor interval: {config.monitor_interval_seconds}s")