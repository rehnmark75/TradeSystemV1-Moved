# configdata/strategies/config_smc_strategy.py
"""
Smart Money Concepts (SMC) Strategy Configuration
Based on TradingView SMC-LuxAlgo script functionality

Features:
- Market Structure Analysis (BOS, ChoCH, Swing Points)
- Order Block Detection (Bullish & Bearish)
- Fair Value Gap (FVG) Identification
- Supply & Demand Zone Analysis
- Institutional Order Flow Analysis
- Multi-timeframe Structure Confirmation
"""

# Strategy enable/disable
SMC_STRATEGY = True

# Main configuration dictionary with multiple presets
SMC_STRATEGY_CONFIG = {
    'default': {
        # Market Structure Detection
        'swing_length': 5,          # Length for swing high/low detection
        'structure_confirmation': 3,  # Bars needed to confirm structure break
        'bos_threshold': 0.00015,   # Break of Structure threshold (price) - Increased for reliability
        'choch_threshold': 0.00015, # Change of Character threshold - Increased for reliability
        'min_structure_significance': 0.5,  # Minimum significance for valid structure break
        
        # Order Block Detection  
        'order_block_length': 3,    # Minimum length for order block
        'order_block_volume_factor': 1.8,  # Volume must be X times average - Increased
        'order_block_buffer': 2,    # Pips buffer around order block zones
        'max_order_blocks': 5,      # Maximum order blocks to track
        'order_block_min_strength': 0.6,  # Minimum strength for valid order block
        
        # Fair Value Gap Detection
        'fvg_min_size': 5,          # Minimum FVG size in pips - Increased
        'fvg_max_age': 20,          # Maximum bars to keep FVG active
        'fvg_fill_threshold': 0.4,  # Percentage filled to consider closed - More strict
        'fvg_volume_confirmation': True,  # Require volume confirmation for FVG
        
        # Supply/Demand Zones
        'zone_min_touches': 2,      # Minimum touches to create zone
        'zone_max_age': 50,         # Maximum bars to keep zone active
        'zone_strength_factor': 1.4, # Volume factor for strong zones - Increased
        
        # Liquidity Analysis (NEW)
        'liquidity_sweep_enabled': True,      # Enable liquidity sweep detection
        'equal_levels_tolerance': 0.5,        # Tolerance for equal highs/lows (pips)
        'min_liquidity_volume': 1.3,          # Min volume for liquidity events
        'liquidity_confirmation_bars': 2,     # Bars to confirm liquidity sweep
        
        # Premium/Discount Analysis (NEW)
        'premium_discount_enabled': True,     # Enable market maker model
        'daily_range_multiplier': 0.618,     # Golden ratio for optimal entry
        'weekly_range_multiplier': 0.382,    # Weekly range analysis
        'premium_threshold': 0.7,            # Threshold for premium zone
        'discount_threshold': 0.3,           # Threshold for discount zone
        
        # Signal Generation
        'confluence_required': 2.5, # Minimum confluence factors needed - Increased
        'min_risk_reward': 1.8,     # Minimum R:R ratio required - Increased  
        'max_distance_to_zone': 8,  # Maximum pips from entry to nearest zone - Reduced
        'min_confidence': 0.65,     # Minimum signal confidence - NEW
        
        # Multi-timeframe Settings
        'use_higher_tf': True,      # Use higher timeframe confirmation
        'higher_tf_multiplier': 4,  # Higher TF = current TF * multiplier
        'structure_alignment_required': True, # Require HTF structure alignment
        'mtf_confluence_weight': 0.8,        # Weight for multi-timeframe confluence
        
        # Session Analysis (NEW)
        'session_filtering_enabled': True,    # Enable session-based filtering
        'london_session_boost': 1.2,         # Confluence boost during London
        'ny_session_boost': 1.3,             # Confluence boost during NY
        'overlap_session_boost': 1.5,        # Boost during London/NY overlap
        'avoid_asian_session': True,         # Avoid signals during Asian session
        
        'description': 'Balanced SMC configuration for trending markets',
        'best_for': ['trending', 'institutional_flow', 'breakouts'],
        'best_volatility_regime': 'medium_to_high',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 10.0,
        'max_pip_volatility': 80.0
    },
    
    'moderate': {
        # Market Structure Detection - Relaxed but still quality
        'swing_length': 4,          # Shorter for more sensitivity
        'structure_confirmation': 2, # Faster confirmation
        'bos_threshold': 0.0001,    # More sensitive to breaks
        'choch_threshold': 0.0001,  # More sensitive to character changes
        'min_structure_significance': 0.4,  # Slightly lower bar
        
        # Order Block Detection - Balanced approach
        'order_block_length': 3,    # Keep standard length
        'order_block_volume_factor': 1.4,  # Relaxed volume requirement
        'order_block_buffer': 2,    # Keep buffer
        'max_order_blocks': 5,      # Keep tracking limit
        'order_block_min_strength': 0.4,  # Lower strength requirement
        'order_block_min_confidence': 0.3, # Lower confidence threshold
        
        # Fair Value Gap Detection - More inclusive
        'fvg_min_size': 3,          # Smaller gaps accepted
        'fvg_max_age': 25,          # Longer validity
        'fvg_fill_threshold': 0.5,  # Moderate fill requirement
        'fvg_volume_confirmation': True,  # Keep volume confirmation
        
        # Supply/Demand Zones - Moderate
        'zone_min_touches': 1,      # Single touch creates zone
        'zone_max_age': 40,         # Longer validity
        'zone_strength_factor': 1.2, # Relaxed strength requirement
        
        # Liquidity Analysis - Keep enabled but relaxed
        'liquidity_sweep_enabled': True,
        'equal_levels_tolerance': 1.0,        # More tolerance for equal levels
        'min_liquidity_volume': 1.2,          # Lower volume requirement
        'liquidity_confirmation_bars': 1,     # Faster confirmation
        
        # Premium/Discount Analysis - Keep but relaxed
        'premium_discount_enabled': True,
        'daily_range_multiplier': 0.618,     # Keep golden ratio
        'weekly_range_multiplier': 0.382,    # Keep weekly analysis
        'premium_threshold': 0.75,           # Slightly higher premium threshold
        'discount_threshold': 0.25,          # Slightly lower discount threshold
        
        # Signal Generation - CRITICAL ADJUSTMENTS
        'confluence_required': 1.0,    # REDUCED to match scalping success
        'min_risk_reward': 1.2,        # Even more accessible
        'max_distance_to_zone': 15,    # More distance allowed
        'min_confidence': 0.35,        # MAJOR REDUCTION for more signals
        
        # Multi-timeframe Settings - Relaxed
        'use_higher_tf': True,         # Keep multi-timeframe
        'higher_tf_multiplier': 4,     # Keep multiplier
        'structure_alignment_required': False, # DISABLE strict alignment
        'mtf_confluence_weight': 0.6,  # Reduced weight
        
        # Session Analysis - Relaxed
        'session_filtering_enabled': False,   # DISABLE session filtering
        'london_session_boost': 1.1,         # Smaller boost
        'ny_session_boost': 1.2,             # Smaller boost  
        'overlap_session_boost': 1.3,        # Smaller boost
        'avoid_asian_session': False,        # ALLOW Asian session
        
        'description': 'Moderate SMC configuration balancing quality with signal frequency',
        'best_for': ['balanced_trading', 'regular_signals', 'quality_frequency_balance'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending_or_ranging',
        'best_session': ['all_sessions'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 100.0
    },
    
    'conservative': {
        'swing_length': 7,
        'structure_confirmation': 5,
        'bos_threshold': 0.00015,
        'choch_threshold': 0.00015,
        'order_block_length': 5,
        'order_block_volume_factor': 2.0,
        'order_block_buffer': 3,
        'max_order_blocks': 3,
        'fvg_min_size': 5,
        'fvg_max_age': 15,
        'fvg_fill_threshold': 0.3,
        'zone_min_touches': 3,
        'zone_max_age': 30,
        'zone_strength_factor': 1.5,
        'confluence_required': 3,
        'min_risk_reward': 2.0,
        'max_distance_to_zone': 5,
        'use_higher_tf': True,
        'higher_tf_multiplier': 4,
        'structure_alignment_required': True,
        'description': 'Conservative SMC for high-confidence signals only',
        'best_for': ['low_volatility', 'risk_averse', 'news_safe'],
        'best_volatility_regime': 'low_to_medium',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 40.0
    },
    
    'aggressive': {
        'swing_length': 3,
        'structure_confirmation': 2,
        'bos_threshold': 0.00005,
        'choch_threshold': 0.00005,
        'order_block_length': 2,
        'order_block_volume_factor': 1.2,
        'order_block_buffer': 1,
        'max_order_blocks': 8,
        'fvg_min_size': 2,
        'fvg_max_age': 30,
        'fvg_fill_threshold': 0.7,
        'zone_min_touches': 1,
        'zone_max_age': 80,
        'zone_strength_factor': 1.1,
        'confluence_required': 1,
        'min_risk_reward': 1.2,
        'max_distance_to_zone': 15,
        'use_higher_tf': False,
        'higher_tf_multiplier': 3,
        'structure_alignment_required': False,
        'description': 'Aggressive SMC for high-frequency trading',
        'best_for': ['scalping', 'high_volatility', 'breakouts'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'any',
        'best_market_regime': 'any',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP', 'CS.D.AUDUSD.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 120.0
    },
    
    'scalping': {
        'swing_length': 2,
        'structure_confirmation': 1,
        'bos_threshold': 0.00003,
        'choch_threshold': 0.00003,
        'order_block_length': 1,
        'order_block_volume_factor': 1.1,
        'order_block_buffer': 0.5,
        'max_order_blocks': 10,
        'fvg_min_size': 1,
        'fvg_max_age': 40,
        'fvg_fill_threshold': 0.8,
        'zone_min_touches': 1,
        'zone_max_age': 100,
        'zone_strength_factor': 1.0,
        'confluence_required': 1,
        'min_risk_reward': 1.0,
        'max_distance_to_zone': 20,
        'use_higher_tf': False,
        'higher_tf_multiplier': 2,
        'structure_alignment_required': False,
        'description': 'High-frequency SMC scalping configuration',
        'best_for': ['scalping', 'quick_moves', 'high_frequency'],
        'best_volatility_regime': 'any',
        'best_trend_strength': 'any',
        'best_market_regime': 'any',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 200.0
    },
    
    'swing': {
        'swing_length': 10,
        'structure_confirmation': 8,
        'bos_threshold': 0.0003,
        'choch_threshold': 0.0003,
        'order_block_length': 8,
        'order_block_volume_factor': 2.5,
        'order_block_buffer': 5,
        'max_order_blocks': 3,
        'fvg_min_size': 8,
        'fvg_max_age': 10,
        'fvg_fill_threshold': 0.2,
        'zone_min_touches': 4,
        'zone_max_age': 20,
        'zone_strength_factor': 2.0,
        'confluence_required': 4,
        'min_risk_reward': 3.0,
        'max_distance_to_zone': 3,
        'use_higher_tf': True,
        'higher_tf_multiplier': 6,
        'structure_alignment_required': True,
        'description': 'Swing trading SMC for longer-term positions',
        'best_for': ['swing_trading', 'position_trading', 'weekly_holds'],
        'best_volatility_regime': 'low_to_medium',
        'best_trend_strength': 'very_strong',
        'best_market_regime': 'strongly_trending',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 15.0,
        'max_pip_volatility': 60.0
    },
    
    'news_safe': {
        'swing_length': 8,
        'structure_confirmation': 6,
        'bos_threshold': 0.0002,
        'choch_threshold': 0.0002,
        'order_block_length': 6,
        'order_block_volume_factor': 2.2,
        'order_block_buffer': 4,
        'max_order_blocks': 2,
        'fvg_min_size': 6,
        'fvg_max_age': 12,
        'fvg_fill_threshold': 0.25,
        'zone_min_touches': 3,
        'zone_max_age': 25,
        'zone_strength_factor': 1.8,
        'confluence_required': 4,
        'min_risk_reward': 2.5,
        'max_distance_to_zone': 4,
        'use_higher_tf': True,
        'higher_tf_multiplier': 5,
        'structure_alignment_required': True,
        'description': 'News-safe SMC avoiding high-impact events',
        'best_for': ['news_avoidance', 'stable_conditions', 'risk_management'],
        'best_volatility_regime': 'low',
        'best_trend_strength': 'medium_to_strong',
        'best_market_regime': 'trending',
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 30.0
    },
    
    'crypto': {
        'swing_length': 4,
        'structure_confirmation': 3,
        'bos_threshold': 0.0001,
        'choch_threshold': 0.0001,
        'order_block_length': 3,
        'order_block_volume_factor': 1.8,
        'order_block_buffer': 2,
        'max_order_blocks': 6,
        'fvg_min_size': 4,
        'fvg_max_age': 25,
        'fvg_fill_threshold': 0.6,
        'zone_min_touches': 2,
        'zone_max_age': 60,
        'zone_strength_factor': 1.4,
        'confluence_required': 2,
        'min_risk_reward': 1.8,
        'max_distance_to_zone': 12,
        'use_higher_tf': True,
        'higher_tf_multiplier': 3,
        'structure_alignment_required': False,
        'description': 'SMC adapted for crypto-style volatility',
        'best_for': ['crypto_pairs', 'high_volatility', '24_7_markets'],
        'best_volatility_regime': 'high',
        'best_trend_strength': 'any',
        'best_market_regime': 'any',
        'preferred_pairs': ['CS.D.BTCUSD.MINI.IP', 'CS.D.ETHUSD.MINI.IP'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 500.0
    }
}

# Active configuration selector  
ACTIVE_SMC_CONFIG = 'moderate'

# Individual feature toggles
SMC_MARKET_STRUCTURE_ENABLED = True
SMC_ORDER_BLOCKS_ENABLED = True
SMC_FAIR_VALUE_GAPS_ENABLED = True
SMC_SUPPLY_DEMAND_ZONES_ENABLED = True
SMC_MULTI_TIMEFRAME_ENABLED = True
SMC_LIQUIDITY_ANALYSIS_ENABLED = True

# Helper functions
def get_smc_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get SMC configuration for specific epic with fallbacks"""
    config = SMC_STRATEGY_CONFIG.get(market_condition, SMC_STRATEGY_CONFIG['default'])
    
    # Adjust for JPY pairs (higher pip values)
    if 'JPY' in epic.upper():
        config = config.copy()
        config['bos_threshold'] *= 100
        config['choch_threshold'] *= 100
        config['fvg_min_size'] *= 0.1  # JPY pips are different
        config['order_block_buffer'] *= 0.1
        config['max_distance_to_zone'] *= 0.1
    
    return config

def get_smc_threshold_for_epic(epic: str) -> float:
    """Get SMC-specific thresholds based on currency pair"""
    if 'JPY' in epic.upper():
        return 0.01  # JPY pairs use different pip scale
    elif any(pair in epic.upper() for pair in ['GBP', 'AUD', 'NZD']):
        return 0.00008  # Slightly higher for volatile pairs
    else:
        return 0.00005  # Default for EUR, USD majors

def get_smc_confluence_factors() -> list:
    """Get available confluence factors for SMC analysis"""
    return [
        'market_structure_break',    # BOS or ChoCH
        'order_block_presence',      # Order block at key level
        'fair_value_gap',           # FVG supporting direction
        'supply_demand_zone',       # Strong S/D zone
        'liquidity_sweep',          # Liquidity grab detected
        'higher_tf_alignment',      # Higher timeframe structure
        'volume_confirmation',      # Volume supporting move
        'equal_highs_lows'         # Equal H/L liquidity
    ]

def validate_smc_config() -> dict:
    """Validate SMC strategy configuration completeness"""
    try:
        required_keys = ['SMC_STRATEGY', 'SMC_STRATEGY_CONFIG', 'ACTIVE_SMC_CONFIG']
        
        for key in required_keys:
            if not globals().get(key):
                return {'valid': False, 'error': f'Missing {key}'}
        
        # Validate active config exists
        active_config = globals().get('ACTIVE_SMC_CONFIG')
        if active_config not in SMC_STRATEGY_CONFIG:
            return {'valid': False, 'error': f'Active config "{active_config}" not found in SMC_STRATEGY_CONFIG'}
        
        # Validate config structure
        required_config_keys = [
            'swing_length', 'structure_confirmation', 'bos_threshold', 'choch_threshold',
            'order_block_length', 'fvg_min_size', 'confluence_required', 'min_risk_reward'
        ]
        
        for config_name, config_data in SMC_STRATEGY_CONFIG.items():
            for key in required_config_keys:
                if key not in config_data:
                    return {'valid': False, 'error': f'Config "{config_name}" missing key "{key}"'}
        
        # Validate feature flags
        feature_flags = [
            'SMC_MARKET_STRUCTURE_ENABLED', 'SMC_ORDER_BLOCKS_ENABLED', 
            'SMC_FAIR_VALUE_GAPS_ENABLED', 'SMC_SUPPLY_DEMAND_ZONES_ENABLED'
        ]
        
        for flag in feature_flags:
            if globals().get(flag) is None:
                return {'valid': False, 'error': f'Missing feature flag {flag}'}
        
        return {
            'valid': True, 
            'config_count': len(SMC_STRATEGY_CONFIG),
            'active_config': active_config,
            'features_enabled': sum(1 for flag in feature_flags if globals().get(flag, False))
        }
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}

# Risk management presets
SMC_RISK_PROFILES = {
    'conservative': {
        'max_risk_per_trade': 0.5,      # % of account
        'max_concurrent_trades': 2,      # Maximum positions
        'profit_target_ratio': 2.0,     # R:R ratio
        'stop_loss_buffer': 5,          # Pips beyond zone
        'breakeven_trigger': 1.0,       # Move to BE at 1:1
        'partial_profit_level': 1.5,    # Take 50% at 1.5:1
    },
    'balanced': {
        'max_risk_per_trade': 1.0,
        'max_concurrent_trades': 3,
        'profit_target_ratio': 1.5,
        'stop_loss_buffer': 3,
        'breakeven_trigger': 0.8,
        'partial_profit_level': 1.2,
    },
    'aggressive': {
        'max_risk_per_trade': 2.0,
        'max_concurrent_trades': 5,
        'profit_target_ratio': 1.2,
        'stop_loss_buffer': 2,
        'breakeven_trigger': 0.5,
        'partial_profit_level': 1.0,
    }
}

ACTIVE_SMC_RISK_PROFILE = 'balanced'