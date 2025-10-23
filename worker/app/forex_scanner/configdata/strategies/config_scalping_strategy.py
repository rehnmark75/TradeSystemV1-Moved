# configdata/strategies/config_scalping_strategy.py
"""
Scalping Strategy Configuration
===============================

High-frequency trading strategy designed for capturing small price movements
within short time frames. Optimized for tight spreads, high liquidity pairs,
and quick entry/exit signals.

Key Features:
- Multiple scalping configurations (ultra_fast, aggressive, conservative, dual_ma)
- Session-based trading controls
- Risk management with position sizing
- Market condition filtering
- Time-based session optimization
"""

# Strategy enable/disable
SCALPING_STRATEGY_ENABLED = True    # 🔥 ENABLED: Linda Raschke MACD 3-10-16 adaptive scalping
SCALPING_MODE = 'linda_raschke'     # Active scalping mode: Linda Raschke MACD 3-10-16

# Scalping Strategy Configurations
SCALPING_STRATEGY_CONFIG = {
    'ultra_fast': {
        'fast_ema': 3,          # Ultra-responsive
        'slow_ema': 8,          # Quick confirmation
        'filter_ema': 21,       # Trend filter
        'timeframes': ['1m'],   # 1-minute scalping only
        'target_pips': 3,       # Small profit targets
        'stop_loss_pips': 2,    # Tight stops
        'max_spread_pips': 1.5, # Maximum allowed spread
        'description': 'Ultra-fast 1-minute scalping with 3/8 EMA crossover',
        'best_for': ['high_liquidity', 'tight_spreads', 'trending_intraday'],
        'best_volatility_regime': 'medium_to_high',
        'best_trend_strength': 'any',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york', 'overlap'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
        'min_pip_volatility': 10.0,
        'max_pip_volatility': 40.0
    },
    'linda_raschke': {
        # 🔥 LINDA RASCHKE MACD 3-10-16 OSCILLATOR
        'macd_fast': 3,         # 3 SMA (NOT EMA!) - Ultra responsive
        'macd_slow': 10,        # 10 SMA (NOT EMA!) - Quick confirmation
        'macd_signal': 16,      # 16 SMA (NOT EMA!) - Smoothing
        'fast_ema': 5,          # EMA for fallback/standard mode
        'slow_ema': 13,         # EMA for fallback/standard mode
        'filter_ema': None,     # No filter - adaptive regime detection
        'timeframes': ['5m'],   # 5-minute optimal for Linda Raschke scalping
        'target_pips': 8,       # Linda Raschke: Quick 8-12 pip targets
        'stop_loss_pips': 6,    # Tight stops for scalping
        'max_spread_pips': 2.0, # Moderate spread tolerance
        'max_bars': 24,         # 2 hours timeout (24 bars * 5min)
        'time_exit_hours': 2.0, # Force exit after 2 hours if no direction
        'breakeven_trigger': 4.0, # Move to BE at 4 pips (50% of target)
        'description': 'Linda Raschke MACD 3-10-16 adaptive scalping with regime detection',
        'signal_types': ['macd_zero_cross', 'macd_signal_cross', 'macd_momentum', 'anti_pattern'],
        'best_for': ['adaptive_trending', 'momentum_continuation', 'pullback_entries'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'any',  # Adaptive to all regimes
        'best_market_regime': 'adaptive',  # Detects trending/ranging dynamically
        'best_session': ['london', 'new_york', 'tokyo'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP',
                           'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 80.0
    },
    'aggressive': {
        'fast_ema': 5,          # Fast but not too noisy
        'slow_ema': 13,         # Fibonacci number
        'filter_ema': 50,       # Medium-term trend filter
        'timeframes': ['1m', '5m'],  # Multi-timeframe
        'target_pips': 5,       # Reasonable profit targets
        'stop_loss_pips': 3,    # Conservative stops
        'max_spread_pips': 2.0, # Moderate spread tolerance
        'description': 'Aggressive scalping with 5/13 EMA crossover',
        'best_for': ['breakouts', 'news_events', 'volatility_spikes'],
        'best_volatility_regime': 'medium_to_high',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 60.0
    },
    'conservative': {
        'fast_ema': 8,          # More stable
        'slow_ema': 20,         # Classic combination
        'filter_ema': 50,       # Trend filter
        'timeframes': ['5m'],   # 5-minute focus
        'target_pips': 8,       # Larger profit targets
        'stop_loss_pips': 5,    # Wider stops
        'max_spread_pips': 2.5, # More spread tolerance
        'description': 'Conservative scalping with 8/20 EMA crossover',
        'best_for': ['stable_trends', 'lower_volatility', 'risk_averse'],
        'best_volatility_regime': 'low_to_medium',
        'best_trend_strength': 'strong',
        'best_market_regime': 'trending',
        'best_session': ['london'],
        'preferred_pairs': ['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
        'min_pip_volatility': 5.0,
        'max_pip_volatility': 30.0
    },
    'dual_ma': {
        'fast_ema': 7,          # Research-based optimal
        'slow_ema': 14,         # Research-based optimal
        'filter_ema': None,     # No filter for simplicity
        'timeframes': ['1m', '5m'],  # Multi-timeframe
        'target_pips': 5,       # Standard profit targets
        'stop_loss_pips': 3,    # Standard stops
        'max_spread_pips': 2.0, # Standard spread tolerance
        'description': 'Simple dual MA crossover scalping (7/14)',
        'best_for': ['simple_execution', 'clear_trends', 'automation'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending'
    },
    'news_trader': {
        'fast_ema': 4,          # Very responsive for news
        'slow_ema': 9,          # Quick confirmation
        'filter_ema': 20,       # Short-term trend filter
        'timeframes': ['1m'],   # 1-minute for news events
        'target_pips': 8,       # Larger targets for volatility
        'stop_loss_pips': 4,    # Wider stops for volatility
        'max_spread_pips': 3.0, # Higher spread tolerance during news
        'description': 'News event scalping with volatility expansion',
        'best_for': ['news_events', 'high_volatility', 'breakouts'],
        'best_session': ['overlap', 'news_times']
    },
    'range_bound': {
        'fast_ema': 12,         # Slower for ranging markets
        'slow_ema': 26,         # Wider separation
        'filter_ema': 100,      # Long-term trend filter
        'timeframes': ['5m', '15m'],  # Longer timeframes
        'target_pips': 6,       # Moderate targets
        'stop_loss_pips': 4,    # Moderate stops
        'max_spread_pips': 2.0, # Standard spread tolerance
        'description': 'Range-bound market scalping with wider EMAs',
        'best_for': ['ranging_markets', 'support_resistance', 'consolidation']
    }
}

# Active scalping configuration
ACTIVE_SCALPING_CONFIG = SCALPING_MODE
SCALPING_TIMEFRAME = '5m'  # Primary scalping timeframe

# Dynamic configuration management
ENABLE_DYNAMIC_SCALPING_CONFIG = True   # Enable dynamic scalping configuration selection
SCALPING_ADAPTIVE_SIZING = True         # Enable adaptive position sizing

# Scalping Risk Management
SCALPING_RISK_MANAGEMENT = {
    'max_trades_per_hour': 20,          # Prevent overtrading
    'max_daily_trades': 100,            # Daily limit
    'max_consecutive_losses': 3,        # Stop after 3 losses
    'min_profit_ratio': 1.5,           # Minimum 1.5:1 reward:risk
    'position_size_percent': 0.5,      # Smaller positions for scalping (0.5% of account)
    'enable_quick_exit': True,          # Enable rapid exit signals
    'max_position_hold_minutes': 15,    # Maximum position hold time
    'break_even_after_pips': 3,         # Move stop to breakeven after 3 pips profit
    'trailing_stop_enabled': True,      # Enable trailing stops
    'trailing_stop_distance': 2,        # 2 pip trailing stop distance

    # Session-based limits
    'session_limits': {
        'london': {
            'start': 8,             # 8:00 UTC
            'end': 17,              # 17:00 UTC
            'max_trades': 40,       # Maximum trades during London session
            'preferred': True       # London session is preferred for scalping
        },
        'new_york': {
            'start': 13,            # 13:00 UTC
            'end': 22,              # 22:00 UTC
            'max_trades': 35,       # Maximum trades during NY session
            'preferred': True       # NY session is preferred for scalping
        },
        'london_ny_overlap': {
            'start': 13,            # 13:00 UTC
            'end': 17,              # 17:00 UTC
            'max_trades': 25,       # Maximum trades during overlap (best time)
            'preferred': True,      # Overlap is the best time for scalping
            'confidence_boost': 0.1 # 10% confidence boost during overlap
        },
        'asian': {
            'start': 0,             # 00:00 UTC
            'end': 9,               # 09:00 UTC
            'max_trades': 15,       # Lower limit during Asian session
            'preferred': False      # Asian session is less preferred
        }
    },

    # Risk controls
    'daily_loss_limit': 200,            # Daily loss limit in account currency
    'consecutive_loss_limit': 50,       # Stop trading after this loss amount
    'drawdown_limit_percent': 3,        # Stop trading at 3% account drawdown
    'cooldown_after_losses': 30         # 30-minute cooldown after max consecutive losses
}

# Scalping Market Conditions
SCALPING_MARKET_CONDITIONS = {
    'min_volume_ratio': 1.2,            # Minimum volume vs average (120%)
    'max_volatility_atr': 0.005,        # Maximum ATR to avoid excessive volatility (0.5%)
    'min_volatility_atr': 0.0008,       # Minimum ATR to ensure movement (0.08%)

    # Preferred pairs for scalping (tight spreads, high liquidity)
    'preferred_pairs': [
        'CS.D.EURUSD.CEEM.IP',          # EUR/USD - most liquid, tight spreads
        'CS.D.GBPUSD.MINI.IP',          # GBP/USD - good volatility, decent spreads
        'CS.D.USDJPY.MINI.IP',          # USD/JPY - popular scalping pair
        'CS.D.AUDUSD.MINI.IP'           # AUD/USD - reasonable spreads, good movement
    ],

    # Avoid these pairs for scalping (wide spreads, low liquidity)
    'avoid_pairs': [
        'CS.D.NZDUSD.MINI.IP',          # NZD/USD - wider spreads
        'CS.D.USDCHF.MINI.IP'           # USD/CHF - lower volatility
    ],

    'avoid_news_minutes': 30,           # Avoid trading 30min before/after major news
    'preferred_sessions': ['london', 'london_new_york_overlap', 'new_york'],
    'max_spread_multiplier': 2.0,       # Maximum spread as multiple of normal spread

    # Market regime filters
    'min_trend_strength': 0.3,          # Minimum trend strength (ADX-like measure)
    'max_ranging_threshold': 0.8,       # Maximum ranging condition threshold
    'volume_surge_threshold': 2.0,      # Volume surge detection (2x normal)

    # Time-based filters
    'avoid_rollover_hours': [22, 23, 0], # Avoid these UTC hours (rollover time)
    'optimal_hours': [8, 9, 10, 13, 14, 15, 16], # Best UTC hours for scalping

    # Economic calendar integration
    'avoid_high_impact_news': True,     # Avoid trading during high impact news
    'news_buffer_minutes': 15,          # Buffer around news events
    'resume_after_news_minutes': 10     # Resume trading X minutes after news
}

# Scalping Signal Quality Settings
SCALPING_SIGNAL_QUALITY = {
    'min_confidence': 0.4,              # Lower confidence for scalping (40%)
    'require_volume_confirmation': True, # Require volume confirmation
    'min_volume_spike': 1.3,            # Minimum volume spike (130% of average)
    'require_momentum_alignment': True,  # Require momentum indicators to align
    'max_signal_age_seconds': 60,       # Maximum signal age for execution (1 minute)

    # EMA alignment requirements
    'require_ema_alignment': True,       # Require EMAs to be aligned with signal
    'min_ema_separation_pips': 1,        # Minimum separation between EMAs
    'require_price_above_filter_ema': True, # Price must be above/below filter EMA

    # Momentum filters
    'macd_histogram_filter': True,       # Use MACD histogram for momentum
    'rsi_overbought': 75,               # RSI overbought level
    'rsi_oversold': 25,                 # RSI oversold level
    'avoid_rsi_extremes': True,         # Avoid trading at RSI extremes

    # Price action filters
    'min_candle_body_ratio': 0.5,       # Minimum candle body to wick ratio
    'require_strong_close': True,       # Require close near high/low
    'max_wick_ratio': 0.3,             # Maximum wick to body ratio

    # Confluence requirements
    'min_confluence_factors': 2,        # Minimum confluence factors required
    'confluence_bonus': 0.1             # Confidence bonus for additional confluence
}

# Strategy weight and integration
STRATEGY_WEIGHT_SCALPING = 0.0         # Weight in combined strategies (disabled by default)
SCALPING_ALLOW_COMBINED = False        # Don't allow in combined strategies (standalone only)
SCALPING_PRIORITY_LEVEL = 3            # Priority level (1=highest, 5=lowest)

# Performance settings
SCALPING_ENABLE_BACKTESTING = True     # Enable strategy in backtests
SCALPING_MIN_DATA_PERIODS = 200        # Minimum data periods required
SCALPING_ENABLE_PERFORMANCE_TRACKING = True # Track strategy performance

# Debug settings
SCALPING_DEBUG_LOGGING = True          # Enable detailed debug logging
SCALPING_LOG_ENTRY_EXIT = True         # Log all entry/exit decisions
SCALPING_LOG_RISK_CALCULATIONS = True  # Log risk management calculations

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_scalping_config(config_name: str = None) -> dict:
    """
    Get the currently active or specified scalping configuration

    Args:
        config_name: Optional specific config name to retrieve

    Returns:
        Dictionary with scalping configuration
    """
    if config_name and config_name in SCALPING_STRATEGY_CONFIG:
        return SCALPING_STRATEGY_CONFIG[config_name]

    return SCALPING_STRATEGY_CONFIG.get(ACTIVE_SCALPING_CONFIG, SCALPING_STRATEGY_CONFIG['aggressive'])

def get_scalping_timeframes(config_name: str = None) -> list:
    """
    Get preferred timeframes for active or specified scalping config

    Args:
        config_name: Optional specific config name

    Returns:
        List of preferred timeframes
    """
    config = get_scalping_config(config_name)
    return config.get('timeframes', ['5m'])

def is_scalping_session(current_hour: int = None) -> dict:
    """
    Check if current time is good for scalping

    Args:
        current_hour: Optional hour to check (UTC), uses current time if None

    Returns:
        Dictionary with session information
    """
    if current_hour is None:
        from datetime import datetime
        import pytz
        now = datetime.now(pytz.UTC)
        current_hour = now.hour

    session_info = {
        'is_scalping_time': False,
        'current_session': None,
        'max_trades': 0,
        'confidence_boost': 0.0,
        'session_preference': 'not_preferred'
    }

    # Check each session
    sessions = SCALPING_RISK_MANAGEMENT['session_limits']

    for session_name, session_config in sessions.items():
        start_hour = session_config['start']
        end_hour = session_config['end']

        # Handle session that crosses midnight
        if start_hour <= end_hour:
            in_session = start_hour <= current_hour <= end_hour
        else:
            in_session = current_hour >= start_hour or current_hour <= end_hour

        if in_session:
            session_info['is_scalping_time'] = True
            session_info['current_session'] = session_name
            session_info['max_trades'] = session_config['max_trades']
            session_info['confidence_boost'] = session_config.get('confidence_boost', 0.0)
            session_info['session_preference'] = 'preferred' if session_config.get('preferred', False) else 'acceptable'
            break

    return session_info

def get_scalping_position_size(account_balance: float, risk_per_trade: float, stop_loss_pips: int, pip_value: float = 1.0) -> float:
    """
    Calculate optimal position size for scalping

    Args:
        account_balance: Account balance in account currency
        risk_per_trade: Risk per trade as percentage (e.g., 0.5 for 0.5%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value per pip for the instrument

    Returns:
        Position size in lots
    """
    try:
        # Calculate position size using risk management formula
        risk_amount = account_balance * (risk_per_trade / 100)
        position_size = risk_amount / (stop_loss_pips * pip_value)

        # Apply scalping position size multiplier
        scalping_multiplier = SCALPING_RISK_MANAGEMENT['position_size_percent'] / 100
        adjusted_position_size = position_size * scalping_multiplier

        # Ensure minimum and maximum position sizes
        min_size = 0.01  # 0.01 lots minimum
        max_size = account_balance * 0.01  # Maximum 1% of account balance

        return max(min_size, min(adjusted_position_size, max_size))

    except Exception:
        return 0.01  # Default minimum position size

def is_valid_scalping_pair(epic: str) -> dict:
    """
    Check if a pair is suitable for scalping

    Args:
        epic: Trading pair epic

    Returns:
        Dictionary with validation results
    """
    preferred = epic in SCALPING_MARKET_CONDITIONS['preferred_pairs']
    avoided = epic in SCALPING_MARKET_CONDITIONS['avoid_pairs']

    return {
        'is_valid': preferred and not avoided,
        'is_preferred': preferred,
        'is_avoided': avoided,
        'recommendation': 'highly_recommended' if preferred else ('avoid' if avoided else 'acceptable')
    }

def get_scalping_risk_limits() -> dict:
    """Get current scalping risk management limits"""
    return SCALPING_RISK_MANAGEMENT.copy()

def validate_scalping_config() -> dict:
    """
    Validate scalping strategy configuration completeness

    Returns:
        Dictionary with validation results
    """
    try:
        # Check required settings
        required_settings = [
            'SCALPING_STRATEGY_ENABLED', 'SCALPING_STRATEGY_CONFIG', 'ACTIVE_SCALPING_CONFIG',
            'SCALPING_RISK_MANAGEMENT', 'SCALPING_MARKET_CONDITIONS'
        ]

        for setting in required_settings:
            if setting not in globals():
                return {'valid': False, 'error': f'Missing required setting: {setting}'}

        # Validate configuration structure
        for config_name, config in SCALPING_STRATEGY_CONFIG.items():
            if not isinstance(config, dict):
                return {'valid': False, 'error': f'Config {config_name} must be dict'}

            required_keys = ['fast_ema', 'slow_ema', 'target_pips', 'stop_loss_pips', 'description']
            for key in required_keys:
                if key not in config:
                    return {'valid': False, 'error': f'Config {config_name} missing {key}'}

            # Validate EMA order
            if config['fast_ema'] >= config['slow_ema']:
                return {'valid': False, 'error': f'Config {config_name}: fast_ema must be < slow_ema'}

            # Validate risk/reward
            if config['target_pips'] <= config['stop_loss_pips']:
                return {'valid': False, 'error': f'Config {config_name}: target_pips should be > stop_loss_pips'}

        # Validate active config exists
        if ACTIVE_SCALPING_CONFIG not in SCALPING_STRATEGY_CONFIG:
            return {'valid': False, 'error': f'Active config {ACTIVE_SCALPING_CONFIG} not found'}

        # Validate risk management settings
        risk_mgmt = SCALPING_RISK_MANAGEMENT
        if not (0 < risk_mgmt['position_size_percent'] <= 5):
            return {'valid': False, 'error': 'position_size_percent must be between 0 and 5'}

        if not (1.0 <= risk_mgmt['min_profit_ratio'] <= 10.0):
            return {'valid': False, 'error': 'min_profit_ratio must be between 1.0 and 10.0'}

        return {
            'valid': True,
            'strategy_enabled': SCALPING_STRATEGY_ENABLED,
            'config_count': len(SCALPING_STRATEGY_CONFIG),
            'active_config': ACTIVE_SCALPING_CONFIG,
            'dynamic_config_enabled': ENABLE_DYNAMIC_SCALPING_CONFIG,
            'preferred_pairs_count': len(SCALPING_MARKET_CONDITIONS['preferred_pairs']),
            'session_count': len(SCALPING_RISK_MANAGEMENT['session_limits'])
        }

    except Exception as e:
        return {'valid': False, 'error': f'Validation exception: {str(e)}'}

def get_scalping_config_summary() -> dict:
    """Get a summary of scalping configuration settings"""
    return {
        'strategy_enabled': SCALPING_STRATEGY_ENABLED,
        'active_config': ACTIVE_SCALPING_CONFIG,
        'dynamic_config_enabled': ENABLE_DYNAMIC_SCALPING_CONFIG,
        'primary_timeframe': SCALPING_TIMEFRAME,
        'total_configurations': len(SCALPING_STRATEGY_CONFIG),
        'available_configs': list(SCALPING_STRATEGY_CONFIG.keys()),
        'max_trades_per_hour': SCALPING_RISK_MANAGEMENT['max_trades_per_hour'],
        'max_daily_trades': SCALPING_RISK_MANAGEMENT['max_daily_trades'],
        'preferred_pairs': SCALPING_MARKET_CONDITIONS['preferred_pairs'],
        'avoid_pairs': SCALPING_MARKET_CONDITIONS['avoid_pairs'],
        'strategy_weight': STRATEGY_WEIGHT_SCALPING,
        'allow_combined': SCALPING_ALLOW_COMBINED,
        'debug_logging': SCALPING_DEBUG_LOGGING,
        'current_config_details': get_scalping_config()
    }

# Scalping strategy frequency presets
SCALPING_FREQUENCY_PRESETS = {
    'ultra_conservative': {
        'max_trades_per_hour': 5,
        'max_daily_trades': 30,
        'min_profit_ratio': 2.0,
        'active_config': 'conservative',
        'description': 'Very few, high quality scalping trades'
    },
    'conservative': {
        'max_trades_per_hour': 10,
        'max_daily_trades': 50,
        'min_profit_ratio': 1.8,
        'active_config': 'conservative',
        'description': 'Conservative scalping approach'
    },
    'balanced': {
        'max_trades_per_hour': 20,
        'max_daily_trades': 100,
        'min_profit_ratio': 1.5,
        'active_config': 'aggressive',
        'description': 'Balanced scalping frequency - default setting'
    },
    'aggressive': {
        'max_trades_per_hour': 35,
        'max_daily_trades': 150,
        'min_profit_ratio': 1.2,
        'active_config': 'ultra_fast',
        'description': 'High frequency scalping'
    }
}

def set_scalping_frequency_preset(preset: str = 'balanced'):
    """
    Apply a scalping frequency preset

    Args:
        preset: 'ultra_conservative', 'conservative', 'balanced', or 'aggressive'
    """
    if preset in SCALPING_FREQUENCY_PRESETS:
        preset_config = SCALPING_FREQUENCY_PRESETS[preset]
        return {
            'preset_applied': preset,
            'settings': preset_config,
            'description': preset_config['description']
        }
    else:
        return {
            'error': f'Unknown preset: {preset}',
            'available_presets': list(SCALPING_FREQUENCY_PRESETS.keys())
        }

print("✅ Scalping Strategy configuration loaded successfully")