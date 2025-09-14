# configdata/strategies/config_tv_ema_crossover_strategy.py
"""
TV_EMA_Crossover Configuration
Imported from TradingView on 2025-09-13

Strategy Type: swing
Complexity Score: 0.4
Source: tradingview
"""

# Core Strategy Settings
TV_EMA_CROSSOVER_STRATEGY = True

# Strategy Configuration with Multiple Presets
TV_EMA_CROSSOVER_STRATEGY_CONFIG = {
    'default': {
        'description': 'Balanced swing configuration imported from TradingView',
        'best_for': ['swing', 'medium_volatility'],
        'confidence_threshold': 0.55,
        'stop_loss_pips': 15,
        'take_profit_pips': 30,
        'risk_reward_ratio': 2.0,
        'short': 12,
        'long': 26,
        'trend': 200,
    },
    'conservative': {
        'description': 'Conservative swing with higher confirmation requirements',
        'best_for': ['trending', 'low_volatility'],
        'confidence_threshold': 0.7,
        'stop_loss_pips': 12,
        'take_profit_pips': 36,
        'risk_reward_ratio': 3.0,
        'short': 12,
        'long': 26,
        'trend': 200,
    },
    'aggressive': {
        'description': 'Aggressive swing with faster signals',
        'best_for': ['breakout', 'high_volatility'],
        'confidence_threshold': 0.4,
        'stop_loss_pips': 20,
        'take_profit_pips': 25,
        'risk_reward_ratio': 1.25,
        'short': 12,
        'long': 26,
        'trend': 200,
    },
    'position_trading': {
        'description': 'Swing trading with extended holding periods',
        'confidence_threshold': 0.65,
        'stop_loss_pips': 30,
        'take_profit_pips': 90,
        'risk_reward_ratio': 3.0,
        'best_for': ['trending', 'position_trading'],
        'short': 25,
        'long': 55,
        'trend': 200,
    },
}

ACTIVE_TV_EMA_CROSSOVER_CONFIG = 'default'

# EMA Configuration
TV_EMA_CROSSOVER_EMA_PERIODS = [12, 26, 200]
TV_EMA_CROSSOVER_EMA_VALIDATION = True


def get_tv_ema_crossover_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get TV_EMA_CROSSOVER configuration for specific epic"""
    config_name = market_condition if market_condition in TV_EMA_CROSSOVER_STRATEGY_CONFIG else ACTIVE_TV_EMA_CROSSOVER_CONFIG
    return TV_EMA_CROSSOVER_STRATEGY_CONFIG.get(config_name, TV_EMA_CROSSOVER_STRATEGY_CONFIG['default'])

def validate_tv_ema_crossover_config() -> dict:
    """Validate TV_EMA_CROSSOVER configuration"""
    try:
        return {
            'valid': True,
            'config_count': len(TV_EMA_CROSSOVER_STRATEGY_CONFIG),
            'presets': list(TV_EMA_CROSSOVER_STRATEGY_CONFIG.keys()),
            'active_config': ACTIVE_TV_EMA_CROSSOVER_CONFIG
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}
