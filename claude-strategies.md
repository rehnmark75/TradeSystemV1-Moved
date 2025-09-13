# Strategy Development Guide

This document provides comprehensive guidance for developing, configuring, and maintaining trading strategies in TradeSystemV1 using the modular configuration pattern.

## Strategy Development Pattern

### Architecture Principles

**IMPORTANT: All new strategies MUST follow this modular configuration pattern to avoid monolithic config.py bloat.**

The system implements a lightweight, modular configuration architecture that achieved an **86% reduction** in main config.py size while providing enhanced functionality and maintainability.

### 1. Modular Configuration Structure

```
configdata/
├── strategies/
│   ├── __init__.py                    # Exports all strategy configs
│   ├── config_ema_strategy.py         # EMA strategy configuration
│   ├── config_macd_strategy.py        # MACD strategy configuration
│   ├── config_zero_lag_strategy.py    # Zero-Lag strategy configuration
│   └── config_[new_strategy].py       # Template for new strategies
├── smc/
│   └── smc_configdata.py             # Smart Money Concepts
└── __init__.py                       # Main Config class with convenience methods
```

**Benefits Achieved:**
- **Modular Architecture**: Each strategy completely self-contained
- **Multiple Presets**: 7 configurations per strategy (default, conservative, aggressive, scalping, swing, news_safe, crypto)
- **Backward Compatibility**: Maintained through Config singleton pattern
- **Easy Testing**: Strategy isolation and comprehensive validation
- **Epic-specific Configurations**: Intelligent fallbacks and pair-specific settings

### 2. Strategy Config File Template

Each strategy MUST have its own dedicated config file following this pattern:

```python
# configdata/strategies/config_[strategy]_strategy.py

# Strategy enable/disable
[STRATEGY]_STRATEGY = True

# Main configuration dictionary with multiple presets
[STRATEGY]_STRATEGY_CONFIG = {
    'default': {
        'short': 21, 'long': 50, 'trend': 200,
        'description': 'Balanced configuration',
        'best_for': ['trending', 'medium_volatility'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {
        'short': 20, 'long': 50, 'trend': 200,
        'description': 'Conservative approach for low volatility',
        'best_for': ['strong_trends', 'low_volatility'],
        'min_pip_volatility': 3.0,
        'max_pip_volatility': 25.0
    },
    'aggressive': {
        'short': 5, 'long': 13, 'trend': 50,
        'description': 'Fast-reacting for high volatility breakouts',
        'best_for': ['breakouts', 'high_volatility'],
        'min_pip_volatility': 20.0,
        'max_pip_volatility': 100.0
    },
    'scalping': {
        'short': 3, 'long': 8, 'trend': 21,
        'description': 'Ultra-fast for scalping strategies',
        'best_for': ['ranging_markets', 'high_frequency']
    },
    'swing': {
        'short': 25, 'long': 55, 'trend': 200,
        'description': 'Slow and steady for swing trading',
        'best_for': ['strong_trends', 'position_trading']
    },
    'news_safe': {
        'short': 15, 'long': 30, 'trend': 200,
        'description': 'Safer configuration during news events',
        'best_for': ['news_events', 'high_volatility']
    },
    'crypto': {
        'short': 7, 'long': 25, 'trend': 99,
        'description': 'Adapted for crypto-like high volatility',
        'best_for': ['high_volatility', 'breakouts']
    }
}

# Active configuration selector
ACTIVE_[STRATEGY]_CONFIG = 'default'

# Individual feature toggles
[STRATEGY]_FEATURE_ENABLED = True
[STRATEGY]_FILTER_ENABLED = False

# Helper functions
def get_[strategy]_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get strategy configuration for specific epic with fallbacks"""
    if ENABLE_DYNAMIC_[STRATEGY]_CONFIG and market_condition in [STRATEGY]_STRATEGY_CONFIG:
        config_name = market_condition
    else:
        config_name = ACTIVE_[STRATEGY]_CONFIG
        
    return [STRATEGY]_STRATEGY_CONFIG.get(config_name, [STRATEGY]_STRATEGY_CONFIG['default'])

def get_[strategy]_threshold_for_epic(epic: str) -> float:
    """Get strategy-specific thresholds based on currency pair"""
    try:
        if 'CS.D.' in epic and '.MINI.IP' in epic:
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
        else:
            pair = epic
            
        # JPY pairs typically require different thresholds
        if 'JPY' in pair:
            return 0.05  # Higher threshold for JPY pairs
        else:
            return 0.00003  # Standard threshold for major pairs
    except Exception:
        return 0.00003  # Default fallback

# Validation function (MANDATORY)
def validate_[strategy]_config() -> dict:
    """Validate strategy configuration completeness"""
    try:
        required_keys = ['[STRATEGY]_STRATEGY', '[STRATEGY]_STRATEGY_CONFIG']
        for key in required_keys:
            if not globals().get(key):
                return {'valid': False, 'error': f'Missing {key}'}
                
        # Validate configuration structure
        for config_name, config in [STRATEGY]_STRATEGY_CONFIG.items():
            if not isinstance(config, dict):
                return {'valid': False, 'error': f'Config {config_name} must be dict'}
            if 'description' not in config:
                return {'valid': False, 'error': f'Config {config_name} missing description'}
                
        return {'valid': True, 'config_count': len([STRATEGY]_STRATEGY_CONFIG)}
    except Exception as e:
        return {'valid': False, 'error': str(e)}
```

### 3. Integration Requirements

#### A. Update strategies/__init__.py

```python
from .config_[strategy]_strategy import *

# Add all exports to __all__
__all__.extend([
    '[STRATEGY]_STRATEGY',
    '[STRATEGY]_STRATEGY_CONFIG', 
    'ACTIVE_[STRATEGY]_CONFIG',
    'get_[strategy]_config_for_epic',
    'get_[strategy]_threshold_for_epic',
    'validate_[strategy]_config'
])

# Add to validation function
def validate_strategy_configs():
    validation_results = {
        'zero_lag': validate_zero_lag_config(),
        'macd': validate_macd_config(),
        'ema': validate_ema_config(),
        '[strategy]': validate_[strategy]_config()  # Add new strategy
    }
    validation_results['overall_valid'] = all(result.get('valid', False) for result in validation_results.values())
    return validation_results
```

#### B. Update main configdata/__init__.py

```python
class Config:
    def __init__(self):
        from configdata import strategies
        self.strategies = strategies
        
    # Add convenience methods for new strategy
    def get_[strategy]_config_for_epic(self, epic: str, condition: str = 'default') -> dict:
        if hasattr(self.strategies, 'get_[strategy]_config_for_epic'):
            return self.strategies.get_[strategy]_config_for_epic(epic, condition)
        # Fallback logic
        return {'short': 21, 'long': 50, 'trend': 200}
        
    def get_[strategy]_threshold_for_epic(self, epic: str) -> float:
        if hasattr(self.strategies, 'get_[strategy]_threshold_for_epic'):
            return self.strategies.get_[strategy]_threshold_for_epic(epic)
        return 0.00003  # Default fallback
```

## Strategy Implementation Examples

### EMA Strategy Configuration

The EMA strategy demonstrates the complete modular configuration pattern:

```python
# configdata/strategies/config_ema_strategy.py

# Core Strategy Settings
SIMPLE_EMA_STRATEGY = True
ACTIVE_EMA_CONFIG = 'aggressive'  # Currently optimized configuration

# Enhanced Dynamic EMA Strategy Configurations
EMA_STRATEGY_CONFIG = {
    'default': {
        'short': 21, 'long': 50, 'trend': 200,
        'description': 'Balanced configuration for medium volatility',
        'best_for': ['trending', 'medium_volatility'],
        'preferred_pairs': ['CS.D.EURUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP']
    },
    'aggressive': {
        'short': 5, 'long': 13, 'trend': 50,
        'description': 'Fast-reacting for high volatility breakouts',
        'best_for': ['breakouts', 'high_volatility'],
        'preferred_pairs': ['CS.D.GBPUSD.MINI.IP', 'CS.D.EURUSD.MINI.IP']
    }
    # ... additional configurations
}

# Multi-Timeframe Validation Settings
TWO_POLE_MTF_VALIDATION = False          # Disabled for faster signals
MOMENTUM_BIAS_ENABLED = False            # Simplified validation
MACD_MOMENTUM_FILTER_ENABLED = True     # MACD confirmation

# MACD Filter Settings (Optimized for signal generation)
MACD_TREND_SENSITIVITY = 'permissive'          # Allow more signals
MACD_VALIDATION_MODE = 'neutral_friendly'      # Reduce blocking
```

### MACD Strategy Configuration

The MACD strategy showcases timeframe-aware optimization:

```python
# configdata/strategies/config_macd_strategy.py

# Timeframe-aware MACD configuration
MACD_PERIODS = {
    'fast_ema': 12,      # Fast EMA period
    'slow_ema': 26,      # Slow EMA period  
    'signal_ema': 9      # Signal EMA period
}

# Advanced MACD settings with database optimization
MACD_HISTOGRAM_THRESHOLD = 0.00003
MACD_ZERO_LINE_FILTER = True
MACD_RSI_FILTER_ENABLED = True
MACD_MOMENTUM_CONFIRMATION = True

# Multi-timeframe MACD validation
MACD_MTF_ENABLED = False                 # Disabled for 15m optimization
MACD_MTF_TIMEFRAMES = ['1h', '4h']
MACD_MTF_MIN_ALIGNMENT = 0.7

# Epic-specific thresholds
MACD_THRESHOLDS = {
    'EURUSD': 0.00003,
    'GBPUSD': 0.00005,
    'USDJPY': 0.003,    # Different scale for JPY pairs
    'DEFAULT': 0.00003
}
```

## Critical Integration Points

### A. Data Fetcher Integration

**MANDATORY Requirements:**
- MUST handle fallback when `[strategy]_strategy=None` parameter
- Use configdata convenience methods as primary source
- Maintain backward compatibility with old config keys

```python
# In data_fetcher.py or similar files
from configdata import config

def get_strategy_periods(epic: str, strategy: str = 'ema'):
    """Get strategy periods with proper fallbacks"""
    try:
        if strategy == 'ema':
            return config.get_ema_config_for_epic(epic)
        elif strategy == 'macd':
            return config.get_macd_config_for_epic(epic)
        else:
            return {'short': 21, 'long': 50, 'trend': 200}  # Safe fallback
    except Exception as e:
        logger.warning(f"Config fallback for {epic}: {e}")
        return {'short': 21, 'long': 50, 'trend': 200}
```

### B. Import Strategy (MANDATORY)

```python
# In files using strategy configs:
from configdata import config        # For strategy configs  
import config as system_config       # For system-level settings

# In TechnicalAnalyzer and similar classes:
from configdata import config        # MUST add this import
```

**CRITICAL**: Fix key mismatches like `'fast'` vs `'fast_ema'` that can cause runtime errors.

## Common Issues to Avoid

### 1. Key Mismatch Errors
```python
# ❌ WRONG: Inconsistent naming
config = {'fast': 12, 'slow_ema': 26}

# ✅ CORRECT: Consistent naming
config = {'fast_ema': 12, 'slow_ema': 26, 'signal_ema': 9}
```

### 2. Missing Import Statements
```python
# ❌ WRONG: Missing configdata import
class TechnicalAnalyzer:
    def get_ema_periods(self):
        return config.EMA_PERIODS  # NameError!

# ✅ CORRECT: Proper import
from configdata import config

class TechnicalAnalyzer:
    def get_ema_periods(self):
        return config.strategies.EMA_PERIODS
```

### 3. Fallback Logic Issues
```python
# ❌ WRONG: No fallback handling
def get_periods(strategy=None):
    return config.get_strategy_config(strategy)  # Fails if strategy=None

# ✅ CORRECT: Proper fallback
def get_periods(strategy=None):
    if strategy is None:
        return {'short': 21, 'long': 50, 'trend': 200}
    return config.get_strategy_config(strategy)
```

### 4. Validation Requirements
```python
# ✅ REQUIRED: Include both period columns AND semantic columns
enhanced_data = {
    # Period-specific columns
    'ema_21': df['close'].ewm(span=21).mean(),
    'ema_50': df['close'].ewm(span=50).mean(), 
    'ema_200': df['close'].ewm(span=200).mean(),
    
    # Semantic columns (for strategy flexibility)
    'ema_short': df['close'].ewm(span=config.short).mean(),
    'ema_long': df['close'].ewm(span=config.long).mean(),
    'ema_trend': df['close'].ewm(span=config.trend).mean()
}
```

### 5. Default Alignment
```python
# ❌ WRONG: Mismatched defaults
hardcoded_fallback = [9, 21, 200]
config_default = [21, 50, 200]

# ✅ CORRECT: Aligned defaults
hardcoded_fallback = [21, 50, 200]  # Match configdata defaults
config_default = [21, 50, 200]
```

## Validation Pattern (MANDATORY)

Each strategy config MUST include comprehensive validation:

### Required Validation Components

```python
def validate_[strategy]_config() -> dict:
    """Comprehensive validation for strategy configuration"""
    try:
        validation_results = []
        
        # 1. Required keys exist and are properly typed
        required_keys = ['[STRATEGY]_STRATEGY', '[STRATEGY]_STRATEGY_CONFIG']
        for key in required_keys:
            if not globals().get(key):
                return {'valid': False, 'error': f'Missing required key: {key}'}
        
        # 2. Value ranges are reasonable
        for config_name, config in [STRATEGY]_STRATEGY_CONFIG.items():
            if 'short' in config and 'long' in config and 'trend' in config:
                if not (config['short'] < config['long'] < config['trend']):
                    return {'valid': False, 'error': f'Invalid EMA order in {config_name}'}
        
        # 3. Epic-specific configurations are valid
        test_epic = 'CS.D.EURUSD.MINI.IP'
        try:
            test_config = get_[strategy]_config_for_epic(test_epic)
            if not isinstance(test_config, dict):
                return {'valid': False, 'error': 'Epic config function returns invalid type'}
        except Exception as e:
            return {'valid': False, 'error': f'Epic config function failed: {e}'}
        
        # 4. Fallback mechanisms work correctly
        try:
            test_threshold = get_[strategy]_threshold_for_epic('INVALID_EPIC')
            if not isinstance(test_threshold, float):
                return {'valid': False, 'error': 'Threshold function returns invalid type'}
        except Exception as e:
            return {'valid': False, 'error': f'Threshold function failed: {e}'}
        
        # 5. Configuration presets are complete and consistent
        required_presets = ['default', 'conservative', 'aggressive']
        for preset in required_presets:
            if preset not in [STRATEGY]_STRATEGY_CONFIG:
                return {'valid': False, 'error': f'Missing required preset: {preset}'}
        
        return {
            'valid': True, 
            'config_count': len([STRATEGY]_STRATEGY_CONFIG),
            'presets': list([STRATEGY]_STRATEGY_CONFIG.keys()),
            'active_config': ACTIVE_[STRATEGY]_CONFIG
        }
        
    except Exception as e:
        return {'valid': False, 'error': f'Validation exception: {str(e)}'}
```

### Example Validation Results

```python
# ✅ Successful validation output
{
    'ema': {'valid': True, 'config_count': 7, 'presets': ['default', 'conservative', 'aggressive', 'scalping', 'swing', 'news_safe', 'crypto'], 'active_config': 'aggressive'},
    'macd': {'valid': True, 'config_count': 5, 'presets': ['default', 'conservative', 'aggressive', 'scalping', 'swing'], 'active_config': 'default'},
    'zero_lag': {'valid': True, 'config_count': 6, 'presets': ['default', 'conservative', 'aggressive', 'scalping', 'swing', 'crypto'], 'active_config': 'default'},
    'overall_valid': True
}
```

## Success Metrics

A properly implemented strategy using this pattern will demonstrate:

### ✅ Technical Metrics
- **No configuration-related runtime errors**
- **All validation tests pass with detailed reporting**
- **Clean separation from system configuration**
- **Multiple working presets (minimum 5 configurations)**
- **Backward compatibility maintained**

### ✅ Performance Metrics  
- **Significant reduction in main config.py size** (target: 80%+ reduction)
- **Fast configuration loading and validation**
- **Epic-specific configuration support**
- **Intelligent fallback mechanisms**

### ✅ Maintainability Metrics
- **Modular architecture with clear boundaries**
- **Comprehensive validation with error reporting**
- **Easy testing and strategy isolation**
- **Documented configuration options and use cases**

## Advanced Configuration Features

### Market Context Awareness

```python
# Epic-specific configuration with market conditions
def get_[strategy]_config_for_market(epic: str, market_condition: dict) -> dict:
    """Get configuration based on market conditions"""
    volatility = market_condition.get('volatility', 'medium')
    trend_strength = market_condition.get('trend_strength', 'medium')
    session = market_condition.get('session', 'london')
    
    # Select optimal configuration based on conditions
    if volatility == 'high' and trend_strength == 'strong':
        return [STRATEGY]_STRATEGY_CONFIG['aggressive']
    elif volatility == 'low' and trend_strength == 'strong':
        return [STRATEGY]_STRATEGY_CONFIG['conservative']
    elif session == 'asian':
        return [STRATEGY]_STRATEGY_CONFIG['news_safe']
    else:
        return [STRATEGY]_STRATEGY_CONFIG['default']
```

### Dynamic Configuration Switching

```python
# Runtime configuration switching based on performance
class ConfigurationManager:
    def __init__(self):
        self.performance_tracker = {}
        
    def suggest_config_change(self, epic: str, current_performance: dict) -> str:
        """Suggest configuration changes based on performance"""
        win_rate = current_performance.get('win_rate', 0)
        profit_factor = current_performance.get('profit_factor', 1)
        
        if win_rate < 0.4:  # Poor win rate
            return 'conservative'  # More selective signals
        elif profit_factor < 1.2:  # Poor profit factor
            return 'aggressive'    # Different risk profile
        else:
            return 'default'       # Keep current approach
```

**This modular pattern prevents monolithic config.py bloat and ensures clean, maintainable, and extensible strategy configurations.**

For optimization integration, see [Dynamic Parameter System](claude-optimization.md).
For architectural context, see [System Architecture](claude-architecture.md).
For command examples, see [Commands & CLI](claude-commands.md).