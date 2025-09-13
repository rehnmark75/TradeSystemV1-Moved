"""
Pine Script to TradeSystemV1 Configuration Mapper

Converts extracted Pine Script patterns and parameters into TradeSystemV1
strategy configurations following the modular configuration pattern.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def to_config(inputs: List[Dict], signals: Dict[str, Any], strategy_name: str = "ImportedFromTV") -> Dict[str, Any]:
    """
    Convert Pine Script analysis to TradeSystemV1 configuration
    
    Args:
        inputs: Extracted input parameters
        signals: Extracted signals and patterns
        strategy_name: Name for the strategy
        
    Returns:
        TradeSystemV1 strategy configuration dictionary
    """
    try:
        config = {
            "name": strategy_name,
            "provenance": {
                "source": "tradingview",
                "imported_at": int(datetime.now().timestamp()),
                "strategy_type": signals.get("strategy_type", "unknown"),
                "complexity_score": signals.get("complexity_score", 0.0)
            },
            "modules": {},
            "filters": {},
            "rules": [],
            "presets": {}
        }
        
        # Configure EMA module
        ema_periods = signals.get("ema_periods", [])
        if ema_periods:
            config["modules"]["ema"] = _configure_ema_module(ema_periods, inputs)
        
        # Configure MACD module
        macd_config = signals.get("macd")
        if macd_config:
            config["modules"]["macd"] = _configure_macd_module(macd_config, inputs)
        
        # Configure RSI module
        rsi_periods = signals.get("rsi_periods", [])
        if rsi_periods:
            config["modules"]["rsi"] = _configure_rsi_module(rsi_periods, inputs)
        
        # Configure Bollinger Bands module
        bb_config = signals.get("bollinger_bands")
        if bb_config:
            config["modules"]["bollinger"] = _configure_bollinger_module(bb_config, inputs)
        
        # Configure FVG module
        if signals.get("mentions_fvg", False):
            config["modules"]["fvg"] = _configure_fvg_module(signals, inputs)
        
        # Configure SMC module
        if signals.get("mentions_smc", False):
            config["modules"]["smc"] = _configure_smc_module(signals, inputs)
        
        # Configure filters
        config["filters"] = _configure_filters(signals, inputs)
        
        # Generate trading rules
        config["rules"] = _generate_trading_rules(signals, inputs)
        
        # Generate strategy presets
        config["presets"] = _generate_strategy_presets(signals, inputs, strategy_name)
        
        logger.info(f"Generated TradeSystemV1 config for {strategy_name} with {len(config['modules'])} modules")
        return config
        
    except Exception as e:
        logger.error(f"Failed to convert to config: {e}")
        return _get_fallback_config(strategy_name)

def _configure_ema_module(ema_periods: List[int], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure EMA module based on extracted periods"""
    try:
        # Sort periods and assign roles
        sorted_periods = sorted(ema_periods)
        
        if len(sorted_periods) >= 3:
            short, long, trend = sorted_periods[0], sorted_periods[1], sorted_periods[-1]
        elif len(sorted_periods) == 2:
            short, long = sorted_periods
            trend = 200  # Default trend EMA
        else:
            short = sorted_periods[0] if sorted_periods else 21
            long = 50    # Default long EMA
            trend = 200  # Default trend EMA
        
        return {
            "enabled": True,
            "periods": sorted_periods,
            "short": short,
            "long": long,
            "trend": trend,
            "validation_enabled": True,
            "distance_filter_enabled": True,
            "description": f"EMA configuration derived from TradingView script with periods {sorted_periods}"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure EMA module: {e}")
        return {"enabled": True, "periods": [21, 50, 200]}

def _configure_macd_module(macd_config: Dict[str, int], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure MACD module based on extracted parameters"""
    try:
        return {
            "enabled": True,
            "fast_ema": macd_config.get("fast", 12),
            "slow_ema": macd_config.get("slow", 26),
            "signal_ema": macd_config.get("signal", 9),
            "histogram_threshold": 0.00003,
            "zero_line_filter": True,
            "momentum_confirmation": True,
            "description": f"MACD configuration: {macd_config['fast']}/{macd_config['slow']}/{macd_config['signal']}"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure MACD module: {e}")
        return {"enabled": True, "fast_ema": 12, "slow_ema": 26, "signal_ema": 9}

def _configure_rsi_module(rsi_periods: List[int], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure RSI module based on extracted periods"""
    try:
        primary_period = rsi_periods[0] if rsi_periods else 14
        
        return {
            "enabled": True,
            "period": primary_period,
            "overbought": 70,
            "oversold": 30,
            "divergence_detection": True,
            "filter_enabled": True,
            "description": f"RSI configuration with period {primary_period}"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure RSI module: {e}")
        return {"enabled": True, "period": 14}

def _configure_bollinger_module(bb_config: Dict[str, Any], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure Bollinger Bands module"""
    try:
        return {
            "enabled": True,
            "length": bb_config.get("length", 20),
            "multiplier": bb_config.get("multiplier", 2.0),
            "squeeze_detection": True,
            "breakout_signals": True,
            "mean_reversion": True,
            "description": f"Bollinger Bands: {bb_config['length']} period, {bb_config['multiplier']} std dev"
        }
        
    except Exception as e:
        logger.error(f"Failed to configure Bollinger module: {e}")
        return {"enabled": True, "length": 20, "multiplier": 2.0}

def _configure_fvg_module(signals: Dict[str, Any], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure Fair Value Gap module"""
    return {
        "enabled": True,
        "detection_enabled": True,
        "mitigation_tracking": True,
        "timeframe_filter": True,
        "min_gap_size": 5.0,  # pips
        "max_gap_age": 24,    # hours
        "description": "Fair Value Gap detection derived from TradingView script"
    }

def _configure_smc_module(signals: Dict[str, Any], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure Smart Money Concepts module"""
    return {
        "enabled": True,
        "bos_detection": True,
        "choch_detection": True,
        "order_block_detection": True,
        "liquidity_analysis": True,
        "structure_timeframes": ["15m", "1h", "4h"],
        "description": "Smart Money Concepts derived from TradingView script"
    }

def _configure_filters(signals: Dict[str, Any], inputs: List[Dict]) -> Dict[str, Any]:
    """Configure strategy filters"""
    filters = {}
    
    # Higher timeframe alignment
    if signals.get("higher_tf"):
        filters["htf_alignment"] = {
            "enabled": True,
            "timeframes": [tf["tf"] for tf in signals["higher_tf"][:2]],  # Limit to 2 TFs
            "description": "Higher timeframe alignment from TradingView script"
        }
    
    # Volume filter if mentioned
    if signals.get("mentions_volume"):
        filters["volume"] = {
            "enabled": True,
            "min_volume_multiplier": 1.5,
            "vwap_filter": True,
            "description": "Volume analysis from TradingView script"
        }
    
    # Support/Resistance filter
    if signals.get("mentions_sr"):
        filters["support_resistance"] = {
            "enabled": True,
            "proximity_threshold": 10,  # pips
            "strength_validation": True,
            "description": "Support/Resistance analysis from TradingView script"
        }
    
    return filters

def _generate_trading_rules(signals: Dict[str, Any], inputs: List[Dict]) -> List[Dict[str, Any]]:
    """Generate trading rules based on extracted signals"""
    rules = []
    
    try:
        # EMA crossover rules
        if signals.get("has_cross_up") and signals.get("ema_periods"):
            rules.append({
                "type": "entry_long",
                "condition": "ema_fast_crosses_above_slow",
                "confirmation": "trend_alignment",
                "description": "Long entry on EMA crossover up"
            })
        
        if signals.get("has_cross_down") and signals.get("ema_periods"):
            rules.append({
                "type": "entry_short", 
                "condition": "ema_fast_crosses_below_slow",
                "confirmation": "trend_alignment",
                "description": "Short entry on EMA crossover down"
            })
        
        # MACD confirmation rules
        if signals.get("macd"):
            rules.append({
                "type": "confirmation",
                "condition": "macd_agrees_with_direction",
                "weight": 0.3,
                "description": "MACD momentum confirmation"
            })
        
        # SMC-based rules
        if signals.get("mentions_smc"):
            rules.append({
                "type": "entry_long",
                "condition": "bos_bullish_with_orderblock",
                "confirmation": "liquidity_sweep",
                "description": "SMC bullish break of structure"
            })
            
            rules.append({
                "type": "entry_short",
                "condition": "bos_bearish_with_orderblock", 
                "confirmation": "liquidity_sweep",
                "description": "SMC bearish break of structure"
            })
        
        # FVG rules
        if signals.get("mentions_fvg"):
            rules.append({
                "type": "entry",
                "condition": "fvg_formed_and_price_return",
                "confirmation": "volume_spike",
                "description": "Fair Value Gap entry signal"
            })
        
        # RSI mean reversion rules
        if signals.get("rsi_periods"):
            rules.append({
                "type": "entry_long",
                "condition": "rsi_oversold_and_divergence",
                "confirmation": "support_level",
                "description": "RSI oversold reversal"
            })
            
            rules.append({
                "type": "entry_short",
                "condition": "rsi_overbought_and_divergence",
                "confirmation": "resistance_level", 
                "description": "RSI overbought reversal"
            })
        
        # Bollinger Bands rules
        if signals.get("bollinger_bands"):
            rules.append({
                "type": "entry",
                "condition": "bb_squeeze_breakout",
                "confirmation": "volume_expansion",
                "description": "Bollinger Bands squeeze breakout"
            })
        
        logger.info(f"Generated {len(rules)} trading rules")
        return rules
        
    except Exception as e:
        logger.error(f"Failed to generate rules: {e}")
        return [{"type": "placeholder", "condition": "manual_review_required"}]

def _generate_strategy_presets(signals: Dict[str, Any], inputs: List[Dict], strategy_name: str) -> Dict[str, Any]:
    """Generate strategy presets following TradeSystemV1 pattern"""
    try:
        strategy_type = signals.get("strategy_type", "trending")
        complexity = signals.get("complexity_score", 0.0)
        
        presets = {
            "default": _create_default_preset(signals, strategy_type),
            "conservative": _create_conservative_preset(signals, strategy_type),
            "aggressive": _create_aggressive_preset(signals, strategy_type)
        }
        
        # Add specialized presets based on strategy type
        if strategy_type == "scalping":
            presets["ultra_fast"] = _create_scalping_preset(signals)
        elif strategy_type == "swing":
            presets["position_trading"] = _create_swing_preset(signals)
        elif strategy_type == "smc":
            presets["institutional"] = _create_smc_preset(signals)
        
        return presets
        
    except Exception as e:
        logger.error(f"Failed to generate presets: {e}")
        return {"default": {"description": "Generated preset - manual review required"}}

def _create_default_preset(signals: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
    """Create default preset configuration"""
    ema_periods = signals.get("ema_periods", [21, 50, 200])
    
    base_config = {
        "description": f"Balanced {strategy_type} configuration imported from TradingView",
        "best_for": [strategy_type, "medium_volatility"],
        "confidence_threshold": 0.55,
        "stop_loss_pips": 15,
        "take_profit_pips": 30,
        "risk_reward_ratio": 2.0
    }
    
    # Add EMA configuration if available
    if ema_periods:
        if len(ema_periods) >= 3:
            base_config.update({
                "short": ema_periods[0],
                "long": ema_periods[1], 
                "trend": ema_periods[-1]
            })
        elif len(ema_periods) >= 2:
            base_config.update({
                "short": ema_periods[0],
                "long": ema_periods[1],
                "trend": 200
            })
    
    return base_config

def _create_conservative_preset(signals: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
    """Create conservative preset with lower risk"""
    default = _create_default_preset(signals, strategy_type)
    
    return {
        **default,
        "description": f"Conservative {strategy_type} with higher confirmation requirements",
        "confidence_threshold": 0.70,
        "stop_loss_pips": 12,
        "take_profit_pips": 36,
        "risk_reward_ratio": 3.0,
        "best_for": ["trending", "low_volatility"]
    }

def _create_aggressive_preset(signals: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
    """Create aggressive preset with higher frequency"""
    default = _create_default_preset(signals, strategy_type)
    
    return {
        **default,
        "description": f"Aggressive {strategy_type} with faster signals",
        "confidence_threshold": 0.40,
        "stop_loss_pips": 20,
        "take_profit_pips": 25,
        "risk_reward_ratio": 1.25,
        "best_for": ["breakout", "high_volatility"]
    }

def _create_scalping_preset(signals: Dict[str, Any]) -> Dict[str, Any]:
    """Create specialized scalping preset"""
    return {
        "description": "Ultra-fast scalping configuration",
        "confidence_threshold": 0.35,
        "stop_loss_pips": 8,
        "take_profit_pips": 12,
        "risk_reward_ratio": 1.5,
        "best_for": ["ranging", "high_frequency"],
        "short": 5,
        "long": 13,
        "trend": 50
    }

def _create_swing_preset(signals: Dict[str, Any]) -> Dict[str, Any]:
    """Create specialized swing trading preset"""
    return {
        "description": "Swing trading with extended holding periods",
        "confidence_threshold": 0.65,
        "stop_loss_pips": 30,
        "take_profit_pips": 90,
        "risk_reward_ratio": 3.0,
        "best_for": ["trending", "position_trading"],
        "short": 25,
        "long": 55,
        "trend": 200
    }

def _create_smc_preset(signals: Dict[str, Any]) -> Dict[str, Any]:
    """Create specialized Smart Money Concepts preset"""
    return {
        "description": "Institutional flow analysis configuration",
        "confidence_threshold": 0.60,
        "stop_loss_pips": 20,
        "take_profit_pips": 60,
        "risk_reward_ratio": 3.0,
        "best_for": ["institutional", "structure_breaks"],
        "smc_enabled": True,
        "order_block_validation": True,
        "liquidity_analysis": True
    }

def _get_fallback_config(strategy_name: str) -> Dict[str, Any]:
    """Return fallback configuration on error"""
    return {
        "name": strategy_name,
        "provenance": {
            "source": "tradingview",
            "error": "Configuration generation failed - manual review required"
        },
        "modules": {
            "ema": {"enabled": True, "periods": [21, 50, 200]}
        },
        "filters": {},
        "rules": [{"type": "manual_review", "condition": "requires_human_analysis"}],
        "presets": {
            "default": {
                "description": "Fallback configuration - requires manual adjustment",
                "confidence_threshold": 0.55
            }
        }
    }

def generate_config_file_content(config: Dict[str, Any]) -> str:
    """
    Generate actual Python config file content following TradeSystemV1 patterns
    
    Args:
        config: Strategy configuration dictionary
        
    Returns:
        Python file content as string
    """
    try:
        strategy_name = config.get("name", "ImportedFromTV").upper().replace(" ", "_")
        
        content = f'''# configdata/strategies/config_{strategy_name.lower()}_strategy.py
"""
{config.get("name", "Imported Strategy")} Configuration
Imported from TradingView on {datetime.now().strftime("%Y-%m-%d")}

Strategy Type: {config.get("provenance", {}).get("strategy_type", "unknown")}
Complexity Score: {config.get("provenance", {}).get("complexity_score", 0.0)}
Source: {config.get("provenance", {}).get("source", "tradingview")}
"""

# Core Strategy Settings
{strategy_name}_STRATEGY = True

# Strategy Configuration with Multiple Presets
{strategy_name}_STRATEGY_CONFIG = {{
'''
        
        # Add presets
        for preset_name, preset_config in config.get("presets", {}).items():
            content += f'''    '{preset_name}': {{
'''
            for key, value in preset_config.items():
                if isinstance(value, str):
                    content += f"        '{key}': '{value}',\n"
                elif isinstance(value, list):
                    content += f"        '{key}': {value},\n"
                else:
                    content += f"        '{key}': {value},\n"
            content += "    },\n"
        
        content += "}\n\n"
        
        # Add active configuration
        content += f"ACTIVE_{strategy_name}_CONFIG = 'default'\n\n"
        
        # Add module configurations
        modules = config.get("modules", {})
        if "ema" in modules:
            ema_config = modules["ema"]
            content += f"# EMA Configuration\n"
            content += f"{strategy_name}_EMA_PERIODS = {ema_config.get('periods', [21, 50, 200])}\n"
            content += f"{strategy_name}_EMA_VALIDATION = {ema_config.get('validation_enabled', True)}\n\n"
        
        if "macd" in modules:
            macd_config = modules["macd"]
            content += f"# MACD Configuration\n"
            content += f"{strategy_name}_MACD_FAST = {macd_config.get('fast_ema', 12)}\n"
            content += f"{strategy_name}_MACD_SLOW = {macd_config.get('slow_ema', 26)}\n"
            content += f"{strategy_name}_MACD_SIGNAL = {macd_config.get('signal_ema', 9)}\n\n"
        
        # Add helper functions
        content += f'''
def get_{strategy_name.lower()}_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get {strategy_name} configuration for specific epic"""
    config_name = market_condition if market_condition in {strategy_name}_STRATEGY_CONFIG else ACTIVE_{strategy_name}_CONFIG
    return {strategy_name}_STRATEGY_CONFIG.get(config_name, {strategy_name}_STRATEGY_CONFIG['default'])

def validate_{strategy_name.lower()}_config() -> dict:
    """Validate {strategy_name} configuration"""
    try:
        return {{
            'valid': True,
            'config_count': len({strategy_name}_STRATEGY_CONFIG),
            'presets': list({strategy_name}_STRATEGY_CONFIG.keys()),
            'active_config': ACTIVE_{strategy_name}_CONFIG
        }}
    except Exception as e:
        return {{'valid': False, 'error': str(e)}}
'''
        
        return content
        
    except Exception as e:
        logger.error(f"Failed to generate config file content: {e}")
        return f"# Error generating configuration: {e}\n# Manual review required"