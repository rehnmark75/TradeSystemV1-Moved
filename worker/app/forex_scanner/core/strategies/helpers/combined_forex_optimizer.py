# core/strategies/helpers/combined_forex_optimizer.py
"""
Combined Strategy Forex Optimizer - MODULAR HELPER
🔥 FOREX OPTIMIZED: Combined strategy specific calculations and optimizations
🏗️ MODULAR: Focused on forex optimization for combined strategy
🎯 MAINTAINABLE: Single responsibility - forex optimization only
⚡ PERFORMANCE: Intelligent caching and efficient calculations
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    from forex_scanner import config


class CombinedForexOptimizer:
    """
    🔥 FOREX OPTIMIZER: Combined strategy specific forex calculations
    
    Handles:
    - Strategy weight normalization
    - Forex-specific confidence adjustments
    - Configuration management
    - Strategy enable/disable logic
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
        # Strategy enable flags from config
        self.strategy_flags = self._load_strategy_enable_flags()
        
        # Base strategy weights from config
        self.base_weights = self._load_base_strategy_weights()
        
        # Forex-specific thresholds
        self.forex_confidence_adjustments = self._load_forex_confidence_adjustments()
        
        # Combined strategy configuration
        self.combined_config = self._load_combined_strategy_config()
        
        # Performance cache
        self._cache = {}
        
        self.logger.debug("✅ CombinedForexOptimizer initialized")

    def _load_strategy_enable_flags(self) -> Dict[str, bool]:
        """Load strategy enable flags from configuration"""
        return {
            'ema_enabled': True,  # Always enabled (core strategy)
            'macd_enabled': True,  # Always enabled (core strategy)
            'kama_enabled': getattr(config, 'KAMA_STRATEGY', False),
            'bb_supertrend_enabled': getattr(config, 'BOLLINGER_SUPERTREND_STRATEGY', False),
            'momentum_bias_enabled': getattr(config, 'MOMENTUM_BIAS_STRATEGY', False),
            'zero_lag_enabled': getattr(config, 'ZERO_LAG_STRATEGY', False)
        }

    def _load_base_strategy_weights(self) -> Dict[str, float]:
        """Load base strategy weights from configuration"""
        return {
            'ema': getattr(config, 'STRATEGY_WEIGHT_EMA', 0.35),
            'macd': getattr(config, 'STRATEGY_WEIGHT_MACD', 0.30),
            'kama': getattr(config, 'STRATEGY_WEIGHT_KAMA', 0.20),
            'bb_supertrend': getattr(config, 'STRATEGY_WEIGHT_BB_SUPERTREND', 0.08),
            'momentum_bias': getattr(config, 'STRATEGY_WEIGHT_MOMENTUM_BIAS', 0.05),
            'zero_lag': getattr(config, 'STRATEGY_WEIGHT_ZERO_LAG', 0.15)
        }

    def _load_forex_confidence_adjustments(self) -> Dict[str, Dict]:
        """Load forex-specific confidence adjustments"""
        return {
            'major_pairs': {
                'confidence_bonus': 0.05,
                'volatility_adjustment': 1.0,
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            },
            'minor_pairs': {
                'confidence_bonus': 0.02,
                'volatility_adjustment': 1.1,
                'pairs': ['EURJPY', 'GBPJPY', 'EURCHF', 'AUDUSD']
            },
            'exotic_pairs': {
                'confidence_bonus': -0.02,
                'volatility_adjustment': 1.3,
                'pairs': ['USDTRY', 'USDZAR', 'USDMXN']
            }
        }

    def _load_combined_strategy_config(self) -> Dict[str, Any]:
        """Load combined strategy configuration"""
        return {
            'mode': getattr(config, 'COMBINED_STRATEGY_MODE', 'consensus'),
            'min_combined_confidence': getattr(config, 'MIN_COMBINED_CONFIDENCE', 0.75),
            'consensus_threshold': getattr(config, 'CONSENSUS_THRESHOLD', 0.7),
            'minimum_bars_required': getattr(config, 'MIN_BARS_COMBINED', 50),
            'enable_safety_filters': getattr(config, 'ENABLE_COMBINED_SAFETY_FILTERS', True),
            'require_volume_confirmation': getattr(config, 'REQUIRE_VOLUME_CONFIRMATION', False)
        }

    def get_strategy_enable_flags(self) -> Dict[str, bool]:
        """Get current strategy enable flags"""
        return self.strategy_flags.copy()

    def get_base_strategy_weights(self) -> Dict[str, float]:
        """Get base strategy weights before normalization"""
        return self.base_weights.copy()

    def get_normalized_strategy_weights(self) -> Dict[str, float]:
        """
        Get normalized strategy weights based on enabled strategies
        
        Returns weights that sum to 1.0 for only enabled strategies
        """
        cache_key = "normalized_weights"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            enabled_strategies = [name for name, enabled in self.strategy_flags.items() if enabled]
            
            # Calculate active weights
            active_weights = {}
            for strategy_name in enabled_strategies:
                # Convert flag name to weight name
                weight_name = strategy_name.replace('_enabled', '')
                if weight_name in self.base_weights:
                    active_weights[weight_name] = self.base_weights[weight_name]
            
            # Normalize weights to sum to 1.0
            total_weight = sum(active_weights.values())
            if total_weight > 0:
                normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
            else:
                # Fallback to equal weights for core strategies
                normalized_weights = {'ema': 0.5, 'macd': 0.5}
                for strategy in active_weights.keys():
                    if strategy not in ['ema', 'macd']:
                        normalized_weights[strategy] = 0.0
            
            # Ensure all strategy names are present (with 0 for disabled)
            final_weights = {}
            for weight_name in self.base_weights.keys():
                final_weights[weight_name] = normalized_weights.get(weight_name, 0.0)
            
            # Cache the result
            self._cache[cache_key] = final_weights
            
            self.logger.debug(f"📊 Normalized weights calculated: {final_weights}")
            return final_weights
            
        except Exception as e:
            self.logger.error(f"❌ Weight normalization failed: {e}")
            # Return safe defaults
            return {'ema': 0.5, 'macd': 0.5, 'kama': 0.0, 'bb_supertrend': 0.0, 'momentum_bias': 0.0, 'zero_lag': 0.0}

    def get_combined_strategy_config(self) -> Dict[str, Any]:
        """Get complete combined strategy configuration"""
        config_copy = self.combined_config.copy()
        
        # Add dynamic consensus threshold based on number of active strategies
        enabled_count = sum(1 for enabled in self.strategy_flags.values() if enabled)
        config_copy['dynamic_consensus_threshold'] = max(0.6, 0.8 - (enabled_count * 0.05))
        
        return config_copy

    def get_minimum_bars_required(self) -> int:
        """Get minimum bars required for combined strategy"""
        return self.combined_config['minimum_bars_required']

    def apply_forex_confidence_adjustments(self, base_confidence: float, epic: str) -> float:
        """
        Apply forex-specific confidence adjustments based on pair type
        
        Args:
            base_confidence: Base confidence score
            epic: Trading epic (e.g., 'CS.D.EURUSD.CEEM.IP')
            
        Returns:
            Adjusted confidence score
        """
        try:
            # Extract pair from epic
            pair = self._extract_pair_from_epic(epic)
            
            # Determine pair type and get adjustments
            pair_type = self._get_forex_pair_type(pair)
            adjustments = self.forex_confidence_adjustments.get(pair_type, {})
            
            # Apply confidence bonus
            confidence_bonus = adjustments.get('confidence_bonus', 0.0)
            adjusted_confidence = base_confidence + confidence_bonus
            
            # Apply volatility adjustment (affects threshold, not direct confidence)
            volatility_adjustment = adjustments.get('volatility_adjustment', 1.0)
            
            # Bound the result
            final_confidence = max(0.1, min(0.98, adjusted_confidence))
            
            if abs(final_confidence - base_confidence) > 0.01:
                self.logger.debug(f"🔧 Forex adjustment for {pair} ({pair_type}): {base_confidence:.3f} → {final_confidence:.3f}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Forex confidence adjustment failed for {epic}: {e}")
            return base_confidence

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic string"""
        try:
            # Remove common prefixes and suffixes
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.IP', '')
            return pair.upper()
        except Exception:
            return epic.upper()

    def _get_forex_pair_type(self, pair: str) -> str:
        """Determine forex pair type (major, minor, exotic)"""
        for pair_type, config in self.forex_confidence_adjustments.items():
            if pair in config.get('pairs', []):
                return pair_type
        
        # Default to minor for unknown pairs
        return 'minor_pairs'

    def calculate_dynamic_threshold(self, market_conditions: Dict) -> float:
        """
        Calculate dynamic confidence threshold based on market conditions
        
        Args:
            market_conditions: Dictionary containing market condition data
            
        Returns:
            Dynamic threshold value
        """
        try:
            base_threshold = self.combined_config['min_combined_confidence']
            
            # Adjust based on volatility
            volatility = market_conditions.get('volatility_regime', 'medium')
            
            if volatility == 'high':
                # Higher threshold in volatile markets (be more selective)
                return min(0.9, base_threshold + 0.1)
            elif volatility == 'low':
                # Lower threshold in stable markets (can be less selective)
                return max(0.6, base_threshold - 0.05)
            else:
                return base_threshold
                
        except Exception as e:
            self.logger.warning(f"⚠️ Dynamic threshold calculation failed: {e}")
            return self.combined_config['min_combined_confidence']

    def calculate_strategy_diversity_bonus(self, contributing_strategies: List[str]) -> float:
        """
        Calculate confidence bonus based on strategy diversity
        
        Args:
            contributing_strategies: List of strategy names contributing to signal
            
        Returns:
            Diversity bonus (0.0 to 0.1)
        """
        try:
            strategy_count = len(contributing_strategies)
            total_possible = sum(1 for enabled in self.strategy_flags.values() if enabled)
            
            if total_possible == 0:
                return 0.0
            
            diversity_ratio = strategy_count / total_possible
            
            # Bonus scale: 0% for 1 strategy, 10% for all strategies
            max_bonus = 0.1
            diversity_bonus = diversity_ratio * max_bonus
            
            return min(max_bonus, diversity_bonus)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Diversity bonus calculation failed: {e}")
            return 0.0

    def optimize_strategy_weights_for_market(self, market_conditions: Dict) -> Dict[str, float]:
        """
        Dynamically optimize strategy weights based on current market conditions
        
        Args:
            market_conditions: Current market condition data
            
        Returns:
            Optimized strategy weights
        """
        try:
            base_weights = self.get_normalized_strategy_weights()
            optimized_weights = base_weights.copy()
            
            volatility = market_conditions.get('volatility_regime', 'medium')
            trend_strength = market_conditions.get('trend_strength', 'medium')
            market_regime = market_conditions.get('market_regime', 'ranging')
            
            # Adjust based on volatility
            if volatility == 'high':
                # KAMA adapts well to volatility, reduce BB sensitivity
                optimized_weights['kama'] *= 1.3
                optimized_weights['bb_supertrend'] *= 0.8
                optimized_weights['ema'] *= 0.9
            elif volatility == 'low':
                # EMA works well in steady conditions
                optimized_weights['ema'] *= 1.2
                optimized_weights['bb_supertrend'] *= 1.1
                optimized_weights['kama'] *= 0.9
            
            # Adjust based on trend strength
            if trend_strength == 'strong':
                # EMA and SuperTrend excel in strong trends
                optimized_weights['ema'] *= 1.3
                optimized_weights['bb_supertrend'] *= 1.2
                optimized_weights['macd'] *= 0.8
            elif trend_strength == 'weak':
                # MACD and KAMA better for weak trends
                optimized_weights['macd'] *= 1.2
                optimized_weights['kama'] *= 1.3
                optimized_weights['ema'] *= 0.8
            
            # Adjust based on market regime
            if market_regime == 'trending':
                optimized_weights['ema'] *= 1.2
                optimized_weights['bb_supertrend'] *= 1.1
            elif market_regime == 'ranging':
                optimized_weights['macd'] *= 1.2
                optimized_weights['kama'] *= 1.1
                optimized_weights['ema'] *= 0.8
            
            # Renormalize weights
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
            
            return optimized_weights
            
        except Exception as e:
            self.logger.error(f"❌ Weight optimization failed: {e}")
            return self.get_normalized_strategy_weights()

    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics and performance metrics"""
        try:
            enabled_strategies = [name for name, enabled in self.strategy_flags.items() if enabled]
            
            return {
                'total_strategies_available': len(self.strategy_flags),
                'enabled_strategies_count': len(enabled_strategies),
                'enabled_strategies': enabled_strategies,
                'normalized_weights': self.get_normalized_strategy_weights(),
                'base_weights': self.base_weights,
                'forex_adjustments_available': len(self.forex_confidence_adjustments),
                'cache_size': len(self._cache),
                'configuration_mode': self.combined_config['mode'],
                'minimum_confidence': self.combined_config['min_combined_confidence'],
                'safety_filters_enabled': self.combined_config['enable_safety_filters'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimization stats collection failed: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear the performance cache"""
        self._cache.clear()
        self.logger.debug("🧹 CombinedForexOptimizer cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        return {
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'memory_efficient': True,
            'cache_type': 'performance_optimization'
        }

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics information"""
        return {
            'module_name': 'CombinedForexOptimizer',
            'initialization_successful': True,
            'configuration_loaded': bool(self.combined_config),
            'strategy_flags_loaded': bool(self.strategy_flags),
            'forex_adjustments_loaded': bool(self.forex_confidence_adjustments),
            'cache_operational': True,
            'optimization_stats': self.get_optimization_stats(),
            'timestamp': datetime.now().isoformat()
        }