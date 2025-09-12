# core/strategies/helpers/bb_forex_optimizer.py
"""
BB Forex Optimizer Module - FIXED: Dynamic Configuration Loading
🔥 FOREX: Forex-specific calculations and optimizations for BB+Supertrend
🎯 FOCUSED: Single responsibility for forex market analysis
📊 COMPREHENSIVE: Market regime detection, efficiency ratios, confidence adjustments

CRITICAL FIX: Now properly loads configuration from config.py instead of using hardcoded values
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
import logging
from datetime import datetime
try:
    import config
except ImportError:
    from forex_scanner import config


class BBForexOptimizer:
    """
    🔥 FOREX: Forex-specific calculations and optimizations for BB+Supertrend strategy
    
    Responsibilities:
    - BB+Supertrend configuration management
    - Forex pair type classification
    - Market regime detection
    - Efficiency ratio calculations
    - Forex-specific confidence adjustments
    - Volatility analysis
    """
    
    def __init__(self, logger: logging.Logger = None, config_name: str = 'default'):
        self.logger = logger or logging.getLogger(__name__)
        self.config_name = config_name
        
        # 🔧 FIX: Initialize cache storage properly
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 🔧 CRITICAL FIX: Load BB+Supertrend configuration dynamically from config.py
        self.bb_configs = self._load_dynamic_bb_configs()
        
        # Validate that the requested config exists
        if self.config_name not in self.bb_configs:
            self.logger.warning(f"⚠️ Config '{self.config_name}' not found, falling back to 'default'")
            self.config_name = 'default'
        
        # Forex-specific scaling factors
        self.forex_scaling = {
            'distance_multiplier': 1000,           # Forex-appropriate scaling
            'volatility_threshold_min': 0.0001,    # 0.01% (forex appropriate)
            'volatility_threshold_max': 0.001,     # 0.1% (forex appropriate)
            'bb_tolerance_factor': 0.05,           # 5% of BB width
            'volume_ratio_threshold': 1.2,         # Lower threshold for forex
            'min_bb_separation': 0.0005,           # Minimum BB separation
            'major_pair_multiplier': 1.15,         # 15% boost for major pairs
            'minor_pair_multiplier': 1.05,         # 5% boost for minor pairs
            'exotic_pair_multiplier': 0.95         # 5% penalty for exotic pairs
        }
        
        # Log current configuration for debugging
        current_config = self.get_bb_config()
        self.logger.info(f"🔥 BB Forex Optimizer initialized with cache support")
        self.logger.info(f"  📊 Config: {self.config_name}")
        self.logger.info(f"  📈 BB Period: {current_config['bb_period']}, Std Dev: {current_config['bb_std_dev']}")
        self.logger.info(f"  📉 SuperTrend: Period {current_config['supertrend_period']}, Multiplier {current_config['supertrend_multiplier']}")
        self.logger.info(f"  🎯 Base Confidence: {current_config['base_confidence']:.1%}")
        self.logger.debug(f"  Cache initialized: {len(self._cache)} entries")

    def _load_dynamic_bb_configs(self) -> Dict:
        """
        🔧 CRITICAL FIX: Load BB configurations dynamically from config.py
        This ensures that any config changes are immediately reflected
        """
        try:
            # First try to get the configurations from config.py
            configs_from_file = getattr(config, 'BB_SUPERTREND_CONFIGS', None)
            
            if configs_from_file:
                self.logger.info(f"✅ Loaded {len(configs_from_file)} BB configs from config.py")
                for config_name, config_data in configs_from_file.items():
                    self.logger.debug(f"  📊 {config_name}: BB({config_data.get('bb_period', 'N/A')}, {config_data.get('bb_std_dev', 'N/A')}) ST({config_data.get('supertrend_period', 'N/A')}, {config_data.get('supertrend_multiplier', 'N/A')})")
                return configs_from_file
            else:
                self.logger.warning("⚠️ No BB_SUPERTREND_CONFIGS found in config.py, using fallback")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load BB configs from config.py: {e}")
        
        # Fallback configurations (only used if config.py loading fails)
        fallback_configs = {
            'conservative': {
                'bb_period': 24,
                'bb_std_dev': 2.8,
                'supertrend_period': 16,
                'supertrend_multiplier': 4.0,
                'base_confidence': 0.75,
                'min_bb_width_pct': 0.0015,
                'max_signals_per_session': 3,
                'require_mtf_confluence': True,
                'min_mtf_confluence': 0.6
            },
            'balanced': {
                'bb_period': 20,
                'bb_std_dev': 2.2,
                'supertrend_period': 12,
                'supertrend_multiplier': 3.2,
                'base_confidence': 0.65,
                'min_bb_width_pct': 0.0012,
                'max_signals_per_session': 5,
                'require_mtf_confluence': True,
                'min_mtf_confluence': 0.45
            },
            'default': {
                'bb_period': 20,
                'bb_std_dev': 2.2,
                'supertrend_period': 10,
                'supertrend_multiplier': 3.0,
                'base_confidence': 0.6,
                'min_bb_width_pct': 0.001,
                'max_signals_per_session': 6,
                'require_mtf_confluence': False,
                'min_mtf_confluence': 0.35
            },
            'aggressive': {
                'bb_period': 16,
                'bb_std_dev': 2.0,
                'supertrend_period': 8,
                'supertrend_multiplier': 2.8,
                'base_confidence': 0.55,
                'min_bb_width_pct': 0.0008,
                'max_signals_per_session': 8,
                'require_mtf_confluence': False,
                'min_mtf_confluence': 0.25
            }
        }
        
        self.logger.warning(f"⚠️ Using fallback BB configs: {list(fallback_configs.keys())}")
        return fallback_configs

    def reload_config(self):
        """
        🔄 Reload configuration from config.py (useful for testing different settings)
        """
        old_config = self.get_bb_config().copy()
        self.bb_configs = self._load_dynamic_bb_configs()
        new_config = self.get_bb_config()
        
        # Log changes
        if old_config != new_config:
            self.logger.info(f"🔄 BB Config reloaded for '{self.config_name}':")
            for key in ['bb_period', 'bb_std_dev', 'supertrend_period', 'supertrend_multiplier', 'base_confidence']:
                if key in old_config and key in new_config:
                    old_val = old_config[key]
                    new_val = new_config[key]
                    if old_val != new_val:
                        self.logger.info(f"  📊 {key}: {old_val} → {new_val}")
            
            # Clear cache since config changed
            self.clear_cache()
        else:
            self.logger.debug(f"🔄 BB Config unchanged after reload")

    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from parameters"""
        try:
            import hashlib
            key_string = f"{prefix}_" + "_".join(str(arg) for arg in args)
            return hashlib.md5(key_string.encode()).hexdigest()[:16]
        except Exception:
            return f"{prefix}_default"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached value is still valid"""
        if cache_key not in self._cache:
            return False
        
        cache_entry = self._cache[cache_key]
        cache_age = (datetime.now() - cache_entry['timestamp']).total_seconds()
        return cache_age < self._cache_timeout

    def _cache_result(self, cache_key: str, value: Any):
        """Cache a calculation result"""
        try:
            self._cache[cache_key] = {
                'value': value,
                'timestamp': datetime.now()
            }
            
            # Clean old entries periodically
            if len(self._cache) > 100:  # Max 100 entries
                self._clean_old_cache_entries()
        except Exception as e:
            self.logger.debug(f"Cache storage failed: {e}")

    def _clean_old_cache_entries(self):
        """Remove expired cache entries"""
        try:
            current_time = datetime.now()
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if (current_time - entry['timestamp']).total_seconds() > self._cache_timeout
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                
            if keys_to_remove:
                self.logger.debug(f"🧹 Cleaned {len(keys_to_remove)} expired forex optimizer cache entries")
        except Exception as e:
            self.logger.debug(f"Cache cleanup failed: {e}")

    def get_bb_config(self) -> Dict:
        """Get current BB+Supertrend configuration with live config reload"""
        # Always reload to ensure we have the latest config
        try:
            # Check if config has been updated in config.py
            latest_configs = getattr(config, 'BB_SUPERTREND_CONFIGS', None)
            if latest_configs and latest_configs != self.bb_configs:
                self.logger.debug(f"🔄 Detected config changes, reloading...")
                self.bb_configs = latest_configs
                self.clear_cache()  # Clear cache when config changes
        except Exception as e:
            self.logger.debug(f"Config reload check failed: {e}")
        
        return self.bb_configs.get(self.config_name, self.bb_configs.get('default', {}))

    def get_forex_pair_type(self, epic: str) -> str:
        """
        🎯 Classify forex pair type for strategy optimization
        """
        cache_key = self._generate_cache_key("pair_type", epic)
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            epic_upper = epic.upper()
            
            # Major pairs
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            if any(pair in epic_upper for pair in major_pairs):
                result = 'MAJOR'
            # Minor pairs (cross currencies)
            elif any(cross in epic_upper for cross in ['EURGBP', 'EURJPY', 'GBPJPY', 'EURCHF', 'AUDJPY']):
                result = 'MINOR'
            # Exotic pairs
            else:
                result = 'EXOTIC'
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.debug(f"Pair type classification failed: {e}")
            result = 'UNKNOWN'
            self._cache_result(cache_key, result)
            return result

    def detect_forex_market_regime(self, current: pd.Series, df: pd.DataFrame = None) -> str:
        """
        📊 Detect current market regime for forex optimization
        """
        cache_key = self._generate_cache_key("market_regime", current.get('close', 0), current.get('atr', 0))
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            atr = current.get('atr', 0.001)
            bb_width = current.get('bb_upper', 0) - current.get('bb_lower', 0)
            
            # Use ATR-based volatility assessment
            if atr > self.forex_scaling['volatility_threshold_max']:
                regime = 'HIGH_VOLATILITY'
            elif atr < self.forex_scaling['volatility_threshold_min']:
                regime = 'LOW_VOLATILITY'
            else:
                regime = 'NORMAL_VOLATILITY'
            
            # Enhance with BB width if available
            if bb_width > 0:
                bb_volatility_ratio = bb_width / current.get('close', 1)
                if bb_volatility_ratio > 0.002:  # 0.2% of price
                    if regime == 'NORMAL_VOLATILITY':
                        regime = 'HIGH_VOLATILITY'
                elif bb_volatility_ratio < 0.0005:  # 0.05% of price
                    if regime == 'NORMAL_VOLATILITY':
                        regime = 'LOW_VOLATILITY'
            
            self._cache_result(cache_key, regime)
            return regime
            
        except Exception as e:
            self.logger.debug(f"Market regime detection failed: {e}")
            result = 'UNKNOWN'
            self._cache_result(cache_key, result)
            return result

    def calculate_forex_efficiency_ratio(self, current: pd.Series, previous: pd.Series) -> float:
        """
        📈 Calculate efficiency ratio optimized for forex markets
        """
        cache_key = self._generate_cache_key("efficiency", current.get('close', 0), previous.get('close', 0))
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            price_change = abs(current.get('close', 0) - previous.get('close', 0))
            atr = current.get('atr', 0.001)
            
            if atr > 0:
                efficiency_ratio = min(price_change / atr, 1.0)
            else:
                efficiency_ratio = 0.5  # Neutral value
            
            # Forex-specific adjustments
            efficiency_ratio *= self.forex_scaling['distance_multiplier']
            efficiency_ratio = min(efficiency_ratio, 1.0)  # Cap at 1.0
            
            self._cache_result(cache_key, efficiency_ratio)
            return efficiency_ratio
            
        except Exception as e:
            self.logger.debug(f"Efficiency ratio calculation failed: {e}")
            result = 0.25  # Conservative fallback
            self._cache_result(cache_key, result)
            return result

    def assess_bb_volatility(self, current: pd.Series) -> str:
        """
        🌡️ Assess Bollinger Band volatility for forex optimization
        """
        cache_key = self._generate_cache_key("bb_volatility", current.get('bb_upper', 0), current.get('bb_lower', 0))
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            bb_upper = current.get('bb_upper', 0)
            bb_lower = current.get('bb_lower', 0)
            current_price = current.get('close', 0)
            
            if bb_upper <= bb_lower or current_price == 0:
                result = 'INSUFFICIENT_DATA'
                self._cache_result(cache_key, result)
                return result
            
            bb_width = bb_upper - bb_lower
            bb_width_pct = bb_width / current_price
            
            # 🔧 ENHANCED: Use config-based thresholds if available
            config_data = self.get_bb_config()
            min_width_pct = config_data.get('min_bb_width_pct', 0.001)
            
            # Forex-appropriate thresholds with config enhancement
            if bb_width_pct > min_width_pct * 2:  # 2x minimum = HIGH
                result = 'HIGH'
            elif bb_width_pct < min_width_pct * 0.5:  # 0.5x minimum = LOW
                result = 'LOW'
            else:
                result = 'NORMAL'
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.debug(f"BB volatility assessment failed: {e}")
            result = 'UNKNOWN'
            self._cache_result(cache_key, result)
            return result

    def is_bb_width_sufficient(self, current: pd.Series) -> bool:
        """
        ✅ Check if BB width is sufficient for reliable signals
        """
        cache_key = self._generate_cache_key("bb_width_check", current.get('bb_upper', 0), current.get('bb_lower', 0))
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            bb_width = current.get('bb_upper', 0) - current.get('bb_lower', 0)
            current_price = current.get('close', 0)
            
            if current_price == 0:
                result = False
                self._cache_result(cache_key, result)
                return result
            
            # Use config-based minimum width if available
            config_data = self.get_bb_config()
            min_width_pct = config_data.get('min_bb_width_pct', 0.001)
            bb_width_pct = bb_width / current_price
            
            result = bb_width_pct >= min_width_pct
            
            self._cache_result(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.debug(f"BB width check failed: {e}")
            result = False
            self._cache_result(cache_key, result)
            return result

    def calculate_bb_position_score(self, current: pd.Series, signal_type: str) -> float:
        """
        📍 Calculate BB position quality score for signal validation
        """
        cache_key = self._generate_cache_key("bb_position", current.get('close', 0), signal_type)
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            bb_upper = current.get('bb_upper', 0)
            bb_lower = current.get('bb_lower', 0)
            current_price = current.get('close', 0)
            
            if bb_upper <= bb_lower or current_price == 0:
                result = 0.0
                self._cache_result(cache_key, result)
                return result
            
            bb_width = bb_upper - bb_lower
            
            if signal_type == 'BULL':
                # For BULL signals, we want price near the lower band
                distance_from_lower = current_price - bb_lower
                position_score = max(0.0, 1.0 - (distance_from_lower / (bb_width * 0.5)))
            else:  # BEAR
                # For BEAR signals, we want price near the upper band
                distance_from_upper = bb_upper - current_price
                position_score = max(0.0, 1.0 - (distance_from_upper / (bb_width * 0.5)))
            
            self._cache_result(cache_key, position_score)
            return position_score
            
        except Exception as e:
            self.logger.debug(f"BB position score calculation failed: {e}")
            result = 0.0
            self._cache_result(cache_key, result)
            return result

    def apply_forex_confidence_adjustments(
        self, 
        base_confidence: float, 
        epic: str, 
        market_regime: str, 
        bb_position_score: float
    ) -> float:
        """
        🎯 Apply forex-specific confidence adjustments
        """
        cache_key = self._generate_cache_key("forex_confidence", epic, market_regime, base_confidence, bb_position_score)
        
        if self._is_cache_valid(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]['value']
        
        self._cache_misses += 1
        
        try:
            adjusted_confidence = base_confidence
            
            # Pair type adjustments
            pair_type = self.get_forex_pair_type(epic)
            if pair_type == 'MAJOR':
                adjusted_confidence *= self.forex_scaling['major_pair_multiplier']
            elif pair_type == 'MINOR':
                adjusted_confidence *= self.forex_scaling['minor_pair_multiplier']
            elif pair_type == 'EXOTIC':
                adjusted_confidence *= self.forex_scaling['exotic_pair_multiplier']
            
            # Market regime adjustments
            if market_regime == 'HIGH_VOLATILITY':
                adjusted_confidence *= 0.9  # Reduce confidence in high volatility
            elif market_regime == 'LOW_VOLATILITY':
                adjusted_confidence *= 1.1  # Increase confidence in low volatility
            
            # BB position score adjustments
            if bb_position_score > 0.8:
                adjusted_confidence *= 1.1  # Strong BB position
            elif bb_position_score < 0.3:
                adjusted_confidence *= 0.9  # Weak BB position
            
            # Ensure reasonable bounds
            adjusted_confidence = max(0.1, min(adjusted_confidence, 0.95))
            
            self._cache_result(cache_key, adjusted_confidence)
            return adjusted_confidence
            
        except Exception as e:
            self.logger.debug(f"Forex confidence adjustments failed: {e}")
            result = base_confidence
            self._cache_result(cache_key, result)
            return result

    def get_bb_tolerance(self, current: pd.Series) -> float:
        """
        📏 Get Bollinger Band tolerance for signal validation
        """
        try:
            bb_width = current.get('bb_upper', 0) - current.get('bb_lower', 0)
            return bb_width * self.forex_scaling['bb_tolerance_factor']
        except Exception as e:
            self.logger.debug(f"BB tolerance calculation failed: {e}")
            return 0.0001  # Default fallback

    def clear_cache(self):
        """🧹 Clear cached calculations"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.debug("🧹 BB Forex Optimizer cache cleared")

    def get_cache_stats(self) -> Dict:
        """📊 Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        return {
            'cached_entries': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_ratio': self._cache_hits / max(total_requests, 1),
            'cache_timeout': self._cache_timeout,
            'forex_scaling_loaded': len(self.forex_scaling),
            'bb_configs_loaded': len(self.bb_configs),
            'current_config': self.config_name,
            'current_config_data': self.get_bb_config()
        }

    def debug_forex_analysis(self, df: pd.DataFrame, epic: str, timeframe: str) -> Dict:
        """
        🔍 Get comprehensive forex analysis for debugging
        """
        try:
            if len(df) == 0:
                return {'error': 'Empty DataFrame'}
            
            current = df.iloc[-1]
            
            debug_info = {
                'forex_analysis': {
                    'pair_type': self.get_forex_pair_type(epic),
                    'market_regime': self.detect_forex_market_regime(current, df),
                    'efficiency_ratio': self.calculate_forex_efficiency_ratio(current, df.iloc[-2] if len(df) > 1 else current),
                    'volatility_assessment': self.assess_bb_volatility(current),
                    'bb_width_sufficient': self.is_bb_width_sufficient(current),
                    'bb_tolerance': self.get_bb_tolerance(current),
                    'current_config': self.get_bb_config(),
                    'cache_stats': self.get_cache_stats()
                }
            }
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}