# core/strategies/helpers/bb_cache.py
"""
BB Cache Module - Extracted from BB Supertrend Strategy
🚀 PERFORMANCE: Intelligent caching and optimization for BB+Supertrend calculations
🎯 FOCUSED: Single responsibility for BB performance optimization
📊 COMPREHENSIVE: Cache management, performance tracking, optimization

This module contains all the caching and performance optimization logic
for BB+Supertrend strategy, extracted for better maintainability and testability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
from datetime import datetime
import hashlib


class BBCache:
    """
    🚀 PERFORMANCE: Intelligent caching system for BB+Supertrend strategy
    
    Responsibilities:
    - Calculation result caching
    - Performance monitoring and optimization
    - Cache invalidation and cleanup
    - Memory management
    - Performance statistics tracking
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache storage
        self._calculation_cache = {}
        self._market_regime_cache = {}
        self._efficiency_ratio_cache = {}
        self._bb_analysis_cache = {}
        
        # Cache configuration
        self._cache_timeout = 300  # 5 minutes
        self._max_cache_entries = 100  # Prevent memory bloat
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0
        self._cache_creation_time = datetime.now()
        
        self.logger.info("🚀 BB Cache system initialized")

    def _generate_cache_key(self, prefix: str, *args) -> str:
        """
        🔑 Generate unique cache key from parameters
        """
        try:
            # Convert all arguments to strings and create hash
            key_string = f"{prefix}_" + "_".join(str(arg) for arg in args)
            return hashlib.md5(key_string.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.debug(f"Cache key generation error: {e}")
            return f"{prefix}_default"

    def _is_cache_valid(self, cache_key: str, cache_dict: Dict = None) -> bool:
        """
        ✅ Check if cached value is still valid
        """
        cache_dict = cache_dict or self._calculation_cache
        
        if cache_key not in cache_dict:
            return False
        
        cache_entry = cache_dict[cache_key]
        cache_age = (datetime.now() - cache_entry['timestamp']).total_seconds()
        
        return cache_age < self._cache_timeout

    def _cache_result(self, cache_key: str, value: Any, cache_dict: Dict = None):
        """
        💾 Cache a calculation result with timestamp
        """
        try:
            cache_dict = cache_dict or self._calculation_cache
            
            cache_dict[cache_key] = {
                'value': value,
                'timestamp': datetime.now()
            }
            
            # Clean old cache entries periodically
            if len(cache_dict) > self._max_cache_entries:
                self._clean_old_cache_entries(cache_dict)
                
        except Exception as e:
            self.logger.debug(f"Cache storage failed: {e}")

    def _clean_old_cache_entries(self, cache_dict: Dict = None):
        """
        🧹 Remove expired cache entries
        """
        try:
            cache_dict = cache_dict or self._calculation_cache
            current_time = datetime.now()
            
            keys_to_remove = [
                key for key, entry in cache_dict.items()
                if (current_time - entry['timestamp']).total_seconds() > self._cache_timeout
            ]
            
            for key in keys_to_remove:
                del cache_dict[key]
                
            if keys_to_remove:
                self.logger.debug(f"🧹 Cleaned {len(keys_to_remove)} expired cache entries")
                
        except Exception as e:
            self.logger.debug(f"Cache cleanup failed: {e}")

    def calculate_efficiency_ratio_cached(self, current: pd.Series, previous: pd.Series, epic: str, timeframe: str) -> float:
        """
        🚀 CACHED: Calculate efficiency ratio with intelligent caching
        """
        # Create cache key
        price_key = f"{current.get('close', 0):.5f}"
        cache_key = self._generate_cache_key("efficiency", epic, timeframe, price_key)
        
        # Check cache first
        if self._is_cache_valid(cache_key, self._efficiency_ratio_cache):
            self._cache_hits += 1
            return self._efficiency_ratio_cache[cache_key]['value']
        
        # Cache miss - calculate
        self._cache_misses += 1
        self._total_calculations += 1
        
        try:
            # Calculate efficiency ratio
            price_change = abs(current.get('close', 0) - previous.get('close', 0))
            atr = current.get('atr', 0.001)
            
            if atr > 0:
                efficiency_ratio = min(1.0, price_change / atr)
                efficiency_ratio = max(0.1, efficiency_ratio)  # Floor at 10%
            else:
                efficiency_ratio = 0.25  # Default reasonable value
            
            # Cache the result
            self._cache_result(cache_key, efficiency_ratio, self._efficiency_ratio_cache)
            return efficiency_ratio
            
        except Exception as e:
            self.logger.error(f"❌ Cached efficiency ratio calculation failed: {e}")
            efficiency_ratio = 0.25
            self._cache_result(cache_key, efficiency_ratio, self._efficiency_ratio_cache)
            return efficiency_ratio

    def detect_market_regime_cached(self, current: pd.Series, df: pd.DataFrame, epic: str, timeframe: str) -> str:
        """
        🚀 CACHED: Market regime detection with caching
        """
        # Create cache key
        price_key = f"{current.get('close', 0):.5f}"
        atr_key = f"{current.get('atr', 0.001):.6f}"
        cache_key = self._generate_cache_key("regime", epic, timeframe, price_key, atr_key)
        
        # Check cache first
        if self._is_cache_valid(cache_key, self._market_regime_cache):
            self._cache_hits += 1
            return self._market_regime_cache[cache_key]['value']
        
        # Cache miss - calculate
        self._cache_misses += 1
        self._total_calculations += 1
        
        try:
            # Use ATR from current candle (already calculated)
            atr = current.get('atr', 0.001)
            current_price = current.get('close', 0)
            
            if current_price <= 0:
                regime = 'ranging'
            else:
                # Calculate volatility ratio using available data
                volatility_ratio = atr / current_price
                
                # BB width as additional volatility measure
                bb_width = current.get('bb_upper', 0) - current.get('bb_lower', 0)
                bb_width_ratio = bb_width / current_price if current_price > 0 else 0
                
                # Combined volatility assessment
                if volatility_ratio > 0.002 or bb_width_ratio > 0.003:  # High volatility
                    regime = 'volatile'
                elif volatility_ratio > 0.001 or bb_width_ratio > 0.002:  # Moderate volatility
                    regime = 'trending'
                elif volatility_ratio < 0.0005 and bb_width_ratio < 0.001:  # Very low volatility
                    regime = 'consolidating'
                else:
                    regime = 'ranging'
            
            # Cache the result
            self._cache_result(cache_key, regime, self._market_regime_cache)
            return regime
            
        except Exception as e:
            self.logger.debug(f"Cached market regime detection failed: {e}")
            regime = 'ranging'
            self._cache_result(cache_key, regime, self._market_regime_cache)
            return regime

    def analyze_bb_position_cached(self, current: pd.Series, signal_type: str, epic: str) -> Dict:
        """
        🚀 CACHED: BB position analysis with caching
        """
        # Create cache key
        price_key = f"{current.get('close', 0):.5f}"
        bb_key = f"{current.get('bb_upper', 0):.5f}_{current.get('bb_lower', 0):.5f}"
        cache_key = self._generate_cache_key("bb_analysis", epic, signal_type, price_key, bb_key)
        
        # Check cache first
        if self._is_cache_valid(cache_key, self._bb_analysis_cache):
            self._cache_hits += 1
            return self._bb_analysis_cache[cache_key]['value']
        
        # Cache miss - calculate
        self._cache_misses += 1
        self._total_calculations += 1
        
        try:
            bb_width = current['bb_upper'] - current['bb_lower']
            bb_middle = current['bb_middle']
            current_price = current['close']
            
            analysis = {
                'bb_width': bb_width,
                'bb_width_pct': bb_width / current_price if current_price > 0 else 0,
                'distance_from_middle': abs(current_price - bb_middle),
                'relative_bb_position': (current_price - bb_middle) / (bb_width / 2) if bb_width > 0 else 0
            }
            
            if signal_type == 'BULL':
                analysis['distance_from_target_band'] = current_price - current['bb_lower']
                analysis['band_proximity_score'] = max(0.0, 1.0 - (analysis['distance_from_target_band'] / (bb_width * 0.5)))
            else:  # BEAR
                analysis['distance_from_target_band'] = current['bb_upper'] - current_price
                analysis['band_proximity_score'] = max(0.0, 1.0 - (analysis['distance_from_target_band'] / (bb_width * 0.5)))
            
            # Cache the result
            self._cache_result(cache_key, analysis, self._bb_analysis_cache)
            return analysis
            
        except Exception as e:
            self.logger.debug(f"Cached BB analysis failed: {e}")
            analysis = {'error': str(e)}
            self._cache_result(cache_key, analysis, self._bb_analysis_cache)
            return analysis

    def get_cache_stats(self) -> Dict:
        """
        📊 Get comprehensive cache performance statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        cache_uptime = (datetime.now() - self._cache_creation_time).total_seconds()
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_ratio': self._cache_hits / max(total_requests, 1),
            'miss_ratio': self._cache_misses / max(total_requests, 1),
            'total_calculations': self._total_calculations,
            'cache_efficiency': (total_requests - self._total_calculations) / max(total_requests, 1),
            'cached_entries': {
                'calculation_cache': len(self._calculation_cache),
                'market_regime_cache': len(self._market_regime_cache),
                'efficiency_ratio_cache': len(self._efficiency_ratio_cache),
                'bb_analysis_cache': len(self._bb_analysis_cache),
                'total': len(self._calculation_cache) + len(self._market_regime_cache) + 
                        len(self._efficiency_ratio_cache) + len(self._bb_analysis_cache)
            },
            'cache_config': {
                'timeout': self._cache_timeout,
                'max_entries': self._max_cache_entries
            },
            'cache_uptime_seconds': cache_uptime,
            'cache_performance': 'excellent' if self._cache_hits / max(total_requests, 1) > 0.8 else
                                'good' if self._cache_hits / max(total_requests, 1) > 0.6 else
                                'poor'
        }

    def clear_cache(self):
        """
        🧹 Clear all cached calculations
        """
        self._calculation_cache.clear()
        self._market_regime_cache.clear()
        self._efficiency_ratio_cache.clear()
        self._bb_analysis_cache.clear()
        
        # Reset performance counters
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calculations = 0
        self._cache_creation_time = datetime.now()
        
        self.logger.info("🧹 BB Cache cleared - all calculations will be recalculated")

    def optimize_cache_settings(self, new_timeout: int = None, new_max_entries: int = None):
        """
        ⚙️ Optimize cache settings based on usage patterns
        """
        try:
            if new_timeout is not None:
                self._cache_timeout = new_timeout
                self.logger.info(f"⚙️ Cache timeout updated to {new_timeout} seconds")
            
            if new_max_entries is not None:
                self._max_cache_entries = new_max_entries
                self.logger.info(f"⚙️ Max cache entries updated to {new_max_entries}")
            
            # Clean up if new max is lower than current entries
            if new_max_entries and len(self._calculation_cache) > new_max_entries:
                self._clean_old_cache_entries()
                
        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")

    def log_cache_performance(self):
        """
        📊 Log cache performance metrics
        """
        try:
            stats = self.get_cache_stats()
            self.logger.info(f"🚀 BB Cache Performance Summary:")
            self.logger.info(f"   Hit Ratio: {stats['hit_ratio']:.1%} ({stats['cache_hits']} hits, {stats['cache_misses']} misses)")
            self.logger.info(f"   Cache Efficiency: {stats['cache_efficiency']:.1%}")
            self.logger.info(f"   Total Cached Entries: {stats['cached_entries']['total']}")
            self.logger.info(f"   Cache Performance: {stats['cache_performance']}")
            
        except Exception as e:
            self.logger.error(f"Cache performance logging failed: {e}")