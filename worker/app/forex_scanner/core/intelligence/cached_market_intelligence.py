# core/intelligence/cached_market_intelligence.py
"""
Cached Market Intelligence Engine
Ultra-fast market intelligence with background workers and intelligent caching.
Optimized for high-frequency trading with minimal latency impact.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import json

try:
    from .market_intelligence import MarketIntelligenceEngine
    from configdata.market_intelligence_config import (
        REGIME_STRATEGY_CONFIDENCE_MODIFIERS,
        ENABLE_PROBABILISTIC_CONFIDENCE_MODIFIERS,
        MIN_CONFIDENCE_MODIFIER
    )
except ImportError:
    # Fallback imports
    pass


class CachedMarketIntelligenceEngine:
    """
    High-performance cached market intelligence engine with background updates.
    Designed for ultra-low latency signal processing.
    """

    def __init__(self, data_fetcher, cache_ttl_seconds: int = 30):
        self.data_fetcher = data_fetcher
        self.cache_ttl_seconds = cache_ttl_seconds
        self.logger = logging.getLogger(__name__)

        # Initialize the underlying market intelligence engine
        try:
            self.intelligence_engine = MarketIntelligenceEngine(data_fetcher)
            self.has_intelligence_engine = True
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize MarketIntelligenceEngine: {e}")
            self.intelligence_engine = None
            self.has_intelligence_engine = False

        # Cache storage
        self._cache_lock = threading.RLock()
        self._regime_cache = {}
        self._confidence_cache = {}
        self._last_update_time = {}

        # Background worker
        self._background_worker_running = False
        self._background_thread = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="MI_Cache")

        # Performance metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._background_updates = 0

        self.logger.info("🚀 CachedMarketIntelligenceEngine initialized")

    def start_background_worker(self, epic_list: List[str], update_interval_seconds: int = 30):
        """Start background worker to update market intelligence cache"""
        if self._background_worker_running:
            self.logger.warning("🔄 Background worker already running")
            return

        self._background_worker_running = True
        self._background_thread = threading.Thread(
            target=self._background_update_worker,
            args=(epic_list, update_interval_seconds),
            daemon=True,
            name="MI_Background_Worker"
        )
        self._background_thread.start()
        self.logger.info(f"🏃 Background worker started - updating every {update_interval_seconds}s")

    def stop_background_worker(self):
        """Stop background worker"""
        self._background_worker_running = False
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5)
        self.logger.info("🛑 Background worker stopped")

    def _background_update_worker(self, epic_list: List[str], update_interval: int):
        """Background worker to update cache periodically"""
        while self._background_worker_running:
            try:
                self.logger.debug("🔄 Background worker updating market intelligence cache...")
                self._update_cache_for_epics(epic_list)
                self._background_updates += 1

                # Sleep in small intervals to allow quick shutdown
                for _ in range(update_interval):
                    if not self._background_worker_running:
                        break
                    time.sleep(1)

            except Exception as e:
                self.logger.error(f"❌ Background worker error: {e}")
                time.sleep(10)  # Wait before retrying

    def _update_cache_for_epics(self, epic_list: List[str]):
        """Update cache for specific epics"""
        if not self.has_intelligence_engine:
            return

        try:
            # Generate fresh intelligence report
            report = self.intelligence_engine.generate_market_intelligence_report(epic_list)

            if report:
                current_time = time.time()

                with self._cache_lock:
                    # Cache the regime analysis
                    market_regime = report.get('market_regime', {})
                    self._regime_cache['global'] = {
                        'data': market_regime,
                        'timestamp': current_time,
                        'epic_list': epic_list.copy()
                    }

                    # Pre-compute confidence modifiers for all strategies
                    self._precompute_confidence_modifiers(market_regime)

                    # Update last update time
                    self._last_update_time['global'] = current_time

                self.logger.debug(f"✅ Cache updated for {len(epic_list)} epics")

        except Exception as e:
            self.logger.error(f"❌ Cache update failed: {e}")

    def _precompute_confidence_modifiers(self, market_regime: Dict):
        """Pre-compute confidence modifiers for all strategy-regime combinations"""
        dominant_regime = market_regime.get('dominant_regime', 'unknown')

        if not ENABLE_PROBABILISTIC_CONFIDENCE_MODIFIERS:
            return

        regime_modifiers = REGIME_STRATEGY_CONFIDENCE_MODIFIERS.get(dominant_regime, {})

        # Store pre-computed modifiers
        cache_key = f"modifiers_{dominant_regime}"
        self._confidence_cache[cache_key] = {
            'modifiers': regime_modifiers,
            'regime': dominant_regime,
            'timestamp': time.time()
        }

    def get_fast_regime_analysis(self, epic_list: List[str] = None) -> Optional[Dict]:
        """
        Ultra-fast regime analysis using cache with fallback to live generation.
        Returns cached data if available and fresh, otherwise generates new data.
        """
        current_time = time.time()

        # Check cache first
        with self._cache_lock:
            cached_regime = self._regime_cache.get('global')

            if cached_regime:
                cache_age = current_time - cached_regime['timestamp']

                if cache_age < self.cache_ttl_seconds:
                    self._cache_hits += 1
                    self.logger.debug(f"🎯 Cache HIT - regime data age: {cache_age:.1f}s")
                    return cached_regime['data']
                else:
                    self.logger.debug(f"⏰ Cache EXPIRED - age: {cache_age:.1f}s > TTL: {self.cache_ttl_seconds}s")

        # Cache miss or expired - generate fresh data
        self._cache_misses += 1

        if not self.has_intelligence_engine:
            self.logger.warning("🚫 No intelligence engine available")
            return None

        try:
            self.logger.debug("🔄 Generating fresh regime analysis...")
            epic_list = epic_list or ['CS.D.EURUSD.CEEM.IP']  # Default epic

            regime_analysis = self.intelligence_engine.analyze_market_regime(epic_list)

            # Cache the fresh data
            with self._cache_lock:
                self._regime_cache['global'] = {
                    'data': regime_analysis,
                    'timestamp': current_time,
                    'epic_list': epic_list
                }
                self._precompute_confidence_modifiers(regime_analysis)

            self.logger.debug("✅ Fresh regime analysis generated and cached")
            return regime_analysis

        except Exception as e:
            self.logger.error(f"❌ Fresh regime analysis failed: {e}")
            return None

    def get_fast_confidence_modifier(self, strategy: str, regime: str = None) -> float:
        """
        Ultra-fast confidence modifier lookup with O(1) performance.
        Returns confidence modifier for strategy-regime combination.
        """
        if not ENABLE_PROBABILISTIC_CONFIDENCE_MODIFIERS:
            return 1.0

        # Use provided regime or get from cache
        if regime is None:
            regime_data = self.get_fast_regime_analysis()
            if not regime_data:
                return 0.7  # Conservative default
            regime = regime_data.get('dominant_regime', 'unknown')

        # Check pre-computed cache
        cache_key = f"modifiers_{regime}"

        with self._cache_lock:
            cached_modifiers = self._confidence_cache.get(cache_key)

            if cached_modifiers:
                modifiers = cached_modifiers['modifiers']
                strategy_lower = strategy.lower()

                # Fast lookup in pre-computed modifiers
                for strategy_key, modifier in modifiers.items():
                    if strategy_key in strategy_lower or strategy_lower in strategy_key:
                        self.logger.debug(f"🎯 Fast modifier lookup: {strategy} in {regime} = {modifier:.1%}")
                        return modifier

        # Fallback to conservative default
        self.logger.debug(f"🔍 No cached modifier for {strategy} in {regime}, using default: 70%")
        return 0.7

    def validate_signal_with_cache(self, signal: Dict) -> Tuple[bool, str, Dict]:
        """
        Ultra-fast signal validation using cached intelligence data.
        Returns (is_valid, reason, intelligence_data)
        """
        epic = signal.get('epic', 'Unknown')
        strategy = signal.get('strategy', 'Unknown')

        # Get cached regime analysis
        regime_data = self.get_fast_regime_analysis([epic])

        if not regime_data:
            return True, "Market intelligence unavailable - allowing signal", {}

        # Fast confidence modifier lookup
        confidence_modifier = self.get_fast_confidence_modifier(strategy)

        # Apply minimum threshold check
        if confidence_modifier < MIN_CONFIDENCE_MODIFIER:
            regime = regime_data.get('dominant_regime', 'unknown')
            reason = f"Strategy '{strategy}' confidence modifier {confidence_modifier:.1%} below minimum {MIN_CONFIDENCE_MODIFIER:.1%} for {regime} regime"
            return False, reason, regime_data

        # Store modifier in signal for potential use
        signal['market_intelligence_confidence_modifier'] = confidence_modifier

        regime = regime_data.get('dominant_regime', 'unknown')
        regime_confidence = regime_data.get('confidence', 0.5)

        success_msg = f"Cached intelligence approved: {regime} regime ({regime_confidence:.1%}), modifier: {confidence_modifier:.1%}"

        return True, success_msg, regime_data

    def get_performance_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': hit_rate,
            'background_updates': self._background_updates,
            'cache_size': len(self._regime_cache) + len(self._confidence_cache),
            'worker_running': self._background_worker_running
        }

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_background_worker()
            self._executor.shutdown(wait=False)
        except:
            pass