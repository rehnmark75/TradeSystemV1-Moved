# core/intelligence/intelligence_fallback_manager.py
"""
Intelligent Fallback Manager for Market Intelligence
5-level fallback hierarchy ensuring signals are never blocked by intelligence processing failures.
Designed for high-reliability automated trading systems.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass


class FallbackLevel(Enum):
    """Fallback hierarchy levels"""
    LEVEL_0_CACHED = "cached_intelligence"      # Fastest: Use cached data
    LEVEL_1_FAST = "fast_generation"           # Fast: Generate with timeout
    LEVEL_2_BASIC = "basic_regime"             # Basic: Simple regime detection
    LEVEL_3_CONFIDENCE = "confidence_only"    # Confidence: Only confidence-based filtering
    LEVEL_4_BYPASS = "full_bypass"            # Bypass: Allow all signals


@dataclass
class FallbackResult:
    """Result from fallback processing"""
    success: bool
    level_used: FallbackLevel
    intelligence_data: Dict
    confidence_modifier: float
    processing_time_ms: float
    reason: str


class IntelligenceFallbackManager:
    """
    Intelligent fallback manager ensuring market intelligence never blocks the system.
    Implements 5-level fallback hierarchy for maximum reliability.
    """

    def __init__(self, cached_engine=None, timeout_ms: int = 50):
        self.cached_engine = cached_engine
        self.timeout_ms = timeout_ms
        self.logger = logging.getLogger(__name__)

        # Fallback statistics
        self.fallback_stats = {
            FallbackLevel.LEVEL_0_CACHED: 0,
            FallbackLevel.LEVEL_1_FAST: 0,
            FallbackLevel.LEVEL_2_BASIC: 0,
            FallbackLevel.LEVEL_3_CONFIDENCE: 0,
            FallbackLevel.LEVEL_4_BYPASS: 0
        }

        # Quality signal thresholds for bypass decisions
        self.high_quality_confidence_threshold = 0.85  # Signals above this bypass intelligence
        self.medium_quality_confidence_threshold = 0.70

        self.logger.info("🛡️ IntelligenceFallbackManager initialized")
        self.logger.info(f"   ⏱️ Timeout: {timeout_ms}ms")
        self.logger.info(f"   🚀 High-quality bypass: {self.high_quality_confidence_threshold:.1%}")

    def process_signal_with_fallback(self, signal: Dict, epic_list: List[str] = None) -> FallbackResult:
        """
        Process signal with intelligent fallback hierarchy.
        Ensures signal is never blocked by intelligence processing failures.
        """
        start_time = time.time()
        epic = signal.get('epic', 'Unknown')
        strategy = signal.get('strategy', 'Unknown')
        confidence = signal.get('confidence_score', 0.0)

        self.logger.debug(f"🛡️ Starting fallback processing for {epic} {strategy} ({confidence:.1%})")

        # LEVEL 0: Try cached intelligence (fastest)
        try:
            result = self._try_cached_intelligence(signal, epic_list)
            if result.success:
                processing_time = (time.time() - start_time) * 1000
                self.fallback_stats[FallbackLevel.LEVEL_0_CACHED] += 1
                self.logger.debug(f"🎯 LEVEL 0 SUCCESS: Cached intelligence ({processing_time:.1f}ms)")
                return self._create_result(True, FallbackLevel.LEVEL_0_CACHED, result.intelligence_data,
                                           result.confidence_modifier, processing_time, "Cached intelligence")
        except Exception as e:
            self.logger.debug(f"🔄 LEVEL 0 FAILED: {e}")

        # LEVEL 1: Try fast generation with timeout (fast)
        try:
            result = self._try_fast_generation(signal, epic_list)
            if result.success:
                processing_time = (time.time() - start_time) * 1000
                self.fallback_stats[FallbackLevel.LEVEL_1_FAST] += 1
                self.logger.debug(f"⚡ LEVEL 1 SUCCESS: Fast generation ({processing_time:.1f}ms)")
                return self._create_result(True, FallbackLevel.LEVEL_1_FAST, result.intelligence_data,
                                           result.confidence_modifier, processing_time, "Fast generation")
        except Exception as e:
            self.logger.debug(f"🔄 LEVEL 1 FAILED: {e}")

        # LEVEL 2: Basic regime detection (basic)
        try:
            result = self._try_basic_regime_detection(signal)
            if result.success:
                processing_time = (time.time() - start_time) * 1000
                self.fallback_stats[FallbackLevel.LEVEL_2_BASIC] += 1
                self.logger.info(f"🔧 LEVEL 2 SUCCESS: Basic regime detection ({processing_time:.1f}ms)")
                return self._create_result(True, FallbackLevel.LEVEL_2_BASIC, result.intelligence_data,
                                           result.confidence_modifier, processing_time, "Basic regime detection")
        except Exception as e:
            self.logger.debug(f"🔄 LEVEL 2 FAILED: {e}")

        # LEVEL 3: Confidence-only filtering (confidence)
        try:
            result = self._try_confidence_only_filtering(signal)
            processing_time = (time.time() - start_time) * 1000
            self.fallback_stats[FallbackLevel.LEVEL_3_CONFIDENCE] += 1
            self.logger.info(f"🎯 LEVEL 3 SUCCESS: Confidence-only filtering ({processing_time:.1f}ms)")
            return self._create_result(True, FallbackLevel.LEVEL_3_CONFIDENCE, result.intelligence_data,
                                       result.confidence_modifier, processing_time, "Confidence-only filtering")
        except Exception as e:
            self.logger.warning(f"🔄 LEVEL 3 FAILED: {e}")

        # LEVEL 4: Full bypass (emergency)
        processing_time = (time.time() - start_time) * 1000
        self.fallback_stats[FallbackLevel.LEVEL_4_BYPASS] += 1
        self.logger.warning(f"🚨 LEVEL 4 EMERGENCY: Full intelligence bypass ({processing_time:.1f}ms)")

        return self._create_result(True, FallbackLevel.LEVEL_4_BYPASS, {}, 1.0, processing_time,
                                   "Emergency bypass - all intelligence processing failed")

    def _try_cached_intelligence(self, signal: Dict, epic_list: List[str] = None) -> FallbackResult:
        """Level 0: Try cached intelligence"""
        if not self.cached_engine:
            raise Exception("No cached engine available")

        is_valid, reason, intelligence_data = self.cached_engine.validate_signal_with_cache(signal)
        confidence_modifier = signal.get('market_intelligence_confidence_modifier', 1.0)

        return FallbackResult(
            success=is_valid,
            level_used=FallbackLevel.LEVEL_0_CACHED,
            intelligence_data=intelligence_data,
            confidence_modifier=confidence_modifier,
            processing_time_ms=0,  # Will be calculated by caller
            reason=reason
        )

    def _try_fast_generation(self, signal: Dict, epic_list: List[str] = None) -> FallbackResult:
        """Level 1: Try fast generation with timeout"""
        if not self.cached_engine or not hasattr(self.cached_engine, 'intelligence_engine'):
            raise Exception("No intelligence engine available")

        epic = signal.get('epic', 'CS.D.EURUSD.CEEM.IP')
        epic_list = epic_list or [epic]

        # Try to generate with timeout
        regime_data = self.cached_engine.get_fast_regime_analysis(epic_list)
        if not regime_data:
            raise Exception("Fast generation failed")

        # Apply probabilistic scoring
        strategy = signal.get('strategy', 'Unknown')
        confidence_modifier = self.cached_engine.get_fast_confidence_modifier(strategy)

        # Check minimum threshold - get from database config
        min_modifier = 0.3  # Default
        try:
            from forex_scanner.services.intelligence_config_service import get_intelligence_config
            config = get_intelligence_config()
            if config:
                min_modifier = getattr(config, 'min_confidence_modifier', 0.3)
        except Exception:
            pass

        if confidence_modifier < min_modifier:
            raise Exception(f"Confidence modifier {confidence_modifier:.1%} below minimum")

        signal['market_intelligence_confidence_modifier'] = confidence_modifier

        return FallbackResult(
            success=True,
            level_used=FallbackLevel.LEVEL_1_FAST,
            intelligence_data=regime_data,
            confidence_modifier=confidence_modifier,
            processing_time_ms=0,
            reason="Fast generation with timeout"
        )

    def _try_basic_regime_detection(self, signal: Dict) -> FallbackResult:
        """Level 2: Basic regime detection using signal characteristics"""
        confidence = signal.get('confidence_score', 0.0)
        strategy = signal.get('strategy', 'Unknown').lower()

        # Simple regime inference based on strategy and confidence
        if 'momentum' in strategy or 'macd' in strategy:
            inferred_regime = 'trending' if confidence > 0.7 else 'medium_volatility'
        elif 'bollinger' in strategy or 'mean_reversion' in strategy:
            inferred_regime = 'ranging' if confidence > 0.6 else 'low_volatility'
        elif 'ema' in strategy:
            inferred_regime = 'trending' if confidence > 0.65 else 'medium_volatility'
        else:
            inferred_regime = 'medium_volatility'  # Safe default

        # Apply basic confidence modifier based on inferred regime
        basic_modifiers = {
            'trending': 0.9,
            'ranging': 0.8,
            'medium_volatility': 0.85,
            'low_volatility': 0.8,
            'high_volatility': 0.9
        }

        confidence_modifier = basic_modifiers.get(inferred_regime, 0.8)

        # Store in signal
        signal['market_intelligence_confidence_modifier'] = confidence_modifier

        intelligence_data = {
            'dominant_regime': inferred_regime,
            'confidence': 0.6,  # Basic detection confidence
            'method': 'basic_inference'
        }

        return FallbackResult(
            success=True,
            level_used=FallbackLevel.LEVEL_2_BASIC,
            intelligence_data=intelligence_data,
            confidence_modifier=confidence_modifier,
            processing_time_ms=0,
            reason=f"Basic regime detection: {inferred_regime}"
        )

    def _try_confidence_only_filtering(self, signal: Dict) -> FallbackResult:
        """Level 3: Confidence-only filtering"""
        confidence = signal.get('confidence_score', 0.0)
        strategy = signal.get('strategy', 'Unknown')

        # High-quality signals get bypass
        if confidence >= self.high_quality_confidence_threshold:
            confidence_modifier = 1.0
            reason = f"High-quality signal bypass ({confidence:.1%})"
        elif confidence >= self.medium_quality_confidence_threshold:
            confidence_modifier = 0.9
            reason = f"Medium-quality signal ({confidence:.1%})"
        else:
            confidence_modifier = 0.8
            reason = f"Standard signal ({confidence:.1%})"

        signal['market_intelligence_confidence_modifier'] = confidence_modifier

        intelligence_data = {
            'dominant_regime': 'unknown',
            'confidence': confidence,
            'method': 'confidence_only'
        }

        return FallbackResult(
            success=True,
            level_used=FallbackLevel.LEVEL_3_CONFIDENCE,
            intelligence_data=intelligence_data,
            confidence_modifier=confidence_modifier,
            processing_time_ms=0,
            reason=reason
        )

    def _create_result(self, success: bool, level: FallbackLevel, intelligence_data: Dict,
                       confidence_modifier: float, processing_time_ms: float, reason: str) -> FallbackResult:
        """Create a standardized fallback result"""
        return FallbackResult(
            success=success,
            level_used=level,
            intelligence_data=intelligence_data,
            confidence_modifier=confidence_modifier,
            processing_time_ms=processing_time_ms,
            reason=reason
        )

    def get_fallback_statistics(self) -> Dict:
        """Get fallback usage statistics"""
        total_calls = sum(self.fallback_stats.values())

        stats = {
            'total_calls': total_calls,
            'level_usage': {},
            'level_percentages': {}
        }

        for level, count in self.fallback_stats.items():
            stats['level_usage'][level.value] = count
            if total_calls > 0:
                stats['level_percentages'][level.value] = (count / total_calls) * 100
            else:
                stats['level_percentages'][level.value] = 0

        return stats

    def reset_statistics(self):
        """Reset fallback statistics"""
        for level in self.fallback_stats:
            self.fallback_stats[level] = 0
        self.logger.info("📊 Fallback statistics reset")