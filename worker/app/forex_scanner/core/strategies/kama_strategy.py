# core/strategies/kama_strategy.py - UPDATED WITH KAMA-SPECIFIC CONFIDENCE
"""
KAMA Strategy Implementation - FIXED CONFIDENCE CALCULATION
🔥 FOREX OPTIMIZED: Confidence thresholds calibrated for forex market volatility
🎯 KAMA-SPECIFIC: Uses dedicated KAMA confidence calculator instead of generic validator
🚫 NO MORE 15.0%: Eliminates the Enhanced Signal Validator that was causing low scores
⚡ PERFORMANCE: Intelligent caching and optimizations
🧠 SMART: KAMA-aware confidence calculation with forex market context

FIXED: Replaced Enhanced Signal Validator with KAMA-specific confidence calculator
that understands KAMA's unique characteristics and forex market dynamics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

# Import our new modular helpers
from .helpers.kama_forex_optimizer import KAMAForexOptimizer
from .helpers.kama_validator import KAMAValidator
from .helpers.kama_cache import KAMACache
from .helpers.kama_signal_detector import KAMASignalDetector
from .helpers.kama_data_helper import KAMADataHelper

# 🔥 NEW: Import KAMA-specific confidence calculator
from .helpers.kama_confidence_calculator import KAMAConfidenceCalculator

try:
    import config
except ImportError:
    from forex_scanner import config


class KAMAStrategy(BaseStrategy):
    """
    🔥 FOREX OPTIMIZED & MODULAR: KAMA (Kaufman's Adaptive Moving Average) strategy implementation
    
    FIXED: Now uses KAMA-specific confidence calculation instead of Enhanced Signal Validator
    This eliminates the "15.0%" confidence scores and rejection issues.
    """
    
    def __init__(self, data_fetcher=None):
        super().__init__('kama_strategy')
        
        # Initialize core components
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        
        # 🏗️ MODULAR: Initialize specialized helper modules
        self.forex_optimizer = KAMAForexOptimizer(logger=self.logger)
        self.validator = KAMAValidator(logger=self.logger, forex_optimizer=self.forex_optimizer)
        self.cache = KAMACache(logger=self.logger)
        self.data_helper = KAMADataHelper(logger=self.logger, forex_optimizer=self.forex_optimizer)
        
        # Initialize signal detector with injected dependencies
        self.signal_detector = KAMASignalDetector(
            logger=self.logger,
            forex_optimizer=self.forex_optimizer,
            validator=self.validator
        )
        
        # 🔥 NEW: Initialize KAMA-specific confidence calculator (REPLACES Enhanced Signal Validator)
        self.confidence_calculator = KAMAConfidenceCalculator(logger=self.logger)
        
        # Get configuration from forex optimizer
        self.er_period = self.forex_optimizer.er_period
        self.fast_sc = self.forex_optimizer.fast_sc
        self.slow_sc = self.forex_optimizer.slow_sc
        self.min_efficiency = self.forex_optimizer.min_efficiency
        self.trend_threshold = self.forex_optimizer.trend_threshold
        self.min_bars = self.forex_optimizer.min_bars
        self.base_confidence = self.forex_optimizer.base_confidence
        
        self.logger.info(f"🔧 MODULAR KAMA Strategy initialized with KAMA-SPECIFIC confidence calculator")
        self.logger.info(f"   ER Period: {self.er_period}, Fast SC: {self.fast_sc}, Slow SC: {self.slow_sc}")
        self.logger.info("✅ KAMA-Specific Confidence Calculator enabled (Enhanced Signal Validator DISABLED)")
        self.logger.info("🚀 Performance caching system enabled")

    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '15m'
    ) -> Optional[Dict]:
        """
        🚀 MODULAR: Main signal detection orchestrator - FIXED CONFIDENCE CALCULATION
        
        This method coordinates between all helper modules to detect KAMA signals
        while using the new KAMA-specific confidence calculator.
        """
        self.logger.info(f"🔍 [MODULAR KAMA] Starting detection for {epic}")

        try:
            # Validate inputs using data helper
            if not self.data_helper.validate_input_data(df, epic, self.min_bars):
                return None
            
            # Ensure KAMA indicators are present
            df_enhanced = self.data_helper.ensure_kama_indicators(df)
            
            # Use signal detector to find signals
            signal_data = self.signal_detector.detect_kama_signal(
                df_enhanced, epic, spread_pips, timeframe
            )
            
            if not signal_data:
                return None
            
            # 🔥 FIXED: Create enhanced signal data for KAMA confidence calculation
            enhanced_signal_data = self.data_helper.create_enhanced_signal_data(
                df_enhanced.iloc[-1], signal_data['signal_type']
            )
            
            # 🎯 FIXED: Use KAMA-specific confidence calculator instead of Enhanced Signal Validator
            confidence = self.calculate_kama_confidence(enhanced_signal_data, df_enhanced, epic)
            
            # 🔧 ADJUSTED: Lower minimum threshold since KAMA calculator is more realistic
            min_threshold = getattr(config, 'KAMA_MIN_CONFIDENCE', 0.25)  # Default 25% instead of 30%
            
            if confidence < min_threshold:
                self.logger.warning(f"[KAMA MODULAR REJECT] {epic} {signal_data['signal_type']} - "
                                  f"confidence {confidence:.1%} < {min_threshold:.1%}")
                return None
            
            # Adjust price for spread
            adjusted_price = self.price_adjuster.calculate_execution_price(
                signal_data['current_price'], signal_data['signal_type'], spread_pips
            )
            
            # Build final signal using data helper
            final_signal = self.data_helper.build_complete_signal(
                signal_data, enhanced_signal_data, adjusted_price, confidence, 
                epic, timeframe, spread_pips
            )
            
            # Add confidence breakdown for debugging
            if self.logger.isEnabledFor(logging.DEBUG):
                confidence_breakdown = self.confidence_calculator.get_confidence_breakdown(
                    enhanced_signal_data, df_enhanced, epic
                )
                final_signal['confidence_breakdown'] = confidence_breakdown
            
            # Log performance stats from cache
            self.cache.log_cache_performance()
            
            self.logger.info(f"🎯 [KAMA MODULAR FIXED] {signal_data['signal_type']} signal for {epic} "
                           f"(confidence: {confidence:.2f}, ER: {signal_data.get('efficiency_ratio', 0):.3f})")
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in modular KAMA signal detection: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    def calculate_confidence(self, signal_data: Dict, df: pd.DataFrame = None, epic: str = None) -> float:
        """
        🎯 FIXED: Use KAMA-specific confidence calculation instead of Enhanced Signal Validator
        
        This method now delegates to the KAMA-specific confidence calculator
        that understands KAMA's unique characteristics.
        """
        return self.calculate_kama_confidence(signal_data, df, epic)

    def calculate_kama_confidence(self, signal_data: Dict, df: pd.DataFrame = None, epic: str = None) -> float:
        """
        🔥 KAMA-SPECIFIC: Advanced confidence calculation using KAMA-specific calculator
        
        This replaces the Enhanced Signal Validator with KAMA-optimized confidence calculation
        that understands efficiency ratios, trend alignment, and KAMA-specific market dynamics.
        """
        try:
            # 🎯 PRIMARY: Use KAMA-specific confidence calculator
            kama_confidence = self.confidence_calculator.calculate_kama_confidence(signal_data, df, epic)
            
            # 🌍 ENHANCEMENT: Apply forex optimizer adjustments
            if self.forex_optimizer:
                forex_enhanced_confidence = self.forex_optimizer.apply_forex_confidence_adjustments(
                    kama_confidence, epic or '', signal_data
                )
                self.logger.debug(f"[KAMA CONFIDENCE] Base: {kama_confidence:.1%} → "
                                f"Forex Enhanced: {forex_enhanced_confidence:.1%}")
            else:
                forex_enhanced_confidence = kama_confidence
            
            # 🔧 FINAL: Apply any additional KAMA-specific adjustments
            final_confidence = self._apply_final_kama_adjustments(
                forex_enhanced_confidence, signal_data, df, epic
            )
            
            self.logger.info(f"[KAMA CONFIDENCE FIXED] Final: {final_confidence:.1%} "
                           f"(Base: {kama_confidence:.1%} → Enhanced: {forex_enhanced_confidence:.1%})")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"[KAMA CONFIDENCE ERROR] {e}")
            # 🚨 FALLBACK: Use simple efficiency-based confidence
            return self._fallback_kama_confidence(signal_data)

    def _apply_final_kama_adjustments(
        self, 
        base_confidence: float, 
        signal_data: Dict, 
        df: pd.DataFrame = None, 
        epic: str = None
    ) -> float:
        """
        🔧 FINAL ADJUSTMENTS: Apply any last-minute KAMA-specific adjustments
        """
        try:
            adjusted_confidence = base_confidence
            
            # Ensure we never go below a reasonable minimum for KAMA
            min_kama_confidence = 0.2  # KAMA should have at least 20% confidence
            
            # Market session adjustment for KAMA (optional)
            session_adjustment = self._get_kama_session_adjustment()
            adjusted_confidence += session_adjustment
            
            # Volatility adjustment for KAMA (works better in volatile markets)
            if df is not None and len(df) > 10:
                volatility_adjustment = self._get_kama_volatility_adjustment(df)
                adjusted_confidence += volatility_adjustment
            
            # Final bounds
            return max(min_kama_confidence, min(0.95, adjusted_confidence))
            
        except Exception as e:
            self.logger.debug(f"Final KAMA adjustments error: {e}")
            return base_confidence

    def _get_kama_session_adjustment(self) -> float:
        """
        ⏰ KAMA SESSION: KAMA performs better during high-volume sessions
        """
        try:
            import pytz
            london_tz = pytz.timezone('Europe/London')
            london_time = datetime.now(london_tz)
            hour = london_time.hour
            
            if 8 <= hour < 17:  # London session - good for KAMA
                return 0.02
            elif 13 <= hour < 22:  # New York session - good for KAMA
                return 0.02
            elif 21 <= hour or hour < 2:  # Sydney/Tokyo overlap - moderate for KAMA
                return 0.01
            else:  # Low volume periods - slightly worse for KAMA
                return -0.01
        except:
            return 0.0

    def _get_kama_volatility_adjustment(self, df: pd.DataFrame) -> float:
        """
        📊 KAMA VOLATILITY: KAMA adapts well to volatility - bonus for volatile periods
        """
        try:
            if len(df) < 10:
                return 0.0
            
            recent_data = df.tail(10)
            volatility = recent_data['close'].std() / recent_data['close'].mean()
            
            if volatility > 0.015:  # High volatility (>1.5%) - KAMA excels
                return 0.03
            elif volatility > 0.01:  # Moderate volatility (1-1.5%) - KAMA good
                return 0.02
            elif volatility < 0.005:  # Low volatility (<0.5%) - KAMA struggles
                return -0.02
            else:
                return 0.0  # Normal volatility
                
        except Exception as e:
            self.logger.debug(f"Volatility adjustment error: {e}")
            return 0.0

    def _fallback_kama_confidence(self, signal_data: Dict) -> float:
        """
        🚨 FALLBACK: Simple KAMA confidence when main calculation fails
        """
        try:
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
            signal_strength = signal_data.get('signal_strength', 0.5)
            
            # Simple KAMA-focused calculation
            fallback_confidence = (efficiency_ratio * 0.7 + signal_strength * 0.3)
            
            # Ensure reasonable bounds for KAMA
            return max(0.25, min(0.85, fallback_confidence))
            
        except:
            return 0.4  # Safe fallback

    # ===== EXISTING METHODS (UNCHANGED) =====
    
    def get_required_indicators(self, epic: str = None) -> List[str]:
        """Get list of required indicators for KAMA strategy"""
        return self.data_helper.get_required_indicators()
    
    def get_semantic_indicators(self) -> List[str]:
        """Required indicators with semantic names"""
        return self.data_helper.get_semantic_indicators()

    def validate_modular_integration(self) -> bool:
        """
        🔧 MODULAR: Validate that all helper modules are properly integrated
        """
        try:
            self.logger.info("🔧 Validating modular KAMA integration...")
            
            # Check that all modules are initialized
            required_modules = [
                ('forex_optimizer', self.forex_optimizer),
                ('validator', self.validator),
                ('cache', self.cache),
                ('data_helper', self.data_helper),
                ('signal_detector', self.signal_detector),
                ('confidence_calculator', self.confidence_calculator)  # NEW: Check KAMA confidence calculator
            ]
            
            for module_name, module_instance in required_modules:
                if module_instance is None:
                    self.logger.error(f"❌ Module {module_name} not properly initialized")
                    return False
                    
            # Check that dependencies are properly injected
            if self.signal_detector.forex_optimizer != self.forex_optimizer:
                self.logger.error("❌ SignalDetector forex_optimizer dependency not properly injected")
                return False
                
            if self.signal_detector.validator != self.validator:
                self.logger.error("❌ SignalDetector validator dependency not properly injected")
                return False
                
            if self.data_helper.forex_optimizer != self.forex_optimizer:
                self.logger.error("❌ DataHelper forex_optimizer dependency not properly injected")
                return False
            
            self.logger.info("✅ All KAMA modules properly integrated and dependencies injected")
            self.logger.info("✅ KAMA-specific confidence calculator integrated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ KAMA module integration validation failed: {e}")
            return False

    def get_performance_stats(self) -> Dict:
        """
        📊 MODULAR: Get comprehensive performance statistics from all modules
        """
        try:
            return {
                'strategy': 'kama_modular_fixed',
                'confidence_calculator': 'kama_specific',  # NEW: Indicate fixed confidence
                'cache_stats': self.cache.get_cache_stats(),
                'forex_optimizer_stats': self.forex_optimizer.get_performance_stats(),
                'validator_stats': self.validator.get_validation_stats(),
                'signal_detector_stats': self.signal_detector.get_detection_stats(),
                'data_helper_stats': self.data_helper.get_data_stats(),
                'modular_integration': self.validate_modular_integration(),
                'error': None
            }
        except Exception as e:
            self.logger.error(f"Performance stats error: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """🚀 MODULAR: Clear all cached calculations across modules"""
        self.cache.clear_cache()
        self.forex_optimizer.clear_cache()
        self.logger.info("🧹 KAMA modular cache cleared - all calculations will be recalculated")

    def debug_signal_detection(self, df: pd.DataFrame, epic: str, spread_pips: float = 1.5, timeframe: str = '15m') -> Dict:
        """
        🔍 MODULAR: Comprehensive debugging using all helper modules - FIXED CONFIDENCE
        """
        try:
            debug_info = {
                'strategy': 'kama_modular_fixed',
                'confidence_method': 'kama_specific_calculator',  # NEW: Show fixed confidence method
                'epic': epic,
                'timeframe': timeframe,
                'spread_pips': spread_pips,
                'validation_steps': [],
                'rejection_reasons': [],
                'module_stats': {}
            }
            
            # Get debug info from each module
            debug_info['data_validation'] = self.data_helper.debug_data_validation(df, epic, self.min_bars)
            debug_info['forex_analysis'] = self.forex_optimizer.debug_forex_analysis(df, epic, timeframe)
            debug_info['signal_detection'] = self.signal_detector.debug_signal_detection(df, epic, spread_pips, timeframe)
            debug_info['cache_performance'] = self.cache.get_cache_stats()
            
            # Try to detect signal
            signal = self.detect_signal(df, epic, spread_pips, timeframe)
            debug_info['signal_result'] = signal
            
            if signal:
                debug_info['validation_steps'].append(f"✅ Signal detected: {signal['signal_type']} with {signal['confidence_score']:.1%} confidence (KAMA-specific)")
                
                # NEW: Add confidence breakdown if available
                if 'confidence_breakdown' in signal:
                    debug_info['confidence_breakdown'] = signal['confidence_breakdown']
            else:
                debug_info['rejection_reasons'].append(f"❌ No signal detected after KAMA-specific validations")
            
            return debug_info
            
        except Exception as e:
            debug_info['error'] = str(e)
            debug_info['rejection_reasons'].append(f"❌ Exception: {e}")
            return debug_info


# ===== CONFIGURATION FOR FIXED KAMA =====

def configure_kama_for_more_signals():
    """
    🔧 CONFIGURATION: Apply settings to get more KAMA signals with the fixed confidence calculator
    
    Call this function to optimize KAMA for signal generation.
    """
    
    # Lower minimum confidence threshold
    config.KAMA_MIN_CONFIDENCE = 0.25  # Down from 0.3 (30%)
    
    # Lower KAMA efficiency requirements
    config.KAMA_MIN_EFFICIENCY = 0.05  # Down from 0.1
    
    # Reduce trend threshold
    config.KAMA_TREND_THRESHOLD = 0.02  # Down from 0.05
    
    # Reduce minimum bars requirement
    config.KAMA_MIN_BARS = 20  # Down from 50
    
    # Set base confidence lower
    config.KAMA_BASE_CONFIDENCE = 0.5  # Down from 0.75
    
    print("🔧 KAMA configuration optimized for more signals:")
    print(f"   Min Confidence: {config.KAMA_MIN_CONFIDENCE:.1%}")
    print(f"   Min Efficiency: {config.KAMA_MIN_EFFICIENCY}")
    print(f"   Trend Threshold: {config.KAMA_TREND_THRESHOLD}")
    print(f"   Min Bars: {config.KAMA_MIN_BARS}")
    print(f"   Base Confidence: {config.KAMA_BASE_CONFIDENCE:.1%}")


# ===== FACTORY FUNCTION =====

def create_kama_strategy(data_fetcher=None):
    """
    🏗️ FACTORY: Create KAMA strategy with fixed confidence calculation
    """
    strategy = KAMAStrategy(data_fetcher=data_fetcher)
    
    # Validate integration
    if not strategy.validate_modular_integration():
        logging.getLogger(__name__).warning("⚠️ Modular integration validation failed")
    
    return strategy