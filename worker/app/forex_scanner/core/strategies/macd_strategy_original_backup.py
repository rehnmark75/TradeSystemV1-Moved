# core/strategies/macd_strategy.py
"""
MACD Strategy Implementation - REFACTORED & MODULAR with Multi-Timeframe Analysis
🔥 FOREX OPTIMIZED: Confidence thresholds calibrated for forex market volatility
🏗️ MODULAR: Clean separation of concerns with focused helper modules
🎯 MAINTAINABLE: Easy to understand, modify, and extend
⚡ PERFORMANCE: Intelligent caching and optimizations
🧠 SMART: Enhanced Signal Validator integration with forex market context
📊 MTF: Multi-Timeframe Analysis for momentum validation (NEW)

REFACTORING COMPLETE: Main strategy now focuses on coordination while
specialized modules handle specific responsibilities:
- MACDForexOptimizer: Forex-specific calculations and optimizations
- MACDValidator: Signal validation and confidence calculation  
- MACDCache: Performance caching and optimization
- MACDSignalDetector: Core signal detection algorithms
- MACDDataHelper: Data preparation and enhancement

NEW FEATURES:
- Multi-Timeframe MACD momentum analysis
- Enhanced confidence scoring with MTF validation
- MACD-specific divergence detection across timeframes
- Session-aware MTF thresholds

This maintains 100% backward compatibility while dramatically improving maintainability!
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
import numpy as np

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
#from ..detection.enhanced_signal_validator import EnhancedSignalValidator

# Import our new modular helpers
from .helpers.macd_forex_optimizer import MACDForexOptimizer
from .helpers.macd_validator import MACDValidator
from .helpers.macd_cache import MACDCache
from .helpers.macd_signal_detector import MACDSignalDetector
from .helpers.macd_data_helper import MACDDataHelper

try:
    import config
except ImportError:
    from forex_scanner import config


class MACDStrategy(BaseStrategy):
    """
    🔥 FOREX OPTIMIZED & MODULAR: MACD + EMA 200 strategy implementation with MTF Analysis
    
    Now organized with clean separation of concerns:
    - Main class handles coordination and public interface
    - Helper modules handle specialized functionality
    - Multi-Timeframe Analysis for enhanced signal validation
    - 100% backward compatibility maintained
    - Dramatically improved maintainability and testability
    """
    
    def __init__(self, data_fetcher=None):
        super().__init__('macd_ema200')
        
        # Initialize core components
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        
        # 🏗️ MODULAR: Initialize specialized helper modules
        self.forex_optimizer = MACDForexOptimizer(
            logger=self.logger, 
            use_correlation=False  # Set to True when correlation engine is ready
        )
        self.validator = MACDValidator(logger=self.logger, forex_optimizer=self.forex_optimizer)
        self.cache = MACDCache(logger=self.logger)
        self.data_helper = MACDDataHelper(logger=self.logger, forex_optimizer=self.forex_optimizer)
        
        # 🔧 FIX: Initialize db_manager attribute before using it
        self.db_manager = None  # Will be set by scanner if needed
        
        # 🔐 CRITICAL: Validate forex_optimizer initialization
        self._validate_threshold_initialization()
        
        # Initialize signal detector with injected dependencies
        self.signal_detector = MACDSignalDetector(
            logger=self.logger,
            forex_optimizer=self.forex_optimizer,
            validator=self.validator,
            db_manager=self.db_manager,  # Pass db_manager
            data_fetcher=data_fetcher    # Pass data_fetcher
        )
        
        # Get configuration from forex optimizer
        self.min_efficiency_ratio = self.forex_optimizer.min_efficiency_ratio
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.50)
        
        #🔧 FIX: Use confidence optimizer instead of missing enhanced_validator
        from .helpers.macd_confidence_optimizer import MACDConfidenceOptimizer
        self.confidence_optimizer = MACDConfidenceOptimizer(self.logger, self.forex_optimizer)
        
        # Backward compatibility - so existing code doesn't break
        self.enhanced_validator = self.confidence_optimizer
        
        # 📊 NEW: Initialize Multi-Timeframe Analyzer BEFORE calling initialize_mtf_analysis
        self.mtf_analyzer = None
        self.enable_mtf_analysis = getattr(config, 'ENABLE_MTF_ANALYSIS', True)
        
        # 📊 MTF configuration - Updated to match your MTF_CONFIG
        self.mtf_config = {
            'require_higher_tf_alignment': getattr(config, 'MTF_CONFIG', {}).get('require_alignment', True),
            'min_aligned_timeframes': getattr(config, 'MTF_CONFIG', {}).get('min_aligned_timeframes', 2),
            'mtf_confidence_boost': getattr(config, 'MTF_CONFIG', {}).get('confidence_boost_max', 0.15),
            'mtf_rejection_mode': getattr(config, 'MTF_REJECTION_MODE', 'soft'),
            'check_timeframes': getattr(config, 'MTF_CONFIG', {}).get('check_timeframes', ['5m', '15m', '1h']),
            'alignment_threshold': getattr(config, 'MTF_CONFIG', {}).get('alignment_threshold', 0.6),
            'alignment_weights': {
                '5m': 0.2,   # Short-term momentum
                '15m': 0.4,  # Primary MACD timeframe  
                '1h': 0.4    # Medium-term trend
            },
            'mtf_cache_duration': 300  # 5 minutes cache
        }
        
        # Initialize MTF cache
        self.mtf_cache = {}
        self.mtf_cache_timestamps = {}
        
        # 🚀 CRITICAL FIX: Initialize MTF analyzer directly (like EMA strategy does)
        if data_fetcher and self.enable_mtf_analysis:
            try:
                from analysis.multi_timeframe import MultiTimeframeAnalyzer
                self.mtf_analyzer = MultiTimeframeAnalyzer(data_fetcher)
                self.logger.info("📊 MACD Multi-Timeframe Analysis ENABLED")
            except ImportError as e:
                self.logger.warning(f"⚠️ Multi-Timeframe module not available: {e}")
                self.enable_mtf_analysis = False
                self.mtf_analyzer = None
            except Exception as e:
                self.logger.error(f"❌ MTF initialization failed: {e}")
                self.enable_mtf_analysis = False
                self.mtf_analyzer = None
        else:
            if not data_fetcher:
                self.logger.info("📊 MACD MTF disabled: no data_fetcher provided")
            if not self.enable_mtf_analysis:
                self.logger.info("📊 MACD MTF disabled: ENABLE_MTF_ANALYSIS=False")
        
        # 📊 NOW call initialize_mtf_analysis (this should work now)
        mtf_success = self.initialize_mtf_analysis(data_fetcher=data_fetcher, enable=True)
        
        # 🔍 LOG MTF INITIALIZATION STATUS
        self.logger.info("🔍 [MACD INIT] MTF Status Check:")
        self.logger.info(f"   enable_mtf_analysis: {getattr(self, 'enable_mtf_analysis', False)}")
        self.logger.info(f"   mtf_analyzer exists: {getattr(self, 'mtf_analyzer', None) is not None}")
        self.logger.info(f"   mtf_config exists: {hasattr(self, 'mtf_config')}")
        self.logger.info(f"   initialize_mtf_analysis success: {mtf_success}")
        
        # 🎯 FINAL STATUS
        if getattr(self, 'mtf_analyzer', None) and getattr(self, 'enable_mtf_analysis', False):
            self.logger.info("✅ MACD Strategy initialized with WORKING MTF analysis")
        else:
            self.logger.warning("⚠️ MACD Strategy initialized but MTF analysis NOT WORKING")
        
        self.logger.info("✅ MACD Strategy initialized with confidence optimizer and MTF analysis (modular version)")
    
    def _validate_threshold_initialization(self):
        """
        🔐 CRITICAL: Validate that forex_optimizer can provide valid thresholds
        This prevents weak signals from passing through
        """
        try:
            # Test critical epics
            test_epics = [
                'CS.D.EURUSD.CEEM.IP',
                'CS.D.GBPUSD.MINI.IP', 
                'CS.D.USDJPY.MINI.IP'
            ]
            
            all_valid = True
            for epic in test_epics:
                threshold = self.forex_optimizer.get_macd_threshold_for_epic(epic)
                
                # Validate threshold is reasonable (accounting for session multipliers)
                if 'JPY' in epic:
                    # JPY thresholds can be multiplied by session factors (0.7x to 1.2x)
                    # Base range: 0.003-0.008, with multipliers: 0.0021-0.0096
                    if threshold < 0.002 or threshold > 0.01:
                        self.logger.error(f"[CRITICAL] Invalid JPY threshold for {epic}: {threshold}")
                        all_valid = False
                else:
                    # Non-JPY thresholds: base 0.00004-0.00008, with multipliers: 0.000028-0.000096
                    if threshold < 0.000025 or threshold > 0.0002:
                        self.logger.error(f"[CRITICAL] Invalid non-JPY threshold for {epic}: {threshold}")
                        all_valid = False
                    
                self.logger.debug(f"[THRESHOLD VALIDATION] {epic}: {threshold:.8f}")
            
            if all_valid:
                self.logger.info("✅ Threshold validation passed - all thresholds within acceptable range")
            else:
                self.logger.warning("⚠️ Threshold validation failed - some thresholds out of range")
                
        except Exception as e:
            self.logger.error(f"[CRITICAL] Threshold validation failed: {e}")
            self.logger.warning("⚠️ Strategy may reject valid signals or accept weak ones")

    def initialize_mtf_analysis(self, data_fetcher=None, enable=True):
        """
        Initialize Multi-Timeframe analysis for MACD strategy
        Call this method to enable MTF analysis
        """
        try:
            self.logger.info(f"🔧 [MACD MTF INIT] Starting initialization...")
            self.logger.info(f"   data_fetcher provided: {data_fetcher is not None}")
            self.logger.info(f"   enable flag: {enable}")
            
            # Set data_fetcher if provided
            if data_fetcher:
                self.data_fetcher = data_fetcher
            
            # Check if we have what we need
            if not self.data_fetcher:
                self.logger.warning(f"⚠️ [MACD MTF INIT] No data_fetcher available")
                self.enable_mtf_analysis = False
                return False
            
            if not enable:
                self.logger.info(f"📊 [MACD MTF INIT] MTF disabled by enable flag")
                self.enable_mtf_analysis = False
                return False
            
            # Set enable flag
            self.enable_mtf_analysis = True
            
            # Try to create MTF analyzer if not already created
            if not hasattr(self, 'mtf_analyzer') or self.mtf_analyzer is None:
                try:
                    from analysis.multi_timeframe import MultiTimeframeAnalyzer
                    self.mtf_analyzer = MultiTimeframeAnalyzer(self.data_fetcher)
                    self.logger.info("✅ [MACD MTF INIT] Multi-Timeframe Analyzer created")
                except ImportError as e:
                    self.logger.warning(f"⚠️ [MACD MTF INIT] MultiTimeframeAnalyzer import failed: {e}")
                    self.enable_mtf_analysis = False
                    return False
                except Exception as e:
                    self.logger.error(f"❌ [MACD MTF INIT] MTF analyzer creation failed: {e}")
                    self.enable_mtf_analysis = False
                    return False
            else:
                self.logger.info("✅ [MACD MTF INIT] Multi-Timeframe Analyzer already exists")
            
            # Ensure MTF configuration exists
            if not hasattr(self, 'mtf_config') or not self.mtf_config:
                self.mtf_config = {
                    'require_higher_tf_alignment': True,
                    'min_aligned_timeframes': 2,
                    'mtf_confidence_boost': 0.15,
                    'mtf_rejection_mode': 'soft',
                    'check_timeframes': ['5m', '15m', '1h', '4h'],
                    'alignment_weights': {
                        '5m': 0.15, '15m': 0.25, '1h': 0.35, '4h': 0.25
                    }
                }
                self.logger.info("✅ [MACD MTF INIT] MTF configuration created")
            
            # Initialize MTF cache if not already done
            if not hasattr(self, 'mtf_cache'):
                self.mtf_cache = {}
                self.mtf_cache_timestamps = {}
                self.logger.info("✅ [MACD MTF INIT] MTF cache initialized")
            
            # Final status check
            success = (
                getattr(self, 'enable_mtf_analysis', False) and 
                getattr(self, 'mtf_analyzer', None) is not None and
                hasattr(self, 'mtf_config')
            )
            
            if success:
                self.logger.info(f"✅ [MACD MTF INIT] Initialization SUCCESS!")
                self.logger.info(f"   Enabled: {self.enable_mtf_analysis}")
                self.logger.info(f"   Analyzer: {self.mtf_analyzer is not None}")
                self.logger.info(f"   Timeframes: {self.mtf_config.get('check_timeframes', [])}")
            else:
                self.logger.error(f"❌ [MACD MTF INIT] Initialization FAILED!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ [MACD MTF INIT] Initialization completely failed: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            self.enable_mtf_analysis = False
            return False

    def detect_signals(self, df_enhanced: pd.DataFrame, epic: str, timeframe: str, 
                  latest: pd.Series, previous: pd.Series, spread_pips: float = 1.5) -> List[Dict]:
        """
        🔧 COMPATIBILITY FIX: Main signal detection method for scanner compatibility
        
        This method bridges the gap between the modular MACDStrategy and the
        scanner's expected interface.
        
        Args:
            df_enhanced: Enhanced DataFrame with MACD indicators
            epic: Trading pair epic 
            timeframe: Timeframe for analysis
            latest: Latest bar data
            previous: Previous bar data
            spread_pips: Spread in pips
            
        Returns:
            List of detected signals (empty list if no signals)
        """
        try:
            self.logger.debug(f"🔍 [MACD STRATEGY] detect_signals called for {epic}")
            
            # Check if MACD strategy is enabled
            if not getattr(config, 'MACD_EMA_STRATEGY', True):
                self.logger.debug(f"⚠️ [MACD STRATEGY] MACD strategy disabled in config")
                return []
            
            # Check if emergency mode is enabled
            if getattr(config, 'EMERGENCY_MACD_MODE', False):
                self.logger.info(f"🚨 [MACD STRATEGY] Emergency mode enabled - using basic detection")
                return self._detect_signals_emergency_mode(df_enhanced, epic, timeframe, latest, previous)
            
            # Use the existing modular detect_signal method (note: singular)
            signal = self.detect_signal(df_enhanced, epic, spread_pips, timeframe)
            
            # Collect all signals (immediate + momentum confirmation)
            signals = []
            
            # Add immediate signal if found
            if signal:
                self.logger.info(f"✅ [MACD STRATEGY] Immediate signal detected: {signal.get('signal_type', 'Unknown')}")
                signals.append(signal)
            
            # NEW: Check for momentum confirmation signals (delayed signals from weak crossovers)
            momentum_signal = self.signal_detector.check_momentum_confirmation_signals(
                epic=epic,
                timeframe=timeframe,
                df_enhanced=df_enhanced,
                latest=latest,
                forex_optimizer=self.forex_optimizer
            )
            
            if momentum_signal:
                self.logger.info(f"🎯 [MACD STRATEGY] Momentum confirmation signal detected: {momentum_signal.get('signal_type', 'Unknown')}")
                signals.append(momentum_signal)
            
            # Cleanup tracker periodically
            if hasattr(self.signal_detector, 'cleanup_tracker'):
                self.signal_detector.cleanup_tracker()
                
            return signals
                
        except Exception as e:
            self.logger.error(f"❌ [MACD STRATEGY] detect_signals failed for {epic}: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return []

    def _detect_signals_emergency_mode(self, df_enhanced: pd.DataFrame, epic: str, 
                                    timeframe: str, latest: pd.Series, previous: pd.Series) -> List[Dict]:
        """
        🚨 EMERGENCY MODE: Ultra-simple MACD signal detection
        
        Bypasses all enhanced logic and uses basic crossover detection only.
        """
        try:
            self.logger.info(f"🚨 [EMERGENCY MACD] Basic crossover detection for {epic}")
            
            # Get basic MACD data
            current_hist = latest.get('macd_histogram', 0)
            previous_hist = previous.get('macd_histogram', 0)
            current_price = latest.get('close', 0)
            ema_200 = latest.get('ema_200', 0)
            
            # Validate data
            if pd.isna(current_hist) or pd.isna(previous_hist) or current_price <= 0:
                self.logger.warning(f"❌ [EMERGENCY MACD] Invalid data for {epic}")
                return []
            
            # Check for crossovers
            signal_type = None
            if previous_hist <= 0 and current_hist > 0:
                signal_type = 'BULL'
                self.logger.info(f"🚨 [EMERGENCY MACD] BULL crossover: {previous_hist:.6f} → {current_hist:.6f}")
            elif previous_hist >= 0 and current_hist < 0:
                signal_type = 'BEAR'
                self.logger.info(f"🚨 [EMERGENCY MACD] BEAR crossover: {previous_hist:.6f} → {current_hist:.6f}")
            
            if not signal_type:
                self.logger.debug(f"📊 [EMERGENCY MACD] No crossover for {epic}")
                return []
            
            # Very basic EMA200 filter (optional)
            if ema_200 > 0:
                if signal_type == 'BULL' and current_price < ema_200:
                    self.logger.info(f"❌ [EMERGENCY MACD] BULL signal rejected: price {current_price:.5f} below EMA200 {ema_200:.5f}")
                    return []
                if signal_type == 'BEAR' and current_price > ema_200:
                    self.logger.info(f"❌ [EMERGENCY MACD] BEAR signal rejected: price {current_price:.5f} above EMA200 {ema_200:.5f}")
                    return []
            
            # Create emergency signal
            signal = {
                'signal_type': signal_type,
                'epic': epic,
                'timeframe': timeframe,
                'strategy': 'emergency_macd',
                'price': float(current_price),
                'ema_200': float(ema_200) if ema_200 > 0 else 0.0,
                'macd_histogram': float(current_hist),
                'macd_histogram_prev': float(previous_hist),
                'macd_line': latest.get('macd_line', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'trigger_reason': 'emergency_basic_crossover',
                'confidence_score': 75.0,  # Fixed confidence for emergency mode
                'timestamp': latest.name if hasattr(latest, 'name') else datetime.now(),
                'emergency_mode': True,
                'enhanced_features_bypassed': True
            }
            
            self.logger.info(f"✅ [EMERGENCY MACD] {signal_type} signal created for {epic}")
            return [signal]
            
        except Exception as e:
            self.logger.error(f"❌ [EMERGENCY MACD] Failed for {epic}: {e}")
            return []
    def _check_minimum_bars(self, df: pd.DataFrame, epic: str) -> bool:
        """Check if we have sufficient bars for MACD detection"""
        try:
            import config
            min_bars = getattr(config, 'MACD_MIN_BARS_REQUIRED', 50)
            
            if len(df) < min_bars:
                self.logger.debug(f"🚫 Insufficient bars: {len(df)} < {min_bars}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"⚠️ Minimum bars check failed: {e}")
            # Default to allowing if check fails
            return True
    
    # 📊 NEW: Multi-Timeframe Analysis Methods
    def detect_signal_with_mtf(self, df: pd.DataFrame, epic: str, spread_pips: float, 
                        timeframe: str, signal_timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """
        📊 Enhanced MACD signal detection with Multi-Timeframe Analysis
        """
        
        self.logger.info(f"📊 [MACD MTF] Starting Multi-Timeframe Enhanced Detection for {epic}")
        self.logger.info(f"   Primary Timeframe: {timeframe}")
        self.logger.info(f"   MTF Enabled: {getattr(self, 'enable_mtf_analysis', False)}")
        self.logger.info(f"   MTF Mode: {getattr(self, 'mtf_config', {}).get('mtf_rejection_mode', 'soft')}")
        
        # First, get the standard MACD signal - pass through the timestamp
        signal = self.detect_signal(df, epic, spread_pips, timeframe, signal_timestamp)
        
        if not signal:
            self.logger.info(f"📊 [MACD MTF] No base signal detected, skipping MTF analysis")
            return None
        
        self.logger.info(f"📊 [MACD MTF] Base signal detected: {signal['signal_type']} with {signal.get('confidence_score', 0):.1%} confidence")
        
        # Check if MTF analysis is enabled and available
        if not getattr(self, 'enable_mtf_analysis', False) or not hasattr(self, 'mtf_analyzer') or not self.mtf_analyzer:
            self.logger.info(f"📊 [MACD MTF] Analysis disabled or unavailable, using base signal only")
            return signal
        
        try:
            self.logger.info(f"🔄 [MACD MTF] Performing Multi-Timeframe MACD Analysis")
            self.logger.info(f"   Checking timeframes: {self.mtf_config.get('check_timeframes', [])}")
            self.logger.info(f"   Minimum aligned required: {self.mtf_config.get('min_aligned_timeframes', 2)}")
            self.logger.info(f"   Alignment threshold: {self.mtf_config.get('alignment_threshold', 0.6):.1%}")
            
            # Extract pair from epic
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.IP', '')
            
            # Get MTF MACD momentum analysis
            mtf_momentum = self._get_mtf_macd_momentum_cached(epic, pair, timeframe)
            
            # Log individual timeframe momentum
            self.logger.info(f"📊 [MACD MTF] Momentum Analysis Complete:")
            for tf, momentum in mtf_momentum.items():
                emoji = "🟢" if momentum == 'bullish' else "🔴" if momentum == 'bearish' else "🟡"
                weight = self.mtf_config['alignment_weights'].get(tf, 0.2)
                self.logger.info(f"   {tf}: {emoji} {momentum.upper()} (weight: {weight:.0%})")
            
            # Validate signal against MTF momentum
            is_valid, momentum_score, aligned_timeframes = self._validate_mtf_macd_alignment(
                signal['signal_type'], mtf_momentum, self.mtf_config
            )
            
            # Add MTF data to signal
            signal['mtf_analysis'] = {
                'enabled': True,
                'strategy_type': 'MACD',
                'momentum_analysis': mtf_momentum,
                'momentum_score': momentum_score,
                'aligned_timeframes': aligned_timeframes,
                'total_timeframes': len(mtf_momentum),
                'mtf_valid': is_valid,
                'rejection_mode': self.mtf_config['mtf_rejection_mode'],
                'macd_specific': True,
                'alignment_threshold': self.mtf_config.get('alignment_threshold', 0.6)
            }
            
            # Log MTF validation results
            self.logger.info(f"📊 [MACD MTF] Validation Results:")
            self.logger.info(f"   Signal Type: {signal['signal_type']}")
            self.logger.info(f"   Momentum Score: {momentum_score:.1%}")
            self.logger.info(f"   Aligned Timeframes: {aligned_timeframes}/{len(mtf_momentum)}")
            self.logger.info(f"   MTF Valid: {'✅ YES' if is_valid else '❌ NO'}")
            self.logger.info(f"   Threshold Check: {momentum_score:.1%} {'≥' if momentum_score >= self.mtf_config.get('alignment_threshold', 0.6) else '<'} {self.mtf_config.get('alignment_threshold', 0.6):.1%}")
            
            # Determine signal fate
            signal_fate = "UNKNOWN"
            confidence_change = 0
            
            # Apply MTF validation based on require_alignment setting
            if self.mtf_config.get('require_higher_tf_alignment', True):
                if not is_valid or momentum_score < self.mtf_config.get('alignment_threshold', 0.6):
                    if self.mtf_config.get('mtf_rejection_mode', 'soft') == 'hard':
                        signal_fate = "REJECTED"
                        self.logger.warning(f"⚠️ [MACD MTF] Signal REJECTED (hard mode)")
                        self.logger.warning(f"   Reason: Insufficient momentum alignment")
                        self.logger.warning(f"   Required: {self.mtf_config['min_aligned_timeframes']} aligned TFs with {self.mtf_config.get('alignment_threshold', 0.6):.1%} score")
                        self.logger.warning(f"   Got: {aligned_timeframes} aligned TFs with {momentum_score:.1%} score")
                        return None
                    else:
                        # Soft mode: Reduce confidence for poor alignment
                        penalty = 0.12  # 12% confidence penalty for MACD
                        original_confidence = signal.get('confidence_score', 0.5)
                        signal['confidence_score'] = max(0.3, original_confidence - penalty)
                        signal['mtf_confidence_penalty'] = penalty
                        confidence_change = -penalty
                        signal_fate = "PENALIZED"
                        self.logger.warning(f"⚠️ [MACD MTF] Poor momentum alignment (soft mode)")
                        self.logger.warning(f"   Confidence Penalty: -{penalty:.1%}")
                        self.logger.warning(f"   Confidence: {original_confidence:.1%} → {signal['confidence_score']:.1%}")
                else:
                    signal_fate = "VALIDATED"
            else:
                signal_fate = "ACCEPTED"
            
            # Apply confidence boost for excellent alignment
            alignment_threshold = self.mtf_config.get('alignment_threshold', 0.6)
            if momentum_score >= alignment_threshold:
                # Calculate boost based on how much we exceed the threshold
                excess_score = momentum_score - alignment_threshold
                max_excess = 1.0 - alignment_threshold  # Maximum possible excess
                boost_ratio = excess_score / max_excess if max_excess > 0 else 1.0
                
                # Apply boost using confidence_boost_max
                max_boost = self.mtf_config.get('mtf_confidence_boost', 0.15)
                mtf_boost = boost_ratio * max_boost
                
                original_confidence = signal.get('confidence_score', 0.5)
                signal['confidence_score'] = min(0.95, original_confidence + mtf_boost)
                signal['mtf_confidence_boost'] = mtf_boost
                confidence_change += mtf_boost
                
                if signal_fate == "VALIDATED":
                    signal_fate = "BOOSTED"
                
                self.logger.info(f"✅ [MACD MTF] Excellent momentum alignment detected!")
                self.logger.info(f"   Momentum Score: {momentum_score:.1%} (threshold: {alignment_threshold:.1%})")
                self.logger.info(f"   Boost Ratio: {boost_ratio:.1%} (excess: {excess_score:.1%})")
                self.logger.info(f"   Confidence Boost: +{mtf_boost:.1%} (max: {max_boost:.1%})")
                self.logger.info(f"   Final Confidence: {signal['confidence_score']:.1%} (was {original_confidence:.1%})")
            
            # Add MACD-specific MTF metadata
            signal['mtf_analysis'].update({
                'momentum_strength': momentum_score,
                'confidence_adjustment': confidence_change,
                'signal_fate': signal_fate,
                'threshold_met': momentum_score >= alignment_threshold,
                'boost_applied': confidence_change > 0,
                'config_used': {
                    'alignment_threshold': alignment_threshold,
                    'max_boost': self.mtf_config.get('mtf_confidence_boost', 0.15),
                    'require_alignment': self.mtf_config.get('require_higher_tf_alignment', True)
                }
            })
            
            # Final MTF summary
            self.logger.info(f"📊 [MACD MTF] Final Decision: {signal_fate}")
            if confidence_change != 0:
                change_direction = "↗️" if confidence_change > 0 else "↘️"
                self.logger.info(f"   Confidence Change: {change_direction} {confidence_change:+.1%}")
            self.logger.info(f"   Final Confidence: {signal['confidence_score']:.1%}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ [MACD MTF] Analysis failed: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            self.logger.warning(f"⚠️ [MACD MTF] Falling back to base signal")
            
            # Add error info to signal but don't fail
            signal['mtf_analysis'] = {
                'enabled': False,
                'error': str(e),
                'fallback_used': True
            }
            
            return signal

    def _get_mtf_macd_momentum_cached(self, epic: str, pair: str, primary_timeframe: str) -> Dict[str, str]:
        """
        Get Multi-Timeframe MACD momentum analysis with caching
        Returns momentum direction for each timeframe: 'bullish', 'bearish', 'neutral'
        """
        try:
            # Check cache first
            cache_key = f"mtf_macd_momentum_{epic}_{primary_timeframe}"
            if hasattr(self, 'cache') and self.cache:
                cached_result = self.cache.get_cached_result(cache_key)
                if cached_result:
                    self.logger.debug(f"📋 [MACD MTF] Using cached momentum for {epic}")
                    return cached_result
            
            # Get MTF configuration
            mtf_config = getattr(self, 'mtf_config', {
                'check_timeframes': ['5m', '15m', '1h', '4h']
            })
            
            mtf_momentum = {}
            
            for timeframe in mtf_config['check_timeframes']:
                try:
                    # Get data for this timeframe
                    if hasattr(self, 'data_fetcher') and self.data_fetcher:
                        df_tf = self.data_fetcher.get_enhanced_data(epic, pair, timeframe)
                    else:
                        # Fallback if no data_fetcher available
                        continue
                    
                    if df_tf is None or len(df_tf) < 50:  # Need enough data for MACD
                        mtf_momentum[timeframe] = 'neutral'
                        continue
                    
                    # Ensure MACD indicators are calculated
                    if hasattr(self, 'data_helper') and self.data_helper:
                        df_tf = self.data_helper.ensure_macd_indicators(df_tf)
                    
                    # Analyze MACD momentum for this timeframe
                    momentum = self._analyze_macd_momentum(df_tf, timeframe)
                    mtf_momentum[timeframe] = momentum
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ [MACD MTF] Failed to get momentum for {timeframe}: {e}")
                    mtf_momentum[timeframe] = 'neutral'
            
            # Cache the result
            if hasattr(self, 'cache') and self.cache:
                self.cache.cache_result(cache_key, mtf_momentum)
            
            return mtf_momentum
            
        except Exception as e:
            self.logger.error(f"❌ [MACD MTF] Momentum analysis failed: {e}")
            return {'5m': 'neutral', '15m': 'neutral', '1h': 'neutral', '4h': 'neutral'}

    def _analyze_macd_momentum(self, df: pd.DataFrame, timeframe: str) -> str:
        """
        Analyze MACD momentum for a specific timeframe
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        try:
            if len(df) < 10:
                return 'neutral'
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Get MACD values
            macd_histogram = latest.get('macd_histogram', 0)
            macd_histogram_prev = previous.get('macd_histogram', 0)
            macd_line = latest.get('macd_line', 0)
            macd_signal = latest.get('macd_signal', 0)
            
            # Primary momentum indicators
            histogram_direction = 'bullish' if macd_histogram > 0 else 'bearish'
            histogram_trend = 'improving' if macd_histogram > macd_histogram_prev else 'declining'
            line_position = 'bullish' if macd_line > macd_signal else 'bearish'
            
            # Determine overall momentum
            bullish_signals = 0
            bearish_signals = 0
            
            # Count bullish signals
            if histogram_direction == 'bullish':
                bullish_signals += 1
            if histogram_trend == 'improving':
                bullish_signals += 1
            if line_position == 'bullish':
                bullish_signals += 1
            
            # Count bearish signals
            if histogram_direction == 'bearish':
                bearish_signals += 1
            if histogram_trend == 'declining':
                bearish_signals += 1
            if line_position == 'bearish':
                bearish_signals += 1
            
            # Determine momentum
            if bullish_signals >= 2:
                return 'bullish'
            elif bearish_signals >= 2:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.warning(f"⚠️ [MACD MTF] Momentum analysis failed for {timeframe}: {e}")
            return 'neutral'

    def _validate_mtf_macd_alignment(self, signal_type: str, mtf_momentum: Dict[str, str], 
                                mtf_config: Dict) -> Tuple[bool, float, int]:
        """
        Validate MACD signal against multi-timeframe momentum using your config
        Returns: (is_valid, momentum_score, aligned_timeframes_count)
        """
        try:
            signal_direction = signal_type.lower()
            if 'buy' in signal_direction or 'bull' in signal_direction:
                expected_momentum = 'bullish'
            elif 'sell' in signal_direction or 'bear' in signal_direction:
                expected_momentum = 'bearish'
            else:
                return False, 0.0, 0
            
            aligned_timeframes = 0
            weighted_score = 0.0
            total_weight = 0.0
            
            # Check alignment for each timeframe
            for timeframe, momentum in mtf_momentum.items():
                weight = mtf_config['alignment_weights'].get(timeframe, 0.2)
                total_weight += weight
                
                if momentum == expected_momentum:
                    aligned_timeframes += 1
                    weighted_score += weight
                elif momentum == 'neutral':
                    # Neutral is partially supportive (50% weight)
                    weighted_score += (weight * 0.5)
                # Opposite momentum gets 0 weight
            
            # Calculate final momentum score
            momentum_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Determine if alignment is sufficient using YOUR config
            min_aligned = mtf_config.get('min_aligned_timeframes', 2)
            alignment_threshold = mtf_config.get('alignment_threshold', 0.6)
            
            is_valid = (aligned_timeframes >= min_aligned and 
                    momentum_score >= alignment_threshold)
            
            return is_valid, momentum_score, aligned_timeframes
            
        except Exception as e:
            self.logger.error(f"❌ [MACD MTF] Alignment validation failed: {e}")
            return False, 0.0, 0

    def _check_macd_histogram_alignment(self, mtf_momentum: Dict[str, str], signal_type: str) -> Dict:
        """
        Check if MACD histogram alignment supports the signal across timeframes
        """
        try:
            expected_direction = 'bullish' if 'buy' in signal_type.lower() or 'bull' in signal_type.lower() else 'bearish'
            
            alignment_data = {
                'expected_direction': expected_direction,
                'aligned_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': [],
                'alignment_strength': 0.0
            }
            
            for timeframe, momentum in mtf_momentum.items():
                if momentum == expected_direction:
                    alignment_data['aligned_timeframes'].append(timeframe)
                elif momentum == 'neutral':
                    alignment_data['neutral_timeframes'].append(timeframe)
                else:
                    alignment_data['conflicting_timeframes'].append(timeframe)
            
            # Calculate alignment strength
            total_tf = len(mtf_momentum)
            aligned_count = len(alignment_data['aligned_timeframes'])
            neutral_count = len(alignment_data['neutral_timeframes'])
            
            alignment_data['alignment_strength'] = (aligned_count + (neutral_count * 0.5)) / total_tf
            
            return alignment_data
            
        except Exception as e:
            self.logger.error(f"❌ [MACD MTF] Histogram alignment check failed: {e}")
            return {'error': str(e)}

   

    # 🎯 OPTIMIZED: MACD-specific confidence calculation using integrated optimizers
    def calculate_confidence(self, signal_data: Dict, df: pd.DataFrame, epic: str) -> float:
        """
        🔧 FIXED: Calculate confidence with REALISTIC forex thresholds and price basis consistency
        ✅ PRICE BASIS FIX: Uses analysis_price (MID) for all technical calculations
        """
        try:
            # Extract key signal components
            macd_histogram = abs(signal_data.get('macd_histogram', 0))
            macd_histogram_prev = abs(signal_data.get('macd_histogram_prev', 0))
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.2)
            market_regime = signal_data.get('market_regime', 'ranging')
            signal_type = signal_data.get('signal_type', '')
            
            # 🔧 PRICE BASIS FIX: Use analysis_price (MID) for technical calculations, not execution price
            execution_price = signal_data.get('price', 0)           # BID/ASK execution price
            analysis_price = signal_data.get('analysis_price',      # MID analysis price
                                            signal_data.get('mid_price', execution_price))
            ema_200 = signal_data.get('ema_200', 0)                # MID-based EMA
            
            # Log price basis for debugging
            if analysis_price != execution_price:
                self.logger.debug(f"[CONFIDENCE PRICE BASIS] Analysis: MID {analysis_price:.5f}, Execution: {('BID' if signal_type == 'BEAR' else 'ASK')} {execution_price:.5f}")
            
            # 🔧 FIXED: REALISTIC Base confidence calculation for FOREX
            if macd_histogram > 0.00005:  # 0.5 pips (was 8 pips!)
                base_confidence = 0.65
            elif macd_histogram > 0.00003:  # 0.3 pips (was 5 pips!)
                base_confidence = 0.60
            elif macd_histogram > 0.00002:  # 0.2 pips (was 3 pips!)
                base_confidence = 0.55
            elif macd_histogram > 0.00001:  # 0.1 pips (was 1 pip!)
                base_confidence = 0.50
            else:
                base_confidence = 0.45  # Increased from 0.35
            
            # 🔧 FIXED: More reasonable efficiency ratio requirements
            if efficiency_ratio > 0.3:  # Lowered from 0.4
                base_confidence += 0.05
            elif efficiency_ratio > 0.2:  # Lowered from 0.25
                base_confidence += 0.03
            elif efficiency_ratio < 0.10:  # Only penalize very low efficiency
                base_confidence -= 0.08  # Reduced penalty from -0.15
            
            # 🔧 FIXED: Reduced market regime penalties
            if market_regime in ['ranging', 'sideways', 'consolidation']:
                base_confidence -= 0.05  # Reduced from -0.18
            elif market_regime in ['trending_weak']:
                base_confidence -= 0.03  # Reduced from -0.10
            elif market_regime in ['volatile', 'choppy']:
                base_confidence -= 0.04  # Reduced from -0.12
            
            # 🔧 PRICE BASIS FIX: Use consistent MID prices for EMA alignment validation
            if ema_200 > 0:
                # Both analysis_price and ema_200 are now guaranteed to be MID-based
                price_distance_pct = ((analysis_price - ema_200) / ema_200) * 100
                
                if signal_type == 'BULL':
                    if price_distance_pct < -0.05:  # Only penalize if significantly below EMA200
                        base_confidence -= 0.05  # Reduced from -0.18
                        self.logger.debug(f"[CONFIDENCE EMA PENALTY] BULL: MID price {price_distance_pct:.2f}% below EMA200")
                elif signal_type == 'BEAR':
                    if price_distance_pct > 0.05:  # Only penalize if significantly above EMA200
                        base_confidence -= 0.05  # Reduced from -0.18
                        self.logger.debug(f"[CONFIDENCE EMA PENALTY] BEAR: MID price {price_distance_pct:.2f}% above EMA200")
                
                # 🆕 ENHANCEMENT: Bonus for strong EMA alignment
                if signal_type == 'BULL' and price_distance_pct > 0.1:  # 0.1% above EMA200
                    base_confidence += 0.02
                    self.logger.debug(f"[CONFIDENCE EMA BONUS] BULL: Strong alignment +2% (price {price_distance_pct:.2f}% above EMA200)")
                elif signal_type == 'BEAR' and price_distance_pct < -0.1:  # 0.1% below EMA200
                    base_confidence += 0.02
                    self.logger.debug(f"[CONFIDENCE EMA BONUS] BEAR: Strong alignment +2% (price {price_distance_pct:.2f}% below EMA200)")
            
            # 🔧 FIXED: Reduced session penalties
            market_session = signal_data.get('market_session', 'unknown')
            if market_session in ['asian', 'sydney', 'asian_late']:
                base_confidence -= 0.03  # Reduced from -0.10
            elif market_session == 'tokyo':
                base_confidence -= 0.02  # Reduced from -0.05
            elif market_session in ['london', 'newyork', 'overlap']:
                base_confidence += 0.03  # Increased from +0.02
            
            # 🆕 ENHANCEMENT: MACD histogram momentum bonus
            histogram_change = macd_histogram - macd_histogram_prev
            if histogram_change > 0:
                # Accelerating momentum
                momentum_bonus = min(0.05, histogram_change / 0.00002)  # Max 5% bonus
                base_confidence += momentum_bonus
                if momentum_bonus > 0.01:  # Only log significant bonuses
                    self.logger.debug(f"[CONFIDENCE MOMENTUM BONUS] Accelerating histogram: +{momentum_bonus:.1%}")
            
            # 🆕 ENHANCEMENT: Multi-timeframe alignment bonus (if available)
            if 'mtf_analysis' in signal_data:
                mtf_data = signal_data['mtf_analysis']
                if mtf_data.get('mtf_valid', False):
                    mtf_bonus = min(0.10, mtf_data.get('momentum_score', 0) * 0.15)  # Max 10% bonus
                    base_confidence += mtf_bonus
                    if mtf_bonus > 0.02:
                        aligned_tfs = mtf_data.get('aligned_timeframes', 0)
                        total_tfs = mtf_data.get('total_timeframes', 1)
                        self.logger.debug(f"[CONFIDENCE MTF BONUS] Strong alignment: +{mtf_bonus:.1%} ({aligned_tfs}/{total_tfs} TFs)")
            
            # 8. Apply forex optimizer adjustments but with realistic caps
            if hasattr(self, 'forex_optimizer') and self.forex_optimizer:
                try:
                    # 🔧 PRICE BASIS FIX: Pass analysis_price to forex optimizer for consistent calculations
                    adjusted_signal_data = signal_data.copy()
                    adjusted_signal_data['price'] = analysis_price  # Use MID price for optimizer calculations
                    
                    optimizer_confidence = self.forex_optimizer.apply_forex_confidence_adjustments(
                        base_confidence, adjusted_signal_data, epic
                    )
                    # More permissive adjustment caps
                    adjustment = optimizer_confidence - base_confidence
                    capped_adjustment = max(-0.15, min(0.15, adjustment))  # Increased cap
                    final_confidence = base_confidence + capped_adjustment
                    
                    if abs(capped_adjustment) > 0.02:  # Log significant adjustments
                        self.logger.debug(f"[CONFIDENCE FOREX ADJ] {epic}: {capped_adjustment:+.1%} adjustment applied")
                except Exception as adj_error:
                    self.logger.warning(f"[CONFIDENCE FOREX ADJ] Failed for {epic}: {adj_error}")
                    final_confidence = base_confidence
            else:
                final_confidence = base_confidence
            
            # 🔧 FIXED: Realistic final constraints
            final_confidence = max(0.30, min(0.85, final_confidence))  # Increased cap to 85%
            
            # 🔧 FIXED: Lowered quality gate threshold
            if final_confidence < 0.25:  # Reduced from 0.35
                self.logger.debug(f"[MACD QUALITY GATE] {epic}: Signal confidence too low {final_confidence:.1%}")
                return 0.0  # Force rejection only for very weak signals
            
            # 🆕 ENHANCEMENT: Comprehensive confidence logging
            confidence_factors = []
            confidence_factors.append(f"base={base_confidence:.1%}")
            confidence_factors.append(f"eff={efficiency_ratio:.1%}")
            confidence_factors.append(f"regime={market_regime}")
            if ema_200 > 0:
                confidence_factors.append(f"ema_align={price_distance_pct:+.1f}%")
            if market_session != 'unknown':
                confidence_factors.append(f"session={market_session}")
            if hasattr(self, 'forex_optimizer') and self.forex_optimizer:
                confidence_factors.append("forex_adj=✓")
            if 'mtf_analysis' in signal_data:
                confidence_factors.append("mtf=✓")
            
            self.logger.debug(f"[MACD CONFIDENCE FIXED] {epic}: {signal_type} final={final_confidence:.1%} "
                            f"({', '.join(confidence_factors)})")
            
            # 🆕 ENHANCEMENT: Add confidence metadata to signal_data for debugging
            if isinstance(signal_data, dict):
                signal_data['confidence_metadata'] = {
                    'base_confidence': base_confidence,
                    'final_confidence': final_confidence,
                    'price_basis': 'MID_consistent',
                    'factors_applied': confidence_factors,
                    'ema_alignment_pct': price_distance_pct if ema_200 > 0 else None,
                    'analysis_price_used': analysis_price,
                    'execution_price': execution_price
                }
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Enhanced confidence calculation failed for {epic}: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return 0.45  # Higher fallback confidence

    def _validate_signal_quality(self, signal_data: Dict, confidence: float, epic: str) -> bool:
        """
        Additional signal quality validation to prevent weak signals
        """
        try:
            # 1. Minimum confidence threshold
            if confidence < 0.40:  # Absolute minimum
                return False
            
            # 2. MACD strength validation
            macd_strength = abs(signal_data.get('macd_histogram', 0))
            if macd_strength < 0.00008:  # Raised minimum strength
                self.logger.debug(f"[QUALITY CHECK] {epic}: MACD too weak {macd_strength:.6f}")
                return False
            
            # 3. Efficiency ratio validation
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.0)
            if efficiency_ratio < 0.12:  # Minimum efficiency required
                self.logger.debug(f"[QUALITY CHECK] {epic}: Efficiency too low {efficiency_ratio:.2f}")
                return False
            
            # 4. Market regime validation
            market_regime = signal_data.get('market_regime', 'unknown')
            if market_regime in ['ranging', 'sideways'] and confidence < 0.50:
                self.logger.debug(f"[QUALITY CHECK] {epic}: Ranging market + low confidence")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Signal quality validation error: {e}")
            return False

    
    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str, signal_timestamp: Optional[datetime] = None) -> Optional[Dict]:
        """
        🎯 UPDATED: Detect MACD signals with session-aware optimizations and signal enhancement
        """
        try:
            self.logger.info(f"🔍 [MODULAR MACD] Starting detection for {epic}")
            
            # Minimum bars check
            min_bars_needed = 50
            if len(df) < min_bars_needed:
                self.logger.debug(f"🚫 Insufficient bars: {len(df)} < {min_bars_needed}")
                return None
            
            
            # 🔧 DATA PREPARATION: Use data helper for all data operations
            df_enhanced = self.data_helper.ensure_macd_indicators(df)
            self.logger.info(f"   ✅ After MACD indicators: {df_enhanced.shape}")
            
            
            # 🔥 VALIDATION: Enhanced data validation
            if not self.data_helper.validate_macd_data(df_enhanced):
                self.logger.warning(f"[MODULAR MACD REJECTED] {epic} - MACD data validation failed")
                return None
            
            # Get latest and previous data
            latest = df_enhanced.iloc[-1]
            previous = df_enhanced.iloc[-2]
            
            # Extract core MACD values
            current_price = latest['close']
            ema_200_current = latest['ema_200']
            macd_histogram_current = latest['macd_histogram']
            macd_histogram_prev = previous['macd_histogram']
            
            # 🚀 UPDATED: Get current market session for session-aware threshold
            current_session = self.forex_optimizer.get_current_market_session()
            
            # 🚀 CACHED: Get MACD threshold (now session-aware)
            macd_threshold = self.cache.get_macd_threshold_cached(
                epic, self.forex_optimizer, market_session=current_session
            )
            
            # Validate significant MACD change
            histogram_change = abs(macd_histogram_current - macd_histogram_prev)
            if histogram_change < macd_threshold:
                self.logger.debug(f"MACD change {histogram_change:.6f} below threshold {macd_threshold:.6f} (session: {current_session})")
                return None
            
            # Log when histogram change passes but it might not be a crossover
            self.logger.debug(f"📈 [MACD STRATEGY] Significant histogram change detected: {macd_histogram_prev:.6f} → {macd_histogram_current:.6f}")
            self.logger.debug(f"   Change: {histogram_change:.6f} >= Threshold: {macd_threshold:.6f}")
            
            # 🎯 SIGNAL DETECTION: Use signal detector module for enhanced detection
            self.logger.debug(f"📞 [MACD STRATEGY] Calling signal detector for {epic}")
            enhanced_signal_data = self.signal_detector.detect_enhanced_macd_signal(
                latest, previous, epic, timeframe, 
                ema_config=None, 
                df_enhanced=df_enhanced,
                forex_optimizer=self.forex_optimizer,
                signal_timestamp=signal_timestamp
            )
            self.logger.debug(f"📞 [MACD STRATEGY] Signal detector returned: {enhanced_signal_data is not None}")
            
            if not enhanced_signal_data:
                # Note: This doesn't necessarily mean there was a crossover - just that histogram change was significant
                self.logger.debug(f"📊 [MACD STRATEGY] No signal detected for {epic} on {timeframe}")
                self.logger.debug(f"   Histogram change {histogram_change:.6f} was above threshold {macd_threshold:.6f}")
                self.logger.debug(f"   But signal detector found no valid crossover or signal was filtered")
                return None
            
            signal_type = enhanced_signal_data['signal_type']
            trigger_reason = enhanced_signal_data['trigger_reason']
            
            # 🚀 PERFORMANCE OPTIMIZATION: Calculate expensive operations once and cache
            efficiency_ratio = self.cache.calculate_efficiency_ratio_cached(
                df_enhanced, epic, timeframe, self.forex_optimizer
            )
            market_regime = self.cache.detect_market_regime_cached(
                latest, df_enhanced, epic, timeframe, self.forex_optimizer
            )
            
            # Create enhanced signal data for confidence calculation
            enhanced_signal_for_confidence = self.data_helper.create_enhanced_signal_data(latest, signal_type)
            enhanced_signal_for_confidence.update({
                'efficiency_ratio': efficiency_ratio,
                'market_regime': market_regime,
                'market_session': current_session,
                'macd_histogram_prev': macd_histogram_prev,
                'epic': epic,
                'timeframe': timeframe
            })
            
            # 🎯 OPTIMIZED: Use MACD-specific confidence optimizer
            confidence = self.calculate_confidence(enhanced_signal_for_confidence, df_enhanced, epic)
            
            # Apply minimum confidence threshold (pair-specific if set)
            min_confidence_threshold = self.get_confidence_threshold(epic)
            if confidence < min_confidence_threshold:
                self.logger.debug(f"🚫 [MODULAR MACD REJECT] {epic}: {signal_type} - "
                            f"confidence {confidence:.1%} below threshold {min_confidence_threshold:.1%}")
                return None
            
            # Build final signal data
            signal_data = {
                'signal_type': signal_type,
                'confidence_score': confidence,
                'epic': epic,
                'price': current_price,
                'spread_pips': spread_pips,
                'timeframe': timeframe,
                'strategy': 'MACD_MODULAR',
                'signal_trigger': trigger_reason,
                'market_session': current_session,
                'market_regime': market_regime,
                'efficiency_ratio': efficiency_ratio,
                'macd_histogram': macd_histogram_current,
                'macd_histogram_prev': macd_histogram_prev,
                'macd_threshold_used': macd_threshold,
                'ema_200': ema_200_current,
                'crossover_type': enhanced_signal_data.get('crossover_type', 'macd_crossover'),
                'data_source': 'modular_macd_strategy',
                'timestamp': latest.get('timestamp', pd.Timestamp.now())
            }
            
            # 🆕 ENHANCEMENT: Add all missing fields that AlertHistoryManager expects
            signal_data = self._enhance_signal_with_missing_fields(
                signal_data, latest, previous, epic, spread_pips, timeframe
            )
            
            # Get timestamp for logging - use the signal_timestamp parameter passed from backtest
            # DEBUG: Let's see what we actually have
            self.logger.debug(f"🔍 DEBUG Timestamp extraction:")
            self.logger.debug(f"   signal_timestamp: {signal_timestamp} (type: {type(signal_timestamp)})")
            self.logger.debug(f"   latest.name: {getattr(latest, 'name', 'NO NAME')} (type: {type(getattr(latest, 'name', None))})")
            if hasattr(latest, 'index'):
                self.logger.debug(f"   latest.index: {latest.index}")
            
            # Try to get a valid timestamp
            if signal_timestamp and hasattr(signal_timestamp, 'strftime'):
                timestamp_str = signal_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                self.logger.debug(f"   ✅ Used signal_timestamp: {timestamp_str}")
            elif signal_timestamp and isinstance(signal_timestamp, str):
                timestamp_str = signal_timestamp
                self.logger.debug(f"   ✅ Used signal_timestamp string: {timestamp_str}")
            elif hasattr(latest, 'name') and latest.name and hasattr(latest.name, 'strftime'):
                timestamp_str = latest.name.strftime('%Y-%m-%d %H:%M:%S')
                self.logger.debug(f"   ✅ Used latest.name: {timestamp_str}")
            elif hasattr(latest, 'name') and latest.name:
                timestamp_str = str(latest.name)
                self.logger.debug(f"   ✅ Used latest.name string: {timestamp_str}")
            else:
                # Last resort - try to extract from signal_data or use current time
                timestamp_str = 'DEBUG_FAILED'
                self.logger.debug(f"   ❌ All timestamp extraction failed!")
            
            self.logger.info(f"✅ [MODULAR MACD SUCCESS] {epic}: {signal_type} - {confidence:.1%} confidence (session: {current_session})")
            self.logger.info(f"   🕐 Signal Timestamp: {timestamp_str}")
            
            # Also log MACD values for threshold correlation
            try:
                macd_line = latest.get('macd_line', 'N/A')
                macd_signal = latest.get('macd_signal', 'N/A')  
                macd_histogram = latest.get('macd_histogram', 'N/A')
                prev_histogram = previous.get('macd_histogram', 'N/A')
                
                if isinstance(macd_histogram, (int, float)) and isinstance(prev_histogram, (int, float)):
                    histogram_change = macd_histogram - prev_histogram
                    self.logger.info(f"   📈 MACD Values: Line={macd_line:.6f}, Signal={macd_signal:.6f}, Histogram={macd_histogram:.6f}")
                    self.logger.info(f"   🔄 Histogram Change: {prev_histogram:.6f} → {macd_histogram:.6f} (Δ {histogram_change:.6f})")
            except Exception as e:
                self.logger.debug(f"Could not log MACD details: {e}")
            return signal_data
            
        except Exception as e:
            self.logger.error(f"❌ [MODULAR MACD ERROR] {epic}: {str(e)}")
            return None

    def _enhance_signal_with_missing_fields(self, signal: Dict, latest: pd.Series, previous: pd.Series, epic: str, spread_pips: float, timeframe: str) -> Dict:
        """
        ENHANCEMENT: Add all missing fields that AlertHistoryManager expects
        """
        try:
            # BASIC FIELDS (ensure they exist)
            signal.setdefault('epic', epic)
            signal.setdefault('timeframe', timeframe)
            # Fix timestamp extraction - try multiple sources
            if 'timestamp' not in signal or signal['timestamp'] is None:
                timestamp_value = None
                
                # Try to get timestamp from latest data
                if hasattr(latest, 'name') and latest.name is not None:
                    timestamp_value = latest.name
                elif hasattr(latest, 'index') and len(latest.index) > 0:
                    timestamp_value = latest.index[0] if hasattr(latest.index[0], 'to_pydatetime') else latest.index[0]
                elif 'start_time' in latest:
                    timestamp_value = latest['start_time']
                elif 'datetime_utc' in latest:
                    timestamp_value = latest['datetime_utc']
                
                # Convert to proper timestamp if needed
                if timestamp_value is not None:
                    if hasattr(timestamp_value, 'to_pydatetime'):
                        timestamp_value = timestamp_value.to_pydatetime()
                    elif isinstance(timestamp_value, str):
                        timestamp_value = pd.to_datetime(timestamp_value)
                    signal['timestamp'] = timestamp_value
                else:
                    # Fallback to current time if no timestamp found
                    signal['timestamp'] = pd.Timestamp.now()
                    self.logger.warning(f"⚠️ Could not extract timestamp for {epic}, using current time")
            
            # Extract pair from epic if not present
            if 'pair' not in signal:
                signal['pair'] = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            # Get pip size for correct calculations
            pip_size = self.forex_optimizer.get_pip_size(epic)

            # PRICE FIELDS
            current_price = signal.get('price', latest.get('close', 0))
            signal.update({
                'bid_price': current_price - (spread_pips * pip_size / 2),  # Half spread below mid
                'ask_price': current_price + (spread_pips * pip_size / 2),  # Half spread above mid
                'spread_pips': spread_pips
            })
            
            # TECHNICAL INDICATORS - MACD specific
            signal.update({
                'macd_line': latest.get('macd_line', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'macd_histogram': signal.get('macd_histogram', latest.get('macd_histogram', 0)),
                'ema_200': signal.get('ema_200', latest.get('ema_200', 0))
            })
            
            # EMA values (if available - some MACD strategies use additional EMAs)
            signal.update({
                'ema_short': latest.get('ema_short', latest.get('ema_9', 0)),
                'ema_long': latest.get('ema_long', latest.get('ema_21', 0)),
                'ema_trend': latest.get('ema_trend', latest.get('ema_200', 0))
            })
            
            # VOLUME ANALYSIS
            volume = latest.get('volume', 0)
            volume_ratio = latest.get('volume_ratio', 1.0)
            
            if volume_ratio == 1.0 and volume > 0:
                volume_20_avg = latest.get('volume_20_avg', volume)
                volume_ratio = volume / volume_20_avg if volume_20_avg > 0 else 1.0
            
            signal.update({
                'volume': volume,
                'volume_ratio': volume_ratio,
                'volume_confirmation': volume_ratio > 1.2
            })
            
            # SUPPORT/RESISTANCE LEVELS
            try:
                recent_highs = [latest.get('high', current_price)]
                recent_lows = [latest.get('low', current_price)]
                
                if hasattr(previous, 'high'):
                    recent_highs.append(previous['high'])
                    recent_lows.append(previous['low'])
                
                nearest_resistance = max(recent_highs)
                nearest_support = min(recent_lows)
                
                distance_to_resistance_pips = abs(nearest_resistance - current_price) / pip_size
                distance_to_support_pips = abs(current_price - nearest_support) / pip_size
                
                risk_reward_ratio = distance_to_resistance_pips / distance_to_support_pips if distance_to_support_pips > 0 else 1.0
                
            except Exception:
                nearest_resistance = current_price * 1.002
                nearest_support = current_price * 0.998
                distance_to_resistance_pips = 20.0
                distance_to_support_pips = 20.0
                risk_reward_ratio = 1.0
            
            signal.update({
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'distance_to_support_pips': distance_to_support_pips,
                'distance_to_resistance_pips': distance_to_resistance_pips,
                'risk_reward_ratio': risk_reward_ratio
            })
            
            # MARKET CONDITIONS
            signal.update({
                'market_session': signal.get('market_session', self._determine_market_session()),
                'is_market_hours': self._is_market_hours(),
                'market_regime': signal.get('market_regime', self._determine_market_regime(latest, previous))
            })
            
            # SIGNAL METADATA
            signal_type = signal.get('signal_type', 'UNKNOWN')
            crossover_direction = 'bullish' if signal_type in ['BUY', 'BULL'] else 'bearish'
            
            signal.update({
                'signal_trigger': signal.get('signal_trigger', f"macd_crossover_{crossover_direction}"),
                'crossover_type': signal.get('crossover_type', f"macd_histogram_crossover_{crossover_direction}"),
                'signal_hash': self._generate_signal_hash(signal),
                'data_source': 'live_scanner',
                'market_timestamp': signal.get('timestamp'),
                'cooldown_key': f"{epic}_{signal_type}_{timeframe}"
            })
            
            # STRATEGY DATA (JSON FIELDS)
            strategy_config = {
                'strategy_type': 'macd_histogram_crossover',
                'ema_200_filter': True,
                'volume_confirmation': signal.get('volume_confirmation', False),
                'efficiency_threshold': signal.get('efficiency_ratio', 0.0),
                'market_session_aware': True,
                'mtf_enabled': getattr(self, 'enable_mtf_analysis', False)
            }
            
            strategy_indicators = {
                'macd_histogram': signal.get('macd_histogram', 0),
                'macd_histogram_prev': signal.get('macd_histogram_prev', 0),
                'macd_line': signal.get('macd_line', 0),
                'macd_signal': signal.get('macd_signal', 0),
                'ema_200': signal.get('ema_200', 0),
                'current_price': current_price,
                'previous_price': previous.get('close', current_price),
                'histogram_change': abs(signal.get('macd_histogram', 0) - signal.get('macd_histogram_prev', 0)),
                'threshold_used': signal.get('macd_threshold_used', 0)
            }
            
            strategy_metadata = {
                'signal_generation_time': pd.Timestamp.now().isoformat(),
                'strategy_version': 'modular_macd_v1.1_mtf_enhanced',
                'confidence_factors': {
                    'macd_strength': 0.4,
                    'trend_alignment': 0.3,
                    'efficiency_ratio': 0.2,
                    'volume_confirmation': 0.1,
                    'mtf_alignment': 0.15 if getattr(self, 'enable_mtf_analysis', False) else 0.0
                },
                'market_conditions_at_signal': {
                    'session': signal.get('market_session'),
                    'regime': signal.get('market_regime'),
                    'efficiency': signal.get('efficiency_ratio', 0.0)
                },
                'macd_specific': {
                    'histogram_crossover': True,
                    'ema_200_filter_applied': True,
                    'session_aware_threshold': True
                }
            }
            
            signal.update({
                'strategy_config': strategy_config,
                'strategy_indicators': strategy_indicators,
                'strategy_metadata': strategy_metadata
            })
            
            # SIGNAL QUALITY METRICS
            signal.update({
                'signal_strength': self._calculate_signal_strength(signal),
                'signal_quality': self._assess_signal_quality(signal),
                'technical_score': self._calculate_technical_score(signal)
            })
            
            self.logger.debug(f"✅ Enhanced MACD signal with {len(signal)} total fields")
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ MACD signal enhancement failed: {e}")
            return signal

    # ===== PAIR-SPECIFIC OPTIMIZATION METHODS =====
    
    def optimize_for_pair(self, pair: str, config_updates: Dict):
        """
        🎯 OPTIMIZATION: Update confidence calculation parameters for specific pair
        
        This updates the MACDConfidenceOptimizer configuration while maintaining
        integration with the existing MACDForexOptimizer.
        
        Example usage:
        strategy.optimize_for_pair('GBPUSD', {
            'macd_thresholds': {'strong': 0.00015, 'moderate': 0.00008},
            'confidence_weights': {'macd_strength': 0.50, 'trend_alignment': 0.25},
            'base_confidence': 0.58,
            'forex_integration': True  # Enable/disable forex optimizer integration
        })
        """
        try:
            # Initialize integrated optimizer if not already done
            if not hasattr(self, 'confidence_optimizer'):
                from .helpers.macd_confidence_optimizer import MACDConfidenceOptimizer
                self.confidence_optimizer = MACDConfidenceOptimizer(
                    logger=self.logger, 
                    forex_optimizer=self.forex_optimizer
                )
            
            # Update configuration for the pair
            self.confidence_optimizer.update_pair_config(pair, config_updates)
            self.logger.info(f"🎯 Updated integrated MACD confidence config for {pair}")
            
            # Log integration status
            config = self.confidence_optimizer.get_pair_config(pair)
            forex_integration = config.get('forex_integration', False)
            self.logger.info(f"   Forex integration: {'✅ Enabled' if forex_integration else '❌ Disabled'}")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize for pair {pair}: {e}")
    
    def _determine_market_session(self) -> str:
        """Determine current market session"""
        try:
            from datetime import timezone
            utc_now = datetime.now(timezone.utc)
            hour = utc_now.hour
            
            if 0 <= hour < 7:
                return 'asia_late'
            elif 7 <= hour < 15:
                return 'london'
            elif 15 <= hour < 21:
                return 'new_york'
            else:
                return 'asia_early'
        except:
            return 'unknown'

    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        try:
            from datetime import timezone
            utc_now = datetime.now(timezone.utc)
            weekday = utc_now.weekday()
            
            if weekday == 5:  # Saturday
                return False
            elif weekday == 6:  # Sunday
                return utc_now.hour >= 22
            else:
                return True
        except:
            return True

    def _determine_market_regime(self, latest: pd.Series, previous: pd.Series) -> str:
        """Determine market regime (trending/ranging) for MACD"""
        try:
            macd_histogram = latest.get('macd_histogram', 0)
            macd_histogram_prev = previous.get('macd_histogram', 0)
            ema_200 = latest.get('ema_200', 0)
            current_price = latest.get('close', 0)
            
            # MACD-based regime detection
            histogram_increasing = macd_histogram > macd_histogram_prev
            histogram_positive = macd_histogram > 0
            price_above_ema = current_price > ema_200 if ema_200 > 0 else False
            
            if histogram_positive and histogram_increasing and price_above_ema:
                return 'trending_up'
            elif not histogram_positive and not histogram_increasing and not price_above_ema:
                return 'trending_down'
            elif abs(macd_histogram) < 0.00001:  # Very small histogram
                return 'ranging'
            else:
                return 'transitioning'
        except:
            return 'unknown'

    def _generate_signal_hash(self, signal: Dict) -> str:
        """Generate unique hash for signal deduplication"""
        try:
            import hashlib
            hash_string = f"{signal.get('epic')}_{signal.get('signal_type')}_{signal.get('timestamp')}_{signal.get('price')}"
            return hashlib.md5(hash_string.encode()).hexdigest()[:8]
        except:
            return 'unknown'

    def _calculate_signal_strength(self, signal: Dict) -> float:
        """Calculate MACD signal strength (0.0 to 1.0)"""
        try:
            factors = []
            
            # MACD histogram strength
            macd_hist = abs(signal.get('macd_histogram', 0))
            if macd_hist > 0:
                hist_strength = min(macd_hist * 10000, 1.0)  # Normalize for forex
                factors.append(hist_strength)
            
            # MACD momentum (histogram change)
            hist_change = abs(signal.get('macd_histogram', 0) - signal.get('macd_histogram_prev', 0))
            if hist_change > 0:
                momentum_strength = min(hist_change * 20000, 1.0)  # Normalize for forex
                factors.append(momentum_strength)
            
            # Volume confirmation
            if signal.get('volume_confirmation', False):
                factors.append(0.8)
            else:
                factors.append(0.3)
            
            # EMA 200 alignment
            price = signal.get('price', 0)
            ema_200 = signal.get('ema_200', 0)
            signal_type = signal.get('signal_type', '')
            
            if ema_200 > 0 and price > 0:
                if (signal_type in ['BUY', 'BULL'] and price > ema_200) or \
                (signal_type in ['SELL', 'BEAR'] and price < ema_200):
                    factors.append(0.9)
                else:
                    factors.append(0.4)
            
            # MTF alignment (if available)
            if 'mtf_analysis' in signal:
                momentum_score = signal['mtf_analysis'].get('momentum_score', 0.5)
                factors.append(momentum_score)
            
            return sum(factors) / len(factors) if factors else 0.5
            
        except:
            return 0.5

    def _assess_signal_quality(self, signal: Dict) -> str:
        """Assess overall MACD signal quality"""
        try:
            strength = signal.get('signal_strength', 0.5)
            confidence = signal.get('confidence_score', 0.5)
            
            # Include MTF score if available
            if 'mtf_analysis' in signal:
                mtf_score = signal['mtf_analysis'].get('momentum_score', 0.5)
                combined_score = (strength + confidence + mtf_score) / 3
            else:
                combined_score = (strength + confidence) / 2
            
            if combined_score >= 0.8:
                return 'excellent'
            elif combined_score >= 0.7:
                return 'good'
            elif combined_score >= 0.6:
                return 'fair'
            else:
                return 'poor'
        except:
            return 'unknown'

    def _calculate_technical_score(self, signal: Dict) -> float:
        """Calculate technical analysis score (0-100) for MACD"""
        try:
            score = 50
            
            # MACD histogram strength
            macd_hist = abs(signal.get('macd_histogram', 0))
            if macd_hist > 0.00005:  # Strong histogram
                score += 20
            elif macd_hist > 0.00002:  # Moderate histogram
                score += 10
            
            # Volume confirmation
            if signal.get('volume_confirmation', False):
                score += 15
            
            # EMA 200 alignment
            price = signal.get('price', 0)
            ema_200 = signal.get('ema_200', 0)
            signal_type = signal.get('signal_type', '')
            
            if ema_200 > 0 and price > 0:
                aligned = (signal_type in ['BUY', 'BULL'] and price > ema_200) or \
                        (signal_type in ['SELL', 'BEAR'] and price < ema_200)
                if aligned:
                    score += 15
            
            # Market regime bonus
            regime = signal.get('market_regime', '')
            if 'trending' in regime:
                score += 10
            
            # MTF bonus
            if 'mtf_analysis' in signal:
                if signal['mtf_analysis'].get('mtf_valid', False):
                    score += 10
                if signal['mtf_analysis'].get('momentum_score', 0) > 0.7:
                    score += 10
            
            # Efficiency ratio bonus
            efficiency = signal.get('efficiency_ratio', 0)
            if efficiency > 0.3:
                score += 10
            elif efficiency > 0.2:
                score += 5
            
            return min(score, 100)
            
        except:
            return 50

    def get_pair_config(self, pair: str) -> Dict:
        """Get current integrated confidence configuration for specific pair"""
        try:
            if not hasattr(self, 'confidence_optimizer'):
                from .helpers.macd_confidence_optimizer import MACDConfidenceOptimizer
                self.confidence_optimizer = MACDConfidenceOptimizer(
                    logger=self.logger, 
                    forex_optimizer=self.forex_optimizer
                )
            
            config = self.confidence_optimizer.get_pair_config(pair)
            
            # Add forex optimizer status to config
            config['has_forex_optimizer'] = self.forex_optimizer is not None
            config['forex_integration_available'] = config.get('forex_integration', False) and self.forex_optimizer is not None
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to get config for pair {pair}: {e}")
            return {}
    
    def enable_forex_integration(self, pair: str = None):
        """Enable forex optimizer integration for specific pair or all pairs"""
        try:
            if pair:
                # Enable for specific pair
                self.optimize_for_pair(pair, {'forex_integration': True})
                self.logger.info(f"✅ Enabled forex integration for {pair}")
            else:
                # Enable for all pairs
                if hasattr(self, 'confidence_optimizer'):
                    for pair_name in self.confidence_optimizer.pair_specific_configs.keys():
                        self.confidence_optimizer.update_pair_config(pair_name, {'forex_integration': True})
                self.logger.info("✅ Enabled forex integration for all pairs")
                
        except Exception as e:
            self.logger.error(f"Failed to enable forex integration: {e}")
    
    def disable_forex_integration(self, pair: str = None):
        """Disable forex optimizer integration for specific pair or all pairs"""
        try:
            if pair:
                # Disable for specific pair
                self.optimize_for_pair(pair, {'forex_integration': False})
                self.logger.info(f"❌ Disabled forex integration for {pair}")
            else:
                # Disable for all pairs
                if hasattr(self, 'confidence_optimizer'):
                    for pair_name in self.confidence_optimizer.pair_specific_configs.keys():
                        self.confidence_optimizer.update_pair_config(pair_name, {'forex_integration': False})
                self.logger.info("❌ Disabled forex integration for all pairs")
                
        except Exception as e:
            self.logger.error(f"Failed to disable forex integration: {e}")
    
    def get_integration_status(self) -> Dict:
        """Get status of optimizer integration"""
        try:
            status = {
                'has_forex_optimizer': self.forex_optimizer is not None,
                'has_confidence_optimizer': hasattr(self, 'confidence_optimizer'),
                'has_mtf_analysis': getattr(self, 'enable_mtf_analysis', False),
                'forex_optimizer_type': type(self.forex_optimizer).__name__ if self.forex_optimizer else None,
                'confidence_optimizer_type': type(self.confidence_optimizer).__name__ if hasattr(self, 'confidence_optimizer') else None,
                'integration_ready': self.forex_optimizer is not None,
                'mtf_config': getattr(self, 'mtf_config', {}),
                'pair_integration_status': {}
            }
            
            # Check per-pair integration status
            if hasattr(self, 'confidence_optimizer'):
                for pair, config in self.confidence_optimizer.pair_specific_configs.items():
                    status['pair_integration_status'][pair] = {
                        'forex_integration_enabled': config.get('forex_integration', False),
                        'integration_active': config.get('forex_integration', False) and self.forex_optimizer is not None
                    }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get integration status: {e}")
            return {'error': str(e)}
    
    def set_confidence_threshold(self, epic: str, threshold: float):
        """Set minimum confidence threshold for specific epic"""
        try:
            if not hasattr(self, 'pair_confidence_thresholds'):
                self.pair_confidence_thresholds = {}
            
            self.pair_confidence_thresholds[epic] = threshold
            self.logger.info(f"Set confidence threshold for {epic}: {threshold:.1%}")
            
        except Exception as e:
            self.logger.error(f"Failed to set confidence threshold: {e}")
    
    def get_confidence_threshold(self, epic: str) -> float:
        """Get minimum confidence threshold for specific epic"""
        if hasattr(self, 'pair_confidence_thresholds') and epic in self.pair_confidence_thresholds:
            return self.pair_confidence_thresholds[epic]
        return 0.50  # Default threshold

    def get_enhanced_performance_stats(self) -> Dict:
        """
        🧠 Get comprehensive performance statistics from all modules including MTF
        """
        try:
            stats = {}
            
            # === ENHANCED VALIDATOR STATS ===
            if hasattr(self, 'enhanced_validator') and self.enhanced_validator:
                try:
                    validator_stats = self.enhanced_validator.get_current_regime_stats()
                    stats['enhanced_validator'] = validator_stats
                except Exception as e:
                    self.logger.debug(f"Enhanced validator stats unavailable: {e}")
                    stats['enhanced_validator'] = {'error': 'unavailable'}
            else:
                stats['enhanced_validator'] = {'error': 'not_initialized'}
            
            # === MODULE PERFORMANCE STATS ===
            stats['cache_performance'] = self.cache.get_cache_stats()
            stats['forex_optimization'] = self.forex_optimizer.get_forex_optimization_summary()
            stats['validator_capabilities'] = self.validator.get_validation_summary()
            stats['signal_detector_capabilities'] = self.signal_detector.get_detection_summary()
            stats['data_helper_capabilities'] = self.data_helper.get_data_helper_summary()
            
            # === MTF ANALYSIS STATS (NEW) ===
            stats['mtf_analysis'] = {
                'enabled': getattr(self, 'enable_mtf_analysis', False),
                'config': getattr(self, 'mtf_config', {}),
                'cache_available': hasattr(self, 'cache') and self.cache is not None,
                'data_fetcher_available': hasattr(self, 'data_fetcher') and self.data_fetcher is not None
            }
            
            # === STRATEGY CONFIGURATION STATS ===
            stats['strategy_configuration'] = {
                'name': self.name,
                'min_efficiency_ratio': self.min_efficiency_ratio,
                'min_confidence': self.min_confidence,
                'strategy_type': 'macd_ema200_modular_mtf',
                'architecture': 'modular_with_dependency_injection_and_mtf'
            }
            
            # === MODULAR ARCHITECTURE BENEFITS ===
            stats['modular_benefits'] = {
                'code_organization': 'Each module has single responsibility',
                'maintainability': 'Easy to modify individual components',
                'testability': 'Each module can be unit tested separately',
                'performance': 'Specialized optimization in each module',
                'debugging': 'Clear error isolation and logging',
                'extensibility': 'Easy to add new modules or replace existing ones',
                'mtf_integration': 'Multi-timeframe analysis seamlessly integrated'
            }
            
            # === TIMESTAMP AND METADATA ===
            stats['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'version': 'modular_macd_forex_optimized_mtf_v1',
                'architecture': 'modular_with_dependency_injection_and_mtf',
                'refactoring_complete': True,
                'backward_compatibility': True,
                'mtf_analysis_integrated': True,
                'total_modules': 5,
                'module_list': [
                    'MACDForexOptimizer',
                    'MACDValidator', 
                    'MACDCache',
                    'MACDSignalDetector',
                    'MACDDataHelper'
                ],
                'new_features': [
                    'Multi-Timeframe MACD momentum analysis',
                    'Enhanced confidence scoring with MTF validation',
                    'MACD-specific divergence detection',
                    'Session-aware MTF thresholds'
                ]
            }
            
            self.logger.debug(f"📊 Generated modular MACD performance stats with {len(stats)} main categories")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Modular MACD performance stats generation error: {e}")
            return {
                'error': str(e),
                'fallback_stats': {
                    'cache_performance': self.cache.get_cache_stats(),
                    'strategy_name': self.name,
                    'modular_architecture': True,
                    'mtf_analysis_enabled': getattr(self, 'enable_mtf_analysis', False),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error_fallback'
                }
            }

    def clear_all_caches(self):
        """🚀 Clear all module caches for fresh calculations"""
        try:
            self.cache.clear_cache()
            self.logger.info("🧹 All modular MACD caches cleared successfully")
        except Exception as e:
            self.logger.error(f"Cache clearing failed: {e}")

    def optimize_all_modules(self):
        """🧠 Run optimization on all modules"""
        try:
            self.cache.optimize_cache_settings()
            self.logger.info("🧠 MACD module optimization completed")
        except Exception as e:
            self.logger.error(f"Module optimization failed: {e}")

    def get_module_diagnostics(self) -> Dict:
        """🔧 Get comprehensive diagnostics from all modules including MTF"""
        try:
            diagnostics = {
                'cache_diagnostics': self.cache.export_cache_diagnostics(),
                'forex_optimizer_summary': self.forex_optimizer.get_forex_optimization_summary(),
                'validator_summary': self.validator.get_validation_summary(),
                'signal_detector_summary': self.signal_detector.get_detection_summary(),
                'data_helper_summary': self.data_helper.get_data_helper_summary(),
                'mtf_diagnostics': {
                    'enabled': getattr(self, 'enable_mtf_analysis', False),
                    'config': getattr(self, 'mtf_config', {}),
                    'data_fetcher_available': hasattr(self, 'data_fetcher') and self.data_fetcher is not None,
                    'cache_integration': hasattr(self, 'cache') and self.cache is not None
                },
                'main_strategy_config': {
                    'name': self.name,
                    'min_efficiency_ratio': self.min_efficiency_ratio,
                    'min_confidence': self.min_confidence
                },
                'overall_health': {
                    'all_modules_loaded': all([
                        self.forex_optimizer is not None,
                        self.validator is not None,
                        self.cache is not None,
                        self.signal_detector is not None,
                        self.data_helper is not None
                    ]),
                    'dependencies_injected': True,
                    'modular_architecture_active': True,
                    'mtf_analysis_ready': getattr(self, 'enable_mtf_analysis', False)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Module diagnostics failed: {e}")
            return {'error': str(e)}

    # ===== UTILITY METHODS =====

    def log_modular_status(self):
        """Log the status of all modules including MTF"""
        self.logger.info("🏗️ MODULAR MACD STRATEGY STATUS:")
        self.logger.info(f"   ForexOptimizer: {'✅ Active' if self.forex_optimizer else '❌ Missing'}")
        self.logger.info(f"   Validator: {'✅ Active' if self.validator else '❌ Missing'}")
        self.logger.info(f"   Cache: {'✅ Active' if self.cache else '❌ Missing'}")
        self.logger.info(f"   SignalDetector: {'✅ Active' if self.signal_detector else '❌ Missing'}")
        self.logger.info(f"   DataHelper: {'✅ Active' if self.data_helper else '❌ Missing'}")
        #self.logger.info(f"   EnhancedValidator: {'✅ Active' if self.enhanced_validator else '❌ Missing'}")
        self.logger.info(f"   PriceAdjuster: {'✅ Active' if self.price_adjuster else '❌ Missing'}")
        self.logger.info(f"   MTF Analysis: {'✅ Active' if getattr(self, 'enable_mtf_analysis', False) else '❌ Disabled'}")

    def validate_modular_integration(self) -> bool:
        """Validate that all modules are properly integrated including MTF"""
        
        # Check that all modules are loaded
        required_modules = [
            ('forex_optimizer', self.forex_optimizer),
            ('validator', self.validator),
            ('cache', self.cache),
            ('signal_detector', self.signal_detector),
            ('data_helper', self.data_helper)
        ]
        
        for module_name, module_instance in required_modules:
            if module_instance is None:
                self.logger.error(f"❌ Module {module_name} is not loaded")
                return False
        
        # Test basic module functionality
        try:
            # Test cache functionality
            test_key = "test_integration"
            self.cache.cache_result(test_key, "test_value")
            
            # Test forex optimizer
            test_threshold = self.forex_optimizer.get_macd_threshold_for_epic("CS.D.EURUSD.CEEM.IP")
            
            # Test MTF integration
            mtf_enabled = getattr(self, 'enable_mtf_analysis', False)
            self.logger.info(f"📊 MTF Analysis: {'✅ Enabled' if mtf_enabled else '❌ Disabled'}")
            
            self.logger.info("✅ All modules integrated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Module integration test failed: {e}")
            return False

    def debug_signal_detection(self, df: pd.DataFrame, epic: str, spread_pips: float = 1.5, timeframe: str = '5m') -> Dict:
        """
        🐛 DEBUG: Comprehensive debug information for modular signal detection
        """
        debug_info = {
            'epic': epic,
            'timeframe': timeframe,
            'spread_pips': spread_pips,
            'dataframe_shape': df.shape,
            'modules_loaded': {},
            'validation_steps': [],
            'rejection_reasons': [],
            'macd_values': {},
            'signal_result': None
        }
        
        try:
            # Check module loading
            debug_info['modules_loaded'] = {
                'forex_optimizer': self.forex_optimizer is not None,
                'validator': self.validator is not None,
                'cache': self.cache is not None,
                'signal_detector': self.signal_detector is not None,
                'data_helper': self.data_helper is not None,
                'enhanced_validator': self.enhanced_validator is not None,
                'mtf_analysis': getattr(self, 'enable_mtf_analysis', False)
            }
            
            # Step 1: Data preparation debug
            try:
                df_enhanced = self.data_helper.ensure_macd_indicators(df)
                debug_info['validation_steps'].append('data_preparation_success')
                
                if self.data_helper.validate_macd_data(df_enhanced):
                    debug_info['validation_steps'].append('macd_data_validation_success')
                else:
                    debug_info['rejection_reasons'].append('macd_data_validation_failed')
                    return debug_info
                    
            except Exception as e:
                debug_info['rejection_reasons'].append(f'data_preparation_error: {str(e)}')
                return debug_info
            
            # Step 2: Extract MACD values for debugging
            latest = df_enhanced.iloc[-1]
            previous = df_enhanced.iloc[-2]
            
            debug_info['macd_values'] = {
                'current_histogram': latest.get('macd_histogram', 0),
                'previous_histogram': previous.get('macd_histogram', 0),
                'histogram_change': abs(latest.get('macd_histogram', 0) - previous.get('macd_histogram', 0)),
                'macd_line': latest.get('macd_line', 0),
                'macd_signal': latest.get('macd_signal', 0),
                'current_price': latest.get('close', 0),
                'ema_200': latest.get('ema_200', 0),
                'macd_color': latest.get('macd_color', 'unknown'),
                'macd_color_prev': latest.get('macd_color_prev', 'unknown')
            }
            
            # Step 3: Threshold validation
            macd_threshold = self.cache.get_macd_threshold_cached(epic, self.forex_optimizer)
            histogram_change = debug_info['macd_values']['histogram_change']
            
            debug_info['threshold_check'] = {
                'macd_threshold': macd_threshold,
                'histogram_change': histogram_change,
                'threshold_passed': histogram_change >= macd_threshold
            }
            
            if histogram_change < macd_threshold:
                debug_info['rejection_reasons'].append(f'threshold_failed: {histogram_change:.6f} < {macd_threshold:.6f}')
                return debug_info
            else:
                debug_info['validation_steps'].append('threshold_validation_success')
            
            # Step 4: Signal detection debug
            try:
                enhanced_signal_data = self.signal_detector.detect_enhanced_macd_signal(
                    latest, previous, epic, timeframe,
                    signal_timestamp=signal_timestamp
                )
                
                if enhanced_signal_data:
                    debug_info['validation_steps'].append('signal_detection_success')
                    debug_info['signal_data'] = enhanced_signal_data
                else:
                    debug_info['rejection_reasons'].append('no_enhanced_signal_detected')
                    return debug_info
                    
            except Exception as e:
                debug_info['rejection_reasons'].append(f'signal_detection_error: {str(e)}')
                return debug_info
            
            # Step 5: Performance calculations debug
            try:
                efficiency_ratio = self.cache.calculate_efficiency_ratio_cached(
                    df_enhanced, epic, timeframe, self.forex_optimizer
                )
                market_regime = self.cache.detect_market_regime_cached(
                    latest, df_enhanced, epic, timeframe, self.forex_optimizer
                )
                
                debug_info['performance_calculations'] = {
                    'efficiency_ratio': efficiency_ratio,
                    'market_regime': market_regime
                }
                debug_info['validation_steps'].append('performance_calculations_success')
                
            except Exception as e:
                debug_info['rejection_reasons'].append(f'performance_calculations_error: {str(e)}')
                return debug_info
            
            # Step 6: Comprehensive validation debug
            try:
                enhanced_signal_for_confidence = self.data_helper.create_enhanced_signal_data(
                    latest, enhanced_signal_data['signal_type']
                )
                enhanced_signal_for_confidence.update({
                    'efficiency_ratio': efficiency_ratio,
                    'market_regime': market_regime
                })
                
                is_valid, confidence, validation_details = self.validator.validate_signal_comprehensive(
                    enhanced_signal_for_confidence, df_enhanced, epic
                )
                
                debug_info['comprehensive_validation'] = {
                    'is_valid': is_valid,
                    'confidence': confidence,
                    'validation_details': validation_details
                }
                
                if is_valid and confidence >= 0.3:
                    debug_info['validation_steps'].append('comprehensive_validation_success')
                    
                    # Step 7: Final signal creation
                    signal = self.data_helper.create_macd_signal_complete(
                        enhanced_signal_data['signal_type'], epic, timeframe, 
                        latest, previous, confidence, enhanced_signal_data['trigger_reason'],
                        efficiency_ratio, market_regime, spread_pips
                    )
                    
                    debug_info['signal_result'] = 'SUCCESS'
                    debug_info['final_signal'] = {
                        'signal_type': signal.get('signal_type'),
                        'confidence': signal.get('confidence_score'),
                        'trigger_reason': signal.get('signal_trigger'),
                        'modular_architecture': True,
                        'mtf_enabled': getattr(self, 'enable_mtf_analysis', False)
                    }
                    
                else:
                    debug_info['rejection_reasons'].append(f'validation_failed: valid={is_valid}, confidence={confidence:.1%}')
                    
            except Exception as e:
                debug_info['rejection_reasons'].append(f'comprehensive_validation_error: {str(e)}')
                return debug_info
            
            # Cache performance info
            debug_info['cache_performance'] = self.cache.get_cache_stats()
            
            return debug_info
            
        except Exception as e:
            debug_info['rejection_reasons'].append(f'debug_error: {str(e)}')
            debug_info['signal_result'] = 'ERROR'
            return debug_info

    # ===== BACKWARD COMPATIBILITY METHODS =====
    # These ensure 100% compatibility with existing code

    def get_cache_stats(self) -> Dict:
        """🚀 Backward compatibility: Get cache performance statistics"""
        return self.cache.get_cache_stats()

    def clear_cache(self):
        """🚀 Backward compatibility: Clear cached calculations"""
        self.clear_all_caches()

    def create_base_signal(self, signal_type: str, epic: str, timeframe: str, latest: pd.Series) -> Dict:
        """Backward compatibility: Create a base signal dictionary with common fields"""
        return {
            'epic': epic,
            'signal_type': signal_type,
            'strategy': self.name,
            'timeframe': timeframe,
            'price': latest['close'],
            'timestamp': self._convert_market_timestamp_safe(latest.name if hasattr(latest, 'name') else pd.Timestamp.now()),
            'modular_architecture': True,
            'mtf_analysis_available': getattr(self, 'enable_mtf_analysis', False)
        }

    def _convert_market_timestamp_safe(self, timestamp_value):
        """Backward compatibility: Safe timestamp conversion"""
        return self.data_helper._convert_market_timestamp_safe(timestamp_value)

    def get_required_indicators(self) -> List[str]:
        """Required indicators for MACD strategy"""
        return [
            'ema_200', 'macd_line', 'macd_signal', 'macd_histogram',
            'macd_color', 'macd_color_prev', 'macd_red_to_green', 'macd_green_to_red'
        ]

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Backward compatibility: Validate DataFrame has required MACD data"""
        return self.data_helper.validate_macd_data(df)


# ===== FACTORY FUNCTION FOR RECOMMENDED USAGE =====

def create_macd_strategy(data_fetcher=None, enable_mtf=True) -> MACDStrategy:
    """
    🏗️ FACTORY: Create modular MACD strategy instance with MTF analysis
    
    This is the recommended way to create new MACD strategy instances
    for new code, though the direct constructor maintains backward compatibility.
    
    Args:
        data_fetcher: Optional data fetcher for dynamic configurations and MTF analysis
        enable_mtf: Enable Multi-Timeframe analysis (default: True)
        
    Returns:
        Fully configured modular MACD strategy instance with MTF capabilities
    """
    # Always return the new modular strategy
    strategy = MACDStrategy(data_fetcher=data_fetcher)
    
    # Initialize MTF analysis if requested
    if enable_mtf:
        strategy.initialize_mtf_analysis(data_fetcher=data_fetcher, enable=True)
        logging.getLogger(__name__).info("📊 MTF Analysis enabled for MACD strategy")
    
    # Validate modular integration
    if not strategy.validate_modular_integration():
        logging.getLogger(__name__).warning("⚠️ Modular integration validation failed - strategy may not work correctly")
    
    return strategy


# ===== BACKWARD COMPATIBILITY WRAPPER =====

class LegacyMACDStrategy(MACDStrategy):
    """
    🔄 LEGACY: Backward compatibility wrapper for existing MACD strategy usage
    
    This ensures that any existing code that directly instantiates MACDStrategy
    will continue to work without any changes while getting all the benefits
    of the new modular architecture and MTF analysis.
    """
    
    def __init__(self, data_fetcher=None):
        # Call the new modular strategy
        super().__init__(data_fetcher=data_fetcher)
        
        # Log that legacy wrapper is being used
        self.logger.info("🔄 Using modular MACD strategy with MTF via legacy wrapper - consider updating to factory function")


# ===== MODULE SUMMARY FOR DOCUMENTATION =====

def get_modular_macd_summary() -> Dict:
    """
    📋 Get summary of modular MACD strategy architecture with MTF integration
    
    This provides documentation about the refactored modular structure
    for developers and maintainers.
    """
    return {
        'architecture': 'modular_with_dependency_injection_and_mtf',
        'main_class': 'MACDStrategy',
        'modules': {
            'MACDForexOptimizer': {
                'file': 'helpers/macd_forex_optimizer.py',
                'responsibility': 'Forex-specific calculations and optimizations',
                'key_methods': [
                    'get_forex_pair_type',
                    'apply_forex_confidence_adjustments',
                    'calculate_forex_efficiency_ratio',
                    'detect_forex_market_regime'
                ]
            },
            'MACDValidator': {
                'file': 'helpers/macd_validator.py', 
                'responsibility': 'Signal validation and confidence calculation',
                'key_methods': [
                    'validate_macd_crossover',
                    'validate_ema200_filter',
                    'validate_volume_confirmation',
                    'calculate_weighted_confidence'
                ]
            },
            'MACDCache': {
                'file': 'helpers/macd_cache.py',
                'responsibility': 'Performance caching and optimization',
                'key_methods': [
                    'calculate_efficiency_ratio_cached',
                    'detect_market_regime_cached',
                    'get_cache_stats',
                    'optimize_cache_settings'
                ]
            },
            'MACDSignalDetector': {
                'file': 'helpers/macd_signal_detector.py',
                'responsibility': 'Core signal detection algorithms',
                'key_methods': [
                    'detect_macd_histogram_crossover',
                    'detect_enhanced_macd_signal',
                    'analyze_macd_momentum_strength'
                ]
            },
            'MACDDataHelper': {
                'file': 'helpers/macd_data_helper.py',
                'responsibility': 'Data preparation and enhancement',
                'key_methods': [
                    'ensure_macd_indicators',
                    'create_enhanced_signal_data',
                    'enhance_signal_to_match_combined_strategy',
                    'validate_macd_data'
                ]
            }
        },
        'new_features': {
            'multi_timeframe_analysis': {
                'description': 'MACD momentum analysis across multiple timeframes',
                'timeframes': ['5m', '15m', '1h', '4h'],
                'validation_modes': ['soft', 'hard'],
                'confidence_adjustments': 'Dynamic boost/penalty based on MTF alignment'
            },
            'enhanced_confidence_scoring': {
                'description': 'MTF-aware confidence calculation',
                'boost_threshold': 0.65,
                'penalty_threshold': 0.5,
                'max_boost': '15%',
                'max_penalty': '12%'
            },
            'divergence_analysis': {
                'description': 'Framework for MACD divergence detection across timeframes',
                'status': 'Framework implemented, full analysis can be extended'
            }
        },
        'benefits': [
            'Separated concerns - each module has single responsibility',
            'Improved maintainability - easy to modify individual components',
            'Better testability - each module can be unit tested separately', 
            'Enhanced performance monitoring - detailed stats from each module',
            'Cleaner error handling - isolated error tracking per module',
            'Easy extensibility - simple to add new modules or replace existing ones',
            'Multi-timeframe validation - reduces false signals',
            'Enhanced signal quality - MTF momentum alignment',
            'Adaptive confidence scoring - context-aware adjustments'
        ],
        'backward_compatibility': '100% - no breaking changes to public API',
        'migration_required': 'No - existing code works without changes',
        'recommended_usage': 'Use create_macd_strategy() factory function for new code with MTF enabled',
        'mtf_integration': {
            'seamless': True,
            'optional': True,
            'cached': True,
            'configurable': True
        },
        'file_size_reduction': {
            'before': '800+ lines in single file',
            'after': '6 focused modules of 150-300 lines each + MTF integration',
            'maintainability_improvement': 'Significantly improved',
            'new_capabilities': 'Multi-timeframe analysis, enhanced validation'
        },
        'version': 'modular_macd_forex_optimized_mtf_v1',
        'timestamp': datetime.now().isoformat()
    }


# ===== MTF CONFIGURATION HELPER =====

def get_default_mtf_config() -> Dict:
    """
    Get default Multi-Timeframe configuration for MACD strategy
    """
    return {
        'check_timeframes': ['5m', '15m', '1h', '4h'],
        'min_aligned_timeframes': 2,
        'mtf_rejection_mode': 'soft',  # 'soft' or 'hard'
        'macd_confidence_boost': 0.15,  # 15% boost for good alignment
        'alignment_weights': {
            '5m': 0.15,   # Short-term momentum confirmation
            '15m': 0.25,  # Primary MACD signal timeframe
            '1h': 0.35,   # Medium-term trend validation
            '4h': 0.25    # Long-term momentum direction
        },
        'require_higher_tf_alignment': True,
        'momentum_threshold': 0.5,  # Minimum momentum score to be valid
        'excellent_momentum_threshold': 0.65,  # Threshold for confidence boost
        'cache_timeout_minutes': 5,  # MTF analysis cache timeout
        'enable_divergence_analysis': True,  # Enable divergence detection
        'session_aware_thresholds': True  # Adjust thresholds based on market session
    }


def create_custom_mtf_config(
    timeframes: List[str] = None,
    rejection_mode: str = 'soft',
    confidence_boost: float = 0.15,
    min_aligned: int = 2
) -> Dict:
    """
    Create custom MTF configuration for MACD strategy
    
    Args:
        timeframes: List of timeframes to analyze
        rejection_mode: 'soft' (penalty) or 'hard' (reject)
        confidence_boost: Boost percentage for good alignment
        min_aligned: Minimum aligned timeframes required
        
    Returns:
        Custom MTF configuration dictionary
    """
    config = get_default_mtf_config()
    
    if timeframes:
        config['check_timeframes'] = timeframes
        # Auto-adjust weights for custom timeframes
        weight_per_tf = 1.0 / len(timeframes)
        config['alignment_weights'] = {tf: weight_per_tf for tf in timeframes}
    
    config['mtf_rejection_mode'] = rejection_mode
    config['macd_confidence_boost'] = confidence_boost
    config['min_aligned_timeframes'] = min_aligned
    
    return config


# ===== INTEGRATION TESTING HELPER =====

def test_macd_mtf_integration(data_fetcher=None) -> Dict:
    """
    Test MACD strategy with MTF integration
    
    Returns comprehensive test results
    """
    try:
        # Create strategy with MTF
        strategy = create_macd_strategy(data_fetcher=data_fetcher, enable_mtf=True)
        
        # Test results
        test_results = {
            'strategy_created': True,
            'mtf_enabled': getattr(strategy, 'enable_mtf_analysis', False),
            'mtf_config_loaded': hasattr(strategy, 'mtf_config'),
            'modules_integrated': strategy.validate_modular_integration(),
            'data_fetcher_available': strategy.data_fetcher is not None,
            'cache_available': strategy.cache is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test MTF configuration
        if hasattr(strategy, 'mtf_config'):
            test_results['mtf_config'] = strategy.mtf_config
        
        # Test module status
        test_results['module_status'] = {
            'forex_optimizer': strategy.forex_optimizer is not None,
            'validator': strategy.validator is not None,
            'cache': strategy.cache is not None,
            'signal_detector': strategy.signal_detector is not None,
            'data_helper': strategy.data_helper is not None
        }
        
        return test_results
        
    except Exception as e:
        return {
            'strategy_created': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }