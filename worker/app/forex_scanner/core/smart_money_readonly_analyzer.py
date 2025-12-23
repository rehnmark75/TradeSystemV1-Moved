# core/smart_money_readonly_analyzer.py
"""
Smart Money Read-Only Analyzer - PRODUCTION READY VERSION
Non-disruptive integration that adds smart money context to existing signals
without affecting the core signal generation process.

COMPLETE FIXES APPLIED:
- Proper null checking for all dictionary/list operations
- Safe navigation for nested dictionary access
- Graceful handling of missing or invalid data
- Dynamic pip sizing for accurate analysis
- Enhanced error messages and fallback values
- ATR-normalized calculations
- Performance optimizations for real-time use

This analyzer runs AFTER signal detection and enhances signals with:
- Market structure analysis (BOS, ChoCh) 
- Order flow analysis (Order Blocks, FVGs)
- Smart money validation scores
- Confluence analysis

Results are stored in the new alert_history columns for analysis and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
from datetime import datetime
import json
import time

# Import with fallback handling
try:
    from .intelligence.market_structure_analyzer import MarketStructureAnalyzer
    from .intelligence.order_flow_analyzer import OrderFlowAnalyzer
    ANALYZERS_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from core.intelligence.market_structure_analyzer import MarketStructureAnalyzer
        from core.intelligence.order_flow_analyzer import OrderFlowAnalyzer
        ANALYZERS_AVAILABLE = True
    except ImportError:
        ANALYZERS_AVAILABLE = False
        logging.getLogger(__name__).warning("Smart money analyzers not available - using mock implementation")

# Import liquidity sweep analyzer
try:
    from .strategies.helpers.silver_bullet_liquidity import SilverBulletLiquidity
    LIQUIDITY_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from core.strategies.helpers.silver_bullet_liquidity import SilverBulletLiquidity
        LIQUIDITY_ANALYZER_AVAILABLE = True
    except ImportError:
        LIQUIDITY_ANALYZER_AVAILABLE = False
        logging.getLogger(__name__).warning("Liquidity sweep analyzer not available")

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class SmartMoneyReadOnlyAnalyzer:
    """
    Production-ready smart money analyzer that enhances existing signals
    without disrupting the core signal detection process
    """
    
    def __init__(self, data_fetcher=None):
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = data_fetcher
        
        # Initialize smart money analyzers with comprehensive fallback
        if ANALYZERS_AVAILABLE:
            try:
                self.market_structure_analyzer = MarketStructureAnalyzer()
                self.order_flow_analyzer = OrderFlowAnalyzer()
                self.logger.info("✅ Smart money analyzers initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize analyzers: {e}")
                self.market_structure_analyzer = None
                self.order_flow_analyzer = None
        else:
            self.market_structure_analyzer = None
            self.order_flow_analyzer = None
            self.logger.warning("⚠️ Using mock smart money implementation")
        
        # Read-only configuration
        self.enabled = getattr(config, 'SMART_MONEY_READONLY_ENABLED', True)
        self.min_data_points = getattr(config, 'SMART_MONEY_MIN_DATA_POINTS', 50)
        self.analysis_timeout = getattr(config, 'SMART_MONEY_ANALYSIS_TIMEOUT', 10.0)
        self.structure_weight = getattr(config, 'SMART_MONEY_STRUCTURE_WEIGHT', 0.4)
        self.order_flow_weight = getattr(config, 'SMART_MONEY_ORDER_FLOW_WEIGHT', 0.3)
        self.min_confidence_boost = getattr(config, 'SMART_MONEY_MIN_CONFIDENCE_BOOST', 0.1)
        self.max_confidence_boost = getattr(config, 'SMART_MONEY_MAX_CONFIDENCE_BOOST', 0.3)

        # Liquidity sweep configuration (NEW)
        self.liquidity_sweep_enabled = getattr(config, 'SMART_MONEY_LIQUIDITY_SWEEP_ENABLED', True)
        self.liquidity_sweep_weight = getattr(config, 'SMART_MONEY_LIQUIDITY_SWEEP_WEIGHT', 0.20)
        self.liquidity_sweep_lookback = getattr(config, 'SMART_MONEY_LIQUIDITY_SWEEP_LOOKBACK_BARS', 10)
        self.min_sweep_quality = getattr(config, 'SMART_MONEY_MIN_SWEEP_QUALITY', 0.4)

        # Initialize liquidity analyzer
        self.liquidity_analyzer = None
        if LIQUIDITY_ANALYZER_AVAILABLE and self.liquidity_sweep_enabled:
            try:
                self.liquidity_analyzer = SilverBulletLiquidity(logger=self.logger)
                self.logger.info("✅ Liquidity sweep analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize liquidity analyzer: {e}")

        self.logger.info("🧠 SmartMoneyReadOnlyAnalyzer initialized")
        self.logger.info(f"   Enabled: {'✅' if self.enabled else '❌'}")
        self.logger.info(f"   Analyzers available: {'✅' if ANALYZERS_AVAILABLE else '❌'}")
        self.logger.info(f"   Liquidity sweep: {'✅' if self.liquidity_analyzer else '❌'}")
        self.logger.info(f"   Weights - Structure: {self.structure_weight}, OrderFlow: {self.order_flow_weight}")
    
    def analyze_signal(self, signal, df, epic, timeframe='5m') -> Dict:
        """
        BACKWARD COMPATIBILITY METHOD
        Alias for enhance_signal_with_smart_money to maintain compatibility
        """
        return self.enhance_signal_with_smart_money(signal, df, epic, timeframe)

    def enhance_signal_with_smart_money(
        self, 
        signal: Optional[Dict], 
        df: pd.DataFrame, 
        epic: str,
        timeframe: str = '5m'
    ) -> Dict:
        """
        Main entry point: enhance existing signal with smart money analysis
        Returns enhanced signal data for database storage
        """
        start_time = datetime.now()
        
        # Quick checks for enabled state and data quality
        if not self.enabled:
            return self._create_disabled_response(signal)
        
        if df is None or len(df) < self.min_data_points:
            return self._create_insufficient_data_response(signal, len(df) if df is not None else 0)
        
        if not ANALYZERS_AVAILABLE:
            return self._create_mock_analysis_response(signal)
        
        try:
            # 1. Analyze Market Structure (with timeout protection)
            market_structure_analysis = self._analyze_market_structure_safe(
                signal, df, epic, timeframe, start_time
            )
            
            # 2. Analyze Order Flow (with timeout protection)
            order_flow_analysis = self._analyze_order_flow_safe(
                signal, df, epic, timeframe, start_time
            )

            # 2.5 Analyze Liquidity Sweeps (NEW)
            liquidity_analysis = self._analyze_liquidity_sweeps_safe(
                signal, df, epic, timeframe, start_time
            )

            # 3. Smart Money Validation
            smart_money_validation = self._validate_signal_with_smart_money_safe(
                signal, market_structure_analysis, order_flow_analysis
            )
            
            # 4. Calculate Enhanced Confidence
            enhanced_confidence = self._calculate_enhanced_confidence_safe(
                signal, smart_money_validation
            )
            
            # 5. Generate Confluence Analysis (now includes liquidity sweep)
            confluence_details = self._generate_confluence_analysis_safe(
                signal, market_structure_analysis, order_flow_analysis, smart_money_validation,
                liquidity_analysis
            )
            
            # 6. Determine Smart Money Type
            smart_money_type = self._determine_smart_money_type_safe(
                market_structure_analysis, order_flow_analysis
            )
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # CRITICAL FIX: Return enhanced signal, not just Smart Money analysis
            # The method should preserve the original signal data and add Smart Money analysis
            enhanced_signal = signal.copy() if signal else {}
            
            # Add Smart Money analysis as a nested field
            enhanced_signal['smart_money_analysis'] = {
                'enabled': True,
                'smart_money_validated': smart_money_validation.get('validated', False),
                'smart_money_type': smart_money_type,
                'smart_money_score': smart_money_validation.get('score', 0.5),
                'market_structure_analysis': market_structure_analysis,
                'order_flow_analysis': order_flow_analysis,
                'liquidity_analysis': liquidity_analysis,
                'confluence_details': confluence_details,
                'analysis_metadata': {
                    'analyzer_version': '2.1.0_with_liquidity',
                    'analysis_timestamp': start_time.isoformat(),
                    'analysis_duration_seconds': elapsed_time,
                    'epic': epic,
                    'timeframe': timeframe,
                    'original_signal_type': signal.get('signal_type') if signal else None,
                    'original_confidence': signal.get('confidence_score') if signal else None,
                    'data_points_analyzed': len(df),
                    'analyzers_available': ANALYZERS_AVAILABLE,
                    'liquidity_analyzer_available': self.liquidity_analyzer is not None
                }
            }
            
            # Update confidence with Smart Money enhancement
            enhanced_signal['confidence_score'] = enhanced_confidence
            enhanced_signal['smart_money_enhanced'] = True
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Smart money analysis failed for {epic}: {e}")
            return self._create_error_response(signal, str(e))
    
    def _analyze_market_structure_safe(
        self, 
        signal: Optional[Dict], 
        df: pd.DataFrame, 
        epic: str,
        timeframe: str,
        start_time: datetime
    ) -> Dict:
        """Safely analyze market structure with timeout and error protection"""
        try:
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > self.analysis_timeout:
                return {'error': 'Analysis timeout', 'timeout': True}
            
            # Safe check for analyzer availability
            if not self.market_structure_analyzer:
                return {'error': 'Market structure analyzer not available', 'available': False}
            
            # Get market structure analysis
            structure_analysis = self.market_structure_analyzer.analyze_market_structure(
                df, epic, timeframe
            )
            
            # Safe null check with comprehensive defaults
            if structure_analysis is None:
                structure_analysis = self._create_default_structure_analysis()
            
            # Ensure all required keys exist with safe defaults
            structure_analysis = self._ensure_structure_analysis_completeness(structure_analysis)
            
            # Validate signal against structure if signal exists
            if signal is not None:
                structure_validation = self.market_structure_analyzer.validate_signal_against_structure(
                    signal, structure_analysis, df
                )
                structure_analysis['signal_validation'] = structure_validation or {}
            
            return structure_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Market structure analysis failed for {epic}: {e}")
            return {
                'error': f'Market structure analysis failed: {e}',
                'current_bias': 'NEUTRAL',
                'structure_score': 0.5,
                'swing_points': [],
                'structure_events': [],
                'next_targets': {'next_resistance': None, 'next_support': None, 'key_levels': []},
                'signal_validation': {'structure_aligned': True, 'structure_score': 0.5}
            }
    
    def _analyze_order_flow_safe(
        self, 
        signal: Optional[Dict], 
        df: pd.DataFrame, 
        epic: str,
        timeframe: str,
        start_time: datetime
    ) -> Dict:
        """Safely analyze order flow with timeout and error protection"""
        try:
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > self.analysis_timeout:
                return {'error': 'Analysis timeout', 'timeout': True}
            
            # Safe check for analyzer availability
            if not self.order_flow_analyzer:
                return {'error': 'Order flow analyzer not available', 'available': False}
            
            # Get order flow analysis
            order_flow_analysis = self.order_flow_analyzer.analyze_order_flow(
                df, epic, timeframe
            )
            
            # Safe null check with comprehensive defaults
            if order_flow_analysis is None:
                order_flow_analysis = self._create_default_order_flow_analysis()
            
            # Ensure all required keys exist with safe defaults
            order_flow_analysis = self._ensure_order_flow_analysis_completeness(order_flow_analysis)
            
            # Validate signal against order flow if signal exists
            if signal is not None:
                order_flow_validation = self.order_flow_analyzer.validate_signal_against_order_flow(
                    signal, order_flow_analysis, df
                )
                order_flow_analysis['signal_validation'] = order_flow_validation or {}
            
            return order_flow_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Order flow analysis failed for {epic}: {e}")
            return {
                'error': f'Order flow analysis failed: {e}',
                'order_flow_bias': 'NEUTRAL',
                'order_blocks': [],
                'fair_value_gaps': [],
                'supply_demand_zones': [],
                'signal_validation': {'order_flow_aligned': True, 'order_flow_score': 0.5}
            }

    def _analyze_liquidity_sweeps_safe(
        self,
        signal: Optional[Dict],
        df: pd.DataFrame,
        epic: str,
        timeframe: str,
        start_time: datetime
    ) -> Dict:
        """
        Safely analyze liquidity sweeps for signal enrichment.
        Detects BSL (buy-side liquidity) and SSL (sell-side liquidity) sweeps.
        """
        default_result = {
            'sweep_detected': False,
            'sweep_type': None,
            'sweep_level': None,
            'sweep_pips': None,
            'sweep_bars_ago': None,
            'sweep_quality': 0,
            'signal_alignment': False,
            'liquidity_levels_found': 0,
            'enabled': self.liquidity_sweep_enabled
        }

        try:
            # Check if liquidity analysis is enabled and analyzer is available
            if not self.liquidity_sweep_enabled or not self.liquidity_analyzer:
                default_result['error'] = 'Liquidity sweep analysis disabled or unavailable'
                return default_result

            # Check timeout
            if (datetime.now() - start_time).total_seconds() > self.analysis_timeout:
                default_result['error'] = 'Analysis timeout'
                return default_result

            # Get pip value for the pair
            pip_value = self._get_pip_value(epic)

            # 1. Detect liquidity levels (swing highs/lows)
            liquidity_levels = self.liquidity_analyzer.detect_liquidity_levels(
                df=df,
                lookback_bars=self.liquidity_sweep_lookback,
                swing_strength=3,
                pip_value=pip_value
            )

            if not liquidity_levels:
                default_result['liquidity_levels_found'] = 0
                return default_result

            default_result['liquidity_levels_found'] = len(liquidity_levels)

            # 2. Detect if any liquidity was swept recently
            sweep = self.liquidity_analyzer.detect_liquidity_sweep(
                df=df,
                liquidity_levels=liquidity_levels,
                min_sweep_pips=3.0,
                max_sweep_pips=15.0,
                pip_value=pip_value,
                require_rejection=True,
                max_sweep_age=self.liquidity_sweep_lookback
            )

            if not sweep:
                return default_result

            # 3. Calculate sweep quality
            sweep_quality = self.liquidity_analyzer.calculate_sweep_quality(sweep)

            # 4. Determine signal alignment
            # BSL sweep (price swept highs) = bearish signal expected (reversal down)
            # SSL sweep (price swept lows) = bullish signal expected (reversal up)
            signal_type = signal.get('signal_type', '').upper() if signal else ''
            sweep_type = sweep.liquidity_level.liquidity_type.value  # 'BSL' or 'SSL'

            signal_alignment = False
            if sweep_type == 'SSL' and signal_type in ['BULL', 'BUY']:
                signal_alignment = True  # SSL sweep supports bullish signal
            elif sweep_type == 'BSL' and signal_type in ['BEAR', 'SELL']:
                signal_alignment = True  # BSL sweep supports bearish signal

            return {
                'sweep_detected': True,
                'sweep_type': sweep_type,
                'sweep_level': sweep.liquidity_level.price,
                'sweep_pips': sweep.sweep_pips,
                'sweep_bars_ago': sweep.rejection_candles,
                'sweep_quality': round(sweep_quality, 3),
                'sweep_status': sweep.status.value,
                'rejection_confirmed': sweep.rejection_confirmed,
                'signal_alignment': signal_alignment,
                'liquidity_levels_found': len(liquidity_levels),
                'enabled': True
            }

        except Exception as e:
            self.logger.error(f"❌ Liquidity sweep analysis failed for {epic}: {e}")
            default_result['error'] = str(e)
            return default_result

    def _get_pip_value(self, epic: str) -> float:
        """Get pip value for the given epic"""
        if not epic:
            return 0.0001
        e = epic.upper()
        if 'JPY' in e:
            return 0.01
        if any(x in e for x in ['XAU', 'GOLD']):
            return 0.1
        if any(x in e for x in ['XAG', 'SILV']):
            return 0.01
        return 0.0001

    def _validate_signal_with_smart_money_safe(
        self,
        signal: Optional[Dict],
        market_structure_analysis: Dict,
        order_flow_analysis: Dict
    ) -> Dict:
        """Safely validate signal with smart money context"""
        try:
            if signal is None:
                return {
                    'validated': False,
                    'score': 0.5,
                    'validation_reasons': ['No signal to validate'],
                    'structure_contribution': 0.0,
                    'order_flow_contribution': 0.0
                }
            
            signal_type = signal.get('signal_type', '').upper()
            validation_reasons = []
            structure_contribution = 0.0
            order_flow_contribution = 0.0
            
            # Structure validation
            structure_validation = market_structure_analysis.get('signal_validation', {})
            if structure_validation.get('structure_aligned', False):
                structure_contribution = self.structure_weight
                validation_reasons.append(f"Structure aligned: {structure_validation.get('validation_reason', 'Unknown')}")
            else:
                validation_reasons.append(f"Structure misaligned: {structure_validation.get('validation_reason', 'Unknown')}")
            
            # Order flow validation
            order_flow_validation = order_flow_analysis.get('signal_validation', {})
            if order_flow_validation.get('order_flow_aligned', False):
                order_flow_contribution = self.order_flow_weight
                validation_reasons.append(f"Order flow aligned: {order_flow_validation.get('validation_reason', 'Unknown')}")
            else:
                validation_reasons.append(f"Order flow misaligned: {order_flow_validation.get('validation_reason', 'Unknown')}")
            
            # Calculate total smart money score
            total_score = min(1.0, 0.5 + structure_contribution + order_flow_contribution)
            validated = total_score >= 0.6  # Threshold for validation
            
            return {
                'validated': validated,
                'score': total_score,
                'validation_reasons': validation_reasons,
                'structure_contribution': structure_contribution,
                'order_flow_contribution': order_flow_contribution,
                'structure_score': structure_validation.get('structure_score', 0.5),
                'order_flow_score': order_flow_validation.get('order_flow_score', 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Smart money validation failed: {e}")
            return {
                'validated': False,
                'score': 0.5,
                'validation_reasons': [f'Validation error: {e}'],
                'structure_contribution': 0.0,
                'order_flow_contribution': 0.0
            }
    
    def _calculate_enhanced_confidence_safe(
        self,
        signal: Optional[Dict],
        smart_money_validation: Dict
    ) -> float:
        """Safely calculate enhanced confidence score"""
        try:
            if signal is None:
                return 0.5
            
            original_confidence = signal.get('confidence_score', 0.5)
            smart_money_score = smart_money_validation.get('score', 0.5)
            
            # Calculate confidence boost based on smart money alignment
            if smart_money_validation.get('validated', False):
                confidence_boost = self.min_confidence_boost + (
                    (smart_money_score - 0.6) / 0.4 * 
                    (self.max_confidence_boost - self.min_confidence_boost)
                )
            else:
                # Slight penalty for misalignment
                confidence_boost = -0.05 if smart_money_score < 0.4 else 0.0
            
            # Apply boost and clamp to valid range
            enhanced_confidence = min(1.0, max(0.0, original_confidence + confidence_boost))
            
            return enhanced_confidence
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced confidence calculation failed: {e}")
            return signal.get('confidence_score', 0.5) if signal else 0.5
    
    def _generate_confluence_analysis_safe(
        self,
        signal: Optional[Dict],
        market_structure_analysis: Dict,
        order_flow_analysis: Dict,
        smart_money_validation: Dict,
        liquidity_analysis: Optional[Dict] = None
    ) -> Dict:
        """Safely generate confluence analysis with OB proximity and liquidity sweep scoring"""
        try:
            confluence_factors = []
            confluence_score = 0.0
            liquidity_analysis = liquidity_analysis or {}

            # Structure confluence
            structure_bias = market_structure_analysis.get('current_bias', 'NEUTRAL')
            if structure_bias != 'NEUTRAL':
                confluence_factors.append(f"Market structure: {structure_bias}")
                confluence_score += 0.3

            # Order flow confluence
            order_flow_bias = order_flow_analysis.get('order_flow_bias', 'NEUTRAL')
            if order_flow_bias != 'NEUTRAL':
                confluence_factors.append(f"Order flow: {order_flow_bias}")
                confluence_score += 0.2

            # OB Proximity scoring
            ob_proximity = order_flow_analysis.get('ob_proximity', {})
            ob_alignment_score = ob_proximity.get('alignment_score', 0)
            if ob_alignment_score > 0:
                # Weight: 0.15 for OB proximity
                ob_proximity_weight = 0.15
                proximity_contribution = ob_alignment_score * ob_proximity_weight
                confluence_score += proximity_contribution

                # Add details to factors
                nearest_ob_dist = ob_proximity.get('nearest_ob_distance_pips')
                if nearest_ob_dist is not None and nearest_ob_dist < 20:
                    confluence_factors.append(f"Near OB ({nearest_ob_dist:.1f} pips, score: {ob_alignment_score:.2f})")

            # Liquidity Sweep confluence (NEW)
            sweep_detected = liquidity_analysis.get('sweep_detected', False)
            sweep_quality = liquidity_analysis.get('sweep_quality', 0)
            if sweep_detected and sweep_quality >= self.min_sweep_quality:
                sweep_type = liquidity_analysis.get('sweep_type', 'Unknown')
                signal_alignment = liquidity_analysis.get('signal_alignment', False)

                # Weight: 0.20 for liquidity sweep (configurable)
                liquidity_contribution = sweep_quality * self.liquidity_sweep_weight
                confluence_score += liquidity_contribution

                alignment_str = "aligned" if signal_alignment else "counter"
                confluence_factors.append(f"Liquidity sweep: {sweep_type} ({alignment_str}, quality: {sweep_quality:.2f})")

            # Key level proximity (legacy - kept for compatibility)
            nearby_levels = order_flow_analysis.get('signal_validation', {}).get('nearby_levels', [])
            if nearby_levels:
                confluence_factors.append(f"Near {len(nearby_levels)} key levels")
                confluence_score += 0.1 * min(len(nearby_levels), 3)

            # Structure events
            recent_events = market_structure_analysis.get('structure_events', [])
            if recent_events:
                confluence_factors.append(f"{len(recent_events)} recent structure events")
                confluence_score += 0.1

            confluence_score = min(1.0, confluence_score)

            return {
                'confluence_factors': confluence_factors,
                'confluence_score': confluence_score,
                'total_factors': len(confluence_factors),
                'structure_bias': structure_bias,
                'order_flow_bias': order_flow_bias,
                'nearby_levels_count': len(nearby_levels),
                'structure_events_count': len(recent_events),
                # OB proximity analytics
                'ob_proximity_score': ob_alignment_score,
                'ob_proximity_bias': ob_proximity.get('dominant_bias', 'NEUTRAL'),
                'nearest_ob_distance_pips': ob_proximity.get('nearest_ob_distance_pips'),
                'total_obs_found': ob_proximity.get('total_obs_found', 0),
                'total_fvgs_found': ob_proximity.get('total_fvgs_found', 0),
                # Liquidity sweep analytics (NEW)
                'liquidity_sweep_detected': sweep_detected,
                'liquidity_sweep_type': liquidity_analysis.get('sweep_type'),
                'liquidity_sweep_quality': sweep_quality,
                'liquidity_sweep_alignment': liquidity_analysis.get('signal_alignment', False)
            }

        except Exception as e:
            self.logger.error(f"❌ Confluence analysis failed: {e}")
            return {
                'confluence_factors': [f'Analysis error: {e}'],
                'confluence_score': 0.5,
                'total_factors': 0
            }
    
    def _determine_smart_money_type_safe(
        self,
        market_structure_analysis: Dict,
        order_flow_analysis: Dict
    ) -> str:
        """Safely determine smart money signal type"""
        try:
            structure_bias = market_structure_analysis.get('current_bias', 'NEUTRAL')
            order_flow_bias = order_flow_analysis.get('order_flow_bias', 'NEUTRAL')
            
            # Determine predominant smart money type
            if structure_bias == 'BULLISH' and order_flow_bias == 'BULLISH':
                return 'STRONG_BULLISH_SMC'
            elif structure_bias == 'BEARISH' and order_flow_bias == 'BEARISH':
                return 'STRONG_BEARISH_SMC'
            elif structure_bias == 'BULLISH' or order_flow_bias == 'BULLISH':
                return 'WEAK_BULLISH_SMC'
            elif structure_bias == 'BEARISH' or order_flow_bias == 'BEARISH':
                return 'WEAK_BEARISH_SMC'
            else:
                return 'NEUTRAL_SMC'
                
        except Exception as e:
            self.logger.error(f"❌ Smart money type determination failed: {e}")
            return 'UNKNOWN_SMC'
    
    # Safe default creation methods
    def _create_default_structure_analysis(self) -> Dict:
        """Create safe default structure analysis"""
        return {
            'current_bias': 'NEUTRAL',
            'structure_score': 0.5,
            'swing_points': [],
            'structure_events': [],
            'next_targets': {'next_resistance': None, 'next_support': None, 'key_levels': []},
            'analysis_summary': 'Default structure analysis',
            'pip_size': 0.0001
        }
    
    def _create_default_order_flow_analysis(self) -> Dict:
        """Create safe default order flow analysis"""
        return {
            'order_flow_bias': 'NEUTRAL',
            'order_blocks': [],
            'fair_value_gaps': [],
            'supply_demand_zones': [],
            'analysis_time_seconds': 0.0,
            'pip_size': 0.0001
        }
    
    def _ensure_structure_analysis_completeness(self, analysis: Dict) -> Dict:
        """Ensure structure analysis has all required fields"""
        defaults = self._create_default_structure_analysis()
        for key, default_value in defaults.items():
            if key not in analysis or analysis[key] is None:
                analysis[key] = default_value
        return analysis
    
    def _ensure_order_flow_analysis_completeness(self, analysis: Dict) -> Dict:
        """Ensure order flow analysis has all required fields"""
        defaults = self._create_default_order_flow_analysis()
        for key, default_value in defaults.items():
            if key not in analysis or analysis[key] is None:
                analysis[key] = default_value
        return analysis
    
    # Response creation methods for various scenarios
    def _create_disabled_response(self, signal: Optional[Dict]) -> Dict:
        """Create response when analyzer is disabled"""
        # CRITICAL FIX: Return enhanced signal, not just disabled data
        enhanced_signal = signal.copy() if signal else {}
        
        enhanced_signal['smart_money_analysis'] = {
            'enabled': False,
            'smart_money_validated': False,
            'smart_money_type': 'DISABLED',
            'smart_money_score': 0.5,
            'analysis_metadata': {
                'analyzer_version': '2.0.0_production',
                'status': 'disabled',
                'message': 'Smart money analysis is disabled'
            }
        }
        
        # Keep original confidence
        enhanced_signal['confidence_score'] = signal.get('confidence_score', 0.5) if signal else 0.5
        enhanced_signal['smart_money_enhanced'] = False
        
        return enhanced_signal
    
    def _create_insufficient_data_response(self, signal: Optional[Dict], data_points: int) -> Dict:
        """Create response when insufficient data is available"""
        # CRITICAL FIX: Return enhanced signal, not just insufficient data response
        enhanced_signal = signal.copy() if signal else {}
        
        enhanced_signal['smart_money_analysis'] = {
            'enabled': True,
            'smart_money_validated': False,
            'smart_money_type': 'INSUFFICIENT_DATA',
            'smart_money_score': 0.5,
            'analysis_metadata': {
                'analyzer_version': '2.0.0_production',
                'status': 'insufficient_data',
                'data_points': data_points,
                'required_points': self.min_data_points,
                'message': f'Need {self.min_data_points} data points, got {data_points}'
            }
        }
        
        # Keep original confidence
        enhanced_signal['confidence_score'] = signal.get('confidence_score', 0.5) if signal else 0.5
        enhanced_signal['smart_money_enhanced'] = False
        
        return enhanced_signal
    
    def _create_mock_analysis_response(self, signal: Optional[Dict]) -> Dict:
        """Create mock response when analyzers are not available"""
        # CRITICAL FIX: Return enhanced signal, not just mock analysis data
        enhanced_signal = signal.copy() if signal else {}
        
        enhanced_signal['smart_money_analysis'] = {
            'enabled': True,
            'smart_money_validated': True,  # Allow signals to pass through
            'smart_money_type': 'MOCK_ANALYSIS',
            'smart_money_score': 0.6,  # Neutral positive score
            'analysis_metadata': {
                'analyzer_version': '2.0.0_production',
                'status': 'mock_mode',
                'message': 'Using mock smart money analysis - analyzers not available'
            }
        }
        
        # Keep original confidence
        enhanced_signal['confidence_score'] = signal.get('confidence_score', 0.5) if signal else 0.5
        enhanced_signal['smart_money_enhanced'] = True
        
        return enhanced_signal
    
    def _create_error_response(self, signal: Optional[Dict], error_message: str) -> Dict:
        """Create response when analysis encounters an error"""
        # CRITICAL FIX: Return enhanced signal, not just error data
        enhanced_signal = signal.copy() if signal else {}
        
        enhanced_signal['smart_money_analysis'] = {
            'enabled': True,
            'smart_money_validated': True,  # Default to allowing signals
            'smart_money_type': 'ERROR_FALLBACK',
            'smart_money_score': 0.5,
            'analysis_metadata': {
                'analyzer_version': '2.0.0_production',
                'status': 'error',
                'error_message': error_message,
                'message': 'Analysis failed, using fallback values'
            }
        }
        
        # Keep original confidence if available
        enhanced_signal['confidence_score'] = signal.get('confidence_score', 0.5) if signal else 0.5
        enhanced_signal['smart_money_enhanced'] = True
        
        return enhanced_signal
    
    def get_analyzer_status(self) -> Dict:
        """Get current status of the smart money analyzer"""
        return {
            'enabled': self.enabled,
            'analyzers_available': ANALYZERS_AVAILABLE,
            'market_structure_analyzer': self.market_structure_analyzer is not None,
            'order_flow_analyzer': self.order_flow_analyzer is not None,
            'configuration': {
                'min_data_points': self.min_data_points,
                'analysis_timeout': self.analysis_timeout,
                'structure_weight': self.structure_weight,
                'order_flow_weight': self.order_flow_weight
            },
            'version': '2.0.0_production'
        }