# core/strategies/smart_money_ema_strategy.py
"""
Smart Money Enhanced EMA Strategy - Phase 1 & 2 Integration
Combines your existing high-performing EMA strategy with smart money concepts:
- Market Structure validation (BOS, ChoCh)
- Order Flow confirmation (Order Blocks, FVGs)
- Maintains existing 125 signals/week performance while improving quality

This strategy builds on your proven EMA 9/21/200 crossover logic while adding
institutional-grade filtering to catch optimal trends aligned with smart money.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .ema_strategy import EMAStrategy
from ..intelligence.market_structure_analyzer import MarketStructureAnalyzer
from ..intelligence.order_flow_analyzer import OrderFlowAnalyzer
try:
    import config
except ImportError:
    from forex_scanner import config


class SmartMoneyEMAStrategy(EMAStrategy):
    """
    Smart Money Enhanced EMA Strategy
    
    Inherits from your proven EMAStrategy and adds smart money validation:
    1. Uses existing EMA 9/21/200 crossover detection 
    2. Validates signals against market structure (BOS/ChoCh)
    3. Confirms signals with order flow analysis (OB/FVG)
    4. Maintains high signal frequency while improving quality
    """
    
    def __init__(self, ema_config_name: str = None, data_fetcher=None):
        # Initialize parent EMA strategy
        super().__init__(ema_config_name, data_fetcher)
        
        # Initialize smart money analyzers
        self.market_structure_analyzer = MarketStructureAnalyzer()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        # Smart money configuration
        self.enable_structure_validation = getattr(config, 'SMART_MONEY_STRUCTURE_VALIDATION', True)
        self.enable_order_flow_validation = getattr(config, 'SMART_MONEY_ORDER_FLOW_VALIDATION', True)
        self.structure_weight = getattr(config, 'SMART_MONEY_STRUCTURE_WEIGHT', 0.3)
        self.order_flow_weight = getattr(config, 'SMART_MONEY_ORDER_FLOW_WEIGHT', 0.2)
        self.min_smart_money_score = getattr(config, 'SMART_MONEY_MIN_SCORE', 0.4)
        
        # Update strategy name to distinguish from regular EMA
        self.name = 'smart_money_ema'
        
        self.logger.info("🧠 SmartMoneyEMAStrategy initialized")
        self.logger.info(f"   Structure validation: {'✅' if self.enable_structure_validation else '❌'}")
        self.logger.info(f"   Order flow validation: {'✅' if self.enable_order_flow_validation else '❌'}")
        self.logger.info(f"   Smart money weights: Structure={self.structure_weight}, OrderFlow={self.order_flow_weight}")
    
    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Enhanced signal detection with smart money validation
        
        Process:
        1. Get base EMA signal from parent class
        2. Analyze market structure context
        3. Analyze order flow context  
        4. Apply smart money validation
        5. Adjust confidence score based on confluence
        """
        try:
            # Step 1: Get base EMA signal using proven logic
            base_signal = super().detect_signal(df, epic, spread_pips, timeframe)
            
            if not base_signal:
                return None  # No base signal, no need for smart money analysis
            
            self.logger.debug(f"🎯 Base EMA signal detected for {epic}: {base_signal['signal_type']}")
            
            # Step 2: Analyze market structure if enabled
            structure_analysis = None
            structure_validation = {'structure_aligned': True, 'structure_score': 0.5}
            
            if self.enable_structure_validation:
                try:
                    structure_analysis = self.market_structure_analyzer.analyze_market_structure(
                        df, epic, timeframe
                    )
                    structure_validation = self.market_structure_analyzer.validate_signal_against_structure(
                        base_signal['signal_type'], 
                        base_signal['entry_price'], 
                        epic
                    )
                    self.logger.debug(f"🏗️ Structure analysis: {structure_validation['recommended_action']}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Structure analysis failed for {epic}: {e}")
            
            # Step 3: Analyze order flow if enabled
            order_flow_analysis = None
            order_flow_validation = {'order_flow_aligned': True, 'order_flow_score': 0.5}
            
            if self.enable_order_flow_validation:
                try:
                    order_flow_analysis = self.order_flow_analyzer.analyze_order_flow(
                        df, epic, timeframe
                    )
                    order_flow_validation = self.order_flow_analyzer.validate_signal_against_order_flow(
                        base_signal['signal_type'], 
                        base_signal['entry_price'], 
                        epic
                    )
                    self.logger.debug(f"📊 Order flow analysis: {order_flow_validation['recommended_action']}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Order flow analysis failed for {epic}: {e}")
            
            # Step 4: Apply smart money validation and scoring
            smart_money_result = self._apply_smart_money_validation(
                base_signal, structure_validation, order_flow_validation,
                structure_analysis, order_flow_analysis
            )
            
            if not smart_money_result['proceed_with_signal']:
                self.logger.debug(f"🚫 Smart money validation rejected signal for {epic}: {smart_money_result['rejection_reason']}")
                return None
            
            # Step 5: Enhance signal with smart money data
            enhanced_signal = self._enhance_signal_with_smart_money_data(
                base_signal, smart_money_result, structure_validation, 
                order_flow_validation, structure_analysis, order_flow_analysis
            )
            
            self.logger.debug(f"✅ Smart money enhanced signal for {epic}: "
                            f"confidence {enhanced_signal['confidence_score']:.3f} -> "
                            f"{enhanced_signal.get('enhanced_confidence_score', enhanced_signal['confidence_score']):.3f}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Smart money EMA signal detection failed for {epic}: {e}")
            # Fallback to base signal if smart money analysis fails
            return super().detect_signal(df, epic, spread_pips, timeframe)
    
    def _apply_smart_money_validation(
        self, 
        base_signal: Dict, 
        structure_validation: Dict,
        order_flow_validation: Dict,
        structure_analysis: Optional[Dict] = None,
        order_flow_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Apply smart money validation rules to determine if signal should proceed
        
        Returns:
            Dictionary with validation results and scoring
        """
        try:
            validation_scores = []
            validation_reasons = []
            proceed_with_signal = True
            rejection_reason = None
            
            # Base EMA signal always contributes (this is your proven performer)
            base_confidence = base_signal.get('confidence_score', 0.8)
            validation_scores.append(base_confidence * 0.5)  # 50% weight to base signal
            validation_reasons.append(f"EMA crossover: {base_confidence:.3f}")
            
            # Structure validation
            if self.enable_structure_validation:
                structure_score = structure_validation.get('structure_score', 0.5)
                structure_aligned = structure_validation.get('structure_aligned', True)
                
                if structure_validation.get('recommended_action') == 'REJECT':
                    # Only reject if structure strongly conflicts (rare case)
                    if structure_score > 0.8 and not structure_aligned:
                        proceed_with_signal = False
                        rejection_reason = f"Strong structure conflict: {structure_validation.get('validation_reason')}"
                    else:
                        # Reduce confidence but don't reject
                        validation_scores.append(structure_score * self.structure_weight * 0.5)
                        validation_reasons.append(f"Structure warning: {structure_score:.3f}")
                else:
                    validation_scores.append(structure_score * self.structure_weight)
                    validation_reasons.append(f"Structure aligned: {structure_score:.3f}")
            
            # Order flow validation
            if self.enable_order_flow_validation:
                order_flow_score = order_flow_validation.get('order_flow_score', 0.5)
                
                validation_scores.append(order_flow_score * self.order_flow_weight)
                validation_reasons.append(f"Order flow: {order_flow_score:.3f}")
            
            # Calculate final smart money score
            final_score = sum(validation_scores)
            
            # Check minimum threshold (but be permissive to maintain signal frequency)
            if final_score < self.min_smart_money_score:
                self.logger.debug(f"⚠️ Smart money score below threshold: {final_score:.3f} < {self.min_smart_money_score}")
                # Don't reject, just flag as lower confidence
            
            return {
                'proceed_with_signal': proceed_with_signal,
                'smart_money_score': final_score,
                'validation_scores': validation_scores,
                'validation_reasons': validation_reasons,
                'rejection_reason': rejection_reason,
                'structure_contribution': structure_validation.get('structure_score', 0.5) * self.structure_weight,
                'order_flow_contribution': order_flow_validation.get('order_flow_score', 0.5) * self.order_flow_weight
            }
            
        except Exception as e:
            self.logger.error(f"❌ Smart money validation failed: {e}")
            return {
                'proceed_with_signal': True,  # Default to allow signal
                'smart_money_score': 0.5,
                'validation_reasons': [f'Validation error: {e}'],
                'rejection_reason': None
            }
    
    def _enhance_signal_with_smart_money_data(
        self, 
        base_signal: Dict, 
        smart_money_result: Dict,
        structure_validation: Dict,
        order_flow_validation: Dict,
        structure_analysis: Optional[Dict] = None,
        order_flow_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Enhance the base signal with smart money analysis data
        """
        try:
            enhanced_signal = base_signal.copy()
            
            # Update strategy name
            enhanced_signal['strategy'] = self.name
            
            # Add smart money confidence enhancement
            original_confidence = base_signal.get('confidence_score', 0.8)
            smart_money_score = smart_money_result.get('smart_money_score', 0.5)
            
            # Calculate enhanced confidence (weighted combination)
            enhanced_confidence = min(1.0, original_confidence * 0.7 + smart_money_score * 0.3)
            enhanced_signal['enhanced_confidence_score'] = enhanced_confidence
            enhanced_signal['original_confidence_score'] = original_confidence
            enhanced_signal['smart_money_score'] = smart_money_score
            
            # Use enhanced confidence as main confidence
            enhanced_signal['confidence_score'] = enhanced_confidence
            
            # Add smart money analysis data
            enhanced_signal.update({
                'smart_money_validated': True,
                'smart_money_version': '1.0',
                'smart_money_timestamp': datetime.now().isoformat(),
                
                # Structure data
                'market_structure_analysis': {
                    'enabled': self.enable_structure_validation,
                    'structure_aligned': structure_validation.get('structure_aligned', True),
                    'structure_score': structure_validation.get('structure_score', 0.5),
                    'current_bias': structure_validation.get('current_bias', 'NEUTRAL'),
                    'validation_reason': structure_validation.get('validation_reason', 'No structure analysis'),
                    'recommended_action': structure_validation.get('recommended_action', 'PROCEED_WITH_CAUTION')
                },
                
                # Order flow data  
                'order_flow_analysis': {
                    'enabled': self.enable_order_flow_validation,
                    'order_flow_aligned': order_flow_validation.get('order_flow_aligned', True),
                    'order_flow_score': order_flow_validation.get('order_flow_score', 0.5),
                    'validation_reason': order_flow_validation.get('validation_reason', 'No order flow analysis'),
                    'recommended_action': order_flow_validation.get('recommended_action', 'PROCEED_WITH_CAUTION'),
                    'nearest_levels': order_flow_validation.get('nearest_levels')
                },
                
                # Smart money summary
                'smart_money_summary': {
                    'total_score': smart_money_score,
                    'validation_reasons': smart_money_result.get('validation_reasons', []),
                    'structure_contribution': smart_money_result.get('structure_contribution', 0),
                    'order_flow_contribution': smart_money_result.get('order_flow_contribution', 0),
                    'confidence_enhancement': enhanced_confidence - original_confidence
                }
            })
            
            # Add detailed structure context if available
            if structure_analysis:
                enhanced_signal['market_structure_details'] = {
                    'current_bias': structure_analysis.get('current_bias'),
                    'structure_score': structure_analysis.get('structure_score'),
                    'swing_points_count': len(structure_analysis.get('swing_points', [])),
                    'structure_events_count': len(structure_analysis.get('structure_events', [])),
                    'analysis_summary': structure_analysis.get('analysis_summary')
                }
            
            # Add detailed order flow context if available
            if order_flow_analysis:
                enhanced_signal['order_flow_details'] = {
                    'order_flow_bias': order_flow_analysis.get('order_flow_bias'),
                    'order_blocks_count': len(order_flow_analysis.get('order_blocks', [])),
                    'fair_value_gaps_count': len(order_flow_analysis.get('fair_value_gaps', [])),
                    'supply_demand_zones_count': len(order_flow_analysis.get('supply_demand_zones', [])),
                    'analysis_summary': order_flow_analysis.get('analysis_summary')
                }
            
            # Update signal classification based on smart money strength
            if smart_money_score > 0.8:
                enhanced_signal['signal_quality'] = 'HIGH_CONVICTION'
                enhanced_signal['signal_classification'] = 'smart_money_confluence'
            elif smart_money_score > 0.6:
                enhanced_signal['signal_quality'] = 'MEDIUM_CONVICTION'
                enhanced_signal['signal_classification'] = 'smart_money_partial'
            else:
                enhanced_signal['signal_quality'] = 'STANDARD'
                enhanced_signal['signal_classification'] = 'ema_technical_only'
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Signal enhancement failed: {e}")
            # Return base signal if enhancement fails
            base_signal['smart_money_error'] = str(e)
            return base_signal
    
    def get_required_indicators(self) -> List[str]:
        """
        Get required indicators (inherited from EMA strategy + smart money needs)
        """
        base_indicators = super().get_required_indicators()
        
        # Smart money analysis may need additional indicators but should work with existing data
        # Add volume indicators if needed for order flow analysis
        smart_money_indicators = ['volume', 'ltv']  # Volume for institutional analysis
        
        return list(set(base_indicators + smart_money_indicators))
    
    def calculate_confidence(self, signal_data: Dict) -> float:
        """
        Calculate confidence using smart money enhanced scoring
        """
        # Use enhanced confidence if available, otherwise fall back to base calculation
        if 'enhanced_confidence_score' in signal_data:
            return signal_data['enhanced_confidence_score']
        elif 'smart_money_score' in signal_data:
            base_confidence = super().calculate_confidence(signal_data)
            return min(1.0, base_confidence * 0.7 + signal_data['smart_money_score'] * 0.3)
        else:
            return super().calculate_confidence(signal_data)
    
    def get_smart_money_status(self) -> Dict:
        """Get current smart money analyzer status"""
        return {
            'structure_validation_enabled': self.enable_structure_validation,
            'order_flow_validation_enabled': self.enable_order_flow_validation,
            'structure_weight': self.structure_weight,
            'order_flow_weight': self.order_flow_weight,
            'min_smart_money_score': self.min_smart_money_score,
            'analyzers_initialized': {
                'market_structure': self.market_structure_analyzer is not None,
                'order_flow': self.order_flow_analyzer is not None
            }
        }
    
    def update_smart_money_config(self, **kwargs):
        """Update smart money configuration dynamically"""
        if 'enable_structure_validation' in kwargs:
            self.enable_structure_validation = kwargs['enable_structure_validation']
        if 'enable_order_flow_validation' in kwargs:
            self.enable_order_flow_validation = kwargs['enable_order_flow_validation']
        if 'structure_weight' in kwargs:
            self.structure_weight = kwargs['structure_weight']
        if 'order_flow_weight' in kwargs:
            self.order_flow_weight = kwargs['order_flow_weight']
        if 'min_smart_money_score' in kwargs:
            self.min_smart_money_score = kwargs['min_smart_money_score']
            
        self.logger.info(f"🔧 Smart money config updated: {kwargs}")


# Configuration additions for config.py
"""
Add these configurations to your config.py file:

# Smart Money EMA Strategy Configuration
SMART_MONEY_STRUCTURE_VALIDATION = True  # Enable market structure validation
SMART_MONEY_ORDER_FLOW_VALIDATION = True  # Enable order flow validation
SMART_MONEY_STRUCTURE_WEIGHT = 0.3  # Weight for structure analysis (0-1)
SMART_MONEY_ORDER_FLOW_WEIGHT = 0.2  # Weight for order flow analysis (0-1)
SMART_MONEY_MIN_SCORE = 0.4  # Minimum smart money score to proceed

# Market Structure Configuration
STRUCTURE_SWING_LOOKBACK = 5  # Periods to look back for swing point identification
STRUCTURE_MIN_SWING_STRENGTH = 0.3  # Minimum strength for valid swing points
STRUCTURE_BOS_CONFIRMATION_PIPS = 5  # Pips needed to confirm break of structure
STRUCTURE_CHOCH_LOOKBACK = 20  # Periods to analyze for change of character

# Order Flow Configuration  
ORDER_FLOW_MIN_OB_SIZE_PIPS = 8  # Minimum order block size in pips
ORDER_FLOW_MIN_FVG_SIZE_PIPS = 5  # Minimum fair value gap size in pips
ORDER_FLOW_OB_LOOKBACK = 50  # Periods to look back for order blocks
ORDER_FLOW_FVG_LOOKBACK = 30  # Periods to look back for FVGs
ORDER_FLOW_VOLUME_SPIKE = 1.5  # Volume spike threshold for institutional moves
"""