# core/strategies/smart_money_macd_strategy.py
"""
Smart Money Enhanced MACD Strategy - Phase 2 Integration
Combines your existing MACD + EMA200 strategy with order flow analysis:
- Validates MACD signals against Order Blocks
- Confirms entries near Fair Value Gaps
- Identifies institutional supply/demand zones
- Maintains existing signal quality while adding smart money context

This strategy builds on your proven MACD histogram + EMA200 logic while adding
order flow confirmation to catch signals aligned with institutional positioning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .macd_strategy import MACDStrategy
from ..intelligence.order_flow_analyzer import OrderFlowAnalyzer
try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class SmartMoneyMACDStrategy(MACDStrategy):
    """
    Smart Money Enhanced MACD Strategy
    
    Inherits from your proven MACDStrategy and adds order flow validation:
    1. Uses existing MACD histogram + EMA200 confirmation logic
    2. Validates signals against order blocks and institutional zones
    3. Confirms entries near fair value gaps
    4. Adds supply/demand zone context
    5. Maintains signal frequency while improving institutional alignment
    """
    
    def __init__(self):
        # Initialize parent MACD strategy
        super().__init__()
        
        # Initialize order flow analyzer
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        # Smart money configuration
        self.enable_order_flow_validation = getattr(config, 'SMART_MACD_ORDER_FLOW_VALIDATION', True)
        self.require_order_block_confluence = getattr(config, 'SMART_MACD_REQUIRE_OB_CONFLUENCE', False)
        self.fvg_proximity_pips = getattr(config, 'SMART_MACD_FVG_PROXIMITY_PIPS', 15)
        self.order_flow_boost_factor = getattr(config, 'SMART_MACD_ORDER_FLOW_BOOST', 1.2)
        self.order_flow_penalty_factor = getattr(config, 'SMART_MACD_ORDER_FLOW_PENALTY', 0.8)
        
        # Update strategy name
        self.name = 'smart_money_macd'
        
        self.logger.info("📊 SmartMoneyMACDStrategy initialized")
        self.logger.info(f"   Order flow validation: {'✅' if self.enable_order_flow_validation else '❌'}")
        self.logger.info(f"   Require OB confluence: {'✅' if self.require_order_block_confluence else '❌'}")
        self.logger.info(f"   FVG proximity: {self.fvg_proximity_pips} pips")
    
    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Enhanced MACD signal detection with order flow validation
        
        Process:
        1. Get base MACD signal from parent class
        2. Analyze order flow context (OB, FVG, Supply/Demand)
        3. Apply order flow validation and confluence checks
        4. Adjust confidence based on institutional alignment
        """
        try:
            # Step 1: Get base MACD signal using proven logic
            base_signal = super().detect_signal(df, epic, spread_pips, timeframe)
            
            if not base_signal:
                return None  # No base signal, no need for order flow analysis
            
            self.logger.debug(f"📈 Base MACD signal detected for {epic}: {base_signal['signal_type']}")
            
            # Step 2: Analyze order flow context if enabled
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
                    self.logger.debug(f"📊 Order flow validation: {order_flow_validation['recommended_action']}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Order flow analysis failed for {epic}: {e}")
            
            # Step 3: Apply order flow validation
            validation_result = self._apply_order_flow_validation(
                base_signal, order_flow_validation, order_flow_analysis
            )
            
            if not validation_result['proceed_with_signal']:
                self.logger.debug(f"🚫 Order flow validation rejected MACD signal for {epic}: {validation_result['rejection_reason']}")
                return None
            
            # Step 4: Enhance signal with order flow data
            enhanced_signal = self._enhance_macd_signal_with_order_flow(
                base_signal, validation_result, order_flow_validation, order_flow_analysis
            )
            
            self.logger.debug(f"✅ Order flow enhanced MACD signal for {epic}: "
                            f"confidence {base_signal['confidence_score']:.3f} -> "
                            f"{enhanced_signal.get('enhanced_confidence_score', base_signal['confidence_score']):.3f}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Smart money MACD signal detection failed for {epic}: {e}")
            # Fallback to base signal if order flow analysis fails
            return super().detect_signal(df, epic, spread_pips, timeframe)
    
    def _apply_order_flow_validation(
        self, 
        base_signal: Dict, 
        order_flow_validation: Dict,
        order_flow_analysis: Optional[Dict] = None
    ) -> Dict:
        """Apply order flow validation to MACD signal"""
        try:
            current_price = base_signal['entry_price']
            signal_type = base_signal['signal_type']
            
            validation_factors = []
            confluence_factors = []
            proceed_with_signal = True
            rejection_reason = None
            
            # Base MACD signal strength
            base_confidence = base_signal.get('confidence_score', 0.75)
            validation_factors.append({
                'factor': 'macd_base',
                'score': base_confidence,
                'weight': 0.6,
                'description': f'MACD histogram + EMA200: {base_confidence:.3f}'
            })
            
            # Order flow validation factors
            if order_flow_analysis:
                order_blocks = order_flow_analysis.get('order_blocks', [])
                fair_value_gaps = order_flow_analysis.get('fair_value_gaps', [])
                supply_demand_zones = order_flow_analysis.get('supply_demand_zones', [])
                
                # Factor 1: Order Block Confluence
                ob_confluence = self._check_order_block_confluence(
                    current_price, signal_type, order_blocks
                )
                if ob_confluence['has_confluence']:
                    validation_factors.append({
                        'factor': 'order_block',
                        'score': ob_confluence['strength'],
                        'weight': 0.25,
                        'description': f"Order block confluence: {ob_confluence['description']}"
                    })
                    confluence_factors.append('order_block')
                
                # Factor 2: Fair Value Gap Proximity
                fvg_proximity = self._check_fvg_proximity(
                    current_price, signal_type, fair_value_gaps
                )
                if fvg_proximity['near_fvg']:
                    validation_factors.append({
                        'factor': 'fair_value_gap',
                        'score': fvg_proximity['strength'],
                        'weight': 0.15,
                        'description': f"Near FVG: {fvg_proximity['description']}"
                    })
                    confluence_factors.append('fair_value_gap')
                
                # Factor 3: Supply/Demand Zone Context
                zone_context = self._check_supply_demand_context(
                    current_price, signal_type, supply_demand_zones
                )
                if zone_context['in_zone']:
                    validation_factors.append({
                        'factor': 'supply_demand',
                        'score': zone_context['strength'],
                        'weight': 0.2,
                        'description': f"In {zone_context['zone_type']} zone: {zone_context['strength']:.3f}"
                    })
                    confluence_factors.append('supply_demand')
            
            # Check if order block confluence is required
            if self.require_order_block_confluence:
                has_ob_confluence = any(f['factor'] == 'order_block' for f in validation_factors)
                if not has_ob_confluence:
                    proceed_with_signal = False
                    rejection_reason = "Required order block confluence not found"
            
            # Calculate weighted order flow score
            order_flow_score = 0.0
            total_weight = 0.0
            
            for factor in validation_factors:
                weighted_score = factor['score'] * factor['weight']
                order_flow_score += weighted_score
                total_weight += factor['weight']
            
            if total_weight > 0:
                order_flow_score = order_flow_score / total_weight
            else:
                order_flow_score = 0.5  # Neutral if no factors
            
            return {
                'proceed_with_signal': proceed_with_signal,
                'order_flow_score': order_flow_score,
                'validation_factors': validation_factors,
                'confluence_factors': confluence_factors,
                'rejection_reason': rejection_reason,
                'confluence_count': len(confluence_factors)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Order flow validation failed: {e}")
            return {
                'proceed_with_signal': True,  # Default to allow
                'order_flow_score': 0.5,
                'validation_factors': [],
                'rejection_reason': None
            }
    
    def _check_order_block_confluence(
        self, 
        current_price: float, 
        signal_type: str, 
        order_blocks: List[Dict]
    ) -> Dict:
        """Check for order block confluence with MACD signal"""
        try:
            pip_size = self._get_pip_size()
            proximity_threshold = 10 * pip_size  # 10 pips proximity
            
            relevant_obs = []
            
            for ob in order_blocks:
                if ob.get('broken', False):
                    continue  # Skip broken order blocks
                
                ob_high = ob['high']
                ob_low = ob['low']
                ob_type = ob['block_type']
                
                # Check if price is near order block
                distance_to_ob = min(
                    abs(current_price - ob_high),
                    abs(current_price - ob_low)
                )
                
                if distance_to_ob <= proximity_threshold:
                    # Check alignment with signal direction
                    if ((signal_type in ['BUY', 'BULL'] and 'BULLISH' in ob_type) or
                        (signal_type in ['SELL', 'BEAR'] and 'BEARISH' in ob_type)):
                        
                        strength = ob.get('strength', 0.5)
                        # Boost strength based on how close we are
                        proximity_factor = 1.0 - (distance_to_ob / proximity_threshold)
                        adjusted_strength = min(1.0, strength * (1.0 + proximity_factor))
                        
                        relevant_obs.append({
                            'type': ob_type,
                            'strength': adjusted_strength,
                            'distance': distance_to_ob,
                            'touched': ob.get('touched', False),
                            'mitigation_count': ob.get('mitigation_count', 0)
                        })
            
            if relevant_obs:
                # Get strongest confluence
                best_ob = max(relevant_obs, key=lambda x: x['strength'])
                return {
                    'has_confluence': True,
                    'strength': best_ob['strength'],
                    'description': f"{best_ob['type']} OB at {best_ob['distance']/pip_size:.1f} pips",
                    'order_block_data': best_ob
                }
            else:
                return {
                    'has_confluence': False,
                    'strength': 0.0,
                    'description': 'No relevant order blocks nearby'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Order block confluence check failed: {e}")
            return {'has_confluence': False, 'strength': 0.0, 'description': 'Error checking OB confluence'}
    
    def _check_fvg_proximity(
        self, 
        current_price: float, 
        signal_type: str, 
        fair_value_gaps: List[Dict]
    ) -> Dict:
        """Check proximity to relevant Fair Value Gaps"""
        try:
            pip_size = self._get_pip_size()
            proximity_threshold = self.fvg_proximity_pips * pip_size
            
            relevant_fvgs = []
            
            for fvg in fair_value_gaps:
                if fvg.get('filled_percentage', 0) > 80:
                    continue  # Skip mostly filled FVGs
                
                fvg_top = fvg['top']
                fvg_bottom = fvg['bottom']
                fvg_type = fvg['gap_type']
                
                # Check if price is near or in FVG
                in_fvg = fvg_bottom <= current_price <= fvg_top
                distance_to_fvg = 0 if in_fvg else min(
                    abs(current_price - fvg_top),
                    abs(current_price - fvg_bottom)
                )
                
                if in_fvg or distance_to_fvg <= proximity_threshold:
                    # Check alignment with signal direction
                    if ((signal_type in ['BUY', 'BULL'] and 'BULLISH' in fvg_type) or
                        (signal_type in ['SELL', 'BEAR'] and 'BEARISH' in fvg_type)):
                        
                        significance = fvg.get('significance', 0.5)
                        fill_percentage = fvg.get('filled_percentage', 0)
                        
                        # Boost significance for unfilled FVGs
                        unfilled_factor = 1.0 - (fill_percentage / 100)
                        adjusted_significance = significance * (1.0 + unfilled_factor * 0.3)
                        
                        # Boost for being inside the gap
                        position_factor = 1.2 if in_fvg else 1.0
                        final_strength = min(1.0, adjusted_significance * position_factor)
                        
                        relevant_fvgs.append({
                            'type': fvg_type,
                            'strength': final_strength,
                            'distance': distance_to_fvg,
                            'in_gap': in_fvg,
                            'fill_percentage': fill_percentage,
                            'size_pips': fvg.get('size_pips', 0)
                        })
            
            if relevant_fvgs:
                # Get strongest FVG confluence
                best_fvg = max(relevant_fvgs, key=lambda x: x['strength'])
                position_desc = "inside" if best_fvg['in_gap'] else f"{best_fvg['distance']/pip_size:.1f} pips from"
                return {
                    'near_fvg': True,
                    'strength': best_fvg['strength'],
                    'description': f"{best_fvg['type']} FVG ({position_desc})",
                    'fvg_data': best_fvg
                }
            else:
                return {
                    'near_fvg': False,
                    'strength': 0.0,
                    'description': 'No relevant FVGs nearby'
                }
                
        except Exception as e:
            self.logger.error(f"❌ FVG proximity check failed: {e}")
            return {'near_fvg': False, 'strength': 0.0, 'description': 'Error checking FVG proximity'}
    
    def _check_supply_demand_context(
        self, 
        current_price: float, 
        signal_type: str, 
        supply_demand_zones: List[Dict]
    ) -> Dict:
        """Check if signal is in relevant supply/demand zone"""
        try:
            relevant_zones = []
            
            for zone in supply_demand_zones:
                if zone.get('broken', False):
                    continue  # Skip broken zones
                
                zone_high = zone['high']
                zone_low = zone['low']
                zone_type = zone['zone_type']
                
                # Check if price is in zone
                in_zone = zone_low <= current_price <= zone_high
                
                if in_zone:
                    # Check alignment with signal direction
                    if ((signal_type in ['BUY', 'BULL'] and zone_type == 'demand') or
                        (signal_type in ['SELL', 'BEAR'] and zone_type == 'supply')):
                        
                        strength = zone.get('strength', 0.5)
                        tests = zone.get('tests', 0)
                        
                        # Stronger zones that haven't been tested much are better
                        test_factor = max(0.5, 1.0 - (tests * 0.1))
                        adjusted_strength = strength * test_factor
                        
                        relevant_zones.append({
                            'type': zone_type,
                            'strength': adjusted_strength,
                            'tests': tests,
                            'zone_size': zone_high - zone_low
                        })
            
            if relevant_zones:
                # Get strongest zone
                best_zone = max(relevant_zones, key=lambda x: x['strength'])
                return {
                    'in_zone': True,
                    'strength': best_zone['strength'],
                    'zone_type': best_zone['type'],
                    'description': f"{best_zone['type']} zone (tested {best_zone['tests']} times)",
                    'zone_data': best_zone
                }
            else:
                return {
                    'in_zone': False,
                    'strength': 0.0,
                    'zone_type': None,
                    'description': 'Not in relevant supply/demand zone'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Supply/demand context check failed: {e}")
            return {'in_zone': False, 'strength': 0.0, 'description': 'Error checking S/D context'}
    
    def _enhance_macd_signal_with_order_flow(
        self, 
        base_signal: Dict, 
        validation_result: Dict,
        order_flow_validation: Dict,
        order_flow_analysis: Optional[Dict] = None
    ) -> Dict:
        """Enhance MACD signal with order flow analysis data"""
        try:
            enhanced_signal = base_signal.copy()
            
            # Update strategy name
            enhanced_signal['strategy'] = self.name
            
            # Calculate enhanced confidence
            base_confidence = base_signal.get('confidence_score', 0.75)
            order_flow_score = validation_result.get('order_flow_score', 0.5)
            confluence_count = validation_result.get('confluence_count', 0)
            
            # Apply confidence adjustment based on order flow
            if order_flow_score > 0.7 and confluence_count >= 2:
                # Strong order flow confluence - boost confidence
                confidence_multiplier = self.order_flow_boost_factor
                signal_quality = 'HIGH_CONVICTION'
            elif order_flow_score > 0.5 and confluence_count >= 1:
                # Some order flow confluence - slight boost
                confidence_multiplier = 1.1
                signal_quality = 'MEDIUM_CONVICTION'
            elif order_flow_score < 0.3:
                # Poor order flow alignment - reduce confidence
                confidence_multiplier = self.order_flow_penalty_factor
                signal_quality = 'LOW_CONVICTION'
            else:
                # Neutral order flow - maintain confidence
                confidence_multiplier = 1.0
                signal_quality = 'STANDARD'
            
            enhanced_confidence = min(1.0, base_confidence * confidence_multiplier)
            
            # Update confidence scores
            enhanced_signal['enhanced_confidence_score'] = enhanced_confidence
            enhanced_signal['original_confidence_score'] = base_confidence
            enhanced_signal['order_flow_score'] = order_flow_score
            enhanced_signal['confidence_score'] = enhanced_confidence  # Use enhanced as main
            
            # Add order flow analysis data
            enhanced_signal.update({
                'smart_money_validated': True,
                'smart_money_type': 'order_flow_macd',
                'smart_money_version': '1.0',
                'smart_money_timestamp': datetime.now().isoformat(),
                
                # Order flow validation results
                'order_flow_analysis': {
                    'enabled': self.enable_order_flow_validation,
                    'order_flow_score': order_flow_score,
                    'confluence_count': confluence_count,
                    'confluence_factors': validation_result.get('confluence_factors', []),
                    'validation_factors': validation_result.get('validation_factors', []),
                    'recommended_action': order_flow_validation.get('recommended_action', 'PROCEED')
                },
                
                # Signal quality and classification
                'signal_quality': signal_quality,
                'signal_classification': f'macd_order_flow_{confluence_count}_confluence',
                'confidence_adjustment': {
                    'original': base_confidence,
                    'multiplier': confidence_multiplier,
                    'enhanced': enhanced_confidence,
                    'adjustment_reason': f'Order flow score: {order_flow_score:.3f}, Confluences: {confluence_count}'
                }
            })
            
            # Add detailed order flow context if available
            if order_flow_analysis:
                enhanced_signal['order_flow_details'] = {
                    'order_flow_bias': order_flow_analysis.get('order_flow_bias'),
                    'active_order_blocks': len([ob for ob in order_flow_analysis.get('order_blocks', []) 
                                              if not ob.get('broken', False)]),
                    'unfilled_fvgs': len([fvg for fvg in order_flow_analysis.get('fair_value_gaps', []) 
                                        if fvg.get('filled_percentage', 0) < 50]),
                    'active_zones': len([zone for zone in order_flow_analysis.get('supply_demand_zones', []) 
                                       if not zone.get('broken', False)]),
                    'nearest_levels': order_flow_validation.get('nearest_levels'),
                    'analysis_summary': order_flow_analysis.get('analysis_summary')
                }
            
            # Add specific confluence details
            validation_factors = validation_result.get('validation_factors', [])
            if validation_factors:
                confluence_details = {}
                for factor in validation_factors:
                    if factor['factor'] != 'macd_base':  # Skip base MACD factor
                        confluence_details[factor['factor']] = {
                            'score': factor['score'],
                            'weight': factor['weight'],
                            'description': factor['description']
                        }
                enhanced_signal['confluence_details'] = confluence_details
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ MACD signal enhancement failed: {e}")
            # Return base signal with error info if enhancement fails
            base_signal['smart_money_error'] = str(e)
            return base_signal
    
    def get_required_indicators(self) -> List[str]:
        """Get required indicators (inherited from MACD + order flow needs)"""
        base_indicators = super().get_required_indicators()
        
        # Order flow analysis benefits from volume data
        order_flow_indicators = ['volume', 'ltv']
        
        return list(set(base_indicators + order_flow_indicators))
    
    def calculate_confidence(self, signal_data: Dict) -> float:
        """Calculate confidence using order flow enhanced scoring"""
        # Use enhanced confidence if available
        if 'enhanced_confidence_score' in signal_data:
            return signal_data['enhanced_confidence_score']
        elif 'order_flow_score' in signal_data:
            base_confidence = super().calculate_confidence(signal_data)
            order_flow_score = signal_data['order_flow_score']
            # Weighted combination favoring base MACD logic
            return min(1.0, base_confidence * 0.75 + order_flow_score * 0.25)
        else:
            return super().calculate_confidence(signal_data)
    
    def _get_pip_size(self) -> float:
        """Get pip size for current currency pair"""
        # Simplified - in practice would determine based on epic
        return 0.0001  # Standard for major pairs
    
    def get_smart_money_status(self) -> Dict:
        """Get current smart money configuration status"""
        return {
            'order_flow_validation_enabled': self.enable_order_flow_validation,
            'require_order_block_confluence': self.require_order_block_confluence,
            'fvg_proximity_pips': self.fvg_proximity_pips,
            'order_flow_boost_factor': self.order_flow_boost_factor,
            'order_flow_penalty_factor': self.order_flow_penalty_factor,
            'analyzer_initialized': self.order_flow_analyzer is not None
        }
    
    def update_smart_money_config(self, **kwargs):
        """Update smart money configuration dynamically"""
        if 'enable_order_flow_validation' in kwargs:
            self.enable_order_flow_validation = kwargs['enable_order_flow_validation']
        if 'require_order_block_confluence' in kwargs:
            self.require_order_block_confluence = kwargs['require_order_block_confluence']
        if 'fvg_proximity_pips' in kwargs:
            self.fvg_proximity_pips = kwargs['fvg_proximity_pips']
        if 'order_flow_boost_factor' in kwargs:
            self.order_flow_boost_factor = kwargs['order_flow_boost_factor']
        if 'order_flow_penalty_factor' in kwargs:
            self.order_flow_penalty_factor = kwargs['order_flow_penalty_factor']
            
        self.logger.info(f"🔧 Smart money MACD config updated: {kwargs}")


# Configuration additions for config.py
"""
Add these configurations to your config.py file:

# Smart Money MACD Strategy Configuration
SMART_MACD_ORDER_FLOW_VALIDATION = True  # Enable order flow validation for MACD
SMART_MACD_REQUIRE_OB_CONFLUENCE = False  # Require order block confluence (strict)
SMART_MACD_FVG_PROXIMITY_PIPS = 15  # Max distance to FVG for confluence
SMART_MACD_ORDER_FLOW_BOOST = 1.2  # Confidence boost for strong order flow alignment
SMART_MACD_ORDER_FLOW_PENALTY = 0.8  # Confidence penalty for poor order flow alignment
"""