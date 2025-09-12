# core/strategies/helpers/combined_signal_detector.py
"""
Combined Strategy Signal Detector - MODULAR HELPER
🔥 SIGNAL DETECTION: Core signal detection logic for combined strategy
🏗️ MODULAR: Focused on signal detection algorithms
🎯 MAINTAINABLE: Single responsibility - signal detection only
⚡ PERFORMANCE: Efficient signal detection with safety filters
🚨 SAFETY: Comprehensive contradiction detection and prevention
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np


class CombinedSignalDetector:
    """
    🔥 SIGNAL DETECTOR: Core signal detection for combined strategy
    
    Handles:
    - Combined signal detection logic
    - Safety filter application
    - Signal enhancement and validation
    - Multiple combination modes (consensus, weighted, dynamic)
    """
    
    def __init__(self, logger: logging.Logger, forex_optimizer=None, validator=None, strategy_manager=None):
        self.logger = logger
        self.forex_optimizer = forex_optimizer
        self.validator = validator
        self.strategy_manager = strategy_manager
        
        # Detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'safety_filter_rejections': 0,
            'consensus_mode_usage': 0,
            'weighted_mode_usage': 0,
            'dynamic_mode_usage': 0,
            'mean_reversion_bypasses': 0
        }
        
        # Detection configuration
        self.detection_config = {
            'enable_safety_filters': True,
            'require_volume_confirmation': False,
            'enable_mean_reversion_bypass': True,
            'min_signal_confidence': 0.6,
            'enable_enhanced_validation': True
        }
        
        self.logger.debug("✅ CombinedSignalDetector initialized")

    def detect_combined_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float, 
        timeframe: str,
        precomputed_signals: Dict[str, Optional[Dict]] = None  # 🔧 ADD THIS LINE
    ) -> Optional[Dict]:
        """
        🔧 SIMPLE CHANGE: Just add one optional parameter to existing method
        """
        try:
            self.detection_stats['total_detections'] += 1
            
            # 🔧 ADD THIS BLOCK: Use precomputed signals if available
            if precomputed_signals is not None:
                self.logger.debug(f"🚀 Using precomputed signals for {epic}")
                all_signals = precomputed_signals
            else:
                # 🔄 KEEP EXISTING LOGIC: Get signals from strategy manager
                if not self.strategy_manager:
                    self.logger.error("❌ Strategy manager not available")
                    return None
                
                all_signals = self.strategy_manager.get_all_strategy_signals(df, epic, spread_pips, timeframe)
            
            # 🔄 KEEP ALL EXISTING LOGIC BELOW THIS POINT - NO CHANGES
            if not all_signals:
                self.logger.debug("📭 No signals from any strategy")
                return None
            
            # Determine combination mode (existing)
            combination_mode = self._get_combination_mode()
            
            # Apply combination logic (existing)
            combined_signal = self._apply_combination_mode(all_signals, combination_mode, epic, timeframe)
            
            if not combined_signal:
                self.logger.debug(f"📭 No signal generated in {combination_mode} mode")
                return None
            
            # Apply safety filters (existing)
            if self.detection_config['enable_safety_filters']:
                safety_result = self._apply_comprehensive_safety_filters(combined_signal, df.iloc[-1], epic)
                
                if not safety_result['passed_all_filters']:
                    self.detection_stats['safety_filter_rejections'] += 1
                    self.logger.info(f"🚫 Signal rejected by safety filters: {safety_result['failed_filters']}")
                    return None
                
                combined_signal['safety_validation'] = safety_result
            
            # Apply final enhancements (existing)
            enhanced_signal = self._apply_final_enhancements(combined_signal, df)
            
            self.detection_stats['successful_detections'] += 1
            self.logger.info(f"✅ Combined signal detected: {enhanced_signal.get('signal_type')} for {epic}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Combined signal detection failed: {e}")
            return None

    def _get_combination_mode(self) -> str:
        """Get the combination mode from configuration"""
        try:
            if self.forex_optimizer:
                config = self.forex_optimizer.get_combined_strategy_config()
                return config.get('mode', 'consensus')
            else:
                return 'consensus'  # Default fallback
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get combination mode: {e}")
            return 'consensus'

    def _apply_combination_mode(
        self, 
        all_signals: Dict[str, Optional[Dict]], 
        combination_mode: str, 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply the specified combination mode"""
        try:
            if combination_mode == 'consensus':
                self.detection_stats['consensus_mode_usage'] += 1
                return self._apply_consensus_mode(all_signals, epic, timeframe)
            elif combination_mode == 'weighted':
                self.detection_stats['weighted_mode_usage'] += 1
                return self._apply_weighted_mode(all_signals, epic, timeframe)
            elif combination_mode == 'dynamic':
                self.detection_stats['dynamic_mode_usage'] += 1
                return self._apply_dynamic_mode(all_signals, epic, timeframe)
            else:
                self.logger.warning(f"⚠️ Unknown combination mode: {combination_mode}, using consensus")
                self.detection_stats['consensus_mode_usage'] += 1
                return self._apply_consensus_mode(all_signals, epic, timeframe)
                
        except Exception as e:
            self.logger.error(f"❌ Combination mode application failed: {e}")
            return None

    def _apply_consensus_mode(
        self, 
        all_signals: Dict[str, Optional[Dict]], 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply consensus combination mode"""
        try:
            if self.validator:
                consensus_result = self.validator.validate_signal_consensus(all_signals, epic)
                
                if consensus_result['consensus_achieved']:
                    # Create combined signal from consensus
                    return self._create_consensus_combined_signal(
                        consensus_result, all_signals, epic, timeframe
                    )
            
            # Fallback to strategy manager combination
            if self.strategy_manager:
                return self.strategy_manager.combine_strategy_results(
                    all_signals, 'consensus', epic, timeframe
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Consensus mode failed: {e}")
            return None

    def _apply_weighted_mode(
        self, 
        all_signals: Dict[str, Optional[Dict]], 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply weighted combination mode"""
        try:
            if self.strategy_manager:
                return self.strategy_manager.combine_strategy_results(
                    all_signals, 'weighted', epic, timeframe
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Weighted mode failed: {e}")
            return None

    def _apply_dynamic_mode(
        self, 
        all_signals: Dict[str, Optional[Dict]], 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply dynamic combination mode"""
        try:
            if self.strategy_manager:
                return self.strategy_manager.combine_strategy_results(
                    all_signals, 'dynamic', epic, timeframe
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic mode failed: {e}")
            return None

    def _create_consensus_combined_signal(
        self, 
        consensus_result: Dict, 
        all_signals: Dict[str, Optional[Dict]], 
        epic: str, 
        timeframe: str
    ) -> Dict:
        """Create a combined signal from consensus results"""
        try:
            # Get the best signal from contributing strategies
            contributing_strategies = consensus_result['contributing_strategies']
            contributing_signals = {k: v for k, v in all_signals.items() if k in contributing_strategies and v is not None}
            
            if not contributing_signals:
                raise ValueError("No contributing signals found")
            
            # Find the best signal (highest confidence)
            best_signal = max(contributing_signals.values(), key=lambda x: x.get('confidence_score', x.get('confidence', 0)))
            
            # Create combined signal
            combined_signal = best_signal.copy()
            consensus_confidence = consensus_result.get('consensus_strength', 0.8)
            combined_signal.update({
                'signal_type': consensus_result['signal_type'],
                'confidence_score': consensus_result.get('consensus_strength', 0.8),
                'combination_mode': 'consensus',
                'contributing_strategies': contributing_strategies,
                'individual_confidences': consensus_result.get('individual_confidences', {}),
                'consensus_strength': consensus_result.get('consensus_strength', 0),
                'strategy_count': consensus_result.get('strategy_count', len(contributing_strategies)),
                'epic': epic,
                'timeframe': timeframe,
                'timestamp': datetime.now(),
                'strategy': f'combined_consensus_{len(contributing_strategies)}'
            })
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"❌ Consensus signal creation failed: {e}")
            return None

    def _apply_comprehensive_safety_filters(
        self, 
        signal: Dict, 
        latest_data: pd.Series, 
        epic: str
    ) -> Dict:
        """Apply comprehensive safety filters to prevent bad signals"""
        try:
            # Use validator if available
            if self.validator:
                return self.validator.apply_safety_filters(signal, latest_data, epic)
            
            # Fallback to basic safety checks
            return self._apply_basic_safety_checks(signal, latest_data, epic)
            
        except Exception as e:
            self.logger.error(f"❌ Safety filters application failed: {e}")
            return {
                'passed_all_filters': False,
                'error': str(e),
                'failed_filters': ['safety_filter_error']
            }

    def _apply_basic_safety_checks(
        self, 
        signal: Dict, 
        latest_data: pd.Series, 
        epic: str
    ) -> Dict:
        """Apply basic safety checks as fallback"""
        try:
            signal_type = signal.get('signal_type', '').upper()
            current_price = signal.get('signal_price', latest_data.get('close', 0))
            
            safety_results = {
                'passed_all_filters': True,
                'failed_filters': [],
                'filter_results': {}
            }
            
            # Basic EMA 200 check
            ema_200 = latest_data.get('ema_200', 0)
            if ema_200 > 0:
                if signal_type in ['BUY', 'BULL'] and current_price <= ema_200:
                    safety_results['passed_all_filters'] = False
                    safety_results['failed_filters'].append('ema200_contradiction')
                elif signal_type in ['SELL', 'BEAR'] and current_price >= ema_200:
                    safety_results['passed_all_filters'] = False
                    safety_results['failed_filters'].append('ema200_contradiction')
            
            # Basic MACD check
            macd_histogram = latest_data.get('macd_histogram', 0)
            if signal_type in ['BUY', 'BULL'] and macd_histogram < -0.0001:
                safety_results['passed_all_filters'] = False
                safety_results['failed_filters'].append('macd_contradiction')
            elif signal_type in ['SELL', 'BEAR'] and macd_histogram > 0.0001:
                safety_results['passed_all_filters'] = False
                safety_results['failed_filters'].append('macd_contradiction')
            
            return safety_results
            
        except Exception as e:
            return {
                'passed_all_filters': False,
                'error': str(e),
                'failed_filters': ['basic_safety_check_error']
            }

    def _apply_final_enhancements(self, signal: Dict, df: pd.DataFrame) -> Dict:
        """Apply final enhancements to the combined signal"""
        try:
            enhanced_signal = signal.copy()
            
            # Add signal generation metadata
            enhanced_signal.update({
                'signal_source': 'combined_strategy_modular',
                'detection_timestamp': datetime.now(),
                'data_bars_used': len(df),
                'enhancement_applied': True,
                'modular_architecture': True
            })
            
            # Add risk management if not present
            if 'stop_loss' not in enhanced_signal or 'take_profit' not in enhanced_signal:
                enhanced_signal = self._add_risk_management(enhanced_signal, df.iloc[-1])
            
            # Add execution metadata
            enhanced_signal = self._add_execution_metadata(enhanced_signal)
            
            # Ensure JSON serializable
            enhanced_signal = self._ensure_json_serializable(enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Final enhancement failed: {e}")
            return signal

    def _add_risk_management(self, signal: Dict, latest_data: pd.Series) -> Dict:
        """Add risk management levels to signal"""
        try:
            signal_type = signal.get('signal_type', '')
            current_price = signal.get('signal_price', latest_data.get('close', 0))
            epic = signal.get('epic', '')
            
            # Calculate pip size
            pip_size = 0.01 if 'JPY' in epic else 0.0001
            
            # Default risk parameters
            default_stop_distance = 20  # pips
            risk_reward_ratio = 2.0
            
            stop_distance_price = default_stop_distance * pip_size
            
            if signal_type in ['BULL', 'BUY']:
                signal['stop_loss'] = current_price - stop_distance_price
                signal['take_profit'] = current_price + (stop_distance_price * risk_reward_ratio)
            elif signal_type in ['BEAR', 'SELL']:
                signal['stop_loss'] = current_price + stop_distance_price
                signal['take_profit'] = current_price - (stop_distance_price * risk_reward_ratio)
            
            # Calculate risk/reward ratio
            if 'stop_loss' in signal and 'take_profit' in signal:
                stop_distance = abs(current_price - signal['stop_loss'])
                target_distance = abs(signal['take_profit'] - current_price)
                if stop_distance > 0:
                    signal['risk_reward_ratio'] = target_distance / stop_distance
                else:
                    signal['risk_reward_ratio'] = risk_reward_ratio
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Risk management addition failed: {e}")
            return signal

    def _add_execution_metadata(self, signal: Dict) -> Dict:
        """Add execution-related metadata"""
        try:
            # Add timing metadata
            signal['alert_timestamp'] = datetime.now()
            signal['processing_timestamp'] = datetime.now().isoformat()
            
            # Add status metadata
            signal['status'] = 'NEW'
            signal['processed'] = False
            signal['alert_level'] = 'MEDIUM'
            
            # Add combined strategy specific metadata
            contributing_count = len(signal.get('contributing_strategies', []))
            signal['alert_message'] = f"Combined {signal.get('signal_type', 'UNKNOWN')} signal from {contributing_count} strategies"
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Execution metadata addition failed: {e}")
            return signal

    def _ensure_json_serializable(self, signal: Dict) -> Dict:
        """Ensure all signal data is JSON serializable"""
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        try:
            return convert_for_json(signal)
        except Exception as e:
            self.logger.error(f"❌ JSON serialization failed: {e}")
            return signal

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get signal detection statistics"""
        try:
            total = self.detection_stats['total_detections']
            successful = self.detection_stats['successful_detections']
            
            return {
                'total_detections': total,
                'successful_detections': successful,
                'failed_detections': total - successful,
                'success_rate_percent': (successful / total * 100) if total > 0 else 0,
                'safety_filter_rejections': self.detection_stats['safety_filter_rejections'],
                'mean_reversion_bypasses': self.detection_stats['mean_reversion_bypasses'],
                'mode_usage': {
                    'consensus': self.detection_stats['consensus_mode_usage'],
                    'weighted': self.detection_stats['weighted_mode_usage'],
                    'dynamic': self.detection_stats['dynamic_mode_usage']
                },
                'detection_config': self.detection_config,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Detection stats collection failed: {e}")
            return {'error': str(e)}

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics information"""
        return {
            'module_name': 'CombinedSignalDetector',
            'initialization_successful': True,
            'forex_optimizer_available': self.forex_optimizer is not None,
            'validator_available': self.validator is not None,
            'strategy_manager_available': self.strategy_manager is not None,
            'detection_config': self.detection_config,
            'detection_stats': self.get_detection_stats(),
            'safety_filters_enabled': self.detection_config['enable_safety_filters'],
            'enhanced_validation_enabled': self.detection_config['enable_enhanced_validation'],
            'timestamp': datetime.now().isoformat()
        }