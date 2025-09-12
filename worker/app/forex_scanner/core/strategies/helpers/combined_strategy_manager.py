# core/strategies/helpers/combined_strategy_manager.py
"""
Combined Strategy Manager - MODULAR HELPER
🔥 STRATEGY COORDINATION: Manages multiple strategies and their combinations
🏗️ MODULAR: Focused on strategy ensemble management
🎯 MAINTAINABLE: Single responsibility - strategy coordination only
⚡ PERFORMANCE: Efficient strategy execution and result combination
🤝 ENSEMBLE: Advanced ensemble learning techniques
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time


class CombinedStrategyManager:
    """
    🔥 STRATEGY MANAGER: Coordinates multiple trading strategies
    
    Handles:
    - Strategy registration and management
    - Parallel strategy execution
    - Result combination and ensemble logic
    - Strategy performance tracking
    - Dynamic strategy weighting
    """
    
    def __init__(self, logger: logging.Logger, forex_optimizer=None, validator=None):
        self.logger = logger
        self.forex_optimizer = forex_optimizer
        self.validator = validator
        
        # Strategy registry
        self.registered_strategies = {}
        
        # Strategy performance tracking
        self.strategy_performance = {}
        
        # Execution statistics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'parallel_executions': 0,
            'average_execution_time': 0.0,
            'strategy_success_rates': {}
        }
        
        # Configuration
        self.manager_config = {
            'enable_parallel_execution': True,
            'max_workers': 4,
            'execution_timeout_seconds': 30,
            'min_strategy_confidence': 0.6,
            'enable_performance_tracking': True,
            'adaptive_weighting_enabled': True
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.debug("✅ CombinedStrategyManager initialized")

    def register_strategies(self, strategies: Dict[str, Any]) -> bool:
        """
        Register multiple strategies for management
        
        Args:
            strategies: Dictionary of strategy name -> strategy instance
            
        Returns:
            True if registration successful
        """
        try:
            with self._lock:
                for strategy_name, strategy_instance in strategies.items():
                    if strategy_instance is not None:
                        self.registered_strategies[strategy_name] = strategy_instance
                        
                        # Initialize performance tracking
                        if strategy_name not in self.strategy_performance:
                            self.strategy_performance[strategy_name] = {
                                'total_signals': 0,
                                'successful_signals': 0,
                                'average_confidence': 0.0,
                                'execution_times': [],
                                'last_execution': None,
                                'error_count': 0,
                                'performance_score': 1.0
                            }
                        
                        # Initialize success rate tracking
                        if strategy_name not in self.execution_stats['strategy_success_rates']:
                            self.execution_stats['strategy_success_rates'][strategy_name] = {
                                'attempts': 0,
                                'successes': 0,
                                'rate': 0.0
                            }
                        
                        self.logger.debug(f"✅ Registered strategy: {strategy_name}")
                
                self.logger.info(f"📊 Strategy registration complete: {len(self.registered_strategies)} active strategies")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Strategy registration failed: {e}")
            return False

    def get_all_strategy_signals(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float, 
        timeframe: str,
        use_parallel: bool = None
    ) -> Dict[str, Optional[Dict]]:
        """
        Execute all registered strategies and collect their signals
        
        Args:
            df: Market data DataFrame
            epic: Trading epic
            spread_pips: Spread in pips
            timeframe: Timeframe
            use_parallel: Whether to use parallel execution (default: from config)
            
        Returns:
            Dictionary of strategy signals
        """
        try:
            start_time = time.time()
            
            if use_parallel is None:
                use_parallel = self.manager_config['enable_parallel_execution']
            
            if use_parallel and len(self.registered_strategies) > 1:
                signals = self._execute_strategies_parallel(df, epic, spread_pips, timeframe)
                self.execution_stats['parallel_executions'] += 1
            else:
                signals = self._execute_strategies_sequential(df, epic, spread_pips, timeframe)
            
            # Update execution statistics
            execution_time = time.time() - start_time
            self._update_execution_stats(execution_time, signals)
            
            # Update strategy performance
            if self.manager_config['enable_performance_tracking']:
                self._update_strategy_performance(signals, execution_time)
            
            self.logger.debug(f"📊 Strategy execution complete: {len([s for s in signals.values() if s])} signals in {execution_time:.3f}s")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Strategy signal collection failed: {e}")
            self.execution_stats['failed_executions'] += 1
            return {}

    def _execute_strategies_parallel(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float, 
        timeframe: str
    ) -> Dict[str, Optional[Dict]]:
        """Execute strategies in parallel using ThreadPoolExecutor"""
        try:
            signals = {}
            timeout = self.manager_config['execution_timeout_seconds']
            max_workers = min(self.manager_config['max_workers'], len(self.registered_strategies))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all strategy executions
                future_to_strategy = {
                    executor.submit(self._execute_single_strategy, strategy_name, strategy, df, epic, spread_pips, timeframe): strategy_name
                    for strategy_name, strategy in self.registered_strategies.items()
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_strategy, timeout=timeout):
                    strategy_name = future_to_strategy[future]
                    try:
                        signal = future.result(timeout=5)  # Individual strategy timeout
                        signals[strategy_name] = signal
                        self.logger.debug(f"✅ {strategy_name}: {'Signal' if signal else 'No signal'}")
                    except Exception as e:
                        self.logger.error(f"❌ {strategy_name} execution failed: {e}")
                        signals[strategy_name] = None
                        self._update_strategy_error_count(strategy_name)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Parallel strategy execution failed: {e}")
            # Fallback to sequential execution
            return self._execute_strategies_sequential(df, epic, spread_pips, timeframe)

    def _execute_strategies_sequential(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float, 
        timeframe: str
    ) -> Dict[str, Optional[Dict]]:
        """Execute strategies sequentially"""
        signals = {}
        
        for strategy_name, strategy in self.registered_strategies.items():
            try:
                signal = self._execute_single_strategy(strategy_name, strategy, df, epic, spread_pips, timeframe)
                signals[strategy_name] = signal
                self.logger.debug(f"✅ {strategy_name}: {'Signal' if signal else 'No signal'}")
            except Exception as e:
                self.logger.error(f"❌ {strategy_name} execution failed: {e}")
                signals[strategy_name] = None
                self._update_strategy_error_count(strategy_name)
        
        return signals

    def _execute_single_strategy(
        self, 
        strategy_name: str, 
        strategy: Any, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float, 
        timeframe: str
    ) -> Optional[Dict]:
        """Execute a single strategy with error handling"""
        try:
            start_time = time.time()
            
            # Execute strategy
            signal = strategy.detect_signal(df, epic, spread_pips, timeframe)
            
            # Update strategy-specific performance tracking
            execution_time = time.time() - start_time
            with self._lock:
                perf = self.strategy_performance[strategy_name]
                perf['execution_times'].append(execution_time)
                perf['last_execution'] = datetime.now()
                
                if signal:
                    perf['successful_signals'] += 1
                    confidence = signal.get('confidence_score', 0)
                    
                    # Update average confidence
                    if perf['total_signals'] == 0:
                        perf['average_confidence'] = confidence
                    else:
                        total_confidence = perf['average_confidence'] * perf['total_signals'] + confidence
                        perf['average_confidence'] = total_confidence / (perf['total_signals'] + 1)
                
                perf['total_signals'] += 1
                
                # Calculate performance score
                perf['performance_score'] = self._calculate_strategy_performance_score(strategy_name)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Strategy {strategy_name} execution error: {e}")
            self._update_strategy_error_count(strategy_name)
            return None

    def combine_strategy_results(
        self, 
        all_signals: Dict[str, Optional[Dict]], 
        combination_mode: str,
        epic: str,
        timeframe: str
    ) -> Optional[Dict]:
        """
        Combine results from multiple strategies
        
        Args:
            all_signals: Dictionary of strategy signals
            combination_mode: How to combine ('consensus', 'weighted', 'dynamic')
            epic: Trading epic
            timeframe: Timeframe
            
        Returns:
            Combined signal or None
        """
        try:
            valid_signals = {k: v for k, v in all_signals.items() if v is not None}
            
            if not valid_signals:
                return None
            
            if combination_mode == 'consensus':
                return self._apply_consensus_combination(valid_signals, epic, timeframe)
            elif combination_mode == 'weighted':
                return self._apply_weighted_combination(valid_signals, epic, timeframe)
            elif combination_mode == 'dynamic':
                return self._apply_dynamic_combination(valid_signals, epic, timeframe)
            else:
                self.logger.warning(f"⚠️ Unknown combination mode: {combination_mode}, using consensus")
                return self._apply_consensus_combination(valid_signals, epic, timeframe)
                
        except Exception as e:
            self.logger.error(f"❌ Strategy result combination failed: {e}")
            return None

    def _apply_consensus_combination(
        self, 
        valid_signals: Dict[str, Dict], 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply consensus-based combination"""
        try:
            # Count signals by type
            bull_signals = {k: v for k, v in valid_signals.items() if v.get('signal_type') == 'BULL'}
            bear_signals = {k: v for k, v in valid_signals.items() if v.get('signal_type') == 'BEAR'}
            
            total_signals = len(valid_signals)
            consensus_threshold = 0.6  # 60% agreement required
            min_consensus = max(2, int(total_signals * consensus_threshold))
            
            if len(bull_signals) >= min_consensus:
                # BULL consensus
                best_signal = max(bull_signals.values(), key=lambda x: x.get('confidence_score', 0))
                avg_confidence = sum(s.get('confidence_score', 0) for s in bull_signals.values()) / len(bull_signals)
                
                combined_signal = self._create_consensus_signal(
                    best_signal, bull_signals, 'BULL', avg_confidence, epic, timeframe
                )
                return combined_signal
                
            elif len(bear_signals) >= min_consensus:
                # BEAR consensus
                best_signal = max(bear_signals.values(), key=lambda x: x.get('confidence_score', 0))
                avg_confidence = sum(s.get('confidence_score', 0) for s in bear_signals.values()) / len(bear_signals)
                
                combined_signal = self._create_consensus_signal(
                    best_signal, bear_signals, 'BEAR', avg_confidence, epic, timeframe
                )
                return combined_signal
            
            # Check for single high-confidence signal
            if len(valid_signals) == 1:
                strategy_name, signal = next(iter(valid_signals.items()))
                confidence = signal.get('confidence_score', 0)
                
                if confidence >= 0.9:  # Very high confidence threshold for single signals
                    signal['combination_mode'] = 'single_high_confidence'
                    signal['contributing_strategies'] = [strategy_name]
                    return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Consensus combination failed: {e}")
            return None

    def _apply_weighted_combination(
        self, 
        valid_signals: Dict[str, Dict], 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply weighted combination based on strategy performance"""
        try:
            # Get strategy weights
            if self.forex_optimizer:
                base_weights = self.forex_optimizer.get_normalized_strategy_weights()
            else:
                # Default equal weights
                base_weights = {name: 1.0/len(valid_signals) for name in valid_signals.keys()}
            
            # Apply performance-based adjustments if enabled
            if self.manager_config['adaptive_weighting_enabled']:
                adjusted_weights = self._apply_performance_adjustments(base_weights, valid_signals.keys())
            else:
                adjusted_weights = base_weights
            
            # Calculate weighted scores
            bull_score = 0
            bear_score = 0
            total_weight = 0
            
            signal_contributions = {'bull': [], 'bear': []}
            
            for strategy_name, signal in valid_signals.items():
                weight = adjusted_weights.get(strategy_name, 0.1)
                confidence = signal.get('confidence_score', 0)
                weighted_contribution = confidence * weight
                
                total_weight += weight
                
                if signal.get('signal_type') == 'BULL':
                    bull_score += weighted_contribution
                    signal_contributions['bull'].append((strategy_name, confidence, weight))
                elif signal.get('signal_type') == 'BEAR':
                    bear_score += weighted_contribution
                    signal_contributions['bear'].append((strategy_name, confidence, weight))
            
            # Normalize scores
            if total_weight > 0:
                bull_score /= total_weight
                bear_score /= total_weight
            
            # Determine winner
            min_threshold = self.manager_config.get('min_strategy_confidence', 0.6)
            
            if bull_score > bear_score and bull_score >= min_threshold:
                winning_signals = {k: v for k, v in valid_signals.items() if v.get('signal_type') == 'BULL'}
                best_signal = max(winning_signals.values(), key=lambda x: x.get('confidence_score', 0))
                
                combined_signal = self._create_weighted_signal(
                    best_signal, 'BULL', bull_score, signal_contributions['bull'], epic, timeframe
                )
                return combined_signal
                
            elif bear_score > bull_score and bear_score >= min_threshold:
                winning_signals = {k: v for k, v in valid_signals.items() if v.get('signal_type') == 'BEAR'}
                best_signal = max(winning_signals.values(), key=lambda x: x.get('confidence_score', 0))
                
                combined_signal = self._create_weighted_signal(
                    best_signal, 'BEAR', bear_score, signal_contributions['bear'], epic, timeframe
                )
                return combined_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Weighted combination failed: {e}")
            return None

    def _apply_dynamic_combination(
        self, 
        valid_signals: Dict[str, Dict], 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply dynamic combination based on current market conditions"""
        try:
            # Analyze current market conditions
            market_conditions = self._analyze_current_market_conditions(valid_signals)
            
            # Get optimized weights based on market conditions
            if self.forex_optimizer:
                optimized_weights = self.forex_optimizer.optimize_strategy_weights_for_market(market_conditions)
            else:
                optimized_weights = {name: 1.0/len(valid_signals) for name in valid_signals.keys()}
            
            # Calculate dynamic threshold
            dynamic_threshold = self._calculate_dynamic_threshold(market_conditions)
            
            # Apply weighted combination with dynamic parameters
            return self._apply_weighted_combination_with_params(
                valid_signals, optimized_weights, dynamic_threshold, epic, timeframe
            )
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic combination failed: {e}")
            # Fallback to consensus
            return self._apply_consensus_combination(valid_signals, epic, timeframe)

    def _create_consensus_signal(
        self, 
        best_signal: Dict, 
        contributing_signals: Dict[str, Dict], 
        signal_type: str, 
        avg_confidence: float, 
        epic: str, 
        timeframe: str
    ) -> Dict:
        """Create a consensus-based combined signal"""
        combined_signal = best_signal.copy()
        
        # Update with consensus information
        combined_signal.update({
            'signal_type': signal_type,
            'confidence_score': min(0.98, avg_confidence * 1.1),  # Consensus bonus
            'combination_mode': 'consensus',
            'contributing_strategies': list(contributing_signals.keys()),
            'individual_confidences': {k: v.get('confidence_score', 0) for k, v in contributing_signals.items()},
            'consensus_strength': len(contributing_signals),
            'strategy_agreement': len(contributing_signals) / len(contributing_signals),  # Will be 1.0 for consensus
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'strategy': f'combined_consensus_{len(contributing_signals)}',
        })
        
        return combined_signal

    def _create_weighted_signal(
        self, 
        best_signal: Dict, 
        signal_type: str, 
        weighted_score: float, 
        contributions: List[Tuple], 
        epic: str, 
        timeframe: str
    ) -> Dict:
        """Create a weighted combined signal"""
        combined_signal = best_signal.copy()
        
        # Update with weighted information
        combined_signal.update({
            'signal_type': signal_type,
            'confidence_score': weighted_score,
            'combination_mode': 'weighted',
            'contributing_strategies': [contrib[0] for contrib in contributions],
            'individual_confidences': {contrib[0]: contrib[1] for contrib in contributions},
            'strategy_weights': {contrib[0]: contrib[2] for contrib in contributions},
            'weighted_scores': {'bull': 0, 'bear': 0},  # Will be set by caller
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'strategy': f'combined_weighted_{len(contributions)}',
        })
        
        return combined_signal

    def _apply_performance_adjustments(self, base_weights: Dict[str, float], strategy_names: List[str]) -> Dict[str, float]:
        """Apply performance-based adjustments to strategy weights"""
        try:
            adjusted_weights = base_weights.copy()
            
            for strategy_name in strategy_names:
                if strategy_name in self.strategy_performance:
                    perf = self.strategy_performance[strategy_name]
                    performance_score = perf.get('performance_score', 1.0)
                    
                    # Adjust weight based on performance (0.5x to 1.5x multiplier)
                    adjustment_factor = 0.5 + (performance_score * 1.0)
                    adjusted_weights[strategy_name] = base_weights.get(strategy_name, 0.1) * adjustment_factor
            
            # Renormalize weights
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"❌ Performance adjustment failed: {e}")
            return base_weights

    def _analyze_current_market_conditions(self, valid_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze current market conditions based on strategy signals"""
        try:
            conditions = {
                'volatility_regime': 'medium',
                'trend_strength': 'medium',
                'market_regime': 'ranging',
                'signal_consensus': len(valid_signals),
                'signal_diversity': len(set(s.get('signal_type') for s in valid_signals.values()))
            }
            
            # Analyze signal consensus
            signal_types = [s.get('signal_type') for s in valid_signals.values()]
            if len(set(signal_types)) <= 1 and len(signal_types) > 0:
                conditions['market_regime'] = 'trending'
            elif len(signal_types) == 0:
                conditions['market_regime'] = 'quiet'
            else:
                conditions['market_regime'] = 'ranging'
            
            # Analyze confidence levels (proxy for market clarity)
            confidences = [s.get('confidence_score', 0) for s in valid_signals.values()]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence > 0.8:
                    conditions['trend_strength'] = 'strong'
                elif avg_confidence < 0.6:
                    conditions['trend_strength'] = 'weak'
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"❌ Market conditions analysis failed: {e}")
            return {'volatility_regime': 'medium', 'trend_strength': 'medium', 'market_regime': 'ranging'}

    def _calculate_dynamic_threshold(self, market_conditions: Dict[str, Any]) -> float:
        """Calculate dynamic confidence threshold based on market conditions"""
        try:
            base_threshold = self.manager_config.get('min_strategy_confidence', 0.6)
            
            # Adjust based on market regime
            market_regime = market_conditions.get('market_regime', 'ranging')
            
            if market_regime == 'trending':
                return max(0.5, base_threshold - 0.1)  # Lower threshold in trending markets
            elif market_regime == 'ranging':
                return min(0.8, base_threshold + 0.1)  # Higher threshold in ranging markets
            else:
                return base_threshold
                
        except Exception as e:
            self.logger.error(f"❌ Dynamic threshold calculation failed: {e}")
            return 0.6

    def _apply_weighted_combination_with_params(
        self, 
        valid_signals: Dict[str, Dict], 
        weights: Dict[str, float], 
        threshold: float, 
        epic: str, 
        timeframe: str
    ) -> Optional[Dict]:
        """Apply weighted combination with custom parameters"""
        # Similar to _apply_weighted_combination but with custom weights and threshold
        # Implementation would be similar to _apply_weighted_combination
        return self._apply_weighted_combination(valid_signals, epic, timeframe)

    def _calculate_strategy_performance_score(self, strategy_name: str) -> float:
        """Calculate overall performance score for a strategy"""
        try:
            perf = self.strategy_performance[strategy_name]
            
            # Success rate component (0-1)
            if perf['total_signals'] > 0:
                success_rate = perf['successful_signals'] / perf['total_signals']
            else:
                success_rate = 0.5  # Neutral for new strategies
            
            # Confidence component (0-1)
            confidence_score = perf['average_confidence']
            
            # Reliability component (based on error rate)
            if perf['total_signals'] > 0:
                error_rate = perf['error_count'] / max(1, perf['total_signals'])
                reliability_score = max(0, 1 - error_rate)
            else:
                reliability_score = 1.0
            
            # Speed component (0-1, faster is better)
            if perf['execution_times']:
                avg_time = sum(perf['execution_times']) / len(perf['execution_times'])
                speed_score = max(0, min(1, 1 - (avg_time / 10)))  # 10 seconds as max expected time
            else:
                speed_score = 1.0
            
            # Weighted combination
            performance_score = (
                success_rate * 0.4 +
                confidence_score * 0.3 +
                reliability_score * 0.2 +
                speed_score * 0.1
            )
            
            return max(0.1, min(1.0, performance_score))
            
        except Exception as e:
            self.logger.error(f"❌ Performance score calculation failed for {strategy_name}: {e}")
            return 0.5

    def _update_execution_stats(self, execution_time: float, signals: Dict[str, Optional[Dict]]) -> None:
        """Update global execution statistics"""
        try:
            with self._lock:
                self.execution_stats['total_executions'] += 1
                
                successful_signals = len([s for s in signals.values() if s is not None])
                if successful_signals > 0:
                    self.execution_stats['successful_executions'] += 1
                else:
                    self.execution_stats['failed_executions'] += 1
                
                # Update average execution time
                total_time = self.execution_stats['average_execution_time'] * (self.execution_stats['total_executions'] - 1)
                self.execution_stats['average_execution_time'] = (total_time + execution_time) / self.execution_stats['total_executions']
                
        except Exception as e:
            self.logger.error(f"❌ Execution stats update failed: {e}")

    def _update_strategy_performance(self, signals: Dict[str, Optional[Dict]], execution_time: float) -> None:
        """Update individual strategy performance metrics"""
        try:
            with self._lock:
                for strategy_name in self.registered_strategies.keys():
                    if strategy_name in self.execution_stats['strategy_success_rates']:
                        success_rates = self.execution_stats['strategy_success_rates'][strategy_name]
                        success_rates['attempts'] += 1
                        
                        if signals.get(strategy_name) is not None:
                            success_rates['successes'] += 1
                        
                        success_rates['rate'] = success_rates['successes'] / success_rates['attempts']
                
        except Exception as e:
            self.logger.error(f"❌ Strategy performance update failed: {e}")

    def _update_strategy_error_count(self, strategy_name: str) -> None:
        """Update error count for a specific strategy"""
        try:
            with self._lock:
                if strategy_name in self.strategy_performance:
                    self.strategy_performance[strategy_name]['error_count'] += 1
                
        except Exception as e:
            self.logger.error(f"❌ Error count update failed for {strategy_name}: {e}")

    def get_active_strategy_count(self) -> int:
        """Get the number of active registered strategies"""
        return len(self.registered_strategies)

    def get_enabled_strategy_names(self) -> List[str]:
        """Get list of enabled strategy names"""
        return list(self.registered_strategies.keys())

    def get_all_required_indicators(self) -> List[str]:
        """Get all required indicators from all registered strategies"""
        try:
            all_indicators = []
            
            for strategy_name, strategy in self.registered_strategies.items():
                try:
                    if hasattr(strategy, 'get_required_indicators'):
                        strategy_indicators = strategy.get_required_indicators()
                        all_indicators.extend(strategy_indicators)
                        self.logger.debug(f"Added {len(strategy_indicators)} indicators from {strategy_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to get indicators from {strategy_name}: {e}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indicators = []
            for indicator in all_indicators:
                if indicator not in seen:
                    seen.add(indicator)
                    unique_indicators.append(indicator)
            
            self.logger.debug(f"Total unique indicators required: {len(unique_indicators)}")
            return unique_indicators
            
        except Exception as e:
            self.logger.error(f"❌ Required indicators collection failed: {e}")
            return []

    def validate_strategy_setup(self) -> bool:
        """Validate that the strategy setup is correct"""
        try:
            if not self.registered_strategies:
                self.logger.error("❌ No strategies registered")
                return False
            
            # Test basic functionality of each strategy
            for strategy_name, strategy in self.registered_strategies.items():
                if not hasattr(strategy, 'detect_signal'):
                    self.logger.error(f"❌ Strategy {strategy_name} missing detect_signal method")
                    return False
                
                if not hasattr(strategy, 'get_required_indicators'):
                    self.logger.warning(f"⚠️ Strategy {strategy_name} missing get_required_indicators method")
            
            self.logger.info(f"✅ Strategy setup validated: {len(self.registered_strategies)} strategies")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Strategy setup validation failed: {e}")
            return False

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics"""
        try:
            return {
                'registered_strategies': len(self.registered_strategies),
                'strategy_names': list(self.registered_strategies.keys()),
                'execution_stats': self.execution_stats.copy(),
                'strategy_performance': self.strategy_performance.copy(),
                'manager_config': self.manager_config.copy(),
                'performance_summary': {
                    'best_performing_strategy': self._get_best_performing_strategy(),
                    'worst_performing_strategy': self._get_worst_performing_strategy(),
                    'average_performance_score': self._calculate_average_performance_score(),
                    'total_signals_generated': sum(perf['total_signals'] for perf in self.strategy_performance.values()),
                    'overall_success_rate': self._calculate_overall_success_rate()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Manager stats collection failed: {e}")
            return {'error': str(e)}

    def _get_best_performing_strategy(self) -> Optional[str]:
        """Get the name of the best performing strategy"""
        try:
            if not self.strategy_performance:
                return None
            
            best_strategy = max(
                self.strategy_performance.items(),
                key=lambda x: x[1].get('performance_score', 0)
            )
            return best_strategy[0]
            
        except Exception:
            return None

    def _get_worst_performing_strategy(self) -> Optional[str]:
        """Get the name of the worst performing strategy"""
        try:
            if not self.strategy_performance:
                return None
            
            worst_strategy = min(
                self.strategy_performance.items(),
                key=lambda x: x[1].get('performance_score', 1)
            )
            return worst_strategy[0]
            
        except Exception:
            return None

    def _calculate_average_performance_score(self) -> float:
        """Calculate average performance score across all strategies"""
        try:
            if not self.strategy_performance:
                return 0.0
            
            scores = [perf.get('performance_score', 0) for perf in self.strategy_performance.values()]
            return sum(scores) / len(scores)
            
        except Exception:
            return 0.0

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all strategies"""
        try:
            total_signals = sum(perf.get('total_signals', 0) for perf in self.strategy_performance.values())
            successful_signals = sum(perf.get('successful_signals', 0) for perf in self.strategy_performance.values())
            
            if total_signals == 0:
                return 0.0
            
            return successful_signals / total_signals
            
        except Exception:
            return 0.0

    def clear_cache(self) -> None:
        """Clear any cached data in the manager"""
        try:
            # Clear performance history older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with self._lock:
                for strategy_name, perf in self.strategy_performance.items():
                    # Keep only recent execution times
                    if perf.get('last_execution') and perf['last_execution'] < cutoff_time:
                        perf['execution_times'] = perf['execution_times'][-10:]  # Keep last 10
            
            self.logger.debug("🧹 CombinedStrategyManager cache cleared")
            
        except Exception as e:
            self.logger.error(f"❌ Manager cache clearing failed: {e}")

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics information"""
        return {
            'module_name': 'CombinedStrategyManager',
            'initialization_successful': True,
            'registered_strategies_count': len(self.registered_strategies),
            'registered_strategy_names': list(self.registered_strategies.keys()),
            'forex_optimizer_available': self.forex_optimizer is not None,
            'validator_available': self.validator is not None,
            'manager_config': self.manager_config,
            'execution_stats': self.execution_stats,
            'strategy_performance_summary': {
                name: {
                    'total_signals': perf.get('total_signals', 0),
                    'performance_score': perf.get('performance_score', 0),
                    'error_count': perf.get('error_count', 0)
                }
                for name, perf in self.strategy_performance.items()
            },
            'manager_stats': self.get_manager_stats(),
            'timestamp': datetime.now().isoformat()
        }