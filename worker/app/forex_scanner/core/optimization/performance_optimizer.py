# core/optimization/performance_optimizer.py
"""
Updated Performance Optimization Engine
Compatible with new modular architecture and Market Intelligence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import json
from datetime import datetime, timedelta
import copy

try:
    import config
except ImportError:
    from forex_scanner import config


class ModernPerformanceOptimizer:
    """Updated performance optimizer compatible with new architecture"""
    
    def __init__(self, signal_detector, alert_history_manager=None):
        self.signal_detector = signal_detector
        self.alert_history = alert_history_manager
        self.logger = logging.getLogger(__name__)
        self.optimization_cache = {}
        
        # Verify required components exist
        if not hasattr(signal_detector, 'backtest_engine'):
            self.logger.warning("⚠️ Backtest engine not found - some features unavailable")
        if not hasattr(signal_detector, 'performance_analyzer'):
            self.logger.warning("⚠️ Performance analyzer not found - using basic metrics")
    
    def optimize_ema_configurations(
        self, 
        epic_list: List[str],
        days: int = 30,
        optimization_metric: str = 'efficiency_score'
    ) -> Dict:
        """
        Optimize EMA configurations using new EMA strategy system
        
        Args:
            epic_list: List of epics to test
            days: Days of historical data
            optimization_metric: 'efficiency_score', 'win_rate', 'profit_factor'
        """
        self.logger.info(f"🔧 Optimizing EMA configurations for {len(epic_list)} pairs")
        
        # Available EMA configurations from config
        available_configs = getattr(config, 'EMA_STRATEGY_CONFIG', {
            'default': {'short': 9, 'long': 21, 'trend': 200},
            'aggressive': {'short': 5, 'long': 13, 'trend': 100},
            'conservative': {'short': 12, 'long': 26, 'trend': 200},
            'scalping': {'short': 3, 'long': 8, 'trend': 50},
            'swing': {'short': 21, 'long': 50, 'trend': 200}
        })
        
        optimization_results = []
        
        for epic in epic_list[:3]:  # Limit to 3 pairs for reasonable execution time
            self.logger.info(f"📊 Testing {epic}...")
            
            epic_results = {
                'epic': epic,
                'config_scores': {},
                'best_config': None,
                'best_score': -1.0
            }
            
            for config_name, ema_params in available_configs.items():
                self.logger.info(f"  Testing {config_name}: {ema_params}")
                
                try:
                    # Test this configuration
                    score, metrics = self._test_ema_configuration(
                        epic, config_name, ema_params, days
                    )
                    
                    epic_results['config_scores'][config_name] = {
                        'score': score,
                        'metrics': metrics
                    }
                    
                    if score > epic_results['best_score']:
                        epic_results['best_score'] = score
                        epic_results['best_config'] = config_name
                    
                    self.logger.info(f"    {config_name}: {score:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"    Error testing {config_name}: {e}")
                    epic_results['config_scores'][config_name] = {
                        'score': -1.0,
                        'error': str(e)
                    }
            
            optimization_results.append(epic_results)
        
        # Analyze results across all pairs
        recommendations = self._analyze_ema_optimization_results(optimization_results)
        
        return {
            'results_by_epic': optimization_results,
            'recommendations': recommendations,
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def _test_ema_configuration(
        self, 
        epic: str, 
        config_name: str, 
        ema_params: Dict,
        days: int
    ) -> Tuple[float, Dict]:
        """Test a specific EMA configuration"""
        try:
            # Use backtest engine if available
            if hasattr(self.signal_detector, 'backtest_engine'):
                results = self.signal_detector.backtest_engine.backtest_epic_time_range(
                    epic=epic,
                    lookback_days=days,
                    timeframe='15m',  # Use 15m for optimization
                    ema_config_override=config_name
                )
                
                if hasattr(self.signal_detector, 'performance_analyzer'):
                    metrics = self.signal_detector.performance_analyzer.analyze_performance(results)
                else:
                    metrics = self._calculate_basic_metrics(results)
                
                score = self._calculate_optimization_score(metrics)
                
            else:
                # Fallback: Use alert history if available
                if self.alert_history:
                    score, metrics = self._test_config_from_alert_history(
                        epic, config_name, days
                    )
                else:
                    # Last resort: synthetic testing
                    score, metrics = self._synthetic_config_test(epic, ema_params, days)
            
            return score, metrics
            
        except Exception as e:
            self.logger.error(f"Error testing {config_name} for {epic}: {e}")
            return -1.0, {'error': str(e)}
    
    def _test_config_from_alert_history(
        self, 
        epic: str, 
        config_name: str, 
        days: int
    ) -> Tuple[float, Dict]:
        """Test configuration using alert history data"""
        try:
            # Get recent alerts for this epic with this config
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # This would require alert history to track EMA config used
            # For now, return basic score
            self.logger.info(f"Testing {config_name} using alert history (basic)")
            
            return 0.5, {'method': 'alert_history', 'signals': 0}
            
        except Exception as e:
            return -1.0, {'error': str(e)}
    
    def _synthetic_config_test(
        self, 
        epic: str, 
        ema_params: Dict, 
        days: int
    ) -> Tuple[float, Dict]:
        """Synthetic testing when backtest engine unavailable"""
        try:
            # Get recent data
            pair_info = config.PAIR_INFO.get(epic, {'pair': 'EURUSD'})
            pair = pair_info['pair']
            
            df = self.signal_detector.data_fetcher.get_enhanced_data(
                epic, pair, timeframe='15m', lookback_hours=days * 24
            )
            
            if df is None or len(df) < 100:
                return -1.0, {'error': 'Insufficient data'}
            
            # Calculate simple metrics based on EMA alignment
            short_ema = f"ema_{ema_params['short']}"
            long_ema = f"ema_{ema_params['long']}"
            trend_ema = f"ema_{ema_params['trend']}"
            
            # Add EMAs if missing
            if short_ema not in df.columns:
                df[short_ema] = df['close'].ewm(span=ema_params['short']).mean()
            if long_ema not in df.columns:
                df[long_ema] = df['close'].ewm(span=ema_params['long']).mean()
            if trend_ema not in df.columns:
                df[trend_ema] = df['close'].ewm(span=ema_params['trend']).mean()
            
            # Calculate alignment score
            latest = df.iloc[-1]
            alignment_score = self._calculate_ema_alignment_score(
                latest['close'], latest[short_ema], 
                latest[long_ema], latest[trend_ema]
            )
            
            # Calculate trend consistency
            trend_score = self._calculate_trend_consistency(df, ema_params)
            
            # Combined score
            score = (alignment_score * 0.6 + trend_score * 0.4)
            
            metrics = {
                'method': 'synthetic',
                'alignment_score': alignment_score,
                'trend_score': trend_score,
                'data_points': len(df)
            }
            
            return score, metrics
            
        except Exception as e:
            self.logger.error(f"Synthetic test error: {e}")
            return -1.0, {'error': str(e)}
    
    def _calculate_ema_alignment_score(
        self, 
        price: float, 
        short_ema: float, 
        long_ema: float, 
        trend_ema: float
    ) -> float:
        """Calculate how well EMAs are aligned"""
        try:
            # Perfect bull alignment: price > short > long > trend
            # Perfect bear alignment: price < short < long < trend
            
            if price > short_ema > long_ema > trend_ema:
                return 1.0  # Perfect bull alignment
            elif price < short_ema < long_ema < trend_ema:
                return 1.0  # Perfect bear alignment
            elif short_ema > long_ema > trend_ema:
                return 0.8  # Good bull alignment, price might be pulling back
            elif short_ema < long_ema < trend_ema:
                return 0.8  # Good bear alignment
            elif short_ema > long_ema:
                return 0.6  # Partial bull alignment
            elif short_ema < long_ema:
                return 0.6  # Partial bear alignment
            else:
                return 0.3  # Mixed signals
                
        except:
            return 0.0
    
    def _calculate_trend_consistency(self, df: pd.DataFrame, ema_params: Dict) -> float:
        """Calculate trend consistency over time"""
        try:
            short_col = f"ema_{ema_params['short']}"
            long_col = f"ema_{ema_params['long']}"
            
            if short_col not in df.columns or long_col not in df.columns:
                return 0.5
            
            # Calculate how often short EMA is above long EMA (bull trend)
            bull_periods = (df[short_col] > df[long_col]).sum()
            total_periods = len(df)
            
            # Consistency is measured as how close we are to 100% or 0%
            # (strong trends are either consistently bull or bear)
            bull_ratio = bull_periods / total_periods
            consistency = max(bull_ratio, 1 - bull_ratio)
            
            return consistency
            
        except:
            return 0.5
    
    def _calculate_basic_metrics(self, results: List[Dict]) -> Dict:
        """Calculate basic metrics when performance analyzer unavailable"""
        if not results:
            return {
                'total_signals': 0,
                'win_rate': 0,
                'average_profit_pips': 0,
                'average_loss_pips': 0,
                'profit_factor': 0
            }
        
        total_signals = len(results)
        winning_signals = sum(1 for r in results if r.get('profit_pips', 0) > 0)
        
        profits = [r.get('profit_pips', 0) for r in results if r.get('profit_pips', 0) > 0]
        losses = [abs(r.get('profit_pips', 0)) for r in results if r.get('profit_pips', 0) < 0]
        
        return {
            'total_signals': total_signals,
            'win_rate': winning_signals / total_signals if total_signals > 0 else 0,
            'average_profit_pips': np.mean(profits) if profits else 0,
            'average_loss_pips': np.mean(losses) if losses else 0,
            'profit_factor': sum(profits) / sum(losses) if losses else 0
        }
    
    def _calculate_optimization_score(self, metrics: Dict) -> float:
        """Calculate optimization score from metrics"""
        if not metrics or metrics.get('total_signals', 0) == 0:
            return -1.0
        
        # Handle different metric types
        if 'method' in metrics:
            # Synthetic metrics
            if metrics['method'] == 'synthetic':
                return metrics.get('alignment_score', 0) * 0.6 + metrics.get('trend_score', 0) * 0.4
            elif metrics['method'] == 'alert_history':
                return 0.5  # Basic score for alert history method
        
        # Standard backtest metrics
        win_rate = metrics.get('win_rate', 0)
        avg_profit = metrics.get('average_profit_pips', 0)
        avg_loss = abs(metrics.get('average_loss_pips', 1))
        total_signals = metrics.get('total_signals', 0)
        
        if avg_loss == 0:
            avg_loss = 1  # Prevent division by zero
        
        # Profit factor
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0
        
        # Sharpe-like ratio
        sharpe = win_rate * avg_profit / (avg_loss * (1 - win_rate)) if (1 - win_rate) > 0 else 0
        
        # Signal frequency bonus (but not too many)
        frequency_bonus = min(1.0, total_signals / 20.0)  # Optimal around 20 signals
        
        # Combined score
        score = (
            win_rate * 0.3 +           # 30% weight on win rate
            profit_factor * 0.25 +     # 25% weight on profit factor  
            min(sharpe, 2.0) * 0.25 +  # 25% weight on sharpe (capped at 2.0)
            frequency_bonus * 0.2      # 20% weight on signal frequency
        )
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _analyze_ema_optimization_results(self, results: List[Dict]) -> Dict:
        """Analyze optimization results and generate recommendations"""
        
        # Count best configurations across all pairs
        config_scores = {}
        config_counts = {}
        
        for epic_result in results:
            best_config = epic_result.get('best_config')
            best_score = epic_result.get('best_score', 0)
            
            if best_config and best_score > 0:
                if best_config not in config_scores:
                    config_scores[best_config] = []
                    config_counts[best_config] = 0
                
                config_scores[best_config].append(best_score)
                config_counts[best_config] += 1
        
        # Find most consistently good configuration
        best_overall_config = None
        best_overall_score = 0
        
        for config_name, scores in config_scores.items():
            avg_score = np.mean(scores)
            consistency = config_counts[config_name] / len(results)
            
            # Combined score: average performance + consistency
            overall_score = avg_score * 0.7 + consistency * 0.3
            
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                best_overall_config = config_name
        
        recommendations = {
            'recommended_ema_config': best_overall_config,
            'confidence': best_overall_score,
            'config_performance': {
                config: {
                    'average_score': np.mean(scores),
                    'pairs_won': counts,
                    'consistency': counts / len(results)
                }
                for config, scores, counts in zip(
                    config_scores.keys(), 
                    config_scores.values(),
                    config_counts.values()
                )
            },
            'immediate_actions': []
        }
        
        # Generate action recommendations
        if best_overall_config and best_overall_score > 0.6:
            current_config = getattr(config, 'ACTIVE_EMA_CONFIG', 'default')
            if best_overall_config != current_config:
                recommendations['immediate_actions'].append(
                    f"Switch from '{current_config}' to '{best_overall_config}' EMA configuration"
                )
        
        if best_overall_score < 0.4:
            recommendations['immediate_actions'].append(
                "Consider developing new EMA configurations - current ones underperforming"
            )
        
        return recommendations
    
    def optimize_confidence_thresholds_modern(
        self,
        epic_list: List[str],
        days: int = 30
    ) -> Dict:
        """Modern confidence threshold optimization using alert history"""
        self.logger.info("🎯 Optimizing confidence thresholds using alert history...")
        
        if not self.alert_history:
            self.logger.warning("⚠️ Alert history not available - skipping threshold optimization")
            return {'error': 'Alert history manager not available'}
        
        try:
            # Get recent alerts
            recent_alerts = self.alert_history.get_recent_alerts(hours=days * 24)
            
            if not recent_alerts:
                return {'error': 'No recent alerts found for optimization'}
            
            # Analyze threshold performance
            threshold_analysis = self._analyze_threshold_performance(recent_alerts)
            
            return {
                'current_performance': threshold_analysis,
                'recommendations': self._generate_threshold_recommendations(threshold_analysis),
                'optimization_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing confidence thresholds: {e}")
            return {'error': str(e)}
    
    def _analyze_threshold_performance(self, alerts: List[Dict]) -> Dict:
        """Analyze how different confidence levels perform"""
        
        # Group alerts by confidence ranges
        confidence_ranges = {
            'very_high': [],  # >0.9
            'high': [],       # 0.8-0.9
            'medium': [],     # 0.6-0.8
            'low': []         # <0.6
        }
        
        for alert in alerts:
            confidence = alert.get('confidence_score', 0)
            
            if confidence > 0.9:
                confidence_ranges['very_high'].append(alert)
            elif confidence > 0.8:
                confidence_ranges['high'].append(alert)
            elif confidence > 0.6:
                confidence_ranges['medium'].append(alert)
            else:
                confidence_ranges['low'].append(alert)
        
        # Calculate performance by range
        performance_by_range = {}
        for range_name, range_alerts in confidence_ranges.items():
            if range_alerts:
                performance_by_range[range_name] = {
                    'count': len(range_alerts),
                    'avg_confidence': np.mean([a.get('confidence_score', 0) for a in range_alerts]),
                    'strategies': list(set(a.get('strategy', 'unknown') for a in range_alerts))
                }
            else:
                performance_by_range[range_name] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'strategies': []
                }
        
        return performance_by_range
    
    def _generate_threshold_recommendations(self, analysis: Dict) -> Dict:
        """Generate threshold recommendations based on analysis"""
        
        recommendations = {
            'suggested_thresholds': {},
            'actions': []
        }
        
        # Analyze signal distribution
        total_signals = sum(data['count'] for data in analysis.values())
        
        if total_signals == 0:
            return recommendations
        
        # Calculate optimal thresholds based on signal quality vs quantity
        very_high_count = analysis['very_high']['count']
        high_count = analysis['high']['count']
        medium_count = analysis['medium']['count']
        
        if very_high_count > total_signals * 0.1:  # If >10% very high confidence
            recommendations['suggested_thresholds']['primary'] = 0.85
            recommendations['actions'].append("Raise threshold to 0.85 - plenty of high-quality signals")
        elif high_count > total_signals * 0.3:     # If >30% high confidence
            recommendations['suggested_thresholds']['primary'] = 0.75
            recommendations['actions'].append("Use 0.75 threshold - good balance of quality vs quantity")
        elif medium_count > total_signals * 0.5:   # If >50% medium confidence
            recommendations['suggested_thresholds']['primary'] = 0.65
            recommendations['actions'].append("Use 0.65 threshold - focus on signal frequency")
        else:
            recommendations['suggested_thresholds']['primary'] = 0.6
            recommendations['actions'].append("Use 0.6 threshold - maximize signal capture")
        
        return recommendations


# Factory function for easy creation
def create_performance_optimizer(signal_detector, alert_history_manager=None):
    """Create a modern performance optimizer"""
    return ModernPerformanceOptimizer(signal_detector, alert_history_manager)


# Usage example
def main():
    """Example usage of updated performance optimizer"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        from core.database import DatabaseManager
        from core.signal_detector import SignalDetector
        from core.alerts.alert_history import AlertHistoryManager
        
        # Initialize components
        db_manager = DatabaseManager(config.DATABASE_URL)
        signal_detector = SignalDetector(db_manager, config.USER_TIMEZONE)
        alert_history = AlertHistoryManager(db_manager)
        
        # Create optimizer
        optimizer = create_performance_optimizer(signal_detector, alert_history)
        
        # Run optimization
        results = optimizer.optimize_ema_configurations(
            epic_list=config.EPIC_LIST[:2],  # Test first 2 pairs
            days=14  # 2 weeks of data
        )
        
        print("🔧 Optimization Results:")
        print(f"Recommended EMA Config: {results['recommendations']['recommended_ema_config']}")
        print(f"Confidence: {results['recommendations']['confidence']:.1%}")
        
        for action in results['recommendations']['immediate_actions']:
            print(f"Action: {action}")
        
    except Exception as e:
        print(f"❌ Error running optimizer: {e}")


if __name__ == "__main__":
    main()