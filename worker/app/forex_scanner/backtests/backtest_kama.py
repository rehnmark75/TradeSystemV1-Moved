#!/usr/bin/env python3
"""
KAMA Strategy Backtest - Standalone Module
Run: python backtest_kama.py --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))

# If we're in backtests/ subdirectory, go up one level to project root
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    # If we're in project root
    project_root = script_dir

sys.path.insert(0, project_root)

from core.database import DatabaseManager
from core.data_fetcher import DataFetcher
from core.strategies.kama_strategy import KAMAStrategy, create_kama_strategy
from performance_analyzer import PerformanceAnalyzer
from signal_analyzer import SignalAnalyzer
try:
    import config
except ImportError:
    from forex_scanner import config


class KAMABacktest:
    """Dedicated KAMA Strategy Backtesting"""
    
    def __init__(self):
        self.logger = logging.getLogger('kama_backtest')
        self.setup_logging()
        
        # Initialize components
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.data_fetcher = DataFetcher(self.db_manager, config.USER_TIMEZONE)
        self.performance_analyzer = PerformanceAnalyzer()
        self.signal_analyzer = SignalAnalyzer()
        self.strategy = None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def initialize_kama_strategy(self, use_modular: bool = True):
        """Initialize KAMA strategy with optional modular architecture"""
        
        if use_modular:
            # Use new modular factory function
            self.strategy = create_kama_strategy(data_fetcher=self.data_fetcher)
            self.logger.info("✅ Modular KAMA Strategy initialized for backtest")
            
            # Validate modular integration
            if hasattr(self.strategy, 'validate_modular_integration'):
                if self.strategy.validate_modular_integration():
                    self.logger.info("   🔧 Modular integration validated successfully")
                else:
                    self.logger.warning("   ⚠️ Modular integration validation failed")
        else:
            # Use legacy direct instantiation
            self.strategy = KAMAStrategy(data_fetcher=self.data_fetcher)
            self.logger.info("✅ Legacy KAMA Strategy initialized for backtest")
        
        # Get KAMA configuration details
        self.logger.info(f"   📊 KAMA Parameters:")
        self.logger.info(f"      ER Period: {getattr(config, 'KAMA_ER_PERIOD', 14)}")
        self.logger.info(f"      Fast SC: {getattr(config, 'KAMA_FAST_SC', 2)}")
        self.logger.info(f"      Slow SC: {getattr(config, 'KAMA_SLOW_SC', 30)}")
        self.logger.info(f"      Min Efficiency: {getattr(config, 'KAMA_MIN_EFFICIENCY', 0.1)}")
        self.logger.info(f"      Trend Threshold: {getattr(config, 'KAMA_TREND_THRESHOLD', 0.05)}")
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        use_modular: bool = True,
        min_confidence: float = None,
        debug_mode: bool = False
    ) -> bool:
        """Run KAMA strategy backtest"""
        
        # Setup epic list
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("🔄 KAMA STRATEGY BACKTEST")
        self.logger.info("=" * 40)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"🏗️ Use modular: {use_modular}")
        self.logger.info(f"🐛 Debug mode: {debug_mode}")
        
        # Override minimum confidence if specified
        original_min_conf = None
        if min_confidence:
            original_min_conf = getattr(config, 'MIN_CONFIDENCE', 0.5)
            config.MIN_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min confidence: {min_confidence:.1%} (was {original_min_conf:.1%})")
        
        try:
            # Initialize strategy
            self.initialize_kama_strategy(use_modular)
            
            all_signals = []
            epic_results = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # Get enhanced data - Need to extract pair from epic
                pair = self._extract_pair_from_epic(current_epic)
                
                df = self.data_fetcher.get_enhanced_data(
                    epic=current_epic,
                    pair=pair,
                    timeframe=timeframe,
                    lookback_hours=days * 24
                )
                
                if df.empty:
                    self.logger.warning(f"❌ No data available for {current_epic}")
                    epic_results[current_epic] = {'signals': 0, 'error': 'No data'}
                    continue
                
                self.logger.info(f"   📊 Data points: {len(df)}")
                self.logger.info(f"   📅 Date range: {df.iloc[0].get('datetime_utc', df.iloc[0].get('start_time', 'Unknown'))} to {df.iloc[-1].get('datetime_utc', df.iloc[-1].get('start_time', 'Unknown'))}")
                
                # Check for KAMA indicators
                kama_indicators = self._check_kama_indicators(df)
                self.logger.info(f"   🔄 KAMA indicators: {kama_indicators}")
                
                # Run KAMA backtest
                signals = self._run_kama_backtest(df, current_epic, timeframe, debug_mode)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                self.logger.info(f"   🎯 KAMA signals found: {len(signals)}")
                
                # Show modular performance stats if available
                if use_modular and hasattr(self.strategy, 'get_performance_stats'):
                    try:
                        perf_stats = self.strategy.get_performance_stats()
                        self._display_modular_stats(perf_stats)
                    except Exception as e:
                        self.logger.debug(f"Could not get performance stats: {e}")
            
            # Display epic-by-epic results
            self._display_epic_results(epic_results)
            
            # Overall analysis
            if all_signals:
                self.logger.info(f"\n✅ TOTAL KAMA SIGNALS: {len(all_signals)}")
                
                # Show individual signals if requested
                if show_signals:
                    self._display_signals(all_signals)
                
                # Performance analysis
                self._analyze_performance(all_signals)
                
                # Show modular component analysis if available
                if use_modular:
                    self._analyze_modular_components(all_signals)
                
                # Restore original config
                if original_min_conf:
                    config.MIN_CONFIDENCE = original_min_conf
                
                return True
            else:
                self.logger.warning("❌ No KAMA signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ KAMA backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_kama_indicators(self, df: pd.DataFrame) -> Dict:
        """Check for KAMA indicators in the DataFrame"""
        kama_cols = [col for col in df.columns if 'kama' in col.lower()]
        er_cols = [col for col in df.columns if 'efficiency' in col.lower() or 'er' in col.lower()]
        
        return {
            'kama_columns': kama_cols,
            'efficiency_ratio_columns': er_cols,
            'indicators_present': len(kama_cols) > 0 and len(er_cols) > 0
        }
    
    def _run_kama_backtest(self, df: pd.DataFrame, epic: str, timeframe: str, debug_mode: bool = False) -> List[Dict]:
        """Run KAMA backtest on historical data"""
        signals = []
        
        # Get minimum bars requirement
        min_bars = getattr(config, 'KAMA_MIN_BARS', 50)
        
        # Process each candle for potential signals
        for i in range(min_bars, len(df)):  # Start from min_bars to ensure enough data
            try:
                # Get data up to current point
                current_data = df.iloc[:i+1].copy()
                
                # Debug mode: Show detailed processing for first few signals
                if debug_mode and len(signals) < 3:
                    self.logger.info(f"🔍 Debug processing candle {i}/{len(df)}")
                    
                    # Check if strategy has debug method
                    if hasattr(self.strategy, 'debug_signal_detection'):
                        debug_info = self.strategy.debug_signal_detection(
                            current_data, epic, config.SPREAD_PIPS, timeframe
                        )
                        self.logger.info(f"   🔧 Debug info: {debug_info}")
                
                # Detect signal using KAMA strategy
                signal = self.strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe
                )
                
                if signal:
                    # Debug: Print signal fields to understand confidence storage
                    if debug_mode:
                        self.logger.info(f"🔍 Raw KAMA signal: {signal}")
                    
                    # Add backtest metadata
                    signal['backtest_timestamp'] = df.iloc[i].get('datetime_utc', df.iloc[i].get('start_time', 'Unknown'))
                    signal['backtest_index'] = i
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close'])
                    }
                    
                    # Ensure timestamp field for analysis
                    if 'timestamp' not in signal:
                        signal['timestamp'] = signal['backtest_timestamp']
                    
                    # Standardize confidence field names
                    confidence_value = signal.get('confidence', signal.get('confidence_score', 0))
                    if confidence_value is not None:
                        signal['confidence'] = confidence_value
                        signal['confidence_score'] = confidence_value
                        
                        if debug_mode:
                            self.logger.info(f"🔍 KAMA confidence: {confidence_value:.3f} ({confidence_value:.1%})")
                    
                    # Add KAMA-specific metrics if available
                    self._add_kama_metrics(signal, df, i)
                    
                    # Add performance metrics by looking ahead
                    enhanced_signal = self._add_performance_metrics(signal, df, i)
                    
                    signals.append(enhanced_signal)
                    
                    self.logger.debug(f"📊 KAMA signal at {signal['backtest_timestamp']}: "
                                    f"{signal.get('signal_type')} (conf: {confidence_value:.1%}, "
                                    f"ER: {signal.get('efficiency_ratio', 0):.3f})")
                    
            except Exception as e:
                if debug_mode:
                    self.logger.warning(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        return signals
    
    def _add_kama_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int):
        """Add KAMA-specific metrics to the signal"""
        try:
            current_row = df.iloc[signal_idx]
            
            # Add KAMA-specific data for analysis
            kama_metrics = {
                'kama_value': signal.get('kama_value', current_row.get('kama', 0)),
                'efficiency_ratio': signal.get('efficiency_ratio', current_row.get('efficiency_ratio', 0)),
                'kama_trend': signal.get('kama_trend', current_row.get('kama_slope', 0)),
                'kama_distance': signal.get('kama_distance', 0),
                'signal_strength': signal.get('signal_strength', 0)
            }
            
            signal.update(kama_metrics)
            
            # Calculate additional KAMA analysis
            if len(df) > signal_idx + 1:
                next_row = df.iloc[signal_idx + 1]
                signal['kama_momentum'] = (next_row.get('kama', 0) - current_row.get('kama', 0)) / current_row.get('kama', 1)
            
        except Exception as e:
            self.logger.debug(f"Error adding KAMA metrics: {e}")
    
    def _add_performance_metrics(self, signal: Dict, df: pd.DataFrame, signal_idx: int) -> Dict:
        """Add performance metrics to signal by looking ahead"""
        try:
            enhanced_signal = signal.copy()
            
            # Get signal details
            entry_price = signal.get('signal_price', signal.get('price', df.iloc[signal_idx]['close']))
            signal_type = signal.get('signal_type', 'UNKNOWN')
            
            # Look ahead for performance (up to 24 hours / 96 bars for 15m timeframe)
            max_lookback = min(96, len(df) - signal_idx - 1)
            
            if max_lookback > 0:
                future_data = df.iloc[signal_idx+1:signal_idx+1+max_lookback]
                
                if signal_type == 'BULL':
                    # For buy signals, look for profit in higher prices
                    max_profit = (future_data['high'].max() - entry_price) * 10000  # Convert to pips
                    max_loss = (entry_price - future_data['low'].min()) * 10000
                elif signal_type == 'BEAR':
                    # For sell signals, look for profit in lower prices
                    max_profit = (entry_price - future_data['low'].min()) * 10000  # Convert to pips
                    max_loss = (future_data['high'].max() - entry_price) * 10000
                else:
                    max_profit = 0
                    max_loss = 0
                
                enhanced_signal.update({
                    'max_profit_pips': round(max_profit, 1),
                    'max_loss_pips': round(max_loss, 1),
                    'profit_loss_ratio': round(max_profit / max_loss, 2) if max_loss > 0 else 0,
                    'lookback_bars': max_lookback
                })
                
                # KAMA-specific performance metrics
                efficiency_at_signal = signal.get('efficiency_ratio', 0)
                if efficiency_at_signal > 0.5:
                    enhanced_signal['high_efficiency_signal'] = True
                elif efficiency_at_signal < 0.2:
                    enhanced_signal['low_efficiency_signal'] = True
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"Error adding performance metrics: {e}")
            return signal
    
    def _display_modular_stats(self, perf_stats: Dict):
        """Display modular component performance stats"""
        try:
            if 'cache_stats' in perf_stats:
                cache_stats = perf_stats['cache_stats']
                hit_ratio = cache_stats.get('hit_ratio', 0)
                self.logger.info(f"   🚀 Cache Performance: {hit_ratio:.1%} hit ratio "
                               f"({cache_stats.get('cache_hits', 0)} hits, {cache_stats.get('cache_misses', 0)} misses)")
            
            if 'forex_optimizer_stats' in perf_stats:
                forex_stats = perf_stats['forex_optimizer_stats']
                self.logger.info(f"   🌍 Forex Optimizer: {forex_stats.get('forex_pairs_supported', 0)} pairs supported")
            
            if 'validator_stats' in perf_stats:
                validator_stats = perf_stats['validator_stats']
                acceptance_rate = validator_stats.get('acceptance_rate', 0)
                self.logger.info(f"   🧠 Validator: {acceptance_rate:.1%} acceptance rate")
                
        except Exception as e:
            self.logger.debug(f"Error displaying modular stats: {e}")
    
    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic"""
        self.logger.info("\n📊 RESULTS BY EPIC:")
        self.logger.info("-" * 30)
        
        for epic, result in epic_results.items():
            if 'error' in result:
                self.logger.info(f"   {epic}: ❌ {result['error']}")
            else:
                self.logger.info(f"   {epic}: {result['signals']} signals")
    
    def _display_signals(self, signals: List[Dict]):
        """Display individual signals"""
        self.logger.info("\n🎯 INDIVIDUAL KAMA SIGNALS:")
        self.logger.info("-" * 70)
        
        try:
            self.signal_analyzer.display_signal_list(signals, max_signals=20)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not use SignalAnalyzer display: {e}")
            # Fallback to manual display with KAMA-specific info
            for i, signal in enumerate(signals[:20], 1):
                timestamp = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                confidence = signal.get('confidence', signal.get('confidence_score', 0))
                price = signal.get('signal_price', signal.get('price', 0))
                efficiency = signal.get('efficiency_ratio', 0)
                kama_trend = signal.get('kama_trend', 0)
                
                self.logger.info(f"{i:2d}. {timestamp} | {epic} | {signal_type} | "
                               f"Conf: {confidence:.1%} | Price: {price:.5f} | "
                               f"ER: {efficiency:.3f} | Trend: {kama_trend:.4f}")
    
    def _analyze_performance(self, signals: List[Dict]):
        """Analyze KAMA strategy performance"""
        try:
            # Create custom KAMA performance analysis
            metrics = self._create_kama_performance_analysis(signals)
            
            self.logger.info("\n📈 KAMA STRATEGY PERFORMANCE:")
            self.logger.info("=" * 40)
            self.logger.info(f"   📊 Total Signals: {metrics.get('total_signals', len(signals))}")
            self.logger.info(f"   🎯 Average Confidence: {metrics.get('avg_confidence', 0):.1%}")
            self.logger.info(f"   📈 Bull Signals: {metrics.get('bull_signals', 0)}")
            self.logger.info(f"   📉 Bear Signals: {metrics.get('bear_signals', 0)}")
            
            # KAMA-specific metrics
            self.logger.info(f"   🔄 Average Efficiency Ratio: {metrics.get('avg_efficiency_ratio', 0):.3f}")
            self.logger.info(f"   📈 High Efficiency Signals (>0.5): {metrics.get('high_efficiency_count', 0)}")
            self.logger.info(f"   📉 Low Efficiency Signals (<0.2): {metrics.get('low_efficiency_count', 0)}")
            
            # Performance metrics if available
            if 'avg_profit' in metrics:
                self.logger.info(f"   💰 Average Profit: {metrics['avg_profit']:.1f} pips")
            if 'avg_loss' in metrics:
                self.logger.info(f"   📉 Average Loss: {metrics['avg_loss']:.1f} pips")
            if 'win_rate' in metrics:
                self.logger.info(f"   🏆 Win Rate: {metrics['win_rate']:.1%}")
            
            # Additional KAMA metrics
            if 'efficiency_win_rate' in metrics:
                self.logger.info(f"   🎯 High Efficiency Win Rate: {metrics['efficiency_win_rate']:.1%}")
            if 'avg_kama_trend' in metrics:
                self.logger.info(f"   📊 Average KAMA Trend: {metrics['avg_kama_trend']:.4f}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ KAMA performance analysis failed: {e}")
            self._fallback_performance_analysis(signals)

    def _create_kama_performance_analysis(self, signals: List[Dict]) -> Dict:
        """Create KAMA-specific performance analysis"""
        if not signals:
            return {}
        
        total_signals = len(signals)
        bull_signals = [s for s in signals if s.get('signal_type') == 'BULL']
        bear_signals = [s for s in signals if s.get('signal_type') == 'BEAR']
        
        # Calculate confidence properly
        confidences = []
        efficiency_ratios = []
        kama_trends = []
        
        for s in signals:
            conf = s.get('confidence', s.get('confidence_score', 0))
            if conf is not None:
                confidences.append(conf)
            
            er = s.get('efficiency_ratio', 0)
            if er is not None:
                efficiency_ratios.append(er)
            
            trend = s.get('kama_trend', 0)
            if trend is not None:
                kama_trends.append(trend)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_efficiency = sum(efficiency_ratios) / len(efficiency_ratios) if efficiency_ratios else 0
        avg_kama_trend = sum(kama_trends) / len(kama_trends) if kama_trends else 0
        
        # KAMA-specific analysis
        high_efficiency_signals = [s for s in signals if s.get('efficiency_ratio', 0) > 0.5]
        low_efficiency_signals = [s for s in signals if s.get('efficiency_ratio', 0) < 0.2]
        
        metrics = {
            'total_signals': total_signals,
            'bull_signals': len(bull_signals),
            'bear_signals': len(bear_signals),
            'avg_confidence': avg_confidence,
            'avg_efficiency_ratio': avg_efficiency,
            'avg_kama_trend': avg_kama_trend,
            'high_efficiency_count': len(high_efficiency_signals),
            'low_efficiency_count': len(low_efficiency_signals)
        }
        
        # Performance metrics from max_profit_pips/max_loss_pips
        profit_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]
        
        if profit_signals:
            profits = [s['max_profit_pips'] for s in profit_signals]
            losses = [s['max_loss_pips'] for s in profit_signals]
            
            avg_profit = sum(profits) / len(profits)
            avg_loss = sum(losses) / len(losses)
            
            # Win rate (assuming 20 pip target)
            winners = [s for s in profit_signals if s['max_profit_pips'] >= 20]
            win_rate = len(winners) / len(profit_signals)
            
            # High efficiency signal performance
            high_eff_profit_signals = [s for s in profit_signals if s.get('efficiency_ratio', 0) > 0.5]
            if high_eff_profit_signals:
                high_eff_winners = [s for s in high_eff_profit_signals if s['max_profit_pips'] >= 20]
                efficiency_win_rate = len(high_eff_winners) / len(high_eff_profit_signals)
            else:
                efficiency_win_rate = 0
            
            metrics.update({
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'win_rate': win_rate,
                'efficiency_win_rate': efficiency_win_rate
            })
        
        return metrics
    
    def _analyze_modular_components(self, signals: List[Dict]):
        """Analyze modular component performance"""
        try:
            if hasattr(self.strategy, 'get_performance_stats'):
                all_stats = self.strategy.get_performance_stats()
                
                self.logger.info("\n🏗️ MODULAR COMPONENT ANALYSIS:")
                self.logger.info("=" * 40)
                
                # Cache performance
                if 'cache_stats' in all_stats:
                    cache_stats = all_stats['cache_stats']
                    self.logger.info(f"   🚀 Cache Hit Ratio: {cache_stats.get('hit_ratio', 0):.1%}")
                    self.logger.info(f"   💾 Cached Entries: {cache_stats.get('cached_entries', {})}")
                
                # Validator performance
                if 'validator_stats' in all_stats:
                    validator_stats = all_stats['validator_stats']
                    self.logger.info(f"   🧠 Validation Acceptance Rate: {validator_stats.get('acceptance_rate', 0):.1%}")
                    self.logger.info(f"   📊 Total Validations: {validator_stats.get('total_validations', 0)}")
                
                # Signal detector performance
                if 'signal_detector_stats' in all_stats:
                    detector_stats = all_stats['signal_detector_stats']
                    self.logger.info(f"   🔍 Total Signal Detections: {detector_stats.get('total_signals', 0)}")
                    self.logger.info(f"   📈 Bull/Bear Ratio: {detector_stats.get('bull_percentage', 0):.1f}%/{detector_stats.get('bear_percentage', 0):.1f}%")
                
        except Exception as e:
            self.logger.debug(f"Modular analysis error: {e}")
    
    def _fallback_performance_analysis(self, signals: List[Dict]):
        """Fallback performance analysis if main analysis fails"""
        total = len(signals)
        bull_count = sum(1 for s in signals if s.get('signal_type') == 'BULL')
        bear_count = total - bull_count
        
        # Calculate basic metrics
        confidences = [s.get('confidence', s.get('confidence_score', 0)) for s in signals if s.get('confidence', s.get('confidence_score', 0)) is not None]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        
        efficiency_ratios = [s.get('efficiency_ratio', 0) for s in signals if s.get('efficiency_ratio', 0) is not None]
        avg_efficiency = sum(efficiency_ratios) / len(efficiency_ratios) if efficiency_ratios else 0
        
        self.logger.info("\n📈 KAMA STRATEGY PERFORMANCE (Basic):")
        self.logger.info("=" * 40)
        self.logger.info(f"   📊 Total Signals: {total}")
        self.logger.info(f"   🎯 Average Confidence: {avg_conf:.1%}")
        self.logger.info(f"   🔄 Average Efficiency Ratio: {avg_efficiency:.3f}")
        self.logger.info(f"   📈 Bull Signals: {bull_count}")
        self.logger.info(f"   📉 Bear Signals: {bear_count}")
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic code"""
        try:
            # Convert "CS.D.EURUSD.CEEM.IP" to "EURUSD"
            if '.D.' in epic and '.MINI.IP' in epic:
                parts = epic.split('.D.')
                if len(parts) > 1:
                    pair_part = parts[1].split('.MINI.IP')[0]
                    return pair_part
            
            # Fallback - try to extract from config.PAIR_INFO
            pair_info = getattr(config, 'PAIR_INFO', {})
            if epic in pair_info:
                return pair_info[epic].get('pair', 'EURUSD')
            
            # Default fallback
            self.logger.warning(f"⚠️ Could not extract pair from {epic}, using EURUSD")
            return 'EURUSD'
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting pair from {epic}: {e}, using EURUSD")
            return 'EURUSD'  


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='KAMA Strategy Backtest')
    
    # Required arguments
    parser.add_argument('--epic', help='Epic to backtest (e.g., CS.D.EURUSD.CEEM.IP)')
    parser.add_argument('--days', type=int, default=7, help='Days to backtest (default: 7)')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (default: 15m)')
    
    # Optional arguments
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--legacy', action='store_true', help='Use legacy KAMA strategy (not modular)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run backtest
    backtest = KAMABacktest()
    
    success = backtest.run_backtest(
        epic=args.epic,
        days=args.days,
        timeframe=args.timeframe,
        show_signals=args.show_signals,
        use_modular=not args.legacy,  # Use modular unless --legacy flag is set
        min_confidence=args.min_confidence,
        debug_mode=args.debug
    )
    
    if success:
        print("\n✅ KAMA backtest completed successfully!")
        print("🎯 Key insights:")
        print("   - Pay attention to efficiency ratio patterns")
        print("   - High efficiency (>0.5) signals typically perform better")
        print("   - Low efficiency (<0.2) signals should be avoided")
        print("   - KAMA works best in trending/volatile markets")
        sys.exit(0)
    else:
        print("\n❌ KAMA backtest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()