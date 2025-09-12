#!/usr/bin/env python3
"""
Combined Strategy Backtest - Standalone Module with CONFIDENCE FIX
Run: python backtest_combined.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m

FIXES:
1. Fixed confidence field mapping issue - signals showing 0.0% confidence
2. Ensured proper confidence field standardization 
3. Added fallback confidence logic for combined signals
4. Enhanced confidence debugging and validation
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List
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
from core.strategies.combined_strategy import CombinedStrategy
from core.backtest.performance_analyzer import PerformanceAnalyzer
from core.backtest.signal_analyzer import SignalAnalyzer
try:
    import config
except ImportError:
    from forex_scanner import config


class CombinedBacktest:
    """Dedicated Combined Strategy Backtesting with CONFIDENCE FIXES"""
    
    def __init__(self):
        self.logger = logging.getLogger('combined_backtest')
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
    
    def _get_datetime_safe(self, row) -> str:
        """Safely get datetime from row, handling different column names"""
        try:
            # Try different possible datetime column names
            for col in ['datetime_utc', 'start_time', 'timestamp', 'datetime']:
                if col in row and row[col] is not None:
                    return str(row[col])
            
            # If no datetime column found, return current time
            from datetime import datetime
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            return 'Unknown'
    
    def _standardize_confidence_fields(self, signal: Dict) -> Dict:
        """
        CONFIDENCE FIX: Standardize confidence field names
        
        The issue was that signals had 'confidence_score' but backtest 
        was looking for 'confidence' field. This ensures both exist.
        """
        if not signal:
            return signal
            
        try:
            # Get confidence value from any available field
            confidence_value = None
            
            # Try different confidence field names in order of preference
            for field in ['confidence_score', 'confidence', 'conf', 'confidence_level']:
                if field in signal and signal[field] is not None:
                    confidence_value = signal[field]
                    break
            
            # If we found a confidence value, ensure both standard fields exist
            if confidence_value is not None:
                # Convert percentage (>1) to decimal if needed
                if confidence_value > 1:
                    confidence_value = confidence_value / 100.0
                
                # Ensure both field names exist for compatibility
                signal['confidence'] = confidence_value
                signal['confidence_score'] = confidence_value
                
                self.logger.debug(f"🔧 Standardized confidence: {confidence_value:.3f} ({confidence_value:.1%})")
            else:
                # No confidence found - use fallback logic for combined signals
                confidence_value = self._calculate_fallback_confidence(signal)
                signal['confidence'] = confidence_value
                signal['confidence_score'] = confidence_value
                
                self.logger.warning(f"⚠️ No confidence found, using fallback: {confidence_value:.1%}")
                
        except Exception as e:
            self.logger.error(f"❌ Confidence standardization failed: {e}")
            # Safe fallback
            signal['confidence'] = 0.6
            signal['confidence_score'] = 0.6
        
        return signal
    
    def _calculate_fallback_confidence(self, signal: Dict) -> float:
        """
        Calculate fallback confidence for combined signals
        
        Uses individual strategy confidences and contributing strategies
        to estimate reasonable confidence when main confidence is missing
        """
        try:
            # Method 1: Use individual strategy confidences
            individual_confidences = signal.get('individual_confidences', {})
            if individual_confidences:
                avg_individual = sum(individual_confidences.values()) / len(individual_confidences)
                
                # Add bonus for multiple strategies
                contributing_count = len(signal.get('contributing_strategies', []))
                ensemble_bonus = min(0.15, contributing_count * 0.05)  # Max 15% bonus
                
                fallback_confidence = min(0.9, avg_individual + ensemble_bonus)
                self.logger.debug(f"🔧 Fallback confidence from individuals: {avg_individual:.1%} + {ensemble_bonus:.1%} = {fallback_confidence:.1%}")
                return fallback_confidence
            
            # Method 2: Use contributing strategy count
            contributing_count = len(signal.get('contributing_strategies', []))
            if contributing_count >= 3:
                return 0.8  # High confidence for 3+ strategies
            elif contributing_count >= 2:
                return 0.7  # Medium-high for 2 strategies
            elif contributing_count >= 1:
                return 0.6  # Medium for 1 strategy
            
            # Method 3: Default fallback
            return 0.5
            
        except Exception as e:
            self.logger.error(f"❌ Fallback confidence calculation failed: {e}")
            return 0.5
    
    def initialize_combined_strategy(self, mode: str = None, min_confidence: float = None):
        """Initialize Combined strategy with specific configuration"""
        
        # Temporarily override combined mode if specified
        original_mode = None
        if mode:
            original_mode = getattr(config, 'COMBINED_STRATEGY_MODE', 'dynamic')
            config.COMBINED_STRATEGY_MODE = mode
            self.logger.info(f"🔧 Using Combined mode: {mode}")
        
        # Temporarily override confidence threshold if specified
        original_confidence = None
        if min_confidence:
            original_confidence = getattr(config, 'MIN_COMBINED_CONFIDENCE', 0.70)
            config.MIN_COMBINED_CONFIDENCE = min_confidence
            self.logger.info(f"🎚️ Min combined confidence: {min_confidence:.1%}")
        
        self.strategy = CombinedStrategy()
        self.logger.info("✅ Combined Strategy initialized for backtest")
        
        # Log strategy configuration
        self._log_strategy_config()
        
        return original_mode, original_confidence
    
    def _log_strategy_config(self):
        """Log current strategy configuration"""
        mode = getattr(config, 'COMBINED_STRATEGY_MODE', 'dynamic')
        min_conf = getattr(config, 'MIN_COMBINED_CONFIDENCE', 0.70)
        
        self.logger.info(f"   🎯 Mode: {mode}")
        self.logger.info(f"   🎚️ Min Confidence: {min_conf:.1%}")
        
        # Log strategy weights
        weights = {
            'EMA': getattr(config, 'STRATEGY_WEIGHT_EMA', 0.30),
            'MACD': getattr(config, 'STRATEGY_WEIGHT_MACD', 0.20),
            'KAMA': getattr(config, 'STRATEGY_WEIGHT_KAMA', 0.20),
            'BB+SuperTrend': getattr(config, 'STRATEGY_WEIGHT_BB_SUPERTREND', 0.05),
            'Zero Lag': getattr(config, 'STRATEGY_WEIGHT_ZERO_LAG', 0.20)
        }
        
        self.logger.info("   ⚖️ Strategy Weights:")
        for strategy, weight in weights.items():
            if weight > 0:
                self.logger.info(f"      {strategy}: {weight:.1%}")
    
    def run_backtest(
        self, 
        epic: str = None, 
        days: int = 7,
        timeframe: str = '15m',
        show_signals: bool = False,
        mode: str = None,
        min_confidence: float = None,
        show_details: bool = False
    ) -> bool:
        """Run Combined strategy backtest"""
        
        # Setup epic list
        epic_list = [epic] if epic else config.EPIC_LIST
        
        self.logger.info("🧪 COMBINED STRATEGY BACKTEST (CONFIDENCE FIXED)")
        self.logger.info("=" * 55)
        self.logger.info(f"📊 Epic(s): {epic_list}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"📅 Days: {days}")
        self.logger.info(f"🎯 Show signals: {show_signals}")
        self.logger.info(f"📋 Show details: {show_details}")
        
        try:
            # Initialize strategy
            original_mode, original_confidence = self.initialize_combined_strategy(mode, min_confidence)
            
            all_signals = []
            epic_results = {}
            strategy_breakdown = {}
            
            for current_epic in epic_list:
                self.logger.info(f"\n📈 Processing {current_epic}")
                
                # Get enhanced data - Need to extract pair from epic
                # Convert epic like "CS.D.EURUSD.MINI.IP" to pair like "EURUSD"
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
                self.logger.info(f"   📅 Date range: {self._get_datetime_safe(df.iloc[0])} to {self._get_datetime_safe(df.iloc[-1])}")
                
                # Run Combined backtest
                signals = self._run_combined_backtest(df, current_epic, timeframe, show_details)
                
                all_signals.extend(signals)
                epic_results[current_epic] = {'signals': len(signals)}
                
                # Track strategy breakdown
                self._update_strategy_breakdown(signals, strategy_breakdown)
                
                self.logger.info(f"   🎯 Combined signals found: {len(signals)}")
            
            # Display epic-by-epic results
            self._display_epic_results(epic_results)
            
            # Display strategy breakdown
            if strategy_breakdown:
                self._display_strategy_breakdown(strategy_breakdown)
            
            # Overall analysis
            if all_signals:
                self.logger.info(f"\n✅ TOTAL COMBINED SIGNALS: {len(all_signals)}")
                
                # Show individual signals if requested
                if show_signals:
                    self._display_signals(all_signals, show_details)
                
                # Performance analysis
                self._analyze_performance(all_signals)
                
                # Restore original config
                if original_mode:
                    config.COMBINED_STRATEGY_MODE = original_mode
                if original_confidence:
                    config.MIN_COMBINED_CONFIDENCE = original_confidence
                
                return True
            else:
                self.logger.warning("❌ No Combined signals found in backtest period")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Combined backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract currency pair from epic code"""
        try:
            # Convert "CS.D.EURUSD.MINI.IP" to "EURUSD"
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
    
    def _run_combined_backtest(self, df: pd.DataFrame, epic: str, timeframe: str, show_details: bool = False) -> List[Dict]:
        """Run Combined backtest on historical data with CONFIDENCE FIXES"""
        signals = []
        
        # Combined strategy needs more data for all indicators
        min_bars = 250  # Increased for combined strategy
        
        for i in range(min_bars, len(df)):
            try:
                # Get data up to current point
                current_data = df.iloc[:i+1].copy()
                
                # Detect signal using Combined strategy
                signal = self.strategy.detect_signal(
                    current_data, epic, config.SPREAD_PIPS, timeframe
                )
                
                if signal:
                    # CONFIDENCE FIX: Standardize confidence fields before processing
                    signal = self._standardize_confidence_fields(signal)
                    
                    # Add backtest metadata
                    signal['backtest_timestamp'] = self._get_datetime_safe(df.iloc[i])
                    signal['backtest_index'] = i
                    signal['candle_data'] = {
                        'open': float(df.iloc[i]['open']),
                        'high': float(df.iloc[i]['high']),
                        'low': float(df.iloc[i]['low']),
                        'close': float(df.iloc[i]['close'])
                    }
                    
                    # Ensure timestamp field for analysis compatibility
                    if 'timestamp' not in signal:
                        signal['timestamp'] = signal['backtest_timestamp']
                    
                    signals.append(signal)
                    
                    # Enhanced logging for combined signals
                    contributing_strategies = signal.get('contributing_strategies', [])
                    confidence = signal.get('confidence_score', 0)
                    
                    if show_details:
                        self.logger.info(f"📊 Combined signal at {signal['backtest_timestamp']}: "
                                       f"{signal.get('signal_type')} (Conf: {confidence:.1%}) "
                                       f"Contributors: {', '.join(contributing_strategies)}")
                    else:
                        self.logger.debug(f"📊 Combined signal at {signal['backtest_timestamp']}: "
                                        f"{signal.get('signal_type')} ({confidence:.1%})")
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing candle {i}: {e}")
                continue
        
        return signals
    
    def _update_strategy_breakdown(self, signals: List[Dict], breakdown: Dict):
        """Update strategy breakdown statistics"""
        for signal in signals:
            contributing = signal.get('contributing_strategies', [])
            for strategy in contributing:
                if strategy not in breakdown:
                    breakdown[strategy] = 0
                breakdown[strategy] += 1
    
    def _display_epic_results(self, epic_results: Dict):
        """Display results by epic"""
        self.logger.info("\n📊 RESULTS BY EPIC:")
        self.logger.info("-" * 30)
        
        for epic, result in epic_results.items():
            if 'error' in result:
                self.logger.info(f"   {epic}: ❌ {result['error']}")
            else:
                self.logger.info(f"   {epic}: {result['signals']} signals")
    
    def _display_strategy_breakdown(self, breakdown: Dict):
        """Display contributing strategy breakdown"""
        self.logger.info("\n📊 CONTRIBUTING STRATEGIES:")
        self.logger.info("-" * 35)
        
        total_contributions = sum(breakdown.values())
        sorted_strategies = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        
        for strategy, count in sorted_strategies:
            percentage = (count / total_contributions) * 100 if total_contributions > 0 else 0
            self.logger.info(f"   {strategy}: {count} contributions ({percentage:.1f}%)")
    
    def _display_signals(self, signals: List[Dict], show_details: bool = False):
        """Display individual signals with FIXED confidence display"""
        self.logger.info("\n🎯 INDIVIDUAL COMBINED SIGNALS:")
        self.logger.info("-" * 55)
        
        for i, signal in enumerate(signals, 1):
            timestamp = signal.get('backtest_timestamp', 'Unknown')
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            
            # CONFIDENCE FIX: Use the standardized confidence field
            confidence = signal.get('confidence_score', signal.get('confidence', 0))
            price = signal.get('price', signal.get('signal_price', signal.get('current_price', 0)))
            
            basic_info = (f"{i:2d}. {timestamp} | {epic} | {signal_type} | "
                         f"Conf: {confidence:.1%} | Price: {price:.5f}")
            
            if show_details:
                contributing = signal.get('contributing_strategies', [])
                individual_confs = signal.get('individual_confidences', {})
                
                self.logger.info(basic_info)
                self.logger.info(f"     Contributors: {', '.join(contributing)}")
                if individual_confs:
                    conf_details = [f"{s}: {c:.1%}" for s, c in individual_confs.items()]
                    self.logger.info(f"     Individual: {', '.join(conf_details)}")
            else:
                self.logger.info(basic_info)
    
    def _analyze_performance(self, signals: List[Dict]):
        """Analyze Combined strategy performance with CONFIDENCE FIXES"""
        try:
            # CONFIDENCE FIX: Ensure all signals have standardized confidence fields
            for signal in signals:
                self._standardize_confidence_fields(signal)
            
            metrics = self.performance_analyzer.analyze_performance(signals)
            
            self.logger.info("\n📈 COMBINED STRATEGY PERFORMANCE (FIXED):")
            self.logger.info("=" * 45)
            self.logger.info(f"   📊 Total Signals: {metrics.get('total_signals', len(signals))}")
            
            # CONFIDENCE FIX: Calculate confidence properly
            avg_confidence = self._calculate_average_confidence(signals)
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")
            
            self.logger.info(f"   📈 Bull Signals: {metrics.get('bull_signals', 0)}")
            self.logger.info(f"   📉 Bear Signals: {metrics.get('bear_signals', 0)}")
            
            # Combined-specific metrics
            consensus_signals = sum(1 for s in signals if 
                                  len(s.get('contributing_strategies', [])) >= 2)
            unanimous_signals = sum(1 for s in signals if 
                                  len(s.get('contributing_strategies', [])) >= 3)
            
            self.logger.info(f"   🤝 Consensus Signals (2+ strategies): {consensus_signals}")
            self.logger.info(f"   🎯 High Consensus (3+ strategies): {unanimous_signals}")
            
            # Signal quality distribution using FIXED confidence
            high_conf = sum(1 for s in signals if s.get('confidence_score', s.get('confidence', 0)) >= 0.8)
            medium_conf = sum(1 for s in signals if 0.6 <= s.get('confidence_score', s.get('confidence', 0)) < 0.8)
            
            self.logger.info(f"   🌟 High Confidence (≥80%): {high_conf}")
            self.logger.info(f"   ⭐ Medium Confidence (60-80%): {medium_conf}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance analysis failed: {e}")
            
            # Basic analysis fallback with CONFIDENCE FIXES
            total = len(signals)
            bull_count = sum(1 for s in signals if s.get('signal_type') == 'BULL')
            bear_count = total - bull_count
            avg_conf = self._calculate_average_confidence(signals)
            
            self.logger.info("\n📈 COMBINED STRATEGY PERFORMANCE (Basic Fixed):")
            self.logger.info("=" * 45)
            self.logger.info(f"   📊 Total Signals: {total}")
            self.logger.info(f"   🎯 Average Confidence: {avg_conf:.1%}")
            self.logger.info(f"   📈 Bull Signals: {bull_count}")
            self.logger.info(f"   📉 Bear Signals: {bear_count}")
    
    def _calculate_average_confidence(self, signals: List[Dict]) -> float:
        """Calculate average confidence from signals with proper field handling"""
        if not signals:
            return 0.0
        
        total_confidence = 0
        valid_signals = 0
        
        for signal in signals:
            # Try both confidence field names
            confidence = signal.get('confidence_score', signal.get('confidence', None))
            
            if confidence is not None and confidence > 0:
                # Convert percentage to decimal if needed
                if confidence > 1:
                    confidence = confidence / 100.0
                
                total_confidence += confidence
                valid_signals += 1
        
        return total_confidence / valid_signals if valid_signals > 0 else 0.0


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Combined Strategy Backtest (Confidence Fixed)')
    
    # Required arguments
    parser.add_argument('--epic', help='Epic to backtest (e.g., CS.D.EURUSD.MINI.IP)')
    parser.add_argument('--days', type=int, default=7, help='Days to backtest (default: 7)')
    parser.add_argument('--timeframe', default='15m', help='Timeframe (default: 15m)')
    
    # Optional arguments
    parser.add_argument('--show-signals', action='store_true', help='Show individual signals')
    parser.add_argument('--show-details', action='store_true', help='Show detailed signal information')
    parser.add_argument('--mode', help='Combined strategy mode (consensus, weighted, dynamic)')
    parser.add_argument('--min-confidence', type=float, help='Minimum confidence threshold')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run backtest
    backtest = CombinedBacktest()
    
    success = backtest.run_backtest(
        epic=args.epic,
        days=args.days,
        timeframe=args.timeframe,
        show_signals=args.show_signals,
        mode=args.mode,
        min_confidence=args.min_confidence,
        show_details=args.show_details
    )
    
    if success:
        print("\n✅ Combined backtest completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Combined backtest failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()