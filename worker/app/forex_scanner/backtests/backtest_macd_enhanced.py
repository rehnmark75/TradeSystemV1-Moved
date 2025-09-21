#!/usr/bin/env python3
"""
Macd Strategy Backtest - Enhanced with Unified Framework
Run: python backtest_macd.py --epic CS.D.EURUSD.MINI.IP --days 7 --timeframe 15m

ENHANCEMENTS:
- Standardized signal format using StandardSignal objects
- Market intelligence integration with regime detection
- Unified parameter management with database optimization
- Smart Money Concepts integration (if available)
- Enhanced performance analytics and display
"""

import sys
import os
import argparse
import logging
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))

# If we're in backtests/ subdirectory, go up one level to project root
if os.path.basename(script_dir) == 'backtests':
    project_root = os.path.dirname(script_dir)
else:
    # If we're in project root
    project_root = script_dir

sys.path.insert(0, project_root)

try:
    from backtests.backtest_base import BacktestBase, StandardSignal, SignalType, MarketConditions
    from core.strategies.macd_strategy import MACDStrategy, create_macd_strategy
except ImportError:
    from forex_scanner.backtests.backtest_base import BacktestBase, StandardSignal, SignalType, MarketConditions
    from forex_scanner.core.strategies.macd_strategy import MACDStrategy, create_macd_strategy

try:
    import config
except ImportError:
    from forex_scanner import config


class EnhancedMACDStrategyBacktest(BacktestBase):
    """Enhanced Macd Strategy Backtesting with Unified Framework"""

    def __init__(self, use_optimal_parameters: bool = True):
        super().__init__('macd', use_optimal_parameters)
        self.logger = logging.getLogger('macd_backtest')
        self.setup_logging()

        # Strategy-specific performance tracking
        self.strategy_performance_stats = {
            'signals_generated': 0,
            'signals_enhanced': 0,
            'market_intelligence_applied': 0,
            'parameter_optimization_used': 0
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def initialize_strategy(self, epic: str = None) -> MACDStrategy:
        """Initialize the macd strategy with enhanced parameters"""
        try:
            # Get parameters using ParameterManager if available
            if hasattr(self, 'parameter_manager') and self.parameter_manager:
                param_set = self.get_parameters(epic=epic)
                strategy_params = param_set.get_all_values() if param_set else {}

                self.logger.info(f"✅ Using ParameterManager for {epic}")
                if param_set:
                    self.logger.info(f"   Parameter confidence: {param_set.confidence_score:.1%}")

            else:
                # Fallback to traditional method
                strategy_params = {}
                if self.use_optimal_parameters:
                    optimal_params = self.get_optimal_parameters(epic)
                    if optimal_params:
                        strategy_params.update({
                            'confidence_threshold': optimal_params.confidence_threshold,
                            'use_optimal_parameters': True
                        })

            # Create strategy with parameters
            strategy = create_macd_strategy(
                data_fetcher=self.data_fetcher,
                backtest_mode=True,
                epic=epic,
                timeframe='15m',
                use_optimized_parameters=self.use_optimal_parameters,
                **strategy_params
            )

            self.logger.info(f"✅ Enhanced Macd strategy initialized for {epic}")
            return strategy

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Macd strategy: {e}")
            raise

    def run_strategy_backtest(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> List[StandardSignal]:
        """
        Run enhanced macd strategy backtest on given data

        Args:
            df: DataFrame with price data and indicators
            epic: Trading pair epic
            spread_pips: Spread in pips
            timeframe: Trading timeframe

        Returns:
            List of StandardSignal objects
        """
        try:
            self.logger.info(f"🎯 Running Enhanced Macd backtest for {epic}")
            self.logger.info(f"   Data points: {len(df)}")
            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Spread: {spread_pips} pips")

            # Initialize strategy for this epic
            strategy = self.initialize_strategy(epic)

            # Detect signals using strategy
            signal = strategy.detect_signal(
                df=df,
                epic=epic,
                spread_pips=spread_pips,
                timeframe=timeframe
            )

            signals = []
            if signal:
                # Get market conditions for this timeframe
                market_conditions = self.get_market_conditions(datetime.now(), df)

                # Create StandardSignal object
                standard_signal = self.standardize_signal(
                    raw_signal=signal,
                    epic=epic,
                    timeframe=timeframe,
                    market_conditions=market_conditions
                )

                # Add strategy-specific metadata
                standard_signal.technical_indicators.update({
                    'macd_specific_analysis': self._analyze_macd_conditions(signal, df),
                    'strategy_confidence_factors': self._get_confidence_factors(signal),
                    'market_regime_suitability': self._assess_regime_suitability(market_conditions, signal)
                })

                # Apply Smart Money enhancement if available
                enhanced_signal = self.enhance_signal_with_smart_money(standard_signal)

                signals.append(enhanced_signal)

                # Update performance statistics
                self._update_strategy_performance_stats(enhanced_signal)

                self.logger.info(f"✅ Found Enhanced Macd signal: {enhanced_signal.signal_type.value} at "
                               f"{enhanced_signal.price} (confidence: {enhanced_signal.confidence:.1%})")
            else:
                self.logger.debug(f"   No signals detected for {epic}")

            return signals

        except Exception as e:
            self.logger.error(f"❌ Enhanced Macd backtest failed for {epic}: {e}")
            return []

    def _analyze_macd_conditions(self, signal: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze macd-specific market conditions"""
        try:
            if len(df) == 0:
                return {'error': 'No data available'}

            # Add macd-specific analysis here
            latest_row = df.iloc[-1]

            analysis = {
                'signal_strength': signal.get('confidence', 0.0),
                'market_context': 'favorable',  # Could be enhanced
                'risk_assessment': 'medium',    # Could be enhanced
                'execution_timing': 'good'      # Could be enhanced
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing macd conditions: {e}")
            return {'error': str(e)}

    def _get_confidence_factors(self, signal: Dict) -> Dict[str, float]:
        """Get confidence factors for the signal"""
        factors = {
            'technical_strength': signal.get('confidence', 0.0),
            'volume_confirmation': 0.5,  # Could be enhanced
            'trend_alignment': 0.5,      # Could be enhanced
            'support_resistance': 0.5   # Could be enhanced
        }

        return factors

    def _assess_regime_suitability(self, market_conditions: MarketConditions, signal: Dict) -> Dict[str, Any]:
        """Assess how suitable current market regime is for this strategy"""
        suitability = {
            'regime': market_conditions.regime.value,
            'suitability_score': 0.7,  # Default - could be enhanced based on strategy type
            'regime_factors': {
                'volatility_match': 'good' if 0.3 < market_conditions.volatility_percentile < 0.8 else 'poor',
                'session_timing': 'good' if market_conditions.session.value in ['london', 'new_york'] else 'fair',
                'trend_strength': 'moderate'
            }
        }

        return suitability

    def _update_strategy_performance_stats(self, signal: StandardSignal):
        """Update strategy-specific performance statistics"""
        self.strategy_performance_stats['signals_generated'] += 1

        if signal.smart_money_analysis:
            self.strategy_performance_stats['signals_enhanced'] += 1

        if signal.market_conditions:
            self.strategy_performance_stats['market_intelligence_applied'] += 1

        if hasattr(self, 'parameter_manager') and self.parameter_manager:
            self.strategy_performance_stats['parameter_optimization_used'] += 1


# Command line interface
def main():
    """Command line interface for the enhanced macd backtest"""
    parser = argparse.ArgumentParser(description='Enhanced Macd Strategy Backtest')
    parser.add_argument('--epic', type=str, help='Epic to backtest (e.g., CS.D.EURUSD.MINI.IP)')
    parser.add_argument('--days', type=int, default=7, help='Number of days to backtest')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (5m, 15m, 1h)')
    parser.add_argument('--show-signals', action='store_true', help='Display individual signals')
    parser.add_argument('--no-optimization', action='store_true', help='Disable parameter optimization')

    args = parser.parse_args()

    # Create and run backtest
    backtest = EnhancedMACDStrategyBacktest(
        use_optimal_parameters=not args.no_optimization
    )

    try:
        result = backtest.run_backtest(
            epic=args.epic,
            days=args.days,
            timeframe=args.timeframe,
            show_signals=args.show_signals
        )

        if result.success:
            print(f"\n🎯 Enhanced Macd Backtest Completed Successfully!")
            print(f"   Total Signals: {result.total_signals}")
            print(f"   Execution Time: {result.execution_time:.2f}s")

            if hasattr(backtest, 'strategy_performance_stats'):
                stats = backtest.strategy_performance_stats
                print(f"   Enhanced Features Used:")
                print(f"     Market Intelligence: {stats['market_intelligence_applied']}/{stats['signals_generated']}")
                print(f"     Smart Money Enhancement: {stats['signals_enhanced']}/{stats['signals_generated']}")
                print(f"     Parameter Optimization: {'✅' if stats['parameter_optimization_used'] else '❌'}")
        else:
            print(f"\n❌ Backtest failed: {result.error_message}")

    except KeyboardInterrupt:
        print("\n⏹️ Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Backtest error: {e}")


if __name__ == "__main__":
    main()
