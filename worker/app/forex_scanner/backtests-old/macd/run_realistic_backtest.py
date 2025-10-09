"""
Realistic MACD Backtest Runner

This backtest exactly replicates live trading conditions:
1. Processes data bar-by-bar like live scanner
2. Implements proper signal spacing and trade management
3. Uses actual MACD strategy without modifications
4. Provides trustworthy results for production deployment

Key Differences from Old Backtest:
- Only calls detect_signal() when new bars arrive (not 600+ times in loop)
- Implements realistic trade lifecycle with slippage and timing
- Uses live strategy settings without backtest-specific modifications
- Provides comprehensive performance analysis
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from .engines.live_simulation_engine import LiveSimulationEngine
from .components.trade_manager import MACDTradeManager


class RealisticMACDBacktest:
    """
    Comprehensive MACD backtest that mirrors live trading exactly

    This backtest can be trusted to represent actual live trading performance.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.results = {}
        self.summary = {}

        self.logger.info("🚀 Realistic MACD Backtest initialized")
        self.logger.info(f"   Test period: {self.config['start_date']} to {self.config['end_date']}")
        self.logger.info(f"   Pairs to test: {len(self.config['test_pairs'])}")
        self.logger.info(f"   Initial balance per pair: ${self.config['initial_balance']:,.2f}")

    def get_default_config(self) -> Dict:
        """Get default backtest configuration"""
        # Use recent dates where real data should exist
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        return {
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': '15m',
            'test_pairs': [
                'CS.D.EURUSD.CEEM.IP',
                'CS.D.GBPUSD.MINI.IP',
                'CS.D.USDJPY.MINI.IP',
                'CS.D.EURJPY.MINI.IP',
                'CS.D.AUDUSD.MINI.IP',
                'CS.D.EURGBP.MINI.IP',
                'CS.D.GBPJPY.MINI.IP',
                'CS.D.AUDJPY.MINI.IP',
                'CS.D.USDCAD.MINI.IP',
                'CS.D.NZDUSD.MINI.IP'
            ],
            'initial_balance': 10000.0,
            'max_trades_per_day': 3,
            'signal_spacing_hours': 4
        }

    def run_pair_backtest(self, epic: str) -> Dict:
        """Run backtest for a single currency pair"""
        try:
            self.logger.info(f"📊 Starting backtest for {epic}")

            # Initialize simulation engine (mirrors live scanner)
            simulation_engine = LiveSimulationEngine(
                epic=epic,
                timeframe=self.config['timeframe']
            )

            # Initialize trade manager (handles complete trade lifecycle)
            trade_manager = MACDTradeManager(
                epic=epic,
                initial_balance=self.config['initial_balance']
            )

            # Load market data
            market_data = simulation_engine.load_market_data(
                self.config['start_date'],
                self.config['end_date']
            )

            self.logger.info(f"   Loaded {len(market_data)} bars for {epic}")

            # Run simulation bar-by-bar (like live trading)
            signals_generated = []
            start_index = 60  # Need enough data for MACD

            for i in range(start_index, len(market_data)):
                current_timestamp = market_data.index[i]
                current_price = market_data.iloc[i]['close']

                # Get data up to current bar (rolling window like live scanner)
                current_data_window = market_data.iloc[:i+1].copy()

                # Process new bar for trade management
                trade_manager.process_bar(current_price, current_timestamp)

                # Check for new signals (exactly like live scanner)
                signal = simulation_engine.process_new_bar(current_timestamp, current_data_window)

                if signal:
                    signals_generated.append(signal)

                    # Open trade if conditions allow
                    new_trade = trade_manager.open_trade(signal, current_price, current_timestamp)

                    if new_trade:
                        self.logger.info(f"   Trade opened: {new_trade.signal_type} #{new_trade.id}")

            # Get final performance results
            performance = trade_manager.get_performance_summary()

            # Compile comprehensive results
            result = {
                'epic': epic,
                'simulation_period': {
                    'start': self.config['start_date'],
                    'end': self.config['end_date'],
                    'days': (self.config['end_date'] - self.config['start_date']).days
                },
                'market_data': {
                    'total_bars': len(market_data),
                    'simulation_bars': len(market_data) - start_index
                },
                'signals': {
                    'total_generated': len(signals_generated),
                    'signals_per_day': len(signals_generated) / ((self.config['end_date'] - self.config['start_date']).days),
                    'signal_details': signals_generated
                },
                'trades': {
                    'total_trades': len(trade_manager.trades),
                    'open_trades': len(trade_manager.open_trades),
                    'closed_trades': len([t for t in trade_manager.trades if t.status.value.startswith('closed')])
                },
                'performance': performance,
                'strategy_config': simulation_engine.strategy.macd_config
            }

            self.logger.info(f"✅ Backtest completed for {epic}")
            self.logger.info(f"   Signals: {len(signals_generated)}")
            self.logger.info(f"   Trades: {result['trades']['total_trades']}")
            if 'win_rate' in performance:
                self.logger.info(f"   Win rate: {performance['win_rate']:.1%}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Backtest failed for {epic}: {e}")
            import traceback
            traceback.print_exc()
            return {'epic': epic, 'error': str(e)}

    def run_full_backtest(self) -> Dict:
        """Run comprehensive backtest across all pairs"""
        try:
            self.logger.info("🚀 Starting FULL REALISTIC MACD BACKTEST")
            self.logger.info("=" * 60)

            all_results = {}
            total_signals = 0
            total_trades = 0
            successful_pairs = 0

            # Test each pair individually
            for epic in self.config['test_pairs']:
                result = self.run_pair_backtest(epic)

                if 'error' not in result:
                    all_results[epic] = result
                    total_signals += result['signals']['total_generated']
                    total_trades += result['trades']['total_trades']
                    successful_pairs += 1
                else:
                    all_results[epic] = result

            # Calculate overall summary
            test_days = (self.config['end_date'] - self.config['start_date']).days
            avg_signals_per_pair = total_signals / successful_pairs if successful_pairs > 0 else 0
            avg_signals_per_day = avg_signals_per_pair / test_days if test_days > 0 else 0

            summary = {
                'backtest_info': {
                    'type': 'Realistic MACD Backtest',
                    'period': f"{self.config['start_date']} to {self.config['end_date']}",
                    'duration_days': test_days,
                    'pairs_tested': len(self.config['test_pairs']),
                    'successful_pairs': successful_pairs
                },
                'signal_analysis': {
                    'total_signals_all_pairs': total_signals,
                    'average_signals_per_pair': avg_signals_per_pair,
                    'signals_per_day_per_pair': avg_signals_per_day,
                    'expected_vs_actual': {
                        'expected_range': '10-30 per pair per week',
                        'actual': f"{avg_signals_per_pair:.1f} per pair per week"
                    }
                },
                'trade_analysis': {
                    'total_trades_all_pairs': total_trades,
                    'average_trades_per_pair': total_trades / successful_pairs if successful_pairs > 0 else 0
                },
                'validation_status': self.validate_results(all_results, avg_signals_per_day),
                'individual_results': all_results
            }

            self.logger.info("✅ FULL BACKTEST COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"   Successful pairs: {successful_pairs}/{len(self.config['test_pairs'])}")
            self.logger.info(f"   Total signals: {total_signals}")
            self.logger.info(f"   Avg signals/pair/day: {avg_signals_per_day:.2f}")
            self.logger.info(f"   Total trades: {total_trades}")

            return summary

        except Exception as e:
            self.logger.error(f"❌ Full backtest failed: {e}")
            raise

    def validate_results(self, results: Dict, avg_signals_per_day: float) -> Dict:
        """Validate backtest results for production readiness"""
        validation = {
            'signal_count_validation': 'UNKNOWN',
            'performance_validation': 'UNKNOWN',
            'production_ready': False,
            'issues': [],
            'recommendations': []
        }

        try:
            # Check signal count (most critical for money loss prevention)
            if avg_signals_per_day > 10:
                validation['signal_count_validation'] = 'FAILED'
                validation['issues'].append(f"Too many signals: {avg_signals_per_day:.1f}/day (expected <5/day)")
                validation['recommendations'].append("Increase signal spacing or tighten filters")
            elif avg_signals_per_day < 0.2:
                validation['signal_count_validation'] = 'WARNING'
                validation['issues'].append(f"Very few signals: {avg_signals_per_day:.1f}/day (expected >0.5/day)")
                validation['recommendations'].append("Consider loosening filters slightly")
            else:
                validation['signal_count_validation'] = 'PASSED'

            # Check individual pair consistency
            signal_counts = []
            for epic, result in results.items():
                if 'error' not in result:
                    daily_signals = result['signals']['signals_per_day']
                    signal_counts.append(daily_signals)

            if signal_counts:
                signal_std = pd.Series(signal_counts).std()
                if signal_std > 5:  # High variance
                    validation['issues'].append(f"High signal variance across pairs (std: {signal_std:.1f})")

            # Overall production readiness
            if validation['signal_count_validation'] == 'PASSED' and len(validation['issues']) <= 1:
                validation['production_ready'] = True
                validation['recommendations'].append("Strategy appears ready for live deployment")
            else:
                validation['recommendations'].append("Further tuning recommended before live deployment")

            return validation

        except Exception as e:
            validation['issues'].append(f"Validation error: {e}")
            return validation

    def display_signals(self, results: Dict):
        """Display signals in EMA backtest format"""
        try:
            all_signals = []

            # Collect all signals from all pairs
            for epic, pair_result in results['individual_results'].items():
                if 'error' not in pair_result and 'signals' in pair_result:
                    signals = pair_result['signals']['signal_details']
                    for signal in signals:
                        signal['epic'] = epic  # Ensure epic is set
                        all_signals.append(signal)

            if not all_signals:
                self.logger.info("⚠️ No signals found to display")
                return

            # Sort by timestamp (newest first)
            all_signals.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)

            self.logger.info("\n🎯 INDIVIDUAL MACD SIGNALS:")
            self.logger.info("=" * 120)
            self.logger.info("#   TIMESTAMP            PAIR     TYPE STRATEGY        PRICE    CONF   MACD     RSI    ADX    ")
            self.logger.info("-" * 120)

            display_signals = all_signals[:20]  # Show max 20 signals

            for i, signal in enumerate(display_signals, 1):
                # Format timestamp
                timestamp = signal.get('timestamp', 'Unknown')
                if isinstance(timestamp, datetime):
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')
                else:
                    timestamp_str = str(timestamp)[:19]

                # Extract pair name
                epic = signal.get('epic', 'Unknown')
                if 'CS.D.' in epic and '.MINI.IP' in epic:
                    pair = epic.split('.D.')[1].split('.MINI.IP')[0]
                else:
                    pair = epic[-6:] if len(epic) >= 6 else epic

                # Format signal type
                signal_type = signal.get('signal_type', 'UNK')
                if signal_type in ['BULL', 'LONG']:
                    type_display = 'BUY'
                elif signal_type in ['BEAR', 'SHORT']:
                    type_display = 'SELL'
                else:
                    type_display = signal_type or 'UNK'

                # Get values
                confidence = signal.get('confidence', 0)
                price = signal.get('close_price', 0)
                macd_hist = signal.get('macd_histogram', 0)
                rsi = signal.get('rsi', 0)
                adx = signal.get('adx', 0)

                row = f"{i:<3} {timestamp_str:<20} {pair:<8} {type_display:<4} {'macd':<15} {price:<8.5f} {confidence:<6.1%} {macd_hist:<8.6f} {rsi:<6.1f} {adx:<6.1f}"
                self.logger.info(row)

            self.logger.info("=" * 120)

            if len(all_signals) > 20:
                self.logger.info(f"📝 Showing latest 20 of {len(all_signals)} total signals (newest first)")
            else:
                self.logger.info(f"📝 Showing all {len(all_signals)} signals (newest first)")

        except Exception as e:
            self.logger.error(f"❌ Error displaying signals: {e}")

    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive backtest report"""
        try:
            report = []
            report.append("🎯 REALISTIC MACD BACKTEST REPORT")
            report.append("=" * 70)
            report.append("")

            # Summary
            summary = results['backtest_info']
            report.append(f"📊 BACKTEST SUMMARY:")
            report.append(f"   Period: {summary['period']}")
            report.append(f"   Duration: {summary['duration_days']} days")
            report.append(f"   Pairs: {summary['successful_pairs']}/{summary['pairs_tested']}")
            report.append("")

            # Signal Analysis
            signals = results['signal_analysis']
            report.append(f"📈 SIGNAL ANALYSIS:")
            report.append(f"   Total signals: {signals['total_signals_all_pairs']}")
            report.append(f"   Avg per pair: {signals['average_signals_per_pair']:.1f}")
            report.append(f"   Signals/day/pair: {signals['signals_per_day_per_pair']:.2f}")
            report.append(f"   Expected range: {signals['expected_vs_actual']['expected_range']}")
            report.append(f"   Actual result: {signals['expected_vs_actual']['actual']}")
            report.append("")

            # Individual Pair Results
            report.append(f"📋 INDIVIDUAL PAIR RESULTS:")
            for epic, result in results['individual_results'].items():
                if 'error' not in result:
                    signals_count = result['signals']['total_generated']
                    signals_per_day = result['signals']['signals_per_day']
                    trades_count = result['trades']['total_trades']

                    status = "✅" if 0.5 <= signals_per_day <= 5 else "⚠️" if signals_per_day < 0.5 else "❌"

                    report.append(f"   {status} {epic}: {signals_count} signals ({signals_per_day:.1f}/day), {trades_count} trades")

                    # Performance metrics if available
                    if 'win_rate' in result['performance']:
                        perf = result['performance']
                        report.append(f"      Performance: {perf['win_rate']:.1%} win rate, {perf['total_pips']:+.1f} pips")
                else:
                    report.append(f"   ❌ {epic}: ERROR - {result['error']}")

            report.append("")

            # Validation
            validation = results['validation_status']
            report.append(f"🔍 VALIDATION RESULTS:")
            report.append(f"   Signal Count: {validation['signal_count_validation']}")
            report.append(f"   Production Ready: {'✅ YES' if validation['production_ready'] else '❌ NO'}")

            if validation['issues']:
                report.append(f"   Issues:")
                for issue in validation['issues']:
                    report.append(f"     - {issue}")

            if validation['recommendations']:
                report.append(f"   Recommendations:")
                for rec in validation['recommendations']:
                    report.append(f"     - {rec}")

            report.append("")
            report.append("=" * 70)

            return "\n".join(report)

        except Exception as e:
            return f"Error generating report: {e}"



def main():
    """Main function to run realistic MACD backtest"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize and run backtest
        backtest = RealisticMACDBacktest()

        # Run full backtest
        results = backtest.run_full_backtest()

        # Generate and display report
        report = backtest.generate_report(results)
        print(report)

        # Display signals in EMA format with UTC timestamps
        backtest.display_signals(results)

        return results

    except Exception as e:
        logging.error(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()