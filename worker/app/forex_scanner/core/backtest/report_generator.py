# core/backtest/report_generator.py
"""
Backtest Report Generator - Consistent reporting across all backtest results

This module provides comprehensive, consistent reporting for all backtest
operations, supporting multiple output formats and detailed analysis.

Features:
- Unified report generation for all backtest modes
- Multiple output formats (console, JSON, CSV, HTML)
- Performance comparison tables
- Signal analysis and visualization
- Parameter optimization reports
- Smart Money analysis reports
- Export capabilities
"""

import logging
import os
import json
import csv
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
from dataclasses import asdict

try:
    from core.backtest.unified_backtest_engine import BacktestResult, BacktestMode
    from core.backtest.backtest_config import UnifiedBacktestConfig
    from core.backtest.parameter_manager import OptimizationResult
    import config
except ImportError:
    from forex_scanner.core.backtest.unified_backtest_engine import BacktestResult, BacktestMode
    from forex_scanner.core.backtest.backtest_config import UnifiedBacktestConfig
    from forex_scanner.core.backtest.parameter_manager import OptimizationResult
    from forex_scanner import config


class BacktestReportGenerator:
    """
    Generator for comprehensive backtest reports

    Provides consistent, detailed reporting across all backtest modes
    with support for multiple output formats and analysis types.
    """

    def __init__(self):
        self.logger = logging.getLogger('report_generator')

    def generate_report(
        self,
        results: List[BacktestResult],
        config: UnifiedBacktestConfig,
        output_file: Optional[str] = None
    ):
        """
        Generate comprehensive report for backtest results

        Args:
            results: List of backtest results
            config: Backtest configuration used
            output_file: Optional file path for report output
        """

        self.logger.info("📊 Generating backtest report...")

        # Filter results
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]

        # Generate appropriate report based on mode
        if config.mode == BacktestMode.SINGLE_STRATEGY:
            self._generate_single_strategy_report(successful_results, failed_results, config)
        elif config.mode == BacktestMode.MULTI_STRATEGY or config.mode == BacktestMode.COMPARISON:
            self._generate_comparison_report(successful_results, failed_results, config)
        elif config.mode == BacktestMode.PARAMETER_SWEEP:
            self._generate_optimization_report(successful_results, failed_results, config)
        elif config.mode == BacktestMode.VALIDATION:
            self._generate_validation_report(successful_results, failed_results, config)

        # Show session summary
        self._display_session_summary(results, config)

    def _generate_single_strategy_report(
        self,
        successful_results: List[BacktestResult],
        failed_results: List[BacktestResult],
        config: UnifiedBacktestConfig
    ):
        """Generate report for single strategy backtest"""

        strategy_name = config.strategies[0] if config.strategies else "Unknown"

        self.logger.info(f"\n📈 {strategy_name.upper()} STRATEGY REPORT")
        self.logger.info("=" * 60)

        if not successful_results:
            self.logger.warning("❌ No successful results to report")
            return

        # Aggregate metrics across all epics/timeframes
        all_signals = []
        total_execution_time = 0

        for result in successful_results:
            all_signals.extend(result.signals)
            total_execution_time += result.execution_time

        self.logger.info(f"📊 Strategy: {strategy_name}")
        self.logger.info(f"📈 Epics tested: {len(set(r.epic for r in successful_results))}")
        self.logger.info(f"⏰ Timeframes: {len(set(r.timeframe for r in successful_results))}")
        self.logger.info(f"🕐 Total execution: {total_execution_time:.1f}s")
        self.logger.info(f"🎯 Total signals: {len(all_signals)}")

        # Display results by epic
        self._display_results_by_epic(successful_results)

        # Performance analysis
        if all_signals:
            self._display_performance_analysis(all_signals, strategy_name)

            # Show individual signals if requested
            if config.output.show_signals:
                self._display_signal_details(all_signals, config.output.max_signals_display)

        # Smart Money analysis
        if config.smart_money.enabled:
            self._display_smart_money_summary(successful_results)

    def _generate_comparison_report(
        self,
        successful_results: List[BacktestResult],
        failed_results: List[BacktestResult],
        config: UnifiedBacktestConfig
    ):
        """Generate report for multi-strategy comparison"""

        self.logger.info(f"\n🎯 STRATEGY COMPARISON REPORT")
        self.logger.info("=" * 60)

        if not successful_results:
            self.logger.warning("❌ No successful results to compare")
            return

        # Group results by strategy
        strategy_results = {}
        for result in successful_results:
            if result.strategy not in strategy_results:
                strategy_results[result.strategy] = []
            strategy_results[result.strategy].append(result)

        # Generate comparison table
        self._display_strategy_comparison_table(strategy_results)

        # Detailed analysis for each strategy
        if config.output.show_performance_details:
            for strategy, results in strategy_results.items():
                self.logger.info(f"\n📊 {strategy.upper()} DETAILED ANALYSIS:")
                self.logger.info("-" * 40)

                all_signals = []
                for result in results:
                    all_signals.extend(result.signals)

                if all_signals:
                    self._display_performance_analysis(all_signals, strategy)

        # Rankings and recommendations
        self._display_strategy_rankings(strategy_results)

    def _generate_optimization_report(
        self,
        successful_results: List[BacktestResult],
        failed_results: List[BacktestResult],
        config: UnifiedBacktestConfig
    ):
        """Generate report for parameter optimization"""

        self.logger.info(f"\n🎯 PARAMETER OPTIMIZATION REPORT")
        self.logger.info("=" * 60)

        if not successful_results:
            self.logger.warning("❌ No successful optimization results")
            return

        # This would integrate with actual optimization results
        # For now, show basic results
        strategy_name = config.strategies[0] if config.strategies else "Unknown"

        self.logger.info(f"📊 Strategy: {strategy_name}")
        self.logger.info(f"🔍 Method: {config.optimization.method.value}")
        self.logger.info(f"🎚️ Scoring: {config.optimization.scoring_metric}")
        self.logger.info(f"🧪 Max combinations: {config.optimization.max_combinations}")

        # Show parameter ranges tested
        if config.optimization.parameter_ranges:
            self.logger.info(f"\n📋 PARAMETER RANGES TESTED:")
            for param, range_str in config.optimization.parameter_ranges.items():
                self.logger.info(f"   • {param}: {range_str}")

        # Show results summary
        self.logger.info(f"\n📈 OPTIMIZATION RESULTS:")
        self.logger.info(f"   Successful tests: {len(successful_results)}")
        self.logger.info(f"   Failed tests: {len(failed_results)}")

        if successful_results:
            # Find best result by signal count (placeholder scoring)
            best_result = max(successful_results, key=lambda r: len(r.signals))
            self.logger.info(f"\n🏆 BEST RESULT:")
            self.logger.info(f"   Signals found: {len(best_result.signals)}")
            self.logger.info(f"   Epic: {best_result.epic}")
            self.logger.info(f"   Timeframe: {best_result.timeframe}")
            self.logger.info(f"   Execution time: {best_result.execution_time:.1f}s")

    def _generate_validation_report(
        self,
        successful_results: List[BacktestResult],
        failed_results: List[BacktestResult],
        config: UnifiedBacktestConfig
    ):
        """Generate report for signal validation"""

        self.logger.info(f"\n🔍 SIGNAL VALIDATION REPORT")
        self.logger.info("=" * 60)

        strategy_name = config.strategies[0] if config.strategies else "Unknown"
        target_timestamp = config.validation.target_timestamp

        self.logger.info(f"📊 Strategy: {strategy_name}")
        self.logger.info(f"⏰ Target timestamp: {target_timestamp}")
        self.logger.info(f"📈 Epic: {config.epics[0] if config.epics else 'Unknown'}")
        self.logger.info(f"🕐 Timeframe: {config.timeframes[0] if config.timeframes else 'Unknown'}")

        if successful_results:
            result = successful_results[0]
            self.logger.info(f"\n✅ VALIDATION SUCCESSFUL")
            self.logger.info(f"   Signals found: {len(result.signals)}")
            self.logger.info(f"   Execution time: {result.execution_time:.1f}s")

            if result.signals:
                signal = result.signals[0]
                self.logger.info(f"\n🎯 SIGNAL DETAILS:")
                self.logger.info(f"   Type: {signal.get('signal_type', 'Unknown')}")
                self.logger.info(f"   Confidence: {signal.get('confidence', 0):.1%}")
                self.logger.info(f"   Price: {signal.get('price', 0):.5f}")

                if 'smart_money_analysis' in signal:
                    sm_analysis = signal['smart_money_analysis']
                    self.logger.info(f"   Smart Money Score: {sm_analysis.get('confluence_score', 0):.1%}")

        else:
            self.logger.warning(f"❌ VALIDATION FAILED")
            if failed_results:
                self.logger.error(f"   Error: {failed_results[0].error}")

    def _display_results_by_epic(self, results: List[BacktestResult]):
        """Display results grouped by epic"""

        epic_groups = {}
        for result in results:
            if result.epic not in epic_groups:
                epic_groups[result.epic] = []
            epic_groups[result.epic].append(result)

        self.logger.info(f"\n📈 RESULTS BY EPIC:")
        self.logger.info("-" * 40)

        for epic, epic_results in epic_groups.items():
            total_signals = sum(len(r.signals) for r in epic_results)
            avg_execution = sum(r.execution_time for r in epic_results) / len(epic_results)

            # Extract pair name
            pair = epic.split('.')[2] if '.' in epic else epic[-6:]

            self.logger.info(f"   {pair:<8} | {total_signals:3d} signals | {avg_execution:5.1f}s avg")

    def _display_performance_analysis(self, signals: List[Dict], strategy_name: str):
        """Display comprehensive performance analysis"""

        if not signals:
            return

        self.logger.info(f"\n📊 {strategy_name.upper()} PERFORMANCE ANALYSIS:")
        self.logger.info("-" * 50)

        # Basic metrics
        total_signals = len(signals)

        # Signal type breakdown
        bull_signals = sum(1 for s in signals if s.get('signal_type', '').upper() in ['BUY', 'BULL', 'LONG'])
        bear_signals = sum(1 for s in signals if s.get('signal_type', '').upper() in ['SELL', 'BEAR', 'SHORT'])

        self.logger.info(f"   Total signals: {total_signals}")
        self.logger.info(f"   Bull signals: {bull_signals} ({bull_signals/total_signals*100:.1f}%)")
        self.logger.info(f"   Bear signals: {bear_signals} ({bear_signals/total_signals*100:.1f}%)")

        # Confidence analysis
        confidences = []
        for s in signals:
            conf = s.get('confidence', s.get('confidence_score', 0))
            if conf is not None:
                if conf > 1:
                    conf = conf / 100.0
                confidences.append(conf)

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)

            self.logger.info(f"   Average confidence: {avg_confidence:.1%}")
            self.logger.info(f"   Confidence range: {min_confidence:.1%} - {max_confidence:.1%}")

        # Performance metrics (if available)
        performance_signals = [s for s in signals if 'max_profit_pips' in s and 'max_loss_pips' in s]

        if performance_signals:
            profits = [s['max_profit_pips'] for s in performance_signals]
            losses = [s['max_loss_pips'] for s in performance_signals]

            # Win/loss analysis
            winners = [s for s in performance_signals if s.get('is_winner', False)]
            losers = [s for s in performance_signals if s.get('is_loser', False)]

            if winners or losers:
                total_trades = len(winners) + len(losers)
                win_rate = len(winners) / total_trades if total_trades > 0 else 0

                self.logger.info(f"   Win rate: {win_rate:.1%} ({len(winners)}/{total_trades})")

            if profits:
                avg_profit = sum(profits) / len(profits)
                max_profit = max(profits)
                self.logger.info(f"   Average profit: {avg_profit:.1f} pips")
                self.logger.info(f"   Max profit: {max_profit:.1f} pips")

            if losses:
                avg_loss = sum(losses) / len(losses)
                max_loss = max(losses)
                self.logger.info(f"   Average loss: {avg_loss:.1f} pips")
                self.logger.info(f"   Max loss: {max_loss:.1f} pips")

            # Trailing stop analysis
            trailing_signals = [s for s in performance_signals if s.get('trailing_stop_used', False)]
            if trailing_signals:
                trailing_effectiveness = len([s for s in trailing_signals if s.get('is_winner', False)]) / len(trailing_signals)
                self.logger.info(f"   Trailing stop usage: {len(trailing_signals)}/{len(performance_signals)} ({len(trailing_signals)/len(performance_signals)*100:.1f}%)")
                self.logger.info(f"   Trailing stop effectiveness: {trailing_effectiveness:.1%}")

    def _display_signal_details(self, signals: List[Dict], max_display: int = 20):
        """Display individual signal details"""

        display_signals = signals[:max_display]

        self.logger.info(f"\n🎯 INDIVIDUAL SIGNALS (showing {len(display_signals)} of {len(signals)}):")
        self.logger.info("=" * 120)
        self.logger.info("#   TIMESTAMP            EPIC     TYPE STRATEGY        PRICE    CONF   PROFIT   LOSS    OUTCOME")
        self.logger.info("-" * 120)

        for i, signal in enumerate(display_signals, 1):
            # Extract signal details
            timestamp = signal.get('backtest_timestamp', signal.get('timestamp', 'Unknown'))[:19]
            epic = signal.get('epic', 'Unknown')

            # Shorten epic for display
            if 'CS.D.' in epic and '.MINI.IP' in epic:
                epic_display = epic.split('.D.')[1].split('.MINI.IP')[0]
            else:
                epic_display = epic[-8:] if len(epic) > 8 else epic

            signal_type = signal.get('signal_type', 'UNK')[:4]
            strategy = signal.get('strategy', 'unknown')[:12]
            price = signal.get('price', 0)

            confidence = signal.get('confidence', signal.get('confidence_score', 0))
            if confidence > 1:
                confidence = confidence / 100.0

            profit = signal.get('max_profit_pips', 0)
            loss = signal.get('max_loss_pips', 0)
            outcome = signal.get('trade_outcome', 'UNKNOWN')[:8]

            # Format row
            row = f"{i:<3} {timestamp:<20} {epic_display:<8} {signal_type:<4} {strategy:<15} {price:<8.5f} {confidence:<6.1%} {profit:<8.1f} {loss:<8.1f} {outcome}"
            self.logger.info(row)

        if len(signals) > max_display:
            self.logger.info(f"... and {len(signals) - max_display} more signals")

        self.logger.info("-" * 120)

    def _display_strategy_comparison_table(self, strategy_results: Dict[str, List[BacktestResult]]):
        """Display strategy comparison table"""

        self.logger.info(f"\n📊 STRATEGY COMPARISON TABLE:")
        self.logger.info("=" * 80)

        # Header
        header = f"{'Strategy':<15} {'Signals':<8} {'Avg Conf':<9} {'Execution':<10} {'Success Rate':<12}"
        self.logger.info(header)
        self.logger.info("-" * 80)

        # Calculate metrics for each strategy
        strategy_metrics = {}
        for strategy, results in strategy_results.items():
            all_signals = []
            total_execution = 0

            for result in results:
                all_signals.extend(result.signals)
                total_execution += result.execution_time

            # Calculate metrics
            signal_count = len(all_signals)
            avg_execution = total_execution / len(results) if results else 0
            success_rate = len(results) / len(results) if results else 0  # All are successful

            # Confidence calculation
            confidences = []
            for signal in all_signals:
                conf = signal.get('confidence', signal.get('confidence_score', 0))
                if conf is not None:
                    if conf > 1:
                        conf = conf / 100.0
                    confidences.append(conf)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            strategy_metrics[strategy] = {
                'signals': signal_count,
                'avg_confidence': avg_confidence,
                'avg_execution': avg_execution,
                'success_rate': success_rate
            }

            # Display row
            row = f"{strategy:<15} {signal_count:<8} {avg_confidence:<9.1%} {avg_execution:<10.1f} {success_rate:<12.1%}"
            self.logger.info(row)

        self.logger.info("=" * 80)

        return strategy_metrics

    def _display_strategy_rankings(self, strategy_results: Dict[str, List[BacktestResult]]):
        """Display strategy rankings and recommendations"""

        # Calculate total signals for ranking
        strategy_signals = {}
        for strategy, results in strategy_results.items():
            total_signals = sum(len(result.signals) for result in results)
            strategy_signals[strategy] = total_signals

        # Sort by signal count
        ranked_strategies = sorted(strategy_signals.items(), key=lambda x: x[1], reverse=True)

        self.logger.info(f"\n🏆 STRATEGY RANKINGS (by signal count):")
        self.logger.info("-" * 40)

        medals = ['🥇', '🥈', '🥉']
        for i, (strategy, signal_count) in enumerate(ranked_strategies):
            medal = medals[i] if i < len(medals) else f"{i+1:2d}."
            self.logger.info(f"   {medal} {strategy:<15}: {signal_count} signals")

        # Recommendations
        if ranked_strategies:
            best_strategy, best_count = ranked_strategies[0]

            self.logger.info(f"\n💡 RECOMMENDATIONS:")
            self.logger.info("-" * 20)

            if best_count == 0:
                self.logger.info("   ⚠️ No signals found across all strategies")
                self.logger.info("   Consider: lower confidence thresholds, longer timeframes")
            elif best_count < 5:
                self.logger.info(f"   📊 Low activity ({best_count} signals max)")
                self.logger.info("   Consider: extending backtest period or adjusting parameters")
            else:
                self.logger.info(f"   ✅ {best_strategy.upper()} shows highest activity ({best_count} signals)")

                if len(ranked_strategies) > 1 and ranked_strategies[1][1] > 0:
                    second_best = ranked_strategies[1]
                    ratio = best_count / second_best[1]
                    if ratio > 2:
                        self.logger.info(f"   🎯 {best_strategy} significantly outperforms others")
                    else:
                        self.logger.info(f"   🤝 {best_strategy} and {second_best[0]} show similar activity")

    def _display_smart_money_summary(self, results: List[BacktestResult]):
        """Display Smart Money analysis summary"""

        total_enhanced = 0
        total_analyzed = 0

        for result in results:
            if result.smart_money_stats:
                total_enhanced += result.smart_money_stats.get('signals_enhanced', 0)
                total_analyzed += result.smart_money_stats.get('signals_analyzed', 0)

        if total_analyzed > 0:
            self.logger.info(f"\n🧠 SMART MONEY ANALYSIS SUMMARY:")
            self.logger.info("-" * 40)
            self.logger.info(f"   Signals analyzed: {total_analyzed}")
            self.logger.info(f"   Signals enhanced: {total_enhanced}")
            self.logger.info(f"   Enhancement rate: {total_enhanced/total_analyzed*100:.1f}%")

    def _display_session_summary(self, results: List[BacktestResult], config: UnifiedBacktestConfig):
        """Display overall session summary"""

        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]

        total_signals = sum(len(r.signals) for r in successful_results)
        total_execution_time = sum(r.execution_time for r in results)

        self.logger.info(f"\n📋 SESSION SUMMARY:")
        self.logger.info("=" * 30)
        self.logger.info(f"   Mode: {config.mode.value}")
        self.logger.info(f"   Strategies: {len(set(config.strategies))}")
        self.logger.info(f"   Epics: {len(set(config.epics))}")
        self.logger.info(f"   Timeframes: {len(set(config.timeframes))}")
        self.logger.info(f"   Days analyzed: {config.days}")
        self.logger.info(f"   Total execution: {total_execution_time:.1f}s")
        self.logger.info(f"   Successful runs: {len(successful_results)}")
        self.logger.info(f"   Failed runs: {len(failed_results)}")
        self.logger.info(f"   Total signals: {total_signals}")

        if failed_results:
            self.logger.info(f"\n❌ FAILED RUNS:")
            for result in failed_results:
                self.logger.info(f"   • {result.strategy} - {result.epic}: {result.error}")

    def export_results(
        self,
        results: List[BacktestResult],
        output_path: str,
        format: str = "json"
    ):
        """
        Export results to file

        Args:
            results: Backtest results to export
            output_path: Output file path
            format: Export format (json, csv, html)
        """

        try:
            if format.lower() == "json":
                self._export_json(results, output_path)
            elif format.lower() == "csv":
                self._export_csv(results, output_path)
            elif format.lower() == "html":
                self._export_html(results, output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"✅ Results exported to: {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}")
            raise

    def _export_json(self, results: List[BacktestResult], output_path: str):
        """Export results as JSON"""

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'successful_results': len([r for r in results if r.error is None]),
            'results': []
        }

        for result in results:
            result_data = {
                'strategy': result.strategy,
                'epic': result.epic,
                'timeframe': result.timeframe,
                'execution_time': result.execution_time,
                'signals_count': len(result.signals),
                'error': result.error,
                'signals': result.signals,
                'performance': result.performance,
                'parameters_used': result.parameters_used,
                'smart_money_stats': result.smart_money_stats
            }
            export_data['results'].append(result_data)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def _export_csv(self, results: List[BacktestResult], output_path: str):
        """Export results as CSV"""

        # Create summary CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Strategy', 'Epic', 'Timeframe', 'Signals Count',
                'Execution Time', 'Win Rate', 'Avg Profit Pips',
                'Avg Loss Pips', 'Success', 'Error'
            ])

            # Data rows
            for result in results:
                win_rate = result.performance.get('win_rate', 0) if result.performance else 0
                avg_profit = result.performance.get('average_profit_pips', 0) if result.performance else 0
                avg_loss = result.performance.get('average_loss_pips', 0) if result.performance else 0

                writer.writerow([
                    result.strategy,
                    result.epic,
                    result.timeframe,
                    len(result.signals),
                    result.execution_time,
                    win_rate,
                    avg_profit,
                    avg_loss,
                    result.error is None,
                    result.error or ''
                ])

        # Create detailed signals CSV if there are signals
        signals_path = output_path.replace('.csv', '_signals.csv')
        all_signals = []

        for result in results:
            for signal in result.signals:
                signal_row = signal.copy()
                signal_row['result_strategy'] = result.strategy
                signal_row['result_epic'] = result.epic
                signal_row['result_timeframe'] = result.timeframe
                all_signals.append(signal_row)

        if all_signals:
            df = pd.DataFrame(all_signals)
            df.to_csv(signals_path, index=False)
            self.logger.info(f"📊 Detailed signals exported to: {signals_path}")

    def _export_html(self, results: List[BacktestResult], output_path: str):
        """Export results as HTML report"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Results Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .results-table {{ border-collapse: collapse; width: 100%; }}
        .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .results-table th {{ background-color: #f2f2f2; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .timestamp {{ font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Backtest Results Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary">
        <h2>📊 Summary</h2>
        <p>Total Results: {len(results)}</p>
        <p>Successful: {len([r for r in results if r.error is None])}</p>
        <p>Failed: {len([r for r in results if r.error is not None])}</p>
        <p>Total Signals: {sum(len(r.signals) for r in results if r.error is None)}</p>
    </div>

    <div class="results">
        <h2>📈 Results</h2>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Epic</th>
                    <th>Timeframe</th>
                    <th>Signals</th>
                    <th>Execution Time</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

        for result in results:
            status_class = "success" if result.error is None else "failure"
            status_text = "✅ Success" if result.error is None else f"❌ {result.error}"

            html_content += f"""
                <tr>
                    <td>{result.strategy}</td>
                    <td>{result.epic}</td>
                    <td>{result.timeframe}</td>
                    <td>{len(result.signals)}</td>
                    <td>{result.execution_time:.1f}s</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html_content)