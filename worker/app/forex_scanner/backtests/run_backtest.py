#!/usr/bin/env python3
"""
Unified Backtest CLI - Single entry point for all backtesting operations

This is the main command-line interface for the unified backtest system.
It provides a consistent, powerful interface for running backtests on any
strategy with advanced features like parameter optimization, Smart Money
analysis, and comprehensive reporting.

Usage Examples:
  # Quick test
  python run_backtest.py --strategy ema --epic EURUSD --days 3

  # Multi-strategy comparison
  python run_backtest.py --compare-strategies ema,macd,kama --epic EURUSD --days 7

  # Parameter optimization
  python run_backtest.py --strategy ema --optimize --param-ranges "confidence:0.4-0.8:0.1"

  # Signal validation
  python run_backtest.py --validate-signal "2025-08-28 19:30:00" --strategy ema --epic EURUSD

  # Use configuration template
  python run_backtest.py --template comprehensive --export results.json

  # Load custom configuration
  python run_backtest.py --config my_config.json
"""

import sys
import os
import argparse
import logging
import json
from typing import List, Optional
from datetime import datetime
import traceback

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
forex_scanner_dir = os.path.dirname(script_dir)
app_dir = os.path.dirname(forex_scanner_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, forex_scanner_dir)
sys.path.insert(0, app_dir)

try:
    from unified_backtest_engine import UnifiedBacktestEngine, BacktestMode
    from strategy_registry import get_strategy_registry
    from parameter_manager import ParameterManager, OptimizationMethod
    from backtest_config import (
        UnifiedBacktestConfig, BacktestConfigManager, ConfigTemplates,
        SmartMoneyConfig, OptimizationConfig, OutputConfig, ValidationConfig
    )
    from report_generator import BacktestReportGenerator
    import sys
    sys.path.append('..')
    import config
except ImportError as e:
    try:
        from forex_scanner.backtests.unified_backtest_engine import UnifiedBacktestEngine, BacktestMode
        from forex_scanner.backtests.strategy_registry import get_strategy_registry
        from forex_scanner.backtests.parameter_manager import ParameterManager, OptimizationMethod
        from forex_scanner.backtests.backtest_config import (
            UnifiedBacktestConfig, BacktestConfigManager, ConfigTemplates,
            SmartMoneyConfig, OptimizationConfig, OutputConfig, ValidationConfig
        )
        from forex_scanner.backtests.report_generator import BacktestReportGenerator
        from forex_scanner import config
    except ImportError:
        print(f"❌ Failed to import required modules: {e}")
        print("   Make sure you're running from the correct directory")
        sys.exit(1)


class UnifiedBacktestCLI:
    """
    Command-line interface for the unified backtest system

    Provides a comprehensive CLI that unifies all backtesting functionality
    into a single, easy-to-use interface.
    """

    def __init__(self):
        self.logger = logging.getLogger('unified_backtest_cli')
        self.engine = UnifiedBacktestEngine()
        self.registry = get_strategy_registry()
        self.parameter_manager = ParameterManager()
        self.config_manager = BacktestConfigManager()

        # Initialize components
        self.setup_logging()
        self.register_strategies()

    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def register_strategies(self):
        """Register strategies with the engine"""
        for name, metadata in self.registry.get_all_strategies().items():
            self.engine.register_strategy(
                name, metadata.strategy_class, None
            )

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser"""

        parser = argparse.ArgumentParser(
            description='Unified Backtest System - Comprehensive trading strategy backtesting',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Quick test
  python run_backtest.py --strategy ema --epic EURUSD --days 3

  # Multi-strategy comparison
  python run_backtest.py --compare-strategies ema,macd,kama

  # Parameter optimization
  python run_backtest.py --strategy ema --optimize --param-ranges "confidence:0.4-0.8:0.1"

  # Signal validation
  python run_backtest.py --validate-signal "2025-08-28 19:30:00" --strategy ema

  # Use template
  python run_backtest.py --template comprehensive

  # Load configuration
  python run_backtest.py --config my_config.json

Available Templates: quick_test, comprehensive, optimization, validation, comparison
            """
        )

        # Mode selection (mutually exclusive)
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument('--strategy',
                               help='Run single strategy backtest')
        mode_group.add_argument('--compare-strategies',
                               help='Compare multiple strategies (comma-separated)')
        mode_group.add_argument('--optimize', action='store_true',
                               help='Run parameter optimization')
        mode_group.add_argument('--validate-signal',
                               help='Validate specific signal (timestamp)')
        mode_group.add_argument('--template',
                               help='Use predefined template')
        mode_group.add_argument('--config',
                               help='Load configuration from file')

        # Core parameters
        parser.add_argument('--epic', '--epics', dest='epics',
                           help='Epic(s) to test (comma-separated or "all")')
        parser.add_argument('--timeframe', '--timeframes', dest='timeframes',
                           help='Timeframe(s) to test (comma-separated)')
        parser.add_argument('--days', type=int, default=7,
                           help='Number of days to backtest (default: 7)')

        # Strategy parameters
        parser.add_argument('--use-optimal-params', action='store_true', default=True,
                           help='Use database optimal parameters (default: True)')
        parser.add_argument('--no-optimal-params', action='store_true',
                           help='Disable database optimal parameters')

        # Smart Money analysis
        parser.add_argument('--smart-money', action='store_true',
                           help='Enable Smart Money Concepts analysis')
        parser.add_argument('--smart-money-confluence', type=float, default=0.6,
                           help='Smart Money confluence threshold (default: 0.6)')

        # Parameter optimization
        parser.add_argument('--param-ranges',
                           help='Parameter ranges for optimization (e.g., "confidence:0.4-0.8:0.1,short_ema:13-34:3")')
        parser.add_argument('--optimization-method', default='grid',
                           choices=['grid', 'random', 'genetic', 'bayesian'],
                           help='Optimization method (default: grid)')
        parser.add_argument('--max-combinations', type=int, default=100,
                           help='Maximum parameter combinations (default: 100)')
        parser.add_argument('--scoring-metric', default='win_rate',
                           choices=['win_rate', 'profit_factor', 'sharpe_ratio', 'total_return'],
                           help='Optimization scoring metric (default: win_rate)')

        # Output and display
        parser.add_argument('--show-signals', action='store_true',
                           help='Display individual signals')
        parser.add_argument('--max-signals', type=int, default=20,
                           help='Maximum signals to display (default: 20)')
        parser.add_argument('--export',
                           help='Export results to file (json, csv, html)')
        parser.add_argument('--export-format', default='json',
                           choices=['json', 'csv', 'html'],
                           help='Export format (default: json)')

        # Signal validation options
        parser.add_argument('--show-raw-data', action='store_true',
                           help='Show raw OHLC data (validation mode)')
        parser.add_argument('--show-calculations', action='store_true', default=True,
                           help='Show detailed calculations (validation mode)')
        parser.add_argument('--show-decision-tree', action='store_true', default=True,
                           help='Show decision tree (validation mode)')

        # System options
        parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
        parser.add_argument('--debug', action='store_true',
                           help='Debug logging')
        parser.add_argument('--quiet', '-q', action='store_true',
                           help='Quiet mode (errors only)')

        # Information commands
        parser.add_argument('--list-strategies', action='store_true',
                           help='List all available strategies')
        parser.add_argument('--list-templates', action='store_true',
                           help='List available configuration templates')
        parser.add_argument('--list-configs', action='store_true',
                           help='List saved configurations')
        parser.add_argument('--strategy-help',
                           help='Show help for specific strategy')

        # Configuration management
        parser.add_argument('--save-config',
                           help='Save current configuration to file')

        return parser

    def parse_epics(self, epics_str: str) -> List[str]:
        """Parse epics string into list"""
        if not epics_str:
            # When no epic specified, use all epics from config.EPIC_LIST (like old backtest files)
            epic_list = getattr(config, 'EPIC_LIST', None)
            if epic_list:
                print(f"🔍 Using all epics from config.EPIC_LIST: {len(epic_list)} epics")
                return epic_list
            else:
                print("⚠️ config.EPIC_LIST not found, using default")
                return [getattr(config, 'DEFAULT_EPIC', 'CS.D.EURUSD.CEEM.IP')]

        if epics_str.lower() == 'all':
            return getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP'])

        # Handle common epic formats
        epics = []
        for epic in epics_str.split(','):
            epic = epic.strip()

            # Convert simple pair to full epic format
            if len(epic) == 6 and epic.isalpha():  # e.g., EURUSD
                # EURUSD uses CEEM format, all others use MINI
                if epic.upper() == 'EURUSD':
                    epic = f"CS.D.{epic}.CEEM.IP"
                else:
                    epic = f"CS.D.{epic}.MINI.IP"

            epics.append(epic)

        return epics

    def parse_timeframes(self, timeframes_str: str) -> List[str]:
        """Parse timeframes string into list"""
        if not timeframes_str:
            return [getattr(config, 'DEFAULT_TIMEFRAME', '15m')]

        return [tf.strip() for tf in timeframes_str.split(',')]

    def run_information_command(self, args) -> bool:
        """Handle information commands that don't run backtests"""

        if args.list_strategies:
            self.display_available_strategies()
            return True

        if args.list_templates:
            self.display_available_templates()
            return True

        if args.list_configs:
            self.display_saved_configs()
            return True

        if args.strategy_help:
            self.display_strategy_help(args.strategy_help)
            return True

        return False

    def display_available_strategies(self):
        """Display all available strategies"""
        print("\n📋 Available Strategies:")
        print("=" * 50)

        strategies = self.registry.get_all_strategies()

        for name, metadata in strategies.items():
            print(f"\n• {name:<15} - {metadata.display_name}")
            print(f"  {metadata.description}")

            # Show capabilities
            capabilities = [cap.value for cap in metadata.capabilities]
            print(f"  Capabilities: {', '.join(capabilities)}")

            # Show timeframes
            print(f"  Timeframes: {', '.join(metadata.timeframes_supported)}")

            # Show optimization support
            opt_status = "✅" if metadata.requires_optimization else "❌"
            mtf_status = "✅" if metadata.has_mtf_support else "❌"
            print(f"  Optimization: {opt_status}  MTF: {mtf_status}")

    def display_available_templates(self):
        """Display available configuration templates"""
        print("\n📋 Available Templates:")
        print("=" * 50)

        templates = {
            'quick_test': 'Quick test configuration for rapid strategy validation',
            'comprehensive': 'Comprehensive multi-strategy analysis with Smart Money',
            'optimization': 'Parameter optimization using grid search',
            'validation': 'Detailed signal validation and inspection',
            'comparison': 'Compare performance across multiple strategies'
        }

        for name, description in templates.items():
            print(f"\n• {name}")
            print(f"  {description}")

            # Show template details
            try:
                template_config = self.config_manager.get_template(name)
                print(f"  Strategies: {', '.join(template_config.strategies)}")
                print(f"  Mode: {template_config.mode.value}")
                print(f"  Days: {template_config.days}")
            except Exception:
                pass

    def display_saved_configs(self):
        """Display saved configurations"""
        print("\n📋 Saved Configurations:")
        print("=" * 50)

        configs = self.config_manager.list_configs()

        if not configs:
            print("No saved configurations found.")
            return

        for config_info in configs:
            print(f"\n• {config_info['filename']}")
            print(f"  Name: {config_info['config_name']}")
            print(f"  Description: {config_info['description']}")
            print(f"  Mode: {config_info['mode']}")
            print(f"  Strategies: {', '.join(config_info['strategies'])}")
            print(f"  Created: {config_info['created_at']}")

    def display_strategy_help(self, strategy_name: str):
        """Display help for a specific strategy"""
        help_text = self.registry.get_strategy_help(strategy_name)
        print(help_text)

    def build_config_from_args(self, args) -> UnifiedBacktestConfig:
        """Build configuration from command line arguments"""

        # Determine mode
        if args.validate_signal:
            mode = BacktestMode.VALIDATION
            strategies = [args.strategy] if args.strategy else ['ema']
        elif args.optimize:
            mode = BacktestMode.PARAMETER_SWEEP
            strategies = [args.strategy] if args.strategy else ['ema']
        elif args.compare_strategies:
            mode = BacktestMode.COMPARISON
            strategies = [s.strip() for s in args.compare_strategies.split(',')]
        else:
            mode = BacktestMode.SINGLE_STRATEGY
            strategies = [args.strategy] if args.strategy else ['ema']

        # Parse epics and timeframes
        epics = self.parse_epics(args.epics)
        timeframes = self.parse_timeframes(args.timeframes)

        # Smart Money configuration
        smart_money_config = SmartMoneyConfig(
            enabled=args.smart_money,
            min_confluence_score=args.smart_money_confluence
        )

        # Optimization configuration
        optimization_config = OptimizationConfig()
        if args.optimize:
            optimization_config.enabled = True
            optimization_config.method = OptimizationMethod(args.optimization_method)
            optimization_config.scoring_metric = args.scoring_metric
            optimization_config.max_combinations = args.max_combinations

            if args.param_ranges:
                optimization_config.parameter_ranges = self.parameter_manager.parse_parameter_ranges(
                    args.param_ranges
                )

        # Output configuration
        output_config = OutputConfig(
            show_signals=args.show_signals,
            max_signals_display=args.max_signals,
            verbose=args.verbose,
            export_format=args.export_format if args.export else None,
            export_path=args.export
        )

        # Validation configuration
        validation_config = ValidationConfig()
        if args.validate_signal:
            validation_config.enabled = True
            validation_config.target_timestamp = args.validate_signal
            validation_config.show_raw_data = args.show_raw_data
            validation_config.show_calculations = args.show_calculations
            validation_config.show_decision_tree = args.show_decision_tree

        # Determine optimal parameters usage
        use_optimal_params = not args.no_optimal_params if args.no_optimal_params else args.use_optimal_params

        return UnifiedBacktestConfig(
            mode=mode,
            strategies=strategies,
            epics=epics,
            timeframes=timeframes,
            days=args.days,
            use_optimal_parameters=use_optimal_params,
            smart_money=smart_money_config,
            optimization=optimization_config,
            output=output_config,
            validation=validation_config
        )

    def run(self, args_list: List[str] = None) -> int:
        """
        Main execution method

        Args:
            args_list: Command line arguments (uses sys.argv if None)

        Returns:
            Exit code (0 for success, 1 for error)
        """

        try:
            # Parse arguments
            parser = self.create_parser()

            if args_list is None:
                args = parser.parse_args()
            else:
                args = parser.parse_args(args_list)

            # Setup logging level
            if args.debug:
                self.setup_logging("DEBUG")
            elif args.verbose:
                self.setup_logging("INFO")
            elif args.quiet:
                self.setup_logging("ERROR")

            # Handle information commands
            if self.run_information_command(args):
                return 0

            # Build configuration
            if args.config:
                # Load from file
                backtest_config = self.config_manager.load_config(args.config)
                self.logger.info(f"📋 Loaded configuration: {args.config}")
            elif args.template:
                # Use template
                backtest_config = self.config_manager.get_template(args.template)
                self.logger.info(f"📋 Using template: {args.template}")
            else:
                # Build from arguments
                backtest_config = self.build_config_from_args(args)

            # Validate configuration
            is_valid, errors = self.config_manager.validate_config(backtest_config)
            if not is_valid:
                self.logger.error("❌ Configuration validation failed:")
                for error in errors:
                    self.logger.error(f"   • {error}")
                return 1

            # Save configuration if requested
            if args.save_config:
                saved_path = self.config_manager.save_config(backtest_config, args.save_config)
                self.logger.info(f"💾 Configuration saved: {saved_path}")

            # Run backtest
            self.logger.info("🚀 Starting unified backtest...")

            results = self.engine.run_backtest(backtest_config)

            if not results:
                self.logger.error("❌ No results generated")
                return 1

            # Generate and display report
            try:
                report_generator = BacktestReportGenerator()
                report_generator.generate_report(results, backtest_config)

                # Export if requested
                if backtest_config.output.export_path:
                    report_generator.export_results(
                        results,
                        backtest_config.output.export_path,
                        backtest_config.output.export_format
                    )

            except ImportError:
                self.logger.warning("⚠️ Report generator not available, showing basic results")
                self._display_basic_results(results, backtest_config)

            # Display session statistics
            stats = self.engine.get_session_statistics()
            self.logger.info(f"\n📊 Session Statistics:")
            self.logger.info(f"   Execution time: {stats['execution_time']:.1f}s")
            self.logger.info(f"   Total signals: {stats['total_signals_found']}")
            self.logger.info(f"   Successful runs: {stats['successful_runs']}")
            self.logger.info(f"   Failed runs: {stats['failed_runs']}")

            return 0

        except KeyboardInterrupt:
            self.logger.info("🛑 Interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            if args.debug if 'args' in locals() else False:
                traceback.print_exc()
            return 1

    def _display_basic_results(self, results, config):
        """Display basic results when report generator is not available"""

        self.logger.info("\n📊 BACKTEST RESULTS:")
        self.logger.info("=" * 50)

        total_signals = 0
        successful_runs = 0

        for result in results:
            if result.error:
                self.logger.error(f"❌ {result.strategy} - {result.epic}: {result.error}")
            else:
                signals_count = len(result.signals)
                total_signals += signals_count
                successful_runs += 1

                self.logger.info(f"✅ {result.strategy} - {result.epic} ({result.timeframe}): "
                               f"{signals_count} signals in {result.execution_time:.1f}s")

                # Show basic performance metrics
                if result.performance:
                    win_rate = result.performance.get('win_rate', 0)
                    avg_profit = result.performance.get('average_profit_pips', 0)
                    total_signals_perf = result.performance.get('total_signals', 0)

                    self.logger.info(f"   Performance: {win_rate:.1%} win rate, "
                                   f"{avg_profit:.1f} avg profit pips, "
                                   f"{total_signals_perf} signals analyzed")

        self.logger.info(f"\n📈 Summary: {total_signals} total signals, "
                        f"{successful_runs}/{len(results)} successful runs")


def main():
    """Main entry point"""
    cli = UnifiedBacktestCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()