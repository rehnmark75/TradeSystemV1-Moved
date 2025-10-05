#!/usr/bin/env python3
# backtest_cli.py
"""
Enhanced Backtest CLI - Standalone command-line interface for the new backtest pipeline
Provides clean, focused interface for backtesting with detailed signal analysis

Usage:
    python backtest_cli.py --days 7                    # Backtest all epics for 7 days
    python backtest_cli.py --epic CS.D.EURUSD.CEEM.IP --days 14 --show-signals
    python backtest_cli.py --days 3 --show-signals --strategy EMA_CROSSOVER
"""

import argparse
import logging
import sys
from datetime import datetime

# Set up path
sys.path.insert(0, '/app/forex_scanner')

try:
    import config
    from commands.enhanced_backtest_commands import EnhancedBacktestCommands
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the correct directory")
    sys.exit(1)


class BacktestCLI:
    """Clean, focused CLI for enhanced backtesting"""

    def __init__(self):
        self.enhanced_backtest = EnhancedBacktestCommands()
        self.setup_logging()

    def setup_logging(self, verbose: bool = False):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""

        parser = argparse.ArgumentParser(
            description="Enhanced Backtest CLI - Test trading strategies with detailed signal analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python backtest_cli.py --days 7                              # Test all pairs for 7 days (strategy only)
  python backtest_cli.py --epic CS.D.EURUSD.CEEM.IP --days 14  # Test EURUSD for 14 days
  python backtest_cli.py --days 3 --show-signals               # Show detailed signal breakdown
  python backtest_cli.py --days 7 --strategy MACD_CROSSOVER    # Test specific strategy
  python backtest_cli.py --quick-test --epic CS.D.GBPUSD.MINI.IP  # Quick 24h test
  python backtest_cli.py --days 3 --pipeline                   # Full pipeline test (with validation)
  python backtest_cli.py --days 7 --strategy MACD --pipeline   # MACD strategy with full pipeline

Mode Comparison:
  Default (Basic):  Fast strategy testing for parameter optimization
  --pipeline:       Full production pipeline with validation, filtering & market intelligence

Signal Display Format:
  When using --show-signals, you'll see detailed breakdown like:

  CS.D.EURUSD.CEEM.IP: 7 signals
  CS.D.GBPUSD.MINI.IP: 15 signals
  ...

  #   TIMESTAMP            PAIR     TYPE STRATEGY        PRICE    CONF   PROFIT   LOSS     R:R
  1   2025-09-24 12:30:00 UTC EEM.IP   SELL ema_modular     1.17439  85.0%  15.0     0.0      inf
  2   2025-09-23 14:30:00 UTC EEM.IP   BUY  ema_modular     1.18000  95.0%  0.0      10.0     0.00
            """
        )

        # Core backtest options
        parser.add_argument(
            '--epic',
            type=str,
            help='Specific epic to test (e.g., CS.D.EURUSD.CEEM.IP). If not provided, tests all epics from config.'
        )

        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Number of days to backtest (default: 7)'
        )

        parser.add_argument(
            '--show-signals',
            action='store_true',
            help='Show detailed signal breakdown with timestamps, prices, and outcomes'
        )

        # Strategy and timeframe options
        parser.add_argument(
            '--strategy',
            type=str,
            default='EMA_CROSSOVER',
            help='Strategy name to use for backtest (default: EMA_CROSSOVER)'
        )

        parser.add_argument(
            '--pipeline',
            action='store_true',
            help='Run full signal pipeline with validation, filtering, and market intelligence (slower but production-ready). Default: basic strategy testing only.'
        )

        parser.add_argument(
            '--timeframe',
            type=str,
            default='15m',
            choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
            help='Trading timeframe (default: 15m)'
        )

        # Display options
        parser.add_argument(
            '--max-signals',
            type=int,
            default=20,
            help='Maximum number of signals to display in detail (default: 20)'
        )

        # Quick test options
        parser.add_argument(
            '--quick-test',
            action='store_true',
            help='Run quick 24-hour backtest with signal details'
        )

        parser.add_argument(
            '--hours',
            type=int,
            default=24,
            help='Hours for quick test (default: 24, used with --quick-test)'
        )

        # System options
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Verbose logging output'
        )

        parser.add_argument(
            '--cleanup',
            action='store_true',
            help='Clean up old test executions before running'
        )

        return parser

    def execute_command(self, args) -> bool:
        """Execute the backtest command based on arguments"""

        try:
            # Setup logging level
            self.setup_logging(args.verbose)

            # Cleanup if requested
            if args.cleanup:
                self.enhanced_backtest.cleanup_test_executions()

            # Quick test mode
            if args.quick_test:
                if not args.epic:
                    print("❌ --epic is required for quick test mode")
                    return False

                return self.enhanced_backtest.quick_enhanced_backtest(
                    epic=args.epic,
                    hours=args.hours,
                    show_signals=True,  # Always show signals for quick test
                    pipeline=args.pipeline
                )

            # Standard backtest mode
            return self.enhanced_backtest.run_enhanced_backtest(
                epic=args.epic,
                days=args.days,
                show_signals=args.show_signals,
                timeframe=args.timeframe,
                strategy=args.strategy,
                max_signals_display=args.max_signals,
                pipeline=args.pipeline
            )

        except KeyboardInterrupt:
            print("\n⚠️ Backtest interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Backtest execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False

    def show_usage_examples(self):
        """Show usage examples"""
        examples = [
            "# Test all currency pairs for 7 days",
            "python backtest_cli.py --days 7",
            "",
            "# Test specific pair with detailed signals",
            "python backtest_cli.py --epic CS.D.EURUSD.CEEM.IP --days 14 --show-signals",
            "",
            "# Quick 24-hour test",
            "python backtest_cli.py --quick-test --epic CS.D.GBPUSD.MINI.IP",
            "",
            "# Test specific strategy with custom timeframe",
            "python backtest_cli.py --strategy MACD_CROSSOVER --timeframe 5m --days 3 --show-signals",
            "",
            "# Clean up and run comprehensive test",
            "python backtest_cli.py --cleanup --days 14 --show-signals --verbose"
        ]

        print("\n📚 Usage Examples:")
        print("=" * 50)
        for example in examples:
            print(example)
        print("=" * 50)

    def run(self):
        """Main run method"""
        parser = self.create_parser()

        # If no arguments provided, show help
        if len(sys.argv) == 1:
            print("🧪 Enhanced Backtest CLI")
            print("=" * 30)
            parser.print_help()
            self.show_usage_examples()
            return 0

        args = parser.parse_args()

        # Execute the command
        success = self.execute_command(args)

        return 0 if success else 1


def main():
    """Main entry point"""
    try:
        cli = BacktestCLI()
        exit_code = cli.run()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()