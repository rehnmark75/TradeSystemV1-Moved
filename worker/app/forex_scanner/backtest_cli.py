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

# Set up path - add both /app and /app/forex_scanner for proper imports
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

try:
    from forex_scanner import config
    from forex_scanner.commands.enhanced_backtest_commands import EnhancedBacktestCommands
except ImportError:
    # Fallback for running from within forex_scanner directory
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
            help='Number of days to backtest (default: 7). Ignored if --start-date and --end-date are provided.'
        )

        parser.add_argument(
            '--start-date',
            type=str,
            help='Start date for backtest in YYYY-MM-DD format (e.g., 2025-10-06). Requires --end-date.'
        )

        parser.add_argument(
            '--end-date',
            type=str,
            help='End date for backtest in YYYY-MM-DD format (e.g., 2025-11-05). Requires --start-date.'
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
            default='SMC_SIMPLE',
            help='Strategy name to use for backtest (default: SMC_SIMPLE)'
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

        parser.add_argument(
            '--csv-export',
            type=str,
            metavar='FILEPATH',
            help='Export all signals to CSV file for detailed analysis (e.g., /tmp/signals.csv)'
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

        # Parameter override options for backtest isolation
        parser.add_argument(
            '--override',
            action='append',
            metavar='PARAM=VALUE',
            help='Override strategy parameter for this backtest only (can be used multiple times). '
                 'Example: --override fixed_stop_loss_pips=12 --override min_confidence=0.55. '
                 'Does NOT affect live trading configuration.'
        )

        # Snapshot support for persistent parameter configurations
        parser.add_argument(
            '--snapshot',
            type=str,
            metavar='NAME',
            help='Load parameter overrides from a saved snapshot. '
                 'Create snapshots with: python snapshot_cli.py create NAME --set PARAM=VALUE. '
                 'List available: python snapshot_cli.py list'
        )

        parser.add_argument(
            '--save-snapshot',
            type=str,
            metavar='NAME',
            help='Save current test results to a snapshot after backtest completes. '
                 'Can be combined with --override to create a new snapshot.'
        )

        # Historical market intelligence replay (Phase 3)
        parser.add_argument(
            '--use-historical-intelligence',
            action='store_true',
            default=True,
            dest='use_historical_intelligence',
            help='Use stored market intelligence from database instead of recalculating (default: True). '
                 'This ensures backtest results match what live trading would have done.'
        )

        parser.add_argument(
            '--no-historical-intelligence',
            action='store_false',
            dest='use_historical_intelligence',
            help='Force recalculation of market intelligence (ignore stored data). '
                 'Useful for testing intelligence calculation changes.'
        )

        return parser

    def _parse_overrides(self, override_args) -> dict:
        """
        Parse --override PARAM=VALUE arguments into a dictionary.

        Args:
            override_args: List of strings like ['fixed_stop_loss_pips=12', 'min_confidence=0.55']

        Returns:
            Dict of parameter overrides with auto-converted types
        """
        if not override_args:
            return None

        overrides = {}
        for item in override_args:
            if '=' not in item:
                print(f"⚠️ Invalid override format '{item}' - expected PARAM=VALUE, skipping")
                continue

            key, value = item.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Auto-convert types
            try:
                if value.lower() in ('true', 'false'):
                    overrides[key] = value.lower() == 'true'
                elif '.' in value:
                    overrides[key] = float(value)
                else:
                    overrides[key] = int(value)
            except ValueError:
                overrides[key] = value  # Keep as string

        if overrides:
            print(f"🧪 Parameter overrides parsed: {len(overrides)} parameters")
            for k, v in overrides.items():
                print(f"   - {k}: {v} ({type(v).__name__})")

        return overrides if overrides else None

    def _load_snapshot_overrides(self, snapshot_name: str) -> dict:
        """
        Load parameter overrides from a saved snapshot.

        Args:
            snapshot_name: Name of the snapshot to load

        Returns:
            Dict of parameter overrides from the snapshot, or None if not found
        """
        try:
            from forex_scanner.services.backtest_config_service import get_backtest_config_service
        except ImportError:
            from services.backtest_config_service import get_backtest_config_service

        service = get_backtest_config_service()
        snapshot = service.get_snapshot(snapshot_name)

        if not snapshot:
            print(f"❌ Snapshot '{snapshot_name}' not found")
            print("   Use 'python snapshot_cli.py list' to see available snapshots")
            return None

        overrides = snapshot.parameter_overrides
        print(f"📦 Loaded snapshot '{snapshot_name}' ({len(overrides)} parameters)")
        for k, v in overrides.items():
            print(f"   - {k}: {v}")

        return overrides

    def _save_test_results_to_snapshot(
        self,
        snapshot_name: str,
        overrides: dict,
        results: dict,
        epic: str = None,
        days: int = None
    ) -> bool:
        """
        Save test results to a snapshot (create new or update existing).

        Args:
            snapshot_name: Name for the snapshot
            overrides: Parameter overrides used in the test
            results: Test results dict
            epic: Epic tested (optional)
            days: Days tested (optional)

        Returns:
            True if saved successfully
        """
        try:
            from forex_scanner.services.backtest_config_service import get_backtest_config_service
        except ImportError:
            from services.backtest_config_service import get_backtest_config_service

        service = get_backtest_config_service()

        # Check if snapshot exists
        existing = service.get_snapshot(snapshot_name)

        if existing:
            # Update existing snapshot with new test results
            success = service.update_test_results(
                name=snapshot_name,
                execution_id=results.get('execution_id', 0),
                results=results,
                epic_tested=epic,
                days_tested=days
            )
            if success:
                print(f"📊 Updated snapshot '{snapshot_name}' with test results")
            return success
        else:
            # Create new snapshot
            if not overrides:
                print(f"⚠️ Cannot create snapshot without parameter overrides")
                print("   Use --override to specify parameters for the new snapshot")
                return False

            snapshot_id = service.create_snapshot(
                name=snapshot_name,
                parameter_overrides=overrides,
                description=f"Created from backtest: {epic or 'all pairs'}, {days} days",
                created_by='backtest_cli'
            )

            if snapshot_id:
                # Update with test results
                service.update_test_results(
                    name=snapshot_name,
                    execution_id=results.get('execution_id', 0),
                    results=results,
                    epic_tested=epic,
                    days_tested=days
                )
                print(f"📦 Created snapshot '{snapshot_name}' with test results")
                return True
            return False

    def execute_command(self, args) -> bool:
        """Execute the backtest command based on arguments"""

        try:
            # Setup logging level
            self.setup_logging(args.verbose)

            # Parse config overrides - snapshot takes precedence, then inline overrides
            config_override = None

            # Load from snapshot if specified
            if getattr(args, 'snapshot', None):
                config_override = self._load_snapshot_overrides(args.snapshot)
                if config_override is None:
                    return False  # Snapshot not found

            # Parse inline overrides (these merge with/override snapshot values)
            inline_overrides = self._parse_overrides(getattr(args, 'override', None))
            if inline_overrides:
                if config_override:
                    # Merge: inline overrides take precedence
                    config_override.update(inline_overrides)
                    print(f"   (merged with {len(inline_overrides)} inline overrides)")
                else:
                    config_override = inline_overrides

            # Validate date range parameters
            if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
                print("❌ Both --start-date and --end-date must be provided together")
                return False

            # Parse dates if provided
            start_date = None
            end_date = None
            if args.start_date and args.end_date:
                try:
                    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
                    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

                    if start_date >= end_date:
                        print("❌ --start-date must be before --end-date")
                        return False

                    # Validate dates are not in the future
                    if end_date > datetime.now():
                        print("❌ --end-date cannot be in the future")
                        return False

                except ValueError as e:
                    print(f"❌ Invalid date format. Use YYYY-MM-DD (e.g., 2025-10-06)")
                    return False

            # Cleanup if requested
            if args.cleanup:
                self.enhanced_backtest.cleanup_test_executions()

            # Get historical intelligence flag
            use_historical_intelligence = getattr(args, 'use_historical_intelligence', True)
            if not use_historical_intelligence:
                print("📚 Historical intelligence: DISABLED (will recalculate from data)")
            else:
                print("📚 Historical intelligence: ENABLED (will replay stored data)")

            # Quick test mode
            if args.quick_test:
                if not args.epic:
                    print("❌ --epic is required for quick test mode")
                    return False

                return self.enhanced_backtest.quick_enhanced_backtest(
                    epic=args.epic,
                    hours=args.hours,
                    show_signals=True,  # Always show signals for quick test
                    pipeline=args.pipeline,
                    config_override=config_override,
                    use_historical_intelligence=use_historical_intelligence
                )

            # Standard backtest mode
            return self.enhanced_backtest.run_enhanced_backtest(
                epic=args.epic,
                days=args.days,
                start_date=start_date,
                end_date=end_date,
                show_signals=args.show_signals,
                timeframe=args.timeframe,
                strategy=args.strategy,
                max_signals_display=args.max_signals,
                pipeline=args.pipeline,
                csv_export=args.csv_export if hasattr(args, 'csv_export') else None,
                config_override=config_override,
                use_historical_intelligence=use_historical_intelligence
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