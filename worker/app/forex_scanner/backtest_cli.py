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
        import sys
        level = logging.DEBUG if verbose else logging.INFO

        # Force reconfigure logging to ensure output goes to stdout
        # This is necessary because other modules may have already configured logging
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Remove existing handlers and add a fresh stdout handler
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

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
            '--scalp',
            action='store_true',
            help='Enable scalping mode with Virtual Stop Loss (VSL) emulation. '
                 'Uses per-pair VSL values: 3 pips for majors, 4 pips for JPY pairs. '
                 'Take profit defaults to 5 pips. Override with --override scalp_tp_pips=X.'
        )

        parser.add_argument(
            '--scalp-offset',
            type=float,
            default=None,
            metavar='PIPS',
            help='Override the limit order offset (momentum confirmation) for scalp mode. '
                 'Default is 1 pip. Higher values require more price movement before entry.'
        )

        parser.add_argument(
            '--scalp-expiry',
            type=int,
            default=None,
            metavar='MINUTES',
            help='Override the limit order expiry time for scalp mode. '
                 'Default is 7 minutes. Shorter values cancel unfilled orders faster.'
        )

        # Scalp Tier Settings
        parser.add_argument(
            '--scalp-htf',
            type=str,
            default=None,
            metavar='TF',
            help='Override TIER 1 HTF timeframe for scalp mode. '
                 'Default is 15m. Options: 5m, 15m, 30m, 1h.'
        )

        parser.add_argument(
            '--scalp-ema',
            type=int,
            default=None,
            metavar='PERIOD',
            help='Override TIER 1 EMA period for scalp mode. '
                 'Default is 20. Common values: 10, 20, 50.'
        )

        parser.add_argument(
            '--scalp-swing-lookback',
            type=int,
            default=None,
            metavar='BARS',
            help='Override TIER 2 swing lookback bars for scalp mode. '
                 'Default is 12. Range: 5-30 bars.'
        )

        parser.add_argument(
            '--scalp-trigger-tf',
            type=str,
            default=None,
            choices=['1m', '5m', '15m'],
            help='Override TIER 2 trigger timeframe for scalp mode. '
                 'Default is 5m.'
        )

        parser.add_argument(
            '--scalp-entry-tf',
            type=str,
            default=None,
            choices=['1m', '5m'],
            help='Override TIER 3 entry timeframe for scalp mode. '
                 'Default is 1m.'
        )

        parser.add_argument(
            '--scalp-confidence',
            type=float,
            default=None,
            metavar='CONF',
            help='Override minimum confidence threshold for scalp mode. '
                 'Default is 0.30 (30%%). Range: 0.0-1.0.'
        )

        parser.add_argument(
            '--scalp-cooldown',
            type=int,
            default=None,
            metavar='MINS',
            help='Override cooldown between scalp trades in minutes. '
                 'Default is 15 minutes.'
        )

        parser.add_argument(
            '--scalp-tolerance',
            type=float,
            default=None,
            metavar='PIPS',
            help='Override swing break tolerance in pips for scalp mode. '
                 'Default is 0.5 pips. Allows entries when price is very close to swing level.'
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
            default=False,
            dest='use_historical_intelligence',
            help='Use stored market intelligence from database instead of recalculating (default: False). '
                 'This ensures backtest results match what live trading would have done.'
        )

        parser.add_argument(
            '--no-historical-intelligence',
            action='store_false',
            dest='use_historical_intelligence',
            help='Force recalculation of market intelligence (ignore stored data). '
                 'Useful for testing intelligence calculation changes.'
        )

        # Parallel execution options
        parser.add_argument(
            '--parallel',
            action='store_true',
            help='Run backtest in parallel using chunked execution. '
                 'Splits the time period into chunks and processes them concurrently.'
        )

        parser.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of parallel workers for chunked execution (default: 4). '
                 'Only used with --parallel flag.'
        )

        parser.add_argument(
            '--chunk-days',
            type=int,
            default=7,
            help='Days per chunk for parallel execution (default: 7). '
                 'Only used with --parallel flag.'
        )

        # Chart generation options
        parser.add_argument(
            '--chart',
            action='store_true',
            help='Generate visual chart with signals plotted on price data.'
        )

        parser.add_argument(
            '--chart-output',
            type=str,
            metavar='FILEPATH',
            help='Save chart to specified file path (e.g., /tmp/backtest_chart.png). '
                 'If not specified, chart is saved to default location.'
        )

        # Parameter variation options
        parser.add_argument(
            '--vary',
            action='append',
            metavar='PARAM=SPEC',
            help='Vary parameter across range or list. '
                 'Formats: param=start:end:step (e.g., fixed_stop_loss_pips=8:12:2) '
                 'OR param=val1,val2,val3 (e.g., min_confidence=0.45,0.50,0.55). '
                 'Can be used multiple times for multiple parameters.'
        )

        parser.add_argument(
            '--vary-json',
            type=str,
            metavar='JSON_OR_FILE',
            help='JSON parameter grid. Can be inline JSON or path to .json file. '
                 'Example: \'{"fixed_stop_loss_pips": [8,10,12], "min_confidence": [0.45,0.50]}\''
        )

        parser.add_argument(
            '--vary-workers',
            type=int,
            default=4,
            metavar='N',
            help='Number of parallel workers for variation testing (default: 4).'
        )

        parser.add_argument(
            '--rank-by',
            type=str,
            default='composite_score',
            choices=['win_rate', 'total_pips', 'profit_factor', 'composite_score', 'expectancy'],
            help='Metric to rank variation results by (default: composite_score).'
        )

        parser.add_argument(
            '--top-n',
            type=int,
            default=10,
            metavar='N',
            help='Show top N variation results (default: 10).'
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

    def _build_scalp_config_overrides(
        self,
        epic: str = None,
        existing_overrides: dict = None,
        scalp_offset: float = None,
        scalp_expiry: int = None,
        scalp_htf: str = None,
        scalp_ema: int = None,
        scalp_swing_lookback: int = None,
        scalp_trigger_tf: str = None,
        scalp_entry_tf: str = None,
        scalp_confidence: float = None,
        scalp_cooldown: int = None,
        scalp_tolerance: float = None
    ) -> dict:
        """
        Build config overrides for scalp mode with Virtual Stop Loss (VSL) emulation.

        VSL allows tighter stop losses than IG's broker minimum by monitoring prices
        in real-time and closing programmatically. In backtest, we emulate this by
        using the VSL values as the stop loss distance.

        Per-pair VSL values (matching live trading):
        - Majors (EURUSD, GBPUSD, etc.): 3 pips
        - JPY pairs (USDJPY, EURJPY, etc.): 4 pips

        Args:
            epic: Specific epic to test (used for display purposes)
            existing_overrides: Any existing overrides (to check for scalp_tp_pips override)
            scalp_offset: Override for limit order offset in pips (default: 1 pip)
            scalp_expiry: Override for limit order expiry in minutes (default: 5 min)
            scalp_htf: Override TIER 1 HTF timeframe (default: 15m)
            scalp_ema: Override TIER 1 EMA period (default: 20)
            scalp_swing_lookback: Override TIER 2 swing lookback bars (default: 12)
            scalp_trigger_tf: Override TIER 2 trigger timeframe (default: 5m)
            scalp_entry_tf: Override TIER 3 entry timeframe (default: 1m)
            scalp_confidence: Override minimum confidence threshold (default: 0.30)
            scalp_cooldown: Override cooldown between trades in minutes (default: 15)
            scalp_tolerance: Override swing break tolerance in pips (default: 0.5)

        Returns:
            Dict of scalp mode config overrides
        """
        try:
            from forex_scanner.config_virtual_stop_backtest import (
                get_vsl_pips, DEFAULT_SCALP_TP_PIPS, PAIR_VSL_CONFIGS
            )
        except ImportError:
            from config_virtual_stop_backtest import (
                get_vsl_pips, DEFAULT_SCALP_TP_PIPS, PAIR_VSL_CONFIGS
            )

        # Get scalp TP from overrides or use default
        scalp_tp = DEFAULT_SCALP_TP_PIPS
        if existing_overrides and 'scalp_tp_pips' in existing_overrides:
            scalp_tp = existing_overrides['scalp_tp_pips']

        # Get scalp offset from CLI arg, existing overrides, or use default (1 pip for scalp mode)
        default_scalp_offset = 1.0  # Default scalp offset is 1 pip (tighter than normal 3 pips)
        if scalp_offset is not None:
            offset = scalp_offset
        elif existing_overrides and 'scalp_limit_offset_pips' in existing_overrides:
            offset = existing_overrides['scalp_limit_offset_pips']
        else:
            offset = default_scalp_offset

        # Get scalp expiry from CLI arg, existing overrides, or use default (7 min for scalp mode)
        default_scalp_expiry = 7  # Default scalp expiry is 7 minutes (faster than normal 45 min)
        if scalp_expiry is not None:
            expiry = scalp_expiry
        elif existing_overrides and 'limit_expiry_minutes' in existing_overrides:
            expiry = existing_overrides['limit_expiry_minutes']
        else:
            expiry = default_scalp_expiry

        overrides = {
            'scalp_mode_enabled': True,
            'scalp_tp_pips': scalp_tp,
            'scalp_limit_offset_pips': offset,  # Limit order offset for momentum confirmation
            'limit_expiry_minutes': expiry,  # Limit order expiry time
            'use_vsl_mode': True,  # Flag for BacktestScanner to use VSL simulation
        }

        # Add tier settings if overridden
        if scalp_htf is not None:
            overrides['scalp_htf_timeframe'] = scalp_htf
        if scalp_ema is not None:
            overrides['scalp_ema_period'] = scalp_ema
        if scalp_swing_lookback is not None:
            overrides['scalp_swing_lookback_bars'] = scalp_swing_lookback
        if scalp_trigger_tf is not None:
            overrides['scalp_trigger_timeframe'] = scalp_trigger_tf
        if scalp_entry_tf is not None:
            overrides['scalp_entry_timeframe'] = scalp_entry_tf
        if scalp_confidence is not None:
            overrides['scalp_min_confidence'] = scalp_confidence
        if scalp_cooldown is not None:
            overrides['scalp_cooldown_minutes'] = scalp_cooldown
        if scalp_tolerance is not None:
            overrides['scalp_swing_break_tolerance_pips'] = scalp_tolerance

        # Display VSL configuration
        print(f"\n🎯 Scalp Mode Configuration (Virtual Stop Loss Emulation):")
        print(f"   Take Profit: {scalp_tp} pips")
        print(f"   Order Offset: {offset} pips (momentum confirmation)")
        print(f"   Order Expiry: {expiry} minutes")

        # Display tier settings
        htf_display = scalp_htf if scalp_htf else "15m (default)"
        ema_display = scalp_ema if scalp_ema else "20 (default)"
        swing_display = scalp_swing_lookback if scalp_swing_lookback else "12 (default)"
        trigger_display = scalp_trigger_tf if scalp_trigger_tf else "5m (default)"
        entry_display = scalp_entry_tf if scalp_entry_tf else "1m (default)"
        conf_display = f"{scalp_confidence:.0%}" if scalp_confidence else "30% (default)"
        cooldown_display = f"{scalp_cooldown} min" if scalp_cooldown else "15 min (default)"
        tolerance_display = f"{scalp_tolerance} pips" if scalp_tolerance else "0.5 pips (default)"
        print(f"   TIER 1 HTF: {htf_display}")
        print(f"   TIER 1 EMA: {ema_display}")
        print(f"   TIER 2 Trigger TF: {trigger_display}")
        print(f"   TIER 2 Swing Lookback: {swing_display} bars")
        print(f"   TIER 3 Entry TF: {entry_display}")
        print(f"   Min Confidence: {conf_display}")
        print(f"   Cooldown: {cooldown_display}")
        print(f"   Swing Break Tolerance: {tolerance_display}")

        if epic:
            # Single epic - show its VSL
            vsl_pips = get_vsl_pips(epic)
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
            print(f"   {pair}: VSL = {vsl_pips} pips")
        else:
            # Multiple epics - show summary
            print(f"   Per-pair Virtual Stop Loss:")
            # Group by VSL value
            majors = [e for e, c in PAIR_VSL_CONFIGS.items() if c['vsl_pips'] == 3.0]
            jpy_pairs = [e for e, c in PAIR_VSL_CONFIGS.items() if c['vsl_pips'] == 4.0]

            if majors:
                major_names = [e.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '') for e in majors[:4]]
                print(f"   - Majors (3 pips): {', '.join(major_names)}...")
            if jpy_pairs:
                jpy_names = [e.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '') for e in jpy_pairs[:4]]
                print(f"   - JPY pairs (4 pips): {', '.join(jpy_names)}")

        print()
        return overrides

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

            # Handle scalp mode with VSL emulation
            if getattr(args, 'scalp', False):
                scalp_offset = getattr(args, 'scalp_offset', None)
                scalp_expiry = getattr(args, 'scalp_expiry', None)
                scalp_htf = getattr(args, 'scalp_htf', None)
                scalp_ema = getattr(args, 'scalp_ema', None)
                scalp_swing_lookback = getattr(args, 'scalp_swing_lookback', None)
                scalp_trigger_tf = getattr(args, 'scalp_trigger_tf', None)
                scalp_entry_tf = getattr(args, 'scalp_entry_tf', None)
                scalp_confidence = getattr(args, 'scalp_confidence', None)
                scalp_cooldown = getattr(args, 'scalp_cooldown', None)
                scalp_tolerance = getattr(args, 'scalp_tolerance', None)
                scalp_overrides = self._build_scalp_config_overrides(
                    args.epic, config_override, scalp_offset, scalp_expiry,
                    scalp_htf, scalp_ema, scalp_swing_lookback,
                    scalp_trigger_tf, scalp_entry_tf, scalp_confidence,
                    scalp_cooldown, scalp_tolerance
                )
                if config_override:
                    config_override.update(scalp_overrides)
                else:
                    config_override = scalp_overrides

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

            # Parameter variation mode
            if getattr(args, 'vary', None) or getattr(args, 'vary_json', None):
                return self._run_param_variations(
                    args=args,
                    start_date=start_date,
                    end_date=end_date,
                    use_historical_intelligence=use_historical_intelligence,
                    base_config_override=config_override  # Pass scalp mode config to merge with variations
                )

            # Parallel backtest mode
            if args.parallel:
                return self._run_parallel_backtest(
                    args=args,
                    start_date=start_date,
                    end_date=end_date,
                    config_override=config_override,
                    use_historical_intelligence=use_historical_intelligence
                )

            # Standard backtest mode
            # If chart is requested, we need the results dict to get execution_id
            need_results = args.chart
            result = self.enhanced_backtest.run_enhanced_backtest(
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
                use_historical_intelligence=use_historical_intelligence,
                return_results=need_results
            )

            # Generate chart if requested (for standard backtest)
            if args.chart and result and isinstance(result, dict):
                self._generate_backtest_chart(args, result)

            # Return bool for success/failure
            if isinstance(result, dict):
                return True
            return result

        except KeyboardInterrupt:
            print("\n⚠️ Backtest interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Backtest execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False

    def _run_param_variations(self, args, start_date, end_date, use_historical_intelligence, base_config_override=None) -> bool:
        """
        Run parallel parameter variation testing.

        Tests multiple parameter combinations for the same epic in parallel.
        Results are ranked by the specified metric.

        Args:
            args: CLI arguments
            start_date: Start date for backtest
            end_date: End date for backtest
            use_historical_intelligence: Whether to use historical intelligence
            base_config_override: Base config overrides (e.g., scalp mode settings) to merge with each variation
        """
        from forex_scanner.core.param_variation import (
            ParameterGridGenerator,
            ParallelVariationRunner,
            VariationRunConfig
        )

        print("\n" + "=" * 70)
        print("🔬 PARAMETER VARIATION MODE")
        print("=" * 70)

        # Validate epic is specified
        if not args.epic:
            print("❌ --epic is required for parameter variation mode")
            return False

        # Parse parameter variations
        try:
            grid_gen = ParameterGridGenerator()
            param_sets = grid_gen.parse_all(
                vary_specs=getattr(args, 'vary', None),
                json_input=getattr(args, 'vary_json', None)
            )
        except ValueError as e:
            print(f"❌ Failed to parse parameter variations: {e}")
            return False

        if not param_sets:
            print("❌ No parameter combinations generated")
            return False

        # Merge base config overrides (e.g., scalp mode) with each variation's params
        # The variation params override the base config where they overlap
        if base_config_override:
            merged_param_sets = []
            for params in param_sets:
                merged = base_config_override.copy()
                merged.update(params)  # Variation params take precedence
                merged_param_sets.append(merged)
            param_sets = merged_param_sets
            print(f"📋 Base config merged: {list(base_config_override.keys())}")

        # Show summary
        total_combinations = len(param_sets)
        if total_combinations > 100:
            print(f"⚠️ Warning: {total_combinations} combinations will take a long time")
            print(f"   Consider reducing parameter ranges or using fewer parameters")

        print(f"\n📊 Configuration:")
        print(f"   Epic: {args.epic}")
        print(f"   Days: {args.days}")
        print(f"   Strategy: {args.strategy}")
        print(f"   Workers: {args.vary_workers}")
        print(f"   Total combinations: {total_combinations}")
        print(f"   Rank by: {args.rank_by}")

        # Show first few combinations
        print(f"\n   Sample combinations:")
        for i, params in enumerate(param_sets[:3]):
            print(f"   [{i+1}] {params}")
        if total_combinations > 3:
            print(f"   ... and {total_combinations - 3} more")

        print("\n" + "-" * 70)

        # Create runner config
        config = VariationRunConfig(
            epic=args.epic,
            days=args.days,
            strategy=args.strategy,
            timeframe=args.timeframe,
            max_workers=args.vary_workers,
            rank_by=args.rank_by,
            top_n=args.top_n,
            use_historical_intelligence=use_historical_intelligence,
            pipeline=args.pipeline,
            start_date=start_date,
            end_date=end_date
        )

        # Run variations
        runner = ParallelVariationRunner(config)
        results = runner.run_variations(param_sets)

        # Display results
        if results:
            print("\n" + "=" * 70)
            print(f"🔬 Parameter Variation Results - {args.epic} ({args.days} days)")
            print("=" * 70)

            # Format and print table
            table = runner.format_results(results)
            print(table)

            print("=" * 70)

            # Show best result
            completed = [r for r in results if r.status == 'completed']
            if completed:
                best = completed[0]
                print(f"\n✨ Best parameters:")
                for k, v in best.params.items():
                    print(f"   {k}: {v}")
                print(f"\n   Win rate: {best.win_rate:.1f}%")
                print(f"   Total pips: {best.total_pips:+.1f}")
                print(f"   Profit factor: {best.profit_factor:.2f}")
                print(f"   Composite score: {best.composite_score:.3f}")

            # Export if requested
            csv_export = getattr(args, 'csv_export', None)
            if csv_export:
                runner.export_results(results, csv_export)
                print(f"\n📤 Results exported to: {csv_export}")

        return True

    def _run_parallel_backtest(self, args, start_date, end_date, config_override, use_historical_intelligence) -> bool:
        """
        Run backtest in parallel across multiple currency pairs.

        Uses concurrent.futures to run backtests for multiple epics simultaneously.
        Much simpler than chunked execution - just parallelizes across pairs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import timedelta

        print("\n" + "=" * 60)
        print("⚡ PARALLEL BACKTEST MODE (Multi-Epic)")
        print("=" * 60)

        # Calculate date range
        if start_date and end_date:
            actual_start = start_date
            actual_end = end_date
        else:
            actual_end = datetime.now()
            actual_start = actual_end - timedelta(days=args.days)

        total_days = (actual_end - actual_start).days

        print(f"📅 Period: {actual_start.date()} to {actual_end.date()} ({total_days} days)")
        print(f"👷 Workers: {args.workers}")

        # Get list of epics to test
        if args.epic:
            # Single epic specified - just run normally (no parallelization benefit)
            print(f"📊 Single epic specified: {args.epic}")
            print("   (Use without --epic to parallelize across all pairs)")
            print("=" * 60 + "\n")
            # Fall back to standard execution for single epic
            return self._run_standard_backtest(args, start_date, end_date, config_override, use_historical_intelligence)

        # Get all configured epics for parallel testing
        # Check multiple possible config attribute names for the epic list
        if hasattr(config, 'EPIC_LIST') and config.EPIC_LIST:
            epic_list = config.EPIC_LIST
        elif hasattr(config, 'CURRENCY_PAIRS') and config.CURRENCY_PAIRS:
            epic_list = config.CURRENCY_PAIRS
        else:
            # Fallback: Default list of major pairs
            epic_list = [
                'CS.D.EURUSD.CEEM.IP',
                'CS.D.GBPUSD.MINI.IP',
                'CS.D.USDJPY.MINI.IP',
                'CS.D.AUDUSD.MINI.IP',
                'CS.D.USDCHF.MINI.IP',
                'CS.D.USDCAD.MINI.IP',
                'CS.D.NZDUSD.MINI.IP',
                'CS.D.EURJPY.MINI.IP',
                'CS.D.AUDJPY.MINI.IP',
            ]

        print(f"📊 Epics: {len(epic_list)} pairs")
        print(f"📈 Strategy: {args.strategy}")
        print("=" * 60 + "\n")

        # CRITICAL FIX (Jan 2026): Pre-load cache for ALL epics BEFORE spawning workers
        # This prevents race conditions where multiple workers try to load/clear the cache simultaneously
        print("🚀 Pre-loading data cache for all epics...")
        try:
            # IMPORTANT: Use same import path as BacktestDataFetcher to share the global cache
            # BacktestDataFetcher tries 'from core.memory_cache' first, so we do the same
            try:
                from core.memory_cache import get_forex_cache, initialize_cache
                from core.database import DatabaseManager
            except ImportError:
                from forex_scanner.core.memory_cache import get_forex_cache, initialize_cache
                from forex_scanner.core.database import DatabaseManager

            db = DatabaseManager(config.DATABASE_URL)
            cache = get_forex_cache(db)
            if cache is None:
                cache = initialize_cache(db, auto_load=False)

            # Load ALL epics for the full date range (not per-epic)
            cache.load_data_for_period(
                start_date=actual_start,
                end_date=actual_end,
                epics=epic_list,  # ALL epics at once
                lookback_hours=168,  # 7 days for indicator warmup
                force_reload=True,  # Force fresh load since we're starting parallel run
                use_backtest_table=True
            )
            print(f"✅ Cache pre-loaded: {cache.stats.total_rows:,} rows for {len(epic_list)} epics")
        except Exception as e:
            print(f"⚠️ Cache pre-load failed: {e}")
            import traceback
            traceback.print_exc()
            print("   Workers will load data individually (slower)")

        # Track results
        all_results = {}
        total_signals = 0
        total_pips = 0.0
        winning_trades = 0
        losing_trades = 0

        def run_single_backtest(epic: str):
            """Run backtest for a single epic and return metrics"""
            try:
                # Run backtest with return_results=True to get actual metrics
                result = self.enhanced_backtest.run_enhanced_backtest(
                    epic=epic,
                    days=args.days,
                    start_date=start_date,
                    end_date=end_date,
                    show_signals=False,  # Don't show individual signals
                    timeframe=args.timeframe,
                    strategy=args.strategy,
                    max_signals_display=0,
                    pipeline=args.pipeline,
                    csv_export=None,
                    config_override=config_override,
                    use_historical_intelligence=use_historical_intelligence,
                    return_results=True  # Get actual results dict
                )

                if result and isinstance(result, dict):
                    return epic, result, None
                elif result:
                    # Got True but no dict - shouldn't happen with return_results=True
                    return epic, {'success': True, 'total_signals': 0, 'total_pips': 0.0}, None
                else:
                    return epic, None, "Backtest failed"
            except Exception as e:
                return epic, None, str(e)

        # Run backtests in parallel
        print("🚀 Running backtests in parallel...")
        completed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_backtest, epic): epic for epic in epic_list}

            for future in as_completed(futures):
                epic, result, error = future.result()
                completed += 1
                progress = f"[{completed}/{len(epic_list)}]"

                if error:
                    print(f"  ❌ {progress} {epic.split('.')[2]}: {error}")
                    all_results[epic] = {'error': error}
                elif result:
                    # Extract metrics from nested structure
                    # Structure:
                    #   - signal_processing: {logged, validated, rejected, errors}
                    #   - backtest_results: {performance_summary: {...}, execution_stats: {...}, signal_summary: {...}}
                    #   - performance_metrics: {total_pips, win_rate}
                    #   - order_logger: BacktestOrderLogger instance with actual trade data
                    signal_proc = result.get('signal_processing', {})
                    backtest_res = result.get('backtest_results', {})
                    exec_stats = backtest_res.get('execution_stats', {})
                    perf_summary = backtest_res.get('performance_summary', {})
                    perf_metrics = result.get('performance_metrics', {})

                    # Signal count from signal_processing (most reliable)
                    signals = (
                        signal_proc.get('logged', 0) or
                        signal_proc.get('validated', 0) or
                        exec_stats.get('total_signals_detected', 0) or
                        perf_summary.get('total_signals', 0) or 0
                    )

                    # Try to get actual trade outcomes from order_logger
                    wins = 0
                    losses = 0
                    breakevens = 0
                    pips = 0.0

                    order_logger = result.get('order_logger')
                    if order_logger and hasattr(order_logger, 'signals'):
                        # Calculate directly from logged signals
                        logged_signals = order_logger.signals
                        for sig in logged_signals:
                            trade_result = sig.get('trade_result', '')
                            pips_gained = sig.get('pips_gained', 0) or 0

                            if trade_result == 'win':
                                wins += 1
                                pips += float(pips_gained)
                            elif trade_result == 'loss':
                                losses += 1
                                pips += float(pips_gained)  # pips_gained is negative for losses
                            elif trade_result == 'breakeven':
                                breakevens += 1

                        signals = len(logged_signals) if logged_signals else signals
                    else:
                        # Fallback: use performance_summary or metrics
                        pips = float(
                            perf_summary.get('total_pips', 0) or
                            perf_metrics.get('total_pips', 0) or 0
                        )

                        win_rate_pct = float(
                            perf_summary.get('avg_win_rate', 0) or
                            perf_metrics.get('win_rate', 0) or 0
                        )
                        # Normalize to 0-1 if it's a percentage > 1
                        if win_rate_pct > 1:
                            win_rate_pct = win_rate_pct / 100

                        # Estimate wins/losses from signals and win rate
                        wins = int(signals * win_rate_pct) if signals > 0 else 0
                        losses = signals - wins

                    total_signals += signals
                    total_pips += pips
                    winning_trades += wins
                    losing_trades += losses

                    # Store flattened result for chart generation
                    all_results[epic] = {
                        'total_signals': signals,
                        'total_pips': pips,
                        'winning_trades': wins,
                        'losing_trades': losses,
                        'raw_result': result
                    }
                    # Show win rate in progress line
                    if wins + losses > 0:
                        wr = wins / (wins + losses) * 100
                        print(f"  ✅ {progress} {epic.split('.')[2]}: {signals} signals, {pips:+.1f} pips, {wr:.0f}% win rate")
                    else:
                        print(f"  ✅ {progress} {epic.split('.')[2]}: {signals} signals, {pips:+.1f} pips")
                else:
                    print(f"  ⚠️ {progress} {epic.split('.')[2]}: No result")
                    all_results[epic] = {}

        # Calculate aggregate statistics
        win_rate = winning_trades / max(winning_trades + losing_trades, 1)

        # Display summary
        print("\n" + "=" * 60)
        print("📊 PARALLEL BACKTEST RESULTS")
        print("=" * 60)
        print(f"Pairs Tested:        {len(epic_list)}")
        print(f"Total Signals:       {total_signals}")
        print(f"Win Rate:            {win_rate:.1%}")
        print(f"Total Pips:          {total_pips:+.1f}")
        print(f"Winning Trades:      {winning_trades}")
        print(f"Losing Trades:       {losing_trades}")
        print("=" * 60)

        # Show per-pair breakdown
        print("\n📈 Per-Pair Results:")
        for epic, result in sorted(all_results.items()):
            pair_name = epic.split('.')[2]
            if 'error' in result:
                print(f"  {pair_name}: ❌ Error - {result.get('error', 'unknown')}")
            else:
                signals = result.get('total_signals', 0)
                pips = result.get('total_pips', 0.0)
                wins = result.get('winning_trades', 0)
                losses = result.get('losing_trades', 0)
                if wins + losses > 0:
                    pair_win_rate = wins / (wins + losses)
                    print(f"  {pair_name}: {signals} signals, {pips:+.1f} pips, {pair_win_rate:.0%} win rate")
                else:
                    print(f"  {pair_name}: {signals} signals, {pips:+.1f} pips")

        # Generate chart if requested (using combined results)
        if args.chart:
            self._generate_parallel_chart(args, all_results, actual_start, actual_end)

        return True

    def _run_standard_backtest(self, args, start_date, end_date, config_override, use_historical_intelligence) -> bool:
        """Run a standard (non-parallel) backtest"""
        result = self.enhanced_backtest.run_enhanced_backtest(
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

        if result and args.chart:
            self._generate_backtest_chart(args, result)

        return result is not None

    def _generate_parallel_chart(self, args, all_results, start_date, end_date):
        """Generate summary chart for parallel backtest results"""
        try:
            from forex_scanner.core.chunked_backtest import BacktestChartGenerator
            from forex_scanner.core.database import DatabaseManager
            from forex_scanner import config as scanner_config
        except ImportError:
            from core.chunked_backtest import BacktestChartGenerator
            from core.database import DatabaseManager
            import config as scanner_config

        print("\n📊 Generating summary chart...")

        # Collect all signals from all results
        all_signals = []
        for epic, result in all_results.items():
            if 'error' not in result and 'signals' in result:
                for sig in result['signals']:
                    sig['epic'] = epic
                    all_signals.append(sig)

        if not all_signals:
            print("⚠️ No signals to chart")
            return

        db_manager = DatabaseManager(scanner_config.DATABASE_URL)
        chart_generator = BacktestChartGenerator(db_manager=db_manager)

        # Determine output path
        chart_path = args.chart_output
        if not chart_path:
            chart_path = f"/tmp/backtest_parallel_{start_date.date()}_{end_date.date()}.png"

        # For now, just show message - full multi-epic chart would need more work
        print(f"📊 Found {len(all_signals)} total signals across all pairs")
        print(f"   (Multi-epic chart generation not yet implemented)")
        print(f"   Use --epic PAIR --chart for single-pair charts")

    def _generate_backtest_chart(self, args, result) -> bool:
        """
        Generate chart for a standard (non-parallel) backtest.

        This is called after a standard backtest completes if --chart is specified.
        """
        try:
            from forex_scanner.core.chunked_backtest import BacktestChartGenerator
            from forex_scanner.core.database import DatabaseManager
            from forex_scanner import config as scanner_config
        except ImportError:
            from core.chunked_backtest import BacktestChartGenerator
            from core.database import DatabaseManager
            import config as scanner_config

        print("\n📊 Generating backtest chart...")

        db_manager = DatabaseManager(scanner_config.DATABASE_URL)

        # Get execution_id from result
        execution_id = None
        if isinstance(result, dict):
            execution_id = result.get('execution_id')
            if not execution_id and 'backtest_results' in result:
                execution_id = result['backtest_results'].get('execution_id')

        if not execution_id:
            print("⚠️ No execution_id found in result - cannot fetch signals")
            return False

        # Convert to plain Python int (numpy.int64 causes psycopg2 issues)
        execution_id = int(execution_id)

        # Fetch signals from database
        query = """
        SELECT
            signal_timestamp as timestamp,
            signal_type as type,
            entry_price,
            exit_price,
            pips_gained as pips,
            trade_result
        FROM backtest_signals
        WHERE execution_id = :exec_id
        ORDER BY signal_timestamp
        """

        try:
            signals_df = db_manager.execute_query(query, {'exec_id': execution_id})
        except Exception as e:
            print(f"⚠️ Failed to fetch signals from database: {e}")
            return False

        if signals_df.empty:
            print("⚠️ No signals found in database for this backtest")
            return False

        # Convert DataFrame to list of dicts for chart generator
        chart_signals = []
        for _, row in signals_df.iterrows():
            pips = float(row['pips']) if row['pips'] else 0
            chart_signals.append({
                'timestamp': row['timestamp'],
                'type': row['type'],
                'entry_price': float(row['entry_price']) if row['entry_price'] else 0,
                'pips': pips,
                'result': 'win' if pips > 0 else 'loss'
            })

        print(f"   Found {len(chart_signals)} signals to plot")
        chart_generator = BacktestChartGenerator(db_manager=db_manager)

        # Get epic from args or result
        epic = args.epic or result.get('epic', 'CS.D.EURUSD.CEEM.IP')

        # Get date range from result or calculate from signals
        start_date = result.get('start_date')
        end_date = result.get('end_date')

        if not start_date and chart_signals:
            timestamps = [s['timestamp'] for s in chart_signals if s['timestamp']]
            if timestamps:
                start_date = min(timestamps)
                end_date = max(timestamps)

        if not start_date or not end_date:
            print("⚠️ Could not determine date range for chart")
            return False

        # Try MinIO upload first (preferred), fall back to disk if specified
        if args.chart_output:
            # User specified output path - save to disk
            result_path = chart_generator.generate_backtest_chart(
                epic=epic,
                start_date=start_date,
                end_date=end_date,
                signals=chart_signals,
                strategy=args.strategy,
                output_path=args.chart_output
            )
            if result_path:
                print(f"✅ Chart saved to: {result_path}")
                return True
            else:
                print("⚠️ Chart generation failed")
                return False
        else:
            # Upload to MinIO
            minio_result = chart_generator.generate_and_upload_chart(
                epic=epic,
                start_date=start_date,
                end_date=end_date,
                signals=chart_signals,
                execution_id=execution_id,
                strategy=args.strategy,
                timeframe=args.timeframe or '15m'
            )

            if minio_result:
                # Save URL to database
                try:
                    update_query = """
                    UPDATE backtest_executions
                    SET chart_url = :chart_url,
                        chart_object_name = :object_name
                    WHERE id = :exec_id
                    """
                    db_manager.execute_query(update_query, {
                        'chart_url': minio_result['url'],
                        'object_name': minio_result['object_name'],
                        'exec_id': execution_id
                    })
                    print(f"✅ Chart uploaded to MinIO: {minio_result['object_name']}")
                    print(f"   URL: {minio_result['url']}")
                    return True
                except Exception as e:
                    print(f"⚠️ Chart uploaded but failed to save URL to database: {e}")
                    return True
            else:
                # MinIO not available - fall back to temp file
                temp_path = f"/tmp/backtest_{epic.replace('.', '_')}_chart.png"
                result_path = chart_generator.generate_backtest_chart(
                    epic=epic,
                    start_date=start_date,
                    end_date=end_date,
                    signals=chart_signals,
                    strategy=args.strategy,
                    output_path=temp_path
                )
                if result_path:
                    print(f"✅ Chart saved to: {result_path} (MinIO unavailable)")
                    return True
                else:
                    print("⚠️ Chart generation failed")
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