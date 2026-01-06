#!/usr/bin/env python3
"""
bt.py - Quick Backtest Wrapper

Simple wrapper for the enhanced backtest CLI with sensible defaults.
Designed for quick and easy backtesting.

Usage:
    python bt.py                                    # 7 days, all epics
    python bt.py EURUSD                            # 7 days, EUR/USD only
    python bt.py EURUSD 14                         # 14 days, EUR/USD only
    python bt.py EURUSD 7 --show-signals          # With detailed signals
    python bt.py --all 3 --show-signals           # All pairs, 3 days, with signals
"""

import sys
import subprocess
import os

def main():
    """Main wrapper function"""

    # Base command
    base_cmd = ["python3", "/app/forex_scanner/backtest_cli.py"]

    args = sys.argv[1:]

    # If no arguments, default to 7 days all pairs
    if not args:
        base_cmd.extend(["--days", "7"])
        print("🧪 Running default backtest: 7 days, all currency pairs")
    else:
        # Parse simplified arguments
        processed_args = []
        i = 0
        epic_specified = False

        while i < len(args):
            arg = args[i]

            # Handle epic shortcuts (e.g., "EURUSD" -> "CS.D.EURUSD.CEEM.IP", others -> "CS.D.PAIR.MINI.IP")
            if arg in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD", "EURJPY", "AUDJPY", "GBPJPY"]:
                # EURUSD uses CEEM, all others use MINI
                if arg == "EURUSD":
                    processed_args.extend(["--epic", f"CS.D.{arg}.CEEM.IP"])
                else:
                    processed_args.extend(["--epic", f"CS.D.{arg}.MINI.IP"])
                print(f"📊 Testing {arg}")
                epic_specified = True

            # Handle --all flag
            elif arg == "--all":
                print("📊 Testing all currency pairs")
                # Don't add --epic, let it default to all

            # Handle strategy shortcuts
            # NOTE: After January 2026 cleanup, only SMC_SIMPLE is active
            # Other strategies have been archived to forex_scanner/archive/disabled_strategies/
            elif arg.upper() in ["SMC", "SMC_SIMPLE", "SMC_EMA"]:
                strategy_mapping = {
                    "SMC": "SMC_SIMPLE",
                    "SMC_SIMPLE": "SMC_SIMPLE",
                    "SMC_EMA": "SMC_SIMPLE",
                }
                strategy_name = strategy_mapping[arg.upper()]
                processed_args.extend(["--strategy", strategy_name])
                print(f"📈 Using {arg.upper()} strategy ({strategy_name})")

            # Handle days as a number
            elif arg.isdigit():
                processed_args.extend(["--days", arg])
                print(f"📅 Testing {arg} days")

            # Handle --override flags (can be used multiple times)
            elif arg == "--override" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🧪 Override: {args[i]}")

            # Handle --snapshot flag for loading saved parameter configs
            elif arg == "--snapshot" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📦 Using snapshot: {args[i]}")

            # Handle --save-snapshot flag for saving test results
            elif arg == "--save-snapshot" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"💾 Will save to snapshot: {args[i]}")

            # Handle historical intelligence flags
            elif arg == "--no-historical-intelligence":
                processed_args.append(arg)
                print(f"📚 Historical intelligence: DISABLED")

            elif arg == "--use-historical-intelligence":
                processed_args.append(arg)
                print(f"📚 Historical intelligence: ENABLED")

            # Pass through other flags
            elif arg.startswith("--"):
                processed_args.append(arg)

                # If it's a flag that takes a value, include the next argument
                if arg in ["--strategy", "--timeframe", "--max-signals", "--hours", "--csv-export"] and i + 1 < len(args):
                    i += 1
                    processed_args.append(args[i])
            else:
                # Unknown argument, pass through
                processed_args.append(arg)

            i += 1

        # If no explicit days were specified and no epic, add default days
        if not any(processed_args[j:j+2] == ["--days", processed_args[j+1]] for j in range(len(processed_args)-1) if processed_args[j] == "--days"):
            # Check if we need to add default days
            pass  # Let CLI handle defaults

        base_cmd.extend(processed_args)

    # Execute the backtest CLI
    try:
        print("🚀 Starting enhanced backtest...")
        print(f"Command: {' '.join(base_cmd)}")
        print("=" * 50)

        result = subprocess.run(base_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return 1

def show_help():
    """Show help message"""
    help_text = """
🧪 bt.py - Quick Backtest Wrapper

Simple interface for running backtests with sensible defaults.

Usage:
  python bt.py [PAIR] [DAYS] [OPTIONS]

Basic Usage:
  python bt.py                           # 7 days, all pairs, EMA strategy (raw signals)
  python bt.py EURUSD                   # 7 days, EUR/USD only, EMA strategy (raw signals)
  python bt.py EURUSD 14                # 14 days, EUR/USD, EMA strategy (raw signals)
  python bt.py EURUSD 7 --show-signals  # With detailed signals (raw signals)
  python bt.py EURUSD 7 --pipeline      # Full pipeline with trade validator (realistic)

Signal Modes:
  Default (raw): Tests strategy signal generation without filters
  --pipeline:    Tests full signal pipeline with trade validator (matches live trading)

Strategy Shortcuts:
  python bt.py EURUSD 7 EMA --show-signals         # EMA Strategy
  python bt.py EURUSD 7 MACD --show-signals        # MACD Strategy
  python bt.py EURUSD 7 BB --show-signals          # Bollinger + Supertrend
  python bt.py EURUSD 7 SMC --show-signals         # Smart Money Concepts (Fast)
  python bt.py EURUSD 7 SMC_STRUCTURE --show-signals  # SMC Pure Structure (Price Action)
  python bt.py EURUSD 7 MOMENTUM --show-signals    # Momentum Strategy

Supported Pairs:
  EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, EURJPY, AUDJPY, GBPJPY

Supported Strategies:
  EMA, MACD, BB, SMC, SMC_STRUCTURE, MOMENTUM, ICHIMOKU, KAMA, ZEROLAG, MEANREV, RANGING, SCALPING, VP (VOLUME_PROFILE)

Additional Options:
  --show-signals     Show detailed signal breakdown
  --pipeline         Use full signal pipeline with trade validator (realistic live simulation)
  --all             Test all pairs (default if no pair specified)
  --strategy NAME   Use specific strategy (full name)
  --timeframe 5m    Use different timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
  --verbose         Verbose output

Parameter Overrides (for backtest testing only - does NOT affect live trading):
  --override PARAM=VALUE   Override strategy parameter (can be used multiple times)
                          Types auto-detected: bool (true/false), float (1.5), int (10), string

  Supported Override Parameters:
    SL/TP: fixed_stop_loss_pips, fixed_take_profit_pips, min_rr_ratio, sl_buffer_pips
    Confidence: min_confidence, max_confidence, high_volume_confidence
    Fibonacci: fib_min, fib_max, fib_pullback_min, fib_pullback_max
    HTF Settings: ema_period, ema_buffer_pips, min_distance_from_ema_pips
    Filters: macd_filter_enabled, allow_asian_session, cooldown_minutes
    See plan file for full list of 40+ overridable parameters

Examples:
  python bt.py GBPUSD 14 EMA --show-signals       # GBP/USD, 14 days, EMA strategy with signals (raw)
  python bt.py EURUSD 7 MACD --pipeline           # EUR/USD, 7 days, MACD with full pipeline
  python bt.py --all 7 MACD --show-signals        # All pairs, 7 days, MACD strategy with signals (raw)
  python bt.py EURUSD 3 SMC --timeframe 5m        # EUR/USD, 3 days, Smart Money on 5m timeframe
  python bt.py AUDUSD 7 MOMENTUM --pipeline        # AUD/USD, 7 days, Momentum with pipeline validation
  python bt.py USDJPY 14 ZEROLAG --show-signals   # USD/JPY, 14 days, Zero Lag strategy

Parameter Override Examples:
  python bt.py EURUSD 14 --override fixed_stop_loss_pips=10 --override min_confidence=0.55
  python bt.py GBPUSD 30 --override fib_min=0.5 --override fib_max=0.7
  python bt.py EURUSD 7 --override macd_filter_enabled=false --show-signals

Config Snapshots (persistent parameter sets):
  # Create a snapshot with specific parameters
  python snapshot_cli.py create tight_sl --set fixed_stop_loss_pips=8 --set min_confidence=0.6

  # Run backtest using a saved snapshot
  python bt.py EURUSD 14 --snapshot tight_sl

  # Run with snapshot + additional overrides (overrides take precedence)
  python bt.py EURUSD 14 --snapshot tight_sl --override min_confidence=0.65

  # List available snapshots
  python snapshot_cli.py list
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        sys.exit(0)

    sys.exit(main())