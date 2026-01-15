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

Parallel Execution:
    python bt.py EURUSD 30 --parallel              # Run in parallel with 4 workers
    python bt.py EURUSD 30 --parallel --workers 8  # Run with 8 workers
    python bt.py EURUSD 30 --parallel --chunk-days 7  # 7-day chunks

Chart Generation:
    python bt.py EURUSD 14 --chart                 # Generate visual chart
    python bt.py EURUSD 14 --chart --chart-output /tmp/chart.png
"""

import sys
import subprocess
import os

def main():
    """Main wrapper function"""

    # Base command - use -u for unbuffered output for real-time progress tracking
    base_cmd = ["python3", "-u", "/app/forex_scanner/backtest_cli.py"]

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

            # Handle scalp mode flag (VSL emulation)
            elif arg == "--scalp":
                processed_args.append(arg)
                print(f"🎯 Scalp mode: ENABLED (Virtual Stop Loss emulation)")

            # Handle scalp offset override
            elif arg == "--scalp-offset" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📍 Scalp offset: {args[i]} pips")

            # Handle scalp expiry override
            elif arg == "--scalp-expiry" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"⏱️ Scalp expiry: {args[i]} minutes")

            # Handle scalp tier settings
            elif arg == "--scalp-htf" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📊 Scalp HTF: {args[i]}")

            elif arg == "--scalp-ema" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📈 Scalp EMA: {args[i]}")

            elif arg == "--scalp-swing-lookback" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔍 Scalp Swing Lookback: {args[i]} bars")

            elif arg == "--scalp-trigger-tf" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔄 Scalp Trigger TF: {args[i]}")

            elif arg == "--scalp-entry-tf" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🎯 Scalp Entry TF: {args[i]}")

            elif arg == "--scalp-confidence" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📊 Scalp Confidence: {args[i]}")

            elif arg == "--scalp-cooldown" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"⏱️ Scalp Cooldown: {args[i]} min")

            elif arg == "--scalp-tolerance" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📏 Scalp Tolerance: {args[i]} pips")

            # Handle scalp signal qualification flags
            elif arg == "--qualification-monitor":
                processed_args.append(arg)
                print(f"📊 Signal Qualification: MONITORING mode (logs only)")

            elif arg == "--qualification-active":
                processed_args.append(arg)
                print(f"🔒 Signal Qualification: ACTIVE mode (blocks low-score signals)")

            elif arg == "--qual-min-score" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📊 Qualification Min Score: {args[i]}")

            elif arg == "--qual-rsi-only":
                processed_args.append(arg)
                print(f"📈 Qualification: RSI filter only")

            elif arg == "--qual-two-pole-only":
                processed_args.append(arg)
                print(f"📈 Qualification: Two-Pole filter only")

            elif arg == "--qual-macd-only":
                processed_args.append(arg)
                print(f"📈 Qualification: MACD filter only")

            # Handle parallel execution flags
            elif arg == "--parallel":
                processed_args.append(arg)
                print(f"⚡ Parallel execution: ENABLED")

            elif arg == "--workers" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"👷 Workers: {args[i]}")

            elif arg == "--chunk-days" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"📦 Chunk size: {args[i]} days")

            # Handle chart generation flags
            elif arg == "--chart":
                processed_args.append(arg)
                print(f"📊 Chart generation: ENABLED")

            elif arg == "--chart-output" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"💾 Chart output: {args[i]}")

            # Handle parameter variation flags
            elif arg == "--vary" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔬 Vary parameter: {args[i]}")

            elif arg == "--vary-json" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔬 Vary JSON: {args[i][:50]}...")

            elif arg == "--vary-workers" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔬 Variation workers: {args[i]}")

            elif arg == "--rank-by" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔬 Rank by: {args[i]}")

            elif arg == "--top-n" and i + 1 < len(args):
                processed_args.append(arg)
                i += 1
                processed_args.append(args[i])
                print(f"🔬 Show top: {args[i]}")

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
  --scalp            Enable scalping mode with Virtual Stop Loss (VSL) emulation
                     Uses 3 pip VSL for majors, 4 pip VSL for JPY pairs, 5 pip TP
  --all             Test all pairs (default if no pair specified)
  --strategy NAME   Use specific strategy (full name)
  --timeframe 5m    Use different timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
  --verbose         Verbose output

Parallel Execution (for faster long-period backtests):
  --parallel         Enable parallel execution (splits into chunks)
  --workers N        Number of parallel workers (default: 4)
  --chunk-days N     Days per chunk (default: 7)

Chart Generation:
  --chart            Generate visual chart with signals plotted
  --chart-output     Save chart to specific file path

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

Parallel Execution Examples:
  python bt.py EURUSD 30 --parallel                    # 30 days with 4 workers (default)
  python bt.py EURUSD 60 --parallel --workers 8        # 60 days with 8 workers
  python bt.py GBPUSD 30 --parallel --chunk-days 10    # 10-day chunks

Chart Generation Examples:
  python bt.py EURUSD 14 --chart                       # Generate chart displayed in terminal
  python bt.py EURUSD 14 --chart --chart-output /tmp/eurusd_backtest.png
  python bt.py EURUSD 30 --parallel --chart            # Parallel backtest with chart

Scalping Mode Examples (VSL Emulation):
  python bt.py EURUSD 7 --scalp                        # Scalp EURUSD with 3 pip VSL, 5 pip TP
  python bt.py USDJPY 7 --scalp                        # Scalp USDJPY with 4 pip VSL (JPY pair)
  python bt.py EURUSD 7 --scalp --show-signals         # Show detailed scalp signals
  python bt.py EURUSD 7 --scalp --override scalp_tp_pips=7  # Override TP to 7 pips
  python bt.py EURUSD 7 --scalp --scalp-offset 2       # Override order offset to 2 pips
  python bt.py EURUSD 7 --scalp --scalp-expiry 10      # Override order expiry to 10 minutes
  python bt.py EURUSD 7 --scalp --scalp-offset 2 --scalp-expiry 15  # Combined offset + expiry

Scalp Tier Settings (per-tier tuning):
  --scalp-htf 15m           TIER 1: HTF timeframe for bias (5m/15m/30m/1h)
  --scalp-ema 30            TIER 1: EMA period (default 20, common: 10, 20, 30, 50)
  --scalp-trigger-tf 5m     TIER 2: Trigger timeframe (1m/5m/15m)
  --scalp-swing-lookback 8  TIER 2: Swing lookback bars (default 12, range 5-30)
  --scalp-entry-tf 1m       TIER 3: Entry timeframe (1m/5m)
  --scalp-confidence 0.35   Minimum confidence threshold (default 0.30)
  --scalp-cooldown 10       Cooldown between trades in minutes (default 15)
  --scalp-tolerance 0.8     Swing break tolerance in pips (default 0.5)

Scalp Tier Tuning Examples:
  python bt.py EURUSD 7 --scalp --scalp-ema 30 --scalp-swing-lookback 8  # Optimized EURUSD settings
  python bt.py GBPUSD 7 --scalp --scalp-htf 15m --scalp-trigger-tf 5m    # Custom timeframes
  python bt.py USDJPY 7 --scalp --scalp-confidence 0.40 --scalp-cooldown 20  # Higher confidence
  python bt.py EURUSD 7 --scalp --scalp-tolerance 1.0  # More tolerant swing break detection

Scalp Signal Qualification (v2.21.0):
  --qualification-monitor    Enable qualification in MONITORING mode (logs results, passes all signals)
  --qualification-active     Enable qualification in ACTIVE mode (blocks signals below threshold)
  --qual-min-score 0.66      Minimum qualification score required (default 0.50 = 50% of filters)
  --qual-rsi-only            Use only RSI momentum filter
  --qual-two-pole-only       Use only Two-Pole oscillator filter
  --qual-macd-only           Use only MACD direction filter

Qualification Examples:
  python bt.py EURUSD 7 --scalp --qualification-monitor      # Log qualification results (recommended first)
  python bt.py EURUSD 7 --scalp --qualification-active       # Block low-score signals
  python bt.py EURUSD 7 --scalp --qualification-active --qual-min-score 0.66  # Require 2/3 filters to pass
  python bt.py EURUSD 7 --scalp --qualification-monitor --qual-rsi-only  # Test RSI filter alone
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
        sys.exit(0)

    sys.exit(main())