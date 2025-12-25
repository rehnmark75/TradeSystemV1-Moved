#!/usr/bin/env python3
"""
bt_stocks.py - Quick Stock Backtest Wrapper

Simple wrapper for the stock backtest CLI with sensible defaults.
Designed for quick and easy backtesting.

Usage:
    python bt_stocks.py                                    # 90 days, all stocks
    python bt_stocks.py AAPL                               # 90 days, AAPL only
    python bt_stocks.py AAPL 30                            # 30 days, AAPL only
    python bt_stocks.py AAPL 30 --show-signals             # With detailed signals
    python bt_stocks.py --all 90                           # All stocks, 90 days
    python bt_stocks.py --all 90 --sector Technology       # Tech stocks only
"""

import sys
import subprocess
import os


def main():
    """Main wrapper function."""

    # Base command
    base_cmd = ["python3", "-m", "stock_scanner.backtest_cli"]

    args = sys.argv[1:]

    # If no arguments, show help
    if not args or args[0] in ["-h", "--help", "help"]:
        show_help()
        return 0

    # Parse simplified arguments
    processed_args = []
    i = 0
    ticker_specified = False
    days_specified = False

    # Common ticker shortcuts
    TICKER_SHORTCUTS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "BAC", "GS", "V", "MA",
        "JNJ", "UNH", "PFE",
        "WMT", "KO", "PG", "MCD",
        "XOM", "CVX", "BA", "CAT",
        "NFLX", "AMD", "INTC", "CRM", "ORCL", "ADBE",
        "DIS", "NKE", "SBUX", "HD", "LOW"
    ]

    # Strategy shortcuts
    STRATEGY_MAP = {
        "EMA": "EMA_PULLBACK",
        "EMA_PULLBACK": "EMA_PULLBACK",
        "PULLBACK": "EMA_PULLBACK",
        "TREND": "EMA_PULLBACK",
    }

    while i < len(args):
        arg = args[i]

        # Handle ticker (uppercase letters)
        if arg.upper() in TICKER_SHORTCUTS or (arg.upper().isalpha() and len(arg) <= 5):
            processed_args.extend(["--ticker", arg.upper()])
            print(f"📊 Testing {arg.upper()}")
            ticker_specified = True

        # Handle --all flag
        elif arg == "--all":
            processed_args.append("--all")
            print("📊 Testing all tradeable stocks")

        # Handle strategy shortcuts
        elif arg.upper() in STRATEGY_MAP:
            strategy_name = STRATEGY_MAP[arg.upper()]
            processed_args.extend(["--strategy", strategy_name])
            print(f"📈 Using {arg.upper()} strategy ({strategy_name})")

        # Handle days as a number
        elif arg.isdigit():
            processed_args.extend(["--days", arg])
            print(f"📅 Testing {arg} days")
            days_specified = True

        # Handle sector filter
        elif arg == "--sector":
            if i + 1 < len(args):
                i += 1
                processed_args.extend(["--sector", args[i]])
                print(f"🏢 Filtering by sector: {args[i]}")

        # Pass through other flags
        elif arg.startswith("--"):
            processed_args.append(arg)

            # If it's a flag that takes a value, include the next argument
            if arg in ["--strategy", "--timeframe", "--csv-export", "--start-date", "--end-date", "--export-execution"] and i + 1 < len(args):
                i += 1
                processed_args.append(args[i])
        else:
            # Unknown argument, pass through
            processed_args.append(arg)

        i += 1

    # Add defaults
    if not days_specified and "--days" not in processed_args:
        processed_args.extend(["--days", "90"])

    # If no ticker or --all specified, default to --all
    if not ticker_specified and "--all" not in processed_args and "--ticker" not in processed_args and "--export-execution" not in processed_args and "--compare" not in processed_args:
        processed_args.append("--all")
        print("📊 Testing all tradeable stocks (default)")

    base_cmd.extend(processed_args)

    # Execute the backtest CLI
    try:
        print("🚀 Starting stock backtest...")
        print(f"Command: {' '.join(base_cmd)}")
        print("=" * 70)

        # Change to the worker/app directory
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        result = subprocess.run(base_cmd, cwd=app_dir)
        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return 1


def show_help():
    """Show help message."""
    help_text = """
🧪 bt_stocks.py - Quick Stock Backtest Wrapper

Simple interface for running stock backtests with sensible defaults.

Usage:
  python bt_stocks.py [TICKER] [DAYS] [OPTIONS]

Basic Usage:
  python bt_stocks.py                           # 90 days, all stocks
  python bt_stocks.py AAPL                      # 90 days, AAPL only
  python bt_stocks.py AAPL 30                   # 30 days, AAPL
  python bt_stocks.py AAPL 30 --show-signals    # With detailed signals
  python bt_stocks.py --all 90                  # All stocks, 90 days

Sector Filtering:
  python bt_stocks.py --all 90 --sector Technology
  python bt_stocks.py --all 90 --sector "Financial,Healthcare"

Strategy Comparison:
  python bt_stocks.py --all 90 --compare EMA_PULLBACK,TREND_MOMENTUM

Export Results:
  python bt_stocks.py AAPL 90 --csv-export /tmp/results.csv
  python bt_stocks.py --export-execution 42 --csv-export /tmp/results.csv

Common Tickers:
  Tech:       AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
  Financial:  JPM, BAC, GS, V, MA
  Healthcare: JNJ, UNH, PFE
  Consumer:   WMT, KO, PG, MCD
  Energy:     XOM, CVX
  Industrial: BA, CAT

Available Strategies:
  EMA_PULLBACK (default) - EMA trend pullback entries

Options:
  --show-signals      Show detailed signal breakdown
  --all               Test all tradeable stocks
  --sector SECTOR     Filter by sector(s)
  --strategy NAME     Use specific strategy
  --timeframe 1d      Candle timeframe (1d, 4h, 1h)
  --csv-export PATH   Export results to CSV
  --compare A,B       Compare multiple strategies
  --verbose           Verbose output

Examples:
  python bt_stocks.py NVDA 60 --show-signals
  python bt_stocks.py --all 90 --sector Technology
  python bt_stocks.py --all 30 --compare EMA_PULLBACK,TREND
"""
    print(help_text)


if __name__ == "__main__":
    sys.exit(main())
