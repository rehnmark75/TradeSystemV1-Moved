#!/usr/bin/env python3
"""
Stock Backtest CLI

Main CLI interface for running stock backtests.

Usage:
    python backtest_cli.py --ticker AAPL --days 90
    python backtest_cli.py --all --days 90 --sector Technology
    python backtest_cli.py --ticker AAPL --days 30 --show-signals
    python backtest_cli.py --compare EMA_PULLBACK,TREND_MOMENTUM --days 90
    python backtest_cli.py --export-execution 42 --csv-export /tmp/results.csv
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, date, timedelta
from typing import List, Optional

from .core.database.async_database_manager import AsyncDatabaseManager
from .core.backtest.backtest_orchestrator import StockBacktestOrchestrator, STRATEGY_REGISTRY
from . import config


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress noisy loggers
    logging.getLogger('asyncpg').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Stock Backtest CLI - Test trading strategies on historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_cli.py --ticker AAPL --days 90
  python backtest_cli.py --all --days 90
  python backtest_cli.py --all --days 90 --sector Technology
  python backtest_cli.py --ticker AAPL --days 30 --show-signals
  python backtest_cli.py --compare EMA_PULLBACK,TREND_MOMENTUM --days 90
  python backtest_cli.py --export-execution 42 --csv-export results.csv
        """
    )

    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        '--ticker', '-t',
        type=str,
        help='Single ticker to backtest (e.g., AAPL)'
    )
    target_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Backtest all tradeable stocks'
    )
    target_group.add_argument(
        '--export-execution',
        type=int,
        metavar='ID',
        help='Export results from a previous execution by ID'
    )

    # Date range
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=90,
        help='Number of days to backtest (default: 90)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )

    # Strategy
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='EMA_PULLBACK',
        help=f'Strategy name (default: EMA_PULLBACK). Available: {", ".join(STRATEGY_REGISTRY.keys())}'
    )

    # Comparison mode
    parser.add_argument(
        '--compare',
        type=str,
        metavar='STRATEGIES',
        help='Compare multiple strategies (comma-separated)'
    )

    # Filters
    parser.add_argument(
        '--sector',
        type=str,
        help='Filter by sector(s), comma-separated (e.g., "Technology,Healthcare")'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1d',
        choices=['1d', '4h', '1h'],
        help='Timeframe (default: 1d)'
    )

    # Output options
    parser.add_argument(
        '--show-signals',
        action='store_true',
        help='Display detailed signal list'
    )
    parser.add_argument(
        '--csv-export',
        type=str,
        metavar='PATH',
        help='Export results to CSV file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser


async def run_backtest(args) -> int:
    """Run the backtest with given arguments."""
    db = None
    try:
        # Initialize database
        db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
        await db.connect()

        # Handle export-execution mode
        if args.export_execution:
            return await handle_export_execution(db, args)

        # Parse date range
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)

        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

        # Determine tickers
        tickers = None
        if args.ticker:
            tickers = [args.ticker.upper()]
        # If --all, tickers remain None (will fetch all tradeable)

        # Handle comparison mode
        if args.compare:
            return await handle_comparison(
                db, args, tickers, start_date, end_date
            )

        # Create orchestrator
        orchestrator = StockBacktestOrchestrator(
            db_manager=db,
            strategy_name=args.strategy,
            timeframe=args.timeframe
        )

        # Print header
        print_header(args, start_date, end_date, tickers)

        # Run backtest
        execution_id = await orchestrator.run(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            sector=args.sector,
            show_progress=True
        )

        # Display results
        await display_results(db, execution_id, args.show_signals)

        # Export if requested
        if args.csv_export:
            await orchestrator.export_results(execution_id, args.csv_export)
            print(f"\nResults exported to: {args.csv_export}")

        return 0

    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        if db:
            await db.close()


async def handle_comparison(db, args, tickers, start_date, end_date) -> int:
    """Handle strategy comparison mode."""
    strategies = [s.strip().upper() for s in args.compare.split(',')]

    print("=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Sector: {args.sector or 'All'}")
    print("=" * 70)

    orchestrator = StockBacktestOrchestrator(db_manager=db)

    execution_ids = await orchestrator.compare_strategies(
        strategies=strategies,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        sector=args.sector
    )

    # Display comparison table
    results_df = await orchestrator.get_comparison_results(execution_ids)

    print("\nCOMPARISON RESULTS")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Signals':>8} {'Trades':>8} {'Win%':>8} {'P&L%':>10} {'PF':>8}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        pf = f"{row['profit_factor']:.2f}" if row['profit_factor'] else 'N/A'
        print(
            f"{row['strategy']:<20} {row['signals']:>8} {row['trades']:>8} "
            f"{row['win_rate']:>7.1f}% {row['total_pnl']:>9.2f}% {pf:>8}"
        )

    print("-" * 70)

    return 0


async def handle_export_execution(db, args) -> int:
    """Handle export of previous execution."""
    from .core.backtest.backtest_order_logger import BacktestOrderLogger

    logger = BacktestOrderLogger(db)

    # Get execution summary
    summary = await logger.get_execution_summary(args.export_execution)
    if not summary:
        print(f"Execution {args.export_execution} not found")
        return 1

    print(f"Exporting execution {args.export_execution}: {summary.get('execution_name')}")

    if args.csv_export:
        success = await logger.export_to_csv(args.export_execution, args.csv_export)
        if success:
            print(f"Exported to: {args.csv_export}")
            return 0
        else:
            print("Export failed")
            return 1
    else:
        # Just display the summary
        await display_results(db, args.export_execution, show_signals=True)
        return 0


def print_header(args, start_date, end_date, tickers):
    """Print backtest header."""
    print()
    print("=" * 70)
    print("STOCK BACKTEST")
    print("=" * 70)
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Period: {start_date} to {end_date} ({(end_date - start_date).days} days)")
    print(f"Timeframe: {args.timeframe}")

    if tickers:
        if len(tickers) <= 5:
            print(f"Tickers: {', '.join(tickers)}")
        else:
            print(f"Tickers: {len(tickers)} stocks")
    else:
        print(f"Tickers: All tradeable stocks")

    if args.sector:
        print(f"Sector: {args.sector}")

    print("=" * 70)
    print()


async def display_results(db, execution_id: int, show_signals: bool = False):
    """Display backtest results."""
    from .core.backtest.backtest_orchestrator import StockBacktestOrchestrator

    orchestrator = StockBacktestOrchestrator(db_manager=db)
    details = await orchestrator.get_execution_details(execution_id)

    if 'error' in details:
        print(f"Error: {details['error']}")
        return

    summary = details['summary']

    print()
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"Execution ID: {execution_id}")
    print(f"Status: {summary.get('status', 'unknown')}")
    print(f"Duration: {summary.get('duration_seconds', 0):.0f} seconds")
    print()

    # Performance metrics
    print(f"Total Signals: {summary.get('total_signals', 0)}")
    print(f"Total Trades: {summary.get('total_trades', 0)}")
    print(f"Winners: {summary.get('winners', 0)} | Losers: {summary.get('losers', 0)}")
    print(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
    print()
    total_pnl = summary.get('total_pnl_percent') or 0
    avg_win = summary.get('avg_win_percent') or 0
    avg_loss = summary.get('avg_loss_percent') or 0
    max_dd = summary.get('max_drawdown_percent') or 0

    print(f"Total P&L: {total_pnl:+.2f}%")
    print(f"Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}%")

    pf = summary.get('profit_factor')
    print(f"Profit Factor: {pf:.2f}" if pf else "Profit Factor: N/A")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print("-" * 70)

    # Quality tier breakdown
    if details.get('breakdown_by_quality'):
        print("\nQUALITY TIER BREAKDOWN")
        print("-" * 50)
        print(f"{'Tier':<8} {'Count':>8} {'Wins':>8} {'Win%':>10} {'Avg P&L':>10}")
        print("-" * 50)

        for row in details['breakdown_by_quality']:
            tier = row.get('quality_tier', 'N/A')
            count = row.get('count', 0)
            wins = row.get('wins', 0)
            win_pct = (wins / count * 100) if count > 0 else 0
            avg_pnl = row.get('avg_pnl', 0) or 0
            print(f"{tier:<8} {count:>8} {wins:>8} {win_pct:>9.1f}% {avg_pnl:>+9.2f}%")

    # Exit reason breakdown
    if details.get('breakdown_by_result'):
        print("\nEXIT REASON BREAKDOWN")
        print("-" * 50)
        print(f"{'Result':<12} {'Reason':<12} {'Count':>8} {'Avg P&L':>10}")
        print("-" * 50)

        for row in details['breakdown_by_result']:
            result = row.get('trade_result', 'N/A') or 'N/A'
            reason = row.get('exit_reason', 'N/A') or 'N/A'
            count = row.get('count', 0)
            avg_pnl = row.get('avg_pnl', 0) or 0
            print(f"{result:<12} {reason:<12} {count:>8} {avg_pnl:>+9.2f}%")

    # Show detailed signals if requested
    if show_signals:
        print("\nTOP 10 PERFORMERS")
        print("-" * 70)
        print(f"{'Date':<12} {'Ticker':<8} {'Entry':>10} {'Exit':>10} {'Result':<8} {'P&L':>10}")
        print("-" * 70)

        for row in details.get('top_performers', []):
            ts = row.get('signal_timestamp')
            date_str = ts.strftime('%Y-%m-%d') if ts else 'N/A'
            print(
                f"{date_str:<12} {row.get('ticker', ''):<8} "
                f"{row.get('entry_price', 0):>10.2f} {row.get('exit_price', 0):>10.2f} "
                f"{row.get('trade_result', ''):<8} {row.get('pnl_percent', 0):>+9.2f}%"
            )

        print("\nWORST 10 PERFORMERS")
        print("-" * 70)
        print(f"{'Date':<12} {'Ticker':<8} {'Entry':>10} {'Exit':>10} {'Result':<8} {'P&L':>10}")
        print("-" * 70)

        for row in details.get('worst_performers', []):
            ts = row.get('signal_timestamp')
            date_str = ts.strftime('%Y-%m-%d') if ts else 'N/A'
            print(
                f"{date_str:<12} {row.get('ticker', ''):<8} "
                f"{row.get('entry_price', 0):>10.2f} {row.get('exit_price', 0):>10.2f} "
                f"{row.get('trade_result', ''):<8} {row.get('pnl_percent', 0):>+9.2f}%"
            )


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if not args.ticker and not args.all and not args.export_execution and not args.compare:
        parser.print_help()
        print("\nError: Must specify --ticker, --all, --compare, or --export-execution")
        sys.exit(1)

    # Run backtest
    exit_code = asyncio.run(run_backtest(args))
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
