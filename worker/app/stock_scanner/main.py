#!/usr/bin/env python3
"""
Stock Scanner CLI

Command-line interface for the stock scanner module.
Provides commands for:
- Syncing instruments from RoboMarkets
- Fetching historical data from yfinance
- Running scans (future)
- Backtesting (future)

Usage:
    python -m stock_scanner.main sync-tickers
    python -m stock_scanner.main fetch-data AAPL --days 60
    python -m stock_scanner.main fetch-all --days 60
    python -m stock_scanner.main status
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

from . import config
from .core.database.async_database_manager import AsyncDatabaseManager
from .core.trading.robomarkets_client import RoboMarketsClient
from .core.data_fetcher import StockDataFetcher

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class StockScannerCLI:
    """
    Stock Scanner Command Line Interface

    Provides commands for managing stock data and running scans.
    """

    def __init__(self, use_local_db: bool = False):
        """
        Initialize CLI

        Args:
            use_local_db: Use local database URL instead of Docker
        """
        self.db_url = config.get_database_url(use_local=use_local_db)
        self.db: Optional[AsyncDatabaseManager] = None
        self.robomarkets: Optional[RoboMarketsClient] = None
        self.data_fetcher: Optional[StockDataFetcher] = None

    async def setup(self):
        """Initialize database and API connections"""
        logger.info("Initializing Stock Scanner...")

        # Database
        self.db = AsyncDatabaseManager(self.db_url)
        await self.db.connect()
        logger.info("Database connected")

        # RoboMarkets client
        if config.ROBOMARKETS_API_KEY and config.ROBOMARKETS_ACCOUNT_ID:
            self.robomarkets = RoboMarketsClient(
                api_key=config.ROBOMARKETS_API_KEY,
                account_id=config.ROBOMARKETS_ACCOUNT_ID,
                base_url=config.ROBOMARKETS_API_URL
            )
            logger.info("RoboMarkets client configured")
        else:
            logger.warning("RoboMarkets credentials not configured")

        # Data fetcher
        self.data_fetcher = StockDataFetcher(
            db_manager=self.db,
            robomarkets_client=self.robomarkets
        )
        logger.info("Data fetcher initialized")

    async def cleanup(self):
        """Cleanup connections"""
        if self.data_fetcher:
            await self.data_fetcher.close()
        if self.robomarkets:
            await self.robomarkets.close()
        if self.db:
            await self.db.close()
        logger.info("Cleanup complete")

    # =========================================================================
    # COMMANDS
    # =========================================================================

    async def cmd_sync_tickers(self):
        """Sync US stocks (NYSE/NASDAQ) from RoboMarkets API to database"""
        print("\n[SYNC] Syncing US stocks from RoboMarkets API...")
        print("[INFO] Filtering: NYSE + NASDAQ only, excluding ETFs")

        try:
            # Sync US stocks directly from API (no account ID needed)
            stats = await self.data_fetcher.sync_us_stocks()

            print(f"\n[OK] Sync complete!")
            print(f"   - Total from API: {stats['total_from_api']:,}")
            print(f"   - US stocks found: {stats['us_stocks_found']:,}")
            print(f"   - Unique tickers: {stats['unique_tickers']:,}")
            print(f"   - NYSE: {stats['nyse_count']:,}")
            print(f"   - NASDAQ: {stats['nasdaq_count']:,}")
            print(f"   - Inserted: {stats['inserted']:,}")
            print(f"   - Updated: {stats['updated']:,}")

            # Show some examples
            tickers = await self.data_fetcher.get_tradeable_tickers()
            if tickers:
                print(f"\n[INFO] Sample tickers in database:")
                for ticker in tickers[:10]:
                    print(f"   - {ticker}")
                if len(tickers) > 10:
                    print(f"   ... and {len(tickers) - 10} more")

            return True

        except Exception as e:
            print(f"[ERROR] Error syncing tickers: {e}")
            logger.exception("Sync error")
            return False

    async def cmd_fetch_data(self, ticker: str, days: int = 60, interval: str = "1h"):
        """Fetch historical data for a single ticker"""
        print(f"\n[FETCH] Fetching {days} days of {interval} data for {ticker}...")

        try:
            count = await self.data_fetcher.fetch_historical_data(
                ticker=ticker,
                days=days,
                interval=interval
            )

            if count > 0:
                print(f"[OK] Fetched {count} candles for {ticker}")

                # Show data coverage
                coverage = await self.data_fetcher.get_data_coverage(ticker)
                if coverage:
                    print(f"\n[INFO] Data coverage for {ticker}:")
                    for tf, info in coverage.items():
                        print(f"   {tf}: {info['count']} candles "
                              f"({info['first'].strftime('%Y-%m-%d')} to "
                              f"{info['last'].strftime('%Y-%m-%d')})")
            else:
                print(f"[WARN] No data fetched for {ticker}")

            return count > 0

        except Exception as e:
            print(f"[ERROR] Error fetching data: {e}")
            logger.exception("Fetch error")
            return False

    async def cmd_fetch_all(self, days: int = 60, interval: str = "1h"):
        """Fetch historical data for all tradeable tickers"""
        print(f"\n[FETCH] Fetching {days} days of {interval} data for all tickers...")

        tickers = await self.data_fetcher.get_tradeable_tickers()

        if not tickers:
            print("[WARN] No tradeable tickers found. Run 'sync-tickers' first.")
            return False

        print(f"[INFO] Fetching data for {len(tickers)} tickers...")

        try:
            results = await self.data_fetcher.fetch_all_tickers(
                days=days,
                interval=interval,
                concurrency=config.MAX_CONCURRENT_FETCHES
            )

            successful = sum(1 for count in results.values() if count > 0)
            total_candles = sum(results.values())

            print(f"\n[OK] Fetch complete:")
            print(f"   - Successful: {successful}/{len(tickers)} tickers")
            print(f"   - Total candles: {total_candles:,}")

            # Show failures
            failures = [t for t, c in results.items() if c == 0]
            if failures:
                print(f"\n[WARN] Failed to fetch: {', '.join(failures[:5])}")
                if len(failures) > 5:
                    print(f"   ... and {len(failures) - 5} more")

            return successful > 0

        except Exception as e:
            print(f"[ERROR] Error fetching data: {e}")
            logger.exception("Fetch all error")
            return False

    async def cmd_fetch_watchlist(self, days: int = 60, interval: str = "1h"):
        """Fetch data for default watchlist (doesn't require RoboMarkets)"""
        print(f"\n[FETCH] Fetching {days} days of {interval} data for watchlist...")
        print(f"[INFO] Watchlist: {len(config.DEFAULT_WATCHLIST)} stocks")

        results = {}
        for i, ticker in enumerate(config.DEFAULT_WATCHLIST, 1):
            print(f"   [{i}/{len(config.DEFAULT_WATCHLIST)}] Fetching {ticker}...", end=" ")
            try:
                count = await self.data_fetcher.fetch_historical_data(
                    ticker=ticker,
                    days=days,
                    interval=interval
                )
                results[ticker] = count
                print(f"OK - {count} candles")
            except Exception as e:
                results[ticker] = 0
                print(f"FAILED - {e}")

        successful = sum(1 for c in results.values() if c > 0)
        total_candles = sum(results.values())

        print(f"\n[OK] Watchlist fetch complete:")
        print(f"   - Successful: {successful}/{len(config.DEFAULT_WATCHLIST)} tickers")
        print(f"   - Total candles: {total_candles:,}")

        return successful > 0

    async def cmd_synthesize(self, ticker: str = None, timeframe: str = "4h"):
        """Synthesize higher timeframe candles"""
        print(f"\n[SYNTH] Synthesizing {timeframe} candles...")

        try:
            if ticker:
                count = await self.data_fetcher.synthesize_candles(ticker, timeframe)
                print(f"[OK] Synthesized {count} {timeframe} candles for {ticker}")
            else:
                results = await self.data_fetcher.synthesize_all_tickers(timeframe)
                total = sum(results.values())
                print(f"[OK] Synthesized {total} {timeframe} candles for {len(results)} tickers")

            return True

        except Exception as e:
            print(f"[ERROR] Error synthesizing: {e}")
            return False

    async def cmd_status(self):
        """Show scanner status and data coverage"""
        print("\n[STATUS] Stock Scanner Status")
        print("=" * 50)

        # Config info
        print(f"\n[CONFIG] Configuration:")
        print(f"   Version: {config.STOCK_SCANNER_VERSION}")
        print(f"   Database: {self.db_url.split('@')[-1]}")
        print(f"   RoboMarkets: {'configured' if self.robomarkets else 'not configured'}")

        # Database health
        print(f"\n[DB] Database:")
        is_healthy = await self.db.health_check()
        print(f"   Status: {'connected' if is_healthy else 'disconnected'}")

        if is_healthy:
            stats = self.db.get_stats()
            print(f"   Pool: {stats.get('pool_used', 0)}/{stats.get('pool_size', 0)} connections")

            # Table counts
            tables = ["stock_instruments", "stock_candles", "stock_signals"]
            for table in tables:
                if await self.db.table_exists(table):
                    count = await self.db.get_table_count(table)
                    print(f"   {table}: {count:,} rows")
                else:
                    print(f"   {table}: not created")

        # Market status
        print(f"\n[MARKET] Market Status:")
        market = config.get_market_status()
        print(f"   Status: {market['status']}")
        print(f"   Time (ET): {market['current_time_et']}")
        if not market['is_open'] and 'next_open' in market:
            print(f"   Next Open: {market['next_open']}")

        # Watchlist
        print(f"\n[LIST] Watchlist: {len(config.DEFAULT_WATCHLIST)} stocks")

        print("\n" + "=" * 50)
        return True

    async def cmd_migrate(self):
        """Run database migrations"""
        print("\n[MIGRATE] Running database migrations...")

        import os
        migration_dir = os.path.join(os.path.dirname(__file__), "migrations")
        migration_file = os.path.join(migration_dir, "001_create_stock_tables.sql")

        if not os.path.exists(migration_file):
            print(f"[ERROR] Migration file not found: {migration_file}")
            return False

        try:
            success = await self.db.run_migration(migration_file)
            if success:
                print("[OK] Migration completed successfully")
            else:
                print("[ERROR] Migration failed")
            return success

        except Exception as e:
            print(f"[ERROR] Migration error: {e}")
            return False

    async def cmd_test_api(self):
        """Test RoboMarkets API connection"""
        print("\n[API] Testing RoboMarkets API connection...")

        if not self.robomarkets:
            print("[ERROR] RoboMarkets client not configured")
            return False

        try:
            # Test connection
            connected = await self.robomarkets.test_connection()
            if connected:
                print("[OK] API connection successful")

                # Get instrument count
                instruments = await self.robomarkets.get_instruments()
                print(f"[INFO] Available instruments: {len(instruments)}")

                # Try a quote
                if instruments:
                    ticker = instruments[0].ticker
                    try:
                        quote = await self.robomarkets.get_quote(ticker)
                        print(f"[INFO] Sample quote ({ticker}): Bid={quote.bid}, Ask={quote.ask}")
                    except Exception as e:
                        print(f"[WARN] Could not get quote: {e}")

                return True
            else:
                print("[ERROR] API connection failed")
                return False

        except Exception as e:
            print(f"[ERROR] API test error: {e}")
            return False

    async def cmd_run_scanners(self, scanner_name: str = None, save: bool = True):
        """Run signal scanners to generate trading signals"""
        print("\n[SCAN] Running Signal Scanners...")
        print("=" * 60)

        try:
            from .scanners import ScannerManager, TradingViewExporter

            # Initialize scanner manager
            enabled = [scanner_name] if scanner_name else None
            manager = ScannerManager(self.db, enabled_scanners=enabled)
            await manager.initialize()

            print(f"[INFO] Enabled scanners: {', '.join(manager.scanner_names)}")

            # Run all scanners
            signals = await manager.run_all_scanners(save_to_db=save)

            # Print results
            stats = manager.get_scan_stats()
            print(f"\n[RESULTS]")
            print(f"   Total Signals: {len(signals)}")
            print(f"   High Quality (A/A+): {stats.get('high_quality_count', 0)}")

            if signals:
                print(f"\n[TOP SIGNALS]")
                for i, signal in enumerate(signals[:10], 1):
                    print(f"   {i}. {signal.ticker:6} | {signal.quality_tier.value:3} | "
                          f"Score: {signal.composite_score:3} | "
                          f"Entry: ${float(signal.entry_price):7.2f} | "
                          f"Stop: ${float(signal.stop_loss):7.2f} | "
                          f"{signal.scanner_name}")

                # Export option
                exporter = TradingViewExporter()
                csv_content = exporter.to_csv(signals)
                print(f"\n[EXPORT] CSV Preview (first 5 lines):")
                for line in csv_content.split('\n')[:6]:
                    print(f"   {line}")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] Scanner error: {e}")
            logger.exception("Scanner error")
            return False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Stock Scanner CLI - Manage stock data and run scans",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local database instead of Docker"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # sync-tickers
    subparsers.add_parser(
        "sync-tickers",
        help="Sync available tickers from RoboMarkets"
    )

    # fetch-data
    fetch_parser = subparsers.add_parser(
        "fetch-data",
        help="Fetch historical data for a ticker"
    )
    fetch_parser.add_argument("ticker", help="Stock ticker symbol")
    fetch_parser.add_argument("--days", type=int, default=60, help="Days of history")
    fetch_parser.add_argument("--interval", default="1h", help="Data interval")

    # fetch-all
    fetch_all_parser = subparsers.add_parser(
        "fetch-all",
        help="Fetch data for all tradeable tickers"
    )
    fetch_all_parser.add_argument("--days", type=int, default=60, help="Days of history")
    fetch_all_parser.add_argument("--interval", default="1h", help="Data interval")

    # fetch-watchlist
    watchlist_parser = subparsers.add_parser(
        "fetch-watchlist",
        help="Fetch data for default watchlist"
    )
    watchlist_parser.add_argument("--days", type=int, default=60, help="Days of history")
    watchlist_parser.add_argument("--interval", default="1h", help="Data interval")

    # synthesize
    synth_parser = subparsers.add_parser(
        "synthesize",
        help="Synthesize higher timeframe candles"
    )
    synth_parser.add_argument("--ticker", help="Specific ticker (optional)")
    synth_parser.add_argument("--timeframe", default="4h", help="Target timeframe (4h or 1d)")

    # status
    subparsers.add_parser("status", help="Show scanner status")

    # migrate
    subparsers.add_parser("migrate", help="Run database migrations")

    # test-api
    subparsers.add_parser("test-api", help="Test RoboMarkets API connection")

    # run-scanners
    scan_parser = subparsers.add_parser(
        "run-scanners",
        help="Run signal scanners to generate trading signals"
    )
    scan_parser.add_argument(
        "--scanner",
        help="Specific scanner to run (trend_momentum, breakout_confirmation, mean_reversion, gap_and_go)"
    )
    scan_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save signals to database"
    )

    return parser


async def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cli = StockScannerCLI(use_local_db=args.local)

    try:
        await cli.setup()

        # Route to command
        if args.command == "sync-tickers":
            success = await cli.cmd_sync_tickers()
        elif args.command == "fetch-data":
            success = await cli.cmd_fetch_data(
                args.ticker, args.days, args.interval
            )
        elif args.command == "fetch-all":
            success = await cli.cmd_fetch_all(args.days, args.interval)
        elif args.command == "fetch-watchlist":
            success = await cli.cmd_fetch_watchlist(args.days, args.interval)
        elif args.command == "synthesize":
            success = await cli.cmd_synthesize(
                getattr(args, "ticker", None),
                args.timeframe
            )
        elif args.command == "status":
            success = await cli.cmd_status()
        elif args.command == "migrate":
            success = await cli.cmd_migrate()
        elif args.command == "test-api":
            success = await cli.cmd_test_api()
        elif args.command == "run-scanners":
            success = await cli.cmd_run_scanners(
                scanner_name=getattr(args, "scanner", None),
                save=not getattr(args, "no_save", False)
            )
        else:
            parser.print_help()
            success = False

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n[WARN] Interrupted by user")
        return 130

    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\n[ERROR] Error: {e}")
        return 1

    finally:
        await cli.cleanup()


def run():
    """Synchronous entry point"""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
