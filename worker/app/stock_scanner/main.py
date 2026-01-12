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

    # =========================================================================
    # CLAUDE AI ANALYSIS COMMANDS
    # =========================================================================

    async def cmd_claude_analyze(
        self,
        signal_id: int = None,
        min_tier: str = 'A',
        max_signals: int = 10,
        level: str = 'standard',
        model: str = None
    ):
        """
        Analyze signals with Claude AI.

        Can analyze a single signal by ID, or batch analyze top signals.
        """
        print("\n[CLAUDE] Claude AI Signal Analysis")
        print("=" * 60)

        try:
            from .scanners import ScannerManager

            manager = ScannerManager(self.db)
            await manager.initialize()

            # Check Claude availability
            claude_stats = manager.get_claude_stats()
            if not claude_stats.get('available', False):
                print("[ERROR] Claude API not available")
                print(f"   Error: {claude_stats.get('error', 'Unknown')}")
                print("\n[INFO] Ensure CLAUDE_API_KEY environment variable is set")
                return False

            print(f"[INFO] Claude API: Available")
            print(f"[INFO] Analysis level: {level}")
            print(f"[INFO] Model: {model or 'sonnet (default)'}")

            if signal_id:
                # Analyze single signal
                print(f"\n[ANALYZE] Analyzing signal ID: {signal_id}")

                analysis = await manager.analyze_single_signal_with_claude(
                    signal_id=signal_id,
                    analysis_level=level,
                    model=model
                )

                if analysis:
                    self._print_claude_analysis(analysis)
                    return True
                else:
                    print("[ERROR] Analysis failed or signal not found")
                    return False

            else:
                # Batch analyze
                print(f"\n[BATCH] Analyzing up to {max_signals} signals (min tier: {min_tier})")

                # Get unanalyzed signals
                signals = await manager.get_unanalyzed_signals(
                    min_tier=min_tier,
                    limit=max_signals
                )

                if not signals:
                    print("[INFO] No unanalyzed signals found matching criteria")
                    return True

                print(f"[INFO] Found {len(signals)} signals to analyze")

                # Analyze
                results = await manager.analyze_signals_with_claude(
                    signals=signals,
                    min_tier=min_tier,
                    max_signals=max_signals,
                    analysis_level=level,
                    model=model
                )

                # Print summary
                print(f"\n[RESULTS] Analyzed {len(results)} signals")
                print("-" * 60)

                for signal, analysis in results:
                    ticker = signal.get('ticker', '???')
                    scanner_tier = signal.get('quality_tier', '?')
                    print(f"   {ticker:6} | Scanner: {scanner_tier} | "
                          f"Claude: {analysis.grade} | {analysis.action} | "
                          f"{analysis.conviction}")

                # Print cost estimate
                total_tokens = sum(a.tokens_used for _, a in results if a.tokens_used)
                avg_latency = sum(a.latency_ms for _, a in results if a.latency_ms) / len(results) if results else 0
                estimated_cost = total_tokens * 0.000003  # ~$3/1M tokens for Sonnet

                print(f"\n[USAGE]")
                print(f"   Total tokens: {total_tokens:,}")
                print(f"   Avg latency: {avg_latency:.0f}ms")
                print(f"   Est. cost: ${estimated_cost:.4f}")

                return True

        except Exception as e:
            print(f"[ERROR] Claude analysis error: {e}")
            logger.exception("Claude analysis error")
            return False

    async def cmd_claude_status(self):
        """Show Claude analysis status and statistics"""
        print("\n[CLAUDE] Claude Analysis Status")
        print("=" * 60)

        try:
            from .scanners import ScannerManager

            manager = ScannerManager(self.db)

            # Get Claude stats
            stats = manager.get_claude_stats()

            print(f"\n[API STATUS]")
            print(f"   Available: {'Yes' if stats.get('available') else 'No'}")

            if not stats.get('available'):
                print(f"   Error: {stats.get('error', 'Unknown')}")
                return True

            print(f"   Total requests: {stats.get('total_requests', 0)}")
            print(f"   Successful: {stats.get('successful_requests', 0)}")
            print(f"   Failed: {stats.get('failed_requests', 0)}")
            print(f"   Total tokens: {stats.get('total_tokens', 0):,}")
            print(f"   Daily requests: {stats.get('daily_requests', 0)}")

            # Get analyzed signals from database
            query = """
                SELECT
                    COUNT(*) as total_analyzed,
                    COUNT(*) FILTER (WHERE claude_grade IN ('A+', 'A')) as high_grade,
                    COUNT(*) FILTER (WHERE claude_action = 'STRONG BUY') as strong_buys,
                    COUNT(*) FILTER (WHERE claude_action = 'BUY') as buys,
                    COUNT(*) FILTER (WHERE claude_action = 'HOLD') as holds,
                    COUNT(*) FILTER (WHERE claude_action = 'AVOID') as avoids,
                    AVG(claude_tokens_used) as avg_tokens,
                    AVG(claude_latency_ms) as avg_latency
                FROM stock_scanner_signals
                WHERE claude_analyzed_at IS NOT NULL
                  AND signal_timestamp >= NOW() - INTERVAL '7 days'
            """

            row = await self.db.fetchrow(query)

            if row and row['total_analyzed'] > 0:
                print(f"\n[ANALYSIS SUMMARY] (Last 7 days)")
                print(f"   Total analyzed: {row['total_analyzed']}")
                print(f"   High grade (A/A+): {row['high_grade']}")
                print(f"   Strong buys: {row['strong_buys']}")
                print(f"   Buys: {row['buys']}")
                print(f"   Holds: {row['holds']}")
                print(f"   Avoids: {row['avoids']}")
                print(f"   Avg tokens/signal: {int(row['avg_tokens'] or 0)}")
                print(f"   Avg latency: {int(row['avg_latency'] or 0)}ms")

            # Get pending signals
            pending_query = """
                SELECT COUNT(*) as pending
                FROM stock_scanner_signals
                WHERE claude_analyzed_at IS NULL
                  AND status = 'active'
                  AND signal_timestamp >= NOW() - INTERVAL '3 days'
            """
            pending_row = await self.db.fetchrow(pending_query)
            pending = pending_row['pending'] if pending_row else 0

            print(f"\n[PENDING]")
            print(f"   Signals awaiting analysis: {pending}")

            return True

        except Exception as e:
            print(f"[ERROR] Status error: {e}")
            logger.exception("Claude status error")
            return False

    async def cmd_claude_list(
        self,
        min_grade: str = None,
        days: int = 7,
        limit: int = 20,
        action_filter: str = None
    ):
        """List signals analyzed by Claude"""
        print("\n[CLAUDE] Claude-Analyzed Signals")
        print("=" * 60)

        try:
            from .scanners import ScannerManager

            manager = ScannerManager(self.db)
            await manager.initialize()

            # Build filter query
            filters = ["claude_analyzed_at IS NOT NULL"]
            params = [limit]

            if min_grade:
                grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
                valid_grades = [g for g, v in grade_order.items() if v >= grade_order.get(min_grade, 1)]
                grades_str = ', '.join(f"'{g}'" for g in valid_grades)
                filters.append(f"claude_grade IN ({grades_str})")

            if action_filter:
                filters.append(f"claude_action = '{action_filter.upper()}'")

            filters.append(f"signal_timestamp >= NOW() - INTERVAL '{days} days'")

            query = f"""
                SELECT
                    id, ticker, scanner_name, quality_tier, composite_score,
                    entry_price, stop_loss, risk_reward_ratio,
                    claude_grade, claude_score, claude_conviction, claude_action,
                    claude_thesis, claude_analyzed_at
                FROM stock_scanner_signals
                WHERE {' AND '.join(filters)}
                ORDER BY claude_score DESC, composite_score DESC
                LIMIT $1
            """

            rows = await self.db.fetch(query, *params)

            if not rows:
                print(f"[INFO] No signals found matching criteria")
                return True

            print(f"[INFO] Found {len(rows)} signals")
            print("-" * 60)

            for row in rows:
                row = dict(row)
                print(f"\n   {row['ticker']:6} | ID: {row['id']}")
                print(f"      Scanner: {row['scanner_name']} | Tier: {row['quality_tier']} | Score: {row['composite_score']}")
                print(f"      Claude: {row['claude_grade']} | Score: {row['claude_score']}/10 | {row['claude_action']}")
                print(f"      Entry: ${float(row['entry_price']):.2f} | Stop: ${float(row['stop_loss']):.2f} | R:R: {float(row['risk_reward_ratio']):.1f}:1")

                # Truncate thesis for display
                thesis = row.get('claude_thesis', '') or ''
                if len(thesis) > 80:
                    thesis = thesis[:77] + "..."
                if thesis:
                    print(f"      Thesis: {thesis}")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] List error: {e}")
            logger.exception("Claude list error")
            return False

    async def cmd_claude_detail(self, signal_id: int):
        """Show detailed Claude analysis for a signal"""
        print(f"\n[CLAUDE] Signal Analysis Detail - ID: {signal_id}")
        print("=" * 60)

        try:
            query = """
                SELECT * FROM stock_scanner_signals
                WHERE id = $1
            """
            row = await self.db.fetchrow(query, signal_id)

            if not row:
                print(f"[ERROR] Signal ID {signal_id} not found")
                return False

            row = dict(row)

            # Signal info
            print(f"\n[SIGNAL INFO]")
            print(f"   Ticker: {row['ticker']}")
            print(f"   Scanner: {row['scanner_name']}")
            print(f"   Quality Tier: {row['quality_tier']}")
            print(f"   Composite Score: {row['composite_score']}")
            print(f"   Entry: ${float(row['entry_price']):.2f}")
            print(f"   Stop Loss: ${float(row['stop_loss']):.2f}")
            print(f"   Risk/Reward: {float(row['risk_reward_ratio']):.2f}:1")
            print(f"   Signal Date: {row['signal_timestamp']}")

            # Claude analysis
            if row.get('claude_analyzed_at'):
                print(f"\n[CLAUDE ANALYSIS]")
                print(f"   Grade: {row['claude_grade']}")
                print(f"   Score: {row['claude_score']}/10")
                print(f"   Conviction: {row['claude_conviction']}")
                print(f"   Action: {row['claude_action']}")
                print(f"   Position: {row['claude_position_rec']}")
                print(f"   Stop Adjust: {row['claude_stop_adjustment']}")
                print(f"   Time Horizon: {row['claude_time_horizon']}")

                print(f"\n[THESIS]")
                thesis = row.get('claude_thesis', 'No thesis provided')
                # Word wrap thesis
                words = thesis.split()
                line = "   "
                for word in words:
                    if len(line) + len(word) > 75:
                        print(line)
                        line = "   " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    print(line)

                if row.get('claude_key_strengths'):
                    print(f"\n[KEY STRENGTHS]")
                    for strength in row['claude_key_strengths']:
                        print(f"   + {strength}")

                if row.get('claude_key_risks'):
                    print(f"\n[KEY RISKS]")
                    for risk in row['claude_key_risks']:
                        print(f"   - {risk}")

                print(f"\n[ANALYSIS META]")
                print(f"   Analyzed: {row['claude_analyzed_at']}")
                print(f"   Model: {row['claude_model']}")
                print(f"   Tokens: {row['claude_tokens_used']}")
                print(f"   Latency: {row['claude_latency_ms']}ms")

            else:
                print(f"\n[INFO] Signal has not been analyzed by Claude yet")
                print(f"   Run: python -m stock_scanner.main claude-analyze --signal-id {signal_id}")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] Detail error: {e}")
            logger.exception("Claude detail error")
            return False

    def _print_claude_analysis(self, analysis):
        """Print a single Claude analysis in detail"""
        print(f"\n[ANALYSIS RESULT]")
        print(f"   Grade: {analysis.grade}")
        print(f"   Score: {analysis.score}/10")
        print(f"   Conviction: {analysis.conviction}")
        print(f"   Action: {analysis.action}")
        print(f"   Position: {analysis.position_recommendation}")
        print(f"   Stop Adjust: {analysis.stop_adjustment}")
        print(f"   Time Horizon: {analysis.time_horizon}")

        if analysis.thesis:
            print(f"\n[THESIS]")
            print(f"   {analysis.thesis}")

        if analysis.key_strengths:
            print(f"\n[STRENGTHS]")
            for s in analysis.key_strengths:
                print(f"   + {s}")

        if analysis.key_risks:
            print(f"\n[RISKS]")
            for r in analysis.key_risks:
                print(f"   - {r}")

        print(f"\n[META]")
        print(f"   Model: {analysis.model}")
        print(f"   Tokens: {analysis.tokens_used}")
        print(f"   Latency: {analysis.latency_ms}ms")

    # =========================================================================
    # DEEP ANALYSIS COMMANDS
    # =========================================================================

    async def cmd_deep_analyze(
        self,
        ticker: str = None,
        signal_id: int = None,
        batch: bool = False,
        min_tier: str = 'A'
    ):
        """
        Run deep analysis on signals.

        Args:
            ticker: Specific ticker to analyze
            signal_id: Specific signal ID to analyze
            batch: Analyze all unanalyzed A+/A signals
            min_tier: Minimum tier for batch analysis
        """
        print("\n[DEEP ANALYSIS] Running Deep Analysis")
        print("=" * 60)

        try:
            from .services.deep_analysis import DeepAnalysisOrchestrator

            orchestrator = DeepAnalysisOrchestrator(self.db)

            if signal_id:
                # Analyze specific signal
                print(f"Analyzing signal ID {signal_id}...")
                result = await orchestrator.analyze_signal(signal_id)

                if result:
                    self._print_deep_analysis_result(result)
                    print(f"\n[OK] Analysis complete: DAQ={result.daq_score} ({result.daq_grade.value})")
                    return True
                else:
                    print(f"[ERROR] Failed to analyze signal {signal_id}")
                    return False

            elif ticker:
                # Find most recent signal for ticker and analyze
                query = """
                    SELECT id FROM stock_scanner_signals
                    WHERE ticker = $1 AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                row = await self.db.fetchrow(query, ticker.upper())

                if not row:
                    print(f"[ERROR] No active signal found for {ticker}")
                    return False

                print(f"Analyzing {ticker} (signal ID {row['id']})...")
                result = await orchestrator.analyze_signal(row['id'])

                if result:
                    self._print_deep_analysis_result(result)
                    print(f"\n[OK] Analysis complete: DAQ={result.daq_score} ({result.daq_grade.value})")
                    return True
                else:
                    print(f"[ERROR] Failed to analyze {ticker}")
                    return False

            else:
                # Batch analyze unanalyzed signals
                print(f"Batch analyzing unanalyzed {min_tier}+ signals...")
                results = await orchestrator.auto_analyze_high_quality_signals(
                    min_tier=min_tier,
                    days_back=1,
                    max_signals=50
                )

                if results:
                    print(f"\n[RESULTS] Analyzed {len(results)} signals:")
                    for r in results[:10]:  # Show first 10
                        print(f"   {r.ticker}: DAQ={r.daq_score} ({r.daq_grade.value})")

                    if len(results) > 10:
                        print(f"   ... and {len(results) - 10} more")

                    avg_daq = sum(r.daq_score for r in results) / len(results)
                    print(f"\n   Average DAQ: {avg_daq:.1f}")
                    return True
                else:
                    print("[INFO] No unanalyzed signals found")
                    return True

        except ImportError as e:
            print(f"[ERROR] Deep analysis module not available: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Deep analysis failed: {e}")
            logger.exception("Deep analysis error")
            return False

    async def cmd_deep_list(
        self,
        limit: int = 20,
        min_daq: int = None,
        min_grade: str = None
    ):
        """List recent deep analyses"""
        print("\n[DEEP ANALYSIS] Recent Analyses")
        print("=" * 60)

        try:
            from .services.deep_analysis import DeepAnalysisOrchestrator

            orchestrator = DeepAnalysisOrchestrator(self.db)
            analyses = await orchestrator.get_recent_analyses(
                limit=limit,
                min_daq_score=min_daq,
                min_grade=min_grade
            )

            if not analyses:
                print("[INFO] No deep analyses found")
                return True

            print(f"\n{'ID':>6} | {'Ticker':^8} | {'DAQ':^5} | {'Grade':^5} | {'Tier':^5} | {'MTF':^4} | {'SMC':^4} | {'News':^4}")
            print("-" * 70)

            for a in analyses:
                print(
                    f"{a['signal_id']:>6} | {a['ticker']:^8} | {a['daq_score']:^5} | {a['daq_grade']:^5} | "
                    f"{a.get('signal_tier', '-'):^5} | {a.get('mtf_score', 0):^4} | {a.get('smc_score', 0):^4} | {a.get('news_score', 0):^4}"
                )

            print(f"\nTotal: {len(analyses)} analyses")
            return True

        except ImportError as e:
            print(f"[ERROR] Deep analysis module not available: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to list analyses: {e}")
            logger.exception("Deep list error")
            return False

    async def cmd_deep_detail(self, signal_id: int):
        """Show detailed deep analysis for a signal"""
        print(f"\n[DEEP ANALYSIS] Signal #{signal_id} Details")
        print("=" * 60)

        try:
            from .services.deep_analysis import DeepAnalysisOrchestrator

            orchestrator = DeepAnalysisOrchestrator(self.db)
            analysis = await orchestrator.get_analysis_by_signal(signal_id)

            if not analysis:
                print(f"[ERROR] No deep analysis found for signal {signal_id}")
                return False

            print(f"\n[SIGNAL] {analysis['ticker']} (ID: {signal_id})")
            print(f"   DAQ Score: {analysis['daq_score']} ({analysis['daq_grade']})")
            print(f"   Analyzed: {analysis['created_at']}")

            print(f"\n[TECHNICAL SCORES]")
            print(f"   Multi-TF:  {analysis.get('mtf_score', '-')}/100")
            print(f"   Volume:    {analysis.get('volume_score', '-')}/100")
            print(f"   SMC:       {analysis.get('smc_score', '-')}/100")

            print(f"\n[FUNDAMENTAL SCORES]")
            print(f"   Quality:   {analysis.get('quality_score', '-')}/100")
            print(f"   Catalyst:  {analysis.get('catalyst_score', '-')}/100")

            print(f"\n[CONTEXTUAL SCORES]")
            print(f"   News:      {analysis.get('news_score', '-')}/100")
            print(f"   Regime:    {analysis.get('regime_score', '-')}/100")
            print(f"   Sector:    {analysis.get('sector_score', '-')}/100")

            print(f"\n[RISK FLAGS]")
            flags = []
            if analysis.get('earnings_within_7d'):
                flags.append("Earnings within 7 days")
            if analysis.get('high_short_interest'):
                flags.append("High short interest")
            if analysis.get('sector_underperforming'):
                flags.append("Sector underperforming")
            if flags:
                for f in flags:
                    print(f"   ! {f}")
            else:
                print("   None")

            if analysis.get('news_summary'):
                print(f"\n[NEWS SUMMARY]")
                print(f"   {analysis['news_summary']}")

            print(f"\n[META]")
            print(f"   Duration: {analysis.get('analysis_duration_ms', 0)}ms")

            return True

        except ImportError as e:
            print(f"[ERROR] Deep analysis module not available: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to get analysis details: {e}")
            logger.exception("Deep detail error")
            return False

    async def cmd_deep_summary(self, days: int = 7):
        """Show deep analysis summary statistics"""
        print(f"\n[DEEP ANALYSIS] Summary Statistics - Last {days} Days")
        print("=" * 60)

        try:
            from .services.deep_analysis import DeepAnalysisOrchestrator

            orchestrator = DeepAnalysisOrchestrator(self.db)
            summary = await orchestrator.get_analysis_summary(days=days)

            if not summary or not summary.get('total_analyses'):
                print("[INFO] No deep analyses found in the specified period")
                return True

            print(f"\n[OVERVIEW]")
            print(f"   Total Analyses: {summary.get('total_analyses', 0)}")
            print(f"   Average DAQ:    {summary.get('avg_daq_score', 0):.1f}")

            print(f"\n[BY GRADE]")
            print(f"   A+: {summary.get('a_plus_count', 0)}")
            print(f"   A:  {summary.get('a_count', 0)}")
            print(f"   B:  {summary.get('b_count', 0)}")
            print(f"   C:  {summary.get('c_count', 0)}")
            print(f"   D:  {summary.get('d_count', 0)}")

            print(f"\n[COMPONENT AVERAGES]")
            print(f"   MTF:      {summary.get('avg_mtf_score', 0):.1f}")
            print(f"   Volume:   {summary.get('avg_volume_score', 0):.1f}")
            print(f"   SMC:      {summary.get('avg_smc_score', 0):.1f}")
            print(f"   Quality:  {summary.get('avg_quality_score', 0):.1f}")
            print(f"   News:     {summary.get('avg_news_score', 0):.1f}")
            print(f"   Regime:   {summary.get('avg_regime_score', 0):.1f}")
            print(f"   Sector:   {summary.get('avg_sector_score', 0):.1f}")

            print(f"\n[PERFORMANCE]")
            print(f"   Avg Duration: {summary.get('avg_duration_ms', 0):.0f}ms")

            return True

        except ImportError as e:
            print(f"[ERROR] Deep analysis module not available: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to get summary: {e}")
            logger.exception("Deep summary error")
            return False

    def _print_deep_analysis_result(self, result):
        """Print formatted deep analysis result"""
        print(f"\n[SIGNAL] {result.ticker}")
        print(f"   DAQ Score: {result.daq_score} ({result.daq_grade.value})")

        print(f"\n[TECHNICAL] (45% weight)")
        print(f"   MTF Confluence: {result.technical.mtf.score}/100 "
              f"({result.technical.mtf.confluence_count}/{result.technical.mtf.total_timeframes} TFs aligned)")
        print(f"   Volume Profile: {result.technical.volume.score}/100 "
              f"(rel_vol: {result.technical.volume.relative_volume:.2f}x)")
        print(f"   SMC Structure:  {result.technical.smc.score}/100 "
              f"({result.technical.smc.smc_trend or 'N/A'})")

        print(f"\n[FUNDAMENTAL] (25% weight)")
        print(f"   Quality:   {result.fundamental.quality.score}/100")
        print(f"   Catalyst:  {result.fundamental.catalyst.score}/100 "
              f"(earnings {'in ' + str(result.fundamental.catalyst.days_to_earnings) + 'd' if result.fundamental.catalyst.days_to_earnings else 'N/A'})")

        print(f"\n[CONTEXTUAL] (30% weight)")
        print(f"   News:      {result.contextual.news.score}/100 "
              f"({result.contextual.news.sentiment_level})")
        print(f"   Regime:    {result.contextual.regime.score}/100 "
              f"({result.contextual.regime.regime.value})")
        print(f"   Sector:    {result.contextual.sector.score}/100 "
              f"({'outperforming' if result.contextual.sector.sector_outperforming else 'underperforming'})")

        if result.errors:
            print(f"\n[ERRORS]")
            for e in result.errors:
                print(f"   ! {e}")

    # =========================================================================
    # WATCHLIST DEEP ANALYSIS COMMANDS
    # =========================================================================

    async def cmd_watchlist_deep_analyze(
        self,
        calculation_date: Optional[str] = None,
        max_tier: int = 2,
        ticker: Optional[str] = None,
        force: bool = False
    ):
        """
        Run deep analysis on watchlist stocks.

        Args:
            calculation_date: Specific date to analyze (YYYY-MM-DD), defaults to latest
            max_tier: Maximum tier to analyze (1-5, default 2)
            ticker: Specific ticker to analyze (optional)
            force: Re-analyze even if already analyzed
        """
        print("\n[WATCHLIST DEEP ANALYSIS] Starting...")
        print("=" * 60)

        try:
            from .services.deep_analysis import DeepAnalysisOrchestrator

            orchestrator = DeepAnalysisOrchestrator(self.db)

            if ticker:
                # Single ticker analysis
                print(f"[INFO] Analyzing single ticker: {ticker}")

                # Find watchlist entry
                date_filter = f"calculation_date = '{calculation_date}'" if calculation_date else """
                    calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
                """

                query = f"""
                    SELECT id, ticker, calculation_date, tier
                    FROM stock_watchlist
                    WHERE {date_filter}
                      AND ticker = $1
                    LIMIT 1
                """
                row = await self.db.fetchrow(query, ticker)

                if not row:
                    print(f"[ERROR] Ticker {ticker} not found in watchlist")
                    return False

                result = await orchestrator.analyze_watchlist_ticker(
                    ticker=row['ticker'],
                    watchlist_id=row['id'],
                    tier=row['tier'],
                    save_to_db=True
                )

                if result:
                    self._print_deep_analysis_result(result)
                    print(f"\n[SUCCESS] DAQ saved to watchlist")
                    return True
                else:
                    print(f"[ERROR] Analysis failed for {ticker}")
                    return False

            else:
                # Batch analysis for top tiers
                results = await orchestrator.auto_analyze_watchlist(
                    max_tier=max_tier,
                    calculation_date=calculation_date,
                    max_tickers=100,
                    skip_analyzed=not force
                )

                if not results:
                    print("[INFO] No watchlist tickers to analyze")
                    return True

                # Summary
                print(f"\n[RESULTS] Analyzed {len(results)} watchlist tickers")
                print("-" * 60)

                # Group by grade
                by_grade = {'A+': [], 'A': [], 'B': [], 'C': [], 'D': []}
                for r in results:
                    by_grade[r.daq_grade.value].append(r)

                for grade in ['A+', 'A', 'B', 'C', 'D']:
                    if by_grade[grade]:
                        print(f"\n[GRADE {grade}] ({len(by_grade[grade])} tickers)")
                        for r in by_grade[grade][:5]:  # Show top 5 per grade
                            print(f"   {r.ticker}: DAQ={r.daq_score}")

                avg_daq = sum(r.daq_score for r in results) / len(results)
                print(f"\n[AVERAGE DAQ] {avg_daq:.1f}")

                return True

        except ImportError as e:
            print(f"[ERROR] Deep analysis module not available: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Watchlist deep analysis failed: {e}")
            logger.exception("Watchlist deep analysis error")
            return False

    async def cmd_watchlist_daq_list(
        self,
        calculation_date: Optional[str] = None,
        min_daq: int = 0,
        limit: int = 50
    ):
        """
        List watchlist stocks with their DAQ scores.

        Args:
            calculation_date: Specific date (YYYY-MM-DD), defaults to latest
            min_daq: Minimum DAQ score filter
            limit: Maximum results to show
        """
        print("\n[WATCHLIST DAQ] DAQ Scores")
        print("=" * 60)

        try:
            date_filter = f"calculation_date = '{calculation_date}'" if calculation_date else """
                calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)
            """

            query = f"""
                SELECT
                    ticker, tier, rank_in_tier, score,
                    daq_score, daq_grade,
                    daq_mtf_score, daq_volume_score, daq_smc_score,
                    daq_quality_score, daq_catalyst_score,
                    daq_news_score, daq_regime_score, daq_sector_score,
                    daq_earnings_risk, daq_analyzed_at
                FROM stock_watchlist
                WHERE {date_filter}
                  AND daq_score IS NOT NULL
                  AND daq_score >= $1
                ORDER BY daq_score DESC, tier, rank_in_tier
                LIMIT $2
            """

            rows = await self.db.fetch(query, min_daq, limit)

            if not rows:
                print("[INFO] No watchlist tickers with DAQ scores found")
                return True

            # Header
            print(f"{'Ticker':<8} {'Tier':>4} {'Score':>6} {'DAQ':>4} {'Grade':>5} "
                  f"{'MTF':>4} {'Vol':>4} {'SMC':>4} {'Qual':>4} {'Cat':>4} "
                  f"{'News':>4} {'Reg':>4} {'Sec':>4}")
            print("-" * 90)

            for row in rows:
                earnings_flag = "!" if row.get('daq_earnings_risk') else " "
                print(
                    f"{row['ticker']:<8} "
                    f"{row['tier']:>4} "
                    f"{row['score']:>6.1f} "
                    f"{row['daq_score']:>4} "
                    f"{row['daq_grade'] or '-':>5} "
                    f"{row.get('daq_mtf_score') or '-':>4} "
                    f"{row.get('daq_volume_score') or '-':>4} "
                    f"{row.get('daq_smc_score') or '-':>4} "
                    f"{row.get('daq_quality_score') or '-':>4} "
                    f"{row.get('daq_catalyst_score') or '-':>4} "
                    f"{row.get('daq_news_score') or '-':>4} "
                    f"{row.get('daq_regime_score') or '-':>4} "
                    f"{row.get('daq_sector_score') or '-':>4} "
                    f"{earnings_flag}"
                )

            print(f"\n[TOTAL] {len(rows)} tickers with DAQ >= {min_daq}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to list watchlist DAQ: {e}")
            logger.exception("Watchlist DAQ list error")
            return False

    # =========================================================================
    # TECHNICAL WATCHLIST DAQ COMMANDS (stock_watchlist_results)
    # =========================================================================

    async def cmd_technical_watchlist_daq(
        self,
        ticker: Optional[str] = None,
        watchlist_type: Optional[str] = None,
        scan_date: Optional[str] = None,
        max_tickers: int = 50,
        force: bool = False
    ):
        """
        Run deep analysis on technical watchlist stocks.

        Args:
            ticker: Specific ticker to analyze (optional)
            watchlist_type: Filter by watchlist type (ema_50_crossover, etc.)
            scan_date: Scan date filter (YYYY-MM-DD), defaults to latest
            max_tickers: Maximum tickers to analyze
            force: Re-analyze even if already analyzed
        """
        print("\n[TECHNICAL WATCHLIST DAQ] Starting...")
        print("=" * 60)

        try:
            from .services.deep_analysis import DeepAnalysisOrchestrator

            orchestrator = DeepAnalysisOrchestrator(self.db)

            if ticker:
                # Single ticker analysis - find in technical watchlist
                print(f"[INFO] Analyzing single ticker: {ticker}")

                conditions = ["ticker = $1"]
                params = [ticker]
                param_idx = 2

                if watchlist_type:
                    conditions.append(f"watchlist_name = ${param_idx}")
                    params.append(watchlist_type)
                    param_idx += 1

                if scan_date:
                    conditions.append(f"DATE(scan_date) = ${param_idx}")
                    params.append(scan_date)
                else:
                    # Latest scan
                    conditions.append("scan_date = (SELECT MAX(scan_date) FROM stock_watchlist_results)")

                where_clause = " AND ".join(conditions)

                query = f"""
                    SELECT id, ticker, watchlist_name, scan_date
                    FROM stock_watchlist_results
                    WHERE {where_clause}
                    ORDER BY scan_date DESC
                    LIMIT 1
                """
                row = await self.db.fetchrow(query, *params)

                if not row:
                    print(f"[ERROR] Ticker {ticker} not found in technical watchlist")
                    return False

                result = await orchestrator.analyze_technical_watchlist_ticker(
                    ticker=row['ticker'],
                    result_id=row['id'],
                    watchlist_name=row['watchlist_name'],
                    save_to_db=True
                )

                if result:
                    self._print_deep_analysis_result(result)
                    print(f"\n[SUCCESS] DAQ saved to technical watchlist")
                    return True
                else:
                    print(f"[ERROR] Analysis failed for {ticker}")
                    return False

            else:
                # Batch analysis
                results = await orchestrator.auto_analyze_technical_watchlist(
                    watchlist_name=watchlist_type,
                    scan_date=scan_date,
                    max_tickers=max_tickers,
                    skip_analyzed=not force
                )

                if not results:
                    print("[INFO] No technical watchlist tickers to analyze")
                    return True

                # Summary
                print(f"\n[RESULTS] Analyzed {len(results)} technical watchlist tickers")
                print("-" * 60)

                # Group by grade
                by_grade = {'A+': [], 'A': [], 'B': [], 'C': [], 'D': []}
                for r in results:
                    by_grade[r.daq_grade.value].append(r)

                for grade in ['A+', 'A', 'B', 'C', 'D']:
                    if by_grade[grade]:
                        print(f"\n[GRADE {grade}] ({len(by_grade[grade])} tickers)")
                        for r in by_grade[grade][:5]:
                            print(f"   {r.ticker}: DAQ={r.daq_score}")

                avg_daq = sum(r.daq_score for r in results) / len(results)
                print(f"\n[AVERAGE DAQ] {avg_daq:.1f}")

                return True

        except ImportError as e:
            print(f"[ERROR] Deep analysis module not available: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Technical watchlist DAQ failed: {e}")
            logger.exception("Technical watchlist DAQ error")
            return False

    async def cmd_technical_watchlist_daq_list(
        self,
        watchlist_type: Optional[str] = None,
        scan_date: Optional[str] = None,
        min_daq: int = 0,
        limit: int = 50
    ):
        """
        List technical watchlist stocks with their DAQ scores.

        Args:
            watchlist_type: Filter by watchlist type
            scan_date: Scan date filter (YYYY-MM-DD), defaults to latest
            min_daq: Minimum DAQ score filter
            limit: Maximum results to show
        """
        print("\n[TECHNICAL WATCHLIST DAQ] DAQ Scores")
        print("=" * 60)

        try:
            conditions = ["daq_score IS NOT NULL", "daq_score >= $1"]
            params = [min_daq]
            param_idx = 2

            if watchlist_type:
                conditions.append(f"watchlist_name = ${param_idx}")
                params.append(watchlist_type)
                param_idx += 1

            if scan_date:
                conditions.append(f"DATE(scan_date) = ${param_idx}")
                params.append(scan_date)
            else:
                # Latest scan for each watchlist type
                conditions.append("scan_date = (SELECT MAX(scan_date) FROM stock_watchlist_results)")

            where_clause = " AND ".join(conditions)

            params.append(limit)
            query = f"""
                SELECT
                    ticker, watchlist_name, price,
                    daq_score, daq_grade,
                    daq_mtf_score, daq_volume_score, daq_smc_score,
                    daq_quality_score, daq_catalyst_score,
                    daq_news_score, daq_regime_score, daq_sector_score,
                    daq_earnings_risk, daq_analyzed_at
                FROM stock_watchlist_results
                WHERE {where_clause}
                ORDER BY daq_score DESC
                LIMIT ${param_idx}
            """

            rows = await self.db.fetch(query, *params)

            if not rows:
                print("[INFO] No technical watchlist tickers with DAQ scores found")
                return True

            # Compact watchlist type names
            type_abbrev = {
                'ema_50_crossover': 'EMA50',
                'ema_20_crossover': 'EMA20',
                'macd_bullish_cross': 'MACD',
                'gap_up_continuation': 'GAP',
                'rsi_oversold_bounce': 'RSI'
            }

            # Header
            print(f"{'Ticker':<8} {'Type':<6} {'Price':>8} {'DAQ':>4} {'Grade':>5} "
                  f"{'MTF':>4} {'Vol':>4} {'SMC':>4} {'Qual':>4} {'Cat':>4} "
                  f"{'News':>4} {'Reg':>4} {'Sec':>4}")
            print("-" * 95)

            for row in rows:
                wl_type = type_abbrev.get(row['watchlist_name'], row['watchlist_name'][:6])
                earnings_flag = "!" if row.get('daq_earnings_risk') else " "
                price = row.get('price') or 0
                print(
                    f"{row['ticker']:<8} "
                    f"{wl_type:<6} "
                    f"${price:>7.2f} "
                    f"{row['daq_score']:>4} "
                    f"{row['daq_grade'] or '-':>5} "
                    f"{row.get('daq_mtf_score') or '-':>4} "
                    f"{row.get('daq_volume_score') or '-':>4} "
                    f"{row.get('daq_smc_score') or '-':>4} "
                    f"{row.get('daq_quality_score') or '-':>4} "
                    f"{row.get('daq_catalyst_score') or '-':>4} "
                    f"{row.get('daq_news_score') or '-':>4} "
                    f"{row.get('daq_regime_score') or '-':>4} "
                    f"{row.get('daq_sector_score') or '-':>4} "
                    f"{earnings_flag}"
                )

            print(f"\n[TOTAL] {len(rows)} tickers with DAQ >= {min_daq}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to list technical watchlist DAQ: {e}")
            logger.exception("Technical watchlist DAQ list error")
            return False

    # =========================================================================
    # BROKER TRADE STATISTICS COMMANDS
    # =========================================================================

    async def cmd_broker_positions(self):
        """Show current open positions from broker"""
        print("\n[BROKER] Open Positions")
        print("=" * 60)

        if not self.robomarkets:
            print("[ERROR] RoboMarkets client not configured")
            return False

        try:
            from .services.broker_trade_analyzer import BrokerTradeAnalyzer

            async with self.robomarkets:
                analyzer = BrokerTradeAnalyzer(self.robomarkets)
                summary = await analyzer.get_open_positions_summary()

            if summary.get("error"):
                print(f"[ERROR] {summary['error']}")
                return False

            print(f"\n[SUMMARY]")
            print(f"   Open Positions: {summary['count']}")
            print(f"   Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
            print(f"   Long: {summary['by_side'].get('long', 0)} | Short: {summary['by_side'].get('short', 0)}")

            if summary['positions']:
                print(f"\n[POSITIONS]")
                print("-" * 60)
                for p in summary['positions']:
                    pnl_color = "+" if p['unrealized_pnl'] >= 0 else ""
                    print(f"   {p['ticker']:6} | {p['side']:5} | Qty: {p['quantity']:>6.1f} | "
                          f"Entry: ${p['entry_price']:>8.2f} | Current: ${p['current_price']:>8.2f} | "
                          f"P&L: {pnl_color}${p['unrealized_pnl']:>8.2f}")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to get positions: {e}")
            logger.exception("Broker positions error")
            return False

    async def cmd_broker_trades(self, days: int = 30, limit: int = 50):
        """Show closed trades from broker"""
        print(f"\n[BROKER] Closed Trades - Last {days} Days")
        print("=" * 60)

        if not self.robomarkets:
            print("[ERROR] RoboMarkets client not configured")
            return False

        try:
            from .services.broker_trade_analyzer import BrokerTradeAnalyzer

            async with self.robomarkets:
                analyzer = BrokerTradeAnalyzer(self.robomarkets)
                summary = await analyzer.get_closed_trades_summary(days=days)

            if summary.get("error"):
                print(f"[ERROR] {summary['error']}")
                return False

            print(f"\n[SUMMARY]")
            print(f"   Total Closed Trades: {summary['count']}")
            print(f"   Total Profit/Loss: ${summary['total_profit']:,.2f}")

            if summary['trades']:
                print(f"\n[RECENT TRADES] (showing up to {limit})")
                print("-" * 60)
                for t in summary['trades'][:limit]:
                    pnl_color = "+" if t['profit'] >= 0 else ""
                    print(f"   {t['ticker']:6} | {t['side']:5} | "
                          f"Open: ${t['open_price']:>8.2f} | Close: ${t['close_price']:>8.2f} | "
                          f"P&L: {pnl_color}${t['profit']:>8.2f} ({pnl_color}{t['profit_pct']:.1f}%) | "
                          f"{t['duration_hours']:.1f}h")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to get trades: {e}")
            logger.exception("Broker trades error")
            return False

    async def cmd_broker_stats(self, days: int = 30):
        """Show comprehensive broker trading statistics"""
        print(f"\n[BROKER] Trading Statistics - Last {days} Days")
        print("=" * 60)

        if not self.robomarkets:
            print("[ERROR] RoboMarkets client not configured")
            return False

        try:
            from .services.broker_trade_analyzer import BrokerTradeAnalyzer

            async with self.robomarkets:
                analyzer = BrokerTradeAnalyzer(self.robomarkets)
                report = await analyzer.generate_report(days=days)

            print(report)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to get statistics: {e}")
            logger.exception("Broker stats error")
            return False

    async def cmd_broker_sync(self, days: int = 30):
        """Sync broker trades to local database"""
        print(f"\n[BROKER] Syncing Trades to Database - Last {days} Days")
        print("=" * 60)

        if not self.robomarkets:
            print("[ERROR] RoboMarkets client not configured")
            return False

        try:
            from .services.broker_trade_analyzer import BrokerTradeSync

            async with self.robomarkets:
                sync = BrokerTradeSync(self.db, self.robomarkets)
                result = await sync.sync_all(days=days)

            print(f"\n[RESULTS]")
            print(f"   Positions: {result['positions']['total']} fetched, "
                  f"{result['positions']['inserted']} new, {result['positions']['updated']} updated")
            print(f"   Trades: {result['trades']['total']} fetched, "
                  f"{result['trades']['inserted']} new, {result['trades']['updated']} updated")
            print(f"\n   Total: {result['total_fetched']} fetched, "
                  f"{result['total_inserted']} inserted, {result['total_updated']} updated")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to sync trades: {e}")
            logger.exception("Broker sync error")
            return False

    async def cmd_broker_db_stats(self, days: int = 30):
        """Show broker statistics from local database"""
        print(f"\n[BROKER] Database Statistics - Last {days} Days")
        print("=" * 60)

        try:
            from .services.broker_trade_analyzer import BrokerTradeSync

            # Create sync instance just for DB queries (no API needed)
            sync = BrokerTradeSync(self.db, None)
            stats = await sync.get_statistics_from_db(days=days)

            print(f"\n[OPEN POSITIONS]")
            print(f"   Count: {stats['open_positions']}")
            print(f"   Unrealized P&L: ${stats['open_unrealized_pnl']:,.2f}")

            print(f"\n[CLOSED TRADES]")
            print(f"   Total: {stats['total_trades']}")
            print(f"   Wins: {stats['winning_trades']} | Losses: {stats['losing_trades']}")
            print(f"   Win Rate: {stats['win_rate']:.1f}%")

            print(f"\n[PROFIT METRICS]")
            print(f"   Net Profit: ${stats['net_profit']:,.2f}")
            print(f"   Total Profit: ${stats['total_profit']:,.2f}")
            print(f"   Total Loss: ${stats['total_loss']:,.2f}")
            print(f"   Profit Factor: {stats['profit_factor']:.2f}")
            print(f"   Expectancy: ${stats['expectancy']:,.2f}")

            print(f"\n[AVERAGES]")
            print(f"   Avg Win: ${stats['avg_win']:,.2f} ({stats['avg_win_pct']:.1f}%)")
            print(f"   Avg Loss: ${stats['avg_loss']:,.2f} ({stats['avg_loss_pct']:.1f}%)")
            print(f"   Avg Duration: {stats['avg_duration_hours']:.1f} hours")

            print(f"\n[BY SIDE]")
            print(f"   Long: {stats['long_trades']} trades, {stats['long_win_rate']:.1f}% win, ${stats['long_profit']:,.2f}")
            print(f"   Short: {stats['short_trades']} trades, {stats['short_win_rate']:.1f}% win, ${stats['short_profit']:,.2f}")

            print("\n" + "=" * 60)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to get DB statistics: {e}")
            logger.exception("Broker DB stats error")
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

    # =========================================================================
    # CLAUDE ANALYSIS COMMANDS
    # =========================================================================

    # claude-analyze
    claude_analyze_parser = subparsers.add_parser(
        "claude-analyze",
        help="Analyze signals with Claude AI"
    )
    claude_analyze_parser.add_argument(
        "--signal-id",
        type=int,
        help="Analyze specific signal by ID"
    )
    claude_analyze_parser.add_argument(
        "--min-tier",
        default="A",
        choices=["A+", "A", "B", "C", "D"],
        help="Minimum quality tier for batch analysis (default: A)"
    )
    claude_analyze_parser.add_argument(
        "--max-signals",
        type=int,
        default=10,
        help="Maximum signals to analyze in batch (default: 10)"
    )
    claude_analyze_parser.add_argument(
        "--level",
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Analysis depth (default: standard)"
    )
    claude_analyze_parser.add_argument(
        "--model",
        choices=["haiku", "sonnet", "opus"],
        help="Claude model to use (default: sonnet)"
    )

    # claude-status
    subparsers.add_parser(
        "claude-status",
        help="Show Claude analysis status and statistics"
    )

    # claude-list
    claude_list_parser = subparsers.add_parser(
        "claude-list",
        help="List signals analyzed by Claude"
    )
    claude_list_parser.add_argument(
        "--min-grade",
        choices=["A+", "A", "B", "C", "D"],
        help="Filter by minimum Claude grade"
    )
    claude_list_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to look back (default: 7)"
    )
    claude_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum signals to show (default: 20)"
    )
    claude_list_parser.add_argument(
        "--action",
        choices=["STRONG BUY", "BUY", "HOLD", "AVOID"],
        help="Filter by Claude action recommendation"
    )

    # claude-detail
    claude_detail_parser = subparsers.add_parser(
        "claude-detail",
        help="Show detailed Claude analysis for a signal"
    )
    claude_detail_parser.add_argument(
        "signal_id",
        type=int,
        help="Signal ID to show details for"
    )

    # =========================================================================
    # DEEP ANALYSIS COMMANDS
    # =========================================================================

    # deep-analyze
    deep_analyze_parser = subparsers.add_parser(
        "deep-analyze",
        help="Run deep analysis on a signal (DAQ scoring)"
    )
    deep_analyze_parser.add_argument(
        "ticker",
        nargs="?",
        help="Specific ticker to analyze (optional - analyzes all A+/A if not provided)"
    )
    deep_analyze_parser.add_argument(
        "--signal-id",
        type=int,
        help="Analyze a specific signal by ID"
    )
    deep_analyze_parser.add_argument(
        "--batch",
        action="store_true",
        help="Analyze all unanalyzed A+/A signals"
    )
    deep_analyze_parser.add_argument(
        "--min-tier",
        default="A",
        choices=["A+", "A", "B"],
        help="Minimum tier for batch analysis (default: A)"
    )

    # deep-list
    deep_list_parser = subparsers.add_parser(
        "deep-list",
        help="List recent deep analyses"
    )
    deep_list_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum results to show (default: 20)"
    )
    deep_list_parser.add_argument(
        "--min-daq",
        type=int,
        help="Minimum DAQ score filter"
    )
    deep_list_parser.add_argument(
        "--min-grade",
        choices=["A+", "A", "B", "C", "D"],
        help="Minimum grade filter"
    )

    # deep-detail
    deep_detail_parser = subparsers.add_parser(
        "deep-detail",
        help="Show detailed deep analysis for a signal"
    )
    deep_detail_parser.add_argument(
        "signal_id",
        type=int,
        help="Signal ID to show deep analysis for"
    )

    # deep-summary
    deep_summary_parser = subparsers.add_parser(
        "deep-summary",
        help="Show deep analysis summary statistics"
    )
    deep_summary_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of history to summarize (default: 7)"
    )

    # =========================================================================
    # WATCHLIST DEEP ANALYSIS COMMANDS
    # =========================================================================

    # watchlist-deep-analyze
    watchlist_deep_parser = subparsers.add_parser(
        "watchlist-deep-analyze",
        help="Run deep analysis on watchlist stocks (DAQ scoring)"
    )
    watchlist_deep_parser.add_argument(
        "ticker",
        nargs="?",
        help="Specific ticker to analyze (optional - analyzes tier 1-2 if not provided)"
    )
    watchlist_deep_parser.add_argument(
        "--date",
        help="Watchlist calculation date (YYYY-MM-DD, default: latest)"
    )
    watchlist_deep_parser.add_argument(
        "--max-tier",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5],
        help="Maximum tier to analyze (default: 2)"
    )
    watchlist_deep_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze even if already analyzed"
    )

    # watchlist-daq-list
    watchlist_daq_list_parser = subparsers.add_parser(
        "watchlist-daq-list",
        help="List watchlist stocks with their DAQ scores"
    )
    watchlist_daq_list_parser.add_argument(
        "--date",
        help="Watchlist calculation date (YYYY-MM-DD, default: latest)"
    )
    watchlist_daq_list_parser.add_argument(
        "--min-daq",
        type=int,
        default=0,
        help="Minimum DAQ score filter (default: 0)"
    )
    watchlist_daq_list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum results to show (default: 50)"
    )

    # technical-watchlist-daq
    tech_wl_daq_parser = subparsers.add_parser(
        "technical-watchlist-daq",
        help="Run deep analysis on technical watchlist stocks (EMA crossovers, MACD, etc.)"
    )
    tech_wl_daq_parser.add_argument(
        "ticker",
        nargs="?",
        help="Optional: analyze a specific ticker"
    )
    tech_wl_daq_parser.add_argument(
        "--type",
        dest="watchlist_type",
        choices=["ema_50_crossover", "ema_20_crossover", "macd_bullish_cross", "gap_up_continuation", "rsi_oversold_bounce"],
        help="Filter by watchlist type"
    )
    tech_wl_daq_parser.add_argument(
        "--date",
        help="Scan date filter (YYYY-MM-DD, default: latest)"
    )
    tech_wl_daq_parser.add_argument(
        "--max",
        type=int,
        default=50,
        help="Maximum tickers to analyze (default: 50)"
    )
    tech_wl_daq_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze even if DAQ score exists"
    )

    # technical-watchlist-daq-list
    tech_wl_daq_list_parser = subparsers.add_parser(
        "technical-watchlist-daq-list",
        help="List technical watchlist stocks with their DAQ scores"
    )
    tech_wl_daq_list_parser.add_argument(
        "--type",
        dest="watchlist_type",
        choices=["ema_50_crossover", "ema_20_crossover", "macd_bullish_cross", "gap_up_continuation", "rsi_oversold_bounce"],
        help="Filter by watchlist type"
    )
    tech_wl_daq_list_parser.add_argument(
        "--date",
        help="Scan date filter (YYYY-MM-DD, default: latest)"
    )
    tech_wl_daq_list_parser.add_argument(
        "--min-daq",
        type=int,
        default=0,
        help="Minimum DAQ score filter (default: 0)"
    )
    tech_wl_daq_list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum results to show (default: 50)"
    )

    # =========================================================================
    # BROKER TRADE STATISTICS COMMANDS
    # =========================================================================

    # broker-positions
    subparsers.add_parser(
        "broker-positions",
        help="Show current open positions from broker"
    )

    # broker-trades
    broker_trades_parser = subparsers.add_parser(
        "broker-trades",
        help="Show closed trades from broker"
    )
    broker_trades_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of trade history to fetch (default: 30)"
    )
    broker_trades_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum trades to display (default: 50)"
    )

    # broker-stats
    broker_stats_parser = subparsers.add_parser(
        "broker-stats",
        help="Show comprehensive trading statistics from broker"
    )
    broker_stats_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to analyze (default: 30)"
    )

    # broker-sync
    broker_sync_parser = subparsers.add_parser(
        "broker-sync",
        help="Sync broker trades to local database"
    )
    broker_sync_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to sync (default: 30)"
    )

    # broker-db-stats
    broker_db_stats_parser = subparsers.add_parser(
        "broker-db-stats",
        help="Show broker statistics from local database"
    )
    broker_db_stats_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to analyze (default: 30)"
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
        # Claude AI commands
        elif args.command == "claude-analyze":
            success = await cli.cmd_claude_analyze(
                signal_id=getattr(args, "signal_id", None),
                min_tier=getattr(args, "min_tier", "A"),
                max_signals=getattr(args, "max_signals", 10),
                level=getattr(args, "level", "standard"),
                model=getattr(args, "model", None)
            )
        elif args.command == "claude-status":
            success = await cli.cmd_claude_status()
        elif args.command == "claude-list":
            success = await cli.cmd_claude_list(
                min_grade=getattr(args, "min_grade", None),
                days=getattr(args, "days", 7),
                limit=getattr(args, "limit", 20),
                action_filter=getattr(args, "action", None)
            )
        elif args.command == "claude-detail":
            success = await cli.cmd_claude_detail(args.signal_id)
        # Broker trade statistics commands
        elif args.command == "broker-positions":
            success = await cli.cmd_broker_positions()
        elif args.command == "broker-trades":
            success = await cli.cmd_broker_trades(
                days=getattr(args, "days", 30),
                limit=getattr(args, "limit", 50)
            )
        elif args.command == "broker-stats":
            success = await cli.cmd_broker_stats(
                days=getattr(args, "days", 30)
            )
        elif args.command == "broker-sync":
            success = await cli.cmd_broker_sync(
                days=getattr(args, "days", 30)
            )
        elif args.command == "broker-db-stats":
            success = await cli.cmd_broker_db_stats(
                days=getattr(args, "days", 30)
            )
        # Deep Analysis commands
        elif args.command == "deep-analyze":
            success = await cli.cmd_deep_analyze(
                ticker=getattr(args, "ticker", None),
                signal_id=getattr(args, "signal_id", None),
                batch=getattr(args, "batch", False),
                min_tier=getattr(args, "min_tier", "A")
            )
        elif args.command == "deep-list":
            success = await cli.cmd_deep_list(
                limit=getattr(args, "limit", 20),
                min_daq=getattr(args, "min_daq", None),
                min_grade=getattr(args, "min_grade", None)
            )
        elif args.command == "deep-detail":
            success = await cli.cmd_deep_detail(
                signal_id=args.signal_id
            )
        elif args.command == "deep-summary":
            success = await cli.cmd_deep_summary(
                days=getattr(args, "days", 7)
            )

        # Watchlist Deep Analysis Commands
        elif args.command == "watchlist-deep-analyze":
            success = await cli.cmd_watchlist_deep_analyze(
                calculation_date=getattr(args, "date", None),
                max_tier=getattr(args, "max_tier", 2),
                ticker=getattr(args, "ticker", None),
                force=getattr(args, "force", False)
            )
        elif args.command == "watchlist-daq-list":
            success = await cli.cmd_watchlist_daq_list(
                calculation_date=getattr(args, "date", None),
                min_daq=getattr(args, "min_daq", 0),
                limit=getattr(args, "limit", 50)
            )

        # Technical Watchlist DAQ Commands
        elif args.command == "technical-watchlist-daq":
            success = await cli.cmd_technical_watchlist_daq(
                ticker=getattr(args, "ticker", None),
                watchlist_type=getattr(args, "watchlist_type", None),
                scan_date=getattr(args, "date", None),
                max_tickers=getattr(args, "max", 50),
                force=getattr(args, "force", False)
            )
        elif args.command == "technical-watchlist-daq-list":
            success = await cli.cmd_technical_watchlist_daq_list(
                watchlist_type=getattr(args, "watchlist_type", None),
                scan_date=getattr(args, "date", None),
                min_daq=getattr(args, "min_daq", 0),
                limit=getattr(args, "limit", 50)
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
