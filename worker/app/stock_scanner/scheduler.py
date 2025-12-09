#!/usr/bin/env python3
"""
Stock Scanner Scheduler - Enhanced Multi-Scan Pipeline

Runs multiple daily stock analysis scans:
1. Full Pipeline (05:00 UTC / 00:00 ET): Complete daily analysis - ready before EU morning
2. Pre-Market (6:00 AM ET): Quick scan for overnight gaps and earnings
3. Intraday (12:30 PM ET): Scanner-only run for momentum plays
4. Post-Market (4:30 PM ET): Quick scan for EOD patterns
5. Weekly (Sunday 1:00 AM ET): Instrument sync and fundamentals

Full Pipeline Stages:
1. Sync 1H candles from yfinance
2. Synthesize daily candles from 1H data
3. Calculate screening metrics (ATR, volume, momentum)
4. Run SMC (Smart Money Concepts) analysis
5. Build watchlist (tiered, scored stocks)
6. Run all signal scanners (ZLMA + 4 scanner strategies)

Usage:
    # Run full pipeline once
    docker exec stock-scheduler python -m stock_scanner.scheduler pipeline

    # Run individual stages
    docker exec stock-scheduler python -m stock_scanner.scheduler sync
    docker exec stock-scheduler python -m stock_scanner.scheduler synthesize
    docker exec stock-scheduler python -m stock_scanner.scheduler metrics
    docker exec stock-scheduler python -m stock_scanner.scheduler smc
    docker exec stock-scheduler python -m stock_scanner.scheduler watchlist
    docker exec stock-scheduler python -m stock_scanner.scheduler signals
    docker exec stock-scheduler python -m stock_scanner.scheduler scanners

    # Run quick scans
    docker exec stock-scheduler python -m stock_scanner.scheduler premarket
    docker exec stock-scheduler python -m stock_scanner.scheduler intraday
    docker exec stock-scheduler python -m stock_scanner.scheduler postmarket

    # Run continuous scheduler
    docker exec stock-scheduler python -m stock_scanner.scheduler run
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, time, timedelta
import pytz

sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.core.data_fetcher import StockDataFetcher
from stock_scanner.core.synthesis.daily_synthesizer import DailyCandleSynthesizer
from stock_scanner.core.metrics.calculator import MetricsCalculator
from stock_scanner.core.screener.watchlist_builder import WatchlistBuilder
from stock_scanner.core.smc.smc_stock_analyzer import SMCStockAnalyzer
from stock_scanner.core.fundamentals.fundamentals_fetcher import FundamentalsFetcher
from stock_scanner.strategies.zlma_trend import ZeroLagMATrendStrategy

# Import scanner manager for running all scanners
try:
    from stock_scanner.scanners.scanner_manager import ScannerManager
    SCANNER_MANAGER_AVAILABLE = True
except ImportError:
    SCANNER_MANAGER_AVAILABLE = False
    ScannerManager = None

# Import performance tracker
try:
    from stock_scanner.services.performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False
    PerformanceTracker = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stock_scheduler")


class StockScheduler:
    """
    Enhanced Stock Scanner Scheduler with multi-scan pipeline.

    Scan Types:
    - full_pipeline: Complete 9-stage analysis (05:00 UTC / 00:00 ET)
    - pre_market: Quick scan for overnight gaps (6:00 AM ET)
    - intraday: Scanner-only run for momentum plays (12:30 PM ET)
    - post_market: Quick scan for EOD patterns (4:30 PM ET)
    - weekly_sync: Instrument and fundamentals refresh (Sunday 1:00 AM ET)

    Full Pipeline Stages:
    1. sync       - Fetch latest 1H candles from yfinance
    2. synthesize - Aggregate 1H -> Daily candles
    3. metrics    - Calculate ATR, volume, momentum indicators
    4. smc        - Run SMC (Smart Money Concepts) analysis
    5. watchlist  - Build tiered, scored watchlist
    6. signals    - Run ZLMA + all scanner strategies

    Schedule (Mon-Fri):
    - 05:00 UTC (00:00 ET): Full pipeline - ready before EU morning
    - 6:00 AM ET: Pre-market scan (gaps, earnings)
    - 12:30 PM ET: Intraday scanner run
    - 4:30 PM ET: Post-market quick scan
    - Sunday 1:00 AM ET: Weekly sync
    """

    # Timezones
    ET = pytz.timezone('America/New_York')
    UTC = pytz.UTC

    # Schedule times - Multiple daily scans
    # Full pipeline runs at 05:00 UTC (00:00 ET) so data is ready before EU morning
    SCHEDULE = {
        'full_pipeline': {
            'time': time(0, 0),       # 00:00 ET = 05:00 UTC
            'type': 'full',
            'description': 'Complete daily pipeline (05:00 UTC)'
        },
        'pre_market': {
            'time': time(6, 0),       # 6:00 AM ET
            'type': 'quick',
            'description': 'Pre-market gap and earnings scan'
        },
        'intraday': {
            'time': time(12, 30),     # 12:30 PM ET
            'type': 'scanner_only',
            'description': 'Intraday momentum scanner'
        },
        'post_market': {
            'time': time(16, 30),     # 4:30 PM ET
            'type': 'quick',
            'description': 'Post-market EOD patterns'
        }
    }

    # Legacy constants for backward compatibility
    PIPELINE_TIME = time(0, 0)    # 00:00 ET (05:00 UTC) - Full pipeline
    WEEKLY_SYNC_TIME = time(1, 0) # 1:00 AM ET on Sunday
    WEEKLY_SYNC_DAY = 6           # Sunday = 6

    def __init__(self):
        self.db: AsyncDatabaseManager = None
        self.fetcher: StockDataFetcher = None
        self.synthesizer: DailyCandleSynthesizer = None
        self.calculator: MetricsCalculator = None
        self.smc_analyzer: SMCStockAnalyzer = None
        self.fundamentals: FundamentalsFetcher = None
        self.watchlist_builder: WatchlistBuilder = None
        self.zlma_strategy: ZeroLagMATrendStrategy = None
        self.scanner_manager: ScannerManager = None
        self.performance_tracker: PerformanceTracker = None
        self.running = False

    async def setup(self):
        """Initialize all pipeline components"""
        logger.info("Initializing Enhanced Stock Scheduler...")

        self.db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
        await self.db.connect()

        # Initialize all components
        self.fetcher = StockDataFetcher(db_manager=self.db)
        self.synthesizer = DailyCandleSynthesizer(db_manager=self.db)
        self.calculator = MetricsCalculator(db_manager=self.db)
        self.smc_analyzer = SMCStockAnalyzer(db_manager=self.db)
        self.fundamentals = FundamentalsFetcher(db_manager=self.db)
        self.watchlist_builder = WatchlistBuilder(db_manager=self.db)
        self.zlma_strategy = ZeroLagMATrendStrategy(db_manager=self.db)

        # Initialize scanner manager if available
        if SCANNER_MANAGER_AVAILABLE:
            self.scanner_manager = ScannerManager(self.db)
            await self.scanner_manager.initialize()
            logger.info("Scanner Manager initialized with scanners: " +
                       ", ".join(self.scanner_manager.scanner_names))

        # Initialize performance tracker if available
        if PERFORMANCE_TRACKER_AVAILABLE:
            self.performance_tracker = PerformanceTracker(self.db)
            logger.info("Performance Tracker initialized")

        logger.info("Enhanced Stock Scheduler initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.fetcher:
            await self.fetcher.close()
        if self.fundamentals:
            await self.fundamentals.close()
        if self.db:
            await self.db.close()
        logger.info("Stock Scheduler cleaned up")

    def get_next_pipeline(self) -> datetime:
        """Get next daily pipeline execution time"""
        now = datetime.now(self.ET)
        target = datetime.combine(now.date(), self.PIPELINE_TIME)
        target = self.ET.localize(target)

        # If we've passed today's time, schedule for tomorrow
        if now >= target:
            target += timedelta(days=1)

        # Skip weekends (Saturday=5, Sunday=6)
        while target.weekday() in (5, 6):
            target += timedelta(days=1)

        return target

    def get_next_weekly_sync(self) -> datetime:
        """Get next weekly instrument sync time"""
        now = datetime.now(self.ET)
        target = datetime.combine(now.date(), self.WEEKLY_SYNC_TIME)
        target = self.ET.localize(target)

        # Find next Sunday
        days_until_sunday = (self.WEEKLY_SYNC_DAY - now.weekday()) % 7
        if days_until_sunday == 0 and now >= target:
            days_until_sunday = 7

        target += timedelta(days=days_until_sunday)
        return target

    async def run_pipeline(self):
        """
        Run the complete daily pipeline.

        Stages:
        1. Sync 1H candles
        2. Synthesize daily candles
        3. Calculate metrics
        4. Build watchlist
        5. Run ZLMA strategy
        """
        logger.info("=" * 80)
        logger.info(" DAILY PIPELINE - Starting")
        logger.info("=" * 80)

        pipeline_start = datetime.now()
        results = {}

        # === STAGE 1: Sync 1H Candles ===
        logger.info("\n[STAGE 1/6] Syncing 1H candle data...")
        try:
            sync_stats = await self._sync_hourly_data()
            results['sync'] = sync_stats
            logger.info(f"[OK] Sync: {sync_stats['successful']} stocks, {sync_stats['total_candles']:,} candles")
        except Exception as e:
            logger.error(f"[FAIL] Sync: {e}")
            results['sync'] = {'error': str(e)}

        # === STAGE 2: Synthesize Daily Candles ===
        logger.info("\n[STAGE 2/6] Synthesizing daily candles...")
        try:
            synth_stats = await self.synthesizer.synthesize_all_daily(incremental=True)
            results['synthesis'] = synth_stats
            logger.info(f"[OK] Synthesis: {synth_stats['total_daily_candles']:,} daily candles")
        except Exception as e:
            logger.error(f"[FAIL] Synthesis: {e}")
            results['synthesis'] = {'error': str(e)}

        # === STAGE 3: Calculate Metrics ===
        logger.info("\n[STAGE 3/6] Calculating screening metrics...")
        try:
            metrics_stats = await self.calculator.calculate_all_metrics()
            results['metrics'] = metrics_stats
            logger.info(f"[OK] Metrics: {metrics_stats['successful']} stocks")
        except Exception as e:
            logger.error(f"[FAIL] Metrics: {e}")
            results['metrics'] = {'error': str(e)}

        # === STAGE 4: SMC Analysis ===
        logger.info("\n[STAGE 4/6] Running SMC analysis...")
        try:
            smc_stats = await self.smc_analyzer.run_analysis_pipeline()
            results['smc'] = smc_stats
            logger.info(f"[OK] SMC: {smc_stats['analyzed']} stocks analyzed")
            logger.info(f"     Bullish: {smc_stats['bullish']}, "
                       f"Bearish: {smc_stats['bearish']}, "
                       f"Neutral: {smc_stats['neutral']}")
        except Exception as e:
            logger.error(f"[FAIL] SMC: {e}")
            results['smc'] = {'error': str(e)}

        # === STAGE 5: Build Watchlist ===
        logger.info("\n[STAGE 5/6] Building watchlist...")
        try:
            watchlist_stats = await self.watchlist_builder.build_watchlist()
            results['watchlist'] = watchlist_stats
            logger.info(f"[OK] Watchlist: {watchlist_stats['passed_filters']} stocks")
            logger.info(f"     Tier 1: {watchlist_stats['tier_1']}, "
                       f"Tier 2: {watchlist_stats['tier_2']}, "
                       f"Tier 3: {watchlist_stats['tier_3']}, "
                       f"Tier 4: {watchlist_stats['tier_4']}")
        except Exception as e:
            logger.error(f"[FAIL] Watchlist: {e}")
            results['watchlist'] = {'error': str(e)}

        # === STAGE 6: ZLMA Strategy Signals ===
        logger.info("\n[STAGE 6/7] Running ZLMA strategy...")
        try:
            signals = await self.zlma_strategy.scan_all_stocks()
            results['zlma_signals'] = {
                'total': len(signals),
                'buy': sum(1 for s in signals if s.signal_type == 'BUY'),
                'sell': sum(1 for s in signals if s.signal_type == 'SELL')
            }
            logger.info(f"[OK] ZLMA Signals: {len(signals)} total "
                       f"(BUY: {results['zlma_signals']['buy']}, SELL: {results['zlma_signals']['sell']})")
        except Exception as e:
            logger.error(f"[FAIL] ZLMA Signals: {e}")
            results['zlma_signals'] = {'error': str(e)}

        # === STAGE 7: All Scanner Strategies ===
        logger.info("\n[STAGE 7/9] Running all scanner strategies...")
        try:
            if self.scanner_manager:
                scanner_signals = await self.scanner_manager.run_all_scanners()
                results['scanner_signals'] = {
                    'total': len(scanner_signals),
                    'by_scanner': self.scanner_manager.get_scan_stats().get('signals_by_scanner', {}),
                    'by_tier': self.scanner_manager.get_scan_stats().get('signals_by_tier', {}),
                    'high_quality': sum(1 for s in scanner_signals if s.is_high_quality)
                }
                logger.info(f"[OK] Scanner Signals: {len(scanner_signals)} total "
                           f"(A/A+: {results['scanner_signals']['high_quality']})")
            else:
                logger.warning("[SKIP] Scanner Manager not available")
                results['scanner_signals'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Scanner Signals: {e}")
            results['scanner_signals'] = {'error': str(e)}

        # === STAGE 8: Performance Tracking ===
        logger.info("\n[STAGE 8/9] Updating signal performance...")
        try:
            if self.performance_tracker:
                status_updates = await self.performance_tracker.update_signal_statuses()
                await self.performance_tracker.record_daily_performance()
                results['performance'] = status_updates
                logger.info(f"[OK] Performance: Updated {sum(status_updates.values())} signals")
            else:
                logger.warning("[SKIP] Performance Tracker not available")
                results['performance'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Performance Tracking: {e}")
            results['performance'] = {'error': str(e)}

        # === STAGE 9: Claude AI Analysis ===
        logger.info("\n[STAGE 9/9] Running Claude AI analysis on top signals...")
        try:
            if self.scanner_manager:
                # Analyze top A+ and A tier signals
                analyzed = await self.scanner_manager.analyze_signals_with_claude(
                    min_tier='A',
                    max_signals=10,
                    analysis_level='standard'
                )
                results['claude_analysis'] = {
                    'analyzed': len(analyzed),
                    'skipped': 0
                }
                logger.info(f"[OK] Claude Analysis: Analyzed {len(analyzed)} signals")
            else:
                logger.warning("[SKIP] Claude Analysis not available")
                results['claude_analysis'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Claude Analysis: {e}")
            results['claude_analysis'] = {'error': str(e)}

        # === Pipeline Summary ===
        total_duration = (datetime.now() - pipeline_start).total_seconds()

        logger.info("\n" + "=" * 80)
        logger.info(" DAILY PIPELINE - Complete")
        logger.info("=" * 80)
        logger.info(f"Total duration: {int(total_duration//60)}m {int(total_duration%60)}s")

        # Log to database
        await self._log_pipeline_run('daily_pipeline', results, total_duration)

        return results

    # Sync configuration to avoid Yahoo Finance rate limiting
    SYNC_CONCURRENCY = 3  # Max concurrent requests (reduced from 5)
    SYNC_BATCH_SIZE = 100  # Process in batches
    SYNC_BATCH_DELAY = 2.0  # Seconds between batches

    async def _sync_hourly_data(self) -> dict:
        """
        Sync 1H candles for stocks needing update.

        Uses batched processing with delays to avoid Yahoo Finance rate limiting.
        Implements retry tracking for failed tickers.
        """
        # Get tickers needing update (older than 20 hours)
        query = """
            SELECT i.ticker
            FROM stock_instruments i
            LEFT JOIN (
                SELECT ticker, MAX(timestamp) as last_ts
                FROM stock_candles WHERE timeframe = '1h'
                GROUP BY ticker
            ) c ON i.ticker = c.ticker
            WHERE i.is_active = TRUE AND i.is_tradeable = TRUE
            AND (c.last_ts IS NULL OR c.last_ts < NOW() - INTERVAL '20 hours')
        """
        rows = await self.db.fetch(query)
        tickers = [row['ticker'] for row in rows]

        if not tickers:
            return {'successful': 0, 'failed': 0, 'total_candles': 0, 'message': 'All up to date'}

        logger.info(f"Updating {len(tickers)} tickers in batches of {self.SYNC_BATCH_SIZE}...")

        # Process in batches to avoid overwhelming Yahoo Finance
        successful = 0
        failed = 0
        total_candles = 0
        failed_tickers = []

        # Split into batches
        batches = [
            tickers[i:i + self.SYNC_BATCH_SIZE]
            for i in range(0, len(tickers), self.SYNC_BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} tickers)...")

            semaphore = asyncio.Semaphore(self.SYNC_CONCURRENCY)

            async def update_ticker(ticker):
                async with semaphore:
                    try:
                        count = await self.fetcher.fetch_historical_data(
                            ticker, days=5, interval='1h'
                        )
                        return (ticker, count, None)
                    except Exception as e:
                        return (ticker, 0, str(e))

            tasks = [update_ticker(t) for t in batch]
            results = await asyncio.gather(*tasks)

            batch_success = 0
            batch_failed = 0
            for ticker, count, error in results:
                if error:
                    failed += 1
                    batch_failed += 1
                    failed_tickers.append(ticker)
                elif count > 0:
                    successful += 1
                    batch_success += 1
                    total_candles += count
                else:
                    failed += 1
                    batch_failed += 1
                    failed_tickers.append(ticker)

            logger.info(f"  Batch {batch_num}: {batch_success} success, {batch_failed} failed")

            # Delay between batches (except for last batch)
            if batch_num < len(batches):
                logger.debug(f"  Waiting {self.SYNC_BATCH_DELAY}s before next batch...")
                await asyncio.sleep(self.SYNC_BATCH_DELAY)

        # Retry failed tickers with longer delays if we have time
        if failed_tickers and len(failed_tickers) <= 500:
            logger.info(f"Retrying {len(failed_tickers)} failed tickers with longer delays...")
            retry_success = 0

            for ticker in failed_tickers[:200]:  # Limit retries
                try:
                    await asyncio.sleep(0.5)  # 500ms delay between retries
                    count = await self.fetcher.fetch_historical_data(
                        ticker, days=5, interval='1h'
                    )
                    if count > 0:
                        retry_success += 1
                        successful += 1
                        failed -= 1
                        total_candles += count
                except Exception:
                    pass

            if retry_success > 0:
                logger.info(f"  Retry recovered {retry_success} tickers")

        return {
            'total_tickers': len(tickers),
            'successful': successful,
            'failed': failed,
            'total_candles': total_candles
        }

    async def run_pre_market_scan(self):
        """
        Pre-market scan (6:00 AM ET) - Quick scan for overnight opportunities.

        Scans for:
        - Overnight gaps (pre-market price vs previous close)
        - Earnings releases from previous night
        - High pre-market volume
        """
        logger.info("=" * 60)
        logger.info("PRE-MARKET SCAN - Checking overnight gaps and earnings")
        logger.info("=" * 60)

        start_time = datetime.now()
        results = {}

        try:
            # Run scanner manager for quick signals
            if self.scanner_manager:
                signals = await self.scanner_manager.run_all_scanners()
                results['scanner_signals'] = {
                    'total': len(signals),
                    'high_quality': sum(1 for s in signals if s.is_high_quality)
                }
                logger.info(f"[OK] Pre-market signals: {len(signals)} "
                           f"(A/A+: {results['scanner_signals']['high_quality']})")

            # Check for earnings releases
            earnings_query = """
                SELECT ticker, name, earnings_date
                FROM stock_instruments
                WHERE earnings_date IS NOT NULL
                AND earnings_date >= CURRENT_DATE - INTERVAL '1 day'
                AND earnings_date <= CURRENT_DATE
                ORDER BY earnings_date DESC
                LIMIT 50
            """
            earnings = await self.db.fetch(earnings_query)
            results['earnings_stocks'] = len(earnings)
            if earnings:
                logger.info(f"[OK] Earnings releases: {len(earnings)} stocks")
                for e in earnings[:5]:
                    logger.info(f"     {e['ticker']}: {e['name'][:30]}")

        except Exception as e:
            logger.error(f"Pre-market scan error: {e}")
            results['error'] = str(e)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nPre-market scan complete in {elapsed:.1f}s")

        await self._log_pipeline_run('pre_market_scan', results, elapsed)
        return results

    async def run_intraday_scan(self):
        """
        Intraday scan (12:30 PM ET) - Scanner-only run for momentum plays.

        Runs all scanner strategies on current data to find:
        - Momentum continuation
        - Breakout confirmations
        - Mean reversion setups
        """
        logger.info("=" * 60)
        logger.info("INTRADAY SCAN - Running momentum scanners")
        logger.info("=" * 60)

        start_time = datetime.now()
        results = {}

        try:
            # Quick sync of latest candles (top tickers only)
            logger.info("[STAGE 1/2] Quick candle sync (top tickers)...")
            top_tickers_query = """
                SELECT ticker FROM stock_watchlist
                WHERE tier IN (1, 2)
                ORDER BY rank_overall
                LIMIT 500
            """
            rows = await self.db.fetch(top_tickers_query)
            top_tickers = [r['ticker'] for r in rows]

            sync_count = 0
            for ticker in top_tickers[:100]:  # Quick sync top 100
                try:
                    count = await self.fetcher.fetch_historical_data(ticker, days=1, interval='1h')
                    if count > 0:
                        sync_count += 1
                except Exception:
                    pass

            results['quick_sync'] = sync_count
            logger.info(f"[OK] Quick sync: {sync_count} tickers updated")

            # Run scanner manager
            logger.info("[STAGE 2/2] Running scanners...")
            if self.scanner_manager:
                signals = await self.scanner_manager.run_all_scanners()
                results['scanner_signals'] = {
                    'total': len(signals),
                    'by_scanner': self.scanner_manager.get_scan_stats().get('signals_by_scanner', {}),
                    'high_quality': sum(1 for s in signals if s.is_high_quality)
                }
                logger.info(f"[OK] Intraday signals: {len(signals)} "
                           f"(A/A+: {results['scanner_signals']['high_quality']})")

        except Exception as e:
            logger.error(f"Intraday scan error: {e}")
            results['error'] = str(e)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nIntraday scan complete in {elapsed:.1f}s")

        await self._log_pipeline_run('intraday_scan', results, elapsed)
        return results

    async def run_post_market_scan(self):
        """
        Post-market scan (4:30 PM ET) - Quick scan for EOD patterns.

        Scans for:
        - End-of-day breakouts/breakdowns
        - Volume spikes
        - Pattern completions
        """
        logger.info("=" * 60)
        logger.info("POST-MARKET SCAN - Checking EOD patterns")
        logger.info("=" * 60)

        start_time = datetime.now()
        results = {}

        try:
            # Run ZLMA strategy for fresh signals
            logger.info("[STAGE 1/2] Running ZLMA strategy...")
            zlma_signals = await self.zlma_strategy.scan_all_stocks()
            results['zlma_signals'] = {
                'total': len(zlma_signals),
                'buy': sum(1 for s in zlma_signals if s.signal_type == 'BUY'),
                'sell': sum(1 for s in zlma_signals if s.signal_type == 'SELL')
            }
            logger.info(f"[OK] ZLMA signals: {len(zlma_signals)}")

            # Run scanner manager
            logger.info("[STAGE 2/2] Running scanners...")
            if self.scanner_manager:
                signals = await self.scanner_manager.run_all_scanners()
                results['scanner_signals'] = {
                    'total': len(signals),
                    'high_quality': sum(1 for s in signals if s.is_high_quality)
                }
                logger.info(f"[OK] Scanner signals: {len(signals)} "
                           f"(A/A+: {results['scanner_signals']['high_quality']})")

        except Exception as e:
            logger.error(f"Post-market scan error: {e}")
            results['error'] = str(e)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nPost-market scan complete in {elapsed:.1f}s")

        await self._log_pipeline_run('post_market_scan', results, elapsed)
        return results

    async def weekly_sync(self):
        """Run weekly instrument sync from RoboMarkets + fundamentals update"""
        logger.info("=" * 60)
        logger.info("WEEKLY SYNC - Syncing instruments and fundamentals")
        logger.info("=" * 60)

        # Part 1: Sync instruments from RoboMarkets
        try:
            logger.info("\n[PART 1/2] Syncing instruments from RoboMarkets...")
            stats = await self.fetcher.sync_us_stocks()

            logger.info(f"Instrument sync complete:")
            logger.info(f"  Total from API: {stats['total_from_api']:,}")
            logger.info(f"  US stocks: {stats['unique_tickers']:,}")
            logger.info(f"  NYSE: {stats['nyse_count']:,}")
            logger.info(f"  NASDAQ: {stats['nasdaq_count']:,}")
            logger.info(f"  New: {stats['inserted']}")
            logger.info(f"  Updated: {stats['updated']}")

        except Exception as e:
            logger.error(f"Instrument sync failed: {e}")

        # Part 2: Update fundamentals data
        try:
            logger.info("\n[PART 2/2] Fetching fundamentals data...")
            fund_stats = await self.fundamentals.run_fundamentals_pipeline()

            logger.info(f"Fundamentals sync complete:")
            logger.info(f"  Total: {fund_stats['total']}")
            logger.info(f"  Successful: {fund_stats['successful']}")
            logger.info(f"  With earnings date: {fund_stats['with_earnings']}")
            logger.info(f"  With beta: {fund_stats['with_beta']}")
            logger.info(f"  With short interest: {fund_stats['with_short_interest']}")

        except Exception as e:
            logger.error(f"Fundamentals sync failed: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("WEEKLY SYNC - Complete")
        logger.info("=" * 60)

    async def _log_pipeline_run(
        self,
        pipeline_name: str,
        results: dict,
        duration_seconds: float
    ):
        """Log pipeline execution to database"""
        import json

        # Determine status
        status = 'success'
        for step_result in results.values():
            if isinstance(step_result, dict) and 'error' in step_result:
                status = 'partial_failure'
                break

        query = """
            INSERT INTO stock_pipeline_log (
                pipeline_name, execution_date, duration_seconds, results, status
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (pipeline_name, execution_date)
            DO UPDATE SET
                duration_seconds = EXCLUDED.duration_seconds,
                results = EXCLUDED.results,
                status = EXCLUDED.status,
                completed_at = NOW()
        """

        try:
            await self.db.execute(
                query,
                pipeline_name,
                datetime.now().date(),
                round(duration_seconds, 2),
                json.dumps(results, default=str),
                status
            )
        except Exception as e:
            logger.error(f"Failed to log pipeline run: {e}")

    def get_next_scheduled_task(self) -> tuple:
        """
        Get the next scheduled task and its time.

        Returns:
            Tuple of (task_name, next_time)
        """
        now = datetime.now(self.ET)
        today = now.date()
        is_weekday = now.weekday() < 5  # Mon-Fri

        candidates = []

        # Add daily scans if weekday
        if is_weekday:
            for task_name, config in self.SCHEDULE.items():
                target = datetime.combine(today, config['time'])
                target = self.ET.localize(target)

                if now >= target:
                    # Schedule for tomorrow
                    target += timedelta(days=1)
                    # Skip weekends
                    while target.weekday() >= 5:
                        target += timedelta(days=1)

                candidates.append((task_name, target))

        # Add weekly sync
        next_weekly = self.get_next_weekly_sync()
        candidates.append(('weekly_sync', next_weekly))

        # Sort by time and return the nearest
        candidates.sort(key=lambda x: x[1])
        return candidates[0]

    async def run(self):
        """Main scheduler loop with multiple daily scans"""
        await self.setup()
        self.running = True

        logger.info("Enhanced Stock Scheduler started")
        logger.info("Schedule (Mon-Fri):")
        for task_name, config in self.SCHEDULE.items():
            et_time = config['time'].strftime('%H:%M')
            # Calculate UTC time (ET is UTC-5 in winter, UTC-4 in summer)
            utc_hour = (config['time'].hour + 5) % 24  # Approximate UTC
            logger.info(f"  {et_time} ET ({utc_hour:02d}:00 UTC) - {task_name}: {config['description']}")
        logger.info(f"  Sunday {self.WEEKLY_SYNC_TIME} ET - weekly_sync: Instrument and fundamentals refresh")

        try:
            while self.running:
                now = datetime.now(self.ET)

                # Get next scheduled task
                next_task, next_time = self.get_next_scheduled_task()

                # Calculate sleep duration
                sleep_seconds = (next_time - now).total_seconds()

                logger.info(f"Next task: {next_task} at {next_time.strftime('%Y-%m-%d %H:%M %Z')}")
                logger.info(f"Sleeping for {int(sleep_seconds//3600)}h {int((sleep_seconds%3600)//60)}m")

                # Sleep until next task (check every minute for shutdown)
                while sleep_seconds > 0 and self.running:
                    await asyncio.sleep(min(60, sleep_seconds))
                    sleep_seconds -= 60

                if not self.running:
                    break

                # Execute the appropriate task
                if next_task == "pre_market":
                    await self.run_pre_market_scan()
                elif next_task == "intraday":
                    await self.run_intraday_scan()
                elif next_task == "post_market":
                    await self.run_post_market_scan()
                elif next_task == "full_pipeline":
                    await self.run_pipeline()
                elif next_task == "weekly_sync":
                    await self.weekly_sync()

        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False


async def run_once(task: str):
    """Run a specific task once"""
    scheduler = StockScheduler()
    await scheduler.setup()

    try:
        if task == "pipeline":
            await scheduler.run_pipeline()
        elif task == "sync":
            await scheduler._sync_hourly_data()
        elif task == "synthesize":
            await scheduler.synthesizer.synthesize_all_daily()
        elif task == "metrics":
            await scheduler.calculator.calculate_all_metrics()
        elif task == "smc":
            await scheduler.smc_analyzer.run_analysis_pipeline()
        elif task == "watchlist":
            await scheduler.watchlist_builder.build_watchlist()
        elif task == "signals":
            await scheduler.zlma_strategy.scan_all_stocks()
        elif task == "scanners":
            if scheduler.scanner_manager:
                await scheduler.scanner_manager.run_all_scanners()
            else:
                print("Scanner manager not available")
        elif task == "premarket":
            await scheduler.run_pre_market_scan()
        elif task == "intraday":
            await scheduler.run_intraday_scan()
        elif task == "postmarket":
            await scheduler.run_post_market_scan()
        elif task == "weekly":
            await scheduler.weekly_sync()
        elif task == "fundamentals":
            await scheduler.fundamentals.run_fundamentals_pipeline()
        else:
            print(f"Unknown task: {task}")
            print("Available: pipeline, sync, synthesize, metrics, smc, watchlist, signals, scanners, premarket, intraday, postmarket, weekly, fundamentals")
    finally:
        await scheduler.cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Stock Scanner Scheduler')
    parser.add_argument('command', nargs='?', default='run',
                       choices=['run', 'pipeline', 'sync', 'synthesize', 'metrics', 'smc',
                               'watchlist', 'signals', 'scanners', 'premarket', 'intraday',
                               'postmarket', 'weekly', 'fundamentals', 'status'],
                       help='Command to execute')
    args = parser.parse_args()

    if args.command == 'run':
        print("Starting Enhanced Stock Scanner Scheduler...")
        print("Press Ctrl+C to stop")

        scheduler = StockScheduler()

        try:
            asyncio.run(scheduler.run())
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            scheduler.stop()

    elif args.command in ['pipeline', 'sync', 'synthesize', 'metrics', 'smc', 'watchlist',
                          'signals', 'scanners', 'premarket', 'intraday', 'postmarket',
                          'weekly', 'fundamentals']:
        print(f"Running {args.command}...")
        asyncio.run(run_once(args.command))

    elif args.command == 'status':
        scheduler = StockScheduler()

        print("\nEnhanced Stock Scanner Scheduler Status")
        print("=" * 60)
        print("\nSchedule (Mon-Fri):")
        for task_name, config in scheduler.SCHEDULE.items():
            et_time = config['time'].strftime('%H:%M')
            utc_hour = (config['time'].hour + 5) % 24  # Approximate UTC
            print(f"  {et_time} ET ({utc_hour:02d}:00 UTC) - {task_name}: {config['description']}")
        print(f"  Sunday {scheduler.WEEKLY_SYNC_TIME} ET - weekly_sync: Instrument refresh")

        print(f"\nFull Pipeline stages:")
        print("  1. sync       - Fetch 1H candles from yfinance")
        print("  2. synthesize - Aggregate 1H -> Daily candles")
        print("  3. metrics    - Calculate ATR, volume, momentum")
        print("  4. smc        - Smart Money Concepts analysis")
        print("  5. watchlist  - Build tiered, scored watchlist")
        print("  6. zlma       - Run ZLMA strategy")
        print("  7. scanners   - Run all scanner strategies")

        print(f"\nNext scheduled tasks:")
        print(f"  Full pipeline: {scheduler.get_next_pipeline()}")
        print(f"  Weekly sync: {scheduler.get_next_weekly_sync()}")

        print("\nAvailable commands:")
        print("  run         - Start continuous scheduler")
        print("  pipeline    - Run full 7-stage pipeline")
        print("  premarket   - Run pre-market scan only")
        print("  intraday    - Run intraday scan only")
        print("  postmarket  - Run post-market scan only")
        print("  scanners    - Run all scanner strategies")
        print("  signals     - Run ZLMA strategy only")
        print("  sync        - Sync candle data only")
        print("  metrics     - Calculate metrics only")
        print("  smc         - Run SMC analysis only")
        print("  watchlist   - Build watchlist only")


if __name__ == '__main__':
    main()
