#!/usr/bin/env python3
"""
Stock Scanner Scheduler - Enhanced Pipeline

Runs the complete daily stock analysis pipeline:
1. Sync 1H candles from yfinance
2. Synthesize daily candles from 1H data
3. Calculate screening metrics (ATR, volume, momentum)
4. Build watchlist (tiered, scored stocks)
5. Run ZLMA strategy for signals

Schedule:
- Daily 10:30 PM ET (Mon-Fri): Full pipeline
- Weekly Sunday 1:00 AM ET: Instrument sync from RoboMarkets

Usage:
    # Run full pipeline once
    docker exec stock-scheduler python -m stock_scanner.scheduler pipeline

    # Run individual stages
    docker exec stock-scheduler python -m stock_scanner.scheduler sync
    docker exec stock-scheduler python -m stock_scanner.scheduler synthesize
    docker exec stock-scheduler python -m stock_scanner.scheduler metrics
    docker exec stock-scheduler python -m stock_scanner.scheduler watchlist
    docker exec stock-scheduler python -m stock_scanner.scheduler signals

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stock_scheduler")


class StockScheduler:
    """
    Enhanced Stock Scanner Scheduler with full analysis pipeline.

    Pipeline stages:
    1. sync       - Fetch latest 1H candles from yfinance
    2. synthesize - Aggregate 1H -> Daily candles
    3. metrics    - Calculate ATR, volume, momentum indicators
    4. smc        - Run SMC (Smart Money Concepts) analysis
    5. watchlist  - Build tiered, scored watchlist
    6. signals    - Run ZLMA strategy for trading signals

    Schedule:
    - Daily 10:30 PM ET (Mon-Fri): Full pipeline after market close
    - Weekly Sunday 1:00 AM ET: Sync instruments from RoboMarkets
    """

    # US Eastern timezone
    ET = pytz.timezone('America/New_York')

    # Schedule times (ET)
    PIPELINE_TIME = time(22, 30)  # 10:30 PM ET - Full pipeline
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
        logger.info("\n[STAGE 6/6] Running ZLMA strategy...")
        try:
            signals = await self.zlma_strategy.scan_all_stocks()
            results['signals'] = {
                'total': len(signals),
                'buy': sum(1 for s in signals if s.signal_type == 'BUY'),
                'sell': sum(1 for s in signals if s.signal_type == 'SELL')
            }
            logger.info(f"[OK] Signals: {len(signals)} total "
                       f"(BUY: {results['signals']['buy']}, SELL: {results['signals']['sell']})")
        except Exception as e:
            logger.error(f"[FAIL] Signals: {e}")
            results['signals'] = {'error': str(e)}

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

    async def run(self):
        """Main scheduler loop"""
        await self.setup()
        self.running = True

        logger.info("Enhanced Stock Scheduler started")
        logger.info(f"  Pipeline time: {self.PIPELINE_TIME} ET (Mon-Fri)")
        logger.info(f"  Weekly sync: Sunday {self.WEEKLY_SYNC_TIME} ET")

        try:
            while self.running:
                now = datetime.now(self.ET)

                # Calculate next scheduled times
                next_pipeline = self.get_next_pipeline()
                next_weekly = self.get_next_weekly_sync()

                # Find which comes first
                if next_weekly < next_pipeline:
                    next_task = "weekly_sync"
                    next_time = next_weekly
                else:
                    next_task = "pipeline"
                    next_time = next_pipeline

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

                # Execute the task
                if next_task == "pipeline":
                    await self.run_pipeline()
                else:
                    await self.weekly_sync()

        except asyncio.CancelledError:
            logger.info("Scheduler cancelled")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
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
        elif task == "weekly":
            await scheduler.weekly_sync()
        elif task == "fundamentals":
            await scheduler.fundamentals.run_fundamentals_pipeline()
        else:
            print(f"Unknown task: {task}")
            print("Available: pipeline, sync, synthesize, metrics, smc, watchlist, signals, weekly, fundamentals")
    finally:
        await scheduler.cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Stock Scanner Scheduler')
    parser.add_argument('command', nargs='?', default='run',
                       choices=['run', 'pipeline', 'sync', 'synthesize', 'metrics', 'smc',
                               'watchlist', 'signals', 'weekly', 'fundamentals', 'status'],
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

    elif args.command in ['pipeline', 'sync', 'synthesize', 'metrics', 'smc', 'watchlist', 'signals', 'weekly', 'fundamentals']:
        print(f"Running {args.command}...")
        asyncio.run(run_once(args.command))

    elif args.command == 'status':
        scheduler = StockScheduler()

        print("\nEnhanced Stock Scanner Scheduler Status")
        print("=" * 50)
        print(f"Pipeline time: {scheduler.PIPELINE_TIME} ET (Mon-Fri)")
        print(f"Weekly sync: Sunday {scheduler.WEEKLY_SYNC_TIME} ET")
        print(f"\nPipeline stages:")
        print("  1. sync       - Fetch 1H candles from yfinance")
        print("  2. synthesize - Aggregate 1H -> Daily candles")
        print("  3. metrics    - Calculate ATR, volume, momentum")
        print("  4. watchlist  - Build tiered, scored watchlist")
        print("  5. signals    - Run ZLMA strategy")
        print(f"\nNext pipeline: {scheduler.get_next_pipeline()}")
        print(f"Next weekly sync: {scheduler.get_next_weekly_sync()}")


if __name__ == '__main__':
    main()
