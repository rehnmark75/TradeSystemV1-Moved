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
from datetime import datetime, time, timedelta, date
import pytz

sys.path.insert(0, '/app')

from stock_scanner import config
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner.core.data_fetcher import StockDataFetcher
from stock_scanner.core.synthesis.daily_synthesizer import DailyCandleSynthesizer
from stock_scanner.core.metrics.calculator import MetricsCalculator
from stock_scanner.scripts.populate_rs import RSPopulator
from stock_scanner.core.metrics.rs_calculator import RSCalculator, MarketRegimeCalculator
from stock_scanner.core.screener.watchlist_builder import WatchlistBuilder
from stock_scanner.core.smc.smc_stock_analyzer import SMCStockAnalyzer
from stock_scanner.core.fundamentals.fundamentals_fetcher import FundamentalsFetcher
from stock_scanner.strategies.zlma_trend import ZeroLagMATrendStrategy
from stock_scanner.core.detection.market_hours import is_trading_day, get_last_trading_day

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

# Import pre-market service
try:
    from stock_scanner.core.news import PreMarketService, PreMarketScanResult
    PREMARKET_SERVICE_AVAILABLE = True
except ImportError:
    PREMARKET_SERVICE_AVAILABLE = False
    PreMarketService = None
    PreMarketScanResult = None

# Import broker trade sync
try:
    from stock_scanner.services.broker_trade_analyzer import BrokerTradeSync
    from stock_scanner.core.trading.robomarkets_client import RoboMarketsClient
    BROKER_SYNC_AVAILABLE = True
except ImportError:
    BROKER_SYNC_AVAILABLE = False
    BrokerTradeSync = None
    RoboMarketsClient = None

# Import deep analysis orchestrator for watchlist DAQ scoring
try:
    from stock_scanner.services.deep_analysis import DeepAnalysisOrchestrator
    DEEP_ANALYSIS_AVAILABLE = True
except ImportError:
    DEEP_ANALYSIS_AVAILABLE = False
    DeepAnalysisOrchestrator = None

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

    Schedule (Mon-Sat):
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
        'premarket_pricing': {
            'time': time(9, 0),       # 9:00 AM ET (30 min before market open)
            'type': 'premarket_finnhub',
            'description': 'Pre-market pricing & news enrichment (Finnhub)'
        },
        'broker_sync_am': {
            'time': time(9, 30),      # 9:30 AM ET (market open)
            'type': 'broker',
            'description': 'Broker trades sync (market open)'
        },
        'intraday': {
            'time': time(12, 30),     # 12:30 PM ET
            'type': 'scanner_only',
            'description': 'Intraday momentum scanner'
        },
        'broker_sync_pm': {
            'time': time(16, 0),      # 4:00 PM ET (market close)
            'type': 'broker',
            'description': 'Broker trades sync (market close)'
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
        self.rs_populator: RSPopulator = None
        self.rs_calculator: RSCalculator = None
        self.market_regime_calculator: MarketRegimeCalculator = None
        self.smc_analyzer: SMCStockAnalyzer = None
        self.fundamentals: FundamentalsFetcher = None
        self.watchlist_builder: WatchlistBuilder = None
        self.zlma_strategy: ZeroLagMATrendStrategy = None
        self.scanner_manager: ScannerManager = None
        self.performance_tracker: PerformanceTracker = None
        self.premarket_service: PreMarketService = None
        self.deep_analysis_orchestrator: DeepAnalysisOrchestrator = None
        self.running = False

    async def setup(self):
        """Initialize all pipeline components"""
        logger.info("Initializing Enhanced Stock Scheduler...")

        self.db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
        await self.db.connect()

        # Initialize all components with rate limiting
        self.fetcher = StockDataFetcher(
            db_manager=self.db,
            max_retries=5,
            request_delay=self.SYNC_REQUEST_DELAY
        )
        self.synthesizer = DailyCandleSynthesizer(db_manager=self.db)
        self.calculator = MetricsCalculator(db_manager=self.db)
        self.rs_populator = RSPopulator(db=self.db)
        self.rs_calculator = RSCalculator(db_manager=self.db)
        self.market_regime_calculator = MarketRegimeCalculator(db_manager=self.db)
        self.smc_analyzer = SMCStockAnalyzer(db_manager=self.db)
        self.fundamentals = FundamentalsFetcher(db_manager=self.db)
        self.watchlist_builder = WatchlistBuilder(db_manager=self.db)
        self.zlma_strategy = ZeroLagMATrendStrategy(db_manager=self.db)

        # Initialize deep analysis orchestrator if available
        if DEEP_ANALYSIS_AVAILABLE:
            self.deep_analysis_orchestrator = DeepAnalysisOrchestrator(self.db)
            logger.info("Deep Analysis Orchestrator initialized")

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

        # Initialize pre-market service if available
        if PREMARKET_SERVICE_AVAILABLE:
            finnhub_api_key = os.getenv('FINNHUB_API_KEY', config.FINNHUB_API_KEY if hasattr(config, 'FINNHUB_API_KEY') else '')
            if finnhub_api_key:
                self.premarket_service = PreMarketService(
                    db_manager=self.db,
                    finnhub_api_key=finnhub_api_key,
                    news_lookback_hours=16,  # Overnight news
                )
                logger.info("Pre-Market Service initialized (Finnhub)")
            else:
                logger.warning("Pre-Market Service skipped - FINNHUB_API_KEY not configured")

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

        # Skip Sunday only (Sunday=6) - Saturday runs for weekend analysis
        while target.weekday() == 6:
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

    async def run_pipeline(self, data_date: date = None):
        """
        Run the complete daily pipeline.

        Args:
            data_date: The trading day to analyze. Defaults to the last trading day.
                       On Saturday, this will be Friday. On Monday, this will be the
                       previous Friday (if market was closed over weekend).

        Stages:
        1. Sync 1H candles
        2. Synthesize daily candles
        3. Calculate metrics
        4. Build watchlist
        5. Run ZLMA strategy

        Signal Deduplication:
        - All signals use data_date (not scan date) for signal_timestamp
        - This ensures Saturday runs and Monday runs don't create duplicates
        - The unique constraint on (ticker, scanner_name, signal_date) prevents duplicates
        """
        # Determine the data date (trading day being analyzed)
        if data_date is None:
            now = datetime.now(self.ET)
            data_date = get_last_trading_day(now).date() if hasattr(get_last_trading_day(now), 'date') else get_last_trading_day(now)

        logger.info("=" * 80)
        logger.info(" DAILY PIPELINE - Starting")
        logger.info("=" * 80)
        logger.info(f" Data Date: {data_date} (trading day being analyzed)")

        pipeline_start = datetime.now()
        results = {'data_date': str(data_date)}

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

        # === STAGE 2b: Synthesize Weekly Candles (for MTF analysis) ===
        logger.info("\n[STAGE 2b] Synthesizing weekly candles...")
        try:
            weekly_stats = await self.synthesizer.synthesize_all_weekly(incremental=True)
            results['weekly_synthesis'] = weekly_stats
            logger.info(f"[OK] Weekly Synthesis: {weekly_stats['total_weekly_candles']:,} weekly candles")
        except Exception as e:
            logger.error(f"[FAIL] Weekly Synthesis: {e}")
            results['weekly_synthesis'] = {'error': str(e)}

        # === STAGE 3: Calculate Metrics ===
        logger.info("\n[STAGE 3/11] Calculating screening metrics...")
        try:
            metrics_stats = await self.calculator.calculate_all_metrics(calculation_date=data_date)
            results['metrics'] = metrics_stats
            logger.info(f"[OK] Metrics: {metrics_stats['successful']} stocks")
        except Exception as e:
            logger.error(f"[FAIL] Metrics: {e}")
            results['metrics'] = {'error': str(e)}

        # === STAGE 4: Relative Strength (RS) Calculation ===
        logger.info("\n[STAGE 4/13] Calculating Relative Strength vs SPY...")
        try:
            rs_stats = await self.rs_populator.populate_rs(calc_date=data_date)
            results['rs'] = rs_stats
            if 'error' in rs_stats:
                logger.warning(f"[WARN] RS: {rs_stats['error']}")
            else:
                logger.info(f"[OK] RS: {rs_stats['updated']} stocks updated")
                logger.info(f"     Elite (90+): {rs_stats['distribution']['elite_90+']}, "
                           f"Strong (70-89): {rs_stats['distribution']['strong_70-89']}, "
                           f"Weak (<40): {rs_stats['distribution']['weak_0-39']}")
        except Exception as e:
            logger.error(f"[FAIL] RS: {e}")
            results['rs'] = {'error': str(e)}

        # === STAGE 4b: Market Regime Calculation ===
        logger.info("\n[STAGE 4b/13] Calculating Market Regime...")
        try:
            regime_stats = await self.market_regime_calculator.calculate_market_regime()
            results['market_regime'] = regime_stats
            if 'error' in regime_stats:
                logger.warning(f"[WARN] Market Regime: {regime_stats['error']}")
            else:
                logger.info(f"[OK] Market Regime: {regime_stats.get('regime', 'unknown')}")
                if 'spy_data' in regime_stats:
                    logger.info(f"     SPY vs SMA200: {regime_stats['spy_data'].get('vs_sma200_pct', 0):.1f}%")
                if 'breadth' in regime_stats:
                    logger.info(f"     % Above SMA200: {regime_stats['breadth'].get('pct_above_sma200', 0):.1f}%")
        except Exception as e:
            logger.error(f"[FAIL] Market Regime: {e}")
            results['market_regime'] = {'error': str(e)}

        # === STAGE 4c: Sector RS Calculation ===
        logger.info("\n[STAGE 4c/13] Calculating Sector RS...")
        try:
            sector_stats = await self.rs_calculator.calculate_sector_rs()
            results['sector_rs'] = sector_stats
            if 'error' in sector_stats:
                logger.warning(f"[WARN] Sector RS: {sector_stats['error']}")
            else:
                logger.info(f"[OK] Sector RS: {sector_stats.get('sectors_processed', 0)} sectors processed")
        except Exception as e:
            logger.error(f"[FAIL] Sector RS: {e}")
            results['sector_rs'] = {'error': str(e)}

        # === STAGE 5: SMC Analysis ===
        logger.info("\n[STAGE 5/13] Running SMC analysis...")
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

        # === STAGE 6: Build Watchlist ===
        logger.info("\n[STAGE 6/11] Building watchlist...")
        try:
            watchlist_stats = await self.watchlist_builder.build_watchlist(calculation_date=data_date)
            results['watchlist'] = watchlist_stats
            logger.info(f"[OK] Watchlist: {watchlist_stats.get('passed_filters', 0)} stocks")
            logger.info(f"     Tier 1: {watchlist_stats['tier_1']}, "
                       f"Tier 2: {watchlist_stats['tier_2']}, "
                       f"Tier 3: {watchlist_stats['tier_3']}, "
                       f"Tier 4: {watchlist_stats['tier_4']}")
        except Exception as e:
            logger.error(f"[FAIL] Watchlist: {e}")
            results['watchlist'] = {'error': str(e)}

        # === STAGE 6b: Watchlist DAQ Scoring ===
        logger.info("\n[STAGE 6b/11] Running watchlist DAQ scoring...")
        try:
            if self.deep_analysis_orchestrator and DEEP_ANALYSIS_AVAILABLE:
                # Run DAQ analysis on ALL tier 1-2 watchlist stocks (no limit)
                # Analysis averages ~600ms/stock with 5 concurrent = ~2min for 800 stocks
                daq_results = await self.deep_analysis_orchestrator.auto_analyze_watchlist(
                    max_tier=2,
                    calculation_date=data_date,
                    max_tickers=2000,  # Effectively no limit for tier 1-2 (~600-800 stocks)
                    skip_analyzed=True  # Don't re-analyze stocks that already have DAQ
                )
                successful = [r for r in daq_results if r.daq_score is not None]
                avg_daq = sum(r.daq_score for r in successful) / max(1, len(successful)) if successful else 0
                results['watchlist_daq'] = {
                    'analyzed': len(daq_results),
                    'successful': len(successful),
                    'avg_daq': round(avg_daq, 1)
                }
                logger.info(f"[OK] Watchlist DAQ: {len(successful)}/{len(daq_results)} stocks scored (avg: {avg_daq:.1f})")
            else:
                logger.warning("[SKIP] Deep Analysis Orchestrator not available")
                results['watchlist_daq'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Watchlist DAQ: {e}")
            results['watchlist_daq'] = {'error': str(e)}

        # === STAGE 6c: ZLMA Strategy Signals ===
        logger.info("\n[STAGE 6c/11] Running ZLMA strategy...")
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

        # === STAGE 7: All Scanner Strategies + Deep Analysis ===
        logger.info("\n[STAGE 7/10] Running all scanner strategies + deep analysis...")
        try:
            if self.scanner_manager:
                scanner_signals = await self.scanner_manager.run_all_scanners(calculation_date=data_date)
                scan_stats = self.scanner_manager.get_scan_stats()
                results['scanner_signals'] = {
                    'total': len(scanner_signals),
                    'by_scanner': scan_stats.get('signals_by_scanner', {}),
                    'by_tier': scan_stats.get('signals_by_tier', {}),
                    'high_quality': sum(1 for s in scanner_signals if s.is_high_quality)
                }
                logger.info(f"[OK] Scanner Signals: {len(scanner_signals)} total "
                           f"(A/A+: {results['scanner_signals']['high_quality']})")

                # Log deep analysis results (automatically run inside run_all_scanners)
                deep_stats = scan_stats.get('deep_analysis', {})
                if deep_stats and not deep_stats.get('error'):
                    results['deep_analysis'] = deep_stats
                    logger.info(f"[OK] Deep Analysis: {deep_stats.get('successful', 0)}/{deep_stats.get('attempted', 0)} "
                               f"signals analyzed (avg DAQ: {deep_stats.get('avg_daq_score', 0):.1f})")
                elif deep_stats.get('error'):
                    results['deep_analysis'] = deep_stats
                    logger.warning(f"[WARN] Deep Analysis error: {deep_stats.get('error')}")
                else:
                    results['deep_analysis'] = {'skipped': True, 'reason': 'no A+/A signals'}
            else:
                logger.warning("[SKIP] Scanner Manager not available")
                results['scanner_signals'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Scanner Signals: {e}")
            results['scanner_signals'] = {'error': str(e)}

        # === STAGE 8: Watchlist Scanner (5 predefined screens) ===
        logger.info("\n[STAGE 8/10] Running watchlist scanner (5 predefined screens)...")
        try:
            if self.scanner_manager:
                watchlist_results = await self.scanner_manager.run_watchlist_scanner(calculation_date=data_date)
                results['watchlist_scanner'] = watchlist_results
                total_wl = sum(watchlist_results.values())
                logger.info(f"[OK] Watchlist Scanner: {total_wl} total matches")
                for wl_name, count in watchlist_results.items():
                    logger.info(f"     {wl_name}: {count}")
            else:
                logger.warning("[SKIP] Scanner Manager not available for watchlist scan")
                results['watchlist_scanner'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Watchlist Scanner: {e}")
            results['watchlist_scanner'] = {'error': str(e)}

        # === STAGE 8b: Technical Watchlist DAQ Scoring ===
        logger.info("\n[STAGE 8b/11] Running technical watchlist DAQ scoring...")
        try:
            if self.deep_analysis_orchestrator and DEEP_ANALYSIS_AVAILABLE:
                # Run DAQ analysis on ALL technical watchlist stocks (no limit)
                tech_wl_results = await self.deep_analysis_orchestrator.auto_analyze_technical_watchlist(
                    watchlist_name=None,  # Analyze all watchlist types
                    scan_date=data_date,  # Pass date object, not string
                    max_tickers=2000,  # Effectively no limit
                    skip_analyzed=True  # Don't re-analyze stocks that already have DAQ
                )
                successful = [r for r in tech_wl_results if r.daq_score is not None]
                avg_daq = sum(r.daq_score for r in successful) / max(1, len(successful)) if successful else 0
                results['technical_watchlist_daq'] = {
                    'analyzed': len(tech_wl_results),
                    'successful': len(successful),
                    'avg_daq': round(avg_daq, 1)
                }
                logger.info(f"[OK] Technical Watchlist DAQ: {len(successful)}/{len(tech_wl_results)} stocks scored (avg: {avg_daq:.1f})")
            else:
                logger.warning("[SKIP] Deep Analysis Orchestrator not available for technical watchlist")
                results['technical_watchlist_daq'] = {'skipped': True}
        except Exception as e:
            logger.error(f"[FAIL] Technical Watchlist DAQ: {e}")
            results['technical_watchlist_daq'] = {'error': str(e)}

        # === STAGE 9: Performance Tracking ===
        logger.info("\n[STAGE 9/11] Updating signal performance...")
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

        # === STAGE 10: Claude AI Analysis ===
        logger.info("\n[STAGE 10/11] Running Claude AI analysis on top signals...")
        try:
            if not config.CLAUDE_ANALYSIS_ENABLED:
                logger.info("[SKIP] Claude Analysis disabled in config")
                results['claude_analysis'] = {'skipped': True, 'reason': 'disabled'}
            elif self.scanner_manager:
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
    # yfinance rate limit is ~2000 requests/hour, so ~33/min = 1.8s between requests
    SYNC_CONCURRENCY = 2  # Max concurrent requests (conservative)
    SYNC_BATCH_SIZE = 50  # Smaller batches for better rate limit handling
    SYNC_BATCH_DELAY = 5.0  # Longer delay between batches (was 2.0)
    SYNC_REQUEST_DELAY = 0.5  # Delay between individual requests (passed to fetcher)

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
            still_failed = []

            for ticker in failed_tickers[:200]:  # Limit retries
                try:
                    await asyncio.sleep(1.0)  # 1s delay between retries (was 0.5)
                    count = await self.fetcher.fetch_historical_data(
                        ticker, days=5, interval='1h'
                    )
                    if count > 0:
                        retry_success += 1
                        successful += 1
                        failed -= 1
                        total_candles += count
                    else:
                        still_failed.append(ticker)
                except Exception:
                    still_failed.append(ticker)

            if retry_success > 0:
                logger.info(f"  Retry recovered {retry_success} tickers")

            # Mark persistently failed tickers as potentially delisted
            if still_failed:
                await self._mark_potentially_delisted(still_failed)

        return {
            'total_tickers': len(tickers),
            'successful': successful,
            'failed': failed,
            'total_candles': total_candles
        }

    async def _mark_potentially_delisted(self, tickers: list):
        """
        Track tickers that consistently fail to sync.
        After 3 consecutive failures, mark them as inactive.
        """
        if not tickers:
            return

        for ticker in tickers:
            # Increment failure count
            query = """
                INSERT INTO stock_sync_failures (ticker, failure_count, last_failure)
                VALUES ($1, 1, NOW())
                ON CONFLICT (ticker) DO UPDATE SET
                    failure_count = stock_sync_failures.failure_count + 1,
                    last_failure = NOW()
                RETURNING failure_count
            """
            try:
                result = await self.db.fetchval(query, ticker)
                if result and result >= 3:
                    # Mark as inactive after 3 failures
                    deactivate_query = """
                        UPDATE stock_instruments
                        SET is_active = FALSE, is_tradeable = FALSE, updated_at = NOW()
                        WHERE ticker = $1
                    """
                    await self.db.execute(deactivate_query, ticker)
                    logger.warning(f"Marked {ticker} as inactive after {result} sync failures")
            except Exception as e:
                # Table might not exist, create it
                if 'does not exist' in str(e):
                    create_query = """
                        CREATE TABLE IF NOT EXISTS stock_sync_failures (
                            ticker VARCHAR(20) PRIMARY KEY,
                            failure_count INT DEFAULT 0,
                            last_failure TIMESTAMP DEFAULT NOW()
                        )
                    """
                    try:
                        await self.db.execute(create_query)
                        logger.info("Created stock_sync_failures table")
                    except Exception:
                        pass

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

    async def run_premarket_pricing_scan(self):
        """
        Pre-market pricing scan (9:00 AM ET) - 30 min before market open.

        Uses Finnhub API to:
        - Verify market is in pre-market session
        - Fetch pre-market quotes for watchlist stocks
        - Detect and classify gaps (vs previous close)
        - Fetch overnight news for gapping stocks
        - Generate pre-market signals with news context
        - Store results for review before market open

        This scan runs at 9:00 AM ET, giving 30 minutes to:
        1. Review gap signals
        2. Analyze news catalysts
        3. Plan entries for market open
        """
        logger.info("=" * 60)
        logger.info("PRE-MARKET PRICING SCAN - Finnhub quotes & news (9:00 AM ET)")
        logger.info("=" * 60)

        start_time = datetime.now()
        results = {}

        if not self.premarket_service:
            logger.warning("[SKIP] Pre-market service not available - check FINNHUB_API_KEY")
            return {'skipped': True, 'reason': 'premarket_service not initialized'}

        try:
            # Run the pre-market scan via PreMarketService
            scan_result = await self.premarket_service.run_premarket_scan(
                force_run=False  # Only run if actually in pre-market session
            )

            results = {
                'is_premarket': scan_result.is_pre_market,
                'market_status': scan_result.market_status.get('session', 'unknown'),
                'quotes_fetched': scan_result.quotes_fetched,
                'quotes_with_gaps': scan_result.quotes_with_gaps,
                'total_signals': len(scan_result.signals),
                'gap_up_signals': scan_result.gap_up_signals,
                'gap_down_signals': scan_result.gap_down_signals,
                'news_catalyst_signals': scan_result.news_catalyst_signals,
                'errors': scan_result.errors,
            }

            if scan_result.is_pre_market:
                logger.info(f"[OK] Pre-market pricing scan complete:")
                logger.info(f"     Quotes fetched: {scan_result.quotes_fetched}")
                logger.info(f"     Stocks with gaps (>1%): {scan_result.quotes_with_gaps}")
                logger.info(f"     Signals generated: {len(scan_result.signals)}")
                logger.info(f"       - Gap UP: {scan_result.gap_up_signals}")
                logger.info(f"       - Gap DOWN: {scan_result.gap_down_signals}")
                logger.info(f"       - News catalyst: {scan_result.news_catalyst_signals}")

                # Log top signals
                if scan_result.signals:
                    logger.info("\n     Top Pre-Market Signals:")
                    for signal in scan_result.signals[:5]:
                        direction = "↑" if signal.direction == "BUY" else "↓"
                        news_tag = f" [NEWS]" if signal.news_count > 0 else ""
                        logger.info(
                            f"       {direction} {signal.symbol}: "
                            f"{signal.quote.gap_percent:+.1f}% gap, "
                            f"{signal.strength} {signal.signal_type}{news_tag}"
                        )
            else:
                logger.info("[INFO] Not in pre-market session - scan skipped")
                logger.info(f"       Market status: {scan_result.market_status}")

        except Exception as e:
            logger.error(f"Pre-market pricing scan error: {e}")
            results['error'] = str(e)
            import traceback
            traceback.print_exc()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nPre-market pricing scan complete in {elapsed:.1f}s")

        await self._log_pipeline_run('premarket_pricing_scan', results, elapsed)
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

    async def run_broker_sync(self, days: int = 30):
        """
        Sync broker trades from RoboMarkets API to local database.

        Runs at:
        - 9:30 AM ET (market open) - catch overnight trades
        - 4:00 PM ET (market close) - sync end-of-day trades

        Syncs:
        - Open positions (current state)
        - Closed trades from last N days
        """
        logger.info("=" * 60)
        logger.info("BROKER SYNC - Syncing trades from RoboMarkets")
        logger.info("=" * 60)

        start_time = datetime.now()
        results = {}

        if not BROKER_SYNC_AVAILABLE:
            logger.warning("[SKIP] Broker sync not available - missing dependencies")
            return {'skipped': True, 'reason': 'dependencies not available'}

        try:
            # Check for API credentials
            api_key = os.getenv('ROBOMARKETS_API_KEY', config.ROBOMARKETS_API_KEY if hasattr(config, 'ROBOMARKETS_API_KEY') else '')
            account_id = os.getenv('ROBOMARKETS_ACCOUNT_ID', config.ROBOMARKETS_ACCOUNT_ID if hasattr(config, 'ROBOMARKETS_ACCOUNT_ID') else '')

            if not api_key or not account_id:
                logger.warning("[SKIP] Broker sync - missing API credentials")
                return {'skipped': True, 'reason': 'missing API credentials'}

            # Initialize broker client and sync service
            client = RoboMarketsClient(api_key=api_key, account_id=account_id)
            broker_sync = BrokerTradeSync(db_manager=self.db, robomarkets_client=client)

            async with client:
                # Run full sync
                sync_result = await broker_sync.sync_all(days=days)

                results = {
                    'positions_fetched': sync_result['positions']['total'],
                    'positions_inserted': sync_result['positions']['inserted'],
                    'positions_updated': sync_result['positions']['updated'],
                    'trades_fetched': sync_result['trades']['total'],
                    'trades_inserted': sync_result['trades']['inserted'],
                    'trades_updated': sync_result['trades']['updated'],
                    'total_synced': sync_result['total_inserted'] + sync_result['total_updated']
                }

                logger.info(f"[OK] Broker sync complete:")
                logger.info(f"     Positions: {results['positions_fetched']} fetched, "
                           f"{results['positions_inserted']} new, {results['positions_updated']} updated")
                logger.info(f"     Trades: {results['trades_fetched']} fetched, "
                           f"{results['trades_inserted']} new, {results['trades_updated']} updated")

        except Exception as e:
            logger.error(f"Broker sync error: {e}")
            results['error'] = str(e)

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nBroker sync complete in {elapsed:.1f}s")

        await self._log_pipeline_run('broker_sync', results, elapsed)
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
        import math

        def sanitize_for_json(obj):
            """
            Recursively sanitize an object for JSON serialization.
            Replaces NaN, Infinity, -Infinity with None.
            """
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(v) for v in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            else:
                return obj

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
            # Sanitize results to remove NaN/Infinity values before JSON serialization
            sanitized_results = sanitize_for_json(results)
            await self.db.execute(
                query,
                pipeline_name,
                datetime.now().date(),
                round(duration_seconds, 2),
                json.dumps(sanitized_results, default=str),
                status
            )
        except Exception as e:
            logger.error(f"Failed to log pipeline run: {e}")

    def _should_run_saturday_analysis(self) -> bool:
        """
        Check if we should run Saturday analysis.

        Saturday runs analyze Friday's data to give you time to review signals
        before Monday. This is useful for weekend analysis/preparation.

        Returns:
            True if today is Saturday and we haven't run analysis for Friday yet
        """
        now = datetime.now(self.ET)

        # Only run on Saturday
        if now.weekday() != 5:  # 5 = Saturday
            return False

        # Get last trading day (should be Friday)
        last_trading = get_last_trading_day(now)
        if hasattr(last_trading, 'date'):
            last_trading = last_trading.date()

        return True

    async def check_and_run_missed_tasks(self):
        """
        Check if any daily tasks were missed today and run them if needed.

        This handles the case where the scheduler starts after the scheduled time
        (e.g., after a weekend or restart). If the task hasn't run today, run it now.

        Also handles Saturday analysis runs for Friday data.
        """
        now = datetime.now(self.ET)
        today = now.date()

        # Special handling for Saturday - run analysis for Friday data
        if now.weekday() == 5:  # Saturday
            last_trading = get_last_trading_day(now)
            if hasattr(last_trading, 'date'):
                data_date = last_trading.date()
            else:
                data_date = last_trading

            # Check if we already ran analysis for the last trading day
            query = """
                SELECT COUNT(*) FROM stock_pipeline_log
                WHERE pipeline_name = 'daily_pipeline'
                AND results->>'data_date' = $1
            """
            try:
                count = await self.db.fetchval(query, str(data_date))

                if count == 0:
                    pipeline_time = self.SCHEDULE['full_pipeline']['time']
                    scheduled = datetime.combine(today, pipeline_time)
                    scheduled = self.ET.localize(scheduled)

                    if now > scheduled:
                        logger.info("=" * 60)
                        logger.info(f"SATURDAY ANALYSIS - Running pipeline for {data_date}")
                        logger.info("=" * 60)
                        await self.run_pipeline(data_date=data_date)
                        return True
                else:
                    logger.info(f"Saturday: Already ran analysis for {data_date}")
            except Exception as e:
                logger.warning(f"Could not check Saturday pipeline status: {e}")
            return False

        # Skip Sunday
        if now.weekday() == 6:  # Sunday
            logger.info("Sunday - skipping missed task check")
            return False

        # Check if today is a trading day (weekday + not a holiday)
        if not is_trading_day(now):
            logger.info("Market holiday - skipping missed task check")
            return False

        # Check if full_pipeline has run today
        query = """
            SELECT COUNT(*) FROM stock_pipeline_log
            WHERE pipeline_name = 'daily_pipeline'
            AND execution_date = $1
        """
        try:
            count = await self.db.fetchval(query, today)

            if count == 0:
                # Check if we're past the scheduled time for full_pipeline
                pipeline_time = self.SCHEDULE['full_pipeline']['time']
                scheduled = datetime.combine(today, pipeline_time)
                scheduled = self.ET.localize(scheduled)

                if now > scheduled:
                    logger.info("=" * 60)
                    logger.info("MISSED TASK DETECTED - Running full_pipeline now")
                    logger.info(f"Scheduled time was: {scheduled.strftime('%H:%M %Z')}")
                    logger.info("=" * 60)
                    await self.run_pipeline()
                    return True
        except Exception as e:
            logger.warning(f"Could not check for missed tasks: {e}")

        return False

    def _is_valid_pipeline_day(self, dt: datetime) -> bool:
        """
        Check if a given day is valid for running the pipeline.

        Valid days are:
        - Trading days (Mon-Fri, not holidays)
        - Saturday (for weekend analysis of Friday data)

        Sunday is NOT valid for pipeline runs.
        """
        weekday = dt.weekday()

        # Saturday is valid for weekend analysis
        if weekday == 5:
            return True

        # Sunday is never valid
        if weekday == 6:
            return False

        # Check if it's a trading day (Mon-Fri, not holiday)
        return is_trading_day(dt)

    def get_next_scheduled_task(self) -> tuple:
        """
        Get the next scheduled task and its time.

        Returns:
            Tuple of (task_name, next_time)

        Note: Runs on trading days AND Saturdays (for weekend analysis).
              Skips Sundays and market holidays.
        """
        now = datetime.now(self.ET)
        today = now.date()

        candidates = []

        # Always add daily scans - they run on trading days AND Saturdays
        for task_name, config in self.SCHEDULE.items():
            target = datetime.combine(today, config['time'])
            target = self.ET.localize(target)

            # If we've passed the time or today is not a valid day, find next valid slot
            if now >= target or not self._is_valid_pipeline_day(now):
                # Start from tomorrow
                if now >= target:
                    target += timedelta(days=1)
                # Skip invalid days (Sundays, holidays)
                max_attempts = 10  # Safety limit
                attempts = 0
                while not self._is_valid_pipeline_day(target) and attempts < max_attempts:
                    target += timedelta(days=1)
                    attempts += 1

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
        logger.info("Schedule (Mon-Sat):")
        for task_name, config in self.SCHEDULE.items():
            et_time = config['time'].strftime('%H:%M')
            # Calculate UTC time (ET is UTC-5 in winter, UTC-4 in summer)
            utc_hour = (config['time'].hour + 5) % 24  # Approximate UTC
            logger.info(f"  {et_time} ET ({utc_hour:02d}:00 UTC) - {task_name}: {config['description']}")
        logger.info(f"  Sunday {self.WEEKLY_SYNC_TIME} ET - weekly_sync: Instrument and fundamentals refresh")

        # Check for missed tasks on startup (e.g., after weekend or restart)
        logger.info("Checking for missed tasks...")
        await self.check_and_run_missed_tasks()

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
                elif next_task == "premarket_pricing":
                    await self.run_premarket_pricing_scan()
                elif next_task == "intraday":
                    await self.run_intraday_scan()
                elif next_task == "post_market":
                    await self.run_post_market_scan()
                elif next_task == "full_pipeline":
                    await self.run_pipeline()
                elif next_task == "weekly_sync":
                    await self.weekly_sync()
                elif next_task in ("broker_sync_am", "broker_sync_pm"):
                    await self.run_broker_sync()

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
        elif task == "rs":
            await scheduler.rs_populator.populate_rs()
        elif task == "market_regime":
            await scheduler.market_regime_calculator.calculate_market_regime()
        elif task == "sector_rs":
            await scheduler.rs_calculator.calculate_sector_rs()
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
        elif task == "premarketpricing":
            await scheduler.run_premarket_pricing_scan()
        elif task == "intraday":
            await scheduler.run_intraday_scan()
        elif task == "postmarket":
            await scheduler.run_post_market_scan()
        elif task == "weekly":
            await scheduler.weekly_sync()
        elif task == "fundamentals":
            await scheduler.fundamentals.run_fundamentals_pipeline()
        elif task == "brokersync":
            await scheduler.run_broker_sync()
        elif task == "techwldaq":
            if scheduler.deep_analysis_orchestrator:
                results = await scheduler.deep_analysis_orchestrator.auto_analyze_technical_watchlist(
                    watchlist_name=None,
                    scan_date=None,
                    max_tickers=2000,  # Effectively no limit
                    skip_analyzed=True
                )
                successful = [r for r in results if r.daq_score is not None]
                avg_daq = sum(r.daq_score for r in successful) / max(1, len(successful)) if successful else 0
                print(f"Analyzed {len(successful)}/{len(results)} technical watchlist stocks (avg DAQ: {avg_daq:.1f})")
            else:
                print("Deep Analysis Orchestrator not available")
        else:
            print(f"Unknown task: {task}")
            print("Available: pipeline, sync, synthesize, metrics, rs, smc, watchlist, signals, scanners, premarket, premarketpricing, intraday, postmarket, weekly, fundamentals, brokersync, techwldaq")
    finally:
        await scheduler.cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced Stock Scanner Scheduler')
    parser.add_argument('command', nargs='?', default='run',
                       choices=['run', 'pipeline', 'sync', 'synthesize', 'metrics', 'rs', 'smc',
                               'watchlist', 'signals', 'scanners', 'premarket', 'premarketpricing',
                               'intraday', 'postmarket', 'weekly', 'fundamentals', 'brokersync',
                               'techwldaq', 'status'],
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

    elif args.command in ['pipeline', 'sync', 'synthesize', 'metrics', 'rs', 'smc', 'watchlist',
                          'signals', 'scanners', 'premarket', 'premarketpricing', 'intraday',
                          'postmarket', 'weekly', 'fundamentals', 'brokersync', 'techwldaq']:
        print(f"Running {args.command}...")
        asyncio.run(run_once(args.command))

    elif args.command == 'status':
        scheduler = StockScheduler()

        print("\nEnhanced Stock Scanner Scheduler Status")
        print("=" * 60)
        print("\nSchedule (Mon-Sat):")
        for task_name, config in scheduler.SCHEDULE.items():
            et_time = config['time'].strftime('%H:%M')
            utc_hour = (config['time'].hour + 5) % 24  # Approximate UTC
            print(f"  {et_time} ET ({utc_hour:02d}:00 UTC) - {task_name}: {config['description']}")
        print(f"  Sunday {scheduler.WEEKLY_SYNC_TIME} ET - weekly_sync: Instrument refresh")

        print(f"\nFull Pipeline stages:")
        print("  1. sync       - Fetch 1H candles from yfinance")
        print("  2. synthesize - Aggregate 1H -> Daily candles")
        print("  3. metrics    - Calculate ATR, volume, momentum")
        print("  4. rs         - Calculate Relative Strength vs SPY")
        print("  5. smc        - Smart Money Concepts analysis")
        print("  6. watchlist  - Build tiered, scored watchlist")
        print("  6b. watchlist_daq - DAQ analysis for main watchlist")
        print("  7. zlma       - Run ZLMA strategy")
        print("  8. scanners   - Run all scanner strategies + DAQ")
        print("  8b. tech_wl_daq - DAQ analysis for technical watchlists")
        print("  9. performance - Update signal performance tracking")
        print("  10. claude    - Run Claude AI analysis on top signals")

        print(f"\nNext scheduled tasks:")
        print(f"  Full pipeline: {scheduler.get_next_pipeline()}")
        print(f"  Weekly sync: {scheduler.get_next_weekly_sync()}")

        print("\nAvailable commands:")
        print("  run              - Start continuous scheduler")
        print("  pipeline         - Run full 11-stage pipeline")
        print("  premarket        - Run pre-market scanner scan only")
        print("  premarketpricing - Run Finnhub pre-market quotes & news (9:00 AM ET scan)")
        print("  intraday         - Run intraday scan only")
        print("  postmarket       - Run post-market scan only")
        print("  scanners         - Run all scanner strategies")
        print("  signals          - Run ZLMA strategy only")
        print("  sync             - Sync candle data only")
        print("  metrics          - Calculate metrics only")
        print("  rs               - Calculate Relative Strength vs SPY")
        print("  smc              - Run SMC analysis only")
        print("  watchlist        - Build watchlist only")
        print("  brokersync       - Sync broker trades to database")
        print("  techwldaq        - Run DAQ analysis for technical watchlists")


if __name__ == '__main__':
    main()
