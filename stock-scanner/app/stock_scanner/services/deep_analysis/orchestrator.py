"""
Deep Analysis Orchestrator

Central coordinator for deep signal analysis. Manages the pipeline:
1. Technical Deep Analysis
2. Fundamental Deep Analysis
3. Contextual Analysis
4. DAQ Score Calculation
5. Database Storage
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .models import (
    DeepAnalysisResult,
    DeepAnalysisConfig,
    DAQGrade,
    TechnicalDeepResult,
    FundamentalDeepResult,
    ContextualDeepResult,
    MTFAnalysisResult,
    VolumeAnalysisResult,
    SMCAnalysisResult,
    QualityScreenResult,
    CatalystAnalysisResult,
    InstitutionalAnalysisResult,
    NewsSentimentResult,
    MarketRegimeResult,
    SectorRotationResult,
    TrendDirection,
    MarketRegime,
)
from .technical_analyzer import TechnicalDeepAnalyzer
from .fundamental_analyzer import FundamentalDeepAnalyzer
from .contextual_analyzer import ContextualDeepAnalyzer

logger = logging.getLogger(__name__)


class DeepAnalysisOrchestrator:
    """
    Central coordinator for deep signal analysis.

    Trigger Conditions:
    - Signal quality_tier in ['A+', 'A']
    - OR manual request via CLI

    Analysis Pipeline:
    1. Technical Deep Analysis (45% weight)
       - Multi-TF Confluence (20%)
       - Volume Profile (10%)
       - SMC Enhancement (15%)
    2. Fundamental Deep Analysis (25% weight)
       - Financial Quality (15%)
       - Catalyst Timing (10%)
    3. Contextual Analysis (30% weight)
       - News Sentiment (10%)
       - Market Regime (10%)
       - Sector Rotation (10%)
    4. Calculate DAQ Score
    5. Store in database

    Usage:
        orchestrator = DeepAnalysisOrchestrator(db_manager)
        result = await orchestrator.analyze_signal(signal_id)
    """

    def __init__(
        self,
        db_manager,
        config: Optional[DeepAnalysisConfig] = None
    ):
        """
        Initialize the deep analysis orchestrator.

        Args:
            db_manager: Database manager instance
            config: Deep analysis configuration
        """
        self.db = db_manager
        self.config = config or DeepAnalysisConfig()

        # Initialize analyzers
        self.technical_analyzer = TechnicalDeepAnalyzer(db_manager, self.config)
        self.fundamental_analyzer = FundamentalDeepAnalyzer(db_manager, self.config)
        self.contextual_analyzer = ContextualDeepAnalyzer(db_manager, self.config)

        # Stats tracking
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_duration_ms': 0,
            'avg_daq_score': 0,
        }

    async def analyze_signal(
        self,
        signal_id: int,
        save_to_db: bool = True,
        force_reanalyze: bool = False
    ) -> Optional[DeepAnalysisResult]:
        """
        Run full deep analysis pipeline for a signal.

        Args:
            signal_id: Signal ID from stock_scanner_signals table
            save_to_db: Whether to save results to database

        Returns:
            DeepAnalysisResult or None if analysis fails
        """
        start_time = time.time()

        # Fetch signal data
        signal = await self._fetch_signal(signal_id)
        if not signal:
            logger.warning(f"Signal {signal_id} not found")
            return None

        ticker = signal.get('ticker')
        logger.info(f"Starting deep analysis for {ticker} (signal_id={signal_id})")

        # Check cooldown (don't re-analyze same ticker within cooldown period)
        if not force_reanalyze and not await self._check_cooldown(ticker):
            logger.info(f"Skipping {ticker} - within cooldown period")
            return None

        try:
            # Get sector for contextual analysis
            sector = await self._get_ticker_sector(ticker)

            # Run all analyzers in parallel
            technical_result, fundamental_result, contextual_result = await asyncio.gather(
                self.technical_analyzer.analyze(ticker, signal),
                self.fundamental_analyzer.analyze(ticker, signal),
                self.contextual_analyzer.analyze(ticker, signal, sector),
            )

            # Create result
            result = DeepAnalysisResult(
                signal_id=signal_id,
                ticker=ticker,
                analysis_timestamp=datetime.now(),
                technical=technical_result,
                fundamental=fundamental_result,
                contextual=contextual_result,
                components_analyzed=['mtf', 'volume', 'smc', 'quality', 'catalyst', 'news', 'regime', 'sector'],
            )

            # Calculate composite DAQ score
            result.calculate_daq_score()

            # Record duration
            result.analysis_duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Deep analysis complete for {ticker}: "
                f"DAQ={result.daq_score} ({result.daq_grade.value}) "
                f"in {result.analysis_duration_ms}ms"
            )

            # Save to database
            if save_to_db:
                await self._save_analysis(result)

            # Update stats
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"Deep analysis failed for {ticker}: {e}")
            self.stats['failed_analyses'] += 1
            return None

    async def analyze_signals_batch(
        self,
        signal_ids: List[int],
        save_to_db: bool = True,
        max_concurrent: int = 5,
        force_reanalyze: bool = False
    ) -> List[DeepAnalysisResult]:
        """
        Run deep analysis for multiple signals with concurrency control.

        Args:
            signal_ids: List of signal IDs to analyze
            save_to_db: Whether to save results
            max_concurrent: Max concurrent analyses

        Returns:
            List of DeepAnalysisResult objects
        """
        logger.info(f"Starting batch deep analysis for {len(signal_ids)} signals")

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(signal_id: int) -> Optional[DeepAnalysisResult]:
            async with semaphore:
                return await self.analyze_signal(
                    signal_id,
                    save_to_db=save_to_db,
                    force_reanalyze=force_reanalyze
                )

        tasks = [analyze_with_limit(sid) for sid in signal_ids]
        all_results = await asyncio.gather(*tasks)

        results = [r for r in all_results if r is not None]

        logger.info(f"Batch analysis complete: {len(results)}/{len(signal_ids)} successful")
        return results

    async def auto_analyze_high_quality_signals(
        self,
        min_tier: str = 'A',
        days_back: int = 1,
        max_signals: int = 50,
        force_reanalyze: bool = False,
        include_non_active: bool = False
    ) -> List[DeepAnalysisResult]:
        """
        Automatically analyze high-quality signals that haven't been analyzed yet.

        This is called by the daily pipeline after scanner strategies run.

        Args:
            min_tier: Minimum quality tier ('A+' or 'A')
            days_back: How many days back to look for signals
            max_signals: Maximum signals to analyze

        Returns:
            List of DeepAnalysisResult objects
        """
        if not self.config.enabled:
            logger.info("Deep analysis is disabled")
            return []

        # Get unanalyzed high-quality signals
        if min_tier == 'B':
            tier_filter = "('A+', 'A', 'B')"
        elif min_tier == 'A':
            tier_filter = "('A+', 'A')"
        else:
            tier_filter = "('A+')"

        status_filter = "IN ('active', 'triggered', 'partial_exit')" if include_non_active else "= 'active'"

        query = f"""
            SELECT s.id, s.ticker, s.quality_tier, s.composite_score
            FROM stock_scanner_signals s
            LEFT JOIN stock_deep_analysis d ON s.id = d.signal_id
            WHERE s.quality_tier IN {tier_filter}
              AND s.signal_timestamp >= NOW() - INTERVAL '{days_back} days'
              AND s.status {status_filter}
              {"AND d.id IS NULL" if not force_reanalyze else ""}
            ORDER BY s.composite_score DESC
            LIMIT $1
        """

        rows = await self.db.fetch(query, max_signals)

        if not rows:
            logger.info("No unanalyzed high-quality signals found")
            return []

        signal_ids = [row['id'] for row in rows]
        logger.info(f"Found {len(signal_ids)} unanalyzed A+/A signals")

        return await self.analyze_signals_batch(
            signal_ids,
            save_to_db=True,
            max_concurrent=self.config.max_concurrent_analyses,
            force_reanalyze=force_reanalyze
        )

    async def _fetch_signal(self, signal_id: int) -> Optional[Dict[str, Any]]:
        """Fetch signal data from database"""
        query = """
            SELECT *
            FROM stock_scanner_signals
            WHERE id = $1
        """
        row = await self.db.fetchrow(query, signal_id)
        return dict(row) if row else None

    async def _get_ticker_sector(self, ticker: str) -> Optional[str]:
        """Get sector for a ticker"""
        query = "SELECT sector FROM stock_instruments WHERE ticker = $1"
        row = await self.db.fetchrow(query, ticker)
        return row['sector'] if row else None

    async def _check_cooldown(self, ticker: str) -> bool:
        """Check if ticker is within cooldown period"""
        if self.config.cooldown_hours <= 0:
            return True

        query = """
            SELECT COUNT(*) as count
            FROM stock_deep_analysis
            WHERE ticker = $1
              AND created_at >= NOW() - INTERVAL '{} hours'
        """.format(self.config.cooldown_hours)

        row = await self.db.fetchrow(query, ticker)
        return row['count'] == 0 if row else True

    async def _save_analysis(self, result: DeepAnalysisResult) -> bool:
        """Save deep analysis result to database"""
        try:
            data = result.to_db_dict()

            # Use INSERT ... ON CONFLICT to handle duplicates
            query = """
                INSERT INTO stock_deep_analysis (
                    signal_id, ticker, analysis_timestamp,
                    daq_score, daq_grade,
                    mtf_score, volume_score, smc_score,
                    quality_score, catalyst_score, institutional_score,
                    news_score, regime_score, sector_score,
                    earnings_within_7d, high_short_interest, low_liquidity,
                    extreme_volatility, sector_underperforming,
                    mtf_details, volume_details, smc_details,
                    fundamental_details, context_details,
                    news_summary, news_articles_count, top_headlines,
                    analysis_duration_ms, components_analyzed, errors
                ) VALUES (
                    $1, $2, $3,
                    $4, $5,
                    $6, $7, $8,
                    $9, $10, $11,
                    $12, $13, $14,
                    $15, $16, $17,
                    $18, $19,
                    $20, $21, $22,
                    $23, $24,
                    $25, $26, $27,
                    $28, $29, $30
                )
                ON CONFLICT (signal_id) DO UPDATE SET
                    daq_score = EXCLUDED.daq_score,
                    daq_grade = EXCLUDED.daq_grade,
                    mtf_score = EXCLUDED.mtf_score,
                    volume_score = EXCLUDED.volume_score,
                    smc_score = EXCLUDED.smc_score,
                    quality_score = EXCLUDED.quality_score,
                    catalyst_score = EXCLUDED.catalyst_score,
                    news_score = EXCLUDED.news_score,
                    regime_score = EXCLUDED.regime_score,
                    sector_score = EXCLUDED.sector_score,
                    analysis_duration_ms = EXCLUDED.analysis_duration_ms,
                    updated_at = NOW()
                RETURNING id
            """

            import json

            result_id = await self.db.fetchval(
                query,
                data['signal_id'],
                data['ticker'],
                data['analysis_timestamp'],
                data['daq_score'],
                data['daq_grade'],
                data['mtf_score'],
                data['volume_score'],
                data['smc_score'],
                data['quality_score'],
                data['catalyst_score'],
                data['institutional_score'],
                data['news_score'],
                data['regime_score'],
                data['sector_score'],
                data['earnings_within_7d'],
                data['high_short_interest'],
                data['low_liquidity'],
                data['extreme_volatility'],
                data['sector_underperforming'],
                json.dumps(data['mtf_details']) if data.get('mtf_details') else None,
                json.dumps(data['volume_details']) if data.get('volume_details') else None,
                json.dumps(data['smc_details']) if data.get('smc_details') else None,
                json.dumps(data['fundamental_details']) if data.get('fundamental_details') else None,
                json.dumps(data['context_details']) if data.get('context_details') else None,
                data['news_summary'],
                data['news_articles_count'],
                json.dumps(data['top_headlines']) if data.get('top_headlines') else None,
                data['analysis_duration_ms'],
                data['components_analyzed'],
                data['errors'],
            )

            if result_id:
                logger.debug(f"Saved deep analysis {result_id} for signal {data['signal_id']}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to save deep analysis: {e}")
            return False

    def _update_stats(self, result: DeepAnalysisResult) -> None:
        """Update internal stats"""
        self.stats['total_analyses'] += 1
        self.stats['successful_analyses'] += 1
        self.stats['total_duration_ms'] += result.analysis_duration_ms

        # Update rolling average DAQ score
        n = self.stats['successful_analyses']
        prev_avg = self.stats['avg_daq_score']
        self.stats['avg_daq_score'] = prev_avg + (result.daq_score - prev_avg) / n

    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return {
            **self.stats,
            'config': {
                'enabled': self.config.enabled,
                'min_tier': self.config.min_tier_for_auto,
                'cooldown_hours': self.config.cooldown_hours,
            }
        }

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    async def get_analysis_by_signal(self, signal_id: int) -> Optional[Dict[str, Any]]:
        """Get deep analysis for a signal"""
        query = """
            SELECT * FROM stock_deep_analysis
            WHERE signal_id = $1
        """
        row = await self.db.fetchrow(query, signal_id)
        return dict(row) if row else None

    async def get_recent_analyses(
        self,
        limit: int = 50,
        min_daq_score: int = None,
        min_grade: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent deep analyses.

        Args:
            limit: Maximum results
            min_daq_score: Minimum DAQ score filter
            min_grade: Minimum grade filter ('A+', 'A', 'B', etc.)

        Returns:
            List of analysis dictionaries
        """
        filters = []
        params = []

        if min_daq_score:
            params.append(min_daq_score)
            filters.append(f"daq_score >= ${len(params)}")

        if min_grade:
            grade_map = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
            min_grade_value = grade_map.get(min_grade, 1)
            valid_grades = [g for g, v in grade_map.items() if v >= min_grade_value]
            grades_str = ', '.join(f"'{g}'" for g in valid_grades)
            filters.append(f"daq_grade IN ({grades_str})")

        params.append(limit)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        query = f"""
            SELECT
                d.*,
                s.ticker,
                s.quality_tier as signal_tier,
                s.composite_score as signal_score,
                s.scanner_name
            FROM stock_deep_analysis d
            JOIN stock_scanner_signals s ON d.signal_id = s.id
            {where_clause}
            ORDER BY d.created_at DESC
            LIMIT ${len(params)}
        """

        rows = await self.db.fetch(query, *params)
        return [dict(row) for row in rows]

    async def get_analysis_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics for deep analyses"""
        query = """
            SELECT
                COUNT(*) as total_analyses,
                AVG(daq_score) as avg_daq_score,
                COUNT(*) FILTER (WHERE daq_grade = 'A+') as a_plus_count,
                COUNT(*) FILTER (WHERE daq_grade = 'A') as a_count,
                COUNT(*) FILTER (WHERE daq_grade = 'B') as b_count,
                COUNT(*) FILTER (WHERE daq_grade = 'C') as c_count,
                COUNT(*) FILTER (WHERE daq_grade = 'D') as d_count,
                AVG(mtf_score) as avg_mtf_score,
                AVG(volume_score) as avg_volume_score,
                AVG(smc_score) as avg_smc_score,
                AVG(quality_score) as avg_quality_score,
                AVG(news_score) as avg_news_score,
                AVG(regime_score) as avg_regime_score,
                AVG(sector_score) as avg_sector_score,
                AVG(analysis_duration_ms) as avg_duration_ms
            FROM stock_deep_analysis
            WHERE created_at >= NOW() - INTERVAL '{} days'
        """.format(days)

        row = await self.db.fetchrow(query)
        return dict(row) if row else {}

    # =========================================================================
    # WATCHLIST ANALYSIS METHODS
    # =========================================================================

    async def analyze_watchlist_ticker(
        self,
        ticker: str,
        watchlist_id: int,
        tier: int,
        save_to_db: bool = True
    ) -> Optional[DeepAnalysisResult]:
        """
        Run deep analysis for a watchlist ticker (not tied to a scanner signal).

        Args:
            ticker: Stock ticker
            watchlist_id: Watchlist row ID
            tier: Watchlist tier (1-5)
            save_to_db: Whether to save results to watchlist table

        Returns:
            DeepAnalysisResult or None if analysis fails
        """
        start_time = time.time()
        logger.info(f"Starting watchlist deep analysis for {ticker} (tier={tier})")

        try:
            # Get sector
            sector = await self._get_ticker_sector(ticker)

            # Create synthetic signal dict for analyzers (using BUY as default direction)
            synthetic_signal = {
                'ticker': ticker,
                'signal_type': 'BUY',  # Default direction for watchlist
                'scanner_name': 'watchlist',
            }

            # Run all analyzers in parallel
            technical_result, fundamental_result, contextual_result = await asyncio.gather(
                self.technical_analyzer.analyze(ticker, synthetic_signal),
                self.fundamental_analyzer.analyze(ticker, synthetic_signal),
                self.contextual_analyzer.analyze(ticker, synthetic_signal, sector),
            )

            # Create result (signal_id=0 since this is watchlist-based)
            result = DeepAnalysisResult(
                signal_id=0,  # No signal ID for watchlist analysis
                ticker=ticker,
                analysis_timestamp=datetime.now(),
                technical=technical_result,
                fundamental=fundamental_result,
                contextual=contextual_result,
                components_analyzed=['mtf', 'volume', 'smc', 'quality', 'catalyst', 'news', 'regime', 'sector'],
            )

            # Calculate composite DAQ score
            result.calculate_daq_score()

            # Record duration
            result.analysis_duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Watchlist deep analysis complete for {ticker}: "
                f"DAQ={result.daq_score} ({result.daq_grade.value}) "
                f"in {result.analysis_duration_ms}ms"
            )

            # Save to watchlist table
            if save_to_db:
                await self._save_watchlist_daq(watchlist_id, result)

            # Update stats
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"Watchlist deep analysis failed for {ticker}: {e}")
            self.stats['failed_analyses'] += 1
            return None

    async def analyze_watchlist_batch(
        self,
        tickers: List[Dict[str, Any]],
        save_to_db: bool = True,
        max_concurrent: int = 5
    ) -> List[DeepAnalysisResult]:
        """
        Run deep analysis for multiple watchlist tickers.

        Args:
            tickers: List of dicts with {id, ticker, calculation_date, tier}
            save_to_db: Whether to save results
            max_concurrent: Max concurrent analyses

        Returns:
            List of DeepAnalysisResult objects
        """
        logger.info(f"Starting batch watchlist analysis for {len(tickers)} tickers")

        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(t: Dict) -> Optional[DeepAnalysisResult]:
            async with semaphore:
                return await self.analyze_watchlist_ticker(
                    ticker=t['ticker'],
                    watchlist_id=t['id'],
                    tier=t['tier'],
                    save_to_db=save_to_db
                )

        tasks = [analyze_with_limit(t) for t in tickers]
        all_results = await asyncio.gather(*tasks)

        results = [r for r in all_results if r is not None]

        logger.info(f"Batch watchlist analysis complete: {len(results)}/{len(tickers)} successful")
        return results

    async def auto_analyze_watchlist(
        self,
        max_tier: int = 2,
        calculation_date: Optional[str] = None,
        max_tickers: int = 50,
        skip_analyzed: bool = True
    ) -> List[DeepAnalysisResult]:
        """
        Automatically analyze top-tier watchlist stocks.

        Args:
            max_tier: Maximum tier to analyze (1 = best, default includes 1 and 2)
            calculation_date: Specific date to analyze (default: latest)
            max_tickers: Maximum tickers to analyze
            skip_analyzed: Skip tickers that already have DAQ scores

        Returns:
            List of DeepAnalysisResult objects
        """
        if not self.config.enabled:
            logger.info("Deep analysis is disabled")
            return []

        # Build query for top-tier watchlist stocks with proper parameterization
        params = []
        param_idx = 1

        if calculation_date:
            date_filter = f"calculation_date = ${param_idx}"
            params.append(calculation_date)
            param_idx += 1
        else:
            date_filter = "calculation_date = (SELECT MAX(calculation_date) FROM stock_watchlist)"

        analyzed_filter = "AND daq_score IS NULL" if skip_analyzed else ""

        query = f"""
            SELECT id, ticker, calculation_date, tier
            FROM stock_watchlist
            WHERE {date_filter}
              AND tier <= ${param_idx}
              {analyzed_filter}
            ORDER BY tier, rank_in_tier
            LIMIT ${param_idx + 1}
        """

        params.extend([max_tier, max_tickers])
        rows = await self.db.fetch(query, *params)

        if not rows:
            logger.info("No unanalyzed watchlist tickers found")
            return []

        tickers = [dict(row) for row in rows]
        logger.info(f"Found {len(tickers)} tier 1-{max_tier} watchlist tickers to analyze")

        return await self.analyze_watchlist_batch(
            tickers,
            save_to_db=True,
            max_concurrent=self.config.max_concurrent_analyses
        )

    async def _save_watchlist_daq(self, watchlist_id: int, result: DeepAnalysisResult) -> bool:
        """Save DAQ results to watchlist table"""
        try:
            query = """
                UPDATE stock_watchlist SET
                    daq_score = $1,
                    daq_grade = $2,
                    daq_mtf_score = $3,
                    daq_volume_score = $4,
                    daq_smc_score = $5,
                    daq_quality_score = $6,
                    daq_catalyst_score = $7,
                    daq_news_score = $8,
                    daq_regime_score = $9,
                    daq_sector_score = $10,
                    daq_earnings_risk = $11,
                    daq_high_short_interest = $12,
                    daq_sector_underperforming = $13,
                    daq_analyzed_at = NOW()
                WHERE id = $14
                RETURNING id
            """

            result_id = await self.db.fetchval(
                query,
                result.daq_score,
                result.daq_grade.value,
                result.technical.mtf.score,
                result.technical.volume.score,
                result.technical.smc.score,
                result.fundamental.quality.score,
                result.fundamental.catalyst.score,
                result.contextual.news.score,
                result.contextual.regime.score,
                result.contextual.sector.score,
                result.earnings_within_7d,
                result.high_short_interest,
                result.sector_underperforming,
                watchlist_id,
            )

            if result_id:
                logger.debug(f"Saved DAQ to watchlist {watchlist_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to save watchlist DAQ: {e}")
            return False

    # =========================================================================
    # TECHNICAL WATCHLIST ANALYSIS (stock_watchlist_results table)
    # =========================================================================

    async def analyze_technical_watchlist_ticker(
        self,
        ticker: str,
        result_id: int,
        watchlist_name: str,
        save_to_db: bool = True,
    ) -> Optional[DeepAnalysisResult]:
        """
        Analyze a single ticker from technical watchlist (stock_watchlist_results).

        Args:
            ticker: Stock ticker symbol
            result_id: The id from stock_watchlist_results table
            watchlist_name: Type of watchlist (ema_50_crossover, macd_bullish_cross, etc.)
            save_to_db: Whether to save results to database

        Returns:
            DeepAnalysisResult or None if analysis fails
        """
        start_time = time.time()
        logger.info(f"Starting technical watchlist deep analysis for {ticker} (type={watchlist_name})")

        try:
            # Get sector
            sector = await self._get_ticker_sector(ticker)

            # Create synthetic signal dict for analyzers (using BUY as default direction)
            synthetic_signal = {
                'ticker': ticker,
                'signal_type': 'BUY',  # Default direction for watchlist
                'scanner_name': watchlist_name,
            }

            # Run all analyzers in parallel
            technical_result, fundamental_result, contextual_result = await asyncio.gather(
                self.technical_analyzer.analyze(ticker, synthetic_signal),
                self.fundamental_analyzer.analyze(ticker, synthetic_signal),
                self.contextual_analyzer.analyze(ticker, synthetic_signal, sector),
            )

            # Create result (signal_id=0 since this is watchlist-based)
            result = DeepAnalysisResult(
                signal_id=0,  # No signal ID for technical watchlist analysis
                ticker=ticker,
                analysis_timestamp=datetime.now(),
                technical=technical_result,
                fundamental=fundamental_result,
                contextual=contextual_result,
                components_analyzed=['mtf', 'volume', 'smc', 'quality', 'catalyst', 'news', 'regime', 'sector'],
            )

            # Calculate composite DAQ score
            result.calculate_daq_score()

            # Record duration
            result.analysis_duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Technical watchlist deep analysis complete for {ticker}: "
                f"DAQ={result.daq_score} ({result.daq_grade.value}) "
                f"in {result.analysis_duration_ms}ms"
            )

            # Save to technical watchlist table
            if save_to_db:
                await self._save_technical_watchlist_daq(result_id, result)

            # Update stats
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"Technical watchlist deep analysis failed for {ticker}: {e}")
            self.stats['failed_analyses'] += 1
            return None

    async def analyze_technical_watchlist_batch(
        self,
        tickers: List[Dict[str, Any]],
        save_to_db: bool = True,
        max_concurrent: int = 5,
    ) -> List[DeepAnalysisResult]:
        """
        Analyze multiple tickers from technical watchlist concurrently.

        Args:
            tickers: List of dicts with keys: ticker, id, watchlist_name
            save_to_db: Whether to save results to database
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of successful DeepAnalysisResult objects
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(item: Dict[str, Any]) -> Optional[DeepAnalysisResult]:
            async with semaphore:
                return await self.analyze_technical_watchlist_ticker(
                    ticker=item["ticker"],
                    result_id=item["id"],
                    watchlist_name=item["watchlist_name"],
                    save_to_db=save_to_db,
                )

        tasks = [analyze_with_semaphore(item) for item in tickers]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(completed):
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for {tickers[i]['ticker']}: {result}")
            elif result:
                results.append(result)

        return results

    async def auto_analyze_technical_watchlist(
        self,
        watchlist_name: Optional[str] = None,
        scan_date: Optional[str] = None,
        max_tickers: int = 50,
        skip_analyzed: bool = True,
    ) -> List[DeepAnalysisResult]:
        """
        Auto-analyze technical watchlist stocks that don't have DAQ scores.

        Args:
            watchlist_name: Filter by watchlist type (None = all types)
            scan_date: Filter by scan date (None = most recent)
            max_tickers: Maximum tickers to analyze
            skip_analyzed: Skip tickers that already have DAQ scores

        Returns:
            List of DeepAnalysisResult objects
        """
        # Build query to get technical watchlist stocks
        conditions = []
        params = []
        param_idx = 1

        if skip_analyzed:
            conditions.append("daq_score IS NULL")

        if watchlist_name:
            conditions.append(f"watchlist_name = ${param_idx}")
            params.append(watchlist_name)
            param_idx += 1

        if scan_date:
            conditions.append(f"DATE(scan_date) = ${param_idx}")
            # Convert string date to date object for asyncpg
            if isinstance(scan_date, str):
                from datetime import datetime
                scan_date_obj = datetime.strptime(scan_date, "%Y-%m-%d").date()
                params.append(scan_date_obj)
            else:
                params.append(scan_date)  # Already a date object
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT id, ticker, watchlist_name, scan_date
            FROM stock_watchlist_results
            WHERE {where_clause}
            ORDER BY scan_date DESC, volume DESC NULLS LAST
            LIMIT {max_tickers}
        """

        rows = await self.db.fetch(query, *params)

        if not rows:
            logger.info("No technical watchlist stocks to analyze")
            return []

        # Prepare ticker list for batch analysis
        tickers = [
            {
                "id": row["id"],
                "ticker": row["ticker"],
                "watchlist_name": row["watchlist_name"],
            }
            for row in rows
        ]

        logger.info(f"Auto-analyzing {len(tickers)} technical watchlist stocks")

        # Run batch analysis
        return await self.analyze_technical_watchlist_batch(tickers, save_to_db=True)

    async def _save_technical_watchlist_daq(
        self, result_id: int, result: DeepAnalysisResult
    ) -> bool:
        """
        Save DAQ scores to stock_watchlist_results table.

        Args:
            result_id: The id from stock_watchlist_results
            result: DeepAnalysisResult with scores

        Returns:
            True if saved successfully
        """
        try:
            query = """
                UPDATE stock_watchlist_results
                SET daq_score = $1,
                    daq_grade = $2,
                    daq_mtf_score = $3,
                    daq_volume_score = $4,
                    daq_smc_score = $5,
                    daq_quality_score = $6,
                    daq_catalyst_score = $7,
                    daq_news_score = $8,
                    daq_regime_score = $9,
                    daq_sector_score = $10,
                    daq_earnings_risk = $11,
                    daq_high_short_interest = $12,
                    daq_sector_underperforming = $13,
                    daq_analyzed_at = NOW()
                WHERE id = $14
                RETURNING id
            """

            updated_id = await self.db.fetchval(
                query,
                result.daq_score,
                result.daq_grade.value,
                result.technical.mtf.score,
                result.technical.volume.score,
                result.technical.smc.score,
                result.fundamental.quality.score,
                result.fundamental.catalyst.score,
                result.contextual.news.score,
                result.contextual.regime.score,
                result.contextual.sector.score,
                result.earnings_within_7d,
                result.high_short_interest,
                result.sector_underperforming,
                result_id,
            )

            if updated_id:
                logger.debug(f"Saved DAQ to technical watchlist result {result_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to save technical watchlist DAQ: {e}")
            return False
