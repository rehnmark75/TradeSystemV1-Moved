"""
Scanner Manager

Orchestrates all signal scanners and provides unified interface for:
- Running all scanners
- Combining and ranking results
- Performance tracking
- Signal management
- Export functionality
- Claude AI analysis integration
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type, Tuple
from collections import defaultdict

from .base_scanner import BaseScanner, SignalSetup, ScannerConfig, QualityTier
from .scoring import SignalScorer
from .strategies import (
    TrendMomentumScanner,
    BreakoutConfirmationScanner,
    MeanReversionScanner,
    GapAndGoScanner,
    EarningsMomentumScanner,
    ShortSqueezeScanner,
    SectorRotationScanner,
    # Forex-adapted strategies
    SMCEmaTrendScanner,
    EMACrossoverScanner,
    MACDMomentumScanner,
)

# Claude analysis imports (lazy load to avoid circular imports)
try:
    from ..services import StockClaudeAnalyzer, ClaudeAnalysis
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    StockClaudeAnalyzer = None
    ClaudeAnalysis = None

logger = logging.getLogger(__name__)


class ScannerManager:
    """
    Central manager for all signal scanners.

    Features:
    - Run all scanners in parallel
    - Deduplicate signals across scanners
    - Rank and filter combined results
    - Track scanner performance
    - Save and manage signals in database

    Usage:
        manager = ScannerManager(db_manager)
        await manager.initialize()

        # Run all scanners
        signals = await manager.run_all_scanners()

        # Get top signals by quality
        top_signals = manager.get_top_signals(limit=20)

        # Export to TradingView
        csv = manager.export_tradingview_csv()
    """

    # Available scanner classes
    SCANNER_CLASSES: Dict[str, Type[BaseScanner]] = {
        'trend_momentum': TrendMomentumScanner,
        'breakout_confirmation': BreakoutConfirmationScanner,
        'mean_reversion': MeanReversionScanner,
        'gap_and_go': GapAndGoScanner,
        'earnings_momentum': EarningsMomentumScanner,
        'short_squeeze': ShortSqueezeScanner,
        'sector_rotation': SectorRotationScanner,
        # Forex-adapted strategies
        'smc_ema_trend': SMCEmaTrendScanner,
        'ema_crossover': EMACrossoverScanner,
        'macd_momentum': MACDMomentumScanner,
    }

    def __init__(
        self,
        db_manager,
        enabled_scanners: List[str] = None,
        scorer: SignalScorer = None
    ):
        """
        Initialize Scanner Manager.

        Args:
            db_manager: AsyncDatabaseManager instance
            enabled_scanners: List of scanner names to enable (None = all)
            scorer: Shared SignalScorer instance
        """
        self.db = db_manager
        self.scorer = scorer or SignalScorer()
        self.enabled_scanners = enabled_scanners or list(self.SCANNER_CLASSES.keys())

        self._scanners: Dict[str, BaseScanner] = {}
        self._latest_signals: List[SignalSetup] = []
        self._scan_stats: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize all enabled scanners"""
        logger.info("Initializing Scanner Manager")

        for scanner_name in self.enabled_scanners:
            if scanner_name in self.SCANNER_CLASSES:
                scanner_class = self.SCANNER_CLASSES[scanner_name]
                self._scanners[scanner_name] = scanner_class(
                    self.db,
                    scorer=self.scorer
                )
                logger.info(f"  Initialized: {scanner_name}")
            else:
                logger.warning(f"  Unknown scanner: {scanner_name}")

        # Verify scanners are registered in database
        await self._verify_scanner_registration()

        logger.info(f"Initialized {len(self._scanners)} scanners")

    async def _verify_scanner_registration(self):
        """Ensure all scanners are registered in database"""
        for scanner_name, scanner in self._scanners.items():
            exists = await scanner.check_scanner_exists()
            if not exists:
                logger.warning(
                    f"Scanner '{scanner_name}' not registered in database. "
                    f"Run migration to add it."
                )

    # =========================================================================
    # SCANNING OPERATIONS
    # =========================================================================

    async def run_all_scanners(
        self,
        calculation_date: datetime = None,
        save_to_db: bool = True
    ) -> List[SignalSetup]:
        """
        Run all enabled scanners and combine results.

        Args:
            calculation_date: Date to scan
            save_to_db: Whether to save signals to database

        Returns:
            Combined list of signals, deduplicated and ranked
        """
        logger.info("=" * 60)
        logger.info("RUNNING ALL SCANNERS")
        logger.info("=" * 60)

        if calculation_date is None:
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        start_time = datetime.now()
        all_signals = []
        scanner_results = {}

        # Run scanners in parallel
        tasks = []
        for scanner_name, scanner in self._scanners.items():
            tasks.append(self._run_scanner_safe(scanner, calculation_date))

        results = await asyncio.gather(*tasks)

        # Collect results
        for scanner_name, signals in zip(self._scanners.keys(), results):
            scanner_results[scanner_name] = signals
            all_signals.extend(signals)
            logger.info(f"  {scanner_name}: {len(signals)} signals")

        # Deduplicate (same ticker from multiple scanners)
        deduplicated = self._deduplicate_signals(all_signals)

        # Sort by composite score
        deduplicated.sort(key=lambda x: x.composite_score, reverse=True)

        # Store latest results
        self._latest_signals = deduplicated

        # Calculate stats
        elapsed = (datetime.now() - start_time).total_seconds()
        self._scan_stats = {
            'scan_date': str(calculation_date),
            'scan_duration_seconds': round(elapsed, 2),
            'total_signals': len(deduplicated),
            'signals_by_scanner': {k: len(v) for k, v in scanner_results.items()},
            'signals_by_tier': self._count_by_tier(deduplicated),
            'high_quality_count': sum(1 for s in deduplicated if s.is_high_quality),
        }

        # Save to database
        if save_to_db and deduplicated:
            saved = await self._save_all_signals(deduplicated)
            self._scan_stats['signals_saved'] = saved

        # Log summary
        self._log_scan_summary(deduplicated)

        return deduplicated

    async def _run_scanner_safe(
        self,
        scanner: BaseScanner,
        calculation_date: datetime
    ) -> List[SignalSetup]:
        """Run a scanner with error handling"""
        try:
            return await scanner.scan(calculation_date)
        except Exception as e:
            logger.error(f"Scanner {scanner.scanner_name} failed: {e}")
            return []

    async def run_single_scanner(
        self,
        scanner_name: str,
        calculation_date: datetime = None,
        save_to_db: bool = True
    ) -> List[SignalSetup]:
        """
        Run a specific scanner by name.

        Args:
            scanner_name: Name of scanner to run
            calculation_date: Date to scan
            save_to_db: Whether to save signals

        Returns:
            List of signals from the scanner
        """
        if scanner_name not in self._scanners:
            raise ValueError(f"Scanner '{scanner_name}' not found or not enabled")

        scanner = self._scanners[scanner_name]
        signals = await scanner.scan(calculation_date)

        if save_to_db and signals:
            await scanner.save_signals(signals)

        return signals

    # =========================================================================
    # SIGNAL MANAGEMENT
    # =========================================================================

    def _deduplicate_signals(
        self,
        signals: List[SignalSetup]
    ) -> List[SignalSetup]:
        """
        Deduplicate signals - keep highest score when same ticker appears.

        If same ticker from multiple scanners, keep the one with highest score.
        """
        ticker_signals: Dict[str, SignalSetup] = {}

        for signal in signals:
            ticker = signal.ticker

            if ticker not in ticker_signals:
                ticker_signals[ticker] = signal
            else:
                # Keep higher score
                if signal.composite_score > ticker_signals[ticker].composite_score:
                    ticker_signals[ticker] = signal

        return list(ticker_signals.values())

    def get_top_signals(
        self,
        limit: int = 20,
        min_tier: str = None,
        scanner_filter: str = None
    ) -> List[SignalSetup]:
        """
        Get top signals from latest scan.

        Args:
            limit: Maximum number of signals
            min_tier: Minimum quality tier (e.g., "B" means B, A, A+)
            scanner_filter: Only include signals from specific scanner

        Returns:
            Filtered and limited list of signals
        """
        signals = self._latest_signals.copy()

        # Filter by scanner
        if scanner_filter:
            signals = [s for s in signals if s.scanner_name == scanner_filter]

        # Filter by tier
        if min_tier:
            tier_order = {'D': 0, 'C': 1, 'B': 2, 'A': 3, 'A+': 4}
            min_tier_value = tier_order.get(min_tier, 0)
            signals = [
                s for s in signals
                if tier_order.get(s.quality_tier.value, 0) >= min_tier_value
            ]

        return signals[:limit]

    def get_signals_by_scanner(self) -> Dict[str, List[SignalSetup]]:
        """Group signals by scanner"""
        by_scanner = defaultdict(list)

        for signal in self._latest_signals:
            by_scanner[signal.scanner_name].append(signal)

        return dict(by_scanner)

    def get_signals_by_tier(self) -> Dict[str, List[SignalSetup]]:
        """Group signals by quality tier"""
        by_tier = defaultdict(list)

        for signal in self._latest_signals:
            by_tier[signal.quality_tier.value].append(signal)

        return dict(by_tier)

    async def _save_all_signals(self, signals: List[SignalSetup]) -> int:
        """Save all signals to database"""
        # Group by scanner and save
        by_scanner = defaultdict(list)
        for signal in signals:
            by_scanner[signal.scanner_name].append(signal)

        total_saved = 0
        for scanner_name, scanner_signals in by_scanner.items():
            if scanner_name in self._scanners:
                saved = await self._scanners[scanner_name].save_signals(scanner_signals)
                total_saved += saved

        return total_saved

    # =========================================================================
    # ACTIVE SIGNAL MANAGEMENT
    # =========================================================================

    async def get_all_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active signals across all scanners"""
        query = """
            SELECT * FROM stock_scanner_signals
            WHERE status = 'active'
            ORDER BY composite_score DESC
        """
        rows = await self.db.fetch(query)
        return [dict(r) for r in rows]

    async def update_signal_status(
        self,
        signal_id: int,
        new_status: str,
        close_price: float = None,
        exit_reason: str = None
    ) -> bool:
        """
        Update a signal's status.

        Args:
            signal_id: Signal ID
            new_status: New status ('triggered', 'partial_exit', 'closed', 'expired', 'cancelled')
            close_price: Price at close (for closed signals)
            exit_reason: Reason for exit

        Returns:
            True if updated successfully
        """
        query = """
            UPDATE stock_scanner_signals
            SET status = $1,
                close_timestamp = CASE WHEN $1 IN ('closed', 'expired', 'cancelled') THEN NOW() ELSE close_timestamp END,
                close_price = COALESCE($2, close_price),
                exit_reason = COALESCE($3, exit_reason)
            WHERE id = $4
            RETURNING id
        """
        result = await self.db.fetchval(query, new_status, close_price, exit_reason, signal_id)
        return result is not None

    async def expire_old_signals(self, days_old: int = 5) -> int:
        """
        Expire signals older than specified days.

        Args:
            days_old: Number of days after which to expire

        Returns:
            Number of signals expired
        """
        query = """
            UPDATE stock_scanner_signals
            SET status = 'expired',
                close_timestamp = NOW(),
                exit_reason = 'Signal expired after {} days'
            WHERE status = 'active'
              AND signal_timestamp < NOW() - INTERVAL '{} days'
            RETURNING id
        """.format(days_old, days_old)

        rows = await self.db.fetch(query)
        expired_count = len(rows)

        if expired_count > 0:
            logger.info(f"Expired {expired_count} old signals")

        return expired_count

    # =========================================================================
    # PERFORMANCE TRACKING
    # =========================================================================

    async def record_daily_performance(self, evaluation_date: datetime = None):
        """
        Record daily performance metrics for each scanner.

        Should be called at end of trading day.
        """
        if evaluation_date is None:
            evaluation_date = datetime.now().date()

        for scanner_name in self._scanners.keys():
            await self._record_scanner_performance(scanner_name, evaluation_date)

    async def _record_scanner_performance(
        self,
        scanner_name: str,
        evaluation_date: datetime
    ):
        """Record performance for a single scanner"""

        # Get closed signals for this period
        query = """
            SELECT
                COUNT(*) as total_signals,
                COUNT(*) FILTER (WHERE status = 'triggered') as triggered,
                COUNT(*) FILTER (WHERE status = 'closed') as closed,
                COUNT(*) FILTER (WHERE status = 'expired') as expired,
                COUNT(*) FILTER (WHERE realized_pnl_pct > 0) as winners,
                COUNT(*) FILTER (WHERE realized_pnl_pct <= 0) as losers,
                AVG(realized_pnl_pct) FILTER (WHERE realized_pnl_pct > 0) as avg_win,
                AVG(realized_pnl_pct) FILTER (WHERE realized_pnl_pct <= 0) as avg_loss,
                AVG(realized_r_multiple) as avg_r,
                MAX(realized_r_multiple) as max_r,
                MIN(realized_r_multiple) as min_r,
                COUNT(*) FILTER (WHERE quality_tier = 'A+') as a_plus_signals,
                COUNT(*) FILTER (WHERE quality_tier = 'A') as a_signals,
                COUNT(*) FILTER (WHERE quality_tier = 'B') as b_signals
            FROM stock_scanner_signals
            WHERE scanner_name = $1
              AND DATE(signal_timestamp) = $2
        """

        row = await self.db.fetchrow(query, scanner_name, evaluation_date)

        if row and row['total_signals'] > 0:
            # Calculate metrics
            total = row['total_signals'] or 1
            winners = row['winners'] or 0
            losers = row['losers'] or 0
            closed = winners + losers

            win_rate = (winners / closed * 100) if closed > 0 else 0
            avg_win = row['avg_win'] or 0
            avg_loss = abs(row['avg_loss'] or 0)
            profit_factor = (avg_win * winners) / (avg_loss * losers) if losers > 0 and avg_loss > 0 else 0

            # Calculate expectancy
            expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

            # Insert performance record
            insert_query = """
                INSERT INTO stock_scanner_performance (
                    scanner_name, evaluation_date, evaluation_period,
                    total_signals, signals_triggered, signals_closed, signals_expired,
                    win_rate, avg_win_pct, avg_loss_pct, profit_factor,
                    avg_r_multiple, max_r_multiple, min_r_multiple, expectancy,
                    a_plus_signals, a_signals, b_signals
                ) VALUES (
                    $1, $2, 'daily',
                    $3, $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12, $13, $14,
                    $15, $16, $17
                )
                ON CONFLICT (scanner_name, evaluation_date, evaluation_period)
                DO UPDATE SET
                    total_signals = EXCLUDED.total_signals,
                    win_rate = EXCLUDED.win_rate,
                    profit_factor = EXCLUDED.profit_factor
            """

            await self.db.execute(
                insert_query,
                scanner_name, evaluation_date, 'daily',
                row['total_signals'], row['triggered'], row['closed'], row['expired'],
                win_rate, avg_win, avg_loss, profit_factor,
                row['avg_r'], row['max_r'], row['min_r'], expectancy,
                row['a_plus_signals'], row['a_signals'], row['b_signals']
            )

    async def get_scanner_performance(
        self,
        scanner_name: str = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get scanner performance history.

        Args:
            scanner_name: Specific scanner (None = all)
            days: Number of days to look back

        Returns:
            List of performance records
        """
        query = """
            SELECT * FROM stock_scanner_performance
            WHERE evaluation_date >= CURRENT_DATE - INTERVAL '{} days'
            {}
            ORDER BY evaluation_date DESC, scanner_name
        """.format(
            days,
            f"AND scanner_name = '{scanner_name}'" if scanner_name else ""
        )

        rows = await self.db.fetch(query)
        return [dict(r) for r in rows]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _count_by_tier(self, signals: List[SignalSetup]) -> Dict[str, int]:
        """Count signals by quality tier"""
        counts = {'A+': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for signal in signals:
            tier = signal.quality_tier.value
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    def _log_scan_summary(self, signals: List[SignalSetup]):
        """Log summary of scan results"""
        stats = self._scan_stats

        logger.info("=" * 60)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Date: {stats.get('scan_date')}")
        logger.info(f"  Duration: {stats.get('scan_duration_seconds')}s")
        logger.info(f"  Total Signals: {stats.get('total_signals')}")
        logger.info(f"  High Quality (A/A+): {stats.get('high_quality_count')}")
        logger.info("")
        logger.info("  By Scanner:")
        for scanner, count in stats.get('signals_by_scanner', {}).items():
            logger.info(f"    {scanner}: {count}")
        logger.info("")
        logger.info("  By Tier:")
        for tier, count in stats.get('signals_by_tier', {}).items():
            if count > 0:
                logger.info(f"    {tier}: {count}")
        logger.info("=" * 60)

    def get_scan_stats(self) -> Dict[str, Any]:
        """Get statistics from latest scan"""
        return self._scan_stats.copy()

    @property
    def scanner_names(self) -> List[str]:
        """Get list of enabled scanner names"""
        return list(self._scanners.keys())

    @property
    def latest_signals(self) -> List[SignalSetup]:
        """Get signals from latest scan"""
        return self._latest_signals.copy()

    # =========================================================================
    # CLAUDE AI ANALYSIS
    # =========================================================================

    async def analyze_signals_with_claude(
        self,
        signals: List[Dict[str, Any]] = None,
        min_tier: str = 'A',
        max_signals: int = 10,
        analysis_level: str = 'standard',
        model: str = None
    ) -> List[Tuple[Dict[str, Any], Any]]:
        """
        Analyze signals with Claude AI.

        Args:
            signals: Signals to analyze (None = get from DB)
            min_tier: Minimum tier to analyze ('A+', 'A', 'B', etc.)
            max_signals: Maximum number of signals to analyze
            analysis_level: 'quick', 'standard', or 'comprehensive'
            model: Override model ('haiku', 'sonnet', 'opus')

        Returns:
            List of (signal, ClaudeAnalysis) tuples
        """
        if not CLAUDE_AVAILABLE:
            logger.warning("Claude analysis not available - missing dependencies")
            return []

        # Get signals if not provided
        if signals is None:
            signals = await self.get_unanalyzed_signals(min_tier=min_tier, limit=max_signals)

        if not signals:
            logger.info("No signals to analyze with Claude")
            return []

        logger.info(f"Starting Claude analysis for {len(signals)} signals")

        # Initialize analyzer with database for chart generation
        analyzer = StockClaudeAnalyzer(
            default_model=model or 'sonnet',
            db_manager=self.db,
            enable_charts=True
        )

        if not analyzer.is_available:
            logger.warning("Claude API not available - check API key")
            return []

        if analyzer.charts_available:
            logger.info("Chart generation enabled for vision analysis")
        else:
            logger.info("Chart generation not available - text-only analysis")

        # Enrich signals with technical, fundamental, and SMC data
        technical_data_list = []
        fundamental_data_list = []

        for signal in signals:
            technical_data = await self._get_signal_technical_data(signal)
            fundamental_data = await self._get_signal_fundamental_data(signal)
            smc_data = await self._get_signal_smc_data(signal)

            # Merge SMC data into technical data for Claude
            if smc_data:
                technical_data['smc'] = smc_data

            technical_data_list.append(technical_data)
            fundamental_data_list.append(fundamental_data)

            ticker = signal.get('ticker', '???')
            has_fundamentals = bool(fundamental_data)
            has_smc = bool(smc_data)
            logger.debug(f"Enriched {ticker}: fundamentals={'yes' if has_fundamentals else 'no'}, smc={'yes' if has_smc else 'no'}")

        # Analyze in batch with enriched data
        results = await analyzer.batch_analyze_signals(
            signals=signals,
            technical_data_list=technical_data_list,
            fundamental_data_list=fundamental_data_list,
            analysis_level=analysis_level,
            max_concurrent=3,
            delay_between_requests=0.5
        )

        # Save analysis results to database
        for signal, analysis in results:
            if analysis:
                await self._save_claude_analysis(signal, analysis)

        logger.info(f"Completed Claude analysis: {len(results)} signals")

        return results

    async def analyze_single_signal_with_claude(
        self,
        signal_id: int,
        analysis_level: str = 'comprehensive',
        model: str = None
    ) -> Optional[Any]:
        """
        Analyze a single signal with Claude AI.

        Args:
            signal_id: Signal ID from database
            analysis_level: 'quick', 'standard', or 'comprehensive'
            model: Override model ('haiku', 'sonnet', 'opus')

        Returns:
            ClaudeAnalysis or None
        """
        if not CLAUDE_AVAILABLE:
            logger.warning("Claude analysis not available - missing dependencies")
            return None

        # Get signal from database
        query = """
            SELECT * FROM stock_scanner_signals
            WHERE id = $1
        """
        row = await self.db.fetchrow(query, signal_id)

        if not row:
            logger.warning(f"Signal {signal_id} not found")
            return None

        signal = dict(row)

        # Get additional technical data if available
        technical_data = await self._get_signal_technical_data(signal)
        fundamental_data = await self._get_signal_fundamental_data(signal)
        smc_data = await self._get_signal_smc_data(signal)

        # Merge SMC data into technical data for Claude
        if smc_data:
            technical_data['smc'] = smc_data

        # Initialize analyzer with database for chart generation
        analyzer = StockClaudeAnalyzer(
            default_model=model or 'sonnet',
            db_manager=self.db,
            enable_charts=True
        )

        if not analyzer.is_available:
            logger.warning("Claude API not available - check API key")
            return None

        # Analyze with chart if available
        analysis = await analyzer.analyze_signal(
            signal=signal,
            technical_data=technical_data,
            fundamental_data=fundamental_data,
            analysis_level=analysis_level,
            model=model,
            include_chart=True  # Enable chart for single signal analysis
        )

        # Save to database
        if analysis:
            await self._save_claude_analysis(signal, analysis)

        return analysis

    async def get_unanalyzed_signals(
        self,
        min_tier: str = 'B',
        limit: int = 20,
        days_back: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get signals that haven't been analyzed by Claude yet.

        Args:
            min_tier: Minimum quality tier
            limit: Maximum number of signals
            days_back: How many days back to look

        Returns:
            List of signal dictionaries
        """
        tier_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
        min_tier_value = tier_order.get(min_tier, 3)

        # Build tier filter
        valid_tiers = [t for t, v in tier_order.items() if v >= min_tier_value]
        tier_filter = ', '.join(f"'{t}'" for t in valid_tiers)

        query = f"""
            SELECT * FROM stock_scanner_signals
            WHERE claude_analyzed_at IS NULL
              AND quality_tier IN ({tier_filter})
              AND signal_timestamp >= NOW() - INTERVAL '{days_back} days'
              AND status = 'active'
            ORDER BY composite_score DESC
            LIMIT $1
        """

        rows = await self.db.fetch(query, limit)
        return [dict(r) for r in rows]

    async def _save_claude_analysis(
        self,
        signal: Dict[str, Any],
        analysis: Any
    ) -> bool:
        """
        Save Claude analysis results to database.

        Args:
            signal: Original signal dictionary
            analysis: ClaudeAnalysis object

        Returns:
            True if saved successfully
        """
        signal_id = signal.get('id')
        if not signal_id:
            logger.warning("Cannot save analysis - signal has no ID")
            return False

        try:
            query = """
                UPDATE stock_scanner_signals
                SET
                    claude_grade = $1,
                    claude_score = $2,
                    claude_conviction = $3,
                    claude_action = $4,
                    claude_thesis = $5,
                    claude_key_strengths = $6,
                    claude_key_risks = $7,
                    claude_position_rec = $8,
                    claude_stop_adjustment = $9,
                    claude_time_horizon = $10,
                    claude_raw_response = $11,
                    claude_analyzed_at = NOW(),
                    claude_tokens_used = $12,
                    claude_latency_ms = $13,
                    claude_model = $14
                WHERE id = $15
                RETURNING id
            """

            result = await self.db.fetchval(
                query,
                analysis.grade,
                analysis.score,
                analysis.conviction,
                analysis.action,
                analysis.thesis,
                analysis.key_strengths,
                analysis.key_risks,
                analysis.position_recommendation,
                analysis.stop_adjustment,
                analysis.time_horizon,
                analysis.raw_response,
                analysis.tokens_used,
                analysis.latency_ms,
                analysis.model,
                signal_id
            )

            if result:
                logger.info(
                    f"Saved Claude analysis for signal {signal_id}: "
                    f"Grade={analysis.grade}, Action={analysis.action}"
                )
                return True
            else:
                logger.warning(f"Failed to save Claude analysis for signal {signal_id}")
                return False

        except Exception as e:
            logger.error(f"Error saving Claude analysis: {e}")
            return False

    async def _get_signal_technical_data(
        self,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract technical data from signal for Claude analysis.

        Args:
            signal: Signal dictionary

        Returns:
            Technical data dictionary
        """
        technical = {}

        # Extract technical fields from signal
        technical_fields = [
            'rsi_14', 'macd_histogram', 'relative_volume', 'trend_strength',
            'atr_stop_distance', 'entry_price', 'stop_loss', 'targets',
            'volume_confirm', 'trend_alignment', 'momentum_score'
        ]

        for field in technical_fields:
            if field in signal and signal[field] is not None:
                technical[field] = signal[field]

        # Add additional context
        if signal.get('confluence_factors'):
            technical['confluence_factors'] = signal['confluence_factors']

        return technical

    async def _get_signal_fundamental_data(
        self,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get fundamental data for signal ticker from stock_instruments table.

        Args:
            signal: Signal dictionary

        Returns:
            Fundamental data dictionary
        """
        ticker = signal.get('ticker')
        if not ticker:
            return {}

        # Try to get fundamental data from stock_instruments table
        try:
            query = """
                SELECT
                    ticker, name, sector, industry,
                    -- Valuation
                    trailing_pe, forward_pe, price_to_book, price_to_sales,
                    peg_ratio, enterprise_to_ebitda, enterprise_value,
                    -- Growth
                    revenue_growth, earnings_growth, earnings_quarterly_growth,
                    -- Profitability
                    profit_margin, operating_margin, gross_margin,
                    return_on_equity, return_on_assets,
                    -- Financial Health
                    debt_to_equity, current_ratio, quick_ratio,
                    -- Risk Metrics
                    beta, short_ratio, short_percent_float,
                    -- Ownership
                    institutional_percent, insider_percent,
                    -- Dividend
                    dividend_yield, dividend_rate, payout_ratio,
                    -- 52-Week Data
                    fifty_two_week_high, fifty_two_week_low, fifty_two_week_change,
                    fifty_day_average, two_hundred_day_average,
                    -- Analyst Data
                    analyst_rating, target_price, target_high, target_low, number_of_analysts,
                    -- Calendar
                    earnings_date, ex_dividend_date,
                    -- Meta
                    fundamentals_updated_at
                FROM stock_instruments
                WHERE ticker = $1
                  AND fundamentals_updated_at IS NOT NULL
            """
            row = await self.db.fetchrow(query, ticker)

            if row:
                # Convert to dict and filter out None values for cleaner output
                data = dict(row)
                return {k: v for k, v in data.items() if v is not None}
        except Exception as e:
            logger.debug(f"Could not fetch fundamentals for {ticker}: {e}")

        return {}

    async def _get_signal_smc_data(
        self,
        signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get SMC (Smart Money Concepts) data for signal ticker from stock_screening_metrics.

        Args:
            signal: Signal dictionary

        Returns:
            SMC data dictionary
        """
        ticker = signal.get('ticker')
        if not ticker:
            return {}

        try:
            query = """
                SELECT
                    smc_trend,
                    smc_bias,
                    last_bos_type,
                    last_bos_date,
                    last_bos_price,
                    last_choch_type,
                    last_choch_date,
                    swing_high,
                    swing_low,
                    swing_high_date,
                    swing_low_date,
                    premium_discount_zone,
                    zone_position,
                    weekly_range_high,
                    weekly_range_low,
                    nearest_ob_type,
                    nearest_ob_price,
                    nearest_ob_distance,
                    smc_confluence_score
                FROM stock_screening_metrics
                WHERE ticker = $1
                  AND smc_trend IS NOT NULL
                ORDER BY calculation_date DESC
                LIMIT 1
            """
            row = await self.db.fetchrow(query, ticker)

            if row:
                data = dict(row)
                return {k: v for k, v in data.items() if v is not None}
        except Exception as e:
            logger.debug(f"Could not fetch SMC data for {ticker}: {e}")

        return {}

    async def get_claude_analyzed_signals(
        self,
        min_grade: str = None,
        days_back: int = 7,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get signals that have been analyzed by Claude.

        Args:
            min_grade: Minimum Claude grade ('A+', 'A', 'B', etc.)
            days_back: How many days back to look
            limit: Maximum number of signals

        Returns:
            List of signal dictionaries with Claude analysis
        """
        grade_filter = ""
        if min_grade:
            grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
            min_grade_value = grade_order.get(min_grade, 1)
            valid_grades = [g for g, v in grade_order.items() if v >= min_grade_value]
            grades_str = ', '.join(f"'{g}'" for g in valid_grades)
            grade_filter = f"AND claude_grade IN ({grades_str})"

        query = f"""
            SELECT * FROM stock_scanner_signals
            WHERE claude_analyzed_at IS NOT NULL
              AND signal_timestamp >= NOW() - INTERVAL '{days_back} days'
              {grade_filter}
            ORDER BY claude_score DESC, composite_score DESC
            LIMIT $1
        """

        rows = await self.db.fetch(query, limit)
        return [dict(r) for r in rows]

    def get_claude_stats(self) -> Dict[str, Any]:
        """Get Claude analysis usage statistics"""
        if not CLAUDE_AVAILABLE:
            return {'available': False, 'error': 'Claude dependencies not installed'}

        analyzer = StockClaudeAnalyzer()
        return {
            'available': analyzer.is_available,
            **analyzer.get_stats()
        }
