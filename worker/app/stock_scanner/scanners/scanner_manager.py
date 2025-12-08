"""
Scanner Manager

Orchestrates all signal scanners and provides unified interface for:
- Running all scanners
- Combining and ranking results
- Performance tracking
- Signal management
- Export functionality
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from collections import defaultdict

from .base_scanner import BaseScanner, SignalSetup, ScannerConfig, QualityTier
from .scoring import SignalScorer
from .strategies import (
    TrendMomentumScanner,
    BreakoutConfirmationScanner,
    MeanReversionScanner,
    GapAndGoScanner,
)

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
