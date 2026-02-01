"""
Strategy Performance Tracker

Tracks and calculates performance metrics for the multi-strategy routing system.
Implements Bayesian fitness scoring with rolling windows for self-tuning behavior.

Features:
- Rolling window performance (7, 14, 30 days)
- Bayesian fitness calculation
- Automatic trading mode adjustments
- Circuit breaker for consecutive losses
- Cooldown and rehabilitation logic

All data is stored in and loaded from the strategy_config database.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from threading import RLock
from contextlib import contextmanager
from decimal import Decimal
from enum import Enum

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode based on fitness score"""
    ACTIVE = "ACTIVE"
    REDUCED = "REDUCED"
    MONITOR_ONLY = "MONITOR_ONLY"
    DISABLED = "DISABLED"


@dataclass
class WindowPerformance:
    """Performance metrics for a specific rolling window"""
    window_days: int
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown_pips: float = 0.0
    sharpe_ratio: Optional[float] = None
    r_multiple: Optional[float] = None

    @property
    def has_min_trades(self) -> bool:
        """Check if window has minimum trades for reliable metrics"""
        return self.total_trades >= 10


@dataclass
class StrategyPerformance:
    """Aggregated performance across all windows for a strategy/regime combo"""
    strategy_name: str
    epic: str
    regime: str
    window_7d: Optional[WindowPerformance] = None
    window_14d: Optional[WindowPerformance] = None
    window_30d: Optional[WindowPerformance] = None

    # Calculated fitness
    fitness_score: float = 0.0
    trading_mode: TradingMode = TradingMode.MONITOR_ONLY
    confidence_modifier: float = 1.0

    # Circuit breaker
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None

    # Timestamps
    last_trade_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class FitnessConfig:
    """Configuration for fitness calculation"""
    # Component weights (must sum to 1.0)
    weight_win_rate: float = 0.30
    weight_profit_factor: float = 0.30
    weight_sharpe: float = 0.20
    weight_r_multiple: float = 0.20

    # Window weights (must sum to 1.0)
    weight_7d: float = 0.50
    weight_14d: float = 0.30
    weight_30d: float = 0.20

    # Trading mode thresholds
    active_threshold: float = 0.65
    reduced_threshold: float = 0.35

    # Confidence modifiers
    active_base_modifier: float = 1.0
    active_max_boost: float = 1.3
    reduced_modifier: float = 0.7

    # Circuit breaker
    consecutive_loss_limit: int = 3
    cooldown_hours: int = 2


class StrategyPerformanceTracker:
    """
    Tracks strategy performance and calculates fitness scores.

    Implements the self-tuning behavior by:
    1. Recording trade outcomes per strategy/epic/regime
    2. Calculating rolling window statistics (7, 14, 30 days)
    3. Computing Bayesian fitness scores
    4. Updating trading modes and confidence modifiers
    5. Triggering cooldowns on consecutive losses
    """

    def __init__(
        self,
        database_url: str = None,
        fitness_config: FitnessConfig = None
    ):
        """
        Initialize the performance tracker.

        Args:
            database_url: PostgreSQL connection string
            fitness_config: Configuration for fitness calculation
        """
        self.database_url = database_url or self._get_default_database_url()
        self.fitness_config = fitness_config or self._load_fitness_config()

        self._lock = RLock()

    def _get_default_database_url(self) -> str:
        """Get database URL from environment"""
        return os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            yield conn
        finally:
            if conn:
                conn.close()

    def _load_fitness_config(self) -> FitnessConfig:
        """Load fitness configuration from database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM router_global_config
                        WHERE is_active = TRUE
                        ORDER BY id DESC LIMIT 1
                    """)
                    row = cur.fetchone()

                    if row:
                        return FitnessConfig(
                            weight_win_rate=float(row['fitness_weight_win_rate']),
                            weight_profit_factor=float(row['fitness_weight_profit_factor']),
                            weight_sharpe=float(row['fitness_weight_sharpe']),
                            weight_r_multiple=float(row['fitness_weight_r_multiple']),
                            weight_7d=float(row['window_7d_weight']),
                            weight_14d=float(row['window_14d_weight']),
                            weight_30d=float(row['window_30d_weight']),
                            active_threshold=float(row['active_min_fitness']),
                            reduced_threshold=float(row['reduced_min_fitness']),
                            active_base_modifier=float(row['active_confidence_modifier']),
                            active_max_boost=float(row['active_max_confidence_boost']),
                            reduced_modifier=float(row['reduced_confidence_modifier']),
                            consecutive_loss_limit=row['consecutive_loss_limit'],
                            cooldown_hours=row['cooldown_hours']
                        )
        except Exception as e:
            logger.warning(f"Failed to load fitness config, using defaults: {e}")

        return FitnessConfig()

    # =========================================================================
    # Trade Recording
    # =========================================================================

    def record_trade_outcome(
        self,
        strategy_name: str,
        epic: str,
        regime: str,
        is_win: bool,
        pips: float,
        r_multiple: Optional[float] = None,
        volatility_state: Optional[str] = None,
        session: Optional[str] = None,
        trade_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Record a trade outcome and update performance metrics.

        Args:
            strategy_name: Name of the strategy that generated the trade
            epic: Trading pair epic
            regime: Market regime when trade was taken
            is_win: Whether the trade was profitable
            pips: Profit/loss in pips (positive for wins, negative for losses)
            r_multiple: Risk-adjusted return (optional)
            volatility_state: Volatility level (high, normal, low)
            session: Trading session (asian, london, new_york)
            trade_timestamp: When the trade closed (defaults to now)

        Returns:
            True if recorded successfully
        """
        timestamp = trade_timestamp or datetime.now()

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Update rolling window performance
                    for window_days in [7, 14, 30]:
                        self._update_window_performance(
                            cur, strategy_name, epic, regime, window_days,
                            is_win, pips, r_multiple, timestamp
                        )

                    # Update consecutive losses and check cooldown
                    self._update_consecutive_losses(
                        cur, strategy_name, epic, regime,
                        is_win, volatility_state, session
                    )

                    # Recalculate fitness score
                    self._recalculate_fitness(
                        cur, strategy_name, epic, regime,
                        volatility_state, session
                    )

                    conn.commit()
                    logger.debug(
                        f"Recorded trade outcome: {strategy_name}/{epic}/{regime} "
                        f"{'WIN' if is_win else 'LOSS'} {pips:+.1f} pips"
                    )
                    return True

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")
            return False

    def _update_window_performance(
        self,
        cur,
        strategy_name: str,
        epic: str,
        regime: str,
        window_days: int,
        is_win: bool,
        pips: float,
        r_multiple: Optional[float],
        timestamp: datetime
    ):
        """Update performance metrics for a specific rolling window"""

        # Calculate period boundaries
        period_end = timestamp
        period_start = timestamp - timedelta(days=window_days)

        # Upsert performance record
        cur.execute("""
            INSERT INTO strategy_regime_performance
            (strategy_name, epic, regime, window_days, total_trades,
             winning_trades, losing_trades, total_pips, last_trade_at,
             period_start, period_end, created_at, updated_at)
            VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (strategy_name, epic, regime, window_days)
            DO UPDATE SET
                total_trades = strategy_regime_performance.total_trades + 1,
                winning_trades = strategy_regime_performance.winning_trades + %s,
                losing_trades = strategy_regime_performance.losing_trades + %s,
                total_pips = strategy_regime_performance.total_pips + %s,
                last_trade_at = %s,
                period_end = %s,
                updated_at = NOW()
        """, (
            strategy_name, epic, regime, window_days,
            1 if is_win else 0,  # winning_trades for insert
            0 if is_win else 1,  # losing_trades for insert
            pips,
            timestamp, period_start, period_end,
            # For update
            1 if is_win else 0,
            0 if is_win else 1,
            pips,
            timestamp, period_end
        ))

        # Recalculate derived metrics (win rate, profit factor)
        cur.execute("""
            UPDATE strategy_regime_performance
            SET
                win_rate = CASE
                    WHEN total_trades > 0 THEN winning_trades::DECIMAL / total_trades
                    ELSE NULL
                END,
                avg_win_pips = CASE
                    WHEN winning_trades > 0 THEN total_pips / winning_trades
                    ELSE NULL
                END
            WHERE strategy_name = %s AND epic = %s AND regime = %s AND window_days = %s
        """, (strategy_name, epic, regime, window_days))

    def _update_consecutive_losses(
        self,
        cur,
        strategy_name: str,
        epic: str,
        regime: str,
        is_win: bool,
        volatility_state: Optional[str],
        session: Optional[str]
    ):
        """Update consecutive loss counter and trigger cooldown if needed"""

        if is_win:
            # Reset consecutive losses on win
            cur.execute("""
                UPDATE regime_fitness_scores
                SET consecutive_losses = 0,
                    last_updated = NOW()
                WHERE strategy_name = %s AND regime = %s
            """, (strategy_name, regime))
        else:
            # Increment consecutive losses
            cur.execute("""
                INSERT INTO regime_fitness_scores
                (strategy_name, regime, volatility_state, session,
                 consecutive_losses, last_updated)
                VALUES (%s, %s, %s, %s, 1, NOW())
                ON CONFLICT (strategy_name, regime, volatility_state, session)
                DO UPDATE SET
                    consecutive_losses = regime_fitness_scores.consecutive_losses + 1,
                    last_updated = NOW()
                RETURNING consecutive_losses
            """, (strategy_name, regime, volatility_state, session))

            result = cur.fetchone()
            consecutive_losses = result[0] if result else 1

            # Check if cooldown should be triggered
            config = self.fitness_config
            if consecutive_losses >= config.consecutive_loss_limit:
                cooldown_until = datetime.now() + timedelta(hours=config.cooldown_hours)
                cur.execute("""
                    UPDATE regime_fitness_scores
                    SET cooldown_until = %s,
                        trading_mode = 'MONITOR_ONLY'
                    WHERE strategy_name = %s AND regime = %s
                """, (cooldown_until, strategy_name, regime))

                logger.warning(
                    f"Circuit breaker triggered: {strategy_name}/{regime} "
                    f"cooldown until {cooldown_until} after {consecutive_losses} losses"
                )

    def _recalculate_fitness(
        self,
        cur,
        strategy_name: str,
        epic: str,
        regime: str,
        volatility_state: Optional[str],
        session: Optional[str]
    ):
        """Recalculate fitness score using Bayesian formula"""

        config = self.fitness_config

        # Get performance for all windows
        windows = {}
        for window_days in [7, 14, 30]:
            cur.execute("""
                SELECT win_rate, profit_factor, sharpe_ratio, r_multiple, total_trades
                FROM strategy_regime_performance
                WHERE strategy_name = %s AND epic = %s AND regime = %s AND window_days = %s
            """, (strategy_name, epic, regime, window_days))

            row = cur.fetchone()
            if row:
                windows[window_days] = {
                    'win_rate': float(row[0]) if row[0] else 0.0,
                    'profit_factor': float(row[1]) if row[1] else 0.0,
                    'sharpe_ratio': float(row[2]) if row[2] else 0.0,
                    'r_multiple': float(row[3]) if row[3] else 0.0,
                    'total_trades': row[4] or 0
                }

        if not windows:
            return  # No data to calculate fitness

        # Calculate weighted fitness score
        fitness_score = self._calculate_bayesian_fitness(windows, config)

        # Determine trading mode based on fitness
        trading_mode = self._determine_trading_mode(fitness_score, config)

        # Calculate confidence modifier
        confidence_modifier = self._calculate_confidence_modifier(fitness_score, trading_mode, config)

        # Update fitness scores table
        cur.execute("""
            INSERT INTO regime_fitness_scores
            (strategy_name, regime, volatility_state, session,
             fitness_score, sample_size, trading_mode, confidence_modifier, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (strategy_name, regime, volatility_state, session)
            DO UPDATE SET
                fitness_score = EXCLUDED.fitness_score,
                sample_size = EXCLUDED.sample_size,
                trading_mode = EXCLUDED.trading_mode,
                confidence_modifier = EXCLUDED.confidence_modifier,
                last_updated = NOW()
        """, (
            strategy_name, regime, volatility_state, session,
            fitness_score,
            sum(w.get('total_trades', 0) for w in windows.values()),
            trading_mode.value,
            confidence_modifier
        ))

    def _calculate_bayesian_fitness(
        self,
        windows: Dict[int, dict],
        config: FitnessConfig
    ) -> float:
        """
        Calculate Bayesian fitness score.

        Formula:
        fitness = w1×win_rate + w2×norm_profit_factor + w3×sharpe + w4×r_multiple

        Where each component is weighted across rolling windows:
        component = window_7d×0.5 + window_14d×0.3 + window_30d×0.2
        """

        def weighted_average(key: str) -> float:
            """Calculate weighted average across windows"""
            total = 0.0
            weight_sum = 0.0

            for days, weight in [(7, config.weight_7d), (14, config.weight_14d), (30, config.weight_30d)]:
                if days in windows and windows[days].get('total_trades', 0) >= 5:
                    total += windows[days].get(key, 0.0) * weight
                    weight_sum += weight

            return total / weight_sum if weight_sum > 0 else 0.0

        # Get weighted components
        win_rate = weighted_average('win_rate')
        profit_factor = weighted_average('profit_factor')
        sharpe = weighted_average('sharpe_ratio')
        r_multiple = weighted_average('r_multiple')

        # Normalize profit factor (cap at 5.0 for scoring)
        norm_pf = min(profit_factor / 5.0, 1.0) if profit_factor > 0 else 0.0

        # Normalize sharpe (cap at 3.0 for scoring)
        norm_sharpe = min(max(sharpe, 0) / 3.0, 1.0)

        # Normalize r_multiple (cap at 3.0 for scoring)
        norm_r = min(max(r_multiple, 0) / 3.0, 1.0)

        # Calculate final fitness
        fitness = (
            config.weight_win_rate * win_rate +
            config.weight_profit_factor * norm_pf +
            config.weight_sharpe * norm_sharpe +
            config.weight_r_multiple * norm_r
        )

        return min(max(fitness, 0.0), 1.0)  # Clamp to [0, 1]

    def _determine_trading_mode(
        self,
        fitness_score: float,
        config: FitnessConfig
    ) -> TradingMode:
        """Determine trading mode based on fitness score"""
        if fitness_score >= config.active_threshold:
            return TradingMode.ACTIVE
        elif fitness_score >= config.reduced_threshold:
            return TradingMode.REDUCED
        else:
            return TradingMode.MONITOR_ONLY

    def _calculate_confidence_modifier(
        self,
        fitness_score: float,
        trading_mode: TradingMode,
        config: FitnessConfig
    ) -> float:
        """Calculate confidence modifier based on fitness and trading mode"""
        if trading_mode == TradingMode.DISABLED:
            return 0.0

        if trading_mode == TradingMode.MONITOR_ONLY:
            return 0.0

        if trading_mode == TradingMode.REDUCED:
            return config.reduced_modifier

        # ACTIVE mode - scale boost based on fitness
        fitness_range = 1.0 - config.active_threshold
        if fitness_range > 0:
            normalized = (fitness_score - config.active_threshold) / fitness_range
            boost_range = config.active_max_boost - config.active_base_modifier
            return config.active_base_modifier + (boost_range * normalized)

        return config.active_base_modifier

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_strategy_performance(
        self,
        strategy_name: str,
        epic: str,
        regime: str
    ) -> Optional[StrategyPerformance]:
        """Get full performance data for a strategy/epic/regime combination"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    perf = StrategyPerformance(
                        strategy_name=strategy_name,
                        epic=epic,
                        regime=regime
                    )

                    # Load window performance
                    for window_days in [7, 14, 30]:
                        cur.execute("""
                            SELECT total_trades, winning_trades, losing_trades,
                                   win_rate, profit_factor, total_pips,
                                   avg_win_pips, avg_loss_pips, max_drawdown_pips,
                                   sharpe_ratio, r_multiple, last_trade_at
                            FROM strategy_regime_performance
                            WHERE strategy_name = %s AND epic = %s
                            AND regime = %s AND window_days = %s
                        """, (strategy_name, epic, regime, window_days))

                        row = cur.fetchone()
                        if row:
                            window = WindowPerformance(
                                window_days=window_days,
                                total_trades=row['total_trades'] or 0,
                                winning_trades=row['winning_trades'] or 0,
                                losing_trades=row['losing_trades'] or 0,
                                total_pips=float(row['total_pips'] or 0),
                                avg_win_pips=float(row['avg_win_pips'] or 0),
                                avg_loss_pips=float(row['avg_loss_pips'] or 0),
                                win_rate=float(row['win_rate']) if row['win_rate'] else None,
                                profit_factor=float(row['profit_factor']) if row['profit_factor'] else None,
                                max_drawdown_pips=float(row['max_drawdown_pips'] or 0),
                                sharpe_ratio=float(row['sharpe_ratio']) if row['sharpe_ratio'] else None,
                                r_multiple=float(row['r_multiple']) if row['r_multiple'] else None
                            )

                            if window_days == 7:
                                perf.window_7d = window
                            elif window_days == 14:
                                perf.window_14d = window
                            else:
                                perf.window_30d = window

                            if row['last_trade_at']:
                                perf.last_trade_at = row['last_trade_at']

                    # Load fitness score
                    cur.execute("""
                        SELECT fitness_score, trading_mode, confidence_modifier,
                               consecutive_losses, cooldown_until
                        FROM regime_fitness_scores
                        WHERE strategy_name = %s AND regime = %s
                        LIMIT 1
                    """, (strategy_name, regime))

                    fitness_row = cur.fetchone()
                    if fitness_row:
                        perf.fitness_score = float(fitness_row['fitness_score'] or 0)
                        perf.trading_mode = TradingMode(fitness_row['trading_mode']) if fitness_row['trading_mode'] else TradingMode.MONITOR_ONLY
                        perf.confidence_modifier = float(fitness_row['confidence_modifier'] or 1.0)
                        perf.consecutive_losses = fitness_row['consecutive_losses'] or 0
                        perf.cooldown_until = fitness_row['cooldown_until']

                    return perf

        except Exception as e:
            logger.error(f"Error getting strategy performance: {e}")
            return None

    def get_regime_performance_summary(
        self,
        strategy_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary across all regimes for a strategy.

        Returns:
            Dict mapping regime to performance metrics
        """
        summary: Dict[str, Dict[str, float]] = {}

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT regime, window_days,
                               SUM(total_trades) as total_trades,
                               AVG(win_rate) as win_rate,
                               AVG(profit_factor) as profit_factor,
                               SUM(total_pips) as total_pips
                        FROM strategy_regime_performance
                        WHERE strategy_name = %s
                        GROUP BY regime, window_days
                        ORDER BY regime, window_days
                    """, (strategy_name,))

                    for row in cur.fetchall():
                        regime = row['regime']
                        if regime not in summary:
                            summary[regime] = {}

                        window_key = f"{row['window_days']}d"
                        summary[regime][window_key] = {
                            'trades': row['total_trades'] or 0,
                            'win_rate': float(row['win_rate'] or 0),
                            'profit_factor': float(row['profit_factor'] or 0),
                            'pips': float(row['total_pips'] or 0)
                        }

        except Exception as e:
            logger.error(f"Error getting regime performance summary: {e}")

        return summary

    def reset_cooldown(
        self,
        strategy_name: str,
        regime: str
    ) -> bool:
        """Manually reset cooldown for a strategy/regime"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE regime_fitness_scores
                        SET cooldown_until = NULL,
                            consecutive_losses = 0,
                            trading_mode = 'MONITOR_ONLY'
                        WHERE strategy_name = %s AND regime = %s
                    """, (strategy_name, regime))
                    conn.commit()
                    logger.info(f"Reset cooldown for {strategy_name}/{regime}")
                    return True
        except Exception as e:
            logger.error(f"Error resetting cooldown: {e}")
            return False

    def rehabilitate_strategy(
        self,
        strategy_name: str,
        regime: str,
        new_mode: TradingMode = TradingMode.REDUCED
    ) -> bool:
        """
        Rehabilitate a strategy from MONITOR_ONLY to REDUCED mode.

        Used after a cooldown period to gradually reintroduce a strategy.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE regime_fitness_scores
                        SET trading_mode = %s,
                            cooldown_until = NULL,
                            consecutive_losses = 0
                        WHERE strategy_name = %s AND regime = %s
                        AND (trading_mode = 'MONITOR_ONLY' OR trading_mode = 'DISABLED')
                    """, (new_mode.value, strategy_name, regime))
                    conn.commit()

                    if cur.rowcount > 0:
                        logger.info(
                            f"Rehabilitated {strategy_name}/{regime} to {new_mode.value}"
                        )
                        return True
                    return False
        except Exception as e:
            logger.error(f"Error rehabilitating strategy: {e}")
            return False


# =============================================================================
# Singleton instance
# =============================================================================

_tracker_instance: Optional[StrategyPerformanceTracker] = None
_tracker_lock = RLock()


def get_performance_tracker(
    database_url: str = None,
    force_new: bool = False
) -> StrategyPerformanceTracker:
    """Get singleton instance of the performance tracker"""
    global _tracker_instance

    with _tracker_lock:
        if _tracker_instance is None or force_new:
            _tracker_instance = StrategyPerformanceTracker(database_url=database_url)
        return _tracker_instance
