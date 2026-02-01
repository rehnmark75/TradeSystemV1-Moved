"""
Strategy Router Service

Routes signals to the most appropriate strategy based on market regime,
session, and historical performance. All configuration is database-driven.

Features:
- Regime-based strategy routing (trending → SMC Simple, ranging → Ranging Market, etc.)
- Enable/disable controls per strategy
- Backtest-only mode for strategy validation
- Performance-based fitness scoring with automatic confidence modifiers
- Circuit breaker for consecutive losses
- Thread-safe with in-memory caching
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from threading import RLock
from contextlib import contextmanager
from enum import Enum

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode based on fitness score"""
    ACTIVE = "ACTIVE"           # Full trading, confidence boost up to 1.3x
    REDUCED = "REDUCED"         # Trade with 0.7x confidence modifier
    MONITOR_ONLY = "MONITOR_ONLY"  # Log signals but don't execute
    DISABLED = "DISABLED"       # Strategy completely disabled


@dataclass
class StrategyInfo:
    """Information about a registered strategy"""
    strategy_name: str
    is_enabled: bool = False
    is_backtest_only: bool = True
    display_name: str = ""
    description: str = ""
    strategy_type: str = "signal"


@dataclass
class RoutingRule:
    """A single routing rule mapping regime to strategy"""
    regime: str
    strategy_name: str
    priority: int = 100
    session: Optional[str] = None
    volatility_state: Optional[str] = None
    adx_min: Optional[float] = None
    adx_max: Optional[float] = None
    min_win_rate: float = 0.40
    min_sample_size: int = 10
    is_active: bool = True


@dataclass
class StrategyFitness:
    """Fitness score and trading mode for a strategy"""
    strategy_name: str
    regime: str
    fitness_score: float = 0.0
    trading_mode: TradingMode = TradingMode.MONITOR_ONLY
    confidence_modifier: float = 1.0
    sample_size: int = 0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None


@dataclass
class RouterConfig:
    """Global configuration for the strategy router"""

    # Master switch
    multi_strategy_enabled: bool = False
    fallback_to_smc_simple: bool = True

    # Fitness calculation weights (Bayesian formula)
    fitness_weight_win_rate: float = 0.30
    fitness_weight_profit_factor: float = 0.30
    fitness_weight_sharpe: float = 0.20
    fitness_weight_r_multiple: float = 0.20

    # Rolling window weights
    window_7d_weight: float = 0.50
    window_14d_weight: float = 0.30
    window_30d_weight: float = 0.20

    # Trading mode thresholds
    active_min_fitness: float = 0.65
    reduced_min_fitness: float = 0.35

    # Confidence modifiers
    active_confidence_modifier: float = 1.0
    active_max_confidence_boost: float = 1.3
    reduced_confidence_modifier: float = 0.7

    # Circuit breaker
    consecutive_loss_limit: int = 3
    cooldown_hours: int = 2
    max_drawdown_percent: float = 15.0
    max_switches_per_48h: int = 3
    min_hours_between_switches: int = 12

    # Performance requirements
    min_sample_size: int = 10
    min_win_rate: float = 0.40


class StrategyRouterService:
    """
    Database-driven strategy routing service.

    Routes signals to the optimal strategy based on:
    - Market regime (trending, ranging, breakout, high_volatility)
    - Trading session (asian, london, new_york)
    - Historical performance (Bayesian fitness scoring)
    - Enable/disable status

    All configuration is stored in the strategy_config database.
    """

    def __init__(
        self,
        database_url: str = None,
        cache_ttl_seconds: int = 120,
        is_backtest: bool = False
    ):
        """
        Initialize the strategy router service.

        Args:
            database_url: PostgreSQL connection string (defaults to env var)
            cache_ttl_seconds: Cache TTL for configuration
            is_backtest: Whether running in backtest mode (allows backtest-only strategies)
        """
        self.database_url = database_url or self._get_default_database_url()
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.is_backtest = is_backtest

        # Thread-safe cache
        self._lock = RLock()
        self._strategies: Dict[str, StrategyInfo] = {}
        self._routing_rules: List[RoutingRule] = []
        self._fitness_scores: Dict[str, StrategyFitness] = {}
        self._router_config: Optional[RouterConfig] = None
        self._cache_timestamp: Optional[datetime] = None

        # Load initial configuration
        self._load_config()

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

    def _should_refresh(self) -> bool:
        """Check if cache has expired"""
        if self._cache_timestamp is None:
            return True
        return datetime.now() - self._cache_timestamp > self.cache_ttl

    def _load_config(self):
        """Load all configuration from database"""
        with self._lock:
            try:
                self._load_strategies()
                self._load_routing_rules()
                self._load_router_config()
                self._load_fitness_scores()
                self._cache_timestamp = datetime.now()
                logger.info(
                    f"Strategy router loaded: {len(self._strategies)} strategies, "
                    f"{len(self._routing_rules)} routing rules"
                )
            except Exception as e:
                logger.error(f"Failed to load strategy router config: {e}")
                # Use defaults if database unavailable
                self._use_defaults()

    def _use_defaults(self):
        """Set default configuration when database unavailable"""
        self._strategies = {
            'SMC_SIMPLE': StrategyInfo(
                strategy_name='SMC_SIMPLE',
                is_enabled=True,
                is_backtest_only=False,
                display_name='SMC Simple',
                description='Primary trending strategy'
            )
        }
        self._routing_rules = [
            RoutingRule(regime='trending', strategy_name='SMC_SIMPLE', priority=10),
            RoutingRule(regime='ranging', strategy_name='SMC_SIMPLE', priority=100),
        ]
        self._router_config = RouterConfig()
        self._cache_timestamp = datetime.now()
        logger.warning("Using default strategy router configuration")

    def _load_strategies(self):
        """Load enabled strategies from database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT strategy_name, is_enabled, is_backtest_only,
                               display_name, description, strategy_type
                        FROM enabled_strategies
                    """)
                    rows = cur.fetchall()

                    self._strategies = {}
                    for row in rows:
                        self._strategies[row['strategy_name']] = StrategyInfo(
                            strategy_name=row['strategy_name'],
                            is_enabled=row['is_enabled'],
                            is_backtest_only=row['is_backtest_only'],
                            display_name=row['display_name'] or row['strategy_name'],
                            description=row['description'] or '',
                            strategy_type=row['strategy_type'] or 'signal'
                        )
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            raise

    def _load_routing_rules(self):
        """Load routing rules from database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT regime, session, volatility_state, adx_min, adx_max,
                               strategy_name, priority, min_win_rate, min_sample_size, is_active
                        FROM strategy_routing_rules
                        WHERE is_active = TRUE
                        ORDER BY priority ASC
                    """)
                    rows = cur.fetchall()

                    self._routing_rules = []
                    for row in rows:
                        self._routing_rules.append(RoutingRule(
                            regime=row['regime'],
                            strategy_name=row['strategy_name'],
                            priority=row['priority'],
                            session=row['session'],
                            volatility_state=row['volatility_state'],
                            adx_min=float(row['adx_min']) if row['adx_min'] else None,
                            adx_max=float(row['adx_max']) if row['adx_max'] else None,
                            min_win_rate=float(row['min_win_rate']) if row['min_win_rate'] else 0.40,
                            min_sample_size=row['min_sample_size'] or 10,
                            is_active=row['is_active']
                        ))
        except Exception as e:
            logger.error(f"Error loading routing rules: {e}")
            raise

    def _load_router_config(self):
        """Load global router configuration"""
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
                        self._router_config = RouterConfig(
                            multi_strategy_enabled=row['multi_strategy_enabled'],
                            fallback_to_smc_simple=row['fallback_to_smc_simple'],
                            fitness_weight_win_rate=float(row['fitness_weight_win_rate']),
                            fitness_weight_profit_factor=float(row['fitness_weight_profit_factor']),
                            fitness_weight_sharpe=float(row['fitness_weight_sharpe']),
                            fitness_weight_r_multiple=float(row['fitness_weight_r_multiple']),
                            window_7d_weight=float(row['window_7d_weight']),
                            window_14d_weight=float(row['window_14d_weight']),
                            window_30d_weight=float(row['window_30d_weight']),
                            active_min_fitness=float(row['active_min_fitness']),
                            reduced_min_fitness=float(row['reduced_min_fitness']),
                            active_confidence_modifier=float(row['active_confidence_modifier']),
                            active_max_confidence_boost=float(row['active_max_confidence_boost']),
                            reduced_confidence_modifier=float(row['reduced_confidence_modifier']),
                            consecutive_loss_limit=row['consecutive_loss_limit'],
                            cooldown_hours=row['cooldown_hours'],
                            max_drawdown_percent=float(row['max_drawdown_percent']),
                            max_switches_per_48h=row['max_switches_per_48h'],
                            min_hours_between_switches=row['min_hours_between_switches'],
                            min_sample_size=row['min_sample_size'],
                            min_win_rate=float(row['min_win_rate'])
                        )
                    else:
                        self._router_config = RouterConfig()
        except Exception as e:
            logger.warning(f"Error loading router config, using defaults: {e}")
            self._router_config = RouterConfig()

    def _load_fitness_scores(self):
        """Load fitness scores from database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("""
                        SELECT strategy_name, regime, volatility_state, session,
                               fitness_score, sample_size, trading_mode,
                               confidence_modifier, consecutive_losses, cooldown_until
                        FROM regime_fitness_scores
                    """)
                    rows = cur.fetchall()

                    self._fitness_scores = {}
                    for row in rows:
                        key = f"{row['strategy_name']}:{row['regime']}:{row['volatility_state']}:{row['session']}"
                        self._fitness_scores[key] = StrategyFitness(
                            strategy_name=row['strategy_name'],
                            regime=row['regime'],
                            fitness_score=float(row['fitness_score']) if row['fitness_score'] else 0.0,
                            trading_mode=TradingMode(row['trading_mode']) if row['trading_mode'] else TradingMode.MONITOR_ONLY,
                            confidence_modifier=float(row['confidence_modifier']) if row['confidence_modifier'] else 1.0,
                            sample_size=row['sample_size'] or 0,
                            consecutive_losses=row['consecutive_losses'] or 0,
                            cooldown_until=row['cooldown_until']
                        )
        except Exception as e:
            logger.warning(f"Error loading fitness scores: {e}")
            self._fitness_scores = {}

    # =========================================================================
    # PUBLIC API: Strategy Routing
    # =========================================================================

    def get_config(self, force_refresh: bool = False) -> RouterConfig:
        """Get current router configuration"""
        with self._lock:
            if force_refresh or self._should_refresh():
                self._load_config()
            return self._router_config or RouterConfig()

    def is_multi_strategy_enabled(self) -> bool:
        """Check if multi-strategy routing is enabled"""
        config = self.get_config()
        return config.multi_strategy_enabled

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """
        Check if a strategy is enabled for the current mode (live/backtest).

        In backtest mode, strategies with is_backtest_only=True are allowed.
        In live mode, only strategies with is_backtest_only=False are allowed.
        """
        with self._lock:
            if self._should_refresh():
                self._load_config()

            if strategy_name not in self._strategies:
                return False

            strategy = self._strategies[strategy_name]

            if not strategy.is_enabled:
                return False

            # In backtest mode, allow backtest-only strategies
            if self.is_backtest:
                return True

            # In live mode, reject backtest-only strategies
            return not strategy.is_backtest_only

    def is_strategy_backtest_only(self, strategy_name: str) -> bool:
        """Check if a strategy is in backtest-only mode"""
        with self._lock:
            if strategy_name not in self._strategies:
                return True  # Unknown strategies are backtest-only
            return self._strategies[strategy_name].is_backtest_only

    def get_strategy_for_regime(
        self,
        regime: str,
        epic: Optional[str] = None,
        session: Optional[str] = None,
        volatility_state: Optional[str] = None,
        adx_value: Optional[float] = None
    ) -> Tuple[Optional[str], float]:
        """
        Get the best strategy for the current market conditions.

        Args:
            regime: Market regime (trending, ranging, breakout, high_volatility, low_volatility)
            epic: Trading pair epic (for pair-specific overrides)
            session: Trading session (asian, london, new_york)
            volatility_state: Volatility level (high, normal, low)
            adx_value: Current ADX value

        Returns:
            Tuple of (strategy_name, confidence_modifier)
        """
        with self._lock:
            if self._should_refresh():
                self._load_config()

            # If multi-strategy is disabled, always return SMC_SIMPLE
            if not self._router_config.multi_strategy_enabled:
                return ('SMC_SIMPLE', 1.0)

            # Find matching routing rules
            candidates = self._get_matching_rules(regime, session, volatility_state, adx_value)

            if not candidates:
                # Fallback to SMC_SIMPLE
                if self._router_config.fallback_to_smc_simple:
                    logger.debug(f"No routing rule for regime={regime}, falling back to SMC_SIMPLE")
                    return ('SMC_SIMPLE', 1.0)
                return (None, 0.0)

            # Evaluate candidates based on fitness and enable status
            for rule in candidates:
                strategy_name = rule.strategy_name

                # Check if strategy is enabled for current mode
                if not self.is_strategy_enabled(strategy_name):
                    continue

                # Check fitness/cooldown
                fitness = self._get_fitness(strategy_name, regime, volatility_state, session)

                # Skip if in cooldown
                if fitness.cooldown_until and datetime.now() < fitness.cooldown_until:
                    logger.debug(f"Strategy {strategy_name} in cooldown until {fitness.cooldown_until}")
                    continue

                # Skip if trading mode is DISABLED
                if fitness.trading_mode == TradingMode.DISABLED:
                    continue

                # Check minimum sample size and win rate
                if fitness.sample_size >= rule.min_sample_size:
                    if fitness.win_rate is not None and fitness.win_rate < rule.min_win_rate:
                        logger.debug(
                            f"Strategy {strategy_name} win rate {fitness.win_rate:.1%} "
                            f"below threshold {rule.min_win_rate:.1%}"
                        )
                        continue

                # Calculate confidence modifier
                confidence_modifier = self._calculate_confidence_modifier(fitness)

                logger.debug(
                    f"Selected strategy {strategy_name} for regime={regime}, "
                    f"fitness={fitness.fitness_score:.2f}, modifier={confidence_modifier:.2f}"
                )

                return (strategy_name, confidence_modifier)

            # No valid strategy found, fallback
            if self._router_config.fallback_to_smc_simple:
                return ('SMC_SIMPLE', 1.0)
            return (None, 0.0)

    def _get_matching_rules(
        self,
        regime: str,
        session: Optional[str],
        volatility_state: Optional[str],
        adx_value: Optional[float]
    ) -> List[RoutingRule]:
        """Get routing rules matching the current conditions, sorted by priority"""
        matches = []

        for rule in self._routing_rules:
            # Must match regime
            if rule.regime != regime:
                continue

            # Session filter (None means any session)
            if rule.session is not None and rule.session != session:
                continue

            # Volatility state filter
            if rule.volatility_state is not None and rule.volatility_state != volatility_state:
                continue

            # ADX range filter
            if adx_value is not None:
                if rule.adx_min is not None and adx_value < rule.adx_min:
                    continue
                if rule.adx_max is not None and adx_value > rule.adx_max:
                    continue

            matches.append(rule)

        # Sort by priority (lower = higher priority)
        matches.sort(key=lambda r: r.priority)
        return matches

    def _get_fitness(
        self,
        strategy_name: str,
        regime: str,
        volatility_state: Optional[str],
        session: Optional[str]
    ) -> StrategyFitness:
        """Get fitness score for a strategy in given conditions"""
        key = f"{strategy_name}:{regime}:{volatility_state}:{session}"

        if key in self._fitness_scores:
            return self._fitness_scores[key]

        # Try without session
        key_no_session = f"{strategy_name}:{regime}:{volatility_state}:None"
        if key_no_session in self._fitness_scores:
            return self._fitness_scores[key_no_session]

        # Try without volatility
        key_no_vol = f"{strategy_name}:{regime}:None:None"
        if key_no_vol in self._fitness_scores:
            return self._fitness_scores[key_no_vol]

        # Return default (MONITOR_ONLY for new strategy/regime combos)
        return StrategyFitness(
            strategy_name=strategy_name,
            regime=regime,
            trading_mode=TradingMode.MONITOR_ONLY
        )

    def _calculate_confidence_modifier(self, fitness: StrategyFitness) -> float:
        """Calculate confidence modifier based on fitness score and trading mode"""
        config = self._router_config or RouterConfig()

        if fitness.trading_mode == TradingMode.DISABLED:
            return 0.0

        if fitness.trading_mode == TradingMode.MONITOR_ONLY:
            return 0.0  # Signals logged but not executed

        if fitness.trading_mode == TradingMode.REDUCED:
            return config.reduced_confidence_modifier

        # ACTIVE mode - apply fitness-based boost
        if fitness.fitness_score >= config.active_min_fitness:
            # Scale boost based on fitness (0.65-1.0 → 1.0-1.3)
            boost_range = config.active_max_confidence_boost - config.active_confidence_modifier
            fitness_range = 1.0 - config.active_min_fitness

            if fitness_range > 0:
                normalized = (fitness.fitness_score - config.active_min_fitness) / fitness_range
                return config.active_confidence_modifier + (boost_range * normalized)

            return config.active_confidence_modifier

        return config.active_confidence_modifier

    def should_execute_trade(
        self,
        strategy_name: str,
        regime: Optional[str] = None,
        volatility_state: Optional[str] = None,
        session: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be executed based on strategy status.

        Returns:
            Tuple of (should_execute, reason)
        """
        # Check if strategy is enabled
        if not self.is_strategy_enabled(strategy_name):
            if strategy_name in self._strategies:
                if self._strategies[strategy_name].is_backtest_only and not self.is_backtest:
                    return (False, f"Strategy {strategy_name} is backtest-only")
            return (False, f"Strategy {strategy_name} is disabled")

        # Check fitness/trading mode
        if regime:
            fitness = self._get_fitness(strategy_name, regime, volatility_state, session)

            if fitness.trading_mode == TradingMode.DISABLED:
                return (False, f"Strategy {strategy_name} is DISABLED for {regime}")

            if fitness.trading_mode == TradingMode.MONITOR_ONLY:
                return (False, f"Strategy {strategy_name} is MONITOR_ONLY for {regime}")

            if fitness.cooldown_until and datetime.now() < fitness.cooldown_until:
                return (False, f"Strategy {strategy_name} in cooldown until {fitness.cooldown_until}")

        return (True, "OK")

    # =========================================================================
    # PUBLIC API: Strategy Information
    # =========================================================================

    def get_all_strategies(self) -> Dict[str, StrategyInfo]:
        """Get all registered strategies"""
        with self._lock:
            if self._should_refresh():
                self._load_config()
            return dict(self._strategies)

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names"""
        with self._lock:
            if self._should_refresh():
                self._load_config()
            return [
                name for name, info in self._strategies.items()
                if self.is_strategy_enabled(name)
            ]

    def get_routing_rules_for_regime(self, regime: str) -> List[RoutingRule]:
        """Get all routing rules for a specific regime"""
        with self._lock:
            if self._should_refresh():
                self._load_config()
            return [r for r in self._routing_rules if r.regime == regime]

    # =========================================================================
    # PUBLIC API: Performance Tracking (for updates from trade outcomes)
    # =========================================================================

    def record_trade_outcome(
        self,
        strategy_name: str,
        epic: str,
        regime: str,
        is_win: bool,
        pips: float,
        volatility_state: str = None,
        session: str = None
    ):
        """
        Record a trade outcome to update fitness scores.

        This is called when a trade closes to update the performance tracking.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Update performance in the appropriate window tables
                    # This is a simplified version - full implementation would
                    # update rolling window statistics

                    # Check for consecutive losses
                    key = f"{strategy_name}:{regime}:{volatility_state}:{session}"

                    if not is_win:
                        # Increment consecutive losses
                        cur.execute("""
                            UPDATE regime_fitness_scores
                            SET consecutive_losses = consecutive_losses + 1,
                                last_updated = NOW()
                            WHERE strategy_name = %s AND regime = %s
                            AND (volatility_state = %s OR (volatility_state IS NULL AND %s IS NULL))
                            AND (session = %s OR (session IS NULL AND %s IS NULL))
                        """, (strategy_name, regime, volatility_state, volatility_state, session, session))

                        # Check if we need to trigger cooldown
                        config = self.get_config()
                        if key in self._fitness_scores:
                            fitness = self._fitness_scores[key]
                            if fitness.consecutive_losses >= config.consecutive_loss_limit:
                                cooldown_until = datetime.now() + timedelta(hours=config.cooldown_hours)
                                cur.execute("""
                                    UPDATE regime_fitness_scores
                                    SET cooldown_until = %s
                                    WHERE strategy_name = %s AND regime = %s
                                """, (cooldown_until, strategy_name, regime))
                                logger.warning(
                                    f"Strategy {strategy_name} entering cooldown for {regime} "
                                    f"after {fitness.consecutive_losses + 1} consecutive losses"
                                )
                    else:
                        # Reset consecutive losses on win
                        cur.execute("""
                            UPDATE regime_fitness_scores
                            SET consecutive_losses = 0,
                                last_updated = NOW()
                            WHERE strategy_name = %s AND regime = %s
                        """, (strategy_name, regime))

                    conn.commit()

        except Exception as e:
            logger.error(f"Error recording trade outcome: {e}")

    def log_strategy_switch(
        self,
        epic: str,
        from_strategy: str,
        to_strategy: str,
        regime: str,
        session: str = None,
        adx_value: float = None,
        volatility_state: str = None,
        switch_reason: str = None
    ):
        """Log a strategy switch for audit purposes"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get fitness scores
                    from_fitness = self._get_fitness(from_strategy, regime, volatility_state, session).fitness_score if from_strategy else None
                    to_fitness = self._get_fitness(to_strategy, regime, volatility_state, session).fitness_score

                    cur.execute("""
                        INSERT INTO strategy_switch_log
                        (epic, from_strategy, to_strategy, regime, session,
                         adx_value, volatility_state, from_fitness, to_fitness, switch_reason)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (epic, from_strategy, to_strategy, regime, session,
                          adx_value, volatility_state, from_fitness, to_fitness, switch_reason))
                    conn.commit()

        except Exception as e:
            logger.error(f"Error logging strategy switch: {e}")


# =============================================================================
# Singleton instance and convenience functions
# =============================================================================

_router_service: Optional[StrategyRouterService] = None
_router_lock = RLock()


def get_strategy_router(
    database_url: str = None,
    is_backtest: bool = False,
    force_new: bool = False
) -> StrategyRouterService:
    """
    Get singleton instance of the strategy router service.

    Args:
        database_url: Database URL (defaults to env var)
        is_backtest: Whether running in backtest mode
        force_new: Force creation of new instance

    Returns:
        StrategyRouterService instance
    """
    global _router_service

    with _router_lock:
        if _router_service is None or force_new:
            _router_service = StrategyRouterService(
                database_url=database_url,
                is_backtest=is_backtest
            )
        elif _router_service.is_backtest != is_backtest:
            # Mode changed, recreate
            _router_service = StrategyRouterService(
                database_url=database_url,
                is_backtest=is_backtest
            )

        return _router_service


def get_strategy_for_conditions(
    regime: str,
    epic: Optional[str] = None,
    session: Optional[str] = None,
    volatility_state: Optional[str] = None,
    adx_value: Optional[float] = None,
    is_backtest: bool = False
) -> Tuple[Optional[str], float]:
    """
    Convenience function to get the best strategy for given conditions.

    Returns:
        Tuple of (strategy_name, confidence_modifier)
    """
    router = get_strategy_router(is_backtest=is_backtest)
    return router.get_strategy_for_regime(
        regime=regime,
        epic=epic,
        session=session,
        volatility_state=volatility_state,
        adx_value=adx_value
    )
