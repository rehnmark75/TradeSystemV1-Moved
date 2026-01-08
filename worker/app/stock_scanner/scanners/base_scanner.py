"""
Base Scanner Abstract Class

Provides the foundation for all signal scanners with common functionality:
- Database connection management
- Signal generation interface
- Risk level calculation (stop-loss, take-profit)
- Signal persistence

All concrete scanners inherit from BaseScanner and implement:
- scan(): Generate signals for the scanner's strategy
- _calculate_entry_levels(): Calculate entry, stop, and targets
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal direction"""
    BUY = "BUY"
    SELL = "SELL"


class QualityTier(Enum):
    """Signal quality tiers"""
    A_PLUS = "A+"  # 85-100: Highest conviction
    A = "A"        # 70-84: High conviction
    B = "B"        # 60-69: Medium conviction
    C = "C"        # 50-59: Lower conviction
    D = "D"        # <50: Speculative

    @classmethod
    def from_score(cls, score: int) -> 'QualityTier':
        """Determine tier from composite score"""
        if score >= 85:
            return cls.A_PLUS
        elif score >= 70:
            return cls.A
        elif score >= 60:
            return cls.B
        elif score >= 50:
            return cls.C
        else:
            return cls.D


@dataclass
class ScannerConfig:
    """Base configuration for all scanners"""

    # Minimum thresholds
    min_score_threshold: int = 70
    max_signals_per_run: int = 50

    # Risk parameters
    max_risk_per_trade_pct: float = 1.5
    default_rr_ratio: float = 2.0  # Default risk-reward ratio

    # Tier limits (1=best liquidity, 4=lowest)
    max_tier: int = 3  # Exclude tier 4 by default

    # Volume requirements
    min_relative_volume: float = 0.8
    high_volume_threshold: float = 1.5

    # ATR for stop calculation
    atr_stop_multiplier: float = 1.5
    max_stop_loss_pct: float = 8.0
    min_stop_loss_pct: float = 1.0

    # Target calculation
    tp1_rr_ratio: float = 2.0  # First target at 2R
    tp2_rr_ratio: float = 3.0  # Second target at 3R

    # ==========================================================================
    # FUNDAMENTAL FILTERS (from stock_instruments)
    # Set to None to disable filter, or set value to enable
    # ==========================================================================

    # Valuation filters
    max_pe_ratio: Optional[float] = None  # Max P/E ratio (e.g., 50)
    min_pe_ratio: Optional[float] = None  # Min P/E ratio (e.g., 5, avoid negative earnings)
    max_peg_ratio: Optional[float] = None  # Max PEG ratio (e.g., 2.0)

    # Growth filters
    min_earnings_growth: Optional[float] = None  # Min YoY earnings growth % (e.g., 0.10 for 10%)
    min_revenue_growth: Optional[float] = None  # Min YoY revenue growth % (e.g., 0.05 for 5%)

    # Profitability filters
    min_profit_margin: Optional[float] = None  # Min profit margin % (e.g., 0.05 for 5%)
    min_roe: Optional[float] = None  # Min return on equity % (e.g., 0.10 for 10%)

    # Financial health filters
    max_debt_to_equity: Optional[float] = None  # Max debt/equity ratio (e.g., 2.0)
    min_current_ratio: Optional[float] = None  # Min current ratio (e.g., 1.0)

    # Short interest filters
    max_short_percent: Optional[float] = None  # Max short % of float (e.g., 30)
    min_short_percent: Optional[float] = None  # Min short % for squeeze plays (e.g., 15)

    # Ownership filters
    min_institutional_pct: Optional[float] = None  # Min institutional ownership % (e.g., 20)

    # Earnings risk filter
    days_to_earnings_min: Optional[int] = None  # Avoid stocks with earnings within N days

    # Include fundamentals data in query (enables fundamental columns in results)
    include_fundamentals: bool = True


@dataclass
class SignalSetup:
    """
    Complete signal setup with entry, stops, and targets.

    This dataclass represents a fully-formed trading signal ready
    for storage in the database or export to TradingView.
    """
    # Identification
    ticker: str
    scanner_name: str
    signal_type: SignalType
    signal_timestamp: datetime = field(default_factory=datetime.now)

    # Entry/Exit Levels
    entry_price: Decimal = Decimal('0')
    stop_loss: Decimal = Decimal('0')
    take_profit_1: Decimal = Decimal('0')
    take_profit_2: Optional[Decimal] = None
    risk_reward_ratio: Decimal = Decimal('0')
    risk_percent: Decimal = Decimal('0')  # % risk from entry to stop

    # Scoring
    composite_score: int = 0
    quality_tier: QualityTier = QualityTier.D
    trend_score: Decimal = Decimal('0')
    momentum_score: Decimal = Decimal('0')
    volume_score: Decimal = Decimal('0')
    pattern_score: Decimal = Decimal('0')
    confluence_score: Decimal = Decimal('0')

    # Context
    setup_description: str = ""
    confluence_factors: List[str] = field(default_factory=list)
    timeframe: str = "daily"
    market_regime: str = ""

    # Position sizing suggestion
    suggested_position_size_pct: Decimal = Decimal('0')
    max_risk_per_trade_pct: Decimal = Decimal('1.5')

    # Raw data for reference
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set quality tier based on composite score"""
        if self.quality_tier == QualityTier.D and self.composite_score > 0:
            self.quality_tier = QualityTier.from_score(self.composite_score)

    @property
    def is_high_quality(self) -> bool:
        """Check if signal is A or A+ quality"""
        return self.quality_tier in [QualityTier.A_PLUS, QualityTier.A]

    @property
    def r_risk_amount(self) -> Decimal:
        """Calculate 1R (risk amount) based on entry and stop"""
        return abs(self.entry_price - self.stop_loss)

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion"""
        import math

        def sanitize_float(val) -> Optional[float]:
            """Convert to float, replacing NaN/Inf with None"""
            if val is None:
                return None
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return None
            return f

        # Convert signal_timestamp to timezone-aware datetime for asyncpg
        ts = self.signal_timestamp
        if isinstance(ts, pd.Timestamp):
            # If pandas Timestamp, convert to Python datetime
            if ts.tz is None:
                # Assume UTC if no timezone
                ts = ts.tz_localize('UTC').to_pydatetime()
            else:
                ts = ts.to_pydatetime()
        elif isinstance(ts, datetime):
            if ts.tzinfo is None:
                # Assume UTC if no timezone
                ts = ts.replace(tzinfo=timezone.utc)

        # Sanitize all float values to prevent NaN from reaching the database
        entry_price = sanitize_float(self.entry_price) or 0.0
        stop_loss = sanitize_float(self.stop_loss)
        take_profit_1 = sanitize_float(self.take_profit_1)

        # If stop_loss or take_profit is None/NaN, calculate reasonable defaults
        if stop_loss is None or stop_loss == 0:
            # Default to 5% below entry for BUY, 5% above for SELL
            if self.signal_type == SignalType.BUY:
                stop_loss = entry_price * 0.95
            else:
                stop_loss = entry_price * 1.05

        if take_profit_1 is None or take_profit_1 == 0:
            # Default to 10% above entry for BUY, 10% below for SELL
            if self.signal_type == SignalType.BUY:
                take_profit_1 = entry_price * 1.10
            else:
                take_profit_1 = entry_price * 0.90

        return {
            'signal_timestamp': ts,
            'scanner_name': self.scanner_name,
            'ticker': self.ticker,
            'signal_type': self.signal_type.value,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': sanitize_float(self.take_profit_2),
            'risk_reward_ratio': sanitize_float(self.risk_reward_ratio) or 0.0,
            'risk_percent': sanitize_float(self.risk_percent) or 0.0,
            'composite_score': self.composite_score,
            'quality_tier': self.quality_tier.value,
            'trend_score': sanitize_float(self.trend_score) or 0.0,
            'momentum_score': sanitize_float(self.momentum_score) or 0.0,
            'volume_score': sanitize_float(self.volume_score) or 0.0,
            'pattern_score': sanitize_float(self.pattern_score) or 0.0,
            'confluence_score': sanitize_float(self.confluence_score) or 0.0,
            'setup_description': self.setup_description,
            'confluence_factors': self.confluence_factors,
            'timeframe': self.timeframe,
            'market_regime': self.market_regime,
            'suggested_position_size_pct': sanitize_float(self.suggested_position_size_pct) or 0.0,
            'max_risk_per_trade_pct': sanitize_float(self.max_risk_per_trade_pct) or 1.5,
        }

    def to_tradingview_dict(self) -> Dict[str, Any]:
        """Convert to TradingView-compatible format"""
        return {
            'symbol': self.ticker,
            'side': 'long' if self.signal_type == SignalType.BUY else 'short',
            'entry': float(self.entry_price),
            'stop': float(self.stop_loss),
            'tp1': float(self.take_profit_1),
            'tp2': float(self.take_profit_2) if self.take_profit_2 else None,
            'risk_pct': float(self.risk_percent),
            'rr_ratio': float(self.risk_reward_ratio),
            'score': self.composite_score,
            'tier': self.quality_tier.value,
            'setup': self.setup_description,
            'factors': ', '.join(self.confluence_factors),
        }


class BaseScanner(ABC):
    """
    Abstract base class for all signal scanners.

    Provides common functionality:
    - Database interaction
    - Signal storage
    - Entry/exit level calculations
    - Quality scoring integration

    Subclasses must implement:
    - scanner_name: Unique identifier for the scanner
    - scan(): Main scanning logic
    - _calculate_entry_levels(): Strategy-specific entry/stop/target calculation
    """

    def __init__(
        self,
        db_manager,
        config: ScannerConfig = None,
        scorer=None
    ):
        """
        Initialize scanner.

        Args:
            db_manager: AsyncDatabaseManager instance
            config: Scanner configuration
            scorer: SignalScorer instance for quality scoring
        """
        self.db = db_manager
        self.config = config or ScannerConfig()
        self.scorer = scorer
        self._signals: List[SignalSetup] = []

    @property
    @abstractmethod
    def scanner_name(self) -> str:
        """Unique name identifying this scanner"""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the scanner strategy"""
        return "Base scanner"

    @abstractmethod
    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute the scanning strategy.

        Args:
            calculation_date: Date to scan (defaults to today)

        Returns:
            List of SignalSetup objects meeting criteria
        """
        pass

    @abstractmethod
    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Calculate entry, stop-loss, and take-profit levels.

        Args:
            candidate: Raw candidate data
            signal_type: BUY or SELL

        Returns:
            Tuple of (entry_price, stop_loss, take_profit_1, take_profit_2)
        """
        pass

    # =========================================================================
    # COMMON METHODS
    # =========================================================================

    def calculate_atr_based_stop(
        self,
        price: Decimal,
        atr: Decimal,
        signal_type: SignalType
    ) -> Decimal:
        """
        Calculate stop-loss using ATR.

        Args:
            price: Entry price
            atr: ATR value (in price terms, not %)
            signal_type: BUY or SELL

        Returns:
            Stop-loss price
        """
        stop_distance = atr * Decimal(str(self.config.atr_stop_multiplier))

        # Apply max/min constraints
        max_stop = price * Decimal(str(self.config.max_stop_loss_pct / 100))
        min_stop = price * Decimal(str(self.config.min_stop_loss_pct / 100))

        stop_distance = max(min(stop_distance, max_stop), min_stop)

        if signal_type == SignalType.BUY:
            return price - stop_distance
        else:
            return price + stop_distance

    def calculate_take_profits(
        self,
        entry: Decimal,
        stop: Decimal,
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate take-profit levels based on R-multiples.

        Args:
            entry: Entry price
            stop: Stop-loss price
            signal_type: BUY or SELL

        Returns:
            Tuple of (tp1, tp2)
        """
        risk = abs(entry - stop)

        if signal_type == SignalType.BUY:
            tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))
            tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))
        else:
            tp1 = entry - (risk * Decimal(str(self.config.tp1_rr_ratio)))
            tp2 = entry - (risk * Decimal(str(self.config.tp2_rr_ratio)))

        return tp1, tp2

    def calculate_position_size(
        self,
        account_risk_pct: Decimal,
        entry: Decimal,
        stop: Decimal,
        quality_tier: QualityTier
    ) -> Decimal:
        """
        Calculate suggested position size based on quality and risk.

        Higher quality signals get larger position sizes.

        Args:
            account_risk_pct: Max % of account to risk
            entry: Entry price
            stop: Stop-loss price
            quality_tier: Signal quality

        Returns:
            Position size as % of portfolio
        """
        risk_per_share_pct = abs(entry - stop) / entry * 100

        # Quality-based position multipliers
        tier_multipliers = {
            QualityTier.A_PLUS: Decimal('1.0'),
            QualityTier.A: Decimal('0.8'),
            QualityTier.B: Decimal('0.6'),
            QualityTier.C: Decimal('0.4'),
            QualityTier.D: Decimal('0.2'),
        }

        multiplier = tier_multipliers.get(quality_tier, Decimal('0.2'))
        adjusted_risk = account_risk_pct * multiplier

        # Position size = (Risk Amount / Risk Per Share %)
        position_size = (adjusted_risk / risk_per_share_pct) * 100

        # Cap at 25% of portfolio max
        return min(position_size, Decimal('25.0'))

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    async def save_signals(self, signals: List[SignalSetup]) -> int:
        """
        Save signals to database.

        Args:
            signals: List of SignalSetup objects

        Returns:
            Number of signals saved
        """
        if not signals:
            return 0

        saved_count = 0

        # Use ON CONFLICT with the unique index (ticker, scanner_name, signal_date)
        # to prevent duplicate signals for the same ticker/scanner/day
        insert_query = """
            INSERT INTO stock_scanner_signals (
                signal_timestamp, scanner_name, ticker, signal_type,
                entry_price, stop_loss, take_profit_1, take_profit_2,
                risk_reward_ratio, risk_percent,
                composite_score, quality_tier,
                trend_score, momentum_score, volume_score, pattern_score, confluence_score,
                setup_description, confluence_factors, timeframe, market_regime,
                suggested_position_size_pct, max_risk_per_trade_pct
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
            )
            ON CONFLICT (ticker, scanner_name, signal_date) DO UPDATE SET
                -- Update if we get a higher quality signal on the same day
                composite_score = CASE
                    WHEN EXCLUDED.composite_score > stock_scanner_signals.composite_score
                    THEN EXCLUDED.composite_score
                    ELSE stock_scanner_signals.composite_score
                END,
                quality_tier = CASE
                    WHEN EXCLUDED.composite_score > stock_scanner_signals.composite_score
                    THEN EXCLUDED.quality_tier
                    ELSE stock_scanner_signals.quality_tier
                END,
                entry_price = CASE
                    WHEN EXCLUDED.composite_score > stock_scanner_signals.composite_score
                    THEN EXCLUDED.entry_price
                    ELSE stock_scanner_signals.entry_price
                END,
                updated_at = NOW()
            WHERE stock_scanner_signals.status = 'active'
            RETURNING id
        """

        for signal in signals:
            try:
                data = signal.to_db_dict()
                result = await self.db.fetchval(
                    insert_query,
                    data['signal_timestamp'],
                    data['scanner_name'],
                    data['ticker'],
                    data['signal_type'],
                    data['entry_price'],
                    data['stop_loss'],
                    data['take_profit_1'],
                    data['take_profit_2'],
                    data['risk_reward_ratio'],
                    data['risk_percent'],
                    data['composite_score'],
                    data['quality_tier'],
                    data['trend_score'],
                    data['momentum_score'],
                    data['volume_score'],
                    data['pattern_score'],
                    data['confluence_score'],
                    data['setup_description'],
                    data['confluence_factors'],
                    data['timeframe'],
                    data['market_regime'],
                    data['suggested_position_size_pct'],
                    data['max_risk_per_trade_pct'],
                )
                if result:
                    saved_count += 1

            except Exception as e:
                logger.error(f"Failed to save signal {signal.ticker}: {e}")

        logger.info(f"Saved {saved_count}/{len(signals)} signals for {self.scanner_name}")
        return saved_count

    async def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active signals for this scanner"""
        query = """
            SELECT * FROM stock_scanner_signals
            WHERE scanner_name = $1
              AND status = 'active'
            ORDER BY composite_score DESC
        """
        rows = await self.db.fetch(query, self.scanner_name)
        return [dict(r) for r in rows]

    async def check_scanner_exists(self) -> bool:
        """Verify scanner is registered in database"""
        query = """
            SELECT 1 FROM stock_signal_scanners
            WHERE scanner_name = $1 AND is_active = true
        """
        result = await self.db.fetchval(query, self.scanner_name)
        return result is not None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    async def get_watchlist_candidates(
        self,
        calculation_date: datetime,
        additional_filters: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get candidates from watchlist with signals and fundamental data.

        Args:
            calculation_date: Date for watchlist data
            additional_filters: Extra SQL WHERE conditions

        Returns:
            List of candidate dictionaries with technical and fundamental data
        """
        # Build fundamental columns selection
        fundamental_columns = ""
        fundamental_join = ""
        fundamental_filters = ""

        if self.config.include_fundamentals:
            fundamental_columns = """,
                -- Valuation metrics
                i.trailing_pe,
                i.forward_pe,
                i.peg_ratio,
                i.price_to_book,
                i.price_to_sales,
                i.enterprise_to_ebitda,
                -- Growth metrics
                i.earnings_growth,
                i.revenue_growth,
                i.earnings_quarterly_growth,
                -- Profitability metrics
                i.profit_margin,
                i.operating_margin,
                i.gross_margin,
                i.return_on_equity,
                i.return_on_assets,
                -- Financial health
                i.debt_to_equity,
                i.current_ratio,
                i.quick_ratio,
                -- Short interest
                i.short_percent_float,
                i.short_ratio,
                i.shares_short,
                -- Ownership
                i.institutional_percent,
                i.insider_percent,
                i.shares_float,
                -- Dividend
                i.dividend_yield,
                i.payout_ratio,
                -- 52-week data (from fundamentals, not watchlist)
                i.fifty_two_week_high as fund_52w_high,
                i.fifty_two_week_low as fund_52w_low,
                i.fifty_two_week_change,
                -- Analyst data
                i.analyst_rating,
                i.target_price,
                i.number_of_analysts,
                -- Company info
                i.sector as company_sector,
                i.industry,
                i.market_cap,
                -- Earnings calendar
                i.earnings_date,
                i.earnings_date_estimated,
                i.fundamentals_updated_at"""

            fundamental_join = """
            LEFT JOIN stock_instruments i ON w.ticker = i.ticker"""

            # Build dynamic fundamental filters
            fundamental_filters = self._build_fundamental_filters()

        base_query = """
            SELECT
                w.ticker,
                w.tier,
                w.score as watchlist_score,
                w.current_price,
                w.atr_percent,
                w.relative_volume,
                w.price_change_20d,
                w.trend_strength,
                w.rsi_signal,
                w.sma20_signal,
                w.sma50_signal,
                w.sma_cross_signal,
                w.macd_cross_signal,
                w.high_low_signal,
                w.gap_signal,
                w.candlestick_pattern,
                w.pct_from_52w_high,
                m.price_change_1d,
                m.price_change_5d,
                m.rsi_14,
                m.macd,
                m.macd_histogram,
                m.adx,
                m.perf_1w,
                m.perf_1m,
                m.atr_14,
                m.sma_20,
                m.sma_50,
                m.sma_200,
                m.avg_volume_20
                {fundamental_columns}
            FROM stock_watchlist w
            LEFT JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            {fundamental_join}
            WHERE w.calculation_date = $1
              AND w.tier <= $2
              AND w.relative_volume >= $3
              {fundamental_filters}
              {additional_filters}
            ORDER BY w.score DESC
        """.format(
            fundamental_columns=fundamental_columns,
            fundamental_join=fundamental_join,
            fundamental_filters=fundamental_filters,
            additional_filters=additional_filters
        )

        rows = await self.db.fetch(
            base_query,
            calculation_date,
            self.config.max_tier,
            self.config.min_relative_volume
        )
        return [dict(r) for r in rows]

    def _build_fundamental_filters(self) -> str:
        """
        Build SQL WHERE clauses for fundamental filters based on config.

        Returns:
            SQL string with AND clauses for each enabled filter
        """
        filters = []

        # Valuation filters
        if self.config.max_pe_ratio is not None:
            filters.append(f"(i.trailing_pe IS NULL OR i.trailing_pe <= {self.config.max_pe_ratio})")
        if self.config.min_pe_ratio is not None:
            filters.append(f"(i.trailing_pe IS NULL OR i.trailing_pe >= {self.config.min_pe_ratio})")
        if self.config.max_peg_ratio is not None:
            filters.append(f"(i.peg_ratio IS NULL OR i.peg_ratio <= {self.config.max_peg_ratio})")

        # Growth filters
        if self.config.min_earnings_growth is not None:
            filters.append(f"(i.earnings_growth IS NULL OR i.earnings_growth >= {self.config.min_earnings_growth})")
        if self.config.min_revenue_growth is not None:
            filters.append(f"(i.revenue_growth IS NULL OR i.revenue_growth >= {self.config.min_revenue_growth})")

        # Profitability filters
        if self.config.min_profit_margin is not None:
            filters.append(f"(i.profit_margin IS NULL OR i.profit_margin >= {self.config.min_profit_margin})")
        if self.config.min_roe is not None:
            filters.append(f"(i.return_on_equity IS NULL OR i.return_on_equity >= {self.config.min_roe})")

        # Financial health filters
        if self.config.max_debt_to_equity is not None:
            filters.append(f"(i.debt_to_equity IS NULL OR i.debt_to_equity <= {self.config.max_debt_to_equity})")
        if self.config.min_current_ratio is not None:
            filters.append(f"(i.current_ratio IS NULL OR i.current_ratio >= {self.config.min_current_ratio})")

        # Short interest filters
        if self.config.max_short_percent is not None:
            filters.append(f"(i.short_percent_float IS NULL OR i.short_percent_float <= {self.config.max_short_percent})")
        if self.config.min_short_percent is not None:
            filters.append(f"(i.short_percent_float IS NOT NULL AND i.short_percent_float >= {self.config.min_short_percent})")

        # Ownership filters
        if self.config.min_institutional_pct is not None:
            filters.append(f"(i.institutional_percent IS NULL OR i.institutional_percent >= {self.config.min_institutional_pct})")

        # Earnings date filter (avoid stocks with earnings within N days)
        if self.config.days_to_earnings_min is not None:
            filters.append(f"""
                (i.earnings_date IS NULL OR
                 i.earnings_date < CURRENT_DATE OR
                 i.earnings_date > CURRENT_DATE + INTERVAL '{self.config.days_to_earnings_min} days')
            """)

        if filters:
            return "AND " + " AND ".join(filters)
        return ""

    def build_confluence_factors(self, candidate: Dict[str, Any]) -> List[str]:
        """
        Build list of confluence factors for a candidate.

        Args:
            candidate: Candidate data dictionary

        Returns:
            List of confluence factor descriptions
        """
        factors = []

        # Trend factors
        sma_cross = candidate.get('sma_cross_signal', '')
        if sma_cross == 'golden_cross':
            factors.append('Golden Cross')
        elif sma_cross == 'bullish':
            factors.append('Above MAs')

        # Momentum factors
        macd_cross = candidate.get('macd_cross_signal', '')
        if macd_cross == 'bullish_cross':
            factors.append('MACD Cross Up')
        elif macd_cross == 'bullish':
            factors.append('MACD Bullish')

        # RSI factors
        rsi_signal = candidate.get('rsi_signal', '')
        if rsi_signal in ['oversold', 'oversold_extreme']:
            factors.append('Oversold RSI')

        # Volume factors
        rel_vol = float(candidate.get('relative_volume', 0) or 0)
        if rel_vol >= self.config.high_volume_threshold:
            factors.append(f'High Volume ({rel_vol:.1f}x)')

        # Pattern factors
        pattern = candidate.get('candlestick_pattern', '')
        if pattern in ['bullish_engulfing', 'hammer', 'dragonfly_doji']:
            factors.append(pattern.replace('_', ' ').title())

        # Position factors
        high_low = candidate.get('high_low_signal', '')
        if high_low == 'near_high':
            factors.append('Near 52W High')
        elif high_low == 'new_high':
            factors.append('New 52W High')

        # Gap factors
        gap = candidate.get('gap_signal', '')
        if gap in ['gap_up', 'gap_up_large']:
            factors.append('Gap Up')

        # =======================================================================
        # FUNDAMENTAL CONFLUENCE FACTORS
        # =======================================================================
        if self.config.include_fundamentals:
            factors.extend(self._build_fundamental_confluence(candidate))

        return factors

    def _build_fundamental_confluence(self, candidate: Dict[str, Any]) -> List[str]:
        """
        Build fundamental-based confluence factors.

        Args:
            candidate: Candidate data with fundamental fields

        Returns:
            List of fundamental confluence descriptions
        """
        factors = []

        # Earnings growth
        earnings_growth = candidate.get('earnings_growth')
        if earnings_growth is not None:
            eg = float(earnings_growth)
            if eg >= 0.25:
                factors.append(f'Strong Earnings Growth (+{eg*100:.0f}%)')
            elif eg >= 0.10:
                factors.append(f'Solid Earnings Growth (+{eg*100:.0f}%)')

        # Revenue growth
        revenue_growth = candidate.get('revenue_growth')
        if revenue_growth is not None:
            rg = float(revenue_growth)
            if rg >= 0.20:
                factors.append(f'High Revenue Growth (+{rg*100:.0f}%)')

        # Return on Equity
        roe = candidate.get('return_on_equity')
        if roe is not None:
            roe_val = float(roe)
            if roe_val >= 0.20:
                factors.append(f'Strong ROE ({roe_val*100:.0f}%)')

        # Low P/E (value)
        pe = candidate.get('trailing_pe')
        if pe is not None:
            pe_val = float(pe)
            if 0 < pe_val < 15:
                factors.append(f'Low P/E ({pe_val:.1f})')

        # PEG ratio (growth at reasonable price)
        peg = candidate.get('peg_ratio')
        if peg is not None:
            peg_val = float(peg)
            if 0 < peg_val < 1.0:
                factors.append(f'Attractive PEG ({peg_val:.2f})')

        # Healthy balance sheet
        debt_eq = candidate.get('debt_to_equity')
        if debt_eq is not None:
            de_val = float(debt_eq)
            if de_val < 0.5:
                factors.append('Low Debt')

        # High short interest (squeeze potential)
        short_pct = candidate.get('short_percent_float')
        if short_pct is not None:
            sp_val = float(short_pct)
            if sp_val >= 15:
                factors.append(f'High Short Interest ({sp_val:.1f}%)')

        # Strong institutional support
        inst_pct = candidate.get('institutional_percent')
        if inst_pct is not None:
            ip_val = float(inst_pct)
            if ip_val >= 70:
                factors.append(f'High Institutional ({ip_val:.0f}%)')

        # Analyst sentiment
        analyst_rating = candidate.get('analyst_rating')
        if analyst_rating and 'buy' in str(analyst_rating).lower():
            factors.append('Analyst: Buy')

        # Dividend yield (income)
        div_yield = candidate.get('dividend_yield')
        if div_yield is not None:
            dy_val = float(div_yield)
            if dy_val >= 3.0:
                factors.append(f'High Dividend ({dy_val:.1f}%)')

        return factors

    async def get_all_active_tickers(self) -> List[str]:
        """
        Get all active stock tickers from stock_instruments.

        Returns:
            List of ticker symbols for all active stocks
        """
        query = """
            SELECT ticker
            FROM stock_instruments
            WHERE is_active = true
            ORDER BY ticker
        """
        rows = await self.db.fetch(query)
        return [r['ticker'] for r in rows]

    def log_scan_summary(
        self,
        candidates_analyzed: int,
        signals_generated: int,
        high_quality_count: int
    ):
        """Log summary of scan results"""
        logger.info("=" * 60)
        logger.info(f"SCAN COMPLETE: {self.scanner_name}")
        logger.info("=" * 60)
        logger.info(f"  Candidates Analyzed: {candidates_analyzed}")
        logger.info(f"  Signals Generated: {signals_generated}")
        logger.info(f"  High Quality (A/A+): {high_quality_count}")
        logger.info("=" * 60)
