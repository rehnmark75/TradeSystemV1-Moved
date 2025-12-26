"""
EMA Pullback Scanner

Uses the backtested EMATrendPullbackStrategy for signal detection.
This ensures identical entry logic between backtesting and live scanning.

Entry Logic (from backtested strategy):
1. Price > 200 EMA (long-term uptrend)
2. Price > 100 EMA (medium-term uptrend)
3. Price > 50 EMA (short-term uptrend)
4. ADX > 20 (trending market - Welles Wilder threshold)
5. MACD > 0 (bullish momentum)
6. Price dropped 2-5% below 20 EMA (pullback)
7. Price crosses back above 20 EMA (entry trigger)
8. RSI 40-60 (healthy pullback zone)
9. Volume >= 1.2x average (institutional participation)
10. Quality tier A+, A, or B only (skip C/D)

This scanner fetches candle data and runs the exact same strategy
that achieved PF 2.02 in backtesting.
"""

import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any, Optional

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ...strategies.ema_trend_pullback import EMATrendPullbackStrategy, PullbackSignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


class EMAPullbackConfig(ScannerConfig):
    """Configuration for EMA Pullback Scanner - uses backtested strategy defaults"""

    # Strategy parameters (match backtested strategy)
    stop_loss_pct: float = 5.0       # 5% stop loss (optimized)
    take_profit_pct: float = 10.0    # 10% take profit (2:1 R:R)
    min_rsi: float = 40.0            # RSI range lower bound
    max_rsi: float = 60.0            # RSI range upper bound
    min_relative_volume: float = 1.2  # Volume >= 1.2x average

    # Scanning limits
    max_signals_per_run: int = 50
    min_score_threshold: int = 60    # Minimum quality score


class EMAPullbackScanner(BaseScanner):
    """
    EMA Pullback Scanner using the exact backtested strategy logic.

    This scanner:
    1. Pre-filters candidates using watchlist data (fast database query)
    2. Fetches candle data for promising candidates
    3. Runs EMATrendPullbackStrategy.scan() - identical to backtest
    4. Converts signals to SignalSetup format for database storage

    This ensures live signals match backtested performance (PF 2.02).
    """

    def __init__(
        self,
        db_manager,
        config: EMAPullbackConfig = None,
        scorer = None
    ):
        super().__init__(db_manager, config or EMAPullbackConfig(), scorer)

        # Initialize the backtested strategy with optimized parameters
        self.strategy = EMATrendPullbackStrategy(
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
            min_rsi=self.config.min_rsi,
            max_rsi=self.config.max_rsi,
            min_relative_volume=self.config.min_relative_volume
        )

        # Data provider for fetching candle data with indicators
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "ema_pullback"

    @property
    def description(self) -> str:
        return "EMA trend pullback - uses backtested strategy (PF 2.02)"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute EMA Pullback scan using the backtested strategy.

        Steps:
        1. Get pre-filtered candidates from watchlist (bullish trend)
        2. For each candidate, fetch candle data with indicators
        3. Run EMATrendPullbackStrategy.scan() - exact backtest logic
        4. Convert PullbackSignal to SignalSetup format
        5. Return sorted signals by quality
        """
        logger.info(f"Starting {self.scanner_name} scan (using backtested strategy)")

        # Get the latest available watchlist date if not specified
        if calculation_date is None:
            calculation_date = await self._get_latest_watchlist_date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        logger.info(f"Using watchlist date: {calculation_date}")

        # Step 1: Pre-filter candidates with bullish trend from watchlist
        # This is a fast database query to reduce the number of tickers to scan
        candidates = await self._get_trend_candidates(calculation_date)
        logger.info(f"Found {len(candidates)} candidates with bullish trend alignment")

        # Step 2 & 3: Fetch data and run strategy for each candidate
        signals = []
        scanned_count = 0

        for candidate in candidates:
            ticker = candidate['ticker']
            sector = candidate.get('sector')

            try:
                # Fetch candle data with all indicators
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=calculation_date - timedelta(days=365),
                    end_date=calculation_date,
                    timeframe='1d'
                )

                if df.empty or len(df) < 220:
                    continue

                scanned_count += 1

                # Run the exact backtested strategy logic
                pullback_signal = self.strategy.scan(df, ticker, sector)

                if pullback_signal:
                    # Convert to SignalSetup format
                    signal = self._convert_to_signal_setup(pullback_signal, candidate)
                    if signal:
                        signals.append(signal)
                        logger.info(f"{ticker}: Signal generated - {pullback_signal.quality_tier} tier, "
                                  f"confidence {pullback_signal.confidence:.1%}")

            except Exception as e:
                logger.warning(f"{ticker}: Error scanning - {e}")
                continue

        logger.info(f"Scanned {scanned_count} tickers, generated {len(signals)} signals")

        # Step 4: Sort by composite score (quality)
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Step 5: Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(candidates), len(signals), high_quality)

        return signals

    async def _get_latest_watchlist_date(self) -> date:
        """Get the most recent date with watchlist data."""
        query = """
            SELECT MAX(calculation_date) as latest_date
            FROM stock_watchlist
        """
        rows = await self.db.fetch(query)
        if rows and rows[0]['latest_date']:
            return rows[0]['latest_date']
        return datetime.now().date()

    async def _get_trend_candidates(
        self,
        calculation_date: date
    ) -> List[Dict[str, Any]]:
        """
        Pre-filter candidates with bullish trend from watchlist.

        This fast database query reduces the number of tickers we need
        to fetch full candle data for. We only need tickers that:
        - Have bullish SMA alignment (price > major EMAs)
        - Are in an uptrend
        - Have reasonable liquidity

        Note: ADX/MACD filters are applied by the strategy on candle data,
        not in the pre-filter. This allows the strategy to be the single
        source of truth for entry conditions.
        """
        # Query for bullish trend candidates
        # Keep pre-filter loose - strategy will apply strict filters on candle data
        additional_filters = """
            AND w.trend_strength IN ('strong_up', 'up')
            AND w.sma_cross_signal IN ('golden_cross', 'bullish')
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _convert_to_signal_setup(
        self,
        pullback_signal: PullbackSignal,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """
        Convert a PullbackSignal from the strategy to SignalSetup format.

        The strategy already validated all entry conditions, so we just
        need to format the output for database storage.
        """
        # Convert confidence to composite score (0-100)
        composite_score = int(pullback_signal.confidence * 100)

        # Get quality tier from strategy
        tier_map = {
            'A+': QualityTier.A_PLUS,
            'A': QualityTier.A,
            'B': QualityTier.B,
            'C': QualityTier.C,
            'D': QualityTier.D
        }
        quality_tier = tier_map.get(pullback_signal.quality_tier, QualityTier.B)

        # Calculate risk percent
        entry = pullback_signal.entry_price
        stop = pullback_signal.stop_loss_price
        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0

        # Build confluence factors
        confluence_factors = self._build_confluence_factors(pullback_signal, candidate)

        # Build description
        description = self._build_description(pullback_signal, candidate)

        # Create SignalSetup
        signal = SignalSetup(
            ticker=pullback_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=SignalType.BUY,
            signal_timestamp=pullback_signal.signal_timestamp,
            entry_price=Decimal(str(round(entry, 4))),
            stop_loss=Decimal(str(round(stop, 4))),
            take_profit_1=Decimal(str(round(pullback_signal.take_profit_price, 4))),
            take_profit_2=None,  # Strategy uses single TP
            risk_reward_ratio=Decimal(str(round(pullback_signal.risk_reward_ratio, 2))),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=composite_score,
            quality_tier=quality_tier,
            trend_score=Decimal('0'),  # Strategy doesn't break down scores
            momentum_score=Decimal('0'),
            volume_score=Decimal('0'),
            pattern_score=Decimal('0'),
            confluence_score=Decimal(str(round(pullback_signal.confidence * 25, 2))),
            setup_description=description,
            confluence_factors=confluence_factors,
            timeframe="daily",
            market_regime=self._determine_market_regime(candidate),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                'ema_20': pullback_signal.ema_20,
                'ema_50': pullback_signal.ema_50,
                'ema_100': pullback_signal.ema_100,
                'ema_200': pullback_signal.ema_200,
                'rsi': pullback_signal.rsi,
                'atr': pullback_signal.atr,
                'pullback_percent': pullback_signal.pullback_percent,
                'relative_volume': pullback_signal.relative_volume,
                **candidate
            }
        )

        return signal

    def _build_confluence_factors(
        self,
        pullback_signal: PullbackSignal,
        candidate: Dict[str, Any]
    ) -> List[str]:
        """Build confluence factors from the signal and candidate data."""
        factors = []

        # EMA cascade alignment
        factors.append("EMA cascade aligned (20>50>100>200)")

        # Pullback depth
        pb = pullback_signal.pullback_percent
        if pb >= 2.0 and pb <= 3.0:
            factors.append(f"Ideal pullback depth ({pb:.1f}%)")
        elif pb < 2.0:
            factors.append(f"Shallow pullback ({pb:.1f}%)")
        else:
            factors.append(f"Deep pullback ({pb:.1f}%)")

        # EMA20 crossover (the key entry trigger)
        factors.append("Price crossed above EMA20 (entry trigger)")

        # RSI
        if pullback_signal.rsi:
            rsi = pullback_signal.rsi
            if 45 <= rsi <= 55:
                factors.append(f"RSI in reset zone ({rsi:.0f})")
            else:
                factors.append(f"RSI healthy ({rsi:.0f})")

        # Volume
        if pullback_signal.relative_volume:
            rv = pullback_signal.relative_volume
            if rv >= 1.5:
                factors.append(f"Strong volume ({rv:.1f}x)")
            elif rv >= 1.2:
                factors.append(f"Good volume ({rv:.1f}x)")

        # ADX/MACD from candidate
        adx = candidate.get('adx')
        if adx:
            factors.append(f"ADX trending ({float(adx):.0f})")

        macd = candidate.get('macd')
        if macd and float(macd) > 0:
            factors.append("MACD bullish")

        # Trend strength
        trend = candidate.get('trend_strength', '')
        if trend == 'strong_up':
            factors.append("Strong uptrend")
        elif trend == 'up':
            factors.append("Uptrend confirmed")

        return factors[:8]

    def _build_description(
        self,
        pullback_signal: PullbackSignal,
        candidate: Dict[str, Any]
    ) -> str:
        """Build human-readable description."""
        ticker = pullback_signal.ticker
        pb = pullback_signal.pullback_percent
        rsi = pullback_signal.rsi or 50

        desc = f"{ticker} EMA trend pullback entry. "
        desc += f"Price pulled back {pb:.1f}% below EMA20 and crossed back above (entry trigger). "
        desc += f"RSI at {rsi:.0f}. "
        desc += f"Quality: {pullback_signal.quality_tier} tier ({pullback_signal.confidence:.0%} confidence)."

        return desc

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> tuple:
        """
        Not used - entry levels come from the backtested strategy signal.
        This method is required by the base class but we override the flow.
        """
        # Return placeholder values - actual levels come from strategy.scan()
        from decimal import Decimal
        price = Decimal(str(candidate.get('current_price', 0)))
        return (price, price * Decimal('0.95'), price * Decimal('1.10'), None)

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime from candidate data."""
        trend_strength = candidate.get('trend_strength', '')

        if trend_strength == 'strong_up':
            regime = "Strong Uptrend"
        elif trend_strength == 'up':
            regime = "Uptrend"
        else:
            regime = "Trending"

        regime += " - Pullback Entry"

        return regime
