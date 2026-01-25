"""
MACD Momentum Scanner

Uses the backtested MACDMomentumStrategy for signal detection.
This ensures identical entry logic between backtesting and live scanning.

Entry Logic (from backtested strategy - PF 1.71):
1. MACD histogram crosses above zero (bullish) or below zero (bearish)
2. Histogram strength >= 2x minimum threshold (strong momentum confirmation)
3. ADX > 25 (trending market - filters out ranging conditions)
4. Price structure: Higher lows (bullish) or lower highs (bearish)
5. Relative volume >= 1.2x (above-average institutional participation)
6. RSI 30-70 (not at extremes)
7. Quality tier A+, A, or B only (skip C/D)

Risk Management:
- Stop: 2.0x ATR from entry
- Target: 5.0x ATR (2.5:1 R:R)

This scanner fetches candle data and runs the exact same strategy
that achieved PF 1.71 in backtesting.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any, Optional

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ...strategies.macd_momentum import MACDMomentumStrategy, MACDMomentumSignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class MACDMomentumConfig(ScannerConfig):
    """Configuration for MACD Momentum Scanner - uses backtested strategy defaults"""

    # Strategy parameters (match backtested strategy)
    stop_loss_atr_mult: float = 2.0    # 2.0x ATR stop loss
    take_profit_atr_mult: float = 5.0  # 5.0x ATR take profit (2.5:1 R:R)
    histogram_min_threshold: float = 0.01  # Minimum histogram value
    min_adx: float = 25.0              # ADX threshold for trending market
    min_rsi: float = 30.0              # RSI range lower bound
    max_rsi: float = 70.0              # RSI range upper bound
    min_relative_volume: float = 1.2   # Volume >= 1.2x average

    # Scanning limits
    max_signals_per_run: int = 50
    min_score_threshold: int = 60      # Minimum quality score


class MACDMomentumScanner(BaseScanner):
    """
    MACD Momentum Scanner using the exact backtested strategy logic.

    This scanner:
    1. Pre-filters candidates using watchlist data (fast database query)
    2. Fetches candle data for promising candidates
    3. Runs MACDMomentumStrategy.scan() - identical to backtest
    4. Converts signals to SignalSetup format for database storage

    This ensures live signals match backtested performance (PF 1.71).
    """

    def __init__(
        self,
        db_manager,
        config: MACDMomentumConfig = None,
        scorer=None
    ):
        super().__init__(db_manager, config or MACDMomentumConfig(), scorer)

        # Initialize the backtested strategy with optimized parameters
        self.strategy = MACDMomentumStrategy(
            stop_loss_atr_mult=self.config.stop_loss_atr_mult,
            take_profit_atr_mult=self.config.take_profit_atr_mult,
            histogram_min_threshold=self.config.histogram_min_threshold,
            min_adx=self.config.min_adx,
            min_rsi=self.config.min_rsi,
            max_rsi=self.config.max_rsi,
            min_relative_volume=self.config.min_relative_volume
        )

        # Data provider for fetching candle data with indicators
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "macd_momentum"

    @property
    def description(self) -> str:
        return "MACD momentum - uses backtested strategy (PF 1.71)"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute MACD Momentum scan using the backtested strategy.

        Steps:
        1. Get ALL active tickers from stock_instruments
        2. For each ticker, fetch candle data with indicators
        3. Run MACDMomentumStrategy.scan() - exact backtest logic
        4. Convert MACDMomentumSignal to SignalSetup format
        5. Return sorted signals by quality
        """
        logger.info(f"Starting {self.scanner_name} scan (using backtested strategy)")

        # Set calculation date
        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        logger.info(f"Scan date: {calculation_date}")

        # Step 1: Get ALL active tickers - no pre-filtering
        tickers = await self.get_all_active_tickers()
        logger.info(f"Scanning {len(tickers)} active stocks for MACD momentum")

        # Step 2 & 3: Fetch data and run strategy for each ticker
        signals = []
        scanned_count = 0

        for ticker in tickers:
            sector = None  # Can be fetched from stock_instruments if needed

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
                macd_signal = self.strategy.scan(df, ticker, sector)

                if macd_signal:
                    # Convert to SignalSetup format (pass ticker info as minimal candidate)
                    candidate = {'ticker': ticker, 'sector': sector}
                    signal = self._convert_to_signal_setup(macd_signal, candidate)
                    if signal:
                        signals.append(signal)
                        logger.info(f"{ticker}: Signal generated - {macd_signal.quality_tier} tier, "
                                    f"confidence {macd_signal.confidence:.1%}")

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
        self.log_scan_summary(len(tickers), len(signals), high_quality)

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

    async def _get_momentum_candidates(
        self,
        calculation_date: date
    ) -> List[Dict[str, Any]]:
        """
        Pre-filter candidates with momentum potential from watchlist.

        This fast database query reduces the number of tickers we need
        to fetch full candle data for. We use loose filters here because
        the strategy will apply strict filters (ADX, histogram, etc.) on
        the actual candle data.
        """
        # Query for candidates that might have MACD momentum signals
        # Keep pre-filter loose - strategy is the source of truth
        additional_filters = """
            AND (
                w.macd_cross_signal IN ('bullish_cross', 'bullish', 'bearish_cross', 'bearish')
                OR w.trend_strength IN ('strong_up', 'up', 'strong_down', 'down')
            )
        """

        return await self.get_watchlist_candidates(
            calculation_date,
            additional_filters
        )

    def _convert_to_signal_setup(
        self,
        macd_signal: MACDMomentumSignal,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """
        Convert a MACDMomentumSignal from the strategy to SignalSetup format.

        The strategy already validated all entry conditions, so we just
        need to format the output for database storage.
        """
        # Convert confidence to composite score (0-100)
        composite_score = int(macd_signal.confidence * 100)

        # Get quality tier from strategy
        tier_map = {
            'A+': QualityTier.A_PLUS,
            'A': QualityTier.A,
            'B': QualityTier.B,
            'C': QualityTier.C,
            'D': QualityTier.D
        }
        quality_tier = tier_map.get(macd_signal.quality_tier, QualityTier.B)

        # Calculate risk percent
        entry = macd_signal.entry_price
        stop = macd_signal.stop_loss_price
        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0

        # Determine signal type
        signal_type = SignalType.BUY if macd_signal.signal_type == 'BUY' else SignalType.SELL

        # Build confluence factors
        confluence_factors = self._build_confluence_factors(macd_signal, candidate)

        # Build description
        description = self._build_description(macd_signal, candidate)

        # Create SignalSetup
        signal = SignalSetup(
            ticker=macd_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=macd_signal.signal_timestamp,
            entry_price=Decimal(str(round(entry, 4))),
            stop_loss=Decimal(str(round(stop, 4))),
            take_profit_1=Decimal(str(round(macd_signal.take_profit_price, 4))),
            take_profit_2=None,  # Strategy uses single TP
            risk_reward_ratio=Decimal(str(round(macd_signal.risk_reward_ratio, 2))),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=composite_score,
            quality_tier=quality_tier,
            trend_score=Decimal('0'),  # Strategy doesn't break down scores
            momentum_score=Decimal('0'),
            volume_score=Decimal('0'),
            pattern_score=Decimal('0'),
            confluence_score=Decimal(str(round(macd_signal.confidence * 25, 2))),
            setup_description=description,
            confluence_factors=confluence_factors,
            timeframe="daily",
            market_regime=self._determine_market_regime(macd_signal, candidate),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                'macd': macd_signal.macd,
                'macd_signal': macd_signal.macd_signal,
                'macd_histogram': macd_signal.macd_histogram,
                'prev_histogram': macd_signal.prev_histogram,
                'rsi': macd_signal.rsi,
                'atr': macd_signal.atr,
                'relative_volume': macd_signal.relative_volume,
                **candidate
            }
        )

        return signal

    def _build_confluence_factors(
        self,
        macd_signal: MACDMomentumSignal,
        candidate: Dict[str, Any]
    ) -> List[str]:
        """Build confluence factors from the signal and candidate data."""
        factors = []

        # MACD histogram cross
        if macd_signal.signal_type == 'BUY':
            factors.append("MACD histogram crossed above zero")
        else:
            factors.append("MACD histogram crossed below zero")

        # Histogram strength
        hist = abs(macd_signal.macd_histogram)
        if hist >= 0.05:
            factors.append(f"Strong histogram ({hist:.3f})")
        elif hist >= 0.02:
            factors.append(f"Good histogram ({hist:.3f})")

        # ADX (from candidate if available)
        adx = candidate.get('adx')
        if adx:
            adx_val = float(adx)
            if adx_val >= 30:
                factors.append(f"Strong trend (ADX {adx_val:.0f})")
            elif adx_val >= 25:
                factors.append(f"Trending (ADX {adx_val:.0f})")

        # RSI
        if macd_signal.rsi:
            rsi = macd_signal.rsi
            if 45 <= rsi <= 55:
                factors.append(f"RSI neutral ({rsi:.0f})")
            elif macd_signal.signal_type == 'BUY' and rsi < 50:
                factors.append(f"RSI has room to rise ({rsi:.0f})")
            elif macd_signal.signal_type == 'SELL' and rsi > 50:
                factors.append(f"RSI has room to fall ({rsi:.0f})")

        # Volume
        if macd_signal.relative_volume:
            rv = macd_signal.relative_volume
            if rv >= 1.5:
                factors.append(f"High volume ({rv:.1f}x)")
            elif rv >= 1.2:
                factors.append(f"Above-avg volume ({rv:.1f}x)")

        # Trend strength from candidate
        trend = candidate.get('trend_strength', '')
        if trend == 'strong_up':
            factors.append("Strong uptrend")
        elif trend == 'up':
            factors.append("Uptrend confirmed")
        elif trend == 'strong_down':
            factors.append("Strong downtrend")
        elif trend == 'down':
            factors.append("Downtrend confirmed")

        return factors[:8]

    def _build_description(
        self,
        macd_signal: MACDMomentumSignal,
        candidate: Dict[str, Any]
    ) -> str:
        """Build human-readable description."""
        ticker = macd_signal.ticker
        direction = "bullish" if macd_signal.signal_type == 'BUY' else "bearish"
        hist = macd_signal.macd_histogram

        desc = f"{ticker} MACD momentum {direction} entry. "
        desc += f"Histogram crossed zero to {hist:.4f}. "

        if macd_signal.rsi:
            desc += f"RSI at {macd_signal.rsi:.0f}. "

        desc += f"Quality: {macd_signal.quality_tier} tier ({macd_signal.confidence:.0%} confidence)."

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
        price = Decimal(str(candidate.get('current_price', 0)))
        return (price, price * Decimal('0.95'), price * Decimal('1.10'), None)

    def _determine_market_regime(
        self,
        macd_signal: MACDMomentumSignal,
        candidate: Dict[str, Any]
    ) -> str:
        """Determine market regime from signal and candidate data."""
        trend_strength = candidate.get('trend_strength', '')

        if trend_strength == 'strong_up':
            regime = "Strong Uptrend"
        elif trend_strength == 'up':
            regime = "Uptrend"
        elif trend_strength == 'strong_down':
            regime = "Strong Downtrend"
        elif trend_strength == 'down':
            regime = "Downtrend"
        else:
            regime = "Transitioning"

        regime += " - MACD Momentum"

        return regime
