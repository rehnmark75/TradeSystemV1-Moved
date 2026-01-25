"""
Gap & Go Scanner

Uses the backtested GapAndGoStrategy for signal detection.
This ensures identical entry logic between backtesting and live scanning.

Entry Criteria (from backtested strategy):
1. Gap up >= 2% (calculated from open vs prev close)
2. Volume surge >= 1.5x average
3. Not gapping into major resistance (overbought at 52W high)
4. Not in death cross (avoid bearish trend gaps)
5. RSI 40-80 (momentum with room to run)

Stop Logic:
- Below gap open (gap must hold)
- Max 5% stop distance

Target:
- TP1: Gap extension (100% of gap size added)
- TP2: 2.5R or higher

Best For:
- Earnings beats
- News catalysts
- Sector momentum days
- Opening range breakouts
"""

import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ..scoring import SignalScorer
from ...strategies.gap_and_go import GapAndGoStrategy, GapSignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class GapAndGoConfig(ScannerConfig):
    """Configuration for Gap & Go Scanner - uses backtested strategy defaults"""

    # Strategy parameters (match backtested strategy)
    min_gap_pct: float = 2.0          # Minimum gap size
    large_gap_pct: float = 4.0        # Large gap threshold
    max_gap_pct: float = 12.0         # Max gap (avoid extremes)
    min_relative_volume: float = 1.5  # Volume surge threshold
    stop_multiplier: float = 0.98     # Stop 2% below gap open
    max_stop_pct: float = 5.0         # Maximum stop loss %
    min_rsi: float = 40.0             # RSI lower bound
    max_rsi: float = 80.0             # RSI upper bound


class GapAndGoScanner(BaseScanner):
    """
    Scans for gap continuation opportunities using the backtested strategy.

    Philosophy:
    - Gaps represent overnight information or catalyst
    - Volume confirms institutional interest
    - Gaps in direction of trend more reliable
    - Quick decisions needed - gaps are time-sensitive

    Uses the exact same logic as the backtested strategy.
    """

    def __init__(
        self,
        db_manager,
        config: GapAndGoConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or GapAndGoConfig(), scorer)

        # Initialize the backtested strategy
        self.strategy = GapAndGoStrategy(
            min_gap_pct=self.config.min_gap_pct,
            large_gap_pct=self.config.large_gap_pct,
            max_gap_pct=self.config.max_gap_pct,
            min_relative_volume=self.config.min_relative_volume,
            stop_multiplier=self.config.stop_multiplier,
            max_stop_pct=self.config.max_stop_pct,
            min_rsi=self.config.min_rsi,
            max_rsi=self.config.max_rsi,
        )

        # Initialize data provider for fetching candles with indicators
        self.data_provider = BacktestDataProvider(db_manager)

        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "gap_and_go"

    @property
    def description(self) -> str:
        return "Gap continuation plays - uses backtested strategy"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute Gap & Go scan using the backtested strategy.

        Steps:
        1. Get ALL active tickers from stock_instruments
        2. For each ticker, fetch candle data with indicators
        3. Run GapAndGoStrategy.scan() - exact backtest logic
        4. Convert GapSignal to SignalSetup format
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
        logger.info(f"Scanning {len(tickers)} active stocks for gaps")

        # Step 2 & 3: Fetch data and run strategy for each ticker
        signals = []
        scanned_count = 0

        for ticker in tickers:
            sector = None  # Can be fetched from stock_instruments if needed

            try:
                # Fetch candle data with all indicators
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=calculation_date - timedelta(days=100),  # Shorter history for gaps
                    end_date=calculation_date,
                    timeframe='1d'
                )

                if df.empty or len(df) < 30:  # Need some history for indicators
                    continue

                scanned_count += 1

                # Run the exact backtested strategy logic
                gap_signal = self.strategy.scan(df, ticker, sector)

                if gap_signal:
                    # Convert to SignalSetup format (pass ticker info as minimal candidate)
                    candidate = {'ticker': ticker, 'sector': sector}
                    signal = self._convert_to_signal_setup(gap_signal, candidate)
                    if signal:
                        signals.append(signal)
                        logger.info(f"{ticker}: Signal generated - {gap_signal.quality_tier} tier, "
                                  f"gap {gap_signal.gap_pct:.1f}%")

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

    def _convert_to_signal_setup(
        self,
        gap_signal: GapSignal,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """
        Convert a GapSignal from the strategy to SignalSetup format.

        The strategy already validated all entry conditions, so we just
        need to format the output for database storage.
        """
        # Convert confidence to composite score (0-100)
        composite_score = int(gap_signal.confidence * 100)

        # Get quality tier from strategy
        tier_map = {
            'A+': QualityTier.A_PLUS,
            'A': QualityTier.A,
            'B': QualityTier.B,
            'C': QualityTier.C,
            'D': QualityTier.D
        }
        quality_tier = tier_map.get(gap_signal.quality_tier, QualityTier.B)

        # Calculate risk percent
        entry = gap_signal.entry_price
        stop = gap_signal.stop_loss_price
        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0

        # Determine signal type
        signal_type = SignalType.BUY if gap_signal.signal_type == 'BUY' else SignalType.SELL

        # Build confluence factors
        confluence_factors = self._build_confluence_factors(gap_signal, candidate)

        # Build description
        description = self._build_description(gap_signal, candidate)

        # Create SignalSetup
        signal = SignalSetup(
            ticker=gap_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=gap_signal.signal_timestamp,
            entry_price=Decimal(str(round(entry, 4))),
            stop_loss=Decimal(str(round(stop, 4))),
            take_profit_1=Decimal(str(round(gap_signal.take_profit_price, 4))),
            take_profit_2=None,
            risk_reward_ratio=Decimal(str(round(gap_signal.risk_reward_ratio, 2))),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=composite_score,
            trend_score=Decimal('0'),  # Strategy handles this internally
            momentum_score=Decimal('0'),
            volume_score=Decimal('0'),
            pattern_score=Decimal('0'),
            confluence_score=Decimal('0'),
            setup_description=description,
            confluence_factors=confluence_factors,
            timeframe="daily",
            market_regime="Gap & Go",
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                'gap_pct': gap_signal.gap_pct,
                'gap_open_price': gap_signal.gap_open_price,
                'relative_volume': gap_signal.relative_volume,
                'atr': gap_signal.atr,
                'rsi': gap_signal.rsi,
                'is_large_gap': gap_signal.is_large_gap,
                'ema_200_above': gap_signal.ema_200_above,
            },
        )

        # Calculate position size
        signal.suggested_position_size_pct = self.calculate_position_size(
            Decimal(str(self.config.max_risk_per_trade_pct)),
            signal.entry_price,
            signal.stop_loss,
            quality_tier
        )

        return signal

    def _build_confluence_factors(
        self,
        gap_signal: GapSignal,
        candidate: Dict[str, Any]
    ) -> List[str]:
        """Build list of confluence factors"""
        factors = ['Gap & Go Setup']

        # Gap size
        if gap_signal.is_large_gap:
            factors.append(f'Large Gap ({gap_signal.gap_pct:.1f}%)')
        else:
            factors.append(f'Gap Up ({gap_signal.gap_pct:.1f}%)')

        # Volume
        if gap_signal.relative_volume >= 2.0:
            factors.append(f'High Volume ({gap_signal.relative_volume:.1f}x)')
        else:
            factors.append(f'Volume Surge ({gap_signal.relative_volume:.1f}x)')

        # Trend context
        if gap_signal.ema_200_above:
            factors.append('Above EMA-200 (Bullish)')

        # RSI
        if gap_signal.rsi:
            if 50 <= gap_signal.rsi <= 70:
                factors.append(f'Good Momentum (RSI {gap_signal.rsi:.0f})')

        return factors

    def _build_description(
        self,
        gap_signal: GapSignal,
        candidate: Dict[str, Any]
    ) -> str:
        """Build human-readable setup description"""
        ticker = gap_signal.ticker
        gap = gap_signal.gap_pct
        vol = gap_signal.relative_volume

        desc = f"{ticker} gap: "

        if gap_signal.is_large_gap:
            desc += f"Large gap +{gap:.1f}%, "
        else:
            desc += f"+{gap:.1f}% gap, "

        desc += f"{vol:.1f}x volume"

        if gap_signal.ema_200_above:
            desc += ", above EMA-200"

        return desc

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Not used - strategy handles entry level calculation"""
        pass
