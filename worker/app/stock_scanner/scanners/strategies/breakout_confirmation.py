"""
Breakout Confirmation Scanner

Uses the backtested BreakoutConfirmationStrategy for signal detection.
This ensures identical entry logic between backtesting and live scanning.

Entry Criteria (from backtested strategy):
1. Near 52-week high (within 5%) or making new high
2. Volume surge (> 1.5x average)
3. Positive gap or strong close (>2% daily gain)
4. MACD momentum expanding (positive histogram)
5. ADX > 22 (trending market)
6. RSI 50-75 (momentum but not overbought)

Stop Logic:
- 2.2x ATR below entry (wider for breakout volatility)
- Max 8% stop distance

Target:
- 4.5x ATR (2:1 R:R)

Best For:
- Strong momentum markets
- Stocks with catalyst (earnings beat, news)
- Sector leaders breaking out
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
from ...strategies.breakout_confirmation import BreakoutConfirmationStrategy, BreakoutSignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class BreakoutConfig(ScannerConfig):
    """Configuration for Breakout Scanner - uses backtested strategy defaults"""

    # Strategy parameters (match backtested strategy)
    stop_loss_atr_mult: float = 2.2   # 2.2x ATR stop loss
    take_profit_atr_mult: float = 4.5  # 4.5x ATR take profit
    max_pct_from_high: float = 5.0     # Within 5% of 52W high
    min_relative_volume: float = 1.5   # Volume surge threshold
    min_daily_change: float = 2.0      # Minimum daily gain %
    min_adx: float = 22.0              # ADX threshold
    min_rsi: float = 50.0              # RSI lower bound
    max_rsi: float = 75.0              # RSI upper bound


class BreakoutConfirmationScanner(BaseScanner):
    """
    Scans for volume-confirmed breakout setups using the backtested strategy.

    Philosophy:
    - Breakouts need volume confirmation to be valid
    - Near 52W highs means limited overhead resistance
    - Gaps add conviction (institutional interest)
    - Failed breakouts get stopped out quickly

    Uses the exact same logic as the backtested strategy (PF varies by market).
    """

    def __init__(
        self,
        db_manager,
        config: BreakoutConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or BreakoutConfig(), scorer)

        # Initialize the backtested strategy
        self.strategy = BreakoutConfirmationStrategy(
            stop_loss_atr_mult=self.config.stop_loss_atr_mult,
            take_profit_atr_mult=self.config.take_profit_atr_mult,
            max_pct_from_high=self.config.max_pct_from_high,
            min_relative_volume=self.config.min_relative_volume,
            min_daily_change=self.config.min_daily_change,
            min_adx=self.config.min_adx,
            min_rsi=self.config.min_rsi,
            max_rsi=self.config.max_rsi,
        )

        # Initialize data provider for fetching candles with indicators
        self.data_provider = BacktestDataProvider(db_manager)

        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "breakout_confirmation"

    @property
    def description(self) -> str:
        return "Volume-confirmed breakouts - uses backtested strategy"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute Breakout scan using the backtested strategy.

        Steps:
        1. Get ALL active tickers from stock_instruments
        2. For each ticker, fetch candle data with indicators
        3. Run BreakoutConfirmationStrategy.scan() - exact backtest logic
        4. Convert BreakoutSignal to SignalSetup format
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
        logger.info(f"Scanning {len(tickers)} active stocks for breakouts")

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

                if df.empty or len(df) < 260:  # Need 252 trading days for 52W high
                    continue

                scanned_count += 1

                # Run the exact backtested strategy logic
                breakout_signal = self.strategy.scan(df, ticker, sector)

                if breakout_signal:
                    # Convert to SignalSetup format (pass ticker info as minimal candidate)
                    candidate = {'ticker': ticker, 'sector': sector}
                    signal = self._convert_to_signal_setup(breakout_signal, candidate)
                    if signal:
                        signals.append(signal)
                        logger.info(f"{ticker}: Signal generated - {breakout_signal.quality_tier} tier, "
                                  f"confidence {breakout_signal.confidence:.1%}")

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
        breakout_signal: BreakoutSignal,
        candidate: Dict[str, Any]
    ) -> Optional[SignalSetup]:
        """
        Convert a BreakoutSignal from the strategy to SignalSetup format.

        The strategy already validated all entry conditions, so we just
        need to format the output for database storage.
        """
        # Convert confidence to composite score (0-100)
        composite_score = int(breakout_signal.confidence * 100)

        # Get quality tier from strategy
        tier_map = {
            'A+': QualityTier.A_PLUS,
            'A': QualityTier.A,
            'B': QualityTier.B,
            'C': QualityTier.C,
            'D': QualityTier.D
        }
        quality_tier = tier_map.get(breakout_signal.quality_tier, QualityTier.B)

        # Calculate risk percent
        entry = breakout_signal.entry_price
        stop = breakout_signal.stop_loss_price
        risk_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0

        # Determine signal type
        signal_type = SignalType.BUY if breakout_signal.signal_type == 'BUY' else SignalType.SELL

        # Build confluence factors
        confluence_factors = self._build_confluence_factors(breakout_signal, candidate)

        # Build description
        description = self._build_description(breakout_signal, candidate)

        # Create SignalSetup
        signal = SignalSetup(
            ticker=breakout_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=breakout_signal.signal_timestamp,
            entry_price=Decimal(str(round(entry, 4))),
            stop_loss=Decimal(str(round(stop, 4))),
            take_profit_1=Decimal(str(round(breakout_signal.take_profit_price, 4))),
            take_profit_2=None,
            risk_reward_ratio=Decimal(str(round(breakout_signal.risk_reward_ratio, 2))),
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
            market_regime="Breakout",
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                'pct_from_52w_high': breakout_signal.pct_from_52w_high,
                'relative_volume': breakout_signal.relative_volume,
                'daily_change_pct': breakout_signal.daily_change_pct,
                'atr': breakout_signal.atr,
                'rsi': breakout_signal.rsi,
                'adx': breakout_signal.adx,
                'has_gap_up': breakout_signal.has_gap_up,
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
        breakout_signal: BreakoutSignal,
        candidate: Dict[str, Any]
    ) -> List[str]:
        """Build list of confluence factors"""
        factors = ['Breakout Setup']

        # Position from high
        if breakout_signal.pct_from_52w_high >= 0:
            factors.append('New 52W High')
        elif breakout_signal.pct_from_52w_high >= -2:
            factors.append('Near 52W High')

        # Volume
        if breakout_signal.relative_volume >= 2.0:
            factors.append(f'High Volume ({breakout_signal.relative_volume:.1f}x)')
        else:
            factors.append(f'Volume Surge ({breakout_signal.relative_volume:.1f}x)')

        # Gap
        if breakout_signal.has_gap_up:
            factors.append('Gap Up')

        # Daily change
        if breakout_signal.daily_change_pct >= 3.0:
            factors.append(f'Strong Move (+{breakout_signal.daily_change_pct:.1f}%)')

        # ADX
        if breakout_signal.adx and breakout_signal.adx >= 30:
            factors.append(f'Strong Trend (ADX {breakout_signal.adx:.0f})')

        return factors

    def _build_description(
        self,
        breakout_signal: BreakoutSignal,
        candidate: Dict[str, Any]
    ) -> str:
        """Build human-readable setup description"""
        ticker = breakout_signal.ticker
        pct = breakout_signal.pct_from_52w_high
        vol = breakout_signal.relative_volume
        change = breakout_signal.daily_change_pct

        desc = f"{ticker} breakout: "

        if pct >= 0:
            desc += "New 52W high, "
        else:
            desc += f"{abs(pct):.1f}% from high, "

        desc += f"{vol:.1f}x volume, +{change:.1f}% today"

        if breakout_signal.has_gap_up:
            desc += ", gap up"

        return desc

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Not used - strategy handles entry level calculation"""
        pass
