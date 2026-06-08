"""
Squeeze Momentum Scanner.

Daily BB/KC squeeze release scanner backed by SqueezeMomentumStrategy so live
scanner output and historical backtests use the same entry logic.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple

from ..base_scanner import BaseScanner, SignalSetup, ScannerConfig, SignalType, QualityTier
from ...strategies.squeeze_momentum import SqueezeMomentumStrategy, SqueezeMomentumSignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class SqueezeMomentumConfig(ScannerConfig):
    bb_length: int = 20
    bb_mult: float = 2.0
    kc_length: int = 20
    kc_mult: float = 1.5
    squeeze_min_bars: int = 3
    squeeze_lookback_bars: int = 8
    require_release_bar: bool = True
    min_momentum_slope_atr: float = 0.03
    min_adx: float = 18.0
    require_adx_rising: bool = True
    min_relative_volume: float = 1.0
    trend_ema: str = "ema_50"
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.5
    min_confidence: float = 0.55
    min_quality_tier: str = "C"
    allow_shorts: bool = True

    max_signals_per_run: int = 50
    min_score_threshold: int = 55


class SqueezeMomentumScanner(BaseScanner):
    """Scan active stock universe for squeeze-release momentum setups."""

    def __init__(self, db_manager, config: SqueezeMomentumConfig = None, scorer=None):
        super().__init__(db_manager, config or SqueezeMomentumConfig(), scorer)
        self.strategy = SqueezeMomentumStrategy(
            bb_length=self.config.bb_length,
            bb_mult=self.config.bb_mult,
            kc_length=self.config.kc_length,
            kc_mult=self.config.kc_mult,
            squeeze_min_bars=self.config.squeeze_min_bars,
            squeeze_lookback_bars=self.config.squeeze_lookback_bars,
            require_release_bar=self.config.require_release_bar,
            min_momentum_slope_atr=self.config.min_momentum_slope_atr,
            min_adx=self.config.min_adx,
            require_adx_rising=self.config.require_adx_rising,
            min_relative_volume=self.config.min_relative_volume,
            trend_ema=self.config.trend_ema,
            stop_loss_atr_mult=self.config.stop_loss_atr_mult,
            take_profit_atr_mult=self.config.take_profit_atr_mult,
            min_confidence=self.config.min_confidence,
            min_quality_tier=self.config.min_quality_tier,
            allow_shorts=self.config.allow_shorts,
        )
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "squeeze_momentum"

    @property
    def description(self) -> str:
        return "BB/KC squeeze release with EMA50, ADX, ATR, and volume filters"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        logger.info("Starting %s scan", self.scanner_name)

        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        tickers = await self.get_all_active_tickers()
        logger.info("Scanning %s active stocks for squeeze momentum", len(tickers))

        signals: List[SignalSetup] = []
        scanned_count = 0

        for ticker in tickers:
            try:
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=calculation_date - timedelta(days=365),
                    end_date=calculation_date,
                    timeframe="1d",
                )
                if df.empty or len(df) < 80:
                    continue

                scanned_count += 1
                sector = await self.data_provider.get_ticker_sector(ticker)
                sqz_signal = self.strategy.scan(df, ticker, sector)
                if not sqz_signal:
                    continue

                signal = self._convert_to_signal_setup(sqz_signal)
                if signal and signal.composite_score >= self.config.min_score_threshold:
                    signals.append(signal)
                    logger.info(
                        "%s: squeeze %s signal, tier=%s, confidence=%.1f%%",
                        ticker,
                        sqz_signal.signal_type,
                        sqz_signal.quality_tier,
                        sqz_signal.confidence * 100,
                    )
            except Exception as exc:
                logger.warning("%s: Error scanning squeeze momentum - %s", ticker, exc)
                continue

        signals.sort(key=lambda x: x.composite_score, reverse=True)
        signals = signals[:self.config.max_signals_per_run]
        high_quality = sum(1 for s in signals if s.is_high_quality)
        logger.info("Scanned %s tickers, generated %s squeeze signals", scanned_count, len(signals))
        self.log_scan_summary(len(tickers), len(signals), high_quality)
        return signals

    def _convert_to_signal_setup(self, sqz_signal: SqueezeMomentumSignal) -> Optional[SignalSetup]:
        composite_score = int(sqz_signal.confidence * 100)
        tier_map = {
            "A+": QualityTier.A_PLUS,
            "A": QualityTier.A,
            "B": QualityTier.B,
            "C": QualityTier.C,
            "D": QualityTier.D,
        }
        quality_tier = tier_map.get(sqz_signal.quality_tier, QualityTier.C)
        signal_type = SignalType.BUY if sqz_signal.signal_type == "BUY" else SignalType.SELL

        entry = Decimal(str(sqz_signal.entry_price))
        stop = Decimal(str(sqz_signal.stop_loss_price))
        tp1 = Decimal(str(sqz_signal.take_profit_price))
        risk = abs(entry - stop)
        risk_percent = (risk / entry * 100) if entry > 0 else Decimal("0")

        trend_score = Decimal(str(min(100, max(0, (sqz_signal.adx or 0) * 3))))
        momentum_score = Decimal(str(min(100, sqz_signal.momentum_slope_atr * 500)))
        volume_score = Decimal(str(min(100, (sqz_signal.relative_volume or 1.0) * 50)))
        pattern_score = Decimal(str(min(100, sqz_signal.squeeze_count / max(self.config.squeeze_lookback_bars, 1) * 100)))
        confluence_score = Decimal(str(composite_score))

        return SignalSetup(
            ticker=sqz_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=sqz_signal.signal_timestamp,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=None,
            risk_reward_ratio=Decimal(str(sqz_signal.risk_reward_ratio)),
            risk_percent=risk_percent,
            composite_score=composite_score,
            quality_tier=quality_tier,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            pattern_score=pattern_score,
            confluence_score=confluence_score,
            setup_description=(
                f"Squeeze release {sqz_signal.signal_type}: "
                f"{sqz_signal.squeeze_count}/{self.config.squeeze_lookback_bars} compression bars, "
                f"slope/ATR {sqz_signal.momentum_slope_atr:.3f}, ADX {sqz_signal.adx}"
            ),
            confluence_factors=[
                "BB/KC squeeze released",
                f"Momentum slope/ATR {sqz_signal.momentum_slope_atr:.3f}",
                f"ADX {sqz_signal.adx}",
                f"Relative volume {sqz_signal.relative_volume}",
            ],
            timeframe="daily",
            market_regime="volatility_expansion",
            suggested_position_size_pct=Decimal(str(min(2.0, max(0.5, sqz_signal.confidence * 2)))),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                "squeeze_count": sqz_signal.squeeze_count,
                "momentum": sqz_signal.momentum,
                "momentum_slope": sqz_signal.momentum_slope,
                "momentum_slope_atr": sqz_signal.momentum_slope_atr,
                "adx": sqz_signal.adx,
                "rsi": sqz_signal.rsi,
                "atr": sqz_signal.atr,
                "relative_volume": sqz_signal.relative_volume,
            },
        )

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Required by BaseScanner. Squeeze scanner uses levels from the
        backtested strategy signal, so this is only a compatibility fallback.
        """
        price = Decimal(str(candidate.get("current_price", 0)))
        if signal_type == SignalType.BUY:
            return price, price * Decimal("0.95"), price * Decimal("1.10"), None
        return price, price * Decimal("1.05"), price * Decimal("0.90"), None
