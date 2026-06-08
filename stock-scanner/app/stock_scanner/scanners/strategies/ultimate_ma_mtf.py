"""
Ultimate MA MTF Scanner.

Scanner wrapper around UltimateMAMTFStrategy. It converts ChrisMoody-style
line-color close conditions into ranked stock scanner signals.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple

from ..base_scanner import BaseScanner, SignalSetup, ScannerConfig, SignalType, QualityTier
from ...strategies.ultimate_ma_mtf import UltimateMAMTFStrategy, UltimateMASignal
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class UltimateMAMTFConfig(ScannerConfig):
    ma_type: int = 2
    ma_length: int = 20
    t3_factor: float = 0.7
    use_second_ma: bool = False
    second_ma_type: int = 2
    second_ma_length: int = 50
    second_t3_factor: float = 0.7
    smoothing_bars: int = 2
    trigger_mode: str = "line_color_close"
    require_ma_slope: bool = True
    require_second_ma_trend: bool = False
    max_price_distance_atr: float = 1.5
    min_ma_slope_atr: float = 0.01
    min_adx: float = 15.0
    min_relative_volume: float = 0.8
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 2.5
    min_confidence: float = 0.55
    min_quality_tier: str = "C"
    allow_shorts: bool = True

    max_signals_per_run: int = 50
    min_score_threshold: int = 55


class UltimateMAMTFScanner(BaseScanner):
    """Scan active stock universe for green/red MA line close setups."""

    def __init__(self, db_manager, config: UltimateMAMTFConfig = None, scorer=None):
        super().__init__(db_manager, config or UltimateMAMTFConfig(), scorer)
        self.strategy = UltimateMAMTFStrategy(
            ma_type=self.config.ma_type,
            ma_length=self.config.ma_length,
            t3_factor=self.config.t3_factor,
            use_second_ma=self.config.use_second_ma,
            second_ma_type=self.config.second_ma_type,
            second_ma_length=self.config.second_ma_length,
            second_t3_factor=self.config.second_t3_factor,
            smoothing_bars=self.config.smoothing_bars,
            trigger_mode=self.config.trigger_mode,
            require_ma_slope=self.config.require_ma_slope,
            require_second_ma_trend=self.config.require_second_ma_trend,
            max_price_distance_atr=self.config.max_price_distance_atr,
            min_ma_slope_atr=self.config.min_ma_slope_atr,
            min_adx=self.config.min_adx,
            min_relative_volume=self.config.min_relative_volume,
            stop_loss_atr_mult=self.config.stop_loss_atr_mult,
            take_profit_atr_mult=self.config.take_profit_atr_mult,
            min_confidence=self.config.min_confidence,
            min_quality_tier=self.config.min_quality_tier,
            allow_shorts=self.config.allow_shorts,
        )
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "ultimate_ma_mtf"

    @property
    def description(self) -> str:
        return "Buy close above green/rising MA, sell close below red/falling MA with ADX, ATR, and volume filters"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        logger.info("Starting %s scan", self.scanner_name)

        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        tickers = await self.get_all_active_tickers()
        logger.info("Scanning %s active stocks for ultimate MA setups", len(tickers))

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
                ma_signal = self.strategy.scan(df, ticker, sector)
                if not ma_signal:
                    continue

                signal = self._convert_to_signal_setup(ma_signal)
                if signal and signal.composite_score >= self.config.min_score_threshold:
                    signals.append(signal)
                    logger.info(
                        "%s: ultimate MA %s signal, trigger=%s, tier=%s, confidence=%.1f%%",
                        ticker,
                        ma_signal.signal_type,
                        ma_signal.trigger,
                        ma_signal.quality_tier,
                        ma_signal.confidence * 100,
                    )
            except Exception as exc:
                logger.warning("%s: Error scanning ultimate MA - %s", ticker, exc)
                continue

        signals.sort(key=lambda x: x.composite_score, reverse=True)
        signals = signals[:self.config.max_signals_per_run]
        high_quality = sum(1 for s in signals if s.is_high_quality)
        logger.info("Scanned %s tickers, generated %s ultimate MA signals", scanned_count, len(signals))
        self.log_scan_summary(len(tickers), len(signals), high_quality)
        return signals

    def _convert_to_signal_setup(self, ma_signal: UltimateMASignal) -> Optional[SignalSetup]:
        composite_score = int(ma_signal.confidence * 100)
        tier_map = {
            "A+": QualityTier.A_PLUS,
            "A": QualityTier.A,
            "B": QualityTier.B,
            "C": QualityTier.C,
            "D": QualityTier.D,
        }
        quality_tier = tier_map.get(ma_signal.quality_tier, QualityTier.C)
        signal_type = SignalType.BUY if ma_signal.signal_type == "BUY" else SignalType.SELL

        entry = Decimal(str(ma_signal.entry_price))
        stop = Decimal(str(ma_signal.stop_loss_price))
        tp1 = Decimal(str(ma_signal.take_profit_price))
        risk = abs(entry - stop)
        risk_percent = (risk / entry * 100) if entry > 0 else Decimal("0")

        trend_score = Decimal(str(min(100, max(0, (ma_signal.adx or 0) * 3))))
        momentum_score = Decimal(str(min(100, ma_signal.ma_slope_atr * 600)))
        volume_score = Decimal(str(min(100, (ma_signal.relative_volume or 1.0) * 50)))
        distance = ma_signal.price_distance_atr or 0.0
        pattern_score = Decimal(str(max(0, min(100, 100 - distance * 35))))
        confluence_score = Decimal(str(composite_score))

        factors = [
            f"{ma_signal.trigger} trigger",
            f"{ma_signal.ma_type}{ma_signal.ma_length} slope/ATR {ma_signal.ma_slope_atr:.3f}",
            f"ADX {ma_signal.adx}",
            f"Relative volume {ma_signal.relative_volume}",
        ]
        if ma_signal.second_ma_type and ma_signal.second_ma_length:
            factors.append(f"Trend aligned with {ma_signal.second_ma_type}{ma_signal.second_ma_length}")

        return SignalSetup(
            ticker=ma_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=ma_signal.signal_timestamp,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=None,
            risk_reward_ratio=Decimal(str(ma_signal.risk_reward_ratio)),
            risk_percent=risk_percent,
            composite_score=composite_score,
            quality_tier=quality_tier,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volume_score=volume_score,
            pattern_score=pattern_score,
            confluence_score=confluence_score,
            setup_description=(
                f"Ultimate MA {ma_signal.signal_type}: {ma_signal.trigger}, "
                f"{ma_signal.ma_type}{ma_signal.ma_length}={ma_signal.ma_value}, "
                f"slope/ATR {ma_signal.ma_slope_atr:.3f}"
            ),
            confluence_factors=factors,
            timeframe="daily",
            market_regime="ma_reclaim",
            suggested_position_size_pct=Decimal(str(min(2.0, max(0.5, ma_signal.confidence * 2)))),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data={
                "trigger": ma_signal.trigger,
                "ma_type": ma_signal.ma_type,
                "ma_length": ma_signal.ma_length,
                "ma_value": ma_signal.ma_value,
                "second_ma_type": ma_signal.second_ma_type,
                "second_ma_length": ma_signal.second_ma_length,
                "second_ma_value": ma_signal.second_ma_value,
                "ma_slope_atr": ma_signal.ma_slope_atr,
                "price_distance_atr": ma_signal.price_distance_atr,
                "adx": ma_signal.adx,
                "rsi": ma_signal.rsi,
                "atr": ma_signal.atr,
                "relative_volume": ma_signal.relative_volume,
            },
        )

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Required by BaseScanner. Ultimate MA scanner uses levels from the
        backtested strategy signal, so this is only a compatibility fallback.
        """
        price = Decimal(str(candidate.get("current_price", 0)))
        if signal_type == SignalType.BUY:
            return price, price * Decimal("0.95"), price * Decimal("1.10"), None
        return price, price * Decimal("1.05"), price * Decimal("0.90"), None
