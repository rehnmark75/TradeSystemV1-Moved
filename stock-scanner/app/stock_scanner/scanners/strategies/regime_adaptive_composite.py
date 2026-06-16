"""
Long-only regime-adaptive composite scanner.

This scanner wraps RegimeAdaptiveCompositeStrategy so live scans and future
backtests can share the same signal logic.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..base_scanner import BaseScanner, QualityTier, ScannerConfig, SignalSetup, SignalType
from ...core.backtest.backtest_data_provider import BacktestDataProvider
from ...strategies.regime_adaptive_composite import (
    RegimeAdaptiveCompositeStrategy,
    RegimeAdaptiveSignal,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeAdaptiveCompositeConfig(ScannerConfig):
    min_score_threshold: int = 62
    max_signals_per_run: int = 30
    lookback_days: int = 365
    min_confidence: float = 0.62
    min_rs_percentile: int = 55
    min_avg_dollar_volume: float = 10_000_000
    min_price: float = 5.0
    max_price: float = 1000.0
    atr_stop_multiplier: float = 1.6
    max_stop_loss_pct: float = 8.0
    max_signal_risk_pct: float = 7.95
    min_trend_relative_volume: float = 1.0
    min_range_relative_volume: float = 0.8
    max_range_adx: float = 30.0
    tp1_rr_ratio: float = 2.4
    tp2_rr_ratio: float = 3.2


class RegimeAdaptiveCompositeScanner(BaseScanner):
    """Scan stocks for long-only trend, compression, and range-reversal setups."""

    def __init__(self, db_manager, config: RegimeAdaptiveCompositeConfig = None, scorer=None):
        super().__init__(db_manager, config or RegimeAdaptiveCompositeConfig(), scorer)
        self.data_provider = BacktestDataProvider(db_manager)
        self.strategy = RegimeAdaptiveCompositeStrategy(
            min_confidence=self.config.min_confidence,
            min_rs_percentile=self.config.min_rs_percentile,
            min_avg_dollar_volume=self.config.min_avg_dollar_volume,
            min_price=self.config.min_price,
            max_price=self.config.max_price,
            max_stop_pct=self.config.max_stop_loss_pct,
            atr_stop_mult=self.config.atr_stop_multiplier,
            take_profit_rr=self.config.tp1_rr_ratio,
            min_trend_relative_volume=self.config.min_trend_relative_volume,
            min_range_relative_volume=self.config.min_range_relative_volume,
            max_range_adx=self.config.max_range_adx,
            max_signal_risk_pct=self.config.max_signal_risk_pct,
        )

    @property
    def scanner_name(self) -> str:
        return "regime_adaptive_composite"

    @property
    def description(self) -> str:
        return "Long-only adaptive scanner for trend, compression, and ranging regimes"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        logger.info("Starting %s scan", self.scanner_name)
        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        candidates = await self._get_candidates(calculation_date)
        signals: List[SignalSetup] = []
        scanned_count = 0

        for candidate in candidates:
            ticker = candidate.get("ticker")
            if not ticker:
                continue
            try:
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=calculation_date - timedelta(days=self.config.lookback_days),
                    end_date=calculation_date,
                    timeframe="1d",
                )
                if df.empty or len(df) < 80:
                    continue
                scanned_count += 1
                composite_signal = self.strategy.scan(df, ticker, candidate)
                if not composite_signal:
                    continue
                signal = self._convert_to_signal_setup(composite_signal)
                if signal and signal.composite_score >= self.config.min_score_threshold:
                    signals.append(signal)
                    logger.info(
                        "%s: long-only %s signal, regime=%s, tier=%s, confidence=%.1f%%",
                        ticker,
                        composite_signal.entry_component,
                        composite_signal.regime,
                        composite_signal.quality_tier,
                        composite_signal.confidence * 100,
                    )
            except Exception as exc:
                logger.warning("%s: Error scanning regime adaptive composite - %s", ticker, exc)

        signals.sort(key=lambda item: item.composite_score, reverse=True)
        signals = signals[:self.config.max_signals_per_run]
        self.log_scan_summary(len(candidates), len(signals), sum(1 for s in signals if s.is_high_quality))
        logger.info("Scanned %s candidates, generated %s long-only composite signals", scanned_count, len(signals))
        return signals

    async def _get_candidates(self, calculation_date) -> List[Dict[str, Any]]:
        query = """
            WITH latest_metrics AS (
                SELECT MAX(calculation_date) AS calculation_date
                FROM stock_screening_metrics
                WHERE calculation_date <= $1
            )
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.rs_percentile, m.rs_trend, m.pct_from_52w_high, m.high_52w,
                m.rsi_14, m.macd_histogram, m.adx, m.sma_20, m.sma_50, m.sma_200,
                m.avg_volume_20, m.avg_dollar_volume,
                i.sector
            FROM stock_screening_metrics m
            JOIN latest_metrics lm ON lm.calculation_date = m.calculation_date
            JOIN stock_instruments i ON i.ticker = m.ticker
            WHERE i.is_active = TRUE
              AND COALESCE(m.current_price, 0) BETWEEN $2 AND $3
              AND COALESCE(m.avg_dollar_volume, 0) >= $4
              AND COALESCE(m.rs_percentile, 0) >= $5
            ORDER BY COALESCE(m.rs_percentile, 0) DESC, COALESCE(m.relative_volume, 0) DESC
            LIMIT 600
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.min_rs_percentile,
        )
        return [dict(row) for row in rows]

    def _convert_to_signal_setup(self, adaptive_signal: RegimeAdaptiveSignal) -> Optional[SignalSetup]:
        entry = Decimal(str(adaptive_signal.entry_price))
        stop = Decimal(str(adaptive_signal.stop_loss_price))
        tp1 = Decimal(str(adaptive_signal.take_profit_price))
        if entry <= 0 or stop <= 0 or tp1 <= 0 or stop >= entry:
            return None

        risk = entry - stop
        tp2 = entry + risk * Decimal(str(self.config.tp2_rr_ratio))
        risk_pct = risk / entry * Decimal("100")
        tier = self._quality_tier(adaptive_signal.quality_tier, adaptive_signal.confluence_score)

        return SignalSetup(
            ticker=adaptive_signal.ticker,
            scanner_name=self.scanner_name,
            signal_type=SignalType.BUY,
            signal_timestamp=adaptive_signal.signal_timestamp,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(adaptive_signal.risk_reward_ratio)),
            risk_percent=Decimal(str(round(float(risk_pct), 2))),
            composite_score=adaptive_signal.confluence_score,
            quality_tier=tier,
            trend_score=Decimal(str(adaptive_signal.trend_score)),
            momentum_score=Decimal(str(adaptive_signal.momentum_score)),
            volume_score=Decimal(str(adaptive_signal.volume_score)),
            pattern_score=Decimal(str(adaptive_signal.pattern_score)),
            confluence_score=Decimal(str(adaptive_signal.confluence_score)),
            setup_description=(
                f"Long-only {adaptive_signal.regime} setup: "
                f"{adaptive_signal.entry_component}"
            ),
            confluence_factors=adaptive_signal.factors,
            timeframe="daily",
            market_regime=adaptive_signal.regime,
            suggested_position_size_pct=self.calculate_position_size(
                Decimal(str(self.config.max_risk_per_trade_pct)),
                entry,
                stop,
                tier,
            ),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=adaptive_signal.raw_data,
        )

    def _quality_tier(self, tier: str, score: int) -> QualityTier:
        mapping = {
            "A+": QualityTier.A_PLUS,
            "A": QualityTier.A,
            "B": QualityTier.B,
            "C": QualityTier.C,
            "D": QualityTier.D,
        }
        return mapping.get(tier, QualityTier.from_score(score))

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        price = Decimal(str(candidate.get("current_price") or 0))
        atr = Decimal(str(candidate.get("atr_14") or 0))
        if price <= 0:
            return Decimal("0"), Decimal("0"), Decimal("0"), None
        if atr <= 0:
            atr = price * Decimal("0.03")
        stop = self.calculate_atr_based_stop(price, atr, SignalType.BUY)
        tp1, tp2 = self.calculate_take_profits(price, stop, SignalType.BUY)
        return price, stop, tp1, tp2
