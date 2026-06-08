"""
Additional stock scanners built on the shared BaseScanner contract.

These scanners lean on data already produced by the daily stock pipeline:
screening metrics, relative strength, sector analysis, fundamentals, news,
and synthesized daily candles.
"""

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base_scanner import BaseScanner, QualityTier, ScannerConfig, SignalSetup, SignalType
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _quality_tier(score: int) -> QualityTier:
    return QualityTier.from_score(score)


@dataclass
class DataDrivenScannerConfig(ScannerConfig):
    max_signals_per_run: int = 30
    min_score_threshold: int = 65
    min_avg_dollar_volume: float = 10_000_000
    min_price: float = 5.0
    max_price: float = 1000.0
    lookback_days: int = 365


class DataDrivenScanner(BaseScanner):
    """Shared helpers for scanners that combine SQL candidates and candles."""

    def __init__(self, db_manager, config: DataDrivenScannerConfig = None, scorer=None):
        super().__init__(db_manager, config or DataDrivenScannerConfig(), scorer)
        self.data_provider = BacktestDataProvider(db_manager)

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        logger.info("Starting %s scan for %s", self.scanner_name, calculation_date)
        candidates = await self._get_candidates(calculation_date)
        signals: List[SignalSetup] = []

        for candidate in candidates:
            ticker = candidate.get("ticker")
            if not ticker:
                continue

            try:
                df = await self._get_candles(ticker, calculation_date)
                if df.empty or len(df) < self.minimum_bars:
                    continue
                signal = self._build_signal(candidate, df, calculation_date)
                if signal:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("%s: error scanning %s: %s", self.scanner_name, ticker, exc)

        signals.sort(key=lambda item: item.composite_score, reverse=True)
        signals = signals[: self.config.max_signals_per_run]
        self.log_scan_summary(len(candidates), len(signals), sum(1 for s in signals if s.is_high_quality))
        return signals

    @property
    def minimum_bars(self) -> int:
        return 60

    async def _get_candles(self, ticker: str, calculation_date: date) -> pd.DataFrame:
        return await self.data_provider.get_historical_data(
            ticker=ticker,
            start_date=calculation_date - timedelta(days=self.config.lookback_days),
            end_date=calculation_date,
            timeframe="1d",
        )

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _build_signal(
        self,
        candidate: Dict[str, Any],
        candles: pd.DataFrame,
        calculation_date: date,
    ) -> Optional[SignalSetup]:
        raise NotImplementedError

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        price = Decimal(str(round(_as_float(candidate.get("current_price")), 4)))
        atr = Decimal(str(round(_as_float(candidate.get("atr_14")), 4)))
        if atr <= 0:
            atr = price * Decimal("0.03")

        stop = self.calculate_atr_based_stop(price, atr, signal_type)
        tp1, tp2 = self.calculate_take_profits(price, stop, signal_type)
        return price, stop, tp1, tp2

    def _make_signal(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType,
        score: int,
        description: str,
        factors: List[str],
        calculation_date: date,
        market_regime: str = "",
    ) -> Optional[SignalSetup]:
        if score < self.config.min_score_threshold:
            return None

        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)
        if entry <= 0 or stop <= 0 or tp1 <= 0:
            return None

        risk = abs(entry - stop)
        reward = abs(tp1 - entry)
        rr = reward / risk if risk > 0 else Decimal("0")
        risk_pct = (risk / entry) * Decimal("100") if entry > 0 else Decimal("0")
        tier = _quality_tier(score)

        return SignalSetup(
            ticker=str(candidate["ticker"]),
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=datetime.combine(calculation_date, datetime.min.time()),
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(round(float(rr), 2))),
            risk_percent=Decimal(str(round(float(risk_pct), 2))),
            composite_score=score,
            quality_tier=tier,
            trend_score=Decimal(str(min(100, score))),
            momentum_score=Decimal(str(min(100, score))),
            volume_score=Decimal(str(min(100, int(_as_float(candidate.get("relative_volume")) * 30)))),
            pattern_score=Decimal(str(min(100, score))),
            confluence_score=Decimal(str(score)),
            setup_description=description,
            confluence_factors=factors,
            timeframe="daily",
            market_regime=market_regime,
            suggested_position_size_pct=self.calculate_position_size(
                Decimal(str(self.config.max_risk_per_trade_pct)),
                entry,
                stop,
                tier,
            ),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=dict(candidate),
        )

    def _base_candidate_filters(self) -> str:
        return """
            AND m.current_price BETWEEN $2 AND $3
            AND COALESCE(m.avg_dollar_volume, 0) >= $4
        """


@dataclass
class EarningsDriftConfig(DataDrivenScannerConfig):
    earnings_window_days: int = 5
    min_gap_or_move_pct: float = 2.0
    min_relative_volume: float = 1.2
    min_rs_percentile: int = 55


class EarningsDriftScanner(DataDrivenScanner):
    """Post-earnings drift scanner for stocks that hold a catalyst move."""

    def __init__(self, db_manager, config: EarningsDriftConfig = None, scorer=None):
        super().__init__(db_manager, config or EarningsDriftConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "earnings_drift"

    @property
    def description(self) -> str:
        return "Post-earnings drift continuation after a held catalyst move"

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = f"""
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.rs_percentile, m.rs_trend, m.gap_percent, m.gap_signal,
                m.rsi_14, m.macd_histogram, m.adx, m.sma_20, m.sma_50,
                m.tv_overall_signal, m.tv_overall_score,
                i.earnings_date, i.earnings_date_estimated, i.sector,
                i.short_percent_float
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON i.ticker = m.ticker
            WHERE m.calculation_date = $1
              AND i.is_active = TRUE
              AND i.earnings_date IS NOT NULL
              AND i.earnings_date BETWEEN $1::date - ($5::int * INTERVAL '1 day') AND $1::date
              {self._base_candidate_filters()}
              AND COALESCE(m.relative_volume, 0) >= $6
              AND COALESCE(m.rs_percentile, 0) >= $7
              AND (
                    ABS(COALESCE(m.gap_percent, 0)) >= $8
                 OR ABS(COALESCE(m.price_change_1d, 0)) >= $8
                 OR ABS(COALESCE(m.price_change_5d, 0)) >= $8
              )
            ORDER BY COALESCE(m.relative_volume, 0) DESC, COALESCE(m.rs_percentile, 0) DESC
            LIMIT 250
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.earnings_window_days,
            self.config.min_relative_volume,
            self.config.min_rs_percentile,
            self.config.min_gap_or_move_pct,
        )
        return [dict(row) for row in rows]

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        close = _as_float(candles.iloc[-1]["close"])
        open_price = _as_float(candles.iloc[-1]["open"])
        prev_close = _as_float(candles.iloc[-2]["close"])
        if close <= 0 or prev_close <= 0:
            return None

        direction = SignalType.BUY if close >= prev_close else SignalType.SELL
        held_gap = close > open_price and close > prev_close if direction == SignalType.BUY else close < open_price and close < prev_close
        if not held_gap:
            return None

        score = 55
        score += min(15, int(abs(_as_float(candidate.get("price_change_1d"))) * 2))
        score += min(15, int(_as_float(candidate.get("relative_volume")) * 5))
        score += min(10, int(_as_float(candidate.get("rs_percentile")) / 10))
        if str(candidate.get("tv_overall_signal") or "").upper() in {"BUY", "STRONG BUY"} and direction == SignalType.BUY:
            score += 8
        if str(candidate.get("tv_overall_signal") or "").upper() in {"SELL", "STRONG SELL"} and direction == SignalType.SELL:
            score += 8

        candidate["current_price"] = close
        description = "Post-earnings drift continuation with held catalyst move"
        factors = [
            f"Earnings date: {candidate.get('earnings_date')}",
            f"1D move: {_as_float(candidate.get('price_change_1d')):.1f}%",
            f"Relative volume: {_as_float(candidate.get('relative_volume')):.1f}x",
            f"RS percentile: {int(_as_float(candidate.get('rs_percentile')))}",
        ]
        return self._make_signal(candidate, direction, min(score, 100), description, factors, calculation_date)


@dataclass
class ShortSqueezeConfig(DataDrivenScannerConfig):
    min_short_percent: float = 15.0
    min_relative_volume: float = 1.5
    min_price_change_1d: float = 2.0
    min_rs_percentile: int = 50


class ShortSqueezeBreakoutScanner(DataDrivenScanner):
    """High short-interest names breaking resistance on unusual volume."""

    def __init__(self, db_manager, config: ShortSqueezeConfig = None, scorer=None):
        super().__init__(db_manager, config or ShortSqueezeConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "short_squeeze_breakout"

    @property
    def description(self) -> str:
        return "Short squeeze breakout candidates with high short float and volume expansion"

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = f"""
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.rs_percentile, m.rs_trend, m.high_low_signal, m.pct_from_52w_high,
                m.rsi_14, m.macd_histogram, m.adx, m.sma_20, m.sma_50, m.sma_200,
                i.short_percent_float, i.short_ratio, i.shares_short, i.sector
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON i.ticker = m.ticker
            WHERE m.calculation_date = $1
              AND i.is_active = TRUE
              {self._base_candidate_filters()}
              AND COALESCE(i.short_percent_float, 0) >= $5
              AND COALESCE(m.relative_volume, 0) >= $6
              AND COALESCE(m.price_change_1d, 0) >= $7
              AND COALESCE(m.rs_percentile, 0) >= $8
            ORDER BY i.short_percent_float DESC, m.relative_volume DESC
            LIMIT 250
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.min_short_percent,
            self.config.min_relative_volume,
            self.config.min_price_change_1d,
            self.config.min_rs_percentile,
        )
        return [dict(row) for row in rows]

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        close = _as_float(candles.iloc[-1]["close"])
        prior_high = float(candles["high"].iloc[-21:-1].max())
        if close <= prior_high:
            return None

        score = 50
        score += min(20, int(_as_float(candidate.get("short_percent_float"))))
        score += min(15, int(_as_float(candidate.get("relative_volume")) * 5))
        score += min(10, int(_as_float(candidate.get("price_change_1d")) * 2))
        score += 8 if _as_float(candidate.get("macd_histogram")) > 0 else 0
        score += 7 if _as_float(candidate.get("adx")) >= 20 else 0

        candidate["current_price"] = close
        factors = [
            f"Short float: {_as_float(candidate.get('short_percent_float')):.1f}%",
            f"Short ratio: {_as_float(candidate.get('short_ratio')):.1f}",
            f"Break above 20D high: {prior_high:.2f}",
            f"Relative volume: {_as_float(candidate.get('relative_volume')):.1f}x",
        ]
        return self._make_signal(
            candidate,
            SignalType.BUY,
            min(score, 100),
            "High short-interest breakout with squeeze pressure",
            factors,
            calculation_date,
        )


@dataclass
class SectorRotationConfig(DataDrivenScannerConfig):
    min_sector_rs_percentile: int = 60
    min_stock_rs_percentile: int = 70
    min_relative_volume: float = 1.0


class SectorRotationLeaderScanner(DataDrivenScanner):
    """Leadership stocks inside improving or leading sectors."""

    def __init__(self, db_manager, config: SectorRotationConfig = None, scorer=None):
        super().__init__(db_manager, config or SectorRotationConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "sector_rotation_leader"

    @property
    def description(self) -> str:
        return "Relative strength leaders in improving or leading sectors"

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = f"""
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.price_change_20d, m.rs_percentile, m.rs_trend,
                m.rsi_14, m.macd_histogram, m.adx, m.sma_20, m.sma_50, m.sma_200,
                m.tv_overall_signal, m.tv_overall_score,
                i.sector,
                s.rs_percentile AS sector_rs_percentile,
                s.rs_trend AS sector_rs_trend,
                s.sector_stage
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON i.ticker = m.ticker
            JOIN sector_analysis s ON s.sector = i.sector AND s.calculation_date = m.calculation_date
            WHERE m.calculation_date = $1
              AND i.is_active = TRUE
              {self._base_candidate_filters()}
              AND COALESCE(m.relative_volume, 0) >= $5
              AND COALESCE(m.rs_percentile, 0) >= $6
              AND COALESCE(s.rs_percentile, 0) >= $7
              AND COALESCE(s.sector_stage, '') IN ('leading', 'improving')
            ORDER BY s.rs_percentile DESC, m.rs_percentile DESC
            LIMIT 250
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.min_relative_volume,
            self.config.min_stock_rs_percentile,
            self.config.min_sector_rs_percentile,
        )
        return [dict(row) for row in rows]

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        close = _as_float(candles.iloc[-1]["close"])
        sma20 = _as_float(candidate.get("sma_20"))
        sma50 = _as_float(candidate.get("sma_50"))
        if not (close > sma50 and (sma20 == 0 or close >= sma20 * 0.98)):
            return None

        score = 45
        score += min(20, int(_as_float(candidate.get("rs_percentile")) / 5))
        score += min(15, int(_as_float(candidate.get("sector_rs_percentile")) / 7))
        score += 10 if candidate.get("sector_stage") == "leading" else 6
        score += 8 if str(candidate.get("rs_trend") or "") == "improving" else 0
        score += 7 if _as_float(candidate.get("macd_histogram")) > 0 else 0

        candidate["current_price"] = close
        factors = [
            f"Sector: {candidate.get('sector')}",
            f"Sector stage: {candidate.get('sector_stage')}",
            f"Sector RS: {int(_as_float(candidate.get('sector_rs_percentile')))}",
            f"Stock RS: {int(_as_float(candidate.get('rs_percentile')))}",
        ]
        return self._make_signal(candidate, SignalType.BUY, min(score, 100), "Sector rotation leadership setup", factors, calculation_date)


@dataclass
class VolatilityContractionConfig(DataDrivenScannerConfig):
    min_score_threshold: int = 70
    max_signals_per_run: int = 20
    contraction_lookback: int = 15
    min_relative_volume: float = 1.3
    max_recent_range_pct: float = 8.0
    min_rs_percentile: int = 60


class VolatilityContractionBreakoutScanner(DataDrivenScanner):
    """Tight range contraction followed by a volume breakout."""

    def __init__(self, db_manager, config: VolatilityContractionConfig = None, scorer=None):
        super().__init__(db_manager, config or VolatilityContractionConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "volatility_contraction_breakout"

    @property
    def description(self) -> str:
        return "Volatility contraction breakout with range compression and volume expansion"

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = f"""
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.rs_percentile, m.rs_trend, m.rsi_14, m.macd_histogram,
                m.adx, m.sma_20, m.sma_50, m.sma_200, m.avg_volume_20,
                i.sector
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON i.ticker = m.ticker
            WHERE m.calculation_date = $1
              AND i.is_active = TRUE
              {self._base_candidate_filters()}
              AND COALESCE(m.relative_volume, 0) >= $5
              AND COALESCE(m.rs_percentile, 0) >= $6
            ORDER BY COALESCE(m.rs_percentile, 0) DESC
            LIMIT 200
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.min_relative_volume,
            self.config.min_rs_percentile,
        )
        return [dict(row) for row in rows]

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        lookback = self.config.contraction_lookback
        if len(candles) < lookback + 30:
            return None

        close = _as_float(candles.iloc[-1]["close"])
        prior_high = float(candles["high"].iloc[-lookback - 1:-1].max())
        recent_high = float(candles["high"].iloc[-lookback - 1:-1].max())
        recent_low = float(candles["low"].iloc[-lookback - 1:-1].min())
        recent_range_pct = ((recent_high - recent_low) / close) * 100 if close > 0 else 999
        ranges = (candles["high"] - candles["low"]) / candles["close"] * 100
        recent_avg_range = float(ranges.iloc[-6:-1].mean())
        older_avg_range = float(ranges.iloc[-26:-6].mean())
        volume = _as_float(candles.iloc[-1]["volume"])
        avg_volume = _as_float(candidate.get("avg_volume_20")) or float(candles["volume"].iloc[-21:-1].mean())

        if close <= prior_high or recent_range_pct > self.config.max_recent_range_pct:
            return None
        if older_avg_range <= 0 or recent_avg_range > older_avg_range * 0.85:
            return None
        if avg_volume <= 0 or volume < avg_volume * self.config.min_relative_volume:
            return None

        score = 55
        score += min(15, int((older_avg_range / max(recent_avg_range, 0.1)) * 4))
        score += min(12, int((volume / avg_volume) * 4))
        score += min(10, int(_as_float(candidate.get("rs_percentile")) / 10))
        score += 8 if _as_float(candidate.get("macd_histogram")) > 0 else 0

        candidate["current_price"] = close
        factors = [
            f"Break above {lookback}D range high: {prior_high:.2f}",
            f"Range contraction: {recent_avg_range:.1f}% vs {older_avg_range:.1f}%",
            f"Volume: {volume / avg_volume:.1f}x average",
            f"RS percentile: {int(_as_float(candidate.get('rs_percentile')))}",
        ]
        return self._make_signal(candidate, SignalType.BUY, min(score, 100), "Volatility contraction breakout", factors, calculation_date)


@dataclass
class HighRetestConfig(DataDrivenScannerConfig):
    max_pct_from_high: float = 8.0
    max_retest_depth_pct: float = 12.0
    min_rs_percentile: int = 55


class HighRetestScanner(DataDrivenScanner):
    """52-week high breakout retests that reclaim short-term structure."""

    def __init__(self, db_manager, config: HighRetestConfig = None, scorer=None):
        super().__init__(db_manager, config or HighRetestConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "high_retest"

    @property
    def description(self) -> str:
        return "52-week high breakout retest and reclaim setup"

    @property
    def minimum_bars(self) -> int:
        return 220

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = f"""
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.rs_percentile, m.rs_trend, m.pct_from_52w_high, m.high_52w,
                m.rsi_14, m.macd_histogram, m.adx, m.sma_20, m.sma_50, m.sma_200,
                i.sector
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON i.ticker = m.ticker
            WHERE m.calculation_date = $1
              AND i.is_active = TRUE
              {self._base_candidate_filters()}
              AND COALESCE(m.rs_percentile, 0) >= $5
              AND COALESCE(m.pct_from_52w_high, -100) >= ($6::numeric * -1)
            ORDER BY m.pct_from_52w_high DESC, m.rs_percentile DESC
            LIMIT 300
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.min_rs_percentile,
            self.config.max_pct_from_high,
        )
        return [dict(row) for row in rows]

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        close = _as_float(candles.iloc[-1]["close"])
        sma20 = _as_float(candidate.get("sma_20"))
        high_52w = _as_float(candidate.get("high_52w")) or float(candles["high"].iloc[-252:].max())
        recent_low = float(candles["low"].iloc[-15:].min())
        retest_depth = ((high_52w - recent_low) / high_52w) * 100 if high_52w > 0 else 999
        reclaimed = close > sma20 and close > float(candles["close"].iloc[-2])

        if retest_depth > self.config.max_retest_depth_pct or not reclaimed:
            return None

        score = 55
        score += min(15, int(_as_float(candidate.get("rs_percentile")) / 7))
        score += 12 if _as_float(candidate.get("pct_from_52w_high")) >= -3 else 6
        score += 10 if _as_float(candidate.get("macd_histogram")) > 0 else 0
        score += min(8, int(_as_float(candidate.get("relative_volume")) * 3))

        candidate["current_price"] = close
        factors = [
            f"Within {abs(_as_float(candidate.get('pct_from_52w_high'))):.1f}% of 52W high",
            f"Retest depth: {retest_depth:.1f}%",
            "Reclaimed SMA20",
            f"RS percentile: {int(_as_float(candidate.get('rs_percentile')))}",
        ]
        return self._make_signal(candidate, SignalType.BUY, min(score, 100), "52-week high retest reclaim", factors, calculation_date)


@dataclass
class RelativeStrengthLeaderConfig(DataDrivenScannerConfig):
    min_rs_percentile: int = 85
    min_rs_5d_gain: int = 10
    min_relative_volume: float = 1.0


class RelativeStrengthLeaderScanner(DataDrivenScanner):
    """Fresh RS leaders with improving rank and constructive technicals."""

    def __init__(self, db_manager, config: RelativeStrengthLeaderConfig = None, scorer=None):
        super().__init__(db_manager, config or RelativeStrengthLeaderConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "relative_strength_leader"

    @property
    def description(self) -> str:
        return "Fresh relative strength leaders with improving RS and constructive trend"

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = f"""
            WITH prev AS (
                SELECT ticker, rs_percentile
                FROM stock_screening_metrics
                WHERE calculation_date = $1::date - INTERVAL '5 days'
            )
            SELECT
                m.ticker, m.current_price, m.atr_14, m.atr_percent,
                m.relative_volume, m.price_change_1d, m.price_change_5d,
                m.price_change_20d, m.rs_percentile, m.rs_trend,
                COALESCE(m.rs_percentile, 0) - COALESCE(prev.rs_percentile, 0) AS rs_5d_change,
                m.rsi_14, m.macd_histogram, m.adx, m.sma_20, m.sma_50, m.sma_200,
                m.tv_overall_signal, m.tv_overall_score,
                i.sector
            FROM stock_screening_metrics m
            JOIN stock_instruments i ON i.ticker = m.ticker
            LEFT JOIN prev ON prev.ticker = m.ticker
            WHERE m.calculation_date = $1
              AND i.is_active = TRUE
              {self._base_candidate_filters()}
              AND COALESCE(m.relative_volume, 0) >= $5
              AND COALESCE(m.rs_percentile, 0) >= $6
              AND (COALESCE(m.rs_trend, '') = 'improving'
                   OR COALESCE(m.rs_percentile, 0) - COALESCE(prev.rs_percentile, 0) >= $7)
            ORDER BY m.rs_percentile DESC, rs_5d_change DESC
            LIMIT 250
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_price,
            self.config.max_price,
            self.config.min_avg_dollar_volume,
            self.config.min_relative_volume,
            self.config.min_rs_percentile,
            self.config.min_rs_5d_gain,
        )
        return [dict(row) for row in rows]

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        close = _as_float(candles.iloc[-1]["close"])
        sma20 = _as_float(candidate.get("sma_20"))
        sma50 = _as_float(candidate.get("sma_50"))
        if close <= max(sma20, sma50):
            return None

        score = 50
        score += min(20, int(_as_float(candidate.get("rs_percentile")) / 5))
        score += min(12, max(0, int(_as_float(candidate.get("rs_5d_change")))))
        score += 8 if _as_float(candidate.get("price_change_20d")) > 0 else 0
        score += 7 if _as_float(candidate.get("macd_histogram")) > 0 else 0
        score += min(8, int(_as_float(candidate.get("relative_volume")) * 3))

        candidate["current_price"] = close
        factors = [
            f"RS percentile: {int(_as_float(candidate.get('rs_percentile')))}",
            f"RS 5D change: {int(_as_float(candidate.get('rs_5d_change')))}",
            "Price above SMA20/SMA50",
            f"20D return: {_as_float(candidate.get('price_change_20d')):.1f}%",
        ]
        return self._make_signal(candidate, SignalType.BUY, min(score, 100), "Fresh relative strength leadership", factors, calculation_date)


@dataclass
class PreMarketCatalystConfig(DataDrivenScannerConfig):
    max_signals_per_run: int = 15
    min_score_threshold: int = 75
    min_gap_pct: float = 3.0
    min_news_sentiment: float = 0.15
    min_news_count: int = 2


class PreMarketCatalystScanner(DataDrivenScanner):
    """Turns stored pre-market gap/news results into normal scanner signals."""

    def __init__(self, db_manager, config: PreMarketCatalystConfig = None, scorer=None):
        super().__init__(db_manager, config or PreMarketCatalystConfig(), scorer)

    @property
    def scanner_name(self) -> str:
        return "premarket_catalyst"

    @property
    def description(self) -> str:
        return "Pre-market gap and news catalyst setup from stored premarket signals"

    async def _get_candidates(self, calculation_date: date) -> List[Dict[str, Any]]:
        query = """
            SELECT
                p.symbol AS ticker,
                COALESCE(p.current_price, m.current_price) AS current_price,
                m.atr_14, m.atr_percent, m.relative_volume, m.rs_percentile,
                m.price_change_1d, m.sma_20, m.sma_50, m.macd_histogram,
                i.sector,
                p.gap_percent, p.signal_type AS premarket_signal_type,
                p.direction, p.confidence, p.news_count, p.news_sentiment_score,
                p.key_headlines
            FROM stock_premarket_signals p
            LEFT JOIN stock_screening_metrics m ON m.ticker = p.symbol
                AND m.calculation_date = (
                    SELECT MAX(calculation_date) FROM stock_screening_metrics
                    WHERE calculation_date <= $1
                )
            JOIN stock_instruments i ON i.ticker = p.symbol
            WHERE DATE(p.generated_at) = $1
              AND i.is_active = TRUE
              AND ABS(COALESCE(p.gap_percent, 0)) >= $2
              AND COALESCE(p.news_count, 0) >= $3
            ORDER BY COALESCE(p.confidence, 0) DESC, ABS(COALESCE(p.gap_percent, 0)) DESC
            LIMIT 50
        """
        try:
            rows = await self.db.fetch(
                query,
                calculation_date,
                self.config.min_gap_pct,
                self.config.min_news_count,
            )
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("premarket_catalyst skipped; stock_premarket_signals unavailable or incompatible: %s", exc)
            return []

    def _build_signal(self, candidate: Dict[str, Any], candles: pd.DataFrame, calculation_date: date) -> Optional[SignalSetup]:
        direction_raw = str(candidate.get("direction") or "").upper()
        signal_type = SignalType.SELL if direction_raw == "SELL" or _as_float(candidate.get("gap_percent")) < 0 else SignalType.BUY
        confidence = _as_float(candidate.get("confidence"), 0.5)
        gap_pct = abs(_as_float(candidate.get("gap_percent")))
        news_sentiment = _as_float(candidate.get("news_sentiment_score"))

        score = 50 + int(confidence * 25) + min(15, int(gap_pct * 2))
        score += 8 if abs(news_sentiment) >= self.config.min_news_sentiment else 0
        score += min(7, int(_as_float(candidate.get("news_count"))))

        factors = [
            f"Pre-market gap: {_as_float(candidate.get('gap_percent')):.1f}%",
            f"News count: {int(_as_float(candidate.get('news_count')))}",
            f"Sentiment score: {news_sentiment:.2f}",
            f"Confidence: {confidence:.0%}",
        ]
        return self._make_signal(candidate, signal_type, min(score, 100), "Pre-market catalyst gap setup", factors, calculation_date)
