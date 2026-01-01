"""
Pocket Pivot Scanner

Identifies Pocket Pivot buy signals - institutional accumulation patterns.

Pocket Pivot Criteria (O'Neil methodology):
1. Volume > max(all down-day volumes in last 10 days)
2. Price within 10% of 10-day high (near highs, not extended)
3. Price > EMA 50 > EMA 200 (uptrending structure)
4. RS percentile > 70 (relative strength leader)
5. Up day (close > open)
6. Not gapping up excessively (gap < 3%)

This scanner finds stocks showing institutional buying interest
within a proper trend structure.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Any, Optional

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ...core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)


@dataclass
class PocketPivotConfig(ScannerConfig):
    """Configuration for Pocket Pivot Scanner"""

    # Pocket Pivot parameters
    lookback_days: int = 10          # Days to check for down-volume max
    max_distance_from_high: float = 10.0  # Max % below 10-day high
    min_rs_percentile: int = 70      # Minimum RS percentile
    max_gap_pct: float = 3.0         # Max gap up to avoid chasing

    # Risk parameters
    stop_loss_pct: float = 7.0       # Stop loss percentage
    take_profit_pct: float = 20.0    # Take profit (3:1 R:R approx)

    # Scanning limits
    max_signals_per_run: int = 30
    min_score_threshold: int = 65


class PocketPivotScanner(BaseScanner):
    """
    Pocket Pivot Scanner - finds institutional accumulation patterns.

    This scanner identifies stocks where:
    - Volume exceeds the maximum down-day volume of the past 10 days
    - Price is near (within 10%) of recent highs
    - Stock is in a proper uptrend (EMA alignment)
    - Stock shows relative strength vs market

    These patterns indicate institutional accumulation within a base.
    """

    def __init__(
        self,
        db_manager,
        config: PocketPivotConfig = None,
        scorer=None
    ):
        super().__init__(db_manager, config or PocketPivotConfig(), scorer)
        self.data_provider = BacktestDataProvider(db_manager)

    @property
    def scanner_name(self) -> str:
        return "pocket_pivot"

    @property
    def description(self) -> str:
        return "Pocket Pivot - institutional accumulation within uptrend"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute Pocket Pivot scan.

        Steps:
        1. Get all active tickers with RS data
        2. Filter by RS percentile > 70
        3. For each candidate, fetch candle data
        4. Check Pocket Pivot criteria
        5. Generate signals for matches
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            calculation_date = datetime.now().date()
        elif isinstance(calculation_date, datetime):
            calculation_date = calculation_date.date()

        logger.info(f"Scan date: {calculation_date}")

        # Get candidates with RS data
        candidates = await self._get_rs_leaders()
        logger.info(f"Found {len(candidates)} RS leaders (>= {self.config.min_rs_percentile} percentile)")

        signals = []
        scanned_count = 0

        for ticker_data in candidates:
            ticker = ticker_data.get('ticker')
            if not ticker:
                continue

            scanned_count += 1
            if scanned_count > 200:  # Limit processing
                break

            try:
                signal = await self._check_pocket_pivot(ticker, ticker_data, calculation_date)
                if signal:
                    signals.append(signal)
                    logger.info(f"Pocket Pivot detected: {ticker}")

                    if len(signals) >= self.config.max_signals_per_run:
                        break

            except Exception as e:
                logger.warning(f"Error checking {ticker}: {e}")
                continue

        # Sort by score
        signals.sort(key=lambda s: s.composite_score, reverse=True)

        logger.info(f"Pocket Pivot scan complete: {len(signals)} signals from {scanned_count} candidates")
        return signals

    async def _get_rs_leaders(self) -> List[Dict[str, Any]]:
        """Get stocks with RS percentile >= threshold."""
        query = """
            SELECT
                m.ticker,
                m.rs_percentile,
                m.rs_trend,
                m.current_price,
                m.ema_50,
                m.ema_200,
                m.atr_percent,
                m.avg_daily_volume,
                m.trend_strength
            FROM stock_screening_metrics m
            WHERE m.calculation_date = (
                SELECT MAX(calculation_date) FROM stock_screening_metrics
            )
            AND m.rs_percentile >= %s
            AND m.current_price > 5  -- Min price filter
            AND m.avg_daily_volume > 500000  -- Min liquidity
            AND m.ema_50 IS NOT NULL
            AND m.ema_200 IS NOT NULL
            ORDER BY m.rs_percentile DESC
        """
        return await self.db_manager.fetch_all(query, (self.config.min_rs_percentile,))

    async def _check_pocket_pivot(
        self,
        ticker: str,
        ticker_data: Dict[str, Any],
        calculation_date: date
    ) -> Optional[SignalSetup]:
        """Check if ticker shows a valid Pocket Pivot pattern."""

        # Get candle data for analysis
        candles = await self.data_provider.get_candles(
            ticker,
            start_date=calculation_date - timedelta(days=30),
            end_date=calculation_date
        )

        if candles is None or len(candles) < 15:
            return None

        # Get the latest candle
        today = candles.iloc[-1]
        today_close = float(today['close'])
        today_open = float(today['open'])
        today_volume = float(today['volume'])
        today_high = float(today['high'])
        today_low = float(today['low'])

        # Check if it's an up day
        if today_close <= today_open:
            return None

        # Get EMA values from metrics
        ema_50 = float(ticker_data.get('ema_50') or 0)
        ema_200 = float(ticker_data.get('ema_200') or 0)

        # Check trend structure: Price > EMA 50 > EMA 200
        if not (today_close > ema_50 > ema_200):
            return None

        # Check gap - don't chase big gaps
        prev_close = float(candles.iloc[-2]['close'])
        gap_pct = ((today_open - prev_close) / prev_close) * 100
        if gap_pct > self.config.max_gap_pct:
            return None

        # Get 10-day high
        lookback = min(self.config.lookback_days, len(candles) - 1)
        recent_high = candles['high'].iloc[-lookback:].max()

        # Check if price is within 10% of high
        distance_from_high = ((recent_high - today_close) / recent_high) * 100
        if distance_from_high > self.config.max_distance_from_high:
            return None

        # POCKET PIVOT CHECK: Volume > max down-day volume
        # Get all down days in lookback period
        down_days = []
        for i in range(-lookback, -1):
            if candles.iloc[i]['close'] < candles.iloc[i]['open']:
                down_days.append(float(candles.iloc[i]['volume']))

        if not down_days:
            # No down days - use average volume as fallback
            max_down_volume = candles['volume'].iloc[-lookback:-1].mean()
        else:
            max_down_volume = max(down_days)

        # Check if today's volume exceeds max down-day volume
        if today_volume <= max_down_volume:
            return None

        # POCKET PIVOT CONFIRMED - Calculate signal parameters
        rs_percentile = ticker_data.get('rs_percentile', 50)
        atr_percent = float(ticker_data.get('atr_percent') or 3.0)

        # Entry at close
        entry_price = Decimal(str(today_close))

        # Stop loss below today's low or 7%, whichever is tighter
        stop_from_low = today_low * 0.98
        stop_from_pct = today_close * (1 - self.config.stop_loss_pct / 100)
        stop_loss = Decimal(str(max(stop_from_low, stop_from_pct)))

        # Take profit at 20%
        take_profit = entry_price * Decimal("1.20")

        # Calculate risk/reward
        risk = float(entry_price - stop_loss)
        reward = float(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 1.0

        # Calculate quality score
        score = self._calculate_score(
            rs_percentile=rs_percentile,
            volume_ratio=today_volume / max_down_volume,
            distance_from_high=distance_from_high,
            atr_percent=atr_percent
        )

        if score < self.config.min_score_threshold:
            return None

        # Determine quality tier
        if score >= 85:
            tier = QualityTier.A_PLUS
        elif score >= 75:
            tier = QualityTier.A
        elif score >= 65:
            tier = QualityTier.B
        else:
            tier = QualityTier.C

        return SignalSetup(
            ticker=ticker,
            signal_type=SignalType.BUY,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit,
            take_profit_2=entry_price * Decimal("1.30"),  # Extended target
            composite_score=score,
            quality_tier=tier,
            risk_reward_ratio=Decimal(str(round(rr_ratio, 2))),
            risk_percent=Decimal(str(self.config.stop_loss_pct)),
            setup_description=f"Pocket Pivot - Volume {today_volume/max_down_volume:.1f}x max down-vol, RS {rs_percentile}",
            confluence_factors=[
                f"RS Percentile: {rs_percentile}",
                f"Volume: {today_volume/max_down_volume:.1f}x max down-day volume",
                f"Distance from high: {distance_from_high:.1f}%",
                f"Trend: Price > EMA50 > EMA200",
                f"ATR: {atr_percent:.1f}%"
            ],
            signal_timestamp=datetime.combine(calculation_date, datetime.min.time()),
            scanner_name=self.scanner_name
        )

    def _calculate_score(
        self,
        rs_percentile: int,
        volume_ratio: float,
        distance_from_high: float,
        atr_percent: float
    ) -> int:
        """Calculate quality score for Pocket Pivot signal."""
        score = 50  # Base score

        # RS bonus (max 25 points)
        if rs_percentile >= 90:
            score += 25
        elif rs_percentile >= 80:
            score += 20
        elif rs_percentile >= 70:
            score += 15
        else:
            score += 10

        # Volume ratio bonus (max 20 points)
        if volume_ratio >= 2.0:
            score += 20
        elif volume_ratio >= 1.5:
            score += 15
        elif volume_ratio >= 1.2:
            score += 10
        else:
            score += 5

        # Distance from high bonus (closer is better, max 15 points)
        if distance_from_high <= 2:
            score += 15
        elif distance_from_high <= 5:
            score += 12
        elif distance_from_high <= 8:
            score += 8
        else:
            score += 5

        # ATR penalty for very volatile stocks
        if atr_percent > 6:
            score -= 5
        elif atr_percent > 5:
            score -= 3

        return min(max(score, 0), 100)


# Import for timedelta
from datetime import timedelta
