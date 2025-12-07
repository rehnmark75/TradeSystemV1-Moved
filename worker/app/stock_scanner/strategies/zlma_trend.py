"""
Zero-Lag MA Trend Strategy

Based on TradingView indicator "Zero-Lag MA Trend Levels [ChartPrime]"

Strategy Logic:
1. Calculate EMA(close, length)
2. Calculate correction factor: close + (close - EMA)
3. Calculate ZLMA = EMA(correction, length) - This is the Zero-Lag MA
4. Signals:
   - BUY: ZLMA crosses above EMA
   - SELL: ZLMA crosses under EMA
5. Trend Levels: Boxes drawn from signal price +/- ATR(200)
6. Confirmation signals when price crosses box boundaries

Key Features:
- Zero-lag moving average reduces delay compared to standard EMA
- ATR-based trend levels provide support/resistance zones
- Good for daily timeframe swing trading
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ZLMAConfig:
    """Configuration for Zero-Lag MA strategy."""
    length: int = 15             # EMA/ZLMA period
    atr_period: int = 14         # ATR period for trend levels (standard 14-day)
    min_atr_percent: float = 2.0 # Minimum ATR % for signal
    min_dollar_volume: float = 10_000_000  # Minimum dollar volume


@dataclass
class ZLMASignal:
    """A signal from the ZLMA strategy."""
    ticker: str
    signal_type: str  # 'BUY' or 'SELL'
    timestamp: datetime
    entry_price: float
    zlma_value: float
    ema_value: float
    atr_value: float
    level_top: float
    level_bottom: float
    stop_loss: float
    take_profit: float
    confidence: float


class ZeroLagMATrendStrategy:
    """
    Zero-Lag Moving Average Trend Strategy.

    Generates signals based on ZLMA/EMA crossovers with ATR-based levels.

    For daily timeframe:
    - BUY when ZLMA crosses above EMA
    - SELL when ZLMA crosses below EMA
    - Stop loss at opposite trend level
    - Take profit at 2x ATR from entry
    """

    def __init__(self, db_manager, config: ZLMAConfig = None):
        self.db = db_manager
        self.config = config or ZLMAConfig()

    async def scan_all_stocks(
        self,
        calculation_date: datetime = None,
        tier_filter: int = None
    ) -> List[ZLMASignal]:
        """
        Scan all stocks (or filtered by tier) for ZLMA signals.

        Args:
            calculation_date: Date to scan (defaults to yesterday)
            tier_filter: Only scan stocks in this tier (1-4)

        Returns:
            List of ZLMASignal objects
        """
        logger.info("=" * 60)
        logger.info("ZLMA STRATEGY SCAN")
        logger.info("=" * 60)

        start_time = datetime.now()

        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        # Get tickers to scan
        if tier_filter:
            tickers = await self._get_tier_tickers(tier_filter, calculation_date)
            logger.info(f"Scanning Tier {tier_filter}: {len(tickers)} stocks")
        else:
            tickers = await self._get_qualified_tickers(calculation_date)
            logger.info(f"Scanning qualified stocks: {len(tickers)}")

        signals = []

        for ticker in tickers:
            signal = await self.scan_ticker(ticker, calculation_date)
            if signal:
                signals.append(signal)
                logger.info(f"  {signal.signal_type}: {ticker} @ {signal.entry_price:.2f}")

        # Save signals to database
        for signal in signals:
            await self._save_signal(signal)

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"\nScan complete:")
        logger.info(f"  Stocks scanned: {len(tickers)}")
        logger.info(f"  Signals found: {len(signals)}")
        logger.info(f"  BUY: {sum(1 for s in signals if s.signal_type == 'BUY')}")
        logger.info(f"  SELL: {sum(1 for s in signals if s.signal_type == 'SELL')}")
        logger.info(f"  Duration: {elapsed:.2f}s")

        return signals

    async def scan_ticker(
        self,
        ticker: str,
        calculation_date: datetime = None
    ) -> Optional[ZLMASignal]:
        """
        Scan a single ticker for ZLMA signal.

        Returns signal if crossover occurred on calculation_date.
        """
        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        # Fetch daily candles (need enough for ATR calculation)
        candles = await self._get_daily_candles(ticker, limit=250)

        if len(candles) < self.config.atr_period + self.config.length + 5:
            return None

        # Convert to numpy arrays
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        timestamps = [c['timestamp'] for c in candles]

        # Check if latest candle is for calculation_date
        latest_date = timestamps[-1].date() if hasattr(timestamps[-1], 'date') else timestamps[-1]
        if latest_date != calculation_date:
            return None  # No data for this date

        # Calculate indicators
        ema = self._calculate_ema(closes, self.config.length)
        zlma = self._calculate_zlma(closes, self.config.length)
        atr = self._calculate_atr(highs, lows, closes, self.config.atr_period)

        if ema is None or zlma is None or atr is None:
            return None

        # Check for crossover on the last candle
        signal_type = self._detect_crossover(zlma, ema)

        if signal_type is None:
            return None

        # Get values at signal
        current_close = closes[-1]
        current_zlma = zlma[-1]
        current_ema = ema[-1]
        current_atr = atr[-1]

        # Calculate trend levels (box bounds)
        if signal_type == 'BUY':
            level_top = current_zlma
            level_bottom = current_zlma - current_atr
            stop_loss = level_bottom
            take_profit = current_close + (2 * current_atr)
        else:  # SELL
            level_top = current_zlma + current_atr
            level_bottom = current_zlma
            stop_loss = level_top
            take_profit = current_close - (2 * current_atr)

        # Calculate confidence based on crossover strength
        crossover_strength = abs(current_zlma - current_ema) / current_atr
        confidence = min(100, max(30, 50 + (crossover_strength * 100)))

        return ZLMASignal(
            ticker=ticker,
            signal_type=signal_type,
            timestamp=timestamps[-1],
            entry_price=current_close,
            zlma_value=round(current_zlma, 4),
            ema_value=round(current_ema, 4),
            atr_value=round(current_atr, 4),
            level_top=round(level_top, 4),
            level_bottom=round(level_bottom, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            confidence=round(confidence, 2)
        )

    def _calculate_ema(
        self,
        prices: np.ndarray,
        period: int
    ) -> Optional[np.ndarray]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None

        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema

    def _calculate_zlma(
        self,
        prices: np.ndarray,
        period: int
    ) -> Optional[np.ndarray]:
        """
        Calculate Zero-Lag Moving Average.

        Formula:
        1. EMA = EMA(close, period)
        2. Correction = close + (close - EMA)
        3. ZLMA = EMA(correction, period)
        """
        if len(prices) < period * 2:
            return None

        # Step 1: Calculate EMA
        ema = self._calculate_ema(prices, period)

        # Step 2: Calculate correction factor
        correction = prices + (prices - ema)

        # Step 3: Calculate ZLMA (EMA of correction)
        zlma = self._calculate_ema(correction, period)

        return zlma

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 200
    ) -> Optional[np.ndarray]:
        """Calculate Average True Range using Wilder's smoothing."""
        if len(closes) < period + 1:
            return None

        # True Range
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        # ATR using Wilder's smoothing
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    def _detect_crossover(
        self,
        zlma: np.ndarray,
        ema: np.ndarray
    ) -> Optional[str]:
        """
        Detect crossover on the last candle.

        Returns:
            'BUY' if ZLMA crossed above EMA
            'SELL' if ZLMA crossed below EMA
            None if no crossover
        """
        if len(zlma) < 2 or len(ema) < 2:
            return None

        # Current and previous values
        zlma_curr = zlma[-1]
        zlma_prev = zlma[-2]
        ema_curr = ema[-1]
        ema_prev = ema[-2]

        # Bullish crossover: ZLMA crosses above EMA
        if zlma_prev <= ema_prev and zlma_curr > ema_curr:
            return 'BUY'

        # Bearish crossover: ZLMA crosses below EMA
        if zlma_prev >= ema_prev and zlma_curr < ema_curr:
            return 'SELL'

        return None

    async def _get_daily_candles(
        self,
        ticker: str,
        limit: int = 250
    ) -> List[Dict[str, Any]]:
        """Get daily candles for a ticker."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized
            WHERE ticker = $1 AND timeframe = '1d'
            ORDER BY timestamp ASC
            LIMIT $2
        """
        rows = await self.db.fetch(query, ticker, limit)
        return [dict(r) for r in rows]

    async def _get_qualified_tickers(
        self,
        calculation_date: datetime
    ) -> List[str]:
        """Get tickers that meet minimum criteria."""
        query = """
            SELECT ticker
            FROM stock_screening_metrics
            WHERE calculation_date = $1
              AND avg_dollar_volume >= $2
              AND atr_percent >= $3
              AND data_quality = 'good'
            ORDER BY avg_dollar_volume DESC
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.min_dollar_volume,
            self.config.min_atr_percent
        )
        return [r['ticker'] for r in rows]

    async def _get_tier_tickers(
        self,
        tier: int,
        calculation_date: datetime
    ) -> List[str]:
        """Get tickers from a specific watchlist tier."""
        query = """
            SELECT ticker
            FROM stock_watchlist
            WHERE calculation_date = $1 AND tier = $2
            ORDER BY rank_in_tier
        """
        rows = await self.db.fetch(query, calculation_date, tier)
        return [r['ticker'] for r in rows]

    async def _save_signal(self, signal: ZLMASignal) -> None:
        """Save signal to database."""
        query = """
            INSERT INTO stock_zlma_signals (
                ticker, signal_timestamp, signal_type,
                zlma_value, ema_value, atr_value,
                level_top, level_bottom,
                entry_price, stop_loss, take_profit,
                confidence
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            )
            ON CONFLICT (ticker, signal_timestamp, signal_type)
            DO UPDATE SET
                zlma_value = EXCLUDED.zlma_value,
                ema_value = EXCLUDED.ema_value,
                atr_value = EXCLUDED.atr_value,
                level_top = EXCLUDED.level_top,
                level_bottom = EXCLUDED.level_bottom,
                entry_price = EXCLUDED.entry_price,
                stop_loss = EXCLUDED.stop_loss,
                take_profit = EXCLUDED.take_profit,
                confidence = EXCLUDED.confidence,
                created_at = NOW()
        """

        await self.db.execute(
            query,
            signal.ticker,
            signal.timestamp,
            signal.signal_type,
            signal.zlma_value,
            signal.ema_value,
            signal.atr_value,
            signal.level_top,
            signal.level_bottom,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.confidence
        )

    async def get_recent_signals(
        self,
        days: int = 7,
        signal_type: str = None
    ) -> List[Dict[str, Any]]:
        """Get recent signals from the database."""
        query = """
            SELECT
                s.*,
                i.name as stock_name,
                m.trend_strength,
                m.relative_volume
            FROM stock_zlma_signals s
            JOIN stock_instruments i ON s.ticker = i.ticker
            LEFT JOIN stock_screening_metrics m
                ON s.ticker = m.ticker
                AND m.calculation_date = DATE(s.signal_timestamp)
            WHERE s.signal_timestamp > NOW() - INTERVAL '%s days'
        """ % days

        if signal_type:
            query += f" AND s.signal_type = '{signal_type}'"

        query += " ORDER BY s.signal_timestamp DESC"

        rows = await self.db.fetch(query)
        return [dict(r) for r in rows]
