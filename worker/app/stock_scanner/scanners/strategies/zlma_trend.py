"""
Zero-Lag MA Trend Scanner

Based on TradingView indicator "Zero-Lag MA Trend Levels [ChartPrime]"

Strategy Logic:
1. Calculate EMA(close, length)
2. Calculate correction factor: close + (close - EMA)
3. Calculate ZLMA = EMA(correction, length) - This is the Zero-Lag MA
4. Signals:
   - BUY: ZLMA crosses above EMA
   - SELL: ZLMA crosses under EMA
5. Trend Levels: Boxes drawn from signal price +/- ATR

Key Features:
- Zero-lag moving average reduces delay compared to standard EMA
- ATR-based trend levels provide support/resistance zones
- Good for daily timeframe swing trading
"""

import numpy as np
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..base_scanner import (
    BaseScanner, SignalSetup, ScannerConfig,
    SignalType, QualityTier
)
from ..scoring import SignalScorer

logger = logging.getLogger(__name__)


@dataclass
class ZLMATrendConfig(ScannerConfig):
    """Configuration specific to ZLMA Trend Scanner"""

    # ZLMA parameters
    zlma_length: int = 15  # EMA/ZLMA period
    atr_period: int = 14   # ATR period for trend levels

    # Signal quality
    min_crossover_strength: float = 0.1  # Min ZLMA-EMA separation as % of ATR
    min_atr_percent: float = 2.0  # Minimum ATR % for volatility

    # Volume requirements
    min_relative_volume: float = 0.8

    # Risk management
    atr_stop_multiplier: float = 1.0  # Stop at ATR level
    atr_tp_multiplier: float = 2.0    # TP at 2x ATR


class ZLMATrendScanner(BaseScanner):
    """
    Zero-Lag Moving Average Trend Scanner.

    Generates signals based on ZLMA/EMA crossovers with ATR-based levels.
    - BUY when ZLMA crosses above EMA
    - SELL when ZLMA crosses below EMA
    - Uses ATR for stop-loss and take-profit levels
    """

    def __init__(
        self,
        db_manager,
        config: ZLMATrendConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or ZLMATrendConfig(), scorer)
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "zlma_trend"

    @property
    def description(self) -> str:
        return "Zero-Lag MA crossover signals with ATR-based trend levels"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute ZLMA trend scan.

        Steps:
        1. Get qualified tickers from watchlist
        2. Calculate ZLMA and EMA for each
        3. Detect crossovers
        4. Score and create signals
        """
        logger.info(f"Starting {self.scanner_name} scan")

        if calculation_date is None:
            from datetime import timedelta
            calculation_date = (datetime.now() - timedelta(days=1)).date()

        # Ensure calculation_date is a date object
        if hasattr(calculation_date, 'date'):
            calculation_date = calculation_date.date()

        # Get qualified tickers
        tickers = await self._get_qualified_tickers(calculation_date)
        logger.info(f"Scanning {len(tickers)} qualified stocks")

        signals = []

        for ticker in tickers:
            signal = await self._scan_ticker(ticker, calculation_date)
            if signal:
                signals.append(signal)
                logger.info(f"  {signal.signal_type.value}: {ticker} @ {signal.entry_price}")

        # Sort by score
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        self.log_scan_summary(len(tickers), len(signals), high_quality)

        return signals

    async def _scan_ticker(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> Optional[SignalSetup]:
        """Scan a single ticker for ZLMA crossover signal."""

        # Get daily candles
        candles = await self._get_daily_candles(ticker, limit=250)

        if len(candles) < self.config.zlma_length * 2 + 10:
            return None

        # Convert to numpy arrays
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])

        # Calculate indicators
        ema = self._calculate_ema(closes, self.config.zlma_length)
        zlma = self._calculate_zlma(closes, self.config.zlma_length)
        atr = self._calculate_atr(highs, lows, closes, self.config.atr_period)

        if ema is None or zlma is None or atr is None:
            return None

        # Detect crossover
        signal_type = self._detect_crossover(zlma, ema)

        if signal_type is None:
            return None

        # Get current values
        current_close = closes[-1]
        current_zlma = zlma[-1]
        current_ema = ema[-1]
        current_atr = atr[-1]

        # Calculate crossover strength
        crossover_strength = abs(current_zlma - current_ema) / current_atr if current_atr > 0 else 0

        if crossover_strength < self.config.min_crossover_strength:
            return None

        # Get candidate data for scoring
        candidate = await self._get_candidate_data(ticker, calculation_date)
        if not candidate:
            candidate = {
                'ticker': ticker,
                'current_price': current_close,
                'atr_percent': (current_atr / current_close) * 100 if current_close > 0 else 0,
            }

        candidate['zlma_value'] = current_zlma
        candidate['ema_value'] = current_ema
        candidate['atr_value'] = current_atr
        candidate['crossover_strength'] = crossover_strength

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)

        # Score the signal
        score_components = self.scorer.score_bullish(candidate) if signal_type == SignalType.BUY else self.scorer.score_bearish(candidate)
        composite_score = score_components.composite_score

        # Boost score based on crossover strength
        strength_bonus = min(15, int(crossover_strength * 30))
        composite_score = min(100, composite_score + strength_bonus)

        # Calculate risk percent
        risk_pct = abs(float(entry) - float(stop)) / float(entry) * 100 if float(entry) > 0 else 0

        # Build description
        description = self._build_description(ticker, signal_type, current_zlma, current_ema, crossover_strength)

        # Create signal
        signal = SignalSetup(
            ticker=ticker,
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(round(self.config.atr_tp_multiplier, 2))),
            risk_percent=Decimal(str(round(risk_pct, 2))),
            composite_score=composite_score,
            trend_score=Decimal(str(round(score_components.weighted_trend, 2))),
            momentum_score=Decimal(str(round(score_components.weighted_momentum, 2))),
            volume_score=Decimal(str(round(score_components.weighted_volume, 2))),
            pattern_score=Decimal(str(round(score_components.weighted_pattern, 2))),
            confluence_score=Decimal(str(round(score_components.weighted_confluence, 2))),
            setup_description=description,
            confluence_factors=self._build_confluence_factors(candidate, signal_type),
            timeframe="daily",
            market_regime=self._determine_market_regime(candidate),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=candidate,
        )

        return signal

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """
        Calculate entry, stop, and take profit levels for ZLMA.

        For ZLMA:
        - Entry: Current price
        - Stop: 1x ATR from ZLMA (at trend level boundary)
        - TP1: 2x ATR from entry
        - TP2: 3x ATR from entry
        """
        current_price = Decimal(str(candidate.get('current_price', 0)))
        zlma = float(candidate.get('zlma_value', current_price))
        atr = float(candidate.get('atr_value', 0))

        entry = current_price
        atr_dec = Decimal(str(atr))

        if signal_type == SignalType.BUY:
            # Stop below ZLMA level
            stop = Decimal(str(zlma)) - atr_dec * Decimal(str(self.config.atr_stop_multiplier))
            tp1 = entry + atr_dec * Decimal(str(self.config.atr_tp_multiplier))
            tp2 = entry + atr_dec * Decimal('3.0')
        else:
            # Stop above ZLMA level
            stop = Decimal(str(zlma)) + atr_dec * Decimal(str(self.config.atr_stop_multiplier))
            tp1 = entry - atr_dec * Decimal(str(self.config.atr_tp_multiplier))
            tp2 = entry - atr_dec * Decimal('3.0')

        return entry, stop, tp1, tp2

    def _calculate_ema(self, prices: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None

        alpha = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema

    def _calculate_zlma(self, prices: np.ndarray, period: int) -> Optional[np.ndarray]:
        """
        Calculate Zero-Lag Moving Average.

        Formula:
        1. EMA = EMA(close, period)
        2. Correction = close + (close - EMA)
        3. ZLMA = EMA(correction, period)
        """
        if len(prices) < period * 2:
            return None

        ema = self._calculate_ema(prices, period)
        correction = prices + (prices - ema)
        zlma = self._calculate_ema(correction, period)

        return zlma

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> Optional[np.ndarray]:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return None

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

    def _detect_crossover(self, zlma: np.ndarray, ema: np.ndarray) -> Optional[SignalType]:
        """Detect crossover on the last candle."""
        if len(zlma) < 2 or len(ema) < 2:
            return None

        zlma_curr, zlma_prev = zlma[-1], zlma[-2]
        ema_curr, ema_prev = ema[-1], ema[-2]

        # Bullish crossover
        if zlma_prev <= ema_prev and zlma_curr > ema_curr:
            return SignalType.BUY

        # Bearish crossover
        if zlma_prev >= ema_prev and zlma_curr < ema_curr:
            return SignalType.SELL

        return None

    async def _get_daily_candles(self, ticker: str, limit: int = 250) -> List[Dict[str, Any]]:
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

    async def _get_qualified_tickers(self, calculation_date: datetime) -> List[str]:
        """Get tickers that meet minimum criteria."""
        query = """
            SELECT w.ticker
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            WHERE w.calculation_date = $1
              AND w.tier <= $2
              AND COALESCE(m.atr_percent, 0) >= $3
              AND COALESCE(m.relative_volume, 0) >= $4
            ORDER BY w.rank_overall
        """
        rows = await self.db.fetch(
            query,
            calculation_date,
            self.config.max_tier,
            self.config.min_atr_percent,
            self.config.min_relative_volume
        )
        return [r['ticker'] for r in rows]

    async def _get_candidate_data(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get full candidate data from watchlist and metrics."""
        query = """
            SELECT
                w.*,
                m.rsi_14, m.rsi_signal,
                m.macd_histogram, m.macd_cross_signal,
                m.trend_strength, m.relative_volume,
                m.atr_percent, m.current_price
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            WHERE w.ticker = $1 AND w.calculation_date = $2
        """
        row = await self.db.fetchrow(query, ticker, calculation_date)
        return dict(row) if row else None

    def _build_description(
        self,
        ticker: str,
        signal_type: SignalType,
        zlma: float,
        ema: float,
        strength: float
    ) -> str:
        """Build human-readable setup description."""
        direction = "bullish" if signal_type == SignalType.BUY else "bearish"
        cross_dir = "above" if signal_type == SignalType.BUY else "below"

        return (
            f"{ticker} ZLMA {direction} crossover. "
            f"ZLMA crossed {cross_dir} EMA with {strength:.1%} ATR separation. "
            f"Zero-lag MA trend reversal signal."
        )

    def _build_confluence_factors(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> List[str]:
        """Build list of confluence factors."""
        factors = ["ZLMA/EMA Crossover"]

        # Trend alignment
        trend = candidate.get('trend_strength', '')
        if signal_type == SignalType.BUY and 'up' in str(trend).lower():
            factors.append("Trend Aligned")
        elif signal_type == SignalType.SELL and 'down' in str(trend).lower():
            factors.append("Trend Aligned")

        # RSI
        rsi = float(candidate.get('rsi_14') or 50)
        if signal_type == SignalType.BUY and 40 <= rsi <= 60:
            factors.append("RSI Neutral")
        elif signal_type == SignalType.SELL and 40 <= rsi <= 60:
            factors.append("RSI Neutral")

        # Volume
        rel_vol = float(candidate.get('relative_volume') or 0)
        if rel_vol >= 1.5:
            factors.append("High Volume")

        # Crossover strength
        strength = float(candidate.get('crossover_strength') or 0)
        if strength >= 0.3:
            factors.append("Strong Crossover")

        return factors

    def _determine_market_regime(self, candidate: Dict[str, Any]) -> str:
        """Determine market regime."""
        trend = candidate.get('trend_strength', '')
        atr_pct = float(candidate.get('atr_percent') or 0)

        if 'strong' in str(trend).lower():
            regime = "Strong Trend"
        elif 'up' in str(trend).lower() or 'down' in str(trend).lower():
            regime = "Trending"
        else:
            regime = "Ranging"

        if atr_pct > 5:
            regime += " (High Vol)"
        elif atr_pct < 2:
            regime += " (Low Vol)"

        return regime

    def log_scan_summary(
        self,
        candidates_count: int,
        signals_count: int,
        high_quality_count: int
    ) -> None:
        """Log scan summary."""
        logger.info(f"\n{self.scanner_name} scan complete:")
        logger.info(f"  Stocks scanned: {candidates_count}")
        logger.info(f"  Signals found: {signals_count}")
        logger.info(f"  High quality (A/A+): {high_quality_count}")
