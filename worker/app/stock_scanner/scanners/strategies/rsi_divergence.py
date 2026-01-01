"""
RSI Divergence Scanner

Adapted from AlphaSuite's bullish_rsi_divergence.py and bearish_rsi_divergence.py patterns.
Identifies potential reversals by detecting price/RSI divergence.

Entry Criteria:
- Bullish: Price makes lower low, RSI makes higher low (momentum improving)
- Bearish: Price makes higher high, RSI makes lower high (momentum weakening)
- Optional trend filter (below 200 SMA for bullish, above for bearish)

Stop Logic:
- Below/above the divergence swing point
- ATR-based safety stop

Target:
- TP1: Prior swing high/low or 200 SMA
- TP2: 3R multiple

Best For:
- Early reversal detection
- Momentum divergence confirmation
- Counter-trend entries with defined risk
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
from ..exclusion_filters import ExclusionFilterEngine, get_mean_reversion_filter

logger = logging.getLogger(__name__)


@dataclass
class RSIDivergenceConfig(ScannerConfig):
    """Configuration for RSI Divergence Scanner"""

    # ==========================================================================
    # RSI Parameters
    # ==========================================================================
    rsi_period: int = 14  # RSI calculation period

    # ==========================================================================
    # Divergence Detection Parameters
    # ==========================================================================
    divergence_lookback: int = 30  # Window to find prior swing
    setup_window: int = 5  # Recent divergence only (last N days)
    min_divergence_bars: int = 5  # Minimum bars between swings
    max_divergence_bars: int = 25  # Maximum bars between swings

    # ==========================================================================
    # Trend Filter
    # ==========================================================================
    require_trend_filter: bool = True  # Require trend context
    # Bullish: below 200 SMA (downtrend reversal)
    # Bearish: above 200 SMA (uptrend reversal)

    # ==========================================================================
    # RSI Thresholds
    # ==========================================================================
    bullish_rsi_max: float = 50.0  # RSI should be relatively low for bullish
    bearish_rsi_min: float = 50.0  # RSI should be relatively high for bearish

    # ==========================================================================
    # Volume Requirements
    # ==========================================================================
    min_relative_volume: float = 0.8

    # ==========================================================================
    # Risk Management
    # ==========================================================================
    atr_stop_multiplier: float = 1.5
    max_stop_loss_pct: float = 8.0

    # Default R:R targets
    tp1_rr_ratio: float = 2.0
    tp2_rr_ratio: float = 3.0

    # ==========================================================================
    # Fundamental Filters
    # ==========================================================================
    max_pe_ratio: float = 60.0
    min_profit_margin: float = 0.0
    max_debt_to_equity: float = 3.0
    days_to_earnings_min: int = 7


class RSIDivergenceScanner(BaseScanner):
    """
    Scans for RSI divergence patterns indicating potential reversals.

    Philosophy:
    - Divergence between price and momentum signals weakening trend
    - Bullish divergence: price lower low but RSI higher low (buyers stepping in)
    - Bearish divergence: price higher high but RSI lower high (sellers emerging)
    - Best when combined with trend context for reversal plays
    """

    def __init__(
        self,
        db_manager,
        config: RSIDivergenceConfig = None,
        scorer: SignalScorer = None
    ):
        super().__init__(db_manager, config or RSIDivergenceConfig(), scorer)
        self.exclusion_filter = get_mean_reversion_filter()
        if scorer is None:
            self.scorer = SignalScorer()

    @property
    def scanner_name(self) -> str:
        return "rsi_divergence"

    @property
    def description(self) -> str:
        return "RSI divergence for reversal detection with trend context"

    async def scan(self, calculation_date: datetime = None) -> List[SignalSetup]:
        """
        Execute RSI divergence scan.

        Steps:
        1. Get qualified tickers from watchlist
        2. Fetch daily candles and calculate RSI
        3. Detect bullish and bearish divergences
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
        logger.info(f"Scanning {len(tickers)} qualified stocks for RSI divergence")

        signals = []

        for ticker in tickers:
            # Check for bullish divergence
            bullish_signal = await self._scan_ticker_bullish(ticker, calculation_date)
            if bullish_signal:
                signals.append(bullish_signal)
                logger.info(f"  BULLISH DIV: {ticker} @ {bullish_signal.entry_price}")

            # Check for bearish divergence
            bearish_signal = await self._scan_ticker_bearish(ticker, calculation_date)
            if bearish_signal:
                signals.append(bearish_signal)
                logger.info(f"  BEARISH DIV: {ticker} @ {bearish_signal.entry_price}")

        # Sort by score
        signals.sort(key=lambda x: x.composite_score, reverse=True)

        # Apply limit
        signals = signals[:self.config.max_signals_per_run]

        # Log summary
        high_quality = sum(1 for s in signals if s.is_high_quality)
        bullish_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        bearish_count = len(signals) - bullish_count
        logger.info(f"\n{self.scanner_name} scan complete:")
        logger.info(f"  Stocks scanned: {len(tickers)}")
        logger.info(f"  Bullish divergences: {bullish_count}")
        logger.info(f"  Bearish divergences: {bearish_count}")
        logger.info(f"  High quality (A/A+): {high_quality}")

        return signals

    async def _scan_ticker_bullish(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> Optional[SignalSetup]:
        """Scan for bullish RSI divergence (lower low price, higher low RSI)."""

        candles = await self._get_daily_candles(ticker, limit=100)

        if len(candles) < self.config.divergence_lookback + self.config.rsi_period + 10:
            return None

        # Convert to numpy arrays
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])

        # Calculate RSI
        rsi_values = self._calculate_rsi(closes, self.config.rsi_period)

        if rsi_values is None:
            return None

        # Detect bullish divergence
        divergence = self._detect_bullish_divergence(closes, lows, rsi_values)

        if divergence is None:
            return None

        current_idx, prior_idx, rsi_current, rsi_prior = divergence

        # Check RSI threshold
        if rsi_current > self.config.bullish_rsi_max:
            return None

        # Get candidate data
        candidate = await self._get_candidate_data(ticker, calculation_date)
        if not candidate:
            candidate = {
                'ticker': ticker,
                'current_price': closes[-1],
                'atr_percent': self._calculate_atr_percent(highs, lows, closes),
            }

        # Check trend filter (bullish divergence works best in downtrend)
        if self.config.require_trend_filter:
            sma200_signal = candidate.get('sma200_signal', '')
            if sma200_signal not in ['below', 'crossed_below']:
                # Not in downtrend - skip
                return None

        # Add divergence data
        candidate['divergence_type'] = 'bullish'
        candidate['rsi_current'] = rsi_current
        candidate['rsi_prior'] = rsi_prior
        candidate['price_current_low'] = lows[current_idx]
        candidate['price_prior_low'] = lows[prior_idx]
        candidate['divergence_bars'] = current_idx - prior_idx
        candidate['candle_timestamp'] = candles[-1].get('timestamp') if candles else None

        return self._create_signal(candidate, SignalType.BUY)

    async def _scan_ticker_bearish(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> Optional[SignalSetup]:
        """Scan for bearish RSI divergence (higher high price, lower high RSI)."""

        candles = await self._get_daily_candles(ticker, limit=100)

        if len(candles) < self.config.divergence_lookback + self.config.rsi_period + 10:
            return None

        # Convert to numpy arrays
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])

        # Calculate RSI
        rsi_values = self._calculate_rsi(closes, self.config.rsi_period)

        if rsi_values is None:
            return None

        # Detect bearish divergence
        divergence = self._detect_bearish_divergence(closes, highs, rsi_values)

        if divergence is None:
            return None

        current_idx, prior_idx, rsi_current, rsi_prior = divergence

        # Check RSI threshold
        if rsi_current < self.config.bearish_rsi_min:
            return None

        # Get candidate data
        candidate = await self._get_candidate_data(ticker, calculation_date)
        if not candidate:
            candidate = {
                'ticker': ticker,
                'current_price': closes[-1],
                'atr_percent': self._calculate_atr_percent(highs, lows, closes),
            }

        # Check trend filter (bearish divergence works best in uptrend)
        if self.config.require_trend_filter:
            sma200_signal = candidate.get('sma200_signal', '')
            if sma200_signal not in ['above', 'crossed_above']:
                # Not in uptrend - skip
                return None

        # Add divergence data
        candidate['divergence_type'] = 'bearish'
        candidate['rsi_current'] = rsi_current
        candidate['rsi_prior'] = rsi_prior
        candidate['price_current_high'] = highs[current_idx]
        candidate['price_prior_high'] = highs[prior_idx]
        candidate['divergence_bars'] = current_idx - prior_idx
        candidate['candle_timestamp'] = candles[-1].get('timestamp') if candles else None

        return self._create_signal(candidate, SignalType.SELL)

    def _detect_bullish_divergence(
        self,
        closes: np.ndarray,
        lows: np.ndarray,
        rsi_values: np.ndarray
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Detect bullish RSI divergence.

        Bullish divergence: Price makes lower low, RSI makes higher low.

        Returns:
            Tuple of (current_idx, prior_idx, rsi_current, rsi_prior) if found
        """
        lookback = self.config.divergence_lookback
        setup_window = self.config.setup_window
        min_bars = self.config.min_divergence_bars
        max_bars = self.config.max_divergence_bars

        # Find swing lows in the data
        swing_lows = self._find_swing_lows(lows, window=3)

        if len(swing_lows) < 2:
            return None

        # Check recent swing lows for divergence
        for i in range(len(swing_lows) - 1, 0, -1):
            current_idx = swing_lows[i]
            prior_idx = swing_lows[i - 1]

            # Must be within setup window
            if len(lows) - current_idx > setup_window:
                continue

            # Check bar distance constraints
            bar_diff = current_idx - prior_idx
            if bar_diff < min_bars or bar_diff > max_bars:
                continue

            # Check for lower low in price
            if lows[current_idx] >= lows[prior_idx]:
                continue  # Not a lower low

            # Check for higher low in RSI
            if rsi_values[current_idx] <= rsi_values[prior_idx]:
                continue  # Not a higher RSI low

            # Bullish divergence found!
            return (
                current_idx,
                prior_idx,
                rsi_values[current_idx],
                rsi_values[prior_idx]
            )

        return None

    def _detect_bearish_divergence(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        rsi_values: np.ndarray
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Detect bearish RSI divergence.

        Bearish divergence: Price makes higher high, RSI makes lower high.

        Returns:
            Tuple of (current_idx, prior_idx, rsi_current, rsi_prior) if found
        """
        lookback = self.config.divergence_lookback
        setup_window = self.config.setup_window
        min_bars = self.config.min_divergence_bars
        max_bars = self.config.max_divergence_bars

        # Find swing highs in the data
        swing_highs = self._find_swing_highs(highs, window=3)

        if len(swing_highs) < 2:
            return None

        # Check recent swing highs for divergence
        for i in range(len(swing_highs) - 1, 0, -1):
            current_idx = swing_highs[i]
            prior_idx = swing_highs[i - 1]

            # Must be within setup window
            if len(highs) - current_idx > setup_window:
                continue

            # Check bar distance constraints
            bar_diff = current_idx - prior_idx
            if bar_diff < min_bars or bar_diff > max_bars:
                continue

            # Check for higher high in price
            if highs[current_idx] <= highs[prior_idx]:
                continue  # Not a higher high

            # Check for lower high in RSI
            if rsi_values[current_idx] >= rsi_values[prior_idx]:
                continue  # Not a lower RSI high

            # Bearish divergence found!
            return (
                current_idx,
                prior_idx,
                rsi_values[current_idx],
                rsi_values[prior_idx]
            )

        return None

    def _find_swing_lows(self, lows: np.ndarray, window: int = 3) -> List[int]:
        """Find swing low indices in the price data."""
        swing_lows = []

        for i in range(window, len(lows) - window):
            is_low = True
            for j in range(1, window + 1):
                if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append(i)

        return swing_lows

    def _find_swing_highs(self, highs: np.ndarray, window: int = 3) -> List[int]:
        """Find swing high indices in the price data."""
        swing_highs = []

        for i in range(window, len(highs) - window):
            is_high = True
            for j in range(1, window + 1):
                if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append(i)

        return swing_highs

    def _calculate_rsi(self, closes: np.ndarray, period: int) -> Optional[np.ndarray]:
        """Calculate RSI values."""
        if len(closes) < period + 1:
            return None

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _create_signal(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Optional[SignalSetup]:
        """Create a SignalSetup from a divergence candidate."""

        # Score the candidate
        if signal_type == SignalType.BUY:
            score_components = self.scorer.score_bullish(candidate)
        else:
            score_components = self.scorer.score_bearish(candidate)

        composite_score = score_components.composite_score

        # Bonus for RSI extremes
        rsi_current = candidate.get('rsi_current', 50)
        if signal_type == SignalType.BUY and rsi_current <= 30:
            composite_score = min(100, composite_score + 10)
        elif signal_type == SignalType.BUY and rsi_current <= 40:
            composite_score = min(100, composite_score + 5)
        elif signal_type == SignalType.SELL and rsi_current >= 70:
            composite_score = min(100, composite_score + 10)
        elif signal_type == SignalType.SELL and rsi_current >= 60:
            composite_score = min(100, composite_score + 5)

        # Apply minimum threshold
        if composite_score < self.config.min_score_threshold:
            return None

        # Calculate entry levels
        entry, stop, tp1, tp2 = self._calculate_entry_levels(candidate, signal_type)

        # Calculate risk percent
        risk_pct = abs(float(entry) - float(stop)) / float(entry) * 100 if float(entry) > 0 else 0

        # Build description
        description = self._build_description(candidate, signal_type)

        # Create signal
        signal = SignalSetup(
            ticker=candidate['ticker'],
            scanner_name=self.scanner_name,
            signal_type=signal_type,
            signal_timestamp=candidate.get('candle_timestamp') or datetime.now(),
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=Decimal(str(round(self.config.tp1_rr_ratio, 2))),
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
            market_regime=self._determine_market_regime(candidate, signal_type),
            max_risk_per_trade_pct=Decimal(str(self.config.max_risk_per_trade_pct)),
            raw_data=candidate,
        )

        # Calculate position size (reduced for reversal plays)
        signal.suggested_position_size_pct = self.calculate_position_size(
            Decimal(str(self.config.max_risk_per_trade_pct * 0.8)),
            entry,
            stop,
            signal.quality_tier
        )

        return signal

    def _calculate_entry_levels(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> Tuple[Decimal, Decimal, Decimal, Optional[Decimal]]:
        """Calculate entry, stop, and take profit levels."""
        current_price = Decimal(str(candidate.get('current_price', 0)))
        atr_percent = float(candidate.get('atr_percent') or 3.0)
        atr = current_price * Decimal(str(atr_percent / 100))

        entry = current_price

        if signal_type == SignalType.BUY:
            # Stop below the divergence low
            div_low = Decimal(str(candidate.get('price_current_low', current_price * Decimal('0.95'))))
            stop = div_low - (atr * Decimal(str(self.config.atr_stop_multiplier / 2)))

            # Max stop constraint
            max_stop = entry * (Decimal('1') - Decimal(str(self.config.max_stop_loss_pct / 100)))
            stop = max(stop, max_stop)

            # Take profits
            risk = entry - stop
            tp1 = entry + (risk * Decimal(str(self.config.tp1_rr_ratio)))
            tp2 = entry + (risk * Decimal(str(self.config.tp2_rr_ratio)))

        else:  # SELL
            # Stop above the divergence high
            div_high = Decimal(str(candidate.get('price_current_high', current_price * Decimal('1.05'))))
            stop = div_high + (atr * Decimal(str(self.config.atr_stop_multiplier / 2)))

            # Max stop constraint
            max_stop = entry * (Decimal('1') + Decimal(str(self.config.max_stop_loss_pct / 100)))
            stop = min(stop, max_stop)

            # Take profits
            risk = stop - entry
            tp1 = entry - (risk * Decimal(str(self.config.tp1_rr_ratio)))
            tp2 = entry - (risk * Decimal(str(self.config.tp2_rr_ratio)))

        return entry, stop, tp1, tp2

    def _calculate_atr_percent(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> float:
        """Calculate ATR as percentage of price."""
        if len(closes) < period + 1:
            return 3.0

        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        atr = np.mean(tr[-period:])
        return (atr / closes[-1]) * 100 if closes[-1] > 0 else 3.0

    async def _get_daily_candles(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
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
        """Get all active tickers for RSI divergence scan."""
        # Scan ALL active stocks - no pre-filtering
        return await self.get_all_active_tickers()

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
                m.atr_percent, m.current_price,
                m.sma200_signal
            FROM stock_watchlist w
            JOIN stock_screening_metrics m ON w.ticker = m.ticker
                AND m.calculation_date = w.calculation_date
            WHERE w.ticker = $1 AND w.calculation_date = $2
        """
        row = await self.db.fetchrow(query, ticker, calculation_date)
        return dict(row) if row else None

    def _build_description(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> str:
        """Build human-readable setup description."""
        ticker = candidate['ticker']
        div_type = candidate.get('divergence_type', 'unknown')
        rsi_current = candidate.get('rsi_current', 50)
        rsi_prior = candidate.get('rsi_prior', 50)
        bars = candidate.get('divergence_bars', 0)

        direction = "bullish" if signal_type == SignalType.BUY else "bearish"

        return (
            f"{ticker} {direction} RSI divergence. "
            f"RSI {rsi_prior:.0f} -> {rsi_current:.0f} over {bars} bars. "
            f"Price made {'lower low' if signal_type == SignalType.BUY else 'higher high'} "
            f"while RSI made {'higher low' if signal_type == SignalType.BUY else 'lower high'}. "
            f"Potential reversal signal."
        )

    def _build_confluence_factors(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> List[str]:
        """Build list of confluence factors."""
        factors = []

        div_type = candidate.get('divergence_type', '')
        if div_type == 'bullish':
            factors.append("Bullish RSI Divergence")
        else:
            factors.append("Bearish RSI Divergence")

        # RSI level
        rsi = candidate.get('rsi_current', 50)
        if signal_type == SignalType.BUY:
            if rsi <= 30:
                factors.append(f"RSI Oversold ({rsi:.0f})")
            elif rsi <= 40:
                factors.append(f"RSI Low ({rsi:.0f})")
        else:
            if rsi >= 70:
                factors.append(f"RSI Overbought ({rsi:.0f})")
            elif rsi >= 60:
                factors.append(f"RSI High ({rsi:.0f})")

        # Trend context
        sma200_signal = candidate.get('sma200_signal', '')
        if signal_type == SignalType.BUY and 'below' in sma200_signal:
            factors.append("Downtrend Reversal")
        elif signal_type == SignalType.SELL and 'above' in sma200_signal:
            factors.append("Uptrend Reversal")

        # Divergence strength
        bars = candidate.get('divergence_bars', 0)
        if bars >= 15:
            factors.append("Strong Divergence (15+ bars)")
        elif bars >= 10:
            factors.append("Moderate Divergence (10+ bars)")

        return factors

    def _determine_market_regime(
        self,
        candidate: Dict[str, Any],
        signal_type: SignalType
    ) -> str:
        """Determine market regime."""
        div_type = candidate.get('divergence_type', '')
        trend = candidate.get('trend_strength', '')

        if signal_type == SignalType.BUY:
            regime = "Bullish Divergence"
        else:
            regime = "Bearish Divergence"

        if 'strong' in str(trend).lower():
            regime += " - Strong Trend"
        elif 'up' in str(trend).lower() or 'down' in str(trend).lower():
            regime += " - Trending"

        return regime
