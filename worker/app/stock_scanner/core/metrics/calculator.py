"""
Stock Metrics Calculator

Calculates screening metrics for stocks:
- ATR (Average True Range)
- Volume averages and relative volume
- Price momentum (1d, 5d, 20d, 60d changes)
- Moving averages (SMA20, SMA50, SMA200, EMA20)
- Trend classification
- RSI, MACD

Enhanced signals (Finviz-inspired):
- RSI signal classification (oversold/overbought)
- SMA signal classification (above/below/crossover)
- SMA cross detection (golden cross/death cross)
- MACD cross signals
- 52-week high/low proximity
- Gap detection
- Candlestick pattern recognition
- Multi-timeframe performance (1W, 1M, 3M, 6M, YTD)
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate screening metrics for stock watchlist filtering.

    Designed to run as a batch job after daily candle synthesis.
    Calculates all technical indicators needed for screening.
    """

    def __init__(self, db_manager):
        self.db = db_manager

    async def calculate_all_metrics(
        self,
        calculation_date: datetime = None,
        ticker: str = None,
        concurrency: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate metrics for all active stocks.

        Args:
            calculation_date: Date to calculate for (defaults to yesterday)
            ticker: Specific ticker or None for all
            concurrency: Number of parallel calculations

        Returns:
            Statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("METRICS CALCULATION")
        logger.info("=" * 60)

        start_time = datetime.now()

        if calculation_date is None:
            calculation_date = datetime.now().date() - timedelta(days=1)

        # Get tickers to process
        if ticker:
            tickers = [ticker]
        else:
            tickers = await self._get_tickers_with_daily_data()

        logger.info(f"Calculating metrics for {len(tickers)} tickers")
        logger.info(f"Calculation date: {calculation_date}")

        # Process with concurrency
        successful = 0
        failed = 0
        semaphore = asyncio.Semaphore(concurrency)

        async def process_ticker(tkr: str):
            async with semaphore:
                try:
                    await self.calculate_ticker_metrics(tkr, calculation_date)
                    return (tkr, True, None)
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics for {tkr}: {e}")
                    return (tkr, False, str(e))

        tasks = [process_ticker(t) for t in tickers]
        results = await asyncio.gather(*tasks)

        for tkr, success, error in results:
            if success:
                successful += 1
            else:
                failed += 1

        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            'total_tickers': len(tickers),
            'successful': successful,
            'failed': failed,
            'calculation_date': str(calculation_date),
            'duration_seconds': round(elapsed, 2),
            'avg_per_ticker_ms': round((elapsed / len(tickers)) * 1000, 2) if tickers else 0
        }

        logger.info(f"\nMetrics calculation complete:")
        logger.info(f"  Successful: {successful}/{len(tickers)}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Duration: {int(elapsed//60)}m {int(elapsed%60)}s")
        logger.info(f"  Avg per ticker: {stats['avg_per_ticker_ms']:.1f}ms")

        return stats

    async def calculate_ticker_metrics(
        self,
        ticker: str,
        calculation_date: datetime
    ) -> None:
        """
        Calculate all metrics for a single ticker.

        Uses vectorized NumPy operations for performance.
        """

        # Fetch daily candles (need 200+ for SMA200)
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized
            WHERE ticker = $1 AND timeframe = '1d'
              AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') <= $2
            ORDER BY timestamp DESC
            LIMIT 250
        """

        rows = await self.db.fetch(query, ticker, calculation_date)

        if len(rows) < 20:
            logger.debug(f"{ticker}: Insufficient data ({len(rows)} days)")
            return

        # Convert to numpy arrays (reversed to oldest first)
        closes = np.array([float(r['close']) for r in reversed(rows)])
        highs = np.array([float(r['high']) for r in reversed(rows)])
        lows = np.array([float(r['low']) for r in reversed(rows)])
        volumes = np.array([int(r['volume'] or 0) for r in reversed(rows)])

        current_price = closes[-1]
        current_volume = volumes[-1]

        # Calculate all metrics
        metrics = {
            'ticker': ticker,
            'calculation_date': calculation_date,
            'current_price': round(current_price, 4),
            'candles_available': len(rows)
        }

        # ATR (14-day)
        atr_14 = self._calculate_atr(highs, lows, closes, period=14)
        if atr_14:
            metrics['atr_14'] = round(atr_14, 4)
            metrics['atr_percent'] = round((atr_14 / current_price) * 100, 2)

        # Historical Volatility (20-day, annualized)
        hv_20 = self._calculate_hv(closes, period=20)
        if hv_20:
            # Cap at 999.99 to fit DECIMAL(5,2) - values above this indicate extreme volatility
            metrics['historical_volatility_20'] = min(round(hv_20, 2), 999.99)

        # Volume metrics
        if len(volumes) >= 20:
            avg_vol_20 = int(np.mean(volumes[-20:]))
            metrics['avg_volume_20'] = avg_vol_20
            metrics['avg_dollar_volume'] = round(avg_vol_20 * current_price, 2)
            metrics['current_volume'] = current_volume
            if avg_vol_20 > 0:
                metrics['relative_volume'] = round(current_volume / avg_vol_20, 2)

        # Price changes
        if len(closes) >= 2:
            metrics['price_change_1d'] = round(((closes[-1] / closes[-2]) - 1) * 100, 2)
        if len(closes) >= 6:
            metrics['price_change_5d'] = round(((closes[-1] / closes[-6]) - 1) * 100, 2)
        if len(closes) >= 21:
            metrics['price_change_20d'] = round(((closes[-1] / closes[-21]) - 1) * 100, 2)
        if len(closes) >= 61:
            metrics['price_change_60d'] = round(((closes[-1] / closes[-61]) - 1) * 100, 2)

        # Moving averages
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            metrics['sma_20'] = round(sma_20, 4)
            metrics['price_vs_sma20'] = round(((current_price / sma_20) - 1) * 100, 2)

        if len(closes) >= 50:
            sma_50 = np.mean(closes[-50:])
            metrics['sma_50'] = round(sma_50, 4)
            metrics['price_vs_sma50'] = round(((current_price / sma_50) - 1) * 100, 2)

        if len(closes) >= 200:
            sma_200 = np.mean(closes[-200:])
            metrics['sma_200'] = round(sma_200, 4)
            metrics['price_vs_sma200'] = round(((current_price / sma_200) - 1) * 100, 2)

        # EMA 20
        if len(closes) >= 20:
            ema_20 = self._calculate_ema(closes, period=20)
            metrics['ema_20'] = round(ema_20, 4)

        # Trend classification
        metrics['trend_strength'] = self._classify_trend(
            current_price,
            metrics.get('sma_20'),
            metrics.get('sma_50'),
            metrics.get('sma_200')
        )

        metrics['ma_alignment'] = self._classify_ma_alignment(
            current_price,
            metrics.get('sma_20'),
            metrics.get('sma_50'),
            metrics.get('sma_200')
        )

        # RSI (14-day)
        rsi_14 = self._calculate_rsi(closes, period=14)
        if rsi_14:
            metrics['rsi_14'] = round(rsi_14, 2)

        # MACD
        macd_vals = self._calculate_macd(closes)
        if macd_vals:
            metrics['macd'] = round(macd_vals['macd'], 6)
            metrics['macd_signal'] = round(macd_vals['signal'], 6)
            metrics['macd_histogram'] = round(macd_vals['histogram'], 6)

        # ADX (14-day) - Average Directional Index for trend strength
        if metrics.get('atr_14'):
            adx = self._calculate_adx(highs, lows, closes, metrics['atr_14'], period=14)
            if adx is not None:
                metrics['adx'] = round(adx, 2)

        # Daily range percent
        if len(highs) >= 1:
            daily_range = ((highs[-1] - lows[-1]) / current_price) * 100
            metrics['daily_range_percent'] = round(daily_range, 2)

        # Weekly range percent (5-day)
        if len(highs) >= 5:
            week_high = np.max(highs[-5:])
            week_low = np.min(lows[-5:])
            weekly_range = ((week_high - week_low) / current_price) * 100
            metrics['weekly_range_percent'] = round(weekly_range, 2)

        # Z-score (50-day)
        if len(closes) >= 50:
            mean_50 = np.mean(closes[-50:])
            std_50 = np.std(closes[-50:])
            if std_50 > 0:
                metrics['z_score_50'] = round((current_price - mean_50) / std_50, 4)

        # Volume percentile
        if len(volumes) >= 60:
            metrics['percentile_volume'] = round(
                np.percentile(volumes[-60:],
                              np.searchsorted(np.sort(volumes[-60:]), current_volume) / 60 * 100),
                2
            )

        # Data quality assessment
        metrics['data_quality'] = 'good' if len(rows) >= 50 else 'incomplete'

        # ============================================================
        # ENHANCED SIGNALS (Finviz-inspired)
        # ============================================================

        # Get opens for gap detection and candlestick patterns
        opens = np.array([float(r['open']) for r in reversed(rows)])

        # RSI Signal Classification
        if metrics.get('rsi_14'):
            metrics['rsi_signal'] = self._classify_rsi_signal(metrics['rsi_14'])

        # SMA Signal Classifications (price relative to SMA)
        if metrics.get('sma_20'):
            metrics['sma20_signal'] = self._classify_sma_signal(
                current_price, metrics['sma_20'],
                closes[-2] if len(closes) >= 2 else None,
                np.mean(closes[-21:-1]) if len(closes) >= 21 else None
            )
        if metrics.get('sma_50'):
            metrics['sma50_signal'] = self._classify_sma_signal(
                current_price, metrics['sma_50'],
                closes[-2] if len(closes) >= 2 else None,
                np.mean(closes[-51:-1]) if len(closes) >= 51 else None
            )
        if metrics.get('sma_200'):
            metrics['sma200_signal'] = self._classify_sma_signal(
                current_price, metrics['sma_200'],
                closes[-2] if len(closes) >= 2 else None,
                np.mean(closes[-201:-1]) if len(closes) >= 201 else None
            )

        # SMA Cross Signal (Golden Cross / Death Cross)
        if len(closes) >= 201:
            metrics['sma_cross_signal'] = self._detect_sma_cross(closes)

        # MACD Cross Signal
        if macd_vals and len(closes) >= 36:  # Need enough data for previous MACD
            prev_macd = self._calculate_macd(closes[:-1])
            if prev_macd:
                metrics['macd_cross_signal'] = self._classify_macd_cross(
                    macd_vals['histogram'],
                    prev_macd['histogram']
                )

        # 52-Week High/Low (need ~252 trading days)
        if len(highs) >= 200:  # Use available data, ideally 252
            lookback = min(len(highs), 252)
            high_52w = float(np.max(highs[-lookback:]))
            low_52w = float(np.min(lows[-lookback:]))
            metrics['high_52w'] = round(high_52w, 4)
            metrics['low_52w'] = round(low_52w, 4)
            metrics['pct_from_52w_high'] = round(((current_price - high_52w) / high_52w) * 100, 2)
            metrics['pct_from_52w_low'] = round(((current_price - low_52w) / low_52w) * 100, 2)
            metrics['high_low_signal'] = self._classify_high_low_signal(
                metrics['pct_from_52w_high'],
                metrics['pct_from_52w_low']
            )

        # Gap Detection
        if len(opens) >= 2 and len(closes) >= 2:
            gap_pct = ((opens[-1] - closes[-2]) / closes[-2]) * 100
            metrics['gap_percent'] = round(gap_pct, 2)
            metrics['gap_signal'] = self._classify_gap_signal(gap_pct)

        # Candlestick Pattern Detection
        if len(opens) >= 2:
            pattern = self._detect_candlestick_pattern(opens, highs, lows, closes)
            if pattern:
                metrics['candlestick_pattern'] = pattern

        # Multi-timeframe Performance
        if len(closes) >= 6:  # 1 week
            metrics['perf_1w'] = round(((closes[-1] / closes[-6]) - 1) * 100, 2)
        if len(closes) >= 22:  # 1 month
            metrics['perf_1m'] = round(((closes[-1] / closes[-22]) - 1) * 100, 2)
        if len(closes) >= 66:  # 3 months
            metrics['perf_3m'] = round(((closes[-1] / closes[-66]) - 1) * 100, 2)
        if len(closes) >= 126:  # 6 months
            metrics['perf_6m'] = round(((closes[-1] / closes[-126]) - 1) * 100, 2)

        # YTD Performance (find first trading day of current year)
        ytd_perf = await self._calculate_ytd_performance(ticker, current_price, calculation_date)
        if ytd_perf is not None:
            metrics['perf_ytd'] = round(ytd_perf, 2)

        # Save to database
        await self._save_metrics(metrics)

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> Optional[float]:
        """Calculate Average True Range using Wilder's smoothing."""
        if len(closes) < period + 1:
            return None

        # True Range
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Wilder's smoothing (exponential with alpha = 1/period)
        atr = np.zeros(len(tr))
        atr[period-1] = np.mean(tr[:period])

        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return float(atr[-1])

    def _calculate_hv(
        self,
        closes: np.ndarray,
        period: int = 20
    ) -> Optional[float]:
        """Calculate Historical Volatility (annualized)."""
        if len(closes) < period + 1:
            return None

        returns = np.diff(np.log(closes))
        hv = np.std(returns[-period:]) * np.sqrt(252) * 100
        return float(hv)

    def _calculate_ema(
        self,
        prices: np.ndarray,
        period: int
    ) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return float(prices[-1])

        alpha = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return float(ema)

    def _calculate_rsi(
        self,
        closes: np.ndarray,
        period: int = 14
    ) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(closes) < period + 1:
            return None

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(
        self,
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Optional[Dict[str, float]]:
        """Calculate MACD indicator."""
        if len(closes) < slow + signal:
            return None

        # EMA calculations
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros(len(data))
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {
            'macd': float(macd_line[-1]),
            'signal': float(signal_line[-1]),
            'histogram': float(histogram[-1])
        }

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        atr: float,
        period: int = 14
    ) -> Optional[float]:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength regardless of direction.
        - ADX > 20: Trending market (Welles Wilder threshold)
        - ADX > 25: Strong trend
        - ADX > 40: Very strong trend
        - ADX < 20: Weak/ranging market
        """
        if len(closes) < period + 1 or atr <= 0:
            return None

        # Calculate directional movement
        high_diff = np.diff(highs)
        low_diff = -np.diff(lows)

        # +DM: positive directional movement
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        # -DM: negative directional movement
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Smooth +DM and -DM using Wilder's smoothing (EMA with alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.zeros(len(data))
            result[period-1] = np.sum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result

        smooth_plus_dm = wilder_smooth(plus_dm, period)
        smooth_minus_dm = wilder_smooth(minus_dm, period)

        # Calculate True Range and smooth it
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        smooth_tr = wilder_smooth(tr, period)

        # Avoid division by zero
        smooth_tr = np.where(smooth_tr == 0, 1, smooth_tr)

        # Calculate +DI and -DI
        plus_di = 100 * smooth_plus_dm / smooth_tr
        minus_di = 100 * smooth_minus_dm / smooth_tr

        # Calculate DX
        di_sum = plus_di + minus_di
        di_sum = np.where(di_sum == 0, 1, di_sum)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum

        # Smooth DX to get ADX
        adx = wilder_smooth(dx, period) / period

        # Return latest ADX value (need enough data)
        if len(adx) >= period * 2:
            return float(adx[-1])
        return None

    def _classify_trend(
        self,
        price: float,
        sma_20: Optional[float],
        sma_50: Optional[float],
        sma_200: Optional[float]
    ) -> str:
        """Classify trend strength based on price vs MAs."""
        if sma_20 is None or sma_50 is None:
            return 'neutral'

        # Strong uptrend: price > sma20 > sma50
        if price > sma_20 > sma_50:
            if sma_200 and sma_50 > sma_200:
                return 'strong_up'
            return 'up'

        # Strong downtrend: price < sma20 < sma50
        if price < sma_20 < sma_50:
            if sma_200 and sma_50 < sma_200:
                return 'strong_down'
            return 'down'

        return 'neutral'

    def _classify_ma_alignment(
        self,
        price: float,
        sma_20: Optional[float],
        sma_50: Optional[float],
        sma_200: Optional[float]
    ) -> str:
        """Classify MA alignment (bullish/bearish/mixed)."""
        if sma_20 is None or sma_50 is None:
            return 'mixed'

        if sma_200:
            if price > sma_20 > sma_50 > sma_200:
                return 'bullish'
            if price < sma_20 < sma_50 < sma_200:
                return 'bearish'
        else:
            if price > sma_20 > sma_50:
                return 'bullish'
            if price < sma_20 < sma_50:
                return 'bearish'

        return 'mixed'

    # ================================================================
    # ENHANCED SIGNAL METHODS (Finviz-inspired)
    # ================================================================

    def _classify_rsi_signal(self, rsi: float) -> str:
        """
        Classify RSI into actionable signals.

        Returns:
            'oversold_extreme': RSI < 20 (strong buy zone)
            'oversold': RSI 20-30 (buy zone)
            'neutral': RSI 30-70
            'overbought': RSI 70-80 (sell zone)
            'overbought_extreme': RSI > 80 (strong sell zone)
        """
        if rsi < 20:
            return 'oversold_extreme'
        elif rsi < 30:
            return 'oversold'
        elif rsi > 80:
            return 'overbought_extreme'
        elif rsi > 70:
            return 'overbought'
        return 'neutral'

    def _classify_sma_signal(
        self,
        current_price: float,
        current_sma: float,
        prev_price: Optional[float],
        prev_sma: Optional[float]
    ) -> str:
        """
        Classify price relationship to SMA.

        Returns:
            'crossed_above': Price just crossed above SMA (bullish)
            'crossed_below': Price just crossed below SMA (bearish)
            'above': Price is above SMA
            'below': Price is below SMA
        """
        price_above = current_price > current_sma

        # Check for crossover
        if prev_price is not None and prev_sma is not None:
            prev_above = prev_price > prev_sma
            if price_above and not prev_above:
                return 'crossed_above'
            if not price_above and prev_above:
                return 'crossed_below'

        return 'above' if price_above else 'below'

    def _detect_sma_cross(self, closes: np.ndarray) -> str:
        """
        Detect Golden Cross or Death Cross (SMA50 vs SMA200).

        Returns:
            'golden_cross': SMA50 just crossed above SMA200 (bullish)
            'death_cross': SMA50 just crossed below SMA200 (bearish)
            'bullish': SMA50 > SMA200
            'bearish': SMA50 < SMA200
        """
        if len(closes) < 201:
            return 'unknown'

        # Current SMAs
        sma_50 = np.mean(closes[-50:])
        sma_200 = np.mean(closes[-200:])

        # Previous day SMAs
        sma_50_prev = np.mean(closes[-51:-1])
        sma_200_prev = np.mean(closes[-201:-1])

        sma50_above = sma_50 > sma_200
        sma50_prev_above = sma_50_prev > sma_200_prev

        # Check for crossover
        if sma50_above and not sma50_prev_above:
            return 'golden_cross'
        if not sma50_above and sma50_prev_above:
            return 'death_cross'

        return 'bullish' if sma50_above else 'bearish'

    def _classify_macd_cross(
        self,
        current_histogram: float,
        prev_histogram: float
    ) -> str:
        """
        Classify MACD histogram crossover.

        Returns:
            'bullish_cross': Histogram crossed above zero
            'bearish_cross': Histogram crossed below zero
            'bullish': Histogram positive
            'bearish': Histogram negative
        """
        curr_positive = current_histogram > 0
        prev_positive = prev_histogram > 0

        if curr_positive and not prev_positive:
            return 'bullish_cross'
        if not curr_positive and prev_positive:
            return 'bearish_cross'

        return 'bullish' if curr_positive else 'bearish'

    def _classify_high_low_signal(
        self,
        pct_from_high: float,
        pct_from_low: float
    ) -> str:
        """
        Classify price position relative to 52-week range.

        Returns:
            'new_high': At or above 52W high
            'near_high': Within 5% of 52W high
            'new_low': At or below 52W low
            'near_low': Within 5% of 52W low
            'middle': In the middle range
        """
        if pct_from_high >= 0:
            return 'new_high'
        elif pct_from_high >= -5:
            return 'near_high'
        elif pct_from_low <= 0:
            return 'new_low'
        elif pct_from_low <= 5:
            return 'near_low'
        return 'middle'

    def _classify_gap_signal(self, gap_pct: float) -> str:
        """
        Classify gap size.

        Returns:
            'gap_up_large': Gap up > 4%
            'gap_up': Gap up 2-4%
            'gap_up_small': Gap up 0.5-2%
            'gap_down_large': Gap down > 4%
            'gap_down': Gap down 2-4%
            'gap_down_small': Gap down 0.5-2%
            'no_gap': Gap < 0.5%
        """
        if gap_pct >= 4:
            return 'gap_up_large'
        elif gap_pct >= 2:
            return 'gap_up'
        elif gap_pct >= 0.5:
            return 'gap_up_small'
        elif gap_pct <= -4:
            return 'gap_down_large'
        elif gap_pct <= -2:
            return 'gap_down'
        elif gap_pct <= -0.5:
            return 'gap_down_small'
        return 'no_gap'

    def _detect_candlestick_pattern(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Optional[str]:
        """
        Detect common candlestick patterns on the latest candle.

        Returns pattern name or None if no significant pattern detected.
        """
        if len(opens) < 2:
            return None

        # Current candle
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        if total_range == 0:
            return None

        body_pct = body / total_range

        # Previous candle
        prev_o, prev_h, prev_l, prev_c = opens[-2], highs[-2], lows[-2], closes[-2]
        prev_body = abs(prev_c - prev_o)

        is_bullish = c > o
        is_bearish = c < o
        prev_bullish = prev_c > prev_o
        prev_bearish = prev_c < prev_o

        # Doji (body < 10% of range)
        if body_pct < 0.1:
            if upper_shadow > body * 2 and lower_shadow > body * 2:
                return 'doji'
            elif upper_shadow > lower_shadow * 2:
                return 'gravestone_doji'
            elif lower_shadow > upper_shadow * 2:
                return 'dragonfly_doji'
            return 'doji'

        # Hammer (small body at top, long lower shadow, bullish reversal)
        if lower_shadow > body * 2 and upper_shadow < body * 0.5:
            if is_bullish:
                return 'hammer'
            else:
                return 'hanging_man'

        # Inverted Hammer / Shooting Star
        if upper_shadow > body * 2 and lower_shadow < body * 0.5:
            if is_bullish:
                return 'inverted_hammer'
            else:
                return 'shooting_star'

        # Engulfing patterns (need to compare with previous candle)
        if prev_body > 0:
            # Bullish Engulfing
            if is_bullish and prev_bearish:
                if o <= prev_c and c >= prev_o and body > prev_body:
                    return 'bullish_engulfing'

            # Bearish Engulfing
            if is_bearish and prev_bullish:
                if o >= prev_c and c <= prev_o and body > prev_body:
                    return 'bearish_engulfing'

        # Marubozu (no or very small shadows)
        shadow_ratio = (upper_shadow + lower_shadow) / total_range
        if shadow_ratio < 0.1:
            if is_bullish:
                return 'bullish_marubozu'
            else:
                return 'bearish_marubozu'

        # Strong candles (body > 70% of range)
        if body_pct > 0.7:
            if is_bullish:
                return 'strong_bullish'
            else:
                return 'strong_bearish'

        return None

    async def _calculate_ytd_performance(
        self,
        ticker: str,
        current_price: float,
        calculation_date
    ) -> Optional[float]:
        """Calculate Year-To-Date performance."""
        try:
            # Get the year from calculation_date
            if isinstance(calculation_date, datetime):
                year = calculation_date.year
            else:
                year = calculation_date.year

            # Query for first trading day of the year
            query = """
                SELECT close
                FROM stock_candles_synthesized
                WHERE ticker = $1 AND timeframe = '1d'
                  AND DATE(timestamp AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York') >= $2
                ORDER BY timestamp ASC
                LIMIT 1
            """
            year_start = date(year, 1, 1)
            rows = await self.db.fetch(query, ticker, year_start)

            if rows:
                year_start_price = float(rows[0]['close'])
                if year_start_price > 0:
                    return ((current_price / year_start_price) - 1) * 100
        except Exception as e:
            logger.debug(f"Error calculating YTD for {ticker}: {e}")

        return None

    async def _get_tickers_with_daily_data(self) -> List[str]:
        """Get tickers that have synthesized daily data."""
        query = """
            SELECT DISTINCT ticker
            FROM stock_candles_synthesized
            WHERE timeframe = '1d'
            ORDER BY ticker
        """
        rows = await self.db.fetch(query)
        return [r['ticker'] for r in rows]

    async def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to database."""
        query = """
            INSERT INTO stock_screening_metrics (
                ticker, calculation_date, current_price,
                atr_14, atr_percent, historical_volatility_20,
                avg_volume_20, avg_dollar_volume, current_volume, relative_volume,
                price_change_1d, price_change_5d, price_change_20d, price_change_60d,
                sma_20, sma_50, sma_200, ema_20,
                price_vs_sma20, price_vs_sma50, price_vs_sma200,
                trend_strength, ma_alignment,
                rsi_14, macd, macd_signal, macd_histogram, adx,
                daily_range_percent, weekly_range_percent,
                z_score_50, percentile_volume,
                data_quality, candles_available,
                rsi_signal, sma20_signal, sma50_signal, sma200_signal,
                sma_cross_signal, macd_cross_signal,
                high_52w, low_52w, pct_from_52w_high, pct_from_52w_low, high_low_signal,
                gap_percent, gap_signal, candlestick_pattern,
                perf_1w, perf_1m, perf_3m, perf_6m, perf_ytd
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
                $29, $30, $31, $32, $33, $34,
                $35, $36, $37, $38, $39, $40,
                $41, $42, $43, $44, $45,
                $46, $47, $48,
                $49, $50, $51, $52, $53
            )
            ON CONFLICT (ticker, calculation_date)
            DO UPDATE SET
                current_price = EXCLUDED.current_price,
                atr_14 = EXCLUDED.atr_14,
                atr_percent = EXCLUDED.atr_percent,
                historical_volatility_20 = EXCLUDED.historical_volatility_20,
                avg_volume_20 = EXCLUDED.avg_volume_20,
                avg_dollar_volume = EXCLUDED.avg_dollar_volume,
                current_volume = EXCLUDED.current_volume,
                relative_volume = EXCLUDED.relative_volume,
                price_change_1d = EXCLUDED.price_change_1d,
                price_change_5d = EXCLUDED.price_change_5d,
                price_change_20d = EXCLUDED.price_change_20d,
                price_change_60d = EXCLUDED.price_change_60d,
                sma_20 = EXCLUDED.sma_20,
                sma_50 = EXCLUDED.sma_50,
                sma_200 = EXCLUDED.sma_200,
                ema_20 = EXCLUDED.ema_20,
                price_vs_sma20 = EXCLUDED.price_vs_sma20,
                price_vs_sma50 = EXCLUDED.price_vs_sma50,
                price_vs_sma200 = EXCLUDED.price_vs_sma200,
                trend_strength = EXCLUDED.trend_strength,
                ma_alignment = EXCLUDED.ma_alignment,
                rsi_14 = EXCLUDED.rsi_14,
                macd = EXCLUDED.macd,
                macd_signal = EXCLUDED.macd_signal,
                macd_histogram = EXCLUDED.macd_histogram,
                adx = EXCLUDED.adx,
                daily_range_percent = EXCLUDED.daily_range_percent,
                weekly_range_percent = EXCLUDED.weekly_range_percent,
                z_score_50 = EXCLUDED.z_score_50,
                percentile_volume = EXCLUDED.percentile_volume,
                data_quality = EXCLUDED.data_quality,
                candles_available = EXCLUDED.candles_available,
                rsi_signal = EXCLUDED.rsi_signal,
                sma20_signal = EXCLUDED.sma20_signal,
                sma50_signal = EXCLUDED.sma50_signal,
                sma200_signal = EXCLUDED.sma200_signal,
                sma_cross_signal = EXCLUDED.sma_cross_signal,
                macd_cross_signal = EXCLUDED.macd_cross_signal,
                high_52w = EXCLUDED.high_52w,
                low_52w = EXCLUDED.low_52w,
                pct_from_52w_high = EXCLUDED.pct_from_52w_high,
                pct_from_52w_low = EXCLUDED.pct_from_52w_low,
                high_low_signal = EXCLUDED.high_low_signal,
                gap_percent = EXCLUDED.gap_percent,
                gap_signal = EXCLUDED.gap_signal,
                candlestick_pattern = EXCLUDED.candlestick_pattern,
                perf_1w = EXCLUDED.perf_1w,
                perf_1m = EXCLUDED.perf_1m,
                perf_3m = EXCLUDED.perf_3m,
                perf_6m = EXCLUDED.perf_6m,
                perf_ytd = EXCLUDED.perf_ytd,
                created_at = NOW()
        """

        await self.db.execute(
            query,
            metrics['ticker'],
            metrics['calculation_date'],
            metrics.get('current_price'),
            metrics.get('atr_14'),
            metrics.get('atr_percent'),
            metrics.get('historical_volatility_20'),
            metrics.get('avg_volume_20'),
            metrics.get('avg_dollar_volume'),
            metrics.get('current_volume'),
            metrics.get('relative_volume'),
            metrics.get('price_change_1d'),
            metrics.get('price_change_5d'),
            metrics.get('price_change_20d'),
            metrics.get('price_change_60d'),
            metrics.get('sma_20'),
            metrics.get('sma_50'),
            metrics.get('sma_200'),
            metrics.get('ema_20'),
            metrics.get('price_vs_sma20'),
            metrics.get('price_vs_sma50'),
            metrics.get('price_vs_sma200'),
            metrics.get('trend_strength'),
            metrics.get('ma_alignment'),
            metrics.get('rsi_14'),
            metrics.get('macd'),
            metrics.get('macd_signal'),
            metrics.get('macd_histogram'),
            metrics.get('adx'),
            metrics.get('daily_range_percent'),
            metrics.get('weekly_range_percent'),
            metrics.get('z_score_50'),
            metrics.get('percentile_volume'),
            metrics.get('data_quality'),
            metrics.get('candles_available'),
            # Enhanced signals
            metrics.get('rsi_signal'),
            metrics.get('sma20_signal'),
            metrics.get('sma50_signal'),
            metrics.get('sma200_signal'),
            metrics.get('sma_cross_signal'),
            metrics.get('macd_cross_signal'),
            metrics.get('high_52w'),
            metrics.get('low_52w'),
            metrics.get('pct_from_52w_high'),
            metrics.get('pct_from_52w_low'),
            metrics.get('high_low_signal'),
            metrics.get('gap_percent'),
            metrics.get('gap_signal'),
            metrics.get('candlestick_pattern'),
            metrics.get('perf_1w'),
            metrics.get('perf_1m'),
            metrics.get('perf_3m'),
            metrics.get('perf_6m'),
            metrics.get('perf_ytd'),
        )
