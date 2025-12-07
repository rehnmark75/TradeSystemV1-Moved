"""
Stock Metrics Calculator

Calculates screening metrics for stocks:
- ATR (Average True Range)
- Volume averages and relative volume
- Price momentum (1d, 5d, 20d, 60d changes)
- Moving averages (SMA20, SMA50, SMA200, EMA20)
- Trend classification
- RSI, MACD
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

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
                rsi_14, macd, macd_signal, macd_histogram,
                daily_range_percent, weekly_range_percent,
                z_score_50, percentile_volume,
                data_quality, candles_available
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18,
                $19, $20, $21, $22, $23, $24, $25, $26, $27,
                $28, $29, $30, $31, $32, $33
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
                daily_range_percent = EXCLUDED.daily_range_percent,
                weekly_range_percent = EXCLUDED.weekly_range_percent,
                z_score_50 = EXCLUDED.z_score_50,
                percentile_volume = EXCLUDED.percentile_volume,
                data_quality = EXCLUDED.data_quality,
                candles_available = EXCLUDED.candles_available,
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
            metrics.get('daily_range_percent'),
            metrics.get('weekly_range_percent'),
            metrics.get('z_score_50'),
            metrics.get('percentile_volume'),
            metrics.get('data_quality'),
            metrics.get('candles_available')
        )
