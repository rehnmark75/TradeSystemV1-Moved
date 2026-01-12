"""
Technical Deep Analyzer

Performs deep technical analysis including:
- Multi-Timeframe (MTF) confluence analysis
- Volume profile analysis
- SMC (Smart Money Concepts) enhancement
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

from .models import (
    TechnicalDeepResult,
    MTFAnalysisResult,
    VolumeAnalysisResult,
    SMCAnalysisResult,
    TimeframeAnalysis,
    TrendDirection,
    DeepAnalysisConfig,
)

logger = logging.getLogger(__name__)


class TechnicalDeepAnalyzer:
    """
    Performs deep technical analysis on stock signals.

    Components:
    1. Multi-Timeframe Confluence (20% of DAQ)
    2. Volume Profile Analysis (10% of DAQ)
    3. SMC Enhancement (15% of DAQ)
    """

    def __init__(self, db_manager, config: DeepAnalysisConfig = None):
        """
        Initialize technical analyzer.

        Args:
            db_manager: Database manager for fetching candle data
            config: Deep analysis configuration
        """
        self.db = db_manager
        self.config = config or DeepAnalysisConfig()

    async def analyze(
        self,
        ticker: str,
        signal: Dict[str, Any],
        candles_by_tf: Optional[Dict[str, pd.DataFrame]] = None
    ) -> TechnicalDeepResult:
        """
        Perform complete technical deep analysis.

        Args:
            ticker: Stock ticker
            signal: Signal data including direction, entry price, etc.
            candles_by_tf: Pre-fetched candles by timeframe (optional)

        Returns:
            TechnicalDeepResult with all component scores
        """
        # Determine signal direction
        signal_direction = self._get_signal_direction(signal)

        # Fetch candles if not provided
        if candles_by_tf is None:
            candles_by_tf = await self._fetch_candles(ticker)

        # Run analysis components
        mtf_result = await self._analyze_mtf(ticker, signal_direction, candles_by_tf)
        volume_result = await self._analyze_volume(ticker, candles_by_tf)
        smc_result = await self._analyze_smc(ticker, signal_direction)

        return TechnicalDeepResult(
            mtf=mtf_result,
            volume=volume_result,
            smc=smc_result
        )

    def _get_signal_direction(self, signal: Dict[str, Any]) -> TrendDirection:
        """Determine signal direction from signal data"""
        signal_type = signal.get('signal_type', '').upper()
        if signal_type in ('BUY', 'LONG'):
            return TrendDirection.BULLISH
        elif signal_type in ('SELL', 'SHORT'):
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    async def _fetch_candles(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Fetch candles for all timeframes"""
        candles = {}
        for tf in self.config.mtf_timeframes:
            candles[tf] = await self._get_candles_with_indicators(ticker, tf)
        return candles

    async def _get_candles_with_indicators(
        self,
        ticker: str,
        timeframe: str,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Fetch candles from database with technical indicators.

        Data sources:
        - 4h: Resampled from 1h candles (stock_candles)
        - 1d: Synthesized daily candles (stock_candles_synthesized)
        - 1w: Synthesized weekly candles (stock_candles_synthesized)

        Args:
            ticker: Stock ticker
            timeframe: Timeframe (4h, 1d, 1w)
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV and indicators
        """
        if timeframe == '4h':
            # Resample 1h to 4h on-the-fly
            df = await self._get_4h_from_1h(ticker, limit)
        elif timeframe in ('1d', '1w'):
            # Fetch from synthesized table
            df = await self._get_synthesized_candles(ticker, timeframe, limit)
        else:
            # Fallback to raw candles table
            df = await self._get_raw_candles(ticker, timeframe, limit)

        if df.empty:
            return df

        # Calculate indicators
        df = self._calculate_indicators(df)

        return df

    async def _get_raw_candles(
        self,
        ticker: str,
        timeframe: str,
        limit: int = 200
    ) -> pd.DataFrame:
        """Fetch raw candles from stock_candles table."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles
            WHERE ticker = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await self.db.fetch(query, ticker, timeframe, limit)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    async def _get_synthesized_candles(
        self,
        ticker: str,
        timeframe: str,
        limit: int = 200
    ) -> pd.DataFrame:
        """Fetch synthesized candles (1d, 1w) from stock_candles_synthesized."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles_synthesized
            WHERE ticker = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await self.db.fetch(query, ticker, timeframe, limit)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    async def _get_4h_from_1h(
        self,
        ticker: str,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Resample 1h candles to 4h on-the-fly.

        Groups by 4-hour blocks aligned to market hours:
        - Block 1: 09:30-13:00 (9:30, 10:30, 11:30, 12:30)
        - Block 2: 13:00-16:00 (13:30, 14:30, 15:30)
        """
        # Fetch enough 1h candles to produce the requested 4h candles
        # Need 4x as many 1h candles, plus buffer for incomplete blocks
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles
            WHERE ticker = $1 AND timeframe = '1h'
            ORDER BY timestamp DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, ticker, limit * 4 + 20)

        if not rows or len(rows) < 4:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values("timestamp").reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Resample to 4H using pandas
        # Use 4H offset, label='right' to get end of period
        df = df.set_index('timestamp')

        resampled = df.resample('4h', label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        resampled = resampled.reset_index()

        # Limit to requested number of candles
        if len(resampled) > limit:
            resampled = resampled.tail(limit).reset_index(drop=True)

        return resampled

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on DataFrame"""
        if df.empty or len(df) < 50:
            return df

        # EMAs
        for period in [20, 50, 200]:
            if len(df) >= period:
                df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD
        if len(df) >= 26:
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI
        if len(df) >= 14:
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.0001)
            df["rsi"] = 100 - (100 / (1 + rs))

        # Volume metrics
        if "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["relative_volume"] = df["volume"] / df["volume_sma"].replace(0, 1)

        return df

    # =========================================================================
    # MULTI-TIMEFRAME ANALYSIS (20% of DAQ)
    # =========================================================================

    async def _analyze_mtf(
        self,
        ticker: str,
        signal_direction: TrendDirection,
        candles_by_tf: Dict[str, pd.DataFrame]
    ) -> MTFAnalysisResult:
        """
        Analyze multi-timeframe confluence.

        Checks if signal direction aligns across 1h, 4h, and daily timeframes.

        Scoring:
        - 3/3 TFs aligned: 100
        - 2/3 TFs aligned: 70
        - 1/3 TFs aligned: 40
        - 0/3 TFs aligned: 10
        """
        timeframes_analysis = {}
        confluence_count = 0

        for tf in self.config.mtf_timeframes:
            df = candles_by_tf.get(tf, pd.DataFrame())

            if df.empty or len(df) < 50:
                # Insufficient data - neutral
                tf_analysis = TimeframeAnalysis(
                    timeframe=tf,
                    trend=TrendDirection.NEUTRAL,
                    ema_aligned=False,
                    macd_bullish=False
                )
            else:
                tf_analysis = self._analyze_single_timeframe(df, tf, signal_direction)

            timeframes_analysis[tf] = tf_analysis

            # Check if this TF aligns with signal direction
            if tf_analysis.trend == signal_direction:
                confluence_count += 1

        # Calculate score
        total_tfs = len(self.config.mtf_timeframes)
        if confluence_count == total_tfs:
            score = 100
        elif confluence_count == total_tfs - 1:
            score = 70
        elif confluence_count == total_tfs - 2:
            score = 40
        else:
            score = 10

        # Build details for database
        details = {
            tf: {
                'trend': analysis.trend.value,
                'ema_aligned': analysis.ema_aligned,
                'macd_bullish': analysis.macd_bullish,
                'rsi': analysis.rsi_level,
                'volume_confirm': analysis.volume_confirm,
            }
            for tf, analysis in timeframes_analysis.items()
        }
        details['confluence'] = f"{confluence_count}/{total_tfs}"
        details['signal_direction'] = signal_direction.value

        return MTFAnalysisResult(
            score=score,
            timeframes=timeframes_analysis,
            confluence_count=confluence_count,
            total_timeframes=total_tfs,
            signal_direction=signal_direction,
            details=details
        )

    def _analyze_single_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        signal_direction: TrendDirection
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe for trend alignment"""
        if df.empty:
            return TimeframeAnalysis(
                timeframe=timeframe,
                trend=TrendDirection.NEUTRAL,
                ema_aligned=False,
                macd_bullish=False
            )

        latest = df.iloc[-1]
        close = latest['close']

        # EMA alignment check
        ema_20 = latest.get('ema_20', close)
        ema_50 = latest.get('ema_50', close)
        ema_200 = latest.get('ema_200', close)

        # Bullish: price > 20 EMA > 50 EMA > 200 EMA
        # Bearish: price < 20 EMA < 50 EMA < 200 EMA
        bullish_ema = (close > ema_20 > ema_50 > ema_200) if all([ema_20, ema_50, ema_200]) else False
        bearish_ema = (close < ema_20 < ema_50 < ema_200) if all([ema_20, ema_50, ema_200]) else False

        # MACD check
        macd_hist = latest.get('macd_histogram', 0)
        macd_bullish = macd_hist > 0 if macd_hist else False

        # RSI check
        rsi = latest.get('rsi')

        # Volume confirmation
        rel_vol = latest.get('relative_volume', 1.0)
        volume_confirm = rel_vol >= 1.2 if rel_vol else False

        # Determine trend
        if bullish_ema and macd_bullish:
            trend = TrendDirection.BULLISH
        elif bearish_ema and not macd_bullish:
            trend = TrendDirection.BEARISH
        else:
            # Mixed signals - check individual components
            bullish_signals = sum([
                close > ema_50 if ema_50 else False,
                macd_bullish,
                rsi > 50 if rsi else False
            ])
            if bullish_signals >= 2:
                trend = TrendDirection.BULLISH
            elif bullish_signals == 0:
                trend = TrendDirection.BEARISH
            else:
                trend = TrendDirection.NEUTRAL

        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            ema_aligned=bullish_ema if signal_direction == TrendDirection.BULLISH else bearish_ema,
            macd_bullish=macd_bullish,
            rsi_level=rsi,
            volume_confirm=volume_confirm,
            details={
                'close': close,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_200': ema_200,
                'macd_histogram': macd_hist,
                'relative_volume': rel_vol,
            }
        )

    # =========================================================================
    # VOLUME PROFILE ANALYSIS (10% of DAQ)
    # =========================================================================

    async def _analyze_volume(
        self,
        ticker: str,
        candles_by_tf: Dict[str, pd.DataFrame]
    ) -> VolumeAnalysisResult:
        """
        Analyze volume profile for accumulation/distribution patterns.

        Scoring:
        - High relative volume (>1.5x) + accumulation: 100
        - Moderate volume (1.2-1.5x) + accumulation: 80
        - High relative volume + distribution: 60
        - Normal volume: 50
        - Low volume (<0.8x): 30
        - Very low volume (<0.5x): 10
        """
        # Use daily candles for volume analysis
        df = candles_by_tf.get('1d', pd.DataFrame())

        if df.empty or len(df) < 20:
            return VolumeAnalysisResult(
                score=50,
                relative_volume=1.0,
                is_accumulation=False,
                is_distribution=False,
                unusual_volume=False,
                volume_trend='stable',
                details={'error': 'Insufficient daily data'}
            )

        latest = df.iloc[-1]
        rel_vol = latest.get('relative_volume', 1.0)

        # Calculate accumulation/distribution
        # Accumulation: more volume on up days than down days (last 20 days)
        recent_df = df.tail(20).copy()
        recent_df['change'] = recent_df['close'].diff()
        recent_df['up_volume'] = recent_df.apply(
            lambda x: x['volume'] if x['change'] > 0 else 0, axis=1
        )
        recent_df['down_volume'] = recent_df.apply(
            lambda x: x['volume'] if x['change'] < 0 else 0, axis=1
        )

        total_up_vol = recent_df['up_volume'].sum()
        total_down_vol = recent_df['down_volume'].sum()

        is_accumulation = total_up_vol > total_down_vol * 1.2  # 20% more volume on up days
        is_distribution = total_down_vol > total_up_vol * 1.2

        # Volume trend (increasing/decreasing)
        vol_5d = df.tail(5)['volume'].mean() if len(df) >= 5 else 0
        vol_20d = df.tail(20)['volume'].mean() if len(df) >= 20 else vol_5d
        if vol_5d > vol_20d * 1.3:
            volume_trend = 'increasing'
        elif vol_5d < vol_20d * 0.7:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'

        unusual_volume = rel_vol >= 2.0

        # Calculate score
        if rel_vol >= 1.5 and is_accumulation:
            score = 100
        elif rel_vol >= 1.2 and is_accumulation:
            score = 80
        elif rel_vol >= 1.5:  # High volume but distribution or neutral
            score = 60
        elif rel_vol >= 0.8:  # Normal volume
            score = 50
        elif rel_vol >= 0.5:  # Low volume
            score = 30
        else:  # Very low volume
            score = 10

        return VolumeAnalysisResult(
            score=score,
            relative_volume=float(rel_vol) if rel_vol else 1.0,
            is_accumulation=is_accumulation,
            is_distribution=is_distribution,
            unusual_volume=unusual_volume,
            volume_trend=volume_trend,
            details={
                'up_volume_20d': float(total_up_vol),
                'down_volume_20d': float(total_down_vol),
                'volume_5d_avg': float(vol_5d),
                'volume_20d_avg': float(vol_20d),
            }
        )

    # =========================================================================
    # SMC (SMART MONEY CONCEPTS) ANALYSIS (15% of DAQ)
    # =========================================================================

    async def _analyze_smc(
        self,
        ticker: str,
        signal_direction: TrendDirection
    ) -> SMCAnalysisResult:
        """
        Enhance signal with Smart Money Concepts data from stock_screening_metrics.

        Scoring:
        - SMC trend aligned + in discount zone + near OB: 100
        - SMC trend aligned + near OB: 80
        - SMC trend aligned: 70
        - SMC neutral: 50
        - SMC trend against signal: 30
        """
        # Fetch SMC data from stock_screening_metrics
        query = """
            SELECT
                smc_trend,
                smc_bias,
                last_bos_type,
                last_bos_date,
                last_bos_price,
                last_choch_type,
                last_choch_date,
                swing_high,
                swing_low,
                swing_high_date,
                swing_low_date,
                premium_discount_zone,
                zone_position,
                weekly_range_high,
                weekly_range_low,
                nearest_ob_type,
                nearest_ob_price,
                nearest_ob_distance,
                smc_confluence_score
            FROM stock_screening_metrics
            WHERE ticker = $1
              AND smc_trend IS NOT NULL
            ORDER BY calculation_date DESC
            LIMIT 1
        """
        row = await self.db.fetchrow(query, ticker)

        if not row:
            return SMCAnalysisResult(
                score=50,
                details={'error': 'No SMC data available'}
            )

        smc_data = dict(row)

        # Extract key fields
        smc_trend = smc_data.get('smc_trend', 'neutral')
        smc_bias = smc_data.get('smc_bias', 'neutral')
        premium_discount = smc_data.get('premium_discount_zone', 'equilibrium')
        zone_position = smc_data.get('zone_position', 0.5)
        nearest_ob_type = smc_data.get('nearest_ob_type')
        nearest_ob_distance = smc_data.get('nearest_ob_distance', 100)
        smc_confluence = smc_data.get('smc_confluence_score', 50)

        # Check trend alignment
        signal_bullish = signal_direction == TrendDirection.BULLISH
        smc_bullish = smc_trend in ('bullish', 'strong_bullish')
        smc_bearish = smc_trend in ('bearish', 'strong_bearish')

        trend_aligned = (signal_bullish and smc_bullish) or (not signal_bullish and smc_bearish)
        trend_against = (signal_bullish and smc_bearish) or (not signal_bullish and smc_bullish)

        # Check zone (bullish signals best in discount, bearish in premium)
        in_favorable_zone = (
            (signal_bullish and premium_discount == 'discount') or
            (not signal_bullish and premium_discount == 'premium')
        )

        # Check proximity to order block
        near_ob = nearest_ob_distance is not None and nearest_ob_distance < 2.0  # Within 2%

        # Calculate score
        if trend_aligned and in_favorable_zone and near_ob:
            score = 100
        elif trend_aligned and near_ob:
            score = 85
        elif trend_aligned and in_favorable_zone:
            score = 80
        elif trend_aligned:
            score = 70
        elif not trend_against:  # Neutral
            score = 50
        else:  # Trend against
            score = 30

        return SMCAnalysisResult(
            score=score,
            smc_trend=smc_trend,
            smc_bias=smc_bias,
            last_bos_type=smc_data.get('last_bos_type'),
            last_bos_date=smc_data.get('last_bos_date'),
            last_choch_type=smc_data.get('last_choch_type'),
            nearest_ob_type=nearest_ob_type,
            nearest_ob_distance=float(nearest_ob_distance) if nearest_ob_distance else None,
            premium_discount_zone=premium_discount,
            zone_position=float(zone_position) if zone_position else None,
            confluence_score=smc_confluence,
            details={
                'trend_aligned': trend_aligned,
                'in_favorable_zone': in_favorable_zone,
                'near_ob': near_ob,
                'swing_high': smc_data.get('swing_high'),
                'swing_low': smc_data.get('swing_low'),
                'weekly_range_high': smc_data.get('weekly_range_high'),
                'weekly_range_low': smc_data.get('weekly_range_low'),
            }
        )
