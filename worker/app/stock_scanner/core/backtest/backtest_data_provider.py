"""
Backtest Data Provider

Retrieves historical stock data with technical indicators for backtesting.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..database.async_database_manager import AsyncDatabaseManager


class BacktestDataProvider:
    """
    Provides historical stock data with technical indicators for backtesting.

    Features:
    - Retrieves data from stock_candles_synthesized table (daily data)
    - Calculates EMAs (20, 50, 100, 200), RSI, ATR
    - Caches data per ticker for performance
    - Includes warmup period for indicator calculation
    """

    # Warmup period in days (for 200 EMA calculation)
    # Reduced to 220 to accommodate smaller datasets while still allowing 200 EMA to stabilize
    WARMUP_DAYS = 220

    # Minimum bars required for valid analysis
    MIN_BARS_REQUIRED = 50

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    async def get_tradeable_tickers(self, sector: Optional[str] = None) -> List[str]:
        """
        Get list of tradeable tickers from stock_instruments table.

        Args:
            sector: Optional sector filter (comma-separated for multiple)

        Returns:
            List of ticker symbols
        """
        if sector:
            sectors = [s.strip() for s in sector.split(',')]
            query = """
                SELECT ticker FROM stock_instruments
                WHERE is_active = TRUE AND is_tradeable = TRUE
                  AND sector = ANY($1)
                ORDER BY ticker
            """
            rows = await self.db.fetch(query, sectors)
        else:
            query = """
                SELECT ticker FROM stock_instruments
                WHERE is_active = TRUE AND is_tradeable = TRUE
                ORDER BY ticker
            """
            rows = await self.db.fetch(query)

        return [row['ticker'] for row in rows]

    async def get_historical_data(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        timeframe: str = '1d',
        include_warmup: bool = True
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data with indicators for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for backtest period
            end_date: End date for backtest period
            timeframe: Candle timeframe ('1d', '4h', '1h')
            include_warmup: Whether to include extra days for indicator warmup

        Returns:
            DataFrame with OHLCV + indicators, or empty DataFrame if no data
        """
        # Check cache first
        cache_key = f"{ticker}_{timeframe}_{end_date}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key].copy()

        # Fetch ALL available data up to end_date
        # This ensures we get maximum warmup for indicators
        # We'll filter to start_date later in the orchestrator
        df = await self._fetch_all_candles(ticker, end_date, timeframe)

        if df.empty:
            self.logger.warning(f"No data found for {ticker} up to {end_date}")
            return df

        # Calculate indicators
        df = self._calculate_indicators(df)

        # Cache the full data
        self._data_cache[cache_key] = df.copy()
        self._cache_timestamps[cache_key] = datetime.now()

        return df

    async def get_data_at_timestamp(
        self,
        ticker: str,
        timestamp: datetime,
        lookback_bars: int = 300
    ) -> pd.DataFrame:
        """
        Get data up to a specific timestamp for point-in-time backtesting.
        Ensures no future data leakage.

        Args:
            ticker: Stock ticker symbol
            timestamp: The timestamp to look back from
            lookback_bars: Number of bars to include

        Returns:
            DataFrame with data UP TO (and including) the timestamp
        """
        query = """
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM stock_candles_synthesized
            WHERE ticker = $1
              AND timeframe = '1d'
              AND timestamp <= $2
            ORDER BY timestamp DESC
            LIMIT $3
        """

        rows = await self.db.fetch(query, ticker, timestamp, lookback_bars)

        if not rows:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate indicators
        df = self._calculate_indicators(df)

        return df

    async def get_future_data(
        self,
        ticker: str,
        from_timestamp: datetime,
        bars: int = 30
    ) -> pd.DataFrame:
        """
        Get future data from a timestamp for trade simulation.
        Used to determine trade outcome after signal.

        Args:
            ticker: Stock ticker symbol
            from_timestamp: The timestamp to look forward from
            bars: Number of future bars to fetch

        Returns:
            DataFrame with future data (excluding the from_timestamp)
        """
        query = """
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM stock_candles_synthesized
            WHERE ticker = $1
              AND timeframe = '1d'
              AND timestamp > $2
            ORDER BY timestamp ASC
            LIMIT $3
        """

        rows = await self.db.fetch(query, ticker, from_timestamp, bars)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    async def _fetch_all_candles(
        self,
        ticker: str,
        end_date: date,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch ALL available candle data up to end_date."""

        # Use synthesized table for daily data, regular for hourly
        if timeframe == '1d':
            table = 'stock_candles_synthesized'
        else:
            table = 'stock_candles'

        query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {table}
            WHERE ticker = $1
              AND timeframe = $2
              AND timestamp <= $3
            ORDER BY timestamp ASC
        """

        rows = await self.db.fetch(
            query,
            ticker,
            timeframe,
            datetime.combine(end_date, datetime.max.time())
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        return df

    async def _fetch_candles(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        timeframe: str
    ) -> pd.DataFrame:
        """Fetch raw candle data from database within a date range."""

        # Use synthesized table for daily data, regular for hourly
        if timeframe == '1d':
            table = 'stock_candles_synthesized'
        else:
            table = 'stock_candles'

        query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {table}
            WHERE ticker = $1
              AND timeframe = $2
              AND timestamp >= $3
              AND timestamp <= $4
            ORDER BY timestamp ASC
        """

        rows = await self.db.fetch(
            query,
            ticker,
            timeframe,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time())
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators on OHLCV data.

        Indicators calculated:
        - EMA 20, 50, 100, 200
        - RSI (14 period)
        - ATR (14 period)
        - Pullback percent from EMA 20
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # RSI (14 period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR (14 period)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Percent distance from EMA 20
        df['pct_from_ema20'] = ((df['close'] - df['ema_20']) / df['ema_20']) * 100

        # Additional helper columns
        df['above_ema_200'] = df['close'] > df['ema_200']
        df['above_ema_100'] = df['close'] > df['ema_100']
        df['above_ema_50'] = df['close'] > df['ema_50']
        df['above_ema_20'] = df['close'] > df['ema_20']

        # Previous bar's EMA 20 position (for crossover detection)
        df['prev_above_ema20'] = df['above_ema_20'].shift(1)

        # Volume moving average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma_20'].replace(0, np.nan)

        # MACD (12, 26, 9) - Standard settings
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # ADX (Average Directional Index) - 14 period
        # Calculate directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        # +DM and -DM
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Smoothed +DI and -DI (using ATR already calculated)
        atr_safe = df['atr'].replace(0, np.nan)
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_safe)
        minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_safe)

        # DX and ADX
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, 1)  # Avoid division by zero
        dx = 100 * abs(plus_di - minus_di) / di_sum
        df['adx'] = dx.ewm(span=14, adjust=False).mean()

        # ZLMA (Zero-Lag Moving Average) - period 15 (default)
        # Formula: ZLMA = EMA(close + (close - EMA(close)), period)
        # This reduces lag by compensating for the EMA's inherent delay
        zlma_period = 15
        ema_for_zlma = df['close'].ewm(span=zlma_period, adjust=False).mean()
        zlma_correction = df['close'] + (df['close'] - ema_for_zlma)
        df['zlma'] = zlma_correction.ewm(span=zlma_period, adjust=False).mean()
        df['ema_15'] = ema_for_zlma  # Keep the base EMA for crossover detection

        # ZLMA crossover detection
        df['zlma_above_ema'] = df['zlma'] > df['ema_15']
        df['prev_zlma_above_ema'] = df['zlma_above_ema'].shift(1)

        # ZLMA-EMA separation (for signal strength)
        df['zlma_ema_diff'] = df['zlma'] - df['ema_15']
        df['zlma_ema_diff_pct'] = (df['zlma_ema_diff'] / df['ema_15']) * 100

        return df

    async def get_ticker_sector(self, ticker: str) -> Optional[str]:
        """Get the sector for a ticker."""
        query = """
            SELECT sector FROM stock_instruments
            WHERE ticker = $1
        """
        row = await self.db.fetchrow(query, ticker)
        return row['sector'] if row else None

    async def get_data_quality(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        timeframe: str = '1d'
    ) -> Dict:
        """
        Check data quality and completeness for a ticker.

        Returns:
            Dict with quality metrics
        """
        query = """
            SELECT
                COUNT(*) as total_bars,
                MIN(timestamp) as first_bar,
                MAX(timestamp) as last_bar,
                COUNT(*) FILTER (WHERE volume = 0) as zero_volume_bars,
                COUNT(*) FILTER (WHERE close IS NULL) as null_close_bars
            FROM stock_candles_synthesized
            WHERE ticker = $1
              AND timeframe = $2
              AND timestamp >= $3
              AND timestamp <= $4
        """

        row = await self.db.fetchrow(
            query,
            ticker,
            timeframe,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time())
        )

        if not row:
            return {
                'ticker': ticker,
                'has_data': False,
                'total_bars': 0,
                'quality_score': 0
            }

        # Calculate expected trading days (rough estimate: ~252 per year)
        total_days = (end_date - start_date).days
        expected_bars = int(total_days * (252 / 365))

        completeness = min(row['total_bars'] / max(expected_bars, 1), 1.0)
        zero_vol_pct = row['zero_volume_bars'] / max(row['total_bars'], 1)

        quality_score = completeness * (1 - zero_vol_pct)

        return {
            'ticker': ticker,
            'has_data': row['total_bars'] > 0,
            'total_bars': row['total_bars'],
            'first_bar': row['first_bar'],
            'last_bar': row['last_bar'],
            'expected_bars': expected_bars,
            'completeness': round(completeness, 4),
            'zero_volume_bars': row['zero_volume_bars'],
            'quality_score': round(quality_score, 4)
        }

    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Data cache cleared")
