"""
Stock Data Fetcher

Fetches historical OHLCV data from yfinance and stores in PostgreSQL.
Also handles data synthesis (1h -> 4h, daily) and technical indicator calculation.

Data Strategy:
- Primary timeframe: 1H (730 days history from yfinance)
- Synthesized: 4H, Daily from 1H data
- Real-time quotes: RoboMarkets API
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import yfinance as yf

from ..trading.robomarkets_client import RoboMarketsClient, Instrument

logger = logging.getLogger(__name__)


class DataFetcherError(Exception):
    """Data fetcher error"""
    pass


class StockDataFetcher:
    """
    Fetches and manages stock market data

    Data sources:
    - yfinance: Historical OHLCV data
    - RoboMarkets: Real-time quotes, instrument list

    Usage:
        fetcher = StockDataFetcher(db_manager, robomarkets_client)
        await fetcher.sync_instruments()
        await fetcher.fetch_historical_data("AAPL", days=60)
        df = await fetcher.get_enhanced_data("AAPL", "1h", lookback=200)
    """

    # yfinance interval mappings
    YFINANCE_INTERVALS = {
        "1m": {"max_days": 7, "interval": "1m"},
        "5m": {"max_days": 60, "interval": "5m"},
        "15m": {"max_days": 60, "interval": "15m"},
        "30m": {"max_days": 60, "interval": "30m"},
        "1h": {"max_days": 730, "interval": "1h"},
        "1d": {"max_days": None, "interval": "1d"},  # Unlimited
    }

    def __init__(
        self,
        db_manager,
        robomarkets_client: RoboMarketsClient = None,
        thread_pool_size: int = 4
    ):
        """
        Initialize data fetcher

        Args:
            db_manager: Async database manager instance
            robomarkets_client: RoboMarkets API client
            thread_pool_size: Size of thread pool for yfinance calls
        """
        self.db = db_manager
        self.robomarkets = robomarkets_client
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._ticker_cache: Dict[str, yf.Ticker] = {}
        self._data_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl = 300  # 5 minutes

    # =========================================================================
    # INSTRUMENT SYNC
    # =========================================================================

    async def sync_instruments(self) -> int:
        """
        Sync available instruments from RoboMarkets to database

        Returns:
            Number of instruments synced
        """
        if not self.robomarkets:
            logger.warning("RoboMarkets client not configured, skipping sync")
            return 0

        try:
            instruments = await self.robomarkets.get_instruments(use_cache=False)
            count = 0

            for inst in instruments:
                await self._upsert_instrument(inst)
                count += 1

            logger.info(f"Synced {count} instruments from RoboMarkets")
            return count

        except Exception as e:
            logger.error(f"Failed to sync instruments: {e}")
            raise DataFetcherError(f"Instrument sync failed: {e}")

    async def _upsert_instrument(self, instrument: Instrument):
        """Insert or update instrument in database"""
        query = """
            INSERT INTO stock_instruments (
                ticker, name, contract_size, min_quantity, max_quantity,
                quantity_step, currency, exchange, is_tradeable,
                robomarkets_ticker, metadata, last_sync
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            ON CONFLICT (ticker) DO UPDATE SET
                name = EXCLUDED.name,
                contract_size = EXCLUDED.contract_size,
                min_quantity = EXCLUDED.min_quantity,
                max_quantity = EXCLUDED.max_quantity,
                quantity_step = EXCLUDED.quantity_step,
                currency = EXCLUDED.currency,
                exchange = EXCLUDED.exchange,
                is_tradeable = EXCLUDED.is_tradeable,
                metadata = EXCLUDED.metadata,
                last_sync = NOW(),
                updated_at = NOW()
        """
        import json
        await self.db.execute(
            query,
            instrument.ticker,
            instrument.name,
            instrument.contract_size,
            instrument.min_quantity,
            instrument.max_quantity,
            instrument.quantity_step,
            instrument.currency,
            instrument.exchange,
            instrument.is_tradeable,
            instrument.ticker,  # robomarkets_ticker same as ticker for now
            json.dumps(instrument.metadata) if instrument.metadata else None
        )

    async def get_tradeable_tickers(self) -> List[str]:
        """Get list of tradeable tickers from database"""
        query = """
            SELECT ticker FROM stock_instruments
            WHERE is_active = TRUE AND is_tradeable = TRUE
            ORDER BY ticker
        """
        rows = await self.db.fetch(query)
        return [row["ticker"] for row in rows]

    # =========================================================================
    # HISTORICAL DATA FETCHING (yfinance)
    # =========================================================================

    def _get_yf_ticker(self, ticker: str) -> yf.Ticker:
        """Get or create yfinance Ticker object"""
        if ticker not in self._ticker_cache:
            self._ticker_cache[ticker] = yf.Ticker(ticker)
        return self._ticker_cache[ticker]

    def _fetch_history_sync(
        self,
        ticker: str,
        period: str = None,
        interval: str = "1h",
        start: datetime = None,
        end: datetime = None
    ) -> pd.DataFrame:
        """
        Synchronous yfinance data fetch (runs in thread pool)

        Args:
            ticker: Stock ticker
            period: Period string (e.g., "60d", "2y")
            interval: Data interval ("1h", "1d", etc.)
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = self._get_yf_ticker(ticker)

            if start and end:
                df = stock.history(start=start, end=end, interval=interval)
            elif period:
                df = stock.history(period=period, interval=interval)
            else:
                df = stock.history(period="60d", interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Add ticker column
            df["ticker"] = ticker

            # Reset index to make datetime a column
            df = df.reset_index()
            df = df.rename(columns={"Date": "timestamp", "Datetime": "timestamp"})

            # Ensure timestamp is timezone-naive UTC
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    async def fetch_historical_data(
        self,
        ticker: str,
        days: int = 60,
        interval: str = "1h"
    ) -> int:
        """
        Fetch historical data from yfinance and store in database

        Args:
            ticker: Stock ticker symbol
            days: Number of days to fetch
            interval: Data interval (default: 1h)

        Returns:
            Number of candles stored
        """
        # Validate interval
        if interval not in self.YFINANCE_INTERVALS:
            raise DataFetcherError(f"Invalid interval: {interval}")

        interval_config = self.YFINANCE_INTERVALS[interval]
        max_days = interval_config["max_days"]

        if max_days and days > max_days:
            logger.warning(
                f"Requested {days} days but {interval} interval max is {max_days} days. "
                f"Limiting to {max_days} days."
            )
            days = max_days

        period = f"{days}d"

        # Fetch data in thread pool (yfinance is synchronous)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            self._executor,
            self._fetch_history_sync,
            ticker,
            period,
            interval,
            None,
            None
        )

        if df.empty:
            logger.warning(f"No data fetched for {ticker}")
            return 0

        # Store in database
        count = await self._store_candles(ticker, interval, df)

        # Log sync
        await self._log_sync(ticker, interval, "full", count, len(df))

        logger.info(f"Fetched {count} {interval} candles for {ticker}")
        return count

    async def fetch_incremental_data(
        self,
        ticker: str,
        interval: str = "1h"
    ) -> int:
        """
        Fetch only new data since last fetch

        Args:
            ticker: Stock ticker symbol
            interval: Data interval

        Returns:
            Number of new candles stored
        """
        # Get last candle timestamp
        query = """
            SELECT MAX(timestamp) as last_ts
            FROM stock_candles
            WHERE ticker = $1 AND timeframe = $2
        """
        result = await self.db.fetchrow(query, ticker, interval)
        last_ts = result["last_ts"] if result else None

        if not last_ts:
            # No existing data, do full fetch
            return await self.fetch_historical_data(ticker, days=60, interval=interval)

        # Fetch from last timestamp to now
        start = last_ts - timedelta(hours=1)  # Overlap by 1 hour for safety
        end = datetime.utcnow()

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            self._executor,
            self._fetch_history_sync,
            ticker,
            None,
            interval,
            start,
            end
        )

        if df.empty:
            return 0

        # Filter out already existing data
        df = df[df["timestamp"] > last_ts]

        if df.empty:
            return 0

        count = await self._store_candles(ticker, interval, df)

        await self._log_sync(ticker, interval, "incremental", count, len(df))

        logger.info(f"Fetched {count} new {interval} candles for {ticker}")
        return count

    async def _store_candles(
        self,
        ticker: str,
        timeframe: str,
        df: pd.DataFrame
    ) -> int:
        """Store candles in database with upsert"""
        if df.empty:
            return 0

        query = """
            INSERT INTO stock_candles (
                ticker, timeframe, timestamp, open, high, low, close, volume
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ticker, timeframe, timestamp) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        count = 0
        for _, row in df.iterrows():
            try:
                await self.db.execute(
                    query,
                    ticker,
                    timeframe,
                    row["timestamp"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row.get("volume", 0))
                )
                count += 1
            except Exception as e:
                logger.error(f"Error storing candle: {e}")

        return count

    async def _log_sync(
        self,
        ticker: str,
        timeframe: str,
        sync_type: str,
        inserted: int,
        fetched: int
    ):
        """Log data sync to database"""
        query = """
            INSERT INTO stock_data_sync_log (
                ticker, timeframe, sync_type, candles_fetched,
                candles_inserted, status
            ) VALUES ($1, $2, $3, $4, $5, 'success')
        """
        try:
            await self.db.execute(query, ticker, timeframe, sync_type, fetched, inserted)
        except Exception as e:
            logger.warning(f"Failed to log sync: {e}")

    # =========================================================================
    # DATA RETRIEVAL
    # =========================================================================

    async def get_candles(
        self,
        ticker: str,
        timeframe: str = "1h",
        limit: int = 200,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        Get candles from database

        Args:
            ticker: Stock ticker
            timeframe: Timeframe (1h, 4h, 1d)
            limit: Maximum candles to return
            start_date: Optional start filter
            end_date: Optional end filter

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles
            WHERE ticker = $1 AND timeframe = $2
        """
        params = [ticker, timeframe]

        if start_date:
            query += f" AND timestamp >= ${len(params) + 1}"
            params.append(start_date)

        if end_date:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(end_date)

        query += f" ORDER BY timestamp DESC LIMIT ${len(params) + 1}"
        params.append(limit)

        rows = await self.db.fetch(query, *params)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    async def get_enhanced_data(
        self,
        ticker: str,
        timeframe: str = "1h",
        lookback: int = 200,
        calculate_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Get candles with technical indicators calculated

        Args:
            ticker: Stock ticker
            timeframe: Timeframe
            lookback: Number of candles
            calculate_indicators: Whether to add indicators

        Returns:
            DataFrame with OHLCV and indicators
        """
        # Check cache
        cache_key = f"{ticker}_{timeframe}_{lookback}"
        if cache_key in self._data_cache:
            df, timestamp = self._data_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self._cache_ttl:
                return df.copy()

        # Get base candles
        df = await self.get_candles(ticker, timeframe, lookback)

        if df.empty:
            return df

        if calculate_indicators:
            df = self._calculate_indicators(df)

        # Update cache
        self._data_cache[cache_key] = (df.copy(), datetime.now())

        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators on DataFrame

        Adds:
        - EMAs (12, 21, 50, 200)
        - MACD
        - RSI
        - Bollinger Bands
        - ATR
        - Volume metrics
        """
        if df.empty or len(df) < 50:
            return df

        # EMAs
        for period in [12, 21, 50, 200]:
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
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        if len(df) >= 20:
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (std * 2)
            df["bb_lower"] = df["bb_middle"] - (std * 2)

        # ATR
        if len(df) >= 14:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = tr.rolling(window=14).mean()

        # Volume metrics
        if "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["relative_volume"] = df["volume"] / df["volume_sma"]

        return df

    # =========================================================================
    # CANDLE SYNTHESIS (1H -> 4H, Daily)
    # =========================================================================

    async def synthesize_candles(
        self,
        ticker: str,
        target_timeframe: str = "4h"
    ) -> int:
        """
        Synthesize higher timeframe candles from 1H data

        Args:
            ticker: Stock ticker
            target_timeframe: Target timeframe (4h or 1d)

        Returns:
            Number of candles synthesized
        """
        if target_timeframe not in ("4h", "1d"):
            raise DataFetcherError(f"Invalid target timeframe: {target_timeframe}")

        # Use database function for synthesis
        query = "SELECT synthesize_candles($1, $2)"

        try:
            result = await self.db.fetchval(query, ticker, target_timeframe)
            logger.info(f"Synthesized {result} {target_timeframe} candles for {ticker}")
            return result or 0
        except Exception as e:
            logger.error(f"Failed to synthesize candles: {e}")
            return 0

    async def synthesize_all_tickers(self, target_timeframe: str = "4h") -> Dict[str, int]:
        """
        Synthesize candles for all tickers in database

        Returns:
            Dict mapping ticker to number of candles synthesized
        """
        tickers = await self.get_tradeable_tickers()
        results = {}

        for ticker in tickers:
            count = await self.synthesize_candles(ticker, target_timeframe)
            results[ticker] = count

        return results

    # =========================================================================
    # BATCH OPERATIONS
    # =========================================================================

    async def fetch_all_tickers(
        self,
        days: int = 60,
        interval: str = "1h",
        concurrency: int = 5
    ) -> Dict[str, int]:
        """
        Fetch historical data for all tradeable tickers

        Args:
            days: Days of history
            interval: Data interval
            concurrency: Max concurrent fetches

        Returns:
            Dict mapping ticker to candle count
        """
        tickers = await self.get_tradeable_tickers()

        if not tickers:
            logger.warning("No tradeable tickers found")
            return {}

        results = {}
        semaphore = asyncio.Semaphore(concurrency)

        async def fetch_with_semaphore(ticker: str):
            async with semaphore:
                try:
                    count = await self.fetch_historical_data(ticker, days, interval)
                    return ticker, count
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                    return ticker, 0

        tasks = [fetch_with_semaphore(t) for t in tickers]
        completed = await asyncio.gather(*tasks)

        for ticker, count in completed:
            results[ticker] = count

        total = sum(results.values())
        logger.info(f"Fetched {total} total candles for {len(tickers)} tickers")

        return results

    async def update_all_tickers(self, concurrency: int = 5) -> Dict[str, int]:
        """
        Incremental update for all tickers

        Returns:
            Dict mapping ticker to new candle count
        """
        tickers = await self.get_tradeable_tickers()
        results = {}
        semaphore = asyncio.Semaphore(concurrency)

        async def update_with_semaphore(ticker: str):
            async with semaphore:
                try:
                    count = await self.fetch_incremental_data(ticker)
                    return ticker, count
                except Exception as e:
                    logger.error(f"Error updating {ticker}: {e}")
                    return ticker, 0

        tasks = [update_with_semaphore(t) for t in tickers]
        completed = await asyncio.gather(*tasks)

        for ticker, count in completed:
            results[ticker] = count

        return results

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get latest close price from database"""
        query = """
            SELECT close FROM stock_candles
            WHERE ticker = $1 AND timeframe = '1h'
            ORDER BY timestamp DESC LIMIT 1
        """
        result = await self.db.fetchval(query, ticker)
        return float(result) if result else None

    async def get_data_coverage(self, ticker: str) -> Dict:
        """Get data coverage info for a ticker"""
        query = """
            SELECT
                timeframe,
                MIN(timestamp) as first_candle,
                MAX(timestamp) as last_candle,
                COUNT(*) as candle_count
            FROM stock_candles
            WHERE ticker = $1
            GROUP BY timeframe
        """
        rows = await self.db.fetch(query, ticker)

        return {
            row["timeframe"]: {
                "first": row["first_candle"],
                "last": row["last_candle"],
                "count": row["candle_count"]
            }
            for row in rows
        }

    def clear_cache(self):
        """Clear data cache"""
        self._data_cache.clear()
        self._ticker_cache.clear()

    async def close(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
        self.clear_cache()
