"""
Watchlist Scanner - 5 Predefined Technical Screens

Runs daily scans for 5 predefined technical setups:
1. EMA 50 Crossover - Price > EMA 200, Price crosses above EMA 50, Volume > 1M/day
2. EMA 20 Crossover - Price > EMA 200, Price crosses above EMA 20, Volume > 1M/day
3. MACD Bullish Cross - MACD crosses from negative to positive, Price > EMA 200, Volume > 1M/day
4. Gap Up Continuation - Gap up > 2% today, Price above VWAP, Price > EMA 200, Volume > 1M/day
5. RSI Oversold Bounce - RSI(14) < 30, Price > EMA 200, Bullish candle pattern, Volume > 1M/day

Crossover Tracking:
- EMA crossovers and MACD crossovers track the original crossover_date
- Each day checks if price is still above the crossed EMA (or MACD still positive)
- If condition fails, entry is marked as 'expired'
- Entries older than 30 days are removed (no longer fresh crossover)
- Gap and RSI are single-day events, no tracking needed

Results are stored in stock_watchlist_results table for Streamlit display.
"""

import logging
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ..core.backtest.backtest_data_provider import BacktestDataProvider

logger = logging.getLogger(__name__)

# Watchlists that track crossover dates (vs single-day events)
CROSSOVER_WATCHLISTS = {'ema_50_crossover', 'ema_20_crossover', 'macd_bullish_cross'}
MAX_CROSSOVER_AGE_DAYS = 30


# Watchlist definitions
WATCHLIST_DEFINITIONS = {
    'ema_50_crossover': {
        'name': 'EMA 50 Crossover',
        'description': 'Price > EMA 200, Price crosses above EMA 50, Volume > 1M/day',
    },
    'ema_20_crossover': {
        'name': 'EMA 20 Crossover',
        'description': 'Price > EMA 200, Price crosses above EMA 20, Volume > 1M/day',
    },
    'macd_bullish_cross': {
        'name': 'MACD Bullish Cross',
        'description': 'MACD crosses from negative to positive, Price > EMA 200, Volume > 1M/day',
    },
    'gap_up_continuation': {
        'name': 'Gap Up Continuation',
        'description': 'Gap up > 2% today, Price > EMA 200, Volume > 1M/day',
    },
    'rsi_oversold_bounce': {
        'name': 'RSI Oversold Bounce',
        'description': 'RSI(14) < 30, Price > EMA 200, Bullish candle, Volume > 1M/day',
    },
}


@dataclass
class WatchlistResult:
    """Result for a stock matching a watchlist criteria"""
    watchlist_name: str
    ticker: str
    scan_date: date
    price: float
    volume: int
    avg_volume: int
    ema_20: float
    ema_50: float
    ema_200: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float
    gap_pct: float
    price_change_1d: float
    vwap: Optional[float] = None
    crossover_date: Optional[date] = None  # Date when crossover first occurred
    status: str = 'active'  # 'active' or 'expired'


@dataclass
class ExistingEntry:
    """Existing active watchlist entry from database"""
    ticker: str
    crossover_date: date
    watchlist_name: str


class WatchlistScanner:
    """
    Scans all active stocks for 5 predefined technical setups.

    Each watchlist has specific criteria and all require:
    - Minimum 1M average daily volume
    - Valid indicator data
    """

    MIN_VOLUME = 1_000_000  # 1M minimum daily volume
    MIN_HISTORY_DAYS = 250  # Need at least 250 days for EMA 200

    def __init__(self, db_manager):
        """
        Initialize the watchlist scanner.

        Args:
            db_manager: Database manager for queries
        """
        self.db = db_manager
        self.data_provider = BacktestDataProvider(db_manager)

    async def get_existing_active_entries(
        self,
        watchlist_name: str
    ) -> Dict[str, ExistingEntry]:
        """
        Get all currently active entries for a watchlist.

        Returns:
            Dict mapping ticker to ExistingEntry
        """
        query = """
            SELECT ticker, crossover_date, watchlist_name
            FROM stock_watchlist_results
            WHERE watchlist_name = $1
            AND status = 'active'
        """
        rows = await self.db.fetch(query, watchlist_name)
        return {
            r['ticker']: ExistingEntry(
                ticker=r['ticker'],
                crossover_date=r['crossover_date'],
                watchlist_name=r['watchlist_name']
            )
            for r in rows
        }

    async def expire_entry(self, watchlist_name: str, ticker: str) -> None:
        """Mark an entry as expired (price dropped below EMA or MACD went negative)."""
        query = """
            UPDATE stock_watchlist_results
            SET status = 'expired'
            WHERE watchlist_name = $1
            AND ticker = $2
            AND status = 'active'
        """
        await self.db.execute(query, watchlist_name, ticker)
        logger.debug(f"Expired {ticker} from {watchlist_name}")

    async def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """
        Remove entries older than max_age_days.

        Returns:
            Number of entries removed
        """
        query = """
            DELETE FROM stock_watchlist_results
            WHERE crossover_date < CURRENT_DATE - INTERVAL '1 day' * $1
            OR status = 'expired'
        """
        result = await self.db.execute(query, max_age_days)
        # Extract count from result if available
        try:
            count = int(result.split()[-1]) if result else 0
        except:
            count = 0
        logger.info(f"Cleaned up entries older than {max_age_days} days or expired")
        return count

    async def get_all_active_tickers(self) -> List[str]:
        """Get all active stock tickers from stock_instruments."""
        query = """
            SELECT ticker
            FROM stock_instruments
            WHERE is_active = true
            ORDER BY ticker
        """
        rows = await self.db.fetch(query)
        return [r['ticker'] for r in rows]

    async def scan_all_watchlists(
        self,
        calculation_date: datetime = None
    ) -> Dict[str, List[WatchlistResult]]:
        """
        Run all 5 watchlist scans with crossover tracking.

        For crossover watchlists (EMA 50/20, MACD):
        - NEW crossovers: Added with crossover_date = today
        - EXISTING entries: Updated if still valid, expired if condition fails
        - Old entries (>30 days) are cleaned up

        For event watchlists (Gap Up, RSI Oversold):
        - Single-day events, always use today's date

        Args:
            calculation_date: Date to run scan for (default: today)

        Returns:
            Dict mapping watchlist_name to list of matching WatchlistResults
        """
        if calculation_date is None:
            # Use today's date - represents when the pipeline ran
            # (even though data is from previous trading day's close)
            calculation_date = datetime.now()

        if isinstance(calculation_date, datetime):
            scan_date = calculation_date.date()
        else:
            scan_date = calculation_date

        logger.info(f"Starting watchlist scans for {scan_date}")

        # Get all active tickers
        tickers = await self.get_all_active_tickers()
        logger.info(f"Scanning {len(tickers)} active stocks for watchlists")

        # Load existing active entries for crossover watchlists
        existing_entries: Dict[str, Dict[str, ExistingEntry]] = {}
        for wl_name in CROSSOVER_WATCHLISTS:
            existing_entries[wl_name] = await self.get_existing_active_entries(wl_name)
            logger.info(f"  {wl_name}: {len(existing_entries[wl_name])} existing active entries")

        # Track which existing entries we've seen (to expire ones that no longer qualify)
        seen_tickers: Dict[str, Set[str]] = {wl: set() for wl in CROSSOVER_WATCHLISTS}

        # Initialize results
        results = {name: [] for name in WATCHLIST_DEFINITIONS.keys()}

        scanned = 0
        errors = 0
        new_crossovers = {wl: 0 for wl in CROSSOVER_WATCHLISTS}
        continued = {wl: 0 for wl in CROSSOVER_WATCHLISTS}

        for ticker in tickers:
            try:
                # Fetch candle data with indicators
                df = await self.data_provider.get_historical_data(
                    ticker=ticker,
                    start_date=scan_date - timedelta(days=365),
                    end_date=scan_date,
                    timeframe='1d'
                )

                if df.empty or len(df) < self.MIN_HISTORY_DAYS:
                    continue

                scanned += 1

                # Check volume requirement first
                avg_volume = df['volume'].tail(20).mean()
                if avg_volume < self.MIN_VOLUME:
                    continue

                # Get latest data
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest

                price = float(latest['close'])
                volume = int(latest['volume'])

                # Get the actual candle date (for accurate crossover_date)
                candle_timestamp = latest.get('timestamp')
                if candle_timestamp is not None:
                    if hasattr(candle_timestamp, 'date'):
                        actual_candle_date = candle_timestamp.date()
                    else:
                        actual_candle_date = scan_date
                else:
                    actual_candle_date = scan_date

                # Get indicators (note: BacktestDataProvider uses 'rsi' not 'rsi_14')
                ema_20 = float(latest.get('ema_20', 0))
                ema_50 = float(latest.get('ema_50', 0))
                ema_200 = float(latest.get('ema_200', 0))
                rsi_14 = float(latest.get('rsi', latest.get('rsi_14', 50)))  # Try both column names
                macd = float(latest.get('macd', 0))
                macd_signal = float(latest.get('macd_signal', latest.get('signal', 0)))
                macd_histogram = float(latest.get('macd_histogram', latest.get('histogram', 0)))

                # Previous values for crossover detection
                prev_macd = float(prev.get('macd', 0))
                prev_macd_signal = float(prev.get('macd_signal', prev.get('signal', 0)))

                # Gap and change calculations
                gap_pct = ((latest['open'] - prev['close']) / prev['close'] * 100) if prev['close'] > 0 else 0
                price_change_1d = ((price - prev['close']) / prev['close'] * 100) if prev['close'] > 0 else 0

                # Base result factory (reused for matching watchlists)
                def make_result(wl_name: str, crossover_dt: date) -> WatchlistResult:
                    return WatchlistResult(
                        watchlist_name=wl_name,
                        ticker=ticker,
                        scan_date=scan_date,
                        price=price,
                        volume=volume,
                        avg_volume=int(avg_volume),
                        ema_20=ema_20,
                        ema_50=ema_50,
                        ema_200=ema_200,
                        rsi_14=rsi_14,
                        macd=macd,
                        macd_signal=macd_signal,
                        macd_histogram=macd_histogram,
                        gap_pct=gap_pct,
                        price_change_1d=price_change_1d,
                        crossover_date=crossover_dt,
                        status='active',
                    )

                # ===== CROSSOVER WATCHLISTS (with tracking) =====

                # 1. EMA 50 Crossover
                wl_name = 'ema_50_crossover'
                is_new_crossover = self._check_ema_50_crossover(price, ema_50, ema_200, prev, latest)
                still_valid = self._check_still_above_ema(price, ema_50, ema_200)

                if ticker in existing_entries[wl_name]:
                    # Existing entry - check if still valid
                    if still_valid:
                        # Continue tracking with original crossover_date
                        seen_tickers[wl_name].add(ticker)
                        existing = existing_entries[wl_name][ticker]
                        results[wl_name].append(make_result(wl_name, existing.crossover_date))
                        continued[wl_name] += 1
                    # If not valid, don't add to seen_tickers (will be expired later)
                elif is_new_crossover:
                    # New crossover - use actual candle date, not scan date
                    results[wl_name].append(make_result(wl_name, actual_candle_date))
                    new_crossovers[wl_name] += 1

                # 2. EMA 20 Crossover
                wl_name = 'ema_20_crossover'
                is_new_crossover = self._check_ema_20_crossover(price, ema_20, ema_200, prev, latest)
                still_valid = self._check_still_above_ema(price, ema_20, ema_200)

                if ticker in existing_entries[wl_name]:
                    if still_valid:
                        seen_tickers[wl_name].add(ticker)
                        existing = existing_entries[wl_name][ticker]
                        results[wl_name].append(make_result(wl_name, existing.crossover_date))
                        continued[wl_name] += 1
                    # If not valid, don't add to seen_tickers (will be expired later)
                elif is_new_crossover:
                    # New crossover - use actual candle date, not scan date
                    results[wl_name].append(make_result(wl_name, actual_candle_date))
                    new_crossovers[wl_name] += 1

                # 3. MACD Bullish Cross (MACD crosses above Signal line)
                wl_name = 'macd_bullish_cross'
                is_new_crossover = self._check_macd_bullish_cross(price, ema_200, macd, macd_signal, prev_macd, prev_macd_signal)
                still_valid = self._check_macd_still_bullish(price, ema_200, macd, macd_signal)

                if ticker in existing_entries[wl_name]:
                    if still_valid:
                        seen_tickers[wl_name].add(ticker)
                        existing = existing_entries[wl_name][ticker]
                        results[wl_name].append(make_result(wl_name, existing.crossover_date))
                        continued[wl_name] += 1
                    # If not valid, don't add to seen_tickers (will be expired later)
                elif is_new_crossover:
                    # New crossover - use actual candle date, not scan date
                    results[wl_name].append(make_result(wl_name, actual_candle_date))
                    new_crossovers[wl_name] += 1

                # ===== EVENT WATCHLISTS (single-day, no tracking) =====

                # 4. Gap Up Continuation
                if self._check_gap_up_continuation(price, ema_200, gap_pct, latest):
                    results['gap_up_continuation'].append(make_result('gap_up_continuation', scan_date))

                # 5. RSI Oversold Bounce
                if self._check_rsi_oversold_bounce(price, ema_200, rsi_14, latest, prev):
                    results['rsi_oversold_bounce'].append(make_result('rsi_oversold_bounce', scan_date))

            except Exception as e:
                errors += 1
                if errors <= 5:  # Only log first 5 errors
                    logger.warning(f"{ticker}: Error scanning - {e}")
                continue

        # Expire entries that no longer qualify
        expired_count = 0
        for wl_name in CROSSOVER_WATCHLISTS:
            for ticker, entry in existing_entries[wl_name].items():
                if ticker not in seen_tickers[wl_name]:
                    # This ticker was in existing entries but didn't qualify today
                    # Either volume dropped or price dropped below EMA
                    await self.expire_entry(wl_name, ticker)
                    expired_count += 1

        # Log summary
        total_matches = sum(len(r) for r in results.values())
        logger.info(f"Watchlist scan complete: {scanned} stocks scanned, {total_matches} total active")
        logger.info(f"  Expired {expired_count} entries that no longer qualify")
        for wl_name in CROSSOVER_WATCHLISTS:
            logger.info(f"  {wl_name}: {new_crossovers[wl_name]} new, {continued[wl_name]} continued")
        for wl_name in ['gap_up_continuation', 'rsi_oversold_bounce']:
            if results[wl_name]:
                logger.info(f"  {wl_name}: {len(results[wl_name])} stocks (event-based)")

        return results

    def _check_still_above_ema(
        self,
        price: float,
        ema: float,
        ema_200: float
    ) -> bool:
        """Check if price is still above both the EMA and EMA 200 (continuation check)."""
        if ema <= 0 or ema_200 <= 0:
            return False
        return price > ema and price > ema_200

    def _check_macd_still_bullish(
        self,
        price: float,
        ema_200: float,
        macd: float,
        macd_signal: float
    ) -> bool:
        """Check if MACD is still above Signal line and price above EMA 200 (continuation check)."""
        if ema_200 <= 0:
            return False
        return macd > macd_signal and price > ema_200

    def _check_ema_50_crossover(
        self,
        price: float,
        ema_50: float,
        ema_200: float,
        prev: pd.Series,
        latest: pd.Series
    ) -> bool:
        """
        Check EMA 50 crossover criteria:
        - Price > EMA 200 (uptrend confirmation)
        - Price crosses above EMA 50 (today above, yesterday below)
        """
        if ema_50 <= 0 or ema_200 <= 0:
            return False

        # Price must be above EMA 200
        if price <= ema_200:
            return False

        # Current price above EMA 50
        if price <= ema_50:
            return False

        # Previous close was below EMA 50 (crossover)
        prev_ema_50 = float(prev.get('ema_50', ema_50))
        if prev['close'] >= prev_ema_50:
            return False

        return True

    def _check_ema_20_crossover(
        self,
        price: float,
        ema_20: float,
        ema_200: float,
        prev: pd.Series,
        latest: pd.Series
    ) -> bool:
        """
        Check EMA 20 crossover criteria:
        - Price > EMA 200 (uptrend confirmation)
        - Price crosses above EMA 20 (today above, yesterday below)
        """
        if ema_20 <= 0 or ema_200 <= 0:
            return False

        # Price must be above EMA 200
        if price <= ema_200:
            return False

        # Current price above EMA 20
        if price <= ema_20:
            return False

        # Previous close was below EMA 20 (crossover)
        prev_ema_20 = float(prev.get('ema_20', ema_20))
        if prev['close'] >= prev_ema_20:
            return False

        return True

    def _check_macd_bullish_cross(
        self,
        price: float,
        ema_200: float,
        macd: float,
        macd_signal: float,
        prev_macd: float,
        prev_macd_signal: float
    ) -> bool:
        """
        Check MACD bullish cross criteria:
        - MACD crosses above Signal line (today MACD > Signal, yesterday MACD < Signal)
        - Price > EMA 200 (uptrend confirmation)
        """
        if ema_200 <= 0:
            return False

        # Price must be above EMA 200
        if price <= ema_200:
            return False

        # MACD must be above Signal line now
        if macd <= macd_signal:
            return False

        # MACD was below Signal line before (crossover)
        if prev_macd >= prev_macd_signal:
            return False

        return True

    def _check_gap_up_continuation(
        self,
        price: float,
        ema_200: float,
        gap_pct: float,
        latest: pd.Series
    ) -> bool:
        """
        Check Gap Up Continuation criteria:
        - Gap up > 2%
        - Price > EMA 200 (uptrend confirmation)
        - Closing above open (green candle = continuation)
        """
        if ema_200 <= 0:
            return False

        # Gap must be > 2%
        if gap_pct < 2.0:
            return False

        # Price must be above EMA 200
        if price <= ema_200:
            return False

        # Price should be closing green (continuation)
        if latest['close'] <= latest['open']:
            return False

        return True

    def _check_rsi_oversold_bounce(
        self,
        price: float,
        ema_200: float,
        rsi_14: float,
        latest: pd.Series,
        prev: pd.Series
    ) -> bool:
        """
        Check RSI Oversold Bounce criteria:
        - RSI < 30 (oversold)
        - Price > EMA 200 (still in uptrend)
        - Bullish candle (close > open)
        """
        if ema_200 <= 0:
            return False

        # RSI must be oversold
        if rsi_14 >= 30:
            return False

        # Price should still be above EMA 200 (pullback in uptrend)
        if price <= ema_200:
            return False

        # Must be a bullish candle (reversal signal)
        if latest['close'] <= latest['open']:
            return False

        return True

    async def save_results(
        self,
        results: Dict[str, List[WatchlistResult]]
    ) -> int:
        """
        Save watchlist scan results to database.

        For crossover watchlists:
        - Uses partial unique index on (watchlist_name, ticker) WHERE status = 'active'
        - Updates scan_date and price data, preserves crossover_date

        For event watchlists:
        - Each day's events are separate entries

        Args:
            results: Dict mapping watchlist_name to list of WatchlistResults

        Returns:
            Total number of records saved
        """
        total_saved = 0

        for watchlist_name, matches in results.items():
            if not matches:
                continue

            is_crossover_wl = watchlist_name in CROSSOVER_WATCHLISTS

            for r in matches:
                if is_crossover_wl:
                    # For crossover watchlists: upsert on (watchlist_name, ticker) where active
                    # This uses the partial unique index
                    query = """
                        INSERT INTO stock_watchlist_results (
                            watchlist_name, ticker, scan_date, crossover_date, status,
                            price, volume, avg_volume,
                            ema_20, ema_50, ema_200,
                            rsi_14, macd, macd_signal, macd_histogram,
                            gap_pct, price_change_1d, vwap
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                        ON CONFLICT (watchlist_name, ticker) WHERE status = 'active'
                        DO UPDATE SET
                            scan_date = EXCLUDED.scan_date,
                            price = EXCLUDED.price,
                            volume = EXCLUDED.volume,
                            avg_volume = EXCLUDED.avg_volume,
                            ema_20 = EXCLUDED.ema_20,
                            ema_50 = EXCLUDED.ema_50,
                            ema_200 = EXCLUDED.ema_200,
                            rsi_14 = EXCLUDED.rsi_14,
                            macd = EXCLUDED.macd,
                            macd_signal = EXCLUDED.macd_signal,
                            macd_histogram = EXCLUDED.macd_histogram,
                            gap_pct = EXCLUDED.gap_pct,
                            price_change_1d = EXCLUDED.price_change_1d,
                            vwap = EXCLUDED.vwap,
                            created_at = NOW()
                    """
                    values = (
                        r.watchlist_name,
                        r.ticker,
                        r.scan_date,
                        r.crossover_date,
                        r.status,
                        r.price,
                        r.volume,
                        r.avg_volume,
                        r.ema_20,
                        r.ema_50,
                        r.ema_200,
                        r.rsi_14,
                        r.macd,
                        r.macd_signal,
                        r.macd_histogram,
                        r.gap_pct,
                        r.price_change_1d,
                        r.vwap,
                    )
                else:
                    # For event watchlists: use UPSERT to handle re-detection of same ticker
                    # The partial unique index on (watchlist_name, ticker) WHERE status = 'active'
                    # prevents duplicates, so we use ON CONFLICT to update existing entries
                    query = """
                        INSERT INTO stock_watchlist_results (
                            watchlist_name, ticker, scan_date, crossover_date, status,
                            price, volume, avg_volume,
                            ema_20, ema_50, ema_200,
                            rsi_14, macd, macd_signal, macd_histogram,
                            gap_pct, price_change_1d, vwap
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                        ON CONFLICT (watchlist_name, ticker) WHERE status = 'active'
                        DO UPDATE SET
                            scan_date = EXCLUDED.scan_date,
                            crossover_date = EXCLUDED.crossover_date,
                            price = EXCLUDED.price,
                            volume = EXCLUDED.volume,
                            avg_volume = EXCLUDED.avg_volume,
                            ema_20 = EXCLUDED.ema_20,
                            ema_50 = EXCLUDED.ema_50,
                            ema_200 = EXCLUDED.ema_200,
                            rsi_14 = EXCLUDED.rsi_14,
                            macd = EXCLUDED.macd,
                            macd_signal = EXCLUDED.macd_signal,
                            macd_histogram = EXCLUDED.macd_histogram,
                            gap_pct = EXCLUDED.gap_pct,
                            price_change_1d = EXCLUDED.price_change_1d,
                            vwap = EXCLUDED.vwap,
                            created_at = NOW()
                    """
                    values = (
                        r.watchlist_name,
                        r.ticker,
                        r.scan_date,
                        r.crossover_date,  # Same as scan_date for events
                        r.status,
                        r.price,
                        r.volume,
                        r.avg_volume,
                        r.ema_20,
                        r.ema_50,
                        r.ema_200,
                        r.rsi_14,
                        r.macd,
                        r.macd_signal,
                        r.macd_histogram,
                        r.gap_pct,
                        r.price_change_1d,
                        r.vwap,
                    )

                try:
                    await self.db.execute(query, *values)
                    total_saved += 1
                except Exception as e:
                    logger.error(f"Failed to save {watchlist_name}/{r.ticker}: {e}")

        logger.info(f"Saved {total_saved} watchlist results to database")
        return total_saved

    async def run(self, calculation_date: datetime = None) -> Dict[str, int]:
        """
        Run all watchlist scans and save results.

        Also cleans up old entries (>30 days) and expired entries.

        Args:
            calculation_date: Date to run scan for (default: today)

        Returns:
            Dict with counts per watchlist
        """
        # First, clean up old and expired entries
        await self.cleanup_old_entries(MAX_CROSSOVER_AGE_DAYS)

        # Run the scans
        results = await self.scan_all_watchlists(calculation_date)
        await self.save_results(results)

        # Enrich results with trading metrics
        enriched_count = await self.enrich_with_trading_metrics()
        logger.info(f"Enriched {enriched_count} watchlist results with trading metrics")

        return {name: len(matches) for name, matches in results.items()}

    async def enrich_with_trading_metrics(self) -> int:
        """
        Enrich active watchlist results with trading metrics from stock_screening_metrics.

        Pulls ATR, support/resistance levels, RS data, and calculates structure-based
        trade plan using swing levels, order blocks, and ATR.

        Returns:
            Number of records enriched
        """
        # Step 1: Pull raw metrics from screening data
        query = """
            UPDATE stock_watchlist_results wr
            SET
                -- ATR and volatility
                atr_14 = sm.atr_14,
                atr_percent = sm.atr_percent,

                -- Support/Resistance from SMC analysis
                swing_high = sm.swing_high,
                swing_low = sm.swing_low,
                swing_high_date = sm.swing_high_date,
                swing_low_date = sm.swing_low_date,

                -- Nearest order block
                nearest_ob_price = sm.nearest_ob_price,
                nearest_ob_type = sm.nearest_ob_type,
                nearest_ob_distance = sm.nearest_ob_distance,

                -- Relative strength
                rs_percentile = sm.rs_percentile,
                rs_trend = sm.rs_trend,

                -- Volume context
                relative_volume = sm.relative_volume,
                avg_daily_change_5d = sm.avg_daily_change_5d,

                -- Entry zone: current price -0.5% to +1%
                suggested_entry_low = wr.price * 0.995,
                suggested_entry_high = wr.price * 1.01

            FROM stock_screening_metrics sm
            WHERE wr.ticker = sm.ticker
            AND wr.status = 'active'
            AND sm.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            RETURNING wr.id
        """

        try:
            result = await self.db.fetch(query)
            enriched_count = len(result) if result else 0

            # Calculate volume trend
            await self._calculate_volume_trend()
            # Calculate structure-based SL/TP levels
            await self._calculate_structure_levels()
            # Calculate trade-ready filter and score
            await self._calculate_trade_ready()

            return enriched_count
        except Exception as e:
            logger.error(f"Failed to enrich watchlist with trading metrics: {e}")
            return 0

    async def _calculate_volume_trend(self) -> None:
        """Calculate volume trend based on relative volume."""
        query = """
            UPDATE stock_watchlist_results
            SET
                volume_trend = CASE
                    WHEN relative_volume >= 1.5 THEN 'accumulation'
                    WHEN relative_volume <= 0.7 THEN 'distribution'
                    ELSE 'neutral'
                END
            WHERE status = 'active'
            AND relative_volume IS NOT NULL
        """
        try:
            await self.db.execute(query)
        except Exception as e:
            logger.error(f"Failed to calculate volume trend: {e}")

    async def _calculate_structure_levels(self) -> None:
        """
        Calculate structure-based SL/TP using swing levels, order blocks, EMA 50, and ATR.

        Stop Loss: Highest structural support below price (swing_low, bullish OB, ema_50),
                   capped at 2x ATR if too wide, floored at 0.5x ATR if too tight.
        Target 1:  Lowest structural resistance above price (swing_high, bearish OB),
                   fallback to 2x risk distance.
        Target 2:  3x risk distance above entry.
        """
        query = """
            UPDATE stock_watchlist_results wr
            SET
                -- Structure-based stop loss
                structure_stop_loss = CASE
                    -- No ATR data: fallback to 3% below
                    WHEN wr.atr_14 IS NULL OR wr.price IS NULL THEN wr.price * 0.97

                    ELSE (
                        SELECT CASE
                            -- If best candidate is too far (> 2x ATR AND > 8% of price), cap at 2x ATR
                            WHEN best_sl IS NOT NULL
                                AND (wr.price - best_sl) > (2.0 * wr.atr_14)
                                AND (wr.price - best_sl) / wr.price > 0.08
                            THEN wr.price - (2.0 * wr.atr_14)

                            -- If best candidate is too tight (< 0.5x ATR), widen to 0.5x ATR
                            WHEN best_sl IS NOT NULL
                                AND (wr.price - best_sl) < (0.5 * wr.atr_14)
                            THEN wr.price - (0.5 * wr.atr_14)

                            -- Good structural level
                            WHEN best_sl IS NOT NULL THEN best_sl

                            -- No structure data: fallback to 1.5x ATR
                            ELSE wr.price - (1.5 * wr.atr_14)
                        END
                        FROM (
                            SELECT MAX(candidate) AS best_sl
                            FROM (
                                SELECT wr.swing_low AS candidate
                                WHERE wr.swing_low IS NOT NULL AND wr.swing_low < wr.price
                                UNION ALL
                                SELECT wr.nearest_ob_price AS candidate
                                WHERE wr.nearest_ob_price IS NOT NULL
                                    AND wr.nearest_ob_type = 'bullish'
                                    AND wr.nearest_ob_price < wr.price
                                UNION ALL
                                SELECT wr.ema_50 AS candidate
                                WHERE wr.ema_50 IS NOT NULL AND wr.ema_50 < wr.price
                            ) candidates
                        ) sl_calc
                    )
                END,

                -- Structure-based target 1 (nearest resistance)
                structure_target_1 = CASE
                    WHEN wr.atr_14 IS NULL OR wr.price IS NULL THEN wr.price * 1.05
                    ELSE COALESCE(
                        (
                            SELECT MIN(candidate)
                            FROM (
                                SELECT wr.swing_high AS candidate
                                WHERE wr.swing_high IS NOT NULL AND wr.swing_high > wr.price
                                UNION ALL
                                SELECT wr.nearest_ob_price AS candidate
                                WHERE wr.nearest_ob_price IS NOT NULL
                                    AND wr.nearest_ob_type = 'bearish'
                                    AND wr.nearest_ob_price > wr.price
                            ) candidates
                        ),
                        -- Fallback: ensure minimum 2:1 R:R
                        NULL
                    )
                END
            WHERE wr.status = 'active'
            AND wr.price IS NOT NULL
        """
        try:
            await self.db.execute(query)
        except Exception as e:
            logger.error(f"Failed to calculate structure levels: {e}")
            return

        # Step 2: Fill in TP1 fallback (2x risk) and TP2, R:R, and populate suggested_ columns
        query2 = """
            UPDATE stock_watchlist_results wr
            SET
                -- If structure_target_1 is NULL or too close, use 2x risk distance
                structure_target_1 = CASE
                    WHEN structure_target_1 IS NOT NULL
                        AND structure_stop_loss IS NOT NULL
                        AND (structure_target_1 - wr.price) >= (wr.price - structure_stop_loss)
                    THEN structure_target_1
                    ELSE wr.price + 2.0 * (wr.price - COALESCE(structure_stop_loss, wr.price * 0.97))
                END,

                -- Target 2: 3x risk distance
                structure_target_2 = wr.price + 3.0 * (wr.price - COALESCE(structure_stop_loss, wr.price * 0.97)),

                -- R:R ratio
                structure_rr_ratio = CASE
                    WHEN structure_stop_loss IS NOT NULL
                        AND (wr.price - structure_stop_loss) > 0
                    THEN ROUND(
                        COALESCE(
                            CASE
                                WHEN structure_target_1 IS NOT NULL
                                    AND (structure_target_1 - wr.price) >= (wr.price - structure_stop_loss)
                                THEN (structure_target_1 - wr.price)
                                ELSE 2.0 * (wr.price - structure_stop_loss)
                            END
                            / (wr.price - structure_stop_loss)
                        , 2.0)::numeric, 2)
                    ELSE 2.00
                END,

                -- Populate existing suggested_ columns with structure values
                suggested_stop_loss = structure_stop_loss,
                suggested_target_1 = CASE
                    WHEN structure_target_1 IS NOT NULL
                        AND structure_stop_loss IS NOT NULL
                        AND (structure_target_1 - wr.price) >= (wr.price - structure_stop_loss)
                    THEN structure_target_1
                    ELSE wr.price + 2.0 * (wr.price - COALESCE(structure_stop_loss, wr.price * 0.97))
                END,
                suggested_target_2 = wr.price + 3.0 * (wr.price - COALESCE(structure_stop_loss, wr.price * 0.97)),
                risk_reward_ratio = CASE
                    WHEN structure_stop_loss IS NOT NULL
                        AND (wr.price - structure_stop_loss) > 0
                    THEN ROUND(
                        COALESCE(
                            CASE
                                WHEN structure_target_1 IS NOT NULL
                                    AND (structure_target_1 - wr.price) >= (wr.price - structure_stop_loss)
                                THEN (structure_target_1 - wr.price)
                                ELSE 2.0 * (wr.price - structure_stop_loss)
                            END
                            / (wr.price - structure_stop_loss)
                        , 2.0)::numeric, 2)
                    ELSE 2.00
                END,
                risk_percent = CASE
                    WHEN structure_stop_loss IS NOT NULL AND wr.price > 0
                    THEN ROUND(((wr.price - structure_stop_loss) / wr.price * 100)::numeric, 2)
                    ELSE 3.00
                END
            WHERE wr.status = 'active'
            AND wr.price IS NOT NULL
        """
        try:
            await self.db.execute(query2)
        except Exception as e:
            logger.error(f"Failed to calculate TP/RR levels: {e}")

    async def _calculate_trade_ready(self) -> None:
        """
        Calculate trade-ready filter and composite score for active watchlist entries.

        Pass/Fail Criteria (all must pass):
        - DAQ grade >= B (score >= 60)
        - RS percentile >= 55
        - Earnings risk = FALSE
        - RSI < 78
        - ATR % between 1.5% and 10%
        - Volume trend != 'distribution'
        - R:R ratio >= 1.5
        - Crossover age 1-10 days (crossover watchlists only)

        Score (0-100): DAQ(30%) + RS(25%) + R:R(20%) + Volume(10%) + Freshness(15%)
        """
        query = """
            UPDATE stock_watchlist_results wr
            SET
                trade_ready = (
                    COALESCE(wr.daq_score, 0) >= 60
                    AND COALESCE(wr.rs_percentile, 0) >= 55
                    AND COALESCE(wr.daq_earnings_risk, TRUE) = FALSE
                    AND COALESCE(wr.rsi_14, 80) < 78
                    AND COALESCE(wr.atr_percent, 0) >= 1.5
                    AND COALESCE(wr.atr_percent, 100) <= 10.0
                    AND COALESCE(wr.volume_trend, 'distribution') != 'distribution'
                    AND COALESCE(wr.structure_rr_ratio, 0) >= 1.5
                    AND (
                        wr.crossover_date IS NULL
                        OR (
                            (CURRENT_DATE - wr.crossover_date) + 1 BETWEEN 1 AND 10
                        )
                    )
                ),

                trade_ready_score = (
                    -- DAQ component (30% weight, max 30 points)
                    LEAST(ROUND(COALESCE(wr.daq_score, 0) * 0.30), 30)

                    -- RS component (25% weight, max 25 points)
                    + LEAST(ROUND(COALESCE(wr.rs_percentile, 0) * 0.25), 25)

                    -- R:R quality (20% weight): 1.5=50, 2.0=70, 3.0+=100
                    + ROUND(LEAST(
                        CASE
                            WHEN COALESCE(wr.structure_rr_ratio, 0) >= 3.0 THEN 100
                            WHEN COALESCE(wr.structure_rr_ratio, 0) >= 2.0 THEN 70 + (COALESCE(wr.structure_rr_ratio, 0) - 2.0) * 30
                            WHEN COALESCE(wr.structure_rr_ratio, 0) >= 1.5 THEN 50 + (COALESCE(wr.structure_rr_ratio, 0) - 1.5) * 40
                            ELSE COALESCE(wr.structure_rr_ratio, 0) * 33.3
                        END
                    , 100) * 0.20)

                    -- Volume confirmation (10%): accumulation=100, neutral=50, distribution=0
                    + ROUND(CASE
                        WHEN wr.volume_trend = 'accumulation' THEN 100
                        WHEN wr.volume_trend = 'neutral' THEN 50
                        ELSE 0
                    END * 0.10)

                    -- Crossover freshness (15%): peak at 3-7 days
                    + ROUND(CASE
                        WHEN wr.crossover_date IS NULL THEN 50
                        WHEN (CURRENT_DATE - wr.crossover_date) + 1 BETWEEN 3 AND 7 THEN 100
                        WHEN (CURRENT_DATE - wr.crossover_date) + 1 BETWEEN 1 AND 2 THEN 70
                        WHEN (CURRENT_DATE - wr.crossover_date) + 1 BETWEEN 8 AND 10 THEN 60
                        ELSE 10
                    END * 0.15)
                )::integer,

                trade_ready_reasons = (
                    SELECT json_agg(json_build_object('criterion', criterion, 'passed', passed))::text
                    FROM (VALUES
                        ('DAQ >= B (60)', COALESCE(wr.daq_score, 0) >= 60),
                        ('RS >= 55', COALESCE(wr.rs_percentile, 0) >= 55),
                        ('No earnings risk', COALESCE(wr.daq_earnings_risk, TRUE) = FALSE),
                        ('RSI < 78', COALESCE(wr.rsi_14, 80) < 78),
                        ('ATR 1.5-10%', COALESCE(wr.atr_percent, 0) >= 1.5 AND COALESCE(wr.atr_percent, 100) <= 10.0),
                        ('No distribution', COALESCE(wr.volume_trend, 'distribution') != 'distribution'),
                        ('R:R >= 1.5', COALESCE(wr.structure_rr_ratio, 0) >= 1.5),
                        ('Age 1-10d', wr.crossover_date IS NULL OR (CURRENT_DATE - wr.crossover_date) + 1 BETWEEN 1 AND 10)
                    ) AS criteria(criterion, passed)
                )
            WHERE wr.status = 'active'
        """
        try:
            await self.db.execute(query)
            # Log summary
            count_query = """
                SELECT
                    COUNT(*) FILTER (WHERE trade_ready = TRUE) as ready,
                    COUNT(*) as total
                FROM stock_watchlist_results
                WHERE status = 'active'
            """
            result = await self.db.fetchrow(count_query)
            if result:
                logger.info(f"Trade-ready filter: {result['ready']}/{result['total']} stocks passed")
        except Exception as e:
            logger.error(f"Failed to calculate trade-ready filter: {e}")
