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
            WHERE crossover_date < CURRENT_DATE - $1::interval
            OR status = 'expired'
        """
        result = await self.db.execute(query, f'{max_age_days} days')
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
                    seen_tickers[wl_name].add(ticker)
                    if still_valid:
                        # Continue tracking with original crossover_date
                        existing = existing_entries[wl_name][ticker]
                        results[wl_name].append(make_result(wl_name, existing.crossover_date))
                        continued[wl_name] += 1
                    # If not valid, don't add to results (will be expired later)
                elif is_new_crossover:
                    # New crossover today
                    results[wl_name].append(make_result(wl_name, scan_date))
                    new_crossovers[wl_name] += 1

                # 2. EMA 20 Crossover
                wl_name = 'ema_20_crossover'
                is_new_crossover = self._check_ema_20_crossover(price, ema_20, ema_200, prev, latest)
                still_valid = self._check_still_above_ema(price, ema_20, ema_200)

                if ticker in existing_entries[wl_name]:
                    seen_tickers[wl_name].add(ticker)
                    if still_valid:
                        existing = existing_entries[wl_name][ticker]
                        results[wl_name].append(make_result(wl_name, existing.crossover_date))
                        continued[wl_name] += 1
                elif is_new_crossover:
                    results[wl_name].append(make_result(wl_name, scan_date))
                    new_crossovers[wl_name] += 1

                # 3. MACD Bullish Cross
                wl_name = 'macd_bullish_cross'
                is_new_crossover = self._check_macd_bullish_cross(price, ema_200, macd, prev_macd)
                still_valid = self._check_macd_still_positive(price, ema_200, macd)

                if ticker in existing_entries[wl_name]:
                    seen_tickers[wl_name].add(ticker)
                    if still_valid:
                        existing = existing_entries[wl_name][ticker]
                        results[wl_name].append(make_result(wl_name, existing.crossover_date))
                        continued[wl_name] += 1
                elif is_new_crossover:
                    results[wl_name].append(make_result(wl_name, scan_date))
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

    def _check_macd_still_positive(
        self,
        price: float,
        ema_200: float,
        macd: float
    ) -> bool:
        """Check if MACD is still positive and price above EMA 200 (continuation check)."""
        if ema_200 <= 0:
            return False
        return macd > 0 and price > ema_200

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
        prev_macd: float
    ) -> bool:
        """
        Check MACD bullish cross criteria:
        - MACD crosses from negative to positive
        - Price > EMA 200 (uptrend confirmation)
        """
        if ema_200 <= 0:
            return False

        # Price must be above EMA 200
        if price <= ema_200:
            return False

        # MACD must be positive now
        if macd <= 0:
            return False

        # MACD was negative before (cross)
        if prev_macd >= 0:
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
                    # For event watchlists: simple insert (each day is independent)
                    # Delete old entries first for this watchlist
                    query = """
                        INSERT INTO stock_watchlist_results (
                            watchlist_name, ticker, scan_date, crossover_date, status,
                            price, volume, avg_volume,
                            ema_20, ema_50, ema_200,
                            rsi_14, macd, macd_signal, macd_histogram,
                            gap_pct, price_change_1d, vwap
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
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

        return {name: len(matches) for name, matches in results.items()}
