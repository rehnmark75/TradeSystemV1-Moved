"""
Market Hours Detection

Handles market session detection for various exchanges.
Important for stock trading since markets have specific open/close times.
"""

from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    CLOSED = "closed"


class Exchange(Enum):
    """Supported exchanges"""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    LSE = "LSE"
    XETRA = "XETRA"


# Market hours configuration (all times in local exchange timezone)
MARKET_SCHEDULES = {
    Exchange.NYSE: {
        "timezone": "America/New_York",
        "pre_market": (time(4, 0), time(9, 30)),
        "regular": (time(9, 30), time(16, 0)),
        "post_market": (time(16, 0), time(20, 0)),
        "trading_days": [0, 1, 2, 3, 4],  # Mon-Fri
    },
    Exchange.NASDAQ: {
        "timezone": "America/New_York",
        "pre_market": (time(4, 0), time(9, 30)),
        "regular": (time(9, 30), time(16, 0)),
        "post_market": (time(16, 0), time(20, 0)),
        "trading_days": [0, 1, 2, 3, 4],
    },
    Exchange.LSE: {
        "timezone": "Europe/London",
        "regular": (time(8, 0), time(16, 30)),
        "trading_days": [0, 1, 2, 3, 4],
    },
    Exchange.XETRA: {
        "timezone": "Europe/Berlin",
        "regular": (time(9, 0), time(17, 30)),
        "trading_days": [0, 1, 2, 3, 4],
    },
}

# US market holidays 2024-2026 (add more as needed)
US_HOLIDAYS = [
    # 2024
    datetime(2024, 1, 1),    # New Year's Day
    datetime(2024, 1, 15),   # MLK Day
    datetime(2024, 2, 19),   # Presidents Day
    datetime(2024, 3, 29),   # Good Friday
    datetime(2024, 5, 27),   # Memorial Day
    datetime(2024, 6, 19),   # Juneteenth
    datetime(2024, 7, 4),    # Independence Day
    datetime(2024, 9, 2),    # Labor Day
    datetime(2024, 11, 28),  # Thanksgiving
    datetime(2024, 12, 25),  # Christmas
    # 2025
    datetime(2025, 1, 1),    # New Year's Day
    datetime(2025, 1, 20),   # MLK Day
    datetime(2025, 2, 17),   # Presidents Day
    datetime(2025, 4, 18),   # Good Friday
    datetime(2025, 5, 26),   # Memorial Day
    datetime(2025, 6, 19),   # Juneteenth
    datetime(2025, 7, 4),    # Independence Day
    datetime(2025, 9, 1),    # Labor Day
    datetime(2025, 11, 27),  # Thanksgiving
    datetime(2025, 12, 25),  # Christmas
    # 2026
    datetime(2026, 1, 1),    # New Year's Day
    datetime(2026, 1, 19),   # MLK Day (3rd Monday)
    datetime(2026, 2, 16),   # Presidents Day (3rd Monday)
    datetime(2026, 4, 3),    # Good Friday
    datetime(2026, 5, 25),   # Memorial Day (last Monday)
    datetime(2026, 6, 19),   # Juneteenth
    datetime(2026, 7, 3),    # Independence Day (observed - July 4 is Saturday)
    datetime(2026, 9, 7),    # Labor Day (1st Monday)
    datetime(2026, 11, 26),  # Thanksgiving (4th Thursday)
    datetime(2026, 12, 25),  # Christmas
]


class MarketHoursChecker:
    """
    Check if markets are open and get session information

    Usage:
        checker = MarketHoursChecker()
        session = checker.get_session(Exchange.NYSE)
        is_open = checker.is_market_open(Exchange.NYSE)
        next_open = checker.get_next_open(Exchange.NYSE)
    """

    def __init__(self):
        """Initialize market hours checker"""
        try:
            import pytz
            self._pytz = pytz
        except ImportError:
            logger.warning("pytz not installed, using basic timezone handling")
            self._pytz = None

    def _get_exchange_time(self, exchange: Exchange) -> datetime:
        """Get current time in exchange timezone"""
        schedule = MARKET_SCHEDULES.get(exchange)
        if not schedule:
            raise ValueError(f"Unknown exchange: {exchange}")

        tz_name = schedule["timezone"]

        if self._pytz:
            tz = self._pytz.timezone(tz_name)
            return datetime.now(tz)
        else:
            # Fallback: assume UTC and apply offset
            # This is simplified and not accurate for DST
            utc_now = datetime.utcnow()
            if "New_York" in tz_name:
                return utc_now - timedelta(hours=5)  # EST
            elif "London" in tz_name:
                return utc_now  # GMT
            elif "Berlin" in tz_name:
                return utc_now + timedelta(hours=1)  # CET
            return utc_now

    def is_holiday(self, exchange: Exchange, dt: datetime = None) -> bool:
        """Check if date is a market holiday"""
        if dt is None:
            dt = self._get_exchange_time(exchange)

        check_date = dt.date() if hasattr(dt, 'date') else dt

        # Check US holidays for NYSE/NASDAQ
        if exchange in (Exchange.NYSE, Exchange.NASDAQ):
            for holiday in US_HOLIDAYS:
                if holiday.date() == check_date:
                    return True

        return False

    def get_session(self, exchange: Exchange, dt: datetime = None) -> MarketSession:
        """
        Get current market session

        Args:
            exchange: Exchange to check
            dt: Optional datetime (defaults to now)

        Returns:
            MarketSession enum value
        """
        schedule = MARKET_SCHEDULES.get(exchange)
        if not schedule:
            return MarketSession.CLOSED

        if dt is None:
            dt = self._get_exchange_time(exchange)

        # Check if trading day
        if dt.weekday() not in schedule["trading_days"]:
            return MarketSession.CLOSED

        # Check holidays
        if self.is_holiday(exchange, dt):
            return MarketSession.CLOSED

        current_time = dt.time() if hasattr(dt, 'time') else time(dt.hour, dt.minute)

        # Check sessions
        if "pre_market" in schedule:
            pre_start, pre_end = schedule["pre_market"]
            if pre_start <= current_time < pre_end:
                return MarketSession.PRE_MARKET

        if "regular" in schedule:
            reg_start, reg_end = schedule["regular"]
            if reg_start <= current_time < reg_end:
                return MarketSession.REGULAR

        if "post_market" in schedule:
            post_start, post_end = schedule["post_market"]
            if post_start <= current_time < post_end:
                return MarketSession.POST_MARKET

        return MarketSession.CLOSED

    def is_market_open(self, exchange: Exchange, include_extended: bool = False) -> bool:
        """
        Check if market is open

        Args:
            exchange: Exchange to check
            include_extended: Include pre/post market sessions

        Returns:
            True if market is open
        """
        session = self.get_session(exchange)

        if include_extended:
            return session in (
                MarketSession.PRE_MARKET,
                MarketSession.REGULAR,
                MarketSession.POST_MARKET
            )

        return session == MarketSession.REGULAR

    def is_regular_session(self, exchange: Exchange) -> bool:
        """Check if in regular trading session"""
        return self.get_session(exchange) == MarketSession.REGULAR

    def get_session_times(self, exchange: Exchange) -> Dict[str, Tuple[time, time]]:
        """Get session start/end times for an exchange"""
        schedule = MARKET_SCHEDULES.get(exchange, {})
        return {
            k: v for k, v in schedule.items()
            if isinstance(v, tuple) and len(v) == 2
        }

    def get_next_open(self, exchange: Exchange) -> Optional[datetime]:
        """
        Get next market open time

        Returns:
            datetime of next market open or None
        """
        schedule = MARKET_SCHEDULES.get(exchange)
        if not schedule or "regular" not in schedule:
            return None

        now = self._get_exchange_time(exchange)
        reg_start, _ = schedule["regular"]
        trading_days = schedule["trading_days"]

        # Start from today
        check_date = now.date()

        for _ in range(7):  # Check up to a week ahead
            if check_date.weekday() in trading_days:
                if not self.is_holiday(exchange, check_date):
                    open_dt = datetime.combine(check_date, reg_start)
                    if self._pytz:
                        tz = self._pytz.timezone(schedule["timezone"])
                        open_dt = tz.localize(open_dt)

                    if open_dt > now:
                        return open_dt

            check_date += timedelta(days=1)

        return None

    def get_next_close(self, exchange: Exchange) -> Optional[datetime]:
        """
        Get next market close time

        Returns:
            datetime of next market close or None
        """
        schedule = MARKET_SCHEDULES.get(exchange)
        if not schedule or "regular" not in schedule:
            return None

        now = self._get_exchange_time(exchange)
        _, reg_end = schedule["regular"]

        if self.is_regular_session(exchange):
            # Market is open, return today's close
            close_dt = datetime.combine(now.date(), reg_end)
            if self._pytz:
                tz = self._pytz.timezone(schedule["timezone"])
                close_dt = tz.localize(close_dt)
            return close_dt

        return None

    def time_until_open(self, exchange: Exchange) -> Optional[timedelta]:
        """Get time remaining until market opens"""
        next_open = self.get_next_open(exchange)
        if next_open:
            now = self._get_exchange_time(exchange)
            if self._pytz and hasattr(next_open, 'tzinfo'):
                return next_open - now
            return next_open.replace(tzinfo=None) - now.replace(tzinfo=None)
        return None

    def time_until_close(self, exchange: Exchange) -> Optional[timedelta]:
        """Get time remaining until market closes"""
        next_close = self.get_next_close(exchange)
        if next_close:
            now = self._get_exchange_time(exchange)
            if self._pytz and hasattr(next_close, 'tzinfo'):
                return next_close - now
            return next_close.replace(tzinfo=None) - now.replace(tzinfo=None)
        return None

    def get_status(self, exchange: Exchange) -> Dict:
        """
        Get comprehensive market status

        Returns:
            Dict with session info, times, etc.
        """
        session = self.get_session(exchange)
        now = self._get_exchange_time(exchange)

        status = {
            "exchange": exchange.value,
            "session": session.value,
            "is_open": session == MarketSession.REGULAR,
            "is_extended_open": session in (
                MarketSession.PRE_MARKET,
                MarketSession.REGULAR,
                MarketSession.POST_MARKET
            ),
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": MARKET_SCHEDULES.get(exchange, {}).get("timezone", "UTC"),
        }

        if session == MarketSession.CLOSED:
            next_open = self.get_next_open(exchange)
            if next_open:
                status["next_open"] = next_open.strftime("%Y-%m-%d %H:%M:%S")
                time_until = self.time_until_open(exchange)
                if time_until:
                    status["time_until_open"] = str(time_until)

        elif session == MarketSession.REGULAR:
            next_close = self.get_next_close(exchange)
            if next_close:
                status["closes_at"] = next_close.strftime("%H:%M:%S")
                time_until = self.time_until_close(exchange)
                if time_until:
                    status["time_until_close"] = str(time_until)

        return status


# Convenience functions
_checker = MarketHoursChecker()


def is_us_market_open(include_extended: bool = False) -> bool:
    """Check if US stock market (NYSE) is open"""
    return _checker.is_market_open(Exchange.NYSE, include_extended)


def get_us_market_session() -> MarketSession:
    """Get current US market session"""
    return _checker.get_session(Exchange.NYSE)


def get_us_market_status() -> Dict:
    """Get US market status"""
    return _checker.get_status(Exchange.NYSE)


def should_scan_stocks() -> bool:
    """
    Determine if we should run stock scans

    Returns True if:
    - Market is in regular session, OR
    - Within 30 minutes of market open (for preparation)
    """
    session = get_us_market_session()

    if session == MarketSession.REGULAR:
        return True

    # Check if we're close to market open
    time_until = _checker.time_until_open(Exchange.NYSE)
    if time_until and time_until <= timedelta(minutes=30):
        return True

    return False


def is_trading_day(dt: datetime = None) -> bool:
    """
    Check if a given date is a trading day (weekday and not a holiday).

    Args:
        dt: Date to check (defaults to today in ET)

    Returns:
        True if it's a trading day, False otherwise
    """
    if dt is None:
        dt = _checker._get_exchange_time(Exchange.NYSE)

    # Check if it's a weekend
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check if it's a holiday
    if _checker.is_holiday(Exchange.NYSE, dt):
        return False

    return True


def get_last_trading_day(dt: datetime = None) -> datetime:
    """
    Get the most recent trading day (excluding today if market hasn't closed).

    Args:
        dt: Reference date (defaults to now in ET)

    Returns:
        datetime of the last trading day
    """
    if dt is None:
        dt = _checker._get_exchange_time(Exchange.NYSE)

    # Start from yesterday
    check_date = dt - timedelta(days=1)

    # Go back until we find a trading day
    for _ in range(7):  # Max 7 days back (should never need more)
        if is_trading_day(check_date):
            return check_date
        check_date -= timedelta(days=1)

    # Fallback (shouldn't happen)
    return dt - timedelta(days=1)
