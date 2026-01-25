"""
Fundamentals Data Fetcher

Fetches comprehensive fundamental data from yfinance and stores in PostgreSQL.

Data includes:
- Earnings & Calendar: Earnings date, ex-dividend date
- Risk Metrics: Beta, short interest, short ratio
- Ownership: Institutional %, insider %, shares outstanding, float
- Valuation: P/E, P/B, P/S, PEG, EV/EBITDA
- Growth: Revenue growth, earnings growth
- Profitability: Profit margins, ROE, ROA
- Financial Health: Debt/Equity, current ratio, quick ratio
- Dividend: Yield, rate, payout ratio
- 52-Week: High, low, change percentage
- Analyst: Ratings, price targets

This data is refreshed weekly as fundamentals don't change frequently.
"""

import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf

logger = logging.getLogger(__name__)


class FundamentalsFetcher:
    """
    Fetches and manages fundamental stock data from yfinance.

    Data sources:
    - yfinance: All fundamental data from Yahoo Finance API

    Usage:
        fetcher = FundamentalsFetcher(db_manager)
        await fetcher.fetch_all_fundamentals(concurrency=10)
    """

    def __init__(self, db_manager, thread_pool_size: int = 4):
        """
        Initialize fundamentals fetcher.

        Args:
            db_manager: Async database manager instance
            thread_pool_size: Size of thread pool for yfinance calls
        """
        self.db = db_manager
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)

    def _fetch_fundamentals_sync(self, ticker: str) -> Optional[Dict]:
        """
        Synchronous yfinance fundamentals fetch (runs in thread pool).

        Args:
            ticker: Stock ticker

        Returns:
            Dict with fundamental data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info:
                return None

            # =================================================================
            # EARNINGS DATE EXTRACTION
            # =================================================================
            earnings_date = None
            earnings_estimated = True

            # Try calendar first (upcoming earnings)
            try:
                calendar = stock.calendar
                if calendar:
                    if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                        earnings_dates = calendar['Earnings Date']
                        if earnings_dates and len(earnings_dates) > 0:
                            ed = earnings_dates[0]
                            if isinstance(ed, date):
                                earnings_date = ed
                            elif isinstance(ed, datetime):
                                earnings_date = ed.date()
                    elif hasattr(calendar, 'columns') and 'Earnings Date' in calendar.columns:
                        earnings_dates = calendar['Earnings Date']
                        if len(earnings_dates) > 0 and earnings_dates[0]:
                            ed = earnings_dates[0]
                            if isinstance(ed, datetime):
                                earnings_date = ed.date()
                            elif hasattr(ed, 'date'):
                                earnings_date = ed.date()
            except Exception:
                pass

            # Fallback to earningsTimestamp from info
            if not earnings_date:
                try:
                    ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
                    if ts:
                        earnings_date = datetime.fromtimestamp(ts).date()
                    if 'isEarningsDateEstimate' in info:
                        earnings_estimated = bool(info['isEarningsDateEstimate'])
                except Exception:
                    pass

            # =================================================================
            # EX-DIVIDEND DATE
            # =================================================================
            ex_dividend_date = None
            try:
                ex_div_ts = info.get('exDividendDate')
                if ex_div_ts:
                    ex_dividend_date = datetime.fromtimestamp(ex_div_ts).date()
            except Exception:
                pass

            # =================================================================
            # BUILD FUNDAMENTALS DICT
            # =================================================================
            fundamentals = {
                'ticker': ticker,

                # Earnings & Calendar
                'earnings_date': earnings_date,
                'earnings_date_estimated': earnings_estimated,
                'ex_dividend_date': ex_dividend_date,

                # Risk Metrics
                'beta': self._safe_get(info, 'beta'),
                'short_ratio': self._safe_get(info, 'shortRatio'),
                # Note: shortPercentOfFloat is already a percentage (e.g., 5.23 = 5.23%)
                # Cap at 999.99 to fit NUMERIC(5,2) column - some heavily shorted stocks exceed 100%
                'short_percent_float': min(self._safe_get(info, 'shortPercentOfFloat') or 0, 999.99) or None,
                'shares_short': self._safe_get_int(info, 'sharesShort'),

                # Ownership
                'institutional_percent': self._safe_get(info, 'heldPercentInstitutions', multiply=100),
                'insider_percent': self._safe_get(info, 'heldPercentInsiders', multiply=100),
                'shares_outstanding': self._safe_get_int(info, 'sharesOutstanding'),
                'shares_float': self._safe_get_int(info, 'floatShares'),

                # Market Cap (stored as string in DB, but we format it nicely)
                'market_cap': self._format_market_cap(info.get('marketCap')),

                # Valuation Metrics
                'forward_pe': self._safe_get(info, 'forwardPE'),
                'trailing_pe': self._safe_get(info, 'trailingPE'),
                'price_to_book': self._safe_get(info, 'priceToBook'),
                'price_to_sales': self._safe_get(info, 'priceToSalesTrailing12Months'),
                'peg_ratio': self._safe_get(info, 'pegRatio'),
                'enterprise_to_ebitda': self._safe_get(info, 'enterpriseToEbitda'),
                'enterprise_to_revenue': self._safe_get(info, 'enterpriseToRevenue'),
                'enterprise_value': self._safe_get_int(info, 'enterpriseValue'),

                # Growth Metrics
                'revenue_growth': self._safe_get(info, 'revenueGrowth', multiply=100),
                'earnings_growth': self._safe_get(info, 'earningsGrowth', multiply=100),
                'earnings_quarterly_growth': self._safe_get(info, 'earningsQuarterlyGrowth', multiply=100),

                # Profitability Metrics
                'profit_margin': self._safe_get(info, 'profitMargins', multiply=100),
                'operating_margin': self._safe_get(info, 'operatingMargins', multiply=100),
                'gross_margin': self._safe_get(info, 'grossMargins', multiply=100),
                'return_on_equity': self._safe_get(info, 'returnOnEquity', multiply=100),
                'return_on_assets': self._safe_get(info, 'returnOnAssets', multiply=100),

                # Financial Health
                'debt_to_equity': self._safe_get(info, 'debtToEquity'),
                'current_ratio': self._safe_get(info, 'currentRatio'),
                'quick_ratio': self._safe_get(info, 'quickRatio'),

                # Dividend Info
                'dividend_yield': self._safe_get(info, 'dividendYield', multiply=100),
                'dividend_rate': self._safe_get(info, 'dividendRate'),
                'payout_ratio': self._safe_get(info, 'payoutRatio', multiply=100),

                # 52-Week Data
                'fifty_two_week_high': self._safe_get(info, 'fiftyTwoWeekHigh'),
                'fifty_two_week_low': self._safe_get(info, 'fiftyTwoWeekLow'),
                'fifty_two_week_change': self._safe_get(info, 'fiftyTwoWeekChange', multiply=100),

                # Moving Averages
                'fifty_day_average': self._safe_get(info, 'fiftyDayAverage'),
                'two_hundred_day_average': self._safe_get(info, 'twoHundredDayAverage'),

                # Analyst Data
                'analyst_rating': self._safe_get_str(info, 'recommendationKey'),
                'target_price': self._safe_get(info, 'targetMeanPrice'),
                'target_high': self._safe_get(info, 'targetHighPrice'),
                'target_low': self._safe_get(info, 'targetLowPrice'),
                'number_of_analysts': self._safe_get_int(info, 'numberOfAnalystOpinions'),

                # Company Info
                'business_summary': self._safe_get_str(info, 'longBusinessSummary', max_len=2000),
                'website': self._safe_get_str(info, 'website', max_len=255),
                'country': self._safe_get_str(info, 'country', max_len=50),
                'city': self._safe_get_str(info, 'city', max_len=100),
                'employee_count': self._safe_get_int(info, 'fullTimeEmployees'),

                # Also update sector/industry if available
                'sector': self._safe_get_str(info, 'sector', max_len=100),
                'industry': self._safe_get_str(info, 'industry', max_len=100),
            }

            return fundamentals

        except Exception as e:
            logger.debug(f"Error fetching fundamentals for {ticker}: {e}")
            return None

    def _safe_get(
        self,
        info: Dict,
        key: str,
        multiply: float = None,
        max_val: float = 99999
    ) -> Optional[float]:
        """Safely extract a numeric value from info dict."""
        try:
            val = info.get(key)
            if val is None or val == 'Infinity' or val == '-Infinity':
                return None
            val = float(val)
            if multiply:
                val = val * multiply
            # Cap extreme values to avoid database overflow
            if val > max_val:
                val = max_val
            elif val < -max_val:
                val = -max_val
            return round(val, 4)
        except (ValueError, TypeError):
            return None

    def _safe_get_int(self, info: Dict, key: str) -> Optional[int]:
        """Safely extract an integer value from info dict."""
        try:
            val = info.get(key)
            if val is None:
                return None
            return int(val)
        except (ValueError, TypeError):
            return None

    def _safe_get_str(
        self,
        info: Dict,
        key: str,
        max_len: int = None
    ) -> Optional[str]:
        """Safely extract a string value from info dict."""
        try:
            val = info.get(key)
            if val is None:
                return None
            val = str(val)
            if max_len and len(val) > max_len:
                val = val[:max_len]
            return val
        except Exception:
            return None

    def _format_market_cap(self, value) -> Optional[str]:
        """Format market cap as human-readable string (e.g., '150.5B', '2.3T')."""
        if value is None:
            return None
        try:
            val = float(value)
            if val >= 1e12:
                return f"{val / 1e12:.1f}T"
            elif val >= 1e9:
                return f"{val / 1e9:.1f}B"
            elif val >= 1e6:
                return f"{val / 1e6:.1f}M"
            else:
                return f"{val:.0f}"
        except (ValueError, TypeError):
            return None

    async def fetch_fundamentals(self, ticker: str) -> Optional[Dict]:
        """
        Fetch fundamentals for a single ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Dict with fundamental data or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_fundamentals_sync,
            ticker
        )

    async def save_fundamentals(self, fundamentals: Dict) -> bool:
        """
        Save fundamentals to database.

        Args:
            fundamentals: Dict with fundamental data

        Returns:
            True if successful
        """
        query = """
            UPDATE stock_instruments SET
                -- Earnings & Calendar
                earnings_date = $2,
                earnings_date_estimated = $3,
                ex_dividend_date = $4,

                -- Risk Metrics
                beta = $5,
                short_ratio = $6,
                short_percent_float = $7,
                shares_short = $8,

                -- Ownership
                institutional_percent = $9,
                insider_percent = $10,
                shares_outstanding = $11,
                shares_float = $12,

                -- Valuation Metrics
                forward_pe = $13,
                trailing_pe = $14,
                price_to_book = $15,
                price_to_sales = $16,
                peg_ratio = $17,
                enterprise_to_ebitda = $18,
                enterprise_to_revenue = $19,
                enterprise_value = $20,

                -- Growth Metrics
                revenue_growth = $21,
                earnings_growth = $22,
                earnings_quarterly_growth = $23,

                -- Profitability Metrics
                profit_margin = $24,
                operating_margin = $25,
                gross_margin = $26,
                return_on_equity = $27,
                return_on_assets = $28,

                -- Financial Health
                debt_to_equity = $29,
                current_ratio = $30,
                quick_ratio = $31,

                -- Dividend Info
                dividend_yield = $32,
                dividend_rate = $33,
                payout_ratio = $34,

                -- 52-Week Data
                fifty_two_week_high = $35,
                fifty_two_week_low = $36,
                fifty_two_week_change = $37,

                -- Moving Averages
                fifty_day_average = $38,
                two_hundred_day_average = $39,

                -- Analyst Data
                analyst_rating = $40,
                target_price = $41,
                target_high = $42,
                target_low = $43,
                number_of_analysts = $44,

                -- Company Info
                business_summary = $45,
                website = $46,
                country = $47,
                city = $48,
                employee_count = $49,

                -- Sector/Industry (update if available)
                sector = COALESCE($50, sector),
                industry = COALESCE($51, industry),

                -- Market Cap
                market_cap = COALESCE($52, market_cap),

                -- Timestamps
                fundamentals_updated_at = NOW(),
                updated_at = NOW()
            WHERE ticker = $1
        """

        try:
            await self.db.execute(
                query,
                fundamentals['ticker'],

                # Earnings & Calendar
                fundamentals['earnings_date'],
                fundamentals['earnings_date_estimated'],
                fundamentals['ex_dividend_date'],

                # Risk Metrics
                fundamentals['beta'],
                fundamentals['short_ratio'],
                fundamentals['short_percent_float'],
                fundamentals['shares_short'],

                # Ownership
                fundamentals['institutional_percent'],
                fundamentals['insider_percent'],
                fundamentals['shares_outstanding'],
                fundamentals['shares_float'],

                # Valuation Metrics
                fundamentals['forward_pe'],
                fundamentals['trailing_pe'],
                fundamentals['price_to_book'],
                fundamentals['price_to_sales'],
                fundamentals['peg_ratio'],
                fundamentals['enterprise_to_ebitda'],
                fundamentals['enterprise_to_revenue'],
                fundamentals['enterprise_value'],

                # Growth Metrics
                fundamentals['revenue_growth'],
                fundamentals['earnings_growth'],
                fundamentals['earnings_quarterly_growth'],

                # Profitability Metrics
                fundamentals['profit_margin'],
                fundamentals['operating_margin'],
                fundamentals['gross_margin'],
                fundamentals['return_on_equity'],
                fundamentals['return_on_assets'],

                # Financial Health
                fundamentals['debt_to_equity'],
                fundamentals['current_ratio'],
                fundamentals['quick_ratio'],

                # Dividend Info
                fundamentals['dividend_yield'],
                fundamentals['dividend_rate'],
                fundamentals['payout_ratio'],

                # 52-Week Data
                fundamentals['fifty_two_week_high'],
                fundamentals['fifty_two_week_low'],
                fundamentals['fifty_two_week_change'],

                # Moving Averages
                fundamentals['fifty_day_average'],
                fundamentals['two_hundred_day_average'],

                # Analyst Data
                fundamentals['analyst_rating'],
                fundamentals['target_price'],
                fundamentals['target_high'],
                fundamentals['target_low'],
                fundamentals['number_of_analysts'],

                # Company Info
                fundamentals['business_summary'],
                fundamentals['website'],
                fundamentals['country'],
                fundamentals['city'],
                fundamentals['employee_count'],

                # Sector/Industry
                fundamentals['sector'],
                fundamentals['industry'],

                # Market Cap
                fundamentals['market_cap'],
            )
            return True
        except Exception as e:
            logger.error(f"Error saving fundamentals for {fundamentals['ticker']}: {e}")
            return False

    async def get_tickers_needing_update(self, days_stale: int = 7) -> List[str]:
        """
        Get tickers that need fundamentals update.

        Args:
            days_stale: Number of days before data is considered stale

        Returns:
            List of tickers needing update
        """
        query = """
            SELECT ticker FROM stock_instruments
            WHERE is_active = TRUE AND is_tradeable = TRUE
            AND (fundamentals_updated_at IS NULL
                 OR fundamentals_updated_at < NOW() - INTERVAL '%s days')
            ORDER BY ticker
        """

        rows = await self.db.fetch(query % days_stale)
        return [row['ticker'] for row in rows]

    async def fetch_all_fundamentals(
        self,
        concurrency: int = 10,
        force: bool = False,
        tickers: List[str] = None
    ) -> Dict:
        """
        Fetch fundamentals for all stocks.

        Args:
            concurrency: Max concurrent fetches
            force: If True, update all stocks regardless of age
            tickers: Optional list of specific tickers to update

        Returns:
            Stats dict with detailed breakdown
        """
        if tickers:
            tickers_to_update = tickers
        elif force:
            query = """
                SELECT ticker FROM stock_instruments
                WHERE is_active = TRUE AND is_tradeable = TRUE
                ORDER BY ticker
            """
            rows = await self.db.fetch(query)
            tickers_to_update = [row['ticker'] for row in rows]
        else:
            tickers_to_update = await self.get_tickers_needing_update()

        if not tickers_to_update:
            logger.info("All fundamentals are up to date")
            return self._empty_stats()

        logger.info(f"Fetching fundamentals for {len(tickers_to_update)} stocks...")

        semaphore = asyncio.Semaphore(concurrency)
        stats = self._empty_stats()
        stats['total'] = len(tickers_to_update)

        async def fetch_and_save(ticker: str) -> Tuple[str, bool, Optional[Dict]]:
            async with semaphore:
                try:
                    fundamentals = await self.fetch_fundamentals(ticker)
                    if fundamentals:
                        success = await self.save_fundamentals(fundamentals)
                        return (ticker, success, fundamentals)
                    return (ticker, False, None)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    return (ticker, False, None)

        tasks = [fetch_and_save(t) for t in tickers_to_update]
        results = await asyncio.gather(*tasks)

        for ticker, success, fundamentals in results:
            if success and fundamentals:
                stats['successful'] += 1
                self._update_stats(stats, fundamentals)
            else:
                stats['failed'] += 1

        logger.info(
            f"Fundamentals sync complete: {stats['successful']}/{stats['total']} successful\n"
            f"  - With earnings: {stats['with_earnings']}\n"
            f"  - With beta: {stats['with_beta']}\n"
            f"  - With short interest: {stats['with_short_interest']}\n"
            f"  - With growth data: {stats['with_growth']}\n"
            f"  - With profitability: {stats['with_profitability']}\n"
            f"  - With analyst ratings: {stats['with_analyst_rating']}"
        )

        return stats

    def _empty_stats(self) -> Dict:
        """Return empty stats dict."""
        return {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'with_earnings': 0,
            'with_beta': 0,
            'with_short_interest': 0,
            'with_growth': 0,
            'with_profitability': 0,
            'with_valuation': 0,
            'with_analyst_rating': 0,
            'with_dividend': 0,
            'with_52w_data': 0,
        }

    def _update_stats(self, stats: Dict, fundamentals: Dict) -> None:
        """Update stats based on fundamentals data."""
        if fundamentals.get('earnings_date'):
            stats['with_earnings'] += 1
        if fundamentals.get('beta'):
            stats['with_beta'] += 1
        if fundamentals.get('short_ratio') or fundamentals.get('short_percent_float'):
            stats['with_short_interest'] += 1
        if fundamentals.get('revenue_growth') or fundamentals.get('earnings_growth'):
            stats['with_growth'] += 1
        if fundamentals.get('profit_margin') or fundamentals.get('return_on_equity'):
            stats['with_profitability'] += 1
        if fundamentals.get('trailing_pe') or fundamentals.get('price_to_book'):
            stats['with_valuation'] += 1
        if fundamentals.get('analyst_rating'):
            stats['with_analyst_rating'] += 1
        if fundamentals.get('dividend_yield'):
            stats['with_dividend'] += 1
        if fundamentals.get('fifty_two_week_high'):
            stats['with_52w_data'] += 1

    async def run_fundamentals_pipeline(self) -> Dict:
        """
        Run the fundamentals pipeline (for weekly sync).

        Returns:
            Stats dict
        """
        logger.info("Starting fundamentals pipeline...")
        return await self.fetch_all_fundamentals(concurrency=10)

    # =========================================================================
    # QUERY METHODS FOR FUNDAMENTAL SCREENING
    # =========================================================================

    async def get_upcoming_earnings(self, days_ahead: int = 14) -> List[Dict]:
        """
        Get stocks with earnings in the next N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of stocks with upcoming earnings
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.earnings_date,
                i.earnings_date_estimated,
                i.analyst_rating,
                m.current_price,
                m.trend_strength
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.earnings_date IS NOT NULL
            AND i.earnings_date >= CURRENT_DATE
            AND i.earnings_date <= CURRENT_DATE + INTERVAL '%s days'
            AND i.is_active = TRUE
            ORDER BY i.earnings_date
        """

        rows = await self.db.fetch(query % days_ahead)
        return [dict(row) for row in rows]

    async def get_high_short_interest(self, min_short_percent: float = 15.0) -> List[Dict]:
        """
        Get stocks with high short interest (squeeze potential).

        Args:
            min_short_percent: Minimum short percent of float

        Returns:
            List of stocks with high short interest
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.short_ratio,
                i.short_percent_float,
                i.shares_short,
                i.shares_float,
                m.current_price,
                m.relative_volume,
                m.trend_strength
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.short_percent_float >= $1
            AND i.is_active = TRUE
            ORDER BY i.short_percent_float DESC
        """

        rows = await self.db.fetch(query, min_short_percent)
        return [dict(row) for row in rows]

    async def get_growth_stocks(
        self,
        min_revenue_growth: float = 20.0,
        min_earnings_growth: float = 15.0
    ) -> List[Dict]:
        """
        Get high-growth stocks.

        Args:
            min_revenue_growth: Minimum revenue growth %
            min_earnings_growth: Minimum earnings growth %

        Returns:
            List of growth stocks
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.revenue_growth,
                i.earnings_growth,
                i.peg_ratio,
                i.forward_pe,
                i.analyst_rating,
                m.current_price,
                m.trend_strength
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE (i.revenue_growth >= $1 OR i.earnings_growth >= $2)
            AND i.is_active = TRUE
            ORDER BY COALESCE(i.earnings_growth, 0) + COALESCE(i.revenue_growth, 0) DESC
        """

        rows = await self.db.fetch(query, min_revenue_growth, min_earnings_growth)
        return [dict(row) for row in rows]

    async def get_value_stocks(
        self,
        max_pe: float = 15.0,
        max_pb: float = 2.0,
        min_dividend_yield: float = 2.0
    ) -> List[Dict]:
        """
        Get value stocks with low valuations.

        Args:
            max_pe: Maximum P/E ratio
            max_pb: Maximum P/B ratio
            min_dividend_yield: Minimum dividend yield %

        Returns:
            List of value stocks
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.trailing_pe,
                i.price_to_book,
                i.dividend_yield,
                i.payout_ratio,
                i.return_on_equity,
                i.debt_to_equity,
                m.current_price,
                m.trend_strength
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.trailing_pe IS NOT NULL
            AND i.trailing_pe > 0
            AND i.trailing_pe <= $1
            AND (i.price_to_book IS NULL OR i.price_to_book <= $2)
            AND (i.dividend_yield IS NULL OR i.dividend_yield >= $3)
            AND i.is_active = TRUE
            ORDER BY i.trailing_pe
        """

        rows = await self.db.fetch(query, max_pe, max_pb, min_dividend_yield)
        return [dict(row) for row in rows]

    async def get_quality_stocks(
        self,
        min_roe: float = 15.0,
        max_debt_to_equity: float = 100.0,
        min_profit_margin: float = 10.0
    ) -> List[Dict]:
        """
        Get high-quality stocks with strong fundamentals.

        Args:
            min_roe: Minimum return on equity %
            max_debt_to_equity: Maximum debt/equity ratio
            min_profit_margin: Minimum profit margin %

        Returns:
            List of quality stocks
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.return_on_equity,
                i.return_on_assets,
                i.profit_margin,
                i.operating_margin,
                i.debt_to_equity,
                i.current_ratio,
                i.analyst_rating,
                m.current_price,
                m.trend_strength
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.return_on_equity >= $1
            AND (i.debt_to_equity IS NULL OR i.debt_to_equity <= $2)
            AND (i.profit_margin IS NULL OR i.profit_margin >= $3)
            AND i.is_active = TRUE
            ORDER BY i.return_on_equity DESC
        """

        rows = await self.db.fetch(query, min_roe, max_debt_to_equity, min_profit_margin)
        return [dict(row) for row in rows]

    async def get_dividend_stocks(
        self,
        min_yield: float = 3.0,
        max_payout_ratio: float = 80.0
    ) -> List[Dict]:
        """
        Get dividend stocks with sustainable yields.

        Args:
            min_yield: Minimum dividend yield %
            max_payout_ratio: Maximum payout ratio % (sustainability)

        Returns:
            List of dividend stocks
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.dividend_yield,
                i.dividend_rate,
                i.payout_ratio,
                i.ex_dividend_date,
                i.profit_margin,
                i.debt_to_equity,
                m.current_price,
                m.trend_strength
            FROM stock_instruments i
            LEFT JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.dividend_yield >= $1
            AND (i.payout_ratio IS NULL OR i.payout_ratio <= $2)
            AND i.is_active = TRUE
            ORDER BY i.dividend_yield DESC
        """

        rows = await self.db.fetch(query, min_yield, max_payout_ratio)
        return [dict(row) for row in rows]

    async def get_near_52w_high(self, within_percent: float = 5.0) -> List[Dict]:
        """
        Get stocks near their 52-week high.

        Args:
            within_percent: Within X% of 52-week high

        Returns:
            List of stocks near 52-week high
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.fifty_two_week_high,
                i.fifty_two_week_low,
                m.current_price,
                ROUND(((m.current_price / i.fifty_two_week_high - 1) * 100)::numeric, 2) as pct_from_high,
                m.relative_volume,
                m.trend_strength
            FROM stock_instruments i
            JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.fifty_two_week_high IS NOT NULL
            AND i.fifty_two_week_high > 0
            AND m.current_price >= i.fifty_two_week_high * (1 - $1 / 100)
            AND i.is_active = TRUE
            ORDER BY (m.current_price / i.fifty_two_week_high) DESC
        """

        rows = await self.db.fetch(query, within_percent)
        return [dict(row) for row in rows]

    async def get_near_52w_low(self, within_percent: float = 10.0) -> List[Dict]:
        """
        Get stocks near their 52-week low (potential value or falling knife).

        Args:
            within_percent: Within X% of 52-week low

        Returns:
            List of stocks near 52-week low
        """
        query = """
            SELECT
                i.ticker,
                i.name,
                i.sector,
                i.fifty_two_week_high,
                i.fifty_two_week_low,
                m.current_price,
                ROUND(((m.current_price / i.fifty_two_week_low - 1) * 100)::numeric, 2) as pct_from_low,
                i.trailing_pe,
                i.return_on_equity,
                m.rsi_14,
                m.trend_strength
            FROM stock_instruments i
            JOIN stock_screening_metrics m ON i.ticker = m.ticker
                AND m.calculation_date = (SELECT MAX(calculation_date) FROM stock_screening_metrics)
            WHERE i.fifty_two_week_low IS NOT NULL
            AND i.fifty_two_week_low > 0
            AND m.current_price <= i.fifty_two_week_low * (1 + $1 / 100)
            AND i.is_active = TRUE
            ORDER BY (m.current_price / i.fifty_two_week_low)
        """

        rows = await self.db.fetch(query, within_percent)
        return [dict(row) for row in rows]

    async def close(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
