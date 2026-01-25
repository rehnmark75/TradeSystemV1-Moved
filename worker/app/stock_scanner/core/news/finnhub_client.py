"""
Finnhub API Client for News Data

Fetches company news from Finnhub's free tier API.
Rate limited to 60 requests/minute (free tier).

API Documentation: https://finnhub.io/docs/api/company-news
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article from Finnhub"""
    headline: str
    summary: str
    source: str
    url: str
    published_at: datetime
    category: str
    ticker: str
    image_url: Optional[str] = None
    finnhub_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "category": self.category,
            "ticker": self.ticker,
            "image_url": self.image_url,
            "finnhub_id": self.finnhub_id,
        }


class FinnhubError(Exception):
    """Base exception for Finnhub API errors"""
    def __init__(self, message: str, code: str = None, response: Dict = None):
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class FinnhubRateLimitError(FinnhubError):
    """Rate limit exceeded error"""
    pass


class FinnhubAuthError(FinnhubError):
    """Authentication error - invalid API key"""
    pass


class FinnhubClient:
    """
    Async client for Finnhub API (Free Tier)

    Features:
    - Rate limiting (30 req/min to stay under 60/min limit)
    - Automatic retry with exponential backoff
    - Response caching
    - Graceful error handling

    Usage:
        client = FinnhubClient(api_key="your_api_key")
        async with client:
            news = await client.get_company_news("AAPL")
    """

    BASE_URL = "https://finnhub.io/api/v1"

    # Rate limiting: 30 req/min (50% of free tier limit)
    MAX_REQUESTS_PER_MINUTE = 30

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        max_requests_per_minute: int = None,
    ):
        """
        Initialize Finnhub client

        Args:
            api_key: Finnhub API key
            base_url: Optional custom base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            max_requests_per_minute: Rate limit (default 30)
        """
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.max_requests_per_minute = max_requests_per_minute or self.MAX_REQUESTS_PER_MINUTE

        self._session: Optional[aiohttp.ClientSession] = None
        self._request_timestamps: deque = deque(maxlen=60)  # Track last 60 requests
        self._cache: Dict[str, tuple] = {}  # {cache_key: (data, timestamp)}
        self._cache_ttl = 3600  # 1 hour cache for news

    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "X-Finnhub-Token": self.api_key,
            "Accept": "application/json",
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _create_session(self):
        """Create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers
            )

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _wait_for_rate_limit(self):
        """
        Wait if necessary to respect rate limits

        Uses a sliding window to track requests per minute.
        """
        now = time.time()

        # Remove timestamps older than 60 seconds
        while self._request_timestamps and (now - self._request_timestamps[0]) > 60:
            self._request_timestamps.popleft()

        # If at limit, wait
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self._request_timestamps[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record this request
        self._request_timestamps.append(time.time())

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key for request"""
        params_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{endpoint}?{params_str}"

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached response if valid"""
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return data
            else:
                del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Any):
        """Cache response data"""
        self._cache[cache_key] = (data, time.time())

    async def _request(
        self,
        endpoint: str,
        params: Dict = None,
        retry_count: int = 0,
        use_cache: bool = True,
    ) -> Any:
        """
        Make an API request with rate limiting and error handling

        Args:
            endpoint: API endpoint path
            params: Query parameters
            retry_count: Current retry attempt
            use_cache: Whether to use cached response

        Returns:
            API response data

        Raises:
            FinnhubError: On API errors
            FinnhubRateLimitError: On rate limit exceeded
            FinnhubAuthError: On authentication errors
        """
        params = params or {}

        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached

        await self._create_session()
        await self._wait_for_rate_limit()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self._session.get(url, params=params) as response:
                # Handle rate limiting
                if response.status == 429:
                    if retry_count < self.max_retries:
                        wait_time = 60  # Wait a full minute on rate limit
                        logger.warning(f"Rate limit hit (429), waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        return await self._request(endpoint, params, retry_count + 1, use_cache)
                    raise FinnhubRateLimitError(
                        "Rate limit exceeded after retries",
                        code="rate_limit",
                    )

                # Handle auth errors
                if response.status == 401 or response.status == 403:
                    raise FinnhubAuthError(
                        "Authentication failed. Check API key.",
                        code="auth_error",
                    )

                # Handle other errors
                if response.status >= 400:
                    error_text = await response.text()
                    raise FinnhubError(
                        f"API error ({response.status}): {error_text}",
                        code=str(response.status),
                    )

                data = await response.json()

                # Cache successful response
                if use_cache:
                    self._set_cache(cache_key, data)

                return data

        except aiohttp.ClientError as e:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Connection error, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                return await self._request(endpoint, params, retry_count + 1, use_cache)
            raise FinnhubError(f"Connection error after retries: {str(e)}")

    async def get_company_news(
        self,
        symbol: str,
        from_date: datetime = None,
        to_date: datetime = None,
        use_cache: bool = True,
    ) -> List[NewsArticle]:
        """
        Get company news articles

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            from_date: Start date (default: 7 days ago)
            to_date: End date (default: today)
            use_cache: Whether to use cached results

        Returns:
            List of NewsArticle objects
        """
        # Default to last 7 days
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=7)

        params = {
            "symbol": symbol.upper(),
            "from": from_date.strftime("%Y-%m-%d"),
            "to": to_date.strftime("%Y-%m-%d"),
        }

        try:
            data = await self._request("/company-news", params, use_cache=use_cache)

            articles = []
            for item in data:
                try:
                    article = NewsArticle(
                        headline=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        source=item.get("source", ""),
                        url=item.get("url", ""),
                        published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                        category=item.get("category", ""),
                        ticker=symbol.upper(),
                        image_url=item.get("image"),
                        finnhub_id=item.get("id"),
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse news article: {e}")
                    continue

            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles

        except FinnhubRateLimitError:
            logger.warning(f"Rate limit hit fetching news for {symbol}")
            raise
        except FinnhubError as e:
            logger.error(f"Failed to fetch news for {symbol}: {e.message}")
            raise

    async def get_market_news(
        self,
        category: str = "general",
        min_id: int = 0,
    ) -> List[NewsArticle]:
        """
        Get general market news

        Args:
            category: News category (general, forex, crypto, merger)
            min_id: Get news with ID greater than this

        Returns:
            List of NewsArticle objects
        """
        params = {
            "category": category,
        }
        if min_id > 0:
            params["minId"] = min_id

        try:
            data = await self._request("/news", params)

            articles = []
            for item in data:
                try:
                    article = NewsArticle(
                        headline=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        source=item.get("source", ""),
                        url=item.get("url", ""),
                        published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                        category=item.get("category", category),
                        ticker="MARKET",  # General market news
                        image_url=item.get("image"),
                        finnhub_id=item.get("id"),
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse market news: {e}")
                    continue

            return articles

        except FinnhubError as e:
            logger.error(f"Failed to fetch market news: {e.message}")
            raise

    async def test_connection(self) -> bool:
        """
        Test API connection and authentication

        Returns:
            True if connection successful
        """
        try:
            # Use market news as a simple test endpoint
            await self._request("/news", {"category": "general"}, use_cache=False)
            logger.info("Finnhub API connection successful")
            return True
        except FinnhubAuthError:
            logger.error("Finnhub API authentication failed - check API key")
            return False
        except FinnhubError as e:
            logger.error(f"Finnhub API connection failed: {e.message}")
            return False

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status

        Returns:
            Dict with rate limit info
        """
        now = time.time()

        # Count requests in last minute
        recent_requests = sum(
            1 for ts in self._request_timestamps
            if (now - ts) <= 60
        )

        return {
            "requests_last_minute": recent_requests,
            "max_per_minute": self.max_requests_per_minute,
            "remaining": max(0, self.max_requests_per_minute - recent_requests),
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        logger.info("Finnhub cache cleared")

    # =========================================================================
    # MARKET STATUS ENDPOINTS
    # =========================================================================

    async def get_market_status(
        self,
        exchange: str = "US",
    ) -> Dict[str, Any]:
        """
        Get current market status from Finnhub.

        API: /stock/market-status

        Args:
            exchange: Exchange code (US, LSE, etc.)

        Returns:
            Dict with:
            - exchange: Exchange code
            - holiday: Holiday name if applicable
            - isOpen: Boolean market open status
            - session: Current session type:
                - "pre-market": 4:00 AM - 9:30 AM ET
                - "regular": 9:30 AM - 4:00 PM ET
                - "post-market": 4:00 PM - 8:00 PM ET
                - null: Market closed
            - timezone: Market timezone
            - t: Unix timestamp
        """
        params = {"exchange": exchange}

        try:
            data = await self._request("/stock/market-status", params, use_cache=False)
            logger.info(f"Market status for {exchange}: {data.get('session', 'closed')}")
            return data
        except FinnhubError as e:
            logger.error(f"Failed to get market status: {e.message}")
            raise

    async def is_pre_market(self, exchange: str = "US") -> bool:
        """
        Check if market is currently in pre-market session.

        Returns:
            True if in pre-market (4:00 AM - 9:30 AM ET)
        """
        try:
            status = await self.get_market_status(exchange)
            return status.get("session") == "pre-market"
        except FinnhubError:
            return False

    async def is_market_open(self, exchange: str = "US") -> bool:
        """
        Check if market is in regular trading hours.

        Returns:
            True if in regular session (9:30 AM - 4:00 PM ET)
        """
        try:
            status = await self.get_market_status(exchange)
            return status.get("session") == "regular"
        except FinnhubError:
            return False

    # =========================================================================
    # QUOTE ENDPOINTS
    # =========================================================================

    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a symbol.

        API: /quote

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with:
            - c: Current price
            - d: Change
            - dp: Percent change
            - h: High price of day
            - l: Low price of day
            - o: Open price of day
            - pc: Previous close price
            - t: Timestamp
        """
        params = {"symbol": symbol.upper()}

        try:
            data = await self._request("/quote", params, use_cache=False)
            return {
                "symbol": symbol.upper(),
                "current_price": data.get("c"),
                "change": data.get("d"),
                "change_percent": data.get("dp"),
                "high": data.get("h"),
                "low": data.get("l"),
                "open": data.get("o"),
                "previous_close": data.get("pc"),
                "timestamp": data.get("t"),
            }
        except FinnhubError as e:
            logger.error(f"Failed to get quote for {symbol}: {e.message}")
            raise

    # =========================================================================
    # ANALYST RECOMMENDATIONS
    # =========================================================================

    async def get_recommendation_trends(self, symbol: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get analyst recommendation trends for a symbol.

        API: /stock/recommendation

        Returns:
            List of dicts with keys:
            - period (YYYY-MM-DD)
            - strongBuy, buy, hold, sell, strongSell
        """
        params = {"symbol": symbol.upper()}

        try:
            data = await self._request("/stock/recommendation", params, use_cache=use_cache)
            if not isinstance(data, list):
                return []
            return data
        except FinnhubError as e:
            logger.error(f"Failed to fetch recommendations for {symbol}: {e.message}")
            raise

    async def get_quotes_batch(
        self,
        symbols: List[str],
        delay_between: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get quotes for multiple symbols with rate limiting.

        Args:
            symbols: List of ticker symbols
            delay_between: Delay between requests (seconds)

        Returns:
            List of quote dicts
        """
        quotes = []
        for symbol in symbols:
            try:
                quote = await self.get_quote(symbol)
                quotes.append(quote)
                await asyncio.sleep(delay_between)
            except FinnhubError as e:
                logger.warning(f"Failed to get quote for {symbol}: {e.message}")
                quotes.append({
                    "symbol": symbol,
                    "error": str(e.message),
                })
        return quotes

    # =========================================================================
    # NEWS SENTIMENT ENDPOINT (Premium)
    # =========================================================================

    async def get_news_sentiment(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get news sentiment for a symbol (Premium feature).

        API: /news-sentiment

        Note: This is a premium endpoint. Free tier will return 403.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with sentiment data including:
            - buzz: Social media buzz metrics
            - companyNewsScore: News sentiment score
            - sectorAverageBullishPercent: Sector comparison
            - sentiment: Overall sentiment data
        """
        params = {"symbol": symbol.upper()}

        try:
            data = await self._request("/news-sentiment", params)
            return data
        except FinnhubAuthError:
            logger.warning(f"News sentiment requires premium API for {symbol}")
            return {"error": "premium_required", "symbol": symbol}
        except FinnhubError as e:
            logger.error(f"Failed to get news sentiment for {symbol}: {e.message}")
            raise
