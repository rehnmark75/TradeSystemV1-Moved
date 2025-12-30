"""
News Enrichment Service

Provides news sentiment enrichment for stock signals.
Handles:
- Automatic enrichment for high-quality (A/A+) signals
- Manual enrichment on demand
- Caching and rate limiting
- Database storage

Usage:
    service = NewsEnrichmentService(db_manager, finnhub_client)
    result = await service.enrich_signal(signal_id)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .finnhub_client import FinnhubClient, FinnhubError, FinnhubRateLimitError, NewsArticle
from .sentiment_analyzer import NewsSentimentAnalyzer, SentimentResult, SentimentLevel

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Result of news enrichment operation"""
    success: bool
    signal_id: int
    ticker: str
    sentiment: Optional[SentimentResult]
    articles_count: int
    error_message: Optional[str] = None
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "signal_id": self.signal_id,
            "ticker": self.ticker,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "articles_count": self.articles_count,
            "error_message": self.error_message,
            "from_cache": self.from_cache,
        }


class NewsEnrichmentService:
    """
    Service for enriching stock signals with news sentiment.

    Features:
    - Automatic enrichment for A/A+ quality signals
    - Manual enrichment for any signal
    - News caching to reduce API calls
    - Rate limit handling with graceful degradation
    - Database persistence
    """

    def __init__(
        self,
        db_manager,
        finnhub_api_key: str,
        lookback_days: int = 7,
        cache_ttl_hours: int = 1,
        min_articles: int = 2,
    ):
        """
        Initialize the news enrichment service.

        Args:
            db_manager: Database manager instance
            finnhub_api_key: Finnhub API key
            lookback_days: Days of news to fetch (default 7)
            cache_ttl_hours: Cache TTL in hours (default 1)
            min_articles: Minimum articles for confident sentiment
        """
        self.db = db_manager
        self.lookback_days = lookback_days
        self.cache_ttl_hours = cache_ttl_hours
        self.min_articles = min_articles

        # Initialize Finnhub client
        self.finnhub = FinnhubClient(api_key=finnhub_api_key)

        # Initialize sentiment analyzer
        self.analyzer = NewsSentimentAnalyzer(min_articles=min_articles)

        # Track enrichment queue for retry
        self._retry_queue: List[Tuple[int, str]] = []  # (signal_id, ticker)

    async def enrich_signal(
        self,
        signal_id: int,
        ticker: str,
        force_refresh: bool = False,
    ) -> EnrichmentResult:
        """
        Enrich a single signal with news sentiment.

        Args:
            signal_id: Signal ID to enrich
            ticker: Stock ticker symbol
            force_refresh: Force fetch even if cache valid

        Returns:
            EnrichmentResult with sentiment data
        """
        try:
            # Check cache first (unless force refresh)
            if not force_refresh:
                cached = await self._get_cached_sentiment(ticker)
                if cached:
                    # Update signal with cached sentiment
                    await self._update_signal_sentiment(signal_id, cached)
                    return EnrichmentResult(
                        success=True,
                        signal_id=signal_id,
                        ticker=ticker,
                        sentiment=cached,
                        articles_count=cached.articles_count,
                        from_cache=True,
                    )

            # Fetch news from Finnhub
            articles = await self._fetch_news(ticker)

            if not articles:
                return EnrichmentResult(
                    success=True,
                    signal_id=signal_id,
                    ticker=ticker,
                    sentiment=None,
                    articles_count=0,
                    error_message="No news articles found",
                )

            # Analyze sentiment
            sentiment = self.analyzer.analyze_articles(articles)

            # Cache the results
            await self._cache_news(ticker, articles, sentiment)

            # Update signal in database
            await self._update_signal_sentiment(signal_id, sentiment)

            # Log the fetch
            await self._log_fetch(ticker, len(articles), True)

            logger.info(
                f"Enriched signal {signal_id} ({ticker}): "
                f"{sentiment.level.value} ({sentiment.score:.2f}) "
                f"from {len(articles)} articles"
            )

            return EnrichmentResult(
                success=True,
                signal_id=signal_id,
                ticker=ticker,
                sentiment=sentiment,
                articles_count=len(articles),
            )

        except FinnhubRateLimitError:
            # Queue for retry later
            self._retry_queue.append((signal_id, ticker))
            logger.warning(f"Rate limited enriching {ticker}, queued for retry")

            return EnrichmentResult(
                success=False,
                signal_id=signal_id,
                ticker=ticker,
                sentiment=None,
                articles_count=0,
                error_message="Rate limited - queued for retry",
            )

        except FinnhubError as e:
            await self._log_fetch(ticker, 0, False, str(e))
            logger.error(f"Failed to enrich {ticker}: {e.message}")

            return EnrichmentResult(
                success=False,
                signal_id=signal_id,
                ticker=ticker,
                sentiment=None,
                articles_count=0,
                error_message=str(e.message),
            )

        except Exception as e:
            logger.exception(f"Unexpected error enriching {ticker}")

            return EnrichmentResult(
                success=False,
                signal_id=signal_id,
                ticker=ticker,
                sentiment=None,
                articles_count=0,
                error_message=f"Unexpected error: {str(e)}",
            )

    async def enrich_high_quality_signals(
        self,
        signals: List[Dict[str, Any]],
    ) -> List[EnrichmentResult]:
        """
        Automatically enrich A+ and A quality signals.

        Args:
            signals: List of signal dicts with id, ticker, quality_tier

        Returns:
            List of EnrichmentResults
        """
        results = []

        # Filter to A+ and A signals only
        high_quality = [
            s for s in signals
            if s.get('quality_tier') in ('A+', 'A')
        ]

        if not high_quality:
            logger.info("No A+ or A signals to enrich")
            return results

        logger.info(f"Auto-enriching {len(high_quality)} high-quality signals")

        # Process sequentially to respect rate limits
        for signal in high_quality:
            result = await self.enrich_signal(
                signal_id=signal['id'],
                ticker=signal['ticker'],
            )
            results.append(result)

            # Small delay between requests
            await asyncio.sleep(0.5)

        return results

    async def process_retry_queue(self) -> List[EnrichmentResult]:
        """
        Process any signals that were queued due to rate limiting.

        Should be called periodically (e.g., every 60 seconds).

        Returns:
            List of EnrichmentResults from retried signals
        """
        if not self._retry_queue:
            return []

        logger.info(f"Processing {len(self._retry_queue)} queued signals")

        results = []
        # Take a copy and clear queue
        queue = self._retry_queue.copy()
        self._retry_queue.clear()

        for signal_id, ticker in queue:
            result = await self.enrich_signal(signal_id, ticker)
            results.append(result)

            # If rate limited again, it will be re-queued
            if not result.success and "Rate limited" in (result.error_message or ""):
                break  # Stop processing if we hit rate limit again

            await asyncio.sleep(1.0)  # Longer delay for retries

        return results

    async def _fetch_news(self, ticker: str) -> List[NewsArticle]:
        """Fetch news articles from Finnhub"""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=self.lookback_days)

        async with self.finnhub:
            articles = await self.finnhub.get_company_news(
                symbol=ticker,
                from_date=from_date,
                to_date=to_date,
            )

        return articles

    async def _get_cached_sentiment(self, ticker: str) -> Optional[SentimentResult]:
        """Get cached sentiment if available and valid"""
        try:
            query = """
                SELECT
                    AVG(sentiment_score) as avg_score,
                    COUNT(*) as article_count,
                    MAX(fetched_at) as last_fetch
                FROM stock_news_cache
                WHERE ticker = $1
                AND expires_at > NOW()
                AND fetched_at > NOW() - INTERVAL '%s hours'
            """ % self.cache_ttl_hours

            row = await self.db.fetchrow(query, ticker)

            if row and row['article_count'] >= self.min_articles:
                # Reconstruct sentiment from cache
                avg_score = float(row['avg_score'] or 0)

                # Classify the score
                if avg_score >= 0.5:
                    level = SentimentLevel.VERY_BULLISH
                elif avg_score >= 0.15:
                    level = SentimentLevel.BULLISH
                elif avg_score <= -0.5:
                    level = SentimentLevel.VERY_BEARISH
                elif avg_score <= -0.15:
                    level = SentimentLevel.BEARISH
                else:
                    level = SentimentLevel.NEUTRAL

                return SentimentResult(
                    score=avg_score,
                    level=level,
                    confidence=0.7,  # Lower confidence for cached
                    articles_count=row['article_count'],
                    factors=[f"From cache ({row['article_count']} articles)"],
                )

            return None

        except Exception as e:
            logger.warning(f"Error checking cache for {ticker}: {e}")
            return None

    async def _cache_news(
        self,
        ticker: str,
        articles: List[NewsArticle],
        sentiment: SentimentResult,
    ):
        """Cache news articles in database"""
        try:
            # Calculate individual sentiment scores and insert
            for article in articles:
                score = self.analyzer.analyze_text(article.headline)

                query = """
                    INSERT INTO stock_news_cache
                    (ticker, headline, summary, source, url, image_url,
                     published_at, category, sentiment_score, finnhub_id,
                     fetched_at, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(),
                            NOW() + INTERVAL '%s hours')
                    ON CONFLICT (ticker, finnhub_id) DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        fetched_at = NOW(),
                        expires_at = NOW() + INTERVAL '%s hours'
                """ % (self.cache_ttl_hours, self.cache_ttl_hours)

                await self.db.execute(
                    query,
                    ticker,
                    article.headline[:500],  # Truncate
                    (article.summary or '')[:2000],
                    article.source[:100] if article.source else None,
                    article.url[:2000] if article.url else None,
                    article.image_url[:500] if article.image_url else None,
                    article.published_at,
                    article.category[:50] if article.category else None,
                    score,
                    article.finnhub_id,
                )

        except Exception as e:
            logger.warning(f"Error caching news for {ticker}: {e}")

    async def _update_signal_sentiment(
        self,
        signal_id: int,
        sentiment: SentimentResult,
    ):
        """Update signal with sentiment data"""
        try:
            query = """
                UPDATE stock_scanner_signals
                SET
                    news_sentiment_score = $1,
                    news_sentiment_level = $2,
                    news_headlines_count = $3,
                    news_factors = $4,
                    news_analyzed_at = NOW(),
                    updated_at = NOW()
                WHERE id = $5
            """

            await self.db.execute(
                query,
                sentiment.score,
                sentiment.level.value,
                sentiment.articles_count,
                sentiment.factors,
                signal_id,
            )

        except Exception as e:
            logger.error(f"Error updating signal {signal_id} with sentiment: {e}")
            raise

    async def _log_fetch(
        self,
        ticker: str,
        articles_fetched: int,
        success: bool,
        error_message: str = None,
    ):
        """Log API fetch for tracking"""
        try:
            query = """
                INSERT INTO stock_news_fetch_log
                (ticker, articles_fetched, success, error_message)
                VALUES ($1, $2, $3, $4)
            """

            await self.db.execute(
                query,
                ticker,
                articles_fetched,
                success,
                error_message,
            )

        except Exception as e:
            logger.warning(f"Error logging fetch for {ticker}: {e}")

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            "finnhub": self.finnhub.get_rate_limit_status(),
            "retry_queue_size": len(self._retry_queue),
        }

    async def cleanup_expired_cache(self) -> int:
        """Clean up expired news cache entries"""
        try:
            result = await self.db.execute(
                "SELECT cleanup_expired_news_cache()"
            )
            deleted = result if isinstance(result, int) else 0
            logger.info(f"Cleaned up {deleted} expired cache entries")
            return deleted
        except Exception as e:
            logger.warning(f"Error cleaning up cache: {e}")
            return 0

    async def get_signal_news(self, signal_id: int) -> Dict[str, Any]:
        """
        Get news data for a signal (for UI display).

        Returns cached articles and sentiment for a signal.
        """
        try:
            # Get signal ticker
            signal_query = """
                SELECT ticker, news_sentiment_score, news_sentiment_level,
                       news_headlines_count, news_factors, news_analyzed_at
                FROM stock_scanner_signals
                WHERE id = $1
            """
            signal = await self.db.fetchrow(signal_query, signal_id)

            if not signal:
                return {"error": "Signal not found"}

            ticker = signal['ticker']

            # Get cached articles
            articles_query = """
                SELECT headline, summary, source, url, published_at,
                       sentiment_score
                FROM stock_news_cache
                WHERE ticker = $1
                AND expires_at > NOW()
                ORDER BY published_at DESC
                LIMIT 10
            """
            articles = await self.db.fetch(articles_query, ticker)

            return {
                "signal_id": signal_id,
                "ticker": ticker,
                "sentiment": {
                    "score": float(signal['news_sentiment_score']) if signal['news_sentiment_score'] else None,
                    "level": signal['news_sentiment_level'],
                    "articles_count": signal['news_headlines_count'],
                    "factors": signal['news_factors'],
                    "analyzed_at": signal['news_analyzed_at'].isoformat() if signal['news_analyzed_at'] else None,
                },
                "articles": [
                    {
                        "headline": a['headline'],
                        "summary": a['summary'],
                        "source": a['source'],
                        "url": a['url'],
                        "published_at": a['published_at'].isoformat() if a['published_at'] else None,
                        "sentiment_score": float(a['sentiment_score']) if a['sentiment_score'] else None,
                    }
                    for a in articles
                ],
            }

        except Exception as e:
            logger.error(f"Error getting news for signal {signal_id}: {e}")
            return {"error": str(e)}
