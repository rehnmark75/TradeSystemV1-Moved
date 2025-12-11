"""
News module for stock scanner

Provides news fetching and sentiment analysis functionality.
"""

from .finnhub_client import FinnhubClient, FinnhubError, FinnhubRateLimitError, NewsArticle
from .sentiment_analyzer import NewsSentimentAnalyzer, SentimentResult, SentimentLevel
from .news_enrichment_service import NewsEnrichmentService, EnrichmentResult

__all__ = [
    # Finnhub client
    "FinnhubClient",
    "FinnhubError",
    "FinnhubRateLimitError",
    "NewsArticle",
    # Sentiment analyzer
    "NewsSentimentAnalyzer",
    "SentimentResult",
    "SentimentLevel",
    # Enrichment service
    "NewsEnrichmentService",
    "EnrichmentResult",
]
