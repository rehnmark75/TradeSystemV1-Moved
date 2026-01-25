"""
News module for stock scanner

Provides news fetching, sentiment analysis, and pre-market scanning functionality.
"""

from .finnhub_client import FinnhubClient, FinnhubError, FinnhubRateLimitError, NewsArticle
from .sentiment_analyzer import NewsSentimentAnalyzer, SentimentResult, SentimentLevel
from .news_enrichment_service import NewsEnrichmentService, EnrichmentResult
from .premarket_service import (
    PreMarketService,
    PreMarketQuote,
    PreMarketSignal,
    PreMarketScanResult,
    GapType,
)

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
    # Pre-market service
    "PreMarketService",
    "PreMarketQuote",
    "PreMarketSignal",
    "PreMarketScanResult",
    "GapType",
]
