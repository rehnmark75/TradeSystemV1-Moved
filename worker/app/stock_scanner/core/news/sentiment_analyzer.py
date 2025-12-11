"""
News Sentiment Analyzer using VADER

Analyzes news headlines and summaries to determine sentiment.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is optimized
for social media and news text.

Features:
- Recency weighting (recent news weighted higher)
- Aggregate sentiment scoring
- Confidence calculation based on article count
- Financial-context adjustments
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from .finnhub_client import NewsArticle

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment classification levels"""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    # Core sentiment score (-1.0 to 1.0)
    score: float

    # Classified sentiment level
    level: SentimentLevel

    # Confidence in the analysis (0.0 to 1.0)
    confidence: float

    # Number of articles analyzed
    articles_count: int

    # Individual article scores for transparency
    article_scores: List[Dict[str, Any]] = field(default_factory=list)

    # Summary factors for confluence
    factors: List[str] = field(default_factory=list)

    # Timestamp of analysis
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "score": round(self.score, 4),
            "level": self.level.value,
            "confidence": round(self.confidence, 4),
            "articles_count": self.articles_count,
            "factors": self.factors,
            "analyzed_at": self.analyzed_at.isoformat(),
        }

    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish"""
        return self.level in (SentimentLevel.BULLISH, SentimentLevel.VERY_BULLISH)

    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish"""
        return self.level in (SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH)

    @property
    def display_label(self) -> str:
        """Get display-friendly label"""
        labels = {
            SentimentLevel.VERY_BULLISH: "Very Bullish",
            SentimentLevel.BULLISH: "Bullish",
            SentimentLevel.NEUTRAL: "Neutral",
            SentimentLevel.BEARISH: "Bearish",
            SentimentLevel.VERY_BEARISH: "Very Bearish",
        }
        return labels.get(self.level, "Unknown")


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment using VADER with financial context

    Features:
    - Recency weighting (recent articles count more)
    - Financial keyword boosting
    - Confidence based on article count and agreement
    - Graceful handling when VADER not installed
    """

    # Sentiment thresholds
    VERY_BULLISH_THRESHOLD = 0.5
    BULLISH_THRESHOLD = 0.15
    BEARISH_THRESHOLD = -0.15
    VERY_BEARISH_THRESHOLD = -0.5

    # Minimum articles for confident analysis
    MIN_ARTICLES_FOR_CONFIDENCE = 2

    # Recency decay (articles lose 10% weight per day)
    RECENCY_DECAY_PER_DAY = 0.1

    # Financial keywords that boost sentiment (positive context)
    POSITIVE_FINANCIAL_KEYWORDS = [
        "beat", "beats", "exceeded", "exceeds", "surpassed",
        "upgrade", "upgraded", "outperform", "buy rating",
        "strong", "growth", "profit", "gains", "rally",
        "breakthrough", "record", "all-time high", "expansion",
        "dividend", "buyback", "acquisition", "partnership",
    ]

    # Financial keywords that indicate negative context
    NEGATIVE_FINANCIAL_KEYWORDS = [
        "miss", "missed", "misses", "below", "downgrade",
        "downgraded", "underperform", "sell rating",
        "weak", "decline", "loss", "losses", "crash",
        "layoffs", "lawsuit", "investigation", "recall",
        "bankruptcy", "default", "warning", "concern",
    ]

    def __init__(self, min_articles: int = None):
        """
        Initialize the sentiment analyzer

        Args:
            min_articles: Minimum articles for confident analysis
        """
        self.min_articles = min_articles or self.MIN_ARTICLES_FOR_CONFIDENCE

        if VADER_AVAILABLE:
            self._analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            self._analyzer = None
            logger.warning(
                "VADER not available. Install with: pip install vaderSentiment"
            )

    def analyze_articles(
        self,
        articles: List[NewsArticle],
        use_recency_weighting: bool = True,
    ) -> SentimentResult:
        """
        Analyze sentiment from a list of news articles

        Args:
            articles: List of NewsArticle objects
            use_recency_weighting: Apply recency decay to older articles

        Returns:
            SentimentResult with aggregated sentiment
        """
        if not articles:
            return SentimentResult(
                score=0.0,
                level=SentimentLevel.NEUTRAL,
                confidence=0.0,
                articles_count=0,
                factors=["No news articles found"],
            )

        if not self._analyzer:
            return self._fallback_analysis(articles)

        article_scores = []
        total_weighted_score = 0.0
        total_weight = 0.0

        now = datetime.now()

        for article in articles:
            # Analyze headline (more important) and summary
            headline_sentiment = self._analyze_text(article.headline)
            summary_sentiment = self._analyze_text(article.summary) if article.summary else 0.0

            # Combined score (headline weighted 70%, summary 30%)
            raw_score = (headline_sentiment * 0.7) + (summary_sentiment * 0.3)

            # Apply financial keyword adjustment
            combined_text = f"{article.headline} {article.summary or ''}"
            keyword_adjustment = self._get_keyword_adjustment(combined_text)
            adjusted_score = max(-1.0, min(1.0, raw_score + keyword_adjustment))

            # Calculate recency weight
            if use_recency_weighting and article.published_at:
                days_old = (now - article.published_at).days
                recency_weight = max(0.1, 1.0 - (days_old * self.RECENCY_DECAY_PER_DAY))
            else:
                recency_weight = 1.0

            article_scores.append({
                "headline": article.headline[:100],  # Truncate for storage
                "raw_score": round(raw_score, 4),
                "adjusted_score": round(adjusted_score, 4),
                "recency_weight": round(recency_weight, 2),
                "source": article.source,
                "published_at": article.published_at.isoformat() if article.published_at else None,
            })

            total_weighted_score += adjusted_score * recency_weight
            total_weight += recency_weight

        # Calculate final aggregated score
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
        else:
            final_score = 0.0

        # Determine sentiment level
        level = self._classify_sentiment(final_score)

        # Calculate confidence
        confidence = self._calculate_confidence(articles, article_scores)

        # Generate factors for confluence display
        factors = self._generate_factors(final_score, level, len(articles), article_scores)

        return SentimentResult(
            score=final_score,
            level=level,
            confidence=confidence,
            articles_count=len(articles),
            article_scores=article_scores,
            factors=factors,
        )

    def _analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text string

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not text or not self._analyzer:
            return 0.0

        try:
            scores = self._analyzer.polarity_scores(text)
            return scores["compound"]  # VADER's compound score (-1 to 1)
        except Exception as e:
            logger.warning(f"Failed to analyze text: {e}")
            return 0.0

    def _get_keyword_adjustment(self, text: str) -> float:
        """
        Adjust sentiment based on financial keywords

        Args:
            text: Combined headline and summary text

        Returns:
            Adjustment value (-0.2 to 0.2)
        """
        text_lower = text.lower()

        positive_count = sum(
            1 for kw in self.POSITIVE_FINANCIAL_KEYWORDS
            if kw in text_lower
        )
        negative_count = sum(
            1 for kw in self.NEGATIVE_FINANCIAL_KEYWORDS
            if kw in text_lower
        )

        # Each keyword adjusts score by 0.05, max 0.2
        adjustment = (positive_count - negative_count) * 0.05
        return max(-0.2, min(0.2, adjustment))

    def _classify_sentiment(self, score: float) -> SentimentLevel:
        """
        Classify score into sentiment level

        Args:
            score: Sentiment score (-1.0 to 1.0)

        Returns:
            SentimentLevel enum
        """
        if score >= self.VERY_BULLISH_THRESHOLD:
            return SentimentLevel.VERY_BULLISH
        elif score >= self.BULLISH_THRESHOLD:
            return SentimentLevel.BULLISH
        elif score <= self.VERY_BEARISH_THRESHOLD:
            return SentimentLevel.VERY_BEARISH
        elif score <= self.BEARISH_THRESHOLD:
            return SentimentLevel.BEARISH
        else:
            return SentimentLevel.NEUTRAL

    def _calculate_confidence(
        self,
        articles: List[NewsArticle],
        article_scores: List[Dict],
    ) -> float:
        """
        Calculate confidence in the sentiment analysis

        Confidence based on:
        - Number of articles (more = higher confidence)
        - Agreement between articles (similar scores = higher confidence)
        - Recency of articles (recent = higher confidence)

        Args:
            articles: Original articles
            article_scores: Calculated scores for each article

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not article_scores:
            return 0.0

        # Base confidence from article count (0-0.4)
        article_factor = min(1.0, len(articles) / 5) * 0.4

        # Agreement factor (0-0.3)
        scores = [s["adjusted_score"] for s in article_scores]
        if len(scores) >= 2:
            variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
            agreement_factor = max(0, 1 - variance) * 0.3
        else:
            agreement_factor = 0.15  # Single article = medium agreement

        # Recency factor (0-0.3)
        now = datetime.now()
        recent_count = sum(
            1 for a in articles
            if a.published_at and (now - a.published_at).days <= 3
        )
        recency_factor = min(1.0, recent_count / max(1, len(articles))) * 0.3

        return min(1.0, article_factor + agreement_factor + recency_factor)

    def _generate_factors(
        self,
        score: float,
        level: SentimentLevel,
        article_count: int,
        article_scores: List[Dict],
    ) -> List[str]:
        """
        Generate human-readable factors for confluence display

        Args:
            score: Final sentiment score
            level: Sentiment level
            article_count: Number of articles analyzed
            article_scores: Individual article scores

        Returns:
            List of factor strings
        """
        factors = []

        # Main sentiment factor
        if level == SentimentLevel.VERY_BULLISH:
            factors.append(f"Strong positive news sentiment ({score:.2f})")
        elif level == SentimentLevel.BULLISH:
            factors.append(f"Positive news sentiment ({score:.2f})")
        elif level == SentimentLevel.VERY_BEARISH:
            factors.append(f"Strong negative news sentiment ({score:.2f})")
        elif level == SentimentLevel.BEARISH:
            factors.append(f"Negative news sentiment ({score:.2f})")
        else:
            factors.append(f"Neutral news sentiment ({score:.2f})")

        # Article count
        factors.append(f"{article_count} news articles analyzed")

        # Highlight most impactful headlines
        if article_scores:
            sorted_scores = sorted(
                article_scores,
                key=lambda x: abs(x["adjusted_score"]),
                reverse=True
            )
            if sorted_scores and abs(sorted_scores[0]["adjusted_score"]) > 0.3:
                top_headline = sorted_scores[0]["headline"]
                if len(top_headline) > 50:
                    top_headline = top_headline[:47] + "..."
                factors.append(f"Key: \"{top_headline}\"")

        return factors

    def _fallback_analysis(self, articles: List[NewsArticle]) -> SentimentResult:
        """
        Simple fallback analysis when VADER is not available

        Uses keyword matching as a basic sentiment indicator.

        Args:
            articles: List of articles to analyze

        Returns:
            SentimentResult with basic analysis
        """
        positive_count = 0
        negative_count = 0

        for article in articles:
            text = f"{article.headline} {article.summary or ''}".lower()

            positive_count += sum(
                1 for kw in self.POSITIVE_FINANCIAL_KEYWORDS
                if kw in text
            )
            negative_count += sum(
                1 for kw in self.NEGATIVE_FINANCIAL_KEYWORDS
                if kw in text
            )

        total = positive_count + negative_count
        if total > 0:
            score = (positive_count - negative_count) / total
        else:
            score = 0.0

        level = self._classify_sentiment(score)

        return SentimentResult(
            score=score,
            level=level,
            confidence=0.3,  # Lower confidence without VADER
            articles_count=len(articles),
            factors=[
                f"Basic analysis (VADER not installed): {level.value}",
                f"Positive keywords: {positive_count}, Negative: {negative_count}",
            ],
        )

    def analyze_text(self, text: str) -> float:
        """
        Public method to analyze a single text string

        Args:
            text: Text to analyze

        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        return self._analyze_text(text)
