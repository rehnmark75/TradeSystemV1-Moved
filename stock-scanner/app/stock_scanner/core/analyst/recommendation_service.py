"""
Analyst Recommendation Service

Fetches Finnhub analyst recommendation trends and stores them in the database.
Designed to enrich new signals without exceeding free-tier rate limits.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..news.finnhub_client import FinnhubClient, FinnhubError, FinnhubRateLimitError

logger = logging.getLogger(__name__)


class AnalystRecommendationService:
    """Service for fetching and storing analyst recommendation trends."""

    def __init__(
        self,
        db_manager,
        finnhub_api_key: str,
        cache_ttl_hours: int = 24,
    ):
        self.db = db_manager
        self.cache_ttl_hours = cache_ttl_hours
        self.finnhub = FinnhubClient(api_key=finnhub_api_key)

    async def enrich_tickers(
        self,
        tickers: List[str],
        max_per_run: int = 20,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch and store recommendations for a batch of tickers.

        Returns:
            Dict with attempted/successful counts.
        """
        attempted = 0
        successful = 0

        for ticker in tickers[:max_per_run]:
            ticker = ticker.upper().strip()
            if not ticker:
                continue

            if not force_refresh:
                cached = await self._has_recent_data(ticker)
                if cached:
                    continue

            attempted += 1
            try:
                trends = await self._fetch_trends(ticker)
                if trends:
                    await self._upsert_trends(ticker, trends)
                    successful += 1
            except FinnhubRateLimitError:
                logger.warning("Finnhub rate limit hit - stopping recommendation enrichment")
                break
            except FinnhubError as e:
                logger.warning(f"Recommendation fetch failed for {ticker}: {e.message}")
            except Exception as e:
                logger.exception(f"Unexpected error for {ticker}: {e}")

        return {"attempted": attempted, "successful": successful}

    async def _fetch_trends(self, ticker: str) -> List[Dict[str, Any]]:
        async with self.finnhub:
            return await self.finnhub.get_recommendation_trends(ticker, use_cache=True)

    async def _has_recent_data(self, ticker: str) -> bool:
        query = """
            SELECT MAX(updated_at) AS updated_at
            FROM stock_analyst_recommendations
            WHERE ticker = $1
        """
        row = await self.db.fetchrow(query, ticker)
        if not row or not row["updated_at"]:
            return False
        updated_at = row["updated_at"]
        # Normalize timestamps to timezone-aware UTC before comparison.
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.cache_ttl_hours)
        return updated_at >= cutoff

    async def _upsert_trends(self, ticker: str, trends: List[Dict[str, Any]]):
        query = """
            INSERT INTO stock_analyst_recommendations (
                ticker, period, strong_buy, buy, hold, sell, strong_sell, source
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8
            )
            ON CONFLICT (ticker, period) DO UPDATE SET
                strong_buy = EXCLUDED.strong_buy,
                buy = EXCLUDED.buy,
                hold = EXCLUDED.hold,
                sell = EXCLUDED.sell,
                strong_sell = EXCLUDED.strong_sell,
                source = EXCLUDED.source,
                updated_at = NOW()
        """
        for trend in trends:
            period = trend.get("period")
            if not period:
                continue
            if isinstance(period, str):
                try:
                    period = datetime.strptime(period, "%Y-%m-%d").date()
                except ValueError:
                    logger.warning(f"Invalid recommendation period for {ticker}: {period}")
                    continue
            await self.db.execute(
                query,
                ticker,
                period,
                trend.get("strongBuy"),
                trend.get("buy"),
                trend.get("hold"),
                trend.get("sell"),
                trend.get("strongSell"),
                "finnhub",
            )
