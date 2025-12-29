"""
Pre-Market Scanning Service

Scans for pre-market opportunities 30 minutes before market open.
Uses Finnhub API for:
- Market status verification
- Pre-market quotes
- Overnight news
- Gap detection

Runs at 9:00 AM ET (30 min before market open at 9:30 AM ET).

Usage:
    service = PreMarketService(db_manager, finnhub_api_key)
    results = await service.run_premarket_scan()
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .finnhub_client import FinnhubClient, FinnhubError, NewsArticle
from .sentiment_analyzer import NewsSentimentAnalyzer, SentimentResult, SentimentLevel

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Pre-market gap classification"""
    GAP_UP_LARGE = "gap_up_large"       # > 5%
    GAP_UP_MEDIUM = "gap_up_medium"     # 2-5%
    GAP_UP_SMALL = "gap_up_small"       # 0.5-2%
    FLAT = "flat"                        # -0.5% to 0.5%
    GAP_DOWN_SMALL = "gap_down_small"   # -0.5% to -2%
    GAP_DOWN_MEDIUM = "gap_down_medium" # -2% to -5%
    GAP_DOWN_LARGE = "gap_down_large"   # < -5%


@dataclass
class PreMarketQuote:
    """Pre-market quote data for a stock"""
    symbol: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    gap_type: GapType = GapType.FLAT
    timestamp: Optional[datetime] = None

    @property
    def gap_percent(self) -> float:
        """Calculate gap percentage from previous close"""
        if self.previous_close and self.previous_close > 0:
            return ((self.current_price - self.previous_close) / self.previous_close) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "previous_close": self.previous_close,
            "change": self.change,
            "change_percent": self.change_percent,
            "gap_percent": self.gap_percent,
            "gap_type": self.gap_type.value,
            "high": self.high,
            "low": self.low,
            "open": self.open_price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class PreMarketSignal:
    """Pre-market trading signal with news context"""
    symbol: str
    quote: PreMarketQuote
    signal_type: str  # GAP_PLAY, MOMENTUM, NEWS_CATALYST, REVERSAL
    direction: str    # BUY or SELL
    strength: str     # STRONG, MODERATE, WEAK

    # News context
    news_count: int = 0
    news_sentiment: Optional[SentimentResult] = None
    key_headlines: List[str] = field(default_factory=list)

    # Technical context
    previous_day_range: Optional[float] = None
    volume_indicator: Optional[str] = None  # HIGH, NORMAL, LOW

    # Entry/Exit levels
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    risk_reward: Optional[float] = None

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quote": self.quote.to_dict(),
            "signal_type": self.signal_type,
            "direction": self.direction,
            "strength": self.strength,
            "news_count": self.news_count,
            "news_sentiment": self.news_sentiment.to_dict() if self.news_sentiment else None,
            "key_headlines": self.key_headlines[:3],  # Top 3 headlines
            "suggested_entry": self.suggested_entry,
            "suggested_stop": self.suggested_stop,
            "suggested_target": self.suggested_target,
            "risk_reward": self.risk_reward,
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class PreMarketScanResult:
    """Result of a pre-market scan"""
    scan_time: datetime
    market_status: Dict[str, Any]
    is_pre_market: bool

    # Quotes
    quotes_fetched: int
    quotes_with_gaps: int

    # Signals generated
    signals: List[PreMarketSignal]
    gap_up_signals: int
    gap_down_signals: int
    news_catalyst_signals: int

    # Errors
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_time": self.scan_time.isoformat(),
            "market_status": self.market_status,
            "is_pre_market": self.is_pre_market,
            "quotes_fetched": self.quotes_fetched,
            "quotes_with_gaps": self.quotes_with_gaps,
            "total_signals": len(self.signals),
            "gap_up_signals": self.gap_up_signals,
            "gap_down_signals": self.gap_down_signals,
            "news_catalyst_signals": self.news_catalyst_signals,
            "signals": [s.to_dict() for s in self.signals],
            "errors": self.errors,
        }


class PreMarketService:
    """
    Service for scanning pre-market opportunities.

    Features:
    - Verifies market is in pre-market session via Finnhub
    - Fetches pre-market quotes for watchlist stocks
    - Detects and classifies gaps
    - Fetches overnight news for gapping stocks
    - Generates pre-market signals with news context
    - Stores results in database

    Schedule: Runs at 9:00 AM ET (30 min before market open)
    """

    # Gap thresholds
    GAP_THRESHOLDS = {
        "large": 5.0,   # 5% or more
        "medium": 2.0,  # 2-5%
        "small": 0.5,   # 0.5-2%
    }

    # Minimum gap for signal generation
    MIN_GAP_FOR_SIGNAL = 1.0  # 1% minimum gap

    def __init__(
        self,
        db_manager,
        finnhub_api_key: str,
        news_lookback_hours: int = 16,  # Overnight news
        min_news_articles: int = 1,
    ):
        """
        Initialize PreMarketService.

        Args:
            db_manager: Database manager instance
            finnhub_api_key: Finnhub API key
            news_lookback_hours: Hours of news to fetch (default 16 = overnight)
            min_news_articles: Minimum articles for news-driven signals
        """
        self.db = db_manager
        self.finnhub = FinnhubClient(api_key=finnhub_api_key)
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.news_lookback_hours = news_lookback_hours
        self.min_news_articles = min_news_articles

    async def run_premarket_scan(
        self,
        tickers: List[str] = None,
        force_run: bool = False,
    ) -> PreMarketScanResult:
        """
        Run a complete pre-market scan.

        Args:
            tickers: List of tickers to scan (default: top watchlist)
            force_run: Run even if not in pre-market session

        Returns:
            PreMarketScanResult with signals and metadata
        """
        logger.info("=" * 60)
        logger.info("PRE-MARKET SCAN - Starting")
        logger.info("=" * 60)

        scan_start = datetime.now()
        errors = []

        # Step 1: Verify market status
        async with self.finnhub:
            try:
                market_status = await self.finnhub.get_market_status("US")
                is_pre_market = market_status.get("session") == "pre-market"

                logger.info(f"Market status: {market_status.get('session', 'unknown')}")

                if not is_pre_market and not force_run:
                    logger.warning("Not in pre-market session. Use force_run=True to override.")
                    return PreMarketScanResult(
                        scan_time=scan_start,
                        market_status=market_status,
                        is_pre_market=False,
                        quotes_fetched=0,
                        quotes_with_gaps=0,
                        signals=[],
                        gap_up_signals=0,
                        gap_down_signals=0,
                        news_catalyst_signals=0,
                        errors=["Not in pre-market session"],
                    )
            except FinnhubError as e:
                logger.error(f"Failed to get market status: {e}")
                market_status = {"error": str(e)}
                is_pre_market = True  # Assume pre-market if can't verify
                errors.append(f"Market status error: {e}")

            # Step 2: Get tickers to scan
            if tickers is None:
                tickers = await self._get_watchlist_tickers()

            logger.info(f"Scanning {len(tickers)} tickers")

            # Step 3: Fetch pre-market quotes
            quotes = await self._fetch_premarket_quotes(tickers)
            logger.info(f"Fetched {len(quotes)} quotes")

            # Step 4: Identify gaps
            gapping_quotes = self._identify_gaps(quotes)
            logger.info(f"Found {len(gapping_quotes)} stocks with significant gaps")

            # Step 5: Fetch news for gapping stocks
            news_by_ticker = await self._fetch_overnight_news(
                [q.symbol for q in gapping_quotes]
            )

            # Step 6: Generate signals
            signals = self._generate_signals(gapping_quotes, news_by_ticker)
            logger.info(f"Generated {len(signals)} pre-market signals")

            # Step 7: Save to database
            await self._save_results(signals)

        # Calculate stats
        gap_up = sum(1 for s in signals if s.direction == "BUY")
        gap_down = sum(1 for s in signals if s.direction == "SELL")
        news_catalyst = sum(1 for s in signals if s.signal_type == "NEWS_CATALYST")

        result = PreMarketScanResult(
            scan_time=scan_start,
            market_status=market_status,
            is_pre_market=is_pre_market,
            quotes_fetched=len(quotes),
            quotes_with_gaps=len(gapping_quotes),
            signals=signals,
            gap_up_signals=gap_up,
            gap_down_signals=gap_down,
            news_catalyst_signals=news_catalyst,
            errors=errors,
        )

        logger.info(f"Pre-market scan complete: {len(signals)} signals "
                   f"(UP: {gap_up}, DOWN: {gap_down}, NEWS: {news_catalyst})")

        return result

    async def _get_watchlist_tickers(self) -> List[str]:
        """Get top tickers from watchlist"""
        query = """
            SELECT ticker FROM stock_watchlist
            WHERE tier IN (1, 2)
            ORDER BY rank_overall
            LIMIT 200
        """
        try:
            rows = await self.db.fetch(query)
            return [r['ticker'] for r in rows]
        except Exception as e:
            logger.warning(f"Could not fetch watchlist: {e}")
            # Fallback to default watchlist
            from stock_scanner import config
            return config.DEFAULT_WATCHLIST

    async def _fetch_premarket_quotes(
        self,
        tickers: List[str],
    ) -> List[PreMarketQuote]:
        """Fetch pre-market quotes for all tickers"""
        quotes = []

        # Batch fetch with rate limiting
        raw_quotes = await self.finnhub.get_quotes_batch(tickers, delay_between=0.3)

        for raw in raw_quotes:
            if "error" in raw:
                continue

            try:
                current = raw.get("current_price", 0)
                prev_close = raw.get("previous_close", 0)

                if not current or not prev_close or prev_close == 0:
                    continue

                change = raw.get("change", 0)
                change_pct = raw.get("change_percent", 0)

                # Classify gap
                gap_pct = ((current - prev_close) / prev_close) * 100
                gap_type = self._classify_gap(gap_pct)

                quote = PreMarketQuote(
                    symbol=raw["symbol"],
                    current_price=current,
                    previous_close=prev_close,
                    change=change,
                    change_percent=change_pct,
                    high=raw.get("high"),
                    low=raw.get("low"),
                    open_price=raw.get("open"),
                    gap_type=gap_type,
                    timestamp=datetime.fromtimestamp(raw.get("timestamp", 0))
                    if raw.get("timestamp") else datetime.now(),
                )
                quotes.append(quote)

            except Exception as e:
                logger.warning(f"Error processing quote for {raw.get('symbol')}: {e}")
                continue

        return quotes

    def _classify_gap(self, gap_percent: float) -> GapType:
        """Classify gap by size"""
        abs_gap = abs(gap_percent)

        if gap_percent > 0:
            if abs_gap >= self.GAP_THRESHOLDS["large"]:
                return GapType.GAP_UP_LARGE
            elif abs_gap >= self.GAP_THRESHOLDS["medium"]:
                return GapType.GAP_UP_MEDIUM
            elif abs_gap >= self.GAP_THRESHOLDS["small"]:
                return GapType.GAP_UP_SMALL
        elif gap_percent < 0:
            if abs_gap >= self.GAP_THRESHOLDS["large"]:
                return GapType.GAP_DOWN_LARGE
            elif abs_gap >= self.GAP_THRESHOLDS["medium"]:
                return GapType.GAP_DOWN_MEDIUM
            elif abs_gap >= self.GAP_THRESHOLDS["small"]:
                return GapType.GAP_DOWN_SMALL

        return GapType.FLAT

    def _identify_gaps(
        self,
        quotes: List[PreMarketQuote],
    ) -> List[PreMarketQuote]:
        """Filter quotes to only significant gaps"""
        gapping = []
        for quote in quotes:
            if abs(quote.gap_percent) >= self.MIN_GAP_FOR_SIGNAL:
                gapping.append(quote)

        # Sort by gap magnitude (largest first)
        gapping.sort(key=lambda x: abs(x.gap_percent), reverse=True)
        return gapping

    async def _fetch_overnight_news(
        self,
        tickers: List[str],
    ) -> Dict[str, List[NewsArticle]]:
        """Fetch overnight news for gapping tickers"""
        news_by_ticker = {}

        to_date = datetime.now()
        from_date = to_date - timedelta(hours=self.news_lookback_hours)

        for ticker in tickers[:50]:  # Limit to top 50 to respect rate limits
            try:
                articles = await self.finnhub.get_company_news(
                    symbol=ticker,
                    from_date=from_date,
                    to_date=to_date,
                    use_cache=False,  # Fresh news
                )
                if articles:
                    news_by_ticker[ticker] = articles
                await asyncio.sleep(0.3)  # Rate limiting
            except FinnhubError as e:
                logger.warning(f"Could not fetch news for {ticker}: {e}")
                continue

        return news_by_ticker

    def _generate_signals(
        self,
        quotes: List[PreMarketQuote],
        news_by_ticker: Dict[str, List[NewsArticle]],
    ) -> List[PreMarketSignal]:
        """Generate pre-market signals from quotes and news"""
        signals = []

        for quote in quotes:
            articles = news_by_ticker.get(quote.symbol, [])

            # Analyze news sentiment if available
            sentiment = None
            key_headlines = []
            if articles:
                sentiment = self.sentiment_analyzer.analyze_articles(articles)
                key_headlines = [a.headline for a in articles[:3]]

            # Determine signal type and direction
            signal_type, direction, strength, confidence = self._classify_signal(
                quote, sentiment, len(articles)
            )

            if signal_type is None:
                continue

            # Calculate entry/stop/target
            entry, stop, target = self._calculate_levels(quote, direction)

            rr = None
            if entry and stop and target and (entry - stop) != 0:
                rr = abs(target - entry) / abs(entry - stop)

            signal = PreMarketSignal(
                symbol=quote.symbol,
                quote=quote,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                news_count=len(articles),
                news_sentiment=sentiment,
                key_headlines=key_headlines,
                suggested_entry=entry,
                suggested_stop=stop,
                suggested_target=target,
                risk_reward=rr,
                confidence=confidence,
            )
            signals.append(signal)

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals

    def _classify_signal(
        self,
        quote: PreMarketQuote,
        sentiment: Optional[SentimentResult],
        news_count: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """
        Classify signal type, direction, and strength.

        Returns:
            (signal_type, direction, strength, confidence)
        """
        gap_pct = quote.gap_percent
        abs_gap = abs(gap_pct)
        confidence = 0.5

        # Base direction from gap
        direction = "BUY" if gap_pct > 0 else "SELL"

        # Signal type classification
        signal_type = "GAP_PLAY"

        # If news supports the gap, it's a news catalyst
        if news_count >= self.min_news_articles and sentiment:
            # News sentiment aligns with gap direction
            bullish_sentiment = sentiment.level in (
                SentimentLevel.BULLISH, SentimentLevel.VERY_BULLISH
            )
            bearish_sentiment = sentiment.level in (
                SentimentLevel.BEARISH, SentimentLevel.VERY_BEARISH
            )

            if (gap_pct > 0 and bullish_sentiment) or (gap_pct < 0 and bearish_sentiment):
                signal_type = "NEWS_CATALYST"
                confidence += 0.15

            # News contradicts gap - potential reversal
            if (gap_pct > 0 and bearish_sentiment) or (gap_pct < 0 and bullish_sentiment):
                signal_type = "REVERSAL"
                direction = "SELL" if gap_pct > 0 else "BUY"
                confidence -= 0.1

        # Strength classification
        if abs_gap >= self.GAP_THRESHOLDS["large"]:
            strength = "STRONG"
            confidence += 0.2
        elif abs_gap >= self.GAP_THRESHOLDS["medium"]:
            strength = "MODERATE"
            confidence += 0.1
        else:
            strength = "WEAK"

        # Adjust confidence based on sentiment confidence
        if sentiment:
            confidence += sentiment.confidence * 0.1

        # Cap confidence
        confidence = min(max(confidence, 0.1), 0.95)

        # Filter out weak signals
        if confidence < 0.4:
            return None, None, None, 0

        return signal_type, direction, strength, confidence

    def _calculate_levels(
        self,
        quote: PreMarketQuote,
        direction: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate entry, stop, and target levels"""
        price = quote.current_price
        prev_close = quote.previous_close

        if direction == "BUY":
            # Entry at current pre-market price
            entry = price
            # Stop below previous close
            stop = prev_close * 0.995  # 0.5% below prev close
            # Target at 2:1 R:R
            risk = entry - stop
            target = entry + (risk * 2)
        else:
            # Entry at current pre-market price
            entry = price
            # Stop above previous close
            stop = prev_close * 1.005  # 0.5% above prev close
            # Target at 2:1 R:R
            risk = stop - entry
            target = entry - (risk * 2)

        return round(entry, 2), round(stop, 2), round(target, 2)

    async def _save_results(self, signals: List[PreMarketSignal]):
        """Save pre-market signals to database"""
        if not signals:
            return

        saved_count = 0
        for signal in signals:
            try:
                # Check if signal already exists for this symbol today
                check_query = """
                    SELECT id FROM stock_premarket_signals
                    WHERE symbol = $1
                    AND DATE(generated_at) = DATE($2)
                """
                existing = await self.db.fetchval(
                    check_query,
                    signal.symbol,
                    signal.generated_at,
                )

                if existing:
                    # Update existing signal
                    update_query = """
                        UPDATE stock_premarket_signals
                        SET signal_type = $1,
                            direction = $2,
                            strength = $3,
                            confidence = $4,
                            gap_percent = $5,
                            gap_type = $6,
                            current_price = $7,
                            previous_close = $8,
                            news_count = $9,
                            news_sentiment_score = $10,
                            news_sentiment_level = $11,
                            key_headlines = $12,
                            suggested_entry = $13,
                            suggested_stop = $14,
                            suggested_target = $15,
                            risk_reward = $16,
                            updated_at = NOW()
                        WHERE id = $17
                    """
                    await self.db.execute(
                        update_query,
                        signal.signal_type,
                        signal.direction,
                        signal.strength,
                        signal.confidence,
                        signal.quote.gap_percent,
                        signal.quote.gap_type.value,
                        signal.quote.current_price,
                        signal.quote.previous_close,
                        signal.news_count,
                        signal.news_sentiment.score if signal.news_sentiment else None,
                        signal.news_sentiment.level.value if signal.news_sentiment else None,
                        signal.key_headlines,
                        signal.suggested_entry,
                        signal.suggested_stop,
                        signal.suggested_target,
                        signal.risk_reward,
                        existing,
                    )
                else:
                    # Insert new signal
                    insert_query = """
                        INSERT INTO stock_premarket_signals
                        (symbol, signal_type, direction, strength, confidence,
                         gap_percent, gap_type, current_price, previous_close,
                         news_count, news_sentiment_score, news_sentiment_level,
                         key_headlines, suggested_entry, suggested_stop, suggested_target,
                         risk_reward, generated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                                $14, $15, $16, $17, $18)
                    """
                    await self.db.execute(
                        insert_query,
                        signal.symbol,
                        signal.signal_type,
                        signal.direction,
                        signal.strength,
                        signal.confidence,
                        signal.quote.gap_percent,
                        signal.quote.gap_type.value,
                        signal.quote.current_price,
                        signal.quote.previous_close,
                        signal.news_count,
                        signal.news_sentiment.score if signal.news_sentiment else None,
                        signal.news_sentiment.level.value if signal.news_sentiment else None,
                        signal.key_headlines,
                        signal.suggested_entry,
                        signal.suggested_stop,
                        signal.suggested_target,
                        signal.risk_reward,
                        signal.generated_at,
                    )
                saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to save signal for {signal.symbol}: {e}")

        logger.info(f"Saved {saved_count} pre-market signals to database")

    async def get_latest_signals(
        self,
        min_confidence: float = 0.5,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get latest pre-market signals from database.

        Args:
            min_confidence: Minimum confidence threshold
            limit: Maximum signals to return

        Returns:
            List of signal dicts
        """
        query = """
            SELECT *
            FROM stock_premarket_signals
            WHERE generated_at >= CURRENT_DATE
            AND confidence >= $1
            ORDER BY confidence DESC, gap_percent DESC
            LIMIT $2
        """

        try:
            rows = await self.db.fetch(query, min_confidence, limit)
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Error fetching pre-market signals: {e}")
            return []

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get Finnhub rate limit status"""
        return self.finnhub.get_rate_limit_status()
