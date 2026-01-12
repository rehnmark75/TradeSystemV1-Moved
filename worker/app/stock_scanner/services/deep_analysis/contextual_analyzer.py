"""
Contextual Deep Analyzer

Performs contextual analysis including:
- News sentiment synthesis
- Market regime detection
- Sector rotation analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import pandas as pd

from .models import (
    ContextualDeepResult,
    NewsSentimentResult,
    MarketRegimeResult,
    SectorRotationResult,
    TrendDirection,
    MarketRegime,
    DeepAnalysisConfig,
)

logger = logging.getLogger(__name__)


# Sector to ETF mapping
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
    # Aliases
    'Financial Services': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Basic Materials': 'XLB',
    'Information Technology': 'XLK',
}


class ContextualDeepAnalyzer:
    """
    Performs contextual analysis on stock signals.

    Components:
    1. News Sentiment (10% of DAQ)
    2. Market Regime (10% of DAQ)
    3. Sector Rotation (10% of DAQ)
    """

    def __init__(self, db_manager, config: Optional[DeepAnalysisConfig] = None):
        """
        Initialize contextual analyzer.

        Args:
            db_manager: Database manager
            config: Deep analysis configuration
        """
        self.db = db_manager
        self.config = config or DeepAnalysisConfig()

    async def analyze(
        self,
        ticker: str,
        signal: Dict[str, Any],
        sector: Optional[str] = None
    ) -> ContextualDeepResult:
        """
        Perform complete contextual deep analysis.

        Args:
            ticker: Stock ticker
            signal: Signal data
            sector: Stock's sector (optional, will be fetched if not provided)

        Returns:
            ContextualDeepResult with all component scores
        """
        # Get sector if not provided
        if not sector:
            sector = await self._get_ticker_sector(ticker)

        # Determine signal direction
        signal_direction = self._get_signal_direction(signal)

        # Run analysis components
        news_result = await self._analyze_news(ticker)
        regime_result = await self._analyze_regime(signal_direction)
        sector_result = await self._analyze_sector(sector)

        return ContextualDeepResult(
            news=news_result,
            regime=regime_result,
            sector=sector_result
        )

    def _get_signal_direction(self, signal: Dict[str, Any]) -> TrendDirection:
        """Determine signal direction from signal data"""
        signal_type = signal.get('signal_type', '').upper()
        if signal_type in ('BUY', 'LONG'):
            return TrendDirection.BULLISH
        elif signal_type in ('SELL', 'SHORT'):
            return TrendDirection.BEARISH
        return TrendDirection.NEUTRAL

    async def _get_ticker_sector(self, ticker: str) -> Optional[str]:
        """Get sector for a ticker from database"""
        query = "SELECT sector FROM stock_instruments WHERE ticker = $1"
        row = await self.db.fetchrow(query, ticker)
        return row['sector'] if row else None

    # =========================================================================
    # NEWS SENTIMENT ANALYSIS (10% of DAQ)
    # =========================================================================

    async def _analyze_news(self, ticker: str) -> NewsSentimentResult:
        """
        Analyze news sentiment from cached Finnhub articles.

        Uses existing news_sentiment data from stock_scanner_signals or
        stock_news_cache if available.

        Scoring:
        - Very bullish (>0.5): 100
        - Bullish (0.15-0.5): 80
        - Neutral (-0.15 to 0.15): 50
        - Bearish (-0.5 to -0.15): 30
        - Very bearish (<-0.5): 10
        """
        # Try to get news sentiment from news cache
        query = """
            SELECT
                ticker,
                headline,
                summary,
                sentiment_score,
                published_at,
                source
            FROM stock_news_cache
            WHERE ticker = $1
              AND published_at >= NOW() - INTERVAL '7 days'
            ORDER BY published_at DESC
            LIMIT 10
        """
        rows = await self.db.fetch(query, ticker)

        if not rows:
            # Try to get from signal's news data
            signal_query = """
                SELECT
                    news_sentiment_score,
                    news_sentiment_level,
                    news_headlines_count,
                    news_factors
                FROM stock_scanner_signals
                WHERE ticker = $1
                  AND news_analyzed_at IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
            """
            signal_row = await self.db.fetchrow(signal_query, ticker)

            if signal_row and signal_row.get('news_sentiment_score') is not None:
                sentiment_value = float(signal_row['news_sentiment_score'])
                sentiment_level = signal_row.get('news_sentiment_level', 'neutral')
                articles_count = signal_row.get('news_headlines_count', 0) or 0
                # news_factors contains headline strings
                news_factors = signal_row.get('news_factors', []) or []
                top_headlines = [{'headline': h, 'sentiment': 0} for h in news_factors[:3]]

                score = self._sentiment_to_score(sentiment_value)

                return NewsSentimentResult(
                    score=score,
                    sentiment_value=sentiment_value,
                    sentiment_level=sentiment_level,
                    articles_count=articles_count,
                    confidence=0.5 if articles_count < 3 else 0.8,
                    top_headlines=top_headlines if top_headlines else [],
                    summary=None,
                    details={'source': 'signal_cache'}
                )

            # No news data available
            return NewsSentimentResult(
                score=50,  # Neutral when no data
                sentiment_value=0.0,
                sentiment_level='neutral',
                articles_count=0,
                confidence=0.0,
                details={'error': 'No news data available'}
            )

        # Calculate aggregate sentiment from cached articles
        articles = [dict(row) for row in rows]
        sentiment_scores = [a.get('sentiment_score', 0) for a in articles if a.get('sentiment_score') is not None]

        if not sentiment_scores:
            avg_sentiment = 0.0
        else:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        # Determine sentiment level
        if avg_sentiment >= 0.5:
            sentiment_level = 'very_bullish'
        elif avg_sentiment >= 0.15:
            sentiment_level = 'bullish'
        elif avg_sentiment <= -0.5:
            sentiment_level = 'very_bearish'
        elif avg_sentiment <= -0.15:
            sentiment_level = 'bearish'
        else:
            sentiment_level = 'neutral'

        score = self._sentiment_to_score(avg_sentiment)

        # Get top 3 headlines
        top_headlines = [
            {
                'headline': a.get('headline', ''),
                'sentiment': a.get('sentiment_score', 0),
                'source': a.get('source', ''),
            }
            for a in articles[:3]
        ]

        # Build summary
        summary = f"{len(articles)} articles in last 7 days. Average sentiment: {sentiment_level} ({avg_sentiment:.2f})"

        return NewsSentimentResult(
            score=score,
            sentiment_value=avg_sentiment,
            sentiment_level=sentiment_level,
            articles_count=len(articles),
            confidence=min(1.0, len(articles) / 5),  # More articles = higher confidence
            top_headlines=top_headlines,
            summary=summary,
            details={'source': 'news_cache'}
        )

    def _sentiment_to_score(self, sentiment: float) -> int:
        """Convert sentiment value (-1 to 1) to score (0-100)"""
        if sentiment >= 0.5:
            return 100
        elif sentiment >= 0.3:
            return 85
        elif sentiment >= 0.15:
            return 70
        elif sentiment >= 0:
            return 55
        elif sentiment >= -0.15:
            return 45
        elif sentiment >= -0.3:
            return 30
        elif sentiment >= -0.5:
            return 20
        else:
            return 10

    # =========================================================================
    # MARKET REGIME ANALYSIS (10% of DAQ)
    # =========================================================================

    async def _analyze_regime(self, signal_direction: TrendDirection) -> MarketRegimeResult:
        """
        Analyze current market regime based on SPY.

        Fetches SPY data to determine overall market direction.

        Scoring (based on alignment with signal):
        - Strong alignment (bullish signal in strong bull market): 100
        - Aligned: 80
        - Neutral market: 60
        - Counter-trend: 40
        - Strong counter-trend: 20
        """
        # Fetch SPY candles from database
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM stock_candles
            WHERE ticker = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await self.db.fetch(query, 'SPY', '1d', 50)

        if not rows or len(rows) < 20:
            return MarketRegimeResult(
                score=60,  # Neutral when no data
                regime=MarketRegime.NEUTRAL,
                spy_trend=TrendDirection.NEUTRAL,
                signal_regime_aligned=True,
                details={'error': 'Insufficient SPY data'}
            )

        df = pd.DataFrame([dict(row) for row in rows])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate metrics
        latest_close = df.iloc[-1]['close']
        close_1w_ago = df.iloc[-5]['close'] if len(df) >= 5 else latest_close
        close_1m_ago = df.iloc[-20]['close'] if len(df) >= 20 else latest_close

        spy_change_1w = ((latest_close - close_1w_ago) / close_1w_ago) * 100
        spy_change_1m = ((latest_close - close_1m_ago) / close_1m_ago) * 100

        # Calculate EMAs
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        latest_ema_20 = df.iloc[-1]['ema_20']
        latest_ema_50 = df.iloc[-1]['ema_50']

        # Determine regime
        if spy_change_1m > 5 and latest_close > latest_ema_20 > latest_ema_50:
            regime = MarketRegime.STRONG_BULLISH
            spy_trend = TrendDirection.BULLISH
        elif spy_change_1m > 2 and latest_close > latest_ema_50:
            regime = MarketRegime.BULLISH
            spy_trend = TrendDirection.BULLISH
        elif spy_change_1m < -5 and latest_close < latest_ema_20 < latest_ema_50:
            regime = MarketRegime.STRONG_BEARISH
            spy_trend = TrendDirection.BEARISH
        elif spy_change_1m < -2 and latest_close < latest_ema_50:
            regime = MarketRegime.BEARISH
            spy_trend = TrendDirection.BEARISH
        else:
            regime = MarketRegime.NEUTRAL
            spy_trend = TrendDirection.NEUTRAL

        # Check alignment with signal
        signal_bullish = signal_direction == TrendDirection.BULLISH
        regime_bullish = regime in (MarketRegime.BULLISH, MarketRegime.STRONG_BULLISH)
        regime_bearish = regime in (MarketRegime.BEARISH, MarketRegime.STRONG_BEARISH)

        # Calculate score
        if signal_bullish:
            if regime == MarketRegime.STRONG_BULLISH:
                score = 100
                aligned = True
            elif regime == MarketRegime.BULLISH:
                score = 85
                aligned = True
            elif regime == MarketRegime.NEUTRAL:
                score = 60
                aligned = True
            elif regime == MarketRegime.BEARISH:
                score = 40
                aligned = False
            else:  # STRONG_BEARISH
                score = 20
                aligned = False
        else:  # Bearish signal
            if regime == MarketRegime.STRONG_BEARISH:
                score = 100
                aligned = True
            elif regime == MarketRegime.BEARISH:
                score = 85
                aligned = True
            elif regime == MarketRegime.NEUTRAL:
                score = 60
                aligned = True
            elif regime == MarketRegime.BULLISH:
                score = 40
                aligned = False
            else:  # STRONG_BULLISH
                score = 20
                aligned = False

        return MarketRegimeResult(
            score=score,
            regime=regime,
            spy_trend=spy_trend,
            spy_change_1w=spy_change_1w,
            spy_change_1m=spy_change_1m,
            signal_regime_aligned=aligned,
            details={
                'spy_close': latest_close,
                'spy_ema_20': latest_ema_20,
                'spy_ema_50': latest_ema_50,
            }
        )

    # =========================================================================
    # SECTOR ROTATION ANALYSIS (10% of DAQ)
    # =========================================================================

    async def _analyze_sector(self, sector: Optional[str]) -> SectorRotationResult:
        """
        Analyze sector rotation and relative strength.

        Compares sector ETF performance to SPY.

        Scoring:
        - Sector outperforming by >3%: 100
        - Sector outperforming by 1-3%: 85
        - Sector inline with market: 60
        - Sector underperforming by 1-3%: 40
        - Sector underperforming by >3%: 20
        """
        if not sector:
            return SectorRotationResult(
                score=50,
                details={'error': 'No sector data available'}
            )

        # Get sector ETF
        sector_etf = SECTOR_ETF_MAP.get(sector)
        if not sector_etf:
            return SectorRotationResult(
                score=50,
                sector=sector,
                details={'error': f'No ETF mapping for sector: {sector}'}
            )

        # Fetch sector ETF and SPY data
        query = """
            SELECT ticker, timestamp, close
            FROM stock_candles
            WHERE ticker IN ($1, $2)
              AND timeframe = $3
            ORDER BY timestamp DESC
            LIMIT $4
        """
        rows = await self.db.fetch(query, sector_etf, 'SPY', '1d', 100)

        if not rows or len(rows) < 40:  # Need at least 20 days for both
            return SectorRotationResult(
                score=50,
                sector=sector,
                sector_etf=sector_etf,
                details={'error': 'Insufficient sector/SPY data'}
            )

        # Separate data by ticker
        sector_data = []
        spy_data = []

        for row in rows:
            if row['ticker'] == sector_etf:
                sector_data.append(dict(row))
            else:
                spy_data.append(dict(row))

        if len(sector_data) < 20 or len(spy_data) < 20:
            return SectorRotationResult(
                score=50,
                sector=sector,
                sector_etf=sector_etf,
                details={'error': 'Insufficient data for comparison'}
            )

        # Sort by timestamp
        sector_data = sorted(sector_data, key=lambda x: x['timestamp'])
        spy_data = sorted(spy_data, key=lambda x: x['timestamp'])

        # Calculate performance
        sector_latest = sector_data[-1]['close']
        sector_1w_ago = sector_data[-5]['close'] if len(sector_data) >= 5 else sector_latest
        sector_1m_ago = sector_data[-20]['close'] if len(sector_data) >= 20 else sector_latest

        spy_latest = spy_data[-1]['close']
        spy_1w_ago = spy_data[-5]['close'] if len(spy_data) >= 5 else spy_latest
        spy_1m_ago = spy_data[-20]['close'] if len(spy_data) >= 20 else spy_latest

        sector_change_1w = ((sector_latest - sector_1w_ago) / sector_1w_ago) * 100
        sector_change_1m = ((sector_latest - sector_1m_ago) / sector_1m_ago) * 100

        spy_change_1m = ((spy_latest - spy_1m_ago) / spy_1m_ago) * 100

        # Calculate relative strength
        sector_rs = sector_change_1m - spy_change_1m

        # Determine if outperforming
        sector_outperforming = sector_rs > 0

        # Calculate score
        if sector_rs > 3:
            score = 100
        elif sector_rs > 1:
            score = 85
        elif sector_rs > -1:
            score = 60
        elif sector_rs > -3:
            score = 40
        else:
            score = 20

        return SectorRotationResult(
            score=score,
            sector=sector,
            sector_etf=sector_etf,
            sector_rs=sector_rs,
            sector_outperforming=sector_outperforming,
            sector_change_1w=sector_change_1w,
            sector_change_1m=sector_change_1m,
            details={
                'spy_change_1m': spy_change_1m,
            }
        )
