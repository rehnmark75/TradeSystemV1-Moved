# News Sentiment Integration

## Overview

The stock scanner now supports news sentiment analysis as a confluence factor for signal enrichment. This feature uses Finnhub's free tier API for news headlines and VADER (Valence Aware Dictionary and sEntiment Reasoner) for local sentiment analysis.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Signal Card    │────▶│  Finnhub API     │────▶│  VADER Analyzer │
│  (Streamlit)    │     │  (News Fetch)    │     │  (Sentiment)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                               │
         │              ┌──────────────────┐             │
         └─────────────▶│  PostgreSQL      │◀────────────┘
                        │  (Cache/Signals) │
                        └──────────────────┘
```

## Components

### 1. Finnhub Client
**File:** `worker/app/stock_scanner/core/news/finnhub_client.py`

- Async HTTP client for Finnhub API
- Rate limiting: 60 requests/minute (free tier)
- Methods:
  - `get_company_news(symbol, from_date, to_date)` - Fetch news articles

### 2. Sentiment Analyzer
**File:** `worker/app/stock_scanner/core/news/sentiment_analyzer.py`

- Uses VADER sentiment analysis (optimized for social/news text)
- Recency weighting (recent news weighted higher)
- Sentiment levels:
  - `very_bullish` (score >= 0.5)
  - `bullish` (score >= 0.15)
  - `neutral` (-0.15 < score < 0.15)
  - `bearish` (score <= -0.15)
  - `very_bearish` (score <= -0.5)

### 3. News Enrichment Service
**File:** `worker/app/stock_scanner/core/news/news_enrichment_service.py`

- Orchestrates news fetching and sentiment analysis
- Handles caching and rate limiting
- Features:
  - Automatic enrichment for A+/A quality signals
  - Manual enrichment on demand
  - Retry queue for rate-limited requests
  - Database persistence

### 4. Scanner Manager Integration
**File:** `worker/app/stock_scanner/scanners/scanner_manager.py`

New methods:
- `enrich_signals_with_news()` - Batch enrich signals
- `enrich_single_signal_news()` - Enrich single signal
- `get_unenriched_signals()` - Find signals needing enrichment
- `get_signal_news_data()` - Get news data for UI display

### 5. Streamlit UI Integration
**File:** `streamlit/pages/stock_scanner.py`

- News sentiment badge on signal cards
- "Enrich with News" / "Refresh News" button
- News Sentiment section in Deep Dive tab
- Direct Finnhub API calls (no docker exec dependency)

## Database Schema

### New Columns in `stock_scanner_signals`
```sql
news_sentiment_score DECIMAL(5,3)    -- -1.0 to 1.0
news_sentiment_level VARCHAR(20)     -- very_bullish/bullish/neutral/bearish/very_bearish
news_headlines_count INTEGER         -- Number of articles analyzed
news_factors TEXT[]                  -- Contributing news factors
news_analyzed_at TIMESTAMP           -- When analysis was performed
```

### New Table: `stock_news_cache`
```sql
CREATE TABLE stock_news_cache (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source VARCHAR(100),
    url TEXT,
    image_url TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    category VARCHAR(50),
    sentiment_score DECIMAL(5,3),
    finnhub_id BIGINT,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(ticker, finnhub_id)
);
```

### New Table: `stock_news_fetch_log`
```sql
CREATE TABLE stock_news_fetch_log (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    articles_fetched INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT true,
    error_message TEXT
);
```

## Configuration

### Environment Variables
```bash
FINNHUB_API_KEY=your_api_key_here  # Get free key at https://finnhub.io/register
```

### Config Settings (`worker/app/stock_scanner/config.py`)
```python
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWS_LOOKBACK_DAYS = 7        # Days of news to fetch
NEWS_CACHE_TTL = 3600         # Cache TTL in seconds (1 hour)
NEWS_MIN_ARTICLES = 2         # Min articles for confident sentiment
NEWS_SENTIMENT_WEIGHT = 0.08  # Weight in composite score (8%)
```

## Triggering Strategy

### Two-Tier Enrichment

| Tier | Trigger | When |
|------|---------|------|
| **Automatic** | A+ and A quality signals | During signal generation |
| **Manual** | Any signal | User clicks "Enrich with News" in UI |

### Why This Works for Daily Trading
- Signals valid for hours/days, not minutes
- Can batch news fetches efficiently
- No urgency = conservative API usage
- Plenty of time for manual review

### Rate Limit Handling
1. **Graceful degradation** - Signal generated without news if limit hit
2. **Queue for retry** - Failed tickers queued, retried after 60s cooldown
3. **Request throttling** - Max 30 req/min (50% of limit) to leave headroom
4. **Fallback** - Signal quality tier unaffected if news unavailable
5. **Manual retry** - UI button always available to fetch/refresh news

## Usage

### UI - Signal Cards
1. Navigate to Stock Scanner page
2. View signal cards - news sentiment badge shows if data exists
3. Click "Enrich with News" to fetch news for a signal
4. News section shows sentiment level, score, and factors

### UI - Deep Dive Tab
1. Open Deep Dive tab for any stock
2. News Sentiment section appears after AI Analysis
3. If news exists, displays sentiment data
4. Click "Fetch News" to get news for stocks without existing data

### CLI (Worker Container)
```python
# In worker container
from stock_scanner.scanners.scanner_manager import ScannerManager

manager = ScannerManager(db)

# Auto-enrich A+/A signals
results = await manager.enrich_signals_with_news()

# Enrich single signal
result = await manager.enrich_single_signal_news(signal_id=123)

# Get news data for display
news_data = await manager.get_signal_news_data(signal_id=123)
```

## Dependencies

### Worker Container
```
finnhub-python>=2.4.0
vaderSentiment>=3.3.2
```

### Streamlit Container
```
vaderSentiment>=3.3.2
```

## API Limits

| Plan | Rate Limit | Estimated Daily Usage |
|------|------------|----------------------|
| Finnhub Free | 60 req/min | ~20-25 calls/day |

Estimated usage breakdown:
- ~5-15 A/A+ signals per daily scan = 5-15 API calls
- Manual enrichment: ~5-10 calls/day
- **Total**: ~20-25 calls/day (well under limit)
