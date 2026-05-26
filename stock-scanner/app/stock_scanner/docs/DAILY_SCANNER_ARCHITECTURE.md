# Daily Stock Scanner Architecture
## High Volume, High Movement Stock Analysis System

**Author**: Financial Data Engineer
**Date**: 2025-12-07
**Version**: 1.0

---

## Executive Summary

This document outlines a comprehensive architecture for a **daily timeframe stock scanner** focused on identifying **high volume, high movement stocks** from a universe of 3,428 US stocks (NYSE + NASDAQ).

### Current State
- **3,428 active US stocks** (no ETFs)
- **1,367,308 hourly candles** stored in PostgreSQL
- **Primary timeframe**: 1H (hourly) from yfinance
- **Target timeframe**: Daily (synthesized from 1H)

### Objectives
1. Synthesize 1H → Daily candles efficiently
2. Identify high-volume stocks with significant price movement
3. Maintain a dynamic, scored watchlist for trading signals
4. Optimize database for fast daily analysis
5. Provide actionable trading signals with context

---

## 1. DATA PIPELINE: 1H → Daily Synthesis

### 1.1 Current Database Schema

```sql
-- Existing table (stores 1H candles from yfinance)
TABLE stock_candles (
    id BIGINT PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,  -- '1h', '4h', '1d'
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC(12,4) NOT NULL,
    high NUMERIC(12,4) NOT NULL,
    low NUMERIC(12,4) NOT NULL,
    close NUMERIC(12,4) NOT NULL,
    volume BIGINT,
    CONSTRAINT unique_stock_candle UNIQUE (ticker, timeframe, timestamp)
);
```

### 1.2 Daily Candle Synthesis Strategy

**Option A: Materialized View (Recommended)**

```sql
-- Create materialized view for daily candles
CREATE MATERIALIZED VIEW stock_daily_candles AS
SELECT
    ticker,
    '1d' as timeframe,
    DATE_TRUNC('day', timestamp) as timestamp,
    (ARRAY_AGG(open ORDER BY timestamp ASC))[1] as open,
    MAX(high) as high,
    MIN(low) as low,
    (ARRAY_AGG(close ORDER BY timestamp DESC))[1] as close,
    SUM(volume) as volume,
    COUNT(*) as hourly_candles,  -- Quality check
    MIN(timestamp) as first_hour,
    MAX(timestamp) as last_hour
FROM stock_candles
WHERE timeframe = '1h'
GROUP BY ticker, DATE_TRUNC('day', timestamp);

-- Indexes for fast access
CREATE INDEX idx_daily_ticker_time ON stock_daily_candles(ticker, timestamp DESC);
CREATE INDEX idx_daily_timestamp ON stock_daily_candles(timestamp DESC);
CREATE UNIQUE INDEX idx_daily_unique ON stock_daily_candles(ticker, timestamp);

-- Refresh function (run after market close)
REFRESH MATERIALIZED VIEW CONCURRENTLY stock_daily_candles;
```

**Benefits**:
- Fast queries (pre-aggregated)
- No duplicate storage (view-based)
- Easy refresh after daily updates
- Supports incremental refresh

**Option B: Database Function (Alternative)**

```sql
-- Function to synthesize daily candles on demand
CREATE OR REPLACE FUNCTION synthesize_daily_candles(
    p_ticker VARCHAR DEFAULT NULL,
    p_days INT DEFAULT 365
) RETURNS TABLE (
    ticker VARCHAR,
    timestamp TIMESTAMP,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        sc.ticker,
        DATE_TRUNC('day', sc.timestamp) as timestamp,
        (ARRAY_AGG(sc.open ORDER BY sc.timestamp ASC))[1] as open,
        MAX(sc.high) as high,
        MIN(sc.low) as low,
        (ARRAY_AGG(sc.close ORDER BY sc.timestamp DESC))[1] as close,
        SUM(sc.volume) as volume
    FROM stock_candles sc
    WHERE sc.timeframe = '1h'
        AND (p_ticker IS NULL OR sc.ticker = p_ticker)
        AND sc.timestamp >= CURRENT_DATE - p_days
    GROUP BY sc.ticker, DATE_TRUNC('day', sc.timestamp)
    ORDER BY timestamp DESC;
END;
$$ LANGUAGE plpgsql;
```

### 1.3 When to Run Synthesis

**Daily Schedule (US Market)**:
```
4:00 PM ET (21:00 UTC) - Market closes
4:30 PM ET (21:30 UTC) - All trades settled
6:00 PM ET (23:00 UTC) - Run incremental 1H update
7:00 PM ET (00:00 UTC) - Synthesize daily candles
7:15 PM ET (00:15 UTC) - Calculate metrics & update watchlist
```

**Cron Schedule**:
```bash
# Run after market close (11 PM UTC = 6 PM ET)
0 23 * * 1-5 docker exec worker-app python -m stock_scanner.scripts.daily_update
30 23 * * 1-5 docker exec worker-app python -m stock_scanner.scripts.synthesize_daily
0 0 * * 2-6 docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist
```

### 1.4 Data Quality & Validation

```sql
-- Quality check: Ensure daily candles have enough hourly data
CREATE VIEW stock_daily_quality AS
SELECT
    ticker,
    timestamp::DATE as date,
    COUNT(*) as hourly_candles,
    CASE
        WHEN COUNT(*) >= 6 THEN 'FULL'      -- Full trading day
        WHEN COUNT(*) >= 3 THEN 'PARTIAL'   -- Partial day
        ELSE 'INCOMPLETE'                   -- Missing data
    END as quality,
    MIN(timestamp)::TIME as first_hour,
    MAX(timestamp)::TIME as last_hour,
    SUM(volume) as total_volume
FROM stock_candles
WHERE timeframe = '1h'
GROUP BY ticker, timestamp::DATE
HAVING COUNT(*) < 7  -- Flag days with < 7 hours
ORDER BY date DESC, ticker;
```

---

## 2. STOCK FILTERING CRITERIA: "High Volume with Movement"

### 2.1 Volume Metrics

**Absolute Volume Threshold**:
```python
# Minimum average daily volume (ADV) filters
VOLUME_TIERS = {
    'mega_liquid': 10_000_000,    # 10M+ shares/day (top 200 stocks)
    'high_liquid': 5_000_000,     # 5M+ shares/day (top 500)
    'liquid': 1_000_000,          # 1M+ shares/day (top 1500)
    'tradeable': 500_000,         # 500K+ (minimum for swing trading)
}

# Dollar volume (better measure of liquidity)
MIN_DOLLAR_VOLUME = 25_000_000  # $25M/day minimum
```

**Relative Volume (RVol)**:
```python
# Relative volume = Today's volume / 20-day average volume
RVOL_THRESHOLDS = {
    'extreme': 3.0,    # 3x average (breaking news/events)
    'high': 2.0,       # 2x average (unusual activity)
    'elevated': 1.5,   # 1.5x average (above normal)
    'normal': 1.0,     # Average
}

# For scanner: stocks with RVol >= 1.2 (20% above average)
MIN_RVOL = 1.2
```

### 2.2 Movement Metrics

**Average True Range (ATR)**:
```python
# ATR as percentage of price (normalized volatility)
ATR_PERCENT_THRESHOLDS = {
    'high_volatility': 3.0,    # 3%+ daily range
    'moderate': 2.0,           # 2%+ daily range
    'low': 1.0,                # 1%+ daily range
}

# For scanner: minimum 1.5% ATR
MIN_ATR_PERCENT = 1.5
```

**Daily Price Range**:
```python
# (High - Low) / Close as percentage
MIN_DAILY_RANGE_PERCENT = 2.0  # Minimum 2% intraday range

# 20-day average range
MIN_AVG_RANGE_PERCENT = 1.5    # Average 1.5% daily range
```

**Bollinger Band Width (Volatility)**:
```python
# BB Width = (Upper - Lower) / Middle * 100
MIN_BB_WIDTH_PERCENT = 4.0  # Minimum 4% BB width
```

### 2.3 Price Filters

```python
# Price range for swing trading
MIN_PRICE = 5.00      # Avoid penny stocks
MAX_PRICE = 1000.00   # Practical limit for position sizing

# Preferred range for optimal liquidity
OPTIMAL_PRICE_RANGE = (10.00, 500.00)
```

### 2.4 Combined Scoring System

```sql
-- Stock quality score (0-100)
CREATE OR REPLACE FUNCTION calculate_stock_score(
    avg_volume BIGINT,
    avg_dollar_volume NUMERIC,
    rvol NUMERIC,
    atr_percent NUMERIC,
    price_range_percent NUMERIC,
    bb_width NUMERIC,
    price NUMERIC
) RETURNS NUMERIC AS $$
DECLARE
    score NUMERIC := 0;
BEGIN
    -- Volume score (30 points max)
    IF avg_dollar_volume >= 100000000 THEN score := score + 30;  -- $100M+
    ELSIF avg_dollar_volume >= 50000000 THEN score := score + 25;
    ELSIF avg_dollar_volume >= 25000000 THEN score := score + 20;
    ELSIF avg_dollar_volume >= 10000000 THEN score := score + 10;
    END IF;

    -- Relative volume score (20 points max)
    IF rvol >= 2.0 THEN score := score + 20;
    ELSIF rvol >= 1.5 THEN score := score + 15;
    ELSIF rvol >= 1.2 THEN score := score + 10;
    END IF;

    -- Volatility score (30 points max)
    IF atr_percent >= 3.0 THEN score := score + 30;
    ELSIF atr_percent >= 2.0 THEN score := score + 20;
    ELSIF atr_percent >= 1.5 THEN score := score + 15;
    END IF;

    -- Range score (10 points max)
    IF price_range_percent >= 3.0 THEN score := score + 10;
    ELSIF price_range_percent >= 2.0 THEN score := score + 7;
    END IF;

    -- Price quality bonus (10 points max)
    IF price BETWEEN 10 AND 500 THEN score := score + 10;
    ELSIF price BETWEEN 5 AND 1000 THEN score := score + 5;
    END IF;

    RETURN score;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

---

## 3. WATCHLIST MANAGEMENT

### 3.1 Dynamic Watchlist Table

```sql
-- Watchlist with calculated metrics
CREATE TABLE stock_watchlist (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,

    -- Ranking metrics
    rank INT,
    score NUMERIC(5,2),

    -- Volume metrics (20-day averages)
    avg_volume BIGINT,
    avg_dollar_volume NUMERIC(15,2),
    current_rvol NUMERIC(6,2),

    -- Volatility metrics
    atr_14 NUMERIC(12,4),
    atr_percent NUMERIC(6,2),
    avg_range_percent NUMERIC(6,2),
    bb_width NUMERIC(6,2),

    -- Price info
    current_price NUMERIC(12,4),
    price_change_1d NUMERIC(6,2),
    price_change_5d NUMERIC(6,2),
    price_change_20d NUMERIC(6,2),

    -- Market cap & sector
    market_cap BIGINT,
    sector VARCHAR(50),

    -- Quality flags
    is_liquid BOOLEAN DEFAULT FALSE,
    is_volatile BOOLEAN DEFAULT FALSE,
    is_trending BOOLEAN DEFAULT FALSE,

    -- Metadata
    first_added TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW(),
    last_alert TIMESTAMP,

    CONSTRAINT fk_ticker FOREIGN KEY (ticker)
        REFERENCES stock_instruments(ticker) ON DELETE CASCADE
);

CREATE INDEX idx_watchlist_rank ON stock_watchlist(rank);
CREATE INDEX idx_watchlist_score ON stock_watchlist(score DESC);
CREATE INDEX idx_watchlist_sector ON stock_watchlist(sector);
```

### 3.2 Daily Watchlist Update Process

```python
# stock_scanner/scripts/calculate_watchlist.py

async def update_watchlist(db: AsyncDatabaseManager):
    """
    Recalculate watchlist daily based on latest metrics.

    Steps:
    1. Calculate metrics for all stocks
    2. Score each stock
    3. Rank top N stocks
    4. Update watchlist table
    """

    # Calculate metrics from daily candles
    query = """
    WITH daily_metrics AS (
        SELECT
            ticker,

            -- Volume metrics (20-day)
            AVG(volume) as avg_volume,
            AVG(volume * close) as avg_dollar_volume,

            -- Latest volume vs average
            (SELECT volume FROM stock_daily_candles sdc2
             WHERE sdc2.ticker = sdc.ticker
             ORDER BY timestamp DESC LIMIT 1) / AVG(volume) as rvol,

            -- ATR (14-day)
            AVG(high - low) as atr_14,
            AVG((high - low) / close * 100) as atr_percent,

            -- Average daily range
            AVG((high - low) / close * 100) as avg_range_percent,

            -- Current price
            (SELECT close FROM stock_daily_candles sdc2
             WHERE sdc2.ticker = sdc.ticker
             ORDER BY timestamp DESC LIMIT 1) as current_price,

            -- Price changes
            ((SELECT close FROM stock_daily_candles sdc2
              WHERE sdc2.ticker = sdc.ticker
              ORDER BY timestamp DESC LIMIT 1) -
             (SELECT close FROM stock_daily_candles sdc2
              WHERE sdc2.ticker = sdc.ticker
              ORDER BY timestamp DESC LIMIT 1 OFFSET 1)) /
             (SELECT close FROM stock_daily_candles sdc2
              WHERE sdc2.ticker = sdc.ticker
              ORDER BY timestamp DESC LIMIT 1 OFFSET 1) * 100 as price_change_1d,

            -- Bollinger Band Width (20-day)
            (STDDEV(close) * 4) / AVG(close) * 100 as bb_width

        FROM stock_daily_candles sdc
        WHERE timestamp >= CURRENT_DATE - INTERVAL '20 days'
        GROUP BY ticker
    ),
    scored_stocks AS (
        SELECT
            dm.*,
            calculate_stock_score(
                dm.avg_volume::BIGINT,
                dm.avg_dollar_volume,
                dm.rvol,
                dm.atr_percent,
                dm.avg_range_percent,
                dm.bb_width,
                dm.current_price
            ) as score
        FROM daily_metrics dm
        WHERE
            -- Minimum filters
            dm.avg_dollar_volume >= 10000000  -- $10M minimum
            AND dm.current_price >= 5
            AND dm.current_price <= 1000
            AND dm.atr_percent >= 1.0
    )
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY score DESC) as rank
    FROM scored_stocks
    WHERE score >= 40  -- Minimum score threshold
    ORDER BY score DESC
    LIMIT 500;  -- Top 500 stocks
    """

    results = await db.fetch(query)

    # Update watchlist table
    await db.execute("TRUNCATE stock_watchlist")

    for row in results:
        await db.execute("""
            INSERT INTO stock_watchlist (
                ticker, rank, score, avg_volume, avg_dollar_volume,
                current_rvol, atr_14, atr_percent, avg_range_percent,
                bb_width, current_price, price_change_1d
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
            row['ticker'], row['rank'], row['score'],
            row['avg_volume'], row['avg_dollar_volume'],
            row['rvol'], row['atr_14'], row['atr_percent'],
            row['avg_range_percent'], row['bb_width'],
            row['current_price'], row['price_change_1d']
        )

    logger.info(f"Updated watchlist with {len(results)} stocks")
```

### 3.3 Watchlist Tiers

```python
# Segment watchlist into tiers for different strategies
WATCHLIST_TIERS = {
    'tier_1': {
        'name': 'Premium Liquidity',
        'score_min': 80,
        'count': 50,
        'description': 'Highest volume, highest movement stocks',
        'use_case': 'Day trading, scalping, quick entries/exits'
    },
    'tier_2': {
        'name': 'High Quality',
        'score_min': 65,
        'count': 150,
        'description': 'Very liquid, good movement',
        'use_case': 'Swing trading, position trading'
    },
    'tier_3': {
        'name': 'Tradeable',
        'score_min': 50,
        'count': 300,
        'description': 'Adequate liquidity and movement',
        'use_case': 'Longer-term positions, lower frequency'
    },
    'tier_4': {
        'name': 'Watchlist',
        'score_min': 40,
        'count': 500,
        'description': 'Minimum requirements met',
        'use_case': 'Monitoring, breakout opportunities'
    }
}
```

### 3.4 Sector Diversification

```python
# Ensure watchlist has sector diversification
MAX_STOCKS_PER_SECTOR = {
    'Technology': 100,
    'Financial': 80,
    'Healthcare': 70,
    'Consumer Cyclical': 60,
    'Consumer Defensive': 50,
    'Industrial': 50,
    'Energy': 40,
    'Other': 50
}

async def apply_sector_limits(stocks: List[dict]) -> List[dict]:
    """
    Limit stocks per sector while maintaining highest scores.
    """
    sector_counts = {}
    filtered = []

    for stock in sorted(stocks, key=lambda s: s['score'], reverse=True):
        sector = stock['sector']
        count = sector_counts.get(sector, 0)
        limit = MAX_STOCKS_PER_SECTOR.get(sector, 50)

        if count < limit:
            filtered.append(stock)
            sector_counts[sector] = count + 1

    return filtered
```

---

## 4. TECHNICAL IMPLEMENTATION

### 4.1 Database Indexes & Optimization

```sql
-- Essential indexes for fast queries
CREATE INDEX idx_candles_ticker_tf_time
    ON stock_candles(ticker, timeframe, timestamp DESC);

CREATE INDEX idx_candles_timeframe_time
    ON stock_candles(timeframe, timestamp DESC);

-- Partial index for recent data (faster queries)
CREATE INDEX idx_candles_recent
    ON stock_candles(ticker, timestamp DESC)
    WHERE timestamp >= CURRENT_DATE - INTERVAL '90 days';

-- Index for volume analysis
CREATE INDEX idx_candles_volume
    ON stock_candles(ticker, timestamp DESC, volume)
    WHERE timeframe = '1d';

-- Covering index for daily metrics
CREATE INDEX idx_daily_metrics
    ON stock_daily_candles(ticker, timestamp DESC)
    INCLUDE (open, high, low, close, volume);
```

### 4.2 Query Optimization Techniques

```sql
-- Use CTEs for complex calculations
WITH recent_candles AS (
    SELECT * FROM stock_daily_candles
    WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
),
volume_stats AS (
    SELECT
        ticker,
        AVG(volume) as avg_volume,
        STDDEV(volume) as volume_stddev
    FROM recent_candles
    GROUP BY ticker
)
SELECT ... FROM recent_candles rc
JOIN volume_stats vs USING (ticker)
WHERE rc.volume > vs.avg_volume * 1.5;

-- Use window functions for rankings
SELECT
    ticker,
    score,
    RANK() OVER (ORDER BY score DESC) as rank,
    PERCENT_RANK() OVER (ORDER BY score DESC) as percentile
FROM stock_watchlist;
```

### 4.3 Incremental Updates vs Full Recalculation

**Incremental Approach** (Recommended for daily):
```python
async def incremental_watchlist_update(db: AsyncDatabaseManager):
    """
    Update only changed metrics, faster than full recalc.
    """
    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).date()

    # Calculate metrics only for new data
    query = """
    UPDATE stock_watchlist w
    SET
        current_price = (
            SELECT close FROM stock_daily_candles
            WHERE ticker = w.ticker
            ORDER BY timestamp DESC LIMIT 1
        ),
        current_rvol = (
            SELECT volume / w.avg_volume
            FROM stock_daily_candles
            WHERE ticker = w.ticker
            AND timestamp::DATE = $1
        ),
        price_change_1d = (
            (SELECT close FROM stock_daily_candles
             WHERE ticker = w.ticker
             ORDER BY timestamp DESC LIMIT 1) -
            (SELECT close FROM stock_daily_candles
             WHERE ticker = w.ticker
             ORDER BY timestamp DESC LIMIT 1 OFFSET 1)
        ) / (SELECT close FROM stock_daily_candles
             WHERE ticker = w.ticker
             ORDER BY timestamp DESC LIMIT 1 OFFSET 1) * 100,
        last_updated = NOW()
    WHERE EXISTS (
        SELECT 1 FROM stock_daily_candles
        WHERE ticker = w.ticker
        AND timestamp::DATE = $1
    )
    """

    await db.execute(query, yesterday)
```

**Full Recalculation** (Weekly on Sunday):
```python
# Full recalc to ensure accuracy
# Run every Sunday to reset rankings
async def full_watchlist_recalc(db: AsyncDatabaseManager):
    """Full watchlist recalculation from scratch."""
    # Drop and rebuild watchlist
    await update_watchlist(db)  # Full calculation query
```

### 4.4 Caching Strategy

```python
from functools import lru_cache
from datetime import datetime, timedelta

class WatchlistCache:
    """
    Cache watchlist data to avoid repeated DB queries.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._cache = {}
        self._timestamps = {}

    def get(self, key: str):
        if key not in self._cache:
            return None

        # Check if expired
        if datetime.now() - self._timestamps[key] > timedelta(seconds=self.ttl):
            del self._cache[key]
            del self._timestamps[key]
            return None

        return self._cache[key]

    def set(self, key: str, value: any):
        self._cache[key] = value
        self._timestamps[key] = datetime.now()

    async def get_watchlist(self, db: AsyncDatabaseManager, tier: str = None):
        """Get cached watchlist or fetch from DB."""
        cache_key = f"watchlist_{tier or 'all'}"

        cached = self.get(cache_key)
        if cached:
            return cached

        # Fetch from DB
        query = """
            SELECT * FROM stock_watchlist
            WHERE ($1::VARCHAR IS NULL OR
                   score >= (SELECT score_min FROM watchlist_tiers WHERE name = $1))
            ORDER BY rank
        """

        results = await db.fetch(query, tier)
        self.set(cache_key, results)

        return results
```

---

## 5. ACTIONABLE OUTPUTS

### 5.1 Daily Watchlist Report

```python
async def generate_daily_report(db: AsyncDatabaseManager) -> dict:
    """
    Generate comprehensive daily watchlist report.
    """
    report = {
        'date': datetime.now().date(),
        'total_stocks': 0,
        'tiers': {},
        'top_movers': [],
        'high_volume': [],
        'breakouts': [],
        'sector_distribution': {}
    }

    # Get tier counts
    for tier_name, tier_config in WATCHLIST_TIERS.items():
        count = await db.fetchval("""
            SELECT COUNT(*) FROM stock_watchlist
            WHERE score >= $1
        """, tier_config['score_min'])

        report['tiers'][tier_name] = {
            'count': count,
            'score_min': tier_config['score_min']
        }

    # Top 10 movers (by score change)
    report['top_movers'] = await db.fetch("""
        SELECT ticker, score, price_change_1d, current_rvol
        FROM stock_watchlist
        ORDER BY ABS(price_change_1d) DESC
        LIMIT 10
    """)

    # High relative volume stocks
    report['high_volume'] = await db.fetch("""
        SELECT ticker, score, current_rvol, avg_dollar_volume
        FROM stock_watchlist
        WHERE current_rvol >= 1.5
        ORDER BY current_rvol DESC
        LIMIT 10
    """)

    # Sector distribution
    report['sector_distribution'] = await db.fetch("""
        SELECT sector, COUNT(*) as count, AVG(score) as avg_score
        FROM stock_watchlist
        GROUP BY sector
        ORDER BY count DESC
    """)

    return report
```

### 5.2 Strategy Integration

```python
# stock_scanner/core/strategies/daily_momentum.py

class DailyMomentumStrategy:
    """
    Example strategy using daily watchlist data.
    """

    async def scan_for_signals(self, db: AsyncDatabaseManager):
        """
        Scan watchlist for momentum signals.

        Signal criteria:
        - Stock in Tier 1 or Tier 2
        - Price above 20-day EMA
        - RVol > 1.3
        - ATR% > 2.0
        - Recent breakout (new 20-day high)
        """

        query = """
        SELECT
            w.ticker,
            w.score,
            w.current_price,
            w.current_rvol,
            w.atr_percent,

            -- Calculate EMA20 from daily candles
            (SELECT AVG(close) FROM (
                SELECT close FROM stock_daily_candles
                WHERE ticker = w.ticker
                ORDER BY timestamp DESC
                LIMIT 20
            ) sub) as ema_20,

            -- 20-day high
            (SELECT MAX(high) FROM stock_daily_candles
             WHERE ticker = w.ticker
             AND timestamp >= CURRENT_DATE - INTERVAL '20 days') as high_20d,

            -- Yesterday's close
            (SELECT close FROM stock_daily_candles
             WHERE ticker = w.ticker
             ORDER BY timestamp DESC
             LIMIT 1 OFFSET 1) as prev_close

        FROM stock_watchlist w
        WHERE
            w.score >= 65  -- Tier 2+
            AND w.current_rvol >= 1.3
            AND w.atr_percent >= 2.0
        """

        candidates = await db.fetch(query)

        signals = []
        for stock in candidates:
            # Check if price above EMA
            if stock['current_price'] < stock['ema_20']:
                continue

            # Check if breaking out
            if stock['current_price'] >= stock['high_20d'] * 0.99:  # Within 1% of high
                signals.append({
                    'ticker': stock['ticker'],
                    'signal': 'BREAKOUT_MOMENTUM',
                    'score': stock['score'],
                    'price': stock['current_price'],
                    'rvol': stock['current_rvol'],
                    'confidence': self._calculate_confidence(stock)
                })

        return signals
```

---

## 6. PERFORMANCE BENCHMARKS

### 6.1 Expected Query Times

```
Query Type                      | Target Time | Optimization
--------------------------------|-------------|---------------------------
Daily candle synthesis (1 stock)| <100ms      | Indexed timestamp + ticker
Daily candle synthesis (all)    | <5 minutes  | Parallel processing
Watchlist calculation (500)     | <30 seconds | Materialized view + CTEs
Single stock metrics            | <50ms       | Covering indexes
Real-time watchlist lookup      | <10ms       | Memory cache
Strategy signal scan            | <2 seconds  | Pre-filtered watchlist
```

### 6.2 Storage Estimates

```
Data Type              | Count       | Size per Row | Total Size
-----------------------|-------------|--------------|------------
1H Candles (current)   | 1.37M       | ~150 bytes   | ~200 MB
Daily Candles (synth)  | 68,000      | ~150 bytes   | ~10 MB
Watchlist              | 500         | ~500 bytes   | ~250 KB
Metrics Cache          | 3,428       | ~300 bytes   | ~1 MB

Total: ~211 MB (very manageable)
```

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1)
- [ ] Create materialized view for daily candles
- [ ] Implement synthesis refresh script
- [ ] Add necessary database indexes
- [ ] Test synthesis accuracy vs yfinance daily data

### Phase 2: Metrics & Scoring (Week 2)
- [ ] Implement scoring function
- [ ] Create watchlist table
- [ ] Build daily metrics calculation script
- [ ] Validate scoring against manual analysis

### Phase 3: Automation (Week 3)
- [ ] Set up cron jobs for daily updates
- [ ] Implement incremental update logic
- [ ] Add monitoring & alerting
- [ ] Create daily report generation

### Phase 4: Strategy Integration (Week 4)
- [ ] Build example strategies using watchlist
- [ ] Integrate with existing signal detection
- [ ] Add backtest support for daily timeframe
- [ ] Performance optimization & tuning

---

## 8. EXAMPLE USAGE

### 8.1 Get Today's Top Stocks

```python
# Get top 20 stocks for trading today
async def get_todays_opportunities(db: AsyncDatabaseManager):
    query = """
    SELECT
        ticker,
        score,
        current_price,
        current_rvol,
        atr_percent,
        price_change_1d,
        sector
    FROM stock_watchlist
    WHERE
        score >= 70
        AND current_rvol >= 1.2
        AND atr_percent >= 2.0
    ORDER BY score DESC
    LIMIT 20
    """

    return await db.fetch(query)
```

### 8.2 Monitor Specific Stock

```python
async def get_stock_context(db: AsyncDatabaseManager, ticker: str):
    """
    Get comprehensive context for a stock.
    """
    query = """
    SELECT
        w.*,
        -- Get last 5 daily candles
        ARRAY(
            SELECT ROW(timestamp, open, high, low, close, volume)
            FROM stock_daily_candles
            WHERE ticker = w.ticker
            ORDER BY timestamp DESC
            LIMIT 5
        ) as recent_candles,

        -- Percentile rank
        PERCENT_RANK() OVER (ORDER BY score) as percentile

    FROM stock_watchlist w
    WHERE w.ticker = $1
    """

    return await db.fetchrow(query, ticker)
```

---

## 9. MONITORING & ALERTS

### 9.1 Data Quality Monitoring

```python
# Daily checks
async def check_data_quality(db: AsyncDatabaseManager):
    """
    Run daily data quality checks.
    """
    checks = {
        'missing_daily_candles': await db.fetchval("""
            SELECT COUNT(DISTINCT ticker)
            FROM stock_instruments
            WHERE is_active = TRUE
            AND ticker NOT IN (
                SELECT DISTINCT ticker
                FROM stock_daily_candles
                WHERE timestamp::DATE = CURRENT_DATE - 1
            )
        """),

        'incomplete_candles': await db.fetchval("""
            SELECT COUNT(*)
            FROM stock_daily_quality
            WHERE date = CURRENT_DATE - 1
            AND quality = 'INCOMPLETE'
        """),

        'watchlist_staleness': await db.fetchval("""
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(last_updated))) / 3600
            FROM stock_watchlist
        """)
    }

    # Alert if issues found
    if checks['missing_daily_candles'] > 100:
        await send_alert(f"Missing daily candles for {checks['missing_daily_candles']} stocks!")

    return checks
```

---

## 10. SUMMARY

### Key Decisions

1. **Synthesis Method**: Materialized view for daily candles (fast, efficient)
2. **Volume Threshold**: $10M minimum daily dollar volume
3. **Movement Threshold**: 1.5% minimum ATR percentage
4. **Watchlist Size**: Top 500 stocks, tiered by score
5. **Update Schedule**: Daily after market close
6. **Scoring System**: 0-100 scale based on volume + volatility
7. **Database Strategy**: Indexed materialized views + caching

### Expected Results

- **500 high-quality stocks** in watchlist
- **Sub-second query times** for strategy scans
- **Automatic daily updates** with quality validation
- **Sector-diversified** opportunity set
- **Scored & ranked** for easy prioritization

### Next Steps

1. Review and approve architecture
2. Create database migrations
3. Implement synthesis scripts
4. Build watchlist calculation
5. Test with historical data
6. Deploy to production
7. Monitor performance

---

**END OF DOCUMENT**
