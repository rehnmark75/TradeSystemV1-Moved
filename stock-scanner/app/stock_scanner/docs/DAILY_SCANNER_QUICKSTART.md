# Daily Stock Scanner - Quick Start Guide

## Overview

The daily stock scanner identifies **high-volume, high-movement stocks** from 3,428 US stocks, synthesizes hourly data into daily candles, and maintains a scored watchlist for trading.

---

## Quick Commands

### 1. Initial Setup (One-Time)

```bash
# Create materialized view for daily candles
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily --create

# Initial watchlist calculation
docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist
```

### 2. Daily Operations

```bash
# Step 1: Update hourly data (after market close)
docker exec worker-app python -m stock_scanner.scripts.daily_update --concurrency 10

# Step 2: Synthesize daily candles
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily

# Step 3: Calculate watchlist
docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist
```

### 3. View Watchlist

```bash
# Connect to database
docker exec postgres psql -U postgres -d stocks

# View top 20 stocks
SELECT rank, ticker, score, current_price, atr_percent, tier
FROM stock_watchlist
ORDER BY rank
LIMIT 20;

# View Tier 1 stocks (premium)
SELECT ticker, score, current_price, avg_dollar_volume
FROM stock_watchlist
WHERE tier = 1
ORDER BY rank;

# Export to CSV
docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist --export
```

---

## Watchlist Tiers

| Tier | Score Range | Count | Description | Use Case |
|------|-------------|-------|-------------|----------|
| 1 | 80-100 | ~50 | Premium liquidity | Day trading, scalping |
| 2 | 65-79 | ~150 | High quality | Swing trading |
| 3 | 50-64 | ~300 | Tradeable | Position trading |
| 4 | 40-49 | ~500 | Watchlist | Monitoring |

---

## Filtering Criteria

### Volume Requirements
- **Minimum**: $10M daily dollar volume
- **Preferred**: $25M+ (high liquidity flag)
- **Premium**: $50M+ (Tier 1/2)

### Movement Requirements
- **Minimum**: 1.0% ATR
- **Preferred**: 1.5% ATR
- **High volatility**: 2.0%+ ATR

### Price Range
- **Minimum**: $5.00
- **Maximum**: $1,000.00
- **Optimal**: $10-$500

---

## Daily Schedule

### Automated Cron Jobs

```bash
# Add to crontab
# Run after US market close (times in UTC)

# 11:00 PM UTC (6:00 PM ET) - Update hourly data
0 23 * * 1-5 docker exec worker-app python -m stock_scanner.scripts.daily_update --concurrency 10

# 11:30 PM UTC (6:30 PM ET) - Synthesize daily candles
30 23 * * 1-5 docker exec worker-app python -m stock_scanner.scripts.synthesize_daily

# 12:00 AM UTC (7:00 PM ET) - Calculate watchlist
0 0 * * 2-6 docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist
```

---

## Database Queries

### Get Top Stocks by Score

```sql
SELECT
    rank,
    ticker,
    score,
    current_price,
    atr_percent,
    avg_dollar_volume / 1000000 as dollar_volume_m,
    tier
FROM stock_watchlist
ORDER BY score DESC
LIMIT 50;
```

### Find High Relative Volume Stocks

```sql
SELECT
    ticker,
    score,
    current_price,
    current_rvol,
    avg_dollar_volume,
    price_change_1d
FROM stock_watchlist
WHERE current_rvol >= 1.5  -- 50% above average
ORDER BY current_rvol DESC
LIMIT 20;
```

### Find Trending Stocks

```sql
SELECT
    ticker,
    score,
    current_price,
    price_change_1d,
    price_change_5d,
    price_change_20d
FROM stock_watchlist
WHERE is_trending = TRUE
  AND price_change_20d > 0  -- Uptrend
ORDER BY price_change_20d DESC
LIMIT 20;
```

### Sector Breakdown

```sql
SELECT
    si.sector,
    COUNT(*) as stock_count,
    AVG(w.score) as avg_score,
    SUM(CASE WHEN w.tier = 1 THEN 1 ELSE 0 END) as tier_1_count
FROM stock_watchlist w
JOIN stock_instruments si ON w.ticker = si.ticker
GROUP BY si.sector
ORDER BY stock_count DESC;
```

### Daily Candle Analysis

```sql
-- Get daily candles for a specific stock
SELECT
    timestamp::DATE as date,
    open,
    high,
    low,
    close,
    volume,
    (high - low) / close * 100 as daily_range_pct
FROM stock_daily_candles
WHERE ticker = 'AAPL'
ORDER BY timestamp DESC
LIMIT 30;
```

### Volume Leaders

```sql
SELECT
    ticker,
    score,
    avg_volume / 1000000 as avg_volume_m,
    avg_dollar_volume / 1000000 as dollar_volume_m,
    current_rvol
FROM stock_watchlist
WHERE is_liquid = TRUE
ORDER BY avg_dollar_volume DESC
LIMIT 20;
```

---

## Python Integration

### Get Watchlist in Code

```python
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner import config

async def get_tier1_stocks():
    """Get Tier 1 stocks for day trading."""
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    try:
        stocks = await db.fetch("""
            SELECT ticker, score, current_price, atr_percent
            FROM stock_watchlist
            WHERE tier = 1
            ORDER BY rank
        """)

        return [dict(s) for s in stocks]

    finally:
        await db.close()


async def get_breakout_candidates():
    """Find stocks near 20-day highs."""
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    try:
        candidates = await db.fetch("""
            SELECT
                w.ticker,
                w.score,
                w.current_price,
                w.atr_percent,
                dc.high_20d
            FROM stock_watchlist w
            JOIN LATERAL (
                SELECT MAX(high) as high_20d
                FROM stock_daily_candles
                WHERE ticker = w.ticker
                AND timestamp >= CURRENT_DATE - INTERVAL '20 days'
            ) dc ON true
            WHERE w.tier <= 2
              AND w.current_price >= dc.high_20d * 0.98  -- Within 2% of high
            ORDER BY w.score DESC
        """)

        return [dict(c) for c in candidates]

    finally:
        await db.close()
```

---

## Monitoring

### Check Data Freshness

```sql
-- Check when watchlist was last updated
SELECT
    MAX(last_updated) as last_update,
    EXTRACT(EPOCH FROM (NOW() - MAX(last_updated))) / 3600 as hours_ago
FROM stock_watchlist;

-- Check latest daily candle date
SELECT
    MAX(timestamp)::DATE as latest_date,
    COUNT(DISTINCT ticker) as ticker_count
FROM stock_daily_candles;

-- Check data quality
SELECT
    timestamp::DATE as date,
    COUNT(*) as candles,
    COUNT(DISTINCT ticker) as tickers,
    AVG(hourly_candles) as avg_hours
FROM stock_daily_candles
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY timestamp::DATE
ORDER BY date DESC;
```

### Validate Synthesis

```bash
# Run validation only
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily --validate-only
```

---

## Troubleshooting

### No Daily Candles

```bash
# Check if materialized view exists
docker exec postgres psql -U postgres -d stocks -c "\dm"

# If missing, create it
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily --create
```

### Empty Watchlist

```sql
-- Check if daily candles exist
SELECT COUNT(*) FROM stock_daily_candles;

-- Check if stocks meet minimum criteria
SELECT COUNT(*)
FROM (
    SELECT ticker, AVG(volume * close) as avg_dv
    FROM stock_daily_candles
    WHERE timestamp >= CURRENT_DATE - INTERVAL '20 days'
    GROUP BY ticker
    HAVING AVG(volume * close) >= 10000000
) sub;
```

### Slow Queries

```sql
-- Analyze materialized view
ANALYZE stock_daily_candles;

-- Rebuild indexes
REINDEX TABLE stock_daily_candles;
REINDEX TABLE stock_watchlist;
```

---

## Performance Tips

1. **Refresh Concurrent**: Use `CONCURRENTLY` for non-blocking refresh
   ```bash
   docker exec worker-app python -m stock_scanner.scripts.synthesize_daily
   ```

2. **Cache Watchlist**: In production, cache watchlist in Redis
   ```python
   # Cache for 1 hour
   redis.setex('watchlist:tier1', 3600, json.dumps(stocks))
   ```

3. **Partial Updates**: For intraday, only recalculate RVol
   ```sql
   UPDATE stock_watchlist SET current_rvol = ...
   ```

4. **Use Indexes**: Always filter by tier or score
   ```sql
   -- Good (uses index)
   SELECT * FROM stock_watchlist WHERE tier = 1;

   -- Bad (full scan)
   SELECT * FROM stock_watchlist WHERE current_price > 100;
   ```

---

## Next Steps

1. **Review Architecture**: Read `DAILY_SCANNER_ARCHITECTURE.md` for full details
2. **Run Initial Setup**: Create materialized view and calculate first watchlist
3. **Set Up Cron**: Automate daily updates
4. **Integrate Strategies**: Use watchlist in trading strategies
5. **Monitor Performance**: Track query times and data quality

---

## Support

For issues or questions:
1. Check logs: `/app/logs/stock_scanner.log`
2. Validate data: Run `--validate-only` scripts
3. Review architecture documentation
