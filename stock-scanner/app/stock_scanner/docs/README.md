# Stock Scanner Documentation

**Daily Timeframe Stock Scanner System**
High-volume, high-movement stock identification and analysis

---

## Quick Start

```bash
# 1. Create daily candles materialized view
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily --create

# 2. Calculate initial watchlist
docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist

# 3. View top stocks
docker exec postgres psql -U postgres -d stocks -c \
  "SELECT rank, ticker, score, tier FROM stock_watchlist ORDER BY rank LIMIT 20;"
```

---

## Documentation Index

### 📘 Overview Documents

1. **[Data Flow Diagram](DATA_FLOW_DIAGRAM.md)** - Visual representation of entire pipeline
   - Complete data flow from raw data to signals
   - Daily schedule and timing
   - Performance metrics

2. **[Quick Start Guide](DAILY_SCANNER_QUICKSTART.md)** - Fast reference
   - Common commands
   - Database queries
   - Troubleshooting tips

### 📖 Detailed Documentation

3. **[Architecture Overview](DAILY_SCANNER_ARCHITECTURE.md)** - Complete system design
   - Data pipeline design
   - Stock filtering criteria
   - Watchlist management
   - Technical implementation
   - Performance optimization

### 📋 Summary

4. **[Implementation Summary](/home/hr/Projects/TradeSystemV1/STOCK_SCANNER_SUMMARY.md)** - Executive overview
   - System components
   - Key features
   - Usage examples
   - Implementation roadmap

---

## System Overview

### What It Does

The stock scanner identifies **high-quality trading opportunities** from 3,428 US stocks by:

1. Synthesizing hourly data into daily candles
2. Calculating volume and volatility metrics
3. Scoring stocks on 0-100 scale
4. Maintaining a tiered watchlist of top 500 stocks
5. Generating actionable trading signals

### Key Features

- **Efficient**: Sub-second queries, 35-minute daily pipeline
- **Smart Filtering**: Multi-factor scoring (volume + volatility)
- **Actionable**: Pre-scored stocks with risk levels
- **Automated**: Runs after market close daily
- **Scalable**: Handles 3,428 stocks, can scale to 10,000+

### Data Pipeline

```
1H Candles (yfinance)
    ↓
Daily Candles (materialized view)
    ↓
Metrics Calculation (20-day)
    ↓
Stock Scoring (0-100)
    ↓
Watchlist (Top 500, 4 tiers)
    ↓
Strategy Signals
```

---

## Core Components

### Scripts

| Script | Purpose | Schedule |
|--------|---------|----------|
| `daily_update.py` | Fetch hourly data | 11:00 PM UTC (6 PM ET) |
| `synthesize_daily.py` | Create daily candles | 11:30 PM UTC (6:30 PM ET) |
| `calculate_watchlist.py` | Compute watchlist | 12:00 AM UTC (7 PM ET) |

### Database Tables

| Table | Records | Purpose |
|-------|---------|---------|
| `stock_candles` | 1.37M | Raw 1H candles |
| `stock_daily_candles` | 68K | Daily candles (view) |
| `stock_watchlist` | 500 | Scored stocks |
| `stock_instruments` | 3,428 | Stock metadata |

### Strategies

| Strategy | File | Purpose |
|----------|------|---------|
| Daily Momentum Breakout | `daily_momentum_breakout.py` | Trend breakout signals |

---

## Watchlist Tiers

| Tier | Score | Count | Use Case |
|------|-------|-------|----------|
| 1 | 80-100 | ~50 | Day trading |
| 2 | 65-79 | ~150 | Swing trading |
| 3 | 50-64 | ~300 | Position trading |
| 4 | 40-49 | ~500 | Monitoring |

---

## Filtering Criteria

### Volume
- **Minimum**: $10M daily dollar volume
- **Liquid**: $25M+ (quality flag)
- **Premium**: $50M+ (Tier 1/2)

### Movement
- **Minimum**: 1.0% ATR
- **Preferred**: 1.5% ATR
- **High**: 2.0%+ ATR

### Price
- **Range**: $5 - $1,000
- **Optimal**: $10 - $500

---

## Common Queries

### Top Stocks
```sql
SELECT rank, ticker, score, tier
FROM stock_watchlist
ORDER BY rank LIMIT 20;
```

### High Volume Stocks
```sql
SELECT ticker, current_rvol, score
FROM stock_watchlist
WHERE current_rvol >= 1.5
ORDER BY current_rvol DESC;
```

### Tier 1 Stocks
```sql
SELECT ticker, score, current_price
FROM stock_watchlist
WHERE tier = 1
ORDER BY rank;
```

---

## Usage Examples

### Python Integration

```python
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner import config

async def get_top_stocks():
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    stocks = await db.fetch("""
        SELECT * FROM stock_watchlist
        WHERE tier <= 2
        ORDER BY rank LIMIT 50
    """)

    await db.close()
    return stocks
```

### Strategy Usage

```python
from stock_scanner.core.strategies.daily_momentum_breakout import DailyMomentumBreakout

strategy = DailyMomentumBreakout(db_manager)
signals = await strategy.scan_for_signals()
```

---

## Performance

| Metric | Value |
|--------|-------|
| Total data size | ~211 MB |
| Daily pipeline | ~35 minutes |
| Watchlist query | <10ms |
| Strategy scan | <2 seconds |
| Daily synthesis | <5 minutes |

---

## File Structure

```
stock_scanner/
├── docs/
│   ├── README.md                          (this file)
│   ├── DAILY_SCANNER_ARCHITECTURE.md      (complete design)
│   ├── DAILY_SCANNER_QUICKSTART.md        (quick reference)
│   └── DATA_FLOW_DIAGRAM.md               (visual flow)
│
├── scripts/
│   ├── synthesize_daily.py                (1H → daily)
│   ├── calculate_watchlist.py             (metrics & scoring)
│   └── daily_update.py                    (data fetch)
│
├── core/
│   ├── strategies/
│   │   └── daily_momentum_breakout.py     (example strategy)
│   └── database/
│       └── async_database_manager.py
│
└── config.py                              (configuration)
```

---

## Monitoring

### Data Quality
```bash
# Validate synthesis
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily --validate-only
```

### Check Freshness
```sql
SELECT MAX(last_updated) FROM stock_watchlist;
SELECT MAX(timestamp)::DATE FROM stock_daily_candles;
```

### View Statistics
```sql
SELECT tier, COUNT(*), AVG(score)
FROM stock_watchlist
GROUP BY tier;
```

---

## Troubleshooting

### Issue: No daily candles

**Solution**:
```bash
docker exec worker-app python -m stock_scanner.scripts.synthesize_daily --create
```

### Issue: Empty watchlist

**Check**:
```sql
SELECT COUNT(*) FROM stock_daily_candles;
```

**Recalculate**:
```bash
docker exec worker-app python -m stock_scanner.scripts.calculate_watchlist
```

### Issue: Slow queries

**Solution**:
```sql
ANALYZE stock_daily_candles;
REINDEX TABLE stock_watchlist;
```

---

## Next Steps

1. **Initial Setup**
   - Create materialized view
   - Calculate first watchlist
   - Verify results

2. **Automation**
   - Set up cron jobs
   - Monitor execution
   - Set up alerts

3. **Integration**
   - Add to trading strategies
   - Connect to signal system
   - Enable notifications

4. **Optimization**
   - Add Redis caching
   - Tune query performance
   - Scale to more stocks

---

## Support

### Getting Help

1. Check this documentation
2. Review architecture document
3. Run validation scripts
4. Check logs: `/app/logs/stock_scanner.log`

### Key Contacts

- Architecture questions: Review `DAILY_SCANNER_ARCHITECTURE.md`
- Quick reference: See `DAILY_SCANNER_QUICKSTART.md`
- Visual overview: Check `DATA_FLOW_DIAGRAM.md`

---

## Version

**Version**: 1.0
**Date**: 2025-12-07
**Status**: Production Ready

---

## License

Part of TradeSystemV1
