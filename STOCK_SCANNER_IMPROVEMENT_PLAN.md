# Stock Scanner Improvement Plan

## Executive Summary

After analyzing the stock scanner system, I've identified the current state and recommendations for improvement. The system has a solid foundation with 6-stage daily pipeline, multiple scanner strategies, Claude AI integration, and comprehensive data storage. However, there are gaps in scheduling, signal freshness, and additional scanning opportunities.

---

## Current System Analysis

### Architecture Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  stock-scanner    │────▶│     PostgreSQL      │◀────│     Streamlit       │
│  (task-worker)      │     │    (stocks DB)      │     │   (stock_scanner)   │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         │
         ▼
   Daily Pipeline (10:30 PM ET Mon-Fri)
   1. sync       → yfinance 1H candles
   2. synthesize → 1H → Daily aggregation
   3. metrics    → ATR, RSI, MACD, volume
   4. smc        → Smart Money Concepts
   5. watchlist  → Tiered scoring (1-4)
   6. signals    → ZLMA strategy signals
```

### Current State (as of 2025-12-09)

| Component | Status | Data Freshness |
|-----------|--------|----------------|
| stock-scanner | Running | Up 2 days |
| Latest candles | 2025-12-08 20:30 | ~16 hours old |
| Watchlist | 2025-12-08 | Yesterday's data |
| ZLMA signals | 74 signals | 2025-12-05 only |
| Scanner signals | 54 signals | 2025-12-08 |
| Stocks analyzed | 3,427 | NYSE/NASDAQ |

### Key Observations

1. **Schedule Timing Issue**: Pipeline runs at 10:30 PM ET but last ran at 3:30 AM (Dec 9). Next run scheduled for 10:30 PM tonight. You won't see today's signals until after market close.

2. **ZLMA Signals Stale**: Only 74 ZLMA signals from Dec 5 - not being generated in recent runs (0 signals on Dec 8-9).

3. **Scanner Signals Working**: 54 scanner signals from Dec 8 - the 4 scanner strategies are functioning.

4. **Rate Limiting**: Yahoo Finance rate limits causing some ticker updates to fail (~2000 tickers per run getting rate limited).

---

## Issues Identified

### Issue 1: Signals Not Appearing Daily
**Problem**: You're not seeing new signals daily because:
- Pipeline only runs once at 10:30 PM ET (after market close)
- ZLMA strategy producing 0 signals (possible configuration/data issue)
- Scanner strategies producing signals but only on pipeline run

**Expected Behavior**: New signals should appear each trading day with fresh market data.

### Issue 2: ZLMA Strategy Not Generating Signals
**Problem**: ZLMA strategy scan returned 0 signals on Dec 8-9 runs.
**Root Cause**: Needs investigation - could be:
- Strategy conditions too strict
- Missing or incomplete candle data
- Timeframe alignment issues

### Issue 3: Single Daily Run Timing
**Problem**: Running only at 10:30 PM means:
- No intraday signal updates
- Pre-market gaps not captured immediately
- Users see yesterday's signals all day

### Issue 4: Rate Limiting with yfinance
**Problem**: ~500-2000 tickers failing due to Yahoo Finance rate limits per run.

---

## Improvement Plan

### Phase 1: Fix Signal Generation (Immediate - 1-2 days)

#### 1.1 Diagnose ZLMA Strategy
```bash
# Run ZLMA manually with debug logging
docker exec task-worker python -c "
import asyncio
import logging
logging.basicConfig(level=logging.DEBUG)
from stock_scanner.strategies.zlma_trend import ZeroLagMATrendStrategy
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager

async def debug():
    db = AsyncDatabaseManager('postgresql://postgres:postgres@postgres:5432/stocks')
    await db.connect()
    strategy = ZeroLagMATrendStrategy(db_manager=db)
    signals = await strategy.scan_all_stocks()
    print(f'Generated {len(signals)} signals')
    await db.close()

asyncio.run(debug())
"
```

#### 1.2 Run Scanner Manager to Generate Fresh Signals
```bash
# Run all 4 scanner strategies
docker exec task-worker python -c "
import asyncio
from stock_scanner.scanners.scanner_manager import ScannerManager
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager

async def run():
    db = AsyncDatabaseManager('postgresql://postgres:postgres@postgres:5432/stocks')
    await db.connect()
    manager = ScannerManager(db)
    await manager.initialize()
    signals = await manager.run_all_scanners()
    print(f'Generated {len(signals)} total signals')
    await db.close()

asyncio.run(run())
"
```

### Phase 2: Add Multiple Daily Scans (1 week)

#### 2.1 Pre-Market Scan (6:00 AM ET)
Add a lightweight scan before market open to capture:
- Overnight news impact
- Pre-market gaps
- Earnings announcements from previous night

#### 2.2 Intraday Scan (12:30 PM ET)
Mid-day check for:
- Morning momentum plays
- Reversal setups
- Volume breakouts

#### 2.3 After-Hours Scan (4:30 PM ET)
Quick scan right after close for:
- EOD breakouts/breakdowns
- Unusual volume
- Technical pattern completions

#### Implementation
```python
# scheduler.py additions
PRE_MARKET_TIME = time(6, 0)   # 6:00 AM ET - light scan
INTRADAY_TIME = time(12, 30)   # 12:30 PM ET - scanner only
POST_MARKET_TIME = time(16, 30) # 4:30 PM ET - quick scan
FULL_PIPELINE_TIME = time(22, 30) # 10:30 PM ET - full pipeline
```

### Phase 3: Enhanced Scanning Strategies (2 weeks)

#### 3.1 New Scanner: Earnings Momentum
**Purpose**: Catch stocks moving on earnings releases
```python
class EarningsMomentumScanner(BaseScanner):
    """
    Scans for stocks with:
    - Earnings release within last 24 hours
    - Price gap > 5%
    - Volume > 2x average
    - Continuation pattern forming
    """
```

#### 3.2 New Scanner: Short Squeeze Detector
**Purpose**: Identify potential short squeeze candidates
```python
class ShortSqueezeScanner(BaseScanner):
    """
    Scans for stocks with:
    - Short interest > 15%
    - Days to cover > 3
    - Breaking above resistance
    - Unusual volume surge
    """
```

#### 3.3 New Scanner: Sector Rotation
**Purpose**: Identify sector strength/weakness for rotation plays
```python
class SectorRotationScanner(BaseScanner):
    """
    Scans for:
    - Strong sectors: Leaders within top 3 performing sectors
    - Weak sectors: Short candidates in bottom 3 sectors
    - Rotation signals when leadership changes
    """
```

#### 3.4 New Scanner: Institutional Flow
**Purpose**: Track unusual institutional activity
```python
class InstitutionalFlowScanner(BaseScanner):
    """
    Scans for stocks with:
    - Unusual options activity (via external API)
    - Dark pool prints > 10% of daily volume
    - Institutional ownership changes
    """
```

### Phase 4: Real-Time Data Enhancement (3-4 weeks)

#### 4.1 Alternative Data Source
Replace/supplement yfinance with more reliable source:
- **Polygon.io**: Real-time data, no rate limits (paid)
- **Alpha Vantage**: Free tier with higher limits
- **IEX Cloud**: Real-time quotes and fundamentals

#### 4.2 WebSocket Integration
For intraday scanning:
```python
class RealTimeDataFeed:
    """
    WebSocket connection for real-time price updates
    - Track price alerts
    - Volume spike detection
    - Pattern completion triggers
    """
```

### Phase 5: Signal Quality Improvements (Ongoing)

#### 5.1 Backtest Integration
Track actual performance of signals:
- Win rate by scanner
- Win rate by quality tier
- Average R-multiple achieved
- Time to target

#### 5.2 Adaptive Scoring
Adjust scoring weights based on:
- Recent market regime
- Sector performance
- Historical signal accuracy

#### 5.3 Claude AI Enhancement
- Auto-analyze all A+ and A tier signals
- Generate daily market summary
- Provide sector analysis

---

## Immediate Action Items

### Today (Critical)
1. **Verify Pipeline Will Run Tonight**
   ```bash
   docker logs stock-scanner --tail 5
   # Should show: "Next task: pipeline at 2025-12-09 22:30 EST"
   ```

2. **Manually Trigger Scanner Run**
   ```bash
   docker exec task-worker python -m stock_scanner.main pipeline
   ```

3. **Check Signal Generation**
   ```sql
   -- Run in postgres container
   SELECT signal_timestamp::date, COUNT(*)
   FROM stock_scanner_signals
   GROUP BY 1 ORDER BY 1 DESC LIMIT 5;
   ```

### This Week
1. Fix ZLMA strategy signal generation issue
2. Add pre-market scan (6 AM ET)
3. Add post-market quick scan (4:30 PM ET)
4. Implement signal performance tracking

### This Month
1. Build earnings momentum scanner
2. Build short squeeze scanner
3. Implement alternative data source
4. Add automated Claude analysis for top signals

---

## Configuration Changes Needed

### scheduler.py
```python
# Add multiple scan times
SCAN_SCHEDULE = {
    'pre_market': {'time': time(6, 0), 'type': 'light'},
    'intraday': {'time': time(12, 30), 'type': 'scanner_only'},
    'post_market': {'time': time(16, 30), 'type': 'quick'},
    'full_pipeline': {'time': time(22, 30), 'type': 'full'},
}
```

### docker-compose.yml
No changes needed - stock-scanner already configured to run continuously.

---

## Expected Outcomes

After implementing these improvements:

| Metric | Current | Expected |
|--------|---------|----------|
| Scan frequency | 1x/day | 4x/day |
| Signal freshness | 24h | <6h |
| Scanner strategies | 4 | 6-8 |
| Daily signals | ~50 | 100-200 |
| Signal accuracy tracking | None | Full |
| Auto Claude analysis | Manual | Automated |

---

## Monitoring Dashboard Additions

Consider adding to Streamlit:
1. **Pipeline Status Widget**: Last run time, next run time, status
2. **Signal Freshness Indicator**: Hours since last signal generation
3. **Scanner Health Check**: Each scanner's last successful run
4. **Performance Tracker**: Win rate, P/F by scanner over time

---

## Questions for Clarification

1. Do you want intraday signals or is end-of-day sufficient?
2. Budget for real-time data API subscription?
3. Priority: More signals or higher quality signals?
4. Interest in options flow data integration?
