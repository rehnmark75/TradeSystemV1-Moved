# Stock Scanner Data Flow Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STOCK SCANNER SYSTEM                             │
│                   Daily Timeframe Analysis Pipeline                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA COLLECTION (Daily - After Market Close)                    │
└─────────────────────────────────────────────────────────────────────────┘

    yfinance API                RoboMarkets API
         │                              │
         │ Fetch 1H candles             │ Stock metadata
         │ (730 days history)           │ (NYSE/NASDAQ)
         ▼                              ▼
    ┌─────────────────────────────────────────┐
    │     daily_update.py                     │
    │  - Incremental fetch                    │
    │  - Concurrency: 5-10                    │
    │  - Duration: ~30 min                    │
    └─────────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────┐
    │   PostgreSQL: stock_candles             │
    │                                         │
    │   ticker    | timeframe | timestamp    │
    │   ------    | --------- | ----------   │
    │   AAPL      | 1h        | 2025-12-07   │
    │   AAPL      | 1h        | 2025-12-06   │
    │   ...       | ...       | ...          │
    │                                         │
    │   Total: 1,367,308 rows                │
    │   Tickers: 3,430                       │
    └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: CANDLE SYNTHESIS (Daily - After 1H Update)                      │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────┐
    │   stock_candles (1H)                    │
    │   - 1,367,308 rows                      │
    │   - Raw hourly data                     │
    └─────────────────────────────────────────┘
                    │
                    │ synthesize_daily.py
                    │ Aggregation Logic:
                    │ - Open: First 1H open
                    │ - High: Max of all 1H highs
                    │ - Low: Min of all 1H lows
                    │ - Close: Last 1H close
                    │ - Volume: Sum of all 1H volumes
                    ▼
    ┌─────────────────────────────────────────┐
    │   MATERIALIZED VIEW:                    │
    │   stock_daily_candles                   │
    │                                         │
    │   SELECT                                │
    │     ticker,                             │
    │     DATE_TRUNC('day', timestamp),       │
    │     FIRST(open) as open,                │
    │     MAX(high) as high,                  │
    │     MIN(low) as low,                    │
    │     LAST(close) as close,               │
    │     SUM(volume) as volume               │
    │   FROM stock_candles                    │
    │   WHERE timeframe = '1h'                │
    │   GROUP BY ticker, day                  │
    │                                         │
    │   Total: ~68,000 daily candles          │
    │   Duration: ~2-3 minutes                │
    └─────────────────────────────────────────┘
                    │
                    │ Indexes created:
                    │ - (ticker, timestamp)
                    │ - (timestamp)
                    │ - (ticker, timestamp, volume)
                    ▼
    ┌─────────────────────────────────────────┐
    │   VALIDATION                            │
    │   - Check hourly_candles count          │
    │   - Verify completeness (>= 6 hours)    │
    │   - Identify missing data               │
    └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: METRICS CALCULATION (Daily - After Synthesis)                   │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────┐
    │   stock_daily_candles                   │
    │   (Last 20 days for each ticker)        │
    └─────────────────────────────────────────┘
                    │
                    │ calculate_watchlist.py
                    │ Calculate for each ticker:
                    ▼
    ┌─────────────────────────────────────────┐
    │   METRICS CALCULATION                   │
    │                                         │
    │   Volume Metrics (20-day):              │
    │   - Avg Volume                          │
    │   - Avg Dollar Volume                   │
    │   - RVol (current / avg)                │
    │                                         │
    │   Volatility Metrics:                   │
    │   - ATR (14-day)                        │
    │   - ATR Percent                         │
    │   - Avg Range Percent                   │
    │   - Bollinger Band Width                │
    │                                         │
    │   Price Metrics:                        │
    │   - Current Price                       │
    │   - 1-day change %                      │
    │   - 5-day change %                      │
    │   - 20-day change %                     │
    │                                         │
    │   Duration: ~20-30 seconds              │
    └─────────────────────────────────────────┘
                    │
                    │ Apply Filters:
                    │ - Dollar volume >= $10M
                    │ - Price: $5 - $1,000
                    │ - ATR >= 1.0%
                    ▼
    ┌─────────────────────────────────────────┐
    │   SCORING ENGINE                        │
    │                                         │
    │   calculate_stock_score()               │
    │                                         │
    │   Components (0-100 scale):             │
    │   ┌──────────────────────────────────┐  │
    │   │ Volume (30 pts)                  │  │
    │   │ - $100M+: 30 pts                 │  │
    │   │ - $50M+:  25 pts                 │  │
    │   │ - $25M+:  20 pts                 │  │
    │   │ - $10M+:  10 pts                 │  │
    │   └──────────────────────────────────┘  │
    │   ┌──────────────────────────────────┐  │
    │   │ Relative Volume (20 pts)         │  │
    │   │ - RVol >= 2.0x: 20 pts           │  │
    │   │ - RVol >= 1.5x: 15 pts           │  │
    │   │ - RVol >= 1.2x: 10 pts           │  │
    │   └──────────────────────────────────┘  │
    │   ┌──────────────────────────────────┐  │
    │   │ Volatility (30 pts)              │  │
    │   │ - ATR >= 3.0%: 30 pts            │  │
    │   │ - ATR >= 2.0%: 20 pts            │  │
    │   │ - ATR >= 1.5%: 15 pts            │  │
    │   └──────────────────────────────────┘  │
    │   ┌──────────────────────────────────┐  │
    │   │ Range (10 pts)                   │  │
    │   │ - Range >= 3.0%: 10 pts          │  │
    │   │ - Range >= 2.0%: 7 pts           │  │
    │   └──────────────────────────────────┘  │
    │   ┌──────────────────────────────────┐  │
    │   │ Price Quality (10 pts)           │  │
    │   │ - $10-500: 10 pts                │  │
    │   │ - $5-1000: 5 pts                 │  │
    │   └──────────────────────────────────┘  │
    │                                         │
    │   Min score for inclusion: 40           │
    └─────────────────────────────────────────┘
                    │
                    │ Rank by score
                    ▼
    ┌─────────────────────────────────────────┐
    │   TIER ASSIGNMENT                       │
    │                                         │
    │   Score 80-100 → Tier 1 (~50 stocks)   │
    │   Score 65-79  → Tier 2 (~100 stocks)  │
    │   Score 50-64  → Tier 3 (~150 stocks)  │
    │   Score 40-49  → Tier 4 (~200 stocks)  │
    │                                         │
    │   Quality Flags:                        │
    │   - is_liquid: $Volume >= $25M          │
    │   - is_volatile: ATR >= 2.0%            │
    │   - is_trending: 20d change >= 10%      │
    └─────────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────┐
    │   PostgreSQL: stock_watchlist           │
    │                                         │
    │   rank | ticker | score | tier | ...   │
    │   ---- | ------ | ----- | ---- | ---   │
    │   1    | NVDA   | 95.3  | 1    | ...   │
    │   2    | TSLA   | 92.1  | 1    | ...   │
    │   3    | AAPL   | 88.7  | 1    | ...   │
    │   ...  | ...    | ...   | ...  | ...   │
    │                                         │
    │   Total: 500 stocks                     │
    │   Updated: Daily                        │
    └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: STRATEGY SIGNAL GENERATION (On-Demand / Real-Time)              │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────┐
    │   stock_watchlist                       │
    │   (Pre-filtered, scored stocks)         │
    └─────────────────────────────────────────┘
                    │
                    │ daily_momentum_breakout.py
                    │ (or other strategies)
                    ▼
    ┌─────────────────────────────────────────┐
    │   STRATEGY FILTERS                      │
    │                                         │
    │   From Watchlist:                       │
    │   - Tier <= 2 (high quality)            │
    │   - Score >= 65                         │
    │   - RVol >= 1.3                         │
    │   - ATR >= 2.0%                         │
    │   - Positive 5-day momentum             │
    │                                         │
    │   Result: ~50-100 candidates            │
    └─────────────────────────────────────────┘
                    │
                    │ For each candidate:
                    │ - Fetch daily candles
                    │ - Calculate indicators
                    ▼
    ┌─────────────────────────────────────────┐
    │   TECHNICAL ANALYSIS                    │
    │                                         │
    │   Calculate:                            │
    │   - EMA 20                              │
    │   - 20-day high/low                     │
    │   - Momentum indicators                 │
    │   - Trend strength                      │
    │                                         │
    │   Check Conditions:                     │
    │   ✓ Price > EMA 20 (uptrend)            │
    │   ✓ Price >= 0.99 * 20d high (breakout)│
    │   ✓ RVol >= 1.3 (high volume)           │
    │   ✓ 5d momentum > 0 (trending)          │
    └─────────────────────────────────────────┘
                    │
                    │ If conditions met:
                    ▼
    ┌─────────────────────────────────────────┐
    │   SIGNAL GENERATION                     │
    │                                         │
    │   Calculate:                            │
    │   - Entry price: Current price          │
    │   - Stop loss: Entry - (2 × ATR)        │
    │   - Take profit: Entry + (6 × ATR)      │
    │   - R:R ratio: (TP - Entry)/(Entry - SL)│
    │                                         │
    │   Confidence Score (0-1):               │
    │   - Watchlist score: 40%                │
    │   - Breakout strength: 20%              │
    │   - Relative volume: 20%                │
    │   - Trend strength: 10%                 │
    │   - Momentum: 10%                       │
    │                                         │
    │   Min confidence: 0.70                  │
    └─────────────────────────────────────────┘
                    │
                    │ Filter by confidence
                    ▼
    ┌─────────────────────────────────────────┐
    │   ACTIONABLE SIGNALS                    │
    │                                         │
    │   {                                     │
    │     ticker: "NVDA",                     │
    │     entry_price: 142.50,                │
    │     stop_loss: 137.80,                  │
    │     take_profit: 156.70,                │
    │     confidence: 0.85,                   │
    │     risk_reward: 3.0,                   │
    │     reason: "Breakout with high vol..." │
    │   }                                     │
    │                                         │
    │   Result: 5-20 high-quality signals     │
    └─────────────────────────────────────────┘
                    │
                    ▼
    ┌─────────────────────────────────────────┐
    │   OUTPUT CHANNELS                       │
    │   - Database (stock_signals)            │
    │   - Telegram notifications              │
    │   - Trading API (auto-execution)        │
    │   - Dashboard / UI                      │
    └─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ DAILY SCHEDULE (US Market - Times in UTC)                               │
└─────────────────────────────────────────────────────────────────────────┘

    21:00 UTC (4:00 PM ET)  ┌─────────────────────────────┐
                            │ US Market Closes             │
                            └─────────────────────────────┘

    23:00 UTC (6:00 PM ET)  ┌─────────────────────────────┐
                            │ Step 1: daily_update.py      │
                            │ - Fetch new 1H candles       │
                            │ - Duration: ~30 min          │
                            └─────────────────────────────┘

    23:30 UTC (6:30 PM ET)  ┌─────────────────────────────┐
                            │ Step 2: synthesize_daily.py  │
                            │ - Refresh materialized view  │
                            │ - Duration: ~2-3 min         │
                            └─────────────────────────────┘

    00:00 UTC (7:00 PM ET)  ┌─────────────────────────────┐
                            │ Step 3: calculate_watchlist  │
                            │ - Compute metrics & scores   │
                            │ - Duration: ~20-30 sec       │
                            └─────────────────────────────┘

    00:30 UTC (7:30 PM ET)  ┌─────────────────────────────┐
                            │ Step 4: Generate signals     │
                            │ - Run strategies             │
                            │ - Send notifications         │
                            └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ DATA QUALITY & MONITORING                                               │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────┐
    │ After Each Step:         │
    │ - Log execution time     │
    │ - Check row counts       │
    │ - Validate data quality  │
    │ - Alert on errors        │
    └──────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │ Quality Checks:                          │
    │ - Missing candles (incomplete days)      │
    │ - Stale data (last_updated > 24h)        │
    │ - Score distribution (tier counts)       │
    │ - Watchlist coverage (# of stocks)       │
    │ - Performance metrics (query times)      │
    └──────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │ Alerts via:                              │
    │ - Database logs                          │
    │ - Telegram                               │
    │ - Email (critical errors)                │
    │ - Monitoring dashboard                   │
    └──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE METRICS                                                      │
└─────────────────────────────────────────────────────────────────────────┘

    Database Size:
    ├─ 1H Candles:        1,367,308 rows  →  ~200 MB
    ├─ Daily Candles:        68,000 rows  →  ~10 MB
    ├─ Watchlist:                500 rows  →  ~250 KB
    └─ Total:                              →  ~211 MB

    Query Performance:
    ├─ Daily synthesis:          1 stock  →  <100ms
    ├─ Daily synthesis:      all stocks  →  <5 min
    ├─ Watchlist calc:              500  →  <30 sec
    ├─ Watchlist lookup:     single tier →  <10ms
    └─ Strategy scan:              200   →  <2 sec

    Processing Time (3,428 stocks):
    ├─ 1H data update:                    →  30 min
    ├─ Daily synthesis:                   →  3 min
    ├─ Watchlist calculation:             →  30 sec
    └─ Total daily pipeline:              →  ~35 min

┌─────────────────────────────────────────────────────────────────────────┐
│ END OF DATA FLOW DIAGRAM                                                │
└─────────────────────────────────────────────────────────────────────────┘
