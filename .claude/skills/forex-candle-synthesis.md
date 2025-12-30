# Forex Candle Synthesis System

This skill provides comprehensive knowledge about how candles are synthesized in the forex_scanner system.

---

## Overview

The forex_scanner uses a **multi-layered candle synthesis architecture**:

1. **Ingestion**: Real-time streaming from IG Markets API → PostgreSQL
2. **Storage**: Normalized 1m base candles with quality scoring
3. **Derivation**: On-demand resampling to 5m, 15m, 1h, 4h with completeness tracking
4. **Caching**: Multi-tier caching (memory, data, indicator, resampled)
5. **Enhancement**: Dynamic technical indicators based on enabled strategies

---

## 1. Data Source: IG Markets API

### Streaming Configuration
**File**: `streamlit/igstream/ig_stream_to_postgres.py`

- **Provider**: IG Markets (trading_ig Python library)
- **Protocol**: Lightstreamer (websocket-based real-time streaming)
- **Subscriptions**: Chart data for multiple currency pairs
- **Resolutions Streamed**: MINUTE_5, MINUTE_15

### Fields Retrieved from IG Markets
```python
BID_OPEN, BID_HIGH, BID_LOW, BID_CLOSE  # OHLC prices
LTP_CLOSE                                 # Last Traded Price
UTM                                       # Unix timestamp (milliseconds)
CONS_TICK_COUNT                          # Volume/tick count
```

### Ingestion Flow
```
CHART:EPIC:RESOLUTION → Parse Fields → INSERT INTO ig_candles
```

---

## 2. Database Storage

### Core Table: `ig_candles`
```sql
CREATE TABLE ig_candles (
    start_time TIMESTAMP,
    epic VARCHAR,          -- e.g., "CS.D.EURUSD.CEEM.IP"
    timeframe INTEGER,     -- Minutes: 5, 15, 60, 240, 1440
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume INTEGER,
    ltv INTEGER,           -- Last traded volume
    quality_score FLOAT,   -- Data quality indicator
    data_source VARCHAR    -- Source identifier
);
```

### Preferred Prices System
**Table**: `preferred_forex_prices`
- Joins with `ig_candles` for enhanced price selection
- Tracks data quality scores
- Provides safety validation: `is_price_safe_for_trading()`
- Handles BID/MID/ASK price preferences

---

## 3. Candle Data Fetching

### Main Data Fetcher
**File**: `worker/app/forex_scanner/core/data_fetcher.py`
**Class**: `DataFetcher`

### Primary Fetch Method
```python
def _fetch_candle_data_optimized(epic, timeframe, lookback_hours, tz_manager)
```

- Queries PostgreSQL with optimized SQL
- Supports timeframes: 5m, 15m, 1h, 4h
- Uses `preferred_forex_prices` for quality-controlled close prices
- Returns pandas DataFrame with OHLCV data

### Typical Query Pattern
```sql
SELECT
    pfp.start_time,
    ic.open, ic.high, ic.low,
    pfp.preferred_price as close,
    ic.ltv,
    pfp.quality_score,
    is_price_safe_for_trading(pfp.epic, pfp.timeframe, pfp.start_time, 10.0) as is_safe_for_trading
FROM preferred_forex_prices pfp
JOIN ig_candles ic ON (
    pfp.epic = ic.epic
    AND pfp.timeframe = ic.timeframe
    AND pfp.start_time = ic.start_time
)
WHERE epic = :epic
  AND timeframe = :timeframe
  AND start_time >= :since
ORDER BY start_time ASC
```

---

## 4. Timeframe Handling & Resampling

### Base Timeframe
The system stores **1-minute candles** as the base and derives all higher timeframes through resampling.

### Timeframe Mapping
```python
timeframe_map = {
    '1m': 1,      # 1 minute (BASE)
    '5m': 5,      # 5 minutes (5 x 1m)
    '15m': 15,    # 15 minutes (15 x 1m)
    '30m': 30,    # 30 minutes (30 x 1m)
    '1h': 60,     # 1 hour (60 x 1m)
    '4h': 240,    # 4 hours (240 x 1m)
    '1d': 1440    # Daily (1440 x 1m)
}
```

### Resampling Methods

All higher timeframes are derived from 1-minute base candles through resampling.

#### 5-Minute Resampling
**Method**: `_resample_to_5m_optimized(df)`
- Source: 5 x 1m candles per 5m period

#### 15-Minute Resampling
**Method**: `_resample_to_15m_optimized(df)`
- Source: 15 x 1m candles per 15m period

```python
df.resample('15min', label='left', closed='left', origin='epoch').agg({
    'open': 'first',    # First open in the period
    'high': 'max',      # Highest high
    'low': 'min',       # Lowest low
    'close': 'last',    # Last close
    'ltv': 'sum'        # Sum of volume
})
```

**Quality Tracking**:
- `actual_1m_candles`: Number of 1m candles found
- `expected_1m_candles`: 15 (for 15m), 60 (for 1h), 240 (for 4h)
- `completeness_ratio`: actual/expected
- `is_complete`: Boolean flag
- `trading_confidence`: 0-100 score
- `suitable_for_entry`: Confidence >= 90%
- `suitable_for_analysis`: Confidence >= 80%

#### 60-Minute (1H) Resampling
**Method**: `_resample_to_60m_optimized(df)`
- Source: 60 x 1m candles per 1H period
- Same aggregation logic as 15m

#### 4-Hour Resampling
**Method**: `_resample_to_4h_optimized(df)`
- Source: 240 x 1m candles per 4H period

### Resampling Principles
```python
label='left'      # Timestamps represent period start
closed='left'     # Left-inclusive intervals
origin='epoch'    # Consistent epoch-based alignment
```

---

## 5. Caching Mechanisms

### A. In-Memory Cache
**File**: `worker/app/forex_scanner/core/memory_cache.py`
**Class**: `InMemoryForexCache`

```python
cache: Dict[epic, Dict[timeframe, pd.DataFrame]] = {
    'CS.D.EURUSD.CEEM.IP': {
        5: DataFrame(...),    # 5m candles
        15: DataFrame(...),   # 15m candles
        60: DataFrame(...)    # 1h candles
    }
}
```

**Features**:
- Loads entire `ig_candles` table into RAM
- Memory limit: 2GB (configurable)
- Float32/int32 compression
- Sub-millisecond lookups

### B. DataFetcher Cache
**Two-Layer Caching**:
1. **Data Cache** (`_data_cache`): 5-minute timeout, caches raw data
2. **Indicator Cache** (`_indicator_cache`): Pre-calculated indicators

### C. Backtest Resampled Cache
**Class**: `BacktestDataFetcher`
```python
_resampled_cache = {}  # Pre-resampled higher timeframes
_backtest_cache = {}   # Historical data cache
_validation_cache = {} # Data quality validation
```

---

## 6. Data Structures

### OHLCV DataFrame Format
```python
DataFrame columns:
    start_time        # Pandas Timestamp (UTC)
    open              # float64
    high              # float64
    low               # float64
    close             # float64
    volume            # int64
    ltv               # int64 (Last traded volume)

# Enhancement columns:
    local_time        # User timezone
    market_session    # London/NY/Asia/Pacific
    user_time         # Display timezone

# Technical indicators (added dynamically):
    ema_9, ema_21, ema_50, ema_200
    macd_line, macd_signal, macd_histogram
    rsi, atr, atr_percentile
    bb_upper, bb_middle, bb_lower
    supertrend, supertrend_direction
```

---

## 7. Multi-Timeframe Analysis

**File**: `worker/app/forex_scanner/analysis/multi_timeframe.py`
**Class**: `MultiTimeframeAnalyzer`

### Workflow
```python
# 1. Fetch multiple timeframes
df_5m = data_fetcher.get_enhanced_data(epic, pair, '5m', lookback_hours=48)
df_15m = data_fetcher.get_enhanced_data(epic, pair, '15m', lookback_hours=168)
df_1h = data_fetcher.get_enhanced_data(epic, pair, '1h', lookback_hours=720)

# 2. Analyze trends across timeframes
# 3. Calculate confluence scores
# 4. Generate enhanced signals with agreement metrics
```

---

## 8. Data Enhancement Pipeline

**Method**: `_enhance_with_analysis_optimized(df, pair, ema_strategy, ema_periods)`

### Enhancement Stages
1. **Base EMAs**: EMA 50, 100, 200 (always)
2. **Strategy-Specific Indicators**: Based on enabled strategies
3. **Support/Resistance**: Pivot points and S/R levels
4. **Volume Analysis**: Spikes, relative volume
5. **Timezone Conversion**: UTC → User timezone

---

## 9. Data Quality & Validation

### Quality Metrics
```python
# Completeness Score
completeness_score = actual_candles / expected_candles

# Quality Score Calculation
quality_score = (
    completeness_score * 0.6
    - price_anomaly_penalty * 0.05
    - volume_anomaly_penalty * 0.02
)
```

### Validation Checks
- High >= Low, High >= Open/Close, Low <= Open/Close
- Gap detection (>3% change)
- Narrow range detection
- Zero/extreme volume detection
- `is_price_safe_for_trading()` SQL function

---

## 10. Key Files Reference

| File | Purpose |
|------|---------|
| `streamlit/igstream/ig_stream_to_postgres.py` | Real-time IG Markets data ingestion |
| `worker/app/forex_scanner/core/data_fetcher.py` | Main candle fetching & enhancement (~2000 lines) |
| `worker/app/forex_scanner/core/backtest_data_fetcher.py` | Backtest-optimized data fetcher |
| `worker/app/forex_scanner/core/memory_cache.py` | In-memory cache for ultra-fast access |
| `worker/app/forex_scanner/analysis/multi_timeframe.py` | Multi-timeframe analysis & confluence |
| `worker/app/forex_scanner/core/database.py` | PostgreSQL database management |

---

## 11. Common Operations

### Fetching Candles for a Pair
```python
from worker.app.forex_scanner.core.data_fetcher import DataFetcher

fetcher = DataFetcher()
df = fetcher.get_enhanced_data(
    epic='CS.D.EURUSD.CEEM.IP',
    pair='EURUSD',
    timeframe='15m',
    lookback_hours=168
)
```

### Manual Resampling
```python
# Resample 1m to 15m
df_15m = df_1m.resample('15min', label='left', closed='left', origin='epoch').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'ltv': 'sum'
})
```

### Checking Data Quality
```python
# Check completeness
if df['trading_confidence'].iloc[-1] >= 90:
    # Safe for entry signals
elif df['trading_confidence'].iloc[-1] >= 80:
    # OK for analysis only
```

---

## 12. Debugging Tips

### Check Raw Data in Database
```sql
SELECT * FROM ig_candles
WHERE epic = 'CS.D.EURUSD.CEEM.IP'
  AND timeframe = 1
ORDER BY start_time DESC
LIMIT 100;
```

### Verify Resampling Quality
```python
# After resampling, check completeness
print(f"Complete candles: {df['is_complete'].sum()}/{len(df)}")
print(f"Avg confidence: {df['trading_confidence'].mean():.1f}%")
```

### Common Issues
1. **Missing data**: Check IG stream connection, gaps in `ig_candles`
2. **Wrong timestamps**: Ensure UTC handling throughout
3. **Stale cache**: Clear `_data_cache` or `_resampled_cache`
4. **Quality warnings**: Check `completeness_ratio` and `quality_score`
