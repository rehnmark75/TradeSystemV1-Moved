# Stream-Based Entry Timing System - Phase 1 (Monitoring)

## Overview

**Goal**: Capture real-time tick data at scalp signal time to analyze whether we could achieve better entries by waiting for optimal tick conditions instead of executing immediately.

**Scope**: Scalp mode trades only
**Container**: fastapi-dev (reuse existing MarketPriceStreamManager)
**Mode**: Monitoring only (no execution change) - collect data first

## Architecture

### Current Flow
```
task-worker: Signal detected → order_executor → HTTP POST → fastapi-dev → Order placed
```

### New Flow (Phase 1)
```
task-worker: Scalp signal detected
  ├─→ HTTP POST to fastapi-dev /api/entry-timing/start-monitoring (async, non-blocking)
  └─→ Execute order immediately (unchanged)
      ↓
fastapi-dev:
  ├─→ Subscribe to epic via MarketPriceStreamManager
  ├─→ Collect ticks for 30 seconds
  ├─→ Calculate: best price, worst price, optimal entry time
  └─→ Store to entry_timing_logs table
```

## Implementation

### New Files

#### 1. Database Migration
**File**: `worker/app/forex_scanner/migrations/create_entry_timing_logs.sql`

```sql
CREATE TABLE IF NOT EXISTS entry_timing_logs (
    id SERIAL PRIMARY KEY,
    monitoring_id UUID NOT NULL UNIQUE,
    alert_id INTEGER,

    -- Signal metadata
    epic VARCHAR(50) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    signal_price DECIMAL(12, 6) NOT NULL,
    signal_timestamp TIMESTAMP NOT NULL,
    signal_confidence DECIMAL(5, 4),

    -- Monitoring
    monitoring_duration_seconds INTEGER DEFAULT 30,
    monitoring_started_at TIMESTAMP NOT NULL,
    monitoring_completed_at TIMESTAMP,

    -- Tick data
    ticks JSONB,
    tick_count INTEGER DEFAULT 0,

    -- Analysis
    best_bid DECIMAL(12, 6),
    best_bid_time TIMESTAMP,
    worst_bid DECIMAL(12, 6),
    best_offer DECIMAL(12, 6),
    worst_offer DECIMAL(12, 6),
    avg_spread_pips DECIMAL(6, 2),
    optimal_entry_price DECIMAL(12, 6),
    optimal_entry_time TIMESTAMP,
    time_to_optimal_ms INTEGER,
    potential_improvement_pips DECIMAL(6, 2),

    -- Status
    status VARCHAR(20) DEFAULT 'monitoring',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_etl_alert_id ON entry_timing_logs(alert_id);
CREATE INDEX idx_etl_epic ON entry_timing_logs(epic);
CREATE INDEX idx_etl_signal_timestamp ON entry_timing_logs(signal_timestamp);
```

#### 2. Entry Timing Service
**File**: `dev-app/services/entry_timing_service.py`

Key components:
- `EntryTimingService` class - manages monitoring sessions
- `TimingSession` dataclass - stores session state and ticks
- Uses existing `MarketPriceStreamManager` for tick subscriptions
- Thread-safe with locks for concurrent signals
- Auto-completes after duration, calculates analysis metrics

#### 3. Entry Timing Router
**File**: `dev-app/routers/entry_timing_router.py`

Endpoints:
- `POST /api/entry-timing/start-monitoring` - Start monitoring for a signal
- `GET /api/entry-timing/status/{monitoring_id}` - Get session status
- `GET /api/entry-timing/analysis/{alert_id}` - Get analysis for an alert

### Files to Modify

#### 1. `dev-app/main.py`
- Import and register `entry_timing_router`

#### 2. `dev-app/config.py`
Add:
```python
ENTRY_TIMING_ENABLED = True
ENTRY_TIMING_MONITORING_DURATION_SECONDS = 30
ENTRY_TIMING_MAX_CONCURRENT_SESSIONS = 10
```

#### 3. `worker/app/forex_scanner/alerts/order_executor.py`
Add non-blocking call before order execution for scalp signals:
```python
# Around line 563, after scalp mode detection
if is_scalp_trade and config.ENTRY_TIMING_ENABLED:
    asyncio.create_task(self._start_entry_timing_monitoring(signal, alert_id))
```

## API Contract

### Start Monitoring
```http
POST /api/entry-timing/start-monitoring
{
    "epic": "CS.D.EURUSD.CEEM.IP",
    "signal_type": "BUY",
    "signal_price": 1.08234,
    "signal_timestamp": "2026-01-15T10:30:00Z",
    "alert_id": 12345,
    "monitoring_duration_seconds": 30
}
```

### Response
```json
{
    "status": "monitoring_started",
    "monitoring_id": "uuid-here",
    "expected_completion": "2026-01-15T10:30:30Z"
}
```

## Analysis Metrics Captured

| Metric | Description |
|--------|-------------|
| `tick_count` | Total ticks received during window |
| `best_bid` / `best_offer` | Best prices during window |
| `worst_bid` / `worst_offer` | Worst prices during window |
| `avg_spread_pips` | Average spread during window |
| `optimal_entry_price` | Best price we could have gotten |
| `time_to_optimal_ms` | Time from signal to optimal price |
| `potential_improvement_pips` | How much better entry could have been |

## Post-Collection Analysis Query

```sql
-- Average potential improvement by pair
SELECT
    epic,
    COUNT(*) as signals,
    AVG(potential_improvement_pips) as avg_improvement,
    AVG(time_to_optimal_ms / 1000.0) as avg_seconds_to_optimal
FROM entry_timing_logs
WHERE status = 'completed'
GROUP BY epic
ORDER BY avg_improvement DESC;
```

## Critical Files Reference

| File | Purpose |
|------|---------|
| [market_price_stream.py](dev-app/services/market_price_stream.py) | Existing stream manager to reuse |
| [order_executor.py](worker/app/forex_scanner/alerts/order_executor.py) | Integration point (~line 563) |
| [virtual_stop_loss_service.py](dev-app/services/virtual_stop_loss_service.py) | Pattern for stream subscriptions |
| [main.py](dev-app/main.py) | Router registration |

## Verification Plan

1. Run migration to create `entry_timing_logs` table
2. Restart fastapi-dev container
3. Trigger a scalp signal (manually or via backtest)
4. Verify monitoring session starts (check logs)
5. Query `entry_timing_logs` table after 30 seconds
6. Confirm tick data and analysis metrics are populated
7. Collect 1-2 weeks of data before Phase 2 decisions

## Future Phases (Not In Scope)

- **Phase 2**: Implement actual stream-qualified entry (wait for optimal conditions)
- **Phase 3**: A/B testing stream-qualified vs immediate execution
- **Phase 4**: Production tuning based on results
