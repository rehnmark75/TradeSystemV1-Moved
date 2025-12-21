# Trade Outcome Analysis Skill

This skill provides comprehensive knowledge for analyzing trade outcomes, understanding why trades won or lost, and improving trading strategy performance.

---

## System Overview

### Key Database Tables

#### 1. `trade_log` - Core Trade Execution Table
**Location**: Defined in `dev-app/services/models.py:20-104`

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `symbol` | String | Trading pair (e.g., CS.D.EURUSD.MINI.IP) |
| `direction` | String | "BUY" or "SELL" |
| `entry_price` | Float | Order entry level (stop-entry price for momentum confirmation) |
| `sl_price` | Float | Stop loss absolute price |
| `tp_price` | Float | Take profit absolute price |
| `limit_price` | Float | Alternative TP storage |
| `timestamp` | DateTime | Trade creation time |
| `closed_at` | DateTime | Trade close time |
| `status` | String | "pending", "tracking", "closed", "partial_closed" |
| `deal_id` | String | IG broker deal ID |
| `deal_reference` | String | IG broker reference |
| `alert_id` | Integer | FK to alert_history.id (links signal to trade) |
| `profit_loss` | Numeric | Final P&L from broker |
| `calculated_pnl` | Numeric | Price-based P&L calculation |
| `pips_gained` | Numeric | Pips profit/loss |
| `moved_to_breakeven` | Boolean | Whether trailing stop moved to BE |
| `partial_close_executed` | Boolean | Whether 50% partial close occurred |
| `partial_close_time` | DateTime | When partial close happened |
| `current_size` | Float | Position size (1.0 → 0.5 after partial) |

#### 2. `alert_history` - Signal/Alert Storage
**Location**: `dev-app/services/models.py:107-192`

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key (referenced by trade_log.alert_id) |
| `epic` | String | IG instrument epic |
| `pair` | String | Currency pair name |
| `signal_type` | String | "BUY" or "SELL" |
| `strategy` | String | Strategy name (EMA, MACD, SMC_STRUCTURE, etc.) |
| `confidence_score` | Numeric | Signal confidence (0.0-1.0) |
| `price` | Numeric | Price at signal time |
| `timeframe` | String | "15m", "1h", "4h" |
| `strategy_indicators` | JSON | All technical indicators at signal time |
| `strategy_metadata` | JSON | Strategy-specific metadata |
| `market_structure_analysis` | JSON | SMC market structure data |
| `order_flow_analysis` | JSON | Order block, FVG, liquidity data |
| `confluence_details` | JSON | Confluence scoring breakdown |
| `claude_score` | Integer | Claude AI scoring (1-10) |
| `claude_approved` | Boolean | Claude approval status |
| `smart_money_validated` | Boolean | SMC validation flag |

#### 3. `broker_transactions` - IG Broker Transaction Records
**Location**: `dev-app/services/models.py:211-236`

Stores raw broker transaction data for P&L reconciliation.

---

## Trade Execution Flow

### Signal → Trade → Outcome Pipeline

```
1. Signal Detection (worker/app/forex_scanner)
   └── Strategy generates signal
         └── Saved to alert_history

2. Trade Execution (dev-app/fastapi-dev container)
   └── OrderExecutor creates trade
         └── Creates trade_log entry with alert_id reference
         └── Places order via IG API
         └── Sets deal_id from IG response

3. Trade Monitoring (dev-app/trailing_class.py)
   └── TrailingStopManager monitors position
         └── Updates sl_price as trailing progresses
         └── Sets moved_to_breakeven when triggered
         └── Handles partial close at profit target

4. P&L Calculation (dev-app/services/)
   └── activity_pnl_correlator.py - Correlates with IG activities
   └── price_based_pnl_calculator.py - Calculates P&L from prices
         └── Updates calculated_pnl, pips_gained, etc.
```

---

## Analysis Endpoints (FastAPI)

### Primary Analysis Endpoints

**Container**: `fastapi-dev`
**Base URL**: `http://localhost:8000`

#### 1. Trade Analysis Router
**File**: `dev-app/routers/trade_analysis_router.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trade-analysis/trade/{trade_id}` | GET | Full trailing stop analysis |
| `/api/trade-analysis/trade/{trade_id}/timeline` | GET | Chronological event timeline |
| `/api/trade-analysis/signal/{trade_id}` | GET | Signal/alert analysis with SMC data |
| `/api/trade-analysis/outcome/{trade_id}` | GET | **MFE/MAE outcome analysis** |

#### 2. Trading Analytics Router
**File**: `dev-app/routers/trading_analytics_router.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trading/deals/calculate-complete-pnl` | POST | Run P&L calculation pipeline |
| `/api/trading/deals/pnl-calculation-status` | GET | P&L calculation coverage |
| `/api/trading/statistics` | GET | Aggregate trading statistics |
| `/api/trading/performance/summary` | GET | Dashboard performance summary |

---

## Outcome Analysis Components

### MFE/MAE Analysis
**Service**: `dev-app/services/trade_analysis_service.py`

```python
# Maximum Favorable Excursion (MFE)
# = Maximum profit the trade reached before exit
# Tells you: "How much profit was available?"

# Maximum Adverse Excursion (MAE)
# = Maximum drawdown the trade experienced
# Tells you: "How much pain did you endure?"

# MFE/MAE Ratio interpretation:
# - Ratio > 2: Good entry, price moved favorably
# - Ratio < 1: Bad entry, price moved against immediately
# - Ratio = 0: Immediate reversal after entry
```

### Entry Quality Assessment
Factors evaluated:
- HTF alignment (4H trend direction)
- Confluence count (SMC factors present)
- Confidence score (strategy confidence)
- Zone quality (premium/discount positioning)

### Exit Quality Assessment
Factors evaluated:
- Exit type (TP hit, SL hit, trailing stop)
- MFE captured percentage
- Missed profit (MFE - actual exit)
- Trailing stages activated

---

## Key Queries for Analysis

### Win/Loss Statistics
```sql
-- Basic win/loss stats for period
SELECT
    COUNT(*) as total_trades,
    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as winners,
    COUNT(CASE WHEN profit_loss < 0 THEN 1 END) as losers,
    ROUND(AVG(CASE WHEN profit_loss > 0 THEN profit_loss END), 2) as avg_win,
    ROUND(AVG(CASE WHEN profit_loss < 0 THEN profit_loss END), 2) as avg_loss,
    ROUND(SUM(profit_loss), 2) as total_pnl
FROM trade_log
WHERE timestamp >= NOW() - INTERVAL '7 days'
AND status = 'closed';
```

### Strategy Performance
```sql
-- Performance by strategy
SELECT
    a.strategy,
    COUNT(t.*) as trades,
    ROUND(100.0 * COUNT(CASE WHEN t.profit_loss > 0 THEN 1 END) / COUNT(*), 1) as win_rate,
    ROUND(SUM(t.profit_loss), 2) as total_pnl,
    ROUND(AVG(t.pips_gained), 1) as avg_pips
FROM trade_log t
JOIN alert_history a ON t.alert_id = a.id
WHERE t.timestamp >= NOW() - INTERVAL '30 days'
AND t.status = 'closed'
GROUP BY a.strategy
ORDER BY total_pnl DESC;
```

### Trailing Stop Effectiveness
```sql
-- Trades that reached breakeven
SELECT
    symbol,
    COUNT(*) as total,
    COUNT(CASE WHEN moved_to_breakeven THEN 1 END) as reached_be,
    ROUND(100.0 * COUNT(CASE WHEN moved_to_breakeven THEN 1 END) / COUNT(*), 1) as be_rate,
    ROUND(AVG(CASE WHEN moved_to_breakeven THEN profit_loss END), 2) as avg_pnl_when_be
FROM trade_log
WHERE status = 'closed'
AND timestamp >= NOW() - INTERVAL '30 days'
GROUP BY symbol;
```

### Signal-to-Trade Correlation
```sql
-- Join trade with its originating signal
SELECT
    t.id as trade_id,
    t.symbol,
    t.direction,
    t.entry_price,
    t.profit_loss,
    t.pips_gained,
    a.strategy,
    a.confidence_score,
    a.signal_trigger,
    a.smart_money_validated,
    a.claude_approved
FROM trade_log t
JOIN alert_history a ON t.alert_id = a.id
WHERE t.status = 'closed'
ORDER BY t.timestamp DESC
LIMIT 20;
```

---

## Streamlit Analytics Dashboard

**File**: `streamlit/pages/unified_analytics.py`

### Available Tabs
1. **Overview** - Key metrics, P&L summary, recent trades
2. **Strategy Analysis** - Per-strategy performance breakdown
3. **Trade Details** - Individual trade inspection
4. **Pair Performance** - Per-currency-pair analysis
5. **Time Analysis** - Performance by session/time
6. **Unfilled Orders** - Limit orders that weren't triggered

### Key Classes
- `UnifiedTradingDashboard` - Main dashboard controller
- `TradingStatistics` - Dataclass for aggregated metrics

---

## Common Analysis Workflows

### 1. Analyze Why a Trade Lost
```bash
# Get full trade analysis
curl http://localhost:8000/api/trade-analysis/outcome/{trade_id}

# Returns:
# - MFE/MAE data (was profit ever available?)
# - Entry quality (was the signal good?)
# - Exit quality (did we exit well?)
# - Learning insights (what went wrong?)
```

### 2. Evaluate Strategy Performance
```bash
docker exec -it postgres psql -U postgres -d trading -c "
SELECT a.strategy, COUNT(*) trades,
       ROUND(100.0 * SUM(CASE WHEN t.profit_loss > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
FROM trade_log t
JOIN alert_history a ON t.alert_id = a.id
WHERE t.status = 'closed'
GROUP BY a.strategy;
"
```

### 3. Check P&L Calculation Coverage
```bash
curl http://localhost:8000/api/trading/deals/pnl-calculation-status?days_back=7
```

### 4. Run P&L Calculation Pipeline
```bash
curl -X POST http://localhost:8000/api/trading/deals/calculate-complete-pnl \
  -H "Content-Type: application/json" \
  -d '{"days_back": 7, "update_trade_log": true}'
```

---

## Trailing Stop Stages

**Config File**: `worker/app/forex_scanner/config_trailing_stops.py`

### Stage Progression
1. **Break-even** (6pts profit): Move SL to entry price
2. **Stage 1** (12-16pts): Lock in 2-8 pips profit
3. **Stage 2** (16-22pts): Lock in 10-12 pips profit
4. **Stage 3** (23+pts): ATR-based trailing (0.8x ATR)

### Partial Close
- Triggers at 13 pips profit (configurable per pair)
- Closes 50% of position
- Remaining 50% continues with trailing

---

## Key Files Reference

| Purpose | File Path |
|---------|-----------|
| Trade log model | `dev-app/services/models.py` |
| Outcome analysis endpoint | `dev-app/routers/trade_analysis_router.py` |
| MFE/MAE calculation | `dev-app/services/trade_analysis_service.py` |
| P&L correlation | `dev-app/services/activity_pnl_correlator.py` |
| Price-based P&L | `dev-app/services/price_based_pnl_calculator.py` |
| Analytics dashboard | `streamlit/pages/unified_analytics.py` |
| Trailing stop config | `worker/app/forex_scanner/config_trailing_stops.py` |
| Order execution | `dev-app/services/trade_automation_service.py` |

---

## Docker Commands

```bash
# Query trade_log directly
docker exec -it postgres psql -U postgres -d trading -c "SELECT * FROM trade_log LIMIT 5;"

# Check recent trades
docker exec -it postgres psql -U postgres -d trading -c "
SELECT id, symbol, direction, profit_loss, status, timestamp
FROM trade_log ORDER BY timestamp DESC LIMIT 10;
"

# View trade with signal details
docker exec -it postgres psql -U postgres -d trading -c "
SELECT t.id, t.symbol, t.profit_loss, a.strategy, a.confidence_score
FROM trade_log t
LEFT JOIN alert_history a ON t.alert_id = a.id
WHERE t.status = 'closed'
ORDER BY t.timestamp DESC LIMIT 10;
"

# Call outcome analysis endpoint
docker exec -it fastapi-dev curl http://localhost:8000/api/trade-analysis/outcome/123

# Run P&L pipeline
docker exec -it fastapi-dev curl -X POST \
  http://localhost:8000/api/trading/deals/calculate-complete-pnl \
  -H "Content-Type: application/json" \
  -d '{"days_back": 7}'
```
