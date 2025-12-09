# Trailing Stop System & Partial Close Documentation

This document covers the trailing stop system implementation in the `fastapi-dev` container, including the progressive stages and partial close functionality.

## Key Files

| File | Purpose |
|------|---------|
| `dev-app/trailing_class.py` | Main trailing stop logic with progressive stages |
| `dev-app/enhanced_trade_processor.py` | Alternative trade processor (similar logic) |
| `dev-app/config.py` | Pair-specific trailing configurations |
| `dev-app/services/adjust_stop_service.py` | IG API integration for stop adjustments |
| `dev-app/services/ig_orders.py` | Partial close API calls |
| `dev-app/trade_monitor.py` | Trade monitoring loop that calls trailing logic |

## Progressive Trailing Stage System

The system uses a 4-stage progressive trailing stop:

```
Entry → Break-Even (6-8 pips) → Stage 1 (10 pips) → Stage 2 (15 pips) → Stage 3 (20+ pips)
                ↓
        Partial Close (13 pips) - Separate trigger
```

### Stage Configuration (Example: EURUSD)

```python
'CS.D.EURUSD.MINI.IP': {
    'early_breakeven_trigger_points': 6,  # Move SL to entry+1 at +6 pips
    'early_breakeven_buffer_points': 1,   # Buffer above entry
    'stage1_trigger_points': 10,          # Lock profit after +10 pts
    'stage1_lock_points': 5,              # Guarantee +5 pts profit
    'stage2_trigger_points': 15,          # Profit lock trigger
    'stage2_lock_points': 10,             # Profit guarantee
    'stage3_trigger_points': 20,          # Start percentage trailing
    'partial_close_trigger_points': 13,   # Partial close at +13 pips
    'partial_close_size': 0.5,            # Close 50% of position
    'enable_partial_close': True,
}
```

### Stage Flow

| Stage | Trigger | Action |
|-------|---------|--------|
| Break-Even | +6-8 pips | Move SL to entry + 1 pip |
| Partial Close | +13 pips | Close 50% of position |
| Stage 1 | +10 pips | Lock 5 pips profit |
| Stage 2 | +15 pips | Lock 10 pips profit |
| Stage 3 | +20 pips | Percentage-based trailing (15-25% retracement) |

## Partial Close System

### How It Works

1. **Trigger**: When profit reaches `partial_close_trigger_points` (default 13 pips)
2. **Action**: Close 50% of position via IG API
3. **Protection**: Move SL to entry + lock_points on remaining 50%
4. **Continuation**: Remaining position continues through Stage 2/3

### Key Code Location

```python
# dev-app/trailing_class.py, line ~1565
partial_close_trigger = trailing_config.get('partial_close_trigger_points', 13)
is_profitable_for_partial_close = profit_points >= partial_close_trigger

if enable_partial_close and is_profitable_for_partial_close:
    # Execute partial close...
```

### Race Condition Prevention

The partial close uses database row locking to prevent duplicate executions:

```python
# Always check database with row lock FIRST
refreshed_trade = db.query(TradeLog).filter(
    TradeLog.id == trade.id
).with_for_update(nowait=False).first()

if refreshed_trade.partial_close_executed:
    # Skip - already executed
```

### Position Verification

Before partial close, the system verifies the position exists on IG:

```python
# dev-app/services/ig_orders.py
async def partial_close_position(...):
    # Check position exists and has sufficient size
    if current_size < size_to_close:
        return {"success": False, "already_closed": True}
```

## Stop Adjustment - Absolute Levels

**CRITICAL**: Stop adjustments use **absolute price levels**, not offsets.

### Correct Implementation

```python
# dev-app/trailing_class.py - _send_stop_adjustment()
def _send_stop_adjustment(self, trade, stop_points, direction_stop, limit_points,
                          new_stop_level=None):  # ← Use this parameter
    if new_stop_level is not None:
        # CORRECT: Send absolute level
        result = adjust_stop_sync(
            epic=trade.symbol,
            new_stop=new_stop_level,  # Absolute price
        )
```

### Why Absolute Levels?

The `adjust_stop_service.py` already supports `new_stop` parameter (line 102-103):
```python
if new_stop is not None:
    payload["stopLevel"] = float(new_stop)
```

Using offsets caused incorrect calculations. Always pass `new_stop_level=<price>`.

## Configuration by Pair Volatility

| Pair Type | BE Trigger | Partial Close | Notes |
|-----------|------------|---------------|-------|
| Major (EURUSD, USDJPY) | 6 pips | 13 pips | Standard |
| GBPUSD | 8 pips | 15 pips | Higher volatility |
| GBPJPY, GBPAUD | 8 pips | 18 pips | Cross pair volatility |
| Other crosses | 7 pips | 13-15 pips | Medium volatility |

## Database Fields

Key fields in `trade_log` table for trailing:

| Field | Purpose |
|-------|---------|
| `moved_to_breakeven` | Boolean - has BE been triggered |
| `moved_to_stage2` | Boolean - has Stage 2 been triggered |
| `partial_close_executed` | Boolean - has partial close been executed |
| `partial_close_time` | Timestamp of partial close |
| `current_size` | Remaining position size after partial close |
| `sl_price` | Current stop loss price |
| `status` | Trade status (tracking, break_even, partial_closed, etc.) |

## Common Issues & Fixes

### Issue: Trailing stop not moving
**Cause**: Using offset calculations instead of absolute levels
**Fix**: Pass `new_stop_level=<price>` to `_send_stop_adjustment()`

### Issue: Partial close executing twice
**Cause**: Race condition - multiple monitoring cycles
**Fix**: Use `with_for_update(nowait=False)` row lock before checking `partial_close_executed`

### Issue: Position fully closed instead of 50%
**Cause**: Duplicate partial close calls
**Fix**:
1. Database lock before execution
2. Verify position size on IG before closing
3. Check `already_closed` in response

### Issue: SQLAlchemy session errors
**Cause**: Trade object detached from session
**Fix**: Query fresh trade with `db.query(TradeLog).filter(...).first()` instead of using passed-in object

## Testing Trailing System

```bash
# Check trailing logs
docker logs fastapi-dev --tail 100 | grep -i "trailing\|stage\|partial\|break"

# Check specific trade
docker exec postgres psql -U postgres -d forex -c \
  "SELECT id, symbol, sl_price, status, moved_to_breakeven, partial_close_executed
   FROM trade_log WHERE id = <trade_id>;"

# Check IG position
docker exec fastapi-dev python -c "
import asyncio
from dependencies import get_ig_auth_headers
import httpx
from config import API_BASE_URL

async def check():
    headers = await get_ig_auth_headers()
    async with httpx.AsyncClient() as client:
        r = await client.get(f'{API_BASE_URL}/positions', headers={
            'X-IG-API-KEY': headers['X-IG-API-KEY'],
            'CST': headers['CST'],
            'X-SECURITY-TOKEN': headers['X-SECURITY-TOKEN'],
        })
        print(r.json())

asyncio.run(check())
"
```

## Recent Updates (December 2025)

1. **Absolute Stop Levels**: Changed from offset-based to absolute price levels for reliability
2. **Partial Close Timing**: Moved from break-even (6 pips) to 13 pips for better profit capture
3. **Race Condition Fix**: Added database row locking to prevent duplicate partial closes
4. **Position Verification**: Added check that position exists before partial close
5. **Session Handling**: Fixed SQLAlchemy detached object issues
