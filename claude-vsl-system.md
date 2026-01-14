# Virtual Stop Loss (VSL) System for Scalping Mode

## Overview

The Virtual Stop Loss system enables tight stop losses for scalp trades by monitoring positions via real-time price streaming and closing them programmatically when price moves against the entry. This bypasses IG Markets' minimum stop loss restrictions (~10 pips) while maintaining a broker-level safety net.

## Problem Solved

IG Markets enforces minimum stop loss distances (~10 pips) that conflict with scalping's tight 5-pip targets. The VSL system provides:
- **Sub-second reaction time** via Lightstreamer streaming (~100ms updates)
- **Configurable virtual SL** (default 4 pips, configurable per-pair)
- **Broker safety net** - IG's actual SL at ~10 pips as crash protection

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI-DEV Container                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IG Lightstreamer ──► MARKET:{epic} subscription               │
│       │                    │                                    │
│       │                    ▼                                    │
│       │         ┌──────────────────────────┐                   │
│       │         │  VirtualStopLossService  │                   │
│       │         │  - Track scalp positions │                   │
│       │         │  - Check VSL on each tick│                   │
│       │         │  - Close if breached     │                   │
│       │         └──────────┬───────────────┘                   │
│       │                    │                                    │
│       │                    ▼                                    │
│       │         ┌──────────────────────────┐                   │
│       │         │  partial_close_position()│ (existing)        │
│       │         │  ig_orders.py            │                   │
│       │         └──────────────────────────┘                   │
│                                                                 │
│  Existing Trade Monitor (30s polling) ─► Trailing stops, BE    │
│                                                                 │
│  Safety Net: IG's actual SL at ~10 pips (broker-enforced)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Configuration (`dev-app/config_virtual_stop.py`)

```python
VIRTUAL_STOP_LOSS_ENABLED = True
DEFAULT_VIRTUAL_SL_PIPS = 4.0  # Close when 4 pips against

PAIR_VIRTUAL_SL_CONFIGS = {
    'CS.D.EURUSD.CEEM.IP': {'virtual_sl_pips': 4.0, 'enabled': True},
    'CS.D.GBPUSD.MINI.IP': {'virtual_sl_pips': 5.0, 'enabled': True},
    # ... other pairs
}

BROKER_SAFETY_SL_PIPS = 10  # IG's actual SL as backup
```

### 2. Market Price Streaming (`dev-app/services/market_price_stream.py`)

- `MarketPriceStreamManager` - Manages Lightstreamer connections
- `MarketPriceListener` - Handles MARKET:{epic} subscription updates
- Provides real-time BID/OFFER prices for VSL calculations

### 3. VSL Service (`dev-app/services/virtual_stop_loss_service.py`)

Core monitoring service with:
- `start()` / `stop()` - Service lifecycle
- `add_scalp_position(trade)` - Register position for monitoring
- `remove_position(trade_id)` - Stop monitoring
- `_on_price_update(price)` - Check VSL breach on each tick
- `_trigger_vsl_close(position, price)` - Execute close via REST API

**VSL Breach Logic:**
```python
if direction == "BUY":
    breached = price.bid <= virtual_sl_price  # BID for closing BUY
else:  # SELL
    breached = price.offer >= virtual_sl_price  # OFFER for closing SELL
```

### 4. REST API (`dev-app/routers/virtual_stop_loss_router.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vsl/status` | GET | Service status and tracked positions |
| `/api/vsl/health` | GET | Health check |
| `/api/vsl/refresh` | POST | Force position sync from database |
| `/api/vsl/position/{trade_id}` | GET | Get specific position details |
| `/api/vsl/position/{trade_id}` | POST | Add position to tracking |
| `/api/vsl/position/{trade_id}` | DELETE | Remove from tracking |

### 5. Database Schema

New columns in `trade_log` table:
```sql
is_scalp_trade BOOLEAN DEFAULT FALSE
virtual_sl_pips FLOAT
virtual_sl_price FLOAT
virtual_sl_triggered BOOLEAN DEFAULT FALSE
virtual_sl_triggered_at TIMESTAMP
```

## Configuration

### Environment Variables (docker-compose.yml)

The VSL service uses PRODUCTION IG credentials for streaming:
```yaml
fastapi-dev:
  environment:
    # Demo credentials for regular trading
    - IG_API_KEY=${DEMO_API_KEY}
    - IG_PWD=${DEMO_PASSWORD}
    # Production credentials for VSL streaming
    - PROD_IG_API_KEY=${PROD_API_KEY}
    - PROD_IG_PWD=${PROD_PASSWORD}
```

### Enabling/Disabling VSL

In `config_virtual_stop.py`:
```python
VIRTUAL_STOP_LOSS_ENABLED = True  # Master switch
```

Per-pair enable/disable:
```python
PAIR_VIRTUAL_SL_CONFIGS = {
    'CS.D.EURUSD.CEEM.IP': {'virtual_sl_pips': 4.0, 'enabled': True},
    'CS.D.GBPUSD.MINI.IP': {'virtual_sl_pips': 5.0, 'enabled': False},  # Disabled
}
```

## Usage

### Automatic Scalp Trade Detection

Trades are automatically marked as scalp trades when:
1. Stop loss is <= 8 pips, OR
2. `is_scalp_trade=True` flag is set in trade request

### Manual Position Management

```bash
# Check VSL status
curl -H "x-apim-gateway: verified" http://localhost:8001/api/vsl/status

# Force position sync
curl -X POST -H "x-apim-gateway: verified" http://localhost:8001/api/vsl/refresh

# Add position to tracking
curl -X POST -H "x-apim-gateway: verified" http://localhost:8001/api/vsl/position/123

# Remove from tracking
curl -X DELETE -H "x-apim-gateway: verified" http://localhost:8001/api/vsl/position/123
```

## How VSL Triggers Work

1. **Position Opens**: Scalp trade created with `is_scalp_trade=True`
2. **VSL Registers**: Service calculates `virtual_sl_price` and subscribes to MARKET:{epic}
3. **Price Streaming**: Lightstreamer sends BID/OFFER updates (~100ms)
4. **Breach Check**: On each tick, service checks if price breaches VSL level
5. **Close Trigger**: If breached, service closes position via `partial_close_position()`
6. **Database Update**: `virtual_sl_triggered=True` and `virtual_sl_triggered_at` recorded

## Safety Features

1. **Broker SL**: IG's actual SL at ~10 pips remains as crash protection
2. **Close Cooldown**: Prevents rapid-fire close attempts (configurable)
3. **Auto-Reconnect**: Lightstreamer reconnects automatically on disconnect
4. **Position Sync**: Periodic sync with database (default 30s)

## Monitoring

### Logs

VSL service logs with `[VSL]` and `[MARKET STREAM]` prefixes:
```
[VSL] 🚀 Starting Virtual Stop Loss Service...
[MARKET STREAM] ✅ Connected to Lightstreamer
[VSL] ✅ Virtual Stop Loss Service started
[VSL] 📊 Tracking 2 scalp positions
[VSL] ⚡ VSL triggered for trade 456 at 1.08523 (entry: 1.08563)
```

### Health Endpoint Response

```json
{
    "healthy": true,
    "running": true,
    "stream_connected": true,
    "positions_tracked": 2,
    "vsl_triggers": 1,
    "price_updates": 15432
}
```

## Files Summary

| File | Purpose |
|------|---------|
| `dev-app/config_virtual_stop.py` | VSL configuration |
| `dev-app/services/market_price_stream.py` | Lightstreamer streaming |
| `dev-app/services/virtual_stop_loss_service.py` | Core VSL logic |
| `dev-app/routers/virtual_stop_loss_router.py` | REST API endpoints |
| `dev-app/migrations/add_virtual_stop_loss_columns.sql` | Database schema |
| `dev-app/dependencies.py` | Production auth for streaming |

## Troubleshooting

### VSL Not Starting

1. Check PRODUCTION credentials in docker-compose.yml
2. Verify Lightstreamer URL (production vs demo)
3. Check container logs: `docker logs fastapi-dev | grep -i vsl`

### Connection Errors

"User/password check failed" - Usually means:
- Using demo credentials with production Lightstreamer URL
- Expired tokens (service auto-refreshes every 55 minutes)

### Positions Not Being Tracked

1. Verify `is_scalp_trade=True` in trade request
2. Check VSL is enabled for the pair in config
3. Force refresh: `POST /api/vsl/refresh`
