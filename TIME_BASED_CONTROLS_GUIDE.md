# Time-Based Trading Controls - Weekend Protection

## Overview
This system implements comprehensive weekend protection by:
1. **Blocking new trades after 20:00 UTC** (task-worker container)
2. **Automatically closing all positions at 20:30 UTC on Fridays** (prod-app container)

## Implementation Details

### 1. Trade Cutoff (task-worker container)

**File**: `worker/app/forex_scanner/core/trading/trade_validator.py`

**Configuration** (`worker/app/forex_scanner/config.py`):
```python
ENABLE_TRADING_TIME_CONTROLS = True    # Enable/disable time-based trading controls
TRADING_CUTOFF_TIME_UTC = 20           # No new trades after this hour UTC (20:00 UTC)
```

**How it works**:
- The `check_trading_hours()` method in `TradeValidator` now checks if current time is past 20:00 UTC
- If past cutoff time, all new signals are rejected with message: `"Trading cutoff reached: XX:XX UTC >= 20:00 UTC (no new trades after cutoff)"`
- Also includes weekend protection logic (no trading Saturday, Sunday before 21:00 UTC)
- Uses UTC time for consistent global operation

### 2. Position Closure (prod-app container)

**Files**:
- `prod-app/services/position_closer.py` - Main position closure service
- `prod-app/routers/position_closer_router.py` - API endpoints
- `prod-app/main.py` - APScheduler integration

**Configuration** (`prod-app/config.py`):
```python
ENABLE_POSITION_CLOSER = True         # Enable/disable automatic position closure
POSITION_CLOSURE_HOUR_UTC = 20        # Hour to close positions (UTC)
POSITION_CLOSURE_MINUTE_UTC = 30      # Minute to close positions (UTC)
POSITION_CLOSURE_WEEKDAY = 4          # Day of week to close positions (4 = Friday, 0 = Monday)

# Safety settings
POSITION_CLOSER_TIMEOUT_SECONDS = 60  # Timeout for position closure operations
POSITION_CLOSER_MAX_RETRY_ATTEMPTS = 3  # Max retries for failed closures
POSITION_CLOSER_RETRY_DELAY_SECONDS = 5  # Delay between retry attempts
```

**How it works**:
1. **APScheduler** runs a cron job every Friday at 20:30 UTC
2. **Position Closer Service** checks if it's the right time and day
3. If conditions met, fetches all open positions via IG Markets API
4. Closes each position individually with market orders
5. Logs all actions and maintains closure history
6. Allows 5-minute execution window (20:30-20:35 UTC)

### 3. Container Architecture

```
task-worker container:
├── Prevents new signals after 20:00 UTC
├── TradeValidator rejects late signals
└── Configuration: TRADING_CUTOFF_TIME_UTC = 20

prod-app container:
├── Closes positions at 20:30 UTC on Fridays
├── Uses IG Markets API for position management
├── APScheduler for automated execution
└── Configuration: Fridays 20:30 UTC
```

## API Endpoints (prod-app container)

### Position Closer Management
- `GET /position-closer/status` - Get service status and next closure time
- `POST /position-closer/check-and-close` - Check time and close if needed
- `POST /position-closer/manual-close-all` - Emergency manual closure (bypasses time checks)
- `GET /position-closer/history` - View recent closure history
- `GET /position-closer/next-closure-time` - Get next scheduled closure
- `GET /position-closer/health` - Health check endpoint

### Example API Usage

```bash
# Check status
curl -H "x-apim-gateway: verified" http://localhost:8000/position-closer/status

# Manual emergency closure (testing only)
curl -X POST -H "x-apim-gateway: verified" http://localhost:8000/position-closer/manual-close-all

# View closure history
curl -H "x-apim-gateway: verified" http://localhost:8000/position-closer/history
```

## Timeline Example (Friday)

| Time (UTC) | Action | Container | Description |
|------------|--------|-----------|-------------|
| 19:59      | ✅ Allow | task-worker | New trades still accepted |
| 20:00      | 🚫 Block | task-worker | No new trades after cutoff |
| 20:29      | 🚫 Block | task-worker | Still blocking new trades |
| 20:30      | 🔒 Close | prod-app | Automatic position closure |
| 20:35      | 🔒 Done | prod-app | All positions closed |

## Safety Features

### TradeValidator Safety
- UTC time consistency across global deployments
- Weekend market closure detection
- Graceful error handling with fail-safe defaults
- Clear rejection messages for debugging

### Position Closer Safety
- Only executes on configured day/time
- 5-minute execution window to handle timing variations
- Individual position closure (no bulk operations)
- Comprehensive error handling and retries
- Manual override capability for emergencies
- Complete audit trail and logging

## Testing

### Test New Trade Cutoff
```python
# In task-worker container, test TradeValidator
validator = TradeValidator()
# Mock time to 20:05 UTC
is_valid, reason = validator.check_trading_hours()
# Should return: False, "Trading cutoff reached: 20:05 UTC >= 20:00 UTC (no new trades after cutoff)"
```

### Test Position Closure
```bash
# Call the manual closure endpoint for testing
curl -X POST -H "x-apim-gateway: verified" http://localhost:8000/position-closer/manual-close-all
```

## Configuration Changes Required

### Docker-compose or Deployment
Update the Dockerfile.fastapi to include APScheduler:
```dockerfile
RUN pip install --no-cache-dir \
    fastapi uvicorn requests httpx pandas \
    sqlalchemy psycopg2-binary apscheduler
```

### Environment Variables (Optional)
You can override defaults via environment variables:
```bash
export TRADING_CUTOFF_TIME_UTC=19        # Earlier cutoff
export POSITION_CLOSURE_HOUR_UTC=21      # Later closure
export ENABLE_TRADING_TIME_CONTROLS=false # Disable cutoff
export ENABLE_POSITION_CLOSER=false      # Disable closure
```

## Monitoring and Logs

### Log Messages to Watch For

**Task-worker (TradeValidator)**:
```
Trading cutoff reached: 20:05 UTC >= 20:00 UTC (no new trades after cutoff)
Weekend: No trading on Saturday
Weekend: Markets closed until Sunday 21:00 UTC (currently 15:30 UTC)
```

**Prod-app (Position Closer)**:
```
✅ Position closer scheduler started - Friday 20:30 UTC
⏰ Scheduled position closure check triggered
🔒 Starting weekend position closure: Friday 20:30 UTC - Weekend position closure time
✅ Closed position: EURUSD (BUY) Deal ID: XXX
🔒 Weekend closure complete: 3 closed, 0 failed
```

### Health Monitoring
- Monitor `/position-closer/health` endpoint
- Check scheduler logs for successful Friday executions
- Verify no new trades after 20:00 UTC in task-worker logs
- Track position closure statistics via API

## Emergency Procedures

### Disable Time Controls
```bash
# Task-worker: Edit config.py
ENABLE_TRADING_TIME_CONTROLS = False

# Prod-app: Edit config.py
ENABLE_POSITION_CLOSER = False
```

### Manual Position Closure
```bash
# Emergency close all positions immediately
curl -X POST -H "x-apim-gateway: verified" \
  http://localhost:8000/position-closer/manual-close-all
```

### Check What Positions Would Be Closed
```bash
# Check current positions without closing
curl -H "x-apim-gateway: verified" \
  http://localhost:8000/orders/positions  # If available
```

## Benefits

✅ **Weekend Protection**: No positions held over weekends
✅ **30-minute Buffer**: Sufficient time between signal cutoff and position closure
✅ **Container Separation**: Each container handles what it's designed for
✅ **Comprehensive Logging**: Full audit trail for compliance
✅ **Emergency Controls**: Manual override capabilities
✅ **Configurable**: Easy to adjust times and enable/disable
✅ **Safe Fallbacks**: Graceful error handling throughout

## Deployment Checklist

- [ ] Update Dockerfile.fastapi to include APScheduler
- [ ] Verify configuration values in both containers
- [ ] Test manual position closure endpoint
- [ ] Monitor logs for first scheduled Friday execution
- [ ] Verify trade cutoff works at 20:00 UTC
- [ ] Set up monitoring alerts for position closure failures
- [ ] Document emergency procedures for operations team

## Future Enhancements

- **Holiday Calendar**: Skip closure on market holidays
- **Position Size Limits**: Only close positions above certain size
- **Partial Closure**: Close percentage of positions instead of all
- **Multi-timezone Support**: Regional market-specific timing
- **SMS/Email Alerts**: Notifications on closure events