# Order Maintenance Scripts

This directory contains maintenance scripts for the trading system.

## Cleanup Stale Orders

**Problem**: Orders stuck in "pending" status block new signals due to cooldown logic.

**Solution**: Automated cleanup script that expires pending orders older than 30 minutes.

### Files

- `cleanup_stale_orders.py` - Main cleanup script
- `/home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh` - Host wrapper script
- Cron job: Runs every 15 minutes

### Usage

#### Manual Cleanup

```bash
# Run inside Docker container
docker exec task-worker python /app/forex_scanner/maintenance/cleanup_stale_orders.py

# Run from host
/home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh

# Dry run (preview without changes)
docker exec task-worker python /app/forex_scanner/maintenance/cleanup_stale_orders.py --dry-run

# Custom age threshold (default: 30 minutes)
docker exec task-worker python /app/forex_scanner/maintenance/cleanup_stale_orders.py --max-age=60
```

#### Automated Cleanup

The script runs automatically via cron:

```bash
# View cron configuration
crontab -l | grep cleanup

# Check logs
tail -f /var/log/cleanup_stale_orders.log

# Test the cron job manually
/home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh
```

### Monitoring

```sql
-- Check for stale pending orders
SELECT
    epic,
    COUNT(*) as stale_count,
    MAX(alert_timestamp) as most_recent
FROM alert_history
WHERE order_status = 'pending'
  AND alert_timestamp < NOW() - INTERVAL '30 minutes'
GROUP BY epic;

-- Check cleanup effectiveness (should be 0 stale orders)
SELECT COUNT(*) as stale_orders
FROM alert_history
WHERE order_status = 'pending'
  AND alert_timestamp < NOW() - INTERVAL '30 minutes';

-- View recent cleanup activity (check order_status transitions)
SELECT
    alert_timestamp,
    epic,
    order_status,
    signal_type
FROM alert_history
WHERE order_status = 'expired'
  AND alert_timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY alert_timestamp DESC
LIMIT 20;
```

### How It Works

1. **Status-Based Cooldown**: The strategy uses order status to determine cooldown periods
   - `pending`: 30 minutes (just generated, brief pause)
   - `placed`: 30 minutes (order working, wait for outcome)
   - `expired`: 30 minutes (didn't fill, prevents spam)
   - `filled`: 4 hours (full cooldown - real trade opened)
   - `rejected`: 15 minutes (order failed, brief pause before retry)

2. **Normal Flow**: fastapi-dev order monitoring should update statuses automatically
   - Pending → Placed (order accepted by broker)
   - Placed → Filled (order executed)
   - Placed → Expired (order timed out)
   - Placed → Rejected (order rejected by broker)

3. **Safety Net**: This cleanup script ensures orders don't stay "pending" forever if:
   - fastapi-dev is down/unhealthy
   - Order monitoring service fails
   - Database update fails
   - Network issues prevent status updates

### Troubleshooting

#### Too many stale orders

If you see many stale orders accumulating:

1. Check fastapi-dev health:
   ```bash
   docker ps | grep fastapi-dev
   docker inspect fastapi-dev --format='{{.State.Health.Status}}'
   ```

2. Check order monitoring logs:
   ```bash
   docker logs fastapi-dev --tail 100 | grep -E "LIMIT SYNC|order.*status"
   ```

3. Restart fastapi-dev if unhealthy:
   ```bash
   docker restart fastapi-dev
   ```

#### Cleanup not running

Check cron is working:

```bash
# View cron status
systemctl status cron

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Manually test the script
/home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh
```

### Integration with Strategy

The strategy checks cooldown before generating signals:

```python
# From smc_simple_strategy.py
def _check_signal_cooldown(self, epic: str, pair: str, check_time: datetime):
    last_signal, order_status = self._get_last_alert_time_from_db(epic)

    if order_status == 'pending':
        required_cooldown = 0.5  # 30 minutes
        if hours_since < required_cooldown:
            return False, "In status-based cooldown"
```

This prevents signal spam while orders are pending, but if orders stay pending too long, it blocks legitimate new signals. The cleanup script ensures orders don't block signals indefinitely.

### Performance Impact

- Cleanup query is indexed and fast (<50ms)
- Runs every 15 minutes with minimal overhead
- Only updates orders older than 30 minutes
- Average load: 47 orders per cleanup (decreases over time)

### Configuration

Adjust cleanup frequency or age threshold:

```bash
# Edit crontab
crontab -e

# Change from every 15 minutes to every 10 minutes:
*/10 * * * * /home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh >> /var/log/cleanup_stale_orders.log 2>&1

# Or change age threshold in the script call:
*/15 * * * * docker exec task-worker python /app/forex_scanner/maintenance/cleanup_stale_orders.py --max-age=45 >> /var/log/cleanup_stale_orders.log 2>&1
```

## Created

- Date: 2026-01-21
- Author: Trading System Maintenance
- Issue: Orders stuck in pending status blocking new signals
- Solution: Automated cleanup via cron
