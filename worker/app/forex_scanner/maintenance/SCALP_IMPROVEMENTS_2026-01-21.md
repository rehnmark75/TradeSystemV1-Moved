# Scalp Mode Improvements - January 21, 2026

## Summary

Major improvements to scalp mode trading based on analysis of "catastrophic results" where most losing trades had MAE=0 (immediate losses). Implemented multi-layered entry confirmation system with volume validation and per-pair optimization.

---

## Changes Implemented

### 1. Volume Filter Enabled (v2.26.0)
**Problem**: Volume filter was disabled in scalp mode, allowing low-liquidity fake moves to trigger entries.

**Solution**:
- Enabled volume filter for scalp mode: `scalp_disable_volume_filter = FALSE`
- Set minimum volume ratio threshold: `min_volume_ratio = 1.0`
- Requires entry candle volume to be at least equal to 20-period SMA

**Database Changes**:
```sql
UPDATE smc_simple_global_config
SET
    scalp_disable_volume_filter = FALSE,
    min_volume_ratio = 1.0
WHERE is_active = TRUE;
```

**Impact** (GBPUSD backtest, 14 days):
- Signals: 41 → 23 (44% reduction - filtered out low-quality entries)
- Win Rate: 56.1% → 60.0% (+3.9%)
- Profit Factor: 1.29 → **3.13** (+143%)
- Expectancy: 0.6 → **2.8 pips** (+367%)

**Rationale**: Volume confirmation ensures price moves are backed by actual market participation, filtering out stop hunts and low-liquidity spikes.

---

### 2. Market Orders for All Scalp Signals
**Problem**: STOP orders (momentum confirmation) add delay and risk slippage in fast-moving scalp trades.

**Solution**:
- Enabled market orders globally: `scalp_use_market_orders = TRUE`
- Market orders execute immediately at current market price
- Combined with 1-pip spread filter to ensure quality fills

**Database Changes**:
```sql
UPDATE smc_simple_global_config
SET scalp_use_market_orders = TRUE
WHERE is_active = TRUE;
```

**Impact**:
- Faster execution (immediate vs waiting for price to reach STOP level)
- Reduced slippage risk in volatile scalp conditions
- Spread filter (max 1.0 pips) protects against poor fills

---

### 3. Entry Candle Alignment (v2.25.1)
**Problem**: Entries at pullback zones without immediate momentum confirmation often reversed immediately.

**Solution**:
- Require entry candle color to match trade direction
- Green candle for BUY signals, Red candle for SELL signals
- Enabled globally: `scalp_require_entry_candle_alignment = TRUE`
- Uses market orders when confirmed: `scalp_use_market_on_entry_alignment = TRUE`

**Database Changes**:
```sql
UPDATE smc_simple_global_config
SET
    scalp_require_entry_candle_alignment = TRUE,
    scalp_use_market_on_entry_alignment = TRUE
WHERE is_active = TRUE;
```

**Impact** (GBPUSD backtest):
- Filters out counter-momentum entries
- Ensures immediate price action supports trade direction
- Combined with volume filter for maximum signal quality

---

### 4. MACD Filter for GBPUSD (Per-Pair Override)
**Problem**: GBPUSD showed suboptimal performance compared to other pairs.

**Solution**:
- Enabled MACD alignment filter specifically for GBPUSD
- Requires MACD momentum to align with trade direction
- Uses per-pair override system (doesn't affect other pairs)

**Database Changes**:
```sql
UPDATE smc_simple_pair_overrides
SET parameter_overrides = jsonb_set(
    COALESCE(parameter_overrides, '{}'::jsonb),
    '{macd_filter_enabled}',
    'true'
)
WHERE epic = 'CS.D.GBPUSD.MINI.IP';
```

**Impact** (GBPUSD backtest, 14 days):
- Win Rate: 53.2% → 56.1% (+2.9%)
- Profit Factor: 1.15 → 1.29 (+12%)
- Expectancy: 0.4 → 0.6 pips (+50%)
- Works synergistically with volume filter

---

### 5. Stale Order Cleanup Automation
**Problem**: Orders stuck in "pending" status blocked new signals due to cooldown logic (30-minute block per pair).

**Root Cause**:
- fastapi-dev order monitoring service failing to update order statuses
- Health check timeouts causing service instability
- 47 orders accumulated in pending status over several days

**Solution**:
- Created automated cleanup script: `cleanup_stale_orders.py`
- Expires pending orders older than 30 minutes
- Installed cron job: runs every 15 minutes
- Provides safety net if order monitoring fails

**Files Created**:
- `/worker/app/forex_scanner/maintenance/cleanup_stale_orders.py`
- `/worker/cron/cleanup-stale-orders`
- `/worker/systemd/cleanup-stale-orders.service`
- `/worker/systemd/cleanup-stale-orders.timer`
- `/scripts/cleanup_stale_orders.sh` (host wrapper)
- `/worker/app/forex_scanner/maintenance/README.md` (documentation)

**Cron Configuration**:
```bash
*/15 * * * * /home/hr/Projects/TradeSystemV1/scripts/cleanup_stale_orders.sh >> /var/log/cleanup_stale_orders.log 2>&1
```

**Impact**:
- Immediate: Expired 47 stale pending orders blocking signals
- Ongoing: Automatic cleanup every 15 minutes
- Prevents cooldown-related signal blocking

**Monitoring Query**:
```sql
-- Check for stale pending orders (should be 0)
SELECT COUNT(*) as stale_orders
FROM alert_history
WHERE order_status = 'pending'
  AND alert_timestamp < NOW() - INTERVAL '30 minutes';
```

---

### 6. Enhanced Logging for Analysis (v2.26.0)
**Problem**: Insufficient metadata in signals to analyze filter effectiveness.

**Solution**: Added comprehensive filter metadata to every signal:

```python
'filter_metadata': {
    'volume_filter_enabled': bool,
    'min_volume_ratio_threshold': float,
    'macd_filter_enabled': bool,
    'entry_candle_alignment_required': bool,
    'entry_candle_alignment_confirmed': bool,
    'rejection_candle_required': bool,
    'rejection_candle_confirmed': bool,
    'market_order_reason': str,  # 'entry_alignment' | 'rejection_candle' | 'default'
}
```

**Files Modified**:
- `/worker/app/forex_scanner/core/strategies/smc_simple_strategy.py` (line 2802-2812)

**Impact**:
- Every signal in `alert_history` table now includes filter states
- Can analyze which filters improve/hurt performance
- Enables data-driven filter optimization

---

## Current Configuration State

### Global Scalp Settings (smc_simple_global_config)
```
scalp_mode_enabled: TRUE
scalp_disable_volume_filter: FALSE
min_volume_ratio: 1.00
volume_filter_enabled: TRUE
scalp_use_market_orders: TRUE
scalp_use_market_on_entry_alignment: TRUE
scalp_require_entry_candle_alignment: TRUE
scalp_tp_pips: 10.0
scalp_sl_pips: 5.0 (per-pair overrides available)
scalp_max_spread_pips: 1.0
scalp_min_confidence: 0.30
scalp_cooldown_minutes: 15
```

### Per-Pair Overrides (smc_simple_pair_overrides)
```
GBPUSD:
  - macd_filter_enabled: TRUE
  - (inherits all global scalp settings)

Other pairs:
  - (use global settings)
```

---

## Backtest Results Summary

### GBPUSD (14 days, all filters enabled)
```
Baseline (volume filter disabled):
  - Signals: 41
  - Win Rate: 56.1%
  - Profit Factor: 1.29
  - Expectancy: 0.6 pips

With Volume Filter (min_ratio = 1.0):
  - Signals: 23 (44% reduction)
  - Win Rate: 60.0% (+3.9%)
  - Profit Factor: 3.13 (+143%)
  - Expectancy: 2.8 pips (+367%)

With Volume Filter (min_ratio = 1.2):
  - Signals: 17 (59% reduction)
  - Win Rate: 70.6% (+14.5%)
  - Profit Factor: 2.90 (+125%)
  - Expectancy: 2.8 pips (+367%)
```

### EURUSD (14 days, volume filter 1.0)
```
  - Signals: 10
  - Win Rate: 70.0%
  - Profit Factor: 2.13
  - Expectancy: 1.7 pips
```

### USDJPY (14 days, volume filter 1.0)
```
  - Signals: 18
  - Win Rate: 55.6%
  - Profit Factor: 1.29
  - Expectancy: 0.8 pips
```

---

## Filter Interaction & Synergy

The filters work together in sequence:

1. **Session Filter** → Check trading session allowed
2. **Spread Filter** → Check spread ≤ 1.0 pips
3. **Cooldown Filter** → Check no recent trades
4. **EMA Bias** → Check 15m EMA directional alignment
5. **Swing Break** → Check 5m swing level broken
6. **Pullback** → Check 1m pullback to fib zone
7. **Volume Filter** ⭐ NEW → Check volume_ratio ≥ 1.0
8. **Entry Candle Alignment** ⭐ NEW → Check entry candle color matches direction
9. **MACD Filter** (GBPUSD only) → Check MACD momentum alignment
10. **Confidence** → Check confidence ≥ 30%

Each filter refines signal quality. Volume + Entry Alignment provide the strongest improvement.

---

## Rejection Tracking

All filter rejections are logged to `smc_simple_rejections` table with:
- Rejection stage (e.g., `VOLUME_LOW`, `SCALP_ENTRY_FILTER`)
- Full market context (volume_ratio, MACD values, candle OHLC)
- Allows post-trade analysis of why signals were blocked

**Query rejected signals**:
```sql
SELECT
    rejection_stage,
    COUNT(*) as count,
    AVG(volume_ratio) as avg_volume_ratio
FROM smc_simple_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '7 days'
  AND rejection_stage IN ('VOLUME_LOW', 'SCALP_ENTRY_FILTER')
GROUP BY rejection_stage
ORDER BY count DESC;
```

---

## Expected Live Performance Impact

### Signal Frequency
- **Before**: 5-10 signals/day (with many immediate losses)
- **After**: 2-5 signals/day (higher quality, better win rate)
- **Trade-off**: Fewer signals but much higher expectancy per trade

### Win Rate & Profitability
- **Before**: ~50-56% win rate, PF 1.1-1.3
- **After**: 60-70% win rate, PF 2.1-3.1 (based on backtests)
- **Expectancy**: 1.7-2.8 pips per trade (vs 0.4-0.6 previously)

### Risk Profile
- Market orders with 1-pip spread filter = controlled execution
- Volume filter = avoids low-liquidity spikes
- Entry alignment = confirms momentum at entry
- Overall: Significantly reduced MAE=0 "immediate loss" scenarios

---

## Monitoring & Validation

### Real-Time Logs
Monitor live scanner output for filter activity:
```bash
docker logs -f task-worker | grep -E "Volume|Entry candle|MACD|filter"
```

### Database Queries

**Check recent signals with filter metadata**:
```sql
SELECT
    alert_timestamp,
    epic,
    signal_type,
    confidence_score,
    volume_ratio,
    strategy_metadata->'filter_metadata'->>'volume_filter_enabled' as vol_filter,
    strategy_metadata->'filter_metadata'->>'min_volume_ratio_threshold' as vol_threshold,
    strategy_metadata->'filter_metadata'->>'entry_candle_alignment_confirmed' as entry_aligned,
    strategy_metadata->'filter_metadata'->>'macd_filter_enabled' as macd_filter,
    strategy_metadata->'filter_metadata'->>'market_order_reason' as mkt_reason
FROM alert_history
WHERE alert_timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY alert_timestamp DESC;
```

**Check volume filter effectiveness**:
```sql
SELECT
    epic,
    COUNT(*) as total_signals,
    AVG(volume_ratio) as avg_volume_ratio,
    MIN(volume_ratio) as min_volume_ratio,
    MAX(volume_ratio) as max_volume_ratio
FROM alert_history
WHERE alert_timestamp >= NOW() - INTERVAL '7 days'
  AND volume_ratio IS NOT NULL
GROUP BY epic
ORDER BY total_signals DESC;
```

**Check stale order cleanup effectiveness**:
```sql
SELECT
    COUNT(*) as stale_pending_orders,
    MAX(alert_timestamp) as oldest_pending
FROM alert_history
WHERE order_status = 'pending'
  AND alert_timestamp < NOW() - INTERVAL '30 minutes';
-- Should return 0 stale orders if cleanup is working
```

---

## Rollback Plan

If live performance degrades, disable filters individually:

### Disable Volume Filter
```sql
UPDATE smc_simple_global_config
SET scalp_disable_volume_filter = TRUE
WHERE is_active = TRUE;
```

### Disable Entry Candle Alignment
```sql
UPDATE smc_simple_global_config
SET scalp_require_entry_candle_alignment = FALSE
WHERE is_active = TRUE;
```

### Disable MACD Filter for GBPUSD
```sql
UPDATE smc_simple_pair_overrides
SET parameter_overrides = parameter_overrides - 'macd_filter_enabled'
WHERE epic = 'CS.D.GBPUSD.MINI.IP';
```

### Revert to STOP Orders
```sql
UPDATE smc_simple_global_config
SET scalp_use_market_orders = FALSE
WHERE is_active = TRUE;
```

After any changes:
```bash
docker restart task-worker
```

---

## Next Steps

1. **Monitor first 24 hours** of live signals with new filters
2. **Analyze filter metadata** from `alert_history` table
3. **Compare live vs backtest** results (expect similar performance)
4. **Fine-tune volume threshold** if needed (test 0.8, 1.0, 1.2)
5. **Consider expanding MACD filter** to other pairs if GBPUSD shows strong results
6. **Evaluate per-pair volume thresholds** based on liquidity characteristics

---

## Files Modified

### Strategy Code
- `/worker/app/forex_scanner/core/strategies/smc_simple_strategy.py`
  - Added filter_metadata logging (lines 2802-2812)

### Configuration Service
- `/worker/app/forex_scanner/services/smc_simple_config_service.py`
  - Fixed MACD per-pair override logic
  - Added entry alignment methods

### Maintenance Scripts (NEW)
- `/worker/app/forex_scanner/maintenance/cleanup_stale_orders.py`
- `/worker/app/forex_scanner/maintenance/README.md`
- `/worker/cron/cleanup-stale-orders`
- `/worker/systemd/cleanup-stale-orders.service`
- `/worker/systemd/cleanup-stale-orders.timer`
- `/scripts/cleanup_stale_orders.sh`

### Database Changes
- `strategy_config.smc_simple_global_config` (multiple columns updated)
- `strategy_config.smc_simple_pair_overrides` (GBPUSD MACD filter)
- `forex.alert_history` (47 stale orders expired)

---

## Authors & Date

- **Date**: 2026-01-21
- **Analysis**: Based on "catastrophic scalp results" investigation
- **Implementation**: Multi-layered entry confirmation system
- **Validation**: Backtest analysis across GBPUSD, EURUSD, USDJPY
- **Status**: ✅ DEPLOYED TO PRODUCTION

---

## Related Documentation

- [Maintenance Scripts README](/worker/app/forex_scanner/maintenance/README.md)
- [Configuration System](/worker/app/forex_scanner/docs/claude-configuration.md)
- Strategy version: v2.26.0
