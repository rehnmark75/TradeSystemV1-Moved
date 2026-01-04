# Configuration System Reference

Complete reference for the TradeSystemV1 configuration system, covering database-driven strategy config, trailing stops, and infrastructure settings.

---

## Quick Reference

| Configuration Type | Source of Truth | Container Owner | Hot Reload |
|-------------------|-----------------|-----------------|------------|
| SMC Strategy Parameters | `strategy_config` database | task-worker | Yes (120s TTL) |
| Trailing Stops (LIVE) | `dev-app/config.py` | fastapi-dev | Container restart |
| Trailing Stops (Backtest) | `config_trailing_stops.py` | task-worker | N/A |
| Infrastructure | `forex_scanner/config.py` | task-worker | Container restart |

### Critical Warnings

```
NEVER edit the wrong container's config file!

- Trailing stops for LIVE trading: dev-app/config.py (fastapi-dev)
- Trailing stops for BACKTESTING: worker/app/forex_scanner/config_trailing_stops.py
- SMC Strategy: Database ONLY (not config files)
```

---

## 1. SMC Simple Strategy Configuration (Database-Driven)

All SMC Simple strategy parameters are stored in the `strategy_config` PostgreSQL database.

### 1.1 Config Service Usage

```python
# Get singleton service instance
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config_service
service = get_smc_simple_config_service()

# Get current config (auto-refreshes every 120s)
config = service.get_config()

# Force refresh from database
config = service.get_config(force_refresh=True)

# Convenience function (same as above)
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
config = get_smc_simple_config()
```

### 1.2 Parameter Categories (~80 parameters)

#### Tier 1: HTF Directional Bias (4H)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `htf_timeframe` | str | "4h" | Higher timeframe for bias |
| `ema_period` | int | 50 | EMA period for trend detection |
| `ema_buffer_pips` | float | 2.5 | Buffer around EMA for validation |
| `require_close_beyond_ema` | bool | True | Require candle close beyond EMA |
| `min_distance_from_ema_pips` | float | 3.0 | Minimum distance from EMA |

#### Tier 2: Entry Trigger (15M)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trigger_timeframe` | str | "15m" | Entry trigger timeframe |
| `swing_lookback_bars` | int | 20 | Bars to look back for swing points |
| `swing_strength_bars` | int | 2 | Bars on each side to confirm swing |
| `require_body_close_break` | bool | False | Require body close beyond swing |
| `wick_tolerance_pips` | float | 3.0 | Wick tolerance for break detection |
| `volume_confirmation_enabled` | bool | True | Enable volume confirmation |
| `volume_sma_period` | int | 20 | SMA period for volume comparison |
| `volume_spike_multiplier` | float | 1.2 | Multiplier for volume spike |

#### Dynamic Swing Lookback

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dynamic_swing_lookback` | bool | True | Adjust lookback based on ATR |
| `swing_lookback_atr_low` | int | 8 | ATR periods for low volatility |
| `swing_lookback_atr_high` | int | 15 | ATR periods for high volatility |
| `swing_lookback_min` | int | 15 | Minimum swing lookback |
| `swing_lookback_max` | int | 30 | Maximum swing lookback |

#### Tier 3: Execution (5M)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entry_timeframe` | str | "5m" | Entry execution timeframe |
| `pullback_enabled` | bool | True | Enable pullback entries |
| `fib_pullback_min` | float | 0.236 | Minimum Fib retracement |
| `fib_pullback_max` | float | 0.700 | Maximum Fib retracement |
| `fib_optimal_zone_min` | float | 0.382 | Optimal zone min (golden) |
| `fib_optimal_zone_max` | float | 0.618 | Optimal zone max (golden) |
| `max_pullback_wait_bars` | int | 12 | Max bars to wait for pullback |
| `pullback_confirmation_bars` | int | 2 | Bars to confirm pullback |

#### Momentum Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `momentum_mode_enabled` | bool | True | Enable momentum entries |
| `momentum_min_depth` | float | -0.50 | Min depth for momentum |
| `momentum_max_depth` | float | 0.0 | Max depth for momentum |
| `momentum_confidence_penalty` | float | 0.05 | Confidence penalty for momentum |

#### ATR-Based Swing Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_atr_swing_validation` | bool | True | Validate swings with ATR |
| `atr_period` | int | 14 | ATR calculation period |
| `min_swing_atr_multiplier` | float | 0.25 | Min swing size as ATR multiple |
| `fallback_min_swing_pips` | float | 5.0 | Fallback min swing size |

#### Momentum Quality Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `momentum_quality_enabled` | bool | True | Enable momentum quality filter |
| `min_breakout_atr_ratio` | float | 0.5 | Min breakout as ATR ratio |
| `min_body_percentage` | float | 0.20 | Min body percentage of candle |

#### Limit Order Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit_order_enabled` | bool | True | Enable limit orders |
| `limit_expiry_minutes` | int | 45 | Order expiry time |
| `pullback_offset_atr_factor` | float | 0.2 | Pullback offset as ATR factor |
| `pullback_offset_min_pips` | float | 2.0 | Min pullback offset |
| `pullback_offset_max_pips` | float | 3.0 | Max pullback offset |
| `momentum_offset_pips` | float | 3.0 | Momentum entry offset |
| `min_risk_after_offset_pips` | float | 5.0 | Min risk after offset |
| `max_sl_atr_multiplier` | float | 3.0 | Max SL as ATR multiple |
| `max_sl_absolute_pips` | float | 30.0 | Absolute max SL |
| `max_risk_after_offset_pips` | float | 55.0 | Max risk after offset |

#### Risk Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_rr_ratio` | float | 1.5 | Minimum risk-reward ratio |
| `optimal_rr_ratio` | float | 2.5 | Optimal risk-reward ratio |
| `max_rr_ratio` | float | 5.0 | Maximum risk-reward ratio |
| `sl_buffer_pips` | int | 6 | Stop loss buffer |
| `sl_atr_multiplier` | float | 1.0 | SL as ATR multiplier |
| `use_atr_stop` | bool | True | Use ATR-based stops |
| `min_tp_pips` | int | 8 | Minimum take profit |
| `use_swing_target` | bool | True | Use swing as TP target |
| `tp_structure_lookback` | int | 50 | Bars to look for structure |
| `risk_per_trade_pct` | float | 1.0 | Risk per trade (%) |

#### Fixed SL/TP Override

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fixed_sl_tp_override_enabled` | bool | True | Enable fixed SL/TP mode |
| `fixed_stop_loss_pips` | float | 9.0 | Global fixed SL (pips) |
| `fixed_take_profit_pips` | float | 15.0 | Global fixed TP (pips) |

#### Session Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_filter_enabled` | bool | True | Enable session filtering |
| `london_session_start` | time | 07:00 | London session start (UTC) |
| `london_session_end` | time | 16:00 | London session end (UTC) |
| `ny_session_start` | time | 12:00 | NY session start (UTC) |
| `ny_session_end` | time | 21:00 | NY session end (UTC) |
| `allowed_sessions` | list | ['london', 'new_york', 'overlap'] | Allowed sessions |
| `block_asian_session` | bool | True | Block Asian session |

#### Signal Limits

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_signals` | int | 3 | Max concurrent signals |
| `signal_cooldown_hours` | int | 3 | Signal cooldown period |

#### Adaptive Cooldown

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adaptive_cooldown_enabled` | bool | True | Enable adaptive cooldown |
| `base_cooldown_hours` | float | 2.0 | Base cooldown hours |
| `cooldown_after_win_multiplier` | float | 0.5 | Multiplier after win |
| `cooldown_after_loss_multiplier` | float | 1.5 | Multiplier after loss |
| `consecutive_loss_penalty_hours` | float | 1.0 | Penalty per consecutive loss |
| `max_consecutive_losses_before_block` | int | 3 | Max losses before block |
| `consecutive_loss_block_hours` | float | 8.0 | Block hours after max losses |
| `win_rate_lookback_trades` | int | 20 | Trades for win rate calc |
| `high_win_rate_threshold` | float | 0.60 | High win rate threshold |
| `low_win_rate_threshold` | float | 0.40 | Low win rate threshold |
| `critical_win_rate_threshold` | float | 0.30 | Critical win rate threshold |
| `high_win_rate_cooldown_reduction` | float | 0.25 | Cooldown reduction at high WR |
| `low_win_rate_cooldown_increase` | float | 0.50 | Cooldown increase at low WR |
| `high_volatility_atr_multiplier` | float | 1.5 | High volatility ATR mult |
| `volatility_cooldown_adjustment` | float | 0.30 | Volatility cooldown adj |
| `strong_trend_cooldown_reduction` | float | 0.30 | Strong trend reduction |
| `session_change_reset_cooldown` | bool | True | Reset on session change |
| `min_cooldown_hours` | float | 1.0 | Minimum cooldown |
| `max_cooldown_hours` | float | 12.0 | Maximum cooldown |

#### Confidence Scoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence_threshold` | float | 0.48 | Minimum confidence to trade |
| `max_confidence_threshold` | float | 0.75 | Maximum confidence (paradox filter) |
| `high_confidence_threshold` | float | 0.75 | High confidence level |
| `confidence_weights` | dict | {ema: 0.20, swing: 0.20, volume: 0.20, pullback: 0.20, rr: 0.20} | Component weights |

#### Volume Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_filter_enabled` | bool | True | Enable volume filtering |
| `min_volume_ratio` | float | 0.50 | Minimum volume ratio |
| `allow_no_volume_data` | bool | True | Allow if no volume data |

#### Dynamic Confidence Adjustments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volume_adjusted_confidence_enabled` | bool | True | Adjust confidence by volume |
| `high_volume_threshold` | float | 0.70 | High volume threshold |
| `atr_adjusted_confidence_enabled` | bool | True | Adjust confidence by ATR |
| `low_atr_threshold` | float | 0.0004 | Low ATR threshold |
| `high_atr_threshold` | float | 0.0008 | High ATR threshold |
| `ema_distance_adjusted_confidence_enabled` | bool | True | Adjust by EMA distance |
| `near_ema_threshold_pips` | float | 20.0 | Near EMA threshold |
| `far_ema_threshold_pips` | float | 30.0 | Far EMA threshold |

#### MACD Alignment Filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `macd_alignment_filter_enabled` | bool | True | Enable MACD filter |
| `macd_alignment_mode` | str | 'momentum' | Mode: 'momentum' or 'histogram' |
| `macd_min_strength` | float | 0.0 | Minimum MACD strength |

#### Logging & Debug

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_debug_logging` | bool | True | Enable debug logs |
| `log_rejected_signals` | bool | True | Log rejected signals |
| `log_swing_detection` | bool | False | Log swing detection |
| `log_ema_checks` | bool | False | Log EMA checks |

#### Rejection Tracking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rejection_tracking_enabled` | bool | True | Track rejections |
| `rejection_batch_size` | int | 50 | Batch size for DB writes |
| `rejection_log_to_console` | bool | False | Log rejections to console |
| `rejection_retention_days` | int | 90 | Days to retain data |

#### Backtest Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backtest_spread_pips` | float | 1.5 | Simulated spread |
| `backtest_slippage_pips` | float | 0.5 | Simulated slippage |

### 1.3 Per-Pair Override System

The system supports per-pair overrides via the `smc_simple_pair_overrides` table.

#### Override Resolution Order

```
1. Check smc_simple_pair_overrides for epic-specific column value
2. Check parameter_overrides JSONB field in pair override
3. Fall back to smc_simple_global_config value
4. Fall back to SMCSimpleConfig dataclass default
```

#### Per-Pair Override Fields

| Field | Type | Description |
|-------|------|-------------|
| `fixed_stop_loss_pips` | NUMERIC(5,1) | Per-pair fixed SL |
| `fixed_take_profit_pips` | NUMERIC(5,1) | Per-pair fixed TP |
| `min_confidence` | DECIMAL(4,3) | Per-pair min confidence |
| `max_confidence` | DECIMAL(4,3) | Per-pair max confidence |
| `sl_buffer_pips` | INTEGER | Per-pair SL buffer |
| `min_volume_ratio` | DECIMAL(4,2) | Per-pair min volume |
| `macd_filter_enabled` | BOOLEAN | Per-pair MACD toggle |
| `allow_asian_session` | BOOLEAN | Allow Asian session |
| `high_volume_confidence` | DECIMAL(4,3) | Confidence at high volume |
| `low_atr_confidence` | DECIMAL(4,3) | Confidence at low ATR |
| `high_atr_confidence` | DECIMAL(4,3) | Confidence at high ATR |
| `near_ema_confidence` | DECIMAL(4,3) | Confidence near EMA |
| `far_ema_confidence` | DECIMAL(4,3) | Confidence far from EMA |
| `blocking_conditions` | JSONB | Complex blocking rules |
| `parameter_overrides` | JSONB | Arbitrary parameter overrides |

#### Per-Pair API Methods

```python
config = get_smc_simple_config()

# Get per-pair fixed SL (returns None if disabled)
sl = config.get_pair_fixed_stop_loss('CS.D.EURUSD.CEEM.IP')

# Get per-pair fixed TP
tp = config.get_pair_fixed_take_profit('CS.D.EURUSD.CEEM.IP')

# Get per-pair minimum confidence
min_conf = config.get_pair_min_confidence('CS.D.EURUSD.CEEM.IP')

# Get per-pair maximum confidence (paradox filter)
max_conf = config.get_pair_max_confidence('CS.D.EURUSD.CEEM.IP')

# Check if MACD filter enabled for pair
macd_on = config.is_macd_filter_enabled('CS.D.EURUSD.CEEM.IP')

# Check if Asian session allowed
asian_ok = config.is_asian_session_allowed('CS.D.USDCHF.MINI.IP')

# Get dynamic confidence based on conditions
threshold = config.get_dynamic_confidence(
    epic='CS.D.EURUSD.CEEM.IP',
    volume_ratio=0.85,
    atr_value=0.0005,
    ema_distance_pips=25.5
)

# Check if signal should be blocked
blocked, reason = config.should_block_signal('CS.D.EURUSD.CEEM.IP', signal_data)
```

#### Blocking Conditions JSONB Structure

```json
{
    "enabled": true,
    "blocking_logic": "any",
    "conditions": {
        "max_ema_distance_pips": 50.0,
        "require_volume_confirmation": true,
        "block_momentum_without_volume": true,
        "min_confidence_override": 0.55
    }
}
```

### 1.4 Database Tables

#### smc_simple_global_config

Main table with ~80 columns for all strategy parameters.

```sql
-- View current config
SELECT version, strategy_status, min_confidence_threshold,
       fixed_stop_loss_pips, fixed_take_profit_pips
FROM smc_simple_global_config
WHERE is_active = TRUE;
```

#### smc_simple_pair_overrides

Per-pair overrides linked to global config.

```sql
-- View per-pair settings
SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips,
       min_confidence, max_confidence, macd_filter_enabled
FROM smc_simple_pair_overrides
WHERE is_enabled = TRUE
ORDER BY epic;
```

#### smc_simple_config_audit

Audit trail for all configuration changes.

```sql
-- View recent changes
SELECT changed_at, changed_by, change_type, field_name,
       old_value, new_value
FROM smc_simple_config_audit
ORDER BY changed_at DESC
LIMIT 20;
```

#### smc_simple_parameter_metadata

UI metadata for parameter display.

```sql
-- Get parameter info
SELECT parameter_name, display_name, category, data_type,
       min_value, max_value, description
FROM smc_simple_parameter_metadata
WHERE category = 'risk_management';
```

---

## 2. Trailing Stop Configuration (File-Based)

### 2.1 Source of Truth

```
LIVE TRADING: dev-app/config.py (fastapi-dev container)
BACKTESTING:  worker/app/forex_scanner/config_trailing_stops.py (task-worker)
```

### 2.2 4-Stage Progressive System

Based on MAE/MFE analysis from December 2025:

| Stage | Trigger | Action | Parameters |
|-------|---------|--------|------------|
| Early BE | +15-20 pips | SL to entry+buffer | `early_breakeven_trigger_points`, `early_breakeven_buffer_points` |
| Stage 1 | +25-30 pips | Lock profit | `stage1_trigger_points`, `stage1_lock_points` |
| Stage 2 | +38-45 pips | Lock more | `stage2_trigger_points`, `stage2_lock_points` |
| Stage 3 | +50-60 pips | ATR trailing | `stage3_trigger_points`, `stage3_atr_multiplier`, `stage3_min_distance` |
| Partial Close | +20-25 pips | Close 40% | `partial_close_trigger_points`, `partial_close_size` |

### 2.3 Configuration Structure

```python
# Per-pair trailing config
{
    'early_breakeven_trigger_points': 15,   # Trigger early BE at +15 pips
    'early_breakeven_buffer_points': 2,     # Move SL to entry+2
    'stage1_trigger_points': 25,            # Stage 1 at +25 pips
    'stage1_lock_points': 12,               # Lock +12 pips
    'stage2_trigger_points': 38,            # Stage 2 at +38 pips
    'stage2_lock_points': 20,               # Lock +20 pips
    'stage3_trigger_points': 50,            # Stage 3 at +50 pips
    'stage3_atr_multiplier': 2.0,           # Trail at 2.0x ATR
    'stage3_min_distance': 8,               # Min 8 pip distance
    'min_trail_distance': 10,               # Min trailing distance
    'break_even_trigger_points': 18,        # Legacy BE trigger
    'enable_partial_close': True,           # Enable partial close
    'partial_close_trigger_points': 20,     # Partial at +20 pips
    'partial_close_size': 0.4,              # Close 40%
}
```

### 2.4 Per-Pair Settings (v3.0.0)

| Pair Category | Early BE Trigger | Stage 1 | Stage 2 | Stage 3 |
|---------------|------------------|---------|---------|---------|
| Major USD (EURUSD, AUDUSD, NZDUSD, USDCAD, USDCHF) | 15 pips | 25 pips | 38 pips | 50 pips |
| GBP pairs (GBPUSD) | 15 pips | 25 pips | 38 pips | 50 pips |
| JPY crosses (USDJPY, EURJPY, AUDJPY, etc.) | 20 pips | 30 pips | 45 pips | 60 pips |

### 2.5 Usage in Code

```python
from config import get_trailing_config_for_epic

# Get config for specific pair
config = get_trailing_config_for_epic('CS.D.EURUSD.CEEM.IP')
early_be = config['early_breakeven_trigger_points']
```

### 2.6 Updating Trailing Config

```bash
# Edit the source file (fastapi-dev container owns this)
nano dev-app/config.py

# Restart container to apply
docker restart fastapi-dev

# Verify
docker exec fastapi-dev python3 -c "
from config import PAIR_TRAILING_CONFIGS
print(PAIR_TRAILING_CONFIGS['CS.D.EURUSD.CEEM.IP'])
"
```

---

## 3. Infrastructure Configuration

Located in `worker/app/forex_scanner/config.py` (infrastructure only after January 2026 cleanup).

### 3.1 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | Main forex database | `postgresql://postgres:postgres@localhost:5432/forex` |
| `STRATEGY_CONFIG_DATABASE_URL` | Strategy config DB | `postgresql://postgres:postgres@postgres:5432/strategy_config` |
| `CLAUDE_API_KEY` | Claude AI integration | None |
| `ORDER_API_URL` | Order placement endpoint | `http://fastapi-dev:8000/orders/place-order` |
| `API_SUBSCRIPTION_KEY` | API authentication key | (default key) |
| `USER_TIMEZONE` | User timezone | `Europe/Stockholm` |
| `MINIO_ENDPOINT` | MinIO object storage | `minio:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | `minioadmin` |
| `MINIO_SECRET_KEY` | MinIO secret key | `minioadmin123` |
| `MINIO_BUCKET_NAME` | Chart storage bucket | `claude-charts` |

### 3.2 Static Pair Mappings

#### PAIR_INFO

```python
PAIR_INFO = {
    'CS.D.EURUSD.CEEM.IP': {'pair': 'EURUSD', 'pip_multiplier': 10000},
    'CS.D.GBPUSD.MINI.IP': {'pair': 'GBPUSD', 'pip_multiplier': 10000},
    'CS.D.USDJPY.MINI.IP': {'pair': 'USDJPY', 'pip_multiplier': 100},
    # ... etc
}
```

#### EPIC_LIST

```python
EPIC_LIST = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.USDCHF.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.AUDJPY.MINI.IP'
]
```

#### EPIC_MAP

```python
# Scanner epic -> Trading API epic
EPIC_MAP = {
    "CS.D.EURUSD.CEEM.IP": "EURUSD.1.MINI",
    "CS.D.GBPUSD.MINI.IP": "GBPUSD.1.MINI",
    "CS.D.USDJPY.MINI.IP": "USDJPY.100.MINI",
    # ... etc
}

REVERSE_EPIC_MAP = {v: k for k, v in EPIC_MAP.items()}
```

---

## 4. Container Ownership Matrix

| Setting Type | Container | File Path | Hot Reload | Restart Required |
|--------------|-----------|-----------|------------|------------------|
| SMC Strategy Parameters | task-worker | Database (strategy_config) | Yes (120s TTL) | No |
| Trailing Stops (LIVE) | fastapi-dev | `dev-app/config.py` | No | Yes |
| Trailing Stops (Backtest) | task-worker | `forex_scanner/config_trailing_stops.py` | No | Yes |
| Epic Lists | task-worker | `forex_scanner/config.py` | No | Yes |
| API URLs | task-worker | `forex_scanner/config.py` | No | Yes |
| Environment Secrets | All | Docker environment | No | Yes |

### Container Access

```bash
# task-worker: Strategy scanning, backtesting
docker exec -it task-worker python /app/forex_scanner/script.py

# fastapi-dev: Trade execution, trailing stops
docker exec -it fastapi-dev python3 -c "from config import PAIR_TRAILING_CONFIGS; print(PAIR_TRAILING_CONFIGS)"

# streamlit: Read-only dashboard (mounts from fastapi-dev)
# Mounts: ./dev-app/config.py:/app/trailing_config.py:ro
```

---

## 5. Common Operations

### View Current SMC Config

```bash
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT version, strategy_status, min_confidence_threshold,
       fixed_stop_loss_pips, fixed_take_profit_pips
FROM smc_simple_global_config WHERE is_active = TRUE;"
```

### Update Global SL/TP

```bash
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_global_config
SET fixed_stop_loss_pips = 10, fixed_take_profit_pips = 18
WHERE is_active = TRUE;"
```

### Set Per-Pair Override

```bash
docker exec postgres psql -U postgres -d strategy_config -c "
UPDATE smc_simple_pair_overrides
SET fixed_stop_loss_pips = 12, fixed_take_profit_pips = 22
WHERE epic = 'CS.D.USDJPY.MINI.IP';"
```

### View Per-Pair Settings

```bash
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic, fixed_stop_loss_pips, fixed_take_profit_pips,
       min_confidence, max_confidence
FROM smc_simple_pair_overrides ORDER BY epic;"
```

### Invalidate Config Cache

```python
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config_service
service = get_smc_simple_config_service()
service.invalidate_cache()
# Or just wait 120 seconds for auto-refresh
```

### View Audit Trail

```bash
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT changed_at, field_name, old_value, new_value
FROM smc_simple_config_audit
ORDER BY changed_at DESC LIMIT 10;"
```

---

## 6. Troubleshooting

### Config Changes Not Taking Effect

1. **SMC Strategy**: Wait 120 seconds for cache TTL, or call `service.invalidate_cache()`
2. **Trailing Stops**: Restart `fastapi-dev` container
3. **Infrastructure**: Restart `task-worker` container

### Wrong Container Edited

If you edited the wrong file:
- `dev-app/config.py` is for LIVE trailing (fastapi-dev)
- `forex_scanner/config_trailing_stops.py` is for BACKTEST only (task-worker)
- SMC settings must be in database, not config files

### Database Connection Issues

```bash
# Test strategy_config connection
docker exec postgres psql -U postgres -d strategy_config -c "SELECT 1;"

# Check service can connect
docker exec task-worker python -c "
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
config = get_smc_simple_config()
print(f'Loaded: v{config.version} from {config.source}')
"
```

### Cache Fallback Behavior

When database is unavailable:
1. Service uses last-known-good cached config
2. If no cache, uses default SMCSimpleConfig()
3. `config.source` indicates: 'database', 'cache', or 'default'

---

## 7. Migration Patterns

### Adding New Parameters

1. Add column to `smc_simple_global_config`:
   ```sql
   ALTER TABLE smc_simple_global_config
   ADD COLUMN new_param_name NUMERIC(5,2) DEFAULT 1.5;
   ```

2. Add to `SMCSimpleConfig` dataclass:
   ```python
   new_param_name: float = 1.5
   ```

3. Add to `_build_config_from_rows` mapping:
   ```python
   direct_mappings = [
       # ... existing ...
       'new_param_name',
   ]
   ```

### Adding Per-Pair Override

1. Add column to `smc_simple_pair_overrides`:
   ```sql
   ALTER TABLE smc_simple_pair_overrides
   ADD COLUMN new_override NUMERIC(5,2);
   ```

2. Add getter method to `SMCSimpleConfig`:
   ```python
   def get_pair_new_override(self, epic: str) -> float:
       if epic in self._pair_overrides:
           override = self._pair_overrides[epic]
           if override.get('new_override') is not None:
               return override['new_override']
       return self.new_param_name
   ```

3. Update `_build_config_from_rows` to include new field in pair override dict.

### Migration File Location

`worker/app/forex_scanner/migrations/`

Template: `migrations/templates/strategy_config_template.sql`
