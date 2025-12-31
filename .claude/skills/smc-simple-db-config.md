# SMC Simple Strategy: Database-Driven Configuration

This skill documents the database-driven configuration system for the SMC Simple strategy, including architecture, common issues, and solutions.

---

## Architecture Overview

### Database: `strategy_config`

The SMC Simple strategy uses a dedicated PostgreSQL database (`strategy_config`) instead of file-based config.

**Tables:**
- `smc_simple_global_config` - Global default parameters (~80 columns)
- `smc_simple_pair_overrides` - Per-pair override settings
- `smc_simple_config_audit` - Change history/audit trail
- `smc_simple_parameter_metadata` - UI rendering metadata

### Key Files

| Location | File | Purpose |
|----------|------|---------|
| Worker | `worker/app/forex_scanner/services/smc_simple_config_service.py` | Config service with caching |
| Streamlit | `streamlit/services/smc_simple_config_service.py` | Streamlit-specific wrapper with save methods |
| Streamlit | `streamlit/components/tabs/smc_config_tab.py` | Configuration management UI |
| Migration | `worker/app/forex_scanner/migrations/create_strategy_config_db.sql` | Database schema |
| Migration | `worker/app/forex_scanner/migrations/seed_smc_simple_config.py` | Seed script |

---

## Configuration Flow

```
1. Strategy Initialization (SMCSimpleStrategy._load_config())
   └── SMCSimpleConfigService.get_config()
         ├── Check in-memory cache (TTL: 120s)
         ├── If expired/missing: query strategy_config database
         └── Fallback to last-known-good if DB unavailable

2. Hot Reload
   └── Strategy re-reads config every ~2 min scan cycle
   └── Changes in Streamlit UI take effect within 2 minutes

3. Per-Pair Resolution
   └── get_effective_config_for_pair(epic)
         ├── Start with global config
         └── Overlay pair-specific overrides
```

---

## Streamlit UI Structure

Located at: **Settings > SMC Config**

**4 Sub-tabs:**
1. **Global Configuration** - Expandable sections for all parameters
2. **Per-Pair Overrides** - Pair-specific settings with inheritance
3. **Effective Config Viewer** - Merged view per pair
4. **Audit Trail** - Recent configuration changes

---

## Common Issues & Solutions

### Issue: "Pending changes" showing when no changes made

**Problem:** Fields with `None` in database show effective/default values but comparison flags them as changed.

**Solution:** Compare against the *effective* value (what's displayed), not the raw DB value.

```python
# WRONG - causes false positives when DB value is None
if not _values_equal(new_value, existing.get('field_name')):
    changes['field_name'] = new_value

# CORRECT - compare against what was actually displayed
field_value = existing.get('field_name')
field_effective = float(field_value) if field_value is not None else default_value
if not _values_equal(new_value, field_effective):
    changes['field_name'] = new_value
```

### Issue: Checkboxes showing unchecked when global is enabled

**Problem:** Pair override field is `None` (no override), but checkbox shows `False` instead of inheriting global.

**Solution:** Check if override value is `None` and fall back to global setting.

```python
macd_override_value = existing.get('macd_filter_enabled')
global_macd = config.get('macd_alignment_filter_enabled', True)
macd_effective = macd_override_value if macd_override_value is not None else global_macd
```

### Issue: Floating point precision false positives

**Problem:** Values like `0.236` and `0.23600000000000002` are flagged as different.

**Solution:** Use tolerance-based comparison.

```python
def _values_equal(new_val, old_val, tolerance: float = 1e-9) -> bool:
    if new_val is None and old_val is None:
        return True
    if new_val is None or old_val is None:
        return False
    if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
        return abs(float(new_val) - float(old_val)) < tolerance
    return new_val == old_val
```

### Issue: "Slice indices must be integers" error

**Problem:** PostgreSQL returns `NUMERIC` columns as `Decimal`, but DataFrame slicing needs `int`.

**Solution:** Convert integer fields explicitly in config service.

```python
int_fields = {
    'ema_period', 'swing_lookback_bars', 'swing_strength_bars',
    'volume_sma_period', 'max_pullback_wait_bars', 'atr_period',
    # ... etc
}

for attr_name in direct_mappings:
    if attr_name in global_row and global_row[attr_name] is not None:
        value = global_row[attr_name]
        if attr_name in int_fields:
            value = int(value)
        elif hasattr(value, 'as_integer_ratio'):  # Decimal to float
            value = float(value)
        setattr(config, attr_name, value)
```

### Issue: Multiselect "default value not in options"

**Problem:** DB has pair like `CS.D.EURUSD.CEEM.IP` but options only include MINI variants.

**Solution:** Merge known pairs with DB pairs to ensure all defaults are valid options.

```python
known_pairs = [
    "CS.D.EURUSD.CEEM.IP",  # EURUSD only available as CEEM
    "CS.D.GBPUSD.MINI.IP",
    # ... etc
]
all_pairs = sorted(set(known_pairs) | set(enabled_pairs))
```

---

## Pair Override Inheritance Rules

When editing per-pair overrides, fields inherit from global config if not explicitly set:

| Field | Inherits From | Default When Null |
|-------|---------------|-------------------|
| Allow Asian Session | `block_asian_session` (inverted) | `not global.block_asian_session` |
| SL Buffer (pips) | `sl_buffer_pips` | Global value |
| Min Confidence | `min_confidence_threshold` | Global value |
| MACD Filter Enabled | `macd_alignment_filter_enabled` | Global value |
| High Volume Confidence | N/A (pair-specific) | 0.45 |
| Low ATR Confidence | N/A (pair-specific) | 0.44 |
| High ATR Confidence | N/A (pair-specific) | 0.52 |
| Near EMA Confidence | N/A (pair-specific) | 0.44 |

**Help text should indicate:**
- "Inherited from global (X)" when using global value
- "Explicit override for this pair" when explicitly set

---

## Database Queries

### Check current global config
```sql
SELECT * FROM smc_simple_global_config WHERE is_active = TRUE;
```

### Check pair overrides
```sql
SELECT * FROM smc_simple_pair_overrides WHERE config_id = 1;
```

### View audit trail
```sql
SELECT * FROM smc_simple_config_audit ORDER BY changed_at DESC LIMIT 20;
```

### Test from container
```bash
docker exec postgres psql -U postgres -d strategy_config -c "SELECT ema_period, min_confidence_threshold FROM smc_simple_global_config WHERE is_active = TRUE;"
```

---

## Service Layer Usage

### Worker Container (task-worker)
```python
from forex_scanner.services.smc_simple_config_service import get_smc_simple_config_service

service = get_smc_simple_config_service()
config = service.get_config()  # Returns SMCSimpleConfig dataclass

# Get effective config for specific pair
effective = service.get_effective_config_for_pair('CS.D.EURUSD.CEEM.IP')
```

### Streamlit Container
```python
from services.smc_simple_config_service import (
    get_global_config,
    get_pair_overrides,
    save_global_config,
    save_pair_override,
)

config = get_global_config()  # Returns dict
overrides = get_pair_overrides(config['id'])

# Save changes
save_global_config(
    config_id=config['id'],
    updates={'min_confidence_threshold': 0.50},
    updated_by='streamlit_user',
    change_reason='Testing new threshold'
)
```

---

## Caching Behavior

- **TTL:** 120 seconds (2 minutes)
- **Force refresh:** `service.get_config(force_refresh=True)`
- **Last-known-good fallback:** If DB unavailable, uses cached config
- **Thread-safe:** Uses `RLock` for concurrent access

---

## Related Files

- Original file-based config: `worker/app/forex_scanner/configdata/strategies/config_smc_simple.py` (1,234 lines, 100+ parameters)
- Strategy using config: `worker/app/forex_scanner/core/strategies/smc_simple_strategy.py`
- DB connection utility: `streamlit/services/db_utils.py`
