# Adding a New Strategy

This guide explains how to add a new trading strategy to the forex scanner.

## Quick Start (5 Steps)

### Step 1: Copy the Template

```bash
# Copy strategy template
cp worker/app/forex_scanner/core/strategies/templates/strategy_template.py \
   worker/app/forex_scanner/core/strategies/my_momentum_strategy.py

# Copy migration template
cp worker/app/forex_scanner/migrations/templates/strategy_config_template.sql \
   worker/app/forex_scanner/migrations/create_my_momentum_config.sql
```

### Step 2: Customize the Strategy File

Edit `my_momentum_strategy.py`:

1. Replace all `TEMPLATE` with `MY_MOMENTUM`
2. Replace all `Template` with `MyMomentum`
3. Replace all `template` with `my_momentum`
4. Implement the `detect_signal()` method
5. Customize the config dataclass parameters

```python
# Example: Change strategy name
@register_strategy('MY_MOMENTUM')  # Auto-registers with registry
class MyMomentumStrategy(StrategyInterface):
    ...
```

### Step 3: Create Database Migration

Edit `create_my_momentum_config.sql`:

1. Replace all `template` with `my_momentum`
2. Replace all `TEMPLATE` with `MY_MOMENTUM`
3. Customize the default parameters

Run the migration:
```bash
docker exec postgres psql -U postgres -d strategy_config \
    -f /app/forex_scanner/migrations/create_my_momentum_config.sql
```

### Step 4: Enable the Strategy

Enable in database:
```sql
-- In strategy_config database
INSERT INTO enabled_strategies (strategy_name, is_enabled)
VALUES ('MY_MOMENTUM', TRUE)
ON CONFLICT (strategy_name) DO UPDATE SET is_enabled = TRUE;
```

Or enable in config.py (legacy fallback):
```python
MY_MOMENTUM_STRATEGY_ENABLED = True
```

### Step 5: Test

```bash
# Verify import works
docker exec -it task-worker python -c \
    "from forex_scanner.core.strategies.my_momentum_strategy import MyMomentumStrategy; print('OK')"

# Verify registry sees it
docker exec -it task-worker python -c "
from forex_scanner.core.strategies.strategy_registry import StrategyRegistry
registry = StrategyRegistry.get_instance()
print(f'Registered: {registry.get_registered_strategies()}')
"

# Run a scan
docker exec -it task-worker python /app/trade_scan.py scan
```

---

## Architecture Overview

### Strategy Registry Pattern

Strategies self-register on import using the `@register_strategy` decorator:

```python
from .strategy_registry import register_strategy, StrategyInterface

@register_strategy('MY_STRATEGY')
class MyStrategy(StrategyInterface):
    @property
    def strategy_name(self) -> str:
        return "MY_STRATEGY"

    def detect_signal(self, **kwargs) -> Optional[Dict]:
        # Your logic here
        pass

    def get_required_timeframes(self) -> List[str]:
        return ['4h', '1h', '15m']
```

### StrategyInterface Requirements

All strategies must implement:

| Method | Description |
|--------|-------------|
| `strategy_name` (property) | Unique identifier (e.g., "MY_MOMENTUM") |
| `detect_signal(**kwargs)` | Main detection logic, returns signal dict or None |
| `get_required_timeframes()` | List of timeframes needed (e.g., ['4h', '15m']) |
| `reset_cooldowns()` (optional) | Reset internal cooldowns for backtesting |
| `flush_rejections()` (optional) | Flush rejection logs to database |

### Signal Dictionary Format

The `detect_signal()` method should return a dict with these fields:

```python
signal = {
    # Required
    'signal': 'BUY',              # 'BUY' or 'SELL'
    'signal_type': 'buy',         # lowercase
    'strategy': 'MY_MOMENTUM',    # Strategy name
    'epic': 'CS.D.EURUSD.CEEM.IP',
    'pair': 'EURUSD',
    'entry_price': 1.08500,
    'stop_loss_pips': 15.0,
    'take_profit_pips': 25.0,
    'confidence_score': 0.75,
    'signal_timestamp': '2026-01-03T10:30:00Z',

    # Optional but recommended
    'strategy_indicators': {
        'ema_50': 1.08450,
        'rsi': 55.2,
        'atr_pips': 8.5,
    },
    'entry_type': 'pullback',     # or 'momentum', 'breakout', etc.
    'htf_bias': 'bullish',
    'version': '1.0.0',
}
```

---

## Configuration System

### Database-Driven Configuration

All strategy settings should be stored in the `strategy_config` database:

```
strategy_config/
├── {strategy}_global_config     # Global parameters
├── {strategy}_pair_overrides    # Per-pair overrides
└── {strategy}_config_audit      # Change history (optional)
```

### Config Service Pattern

Use a singleton config service with caching:

```python
class MyMomentumConfigService:
    _instance = None
    _config = None
    _cache_ttl_seconds = 300  # 5 minutes

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_config(self) -> MyMomentumConfig:
        # Check cache, refresh if needed
        ...
```

### Per-Pair Overrides

Support per-pair customization:

```python
def get_pair_stop_loss(self, epic: str) -> float:
    """Get SL for pair (falls back to global default)"""
    pair_override = self._get_pair_override(epic)
    if pair_override and pair_override.fixed_stop_loss_pips:
        return pair_override.fixed_stop_loss_pips
    return self.config.fixed_stop_loss_pips
```

---

## Best Practices

### 1. Cooldown Management

Prevent signal spam with per-pair cooldowns:

```python
def _check_cooldown(self, epic: str) -> bool:
    if epic not in self._cooldowns:
        return True
    if datetime.now() >= self._cooldowns[epic]:
        del self._cooldowns[epic]
        return True
    return False

def _set_cooldown(self, epic: str) -> None:
    self._cooldowns[epic] = datetime.now() + timedelta(minutes=60)
```

### 2. Rejection Logging

Track why signals are rejected for analysis:

```python
def _log_rejection(self, epic: str, reason: str, value: Any = None):
    self._pending_rejections.append({
        'epic': epic,
        'strategy': self.strategy_name,
        'reason': reason,
        'value': value,
        'timestamp': datetime.now().isoformat()
    })
```

### 3. Confidence Scoring

Use multi-component confidence (each ~20% weight):

```python
def _calculate_confidence(self, ...) -> float:
    components = {
        'trend_alignment': 0.20 if htf_aligned else 0.0,
        'signal_quality': signal_score * 0.20,
        'entry_timing': entry_score * 0.20,
        'volume_confirmation': vol_score * 0.20,
        'risk_reward': rr_score * 0.20,
    }
    return sum(components.values())
```

### 4. ATR-Based Calculations

Use ATR for dynamic SL/TP:

```python
def _calculate_sl(self, df: pd.DataFrame, direction: str) -> float:
    atr = df['atr'].iloc[-1]
    base_sl = atr * self.config.sl_atr_multiplier

    # Apply pair-specific buffer
    buffer = self.get_pair_sl_buffer(epic)

    # Cap at reasonable maximum
    return min(base_sl + buffer, 30.0)  # Max 30 pips
```

---

## Testing Checklist

Before deploying a new strategy:

- [ ] Strategy imports without errors
- [ ] Registry shows strategy as registered
- [ ] Config loads from database correctly
- [ ] Per-pair overrides work
- [ ] Signal dict has all required fields
- [ ] Backtest produces reasonable results
- [ ] Cooldowns prevent signal spam
- [ ] Container restarts cleanly

```bash
# Full test sequence
docker restart task-worker
docker logs task-worker --tail 50  # Check for errors

docker exec -it task-worker python /app/trade_scan.py scan
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 7 MY_MOMENTUM
```

---

## Example: Adding RSI Divergence Strategy

### 1. Create strategy file

```python
# core/strategies/rsi_divergence_strategy.py

@register_strategy('RSI_DIVERGENCE')
class RSIDivergenceStrategy(StrategyInterface):
    def __init__(self, config=None, logger=None, db_manager=None):
        self.config = get_rsi_divergence_config()
        self._cooldowns = {}

    @property
    def strategy_name(self) -> str:
        return "RSI_DIVERGENCE"

    def get_required_timeframes(self) -> List[str]:
        return ['4h', '1h']

    def detect_signal(self, df_trigger=None, df_4h=None, epic="", **kwargs):
        # Check for bullish divergence
        if self._detect_bullish_divergence(df_trigger):
            if self._confirm_with_htf(df_4h):
                return self._build_signal(epic, 'BUY', ...)

        # Check for bearish divergence
        if self._detect_bearish_divergence(df_trigger):
            if self._confirm_with_htf(df_4h):
                return self._build_signal(epic, 'SELL', ...)

        return None
```

### 2. Create migration

```sql
-- migrations/create_rsi_divergence_config.sql
CREATE TABLE rsi_divergence_global_config (...);

INSERT INTO rsi_divergence_global_config (parameter_name, parameter_value, ...)
VALUES
    ('rsi_period', '14', 'int', ...),
    ('divergence_lookback', '20', 'int', ...),
    ('min_confidence', '0.65', 'float', ...);
```

### 3. Enable and test

```bash
docker exec postgres psql -U postgres -d strategy_config \
    -f /app/forex_scanner/migrations/create_rsi_divergence_config.sql

docker exec -it task-worker python /app/trade_scan.py scan
```

---

## Troubleshooting

### Strategy Not Detected

Check if it's registered:
```python
from forex_scanner.core.strategies.strategy_registry import StrategyRegistry
registry = StrategyRegistry.get_instance()
print(registry.get_registered_strategies())
```

### Config Not Loading

Check database connection:
```bash
docker exec postgres psql -U postgres -d strategy_config -c \
    "SELECT * FROM my_momentum_global_config LIMIT 5;"
```

### Signal Not Appearing

Enable debug logging:
```python
import logging
logging.getLogger('forex_scanner.core.strategies').setLevel(logging.DEBUG)
```

Check signal detector logs:
```bash
docker logs task-worker 2>&1 | grep "MY_MOMENTUM"
```
