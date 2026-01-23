# Dynamic Trailing Stop System - Implementation Plan

**Status**: Ready for Implementation
**Priority**: High
**Created**: January 22, 2026
**Estimated Implementation**: 4-5 weeks (phased rollout)

---

## Executive Summary

Create a **market-adaptive trailing stop system** that adjusts stop placement based on real-time market conditions (volatility, session, regime) while maintaining a **guaranteed profit lock** for capital protection.

### Key Features

1. **🛡️ Guaranteed Profit Lock** ✅ **IMPLEMENTED**
   - Any trade reaching **+10 pips** must move SL to **entry + 1 pip** minimum
   - Ensures profitable trades NEVER go into loss
   - Applies to BOTH fixed and dynamic systems

2. **⚡ Scalp-Optimized Multipliers** (Tighter than regular trades)
   - Low volatility: **0.6x** vs 0.7x regular
   - Normal volatility: **0.8x** vs 1.0x regular
   - High volatility: **1.0x** vs 1.3x regular
   - Extreme volatility: **1.2x** vs 1.6x regular

3. **📊 Volatility-Adaptive Stages**
   - Base pips × volatility factor × session factor
   - ATR bounds validation (0.8× - 4.0× ATR)
   - Per-pair base configurations

4. **🔄 Easy Toggle**
   - `ENABLE_DYNAMIC_TRAILING = False` (start disabled)
   - Runs alongside existing fixed system
   - Per-pair override capability

---

## Current State Analysis

### ✅ What We Have

| Feature | Status | Location |
|---------|--------|----------|
| ATR Calculation | ✅ Done | `alert_history.atr` column |
| Market Intelligence | ✅ Done | `market_intelligence.py`, volatility regimes |
| Session Tracking | ✅ Done | `alert_history.market_session` |
| Guaranteed Profit Lock | ✅ **IMPLEMENTED** | `trailing_class.py:1504` |
| Per-Pair Scalp SL/TP | ✅ **IMPLEMENTED** | Database + strategy v2.30.0 |
| Progressive Trailing | ✅ Done | 3-stage system in `trailing_class.py` |

### ❌ What We Need

| Feature | Status | Priority |
|---------|--------|----------|
| Dynamic Stage Calculator | ❌ Not Built | **HIGH** |
| Market Conditions Storage | ❌ Not Built | HIGH |
| Config Service Integration | ❌ Not Built | HIGH |
| Backtest Replay | ❌ Not Built | MEDIUM |
| Dashboard Analytics | ❌ Not Built | LOW |

---

## Design: Volatility-Adjusted Dynamic Stages

### Formula

```python
adjusted_stage = base_pips × volatility_factor × session_factor

# With ATR bounds validation:
final_stage = max(atr × 0.8, min(adjusted_stage, atr × 4.0))
```

### Volatility Multipliers

**Regular Trades:**
| Volatility | Factor | Example (15 pip base) |
|------------|--------|----------------------|
| Low | 0.7 | 10.5 pips |
| Normal | 1.0 | 15.0 pips |
| High | 1.3 | 19.5 pips |
| Extreme | 1.6 | 24.0 pips |

**Scalp Trades (Tighter):**
| Volatility | Factor | Example (10 pip base) |
|------------|--------|----------------------|
| Low | 0.6 | 6.0 pips |
| Normal | 0.8 | 8.0 pips |
| High | 1.0 | 10.0 pips |
| Extreme | 1.2 | 12.0 pips |

### Session Multipliers (Optional)

| Session | Factor | Notes |
|---------|--------|-------|
| Asian | 0.9 | Lower volatility → tighter stops |
| London | 1.1 | Higher volatility → wider stops |
| New York | 1.1 | Higher volatility → wider stops |
| Overlap | 1.2 | Highest volatility → widest stops |

---

## Implementation Phases

### **Phase 1: Database & Infrastructure** (Week 1)

#### 1.1 Extend TradeLog Model
**File**: `dev-app/services/models.py`

```python
# Market conditions at entry
atr_at_entry = Column(Float, nullable=True)
volatility_regime = Column(String(20), nullable=True)  # 'low', 'normal', 'high', 'extreme'
market_regime_at_entry = Column(String(50), nullable=True)  # 'trending', 'ranging', 'breakout'
session_at_entry = Column(String(20), nullable=True)  # 'asian', 'london', 'new_york', 'overlap'

# Calculated dynamic stages (stored as JSON)
calculated_stages = Column(JSON, nullable=True)
trailing_calculation_method = Column(String(50), nullable=True)  # 'fixed', 'dynamic_volatility'
```

**Migration SQL:**
```sql
ALTER TABLE trade_log ADD COLUMN atr_at_entry FLOAT;
ALTER TABLE trade_log ADD COLUMN volatility_regime VARCHAR(20);
ALTER TABLE trade_log ADD COLUMN market_regime_at_entry VARCHAR(50);
ALTER TABLE trade_log ADD COLUMN session_at_entry VARCHAR(20);
ALTER TABLE trade_log ADD COLUMN calculated_stages JSON;
ALTER TABLE trade_log ADD COLUMN trailing_calculation_method VARCHAR(50);
```

#### 1.2 Create DynamicTrailingCalculator Service
**New File**: `dev-app/services/dynamic_trailing_calculator.py`

Key classes:
- `MarketConditions` - ATR, volatility regime, session, epic
- `CalculatedStages` - Dynamically computed stage values
- `DynamicTrailingCalculator` - Core calculation logic

**Key Method:**
```python
def calculate_stages(
    market_conditions: MarketConditions,
    base_config: Dict,
    is_scalp_trade: bool = False
) -> CalculatedStages
```

---

### **Phase 2: Trade Creation Integration** (Week 2)

#### 2.1 Update orders_router.py
**File**: `dev-app/routers/orders_router.py` (around line 650-680)

```python
# After creating trade_log, before db.add()

if alert_id and ENABLE_DYNAMIC_TRAILING:
    try:
        # Query alert_history for market conditions
        alert = db.query(AlertHistory).filter(AlertHistory.id == alert_id).first()

        if alert and alert.atr:
            market_conditions = MarketConditions(
                atr=alert.atr,
                volatility_regime=alert.volatility_regime or 'normal',
                market_regime=alert.market_regime or 'unknown',
                session=alert.market_session or 'unknown',
                epic=symbol
            )

            # Calculate dynamic stages
            calculator = DynamicTrailingCalculator()
            calculated = calculator.calculate_stages(
                market_conditions, base_config, is_scalp
            )

            # Store in trade_log
            trade_log.atr_at_entry = market_conditions.atr
            trade_log.volatility_regime = market_conditions.volatility_regime
            trade_log.calculated_stages = {
                'early_be_trigger': calculated.early_be_trigger,
                'stage1_trigger': calculated.stage1_trigger,
                'stage2_trigger': calculated.stage2_trigger,
                ...
            }
            trade_log.trailing_calculation_method = 'dynamic_volatility'
```

---

### **Phase 3: Trailing Execution** (Week 2-3)

#### 3.1 Update enhanced_trade_processor.py
**File**: `dev-app/enhanced_trade_processor.py`

```python
def get_config_for_trade(self, trade: TradeLog) -> TrailingConfig:
    """
    Priority:
    1. Dynamic (if enabled AND calculated_stages exist)
    2. Fixed pair-specific configs
    """
    if (ENABLE_DYNAMIC_TRAILING and
        hasattr(trade, 'calculated_stages') and
        trade.calculated_stages):

        # Use dynamically calculated stages
        calc = trade.calculated_stages
        return TrailingConfig(
            early_breakeven_trigger_points=calc['early_be_trigger'],
            stage1_trigger_points=calc['stage1_trigger'],
            ...
        )

    # Fall back to fixed config
    return TrailingConfig.from_epic(trade.symbol, is_scalp)
```

---

### **Phase 4: Configuration & Feature Flags** (Week 1)

**File**: `dev-app/config.py`

```python
# ================== DYNAMIC TRAILING CONFIGURATION ==================
# Master toggle: Switch between fixed and dynamic systems
ENABLE_DYNAMIC_TRAILING = False  # Start disabled, enable after backtesting

# Enable session-based adjustment
ENABLE_SESSION_ADJUSTMENT = True

# Volatility multipliers - REGULAR TRADES
VOLATILITY_MULTIPLIERS = {
    'low': 0.7,
    'normal': 1.0,
    'high': 1.3,
    'extreme': 1.6
}

# Volatility multipliers - SCALP TRADES (tighter)
SCALP_VOLATILITY_MULTIPLIERS = {
    'low': 0.6,
    'normal': 0.8,
    'high': 1.0,
    'extreme': 1.2
}

# Session multipliers
SESSION_MULTIPLIERS = {
    'asian': 0.9,
    'london': 1.1,
    'new_york': 1.1,
    'overlap': 1.2
}

# ATR bounds
DYNAMIC_TRAILING_ATR_MIN_MULTIPLIER = 0.8
DYNAMIC_TRAILING_ATR_MAX_MULTIPLIER = 4.0

# Per-pair override (force specific pairs to use fixed)
FORCE_FIXED_TRAILING_PAIRS = []
```

---

### **Phase 5: Backtesting Support** (Week 3-4)

#### 5.1 Ensure Backtest Captures Market Conditions
**File**: `worker/app/forex_scanner/bt.py`

- Store ATR, volatility_regime, market_regime, session with each signal
- Call DynamicTrailingCalculator with historical conditions during backtest trade creation
- Generate comparison report: dynamic vs fixed performance

---

### **Phase 6: Testing & Validation** (Week 4-5)

#### 6.1 Unit Tests
**File**: `tests/test_dynamic_trailing_calculator.py`

```python
def test_low_volatility_adjustment():
    """Low volatility should result in tighter stops"""
    conditions = MarketConditions(atr=8.0, volatility_regime='low', ...)
    result = calculator.calculate_stages(conditions, {'early_be_trigger': 15})

    # 15 × 0.7 (volatility) × 0.9 (session) = 9.45
    # But must be >= 0.8 × ATR = 6.4
    assert result.early_be_trigger < 15  # Tighter
    assert result.early_be_trigger >= 6.4  # Respects ATR minimum

def test_scalp_mode_uses_tighter_multipliers():
    """Scalp trades should use 0.6-1.2x instead of 0.7-1.6x"""
    conditions = MarketConditions(atr=10.0, volatility_regime='normal', ...)

    regular_result = calculator.calculate_stages(conditions, base_config, is_scalp=False)
    scalp_result = calculator.calculate_stages(conditions, base_config, is_scalp=True)

    # Scalp should be tighter
    assert scalp_result.stage1_trigger < regular_result.stage1_trigger
```

#### 6.2 Historical Backtesting (30-90 days)

Compare performance metrics:
- **Baseline**: Fixed trailing (current system)
- **Test 1**: Fixed + Guaranteed Profit Lock (isolate benefit)
- **Test 2**: Dynamic + Guaranteed Profit Lock
- **Variants**: Different multiplier combinations

**Key Metrics:**
- Win rate by volatility regime
- Profit capture percentage
- Premature stop-out reduction
- Expectancy improvement (scalp vs regular)
- Guaranteed profit lock validation (trades at +10 pips should never lose)

---

## Rollout Strategy

### **Step 1: Enable Guaranteed Profit Lock ONLY** ✅ **DONE**

- `ENABLE_GUARANTEED_PROFIT_LOCK = True`
- `ENABLE_DYNAMIC_TRAILING = False`
- Test for 3-5 days on ALL pairs
- **Verify**: Zero trades go into loss after hitting +10 pips

### **Step 2: Infrastructure Deployment** (Week 5)

- Deploy database migration
- Deploy DynamicTrailingCalculator service
- Deploy with `ENABLE_DYNAMIC_TRAILING = False` initially
- Verify no breaking changes

### **Step 3: Controlled Rollout** (Week 6+)

**Day 1-3: Enable for EURUSD Only**
```python
ENABLE_DYNAMIC_TRAILING = True
FORCE_FIXED_TRAILING_PAIRS = [
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    # ... all except EURUSD
]
```

**Monitor:**
- Verify calculated_stages populated correctly
- Check trailing stops execute at expected levels
- Compare live metrics vs backtest predictions

**Day 4-7: Expand to 3 Pairs**
- Add GBPUSD, USDJPY if EURUSD shows improvement

**Week 2+: Full Rollout**
- Enable for all pairs if 10%+ expectancy improvement observed

---

## Risk Mitigation

### 1. Fallback Mechanisms

- No ATR data → use fixed config
- Calculation fails → use fixed config
- Unreasonable values → cap at ATR bounds
- Existing trades without stages → use fixed config

### 2. Monitoring & Alerts

- Log every dynamic calculation (inputs + outputs)
- Alert if stages outside expected ranges
- Dashboard: dynamic vs fixed performance comparison
- Weekly report on calculation method distribution

### 3. Kill Switch

- `ENABLE_DYNAMIC_TRAILING = False` → instant revert
- Per-pair override via `FORCE_FIXED_TRAILING_PAIRS`
- Database flag to disable for specific trades

---

## Expected Outcomes

### Benefits

1. **Better Capital Preservation**: Tighter stops in low volatility
2. **Improved Profit Capture**: Wider stops in high volatility let winners run
3. **Session Optimization**: Asian session gets tighter stops
4. **Market-Adaptive**: Automatically adjusts to changing conditions
5. **Data-Driven**: Uses actual ATR instead of arbitrary pip values

### Success Criteria

**Primary Goals:**
1. ✅ **Zero trades go into loss after +10 pips** (profit lock validates)
2. **10%+ improvement in expectancy** vs fixed trailing for scalps
3. **15%+ reduction in premature stop-outs** during high volatility
4. **20%+ increase in profit capture** during trending markets
5. **No increase in losing trade frequency**

**Scalping-Specific:**
- Profit lock speed: Average time to +10 pip protection < 5 minutes
- Protected trade count: >40% of scalp trades reach profit lock
- Average scalp profit: +15% with tighter low-volatility stops

---

## Critical Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `dev-app/services/models.py` | TradeLog extensions | ❌ TODO |
| `dev-app/services/dynamic_trailing_calculator.py` | **NEW** - Core logic | ❌ TODO |
| `dev-app/routers/orders_router.py` | Market data extraction | ❌ TODO |
| `dev-app/enhanced_trade_processor.py` | Load calculated stages | ❌ TODO |
| `dev-app/config.py` | Feature flags & multipliers | ✅ Profit lock done |
| `dev-app/trailing_class.py` | Guaranteed profit lock | ✅ **IMPLEMENTED** |
| `worker/app/forex_scanner/bt.py` | Backtest support | ❌ TODO |

---

## Alternative Future Enhancements

### Machine Learning-Based Calculation

- Train ML model on historical trades
- Predict optimal stage values based on features
- Requires significant data and validation

### Regime-Specific Strategies

- Trending: Wider stages, let winners run
- Ranging: Tighter stages, quick profit taking
- Breakout: Very wide initial, aggressive trail after confirmation

### Real-Time Volatility Adjustment

- Re-calculate stages if volatility regime changes dramatically during trade
- Example: Trade entered in "normal", market shifts to "extreme"

---

## Next Steps (Immediate Actions)

1. ✅ **Guaranteed Profit Lock** - IMPLEMENTED
2. ✅ **Per-Pair Scalp SL/TP** - IMPLEMENTED
3. ❌ **Database Migration** - Add market condition fields to trade_log
4. ❌ **Create DynamicTrailingCalculator** - Build the core calculation service
5. ❌ **Historical Backtest** - Validate with 30-90 days of data
6. ❌ **Controlled Rollout** - Start with 1-2 pairs

---

**Document Version**: 1.0
**Last Updated**: January 22, 2026
**Status**: Ready for Phase 1 Implementation
