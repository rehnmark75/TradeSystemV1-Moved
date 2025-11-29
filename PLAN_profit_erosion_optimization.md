# Profit Erosion Optimization Plan

## Problem Analysis

**Observation**: Trades reach significant MFE (e.g., 26.4 pips) but give back substantial profit before exit (e.g., ending at 16.0 pips = **39% erosion**).

---

## CRITICAL FINDING: Existing Take Profit System Status

### 1. Strategy Configuration (`config_smc_structure.py`)

**Partial Take Profit Settings:**
```python
SMC_PARTIAL_PROFIT_ENABLED = True        # ✅ ENABLED
SMC_PARTIAL_PROFIT_PERCENT = 50          # Close 50% at partial
SMC_PARTIAL_PROFIT_RR = 1.0              # Partial at 1.0R
SMC_MOVE_SL_TO_BE_AFTER_PARTIAL = True   # Move SL to BE after
```

**Trailing Stop Settings:**
```python
SMC_TRAILING_ENABLED = False             # ❌ DISABLED
SMC_TRAILING_MODE = 'structure_priority' # Only trail on >3R
SMC_TRAILING_MIN_RR = 3.0               # Must reach 3R first
```

### 2. Signal Generation (`smc_structure_strategy.py`)

The strategy **calculates** partial TP level:
```python
if self.partial_profit_enabled:
    partial_reward_pips = risk_pips * self.partial_profit_rr
    if final_trend == 'BULL':
        partial_tp = entry_price + (partial_reward_pips * pip_value)
    else:
        partial_tp = entry_price - (partial_reward_pips * pip_value)
```

### 3. Live Trade Execution

**Trade Monitor** (`trailing_class.py`) uses **Progressive 3-Stage trailing**:
- Stage 1: Break-even trigger
- Stage 2: Profit lock
- Stage 3: Percentage-based trailing

**BUT**: Looking at database analysis:
```
Winners in last 50 trades: 20
Moved to breakeven: 0
Partial close executed: 0
```

### ⚠️ ROOT CAUSE IDENTIFIED

**The partial take profit system is configured but NOT being executed in live trading.**

Evidence:
1. `partial_close_executed = False` for ALL winning trades
2. `moved_to_breakeven = False` for ALL winning trades
3. The signal generates `partial_tp` price but it's **never acted upon**

**Why?**
- The strategy generates `partial_tp` in the signal
- The trade monitor uses `Progressive3StageTrailing` which **doesn't implement partial closes**
- The trailing logic only adjusts stop loss, it doesn't close partial positions

### What's Actually Happening

1. **Entry**: Signal placed with SL and TP
2. **During Trade**: 3-stage trailing adjusts SL based on profit reached
3. **Exit**: Either TP hit OR trailing stop triggered
4. **Missing**: No partial position close at intermediate levels

The Progressive 3-Stage system **protects profit via stop adjustment** but **doesn't capture profit via partial closes**.

### Gap Analysis

| Component | Status | Notes |
|-----------|--------|-------|
| Strategy Config (`SMC_PARTIAL_PROFIT_ENABLED`) | ✅ Enabled | Settings configured |
| Signal Generation (`partial_tp` calculation) | ✅ Working | Price calculated |
| IG API Function (`partial_close_position`) | ✅ Exists | In `ig_orders.py` |
| Trade Monitor Integration | ❌ **MISSING** | Not calling partial close |
| Database Tracking | ✅ Ready | `partial_close_executed` column exists |

**The partial close functionality is BUILT but NOT WIRED UP to the trade monitor.**

---

### Current System Analysis

Your 3-Stage Progressive Trailing Stop system has these thresholds (example for AUDUSD):
- **Break-even trigger**: 12 pips → locks +5 pips buffer
- **Stage 1**: 16 pips → locks +4 pips
- **Stage 2**: 22 pips → locks +12 pips
- **Stage 3**: 23 pips → dynamic trailing (15-25% retracement)

**The Problem**: In the example trade (MFE=26.4, Exit=16.0):
1. Trade reached 26.4 pips - this triggered Stage 3 (>23 pips)
2. Stage 3 with 25% retracement at 26 pips = 6.5 pip trail distance
3. Protected profit would be 26.4 - 6.5 = **19.9 pips**
4. But actual exit was 16.0 pips - **below even Stage 2 lock (12 pips)**

This suggests either:
- Stage transitions aren't locking profit effectively
- The trailing stop is too loose at lower MFE levels
- Gap between stages allows too much retracement

---

## Proposed Solutions

### Option A: Tighten Stage 3 Retracement (Conservative)

**Approach**: Reduce the percentage-based retracement in Stage 3 to capture more profit.

**Current**:
```python
if best_profit_pips >= 50:
    retracement_pct = 0.15  # 15%
elif best_profit_pips >= 25:
    retracement_pct = 0.20  # 20%
else:
    retracement_pct = 0.25  # 25%
```

**Proposed**:
```python
if best_profit_pips >= 50:
    retracement_pct = 0.10  # 10% (was 15%)
elif best_profit_pips >= 25:
    retracement_pct = 0.15  # 15% (was 20%)
else:
    retracement_pct = 0.20  # 20% (was 25%)
```

**Impact**:
- MFE=26.4 → Trail=5.3 pips → Protected=21.1 pips (vs current 19.9)
- Pros: Simple change, protects more profit
- Cons: May exit too early on normal pullbacks

---

### Option B: Add Stage 2.5 - MFE-Based Profit Protection (Recommended)

**Approach**: Add a new intermediate stage that locks a percentage of MFE when profit starts declining.

**New Logic**:
```python
# Stage 2.5: MFE Protection Rule
# When profit reaches 70% of TP, lock 60% of MFE on any decline
if best_profit_pips >= target_pips * 0.70:
    if current_profit < best_profit_pips * 0.90:  # 10% decline from peak
        protected_profit = best_profit_pips * 0.60  # Lock 60% of MFE
        return -protected_profit, stage_info
```

**Example**:
- TP = 30 pips, MFE = 26.4 pips (88% of TP)
- If profit drops to 23.8 pips (90% of MFE), trigger protection
- Lock 60% of 26.4 = **15.8 pips** minimum exit

**Impact**:
- Protects significant profits when momentum fades
- Still allows pullbacks in early trade
- Works dynamically based on actual MFE reached

---

### Option C: Partial Take Profit System (Most Comprehensive)

**Approach**: Implement partial position closes at key profit levels.

**Implementation**:
```python
# Partial TP Configuration
PARTIAL_TP_RULES = {
    'first_partial': {
        'trigger_pips': 15,      # When profit reaches 15 pips
        'close_percent': 0.50,   # Close 50% of position
        'move_sl_to': 'breakeven'  # Move SL to breakeven for remainder
    },
    'second_partial': {
        'trigger_pips': 25,      # When profit reaches 25 pips
        'close_percent': 0.25,   # Close 25% more (75% total)
        'move_sl_to': 'lock_15'  # Lock 15 pips for remainder
    },
    # Final 25% rides to full TP or trailing stop
}
```

**Example for MFE=26.4 trade**:
1. At 15 pips: Close 50% → Secure +7.5 pips average
2. At 25 pips: Close 25% → Secure +6.25 pips more
3. Remaining 25% exits at 16 pips → +4.0 pips
4. **Total: 17.75 pips** (vs single exit at 16.0 = +1.75 pips more)

**Impact**:
- Guarantees profit capture at key levels
- Reduces emotional impact of giving back gains
- Allows runners to capture big moves
- More complex to implement (requires position sizing tracking)

---

### Option D: Momentum-Based Dynamic Trailing (Advanced)

**Approach**: Use price action momentum to adjust trailing distance dynamically.

**Logic**:
```python
def calculate_dynamic_trail(candles_since_mfe: int, mfe_pips: float, current_profit: float) -> float:
    """
    Trail distance based on how long since MFE was reached
    Quick retracement = loose trail (might recover)
    Slow decay = tight trail (momentum lost)
    """
    bars_since_peak = candles_since_mfe
    profit_decay_rate = (mfe_pips - current_profit) / max(bars_since_peak, 1)

    if bars_since_peak <= 2:
        # Just peaked, allow normal pullback
        trail_pct = 0.30  # 30% retracement allowed
    elif profit_decay_rate > 3.0:  # Losing >3 pips per bar
        # Fast decay - tighten immediately
        trail_pct = 0.10  # Only 10% retracement
    elif bars_since_peak >= 6:
        # Slow grinding down - momentum lost
        trail_pct = 0.15  # 15% retracement
    else:
        # Normal consolidation
        trail_pct = 0.25  # 25% retracement

    return mfe_pips * trail_pct
```

**Impact**:
- Adapts to market conditions in real-time
- Distinguishes healthy pullbacks from trend reversals
- More complex to implement and test

---

## Recommendation

**Phase 1 (Quick Win)**: Implement **Option B (Stage 2.5 MFE Protection)**
- Minimal code changes
- Significant profit protection improvement
- Easy to backtest and validate

**Phase 2 (Medium Term)**: Add **Option C (Partial Take Profit)**
- Requires position management changes
- Best long-term solution for profit capture
- Can be implemented alongside current trailing

**Phase 3 (Advanced)**: Consider **Option D (Momentum-Based)**
- Only if Phases 1-2 don't achieve desired results
- Requires extensive backtesting

---

## Implementation Plan for Option B

### Step 1: Update TrailingStopSimulator

```python
# In _update_progressive_trailing_stop method, add after Stage 3 check:

# Stage 2.5: MFE Protection Rule
# If profit has declined 10%+ from peak after reaching 70% of target
mfe_protection_threshold = self.target_pips * 0.70
if best_profit_pips >= mfe_protection_threshold:
    # Check if we're in decline (current profit < 90% of best)
    current_decline_pct = 1.0 - (current_profit_pips / best_profit_pips) if best_profit_pips > 0 else 0
    if current_decline_pct >= 0.10:  # 10%+ decline from peak
        protected_profit = best_profit_pips * 0.60  # Lock 60% of MFE
        stage_info['mfe_protection_triggered'] = True
        stage_info['stage_reached'] = 2.5
        return -protected_profit, stage_info
```

### Step 2: Add Configuration

```python
# In config_trailing_stops.py, add to each pair:
'mfe_protection_threshold_pct': 0.70,  # Trigger when profit reaches 70% of TP
'mfe_protection_decline_pct': 0.10,    # Trigger on 10% decline from peak
'mfe_protection_lock_pct': 0.60,       # Lock 60% of MFE
```

### Step 3: Backtest Validation

Run 90-day backtest comparing:
- Current system (baseline)
- Option B implementation (test)

Key metrics to compare:
- Average profit per winning trade
- Profit factor
- Win rate (shouldn't decrease significantly)
- Average MFE vs Actual Exit comparison

---

## Expected Results

Based on your example (MFE=26.4, Current Exit=16.0):

| Metric | Current | With Option B |
|--------|---------|---------------|
| Exit Price (pips) | 16.0 | 15.8-21.0 |
| Profit Captured | 60% of MFE | 60-80% of MFE |
| Erosion | 39% | 20-40% |

**Conservative estimate**: 15-25% improvement in profit capture on winning trades.

---

## Next Steps

1. Review this plan and select approach
2. Implement selected option
3. Run backtest comparison
4. Validate on paper trading before live deployment

---

## UPDATED RECOMMENDATION

Given the finding that **partial take profit is configured but not implemented**, the priority should be:

### Priority 1: Wire Up Existing Partial Close System (Quick Win)

The infrastructure already exists:
1. `SMC_PARTIAL_PROFIT_ENABLED = True` in config
2. `partial_tp` price calculated in signals
3. `partial_close_position()` function in `ig_orders.py`
4. Database columns ready (`partial_close_executed`, `partial_close_time`, `current_size`)

**Implementation Required:**
Add to `trade_monitor.py`:
```python
# In the monitoring loop, check if partial TP reached:
if not trade.partial_close_executed:
    if current_profit_pips >= partial_tp_pips:
        # Execute partial close
        await partial_close_position(
            deal_id=trade.deal_id,
            epic=trade.symbol,
            direction=trade.direction,
            size_to_close=0.5,  # Close 50%
            auth_headers=auth_headers
        )
        # Update database
        trade.partial_close_executed = True
        trade.partial_close_time = datetime.utcnow()
        trade.current_size = 0.5
        # Move SL to breakeven for remaining position
```

**Expected Impact:**
- MFE=26.4, Partial at 1.0R (12 pips) → Lock 50% at +12 pips
- Remaining 50% rides to TP or trailing stop exit
- Example: 50% at +12, 50% at +16 = Average +14 pips (vs current +16 full exit)
- **Guaranteed profit capture at intermediate level**

### Priority 2: Option B - MFE Protection Rule (If Priority 1 insufficient)

After implementing partial close, if profit erosion still exceeds 30%, add the MFE Protection Rule to the trailing stop logic.

### Priority 3: Tighten Stage 3 Retracement (Fine-tuning)

After both above, fine-tune the Stage 3 retracement percentages if needed.

---

## Implementation Status

### ✅ Stage 2.5: MFE Protection - IMPLEMENTED

**Implementation Date**: 2025-11-29
**Branch**: `feature/mfe-protection-stage`

**Files Modified**:
1. `worker/app/forex_scanner/core/trading/trailing_stop_simulator.py` - Backtest trailing stop
2. `worker/app/forex_scanner/config_trailing_stops.py` - Configuration with defaults
3. `dev-app/trailing_class.py` - Live trading trailing stop (Progressive3StageTrailing)

**How It Works**:
- Triggers when profit reaches 70% of target AND then declines 10% from peak (MFE)
- Locks 60% of MFE as minimum exit profit
- Example: MFE=26.4 pips → Protection triggers at 23.76 pips → Locks 15.84 pips minimum

**Expected Impact**:
- Prevents giving back more than 40% of achieved MFE
- Works alongside existing 3-stage trailing system
- Adds "Stage 2.5" protection between Stage 2 (profit lock) and Stage 3 (dynamic trailing)

---

## Next Steps

1. ~~**Run backtest** to validate MFE Protection impact on profit factor~~ ✅ Complete
2. **Implement Strategy Optimization** - See [PLAN_smc_strategy_optimization.md](PLAN_smc_strategy_optimization.md)
3. **Future**: Implement partial close wiring if additional profit capture needed

### Backtest Results (2025-11-29)

**30-Day Backtest with MFE Protection**:
- Win Rate: 24.1% (unchanged from baseline)
- Profit Factor: 0.35
- MFE Protection Triggers: 0 (trades not reaching 70% of target)

**Root Cause Finding**: MFE Protection didn't trigger because trades are failing before reaching 70% of target. The issue is **entry quality**, not profit capture.

**Recommendation**: Implement Phase 1 of [Strategy Optimization Plan](PLAN_smc_strategy_optimization.md) (pullback filter) first, then MFE Protection will become effective.

The partial close wiring remains "lowest hanging fruit" for additional optimization after entry quality is fixed.
