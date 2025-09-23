# Break-Even Distance Logic Explained

## User's Question
> "I thought I should follow the min distance allowed in the table but I just saw for AUDUSD where the table said 2 steps it moved directly to 6 pts"

## Answer Summary
**The system uses a hierarchy: IG's minimum distance OVERRIDES configuration table values for better trade management.**

## Detailed Analysis

### 🔄 How Break-Even Distance is Determined

When a trade reaches break-even trigger, the system follows this hierarchy:

1. **Primary**: Use IG's `min_stop_distance_points` (from broker API)
2. **Fallback**: Use configuration `stage1_lock_points` (from table)

### 📊 Evidence from Database

**AUDUSD Historical Data:**
```sql
-- Historical trades (when IG minimum was 6 points)
ID 269:  min_stop_distance_points = 6, moved 6 points to break-even
ID 1113: min_stop_distance_points = 6, but didn't reach break-even
ID 1107: min_stop_distance_points = 6, but didn't reach break-even

-- Recent trades (IG minimum now 2 points)
ID 1162: min_stop_distance_points = 2
ID 1157: min_stop_distance_points = 2
ID 1148: min_stop_distance_points = 2
```

### 🎯 Code Implementation

**Location:** `trailing_class.py:698-711` in `_calculate_stage1_trail()` method

```python
# ✅ ENHANCEMENT: Use IG's minimum stop distance for better trade evolution
ig_min_distance = getattr(trade, 'min_stop_distance_points', None)
if ig_min_distance:
    lock_points = max(1, round(ig_min_distance))  # IG minimum takes priority
    self.logger.info(f"🎯 [STAGE 1 IG MIN] Trade {trade.id}: Using IG minimum distance {lock_points}pts")
else:
    lock_points = self.config.stage1_lock_points    # Fallback to config
    self.logger.info(f"⚠️ [STAGE 1 FALLBACK] Trade {trade.id}: No IG minimum distance, using config {lock_points}pts")
```

### 📋 Configuration Table vs Reality

**What the "table" shows (BALANCED_PROGRESSIVE_CONFIG):**
- `stage1_lock_points = 2` for AUDUSD

**What actually happens:**
- If IG says minimum = 6 points → Trade moves 6 points
- If IG says minimum = 2 points → Trade moves 2 points
- If IG data unavailable → Falls back to table value (2 points)

### ✅ Test Results Verification

Our test script `test_audusd_breakeven_analysis.py` confirms:

```
📊 Test Case 1: Historical AUDUSD (IG minimum = 6 points)
Entry Price: 0.65365
Configuration says: 2 points
IG Minimum Distance: 6.0 points
Expected Stop: 0.65425 (IG minimum used)
Actual Stop: 0.65425
Distance moved: 6.0 points
✅ MATCHES USER'S OBSERVATION

📊 Test Case 2: Current AUDUSD (IG minimum = 2 points)
Entry Price: 0.66064
Configuration says: 2 points
IG Minimum Distance: 2.0 points
Expected Stop: 0.66084 (IG minimum used)
Actual Stop: 0.66084
Distance moved: 2.0 points
✅ MATCHES CONFIGURATION
```

### 🎯 Why This Design Makes Sense

1. **Broker Compliance**: IG's minimum distances are regulatory requirements
2. **Dynamic Adaptation**: IG can change minimum distances based on market conditions
3. **Better Trade Evolution**: Often gives trades more room to develop profitably
4. **Fallback Safety**: Configuration table provides backup when broker data unavailable

### 📅 Timeline of Changes

- **Historical**: IG required 6-point minimum for AUDUSD
- **Current**: IG requires 2-point minimum for AUDUSD
- **System**: Always uses IG's current requirement, not the static table value

## Conclusion

**The user's observation is CORRECT and EXPECTED behavior.**

- Table shows 2 points (fallback configuration)
- IG required 6 points minimum (at that time)
- System correctly used 6 points instead of 2
- This is working as designed for regulatory compliance and better trade management

### 🔍 How to Check Current Behavior

```sql
-- Check current AUDUSD IG minimum distance
SELECT id, symbol, min_stop_distance_points, timestamp
FROM trade_log
WHERE symbol LIKE '%AUDUSD%'
ORDER BY timestamp DESC LIMIT 1;
```

The "table" is a fallback configuration - the actual behavior is determined by IG's real-time minimum distance requirements.