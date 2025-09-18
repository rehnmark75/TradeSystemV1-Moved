# Migration to 1-Minute Base Data Plan

## 🎉 **Great Discovery!**

**IG Lightstreamer DOES support 1-minute chart streaming** using `"1MINUTE"` format!
The original plan is back on track.

## 🎯 **Current Status**

### Phase 1: Parallel Collection (READY TO DEPLOY)
- ✅ **Chart streamer updated** to collect both 5m and 1m data (`"1MINUTE"` format)
- ✅ **Backfill service updated** to handle 1m gaps
- ✅ **Synthesis functions added** for 1m → 5m validation
- 🚀 **Next**: Restart streaming services to begin 1m collection

### Alternative Approaches for 1m Data
1. **Periodic REST API fetches** - Pull 1m candles every few minutes via REST API
2. **Tick-level streaming** - Use IG's tick streaming and aggregate to 1m candles
3. **5m base approach** - Continue with current 5m streaming + synthesis (recommended)

## 📊 **Revised Strategy**

### Current Architecture (Recommended)
```
[IG Chart Stream] → [5m candles] → Database (timeframe=5)
                                 ↓
                    [Synthesize 15m, 60m on-demand]
```

### Alternative: REST API 1m Collection
```
[IG REST API] → [1m candles] → Database (timeframe=1)
              ↓ (periodic polling)
[Synthesize all timeframes from 1m base]
```

### Hybrid Approach
```
[IG Chart Stream] → [5m candles] → Database (real-time)
[IG REST API]     → [1m candles] → Database (periodic backfill)
```

## 🧪 **Validation Tests**

### Test 1: Data Volume
- Expect ~5x more 1m candles than 5m candles
- Monitor storage impact and streaming performance

### Test 2: Synthesis Accuracy
- Compare `synthesize_5m_from_1m()` output vs direct 5m candles
- Target: <0.1 pip difference in OHLC values
- Test across different market conditions (trending, ranging, volatile)

### Test 3: Performance Impact
- Measure synthesis speed for different timeframe combinations
- Ensure real-time trading performance is not degraded

## 📈 **Benefits of 1m Base Architecture**

### Immediate Benefits
- **Perfect consistency**: All timeframes mathematically derived from 1m base
- **Higher precision**: Better entry/exit timing for strategies
- **Flexibility**: Easy to add custom timeframes (2m, 3m, 10m, etc.)

### Long-term Benefits
- **Simplified codebase**: Single synthesis logic vs multiple streaming endpoints
- **Better backtesting**: Higher precision historical analysis
- **Scalability**: Easier to add new trading pairs (only need 1m stream)

## 🔧 **Implementation Details**

### Files Modified
- `stream-app/igstream/chart_streamer.py`: Added 1m timeframe collection
- `stream-app/igstream/auto_backfill.py`: Added 1m gap detection and mapping
- `streamlit/utils/helpers.py`: Added `synthesize_5m_from_1m()` function

### Database Schema
- **No changes required**: `ig_candles` table already supports timeframe=1
- **Storage impact**: Expect ~20% increase in total candle storage

### Monitoring Points
- Stream connection stability for 1m data
- Gap detection accuracy for 1m timeframe
- Synthesis performance metrics

## 🚨 **Rollback Plan**

If 1m collection causes issues:
1. **Immediate**: Comment out `1: "MINUTE"` from chart_streamer.py
2. **Data cleanup**: `DELETE FROM ig_candles WHERE timeframe = 1`
3. **Backfill**: Revert to `timeframes=[5]` in auto_backfill.py

## 📋 **Next Actions**

1. **Restart streaming services** to begin 1m data collection
2. **Monitor logs** for successful 1m candle creation
3. **Check database** after 24 hours for expected 1m data volume
4. **Begin validation** once sufficient 1m data collected

---
*Created: 2025-09-18*
*Purpose: Track migration from 5m to 1m base data architecture*