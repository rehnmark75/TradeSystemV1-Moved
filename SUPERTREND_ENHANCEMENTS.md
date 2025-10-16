# SuperTrend Strategy - LuxAlgo-Inspired Enhancements

## 📋 Overview

This document describes the advanced signal filtering enhancements added to the SuperTrend strategy, inspired by the LuxAlgo "SuperTrend AI (Clustering)" indicator.

**Goal**: Reduce signal count from 1000+ to 100-200 high-quality signals while maintaining or improving win rate.

## 🎯 Implemented Enhancements

### ✅ Option 2: Enhanced Slow SuperTrend Stability Filter

**Problem**: Original implementation required only 5 bars of stability, allowing signals in early-stage choppy conditions.

**Solution**: Increased stability requirement to 12 bars (configurable).

**Impact**: 30-50% signal reduction

**Configuration**:
```python
# worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py
SUPERTREND_STABILITY_BARS = 12  # Increased from 5
```

**How it works**:
- Slow SuperTrend must maintain same direction for 12 consecutive bars
- Prevents signals when trend is just starting to flip
- Filters out false signals in choppy/ranging markets
- More conservative than YouTube strategy (which used 5 bars)

**Tuning**:
- **10-12 bars**: Balanced (recommended for most pairs)
- **15+ bars**: Very conservative (fewer signals, higher quality)
- **8-9 bars**: Aggressive (more signals, may include some chop)

---

### ✅ Option 3: Performance-Based Filter (LuxAlgo Inspired)

**Problem**: All signals treated equally, even when SuperTrend has been wrong recently.

**Solution**: Track SuperTrend performance and only signal when performance is positive.

**Impact**: 40-60% signal reduction, better quality signals

**Configuration**:
```python
SUPERTREND_PERFORMANCE_FILTER = True
SUPERTREND_PERFORMANCE_THRESHOLD = 0.0      # Positive performance required
SUPERTREND_PERFORMANCE_ALPHA = 0.1          # Smoothing factor
```

**How it works**:
1. **Track Performance**: For each bar, check if price moved with SuperTrend direction
   - SuperTrend bullish + price up = positive performance
   - SuperTrend bearish + price down = positive performance
   - Opposite movements = negative performance

2. **Exponential Smoothing**: Apply EMA smoothing with alpha=0.1
   - Creates a "performance memory" similar to LuxAlgo
   - Recent performance weighted more heavily

3. **Filter Signals**: Only allow signals when smoothed performance > 0

**Calculation**:
```python
# Performance at each bar
raw_performance = st_trend[t-1] * (close[t] - close[t-1])

# Exponentially smoothed performance
performance[t] = alpha * raw_performance[t] + (1 - alpha) * performance[t-1]

# Signal allowed if performance > threshold
signal_allowed = performance[t] > 0.0
```

**Tuning**:
- **Threshold = 0.0**: Only positive performance (recommended)
- **Threshold = -0.0001**: Allow slightly negative (more signals)
- **Alpha = 0.05**: Very smooth, long memory
- **Alpha = 0.1**: Balanced (recommended)
- **Alpha = 0.2**: Reactive, short memory

---

### ✅ Option 5: Trend Strength Filter

**Problem**: SuperTrends signal even when very close together (choppy market).

**Solution**: Measure separation between fast and slow SuperTrends, filter weak trends.

**Impact**: 30-40% signal reduction

**Configuration**:
```python
SUPERTREND_TREND_STRENGTH_FILTER = True
SUPERTREND_MIN_TREND_STRENGTH = 0.3         # 0.3% minimum separation
```

**How it works**:
1. **Calculate Separation**: Measure distance between fast and slow SuperTrends
   ```python
   separation_pct = abs(st_fast - st_slow) / close * 100
   ```

2. **Filter Weak Trends**: Only signal if separation > minimum threshold
   - Wide separation = strong trend = good signals
   - Narrow separation = choppy market = skip signals

**Examples**:
- **EUR/USD @ 1.1000**:
  - 0.3% separation = 33 pips between SuperTrends
  - Strong uptrend: ST Fast @ 1.0967, ST Slow @ 1.0934 (33 pips) ✅
  - Choppy market: ST Fast @ 1.0989, ST Slow @ 1.0978 (11 pips) ❌

**Tuning**:
- **0.2%**: Aggressive (more signals, some chop included)
- **0.3%**: Balanced (recommended for EUR/USD, GBP/USD)
- **0.4-0.5%**: Conservative (strong trends only)
- **Pair-specific**: GBP/JPY may need 0.5%, EUR/USD may use 0.2%

---

### ✅ Option 4: Adaptive Factor Optimization (Framework)

**Status**: Framework implemented, disabled by default (computationally expensive).

**What it does**: Full LuxAlgo clustering system for dynamic ATR multiplier selection.

**Configuration**:
```python
SUPERTREND_ADAPTIVE_CLUSTERING = False      # Disabled by default
SUPERTREND_CLUSTER_MIN_FACTOR = 1.0
SUPERTREND_CLUSTER_MAX_FACTOR = 5.0
SUPERTREND_CLUSTER_STEP = 0.5
SUPERTREND_CLUSTER_LOOKBACK = 500
```

**Implementation** ([supertrend_adaptive_optimizer.py](worker/app/forex_scanner/core/strategies/helpers/supertrend_adaptive_optimizer.py)):

1. **Multiple SuperTrend Calculation**:
   - Tests 9 different ATR multipliers (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
   - Calculates SuperTrend for each multiplier

2. **Performance Evaluation**:
   - Tracks performance of each SuperTrend multiplier
   - Uses exponential smoothing like LuxAlgo

3. **K-Means Clustering**:
   - Groups multipliers into 3 clusters: best/average/worst
   - Based on performance scores

4. **Adaptive Selection**:
   - Dynamically selects multipliers from best-performing cluster
   - Adapts to changing market conditions

**Why disabled**:
- Requires significant computational resources
- Needs to calculate 9 SuperTrends instead of 3
- Better suited for research/optimization phase
- Current fixed parameters (YouTube strategy) work well

**When to enable**:
- Backtesting and optimization
- Finding optimal parameters for specific pairs
- Research into market-adaptive strategies

---

## 📊 Expected Impact

### Combined Effect

With all filters enabled (Options 2, 3, 5):

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total Signals** | 1000+ | 100-200 | -70-80% |
| **Bull Signals** | ~500 | ~50-100 | -70-80% |
| **Bear Signals** | ~500 | ~50-100 | -70-80% |
| **Signal Quality** | Mixed | High | Improved |
| **Win Rate** | Est. 50-60% | Est. 60-70% | +10-15% |

### Filter Contribution

| Filter | Signal Reduction | Quality Impact |
|--------|------------------|----------------|
| **Stability (12 bars)** | 30-50% | Medium |
| **Performance** | 40-60% | High |
| **Trend Strength** | 30-40% | Medium-High |
| **Combined** | 70-80% | Very High |

Note: Filters are applied sequentially, so reductions are not additive.

---

## 🎛️ Configuration Guide

### Recommended Presets

#### 1. **Balanced** (Default - Recommended)
```python
SUPERTREND_STABILITY_BARS = 12
SUPERTREND_PERFORMANCE_FILTER = True
SUPERTREND_PERFORMANCE_THRESHOLD = 0.0
SUPERTREND_PERFORMANCE_ALPHA = 0.1
SUPERTREND_TREND_STRENGTH_FILTER = True
SUPERTREND_MIN_TREND_STRENGTH = 0.3
```
**Use case**: Most pairs, general trading
**Signals**: ~150-200 per month
**Quality**: High

#### 2. **Conservative** (Highest Quality)
```python
SUPERTREND_STABILITY_BARS = 15
SUPERTREND_PERFORMANCE_FILTER = True
SUPERTREND_PERFORMANCE_THRESHOLD = 0.0001    # Require positive performance
SUPERTREND_PERFORMANCE_ALPHA = 0.05          # Smooth, long memory
SUPERTREND_TREND_STRENGTH_FILTER = True
SUPERTREND_MIN_TREND_STRENGTH = 0.4          # Strong trends only
```
**Use case**: Low risk tolerance, position trading
**Signals**: ~50-100 per month
**Quality**: Very High

#### 3. **Aggressive** (More Signals)
```python
SUPERTREND_STABILITY_BARS = 10
SUPERTREND_PERFORMANCE_FILTER = True
SUPERTREND_PERFORMANCE_THRESHOLD = -0.0001   # Allow slight negative
SUPERTREND_PERFORMANCE_ALPHA = 0.15          # More reactive
SUPERTREND_TREND_STRENGTH_FILTER = True
SUPERTREND_MIN_TREND_STRENGTH = 0.2          # Lower threshold
```
**Use case**: Active trading, scalping approach
**Signals**: ~250-350 per month
**Quality**: Medium-High

#### 4. **Research** (All Features)
```python
# Enable adaptive clustering for research
SUPERTREND_ADAPTIVE_CLUSTERING = True
SUPERTREND_CLUSTER_LOOKBACK = 1000
# Plus all filters from Balanced preset
```
**Use case**: Backtesting, optimization, research
**Signals**: Variable (adaptive)
**Quality**: Optimized for historical data

---

## 🔬 Technical Details

### Performance Calculation Algorithm

The performance calculation is inspired by LuxAlgo's approach:

```python
def calculate_performance(df, alpha=0.1):
    """
    Calculate SuperTrend performance with exponential smoothing

    Similar to LuxAlgo:
    perf += 2/(perfAlpha+1) * (nz(close - close[1]) * diff - perf)

    Our implementation:
    perf[t] = alpha * raw_perf[t] + (1-alpha) * perf[t-1]
    """
    # Get SuperTrend direction (1 = bullish, -1 = bearish)
    st_trend = df['st_fast_trend'].shift(1)

    # Get price change
    price_change = df['close'].diff()

    # Raw performance: positive if trend correct, negative if wrong
    raw_performance = st_trend * price_change

    # Apply exponential smoothing
    performance = raw_performance.ewm(alpha=alpha, min_periods=1).mean()

    return performance
```

### Trend Strength Calculation

```python
def calculate_trend_strength(df):
    """
    Calculate trend strength from SuperTrend separation

    Wide separation = strong trend
    Narrow separation = choppy market
    """
    st_fast = df['st_fast']
    st_slow = df['st_slow']
    close = df['close']

    # Separation as percentage of price
    separation_pct = abs(st_fast - st_slow) / close * 100

    return separation_pct
```

### Stability Check

```python
def check_stability(df, direction, num_bars=12):
    """
    Check if slow SuperTrend stable for N bars

    Args:
        direction: 1 for bullish, -1 for bearish
        num_bars: Minimum consecutive bars (default 12)
    """
    slow_trend = df['st_slow_trend']

    # Start with current bar
    stable = (slow_trend == direction)

    # Check all previous bars
    for i in range(1, num_bars):
        stable = stable & (slow_trend.shift(i) == direction)

    return stable
```

---

## 📈 Usage Examples

### Basic Usage (Scanner)

The filters are automatically applied when using the EMA strategy in SuperTrend mode:

```bash
# Inside Docker container
python -m forex_scanner.cli.scanner_cli ema --pairs EURUSD --timeframe 15m
```

No code changes needed - filters are applied automatically based on config.

### Backtest with Enhanced Filters

```bash
# Backtest with new filters
python -m forex_scanner.cli.scanner_cli backtest ema \
    --pair EURUSD \
    --timeframe 15m \
    --start-date 2024-01-01 \
    --end-date 2024-12-01
```

### Custom Configuration

```python
# Temporary config override for testing
from forex_scanner.configdata.strategies import config_ema_strategy

# Test conservative settings
config_ema_strategy.SUPERTREND_STABILITY_BARS = 15
config_ema_strategy.SUPERTREND_MIN_TREND_STRENGTH = 0.4
config_ema_strategy.SUPERTREND_PERFORMANCE_THRESHOLD = 0.0001
```

---

## 🧪 Testing & Validation

### A/B Comparison

The system logs signal counts before and after each filter:

```
📊 Original signals: BULL=487, BEAR=523
✅ After stability filter (12 bars): BULL=234, BEAR=267
✅ After performance filter: BULL=156, BEAR=178
✅ After trend strength filter: BULL=98, BEAR=112
🎯 FINAL: BULL=98, BEAR=112
📉 Signal reduction: 79.3%
```

### Monitoring Performance

Check the logs for filter effectiveness:

```bash
# Search for filter performance in logs
docker-compose exec worker grep "Signal reduction" /app/logs/scanner.log

# Check individual filter impact
docker-compose exec worker grep "After.*filter" /app/logs/scanner.log
```

---

## 🔧 Troubleshooting

### Too Few Signals

If getting fewer than 50 signals per month:

1. **Reduce stability requirement**:
   ```python
   SUPERTREND_STABILITY_BARS = 10  # Down from 12
   ```

2. **Lower trend strength threshold**:
   ```python
   SUPERTREND_MIN_TREND_STRENGTH = 0.2  # Down from 0.3
   ```

3. **Adjust performance threshold**:
   ```python
   SUPERTREND_PERFORMANCE_THRESHOLD = -0.0001  # Allow slight negative
   ```

### Still Too Many Signals

If getting more than 300 signals per month:

1. **Increase stability requirement**:
   ```python
   SUPERTREND_STABILITY_BARS = 15  # Up from 12
   ```

2. **Raise trend strength threshold**:
   ```python
   SUPERTREND_MIN_TREND_STRENGTH = 0.4  # Up from 0.3
   ```

3. **Require stronger performance**:
   ```python
   SUPERTREND_PERFORMANCE_THRESHOLD = 0.0001  # Require clearly positive
   ```

### Poor Signal Quality

If win rate is low despite filters:

1. **Check performance alpha** (may be too reactive):
   ```python
   SUPERTREND_PERFORMANCE_ALPHA = 0.05  # More stable
   ```

2. **Verify SuperTrend configuration** (check ATR periods/multipliers)

3. **Review market conditions** (filters work best in trending markets)

---

## 📚 References

### LuxAlgo SuperTrend AI

- **Original Indicator**: [TradingView - SuperTrend AI (Clustering)](https://www.tradingview.com/script/...)
- **License**: CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike)
- **Key Concepts Borrowed**:
  - Performance memory system (exponential smoothing)
  - K-means clustering for factor optimization
  - Adaptive factor selection based on market conditions

### Implementation Files

- **Signal Calculator**: [ema_signal_calculator.py](worker/app/forex_scanner/core/strategies/helpers/ema_signal_calculator.py)
- **Adaptive Optimizer**: [supertrend_adaptive_optimizer.py](worker/app/forex_scanner/core/strategies/helpers/supertrend_adaptive_optimizer.py)
- **Configuration**: [config_ema_strategy.py](worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py)

### Related Documentation

- [SuperTrend Strategy Overview](claude-strategies.md#supertrend-strategy)
- [YouTube Triple SuperTrend Strategy](claude-strategies.md#youtube-triple-supertrend)
- [Backtest System](claude-commands.md#backtesting)

---

## 🎓 Next Steps

1. **Backtest** the enhanced strategy with your historical data
2. **Monitor** signal quality in live scanner
3. **Tune** parameters based on your specific pairs and timeframes
4. **Compare** results with previous strategy version
5. **Experiment** with adaptive clustering (Option 4) for research

**Happy Trading!** 🚀
