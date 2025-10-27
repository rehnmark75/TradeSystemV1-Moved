# MACD Simple Crossover Strategy

## 🎯 Strategy Overview

A clean, straightforward MACD crossover strategy with H4 trend filter for high-quality trade entries on 1-hour timeframe.

### Core Philosophy
**Keep it Simple**: H4 trend validation + MACD crossover = signal. No complex confluence, patterns, or Fibonacci calculations.

### Your Trading Pairs (9 pairs)
- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD
- EURJPY, AUDJPY

---

## 📊 Strategy Components

### 1. H4 MACD Trend Filter (Higher Timeframe Validation)
**Purpose**: Only trade with higher timeframe momentum

**How it works**:
- Calculates MACD (12, 26, 9) on H4 timeframe
- Identifies trend direction: Bullish (MACD > Signal) or Bearish (MACD < Signal)
- Requires histogram to be expanding (building momentum)
- **BULL signals** require H4 bullish trend
- **BEAR signals** require H4 bearish trend

**Why it's important**: Prevents counter-trend trades that fight the bigger picture.

---

### 2. MACD Crossover Detection (Entry Trigger)
**Purpose**: Identify precise entry point on strategy timeframe

**How it works**:
- **Bullish Crossover**: MACD line crosses above Signal line
  - Previous bar: MACD ≤ Signal
  - Current bar: MACD > Signal

- **Bearish Crossover**: MACD line crosses below Signal line
  - Previous bar: MACD ≥ Signal
  - Current bar: MACD < Signal

**Why crossovers**: Clear, objective entry signal when momentum shifts.

---

### 3. Trend Alignment Validation
**Purpose**: Only take signals in direction of H4 trend

**Rules**:
- BULL signal + H4 bearish trend = ❌ REJECTED
- BEAR signal + H4 bullish trend = ❌ REJECTED
- BULL signal + H4 neutral trend = ❌ REJECTED
- BEAR signal + H4 neutral trend = ❌ REJECTED
- BULL signal + H4 bullish trend = ✅ VALID
- BEAR signal + H4 bearish trend = ✅ VALID

---

### 4. ATR-Based Stop Loss & Take Profit
**Purpose**: Dynamic position sizing based on market volatility

**Stop Loss** (1.5x ATR):
- **BULL**: Entry - (1.5 × ATR)
- **BEAR**: Entry + (1.5 × ATR)
- **Min**: 10 pips (spread + slippage buffer)
- **Max**: 30 pips (risk control)

**Take Profit** (3.0x ATR):
- **BULL**: Entry + (3.0 × ATR)
- **BEAR**: Entry - (3.0 × ATR)
- **Target R:R**: 2:1 minimum

**Why ATR**: Adjusts to market conditions - tighter stops in calm markets, wider in volatile markets.

---

### 5. Simple Confidence Scoring
**Purpose**: Rank signal quality for position sizing

**Base Confidence**: 60% (simpler strategy = higher base trust)

**Bonuses**:
- **+10%** if H4 histogram expanding (strong momentum)
- **+10%** if H4 histogram magnitude > 0.0001 (strong trend)

**Maximum**: 80% confidence

**Confidence Levels**:
- **60%**: Basic signal (H4 trend valid, MACD crossover)
- **70%**: Good signal (+ expanding histogram OR strong histogram)
- **80%**: Excellent signal (+ both expanding AND strong histogram)

---

## 🔄 Complete Entry Process (Step-by-Step)

### Example: BULL Signal on NZDUSD

**Step 1: H4 MACD Trend Check**
```
H4 MACD line: 0.000296 (above Signal line: 0.000294)
H4 Histogram: 0.000115 (positive, expanding)
Result: H4 trend = BULLISH ✅
Signal direction allowed: BULL
```

**Step 2: Detect MACD Crossover on 1H**
```
Previous 1H bar: MACD (0.000290) ≤ Signal (0.000292)
Current 1H bar: MACD (0.000296) > Signal (0.000294)

Crossover detected: BULLISH ✅
Signal direction: BULL
```

**Step 3: Validate Against H4 Trend**
```
Signal direction: BULL
H4 trend: bullish
Alignment: ✅ VALID
```

**Step 4: Calculate SL/TP**
```
Entry price: 0.57632
ATR (14-period): 0.00097

Stop Loss: 0.57632 - (1.5 × 0.00097) = 0.57486 (14.6 pips)
Take Profit: 0.57632 + (3.0 × 0.00097) = 0.57925 (29.3 pips)

Risk: 14.6 pips
Reward: 29.3 pips
R:R: 2.00:1 ✅ (> 2:1 required)
```

**Step 5: Calculate Confidence**
```
Base confidence: 60%
+ H4 histogram expanding: +10%
+ H4 histogram strong (0.000115): +10%
= Total: 80% confidence ✅
```

**Step 6: Signal Generated**
```
🎯 BULL SIGNAL
Entry: 0.57632
SL: 0.57486 (14.6 pips)
TP: 0.57925 (29.3 pips)
R:R: 1:2.00
Confidence: 70%
H4 Trend: bullish (histogram: 0.000115)
MACD Crossover: 0.000296 > 0.000294
```

---

## ⚙️ Configuration Settings

### Main Configuration ([config_macd_strategy.py](worker/app/forex_scanner/configdata/strategies/config_macd_strategy.py))

```python
# MACD Parameters
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# H4 Filter
MACD_CONFLUENCE_H4_FILTER_ENABLED = True
MACD_CONFLUENCE_H4_REQUIRE_EXPANSION = True

# Risk Management
MACD_CONFLUENCE_MIN_STOP_PIPS = 10.0
MACD_CONFLUENCE_MAX_STOP_PIPS = 30.0
MACD_CONFLUENCE_MIN_RR_RATIO = 2.0

# ATR Multipliers
MACD_STOP_ATR_MULTIPLIER = 1.5  # SL distance
MACD_TP_ATR_MULTIPLIER = 3.0    # TP distance
```

---

## 🧪 Testing

### Single-Pair Backtest
```bash
docker-compose exec worker python worker/app/forex_scanner/main.py backtest \
    --epic EURUSD \
    --strategy macd \
    --days 30 \
    --timeframe 1h
```

### Multi-Pair Backtest (All 9 Pairs)
```bash
docker-compose exec worker python worker/app/forex_scanner/main.py backtest \
    --strategy macd \
    --days 30 \
    --timeframe 1h \
    --pipeline
```

Or use the shorthand:
```bash
cd /app/forex_scanner
python bt.py --all 30 MACD --pipeline --timeframe 1h --show-signals
```

---

## 📊 Backtest Results (October 2025)

**Test Period**: 30 days (Sept 25 - Oct 27, 2025)
**Timeframe**: 1 hour
**Pairs**: 9 currency pairs

### Performance Summary
- **Total Signals**: 3 (0.01 signals/day/pair)
- **Validation Rate**: 100% (all signals passed validation)
- **Average Confidence**: 70%
- **Signal Distribution**: 2 Bull, 1 Bear

### Trade Outcomes
- **Winners**: 0
- **Losers**: 2 (avg -7.7 pips)
- **Breakeven**: 1
- **Win Rate**: 0%

### Signal Frequency
**Very Low** - This is expected with H4 filter. The strategy waits for:
1. Clear H4 trend direction
2. MACD crossover on 1H
3. Alignment between the two

This results in fewer but potentially higher quality signals.

---

## 🎯 Key Differences from Complex Confluence Version

| Aspect | Complex Confluence | Simple Crossover |
|--------|-------------------|------------------|
| **Entry trigger** | Candlestick pattern at Fib zone | MACD crossover |
| **Components** | Fib + confluence + patterns | H4 filter + crossover |
| **Code complexity** | 606 lines + 4 helpers | 606 lines (simplified) |
| **Signal frequency** | Very low (~3/month) | Very low (~3/month) |
| **Confidence calculation** | Multi-factor (50-90%) | Simple (60-80%) |
| **SL/TP method** | 15M swing or ATR | ATR only |
| **Required indicators** | MACD, Fib, EMAs, patterns | MACD only |

---

## 💡 Strategy Strengths

✅ **Simple and Clear**: Only 5 steps, easy to understand and verify
✅ **Multi-timeframe**: H4 validates, 1H triggers
✅ **Objective Entries**: No subjective pattern interpretation
✅ **Risk-Controlled**: Tight ATR-based stops, minimum 2:1 R:R
✅ **Dynamic Sizing**: ATR adjusts to market conditions
✅ **100% Validation**: All signals pass trade validator

---

## ⚠️ Strategy Weaknesses

⚠️ **Very Low Frequency**: ~3 signals per month across 9 pairs (may miss trends)
⚠️ **Requires Clear Trends**: Neutral H4 = no signals
⚠️ **Crossover Lag**: Entry after momentum already shifted
⚠️ **No Pattern Confirmation**: Relies solely on MACD
⚠️ **Limited Testing**: Only 30 days, small sample size

---

## 🚀 Next Steps

### Immediate Priorities
1. ✅ **Completed**: Simplified strategy implementation
2. ⏳ **Test with histogram filters**: Add minimum histogram size thresholds (user's request: "Later we can work with the histogram sizes")
3. ⏳ **Optimize parameters**: Test different ATR multipliers per pair
4. ⏳ **Extended backtest**: Run 90-180 day test for larger sample size

### Future Enhancements
- Add minimum histogram magnitude threshold (filter weak crossovers)
- Test different timeframe combinations (H4/1H vs H4/15M)
- Add session filters (London/NY only)
- Compare against legacy MACD strategy

---

## 🛠️ Customization Options

### More Signals (Lower Quality)
```python
MACD_CONFLUENCE_H4_FILTER_ENABLED = False  # Remove H4 filter
# or
MACD_CONFLUENCE_H4_REQUIRE_EXPANSION = False  # Allow non-expanding histograms
```

### Higher Quality (Fewer Signals)
```python
MACD_CONFLUENCE_MIN_RR_RATIO = 3.0  # Require 3:1 R:R
MACD_MIN_HISTOGRAM_MAGNITUDE = 0.0001  # Require strong histogram
```

### Tighter Stops
```python
MACD_STOP_ATR_MULTIPLIER = 1.0  # Tighter SL (1.0x ATR)
MACD_TP_ATR_MULTIPLIER = 2.0    # Lower TP (2.0x ATR)
```

### Wider Stops
```python
MACD_STOP_ATR_MULTIPLIER = 2.0  # Wider SL (2.0x ATR)
MACD_TP_ATR_MULTIPLIER = 4.0    # Higher TP (4.0x ATR)
```

---

**Built on October 27, 2025** 🚀

Strategy designed for **1H timeframe** trading on **9 currency pairs** with focus on **simplicity** and **trend alignment**.

Ready to optimize! 🎯
