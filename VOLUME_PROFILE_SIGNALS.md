# 📊 Volume Profile Strategy - Signal Triggers

## Overview

The Volume Profile strategy generates BUY/SELL signals based on institutional volume analysis. It identifies support/resistance at volume clusters and generates mean reversion + breakout signals.

---

## 🎯 Signal Types (Priority Order)

### 1. **HVN Bounce** (Highest Priority)
*Bounce at High Volume Node (institutional support/resistance)*

#### 🟢 **BUY Signal Triggers:**
- ✅ Price is **at or near HVN zone** (within 5 pips by default)
- ✅ Price is **below POC** (Point of Control)
- ✅ HVN acts as **support** (institutions accumulated here)
- ✅ Confidence >= 60% (default)
- ✅ Risk:Reward >= 1.5:1

**Logic:** When price drops to a High Volume Node below POC, it's hitting institutional support where heavy buying occurred historically. Expect a bounce UP.

#### 🔴 **SELL Signal Triggers:**
- ✅ Price is **at or near HVN zone** (within 5 pips by default)
- ✅ Price is **above POC** (Point of Control)
- ✅ HVN acts as **resistance** (institutions distributed here)
- ✅ Confidence >= 60% (default)
- ✅ Risk:Reward >= 1.5:1

**Logic:** When price rises to a High Volume Node above POC, it's hitting institutional resistance where heavy selling occurred historically. Expect a bounce DOWN.

---

### 2. **POC Reversion** (Mean Reversion)
*Price reverts to Point of Control (volume magnet effect)*

#### 🟢 **BUY Signal Triggers:**
- ✅ Price is **below POC** (10-50 pips away)
- ✅ POC acts as a **magnet** pulling price up
- ✅ Confidence >= 60% (adjusted to 57% = 0.95 multiplier)
- ✅ Target = POC price level
- ✅ Risk:Reward >= 1.5:1

**Logic:** POC is the highest volume price level. When price moves away from POC, there's a tendency to revert back to this equilibrium level. Buy when below POC to capture mean reversion UP.

#### 🔴 **SELL Signal Triggers:**
- ✅ Price is **above POC** (10-50 pips away)
- ✅ POC acts as a **magnet** pulling price down
- ✅ Confidence >= 60% (adjusted to 57% = 0.95 multiplier)
- ✅ Target = POC price level
- ✅ Risk:Reward >= 1.5:1

**Logic:** When price is above POC, expect mean reversion back down to the volume magnet. Sell to capture the move down to POC.

**Distance Rules:**
- Too close (<10 pips): No signal (already at POC)
- Sweet spot (10-50 pips): Signal generated
- Too far (>50 pips): No signal (unlikely to reach POC)

---

### 3. **Value Area Breakout**
*Price breaks out of fair value range (VAH/VAL boundaries)*

#### 🟢 **BUY Signal Triggers:**
- ✅ Current price **> VAH** (Value Area High)
- ✅ Previous bar price **<= VAH** (just broke out)
- ✅ Breaking **above** the 70% volume zone (bullish breakout)
- ✅ Confidence >= 60% (adjusted to 54% = 0.90 multiplier)
- ✅ Risk:Reward >= 1.5:1

**Logic:** Value Area contains 70% of trading volume (fair value range). Breaking above VAH signals strong bullish momentum - institutions are driving price higher beyond fair value. Trend acceleration signal.

#### 🔴 **SELL Signal Triggers:**
- ✅ Current price **< VAL** (Value Area Low)
- ✅ Previous bar price **>= VAL** (just broke down)
- ✅ Breaking **below** the 70% volume zone (bearish breakout)
- ✅ Confidence >= 60% (adjusted to 54% = 0.90 multiplier)
- ✅ Risk:Reward >= 1.5:1

**Logic:** Breaking below VAL signals strong bearish momentum - institutions are driving price lower beyond fair value. Trend acceleration signal.

---

### 4. **LVN Breakout** (Lowest Priority, Optional)
*Price breaks through Low Volume Node (weak resistance zones)*

#### 🟢 **BUY Signal Triggers:**
- ✅ Price is **at or near LVN zone** (within 5 pips by default)
- ✅ Price momentum is **positive** (current > previous bar)
- ✅ LVN = weak zone with **minimal volume** (fast breakout expected)
- ✅ Confidence >= 60% (adjusted to 51% = 0.85 multiplier)
- ✅ Risk:Reward >= 1.5:1

**Logic:** LVN zones have very low volume (rejection zones). Price tends to move quickly through these areas with minimal resistance. Positive momentum through LVN = continuation signal UP.

#### 🔴 **SELL Signal Triggers:**
- ✅ Price is **at or near LVN zone** (within 5 pips by default)
- ✅ Price momentum is **negative** (current < previous bar)
- ✅ LVN = weak zone with **minimal volume** (fast breakout expected)
- ✅ Confidence >= 60% (adjusted to 51% = 0.85 multiplier)
- ✅ Risk:Reward >= 1.5:1

**Logic:** Negative momentum through LVN = continuation signal DOWN. Little support to stop the decline.

**Note:** LVN breakouts are disabled by default (`enable_lvn_breakout = False` in default config). Enable in aggressive/crypto presets.

---

## 📊 Key Volume Profile Concepts

### **POC (Point of Control)**
- Price level with **highest volume**
- Acts as a **magnet** (price gravitates toward it)
- Strong support/resistance depending on approach direction

### **VAH/VAL (Value Area High/Low)**
- Boundaries containing **70% of volume**
- Represents **fair value range**
- Inside VA = balanced market
- Outside VA = extreme/trending market

### **HVN (High Volume Node)**
- Price zones with **peak volume**
- Institutional accumulation/distribution zones
- Act as strong **support** (below POC) or **resistance** (above POC)
- Detected using scipy peak detection algorithm

### **LVN (Low Volume Node)**
- Price zones with **minimal volume**
- Rejection zones (price moved through quickly)
- Weak support/resistance = **breakout opportunities**
- Detected using valley detection algorithm

---

## 🎚️ Confidence Scoring

Base confidence starts at **50%** and is adjusted based on:

### **Confidence Boosters (+):**
- ✅ **Strong HVN** (+20% × HVN strength): Stronger volume node = higher confidence
- ✅ **At POC** (+15%): Price at the volume magnet
- ✅ **Favorable Value Area position** (+10%): Price below VA for BUY, above VA for SELL
- ✅ **Volume skewness alignment** (+10%): Volume distribution supports signal direction
- ✅ **Multiple HVN confluence** (+10%): Multiple volume nodes nearby

### **Confidence Penalties (-):**
- ❌ **At LVN** (-15%): Weak support/resistance area
- ❌ **Too far from Value Area** (-10%): >20 pips outside VA boundaries
- ❌ **Counter-trend** (-15%): Signal against trend direction

### **Minimum Confidence Thresholds (by preset):**
- **Default:** 60%
- **Conservative:** 70%
- **Aggressive:** 50%
- **Scalping:** 55%
- **Swing:** 65%
- **News Safe:** 75%
- **Crypto:** 60%

---

## 🛡️ Risk Management

### **Stop Loss Calculation**

**Method: HVN-Based (default)**
- **BUY:** Stop below nearest HVN support (+ 2 pip buffer)
- **SELL:** Stop above nearest HVN resistance (+ 2 pip buffer)
- Fallback: Use VAL (for BUY) or VAH (for SELL)

**Constraints:**
- Min stop: 10 pips
- Max stop: 40 pips (pair-specific)

**Alternative: ATR-Based**
- Stop = 1.5× ATR(14)

### **Take Profit Calculation**

**Method: HVN-Based (default)**
- **BUY:** Target next HVN above or POC (if closer) or VAH
- **SELL:** Target next HVN below or POC (if closer) or VAL

**Constraints:**
- Min Risk:Reward ratio: 1.5:1
- Max target: 60 pips

**Alternative: ATR-Based**
- Target = 3.0× ATR(14)

---

## 🔧 Configuration Presets

| Preset | Lookback | Min Conf | HVN Threshold | Best For |
|--------|----------|----------|---------------|----------|
| **default** | 50 bars | 60% | 5 pips | Balanced trading |
| **conservative** | 100 bars | 70% | 3 pips | High quality signals |
| **aggressive** | 30 bars | 50% | 8 pips | More signals |
| **scalping** | 20 bars | 55% | 3 pips | Fast entries |
| **swing** | 100 bars | 65% | 8 pips | Position trading |
| **news_safe** | 50 bars | 75% | 2 pips | High volatility |
| **crypto** | 40 bars | 60% | 15 pips | Wide zones |

---

## 📈 Example Scenarios

### **Scenario 1: HVN Bounce BUY**
```
Current Price: 1.0850
POC: 1.0900
HVN Zone: 1.0845-1.0855 (center: 1.0850)
VAH: 1.0920
VAL: 1.0880

✅ BUY Signal Generated:
- Price at HVN (1.0850 in zone)
- Price below POC (1.0850 < 1.0900)
- HVN acts as support
- Entry: 1.0850
- Stop: 1.0843 (below HVN - 2 pip buffer)
- Target: 1.0900 (POC) or next HVN above
```

### **Scenario 2: POC Reversion SELL**
```
Current Price: 1.0945
POC: 1.0900
VAH: 1.0920
VAL: 1.0880

✅ SELL Signal Generated:
- Price 45 pips above POC (10-50 pip sweet spot)
- Mean reversion expected
- Entry: 1.0945
- Stop: 1.0960 (15 pips, 1.5× ATR)
- Target: 1.0900 (POC)
- R:R: 45/15 = 3:1 ✅
```

### **Scenario 3: Value Area Breakout BUY**
```
Previous Bar Close: 1.0918
Current Price: 1.0923
VAH: 1.0920
POC: 1.0900

✅ BUY Signal Generated:
- Previous bar <= VAH (1.0918)
- Current price > VAH (1.0923)
- Fresh breakout above 70% volume zone
- Entry: 1.0923
- Stop: 1.0910 (13 pips)
- Target: Next HVN at 1.0950 (27 pips)
- R:R: 27/13 = 2.08:1 ✅
```

---

## 🚀 Quick Reference

**Buy when:**
- Price at HVN **below** POC (support bounce)
- Price **below** POC by 10-50 pips (mean reversion up)
- Price breaks **above** VAH (bullish breakout)
- Price at LVN with **positive** momentum (continuation up)

**Sell when:**
- Price at HVN **above** POC (resistance bounce)
- Price **above** POC by 10-50 pips (mean reversion down)
- Price breaks **below** VAL (bearish breakout)
- Price at LVN with **negative** momentum (continuation down)

**Always check:**
- ✅ Confidence >= threshold
- ✅ Risk:Reward >= 1.5:1
- ✅ Stop loss within min/max limits
- ✅ Not filtered by safety checks
