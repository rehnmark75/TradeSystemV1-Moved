# EMA Trend Pullback Strategy - Optimization Results

**Date:** 2025-12-25
**Backtest Period:** 90 days
**Stocks Tested:** 3,413 tradeable stocks

---

## Phase 3: Multi-Filter Optimization - PF 2.0 Achieved! (Latest)

### Progressive Filter Testing

| Configuration | Win Rate | Profit Factor | Signals | Max DD | Avg Win |
|---------------|----------|---------------|---------|--------|---------|
| ADX>20 + MACD>0 (baseline) | 47.5% | 1.38 | 543 | 278% | +7.61% |
| + RSI 40-60 | 48.1% | 1.38 | 443 | 212% | +7.53% |
| + Skip C/D tiers | 48.1% | 1.38 | 443 | 212% | +7.53% |
| + Volume ≥ 1.0x | 49.2% | 1.63 | 188 | 80% | +8.56% |
| **+ Volume ≥ 1.2x** | **50.4%** | **2.02** | **120** | **53%** | **+9.85%** |

### Final Configuration (PF 2.02)

**Filters applied:**
1. **ADX > 20** - Trending market confirmation
2. **MACD > 0** - Bullish momentum
3. **RSI 40-60** - Healthy pullback zone (not panic or exhaustion)
4. **Skip C/D quality tiers** - Only A+, A, B signals
5. **Volume ≥ 1.2x average** - Institutional participation

**Results:**
- **Profit Factor:** 1.31 → **2.02** (+54%)
- **Win Rate:** 47.2% → **50.4%** (+3.2%)
- **Max Drawdown:** 619% → **53.9%** (-91%)
- **Avg Win:** +7.23% → **+9.85%** (+36%)
- **Signal Count:** 1,284 → 120 (-91%)

**Exit Breakdown:**
- TP Hit: 31 trades (+9.58% avg)
- Timeout Wins: 20 trades (+2.37% avg)
- Gap TP: 9 trades (+27.38% avg)
- SL Hit: 43 trades (-5.39% avg)
- Timeout Losses: 10 trades (-1.96% avg)

---

## Phase 2: ADX + MACD Filter Optimization

### Filter Combination Results (90-day backtest)

| Configuration | Win Rate | Profit Factor | Total P&L | Signals | Max DD |
|---------------|----------|---------------|-----------|---------|--------|
| **Baseline (no filters)** | 47.2% | 1.31 | +1,038.77% | 1,284 | 619% |
| ADX>25 + MACD>0 | 46.2% | 1.25 | +211.50% | 322 | 172% |
| **ADX>20 + MACD>0** | **47.5%** | **1.38** | +535.56% | 543 | **278%** |
| ADX>20 only | 47.8% | 1.34 | +806.00% | 935 | 408% |
| ADX>20 + MACD hist>0 | 47.7% | 1.35 | +248.78% | 285 | 160% |
| MACD>0 only | 45.1% | 1.23 | +458.28% | 719 | 439% |

### Best Configuration: ADX > 20 + MACD > 0

**Improvements over baseline:**
- **Profit Factor:** 1.31 → 1.38 (+5.3%)
- **Max Drawdown:** 619% → 278% (-55%)
- **Signal Quality:** Higher selectivity (58% fewer signals)

**Why this works:**
1. **ADX > 20** - Ensures we only enter during trending markets (Welles Wilder threshold)
2. **MACD > 0** - Confirms bullish momentum, avoiding counter-trend entries

---

## Phase 1: SL/TP Optimization (30-day backtest)

**Signals Generated:** 448

## Optimization Matrix

| SL | TP | R:R | Win Rate | Total P&L | Profit Factor | Max DD | Avg Win | Avg Loss |
|----|----|-----|----------|-----------|---------------|--------|---------|----------|
| 2% | 4% | 2:1 | 44.5% | +37.85% | 1.07 | 73.25% | +2.85% | +2.33% |
| 2% | 6% | 3:1 | 42.7% | +178.14% | 1.31 | 79.55% | +4.01% | +2.31% |
| 2% | 8% | 4:1 | 41.3% | +263.47% | 1.46 | 77.02% | +4.67% | +2.32% |
| 2.5% | 10% | 4:1 | 42.9% | +254.80% | 1.39 | 80.49% | +4.85% | +2.70% |
| 3% | 6% | 2:1 | 49.5% | +209.22% | 1.33 | 85.70% | +3.93% | +3.01% |
| 3% | 9% | 3:1 | 47.2% | +311.18% | 1.47 | 84.05% | +4.75% | +3.02% |
| 4% | 6% | 1.5:1 | 53.0% | +203.11% | 1.29 | 89.86% | +3.93% | +3.56% |
| 4% | 8% | 2:1 | 51.6% | +314.86% | 1.44 | 90.21% | +4.61% | +3.58% |
| **5%** | **10%** | **2:1** | **54.6%** | **+447.58%** | **1.61** | **70.63%** | +4.98% | +3.90% |

## Key Findings

### Best Overall: 5% SL / 10% TP (2:1 R:R)
- **Highest P&L: +447.58%**
- **Best Profit Factor: 1.61**
- **Best Win Rate: 54.6%**
- **Lowest Max Drawdown: 70.63%**

The wider stops give trades room to breathe during pullback consolidation before continuing the trend.

### Best for Tight Risk: 2% SL / 8% TP (4:1 R:R)
- Good P&L: +263.47%
- Strong Profit Factor: 1.46
- Lower individual trade risk

### Sweet Spot Balanced: 3% SL / 9% TP (3:1 R:R)
- Solid P&L: +311.18%
- Excellent Profit Factor: 1.47
- Good balance of win rate and reward

## Exit Reason Analysis (5% SL / 10% TP)

| Result | Reason | Count | Avg P&L |
|--------|--------|-------|---------|
| WIN | TIMEOUT | 162 | +2.66% |
| LOSS | TIMEOUT | 91 | -1.82% |
| LOSS | SL_HIT | 88 | -5.39% |
| WIN | TP_HIT | 61 | +9.58% |
| WIN | GAP_TP | 15 | +11.26% |
| LOSS | GAP_SL | 10 | -9.68% |
| BREAKEVEN | TIMEOUT | 9 | +0.02% |

## Quality Tier Performance (5% SL / 10% TP)

| Tier | Count | Wins | Win% | Avg P&L |
|------|-------|------|------|---------|
| A | 185 | 98 | 53.0% | +0.84% |
| A+ | 158 | 85 | 53.8% | +0.66% |
| B | 88 | 45 | 51.1% | +1.67% |
| C | 16 | 10 | 62.5% | +3.78% |

## Recommendations

1. **Default Configuration:** 5% SL / 10% TP
2. **Conservative Alternative:** 2% SL / 8% TP for smaller account risk
3. **Aggressive Alternative:** 3% SL / 9% TP for higher frequency

## Notes

- All tests used the same 448 signals from the EMA Trend Pullback strategy
- Pullback threshold: 2-5% below 20 EMA
- Entry: Price crosses back above 20 EMA while above 50/100/200 EMAs
- Max holding period: 20 days
- Commission: 0.1%, Slippage: 0.1%
