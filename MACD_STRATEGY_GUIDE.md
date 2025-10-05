# MACD Strategy - Complete Configuration Guide

## Overview
Clean MACD strategy rebuilt from scratch with zero-line crossover detection, configurable EMA trend filter, and comprehensive validation.

## Key Features
1. **Standard MACD (12, 26, 9)** zero-line crossover detection
2. **Configurable EMA trend filter** (disabled by default)
3. **ADX filter** with Wilder's smoothing (ADX >= 20)
4. **RSI filter** (30-70 range, not extreme)
5. **Swing proximity validation** (8 pips minimum distance)
6. **Histogram direction check** for momentum confirmation
7. **One signal per crossover event** (no duplicates)

## Critical Bug Fixes Implemented

### 1. ADX Calculation Fixed
- **Issue**: Was using simple rolling mean instead of Wilder's smoothing
- **Fix**: Now uses exponential weighted mean (EWM) with alpha=1/14
- **Impact**: ADX values now match TradingView's standard calculation
- **Location**: `worker/app/forex_scanner/core/strategies/macd_strategy.py:366-400`

```python
# Correct Wilder's Smoothing (RMA)
atr = tr.ewm(alpha=1/period, adjust=False).mean()
plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
df['adx'] = dx.ewm(alpha=1/period, adjust=False).mean()
```

### 2. NaN Validation Added
- **Issue**: NaN ADX values bypassed filter (NaN < 25 returns False in Python)
- **Fix**: Explicit `pd.isna()` checks before threshold comparison
- **Location**: `worker/app/forex_scanner/core/strategies/macd_strategy.py:120-133`

## Configuration

### File Location
`worker/app/forex_scanner/configdata/strategies/config_macd_strategy.py`

### EMA Filter Configuration (Lines 361-371)

```python
MACD_EMA_FILTER = {
    'enabled': False,  # Default: disabled for more signals
    'ema_period': 50,  # Options: 50 (faster), 100 (balanced), 200 (slower)
}
```

**To enable EMA filter**, edit the config file and set:
```python
'enabled': True,
'ema_period': 50,  # or 100, or 200
```

### Swing Validation Configuration (Lines 351-359)

```python
MACD_SWING_VALIDATION = {
    'enabled': True,
    'min_distance_pips': 8,  # Minimum distance from swing high/low
    'lookback_swings': 5,
    'swing_length': 5,
    'strict_mode': False,  # False = reduce confidence, True = reject entirely
}
```

## Signal Quality Results (7-day backtest, 9 pairs)

### With EMA Filter Disabled (Default)
- **Total signals**: 66
- **Validated signals**: 27 (40.9% pass rate)
- **Average confidence**: 72.0%
- **Bull/Bear ratio**: 32/34 (balanced)
- **Signal frequency**: ~0.98 signals/day/pair
- **Rejection reason**: All 39 rejections due to S/R level conflicts

### With EMA 50 Filter Enabled
- **Total signals**: 12
- **Validated signals**: 9 (75.0% pass rate)
- **Signal frequency**: ~0.19 signals/day/pair
- **Quality**: Significantly higher (75% validation rate)
- **Trade-off**: Fewer signals but better quality

### With EMA 100 Filter Enabled
- **Total signals**: 48
- **Validated signals**: 23 (47.9% pass rate)
- **Signal frequency**: ~0.76 signals/day/pair
- **Balance**: Moderate quantity with good quality

## Current Thresholds (Optimized)

| Parameter | Value | Notes |
|-----------|-------|-------|
| ADX Minimum | 20 | Lowered from 25 for more signals |
| Confidence Minimum | 60% | Balance between quality and quantity |
| RSI Range | 30-70 | Skip extreme overbought/oversold zones |
| Swing Distance | 8 pips | Minimum distance from recent swing points |
| EMA Filter | Disabled | Default: off for maximum signals |

## Backtest Commands

### Full Pipeline Mode (Recommended)
Includes strategy filters + TradeValidator (S/R, news, freshness):
```bash
docker compose exec task-worker python /app/forex_scanner/bt.py --all 7 MACD --pipeline --timeframe 15m --show-signals
```

### Strategy Filters Only
Tests strategy filters without full validation chain:
```bash
docker compose exec task-worker python /app/forex_scanner/bt.py --all 7 MACD --timeframe 15m --show-signals
```

## Strategy Behavior

### Signal Generation
- **BULL Signal**: MACD line crosses above zero with ADX >= 20, RSI 30-70, not near swing high
- **BEAR Signal**: MACD line crosses below zero with ADX >= 20, RSI 30-70, not near swing low

### Validation Chain
1. **Strategy Filters** (in MACD strategy):
   - Zero-line crossover detection
   - ADX >= 20 (trending market)
   - RSI 30-70 (not extreme)
   - EMA filter (if enabled)
   - Swing proximity check (8 pips)
   - Histogram direction confirmation
   - Confidence >= 60%

2. **TradeValidator** (in pipeline mode):
   - S/R level validation
   - News event filter
   - EMA 200 trend filter (MACD bypasses this for counter-trend trades)
   - Freshness check

### Counter-Trend Trading
MACD strategy can trade counter-trend (momentum reversals). It has special exception logic in TradeValidator that bypasses the EMA 200 filter, allowing it to catch momentum reversals against the main trend.

## Performance Metrics

| Metric | EMA Disabled | EMA 50 | EMA 100 |
|--------|--------------|--------|---------|
| Total Signals | 66 | 12 | 48 |
| Validated | 27 (40.9%) | 9 (75.0%) | 23 (47.9%) |
| Signals/Day/Pair | 0.98 | 0.19 | 0.76 |
| Avg Profit | 152.5 pips | - | - |
| Avg Loss | 98.9 pips | - | - |

## Recommendations

### For Maximum Signals
Keep EMA filter **disabled** (default setting):
- 66 signals over 7 days
- 40.9% validation rate
- Good for active trading

### For Quality Signals
Enable **EMA 50 filter**:
```python
MACD_EMA_FILTER = {
    'enabled': True,
    'ema_period': 50,
}
```
- 12 signals over 7 days
- 75% validation rate
- Best signal quality

### For Balanced Approach
Enable **EMA 100 filter**:
```python
MACD_EMA_FILTER = {
    'enabled': True,
    'ema_period': 100,
}
```
- 48 signals over 7 days
- ~48% validation rate
- Good balance of quantity and quality

## File Locations

| Component | Path |
|-----------|------|
| Strategy | `worker/app/forex_scanner/core/strategies/macd_strategy.py` |
| Configuration | `worker/app/forex_scanner/configdata/strategies/config_macd_strategy.py` |
| Data Fetcher | `worker/app/forex_scanner/core/data_fetcher.py` |
| Trade Validator | `worker/app/forex_scanner/core/trading/trade_validator.py` |

## Key Code Sections

### ADX Calculation with Wilder's Smoothing
Location: `macd_strategy.py:366-400`

### EMA Filter Logic
Location: `macd_strategy.py:154-186`

### Signal Validation
Location: `macd_strategy.py:95-187`

### Zero-Line Crossover Detection
Location: `macd_strategy.py:305-365`

## Troubleshooting

### Too Many Signals
- Enable EMA filter (50 or 100)
- Increase ADX threshold in strategy code
- Increase confidence threshold

### Too Few Signals
- Disable EMA filter (default)
- Lower ADX threshold (currently 20)
- Check ADX calculation is working correctly

### Signals Don't Match TradingView
- Verify ADX calculation uses Wilder's smoothing
- Check for NaN values in ADX column
- Ensure MACD parameters are (12, 26, 9)

## Recent Optimization History

| Date | Change | Impact |
|------|--------|--------|
| 2025-10-05 | Fixed ADX calculation (Wilder's smoothing) | ADX values now match TradingView |
| 2025-10-05 | Added NaN validation for ADX | Prevents invalid signals |
| 2025-10-05 | Made EMA filter configurable | User can choose signal quantity vs quality |
| 2025-10-05 | Lowered ADX from 25 to 20 | More signals while maintaining trend requirement |
| 2025-10-05 | Lowered confidence from 65% to 60% | Balanced quality and quantity |

---

**Last Updated**: 2025-10-05
**Strategy Version**: 2.0 (Rebuilt with configurable EMA filter)
