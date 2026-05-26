# Unified Parameter Optimizer

## Overview

The Unified Parameter Optimizer is a comprehensive CLI tool that analyzes historical rejection outcomes and trade data to generate data-driven parameter recommendations for the SMC Simple strategy. It identifies over-restrictive filters, suggests optimal thresholds, and generates ready-to-apply SQL statements.

## Quick Start

```bash
# Basic analysis (30 days, all enabled epics)
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py

# Verbose output with full analysis
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py --days 30 -v

# Analyze specific epic
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py --epic EURUSD

# Export SQL to file
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py --export-sql /tmp/optimizations.sql
```

## Architecture

```
worker/app/forex_scanner/optimization/
├── unified_parameter_optimizer.py     # Main CLI + orchestrator
├── data_collectors/
│   ├── base_collector.py              # Base class with DB access
│   ├── trade_collector.py             # Collects from trade_log + alert_history
│   ├── rejection_collector.py         # Collects from smc_simple_rejections + outcomes
│   └── market_intel_collector.py      # Collects from market_intelligence_history
├── analyzers/
│   ├── correlation_analyzer.py        # Parameter correlation analysis
│   ├── direction_analyzer.py          # BULL vs BEAR performance
│   └── regime_analyzer.py             # Market regime correlation
└── generators/
    ├── sql_generator.py               # Generates UPDATE statements
    └── report_generator.py            # Formats console output
```

## Data Sources

### 1. Rejection Outcomes (`smc_simple_rejections` + `smc_rejection_outcomes`)

The primary data source. Contains signals that were rejected by various filters, with tracked outcomes showing what would have happened if the signal was taken.

**Key columns analyzed:**
- `rejection_stage` - Which filter rejected the signal (TIER2_SWING, MACD_FILTER, SMC_CONFLICT, etc.)
- `would_be_winner` - Whether the rejected signal would have hit TP (true) or SL (false)
- `potential_profit_pips` - Pips that would have been gained/lost

**50+ indicator columns collected:**
- Core: `confidence_score`, `volume_ratio`, `pullback_depth`
- MACD: `macd_histogram`, `macd_line`, `macd_signal`, `macd_aligned`
- Trend: `adx_value`, `efficiency_ratio`, `kama_er`
- Volatility: `atr_percentile`, `bb_width_percentile`, `bb_percent_b`
- Momentum: `stoch_k`, `stoch_d`, `rsi_zone`
- Structure: `swing_range_pips`, `ema_distance_pips`

### 2. Trade Outcomes (`trade_log` + `alert_history`)

Actual executed trades with real outcomes. Used for SL/TP optimization via MFE/MAE analysis.

### 3. Market Intelligence (`market_intelligence_history`)

Market regime context for regime-based filtering recommendations.

## Analysis Methods

### 1. Rejection Stage Analysis

Identifies filters that reject too many winning signals:

```
TIER2_SWING rejections: 65% win rate (187 samples)
  → Recommendation: Reduce min_swing_atr_multiplier from 0.25 to 0.15
```

**Detected stages and recommendations:**
| Stage Pattern | Parameter | Action |
|--------------|-----------|--------|
| TIER2/SWING | `min_swing_atr_multiplier` | Reduce by 40% |
| MACD | `macd_filter_enabled` | Consider disabling |
| SMC_CONFLICT | `smc_conflict_tolerance` | Increase to 1 |
| CONFIDENCE_CAP | `max_confidence` | Raise to 0.95 |

### 2. Numeric Parameter Binning

Divides parameter values into quartiles and compares win rates:

```
volume_ratio analysis:
  Q1 (0.15-0.25): 58% win rate, 45 samples  ← Being over-rejected!
  Q2 (0.25-0.35): 52% win rate, 38 samples
  Q3 (0.35-0.50): 48% win rate, 42 samples
  Q4 (0.50+):     45% win rate, 31 samples

  → Recommendation: Lower min_volume_ratio to capture Q1 winners
```

### 3. Categorical Parameter Analysis

Analyzes win rates by category for parameters like:
- `volatility_state`: low/medium/high
- `market_regime_detected`: trending/ranging/breakout
- `adx_trend_strength`: weak/moderate/strong
- `rsi_zone`: oversold/neutral/overbought

### 4. MACD Histogram Strength Analysis

**Key insight:** It's not just positive/negative MACD that matters, but the strength of the histogram.

```
Weak momentum (|histogram| < median): 62% win rate
Strong momentum (|histogram| > median): 55% win rate

→ Paradox detected! Weak momentum signals are being over-rejected.
```

### 5. ADX Trend Strength Buckets

```
ADX 0-20 (weak):      58% win rate ← Consider including
ADX 20-25 (developing): 52% win rate
ADX 25-40 (strong):    48% win rate
ADX 40+ (very strong): 45% win rate
```

### 6. Direction Analysis (BULL vs BEAR)

Compares performance by trade direction per epic:

```
EURUSD:
  BULL: 48% win rate (52 trades)
  BEAR: 62% win rate (37 trades)

  → Enable direction_overrides_enabled
  → Tighten BULL filters, relax BEAR filters
```

### 7. Regime Analysis

Identifies market regimes where the strategy underperforms:

```
EURUSD regime performance:
  trending: 61% win rate ✓
  ranging:  25% win rate ✗ RECOMMEND BLOCKING
  breakout: 55% win rate
```

## Configuration Loading

The optimizer loads current per-pair configuration from `smc_simple_pair_overrides` table in the `strategy_config` database:

**Parameters loaded:**
- SL/TP: `fixed_stop_loss_pips`, `fixed_take_profit_pips`, `sl_buffer_pips`
- Confidence: `min_confidence`, `max_confidence`, `min_confidence_bull/bear`
- Volume: `min_volume_ratio`, `min_volume_ratio_bull/bear`
- Fib: `fib_pullback_min_bull/bear`, `fib_pullback_max_bull/bear`
- Momentum: `momentum_min_depth_bull/bear`
- Swing: `min_swing_atr_multiplier`, `swing_lookback_bars`
- Filters: `macd_filter_enabled`, `smc_conflict_tolerance`, `direction_overrides_enabled`

**Important:** The optimizer shows actual current values from the database, not defaults. If you change a parameter in the database, subsequent optimizer runs will show the updated current value.

## Output Format

### Console Report

```
════════════════════════════════════════════════════════════════════════════════
                    UNIFIED PARAMETER OPTIMIZATION REPORT
                    Period: 30 days | Generated: 2026-01-08 18:22:01
════════════════════════════════════════════════════════════════════════════════

SUMMARY:
  Epics Analyzed: 8
  Total Trades: 0
  Rejections Analyzed: 3488

════════════════════════════════════════════════════════════════════════════════
USDCAD (CS.D.USDCAD.MINI.IP)
════════════════════════════════════════════════════════════════════════════════

PARAMETER RECOMMENDATIONS:
┌───────────────────────┬─────────┬─────────────┬────────────┬──────────────┐
│ Parameter             │ Current │ Recommended │ Confidence │ Est. Impact  │
├───────────────────────┼─────────┼─────────────┼────────────┼──────────────┤
│ min_swing_atr_multipli│     0.15│         0.09│        90% │ +1500 pips/mo│
└───────────────────────┴─────────┴─────────────┴────────────┴──────────────┘

SQL (copy to apply):
────────────────────────────────────────────────────────────────────────────────
UPDATE smc_simple_pair_overrides SET
    min_swing_atr_multiplier = 0.09,
    change_reason = 'Unified optimizer: 1 changes on 2026-01-08',
    updated_at = NOW()
WHERE epic = 'CS.D.USDCAD.MINI.IP';
```

### SQL Generation

The optimizer generates two types of SQL:

**1. Per-Pair Parameters** (directly applicable):
```sql
UPDATE smc_simple_pair_overrides SET
    min_swing_atr_multiplier = 0.09,
    min_confidence = 0.42,
    change_reason = 'Unified optimizer: 2 changes on 2026-01-08',
    updated_at = NOW()
WHERE epic = 'CS.D.USDCAD.MINI.IP';
```

**2. Global Config Suggestions** (commented, requires manual review):
```sql
-- GLOBAL CONFIG SUGGESTION for CS.D.USDCAD.MINI.IP:
-- These parameters are global (affect all pairs). Consider adding per-pair columns.
-- UPDATE smc_simple_global_config SET
--     macd_min_histogram_strength = 0.00005,
--     change_reason = 'Unified optimizer suggestion',
--     updated_at = NOW()
-- WHERE is_active = TRUE;
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--days` | 30 | Days of historical data to analyze |
| `--epic` | all enabled | Specific epic(s) to analyze |
| `--min-sample-size` | 20 | Minimum samples for recommendation |
| `--min-confidence` | 0.70 | Minimum confidence level (0-1) |
| `--direction-analysis` | false | Include BULL/BEAR breakdown |
| `--include-regime-analysis` | false | Include regime filter recommendations |
| `--export-sql FILE` | - | Export SQL to file |
| `--export-json FILE` | - | Export recommendations as JSON |
| `-v, --verbose` | false | Detailed logging output |

## Per-Pair Parameters

These parameters can be set per-pair in `smc_simple_pair_overrides`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_confidence` | decimal | Minimum confidence threshold |
| `max_confidence` | decimal | Maximum confidence cap |
| `min_volume_ratio` | decimal | Minimum volume ratio |
| `fixed_stop_loss_pips` | numeric | Fixed SL in pips |
| `fixed_take_profit_pips` | numeric | Fixed TP in pips |
| `sl_buffer_pips` | integer | SL buffer |
| `macd_filter_enabled` | boolean | Enable/disable MACD filter |
| `fib_pullback_min_bull/bear` | decimal | Min fib pullback by direction |
| `fib_pullback_max_bull/bear` | decimal | Max fib pullback by direction |
| `min_volume_ratio_bull/bear` | decimal | Volume ratio by direction |
| `min_confidence_bull/bear` | decimal | Confidence by direction |
| `momentum_min_depth_bull/bear` | decimal | Momentum depth by direction |
| `direction_overrides_enabled` | boolean | Enable direction-specific params |
| `min_swing_atr_multiplier` | decimal | Swing filter strictness |
| `swing_lookback_bars` | integer | Bars for swing detection |
| `smc_conflict_tolerance` | integer | Allowed SMC conflicts (0=strict) |

## Recommendation Confidence Calculation

Confidence is calculated based on:
1. **Sample size**: More samples = higher confidence
2. **Win rate difference**: Larger gap between best/worst = higher confidence
3. **Statistical significance**: Minimum thresholds enforced

Formula:
```python
confidence = min(0.90, 0.5 + (sample_size / 50) + (win_rate_diff / 2))
```

Recommendations below `--min-confidence` (default 0.70) are filtered out.

## Workflow

### 1. Regular Optimization Cycle

```bash
# Run weekly optimization analysis
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py --days 30 -v

# Review recommendations
# Apply selected SQL statements manually

# Restart to pick up changes
docker restart task-worker
```

### 2. Investigating Specific Epic

```bash
# Check why EURUSD is underperforming
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py \
    --epic EURUSD --direction-analysis --include-regime-analysis -v
```

### 3. Batch Apply with Review

```bash
# Export all recommendations
docker exec task-worker python /app/forex_scanner/optimization/unified_parameter_optimizer.py \
    --days 30 --export-sql /tmp/optimizations.sql

# Review the file
docker exec task-worker cat /tmp/optimizations.sql

# Apply after review
docker exec postgres psql -U postgres -d strategy_config -f /tmp/optimizations.sql
```

## Integration with Streamlit UI

The Streamlit Parameter Optimizer tab provides a complementary UI:
- Visual parameter adjustment
- Real-time preview of changes
- Direct database updates via UI

The CLI optimizer provides deeper statistical analysis while Streamlit provides interactive parameter management.

## Database Tables

### Input Tables (forex database)

| Table | Purpose |
|-------|---------|
| `smc_simple_rejections` | Rejected signal snapshots with all indicators |
| `smc_rejection_outcomes` | Tracked outcomes (HIT_TP/HIT_SL) for rejections |
| `trade_log` | Actual executed trades |
| `alert_history` | Signal details at time of alert |
| `market_intelligence_history` | Market regime snapshots |

### Config Tables (strategy_config database)

| Table | Purpose |
|-------|---------|
| `smc_simple_pair_overrides` | Per-pair parameter overrides |
| `smc_simple_global_config` | Global strategy parameters |
| `smc_simple_config_audit` | Change history |

## Troubleshooting

### "No rejection data found"

Check that rejection tracking is enabled and data exists:
```bash
docker exec postgres psql -U postgres -d forex -c "
SELECT COUNT(*) FROM smc_simple_rejections
WHERE scan_timestamp >= NOW() - INTERVAL '30 days'"
```

### Recommendations show stale values

The optimizer loads config from the database on each run. If values appear stale:
1. Verify the change was saved to `smc_simple_pair_overrides`
2. Ensure the parameter column is included in `_load_current_config()` query

### High-confidence recommendations not appearing

Check if:
- Sample size meets `--min-sample-size` threshold (default 20)
- Win rate difference is significant (>15% for numeric, >20% for categorical)
- Overall win rate for rejected signals is >55%

## See Also

- [Adding New Strategies](adding_new_strategy.md) - Strategy development guide
- [Multi-Epic Optimizer](multi_epic_optimizer.md) - Parallel backtesting optimizer
- CLAUDE.md - Main project documentation
