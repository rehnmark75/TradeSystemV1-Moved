# Multi-Epic Parameter Optimizer

A comprehensive parameter optimization system that runs backtests across all enabled currency pairs to find optimal strategy configurations.

## Overview

The multi-epic optimizer automates the process of finding the best parameters for the SMC Simple strategy across all 9 enabled currency pairs. It:

- Tests multiple parameter combinations per epic
- Uses smart R:R filtering to skip invalid combinations
- Shows real-time progress with ETA
- Supports pause/resume on interruption
- Auto-creates snapshots for best configurations
- Exports results to CSV for analysis

## Quick Start

```bash
# See what will be tested (no execution)
docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --dry-run

# Run fast mode optimization (~7 hours)
docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --days 30

# Test specific pairs only
docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --mode fast --epics EURUSD GBPUSD
```

## Optimization Modes

| Mode | Combinations/Epic | Total Tests (9 epics) | Est. Time | Use Case |
|------|-------------------|----------------------|-----------|----------|
| **fast** | ~16 | ~144 | ~7 hours | Quick validation, initial testing |
| **medium** | ~48 | ~432 | ~22 hours | Balanced depth/time |
| **extended** | ~102 | ~918 | ~46 hours | Thorough optimization |

### Parameters by Mode

**Fast Mode** (core SL/TP/Confidence):
```
fixed_stop_loss_pips: [8, 10, 12]
fixed_take_profit_pips: [16, 20, 25]
min_confidence: [0.48, 0.52]
```

**Medium Mode** (+ SL buffer):
```
fixed_stop_loss_pips: [8, 10, 12]
fixed_take_profit_pips: [16, 20, 25]
min_confidence: [0.45, 0.50, 0.55]
sl_buffer_pips: [4, 6]
```

**Extended Mode** (+ wider ranges):
```
fixed_stop_loss_pips: [7, 8, 9, 10, 12]
fixed_take_profit_pips: [14, 16, 20, 25]
min_confidence: [0.45, 0.50, 0.55]
sl_buffer_pips: [4, 6]
```

## CLI Reference

```
usage: multi_epic_optimizer.py [-h] [--mode {fast,medium,extended}]
                               [--days DAYS] [--epics EPICS [EPICS ...]]
                               [--resume RESUME] [--output-dir OUTPUT_DIR]
                               [--no-snapshots] [--dry-run]

Multi-Epic Parameter Optimization

optional arguments:
  -h, --help            show this help message and exit
  --mode {fast,medium,extended}
                        Optimization mode (default: fast)
  --days DAYS           Backtest period in days (default: 30)
  --epics EPICS [EPICS ...]
                        Specific epic shortcuts to test (e.g., EURUSD GBPUSD)
  --resume RESUME       Resume interrupted run by run ID
  --output-dir OUTPUT_DIR
                        Output directory for CSV exports
  --no-snapshots        Do not create snapshots for best configs
  --dry-run             Show what would be tested without running
```

### Epic Shortcuts

Use these shortcuts instead of full epic IDs:

| Shortcut | Full Epic ID |
|----------|-------------|
| EURUSD | CS.D.EURUSD.CEEM.IP |
| GBPUSD | CS.D.GBPUSD.MINI.IP |
| USDJPY | CS.D.USDJPY.MINI.IP |
| AUDUSD | CS.D.AUDUSD.MINI.IP |
| USDCHF | CS.D.USDCHF.MINI.IP |
| USDCAD | CS.D.USDCAD.MINI.IP |
| NZDUSD | CS.D.NZDUSD.MINI.IP |
| EURJPY | CS.D.EURJPY.MINI.IP |
| AUDJPY | CS.D.AUDJPY.MINI.IP |

## Progress Display

During optimization, a progress bar shows real-time status:

```
============================================================
📈 OPTIMIZING EURUSD (1/9)
============================================================
EURUSD [████████████░░░░░░░░░░░░░░░░░░] 42.0% (7/16) | Overall: 4.9% | ETA: 6.2h | SL=10, TP=20, Conf=52%
  ✅ Signals=45, Win=58%, PF=1.85, Pips=+125.5 | Avg: 178s/test
```

**Progress elements:**
- Epic name and position (1/9)
- Visual progress bar for current epic
- Percentage complete (current epic and overall)
- ETA (estimated time remaining)
- Current parameters being tested
- Last result (signals, win rate, profit factor, pips)
- Average time per test

## Pause & Resume

Press `Ctrl+C` to gracefully pause optimization. Progress is saved to the database.

```bash
# Resume an interrupted run
docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py --resume 12345
```

The optimizer will continue from where it left off.

## Output

### Auto-Created Snapshots

When optimization completes for each epic, a snapshot is automatically created with the best parameters:

```
📦 Snapshot: eurusd_optimized_20260107
```

Use snapshots in backtests:
```bash
docker exec -it task-worker python /app/forex_scanner/bt.py EURUSD 14 --snapshot eurusd_optimized_20260107
```

### CSV Export

Results are exported to CSV for analysis:
```
/app/forex_scanner/optimization_results/optimization_123_20260107_1430.csv
```

CSV columns:
- epic, fixed_stop_loss_pips, fixed_take_profit_pips, min_confidence
- rr_ratio, sl_buffer_pips
- total_signals, winners, losers
- win_rate, profit_factor, total_pips
- composite_score, status, duration_seconds

### Final Report

After completion, a cross-epic comparison is displayed:

```
================================================================================
CROSS-EPIC PARAMETER COMPARISON
================================================================================

Epic       SL    TP    R:R    Conf   Signals  Win%     PF       Pips         Score
----------------------------------------------------------------------------------------------------
EURUSD     8     16    2.00   52%    45       58.0%    1.85     +125.5       0.7234
GBPUSD     10    20    2.00   50%    52       55.0%    1.72     +98.2        0.6891
USDJPY     8     20    2.50   48%    38       52.0%    1.55     +67.4        0.6123
...

🏆 BEST PERFORMING PAIR: EURUSD
   Parameters: SL=8, TP=16, Conf=52%
   Performance: Win Rate=58.0%, PF=1.85, Pips=+125.5
   Snapshot: eurusd_optimized_20260107
```

## Composite Scoring

Results are ranked using a composite score:

```
composite_score = (
    win_rate * 0.30 +                          # 30% weight
    min(profit_factor / 3.0, 1.0) * 0.30 +    # 30% weight (PF normalized, cap at 3)
    (total_pips / 100) * 0.20 +               # 20% weight (pips normalized)
    min(signal_count / 50, 1.0) * 0.20        # 20% weight (signal count, cap at 50)
)
```

Higher scores indicate better parameter combinations.

## Database Tables

Optimization data is stored in three tables:

### optimization_runs
Master record for each optimization run:
- Run name, mode, status
- Progress tracking (for resume)
- Best overall results

### optimization_results
Individual test results:
- Epic, parameters, execution_id
- Win rate, profit factor, pips
- Composite score

### optimization_best_params
Best parameters per epic per run:
- Best params (JSONB)
- Performance metrics
- Snapshot reference

## R:R Filtering

Invalid parameter combinations are automatically filtered:

- **Minimum R:R**: 1.5 (TP must be at least 1.5× SL)
- **Maximum R:R**: 4.0 (avoid unrealistic targets)

This reduces the number of tests by ~40%.

## Examples

### Quick Test on Specific Pairs
```bash
# Test only majors with fast mode
docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py \
    --mode fast --days 14 --epics EURUSD GBPUSD USDJPY
```

### Extended Overnight Run
```bash
# Run in background with log file
docker exec task-worker python /app/forex_scanner/multi_epic_optimizer.py \
    --mode extended --days 30 2>&1 | tee optimization_output.log &
```

### Disable Auto-Snapshots
```bash
# Just analyze, don't create snapshots
docker exec -it task-worker python /app/forex_scanner/multi_epic_optimizer.py \
    --mode fast --no-snapshots
```

## Querying Results

### Recent Optimization Runs
```sql
SELECT id, run_name, run_mode, status,
       completed_combinations, total_combinations,
       best_overall_epic, best_overall_score
FROM optimization_runs
ORDER BY created_at DESC LIMIT 5;
```

### Best Parameters for an Epic
```sql
SELECT run_id, best_params, win_rate, profit_factor, total_pips, snapshot_name
FROM optimization_best_params
WHERE epic = 'CS.D.EURUSD.CEEM.IP'
ORDER BY composite_score DESC LIMIT 3;
```

### All Results for a Run
```sql
SELECT epic, params_tested, win_rate, profit_factor, total_pips, composite_score
FROM optimization_results
WHERE run_id = 123 AND status = 'completed'
ORDER BY composite_score DESC;
```

## Troubleshooting

### "No results found" errors
The backtest may have failed. Check:
- Database connectivity
- Sufficient candle data for the backtest period
- Strategy configuration is valid

### Slow performance
Each backtest takes ~3 minutes. To speed up:
- Use `--mode fast` for fewer combinations
- Test fewer epics with `--epics`
- Use shorter periods with `--days 14`

### Resuming fails
Ensure the run ID exists:
```sql
SELECT id, status FROM optimization_runs WHERE id = 12345;
```

## Files

| File | Purpose |
|------|---------|
| `multi_epic_optimizer.py` | Main optimizer script |
| `optimization_config.py` | Parameter tier definitions |
| `migrations/create_multi_epic_optimization_tables.sql` | Database schema |
| `optimization_results/` | CSV export directory |
