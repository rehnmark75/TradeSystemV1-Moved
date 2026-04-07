# Backtest Runner - How to Run and Capture Backtest Output

This skill ensures backtests are run correctly inside Docker and their output is reliably captured.

---

## Running a Backtest

**Always run inside the `task-worker` container:**

```bash
docker exec -it task-worker python -u /app/forex_scanner/bt.py EURUSD 7
```

### Common Examples

```bash
# Single pair, 7 days (default)
docker exec -it task-worker python -u /app/forex_scanner/bt.py EURUSD 7

# With signal details
docker exec -it task-worker python -u /app/forex_scanner/bt.py EURUSD 7 --show-signals

# Scalp mode (recommended for live comparison)
docker exec -it task-worker python -u /app/forex_scanner/bt.py EURUSD 7 --scalp --timeframe 5m

# With parameter overrides
docker exec -it task-worker python -u /app/forex_scanner/bt.py EURUSD 14 --override fixed_stop_loss_pips=10 --override min_confidence=0.55

# Multiple pairs (no pair = all enabled pairs)
docker exec -it task-worker python -u /app/forex_scanner/bt.py 7
```

### Pair Shortcuts
EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD, EURJPY, AUDJPY, GBPJPY

### Key Flags
| Flag | Purpose |
|------|---------|
| `--scalp` | Scalp mode with optimized SL/TP |
| `--timeframe 5m` | Scan interval (use 5m for live comparison) |
| `--show-signals` | Show detailed signal table |
| `--override PARAM=VALUE` | Override strategy parameter (repeatable) |
| `--pipeline` | Full pipeline with trade validator |
| `--parallel --workers 8` | Parallel execution for long periods |

---

## Capturing Output (CRITICAL)

Backtest output is large (often 5000+ lines). The Bash tool has a 2-minute default timeout, and output can be truncated. Follow these rules:

### Rule 1: Always Use Background Mode for Backtests

Backtests take 30-120+ seconds. Always use `run_in_background=true` on the Bash tool and redirect output to a file:

```bash
docker exec task-worker python -u /app/forex_scanner/bt.py EURUSD 7 > /tmp/bt_eurusd_7d.txt 2>&1
```

Then read the results file with the Read tool after the command completes.

### Rule 2: Read the Summary Section (End of Output)

The performance summary is always at the END of the output. Read the last 100-150 lines to get results:

```
Read tool: /tmp/bt_eurusd_7d.txt (offset from end)
```

### Rule 3: Key Metrics to Extract

Look for these patterns in the output:

```
Total Signals: N
Win Rate: X.X%
Profit Factor: X.XX
Expectancy: X.XX pips
Total PnL: X.XX pips
Winners: N / Losers: N
```

### Rule 4: For Multiple Pair Backtests

Run each pair separately and in parallel when possible:

```bash
# Run 3 pairs in parallel (3 separate Bash calls with run_in_background)
docker exec task-worker python -u /app/forex_scanner/bt.py EURUSD 14 > /tmp/bt_eurusd.txt 2>&1
docker exec task-worker python -u /app/forex_scanner/bt.py GBPUSD 14 > /tmp/bt_gbpusd.txt 2>&1
docker exec task-worker python -u /app/forex_scanner/bt.py USDJPY 14 > /tmp/bt_usdjpy.txt 2>&1
```

### Rule 5: Timeout

Set `timeout: 300000` (5 minutes) for single-pair backtests, `timeout: 600000` (10 minutes) for multi-pair or long periods (30+ days).

### Rule 6: Don't Use `-it` When Redirecting

When redirecting to a file, drop the `-it` flags (they cause issues with non-interactive execution):

```bash
# CORRECT for background/redirect
docker exec task-worker python -u /app/forex_scanner/bt.py EURUSD 7 > /tmp/bt_output.txt 2>&1

# CORRECT for interactive (no redirect)
docker exec -it task-worker python -u /app/forex_scanner/bt.py EURUSD 7
```

---

## Output File Naming Convention

Use consistent naming so results are easy to find:

```
/tmp/bt_{pair}_{days}d.txt          # Basic: /tmp/bt_eurusd_7d.txt
/tmp/bt_{pair}_{days}d_scalp.txt    # Scalp: /tmp/bt_eurusd_7d_scalp.txt
/tmp/bt_{pair}_{days}d_{label}.txt  # Custom: /tmp/bt_eurusd_14d_tight_sl.txt
```

---

## Interpreting Results

### Good Performance Indicators
- Win Rate > 50%
- Profit Factor > 1.5
- Positive expectancy (pips per trade)
- Consistent across multiple timeframes

### Red Flags
- Win Rate < 45%
- Profit Factor < 1.0 (losing money)
- Negative expectancy
- Large drawdowns relative to gains

### Comparing Before/After Changes

When testing parameter changes, always:
1. Run baseline first (current config)
2. Run modified version
3. Compare: Win Rate, PF, Expectancy, Total PnL, Trade Count
4. Ensure trade count is sufficient (>20 trades minimum for statistical relevance)
