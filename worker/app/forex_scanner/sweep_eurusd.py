#!/usr/bin/env python3
"""
EURUSD Parameter Sweep v2.41.0
Finds optimal MFI slope, swing significance, and SL/TP for EURUSD.
All overrides applied at epic level via smc_simple_pair_overrides.parameter_overrides.
"""
import subprocess
import json
import re
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

EPIC = 'CS.D.EURUSD.CEEM.IP'
DAYS = 30
TIMEFRAME = '5m'

DB_CONF = dict(host='postgres', port=5432, dbname='strategy_config', user='postgres', password='postgres')

def get_conn():
    return psycopg2.connect(**DB_CONF)

def get_current_state():
    """Snapshot current EURUSD DB settings."""
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT fixed_stop_loss_pips, fixed_take_profit_pips, parameter_overrides "
            "FROM smc_simple_pair_overrides WHERE epic=%s", (EPIC,)
        )
        row = cur.fetchone()
        return dict(row) if row else {}

def apply_db_override(sl=None, tp=None, extra_params: dict = None):
    """Update EURUSD pair override row. Only changes specified fields."""
    with get_conn() as conn, conn.cursor() as cur:
        if sl is not None:
            cur.execute(
                "UPDATE smc_simple_pair_overrides SET fixed_stop_loss_pips=%s WHERE epic=%s",
                (sl, EPIC)
            )
        if tp is not None:
            cur.execute(
                "UPDATE smc_simple_pair_overrides SET fixed_take_profit_pips=%s WHERE epic=%s",
                (tp, EPIC)
            )
        if extra_params is not None:
            # Merge into existing parameter_overrides JSONB
            for key, val in extra_params.items():
                cur.execute(
                    "UPDATE smc_simple_pair_overrides "
                    "SET parameter_overrides = parameter_overrides || %s::jsonb "
                    "WHERE epic=%s",
                    (json.dumps({key: val}), EPIC)
                )
        conn.commit()

def remove_db_params(keys: list):
    """Remove keys from parameter_overrides JSONB."""
    with get_conn() as conn, conn.cursor() as cur:
        for key in keys:
            cur.execute(
                "UPDATE smc_simple_pair_overrides "
                "SET parameter_overrides = parameter_overrides - %s "
                "WHERE epic=%s",
                (key, EPIC)
            )
        conn.commit()

def restore_state(original):
    """Restore original DB state."""
    apply_db_override(
        sl=original.get('fixed_stop_loss_pips'),
        tp=original.get('fixed_take_profit_pips')
    )
    remove_db_params([
        'mfi_filter_mode', 'mfi_min_slope',
        'swing_significance_filter_mode', 'min_swing_significance'
    ])
    print("  [DB restored]")

def run_backtest(label: str, extra_overrides: dict = None) -> dict:
    """Run backtest and return parsed performance metrics."""
    cmd = [
        'python', '/app/forex_scanner/bt.py',
        'EURUSD', str(DAYS),
        '--timeframe', TIMEFRAME,
    ]
    if extra_overrides:
        for k, v in extra_overrides.items():
            cmd += ['--override', f'{k}={v}']

    print(f"\n  Running: {label}")
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=600
    )
    output = result.stdout + result.stderr

    metrics = {'label': label}
    patterns = {
        'signals': r'Total Signals:\s*(\d+)',
        'winners': r'Winners:\s*(\d+)',
        'losers': r'Losers:\s*(\d+)',
        'win_rate': r'Win Rate:\s*([\d.]+)%',
        'profit_factor': r'Profit Factor:\s*([\d.]+)',
        'expectancy': r'Expectancy:\s*([\d.]+)\s*pips',
        'avg_win': r'Average Profit per Winner:\s*([\d.]+)\s*pips',
        'avg_loss': r'Average Loss per Loser:\s*([\d.]+)\s*pips',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            metrics[key] = float(m.group(1))

    if 'profit_factor' not in metrics:
        metrics['error'] = 'No results found'
        print(f"    ERROR: {output[-500:]}")

    return metrics

def fmt(metrics: dict) -> str:
    if 'error' in metrics:
        return f"  ERROR: {metrics['error']}"
    return (
        f"  Signals={int(metrics.get('signals',0)):3d}  "
        f"WR={metrics.get('win_rate',0):.1f}%  "
        f"PF={metrics.get('profit_factor',0):.2f}  "
        f"Exp={metrics.get('expectancy',0):.1f}p  "
        f"AvgW={metrics.get('avg_win',0):.1f}p  "
        f"AvgL={metrics.get('avg_loss',0):.1f}p"
    )


def main():
    print(f"=== EURUSD Parameter Sweep ({DAYS}d, {TIMEFRAME}) ===\n")

    original = get_current_state()
    print(f"Original state: SL={original.get('fixed_stop_loss_pips')} TP={original.get('fixed_take_profit_pips')}")

    results = []

    # ─── GROUP 1: FILTER COMBINATIONS (SL=12, TP=18) ──────────────────────────
    print("\n--- GROUP 1: Filter Mode Combinations (SL=12 TP=18) ---")

    # 1a. Baseline (both MONITORING)
    restore_state(original)
    r = run_backtest("1a. Baseline: both MONITORING")
    results.append(r); print(fmt(r))

    # 1b. Sig ACTIVE only (slope=0.40, MFI monitoring)
    apply_db_override(extra_params={
        'swing_significance_filter_mode': 'ACTIVE',
        'min_swing_significance': 0.40,
        'mfi_filter_mode': 'MONITORING',
    })
    r = run_backtest("1b. Sig ACTIVE only (min=0.40)")
    results.append(r); print(fmt(r))

    # 1c. MFI ACTIVE only (slope=-15, Sig monitoring)
    apply_db_override(extra_params={
        'swing_significance_filter_mode': 'MONITORING',
        'mfi_filter_mode': 'ACTIVE',
        'mfi_min_slope': -15.0,
    })
    r = run_backtest("1c. MFI ACTIVE only (slope=-15)")
    results.append(r); print(fmt(r))

    # 1d. Both ACTIVE: slope=-15, sig=0.40
    apply_db_override(extra_params={
        'swing_significance_filter_mode': 'ACTIVE',
        'min_swing_significance': 0.40,
        'mfi_filter_mode': 'ACTIVE',
        'mfi_min_slope': -15.0,
    })
    r = run_backtest("1d. Both ACTIVE: slope=-15, sig=0.40")
    results.append(r); print(fmt(r))

    # 1e. Both ACTIVE: slope=-20, sig=0.40
    apply_db_override(extra_params={
        'swing_significance_filter_mode': 'ACTIVE',
        'min_swing_significance': 0.40,
        'mfi_filter_mode': 'ACTIVE',
        'mfi_min_slope': -20.0,
    })
    r = run_backtest("1e. Both ACTIVE: slope=-20, sig=0.40")
    results.append(r); print(fmt(r))

    # 1f. Both ACTIVE: slope=-15, sig=0.45
    apply_db_override(extra_params={
        'swing_significance_filter_mode': 'ACTIVE',
        'min_swing_significance': 0.45,
        'mfi_filter_mode': 'ACTIVE',
        'mfi_min_slope': -15.0,
    })
    r = run_backtest("1f. Both ACTIVE: slope=-15, sig=0.45")
    results.append(r); print(fmt(r))

    # 1g. Both ACTIVE: slope=-15, sig=0.35
    apply_db_override(extra_params={
        'swing_significance_filter_mode': 'ACTIVE',
        'min_swing_significance': 0.35,
        'mfi_filter_mode': 'ACTIVE',
        'mfi_min_slope': -15.0,
    })
    r = run_backtest("1g. Both ACTIVE: slope=-15, sig=0.35")
    results.append(r); print(fmt(r))

    # ─── GROUP 2: SL/TP SWEEP (best filters from group 1) ─────────────────────
    print("\n--- GROUP 2: SL/TP Ratios (Both ACTIVE slope=-15 sig=0.40) ---")

    best_filters = {
        'swing_significance_filter_mode': 'ACTIVE',
        'min_swing_significance': 0.40,
        'mfi_filter_mode': 'ACTIVE',
        'mfi_min_slope': -15.0,
    }

    sltp_combos = [
        (10, 20, "2:1 R:R"),
        (12, 24, "2:1 R:R"),
        (10, 25, "2.5:1 R:R"),
        (12, 30, "2.5:1 R:R"),
        (8,  20, "2.5:1 R:R tight SL"),
        (10, 22, "2.2:1 R:R"),
    ]
    for sl, tp, rr_label in sltp_combos:
        apply_db_override(sl=sl, tp=tp, extra_params=best_filters)
        r = run_backtest(f"2. SL={sl} TP={tp} ({rr_label})")
        results.append(r); print(fmt(r))

    # ─── RESTORE ──────────────────────────────────────────────────────────────
    restore_state(original)

    # ─── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'Config':<45} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>7} {'AvgW':>6} {'AvgL':>6}")
    print("="*90)
    for r in results:
        if 'error' not in r:
            print(
                f"{r['label']:<45} "
                f"{int(r.get('signals',0)):>5} "
                f"{r.get('win_rate',0):>5.1f}% "
                f"{r.get('profit_factor',0):>6.2f} "
                f"{r.get('expectancy',0):>7.1f} "
                f"{r.get('avg_win',0):>6.1f} "
                f"{r.get('avg_loss',0):>6.1f}"
            )

    # Find best by profit factor and by expectancy
    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= 30]
    if valid:
        best_pf = max(valid, key=lambda x: x.get('profit_factor', 0))
        best_exp = max(valid, key=lambda x: x.get('expectancy', 0))
        print(f"\n🏆 Best PF:         {best_pf['label']} → PF={best_pf['profit_factor']:.2f}")
        print(f"🏆 Best Expectancy: {best_exp['label']} → {best_exp['expectancy']:.1f} pips/trade")


if __name__ == '__main__':
    main()
