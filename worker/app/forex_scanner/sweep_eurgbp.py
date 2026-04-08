#!/usr/bin/env python3
"""
EURGBP Structured Parameter Sweep
Groups tested in order — each group builds on the best result from the previous.

Group 1: ATR normalization (core R:R fix — AvgWin 4.9p vs AvgLoss 9.1p)
Group 2: Confidence thresholds (filter low-quality entries)
Group 3: MFI filter (momentum quality gate)
Group 4: Swing significance (swing break quality)
Group 5: Best combination

JSONB keys: atr_normalized_sl_tp_enabled, regime_sl_tp_enabled,
            mfi_filter_mode, mfi_min_slope,
            swing_significance_filter_mode, min_swing_significance
Direct cols: min_confidence, max_confidence,
             fixed_stop_loss_pips, fixed_take_profit_pips
"""
import subprocess
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor

EPIC = 'CS.D.EURGBP.MINI.IP'
DAYS = 30
TIMEFRAME = '5m'

DB_CONF = dict(host='postgres', port=5432, dbname='strategy_config', user='postgres', password='postgres')

JSONB_SWEEP_KEYS = [
    'atr_normalized_sl_tp_enabled', 'regime_sl_tp_enabled',
    'mfi_filter_mode', 'mfi_min_slope',
    'swing_significance_filter_mode', 'min_swing_significance',
    'min_breakout_atr_ratio',
]

def get_conn():
    return psycopg2.connect(**DB_CONF)

def get_current_state():
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT min_confidence, max_confidence, "
            "       fixed_stop_loss_pips, fixed_take_profit_pips, parameter_overrides "
            "FROM smc_simple_pair_overrides WHERE epic=%s", (EPIC,)
        )
        return dict(cur.fetchone())

def set_direct(min_conf=None, max_conf=None, sl=None, tp=None):
    with get_conn() as conn, conn.cursor() as cur:
        if min_conf is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET min_confidence=%s WHERE epic=%s", (min_conf, EPIC))
        if max_conf is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET max_confidence=%s WHERE epic=%s", (max_conf, EPIC))
        if sl is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET fixed_stop_loss_pips=%s WHERE epic=%s", (sl, EPIC))
        if tp is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET fixed_take_profit_pips=%s WHERE epic=%s", (tp, EPIC))
        conn.commit()

def set_jsonb(params: dict):
    with get_conn() as conn, conn.cursor() as cur:
        for key, val in params.items():
            cur.execute(
                "UPDATE smc_simple_pair_overrides "
                "SET parameter_overrides = parameter_overrides || %s::jsonb WHERE epic=%s",
                (json.dumps({key: val}), EPIC)
            )
        conn.commit()

def remove_jsonb(keys: list):
    with get_conn() as conn, conn.cursor() as cur:
        for key in keys:
            cur.execute(
                "UPDATE smc_simple_pair_overrides "
                "SET parameter_overrides = parameter_overrides - %s WHERE epic=%s",
                (key, EPIC)
            )
        conn.commit()

def restore_state(original):
    set_direct(
        min_conf=original['min_confidence'],
        max_conf=original['max_confidence'],
        sl=original['fixed_stop_loss_pips'],
        tp=original['fixed_take_profit_pips'],
    )
    remove_jsonb(JSONB_SWEEP_KEYS)
    print("  [DB restored]")

def apply(jsonb: dict = None, min_conf=None, max_conf=None, sl=None, tp=None):
    """Apply a combination of overrides in one call."""
    if jsonb:
        set_jsonb(jsonb)
    set_direct(min_conf=min_conf, max_conf=max_conf, sl=sl, tp=tp)

def run_bt(label: str) -> dict:
    cmd = ['python', '/app/forex_scanner/bt.py', 'EURGBP', str(DAYS), '--timeframe', TIMEFRAME]
    print(f"  Running: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    metrics = {'label': label}
    patterns = {
        'signals':       r'Total Signals:\s*(\d+)',
        'winners':       r'Winners:\s*(\d+)',
        'losers':        r'Losers:\s*(\d+)',
        'win_rate':      r'Win Rate:\s*([\d.]+)%',
        'profit_factor': r'Profit Factor:\s*([\d.]+)',
        'expectancy':    r'Expectancy:\s*([+-]?[\d.]+)\s*pips',
        'avg_win':       r'Average Profit per Winner:\s*([\d.]+)\s*pips',
        'avg_loss':      r'Average Loss per Loser:\s*([\d.]+)\s*pips',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            metrics[key] = float(m.group(1))

    if 'profit_factor' not in metrics:
        metrics['error'] = 'No results'
        print(f"    ERROR: {output[-500:]}")
    else:
        print(f"    → Sigs={int(metrics.get('signals',0))}  WR={metrics.get('win_rate',0):.1f}%  "
              f"PF={metrics.get('profit_factor',0):.2f}  Exp={metrics.get('expectancy',0):+.1f}p  "
              f"AvgW={metrics.get('avg_win',0):.1f}p  AvgL={metrics.get('avg_loss',0):.1f}p")
    return metrics

def fmt_row(m: dict) -> str:
    if 'error' in m:
        return f"{'ERROR':<45} {'':>5} {'':>6} {'':>6} {'':>8} {'':>6} {'':>6}"
    return (
        f"{m['label']:<45} "
        f"{int(m.get('signals',0)):>5} "
        f"{m.get('win_rate',0):>5.1f}% "
        f"{m.get('profit_factor',0):>6.2f} "
        f"{m.get('expectancy',0):>+8.1f} "
        f"{m.get('avg_win',0):>6.1f} "
        f"{m.get('avg_loss',0):>6.1f}"
    )

def best(results, min_signals=15):
    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= min_signals]
    if not valid:
        return None
    return max(valid, key=lambda x: x.get('profit_factor', 0))

def print_summary(results):
    print(f"\n{'Config':<45} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("-" * 90)
    for r in results:
        print(fmt_row(r))
    b = best(results)
    if b:
        print(f"\n  >>> Best PF: [{b['label']}]  PF={b['profit_factor']:.2f}  WR={b.get('win_rate',0):.1f}%")


def main():
    print(f"=== EURGBP Structured Sweep ({DAYS}d, {TIMEFRAME}) ===")
    print("Problem: AvgWin=4.9p vs AvgLoss=9.1p (ATR norm compresses targets)\n")

    original = get_current_state()
    print(f"Original: SL={original['fixed_stop_loss_pips']} TP={original['fixed_take_profit_pips']} "
          f"min_conf={original['min_confidence']} max_conf={original['max_confidence']}")
    print(f"JSONB: {original['parameter_overrides']}\n")

    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1: ATR NORMALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("GROUP 1: ATR Normalization (fixing adverse R:R)")
    print("=" * 70)
    g1 = []

    # 1a: Baseline (ATR norm on, current)
    restore_state(original)
    g1.append(run_bt("1a. Baseline (ATR norm ON, SL=7 TP=10)"))

    # 1b: ATR norm off — fixed SL/TP used directly
    restore_state(original)
    set_jsonb({'atr_normalized_sl_tp_enabled': False})
    g1.append(run_bt("1b. ATR norm OFF (SL=7 TP=10)"))

    # 1c: ATR norm off + regime SL/TP off (remove both scalers)
    restore_state(original)
    set_jsonb({'atr_normalized_sl_tp_enabled': False, 'regime_sl_tp_enabled': False})
    g1.append(run_bt("1c. ATR norm OFF + regime OFF (SL=7 TP=10)"))

    # 1d: ATR norm off + wider TP for better R:R
    restore_state(original)
    set_jsonb({'atr_normalized_sl_tp_enabled': False})
    set_direct(sl=7, tp=12)
    g1.append(run_bt("1d. ATR norm OFF (SL=7 TP=12, 1.71:1)"))

    # 1e: ATR norm off + 2:1 R:R
    restore_state(original)
    set_jsonb({'atr_normalized_sl_tp_enabled': False})
    set_direct(sl=7, tp=14)
    g1.append(run_bt("1e. ATR norm OFF (SL=7 TP=14, 2:1)"))

    # 1f: ATR norm off + wider SL + 2:1 R:R
    restore_state(original)
    set_jsonb({'atr_normalized_sl_tp_enabled': False})
    set_direct(sl=9, tp=18)
    g1.append(run_bt("1f. ATR norm OFF (SL=9 TP=18, 2:1)"))

    all_results.extend(g1)
    print_summary(g1)
    best_g1 = best(g1)

    # Extract best ATR norm settings to carry forward
    g1_jsonb = {'atr_normalized_sl_tp_enabled': False}  # assume off is better
    g1_sl = best_g1['label'].split('SL=')[1].split(' ')[0] if best_g1 and 'SL=' in best_g1['label'] else 7
    g1_tp = best_g1['label'].split('TP=')[1].split(',')[0].split(')')[0] if best_g1 and 'TP=' in best_g1['label'] else 10
    # Use the baseline ATR norm OFF config (1b) as base if regime-off wasn't better
    # Carry forward: ATR norm OFF, best SL/TP from g1
    base_jsonb = dict(g1_jsonb)
    base_sl = float(g1_sl) if best_g1 else 7.0
    base_tp = float(g1_tp) if best_g1 else 10.0

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2: CONFIDENCE THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"GROUP 2: Confidence Thresholds (base: ATR norm OFF, SL={base_sl} TP={base_tp})")
    print("=" * 70)
    g2 = []

    conf_tests = [
        (None, None,  "2a. Baseline conf (default)"),
        (0.52, None,  "2b. min_conf=0.52"),
        (0.55, None,  "2c. min_conf=0.55"),
        (0.58, None,  "2d. min_conf=0.58"),
        (0.60, None,  "2e. min_conf=0.60"),
        (0.62, None,  "2f. min_conf=0.62"),
        (0.55, 0.64,  "2g. min=0.55 max=0.64 (cap high conf)"),
        (0.58, 0.64,  "2h. min=0.58 max=0.64"),
        (0.60, 0.64,  "2i. min=0.60 max=0.64"),
    ]

    for min_c, max_c, label in conf_tests:
        restore_state(original)
        set_jsonb(base_jsonb)
        set_direct(sl=base_sl, tp=base_tp, min_conf=min_c, max_conf=max_c)
        g2.append(run_bt(label))

    all_results.extend(g2)
    print_summary(g2)
    best_g2 = best(g2)

    # Extract best confidence settings
    best_min_conf = None
    best_max_conf = None
    if best_g2:
        lbl = best_g2['label']
        if 'min_conf=' in lbl:
            best_min_conf = float(lbl.split('min_conf=')[1].split()[0])
        if 'min=' in lbl:
            best_min_conf = float(lbl.split('min=')[1].split()[0])
        if 'max=' in lbl:
            best_max_conf = float(lbl.split('max=')[1].split()[0])

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3: MFI FILTER
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"GROUP 3: MFI Filter (base: +Group2 best, min_conf={best_min_conf})")
    print("=" * 70)
    g3 = []

    mfi_tests = [
        ({}, "3a. No MFI filter"),
        ({'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -5.0},  "3b. MFI ACTIVE slope=-5"),
        ({'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -10.0}, "3c. MFI ACTIVE slope=-10"),
        ({'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -15.0}, "3d. MFI ACTIVE slope=-15"),
        ({'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -20.0}, "3e. MFI ACTIVE slope=-20"),
        ({'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': 0.0},   "3f. MFI ACTIVE slope=0 (positive only)"),
    ]

    for extra_jsonb, label in mfi_tests:
        restore_state(original)
        combined = {**base_jsonb, **extra_jsonb}
        set_jsonb(combined)
        set_direct(sl=base_sl, tp=base_tp, min_conf=best_min_conf, max_conf=best_max_conf)
        g3.append(run_bt(label))

    all_results.extend(g3)
    print_summary(g3)
    best_g3 = best(g3)

    # Extract best MFI params
    best_mfi = {}
    if best_g3 and 'slope=' in best_g3['label']:
        slope = float(best_g3['label'].split('slope=')[1].split()[0])
        best_mfi = {'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': slope}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4: SWING SIGNIFICANCE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"GROUP 4: Swing Significance (base: +Group3 best)")
    print("=" * 70)
    g4 = []

    sig_tests = [
        ({}, "4a. No sig filter"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.30}, "4b. Sig ACTIVE min=0.30"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.35}, "4c. Sig ACTIVE min=0.35"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.40}, "4d. Sig ACTIVE min=0.40"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.45}, "4e. Sig ACTIVE min=0.45"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.50}, "4f. Sig ACTIVE min=0.50"),
    ]

    for extra_jsonb, label in sig_tests:
        restore_state(original)
        combined = {**base_jsonb, **best_mfi, **extra_jsonb}
        set_jsonb(combined)
        set_direct(sl=base_sl, tp=base_tp, min_conf=best_min_conf, max_conf=best_max_conf)
        g4.append(run_bt(label))

    all_results.extend(g4)
    print_summary(g4)
    best_g4 = best(g4)

    best_sig = {}
    if best_g4 and 'min=' in best_g4['label']:
        sig_val = float(best_g4['label'].split('min=')[1])
        best_sig = {'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': sig_val}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5: BEST COMBINATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 5: Best Combination")
    print("=" * 70)
    g5 = []

    restore_state(original)
    best_combo = {**base_jsonb, **best_mfi, **best_sig}
    set_jsonb(best_combo)
    set_direct(sl=base_sl, tp=base_tp, min_conf=best_min_conf, max_conf=best_max_conf)
    g5.append(run_bt("5a. Full best combo"))

    # Also test with min_breakout_atr_ratio tighter (stronger breakout required)
    restore_state(original)
    set_jsonb({**best_combo, 'min_breakout_atr_ratio': 0.7})
    set_direct(sl=base_sl, tp=base_tp, min_conf=best_min_conf, max_conf=best_max_conf)
    g5.append(run_bt("5b. Best combo + breakout_atr_ratio=0.7"))

    restore_state(original)
    set_jsonb({**best_combo, 'min_breakout_atr_ratio': 0.9})
    set_direct(sl=base_sl, tp=base_tp, min_conf=best_min_conf, max_conf=best_max_conf)
    g5.append(run_bt("5c. Best combo + breakout_atr_ratio=0.9"))

    all_results.extend(g5)
    print_summary(g5)

    # ══════════════════════════════════════════════════════════════════════════
    # RESTORE + FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    restore_state(original)

    print("\n" + "=" * 90)
    print("FULL RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<45} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("=" * 90)
    for r in all_results:
        print(fmt_row(r))

    overall_best = best(all_results)
    if overall_best:
        print(f"\n  >>> OVERALL BEST: [{overall_best['label']}]")
        print(f"      PF={overall_best['profit_factor']:.2f}  "
              f"WR={overall_best.get('win_rate',0):.1f}%  "
              f"Exp={overall_best.get('expectancy',0):+.1f}p  "
              f"Signals={int(overall_best.get('signals',0))}")


if __name__ == '__main__':
    main()
