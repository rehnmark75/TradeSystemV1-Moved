#!/usr/bin/env python3
"""
EURGBP Sweep - Groups 4 & 5 continuation
Base from completed groups:
  - ATR norm OFF + regime SL/TP OFF (1c best)
  - min_conf=0.60, max_conf=0.64 (2i best)
  - MFI ACTIVE slope=0 (3f best)

Also tests the Group 1 winner (regime OFF) compounded with Groups 2+3 best.
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

# Best settings from Groups 1-3
BEST_BASE = {
    'atr_normalized_sl_tp_enabled': False,
    'regime_sl_tp_enabled': False,
    'mfi_filter_mode': 'ACTIVE',
    'mfi_min_slope': 0.0,
}
BEST_MIN_CONF = 0.60
BEST_MAX_CONF = 0.64
BEST_SL = 7.0
BEST_TP = 10.0

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
        print(f"    ERROR: {output[-300:]}")
    else:
        print(f"    → Sigs={int(metrics.get('signals',0))}  WR={metrics.get('win_rate',0):.1f}%  "
              f"PF={metrics.get('profit_factor',0):.2f}  Exp={metrics.get('expectancy',0):+.1f}p  "
              f"AvgW={metrics.get('avg_win',0):.1f}p  AvgL={metrics.get('avg_loss',0):.1f}p")
    return metrics

def fmt_row(m: dict) -> str:
    if 'error' in m:
        return f"{'ERROR':<50}"
    return (
        f"{m['label']:<50} "
        f"{int(m.get('signals',0)):>5} "
        f"{m.get('win_rate',0):>5.1f}% "
        f"{m.get('profit_factor',0):>6.2f} "
        f"{m.get('expectancy',0):>+8.1f} "
        f"{m.get('avg_win',0):>6.1f} "
        f"{m.get('avg_loss',0):>6.1f}"
    )

def best(results, min_signals=15):
    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= min_signals]
    return max(valid, key=lambda x: x.get('profit_factor', 0)) if valid else None

def print_summary(results):
    print(f"\n{'Config':<50} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("-" * 95)
    for r in results:
        print(fmt_row(r))
    b = best(results)
    if b:
        print(f"\n  >>> Best: [{b['label']}]  PF={b['profit_factor']:.2f}  WR={b.get('win_rate',0):.1f}%  Exp={b.get('expectancy',0):+.1f}p")


def main():
    print(f"=== EURGBP Sweep Groups 4+5 Continuation ({DAYS}d, {TIMEFRAME}) ===")
    print(f"Base: ATR norm OFF + regime OFF + MFI slope=0 + conf 0.60-0.64\n")

    original = get_current_state()
    print(f"Original: SL={original['fixed_stop_loss_pips']} TP={original['fixed_take_profit_pips']} "
          f"min_conf={original['min_confidence']} max_conf={original['max_confidence']}")

    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4: SWING SIGNIFICANCE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 4: Swing Significance")
    print("=" * 70)
    g4 = []

    sig_tests = [
        ({}, "4a. No sig filter"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.30}, "4b. Sig min=0.30"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.35}, "4c. Sig min=0.35"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.40}, "4d. Sig min=0.40"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.45}, "4e. Sig min=0.45"),
        ({'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.50}, "4f. Sig min=0.50"),
    ]

    for extra_jsonb, label in sig_tests:
        restore_state(original)
        set_jsonb({**BEST_BASE, **extra_jsonb})
        set_direct(sl=BEST_SL, tp=BEST_TP, min_conf=BEST_MIN_CONF, max_conf=BEST_MAX_CONF)
        g4.append(run_bt(label))

    all_results.extend(g4)
    print_summary(g4)
    best_g4 = best(g4)

    best_sig = {}
    if best_g4 and 'min=' in best_g4['label']:
        sig_val = float(best_g4['label'].split('min=')[1].split()[0])
        best_sig = {'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': sig_val}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5: BEST COMBINATIONS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 5: Best Combinations")
    print("=" * 70)
    g5 = []

    full_base = {**BEST_BASE, **best_sig}

    # 5a: Full compounded best
    restore_state(original)
    set_jsonb(full_base)
    set_direct(sl=BEST_SL, tp=BEST_TP, min_conf=BEST_MIN_CONF, max_conf=BEST_MAX_CONF)
    g5.append(run_bt("5a. Full best (ATR/regime off + MFI=0 + sig + conf 0.60-0.64)"))

    # 5b: breakout_atr_ratio=0.7
    restore_state(original)
    set_jsonb({**full_base, 'min_breakout_atr_ratio': 0.7})
    set_direct(sl=BEST_SL, tp=BEST_TP, min_conf=BEST_MIN_CONF, max_conf=BEST_MAX_CONF)
    g5.append(run_bt("5b. Full best + breakout_atr=0.7"))

    # 5c: breakout_atr_ratio=0.9
    restore_state(original)
    set_jsonb({**full_base, 'min_breakout_atr_ratio': 0.9})
    set_direct(sl=BEST_SL, tp=BEST_TP, min_conf=BEST_MIN_CONF, max_conf=BEST_MAX_CONF)
    g5.append(run_bt("5c. Full best + breakout_atr=0.9"))

    # 5d: Try wider TP with best filters
    restore_state(original)
    set_jsonb(full_base)
    set_direct(sl=BEST_SL, tp=14.0, min_conf=BEST_MIN_CONF, max_conf=BEST_MAX_CONF)
    g5.append(run_bt("5d. Full best + TP=14"))

    # 5e: Regime OFF was the biggest G1 winner — test without MFI to see pure regime effect
    restore_state(original)
    set_jsonb({'atr_normalized_sl_tp_enabled': False, 'regime_sl_tp_enabled': False, **best_sig})
    set_direct(sl=BEST_SL, tp=BEST_TP, min_conf=BEST_MIN_CONF, max_conf=BEST_MAX_CONF)
    g5.append(run_bt("5e. ATR/regime off + sig + conf (no MFI)"))

    all_results.extend(g5)
    print_summary(g5)

    # ══════════════════════════════════════════════════════════════════════════
    # RESTORE
    # ══════════════════════════════════════════════════════════════════════════
    restore_state(original)
    print("\n  [DB restored to original]")

    # ══════════════════════════════════════════════════════════════════════════
    # FULL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("GROUPS 4+5 SUMMARY")
    print("=" * 95)
    print(f"{'Config':<50} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("=" * 95)
    for r in all_results:
        print(fmt_row(r))

    # Include prior group bests for context
    prior = [
        {'label': '[G1] ATR norm OFF + regime OFF',         'signals': 258, 'win_rate': 0, 'profit_factor': 1.13, 'expectancy': 0.4,  'avg_win': 6.1, 'avg_loss': 7.1},
        {'label': '[G2] + min_conf=0.60 max_conf=0.64',     'signals': 172, 'win_rate': 0, 'profit_factor': 1.04, 'expectancy': 0.1,  'avg_win': 6.8, 'avg_loss': 7.6},
        {'label': '[G3] + MFI ACTIVE slope=0',              'signals': 89,  'win_rate': 0, 'profit_factor': 1.14, 'expectancy': 0.5,  'avg_win': 7.4, 'avg_loss': 7.6},
    ]
    print("\n--- Prior group bests for reference ---")
    for r in prior:
        print(fmt_row(r))

    overall_best = best(all_results + prior)
    if overall_best:
        print(f"\n  >>> OVERALL BEST: [{overall_best['label']}]")
        print(f"      PF={overall_best['profit_factor']:.2f}  "
              f"Signals={int(overall_best.get('signals',0))}  "
              f"Exp={overall_best.get('expectancy',0):+.1f}p/trade")


if __name__ == '__main__':
    main()
