#!/usr/bin/env python3
"""
EURGBP RSI & Two-Pole Filter Sweep
Base: ATR norm OFF + regime SL/TP OFF (PF=1.13, 258 signals)

Group 1: Two-pole filter — on/off and threshold tightness
Group 2: RSI range — tighten/loosen bull/bear ranges
Group 3: RSI entry thresholds — block overbought/oversold entries
Group 4: MACD filter — on/off
Group 5: Best combination
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
    'scalp_two_pole_filter_enabled', 'scalp_two_pole_bull_threshold', 'scalp_two_pole_bear_threshold',
    'scalp_rsi_filter_enabled', 'scalp_rsi_bull_min', 'scalp_rsi_bull_max',
    'scalp_rsi_bear_min', 'scalp_rsi_bear_max',
    'scalp_entry_rsi_buy_max', 'scalp_entry_rsi_sell_min',
    'scalp_macd_filter_enabled',
]

# Base from previous sweep (best G1 result)
BASE = {
    'atr_normalized_sl_tp_enabled': False,
    'regime_sl_tp_enabled': False,
}

def get_conn():
    return psycopg2.connect(**DB_CONF)

def get_current_state():
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT min_confidence, max_confidence, fixed_stop_loss_pips, "
            "       fixed_take_profit_pips, parameter_overrides "
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
    # Re-apply base settings
    set_jsonb(BASE)

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
        return f"{'ERROR':<52}"
    return (
        f"{m['label']:<52} "
        f"{int(m.get('signals',0)):>5} "
        f"{m.get('win_rate',0):>5.1f}% "
        f"{m.get('profit_factor',0):>6.2f} "
        f"{m.get('expectancy',0):>+8.1f} "
        f"{m.get('avg_win',0):>6.1f} "
        f"{m.get('avg_loss',0):>6.1f}"
    )

def best(results, min_signals=20):
    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= min_signals]
    return max(valid, key=lambda x: x.get('profit_factor', 0)) if valid else None

def print_summary(results, title=''):
    if title:
        print(f"\n  --- {title} ---")
    print(f"  {'Config':<52} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("  " + "-" * 95)
    for r in results:
        print("  " + fmt_row(r))
    b = best(results)
    if b:
        print(f"\n  >>> Best: [{b['label']}]  PF={b['profit_factor']:.2f}  "
              f"WR={b.get('win_rate',0):.1f}%  Sigs={int(b.get('signals',0))}  Exp={b.get('expectancy',0):+.1f}p")
    return b


def main():
    print(f"=== EURGBP RSI & Two-Pole Filter Sweep ({DAYS}d, {TIMEFRAME}) ===")
    print(f"Base: ATR norm OFF + regime OFF (PF=1.13 baseline)\n")

    original = get_current_state()
    print(f"Original DB: SL={original['fixed_stop_loss_pips']} TP={original['fixed_take_profit_pips']} "
          f"min_conf={original['min_confidence']} max_conf={original['max_confidence']}")
    print(f"Global defaults: two-pole ON ±0.30 | RSI ON bull=40-75 bear=25-60\n")

    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1: TWO-POLE FILTER
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("GROUP 1: Two-Pole Filter (on/off + threshold tightness)")
    print("=" * 70)
    g1 = []

    two_pole_tests = [
        ({},                                                                          "1a. Baseline (two-pole ON ±0.30)"),
        ({'scalp_two_pole_filter_enabled': False},                                    "1b. Two-pole OFF"),
        ({'scalp_two_pole_bull_threshold': -0.10, 'scalp_two_pole_bear_threshold': 0.10}, "1c. Two-pole ±0.10 (tight)"),
        ({'scalp_two_pole_bull_threshold': -0.20, 'scalp_two_pole_bear_threshold': 0.20}, "1d. Two-pole ±0.20"),
        ({'scalp_two_pole_bull_threshold': -0.40, 'scalp_two_pole_bear_threshold': 0.40}, "1e. Two-pole ±0.40 (loose)"),
        ({'scalp_two_pole_bull_threshold': -0.50, 'scalp_two_pole_bear_threshold': 0.50}, "1f. Two-pole ±0.50 (loosest)"),
    ]

    for extra, label in two_pole_tests:
        restore_state(original)
        set_jsonb({**BASE, **extra})
        g1.append(run_bt(label))

    all_results.extend(g1)
    best_g1 = print_summary(g1, "Group 1 Results")
    best_two_pole = {k: v for k, v in (best_g1['label'].split('±') and {}).items()} if best_g1 else {}
    # Determine best two-pole override from best result label
    two_pole_override = {}
    if best_g1:
        lbl = best_g1['label']
        if 'OFF' in lbl:
            two_pole_override = {'scalp_two_pole_filter_enabled': False}
        elif '±' in lbl:
            thresh = float(lbl.split('±')[1].split()[0].rstrip(')'))
            two_pole_override = {'scalp_two_pole_bull_threshold': -thresh, 'scalp_two_pole_bear_threshold': thresh}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2: RSI RANGE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 2: RSI Range (bull/bear entry zones)")
    print("=" * 70)
    g2 = []

    rsi_range_tests = [
        ({},                                                                                        "2a. Baseline (bull=40-75, bear=25-60)"),
        ({'scalp_rsi_filter_enabled': False},                                                       "2b. RSI filter OFF"),
        ({'scalp_rsi_bull_min': 45, 'scalp_rsi_bull_max': 70, 'scalp_rsi_bear_min': 30, 'scalp_rsi_bear_max': 55}, "2c. bull=45-70 bear=30-55 (tighter)"),
        ({'scalp_rsi_bull_min': 50, 'scalp_rsi_bull_max': 70, 'scalp_rsi_bear_min': 30, 'scalp_rsi_bear_max': 50}, "2d. bull=50-70 bear=30-50 (strict)"),
        ({'scalp_rsi_bull_min': 40, 'scalp_rsi_bull_max': 65, 'scalp_rsi_bear_min': 35, 'scalp_rsi_bear_max': 60}, "2e. bull=40-65 bear=35-60 (mid cap)"),
        ({'scalp_rsi_bull_min': 35, 'scalp_rsi_bull_max': 80, 'scalp_rsi_bear_min': 20, 'scalp_rsi_bear_max': 65}, "2f. bull=35-80 bear=20-65 (loose)"),
    ]

    for extra, label in rsi_range_tests:
        restore_state(original)
        set_jsonb({**BASE, **two_pole_override, **extra})
        g2.append(run_bt(label))

    all_results.extend(g2)
    best_g2 = print_summary(g2, "Group 2 Results")
    rsi_range_override = {}
    if best_g2:
        lbl = best_g2['label']
        if 'OFF' in lbl:
            rsi_range_override = {'scalp_rsi_filter_enabled': False}
        elif 'bull=' in lbl:
            bull_min = int(lbl.split('bull=')[1].split('-')[0])
            bull_max = int(lbl.split('bull=')[1].split('-')[1].split()[0])
            bear_min = int(lbl.split('bear=')[1].split('-')[0])
            bear_max = int(lbl.split('bear=')[1].split('-')[1].split()[0])
            rsi_range_override = {
                'scalp_rsi_bull_min': bull_min, 'scalp_rsi_bull_max': bull_max,
                'scalp_rsi_bear_min': bear_min, 'scalp_rsi_bear_max': bear_max,
            }

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3: RSI ENTRY THRESHOLDS (block overbought/oversold)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 3: RSI Entry Thresholds (block extreme RSI at entry)")
    print("=" * 70)
    g3 = []

    rsi_entry_tests = [
        ({},                                                                     "3a. Baseline (entry thresholds disabled)"),
        ({'scalp_entry_rsi_buy_max': 70, 'scalp_entry_rsi_sell_min': 30},       "3b. entry buy<=70 sell>=30"),
        ({'scalp_entry_rsi_buy_max': 65, 'scalp_entry_rsi_sell_min': 35},       "3c. entry buy<=65 sell>=35"),
        ({'scalp_entry_rsi_buy_max': 60, 'scalp_entry_rsi_sell_min': 40},       "3d. entry buy<=60 sell>=40 (neutral zone)"),
        ({'scalp_entry_rsi_buy_max': 75, 'scalp_entry_rsi_sell_min': 25},       "3e. entry buy<=75 sell>=25 (loose)"),
    ]

    for extra, label in rsi_entry_tests:
        restore_state(original)
        set_jsonb({**BASE, **two_pole_override, **rsi_range_override, **extra})
        g3.append(run_bt(label))

    all_results.extend(g3)
    best_g3 = print_summary(g3, "Group 3 Results")
    rsi_entry_override = {}
    if best_g3 and 'buy<=' in best_g3['label']:
        buy_max = float(best_g3['label'].split('buy<=')[1].split()[0])
        sell_min = float(best_g3['label'].split('sell>=')[1].split()[0].rstrip(')'))
        rsi_entry_override = {'scalp_entry_rsi_buy_max': buy_max, 'scalp_entry_rsi_sell_min': sell_min}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4: MACD FILTER
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 4: MACD Filter")
    print("=" * 70)
    g4 = []

    macd_tests = [
        ({},                                        "4a. Baseline (MACD as global)"),
        ({'scalp_macd_filter_enabled': False},       "4b. MACD OFF"),
        ({'scalp_macd_filter_enabled': True},        "4c. MACD ON (force enable)"),
    ]

    for extra, label in macd_tests:
        restore_state(original)
        set_jsonb({**BASE, **two_pole_override, **rsi_range_override, **rsi_entry_override, **extra})
        g4.append(run_bt(label))

    all_results.extend(g4)
    best_g4 = print_summary(g4, "Group 4 Results")
    macd_override = {}
    if best_g4 and 'OFF' in best_g4['label']:
        macd_override = {'scalp_macd_filter_enabled': False}
    elif best_g4 and 'force' in best_g4['label']:
        macd_override = {'scalp_macd_filter_enabled': True}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5: BEST COMBINATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 5: Best Combination")
    print("=" * 70)
    g5 = []

    full_best = {**BASE, **two_pole_override, **rsi_range_override, **rsi_entry_override, **macd_override}

    restore_state(original)
    set_jsonb(full_best)
    g5.append(run_bt("5a. Full compounded best"))

    # Also test: previous sweep best (ATR/regime off only) for comparison
    restore_state(original)
    set_jsonb(BASE)
    g5.append(run_bt("5b. G1-only base (ATR/regime off, no filter changes)"))

    all_results.extend(g5)
    print_summary(g5, "Group 5 Results")

    # ══════════════════════════════════════════════════════════════════════════
    # RESTORE + FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    restore_state(original)
    print("\n  [DB restored to original + base ATR/regime off]")

    print("\n" + "=" * 100)
    print("FULL SUMMARY")
    print("=" * 100)
    print(f"  {'Config':<52} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("  " + "=" * 95)
    for r in all_results:
        print("  " + fmt_row(r))

    overall_best = best(all_results)
    if overall_best:
        print(f"\n  >>> OVERALL BEST: [{overall_best['label']}]")
        print(f"      PF={overall_best['profit_factor']:.2f}  "
              f"Signals={int(overall_best.get('signals',0))}  "
              f"Exp={overall_best.get('expectancy',0):+.1f}p/trade")
        print(f"\n  Applied overrides:")
        print(f"      two_pole: {two_pole_override or 'no change'}")
        print(f"      rsi_range: {rsi_range_override or 'no change'}")
        print(f"      rsi_entry: {rsi_entry_override or 'no change'}")
        print(f"      macd: {macd_override or 'no change'}")


if __name__ == '__main__':
    main()
