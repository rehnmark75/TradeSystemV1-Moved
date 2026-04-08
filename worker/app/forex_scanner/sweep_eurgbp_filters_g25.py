#!/usr/bin/env python3
"""
EURGBP Filter Sweep - Groups 2-5 continuation
G1 result: Two-pole filter has no effect (PF identical across all variants)
           → two_pole_override = {} (leave at global default)

Base: ATR norm OFF + regime OFF (PF=1.09 with conf 0.60/0.64 from previous state)
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

BASE = {'atr_normalized_sl_tp_enabled': False, 'regime_sl_tp_enabled': False}

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

def set_direct(min_conf=None, max_conf=None):
    with get_conn() as conn, conn.cursor() as cur:
        if min_conf is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET min_confidence=%s WHERE epic=%s", (min_conf, EPIC))
        if max_conf is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET max_confidence=%s WHERE epic=%s", (max_conf, EPIC))
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

def restore(original):
    set_direct(min_conf=original['min_confidence'], max_conf=original['max_confidence'])
    remove_jsonb(JSONB_SWEEP_KEYS)
    set_jsonb(BASE)

def run_bt(label: str) -> dict:
    cmd = ['python', '/app/forex_scanner/bt.py', 'EURGBP', str(DAYS), '--timeframe', TIMEFRAME]
    print(f"  Running: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    metrics = {'label': label}
    for key, pattern in [
        ('signals',       r'Total Signals:\s*(\d+)'),
        ('win_rate',      r'Win Rate:\s*([\d.]+)%'),
        ('profit_factor', r'Profit Factor:\s*([\d.]+)'),
        ('expectancy',    r'Expectancy:\s*([+-]?[\d.]+)\s*pips'),
        ('avg_win',       r'Average Profit per Winner:\s*([\d.]+)\s*pips'),
        ('avg_loss',      r'Average Loss per Loser:\s*([\d.]+)\s*pips'),
    ]:
        m = re.search(pattern, output)
        if m:
            metrics[key] = float(m.group(1))
    if 'profit_factor' not in metrics:
        metrics['error'] = 'No results'
        print(f"    ERROR: {output[-200:]}")
    else:
        print(f"    → Sigs={int(metrics.get('signals',0))}  WR={metrics.get('win_rate',0):.1f}%  "
              f"PF={metrics.get('profit_factor',0):.2f}  Exp={metrics.get('expectancy',0):+.1f}p  "
              f"AvgW={metrics.get('avg_win',0):.1f}p  AvgL={metrics.get('avg_loss',0):.1f}p")
    return metrics

def fmt(m):
    if 'error' in m:
        return f"  {'ERROR':<52}"
    return (f"  {m['label']:<52} {int(m.get('signals',0)):>5} "
            f"{m.get('win_rate',0):>5.1f}% {m.get('profit_factor',0):>6.2f} "
            f"{m.get('expectancy',0):>+8.1f} {m.get('avg_win',0):>6.1f} {m.get('avg_loss',0):>6.1f}")

def best(results, min_sigs=20):
    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= min_sigs]
    return max(valid, key=lambda x: x.get('profit_factor', 0)) if valid else None

def summarize(results, title):
    print(f"\n  --- {title} ---")
    print(f"  {'Config':<52} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("  " + "-" * 95)
    for r in results:
        print(fmt(r))
    b = best(results)
    if b:
        print(f"\n  >>> Best: [{b['label']}]  PF={b['profit_factor']:.2f}  "
              f"Sigs={int(b.get('signals',0))}  Exp={b.get('expectancy',0):+.1f}p")
    return b


def main():
    print(f"=== EURGBP Filter Sweep G2-5 ({DAYS}d, {TIMEFRAME}) ===")
    print(f"G1 result: Two-pole has no effect → leave at default\n")

    original = get_current_state()
    print(f"DB state: SL={original['fixed_stop_loss_pips']} TP={original['fixed_take_profit_pips']} "
          f"min_conf={original['min_confidence']} max_conf={original['max_confidence']}\n")

    all_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2: RSI RANGE
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("GROUP 2: RSI Range (bull/bear entry zones) — global: bull=40-75 bear=25-60")
    print("=" * 70)
    g2 = []

    for extra, label in [
        ({},                                                                                        "2a. Baseline (bull=40-75 bear=25-60)"),
        ({'scalp_rsi_filter_enabled': False},                                                       "2b. RSI filter OFF"),
        ({'scalp_rsi_bull_min': 45, 'scalp_rsi_bull_max': 70, 'scalp_rsi_bear_min': 30, 'scalp_rsi_bear_max': 55}, "2c. bull=45-70 bear=30-55"),
        ({'scalp_rsi_bull_min': 50, 'scalp_rsi_bull_max': 70, 'scalp_rsi_bear_min': 30, 'scalp_rsi_bear_max': 50}, "2d. bull=50-70 bear=30-50 (strict)"),
        ({'scalp_rsi_bull_min': 40, 'scalp_rsi_bull_max': 65, 'scalp_rsi_bear_min': 35, 'scalp_rsi_bear_max': 60}, "2e. bull=40-65 bear=35-60"),
        ({'scalp_rsi_bull_min': 35, 'scalp_rsi_bull_max': 80, 'scalp_rsi_bear_min': 20, 'scalp_rsi_bear_max': 65}, "2f. bull=35-80 bear=20-65 (loose)"),
    ]:
        restore(original)
        set_jsonb({**BASE, **extra})
        g2.append(run_bt(label))

    all_results.extend(g2)
    best_g2 = summarize(g2, "Group 2 Results")
    rsi_range = {}
    if best_g2 and 'bull=' in best_g2['label'] and 'Baseline' not in best_g2['label']:
        lbl = best_g2['label']
        rsi_range = {
            'scalp_rsi_bull_min': int(lbl.split('bull=')[1].split('-')[0]),
            'scalp_rsi_bull_max': int(lbl.split('bull=')[1].split('-')[1].split()[0]),
            'scalp_rsi_bear_min': int(lbl.split('bear=')[1].split('-')[0]),
            'scalp_rsi_bear_max': int(lbl.split('bear=')[1].split('-')[1].split()[0]),
        }
    elif best_g2 and 'OFF' in best_g2['label']:
        rsi_range = {'scalp_rsi_filter_enabled': False}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3: RSI ENTRY THRESHOLDS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 3: RSI Entry Thresholds (block overbought/oversold at entry)")
    print("=" * 70)
    g3 = []

    for extra, label in [
        ({},                                                               "3a. Baseline (disabled, buy<=100 sell>=0)"),
        ({'scalp_entry_rsi_buy_max': 75, 'scalp_entry_rsi_sell_min': 25}, "3b. buy<=75 sell>=25"),
        ({'scalp_entry_rsi_buy_max': 70, 'scalp_entry_rsi_sell_min': 30}, "3c. buy<=70 sell>=30"),
        ({'scalp_entry_rsi_buy_max': 65, 'scalp_entry_rsi_sell_min': 35}, "3d. buy<=65 sell>=35"),
        ({'scalp_entry_rsi_buy_max': 60, 'scalp_entry_rsi_sell_min': 40}, "3e. buy<=60 sell>=40"),
    ]:
        restore(original)
        set_jsonb({**BASE, **rsi_range, **extra})
        g3.append(run_bt(label))

    all_results.extend(g3)
    best_g3 = summarize(g3, "Group 3 Results")
    rsi_entry = {}
    if best_g3 and 'buy<=' in best_g3['label'] and 'Baseline' not in best_g3['label']:
        lbl = best_g3['label']
        rsi_entry = {
            'scalp_entry_rsi_buy_max': float(lbl.split('buy<=')[1].split()[0]),
            'scalp_entry_rsi_sell_min': float(lbl.split('sell>=')[1].split()[0]),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4: MACD FILTER
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 4: MACD Filter")
    print("=" * 70)
    g4 = []

    for extra, label in [
        ({},                                   "4a. Baseline (MACD global default)"),
        ({'scalp_macd_filter_enabled': False},  "4b. MACD OFF"),
        ({'scalp_macd_filter_enabled': True},   "4c. MACD ON"),
    ]:
        restore(original)
        set_jsonb({**BASE, **rsi_range, **rsi_entry, **extra})
        g4.append(run_bt(label))

    all_results.extend(g4)
    best_g4 = summarize(g4, "Group 4 Results")
    macd = {}
    if best_g4 and 'OFF' in best_g4['label']:
        macd = {'scalp_macd_filter_enabled': False}
    elif best_g4 and 'ON' in best_g4['label']:
        macd = {'scalp_macd_filter_enabled': True}

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5: BEST COMBINATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GROUP 5: Best Combination")
    print("=" * 70)
    g5 = []

    full = {**BASE, **rsi_range, **rsi_entry, **macd}

    restore(original)
    set_jsonb(full)
    g5.append(run_bt("5a. Full compounded best"))

    # Clean reference: just base (ATR/regime off, no filter changes)
    restore(original)
    set_jsonb(BASE)
    g5.append(run_bt("5b. Base only (ATR/regime off)"))

    all_results.extend(g5)
    summarize(g5, "Group 5 Results")

    # ══════════════════════════════════════════════════════════════════════════
    # RESTORE + FULL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    restore(original)
    print("\n  [DB restored]")

    print("\n" + "=" * 100)
    print("FULL SUMMARY (G2-5)")
    print("=" * 100)
    print(f"  {'Config':<52} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>8} {'AvgW':>6} {'AvgL':>6}")
    print("  " + "=" * 95)
    for r in all_results:
        print(fmt(r))

    ob = best(all_results)
    if ob:
        print(f"\n  >>> OVERALL BEST: [{ob['label']}]  PF={ob['profit_factor']:.2f}  "
              f"Sigs={int(ob.get('signals',0))}  Exp={ob.get('expectancy',0):+.1f}p/trade")
        print(f"\n  Filter overrides to apply:")
        if rsi_range:  print(f"    RSI range:  {rsi_range}")
        if rsi_entry:  print(f"    RSI entry:  {rsi_entry}")
        if macd:       print(f"    MACD:       {macd}")
        if not any([rsi_range, rsi_entry, macd]):
            print(f"    (none — base settings already optimal)")


if __name__ == '__main__':
    main()
