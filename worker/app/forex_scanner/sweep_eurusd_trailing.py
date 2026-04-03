#!/usr/bin/env python3
"""
EURUSD Trailing Stop Parameter Sweep
Tests various BE trigger / stage lock combinations to maximise R:R.
Modifies config_trailing_stops.py EURUSD entry, runs 30d backtest, restores.
"""
import subprocess
import re
import shutil
import os

CONFIG_FILE = '/app/forex_scanner/config_trailing_stops.py'
CONFIG_HOST = CONFIG_FILE  # script runs inside container, write directly
DAYS = 30
TIMEFRAME = '5m'

# ── test matrix ──────────────────────────────────────────────────────────────
# Each entry: (label, be, s1_trigger, s1_lock, s2_trigger, s2_lock, min_trail)
CONFIGS = [
    # label                          BE   S1t  S1l  S2t  S2l  trail
    ("Baseline (current)",           20,  25,  10,  35,  22,  15),
    ("Tight BE=10",                  10,  18,   8,  28,  16,  10),
    ("Tight BE=12",                  12,  20,  10,  30,  18,  12),
    ("Tight BE=15",                  15,  22,  10,  32,  20,  13),
    ("Loose BE=25",                  25,  30,  12,  40,  25,  18),
    ("Tight BE=15 + big locks",      15,  25,  15,  35,  28,  13),
    ("Wide stages (let run)",        20,  32,  15,  48,  32,  15),
    ("Tight BE=15 + wide stages",    15,  30,  15,  45,  30,  13),
    ("BE=18 + aggressive S2 lock",   18,  25,  12,  38,  28,  15),
    ("Very wide (BE=25 + big run)",  25,  35,  15,  52,  35,  18),
]

EURUSD_TEMPLATE = """    'CS.D.EURUSD.CEEM.IP': {{
        'stage1_trigger_points': {s1t},
        'stage1_lock_points': {s1l},
        'stage2_trigger_points': {s2t},
        'stage2_lock_points': {s2l},
        'stage3_trigger_points': {s3t},
        'stage3_atr_multiplier': 0.8,
        'stage3_min_distance': 4,
        'min_trail_distance': {trail},
        'break_even_trigger_points': {be},
    }},"""


def patch_config(be, s1t, s1l, s2t, s2l, trail):
    """Read config file, replace EURUSD section, write back."""
    with open(CONFIG_HOST, 'r') as f:
        content = f.read()

    new_section = EURUSD_TEMPLATE.format(
        be=be, s1t=s1t, s1l=s1l, s2t=s2t, s2l=s2l,
        s3t=s2t + 10,  # stage3 = stage2 + 10
        trail=trail
    )

    # Replace from 'CS.D.EURUSD.CEEM.IP': { ... }, (first pair entry)
    patched = re.sub(
        r"    'CS\.D\.EURUSD\.CEEM\.IP': \{.*?\},",
        new_section,
        content,
        count=1,
        flags=re.DOTALL
    )
    with open(CONFIG_HOST, 'w') as f:
        f.write(patched)


def copy_to_container():
    pass  # script runs inside container — writes directly to CONFIG_HOST = CONFIG_FILE


def run_backtest(label):
    result = subprocess.run(
        ['python', '/app/forex_scanner/bt.py', 'EURUSD', str(DAYS),
         '--timeframe', TIMEFRAME],
        capture_output=True, text=True, timeout=600
    )
    output = result.stdout + result.stderr

    metrics = {'label': label}
    for key, pat in [
        ('signals',        r'Total Signals:\s*(\d+)'),
        ('winners',        r'Winners:\s*(\d+)\s'),
        ('losers',         r'Losers:\s*(\d+)\s'),
        ('profit_factor',  r'Profit Factor:\s*([\d.]+)'),
        ('expectancy',     r'Expectancy:\s*([\d.]+)\s*pips'),
        ('avg_win',        r'Average Profit per Winner:\s*([\d.]+)\s*pips'),
        ('avg_loss',       r'Average Loss per Loser:\s*([\d.]+)\s*pips'),
    ]:
        m = re.search(pat, output)
        if m:
            metrics[key] = float(m.group(1))

    if 'profit_factor' not in metrics:
        metrics['error'] = output[-300:]
    return metrics


def fmt(m):
    if 'error' in m:
        return f"  ERROR"
    wins = int(m.get('winners', 0))
    losses = int(m.get('losers', 0))
    total = wins + losses
    wr = (wins / total * 100) if total else 0
    rr = m.get('avg_win', 0) / m.get('avg_loss', 1)
    return (
        f"  Sigs={int(m.get('signals',0)):3d}  "
        f"WR={wr:.1f}%  "
        f"PF={m.get('profit_factor',0):.2f}  "
        f"Exp={m.get('expectancy',0):.1f}p  "
        f"AvgW={m.get('avg_win',0):.1f}p  "
        f"AvgL={m.get('avg_loss',0):.1f}p  "
        f"R:R={rr:.2f}"
    )


def main():
    print(f"=== EURUSD Trailing Stop Sweep ({DAYS}d {TIMEFRAME}) ===\n")
    print("Params: BE=breakeven trigger, S1/S2=stage triggers+locks, Trail=min distance\n")

    # Backup original
    backup = CONFIG_HOST + '.bak'
    shutil.copy2(CONFIG_HOST, backup)
    print(f"Backed up config to {backup}\n")

    results = []
    try:
        for cfg in CONFIGS:
            label, be, s1t, s1l, s2t, s2l, trail = cfg
            print(f"  Running: {label}  (BE={be} S1={s1t}/{s1l} S2={s2t}/{s2l} trail={trail})")
            patch_config(be, s1t, s1l, s2t, s2l, trail)
            copy_to_container()
            m = run_backtest(label)
            results.append(m)
            print(fmt(m))
    finally:
        # Always restore
        shutil.copy2(backup, CONFIG_HOST)
        copy_to_container()
        print("\n  [Config restored]")

    # Summary table
    print("\n" + "="*110)
    print(f"{'Config':<38} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp':>6} {'AvgW':>6} {'AvgL':>6} {'R:R':>6}")
    print("="*110)
    for m in results:
        if 'error' not in m:
            wins = int(m.get('winners', 0))
            losses = int(m.get('losers', 0))
            total = wins + losses
            wr = (wins / total * 100) if total else 0
            rr = m.get('avg_win', 0) / m.get('avg_loss', 1)
            print(
                f"{m['label']:<38} "
                f"{int(m.get('signals',0)):>5} "
                f"{wr:>5.1f}% "
                f"{m.get('profit_factor',0):>6.2f} "
                f"{m.get('expectancy',0):>6.1f} "
                f"{m.get('avg_win',0):>6.1f} "
                f"{m.get('avg_loss',0):>6.1f} "
                f"{rr:>6.2f}"
            )

    valid = [m for m in results if 'error' not in m and m.get('signals', 0) >= 20]
    if valid:
        best_pf  = max(valid, key=lambda x: x.get('profit_factor', 0))
        best_exp = max(valid, key=lambda x: x.get('expectancy', 0))
        best_rr  = max(valid, key=lambda x: x.get('avg_win', 0) / max(x.get('avg_loss', 1), 1))
        print(f"\n🏆 Best PF:         {best_pf['label']}  → PF={best_pf['profit_factor']:.2f}")
        print(f"🏆 Best Expectancy: {best_exp['label']}  → {best_exp['expectancy']:.1f} pips/trade")
        print(f"🏆 Best R:R:        {best_rr['label']}  → {best_rr.get('avg_win',0)/max(best_rr.get('avg_loss',1),1):.2f}")


if __name__ == '__main__':
    main()
