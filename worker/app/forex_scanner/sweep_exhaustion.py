#!/usr/bin/env python3
"""
EURUSD Impulse Quality Filter Sweep — Fine-tune around 0.50 threshold.
"""
import subprocess
import re
import psycopg2

EPIC = 'CS.D.EURUSD.CEEM.IP'
DAYS = 14
TIMEFRAME = '5m'
DB_CONF = dict(host='postgres', port=5432, dbname='strategy_config',
               user='postgres', password='postgres')

CONFIGS = [
    ("Baseline (MONITORING)",      'MONITORING', 0.35),
    ("ACTIVE score>=0.48",         'ACTIVE',     0.48),
    ("ACTIVE score>=0.50",         'ACTIVE',     0.50),
    ("ACTIVE score>=0.52",         'ACTIVE',     0.52),
    ("ACTIVE score>=0.55",         'ACTIVE',     0.55),
    ("ACTIVE score>=0.58",         'ACTIVE',     0.58),
    ("ACTIVE score>=0.60",         'ACTIVE',     0.60),
]


def set_db(mode, min_score):
    with psycopg2.connect(**DB_CONF) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE smc_simple_pair_overrides
                SET parameter_overrides = parameter_overrides
                    || jsonb_build_object(
                        'impulse_quality_filter_mode', %s,
                        'min_impulse_score', %s::text::numeric
                    )
                WHERE epic = %s
            """, (mode, min_score, EPIC))
        conn.commit()


def run_backtest(label):
    result = subprocess.run(
        ['python', '/app/forex_scanner/bt.py', 'EURUSD', str(DAYS),
         '--timeframe', TIMEFRAME],
        capture_output=True, text=True, timeout=300
    )
    output = result.stdout + result.stderr
    metrics = {'label': label}
    for key, pat in [
        ('signals',       r'Total Signals:\s*(\d+)'),
        ('winners',       r'Winners:\s*(\d+)\s'),
        ('losers',        r'Losers:\s*(\d+)\s'),
        ('profit_factor', r'Profit Factor:\s*([\d.]+)'),
        ('expectancy',    r'Expectancy:\s*([\d.]+)\s*pips'),
        ('avg_win',       r'Average Profit per Winner:\s*([\d.]+)\s*pips'),
        ('avg_loss',      r'Average Loss per Loser:\s*([\d.]+)\s*pips'),
    ]:
        m = re.search(pat, output)
        if m:
            metrics[key] = float(m.group(1))
    metrics['blocked'] = float(output.count('Impulse quality too low'))
    if 'profit_factor' not in metrics:
        metrics['error'] = output[-400:]
    return metrics


def main():
    print(f"=== EURUSD Impulse Quality Fine-Tune ({DAYS}d {TIMEFRAME}) ===\n")
    results = []
    for i, (label, mode, min_score) in enumerate(CONFIGS):
        print(f"  [{i+1}/{len(CONFIGS)}] {label}")
        set_db(mode, min_score)
        m = run_backtest(label)
        results.append(m)
        wins = int(m.get('winners', 0))
        losses = int(m.get('losers', 0))
        total = wins + losses
        wr = (wins / total * 100) if total else 0
        rr = m.get('avg_win', 0) / max(m.get('avg_loss', 1), 1)
        blk = int(m.get('blocked', 0))
        if 'error' in m:
            print(f"  ERROR: {m['error'][:120]}")
        else:
            print(f"  Sigs={int(m.get('signals',0)):3d}  Blk={blk:2d}  WR={wr:.1f}%  PF={m.get('profit_factor',0):.2f}  Exp={m.get('expectancy',0):.1f}p  R:R={rr:.2f}")

    set_db('MONITORING', 0.35)
    print("\n  [DB restored to MONITORING]")

    print("\n" + "="*110)
    print(f"{'Config':<30} {'Sigs':>5} {'Blk':>4} {'WR%':>6} {'PF':>6} {'Exp':>6} {'AvgW':>7} {'AvgL':>7} {'R:R':>6}")
    print("="*110)
    for m in results:
        if 'error' not in m:
            wins = int(m.get('winners', 0))
            losses = int(m.get('losers', 0))
            total = wins + losses
            wr = (wins / total * 100) if total else 0
            rr = m.get('avg_win', 0) / max(m.get('avg_loss', 1), 1)
            print(f"{m['label']:<30} {int(m.get('signals',0)):>5} {int(m.get('blocked',0)):>4} {wr:>5.1f}% {m.get('profit_factor',0):>6.2f} {m.get('expectancy',0):>6.1f} {m.get('avg_win',0):>7.1f} {m.get('avg_loss',0):>7.1f} {rr:>6.2f}")

    valid = [m for m in results if 'error' not in m and m.get('signals', 0) >= 5]
    if len(valid) >= 2:
        baseline = valid[0]
        best = max(valid[1:], key=lambda x: x.get('profit_factor', 0))
        print(f"\nBaseline: PF={baseline.get('profit_factor',0):.2f}  Exp={baseline.get('expectancy',0):.1f}p  Sigs={int(baseline.get('signals',0))}")
        print(f"Best:     {best['label']}  -> PF={best['profit_factor']:.2f} ({best['profit_factor']-baseline.get('profit_factor',0):+.2f})  Exp={best['expectancy']:.1f}p  Sigs={int(best.get('signals',0))}")


if __name__ == '__main__':
    main()
