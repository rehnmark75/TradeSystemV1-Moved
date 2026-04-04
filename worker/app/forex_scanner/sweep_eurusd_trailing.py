#!/usr/bin/env python3
"""
EURUSD Fixed SL/TP Parameter Sweep
Tests fixed_stop_loss_pips / fixed_take_profit_pips combinations via DB update.

The backtest for SMC_SIMPLE disables trailing entirely when signals carry
risk_pips/reward_pips — it exits at fixed TP/SL only.  So R:R is controlled
solely by these two DB values.  This sweep finds the combination with the
best Profit Factor and Expectancy.

Current EURUSD baseline: SL=12, TP=18
"""
import subprocess
import re
import psycopg2

EPIC = 'CS.D.EURUSD.CEEM.IP'
DAYS = 30
TIMEFRAME = '5m'

DB_CONF = dict(host='postgres', port=5432, dbname='strategy_config',
               user='postgres', password='postgres')

# ── test matrix ──────────────────────────────────────────────────────────────
# (label, sl_pips, tp_pips)  — R:R = tp/sl
CONFIGS = [
    ("Baseline  SL=12 TP=18  R:R=1.50",  12, 18),
    ("SL=10 TP=18  R:R=1.80",            10, 18),
    ("SL=10 TP=20  R:R=2.00",            10, 20),
    ("SL=10 TP=22  R:R=2.20",            10, 22),
    ("SL=12 TP=20  R:R=1.67",            12, 20),
    ("SL=12 TP=22  R:R=1.83",            12, 22),
    ("SL=12 TP=24  R:R=2.00",            12, 24),
    ("SL=15 TP=20  R:R=1.33",            15, 20),
    ("SL=15 TP=25  R:R=1.67",            15, 25),
    ("SL=15 TP=30  R:R=2.00",            15, 30),
    ("SL=8  TP=18  R:R=2.25",             8, 18),
    ("SL=8  TP=20  R:R=2.50",             8, 20),
]


def set_sl_tp(sl, tp):
    with psycopg2.connect(**DB_CONF) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE smc_simple_pair_overrides
                SET fixed_stop_loss_pips = %s, fixed_take_profit_pips = %s
                WHERE epic = %s
            """, (sl, tp, EPIC))
        conn.commit()


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
        metrics['error'] = output[-400:]
    return metrics


def fmt(m):
    if 'error' in m:
        return f"  ERROR: {m['error'][:120]}"
    wins = int(m.get('winners', 0))
    losses = int(m.get('losers', 0))
    total = wins + losses
    wr = (wins / total * 100) if total else 0
    rr = m.get('avg_win', 0) / max(m.get('avg_loss', 1), 1)
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
    print(f"=== EURUSD Fixed SL/TP Sweep ({DAYS}d {TIMEFRAME}) ===\n")
    print("Sweeping fixed_stop_loss_pips / fixed_take_profit_pips in DB\n")
    print("NOTE: Cache is 60s — each backtest subprocess starts fresh (no cache issue)\n")

    original_sl, original_tp = 12, 18  # baseline to restore

    results = []
    try:
        for cfg in CONFIGS:
            label, sl, tp = cfg
            print(f"  Running: {label}")
            set_sl_tp(sl, tp)
            m = run_backtest(label)
            results.append(m)
            print(fmt(m))
    finally:
        set_sl_tp(original_sl, original_tp)
        print("\n  [DB restored to SL=12 TP=18]")

    # Summary table
    print("\n" + "="*115)
    print(f"{'Config':<38} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp':>6} {'AvgW':>7} {'AvgL':>7} {'R:R':>6}")
    print("="*115)
    for m in results:
        if 'error' not in m:
            wins = int(m.get('winners', 0))
            losses = int(m.get('losers', 0))
            total = wins + losses
            wr = (wins / total * 100) if total else 0
            rr = m.get('avg_win', 0) / max(m.get('avg_loss', 1), 1)
            print(
                f"{m['label']:<38} "
                f"{int(m.get('signals',0)):>5} "
                f"{wr:>5.1f}% "
                f"{m.get('profit_factor',0):>6.2f} "
                f"{m.get('expectancy',0):>6.1f} "
                f"{m.get('avg_win',0):>7.1f} "
                f"{m.get('avg_loss',0):>7.1f} "
                f"{rr:>6.2f}"
            )

    valid = [m for m in results if 'error' not in m and m.get('signals', 0) >= 20]
    if valid:
        best_pf  = max(valid, key=lambda x: x.get('profit_factor', 0))
        best_exp = max(valid, key=lambda x: x.get('expectancy', 0))
        best_rr  = max(valid, key=lambda x: x.get('avg_win', 0) / max(x.get('avg_loss', 1), 1))
        print(f"\nBest PF:         {best_pf['label']}  -> PF={best_pf['profit_factor']:.2f}")
        print(f"Best Expectancy: {best_exp['label']}  -> Exp={best_exp['expectancy']:.1f} pips/trade")
        print(f"Best R:R:        {best_rr['label']}  -> R:R={best_rr.get('avg_win',0)/max(best_rr.get('avg_loss',1),1):.2f}")


if __name__ == '__main__':
    main()
