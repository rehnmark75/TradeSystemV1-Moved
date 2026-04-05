#!/usr/bin/env python3
"""
USDJPY Parameter Sweep v2.47.0
Finds optimal max_confidence, blocked hours, SL/TP for USDJPY.
Based on 60-day analysis showing:
  - Confidence >= 0.60 has 31.8% WR (inversely predictive)
  - London session (07-12) is -233 P&L (77% of losses)
  - BUY direction is 30.8% WR vs SELL 50.0% WR
  - Breakout regime is 14.3% WR (already blocked in code)
  - Current SL/TP 8/14 produces 0.957 W/L ratio (needs widening)
"""
import subprocess
import json
import re
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

EPIC = 'CS.D.USDJPY.MINI.IP'
DAYS = 30
TIMEFRAME = '5m'

DB_CONF = dict(host='postgres', port=5432, dbname='strategy_config', user='postgres', password='postgres')


def get_conn():
    return psycopg2.connect(**DB_CONF)


def get_current_state():
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            "SELECT fixed_stop_loss_pips, fixed_take_profit_pips, max_confidence, "
            "scalp_blocked_hours_utc, parameter_overrides "
            "FROM smc_simple_pair_overrides WHERE epic=%s", (EPIC,)
        )
        row = cur.fetchone()
        return dict(row) if row else {}


def apply_db_override(sl=None, tp=None, max_conf=None, blocked_hours=None, extra_params=None):
    with get_conn() as conn, conn.cursor() as cur:
        if sl is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET fixed_stop_loss_pips=%s WHERE epic=%s", (sl, EPIC))
        if tp is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET fixed_take_profit_pips=%s WHERE epic=%s", (tp, EPIC))
        if max_conf is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET max_confidence=%s WHERE epic=%s", (max_conf, EPIC))
        if blocked_hours is not None:
            cur.execute("UPDATE smc_simple_pair_overrides SET scalp_blocked_hours_utc=%s WHERE epic=%s", (blocked_hours, EPIC))
        if extra_params is not None:
            for key, val in extra_params.items():
                cur.execute(
                    "UPDATE smc_simple_pair_overrides "
                    "SET parameter_overrides = parameter_overrides || %s::jsonb "
                    "WHERE epic=%s",
                    (json.dumps({key: val}), EPIC)
                )
        conn.commit()


def restore_state(original):
    apply_db_override(
        sl=original.get('fixed_stop_loss_pips'),
        tp=original.get('fixed_take_profit_pips'),
        max_conf=original.get('max_confidence'),
        blocked_hours=original.get('scalp_blocked_hours_utc'),
    )
    # Remove any test params we added
    with get_conn() as conn, conn.cursor() as cur:
        for key in ['scalp_min_adx']:
            cur.execute(
                "UPDATE smc_simple_pair_overrides SET parameter_overrides = parameter_overrides - %s WHERE epic=%s",
                (key, EPIC)
            )
        conn.commit()
    print("  [DB restored]")


def run_backtest(label: str) -> dict:
    cmd = ['python', '/app/forex_scanner/bt.py', 'USDJPY', str(DAYS), '--scalp', '--timeframe', TIMEFRAME]
    print(f"\n  Running: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    metrics = {'label': label}
    patterns = {
        'signals': r'Total Signals:\s*(\d+)',
        'winners': r'Winners:\s*(\d+)',
        'losers': r'Losers:\s*(\d+)',
        'win_rate': r'Win Rate:\s*([\d.]+)%',
        'profit_factor': r'Profit Factor:\s*([\d.]+)',
        'expectancy': r'Expectancy:\s*([-\d.]+)\s*pips',
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
    print(f"=== USDJPY Parameter Sweep ({DAYS}d, {TIMEFRAME}) ===\n")

    original = get_current_state()
    print(f"Original: SL={original.get('fixed_stop_loss_pips')} TP={original.get('fixed_take_profit_pips')} "
          f"MaxConf={original.get('max_confidence')} Hours={original.get('scalp_blocked_hours_utc')}")

    results = []

    # ─── GROUP 1: MAX CONFIDENCE SWEEP ────────────────────────────────────────
    print("\n--- GROUP 1: Max Confidence Sweep ---")

    for mc in [0.64, 0.62, 0.60, 0.59, 0.57, 0.55]:
        restore_state(original)
        apply_db_override(max_conf=mc)
        r = run_backtest(f"1. max_conf={mc}")
        results.append(r); print(fmt(r))

    # ─── GROUP 2: BLOCKED HOURS SWEEP ─────────────────────────────────────────
    print("\n--- GROUP 2: Blocked Hours Sweep (best conf from G1) ---")
    restore_state(original)

    hour_configs = [
        ('0,1,2,3,22,23', 'baseline (no London block)'),
        ('0,1,2,3,7,8,9,22,23', 'block 7-9 (London open)'),
        ('0,1,2,3,7,8,9,10,11,22,23', 'block 7-11 (full London)'),
        ('0,1,2,3,7,8,9,14,15,22,23', 'block 7-9 + 14-15 (overlap)'),
        ('0,1,2,3,8,9,14,15,22,23', 'block 8-9 + 14-15'),
    ]
    for hours, desc in hour_configs:
        restore_state(original)
        apply_db_override(blocked_hours=hours)
        r = run_backtest(f"2. hours={desc}")
        results.append(r); print(fmt(r))

    # ─── GROUP 3: SL/TP SWEEP ─────────────────────────────────────────────────
    print("\n--- GROUP 3: SL/TP Sweep ---")

    sltp_combos = [
        (8,  14, "current 1.75:1"),
        (10, 18, "1.8:1"),
        (12, 20, "1.67:1"),
        (12, 22, "1.83:1"),
        (15, 22, "1.47:1 wide SL"),
        (10, 22, "2.2:1"),
        (10, 16, "1.6:1"),
    ]
    for sl, tp, rr_label in sltp_combos:
        restore_state(original)
        apply_db_override(sl=sl, tp=tp)
        r = run_backtest(f"3. SL={sl} TP={tp} ({rr_label})")
        results.append(r); print(fmt(r))

    # ─── GROUP 4: MIN ADX SWEEP ───────────────────────────────────────────────
    print("\n--- GROUP 4: Min ADX Sweep ---")

    for adx in [0, 18, 20, 22, 25]:
        restore_state(original)
        if adx > 0:
            apply_db_override(extra_params={'scalp_min_adx': float(adx)})
        r = run_backtest(f"4. min_adx={adx}")
        results.append(r); print(fmt(r))

    # ─── RESTORE ──────────────────────────────────────────────────────────────
    restore_state(original)

    # ─── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "="*95)
    print(f"{'Config':<45} {'Sigs':>5} {'WR%':>6} {'PF':>6} {'Exp(p)':>7} {'AvgW':>6} {'AvgL':>6}")
    print("="*95)
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

    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= 20]
    if valid:
        best_pf = max(valid, key=lambda x: x.get('profit_factor', 0))
        best_exp = max(valid, key=lambda x: x.get('expectancy', 0))
        print(f"\n🏆 Best PF:         {best_pf['label']} → PF={best_pf['profit_factor']:.2f}")
        print(f"🏆 Best Expectancy: {best_exp['label']} → {best_exp['expectancy']:.1f} pips/trade")


if __name__ == '__main__':
    main()
