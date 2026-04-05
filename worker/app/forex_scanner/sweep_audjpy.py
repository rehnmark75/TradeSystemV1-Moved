#!/usr/bin/env python3
"""
AUDJPY Parameter Sweep v2.47.0
Current: SL=12 TP=18, max_conf=0.64, blocked=7-10,14-15
Has: two-pole filter, require_body_close_break, require_rejection_candle
1h ATR ~15 pips (JPY pair — default ref=8 is too low).
"""
import subprocess
import json
import re
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

EPIC = 'CS.D.AUDJPY.MINI.IP'
PAIR = 'AUDJPY'
DAYS = 30
TIMEFRAME = '5m'
PROGRESS_FILE = '/tmp/sweep_audjpy_progress.log'
RESULTS_FILE = '/tmp/sweep_audjpy_results.txt'

DB_CONF = dict(host='postgres', port=5432, dbname='strategy_config', user='postgres', password='postgres')

SWEEP_JSONB_KEYS = [
    'mfi_filter_mode', 'mfi_min_slope',
    'swing_significance_filter_mode', 'min_swing_significance',
    'scalp_min_adx',
    'atr_reference_pips',
    'regime_sl_tp_enabled', 'atr_normalized_sl_tp_enabled',
    'atr_scale_min', 'atr_scale_max',
    'trending_sl_mult', 'trending_tp_mult',
    'ranging_sl_mult', 'ranging_tp_mult',
]


def log(msg):
    print(msg, flush=True)
    with open(PROGRESS_FILE, 'a') as f:
        f.write(msg + '\n')


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
    with get_conn() as conn, conn.cursor() as cur:
        for key in SWEEP_JSONB_KEYS:
            cur.execute(
                "UPDATE smc_simple_pair_overrides SET parameter_overrides = parameter_overrides - %s WHERE epic=%s",
                (key, EPIC)
            )
        conn.commit()
    orig_params = original.get('parameter_overrides', {})
    if orig_params:
        with get_conn() as conn, conn.cursor() as cur:
            for key in SWEEP_JSONB_KEYS:
                if key in orig_params:
                    cur.execute(
                        "UPDATE smc_simple_pair_overrides "
                        "SET parameter_overrides = parameter_overrides || %s::jsonb "
                        "WHERE epic=%s",
                        (json.dumps({key: orig_params[key]}), EPIC)
                    )
            conn.commit()


def run_backtest(label: str, test_num: int, total: int) -> dict:
    cmd = ['python', '/app/forex_scanner/bt.py', PAIR, str(DAYS), '--scalp', '--timeframe', TIMEFRAME]
    log(f"\n  [{test_num}/{total}] Running: {label}")
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

    if metrics.get('win_rate', 0) == 0 and metrics.get('winners', 0) > 0 and metrics.get('signals', 0) > 0:
        metrics['win_rate'] = round(100.0 * metrics['winners'] / metrics['signals'], 1)

    if 'profit_factor' not in metrics:
        metrics['error'] = 'No results found'
        log(f"    ERROR: {output[-500:]}")

    return metrics


def fmt(m: dict) -> str:
    if 'error' in m:
        return f"  ERROR: {m['error']}"
    return (
        f"  Signals={int(m.get('signals',0)):3d}  "
        f"W={int(m.get('winners',0))}  L={int(m.get('losers',0))}  "
        f"WR={m.get('win_rate',0):.1f}%  "
        f"PF={m.get('profit_factor',0):.2f}  "
        f"Exp={m.get('expectancy',0):.1f}p  "
        f"AvgW={m.get('avg_win',0):.1f}p  "
        f"AvgL={m.get('avg_loss',0):.1f}p"
    )


def main():
    with open(PROGRESS_FILE, 'w') as f:
        f.write('')

    TOTAL_TESTS = 40

    log(f"=== AUDJPY Parameter Sweep ({DAYS}d, {TIMEFRAME}) — {TOTAL_TESTS} tests ===\n")

    original = get_current_state()
    log(f"Original: SL={original.get('fixed_stop_loss_pips')} TP={original.get('fixed_take_profit_pips')} "
        f"MaxConf={original.get('max_confidence')} Hours={original.get('scalp_blocked_hours_utc')}")

    results = []
    n = 0

    # ─── GROUP 1: BASELINE ────────────────────────────────────────────────────
    log("\n--- GROUP 1: Baseline ---")
    restore_state(original)
    n += 1; r = run_backtest("1. BASELINE (current)", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    # ─── GROUP 2: ATR REFERENCE PIPS ──────────────────────────────────────────
    # AUDJPY 1h ATR ~15 pips. Default ref=8 is way too low.
    log("\n--- GROUP 2: ATR Reference Pips ---")
    for atr_ref in [8, 12, 15, 18]:
        restore_state(original)
        apply_db_override(extra_params={'atr_reference_pips': float(atr_ref), 'atr_normalized_sl_tp_enabled': True})
        n += 1; r = run_backtest(f"2. atr_ref={atr_ref}", n, TOTAL_TESTS)
        results.append(r); log(fmt(r))

    # ─── GROUP 3: MFI FILTER ─────────────────────────────────────────────────
    log("\n--- GROUP 3: MFI Filter Sweep ---")
    for slope in [-10.0, -15.0, -20.0, -25.0]:
        restore_state(original)
        apply_db_override(extra_params={'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': slope})
        n += 1; r = run_backtest(f"3. MFI slope={slope}", n, TOTAL_TESTS)
        results.append(r); log(fmt(r))

    # ─── GROUP 4: SWING SIGNIFICANCE ─────────────────────────────────────────
    log("\n--- GROUP 4: Swing Significance Sweep ---")
    for sig in [0.30, 0.35, 0.40, 0.45, 0.50]:
        restore_state(original)
        apply_db_override(extra_params={'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': sig})
        n += 1; r = run_backtest(f"4. SwingSig={sig}", n, TOTAL_TESTS)
        results.append(r); log(fmt(r))

    # ─── GROUP 5: MIN ADX ────────────────────────────────────────────────────
    log("\n--- GROUP 5: Min ADX Sweep ---")
    for adx in [0, 18, 20, 22, 25, 28]:
        restore_state(original)
        if adx > 0:
            apply_db_override(extra_params={'scalp_min_adx': float(adx)})
        n += 1; r = run_backtest(f"5. min_adx={adx}", n, TOTAL_TESTS)
        results.append(r); log(fmt(r))

    # ─── GROUP 6: MAX CONFIDENCE ─────────────────────────────────────────────
    log("\n--- GROUP 6: Max Confidence Sweep ---")
    for mc in [0.64, 0.62, 0.60, 0.58, 0.55]:
        restore_state(original)
        apply_db_override(max_conf=mc)
        n += 1; r = run_backtest(f"6. max_conf={mc}", n, TOTAL_TESTS)
        results.append(r); log(fmt(r))

    # ─── GROUP 7: DYNAMIC SL/TP ─────────────────────────────────────────────
    log("\n--- GROUP 7: Dynamic SL/TP ---")

    restore_state(original)
    apply_db_override(extra_params={'regime_sl_tp_enabled': True, 'atr_normalized_sl_tp_enabled': False})
    n += 1; r = run_backtest("7a. Regime SL/TP only", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'regime_sl_tp_enabled': False, 'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 15.0})
    n += 1; r = run_backtest("7b. ATR norm only (ref=15)", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'regime_sl_tp_enabled': True, 'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 15.0})
    n += 1; r = run_backtest("7c. Both: regime+ATR(15)", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'regime_sl_tp_enabled': True, 'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 12.0})
    n += 1; r = run_backtest("7d. Both: regime+ATR(12)", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'regime_sl_tp_enabled': True, 'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 15.0, 'atr_scale_min': 0.8, 'atr_scale_max': 1.3})
    n += 1; r = run_backtest("7e. Both(15) scale 0.8-1.3", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'regime_sl_tp_enabled': True, 'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 15.0,
                                     'trending_sl_mult': 1.0, 'trending_tp_mult': 1.2, 'ranging_sl_mult': 1.1, 'ranging_tp_mult': 0.8})
    n += 1; r = run_backtest("7f. Both(15)+regime mults", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    # ─── GROUP 8: BLOCKED HOURS ──────────────────────────────────────────────
    # Current blocks: 7,8,9,10,14,15 (early London + overlap)
    log("\n--- GROUP 8: Blocked Hours Sweep ---")
    hour_configs = [
        ('7,8,9,10,14,15', 'current'),
        ('7,8,9,10,11,14,15', 'cur+hour11'),
        ('7,8,9,10,14,15,16,17', 'cur+late NY'),
        ('7,8,9,10,14,15,0,1,2,3,22,23', 'cur+deep Asian'),
        ('14,15', 'overlap only'),
        ('', 'no blocks'),
    ]
    for hours, desc in hour_configs:
        restore_state(original)
        apply_db_override(blocked_hours=hours)
        n += 1; r = run_backtest(f"8. hours: {desc}", n, TOTAL_TESTS)
        results.append(r); log(fmt(r))

    # ─── GROUP 9: COMBINED ───────────────────────────────────────────────────
    log("\n--- GROUP 9: Combined Tests ---")

    restore_state(original)
    apply_db_override(extra_params={'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -15.0,
                                     'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.40})
    n += 1; r = run_backtest("9a. MFI-15+SwSig0.40", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -15.0,
                                     'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.40,
                                     'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 15.0})
    n += 1; r = run_backtest("9b. MFI-15+SwSig0.40+ATR15", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    restore_state(original)
    apply_db_override(extra_params={'mfi_filter_mode': 'ACTIVE', 'mfi_min_slope': -15.0,
                                     'swing_significance_filter_mode': 'ACTIVE', 'min_swing_significance': 0.40,
                                     'regime_sl_tp_enabled': True, 'atr_normalized_sl_tp_enabled': True, 'atr_reference_pips': 15.0})
    n += 1; r = run_backtest("9c. All+regime+ATR(15)", n, TOTAL_TESTS)
    results.append(r); log(fmt(r))

    # ─── RESTORE ──────────────────────────────────────────────────────────────
    restore_state(original)
    log("\n✅ DB restored to original state")

    # ─── FINAL SUMMARY ────────────────────────────────────────────────────────
    summary = []
    summary.append("=" * 110)
    summary.append(f"{'Config':<45} {'Sigs':>5} {'W':>4} {'L':>4} {'WR%':>6} {'PF':>6} {'Exp(p)':>7} {'AvgW':>6} {'AvgL':>6}")
    summary.append("=" * 110)
    for r in results:
        if 'error' not in r:
            summary.append(
                f"{r['label']:<45} "
                f"{int(r.get('signals', 0)):>5} "
                f"{int(r.get('winners', 0)):>4} "
                f"{int(r.get('losers', 0)):>4} "
                f"{r.get('win_rate', 0):>5.1f}% "
                f"{r.get('profit_factor', 0):>6.2f} "
                f"{r.get('expectancy', 0):>7.1f} "
                f"{r.get('avg_win', 0):>6.1f} "
                f"{r.get('avg_loss', 0):>6.1f}"
            )

    valid = [r for r in results if 'error' not in r and r.get('signals', 0) >= 10]
    if valid:
        best_pf = max(valid, key=lambda x: x.get('profit_factor', 0))
        best_exp = max(valid, key=lambda x: x.get('expectancy', 0))
        summary.append(f"\n🏆 Best PF (>=10 sigs):  {best_pf['label']} → PF={best_pf['profit_factor']:.2f}")
        summary.append(f"🏆 Best Exp (>=10 sigs): {best_exp['label']} → {best_exp['expectancy']:.1f} pips/trade")

    summary_text = '\n'.join(summary)
    log(f"\n{summary_text}")

    with open(RESULTS_FILE, 'w') as f:
        f.write(summary_text + '\n')
    log(f"\n📁 Results saved to {RESULTS_FILE}")


if __name__ == '__main__':
    main()
