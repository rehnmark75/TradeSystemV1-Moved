#!/usr/bin/env python3
"""
LPF Per-Pair Threshold Sweep

Finds optimal per-pair LPF thresholds and rule configurations by combining
backtest trade outcomes with LPF decision data.

Modes:
  --backtest     Run fresh backtests first (LPF in monitor mode so all signals
                 get trade outcomes), then sweep thresholds. Most accurate.
  (default)      Replay existing loss_prevention_decisions data only.

Usage:
  python sweep_lpf_per_pair.py --backtest                # All pairs, 30d backtest first
  python sweep_lpf_per_pair.py --backtest --days 14      # All pairs, 14d
  python sweep_lpf_per_pair.py --backtest EURUSD         # Single pair
  python sweep_lpf_per_pair.py                           # Replay existing data only
  python sweep_lpf_per_pair.py --recommend               # Output SQL for recommended config
"""
import subprocess
import sys
import json
import re
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 required. Run inside task-worker container.")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────

STRATEGY_DB = os.getenv(
    'STRATEGY_CONFIG_DATABASE_URL',
    'postgresql://postgres:postgres@postgres:5432/strategy_config'
)
FOREX_DB = os.getenv(
    'FOREX_DATABASE_URL',
    'postgresql://postgres:postgres@postgres:5432/forex'
)

THRESHOLDS = [round(t * 0.05, 2) for t in range(4, 19)]  # 0.20 to 0.90

ALL_EPICS = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.USDJPY.MINI.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDCAD.MINI.IP',
    'CS.D.EURJPY.MINI.IP',
    'CS.D.AUDUSD.MINI.IP',
    'CS.D.AUDJPY.MINI.IP',
    'CS.D.NZDUSD.MINI.IP',
]

EPIC_SHORT = {
    'CS.D.EURUSD.CEEM.IP': 'EURUSD',
    'CS.D.USDJPY.MINI.IP': 'USDJPY',
    'CS.D.GBPUSD.MINI.IP': 'GBPUSD',
    'CS.D.USDCAD.MINI.IP': 'USDCAD',
    'CS.D.EURJPY.MINI.IP': 'EURJPY',
    'CS.D.AUDUSD.MINI.IP': 'AUDUSD',
    'CS.D.AUDJPY.MINI.IP': 'AUDJPY',
    'CS.D.NZDUSD.MINI.IP': 'NZDUSD',
    'CS.D.USDCHF.MINI.IP': 'USDCHF',
}

SHORT_TO_EPIC = {v: k for k, v in EPIC_SHORT.items()}


# ── Database Helpers ─────────────────────────────────────────────────────────

def get_strategy_conn():
    return psycopg2.connect(STRATEGY_DB)

def get_forex_conn():
    return psycopg2.connect(FOREX_DB)


# ── Backtest Runner ──────────────────────────────────────────────────────────

def set_lpf_block_mode(mode: str):
    """Set LPF block mode ('monitor' or 'block')."""
    with get_strategy_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE loss_prevention_config SET block_mode = %s WHERE id = 1",
                (mode,)
            )
        conn.commit()
    print(f"  LPF block_mode set to '{mode}'")


def get_lpf_block_mode() -> str:
    """Get current LPF block mode."""
    with get_strategy_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT block_mode FROM loss_prevention_config WHERE id = 1")
            row = cur.fetchone()
            return row[0] if row else 'monitor'


def clear_backtest_decisions(epic: str, cutoff: datetime):
    """Clear old backtest LPF decisions for a pair (keep live ones)."""
    with get_strategy_conn() as conn:
        with conn.cursor() as cur:
            # Backtest decisions have signal_timestamp in the past but created_at is recent
            # Clear decisions created after cutoff (the ones we're about to regenerate)
            cur.execute(
                "DELETE FROM loss_prevention_decisions WHERE epic = %s AND created_at >= %s",
                (epic, cutoff)
            )
            deleted = cur.rowcount
        conn.commit()
    return deleted


def run_backtest(pair_short: str, days: int) -> dict:
    """Run bt.py for a pair and parse results."""
    cmd = [
        'python', '/app/forex_scanner/bt.py',
        pair_short, str(days),
        '--timeframe', '5m',
        '--pipeline',
    ]
    print(f"\n  Running backtest: {pair_short} {days}d ...")
    start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    elapsed = time.time() - start

    metrics = {'pair': pair_short, 'elapsed': round(elapsed, 1)}
    patterns = {
        'signals': r'Total Signals:\s*(\d+)',
        'winners': r'Winners:\s*(\d+)',
        'losers': r'Losers:\s*(\d+)',
        'win_rate': r'Win Rate:\s*([\d.]+)%',
        'profit_factor': r'Profit Factor:\s*([\d.]+)',
        'expectancy': r'Expectancy:\s*([\d.]+)\s*pips',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            metrics[key] = float(m.group(1))

    if 'signals' not in metrics:
        metrics['error'] = 'No results parsed'
        # Show last 300 chars of output for debugging
        print(f"    ERROR: could not parse results")
        print(f"    Last output: ...{output[-300:]}")
    else:
        s = int(metrics.get('signals', 0))
        w = int(metrics.get('winners', 0))
        wr = metrics.get('win_rate', 0)
        pf = metrics.get('profit_factor', 0)
        print(f"    {s} signals, {w}W, WR={wr:.1f}%, PF={pf:.2f} ({elapsed:.0f}s)")

    return metrics


def load_backtest_decisions(epic: str, after: datetime) -> List[Dict]:
    """Load LPF decisions created after a cutoff (from the backtest we just ran)."""
    with get_strategy_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT epic, signal_type, confidence, total_penalty,
                       triggered_rules, decision, signal_timestamp
                FROM loss_prevention_decisions
                WHERE epic = %s AND created_at >= %s
                ORDER BY signal_timestamp
            """, (epic, after))
            rows = [dict(r) for r in cur.fetchall()]

    for row in rows:
        tr = row['triggered_rules']
        if isinstance(tr, str):
            tr = json.loads(tr)
        row['triggered_rules'] = tr or []

    return rows


def load_backtest_outcomes(epic: str, days: int) -> List[Dict]:
    """Load backtest signal outcomes from the most recent execution for this epic."""
    with get_forex_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find the most recent execution for this epic
            cur.execute("""
                SELECT execution_id
                FROM backtest_signals
                WHERE epic = %s
                GROUP BY execution_id
                ORDER BY MAX(created_at) DESC
                LIMIT 1
            """, (epic,))
            row = cur.fetchone()
            if not row:
                return []

            exec_id = row['execution_id']

            cur.execute("""
                SELECT epic, signal_type, signal_timestamp, confidence_score,
                       pips_gained, trade_result, validation_passed
                FROM backtest_signals
                WHERE execution_id = %s AND epic = %s
                ORDER BY signal_timestamp
            """, (exec_id, epic))
            return [dict(r) for r in cur.fetchall()]


def join_decisions_with_outcomes(decisions: List[Dict],
                                outcomes: List[Dict]) -> List[Dict]:
    """
    Join LPF decisions with backtest outcomes on epic + signal_timestamp.
    Returns merged records with both triggered_rules AND trade_result.
    """
    # Build outcome lookup with multiple keys for fuzzy timestamp matching
    # Backtests can have slight timestamp differences (seconds/minutes)
    outcome_lookup = {}
    for o in outcomes:
        ts = o.get('signal_timestamp')
        if ts:
            # Store at multiple granularities for matching
            key_exact = ts.replace(microsecond=0)
            key_minute = ts.replace(second=0, microsecond=0)
            # Also try +/- 5 minutes for drift
            for offset_min in range(-5, 6):
                key_fuzzy = key_minute.replace(
                    minute=(key_minute.minute + offset_min) % 60,
                    hour=key_minute.hour + ((key_minute.minute + offset_min) // 60)
                )
                if key_fuzzy not in outcome_lookup:
                    outcome_lookup[key_fuzzy] = o
            outcome_lookup[key_exact] = o
            outcome_lookup[key_minute] = o

    joined = []
    matched = 0
    for d in decisions:
        ts = d.get('signal_timestamp')
        merged = {**d, 'pips_gained': None, 'trade_result': None}
        if ts:
            key = ts.replace(second=0, microsecond=0)
            outcome = outcome_lookup.get(key)
            if not outcome:
                key = ts.replace(microsecond=0)
                outcome = outcome_lookup.get(key)
            if outcome:
                matched += 1
                merged['pips_gained'] = outcome.get('pips_gained')
                merged['trade_result'] = outcome.get('trade_result')
        joined.append(merged)

    return joined, matched


# ── Data Loading (replay mode) ──���────────────────────────────────────────────

def load_all_decisions(pair_filter: Optional[str] = None) -> List[Dict]:
    """Load all LPF decisions with triggered_rules from strategy_config DB."""
    with get_strategy_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT epic, signal_type, confidence, total_penalty,
                       triggered_rules, decision, signal_timestamp
                FROM loss_prevention_decisions
                WHERE triggered_rules IS NOT NULL
            """
            params = []
            if pair_filter:
                sql += " AND epic LIKE %s"
                params.append(f'%{pair_filter}%')
            sql += " ORDER BY created_at"
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    for row in rows:
        tr = row['triggered_rules']
        if isinstance(tr, str):
            tr = json.loads(tr)
        row['triggered_rules'] = tr or []

    return rows


def load_live_outcomes() -> Dict[str, List[Dict]]:
    """Load alert_history + trade_log outcomes, keyed by epic."""
    outcomes_by_epic = defaultdict(list)
    with get_forex_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT ah.id, ah.epic, ah.signal_type, ah.alert_timestamp,
                       ah.lpf_penalty, tl.profit_loss
                FROM alert_history ah
                INNER JOIN trade_log tl ON ah.id = tl.alert_id
                WHERE ah.lpf_penalty IS NOT NULL
                  AND tl.profit_loss IS NOT NULL
            """)
            for row in cur.fetchall():
                outcomes_by_epic[row['epic']].append(dict(row))
    return dict(outcomes_by_epic)


def load_current_config() -> Tuple[Dict, Dict]:
    """Load current global LPF config and per-pair configs."""
    with get_strategy_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM loss_prevention_config WHERE id = 1")
            global_config = dict(cur.fetchone() or {})

            cur.execute("SELECT * FROM loss_prevention_pair_config")
            pair_configs = {r['epic']: dict(r) for r in cur.fetchall()}

    return global_config, pair_configs


# ── Replay Engine ────────────────────────────────────────────────────────────

def reaggregate_penalty(triggered_rules: List[Dict],
                        disabled_rules: set = None,
                        penalty_overrides: dict = None) -> float:
    """Re-compute total penalty from triggered_rules with optional overrides."""
    disabled_rules = disabled_rules or set()
    penalty_overrides = penalty_overrides or {}

    category_penalties = {}
    for rule in triggered_rules:
        name = rule.get('rule_name', '')
        if name in disabled_rules:
            continue

        penalty = float(penalty_overrides.get(name, rule.get('penalty', 0)))
        cat = rule.get('category', '?')

        if cat not in category_penalties:
            category_penalties[cat] = penalty
        else:
            if penalty < 0:
                category_penalties[cat] = min(category_penalties[cat], penalty)
            else:
                category_penalties[cat] = max(category_penalties[cat], penalty)

    return sum(category_penalties.values())


def sweep_threshold_signals(decisions: List[Dict]) -> Dict[float, Dict]:
    """Simulate each threshold and count allowed/blocked signals."""
    results = {}
    for threshold in THRESHOLDS:
        allowed = 0
        blocked = 0
        for d in decisions:
            penalty = reaggregate_penalty(d['triggered_rules'])
            if penalty >= threshold:
                blocked += 1
            else:
                allowed += 1
        total = allowed + blocked
        results[threshold] = {
            'threshold': threshold,
            'allowed': allowed,
            'blocked': blocked,
            'total': total,
            'block_rate': round(100 * blocked / total, 1) if total else 0,
        }
    return results


def sweep_threshold_outcomes(joined_data: List[Dict]) -> Dict[float, Dict]:
    """
    Sweep thresholds using joined decisions+outcomes.
    For each threshold, show WR/PF/PnL of signals that would pass.
    """
    # Only use records that have trade outcomes
    with_outcome = [d for d in joined_data if d.get('pips_gained') is not None]
    if not with_outcome:
        return {}

    results = {}
    for threshold in THRESHOLDS:
        wins = 0
        losses = 0
        win_pips = 0.0
        loss_pips = 0.0
        total_pips = 0.0

        for d in with_outcome:
            penalty = reaggregate_penalty(d['triggered_rules'])
            if penalty >= threshold:
                continue  # Blocked at this threshold

            pips = float(d['pips_gained'] or 0)
            total_pips += pips
            result = d.get('trade_result', '')
            if result == 'win' or pips > 0:
                wins += 1
                win_pips += abs(pips)
            elif result == 'loss' or pips < 0:
                losses += 1
                loss_pips += abs(pips)

        total = wins + losses
        wr = round(100 * wins / total, 1) if total else 0
        pf = round(win_pips / loss_pips, 2) if loss_pips > 0 else (99.99 if win_pips > 0 else 0)
        exp = round(total_pips / total, 1) if total else 0

        results[threshold] = {
            'threshold': threshold,
            'trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'profit_factor': pf,
            'total_pips': round(total_pips, 1),
            'expectancy': exp,
        }
    return results


def sweep_threshold_live(outcomes: List[Dict]) -> Dict[float, Dict]:
    """Sweep using live trade outcomes (alert_history + trade_log). Fallback mode."""
    if not outcomes:
        return {}

    results = {}
    for threshold in THRESHOLDS:
        wins = 0
        losses = 0
        win_pnl = 0.0
        loss_pnl = 0.0
        total_pnl = 0.0

        for o in outcomes:
            penalty = float(o.get('lpf_penalty', 0))
            if penalty >= threshold:
                continue
            pnl = float(o['profit_loss'])
            total_pnl += pnl
            if pnl > 0:
                wins += 1
                win_pnl += pnl
            else:
                losses += 1
                loss_pnl += abs(pnl)

        total = wins + losses
        wr = round(100 * wins / total, 1) if total else 0
        pf = round(win_pnl / loss_pnl, 2) if loss_pnl > 0 else (99.99 if win_pnl > 0 else 0)
        results[threshold] = {
            'threshold': threshold,
            'trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'profit_factor': pf,
            'total_pnl': round(total_pnl, 2),
        }
    return results


# ── Rule Impact Analysis ─────────────────────────────────────────────────────

def analyze_rule_frequency(decisions: List[Dict]) -> Dict[str, Dict]:
    """Count how often each rule triggers and its contribution."""
    rule_stats = defaultdict(lambda: {'count': 0, 'total_penalty': 0.0, 'category': ''})
    for d in decisions:
        for rule in d['triggered_rules']:
            name = rule.get('rule_name', '')
            rule_stats[name]['count'] += 1
            rule_stats[name]['total_penalty'] += float(rule.get('penalty', 0))
            rule_stats[name]['category'] = rule.get('category', '?')
    return dict(rule_stats)


def analyze_rule_win_rate(joined_data: List[Dict]) -> Dict[str, Dict]:
    """For each rule, compute WR when it triggers vs when it doesn't."""
    with_outcome = [d for d in joined_data if d.get('trade_result') is not None]
    if not with_outcome:
        return {}

    # Collect all rule names
    all_rules = set()
    for d in with_outcome:
        for r in d['triggered_rules']:
            all_rules.add(r.get('rule_name', ''))

    rule_wr = {}
    for rule_name in all_rules:
        triggered_wins = 0
        triggered_losses = 0
        not_triggered_wins = 0
        not_triggered_losses = 0
        triggered_pips = 0.0
        not_triggered_pips = 0.0

        for d in with_outcome:
            rule_names = [r.get('rule_name', '') for r in d['triggered_rules']]
            pips = float(d.get('pips_gained', 0) or 0)
            is_win = d.get('trade_result') == 'win' or pips > 0

            if rule_name in rule_names:
                triggered_pips += pips
                if is_win:
                    triggered_wins += 1
                else:
                    triggered_losses += 1
            else:
                not_triggered_pips += pips
                if is_win:
                    not_triggered_wins += 1
                else:
                    not_triggered_losses += 1

        t_total = triggered_wins + triggered_losses
        nt_total = not_triggered_wins + not_triggered_losses

        rule_wr[rule_name] = {
            'triggered_wr': round(100 * triggered_wins / t_total, 1) if t_total else None,
            'triggered_count': t_total,
            'triggered_pips': round(triggered_pips, 1),
            'not_triggered_wr': round(100 * not_triggered_wins / nt_total, 1) if nt_total else None,
            'not_triggered_count': nt_total,
            'not_triggered_pips': round(not_triggered_pips, 1),
        }

    return rule_wr


def simulate_rule_disable(decisions: List[Dict], rule_name: str,
                          current_threshold: float) -> Dict:
    """Show impact of disabling a single rule."""
    original_blocked = 0
    new_blocked = 0
    freed = 0

    for d in decisions:
        orig_penalty = reaggregate_penalty(d['triggered_rules'])
        new_penalty = reaggregate_penalty(d['triggered_rules'], disabled_rules={rule_name})

        if orig_penalty >= current_threshold:
            original_blocked += 1
        if new_penalty >= current_threshold:
            new_blocked += 1
        if orig_penalty >= current_threshold and new_penalty < current_threshold:
            freed += 1

    return {
        'rule_name': rule_name,
        'original_blocked': original_blocked,
        'new_blocked': new_blocked,
        'freed': freed,
        'total': len(decisions),
    }


# ── Output Formatting ────────────────────────────────────────────────────────

def print_header(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def print_threshold_sweep(epic: str, sweep: Dict[float, Dict],
                          current_threshold: float):
    short = EPIC_SHORT.get(epic, epic)
    print(f"\n--- {short} Signal Threshold Sweep ---")
    print(f"{'Thresh':>7} {'Allowed':>8} {'Blocked':>8} {'Block%':>7}")
    print(f"{'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for t, s in sorted(sweep.items()):
        marker = ' <--' if abs(t - current_threshold) < 0.01 else ''
        print(f"  {t:.2f}  {s['allowed']:>6}   {s['blocked']:>6}   {s['block_rate']:>5.1f}%{marker}")


def print_outcome_sweep(epic: str, sweep: Dict[float, Dict],
                        current_threshold: float, label: str = "Outcome"):
    short = EPIC_SHORT.get(epic, epic)
    if not sweep or all(s['trades'] == 0 for s in sweep.values()):
        print(f"\n--- {short} {label} Sweep: No data ---")
        return

    has_pips = 'total_pips' in next(iter(sweep.values()))
    pnl_col = 'Pips' if has_pips else 'PnL'
    pnl_key = 'total_pips' if has_pips else 'total_pnl'
    exp_col = 'Exp/trade' if has_pips else ''

    print(f"\n--- {short} {label} Sweep ---")
    hdr = f"{'Thresh':>7} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'PF':>6} {pnl_col:>8}"
    if has_pips:
        hdr += f" {exp_col:>9}"
    print(hdr)
    print(f"{'-'*7} {'-'*7} {'-'*5} {'-'*6} {'-'*6} {'-'*8}" + (f" {'-'*9}" if has_pips else ""))

    best_pf = 0
    best_t = None
    for t, s in sorted(sweep.items()):
        if s['trades'] >= 5 and s['profit_factor'] > best_pf:
            best_pf = s['profit_factor']
            best_t = t

    for t, s in sorted(sweep.items()):
        marker = ''
        if abs(t - current_threshold) < 0.01:
            marker = ' <--'
        elif best_t is not None and abs(t - best_t) < 0.01 and s['trades'] >= 5:
            marker = ' ***'
        pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] < 99 else 'inf'
        line = f"  {t:.2f}  {s['trades']:>5}   {s['wins']:>3}   {s['win_rate']:>5.1f} {pf_str:>6} {s[pnl_key]:>8.1f}"
        if has_pips:
            line += f" {s.get('expectancy', 0):>8.1f}p"
        print(f"{line}{marker}")

    if best_t is not None:
        print(f"  *** = best PF ({best_pf:.2f}) with >= 5 trades")


def print_rule_frequency(epic: str, rule_stats: Dict[str, Dict],
                         total_decisions: int, rule_wr: Dict[str, Dict] = None):
    short = EPIC_SHORT.get(epic, epic)
    has_wr = rule_wr and any(r.get('triggered_wr') is not None for r in rule_wr.values())

    print(f"\n--- {short} Rule Analysis ({total_decisions} signals) ---")
    hdr = f"{'Rule':>28} {'Cat':>4} {'Hits':>5} {'Rate%':>6} {'Pen':>6}"
    if has_wr:
        hdr += f" {'TrigWR':>7} {'!TrigWR':>8} {'TrigPips':>9}"
    print(hdr)
    print("-" * len(hdr))

    sorted_rules = sorted(rule_stats.items(), key=lambda x: -x[1]['count'])
    for name, stats in sorted_rules:
        rate = 100 * stats['count'] / total_decisions if total_decisions else 0
        avg_pen = stats['total_penalty'] / stats['count'] if stats['count'] else 0
        line = f"  {name:>26}    {stats['category']}  {stats['count']:>3}  {rate:>5.1f}  {avg_pen:>+5.2f}"
        if has_wr and name in rule_wr:
            rw = rule_wr[name]
            twr = f"{rw['triggered_wr']:.0f}%" if rw['triggered_wr'] is not None else '  -'
            ntwr = f"{rw['not_triggered_wr']:.0f}%" if rw['not_triggered_wr'] is not None else '  -'
            tp = f"{rw['triggered_pips']:+.0f}" if rw['triggered_count'] > 0 else '  -'
            line += f"  {twr:>6}   {ntwr:>6}  {tp:>8}"
        print(line)


def print_rule_disable_impact(epic: str, impacts: List[Dict], current_threshold: float):
    short = EPIC_SHORT.get(epic, epic)
    print(f"\n--- {short} Rule Disable Impact (threshold={current_threshold:.2f}) ---")
    print(f"{'Rule':>28} {'CurBlocked':>11} {'Freed':>6} {'NewBlocked':>11}")
    print(f"{'-'*28} {'-'*11} {'-'*6} {'-'*11}")

    sorted_impacts = sorted(impacts, key=lambda x: -x['freed'])
    for imp in sorted_impacts:
        if imp['freed'] == 0:
            continue
        print(f"  {imp['rule_name']:>26}  {imp['original_blocked']:>9}   {imp['freed']:>4}   {imp['new_blocked']:>9}")


def print_recommendations(recommendations: Dict[str, Dict]):
    print_header("RECOMMENDED PER-PAIR LPF CONFIG")
    print("\n-- Apply per-pair LPF config (run in strategy_config DB)")

    for epic, rec in sorted(recommendations.items()):
        short = EPIC_SHORT.get(epic, epic)
        threshold = rec.get('threshold')
        disabled = rec.get('disabled_rules', [])
        notes = rec.get('notes', '')

        if threshold is None and not disabled:
            continue

        disabled_sql = "ARRAY[" + ",".join(f"'{r}'" for r in disabled) + "]" if disabled else 'NULL'
        threshold_sql = str(threshold) if threshold else 'NULL'

        print(f"""
-- {short}: {notes}
INSERT INTO loss_prevention_pair_config (epic, penalty_threshold, disabled_rules, notes)
VALUES ('{epic}', {threshold_sql}, {disabled_sql}, '{notes}')
ON CONFLICT (epic) DO UPDATE SET
    penalty_threshold = EXCLUDED.penalty_threshold,
    disabled_rules = EXCLUDED.disabled_rules,
    notes = EXCLUDED.notes,
    updated_at = NOW();""")


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_pair(epic: str, decisions: List[Dict], outcome_sweep: Dict,
                 rule_wr: Dict, global_threshold: float,
                 pair_config: Optional[Dict]) -> Optional[Dict]:
    """Full analysis for one pair. Returns recommendation."""
    short = EPIC_SHORT.get(epic, epic)
    current_threshold = global_threshold
    if pair_config and pair_config.get('penalty_threshold') is not None:
        current_threshold = float(pair_config['penalty_threshold'])

    total = len(decisions)
    if total < 5:
        print(f"\n--- {short}: Only {total} decisions -- skipping (need >= 5) ---")
        return None

    # 1. Signal threshold sweep
    signal_sweep = sweep_threshold_signals(decisions)
    print_threshold_sweep(epic, signal_sweep, current_threshold)

    # 2. Outcome threshold sweep
    print_outcome_sweep(epic, outcome_sweep, current_threshold, "Backtest Outcome")

    # 3. Rule frequency + win rate
    rule_stats = analyze_rule_frequency(decisions)
    print_rule_frequency(epic, rule_stats, total, rule_wr)

    # 4. Rule disable impact
    impacts = []
    for rule_name in rule_stats.keys():
        imp = simulate_rule_disable(decisions, rule_name, current_threshold)
        impacts.append(imp)
    print_rule_disable_impact(epic, impacts, current_threshold)

    # 5. Build recommendation
    rec = build_recommendation(epic, signal_sweep, outcome_sweep, rule_stats,
                               rule_wr, impacts, current_threshold, total)
    return rec


def build_recommendation(epic, signal_sweep, outcome_sweep, rule_stats,
                         rule_wr, impacts, current_threshold, total_decisions) -> Dict:
    """Heuristic recommendation. Prefers outcome data over signal-level."""
    short = EPIC_SHORT.get(epic, epic)
    rec = {'epic': epic, 'notes': ''}

    # Find best threshold from outcome data (minimum 8 trades, best PF)
    best_pf = 0
    best_threshold = current_threshold
    best_trades = 0
    has_outcome_data = False

    for t, s in outcome_sweep.items():
        if s['trades'] >= 8:
            has_outcome_data = True
            if s['profit_factor'] > best_pf:
                best_pf = s['profit_factor']
                best_threshold = t
                best_trades = s['trades']

    if has_outcome_data and abs(best_threshold - current_threshold) > 0.04:
        rec['threshold'] = best_threshold
        wr = outcome_sweep[best_threshold]['win_rate']
        rec['notes'] += f"threshold {current_threshold:.2f}->{best_threshold:.2f} (PF={best_pf:.2f} WR={wr:.0f}% n={best_trades})"
    elif not has_outcome_data:
        # Signal-level fallback: target 15-25% block rate
        for t, s in sorted(signal_sweep.items()):
            if 15 <= s['block_rate'] <= 25:
                if abs(t - current_threshold) > 0.04:
                    rec['threshold'] = t
                    rec['notes'] += f"threshold {current_threshold:.2f}->{t:.2f} (targets {s['block_rate']:.0f}% block rate, no outcome data)"
                break

    # Find rules that HURT performance when triggered (lower WR than not-triggered)
    disabled = []
    if rule_wr:
        for rule_name, rw in rule_wr.items():
            if rw['triggered_wr'] is None or rw['not_triggered_wr'] is None:
                continue
            if rw['triggered_count'] < 3:
                continue
            # Rule makes things worse: triggered WR is lower AND we freed signals
            if rw['triggered_wr'] < rw['not_triggered_wr']:
                # Check if disabling actually frees signals
                imp = next((i for i in impacts if i['rule_name'] == rule_name), None)
                if imp and imp['freed'] >= 2:
                    disabled.append(rule_name)

    if disabled:
        rec['disabled_rules'] = disabled
        if rec['notes']:
            rec['notes'] += '; '
        rec['notes'] += f"disable {disabled}"

    if not rec.get('threshold') and not rec.get('disabled_rules'):
        rec['notes'] = f"current config OK (threshold={current_threshold:.2f})"

    print(f"\n  >> {short}: {rec['notes']}")
    return rec


# ── Main ─────────────────────────────────────────────────────────────────────

def main_backtest(pair_filter: Optional[str], days: int, show_recommend: bool):
    """Backtest-first mode: run backtests in monitor mode, then sweep."""
    print_header(f"LPF Per-Pair Sweep (backtest {days}d, monitor mode)")

    # Determine epics
    if pair_filter:
        epic = SHORT_TO_EPIC.get(pair_filter)
        if not epic:
            # Try matching
            for e, s in EPIC_SHORT.items():
                if pair_filter in s or pair_filter in e:
                    epic = e
                    break
        if not epic:
            print(f"ERROR: Unknown pair '{pair_filter}'")
            return
        epics = [epic]
    else:
        epics = ALL_EPICS

    global_config, pair_configs = load_current_config()
    global_threshold = float(global_config.get('penalty_threshold', 0.60))
    original_mode = get_lpf_block_mode()

    # Step 1: Set LPF to monitor mode so all signals pass and get trade outcomes
    print(f"\nStep 1: Set LPF to monitor mode (was '{original_mode}')")
    if original_mode != 'monitor':
        set_lpf_block_mode('monitor')

    try:
        # Step 2: Run backtests per pair
        print(f"\nStep 2: Running backtests ({len(epics)} pairs, {days}d each)")
        backtest_start = datetime.now()
        bt_metrics = {}

        for epic in epics:
            short = EPIC_SHORT.get(epic, epic)
            # Clear old decisions for this pair so we get fresh data
            deleted = clear_backtest_decisions(epic, backtest_start)
            if deleted:
                print(f"  Cleared {deleted} old decisions for {short}")

            metrics = run_backtest(short, days)
            bt_metrics[epic] = metrics

        # Step 3: Load fresh decisions + outcomes and analyze
        print(f"\nStep 3: Loading fresh data and analyzing...")
        recommendations = {}

        for epic in epics:
            short = EPIC_SHORT.get(epic, epic)

            # Load decisions created during our backtest run
            decisions = load_backtest_decisions(epic, backtest_start)
            # Load outcomes from most recent backtest execution
            outcomes = load_backtest_outcomes(epic, days)

            # Join them
            joined, matched = join_decisions_with_outcomes(decisions, outcomes)
            with_outcome = len([j for j in joined if j.get('trade_result') is not None])

            print(f"\n  {short}: {len(decisions)} decisions, {len(outcomes)} outcomes, {matched} matched")

            if not decisions:
                print(f"  {short}: No LPF decisions from backtest -- skipping")
                continue

            # Build outcome sweep from joined data
            outcome_sweep = sweep_threshold_outcomes(joined)
            rule_wr = analyze_rule_win_rate(joined)

            rec = analyze_pair(epic, decisions, outcome_sweep, rule_wr,
                               global_threshold, pair_configs.get(epic))
            if rec:
                recommendations[epic] = rec

        # Print recommendations
        if show_recommend and recommendations:
            print_recommendations(recommendations)

        # Summary
        print_header("SUMMARY")
        print(f"\n{'Pair':>8} {'Signals':>8} {'Matched':>8} {'Current':>8} {'Recommended':>12}")
        print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
        for epic in epics:
            short = EPIC_SHORT.get(epic, epic)
            m = bt_metrics.get(epic, {})
            signals = int(m.get('signals', 0))
            decisions = load_backtest_decisions(epic, backtest_start)
            outcomes = load_backtest_outcomes(epic, days)
            joined, matched = join_decisions_with_outcomes(decisions, outcomes)

            pc = pair_configs.get(epic)
            curr = float(pc['penalty_threshold']) if pc and pc.get('penalty_threshold') else global_threshold
            rec = recommendations.get(epic, {})
            new_t = rec.get('threshold', curr)
            marker = ' *' if abs(new_t - curr) > 0.01 else ''
            print(f"  {short:>6}   {signals:>6}   {matched:>6}    {curr:.2f}       {new_t:.2f}{marker}")

        print(f"\n  * = changed from current setting")
        if not show_recommend:
            print(f"\n  Run with --recommend to generate SQL statements")

    finally:
        # Step 4: Restore LPF mode
        if original_mode != 'monitor':
            print(f"\nRestoring LPF to '{original_mode}' mode")
            set_lpf_block_mode(original_mode)


def main_replay(pair_filter: Optional[str], show_recommend: bool):
    """Replay mode: use existing decision data only."""
    print_header("LPF Per-Pair Sweep (replay existing data)")

    decisions = load_all_decisions(pair_filter)
    live_outcomes = load_live_outcomes()
    global_config, pair_configs = load_current_config()
    global_threshold = float(global_config.get('penalty_threshold', 0.60))

    print(f"\n  Decisions: {len(decisions)}")
    print(f"  Global threshold: {global_threshold}")
    print(f"  Per-pair configs: {len(pair_configs)}")

    decisions_by_epic = defaultdict(list)
    for d in decisions:
        decisions_by_epic[d['epic']].append(d)

    recommendations = {}
    for epic in sorted(decisions_by_epic.keys()):
        pair_decisions = decisions_by_epic[epic]
        pair_outcomes = live_outcomes.get(epic, [])

        # Use live outcomes for sweep
        outcome_sweep = sweep_threshold_live(pair_outcomes)

        rec = analyze_pair(epic, pair_decisions, outcome_sweep, {},
                           global_threshold, pair_configs.get(epic))
        if rec:
            recommendations[epic] = rec

    if show_recommend and recommendations:
        print_recommendations(recommendations)

    # Summary
    print_header("SUMMARY")
    print(f"\n{'Pair':>8} {'Decisions':>10} {'Outcomes':>9} {'Current':>8} {'Recommended':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*12}")
    for epic in sorted(decisions_by_epic.keys()):
        short = EPIC_SHORT.get(epic, epic)
        n_dec = len(decisions_by_epic[epic])
        n_out = len(live_outcomes.get(epic, []))
        pc = pair_configs.get(epic)
        curr = float(pc['penalty_threshold']) if pc and pc.get('penalty_threshold') else global_threshold
        rec = recommendations.get(epic, {})
        new_t = rec.get('threshold', curr)
        marker = ' *' if abs(new_t - curr) > 0.01 else ''
        print(f"  {short:>6}   {n_dec:>8}    {n_out:>6}    {curr:.2f}       {new_t:.2f}{marker}")

    print(f"\n  * = changed from current setting")
    if not show_recommend:
        print(f"\n  Run with --recommend to generate SQL")


def main():
    pair_filter = None
    show_recommend = '--recommend' in sys.argv
    backtest_mode = '--backtest' in sys.argv
    days = 30

    # Parse args
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--days' and i < len(sys.argv) - 1:
            days = int(sys.argv[i + 1])
        elif arg.startswith('--'):
            continue
        elif not arg.isdigit():
            pair_filter = arg.upper()

    if backtest_mode:
        main_backtest(pair_filter, days, show_recommend)
    else:
        main_replay(pair_filter, show_recommend)


if __name__ == '__main__':
    main()
