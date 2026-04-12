#!/usr/bin/env python3
"""
Config Promotion CLI — Compare and promote demo config to live.

Usage:
    # SMC Simple strategy config (original commands, still work)
    python -m forex_scanner.commands.promote_config diff
    python -m forex_scanner.commands.promote_config promote [--confirm]
    python -m forex_scanner.commands.promote_config status

    # Scanner / TradeValidator / Claude AI settings
    python -m forex_scanner.commands.promote_config scanner diff
    python -m forex_scanner.commands.promote_config scanner promote [--confirm]
    python -m forex_scanner.commands.promote_config scanner status

    # Loss Prevention Filter (global config + rules + pair overrides)
    python -m forex_scanner.commands.promote_config lpf diff
    python -m forex_scanner.commands.promote_config lpf promote [--confirm]
    python -m forex_scanner.commands.promote_config lpf status
"""

import os
import sys
import json
import argparse
import psycopg2
import psycopg2.extras
from datetime import datetime


DB_URL = os.getenv(
    'STRATEGY_CONFIG_DATABASE_URL',
    'postgresql://postgres:postgres@postgres:5432/strategy_config'
)

# Columns to skip when comparing/copying (auto-managed)
SKIP_COLUMNS = {'id', 'created_at', 'updated_at', 'updated_by', 'change_reason', 'config_set'}


def get_connection():
    return psycopg2.connect(DB_URL)


# =============================================================================
# SMC SIMPLE helpers (unchanged)
# =============================================================================

def get_active_config(conn, config_set: str) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM smc_simple_global_config WHERE is_active = TRUE AND config_set = %s LIMIT 1",
            (config_set,)
        )
        row = cur.fetchone()
        if not row:
            print(f"No active config found for config_set='{config_set}'")
            sys.exit(1)
        return dict(row)


def get_pair_overrides(conn, config_id: int) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT * FROM smc_simple_pair_overrides WHERE config_id = %s ORDER BY epic",
            (config_id,)
        )
        rows = cur.fetchall()
        return {row['epic']: dict(row) for row in rows}


def cmd_status(args):
    conn = get_connection()
    live = get_active_config(conn, 'live')
    demo = get_active_config(conn, 'demo')

    print(f"\n{'='*60}")
    print(f"  Live config: id={live['id']}, version={live['version']}")
    print(f"  Demo config: id={demo['id']}, version={demo['version']}")
    print(f"{'='*60}")

    live_overrides = get_pair_overrides(conn, live['id'])
    demo_overrides = get_pair_overrides(conn, demo['id'])

    print(f"\n  Live pairs ({len(live_overrides)}):")
    for epic, ov in live_overrides.items():
        enabled = "ON" if ov.get('is_enabled') else "OFF"
        sl = ov.get('fixed_stop_loss_pips', '-')
        tp = ov.get('fixed_take_profit_pips', '-')
        print(f"    {epic}: {enabled}  SL={sl}  TP={tp}")

    print(f"\n  Demo pairs ({len(demo_overrides)}):")
    for epic, ov in demo_overrides.items():
        enabled = "ON" if ov.get('is_enabled') else "OFF"
        sl = ov.get('fixed_stop_loss_pips', '-')
        tp = ov.get('fixed_take_profit_pips', '-')
        print(f"    {epic}: {enabled}  SL={sl}  TP={tp}")

    conn.close()


def cmd_diff(args):
    conn = get_connection()
    live = get_active_config(conn, 'live')
    demo = get_active_config(conn, 'demo')

    # Global config diff
    diffs = []
    for key in live:
        if key in SKIP_COLUMNS:
            continue
        live_val = live.get(key)
        demo_val = demo.get(key)
        if str(live_val) != str(demo_val):
            diffs.append((key, live_val, demo_val))

    if diffs:
        print(f"\n{'='*60}")
        print(f"  Global config differences (demo vs live)")
        print(f"{'='*60}")
        print(f"  {'Parameter':<45} {'Live':>12} {'Demo':>12}")
        print(f"  {'-'*45} {'-'*12} {'-'*12}")
        for key, live_val, demo_val in sorted(diffs):
            lv = str(live_val)[:12] if live_val is not None else 'NULL'
            dv = str(demo_val)[:12] if demo_val is not None else 'NULL'
            print(f"  {key:<45} {lv:>12} {dv:>12}")
        print(f"\n  Total: {len(diffs)} parameter(s) differ")
    else:
        print("\n  Global configs are identical.")

    # Pair override diff
    live_overrides = get_pair_overrides(conn, live['id'])
    demo_overrides = get_pair_overrides(conn, demo['id'])

    pair_diffs = []
    all_epics = set(live_overrides.keys()) | set(demo_overrides.keys())
    for epic in sorted(all_epics):
        lo = live_overrides.get(epic, {})
        do = demo_overrides.get(epic, {})
        for key in set(list(lo.keys()) + list(do.keys())):
            if key in SKIP_COLUMNS | {'id', 'config_id', 'created_at', 'updated_at', 'updated_by', 'change_reason'}:
                continue
            if str(lo.get(key)) != str(do.get(key)):
                pair_diffs.append((epic, key, lo.get(key), do.get(key)))

    if pair_diffs:
        print(f"\n{'='*60}")
        print(f"  Pair override differences (demo vs live)")
        print(f"{'='*60}")
        for epic, key, live_val, demo_val in pair_diffs:
            short_epic = epic.split('.')[-2] if '.' in epic else epic
            lv = str(live_val)[:15] if live_val is not None else 'NULL'
            dv = str(demo_val)[:15] if demo_val is not None else 'NULL'
            print(f"  {short_epic:<10} {key:<35} live={lv:<15} demo={dv}")
        print(f"\n  Total: {len(pair_diffs)} pair override(s) differ")
    else:
        print("\n  Pair overrides are identical.")

    conn.close()


def cmd_promote(args):
    conn = get_connection()
    live = get_active_config(conn, 'live')
    demo = get_active_config(conn, 'demo')

    # Calculate what would change
    changes = []
    for key in demo:
        if key in SKIP_COLUMNS:
            continue
        if str(live.get(key)) != str(demo.get(key)):
            changes.append(key)

    if not changes:
        print("\n  No differences to promote. Configs are identical.")
        conn.close()
        return

    print(f"\n  Promoting {len(changes)} parameter(s) from demo -> live")
    for key in sorted(changes):
        print(f"    {key}: {live.get(key)} -> {demo.get(key)}")

    if not args.confirm:
        print(f"\n  DRY RUN — add --confirm to apply changes")
        conn.close()
        return

    # Apply changes
    set_clauses = []
    values = []
    for key in changes:
        set_clauses.append(f"{key} = %s")
        values.append(demo[key])

    set_clauses.append("updated_at = %s")
    values.append(datetime.utcnow())
    set_clauses.append("updated_by = %s")
    values.append('promote_config_cli')
    set_clauses.append("change_reason = %s")
    values.append(f'Promoted from demo config (id={demo["id"]}) at {datetime.utcnow().isoformat()}')

    values.append(live['id'])

    sql = f"UPDATE smc_simple_global_config SET {', '.join(set_clauses)} WHERE id = %s"

    with conn.cursor() as cur:
        cur.execute(sql, values)

    # Also promote pair overrides
    demo_overrides = get_pair_overrides(conn, demo['id'])
    live_overrides = get_pair_overrides(conn, live['id'])

    pair_changes = 0
    for epic, demo_ov in demo_overrides.items():
        live_ov = live_overrides.get(epic)
        if not live_ov:
            continue
        pair_updates = []
        pair_values = []
        for key in demo_ov:
            if key in SKIP_COLUMNS | {'id', 'config_id', 'created_at', 'updated_at', 'updated_by', 'change_reason'}:
                continue
            if str(demo_ov.get(key)) != str(live_ov.get(key)):
                pair_updates.append(f"{key} = %s")
                pair_values.append(demo_ov[key])
                pair_changes += 1

        if pair_updates:
            pair_updates.append("updated_by = %s")
            pair_values.append('promote_config_cli')
            pair_updates.append("change_reason = %s")
            pair_values.append(f'Promoted from demo at {datetime.utcnow().isoformat()}')
            pair_values.append(live_ov['id'])

            sql = f"UPDATE smc_simple_pair_overrides SET {', '.join(pair_updates)} WHERE id = %s"
            with conn.cursor() as cur:
                cur.execute(sql, pair_values)

    conn.commit()
    conn.close()

    print(f"\n  PROMOTED: {len(changes)} global params + {pair_changes} pair override changes")
    print(f"  Restart task-worker-live to pick up new config.")


# =============================================================================
# GENERIC helpers for scanner + LPF
# =============================================================================

def _get_single_row(conn, table: str, config_set: str) -> dict:
    """Get a single row from a config_set-scoped table."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"SELECT * FROM {table} WHERE config_set = %s LIMIT 1",
            (config_set,)
        )
        row = cur.fetchone()
        if not row:
            print(f"No row found in {table} for config_set='{config_set}'")
            sys.exit(1)
        return dict(row)


def _get_rows(conn, table: str, config_set: str, order_by: str = 'id') -> list:
    """Get all rows from a config_set-scoped table."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"SELECT * FROM {table} WHERE config_set = %s ORDER BY {order_by}",
            (config_set,)
        )
        return [dict(r) for r in cur.fetchall()]


def _diff_rows(live: dict, demo: dict, skip: set = None) -> list:
    """Return list of (key, live_val, demo_val) for differing keys."""
    skip = skip or SKIP_COLUMNS
    diffs = []
    all_keys = set(live.keys()) | set(demo.keys())
    for key in sorted(all_keys):
        if key in skip:
            continue
        lv = live.get(key)
        dv = demo.get(key)
        if str(lv) != str(dv):
            diffs.append((key, lv, dv))
    return diffs


def _print_diff_table(diffs: list, label: str):
    if diffs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  {'Parameter':<45} {'Live':>15} {'Demo':>15}")
        print(f"  {'-'*45} {'-'*15} {'-'*15}")
        for key, lv, dv in diffs:
            ls = str(lv)[:15] if lv is not None else 'NULL'
            ds = str(dv)[:15] if dv is not None else 'NULL'
            print(f"  {key:<45} {ls:>15} {ds:>15}")
        print(f"\n  Total: {len(diffs)} difference(s)")
    else:
        print(f"\n  {label}: identical")


def _apply_update(conn, table: str, changes: list, demo_row: dict, live_id: int):
    """Apply column changes to the live row of a table."""
    set_clauses = []
    values = []
    for key, _live_val, demo_val in changes:
        set_clauses.append(f"{key} = %s")
        values.append(demo_val)
    set_clauses.append("updated_at = %s")
    values.append(datetime.utcnow())
    values.append(live_id)
    sql = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE id = %s"
    with conn.cursor() as cur:
        cur.execute(sql, values)


# =============================================================================
# SCANNER subcommand
# =============================================================================

def cmd_scanner_status(args):
    conn = get_connection()
    live = _get_single_row(conn, 'scanner_global_config', 'live')
    demo = _get_single_row(conn, 'scanner_global_config', 'demo')
    print(f"\n{'='*60}")
    print(f"  scanner_global_config")
    print(f"  Live: id={live['id']}, version={live.get('version','?')}, "
          f"min_confidence={live.get('min_confidence')}, "
          f"require_claude_approval={live.get('require_claude_approval')}")
    print(f"  Demo: id={demo['id']}, version={demo.get('version','?')}, "
          f"min_confidence={demo.get('min_confidence')}, "
          f"require_claude_approval={demo.get('require_claude_approval')}")
    conn.close()


def cmd_scanner_diff(args):
    conn = get_connection()
    live = _get_single_row(conn, 'scanner_global_config', 'live')
    demo = _get_single_row(conn, 'scanner_global_config', 'demo')
    diffs = _diff_rows(live, demo)
    _print_diff_table(diffs, "scanner_global_config differences (demo vs live)")
    conn.close()


def cmd_scanner_promote(args):
    conn = get_connection()
    live = _get_single_row(conn, 'scanner_global_config', 'live')
    demo = _get_single_row(conn, 'scanner_global_config', 'demo')
    diffs = _diff_rows(live, demo)

    if not diffs:
        print("\n  scanner_global_config: no differences to promote.")
        conn.close()
        return

    print(f"\n  Promoting {len(diffs)} scanner parameter(s) from demo -> live:")
    for key, lv, dv in diffs:
        print(f"    {key}: {lv} -> {dv}")

    if not args.confirm:
        print("\n  DRY RUN — add --confirm to apply")
        conn.close()
        return

    _apply_update(conn, 'scanner_global_config', diffs, demo, live['id'])
    conn.commit()
    conn.close()
    print(f"\n  PROMOTED: {len(diffs)} scanner parameter(s) to live")
    print("  Restart task-worker-live to pick up new config.")


# =============================================================================
# LPF subcommand
# =============================================================================

def cmd_lpf_status(args):
    conn = get_connection()
    for cs in ('live', 'demo'):
        cfg = _get_single_row(conn, 'loss_prevention_config', cs)
        rules = _get_rows(conn, 'loss_prevention_rules', cs, order_by='category, penalty DESC')
        pairs = _get_rows(conn, 'loss_prevention_pair_config', cs, order_by='epic')
        enabled_rules = [r for r in rules if r.get('is_enabled')]
        print(f"\n  [{cs.upper()}] loss_prevention_config id={cfg['id']}")
        print(f"    is_enabled={cfg.get('is_enabled')}  block_mode={cfg.get('block_mode')}  "
              f"penalty_threshold={cfg.get('penalty_threshold')}")
        print(f"    rules: {len(rules)} total, {len(enabled_rules)} enabled")
        print(f"    pair_configs: {len(pairs)}")
    conn.close()


def cmd_lpf_diff(args):
    conn = get_connection()

    # 1. Global config diff
    live_cfg = _get_single_row(conn, 'loss_prevention_config', 'live')
    demo_cfg = _get_single_row(conn, 'loss_prevention_config', 'demo')
    diffs = _diff_rows(live_cfg, demo_cfg)
    _print_diff_table(diffs, "loss_prevention_config differences (demo vs live)")

    # 2. Rules diff
    live_rules = {r['rule_name']: r for r in _get_rows(conn, 'loss_prevention_rules', 'live', 'rule_name')}
    demo_rules = {r['rule_name']: r for r in _get_rows(conn, 'loss_prevention_rules', 'demo', 'rule_name')}

    rule_diffs = []
    all_names = sorted(set(live_rules) | set(demo_rules))
    rule_skip = SKIP_COLUMNS | {'id'}
    for name in all_names:
        lr = live_rules.get(name, {})
        dr = demo_rules.get(name, {})
        for key in sorted(set(list(lr.keys()) + list(dr.keys()))):
            if key in rule_skip:
                continue
            if str(lr.get(key)) != str(dr.get(key)):
                rule_diffs.append((name, key, lr.get(key), dr.get(key)))

    if rule_diffs:
        print(f"\n{'='*60}")
        print(f"  loss_prevention_rules differences (demo vs live)")
        print(f"{'='*60}")
        for name, key, lv, dv in rule_diffs:
            ls = str(lv)[:15] if lv is not None else 'MISSING'
            ds = str(dv)[:15] if dv is not None else 'MISSING'
            print(f"  {name:<35} {key:<20} live={ls:<15} demo={ds}")
        print(f"\n  Total: {len(rule_diffs)} rule difference(s)")
    else:
        print("\n  loss_prevention_rules: identical")

    # 3. Pair config diff
    live_pairs = {r['epic']: r for r in _get_rows(conn, 'loss_prevention_pair_config', 'live', 'epic')}
    demo_pairs = {r['epic']: r for r in _get_rows(conn, 'loss_prevention_pair_config', 'demo', 'epic')}

    pair_diffs = []
    all_epics = sorted(set(live_pairs) | set(demo_pairs))
    pair_skip = SKIP_COLUMNS | {'id'}
    for epic in all_epics:
        lp = live_pairs.get(epic, {})
        dp = demo_pairs.get(epic, {})
        for key in sorted(set(list(lp.keys()) + list(dp.keys()))):
            if key in pair_skip:
                continue
            if str(lp.get(key)) != str(dp.get(key)):
                short = epic.split('.')[-2] if '.' in epic else epic
                pair_diffs.append((short, key, lp.get(key), dp.get(key)))

    if pair_diffs:
        print(f"\n{'='*60}")
        print(f"  loss_prevention_pair_config differences (demo vs live)")
        print(f"{'='*60}")
        for epic, key, lv, dv in pair_diffs:
            ls = str(lv)[:15] if lv is not None else 'MISSING'
            ds = str(dv)[:15] if dv is not None else 'MISSING'
            print(f"  {epic:<10} {key:<30} live={ls:<15} demo={ds}")
        print(f"\n  Total: {len(pair_diffs)} pair config difference(s)")
    else:
        print("\n  loss_prevention_pair_config: identical")

    conn.close()


def cmd_lpf_promote(args):
    conn = get_connection()

    # 1. Global config
    live_cfg = _get_single_row(conn, 'loss_prevention_config', 'live')
    demo_cfg = _get_single_row(conn, 'loss_prevention_config', 'demo')
    cfg_diffs = _diff_rows(live_cfg, demo_cfg)

    # 2. Rules
    live_rules = {r['rule_name']: r for r in _get_rows(conn, 'loss_prevention_rules', 'live', 'rule_name')}
    demo_rules = {r['rule_name']: r for r in _get_rows(conn, 'loss_prevention_rules', 'demo', 'rule_name')}
    rule_skip = SKIP_COLUMNS | {'id'}

    rule_changes: list[tuple] = []  # (rule_name, key, demo_val, live_id or None)
    all_rule_names = sorted(set(live_rules) | set(demo_rules))
    for name in all_rule_names:
        lr = live_rules.get(name)
        dr = demo_rules.get(name)
        if not dr:
            print(f"  WARNING: rule '{name}' exists in live but not demo — skipping")
            continue
        if not lr:
            print(f"  WARNING: rule '{name}' exists in demo but not live — skipping (add manually if desired)")
            continue
        for key in sorted(set(list(lr.keys()) + list(dr.keys()))):
            if key in rule_skip:
                continue
            if str(lr.get(key)) != str(dr.get(key)):
                rule_changes.append((name, key, dr.get(key), lr['id']))

    # 3. Pair config
    live_pairs = {r['epic']: r for r in _get_rows(conn, 'loss_prevention_pair_config', 'live', 'epic')}
    demo_pairs = {r['epic']: r for r in _get_rows(conn, 'loss_prevention_pair_config', 'demo', 'epic')}
    pair_skip = SKIP_COLUMNS | {'id'}

    pair_changes: list[tuple] = []  # (epic, key, demo_val, live_id)
    for epic in sorted(set(live_pairs) | set(demo_pairs)):
        lp = live_pairs.get(epic)
        dp = demo_pairs.get(epic)
        if not dp or not lp:
            continue
        for key in sorted(set(list(lp.keys()) + list(dp.keys()))):
            if key in pair_skip:
                continue
            if str(lp.get(key)) != str(dp.get(key)):
                pair_changes.append((epic, key, dp.get(key), lp['id']))

    total = len(cfg_diffs) + len(rule_changes) + len(pair_changes)
    if total == 0:
        print("\n  LPF: no differences to promote.")
        conn.close()
        return

    print(f"\n  LPF promotion summary ({total} change(s)):")
    if cfg_diffs:
        print(f"\n  loss_prevention_config ({len(cfg_diffs)} change(s)):")
        for key, lv, dv in cfg_diffs:
            print(f"    {key}: {lv} -> {dv}")
    if rule_changes:
        print(f"\n  loss_prevention_rules ({len(rule_changes)} change(s)):")
        for name, key, dv, _id in rule_changes:
            print(f"    [{name}] {key} -> {dv}")
    if pair_changes:
        print(f"\n  loss_prevention_pair_config ({len(pair_changes)} change(s)):")
        for epic, key, dv, _id in pair_changes:
            short = epic.split('.')[-2] if '.' in epic else epic
            print(f"    [{short}] {key} -> {dv}")

    if not args.confirm:
        print("\n  DRY RUN — add --confirm to apply")
        conn.close()
        return

    # Apply
    if cfg_diffs:
        _apply_update(conn, 'loss_prevention_config', cfg_diffs, demo_cfg, live_cfg['id'])

    if rule_changes:
        # Group by rule id for efficient updates
        by_id: dict[int, list] = {}
        for name, key, dv, live_id in rule_changes:
            by_id.setdefault(live_id, []).append((key, dv))
        for live_id, col_vals in by_id.items():
            set_clauses = [f"{k} = %s" for k, _ in col_vals] + ["updated_at = %s"]
            values = [v for _, v in col_vals] + [datetime.utcnow(), live_id]
            sql = f"UPDATE loss_prevention_rules SET {', '.join(set_clauses)} WHERE id = %s"
            with conn.cursor() as cur:
                cur.execute(sql, values)

    if pair_changes:
        by_id2: dict[int, list] = {}
        for epic, key, dv, live_id in pair_changes:
            by_id2.setdefault(live_id, []).append((key, dv))
        for live_id, col_vals in by_id2.items():
            set_clauses = [f"{k} = %s" for k, _ in col_vals] + ["updated_at = %s"]
            values = [v for _, v in col_vals] + [datetime.utcnow(), live_id]
            sql = f"UPDATE loss_prevention_pair_config SET {', '.join(set_clauses)} WHERE id = %s"
            with conn.cursor() as cur:
                cur.execute(sql, values)

    conn.commit()
    conn.close()
    print(f"\n  PROMOTED: {len(cfg_diffs)} config + {len(rule_changes)} rules + {len(pair_changes)} pair configs")
    print("  Restart task-worker-live to pick up new LPF config.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Config Promotion CLI — Demo to Live')
    subparsers = parser.add_subparsers(dest='command', help='Domain or action')

    # ── Legacy top-level SMC commands (backwards compatible) ──────────────────
    subparsers.add_parser('status', help='[SMC] Show both live and demo configs')
    subparsers.add_parser('diff', help='[SMC] Show differences between live and demo')
    promote_parser = subparsers.add_parser('promote', help='[SMC] Promote demo config to live')
    promote_parser.add_argument('--confirm', action='store_true', help='Actually apply (default: dry run)')

    # ── scanner subcommand ────────────────────────────────────────────────────
    scanner_parser = subparsers.add_parser('scanner', help='Scanner / TradeValidator / Claude AI config')
    scanner_sub = scanner_parser.add_subparsers(dest='action', help='scanner action')
    scanner_sub.add_parser('status', help='Show both scanner configs')
    scanner_sub.add_parser('diff', help='Show scanner config differences')
    scanner_promote = scanner_sub.add_parser('promote', help='Promote scanner demo -> live')
    scanner_promote.add_argument('--confirm', action='store_true', help='Actually apply (default: dry run)')

    # ── lpf subcommand ────────────────────────────────────────────────────────
    lpf_parser = subparsers.add_parser('lpf', help='Loss Prevention Filter config')
    lpf_sub = lpf_parser.add_subparsers(dest='action', help='lpf action')
    lpf_sub.add_parser('status', help='Show LPF status for both environments')
    lpf_sub.add_parser('diff', help='Show LPF differences (config + rules + pair overrides)')
    lpf_promote = lpf_sub.add_parser('promote', help='Promote LPF demo -> live')
    lpf_promote.add_argument('--confirm', action='store_true', help='Actually apply (default: dry run)')

    args = parser.parse_args()

    # Route
    if args.command == 'status':
        cmd_status(args)
    elif args.command == 'diff':
        cmd_diff(args)
    elif args.command == 'promote':
        cmd_promote(args)
    elif args.command == 'scanner':
        action = getattr(args, 'action', None)
        if action == 'status':
            cmd_scanner_status(args)
        elif action == 'diff':
            cmd_scanner_diff(args)
        elif action == 'promote':
            cmd_scanner_promote(args)
        else:
            scanner_parser.print_help()
    elif args.command == 'lpf':
        action = getattr(args, 'action', None)
        if action == 'status':
            cmd_lpf_status(args)
        elif action == 'diff':
            cmd_lpf_diff(args)
        elif action == 'promote':
            cmd_lpf_promote(args)
        else:
            lpf_parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
