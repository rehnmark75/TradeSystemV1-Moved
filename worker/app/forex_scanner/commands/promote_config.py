#!/usr/bin/env python3
"""
Config Promotion CLI — Compare and promote demo config to live.

Usage:
    python -m forex_scanner.commands.promote_config diff        # Show differences
    python -m forex_scanner.commands.promote_config promote     # Promote demo -> live (dry run)
    python -m forex_scanner.commands.promote_config promote --confirm  # Actually promote
    python -m forex_scanner.commands.promote_config status      # Show both configs
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


def main():
    parser = argparse.ArgumentParser(description='Config Promotion CLI — Demo to Live')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('status', help='Show both live and demo configs')
    subparsers.add_parser('diff', help='Show differences between live and demo')

    promote_parser = subparsers.add_parser('promote', help='Promote demo config to live')
    promote_parser.add_argument('--confirm', action='store_true', help='Actually apply changes (default: dry run)')

    args = parser.parse_args()

    if args.command == 'status':
        cmd_status(args)
    elif args.command == 'diff':
        cmd_diff(args)
    elif args.command == 'promote':
        cmd_promote(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
