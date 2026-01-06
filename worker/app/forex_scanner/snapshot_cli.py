#!/usr/bin/env python3
"""
snapshot_cli.py - Config Snapshot Management CLI

Manage named parameter configurations for backtest testing without affecting live trading.

Usage:
    python snapshot_cli.py create "tight_sl" --set fixed_stop_loss_pips=8 --desc "Testing tighter SL"
    python snapshot_cli.py list
    python snapshot_cli.py show tight_sl
    python snapshot_cli.py delete tight_sl
    python snapshot_cli.py promote tight_sl --confirm
"""

import argparse
import sys
import json
from datetime import datetime

# Set up path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/forex_scanner')

try:
    from forex_scanner.services.backtest_config_service import get_backtest_config_service
except ImportError:
    from services.backtest_config_service import get_backtest_config_service


def parse_set_args(set_args):
    """Parse --set PARAM=VALUE arguments into a dictionary"""
    if not set_args:
        return {}

    overrides = {}
    for item in set_args:
        if '=' not in item:
            print(f"⚠️ Invalid format '{item}' - expected PARAM=VALUE, skipping")
            continue

        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()

        # Auto-convert types
        try:
            if value.lower() in ('true', 'false'):
                overrides[key] = value.lower() == 'true'
            elif '.' in value:
                overrides[key] = float(value)
            else:
                overrides[key] = int(value)
        except ValueError:
            overrides[key] = value  # Keep as string

    return overrides


def cmd_create(args):
    """Create a new snapshot"""
    service = get_backtest_config_service()

    overrides = parse_set_args(args.set)
    if not overrides:
        print("❌ No parameters specified. Use --set PARAM=VALUE to set parameters.")
        return 1

    print(f"\n📦 Creating snapshot '{args.name}'")
    print(f"   Description: {args.description or '(none)'}")
    print(f"   Parameters: {len(overrides)}")
    for k, v in overrides.items():
        print(f"      - {k}: {v} ({type(v).__name__})")

    tags = args.tags.split(',') if args.tags else []

    snapshot_id = service.create_snapshot(
        name=args.name,
        parameter_overrides=overrides,
        description=args.description,
        created_by=args.created_by or 'cli',
        tags=tags
    )

    if snapshot_id:
        print(f"\n✅ Created snapshot '{args.name}' (ID: {snapshot_id})")
        print(f"\n   Use it with: python bt.py EURUSD 7 --snapshot {args.name}")
        return 0
    else:
        print(f"\n❌ Failed to create snapshot")
        return 1


def cmd_list(args):
    """List all snapshots"""
    service = get_backtest_config_service()

    snapshots = service.list_snapshots(
        include_inactive=args.all,
        limit=args.limit
    )

    if not snapshots:
        print("\n📦 No snapshots found")
        print("   Create one with: python snapshot_cli.py create NAME --set PARAM=VALUE")
        return 0

    print(f"\n📦 Config Snapshots ({len(snapshots)} found)")
    print("=" * 90)
    print(f"{'NAME':<25} {'PARAMS':<8} {'TESTS':<6} {'WIN RATE':<10} {'CREATED':<20} {'STATUS'}")
    print("-" * 90)

    for snap in snapshots:
        param_count = len(snap.parameter_overrides)
        win_rate = "-"
        if snap.test_results and snap.test_results.get('win_rate'):
            win_rate = f"{snap.test_results['win_rate']:.1%}"

        created = snap.created_at.strftime('%Y-%m-%d %H:%M') if snap.created_at else '-'

        status = []
        if snap.is_promoted:
            status.append("📈 PROMOTED")
        if not snap.is_active:
            status.append("❌ INACTIVE")
        status_str = " ".join(status) if status else "✅ ACTIVE"

        print(f"{snap.snapshot_name:<25} {param_count:<8} {snap.test_count:<6} {win_rate:<10} {created:<20} {status_str}")

    print("=" * 90)
    return 0


def cmd_show(args):
    """Show snapshot details"""
    service = get_backtest_config_service()

    snapshot = service.get_snapshot(args.name)
    if not snapshot:
        print(f"\n❌ Snapshot '{args.name}' not found")
        return 1

    print(f"\n📦 Snapshot: {snapshot.snapshot_name}")
    print("=" * 60)
    print(f"   ID: {snapshot.id}")
    print(f"   Description: {snapshot.description or '(none)'}")
    print(f"   Created: {snapshot.created_at}")
    print(f"   Created by: {snapshot.created_by}")
    print(f"   Base version: {snapshot.base_config_version or 'unknown'}")
    print(f"   Test count: {snapshot.test_count}")
    print(f"   Is promoted: {snapshot.is_promoted}")
    print(f"   Tags: {', '.join(snapshot.tags) if snapshot.tags else '(none)'}")

    print(f"\n📊 Parameter Overrides ({len(snapshot.parameter_overrides)}):")
    print("-" * 40)
    for param, value in snapshot.parameter_overrides.items():
        print(f"   {param}: {value}")

    if snapshot.test_results:
        print(f"\n📈 Last Test Results:")
        print("-" * 40)
        for key, value in snapshot.test_results.items():
            if isinstance(value, float):
                if 'rate' in key.lower():
                    print(f"   {key}: {value:.1%}")
                else:
                    print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

    # Show test history
    history = service.get_test_history(args.name, limit=5)
    if history:
        print(f"\n📜 Recent Test History:")
        print("-" * 60)
        for test in history:
            date = test['tested_at'].strftime('%Y-%m-%d %H:%M') if test['tested_at'] else '-'
            win_rate = f"{test['win_rate']:.1%}" if test['win_rate'] else '-'
            pf = f"{test['profit_factor']:.2f}" if test['profit_factor'] else '-'
            print(f"   {date} | {test['epic_tested'] or 'all'} | {test['days_tested'] or '-'}d | "
                  f"Signals: {test['total_signals']} | Win: {win_rate} | PF: {pf}")

    print("\n" + "=" * 60)
    print(f"Use with: python bt.py EURUSD 7 --snapshot {snapshot.snapshot_name}")
    return 0


def cmd_delete(args):
    """Delete a snapshot"""
    service = get_backtest_config_service()

    snapshot = service.get_snapshot(args.name)
    if not snapshot:
        print(f"\n❌ Snapshot '{args.name}' not found")
        return 1

    if not args.confirm:
        print(f"\n⚠️ This will {'permanently delete' if args.hard else 'deactivate'} snapshot '{args.name}'")
        print(f"   Use --confirm to proceed")
        return 1

    success = service.delete_snapshot(args.name, hard_delete=args.hard)
    if success:
        action = "Deleted" if args.hard else "Deactivated"
        print(f"\n✅ {action} snapshot '{args.name}'")
        return 0
    else:
        print(f"\n❌ Failed to delete snapshot")
        return 1


def cmd_promote(args):
    """Promote snapshot to live configuration"""
    service = get_backtest_config_service()

    snapshot = service.get_snapshot(args.name)
    if not snapshot:
        print(f"\n❌ Snapshot '{args.name}' not found")
        return 1

    dry_run = not args.confirm

    print(f"\n📈 Promoting snapshot '{args.name}' to live configuration")
    print("=" * 60)

    result = service.promote_to_live(
        name=args.name,
        promoted_by=args.promoted_by or 'cli',
        notes=args.notes,
        dry_run=dry_run
    )

    if not result.get('success', False) and 'error' in result:
        print(f"\n❌ {result['error']}")
        return 1

    print(f"\n   Parameters to change:")
    for param in result.get('parameters_to_change', []):
        value = result.get('overrides', {}).get(param)
        print(f"      - {param}: {value}")

    if result.get('last_test_results'):
        print(f"\n   Last test results:")
        for key, value in result['last_test_results'].items():
            if isinstance(value, float):
                print(f"      - {key}: {value:.2f}")
            else:
                print(f"      - {key}: {value}")

    if dry_run:
        print(f"\n⚠️ DRY RUN - No changes made")
        print(f"   Add --confirm to apply these changes to live configuration")
    else:
        print(f"\n{result.get('message', 'Promotion complete')}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Config Snapshot Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a snapshot with tighter stop loss
  python snapshot_cli.py create tight_sl --set fixed_stop_loss_pips=8 --set sl_buffer_pips=1 \\
      --desc "Testing tighter SL for EURUSD"

  # List all snapshots
  python snapshot_cli.py list

  # Show snapshot details
  python snapshot_cli.py show tight_sl

  # Delete a snapshot
  python snapshot_cli.py delete tight_sl --confirm

  # Promote to live (dry run first)
  python snapshot_cli.py promote tight_sl
  python snapshot_cli.py promote tight_sl --confirm

Use snapshots in backtests:
  python bt.py EURUSD 14 --snapshot tight_sl
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new snapshot')
    create_parser.add_argument('name', help='Unique snapshot name')
    create_parser.add_argument('--set', action='append', metavar='PARAM=VALUE',
                               help='Set parameter override (can be used multiple times)')
    create_parser.add_argument('--desc', '--description', dest='description',
                               help='Snapshot description')
    create_parser.add_argument('--tags', help='Comma-separated tags')
    create_parser.add_argument('--created-by', help='Creator name')

    # List command
    list_parser = subparsers.add_parser('list', help='List all snapshots')
    list_parser.add_argument('--all', action='store_true', help='Include inactive snapshots')
    list_parser.add_argument('--limit', type=int, default=50, help='Max results')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show snapshot details')
    show_parser.add_argument('name', help='Snapshot name')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a snapshot')
    delete_parser.add_argument('name', help='Snapshot name')
    delete_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    delete_parser.add_argument('--hard', action='store_true', help='Permanently delete')

    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote snapshot to live config')
    promote_parser.add_argument('name', help='Snapshot name')
    promote_parser.add_argument('--confirm', action='store_true', help='Actually apply changes')
    promote_parser.add_argument('--notes', help='Promotion notes')
    promote_parser.add_argument('--promoted-by', help='Who is promoting')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Route to command handler
    commands = {
        'create': cmd_create,
        'list': cmd_list,
        'show': cmd_show,
        'delete': cmd_delete,
        'promote': cmd_promote
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
