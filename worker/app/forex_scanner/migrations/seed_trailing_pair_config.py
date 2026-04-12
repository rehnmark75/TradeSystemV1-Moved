#!/usr/bin/env python3
"""
Seed the trailing_pair_config table from the legacy Python dicts in
dev-app/config.py.

Run inside the fastapi-dev container (has access to dev-app/config.py):
    docker exec -it fastapi-dev python3 /app/migrations_seed_trailing_pair_config.py

or directly via docker:
    docker cp worker/app/forex_scanner/migrations/seed_trailing_pair_config.py \
      fastapi-dev:/tmp/seed.py
    docker exec fastapi-dev python3 /tmp/seed.py

Idempotent: uses ON CONFLICT DO NOTHING — only inserts missing rows.
Creates rows for BOTH config_set='demo' and config_set='live' with identical
values (the user diverges them later via the UI).
"""

import os
import sys

# Ensure /app is importable (script run location-agnostic)
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

import psycopg2
from psycopg2.extras import execute_values

from config import (  # noqa: E402  (after sys.path fix)
    PAIR_TRAILING_CONFIGS,
    SCALP_TRAILING_CONFIGS,
    DEFAULT_TRAILING_CONFIG,
    DEFAULT_SCALP_TRAILING_CONFIG,
)


FIELDS = [
    'early_breakeven_trigger_points',
    'early_breakeven_buffer_points',
    'stage1_trigger_points',
    'stage1_lock_points',
    'stage2_trigger_points',
    'stage2_lock_points',
    'stage3_trigger_points',
    'stage3_atr_multiplier',
    'stage3_min_distance',
    'min_trail_distance',
    'break_even_trigger_points',
    'enable_partial_close',
    'partial_close_trigger_points',
    'partial_close_size',
]


def row_tuple(config_set, epic, is_scalp, cfg):
    return (
        config_set,
        epic,
        is_scalp,
        *[cfg.get(f) for f in FIELDS],
        'seed-script',
        'Initial seed from legacy Python dicts',
    )


def main():
    db_url = os.getenv(
        'STRATEGY_CONFIG_DATABASE_URL',
        'postgresql://postgres:postgres@postgres:5432/strategy_config',
    )
    rows = []

    # Regular trailing configs
    for epic, cfg in PAIR_TRAILING_CONFIGS.items():
        for config_set in ('demo', 'live'):
            rows.append(row_tuple(config_set, epic, False, cfg))

    # Scalp trailing configs
    for epic, cfg in SCALP_TRAILING_CONFIGS.items():
        for config_set in ('demo', 'live'):
            rows.append(row_tuple(config_set, epic, True, cfg))

    # DEFAULT fallback rows
    for config_set in ('demo', 'live'):
        rows.append(row_tuple(config_set, 'DEFAULT', False, DEFAULT_TRAILING_CONFIG))
        rows.append(row_tuple(config_set, 'DEFAULT', True, DEFAULT_SCALP_TRAILING_CONFIG))

    insert_cols = ['config_set', 'epic', 'is_scalp', *FIELDS, 'updated_by', 'change_reason']
    sql = (
        f"INSERT INTO trailing_pair_config ({', '.join(insert_cols)}) "
        f"VALUES %s "
        f"ON CONFLICT (config_set, epic, is_scalp) DO NOTHING "
        f"RETURNING id, config_set, epic, is_scalp"
    )

    print(f"Seeding {len(rows)} rows into trailing_pair_config...")
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            inserted = execute_values(cur, sql, rows, fetch=True)
            print(f"Inserted {len(inserted)} new rows (conflicts skipped).")
            for row in inserted:
                print(f"  id={row[0]}  {row[1]:4}  {row[2]:30}  scalp={row[3]}")
        conn.commit()


if __name__ == '__main__':
    main()
