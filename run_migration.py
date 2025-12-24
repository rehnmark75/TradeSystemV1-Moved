#!/usr/bin/env python3
"""
Migration script for early breakeven tracking (v2.8.0)
Adds columns to track early breakeven execution before partial close.

Run inside fastapi-dev container:
docker exec fastapi-dev python3 /app/run_migration.py
"""
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/trading')
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Check if columns exist
    result = conn.execute(text("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'trade_log'
        AND column_name IN ('early_be_executed', 'early_be_time')
    """))
    existing = [row[0] for row in result.fetchall()]
    print(f'Existing columns: {existing}')

    if 'early_be_executed' not in existing:
        conn.execute(text('ALTER TABLE trade_log ADD COLUMN early_be_executed BOOLEAN NOT NULL DEFAULT FALSE'))
        print('Added early_be_executed column')
    else:
        print('early_be_executed column already exists')

    if 'early_be_time' not in existing:
        conn.execute(text('ALTER TABLE trade_log ADD COLUMN early_be_time TIMESTAMP'))
        print('Added early_be_time column')
    else:
        print('early_be_time column already exists')

    conn.commit()
    print('Migration complete!')
