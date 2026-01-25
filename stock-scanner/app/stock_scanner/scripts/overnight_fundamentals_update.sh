#!/bin/bash
# Overnight Fundamentals Update Script
# Runs with conservative settings to avoid rate limiting
# Designed to be run via cron at 2 AM when rate limits have reset

LOG_FILE="/app/stock_scanner/logs/overnight_fundamentals_$(date +%Y%m%d).log"

echo "========================================" >> $LOG_FILE
echo "OVERNIGHT FUNDAMENTALS UPDATE" >> $LOG_FILE
echo "Started: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE

# Step 1: Update missing/stale tickers (NOT --force)
echo "" >> $LOG_FILE
echo "[STEP 1] Updating missing/stale tickers..." >> $LOG_FILE
python3 -u /app/stock_scanner/scripts/update_fundamentals.py \
    --concurrency 1 \
    --delay 2.0 \
    --max-retries 3 \
    >> $LOG_FILE 2>&1

# Wait 5 minutes for rate limit cooldown
echo "" >> $LOG_FILE
echo "[COOLDOWN] Waiting 5 minutes before retry round..." >> $LOG_FILE
sleep 300

# Step 2: Retry any failures from step 1
echo "" >> $LOG_FILE
echo "[STEP 2] Retrying failed tickers..." >> $LOG_FILE
python3 -u /app/stock_scanner/scripts/update_fundamentals.py \
    --retry-failed \
    --concurrency 1 \
    --delay 3.0 \
    --max-retries 2 \
    >> $LOG_FILE 2>&1

# Step 3: Report final coverage
echo "" >> $LOG_FILE
echo "[STEP 3] Final coverage report..." >> $LOG_FILE
python3 -c "
import asyncio
import sys
sys.path.insert(0, '/app')
from stock_scanner.core.database.async_database_manager import AsyncDatabaseManager
from stock_scanner import config

async def report():
    db = AsyncDatabaseManager(config.STOCKS_DATABASE_URL)
    await db.connect()

    total = await db.fetchval('SELECT COUNT(*) FROM stock_instruments WHERE is_active = TRUE AND is_tradeable = TRUE')
    with_fund = await db.fetchval('SELECT COUNT(*) FROM stock_instruments WHERE is_active = TRUE AND is_tradeable = TRUE AND sector IS NOT NULL')
    with_cap = await db.fetchval('SELECT COUNT(*) FROM stock_instruments WHERE is_active = TRUE AND is_tradeable = TRUE AND market_cap IS NOT NULL')
    never_updated = await db.fetchval('SELECT COUNT(*) FROM stock_instruments WHERE is_active = TRUE AND is_tradeable = TRUE AND fundamentals_updated_at IS NULL')

    print(f'=== FINAL COVERAGE REPORT ===')
    print(f'Total active stocks: {total}')
    print(f'With sector: {with_fund} ({100*with_fund/total:.1f}%)')
    print(f'With market cap: {with_cap} ({100*with_cap/total:.1f}%)')
    print(f'Never updated: {never_updated}')

    await db.close()

asyncio.run(report())
" >> $LOG_FILE 2>&1

echo "" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "COMPLETED: $(date)" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
