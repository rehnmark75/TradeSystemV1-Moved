#!/bin/bash
# =============================================================================
# Azure Backtest Data Push Script
# Pushes local ig_candles and strategy_config data to Azure VM
# =============================================================================

set -euo pipefail

# Configuration
AZURE_VM_IP="${AZURE_VM_IP:?Error: Set AZURE_VM_IP environment variable}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/azure_backtest_rsa}"
DAYS_TO_SYNC="${1:-30}"  # Default: sync last 30 days of candles
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
TEMP_DIR="/tmp/azure_backtest_push_${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Cleanup function
cleanup() {
    echo -e "${BLUE}Cleaning up temporary files...${NC}"
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

echo -e "${BLUE}=== Pushing Data to Azure Backtest VM ===${NC}"
echo "VM IP: ${AZURE_VM_IP}"
echo "Days to sync: ${DAYS_TO_SYNC}"
echo ""

# Create temp directory
mkdir -p "${TEMP_DIR}"

# =============================================================================
# 1. Export ig_candles (1m data)
# =============================================================================
echo -e "${BLUE}[1/5] Exporting ig_candles (last ${DAYS_TO_SYNC} days)...${NC}"

# Get row count first
ROW_COUNT=$(docker exec postgres psql -U postgres -d forex -t -c "
    SELECT COUNT(*) FROM ig_candles
    WHERE timeframe = 1 AND start_time > NOW() - INTERVAL '${DAYS_TO_SYNC} days'
")
echo "  Found ${ROW_COUNT} rows to export"

# Export to CSV
docker exec postgres psql -U postgres -d forex -c "\COPY (
    SELECT start_time, epic, timeframe, open, high, low, close, volume, ltv
    FROM ig_candles
    WHERE timeframe = 1 AND start_time > NOW() - INTERVAL '${DAYS_TO_SYNC} days'
    ORDER BY start_time
) TO STDOUT WITH CSV HEADER" > "${TEMP_DIR}/ig_candles.csv"

# Compress
CANDLE_SIZE=$(du -h "${TEMP_DIR}/ig_candles.csv" | cut -f1)
echo "  CSV size: ${CANDLE_SIZE}"
gzip "${TEMP_DIR}/ig_candles.csv"
COMPRESSED_SIZE=$(du -h "${TEMP_DIR}/ig_candles.csv.gz" | cut -f1)
echo "  Compressed size: ${COMPRESSED_SIZE}"

echo -e "${GREEN}  ig_candles exported successfully${NC}"

# =============================================================================
# 1b. Export ig_candles_backtest (pre-resampled 5m/15m/1h/4h data)
# =============================================================================
echo -e "${BLUE}[1b/6] Exporting ig_candles_backtest (pre-resampled data)...${NC}"

BACKTEST_COUNT=$(docker exec postgres psql -U postgres -d forex -t -c "
    SELECT COUNT(*) FROM ig_candles_backtest
    WHERE start_time > NOW() - INTERVAL '${DAYS_TO_SYNC} days'
" 2>/dev/null || echo "0")
echo "  Found ${BACKTEST_COUNT} pre-resampled candles"

if [[ "${BACKTEST_COUNT}" -gt 0 ]]; then
    docker exec postgres psql -U postgres -d forex -c "\COPY (
        SELECT start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from
        FROM ig_candles_backtest
        WHERE start_time > NOW() - INTERVAL '${DAYS_TO_SYNC} days'
        ORDER BY epic, timeframe, start_time
    ) TO STDOUT WITH CSV HEADER" > "${TEMP_DIR}/ig_candles_backtest.csv"

    BACKTEST_SIZE=$(du -h "${TEMP_DIR}/ig_candles_backtest.csv" | cut -f1)
    echo "  CSV size: ${BACKTEST_SIZE}"
    gzip "${TEMP_DIR}/ig_candles_backtest.csv"
    BACKTEST_COMPRESSED=$(du -h "${TEMP_DIR}/ig_candles_backtest.csv.gz" | cut -f1)
    echo "  Compressed size: ${BACKTEST_COMPRESSED}"
    echo -e "${GREEN}  ig_candles_backtest exported successfully${NC}"
else
    echo -e "${YELLOW}  No ig_candles_backtest data to export (will be generated on first backtest)${NC}"
fi

# =============================================================================
# 2. Export strategy_config tables
# =============================================================================
echo -e "${BLUE}[2/6] Exporting strategy_config tables...${NC}"

docker exec postgres pg_dump -U postgres -d strategy_config \
    --table=smc_simple_global_config \
    --table=smc_simple_pair_overrides \
    --table=scanner_global_config \
    --table=intelligence_global_config \
    --clean --if-exists \
    > "${TEMP_DIR}/strategy_config.sql"

CONFIG_SIZE=$(du -h "${TEMP_DIR}/strategy_config.sql" | cut -f1)
echo "  strategy_config dump size: ${CONFIG_SIZE}"
echo -e "${GREEN}  strategy_config exported successfully${NC}"

# =============================================================================
# 3. Export backtest table schemas (if they don't exist on Azure)
# =============================================================================
echo -e "${BLUE}[3/6] Exporting backtest table schemas...${NC}"

docker exec postgres pg_dump -U postgres -d forex \
    --table=backtest_executions \
    --table=backtest_signals \
    --table=backtest_performance \
    --table=backtest_parallel_runs \
    --table=backtest_job_queue \
    --schema-only \
    --clean --if-exists \
    > "${TEMP_DIR}/backtest_schema.sql"

SCHEMA_SIZE=$(du -h "${TEMP_DIR}/backtest_schema.sql" | cut -f1)
echo "  backtest_schema.sql size: ${SCHEMA_SIZE}"

# Also export ig_candles schema
docker exec postgres pg_dump -U postgres -d forex \
    --table=ig_candles \
    --schema-only \
    --clean --if-exists \
    > "${TEMP_DIR}/ig_candles_schema.sql"

# Export backtest functions
echo "  Exporting backtest functions..."
docker exec postgres pg_dump -U postgres -d forex \
    --schema-only \
    --no-owner \
    --no-privileges \
    -t 'dummy_table_that_does_not_exist' 2>/dev/null || true

# Export functions manually
docker exec postgres psql -U postgres -d forex -c "\sf calculate_backtest_performance" > "${TEMP_DIR}/backtest_functions.sql" 2>/dev/null || true
docker exec postgres psql -U postgres -d forex -c "\sf get_backtest_summary" >> "${TEMP_DIR}/backtest_functions.sql" 2>/dev/null || true
docker exec postgres psql -U postgres -d forex -c "\sf update_parallel_run_progress" >> "${TEMP_DIR}/backtest_functions.sql" 2>/dev/null || true

echo -e "${GREEN}  Schemas exported${NC}"

# =============================================================================
# 4. Upload to Azure VM
# =============================================================================
echo -e "${BLUE}[4/6] Uploading files to Azure VM...${NC}"

# Ensure sync directory exists on Azure
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "azureuser@${AZURE_VM_IP}" \
    "mkdir -p /data/sync"

# Upload files
scp -i "${SSH_KEY}" \
    "${TEMP_DIR}/ig_candles.csv.gz" \
    "${TEMP_DIR}/strategy_config.sql" \
    "${TEMP_DIR}/backtest_schema.sql" \
    "${TEMP_DIR}/ig_candles_schema.sql" \
    "azureuser@${AZURE_VM_IP}:/data/sync/"

# Upload ig_candles_backtest if it exists
if [[ -f "${TEMP_DIR}/ig_candles_backtest.csv.gz" ]]; then
    scp -i "${SSH_KEY}" \
        "${TEMP_DIR}/ig_candles_backtest.csv.gz" \
        "azureuser@${AZURE_VM_IP}:/data/sync/"
fi

echo -e "${GREEN}  Files uploaded to /data/sync/${NC}"

# =============================================================================
# 5. Import on Azure VM
# =============================================================================
echo -e "${BLUE}[5/6] Importing data on Azure VM...${NC}"

ssh -i "${SSH_KEY}" "azureuser@${AZURE_VM_IP}" << 'REMOTE_SCRIPT'
cd /data/sync

echo "  Files in /data/sync:"
ls -la /data/sync/

echo ""
echo "  Decompressing candles..."
gunzip -f ig_candles.csv.gz || true

echo "  Creating required functions..."
docker exec postgres psql -U postgres -d forex -c "
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ language 'plpgsql';

CREATE OR REPLACE FUNCTION update_backtest_executions_updated_at()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ language 'plpgsql';

CREATE OR REPLACE FUNCTION update_backtest_performance_updated_at()
RETURNS TRIGGER AS \$\$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
\$\$ language 'plpgsql';
" || true

echo "  Creating ig_candles table..."
docker exec -i postgres psql -U postgres -d forex < /data/sync/ig_candles_schema.sql || true

echo "  Truncating existing candles (clean import)..."
docker exec postgres psql -U postgres -d forex -c "TRUNCATE ig_candles;" 2>/dev/null || true

echo "  Importing ig_candles..."
docker exec -i postgres psql -U postgres -d forex -c "\COPY ig_candles(start_time, epic, timeframe, open, high, low, close, volume, ltv) FROM '/sync/ig_candles.csv' WITH CSV HEADER"

CANDLE_COUNT=\$(docker exec postgres psql -U postgres -d forex -t -c "SELECT COUNT(*) FROM ig_candles;")
echo "  ig_candles imported: \${CANDLE_COUNT} rows"

# Import ig_candles_backtest - always attempt it
echo ""
echo "  === Importing ig_candles_backtest ==="

# Decompress if .gz exists
if [[ -f /data/sync/ig_candles_backtest.csv.gz ]]; then
    echo "  Decompressing ig_candles_backtest.csv.gz..."
    gunzip -f /data/sync/ig_candles_backtest.csv.gz
fi

# Check if CSV exists now
if [[ -f /data/sync/ig_candles_backtest.csv ]]; then
    echo "  Found ig_candles_backtest.csv"

    echo "  Creating ig_candles_backtest table..."
    docker exec postgres psql -U postgres -d forex -c "
    CREATE TABLE IF NOT EXISTS ig_candles_backtest (
        start_time timestamp without time zone NOT NULL,
        epic character varying NOT NULL,
        timeframe integer NOT NULL,
        open double precision NOT NULL,
        high double precision NOT NULL,
        low double precision NOT NULL,
        close double precision NOT NULL,
        volume integer NOT NULL,
        ltv integer,
        resampled_from integer DEFAULT 1,
        created_at timestamp without time zone DEFAULT now(),
        PRIMARY KEY (start_time, epic, timeframe)
    );
    CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic ON ig_candles_backtest(epic);
    CREATE INDEX IF NOT EXISTS idx_backtest_candles_epic_tf_time ON ig_candles_backtest(epic, timeframe, start_time DESC);
    " || true

    echo "  Truncating existing backtest candles..."
    docker exec postgres psql -U postgres -d forex -c "TRUNCATE ig_candles_backtest;" 2>/dev/null || true

    echo "  Importing ig_candles_backtest..."
    docker exec -i postgres psql -U postgres -d forex -c "\COPY ig_candles_backtest(start_time, epic, timeframe, open, high, low, close, volume, ltv, resampled_from) FROM '/sync/ig_candles_backtest.csv' WITH CSV HEADER"

    BACKTEST_COUNT=\$(docker exec postgres psql -U postgres -d forex -t -c "SELECT COUNT(*) FROM ig_candles_backtest;")
    echo "  ig_candles_backtest imported: \${BACKTEST_COUNT} rows"

    echo "  Breakdown by timeframe:"
    docker exec postgres psql -U postgres -d forex -c "SELECT timeframe, COUNT(*) as count FROM ig_candles_backtest GROUP BY timeframe ORDER BY timeframe;"
else
    echo "  ERROR: ig_candles_backtest.csv not found!"
    echo "  Files in /data/sync:"
    ls -la /data/sync/
fi

echo "  Creating backtest tables..."
docker exec -i postgres psql -U postgres -d forex < /data/sync/backtest_schema.sql

echo "  Creating trade_log table..."
docker exec postgres psql -U postgres -d forex -c "
CREATE TABLE IF NOT EXISTS trade_log (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(100) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    limit_price DOUBLE PRECISION,
    sl_price DOUBLE PRECISION,
    tp_price DOUBLE PRECISION,
    direction VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITHOUT TIME ZONE,
    moved_to_breakeven BOOLEAN NOT NULL DEFAULT FALSE,
    deal_id VARCHAR(100),
    deal_reference VARCHAR(100),
    endpoint VARCHAR(200),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    trigger_distance DOUBLE PRECISION,
    min_stop_distance_points DOUBLE PRECISION,
    trigger_time TIMESTAMP WITHOUT TIME ZONE,
    last_trigger_price DOUBLE PRECISION,
    monitor_until TIMESTAMP WITHOUT TIME ZONE,
    closed_at TIMESTAMP WITHOUT TIME ZONE,
    alert_id INTEGER,
    profit_loss NUMERIC(12,2),
    pnl_currency VARCHAR(10),
    updated_at TIMESTAMP WITHOUT TIME ZONE,
    pnl_updated_at TIMESTAMP WITHOUT TIME ZONE,
    position_reference VARCHAR(20),
    activity_correlated BOOLEAN,
    lifecycle_duration_minutes INTEGER,
    pips_gained DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_trade_log_symbol ON trade_log(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_log_status ON trade_log(status);
CREATE INDEX IF NOT EXISTS idx_trade_log_closed_at ON trade_log(closed_at DESC);
"

echo "  Creating strategy_config database if needed..."
docker exec postgres psql -U postgres -c "CREATE DATABASE strategy_config;" 2>/dev/null || true

echo "  Importing strategy_config..."
docker exec -i postgres psql -U postgres -d strategy_config < /data/sync/strategy_config.sql

# Verify strategy_config import
echo "  Verifying strategy_config import..."
GLOBAL_CONFIG_COUNT=\$(docker exec postgres psql -U postgres -d strategy_config -t -c "SELECT COUNT(*) FROM smc_simple_global_config WHERE is_active = TRUE;" 2>/dev/null || echo "0")
PAIR_OVERRIDE_COUNT=\$(docker exec postgres psql -U postgres -d strategy_config -t -c "SELECT COUNT(*) FROM smc_simple_pair_overrides;" 2>/dev/null || echo "0")
SCANNER_CONFIG_COUNT=\$(docker exec postgres psql -U postgres -d strategy_config -t -c "SELECT COUNT(*) FROM scanner_global_config WHERE is_active = TRUE;" 2>/dev/null || echo "0")

echo "    smc_simple_global_config (active): \${GLOBAL_CONFIG_COUNT}"
echo "    smc_simple_pair_overrides: \${PAIR_OVERRIDE_COUNT}"
echo "    scanner_global_config (active): \${SCANNER_CONFIG_COUNT}"

if [[ "\${GLOBAL_CONFIG_COUNT}" -lt 1 ]] || [[ "\${PAIR_OVERRIDE_COUNT}" -lt 1 ]]; then
    echo ""
    echo "  *** WARNING: strategy_config import may have failed! ***"
    echo "  Expected at least 1 active global config and pair overrides."
    echo "  Please verify the configuration on the Azure VM."
else
    echo "  strategy_config import verified successfully"
fi

echo "  Creating backtest functions..."
docker exec postgres psql -U postgres -d forex -c "
CREATE OR REPLACE FUNCTION public.calculate_backtest_performance(p_execution_id integer, p_epic character varying DEFAULT NULL::character varying)
 RETURNS void
 LANGUAGE plpgsql
AS \\\$function\\\$
DECLARE
    rec RECORD;
BEGIN
    FOR rec IN
        SELECT epic, timeframe, strategy_name
        FROM backtest_signals
        WHERE execution_id = p_execution_id
          AND (p_epic IS NULL OR epic = p_epic)
        GROUP BY epic, timeframe, strategy_name
    LOOP
        INSERT INTO backtest_performance (
            execution_id, epic, timeframe, strategy_name,
            total_signals, bull_signals, bear_signals, validated_signals,
            winning_trades, losing_trades, breakeven_trades, win_rate,
            total_pips, avg_win_pips, avg_loss_pips, max_win_pips, max_loss_pips,
            profit_factor, expectancy_per_trade, max_drawdown_pips,
            avg_trade_duration_minutes, signal_frequency_per_day,
            data_completeness_score
        )
        SELECT
            p_execution_id, rec.epic, rec.timeframe, rec.strategy_name,
            COUNT(*), COUNT(CASE WHEN signal_type = 'BULL' THEN 1 END),
            COUNT(CASE WHEN signal_type = 'BEAR' THEN 1 END),
            COUNT(CASE WHEN validation_passed = TRUE THEN 1 END),
            COUNT(CASE WHEN trade_result = 'win' THEN 1 END),
            COUNT(CASE WHEN trade_result = 'loss' THEN 1 END),
            COUNT(CASE WHEN trade_result = 'breakeven' THEN 1 END),
            CASE WHEN COUNT(CASE WHEN trade_result IN ('win', 'loss') THEN 1 END) > 0
                 THEN COUNT(CASE WHEN trade_result = 'win' THEN 1 END)::DECIMAL /
                      COUNT(CASE WHEN trade_result IN ('win', 'loss') THEN 1 END)
                 ELSE NULL END,
            COALESCE(SUM(pips_gained), 0),
            AVG(CASE WHEN trade_result = 'win' THEN pips_gained END),
            AVG(CASE WHEN trade_result = 'loss' THEN pips_gained END),
            MAX(CASE WHEN trade_result = 'win' THEN pips_gained END),
            MIN(CASE WHEN trade_result = 'loss' THEN pips_gained END),
            CASE WHEN SUM(CASE WHEN trade_result = 'loss' THEN ABS(pips_gained) END) > 0
                 THEN SUM(CASE WHEN trade_result = 'win' THEN pips_gained ELSE 0 END) /
                      SUM(CASE WHEN trade_result = 'loss' THEN ABS(pips_gained) ELSE 0 END)
                 ELSE NULL END,
            AVG(pips_gained), 0, AVG(holding_time_minutes),
            COUNT(*)::DECIMAL / GREATEST(1, EXTRACT(days FROM MAX(signal_timestamp) - MIN(signal_timestamp))),
            AVG(data_completeness)
        FROM backtest_signals
        WHERE execution_id = p_execution_id AND epic = rec.epic AND timeframe = rec.timeframe AND strategy_name = rec.strategy_name
        ON CONFLICT (execution_id, epic, timeframe, strategy_name) DO UPDATE SET
            total_signals = EXCLUDED.total_signals, bull_signals = EXCLUDED.bull_signals, bear_signals = EXCLUDED.bear_signals,
            validated_signals = EXCLUDED.validated_signals, winning_trades = EXCLUDED.winning_trades, losing_trades = EXCLUDED.losing_trades,
            breakeven_trades = EXCLUDED.breakeven_trades, win_rate = EXCLUDED.win_rate, total_pips = EXCLUDED.total_pips,
            avg_win_pips = EXCLUDED.avg_win_pips, avg_loss_pips = EXCLUDED.avg_loss_pips, max_win_pips = EXCLUDED.max_win_pips,
            max_loss_pips = EXCLUDED.max_loss_pips, profit_factor = EXCLUDED.profit_factor, expectancy_per_trade = EXCLUDED.expectancy_per_trade,
            avg_trade_duration_minutes = EXCLUDED.avg_trade_duration_minutes, signal_frequency_per_day = EXCLUDED.signal_frequency_per_day,
            data_completeness_score = EXCLUDED.data_completeness_score, updated_at = NOW();
    END LOOP;
END;
\\\$function\\\$;

CREATE OR REPLACE FUNCTION public.get_backtest_summary(p_execution_id integer)
 RETURNS TABLE(execution_name character varying, strategy character varying, status character varying, total_signals bigint, total_validated_signals bigint, avg_win_rate numeric, total_pips numeric, avg_profit_factor numeric, data_quality numeric)
 LANGUAGE plpgsql
AS \\\$function\\\$
BEGIN
    RETURN QUERY
    SELECT be.execution_name, be.strategy_name, be.status,
        COALESCE(SUM(bp.total_signals), 0), COALESCE(SUM(bp.validated_signals), 0),
        AVG(bp.win_rate), COALESCE(SUM(bp.total_pips), 0), AVG(bp.profit_factor), be.quality_score
    FROM backtest_executions be
    LEFT JOIN backtest_performance bp ON be.id = bp.execution_id
    WHERE be.id = p_execution_id
    GROUP BY be.id, be.execution_name, be.strategy_name, be.status, be.quality_score;
END;
\\\$function\\\$;

CREATE OR REPLACE FUNCTION public.update_parallel_run_progress(p_run_id integer, p_chunk_execution_id integer)
 RETURNS void
 LANGUAGE plpgsql
AS \\\$function\\\$
BEGIN
    UPDATE backtest_parallel_runs
    SET chunk_execution_ids = array_append(chunk_execution_ids, p_chunk_execution_id),
        completed_chunks = completed_chunks + 1,
        status = CASE WHEN completed_chunks + 1 >= total_chunks THEN 'aggregating' ELSE 'running' END
    WHERE id = p_run_id;
END;
\\\$function\\\$;
"

echo "  Verifying import..."
CANDLE_COUNT=\$(docker exec postgres psql -U postgres -d forex -t -c "SELECT COUNT(*) FROM ig_candles;")
echo "  ig_candles row count: \${CANDLE_COUNT}"

BACKTEST_CANDLE_COUNT=\$(docker exec postgres psql -U postgres -d forex -t -c "SELECT COUNT(*) FROM ig_candles_backtest;" 2>/dev/null || echo "0")
echo "  ig_candles_backtest row count: \${BACKTEST_CANDLE_COUNT}"

# Final strategy_config verification with actual values
echo ""
echo "  === FINAL IMPORT VERIFICATION ==="
FINAL_GLOBAL=\$(docker exec postgres psql -U postgres -d strategy_config -t -c "SELECT COUNT(*) FROM smc_simple_global_config WHERE is_active = TRUE;" 2>/dev/null || echo "ERROR")
FINAL_PAIRS=\$(docker exec postgres psql -U postgres -d strategy_config -t -c "SELECT COUNT(*) FROM smc_simple_pair_overrides;" 2>/dev/null || echo "ERROR")
FINAL_SCANNER=\$(docker exec postgres psql -U postgres -d strategy_config -t -c "SELECT COUNT(*) FROM scanner_global_config WHERE is_active = TRUE;" 2>/dev/null || echo "ERROR")

echo "  ig_candles:              \${CANDLE_COUNT} rows"
echo "  ig_candles_backtest:     \${BACKTEST_CANDLE_COUNT} rows"
echo "  smc_simple_global_config: \${FINAL_GLOBAL} active config(s)"
echo "  smc_simple_pair_overrides: \${FINAL_PAIRS} pair(s)"
echo "  scanner_global_config:   \${FINAL_SCANNER} active config(s)"

# Show per-pair settings as final confirmation
echo ""
echo "  Per-pair SL/TP settings imported:"
docker exec postgres psql -U postgres -d strategy_config -c "
SELECT epic,
       COALESCE(fixed_stop_loss_pips::text, 'default') as sl_pips,
       COALESCE(fixed_take_profit_pips::text, 'default') as tp_pips,
       COALESCE(min_confidence::text, 'default') as min_conf
FROM smc_simple_pair_overrides
ORDER BY epic;" 2>/dev/null || echo "  Could not query pair overrides"

echo ""
echo "  Cleanup sync files..."
rm -f /data/sync/*.csv /data/sync/*.sql /data/sync/*.gz
REMOTE_SCRIPT

echo ""
echo -e "${GREEN}=== Data Push Complete ===${NC}"
echo ""
echo "Summary:"
echo "  - ig_candles: ${ROW_COUNT} rows (last ${DAYS_TO_SYNC} days, 1m data)"
echo "  - ig_candles_backtest: ${BACKTEST_COUNT:-0} rows (pre-resampled 5m/15m/1h/4h)"
echo "  - strategy_config: All config tables"
echo "  - backtest schemas: Created if missing"
echo ""
echo "Next: Run backtest with ./scripts/azure_backtest.sh run EURUSD 14"
