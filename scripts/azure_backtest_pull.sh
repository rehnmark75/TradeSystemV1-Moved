#!/bin/bash
# =============================================================================
# Azure Backtest Results Pull Script
# Pulls backtest results from Azure VM back to local database
# Results get ID offset of +1,000,000 to avoid conflicts
# =============================================================================

set -euo pipefail

# Configuration
AZURE_VM_IP="${AZURE_VM_IP:?Error: Set AZURE_VM_IP environment variable}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/azure_backtest_rsa}"
EXECUTION_ID="${1:-}"  # Optional: specific execution ID to pull
ID_OFFSET=1000000      # Offset to add to Azure IDs
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
TEMP_DIR="/tmp/azure_backtest_pull_${TIMESTAMP}"

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

echo -e "${BLUE}=== Pulling Results from Azure Backtest VM ===${NC}"
echo "VM IP: ${AZURE_VM_IP}"
echo "ID Offset: +${ID_OFFSET}"
echo ""

# Create temp directory
mkdir -p "${TEMP_DIR}"

# =============================================================================
# 1. List available executions on Azure (if no ID specified)
# =============================================================================
if [[ -z "${EXECUTION_ID}" ]]; then
    echo -e "${BLUE}[1/4] Listing recent backtest executions on Azure...${NC}"

    ssh -i "${SSH_KEY}" "azureuser@${AZURE_VM_IP}" << 'REMOTE_SCRIPT'
docker exec postgres psql -U postgres -d forex -c "
    SELECT
        id,
        execution_name,
        strategy_name,
        status,
        to_char(start_time, 'YYYY-MM-DD HH24:MI') as started,
        COALESCE(total_combinations, 0) as signals
    FROM backtest_executions
    ORDER BY start_time DESC
    LIMIT 20;
"
REMOTE_SCRIPT

    echo ""
    echo -e "${YELLOW}Usage: $0 <execution_id>${NC}"
    echo "Example: $0 42"
    echo ""
    echo "To pull ALL recent results: $0 all"
    exit 0
fi

# =============================================================================
# 2. Export results from Azure
# =============================================================================
echo -e "${BLUE}[2/4] Exporting results from Azure...${NC}"

if [[ "${EXECUTION_ID}" == "all" ]]; then
    echo "  Exporting ALL backtest results..."

    ssh -i "${SSH_KEY}" "azureuser@${AZURE_VM_IP}" << 'REMOTE_SCRIPT'
docker exec postgres pg_dump -U postgres -d forex \
    --table=backtest_executions \
    --table=backtest_signals \
    --table=backtest_performance \
    --table=backtest_parallel_runs \
    --data-only \
    --column-inserts \
    > /data/sync/backtest_results.sql
REMOTE_SCRIPT
else
    echo "  Exporting execution ID: ${EXECUTION_ID}"

    ssh -i "${SSH_KEY}" "azureuser@${AZURE_VM_IP}" << REMOTE_SCRIPT
# Export specific execution and its related data
docker exec postgres psql -U postgres -d forex -c "\COPY (
    SELECT * FROM backtest_executions WHERE id = ${EXECUTION_ID}
) TO STDOUT WITH CSV HEADER" > /data/sync/executions.csv

docker exec postgres psql -U postgres -d forex -c "\COPY (
    SELECT * FROM backtest_signals WHERE execution_id = ${EXECUTION_ID}
) TO STDOUT WITH CSV HEADER" > /data/sync/signals.csv

docker exec postgres psql -U postgres -d forex -c "\COPY (
    SELECT * FROM backtest_performance WHERE execution_id = ${EXECUTION_ID}
) TO STDOUT WITH CSV HEADER" > /data/sync/performance.csv

# Check if parallel runs table exists and has data
docker exec postgres psql -U postgres -d forex -c "\COPY (
    SELECT * FROM backtest_parallel_runs WHERE ${EXECUTION_ID} = ANY(chunk_execution_ids)
) TO STDOUT WITH CSV HEADER" > /data/sync/parallel_runs.csv 2>/dev/null || touch /data/sync/parallel_runs.csv
REMOTE_SCRIPT
fi

echo -e "${GREEN}  Results exported on Azure${NC}"

# =============================================================================
# 3. Download results
# =============================================================================
echo -e "${BLUE}[3/4] Downloading results...${NC}"

if [[ "${EXECUTION_ID}" == "all" ]]; then
    scp -i "${SSH_KEY}" "azureuser@${AZURE_VM_IP}:/data/sync/backtest_results.sql" "${TEMP_DIR}/"
else
    scp -i "${SSH_KEY}" \
        "azureuser@${AZURE_VM_IP}:/data/sync/executions.csv" \
        "azureuser@${AZURE_VM_IP}:/data/sync/signals.csv" \
        "azureuser@${AZURE_VM_IP}:/data/sync/performance.csv" \
        "azureuser@${AZURE_VM_IP}:/data/sync/parallel_runs.csv" \
        "${TEMP_DIR}/"
fi

echo -e "${GREEN}  Results downloaded${NC}"

# =============================================================================
# 4. Transform and import locally
# =============================================================================
echo -e "${BLUE}[4/4] Importing results locally (with ID offset +${ID_OFFSET})...${NC}"

if [[ "${EXECUTION_ID}" == "all" ]]; then
    # For full dump, transform IDs using sed
    echo "  Transforming IDs in SQL dump..."

    # This is complex for SQL dumps - we need to offset the IDs
    # Using Python for cleaner transformation
    python3 << PYTHON_SCRIPT
import re

with open("${TEMP_DIR}/backtest_results.sql", 'r') as f:
    content = f.read()

# Offset execution IDs
# Match: VALUES (123, ... and change to VALUES (1000123, ...
def offset_id(match):
    old_id = int(match.group(1))
    new_id = old_id + ${ID_OFFSET}
    return f"VALUES ({new_id},"

# For backtest_executions: offset the id column
content = re.sub(r"INSERT INTO.*backtest_executions.*VALUES \((\d+),",
                 lambda m: m.group(0).replace(f"VALUES ({m.group(1)},", f"VALUES ({int(m.group(1)) + ${ID_OFFSET}},"),
                 content)

# For other tables: offset execution_id references
content = re.sub(r"execution_id = (\d+)",
                 lambda m: f"execution_id = {int(m.group(1)) + ${ID_OFFSET}}",
                 content)

with open("${TEMP_DIR}/backtest_results_offset.sql", 'w') as f:
    f.write(content)

print("  IDs transformed successfully")
PYTHON_SCRIPT

    docker exec -i postgres psql -U postgres -d forex < "${TEMP_DIR}/backtest_results_offset.sql"

else
    # For specific execution, use CSV import with ID transformation
    NEW_EXECUTION_ID=$((EXECUTION_ID + ID_OFFSET))
    echo "  Original ID: ${EXECUTION_ID} -> New ID: ${NEW_EXECUTION_ID}"

    # Transform executions CSV
    if [[ -s "${TEMP_DIR}/executions.csv" ]]; then
        echo "  Importing backtest_executions..."
        # Read header and transform data
        python3 << PYTHON_SCRIPT
import csv

with open("${TEMP_DIR}/executions.csv", 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if rows:
    with open("${TEMP_DIR}/executions_offset.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row['id'] = int(row['id']) + ${ID_OFFSET}
            writer.writerow(row)
    print(f"  Transformed {len(rows)} execution(s)")
PYTHON_SCRIPT

        if [[ -f "${TEMP_DIR}/executions_offset.csv" ]]; then
            docker cp "${TEMP_DIR}/executions_offset.csv" postgres:/tmp/executions.csv
            docker exec postgres psql -U postgres -d forex -c "\COPY backtest_executions FROM '/tmp/executions.csv' WITH CSV HEADER"
        fi
    fi

    # Transform signals CSV
    if [[ -s "${TEMP_DIR}/signals.csv" ]]; then
        echo "  Importing backtest_signals..."
        python3 << PYTHON_SCRIPT
import csv

with open("${TEMP_DIR}/signals.csv", 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if rows:
    with open("${TEMP_DIR}/signals_offset.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row['id'] = int(row['id']) + ${ID_OFFSET} * 1000  # Larger offset for signals
            row['execution_id'] = int(row['execution_id']) + ${ID_OFFSET}
            writer.writerow(row)
    print(f"  Transformed {len(rows)} signal(s)")
PYTHON_SCRIPT

        if [[ -f "${TEMP_DIR}/signals_offset.csv" ]]; then
            docker cp "${TEMP_DIR}/signals_offset.csv" postgres:/tmp/signals.csv
            docker exec postgres psql -U postgres -d forex -c "\COPY backtest_signals FROM '/tmp/signals.csv' WITH CSV HEADER"
        fi
    fi

    # Transform performance CSV
    if [[ -s "${TEMP_DIR}/performance.csv" ]]; then
        echo "  Importing backtest_performance..."
        python3 << PYTHON_SCRIPT
import csv

with open("${TEMP_DIR}/performance.csv", 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if rows:
    with open("${TEMP_DIR}/performance_offset.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            row['id'] = int(row['id']) + ${ID_OFFSET}
            row['execution_id'] = int(row['execution_id']) + ${ID_OFFSET}
            writer.writerow(row)
    print(f"  Transformed {len(rows)} performance record(s)")
PYTHON_SCRIPT

        if [[ -f "${TEMP_DIR}/performance_offset.csv" ]]; then
            docker cp "${TEMP_DIR}/performance_offset.csv" postgres:/tmp/performance.csv
            docker exec postgres psql -U postgres -d forex -c "\COPY backtest_performance FROM '/tmp/performance.csv' WITH CSV HEADER"
        fi
    fi
fi

# Cleanup remote sync files
ssh -i "${SSH_KEY}" "azureuser@${AZURE_VM_IP}" "rm -f /data/sync/*.csv /data/sync/*.sql"

echo ""
echo -e "${GREEN}=== Results Pull Complete ===${NC}"
echo ""
if [[ "${EXECUTION_ID}" == "all" ]]; then
    echo "All Azure backtest results imported with ID offset +${ID_OFFSET}"
else
    echo "Execution ${EXECUTION_ID} imported as ID ${NEW_EXECUTION_ID}"
    echo ""
    echo "View results:"
    echo "  docker exec postgres psql -U postgres -d forex -c \"SELECT * FROM backtest_executions WHERE id = ${NEW_EXECUTION_ID};\""
fi
