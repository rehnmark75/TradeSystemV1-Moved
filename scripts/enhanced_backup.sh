#!/bin/bash

# Enhanced Backup Script for TradeSystemV1
# Backs up PostgreSQL databases + Vector DB + Logs + PgAdmin config

set -euo pipefail

BACKUP_BASE_DIR="/app/postgresbackup"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
DATE_FOLDER=$(date '+%Y-%m-%d')
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Create daily backup directory
BACKUP_DIR="${BACKUP_BASE_DIR}/${DATE_FOLDER}"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting enhanced TradeSystemV1 backup..."
echo "📁 Backup date folder: ${DATE_FOLDER}"

# Create backup directories
mkdir -p "${BACKUP_BASE_DIR}"
mkdir -p "${BACKUP_DIR}"

# ======================
# 1. PostgreSQL Databases
# ======================
echo -e "${BLUE}📊 Backing up PostgreSQL databases...${NC}"

# All databases to backup (includes tables, views, functions, triggers, sequences, indexes)
DATABASES=("forex" "forex_config" "stocks" "strategy_config")

# pg_dump options explanation:
# --verbose: Show progress
# --clean: Include DROP statements before CREATE
# --if-exists: Add IF EXISTS to DROP statements
# --create: Include CREATE DATABASE statement
# --format=plain: SQL script format (readable, portable)
# --no-owner: Don't include ownership commands (for portability)
# --no-privileges: Don't include GRANT/REVOKE (for portability)
# Note: pg_dump automatically includes ALL schema objects:
#   - Tables with data
#   - Views (including materialized views)
#   - Functions and procedures
#   - Triggers
#   - Sequences
#   - Indexes
#   - Constraints (foreign keys, unique, check)
#   - Custom types and domains
#   - Extensions

backup_single_database() {
    local db_name=$1
    echo "  → Backing up ${db_name} database..."

    # Per-database data-exclusion list.
    # We keep the SCHEMA (so a restore recreates the table with its indexes)
    # but skip the DATA. Currently:
    #   forex.ig_candles_backtest — derived/resampled historical candles,
    #       re-populatable via scripts/dukascopy_download.py + dukascopy_push_local.sh.
    #       Excluding saves ~8-10 GB per backup generation.
    local exclude_args=()
    case "${db_name}" in
        forex)
            exclude_args+=(--exclude-table-data='public.ig_candles_backtest')
            ;;
    esac

    # pg_dump with comprehensive options for complete backup:
    # - Tables, data, indexes, constraints
    # - Views (regular and materialized)
    # - Functions, procedures, triggers
    # - Sequences, custom types, domains
    # - Extensions (uuid-ossp, etc.)
    if docker exec "${POSTGRES_CONTAINER}" pg_dump \
        -U "${POSTGRES_USER}" \
        -d "${db_name}" \
        --verbose \
        --clean \
        --if-exists \
        --create \
        --format=plain \
        --no-owner \
        --no-privileges \
        "${exclude_args[@]}" \
        > "${BACKUP_DIR}/${db_name}_backup_${TIMESTAMP}.sql" 2>/dev/null; then

        gzip "${BACKUP_DIR}/${db_name}_backup_${TIMESTAMP}.sql"

        # Show what was backed up
        local size=$(stat -f%z "${BACKUP_DIR}/${db_name}_backup_${TIMESTAMP}.sql.gz" 2>/dev/null || stat -c%s "${BACKUP_DIR}/${db_name}_backup_${TIMESTAMP}.sql.gz")
        echo -e "${GREEN}  ✅ ${db_name} database backed up ($(numfmt --to=iec ${size}))${NC}"
        return 0
    else
        echo -e "${RED}  ❌ ${db_name} database backup failed${NC}"
        return 1
    fi
}

# Backup all databases
BACKUP_FAILED=0
for db in "${DATABASES[@]}"; do
    if ! backup_single_database "${db}"; then
        BACKUP_FAILED=1
    fi
done

if [[ ${BACKUP_FAILED} -eq 1 ]]; then
    echo -e "${RED}❌ One or more database backups failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All ${#DATABASES[@]} databases backed up successfully${NC}"

# ======================
# 2. Vector Database
# ======================
echo -e "${BLUE}🤖 Backing up Vector Database (ChromaDB)...${NC}"

# Create temp directory for additional backups
TEMP_DIR="${BACKUP_DIR}/temp_${TIMESTAMP}"
mkdir -p "${TEMP_DIR}"

if [[ -d "/app/vector-db/vectordb" ]]; then
    echo "  → Copying vector database files..."
    cp -r /app/vector-db/vectordb "${TEMP_DIR}/"
    echo -e "${GREEN}  ✅ Vector database backed up${NC}"
else
    echo -e "${YELLOW}  ⚠️  Vector database directory not found, skipping${NC}"
fi

if [[ -d "/app/vector-db/data" ]]; then
    echo "  → Copying vector database data..."
    cp -r /app/vector-db/data "${TEMP_DIR}/"
    echo -e "${GREEN}  ✅ Vector database data backed up${NC}"
else
    echo -e "${YELLOW}  ⚠️  Vector database data directory not found, skipping${NC}"
fi

# ======================
# 3. Application Logs
# ======================
echo -e "${BLUE}📝 Backing up Application Logs...${NC}"

# List of log directories to backup
LOG_DIRS=(
    "/app/logs/worker"
    "/app/logs/vector-db"
    "/app/logs/tradingview"
    "/app/logs/economic-calendar"
    "/app/logs/dev"
    "/app/logs/prod"
    "/app/logs/stream"
    "/app/logs/backup"
)

mkdir -p "${TEMP_DIR}/logs"

for log_dir in "${LOG_DIRS[@]}"; do
    if [[ -d "$log_dir" ]]; then
        dir_name=$(basename "$log_dir")
        echo "  → Backing up $dir_name logs..."
        cp -r "$log_dir" "${TEMP_DIR}/logs/" 2>/dev/null || echo -e "${YELLOW}    ⚠️  Some $dir_name logs inaccessible${NC}"
        echo -e "${GREEN}    ✅ $dir_name logs backed up${NC}"
    else
        echo -e "${YELLOW}  ⚠️  Log directory $log_dir not found, skipping${NC}"
    fi
done

# ======================
# 4. PgAdmin Configuration
# ======================
echo -e "${BLUE}⚙️  Backing up PgAdmin Configuration...${NC}"

# Create PgAdmin backup using docker cp
echo "  → Extracting PgAdmin configuration..."
if docker exec pgadmin tar -czf /tmp/pgadmin_config_${TIMESTAMP}.tar.gz -C /var/lib/pgadmin . 2>/dev/null; then
    docker cp pgadmin:/tmp/pgadmin_config_${TIMESTAMP}.tar.gz "${TEMP_DIR}/"
    docker exec pgadmin rm -f /tmp/pgadmin_config_${TIMESTAMP}.tar.gz
    echo -e "${GREEN}  ✅ PgAdmin configuration backed up${NC}"
else
    echo -e "${YELLOW}  ⚠️  PgAdmin configuration backup failed, continuing...${NC}"
fi

# ======================
# 5. Compress Additional Backups
# ======================
echo -e "${BLUE}📦 Compressing additional backup components...${NC}"

if [[ -d "${TEMP_DIR}" ]] && [[ "$(ls -A ${TEMP_DIR})" ]]; then
    echo "  → Creating compressed archive..."
    tar -czf "${BACKUP_DIR}/additional_backup_${TIMESTAMP}.tar.gz" -C "${TEMP_DIR}" .

    # Verify the archive
    if tar -tzf "${BACKUP_DIR}/additional_backup_${TIMESTAMP}.tar.gz" >/dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Additional backup archive created and verified${NC}"
        # Clean up temp directory
        rm -rf "${TEMP_DIR}"
    else
        echo -e "${RED}  ❌ Additional backup archive verification failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}  ⚠️  No additional components to compress${NC}"
    rm -rf "${TEMP_DIR}"
fi

# ======================
# 6. Backup Summary
# ======================
echo
echo -e "${GREEN}🎉 Enhanced backup completed successfully!${NC}"
echo -e "${BLUE}📋 Backup Summary:${NC}"
echo "================================"

# Show all backup files created
for file in "${BACKUP_DIR}"/*_${TIMESTAMP}*; do
    if [[ -f "$file" ]]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
        echo "$(basename "$file"): $(numfmt --to=iec ${size})"
    fi
done

echo
echo -e "${BLUE}📍 Backup Location:${NC} ${BACKUP_DIR}"
echo -e "${BLUE}🕐 Timestamp:${NC} ${TIMESTAMP}"

# ======================
# 7. Integrity Verification
# ======================
echo
echo -e "${BLUE}🔍 Verifying backup integrity...${NC}"

# Verify all PostgreSQL database backups
for db in "${DATABASES[@]}"; do
    db_backup="${BACKUP_DIR}/${db}_backup_${TIMESTAMP}.sql.gz"
    if [[ -f "$db_backup" ]]; then
        if gzip -t "$db_backup" 2>/dev/null; then
            # Verify SQL content contains pg_dump header
            # Use a subshell to avoid pipefail issues with head closing pipe early
            if (zcat "$db_backup" 2>/dev/null || true) | head -50 | grep -q "PostgreSQL database dump"; then
                echo -e "${GREEN}  ✅ $(basename "$db_backup") - integrity OK${NC}"
            else
                echo -e "${RED}  ❌ $(basename "$db_backup") - SQL content invalid${NC}"
                exit 1
            fi
        else
            echo -e "${RED}  ❌ $(basename "$db_backup") - gzip integrity FAILED${NC}"
            exit 1
        fi
    else
        echo -e "${RED}  ❌ ${db}_backup_${TIMESTAMP}.sql.gz - file missing${NC}"
        exit 1
    fi
done

# Verify additional backup archive
if [[ -f "${BACKUP_DIR}/additional_backup_${TIMESTAMP}.tar.gz" ]]; then
    if tar -tzf "${BACKUP_DIR}/additional_backup_${TIMESTAMP}.tar.gz" >/dev/null 2>&1; then
        echo -e "${GREEN}  ✅ additional_backup_${TIMESTAMP}.tar.gz - integrity OK${NC}"
    else
        echo -e "${RED}  ❌ additional_backup_${TIMESTAMP}.tar.gz - integrity FAILED${NC}"
        exit 1
    fi
fi

# ======================
# 8. Upload to Azure Blob Storage
# ======================
upload_to_azure() {
    # Check if Azure credentials are configured
    if [[ -z "${AZURE_STORAGE_ACCOUNT:-}" ]] || [[ -z "${AZURE_STORAGE_KEY:-}" ]]; then
        echo -e "${YELLOW}  ⚠️  Azure credentials not configured, skipping cloud upload${NC}"
        return 0
    fi

    echo -e "${BLUE}☁️  Uploading backups to Azure Blob Storage...${NC}"
    echo "  → Storage Account: ${AZURE_STORAGE_ACCOUNT}"
    echo "  → Container: ${AZURE_STORAGE_CONTAINER:-klirrbackup}"

    local container="${AZURE_STORAGE_CONTAINER:-klirrbackup}"
    local upload_count=0
    local upload_failed=0

    # Upload all backup files from today's folder
    for file in "${BACKUP_DIR}"/*_${TIMESTAMP}*; do
        if [[ -f "$file" ]]; then
            local filename=$(basename "$file")
            local blob_path="${DATE_FOLDER}/${filename}"
            local file_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")

            echo "  → Uploading ${filename} ($(numfmt --to=iec ${file_size}))..."
            if az storage blob upload \
                --account-name "${AZURE_STORAGE_ACCOUNT}" \
                --account-key "${AZURE_STORAGE_KEY}" \
                --container-name "${container}" \
                --name "${blob_path}" \
                --file "$file" \
                --overwrite true \
                --only-show-errors 2>/dev/null; then
                echo -e "${GREEN}    ✅ Uploaded ${filename}${NC}"
                upload_count=$((upload_count + 1))
            else
                echo -e "${RED}    ❌ Failed to upload ${filename}${NC}"
                upload_failed=$((upload_failed + 1))
            fi
        fi
    done

    if [[ ${upload_failed} -gt 0 ]]; then
        echo -e "${YELLOW}  ⚠️  ${upload_count} files uploaded, ${upload_failed} failed${NC}"
    else
        echo -e "${GREEN}  ✅ All ${upload_count} backup files uploaded to Azure${NC}"
    fi
}

echo
upload_to_azure

# ======================
# 9. Azure Retention Cleanup (60 days)
# ======================
cleanup_azure_old_backups() {
    # Check if Azure credentials are configured
    if [[ -z "${AZURE_STORAGE_ACCOUNT:-}" ]] || [[ -z "${AZURE_STORAGE_KEY:-}" ]]; then
        return 0
    fi

    local retention_days="${AZURE_RETENTION_DAYS:-60}"
    local cutoff_date=$(date -d "${retention_days} days ago" '+%Y-%m-%d' 2>/dev/null)
    local container="${AZURE_STORAGE_CONTAINER:-klirrbackup}"

    echo
    echo -e "${BLUE}🧹 Cleaning up Azure backups older than ${retention_days} days (before ${cutoff_date})...${NC}"

    local deleted_count=0

    # Get list of all blobs and filter by date
    az storage blob list \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --account-key "${AZURE_STORAGE_KEY}" \
        --container-name "${container}" \
        --query "[].name" -o tsv 2>/dev/null | \
    while read -r blob_name; do
        # Extract date from blob path (format: YYYY-MM-DD/filename)
        local blob_date=$(echo "${blob_name}" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}' || echo "")

        if [[ -n "${blob_date}" ]] && [[ "${blob_date}" < "${cutoff_date}" ]]; then
            echo "  → Deleting old backup: ${blob_name}"
            if az storage blob delete \
                --account-name "${AZURE_STORAGE_ACCOUNT}" \
                --account-key "${AZURE_STORAGE_KEY}" \
                --container-name "${container}" \
                --name "${blob_name}" \
                --only-show-errors 2>/dev/null; then
                deleted_count=$((deleted_count + 1))
            fi
        fi
    done

    echo -e "${GREEN}  ✅ Azure cleanup completed${NC}"
}

cleanup_azure_old_backups

# ======================
# 10. Cleanup Old Local Backups (Retention Policy)
# ======================
echo
echo -e "${BLUE}🧹 Applying local retention policy...${NC}"

# Retention settings
DAILY_RETENTION=7    # Keep 7 daily backups
WEEKLY_RETENTION=4   # Keep 4 weekly backups (Sundays)
MONTHLY_RETENTION=12 # Keep 12 monthly backups (1st of month)

cleanup_old_backups() {
    echo "  → Scanning backup folders..."

    # Get all backup date folders sorted by date (newest first)
    local -a all_folders
    mapfile -t all_folders < <(find "${BACKUP_BASE_DIR}" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]" | sort -r)

    local total_folders=${#all_folders[@]}
    echo "  → Found ${total_folders} backup folders"

    if [[ ${total_folders} -le ${DAILY_RETENTION} ]]; then
        echo -e "${GREEN}  ✅ All folders within retention policy (${total_folders}/${DAILY_RETENTION})${NC}"
        return 0
    fi

    # Arrays to track which folders to keep
    local -a keep_folders=()

    # Keep daily backups (last N days)
    local daily_count=0
    for folder in "${all_folders[@]}"; do
        if [[ ${daily_count} -lt ${DAILY_RETENTION} ]]; then
            keep_folders+=("${folder}")
            daily_count=$((daily_count + 1))
        fi
    done

    # Keep weekly backups (Sundays) - check folders older than daily retention
    local weekly_count=0
    for folder in "${all_folders[@]:${DAILY_RETENTION}}"; do
        # Extract date from folder name (YYYY-MM-DD)
        local folder_date=$(basename "${folder}")
        # Check if this date was a Sunday (day of week = 0)
        local day_of_week=$(date -d "${folder_date}" '+%w' 2>/dev/null || echo "1")

        if [[ ${day_of_week} -eq 0 ]] && [[ ${weekly_count} -lt ${WEEKLY_RETENTION} ]]; then
            if [[ ! " ${keep_folders[*]} " =~ " ${folder} " ]]; then
                keep_folders+=("${folder}")
                weekly_count=$((weekly_count + 1))
            fi
        fi
    done

    # Keep monthly backups (1st of month) - check all folders
    local monthly_count=0
    for folder in "${all_folders[@]}"; do
        # Extract date from folder name (YYYY-MM-DD)
        local folder_date=$(basename "${folder}")
        # Check if this is the 1st of the month
        local day_of_month=$(date -d "${folder_date}" '+%d' 2>/dev/null || echo "02")

        if [[ ${day_of_month} -eq 01 ]] && [[ ${monthly_count} -lt ${MONTHLY_RETENTION} ]]; then
            if [[ ! " ${keep_folders[*]} " =~ " ${folder} " ]]; then
                keep_folders+=("${folder}")
                monthly_count=$((monthly_count + 1))
            fi
        fi
    done

    # Remove folders not in keep list
    local removed_count=0
    for folder in "${all_folders[@]}"; do
        if [[ ! " ${keep_folders[*]} " =~ " ${folder} " ]]; then
            echo "  → Removing old backup folder: $(basename "${folder}")"
            rm -rf "${folder}"
            removed_count=$((removed_count + 1))
        fi
    done

    echo -e "${GREEN}  ✅ Retention cleanup completed: kept ${#keep_folders[@]} folders, removed ${removed_count} folders${NC}"
}

# Run cleanup
cleanup_old_backups

echo
echo -e "${GREEN}✨ All backup integrity checks passed!${NC}"
echo -e "${GREEN}🎯 Enhanced backup system ready for production use.${NC}"