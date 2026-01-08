#!/bin/bash

# Azure-Only Backup Script for TradeSystemV1
# Streams backups directly to Azure Blob Storage using /tmp for staging
# No dependency on external drives

set -euo pipefail

# Use /tmp for temporary staging (container's ephemeral storage)
TEMP_BASE="/tmp/backup_staging"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
DATE_FOLDER=$(date '+%Y-%m-%d')
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Create temp staging directory
STAGING_DIR="${TEMP_BASE}/${TIMESTAMP}"
mkdir -p "${STAGING_DIR}"

# Cleanup function to ensure temp files are removed
cleanup() {
    echo "Cleaning up staging directory..."
    rm -rf "${STAGING_DIR}"
    rm -rf "${TEMP_BASE}"
}
trap cleanup EXIT

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting Azure-direct backup for TradeSystemV1..."
echo "📁 Date folder: ${DATE_FOLDER}"
echo "📦 Staging directory: ${STAGING_DIR}"

# Check Azure credentials
if [[ -z "${AZURE_STORAGE_ACCOUNT:-}" ]] || [[ -z "${AZURE_STORAGE_KEY:-}" ]]; then
    echo -e "${RED}❌ Azure credentials not configured!${NC}"
    echo "Required environment variables: AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY"
    exit 1
fi

CONTAINER="${AZURE_STORAGE_CONTAINER:-klirrbackup}"
echo "☁️  Target: ${AZURE_STORAGE_ACCOUNT}/${CONTAINER}/${DATE_FOLDER}/"

# ======================
# 1. PostgreSQL Databases
# ======================
echo -e "${BLUE}📊 Backing up PostgreSQL databases...${NC}"

DATABASES=("forex" "forex_config" "stocks" "strategy_config")

backup_and_upload_database() {
    local db_name=$1
    local sql_file="${STAGING_DIR}/${db_name}_backup_${TIMESTAMP}.sql"
    local gz_file="${sql_file}.gz"
    local blob_path="${DATE_FOLDER}/${db_name}_backup_${TIMESTAMP}.sql.gz"

    echo "  → Backing up ${db_name}..."

    # Dump database to staging
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
        > "${sql_file}" 2>/dev/null; then

        # Compress
        gzip "${sql_file}"

        local size=$(stat -c%s "${gz_file}" 2>/dev/null || stat -f%z "${gz_file}")
        echo "  → Uploading ${db_name} ($(numfmt --to=iec ${size}))..."

        # Upload to Azure
        if az storage blob upload \
            --account-name "${AZURE_STORAGE_ACCOUNT}" \
            --account-key "${AZURE_STORAGE_KEY}" \
            --container-name "${CONTAINER}" \
            --name "${blob_path}" \
            --file "${gz_file}" \
            --overwrite true \
            --only-show-errors 2>/dev/null; then
            echo -e "${GREEN}  ✅ ${db_name} uploaded to Azure${NC}"
            # Remove local file immediately to save space
            rm -f "${gz_file}"
            return 0
        else
            echo -e "${RED}  ❌ ${db_name} upload failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}  ❌ ${db_name} dump failed${NC}"
        return 1
    fi
}

# Backup all databases
BACKUP_FAILED=0
for db in "${DATABASES[@]}"; do
    if ! backup_and_upload_database "${db}"; then
        BACKUP_FAILED=1
    fi
done

if [[ ${BACKUP_FAILED} -eq 1 ]]; then
    echo -e "${RED}❌ One or more database backups failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All ${#DATABASES[@]} databases backed up to Azure${NC}"

# ======================
# 2. Vector Database
# ======================
echo -e "${BLUE}🤖 Backing up Vector Database (ChromaDB)...${NC}"

ADDITIONAL_DIR="${STAGING_DIR}/additional"
mkdir -p "${ADDITIONAL_DIR}"

if [[ -d "/app/vector-db/vectordb" ]]; then
    echo "  → Copying vector database files..."
    cp -r /app/vector-db/vectordb "${ADDITIONAL_DIR}/"
    echo -e "${GREEN}  ✅ Vector database staged${NC}"
else
    echo -e "${YELLOW}  ⚠️  Vector database directory not found, skipping${NC}"
fi

if [[ -d "/app/vector-db/data" ]]; then
    echo "  → Copying vector database data..."
    cp -r /app/vector-db/data "${ADDITIONAL_DIR}/"
    echo -e "${GREEN}  ✅ Vector database data staged${NC}"
else
    echo -e "${YELLOW}  ⚠️  Vector database data directory not found, skipping${NC}"
fi

# ======================
# 3. Application Logs
# ======================
echo -e "${BLUE}📝 Backing up Application Logs...${NC}"

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

mkdir -p "${ADDITIONAL_DIR}/logs"

for log_dir in "${LOG_DIRS[@]}"; do
    if [[ -d "$log_dir" ]]; then
        dir_name=$(basename "$log_dir")
        cp -r "$log_dir" "${ADDITIONAL_DIR}/logs/" 2>/dev/null || true
    fi
done
echo -e "${GREEN}  ✅ Logs staged${NC}"

# ======================
# 4. PgAdmin Configuration
# ======================
echo -e "${BLUE}⚙️  Backing up PgAdmin Configuration...${NC}"

if docker exec pgadmin tar -czf /tmp/pgadmin_config_${TIMESTAMP}.tar.gz -C /var/lib/pgadmin . 2>/dev/null; then
    docker cp pgadmin:/tmp/pgadmin_config_${TIMESTAMP}.tar.gz "${ADDITIONAL_DIR}/"
    docker exec pgadmin rm -f /tmp/pgadmin_config_${TIMESTAMP}.tar.gz
    echo -e "${GREEN}  ✅ PgAdmin configuration staged${NC}"
else
    echo -e "${YELLOW}  ⚠️  PgAdmin configuration backup failed, continuing...${NC}"
fi

# ======================
# 5. Compress & Upload Additional Backups
# ======================
echo -e "${BLUE}📦 Compressing and uploading additional components...${NC}"

ADDITIONAL_ARCHIVE="${STAGING_DIR}/additional_backup_${TIMESTAMP}.tar.gz"

if [[ -d "${ADDITIONAL_DIR}" ]] && [[ "$(ls -A ${ADDITIONAL_DIR})" ]]; then
    echo "  → Creating compressed archive..."
    tar -czf "${ADDITIONAL_ARCHIVE}" -C "${ADDITIONAL_DIR}" .

    # Verify archive
    if tar -tzf "${ADDITIONAL_ARCHIVE}" >/dev/null 2>&1; then
        SIZE=$(stat -c%s "${ADDITIONAL_ARCHIVE}" 2>/dev/null || stat -f%z "${ADDITIONAL_ARCHIVE}")
        echo "  → Uploading additional backup ($(numfmt --to=iec ${SIZE}))..."

        if az storage blob upload \
            --account-name "${AZURE_STORAGE_ACCOUNT}" \
            --account-key "${AZURE_STORAGE_KEY}" \
            --container-name "${CONTAINER}" \
            --name "${DATE_FOLDER}/additional_backup_${TIMESTAMP}.tar.gz" \
            --file "${ADDITIONAL_ARCHIVE}" \
            --overwrite true \
            --only-show-errors 2>/dev/null; then
            echo -e "${GREEN}  ✅ Additional backup uploaded to Azure${NC}"
        else
            echo -e "${RED}  ❌ Additional backup upload failed${NC}"
        fi
    else
        echo -e "${RED}  ❌ Additional backup archive verification failed${NC}"
    fi

    # Clean up staging
    rm -rf "${ADDITIONAL_DIR}"
    rm -f "${ADDITIONAL_ARCHIVE}"
else
    echo -e "${YELLOW}  ⚠️  No additional components to compress${NC}"
fi

# ======================
# 6. Azure Retention Cleanup
# ======================
cleanup_azure_old_backups() {
    local retention_days="${AZURE_RETENTION_DAYS:-60}"
    local cutoff_date=$(date -d "${retention_days} days ago" '+%Y-%m-%d' 2>/dev/null)

    echo
    echo -e "${BLUE}🧹 Cleaning up Azure backups older than ${retention_days} days (before ${cutoff_date})...${NC}"

    # Get list of all blobs and filter by date
    az storage blob list \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --account-key "${AZURE_STORAGE_KEY}" \
        --container-name "${CONTAINER}" \
        --query "[].name" -o tsv 2>/dev/null | \
    while read -r blob_name; do
        # Extract date from blob path (format: YYYY-MM-DD/filename)
        local blob_date=$(echo "${blob_name}" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}' || echo "")

        if [[ -n "${blob_date}" ]] && [[ "${blob_date}" < "${cutoff_date}" ]]; then
            echo "  → Deleting old backup: ${blob_name}"
            az storage blob delete \
                --account-name "${AZURE_STORAGE_ACCOUNT}" \
                --account-key "${AZURE_STORAGE_KEY}" \
                --container-name "${CONTAINER}" \
                --name "${blob_name}" \
                --only-show-errors 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}  ✅ Azure cleanup completed${NC}"
}

cleanup_azure_old_backups

# ======================
# 7. Backup Summary
# ======================
echo
echo -e "${GREEN}🎉 Azure backup completed successfully!${NC}"
echo -e "${BLUE}📋 Backup Summary:${NC}"
echo "================================"
echo "Storage Account: ${AZURE_STORAGE_ACCOUNT}"
echo "Container: ${CONTAINER}"
echo "Blob Path: ${DATE_FOLDER}/"
echo "Databases: ${DATABASES[*]}"
echo "Timestamp: ${TIMESTAMP}"

# List uploaded blobs for today
echo
echo -e "${BLUE}📁 Uploaded files:${NC}"
az storage blob list \
    --account-name "${AZURE_STORAGE_ACCOUNT}" \
    --account-key "${AZURE_STORAGE_KEY}" \
    --container-name "${CONTAINER}" \
    --prefix "${DATE_FOLDER}/" \
    --query "[].{name:name, size:properties.contentLength}" \
    --output table 2>/dev/null || echo "  (Could not list blobs)"

echo
echo -e "${GREEN}✨ All backups stored in Azure Blob Storage!${NC}"
