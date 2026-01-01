#!/bin/bash

# Simplified backup script for testing and immediate use
set -euo pipefail

BACKUP_DIR="/app/postgresbackup"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# All databases to backup
DATABASES=("forex" "forex_config" "stocks" "strategy_config")

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🔄 Starting simple database backup..."
echo "📦 Databases: ${DATABASES[*]}"

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Backup all databases
BACKUP_FAILED=0
for db in "${DATABASES[@]}"; do
    echo "📊 Backing up ${db} database..."
    if docker exec "${POSTGRES_CONTAINER}" pg_dump -U "${POSTGRES_USER}" -d "${db}" --verbose --clean --if-exists --create --format=plain --no-owner --no-privileges > "${BACKUP_DIR}/${db}_simple_backup_${TIMESTAMP}.sql" 2>/dev/null; then
        gzip "${BACKUP_DIR}/${db}_simple_backup_${TIMESTAMP}.sql"
        echo -e "${GREEN}✅ ${db} backup completed${NC}"
    else
        echo -e "${RED}❌ ${db} backup failed${NC}"
        BACKUP_FAILED=1
    fi
done

if [[ ${BACKUP_FAILED} -eq 1 ]]; then
    echo -e "${RED}❌ One or more backups failed${NC}"
    exit 1
fi

# Show results
echo "📋 Backup Summary:"
ls -lah "${BACKUP_DIR}"/*simple_backup_${TIMESTAMP}.sql.gz 2>/dev/null | while read -r line; do
    echo "  $line"
done

# Test integrity
echo "🔍 Testing backup integrity..."
for file in "${BACKUP_DIR}"/*simple_backup_${TIMESTAMP}.sql.gz; do
    if [[ -f "$file" ]]; then
        if gzip -t "$file" 2>/dev/null; then
            echo -e "${GREEN}✅ $(basename "$file") - integrity OK${NC}"
        else
            echo -e "${RED}❌ $(basename "$file") - integrity FAILED${NC}"
            exit 1
        fi
    fi
done

echo -e "${GREEN}🎉 All ${#DATABASES[@]} databases backed up successfully!${NC}"