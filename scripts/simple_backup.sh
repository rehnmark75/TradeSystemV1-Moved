#!/bin/bash

# Simplified backup script for testing and immediate use
set -euo pipefail

BACKUP_DIR="/app/postgresbackup"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🔄 Starting simple database backup..."

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Backup forex database
echo "📊 Backing up forex database..."
if docker exec "${POSTGRES_CONTAINER}" pg_dump -U "${POSTGRES_USER}" -d forex --verbose --clean --if-exists --create --format=plain > "${BACKUP_DIR}/forex_simple_backup_${TIMESTAMP}.sql" 2>/dev/null; then
    # Compress backup
    gzip "${BACKUP_DIR}/forex_simple_backup_${TIMESTAMP}.sql"
    echo -e "${GREEN}✅ Forex backup completed: forex_simple_backup_${TIMESTAMP}.sql.gz${NC}"
else
    echo -e "${RED}❌ Forex backup failed${NC}"
    exit 1
fi

# Backup forex_config database
echo "⚙️  Backing up forex_config database..."
if docker exec "${POSTGRES_CONTAINER}" pg_dump -U "${POSTGRES_USER}" -d forex_config --verbose --clean --if-exists --create --format=plain > "${BACKUP_DIR}/forex_config_simple_backup_${TIMESTAMP}.sql" 2>/dev/null; then
    # Compress backup
    gzip "${BACKUP_DIR}/forex_config_simple_backup_${TIMESTAMP}.sql"
    echo -e "${GREEN}✅ Forex_config backup completed: forex_config_simple_backup_${TIMESTAMP}.sql.gz${NC}"
else
    echo -e "${RED}❌ Forex_config backup failed${NC}"
    exit 1
fi

# Show results
echo "📋 Backup Summary:"
ls -lah "${BACKUP_DIR}"/*simple_backup_${TIMESTAMP}.sql.gz | while read -r line; do
    echo "  $line"
done

# Test integrity
echo "🔍 Testing backup integrity..."
for file in "${BACKUP_DIR}"/*simple_backup_${TIMESTAMP}.sql.gz; do
    if gzip -t "$file" 2>/dev/null; then
        echo -e "${GREEN}✅ $(basename "$file") - integrity OK${NC}"
    else
        echo -e "${RED}❌ $(basename "$file") - integrity FAILED${NC}"
        exit 1
    fi
done

echo -e "${GREEN}🎉 All backups completed successfully!${NC}"