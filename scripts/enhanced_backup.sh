#!/bin/bash

# Enhanced Backup Script for TradeSystemV1
# Backs up PostgreSQL databases + Vector DB + Logs + PgAdmin config

set -euo pipefail

BACKUP_DIR="/app/postgresbackup"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting enhanced TradeSystemV1 backup..."

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# ======================
# 1. PostgreSQL Databases
# ======================
echo -e "${BLUE}📊 Backing up PostgreSQL databases...${NC}"

# Backup forex database
echo "  → Backing up forex database..."
if docker exec "${POSTGRES_CONTAINER}" pg_dump -U "${POSTGRES_USER}" -d forex --verbose --clean --if-exists --create --format=plain > "${BACKUP_DIR}/forex_backup_${TIMESTAMP}.sql" 2>/dev/null; then
    gzip "${BACKUP_DIR}/forex_backup_${TIMESTAMP}.sql"
    echo -e "${GREEN}  ✅ Forex database backed up${NC}"
else
    echo -e "${RED}  ❌ Forex database backup failed${NC}"
    exit 1
fi

# Backup forex_config database
echo "  → Backing up forex_config database..."
if docker exec "${POSTGRES_CONTAINER}" pg_dump -U "${POSTGRES_USER}" -d forex_config --verbose --clean --if-exists --create --format=plain > "${BACKUP_DIR}/forex_config_backup_${TIMESTAMP}.sql" 2>/dev/null; then
    gzip "${BACKUP_DIR}/forex_config_backup_${TIMESTAMP}.sql"
    echo -e "${GREEN}  ✅ Forex_config database backed up${NC}"
else
    echo -e "${RED}  ❌ Forex_config database backup failed${NC}"
    exit 1
fi

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

# Verify PostgreSQL backups
for db_backup in "${BACKUP_DIR}"/forex*_${TIMESTAMP}.sql.gz; do
    if [[ -f "$db_backup" ]]; then
        if gzip -t "$db_backup" 2>/dev/null; then
            echo -e "${GREEN}  ✅ $(basename "$db_backup") - integrity OK${NC}"
        else
            echo -e "${RED}  ❌ $(basename "$db_backup") - integrity FAILED${NC}"
            exit 1
        fi
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

echo
echo -e "${GREEN}✨ All backup integrity checks passed!${NC}"
echo -e "${GREEN}🎯 Enhanced backup system ready for production use.${NC}"