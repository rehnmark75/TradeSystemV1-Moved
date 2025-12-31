#!/bin/bash

# Health check script for enhanced backup service
# Checks for recent backup files instead of log files
# Returns 0 for healthy, 1 for warning, 2 for critical

BACKUP_BASE_DIR="/app/postgresbackup"

# Check if backup directory exists
if [[ ! -d "${BACKUP_BASE_DIR}" ]]; then
    echo "CRITICAL: Backup directory not found"
    exit 2
fi

# Get today's and yesterday's date folders
TODAY=$(date '+%Y-%m-%d')
YESTERDAY=$(date -d '1 day ago' '+%Y-%m-%d' 2>/dev/null || date -v-1d '+%Y-%m-%d')

# Check for backups in today's or yesterday's folder
RECENT_BACKUPS=0

for DATE_FOLDER in "${TODAY}" "${YESTERDAY}"; do
    BACKUP_DIR="${BACKUP_BASE_DIR}/${DATE_FOLDER}"
    if [[ -d "${BACKUP_DIR}" ]]; then
        # Count .sql.gz backup files
        BACKUP_COUNT=$(find "${BACKUP_DIR}" -name "*.sql.gz" -type f -mtime -2 | wc -l)
        RECENT_BACKUPS=$((RECENT_BACKUPS + BACKUP_COUNT))
    fi
done

# Expect at least 2 database backups (forex + forex_config)
if [[ ${RECENT_BACKUPS} -ge 2 ]]; then
    echo "HEALTHY: Found ${RECENT_BACKUPS} recent backup files (last 48 hours)"
    exit 0
elif [[ ${RECENT_BACKUPS} -gt 0 ]]; then
    echo "WARNING: Only found ${RECENT_BACKUPS} recent backup files (expected 2+)"
    exit 1
else
    echo "CRITICAL: No recent backup files found (last 48 hours)"
    exit 2
fi