#!/bin/bash

# Health check script for enhanced backup service
# Checks for recent backup files instead of log files
# Returns 0 for healthy, 1 for warning, 2 for critical

BACKUP_BASE_DIR="/app/postgresbackup"

# Expected databases to backup
EXPECTED_DATABASES=("forex" "forex_config" "stocks" "strategy_config")
EXPECTED_COUNT=${#EXPECTED_DATABASES[@]}

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
FOUND_DATABASES=()

for DATE_FOLDER in "${TODAY}" "${YESTERDAY}"; do
    BACKUP_DIR="${BACKUP_BASE_DIR}/${DATE_FOLDER}"
    if [[ -d "${BACKUP_DIR}" ]]; then
        # Count .sql.gz backup files and track which databases are backed up
        for db in "${EXPECTED_DATABASES[@]}"; do
            if find "${BACKUP_DIR}" -name "${db}_backup_*.sql.gz" -type f -mtime -2 | grep -q .; then
                if [[ ! " ${FOUND_DATABASES[*]} " =~ " ${db} " ]]; then
                    FOUND_DATABASES+=("${db}")
                    RECENT_BACKUPS=$((RECENT_BACKUPS + 1))
                fi
            fi
        done
    fi
done

# Report status based on expected database count (4 databases)
if [[ ${RECENT_BACKUPS} -ge ${EXPECTED_COUNT} ]]; then
    echo "HEALTHY: Found all ${EXPECTED_COUNT} database backups (${FOUND_DATABASES[*]})"
    exit 0
elif [[ ${RECENT_BACKUPS} -ge 2 ]]; then
    MISSING=()
    for db in "${EXPECTED_DATABASES[@]}"; do
        if [[ ! " ${FOUND_DATABASES[*]} " =~ " ${db} " ]]; then
            MISSING+=("${db}")
        fi
    done
    echo "WARNING: Found ${RECENT_BACKUPS}/${EXPECTED_COUNT} databases. Missing: ${MISSING[*]}"
    exit 1
elif [[ ${RECENT_BACKUPS} -gt 0 ]]; then
    echo "WARNING: Only found ${RECENT_BACKUPS}/${EXPECTED_COUNT} database backups (last 48 hours)"
    exit 1
else
    echo "CRITICAL: No recent backup files found (last 48 hours)"
    exit 2
fi