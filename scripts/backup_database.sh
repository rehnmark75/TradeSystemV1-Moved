#!/bin/bash

# PostgreSQL Database Backup Script for TradeSystemV1
# Backs up both forex and forex_config databases with retention management

set -euo pipefail

# Configuration
BACKUP_DIR="/app/postgresbackup"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
LOG_FILE="/app/logs/backup.log"
MAX_RETRIES=3
RETRY_DELAY=5

# Retention configuration
DAILY_RETENTION=7    # Keep 7 daily backups
WEEKLY_RETENTION=4   # Keep 4 weekly backups (Sundays)
MONTHLY_RETENTION=12 # Keep 12 monthly backups (1st of month)

# Databases to backup (all 4 databases)
DATABASES=("forex" "forex_config" "stocks" "strategy_config")

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Check if running inside Docker
check_environment() {
    if [[ ! -f /.dockerenv ]]; then
        error_exit "This script must be run inside a Docker container with access to PostgreSQL"
    fi

    # Check if PostgreSQL container is accessible
    if ! docker exec "${POSTGRES_CONTAINER}" pg_isready -U "${POSTGRES_USER}" &>/dev/null; then
        error_exit "PostgreSQL container '${POSTGRES_CONTAINER}' is not accessible or not ready"
    fi

    # Create backup directory if it doesn't exist
    mkdir -p "${BACKUP_DIR}"

    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "${LOG_FILE}")"

    log "INFO" "Environment check passed"
}

# Generate backup filename
generate_filename() {
    local database=$1
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    echo "${database}_backup_${timestamp}.sql"
}

# Perform database backup
backup_database() {
    local database=$1
    local filename=$2
    local temp_file="${BACKUP_DIR}/${filename}.tmp"
    local final_file="${BACKUP_DIR}/${filename}"
    local compressed_file="${final_file}.gz"

    log "INFO" "Starting backup of database: ${database}"

    # Retry mechanism for backup
    local retry_count=0
    while [[ ${retry_count} -lt ${MAX_RETRIES} ]]; do
        if docker exec "${POSTGRES_CONTAINER}" pg_dump \
            -U "${POSTGRES_USER}" \
            -d "${database}" \
            --verbose \
            --clean \
            --if-exists \
            --create \
            --format=plain \
            > "${temp_file}" 2>/dev/null; then

            # Move temp file to final location
            mv "${temp_file}" "${final_file}"

            # Compress the backup
            if gzip "${final_file}"; then
                log "INFO" "Successfully backed up ${database} to ${compressed_file}"

                # Verify the backup
                if verify_backup "${compressed_file}"; then
                    log "INFO" "Backup verification passed for ${database}"
                    return 0
                else
                    log "WARN" "Backup verification failed for ${database}, retrying..."
                    rm -f "${compressed_file}"
                fi
            else
                log "WARN" "Failed to compress backup for ${database}, retrying..."
                rm -f "${final_file}"
            fi
        else
            log "WARN" "Backup attempt $((retry_count + 1)) failed for ${database}"
        fi

        retry_count=$((retry_count + 1))
        if [[ ${retry_count} -lt ${MAX_RETRIES} ]]; then
            log "INFO" "Retrying in ${RETRY_DELAY} seconds..."
            sleep ${RETRY_DELAY}
        fi
    done

    # Clean up temp file if it exists
    rm -f "${temp_file}"
    error_exit "Failed to backup ${database} after ${MAX_RETRIES} attempts"
}

# Verify backup integrity
verify_backup() {
    local backup_file=$1

    # Check if file exists and is not empty
    if [[ ! -f "${backup_file}" ]] || [[ ! -s "${backup_file}" ]]; then
        return 1
    fi

    # Check if gzip file is valid
    if ! gzip -t "${backup_file}" &>/dev/null; then
        return 1
    fi

    # Check if SQL content is valid
    if ! zcat "${backup_file}" | head -20 | grep -q "PostgreSQL database dump"; then
        return 1
    fi

    return 0
}

# Calculate backup size and statistics
calculate_backup_stats() {
    local backup_file=$1
    local uncompressed_size
    local compressed_size
    local compression_ratio

    compressed_size=$(stat -f%z "${backup_file}" 2>/dev/null || stat -c%s "${backup_file}")
    uncompressed_size=$(zcat "${backup_file}" | wc -c)
    compression_ratio=$(echo "scale=1; ${compressed_size} * 100 / ${uncompressed_size}" | bc)

    log "INFO" "Backup stats - Compressed: $(numfmt --to=iec ${compressed_size}), Uncompressed: $(numfmt --to=iec ${uncompressed_size}), Ratio: ${compression_ratio}%"
}

# Clean old backups based on retention policy
cleanup_old_backups() {
    local database=$1

    log "INFO" "Starting cleanup for ${database} backups"

    # Get current date components
    local current_day=$(date +%d)
    local current_dow=$(date +%w)  # Day of week (0=Sunday)

    # Arrays to track which backups to keep
    local -a keep_files=()

    # Find all backup files for this database, sorted by modification time (newest first)
    local -a all_files
    mapfile -t all_files < <(find "${BACKUP_DIR}" -name "${database}_backup_*.sql.gz" -type f -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)

    log "INFO" "Found ${#all_files[@]} backup files for ${database}"

    # Keep daily backups (last N days)
    local daily_count=0
    for file in "${all_files[@]}"; do
        if [[ ${daily_count} -lt ${DAILY_RETENTION} ]]; then
            keep_files+=("${file}")
            daily_count=$((daily_count + 1))
        fi
    done

    # Keep weekly backups (Sundays)
    local weekly_count=0
    for file in "${all_files[@]}"; do
        local file_dow=$(date -r "${file}" +%w)
        if [[ ${file_dow} -eq 0 ]] && [[ ${weekly_count} -lt ${WEEKLY_RETENTION} ]]; then
            if [[ ! " ${keep_files[*]} " =~ " ${file} " ]]; then
                keep_files+=("${file}")
                weekly_count=$((weekly_count + 1))
            fi
        fi
    done

    # Keep monthly backups (1st of month)
    local monthly_count=0
    for file in "${all_files[@]}"; do
        local file_day=$(date -r "${file}" +%d)
        if [[ ${file_day} -eq 01 ]] && [[ ${monthly_count} -lt ${MONTHLY_RETENTION} ]]; then
            if [[ ! " ${keep_files[*]} " =~ " ${file} " ]]; then
                keep_files+=("${file}")
                monthly_count=$((monthly_count + 1))
            fi
        fi
    done

    # Remove files not in keep list
    local removed_count=0
    for file in "${all_files[@]}"; do
        if [[ ! " ${keep_files[*]} " =~ " ${file} " ]]; then
            log "INFO" "Removing old backup: $(basename "${file}")"
            rm -f "${file}"
            removed_count=$((removed_count + 1))
        fi
    done

    log "INFO" "Cleanup completed for ${database}: kept ${#keep_files[@]} files, removed ${removed_count} files"
}

# Generate backup report
generate_report() {
    local -a backup_files=("$@")
    local total_size=0

    log "INFO" "=== Backup Report ==="
    log "INFO" "Backup Date: $(date '+%Y-%m-%d %H:%M:%S')"
    log "INFO" "Databases: ${DATABASES[*]}"

    for file in "${backup_files[@]}"; do
        if [[ -f "${file}" ]]; then
            local size=$(stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}")
            total_size=$((total_size + size))
            log "INFO" "- $(basename "${file}"): $(numfmt --to=iec ${size})"
            calculate_backup_stats "${file}"
        fi
    done

    log "INFO" "Total backup size: $(numfmt --to=iec ${total_size})"
    log "INFO" "Backup location: ${BACKUP_DIR}"
    log "INFO" "===================="
}

# Main backup function
main() {
    local start_time=$(date +%s)
    local -a backup_files=()

    log "INFO" "Starting database backup process"

    # Environment checks
    check_environment

    # Backup each database
    for database in "${DATABASES[@]}"; do
        local filename=$(generate_filename "${database}")
        backup_database "${database}" "${filename}"
        backup_files+=("${BACKUP_DIR}/${filename}.gz")

        # Cleanup old backups for this database
        cleanup_old_backups "${database}"
    done

    # Generate report
    generate_report "${backup_files[@]}"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log "INFO" "Backup process completed in ${duration} seconds"
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -d, --dry-run  Show what would be done without executing"
    echo "  -v, --verify   Verify existing backups only"
    echo "  --cleanup      Run cleanup only"
    echo "  --report       Generate report of existing backups"
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    -d|--dry-run)
        log "INFO" "DRY RUN MODE - No actual backups will be performed"
        check_environment
        for database in "${DATABASES[@]}"; do
            filename=$(generate_filename "${database}")
            log "INFO" "Would backup ${database} to ${filename}.gz"
        done
        exit 0
        ;;
    -v|--verify)
        log "INFO" "Verifying existing backups"
        for database in "${DATABASES[@]}"; do
            latest_backup=$(find "${BACKUP_DIR}" -name "${database}_backup_*.sql.gz" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
            if [[ -n "${latest_backup}" ]]; then
                if verify_backup "${latest_backup}"; then
                    log "INFO" "✓ Latest backup for ${database} is valid: $(basename "${latest_backup}")"
                else
                    log "ERROR" "✗ Latest backup for ${database} is corrupted: $(basename "${latest_backup}")"
                fi
            else
                log "WARN" "No backups found for ${database}"
            fi
        done
        exit 0
        ;;
    --cleanup)
        log "INFO" "Running cleanup only"
        check_environment
        for database in "${DATABASES[@]}"; do
            cleanup_old_backups "${database}"
        done
        exit 0
        ;;
    --report)
        log "INFO" "Generating backup report"
        for database in "${DATABASES[@]}"; do
            mapfile -t backup_files < <(find "${BACKUP_DIR}" -name "${database}_backup_*.sql.gz" -type f)
            if [[ ${#backup_files[@]} -gt 0 ]]; then
                log "INFO" "=== ${database} backups ==="
                for file in "${backup_files[@]}"; do
                    size=$(stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}")
                    backup_date=$(date -r "${file}" '+%Y-%m-%d %H:%M:%S')
                    log "INFO" "$(basename "${file}"): $(numfmt --to=iec ${size}) (${backup_date})"
                done
            else
                log "INFO" "No backups found for ${database}"
            fi
        done
        exit 0
        ;;
    "")
        # Run normal backup
        main
        ;;
    *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac