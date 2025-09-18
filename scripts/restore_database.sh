#!/bin/bash

# PostgreSQL Database Restore Script for TradeSystemV1
# Restores forex and/or forex_config databases from backup files

set -euo pipefail

# Configuration
BACKUP_DIR="/app/postgresbackup"
POSTGRES_CONTAINER="postgres"
POSTGRES_USER="postgres"
LOG_FILE="/app/logs/restore.log"

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

# Show usage information
usage() {
    cat << EOF
Database Restore Script for TradeSystemV1

Usage: $0 [OPTIONS] <backup-file>

Options:
    -d, --database NAME     Target database name (forex or forex_config)
    -f, --force            Skip confirmation prompts
    -c, --clean            Drop and recreate database before restore
    -v, --verify           Verify restore after completion
    -b, --backup-before    Create backup before restore
    -h, --help             Show this help message

Examples:
    # Restore specific backup file
    $0 forex_backup_20250918_120000.sql.gz

    # Restore with verification
    $0 --verify forex_backup_20250918_120000.sql.gz

    # Restore latest backup for specific database
    $0 --database forex --latest

    # Force restore with cleanup
    $0 --force --clean forex_backup_20250918_120000.sql.gz

    # List available backups
    $0 --list

Available backup files:
$(find "${BACKUP_DIR}" -name "*.sql.gz" 2>/dev/null | sort -r | head -10)

EOF
}

# Check prerequisites
check_prerequisites() {
    # Check if running with proper access
    if [[ $EUID -eq 0 ]]; then
        log "WARN" "Running as root - this may cause permission issues"
    fi

    # Check if PostgreSQL container is accessible
    if ! docker exec "${POSTGRES_CONTAINER}" pg_isready -U "${POSTGRES_USER}" &>/dev/null; then
        error_exit "PostgreSQL container '${POSTGRES_CONTAINER}' is not accessible or not ready"
    fi

    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "${LOG_FILE}")"

    log "INFO" "Prerequisites check passed"
}

# List available backup files
list_backups() {
    log "INFO" "Available backup files:"

    if [[ ! -d "${BACKUP_DIR}" ]]; then
        log "WARN" "Backup directory does not exist: ${BACKUP_DIR}"
        return 1
    fi

    local -a backup_files
    mapfile -t backup_files < <(find "${BACKUP_DIR}" -name "*.sql.gz" -type f -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-)

    if [[ ${#backup_files[@]} -eq 0 ]]; then
        log "WARN" "No backup files found in ${BACKUP_DIR}"
        return 1
    fi

    for file in "${backup_files[@]}"; do
        local size=$(stat -f%z "${file}" 2>/dev/null || stat -c%s "${file}")
        local date=$(date -r "${file}" '+%Y-%m-%d %H:%M:%S')
        local basename=$(basename "${file}")
        printf "  %-40s %10s %s\n" "${basename}" "$(numfmt --to=iec ${size})" "${date}"
    done
}

# Find latest backup for database
find_latest_backup() {
    local database=$1
    local latest_backup

    latest_backup=$(find "${BACKUP_DIR}" -name "${database}_backup_*.sql.gz" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

    if [[ -n "${latest_backup}" ]]; then
        echo "${latest_backup}"
        return 0
    else
        return 1
    fi
}

# Verify backup file integrity
verify_backup_file() {
    local backup_file=$1

    log "INFO" "Verifying backup file integrity: $(basename "${backup_file}")"

    # Check if file exists
    if [[ ! -f "${backup_file}" ]]; then
        error_exit "Backup file does not exist: ${backup_file}"
    fi

    # Check if file is not empty
    if [[ ! -s "${backup_file}" ]]; then
        error_exit "Backup file is empty: ${backup_file}"
    fi

    # Check gzip integrity
    if ! gzip -t "${backup_file}" &>/dev/null; then
        error_exit "Backup file is corrupted (gzip test failed): ${backup_file}"
    fi

    # Check SQL content
    if ! zcat "${backup_file}" | head -20 | grep -q "PostgreSQL database dump"; then
        error_exit "Backup file does not appear to be a valid PostgreSQL dump: ${backup_file}"
    fi

    log "INFO" "Backup file verification passed"
}

# Extract database name from backup filename
extract_database_name() {
    local backup_file=$1
    local basename=$(basename "${backup_file}")

    # Extract database name from filename pattern: database_backup_timestamp.sql.gz
    if [[ "${basename}" =~ ^(forex|forex_config)_backup_.*\.sql\.gz$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        error_exit "Cannot determine database name from filename: ${basename}"
    fi
}

# Create pre-restore backup
create_pre_restore_backup() {
    local database=$1

    log "INFO" "Creating pre-restore backup of ${database}"

    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_filename="${database}_pre_restore_backup_${timestamp}.sql.gz"
    local backup_path="${BACKUP_DIR}/${backup_filename}"

    if docker exec "${POSTGRES_CONTAINER}" pg_dump \
        -U "${POSTGRES_USER}" \
        -d "${database}" \
        --verbose \
        --clean \
        --if-exists \
        --create \
        --format=plain | gzip > "${backup_path}"; then

        log "INFO" "Pre-restore backup created: ${backup_filename}"
        echo "${backup_path}"
    else
        error_exit "Failed to create pre-restore backup"
    fi
}

# Drop and recreate database
recreate_database() {
    local database=$1

    log "INFO" "Dropping and recreating database: ${database}"

    # Terminate active connections
    docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d postgres -c "
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '${database}' AND pid <> pg_backend_pid();
    " &>/dev/null || true

    # Drop database
    docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${database};" || {
        error_exit "Failed to drop database: ${database}"
    }

    # Create database
    docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d postgres -c "CREATE DATABASE ${database};" || {
        error_exit "Failed to create database: ${database}"
    }

    log "INFO" "Database ${database} recreated successfully"
}

# Restore database from backup
restore_database() {
    local backup_file=$1
    local target_database=$2
    local clean_restore=$3

    log "INFO" "Starting restore of ${target_database} from $(basename "${backup_file}")"

    # If clean restore requested, recreate database
    if [[ "${clean_restore}" == "true" ]]; then
        recreate_database "${target_database}"
    fi

    # Restore database
    local temp_sql="/tmp/restore_${target_database}_$$.sql"

    # Extract SQL from backup
    if ! zcat "${backup_file}" > "${temp_sql}"; then
        rm -f "${temp_sql}"
        error_exit "Failed to extract SQL from backup file"
    fi

    # Execute restore
    if docker exec -i "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d postgres < "${temp_sql}"; then
        log "INFO" "Database restore completed successfully"
    else
        rm -f "${temp_sql}"
        error_exit "Database restore failed"
    fi

    # Cleanup
    rm -f "${temp_sql}"
}

# Verify restored database
verify_restored_database() {
    local database=$1

    log "INFO" "Verifying restored database: ${database}"

    # Check if database exists
    if ! docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -lqt | cut -d \| -f 1 | grep -qw "${database}"; then
        error_exit "Database ${database} does not exist after restore"
    fi

    # Check basic connectivity
    if ! docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${database}" -c "SELECT 1;" &>/dev/null; then
        error_exit "Cannot connect to restored database: ${database}"
    fi

    # Count tables (basic sanity check)
    local table_count
    table_count=$(docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${database}" -t -c "
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog');
    " | tr -d ' ')

    if [[ "${table_count}" -eq 0 ]]; then
        log "WARN" "No user tables found in restored database - this might indicate an issue"
    else
        log "INFO" "Restored database contains ${table_count} user tables"
    fi

    # Database-specific checks
    case "${database}" in
        forex)
            # Check for key forex tables
            local key_tables=("ig_candles" "trade_log")
            for table in "${key_tables[@]}"; do
                if docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${database}" -c "\dt ${table}" &>/dev/null; then
                    local row_count
                    row_count=$(docker exec "${POSTGRES_CONTAINER}" psql -U "${POSTGRES_USER}" -d "${database}" -t -c "SELECT COUNT(*) FROM ${table};" | tr -d ' ')
                    log "INFO" "Table ${table}: ${row_count} rows"
                else
                    log "WARN" "Key table ${table} not found in restored database"
                fi
            done
            ;;
        forex_config)
            # Check for configuration tables
            log "INFO" "Forex config database verification completed"
            ;;
    esac

    log "INFO" "Database verification completed"
}

# Main restore function
main() {
    local backup_file=""
    local target_database=""
    local force_restore=false
    local clean_restore=false
    local verify_after=false
    local backup_before=false
    local use_latest=false
    local list_only=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--database)
                target_database="$2"
                shift 2
                ;;
            -f|--force)
                force_restore=true
                shift
                ;;
            -c|--clean)
                clean_restore=true
                shift
                ;;
            -v|--verify)
                verify_after=true
                shift
                ;;
            -b|--backup-before)
                backup_before=true
                shift
                ;;
            --latest)
                use_latest=true
                shift
                ;;
            --list)
                list_only=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                error_exit "Unknown option: $1"
                ;;
            *)
                backup_file="$1"
                shift
                ;;
        esac
    done

    # Initialize logging
    log "INFO" "Starting database restore process"

    # Check prerequisites
    check_prerequisites

    # Handle list request
    if [[ "${list_only}" == "true" ]]; then
        list_backups
        exit 0
    fi

    # Handle latest backup request
    if [[ "${use_latest}" == "true" ]]; then
        if [[ -z "${target_database}" ]]; then
            error_exit "Database name required when using --latest option"
        fi

        if ! backup_file=$(find_latest_backup "${target_database}"); then
            error_exit "No backup files found for database: ${target_database}"
        fi

        log "INFO" "Using latest backup: $(basename "${backup_file}")"
    fi

    # Validate backup file
    if [[ -z "${backup_file}" ]]; then
        error_exit "Backup file required. Use --help for usage information."
    fi

    # Convert to absolute path if needed
    if [[ "${backup_file}" != /* ]]; then
        backup_file="${BACKUP_DIR}/${backup_file}"
    fi

    # Verify backup file
    verify_backup_file "${backup_file}"

    # Determine target database if not specified
    if [[ -z "${target_database}" ]]; then
        target_database=$(extract_database_name "${backup_file}")
        log "INFO" "Target database determined from filename: ${target_database}"
    fi

    # Validate target database
    if [[ ! "${target_database}" =~ ^(forex|forex_config)$ ]]; then
        error_exit "Invalid target database: ${target_database}. Must be 'forex' or 'forex_config'"
    fi

    # Confirmation prompt (unless forced)
    if [[ "${force_restore}" != "true" ]]; then
        echo
        echo -e "${YELLOW}WARNING: This will restore the ${target_database} database!${NC}"
        echo -e "Source: ${BLUE}$(basename "${backup_file}")${NC}"
        echo -e "Target: ${BLUE}${target_database}${NC}"
        echo -e "Clean restore: ${BLUE}${clean_restore}${NC}"
        echo
        read -p "Are you sure you want to continue? (yes/no): " confirm

        if [[ "${confirm}" != "yes" ]]; then
            log "INFO" "Restore cancelled by user"
            exit 0
        fi
    fi

    # Create pre-restore backup if requested
    if [[ "${backup_before}" == "true" ]]; then
        local pre_restore_backup
        pre_restore_backup=$(create_pre_restore_backup "${target_database}")
        log "INFO" "Pre-restore backup saved: $(basename "${pre_restore_backup}")"
    fi

    # Perform restore
    local start_time=$(date +%s)
    restore_database "${backup_file}" "${target_database}" "${clean_restore}"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Verify restore if requested
    if [[ "${verify_after}" == "true" ]]; then
        verify_restored_database "${target_database}"
    fi

    # Success message
    log "INFO" "Database restore completed successfully in ${duration} seconds"
    echo
    echo -e "${GREEN}✅ Restore completed successfully!${NC}"
    echo -e "Database: ${target_database}"
    echo -e "Source: $(basename "${backup_file}")"
    echo -e "Duration: ${duration} seconds"
    echo
}

# Run main function with all arguments
main "$@"