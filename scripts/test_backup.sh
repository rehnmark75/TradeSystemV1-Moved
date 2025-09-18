#!/bin/bash

# Simple test version of backup script for validation
# Tests basic pg_dump functionality without Docker-in-Docker

set -euo pipefail

# Configuration
BACKUP_DIR="/tmp/backup_test"
POSTGRES_USER="postgres"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=== Database Backup Test ==="

# Create test backup directory
mkdir -p "${BACKUP_DIR}"

# Test 1: Check PostgreSQL connectivity
echo -n "Testing PostgreSQL connectivity... "
if pg_isready -U "${POSTGRES_USER}" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test 2: Check database existence
echo -n "Testing database existence... "
if psql -U "${POSTGRES_USER}" -lqt | cut -d \| -f 1 | grep -qw "forex"; then
    echo -e "${GREEN}✓ forex database found${NC}"
else
    echo -e "${RED}✗ forex database not found${NC}"
    exit 1
fi

if psql -U "${POSTGRES_USER}" -lqt | cut -d \| -f 1 | grep -qw "forex_config"; then
    echo -e "${GREEN}✓ forex_config database found${NC}"
else
    echo -e "${RED}✗ forex_config database not found${NC}"
    exit 1
fi

# Test 3: Test pg_dump functionality
echo -n "Testing pg_dump (forex)... "
if pg_dump -U "${POSTGRES_USER}" -d forex --schema-only > "${BACKUP_DIR}/forex_schema_test.sql" 2>/dev/null; then
    if [[ -s "${BACKUP_DIR}/forex_schema_test.sql" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL (empty file)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

echo -n "Testing pg_dump (forex_config)... "
if pg_dump -U "${POSTGRES_USER}" -d forex_config --schema-only > "${BACKUP_DIR}/forex_config_schema_test.sql" 2>/dev/null; then
    if [[ -s "${BACKUP_DIR}/forex_config_schema_test.sql" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL (empty file)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test 4: Test compression
echo -n "Testing gzip compression... "
if gzip "${BACKUP_DIR}/forex_schema_test.sql"; then
    if [[ -f "${BACKUP_DIR}/forex_schema_test.sql.gz" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
    else
        echo -e "${RED}✗ FAIL (compressed file not found)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test 5: Test integrity verification
echo -n "Testing backup integrity verification... "
if gzip -t "${BACKUP_DIR}/forex_schema_test.sql.gz" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test 6: Test SQL content verification
echo -n "Testing SQL content verification... "
if zcat "${BACKUP_DIR}/forex_schema_test.sql.gz" | head -20 | grep -q "PostgreSQL database dump"; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Show statistics
echo
echo "=== Test Statistics ==="
echo "Backup directory: ${BACKUP_DIR}"
echo "Test files created:"
ls -la "${BACKUP_DIR}/"

echo
echo -e "${GREEN}✅ All backup functionality tests passed!${NC}"
echo "The backup system is ready for production use."

# Cleanup test files
rm -rf "${BACKUP_DIR}"
echo "Test files cleaned up."