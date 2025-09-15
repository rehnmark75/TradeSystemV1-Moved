#!/bin/bash
# LuxAlgo Collection Downloader Management Script
#
# This script manages the download and import of LuxAlgo indicators into your
# TradingView database, providing comprehensive collection management.

set -e

# Configuration
SCRIPT_DIR="/home/hr/Projects/TradeSystemV1/tradingview"
LOG_DIR="/home/hr/Projects/TradeSystemV1/logs/tradingview"
LUXALGO_SCRIPT="$SCRIPT_DIR/fetch_luxalgo_indicators.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    local status=$1
    local message=$2
    case $status in
        "ok") echo -e "${GREEN}✅ $message${NC}" ;;
        "warn") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "info") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "error") echo -e "${RED}❌ $message${NC}" ;;
    esac
}

print_header() {
    echo -e "${BLUE}"
    echo "🔥 LuxAlgo Collection Downloader for TradeSystemV1"
    echo "================================================="
    echo -e "${NC}"
}

check_prerequisites() {
    print_status "info" "Checking prerequisites..."

    # Check if TradingView service is running
    if curl -f -s http://localhost:8080/health > /dev/null; then
        print_status "ok" "TradingView API service is running"
    else
        print_status "error" "TradingView API service is not running"
        echo "   Please start it: docker-compose up -d tradingview"
        return 1
    fi

    # Check database connectivity
    if docker exec postgres psql -U postgres -d forex -c '\q' > /dev/null 2>&1; then
        print_status "ok" "PostgreSQL database is accessible"
    else
        print_status "error" "PostgreSQL database is not accessible"
        echo "   Please start it: docker-compose up -d postgres"
        return 1
    fi

    # Check Python script exists
    if [ -f "$LUXALGO_SCRIPT" ]; then
        print_status "ok" "LuxAlgo downloader script found"
    else
        print_status "error" "LuxAlgo downloader script not found at $LUXALGO_SCRIPT"
        return 1
    fi

    # Check log directory
    mkdir -p "$LOG_DIR"
    print_status "ok" "Log directory ready"

    return 0
}

show_collection_status() {
    print_status "info" "Checking current collection status..."

    # Check if we have any LuxAlgo scripts in database
    local luxalgo_count=$(docker exec postgres psql -U postgres -d forex -t -c "SELECT COUNT(*) FROM tvscripts WHERE is_luxalgo = TRUE;" 2>/dev/null | tr -d ' \n' || echo "0")

    if [ "$luxalgo_count" != "" ] && [ "$luxalgo_count" -gt 0 ] 2>/dev/null; then
        print_status "ok" "Found $luxalgo_count LuxAlgo scripts in database"

        # Get detailed stats
        echo "📊 Current Collection Statistics:"
        docker exec postgres psql -U postgres -d forex -c "
            SELECT
                'Total Scripts' as metric, COUNT(*)::text as value FROM tvscripts
            UNION ALL
            SELECT 'LuxAlgo Scripts', COUNT(*)::text FROM tvscripts WHERE is_luxalgo = TRUE
            UNION ALL
            SELECT 'Open Source', COUNT(*)::text FROM tvscripts WHERE open_source = TRUE
            UNION ALL
            SELECT 'LuxAlgo Open Source', COUNT(*)::text FROM tvscripts WHERE is_luxalgo = TRUE AND open_source = TRUE;
        " 2>/dev/null || true
    else
        print_status "warn" "No LuxAlgo scripts found in database"
        echo "   This appears to be a fresh installation"
    fi
}

download_luxalgo_collection() {
    print_status "info" "Starting LuxAlgo collection download..."

    # Ensure services are ready
    if ! check_prerequisites; then
        return 1
    fi

    # Create backup before major operation
    print_status "info" "Creating database backup..."
    docker exec postgres pg_dump -U postgres forex > "/tmp/forex_backup_$(date +%Y%m%d_%H%M%S).sql" 2>/dev/null || true

    # Run the Python downloader
    print_status "info" "Executing LuxAlgo downloader..."
    echo "📥 This may take 10-20 minutes depending on the number of indicators..."
    echo "📋 Check logs at: $LOG_DIR/luxalgo_downloader.log"

    if python3 "$LUXALGO_SCRIPT"; then
        print_status "ok" "LuxAlgo collection download completed successfully!"

        # Show updated stats
        echo ""
        show_collection_status

        print_status "ok" "Collection is now available in your Streamlit interface"
        echo "   🔗 Access: http://localhost:8501 → TradingView Importer"

        return 0
    else
        print_status "error" "LuxAlgo collection download failed"
        echo "   📋 Check logs: $LOG_DIR/luxalgo_downloader.log"
        return 1
    fi
}

search_luxalgo_indicators() {
    print_status "info" "Searching for LuxAlgo indicators in current collection..."

    docker exec postgres psql -U postgres -d forex -c "
        SELECT
            title,
            author,
            luxalgo_category,
            likes,
            CASE WHEN open_source THEN 'Yes' ELSE 'No' END as open_source
        FROM tvscripts
        WHERE is_luxalgo = TRUE
        ORDER BY likes DESC
        LIMIT 20;
    " 2>/dev/null || print_status "error" "Failed to query database"
}

show_luxalgo_categories() {
    print_status "info" "LuxAlgo indicator categories:"

    docker exec postgres psql -U postgres -d forex -c "
        SELECT
            luxalgo_category as \"Category\",
            COUNT(*) as \"Count\",
            AVG(likes)::int as \"Avg Likes\"
        FROM tvscripts
        WHERE is_luxalgo = TRUE AND luxalgo_category IS NOT NULL
        GROUP BY luxalgo_category
        ORDER BY COUNT(*) DESC;
    " 2>/dev/null || print_status "error" "Failed to query database"
}

test_integration() {
    print_status "info" "Testing LuxAlgo integration with Streamlit..."

    # Test TradingView API search for LuxAlgo
    print_status "info" "Testing API search for 'luxalgo'..."
    local search_result=$(curl -s -X POST "http://localhost:8080/api/tvscripts/search?query=luxalgo&limit=5")

    if echo "$search_result" | grep -q "results"; then
        local result_count=$(echo "$search_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(len(data.get('results', [])))
except:
    print('0')
" 2>/dev/null)
        print_status "ok" "API search working - found $result_count results"
    else
        print_status "error" "API search failed"
    fi

    # Test database query
    local db_count=$(docker exec postgres psql -U postgres -d forex -t -c "SELECT COUNT(*) FROM tvscripts WHERE is_luxalgo = TRUE;" 2>/dev/null | tr -d ' ' || echo "0")
    print_status "ok" "Database contains $db_count LuxAlgo scripts"

    print_status "ok" "Integration test completed"
}

show_help() {
    print_header
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  download      Download complete LuxAlgo collection"
    echo "  status        Show current collection status"
    echo "  search        Search and display LuxAlgo indicators"
    echo "  categories    Show LuxAlgo indicator categories"
    echo "  test          Test integration functionality"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 download       # Download all LuxAlgo indicators"
    echo "  $0 status         # Check current collection"
    echo "  $0 search         # Browse downloaded indicators"
    echo ""
    echo "Files:"
    echo "  Script: $LUXALGO_SCRIPT"
    echo "  Logs: $LOG_DIR/luxalgo_downloader.log"
}

# Main command handling
case "${1:-help}" in
    download)
        print_header
        download_luxalgo_collection
        ;;
    status)
        print_header
        show_collection_status
        ;;
    search)
        print_header
        search_luxalgo_indicators
        ;;
    categories)
        print_header
        show_luxalgo_categories
        ;;
    test)
        print_header
        test_integration
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac