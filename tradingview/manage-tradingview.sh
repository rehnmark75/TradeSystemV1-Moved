#!/bin/bash
"""
TradingView Service Management Script

Provides easy management commands for the containerized TradingView service.
"""

set -e

SERVICE_NAME="tradingview"
API_URL="http://localhost:8080"

show_help() {
    echo "🎯 TradingView Service Management"
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start          Start the TradingView service"
    echo "  stop           Stop the TradingView service"
    echo "  restart        Restart the TradingView service"
    echo "  status         Show service status"
    echo "  logs           Show service logs"
    echo "  shell          Open shell in container"
    echo "  health         Check service health"
    echo "  stats          Show library statistics"
    echo "  search [TERM]  Search for scripts"
    echo "  rebuild        Rebuild and restart service"
    echo "  cleanup        Stop and remove service data"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 search EMA"
    echo "  $0 search \"momentum indicators\""
    echo "  $0 stats"
}

start_service() {
    echo "🚀 Starting TradingView service..."
    docker-compose up -d $SERVICE_NAME
    
    echo "⏳ Waiting for service to be ready..."
    sleep 5
    
    if check_health_quiet; then
        echo "✅ TradingView service started successfully"
        echo "   API: $API_URL"
        echo "   Docs: $API_URL/docs"
    else
        echo "❌ Service failed to start properly"
        echo "📋 Checking logs..."
        docker-compose logs --tail=20 $SERVICE_NAME
        return 1
    fi
}

stop_service() {
    echo "🛑 Stopping TradingView service..."
    docker-compose stop $SERVICE_NAME
    echo "✅ Service stopped"
}

restart_service() {
    echo "🔄 Restarting TradingView service..."
    stop_service
    start_service
}

show_status() {
    echo "📊 TradingView Service Status:"
    if docker-compose ps $SERVICE_NAME | grep -q "Up"; then
        echo "   Status: ✅ Running"
        
        if check_health_quiet; then
            echo "   Health: ✅ Healthy"
        else
            echo "   Health: ❌ Unhealthy"
        fi
        
        echo "   Ports: $(docker-compose ps $SERVICE_NAME | grep -o '0.0.0.0:[0-9]*->[0-9]*' | tr '\n' ' ')"
    else
        echo "   Status: ❌ Not running"
    fi
}

show_logs() {
    echo "📋 TradingView Service Logs:"
    docker-compose logs --tail=50 -f $SERVICE_NAME
}

open_shell() {
    echo "🐚 Opening shell in TradingView container..."
    docker-compose exec $SERVICE_NAME bash
}

check_health_quiet() {
    curl -f -s $API_URL/health > /dev/null 2>&1
}

check_health() {
    echo "🏥 Checking TradingView service health..."
    
    if check_health_quiet; then
        echo "✅ Service is healthy"
        curl -s $API_URL/health | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"   Database: {'✅' if data.get('database_available') else '❌'}\"
print(f\"   Integration: {'✅' if data.get('integration_available') else '❌'}\")"
    else
        echo "❌ Service is unhealthy or not responding"
        return 1
    fi
}

show_stats() {
    echo "📊 TradingView Library Statistics:"
    
    if ! check_health_quiet; then
        echo "❌ Service not available"
        return 1
    fi
    
    curl -s $API_URL/api/tvscripts/stats | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"   Database: {data.get('database_type', 'unknown')}\")
    print(f\"   Total Scripts: {data['total_scripts']}\")
    
    print(f\"   Script Types:\")
    for stype, count in data.get('script_types', {}).items():
        print(f\"     {stype}: {count}\")
    
    print(f\"   Categories:\")
    for cat, count in data.get('categories', {}).items():
        print(f\"     {cat}: {count}\")
    
    print(f\"   Top Scripts:\")
    for i, script in enumerate(data.get('top_scripts', [])[:3], 1):
        author = script.get('author', 'Unknown')
        title = script.get('title', 'Unknown')
        likes = script.get('likes', 0)
        print(f\"     {i}. {title} by {author} ({likes:,} likes)\")
        
    avg_likes = data.get('averages', {}).get('likes', 0)
    avg_views = data.get('averages', {}).get('views', 0)
    print(f\"   Averages: {avg_likes:,.0f} likes, {avg_views:,.0f} views\")
    
except Exception as e:
    print(f\"Error parsing stats: {e}\")
    print(sys.stdin.read())
"
}

search_scripts() {
    local query="$1"
    if [ -z "$query" ]; then
        echo "❌ Please provide a search term"
        echo "Usage: $0 search [TERM]"
        return 1
    fi
    
    echo "🔍 Searching for: '$query'"
    
    if ! check_health_quiet; then
        echo "❌ Service not available"
        return 1
    fi
    
    curl -s -X POST "$API_URL/api/tvscripts/search?query=$query&limit=5" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    results = data['results']
    print(f\"Found {data['count']} results:\")
    for i, script in enumerate(results, 1):
        print(f\"  {i}. {script['title']} by {script['author']}\")
        print(f\"     Type: {script['strategy_type']}, Likes: {script['likes']:,}\")
        print(f\"     {script['description'][:80]}...\")
        print()
except Exception as e:
    print(f\"Error parsing results: {e}\")
    print(sys.stdin.read())
"
}

rebuild_service() {
    echo "🏗️ Rebuilding TradingView service..."
    docker-compose stop $SERVICE_NAME
    docker-compose build --no-cache $SERVICE_NAME
    start_service
}

cleanup_service() {
    echo "🧹 Cleaning up TradingView service..."
    read -p "This will remove all service data. Continue? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down $SERVICE_NAME
        docker-compose rm -f $SERVICE_NAME
        docker volume rm -f tradesystemv1_tradingview_data 2>/dev/null || true
        echo "✅ Service data cleaned up"
    else
        echo "❌ Cleanup cancelled"
    fi
}

# Main command handling
case "$1" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    health)
        check_health
        ;;
    stats)
        show_stats
        ;;
    search)
        search_scripts "$2"
        ;;
    rebuild)
        rebuild_service
        ;;
    cleanup)
        cleanup_service
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac