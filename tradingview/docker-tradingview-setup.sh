#!/bin/bash
"""
TradingView Service Setup Script

Sets up the containerized TradingView integration service within
your existing docker-compose environment.
"""

set -e

echo "🚀 Setting up TradingView containerized service..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs/tradingview
mkdir -p data

# Copy existing database if it exists
if [ -f "data/tvscripts.db" ]; then
    echo "✅ Using existing TradingView database"
else
    echo "📋 Database will be created on first container start"
fi

# Build and start the TradingView service
echo "🏗️ Building TradingView container..."
docker-compose build tradingview

echo "🚀 Starting TradingView service..."
docker-compose up -d tradingview

# Wait for service to be ready
echo "⏳ Waiting for TradingView service to be ready..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ TradingView service is healthy!"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts - waiting for service..."
        sleep 5
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Service health check failed"
    echo "📋 Checking container logs..."
    docker-compose logs tradingview
    exit 1
fi

# Test the service
echo "🧪 Testing TradingView API endpoints..."

echo "1️⃣ Testing health endpoint..."
curl -s http://localhost:8080/health | python3 -m json.tool

echo "2️⃣ Testing root endpoint..."
curl -s http://localhost:8080/ | python3 -m json.tool

echo "3️⃣ Testing stats endpoint..."
curl -s http://localhost:8080/api/tvscripts/stats | python3 -m json.tool

echo "4️⃣ Testing search endpoint..."
curl -s -X POST "http://localhost:8080/api/tvscripts/search?query=EMA&limit=3" | python3 -m json.tool

echo ""
echo "🎉 TradingView service setup complete!"
echo ""
echo "📋 Service Information:"
echo "   • API Server: http://localhost:8080"
echo "   • API Docs: http://localhost:8080/docs"
echo "   • Health Check: http://localhost:8080/health"
echo "   • Streamlit UI: http://localhost:8502 (when started)"
echo ""
echo "🔧 Available Commands:"
echo "   • View logs: docker-compose logs tradingview"
echo "   • Restart: docker-compose restart tradingview"
echo "   • Shell access: docker-compose exec tradingview bash"
echo ""
echo "🔍 Integration with existing services:"
echo "   • Task worker can access via: http://tradingview:8080/api/tvscripts"
echo "   • Environment variable added: TRADINGVIEW_API_URL"