# 🐳 TradingView Docker Integration Guide

Your TradingView integration is now **fully containerized** and integrated into your existing docker-compose infrastructure!

## 🚀 Quick Start

### **1. Setup (One Time)**
```bash
# Run the setup script
./docker-tradingview-setup.sh
```

This will:
- ✅ Build the TradingView container
- ✅ Start the service
- ✅ Initialize database with 15 scripts (5 strategies + 10 indicators)
- ✅ Run health checks
- ✅ Test all endpoints

### **2. Daily Usage**
```bash
# Manage the service
./manage-tradingview.sh [command]

# Examples:
./manage-tradingview.sh status     # Check if running
./manage-tradingview.sh search EMA # Search for EMA scripts
./manage-tradingview.sh stats      # View library statistics
./manage-tradingview.sh logs       # View service logs
```

## 📊 Service Architecture

### **New Container: `tradingview`**
- **API Server**: http://localhost:8080
- **Database**: SQLite with 15 TradingView scripts
- **Integration**: Connected to your existing PostgreSQL
- **Health Monitoring**: Built-in health checks

### **Integration Points**
```yaml
# Updated docker-compose.yml includes:
tradingview:
  ports:
    - "8080:8080"  # FastAPI API
    - "8502:8501"  # Streamlit UI
  environment:
    - TRADINGVIEW_API_URL=http://tradingview:8080/api/tvscripts
  volumes:
    - tradingview_data:/app/data  # Persistent database
```

## 🔧 Available Endpoints

### **API Endpoints (Port 8080)**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/api/tvscripts/search` | POST | Search scripts |
| `/api/tvscripts/script/{slug}` | GET | Get script details |
| `/api/tvscripts/stats` | GET | Library statistics |
| `/api/tvscripts/analyze` | POST | Analyze script code |

### **Web Interfaces**
- **API Documentation**: http://localhost:8080/docs
- **API Explorer**: http://localhost:8080/redoc
- **Health Dashboard**: http://localhost:8080/health

## 🎯 Integration with Your Existing Services

### **1. Task Worker Integration**
Your task-worker now has access via environment variable:
```bash
docker exec task-worker curl http://tradingview:8080/api/tvscripts/stats
```

### **2. Strategy Enhancement Workflow**
```bash
# 1. Search for community strategies
./manage-tradingview.sh search "EMA crossover"

# 2. Get detailed parameters
curl http://localhost:8080/api/tvscripts/script/triple-ema-system

# 3. Add to your strategy configs
# Edit: worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py

# 4. Test with your scanner
docker exec forex_scanner python -m forex_scanner.main --strategy ema --preset triple_ema_tv
```

### **3. Automated Parameter Discovery**
```python
# In your optimization scripts:
import requests

# Get community-proven parameters
response = requests.get('http://tradingview:8080/api/tvscripts/search?query=EMA&limit=10')
strategies = response.json()['results']

# Extract popular parameter ranges
ema_periods = []
for strategy in strategies:
    # Extract EMA periods from strategy code
    # Use for optimization bounds
```

## 📋 Management Commands

### **Service Management**
```bash
./manage-tradingview.sh start      # Start service
./manage-tradingview.sh stop       # Stop service  
./manage-tradingview.sh restart    # Restart service
./manage-tradingview.sh status     # Show status
./manage-tradingview.sh health     # Health check
```

### **Data Operations**
```bash
./manage-tradingview.sh stats              # Library statistics
./manage-tradingview.sh search "MACD"      # Search scripts
./manage-tradingview.sh logs               # View logs
```

### **Development**
```bash
./manage-tradingview.sh shell      # Container shell access
./manage-tradingview.sh rebuild    # Rebuild container
./manage-tradingview.sh cleanup    # Remove all data
```

## 🔄 Integration Workflow

### **Enhanced Strategy Development**
1. **Discovery**: Search community scripts for your indicators
2. **Analysis**: Extract proven parameter ranges  
3. **Implementation**: Add community parameters as new presets
4. **Testing**: A/B test community vs optimized parameters
5. **Optimization**: Use community ranges as starting points

### **Example: EMA Strategy Enhancement**
```bash
# 1. Find EMA strategies
./manage-tradingview.sh search EMA

# 2. Get Triple EMA details (8, 21, 55 periods)
curl http://localhost:8080/api/tvscripts/script/triple-ema-system

# 3. Add to your config
# File: worker/app/forex_scanner/configdata/strategies/config_ema_strategy.py
'triple_ema_tv': {
    'short': 8,      # Community proven
    'long': 21,      # 420 likes validation
    'trend': 55,     # Popular parameters
    'description': 'TradingView Triple EMA - community validated'
}

# 4. Test immediately
docker exec forex_scanner python -m forex_scanner.main --strategy ema --preset triple_ema_tv
```

## 📊 Data Persistence

### **Database Storage**
- **Volume**: `tradingview_data` (persistent across restarts)
- **Local Backup**: `./data/tvscripts.db` (mounted from host)
- **Scripts**: 15 total (5 strategies + 10 indicators)

### **Backup & Restore**
```bash
# Backup database
docker cp tradingview:/app/data/tvscripts.db ./backup_tvscripts.db

# Restore database  
docker cp ./backup_tvscripts.db tradingview:/app/data/tvscripts.db
./manage-tradingview.sh restart
```

## 🚀 Production Deployment

### **Resource Requirements**
- **CPU**: 0.5 cores
- **Memory**: 512MB
- **Storage**: 100MB (database + logs)
- **Network**: Internal only (no external API calls)

### **Monitoring**
```bash
# Health monitoring
while true; do
  ./manage-tradingview.sh health
  sleep 30
done

# Performance monitoring  
docker stats tradingview
```

### **Scaling**
The service is designed to be:
- **Stateless**: All data in persistent volumes
- **Scalable**: Can run multiple instances behind load balancer
- **Resilient**: Auto-restart on failure

## 🎉 Benefits of Containerization

### **✅ Advantages Over Host Files**
1. **Isolation**: No host dependency conflicts
2. **Portability**: Works on any Docker environment
3. **Scalability**: Easy to scale or replicate
4. **Management**: Integrated with your existing stack
5. **Persistence**: Data survives container restarts
6. **Monitoring**: Built-in health checks and logging

### **🔧 Integration Benefits**
1. **Service Discovery**: Other containers can access via `http://tradingview:8080`
2. **Network Isolation**: Secure internal communication
3. **Resource Management**: Docker handles CPU/memory limits
4. **Logging**: Centralized log management
5. **Backup**: Volume-based data persistence

## 🎯 Next Steps

1. **Immediate**: Run `./docker-tradingview-setup.sh` to get started
2. **Integration**: Use the search API in your optimization scripts  
3. **Enhancement**: Add community parameters to your strategy configs
4. **Monitoring**: Set up health check monitoring
5. **Scaling**: Consider multiple instances for high availability

Your TradingView integration is now **production-ready** and **fully integrated** with your trading infrastructure! 🚀