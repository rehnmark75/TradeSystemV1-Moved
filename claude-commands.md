# Commands & CLI Reference

This document provides comprehensive command reference for managing Docker services, running CLI operations, and executing optimization tasks in TradeSystemV1.

## Docker Services Management

### Service Control
```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d fastapi-dev    # Development API server
docker-compose up -d fastapi-prod   # Production API server
docker-compose up -d fastapi-stream # Streaming service
docker-compose up -d task-worker    # Forex scanner worker
docker-compose up -d streamlit      # Dashboard UI

# View logs
docker-compose logs -f [service-name]

# Restart services
docker-compose restart [service-name]
```

### Database Access
```bash
# PostgreSQL access
docker exec -it postgres psql -U postgres -d forex
docker exec -it postgres psql -U postgres -d forex_config

# Quick database queries
docker exec postgres psql -U postgres -d forex -c "SELECT COUNT(*) FROM ig_candles;"
docker exec postgres psql -U postgres -d forex -c "SELECT DISTINCT epic FROM ema_best_parameters LIMIT 10;"
```

### Service Monitoring
```bash
# Check service status
docker-compose ps

# View resource usage
docker stats

# Check container health
docker-compose exec task-worker python -c "import pandas; print('✅ Dependencies OK')"
```

## Forex Scanner CLI

**IMPORTANT: All forex scanner commands must be run inside Docker containers, NOT on the host system.**
The host system lacks the required Python dependencies (pandas, etc.) and proper environment setup.

### Core Scanner Operations
```bash
# CORRECT: Run commands inside the task-worker container
docker-compose exec -T task-worker python -m forex_scanner.main scan
docker-compose exec -T task-worker python -m forex_scanner.main live
docker-compose exec -T task-worker python -m forex_scanner.main backtest --days 30

# WRONG: These will fail on the host system
# python -m forex_scanner.main scan  ❌ ModuleNotFoundError: No module named 'pandas'
# python forex_scanner/backtests/backtest_ema.py  ❌ Missing dependencies
```

### Strategy Backtesting
```bash
# EMA strategy backtests
docker-compose exec -T task-worker python forex_scanner/backtests/backtest_ema.py --epic "CS.D.EURUSD.MINI.IP" --days 30 --show-signals

# MACD strategy backtests  
docker-compose exec -T task-worker python forex_scanner/backtests/backtest_macd.py --epic "CS.D.AUDUSD.MINI.IP" --days 15 --show-signals

# Signal validation (specific timestamps)
docker-compose exec -T task-worker python forex_scanner/backtests/backtest_ema.py --epic "CS.D.EURUSD.MINI.IP" --validate-signal "2025-08-29 11:55:00"

# Complex commands with bash wrapper
docker-compose exec -T task-worker bash -c "python forex_scanner/backtests/backtest_ema.py --epic 'CS.D.AUDUSD.MINI.IP' --validate-signal '2025-08-29 11:55:00' --show-raw-data --show-calculations"
```

### Live Trading Operations
```bash
# Start live scanner (production)
docker-compose exec -T task-worker python -m forex_scanner.main live --mode production

# Start live scanner (paper trading)
docker-compose exec -T task-worker python -m forex_scanner.main live --mode paper

# Monitor live signals
docker-compose logs -f task-worker | grep "SIGNAL DETECTED"
```

## Parameter Optimization System

**IMPORTANT: The system uses dynamic, database-driven parameters instead of static config files.**

### EMA Parameter Optimization
```bash
# Single epic optimization (quick test)
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --epic CS.D.EURUSD.CEEM.IP --quick-test --days 7

# Single epic optimization (full test - 14,406 combinations)
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --epic CS.D.EURUSD.CEEM.IP --days 30

# Optimize all epics (comprehensive)
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --all-epics --days 30

# Parallel optimization (faster)
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --all-epics --days 30 --parallel
```

### MACD Parameter Optimization
```bash
# MACD optimization for specific epic
docker exec task-worker python forex_scanner/optimization/optimize_macd_parameters.py --epic CS.D.AUDUSD.MINI.IP --days 30

# MACD timeframe-specific optimization
docker exec task-worker python forex_scanner/optimization/optimize_macd_parameters.py --epic CS.D.AUDUSD.MINI.IP --timeframe 15m --days 30

# All MACD epics optimization
docker exec task-worker python forex_scanner/optimization/optimize_macd_parameters.py --all-epics --days 30
```

### Optimization Analysis & Results
```bash
# View optimization summary
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --summary

# Detailed epic analysis
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --epic CS.D.EURUSD.CEEM.IP --top-n 10

# View optimization runs history
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --runs

# Generate insights and recommendations
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --insights
```

### Dynamic Parameter System Testing
```bash
# Test dynamic parameter integration
docker exec task-worker python forex_scanner/optimization/test_dynamic_integration.py

# Check optimization status for all epics
docker exec task-worker python forex_scanner/optimization/dynamic_scanner_integration.py

# Test MACD parameter service
docker exec task-worker python forex_scanner/optimization/test_macd_optimization_system.py

# Verify parameter fallback mechanisms
docker exec task-worker python -c "
from forex_scanner.optimization.optimal_parameter_service import get_epic_optimal_parameters
params = get_epic_optimal_parameters('CS.D.EURUSD.CEEM.IP')
print(f'EMA: {params.ema_config}, Confidence: {params.confidence_threshold:.0%}')
"
```

## Development Commands

### Local Development Setup
```bash
# Install dependencies (in respective directories)
pip install -r requirements.txt

# Database migrations (if needed)
alembic upgrade head

# Run FastAPI locally (development)
cd dev-app
uvicorn main:app --reload --port 8001

# Run Streamlit locally
cd streamlit
streamlit run streamlit_app.py
```

### Testing & Validation
```bash
# Test strategy configurations
docker exec task-worker python -c "
from configdata.strategies import validate_strategy_configs
results = validate_strategy_configs()
print('✅' if results['overall_valid'] else '❌', 'Strategy configs:', results)
"

# Test database connections
docker exec task-worker python -c "
from core.database import DatabaseManager
import config
db = DatabaseManager(config.DATABASE_URL)
with db.get_connection() as conn:
    result = conn.execute('SELECT COUNT(*) FROM ig_candles').fetchone()
    print(f'✅ Database OK: {result[0]} candles')
"

# Validate optimization data
docker exec task-worker python -c "
from forex_scanner.optimization.optimal_parameter_service import get_macd_optimal_parameters
params = get_macd_optimal_parameters('CS.D.AUDUSD.MINI.IP', '15m')
print(f'✅ MACD optimization: {params.fast_ema}/{params.slow_ema}/{params.signal_ema}')
"
```

### Code Quality & Debugging
```bash
# Clear Python cache
docker-compose exec -T task-worker find /app -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Check imports and dependencies
docker exec task-worker python -c "
import sys
import forex_scanner
print(f'✅ forex_scanner path: {forex_scanner.__file__}')
print(f'✅ Python path: {sys.path[:3]}')
"

# Memory and performance monitoring
docker exec task-worker python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
print(f'CPU: {process.cpu_percent()}%')
"
```

## Troubleshooting Commands

### Common Issues
```bash
# Fix Docker permission issues
sudo chmod 666 /var/run/docker.sock

# Reset Docker environment
docker-compose down
docker system prune -f
docker-compose up -d

# Check service connectivity
docker exec task-worker curl -f http://fastapi-dev:8001/health || echo "❌ API unreachable"
docker exec postgres psql -U postgres -c "SELECT 1" || echo "❌ Database unreachable"

# Verify configuration loading
docker exec task-worker python -c "
from configdata import config
print(f'✅ EMA periods: {config.strategies.EMA_STRATEGY_CONFIG[\"aggressive\"]}')
print(f'✅ Database URL: {config.DATABASE_URL[:20]}...')
"
```

### Log Analysis
```bash
# Search for specific errors
docker-compose logs task-worker | grep -E "(ERROR|FAILED|❌)"

# Monitor signal generation
docker-compose logs -f task-worker | grep -E "(SIGNAL DETECTED|✅.*signal)"

# Track optimization progress
docker-compose logs -f task-worker | grep -E "(optimization|parameters|Score:)"

# Database query logs
docker-compose logs postgres | grep -E "(ERROR|WARN)"
```

### Performance Monitoring
```bash
# Monitor container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check disk usage
docker system df
docker exec postgres du -sh /var/lib/postgresql/data

# Database performance
docker exec postgres psql -U postgres -d forex -c "
SELECT schemaname,tablename,n_live_tup,n_dead_tup 
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC LIMIT 10;
"
```

## Quick Reference

### Most Common Commands
```bash
# Start system
docker-compose up -d

# Run EMA backtest
docker exec task-worker python forex_scanner/backtests/backtest_ema.py --epic "CS.D.EURUSD.MINI.IP" --days 30

# Optimize parameters
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --epic CS.D.EURUSD.CEEM.IP --quick-test

# Check results
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --summary

# Monitor logs
docker-compose logs -f task-worker
```

For more detailed system information, see [Architecture Overview](claude-architecture.md).
For strategy-specific commands, see [Strategy Development](claude-strategies.md).
For optimization details, see [Dynamic Parameter System](claude-optimization.md).