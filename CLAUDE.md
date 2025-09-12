# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Docker Services Management
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

# Database access
docker exec -it postgres psql -U postgres -d forex
docker exec -it postgres psql -U postgres -d forex_config
```

### Forex Scanner CLI
**IMPORTANT: All forex scanner commands must be run inside Docker containers, NOT on the host system.**
The host system lacks the required Python dependencies (pandas, etc.) and proper environment setup.

```bash
# CORRECT: Run commands inside the task-worker container
docker-compose exec -T task-worker python -m forex_scanner.main scan
docker-compose exec -T task-worker python -m forex_scanner.main live
docker-compose exec -T task-worker python -m forex_scanner.main backtest --days 30

# CORRECT: Run backtests inside container
docker-compose exec -T task-worker python forex_scanner/backtests/backtest_ema.py --epic "CS.D.EURUSD.MINI.IP" --validate-signal "2025-08-29 11:55:00"

# WRONG: These will fail on the host system
# python -m forex_scanner.main scan  ❌ ModuleNotFoundError: No module named 'pandas'
# python forex_scanner/backtests/backtest_ema.py  ❌ Missing dependencies

# Alternative: Run with bash wrapper for complex commands
docker-compose exec -T task-worker bash -c "python forex_scanner/backtests/backtest_ema.py --epic 'CS.D.AUDUSD.MINI.IP' --validate-signal '2025-08-29 11:55:00' --show-raw-data --show-calculations"
```

### EMA Parameter Optimization System
**IMPORTANT: The system now uses dynamic, database-driven parameters instead of static config files.**

```bash
# Run parameter optimization for single epic
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --epic CS.D.EURUSD.CEEM.IP --quick-test --days 7

# Run full optimization (14,406 combinations)
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --epic CS.D.EURUSD.CEEM.IP --days 30

# Optimize all epics
docker exec task-worker python forex_scanner/optimization/optimize_ema_parameters.py --all-epics --days 30

# View optimization results
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --summary
docker exec task-worker python forex_scanner/optimization/optimization_analysis.py --epic CS.D.EURUSD.CEEM.IP --top-n 10

# Test dynamic parameter system
docker exec task-worker python forex_scanner/optimization/test_dynamic_integration.py

# Check optimization status
docker exec task-worker python forex_scanner/optimization/dynamic_scanner_integration.py
```

### Development Commands
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

## Architecture Overview

### Service Architecture
The system follows a microservices architecture with Docker containers:

1. **FastAPI Services** (dev-app, prod-app, stream-app)
   - REST APIs for trading operations
   - WebSocket streaming for real-time data
   - Integration with IG Markets API
   - Database: PostgreSQL with two schemas (forex, forex_config)

2. **Task Worker** (worker/app)
   - Forex scanner with multiple trading strategies (EMA, MACD, KAMA, Smart Money)
   - Claude AI integration for signal analysis
   - Backtesting engine for strategy validation
   - Alert system with deduplication

3. **Streamlit Dashboard** (streamlit/)
   - Real-time trading visualization
   - Performance analytics
   - Log monitoring

4. **Supporting Services**
   - PostgreSQL database
   - Nginx reverse proxy with SSL
   - PgAdmin for database management

### Key Database Models
- **TradeLog** (dev-app/services/models.py): Core trading record with P&L tracking
- **IGCandle**: Market data with composite primary key (time, epic, timeframe)
- **Candle**: Historical price data storage

### Trading System Components

#### Signal Detection Pipeline
1. Data fetching from database/API
2. Technical indicator calculation (EMAs, MACD, Two-Pole Oscillator, Momentum Bias Index)
3. Multi-timeframe analysis for trend confirmation
4. 5-layer signal validation system
5. Alert generation with deduplication
6. Order execution via API

#### EMA Strategy - Advanced 5-Layer Validation System
The core EMA strategy (`worker/app/forex_scanner/core/strategies/ema_strategy.py`) implements a sophisticated multi-layer validation system:

**Signal Requirements:**
- **BULL**: Price crosses above EMA(21), with EMA(21) > EMA(50) > EMA(200)
- **BEAR**: Price crosses below EMA(21), with EMA(21) < EMA(50) < EMA(200)

**5-Layer Validation System:**
1. **EMA Crossover Detection**: Price crossing short EMA with proper trend alignment
2. **15m Two-Pole Oscillator**: Momentum validation (GREEN for BUY, PURPLE for SELL)  
3. **1H Two-Pole Oscillator**: Higher timeframe momentum confirmation
4. **Momentum Bias Index**: Final momentum validation (AlgoAlpha's indicator)
5. **EMA 200 Trend Filter**: Major trend direction validation

**Key Features:**
- Multi-timeframe momentum confirmation (15m + 1H)
- Extremely selective signal generation (high win rate, fewer signals)
- Protects against false breakouts and unstable momentum
- Configurable boundary requirements for momentum strength

**Configuration:**
```python
# Two-Pole Oscillator
TWO_POLE_OSCILLATOR_ENABLED = True
TWO_POLE_MTF_VALIDATION = True        # 1H timeframe validation

# Momentum Bias Index  
MOMENTUM_BIAS_ENABLED = True
MOMENTUM_BIAS_REQUIRE_ABOVE_BOUNDARY = False  # Optional stricter filtering

# EMA 200 Trend Filter
EMA_200_TREND_FILTER_ENABLED = True
```

**Performance Characteristics:**
- Very low signal frequency due to strict validation
- High win rates (often 100% in testing)
- Excellent risk management through multi-layer filtering
- Prevents trading against momentum or trend direction

#### Key Configuration Files
- `worker/app/forex_scanner/config.py`: Scanner settings, trading pairs, thresholds
- `dev-app/config.py`: API endpoints, database connections
- `docker-compose.yml`: Service orchestration and networking

### Important Patterns

#### Service Communication
- Internal services communicate via Docker network (lab-net)
- External API calls to IG Markets for trading operations
- PostgreSQL as central data store
- Redis/file-based caching for performance

#### Error Handling
- Comprehensive logging with RotatingFileHandler
- Retry mechanisms for API calls
- Database transaction management with proper rollback

#### Security Considerations
- Azure Key Vault integration for secrets
- SSL/TLS via Nginx and Certbot
- Environment-based configuration
- No hardcoded credentials

### Testing Approach
The system uses backtesting for strategy validation rather than unit tests:
- Historical data analysis over configurable periods
- Performance metrics calculation
- Strategy comparison tools
- Real-time monitoring and logging for production validation

## Strategy Development - Lightweight Configuration Pattern

### Architecture Principles

**IMPORTANT: All new strategies MUST follow this modular configuration pattern to avoid monolithic config.py bloat.**

#### 1. Modular Configuration Structure
```
configdata/
├── strategies/
│   ├── __init__.py (exports all strategy configs)
│   ├── config_ema_strategy.py (dedicated EMA config)
│   ├── config_macd_strategy.py (dedicated MACD config)
│   └── config_zero_lag_strategy.py (dedicated Zero-Lag config)
├── smc/
│   └── smc_configdata.py (Smart Money Concepts)
└── __init__.py (main Config class with convenience methods)
```

#### 2. Strategy Config File Template
Each strategy MUST have its own dedicated config file following this pattern:

```python
# configdata/strategies/config_[strategy]_strategy.py

# Strategy enable/disable
[STRATEGY]_STRATEGY = True

# Main configuration dictionary with multiple presets
[STRATEGY]_STRATEGY_CONFIG = {
    'default': {
        'short': 21, 'long': 50, 'trend': 200,
        'description': 'Balanced configuration',
        'best_for': ['trending', 'medium_volatility'],
        'best_volatility_regime': 'medium',
        'best_trend_strength': 'medium',
        'best_market_regime': 'trending',
        'best_session': ['london', 'new_york'],
        'preferred_pairs': ['CS.D.EURUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP'],
        'min_pip_volatility': 8.0,
        'max_pip_volatility': 50.0
    },
    'conservative': {...},
    'aggressive': {...},
    'scalping': {...},
    'swing': {...},
    'news_safe': {...},
    'crypto': {...}
}

# Active configuration selector
ACTIVE_[STRATEGY]_CONFIG = 'default'

# Individual feature toggles
[STRATEGY]_FEATURE_ENABLED = True
[STRATEGY]_FILTER_ENABLED = False

# Helper functions
def get_[strategy]_config_for_epic(epic: str, market_condition: str = 'default') -> dict:
    """Get strategy configuration for specific epic with fallbacks"""
    return [STRATEGY]_STRATEGY_CONFIG.get(market_condition, [STRATEGY]_STRATEGY_CONFIG['default'])

def get_[strategy]_threshold_for_epic(epic: str) -> float:
    """Get strategy-specific thresholds based on currency pair"""
    # Implementation with JPY vs non-JPY logic

# Validation function  
def validate_[strategy]_config() -> dict:
    """Validate strategy configuration completeness"""
    try:
        required_keys = ['[STRATEGY]_STRATEGY', '[STRATEGY]_STRATEGY_CONFIG']
        for key in required_keys:
            if not globals().get(key):
                return {'valid': False, 'error': f'Missing {key}'}
        return {'valid': True, 'config_count': len([STRATEGY]_STRATEGY_CONFIG)}
    except Exception as e:
        return {'valid': False, 'error': str(e)}
```

#### 3. Integration Requirements

**A. Update strategies/__init__.py:**
```python
from .config_[strategy]_strategy import *

# Add all exports to __all__
__all__.extend([
    '[STRATEGY]_STRATEGY',
    '[STRATEGY]_STRATEGY_CONFIG', 
    'ACTIVE_[STRATEGY]_CONFIG',
    'get_[strategy]_config_for_epic',
    'get_[strategy]_threshold_for_epic',
    'validate_[strategy]_config'
])

# Add to validation function
def validate_strategy_configs():
    validation_results = {
        'zero_lag': validate_zero_lag_config(),
        'macd': validate_macd_config(),
        'ema': validate_ema_config(),
        '[strategy]': validate_[strategy]_config()  # Add new strategy
    }
    validation_results['overall_valid'] = all(result.get('valid', False) for result in validation_results.values())
    return validation_results
```

**B. Update main configdata/__init__.py:**
```python
class Config:
    def __init__(self):
        from configdata import strategies
        self.strategies = strategies
        
        # Add convenience methods for new strategy
    def get_[strategy]_config_for_epic(self, epic: str, condition: str = 'default') -> dict:
        if hasattr(self.strategies, 'get_[strategy]_config_for_epic'):
            return self.strategies.get_[strategy]_config_for_epic(epic, condition)
        # Fallback logic
        return {'short': 21, 'long': 50, 'trend': 200}
        
    def get_[strategy]_threshold_for_epic(self, epic: str) -> float:
        if hasattr(self.strategies, 'get_[strategy]_threshold_for_epic'):
            return self.strategies.get_[strategy]_threshold_for_epic(epic)
        return 0.00003  # Default fallback
```

#### 4. Critical Integration Points

**A. Data Fetcher Integration:**
- MUST handle fallback when `[strategy]_strategy=None` parameter
- Use configdata convenience methods as primary source
- Maintain backward compatibility with old config keys
- **CRITICAL**: Fix key mismatches (e.g., `'fast'` vs `'fast_ema'` seen in MACD refactor)

**B. Import Strategy (MANDATORY):**
```python
# In files using strategy configs:
from configdata import config        # For strategy configs  
import config as system_config       # For system-level settings

# In TechnicalAnalyzer and similar classes:
from configdata import config        # MUST add this import
```

#### 5. Common Issues to Avoid

1. **Key Mismatch**: Ensure consistent naming (`'fast'` vs `'fast_ema'` caused runtime errors)
2. **Missing Imports**: MUST add `from configdata import config` to TechnicalAnalyzer
3. **Fallback Logic**: Always provide fallbacks when strategy=None in data fetcher
4. **Validation**: Include both period columns (`ema_21`, `ema_50`, `ema_200`) AND semantic columns (`ema_short`, `ema_long`, `ema_trend`)
5. **Default Alignment**: Ensure hardcoded defaults match configdata defaults (e.g., `[21, 50, 200]` not `[9, 21, 200]`)

#### 6. Benefits Achieved

- **86% reduction** in main config.py size (EMA: 10,078 → 1,394 lines)
- **Modular architecture** - each strategy completely self-contained
- **Multiple presets** per strategy (7 configs: default, conservative, aggressive, scalping, swing, news_safe, crypto)
- **Backward compatibility** maintained through Config singleton pattern
- **Easy testing** and **strategy isolation**
- **Comprehensive validation** system with detailed error reporting
- **Epic-specific configurations** with intelligent fallbacks

#### 7. Validation Pattern (MANDATORY)

Each strategy config MUST include comprehensive validation covering:
- Required configuration keys exist and are properly typed
- Value ranges are reasonable (e.g., EMAs: short < long < trend)
- Epic-specific configurations are valid
- Fallback mechanisms work correctly
- Configuration presets are complete and consistent

**Example Validation Results:**
```python
✅ ZeroLag configuration validation passed
✅ MACD configuration validation passed  
✅ EMA configuration validation passed
✅ Strategy configurations loaded successfully
```

#### 8. Success Metrics

A properly implemented strategy using this pattern will show:
- ✅ No configuration-related runtime errors
- ✅ All validation tests pass
- ✅ Clean separation from system config
- ✅ Multiple working presets
- ✅ Backward compatibility maintained
- ✅ Significant reduction in main config.py size

**This pattern prevents the monolithic config.py anti-pattern and ensures clean, maintainable, and extensible strategy configurations.**

## Dynamic Parameter System (Latest Enhancement)

### Overview
The system has been enhanced with a revolutionary dynamic parameter system that automatically uses optimal parameters from optimization results instead of static config files. This creates a truly intelligent, self-optimizing trading system.

### Architecture Components

#### 1. OptimalParameterService (`optimization/optimal_parameter_service.py`)
- **Purpose**: Retrieves optimal parameters from database with intelligent caching
- **Features**: Market context awareness, fallback mechanisms, performance tracking
- **Caching**: 30-minute cache duration for performance optimization

```python
from optimization.optimal_parameter_service import get_epic_optimal_parameters

# Get optimal parameters for any epic
params = get_epic_optimal_parameters('CS.D.EURUSD.CEEM.IP')
# Returns: EMA config, confidence, timeframe, SL/TP, performance score
```

#### 2. Enhanced EMA Strategy (`core/strategies/ema_strategy.py`)
- **Dynamic Parameter Loading**: Automatically uses optimal parameters per epic
- **Backward Compatibility**: Falls back to config files when optimization data unavailable
- **Epic-specific Configuration**: Different optimal settings for each trading pair

```python
# NEW: Dynamic parameter integration
strategy = EMAStrategy(
    epic='CS.D.EURUSD.CEEM.IP',
    use_optimal_parameters=True  # Uses database-driven parameters
)
# Automatically gets optimal: EMA periods, confidence, SL/TP levels
```

#### 3. DynamicEMAScanner (`optimization/dynamic_scanner_integration.py`)
- **Intelligent Scanning**: Automatically creates optimized strategies for each epic
- **Optimization Reporting**: Shows which epics are optimized vs need optimization
- **Smart Recommendations**: Suggests when to re-optimize based on performance

```python
scanner = DynamicEMAScanner()
signals = scanner.scan_all_optimized_epics()  # Uses optimal parameters per epic
scanner.print_optimization_status()  # Shows system readiness
```

### Database Schema (PostgreSQL)
```sql
-- Core optimization tables
ema_optimization_runs     -- Track optimization sessions
ema_optimization_results  -- Store parameter test results (14,406 combinations)
ema_best_parameters      -- Best configurations per epic with performance metrics

-- Enhanced with market context awareness
ALTER TABLE ema_best_parameters ADD COLUMN market_regime VARCHAR(20);
ALTER TABLE ema_best_parameters ADD COLUMN session_preference VARCHAR(50);
ALTER TABLE ema_best_parameters ADD COLUMN volatility_range VARCHAR(20);
```

### Key Features

#### Market Context Awareness
```python
conditions = MarketConditions(
    volatility_level='high',    # Adjusts SL/TP for volatility
    market_regime='trending',   # Different params for trending vs ranging
    session='london',          # Session-specific optimization
    news_impact='high'         # News-aware adjustments
)
params = service.get_epic_parameters(epic, conditions)
```

#### Performance Tracking & Auto-Suggestions
```python
# System monitors performance and suggests updates
suggestions = service.suggest_parameter_updates('CS.D.EURUSD.CEEM.IP')
# Returns: needs_update, performance_improvement, suggested_config, reason
```

#### Comprehensive Parameter Grid (14,406 combinations per epic)
```python
optimization_grid = {
    'ema_configs': ['default', 'aggressive', 'conservative', 'scalping', 'swing', 'news_safe', 'crypto'],
    'confidence_levels': [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    'timeframes': ['5m', '15m', '1h'],
    'smart_money_options': [True, False],
    'stop_loss_levels': [5, 8, 10, 12, 15, 20, 25],
    'take_profit_levels': [10, 15, 20, 25, 30, 40, 50],
    'risk_reward_ratios': [1.5, 2.0, 2.5, 3.0]
}
```

### Production Benefits

1. **Intelligent Automation**: No manual config updates - parameters from optimization results
2. **Epic-specific Optimization**: Each pair uses individually optimized settings
3. **Performance-driven Decisions**: Parameters chosen based on actual backtesting results
4. **Market Context Adaptation**: Parameters adjust for volatility and market conditions
5. **Continuous Improvement**: Built-in tracking and suggestions for re-optimization
6. **Graceful Fallbacks**: Handles missing optimization data intelligently

### Migration from Static to Dynamic Parameters

#### Old Approach (Manual Config Files)
```python
# Static configuration in config files
EMA_STRATEGY_CONFIG = {
    'default': {'short': 21, 'long': 50, 'trend': 200},
    'aggressive': {'short': 12, 'long': 26, 'trend': 200},
    # Manual updates needed for each epic
}
```

#### New Approach (Database-driven Optimization)
```python
# Automatic optimal parameter retrieval
strategy = EMAStrategy(epic='CS.D.EURUSD.CEEM.IP', use_optimal_parameters=True)
# Automatically gets: optimal EMA config, 55% confidence, 10/20 SL/TP, 1.575 performance score
```

### Testing & Validation
Comprehensive test suite validates all components:
```bash
docker exec task-worker python forex_scanner/optimization/test_dynamic_integration.py
# Tests: Parameter service, dynamic strategy, scanner integration, performance tracking
```

### Implementation Status: ✅ PRODUCTION READY
- All 5 test suites passed (5/5)
- Caching system functional
- Fallback mechanisms working
- Market context integration complete
- Performance tracking operational

## Market Intelligence System

### Overview
The Market Intelligence system provides comprehensive market analysis and trade context evaluation in the Streamlit TradingView chart interface. It works independently of the forex_scanner module, using simplified analysis algorithms when the full MarketIntelligenceEngine is not available.

### Architecture

#### Components
1. **StreamlitDataFetcher** (`streamlit/services/data_fetcher_adapter.py`)
   - Adapts Streamlit's database connection for intelligence analysis
   - Maps timeframes correctly (5m=5, 15m=5, 1h=60 as integers)
   - Enhances data with technical indicators (EMAs, ATR, volatility)
   - SQL queries optimized for ig_candles table structure

2. **MarketIntelligenceService** (`streamlit/services/market_intelligence_service.py`)
   - Main service wrapper for market intelligence
   - Provides both full and simplified analysis modes
   - 5-minute caching mechanism for performance
   - Graceful fallback when forex_scanner module unavailable

3. **TradingView Chart Integration** (`streamlit/pages/tvchart.py`)
   - Sidebar controls for intelligence features
   - Session and regime visualization above chart
   - Comprehensive trade context analysis with tabs
   - Proper handling of simplified data structures

### Features

#### Market Regime Analysis
- **Detection Methods**: EMA-based trend analysis (21/50/200)
- **Regime Types**: trending_up, trending_down, ranging, breakout
- **Confidence Scoring**: 0.6-0.7 based on indicator alignment
- **Volatility Calculation**: ATR-based with percentile ranking

#### Session Analysis
- **Sessions**: Asian (22:00-08:00 UTC), London (08:00-16:00 UTC), New York (13:00-22:00 UTC)
- **Overlap Detection**: London/New York (13:00-16:00 UTC)
- **Volatility Mapping**:
  - Peak: London/New York overlap
  - High: London or New York single session
  - Low: Asian session
  - Minimal: Off hours
- **Characteristics**: Session-specific trading behaviors and active pairs

#### Trade Context Analysis
- **Quality Scoring**: Combined regime (60%) and session (40%) alignment
- **Regime Alignment**: 
  - BUY: Good for trending_up (80% score), breakout (70% score)
  - SELL: Good for trending_down (80% score), breakout (70% score)
  - Ranging: Neutral (40% score)
- **Session Scoring**:
  - London/NY Overlap: 90%
  - London or NY: 70%
  - Asian: 50%
  - Off hours: 20%
- **Success Factors**: Context-aware list of positive trade aspects
- **Improvement Suggestions**: Actionable recommendations for better trades

### Database Schema
The system expects the following ig_candles table structure:
```sql
- start_time (timestamp)
- epic (varchar)
- timeframe (integer: 5, 60)
- open, high, low, close (numeric)
- volume, ltv (numeric)
```

### Usage in Streamlit

#### Enable Features
```python
# In sidebar
show_market_regime = st.sidebar.checkbox("Show Market Regime", value=False)
show_session_analysis = st.sidebar.checkbox("Show Session Info", value=True)
show_trade_context = st.sidebar.checkbox("Analyze Trade Context", value=False)
```

#### Get Intelligence Service
```python
from services.market_intelligence_service import get_intelligence_service

intelligence_service = get_intelligence_service()
intelligence_service.data_fetcher.engine = engine  # Pass DB engine

# Get analyses
regime_data = intelligence_service.get_regime_for_timeframe(epic, '15m')
session_data = intelligence_service.get_session_analysis()
trade_context = intelligence_service.analyze_trade_context(trade_data, epic)
```

### Simplified Analysis Mode
When forex_scanner module is not available (Streamlit container), the system automatically uses simplified analysis:

1. **Regime Detection**: Basic EMA crossover analysis
2. **Session Detection**: UTC time-based without market data
3. **Trade Context**: Rule-based scoring without ML
4. **Volatility**: Rolling standard deviation instead of complex indicators

### Display Components

#### Session Bar
Shows current trading session with volatility assessment:
- Format: `[emoji] SESSION NAME | Volatility Level | Characteristic`
- Example: `🟢 LONDON SESSION | High Volatility | High volatility`

#### Regime Display
Shows dominant market regime with confidence:
- Format: `[emoji] REGIME | Confidence% | Description`
- Example: `📈 TRENDING UP | 70% confidence | Strong upward momentum`

#### Trade Context Tabs
1. **Individual Trades**: Trade-by-trade analysis with scores
2. **Regime Performance**: Aggregated performance by market regime
3. **Trade Summary**: Overall statistics and recommendations

### Performance Considerations
- 5-minute cache duration for all analyses
- Limit to 5 pairs for batch regime analysis
- 48-hour default lookback for data fetching
- Fallback to 5m data when 15m unavailable

### Common Issues and Solutions

1. **"Unknown Volatility" in session display**
   - Solution: Check session_characteristics field instead of session_config
   - Fixed in tvchart.py lines 897-932

2. **SQL column errors**
   - Issue: Column names differ from expected (open vs open_price_mid)
   - Solution: Use correct column names in data_fetcher_adapter.py

3. **Timeframe type errors**
   - Issue: Database stores timeframes as integers (5, 60) not strings ('5m', '1h')
   - Solution: Map strings to integers in data_fetcher_adapter.py

4. **Missing fields in trade context**
   - Issue: UI expects specific fields (entry_price, regime_alignment.score, etc.)
   - Solution: Ensure all fields provided in _get_simplified_trade_context()

### Testing
```bash
# Restart Streamlit to load changes
docker-compose restart streamlit

# Check logs for errors
docker-compose logs --tail=30 streamlit | grep -E "(Error|error|❌|⚠️)"

# Clear Python cache if needed
docker-compose exec -T streamlit find /app -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
```