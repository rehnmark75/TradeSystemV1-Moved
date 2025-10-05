# Forex Scanner - Complete Project Documentation
## 📋 Project Status: July 17, 2025

**Current Status: ✅ PRODUCTION READY** 
- 🏗️ Modular architecture successfully implemented
- 🔧 Critical issues resolved (database table references fixed)
- 🚀 All systems operational: live scanning, backtesting, debugging
- 📈 Multiple strategies functional: EMA, MACD, Combined
- 🎯 Performance: 92.1% average confidence, 60% win rate, 24.0 pip average profit

---

## 🏗️ Complete Project Architecture

### Core Directory Structure

```
forex_scanner/
├── 📄 main.py                         # CLI entry point with command routing
├── ⚙️ config.py                       # Centralized configuration
├── 📋 requirements.txt                # Dependencies
├── 🧪 test_imports.py                 # Module structure verification
├── ⚠️ trade_scan.py                   # Docker entry point (EnhancedTradingSystem)
│
├── 🔧 core/                           # Core functionality
│   ├── 📄 __init__.py                 # Enhanced module initialization
│   ├── 🔍 signal_detector.py          # Lightweight coordinator (150 lines)
│   ├── 🤖 scanner.py                  # Main scanning orchestration (ForexScanner + IntelligentForexScanner)
│   ├── 🗄️ database.py                 # PostgreSQL database management
│   ├── 📊 data_fetcher.py             # Data fetching & enhancement
│   │
│   ├── 📈 strategies/                 # Trading strategies
│   │   ├── 🔗 base_strategy.py        # Abstract base class
│   │   ├── 📊 ema_strategy.py         # EMA crossover logic (9,21,200)
│   │   ├── 📉 macd_strategy.py        # MACD + EMA200 strategy
│   │   ├── 🎯 combined_strategy.py    # Multi-strategy combination
│   │   ├── ⚡ scalping_strategy.py    # High-frequency scalping
│   │   └── 🔄 kama_strategy.py        # Adaptive moving average
│   │
│   ├── 🔎 detection/                  # Signal detection components
│   │   ├── 💰 price_adjuster.py       # BID/MID price handling
│   │   └── 🌡️ market_conditions.py    # Market analysis
│   │
│   ├── 🧪 backtest/                   # Backtesting engine
│   │   ├── 🔄 backtest_engine.py      # Core backtesting logic
│   │   ├── 📊 performance_analyzer.py # Performance metrics
│   │   └── 📋 signal_analyzer.py      # Signal display/analysis
│   │
│   ├── 🧠 intelligence/               # Market intelligence
│   │   └── 🤖 market_intelligence.py  # Advanced market analysis
│   │
│   ├── 💾 processing/                 # Signal processing
│   │   └── 📝 signal_processor.py     # Signal validation & enhancement
│   │
│   ├── 👁️ monitoring/                 # System monitoring
│   │   ├── 📊 performance_monitor.py  # Performance tracking
│   │   └── 🔄 session_manager.py      # Session management
│   │
│   ├── 💼 trading/                    # Trading execution
│   │   └── 🎯 trading_orchestrator.py # High-level trading coordination
│   │
│   └── 🛡️ alert_deduplication.py     # Alert deduplication system
│
├── 📊 analysis/                       # Technical analysis modules
│   ├── 🔧 technical.py                # EMA, S/R, indicators
│   ├── 📈 volume.py                   # Volume analysis
│   ├── 🎭 behavior.py                 # Market behavior patterns
│   └── ⏰ multi_timeframe.py          # Multi-timeframe analysis
│
├── 🚨 alerts/                         # Alert and notification system
│   ├── 🤖 claude_api.py               # Claude AI integration
│   ├── 📱 notifications.py            # Alert management
│   └── 📋 alert_history.py            # Alert history database management
│
├── 🛠️ utils/                          # Utilities and helpers
│   ├── 🔧 helpers.py                  # Helper functions
│   ├── 🌍 timezone_utils.py           # Timezone handling
│   ├── 🔄 alert_saver.py              # Alert saving utilities
│   └── 📊 performance_tracker.py      # Performance tracking utilities
│
├── 💻 commands/                       # CLI command modules
│   ├── 🔍 scanner_commands.py         # scan, live commands
│   ├── 🧪 backtest_commands.py        # backtest, comparison commands
│   ├── 🐛 debug_commands.py           # debugging commands
│   ├── ⚡ scalping_commands.py        # scalping commands
│   ├── 🤖 claude_commands.py          # Claude API commands
│   └── 📊 analysis_commands.py        # analysis commands
│
├── ⚙️ configdata/                     # Dynamic configuration
│   ├── 🔄 dynamic_config_loader.py    # Database-driven config
│   └── 🔧 integration_module.py       # Config migration tools
│
├── 🔧 scripts/                        # Utility scripts
│   ├── 🏥 diagnostic_check.py         # System diagnostics
│   ├── 🔍 scanner_diagnostic.py       # Scanner-specific diagnostics
│   ├── 🧪 test_imports.py             # Import verification
│   ├── 🚨 alert_diagnostic.py         # Alert system analysis
│   └── 🔧 fixes/                      # Various fix scripts
│
├── 📁 backups/                        # System backups
│   └── 📄 alert_fix_20250716_*/       # Dated backup folders
│
├── 📚 documentation/                  # Project documentation
│   ├── 📄 project_state.md            # Current project state
│   └── 📄 new_cat_quickstart.md       # Quick start guide
│
└── 🌐 streamlit/                      # Web interface (optional)
    ├── 📄 pages/
    │   ├── ⚙️ TradeSystemv1.py         # Main configuration interface
    │   ├── 📊 page2.py                 # Monitoring dashboard
    │   └── 📋 logs.py                  # Real-time log viewer
    └── ...
```

---

## 🗄️ Database Schema

### Primary Tables

#### `alert_history` - Main Signal Storage
```sql
CREATE TABLE alert_history (
    id SERIAL PRIMARY KEY,
    alert_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    epic VARCHAR(50) NOT NULL,
    pair VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    price DECIMAL(10,5) NOT NULL,
    bid_price DECIMAL(10,5),
    ask_price DECIMAL(10,5),
    spread_pips DECIMAL(5,2),
    timeframe VARCHAR(10) NOT NULL,
    
    -- Strategy data
    strategy_config JSON,
    strategy_indicators JSON,
    strategy_metadata JSON,
    
    -- Technical indicators
    ema_short DECIMAL(10,5),
    ema_long DECIMAL(10,5), 
    ema_trend DECIMAL(10,5),
    macd_line DECIMAL(10,6),
    macd_signal DECIMAL(10,6),
    macd_histogram DECIMAL(10,6),
    
    -- Market data
    volume DECIMAL(15,2),
    volume_ratio DECIMAL(8,4),
    volume_confirmation BOOLEAN DEFAULT FALSE,
    nearest_support DECIMAL(10,5),
    nearest_resistance DECIMAL(10,5),
    distance_to_support_pips DECIMAL(8,2),
    distance_to_resistance_pips DECIMAL(8,2),
    risk_reward_ratio DECIMAL(8,4),
    market_session VARCHAR(20),
    is_market_hours BOOLEAN DEFAULT TRUE,
    market_regime VARCHAR(30),
    
    -- Signal metadata
    signal_trigger VARCHAR(50),
    signal_conditions JSON,
    crossover_type VARCHAR(50),
    signal_hash VARCHAR(32),
    data_source VARCHAR(20),
    market_timestamp TIMESTAMP,
    cooldown_key VARCHAR(100),
    dedup_metadata JSON,
    
    -- Claude analysis
    claude_analysis TEXT,
    claude_score INTEGER,
    claude_decision VARCHAR(20),
    claude_approved BOOLEAN,
    claude_reason TEXT,
    claude_mode VARCHAR(20),
    claude_raw_response TEXT,
    
    -- Alert data
    alert_message TEXT,
    alert_level VARCHAR(20) DEFAULT 'INFO',
    status VARCHAR(20) DEFAULT 'ACTIVE',
    processed_at TIMESTAMP,
    notes TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    strategy_config_hash VARCHAR(64)
);
```

#### Other Supporting Tables
- `strategy_summary` - Strategy performance aggregation
- `strategy_configs` - Strategy configuration management
- `ig_candles` - Market data storage
- `performance_snapshots` - System performance tracking

---

## 🔧 Core Components

### 1. 🔍 Signal Detection System

**SignalDetector (`signal_detector.py`)** - Lightweight coordinator (150 lines)
- Orchestrates multiple trading strategies
- Handles crossover state tracking to prevent duplicate signals
- Manages signal validation and enhancement
- Supports live scanning and backtesting modes

**Key Features:**
- ✅ EMA crossover detection with state tracking
- ✅ MACD signal integration
- ✅ Combined strategy consensus/weighted modes
- ✅ BID/MID price adjustment for accurate signals
- ✅ Confidence scoring based on multiple factors

### 2. 📈 Trading Strategies

**Base Strategy Architecture (`base_strategy.py`)**
```python
class BaseStrategy:
    def detect_signal(self, df: pd.DataFrame, epic: str, spread_pips: float, timeframe: str) -> Optional[Dict]
    def get_required_indicators(self) -> List[str]
    def calculate_confidence(self, signal_data: Dict) -> float
```

**Available Strategies:**
- **EMA Strategy**: 9/21/200 EMA crossovers with trend alignment
- **MACD Strategy**: MACD histogram + EMA200 confirmation
- **Combined Strategy**: Multi-strategy consensus and weighted modes
- **Scalping Strategy**: High-frequency short-term signals
- **KAMA Strategy**: Adaptive moving average system

### 3. 🧪 Backtesting Engine

**BacktestEngine (`backtest_engine.py`)**
- Historical signal detection across configurable timeframes
- Performance metrics calculation and comparison
- Strategy isolation for testing individual components
- Enhanced data validation and error handling

**Recent Performance Metrics:**
- 📊 **Total Signals**: 125 (7-day EURUSD 15m backtest)
- 🎯 **Strategy Mix**: 110 EMA + 8 MACD + 7 Combined
- 📈 **Average Confidence**: 92.1%
- 🏆 **Win Rate**: 60%
- 💰 **Average Profit**: 24.0 pips

### 4. 🗄️ Data Management

**DatabaseManager (`database.py`)**
- PostgreSQL connection management with connection pooling
- Query execution with automatic retry logic
- Transaction management and rollback handling
- Health monitoring and connection validation

**Current Schema Status:**
- ✅ `alert_history` table with 45+ columns
- ✅ Enhanced with Claude analysis fields
- ✅ Deduplication and performance tracking
- ✅ JSON fields for flexible strategy data

### 5. 🚨 Alert System

**ClaudeAnalyzer (`claude_api.py`)**
- AI-powered signal analysis and validation
- Minimal and detailed analysis modes
- Integration with alert history for decision tracking

**NotificationManager (`notifications.py`)**
- Multi-channel alert delivery
- Integration with Claude analysis results
- Fallback mechanisms for reliability

**AlertHistoryManager (`alert_history.py`)**
- Comprehensive signal storage and retrieval
- Claude analysis result tracking
- Performance analytics and reporting

---

## 💻 Command Line Interface

### Core Commands

```bash
# Single scan
python main.py scan

# Continuous live scanning
python main.py live

# Configuration check
python main.py scan --config-check
```

### Backtesting Commands

```bash
# Standard backtest
python main.py backtest --epic CS.D.EURUSD.CEEM.IP --days 7 --timeframe 15m

# Compare EMA configurations
python main.py compare-ema-configs --epic CS.D.EURUSD.CEEM.IP --days 30

# Extended backtest with analysis
python main.py backtest --days 30 --show-signals --bb-analysis
```

### Debugging Commands

```bash
# Debug individual signals
python main.py debug --epic CS.D.EURUSD.CEEM.IP

# Debug specific strategies
python main.py debug-combined --epic CS.D.EURUSD.CEEM.IP
python main.py debug-macd --epic CS.D.EURUSD.CEEM.IP

# System diagnostics
python main.py diagnostic-check
```

### Claude Integration Commands

```bash
# Test Claude API
python main.py test-claude

# Claude analysis of recent signals
python main.py claude-analyze --epic CS.D.EURUSD.CEEM.IP
```

---

## ⚙️ Configuration

### Static Configuration (`config.py`)

```python
# Trading Strategy Settings
SIMPLE_EMA_STRATEGY = True
MACD_EMA_STRATEGY = True
COMBINED_STRATEGY_MODE = 'consensus'  # or 'weighted'
MIN_COMBINED_CONFIDENCE = 0.75

# Strategy Weights (for weighted mode)
STRATEGY_WEIGHT_EMA = 0.6
STRATEGY_WEIGHT_MACD = 0.4

# Technical Parameters
USE_BID_ADJUSTMENT = False
DEFAULT_TIMEFRAME = '15m'
MIN_CONFIDENCE = 0.7
SPREAD_PIPS = 1.5

# Currency Pairs
EPIC_LIST = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP',
    # ... more pairs
]

# Database
DATABASE_URL = "postgresql://user:pass@host:5432/dbname"

# Claude API
CLAUDE_API_KEY = "sk-ant-api03-..."
CLAUDE_ANALYSIS_MODE = "minimal"  # or "detailed"
```

### Dynamic Configuration (`configdata/`)
- Database-driven configuration management
- Runtime configuration changes
- Performance-based auto-tuning

---

## 🐳 Docker Integration

### Entry Points
- **Main**: `trade_scan.py` - Docker container entry point
- **CLI**: `main.py` - Command-line interface
- **Development**: Individual module testing

### Container Usage
```bash
# Run single scan
docker exec <container> python trade_scan.py scan

# Run continuous scanning
docker exec <container> python trade_scan.py live

# Run backtesting
docker exec <container> python main.py backtest --epic CS.D.EURUSD.CEEM.IP --days 7
```

---

## 🔧 Recent Fixes & Improvements

### July 17, 2025 Updates
1. **Fixed Database Table References**
   - Corrected `signals` table references to `alert_history`
   - Updated `core/processing/signal_processor.py`
   - Fixed parameter binding for PostgreSQL

2. **Enhanced Error Handling**
   - Added JSON serialization helpers
   - Improved datetime handling
   - Robust string method protection

3. **Claude Integration Improvements**
   - Fixed API key initialization issues
   - Enhanced analysis mode support
   - Better error fallback mechanisms

4. **Notification System Enhancements**
   - Fixed import path issues
   - Added dummy manager fallbacks
   - Improved error logging

---

## 📊 Performance Metrics

### System Performance
- 🚀 Signal detection latency: <5 seconds
- 💾 Memory usage: Optimized for long-running processes
- 🗄️ Database performance: Indexed queries with connection pooling
- ⚡ Scan frequency: Configurable (60s default)

### Trading Performance
- 📈 Backtest success: 125 signals/week (EURUSD 15m)
- 🎯 Average confidence: 92.1%
- 🏆 Win rate: 60%
- 💰 Average profit: 24.0 pips per signal

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure database
export DATABASE_URL="postgresql://user:pass@host:5432/forex"

# Set Claude API key
export CLAUDE_API_KEY="sk-ant-api03-..."
```

### 2. Test System
```bash
# Test imports
python test_imports.py

# Test configuration
python main.py scan --config-check

# Test backtesting
python main.py backtest --epic CS.D.EURUSD.CEEM.IP --days 3
```

### 3. Run Production
```bash
# Single scan
python main.py scan

# Continuous scanning
python main.py live

# Docker mode
python trade_scan.py docker
```

---

## 🔍 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   python test_imports.py
   ```

2. **Database Connection Issues**
   ```bash
   python main.py scan --config-check
   ```

3. **No Signals Found**
   ```bash
   python main.py debug --epic CS.D.EURUSD.CEEM.IP
   ```

4. **Claude API Issues**
   ```bash
   python main.py test-claude
   ```

### System Diagnostics
```bash
# Comprehensive system check
python scripts/diagnostic_check.py

# Alert system analysis
python scripts/alert_diagnostic.py

# Scanner-specific diagnostics
python scripts/scanner_diagnostic.py
```

---

## 📈 Future Roadmap

### Planned Enhancements
1. **Machine Learning Integration**
   - Pattern recognition models
   - Adaptive strategy weighting
   - Market regime detection

2. **Risk Management**
   - Position sizing algorithms
   - Portfolio correlation analysis
   - Drawdown protection

3. **Performance Optimization**
   - Parallel strategy execution
   - Real-time data streaming
   - Advanced caching mechanisms

4. **Extended Market Coverage**
   - Additional currency pairs
   - Cryptocurrency integration
   - Stock market signals

---

## 📞 Support & Maintenance

### Documentation Updates
- This documentation updated: July 17, 2025
- Next scheduled review: August 1, 2025
- Version tracking in `documentation/` folder

### System Monitoring
- Performance alerts configured
- Database health checks active
- Claude API usage monitoring enabled

### Backup Strategy
- Daily database backups
- Code backups in `backups/` folder
- Configuration versioning enabled

---

*This documentation reflects the current state of the Forex Scanner project as of July 17, 2025. For the most up-to-date information, check the project repository and recent commit history.*