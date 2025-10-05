# 🚀 Forex Scanner - EMA Signal Detection System

An automated forex trading signal detection system based on EMA (Exponential Moving Average) crossover strategy with AI-powered analysis through Claude API integration.

## ✨ Features

- **🎯 EMA Crossover Strategy**: 9, 21, 200 period EMAs for trend identification
- **📊 BID/MID Price Handling**: Automatic adjustment for accurate signal detection
- **🤖 AI Analysis**: Claude API integration for intelligent signal analysis
- **📈 Multi-timeframe Analysis**: 1m, 5m, 15m, 1h, 4h, daily trend alignment
- **📊 Volume Confirmation**: Volume ratio analysis for signal strength
- **🔍 Support/Resistance**: Dynamic S/R level detection
- **📱 Real-time Alerts**: Console, file, and API notifications
- **🔄 Backtesting**: Historical signal analysis and performance metrics
- **🛡️ Risk Management**: Position sizing and risk assessment

## 🏗️ Architecture

```
forex_scanner/
├── main.py                 # Entry point
├── config.py               # Configuration
├── core/                   # Core functionality
│   ├── scanner.py          # Main scanner class
│   ├── signal_detector.py  # Signal detection logic
│   ├── database.py         # Database management
│   └── data_fetcher.py     # Data fetching
├── analysis/               # Technical analysis modules
│   ├── technical.py        # EMA, S/R, indicators
│   ├── volume.py           # Volume analysis
│   ├── behavior.py         # Market behavior
│   └── multi_timeframe.py  # Multi-TF analysis
├── alerts/                 # Alert system
│   ├── claude_api.py       # Claude integration
│   ├── notifications.py    # Alert management
│   └── order_manager.py    # Order execution
└── utils/                  # Utilities
    ├── helpers.py          # Helper functions
    └── timezone_utils.py   # Timezone handling
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository>
cd forex_scanner
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file or modify `config.py`:

```python
# Database
DATABASE_URL = "postgresql://user:pass@localhost/forex_db"

# API Keys
CLAUDE_API_KEY = "your_anthropic_api_key"

# Trading pairs
EPIC_LIST = [
    'CS.D.EURUSD.CEEM.IP',
    'CS.D.GBPUSD.MINI.IP',
    'CS.D.USDJPY.MINI.IP'
]

# Scanner settings
SCAN_INTERVAL = 60  # seconds
SPREAD_PIPS = 1.5
MIN_CONFIDENCE = 0.6
```

### 3. Usage

#### Check Configuration
```bash
python main.py scan --config-check
```

#### Single Scan
```bash
python main.py scan
```

#### Continuous Scanning
```bash
python main.py live
```

#### Backtesting
```bash
python main.py backtest --days 30
python main.py backtest --epic CS.D.EURUSD.CEEM.IP --days 7
python main.py debug --epic CS.D.EURUSD.CEEM.IP --timestamp "2025-06-26 09:40"
python main.py claude-timestamp --epic CS.D.EURUSD.CEEM.IP --timestamp "2025-06-27 08:20"
```

#### Test Claude Integration
```bash
python main.py test-claude
```

## 📊 Signal Criteria

### BULL Signal Requirements
- ✅ Price above EMA 9
- ✅ EMA 9 above EMA 21  
- ✅ EMA 9 above EMA 200
- ✅ EMA 21 above EMA 200
- ✅ New crossover (price crosses above EMA 9)

### BEAR Signal Requirements
- ✅ Price below EMA 9
- ✅ EMA 21 above EMA 9
- ✅ EMA 200 above EMA 9  
- ✅ EMA 200 above EMA 21
- ✅ New crossover (price crosses below EMA 9)

### Confidence Scoring
```
Confidence = Base(0.5) + EMA_Separation_Bonus + Volume_Bonus
- EMA Separation: Up to 30% bonus based on EMA distance
- Volume Bonus: Up to 15% bonus for above-average volume  
- Maximum confidence: 95%
```

## 🔧 Advanced Usage

### Programmatic Usage

```python
from core.scanner import ForexScanner
from core.database import DatabaseManager

# Initialize scanner
db_manager = DatabaseManager("your_db_url")
scanner = ForexScanner(
    db_manager=db_manager,
    epic_list=['CS.D.EURUSD.CEEM.IP'],
    claude_api_key="your_key"
)

# Single scan
signals = scanner.scan