# Modular Backtest System

A unified, extensible backtesting platform that consolidates all trading strategies into a single interface with advanced charting and analysis capabilities.

## 🎯 Overview

The Modular Backtest System rebuilds the individual backtest files into a common, reusable framework that:

- **Unifies Strategy Execution**: Run any strategy through the same interface
- **Provides Interactive Charts**: Uses TradingView Lightweight Charts (same as tvchart page)
- **Offers Advanced Filtering**: Interactive table with comprehensive signal analysis
- **Enables Easy Extension**: Add new strategies with minimal effort
- **Supports Multiple Export Formats**: CSV, Excel, JSON, ZIP packages

## 🏗️ Architecture

### Core Components

```
streamlit/
├── pages/
│   └── backtest_runner.py          # Main Streamlit interface
├── services/
│   ├── backtest_service.py         # Core framework (Registry, Runner, Processor)
│   ├── backtest_chart_service.py   # TradingView chart integration
│   ├── backtest_init.py            # System initialization
│   └── strategy_adapters/          # Strategy wrapper classes
│       ├── base_adapter.py         # Abstract base class
│       ├── generic_adapter.py      # Generic wrapper for any strategy
│       ├── ema_adapter.py          # EMA strategy adapter
│       └── macd_adapter.py         # MACD strategy adapter
└── components/
    ├── results_table.py            # Interactive results table
    ├── chart_sync.py               # Chart-table synchronization
    └── export_manager.py           # Export functionality
```

### Key Classes

- **BacktestRegistry**: Discovers and manages available strategies
- **BacktestRunner**: Executes backtests with progress tracking
- **BacktestResultProcessor**: Standardizes and analyzes results
- **BaseStrategyAdapter**: Abstract interface for strategy wrappers
- **BacktestChartService**: Converts results to TradingView format

## 🚀 Usage

### Running Backtests

1. **Access the Interface**
   ```
   Navigate to: streamlit/pages/backtest_runner.py
   ```

2. **Select Strategy**
   - Choose from automatically discovered strategies
   - Configure strategy-specific parameters
   - Set basic parameters (epic, timeframe, days)

3. **Execute Backtest**
   - Click "Run Backtest"
   - Monitor real-time progress
   - View results automatically

### Viewing Results

**Chart Tab**:
- TradingView-style candlestick chart
- Signal markers with color coding (green=profit, red=loss)
- Multiple indicator overlays (EMAs, MACD, etc.)

**Signals Tab**:
- Interactive table with filtering
- Search functionality
- Export capabilities

**Analysis Tab**:
- Performance metrics
- Strategy-specific insights
- Detailed breakdowns

## 🔧 Adding New Strategies

### Method 1: Create Specific Adapter

```python
# streamlit/services/strategy_adapters/your_strategy_adapter.py

from ..backtest_service import BaseStrategyAdapter, StrategyInfo, BacktestConfig, BacktestResult

class YourStrategyAdapter(BaseStrategyAdapter):
    def __init__(self):
        super().__init__("your_strategy")

    def get_strategy_info(self) -> StrategyInfo:
        return StrategyInfo(
            name="your_strategy",
            display_name="Your Strategy",
            description="Your strategy description",
            module_path="forex_scanner.backtests.backtest_your_strategy",
            class_name="YourStrategyBacktest",
            parameters={
                'custom_param': {
                    "type": "number",
                    "default": 10,
                    "description": "Custom parameter"
                }
            }
        )

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        # Your implementation here
        pass
```

### Method 2: Use Generic Adapter

The system automatically creates generic adapters for any backtest file following the pattern:
- File: `worker/app/forex_scanner/backtests/backtest_*.py`
- Class: Inherits from `BacktestBase` or has `run_backtest` method

### Method 3: Register at Runtime

```python
from services.backtest_service import get_backtest_registry

registry = get_backtest_registry()
registry.register_adapter('strategy_name', YourStrategyAdapter())
```

## 📊 Chart Integration

The system uses the same TradingView Lightweight Charts library as the main tvchart page:

### Signal Markers
- **Green**: Profitable signals
- **Red**: Loss signals
- **Yellow**: Running/breakeven signals
- **Size**: Indicates confidence/importance

### Indicator Support
- **EMAs**: Dynamic periods based on strategy config
- **MACD**: Line, signal, histogram
- **Zero Lag EMA**: Trend-based coloring
- **Custom indicators**: Easily extensible

### Chart Features
- **Zoom/Pan**: Full TradingView interaction
- **Multi-pane**: Price + indicator charts
- **Synchronized time**: All charts aligned
- **Export ready**: Chart configurations exportable

## 📈 Results Table Features

### Filtering Options
- **Direction**: BUY/SELL signals
- **Signal Type**: Strategy-specific types
- **Profitability**: Profitable/Loss/Breakeven
- **Confidence Range**: Min/max confidence
- **Date Range**: Time-based filtering
- **Text Search**: Search across all fields

### Table Columns
- **Timestamp**: Signal generation time
- **Type**: Signal type identifier
- **Direction**: BUY/SELL
- **Entry Price**: Signal entry price
- **Confidence**: Signal confidence %
- **Max Profit/Loss**: Historical performance
- **Net Profit**: Profit - Loss calculation
- **Strategy**: Originating strategy

### Interactive Features
- **Multi-select**: Select multiple signals
- **Sort**: Click column headers
- **Export**: Current filtered data
- **Statistics**: Live summary updates

## 🔄 Export Capabilities

### Signal Data
- **CSV**: Spreadsheet-compatible format
- **Excel**: Multi-sheet with formatting
- **JSON**: API-compatible format

### Performance Metrics
- **JSON**: Structured performance data
- **Text Report**: Human-readable summary

### Complete Packages
- **ZIP**: All data + configuration
- **Reproduction Scripts**: Python code to recreate

### Configuration Export
- **JSON Config**: Backtest parameters
- **Python Script**: Executable reproduction

## ⚙️ Configuration

### Strategy Parameters

Each strategy adapter defines its own parameters:

```python
parameters = {
    'epic': {
        "type": "select",
        "default": "CS.D.EURUSD.MINI.IP",
        "description": "Trading pair"
    },
    'confidence_threshold': {
        "type": "number",
        "default": 0.7,
        "min": 0.1,
        "max": 1.0,
        "description": "Minimum confidence"
    },
    'enable_smart_money': {
        "type": "boolean",
        "default": False,
        "description": "Enable smart money analysis"
    }
}
```

### Supported Parameter Types
- **select**: Dropdown with options
- **number**: Numeric input with min/max/step
- **boolean**: Checkbox
- **text**: Text input

## 🔍 Discovery System

The system automatically discovers strategies by:

1. **Scanning**: `worker/app/forex_scanner/backtests/backtest_*.py`
2. **Analyzing**: File structure and class definitions
3. **Extracting**: Parameter definitions and descriptions
4. **Registering**: Available strategies in registry

### Discovery Rules
- **Filename**: Must start with `backtest_`
- **Class**: Must have `run_backtest` method
- **Parameters**: Extracted from argparse or docstrings

## 🚨 Error Handling

### Strategy Loading
- **Import failures**: Graceful fallback to generic adapter
- **Missing dependencies**: Warning with disabled features
- **Configuration errors**: Validation with helpful messages

### Execution Errors
- **Timeout handling**: Configurable execution timeouts
- **Resource management**: Automatic cleanup
- **Progress tracking**: Real-time status updates

### Chart Rendering
- **Missing data**: Minimal chart with notifications
- **Large datasets**: Automatic data sampling
- **Browser compatibility**: Fallback options

## 🔧 Development

### Running System
```bash
# Start Streamlit
streamlit run streamlit/pages/backtest_runner.py

# Or from main app
streamlit run streamlit/streamlit_app.py
# Then navigate to "Backtest Runner" page
```

### Testing New Adapters
```python
# Test adapter registration
from services.backtest_init import initialize_backtest_system, get_system_status

initialize_backtest_system()
status = get_system_status()
print(f"Available strategies: {status['available_strategies']}")

# Test backtest execution
from services.backtest_service import BacktestConfig, get_backtest_runner

config = BacktestConfig(
    strategy_name="ema",
    epic="CS.D.EURUSD.MINI.IP",
    days=7,
    timeframe="15m"
)

runner = get_backtest_runner()
result = runner.run_backtest(config)
print(f"Success: {result.success}, Signals: {result.total_signals}")
```

### Adding Dependencies
```python
# Strategy-specific imports in adapters
try:
    from forex_scanner.backtests.backtest_your_strategy import YourStrategy
except ImportError:
    # Handle gracefully
    YourStrategy = None
```

## 📝 API Reference

### Core Classes

**BacktestConfig**
```python
@dataclass
class BacktestConfig:
    strategy_name: str
    epic: str = "CS.D.EURUSD.MINI.IP"
    days: int = 7
    timeframe: str = "15m"
    parameters: Dict[str, Any] = field(default_factory=dict)
```

**BacktestResult**
```python
@dataclass
class BacktestResult:
    strategy_name: str
    epic: str
    timeframe: str
    total_signals: int
    signals: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    chart_data: Optional[pd.DataFrame] = None
```

### Service Functions

```python
# Get registry instance
registry = get_backtest_registry()

# Get runner instance
runner = get_backtest_runner()

# Get chart service
chart_service = get_chart_service()

# Get result processor
processor = get_result_processor()
```

## 🎯 Future Enhancements

### Planned Features
- **Real-time backtests**: Live data integration
- **Strategy comparison**: Side-by-side analysis
- **Parameter optimization**: Automated parameter tuning
- **Portfolio backtests**: Multi-strategy portfolios
- **Risk analysis**: VaR, drawdown analysis
- **Machine learning**: AI-enhanced strategies

### Extension Points
- **Custom indicators**: Plugin architecture
- **Export formats**: Additional output formats
- **Chart libraries**: Alternative chart engines
- **Data sources**: Multiple data providers
- **Execution engines**: Distributed processing

## 🆘 Troubleshooting

### Common Issues

**Strategy Not Found**
```python
# Check if strategy is discovered
registry = get_backtest_registry()
print(f"Available: {registry.list_strategies()}")

# Force rediscovery
registry.discover_strategies()
```

**Import Errors**
```python
# Check Python path
import sys
print(sys.path)

# Add worker path manually
sys.path.insert(0, '/path/to/worker/app')
```

**Chart Not Rendering**
```python
# Check if charts library available
try:
    from streamlit_lightweight_charts_ntf import renderLightweightCharts
    print("✅ Charts available")
except ImportError:
    print("❌ Charts not available")
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
result = runner.run_backtest(config)
```

## 📞 Support

For issues, feature requests, or questions:

1. **Check logs**: Review Streamlit console output
2. **Verify setup**: Ensure all dependencies installed
3. **Test individually**: Try strategy adapters separately
4. **Check configuration**: Validate parameter formats

---

**Generated by**: Modular Backtest System v1.0
**Last Updated**: 2024-01-20
**Documentation Version**: 1.0.0