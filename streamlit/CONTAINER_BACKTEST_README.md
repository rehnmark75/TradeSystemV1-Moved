# Container-Aware Backtest System

A unified backtesting platform designed to work within the containerized TradeSystemV1 architecture.

## 🎯 Overview

This system provides a modular backtest framework that works entirely within the Streamlit container, using database connections to access historical data and trading signals without requiring access to the worker container's code.

## 🏗️ Architecture

### Container Design
```
┌─────────────────────┐    ┌─────────────────────┐
│   Streamlit         │    │     Worker          │
│   Container         │    │     Container       │
│                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ Backtest System │ │    │ │ Live Trading    │ │
│ │                 │ │    │ │ Strategies      │ │
│ │ - Historical    │ │◄──►│ │                 │ │
│ │   Analysis      │ │    │ │ - EMA           │ │
│ │ - Simple        │ │    │ │ - MACD          │ │
│ │   Strategies    │ │    │ │ - Combined      │ │
│ │ - Chart Viewer  │ │    │ └─────────────────┘ │
│ └─────────────────┘ │    └─────────────────────┘
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   PostgreSQL        │
│   Database          │
│                     │
│ - Candle Data       │
│ - Trade Logs        │
│ - Alert History     │
└─────────────────────┘
```

## 🚀 Quick Start

### 1. Access the System
Navigate to the Streamlit app and look for:
- **Main Interface**: `pages/backtest_runner.py`
- **Test Interface**: `pages/test_backtest_system.py`

### 2. Available Strategies

The system includes three built-in strategies that work with database data:

#### 📊 Historical Signal Analysis
- **Description**: Analyzes actual trading signals from the database
- **Data Source**: `trade_log` and `alert_history` tables
- **Use Case**: Review performance of historical trades

#### 📈 Simple Moving Average Crossover
- **Description**: Basic MA crossover strategy using price data
- **Data Source**: `ig_candles` table
- **Use Case**: Test classic technical analysis approach

#### 🚀 Price Breakout Strategy
- **Description**: Detects breakouts from recent high/low levels
- **Data Source**: `ig_candles` table
- **Use Case**: Identify momentum trading opportunities

### 3. Running a Backtest

1. **Select Strategy**: Choose from the dropdown
2. **Configure Parameters**: Set epic, timeframe, days, and strategy-specific settings
3. **Execute**: Click "Run Backtest"
4. **Analyze Results**: View charts, signals table, and performance metrics

## 📊 Features

### Interactive Charts
- **TradingView Integration**: Same `streamlit_lightweight_charts_ntf` as main tvchart
- **Signal Markers**: Color-coded based on profitability
- **Multiple Timeframes**: 5m, 15m, 1h support
- **Indicator Overlays**: Moving averages, breakout levels

### Results Table
- **Comprehensive Filtering**: Direction, type, profitability, dates
- **Real-time Search**: Find specific signals quickly
- **Export Options**: CSV, Excel, JSON formats
- **Performance Analytics**: Win rate, average profit, risk metrics

### Export Capabilities
- **Signal Data**: CSV and Excel exports
- **Performance Reports**: JSON and text formats
- **Complete Packages**: ZIP files with all data
- **Configuration Files**: Reproduce backtests

## 🔧 Configuration

### Environment Variables
```bash
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/forex
```

### Strategy Parameters

Each strategy has configurable parameters:

```python
# Historical Signal Analysis
{
    'min_confidence': 0.7,  # Filter signals by confidence
    'days': 7,              # Lookback period
    'epic': 'CS.D.EURUSD.MINI.IP'
}

# MA Crossover
{
    'fast_ma': 10,          # Fast moving average period
    'slow_ma': 20,          # Slow moving average period
    'days': 7
}

# Price Breakout
{
    'lookback_periods': 20,      # Periods for high/low calculation
    'breakout_threshold': 0.5,   # Minimum breakout in pips
    'days': 7
}
```

## 📈 Data Sources

### 1. Candle Data (`ig_candles`)
```sql
SELECT start_time, open, high, low, close
FROM ig_candles
WHERE timeframe = :tf AND epic = :epic
ORDER BY start_time DESC
```

### 2. Trade Logs (`trade_log`)
```sql
SELECT timestamp, entry_price, direction, profit_loss, status
FROM trade_log
WHERE symbol = :epic AND timestamp >= :min_time
```

### 3. Alert History (`alert_history`)
```sql
SELECT strategy, signal_type, confidence_score
FROM alert_history
WHERE epic = :epic
```

## 🧪 Testing

### Test Page
Access `pages/test_backtest_system.py` to:
1. **Verify Imports**: Check all modules load correctly
2. **Test Database**: Confirm connection to PostgreSQL
3. **Run Sample Backtests**: Execute each strategy
4. **View Results**: See signals and performance data

### Manual Testing
```python
# In Streamlit container
from services.container_backtest_service import get_container_backtest_service, BacktestConfig

service = get_container_backtest_service()
config = BacktestConfig(
    strategy_name='historical_signals',
    epic='CS.D.EURUSD.MINI.IP',
    days=7
)
result = service.run_backtest(config)
```

## 🔄 Adding New Strategies

### Step 1: Define Strategy Logic
```python
def _run_my_custom_backtest(self, config: BacktestConfig) -> BacktestResult:
    # Get data from database
    chart_data = get_candle_data(self.engine, tf_minutes, config.epic, limit=days*24*4)

    # Apply your logic
    signals = []
    for i, row in chart_data.iterrows():
        if your_condition(row):
            signal = {
                'timestamp': row['start_time'],
                'direction': 'BUY' or 'SELL',
                'entry_price': row['close'],
                # ... other fields
            }
            signals.append(signal)

    # Return results
    return BacktestResult(...)
```

### Step 2: Register Strategy
```python
# In get_available_strategies()
'my_strategy': StrategyInfo(
    name='my_strategy',
    display_name='My Custom Strategy',
    description='Description of what it does',
    parameters={
        'param1': {
            "type": "number",
            "default": 10,
            "description": "Parameter description"
        }
    }
)
```

### Step 3: Add Execution
```python
# In run_backtest()
elif config.strategy_name == 'my_strategy':
    result = self._run_my_custom_backtest(config)
```

## 📊 Performance Metrics

### Standard Metrics
- **Total Signals**: Number of signals generated
- **Win Rate**: Percentage of profitable signals
- **Average Confidence**: Mean confidence score
- **Average Profit**: Mean profit in pips
- **Risk/Reward Ratio**: Average R:R across signals

### Signal Fields
Each signal includes:
```python
{
    'timestamp': datetime,
    'epic': str,
    'signal_type': str,
    'direction': 'BUY'|'SELL',
    'entry_price': float,
    'confidence': float,
    'strategy': str,
    'timeframe': str,
    'max_profit_pips': float,
    'max_loss_pips': float,
    'profit_loss_ratio': float
}
```

## 🚨 Troubleshooting

### Common Issues

**No Database Connection**
```python
# Check environment variable
import os
print(os.getenv('DATABASE_URL'))

# Test connection manually
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
```

**No Data Available**
```python
# Check if data exists
from services.data import get_candle_data
df = get_candle_data(engine, 15, 'CS.D.EURUSD.MINI.IP', limit=100)
print(f"Found {len(df)} candles")
```

**Import Errors**
- Ensure running within Streamlit container
- Check that all required packages are installed in container
- Verify file paths are correct

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
service = get_container_backtest_service()
result = service.run_backtest(config)
```

## 🎯 Limitations & Design Decisions

### Container Constraints
- **No Worker Code Access**: Can't import worker strategies directly
- **Database Only**: Relies on historical data in PostgreSQL
- **Simplified Strategies**: Uses basic technical analysis
- **Performance Estimation**: Looks ahead in historical data

### Benefits
- **Container Native**: Works within existing architecture
- **Real Data**: Uses actual trading data from database
- **Fast Execution**: No complex external dependencies
- **Easy Extension**: Simple to add new strategies

## 🔮 Future Enhancements

### Planned Features
- **API Bridge**: Connect to worker container for live strategy access
- **Real-time Data**: Stream live price updates
- **Advanced Indicators**: More technical analysis tools
- **Portfolio Backtests**: Multi-strategy combinations
- **Risk Analysis**: VaR, drawdown calculations

### Extension Points
- **Custom Indicators**: Plugin system for new indicators
- **Data Sources**: Additional market data providers
- **Export Formats**: More output options
- **Visualization**: Enhanced chart features

## 📞 Support

### Debugging Steps
1. **Check Test Page**: Visit `pages/test_backtest_system.py`
2. **Verify Database**: Ensure PostgreSQL is accessible
3. **Check Logs**: Review Streamlit console output
4. **Test Individual Components**: Use test functions

### Getting Help
- Check the test page for system status
- Review database connection settings
- Verify container environment variables
- Test with simple strategies first

---

**Generated by**: Container-Aware Backtest System v1.0
**Last Updated**: 2024-01-20
**Architecture**: Containerized, Database-Driven
**Status**: Production Ready ✅