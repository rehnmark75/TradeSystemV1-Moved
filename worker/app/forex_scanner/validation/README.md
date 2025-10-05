# Signal Validation and Replay System

A comprehensive signal validation and replay system for validating historical trading signals. This system allows you to recreate exact scanner conditions at any historical timestamp and validate why alerts were generated.

## Overview

The validation system provides:
- **Historical Data Replay**: Recreate market conditions as they existed at specific timestamps
- **Scanner State Recreation**: Restore scanner configurations, strategy parameters, and system settings
- **Signal Detection Replay**: Re-run the complete signal detection pipeline
- **Comprehensive Reporting**: Detailed validation reports with market analysis and decision paths
- **Batch Processing**: Validate multiple signals across different epics and timeframes
- **Time Series Analysis**: Analyze signal patterns across time ranges

## Quick Start

### Basic Usage

```bash
# Validate a single signal
python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "2025-01-15 14:30:00" \
    --epic "CS.D.EURUSD.CEEM.IP" \
    --show-calculations

# Validate all epics at a timestamp
python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "2025-01-15 14:30:00" \
    --all-epics \
    --compare-with-stored

# Debug specific strategy
python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "2025-01-15 14:30:00" \
    --epic "CS.D.EURUSD.CEEM.IP" \
    --strategy "EMA" \
    --debug-mode \
    --show-intermediate-steps
```

### Time Series Analysis

```bash
# Analyze signals over time range
python -m forex_scanner.validation.signal_replay_validator \
    --epic "CS.D.EURUSD.CEEM.IP" \
    --start-time "2025-01-15 08:00:00" \
    --end-time "2025-01-15 18:00:00" \
    --interval 30 \
    --time-series
```

## System Architecture

### Core Components

1. **HistoricalDataManager** (`historical_data_manager.py`)
   - Fetches and manages historical candle data
   - Calculates technical indicators with historical accuracy
   - Provides market state recreation at specific timestamps

2. **ScannerStateRecreator** (`scanner_state_recreator.py`) 
   - Recreates scanner configuration as it existed historically
   - Manages strategy parameters and system settings
   - Handles configuration time-travel and restoration

3. **ReplayEngine** (`replay_engine.py`)
   - Core orchestration of the replay process
   - Coordinates data retrieval, state recreation, and signal detection
   - Manages parallel processing and performance optimization

4. **ValidationReporter** (`validation_reporter.py`)
   - Generates comprehensive validation reports
   - Provides multiple output formats (text, JSON)
   - Creates detailed market analysis and decision path documentation

5. **SignalReplayValidator** (`signal_replay_validator.py`)
   - Main CLI interface and high-level API
   - Command-line argument parsing and workflow management
   - Integration with existing scanner infrastructure

### Data Flow

```
1. Parse CLI arguments and validate inputs
2. Initialize database connections and components
3. Recreate historical scanner state for target timestamp
4. Fetch historical market data and calculate indicators
5. Replay signal detection using recreated components
6. Process signals through full pipeline (Smart Money, filters, etc.)
7. Compare results with stored historical alerts
8. Generate comprehensive validation report
9. Export results and cleanup resources
```

## Command-Line Interface

### Required Arguments

One of:
- `--timestamp "YYYY-MM-DD HH:MM:SS"` - Validate signals at specific timestamp
- `--time-series` - Run time series validation (requires `--start-time` and `--end-time`)

### Epic Selection

One of:
- `--epic "CS.D.EURUSD.CEEM.IP"` - Validate single epic
- `--all-epics` - Validate all configured epics
- `--epic-list "CS.D.EURUSD.CEEM.IP" "CS.D.GBPUSD.MINI.IP"` - Validate specific list

### Analysis Options

- `--timeframe {5m,15m,30m,1h}` - Timeframe for analysis (default: 15m)
- `--strategy {ema,macd,kama,zero_lag,momentum_bias}` - Focus on specific strategy
- `--show-calculations` - Show detailed technical calculations
- `--show-raw-data` - Include raw market data in output
- `--show-intermediate-steps` - Show decision path steps
- `--debug-mode` - Enable verbose debug logging

### Comparison and Export

- `--compare-with-stored` - Compare with stored alerts (default: True)
- `--no-compare` - Skip comparison with stored alerts
- `--export-json` - Export results to JSON format
- `--output-file FILE` - Save output to file instead of stdout

### Performance Options

- `--no-parallel` - Disable parallel processing for batch operations
- `--show-performance` - Show performance statistics after validation

### Time Series Options

- `--start-time "YYYY-MM-DD HH:MM:SS"` - Start time for time series
- `--end-time "YYYY-MM-DD HH:MM:SS"` - End time for time series  
- `--interval MINUTES` - Interval between validation points (default: 60)

## Configuration

### Replay Configuration (`replay_config.py`)

Key configuration options:

```python
# Data fetching
DEFAULT_LOOKBACK_BARS = 500  # Historical bars to fetch
MIN_LOOKBACK_BARS = 100     # Minimum bars for indicators

# Performance  
ENABLE_PARALLEL_PROCESSING = True
MAX_CONCURRENT_EPICS = 5
ENABLE_DATA_CACHING = True

# Error handling
MAX_RETRIES = 3
CONTINUE_ON_ERROR = True
DETAILED_ERROR_LOGGING = True

# Feature flags
ENABLE_SMART_MONEY_VALIDATION = True
ENABLE_CLAUDE_ANALYSIS_REPLAY = True
ENABLE_CONFIGURATION_TIME_TRAVEL = True
```

### Strategy-Specific Configuration

Each strategy has specific requirements:

```python
STRATEGY_REPLAY_CONFIG = {
    'ema': {
        'min_bars_required': 200,
        'indicators_needed': ['ema_21', 'ema_50', 'ema_200', 'two_pole_oscillator'],
        'mtf_analysis': True
    },
    'macd': {
        'min_bars_required': 100, 
        'indicators_needed': ['macd', 'macd_signal', 'macd_histogram', 'ema_200'],
        'mtf_analysis': True
    }
}
```

## Output Formats

### Validation Report

```
🔍 Signal Validation Report
═══════════════════════════════════════════════════════════
📅 Target Timestamp: 2025-01-15 14:30:00 UTC
🎯 Epic: CS.D.EURUSD.CEEM.IP
⚙️  Strategy: EMA Strategy

📊 Market Conditions at Timestamp:
   Price: 1.03456 (OHLC: 1.03400/1.03480/1.03390/1.03456)
   EMA(21): 1.03420 | EMA(50): 1.03380 | EMA(200): 1.03250
   Trend: BULLISH (EMA alignment confirmed)

🔬 Signal Detection Analysis:
   Signal Type: BULL
   Confidence: 74.0%
   Entry Price: 1.03456
   📊 EMA Analysis:
     EMA_21: 1.03420
     EMA_50: 1.03380
     EMA_200: 1.03250
     ✓ Trend Alignment: Confirmed
     ✓ Momentum: Confirmed

🧠 Smart Money Analysis:
   Validated: ✅
   Type: BOS_CONTINUATION
   Score: 0.76
   Enhanced Confidence: 68.0% → 74.0%

🎯 Final Validation Result:
   ✅ VALIDATION PASSED: Replay matches stored alert exactly

⏱️  Processing time: 234.5ms
📝 Report generated: 2025-01-16 10:30:15
```

### JSON Export

```json
{
  "export_timestamp": "2025-01-16T10:30:15.123Z",
  "total_results": 1,
  "results": [
    {
      "success": true,
      "epic": "CS.D.EURUSD.CEEM.IP", 
      "timestamp": "2025-01-15T14:30:00Z",
      "signal_detected": true,
      "processing_time_ms": 234.5,
      "signal_data": {
        "signal_type": "BULL",
        "confidence_score": 0.74,
        "strategy": "EMA",
        "entry_price": 1.03456,
        "smart_money_validated": true,
        "smart_money_score": 0.76
      },
      "market_state": {
        "price": {"close": 1.03456, "open": 1.03400},
        "indicators": {"ema_21": 1.03420, "ema_50": 1.03380},
        "trend": {"direction": "BULLISH", "ema_alignment": true}
      }
    }
  ]
}
```

## Error Handling

The system includes comprehensive error handling:

### Error Types

- `ValidationError` - Base validation error
- `DataRetrievalError` - Historical data fetch failures
- `StateRecreationError` - Scanner state recreation failures  
- `SignalDetectionError` - Signal detection failures
- `ConfigurationError` - Configuration issues

### Recovery Strategies

- **Automatic Retry** - Network and temporary failures
- **Graceful Degradation** - Continue with reduced functionality
- **Skip and Continue** - Skip problematic data points
- **Fail Fast** - Critical configuration errors

### Error Reporting

```python
from validation.error_handling import ValidationErrorHandler

error_handler = ValidationErrorHandler()

# Get error summary
summary = error_handler.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"Most common: {summary['most_common_error']}")
```

## Integration with Existing System

### Database Requirements

The validation system requires access to:
- `ig_candles` table - Historical market data
- `alert_history` table - Stored alerts for comparison
- Configuration data - Strategy settings and system parameters

### Docker Integration  

Run inside the task-worker container:

```bash
docker-compose exec -T task-worker python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "2025-01-15 14:30:00" \
    --epic "CS.D.EURUSD.CEEM.IP"
```

## Performance Considerations

### Optimization Features

- **Data Caching** - Cache frequently accessed historical data
- **Parallel Processing** - Process multiple epics simultaneously
- **Lazy Loading** - Load indicators only when needed
- **Query Optimization** - Efficient database queries with proper indexing

### Performance Monitoring

```python
validator = SignalReplayValidator()

# After validation operations
perf_stats = validator.get_performance_report()
print(perf_stats)
```

Expected performance:
- Single signal validation: 200-500ms
- Batch validation (9 epics): 2-5 seconds
- Time series (24 hours, 1-hour intervals): 30-60 seconds

## Troubleshooting

### Common Issues

1. **"No historical data found"**
   - Check timestamp is within available data range
   - Verify epic name format (CS.D.EURUSD.CEEM.IP)
   - Ensure database connection is working

2. **"Insufficient data for indicators"**
   - Increase lookback period in configuration
   - Check data availability before target timestamp
   - Verify timeframe is supported

3. **"Configuration time-travel failed"**
   - Disable configuration time-travel in config
   - Check for missing configuration keys
   - Verify strategy availability at timestamp

4. **Performance Issues**
   - Enable parallel processing for batch operations
   - Increase data cache size
   - Use smaller time windows for time series

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "2025-01-15 14:30:00" \
    --epic "CS.D.EURUSD.CEEM.IP" \
    --debug-mode \
    --verbose
```

## Development

### Adding New Validation Features

1. **Custom Validators**: Extend ValidationResult class
2. **New Report Formats**: Add methods to ValidationReporter
3. **Additional Strategies**: Update STRATEGY_REPLAY_CONFIG
4. **Custom Error Types**: Inherit from ValidationError

### Testing

```bash
# Test with recent timestamp (ensure data availability)
python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "$(date -d '2 hours ago' '+%Y-%m-%d %H:00:00')" \
    --epic "CS.D.EURUSD.CEEM.IP" \
    --debug-mode

# Test batch processing
python -m forex_scanner.validation.signal_replay_validator \
    --timestamp "$(date -d '2 hours ago' '+%Y-%m-%d %H:00:00')" \
    --all-epics \
    --show-performance
```

## License

Part of the TradeSystemV1 forex scanner project. See main project for licensing information.