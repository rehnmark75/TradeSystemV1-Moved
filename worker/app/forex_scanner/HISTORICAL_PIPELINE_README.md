# Historical Scanner Pipeline Backtesting

## Overview

The new Historical Scanner Pipeline backtesting system provides **realistic backtesting** by using the complete production scanner pipeline on historical data from the `ig_candles` table.

## Problem Solved

**Before:** Backtests used simplified strategy calls that bypassed the real trading pipeline, making results unrealistic.

**Now:** Backtests use the exact same scanner pipeline as production, including:
- SignalProcessor (Smart Money analysis)
- EnhancedSignalValidator (confidence scoring, market efficiency checks)
- TradeValidator (trend filters, S/R validation, news filtering)
- TradingOrchestrator (Claude analysis, risk management)
- Complete deduplication and validation logic

## Architecture

```
Historical Data (ig_candles) → HistoricalScannerEngine → Full Scanner Pipeline → Trade Decision Logging
```

### Key Components

1. **HistoricalScannerEngine** - Main orchestrator that steps through historical timestamps
2. **Modified DataFetcher** - Supports historical timestamp constraints to prevent lookahead bias
3. **Enhanced TradingOrchestrator** - Logs trade decisions instead of executing them
4. **backtest_trades table** - Stores comprehensive trade decision data

## Usage

### 1. Run Database Migration (First Time Only)

```bash
cd /home/hr/Projects/TradeSystemV1/worker/app/forex_scanner/migrations
python run_backtest_trades_migration.py
```

### 2. Run Historical Pipeline Backtests

#### Momentum Strategy with Full Pipeline
```bash
python backtest_momentum.py --pipeline --epic CS.D.EURUSD.MINI.IP --days 7
```

#### All Strategies Comparison (Traditional)
```bash
python backtest_all.py --epic CS.D.EURUSD.MINI.IP --days 7
```

#### Test the System
```bash
python test_historical_pipeline.py
```

## Backtest Modes Comparison

| Mode | Command | Description | Realism Level |
|------|---------|-------------|---------------|
| **Pipeline** | `--pipeline` | Complete scanner pipeline | 🌟🌟🌟🌟🌟 Highest |
| **Strategy** | `--backtest` | Strategy-only testing | 🌟🌟🌟 Medium |
| **Simulate Live** | `--simulate-live` | Signal detector simulation | 🌟🌟🌟🌟 High |

## Features

### ✅ Realistic Pipeline Processing
- Uses exact same validation logic as production
- All signals go through complete pipeline
- No shortcuts or simplifications

### ✅ No Lookahead Bias
- DataFetcher constrained to historical timestamp
- Scanner only sees data available at each point in time
- Chronological processing maintains temporal integrity

### ✅ Comprehensive Decision Logging
- Complete trade decision context captured
- All validation results stored
- Claude analysis results included
- Rejection reasons detailed

### ✅ Database Integration
- Results stored in `backtest_trades` table
- Indexes for efficient analysis
- Views for summary reporting

## Database Schema

The `backtest_trades` table captures:

- **Decision Context**: Action (BUY/SELL/REJECT), reason, pipeline stage
- **Signal Data**: Original and validated signal information
- **Validation Results**: Confidence scores, intelligence filtering, risk management
- **Claude Analysis**: AI recommendations and reasoning (if enabled)
- **Market Context**: Regime, session, volatility level
- **Performance Tracking**: For future analysis and optimization

## Benefits

1. **Accurate Performance Prediction** - Results closely match live trading
2. **Complete Decision Analysis** - Understand why trades were approved/rejected
3. **Pipeline Debugging** - Identify bottlenecks in real trading logic
4. **Strategy Comparison** - Compare strategies using identical infrastructure
5. **Risk Assessment** - Test validation rules before live deployment

## Example Output

```
🚀 HISTORICAL SCANNER PIPELINE BACKTEST
====================================================
📊 Configuration:
   Epic(s): ['CS.D.EURUSD.MINI.IP']
   Period: 2025-09-18 to 2025-09-25
   Days: 7
   Timeframe: 15m

🎉 HISTORICAL PIPELINE BACKTEST RESULTS
====================================================
📊 Scan Statistics:
   Total scans: 672
   Signals detected: 12
   Trade decisions: 12
   Approved trades: 8
   Rejected trades: 4
   Approval rate: 66.7%

💰 Sample Approved Trades:
   1. BUY CS.D.EURUSD.MINI.IP (momentum_strategy) - Confidence: 78.5%
   2. SELL CS.D.EURUSD.MINI.IP (momentum_strategy) - Confidence: 82.1%
```

## File Structure

```
forex_scanner/
├── core/backtest/
│   └── historical_scanner_engine.py    # Main orchestrator
├── core/
│   ├── data_fetcher.py                  # Enhanced with historical constraints
│   └── trading/
│       └── trading_orchestrator.py     # Enhanced with decision logging
├── backtests/
│   ├── backtest_momentum.py            # Updated with --pipeline mode
│   └── backtest_all.py                 # Compatible with pipeline mode
├── migrations/
│   ├── create_backtest_trades_table.sql
│   └── run_backtest_trades_migration.py
└── test_historical_pipeline.py         # Demonstration script
```

## Next Steps

1. **Run the test**: `python test_historical_pipeline.py`
2. **Try pipeline mode**: `python backtest_momentum.py --pipeline --days 3`
3. **Compare modes**: Run same backtest with `--backtest` vs `--pipeline`
4. **Analyze results**: Query `backtest_trades` table for detailed analysis

This system bridges the gap between backtesting and live trading, providing confidence that backtest results will translate to real performance.