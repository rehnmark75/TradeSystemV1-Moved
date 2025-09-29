# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Documentation Structure

The TradeSystemV1 documentation has been organized into focused, specialized files for better navigation and maintainability:

### 🚀 Quick Start
- **[Overview & Navigation](claude-overview.md)** - Start here for system overview and navigation hub

### 📖 Core Documentation
- **[Commands & CLI](claude-commands.md)** - Docker services, CLI usage, optimization commands
- **[Architecture Overview](claude-architecture.md)** - System design, services, and database models

### 🔧 Development
- **[Strategy Development](claude-strategies.md)** - Trading strategy patterns and configuration guidelines
- **[Development Best Practices](claude-development.md)** - Patterns, testing, and code organization

### ⚡ Advanced Features
- **[Dynamic Parameter Optimization](claude-optimization.md)** - Database-driven parameter system
- **[Market Intelligence](claude-intelligence.md)** - Real-time market analysis and trade context

## 🎯 Quick Navigation

| I want to... | Go to |
|--------------|-------|
| **Start the system** | [Commands & CLI](claude-commands.md#docker-services-management) |
| **Run forex scanner** | [Commands & CLI](claude-commands.md#forex-scanner-cli) |
| **Understand the architecture** | [Architecture Overview](claude-architecture.md) |
| **Develop a new strategy** | [Strategy Development](claude-strategies.md) |
| **Optimize parameters** | [Dynamic Parameter Optimization](claude-optimization.md) |
| **Use market intelligence** | [Market Intelligence](claude-intelligence.md) |
| **Follow best practices** | [Development Best Practices](claude-development.md) |

## ⚠️ Important Notes

- **Docker Required**: All forex scanner commands must be run inside Docker containers
- **Database-Driven**: The system uses dynamic, optimized parameters from PostgreSQL
- **Modular Design**: New strategies follow lightweight configuration patterns
- **Real-time Intelligence**: Market analysis and trade context evaluation available

## 🤖 Agent Configuration

**Automatic Agent Usage:**

- **trading-strategy-analyst**: Automatically use this agent for any task involving:
  - Strategy performance analysis, backtest result evaluation, trading strategy optimization
  - Win rate, profit/loss analysis, strategy parameter tuning, market regime performance assessment
  - Keywords: strategy, backtest, performance, win rate, profit, loss, optimization, trading, analysis, momentum, RSI, MACD, EMA, SMA, bollinger, stochastic, parameters, signals, entry, exit, stop loss, take profit
  - File patterns: */strategies/*, */backtests/*, *backtest*.py, *strategy*.py, *config_*.py

- **devops-engineer**: Automatically use this agent for any task involving:
  - Docker orchestration, CI/CD pipelines, infrastructure automation, production deployment
  - System monitoring, alerting, performance optimization, infrastructure scaling
  - Keywords: docker, kubernetes, deployment, infrastructure, monitoring, CI/CD, pipeline, container, scaling, production, devops, automation
  - File patterns: docker-compose*.yml, Dockerfile*, k8s/*, .github/workflows/*, .gitlab-ci.yml

- **quantitative-researcher**: Automatically use this agent for any task involving:
  - Mathematical modeling, statistical analysis, hypothesis testing, risk modeling
  - Research methodology, alternative data analysis, factor modeling, portfolio optimization
  - Keywords: statistical, mathematical, model, research, hypothesis, risk, factor, correlation, regression, optimization, quantitative, econometric, simulation
  - File patterns: */research/*, */models/*, *analysis*.py, *research*.py, *model*.py

- **real-time-systems-engineer**: Automatically use this agent for any task involving:
  - Ultra-low latency optimization, high-frequency processing, concurrent programming
  - Real-time data processing, performance optimization, memory management
  - Keywords: latency, real-time, concurrent, performance, optimization, threading, memory, high-frequency, microsecond, lock-free
  - File patterns: */real_time/*, */concurrent/*, */performance/*, *latency*.py, *concurrent*.py

- **financial-data-engineer**: Automatically use this agent for any task involving:
  - Market data feeds, tick data processing, financial data normalization, order book management
  - Financial database optimization, time series data, market microstructure
  - Keywords: market data, tick data, financial, forex, currency, price, quote, trade, order book, time series, OHLCV
  - File patterns: */market_data/*, */feeds/*, */financial/*, *market*.py, *price*.py, *forex*.py

## 📈 System Status

**Recent Enhancements:**
- ✅ Dynamic parameter optimization system (database-driven)
- ✅ Market intelligence with regime detection
- ✅ Modular strategy configuration architecture
- ✅ Comprehensive documentation split

For detailed setup and usage instructions, start with the [Overview & Navigation](claude-overview.md).
- Add information about the new backtest system to memory
- memorize the new backtest system including the --pipeline switch and how the result is summarized when a backtest completes