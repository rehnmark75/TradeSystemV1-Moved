# TradeSystemV1 Documentation Hub

This documentation provides comprehensive guidance for Claude Code when working with the TradeSystemV1 codebase - a sophisticated algorithmic trading system with dynamic optimization and market intelligence capabilities.

## 📚 Documentation Structure

### 🚀 Quick Start
- **[Commands & CLI](claude-commands.md)** - Docker services, CLI usage, optimization commands
- **[Architecture Overview](claude-architecture.md)** - System design, services, and database models
- **[Configuration System](claude-configuration.md)** - Database-driven config, trailing stops, infrastructure

### 🛠️ Development
- **[Strategy Development](claude-strategies.md)** - Trading strategy patterns and configuration guidelines
- **[Development Best Practices](claude-development.md)** - Patterns, testing, and code organization

### 🎯 Advanced Systems
- **[Dynamic Parameter Optimization](claude-optimization.md)** - Database-driven parameter system
- **[Market Intelligence](claude-intelligence.md)** - Real-time market analysis and trade context

## 🎯 System Overview

TradeSystemV1 is a production-ready algorithmic trading system featuring:

- **Multi-Strategy Trading**: EMA, MACD, Zero-Lag, Smart Money Concepts
- **Dynamic Optimization**: Database-driven parameter system (14,406+ combinations tested per epic)
- **Market Intelligence**: Real-time regime detection and trade context analysis
- **Microservices Architecture**: Docker-based with FastAPI, Streamlit, and PostgreSQL
- **Advanced Validation**: 5-layer signal validation with multi-timeframe confirmation

## 🏗️ Core Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| **FastAPI Services** | REST APIs, WebSocket streaming, IG Markets integration | [Architecture](claude-architecture.md) |
| **Task Worker** | Strategy execution, signal detection, backtesting | [Commands](claude-commands.md) |
| **Streamlit Dashboard** | Real-time visualization and performance analytics | [Intelligence](claude-intelligence.md) |
| **Optimization Engine** | Dynamic parameter optimization and testing | [Optimization](claude-optimization.md) |
| **Strategy Framework** | Modular trading strategy development | [Strategies](claude-strategies.md) |

## 🚦 Quick Navigation

### Common Tasks
- **Run a backtest**: See [Commands](claude-commands.md#forex-scanner-cli)
- **Optimize strategy parameters**: See [Optimization](claude-optimization.md#ema-parameter-optimization-system)
- **Develop new strategy**: See [Strategies](claude-strategies.md#strategy-development-pattern)
- **View system architecture**: See [Architecture](claude-architecture.md#service-architecture)
- **Analyze market conditions**: See [Intelligence](claude-intelligence.md#market-regime-analysis)

### Development Workflows
- **Setup development environment**: [Development](claude-development.md#development-setup)
- **Debug Docker services**: [Commands](claude-commands.md#docker-services-management)
- **Monitor system performance**: [Architecture](claude-architecture.md#performance-monitoring)
- **Validate strategy configurations**: [Strategies](claude-strategies.md#validation-pattern)

## 📈 Recent Enhancements

- ✅ **Dynamic Parameter System**: Automatically uses optimal parameters from optimization results
- ✅ **MACD Database Integration**: Fixed parameter service for 12/24/8 optimized parameters  
- ✅ **EMA Strategy Optimization**: Reduced late entries with aggressive 5/13/50 configuration
- ✅ **Market Intelligence**: Real-time regime detection and trade context analysis
- ✅ **Modular Configuration**: 86% reduction in config.py size through strategy separation

## 🔧 Development Notes

This documentation is designed for Claude Code to understand and work with the TradeSystemV1 codebase effectively. Each section provides:

- **Practical commands** with real examples
- **Architecture insights** for understanding system design
- **Development patterns** for maintaining code quality
- **Troubleshooting guides** for common issues
- **Cross-references** between related concepts

For detailed implementation guidance, start with the most relevant section above based on your current task.