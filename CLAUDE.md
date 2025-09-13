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

## 📈 System Status

**Recent Enhancements:**
- ✅ Dynamic parameter optimization system (database-driven)
- ✅ Market intelligence with regime detection
- ✅ Modular strategy configuration architecture
- ✅ Comprehensive documentation split

For detailed setup and usage instructions, start with the [Overview & Navigation](claude-overview.md).