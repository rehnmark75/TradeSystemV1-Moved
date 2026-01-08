# Unified Parameter Optimizer package
"""
Unified Parameter Optimization for SMC Simple Strategy

This package provides comprehensive parameter optimization by analyzing:
- Trade outcomes (from trade_log + alert_history)
- Rejection outcomes (from smc_simple_rejections + smc_rejection_outcomes)
- Market intelligence (from market_intelligence_history)

Components:
- data_collectors: Collect data from various database tables
- analyzers: Analyze correlations, direction performance, and regime impact
- generators: Generate SQL updates and formatted reports

Usage:
    from forex_scanner.optimization import UnifiedParameterOptimizer

    optimizer = UnifiedParameterOptimizer(days=30)
    results = optimizer.run()
    optimizer.print_report(results)
"""

from .unified_parameter_optimizer import UnifiedParameterOptimizer

__all__ = ['UnifiedParameterOptimizer']
