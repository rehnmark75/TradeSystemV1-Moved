"""
Forex Scanner Services Package

Contains service classes for database-driven configuration and operations.
"""

from .smc_simple_config_service import (
    SMCSimpleConfig,
    SMCSimpleConfigService,
    get_smc_simple_config_service,
    get_smc_simple_config,
)

from .minio_client import (
    MinIOChartClient,
    get_minio_client,
    upload_vision_chart,
)

from .strategy_router_service import (
    StrategyRouterService,
    TradingMode,
    StrategyInfo,
    RoutingRule,
    StrategyFitness,
    RouterConfig,
    get_strategy_router,
    get_strategy_for_conditions,
)

from .strategy_performance_tracker import (
    StrategyPerformanceTracker,
    WindowPerformance,
    StrategyPerformance,
    FitnessConfig,
    get_performance_tracker,
)

__all__ = [
    # SMC Simple Config
    'SMCSimpleConfig',
    'SMCSimpleConfigService',
    'get_smc_simple_config_service',
    'get_smc_simple_config',
    # MinIO
    'MinIOChartClient',
    'get_minio_client',
    'upload_vision_chart',
    # Strategy Router
    'StrategyRouterService',
    'TradingMode',
    'StrategyInfo',
    'RoutingRule',
    'StrategyFitness',
    'RouterConfig',
    'get_strategy_router',
    'get_strategy_for_conditions',
    # Performance Tracker
    'StrategyPerformanceTracker',
    'WindowPerformance',
    'StrategyPerformance',
    'FitnessConfig',
    'get_performance_tracker',
]
