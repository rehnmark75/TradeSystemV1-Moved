"""
Chunked Backtest Package

Provides parallel backtest execution with chunking and result aggregation.

Main components:
- ChunkCoordinator: Splits date ranges into parallel chunks
- ResultAggregator: Combines chunk results into unified metrics
- BacktestChartGenerator: Creates visual charts with signals
"""

from .chunk_coordinator import (
    ChunkCoordinator,
    ChunkConfig,
    ParallelRunConfig,
    ChunkResult,
    create_chunk_coordinator,
)

from .result_aggregator import (
    ResultAggregator,
    AggregatedMetrics,
    SignalRecord,
    create_result_aggregator,
)

from .backtest_chart_generator import (
    BacktestChartGenerator,
    create_backtest_chart_generator,
)

__all__ = [
    # Coordinator
    'ChunkCoordinator',
    'ChunkConfig',
    'ParallelRunConfig',
    'ChunkResult',
    'create_chunk_coordinator',

    # Aggregator
    'ResultAggregator',
    'AggregatedMetrics',
    'SignalRecord',
    'create_result_aggregator',

    # Chart Generator
    'BacktestChartGenerator',
    'create_backtest_chart_generator',
]
