"""
Parameter Variation Module

Provides parallel parameter testing for backtests.
Allows testing multiple parameter combinations simultaneously.
"""

from .parameter_grid import ParameterGridGenerator, GridSpec
from .variation_result import VariationResult, ResultRanker
from .variation_runner import ParallelVariationRunner, VariationRunConfig

__all__ = [
    'ParameterGridGenerator',
    'GridSpec',
    'VariationResult',
    'ResultRanker',
    'ParallelVariationRunner',
    'VariationRunConfig',
]
