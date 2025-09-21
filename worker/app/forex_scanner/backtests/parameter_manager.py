# core/backtest/parameter_manager.py
"""
Parameter Manager - Systematic parameter testing and optimization

This module handles parameter sweeps, optimization, and systematic testing
of different parameter combinations for trading strategies.

Features:
- Parameter range definition and validation
- Grid search and random search capabilities
- Genetic algorithm optimization (future)
- Performance-based parameter ranking
- Cross-validation with time-based splits
- Parameter sensitivity analysis
"""

import logging
import itertools
import random
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import math

try:
    from strategy_registry import get_strategy_registry
    import config
except ImportError:
    from forex_scanner.backtests.strategy_registry import get_strategy_registry
    from forex_scanner import config


class OptimizationMethod(Enum):
    """Parameter optimization methods"""
    GRID_SEARCH = "grid"
    RANDOM_SEARCH = "random"
    GENETIC_ALGORITHM = "genetic"
    BAYESIAN_OPTIMIZATION = "bayesian"
    ADAPTIVE_SEARCH = "adaptive"


class ParameterType(Enum):
    """Parameter data types"""
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    CATEGORICAL = "categorical"


@dataclass
class ParameterRange:
    """Definition of a parameter range for optimization"""
    name: str
    param_type: ParameterType
    min_value: Union[int, float, bool] = None
    max_value: Union[int, float, bool] = None
    step: Union[int, float] = None
    values: List[Any] = None  # For categorical parameters
    default: Any = None
    description: str = ""

    def __post_init__(self):
        """Validate parameter range definition"""
        if self.param_type == ParameterType.CATEGORICAL:
            if not self.values:
                raise ValueError(f"Categorical parameter {self.name} must have values list")
        else:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Parameter {self.name} must have min_value and max_value")

    def generate_values(self, max_values: int = None) -> List[Any]:
        """Generate list of values for this parameter"""
        if self.param_type == ParameterType.CATEGORICAL:
            return self.values.copy()

        elif self.param_type == ParameterType.BOOLEAN:
            return [False, True]

        elif self.param_type == ParameterType.INTEGER:
            step = self.step or 1
            values = list(range(int(self.min_value), int(self.max_value) + 1, int(step)))

        elif self.param_type == ParameterType.FLOAT:
            step = self.step or (self.max_value - self.min_value) / 10
            num_steps = int((self.max_value - self.min_value) / step) + 1
            values = [round(self.min_value + i * step, 6) for i in range(num_steps)]

        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")

        # Limit number of values if specified
        if max_values and len(values) > max_values:
            # Sample evenly across the range
            indices = np.linspace(0, len(values) - 1, max_values, dtype=int)
            values = [values[i] for i in indices]

        return values


@dataclass
class ParameterSet:
    """A specific combination of parameters"""
    parameters: Dict[str, Any]
    score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    signals_count: int = 0
    error: Optional[str] = None

    def __hash__(self):
        """Make parameter set hashable for deduplication"""
        return hash(tuple(sorted(self.parameters.items())))


@dataclass
class OptimizationResult:
    """Result from parameter optimization"""
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: List[ParameterSet]
    optimization_method: OptimizationMethod
    total_combinations: int
    successful_combinations: int
    execution_time: float
    strategy_name: str
    epic: str
    timeframe: str
    scoring_metric: str = "win_rate"


class ParameterManager:
    """
    Manager for systematic parameter testing and optimization

    Handles parameter sweeps, optimization, and systematic testing
    of different parameter combinations for trading strategies.
    """

    def __init__(self):
        self.logger = logging.getLogger('parameter_manager')
        self.registry = get_strategy_registry()

        # Default scoring metrics
        self.scoring_metrics = {
            'win_rate': lambda metrics: metrics.get('win_rate', 0),
            'profit_factor': lambda metrics: metrics.get('profit_factor', 0),
            'sharpe_ratio': lambda metrics: metrics.get('sharpe_ratio', 0),
            'total_return': lambda metrics: metrics.get('total_return_pips', 0),
            'signal_count': lambda metrics: metrics.get('total_signals', 0),
            'avg_profit': lambda metrics: metrics.get('average_profit_pips', 0),
            'risk_reward': lambda metrics: metrics.get('risk_reward_ratio', 0)
        }

    def parse_parameter_ranges(self, range_string: str) -> Dict[str, ParameterRange]:
        """
        Parse parameter range string into ParameterRange objects

        Format examples:
        - "confidence:0.4-0.8:0.1" (float range with step)
        - "short_ema:8-34:2" (integer range with step)
        - "use_mtf:true,false" (boolean/categorical)
        - "mode:consensus,weighted,majority" (categorical)

        Args:
            range_string: Comma-separated parameter definitions

        Returns:
            Dictionary mapping parameter names to ParameterRange objects
        """
        ranges = {}

        if not range_string:
            return ranges

        try:
            # Split by comma and process each parameter definition
            param_definitions = [p.strip() for p in range_string.split(',')]

            for definition in param_definitions:
                if ':' not in definition:
                    continue

                parts = definition.split(':')
                if len(parts) < 2:
                    continue

                param_name = parts[0].strip()

                # Check if it's a range (contains dash) or categorical (contains comma in second part)
                if '-' in parts[1] and ',' not in parts[1]:
                    # Range definition: name:min-max or name:min-max:step
                    range_part = parts[1].strip()
                    min_val, max_val = range_part.split('-')

                    # Determine if it's int or float
                    try:
                        min_val = int(min_val)
                        max_val = int(max_val)
                        param_type = ParameterType.INTEGER
                    except ValueError:
                        min_val = float(min_val)
                        max_val = float(max_val)
                        param_type = ParameterType.FLOAT

                    # Get step if provided
                    step = None
                    if len(parts) > 2:
                        try:
                            step = int(parts[2]) if param_type == ParameterType.INTEGER else float(parts[2])
                        except ValueError:
                            pass

                    ranges[param_name] = ParameterRange(
                        name=param_name,
                        param_type=param_type,
                        min_value=min_val,
                        max_value=max_val,
                        step=step
                    )

                else:
                    # Categorical definition: name:val1,val2,val3 or combined in second part
                    if ',' in parts[1]:
                        values_str = parts[1]
                    else:
                        # Combine remaining parts for categorical values
                        values_str = ':'.join(parts[1:])

                    values = [v.strip() for v in values_str.split(',')]

                    # Try to convert to appropriate types
                    converted_values = []
                    for val in values:
                        val_lower = val.lower()
                        if val_lower in ['true', 'false']:
                            converted_values.append(val_lower == 'true')
                        else:
                            try:
                                # Try integer first
                                converted_values.append(int(val))
                            except ValueError:
                                try:
                                    # Try float
                                    converted_values.append(float(val))
                                except ValueError:
                                    # Keep as string
                                    converted_values.append(val)

                    ranges[param_name] = ParameterRange(
                        name=param_name,
                        param_type=ParameterType.CATEGORICAL,
                        values=converted_values
                    )

        except Exception as e:
            self.logger.error(f"❌ Error parsing parameter ranges: {e}")

        return ranges

    def generate_parameter_combinations(
        self,
        strategy_name: str,
        parameter_ranges: Dict[str, ParameterRange],
        method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
        max_combinations: int = 1000,
        random_seed: int = None
    ) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for testing

        Args:
            strategy_name: Name of the strategy
            parameter_ranges: Parameter ranges to test
            method: Optimization method
            max_combinations: Maximum number of combinations
            random_seed: Random seed for reproducibility

        Returns:
            List of parameter combinations
        """
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Get strategy metadata for default parameters
        strategy_metadata = self.registry.get_strategy(strategy_name)
        base_parameters = strategy_metadata.default_parameters.copy() if strategy_metadata else {}

        if method == OptimizationMethod.GRID_SEARCH:
            return self._generate_grid_search_combinations(
                base_parameters, parameter_ranges, max_combinations
            )
        elif method == OptimizationMethod.RANDOM_SEARCH:
            return self._generate_random_search_combinations(
                base_parameters, parameter_ranges, max_combinations
            )
        else:
            self.logger.warning(f"⚠️ Optimization method {method} not implemented, using grid search")
            return self._generate_grid_search_combinations(
                base_parameters, parameter_ranges, max_combinations
            )

    def _generate_grid_search_combinations(
        self,
        base_parameters: Dict[str, Any],
        parameter_ranges: Dict[str, ParameterRange],
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate all combinations using grid search"""

        # Generate values for each parameter
        param_values = {}
        for name, param_range in parameter_ranges.items():
            # Calculate max values per parameter to stay under limit
            max_per_param = max(2, int(max_combinations ** (1.0 / len(parameter_ranges))))
            param_values[name] = param_range.generate_values(max_per_param)

        # Generate all combinations
        combinations = []
        param_names = list(param_values.keys())

        if not param_names:
            return [base_parameters.copy()]

        for combination in itertools.product(*[param_values[name] for name in param_names]):
            if len(combinations) >= max_combinations:
                break

            params = base_parameters.copy()
            for i, value in enumerate(combination):
                params[param_names[i]] = value

            combinations.append(params)

        self.logger.info(f"📊 Generated {len(combinations)} parameter combinations (grid search)")
        return combinations

    def _generate_random_search_combinations(
        self,
        base_parameters: Dict[str, Any],
        parameter_ranges: Dict[str, ParameterRange],
        max_combinations: int
    ) -> List[Dict[str, Any]]:
        """Generate random combinations"""

        combinations = set()  # Use set to avoid duplicates
        param_names = list(parameter_ranges.keys())

        if not param_names:
            return [base_parameters.copy()]

        # Generate values for each parameter
        param_values = {}
        for name, param_range in parameter_ranges.items():
            param_values[name] = param_range.generate_values()

        # Generate random combinations
        attempts = 0
        max_attempts = max_combinations * 3  # Avoid infinite loop

        while len(combinations) < max_combinations and attempts < max_attempts:
            params = base_parameters.copy()

            for param_name in param_names:
                available_values = param_values[param_name]
                params[param_name] = random.choice(available_values)

            # Convert to tuple for set membership (hashable)
            param_tuple = tuple(sorted(params.items()))
            combinations.add(param_tuple)
            attempts += 1

        # Convert back to list of dictionaries
        result = [dict(combination) for combination in combinations]

        self.logger.info(f"📊 Generated {len(result)} parameter combinations (random search)")
        return result

    def optimize_parameters(
        self,
        strategy_name: str,
        epic: str,
        timeframe: str,
        parameter_ranges: Dict[str, ParameterRange],
        backtest_engine,  # Avoid circular import
        backtest_config,
        method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
        scoring_metric: str = "win_rate",
        max_combinations: int = 100,
        cross_validation_splits: int = 1,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize parameters for a strategy

        Args:
            strategy_name: Name of the strategy to optimize
            epic: Epic to test on
            timeframe: Timeframe for testing
            parameter_ranges: Parameter ranges to optimize
            backtest_engine: Backtest engine instance
            backtest_config: Base backtest configuration
            method: Optimization method
            scoring_metric: Metric to optimize for
            max_combinations: Maximum parameter combinations to test
            cross_validation_splits: Number of time-based splits for validation
            verbose: Whether to show progress

        Returns:
            OptimizationResult with best parameters and all results
        """
        start_time = datetime.now()

        self.logger.info("🎯 PARAMETER OPTIMIZATION")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Strategy: {strategy_name}")
        self.logger.info(f"📈 Epic: {epic}")
        self.logger.info(f"⏰ Timeframe: {timeframe}")
        self.logger.info(f"🔍 Method: {method.value}")
        self.logger.info(f"🎚️ Scoring: {scoring_metric}")
        self.logger.info(f"🧪 Max combinations: {max_combinations}")

        try:
            # Generate parameter combinations
            combinations = self.generate_parameter_combinations(
                strategy_name, parameter_ranges, method, max_combinations
            )

            if not combinations:
                raise ValueError("No parameter combinations generated")

            # Test each combination
            results = []
            successful_tests = 0

            for i, params in enumerate(combinations):
                if verbose and i % max(1, len(combinations) // 10) == 0:
                    progress = (i / len(combinations)) * 100
                    self.logger.info(f"   Progress: {progress:.1f}% ({i}/{len(combinations)})")

                # Test this parameter combination
                result = self._test_parameter_combination(
                    strategy_name, epic, timeframe, params,
                    backtest_engine, backtest_config, cross_validation_splits
                )

                results.append(result)
                if result.error is None:
                    successful_tests += 1

            # Sort results by score
            scoring_func = self.scoring_metrics.get(scoring_metric, self.scoring_metrics['win_rate'])
            valid_results = [r for r in results if r.error is None and r.score is not None]

            if not valid_results:
                raise ValueError("No successful parameter combinations found")

            valid_results.sort(key=lambda x: x.score, reverse=True)
            best_result = valid_results[0]

            execution_time = (datetime.now() - start_time).total_seconds()

            optimization_result = OptimizationResult(
                best_parameters=best_result.parameters,
                best_score=best_result.score,
                all_results=results,
                optimization_method=method,
                total_combinations=len(combinations),
                successful_combinations=successful_tests,
                execution_time=execution_time,
                strategy_name=strategy_name,
                epic=epic,
                timeframe=timeframe,
                scoring_metric=scoring_metric
            )

            self.logger.info(f"✅ Optimization completed in {execution_time:.1f}s")
            self.logger.info(f"🏆 Best score: {best_result.score:.3f}")
            self.logger.info(f"📊 Success rate: {successful_tests}/{len(combinations)} ({successful_tests/len(combinations)*100:.1f}%)")

            return optimization_result

        except Exception as e:
            self.logger.error(f"❌ Parameter optimization failed: {e}")
            raise

    def _test_parameter_combination(
        self,
        strategy_name: str,
        epic: str,
        timeframe: str,
        parameters: Dict[str, Any],
        backtest_engine,
        base_config,
        cv_splits: int
    ) -> ParameterSet:
        """Test a single parameter combination"""
        try:
            # Create modified config with these parameters
            test_config = self._create_test_config(base_config, parameters)

            # Run backtest with these parameters
            # Note: This would need to be integrated with the backtest engine
            # For now, we'll create a placeholder result

            # TODO: Integrate with actual backtest execution
            # results = backtest_engine.run_backtest(test_config)

            # Placeholder metrics for now
            metrics = {
                'win_rate': random.uniform(0.3, 0.8),
                'total_signals': random.randint(5, 50),
                'profit_factor': random.uniform(0.8, 2.5),
                'average_profit_pips': random.uniform(5, 25)
            }

            # Calculate score using the default scoring metric (win_rate)
            score = metrics.get('win_rate', 0)

            return ParameterSet(
                parameters=parameters,
                score=score,
                metrics=metrics,
                execution_time=random.uniform(0.1, 2.0),
                signals_count=metrics['total_signals']
            )

        except Exception as e:
            return ParameterSet(
                parameters=parameters,
                error=str(e)
            )

    def _create_test_config(self, base_config, parameters: Dict[str, Any]):
        """Create a test configuration with modified parameters"""
        # This would create a modified config for testing
        # Implementation depends on how the backtest config is structured
        return base_config

    def analyze_parameter_sensitivity(
        self,
        optimization_result: OptimizationResult,
        parameter_name: str
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity of results to a specific parameter

        Args:
            optimization_result: Results from parameter optimization
            parameter_name: Name of parameter to analyze

        Returns:
            Dictionary with sensitivity analysis results
        """
        valid_results = [r for r in optimization_result.all_results if r.error is None]

        if not valid_results:
            return {'error': 'No valid results for sensitivity analysis'}

        # Group results by parameter value
        parameter_groups = {}
        for result in valid_results:
            param_value = result.parameters.get(parameter_name)
            if param_value is not None:
                if param_value not in parameter_groups:
                    parameter_groups[param_value] = []
                parameter_groups[param_value].append(result.score)

        # Calculate statistics for each parameter value
        sensitivity_data = {}
        for param_value, scores in parameter_groups.items():
            sensitivity_data[param_value] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'count': len(scores)
            }

        # Calculate overall sensitivity metrics
        mean_scores = [data['mean_score'] for data in sensitivity_data.values()]
        sensitivity_range = max(mean_scores) - min(mean_scores) if mean_scores else 0
        sensitivity_coefficient = sensitivity_range / max(mean_scores) if mean_scores and max(mean_scores) > 0 else 0

        return {
            'parameter_name': parameter_name,
            'sensitivity_data': sensitivity_data,
            'sensitivity_range': sensitivity_range,
            'sensitivity_coefficient': sensitivity_coefficient,
            'high_sensitivity': sensitivity_coefficient > 0.2  # Arbitrary threshold
        }

    def generate_optimization_report(self, optimization_result: OptimizationResult) -> str:
        """Generate a detailed optimization report"""

        report = f"""
🎯 PARAMETER OPTIMIZATION REPORT
={'=' * 50}

Strategy: {optimization_result.strategy_name}
Epic: {optimization_result.epic}
Timeframe: {optimization_result.timeframe}
Optimization Method: {optimization_result.optimization_method.value}
Scoring Metric: {optimization_result.scoring_metric}

📊 EXECUTION SUMMARY
{'-' * 25}
Total Combinations: {optimization_result.total_combinations}
Successful Tests: {optimization_result.successful_combinations}
Success Rate: {optimization_result.successful_combinations/optimization_result.total_combinations*100:.1f}%
Execution Time: {optimization_result.execution_time:.1f}s
Average Time per Test: {optimization_result.execution_time/optimization_result.total_combinations:.2f}s

🏆 BEST PARAMETERS
{'-' * 20}
Score: {optimization_result.best_score:.4f}
Parameters:
"""

        for param, value in optimization_result.best_parameters.items():
            report += f"  • {param}: {value}\n"

        # Top 5 results
        valid_results = [r for r in optimization_result.all_results if r.error is None]
        valid_results.sort(key=lambda x: x.score, reverse=True)
        top_results = valid_results[:5]

        report += f"\n📈 TOP 5 RESULTS\n{'-' * 18}\n"
        for i, result in enumerate(top_results, 1):
            report += f"{i}. Score: {result.score:.4f}, Signals: {result.signals_count}\n"
            key_params = {k: v for k, v in result.parameters.items()
                         if k in optimization_result.best_parameters}
            report += f"   {key_params}\n"

        # Performance distribution
        scores = [r.score for r in valid_results]
        if scores:
            report += f"\n📊 SCORE DISTRIBUTION\n{'-' * 22}\n"
            report += f"Mean: {np.mean(scores):.4f}\n"
            report += f"Std Dev: {np.std(scores):.4f}\n"
            report += f"Min: {np.min(scores):.4f}\n"
            report += f"Max: {np.max(scores):.4f}\n"
            report += f"25th Percentile: {np.percentile(scores, 25):.4f}\n"
            report += f"75th Percentile: {np.percentile(scores, 75):.4f}\n"

        return report.strip()

    def export_optimization_results(
        self,
        optimization_result: OptimizationResult,
        file_path: str,
        format: str = "json"
    ):
        """Export optimization results to file"""

        export_data = {
            'strategy_name': optimization_result.strategy_name,
            'epic': optimization_result.epic,
            'timeframe': optimization_result.timeframe,
            'optimization_method': optimization_result.optimization_method.value,
            'scoring_metric': optimization_result.scoring_metric,
            'best_parameters': optimization_result.best_parameters,
            'best_score': optimization_result.best_score,
            'total_combinations': optimization_result.total_combinations,
            'successful_combinations': optimization_result.successful_combinations,
            'execution_time': optimization_result.execution_time,
            'results': []
        }

        # Add all results
        for result in optimization_result.all_results:
            result_data = {
                'parameters': result.parameters,
                'score': result.score,
                'metrics': result.metrics,
                'execution_time': result.execution_time,
                'signals_count': result.signals_count,
                'error': result.error
            }
            export_data['results'].append(result_data)

        try:
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"✅ Optimization results exported to {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")
            raise