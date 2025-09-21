"""
Generic Strategy Adapter
Wraps existing backtest classes to work with the unified system
"""

import os
import sys
import importlib
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add worker path for accessing backtest modules
# Calculate correct path: streamlit/services/strategy_adapters/generic_adapter.py -> TradeSystemV1/worker/app
CURRENT_FILE = os.path.abspath(__file__)
STRATEGY_ADAPTERS_DIR = os.path.dirname(CURRENT_FILE)  # strategy_adapters/
SERVICES_DIR = os.path.dirname(STRATEGY_ADAPTERS_DIR)  # services/
STREAMLIT_DIR = os.path.dirname(SERVICES_DIR)  # streamlit/
PROJECT_ROOT = os.path.dirname(STREAMLIT_DIR)  # TradeSystemV1/
WORKER_PATH = os.path.join(PROJECT_ROOT, 'worker', 'app')

if WORKER_PATH not in sys.path:
    sys.path.insert(0, WORKER_PATH)

try:
    # Import config first
    import config as worker_config
except ImportError:
    try:
        from forex_scanner import config as worker_config
    except ImportError:
        worker_config = None

from ..backtest_service import BaseStrategyAdapter, StrategyInfo, BacktestConfig, BacktestResult


class GenericStrategyAdapter(BaseStrategyAdapter):
    """Generic adapter that can wrap any existing backtest class"""

    def __init__(self, strategy_name: str, strategy_info: StrategyInfo):
        super().__init__(strategy_name)
        self.strategy_info = strategy_info
        self._backtest_class = None
        self._epic_list = None

    def get_strategy_info(self) -> StrategyInfo:
        """Return information about this strategy"""
        return self.strategy_info

    def get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for this strategy"""
        return {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'days': 7,
            'timeframe': '15m',
            'show_signals': True
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for this strategy"""
        try:
            # Basic validation
            if 'epic' in parameters and not isinstance(parameters['epic'], str):
                return False
            if 'days' in parameters and (not isinstance(parameters['days'], int) or parameters['days'] < 1):
                return False
            if 'timeframe' in parameters and parameters['timeframe'] not in ['5m', '15m', '1h']:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Execute the backtest with given configuration"""
        start_time = datetime.now()

        try:
            # Load the backtest class
            backtest_instance = self._load_backtest_class()
            if not backtest_instance:
                raise Exception(f"Could not load backtest class for {self.strategy_name}")

            # Run the backtest
            success = self._execute_backtest(backtest_instance, config)

            if success:
                # Extract results - this would need to be customized per strategy
                signals = self._extract_signals(backtest_instance)
                chart_data = self._extract_chart_data(backtest_instance)

                # Calculate performance metrics
                from ..backtest_service import get_result_processor
                processor = get_result_processor()
                processed_signals = processor.process_signals(signals)
                performance_metrics = processor.calculate_performance_metrics(processed_signals)

                return BacktestResult(
                    strategy_name=config.strategy_name,
                    epic=config.epic,
                    timeframe=config.timeframe,
                    total_signals=len(processed_signals),
                    signals=processed_signals,
                    performance_metrics=performance_metrics,
                    execution_time=0,  # Will be set by runner
                    success=True,
                    chart_data=chart_data
                )
            else:
                raise Exception("Backtest execution failed")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=0,
                signals=[],
                performance_metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _load_backtest_class(self):
        """Load the actual backtest class"""
        try:
            if self._backtest_class is not None:
                return self._backtest_class

            # Import the module
            module = importlib.import_module(self.strategy_info.module_path)

            # Try to find the right class
            backtest_class = None

            # Look for specific class patterns
            class_patterns = [
                f"{self.strategy_name.upper()}Backtest",
                f"{self.strategy_name.title().replace('_', '')}Backtest",
                "BacktestBase",  # Fallback to base class
            ]

            for pattern in class_patterns:
                if hasattr(module, pattern):
                    backtest_class = getattr(module, pattern)
                    break

            # If no specific class found, look for any class that looks like a backtest
            if not backtest_class:
                for name in dir(module):
                    obj = getattr(module, name)
                    if (hasattr(obj, '__call__') and
                        hasattr(obj, 'run_backtest') and
                        name != 'BacktestBase'):
                        backtest_class = obj
                        break

            if backtest_class:
                # Try to instantiate it
                if hasattr(backtest_class, '__init__'):
                    # Check if it's a class that needs to be instantiated
                    try:
                        if 'strategy_name' in backtest_class.__init__.__code__.co_varnames:
                            self._backtest_class = backtest_class(self.strategy_name)
                        else:
                            self._backtest_class = backtest_class()
                    except Exception as e:
                        self.logger.warning(f"Could not instantiate {backtest_class.__name__}: {e}")
                        # Try without parameters
                        try:
                            self._backtest_class = backtest_class()
                        except:
                            self._backtest_class = backtest_class
                else:
                    self._backtest_class = backtest_class

                return self._backtest_class

            self.logger.error(f"No suitable backtest class found in {self.strategy_info.module_path}")
            return None

        except ImportError as e:
            self.logger.error(f"Could not import {self.strategy_info.module_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading backtest class: {e}")
            return None

    def _execute_backtest(self, backtest_instance, config: BacktestConfig) -> bool:
        """Execute the backtest using the instance"""
        try:
            # Different strategies might have different method signatures
            method_name = 'run_backtest'

            if hasattr(backtest_instance, method_name):
                method = getattr(backtest_instance, method_name)

                # Try to call with different parameter combinations
                try:
                    # Try with all parameters
                    return method(
                        epic=config.epic,
                        days=config.days,
                        timeframe=config.timeframe,
                        show_signals=config.parameters.get('show_signals', True),
                        **config.parameters
                    )
                except TypeError:
                    try:
                        # Try with basic parameters
                        return method(
                            epic=config.epic,
                            days=config.days,
                            timeframe=config.timeframe,
                            show_signals=config.parameters.get('show_signals', True)
                        )
                    except TypeError:
                        try:
                            # Try with just epic
                            return method(epic=config.epic)
                        except TypeError:
                            # Try with no parameters
                            return method()

            # If no run_backtest method, try to call the instance directly
            elif callable(backtest_instance):
                return backtest_instance(
                    epic=config.epic,
                    days=config.days,
                    timeframe=config.timeframe
                )

            self.logger.error(f"No executable method found for {self.strategy_name}")
            return False

        except Exception as e:
            self.logger.error(f"Error executing backtest: {e}")
            return False

    def _extract_signals(self, backtest_instance) -> List[Dict[str, Any]]:
        """Extract signals from the backtest instance"""
        signals = []

        try:
            # Look for common signal storage patterns
            signal_attributes = [
                'signals', 'all_signals', 'backtest_signals',
                'results', 'signal_results', '_signals'
            ]

            for attr in signal_attributes:
                if hasattr(backtest_instance, attr):
                    signal_data = getattr(backtest_instance, attr)
                    if isinstance(signal_data, list) and signal_data:
                        signals = signal_data
                        break

            # If no signals found in instance, try to get them from modules that might store them
            if not signals:
                # Some backtests might store signals in global variables or other ways
                # This would need customization per strategy
                pass

            self.logger.info(f"Extracted {len(signals)} signals from {self.strategy_name}")

        except Exception as e:
            self.logger.warning(f"Could not extract signals from {self.strategy_name}: {e}")

        return signals

    def _extract_chart_data(self, backtest_instance) -> Optional[Any]:
        """Extract chart data from the backtest instance"""
        try:
            # Look for chart data attributes
            chart_attributes = [
                'chart_data', 'df', 'data', 'price_data',
                'historical_data', '_chart_data'
            ]

            for attr in chart_attributes:
                if hasattr(backtest_instance, attr):
                    chart_data = getattr(backtest_instance, attr)
                    if chart_data is not None:
                        return chart_data

        except Exception as e:
            self.logger.warning(f"Could not extract chart data from {self.strategy_name}: {e}")

        return None