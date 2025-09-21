"""
Modular Backtest Service
Core framework for unified backtest execution across all strategies
"""

import os
import sys
import inspect
import importlib
import logging
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import traceback

# Add worker path for accessing backtest modules
# Calculate correct path: streamlit/services/backtest_service.py -> TradeSystemV1/worker/app
CURRENT_FILE = os.path.abspath(__file__)
STREAMLIT_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE))  # Go up from services/ to streamlit/
PROJECT_ROOT = os.path.dirname(STREAMLIT_DIR)  # Go up from streamlit/ to TradeSystemV1/
WORKER_PATH = os.path.join(PROJECT_ROOT, 'worker', 'app')

if WORKER_PATH not in sys.path:
    sys.path.insert(0, WORKER_PATH)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    strategy_name: str
    epic: str = "CS.D.EURUSD.MINI.IP"
    days: int = 7
    timeframe: str = "15m"
    show_signals: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Standardized backtest result"""
    strategy_name: str
    epic: str
    timeframe: str
    total_signals: int
    signals: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    chart_data: Optional[pd.DataFrame] = None


@dataclass
class StrategyInfo:
    """Information about a discovered strategy"""
    name: str
    display_name: str
    description: str
    module_path: str
    class_name: str
    parameters: Dict[str, Any]
    adapter_class: Optional[Type] = None


class BaseStrategyAdapter(ABC):
    """Abstract base class for strategy adapters"""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"adapter_{strategy_name}")

    @abstractmethod
    def get_strategy_info(self) -> StrategyInfo:
        """Return information about this strategy"""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for this strategy"""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for this strategy"""
        pass

    @abstractmethod
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Execute the backtest with given configuration"""
        pass


class BacktestRegistry:
    """Registry for managing available backtest strategies"""

    def __init__(self):
        self.strategies: Dict[str, StrategyInfo] = {}
        self.adapters: Dict[str, BaseStrategyAdapter] = {}
        self.logger = logging.getLogger(__name__)

    def discover_strategies(self) -> Dict[str, StrategyInfo]:
        """Automatically discover available backtest strategies"""
        self.logger.info("🔍 Discovering available backtest strategies...")

        # Path to backtests directory
        backtests_path = os.path.join(WORKER_PATH, 'forex_scanner', 'backtests')

        if not os.path.exists(backtests_path):
            self.logger.warning(f"❌ Backtests directory not found: {backtests_path}")
            return {}

        strategies_found = 0

        # Scan for backtest files
        for filename in os.listdir(backtests_path):
            if filename.startswith('backtest_') and filename.endswith('.py') and filename != 'backtest_base.py':
                try:
                    strategy_name = filename.replace('backtest_', '').replace('.py', '')
                    strategy_info = self._analyze_strategy_file(backtests_path, filename, strategy_name)

                    if strategy_info:
                        self.strategies[strategy_name] = strategy_info
                        strategies_found += 1
                        self.logger.info(f"✅ Discovered strategy: {strategy_info.display_name}")

                except Exception as e:
                    self.logger.error(f"❌ Error analyzing {filename}: {str(e)}")
                    continue

        self.logger.info(f"🎯 Discovery complete: {strategies_found} strategies found")
        return self.strategies

    def _analyze_strategy_file(self, base_path: str, filename: str, strategy_name: str) -> Optional[StrategyInfo]:
        """Analyze a strategy file to extract information"""
        try:
            # Read the file to extract metadata
            file_path = os.path.join(base_path, filename)

            with open(file_path, 'r') as f:
                content = f.read()

            # Extract description from docstring or comments
            description = self._extract_description(content, strategy_name)

            # Create strategy info
            strategy_info = StrategyInfo(
                name=strategy_name,
                display_name=strategy_name.replace('_', ' ').title(),
                description=description,
                module_path=f"forex_scanner.backtests.backtest_{strategy_name}",
                class_name=self._guess_class_name(strategy_name),
                parameters=self._extract_parameters(content)
            )

            return strategy_info

        except Exception as e:
            self.logger.error(f"Error analyzing {filename}: {str(e)}")
            return None

    def _extract_description(self, content: str, strategy_name: str) -> str:
        """Extract description from file content"""
        lines = content.split('\n')

        # Look for docstring at the top
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                description_lines = []
                for j in range(i+1, len(lines)):
                    if '"""' in lines[j] or "'''" in lines[j]:
                        break
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        description_lines.append(lines[j].strip())

                if description_lines:
                    return ' '.join(description_lines[:2])  # First 2 non-empty lines

        # Fallback to strategy name
        return f"{strategy_name.replace('_', ' ').title()} trading strategy"

    def _guess_class_name(self, strategy_name: str) -> str:
        """Guess the main class name for a strategy"""
        # Most backtests follow the pattern: StrategyBacktest or just use the base class
        return f"{strategy_name.title().replace('_', '')}Backtest"

    def _extract_parameters(self, content: str) -> Dict[str, Any]:
        """Extract configurable parameters from file content"""
        parameters = {}

        # Look for common parameter patterns in the code
        lines = content.split('\n')
        for line in lines:
            line = line.strip()

            # Look for argparse parameters
            if 'add_argument' in line and '--' in line:
                try:
                    # Extract parameter name
                    if '--epic' in line:
                        parameters['epic'] = {"type": "select", "default": "CS.D.EURUSD.MINI.IP"}
                    elif '--days' in line:
                        parameters['days'] = {"type": "number", "default": 7, "min": 1, "max": 30}
                    elif '--timeframe' in line:
                        parameters['timeframe'] = {"type": "select", "default": "15m", "options": ["5m", "15m", "1h"]}
                    elif '--show-signals' in line:
                        parameters['show_signals'] = {"type": "boolean", "default": True}
                except:
                    continue

        # Add default parameters if none found
        if not parameters:
            parameters = {
                'epic': {"type": "select", "default": "CS.D.EURUSD.MINI.IP"},
                'days': {"type": "number", "default": 7, "min": 1, "max": 30},
                'timeframe': {"type": "select", "default": "15m", "options": ["5m", "15m", "1h"]},
                'show_signals': {"type": "boolean", "default": True}
            }

        return parameters

    def register_adapter(self, strategy_name: str, adapter: BaseStrategyAdapter):
        """Register a strategy adapter"""
        self.adapters[strategy_name] = adapter
        self.logger.info(f"✅ Registered adapter for {strategy_name}")

    def get_strategy(self, strategy_name: str) -> Optional[StrategyInfo]:
        """Get strategy information by name"""
        return self.strategies.get(strategy_name)

    def get_adapter(self, strategy_name: str) -> Optional[BaseStrategyAdapter]:
        """Get strategy adapter by name"""
        return self.adapters.get(strategy_name)

    def list_strategies(self) -> List[str]:
        """List all available strategy names"""
        return list(self.strategies.keys())


class BacktestRunner:
    """Unified backtest execution engine"""

    def __init__(self, registry: BacktestRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, config: BacktestConfig, progress_callback=None) -> BacktestResult:
        """Execute a backtest with the given configuration"""
        start_time = datetime.now()

        try:
            self.logger.info(f"🚀 Starting backtest: {config.strategy_name}")

            # Get strategy adapter
            adapter = self.registry.get_adapter(config.strategy_name)
            if not adapter:
                # Try to create a generic adapter
                adapter = self._create_generic_adapter(config.strategy_name)
                if not adapter:
                    raise ValueError(f"No adapter found for strategy: {config.strategy_name}")

            # Update progress
            if progress_callback:
                progress_callback(10, "Initializing strategy...")

            # Validate parameters
            if not adapter.validate_parameters(config.parameters):
                raise ValueError("Invalid parameters provided")

            if progress_callback:
                progress_callback(20, "Validating parameters...")

            # Execute backtest
            if progress_callback:
                progress_callback(30, "Running backtest...")

            result = adapter.run_backtest(config)

            if progress_callback:
                progress_callback(90, "Processing results...")

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            if progress_callback:
                progress_callback(100, "Complete!")

            self.logger.info(f"✅ Backtest complete: {result.total_signals} signals in {execution_time:.2f}s")
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self.logger.error(f"❌ Backtest failed: {error_msg}")
            self.logger.error(traceback.format_exc())

            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=0,
                signals=[],
                performance_metrics={},
                execution_time=execution_time,
                success=False,
                error_message=error_msg
            )

    def _create_generic_adapter(self, strategy_name: str) -> Optional[BaseStrategyAdapter]:
        """Create a generic adapter for legacy backtest classes"""
        try:
            from .strategy_adapters.generic_adapter import GenericStrategyAdapter

            strategy_info = self.registry.get_strategy(strategy_name)
            if strategy_info:
                adapter = GenericStrategyAdapter(strategy_name, strategy_info)
                self.registry.register_adapter(strategy_name, adapter)
                return adapter
        except ImportError:
            self.logger.warning(f"Could not create generic adapter for {strategy_name}")

        return None


class BacktestResultProcessor:
    """Processes and standardizes backtest results"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and standardize signal data"""
        processed_signals = []

        for signal in signals:
            try:
                processed_signal = self._standardize_signal(signal)
                processed_signals.append(processed_signal)
            except Exception as e:
                self.logger.warning(f"Error processing signal: {e}")
                continue

        return processed_signals

    def _standardize_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize a single signal to common format"""
        standardized = {
            'timestamp': signal.get('timestamp'),
            'epic': signal.get('epic', ''),
            'signal_type': signal.get('signal_type', 'UNKNOWN'),
            'direction': signal.get('direction', ''),
            'entry_price': float(signal.get('entry_price', 0)),
            'confidence': float(signal.get('confidence', 0)),
            'strategy': signal.get('strategy', ''),
            'timeframe': signal.get('timeframe', ''),
            'spread_pips': float(signal.get('spread_pips', 0)),
            'max_profit_pips': float(signal.get('max_profit_pips', 0)),
            'max_loss_pips': float(signal.get('max_loss_pips', 0)),
            'profit_loss_ratio': float(signal.get('profit_loss_ratio', 0)),
        }

        # Add optional fields if available
        optional_fields = [
            'stop_loss_pips', 'take_profit_pips', 'risk_reward_ratio',
            'volume', 'lookback_bars', 'market_condition'
        ]

        for field in optional_fields:
            if field in signal:
                standardized[field] = signal[field]

        return standardized

    def calculate_performance_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not signals:
            return {}

        total_signals = len(signals)

        # Profit/Loss analysis
        profitable_signals = [s for s in signals if s.get('max_profit_pips', 0) > s.get('max_loss_pips', 0)]
        win_rate = len(profitable_signals) / total_signals if total_signals > 0 else 0

        # Confidence analysis
        avg_confidence = sum(s.get('confidence', 0) for s in signals) / total_signals if total_signals > 0 else 0

        # Profit metrics
        total_profit = sum(s.get('max_profit_pips', 0) for s in signals)
        total_loss = sum(s.get('max_loss_pips', 0) for s in signals)
        avg_profit = total_profit / total_signals if total_signals > 0 else 0

        # Risk/Reward
        avg_rr = sum(s.get('profit_loss_ratio', 0) for s in signals) / total_signals if total_signals > 0 else 0

        return {
            'total_signals': total_signals,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'avg_profit_pips': avg_profit,
            'total_profit_pips': total_profit,
            'total_loss_pips': total_loss,
            'avg_risk_reward': avg_rr,
            'profitable_signals': len(profitable_signals),
            'losing_signals': total_signals - len(profitable_signals)
        }


# Global registry instance
_backtest_registry = None

def get_backtest_registry() -> BacktestRegistry:
    """Get the global backtest registry instance"""
    global _backtest_registry
    if _backtest_registry is None:
        _backtest_registry = BacktestRegistry()
        _backtest_registry.discover_strategies()
    return _backtest_registry


def get_backtest_runner() -> BacktestRunner:
    """Get a backtest runner instance"""
    registry = get_backtest_registry()
    return BacktestRunner(registry)


def get_result_processor() -> BacktestResultProcessor:
    """Get a result processor instance"""
    return BacktestResultProcessor()