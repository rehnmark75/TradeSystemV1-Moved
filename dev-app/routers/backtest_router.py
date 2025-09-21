"""
Backtest API Router
Provides API endpoints to execute existing backtest strategies
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import os
import sys
import logging
import json
from datetime import datetime

# Add worker app to path
sys.path.insert(0, '/app')
WORKER_PATH = '/app/worker/app'
if WORKER_PATH not in sys.path:
    sys.path.insert(0, WORKER_PATH)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])
logger = logging.getLogger(__name__)

# Request/Response models
class BacktestRequest(BaseModel):
    strategy_name: str
    epic: str = "CS.D.EURUSD.MINI.IP"
    days: int = 7
    timeframe: str = "15m"
    show_signals: bool = True
    parameters: Dict[str, Any] = {}

class BacktestResponse(BaseModel):
    success: bool
    strategy_name: str
    epic: str
    timeframe: str
    total_signals: int
    signals: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None

class StrategyInfo(BaseModel):
    name: str
    display_name: str
    description: str
    parameters: Dict[str, Any]
    available: bool

# Strategy discovery and execution
class BacktestExecutor:
    """Executor for running existing backtest strategies"""

    def __init__(self):
        self.available_strategies = {}
        self._discover_strategies()

    def _discover_strategies(self):
        """Discover available backtest strategies"""
        try:
            # Define known strategies with their configurations
            strategies = {
                'ema': {
                    'module': 'forex_scanner.backtests.backtest_ema',
                    'class': 'EMABacktest',
                    'display_name': 'Enhanced EMA Strategy',
                    'description': 'EMA crossover strategy with smart money analysis and database optimization',
                    'parameters': {
                        'ema_config': {
                            'type': 'select',
                            'options': ['aggressive', 'conservative', 'scalping', None],
                            'default': None,
                            'description': 'EMA configuration preset'
                        },
                        'min_confidence': {
                            'type': 'number',
                            'default': 0.7,
                            'min': 0.1,
                            'max': 1.0,
                            'description': 'Minimum confidence threshold'
                        },
                        'enable_smart_money': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Enable Smart Money Concepts analysis'
                        },
                        'use_optimal_parameters': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Use database-optimized parameters'
                        }
                    }
                },
                'macd': {
                    'module': 'forex_scanner.backtests.backtest_macd',
                    'class': 'EnhancedMACDBacktest',
                    'display_name': 'Enhanced MACD Strategy',
                    'description': 'MACD oscillator strategy with smart money analysis and database optimization',
                    'parameters': {
                        'min_confidence': {
                            'type': 'number',
                            'default': 0.7,
                            'min': 0.1,
                            'max': 1.0,
                            'description': 'Minimum confidence threshold'
                        },
                        'enable_smart_money': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Enable Smart Money Concepts analysis'
                        },
                        'enable_forex_integration': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Enable forex-specific optimizations'
                        },
                        'use_optimal_parameters': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Use database-optimized parameters'
                        }
                    }
                },
                'combined': {
                    'module': 'forex_scanner.backtests.backtest_combined',
                    'class': 'CombinedBacktest',
                    'display_name': 'Combined Strategy',
                    'description': 'Multi-strategy approach combining EMA and MACD signals',
                    'parameters': {
                        'combination_mode': {
                            'type': 'select',
                            'options': ['consensus', 'weighted'],
                            'default': 'consensus',
                            'description': 'Strategy combination method'
                        },
                        'min_confidence': {
                            'type': 'number',
                            'default': 0.7,
                            'min': 0.1,
                            'max': 1.0,
                            'description': 'Minimum confidence threshold'
                        }
                    }
                },
                'bb_supertrend': {
                    'module': 'forex_scanner.backtests.backtest_bb_supertrend',
                    'class': 'BBSupertrendBacktest',
                    'display_name': 'Bollinger Bands + Supertrend',
                    'description': 'Combined Bollinger Bands and Supertrend strategy',
                    'parameters': {
                        'bb_config': {
                            'type': 'select',
                            'options': ['default', 'tight', 'wide'],
                            'default': 'default',
                            'description': 'Bollinger Bands configuration'
                        }
                    }
                },
                'kama': {
                    'module': 'forex_scanner.backtests.backtest_kama',
                    'class': 'KAMABacktest',
                    'display_name': 'KAMA Strategy',
                    'description': 'Kaufman Adaptive Moving Average strategy',
                    'parameters': {
                        'kama_period': {
                            'type': 'number',
                            'default': 14,
                            'min': 5,
                            'max': 50,
                            'description': 'KAMA calculation period'
                        }
                    }
                },
                'smc': {
                    'module': 'forex_scanner.backtests.backtest_smc',
                    'class': 'SMCBacktest',
                    'display_name': 'Smart Money Concepts',
                    'description': 'Pure Smart Money Concepts trading strategy',
                    'parameters': {
                        'structure_detection': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Enable market structure detection'
                        }
                    }
                }
            }

            # Test availability of each strategy
            for name, config in strategies.items():
                try:
                    # Try to import the module
                    import importlib
                    module = importlib.import_module(config['module'])
                    strategy_class = getattr(module, config['class'], None)

                    if strategy_class:
                        self.available_strategies[name] = {
                            **config,
                            'available': True,
                            'strategy_class': strategy_class
                        }
                        logger.info(f"✅ Strategy available: {config['display_name']}")
                    else:
                        logger.warning(f"⚠️ Class {config['class']} not found in {config['module']}")

                except ImportError as e:
                    logger.warning(f"⚠️ Strategy {name} not available: {e}")
                except Exception as e:
                    logger.error(f"❌ Error checking strategy {name}: {e}")

            logger.info(f"🎯 Strategy discovery complete: {len(self.available_strategies)} available")

        except Exception as e:
            logger.error(f"❌ Strategy discovery failed: {e}")

    def get_available_strategies(self) -> Dict[str, StrategyInfo]:
        """Get list of available strategies"""
        strategies = {}
        for name, config in self.available_strategies.items():
            strategies[name] = StrategyInfo(
                name=name,
                display_name=config['display_name'],
                description=config['description'],
                parameters=config['parameters'],
                available=config.get('available', False)
            )
        return strategies

    def execute_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """Execute a backtest using the specified strategy"""
        start_time = datetime.now()

        try:
            if request.strategy_name not in self.available_strategies:
                raise HTTPException(
                    status_code=404,
                    detail=f"Strategy '{request.strategy_name}' not found"
                )

            strategy_config = self.available_strategies[request.strategy_name]
            if not strategy_config.get('available', False):
                raise HTTPException(
                    status_code=503,
                    detail=f"Strategy '{request.strategy_name}' is not available"
                )

            # Get the strategy class
            strategy_class = strategy_config['strategy_class']

            # Initialize the strategy
            strategy_instance = strategy_class()

            # Prepare parameters for the strategy
            params = {
                'epic': request.epic,
                'days': request.days,
                'timeframe': request.timeframe,
                'show_signals': request.show_signals,
                **request.parameters
            }

            # Execute the backtest
            logger.info(f"🚀 Executing {request.strategy_name} backtest: {request.epic}")
            success = strategy_instance.run_backtest(**params)

            if not success:
                raise Exception("Backtest execution returned False")

            # Extract results from the strategy instance
            signals = self._extract_signals(strategy_instance)
            performance_metrics = self._extract_performance_metrics(strategy_instance)

            execution_time = (datetime.now() - start_time).total_seconds()

            return BacktestResponse(
                success=True,
                strategy_name=request.strategy_name,
                epic=request.epic,
                timeframe=request.timeframe,
                total_signals=len(signals),
                signals=signals,
                performance_metrics=performance_metrics,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Backtest execution failed: {str(e)}")

            return BacktestResponse(
                success=False,
                strategy_name=request.strategy_name,
                epic=request.epic,
                timeframe=request.timeframe,
                total_signals=0,
                signals=[],
                performance_metrics={},
                execution_time=execution_time,
                error_message=str(e)
            )

    def _extract_signals(self, strategy_instance) -> List[Dict[str, Any]]:
        """Extract signals from the strategy instance"""
        signals = []

        try:
            # Check various attributes where signals might be stored
            signal_sources = [
                'all_signals', 'signals', 'backtest_signals',
                'enhanced_signals', '_signals'
            ]

            for attr in signal_sources:
                if hasattr(strategy_instance, attr):
                    signal_data = getattr(strategy_instance, attr)
                    if isinstance(signal_data, list) and signal_data:
                        signals = signal_data
                        logger.info(f"✅ Found {len(signals)} signals in {attr}")
                        break

            # If no signals found, try to access through sub-components
            if not signals and hasattr(strategy_instance, 'signal_analyzer'):
                analyzer = strategy_instance.signal_analyzer
                if hasattr(analyzer, 'signals'):
                    signals = analyzer.signals

            # Convert to serializable format
            serializable_signals = []
            for signal in signals:
                if isinstance(signal, dict):
                    # Convert datetime objects to strings
                    serializable_signal = {}
                    for key, value in signal.items():
                        if isinstance(value, datetime):
                            serializable_signal[key] = value.isoformat()
                        elif hasattr(value, 'item'):  # numpy types
                            serializable_signal[key] = value.item()
                        else:
                            serializable_signal[key] = value
                    serializable_signals.append(serializable_signal)

            return serializable_signals

        except Exception as e:
            logger.warning(f"⚠️ Error extracting signals: {e}")
            return []

    def _extract_performance_metrics(self, strategy_instance) -> Dict[str, Any]:
        """Extract performance metrics from the strategy instance"""
        try:
            # Check if the instance has a performance analyzer
            if hasattr(strategy_instance, 'performance_analyzer'):
                analyzer = strategy_instance.performance_analyzer
                if hasattr(analyzer, 'get_metrics'):
                    return analyzer.get_metrics()

            # Try to extract from all_signals
            signals = self._extract_signals(strategy_instance)
            if signals:
                return self._calculate_basic_metrics(signals)

            return {}

        except Exception as e:
            logger.warning(f"⚠️ Error extracting performance metrics: {e}")
            return {}

    def _calculate_basic_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic performance metrics from signals"""
        if not signals:
            return {}

        total_signals = len(signals)
        profitable_signals = [s for s in signals if s.get('max_profit_pips', 0) > s.get('max_loss_pips', 0)]
        win_rate = len(profitable_signals) / total_signals if total_signals > 0 else 0

        avg_confidence = sum(s.get('confidence', 0) for s in signals) / total_signals if total_signals > 0 else 0
        total_profit = sum(s.get('max_profit_pips', 0) for s in signals)
        avg_profit = total_profit / total_signals if total_signals > 0 else 0

        return {
            'total_signals': total_signals,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'avg_profit_pips': avg_profit,
            'profitable_signals': len(profitable_signals),
            'losing_signals': total_signals - len(profitable_signals)
        }


# Global executor instance
_executor = None

def get_executor() -> BacktestExecutor:
    """Get the global backtest executor instance"""
    global _executor
    if _executor is None:
        _executor = BacktestExecutor()
    return _executor


# API Endpoints
@router.get("/strategies", response_model=Dict[str, StrategyInfo])
async def get_available_strategies():
    """Get list of available backtest strategies"""
    try:
        executor = get_executor()
        strategies = executor.get_available_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Execute a backtest with the specified parameters"""
    try:
        executor = get_executor()
        result = executor.execute_backtest(request)
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        executor = get_executor()
        strategies = executor.get_available_strategies()
        return {
            "status": "healthy",
            "available_strategies": len(strategies),
            "strategies": list(strategies.keys())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }