#!/usr/bin/env python3
"""
Task Worker API Server
FastAPI server to expose existing backtest strategies via HTTP API
"""

import sys
import os
from pathlib import Path

# Add forex_scanner to path
current_dir = Path(__file__).parent
forex_scanner_path = current_dir / "forex_scanner"
if str(forex_scanner_path) not in sys.path:
    sys.path.insert(0, str(forex_scanner_path))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
import traceback

# Import the backtest router from the dev-app (we'll adapt it)
# But first let's create a simple version that works with the forex_scanner

app = FastAPI(
    title="Task Worker Backtest API",
    description="API for executing existing forex scanner backtest strategies",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class BacktestRequest(BaseModel):
    strategy_name: str
    epic: Optional[str] = None  # None means run on all epics from config
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

class BacktestExecutor:
    """Executor for running existing backtest strategies from forex_scanner"""

    def __init__(self):
        self.available_strategies = {}
        self._discover_strategies()

    def _discover_strategies(self):
        """Discover available backtest strategies"""
        try:
            # Define known strategies with their complete backtest configurations
            strategies = {
                'ema': {
                    'module': 'backtests.backtest_ema',
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
                    'module': 'backtests.backtest_macd',
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
                    'module': 'backtests.backtest_combined',
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
                    'module': 'backtests.backtest_bb_supertrend',
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
                    'module': 'backtests.backtest_kama',
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
                    'module': 'backtests.backtest_smc',
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
                },
                'ichimoku': {
                    'module': 'backtests.backtest_ichimoku_enhanced',
                    'class': 'EnhancedIchimokuStrategyBacktest',
                    'display_name': 'Ichimoku Cloud Strategy (Enhanced)',
                    'description': 'Enhanced Ichimoku Kinko Hyo analysis with unified framework integration',
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
                        'cloud_breakout_only': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Only trade cloud breakouts (ignore TK line crossovers)'
                        }
                    }
                },
                'zero_lag': {
                    'module': 'backtests.backtest_zero_lag',
                    'class': 'ZeroLagBacktest',
                    'display_name': 'Zero Lag + Squeeze Momentum',
                    'description': 'Zero lag EMA strategy with squeeze momentum indicator for enhanced timing',
                    'parameters': {
                        'min_confidence': {
                            'type': 'number',
                            'default': 0.7,
                            'min': 0.1,
                            'max': 1.0,
                            'description': 'Minimum confidence threshold'
                        },
                        'enable_squeeze_momentum': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Enable squeeze momentum indicator'
                        },
                        'enable_smart_money': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Enable Smart Money Concepts analysis'
                        }
                    }
                },
                'mean_reversion': {
                    'module': 'backtests.backtest_mean_reversion',
                    'class': 'MeanReversionBacktest',
                    'display_name': 'Mean Reversion Strategy (Enhanced)',
                    'description': 'Multi-oscillator confluence mean reversion strategy with enhanced framework integration',
                    'parameters': {
                        'min_confidence': {
                            'type': 'number',
                            'default': 0.7,
                            'min': 0.1,
                            'max': 1.0,
                            'description': 'Minimum confidence threshold'
                        },
                        'oscillator_confluence': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Require multiple oscillator confirmation'
                        },
                        'use_optimal_parameters': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Use database-optimized parameters'
                        }
                    }
                },
                'ema_enhanced': {
                    'module': 'backtests.backtest_ema_enhanced',
                    'class': 'EnhancedEmaBacktest',
                    'display_name': 'EMA Strategy (Enhanced Framework)',
                    'description': 'Enhanced EMA strategy with unified framework, market intelligence, and standardized output',
                    'parameters': {
                        'ema_config': {
                            'type': 'select',
                            'options': ['aggressive', 'conservative', 'scalping'],
                            'default': 'aggressive',
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
                'macd_enhanced': {
                    'module': 'backtests.backtest_macd_enhanced',
                    'class': 'EnhancedMacdBacktest',
                    'display_name': 'MACD Strategy (Enhanced Framework)',
                    'description': 'Enhanced MACD strategy with unified framework, market intelligence, and standardized output',
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
                'ichimoku_enhanced': {
                    'module': 'backtests.backtest_ichimoku_enhanced',
                    'class': 'EnhancedIchimokuStrategyBacktest',
                    'display_name': 'Ichimoku Strategy (Enhanced Framework)',
                    'description': 'Enhanced Ichimoku strategy with unified framework, market intelligence, and standardized output',
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
                        'cloud_breakout_only': {
                            'type': 'boolean',
                            'default': False,
                            'description': 'Only trade cloud breakouts (ignore TK line crossovers)'
                        }
                    }
                },
                'all': {
                    'module': 'backtests.backtest_all',
                    'class': 'AllStrategiesBacktest',
                    'display_name': 'All Strategies Combined',
                    'description': 'Run all available strategies and combine results for comprehensive analysis',
                    'parameters': {
                        'min_confidence': {
                            'type': 'number',
                            'default': 0.7,
                            'min': 0.1,
                            'max': 1.0,
                            'description': 'Minimum confidence threshold'
                        },
                        'strategy_selection': {
                            'type': 'select',
                            'options': ['all', 'top_performers', 'trend_following', 'mean_reversion'],
                            'default': 'all',
                            'description': 'Which strategy subset to run'
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
        """Execute a backtest using the specified strategy with enhanced format support"""
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

            # Get the backtest class
            backtest_class = strategy_config['strategy_class']

            # Initialize the backtest class
            backtest_instance = backtest_class()
            logger.info(f"✅ Initialized {request.strategy_name} backtest class")

            # Prepare parameters for the backtest
            backtest_params = {
                'epic': request.epic,  # None means all epics
                'days': request.days,
                'timeframe': request.timeframe,
                'show_signals': request.show_signals
            }

            # Add strategy-specific parameters
            import inspect
            run_backtest_method = getattr(backtest_instance, 'run_backtest')
            method_signature = inspect.signature(run_backtest_method)
            supported_params = set(method_signature.parameters.keys())

            logger.info(f"ℹ️ Strategy {request.strategy_name} supports parameters: {supported_params}")

            # Add user parameters to backtest
            for param_name, param_value in request.parameters.items():
                if param_name in supported_params:
                    backtest_params[param_name] = param_value
                    logger.info(f"✅ Added supported parameter: {param_name}={param_value}")

            # Pass user parameters to backtest for ParameterManager
            if hasattr(backtest_instance, 'parameter_manager') and backtest_instance.parameter_manager:
                backtest_params['user_parameters'] = request.parameters

            logger.info(f"📊 Backtest parameters: {backtest_params}")

            # Execute the backtest
            logger.info(f"🚀 Running {request.strategy_name} backtest on {request.epic}")
            result = backtest_instance.run_backtest(**backtest_params)

            # Handle both new StandardBacktestResult and legacy boolean returns
            if hasattr(result, 'success'):
                # New enhanced format - StandardBacktestResult
                logger.info("✅ Received StandardBacktestResult format")

                return BacktestResponse(
                    success=result.success,
                    strategy_name=result.strategy_name,
                    epic=result.epic,
                    timeframe=result.timeframe,
                    total_signals=result.total_signals,
                    signals=result.signals,  # Uses properties for compatibility
                    performance_metrics=result.performance_metrics,
                    execution_time=result.execution_time,
                    error_message=result.error_message
                )

            elif isinstance(result, bool):
                # Legacy format - boolean return
                logger.info("⚠️ Received legacy boolean format, extracting signals manually")

                if not result:
                    raise Exception("Backtest execution returned False")

                # Extract signals using legacy method
                signals = self._extract_signals_from_backtest(backtest_instance)
                performance_metrics = self._extract_performance_metrics_from_backtest(backtest_instance)
                execution_time = (datetime.now() - start_time).total_seconds()

                return BacktestResponse(
                    success=True,
                    strategy_name=request.strategy_name,
                    epic=request.epic or "ALL_EPICS",
                    timeframe=request.timeframe,
                    total_signals=len(signals),
                    signals=signals,
                    performance_metrics=performance_metrics,
                    execution_time=execution_time
                )

            else:
                raise Exception(f"Unexpected result type from backtest: {type(result)}")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Backtest execution failed: {str(e)}")

            return BacktestResponse(
                success=False,
                strategy_name=request.strategy_name,
                epic=request.epic or "ALL_EPICS",
                timeframe=request.timeframe,
                total_signals=0,
                signals=[],
                performance_metrics={},
                execution_time=execution_time,
                error_message=str(e)
            )

    def _extract_signals_from_backtest(self, backtest_instance) -> List[Dict[str, Any]]:
        """Extract signals from complete backtest instance"""
        signals = []

        try:
            # The backtest classes store signals in different ways, let's check all possibilities
            signal_sources = [
                'all_signals',      # Primary location for EMA backtest
                'signals',          # Alternative location
                'backtest_signals', # Another alternative
                'enhanced_signals', # For enhanced strategies
                '_signals'          # Private attribute version
            ]

            # Check instance attributes
            for attr in signal_sources:
                if hasattr(backtest_instance, attr):
                    signal_data = getattr(backtest_instance, attr)
                    if isinstance(signal_data, list) and signal_data:
                        signals = signal_data
                        logger.info(f"✅ Found {len(signals)} signals in {attr} attribute")
                        break
                    elif signal_data is not None:
                        logger.warning(f"⚠️ {attr} attribute exists but is not a list or is empty: {type(signal_data)}")

            # If no signals found in attributes, check if the backtest has stored them during execution
            # Some backtests might store signals in analyzers or other components
            if not signals:
                logger.info("🔍 Checking signal analyzer and performance analyzer for signals")

                # Check signal analyzer
                if hasattr(backtest_instance, 'signal_analyzer'):
                    analyzer = backtest_instance.signal_analyzer
                    for attr in ['signals', 'all_signals', 'enhanced_signals']:
                        if hasattr(analyzer, attr):
                            signal_data = getattr(analyzer, attr)
                            if isinstance(signal_data, list) and signal_data:
                                signals = signal_data
                                logger.info(f"✅ Found {len(signals)} signals in signal_analyzer.{attr}")
                                break

            # Try accessing through strategy instance within backtest
            if not signals and hasattr(backtest_instance, 'strategy'):
                logger.info("🔍 Trying to extract signals from strategy within backtest instance")
                strategy = backtest_instance.strategy
                for attr in signal_sources:
                    if hasattr(strategy, attr):
                        signal_data = getattr(strategy, attr)
                        if isinstance(signal_data, list) and signal_data:
                            signals = signal_data
                            logger.info(f"✅ Found {len(signals)} signals in strategy.{attr}")
                            break

            # Debug: Show what attributes are available if no signals found
            if not signals:
                available_attrs = [attr for attr in dir(backtest_instance) if not attr.startswith('__')]
                logger.warning(f"⚠️ No signals found. Available attributes: {available_attrs}")

            # Convert to serializable format
            serializable_signals = []
            for signal in signals:
                if isinstance(signal, dict):
                    # Convert datetime objects to strings and handle numpy types
                    serializable_signal = {}
                    for key, value in signal.items():
                        if isinstance(value, datetime):
                            serializable_signal[key] = value.isoformat()
                        elif hasattr(value, 'item'):  # numpy types
                            try:
                                serializable_signal[key] = value.item()
                            except:
                                serializable_signal[key] = str(value)
                        elif isinstance(value, (int, float, str, bool)) or value is None:
                            serializable_signal[key] = value
                        else:
                            # Convert other types to string as fallback
                            serializable_signal[key] = str(value)
                    serializable_signals.append(serializable_signal)

            logger.info(f"📊 Extracted {len(serializable_signals)} serializable signals from backtest")
            return serializable_signals

        except Exception as e:
            logger.error(f"❌ Error extracting signals from backtest: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_performance_metrics_from_backtest(self, backtest_instance) -> Dict[str, Any]:
        """Extract performance metrics from complete backtest instance"""
        try:
            # Check if the backtest instance has a performance analyzer
            if hasattr(backtest_instance, 'performance_analyzer'):
                analyzer = backtest_instance.performance_analyzer
                if hasattr(analyzer, 'get_metrics'):
                    return analyzer.get_metrics()

            # Try to extract from signals
            signals = self._extract_signals_from_backtest(backtest_instance)
            if signals:
                return self._calculate_basic_metrics(signals)

            return {}

        except Exception as e:
            logger.warning(f"⚠️ Error extracting performance metrics from backtest: {e}")
            return {}

    def _extract_signals(self, strategy_instance) -> List[Dict[str, Any]]:
        """DEPRECATED: Extract signals from the strategy instance (kept for compatibility)"""
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
        """DEPRECATED: Extract performance metrics from the strategy instance (kept for compatibility)"""
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
@app.get("/api/backtest/strategies", response_model=Dict[str, StrategyInfo])
async def get_available_strategies():
    """Get list of available backtest strategies"""
    try:
        executor = get_executor()
        strategies = executor.get_available_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Execute a backtest with the specified parameters"""
    try:
        executor = get_executor()
        result = executor.execute_backtest(request)
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/health")
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Task Worker Backtest API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/api/backtest/strategies",
            "/api/backtest/run",
            "/api/backtest/health"
        ]
    }

if __name__ == "__main__":
    import uvicorn

    # Log startup
    logger.info("🚀 Starting Task Worker Backtest API Server")
    logger.info("📊 Real forex scanner strategies will be available")

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8007,
        log_level="info"
    )