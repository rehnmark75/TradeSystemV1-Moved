"""
Worker Backtest Service
Connects to the worker container API to execute existing strategies
"""

import os
import logging
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class BacktestConfig:
    """Configuration for a worker backtest run"""
    strategy_name: str
    epic: Optional[str] = None  # None means run on all epics from config
    days: int = 7
    timeframe: str = "15m"
    show_signals: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Worker backtest result"""
    strategy_name: str
    epic: Optional[str]  # Can be None for all epics runs
    timeframe: str
    total_signals: int
    signals: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    chart_data: Optional[Any] = None

@dataclass
class StrategyInfo:
    """Information about an available strategy"""
    name: str
    display_name: str
    description: str
    parameters: Dict[str, Any]

class WorkerBacktestService:
    """Service that connects to worker container to execute existing strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.worker_base_url = self._get_worker_url()
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minute timeout for backtests

        # Add required gateway header for worker container access
        self.session.headers.update({
            'x-apim-gateway': 'verified'
        })

    def _get_worker_url(self) -> str:
        """Get the worker container URL"""
        # In docker-compose, containers can reach each other by name
        # We'll use the task-worker container for real strategies
        return "http://task-worker:8007"

    def get_available_strategies(self) -> Dict[str, StrategyInfo]:
        """Get available strategies from the worker container"""
        try:
            url = f"{self.worker_base_url}/api/backtest/strategies"
            self.logger.info(f"🔍 Fetching strategies from: {url}")

            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            strategies = {}

            for name, strategy_data in data.items():
                strategies[name] = StrategyInfo(
                    name=strategy_data['name'],
                    display_name=strategy_data['display_name'],
                    description=strategy_data['description'],
                    parameters=strategy_data['parameters']
                )

            self.logger.info(f"✅ Retrieved {len(strategies)} strategies from worker")
            return strategies

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Failed to fetch strategies from worker: {e}")
            return self._get_fallback_strategies()
        except Exception as e:
            self.logger.error(f"❌ Error getting strategies: {e}")
            return self._get_fallback_strategies()

    def _get_fallback_strategies(self) -> Dict[str, StrategyInfo]:
        """Fallback strategies if worker is not available"""
        self.logger.warning("⚠️ Using fallback strategies (worker not available)")
        return {
            'ema': StrategyInfo(
                name='ema',
                display_name='Enhanced EMA Strategy (Offline)',
                description='EMA crossover strategy - worker connection unavailable',
                parameters={
                    'epic': {'type': 'select', 'default': 'CS.D.EURUSD.MINI.IP'},
                    'days': {'type': 'number', 'default': 7, 'min': 1, 'max': 30},
                    'timeframe': {'type': 'select', 'default': '15m', 'options': ['5m', '15m', '1h']}
                }
            ),
            'macd': StrategyInfo(
                name='macd',
                display_name='Enhanced MACD Strategy (Offline)',
                description='MACD oscillator strategy - worker connection unavailable',
                parameters={
                    'epic': {'type': 'select', 'default': 'CS.D.EURUSD.MINI.IP'},
                    'days': {'type': 'number', 'default': 7, 'min': 1, 'max': 30},
                    'timeframe': {'type': 'select', 'default': '15m', 'options': ['5m', '15m', '1h']}
                }
            )
        }

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Execute a backtest using the worker container"""
        start_time = datetime.now()

        try:
            url = f"{self.worker_base_url}/api/backtest/run"
            self.logger.info(f"🚀 Running backtest: {config.strategy_name} on {config.epic}")

            # Prepare request payload
            payload = {
                "strategy_name": config.strategy_name,
                "epic": config.epic,
                "days": config.days,
                "timeframe": config.timeframe,
                "show_signals": config.show_signals,
                "parameters": config.parameters
            }

            self.logger.info(f"📤 Sending request to: {url}")
            self.logger.info(f"📊 Payload: {json.dumps(payload, indent=2)}")

            # Send request to worker
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()

            if data.get('success', False):
                self.logger.info(f"✅ Backtest completed: {data.get('total_signals', 0)} signals")

                return BacktestResult(
                    strategy_name=data['strategy_name'],
                    epic=data['epic'],
                    timeframe=data['timeframe'],
                    total_signals=data['total_signals'],
                    signals=data['signals'],
                    performance_metrics=data['performance_metrics'],
                    execution_time=data['execution_time'],
                    success=True
                )
            else:
                error_msg = data.get('error_message', 'Unknown error')
                self.logger.error(f"❌ Worker backtest failed: {error_msg}")

                return BacktestResult(
                    strategy_name=config.strategy_name,
                    epic=config.epic,
                    timeframe=config.timeframe,
                    total_signals=0,
                    signals=[],
                    performance_metrics={},
                    execution_time=data.get('execution_time', 0),
                    success=False,
                    error_message=error_msg
                )

        except requests.exceptions.Timeout:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = "Backtest timed out after 5 minutes"
            self.logger.error(f"⏰ {error_msg}")

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

        except requests.exceptions.ConnectionError:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = "Cannot connect to worker container. Is it running?"
            self.logger.error(f"🔗 {error_msg}")

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

        except requests.exceptions.RequestException as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Request failed: {str(e)}"
            self.logger.error(f"📡 {error_msg}")

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

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")

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

    def check_worker_health(self) -> Dict[str, Any]:
        """Check if the worker container is healthy and responsive"""
        try:
            url = f"{self.worker_base_url}/api/backtest/health"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            self.logger.info(f"✅ Worker health check: {data.get('status', 'unknown')}")
            return data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Worker health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "available_strategies": 0
            }

    def get_worker_info(self) -> Dict[str, Any]:
        """Get information about the worker container"""
        health = self.check_worker_health()
        return {
            "worker_url": self.worker_base_url,
            "health_status": health.get("status", "unknown"),
            "available_strategies": health.get("available_strategies", 0),
            "strategy_names": health.get("strategies", []),
            "connection_timeout": self.session.timeout
        }


# Global instance
_worker_backtest_service = None

def get_worker_backtest_service() -> WorkerBacktestService:
    """Get the global worker backtest service instance"""
    global _worker_backtest_service
    if _worker_backtest_service is None:
        _worker_backtest_service = WorkerBacktestService()
    return _worker_backtest_service