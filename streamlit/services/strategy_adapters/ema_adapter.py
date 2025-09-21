"""
EMA Strategy Adapter
Specific adapter for the Enhanced EMA Strategy Backtest
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add worker path
# Calculate correct path: streamlit/services/strategy_adapters/ema_adapter.py -> TradeSystemV1/worker/app
CURRENT_FILE = os.path.abspath(__file__)
STRATEGY_ADAPTERS_DIR = os.path.dirname(CURRENT_FILE)  # strategy_adapters/
SERVICES_DIR = os.path.dirname(STRATEGY_ADAPTERS_DIR)  # services/
STREAMLIT_DIR = os.path.dirname(SERVICES_DIR)  # streamlit/
PROJECT_ROOT = os.path.dirname(STREAMLIT_DIR)  # TradeSystemV1/
WORKER_PATH = os.path.join(PROJECT_ROOT, 'worker', 'app')

if WORKER_PATH not in sys.path:
    sys.path.insert(0, WORKER_PATH)

from ..backtest_service import BaseStrategyAdapter, StrategyInfo, BacktestConfig, BacktestResult


class EMAStrategyAdapter(BaseStrategyAdapter):
    """Adapter for the Enhanced EMA Strategy"""

    def __init__(self):
        super().__init__("ema")
        self._backtest_instance = None

    def get_strategy_info(self) -> StrategyInfo:
        """Return information about the EMA strategy"""
        return StrategyInfo(
            name="ema",
            display_name="Enhanced EMA Strategy",
            description="EMA crossover strategy with smart money analysis and database optimization",
            module_path="forex_scanner.backtests.backtest_ema",
            class_name="EMABacktest",
            parameters={
                'epic': {
                    "type": "select",
                    "default": "CS.D.EURUSD.MINI.IP",
                    "description": "Trading pair to analyze"
                },
                'days': {
                    "type": "number",
                    "default": 7,
                    "min": 1,
                    "max": 30,
                    "description": "Number of days of historical data"
                },
                'timeframe': {
                    "type": "select",
                    "default": "15m",
                    "options": ["5m", "15m", "1h"],
                    "description": "Chart timeframe"
                },
                'show_signals': {
                    "type": "boolean",
                    "default": True,
                    "description": "Display individual signals"
                },
                'ema_config': {
                    "type": "select",
                    "default": None,
                    "options": ["aggressive", "conservative", "scalping", None],
                    "description": "EMA configuration preset"
                },
                'min_confidence': {
                    "type": "number",
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "description": "Minimum confidence threshold"
                },
                'enable_smart_money': {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable Smart Money Concepts analysis"
                },
                'use_optimal_parameters': {
                    "type": "boolean",
                    "default": True,
                    "description": "Use database-optimized parameters"
                }
            }
        )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for the EMA strategy"""
        return {
            'epic': 'CS.D.EURUSD.MINI.IP',
            'days': 7,
            'timeframe': '15m',
            'show_signals': True,
            'ema_config': None,
            'min_confidence': 0.7,
            'enable_smart_money': False,
            'use_optimal_parameters': True
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters for the EMA strategy"""
        try:
            # Validate epic
            if 'epic' in parameters and not isinstance(parameters['epic'], str):
                self.logger.error("Epic must be a string")
                return False

            # Validate days
            if 'days' in parameters:
                days = parameters['days']
                if not isinstance(days, int) or days < 1 or days > 30:
                    self.logger.error("Days must be an integer between 1 and 30")
                    return False

            # Validate timeframe
            if 'timeframe' in parameters:
                if parameters['timeframe'] not in ['5m', '15m', '1h']:
                    self.logger.error("Timeframe must be one of: 5m, 15m, 1h")
                    return False

            # Validate confidence
            if 'min_confidence' in parameters:
                conf = parameters['min_confidence']
                if not isinstance(conf, (int, float)) or conf < 0.1 or conf > 1.0:
                    self.logger.error("Min confidence must be between 0.1 and 1.0")
                    return False

            # Validate EMA config
            if 'ema_config' in parameters and parameters['ema_config'] is not None:
                if parameters['ema_config'] not in ['aggressive', 'conservative', 'scalping']:
                    self.logger.error("EMA config must be one of: aggressive, conservative, scalping")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Execute the EMA backtest"""
        start_time = datetime.now()

        try:
            # Load the EMA backtest class
            self._load_backtest_class()

            if not self._backtest_instance:
                raise Exception("Could not load EMA backtest class")

            # Prepare parameters
            params = self._prepare_parameters(config)

            # Execute the backtest
            self.logger.info(f"🚀 Running EMA backtest for {config.epic}")

            success = self._backtest_instance.run_backtest(**params)

            if not success:
                raise Exception("EMA backtest execution returned False")

            # Extract results
            signals = self._extract_signals()
            chart_data = self._extract_chart_data()

            # Process results
            from ..backtest_service import get_result_processor
            processor = get_result_processor()
            processed_signals = processor.process_signals(signals)
            performance_metrics = processor.calculate_performance_metrics(processed_signals)

            # Add EMA-specific metrics
            performance_metrics.update(self._get_ema_specific_metrics())

            execution_time = (datetime.now() - start_time).total_seconds()

            return BacktestResult(
                strategy_name=config.strategy_name,
                epic=config.epic,
                timeframe=config.timeframe,
                total_signals=len(processed_signals),
                signals=processed_signals,
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                success=True,
                chart_data=chart_data
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ EMA backtest failed: {str(e)}")

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
        """Load the EMA backtest class"""
        try:
            if self._backtest_instance is not None:
                return

            from forex_scanner.backtests.backtest_ema import EMABacktest
            self._backtest_instance = EMABacktest()
            self.logger.info("✅ EMA backtest class loaded successfully")

        except ImportError as e:
            self.logger.error(f"❌ Could not import EMA backtest class: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error loading EMA backtest class: {e}")
            raise

    def _prepare_parameters(self, config: BacktestConfig) -> Dict[str, Any]:
        """Prepare parameters for the EMA backtest"""
        params = {
            'epic': config.epic,
            'days': config.parameters.get('days', 7),
            'timeframe': config.timeframe,
            'show_signals': config.parameters.get('show_signals', True),
            'ema_config': config.parameters.get('ema_config'),
            'min_confidence': config.parameters.get('min_confidence', 0.7),
            'enable_smart_money': config.parameters.get('enable_smart_money', False),
            'use_optimal_parameters': config.parameters.get('use_optimal_parameters', True)
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        return params

    def _extract_signals(self) -> List[Dict[str, Any]]:
        """Extract signals from the EMA backtest instance"""
        signals = []

        try:
            # Check various attributes where signals might be stored
            signal_sources = [
                'all_signals', 'signals', 'backtest_signals',
                'enhanced_signals', 'ema_signals', '_signals'
            ]

            for attr in signal_sources:
                if hasattr(self._backtest_instance, attr):
                    signal_data = getattr(self._backtest_instance, attr)
                    if isinstance(signal_data, list) and signal_data:
                        signals = signal_data
                        self.logger.info(f"✅ Found {len(signals)} signals in {attr}")
                        break

            # If no signals found, try to access through sub-components
            if not signals and hasattr(self._backtest_instance, 'signal_analyzer'):
                analyzer = self._backtest_instance.signal_analyzer
                if hasattr(analyzer, 'signals'):
                    signals = analyzer.signals

            # Convert signals to standard format
            standardized_signals = []
            for signal in signals:
                if isinstance(signal, dict):
                    # Ensure required fields are present
                    std_signal = {
                        'timestamp': signal.get('timestamp'),
                        'epic': signal.get('epic', ''),
                        'signal_type': signal.get('signal_type', 'EMA'),
                        'direction': signal.get('direction', ''),
                        'entry_price': float(signal.get('entry_price', 0)),
                        'confidence': float(signal.get('confidence', 0)),
                        'strategy': 'EMA',
                        'timeframe': signal.get('timeframe', ''),
                        'spread_pips': float(signal.get('spread_pips', 0)),
                        'max_profit_pips': float(signal.get('max_profit_pips', 0)),
                        'max_loss_pips': float(signal.get('max_loss_pips', 0)),
                        'profit_loss_ratio': float(signal.get('profit_loss_ratio', 0)),
                    }

                    # Add any additional EMA-specific fields
                    ema_fields = ['ema_config', 'smart_money_score', 'market_structure']
                    for field in ema_fields:
                        if field in signal:
                            std_signal[field] = signal[field]

                    standardized_signals.append(std_signal)

            return standardized_signals

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting signals: {e}")
            return []

    def _extract_chart_data(self) -> Optional[Any]:
        """Extract chart data from the EMA backtest instance"""
        try:
            # Look for chart data in various places
            chart_sources = [
                'chart_data', 'df', 'price_data', 'historical_data',
                'enhanced_data', '_chart_data'
            ]

            for attr in chart_sources:
                if hasattr(self._backtest_instance, attr):
                    chart_data = getattr(self._backtest_instance, attr)
                    if chart_data is not None:
                        self.logger.info(f"✅ Found chart data in {attr}")
                        return chart_data

            # Try to get data from sub-components
            if hasattr(self._backtest_instance, 'data_fetcher'):
                # The data might be in the data fetcher's last query
                pass

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting chart data: {e}")

        return None

    def _get_ema_specific_metrics(self) -> Dict[str, Any]:
        """Get EMA-specific performance metrics"""
        metrics = {}

        try:
            # Check if smart money analysis was used
            if hasattr(self._backtest_instance, 'smart_money_stats'):
                smc_stats = self._backtest_instance.smart_money_stats
                metrics['smart_money_analysis'] = {
                    'signals_analyzed': smc_stats.get('signals_analyzed', 0),
                    'signals_enhanced': smc_stats.get('signals_enhanced', 0),
                    'analysis_failures': smc_stats.get('analysis_failures', 0),
                    'enhancement_rate': (
                        smc_stats.get('signals_enhanced', 0) / max(smc_stats.get('signals_analyzed', 1), 1)
                    )
                }

            # Check if optimal parameters were used
            if hasattr(self._backtest_instance, 'strategy') and self._backtest_instance.strategy:
                strategy = self._backtest_instance.strategy
                if hasattr(strategy, 'use_optimal_parameters'):
                    metrics['used_optimal_parameters'] = strategy.use_optimal_parameters

        except Exception as e:
            self.logger.warning(f"⚠️ Error getting EMA-specific metrics: {e}")

        return metrics