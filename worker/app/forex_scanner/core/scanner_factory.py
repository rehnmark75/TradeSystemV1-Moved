# core/scanner_factory.py
"""
Scanner Factory - Mode switching architecture for live vs backtest scanners
Provides unified interface for creating scanners in different modes
"""

import logging
from typing import Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum

try:
    import config
    from core.database import DatabaseManager
    from core.scanner import IntelligentForexScanner
    from core.backtest_scanner import BacktestScanner
    from core.trading.trading_orchestrator import TradingOrchestrator
    from core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.scanner import IntelligentForexScanner
    from forex_scanner.core.backtest_scanner import BacktestScanner
    from forex_scanner.core.trading.trading_orchestrator import TradingOrchestrator
    from forex_scanner.core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator


class ScannerMode(Enum):
    """Scanner operation modes"""
    LIVE = "live"
    BACKTEST = "backtest"


class ScannerFactory:
    """
    Factory for creating scanners in different modes
    Ensures consistent configuration and component selection
    """

    def __init__(self, db_manager: DatabaseManager = None, logger: logging.Logger = None):
        self.db_manager = db_manager or DatabaseManager(config.DATABASE_URL)
        self.logger = logger or logging.getLogger(__name__)

    def create_scanner(self,
                      mode: Union[ScannerMode, str],
                      scanner_config: Dict[str, Any] = None,
                      backtest_config: Dict[str, Any] = None,
                      **kwargs) -> Union[IntelligentForexScanner, BacktestScanner]:
        """
        Create scanner instance based on mode

        Args:
            mode: Scanner mode (LIVE or BACKTEST)
            scanner_config: Scanner configuration parameters
            backtest_config: Backtest-specific configuration (required for backtest mode)
            **kwargs: Additional parameters passed to scanner constructor

        Returns:
            Scanner instance appropriate for the mode
        """

        # Normalize mode
        if isinstance(mode, str):
            mode = ScannerMode(mode.lower())

        # Set default configurations
        scanner_config = scanner_config or {}
        default_config = self._get_default_scanner_config()
        merged_config = {**default_config, **scanner_config, **kwargs}

        self.logger.info(f"🏭 Creating {mode.value} scanner")

        if mode == ScannerMode.LIVE:
            return self._create_live_scanner(merged_config)

        elif mode == ScannerMode.BACKTEST:
            if not backtest_config:
                raise ValueError("backtest_config is required for backtest mode")

            return self._create_backtest_scanner(merged_config, backtest_config)

        else:
            raise ValueError(f"Unsupported scanner mode: {mode}")

    def create_trading_orchestrator(self,
                                  mode: Union[ScannerMode, str],
                                  orchestrator_config: Dict[str, Any] = None,
                                  backtest_config: Dict[str, Any] = None,
                                  **kwargs) -> Union[TradingOrchestrator, BacktestTradingOrchestrator]:
        """
        Create trading orchestrator based on mode

        Args:
            mode: Operating mode (LIVE or BACKTEST)
            orchestrator_config: Orchestrator configuration
            backtest_config: Backtest-specific configuration (required for backtest mode)
            **kwargs: Additional parameters

        Returns:
            Trading orchestrator appropriate for the mode
        """

        # Normalize mode
        if isinstance(mode, str):
            mode = ScannerMode(mode.lower())

        orchestrator_config = orchestrator_config or {}
        merged_config = {**orchestrator_config, **kwargs}

        self.logger.info(f"🏭 Creating {mode.value} trading orchestrator")

        if mode == ScannerMode.LIVE:
            return self._create_live_orchestrator(merged_config)

        elif mode == ScannerMode.BACKTEST:
            if not backtest_config:
                raise ValueError("backtest_config is required for backtest mode")

            execution_id = backtest_config.get('execution_id')
            if not execution_id:
                raise ValueError("execution_id is required in backtest_config")

            return self._create_backtest_orchestrator(execution_id, backtest_config, merged_config)

        else:
            raise ValueError(f"Unsupported orchestrator mode: {mode}")

    def _create_live_scanner(self, config_params: Dict[str, Any]) -> IntelligentForexScanner:
        """Create live scanner with optimized configuration"""

        live_config = {
            'db_manager': self.db_manager,
            'intelligence_mode': 'live_only',
            'use_bid_adjustment': config_params.get('use_bid_adjustment', True),
            'enable_deduplication': config_params.get('enable_deduplication', True),
            'enable_smart_money': config_params.get('enable_smart_money', True),
            'use_signal_processor': config_params.get('use_signal_processor', True),
            **config_params
        }

        scanner = IntelligentForexScanner(**live_config)

        self.logger.info("✅ Live scanner created with configuration:")
        self.logger.info(f"   Epic list: {len(scanner.epic_list)} pairs")
        self.logger.info(f"   Min confidence: {scanner.min_confidence:.1%}")
        self.logger.info(f"   Intelligence mode: {live_config['intelligence_mode']}")
        self.logger.info(f"   Deduplication: {'✅' if live_config['enable_deduplication'] else '❌'}")
        self.logger.info(f"   Smart money: {'✅' if live_config['enable_smart_money'] else '❌'}")

        return scanner

    def _create_backtest_scanner(self,
                                config_params: Dict[str, Any],
                                backtest_config: Dict[str, Any]) -> BacktestScanner:
        """Create backtest scanner with optimized configuration"""

        # Validate backtest configuration
        self._validate_backtest_config(backtest_config)

        backtest_scanner_config = {
            'db_manager': self.db_manager,
            'intelligence_mode': 'backtest_consistent',
            'enable_deduplication': config_params.get('enable_deduplication', False),  # Usually disabled for backtests
            'enable_smart_money': config_params.get('enable_smart_money', True),
            'use_signal_processor': config_params.get('use_signal_processor', True),
            **config_params
        }

        scanner = BacktestScanner(backtest_config, **backtest_scanner_config)

        self.logger.info("✅ Backtest scanner created with configuration:")
        self.logger.info(f"   Execution ID: {backtest_config['execution_id']}")
        self.logger.info(f"   Strategy: {backtest_config['strategy_name']}")
        self.logger.info(f"   Period: {backtest_config['start_date']} to {backtest_config['end_date']}")
        self.logger.info(f"   Epic list: {len(scanner.epic_list)} pairs")
        self.logger.info(f"   Timeframe: {backtest_config.get('timeframe', '15m')}")
        self.logger.info(f"   Smart money: {'✅' if backtest_scanner_config['enable_smart_money'] else '❌'}")

        return scanner

    def _create_live_orchestrator(self, config_params: Dict[str, Any]) -> TradingOrchestrator:
        """Create live trading orchestrator"""

        try:
            orchestrator = TradingOrchestrator(
                db_manager=self.db_manager,
                logger=self.logger,
                **config_params
            )

            self.logger.info("✅ Live trading orchestrator created")
            return orchestrator

        except Exception as e:
            self.logger.error(f"Error creating live orchestrator: {e}")
            raise

    def _create_backtest_orchestrator(self,
                                    execution_id: int,
                                    backtest_config: Dict[str, Any],
                                    config_params: Dict[str, Any]) -> BacktestTradingOrchestrator:
        """Create backtest trading orchestrator"""

        try:
            # Extract pipeline_mode from backtest_config to ensure it's passed
            pipeline_mode = backtest_config.get('pipeline_mode', False)
            orchestrator = BacktestTradingOrchestrator(
                execution_id=execution_id,
                backtest_config=backtest_config,
                db_manager=self.db_manager,
                logger=self.logger,
                pipeline_mode=pipeline_mode,
                **config_params
            )

            self.logger.info("✅ Backtest trading orchestrator created")
            return orchestrator

        except Exception as e:
            self.logger.error(f"Error creating backtest orchestrator: {e}")
            raise

    def _get_default_scanner_config(self) -> Dict[str, Any]:
        """Get default scanner configuration"""

        return {
            'epic_list': getattr(config, 'EPIC_LIST', []),
            'min_confidence': getattr(config, 'MIN_CONFIDENCE', 0.7),
            'scan_interval': getattr(config, 'SCAN_INTERVAL_SECONDS', 60),
            'spread_pips': getattr(config, 'SPREAD_PIPS', 1.5),
            'user_timezone': getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm'),
        }

    def _validate_backtest_config(self, backtest_config: Dict[str, Any]):
        """Validate backtest configuration parameters"""

        required_fields = ['execution_id', 'strategy_name', 'start_date', 'end_date']

        for field in required_fields:
            if field not in backtest_config:
                raise ValueError(f"Missing required backtest configuration field: {field}")

        # Validate dates
        start_date = backtest_config['start_date']
        end_date = backtest_config['end_date']

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        # Validate execution_id
        execution_id = backtest_config['execution_id']
        if not isinstance(execution_id, int) or execution_id <= 0:
            raise ValueError("execution_id must be a positive integer")

        self.logger.debug("✅ Backtest configuration validation passed")

    def create_backtest_execution(self,
                                strategy_name: str,
                                start_date: datetime,
                                end_date: datetime,
                                epics: list = None,
                                timeframe: str = '15m',
                                execution_name: str = None) -> int:
        """
        Create a new backtest execution record and return execution_id

        Args:
            strategy_name: Name of the trading strategy
            start_date: Backtest start date
            end_date: Backtest end date
            epics: List of epics to test (defaults to config.EPIC_LIST)
            timeframe: Trading timeframe
            execution_name: Optional custom name for the execution

        Returns:
            execution_id: ID of the created execution record
        """

        epics = epics or getattr(config, 'EPIC_LIST', [])
        execution_name = execution_name or f"{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        try:
            # Convert Python lists to PostgreSQL array strings
            epics_array = epics if isinstance(epics, list) else [epics] if epics else []
            timeframes_array = [timeframe] if isinstance(timeframe, str) else timeframe

            # Format as PostgreSQL array strings
            epics_pg_array = '{' + ','.join(f'"{epic}"' for epic in epics_array) + '}'
            timeframes_pg_array = '{' + ','.join(f'"{tf}"' for tf in timeframes_array) + '}'

            query = """
            INSERT INTO backtest_executions (
                execution_name, strategy_name, data_start_date, data_end_date,
                epics_tested, timeframes, status
            ) VALUES (
                :execution_name, :strategy_name, :start_date, :end_date,
                CAST(:epics_tested AS text[]), CAST(:timeframes AS text[]), 'running'
            ) RETURNING id
            """

            params = {
                'execution_name': execution_name,
                'strategy_name': strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'epics_tested': epics_pg_array,
                'timeframes': timeframes_pg_array
            }

            result_df = self.db_manager.execute_query(query, params)

            if result_df.empty:
                raise Exception("Failed to create backtest execution - no result returned")

            execution_id = result_df.iloc[0]['id']

            self.logger.info(f"✅ Created backtest execution {execution_id}:")
            self.logger.info(f"   Name: {execution_name}")
            self.logger.info(f"   Strategy: {strategy_name}")
            self.logger.info(f"   Period: {start_date} to {end_date}")
            self.logger.info(f"   Epics: {len(epics)} pairs")

            return execution_id

        except Exception as e:
            self.logger.error(f"Error creating backtest execution: {e}")
            raise

    def run_complete_backtest_workflow(self,
                                     strategy_name: str,
                                     start_date: datetime,
                                     end_date: datetime,
                                     epics: list = None,
                                     timeframe: str = '15m',
                                     execution_name: str = None,
                                     scanner_config: Dict = None,
                                     orchestrator_config: Dict = None) -> Dict:
        """
        Complete backtest workflow: create execution, run backtest, return results

        This is a high-level convenience method that handles the entire backtest process
        """

        self.logger.info(f"🚀 Starting complete backtest workflow for {strategy_name}")

        try:
            # Step 1: Create backtest execution
            execution_id = self.create_backtest_execution(
                strategy_name, start_date, end_date, epics, timeframe, execution_name
            )

            # Step 2: Create backtest configuration
            backtest_config = {
                'execution_id': execution_id,
                'strategy_name': strategy_name,
                'start_date': start_date,
                'end_date': end_date,
                'epics': epics or getattr(config, 'EPIC_LIST', []),
                'timeframe': timeframe
            }

            # Step 3: Create and run backtest orchestrator
            with self.create_trading_orchestrator(
                ScannerMode.BACKTEST,
                orchestrator_config,
                backtest_config
            ) as orchestrator:

                results = orchestrator.run_backtest_orchestration()

            self.logger.info(f"✅ Complete backtest workflow finished for execution {execution_id}")
            return results

        except Exception as e:
            self.logger.error(f"❌ Backtest workflow failed: {e}")
            raise

    def get_factory_info(self) -> Dict[str, Any]:
        """Get factory configuration and status information"""

        return {
            'supported_modes': [mode.value for mode in ScannerMode],
            'default_config': self._get_default_scanner_config(),
            'database_connected': self.db_manager is not None,
            'factory_version': '1.0.0'
        }


# Global factory instance
_global_factory = None


def get_scanner_factory(db_manager: DatabaseManager = None,
                       logger: logging.Logger = None) -> ScannerFactory:
    """Get singleton scanner factory instance"""

    global _global_factory

    if _global_factory is None:
        _global_factory = ScannerFactory(db_manager, logger)

    return _global_factory


# Convenience functions
def create_live_scanner(scanner_config: Dict = None, **kwargs) -> IntelligentForexScanner:
    """Convenience function to create live scanner"""
    factory = get_scanner_factory()
    return factory.create_scanner(ScannerMode.LIVE, scanner_config, **kwargs)


def create_backtest_scanner(backtest_config: Dict,
                          scanner_config: Dict = None,
                          **kwargs) -> BacktestScanner:
    """Convenience function to create backtest scanner"""
    factory = get_scanner_factory()
    return factory.create_scanner(ScannerMode.BACKTEST, scanner_config, backtest_config, **kwargs)


def run_quick_backtest(strategy_name: str,
                      start_date: datetime,
                      end_date: datetime,
                      **kwargs) -> Dict:
    """Convenience function to run a quick backtest"""
    factory = get_scanner_factory()
    return factory.run_complete_backtest_workflow(
        strategy_name, start_date, end_date, **kwargs
    )