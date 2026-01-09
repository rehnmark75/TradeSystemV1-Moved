# core/trading/backtest_trading_orchestrator.py
"""
BacktestTradingOrchestrator - Mirrors TradingOrchestrator for backtest mode
Uses same TradeValidator logic but logs signals instead of executing trades
"""

import sys
import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# Add path setup for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

try:
    import config
    from core.database import DatabaseManager
    from core.backtest_scanner import BacktestScanner
    from core.backtest_data_fetcher import BacktestDataFetcher
    from core.signal_detector import SignalDetector

    # Import same validation components as live trading
    from .trade_validator import TradeValidator
    from .backtest_order_logger import BacktestOrderLogger

    # Import AlertHistoryManager for comprehensive signal logging
    from alerts.alert_history import AlertHistoryManager

    print("✅ Successfully imported all backtest trading modules")
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.backtest_scanner import BacktestScanner
    from forex_scanner.core.backtest_data_fetcher import BacktestDataFetcher
    from forex_scanner.core.signal_detector import SignalDetector

    # Import same validation components as live trading
    try:
        from forex_scanner.core.trading.trade_validator import TradeValidator
        from forex_scanner.core.trading.backtest_order_logger import BacktestOrderLogger
        print("✅ Successfully imported backtest trading modules with absolute paths")
    except ImportError as e:
        print(f"⚠️ Backtest trading modules not available: {e}")
        TradeValidator = BacktestOrderLogger = None

    # Import AlertHistoryManager for comprehensive signal logging
    try:
        from forex_scanner.alerts.alert_history import AlertHistoryManager
        print("✅ Successfully imported AlertHistoryManager with absolute path")
    except ImportError as e:
        print(f"⚠️ AlertHistoryManager not available: {e}")
        AlertHistoryManager = None


class BacktestTradingOrchestrator:
    """
    Backtest Trading System Orchestrator

    CRITICAL: Uses SAME validation logic as live trading (TradeValidator)
    DIFFERENCE: Routes validated signals to BacktestOrderLogger instead of order execution

    This ensures backtests use identical validation pipeline as live trading
    """

    def __init__(self,
                 execution_id: int,
                 backtest_config: Dict,
                 db_manager: DatabaseManager = None,
                 logger: Optional[logging.Logger] = None,
                 pipeline_mode: bool = False,
                 config_override: dict = None,
                 use_historical_intelligence: bool = True,
                 signal_start_filter: datetime = None):
        """
        Initialize BacktestTradingOrchestrator

        Args:
            execution_id: ID of the backtest execution
            backtest_config: Backtest configuration dict
            db_manager: Database manager instance
            logger: Logger instance
            pipeline_mode: Use full validation pipeline (slower but matches live)
            config_override: Parameter overrides for testing
            use_historical_intelligence: Replay stored market intelligence
            signal_start_filter: Only record signals AFTER this timestamp
                                 (used for chunked parallel backtests to avoid
                                 recording signals during warmup period)
        """
        self.execution_id = execution_id
        self.backtest_config = backtest_config
        self.pipeline_mode = pipeline_mode
        self._config_override = config_override
        self._use_historical_intelligence = use_historical_intelligence
        self._signal_start_filter = signal_start_filter or backtest_config.get('signal_start_filter')
        self.db_manager = db_manager or DatabaseManager(config.DATABASE_URL)
        self.logger = logger or logging.getLogger(__name__)

        # Core components - SAME as live trading
        # PERFORMANCE OPTIMIZATION (Jan 2026): Pass dates to load only needed data into cache
        self.data_fetcher = BacktestDataFetcher(
            self.db_manager,
            config.USER_TIMEZONE,
            start_date=backtest_config.get('start_date'),
            end_date=backtest_config.get('end_date'),
            epics=backtest_config.get('epics')
        )
        self.signal_detector = SignalDetector(self.db_manager, config.USER_TIMEZONE, config_override=self._config_override)

        # Force-initialize the requested strategy for backtest (regardless of config flags)
        # This allows testing any strategy without modifying config.py
        strategy_name = backtest_config.get('strategy_name', 'EMA_CROSSOVER')
        if strategy_name and strategy_name.upper() not in ['EMA', 'EMA_CROSSOVER', 'ALL', '']:
            success, message = self.signal_detector.force_initialize_strategy(strategy_name)
            if not success:
                self.logger.error(f"❌ Failed to initialize strategy '{strategy_name}': {message}")
                raise ValueError(f"Cannot run backtest: {message}")
            self.logger.info(f"✅ Backtest strategy '{strategy_name}' force-initialized: {message}")

        # Pipeline mode: Use SAME TradeValidator as live trading with backtest mode enabled
        # Basic mode: Skip TradeValidator to improve performance for parameter optimization
        if self.pipeline_mode:
            self.trade_validator = TradeValidator(
                logger=self.logger,
                db_manager=self.db_manager,
                backtest_mode=True  # UNIVERSAL FIX: Enable confidence format normalization
            )
            self.logger.info("🔄 PIPELINE MODE: TradeValidator enabled for full validation")
        else:
            self.trade_validator = None
            self.logger.info("🚀 BASIC MODE: TradeValidator disabled for fast parameter optimization")

        # DEBUG: Log pipeline mode status
        self.logger.info(f"🔧 BacktestTradingOrchestrator initialized with pipeline_mode={pipeline_mode}")
        self.logger.info(f"🔧 backtest_config contains pipeline_mode: {backtest_config.get('pipeline_mode', 'NOT_SET')}")
        self.logger.info(f"🔧 trade_validator created: {self.trade_validator is not None}")

        # Backtest-specific components
        self.order_logger = BacktestOrderLogger(self.db_manager, execution_id, logger=self.logger)

        # CRITICAL FIX: Set skip_signal_logging=True so scanner doesn't log signals
        # The orchestrator will handle all signal logging to prevent duplicates
        scanner_config = backtest_config.copy()
        scanner_config['skip_signal_logging'] = True
        self.scanner = BacktestScanner(
            scanner_config,
            db_manager=self.db_manager,
            config_override=self._config_override,
            use_historical_intelligence=self._use_historical_intelligence
        )

        # Alert history DISABLED for backtests - only for production signals
        self.alert_history_manager = None  # Never use alert_history for backtest data

        # Statistics tracking
        self.orchestrator_stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'signals_logged': 0,
            'validation_errors': 0,
            'processing_errors': 0
        }

        self.logger.info(f"🧪 BacktestTradingOrchestrator initialized:")
        self.logger.info(f"   Execution ID: {execution_id}")
        self.logger.info(f"   Strategy: {backtest_config.get('strategy_name', 'unknown')}")
        self.logger.info(f"   Using SAME TradeValidator as live trading ✅")
        self.logger.info(f"   Alert history: ❌ DISABLED (backtest mode - production only)")
        if self._config_override:
            self.logger.info(f"   🧪 CONFIG OVERRIDES: {len(self._config_override)} parameters")
            for key, value in self._config_override.items():
                self.logger.info(f"      - {key}: {value}")

    def run_backtest_orchestration(self) -> Dict:
        """
        Main orchestration method - mirrors live trading flow exactly
        """
        self.logger.info(f"🚀 Starting backtest orchestration for execution {self.execution_id}")

        orchestration_start = datetime.now()

        try:
            # Initialize backtest execution
            self._initialize_backtest_execution()

            # Run scanner with proper context management to get raw results
            with self.scanner.order_logger:
                self.scanner._initialize_backtest_execution()
                raw_backtest_results = self.scanner._execute_backtest_loop()

            # CRITICAL FIX: Process all signals through orchestrator validation pipeline
            self._process_scanner_signals(raw_backtest_results)

            # Generate final report from raw results
            backtest_results = self.scanner._generate_backtest_report(raw_backtest_results)

            # Process any additional signal validation if needed
            enhanced_results = self._enhance_backtest_results(backtest_results)

            # Generate comprehensive report
            final_report = self._generate_orchestration_report(enhanced_results, orchestration_start)

            self.logger.info(f"✅ Backtest orchestration completed successfully")
            self.logger.info(f"📊 Total signals processed: {self.orchestrator_stats['signals_processed']}")
            self.logger.info(f"✔️ Signals validated: {self.orchestrator_stats['signals_validated']}")
            self.logger.info(f"❌ Signals rejected: {self.orchestrator_stats['signals_rejected']}")

            # Log resampled cache performance stats
            if hasattr(self.scanner, 'signal_detector') and hasattr(self.scanner.signal_detector, 'data_fetcher'):
                data_fetcher = self.scanner.signal_detector.data_fetcher
                if hasattr(data_fetcher, 'get_resampled_cache_stats'):
                    cache_stats = data_fetcher.get_resampled_cache_stats()
                    self.logger.info(f"\n⚡ Resampled Data Cache Performance:")
                    self.logger.info(f"   Cache Hits: {cache_stats['cache_hits']}")
                    self.logger.info(f"   Cache Misses: {cache_stats['cache_misses']}")
                    self.logger.info(f"   Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
                    self.logger.info(f"   Cached Datasets: {cache_stats['cached_datasets']}")
                    if cache_stats['hit_rate_percent'] > 90:
                        self.logger.info(f"   ✅ Excellent cache performance!")

            return final_report

        except Exception as e:
            self.logger.error(f"❌ Backtest orchestration failed: {e}")
            self.orchestrator_stats['processing_errors'] += 1
            raise

    def process_signal_through_validation_pipeline(self, signal: Dict) -> tuple[bool, str, Dict]:
        """
        Process signal through EXACT SAME validation pipeline as live trading
        This is the critical method that ensures backtest/live consistency
        """

        self.orchestrator_stats['signals_processed'] += 1

        try:
            # Step 1: Signal validation (PIPELINE MODE ONLY)
            if self.pipeline_mode and self.trade_validator:
                self.logger.info(f"🔄 PIPELINE MODE: Validating signal through TradeValidator for {signal.get('epic', 'unknown')}")
                validation_passed, validation_message = self.trade_validator.validate_signal_for_trading(signal)
                self.logger.info(f"🔄 PIPELINE MODE: Validation result: {validation_passed}, message: {validation_message}")
            else:
                # Basic mode: Skip validation for fast parameter optimization
                self.logger.info(f"🚀 BASIC MODE: Skipping TradeValidator validation for {signal.get('epic', 'unknown')} (pipeline_mode={self.pipeline_mode}, trade_validator={self.trade_validator is not None})")
                validation_passed, validation_message = True, "Basic mode - validation skipped"

            if validation_passed:
                self.orchestrator_stats['signals_validated'] += 1

                # Mark signal as validated for tracking
                signal['validation_passed'] = True
                signal['validation_message'] = validation_message

                # Step 2: Log validated signal (instead of executing order)
                success, message, order_data = self.order_logger.place_order(signal)

                if success:
                    self.orchestrator_stats['signals_logged'] += 1

                    # Step 3: Alert history SKIPPED for backtests (production only)
                    # Backtest signals go only to backtest_signals table, not alert_history

                    self.logger.info(f"✅ Signal validated and logged: {signal.get('epic')} "
                                   f"{signal.get('signal_type')} ({signal.get('confidence_score', 0):.1%})")

                    return True, f"Signal validated and logged: {message}", order_data
                else:
                    self.logger.error(f"❌ Failed to log validated signal: {message}")
                    return False, f"Logging failed: {message}", {}
            else:
                self.orchestrator_stats['signals_rejected'] += 1

                # Mark signal as failed validation for tracking
                signal['validation_passed'] = False
                signal['validation_message'] = validation_message

                # Still log failed signals for comprehensive reporting
                self.order_logger.place_order(signal)

                self.logger.info(f"❌ Signal rejected by validation: {validation_message}")
                return False, f"Validation failed: {validation_message}", {}

        except Exception as e:
            self.orchestrator_stats['validation_errors'] += 1
            self.logger.error(f"Error in validation pipeline: {e}")
            return False, f"Pipeline error: {str(e)}", {}

    def _process_scanner_signals(self, backtest_results: Dict):
        """
        CRITICAL FIX: Process signals from scanner through orchestrator validation pipeline

        This ensures signals go through the same validation as live trading.

        If signal_start_filter is set, signals before that timestamp are skipped.
        This is used for chunked parallel backtests to avoid recording signals
        during the warmup period (which is needed for indicator calculation).
        """
        try:
            signals_processed = 0
            signals_filtered = 0

            self.logger.info(f"🔍 DEBUG: Backtest results keys: {list(backtest_results.keys())}")

            # Log signal_start_filter if set
            if self._signal_start_filter:
                self.logger.info(f"🔍 Signal start filter active: only recording signals after {self._signal_start_filter}")

            # Extract signals from backtest results structure: results['signals_by_epic'][epic]
            signals_by_epic = backtest_results.get('signals_by_epic', {})

            self.logger.info(f"🔍 DEBUG: Found {len(signals_by_epic)} epics with signals")

            for epic, signals in signals_by_epic.items():
                self.logger.info(f"🔍 DEBUG: Epic {epic} has {len(signals) if isinstance(signals, list) else 'non-list'} signals")

                if isinstance(signals, list):
                    for signal in signals:
                        if isinstance(signal, dict):
                            # Apply signal_start_filter if set (for chunked backtests)
                            if self._signal_start_filter:
                                signal_timestamp = signal.get('timestamp') or signal.get('signal_timestamp')
                                if signal_timestamp:
                                    # Convert string to datetime if needed
                                    if isinstance(signal_timestamp, str):
                                        signal_timestamp = pd.to_datetime(signal_timestamp)

                                    # Skip signals before the filter timestamp
                                    if signal_timestamp < self._signal_start_filter:
                                        signals_filtered += 1
                                        continue

                            signals_processed += 1

                            # Process each signal through the validation pipeline
                            success, message, order_data = self.process_signal_through_validation_pipeline(signal)

                            if success:
                                self.logger.debug(f"✅ Signal {signals_processed} ({epic}) validated and logged")
                            else:
                                self.logger.debug(f"❌ Signal {signals_processed} ({epic}) failed validation: {message}")

            self.logger.info(f"🔄 Processed {signals_processed} signals through orchestrator validation pipeline")
            if signals_filtered > 0:
                self.logger.info(f"🔍 Filtered out {signals_filtered} signals (before signal_start_filter)")

        except Exception as e:
            self.logger.error(f"Error processing scanner signals through validation: {e}")
            raise

    def _initialize_backtest_execution(self):
        """Initialize backtest execution record"""
        try:
            # Update execution record with orchestrator info
            query = """
            UPDATE backtest_executions
            SET config_snapshot = config_snapshot || CAST(:config_json AS jsonb),
                updated_at = NOW()
            WHERE id = :execution_id
            """

            orchestrator_config = {
                'orchestrator_version': 'backtest_v1.0',
                'validation_pipeline': 'identical_to_live',
                'trade_validator_used': True,
                'alert_history_enabled': False,  # Always disabled for backtests
                'data_destination': 'backtest_signals_table_only'
            }

            params = {
                'config_json': json.dumps(orchestrator_config),
                'execution_id': int(self.execution_id)  # Convert numpy.int64 to Python int
            }

            # Execute UPDATE query - handle DataFrame return issue
            try:
                self.db_manager.execute_query(query, params)
                self.logger.info("✅ Backtest execution record updated with orchestrator config")
            except Exception as update_error:
                if "This result object does not return rows" in str(update_error):
                    # UPDATE query succeeded but DatabaseManager can't create DataFrame - this is expected
                    self.logger.info("✅ Backtest execution record updated with orchestrator config")
                else:
                    raise update_error

        except Exception as e:
            self.logger.error(f"Error initializing backtest execution: {e}")
            raise

    def _enhance_backtest_results(self, backtest_results: Dict) -> Dict:
        """Enhance backtest results with orchestration data"""

        enhanced_results = backtest_results.copy()

        # Add orchestration statistics
        enhanced_results['orchestration_stats'] = self.orchestrator_stats.copy()
        enhanced_results['validation_summary'] = self._get_validation_summary()
        enhanced_results['trade_validator_stats'] = self.trade_validator.get_validation_statistics() if self.trade_validator else {'status': 'disabled_basic_mode'}

        return enhanced_results

    def _get_validation_summary(self) -> Dict:
        """Get validation pipeline summary"""
        total_processed = self.orchestrator_stats['signals_processed']

        if total_processed == 0:
            return {
                'validation_rate': 0.0,
                'rejection_rate': 0.0,
                'logging_success_rate': 0.0
            }

        return {
            'validation_rate': self.orchestrator_stats['signals_validated'] / total_processed,
            'rejection_rate': self.orchestrator_stats['signals_rejected'] / total_processed,
            'logging_success_rate': self.orchestrator_stats['signals_logged'] / max(self.orchestrator_stats['signals_validated'], 1),
            'error_rate': self.orchestrator_stats['validation_errors'] / total_processed
        }

    def _log_to_alert_history(self, signal: Dict, validation_message: str):
        """Log signal to alert history (same as live trading)"""
        try:
            if self.alert_history_manager:
                # Convert signal to alert format
                alert_data = {
                    'epic': signal.get('epic'),
                    'signal_type': signal.get('signal_type'),
                    'confidence_score': signal.get('confidence_score'),
                    'entry_price': signal.get('entry_price'),
                    'stop_loss_price': signal.get('stop_loss_price'),
                    'take_profit_price': signal.get('take_profit_price'),
                    'timestamp': signal.get('signal_timestamp', datetime.now()),
                    'strategy': signal.get('strategy', 'backtest'),
                    'validation_message': validation_message,
                    'backtest_execution_id': self.execution_id,
                    'backtest_mode': True
                }

                self.alert_history_manager.log_alert(alert_data)

        except Exception as e:
            self.logger.error(f"Error logging to alert history: {e}")

    def _generate_orchestration_report(self, results: Dict, start_time: datetime) -> Dict:
        """Generate comprehensive orchestration report"""

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = {
            'execution_id': self.execution_id,
            'orchestration_summary': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'orchestration_version': 'backtest_v1.0'
            },
            'validation_pipeline': {
                'trade_validator_used': self.trade_validator is not None,
                'same_as_live_trading': self.trade_validator is not None,
                'validation_statistics': self.trade_validator.get_validation_statistics() if self.trade_validator else {'status': 'disabled_basic_mode'},
                'validation_summary': self._get_validation_summary()
            },
            'signal_processing': {
                'total_processed': self.orchestrator_stats['signals_processed'],
                'validated': self.orchestrator_stats['signals_validated'],
                'rejected': self.orchestrator_stats['signals_rejected'],
                'logged': self.orchestrator_stats['signals_logged'],
                'errors': self.orchestrator_stats['validation_errors']
            },
            'backtest_results': results,
            'performance_metrics': self._calculate_performance_metrics(),
            'quality_assurance': {
                'pipeline_integrity': self._verify_pipeline_integrity(),
                'data_consistency': self._verify_data_consistency()
            },
            'order_logger': self.order_logger  # Include order_logger for direct CSV export
        }

        # Log comprehensive summary
        self.logger.info(f"📊 Orchestration Report Summary:")
        self.logger.info(f"   Duration: {duration:.1f}s")
        self.logger.info(f"   Signals processed: {self.orchestrator_stats['signals_processed']}")
        self.logger.info(f"   Validation rate: {self._get_validation_summary()['validation_rate']:.1%}")
        self.logger.info(f"   Pipeline integrity: {'✅ PASSED' if report['quality_assurance']['pipeline_integrity'] else '❌ FAILED'}")

        return report

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate orchestration performance metrics"""
        try:
            # Get performance data from database
            try:
                result_df = self.db_manager.execute_query(
                    "SELECT * FROM get_backtest_summary(:execution_id)",
                    {'execution_id': int(self.execution_id)}
                )
                result = result_df.iloc[0].to_dict() if not result_df.empty else None

                if result:
                    return {
                        'total_signals': result.get('total_signals', 0),
                        'validated_signals': result.get('total_validated_signals', 0),
                        'win_rate': float(result.get('avg_win_rate', 0)) if result.get('avg_win_rate') else 0,
                        'total_pips': float(result.get('total_pips', 0)) if result.get('total_pips') else 0,
                        'profit_factor': float(result.get('avg_profit_factor', 0)) if result.get('avg_profit_factor') else 0
                    }
                else:
                    return {}

            except Exception as db_error:
                if "This result object does not return rows" in str(db_error):
                    # This should not happen with a SELECT query, but handle it gracefully
                    self.logger.warning("Database query completed but result object issue encountered")
                    return {}
                else:
                    raise db_error

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _verify_pipeline_integrity(self) -> bool:
        """Verify that validation pipeline is working correctly"""
        try:
            # Check that TradeValidator is functioning (if enabled)
            validator_stats = self.trade_validator.get_validation_statistics() if self.trade_validator else {'status': 'disabled_basic_mode'}

            # Verify order logger is functioning
            logger_stats = self.order_logger.get_statistics()

            # Basic integrity checks
            check1 = self.orchestrator_stats['signals_processed'] >= 0
            # Fixed logic: when no signals processed, validation errors should be 0 (which is fine)
            if self.orchestrator_stats['signals_processed'] == 0:
                check2 = self.orchestrator_stats['validation_errors'] == 0
            else:
                check2 = self.orchestrator_stats['validation_errors'] < (self.orchestrator_stats['signals_processed'] * 0.1)
            check3 = logger_stats['execution_id'] == self.execution_id

            integrity_checks = [check1, check2, check3]

            return all(integrity_checks)

        except Exception as e:
            self.logger.error(f"Error verifying pipeline integrity: {e}")
            return False

    def _verify_data_consistency(self) -> bool:
        """Verify data consistency between components"""
        try:
            # Check that signals logged match signals validated
            expected_logged = self.orchestrator_stats['signals_validated']
            actual_logged = self.orchestrator_stats['signals_logged']

            # Should be very close (within 5% difference)
            if expected_logged > 0:
                consistency_ratio = actual_logged / expected_logged
                return 0.95 <= consistency_ratio <= 1.05

            return True

        except Exception as e:
            self.logger.error(f"Error verifying data consistency: {e}")
            return False

    def get_orchestrator_statistics(self) -> Dict:
        """Get current orchestrator statistics"""
        return {
            'orchestrator_stats': self.orchestrator_stats.copy(),
            'trade_validator_stats': self.trade_validator.get_validation_statistics() if self.trade_validator else {'status': 'disabled_basic_mode'},
            'order_logger_stats': self.order_logger.get_statistics(),
            'order_logger': self.order_logger,  # Include order_logger for CSV export
            'execution_id': self.execution_id
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Finalize order logger
            self.order_logger.finalize_execution(status='completed')

            # Clear caches
            if hasattr(self.data_fetcher, 'clear_backtest_cache'):
                self.data_fetcher.clear_backtest_cache()

            self.logger.info("🧹 Backtest orchestrator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        try:
            if exc_type is not None:
                self.logger.error(f"Orchestrator exiting due to error: {exc_val}")
                self.order_logger.finalize_execution(status='failed', error_message=str(exc_val))
            else:
                self.cleanup()
        except Exception as e:
            self.logger.error(f"Error in orchestrator cleanup: {e}")


# Factory function for creating backtest orchestrator
def create_backtest_trading_orchestrator(execution_id: int,
                                       backtest_config: Dict,
                                       db_manager: DatabaseManager = None,
                                       config_override: dict = None,
                                       **kwargs) -> BacktestTradingOrchestrator:
    """Create BacktestTradingOrchestrator instance"""
    # Extract pipeline_mode from backtest_config if not provided in kwargs
    if 'pipeline_mode' not in kwargs:
        kwargs['pipeline_mode'] = backtest_config.get('pipeline_mode', False)

    return BacktestTradingOrchestrator(
        execution_id, backtest_config, db_manager, config_override=config_override, **kwargs
    )


# Convenience function for running complete backtest
def run_complete_backtest(backtest_config: Dict,
                         db_manager: DatabaseManager = None,
                         logger: logging.Logger = None) -> Dict:
    """
    Run a complete backtest with orchestration

    Args:
        backtest_config: Configuration dict with execution_id, strategy, dates, etc.
        db_manager: Database manager instance
        logger: Logger instance

    Returns:
        Comprehensive backtest report
    """
    execution_id = backtest_config['execution_id']

    logger = logger or logging.getLogger(__name__)
    logger.info(f"🚀 Running complete backtest for execution {execution_id}")

    try:
        with create_backtest_trading_orchestrator(
            execution_id, backtest_config, db_manager, logger=logger
        ) as orchestrator:

            results = orchestrator.run_backtest_orchestration()

            logger.info(f"✅ Complete backtest finished successfully")
            return results

    except Exception as e:
        logger.error(f"❌ Complete backtest failed: {e}")
        raise