# core/trading/backtest_order_logger.py
"""
BacktestOrderLogger - Replaces OrderManager for backtest mode
Logs signals to database instead of executing trades
Maintains same interface as OrderManager for compatibility
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from decimal import Decimal

try:
    import config
    from core.database import DatabaseManager
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager


class BacktestOrderLogger:
    """
    Logs trading signals to backtest database tables instead of executing orders
    Maintains same interface as OrderManager for drop-in replacement
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 execution_id: int,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.execution_id = execution_id
        self.logger = logger or logging.getLogger(__name__)

        # Statistics
        self.signals_logged = 0
        self.validation_passed = 0
        self.validation_failed = 0

        # Finalization tracking
        self._finalized = False

        # Initialize execution if not exists
        self._ensure_execution_exists()

    def _ensure_execution_exists(self):
        """Ensure the execution_id exists in backtest_executions table"""
        try:
            query = """
            SELECT id FROM backtest_executions WHERE id = :execution_id
            """
            result_df = self.db_manager.execute_query(query, {'execution_id': int(self.execution_id)})

            if result_df.empty:
                self.logger.warning(f"Execution ID {self.execution_id} not found in database")
        except Exception as e:
            self.logger.error(f"Error checking execution existence: {e}")

    def place_order(self, signal: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict]]:
        """
        Log trading signal instead of placing order
        Returns same format as OrderManager for compatibility
        """
        try:
            # Log the signal to backtest_signals table
            success = self._log_backtest_signal(signal)

            if success:
                self.signals_logged += 1
                if signal.get('validation_passed', False):
                    self.validation_passed += 1
                else:
                    self.validation_failed += 1

                self.logger.info(f"📊 Logged backtest signal: {signal.get('epic')} {signal.get('signal_type')} "
                               f"({signal.get('confidence_score', 0):.1%})")

                # Return success with mock order data
                mock_order = {
                    'order_id': f"backtest_{self.execution_id}_{self.signals_logged}",
                    'status': 'logged',
                    'epic': signal.get('epic'),
                    'direction': signal.get('signal_type'),
                    'size': signal.get('position_size', 1.0),
                    'level': signal.get('entry_price'),
                    'stop_level': signal.get('stop_loss_price'),
                    'limit_level': signal.get('take_profit_price'),
                    'logged_at': datetime.now(timezone.utc).isoformat()
                }

                return True, "Signal logged successfully", mock_order
            else:
                return False, "Failed to log signal", None

        except Exception as e:
            self.logger.error(f"Error logging backtest signal: {e}")
            return False, f"Error logging signal: {str(e)}", None

    def _log_backtest_signal(self, signal: Dict[str, Any]) -> bool:
        """Log signal to backtest_signals table"""
        try:
            # Extract signal data with defaults
            epic = signal.get('epic', 'UNKNOWN')
            timeframe = signal.get('timeframe', '15m')
            signal_timestamp = signal.get('signal_timestamp') or signal.get('timestamp') or datetime.now(timezone.utc)
            signal_type = self._normalize_signal_type(signal.get('signal_type', ''))
            strategy_name = signal.get('strategy', 'unknown')

            # Market data
            open_price = self._to_decimal(signal.get('open_price') or signal.get('current_price', 0))
            high_price = self._to_decimal(signal.get('high_price', 0))
            low_price = self._to_decimal(signal.get('low_price', 0))
            close_price = self._to_decimal(signal.get('close_price') or signal.get('current_price', 0))
            volume = signal.get('volume', 0)

            # Signal characteristics
            confidence_score = self._to_decimal(signal.get('confidence_score', 0))
            signal_strength = self._to_decimal(signal.get('signal_strength'))

            # Technical indicators
            indicator_values = self._extract_indicator_values(signal)

            # Trade parameters
            entry_price = self._to_decimal(signal.get('entry_price') or signal.get('current_price', 0))
            stop_loss_price = self._to_decimal(signal.get('stop_loss_price'))
            take_profit_price = self._to_decimal(signal.get('take_profit_price'))
            risk_reward_ratio = self._to_decimal(signal.get('risk_reward_ratio'))

            # Trade outcome (usually null for initial signals)
            exit_price = self._to_decimal(signal.get('exit_price'))
            exit_timestamp = signal.get('exit_timestamp')
            exit_reason = signal.get('exit_reason')
            pips_gained = self._to_decimal(signal.get('pips_gained'))
            trade_result = signal.get('trade_result')

            # Performance metrics
            holding_time_minutes = signal.get('holding_time_minutes')
            max_favorable_excursion_pips = self._to_decimal(signal.get('max_favorable_excursion_pips'))
            max_adverse_excursion_pips = self._to_decimal(signal.get('max_adverse_excursion_pips'))

            # Data quality
            data_completeness = self._to_decimal(signal.get('data_completeness', 1.0))
            validation_flags = signal.get('validation_flags', [])

            # Validation results
            validation_passed = signal.get('validation_passed', False)
            validation_reasons = signal.get('validation_reasons', [])
            trade_validator_version = signal.get('trade_validator_version', 'unknown')

            # Market intelligence
            market_intelligence = signal.get('market_intelligence', {})
            smart_money_score = self._to_decimal(signal.get('smart_money_score'))
            smart_money_validated = signal.get('smart_money_validated', False)

            # Convert timestamp to ensure timezone awareness
            if isinstance(signal_timestamp, str):
                signal_timestamp = datetime.fromisoformat(signal_timestamp.replace('Z', '+00:00'))
            elif signal_timestamp.tzinfo is None:
                signal_timestamp = signal_timestamp.replace(tzinfo=timezone.utc)

            # Insert into database
            query = """
            INSERT INTO backtest_signals (
                execution_id, epic, timeframe, signal_timestamp, signal_type, strategy_name,
                open_price, high_price, low_price, close_price, volume,
                confidence_score, signal_strength, indicator_values,
                entry_price, stop_loss_price, take_profit_price, risk_reward_ratio,
                exit_price, exit_timestamp, exit_reason, pips_gained, trade_result,
                holding_time_minutes, max_favorable_excursion_pips, max_adverse_excursion_pips,
                data_completeness, validation_flags,
                validation_passed, validation_reasons, trade_validator_version,
                market_intelligence, smart_money_score, smart_money_validated
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            """

            params = (
                self.execution_id, epic, timeframe, signal_timestamp, signal_type, strategy_name,
                open_price, high_price, low_price, close_price, volume,
                confidence_score, signal_strength, json.dumps(indicator_values),
                entry_price, stop_loss_price, take_profit_price, risk_reward_ratio,
                exit_price, exit_timestamp, exit_reason, pips_gained, trade_result,
                holding_time_minutes, max_favorable_excursion_pips, max_adverse_excursion_pips,
                data_completeness, validation_flags,
                validation_passed, validation_reasons, trade_validator_version,
                json.dumps(market_intelligence), smart_money_score, smart_money_validated
            )

            self.db_manager.execute_query(query, params)
            return True

        except Exception as e:
            self.logger.error(f"Error inserting backtest signal: {e}")
            self.logger.error(f"Signal data: {signal}")
            return False

    def _normalize_signal_type(self, signal_type: str) -> str:
        """Normalize signal type to BULL/BEAR"""
        if not signal_type:
            return 'BULL'  # Default

        signal_type = str(signal_type).upper()

        # Map various signal types to BULL/BEAR
        if signal_type in ['BUY', 'LONG', 'BULL', 'UP']:
            return 'BULL'
        elif signal_type in ['SELL', 'SHORT', 'BEAR', 'DOWN']:
            return 'BEAR'
        else:
            return signal_type  # Keep as-is if already BULL/BEAR

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        """Convert value to Decimal or None"""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            return None

    def _extract_indicator_values(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical indicator values from signal"""
        indicator_values = {}

        # Common indicator fields to extract
        indicator_fields = [
            'ema_5', 'ema_13', 'ema_21', 'ema_50', 'ema_200',
            'macd_line', 'macd_signal', 'macd_histogram',
            'rsi', 'atr', 'bollinger_upper', 'bollinger_lower',
            'support_level', 'resistance_level',
            'momentum_score', 'trend_strength'
        ]

        for field in indicator_fields:
            if field in signal:
                value = signal[field]
                if value is not None:
                    try:
                        indicator_values[field] = float(value)
                    except (ValueError, TypeError):
                        indicator_values[field] = str(value)

        # Extract nested indicator data
        if 'indicators' in signal and isinstance(signal['indicators'], dict):
            indicator_values.update(signal['indicators'])

        return indicator_values

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """
        Mock order cancellation for compatibility
        In backtest mode, this doesn't do anything meaningful
        """
        self.logger.debug(f"Mock cancel order: {order_id}")
        return True, "Order cancelled (backtest mode)"

    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Mock order modification for compatibility
        In backtest mode, this could potentially update the logged signal
        """
        self.logger.debug(f"Mock modify order: {order_id}, modifications: {modifications}")
        return True, "Order modified (backtest mode)"

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Mock order status for compatibility
        """
        return {
            'order_id': order_id,
            'status': 'logged',
            'epic': 'UNKNOWN',
            'direction': 'UNKNOWN',
            'logged_at': datetime.now(timezone.utc).isoformat()
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Mock positions for compatibility
        In backtest mode, return empty list or logged signals
        """
        return []

    def close_position(self, epic: str) -> Tuple[bool, str]:
        """
        Mock position closing for compatibility
        """
        self.logger.debug(f"Mock close position: {epic}")
        return True, "Position closed (backtest mode)"

    def update_execution_stats(self, completed_combinations: int = None,
                             total_candles_processed: int = None,
                             data_gaps_detected: int = None,
                             memory_usage_mb: int = None) -> bool:
        """Update execution statistics in database"""
        try:
            updates = []
            params = {}

            if completed_combinations is not None:
                updates.append("completed_combinations = :completed_combinations")
                params['completed_combinations'] = completed_combinations

            if total_candles_processed is not None:
                updates.append("total_candles_processed = :total_candles_processed")
                params['total_candles_processed'] = total_candles_processed

            if data_gaps_detected is not None:
                updates.append("data_gaps_detected = :data_gaps_detected")
                params['data_gaps_detected'] = data_gaps_detected

            if memory_usage_mb is not None:
                updates.append("memory_usage_mb = :memory_usage_mb")
                params['memory_usage_mb'] = memory_usage_mb

            if updates:
                updates.append("updated_at = NOW()")
                params['execution_id'] = int(self.execution_id)

                query = f"""
                UPDATE backtest_executions
                SET {', '.join(updates)}
                WHERE id = :execution_id
                """

                # Handle UPDATE query exception
                try:
                    self.db_manager.execute_query(query, params)
                except Exception as update_error:
                    if "This result object does not return rows" in str(update_error):
                        # UPDATE query succeeded but DatabaseManager can't create DataFrame - this is expected
                        pass
                    else:
                        raise update_error
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error updating execution stats: {e}")
            return False

    def finalize_execution(self, status: str = 'completed', error_message: str = None) -> bool:
        """Mark execution as completed and calculate final statistics"""
        if self._finalized:
            self.logger.debug(f"Execution {self.execution_id} already finalized, skipping")
            return True

        try:
            self._finalized = True
            # Calculate execution duration
            duration_query = """
            UPDATE backtest_executions
            SET end_time = NOW(),
                execution_duration_seconds = EXTRACT(EPOCH FROM (NOW() - start_time))::INTEGER,
                status = :status,
                error_message = :error_message,
                updated_at = NOW()
            WHERE id = :execution_id
            """

            params = {
                'status': status,
                'error_message': error_message,
                'execution_id': int(self.execution_id)  # Convert numpy.int64 to Python int
            }

            # Execute UPDATE query - handle DataFrame return issue
            try:
                self.db_manager.execute_query(duration_query, params)
            except Exception as update_error:
                if "This result object does not return rows" in str(update_error):
                    # UPDATE query succeeded but DatabaseManager can't create DataFrame - this is expected
                    pass
                else:
                    raise update_error

            # Calculate performance metrics
            perf_params = {'execution_id': int(self.execution_id)}
            try:
                self.db_manager.execute_query("SELECT calculate_backtest_performance(:execution_id)", perf_params)
            except Exception as perf_error:
                if "This result object does not return rows" in str(perf_error):
                    # Performance function called successfully
                    pass
                else:
                    raise perf_error

            self.logger.info(f"✅ Backtest execution {self.execution_id} finalized: {status}")
            self.logger.info(f"📊 Signals logged: {self.signals_logged}, "
                           f"Validated: {self.validation_passed}, "
                           f"Failed validation: {self.validation_failed}")

            return True

        except Exception as e:
            self.logger.error(f"Error finalizing execution: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return {
            'execution_id': self.execution_id,
            'signals_logged': self.signals_logged,
            'validation_passed': self.validation_passed,
            'validation_failed': self.validation_failed,
            'validation_rate': self.validation_passed / max(1, self.signals_logged)
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finalize execution"""
        if exc_type is not None:
            self.finalize_execution(status='failed', error_message=str(exc_val))
        else:
            self.finalize_execution(status='completed')


# Factory function for compatibility
def create_backtest_order_logger(db_manager: DatabaseManager,
                                execution_id: int,
                                **kwargs) -> BacktestOrderLogger:
    """Create BacktestOrderLogger instance"""
    return BacktestOrderLogger(db_manager, execution_id, **kwargs)