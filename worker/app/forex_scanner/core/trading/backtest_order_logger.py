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


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Decimal and datetime objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'isoformat'):  # Handle other date/time types
            return obj.isoformat()
        return super().default(obj)


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

        # Signal tracking for comprehensive reporting
        self.signals_by_epic = {}  # epic -> signal count
        self.signals_by_strategy = {}  # strategy -> signal count
        self.all_signals = []  # Store all signals for detailed reporting
        self.bull_signals = 0
        self.bear_signals = 0

        # Validation failure tracking
        self.validation_failures = {}  # validation_step -> count

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
        Log trading signal to console only (no database)
        Returns same format as OrderManager for compatibility
        """
        try:
            # Extract signal data
            epic = signal.get('epic', 'UNKNOWN')
            signal_type = signal.get('signal_type', 'UNKNOWN').upper()
            confidence = signal.get('confidence_score', 0)
            price = signal.get('current_price', 0)
            timestamp = signal.get('signal_timestamp', 'UNKNOWN')
            strategy = signal.get('strategy', 'UNKNOWN')

            # Update statistics
            self.signals_logged += 1
            if signal.get('validation_passed', False):
                self.validation_passed += 1
            else:
                self.validation_failed += 1

            # Track by epic
            if epic not in self.signals_by_epic:
                self.signals_by_epic[epic] = 0
            self.signals_by_epic[epic] += 1

            # Track by strategy
            if strategy not in self.signals_by_strategy:
                self.signals_by_strategy[strategy] = 0
            self.signals_by_strategy[strategy] += 1

            # Track signal types
            if signal_type in ['BULL', 'BUY', 'LONG']:
                self.bull_signals += 1
                signal_type = 'BUY'  # Normalize for display
            elif signal_type in ['BEAR', 'SELL', 'SHORT']:
                self.bear_signals += 1
                signal_type = 'SELL'  # Normalize for display

            # Store complete signal data for detailed reporting
            validation_passed = signal.get('validation_passed', False)
            validation_message = signal.get('validation_message', '')

            # Initialize variables
            failure_reason = 'Unknown'
            detailed_reason = 'N/A'

            # Track validation failures for summary
            if not validation_passed and validation_message:
                # Extract the validation step that failed (first part before ':')
                failure_reason = validation_message.split(':')[0] if ':' in validation_message else validation_message
                failure_reason = failure_reason.strip()
                if failure_reason not in self.validation_failures:
                    self.validation_failures[failure_reason] = 0
                self.validation_failures[failure_reason] += 1

                # Store detailed rejection reason (full validation message)
                detailed_reason = validation_message.strip() if validation_message else 'Unknown'
            elif not validation_passed:
                # No validation message but failed - use generic reason
                failure_reason = 'Validation Failed'
                detailed_reason = 'No specific validation message provided'

            # Prefer actual trade simulation results (profit/loss) over MFE/MAE metrics
            # The 'profit' and 'loss' fields are set by the trailing/VSL simulators
            # Only fall back to max_profit_pips/max_loss_pips if actual results not available
            profit_pips = signal.get('profit') or signal.get('max_profit_pips') or signal.get('potential_profit_pips', 0)
            loss_pips = signal.get('loss') or signal.get('max_loss_pips') or signal.get('potential_loss_pips', 0)

            # Get pips_gained from various possible sources
            pips_gained = signal.get('pips_gained', 0)
            if pips_gained == 0:
                # Fallback: calculate from trade result
                trade_result = signal.get('trade_result', '')
                if trade_result == 'win':
                    pips_gained = profit_pips if profit_pips > 0 else signal.get('limit_distance', 15)
                elif trade_result == 'loss':
                    pips_gained = -loss_pips if loss_pips > 0 else -signal.get('stop_distance', 9)

            signal_record = {
                'id': self.signals_logged,
                'timestamp': timestamp,
                'epic': epic,
                'signal_type': signal_type,
                'strategy': strategy,
                'price': price,
                'confidence': confidence,
                'profit': profit_pips,
                'loss': loss_pips,
                'pips_gained': pips_gained,  # Critical for variation testing
                'risk_reward': signal.get('profit_loss_ratio') or signal.get('risk_reward_ratio', 0),
                'validation_passed': validation_passed,
                'rejection_reason': 'N/A' if validation_passed else (failure_reason or 'Unknown'),
                'detailed_rejection_reason': detailed_reason,
                # Add configured SL/TP from strategy
                'stop_distance': signal.get('stop_distance', 0),
                'limit_distance': signal.get('limit_distance', 0),
                # Add simulation metadata if available
                'exit_reason': signal.get('exit_reason'),
                'trade_result': signal.get('trade_result'),
                'trailing_stop_used': signal.get('trailing_stop_used', False),
                'best_profit_achieved': signal.get('best_profit_achieved', 0)
            }
            self.all_signals.append(signal_record)

            # Enhanced console logging
            self.logger.info(f"🎯 BACKTEST SIGNAL #{self.signals_logged}: {epic} {signal_type}")
            self.logger.info(f"   📊 Confidence: {confidence:.1%} | Price: {price:.5f} | Strategy: {strategy}")
            self.logger.info(f"   ⏰ Timestamp: {timestamp}")

            # Show configured SL/TP if available
            stop_distance = signal.get('stop_distance', 0)
            limit_distance = signal.get('limit_distance', 0)
            if stop_distance > 0 or limit_distance > 0:
                self.logger.info(f"   🎯 Configured SL/TP: {stop_distance:.0f} / {limit_distance:.0f} pips")

            # ✅ FIX: Save signal to database for CSV export functionality
            try:
                db_saved = self._log_backtest_signal(signal)
                if db_saved:
                    self.logger.info(f"   💾 Signal saved to database (ID: {self.execution_id})")
                else:
                    self.logger.warning(f"   ⚠️ Signal DB save returned False")
            except Exception as e:
                self.logger.error(f"   ❌ DB save exception: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

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

            return True, "Signal logged to database and console", mock_order

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

            # Normalize trade_result to lowercase for database constraint (win/loss/breakeven)
            if trade_result:
                trade_result = trade_result.lower()
                # Fix common variations: "lose" -> "loss", "winner" -> "win"
                if trade_result == 'lose':
                    trade_result = 'loss'
                elif trade_result == 'winner':
                    trade_result = 'win'

            # Truncate exit_reason if too long for VARCHAR(10)
            if exit_reason and len(exit_reason) > 10:
                exit_reason = exit_reason[:10]

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
                int(self.execution_id), epic, timeframe, signal_timestamp, signal_type, strategy_name,
                open_price, high_price, low_price, close_price, volume,
                confidence_score, signal_strength, json.dumps(indicator_values, cls=DecimalEncoder),
                entry_price, stop_loss_price, take_profit_price, risk_reward_ratio,
                exit_price, exit_timestamp, exit_reason, pips_gained, trade_result,
                holding_time_minutes, max_favorable_excursion_pips, max_adverse_excursion_pips,
                data_completeness, json.dumps(validation_flags if isinstance(validation_flags, list) else [], cls=DecimalEncoder),
                validation_passed, json.dumps(validation_reasons if isinstance(validation_reasons, list) else [], cls=DecimalEncoder), trade_validator_version,
                json.dumps(market_intelligence, cls=DecimalEncoder), smart_money_score, smart_money_validated
            )

            # Use raw psycopg2 connection for INSERT with positional parameters (%s placeholders)
            raw_conn = self.db_manager.engine.raw_connection()
            try:
                cursor = raw_conn.cursor()
                cursor.execute(query, params)
                raw_conn.commit()
                cursor.close()
            finally:
                raw_conn.close()
            return True

        except Exception as e:
            self.logger.error(f"Error inserting backtest signal: {e}")
            self.logger.error(f"  trade_result={trade_result!r}, exit_reason={exit_reason!r}")
            import traceback
            self.logger.error(traceback.format_exc())
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

            # Generate comprehensive summary report
            self._generate_comprehensive_summary()

            return True

        except Exception as e:
            self.logger.error(f"Error finalizing execution: {e}")
            return False

    def _generate_comprehensive_summary(self):
        """Generate comprehensive backtest summary in the old system format"""
        if self.signals_logged == 0:
            self.logger.info("📊 No signals generated during backtest")
            return

        # Get primary strategy name (most common)
        primary_strategy = max(self.signals_by_strategy.items(), key=lambda x: x[1])[0] if self.signals_by_strategy else 'unknown'
        strategy_display = primary_strategy.upper()

        self.logger.info("")
        self.logger.info("📊 RESULTS BY EPIC:")
        self.logger.info("------------------------------")

        # Sort epics by signal count (descending)
        sorted_epics = sorted(self.signals_by_epic.items(), key=lambda x: x[1], reverse=True)
        for epic, count in sorted_epics:
            # Clean up epic display name
            epic_display = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
            self.logger.info(f"   {epic_display}: {count} signals")

        self.logger.info("")
        self.logger.info(f"✅ TOTAL {strategy_display} SIGNALS: {self.signals_logged}")

        # Add filtered signal sections
        self._display_filtered_signal_sections(strategy_display)

        # Performance summary
        self.logger.info("")
        self.logger.info(f"📈 {strategy_display} STRATEGY PERFORMANCE:")
        self.logger.info("=" * 50)
        self.logger.info(f"   📊 Total Signals: {self.signals_logged}")

        # Calculate average confidence
        if self.all_signals:
            avg_confidence = sum(s['confidence'] for s in self.all_signals) / len(self.all_signals)
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence:.1%}")

        self.logger.info(f"   📈 Bull Signals: {self.bull_signals}")
        self.logger.info(f"   📉 Bear Signals: {self.bear_signals}")

        # Calculate basic performance metrics - FIXED: Calculate separately for winners/losers
        winners = [s for s in self.all_signals if s.get('profit', 0) > 0 and s.get('loss', 0) == 0]
        losers = [s for s in self.all_signals if s.get('loss', 0) > 0 and s.get('profit', 0) == 0]
        breakevens = [s for s in self.all_signals if s.get('profit', 0) == 0 and s.get('loss', 0) == 0]
        validated_signals = sum(1 for s in self.all_signals if s.get('validation_passed', False))

        if self.all_signals:
            validation_rate = validated_signals / len(self.all_signals)

            # Calculate average profit PER WINNER (not across all signals)
            if winners:
                total_profit_winners = sum(s.get('profit', 0) for s in winners)
                avg_profit_per_winner = total_profit_winners / len(winners)
                self.logger.info(f"   💰 Average Profit per Winner: {avg_profit_per_winner:.1f} pips")
                self.logger.info(f"   ✅ Winners: {len(winners)}")
            else:
                self.logger.info(f"   💰 No winning trades")

            # Calculate average loss PER LOSER (not across all signals)
            if losers:
                total_loss_losers = sum(s.get('loss', 0) for s in losers)
                avg_loss_per_loser = total_loss_losers / len(losers)
                self.logger.info(f"   📉 Average Loss per Loser: {avg_loss_per_loser:.1f} pips")
                self.logger.info(f"   ❌ Losers: {len(losers)}")
            else:
                self.logger.info(f"   📉 No losing trades")

            if breakevens:
                self.logger.info(f"   ➖ Breakeven: {len(breakevens)}")

            # Calculate win rate
            total_closed = len(winners) + len(losers) + len(breakevens)
            if total_closed > 0:
                win_rate = len(winners) / total_closed
                self.logger.info(f"   🎯 Win Rate: {win_rate:.1%}")

            # Calculate profit factor
            if losers and winners:
                total_profit = sum(s.get('profit', 0) for s in winners)
                total_loss = sum(s.get('loss', 0) for s in losers)
                if total_loss > 0:
                    profit_factor = total_profit / total_loss
                    self.logger.info(f"   📊 Profit Factor: {profit_factor:.2f}")

                # Expectancy (average profit per trade)
                expectancy = (total_profit - total_loss) / total_closed
                self.logger.info(f"   💵 Expectancy: {expectancy:.1f} pips per trade")

            self.logger.info(f"   🏆 Validation Rate: {validation_rate:.1%}")
            self.logger.info(f"   📊 Signal Breakdown:")
            self.logger.info(f"      ✅ Validated: {validated_signals} signals")
            self.logger.info(f"      ❌ Failed Validation: {self.signals_logged - validated_signals} signals")

            # Add detailed validation failure breakdown
            if self.validation_failures:
                self.logger.info(f"      🔍 Validation Failure Breakdown:")
                # Sort by count (descending) for most common failures first
                sorted_failures = sorted(self.validation_failures.items(), key=lambda x: x[1], reverse=True)
                for failure_type, count in sorted_failures:
                    self.logger.info(f"         • {failure_type}: {count} signals")
            elif self.signals_logged - validated_signals > 0:
                self.logger.info(f"      🔍 Validation Failure Breakdown: No detailed reasons available")

    def _display_filtered_signal_sections(self, strategy_display: str):
        """Display separate sections for validated and rejected signals"""
        # Filter signals by validation status
        validated_signals = [s for s in self.all_signals if s.get('validation_passed', False)]
        rejected_signals = [s for s in self.all_signals if not s.get('validation_passed', False)]

        # Sort both by timestamp (newest first) and limit to 15 each
        validated_signals = sorted(validated_signals, key=lambda x: x['timestamp'], reverse=True)[:15]
        rejected_signals = sorted(rejected_signals, key=lambda x: x['timestamp'], reverse=True)[:15]

        # Display validated signals section
        if validated_signals:
            self.logger.info("")
            self.logger.info(f"✅ VALIDATED {strategy_display} SIGNALS ({len(validated_signals)} signals):")
            self.logger.info("=" * 170)
            self.logger.info(f"{'#':3} {'TIMESTAMP':19} {'PAIR':12} {'TYPE':4} {'STRATEGY':12} {'PRICE':10} {'CONF':6} {'SL':5} {'TP':5} {'ACT_P':7} {'ACT_L':7} {'R:R':6}")
            self.logger.info("-" * 170)

            for signal in validated_signals:
                self._format_signal_row(signal, include_rejection=False)

            self.logger.info("=" * 170)
            self.logger.info(f"📝 {len(validated_signals)} validated signals shown (newest first)")
            self.logger.info(f"    SL/TP = Configured Stop Loss/Take Profit | ACT_P/ACT_L = Actual Profit/Loss from simulation")

        # Display rejected signals section
        if rejected_signals:
            self.logger.info("")
            self.logger.info(f"❌ REJECTED {strategy_display} SIGNALS ({len(rejected_signals)} signals):")
            self.logger.info("=" * 230)
            self.logger.info(f"{'#':3} {'TIMESTAMP':19} {'PAIR':12} {'TYPE':4} {'STRATEGY':12} {'PRICE':10} {'CONF':6} {'SL':5} {'TP':5} {'ACT_P':7} {'ACT_L':7} {'R:R':6} {'DETAILED REJECTION REASON':65}")
            self.logger.info("-" * 230)

            for signal in rejected_signals:
                self._format_signal_row(signal, include_rejection=True, use_detailed=True)

            self.logger.info("=" * 230)
            self.logger.info(f"📝 {len(rejected_signals)} rejected signals shown (newest first)")
            self.logger.info(f"    SL/TP = Configured Stop Loss/Take Profit | ACT_P/ACT_L = Actual Profit/Loss from simulation")

    def _format_signal_row(self, signal: Dict, include_rejection: bool = True, use_detailed: bool = False):
        """Format a single signal row for display"""
        # Format timestamp
        if hasattr(signal['timestamp'], 'strftime'):
            timestamp_str = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            timestamp_str = str(signal['timestamp'])[:19] + ' UTC'

        # Clean up epic display
        epic_display = signal['epic'].replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
        if len(epic_display) > 12:
            epic_display = epic_display[:12]

        # Format strategy display
        strategy_display = signal['strategy']
        if len(strategy_display) > 12:
            strategy_display = strategy_display[:12]

        # Get configured SL/TP (from signal creation)
        configured_sl = signal.get('stop_distance', 0)  # In pips
        configured_tp = signal.get('limit_distance', 0)  # In pips

        # Get actual simulation results
        actual_profit = signal.get('profit', 0)
        actual_loss = signal.get('loss', 0)
        risk_reward = signal.get('risk_reward', 0)

        # Calculate R:R ratio display (based on actual results)
        if actual_loss > 0:
            rr_display = f"{actual_profit/actual_loss:.2f}"
        elif actual_profit > 0:
            rr_display = "inf"
        else:
            rr_display = "0.00"

        # Base signal info with both configured and actual values
        base_info = (f"{signal['id']:3} {timestamp_str:19} {epic_display:12} {signal['signal_type']:4} "
                    f"{strategy_display:12} {signal['price']:8.5f}  {signal['confidence']:5.1%} "
                    f"{configured_sl:4.0f}  {configured_tp:4.0f}  "
                    f"{actual_profit:6.1f}  {actual_loss:6.1f}  {rr_display:6}")

        # Add rejection reason if requested
        if include_rejection:
            if use_detailed:
                # Use detailed rejection reason for comprehensive debugging
                detailed_reason = signal.get('detailed_rejection_reason', 'N/A')
                if len(detailed_reason) > 65:
                    detailed_reason = detailed_reason[:62] + '...'
                self.logger.info(f"{base_info} {detailed_reason:65}")
            else:
                # Use basic rejection reason for compact display
                rejection_reason = signal.get('rejection_reason', 'N/A')
                if len(rejection_reason) > 25:
                    rejection_reason = rejection_reason[:22] + '...'
                self.logger.info(f"{base_info} {rejection_reason:25}")
        else:
            self.logger.info(base_info)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics"""
        return {
            'execution_id': self.execution_id,
            'signals_logged': self.signals_logged,
            'validation_passed': self.validation_passed,
            'validation_failed': self.validation_failed,
            'validation_rate': self.validation_passed / max(1, self.signals_logged),
            'signals_by_epic': self.signals_by_epic,
            'signals_by_strategy': self.signals_by_strategy,
            'bull_signals': self.bull_signals,
            'bear_signals': self.bear_signals
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics for variation testing.

        Returns statistics in the format expected by ParallelVariationRunner._extract_metrics()
        This enables proper result extraction for parameter variation tests.

        Added: Jan 2026 - Critical fix for variation testing showing 0 signals
        """
        # Calculate wins and losses from all_signals
        wins = []
        losses = []

        for sig in self.all_signals:
            pips = sig.get('pips_gained', 0)
            if pips is None:
                continue
            if pips > 0:
                wins.append(pips)
            elif pips < 0:
                losses.append(pips)

        total_signals = len(self.all_signals)
        winning_trades = len(wins)
        losing_trades = len(losses)
        total_pips = sum(wins) + sum(losses)

        # Calculate win rate
        if total_signals > 0:
            win_rate = (winning_trades / total_signals) * 100
        else:
            win_rate = 0.0

        # Calculate averages
        avg_win_pips = sum(wins) / len(wins) if wins else 0.0
        avg_loss_pips = sum(losses) / len(losses) if losses else 0.0

        # Calculate profit factor
        total_wins_amount = sum(wins)
        total_losses_amount = abs(sum(losses))
        if total_losses_amount > 0:
            profit_factor = total_wins_amount / total_losses_amount
        else:
            profit_factor = float('inf') if total_wins_amount > 0 else 0.0

        # Calculate max drawdown (simplified - running minimum)
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for sig in self.all_signals:
            pips = sig.get('pips_gained', 0) or 0
            cumulative += pips
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'total_signals': total_signals,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pips': total_pips,
            'win_rate': win_rate,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.0,
            'avg_win_pips': avg_win_pips,
            'avg_loss_pips': avg_loss_pips,
            'max_drawdown_pips': max_drawdown,
            'bull_signals': self.bull_signals,
            'bear_signals': self.bear_signals
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
    def export_signals_to_csv(self, csv_path: str) -> bool:
        """
        Export all signals collected during backtest to CSV file
        This bypasses database storage and exports directly from memory
        
        Args:
            csv_path: Full path where CSV file should be created
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            import pandas as pd
            import os
            from datetime import datetime
            
            if not self.all_signals:
                self.logger.warning(f"⚠️ No signals to export - all_signals list is empty")
                return False
            
            # Ensure directory exists
            csv_dir = os.path.dirname(csv_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)
                self.logger.info(f"   📁 Created directory: {csv_dir}")
            
            # Convert signals to DataFrame
            df = pd.DataFrame(self.all_signals)
            
            # Export to CSV
            df.to_csv(csv_path, index=False)
            
            # Calculate statistics
            total_signals = len(df)
            completed_trades = df[df['trade_result'].notna()]
            winners = len(completed_trades[completed_trades['trade_result'] == 'win'])
            losers = len(completed_trades[completed_trades['trade_result'] == 'loss'])
            breakeven = len(completed_trades[completed_trades['trade_result'] == 'breakeven'])
            win_rate = (winners / max(winners + losers, 1)) * 100 if (winners + losers) > 0 else 0
            
            self.logger.info(f"\n📁 CSV EXPORT SUCCESSFUL:")
            self.logger.info("=" * 60)
            self.logger.info(f"   📂 File: {csv_path}")
            self.logger.info(f"   📊 Total Signals: {total_signals}")
            self.logger.info(f"   ✅ Winners: {winners}")
            self.logger.info(f"   ❌ Losers: {losers}")
            self.logger.info(f"   ⚖️  Breakeven: {breakeven}")
            self.logger.info(f"   🎯 Win Rate: {win_rate:.1f}%")
            self.logger.info(f"   🔍 Execution ID: {self.execution_id}")
            self.logger.info("=" * 60)
            
            # Show columns
            self.logger.info(f"\n📋 Available Columns ({len(df.columns)}):")
            for col in df.columns:
                self.logger.info(f"   • {col}")
            
            self.logger.info(f"\n💡 Analysis Tips:")
            self.logger.info(f"   import pandas as pd")
            self.logger.info(f"   df = pd.read_csv('{csv_path}')")
            self.logger.info(f"   df[df['trade_result'] == 'loss']  # Analyze losers")
            self.logger.info(f"   df[df['trade_result'] == 'win']   # Analyze winners")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export signals to CSV: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
