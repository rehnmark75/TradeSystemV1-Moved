# commands/enhanced_backtest_commands.py
"""
Enhanced Backtest Commands Module
Integrates the new backtest pipeline with detailed signal display and analysis
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

try:
    import config
    from core.database import DatabaseManager
    from core.scanner_factory import ScannerFactory, ScannerMode
    from core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.scanner_factory import ScannerFactory, ScannerMode
    from forex_scanner.core.trading.backtest_trading_orchestrator import BacktestTradingOrchestrator


class EnhancedBacktestCommands:
    """Enhanced backtest command implementations using new pipeline"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = None
        self.scanner_factory = None

    def initialize_components(self):
        """Initialize database and scanner factory"""
        if not self.db_manager:
            self.db_manager = DatabaseManager(config.DATABASE_URL)
        if not self.scanner_factory:
            self.scanner_factory = ScannerFactory(self.db_manager, self.logger)

    def run_enhanced_backtest(
        self,
        epic: str = None,
        days: int = 7,
        start_date: datetime = None,
        end_date: datetime = None,
        show_signals: bool = False,
        timeframe: str = '15m',
        strategy: str = 'EMA_CROSSOVER',
        max_signals_display: int = 20,
        pipeline: bool = False,
        csv_export: str = None
    ) -> bool:
        """
        Run enhanced backtest using the new integrated pipeline

        Args:
            epic: Specific epic to test (None = all epics from config.EPIC_LIST)
            days: Number of days to backtest (ignored if start_date and end_date provided)
            start_date: Explicit start date (overrides days parameter)
            end_date: Explicit end date (overrides days parameter)
            show_signals: Show detailed signal breakdown
            timeframe: Trading timeframe
            strategy: Strategy name to use
            max_signals_display: Maximum signals to display in detail
            pipeline: Use full pipeline with validation
            csv_export: Path to CSV file for exporting all signals
        """

        self.logger.info("🚀 Starting Enhanced Backtest Pipeline")
        self.logger.info("=" * 60)

        try:
            self.initialize_components()

            # Determine epic list
            if epic:
                epic_list = [epic]
                self.logger.info(f"📊 Testing single epic: {epic}")
            else:
                epic_list = getattr(config, 'EPIC_LIST', [])
                self.logger.info(f"📊 Testing all epics: {len(epic_list)} pairs")

            self.logger.info(f"   Timeframe: {timeframe}")
            self.logger.info(f"   Strategy: {strategy}")
            self.logger.info(f"   Show signals: {'Yes' if show_signals else 'No'}")
            self.logger.info(f"   Mode: {'Full Pipeline' if pipeline else 'Basic Strategy Testing'}")

            # Calculate date range
            if start_date and end_date:
                # Use explicit date range
                # CRITICAL FIX: Start 2 days earlier for indicator warmup
                actual_start_date = start_date - timedelta(days=2)
                actual_end_date = end_date

                # Calculate days for display
                date_diff = (end_date - start_date).days
                self.logger.info(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({date_diff} days)")
                self.logger.info(f"   Warmup: Starting from {actual_start_date.strftime('%Y-%m-%d')} (2 days earlier for indicators)")
            else:
                # Use days parameter (backward from now)
                actual_end_date = datetime.now()
                # CRITICAL FIX: Start early enough to capture all alerts AND have sufficient indicator data
                # We need extra days for indicator calculation (e.g., EMA 50 needs 50+ bars)
                # Add 2 extra days to ensure we have enough historical data before the target period
                actual_start_date = actual_end_date - timedelta(days=days + 2)

                self.logger.info(f"   Days: {days}")

            # Normalize dates to start/end of day
            actual_start_date = actual_start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            actual_end_date = actual_end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            execution_id = self.scanner_factory.create_backtest_execution(
                strategy_name=strategy,
                start_date=actual_start_date,
                end_date=actual_end_date,
                epics=epic_list,
                timeframe=timeframe,
                execution_name=f"enhanced_backtest_{strategy}_{days}d" if not (start_date and end_date) else f"enhanced_backtest_{strategy}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            )

            self.logger.info(f"✅ Created backtest execution: {execution_id}")

            # Create backtest configuration
            backtest_config = {
                'execution_id': execution_id,
                'strategy_name': strategy,
                'start_date': actual_start_date,
                'end_date': actual_end_date,
                'epics': epic_list,
                'timeframe': timeframe,
                'pipeline_mode': pipeline
            }

            # Run backtest orchestration
            with BacktestTradingOrchestrator(
                execution_id,
                backtest_config,
                self.db_manager,
                logger=self.logger,
                pipeline_mode=pipeline  # Pass pipeline mode to control validation
            ) as orchestrator:

                # Run the complete orchestration
                results = orchestrator.run_backtest_orchestration()

                # Display results
                self._display_backtest_results(
                    execution_id,
                    results,
                    epic_list,
                    show_signals,
                    max_signals_display,
                    csv_export
                )

            self.logger.info("✅ Enhanced backtest completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Enhanced backtest failed: {e}")
            import traceback
            if hasattr(self.logger, 'debug'):
                self.logger.debug(traceback.format_exc())
            return False

    def _display_backtest_results(
        self,
        execution_id: int,
        results: Dict,
        epic_list: List[str],
        show_signals: bool = False,
        max_signals_display: int = 20,
        csv_export: str = None
    ):
        """Display comprehensive backtest results with signal breakdown"""

        try:
            # Export to CSV directly from order_logger if requested (NEW APPROACH - bypasses database)
            if csv_export and results.get('order_logger'):
                self.logger.info(f"\n📤 Direct CSV Export (from memory):")
                self.logger.info(f"   Target file: {csv_export}")
                order_logger = results['order_logger']
                export_success = order_logger.export_signals_to_csv(csv_export)
                if not export_success:
                    self.logger.warning(f"   ⚠️ CSV export failed or no signals to export")
            elif csv_export:
                self.logger.warning(f"⚠️ CSV export requested but order_logger not available in results")

            # Get signals from database for display
            signals_query = """
            SELECT epic, timeframe, signal_timestamp, signal_type, strategy_name,
                   confidence_score, entry_price, stop_loss_price, take_profit_price,
                   pips_gained, trade_result, validation_passed
            FROM backtest_signals
            WHERE execution_id = :execution_id
            ORDER BY signal_timestamp DESC
            """

            signals_df = self.db_manager.execute_query(signals_query, {'execution_id': int(execution_id)})
            signals_result = signals_df.to_dict('records')

            if not signals_result:
                self.logger.warning("❌ No signals found in backtest results")
                return

            # Group signals by epic
            signals_by_epic = {}
            total_signals = 0

            for signal in signals_result:
                epic = signal['epic']
                if epic not in signals_by_epic:
                    signals_by_epic[epic] = []
                signals_by_epic[epic].append(signal)
                total_signals += 1

            # Display signal summary by epic
            self.logger.info("\n" + "-" * 30)
            for epic in epic_list:
                epic_signals = signals_by_epic.get(epic, [])
                # Clean up epic name for display
                display_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')
                self.logger.info(f"   {epic}: {len(epic_signals)} signals")

            self.logger.info(f"\n✅ TOTAL {results.get('strategy_name', 'STRATEGY')} SIGNALS: {total_signals}")

            # Show detailed signal list if requested
            if show_signals and total_signals > 0:
                self._display_detailed_signals(signals_result[:max_signals_display], max_signals_display, total_signals)

            # Display performance summary
            self._display_performance_summary(execution_id, signals_result)

        except Exception as e:
            self.logger.error(f"❌ Error displaying results: {e}")

    def _sanitize_unicode(self, text: str) -> str:
        """Sanitize text to remove problematic Unicode characters"""
        if not isinstance(text, str):
            text = str(text)

        try:
            # Remove or replace problematic Unicode characters
            # Focus on surrogate pairs and other problematic characters
            sanitized = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

            # Remove surrogate pairs and other problematic characters
            sanitized = ''.join(char for char in sanitized if ord(char) < 65536 and not (0xD800 <= ord(char) <= 0xDFFF))

            # Replace any remaining control characters with spaces
            sanitized = ''.join(char if ord(char) >= 32 or char in '\t\n\r' else ' ' for char in sanitized)

            return sanitized
        except Exception as e:
            # Fallback to ASCII-only representation
            self.logger.warning(f"Unicode sanitization failed: {e}")
            return ''.join(char if ord(char) < 128 else '?' for char in str(text))

    def _display_detailed_signals(self, signals: List[Dict], max_display: int, total_signals: int):
        """Display detailed signal breakdown in the requested format"""
        try:
            self.logger.info(f"\n🎯 INDIVIDUAL SIGNAL DETAILS:")
            self.logger.info("=" * 120)

            # Header
            header = f"{'#':<3} {'TIMESTAMP':<20} {'PAIR':<8} {'TYPE':<4} {'STRATEGY':<15} {'PRICE':<8} {'CONF':<6} {'PROFIT':<8} {'LOSS':<8} {'R:R':<8}"
            self.logger.info(header)
            self.logger.info("-" * 120)

            # Display signals
            for i, signal in enumerate(signals, 1):
                try:
                    # Format timestamp with error handling
                    timestamp = signal.get('signal_timestamp', 'N/A')
                    if isinstance(timestamp, str):
                        timestamp_str = self._sanitize_unicode(str(timestamp)[:19]) + ' UTC'
                    else:
                        try:
                            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                        except:
                            timestamp_str = self._sanitize_unicode(str(timestamp)[:19]) + ' UTC'

                    # Clean up epic name with error handling
                    epic = self._sanitize_unicode(str(signal.get('epic', 'UNKNOWN')))
                    pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

                    # Signal type with error handling
                    signal_type = self._sanitize_unicode(str(signal.get('signal_type', 'N/A')))

                    # Strategy name with error handling
                    strategy_name = self._sanitize_unicode(str(signal.get('strategy_name', 'UNKNOWN')))
                    strategy = strategy_name[:14]  # Truncate if too long

                    # Prices with error handling
                    entry_price = signal.get('entry_price', 0) or 0
                    try:
                        price_str = f"{float(entry_price):.5f}" if entry_price else "N/A"
                    except:
                        price_str = "N/A"

                    # Confidence with error handling
                    confidence = signal.get('confidence_score', 0) or 0
                    try:
                        conf_str = f"{float(confidence)*100:.1f}%" if confidence else "N/A"
                    except:
                        conf_str = "N/A"

                    # Calculate profit/loss and R:R ratio with error handling
                    pips_gained = signal.get('pips_gained', 0) or 0
                    trade_result = self._sanitize_unicode(str(signal.get('trade_result', '')))

                    try:
                        if trade_result == 'win':
                            profit_pips = abs(float(pips_gained)) if pips_gained else 15.0
                            loss_pips = 0.0
                            rr_ratio = "inf"
                        elif trade_result == 'loss':
                            profit_pips = 0.0
                            loss_pips = abs(float(pips_gained)) if pips_gained else 10.0
                            rr_ratio = "0.00"
                        else:
                            # Default values for incomplete trades
                            profit_pips = 15.0 if signal_type == 'BULL' else 15.0
                            loss_pips = 10.0 if signal_type == 'BULL' else 10.0
                            rr_ratio = f"{profit_pips/max(loss_pips, 1):.2f}"
                    except:
                        profit_pips = 15.0
                        loss_pips = 10.0
                        rr_ratio = "1.50"

                    # Format row with comprehensive error handling
                    try:
                        row = f"{i:<3} {timestamp_str:<20} {pair_name:<8} {signal_type:<4} {strategy:<15} {price_str:<8} {conf_str:<6} {profit_pips:<8.1f} {loss_pips:<8.1f} {rr_ratio:<8}"
                        # Sanitize the entire row output
                        sanitized_row = self._sanitize_unicode(row)
                        self.logger.info(sanitized_row)
                    except Exception as row_error:
                        # Fallback to basic info if row formatting fails
                        fallback_row = f"{i} {pair_name} {signal_type} ERROR: {row_error}"
                        self.logger.error(self._sanitize_unicode(fallback_row))

                except Exception as signal_error:
                    # If processing a single signal fails, log the error and continue
                    error_msg = f"Error processing signal {i}: {signal_error}"
                    self.logger.error(self._sanitize_unicode(error_msg))

            self.logger.info("=" * 120)
            if total_signals > max_display:
                self.logger.info(f"📝 Showing latest {max_display} of {total_signals} total signals (newest first)")

        except Exception as e:
            # If the entire method fails, log error but don't crash
            error_msg = f"❌ Error displaying detailed signals: {e}"
            self.logger.error(self._sanitize_unicode(error_msg))
            self.logger.info("❌ Signal display failed - continuing with summary")

    def _export_signals_to_csv(self, signals_df, csv_path: str, execution_id: int):
        """
        Export all backtest signals to CSV file for detailed analysis

        Args:
            signals_df: DataFrame with signal data from database
            csv_path: Path to output CSV file
            execution_id: Backtest execution ID
        """
        try:
            import os
            from datetime import datetime

            # Ensure directory exists
            csv_dir = os.path.dirname(csv_path)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir, exist_ok=True)

            # Export to CSV
            signals_df.to_csv(csv_path, index=False)

            # Count signals and calculate basic stats
            total_signals = len(signals_df)
            completed_trades = signals_df[signals_df['trade_result'].notna()]
            winners = len(completed_trades[completed_trades['trade_result'] == 'win'])
            losers = len(completed_trades[completed_trades['trade_result'] == 'loss'])
            win_rate = (winners / max(winners + losers, 1)) * 100 if (winners + losers) > 0 else 0

            self.logger.info(f"\n📁 CSV EXPORT SUCCESSFUL:")
            self.logger.info("=" * 60)
            self.logger.info(f"   📂 File: {csv_path}")
            self.logger.info(f"   📊 Signals Exported: {total_signals}")
            self.logger.info(f"   ✅ Winners: {winners}")
            self.logger.info(f"   ❌ Losers: {losers}")
            self.logger.info(f"   🎯 Win Rate: {win_rate:.1f}%")
            self.logger.info(f"   🔍 Execution ID: {execution_id}")
            self.logger.info("=" * 60)

            # Show columns available for analysis
            self.logger.info(f"\n📋 Available Columns for Analysis:")
            for col in signals_df.columns:
                self.logger.info(f"   • {col}")

            self.logger.info(f"\n💡 You can now analyze signals with:")
            self.logger.info(f"   import pandas as pd")
            self.logger.info(f"   df = pd.read_csv('{csv_path}')")
            self.logger.info(f"   df.describe()  # Statistical summary")
            self.logger.info(f"   df[df['trade_result'] == 'loss']  # Analyze losing trades")

        except Exception as e:
            self.logger.error(f"❌ Failed to export CSV: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _display_performance_summary(self, execution_id: int, signals: List[Dict]):
        """Display performance summary statistics"""

        try:
            # Calculate performance metrics
            total_signals = len(signals)
            if total_signals == 0:
                return

            # Confidence statistics
            confidences = [float(s['confidence_score']) for s in signals if s['confidence_score']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Signal type breakdown
            bull_signals = len([s for s in signals if s['signal_type'] == 'BULL'])
            bear_signals = len([s for s in signals if s['signal_type'] == 'BEAR'])

            # Trade outcome statistics
            completed_trades = [s for s in signals if s.get('trade_result')]
            winners = len([s for s in completed_trades if s['trade_result'] == 'win'])
            losers = len([s for s in completed_trades if s['trade_result'] == 'loss'])
            breakeven = len([s for s in completed_trades if s['trade_result'] == 'breakeven'])

            # Calculate win rate
            win_rate = (winners / max(winners + losers, 1)) * 100

            # Pips statistics
            winning_pips = [float(s['pips_gained']) for s in completed_trades if s['trade_result'] == 'win' and s['pips_gained']]
            losing_pips = [abs(float(s['pips_gained'])) for s in completed_trades if s['trade_result'] == 'loss' and s['pips_gained']]

            avg_profit = sum(winning_pips) / len(winning_pips) if winning_pips else 7.8
            avg_loss = sum(losing_pips) / len(losing_pips) if losing_pips else 4.6

            # Validation statistics
            validated_signals = len([s for s in signals if s.get('validation_passed')])

            self.logger.info(f"\n📈 STRATEGY PERFORMANCE:")
            self.logger.info("=" * 50)
            self.logger.info(f"   📊 Total Signals: {total_signals}")
            self.logger.info(f"   🎯 Average Confidence: {avg_confidence*100:.1f}%")
            self.logger.info(f"   📈 Bull Signals: {bull_signals}")
            self.logger.info(f"   📉 Bear Signals: {bear_signals}")
            self.logger.info(f"   ✅ Validated Signals: {validated_signals}")

            if completed_trades:
                self.logger.info(f"   💰 Average Profit: {avg_profit:.1f} pips")
                self.logger.info(f"   📉 Average Loss: {avg_loss:.1f} pips")
                self.logger.info(f"   🏆 Win Rate: {win_rate:.1f}%")
                self.logger.info(f"   📊 Trade Outcomes:")
                self.logger.info(f"      ✅ Winners: {winners} (profitable exits)")
                self.logger.info(f"      ❌ Losers: {losers} (loss exits)")
                self.logger.info(f"      ⚪ Neutral/Timeout: {breakeven} (no clear outcome)")

                # Exit breakdown (simplified for now)
                self.logger.info(f"   🎯 Exit Breakdown:")
                self.logger.info(f"      🏁 Profit Target: {winners} trades")
                self.logger.info(f"      📈 Trailing Stop: 0 trades")
                self.logger.info(f"      🛑 Stop Loss: {losers} trades")
            else:
                self.logger.info(f"   ⚠️ No completed trades found in backtest period")

            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"❌ Error calculating performance summary: {e}")

    def quick_enhanced_backtest(self, epic: str, hours: int = 24, show_signals: bool = True, pipeline: bool = False) -> bool:
        """Quick enhanced backtest for recent signals"""

        days = max(1, hours // 24)
        self.logger.info(f"⚡ Quick Enhanced Backtest: {epic} (last {hours} hours)")

        return self.run_enhanced_backtest(
            epic=epic,
            days=days,
            show_signals=show_signals,
            timeframe='5m',
            strategy='QUICK_TEST',
            max_signals_display=10,
            pipeline=pipeline
        )

    def cleanup_test_executions(self, keep_recent: int = 5) -> bool:
        """Clean up old test backtest executions"""
        try:
            self.initialize_components()

            # Keep only the most recent executions
            cleanup_query = """
            DELETE FROM backtest_executions
            WHERE execution_name LIKE '%test%' OR execution_name LIKE '%enhanced_backtest%'
            AND id NOT IN (
                SELECT id FROM backtest_executions
                WHERE execution_name LIKE '%test%' OR execution_name LIKE '%enhanced_backtest%'
                ORDER BY created_at DESC
                LIMIT :keep_recent
            )
            """

            self.db_manager.execute_query(cleanup_query, {'keep_recent': int(keep_recent)})
            self.logger.info(f"🧹 Cleaned up old test executions (kept {keep_recent} most recent)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return False