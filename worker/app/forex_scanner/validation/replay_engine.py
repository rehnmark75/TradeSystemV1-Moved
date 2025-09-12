# validation/replay_engine.py
"""
Replay Engine for Signal Validation

This module provides the core orchestration logic for signal validation
and replay operations. It coordinates between data retrieval, state recreation,
signal processing, and validation reporting.
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from forex_scanner.core.database import DatabaseManager
from forex_scanner.core.scanner import IntelligentForexScanner
from forex_scanner.core.signal_detector import SignalDetector
from forex_scanner.core.processing.signal_processor import SignalProcessor
from forex_scanner.alerts.alert_history import AlertHistoryManager
from forex_scanner.utils.scanner_utils import make_json_serializable

from .historical_data_manager import HistoricalDataManager
from .scanner_state_recreator import ScannerStateRecreator
from .validation_reporter import ValidationReporter, ValidationResult
from .replay_config import ReplayConfig, PERFORMANCE_CONFIG
from .historical_data_fetcher import HistoricalDataFetcher


class ReplayEngine:
    """
    Core engine for signal validation and replay operations
    
    This class orchestrates the complete validation process:
    - Historical data retrieval and preparation
    - Scanner state recreation
    - Signal detection replay
    - Result comparison and validation
    - Report generation
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        user_timezone: str = 'Europe/Stockholm',
        enable_parallel_processing: bool = None
    ):
        """
        Initialize the replay engine
        
        Args:
            db_manager: Database manager for data access
            user_timezone: User's timezone for timestamp handling
            enable_parallel_processing: Whether to enable parallel processing
        """
        self.db_manager = db_manager
        self.user_timezone = user_timezone
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.historical_data_manager = HistoricalDataManager(db_manager, user_timezone)
        self.state_recreator = ScannerStateRecreator(db_manager, user_timezone) 
        self.reporter = ValidationReporter()
        
        # Initialize alert history manager for stored alert comparison
        try:
            self.alert_history = AlertHistoryManager(db_manager)
        except Exception as e:
            self.logger.warning(f"⚠️ AlertHistoryManager not available: {e}")
            self.alert_history = None
        
        # Performance configuration
        self.enable_parallel = enable_parallel_processing if enable_parallel_processing is not None else PERFORMANCE_CONFIG['enable_parallel_processing']
        self.max_concurrent = PERFORMANCE_CONFIG['max_concurrent_epics']
        
        # Statistics tracking
        self.stats = {
            'validations_performed': 0,
            'signals_detected': 0,
            'validation_failures': 0,
            'total_processing_time_ms': 0.0,
            'cache_hits': 0,
            'db_queries': 0,
            'parallel_executions': 0
        }
        
        self.logger.info(f"🚀 ReplayEngine initialized")
        self.logger.info(f"   Timezone: {user_timezone}")
        self.logger.info(f"   Parallel processing: {'✅' if self.enable_parallel else '❌'}")
        self.logger.info(f"   Max concurrent: {self.max_concurrent}")
    
    def validate_signal_at_timestamp(
        self,
        epic: str,
        timestamp: datetime,
        timeframe: str = '15m',
        strategy_filter: str = None,
        compare_with_stored: bool = True,
        debug_mode: bool = False
    ) -> ValidationResult:
        """
        Validate a signal at a specific timestamp for a single epic
        
        Args:
            epic: Epic code to validate
            timestamp: Target timestamp for validation
            timeframe: Timeframe for analysis
            strategy_filter: Specific strategy to focus on
            compare_with_stored: Whether to compare with stored alerts
            debug_mode: Enable debug mode for detailed logging
            
        Returns:
            ValidationResult object with complete validation results
        """
        start_time = time.time()
        self.stats['validations_performed'] += 1
        
        try:
            # Ensure timestamp has timezone
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            self.logger.info(f"🔍 Validating signal: {epic} @ {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            if debug_mode:
                self.logger.setLevel(logging.DEBUG)
            
            # Step 1: Recreate scanner state
            scanner_state = self.state_recreator.recreate_scanner_state(
                timestamp=timestamp,
                epic_list=[epic],
                strategy_filter=strategy_filter
            )
            
            if not scanner_state:
                return ValidationResult(
                    success=False,
                    epic=epic,
                    timestamp=timestamp,
                    signal_detected=False,
                    error_message="Failed to recreate scanner state"
                )
            
            # Step 2: Get historical market data
            market_state = self.historical_data_manager.get_market_state_at_timestamp(
                epic=epic,
                timestamp=timestamp,
                timeframe=timeframe
            )
            
            if not market_state:
                return ValidationResult(
                    success=False,
                    epic=epic,
                    timestamp=timestamp,
                    signal_detected=False,
                    error_message="Failed to retrieve historical market data"
                )
            
            # Step 3: Get enhanced historical data for signal detection
            enhanced_data = self.historical_data_manager.get_enhanced_historical_data(
                epic=epic,
                target_timestamp=timestamp,
                timeframe=timeframe,
                strategy=strategy_filter
            )
            
            if enhanced_data is None:
                return ValidationResult(
                    success=False,
                    epic=epic,
                    timestamp=timestamp,
                    signal_detected=False,
                    error_message="Failed to get enhanced historical data"
                )
            
            # Step 4: Recreate signal detection components
            signal_detector = self.state_recreator.create_signal_detector(scanner_state)
            signal_processor = self.state_recreator.create_signal_processor(scanner_state)
            
            if not signal_detector:
                return ValidationResult(
                    success=False,
                    epic=epic,
                    timestamp=timestamp,
                    signal_detected=False,
                    error_message="Failed to create signal detector"
                )
            
            # ARCHITECTURAL FIX: Replace SignalDetector's DataFetcher with HistoricalDataFetcher
            # This ensures the live strategy code uses our prepared historical data
            self.logger.info("🔧 Injecting HistoricalDataFetcher into SignalDetector")
            historical_data_fetcher = HistoricalDataFetcher(enhanced_data, self.user_timezone)
            
            # Replace the DataFetcher in SignalDetector and ALL its strategies
            signal_detector.data_fetcher = historical_data_fetcher
            
            # Update all strategies that use data_fetcher
            strategies_updated = []
            
            if hasattr(signal_detector, 'ema_strategy') and signal_detector.ema_strategy:
                signal_detector.ema_strategy.data_fetcher = historical_data_fetcher
                strategies_updated.append('EMA')
                
            if hasattr(signal_detector, 'macd_strategy') and signal_detector.macd_strategy:
                signal_detector.macd_strategy.data_fetcher = historical_data_fetcher
                strategies_updated.append('MACD')
                
            if hasattr(signal_detector, 'zero_lag_strategy') and signal_detector.zero_lag_strategy:
                signal_detector.zero_lag_strategy.data_fetcher = historical_data_fetcher
                strategies_updated.append('Zero-Lag')
                
            if hasattr(signal_detector, 'combined_strategy') and signal_detector.combined_strategy:
                signal_detector.combined_strategy.data_fetcher = historical_data_fetcher
                strategies_updated.append('Combined')
                
            if hasattr(signal_detector, 'kama_strategy') and signal_detector.kama_strategy:
                signal_detector.kama_strategy.data_fetcher = historical_data_fetcher
                strategies_updated.append('KAMA')
                
            if hasattr(signal_detector, 'bb_supertrend_strategy') and signal_detector.bb_supertrend_strategy:
                signal_detector.bb_supertrend_strategy.data_fetcher = historical_data_fetcher
                strategies_updated.append('BB-Supertrend')
                
            # Note: scalping_strategy and momentum_bias_strategy don't use data_fetcher
            # They work directly with the DataFrame passed to them
                
            self.logger.info(f"✅ Live strategies now using historical data: {', '.join(strategies_updated)}")
            
            # Step 5: Perform signal detection replay
            detection_result = self._replay_signal_detection(
                signal_detector=signal_detector,
                enhanced_data=enhanced_data,
                epic=epic,
                timestamp=timestamp,
                timeframe=timeframe,
                scanner_state=scanner_state,
                debug_mode=debug_mode
            )
            
            # Step 6: Process through SignalProcessor if available
            if detection_result['signal_detected'] and signal_processor:
                processed_signal = self._replay_signal_processing(
                    signal_processor=signal_processor,
                    signal_data=detection_result['signal_data'],
                    debug_mode=debug_mode
                )
                detection_result['signal_data'] = processed_signal
                detection_result['processing_applied'] = True
            
            # Step 7: Compare with stored alerts if requested
            comparison_result = None
            if compare_with_stored:
                comparison_result = self._compare_with_stored_alerts(
                    epic=epic,
                    timestamp=timestamp,
                    detected_signal=detection_result.get('signal_data'),
                    timeframe=timeframe
                )
            
            # Step 8: Build validation result
            processing_time_ms = (time.time() - start_time) * 1000
            self.stats['total_processing_time_ms'] += processing_time_ms
            
            if detection_result['signal_detected']:
                self.stats['signals_detected'] += 1
            
            result = ValidationResult(
                success=True,
                epic=epic,
                timestamp=timestamp,
                signal_detected=detection_result['signal_detected'],
                signal_data=detection_result.get('signal_data'),
                market_state=market_state,
                decision_path=detection_result.get('decision_path'),
                comparison_result=comparison_result,
                processing_time_ms=processing_time_ms
            )
            
            self.logger.info(f"✅ Validation completed in {processing_time_ms:.1f}ms")
            return result
            
        except Exception as e:
            self.stats['validation_failures'] += 1
            processing_time_ms = (time.time() - start_time) * 1000
            
            self.logger.error(f"❌ Validation failed for {epic}: {e}")
            if debug_mode:
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ValidationResult(
                success=False,
                epic=epic,
                timestamp=timestamp,
                signal_detected=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms
            )
        
        finally:
            # Cleanup
            try:
                self.state_recreator.restore_original_configuration()
            except Exception as e:
                self.logger.warning(f"⚠️ Error restoring configuration: {e}")
    
    def validate_batch_signals(
        self,
        epic_list: List[str],
        timestamp: datetime,
        timeframe: str = '15m',
        strategy_filter: str = None,
        compare_with_stored: bool = True,
        show_progress: bool = True
    ) -> List[ValidationResult]:
        """
        Validate signals for multiple epics at the same timestamp
        
        Args:
            epic_list: List of epics to validate
            timestamp: Target timestamp for validation
            timeframe: Timeframe for analysis
            strategy_filter: Specific strategy to focus on
            compare_with_stored: Whether to compare with stored alerts
            show_progress: Whether to show progress indicators
            
        Returns:
            List of ValidationResult objects
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Batch validation: {len(epic_list)} epics @ {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            results = []
            
            if self.enable_parallel and len(epic_list) > 1:
                # Parallel execution
                results = self._execute_batch_parallel(
                    epic_list=epic_list,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    strategy_filter=strategy_filter,
                    compare_with_stored=compare_with_stored,
                    show_progress=show_progress
                )
                self.stats['parallel_executions'] += 1
            else:
                # Sequential execution
                results = self._execute_batch_sequential(
                    epic_list=epic_list,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    strategy_filter=strategy_filter,
                    compare_with_stored=compare_with_stored,
                    show_progress=show_progress
                )
            
            batch_time = time.time() - start_time
            successful = sum(1 for r in results if r.success)
            signals_found = sum(1 for r in results if r.signal_detected)
            
            self.logger.info(f"✅ Batch validation completed in {batch_time:.2f}s")
            self.logger.info(f"   Results: {successful}/{len(epic_list)} successful, {signals_found} signals detected")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Batch validation failed: {e}")
            return []
    
    def validate_time_series(
        self,
        epic: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        interval_minutes: int = 60,
        timeframe: str = '15m',
        strategy_filter: str = None
    ) -> List[ValidationResult]:
        """
        Validate signals across a time series for pattern analysis
        
        Args:
            epic: Epic code to validate
            start_timestamp: Start of time series
            end_timestamp: End of time series
            interval_minutes: Interval between validation points
            timeframe: Timeframe for analysis
            strategy_filter: Specific strategy to focus on
            
        Returns:
            List of ValidationResult objects across time series
        """
        try:
            self.logger.info(f"📈 Time series validation: {epic}")
            self.logger.info(f"   Range: {start_timestamp.strftime('%Y-%m-%d %H:%M')} → {end_timestamp.strftime('%Y-%m-%d %H:%M')}")
            self.logger.info(f"   Interval: {interval_minutes} minutes")
            
            # Generate timestamp list
            timestamps = []
            current = start_timestamp
            while current <= end_timestamp:
                timestamps.append(current)
                current += timedelta(minutes=interval_minutes)
            
            self.logger.info(f"   Validation points: {len(timestamps)}")
            
            # Validate each timestamp
            results = []
            for i, ts in enumerate(timestamps, 1):
                self.logger.info(f"📊 Validating point {i}/{len(timestamps)}: {ts.strftime('%H:%M')}")
                
                result = self.validate_signal_at_timestamp(
                    epic=epic,
                    timestamp=ts,
                    timeframe=timeframe,
                    strategy_filter=strategy_filter,
                    compare_with_stored=False,  # Skip comparison for time series
                    debug_mode=False
                )
                
                results.append(result)
            
            # Summary statistics
            successful = sum(1 for r in results if r.success)
            signals_found = sum(1 for r in results if r.signal_detected)
            
            self.logger.info(f"✅ Time series validation completed")
            self.logger.info(f"   {successful}/{len(timestamps)} successful, {signals_found} signals detected")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Time series validation failed: {e}")
            return []
    
    def _replay_signal_detection(
        self,
        signal_detector: SignalDetector,
        enhanced_data: Any,
        epic: str,
        timestamp: datetime,
        timeframe: str,
        scanner_state: Dict[str, Any],
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """Replay signal detection process"""
        try:
            self.logger.debug(f"🔬 Replaying signal detection for {epic}")
            
            # Get pair information
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            # Get configuration from scanner state
            config_data = scanner_state.get('configuration', {})
            spread_pips = config_data.get('spread_pips', 1.5)
            use_bid_adjustment = config_data.get('use_bid_adjustment', False)
            
            # Decision path tracking
            decision_path = []
            
            # Determine which detection method to use
            if use_bid_adjustment:
                self.logger.debug("🔧 Using BID-adjusted signal detection")
                signal = signal_detector.detect_signals_bid_adjusted(epic, pair, spread_pips, timeframe)
                decision_path.append({
                    'step': 'detection_method',
                    'status': 'SELECTED',
                    'description': f'BID-adjusted detection (spread: {spread_pips} pips)'
                })
            else:
                self.logger.debug("🔧 Using MID-price signal detection")
                signal = signal_detector.detect_signals_mid_prices(epic, pair, timeframe)
                decision_path.append({
                    'step': 'detection_method',
                    'status': 'SELECTED',
                    'description': 'MID-price detection'
                })
            
            if signal:
                self.logger.debug(f"✅ Signal detected: {signal.get('signal_type')} (confidence: {signal.get('confidence_score', 0):.3f})")
                
                decision_path.append({
                    'step': 'signal_detection',
                    'status': 'PASS',
                    'description': f"Signal detected: {signal.get('signal_type')} @ {signal.get('confidence_score', 0):.1%}"
                })
                
                # Clean and prepare signal data
                signal = make_json_serializable(signal)
                
                return {
                    'signal_detected': True,
                    'signal_data': signal,
                    'decision_path': decision_path
                }
            else:
                self.logger.debug("ℹ️ No signal detected")
                
                decision_path.append({
                    'step': 'signal_detection',
                    'status': 'FAIL',
                    'description': 'No signal met detection criteria'
                })
                
                return {
                    'signal_detected': False,
                    'signal_data': None,
                    'decision_path': decision_path
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error in signal detection replay: {e}")
            return {
                'signal_detected': False,
                'signal_data': None,
                'error': str(e),
                'decision_path': [{'step': 'error', 'status': 'FAIL', 'description': f'Detection error: {str(e)}'}]
            }
    
    def _replay_signal_processing(
        self,
        signal_processor: SignalProcessor,
        signal_data: Dict[str, Any],
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """Replay signal processing through SignalProcessor"""
        try:
            if not ReplayConfig.is_feature_enabled('enable_smart_money_validation'):
                return signal_data
            
            self.logger.debug("🧠 Replaying signal processing (Smart Money, filters, etc.)")
            
            # Process signal through full pipeline
            processed_signal = signal_processor.process_signal(signal_data)
            
            if processed_signal:
                # Check processing results
                processing_result = processed_signal.get('processing_result', {})
                
                if processing_result.get('smart_money_analyzed'):
                    self.logger.debug(f"✅ Smart Money applied: score {processed_signal.get('smart_money_score', 0):.3f}")
                
                if processing_result.get('claude_analyzed'):
                    self.logger.debug("✅ Claude analysis applied")
                
                return processed_signal
            else:
                self.logger.debug("⚠️ Signal filtered out during processing")
                return signal_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in signal processing replay: {e}")
            return signal_data
    
    def _compare_with_stored_alerts(
        self,
        epic: str,
        timestamp: datetime,
        detected_signal: Dict[str, Any],
        timeframe: str,
        time_window_minutes: int = 30
    ) -> Dict[str, Any]:
        """Compare detected signal with stored historical alerts"""
        try:
            self.logger.debug(f"🔍 Comparing with stored alerts for {epic}")
            
            # Define search window
            window_start = timestamp - timedelta(minutes=time_window_minutes)
            window_end = timestamp + timedelta(minutes=time_window_minutes)
            
            # Search for matching alerts - REAL IMPLEMENTATION
            self.logger.info(f"🔍 Searching for stored alerts in window: {window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}")
            comparison_result = self._search_stored_alerts(epic, timestamp, window_start, window_end, detected_signal)
            
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"❌ Error comparing with stored alerts: {e}")
            return {'error': str(e)}
    
    def _execute_batch_parallel(
        self,
        epic_list: List[str],
        timestamp: datetime,
        timeframe: str,
        strategy_filter: str,
        compare_with_stored: bool,
        show_progress: bool
    ) -> List[ValidationResult]:
        """Execute batch validation in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent, len(epic_list))) as executor:
            # Submit all tasks
            future_to_epic = {
                executor.submit(
                    self.validate_signal_at_timestamp,
                    epic, timestamp, timeframe, strategy_filter, compare_with_stored, False
                ): epic for epic in epic_list
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_epic):
                epic = future_to_epic[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Parallel execution failed for {epic}: {e}")
                    results.append(ValidationResult(
                        success=False,
                        epic=epic,
                        timestamp=timestamp,
                        signal_detected=False,
                        error_message=str(e)
                    ))
                
                completed += 1
                if show_progress:
                    self.logger.info(f"📊 Progress: {completed}/{len(epic_list)} completed")
        
        return results
    
    def _execute_batch_sequential(
        self,
        epic_list: List[str],
        timestamp: datetime,
        timeframe: str,
        strategy_filter: str,
        compare_with_stored: bool,
        show_progress: bool
    ) -> List[ValidationResult]:
        """Execute batch validation sequentially"""
        results = []
        
        for i, epic in enumerate(epic_list, 1):
            if show_progress:
                self.logger.info(f"📊 Processing {i}/{len(epic_list)}: {epic}")
            
            result = self.validate_signal_at_timestamp(
                epic=epic,
                timestamp=timestamp,
                timeframe=timeframe,
                strategy_filter=strategy_filter,
                compare_with_stored=compare_with_stored,
                debug_mode=False
            )
            
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_time_seconds = self.stats['total_processing_time_ms'] / 1000
        
        return {
            'validations_performed': self.stats['validations_performed'],
            'signals_detected': self.stats['signals_detected'],
            'validation_failures': self.stats['validation_failures'],
            'success_rate': (self.stats['validations_performed'] - self.stats['validation_failures']) / max(self.stats['validations_performed'], 1) * 100,
            'signal_detection_rate': self.stats['signals_detected'] / max(self.stats['validations_performed'], 1) * 100,
            'total_processing_time_seconds': total_time_seconds,
            'average_processing_time_ms': self.stats['total_processing_time_ms'] / max(self.stats['validations_performed'], 1),
            'cache_hits': self.stats['cache_hits'],
            'db_queries': self.stats['db_queries'],
            'parallel_executions': self.stats['parallel_executions']
        }
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.stats.items()}
        self.logger.info("📊 Performance statistics reset")
    
    def _search_stored_alerts(
        self, 
        epic: str, 
        target_timestamp: datetime, 
        window_start: datetime, 
        window_end: datetime, 
        signal_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Search for stored alerts in the database within the time window
        
        Args:
            epic: Epic to search for
            target_timestamp: Target timestamp
            window_start: Start of search window
            window_end: End of search window
            signal_data: Current signal data to compare with
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Query the alert_history table
            query = """
            SELECT alert_timestamp, epic, signal_type, confidence_score, strategy, price,
                   ema_short, ema_long, ema_trend, crossover_type
            FROM alert_history 
            WHERE epic = %s 
            AND alert_timestamp BETWEEN %s AND %s
            ORDER BY alert_timestamp ASC
            """
            
            import pandas as pd
            
            # Execute query using database manager
            df = pd.read_sql_query(
                query,
                self.db_manager.get_connection(),
                params=(epic, window_start, window_end)
            )
            
            alerts_found = len(df)
            self.logger.info(f"📊 Found {alerts_found} stored alerts in search window")
            
            if alerts_found == 0:
                return {
                    'stored_alert_found': False,
                    'search_details': {
                        'epic': epic,
                        'time_window': f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}",
                        'alerts_checked': 0
                    },
                    'matches': {}
                }
            
            # Find the closest match by timestamp
            best_match = None
            closest_time_diff = None
            
            for _, alert in df.iterrows():
                alert_time = pd.to_datetime(alert['alert_timestamp'])
                time_diff = abs((target_timestamp - alert_time).total_seconds())
                
                if closest_time_diff is None or time_diff < closest_time_diff:
                    closest_time_diff = time_diff
                    best_match = alert
            
            if best_match is not None:
                self.logger.info(f"✅ Found matching stored alert at {best_match['alert_timestamp']}")
                
                # Compare signals
                matches = {}
                if signal_data:
                    stored_signal = best_match['signal_type']
                    current_signal = signal_data.get('signal_type', 'UNKNOWN')
                    matches['signal_type'] = stored_signal == current_signal
                    
                    stored_confidence = float(best_match['confidence_score'])
                    current_confidence = signal_data.get('confidence_score', 0.0)
                    confidence_diff = abs(stored_confidence - current_confidence)
                    matches['confidence'] = confidence_diff < 0.05  # 5% tolerance
                    
                    matches['strategy'] = best_match['strategy'] == signal_data.get('strategy', '')
                
                return {
                    'stored_alert_found': True,
                    'stored_alert': {
                        'id': None,  # Would need to add ID column
                        'signal_type': best_match['signal_type'],
                        'confidence_score': float(best_match['confidence_score']),
                        'strategy': best_match['strategy'],
                        'timestamp': str(best_match['alert_timestamp']),
                        'price': float(best_match['price']) if best_match['price'] else None,
                        'ema_values': {
                            'short': float(best_match['ema_short']) if best_match['ema_short'] else None,
                            'long': float(best_match['ema_long']) if best_match['ema_long'] else None,
                            'trend': float(best_match['ema_trend']) if best_match['ema_trend'] else None
                        }
                    },
                    'search_details': {
                        'epic': epic,
                        'time_window': f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}",
                        'alerts_checked': alerts_found,
                        'time_difference_seconds': int(closest_time_diff)
                    },
                    'matches': matches,
                    'match_quality': self._assess_match_quality(matches) if matches else 'N/A'
                }
            else:
                return {
                    'stored_alert_found': False,
                    'search_details': {
                        'epic': epic,
                        'time_window': f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}",
                        'alerts_checked': alerts_found
                    },
                    'matches': {}
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error searching stored alerts: {e}")
            return {
                'stored_alert_found': False,
                'search_details': {
                    'epic': epic,
                    'time_window': f"{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}",
                    'alerts_checked': 0,
                    'error': str(e)
                },
                'matches': {}
            }
    
    def _assess_match_quality(self, matches: Dict[str, bool]) -> str:
        """Assess the quality of the match based on comparison results"""
        if not matches:
            return 'N/A'
        
        match_count = sum(matches.values())
        total_checks = len(matches)
        match_ratio = match_count / total_checks
        
        if match_ratio >= 0.9:
            return 'EXCELLENT'
        elif match_ratio >= 0.7:
            return 'GOOD'
        elif match_ratio >= 0.5:
            return 'FAIR'
        else:
            return 'POOR'

    def cleanup_resources(self):
        """Cleanup resources and restore original state"""
        try:
            self.state_recreator.cleanup_recreations()
            self.historical_data_manager.clear_cache()
            self.logger.info("🧹 ReplayEngine resources cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up resources: {e}")