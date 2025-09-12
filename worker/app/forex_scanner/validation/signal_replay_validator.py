# validation/signal_replay_validator.py
"""
Signal Replay Validator - Main Entry Point

This module provides the main command-line interface for signal validation
and replay operations. It orchestrates the complete validation process
and provides comprehensive reporting.

Usage:
    python -m forex_scanner.validation.signal_replay_validator \
        --timestamp "2025-01-15 14:30:00" \
        --epic "CS.D.EURUSD.MINI.IP" \
        --show-calculations \
        --show-raw-data

    python -m forex_scanner.validation.signal_replay_validator \
        --timestamp "2025-01-15 14:30:00" \
        --all-epics \
        --compare-with-stored
"""

import argparse
import logging
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add forex_scanner to path - ensure we can import the parent modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from forex_scanner.core.database import DatabaseManager
from forex_scanner import config

try:
    from .replay_engine import ReplayEngine
    from .validation_reporter import ValidationReporter, ValidationResult
    from .replay_config import ReplayConfig
except ImportError:
    # Handle direct execution or module loading issues
    try:
        from forex_scanner.validation.replay_engine import ReplayEngine
        from forex_scanner.validation.validation_reporter import ValidationReporter, ValidationResult
        from forex_scanner.validation.replay_config import ReplayConfig
    except ImportError:
        # Fallback for direct file execution
        from replay_engine import ReplayEngine
        from validation_reporter import ValidationReporter, ValidationResult
        from replay_config import ReplayConfig


class SignalReplayValidator:
    """
    Main interface for signal replay and validation operations
    
    This class provides high-level methods for validating signals,
    generating reports, and managing validation workflows.
    """
    
    def __init__(self, db_url: str = None, user_timezone: str = 'Europe/Stockholm'):
        """
        Initialize the signal replay validator
        
        Args:
            db_url: Database connection URL (uses config default if None)
            user_timezone: User's timezone for timestamp handling
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize database manager
        database_url = db_url or getattr(config, 'DATABASE_URL', '')
        if not database_url:
            raise ValueError("Database URL not provided and not found in config")
        
        try:
            self.db_manager = DatabaseManager(database_url)
            self.logger.info("✅ Database connection established")
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            raise
        
        # Initialize replay engine
        self.replay_engine = ReplayEngine(
            db_manager=self.db_manager,
            user_timezone=user_timezone
        )
        
        # Initialize reporter
        self.reporter = ValidationReporter(use_colors=True)
        
        self.logger.info(f"🚀 SignalReplayValidator initialized")
        self.logger.info(f"   Timezone: {user_timezone}")
    
    def validate_single_signal(
        self,
        epic: str,
        timestamp: datetime,
        timeframe: str = '15m',
        strategy: str = None,
        show_calculations: bool = False,
        show_raw_data: bool = False,
        show_intermediate_steps: bool = False,
        compare_with_stored: bool = True,
        export_json: bool = False,
        output_file: str = None
    ) -> ValidationResult:
        """
        Validate a single signal and generate report
        
        Args:
            epic: Epic code to validate
            timestamp: Target timestamp
            timeframe: Timeframe for analysis
            strategy: Specific strategy to focus on
            show_calculations: Show detailed calculations
            show_raw_data: Include raw market data
            show_intermediate_steps: Show decision steps
            compare_with_stored: Compare with stored alerts
            export_json: Export results to JSON
            output_file: Output file path
            
        Returns:
            ValidationResult object
        """
        try:
            self.logger.info(f"🔍 Validating single signal: {epic}")
            
            # Perform validation
            result = self.replay_engine.validate_signal_at_timestamp(
                epic=epic,
                timestamp=timestamp,
                timeframe=timeframe,
                strategy_filter=strategy,
                compare_with_stored=compare_with_stored,
                debug_mode=show_intermediate_steps
            )
            
            # Generate report
            report = self.reporter.generate_validation_report(
                result=result,
                show_calculations=show_calculations,
                show_raw_data=show_raw_data,
                show_intermediate_steps=show_intermediate_steps
            )
            
            # Output report
            if output_file:
                self._save_to_file(report, output_file)
                self.logger.info(f"📝 Report saved to: {output_file}")
            else:
                print(report)
            
            # Export JSON if requested
            if export_json:
                json_data = self.reporter.export_to_json(result)
                json_file = output_file.replace('.txt', '.json') if output_file else f"validation_{epic}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                self._save_to_file(json_data, json_file)
                self.logger.info(f"📄 JSON exported to: {json_file}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Single signal validation failed: {e}")
            raise
    
    def validate_real_trade_signal(
        self,
        epic: str,
        timestamp: datetime,
        timeframe: str = '15m',
        strategy: str = None,
        show_calculations: bool = False,
        show_raw_data: bool = False,
        show_trade_outcome: bool = False,
        compare_with_live: bool = False,
        analyze_outcome: bool = False,
        timeframe_context: List[str] = None,
        show_risk_reward: bool = False,
        export_json: bool = False,
        output_file: str = None
    ) -> ValidationResult:
        """
        Validate a real trade signal with enhanced analysis
        
        This method provides comprehensive analysis for signals that resulted in actual trades,
        including outcome analysis, live data comparison, and multi-timeframe context.
        
        Args:
            epic: Epic code to validate
            timestamp: Exact timestamp when the trade was placed
            timeframe: Primary timeframe for analysis
            strategy: Specific strategy to focus on
            show_calculations: Show detailed calculations
            show_raw_data: Include raw market data
            show_trade_outcome: Show what happened after the signal
            compare_with_live: Compare with stored live signal data
            analyze_outcome: Perform comprehensive outcome analysis
            timeframe_context: List of timeframes for context analysis (e.g., ['15m', '1h', '4h'])
            show_risk_reward: Calculate risk/reward based on price movement
            export_json: Export results to JSON
            output_file: Output file path
            
        Returns:
            ValidationResult object with enhanced real trade analysis
        """
        try:
            self.logger.info(f"🎯 Analyzing real trade signal: {epic} @ {timestamp}")
            self.logger.info(f"   Enhanced analysis enabled with outcome tracking")
            
            # Step 1: Standard signal validation
            result = self.replay_engine.validate_signal_at_timestamp(
                epic=epic,
                timestamp=timestamp,
                timeframe=timeframe,
                strategy_filter=strategy,
                compare_with_stored=compare_with_live,
                debug_mode=True  # Always enable debug for real trade analysis
            )
            
            # Step 2: Enhanced analysis for real trades
            if result.success and result.signal_detected:
                # Add real trade analysis data to the result
                result.real_trade_analysis = {}
                
                # Step 3: Trade outcome analysis
                if show_trade_outcome or analyze_outcome:
                    self.logger.info("📊 Analyzing trade outcome...")
                    outcome_data = self._analyze_trade_outcome(epic, timestamp, timeframe)
                    result.real_trade_analysis['outcome'] = outcome_data
                
                # Step 4: Live signal comparison  
                if compare_with_live:
                    self.logger.info("🔄 Comparing with live signal data...")
                    live_comparison = self._compare_with_live_data(epic, timestamp)
                    result.real_trade_analysis['live_comparison'] = live_comparison
                
                # Step 5: Multi-timeframe context
                if timeframe_context:
                    self.logger.info(f"📈 Analyzing multi-timeframe context: {timeframe_context}")
                    context_data = self._analyze_timeframe_context(epic, timestamp, timeframe_context)
                    result.real_trade_analysis['timeframe_context'] = context_data
                
                # Step 6: Risk/reward analysis
                if show_risk_reward:
                    self.logger.info("⚖️ Calculating risk/reward ratio...")
                    risk_reward_data = self._calculate_risk_reward(epic, timestamp, result.signal_data)
                    result.real_trade_analysis['risk_reward'] = risk_reward_data
            
            # Generate enhanced report
            report = self._generate_real_trade_report(
                result=result,
                show_calculations=show_calculations,
                show_raw_data=show_raw_data,
                show_trade_outcome=show_trade_outcome,
                analyze_outcome=analyze_outcome,
                show_risk_reward=show_risk_reward
            )
            
            # Output report
            if output_file:
                self._save_to_file(report, output_file)
                self.logger.info(f"📝 Real trade analysis saved to: {output_file}")
            else:
                print(report)
            
            # Export JSON if requested
            if export_json:
                json_data = self._export_real_trade_json(result)
                json_file = output_file.replace('.txt', '.json') if output_file else f"real_trade_analysis_{epic}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                self._save_to_file(json_data, json_file)
                self.logger.info(f"📄 Real trade JSON exported to: {json_file}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Real trade signal validation failed: {e}")
            raise
    
    def validate_batch_signals(
        self,
        epic_list: List[str],
        timestamp: datetime,
        timeframe: str = '15m',
        strategy: str = None,
        compare_with_stored: bool = True,
        show_progress: bool = True,
        export_json: bool = False,
        output_file: str = None
    ) -> List[ValidationResult]:
        """
        Validate multiple signals and generate summary report
        
        Args:
            epic_list: List of epics to validate
            timestamp: Target timestamp
            timeframe: Timeframe for analysis
            strategy: Specific strategy to focus on
            compare_with_stored: Compare with stored alerts
            show_progress: Show progress indicators
            export_json: Export results to JSON
            output_file: Output file path
            
        Returns:
            List of ValidationResult objects
        """
        try:
            self.logger.info(f"🔍 Validating batch signals: {len(epic_list)} epics")
            
            # Perform batch validation
            results = self.replay_engine.validate_batch_signals(
                epic_list=epic_list,
                timestamp=timestamp,
                timeframe=timeframe,
                strategy_filter=strategy,
                compare_with_stored=compare_with_stored,
                show_progress=show_progress
            )
            
            # Generate summary report
            summary = self.reporter.generate_batch_summary(
                results=results,
                show_statistics=True,
                show_failures=True
            )
            
            # Output report
            if output_file:
                self._save_to_file(summary, output_file)
                self.logger.info(f"📝 Summary report saved to: {output_file}")
            else:
                print(summary)
            
            # Export JSON if requested
            if export_json:
                json_data = self.reporter.export_to_json(results)
                json_file = output_file.replace('.txt', '.json') if output_file else f"batch_validation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                self._save_to_file(json_data, json_file)
                self.logger.info(f"📄 JSON exported to: {json_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Batch validation failed: {e}")
            raise
    
    def validate_time_series(
        self,
        epic: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        interval_minutes: int = 60,
        timeframe: str = '15m',
        strategy: str = None,
        export_json: bool = False,
        output_file: str = None
    ) -> List[ValidationResult]:
        """
        Validate signals across a time series
        
        Args:
            epic: Epic to validate
            start_timestamp: Start of time series
            end_timestamp: End of time series
            interval_minutes: Interval between validations
            timeframe: Timeframe for analysis
            strategy: Specific strategy to focus on
            export_json: Export results to JSON
            output_file: Output file path
            
        Returns:
            List of ValidationResult objects
        """
        try:
            self.logger.info(f"📈 Time series validation: {epic}")
            
            # Perform time series validation
            results = self.replay_engine.validate_time_series(
                epic=epic,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                interval_minutes=interval_minutes,
                timeframe=timeframe,
                strategy_filter=strategy
            )
            
            # Generate time series report
            report_lines = []
            report_lines.append(f"🕒 Time Series Validation Report")
            report_lines.append(f"=" * 80)
            report_lines.append(f"Epic: {epic}")
            report_lines.append(f"Time Range: {start_timestamp.strftime('%Y-%m-%d %H:%M')} → {end_timestamp.strftime('%Y-%m-%d %H:%M')}")
            report_lines.append(f"Interval: {interval_minutes} minutes")
            report_lines.append(f"Total Points: {len(results)}")
            report_lines.append("")
            
            # Statistics
            successful = sum(1 for r in results if r.success)
            signals_found = sum(1 for r in results if r.signal_detected)
            
            report_lines.append(f"📊 Summary:")
            report_lines.append(f"   Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
            report_lines.append(f"   Signals Found: {signals_found}/{len(results)} ({signals_found/len(results)*100:.1f}%)")
            report_lines.append("")
            
            # Signal details
            if signals_found > 0:
                report_lines.append(f"🎯 Detected Signals:")
                for result in results:
                    if result.signal_detected:
                        signal_type = result.signal_data.get('signal_type', 'UNKNOWN')
                        confidence = result.signal_data.get('confidence_score', 0)
                        timestamp_str = result.timestamp.strftime('%H:%M:%S')
                        report_lines.append(f"   {timestamp_str}: {signal_type} ({confidence:.1%})")
            
            report = "\n".join(report_lines)
            
            # Output report
            if output_file:
                self._save_to_file(report, output_file)
                self.logger.info(f"📝 Time series report saved to: {output_file}")
            else:
                print(report)
            
            # Export JSON if requested
            if export_json:
                json_data = self.reporter.export_to_json(results)
                json_file = output_file.replace('.txt', '.json') if output_file else f"timeseries_{epic}_{start_timestamp.strftime('%Y%m%d')}.json"
                self._save_to_file(json_data, json_file)
                self.logger.info(f"📄 JSON exported to: {json_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Time series validation failed: {e}")
            raise
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        stats = self.replay_engine.get_performance_stats()
        
        lines = []
        lines.append("⚡ Performance Report")
        lines.append("=" * 40)
        lines.append(f"Validations: {stats['validations_performed']}")
        lines.append(f"Success Rate: {stats['success_rate']:.1f}%")
        lines.append(f"Signal Detection Rate: {stats['signal_detection_rate']:.1f}%")
        lines.append(f"Average Processing Time: {stats['average_processing_time_ms']:.1f}ms")
        lines.append(f"Total Processing Time: {stats['total_processing_time_seconds']:.2f}s")
        lines.append(f"Cache Hits: {stats['cache_hits']}")
        lines.append(f"Parallel Executions: {stats['parallel_executions']}")
        
        return "\n".join(lines)
    
    def _save_to_file(self, content: str, filename: str) -> None:
        """Save content to file"""
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.error(f"❌ Failed to save file {filename}: {e}")
            raise
    
    def _analyze_trade_outcome(self, epic: str, signal_timestamp: datetime, timeframe: str) -> Dict[str, Any]:
        """Analyze what happened after the trade signal was placed"""
        try:
            from .historical_data_manager import HistoricalDataManager
            
            # Initialize historical data manager for precise data fetching
            data_manager = HistoricalDataManager(self.db_manager)
            
            # Fetch price data for next 1h, 4h, and 24h after signal
            outcome_data = {
                'signal_price': None,
                'price_movements': {},
                'profit_loss_analysis': {},
                'volatility_analysis': {},
                'session_analysis': {}
            }
            
            # Get signal price (close price at signal timestamp)
            signal_data = data_manager.get_market_data_at_timestamp(epic, signal_timestamp, timeframe)
            if signal_data and not signal_data.empty:
                outcome_data['signal_price'] = float(signal_data.iloc[-1]['close'])
                self.logger.info(f"📍 Signal price: {outcome_data['signal_price']}")
            
            # Analyze price movements at different intervals
            intervals = [
                ('1h', signal_timestamp + timedelta(hours=1)),
                ('4h', signal_timestamp + timedelta(hours=4)),  
                ('24h', signal_timestamp + timedelta(hours=24)),
                ('1w', signal_timestamp + timedelta(days=7))
            ]
            
            for interval_name, end_time in intervals:
                try:
                    price_data = data_manager.get_market_data_range(epic, signal_timestamp, end_time, timeframe)
                    if price_data is not None and not price_data.empty:
                        high_price = float(price_data['high'].max())
                        low_price = float(price_data['low'].min())
                        end_price = float(price_data.iloc[-1]['close'])
                        
                        if outcome_data['signal_price']:
                            pips_high = (high_price - outcome_data['signal_price']) * 10000
                            pips_low = (low_price - outcome_data['signal_price']) * 10000
                            pips_end = (end_price - outcome_data['signal_price']) * 10000
                            
                            outcome_data['price_movements'][interval_name] = {
                                'high_price': high_price,
                                'low_price': low_price,
                                'end_price': end_price,
                                'pips_to_high': round(pips_high, 1),
                                'pips_to_low': round(pips_low, 1),
                                'pips_to_end': round(pips_end, 1),
                                'max_favorable': max(pips_high, abs(pips_low)),
                                'max_adverse': min(pips_high, pips_low)
                            }
                            
                            self.logger.info(f"📈 {interval_name}: High +{pips_high:.1f} pips, Low {pips_low:.1f} pips, End {pips_end:.1f} pips")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not analyze {interval_name} outcome: {e}")
                    outcome_data['price_movements'][interval_name] = {'error': str(e)}
            
            return outcome_data
            
        except Exception as e:
            self.logger.error(f"❌ Trade outcome analysis failed: {e}")
            return {'error': str(e)}
    
    def _compare_with_live_data(self, epic: str, timestamp: datetime) -> Dict[str, Any]:
        """Compare replayed signal with stored live signal data"""
        try:
            # Query alert_history table for signals around this timestamp
            # Allow ±2 minutes window to account for timing differences
            start_window = timestamp - timedelta(minutes=2)
            end_window = timestamp + timedelta(minutes=2)
            
            query = """
            SELECT * FROM alert_history 
            WHERE epic = %s 
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp ASC
            """
            
            import pandas as pd
            with self.db_manager.get_engine().connect() as conn:
                stored_signals = pd.read_sql(query, conn, params=(epic, start_window, end_window))
            
            comparison_data = {
                'found_stored_signals': len(stored_signals),
                'closest_match': None,
                'comparison_results': {},
                'confidence_comparison': None
            }
            
            if not stored_signals.empty:
                # Find closest timestamp match
                stored_signals['time_diff'] = abs((pd.to_datetime(stored_signals['timestamp']) - timestamp).dt.total_seconds())
                closest_idx = stored_signals['time_diff'].idxmin()
                closest_signal = stored_signals.iloc[closest_idx]
                
                comparison_data['closest_match'] = {
                    'timestamp': str(closest_signal['timestamp']),
                    'time_difference_seconds': int(closest_signal['time_diff']),
                    'signal_type': closest_signal.get('signal_type', 'Unknown'),
                    'confidence': closest_signal.get('confidence_score', None),
                    'strategy': closest_signal.get('strategy', 'Unknown')
                }
                
                self.logger.info(f"🔍 Found closest stored signal: {closest_signal['signal_type']} at {closest_signal['timestamp']} ({closest_signal['time_diff']:.0f}s difference)")
                
                # Compare confidence scores and other metrics
                if 'confidence_score' in closest_signal:
                    comparison_data['confidence_comparison'] = {
                        'stored_confidence': float(closest_signal['confidence_score']),
                        'difference_analysis': 'Available after replay validation'
                    }
            else:
                self.logger.warning(f"⚠️ No stored signals found for {epic} around {timestamp}")
                comparison_data['found_stored_signals'] = 0
            
            return comparison_data
            
        except Exception as e:
            self.logger.error(f"❌ Live data comparison failed: {e}")
            return {'error': str(e)}
    
    def _analyze_timeframe_context(self, epic: str, timestamp: datetime, timeframes: List[str]) -> Dict[str, Any]:
        """Analyze signal context across multiple timeframes"""
        try:
            from .historical_data_manager import HistoricalDataManager
            data_manager = HistoricalDataManager(self.db_manager)
            
            context_data = {}
            
            for tf in timeframes:
                try:
                    self.logger.info(f"📊 Analyzing {tf} context...")
                    
                    # Get data for context analysis (more data for higher timeframes)
                    lookback_hours = {'15m': 8, '1h': 24, '4h': 168, '1d': 720}.get(tf, 24)
                    start_time = timestamp - timedelta(hours=lookback_hours)
                    
                    market_data = data_manager.get_market_data_range(epic, start_time, timestamp, tf)
                    
                    if market_data is not None and not market_data.empty:
                        # Calculate context metrics
                        current_price = float(market_data.iloc[-1]['close'])
                        
                        # Trend analysis
                        if len(market_data) >= 20:
                            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
                            trend_direction = "UP" if current_price > sma_20 else "DOWN"
                        else:
                            trend_direction = "INSUFFICIENT_DATA"
                        
                        # Support/Resistance levels
                        recent_high = market_data['high'].tail(20).max()
                        recent_low = market_data['low'].tail(20).min()
                        
                        # Volatility
                        volatility = market_data['close'].tail(20).std() if len(market_data) >= 20 else None
                        
                        context_data[tf] = {
                            'current_price': current_price,
                            'trend_direction': trend_direction,
                            'recent_high': float(recent_high),
                            'recent_low': float(recent_low),
                            'volatility': float(volatility) if volatility else None,
                            'distance_to_high_pips': round((recent_high - current_price) * 10000, 1),
                            'distance_to_low_pips': round((current_price - recent_low) * 10000, 1),
                            'candles_analyzed': len(market_data)
                        }
                        
                        self.logger.info(f"   {tf}: {trend_direction} trend, {context_data[tf]['distance_to_high_pips']}p to high, {context_data[tf]['distance_to_low_pips']}p above low")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not analyze {tf} context: {e}")
                    context_data[tf] = {'error': str(e)}
            
            return context_data
            
        except Exception as e:
            self.logger.error(f"❌ Timeframe context analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_reward(self, epic: str, timestamp: datetime, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk/reward ratio based on subsequent price movement"""
        try:
            # This will use the trade outcome data to calculate risk/reward
            outcome_data = self._analyze_trade_outcome(epic, timestamp, '15m')
            
            risk_reward_data = {
                'analysis_available': False,
                'risk_reward_ratios': {},
                'trade_quality_assessment': {}
            }
            
            if outcome_data.get('signal_price') and outcome_data.get('price_movements'):
                risk_reward_data['analysis_available'] = True
                signal_price = outcome_data['signal_price']
                signal_type = signal_data.get('signal_type', 'UNKNOWN')
                
                # Calculate R:R for different time horizons
                for interval, movement in outcome_data['price_movements'].items():
                    if 'error' not in movement:
                        if signal_type.upper() == 'BULL':
                            # For BULL signals, reward is upward movement, risk is downward
                            max_reward = movement['pips_to_high']
                            max_risk = abs(movement['pips_to_low']) if movement['pips_to_low'] < 0 else 0
                        else:
                            # For BEAR signals, reward is downward movement, risk is upward  
                            max_reward = abs(movement['pips_to_low']) if movement['pips_to_low'] < 0 else 0
                            max_risk = movement['pips_to_high'] if movement['pips_to_high'] > 0 else 0
                        
                        if max_risk > 0:
                            risk_reward_ratio = round(max_reward / max_risk, 2)
                        else:
                            risk_reward_ratio = float('inf') if max_reward > 0 else 0
                        
                        risk_reward_data['risk_reward_ratios'][interval] = {
                            'max_reward_pips': round(max_reward, 1),
                            'max_risk_pips': round(max_risk, 1),
                            'risk_reward_ratio': risk_reward_ratio,
                            'end_result_pips': round(movement['pips_to_end'], 1)
                        }
                        
                        self.logger.info(f"⚖️ {interval} R:R: {max_reward:.1f}:{max_risk:.1f} = {risk_reward_ratio:.2f}")
                
                # Overall trade quality assessment
                hour_1_data = risk_reward_data['risk_reward_ratios'].get('1h', {})
                if hour_1_data:
                    if hour_1_data.get('risk_reward_ratio', 0) >= 2.0:
                        quality = "EXCELLENT"
                    elif hour_1_data.get('risk_reward_ratio', 0) >= 1.5:
                        quality = "GOOD"  
                    elif hour_1_data.get('risk_reward_ratio', 0) >= 1.0:
                        quality = "FAIR"
                    else:
                        quality = "POOR"
                    
                    risk_reward_data['trade_quality_assessment'] = {
                        'overall_quality': quality,
                        'primary_ratio': hour_1_data.get('risk_reward_ratio', 0),
                        'assessment_timeframe': '1h'
                    }
            
            return risk_reward_data
            
        except Exception as e:
            self.logger.error(f"❌ Risk/reward calculation failed: {e}")
            return {'error': str(e)}
    
    def _generate_real_trade_report(self, result: ValidationResult, **kwargs) -> str:
        """Generate comprehensive report for real trade analysis"""
        try:
            # Start with standard validation report
            base_report = self.reporter.generate_validation_report(
                result=result,
                show_calculations=kwargs.get('show_calculations', False),
                show_raw_data=kwargs.get('show_raw_data', False),
                show_intermediate_steps=True  # Always show steps for real trades
            )
            
            # Add enhanced real trade analysis sections
            enhanced_sections = []
            
            if hasattr(result, 'real_trade_analysis') and result.real_trade_analysis:
                enhanced_sections.append("\n" + "="*80)
                enhanced_sections.append("🎯 REAL TRADE ANALYSIS")
                enhanced_sections.append("="*80)
                
                # Trade outcome analysis
                if 'outcome' in result.real_trade_analysis:
                    outcome = result.real_trade_analysis['outcome']
                    enhanced_sections.append("\n📊 TRADE OUTCOME ANALYSIS")
                    enhanced_sections.append("-" * 40)
                    
                    if outcome.get('signal_price'):
                        enhanced_sections.append(f"Signal Price: {outcome['signal_price']}")
                    
                    for interval, movement in outcome.get('price_movements', {}).items():
                        if 'error' not in movement:
                            enhanced_sections.append(f"\n{interval.upper()} Performance:")
                            enhanced_sections.append(f"  High: +{movement['pips_to_high']:.1f} pips")
                            enhanced_sections.append(f"  Low:  {movement['pips_to_low']:.1f} pips") 
                            enhanced_sections.append(f"  End:  {movement['pips_to_end']:.1f} pips")
                
                # Live comparison
                if 'live_comparison' in result.real_trade_analysis:
                    comparison = result.real_trade_analysis['live_comparison']
                    enhanced_sections.append("\n🔄 LIVE SIGNAL COMPARISON")
                    enhanced_sections.append("-" * 40)
                    
                    if comparison.get('closest_match'):
                        match = comparison['closest_match']
                        enhanced_sections.append(f"Stored Signal Found: {match['signal_type']}")
                        enhanced_sections.append(f"Time Difference: {match['time_difference_seconds']}s")
                        enhanced_sections.append(f"Stored Confidence: {match.get('confidence', 'N/A')}")
                        enhanced_sections.append(f"Strategy: {match.get('strategy', 'Unknown')}")
                    else:
                        enhanced_sections.append("No matching stored signals found")
                
                # Risk/reward analysis
                if 'risk_reward' in result.real_trade_analysis:
                    rr = result.real_trade_analysis['risk_reward']
                    enhanced_sections.append("\n⚖️ RISK/REWARD ANALYSIS")
                    enhanced_sections.append("-" * 40)
                    
                    if rr.get('analysis_available'):
                        for interval, ratio_data in rr.get('risk_reward_ratios', {}).items():
                            enhanced_sections.append(f"\n{interval.upper()}:")
                            enhanced_sections.append(f"  Max Reward: +{ratio_data['max_reward_pips']:.1f} pips")
                            enhanced_sections.append(f"  Max Risk:   -{ratio_data['max_risk_pips']:.1f} pips") 
                            enhanced_sections.append(f"  R:R Ratio:  {ratio_data['risk_reward_ratio']:.2f}")
                            enhanced_sections.append(f"  Final:      {ratio_data['end_result_pips']:.1f} pips")
                        
                        if 'trade_quality_assessment' in rr:
                            assessment = rr['trade_quality_assessment']
                            enhanced_sections.append(f"\nTrade Quality: {assessment.get('overall_quality', 'UNKNOWN')}")
                            enhanced_sections.append(f"Primary R:R: {assessment.get('primary_ratio', 0):.2f}")
                    else:
                        enhanced_sections.append("Risk/reward analysis not available")
                
                # Timeframe context
                if 'timeframe_context' in result.real_trade_analysis:
                    context = result.real_trade_analysis['timeframe_context']
                    enhanced_sections.append("\n📈 MULTI-TIMEFRAME CONTEXT")
                    enhanced_sections.append("-" * 40)
                    
                    for tf, data in context.items():
                        if 'error' not in data:
                            enhanced_sections.append(f"\n{tf.upper()} Context:")
                            enhanced_sections.append(f"  Trend: {data.get('trend_direction', 'Unknown')}")
                            enhanced_sections.append(f"  Distance to High: +{data.get('distance_to_high_pips', 0):.1f} pips")
                            enhanced_sections.append(f"  Distance to Low:  +{data.get('distance_to_low_pips', 0):.1f} pips")
                            if data.get('volatility'):
                                enhanced_sections.append(f"  Volatility: {data['volatility']:.5f}")
            
            # Combine base report with enhanced sections
            return base_report + "\n".join(enhanced_sections)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate real trade report: {e}")
            return f"Report generation failed: {e}"
    
    def _export_real_trade_json(self, result: ValidationResult) -> str:
        """Export real trade analysis to JSON format"""
        try:
            # Start with base JSON export
            base_json = json.loads(self.reporter.export_to_json(result))
            
            # Add real trade analysis data
            if hasattr(result, 'real_trade_analysis') and result.real_trade_analysis:
                base_json['real_trade_analysis'] = result.real_trade_analysis
                base_json['analysis_type'] = 'real_trade_validation'
                base_json['enhanced_features'] = [
                    'trade_outcome_analysis',
                    'live_signal_comparison', 
                    'multi_timeframe_context',
                    'risk_reward_calculation'
                ]
            
            return json.dumps(base_json, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export real trade JSON: {e}")
            return json.dumps({'error': str(e)})

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.replay_engine.cleanup_resources()
            self.logger.info("🧹 SignalReplayValidator cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='Signal Replay Validator - Validate historical trading signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single signal
  python -m forex_scanner.validation.signal_replay_validator \\
    --timestamp "2025-01-15 14:30:00" \\
    --epic "CS.D.EURUSD.MINI.IP" \\
    --show-calculations \\
    --show-raw-data

  # Validate all epics at timestamp
  python -m forex_scanner.validation.signal_replay_validator \\
    --timestamp "2025-01-15 14:30:00" \\
    --all-epics \\
    --compare-with-stored

  # Debug specific strategy
  python -m forex_scanner.validation.signal_replay_validator \\
    --timestamp "2025-01-15 14:30:00" \\
    --epic "CS.D.EURUSD.MINI.IP" \\
    --strategy "EMA" \\
    --debug-mode \\
    --show-intermediate-steps

  # Time series analysis
  python -m forex_scanner.validation.signal_replay_validator \\
    --epic "CS.D.EURUSD.MINI.IP" \\
    --start-time "2025-01-15 08:00:00" \\
    --end-time "2025-01-15 18:00:00" \\
    --interval 30 \\
    --time-series

  # Real trade analysis (NEW)
  python -m forex_scanner.validation.signal_replay_validator \\
    --timestamp "2025-09-04 17:31:42" \\
    --epic "CS.D.EURUSD.MINI.IP" \\
    --real-trade \\
    --show-trade-outcome \\
    --compare-with-live \\
    --show-calculations

  # Comprehensive outcome analysis
  python -m forex_scanner.validation.signal_replay_validator \\
    --timestamp "2025-09-04 17:31:42" \\
    --epic "CS.D.EURUSD.MINI.IP" \\
    --real-trade \\
    --analyze-outcome \\
    --timeframe-context "15m,1h,4h" \\
    --show-risk-reward
        """
    )
    
    # Required arguments (mutually exclusive groups)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--timestamp', type=str,
                      help='Target timestamp (format: "YYYY-MM-DD HH:MM:SS")')
    group.add_argument('--time-series', action='store_true',
                      help='Run time series validation (requires --start-time and --end-time)')
    
    # Epic selection
    epic_group = parser.add_mutually_exclusive_group()
    epic_group.add_argument('--epic', type=str,
                           help='Single epic to validate (e.g., "CS.D.EURUSD.MINI.IP")')
    epic_group.add_argument('--all-epics', action='store_true',
                           help='Validate all configured epics')
    epic_group.add_argument('--epic-list', type=str, nargs='+',
                           help='List of epics to validate')
    
    # Time series arguments
    parser.add_argument('--start-time', type=str,
                       help='Start time for time series (format: "YYYY-MM-DD HH:MM:SS")')
    parser.add_argument('--end-time', type=str,
                       help='End time for time series (format: "YYYY-MM-DD HH:MM:SS")')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval in minutes for time series (default: 60)')
    
    # Analysis options
    parser.add_argument('--timeframe', type=str, default='15m',
                       choices=['5m', '15m', '30m', '1h'],
                       help='Timeframe for analysis (default: 15m)')
    parser.add_argument('--strategy', type=str,
                       choices=['ema', 'macd', 'kama', 'zero_lag', 'momentum_bias'],
                       help='Focus on specific strategy')
    
    # Output options
    parser.add_argument('--show-calculations', action='store_true',
                       help='Show detailed calculations')
    parser.add_argument('--show-raw-data', action='store_true',
                       help='Include raw market data in output')
    parser.add_argument('--show-intermediate-steps', action='store_true',
                       help='Show intermediate decision steps')
    parser.add_argument('--debug-mode', action='store_true',
                       help='Enable debug mode with verbose logging')
    
    # Comparison options
    parser.add_argument('--compare-with-stored', action='store_true', default=True,
                       help='Compare with stored alerts (default: True)')
    parser.add_argument('--no-compare', action='store_true',
                       help='Skip comparison with stored alerts')
    
    # Real trade analysis options
    parser.add_argument('--real-trade', action='store_true',
                       help='Analyze a real trade that was placed (enables enhanced analysis)')
    parser.add_argument('--show-trade-outcome', action='store_true',
                       help='Show what happened after the signal (price movement, P&L analysis)')
    parser.add_argument('--compare-with-live', action='store_true',
                       help='Compare replayed results with live signal data from alert history')
    parser.add_argument('--analyze-outcome', action='store_true',
                       help='Comprehensive outcome analysis with risk/reward calculation')
    parser.add_argument('--timeframe-context', type=str,
                       help='Show multi-timeframe context (e.g., "15m,1h,4h")')
    parser.add_argument('--show-risk-reward', action='store_true',
                       help='Calculate and show risk/reward ratio based on subsequent price action')
    
    # Export options
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON format')
    parser.add_argument('--output-file', type=str,
                       help='Output file path (default: stdout)')
    
    # Performance options
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--show-performance', action='store_true',
                       help='Show performance statistics')
    
    # Logging options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress non-essential output')
    
    return parser


def setup_logging(verbose: bool = False, quiet: bool = False, debug_mode: bool = False) -> None:
    """Setup logging configuration"""
    if quiet:
        level = logging.WARNING
    elif debug_mode or verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string to datetime object"""
    try:
        # Try with seconds
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        try:
            # Try without seconds
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {timestamp_str}. Use 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM'")
    
    # Add UTC timezone
    return dt.replace(tzinfo=timezone.utc)


def get_epic_list(args) -> List[str]:
    """Get list of epics based on arguments"""
    if args.epic:
        return [args.epic]
    elif args.epic_list:
        return args.epic_list
    elif args.all_epics:
        return ReplayConfig.DEFAULT_EPIC_LIST
    else:
        # Default to EURUSD if no epic specified for time series
        return ['CS.D.EURUSD.MINI.IP']


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        verbose=args.verbose,
        quiet=args.quiet,
        debug_mode=args.debug_mode
    )
    
    logger = logging.getLogger(__name__)
    validator = None
    
    try:
        # Initialize validator
        validator = SignalReplayValidator()
        
        # Handle comparison options
        compare_with_stored = args.compare_with_stored and not args.no_compare
        
        if args.time_series:
            # Time series validation
            if not args.start_time or not args.end_time:
                parser.error("Time series validation requires --start-time and --end-time")
            
            start_ts = parse_timestamp(args.start_time)
            end_ts = parse_timestamp(args.end_time)
            epic_list = get_epic_list(args)
            
            if len(epic_list) > 1:
                parser.error("Time series validation supports only one epic at a time")
            
            results = validator.validate_time_series(
                epic=epic_list[0],
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                interval_minutes=args.interval,
                timeframe=args.timeframe,
                strategy=args.strategy,
                export_json=args.export_json,
                output_file=args.output_file
            )
            
        else:
            # Single timestamp validation
            target_ts = parse_timestamp(args.timestamp)
            epic_list = get_epic_list(args)
            
            if len(epic_list) == 1:
                # Check if this is real trade analysis
                if args.real_trade:
                    # Real trade signal validation with enhanced analysis
                    logger.info("🎯 Real trade analysis mode enabled")
                    
                    # Parse timeframe context if provided
                    timeframe_context = None
                    if args.timeframe_context:
                        timeframe_context = [tf.strip() for tf in args.timeframe_context.split(',')]
                        logger.info(f"📈 Multi-timeframe context: {timeframe_context}")
                    
                    result = validator.validate_real_trade_signal(
                        epic=epic_list[0],
                        timestamp=target_ts,
                        timeframe=args.timeframe,
                        strategy=args.strategy,
                        show_calculations=args.show_calculations,
                        show_raw_data=args.show_raw_data,
                        show_trade_outcome=args.show_trade_outcome,
                        compare_with_live=args.compare_with_live,
                        analyze_outcome=args.analyze_outcome,
                        timeframe_context=timeframe_context,
                        show_risk_reward=args.show_risk_reward,
                        export_json=args.export_json,
                        output_file=args.output_file
                    )
                else:
                    # Standard single epic validation
                    result = validator.validate_single_signal(
                        epic=epic_list[0],
                        timestamp=target_ts,
                        timeframe=args.timeframe,
                        strategy=args.strategy,
                        show_calculations=args.show_calculations,
                        show_raw_data=args.show_raw_data,
                        show_intermediate_steps=args.show_intermediate_steps,
                        compare_with_stored=compare_with_stored,
                        export_json=args.export_json,
                        output_file=args.output_file
                    )
                
                if not result.success:
                    logger.error(f"❌ Validation failed: {result.error_message}")
                    sys.exit(1)
            
            else:
                # Batch validation
                results = validator.validate_batch_signals(
                    epic_list=epic_list,
                    timestamp=target_ts,
                    timeframe=args.timeframe,
                    strategy=args.strategy,
                    compare_with_stored=compare_with_stored,
                    show_progress=not args.quiet,
                    export_json=args.export_json,
                    output_file=args.output_file
                )
                
                # Check if any validations failed
                failures = [r for r in results if not r.success]
                if failures:
                    logger.warning(f"⚠️ {len(failures)}/{len(results)} validations failed")
        
        # Show performance statistics if requested
        if args.show_performance:
            perf_report = validator.get_performance_report()
            print("\n" + perf_report)
        
        logger.info("✅ Validation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("🛑 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        if args.debug_mode:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Cleanup
        if validator:
            validator.cleanup()


if __name__ == '__main__':
    main()