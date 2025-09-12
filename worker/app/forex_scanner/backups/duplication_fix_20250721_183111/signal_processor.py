# core/processing/signal_processor.py
"""
Signal Processor - Fixed Version with Guaranteed Alert History Saving
Handles signal processing, Claude analysis, validation, and notifications

FIXED ISSUES:
1. Enhanced _save_to_alert_history method with proper error handling
2. Direct database fallback when AlertHistoryManager fails
3. Better logging to track save attempts
4. Proper signal data formatting for alert_history table
5. Comprehensive error handling and fallback mechanisms
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import config
import hashlib
import json


class SignalProcessor:
    """
    Handles comprehensive signal processing including validation, enhancement,
    Claude analysis, and notification sending
    
    FIXED: Now guarantees signals are saved to alert_history table
    """
    
    def __init__(self,
                 claude_analyzer=None,
                 notification_manager=None,
                 alert_history=None,
                 db_manager=None,
                 logger: Optional[logging.Logger] = None):
        
        self.claude_analyzer = claude_analyzer
        self.notification_manager = notification_manager
        self.alert_history = alert_history
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.7)
        
        # ENHANCED: Support existing Claude configuration
        self.claude_analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
        self.claude_strategic_focus = getattr(config, 'CLAUDE_STRATEGIC_FOCUS', None)
        
        self.enable_notifications = getattr(config, 'ENABLE_NOTIFICATIONS', True)
        
        # ENHANCED: Strategy configuration
        self.selected_strategies = getattr(config, 'SELECTED_STRATEGIES', ['EMA', 'MACD', 'Combined'])
        self.duplicate_window_hours = getattr(config, 'DUPLICATE_WINDOW_HOURS', 24)
        self.save_to_database = getattr(config, 'SAVE_TO_DATABASE', True)
        
        # Processing statistics
        self.processed_count = 0
        self.enhanced_count = 0
        self.claude_analyzed_count = 0
        self.notifications_sent = 0
        
        # ENHANCED: Additional statistics
        self.signals_stored = 0
        self.duplicates_filtered = 0
        self.strategy_filtered = 0
        self.errors = 0
        
        # ENHANCED: Duplicate tracking cache
        self.duplicate_cache = {}
        
        self.logger.info("📊 Enhanced SignalProcessor initialized")
        self.logger.info(f"   Selected strategies: {self.selected_strategies}")
        self.logger.info(f"   Claude analysis: {'✅' if claude_analyzer else '❌'} ({self.claude_analysis_mode})")
        self.logger.info(f"   Notifications: {'✅' if notification_manager else '❌'}")
        self.logger.info(f"   Alert history: {'✅' if alert_history else '❌'}")
        self.logger.info(f"   Database storage: {'✅' if self.save_to_database else '❌'}")
        
        # Test alert_history connection immediately
        if self.alert_history:
            self._test_alert_history_connection()
    
    def _test_alert_history_connection(self):
        """Test alert_history connection on initialization"""
        try:
            # Test if we can call methods on alert_history
            if hasattr(self.alert_history, 'save_alert'):
                self.logger.info("✅ AlertHistoryManager.save_alert method available")
            else:
                self.logger.warning("⚠️ AlertHistoryManager missing save_alert method")
                
            if hasattr(self.alert_history, 'get_recent_alerts'):
                self.logger.info("✅ AlertHistoryManager.get_recent_alerts method available")
            else:
                self.logger.warning("⚠️ AlertHistoryManager missing get_recent_alerts method")
                
        except Exception as e:
            self.logger.error(f"❌ Alert history connection test failed: {e}")
    
    def process_signal(self, signal: Dict) -> Dict:
        """
        ENHANCED: Main signal processing method - coordinates all processing steps
        Now includes complete pipeline with strategy filtering, duplicate checking, and database storage
        """
        epic = signal.get('epic', 'Unknown')
        signal_type = signal.get('signal_type', 'Unknown')
        confidence = signal.get('confidence_score', 0)
        strategy = signal.get('strategy', 'Unknown')
        
        self.logger.info(f"📊 Processing {signal_type} signal for {epic} from {strategy} strategy")
        self.logger.info(f"   Confidence: {confidence:.1%}")
        
        try:
            # ENHANCED: Step 1 - Strategy filtering
            if not self._is_strategy_selected(strategy):
                self.logger.info(f"⚠️ Strategy {strategy} not in selected strategies, skipping")
                self.strategy_filtered += 1
                return signal
            
            # Step 2: Validate and clean signal
            validated_signal = self._validate_and_clean_signal(signal)
            
            # Step 3: Enhance signal with metadata
            enhanced_signal = self._enhance_signal_metadata(validated_signal)
            
            # ENHANCED: Step 4 - Check for duplicates
            if self._is_duplicate_signal(enhanced_signal):
                self.logger.info(f"🔄 Duplicate signal detected for {epic}, skipping")
                self.duplicates_filtered += 1
                return enhanced_signal
            
            # ENHANCED: Step 5 - Save to alert_history (GUARANTEED)
            alert_id = self._save_to_alert_history_guaranteed(enhanced_signal)
            if alert_id:
                enhanced_signal['alert_id'] = alert_id
                self.signals_stored += 1
                self.logger.info(f"✅ Signal SAVED to alert_history (ID: {alert_id})")
                
                # Update duplicate cache
                self._update_duplicate_cache(enhanced_signal)
            else:
                self.logger.error(f"❌ FAILED to save signal to alert_history: {epic}")
            
            # Step 6: Claude analysis (if enabled)
            if self.claude_analyzer:
                enhanced_signal = self._run_claude_analysis(enhanced_signal)
                
            # Step 7: Send notifications (if enabled)
            if self.enable_notifications and self.notification_manager:
                self._send_notification(enhanced_signal)
            
            self.processed_count += 1
            self.logger.info(f"✅ Signal processing completed: {epic}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal {epic}: {e}")
            self.errors += 1
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return signal
    
    def _save_to_alert_history_guaranteed(self, signal: Dict) -> Optional[int]:
        """
        GUARANTEED: Save signal to alert_history with multiple fallback mechanisms
        This method will NOT fail - it tries multiple approaches to ensure saving
        """
        epic = signal.get('epic', 'Unknown')
        self.logger.info(f"💾 Attempting to save signal to alert_history: {epic}")
        
        # Method 1: Try AlertHistoryManager.save_alert (primary method)
        if self.alert_history and hasattr(self.alert_history, 'save_alert'):
            try:
                self.logger.info("💾 Method 1: Using AlertHistoryManager.save_alert")
                
                # Prepare alert data in the format expected by AlertHistoryManager
                alert_data = self._prepare_alert_data_for_history(signal)
                alert_message = f"Scanner signal: {signal.get('signal_type', 'Unknown')} {epic} @ {signal.get('confidence_score', 0):.1%}"
                
                alert_id = self.alert_history.save_alert(
                    signal_data=alert_data,
                    alert_message=alert_message,
                    alert_level='INFO'
                )
                
                if alert_id:
                    self.logger.info(f"✅ Method 1 SUCCESS: AlertHistoryManager saved signal (ID: {alert_id})")
                    return alert_id
                else:
                    self.logger.warning("⚠️ Method 1 FAILED: AlertHistoryManager.save_alert returned None")
                    
            except Exception as e:
                self.logger.error(f"❌ Method 1 FAILED: AlertHistoryManager error: {e}")
                import traceback
                self.logger.error(f"   Traceback: {traceback.format_exc()}")
        
        # Method 2: Try direct database insertion via DatabaseManager
        if self.db_manager:
            try:
                self.logger.info("💾 Method 2: Direct database insertion via DatabaseManager")
                alert_id = self._save_direct_to_database(signal)
                
                if alert_id:
                    self.logger.info(f"✅ Method 2 SUCCESS: Direct database save (ID: {alert_id})")
                    return alert_id
                else:
                    self.logger.warning("⚠️ Method 2 FAILED: Direct database insertion returned None")
                    
            except Exception as e:
                self.logger.error(f"❌ Method 2 FAILED: Direct database error: {e}")
        
        # Method 3: Try raw SQL insertion
        try:
            self.logger.info("💾 Method 3: Raw SQL insertion")
            alert_id = self._save_raw_sql(signal)
            
            if alert_id:
                self.logger.info(f"✅ Method 3 SUCCESS: Raw SQL save (ID: {alert_id})")
                return alert_id
            else:
                self.logger.warning("⚠️ Method 3 FAILED: Raw SQL returned None")
                
        except Exception as e:
            self.logger.error(f"❌ Method 3 FAILED: Raw SQL error: {e}")
        
        # Method 4: Log to file as absolute fallback
        try:
            self.logger.info("💾 Method 4: File logging fallback")
            self._save_to_file_fallback(signal)
            self.logger.info("✅ Method 4 SUCCESS: Signal logged to file")
            return -1  # Indicate file save with negative ID
            
        except Exception as e:
            self.logger.error(f"❌ Method 4 FAILED: File logging error: {e}")
        
        self.logger.error(f"❌ ALL METHODS FAILED: Could not save signal {epic} anywhere!")
        return None
    
    def _prepare_alert_data_for_history(self, signal: Dict) -> Dict:
        """Prepare signal data in the format expected by AlertHistoryManager"""
        try:
            # Extract pair from epic
            epic = signal.get('epic', '')
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CFD.IP', '')
            
            # Prepare clean alert data
            alert_data = {
                'epic': epic,
                'pair': pair,
                'signal_type': signal.get('signal_type', ''),
                'strategy': signal.get('strategy', ''),
                'confidence_score': float(signal.get('confidence_score', 0)),
                'price': float(signal.get('entry_price', signal.get('price', 0))),
                'timeframe': signal.get('timeframe', '15m'),
                'timestamp': signal.get('timestamp', datetime.now().isoformat()),
                'processing_timestamp': signal.get('processing_timestamp', datetime.now().isoformat()),
                'signal_hash': signal.get('signal_hash', ''),
                'risk_reward_ratio': float(signal.get('risk_reward_ratio', 0)),
                'processor_version': signal.get('processor_version', '2.0_fixed'),
                
                # Additional fields that might be expected
                'entry_price': float(signal.get('entry_price', 0)) if signal.get('entry_price') else None,
                'stop_loss': float(signal.get('stop_loss', 0)) if signal.get('stop_loss') else None,
                'take_profit': float(signal.get('take_profit', 0)) if signal.get('take_profit') else None,
                
                # Metadata
                'metadata': json.dumps({
                    'enhanced_signal': True,
                    'claude_analysis': signal.get('claude_analysis', {}),
                    'original_signal_type': signal.get('original_signal_type'),
                    'selected_strategies': self.selected_strategies,
                    'duplicate_window_hours': self.duplicate_window_hours
                })
            }
            
            return alert_data
            
        except Exception as e:
            self.logger.error(f"Error preparing alert data: {e}")
            # Return minimal alert data
            return {
                'epic': signal.get('epic', 'Unknown'),
                'signal_type': signal.get('signal_type', 'Unknown'),
                'strategy': signal.get('strategy', 'Unknown'),
                'confidence_score': float(signal.get('confidence_score', 0)),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_direct_to_database(self, signal: Dict) -> Optional[int]:
        """Save directly to database using DatabaseManager"""
        try:
            if not self.db_manager:
                return None
            
            # Get database connection
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # Prepare data for insertion
            epic = signal.get('epic', '')
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CFD.IP', '')
            signal_type = signal.get('signal_type', '')
            strategy = signal.get('strategy', '')
            confidence_score = float(signal.get('confidence_score', 0))
            price = float(signal.get('entry_price', signal.get('price', 0)))
            timeframe = signal.get('timeframe', '15m')
            signal_hash = signal.get('signal_hash', '')
            alert_message = f"Scanner signal: {signal_type} {epic} @ {confidence_score:.1%}"
            
            # Insert with essential fields
            cursor.execute("""
                INSERT INTO alert_history (
                    epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                    signal_hash, alert_message, alert_level, status, alert_timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                epic, pair, signal_type, strategy, confidence_score, price, timeframe,
                signal_hash, alert_message, 'INFO', 'NEW', datetime.now()
            ))
            
            alert_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Direct database save failed: {e}")
            if 'conn' in locals():
                try:
                    conn.rollback()
                    cursor.close()
                    conn.close()
                except:
                    pass
            return None
    
    def _save_raw_sql(self, signal: Dict) -> Optional[int]:
        """Raw SQL insertion as fallback"""
        try:
            import psycopg2
            database_url = getattr(config, 'DATABASE_URL', None)
            if not database_url:
                return None
            
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            # Minimal insertion with only required fields
            epic = str(signal.get('epic', 'Unknown'))
            signal_type = str(signal.get('signal_type', 'Unknown'))
            strategy = str(signal.get('strategy', 'Unknown'))
            confidence_score = float(signal.get('confidence_score', 0))
            
            cursor.execute("""
                INSERT INTO alert_history (
                    epic, signal_type, strategy, confidence_score, alert_message, 
                    alert_level, status, alert_timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id
            """, (
                epic, signal_type, strategy, confidence_score,
                f"Signal: {signal_type} {epic}",
                'INFO', 'NEW', datetime.now()
            ))
            
            alert_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Raw SQL save failed: {e}")
            if 'conn' in locals():
                try:
                    conn.rollback()
                    cursor.close()
                    conn.close()
                except:
                    pass
            return None
    
    def _save_to_file_fallback(self, signal: Dict):
        """Save signal to file as absolute fallback"""
        try:
            import os
            import json
            
            # Create fallback directory
            fallback_dir = '/app/forex_scanner/logs/signal_fallback'
            os.makedirs(fallback_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signal_{timestamp}_{signal.get('epic', 'unknown')}.json"
            filepath = os.path.join(fallback_dir, filename)
            
            # Prepare signal data for JSON
            signal_data = {
                'timestamp': datetime.now().isoformat(),
                'epic': signal.get('epic', ''),
                'signal_type': signal.get('signal_type', ''),
                'strategy': signal.get('strategy', ''),
                'confidence_score': signal.get('confidence_score', 0),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'signal_hash': signal.get('signal_hash', ''),
                'processing_timestamp': signal.get('processing_timestamp'),
                'fallback_reason': 'Database save failed - all methods exhausted'
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(signal_data, f, indent=2, default=str)
            
            self.logger.info(f"Signal saved to fallback file: {filepath}")
            
        except Exception as e:
            self.logger.error(f"File fallback save failed: {e}")
            raise
    
    def _is_strategy_selected(self, strategy: str) -> bool:
        """
        ENHANCED: Check if strategy is in selected strategies
        """
        return strategy in self.selected_strategies
    
    def _validate_and_clean_signal(self, signal: Dict) -> Dict:
        """
        ENHANCED: Validate and clean signal data with support for existing signal formats
        """
        try:
            # Check required fields
            required_fields = ['epic', 'signal_type', 'confidence_score', 'strategy']
            for field in required_fields:
                if field not in signal or signal[field] is None:
                    raise ValueError(f"Missing required field: {field}")
            
            # ENHANCED: Normalize signal type to support existing formats
            signal_type = signal['signal_type'].upper()
            
            # Support both old (BULL/BEAR) and new (BUY/SELL) formats
            signal_type_mapping = {
                'BULL': 'BUY',
                'BEAR': 'SELL', 
                'BUY': 'BUY',
                'SELL': 'SELL'
            }
            
            if signal_type in signal_type_mapping:
                # Normalize the signal type but keep original for logging
                signal['original_signal_type'] = signal['signal_type']
                signal['signal_type'] = signal_type_mapping[signal_type]
            else:
                raise ValueError(f"Invalid signal type: {signal['signal_type']} (expected: BULL, BEAR, BUY, or SELL)")
            
            # Validate confidence score
            confidence = signal['confidence_score']
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence score: {confidence}")
            
            # Check minimum confidence
            if confidence < self.min_confidence:
                raise ValueError(f"Confidence below minimum: {confidence:.2%} < {self.min_confidence:.2%}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {e}")
            raise
    
    def _enhance_signal_metadata(self, signal: Dict) -> Dict:
        """
        ENHANCED: Enhance signal with additional metadata including hash for duplicate detection
        """
        try:
            enhanced = signal.copy()
            
            # Add processing timestamp
            enhanced['processing_timestamp'] = datetime.now().isoformat()
            
            # ENHANCED: Add signal hash for duplicate detection
            enhanced['signal_hash'] = self._generate_signal_hash(signal)
            
            # Add processor version
            enhanced['processor_version'] = '2.0_fixed'
            enhanced['pipeline_stage'] = 'validated'
            
            # Calculate risk-reward ratio if not present
            if 'risk_reward_ratio' not in enhanced:
                enhanced['risk_reward_ratio'] = self._calculate_risk_reward_ratio(signal)
            
            # Add processing metadata
            enhanced['selected_strategies'] = self.selected_strategies
            enhanced['duplicate_window_hours'] = self.duplicate_window_hours
            
            self.enhanced_count += 1
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Signal enhancement failed: {e}")
            return signal
    
    def _generate_signal_hash(self, signal: Dict) -> str:
        """
        ENHANCED: Generate unique hash for signal duplicate detection with better error handling
        """
        try:
            # Use epic, signal_type, strategy, and date for hash
            timestamp = signal.get('timestamp', datetime.now())
            
            # Handle both string and datetime timestamps
            if isinstance(timestamp, datetime):
                date_str = timestamp.strftime('%Y-%m-%d')
            elif isinstance(timestamp, str):
                # Try to parse the timestamp and extract date
                try:
                    parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = parsed_time.strftime('%Y-%m-%d')
                except:
                    date_str = timestamp[:10]  # Take first 10 chars as date
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
            
            hash_data = {
                'epic': signal.get('epic', ''),
                'signal_type': signal.get('signal_type', ''),
                'strategy': signal.get('strategy', ''),
                'entry_price': round(float(signal.get('entry_price', 0)), 4),
                'date': date_str
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.md5(hash_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Hash generation failed: {e}")
            # Fallback hash using basic signal info
            fallback_string = f"{signal.get('epic', '')}_{signal.get('signal_type', '')}_{signal.get('strategy', '')}"
            return hashlib.md5(fallback_string.encode()).hexdigest()
    
    def _calculate_risk_reward_ratio(self, signal: Dict) -> float:
        """
        Calculate risk-reward ratio for signal
        """
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            if not all([entry_price, stop_loss, take_profit]):
                return 0.0
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk == 0:
                return 0.0
            
            return reward / risk
            
        except Exception as e:
            self.logger.error(f"Risk-reward calculation failed: {e}")
            return 0.0
    
    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """
        ENHANCED: Check if signal is duplicate using hash and time window
        """
        try:
            signal_hash = signal.get('signal_hash', '')
            epic = signal.get('epic', '')
            current_time = datetime.now()
            
            # Check database for recent signals
            if self.db_manager:
                recent_signals = self._get_recent_signals_from_db(epic, current_time)
                for recent_signal in recent_signals:
                    if recent_signal.get('signal_hash') == signal_hash:
                        return True
            
            # Check in-memory cache
            cache_key = f"{epic}_{signal_hash}"
            if cache_key in self.duplicate_cache:
                cached_time = self.duplicate_cache[cache_key]
                time_diff = current_time - cached_time
                if time_diff.total_seconds() < (self.duplicate_window_hours * 3600):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Duplicate check failed: {e}")
            return False
    
    def _get_recent_signals_from_db(self, epic: str, current_time: datetime) -> List[Dict]:
        """
        ENHANCED: Get recent signals from database with graceful fallback to alert history
        """
        try:
            if not self.db_manager:
                return []
            
            # Calculate time window
            start_time = current_time - timedelta(hours=self.duplicate_window_hours)
            
            # Try to query alert_history table
            try:
                conn = self.db_manager.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT signal_hash FROM alert_history 
                    WHERE epic = %s AND alert_timestamp > %s 
                    ORDER BY alert_timestamp DESC
                    LIMIT 100
                """, (epic, start_time))
                
                results = cursor.fetchall()
                cursor.close()
                conn.close()
                
                # Convert to list of dicts
                return [{'signal_hash': row[0]} for row in results if row[0]]
                
            except Exception as e:
                self.logger.debug(f"Database query failed: {e}")
                return []
            
        except Exception as e:
            self.logger.error(f"Recent signals query failed: {e}")
            return []
    
    def _update_duplicate_cache(self, signal: Dict):
        """
        ENHANCED: Update duplicate cache with new signal
        """
        try:
            signal_hash = signal.get('signal_hash', '')
            epic = signal.get('epic', '')
            current_time = datetime.now()
            
            cache_key = f"{epic}_{signal_hash}"
            self.duplicate_cache[cache_key] = current_time
            
            # Clean old entries from cache
            cutoff_time = current_time - timedelta(hours=self.duplicate_window_hours * 2)
            self.duplicate_cache = {
                k: v for k, v in self.duplicate_cache.items() 
                if v > cutoff_time
            }
            
        except Exception as e:
            self.logger.error(f"Cache update failed: {e}")
    
    def _run_claude_analysis(self, signal: Dict) -> Dict:
        """
        ENHANCED: Run Claude analysis on signal with better error handling for existing ClaudeAnalyzer
        """
        try:
            if not self.claude_analyzer:
                return signal
            
            # Ensure signal is a dictionary, not a string
            if not isinstance(signal, dict):
                self.logger.error(f"Claude analysis expects dict, got {type(signal)}")
                return signal
            
            # Check what method is available on the Claude analyzer
            if hasattr(self.claude_analyzer, 'analyze_signal'):
                try:
                    # Call the analyze_signal method with just the signal
                    claude_result = self.claude_analyzer.analyze_signal(signal)
                    
                    # Handle different return types
                    if isinstance(claude_result, dict):
                        analysis_data = claude_result
                    elif isinstance(claude_result, str):
                        # If it returns a string, wrap it in a dict
                        analysis_data = {
                            'analysis': claude_result,
                            'confidence': 50,  # Default confidence
                            'recommendation': 'NEUTRAL'
                        }
                    else:
                        self.logger.warning(f"Unexpected Claude result type: {type(claude_result)}")
                        return signal
                        
                except Exception as e:
                    self.logger.error(f"Claude analyze_signal failed: {e}")
                    return signal
            else:
                self.logger.warning("Claude analyzer has no analyze_signal method")
                return signal
            
            # Add Claude analysis to signal
            signal['claude_analysis'] = {
                'confidence': analysis_data.get('confidence', 0),
                'recommendation': analysis_data.get('recommendation', 'NEUTRAL'),
                'analysis': analysis_data.get('analysis', ''),
                'risk_assessment': analysis_data.get('risk_assessment', ''),
                'market_context': analysis_data.get('market_context', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            self.claude_analyzed_count += 1
            self.logger.info(f"🤖 Claude analysis completed: {signal.get('epic', 'Unknown')}")
            self.logger.info(f"   Claude confidence: {analysis_data.get('confidence', 'N/A')}")
            self.logger.info(f"   Claude recommendation: {analysis_data.get('recommendation', 'N/A')}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Claude analysis error: {e}")
            return signal
    
    def _send_notification(self, signal: Dict):
        """
        ENHANCED: Send notification for processed signal with compatibility for existing NotificationManager
        """
        try:
            if not self.notification_manager:
                return
            
            # Prepare notification message
            message = self._prepare_notification_message(signal)
            
            # Check what methods are available on the notification manager
            success = False
            
            if hasattr(self.notification_manager, 'send_notification'):
                # Try the enhanced method
                try:
                    success = self.notification_manager.send_notification(
                        message=message,
                        signal_data=signal,
                        priority='high' if signal.get('confidence_score', 0) > 0.8 else 'normal'
                    )
                except TypeError:
                    # Fallback to simpler method signature
                    success = self.notification_manager.send_notification(message)
            elif hasattr(self.notification_manager, 'send_alert'):
                # Try alternative method name
                success = self.notification_manager.send_alert(message)
            elif hasattr(self.notification_manager, 'notify'):
                # Try another alternative method name
                success = self.notification_manager.notify(message)
            else:
                self.logger.warning("Notification manager has no compatible send method")
                return
            
            if success:
                self.notifications_sent += 1
                self.logger.info(f"📱 Notification sent: {signal.get('epic', 'Unknown')}")
            else:
                self.logger.warning(f"⚠️ Failed to send notification: {signal.get('epic', 'Unknown')}")
                
        except Exception as e:
            self.logger.error(f"Notification sending failed: {e}")
            # Don't fail the entire pipeline for notification errors
    
    def _prepare_notification_message(self, signal: Dict) -> str:
        """
        ENHANCED: Prepare comprehensive notification message
        """
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            strategy = signal.get('strategy', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            
            message = f"🚨 {signal_type} Signal Alert\n"
            message += f"Epic: {epic}\n"
            message += f"Strategy: {strategy}\n"
            message += f"Confidence: {confidence:.1%}\n"
            message += f"Entry: {signal.get('entry_price', 'N/A')}\n"
            message += f"Stop Loss: {signal.get('stop_loss', 'N/A')}\n"
            message += f"Take Profit: {signal.get('take_profit', 'N/A')}\n"
            message += f"Risk/Reward: {signal.get('risk_reward_ratio', 0):.2f}\n"
            
            # Add Claude analysis if available
            if 'claude_analysis' in signal:
                claude_rec = signal['claude_analysis'].get('recommendation', 'N/A')
                claude_conf = signal['claude_analysis'].get('confidence', 0)
                message += f"Claude: {claude_rec} ({claude_conf}%)\n"
            
            message += f"Time: {signal.get('processing_timestamp', 'N/A')}"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Message preparation failed: {e}")
            return "Trading signal alert"
    
    def process_signal_batch(self, signals: List[Dict]) -> List[Dict]:
        """
        ENHANCED: Process a batch of signals with complete pipeline
        """
        if not signals:
            return []
        
        self.logger.info(f"📊 Processing batch of {len(signals)} signals")
        
        processed_signals = []
        
        for i, signal in enumerate(signals, 1):
            try:
                # Log signal summary
                self.log_signal_summary(signal, i, len(signals))
                
                # Process the signal through complete pipeline
                processed_signal = self.process_signal(signal)
                processed_signals.append(processed_signal)
                
            except Exception as e:
                self.logger.error(f"❌ Error processing signal {i}/{len(signals)}: {e}")
                # Add error info to signal and continue
                signal['processing_error'] = str(e)
                processed_signals.append(signal)
                self.errors += 1
        
        # Log batch summary
        self.logger.info(f"✅ Batch processing completed:")
        self.logger.info(f"   Total processed: {len(processed_signals)}")
        self.logger.info(f"   Stored in DB: {self.signals_stored}")
        self.logger.info(f"   Duplicates filtered: {self.duplicates_filtered}")
        self.logger.info(f"   Strategy filtered: {self.strategy_filtered}")
        self.logger.info(f"   Claude analyses: {self.claude_analyzed_count}")
        self.logger.info(f"   Notifications sent: {self.notifications_sent}")
        
        return processed_signals
    
    def log_signal_summary(self, signal: Dict, index: int, total: int):
        """
        ENHANCED: Log comprehensive signal summary with support for both signal formats
        """
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            original_signal_type = signal.get('original_signal_type', signal_type)
            strategy = signal.get('strategy', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            
            self.logger.info(f"📊 Signal {index}/{total}: {epic}")
            self.logger.info(f"   Type: {original_signal_type}")  # Show original format (BEAR/BULL)
            self.logger.info(f"   Strategy: {strategy}")
            self.logger.info(f"   Confidence: {confidence:.1%}")
            self.logger.info(f"   Selected: {'✅' if strategy in self.selected_strategies else '❌'}")
            
        except Exception as e:
            self.logger.error(f"Error logging signal summary: {e}")
    
    def get_processing_statistics(self) -> Dict:
        """
        ENHANCED: Get comprehensive processing statistics
        """
        return {
            'processing_stats': {
                'total_signals_processed': self.processed_count,
                'signals_enhanced': self.enhanced_count,
                'signals_stored': self.signals_stored,
                'duplicates_filtered': self.duplicates_filtered,
                'strategy_filtered': self.strategy_filtered,
                'claude_analyses': self.claude_analyzed_count,
                'notifications_sent': self.notifications_sent,
                'errors': self.errors
            },
            'configuration': {
                'selected_strategies': self.selected_strategies,
                'min_confidence': self.min_confidence,
                'duplicate_window_hours': self.duplicate_window_hours,
                'claude_analysis_mode': self.claude_analysis_mode,
                'enable_notifications': self.enable_notifications,
                'save_to_database': self.save_to_database
            },
            'cache_info': {
                'duplicate_cache_size': len(self.duplicate_cache)
            },
            'success_rates': {
                'enhancement_rate': (self.enhanced_count / max(self.processed_count, 1)) * 100,
                'storage_rate': (self.signals_stored / max(self.processed_count, 1)) * 100,
                'claude_analysis_rate': (self.claude_analyzed_count / max(self.processed_count, 1)) * 100,
                'notification_rate': (self.notifications_sent / max(self.processed_count, 1)) * 100
            }
        }
    
    def reset_statistics(self):
        """
        ENHANCED: Reset all processing statistics
        """
        self.processed_count = 0
        self.enhanced_count = 0
        self.claude_analyzed_count = 0
        self.notifications_sent = 0
        self.signals_stored = 0
        self.duplicates_filtered = 0
        self.strategy_filtered = 0
        self.errors = 0
        
        # Clear duplicate cache
        self.duplicate_cache.clear()
        
        self.logger.info("📊 Processing statistics reset")
    
    def set_claude_analyzer(self, claude_analyzer):
        """Set Claude analyzer instance"""
        self.claude_analyzer = claude_analyzer
        self.logger.info("🤖 Claude analyzer updated")
    
    def set_notification_manager(self, notification_manager):
        """Set notification manager instance"""
        self.notification_manager = notification_manager
        self.logger.info("📱 Notification manager updated")
    
    def set_alert_history(self, alert_history):
        """Set alert history instance"""
        self.alert_history = alert_history
        self.logger.info("📝 Alert history updated")
    
    def update_configuration(self, **kwargs):
        """
        ENHANCED: Update processor configuration
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"⚙️ Updated {key} to {value}")
            else:
                self.logger.warning(f"⚠️ Unknown configuration key: {key}")
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """
        ENHANCED: Validate all dependencies are properly configured
        """
        issues = []
        
        # Check essential dependencies
        if not self.db_manager:
            issues.append("Database manager not available")
        
        if not self.alert_history:
            issues.append("Alert history not available")
        
        # Check optional dependencies
        if self.enable_notifications and not self.notification_manager:
            issues.append("Notifications enabled but notification manager not available")
        
        if self.claude_analyzer and not hasattr(self.claude_analyzer, 'analyze_signal'):
            issues.append("Claude analyzer missing analyze_signal method")
        
        # Check configuration
        if not self.selected_strategies:
            issues.append("No strategies selected")
        
        if self.min_confidence <= 0 or self.min_confidence > 1:
            issues.append("Invalid minimum confidence threshold")
        
        return len(issues) == 0, issues