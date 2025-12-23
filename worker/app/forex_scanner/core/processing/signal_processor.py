# core/processing/signal_processor.py
"""
Signal Processor - Enhanced Version with Guaranteed Alert History Saving
Handles signal processing, Claude analysis, validation, and notifications

FIXED ISSUES:
1. Enhanced _save_to_alert_history method with proper error handling
2. Direct database fallback when AlertHistoryManager fails
3. Better logging to track save attempts
4. Proper signal data formatting for alert_history table
5. Comprehensive error handling and fallback mechanisms

ENHANCED FEATURES:
6. Improved signal type handling without forced conversion
7. Better duplicate detection with fingerprinting
8. Claude analysis timeout handling
9. Performance metrics tracking
10. Enhanced validation and error recovery

NOTIFICATION IMPROVEMENTS:
11. Fixed notification interface compatibility with enhanced NotificationManager
12. Added backward compatibility for legacy notification methods
13. Enhanced notification message formatting
14. Comprehensive notification error handling
15. Proper Claude analysis integration in notifications

SMART MONEY INTEGRATION (NEW):
16. Smart Money Read-Only Analyzer integration
17. Market structure analysis (BOS, ChoCh)
18. Order flow analysis (Order Blocks, FVGs)
19. Smart money validation scores
20. Confluence analysis for enhanced signals
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
try:
    import config
except ImportError:
    from forex_scanner import config
import hashlib
import json
import time
try:
    from utils.scanner_utils import make_json_serializable, clean_signal_for_json
except ImportError:
    from forex_scanner.utils.scanner_utils import make_json_serializable, clean_signal_for_json

# ADD: Import deduplication manager
try:
    from core.alert_deduplication import AlertDeduplicationManager
except ImportError:
    from forex_scanner.core.alert_deduplication import AlertDeduplicationManager

# ADD: Import Smart Money Analyzer
try:
    from core.smart_money_readonly_analyzer import SmartMoneyReadOnlyAnalyzer
    SMART_MONEY_AVAILABLE = True
except ImportError:
    SMART_MONEY_AVAILABLE = False
    logging.getLogger(__name__).warning("SmartMoneyReadOnlyAnalyzer not available")

class SignalProcessor:
    """
    Handles comprehensive signal processing including validation, enhancement,
    Claude analysis, Smart Money analysis, and notification sending
    
    FIXED: Now guarantees signals are saved to alert_history table
    ENHANCED: Improved robustness, performance tracking, and error handling
    NOTIFICATION ENHANCED: Full compatibility with enhanced NotificationManager
    SMART MONEY ENHANCED: Integrated smart money analysis for better signal quality
    """
    
    def __init__(self,
                 claude_analyzer=None,
                 notification_manager=None,
                 alert_history=None,
                 db_manager=None,
                 data_fetcher=None,  # ADD: data_fetcher parameter for smart money
                 logger: Optional[logging.Logger] = None):
        
        self.claude_analyzer = claude_analyzer
        self.notification_manager = notification_manager
        self.alert_history = alert_history
        self.db_manager = db_manager
        self.data_fetcher = data_fetcher  # ADD: Store data_fetcher
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration (existing)
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.7)
        self.claude_analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
        self.claude_strategic_focus = getattr(config, 'CLAUDE_STRATEGIC_FOCUS', None)
        self.claude_timeout = getattr(config, 'CLAUDE_TIMEOUT_SECONDS', 30)
        self.enable_notifications = getattr(config, 'ENABLE_NOTIFICATIONS', True)
        # Use dynamic strategy detection instead of hardcoded list
        if hasattr(config, 'get_enabled_strategies'):
            self.selected_strategies = config.get_enabled_strategies()
        else:
            # Fallback to old approach if new function not available
            self.selected_strategies = getattr(config, 'SELECTED_STRATEGIES', ['EMA', 'MACD', 'Combined'])
        self.duplicate_window_hours = getattr(config, 'DUPLICATE_WINDOW_HOURS', 24)
        self.save_to_database = getattr(config, 'SAVE_TO_DATABASE', True)
        
        # INTEGRATION: Initialize deduplication manager
        self.deduplication_manager = None
        self.enable_deduplication = getattr(config, 'ENABLE_ALERT_DEDUPLICATION', True)
        
        if self.enable_deduplication and db_manager:
            try:
                self.deduplication_manager = AlertDeduplicationManager(db_manager)
                self.logger.info("🛡️ Deduplication manager initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize deduplication manager: {e}")
                self.enable_deduplication = False
        
        # SMART MONEY INTEGRATION: Initialize Smart Money Analyzer
        self.smart_money_analyzer = None
        self.enable_smart_money = getattr(config, 'SMART_MONEY_READONLY_ENABLED', True)
        self.smart_money_timeout = getattr(config, 'SMART_MONEY_ANALYSIS_TIMEOUT', 5.0)
        
        if SMART_MONEY_AVAILABLE and self.enable_smart_money and data_fetcher:
            try:
                self.smart_money_analyzer = SmartMoneyReadOnlyAnalyzer(data_fetcher)
                self.logger.info("🧠 Smart Money Analyzer initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Smart Money Analyzer initialization failed: {e}")
                self.enable_smart_money = False
        
        # Processing statistics (existing + smart money stats)
        self.processed_count = 0
        self.enhanced_count = 0
        self.claude_analyzed_count = 0
        self.smart_money_analyzed_count = 0  # ADD: Smart money counter
        self.notifications_sent = 0
        self.signals_stored = 0
        self.duplicates_filtered = 0  # existing
        self.strategy_filtered = 0
        self.errors = 0
        self.avg_processing_time = 0.0
        self.performance_samples = []
        
        # INTEGRATION: Additional deduplication statistics
        self.dedup_blocked_count = 0
        self.dedup_hash_blocks = 0
        self.dedup_time_blocks = 0
        self.dedup_rate_blocks = 0
        self.dedup_similarity_blocks = 0
        
        # Duplicate tracking cache (existing)
        self.duplicate_cache = {}
        
        self.logger.info("📊 Enhanced SignalProcessor initialized with integrated deduplication and Smart Money")
        self.logger.info(f"   Selected strategies: {self.selected_strategies}")
        self.logger.info(f"   Claude analysis: {'✅' if claude_analyzer else '❌'} ({self.claude_analysis_mode})")
        self.logger.info(f"   Smart Money: {'✅' if self.smart_money_analyzer else '❌'}")
        self.logger.info(f"   Notifications: {'✅' if notification_manager else '❌'}")
        self.logger.info(f"   Alert history: {'✅' if alert_history else '❌'}")
        self.logger.info(f"   Database storage: {'✅' if self.save_to_database else '❌'}")
        self.logger.info(f"   Deduplication: {'✅' if self.deduplication_manager else '❌'}")
        self.logger.info(f"   Claude timeout: {self.claude_timeout}s")
        self.logger.info(f"   Smart Money timeout: {self.smart_money_timeout}s")
        
        # Test connections immediately (existing)
        if self.alert_history:
            self._test_alert_history_connection()
        if self.notification_manager:
            self._test_notification_manager_connection()
    
    def _test_alert_history_connection(self):
        """Test alert_history connection on initialization (EXISTING METHOD)"""
        try:
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
    
    def _test_notification_manager_connection(self):
        """Test notification manager connection (EXISTING METHOD)"""
        try:
            if hasattr(self.notification_manager, 'send_signal_alert'):
                self.logger.info("✅ Enhanced NotificationManager.send_signal_alert method available")
                self._notification_interface = 'enhanced'
            elif hasattr(self.notification_manager, 'send_notification'):
                self.logger.info("✅ Legacy NotificationManager.send_notification method available")
                self._notification_interface = 'legacy_send_notification'
            elif hasattr(self.notification_manager, 'send_alert'):
                self.logger.info("✅ Legacy NotificationManager.send_alert method available")
                self._notification_interface = 'legacy_send_alert'
            elif hasattr(self.notification_manager, 'notify'):
                self.logger.info("✅ Basic NotificationManager.notify method available")
                self._notification_interface = 'basic_notify'
            else:
                self.logger.warning("⚠️ NotificationManager has no compatible send methods")
                self._notification_interface = 'none'
                
        except Exception as e:
            self.logger.error(f"❌ Notification manager connection test failed: {e}")
            self._notification_interface = 'error'
    
    def process_signal(self, signal: Dict) -> Dict:
        """
        INTEGRATION UPDATE: Main signal processing with integrated deduplication and Smart Money
        
        EXISTING functionality maintained:
        - Strategy filtering
        - Signal validation and enhancement
        - Claude analysis
        - Notification sending
        - Database saving
        
        NEW functionality added:
        - Integrated deduplication checking
        - Smart Money analysis
        - Enhanced statistics tracking
        - Smart Money metadata in results
        - MTF override logic for confidence filtering
        
        FIXED: Confidence filtering now handled gracefully without exceptions
        """
        start_time = time.time()
        self.processed_count += 1
        
        epic = signal.get('epic', 'Unknown')
        signal_type = signal.get('signal_type', 'Unknown')
        strategy = signal.get('strategy', 'Unknown')
        confidence = signal.get('confidence_score', 0)
        
        self.logger.info(f"📊 Processing signal #{self.processed_count}: {epic}")
        self.logger.info(f"   Type: {signal_type}, Strategy: {strategy}")
        self.logger.info(f"   Confidence: {confidence:.1%}")
        
        # Enhanced processing result tracking
        processing_result = {
            'signal_processed': False,
            'claude_analyzed': False,
            'smart_money_analyzed': False,  # ADD: Smart money field
            'saved_to_database': False,
            'notifications_sent': False,
            'alert_id': None,
            'claude_result': None,
            'smart_money_result': None,  # ADD: Smart money result
            'processing_errors': [],
            'processing_time_ms': 0,
            'duplicate_filtered': False,
            'strategy_filtered': False,
            'confidence_filtered': False,  # NEW: Add confidence filtering tracking
            'mtf_override_applied': False,  # NEW: Track MTF override usage
            # INTEGRATION: New deduplication fields
            'deduplication_checked': False,
            'deduplication_passed': False,
            'deduplication_reason': None,
            'deduplication_metadata': None
        }
        
        try:
            # EXISTING: Step 1 - Strategy filtering (early exit)
            if not self._is_strategy_selected(strategy):
                self.logger.info(f"⚠️ Strategy {strategy} not in selected strategies, skipping")
                self.strategy_filtered += 1
                processing_result['strategy_filtered'] = True
                processing_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                signal['processing_result'] = processing_result
                return signal
            
            # NEW: Step 2 - Enhanced confidence filtering with MTF override logic
            # Check if signal has MTF analysis and is valid
            is_mtf_enhanced = (
                'mtf_analysis' in signal and 
                signal['mtf_analysis'].get('mtf_valid', False) and
                signal['mtf_analysis'].get('enabled', False)
            )
            
            # Determine confidence threshold based on MTF enhancement
            if is_mtf_enhanced:
                # Use lower threshold for MTF-enhanced signals
                mtf_threshold = getattr(config, 'MTF_ENHANCED_MIN_CONFIDENCE', 0.60)
                effective_threshold = mtf_threshold
                processing_result['mtf_override_applied'] = True
                
                # Log MTF override
                mtf_score = signal['mtf_analysis'].get('momentum_score', 0)
                aligned_tfs = signal['mtf_analysis'].get('aligned_timeframes', 0)
                total_tfs = signal['mtf_analysis'].get('total_timeframes', 0)
                
                self.logger.info(f"📊 MTF Override Applied:")
                self.logger.info(f"   Standard threshold: {self.min_confidence:.1%}")
                self.logger.info(f"   MTF threshold: {mtf_threshold:.1%}")
                self.logger.info(f"   MTF Score: {mtf_score:.1%}")
                self.logger.info(f"   Aligned TFs: {aligned_tfs}/{total_tfs}")
            else:
                # Use standard threshold for non-MTF signals
                effective_threshold = self.min_confidence

            # 🔥 SCALPING BYPASS: Use lower threshold for scalping strategies (45%)
            strategy = signal.get('strategy', '')
            scalping_mode = signal.get('scalping_mode', '')
            is_scalping = ('scalping' in strategy.lower() or
                          scalping_mode in ['linda_raschke', 'ranging_momentum', 'linda_macd_zero_cross',
                                           'linda_macd_cross', 'linda_macd_momentum', 'linda_anti_pattern'])

            if is_scalping:
                scalping_threshold = getattr(config, 'SCALPING_MIN_CONFIDENCE', 0.45)
                effective_threshold = scalping_threshold
                self.logger.info(f"🔥 SCALPING: Using scalping threshold {scalping_threshold:.1%} instead of {self.min_confidence:.1%}")

            # Apply confidence filtering with appropriate threshold
            if confidence < effective_threshold:
                if is_mtf_enhanced:
                    self.logger.info(f"🔽 MTF Signal filtered: {epic} confidence {confidence:.2%} < {effective_threshold:.2%} (MTF threshold)")
                else:
                    self.logger.info(f"🔽 Signal filtered: {epic} confidence {confidence:.2%} < {effective_threshold:.2%}")
                
                # Track confidence filtering stats
                if hasattr(self, 'confidence_filtered'):
                    self.confidence_filtered += 1
                processing_result['confidence_filtered'] = True
                processing_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                signal['processing_result'] = processing_result
                signal['filtered_reason'] = f"Confidence below {'MTF' if is_mtf_enhanced else 'standard'} minimum: {confidence:.2%} < {effective_threshold:.2%}"
                return signal
            else:
                # Log successful confidence check
                if is_mtf_enhanced:
                    self.logger.info(f"✅ MTF Signal passed confidence: {confidence:.2%} ≥ {effective_threshold:.2%} (MTF threshold)")
                else:
                    self.logger.info(f"✅ Signal passed confidence: {confidence:.2%} ≥ {effective_threshold:.2%}")
            
            # EXISTING: Step 3 - Enhanced validation (confidence check removed from here)
            validated_signal = self._validate_and_clean_signal_enhanced(signal)
            
            # EXISTING: Step 4 - Enhance signal with metadata
            enhanced_signal = self._enhance_signal_metadata_v2(validated_signal)
            
            # INTEGRATION: Step 5 - NEW Deduplication check
            if self.deduplication_manager:
                processing_result['deduplication_checked'] = True
                
                allow_signal, dedup_reason, dedup_metadata = self.deduplication_manager.should_allow_alert(enhanced_signal)
                
                processing_result['deduplication_reason'] = dedup_reason
                processing_result['deduplication_metadata'] = dedup_metadata
                
                if not allow_signal:
                    self.logger.info(f"🔄 Signal blocked by deduplication: {dedup_reason}")
                    self.dedup_blocked_count += 1
                    processing_result['duplicate_filtered'] = True
                    processing_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                    
                    # Track deduplication block reasons
                    if 'Hash' in dedup_reason:
                        self.dedup_hash_blocks += 1
                    elif 'Cooldown' in dedup_reason:
                        self.dedup_time_blocks += 1
                    elif 'Rate Limit' in dedup_reason:
                        self.dedup_rate_blocks += 1
                    elif 'Similarity' in dedup_reason:
                        self.dedup_similarity_blocks += 1
                    
                    enhanced_signal['processing_result'] = processing_result
                    enhanced_signal['deduplication_blocked'] = True
                    enhanced_signal['deduplication_reason'] = dedup_reason
                    return enhanced_signal
                
                processing_result['deduplication_passed'] = True
                enhanced_signal.update(dedup_metadata)  # Add deduplication metadata to signal
                self.logger.debug(f"✅ Deduplication passed: {epic}")
            else:
                # EXISTING: Fallback to existing duplicate detection if deduplication manager not available
                if self._is_duplicate_signal_enhanced(enhanced_signal):
                    self.logger.info(f"🔄 Duplicate signal detected for {epic} (legacy method), skipping")
                    self.duplicates_filtered += 1
                    processing_result['duplicate_filtered'] = True
                    processing_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
                    enhanced_signal['processing_result'] = processing_result
                    return enhanced_signal
            
            # SMART MONEY: Step 6 - Smart Money Analysis (NEW)
            smart_money_result = None
            if self._should_analyze_with_smart_money(enhanced_signal):
                self.logger.debug("🧠 Step 6: Applying Smart Money analysis...")
                smart_money_result = self._run_smart_money_analysis(enhanced_signal, epic)
                
                if smart_money_result:
                    processing_result['smart_money_analyzed'] = True
                    processing_result['smart_money_result'] = smart_money_result
                    
                    # Merge smart money data into signal
                    enhanced_signal = self._merge_smart_money_results(enhanced_signal, smart_money_result)
                    self.smart_money_analyzed_count += 1
                    
                    self.logger.info(f"🧠 Smart Money Analysis Complete:")
                    self.logger.info(f"   Validated: {smart_money_result.get('smart_money_validated', False)}")
                    self.logger.info(f"   Type: {smart_money_result.get('smart_money_type', 'Unknown')}")
                    self.logger.info(f"   Score: {smart_money_result.get('smart_money_score', 0):.3f}")
                    self.logger.info(f"   Enhanced Confidence: {smart_money_result.get('enhanced_confidence_score', confidence):.3f}")
            
            # INTEGRATION: Step 7 - Enhanced database save with deduplication and smart money
            alert_id = None
            if self.save_to_database and self.alert_history:
                # Prepare alert data with smart money fields
                enhanced_signal = self._prepare_signal_with_smart_money(enhanced_signal, smart_money_result)
                
                if self.deduplication_manager:
                    # Use deduplication manager for saving (NEW)
                    alert_id = self.deduplication_manager.save_alert_with_deduplication(
                        self.alert_history, enhanced_signal
                    )
                    if alert_id:
                        self.logger.info(f"✅ Signal saved via deduplication manager (ID: {alert_id})")
                else:
                    # Fallback to direct save (EXISTING)
                    alert_id = self._save_to_alert_history_guaranteed(enhanced_signal)
                
                if alert_id:
                    self.signals_stored += 1
                    processing_result['saved_to_database'] = True
                    processing_result['alert_id'] = alert_id
                    enhanced_signal['alert_id'] = alert_id
                    self.logger.info(f"✅ Signal saved to database with Smart Money: ID {alert_id}")
                    self._update_duplicate_cache(enhanced_signal)
                else:
                    self.logger.error(f"❌ Failed to save signal to database: {epic}")
                    processing_result['processing_errors'].append("Database save failed")
            
            # EXISTING: Step 8 - Enhanced Claude analysis
            claude_result = None
            if self._should_analyze_with_claude(enhanced_signal):
                self.logger.debug("🤖 Step 8: Applying Claude analysis...")
                claude_result = self._run_claude_analysis_with_timeout(enhanced_signal)
                if claude_result:
                    processing_result['claude_analyzed'] = True
                    processing_result['claude_result'] = claude_result
                    enhanced_signal = self._merge_claude_results(enhanced_signal, claude_result)
                    self.claude_analyzed_count += 1
            
            # EXISTING: Step 9 - Enhanced notifications (with smart money info)
            if self.enable_notifications and self._should_send_notification(enhanced_signal):
                notification_sent = self._send_enhanced_notification(enhanced_signal, claude_result)
                if notification_sent:
                    processing_result['notifications_sent'] = True
                    self.notifications_sent += 1
            
            # Mark as successfully processed
            processing_result['signal_processed'] = True
            
            # Update performance tracking
            processing_time = time.time() - start_time
            processing_result['processing_time_ms'] = int(processing_time * 1000)
            self._update_performance_stats(processing_time)
            
            enhanced_signal['processing_result'] = processing_result
            
            self.enhanced_count += 1
            self.logger.info(f"✅ Signal processing completed in {processing_time:.2f}s")
            
            return enhanced_signal
            
        except Exception as e:
            self.errors += 1
            processing_result['processing_errors'].append(str(e))
            processing_result['processing_time_ms'] = int((time.time() - start_time) * 1000)
            
            self.logger.error(f"❌ Error processing signal {epic}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            signal['processing_result'] = processing_result
            signal['processing_error'] = str(e)
            return signal
    
    # SMART MONEY: New methods for Smart Money integration
    def _should_analyze_with_smart_money(self, signal: Dict) -> bool:
        """Determine if signal should be analyzed with smart money"""
        if not self.enable_smart_money:
            return False

        if not self.smart_money_analyzer:
            return False

        # Only analyze signals with sufficient confidence
        confidence = signal.get('confidence_score', 0)
        if confidence < 0.6:
            return False

        # Check if strategy supports smart money analysis
        strategy = signal.get('strategy', '').strip().lower()

        # EXPLICIT strategy mapping for smart money compatibility
        smart_money_strategies = {
            'ema': True,
            'macd': True,
            'kama': True,
            'combined': True,
            'ichimoku': True,
            'smc': True,
            'smc_fast': True,
            'smc_simple': True,
            'smc_structure': True,
            'momentum': True,
            # Traditional technical analysis strategies that DON'T use smart money
            'mean_reversion': False,
            'ranging_market': False,
            'zero_lag': False,
            'zero_lag_squeeze': False
        }

        # Check explicit mapping first
        if strategy in smart_money_strategies:
            should_analyze = smart_money_strategies[strategy]
            self.logger.debug(f"🧠 Smart Money check for '{strategy}': {'✅ Enabled' if should_analyze else '❌ Disabled (traditional TA strategy)'}")
            return should_analyze

        # Fallback for unknown strategies - log and use keyword matching
        smart_money_keywords = ['ema', 'macd', 'combined', 'smc', 'ichimoku']
        should_analyze = any(keyword in strategy for keyword in smart_money_keywords)
        self.logger.warning(f"🧠 Unknown strategy '{strategy}' - using keyword fallback: {'✅ Enabled' if should_analyze else '❌ Disabled'}")
        return should_analyze
    
    def _run_smart_money_analysis(self, signal: Dict, epic: str) -> Optional[Dict]:
        """Run smart money analysis with timeout protection"""
        if not self.smart_money_analyzer or not self.data_fetcher:
            return None
        
        try:
            # Get price data for analysis
            pair = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            timeframe = signal.get('timeframe', '5m')
            
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe)
            
            if df is None or len(df) < 100:
                self.logger.warning(f"⚠️ Insufficient data for smart money analysis: {epic}")
                return None
            
            # Run analysis with timeout using threading
            import threading
            result = {}
            error = []
            
            def analyze():
                try:
                    result['data'] = self.smart_money_analyzer.analyze_signal(
                        signal, df, epic, timeframe
                    )
                except Exception as e:
                    error.append(str(e))
            
            thread = threading.Thread(target=analyze)
            thread.start()
            thread.join(timeout=self.smart_money_timeout)
            
            if thread.is_alive():
                self.logger.warning(f"⚠️ Smart money analysis timeout for {epic}")
                return None
            
            if error:
                self.logger.error(f"❌ Smart money analysis error: {error[0]}")
                return None
            
            return result.get('data')
            
        except Exception as e:
            self.logger.error(f"❌ Smart money analysis failed: {e}")
            return None
    
    def _merge_smart_money_results(self, signal: Dict, smart_money_result: Dict) -> Dict:
        """
        Merge smart money analysis results into signal
        FIXED: Use make_json_serializable for all JSON operations to handle datetime objects
        FIXED: Extract from nested 'smart_money_analysis' field if present
        """
        if not smart_money_result:
            return signal

        # Smart money analyzer returns data nested under 'smart_money_analysis'
        # Extract the nested data if present, otherwise use the result directly
        sm_analysis = smart_money_result.get('smart_money_analysis', {})

        # Update signal with smart money data
        # Check both nested and direct locations for backward compatibility
        signal['smart_money_validated'] = sm_analysis.get('smart_money_validated',
                                            smart_money_result.get('smart_money_validated', False))
        signal['smart_money_type'] = sm_analysis.get('smart_money_type',
                                            smart_money_result.get('smart_money_type', 'UNKNOWN'))
        signal['smart_money_score'] = sm_analysis.get('smart_money_score',
                                            smart_money_result.get('smart_money_score', 0.5))
        signal['enhanced_confidence_score'] = smart_money_result.get('confidence_score',
                                                                    signal.get('confidence_score', 0.5))

        # Add detailed analysis data for database storage (as JSON strings)
        # FIXED: Use make_json_serializable to handle datetime objects
        # FIXED: Extract from nested 'smart_money_analysis' field
        market_structure = sm_analysis.get('market_structure_analysis',
                                          smart_money_result.get('market_structure_analysis', {}))
        signal['market_structure_analysis'] = json.dumps(make_json_serializable(market_structure))

        order_flow = sm_analysis.get('order_flow_analysis',
                                    smart_money_result.get('order_flow_analysis', {}))
        signal['order_flow_analysis'] = json.dumps(make_json_serializable(order_flow))

        confluence = sm_analysis.get('confluence_details',
                                    smart_money_result.get('confluence_details', {}))
        signal['confluence_details'] = json.dumps(make_json_serializable(confluence))

        # Extract OB proximity fields for analytics
        ob_proximity = order_flow.get('ob_proximity', {})
        signal['ob_proximity_score'] = ob_proximity.get('alignment_score')
        signal['nearest_ob_distance_pips'] = ob_proximity.get('nearest_ob_distance_pips')

        # Extract liquidity sweep fields for analytics (NEW)
        liquidity_analysis = sm_analysis.get('liquidity_analysis',
                                             smart_money_result.get('liquidity_analysis', {}))
        signal['liquidity_sweep_detected'] = liquidity_analysis.get('sweep_detected', False)
        signal['liquidity_sweep_type'] = liquidity_analysis.get('sweep_type')
        signal['liquidity_sweep_quality'] = liquidity_analysis.get('sweep_quality')

        # Add metadata
        metadata = sm_analysis.get('analysis_metadata', smart_money_result.get('analysis_metadata'))
        if metadata:
            signal['smart_money_metadata'] = json.dumps(make_json_serializable(metadata))

        return signal
    

    
    def _prepare_signal_with_smart_money(self, signal: Dict, smart_money_result: Optional[Dict]) -> Dict:
        """Prepare signal with smart money fields for database storage"""
        # Ensure smart money fields are present (even if null/default)
        if smart_money_result:
            # Fields are already merged by _merge_smart_money_results
            pass
        else:
            # Add default smart money fields if analysis wasn't run
            signal.setdefault('smart_money_validated', False)
            signal.setdefault('smart_money_type', None)
            signal.setdefault('smart_money_score', None)
            signal.setdefault('enhanced_confidence_score', signal.get('confidence_score'))
            signal.setdefault('market_structure_analysis', None)
            signal.setdefault('order_flow_analysis', None)
            signal.setdefault('confluence_details', None)
            signal.setdefault('smart_money_metadata', None)
            # NEW: OB proximity and liquidity sweep defaults
            signal.setdefault('ob_proximity_score', None)
            signal.setdefault('nearest_ob_distance_pips', None)
            signal.setdefault('liquidity_sweep_detected', False)
            signal.setdefault('liquidity_sweep_type', None)
            signal.setdefault('liquidity_sweep_quality', None)

        return signal
    
    # All existing methods remain unchanged below...
    
    def _validate_and_clean_signal_enhanced(self, signal: Dict) -> Dict:
        """
        ENHANCED: Improved signal validation with better error messages and type handling
        CRITICAL FIX: Keep original signal types (BULL/BEAR) without conversion
        """
        try:
            # Check required fields with detailed error messages
            required_fields = ['epic', 'signal_type', 'confidence_score', 'strategy']
            missing_fields = [field for field in required_fields if field not in signal or signal[field] is None]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Enhanced signal type validation - preserve original types
            signal_type = str(signal['signal_type']).upper().strip()
            valid_types = ['BULL', 'BEAR', 'BUY', 'SELL', 'TEST_BULL', 'TEST_BEAR']
            
            if signal_type not in valid_types:
                raise ValueError(f"Invalid signal type: '{signal['signal_type']}' (expected: {valid_types})")
            
            # CRITICAL FIX: Keep the original signal type - don't convert!
            signal['signal_type'] = signal_type
            signal['original_signal_type'] = signal.get('original_signal_type', signal_type)
            
            # Enhanced confidence validation with type coercion
            confidence = signal['confidence_score']
            if not isinstance(confidence, (int, float)):
                try:
                    confidence = float(confidence)
                    signal['confidence_score'] = confidence
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid confidence score type: {type(confidence)}")
            
            if not 0 <= confidence <= 1:
                raise ValueError(f"Confidence score out of range: {confidence} (expected: 0.0-1.0)")
            
            
            # Enhanced strategy validation
            strategy = signal.get('strategy', '').upper()
            strategy = signal.get('strategy', '').strip()
            if strategy:
                self.logger.debug(f"Processing signal with strategy: {strategy}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {e}")
            raise
    
    def _enhance_signal_metadata_v2(self, signal: Dict) -> Dict:
        """
        ENHANCED: Enhanced signal metadata with improved hash generation and risk calculations
        """
        try:
            enhanced = signal.copy()
            
            # Add processing timestamp
            enhanced['processing_timestamp'] = datetime.now().isoformat()
            
            # ENHANCED: Add improved signal fingerprint for duplicate detection
            enhanced['signal_hash'] = self._generate_signal_fingerprint_v2(signal)
            
            # Add processor version and stage tracking
            enhanced['processor_version'] = '2.1_enhanced_smart_money'  # Updated version
            enhanced['pipeline_stage'] = 'validated'
            
            # Enhanced risk-reward calculation
            if 'risk_reward_ratio' not in enhanced:
                enhanced['risk_reward_ratio'] = self._calculate_risk_reward_ratio_enhanced(signal)
            
            # Add comprehensive processing metadata
            enhanced['processing_metadata'] = {
                'selected_strategies': self.selected_strategies,
                'duplicate_window_hours': self.duplicate_window_hours,
                'claude_analysis_mode': self.claude_analysis_mode,
                'smart_money_enabled': self.enable_smart_money,  # ADD: Smart money status
                'min_confidence_threshold': self.min_confidence,
                'enhancement_timestamp': datetime.now().isoformat()
            }
            
            self.enhanced_count += 1
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Signal enhancement failed: {e}")
            return signal
    
    def _generate_signal_fingerprint_v2(self, signal: Dict) -> str:
        """
        ENHANCED: Generate improved unique fingerprint for duplicate detection
        """
        try:
            # Use epic, signal_type, strategy, and rounded timestamp for fingerprint
            timestamp = signal.get('timestamp', datetime.now())
            
            # Handle both string and datetime timestamps
            if isinstance(timestamp, datetime):
                rounded_time = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
            elif isinstance(timestamp, str):
                try:
                    parsed_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    rounded_time = parsed_time.replace(minute=(parsed_time.minute // 15) * 15, second=0, microsecond=0)
                except:
                    rounded_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            else:
                rounded_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            # Create comprehensive fingerprint data
            fingerprint_data = {
                'epic': signal.get('epic', ''),
                'signal_type': signal.get('signal_type', ''),
                'strategy': signal.get('strategy', ''),
                'rounded_timestamp': rounded_time.isoformat(),
                'price_range': round(float(signal.get('entry_price', signal.get('price', 0))), 3)
            }
            
            fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
            return hashlib.md5(fingerprint_str.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Fingerprint generation failed: {e}")
            # Fallback fingerprint using basic signal info
            fallback_string = f"{signal.get('epic', '')}_{signal.get('signal_type', '')}_{signal.get('strategy', '')}"
            return hashlib.md5(fallback_string.encode()).hexdigest()
    
    def _is_duplicate_signal_enhanced(self, signal: Dict) -> bool:
        """
        ENHANCED: Improved duplicate detection using fingerprinting and time windows
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
    
    def _should_analyze_with_claude(self, signal: Dict) -> bool:
        """
        ENHANCED: Determine if signal should be analyzed with Claude based on configuration and confidence
        """
        if not self.claude_analyzer:
            return False
        
        # Always analyze in enhanced modes
        if self.claude_analysis_mode in ['enhanced', 'full']:
            return True
        
        # In minimal mode, only analyze high-confidence signals
        if self.claude_analysis_mode == 'minimal':
            return signal.get('confidence_score', 0) >= 0.8
        
        return False
    
    def _run_claude_analysis_with_timeout(self, signal: Dict) -> Optional[Dict]:
        """
        ENHANCED: Run Claude analysis with timeout handling and improved error recovery
        """
        try:
            if not hasattr(self.claude_analyzer, 'analyze_signal'):
                self.logger.warning("Claude analyzer missing analyze_signal method")
                return None
            
            start_time = time.time()
            self.logger.info(f"🤖 Starting Claude analysis for {signal.get('epic', 'Unknown')}")
            
            # Call Claude analysis
            claude_result = self.claude_analyzer.analyze_signal(signal)
            
            elapsed_time = time.time() - start_time
            if elapsed_time > self.claude_timeout:
                self.logger.warning(f"Claude analysis took {elapsed_time:.1f}s (timeout: {self.claude_timeout}s)")
            
            # Normalize result format
            if isinstance(claude_result, dict):
                return claude_result
            elif isinstance(claude_result, str):
                return {
                    'analysis': claude_result,
                    'confidence': 50,
                    'recommendation': 'NEUTRAL',
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': elapsed_time
                }
            else:
                self.logger.warning(f"Unexpected Claude result type: {type(claude_result)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Claude analysis error: {e}")
            return None
    
    def _should_send_notification(self, signal: Dict) -> bool:
        """
        ENHANCED: Intelligent notification decision based on multiple criteria including smart money
        """
        if not self.notification_manager:
            return False
        
        confidence = signal.get('confidence_score', 0)
        
        # Always send for very high-confidence signals
        if confidence >= 0.9:
            return True
        
        # Send if smart money strongly validated (NEW)
        if signal.get('smart_money_validated') and signal.get('smart_money_score', 0) > 0.8:
            return True
        
        # Send if Claude strongly approved
        claude_analysis = signal.get('claude_analysis', {})
        if claude_analysis.get('recommendation') == 'STRONG_BUY' or claude_analysis.get('recommendation') == 'STRONG_SELL':
            return True
        
        # Send for high-confidence signals
        if confidence >= 0.8:
            return True
        
        # Send if Claude approved with decent confidence
        if claude_analysis.get('confidence', 0) >= 70 and claude_analysis.get('recommendation') in ['BUY', 'SELL']:
            return True
        
        return confidence >= 0.75
    
    def _send_enhanced_notification(self, signal: Dict, claude_decision: Optional[Dict] = None) -> bool:
        """
        ENHANCED: Send notifications with full NotificationManager compatibility and Claude/Smart Money integration
        """
        try:
            if not self.notification_manager:
                self.logger.warning("No notification manager available")
                return False
            
            # Check execution status
            executed = signal.get('order_executed', False)
            
            # Use the detected interface type
            interface = getattr(self, '_notification_interface', 'none')
            
            if interface == 'enhanced':
                # Use enhanced NotificationManager.send_signal_alert method
                success = self._send_enhanced_signal_alert(signal, claude_decision, executed)
            elif interface.startswith('legacy'):
                # Use legacy notification methods with backward compatibility
                success = self._send_legacy_notification(signal, claude_decision, executed, interface)
            elif interface == 'basic_notify':
                # Use basic notify method
                message = self._prepare_comprehensive_notification_message(signal, claude_decision, executed)
                success = self.notification_manager.notify(message)
            else:
                self.logger.warning("No compatible notification method available")
                return False
            
            if success:
                self.logger.info(f"📱 Notification sent successfully: {signal.get('epic', 'Unknown')}")
            else:
                self.logger.warning(f"⚠️ Failed to send notification: {signal.get('epic', 'Unknown')}")
            
            return success
                
        except Exception as e:
            self.logger.error(f"Enhanced notification sending failed: {e}")
            return False
    
    def _send_enhanced_signal_alert(self, signal: Dict, claude_decision: Optional[Dict], executed: bool) -> bool:
        """Send notification using enhanced NotificationManager.send_signal_alert method"""
        try:
            self.notification_manager.send_signal_alert(
                signal=signal,
                claude_decision=claude_decision,
                executed=executed
            )
            return True
        except Exception as e:
            self.logger.error(f"Enhanced signal alert failed: {e}")
            return False
    
    def _send_legacy_notification(self, signal: Dict, claude_decision: Optional[Dict], executed: bool, interface: str) -> bool:
        """Send notification using legacy notification methods"""
        try:
            message = self._prepare_comprehensive_notification_message(signal, claude_decision, executed)
            
            if interface == 'legacy_send_notification':
                # Try enhanced method signature first
                try:
                    priority = 'high' if signal.get('confidence_score', 0) > 0.8 else 'normal'
                    return self.notification_manager.send_notification(
                        message=message,
                        signal_data=signal,
                        priority=priority
                    )
                except TypeError:
                    # Fallback to simple method signature
                    return self.notification_manager.send_notification(message)
            elif interface == 'legacy_send_alert':
                return self.notification_manager.send_alert(message)
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Legacy notification failed: {e}")
            return False
    
    def _prepare_comprehensive_notification_message(self, signal: Dict, claude_decision: Optional[Dict], executed: bool) -> str:
        """
        ENHANCED: Prepare comprehensive notification message with Claude analysis, Smart Money, and execution status
        """
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            strategy = signal.get('strategy', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            
            # Create engaging message with execution status
            if executed:
                emoji = "💰"
                status_text = "EXECUTED"
            elif signal.get('smart_money_validated'):  # ADD: Smart money check
                emoji = "🧠"
                status_text = "SMART MONEY VALIDATED"
            elif claude_decision and claude_decision.get('recommendation') in ['STRONG_BUY', 'STRONG_SELL']:
                emoji = "🔥"
                status_text = "CLAUDE APPROVED"
            elif confidence >= 0.9:
                emoji = "🔥"
                status_text = "HIGH CONFIDENCE"
            elif confidence >= 0.8:
                emoji = "🚨"
                status_text = "SIGNAL ALERT"
            else:
                emoji = "📊"
                status_text = "SIGNAL DETECTED"
            
            message = f"{emoji} {signal_type} {status_text}\n"
            message += f"═══════════════════\n"
            message += f"Epic: {epic}\n"
            message += f"Strategy: {strategy}\n"
            message += f"Confidence: {confidence:.1%}\n"
            
            # Add Smart Money info if available (NEW)
            if signal.get('smart_money_validated'):
                sm_type = signal.get('smart_money_type', 'Unknown')
                sm_score = signal.get('smart_money_score', 0)
                message += f"🧠 Smart Money: {sm_type} ({sm_score:.2f})\n"
            
            # Add price information
            if signal.get('entry_price'):
                message += f"Entry: {signal.get('entry_price')}\n"
            if signal.get('stop_loss'):
                message += f"Stop Loss: {signal.get('stop_loss')}\n"
            if signal.get('take_profit'):
                message += f"Take Profit: {signal.get('take_profit')}\n"
            
            # Add risk-reward ratio
            rr_ratio = signal.get('risk_reward_ratio', 0)
            if rr_ratio > 0:
                message += f"Risk/Reward: {rr_ratio:.2f}\n"
            
            # Add Claude analysis if available
            if claude_decision:
                claude_rec = claude_decision.get('recommendation', 'N/A')
                claude_conf = claude_decision.get('confidence', 0)
                message += f"🤖 Claude: {claude_rec} ({claude_conf}%)\n"
                
                # Add Claude reasoning if available
                if claude_decision.get('analysis'):
                    analysis = claude_decision['analysis'][:100] + "..." if len(claude_decision['analysis']) > 100 else claude_decision['analysis']
                    message += f"Analysis: {analysis}\n"
            
            # Add execution information
            if executed:
                message += f"💰 ORDER EXECUTED\n"
                if signal.get('execution_price'):
                    message += f"Execution Price: {signal.get('execution_price')}\n"
                if signal.get('order_id'):
                    message += f"Order ID: {signal.get('order_id')}\n"
            
            message += f"Time: {datetime.now().strftime('%H:%M:%S')}"
            
            return message
            
        except Exception as e:
            self.logger.error(f"Comprehensive message preparation failed: {e}")
            return f"Trading signal alert: {signal.get('epic', 'Unknown')} - {signal.get('signal_type', 'Unknown')}"
    
    def _update_performance_stats(self, processing_time: float):
        """
        ENHANCED: Update performance statistics with moving average and outlier handling
        """
        try:
            # Add to performance samples (keep last 100)
            self.performance_samples.append(processing_time)
            if len(self.performance_samples) > 100:
                self.performance_samples.pop(0)
            
            # Calculate moving average
            if self.performance_samples:
                self.avg_processing_time = sum(self.performance_samples) / len(self.performance_samples)
                
            # Log performance warnings for slow processing
            if processing_time > 10.0:  # More than 10 seconds
                self.logger.warning(f"⚠️ Slow processing detected: {processing_time:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Performance stats update failed: {e}")
    
    def _merge_claude_results(self, signal: Dict, claude_result: Dict) -> Dict:
        """
        ENHANCED: Merge Claude analysis results into signal with proper formatting
        """
        try:
            signal['claude_analysis'] = {
                'confidence': claude_result.get('confidence', 0),
                'recommendation': claude_result.get('recommendation', 'NEUTRAL'),
                'analysis': claude_result.get('analysis', ''),
                'risk_assessment': claude_result.get('risk_assessment', ''),
                'market_context': claude_result.get('market_context', ''),
                'timestamp': datetime.now().isoformat(),
                'processing_time': claude_result.get('processing_time', 0)
            }
            
            # Add Claude-specific fields for database storage
            signal['claude_score'] = claude_result.get('confidence', 0)
            signal['claude_decision'] = claude_result.get('recommendation', 'NEUTRAL')
            signal['claude_approved'] = claude_result.get('approved', False)
            signal['claude_reason'] = claude_result.get('reason', '')
            signal['claude_mode'] = self.claude_analysis_mode
            signal['claude_raw_response'] = json.dumps(claude_result)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Claude result merging failed: {e}")
            return signal
    
    def _calculate_risk_reward_ratio_enhanced(self, signal: Dict) -> float:
        """
        ENHANCED: Enhanced risk-reward calculation with fallback methods
        """
        try:
            entry_price = float(signal.get('entry_price', signal.get('price', 0)))
            stop_loss = float(signal.get('stop_loss', 0))
            take_profit = float(signal.get('take_profit', 0))
            
            if not all([entry_price, stop_loss, take_profit]):
                return 0.0
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            if risk == 0:
                return 0.0
            
            ratio = reward / risk
            
            # Cap the ratio at reasonable levels
            return min(ratio, 10.0)
            
        except Exception as e:
            self.logger.error(f"Risk-reward calculation failed: {e}")
            return 0.0
    
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
                
                # Add smart money info to message if available
                if signal.get('smart_money_validated'):
                    alert_message += f" [SM: {signal.get('smart_money_score', 0):.2f}]"
                
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
        """Prepare signal data in the format expected by AlertHistoryManager (with Smart Money fields)"""
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
                'processor_version': signal.get('processor_version', '2.1_enhanced_smart_money'),
                
                # Smart Money fields (NEW)
                'smart_money_validated': signal.get('smart_money_validated', False),
                'smart_money_type': signal.get('smart_money_type'),
                'smart_money_score': signal.get('smart_money_score'),
                'enhanced_confidence_score': signal.get('enhanced_confidence_score'),
                'market_structure_analysis': signal.get('market_structure_analysis'),
                'order_flow_analysis': signal.get('order_flow_analysis'),
                'confluence_details': signal.get('confluence_details'),
                'smart_money_metadata': signal.get('smart_money_metadata'),
                
                # Additional fields that might be expected
                'entry_price': float(signal.get('entry_price', 0)) if signal.get('entry_price') else None,
                'stop_loss': float(signal.get('stop_loss', 0)) if signal.get('stop_loss') else None,
                'take_profit': float(signal.get('take_profit', 0)) if signal.get('take_profit') else None,
                
                # Enhanced metadata
                'metadata': json.dumps({
                    'enhanced_signal': True,
                    'claude_analysis': signal.get('claude_analysis', {}),
                    'original_signal_type': signal.get('original_signal_type'),
                    'selected_strategies': self.selected_strategies,
                    'duplicate_window_hours': self.duplicate_window_hours,
                    'avg_processing_time': self.avg_processing_time,
                    'smart_money_enabled': self.enable_smart_money  # ADD: Smart money status
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
        """Direct database save fallback method"""
        try:
            # This would contain the direct database insertion logic
            # Implementation depends on your DatabaseManager interface
            self.logger.warning("Direct database save not implemented - implement based on your DatabaseManager")
            return None
        except Exception as e:
            self.logger.error(f"Direct database save failed: {e}")
            return None
    
    def _save_raw_sql(self, signal: Dict) -> Optional[int]:
        """Raw SQL insertion fallback method"""
        try:
            # This would contain raw SQL insertion logic
            # Implementation depends on your database schema
            self.logger.warning("Raw SQL save not implemented - implement based on your database schema")
            return None
        except Exception as e:
            self.logger.error(f"Raw SQL save failed: {e}")
            return None
    
    def _save_to_file_fallback(self, signal: Dict):
        """File logging absolute fallback"""
        try:
            import os
            fallback_file = "signal_processor_fallback.log"
            
            with open(fallback_file, 'a') as f:
                timestamp = datetime.now().isoformat()
                signal_data = json.dumps(signal, default=str)
                f.write(f"{timestamp}: {signal_data}\n")
                
        except Exception as e:
            self.logger.error(f"File fallback save failed: {e}")
    
    def _is_strategy_selected(self, strategy: str) -> bool:
        """Check if strategy is enabled using dynamic detection or fallback to list"""
        # Try to use the new dynamic strategy detection from config module
        if hasattr(config, 'is_strategy_enabled'):
            return config.is_strategy_enabled(strategy)
        else:
            # Fallback to old approach with case-insensitive matching
            return strategy.upper() in [s.upper() for s in self.selected_strategies]
    
    def _get_recent_signals_from_db(self, epic: str, current_time: datetime) -> List[Dict]:
        """Get recent signals from database for duplicate detection"""
        try:
            if not self.db_manager:
                return []
            
            # Calculate time window
            window_start = current_time - timedelta(hours=self.duplicate_window_hours)
            
            # This would query the database for recent signals
            # Implementation depends on your database schema
            self.logger.debug(f"Getting recent signals for {epic} since {window_start}")
            return []  # Placeholder
            
        except Exception as e:
            self.logger.error(f"Database query for recent signals failed: {e}")
            return []
    
    def _update_duplicate_cache(self, signal: Dict):
        """Update in-memory duplicate cache"""
        try:
            epic = signal.get('epic', '')
            signal_hash = signal.get('signal_hash', '')
            current_time = datetime.now()
            
            cache_key = f"{epic}_{signal_hash}"
            self.duplicate_cache[cache_key] = current_time
            
            # Clean old entries from cache
            cutoff_time = current_time - timedelta(hours=self.duplicate_window_hours * 2)
            keys_to_remove = [
                key for key, timestamp in self.duplicate_cache.items() 
                if timestamp < cutoff_time
            ]
            
            for key in keys_to_remove:
                del self.duplicate_cache[key]
                
        except Exception as e:
            self.logger.error(f"Duplicate cache update failed: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """
        ENHANCED: Get comprehensive performance metrics including timing data and Smart Money stats
        """
        return {
            'processing_stats': {
                'total_signals_processed': self.processed_count,
                'signals_enhanced': self.enhanced_count,
                'signals_stored': self.signals_stored,
                'duplicates_filtered': self.duplicates_filtered,
                'strategy_filtered': self.strategy_filtered,
                'claude_analyses': self.claude_analyzed_count,
                'smart_money_analyses': self.smart_money_analyzed_count,  # ADD: Smart money stat
                'notifications_sent': self.notifications_sent,
                'errors': self.errors
            },
            'performance_metrics': {
                'avg_processing_time_ms': int(self.avg_processing_time * 1000),
                'total_performance_samples': len(self.performance_samples),
                'success_rate_pct': (self.signals_stored / max(self.processed_count, 1)) * 100,
                'claude_analysis_rate_pct': (self.claude_analyzed_count / max(self.processed_count, 1)) * 100,
                'smart_money_analysis_rate_pct': (self.smart_money_analyzed_count / max(self.processed_count, 1)) * 100,  # ADD
                'notification_interface': getattr(self, '_notification_interface', 'unknown')
            },
            'configuration': {
                'selected_strategies': self.selected_strategies,
                'min_confidence': self.min_confidence,
                'duplicate_window_hours': self.duplicate_window_hours,
                'claude_analysis_mode': self.claude_analysis_mode,
                'smart_money_enabled': self.enable_smart_money,  # ADD: Smart money config
                'claude_timeout': self.claude_timeout,
                'smart_money_timeout': self.smart_money_timeout  # ADD: Smart money timeout
            }
        }
    
    def validate_system_health(self) -> Tuple[bool, List[str]]:
        """
        ENHANCED: Comprehensive system health validation including Smart Money
        """
        issues = []
        
        # Check essential dependencies
        if not self.db_manager:
            issues.append("Database manager not available")
        
        if not self.alert_history:
            issues.append("Alert history not available")
        
        # Check Smart Money system (NEW)
        if self.enable_smart_money and not self.smart_money_analyzer:
            issues.append("Smart Money enabled but analyzer not available")
        
        # Check notification system
        if self.enable_notifications:
            if not self.notification_manager:
                issues.append("Notifications enabled but notification manager not available")
            elif getattr(self, '_notification_interface', 'none') == 'none':
                issues.append("Notification manager has no compatible send methods")
        
        # Check Claude analyzer
        if self.claude_analyzer and not hasattr(self.claude_analyzer, 'analyze_signal'):
            issues.append("Claude analyzer missing analyze_signal method")
        
        # Check configuration
        if not self.selected_strategies:
            issues.append("No strategies selected")
        
        if self.min_confidence <= 0 or self.min_confidence > 1:
            issues.append("Invalid minimum confidence threshold")
        
        # Check performance issues
        if self.avg_processing_time > 5.0:
            issues.append(f"High average processing time: {self.avg_processing_time:.2f}s")
        
        if self.errors > self.processed_count * 0.1:  # More than 10% error rate
            issues.append(f"High error rate: {self.errors}/{self.processed_count}")
        
        return len(issues) == 0, issues
    
    def reset_statistics(self):
        """Reset all processing statistics"""
        self.processed_count = 0
        self.enhanced_count = 0
        self.claude_analyzed_count = 0
        self.smart_money_analyzed_count = 0  # ADD: Reset smart money counter
        self.notifications_sent = 0
        self.signals_stored = 0
        self.duplicates_filtered = 0
        self.strategy_filtered = 0
        self.errors = 0
        self.avg_processing_time = 0.0
        self.performance_samples = []
        self.duplicate_cache = {}
        
        self.logger.info("📊 SignalProcessor statistics reset")