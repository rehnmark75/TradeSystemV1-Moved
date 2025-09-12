# core/processing/signal_processor.py
"""
Signal Processor - Extracted from IntelligentForexScanner
Handles signal processing, Claude analysis, validation, and notifications
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import config
import re
from alerts.alert_history import AlertHistoryManager  


class SignalProcessor:
    """
    Handles comprehensive signal processing including validation, enhancement,
    Claude analysis, and notification sending
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
        self.claude_analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
        self.enable_notifications = getattr(config, 'ENABLE_NOTIFICATIONS', True)
        
        # Processing statistics
        self.processed_count = 0
        self.enhanced_count = 0
        self.claude_analyzed_count = 0
        self.notifications_sent = 0
        
        self.logger.info("📊 SignalProcessor initialized")
        self.logger.info(f"   Claude analysis: {'✅' if claude_analyzer else '❌'} ({self.claude_analysis_mode})")
        self.logger.info(f"   Notifications: {'✅' if notification_manager else '❌'}")
        self.logger.info(f"   Alert history: {'✅' if alert_history else '❌'}")
    
    def process_signal(self, signal: Dict) -> Dict:
        """
        ENHANCED: Main signal processing method - coordinates all processing steps
        Now includes proper Claude analysis integration and enhanced database storage
        """
        epic = signal.get('epic', 'Unknown')
        signal_type = signal.get('signal_type', 'Unknown')
        confidence = signal.get('confidence_score', 0)
        
        self.logger.info(f"📊 Processing {signal_type} signal for {epic}")
        self.logger.info(f"   Confidence: {confidence:.1%}")
        
        # Initialize processing result tracking
        processing_result = {
            'signal_processed': False,
            'claude_analyzed': False,
            'saved_to_database': False,
            'notifications_sent': False,
            'trade_approved': False,
            'alert_id': None,
            'claude_result': None,
            'processing_errors': []
        }
        
        try:
            # Step 1: Validate and clean signal
            self.logger.debug("🔍 Step 1: Validating signal...")
            validated_signal = self._validate_and_clean_signal(signal)
            
            # Step 2: Enhance signal with metadata
            self.logger.debug("📝 Step 2: Enhancing signal metadata...")
            enhanced_signal = self._enhance_signal_metadata(validated_signal)
            
            # Step 3: Apply Claude analysis EARLY (before database save)
            # This allows us to store Claude results in the database immediately
            self.logger.debug("🤖 Step 3: Applying Claude analysis...")
            claude_result = self._apply_claude_analysis_enhanced(enhanced_signal)
            
            if claude_result:
                processing_result['claude_analyzed'] = True
                processing_result['claude_result'] = claude_result
                
                # Add Claude results to the signal for database storage
                enhanced_signal.update({
                    'claude_analysis': claude_result.get('raw_response', ''),
                    'claude_score': claude_result.get('score'),
                    'claude_decision': claude_result.get('decision'),
                    'claude_approved': claude_result.get('approved', False),
                    'claude_reason': claude_result.get('reason'),
                    'claude_mode': claude_result.get('mode', 'minimal')
                })
                
                self.logger.info(f"🤖 Claude Analysis: {claude_result.get('decision', 'N/A')} "
                            f"(Score: {claude_result.get('score', 'N/A')}/10)")
            else:
                self.logger.warning("⚠️ No Claude analysis available")
            
            # Step 4: Save signal to enhanced database with Claude results
            self.logger.debug("💾 Step 4: Saving to database...")
            alert_id = self._save_signal_to_database_enhanced(enhanced_signal, claude_result)
            
            if alert_id:
                processing_result['saved_to_database'] = True
                processing_result['alert_id'] = alert_id
                enhanced_signal['alert_id'] = alert_id
                self.logger.info(f"💾 Alert saved with ID: {alert_id}")
            else:
                self.logger.warning("⚠️ Database save failed")
                processing_result['processing_errors'].append("Database save failed")
            
            # Step 5: Evaluate trade approval (enhanced with Claude integration)
            self.logger.debug("⚖️ Step 5: Evaluating trade approval...")
            trade_approved = self._evaluate_trade_approval_enhanced(enhanced_signal, claude_result)
            
            processing_result['trade_approved'] = trade_approved
            enhanced_signal['trade_approved'] = trade_approved
            
            # Log trade decision
            if trade_approved:
                approval_reason = self._get_trade_approval_reason(enhanced_signal, claude_result)
                self.logger.info(f"✅ Trade APPROVED: {approval_reason}")
            else:
                rejection_reason = self._get_trade_rejection_reason(enhanced_signal, claude_result)
                self.logger.info(f"❌ Trade REJECTED: {rejection_reason}")
            
            # Step 6: Send enhanced notifications (include Claude analysis)
            self.logger.debug("📢 Step 6: Sending notifications...")
            notification_success = self._send_notifications_enhanced(enhanced_signal, claude_result)
            
            if notification_success:
                processing_result['notifications_sent'] = True
                self.logger.info("📢 Notifications sent successfully")
            else:
                self.logger.warning("⚠️ Notification sending failed")
                processing_result['processing_errors'].append("Notification sending failed")
            
            # Step 7: Update processing statistics (enhanced with Claude stats)
            self.logger.debug("📊 Step 7: Updating statistics...")
            self._update_processing_stats_enhanced(enhanced_signal, claude_result, processing_result)
            
            # Step 8: Execute trade if approved (new step)
            if trade_approved and getattr(self, 'enable_trading', False):
                self.logger.debug("💰 Step 8: Executing trade...")
                execution_result = self._execute_trade_if_approved(enhanced_signal, claude_result)
                processing_result['trade_executed'] = execution_result.get('executed', False)
                enhanced_signal['execution_result'] = execution_result
            
            # Mark processing as successful
            processing_result['signal_processed'] = True
            enhanced_signal['processing_result'] = processing_result
            
            self.logger.info(f"✅ Signal processing completed for {epic}")
            self.processed_count += 1
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signal for {epic}: {e}")
            
            # Enhanced error handling
            processing_result['processing_errors'].append(str(e))
            processing_result['signal_processed'] = False
            
            signal['processing_error'] = str(e)
            signal['processing_result'] = processing_result
            
            # Try to save error to database for tracking
            try:
                self._save_processing_error(signal, str(e))
            except:
                self.logger.error("Failed to save processing error to database")
            
            return signal

    # ENHANCED HELPER METHODS

    def _apply_claude_analysis_enhanced(self, signal: Dict) -> Optional[Dict]:
        """
        ENHANCED: Apply Claude analysis with better error handling and result structure
        """
        if not hasattr(self, 'claude_analyzer') or not self.claude_analyzer:
            self.logger.debug("No Claude analyzer available")
            return None
        
        try:
            # Use minimal analysis for fast approve/reject decisions
            analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
            
            if analysis_mode == 'minimal' and hasattr(self.claude_analyzer, 'analyze_signal_minimal'):
                claude_result = self.claude_analyzer.analyze_signal_minimal(signal)
                
                if claude_result:
                    self.logger.info(f"🤖 Claude Minimal Analysis:")
                    self.logger.info(f"   Score: {claude_result.get('score', 'N/A')}/10")
                    self.logger.info(f"   Decision: {claude_result.get('decision', 'N/A')}")
                    self.logger.info(f"   Approved: {claude_result.get('approved', False)}")
                    self.logger.info(f"   Reason: {claude_result.get('reason', 'N/A')}")
                    
                    return claude_result
                else:
                    self.logger.warning("⚠️ Claude minimal analysis returned no result")
                    return None
            
            else:
                # Fallback to full analysis
                claude_analysis = self.claude_analyzer.analyze_signal(signal)
                
                if claude_analysis:
                    self.logger.info(f"🤖 Claude Full Analysis: {len(claude_analysis)} characters")
                    
                    # Try to extract structured data from full analysis
                    extracted_data = self._extract_decision_from_full_analysis(claude_analysis)
                    
                    return {
                        'mode': 'full',
                        'raw_response': claude_analysis,
                        'analysis': claude_analysis,
                        **extracted_data
                    }
                else:
                    self.logger.warning("⚠️ Claude full analysis returned no result")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Claude analysis failed: {e}")
            return None

    def _save_signal_to_database_enhanced(self, signal: Dict, claude_result: Optional[Dict] = None) -> Optional[int]:
        """
        ENHANCED: Save signal to database using the enhanced AlertHistoryManager
        """
        if not hasattr(self, 'alert_history_manager'):
            self.logger.warning("⚠️ No alert_history_manager available")
            return None
        
        try:
            # Create enhanced alert message
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            strategy = signal.get('strategy', 'Unknown')
            
            alert_message = f"{signal_type} signal for {epic} @ {confidence:.1%} ({strategy})"
            
            # Add Claude information to alert message if available
            if claude_result:
                claude_decision = claude_result.get('decision', 'N/A')
                claude_score = claude_result.get('score', 'N/A')
                alert_message += f" - Claude: {claude_decision} ({claude_score}/10)"
            
            # Determine alert level based on confidence and Claude approval
            alert_level = self._determine_alert_level(signal, claude_result)
            
            # Save using enhanced AlertHistoryManager
            alert_id = self.alert_history_manager.save_alert(
                signal=signal,
                alert_message=alert_message,
                alert_level=alert_level,
                claude_result=claude_result  # This is the key enhancement!
            )
            
            if alert_id:
                self.logger.info(f"✅ Enhanced alert saved: ID #{alert_id}")
            else:
                self.logger.error("❌ Failed to save enhanced alert")
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced database save failed: {e}")
            return None

    def _evaluate_trade_approval_enhanced(self, signal: Dict, claude_result: Optional[Dict] = None) -> bool:
        """
        ENHANCED: Evaluate trade approval with Claude integration
        """
        try:
            # Check basic confidence threshold
            confidence = signal.get('confidence_score', 0)
            min_confidence = getattr(config, 'MIN_CONFIDENCE_FOR_TRADING', 0.75)
            confidence_passed = confidence >= min_confidence
            
            # Check Claude approval if required
            claude_approval_required = getattr(config, 'CLAUDE_REQUIRE_APPROVAL_FOR_TRADING', True)
            claude_passed = True  # Default to True if Claude not required
            
            if claude_approval_required and claude_result:
                claude_approved = claude_result.get('approved', False)
                claude_score = claude_result.get('score', 0)
                min_claude_score = getattr(config, 'CLAUDE_MIN_SCORE_FOR_APPROVAL', 5)
                
                claude_passed = claude_approved and claude_score >= min_claude_score
                
                self.logger.debug(f"Claude approval check: approved={claude_approved}, "
                                f"score={claude_score}, min_score={min_claude_score}, passed={claude_passed}")
            
            elif claude_approval_required and not claude_result:
                # Claude required but not available
                claude_passed = False
                self.logger.warning("⚠️ Claude approval required but no Claude analysis available")
            
            # Final approval decision
            trade_approved = confidence_passed and claude_passed
            
            self.logger.debug(f"Trade approval evaluation: confidence_passed={confidence_passed}, "
                            f"claude_passed={claude_passed}, final_approved={trade_approved}")
            
            return trade_approved
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating trade approval: {e}")
            return False

    def _send_notifications_enhanced(self, signal: Dict, claude_result: Optional[Dict] = None) -> bool:
        """
        ENHANCED: Send notifications with Claude analysis information
        """
        try:
            # Get basic signal information
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            price = signal.get('price', 'N/A')
            
            # Create enhanced notification message
            notification_title = f"{signal_type} Signal: {epic}"
            
            notification_body = f"""
    📊 Signal Details:
    • Pair: {epic}
    • Type: {signal_type}
    • Price: {price}
    • Confidence: {confidence:.1%}
    • Strategy: {signal.get('strategy', 'Unknown')}
    """
            
            # Add Claude analysis to notification if available
            if claude_result:
                claude_decision = claude_result.get('decision', 'N/A')
                claude_score = claude_result.get('score', 'N/A')
                claude_reason = claude_result.get('reason', 'No reason provided')
                
                notification_body += f"""
    🤖 Claude Analysis:
    • Decision: {claude_decision}
    • Score: {claude_score}/10
    • Reason: {claude_reason}
    """
            
            # Add trade approval status
            trade_approved = signal.get('trade_approved', False)
            approval_emoji = "✅" if trade_approved else "❌"
            approval_text = "APPROVED" if trade_approved else "REJECTED"
            notification_body += f"\n{approval_emoji} Trade Status: {approval_text}"
            
            # Send notification using your existing notification system
            if hasattr(self, 'notification_manager'):
                success = self.notification_manager.send_notification(
                    title=notification_title,
                    message=notification_body,
                    level='HIGH' if trade_approved else 'MEDIUM'
                )
                return success
            else:
                # Fallback: log the notification
                self.logger.info(f"📢 Notification: {notification_title}")
                self.logger.info(notification_body)
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Enhanced notification failed: {e}")
            return False

    def _update_processing_stats_enhanced(self, signal: Dict, claude_result: Optional[Dict], processing_result: Dict):
        """
        ENHANCED: Update processing statistics including Claude analysis stats
        """
        try:
            # Update existing stats
            if hasattr(self, 'daily_stats'):
                self.daily_stats['signals_processed'] = self.daily_stats.get('signals_processed', 0) + 1
                
                if signal.get('signal_type') == 'BULL':
                    self.daily_stats['bull_signals'] = self.daily_stats.get('bull_signals', 0) + 1
                elif signal.get('signal_type') == 'BEAR':
                    self.daily_stats['bear_signals'] = self.daily_stats.get('bear_signals', 0) + 1
                
                # ENHANCED: Add Claude statistics
                if claude_result:
                    self.daily_stats['claude_analyses'] = self.daily_stats.get('claude_analyses', 0) + 1
                    
                    if claude_result.get('approved'):
                        self.daily_stats['claude_approved'] = self.daily_stats.get('claude_approved', 0) + 1
                    else:
                        self.daily_stats['claude_rejected'] = self.daily_stats.get('claude_rejected', 0) + 1
                    
                    # Track Claude score distribution
                    claude_score = claude_result.get('score', 0)
                    if claude_score >= 8:
                        self.daily_stats['claude_high_scores'] = self.daily_stats.get('claude_high_scores', 0) + 1
                    elif claude_score <= 3:
                        self.daily_stats['claude_low_scores'] = self.daily_stats.get('claude_low_scores', 0) + 1
                
                # Track trade approvals
                if signal.get('trade_approved'):
                    self.daily_stats['trades_approved'] = self.daily_stats.get('trades_approved', 0) + 1
                
                # Track processing errors
                if processing_result.get('processing_errors'):
                    self.daily_stats['processing_errors'] = self.daily_stats.get('processing_errors', 0) + 1
            
        except Exception as e:
            self.logger.error(f"❌ Error updating enhanced processing stats: {e}")

    def _determine_alert_level(self, signal: Dict, claude_result: Optional[Dict] = None) -> str:
        """
        ENHANCED: Determine alert level based on confidence and Claude analysis
        """
        confidence = signal.get('confidence_score', 0)
        
        # Base level on confidence
        if confidence >= 0.9:
            base_level = 'HIGH'
        elif confidence >= 0.75:
            base_level = 'MEDIUM'
        else:
            base_level = 'LOW'
        
        # Adjust based on Claude analysis
        if claude_result:
            claude_approved = claude_result.get('approved', False)
            claude_score = claude_result.get('score', 0)
            
            if claude_approved and claude_score >= 8:
                return 'HIGH'  # Claude strongly approves
            elif claude_approved and claude_score >= 6:
                return 'MEDIUM'  # Claude moderately approves
            elif not claude_approved:
                return 'LOW'  # Claude rejects
        
        return base_level

    def _get_trade_approval_reason(self, signal: Dict, claude_result: Optional[Dict] = None) -> str:
        """Get human-readable reason for trade approval"""
        reasons = []
        
        confidence = signal.get('confidence_score', 0)
        reasons.append(f"Confidence {confidence:.1%}")
        
        if claude_result and claude_result.get('approved'):
            claude_score = claude_result.get('score', 0)
            reasons.append(f"Claude approved ({claude_score}/10)")
        
        return ", ".join(reasons)

    def _get_trade_rejection_reason(self, signal: Dict, claude_result: Optional[Dict] = None) -> str:
        """Get human-readable reason for trade rejection"""
        reasons = []
        
        confidence = signal.get('confidence_score', 0)
        min_confidence = getattr(config, 'MIN_CONFIDENCE_FOR_TRADING', 0.75)
        
        if confidence < min_confidence:
            reasons.append(f"Low confidence ({confidence:.1%} < {min_confidence:.1%})")
        
        if claude_result and not claude_result.get('approved'):
            claude_reason = claude_result.get('reason', 'Unknown reason')
            reasons.append(f"Claude rejected: {claude_reason}")
        elif getattr(config, 'CLAUDE_REQUIRE_APPROVAL_FOR_TRADING', True) and not claude_result:
            reasons.append("No Claude analysis available")
        
        return ", ".join(reasons) if reasons else "Unknown reason"

    def _execute_trade_if_approved(self, signal: Dict, claude_result: Optional[Dict] = None) -> Dict:
        """
        Execute trade if approved (placeholder for your trading logic)
        """
        execution_result = {
            'executed': False,
            'timestamp': datetime.now().isoformat(),
            'error': None
        }
        
        try:
            if not signal.get('trade_approved', False):
                execution_result['error'] = "Trade not approved"
                return execution_result
            
            # Here you would implement your actual trading execution logic
            # For now, this is a placeholder
            
            epic = signal.get('epic')
            signal_type = signal.get('signal_type')
            price = signal.get('price')
            
            self.logger.info(f"💰 EXECUTING TRADE: {signal_type} {epic} @ {price}")
            
            if claude_result:
                self.logger.info(f"   Claude Score: {claude_result.get('score')}/10")
                self.logger.info(f"   Claude Reason: {claude_result.get('reason')}")
            
            # Mock execution success
            execution_result['executed'] = True
            execution_result['epic'] = epic
            execution_result['signal_type'] = signal_type
            execution_result['price'] = price
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"❌ Trade execution failed: {e}")
            execution_result['error'] = str(e)
            return execution_result

    def _extract_decision_from_full_analysis(self, analysis_text: str) -> Dict:
        """
        Extract structured decision data from full Claude analysis text
        """
        extracted = {
            'score': None,
            'decision': None,
            'approved': False,
            'reason': 'Full analysis provided'
        }
        
        if not analysis_text:
            return extracted
        
        text_upper = analysis_text.upper()
        
        # Try to extract approval/rejection
        if 'APPROVE' in text_upper and 'REJECT' not in text_upper:
            extracted['decision'] = 'APPROVE'
            extracted['approved'] = True
        elif 'REJECT' in text_upper:
            extracted['decision'] = 'REJECT'
            extracted['approved'] = False
        
        # Try to extract score
        import re
        score_patterns = [
            r'(?:score|rating):\s*(\d+)(?:/10)?',
            r'(\d+)/10',
            r'score\s+(\d+)',
            r'rating\s+(\d+)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        extracted['score'] = score
                        break
                except ValueError:
                    continue
        
        return extracted

    def _save_processing_error(self, signal: Dict, error_message: str):
        """
        Save processing error to database for tracking
        """
        try:
            if hasattr(self, 'alert_history_manager'):
                # Create error signal for database storage
                error_signal = signal.copy()
                error_signal['processing_error'] = error_message
                error_signal['alert_level'] = 'ERROR'
                
                self.alert_history_manager.save_alert(
                    signal=error_signal,
                    alert_message=f"Processing error for {signal.get('epic', 'Unknown')}: {error_message}",
                    alert_level='ERROR'
                )
        except Exception as e:
            self.logger.error(f"Failed to save processing error: {e}")
    
    def _validate_and_clean_signal(self, signal: Dict) -> Dict:
        """Validate signal data and clean for JSON compatibility"""
        try:
            # Import utility function for JSON cleaning
            from utils.scanner_utils import clean_signal_for_json
            
            cleaned_signal = clean_signal_for_json(signal)
            
            # Additional validation
            required_fields = ['epic', 'signal_type', 'confidence_score']
            for field in required_fields:
                if field not in cleaned_signal:
                    self.logger.warning(f"⚠️ Missing required field: {field}")
                    cleaned_signal[field] = 'Unknown' if field != 'confidence_score' else 0.0
            
            # Ensure confidence is within valid range
            confidence = cleaned_signal.get('confidence_score', 0)
            if not 0 <= confidence <= 1:
                self.logger.warning(f"⚠️ Invalid confidence score: {confidence}, clamping to [0,1]")
                cleaned_signal['confidence_score'] = max(0, min(1, confidence))
            
            # Add processing timestamp
            cleaned_signal['processing_timestamp'] = datetime.now().isoformat()
            
            self.logger.debug(f"✅ Signal validated and cleaned for {cleaned_signal.get('epic')}")
            return cleaned_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error validating signal: {e}")
            return signal
    
    def _enhance_signal_metadata(self, signal: Dict) -> Dict:
        """Enhance signal with additional metadata"""
        try:
            enhanced_signal = signal.copy()
            
            # Add processing metadata
            enhanced_signal.update({
                'processor_version': '2.0',
                'processing_stage': 'enhanced',
                'enhancement_timestamp': datetime.now().isoformat()
            })
            
            # Add market context if available
            epic = signal.get('epic')
            if epic:
                enhanced_signal.update(self._get_market_context(epic))
            
            # Add signal quality metrics
            enhanced_signal.update(self._calculate_signal_quality_metrics(signal))
            
            # Add risk metrics
            enhanced_signal.update(self._calculate_risk_metrics(signal))
            
            self.enhanced_count += 1
            self.logger.debug(f"✅ Signal enhanced with metadata for {epic}")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error enhancing signal metadata: {e}")
            return signal
    
    def _get_market_context(self, epic: str) -> Dict:
        """Get market context for the epic"""
        try:
            context = {}
            
            # Get pair info
            pair_info = getattr(config, 'PAIR_INFO', {}).get(epic, {})
            if pair_info:
                context['pair_info'] = pair_info
            
            # Add market session info
            from datetime import datetime
            current_time = datetime.now()
            context['market_session'] = self._determine_market_session(current_time)
            
            # Add volatility context (if available)
            context['market_volatility'] = self._get_current_volatility_context(epic)
            
            return {'market_context': context}
            
        except Exception as e:
            self.logger.debug(f"Could not get market context for {epic}: {e}")
            return {}
    
    def _determine_market_session(self, timestamp: datetime) -> str:
        """Determine current market session"""
        try:
            hour = timestamp.hour
            
            # Simplified session determination (UTC time)
            if 22 <= hour or hour < 6:
                return 'SYDNEY'
            elif 6 <= hour < 8:
                return 'TOKYO'
            elif 8 <= hour < 16:
                return 'LONDON'
            elif 16 <= hour < 22:
                return 'NEW_YORK'
            else:
                return 'UNKNOWN'
                
        except Exception:
            return 'UNKNOWN'
    
    def _get_current_volatility_context(self, epic: str) -> str:
        """Get current volatility context"""
        try:
            # This would typically query recent price data
            # For now, return a placeholder
            return 'NORMAL'
        except Exception:
            return 'UNKNOWN'
    
    def _calculate_signal_quality_metrics(self, signal: Dict) -> Dict:
        """Calculate signal quality metrics"""
        try:
            quality_metrics = {}
            
            # Base quality from confidence
            confidence = signal.get('confidence_score', 0)
            quality_metrics['base_quality'] = confidence
            
            # Strategy quality
            strategy = signal.get('strategy', 'unknown')
            quality_metrics['strategy_quality'] = self._get_strategy_quality_score(strategy)
            
            # Multi-timeframe quality (if available)
            confluence_score = signal.get('confluence_score', 0)
            if confluence_score > 0:
                quality_metrics['mtf_quality'] = confluence_score
            
            # Combined quality score
            quality_scores = [v for v in quality_metrics.values() if v > 0]
            if quality_scores:
                quality_metrics['combined_quality'] = sum(quality_scores) / len(quality_scores)
            else:
                quality_metrics['combined_quality'] = confidence
            
            return {'quality_metrics': quality_metrics}
            
        except Exception as e:
            self.logger.debug(f"Error calculating quality metrics: {e}")
            return {}
    
    def _get_strategy_quality_score(self, strategy: str) -> float:
        """Get quality score based on strategy type"""
        strategy_scores = {
            'EMA_CROSSOVER': 0.8,
            'MACD_EMA': 0.85,
            'COMBINED': 0.9,
            'SCALPING': 0.75,
            'unknown': 0.5
        }
        return strategy_scores.get(strategy, 0.7)
    
    def _calculate_risk_metrics(self, signal: Dict) -> Dict:
        """Calculate risk-related metrics"""
        try:
            risk_metrics = {}
            
            # Confidence-based risk
            confidence = signal.get('confidence_score', 0)
            risk_metrics['confidence_risk'] = 1.0 - confidence
            
            # Time-based risk (market hours)
            current_hour = datetime.now().hour
            if 8 <= current_hour <= 16:  # London session
                risk_metrics['time_risk'] = 0.2  # Low risk
            elif 16 <= current_hour <= 20:  # NY session
                risk_metrics['time_risk'] = 0.3  # Medium risk
            else:
                risk_metrics['time_risk'] = 0.6  # Higher risk
            
            # Volatility risk (placeholder)
            risk_metrics['volatility_risk'] = 0.4  # Medium default
            
            # Combined risk score
            risk_scores = list(risk_metrics.values())
            risk_metrics['combined_risk'] = sum(risk_scores) / len(risk_scores)
            
            return {'risk_metrics': risk_metrics}
            
        except Exception as e:
            self.logger.debug(f"Error calculating risk metrics: {e}")
            return {}
    
    def _save_signal_to_database(self, signal: Dict):
        """Save signal to database if not already saved"""
        try:
            # Check if signal already has an alert_id (already saved)
            if signal.get('alert_id'):
                self.logger.debug(f"Signal already saved with ID {signal['alert_id']}")
                return
            
            if not self.alert_history:
                self.logger.debug("No alert history manager available")
                return
            
            # Save signal
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            
            alert_message = f"{signal_type} signal for {epic} at {confidence:.1%} confidence"
            
            from utils.scanner_utils import save_signal_with_logging
            alert_id = save_signal_with_logging(self.alert_history, signal, self.logger)
            
            if alert_id:
                signal['alert_id'] = alert_id
                self.logger.debug(f"✅ Signal saved to database with ID {alert_id}")
            else:
                self.logger.warning("⚠️ Failed to save signal to database")
                
        except Exception as e:
            self.logger.error(f"❌ Error saving signal to database: {e}")
    
    def _apply_claude_analysis(self, signal: Dict) -> Dict:
        """Apply Claude analysis to signal"""
        if not self.claude_analyzer:
            self.logger.debug("No Claude analyzer available")
            return signal
        
        try:
            self.logger.info("🤖 Applying Claude analysis...")
            
            claude_enhanced_signal = signal.copy()
            
            if self.claude_analysis_mode == 'minimal':
                claude_result = self.claude_analyzer.analyze_signal_minimal(signal)
                
                if claude_result:
                    claude_enhanced_signal.update({
                        'claude_analysis': claude_result.get('raw_response', 'Analysis completed'),
                        'claude_quality_score': claude_result.get('score'),
                        'claude_decision': claude_result.get('decision'),
                        'claude_approved': claude_result.get('approved', False),
                        'claude_reason': claude_result.get('reason', 'No reason provided')
                    })
                    
                    self.logger.info(f"   Claude Score: {claude_result.get('score')}/10")
                    self.logger.info(f"   Claude Decision: {'✅' if claude_result.get('approved') else '❌'} {claude_result.get('decision')}")
                    self.logger.info(f"   Claude Reason: {claude_result.get('reason')}")
                    
                else:
                    self.logger.warning("⚠️ No Claude minimal analysis received")
                    
            else:
                # Full Claude analysis
                claude_analysis = self.claude_analyzer.analyze_signal(signal)
                
                if claude_analysis:
                    # Extract Claude metrics using utility functions
                    from utils.scanner_utils import ClaudeIntegration
                    claude_integration = ClaudeIntegration(self.logger)
                    
                    claude_quality_score = claude_integration.extract_claude_quality_score(claude_analysis)
                    claude_decision = claude_integration.extract_claude_decision(claude_analysis)
                    claude_approved = claude_decision == 'TRADE'
                    
                    claude_enhanced_signal.update({
                        'claude_analysis': claude_analysis,
                        'claude_quality_score': claude_quality_score,
                        'claude_decision': claude_decision,
                        'claude_approved': claude_approved
                    })
                    
                    self.logger.info(f"   Claude Quality Score: {claude_quality_score}/10")
                    self.logger.info(f"   Claude Decision: {'✅' if claude_approved else '❌'} {claude_decision}")
                    
                else:
                    self.logger.warning("⚠️ No Claude analysis received")
            
            # Update database with Claude analysis
            if claude_enhanced_signal.get('alert_id') and claude_enhanced_signal.get('claude_analysis'):
                try:
                    from utils.scanner_utils import update_alert_with_claude
                    update_alert_with_claude(
                        self.db_manager, 
                        claude_enhanced_signal['alert_id'], 
                        claude_enhanced_signal['claude_analysis'], 
                        self.logger
                    )
                    self.logger.debug(f"✅ Alert {claude_enhanced_signal['alert_id']} updated with Claude analysis")
                except Exception as e:
                    self.logger.error(f"❌ Error updating alert with Claude analysis: {e}")
            
            self.claude_analyzed_count += 1
            return claude_enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Claude analysis failed: {e}")
            return signal
    
    def _evaluate_trade_approval(self, signal: Dict) -> bool:
        """Evaluate if trade should be approved"""
        try:
            # Use utility function for trade approval
            from utils.scanner_utils import evaluate_trade_approval
            
            claude_quality_score = signal.get('claude_quality_score')
            claude_decision = signal.get('claude_decision')
            
            trade_approved = evaluate_trade_approval(
                signal, 
                claude_quality_score, 
                claude_decision, 
                self.min_confidence, 
                config
            )
            
            # Override with Claude's direct approval if available
            if 'claude_approved' in signal:
                trade_approved = trade_approved and signal['claude_approved']
                self.logger.info(f"✅ Trade approval updated with Claude decision: {trade_approved}")
            
            if trade_approved:
                self.logger.info(f"✅ Signal approved for trading")
            else:
                self.logger.info(f"❌ Signal NOT approved for trading")
            
            return trade_approved
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating trade approval: {e}")
            return False
    
    def _send_notifications(self, signal: Dict):
        """Send notifications for the signal"""
        if not self.enable_notifications or not self.notification_manager:
            self.logger.debug("Notifications disabled or manager not available")
            return
        
        try:
            epic = signal.get('epic')
            claude_quality_score = signal.get('claude_quality_score')
            claude_decision = signal.get('claude_decision')
            trade_approved = signal.get('trade_approved', False)
            
            # Use utility function for enhanced notifications
            from utils.scanner_utils import send_enhanced_notification
            
            send_enhanced_notification(
                self.notification_manager, 
                signal, 
                claude_quality_score, 
                claude_decision, 
                trade_approved, 
                self.logger
            )
            
            self.notifications_sent += 1
            self.logger.info(f"📢 Notification sent for {epic} signal")
            
        except Exception as e:
            self.logger.error(f"❌ Notification failed: {e}")
    
    def _update_processing_stats(self, signal: Dict):
        """Update processing statistics"""
        try:
            # This could be expanded to track more detailed statistics
            epic = signal.get('epic')
            confidence = signal.get('confidence_score', 0)
            trade_approved = signal.get('trade_approved', False)
            
            # Log processing stats periodically
            if self.processed_count % 10 == 0:
                self.logger.info(f"📊 Processing stats: {self.processed_count} processed, "
                               f"{self.enhanced_count} enhanced, {self.claude_analyzed_count} analyzed, "
                               f"{self.notifications_sent} notifications sent")
                
        except Exception as e:
            self.logger.debug(f"Error updating processing stats: {e}")
    
    def log_signal_summary(self, signal: Dict, index: int, total: int):
        """
        Log detailed signal summary
        Extracted from _log_signal_summary in IntelligentForexScanner
        """
        try:
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            strategy = signal.get('strategy', 'Unknown')
            signal_time = signal.get('timestamp_display', 'Unknown time')
            
            self.logger.info(f"🚀 Processing signal {index}/{total}:")
            self.logger.info(f"📊 [{index}/{total}] {epic} {signal_type} ({strategy}) - {confidence:.1f}%")
            self.logger.info(f"   📅 Signal time: {signal_time}")
            
            if 'intelligence_score' in signal:
                self.logger.info(f"   🧠 Intelligence: {signal['intelligence_score']:.1%}")
            
            if 'signal_hash' in signal:
                self.logger.info(f"   🛡️ Dedup hash: {signal['signal_hash'][:8]}...")
            
            if 'price_mid' in signal:
                exec_price = signal.get('execution_price', 'N/A')
                self.logger.info(f"   💰 MID: {signal['price_mid']:.5f}, EXEC: {exec_price}")
            
            if 'claude_quality_score' in signal:
                claude_approved = '✅' if signal.get('claude_approved') else '❌'
                self.logger.info(f"   🤖 Claude: {signal['claude_quality_score']}/10 {claude_approved}")
            
            if 'trade_approved' in signal:
                approval_status = '✅ APPROVED' if signal['trade_approved'] else '❌ REJECTED'
                self.logger.info(f"   💼 Trade: {approval_status}")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging signal summary: {e}")
    
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics"""
        return {
            'processed_count': self.processed_count,
            'enhanced_count': self.enhanced_count,
            'claude_analyzed_count': self.claude_analyzed_count,
            'notifications_sent': self.notifications_sent,
            'claude_analysis_mode': self.claude_analysis_mode,
            'min_confidence_threshold': self.min_confidence,
            'notifications_enabled': self.enable_notifications
        }
    
    def reset_processing_stats(self):
        """Reset processing statistics"""
        self.processed_count = 0
        self.enhanced_count = 0
        self.claude_analyzed_count = 0
        self.notifications_sent = 0
        self.logger.info("📊 Processing statistics reset")
    
    def validate_signal_format(self, signal: Dict) -> Tuple[bool, List[str]]:
        """Validate signal format and return validation results"""
        errors = []
        
        try:
            # Check required fields
            required_fields = ['epic', 'signal_type', 'confidence_score']
            for field in required_fields:
                if field not in signal:
                    errors.append(f"Missing required field: {field}")
                elif signal[field] is None:
                    errors.append(f"Field {field} is None")
            
            # Validate field types and values
            if 'confidence_score' in signal:
                confidence = signal['confidence_score']
                if not isinstance(confidence, (int, float)):
                    errors.append(f"confidence_score must be numeric, got {type(confidence)}")
                elif not 0 <= confidence <= 1:
                    errors.append(f"confidence_score must be between 0 and 1, got {confidence}")
            
            if 'signal_type' in signal:
                signal_type = signal['signal_type']
                if signal_type not in ['BUY', 'SELL']:
                    errors.append(f"signal_type must be 'BUY' or 'SELL', got '{signal_type}'")
            
            # Check for common timestamp fields
            timestamp_fields = ['timestamp', 'market_timestamp', 'processing_timestamp']
            has_timestamp = any(field in signal for field in timestamp_fields)
            if not has_timestamp:
                errors.append("No timestamp field found")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def process_signal_batch(self, signals: List[Dict]) -> List[Dict]:
        """Process a batch of signals efficiently"""
        if not signals:
            return []
        
        self.logger.info(f"📊 Processing batch of {len(signals)} signals")
        
        processed_signals = []
        
        for i, signal in enumerate(signals, 1):
            try:
                # Log signal summary first
                self.log_signal_summary(signal, i, len(signals))
                
                # Process the signal
                processed_signal = self.process_signal(signal)
                processed_signals.append(processed_signal)
                
            except Exception as e:
                self.logger.error(f"❌ Error processing signal {i}/{len(signals)}: {e}")
                # Add error info to signal and continue
                signal['processing_error'] = str(e)
                processed_signals.append(signal)
        
        self.logger.info(f"✅ Batch processing completed: {len(processed_signals)} signals processed")
        return processed_signals
    
    def set_claude_analyzer(self, claude_analyzer):
        """Set Claude analyzer (dependency injection)"""
        self.claude_analyzer = claude_analyzer
        self.logger.info("🤖 Claude analyzer updated")
    
    def set_notification_manager(self, notification_manager):
        """Set notification manager (dependency injection)"""
        self.notification_manager = notification_manager
        self.logger.info("📢 Notification manager updated")
    
    def set_alert_history(self, alert_history):
        """Set alert history manager (dependency injection)"""
        self.alert_history = alert_history
        self.logger.info("💾 Alert history manager updated")
    
    def update_configuration(self, **kwargs):
        """Update processor configuration dynamically"""
        updated_items = []
        
        if 'claude_analysis_mode' in kwargs:
            self.claude_analysis_mode = kwargs['claude_analysis_mode']
            updated_items.append(f"Claude mode: {self.claude_analysis_mode}")
        
        if 'min_confidence' in kwargs:
            self.min_confidence = kwargs['min_confidence']
            updated_items.append(f"Min confidence: {self.min_confidence:.1%}")
        
        if 'enable_notifications' in kwargs:
            self.enable_notifications = kwargs['enable_notifications']
            status = "enabled" if self.enable_notifications else "disabled"
            updated_items.append(f"Notifications: {status}")
        
        if updated_items:
            self.logger.info(f"📝 SignalProcessor configuration updated:")
            for item in updated_items:
                self.logger.info(f"   {item}")
        
        return updated_items
    
    def get_configuration(self) -> Dict:
        """Get current processor configuration"""
        return {
            'claude_analysis_mode': self.claude_analysis_mode,
            'min_confidence': self.min_confidence,
            'enable_notifications': self.enable_notifications,
            'claude_analyzer_available': self.claude_analyzer is not None,
            'notification_manager_available': self.notification_manager is not None,
            'alert_history_available': self.alert_history is not None,
            'db_manager_available': self.db_manager is not None
        }
    
    def test_claude_analysis(self, test_signal: Dict = None) -> Dict:
        """Test Claude analysis with a sample signal"""
        if not self.claude_analyzer:
            return {'error': 'Claude analyzer not available'}
        
        if test_signal is None:
            test_signal = {
                'epic': 'CS.D.EURUSD.MINI.IP',
                'signal_type': 'BUY',
                'strategy': 'TEST',
                'confidence_score': 0.75,
                'price_mid': 1.1000,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            self.logger.info("🧪 Testing Claude analysis...")
            
            if self.claude_analysis_mode == 'minimal':
                result = self.claude_analyzer.analyze_signal_minimal(test_signal)
                
                return {
                    'success': True,
                    'mode': 'minimal',
                    'result': result,
                    'test_signal': test_signal
                }
            else:
                result = self.claude_analyzer.analyze_signal(test_signal)
                
                return {
                    'success': True,
                    'mode': 'full',
                    'result': result,
                    'test_signal': test_signal
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'test_signal': test_signal
            }
    
    def test_notification_system(self, test_signal: Dict = None) -> Dict:
        """Test notification system with a sample signal"""
        if not self.notification_manager:
            return {'error': 'Notification manager not available'}
        
        if test_signal is None:
            test_signal = {
                'epic': 'CS.D.EURUSD.MINI.IP',
                'signal_type': 'BUY',
                'strategy': 'TEST',
                'confidence_score': 0.75,
                'claude_quality_score': 8,
                'claude_decision': 'TRADE',
                'trade_approved': True,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            self.logger.info("🧪 Testing notification system...")
            
            # Test basic notification
            self.notification_manager.send_signal_alert(
                test_signal,
                f"TEST: {test_signal['signal_type']} signal for {test_signal['epic']}"
            )
            
            return {
                'success': True,
                'message': 'Test notification sent successfully',
                'test_signal': test_signal
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'test_signal': test_signal
            }