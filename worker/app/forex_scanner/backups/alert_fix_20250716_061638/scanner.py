# core/scanner.py (REFACTORED VERSION)
"""
Refactored Intelligent Forex Scanner - Clean Orchestration
Uses extracted components for modular, maintainable architecture
"""

import sys
import os
sys.path.insert(0, '/app/forex_scanner')

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import config


class IntelligentForexScanner:
    """
    Refactored intelligent forex scanner using extracted components
    Clean orchestration of specialized components (~200 lines vs 1300+)
    """
    
    def __init__(
        self,
        db_manager,
        epic_list: List[str] = None,
        intelligence_mode: str = 'backtest_consistent',
        scan_interval: int = 60,
        min_confidence: float = None,
        spread_pips: float = None,
        use_bid_adjustment: bool = None,
        user_timezone: str = 'Europe/Stockholm',
        intelligence_preset: str = None,
        deduplication_config = None,
        **kwargs
    ):
        # Core configuration
        self.db_manager = db_manager
        self.epic_list = epic_list or getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.MINI.IP'])
        self.intelligence_mode = intelligence_mode
        self.scan_interval = scan_interval
        self.min_confidence = min_confidence or getattr(config, 'MIN_CONFIDENCE', 0.7)
        self.spread_pips = spread_pips or getattr(config, 'SPREAD_PIPS', 1.5)
        self.use_bid_adjustment = use_bid_adjustment if use_bid_adjustment is not None else getattr(config, 'USE_BID_ADJUSTMENT', False)
        
        # Runtime state
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self._initialize_components(user_timezone, intelligence_preset, deduplication_config)
        
        # Log initialization
        self._log_initialization()
    
    def _initialize_components(self, user_timezone: str, intelligence_preset: str, deduplication_config):
        """Initialize all components using dependency injection"""
        try:
            # 1. Initialize timezone manager
            from utils.scanner_utils import EnhancedTimezoneManager
            self.timezone_manager = EnhancedTimezoneManager(user_timezone)
            
            # 2. Initialize core detection components
            from core.signal_detector import SignalDetector
            from alerts.alert_history import AlertHistoryManager
            
            self.signal_detector = SignalDetector(self.db_manager, user_timezone)
            self.alert_history = AlertHistoryManager(self.db_manager)
            
            # 3. Initialize deduplication manager
            self._initialize_deduplication_manager(deduplication_config)
            
            # 4. Initialize intelligence manager
            from core.intelligence.intelligence_manager import IntelligenceManager
            self.intelligence_manager = IntelligenceManager(
                intelligence_preset=intelligence_preset,
                db_manager=self.db_manager,
                signal_detector=self.signal_detector,
                logger=self.logger
            )
            
            # 5. Initialize scanner controller
            from core.scanning.scanner_controller import ScannerController
            self.scanner_controller = ScannerController(
                signal_detector=self.signal_detector,
                intelligence_manager=self.intelligence_manager,
                deduplication_manager=self.deduplication_manager,
                timezone_manager=self.timezone_manager,
                epic_list=self.epic_list,
                min_confidence=self.min_confidence,
                spread_pips=self.spread_pips,
                use_bid_adjustment=self.use_bid_adjustment,
                logger=self.logger
            )
            
            # 6. Initialize signal processor
            claude_analyzer = self._initialize_claude_analyzer()
            notification_manager = self._initialize_notification_manager()
            
            from core.processing.signal_processor import SignalProcessor
            self.signal_processor = SignalProcessor(
                claude_analyzer=claude_analyzer,
                notification_manager=notification_manager,
                alert_history=self.alert_history,
                db_manager=self.db_manager,
                logger=self.logger
            )
            
            # 7. Initialize session manager
            from core.monitoring.session_manager import SessionManager
            self.session_manager = SessionManager(
                timezone_manager=self.timezone_manager,
                logger=self.logger
            )
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _initialize_deduplication_manager(self, deduplication_config):
        """Initialize deduplication manager"""
        try:
            from core.alert_deduplication import AlertDeduplicationManager, AlertCooldownConfig
            
            if deduplication_config is None:
                deduplication_config = AlertCooldownConfig(
                    epic_signal_cooldown_minutes=getattr(config, 'ALERT_COOLDOWN_MINUTES', 5),
                    strategy_cooldown_minutes=getattr(config, 'STRATEGY_COOLDOWN_MINUTES', 3),
                    global_cooldown_seconds=getattr(config, 'GLOBAL_COOLDOWN_SECONDS', 30),
                    max_alerts_per_hour=getattr(config, 'MAX_ALERTS_PER_HOUR', 50),
                    max_alerts_per_epic_hour=getattr(config, 'MAX_ALERTS_PER_EPIC_HOUR', 6)
                )
            
            self.deduplication_manager = AlertDeduplicationManager(self.db_manager, deduplication_config)
            self.logger.info("🛡️ Deduplication manager initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize deduplication manager: {e}")
            self.deduplication_manager = None
    
    def _initialize_claude_analyzer(self):
        """Initialize Claude analyzer"""
        try:
            from alerts.claude_api import create_claude_analyzer, create_minimal_claude_analyzer
            
            api_key = getattr(config, 'CLAUDE_API_KEY', None)
            if not api_key:
                self.logger.warning("⚠️ No Claude API key - Claude analysis disabled")
                return None
            
            analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
            
            if analysis_mode == 'minimal':
                analyzer = create_minimal_claude_analyzer(api_key, auto_save=True)
                self.logger.info("✅ Claude minimal analyzer initialized")
            else:
                analyzer = create_claude_analyzer(api_key, auto_save=True)
                self.logger.info("✅ Claude full analyzer initialized")
            
            return analyzer
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Claude analyzer: {e}")
            return None
    
    def _initialize_notification_manager(self):
        """Initialize notification manager"""
        try:
            from alerts.notifications import NotificationManager
            manager = NotificationManager()
            self.logger.info("📱 Notification manager initialized")
            return manager
        except ImportError:
            self.logger.debug("💡 Notification manager not available")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Notification manager initialization failed: {e}")
            return None
    
    def _log_initialization(self):
        """Log initialization summary"""
        current_local = self.timezone_manager.get_current_local_time()
        tz_name = current_local.strftime('%Z')
        
        self.logger.info(f"🧠 Refactored intelligent scanner initialized")
        self.logger.info(f"   Mode: {self.intelligence_mode}")
        self.logger.info(f"   Epic count: {len(self.epic_list)}")
        self.logger.info(f"   Min confidence: {self.min_confidence:.1%}")
        self.logger.info(f"   Spread: {self.spread_pips} pips")
        self.logger.info(f"   User timezone: {self.timezone_manager.user_timezone} ({tz_name})")
        
        # Component status
        self.logger.info(f"   Components initialized:")
        self.logger.info(f"      Scanner Controller: ✅")
        self.logger.info(f"      Intelligence Manager: ✅")
        self.logger.info(f"      Signal Processor: ✅")
        self.logger.info(f"      Session Manager: ✅")
        self.logger.info(f"      Deduplication: {'✅' if self.deduplication_manager else '❌'}")
    
    def scan_once(self) -> List[Dict]:
        """
        Main scan method - clean orchestration using components
        Delegates to ScannerController for all scanning logic
        """
        self.logger.info(f"🔍 Starting scan (intelligence: {self.intelligence_mode})")
        
        # Delegate to scanner controller
        signals = self.scanner_controller.scan_once(intelligence_mode=self.intelligence_mode)
        
        # Process each signal using signal processor
        if signals:
            processed_signals = []
            for i, signal in enumerate(signals, 1):
                self.signal_processor.log_signal_summary(signal, i, len(signals))
                processed_signal = self.signal_processor.process_signal(signal)
                processed_signals.append(processed_signal)
            
            self.logger.info(f"✅ Scan completed: {len(processed_signals)} signals processed")
            return processed_signals
        else:
            self.logger.info("✓ Scan completed: no new signals")
            return []
    
    def start_continuous_scan(self):
        """Start continuous scanning - clean session management"""
        self.logger.info(f"🚀 Starting continuous scanner")
        self.logger.info(f"   Interval: {self.scan_interval}s")
        
        # Start session
        self.session_manager.start_session()
        self.running = True
        
        try:
            while self.running:
                # Perform scan
                signals = self.scan_once()
                
                # Update session statistics
                self.session_manager.update_scan_stats(signals)
                
                # Sleep until next scan
                if self.running:
                    time.sleep(self.scan_interval)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Scanner stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Scanner error: {e}")
            raise
        finally:
            self.running = False
            self.session_manager.stop_session()
    
    def stop(self):
        """Stop the scanner"""
        self.running = False
        self.logger.info("🛑 Scanner stop requested")
    
    # Delegate methods to appropriate components
    
    def get_scanner_status(self) -> Dict:
        """Get comprehensive scanner status"""
        base_status = {
            'running': self.running,
            'intelligence_mode': self.intelligence_mode,
            'scan_interval': self.scan_interval,
            'refactored_version': True,
            'components': {
                'scanner_controller': self.scanner_controller.get_scanning_status(),
                'intelligence_manager': self.intelligence_manager.get_intelligence_status(),
                'signal_processor': self.signal_processor.get_processing_stats(),
                'session_manager': self.session_manager.get_session_status() if hasattr(self.session_manager, 'get_session_status') else {},
                'deduplication_manager': self.deduplication_manager.get_deduplication_stats() if self.deduplication_manager else {'enabled': False}
            }
        }
        
        return base_status
    
    def get_serializable_status(self):
        """Get JSON-serializable status"""
        try:
            from utils.scanner_utils import clean_signal_for_json
            status = self.get_scanner_status()
            return clean_signal_for_json(status)
        except Exception as e:
            self.logger.error(f"❌ Error getting serializable status: {e}")
            return {'error': str(e)}
    
    def update_configuration(self, **kwargs):
        """Update configuration across all components"""
        updated_components = []
        
        # Update scanner controller
        if any(key in kwargs for key in ['epic_list', 'min_confidence', 'spread_pips', 'use_bid_adjustment']):
            controller_updates = self.scanner_controller.update_configuration(**kwargs)
            if controller_updates:
                updated_components.append(f"Scanner Controller: {', '.join(controller_updates)}")
        
        # Update intelligence manager
        if 'intelligence_preset' in kwargs:
            if self.intelligence_manager.change_intelligence_preset(kwargs['intelligence_preset']):
                updated_components.append(f"Intelligence Manager: preset changed to {kwargs['intelligence_preset']}")
        
        # Update signal processor
        processor_keys = ['claude_analysis_mode', 'enable_notifications']
        if any(key in kwargs for key in processor_keys):
            processor_kwargs = {k: v for k, v in kwargs.items() if k in processor_keys}
            processor_updates = self.signal_processor.update_configuration(**processor_kwargs)
            if processor_updates:
                updated_components.append(f"Signal Processor: {', '.join(processor_updates)}")
        
        # Update intelligence mode
        if 'intelligence_mode' in kwargs:
            if kwargs['intelligence_mode'] in ['disabled', 'backtest_consistent', 'live_only', 'enhanced']:
                self.intelligence_mode = kwargs['intelligence_mode']
                updated_components.append(f"Intelligence mode: {self.intelligence_mode}")
            else:
                self.logger.warning(f"Invalid intelligence mode: {kwargs['intelligence_mode']}")
        
        # Update scan interval
        if 'scan_interval' in kwargs:
            self.scan_interval = kwargs['scan_interval']
            updated_components.append(f"Scan interval: {self.scan_interval}s")
        
        if updated_components:
            self.logger.info(f"📝 Configuration updated:")
            for component in updated_components:
                self.logger.info(f"   {component}")
        
        return updated_components
    
    # Intelligence management methods
    def change_intelligence_preset(self, preset_name: str) -> bool:
        """Change intelligence preset"""
        return self.intelligence_manager.change_intelligence_preset(preset_name)
    
    def get_intelligence_status(self) -> Dict:
        """Get intelligence status"""
        return self.intelligence_manager.get_intelligence_status()
    
    def test_intelligence_thresholds(self, epic: str, test_presets: List[str] = None) -> Dict:
        """Test intelligence thresholds"""
        return self.intelligence_manager.test_intelligence_thresholds(epic, test_presets)
    
    # Deduplication management methods
    def get_deduplication_stats(self) -> Dict:
        """Get deduplication statistics"""
        if self.deduplication_manager:
            return self.deduplication_manager.get_deduplication_stats()
        else:
            return {'enabled': False, 'message': 'Deduplication not enabled'}
    
    def test_signal_deduplication(self, epic: str) -> Dict:
        """Test signal deduplication"""
        if self.deduplication_manager:
            try:
                test_signal = {
                    'epic': epic,
                    'signal_type': 'BUY',
                    'strategy': 'TEST',
                    'confidence_score': 0.75,
                    'price': 1.1000,
                    'timestamp': datetime.now().isoformat()
                }
                
                allow, reason, metadata = self.deduplication_manager.should_allow_alert(test_signal)
                
                return {
                    'test_signal': test_signal,
                    'would_allow': allow,
                    'reason': reason,
                    'metadata': metadata
                }
            except Exception as e:
                return {'error': f'Failed to test deduplication: {e}'}
        else:
            return {'error': 'Deduplication not enabled'}
    
    def reset_deduplication_cooldowns(self, epic: str = None, strategy: str = None) -> bool:
        """Reset deduplication cooldowns"""
        if self.deduplication_manager:
            return self.deduplication_manager.reset_cooldowns(epic, strategy)
        else:
            self.logger.warning("⚠️ Deduplication not enabled")
            return False
    

    # Legacy compatibility methods (delegate to components)
    def _scan_single_epic(self, epic: str, enable_multi_tf: bool = False) -> Optional[Dict]:
        """Legacy method - delegates to scanner controller"""
        try:
            return self.scanner_controller._scan_single_epic(epic)
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_single_epic for {epic}: {e}")
            return None
    
    def _scan_without_intelligence(self) -> List[Dict]:
        """Legacy method - delegates to scanner controller"""
        try:
            return self.scanner_controller._detect_raw_signals()
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_without_intelligence: {e}")
            return []
    
    def _scan_with_backtest_intelligence(self) -> List[Dict]:
        """Legacy method - delegates to scanner controller with backtest intelligence"""
        try:
            raw_signals = self.scanner_controller._detect_raw_signals()
            return self.intelligence_manager.apply_backtest_intelligence_filtering(raw_signals)
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_with_backtest_intelligence: {e}")
            return []
    
    def _scan_with_configurable_intelligence(self) -> List[Dict]:
        """Legacy method - delegates to scanner controller with configurable intelligence"""
        try:
            raw_signals = self.scanner_controller._detect_raw_signals()
            return self.intelligence_manager.apply_intelligence_filtering(raw_signals)
        except Exception as e:
            self.logger.error(f"❌ Error in _scan_with_configurable_intelligence: {e}")
            return []
    
    def _process_signal(self, signal: Dict):
        """Legacy method - delegates to signal processor"""
        try:
            return self.signal_processor.process_signal(signal)
        except Exception as e:
            self.logger.error(f"❌ Error in _process_signal: {e}")
            return signal
    
    def _apply_deduplication_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Legacy method - delegates to scanner controller"""
        try:
            return self.scanner_controller._apply_deduplication_filtering(signals)
        except Exception as e:
            self.logger.error(f"❌ Error in _apply_deduplication_filtering: {e}")
            return signals
    
    def _log_signal_summary(self, signal: Dict, index: int, total: int):
        """Legacy method - delegates to signal processor"""
        try:
            return self.signal_processor.log_signal_summary(signal, index, total)
        except Exception as e:
            self.logger.error(f"❌ Error in _log_signal_summary: {e}")

    
        # Testing and diagnostics methods
    def validate_components(self) -> Dict:
        """Validate all components"""
        validation_results = {}
        
        # Validate scanner controller
        try:
            valid, issues = self.scanner_controller.validate_dependencies()
            validation_results['scanner_controller'] = {
                'valid': valid,
                'issues': issues
            }
        except Exception as e:
            validation_results['scanner_controller'] = {
                'valid': False,
                'issues': [f'Validation failed: {e}']
            }
        
        # Validate intelligence manager
        try:
            valid, issues = self.intelligence_manager.validate_intelligence_config()
            validation_results['intelligence_manager'] = {
                'valid': valid,
                'issues': issues
            }
        except Exception as e:
            validation_results['intelligence_manager'] = {
                'valid': False,
                'issues': [f'Validation failed: {e}']
            }
        
        # Overall validation
        all_valid = all(result['valid'] for result in validation_results.values())
        
        return {
            'overall_valid': all_valid,
            'component_results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_scan_epic(self, epic: str) -> Dict:
        """Test scanning for specific epic"""
        return self.scanner_controller.test_epic_scan(epic)
    
    def test_claude_analysis(self) -> Dict:
        """Test Claude analysis"""
        return self.signal_processor.test_claude_analysis()
    
    def test_notification_system(self) -> Dict:
        """Test notification system"""
        return self.signal_processor.test_notification_system()
    
    def get_epic_performance(self) -> Dict:
        """Get epic performance statistics"""
        return self.scanner_controller.get_epic_performance()


# Keep the simple ForexScanner class unchanged
class ForexScanner:
    """Simple forex scanner - kept unchanged for backward compatibility"""
    
    def __init__(
        self,
        db_manager,
        epic_list: List[str],
        scan_interval: int = 60,
        claude_api_key: Optional[str] = None,
        enable_claude_analysis: bool = True,
        use_bid_adjustment: bool = True,
        spread_pips: float = 1.5,
        min_confidence: float = 0.6,
        user_timezone: str = 'Europe/Stockholm'
    ):
        self.db_manager = db_manager
        self.epic_list = epic_list
        self.scan_interval = scan_interval
        self.use_bid_adjustment = use_bid_adjustment
        self.spread_pips = spread_pips
        self.min_confidence = min_confidence
        
        # Initialize timezone manager
        from utils.timezone_utils import TimezoneManager
        self.timezone_manager = TimezoneManager(user_timezone)
        
        # Initialize components with timezone awareness
        from core.signal_detector import SignalDetector
        from alerts.claude_api import ClaudeAnalyzer
        from alerts.notifications import NotificationManager
        
        self.signal_detector = SignalDetector(db_manager, user_timezone)
        self.claude_analyzer = ClaudeAnalyzer(claude_api_key) if claude_api_key else None
        self.notification_manager = NotificationManager()
        
        # State tracking
        self.last_signals = {}
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🔧 Simple scanner initialized:")
        self.logger.info(f"  Epic pairs: {len(epic_list)}")
        self.logger.info(f"  Scan interval: {scan_interval}s")
        self.logger.info(f"  BID adjustment: {use_bid_adjustment}")
        self.logger.info(f"  Claude analysis: {claude_api_key is not None}")
    
    # ... rest of ForexScanner implementation unchanged


# For backward compatibility
IntelligentForexScanner_Legacy = IntelligentForexScanner  # Alias