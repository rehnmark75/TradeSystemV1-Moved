# core/scanning/scanner_controller.py
"""
Scanner Controller - Extracted from IntelligentForexScanner
Handles scanning orchestration, signal coordination, and filtering pipeline
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import config


class ScannerController:
    """
    Controls the scanning process, coordinates signal detection, and manages filtering pipeline
    Extracted from IntelligentForexScanner to provide modular scanning orchestration
    """
    
    def __init__(self,
                 signal_detector,
                 intelligence_manager=None,
                 deduplication_manager=None,
                 timezone_manager=None,
                 epic_list: List[str] = None,
                 min_confidence: float = None,
                 spread_pips: float = None,
                 use_bid_adjustment: bool = None,
                 logger: Optional[logging.Logger] = None):
        
        # Core dependencies
        self.signal_detector = signal_detector
        self.intelligence_manager = intelligence_manager
        self.deduplication_manager = deduplication_manager
        self.timezone_manager = timezone_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.epic_list = epic_list or getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP'])
        self.min_confidence = min_confidence or getattr(config, 'MIN_CONFIDENCE', 0.7)
        self.spread_pips = spread_pips or getattr(config, 'SPREAD_PIPS', 1.5)
        self.use_bid_adjustment = use_bid_adjustment if use_bid_adjustment is not None else getattr(config, 'USE_BID_ADJUSTMENT', False)
        
        # Scanning state
        self.last_signals = {}  # Track last signal time for each epic
        self.scan_count = 0
        self.total_signals_detected = 0
        self.total_signals_filtered = 0
        
        # Multi-timeframe configuration
        self.enable_multi_timeframe = getattr(config, 'ENABLE_MULTI_TIMEFRAME_ANALYSIS', False)
        self.default_timeframe = getattr(config, 'DEFAULT_TIMEFRAME', '5m')
        
        self.logger.info("🔍 ScannerController initialized")
        self.logger.info(f"   Epic pairs: {len(self.epic_list)}")
        self.logger.info(f"   Min confidence: {self.min_confidence:.1%}")
        self.logger.info(f"   Multi-timeframe: {self.enable_multi_timeframe}")
        self.logger.info(f"   Intelligence: {'✅' if intelligence_manager else '❌'}")
        self.logger.info(f"   Deduplication: {'✅' if deduplication_manager else '❌'}")
    
    def scan_once(self, intelligence_mode: str = 'backtest_consistent') -> List[Dict]:
        """
        Main scanning method with configurable intelligence modes
        Extracted from scan_once in IntelligentForexScanner
        """
        self.scan_count += 1
        self.logger.info(f"🔍 Scanning {len(self.epic_list)} epics (scan #{self.scan_count})")
        self.logger.info(f"   Min confidence: {self.min_confidence * 100:.1f}%")
        self.logger.info(f"   Intelligence mode: {intelligence_mode}")
        
        # Step 1: Detect raw signals from all epics
        raw_signals = self._detect_raw_signals()
        
        if not raw_signals:
            self.logger.info("📊 No signals detected this scan")
            return []
        
        self.logger.info(f"📊 Raw signals detected: {len(raw_signals)}")
        
        # Step 2: Normalize timestamps
        normalized_signals = self._normalize_signal_timestamps(raw_signals)
        
        # Step 3: Apply confidence filtering
        confidence_filtered = self._apply_confidence_filtering(normalized_signals)
        
        # Step 4: Apply intelligence filtering based on mode
        intelligence_filtered = self._apply_intelligence_filtering(confidence_filtered, intelligence_mode)
        
        # Step 5: Apply deduplication filtering
        deduplicated_signals = self._apply_deduplication_filtering(intelligence_filtered)
        
        # Step 6: Apply legacy timestamp filtering as backup
        new_signals = self._apply_legacy_timestamp_filtering(deduplicated_signals)
        
        # Update statistics
        self.total_signals_detected += len(raw_signals)
        self.total_signals_filtered += len(raw_signals) - len(new_signals)
        
        if new_signals:
            self.logger.info(f"✅ Final result: {len(new_signals)} new qualified signals")
        else:
            self.logger.info("✓ No new signals after all filtering")
        
        return new_signals
    
    def _detect_raw_signals(self) -> List[Dict]:
        """Detect raw signals from all configured epics"""
        raw_signals = []
        
        for epic in self.epic_list:
            try:
                signal = self._scan_single_epic(epic)
                if signal:
                    raw_signals.append(signal)
                    self.logger.debug(f"✅ Signal detected for {epic}")
                else:
                    self.logger.debug(f"📊 No signal for {epic}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error scanning {epic}: {e}")
        
        return raw_signals
    
    def scan_single_epic(self, epic: str, enable_multi_tf: bool = False) -> Optional[Dict]:
        """Public method to scan a single epic"""
        return self._scan_single_epic(epic)
    
    def _scan_single_epic(self, epic: str) -> Optional[Dict]:
        """
        Scan a single epic for signals with enhanced MTF handling
        Extracted from _scan_single_epic in IntelligentForexScanner
        """
        try:
            # Get pair information
            pair_info = getattr(config, 'PAIR_INFO', {}).get(epic, {'pair': 'EURUSD', 'pip_multiplier': 10000})
            pair_name = pair_info['pair']
            
            if not pair_name or pair_name == 'EURUSD':
                pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '')
            
            signal = None
            
            # Try multi-timeframe analysis if enabled
            if self.enable_multi_timeframe:
                signal = self._attempt_multi_timeframe_scan(epic, pair_name)
            
            # Fallback to single-timeframe if MTF failed or disabled
            if not signal:
                signal = self._attempt_single_timeframe_scan(epic, pair_name)
            
            # Add scanning metadata
            if signal:
                signal.update({
                    'scan_timestamp': datetime.now().isoformat(),
                    'scan_count': self.scan_count,
                    'scanner_version': '2.0'
                })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error in single epic scan for {epic}: {e}")
            return None
    
    def _attempt_multi_timeframe_scan(self, epic: str, pair_name: str) -> Optional[Dict]:
        """Attempt multi-timeframe signal detection"""
        try:
            self.logger.debug(f"🔍 MTF scan: {epic}")
            
            signal = self.signal_detector.detect_signals_multi_timeframe(
                epic=epic,
                pair=pair_name,
                spread_pips=self.spread_pips,
                primary_timeframe=self.default_timeframe
            )
            
            if signal:
                confidence = signal.get('confidence_score', 0)
                strategy = signal.get('strategy', 'unknown')
                confluence = signal.get('confluence_score', 0)
                timeframes = signal.get('timeframes_analyzed', [])
                
                self.logger.info(f"🎯 MTF Signal: {signal['signal_type']} {epic}")
                self.logger.info(f"   📊 Strategy: {strategy}, Confidence: {confidence:.1%}")
                self.logger.info(f"   🤝 Confluence: {confluence:.1%}, TFs: {', '.join(timeframes)}")
                
                signal.update({
                    'mtf_enabled': True,
                    'mtf_success': True,
                    'scanning_mode': 'multi_timeframe'
                })
                
                return signal
            else:
                self.logger.debug(f"📊 No MTF signal for {epic}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ MTF scan failed for {epic}: {e}")
            return None
    
    def _attempt_single_timeframe_scan(self, epic: str, pair_name: str) -> Optional[Dict]:
        """Attempt single-timeframe signal detection"""
        try:
            self.logger.debug(f"🔍 Single TF scan: {epic}")
            
            if self.use_bid_adjustment:
                signal = self.signal_detector.detect_signals_bid_adjusted(
                    epic=epic,
                    pair=pair_name,
                    spread_pips=self.spread_pips,
                    timeframe=self.default_timeframe
                )
            else:
                signal = self.signal_detector.detect_signals_mid_prices(
                    epic=epic,
                    pair=pair_name,
                    timeframe=self.default_timeframe
                )
            
            if signal:
                confidence = signal.get('confidence_score', 0)
                strategy = signal.get('strategy', 'unknown')
                
                self.logger.info(f"🎯 Signal: {signal['signal_type']} {epic}")
                self.logger.info(f"   📊 Strategy: {strategy}, Confidence: {confidence:.1%}")
                
                signal.update({
                    'mtf_enabled': self.enable_multi_timeframe,
                    'mtf_success': False,
                    'scanning_mode': 'single_timeframe',
                    'timeframe': self.default_timeframe
                })
                
                return signal
            else:
                self.logger.debug(f"📊 No single TF signal for {epic}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Single TF scan failed for {epic}: {e}")
            return None
    
    def _normalize_signal_timestamps(self, signals: List[Dict]) -> List[Dict]:
        """Normalize signal timestamps using timezone manager"""
        if not self.timezone_manager:
            return signals
        
        normalized_signals = []
        for signal in signals:
            try:
                normalized_signal = self.timezone_manager.normalize_signal_timestamp(signal)
                normalized_signals.append(normalized_signal)
            except Exception as e:
                self.logger.warning(f"⚠️ Timestamp normalization failed for {signal.get('epic')}: {e}")
                normalized_signals.append(signal)
        
        return normalized_signals
    
    def _apply_confidence_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Apply confidence threshold filtering"""
        if not signals:
            return []
        
        qualified_signals = [
            signal for signal in signals 
            if signal.get('confidence_score', 0) >= self.min_confidence
        ]
        
        self.logger.info(f"📊 Confidence filter: {len(qualified_signals)}/{len(signals)} signals passed ({self.min_confidence:.1%} threshold)")
        
        return qualified_signals
    
    def _apply_intelligence_filtering(self, signals: List[Dict], intelligence_mode: str) -> List[Dict]:
        """Apply intelligence filtering based on mode"""
        if not signals or not self.intelligence_manager:
            self.logger.debug("📊 Skipping intelligence filtering - no signals or no manager")
            return signals
        
        try:
            if intelligence_mode == 'disabled':
                self.logger.info("🧠 Intelligence filtering: DISABLED - all signals pass")
                return signals
            
            elif intelligence_mode == 'backtest_consistent':
                self.logger.info("🧠 Applying backtest-consistent intelligence filtering")
                return self.intelligence_manager.apply_backtest_intelligence_filtering(signals)
            
            elif intelligence_mode in ['live_only', 'enhanced']:
                self.logger.info(f"🧠 Applying {intelligence_mode} intelligence filtering")
                return self.intelligence_manager.apply_intelligence_filtering(signals)
            
            else:
                self.logger.warning(f"⚠️ Unknown intelligence mode: {intelligence_mode}, using default")
                return self.intelligence_manager.apply_intelligence_filtering(signals)
                
        except Exception as e:
            self.logger.error(f"❌ Intelligence filtering failed: {e}")
            return signals  # Return unfiltered signals on error
    
    def _apply_deduplication_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Apply deduplication filtering"""
        if not signals or not self.deduplication_manager:
            if not self.deduplication_manager:
                self.logger.debug("🛡️ Deduplication disabled - all signals passed")
            return signals
        
        try:
            deduplicated_signals = []
            self.logger.info(f"🛡️ Applying deduplication to {len(signals)} signals")
            
            for signal in signals:
                allow, reason, metadata = self.deduplication_manager.should_allow_alert(signal)
                
                if allow:
                    # Add deduplication metadata to signal
                    signal.update({
                        'signal_hash': metadata['signal_hash'],
                        'cooldown_key': metadata['cooldown_key'],
                        'dedup_approved': True
                    })
                    deduplicated_signals.append(signal)
                else:
                    epic = signal.get('epic', 'Unknown')
                    self.logger.info(f"🚫 Signal blocked by deduplication: {epic} - {reason}")
            
            self.logger.info(f"🛡️ Deduplication result: {len(deduplicated_signals)}/{len(signals)} signals passed")
            return deduplicated_signals
            
        except Exception as e:
            self.logger.error(f"❌ Deduplication filtering failed: {e}")
            return signals  # Return unfiltered signals on error
    
    def _apply_legacy_timestamp_filtering(self, signals: List[Dict]) -> List[Dict]:
        """Apply legacy timestamp filtering as backup"""
        if not signals:
            return []
        
        new_signals = []
        
        for signal in signals:
            signal_time = signal.get('timestamp')
            epic = signal.get('epic')
            
            if signal_time and epic:
                try:
                    from utils.scanner_utils import is_new_signal
                    
                    if is_new_signal(self.last_signals, epic, signal_time, self.logger):
                        self.last_signals[epic] = signal_time
                        new_signals.append(signal)
                    else:
                        self.logger.debug(f"🕐 Legacy filter: {epic} signal too recent")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Legacy timestamp filtering failed for {epic}: {e}")
                    new_signals.append(signal)  # Include signal on error
            else:
                # Include signals without proper timestamp/epic info
                self.logger.warning(f"⚠️ Signal missing timestamp or epic info")
                new_signals.append(signal)
        
        if len(new_signals) != len(signals):
            self.logger.info(f"🕐 Legacy timestamp filter: {len(new_signals)}/{len(signals)} signals passed")
        
        return new_signals
    
    def get_scanning_status(self) -> Dict:
        """Get current scanning status and statistics"""
        return {
            'epic_count': len(self.epic_list),
            'scan_count': self.scan_count,
            'total_signals_detected': self.total_signals_detected,
            'total_signals_filtered': self.total_signals_filtered,
            'last_signals': dict(self.last_signals),
            'configuration': {
                'min_confidence': self.min_confidence,
                'spread_pips': self.spread_pips,
                'use_bid_adjustment': self.use_bid_adjustment,
                'enable_multi_timeframe': self.enable_multi_timeframe,
                'default_timeframe': self.default_timeframe
            },
            'components': {
                'signal_detector_available': self.signal_detector is not None,
                'intelligence_manager_available': self.intelligence_manager is not None,
                'deduplication_manager_available': self.deduplication_manager is not None,
                'timezone_manager_available': self.timezone_manager is not None
            }
        }
    
    def update_configuration(self, **kwargs):
        """Update scanner controller configuration"""
        updated_items = []
        
        if 'epic_list' in kwargs:
            self.epic_list = kwargs['epic_list']
            updated_items.append(f"Epic list: {len(self.epic_list)} pairs")
        
        if 'min_confidence' in kwargs:
            self.min_confidence = kwargs['min_confidence']
            updated_items.append(f"Min confidence: {self.min_confidence:.1%}")
        
        if 'spread_pips' in kwargs:
            self.spread_pips = kwargs['spread_pips']
            updated_items.append(f"Spread: {self.spread_pips} pips")
        
        if 'use_bid_adjustment' in kwargs:
            self.use_bid_adjustment = kwargs['use_bid_adjustment']
            status = "enabled" if self.use_bid_adjustment else "disabled"
            updated_items.append(f"BID adjustment: {status}")
        
        if 'enable_multi_timeframe' in kwargs:
            self.enable_multi_timeframe = kwargs['enable_multi_timeframe']
            status = "enabled" if self.enable_multi_timeframe else "disabled"
            updated_items.append(f"Multi-timeframe: {status}")
        
        if 'default_timeframe' in kwargs:
            self.default_timeframe = kwargs['default_timeframe']
            updated_items.append(f"Default timeframe: {self.default_timeframe}")
        
        if updated_items:
            self.logger.info(f"📝 ScannerController configuration updated:")
            for item in updated_items:
                self.logger.info(f"   {item}")
        
        return updated_items
    
    def reset_scanning_state(self):
        """Reset scanning state (useful for testing)"""
        self.last_signals.clear()
        self.scan_count = 0
        self.total_signals_detected = 0
        self.total_signals_filtered = 0
        self.logger.info("🔄 Scanning state reset")
    
    def test_epic_scan(self, epic: str) -> Dict:
        """Test scanning for a specific epic"""
        try:
            self.logger.info(f"🧪 Testing scan for {epic}")
            
            start_time = datetime.now()
            signal = self._scan_single_epic(epic)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'epic': epic,
                'success': signal is not None,
                'signal': signal,
                'scan_duration_seconds': duration,
                'timestamp': end_time.isoformat()
            }
            
            if signal:
                self.logger.info(f"✅ Test scan successful for {epic}: {signal['signal_type']}")
            else:
                self.logger.info(f"📊 Test scan completed for {epic}: no signal")
            
            return result
            
        except Exception as e:
            error_result = {
                'epic': epic,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.error(f"❌ Test scan failed for {epic}: {e}")
            return error_result
    
    def get_epic_performance(self) -> Dict:
        """Get performance statistics by epic"""
        try:
            epic_stats = {}
            
            for epic in self.epic_list:
                last_signal_time = self.last_signals.get(epic)
                
                epic_stats[epic] = {
                    'has_recent_signal': last_signal_time is not None,
                    'last_signal_time': last_signal_time,
                    'pair_info': getattr(config, 'PAIR_INFO', {}).get(epic, {})
                }
            
            return {
                'total_epics': len(self.epic_list),
                'epics_with_signals': len([e for e in epic_stats.values() if e['has_recent_signal']]),
                'epic_details': epic_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': f'Failed to get epic performance: {e}',
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate that all required dependencies are available"""
        issues = []
        
        if not self.signal_detector:
            issues.append("Signal detector not available")
        
        if not self.epic_list:
            issues.append("No epics configured for scanning")
        
        if self.min_confidence < 0 or self.min_confidence > 1:
            issues.append(f"Invalid min_confidence: {self.min_confidence}")
        
        if self.spread_pips < 0:
            issues.append(f"Invalid spread_pips: {self.spread_pips}")
        
        # Test signal detector if available
        if self.signal_detector:
            try:
                # Try to call a method to verify it's working
                hasattr(self.signal_detector, 'detect_signals_mid_prices')
            except Exception as e:
                issues.append(f"Signal detector validation failed: {e}")
        
        return len(issues) == 0, issues