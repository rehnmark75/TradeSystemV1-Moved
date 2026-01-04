# core/scanner.py
"""
IntelligentForexScanner with Enhanced Features
Maintains backward compatibility while adding new capabilities
UPDATED: Integrated SignalProcessor for Smart Money analysis

CRITICAL: Database-driven configuration - NO FALLBACK to config.py
All settings must come from scanner_global_config table.
"""

import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import pandas as pd

try:
    import config
    from core.signal_detector import SignalDetector
    from core.database import DatabaseManager
except ImportError:
    # Fallback for validation system and other imports
    from forex_scanner import config
    from forex_scanner.core.signal_detector import SignalDetector
    from forex_scanner.core.database import DatabaseManager

# Optional enhancement imports
try:
    from core.alert_deduplication import AlertDeduplicationManager
    DEDUP_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.core.alert_deduplication import AlertDeduplicationManager
        DEDUP_AVAILABLE = True
    except ImportError:
        DEDUP_AVAILABLE = False
        AlertDeduplicationManager = None

try:
    from core.smart_money_integration import add_smart_money_to_signals
    SMART_MONEY_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.core.smart_money_integration import add_smart_money_to_signals
        SMART_MONEY_AVAILABLE = True
    except ImportError:
        SMART_MONEY_AVAILABLE = False
        add_smart_money_to_signals = None

# ADD: Import SignalProcessor for Smart Money integration
try:
    from core.processing.signal_processor import SignalProcessor
    SIGNAL_PROCESSOR_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.core.processing.signal_processor import SignalProcessor
        SIGNAL_PROCESSOR_AVAILABLE = True
    except ImportError:
        SIGNAL_PROCESSOR_AVAILABLE = False
        SignalProcessor = None

# ADD: Import Market Intelligence components for comprehensive market analysis
try:
    from core.intelligence.market_intelligence import MarketIntelligenceEngine
    from core.intelligence.market_intelligence_history_manager import MarketIntelligenceHistoryManager
    MARKET_INTELLIGENCE_AVAILABLE = True
except ImportError:
    try:
        from forex_scanner.core.intelligence.market_intelligence import MarketIntelligenceEngine
        from forex_scanner.core.intelligence.market_intelligence_history_manager import MarketIntelligenceHistoryManager
        MARKET_INTELLIGENCE_AVAILABLE = True
    except ImportError:
        MARKET_INTELLIGENCE_AVAILABLE = False
        MarketIntelligenceEngine = None
        MarketIntelligenceHistoryManager = None

# Import config services for database-driven settings
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
    CONFIG_SERVICES_AVAILABLE = True
except ImportError:
    try:
        from services.scanner_config_service import get_scanner_config
        from services.smc_simple_config_service import get_smc_simple_config
        CONFIG_SERVICES_AVAILABLE = True
    except ImportError:
        CONFIG_SERVICES_AVAILABLE = False

# Import intelligence config service for cleanup settings
try:
    from forex_scanner.services.intelligence_config_service import get_intelligence_config
    INTELLIGENCE_CONFIG_SERVICE_AVAILABLE = True
except ImportError:
    try:
        from services.intelligence_config_service import get_intelligence_config
        INTELLIGENCE_CONFIG_SERVICE_AVAILABLE = True
    except ImportError:
        INTELLIGENCE_CONFIG_SERVICE_AVAILABLE = False

# Import market hours utility to skip scanning when forex market is closed
try:
    from forex_scanner.utils.timezone_utils import is_market_hours
    MARKET_HOURS_AVAILABLE = True
except ImportError:
    try:
        from utils.timezone_utils import is_market_hours
        MARKET_HOURS_AVAILABLE = True
    except ImportError:
        MARKET_HOURS_AVAILABLE = False
        is_market_hours = None


class IntelligentForexScanner:
    """
    Enhanced forex scanner with optional deduplication and smart money analysis
    Maintains backward compatibility with existing system
    UPDATED: Now uses SignalProcessor for Smart Money analysis
    """
    
    def __init__(self, 
                 db_manager=None,
                 epic_list: List[str] = None,
                 min_confidence: float = None,
                 scan_interval: int = 60,
                 use_bid_adjustment: bool = None,
                 spread_pips: float = None,
                 user_timezone: str = 'Europe/Stockholm',
                 intelligence_mode: str = 'backtest_consistent',
                 **kwargs):
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # ✅ CRITICAL: Database-driven configuration - NO FALLBACK to config.py
        if not CONFIG_SERVICES_AVAILABLE:
            raise RuntimeError(
                "❌ CRITICAL: Scanner config service not available - database is REQUIRED, no fallback allowed"
            )

        try:
            self._scanner_cfg = get_scanner_config()
            self._smc_cfg = get_smc_simple_config()
        except Exception as e:
            raise RuntimeError(
                f"❌ CRITICAL: Failed to load config from database: {e} - no fallback allowed"
            )

        if not self._scanner_cfg or not self._smc_cfg:
            raise RuntimeError(
                "❌ CRITICAL: Config returned None - database is REQUIRED, no fallback allowed"
            )

        # Core configuration from database - NO FALLBACK
        self.db_manager = db_manager
        self.epic_list = epic_list or self._smc_cfg.enabled_pairs
        self.min_confidence = min_confidence if min_confidence is not None else self._scanner_cfg.min_confidence
        self.scan_interval = scan_interval
        self.use_bid_adjustment = use_bid_adjustment if use_bid_adjustment is not None else False
        self.spread_pips = spread_pips if spread_pips is not None else 1.5
        self.user_timezone = user_timezone
        self.intelligence_mode = intelligence_mode

        self.logger.info("[CONFIG:DB] ✅ Scanner config loaded from database (NO FALLBACK)")
        
        # Initialize signal detector
        self.signal_detector = self._initialize_signal_detector(db_manager, user_timezone)

        # Initialize deduplication manager FIRST (shared with SignalProcessor)
        # Get dedup setting from database - NO FALLBACK
        self.deduplication_manager = None
        self.enable_deduplication = self._scanner_cfg.enable_alert_deduplication and DEDUP_AVAILABLE

        if self.enable_deduplication and db_manager and DEDUP_AVAILABLE:
            try:
                self.deduplication_manager = AlertDeduplicationManager(db_manager)
                self.logger.info("🛡️ Deduplication manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize deduplication: {e}")
                self.enable_deduplication = False

        # Initialize SignalProcessor for Smart Money analysis (pass shared dedup manager)
        self.signal_processor = None
        self.use_signal_processor = True and SIGNAL_PROCESSOR_AVAILABLE  # Always use signal processor

        if self.use_signal_processor and SIGNAL_PROCESSOR_AVAILABLE:
            try:
                # Get data_fetcher from signal_detector if available
                data_fetcher = getattr(self.signal_detector, 'data_fetcher', None)

                # If no data_fetcher from signal_detector, create one
                if not data_fetcher and db_manager:
                    try:
                        from core.data_fetcher import DataFetcher
                        data_fetcher = DataFetcher(db_manager)
                    except ImportError:
                        try:
                            from forex_scanner.core.data_fetcher import DataFetcher
                            data_fetcher = DataFetcher(db_manager)
                        except ImportError:
                            self.logger.warning("DataFetcher not available for SignalProcessor")

                # Initialize SignalProcessor with shared deduplication manager
                self.signal_processor = SignalProcessor(
                    db_manager=db_manager,
                    data_fetcher=data_fetcher,  # CRITICAL for Smart Money!
                    alert_history=getattr(self, 'alert_history', None),
                    claude_analyzer=getattr(self, 'claude_analyzer', None),
                    notification_manager=getattr(self, 'notification_manager', None),
                    deduplication_manager=self.deduplication_manager  # Pass shared instance
                )

                self.logger.info("📊 SignalProcessor initialized")
                self.logger.info(f"   Smart Money: {'✅' if self.signal_processor.smart_money_analyzer else '❌'}")
                self.logger.info(f"   Data Fetcher: {'✅' if data_fetcher else '❌'}")

            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize SignalProcessor: {e}")
                self.use_signal_processor = False
        
        # Initialize optional smart money (keep for backward compatibility)
        # Get setting from database - NO FALLBACK
        self.enable_smart_money = self._scanner_cfg.smart_money_readonly_enabled and SMART_MONEY_AVAILABLE
        if self.enable_smart_money:
            self.logger.info("✅ Smart money analysis enabled")

        # ADD: Initialize Market Intelligence components for comprehensive market analysis
        self.market_intelligence_engine = None
        self.market_intelligence_history = None
        # Market intelligence is always enabled if available (no config toggle needed)
        self.enable_market_intelligence = MARKET_INTELLIGENCE_AVAILABLE

        if self.enable_market_intelligence and db_manager and MARKET_INTELLIGENCE_AVAILABLE:
            try:
                # Get data_fetcher from signal_detector if available
                data_fetcher = getattr(self.signal_detector, 'data_fetcher', None)

                # If no data_fetcher from signal_detector, create one
                if not data_fetcher and db_manager:
                    try:
                        from core.data_fetcher import DataFetcher
                        data_fetcher = DataFetcher(db_manager)
                    except ImportError:
                        try:
                            from forex_scanner.core.data_fetcher import DataFetcher
                            data_fetcher = DataFetcher(db_manager)
                        except ImportError:
                            data_fetcher = None

                if data_fetcher:
                    # Initialize Market Intelligence Engine
                    self.market_intelligence_engine = MarketIntelligenceEngine(data_fetcher)

                    # Initialize Market Intelligence History Manager
                    self.market_intelligence_history = MarketIntelligenceHistoryManager(db_manager)

                    self.logger.info("🧠 Market Intelligence components initialized")
                    self.logger.info(f"   Engine: {'✅' if self.market_intelligence_engine else '❌'}")
                    self.logger.info(f"   History: {'✅' if self.market_intelligence_history else '❌'}")
                else:
                    self.logger.warning("⚠️ DataFetcher not available for Market Intelligence")
                    self.enable_market_intelligence = False

            except Exception as e:
                self.logger.warning(f"⚠️ Could not initialize Market Intelligence: {e}")
                self.enable_market_intelligence = False

        # Scanner state
        self.running = False
        self.last_signals = {}
        
        # Statistics (updated with new counters)
        self.stats = {
            'scans_completed': 0,
            'signals_detected': 0,
            'signals_processed': 0,
            'signals_filtered_confidence': 0,
            'signals_filtered_dedup': 0,
            'errors': 0,
            'timestamp_conversions': 0,
            'smart_money_enhanced': 0,
            'signal_processor_used': 0,  # ADD: Track SignalProcessor usage
            'smart_money_validated': 0,  # ADD: Track validated signals
            'market_intelligence_generated': 0,  # ADD: Track market intelligence generation
            'market_intelligence_stored': 0,     # ADD: Track successful storage
            'market_intelligence_errors': 0     # ADD: Track storage errors
        }

        self.logger.info(f"🔍 IntelligentForexScanner initialized")
        self.logger.info(f"   Epics: {len(self.epic_list)}")
        self.logger.info(f"   Min confidence: {self.min_confidence:.1%}")
        self.logger.info(f"   Deduplication: {'✅' if self.enable_deduplication else '❌'}")
        self.logger.info(f"   Smart money: {'✅' if self.enable_smart_money else '❌'}")
        self.logger.info(f"   SignalProcessor: {'✅' if self.use_signal_processor else '❌'}")
        self.logger.info(f"   Market Intelligence: {'✅' if self.enable_market_intelligence else '❌'}")

    def _initialize_signal_detector(self, db_manager, user_timezone):
        """Initialize signal detector with fallback"""
        try:
            if db_manager:
                return SignalDetector(db_manager, user_timezone)
            else:
                # Try to create with temporary db manager
                # NOTE: DATABASE_URL is an environment variable that stays in config.py
                try:
                    from core.database import DatabaseManager
                except ImportError:
                    from forex_scanner.core.database import DatabaseManager
                temp_db = DatabaseManager(getattr(config, 'DATABASE_URL', ''))  # ENV VAR - stays in config
                return SignalDetector(temp_db, user_timezone)
        except Exception as e:
            self.logger.warning(f"⚠️ Signal detector init warning: {e}")
            # Return basic signal detector without db
            return SignalDetector(None, user_timezone)
    
    def _convert_timestamp_safe(self, timestamp_value) -> Optional[datetime]:
        """Safely convert various timestamp formats"""
        if timestamp_value is None:
            return None
            
        try:
            self.stats['timestamp_conversions'] += 1
            
            # Already datetime
            if isinstance(timestamp_value, datetime):
                if timestamp_value.tzinfo is None:
                    return timestamp_value.replace(tzinfo=timezone.utc)
                return timestamp_value
            
            # String timestamp
            if isinstance(timestamp_value, str):
                if 'T' in timestamp_value:
                    return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                return datetime.fromisoformat(timestamp_value).replace(tzinfo=timezone.utc)
            
            # Unix timestamp (validate range)
            if isinstance(timestamp_value, (int, float)):
                # Valid range: 2020-2030
                if 1577836800 <= timestamp_value <= 1893456000:
                    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
                # Invalid - use current time
                return datetime.now(tz=timezone.utc)
            
            # Pandas timestamp
            if hasattr(timestamp_value, 'to_pydatetime'):
                dt = timestamp_value.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            
            # Unknown - use current time
            return datetime.now(tz=timezone.utc)
            
        except Exception:
            return datetime.now(tz=timezone.utc)
    
    def _scan_single_epic(self, epic: str, enable_multi_timeframe: bool = False) -> Optional[Dict]:
        """Scan single epic for signals"""
        try:
            # Get pair info - extract pair name from epic
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

            # Get default timeframe from database - NO FALLBACK
            default_tf = self._scanner_cfg.default_timeframe

            # Detect signals
            if self.use_bid_adjustment:
                signal = self.signal_detector.detect_signals_bid_adjusted(
                    epic, pair_name, self.spread_pips, default_tf
                )
            else:
                signal = self.signal_detector.detect_signals_mid_prices(
                    epic, pair_name, default_tf
                )
            
            if signal:
                # Fix timestamps
                for field in ['market_timestamp', 'timestamp']:
                    if field in signal:
                        signal[field] = self._convert_timestamp_safe(signal[field])
                
                # Return first signal if list
                if isinstance(signal, list):
                    return signal[0] if signal else None
                return signal
                
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning {epic}: {e}")
            self.stats['errors'] += 1
            return None
    
    def scan_once(self, scan_type: str = 'live') -> List[Dict]:
        """
        Perform one complete scan of all epics
        UPDATED: Now processes signals through SignalProcessor for Smart Money analysis
        UPDATED: Skips scanning when forex market is closed (weekends)
        """
        # Check if forex market is open before scanning
        if MARKET_HOURS_AVAILABLE and is_market_hours is not None:
            if not is_market_hours():
                # Market is closed - skip all scanning, intelligence, and performance data
                if not hasattr(self, '_last_market_closed_log') or \
                   (datetime.now() - self._last_market_closed_log).total_seconds() > 3600:
                    self.logger.info("🌙 Forex market is closed - skipping scan (no signals, intelligence, or performance data will be generated)")
                    self._last_market_closed_log = datetime.now()
                return []

        scan_start = datetime.now()
        self.stats['scans_completed'] += 1

        try:
            self.logger.info(f"🔍 Starting scan #{self.stats['scans_completed']}")
            
            # Step 1: Detect raw signals
            raw_signals = []
            for epic in self.epic_list:
                signals = self._detect_signals_for_epic(epic)
                if signals:
                    if isinstance(signals, list):
                        raw_signals.extend(signals)
                    else:
                        raw_signals.append(signals)
            
            if not raw_signals:
                self.logger.info("✓ No signals detected")
                # Still capture market intelligence even with no signals
                self._capture_scan_market_intelligence(scan_start, [])
                return []
            
            self.logger.info(f"📊 {len(raw_signals)} raw signals detected")
            self.stats['signals_detected'] += len(raw_signals)
            
            # Step 2: Process through SignalProcessor if available (NEW!)
            processed_signals = []
            
            if self.use_signal_processor and self.signal_processor:
                self.logger.debug("📊 Processing signals through SignalProcessor...")
                
                for signal in raw_signals:
                    try:
                        # Process signal through SignalProcessor (includes Smart Money analysis)
                        processed = self.signal_processor.process_signal(signal)
                        
                        if processed:
                            self.stats['signal_processor_used'] += 1
                            
                            # Check if smart money was applied
                            processing_result = processed.get('processing_result', {})
                            if processing_result.get('smart_money_analyzed'):
                                self.stats['smart_money_enhanced'] += 1
                                self.logger.info(f"🧠 Smart Money applied to {processed.get('epic')}")
                                self.logger.debug(f"   Score: {processed.get('smart_money_score', 0):.3f}")
                                self.logger.debug(f"   Type: {processed.get('smart_money_type', 'Unknown')}")
                                
                                if processed.get('smart_money_validated'):
                                    self.stats['smart_money_validated'] += 1
                            
                            # Add to processed signals if not filtered out
                            if not processing_result.get('strategy_filtered') and not processing_result.get('duplicate_filtered'):
                                processed_signals.append(processed)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing signal through SignalProcessor: {e}")
                        # Add original signal if processing fails
                        processed_signals.append(signal)
                
                self.logger.info(f"📊 {len(processed_signals)} signals after SignalProcessor")
            else:
                # Fallback to original processing if SignalProcessor not available
                processed_signals = raw_signals
                self.logger.debug("⚠️ SignalProcessor not available, using original processing")
            
            # Step 3: Filter by confidence (for signals not processed by SignalProcessor)
            filtered_signals = []
            for signal in processed_signals:
                # Skip if already processed by SignalProcessor (it handles confidence filtering)
                if signal.get('processing_result'):
                    filtered_signals.append(signal)
                elif signal.get('confidence_score', 0) >= self.min_confidence:
                    filtered_signals.append(signal)
                else:
                    self.stats['signals_filtered_confidence'] += 1
            
            # Step 4: Apply deduplication if available and not already done by SignalProcessor
            if self.enable_deduplication and self.deduplication_manager:
                deduplicated = []
                for signal in filtered_signals:
                    # Skip if already deduplicated by SignalProcessor
                    if signal.get('processing_result', {}).get('deduplication_checked'):
                        deduplicated.append(signal)
                    else:
                        allow, reason, metadata = self.deduplication_manager.should_allow_alert(signal)
                        if allow:
                            signal.update(metadata)
                            deduplicated.append(signal)
                        else:
                            self.stats['signals_filtered_dedup'] += 1
                filtered_signals = deduplicated
            
            # Step 5: Apply smart money if available and not already done by SignalProcessor
            # (This is for backward compatibility - SignalProcessor handles this better)
            if self.enable_smart_money and add_smart_money_to_signals and not self.use_signal_processor:
                try:
                    filtered_signals = add_smart_money_to_signals(
                        filtered_signals, 
                        self.signal_detector.data_fetcher,
                        self.db_manager
                    )
                    self.stats['smart_money_enhanced'] += len(filtered_signals)
                except Exception as e:
                    self.logger.warning(f"⚠️ Smart money enhancement failed: {e}")
            
            # Step 6: Prepare signals for output
            clean_signals = []
            for signal in filtered_signals:
                clean_signal = self._prepare_signal(signal)
                clean_signals.append(clean_signal)
                self.stats['signals_processed'] += 1
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            
            if clean_signals:
                self.logger.info(f"✅ Scan completed in {scan_duration:.2f}s: {len(clean_signals)} signals ready")
                
                # Log signal details with Smart Money info
                for signal in clean_signals:
                    epic = signal.get('epic', 'Unknown')
                    signal_type = signal.get('signal_type', 'Unknown')
                    confidence = signal.get('confidence_score', 0)
                    
                    # Check for smart money validation
                    if signal.get('smart_money_validated'):
                        sm_score = signal.get('smart_money_score', 0)
                        sm_type = signal.get('smart_money_type', 'Unknown')
                        self.logger.info(f"   📊 {epic} {signal_type} ({confidence:.1%}) 🧠 SM: {sm_type} ({sm_score:.2f})")
                    else:
                        self.logger.info(f"   📊 {epic} {signal_type} ({confidence:.1%})")
                
                # Log Smart Money statistics if any
                if self.stats['smart_money_validated'] > 0:
                    self.logger.info(f"🧠 Smart Money validated: {self.stats['smart_money_validated']}/{len(clean_signals)} signals")

                # NEW: Log Market Intelligence summary for analysis
                self._log_market_intelligence_summary(clean_signals)

            # ADD: Generate and store market intelligence for this scan cycle
            self._capture_scan_market_intelligence(scan_start, clean_signals)

            # Periodic cleanup of old intelligence records (once per ~24 hours)
            # At ~600 scans/day (every 2.5 minutes), check every 500 scans
            self._maybe_cleanup_old_intelligence_records()

            return clean_signals

        except Exception as e:
            self.logger.error(f"❌ Scan error: {e}")
            self.stats['errors'] += 1
            return []
    
    def _detect_signals_for_epic(self, epic: str) -> Optional[Dict]:
        """Detect signals using all strategies for an epic"""
        try:
            pair_name = epic.replace('CS.D.', '').replace('.MINI.IP', '').replace('.CEEM.IP', '')

            # Get default timeframe from database - NO FALLBACK
            default_tf = self._scanner_cfg.default_timeframe

            # Use detect_signals_all_strategies if available
            if hasattr(self.signal_detector, 'detect_signals_all_strategies'):
                signals = self.signal_detector.detect_signals_all_strategies(
                    epic, pair_name, self.spread_pips, default_tf
                )
            else:
                # Fallback to single strategy
                if self.use_bid_adjustment:
                    signals = self.signal_detector.detect_signals_bid_adjusted(
                        epic, pair_name, self.spread_pips, default_tf
                    )
                else:
                    signals = self.signal_detector.detect_signals_mid_prices(
                        epic, pair_name, default_tf
                    )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting signals for {epic}: {e}")
            return None
    
    def _prepare_signal(self, signal: Dict) -> Dict:
        """Prepare signal for processing"""
        clean_signal = signal.copy()
        
        # Ensure required fields
        clean_signal['scanner_timestamp'] = datetime.now().isoformat()
        clean_signal['scanner_version'] = 'enhanced_v1.3_with_signal_processor'  # Updated version
        clean_signal['scanner_validated'] = True
        
        # Ensure timestamps are safe
        for field in ['market_timestamp', 'timestamp']:
            if field in clean_signal:
                clean_signal[field] = self._convert_timestamp_safe(clean_signal[field])
        
        # Add processing flags (updated to show SignalProcessor usage)
        clean_signal['processing_pipeline'] = {
            'raw_detection': True,
            'confidence_filtered': True,
            'dedup_filtered': self.enable_deduplication,
            'smart_money_enhanced': self.enable_smart_money or bool(signal.get('smart_money_validated')),
            'signal_processor_used': self.use_signal_processor,
            'ready_for_execution': True
        }
        
        return clean_signal

    def _log_market_intelligence_summary(self, signals: List[Dict]) -> None:
        """
        📊 Log market intelligence summary for analysis
        Shows market conditions captured during this scan
        """
        try:
            if not signals:
                return

            # Collect market intelligence data from signals
            regimes = []
            sessions = []
            volatility_levels = []
            intelligence_sources = []
            strategies_with_intelligence = set()

            for signal in signals:
                epic = signal.get('epic', 'Unknown')
                strategy = signal.get('strategy', 'unknown')

                # Check if signal has market intelligence
                market_intelligence = signal.get('market_intelligence', {})
                if market_intelligence:
                    strategies_with_intelligence.add(strategy)

                    # Collect regime data
                    regime_analysis = market_intelligence.get('regime_analysis', {})
                    if regime_analysis.get('dominant_regime'):
                        regimes.append({
                            'epic': epic,
                            'regime': regime_analysis.get('dominant_regime'),
                            'confidence': regime_analysis.get('confidence', 0.5)
                        })

                    # Collect session data
                    session_analysis = market_intelligence.get('session_analysis', {})
                    if session_analysis.get('current_session'):
                        sessions.append(session_analysis.get('current_session'))

                    # Collect volatility data
                    volatility = market_intelligence.get('volatility_level')
                    if volatility:
                        volatility_levels.append(volatility)

                    # Collect intelligence source
                    source = market_intelligence.get('intelligence_source')
                    if source:
                        intelligence_sources.append(source)

            # Log summary if any intelligence was captured
            if regimes or sessions or volatility_levels:
                self.logger.info("📊 Market Intelligence Summary:")

                # Log strategies with intelligence
                if strategies_with_intelligence:
                    self.logger.info(f"   🧠 Strategies with intelligence: {', '.join(sorted(strategies_with_intelligence))}")

                # Log regime distribution
                if regimes:
                    regime_counts = {}
                    total_confidence = 0
                    for r in regimes:
                        regime = r['regime']
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1
                        total_confidence += r['confidence']

                    avg_confidence = total_confidence / len(regimes) if regimes else 0
                    regime_summary = ', '.join([f"{regime}({count})" for regime, count in regime_counts.items()])
                    self.logger.info(f"   📈 Market Regimes: {regime_summary} | Avg Confidence: {avg_confidence:.1%}")

                # Log session distribution
                if sessions:
                    session_counts = {}
                    for session in sessions:
                        session_counts[session] = session_counts.get(session, 0) + 1
                    session_summary = ', '.join([f"{session}({count})" for session, count in session_counts.items()])
                    self.logger.info(f"   🕐 Trading Sessions: {session_summary}")

                # Log volatility distribution
                if volatility_levels:
                    vol_counts = {}
                    for vol in volatility_levels:
                        vol_counts[vol] = vol_counts.get(vol, 0) + 1
                    vol_summary = ', '.join([f"{vol}({count})" for vol, count in vol_counts.items()])
                    self.logger.info(f"   📊 Volatility Levels: {vol_summary}")

                # Log intelligence sources
                if intelligence_sources:
                    source_counts = {}
                    for source in intelligence_sources:
                        source_type = 'Strategy' if 'MarketIntelligenceEngine' in source else 'Universal'
                        source_counts[source_type] = source_counts.get(source_type, 0) + 1
                    source_summary = ', '.join([f"{source}({count})" for source, count in source_counts.items()])
                    self.logger.info(f"   🔍 Intelligence Sources: {source_summary}")

                # Log specific regime details for analysis
                if regimes:
                    self.logger.info("   📋 Regime Details:")
                    for r in regimes:
                        self.logger.info(f"      {r['epic']}: {r['regime']} ({r['confidence']:.1%})")

            else:
                self.logger.info("📊 Market Intelligence: No intelligence data captured (engine may be disabled)")

        except Exception as e:
            self.logger.warning(f"⚠️ Error logging market intelligence summary: {e}")

    def _capture_scan_market_intelligence(self, scan_start: datetime, signals: List[Dict]) -> None:
        """
        🧠 Generate and store comprehensive market intelligence for this scan cycle
        This captures market conditions regardless of whether signals were detected
        """
        if not self.enable_market_intelligence or not self.market_intelligence_engine or not self.market_intelligence_history:
            return

        try:
            scan_duration = (datetime.now() - scan_start).total_seconds()
            self.logger.debug(f"🧠 Generating market intelligence for scan cycle...")

            # Generate comprehensive market intelligence report
            intelligence_report = self.market_intelligence_engine.generate_market_intelligence_report(self.epic_list)

            if intelligence_report:
                self.stats['market_intelligence_generated'] += 1

                # Generate unique scan cycle ID
                scan_cycle_id = f"scan_{scan_start.strftime('%Y%m%d_%H%M%S')}_{len(signals)}signals"

                # Store market intelligence in dedicated table
                record_id = self.market_intelligence_history.save_market_intelligence(
                    intelligence_report=intelligence_report,
                    epic_list=self.epic_list,
                    scan_cycle_id=scan_cycle_id
                )

                if record_id:
                    self.stats['market_intelligence_stored'] += 1

                    # Extract key information for logging
                    market_regime = intelligence_report.get('market_regime', {})
                    dominant_regime = market_regime.get('dominant_regime', 'unknown')
                    confidence = market_regime.get('confidence', 0.5)

                    session_analysis = intelligence_report.get('session_analysis', {})
                    current_session = session_analysis.get('current_session', 'unknown')

                    self.logger.debug(f"Market Intelligence stored: {dominant_regime} regime ({confidence:.1%}) "
                                    f"during {current_session} session - Record #{record_id}")

                    # Log additional insights if confidence is high
                    if confidence > 0.8:
                        market_strength = market_regime.get('market_strength', {})
                        market_bias = market_strength.get('market_bias', 'neutral')
                        self.logger.debug(f"High confidence analysis: Market bias = {market_bias}")

                else:
                    self.stats['market_intelligence_errors'] += 1
                    self.logger.warning("⚠️ Failed to store market intelligence data")

            else:
                self.stats['market_intelligence_errors'] += 1
                self.logger.warning("⚠️ Failed to generate market intelligence report")

        except Exception as e:
            self.stats['market_intelligence_errors'] += 1
            self.logger.error(f"❌ Error capturing market intelligence: {e}")
            import traceback
            self.logger.debug(f"   Traceback: {traceback.format_exc()}")

    def _maybe_cleanup_old_intelligence_records(self):
        """
        Periodically cleanup old intelligence records to prevent database bloat.

        Runs once every ~500 scans (approximately once per day at 2.5 minute intervals).
        Uses retention days from database config (default: 60 days).
        """
        # Only run every 500 scans (~24 hours at 2.5 min intervals)
        if self.stats['scans_completed'] % 500 != 0:
            return

        # Skip if no market intelligence history manager
        if not self.market_intelligence_history:
            return

        try:
            # Get retention days from database config
            retention_days = 60  # Default
            if INTELLIGENCE_CONFIG_SERVICE_AVAILABLE:
                try:
                    intel_config = get_intelligence_config()
                    retention_days = intel_config.intelligence_cleanup_days
                except Exception:
                    pass

            # Run cleanup
            deleted_count = self.market_intelligence_history.cleanup_old_records(
                days_to_keep=retention_days
            )

            if deleted_count and deleted_count > 0:
                self.logger.info(
                    f"Cleaned {deleted_count} old intelligence records "
                    f"(keeping {retention_days} days)"
                )

        except Exception as e:
            self.logger.warning(f"Intelligence cleanup failed: {e}")

    def _calculate_boundary_aligned_sleep(self, scan_duration: float) -> float:
        """
        Calculate sleep time to align with 15m candle boundaries.

        When SCAN_ALIGN_TO_BOUNDARIES is enabled, scans are timed to occur
        shortly after 15m candle closes (:00, :15, :30, :45 + offset).

        Returns:
            float: Seconds to sleep before next scan
        """
        from datetime import datetime, timedelta

        # Get offset from database config - NO FALLBACK
        offset_seconds = self._scanner_cfg.scan_boundary_offset_seconds

        now = datetime.utcnow()
        current_minute = now.minute
        current_second = now.second

        # Find next 15m boundary (0, 15, 30, 45)
        next_boundary_minute = ((current_minute // 15) + 1) * 15 % 60

        # Calculate time to next boundary
        if next_boundary_minute == 0 and current_minute >= 45:
            # Boundary is at next hour
            next_boundary = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_boundary = now.replace(minute=next_boundary_minute, second=0, microsecond=0)
            if next_boundary < now:
                next_boundary += timedelta(hours=1)

        # Add offset (scan after boundary to allow data to settle)
        target_scan_time = next_boundary + timedelta(seconds=offset_seconds)

        # Calculate sleep time
        sleep_seconds = (target_scan_time - now).total_seconds()

        # If we're already past the target time for this boundary, calculate for next
        if sleep_seconds < 0:
            sleep_seconds += 15 * 60  # Add 15 minutes

        # Log boundary alignment info
        self.logger.debug(f"⏰ Boundary scan: next at {target_scan_time.strftime('%H:%M:%S')} UTC (sleeping {sleep_seconds:.0f}s)")

        return max(10, sleep_seconds)  # Minimum 10 seconds between scans

    def start_continuous_scanning(self):
        """Start continuous scanning"""
        self.running = True

        # Check if boundary-aligned scanning is enabled from DATABASE (not config file)
        scanner_config = get_scanner_config()
        boundary_aligned = scanner_config.scan_align_to_boundaries
        boundary_offset = scanner_config.scan_boundary_offset_seconds
        use_1m_base = scanner_config.use_1m_base_synthesis

        self.logger.info(f"🚀 Starting continuous scanning")
        self.logger.info(f"   Interval: {self.scan_interval}s")
        self.logger.info(f"   Epics: {len(self.epic_list)}")
        self.logger.info(f"   SignalProcessor: {'✅ Active' if self.use_signal_processor else '❌ Inactive'}")
        self.logger.info(f"   1m Base Synthesis: {'✅ Enabled' if use_1m_base else '❌ Disabled (5m base)'}")
        self.logger.info(f"   Boundary Scanning: {'✅ Enabled (offset: ' + str(boundary_offset) + 's)' if boundary_aligned else '❌ Disabled'}")

        try:
            while self.running:
                scan_start = time.time()

                # Check if market is open - if closed, use longer sleep interval
                market_open = True
                if MARKET_HOURS_AVAILABLE and is_market_hours is not None:
                    market_open = is_market_hours()

                # Perform scan (will return empty if market closed)
                signals = self.scan_once('live')

                if signals:
                    self.logger.info(f"📤 {len(signals)} signals detected")

                    # Log Smart Money summary if available
                    sm_validated = sum(1 for s in signals if s.get('smart_money_validated'))
                    if sm_validated > 0:
                        self.logger.info(f"🧠 Smart Money validated: {sm_validated}/{len(signals)}")

                # Calculate sleep time based on mode
                scan_duration = time.time() - scan_start

                if not market_open:
                    # Market is closed - use longer sleep interval (5 minutes)
                    # to reduce resource usage during weekends
                    sleep_time = 300  # 5 minutes
                elif boundary_aligned:
                    # Boundary-aligned scanning: sync with 15m candle closes
                    sleep_time = self._calculate_boundary_aligned_sleep(scan_duration)
                else:
                    # Standard fixed interval scanning
                    sleep_time = max(0, self.scan_interval - scan_duration)

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("🛑 Scanning stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Scanning error: {e}")
        finally:
            self.running = False
    
    def start_continuous_scan(self):
        """Legacy method name for compatibility"""
        self.start_continuous_scanning()
    
    def stop_scanning(self):
        """Stop scanning"""
        self.running = False
        self.logger.info("🛑 Scanner stop requested")
    
    def stop(self):
        """Legacy stop method"""
        self.stop_scanning()
    
    def get_scanner_status(self) -> Dict:
        """Get scanner status (updated with SignalProcessor info)"""
        return {
            'running': self.running,
            'epic_count': len(self.epic_list),
            'scan_interval': self.scan_interval,
            'min_confidence': self.min_confidence,
            'use_bid_adjustment': self.use_bid_adjustment,
            'spread_pips': self.spread_pips,
            'user_timezone': self.user_timezone,
            'intelligence_mode': self.intelligence_mode,
            'deduplication_enabled': self.enable_deduplication,
            'smart_money_enabled': self.enable_smart_money,
            'signal_processor_enabled': self.use_signal_processor,  # ADD
            'signal_processor_available': self.signal_processor is not None,  # ADD
            'stats': self.stats.copy(),
            'last_scan_time': datetime.now().isoformat(),
            'architecture_version': 'enhanced_v1.3_with_signal_processor'  # Updated
        }
    
    def get_statistics(self) -> Dict:
        """Get scanner statistics (updated with SignalProcessor stats)"""
        scans = max(1, self.stats['scans_completed'])
        detected = self.stats['signals_detected']
        processed = self.stats['signals_processed']
        
        return {
            'scans_completed': scans,
            'signals_detected': detected,
            'signals_processed': processed,
            'signals_filtered_confidence': self.stats['signals_filtered_confidence'],
            'signals_filtered_dedup': self.stats['signals_filtered_dedup'],
            'signal_processor_used': self.stats['signal_processor_used'],  # ADD
            'errors': self.stats['errors'],
            'timestamp_conversions': self.stats['timestamp_conversions'],
            'smart_money_enhanced': self.stats['smart_money_enhanced'],
            'smart_money_validated': self.stats['smart_money_validated'],  # ADD
            'detection_rate': detected / scans,
            'processing_success_rate': processed / max(1, detected) if detected > 0 else 0,
            'smart_money_success_rate': self.stats['smart_money_validated'] / max(1, self.stats['smart_money_enhanced']) if self.stats['smart_money_enhanced'] > 0 else 0  # ADD
        }
    
    def update_configuration(self, **kwargs):
        """Update scanner configuration"""
        config_updated = False
        
        # Update allowed parameters (added use_signal_processor)
        params = [
            'scan_interval', 'min_confidence', 'epic_list',
            'use_bid_adjustment', 'spread_pips', 'intelligence_mode',
            'enable_deduplication', 'enable_smart_money', 'use_signal_processor'
        ]
        
        for param in params:
            if param in kwargs:
                setattr(self, param, kwargs[param])
                self.logger.info(f"📝 Updated {param}: {kwargs[param]}")
                config_updated = True
        
        return config_updated
    
    def _is_new_signal(self, epic: str, signal_time) -> bool:
        """Check if signal is new"""
        if epic not in self.last_signals:
            self.last_signals[epic] = signal_time
            return True
        
        if self.last_signals[epic] != signal_time:
            self.last_signals[epic] = signal_time
            return True
        
        return False


class ForexScanner(IntelligentForexScanner):
    """
    Legacy ForexScanner class for backward compatibility
    Redirects to IntelligentForexScanner
    """
    
    def __init__(self, 
                 db_manager=None,
                 epic_list: List[str] = None,
                 scan_interval: int = 60,
                 claude_api_key: Optional[str] = None,
                 enable_claude_analysis: bool = None,
                 use_bid_adjustment: bool = True,
                 spread_pips: float = 1.5,
                 min_confidence: float = 0.7,
                 user_timezone: str = 'Europe/Stockholm',
                 **kwargs):
        
        logger = logging.getLogger(__name__)
        
        # Claude analysis setting from database - NO FALLBACK to config.py
        if enable_claude_analysis is not None:
            # Use the explicitly passed value
            self.enable_claude_analysis = enable_claude_analysis
        else:
            # Read from database - REQUIRED, NO FALLBACK
            if not CONFIG_SERVICES_AVAILABLE:
                raise RuntimeError(
                    "❌ CRITICAL: Scanner config service not available - database is REQUIRED, no fallback allowed"
                )
            try:
                scanner_cfg = get_scanner_config()
                self.enable_claude_analysis = scanner_cfg.require_claude_approval
            except Exception as e:
                raise RuntimeError(
                    f"❌ CRITICAL: Failed to load scanner config from database: {e} - no fallback allowed"
                )

        # Warn about deprecated parameters
        if claude_api_key or enable_claude_analysis:
            logger.info("📝 Claude parameters now handled by TradingOrchestrator")
        
        # Initialize with IntelligentForexScanner
        super().__init__(
            db_manager=db_manager,
            epic_list=epic_list,
            min_confidence=min_confidence,
            scan_interval=scan_interval,
            use_bid_adjustment=use_bid_adjustment,
            spread_pips=spread_pips,
            user_timezone=user_timezone,
            **kwargs
        )
        
        logger.info("✅ ForexScanner initialized (using IntelligentForexScanner)")


# Export both classes
__all__ = ['IntelligentForexScanner', 'ForexScanner']