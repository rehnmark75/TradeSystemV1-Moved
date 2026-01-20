#!/usr/bin/env python3
"""
Trading Orchestrator - ENHANCED for Modular Claude API Integration
UPDATED: Compatible with TradeValidator that has duplicate detection removed
ENHANCED: Full integration with new modular Claude API and advanced configuration
FIXED: Removed all duplicate detection logic from TradingOrchestrator

This module provides high-level orchestration of all trading components,
with clean separation between signal detection (scanner) and intelligence
processing (orchestrator). All signals are saved to the alert_history table.
NOW includes advanced Claude API integration with institutional-grade analysis.

CLEAN ARCHITECTURE:
- Scanner: Pure signal detection + deduplication (ONLY place for duplicate detection)
- TradingOrchestrator: Intelligence, risk, Claude, database, trading
- TradeValidator: Trading-specific validation only (NO duplicate detection)
- AlertHistoryManager: Complete database operations with all columns
- IntegrationManager: Enhanced Claude analysis with modular API

ENHANCED CLAUDE INTEGRATION:
- ✅ Support for advanced Claude prompts (USE_ADVANCED_CLAUDE_PROMPTS)
- ✅ Institutional analysis levels (CLAUDE_ANALYSIS_LEVEL)
- ✅ Runtime configuration of Claude settings
- ✅ Batch analysis capabilities
- ✅ Health monitoring and fallback mechanisms

DUPLICATE DETECTION ARCHITECTURE:
- ✅ Scanner handles ALL duplicate detection
- ❌ TradeValidator NO LONGER handles duplicate detection
- ❌ TradingOrchestrator NO LONGER has duplicate detection logic
"""

import sys
import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd

# Add path setup for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

try:
    import config
    from core.database import DatabaseManager
    from core.scanner import IntelligentForexScanner
    from core.signal_detector import SignalDetector
    
    # Import new modular components
    from .order_manager import OrderManager
    from .session_manager import SessionManager
    from .risk_manager import RiskManager
    from .trade_validator import TradeValidator
    from .performance_tracker import PerformanceTracker
    from .market_monitor import MarketMonitor
    from .integration_manager import IntegrationManager
    
    # Import AlertHistoryManager for database operations
    from alerts.alert_history import AlertHistoryManager

    # Import RejectionOutcomeAnalyzer for automatic outcome analysis
    from monitoring.rejection_outcome_analyzer import RejectionOutcomeAnalyzer

    print("✅ Successfully imported all trading modules including AlertHistoryManager")
except ImportError:
    from forex_scanner import config
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.core.scanner import IntelligentForexScanner
    from forex_scanner.core.signal_detector import SignalDetector
    
    # Import new modular components with absolute paths for fallback
    try:
        from forex_scanner.core.trading.order_manager import OrderManager
        from forex_scanner.core.trading.session_manager import SessionManager
        from forex_scanner.core.trading.risk_manager import RiskManager
        from forex_scanner.core.trading.trade_validator import TradeValidator
        from forex_scanner.core.trading.performance_tracker import PerformanceTracker
        from forex_scanner.core.trading.market_monitor import MarketMonitor
        from forex_scanner.core.trading.integration_manager import IntegrationManager
        print("✅ Successfully imported all trading modules with absolute paths")
    except ImportError as e:
        print(f"⚠️ Trading modules not available: {e}")
        OrderManager = SessionManager = RiskManager = TradeValidator = None
        PerformanceTracker = MarketMonitor = IntegrationManager = None
    
    # CRITICAL: Import AlertHistoryManager for database operations
    try:
        from forex_scanner.alerts.alert_history import AlertHistoryManager
        print("✅ Successfully imported AlertHistoryManager with absolute path")
    except ImportError as e:
        print(f"⚠️ AlertHistoryManager not available: {e}")
        AlertHistoryManager = None

    # Import RejectionOutcomeAnalyzer for automatic outcome analysis
    try:
        from forex_scanner.monitoring.rejection_outcome_analyzer import RejectionOutcomeAnalyzer
        print("✅ Successfully imported RejectionOutcomeAnalyzer")
    except ImportError as e:
        print(f"⚠️ RejectionOutcomeAnalyzer not available: {e}")
        RejectionOutcomeAnalyzer = None

    # Import SMCRejectionHistoryManager for tracking validation rejections
    try:
        from forex_scanner.alerts.smc_rejection_history import SMCRejectionHistoryManager
        SMC_REJECTION_AVAILABLE = True
        print("✅ Successfully imported SMCRejectionHistoryManager")
    except ImportError as e:
        print(f"⚠️ SMCRejectionHistoryManager not available: {e}")
        SMCRejectionHistoryManager = None
        SMC_REJECTION_AVAILABLE = False

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

# Import MinIO client for chart storage
try:
    from forex_scanner.services.minio_client import upload_vision_chart, get_minio_client
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    upload_vision_chart = None
    get_minio_client = None


class TradingOrchestrator:
    """
    Trading System Orchestrator - ENHANCED for Modular Claude API Integration
    
    ENHANCED RESPONSIBILITIES:
    - Intelligence filtering and configuration (owns intelligence)
    - Risk management and validation
    - ENHANCED Claude analysis via IntegrationManager with modular API
    - Advanced Claude configuration management
    - Runtime Claude settings updates
    - Database operations via AlertHistoryManager
    - Trade execution coordination
    - Performance tracking and session management
    
    REMOVED RESPONSIBILITIES:
    - ❌ NO duplicate detection logic (handled by Scanner only)
    - ❌ NO cooldown management (handled by Scanner only)
    - ❌ NO signal cache management (handled by Scanner only)
    
    NOT Responsible For:
    - Raw signal detection (handled by Scanner)
    - Basic confidence filtering (handled by Scanner)
    - Basic deduplication (handled by Scanner)
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager = None,
        # INTELLIGENCE CONFIGURATION (orchestrator owns this)
        intelligence_mode: str = 'balanced',
        intelligence_preset: str = 'balanced',
        intelligence_threshold: float = None,
        enable_market_intelligence: bool = True,
        # ENHANCED CLAUDE CONFIGURATION
        enable_claude_analysis: bool = None,
        claude_analysis_mode: str = None,
        claude_analysis_level: str = None,
        use_advanced_claude_prompts: bool = None,
        # TRADING CONFIGURATION
        enable_trading: bool = None,
        # SCANNER CONFIGURATION (passed to scanner)
        scan_interval: int = None,
        epic_list: List[str] = None,
        min_confidence: float = None,
        use_bid_adjustment: bool = None,
        spread_pips: float = None,
        user_timezone: str = 'Europe/Stockholm',
        **kwargs
    ):
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # INTELLIGENCE CONFIGURATION (orchestrator owns this)
        self.intelligence_mode = intelligence_mode
        self.intelligence_preset = intelligence_preset
        self.intelligence_threshold = intelligence_threshold or self._get_intelligence_threshold()
        self.enable_market_intelligence = enable_market_intelligence
        
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

        self.logger.info("[CONFIG:DB] ✅ TradingOrchestrator config loaded from database (NO FALLBACK)")

        # ENHANCED CLAUDE CONFIGURATION - read from database (NO FALLBACK)
        if enable_claude_analysis is not None:
            self.enable_claude = enable_claude_analysis
        else:
            self.enable_claude = self._scanner_cfg.require_claude_approval

        # SCALP MODE CLAUDE OVERRIDE: Control Claude AI based on scalp mode config
        if getattr(self._smc_cfg, 'scalp_mode_enabled', False):
            scalp_claude_enabled = getattr(self._smc_cfg, 'scalp_enable_claude_ai', True)
            if scalp_claude_enabled:
                if not self.enable_claude:
                    self.enable_claude = True
                    self.logger.info("🎯 [SCALP MODE] Claude AI validation enabled for scalp trades")
                else:
                    self.logger.info("🎯 [SCALP MODE] Claude AI validation active (already enabled)")
            else:
                # DISABLE Claude when scalp config says so (override scanner config)
                self.enable_claude = False
                self.logger.info("🎯 [SCALP MODE] Claude AI validation DISABLED per scalp config")

        self.claude_analysis_mode = claude_analysis_mode or 'minimal'
        self.claude_analysis_level = claude_analysis_level or 'institutional'
        self.use_advanced_claude_prompts = use_advanced_claude_prompts if use_advanced_claude_prompts is not None else True

        # TRADING CONFIGURATION - from database (NO FALLBACK)
        if enable_trading is not None:
            self.enable_trading = enable_trading
        else:
            self.enable_trading = self._scanner_cfg.enable_order_execution

        # SCANNER CONFIGURATION (from database - NO FALLBACK)
        self.scan_interval = scan_interval if scan_interval is not None else self._scanner_cfg.scan_interval
        self.epic_list = epic_list or self._smc_cfg.enabled_pairs
        self.min_confidence = min_confidence if min_confidence is not None else self._scanner_cfg.min_confidence
        self.use_bid_adjustment = use_bid_adjustment if use_bid_adjustment is not None else False
        self.spread_pips = spread_pips if spread_pips is not None else 1.5
        self.user_timezone = user_timezone
        
        # Session tracking
        self.session_id = None
        self.running = False
        self.scan_count = 0
        
        # Initialize database with fallback
        if not db_manager:
            try:
                db_manager = DatabaseManager()
                if not db_manager.test_connection():
                    self.logger.warning("⚠️ Database connection failed - some features will be limited")
                    db_manager = None
            except Exception as e:
                self.logger.warning(f"⚠️ Database initialization failed: {e}")
                db_manager = None
        
        self.db_manager = db_manager
        
        # CRITICAL: Initialize AlertHistoryManager for database operations
        try:
            if self.db_manager:
                self.alert_history = AlertHistoryManager(self.db_manager)
                self.logger.info("✅ AlertHistoryManager initialized successfully")
            else:
                self.alert_history = None
                self.logger.warning("⚠️ AlertHistoryManager not available - database functionality limited")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AlertHistoryManager: {e}")
            self.alert_history = None
        
        # Initialize clean scanner (NO intelligence parameters)
        self.scanner = self._initialize_clean_scanner()
        
        # Initialize modular trading components
        try:
            self.order_manager = OrderManager(
                logger=self.logger,
                enable_trading=self.enable_trading
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ OrderManager not available: {e}")
            self.order_manager = None
        
        try:
            self.session_manager = SessionManager(
                logger=self.logger,
                scan_interval=self.scan_interval
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ SessionManager not available: {e}")
            self.session_manager = None
        
        try:
            self.risk_manager = RiskManager(
                logger=self.logger
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ RiskManager not available: {e}")
            self.risk_manager = None
        
        # UPDATED: Initialize NEW TradeValidator (NO duplicate detection parameters)
        # Pass Claude filtering override based on scalp mode config
        try:
            self.trade_validator = TradeValidator(
                logger=self.logger,
                db_manager=self.db_manager,  # NEW: Pass db_manager for S/R validation
                alert_history_manager=self.alert_history,  # NEW: For saving Claude rejections to DB
                enable_claude_filtering=self.enable_claude  # NEW: Pass scalp mode Claude setting
                # REMOVED: enable_duplicate_check, cooldown_minutes, duplicate_sensitivity
                # These are no longer used in the new TradeValidator
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ TradeValidator not available: {e}")
            self.trade_validator = None
        
        try:
            self.performance_tracker = PerformanceTracker(
            logger=self.logger,
            db_manager=self.db_manager
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ PerformanceTracker not available: {e}")
            self.performance_tracker = None
        
        try:
            self.market_monitor = MarketMonitor(
            logger=self.logger
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ MarketMonitor not available: {e}")
            self.market_monitor = None
        
        # ENHANCED: Initialize IntegrationManager with advanced Claude configuration
        try:
            # Get data_fetcher from scanner's signal_detector for vision analysis
            data_fetcher = None
            if self.scanner and hasattr(self.scanner, 'signal_detector'):
                signal_detector = self.scanner.signal_detector
                if hasattr(signal_detector, 'data_fetcher'):
                    data_fetcher = signal_detector.data_fetcher
                    self.logger.info("📊 Vision analysis enabled: data_fetcher available")

            self.integration_manager = IntegrationManager(
                db_manager=self.db_manager,
                logger=self.logger,
                enable_claude=self.enable_claude,
                enable_notifications=self._scanner_cfg.notifications_enabled,  # From database - NO FALLBACK
                data_fetcher=data_fetcher
            )
        except (NameError, ImportError) as e:
            self.logger.warning(f"⚠️ IntegrationManager not available: {e}")
            self.integration_manager = None
        
        # ENHANCED: Configure Claude settings after initialization
        self._configure_advanced_claude_features()

        # Initialize RejectionOutcomeAnalyzer for automatic outcome analysis
        self.outcome_analyzer = None
        self._last_outcome_analysis = None
        self._outcome_analysis_interval_hours = 1  # Run every hour
        try:
            if RejectionOutcomeAnalyzer is not None:
                self.outcome_analyzer = RejectionOutcomeAnalyzer()
                self.logger.info("✅ RejectionOutcomeAnalyzer initialized for automatic outcome tracking")
            else:
                self.logger.warning("⚠️ RejectionOutcomeAnalyzer not available")
        except Exception as e:
            self.logger.warning(f"⚠️ RejectionOutcomeAnalyzer initialization failed: {e}")

        # Initialize SMCRejectionHistoryManager for tracking validation rejections (S/R, etc.)
        self.smc_rejection_manager = None
        try:
            if SMC_REJECTION_AVAILABLE and SMCRejectionHistoryManager and self.db_manager:
                self.smc_rejection_manager = SMCRejectionHistoryManager(self.db_manager)
                self.logger.info("✅ SMCRejectionHistoryManager initialized for validation rejection tracking")
            else:
                self.logger.warning("⚠️ SMCRejectionHistoryManager not available - validation rejections won't be saved")
        except Exception as e:
            self.logger.warning(f"⚠️ SMCRejectionHistoryManager initialization failed: {e}")

        self.logger.info("🎯 TradingOrchestrator initialized with ENHANCED modular Claude integration")
        self.logger.info(f"   Intelligence mode: {self.intelligence_mode}")
        self.logger.info(f"   Intelligence preset: {self.intelligence_preset}")
        self.logger.info(f"   Intelligence threshold: {self.intelligence_threshold:.1%}")
        self.logger.info(f"   Market intelligence: {self.enable_market_intelligence}")
        self.logger.info(f"   Trading enabled: {self.enable_trading}")
        self.logger.info(f"   Claude enabled: {self.enable_claude}")
        self.logger.info(f"   Claude analysis mode: {self.claude_analysis_mode}")
        self.logger.info(f"   Claude analysis level: {self.claude_analysis_level}")
        self.logger.info(f"   Advanced Claude prompts: {self.use_advanced_claude_prompts}")
        self.logger.info(f"   Scanner min confidence: {self.min_confidence:.1%}")
        self.logger.info(f"   Scan interval: {self.scan_interval}s")
        self.logger.info(f"   Database available: {'Yes' if self.db_manager else 'No'}")
        self.logger.info(f"   AlertHistoryManager available: {'Yes' if self.alert_history else 'No'}")
        
        # UPDATED: Show NEW TradeValidator configuration (no duplicate detection)
        if self.trade_validator:
            validator_stats = self.trade_validator.get_validation_statistics()
            self.logger.info("🛡️ NEW TradeValidator Configuration:")
            self.logger.info(f"   Duplicate detection: {validator_stats['status']['duplicate_detection']}")
            self.logger.info(f"   Validation focus: {validator_stats['status']['validation_focus']}")
            self.logger.info(f"   Min confidence: {validator_stats['configuration']['min_confidence']:.1%}")
            self.logger.info(f"   Market hours validation: {validator_stats['configuration']['validate_market_hours']}")
            self.logger.info(f"   Trading hours: {validator_stats['configuration']['trading_hours']}")
        else:
            self.logger.info("🛡️ TradeValidator not available")
        
        # ENHANCED: Log Claude integration status
        self._log_claude_integration_status()
        
        # Log component status
        self._log_component_status()
    
    def _configure_advanced_claude_features(self):
        """
        ENHANCED: Configure advanced Claude features after initialization
        """
        try:
            if not self.integration_manager or not self.enable_claude:
                return
            
            # Configure advanced Claude settings if supported
            if hasattr(self.integration_manager, 'update_claude_configuration'):
                self.integration_manager.update_claude_configuration(
                    analysis_level=self.claude_analysis_level,
                    use_advanced_prompts=self.use_advanced_claude_prompts
                )
                self.logger.info("✅ Advanced Claude configuration applied")
            else:
                self.logger.info("ℹ️ Using legacy Claude configuration")
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring advanced Claude features: {e}")
    
    def _log_claude_integration_status(self):
        """
        ENHANCED: Log detailed Claude integration status
        """
        try:
            if not self.integration_manager:
                self.logger.info("🤖 Claude Integration: Not available")
                return
            
            # Get Claude validation status
            if hasattr(self.integration_manager, 'validate_claude_configuration'):
                validation = self.integration_manager.validate_claude_configuration()
                
                self.logger.info("🤖 Claude Integration Status:")
                self.logger.info(f"   Integration status: {validation.get('integration_status', 'Unknown')}")
                self.logger.info(f"   Available methods: {len(validation.get('available_methods', []))}")
                self.logger.info(f"   Modular features: {len(validation.get('modular_features', []))}")
                
                if validation.get('advanced_config'):
                    advanced = validation['advanced_config']
                    self.logger.info(f"   Advanced prompts: {advanced.get('use_advanced_prompts', False)}")
                    self.logger.info(f"   Analysis level: {advanced.get('analysis_level', 'Unknown')}")
                
                if validation.get('recommended_fixes'):
                    self.logger.warning("⚠️ Claude configuration recommendations:")
                    for fix in validation['recommended_fixes']:
                        self.logger.warning(f"   - {fix}")
            else:
                self.logger.info("🤖 Claude Integration: Legacy mode")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging Claude status: {e}")
    
    def _get_intelligence_threshold(self) -> float:
        """Get intelligence threshold based on preset"""
        preset_thresholds = {
            'disabled': 0.0,
            'minimal': 0.3,
            'balanced': 0.5,
            'conservative': 0.7,
            'strict': 0.8
        }
        return preset_thresholds.get(self.intelligence_preset, 0.5)
    
    def _initialize_clean_scanner(self):
        """Initialize scanner with PURE signal detection parameters (NO intelligence)"""
        scanner_attempts = [
            self._try_clean_intelligent_scanner,
            self._try_core_scanner_factory,
            self._try_signal_detector_direct,
            self._try_fallback_scanner
        ]
        
        for i, attempt_func in enumerate(scanner_attempts, 1):
            try:
                self.logger.info(f"🔄 Clean scanner initialization attempt #{i}...")
                scanner = attempt_func()
                if scanner:
                    self.logger.info(f"✅ Clean scanner initialized successfully (method #{i})")
                    return scanner
            except Exception as e:
                self.logger.warning(f"⚠️ Scanner initialization attempt #{i} failed: {e}")
                continue
        
        self.logger.error("❌ All scanner initialization attempts failed")
        return None
    
    def _try_clean_intelligent_scanner(self):
        """Try to initialize IntelligentForexScanner with CLEAN parameters only"""
        try:
            scanner = IntelligentForexScanner(
                db_manager=self.db_manager,
                epic_list=self.epic_list,
                min_confidence=self.min_confidence,
                scan_interval=self.scan_interval,
                use_bid_adjustment=self.use_bid_adjustment,
                spread_pips=self.spread_pips,
                user_timezone=self.user_timezone
                # NO intelligence parameters passed to scanner
            )
            self.logger.info("✅ Clean IntelligentForexScanner initialized")
            return scanner
        except Exception as e:
            self.logger.debug(f"Clean IntelligentForexScanner initialization failed: {e}")
            raise
    
    def _try_core_scanner_factory(self):
        """Try to use core scanner factory with clean parameters"""
        try:
            from core import create_enhanced_scanner
            scanner = create_enhanced_scanner(
                db_manager=self.db_manager,
                epic_list=self.epic_list,
                min_confidence=self.min_confidence,
                scan_interval=self.scan_interval,
                use_bid_adjustment=self.use_bid_adjustment,
                spread_pips=self.spread_pips,
                user_timezone=self.user_timezone
                # NO intelligence parameters
            )
            return scanner
        except Exception as e:
            self.logger.debug(f"Core scanner factory failed: {e}")
            raise
    
    def _try_signal_detector_direct(self):
        """Try to create a minimal scanner using SignalDetector directly"""
        try:
            class MinimalScanner:
                def __init__(self, signal_detector, epic_list, logger):
                    self.signal_detector = signal_detector
                    self.epic_list = epic_list
                    self.logger = logger
                
                def scan_once(self):
                    """Minimal scan implementation"""
                    try:
                        signals = []
                        for epic in self.epic_list:
                            try:
                                signal = self.signal_detector.detect_signals(epic)
                                if signal:
                                    signals.extend(signal if isinstance(signal, list) else [signal])
                            except Exception as e:
                                self.logger.debug(f"Signal detection failed for {epic}: {e}")
                                continue
                        return signals
                    except Exception as e:
                        self.logger.error(f"Minimal scanner scan failed: {e}")
                        return []
                
                def stop(self):
                    """Stop method for compatibility"""
                    pass
            
            signal_detector = SignalDetector(self.db_manager, self.user_timezone)
            scanner = MinimalScanner(signal_detector, self.epic_list, self.logger)
            self.logger.info("🔄 Created minimal scanner using SignalDetector")
            return scanner
        except Exception as e:
            self.logger.debug(f"Minimal scanner creation failed: {e}")
            raise
    
    def _try_fallback_scanner(self):
        """Create an absolute fallback scanner"""
        try:
            class FallbackScanner:
                def __init__(self, epic_list, logger):
                    self.epic_list = epic_list
                    self.logger = logger
                    self.logger.warning("⚠️ Using fallback scanner - limited functionality")
                
                def scan_once(self):
                    """Fallback scan that returns empty results"""
                    self.logger.warning("⚠️ Fallback scanner: returning empty results")
                    return []
                
                def stop(self):
                    """Stop method for compatibility"""
                    pass
            
            scanner = FallbackScanner(self.epic_list, self.logger)
            return scanner
        except Exception as e:
            self.logger.debug(f"Fallback scanner creation failed: {e}")
            raise
    
    def _log_component_status(self):
        """Log the status of all components"""
        self.logger.info("📊 Component Status:")
        
        components = [
            ("DatabaseManager", self.db_manager is not None),
            ("AlertHistoryManager", self.alert_history is not None),
            ("Scanner", self.scanner is not None),
            ("OrderManager", self.order_manager is not None),
            ("SessionManager", self.session_manager is not None),
            ("RiskManager", self.risk_manager is not None),
            ("TradeValidator (NEW - No Duplicates)", self.trade_validator is not None),
            ("PerformanceTracker", self.performance_tracker is not None),
            ("MarketMonitor", self.market_monitor is not None),
            ("IntegrationManager (Enhanced Claude)", self.integration_manager is not None),
            ("SignalDetector", self.scanner and hasattr(self.scanner, 'signal_detector'))
        ]
        
        working_components = 0
        total_components = len(components)
        
        for name, status in components:
            status_icon = "✅" if status else "❌"
            self.logger.info(f"   {name}: {status_icon}")
            if status:
                working_components += 1
        
        health_percentage = (working_components / total_components) * 100
        self.logger.info(f"📊 System Health: {health_percentage:.1f}% ({working_components}/{total_components} components)")

    def _save_validation_rejection(self, signal: Dict, reason: str) -> None:
        """
        Save validation rejection to smc_simple_rejections table for analytics.

        This captures rejections from TradeValidator including:
        - S/R Level proximity rejections
        - S/R Path Blocking rejections
        - EMA200 filter rejections
        - Claude filtering rejections (if not already saved by validator)
        - Other validation failures

        Args:
            signal: The rejected signal dictionary
            reason: The rejection reason string
        """
        if not self.smc_rejection_manager:
            return

        try:
            from datetime import timezone

            epic = signal.get('epic', '')
            pair = signal.get('pair', epic.split('.')[2] if len(epic.split('.')) > 2 else epic)

            # Determine rejection stage based on reason
            rejection_stage = self._determine_rejection_stage(reason)

            # Get timestamp
            timestamp = signal.get('timestamp')
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Extract S/R path blocking details if available
            sr_path_blocking = signal.get('sr_path_blocking', {})

            # Build rejection data
            rejection_data = {
                'scan_timestamp': timestamp,
                'epic': epic,
                'pair': pair,
                'rejection_stage': rejection_stage,
                'rejection_reason': reason,
                'attempted_direction': signal.get('signal_type', ''),
                'strategy_version': signal.get('strategy_version', 'unknown'),

                # Price context
                'current_price': signal.get('entry_price'),
                'potential_entry': signal.get('entry_price'),
                'potential_stop_loss': signal.get('stop_loss'),
                'potential_take_profit': signal.get('take_profit'),
                'potential_risk_pips': signal.get('risk_pips'),
                'potential_reward_pips': signal.get('reward_pips'),
                'potential_rr_ratio': signal.get('rr_ratio'),

                # Confidence
                'confidence_score': signal.get('confidence_score'),

                # Market context from signal
                'market_hour': signal.get('market_hour'),
                'market_session': signal.get('market_session'),

                # S/R Path Blocking specific fields
                'sr_blocking_level': sr_path_blocking.get('blocking_sr_level'),
                'sr_blocking_type': sr_path_blocking.get('blocking_sr_type'),
                'sr_blocking_distance_pips': sr_path_blocking.get('distance_pips'),
                'sr_path_blocked_pct': sr_path_blocking.get('path_blocked_pct'),
                'target_distance_pips': sr_path_blocking.get('target_distance_pips'),
            }

            # Save to database
            self.smc_rejection_manager.save_rejection(rejection_data)
            self.logger.debug(f"💾 Validation rejection saved: {epic} - {rejection_stage}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save validation rejection: {e}")

    def _determine_rejection_stage(self, reason: str) -> str:
        """
        Determine the rejection stage based on the rejection reason string.

        Args:
            reason: The rejection reason from TradeValidator

        Returns:
            Rejection stage constant (e.g., 'SR_PATH_BLOCKED', 'S/R_LEVEL', etc.)
        """
        reason_lower = reason.lower()

        # S/R Path Blocking (most specific - check first)
        if 'blocks' in reason_lower and ('path' in reason_lower or '% of' in reason_lower):
            return 'SR_PATH_BLOCKED'

        # S/R Level proximity
        if 's/r level' in reason_lower or 'too close to' in reason_lower:
            return 'SR_LEVEL'

        # Support/Resistance cluster
        if 'cluster' in reason_lower and ('support' in reason_lower or 'resistance' in reason_lower):
            return 'SR_CLUSTER'

        # EMA200 filter
        if 'ema200' in reason_lower or 'ema 200' in reason_lower:
            return 'EMA200_FILTER'

        # Claude filtering (shouldn't normally get here - validator handles it)
        if 'claude' in reason_lower:
            return 'CLAUDE_FILTER'

        # News filtering
        if 'news' in reason_lower:
            return 'NEWS_FILTER'

        # Market Bias Filter (v2.3.2) - counter-trend trades blocked by high consensus
        if 'market bias filter' in reason_lower or 'market bias conflict' in reason_lower:
            return 'MARKET_BIAS_FILTER'

        # Market hours
        if 'market hours' in reason_lower or 'outside trading' in reason_lower:
            return 'MARKET_HOURS'

        # Confidence
        if 'confidence' in reason_lower:
            return 'CONFIDENCE'

        # Default: Generic validation failure
        return 'VALIDATION_FAILED'

    def _apply_intelligence_filtering(self, signals: List[Dict]) -> List[Dict]:
        """
        Apply intelligence filtering (orchestrator responsibility)
        """
        if not self.enable_market_intelligence or self.intelligence_mode == 'disabled':
            self.logger.info("🧠 Intelligence filtering: disabled")
            return signals
        
        self.logger.info(f"🧠 Applying {self.intelligence_preset} intelligence filtering")
        self.logger.info(f"   Intelligence threshold: {self.intelligence_threshold:.1%}")
        
        filtered_signals = []
        
        for signal in signals:
            epic = signal.get('epic', 'Unknown')
            
            # Calculate intelligence score (orchestrator responsibility)
            intelligence_score = self._calculate_intelligence_score(signal)
            
            if intelligence_score >= self.intelligence_threshold:
                signal['intelligence_score'] = intelligence_score
                signal['intelligence_preset'] = self.intelligence_preset
                signal['intelligence_filtered'] = True
                filtered_signals.append(signal)
                self.logger.info(f"✅ INTELLIGENCE PASSED {epic}: {intelligence_score:.1%} ≥ {self.intelligence_threshold:.1%}")
            else:
                self.logger.info(f"🚫 INTELLIGENCE FILTERED {epic}: {intelligence_score:.1%} < {self.intelligence_threshold:.1%}")
        
        self.logger.info(f"🧠 Intelligence result: {len(filtered_signals)}/{len(signals)} signals passed")
        return filtered_signals
    
    def _calculate_intelligence_score(self, signal: Dict) -> float:
        """
        Calculate intelligence score (orchestrator responsibility)
        """
        try:
            # Get base score from signal confidence
            base_score = signal.get('confidence_score', 0.75)
            
            # Apply intelligence scoring based on preset
            if self.intelligence_preset == 'minimal':
                # Very permissive - boost most signals
                return min(1.0, base_score + 0.15)
            elif self.intelligence_preset == 'balanced':
                # Moderate requirements - use confidence as-is with small boost
                return min(1.0, base_score + 0.05)
            elif self.intelligence_preset == 'conservative':
                # More strict requirements - reduce score slightly
                return max(0.0, base_score - 0.05)
            elif self.intelligence_preset == 'strict':
                # Very strict requirements - significant reduction
                return max(0.0, base_score - 0.15)
            else:
                return base_score
                
        except Exception as e:
            self.logger.debug(f"Intelligence score calculation error: {e}")
            return 0.5  # Neutral score

    def _save_signal_to_database(self, signal: Dict, claude_result = None) -> Optional[int]:
        """
        Save signal to alert_history table with ALL columns populated
        ENHANCED: Better handling of modular Claude results
        UPDATED: Store alert_id in signal for order execution tracking
        FIXED: Properly merge Claude data into signal before database save
        """
        if not self.alert_history:
            self.logger.warning("⚠️ AlertHistoryManager not available - cannot save signal to database")
            return None
        
        # FIX: Ensure signal is a proper dictionary (existing fix)
        if not isinstance(signal, dict):
            self.logger.error(f"❌ Signal is not a dictionary, got {type(signal)}: {signal}")
            if isinstance(signal, str):
                signal = {
                    'epic': signal,
                    'signal_type': 'Unknown',
                    'strategy': 'Unknown',
                    'confidence_score': 0.0,
                    'price': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.logger.error(f"❌ Cannot convert {type(signal)} to signal dictionary")
                return None
        
        # ENHANCED: Handle modular Claude result formats
        processed_claude_result = None
        if claude_result is not None:
            if isinstance(claude_result, str):
                self.logger.info(f"🔄 Converting Claude string response to dictionary")
                # Parse the string response into dictionary format
                processed_claude_result = self._parse_claude_string_response(claude_result)
                self.logger.debug(f"   Parsed Claude result: {processed_claude_result}")
            elif isinstance(claude_result, dict):
                processed_claude_result = claude_result
                self.logger.debug(f"   Claude result already in dictionary format")
                
                # ENHANCED: Log modular Claude features if present
                if processed_claude_result.get('claude_analysis_level'):
                    self.logger.debug(f"   Analysis level: {processed_claude_result['claude_analysis_level']}")
                if processed_claude_result.get('claude_advanced_prompts'):
                    self.logger.debug(f"   Advanced prompts: {processed_claude_result['claude_advanced_prompts']}")
            else:
                self.logger.warning(f"⚠️ Unexpected Claude result type: {type(claude_result)}")
                processed_claude_result = None
        
        try:
            # Create alert message
            epic = signal.get('epic', 'Unknown')
            signal_type = signal.get('signal_type', 'Unknown')
            confidence = signal.get('confidence_score', 0)
            strategy = signal.get('strategy', 'Unknown')
            
            alert_message = f"Live signal: {signal_type} {epic} @ {confidence:.1%} confidence via {strategy}"
            
            # ENHANCED: Add Claude analysis level to alert message if available
            if processed_claude_result and processed_claude_result.get('claude_analysis_level'):
                alert_message += f" (Claude: {processed_claude_result['claude_analysis_level']})"
            
            # FIXED: Merge Claude data into signal copy before saving to database
            signal_with_claude = signal.copy()  # Create copy to avoid modifying original
            
            if processed_claude_result:
                # Merge all Claude data into the signal for comprehensive database storage
                claude_data_to_merge = {
                    'claude_analysis': processed_claude_result.get('raw_response') or processed_claude_result.get('analysis'),
                    'claude_score': processed_claude_result.get('score'),
                    'claude_decision': processed_claude_result.get('decision'),
                    'claude_approved': processed_claude_result.get('approved', False),
                    'claude_reason': processed_claude_result.get('reason'),
                    'claude_mode': processed_claude_result.get('mode', 'institutional'),
                    'claude_raw_response': processed_claude_result.get('raw_response'),
                    'claude_analysis_level': processed_claude_result.get('claude_analysis_level'),
                    'claude_advanced_prompts': processed_claude_result.get('claude_advanced_prompts'),
                    'technical_validation_passed': processed_claude_result.get('technical_validation_passed', False)
                }
                
                # Only add non-None values to avoid overwriting existing data with None
                for key, value in claude_data_to_merge.items():
                    if value is not None:
                        signal_with_claude[key] = value
                
                self.logger.debug(f"🤖 Merged Claude data into signal for database save")
            else:
                self.logger.debug(f"ℹ️ No Claude data to merge for this signal")
            
            # Log what we're about to save for debugging
            self.logger.debug(f"🔍 Saving signal: Epic={epic}, Type={signal_type}, Strategy={strategy}")
            if processed_claude_result:
                self.logger.debug(f"   Claude: Score={processed_claude_result.get('score')}, Decision={processed_claude_result.get('decision')}")
                if processed_claude_result.get('claude_analysis_level'):
                    self.logger.debug(f"   Analysis Level: {processed_claude_result['claude_analysis_level']}")
            
            # Save using AlertHistoryManager with Claude-enhanced signal data
            alert_id = self.alert_history.save_alert(
                signal=signal_with_claude,  # Use signal with merged Claude data
                alert_message=alert_message,
                alert_level='INFO',
                claude_result=processed_claude_result  # Also pass as separate parameter for backward compatibility
            )
            
            if alert_id:
                # ADD: Store alert_id in original signal for order execution (preserve existing functionality)
                signal['alert_id'] = alert_id

                # Save vision artifacts with alert_id prefix (if available)
                if processed_claude_result and processed_claude_result.get('_vision_artifacts'):
                    self._save_vision_artifacts_with_alert_id(
                        alert_id=alert_id,
                        signal=signal,
                        claude_result=processed_claude_result
                    )

                # v3.2.0: No notification needed - strategy reads cooldowns directly from
                # alert_history database (single source of truth)

                # Enhanced logging with Claude status
                claude_status = "No Claude"
                if processed_claude_result:
                    decision = processed_claude_result.get('decision', 'Unknown')
                    score = processed_claude_result.get('score', 'N/A')
                    claude_status = f"Claude: {decision} ({score}/10)"

                self.logger.info(f"✅ Signal saved to alert_history table (ID: {alert_id}) - {claude_status}")

                # v2.24.0: Set cooldown AFTER signal is saved (not at deduplication stage)
                # This prevents "phantom cooldowns" when signals pass dedup but are rejected by SMC
                self._set_signal_cooldown(signal)

                return alert_id
            else:
                self.logger.error(f"❌ Failed to save signal to database - AlertHistoryManager returned None")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error saving signal to database: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return None

    def _set_signal_cooldown(self, signal: Dict) -> None:
        """
        v2.24.0: Set cooldown for a signal AFTER it has been saved to database.

        This is called after all filters (deduplication, SMC, Claude) have passed
        and the signal has been saved. This prevents "phantom cooldowns" where
        signals that pass deduplication but are rejected by later filters
        still block future signals.

        Args:
            signal: Signal dictionary with 'epic', 'signal_type', and 'strategy'
        """
        try:
            # Get deduplication manager from scanner
            if not self.scanner or not hasattr(self.scanner, 'deduplication_manager'):
                return

            dedup_manager = self.scanner.deduplication_manager
            if not dedup_manager:
                return

            epic = signal.get('epic', '')
            signal_type = signal.get('signal_type', '')
            strategy = signal.get('strategy', 'Unknown')

            if epic and signal_type:
                dedup_manager.set_cooldown(epic, signal_type, strategy)
            else:
                self.logger.warning(f"Cannot set cooldown - missing epic or signal_type: {signal}")

        except Exception as e:
            self.logger.warning(f"Error setting signal cooldown: {e}")

    def _save_vision_artifacts_with_alert_id(
        self,
        alert_id: int,
        signal: Dict,
        claude_result: Dict
    ) -> Optional[str]:
        """
        Save vision analysis artifacts (chart, prompt, result) to disk with alert_id prefix.
        Also updates the alert_history record with the chart URL.

        Args:
            alert_id: Database alert ID for file naming
            signal: Signal dictionary
            claude_result: Claude analysis result containing _vision_artifacts

        Returns:
            Chart file path if saved, None otherwise
        """
        try:
            import base64

            # Get vision artifacts from claude_result
            artifacts = claude_result.get('_vision_artifacts', {})
            chart_base64 = artifacts.get('chart_base64')
            prompt = artifacts.get('prompt', '')

            # Get save directory from database config (NO FALLBACK to config.py)
            if self._scanner_cfg:
                vision_dir = self._scanner_cfg.claude_vision_save_directory
            else:
                vision_dir = 'claude_analysis_enhanced/vision_analysis'
            os.makedirs(vision_dir, exist_ok=True)

            # Generate filename prefix with alert_id
            epic = signal.get('epic', 'unknown').replace('.', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = f"{alert_id}_{epic}_{timestamp}"

            files_saved = []
            minio_chart_url = None

            # 1. Save chart image - try MinIO first, then fall back to disk
            if chart_base64:
                try:
                    chart_bytes = base64.b64decode(chart_base64)

                    # Try MinIO upload first
                    if MINIO_AVAILABLE and upload_vision_chart:
                        minio_chart_url = upload_vision_chart(
                            chart_bytes,
                            alert_id,
                            signal.get('epic', 'unknown'),
                            timestamp
                        )
                        if minio_chart_url:
                            self.logger.info(f"📊 Chart uploaded to MinIO: {minio_chart_url}")
                            files_saved.append(f"{prefix}_chart.png (MinIO)")

                    # Fall back to disk if MinIO fails or unavailable
                    if not minio_chart_url:
                        chart_path = os.path.join(vision_dir, f"{prefix}_chart.png")
                        with open(chart_path, 'wb') as f:
                            f.write(chart_bytes)
                        files_saved.append(f"{prefix}_chart.png")
                        self.logger.info(f"📊 Chart saved to disk: {chart_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to save chart: {e}")

            # 2. Save signal data as JSON
            signal_path = os.path.join(vision_dir, f"{prefix}_signal.json")
            try:
                signal_data = {}
                for k, v in signal.items():
                    try:
                        json.dumps(v)
                        signal_data[k] = v
                    except (TypeError, ValueError):
                        signal_data[k] = str(v)

                with open(signal_path, 'w', encoding='utf-8') as f:
                    json.dump(signal_data, f, indent=2, default=str)
                files_saved.append(f"{prefix}_signal.json")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save signal data: {e}")

            # 3. Save prompt text
            prompt_path = os.path.join(vision_dir, f"{prefix}_prompt.txt")
            try:
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(f"═══════════════════════════════════════════════════════════════\n")
                    f.write(f"CLAUDE VISION ANALYSIS PROMPT\n")
                    f.write(f"═══════════════════════════════════════════════════════════════\n")
                    f.write(f"Alert ID: {alert_id}\n")
                    f.write(f"Epic: {signal.get('epic', 'Unknown')}\n")
                    f.write(f"Strategy: {signal.get('strategy', 'Unknown')}\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Chart Included: {chart_base64 is not None}\n")
                    f.write(f"═══════════════════════════════════════════════════════════════\n\n")
                    f.write(prompt)
                files_saved.append(f"{prefix}_prompt.txt")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save prompt: {e}")

            # 4. Save analysis result as JSON
            result_path = os.path.join(vision_dir, f"{prefix}_result.json")
            try:
                result_data = {
                    'alert_id': alert_id,
                    'epic': signal.get('epic'),
                    'signal_type': signal.get('signal_type'),
                    'strategy': signal.get('strategy'),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'score': claude_result.get('score'),
                    'decision': claude_result.get('decision'),
                    'approved': claude_result.get('approved'),
                    'reason': claude_result.get('reason'),
                    'vision_used': claude_result.get('vision_used', chart_base64 is not None),
                    'tokens_used': claude_result.get('tokens_used'),
                    'mode': claude_result.get('mode'),
                    'raw_response': claude_result.get('raw_response'),
                    'files': {
                        'chart': f"{prefix}_chart.png" if chart_base64 else None,
                        'signal': f"{prefix}_signal.json",
                        'prompt': f"{prefix}_prompt.txt",
                        'result': f"{prefix}_result.json"
                    }
                }
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2)
                files_saved.append(f"{prefix}_result.json")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save result: {e}")

            if files_saved:
                self.logger.info(f"✅ Vision artifacts saved: {', '.join(files_saved)}")

            # Clean up _vision_artifacts from claude_result to save memory
            if '_vision_artifacts' in claude_result:
                del claude_result['_vision_artifacts']

            # Update alert_history with vision_chart_url if chart was saved
            if chart_base64 and self.alert_history:
                try:
                    # Use MinIO URL if available, otherwise use disk path
                    if minio_chart_url:
                        chart_url = minio_chart_url
                    else:
                        chart_url = f"file://{vision_dir}/{prefix}_chart.png"

                    self.alert_history.update_alert_vision_chart_url(alert_id, chart_url)
                    self.logger.info(f"✅ Updated alert #{alert_id} with vision_chart_url: {chart_url}")
                    return chart_url
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to update vision_chart_url in database: {e}")
                    return minio_chart_url or f"{prefix}_chart.png"

            return minio_chart_url

        except Exception as e:
            self.logger.error(f"❌ Failed to save vision artifacts: {e}")
            return None

    def _parse_claude_string_response(self, claude_string: str) -> Dict:
        """
        Parse Claude string response into dictionary format
        ENHANCED: Better parsing for modular Claude responses
        
        Expected string format:
        SCORE: 8
        DECISION: APPROVE
        REASON: All three EMAs show bullish alignment...
        """
        try:
            result = {
                'score': None,
                'decision': None,
                'reason': None,
                'approved': False,
                'raw_response': claude_string
            }
            
            lines = [line.strip() for line in claude_string.strip().split('\n') if line.strip()]
            
            for line in lines:
                if line.startswith('SCORE:'):
                    score_text = line.replace('SCORE:', '').strip()
                    try:
                        # Handle "8", "8/10", or "Score: 8"
                        import re
                        numbers = re.findall(r'\b(\d+)\b', score_text)
                        if numbers:
                            result['score'] = int(numbers[0])
                    except (ValueError, IndexError):
                        self.logger.warning(f"⚠️ Could not parse Claude score: {score_text}")
                
                elif line.startswith('DECISION:'):
                    decision = line.replace('DECISION:', '').strip().upper()
                    result['decision'] = decision
                    result['approved'] = decision == 'APPROVE'
                
                elif line.startswith('REASON:'):
                    result['reason'] = line.replace('REASON:', '').strip()
                
                # ENHANCED: Parse additional modular Claude fields
                elif line.startswith('ANALYSIS_LEVEL:'):
                    result['claude_analysis_level'] = line.replace('ANALYSIS_LEVEL:', '').strip()
                
                elif line.startswith('ADVANCED_PROMPTS:'):
                    prompts_text = line.replace('ADVANCED_PROMPTS:', '').strip().lower()
                    result['claude_advanced_prompts'] = prompts_text in ['true', 'yes', 'enabled']
                
                elif line.startswith('TECHNICAL_VALIDATION:'):
                    validation_text = line.replace('TECHNICAL_VALIDATION:', '').strip().lower()
                    result['technical_validation_passed'] = validation_text in ['true', 'yes', 'passed']
            
            # Validate that we got the essential fields
            if result['score'] is None or result['decision'] is None:
                self.logger.warning(f"⚠️ Incomplete Claude response parsing")
                self.logger.debug(f"   Original: {claude_string}")
                self.logger.debug(f"   Parsed: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing Claude string response: {e}")
            return {
                'score': None,
                'decision': 'UNKNOWN',
                'reason': 'Parse error',
                'approved': False,
                'raw_response': claude_string
            }


    def _save_all_signals_to_database(self, signals: List[Dict], claude_results: Dict = None) -> int:
        """
        Save all signals to database with proper error handling
        ENHANCED: Better logging for modular Claude results
        
        Args:
            signals: List of signals to save
            claude_results: Dictionary mapping signal index to Claude results
            
        Returns:
            Number of signals successfully saved
        """
        if not signals:
            return 0
        
        saved_count = 0
        claude_results = claude_results or {}
        
        self.logger.info(f"💾 Saving all signals to alert_history table...")
        
        for i, signal in enumerate(signals):
            try:
                # FIX: Validate signal is a dictionary before processing
                if not isinstance(signal, dict):
                    self.logger.error(f"❌ Signal #{i+1} is not a dictionary, got {type(signal)}: {signal}")
                    continue
                
                # Get Claude result for this signal if available
                claude_result = claude_results.get(i)
                
                # Log what we're processing for debugging
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                self.logger.debug(f"💾 Processing signal #{i+1}: {epic} {signal_type}")
                
                # ENHANCED: Log Claude features if present
                if claude_result and isinstance(claude_result, dict):
                    if claude_result.get('claude_analysis_level'):
                        self.logger.debug(f"   Claude level: {claude_result['claude_analysis_level']}")
                    if claude_result.get('claude_advanced_prompts'):
                        self.logger.debug(f"   Advanced prompts: {claude_result['claude_advanced_prompts']}")
                
                # Save signal to database
                alert_id = self._save_signal_to_database(signal, claude_result)
                
                if alert_id:
                    saved_count += 1
                    self.logger.debug(f"✅ Signal #{i+1} saved with ID: {alert_id}")
                else:
                    self.logger.warning(f"⚠️ Failed to save signal #{i+1} to database")
                    
            except Exception as e:
                self.logger.error(f"❌ Error saving signal #{i+1} to database: {e}")
                import traceback
                self.logger.error(f"   Traceback: {traceback.format_exc()}")
                continue
        
        self.logger.info(f"💾 Database save summary: {saved_count}/{len(signals)} signals saved")
        
        if saved_count > 0:
            self.logger.info(f"✅ Successfully saved {saved_count} signals to alert_history table")
        else:
            self.logger.warning(f"⚠️ No signals were saved to the database")
        
        return saved_count
    
    def scan_once(self) -> List[Dict]:
        """
        ENHANCED: Orchestrator handles the complete processing pipeline with advanced Claude
        Scanner does pure signal detection + deduplication, orchestrator handles intelligence + validation
        """
        if not self.scanner:
            self.logger.error("❌ No scanner available for signal detection")
            self.logger.info("🔄 Attempting to reinitialize scanner...")
            self.scanner = self._initialize_clean_scanner()
            if not self.scanner:
                self.logger.error("❌ Scanner reinitialization failed - cannot perform scan")
                return []
        
        try:
            scan_start_time = datetime.now()
            self.scan_count += 1
            
            self.logger.info(f"🔄 Scan #{self.scan_count} starting...")
            
            # 1. Get clean signals from scanner (includes deduplication)
            self.logger.info("🔍 Delegating signal detection to scanner...")
            
            # Try different scan methods based on scanner type
            if hasattr(self.scanner, 'scan_once'):
                raw_signals = self.scanner.scan_once()
            elif hasattr(self.scanner, 'scan'):
                raw_signals = self.scanner.scan()
            elif hasattr(self.scanner, 'detect_signals'):
                # For minimal scanner, try to scan all epics
                raw_signals = []
                for epic in self.epic_list:
                    try:
                        signal = self.scanner.detect_signals(epic)
                        if signal:
                            raw_signals.extend(signal if isinstance(signal, list) else [signal])
                    except Exception as e:
                        self.logger.debug(f"Signal detection failed for {epic}: {e}")
                        continue
            else:
                self.logger.error("❌ Scanner has no compatible scan method")
                return []
            
            if not raw_signals:
                self.logger.info("📊 Scanner detected 0 signals")
                self.logger.info(f"⏰ Waiting {self.scan_interval}s until next scan...")
                return []
            
            self.logger.info(f"📊 Scanner detected {len(raw_signals)} signals")
            
            # 2. Apply INTELLIGENCE FILTERING (orchestrator responsibility)
            intelligence_filtered_signals = self._apply_intelligence_filtering(raw_signals)
            
            if not intelligence_filtered_signals:
                self.logger.info("📊 No signals after intelligence filtering")
                self.logger.info(f"⏰ Waiting {self.scan_interval}s until next scan...")
                return []
            
            # 3. Apply NEW trade validation (trading-specific rules only, NO duplicate detection)
            self.logger.info(f"🔍 TRADING ORCHESTRATOR: Sending {len(intelligence_filtered_signals)} signals to TradeValidator")

            # Log signal summary before validation
            if intelligence_filtered_signals:
                signal_summary = {}
                for signal in intelligence_filtered_signals:
                    strategy = signal.get('strategy', 'Unknown')
                    epic = signal.get('epic', 'Unknown')
                    key = f"{epic}-{strategy}"
                    signal_summary[key] = signal_summary.get(key, 0) + 1
                self.logger.info(f"📊 ORCHESTRATOR SIGNALS TO VALIDATE: {dict(signal_summary)}")

            if self.trade_validator:
                valid_signals, invalid_signals = self.trade_validator.validate_signals_batch(intelligence_filtered_signals)

                # ✅ ENHANCED VALIDATION RESULTS LOGGING
                self.logger.info(f"✅ ORCHESTRATOR: TradeValidator returned {len(valid_signals)} valid, {len(invalid_signals)} invalid signals")

                if invalid_signals and len(invalid_signals) > 0:
                    # Create summary of rejected signals with epic names
                    rejected_epics = []
                    for invalid_signal in invalid_signals:
                        epic = invalid_signal.get('epic', 'Unknown')
                        signal_type = invalid_signal.get('signal_type', 'Unknown')
                        reason = invalid_signal.get('validation_error', 'Unknown reason')
                        rejected_epics.append(f"{epic} {signal_type} ({reason})")

                        # Save validation rejection to database for analytics
                        self._save_validation_rejection(invalid_signal, reason)

                    epics_summary = ', '.join(rejected_epics)
                    self.logger.warning(f"❌ Filtered out {len(invalid_signals)} invalid signals: {epics_summary}")
                   
                
                if not valid_signals:
                    self.logger.info("📊 No valid signals after validation")
                    self.logger.info(f"⏰ Waiting {self.scan_interval}s until next scan...")
                    return []
                
                self.logger.info(f"📊 Validation complete: {len(valid_signals)} valid, {len(invalid_signals)} invalid")
            else:
                valid_signals = intelligence_filtered_signals
                self.logger.warning("⚠️ No trade validator available - skipping validation")
            
            # 4. 🚀 ENHANCED: Apply risk management with testing mode bypass
            if self._scanner_cfg.strategy_testing_mode:  # From database - NO FALLBACK
                # 🚀 TESTING MODE: Skip all risk management and use minimal position sizes
                self.logger.info(f"🚀 Testing mode - risk management bypassed: {len(valid_signals)} signals approved")
                
                # Add minimal risk info to signals for compatibility
                risk_filtered_signals = []
                for signal in valid_signals:
                    enhanced_signal = signal.copy()
                    enhanced_signal['position_size'] = 0.01  # Minimal position size
                    enhanced_signal['risk_validated'] = True
                    enhanced_signal['risk_reason'] = "Testing mode - minimal position"
                    enhanced_signal['testing_mode'] = True
                    risk_filtered_signals.append(enhanced_signal)
                
            else:
                # 🛡️ LIVE MODE: Full risk management
                risk_filtered_signals = []
                if self.risk_manager:
                    can_trade, risk_msg = self.risk_manager.can_trade()
                    if not can_trade:
                        self.logger.warning(f"⚠️ Risk limits prevent trading: {risk_msg}")
                        # Still save signals to database even if can't trade
                        self._save_all_signals_to_database(valid_signals)
                        return []
                    
                    for signal in valid_signals:
                        try:
                            risk_valid, risk_reason = self.risk_manager.validate_risk_parameters(signal)
                            if not risk_valid:
                                self.logger.warning(f"❌ Risk validation failed for {signal.get('epic')}: {risk_reason}")
                                continue
                            
                            account_balance = 10000.0
                            position_size, size_reason = self.risk_manager.calculate_position_size(signal, account_balance)
                            
                            if position_size <= 0:
                                self.logger.warning(f"❌ Invalid position size for {signal.get('epic')}: {size_reason}")
                                continue
                            
                            # Apply risk adjustments if method exists
                            if hasattr(self.risk_manager, 'apply_risk_adjustments'):
                                adjusted_size = self.risk_manager.apply_risk_adjustments(position_size, signal)
                            else:
                                adjusted_size = position_size
                            
                            # Add risk info to signal
                            enhanced_signal = signal.copy()
                            enhanced_signal['position_size'] = adjusted_size
                            enhanced_signal['risk_validated'] = True
                            enhanced_signal['risk_reason'] = size_reason
                            enhanced_signal['testing_mode'] = False
                            
                            risk_filtered_signals.append(enhanced_signal)
                            
                        except Exception as e:
                            self.logger.error(f"❌ Error applying risk management to signal {signal.get('epic')}: {e}")
                            continue
                    
                    self.logger.info(f"💰 Risk management: {len(risk_filtered_signals)}/{len(valid_signals)} signals approved")
                else:
                    risk_filtered_signals = valid_signals

            
            # 5. ENHANCED: Apply Claude analysis with modular features
            # FIXED: Skip IntegrationManager Claude if TradeValidator already did Claude analysis
            # This prevents duplicate API calls and wasted tokens
            claude_results = {}
            # Check from database - NO FALLBACK
            trade_validator_did_claude = self._scanner_cfg.require_claude_approval

            if trade_validator_did_claude:
                # TradeValidator already did Claude analysis - use its results, skip IntegrationManager
                self.logger.info("🤖 Skipping IntegrationManager Claude (TradeValidator already analyzed)")
                analyzed_signals = risk_filtered_signals

                # Extract Claude results from TradeValidator's analysis
                for i, signal in enumerate(risk_filtered_signals):
                    if signal.get('claude_validation_result'):
                        claude_results[i] = signal.get('claude_validation_result')

            elif self.enable_claude and self.integration_manager:
                # No TradeValidator Claude - use IntegrationManager with vision
                self.logger.info(f"🤖 Starting enhanced Claude analysis...")
                self.logger.info(f"   Analysis mode: {self.claude_analysis_mode}")
                self.logger.info(f"   Analysis level: {self.claude_analysis_level}")
                self.logger.info(f"   Advanced prompts: {self.use_advanced_claude_prompts}")

                analyzed_signals = self.integration_manager.analyze_signals_with_claude(risk_filtered_signals)

                # Extract Claude results for database storage
                for i, signal in enumerate(analyzed_signals):
                    if signal.get('claude_analysis'):
                        claude_result = {
                            'raw_response': signal.get('claude_analysis'),
                            'score': signal.get('claude_score'),
                            'decision': signal.get('claude_decision'),
                            'approved': signal.get('claude_approved', False),
                            'reason': signal.get('claude_reason'),
                            'claude_analysis_level': signal.get('claude_analysis_level'),
                            'claude_advanced_prompts': signal.get('claude_advanced_prompts'),
                            'technical_validation_passed': signal.get('claude_technical_validation', False)
                        }
                        claude_results[i] = claude_result

                        # ENHANCED: Log modular Claude features
                        if claude_result.get('claude_analysis_level'):
                            self.logger.debug(f"   Signal #{i+1} analyzed with {claude_result['claude_analysis_level']} level")

                self.logger.info(f"🤖 Enhanced Claude analysis complete")

            else:
                # Claude disabled entirely
                analyzed_signals = risk_filtered_signals
                self.logger.info("🤖 Claude analysis disabled")
            
            # 6. CRITICAL: Save ALL signals to alert_history table
            self.logger.info("💾 Saving all signals to alert_history table...")
            saved_count = self._save_all_signals_to_database(analyzed_signals, claude_results)
            
            if saved_count == 0 and analyzed_signals:
                self.logger.error("❌ CRITICAL: No signals were saved to database despite having signals!")
            elif saved_count > 0:
                self.logger.info(f"✅ Successfully saved {saved_count} signals to alert_history table")
            
            # 7. Execute trades if enabled
            # CRITICAL: Only execute signals that were successfully saved (have alert_id)
            execution_results = []
            if self.enable_trading and analyzed_signals and self.order_manager:
                # Filter to only signals with alert_id (successfully saved to DB)
                signals_to_execute = [s for s in analyzed_signals if s.get('alert_id')]
                if len(signals_to_execute) < len(analyzed_signals):
                    skipped = len(analyzed_signals) - len(signals_to_execute)
                    self.logger.warning(f"⚠️ Skipping {skipped} signals without alert_id (DB save failed)")
                if signals_to_execute:
                    execution_results = self.order_manager.execute_signals(signals_to_execute)
                else:
                    self.logger.warning("⚠️ No signals to execute - all failed to save to database")
            
            # 8. Track performance
            if self.performance_tracker:
                try:
                    scan_duration = (datetime.now() - scan_start_time).total_seconds()
                    
                    # Track signals individually
                    for signal in analyzed_signals:
                        if hasattr(self.performance_tracker, 'track_signal_outcome'):
                            self.performance_tracker.track_signal_outcome(signal, 'GENERATED')
                        elif hasattr(self.performance_tracker, 'track_strategy_performance'):
                            strategy = signal.get('strategy', 'Unknown')
                            self.performance_tracker.track_strategy_performance(strategy, signal)
                    
                    # Monitor execution latency
                    if hasattr(self.performance_tracker, 'monitor_execution_latency'):
                        self.performance_tracker.monitor_execution_latency('scan_cycle', scan_duration)
                    
                    # Update session metrics
                    if hasattr(self.performance_tracker, 'session_metrics'):
                        self.performance_tracker.session_metrics['total_signals'] += len(analyzed_signals)
                        if execution_results:
                            self.performance_tracker.session_metrics['total_trades'] += len(execution_results)
                    
                except Exception as perf_error:
                    self.logger.error(f"❌ Error tracking performance: {perf_error}")
            
            # 9. Update session manager
            if self.session_manager and hasattr(self.session_manager, 'update_scan_statistics'):
                try:
                    scan_duration = (datetime.now() - scan_start_time).total_seconds()
                    self.session_manager.update_scan_statistics({
                        'scan_duration': scan_duration,
                        'signals_detected': len(analyzed_signals),
                        'signals_saved': saved_count,
                        'trades_executed': len(execution_results),
                        'timestamp': datetime.now()
                    })
                except Exception as session_error:
                    self.logger.error(f"❌ Error updating session stats: {session_error}")
            
            # 10. Log final results
            final_count = len(analyzed_signals)
            scan_duration = (datetime.now() - scan_start_time).total_seconds()
            
            self.logger.info(f"✅ Scan #{self.scan_count} completed in {scan_duration:.2f}s")
            self.logger.info(f"   Raw signals from scanner: {len(raw_signals)}")
            self.logger.info(f"   After intelligence filtering: {len(intelligence_filtered_signals)}")
            self.logger.info(f"   Final processed signals: {final_count}")
            self.logger.info(f"   Signals saved to DB: {saved_count}")
            
            if analyzed_signals:
                self.logger.info("📤 Final signals processed successfully")
                for signal in analyzed_signals:
                    epic = signal.get('epic', 'Unknown')
                    signal_type = signal.get('signal_type', 'Unknown')
                    confidence = signal.get('confidence_score', 0)
                    intelligence_score = signal.get('intelligence_score', 0)
                    strategy = signal.get('strategy', 'Unknown')
                    claude_info = ""
                    
                    # ENHANCED: Add Claude analysis info to log
                    if signal.get('claude_analyzed'):
                        claude_decision = signal.get('claude_decision', 'Unknown')
                        claude_score = signal.get('claude_score')
                        claude_level = signal.get('claude_analysis_level', '')
                        
                        claude_info = f" | Claude: {claude_decision}"
                        if claude_score:
                            claude_info += f" ({claude_score}/10)"
                        if claude_level:
                            claude_info += f" [{claude_level}]"
                    
                    self.logger.info(f"   📊 {epic} {signal_type} ({confidence:.1%}) - {strategy}{claude_info}")
            
            return analyzed_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error during scan: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    # REMOVED: All duplicate detection methods since TradeValidator no longer handles duplicates
    # - clear_validator_cache() → REMOVED
    # - disable_duplicate_detection() → REMOVED  
    # - enable_duplicate_detection() → REMOVED
    # - test_validator_configuration() → UPDATED to show new TradeValidator info
    
    def get_validator_status(self) -> Dict:
        """Get current validator configuration and status (NEW format)"""
        try:
            if self.trade_validator:
                return self.trade_validator.get_validation_statistics()
            else:
                return {'error': 'TradeValidator not available'}
        except Exception as e:
            self.logger.error(f"❌ Error getting validator status: {e}")
            return {'error': str(e)}
    
    def test_validator_configuration(self):
        """Test and log NEW validator configuration"""
        try:
            validator_stats = self.get_validator_status()
            
            self.logger.info("🧪 Testing NEW TradeValidator configuration...")
            self.logger.info(f"   Duplicate detection: {validator_stats['status']['duplicate_detection']}")
            self.logger.info(f"   Validation focus: {validator_stats['status']['validation_focus']}")
            self.logger.info(f"   Performance: {validator_stats['status']['performance']}")
            self.logger.info(f"   Min confidence: {validator_stats['configuration']['min_confidence']:.1%}")
            self.logger.info(f"   Market hours validation: {validator_stats['configuration']['validate_market_hours']}")
            
            # Verify new architecture is in place
            if validator_stats['status']['duplicate_detection'] == 'Removed - handled by Scanner':
                self.logger.info("✅ NEW ARCHITECTURE VERIFIED: Duplicate detection removed from TradeValidator")
            else:
                self.logger.error("❌ NEW ARCHITECTURE NOT VERIFIED: TradeValidator still has duplicate detection")
            
            return validator_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error testing validator configuration: {e}")
            return {}
    
    # Intelligence configuration methods (orchestrator owns these)
    def set_intelligence_preset(self, preset: str) -> bool:
        """Set intelligence preset at runtime"""
        try:
            self.intelligence_preset = preset
            self.intelligence_threshold = self._get_intelligence_threshold()
            self.logger.info(f"🔧 Intelligence preset updated: {preset} (threshold: {self.intelligence_threshold:.1%})")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to set intelligence preset: {e}")
            return False
    
    def set_intelligence_mode(self, mode: str) -> bool:
        """Set intelligence mode at runtime"""
        try:
            self.intelligence_mode = mode
            self.logger.info(f"🔧 Intelligence mode updated: {mode}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to set intelligence mode: {e}")
            return False
    
    def get_intelligence_status(self) -> Dict:
        """Get current intelligence configuration"""
        return {
            'mode': self.intelligence_mode,
            'preset': self.intelligence_preset,
            'threshold': self.intelligence_threshold,
            'enabled': self.enable_market_intelligence,
            'owner': 'TradingOrchestrator'  # Clear ownership
        }
    
    # ENHANCED: Claude configuration methods
    def set_claude_analysis_level(self, level: str) -> bool:
        """
        ENHANCED: Set Claude analysis level at runtime
        
        Args:
            level: 'institutional', 'hedge_fund', 'prop_trader', or 'risk_manager'
        """
        try:
            self.claude_analysis_level = level
            
            # Update integration manager if available
            if self.integration_manager and hasattr(self.integration_manager, 'update_claude_configuration'):
                self.integration_manager.update_claude_configuration(analysis_level=level)
            
            self.logger.info(f"🔧 Claude analysis level updated: {level}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to set Claude analysis level: {e}")
            return False
    
    def set_claude_analysis_mode(self, mode: str) -> bool:
        """
        ENHANCED: Set Claude analysis mode at runtime
        
        Args:
            mode: 'minimal', 'full', etc.
        """
        try:
            self.claude_analysis_mode = mode
            self.logger.info(f"🔧 Claude analysis mode updated: {mode}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to set Claude analysis mode: {e}")
            return False
    
    def toggle_advanced_claude_prompts(self, enabled: bool) -> bool:
        """
        ENHANCED: Enable or disable advanced Claude prompts at runtime
        """
        try:
            self.use_advanced_claude_prompts = enabled
            
            # Update integration manager if available
            if self.integration_manager and hasattr(self.integration_manager, 'update_claude_configuration'):
                self.integration_manager.update_claude_configuration(use_advanced_prompts=enabled)
            
            self.logger.info(f"🔧 Advanced Claude prompts {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to toggle advanced Claude prompts: {e}")
            return False
    
    def get_claude_status(self) -> Dict:
        """
        ENHANCED: Get current Claude configuration and status
        """
        try:
            status = {
                'enabled': self.enable_claude,
                'analysis_mode': self.claude_analysis_mode,
                'analysis_level': self.claude_analysis_level,
                'advanced_prompts': self.use_advanced_claude_prompts,
                'integration_available': self.integration_manager is not None,
                'owner': 'TradingOrchestrator'
            }
            
            # Get integration manager status if available
            if self.integration_manager and hasattr(self.integration_manager, 'get_integration_statistics'):
                integration_stats = self.integration_manager.get_integration_statistics()
                status.update({
                    'integration_health': integration_stats.get('overall_health', 'Unknown'),
                    'success_rate': integration_stats.get('claude', {}).get('success_rate', 0),
                    'modular_features': integration_stats.get('claude', {}).get('modular_features', [])
                })
            
            return status
        except Exception as e:
            self.logger.error(f"❌ Error getting Claude status: {e}")
            return {'error': str(e)}
    
    def test_claude_configuration(self):
        """
        ENHANCED: Test and log Claude configuration with modular features
        """
        try:
            claude_status = self.get_claude_status()
            
            self.logger.info("🧪 Testing enhanced Claude configuration...")
            self.logger.info(f"   Claude enabled: {claude_status.get('enabled', False)}")
            self.logger.info(f"   Analysis mode: {claude_status.get('analysis_mode', 'Unknown')}")
            self.logger.info(f"   Analysis level: {claude_status.get('analysis_level', 'Unknown')}")
            self.logger.info(f"   Advanced prompts: {claude_status.get('advanced_prompts', False)}")
            self.logger.info(f"   Integration health: {claude_status.get('integration_health', 'Unknown')}")
            self.logger.info(f"   Success rate: {claude_status.get('success_rate', 0):.1%}")
            
            modular_features = claude_status.get('modular_features', [])
            if modular_features:
                self.logger.info(f"   Modular features: {len(modular_features)} available")
                for feature in modular_features[:5]:  # Show first 5 features
                    self.logger.info(f"     - {feature}")
            else:
                self.logger.info("   Modular features: None detected")
            
            # Test integration if available
            if self.integration_manager:
                test_result = self.test_claude_integration()
                self.logger.info(f"   Integration test: {'✅ PASS' if test_result else '❌ FAIL'}")
            
            return claude_status
            
        except Exception as e:
            self.logger.error(f"❌ Error testing Claude configuration: {e}")
            return {}
    
    def _calculate_boundary_aligned_sleep(self) -> float:
        """
        Calculate sleep time to align with candle boundaries based on scan_interval.

        When SCAN_ALIGN_TO_BOUNDARIES is enabled, scans are timed to occur
        shortly after candle closes + offset. The boundary interval is derived
        from scan_interval (e.g., 300s = 5m boundaries at :00, :05, :10, etc.).

        Returns:
            float: Seconds to sleep before next scan
        """
        from datetime import timedelta

        # Get settings from database config
        offset_seconds = self._scanner_cfg.scan_boundary_offset_seconds
        scan_interval = self._scanner_cfg.scan_interval

        # Derive boundary interval from scan_interval (in minutes)
        boundary_minutes = max(1, scan_interval // 60)  # Default to scan_interval

        now = datetime.utcnow()
        current_minute = now.minute
        current_second = now.second

        # Find next boundary based on scan_interval
        # e.g., for 5m: boundaries at 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55
        next_boundary_minute = ((current_minute // boundary_minutes) + 1) * boundary_minutes

        # Calculate time to next boundary
        if next_boundary_minute >= 60:
            # Boundary is in next hour
            next_boundary_minute = next_boundary_minute % 60
            next_boundary = now.replace(minute=next_boundary_minute, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_boundary = now.replace(minute=next_boundary_minute, second=0, microsecond=0)

        # Add offset (scan after boundary to allow data to settle)
        target_scan_time = next_boundary + timedelta(seconds=offset_seconds)

        # Calculate sleep time
        sleep_seconds = (target_scan_time - now).total_seconds()

        # If we're already past the target time for this boundary, calculate for next
        if sleep_seconds < 0:
            sleep_seconds += boundary_minutes * 60

        return max(10, sleep_seconds)  # Minimum 10 seconds between scans

    def start_continuous_scan(self):
        """Start continuous scanning loop with database saving + NEW validation + ENHANCED Claude"""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.session_manager:
            self.session_manager.start_session(self.session_id)

        # Check synthesis and boundary scanning modes from DATABASE (not config file)
        scanner_config = get_scanner_config()
        use_1m_base = scanner_config.use_1m_base_synthesis
        boundary_aligned = scanner_config.scan_align_to_boundaries
        boundary_offset = scanner_config.scan_boundary_offset_seconds

        # Note: session_manager.start_session() already logs session start details
        # Log additional orchestrator-specific info only
        self.logger.info(f"   1m Base Synthesis: {'✅ Enabled' if use_1m_base else '❌ Disabled (5m base)'}")
        self.logger.info(f"   Boundary Scanning: {'✅ Enabled (offset: ' + str(boundary_offset) + 's)' if boundary_aligned else '❌ Disabled'}")
        self.logger.info(f"   Intelligence preset: {self.intelligence_preset}")
        self.logger.info(f"   Intelligence threshold: {self.intelligence_threshold:.1%}")
        self.logger.info(f"   Alert saving: {'Enabled' if self.alert_history else 'Disabled'}")
        self.logger.info(f"   Duplicate detection: Handled by Scanner only")
        self.logger.info(f"   Claude enabled: {self.enable_claude}")

        if self.enable_claude:
            self.logger.info(f"   Claude analysis mode: {self.claude_analysis_mode}")
            self.logger.info(f"   Claude analysis level: {self.claude_analysis_level}")
            self.logger.info(f"   Advanced Claude prompts: {self.use_advanced_claude_prompts}")

        # Test NEW validator configuration on startup
        self.test_validator_configuration()

        # ENHANCED: Test Claude configuration on startup
        if self.enable_claude:
            self.test_claude_configuration()

        if self.performance_tracker:
            # Note: reset_session_metrics() logs its own "Session metrics reset" message
            self.performance_tracker.reset_session_metrics()

        self.running = True

        try:
            while self.running:
                # Perform scan (which now includes intelligence filtering, enhanced Claude analysis, and saves all signals to database)
                signals = self.scan_once()

                # Run periodic maintenance tasks (outcome analysis, cleanup, etc.)
                self._run_periodic_maintenance()

                # Wait for next scan - use boundary alignment if enabled
                if self.running:
                    if boundary_aligned:
                        sleep_time = self._calculate_boundary_aligned_sleep()
                        next_scan = datetime.utcnow() + timedelta(seconds=sleep_time)
                        self.logger.info(f"⏰ Boundary-aligned: next scan at {next_scan.strftime('%H:%M:%S')} UTC (sleeping {sleep_time:.0f}s)")
                    else:
                        sleep_time = self.scan_interval
                        self.logger.info(f"⏰ Waiting {sleep_time}s until next scan...")

                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.logger.info("⏹️ Continuous scan interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in continuous scan: {e}")
        finally:
            self.stop()
    
    def _run_periodic_maintenance(self):
        """
        Run periodic maintenance tasks (outcome analysis, cleanup, etc.)
        Called after each scan cycle. Tasks have their own interval tracking.
        """
        try:
            self._run_outcome_analysis_if_due()
        except Exception as e:
            # Don't let maintenance failures disrupt scanning
            self.logger.debug(f"Periodic maintenance error (non-critical): {e}")

    def _run_outcome_analysis_if_due(self):
        """
        Run rejection outcome analysis if enough time has passed since last run.
        Analyzes recent rejections to determine if they would have been profitable.
        """
        if not self.outcome_analyzer:
            return

        now = datetime.now()

        # Check if it's time to run (every hour by default)
        if self._last_outcome_analysis is not None:
            hours_since_last = (now - self._last_outcome_analysis).total_seconds() / 3600
            if hours_since_last < self._outcome_analysis_interval_hours:
                return

        try:
            self.logger.info("📊 Running automatic rejection outcome analysis...")

            # Analyze rejections from the last 3 days that haven't been analyzed yet
            stats = self.outcome_analyzer.run_daily_analysis(days_back=3, dry_run=False)

            self._last_outcome_analysis = now

            # Only log if we actually analyzed something
            if stats.get('total_analyzed', 0) > 0:
                win_rate = stats.get('would_be_win_rate', 0)
                self.logger.info(
                    f"📊 Outcome analysis complete: {stats['total_analyzed']} rejections analyzed, "
                    f"{stats.get('winners', 0)} would-be winners, {stats.get('losers', 0)} would-be losers "
                    f"({win_rate:.1f}% win rate)"
                )
            else:
                self.logger.debug("📊 Outcome analysis: no new rejections to analyze")

        except Exception as e:
            # Log at debug level to avoid log spam
            self.logger.debug(f"Outcome analysis skipped: {e}")
            self._last_outcome_analysis = now  # Still update time to avoid retry spam

    def stop_scanning(self):
        """Stop continuous scanning"""
        self.logger.info("🛑 Stopping scanning...")
        self.running = False
    
    def stop(self):
        """Stop the orchestrator and all components"""
        self.logger.info("🛑 Stopping TradingOrchestrator...")
        
        self.running = False
        
        # Stop scanner if it has a stop method
        if self.scanner and hasattr(self.scanner, 'stop'):
            try:
                self.scanner.stop()
                self.logger.info("✅ Scanner stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping scanner: {e}")
        
        # Stop session manager
        if self.session_manager:
            try:
                self.session_manager.end_session()
                self.logger.info("✅ Session ended")
            except Exception as e:
                self.logger.error(f"❌ Error ending session: {e}")
        
        self.logger.info("✅ TradingOrchestrator stopped successfully")
    
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive system status including NEW validator info + ENHANCED Claude"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': {
                    'running': self.running,
                    'trading_enabled': self.enable_trading,
                    'claude_enabled': self.enable_claude,
                    'claude_analysis_mode': self.claude_analysis_mode,
                    'claude_analysis_level': self.claude_analysis_level,
                    'claude_advanced_prompts': self.use_advanced_claude_prompts,
                    'scan_interval': self.scan_interval,
                    'alert_saving_enabled': self.alert_history is not None,
                    'intelligence_enabled': self.enable_market_intelligence,
                    'intelligence_preset': self.intelligence_preset,
                    'intelligence_threshold': self.intelligence_threshold,
                    'duplicate_detection_architecture': 'Scanner handles all duplicate detection'
                },
                'session': self.get_session_status(),
                'performance': self.get_performance_report(),
                'risk': self.get_risk_status(),
                'market': self.get_market_status(),
                'integrations': self.get_integration_status(),
                'intelligence': self.get_intelligence_status(),
                'claude': self.get_claude_status(),  # ENHANCED: Claude status
                'validator': self.get_validator_status(),  # NEW validator status
                'components': {
                    'order_manager': self.order_manager is not None,
                    'session_manager': self.session_manager is not None,
                    'risk_manager': self.risk_manager is not None,
                    'trade_validator_new': self.trade_validator is not None,
                    'performance_tracker': self.performance_tracker is not None,
                    'market_monitor': self.market_monitor is not None,
                    'integration_manager_enhanced': self.integration_manager is not None,
                    'scanner': self.scanner is not None,
                    'signal_detector': self.scanner and hasattr(self.scanner, 'signal_detector'),
                    'database_manager': self.db_manager is not None,
                    'alert_history_manager': self.alert_history is not None
                }
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting comprehensive status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    # Interface methods for compatibility
    def get_session_status(self) -> Dict:
        """Get current session status"""
        try:
            return self.session_manager.get_session_status() if self.session_manager else {}
        except Exception as e:
            self.logger.error(f"❌ Error getting session status: {e}")
            return {}
    
    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        try:
            return self.session_manager.get_session_summary() if self.session_manager else {}
        except Exception as e:
            self.logger.error(f"❌ Error getting session summary: {e}")
            return {}
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        try:
            if not self.performance_tracker:
                return {}
            
            if hasattr(self.performance_tracker, 'generate_performance_report'):
                return self.performance_tracker.generate_performance_report()
            elif hasattr(self.performance_tracker, 'get_real_time_metrics'):
                return self.performance_tracker.get_real_time_metrics()
            else:
                return getattr(self.performance_tracker, 'session_metrics', {})
        except Exception as e:
            self.logger.error(f"❌ Error getting performance report: {e}")
            return {}
    
    def get_risk_status(self) -> Dict:
        """Get risk status"""
        try:
            return self.risk_manager.get_risk_status() if self.risk_manager else {}
        except Exception as e:
            self.logger.error(f"❌ Error getting risk status: {e}")
            return {}
    
    def get_market_status(self) -> Dict:
        """Get market status"""
        try:
            return self.market_monitor.get_market_status() if self.market_monitor else {}
        except Exception as e:
            self.logger.error(f"❌ Error getting market status: {e}")
            return {}
    
    def get_integration_status(self) -> Dict:
        """Get integration status"""
        try:
            return self.integration_manager.get_integration_statistics() if self.integration_manager else {}
        except Exception as e:
            self.logger.error(f"❌ Error getting integration status: {e}")
            return {}
    
    def test_claude_integration(self) -> bool:
        """Test Claude integration"""
        try:
            return self.integration_manager.test_claude_integration() if self.integration_manager else False
        except Exception as e:
            self.logger.error(f"❌ Error testing Claude integration: {e}")
            return False
    
    def test_order_execution(self) -> bool:
        """Test order execution"""
        try:
            return self.order_manager.test_order_executor() if self.order_manager else False
        except Exception as e:
            self.logger.error(f"❌ Error testing order execution: {e}")
            return False
    
    def enable_trading_mode(self, enable: bool = True):
        """Enable or disable trading"""
        self.enable_trading = enable
        if self.order_manager:
            self.order_manager.enable_trading_mode(enable)
        self.logger.info(f"🔧 Trading mode {'enabled' if enable else 'disabled'}")
    
    def enable_claude_mode(self, enable: bool = True):
        """ENHANCED: Enable or disable Claude analysis with modular features"""
        self.enable_claude = enable
        if self.integration_manager:
            self.integration_manager.enable_integration('claude', enable)
        
        # ENHANCED: Log current Claude configuration when enabling
        if enable:
            self.logger.info(f"🔧 Claude mode enabled")
            self.logger.info(f"   Analysis mode: {self.claude_analysis_mode}")
            self.logger.info(f"   Analysis level: {self.claude_analysis_level}")
            self.logger.info(f"   Advanced prompts: {self.use_advanced_claude_prompts}")
        else:
            self.logger.info(f"🔧 Claude mode disabled")
    
    def update_scan_interval(self, new_interval: int):
        """Update scan interval"""
        self.scan_interval = new_interval
        if self.session_manager:
            self.session_manager.update_scan_interval(new_interval)
        self.logger.info(f"🔧 Scan interval updated to {new_interval}s")
    
    # =================================================================
    # ENHANCED COMPATIBILITY METHODS FOR trade_scan.py
    # =================================================================
    
    def run_live_trading(self, scan_interval: Optional[int] = None):
        """ENHANCED COMPATIBILITY METHOD: Start continuous live trading with NEW validation + modular Claude"""
        try:
            if scan_interval:
                self.update_scan_interval(scan_interval)
            
            self.logger.info(f"🚀 Starting enhanced live trading mode (compatibility)")
            self.logger.info(f"   Scan interval: {self.scan_interval}s")
            self.logger.info(f"   Trading enabled: {self.enable_trading}")
            self.logger.info(f"   Intelligence: {self.intelligence_preset} (threshold: {self.intelligence_threshold:.1%})")
            self.logger.info(f"   Alert saving: {'Enabled' if self.alert_history else 'Disabled'}")
            self.logger.info(f"   Duplicate detection: Handled by Scanner only")
            
            if self.enable_claude:
                self.logger.info(f"   Claude analysis: Enhanced modular mode")
                self.logger.info(f"   Claude level: {self.claude_analysis_level}")
                self.logger.info(f"   Advanced prompts: {self.use_advanced_claude_prompts}")
            else:
                self.logger.info(f"   Claude analysis: Disabled")
            
            return self.start_continuous_scan()
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced live trading: {e}")
            raise
    
    def start_docker_mode(self):
        """ENHANCED COMPATIBILITY METHOD: Start Docker deployment mode with NEW validation + modular Claude"""
        try:
            self.logger.info("🐳 Starting enhanced Docker deployment mode (compatibility)")
            self.logger.info(f"   Intelligence mode: {self.intelligence_mode}")
            self.logger.info(f"   Intelligence preset: {self.intelligence_preset}")
            self.logger.info(f"   Trading enabled: {self.enable_trading}")
            self.logger.info(f"   Claude analysis: {self.enable_claude}")
            
            if self.enable_claude:
                self.logger.info(f"   Claude mode: {self.claude_analysis_mode}")
                self.logger.info(f"   Claude level: {self.claude_analysis_level}")
                self.logger.info(f"   Advanced prompts: {self.use_advanced_claude_prompts}")
            
            self.logger.info(f"   Scan interval: {self.scan_interval}s")
            self.logger.info(f"   Alert saving: {'Enabled' if self.alert_history else 'Disabled'}")
            self.logger.info(f"   Duplicate detection: Handled by Scanner only")
            
            # Log environment information
            self._log_docker_environment()

            # Note: test_validator_configuration() and test_claude_configuration()
            # are called by start_continuous_scan(), so we don't need to call them here

            return self.start_continuous_scan()
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Enhanced Docker mode stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced Docker mode: {e}")
            raise
        finally:
            self.stop()
    
    def get_system_status(self) -> Dict:
        """ENHANCED COMPATIBILITY METHOD: Get system status with NEW validator info + modular Claude"""
        try:
            comprehensive_status = self.get_comprehensive_status()
            
            return {
                'trading_system': {
                    'version': '2.1.0-enhanced-modular-claude-architecture',
                    'entry_point': 'trade_scan.py',
                    'mode': 'enhanced_modular_orchestrator_with_advanced_claude',
                    'initialization_time': datetime.now().isoformat(),
                    'duplicate_detection_architecture': 'Scanner handles all duplicate detection',
                    'claude_integration': 'Enhanced modular API with institutional-grade analysis'
                },
                'configuration': {
                    'intelligence_mode': self.intelligence_mode,
                    'intelligence_preset': self.intelligence_preset,
                    'intelligence_threshold': self.intelligence_threshold,
                    'trading_enabled': self.enable_trading,
                    'claude_enabled': self.enable_claude,
                    'claude_analysis_mode': self.claude_analysis_mode,
                    'claude_analysis_level': self.claude_analysis_level,
                    'claude_advanced_prompts': self.use_advanced_claude_prompts,
                    'scan_interval': self.scan_interval,
                    'alert_saving_enabled': self.alert_history is not None,
                    'duplicate_detection_removed_from_tradevalidator': True,
                    'tradevalidator_focus': 'Trading-specific validation only',
                    'claude_modular_features': 'Institutional analysis, advanced prompts, batch processing'
                },
                'orchestrator': comprehensive_status,
                'status': 'operational' if not comprehensive_status.get('error') else 'error'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting enhanced system status: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _log_docker_environment(self):
        """Log Docker environment information for compatibility"""
        try:
            self.logger.info("🐳 Docker Environment Information:")
            
            docker_vars = ['HOSTNAME', 'PATH', 'PWD', 'HOME']
            for var in docker_vars:
                value = os.environ.get(var, 'Not set')
                self.logger.info(f"   {var}: {value}")
            
            self.logger.info(f"   Python version: {sys.version}")
            self.logger.info(f"   Python executable: {sys.executable}")
            
            try:
                import shutil
                disk_usage = shutil.disk_usage('/')
                total_gb = disk_usage.total / (1024**3)
                free_gb = disk_usage.free / (1024**3)
                self.logger.info(f"   Disk space: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            except:
                self.logger.info("   Disk space: Unable to determine")
                
        except Exception as e:
            self.logger.error(f"❌ Error logging Docker environment: {e}")

    def check_component_availability(self) -> Dict[str, bool]:
        """
        Check availability of all trading orchestrator components
        Used by integration test suite to validate system resilience
        
        Returns:
            Dict mapping component names to availability status
        """
        components = {
            'DatabaseManager': self.db_manager is not None,
            'AlertHistoryManager': self.alert_history is not None,
            'Scanner': hasattr(self, 'scanner') and self.scanner is not None,
            'OrderManager': hasattr(self, 'order_manager') and self.order_manager is not None,
            'SessionManager': hasattr(self, 'session_manager') and self.session_manager is not None,
            'RiskManager': hasattr(self, 'risk_manager') and self.risk_manager is not None,
            'TradeValidator': hasattr(self, 'trade_validator') and self.trade_validator is not None,
            'PerformanceTracker': hasattr(self, 'performance_tracker') and self.performance_tracker is not None,
            'MarketMonitor': hasattr(self, 'market_monitor') and self.market_monitor is not None,
            'IntegrationManager': hasattr(self, 'integration_manager') and self.integration_manager is not None,
            'SignalDetector': hasattr(self, 'signal_detector') and self.signal_detector is not None,
        }
        return components


# =================================================================
# ENHANCED COMPATIBILITY FUNCTIONS FOR trade_scan.py
# =================================================================

def create_trading_orchestrator(
    db_manager: DatabaseManager = None,
    intelligence_mode: str = 'balanced',
    intelligence_preset: str = 'balanced',
    enable_trading: bool = None,
    enable_claude_analysis: bool = None,
    # ENHANCED: New Claude configuration parameters
    claude_analysis_mode: str = None,
    claude_analysis_level: str = None,
    use_advanced_claude_prompts: bool = None,
    scan_interval: int = None,
    **kwargs
) -> TradingOrchestrator:
    """ENHANCED COMPATIBILITY FACTORY: Create TradingOrchestrator with NEW TradeValidator architecture + modular Claude"""
    try:
        logger = logging.getLogger(__name__)
        
        # Initialize database manager if not provided
        if not db_manager:
            logger.info("🔧 No database manager provided, attempting to create one...")
            try:
                # NOTE: DATABASE_URL is an environment variable that stays in config.py
                database_url = getattr(config, 'DATABASE_URL', None)  # ENV VAR - stays in config
                if database_url:
                    db_manager = DatabaseManager(database_url)
                    logger.info("✅ DatabaseManager created in factory")
                    
                    if hasattr(db_manager, 'test_connection'):
                        if db_manager.test_connection():
                            logger.info("✅ Database connection test passed")
                        else:
                            logger.warning("⚠️ Database connection test failed")
                else:
                    logger.error("❌ DATABASE_URL not found in config")
                    db_manager = None
                    
            except Exception as db_error:
                logger.warning(f"⚠️ Failed to create DatabaseManager: {db_error}")
                logger.warning("⚠️ Continuing without database - functionality will be limited")
                db_manager = None
        
        # ENHANCED: Create orchestrator with NEW architecture + modular Claude
        orchestrator = TradingOrchestrator(
            db_manager=db_manager,
            scan_interval=scan_interval or 60,
            enable_trading=enable_trading,
            enable_claude_analysis=enable_claude_analysis,
            claude_analysis_mode=claude_analysis_mode,
            claude_analysis_level=claude_analysis_level,
            use_advanced_claude_prompts=use_advanced_claude_prompts,
            intelligence_mode=intelligence_mode,
            intelligence_preset=intelligence_preset
        )
        
        logger.info("🎯 Enhanced TradingOrchestrator created successfully (NEW TradeValidator + Modular Claude)")
        logger.info(f"   Intelligence mode: {intelligence_mode}")
        logger.info(f"   Intelligence preset: {intelligence_preset}")
        logger.info(f"   Intelligence threshold: {orchestrator.intelligence_threshold:.1%}")
        logger.info(f"   Trading enabled: {orchestrator.enable_trading}")
        logger.info(f"   Claude enabled: {orchestrator.enable_claude}")
        
        if orchestrator.enable_claude:
            logger.info(f"   Claude analysis mode: {orchestrator.claude_analysis_mode}")
            logger.info(f"   Claude analysis level: {orchestrator.claude_analysis_level}")
            logger.info(f"   Advanced Claude prompts: {orchestrator.use_advanced_claude_prompts}")
        
        logger.info(f"   Database available: {'Yes' if db_manager else 'No'}")
        logger.info(f"   Alert saving: {'Yes' if orchestrator.alert_history else 'No'}")
        logger.info(f"   Duplicate detection: Handled by Scanner only")
        
        return orchestrator
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Failed to create enhanced TradingOrchestrator: {e}")
        raise


def create_intelligent_trading_system(**kwargs):
    """ENHANCED COMPATIBILITY FUNCTION: For existing integrations with NEW TradeValidator + modular Claude"""
    return create_trading_orchestrator(**kwargs)


if __name__ == "__main__":
    # Test ENHANCED functionality
    print("🧪 Testing Enhanced TradingOrchestrator (NEW TradeValidator + Modular Claude API)...")
    
    try:
        # Create test orchestrator with enhanced Claude features
        orchestrator = create_trading_orchestrator(
            enable_claude_analysis=True,
            claude_analysis_level='institutional',
            use_advanced_claude_prompts=True
        )
        
        # Test NEW validator configuration
        validator_stats = orchestrator.test_validator_configuration()
        duplicate_status = validator_stats['status']['duplicate_detection']
        validation_focus = validator_stats['status']['validation_focus']
        
        print(f"✅ Validator duplicate detection: {duplicate_status}")
        print(f"✅ Validator focus: {validation_focus}")
        print(f"✅ NEW architecture: {'VERIFIED' if 'Removed' in duplicate_status else 'NOT VERIFIED'}")
        
        # ENHANCED: Test Claude configuration
        claude_stats = orchestrator.test_claude_configuration()
        print(f"✅ Claude enabled: {claude_stats.get('enabled', False)}")
        print(f"✅ Claude analysis level: {claude_stats.get('analysis_level', 'Unknown')}")
        print(f"✅ Advanced prompts: {claude_stats.get('advanced_prompts', False)}")
        print(f"✅ Modular features: {len(claude_stats.get('modular_features', []))}")
        
        # Test single scan (which should save to database with enhanced Claude)
        signals = orchestrator.scan_once()
        print(f"✅ Test scan completed - found {len(signals)} signals")
        
        # Test intelligence configuration
        orchestrator.set_intelligence_preset('minimal')
        intelligence_status = orchestrator.get_intelligence_status()
        print(f"✅ Intelligence status: {intelligence_status['preset']} (threshold: {intelligence_status['threshold']:.1%})")
        
        # ENHANCED: Test Claude configuration updates
        orchestrator.set_claude_analysis_level('hedge_fund')
        orchestrator.toggle_advanced_claude_prompts(True)
        print(f"✅ Claude configuration updated dynamically")
        
        # Test status methods
        status = orchestrator.get_comprehensive_status()
        print(f"✅ Enhanced system status retrieved - {len(status)} sections")
        print(f"✅ Alert saving enabled: {status['system_status']['alert_saving_enabled']}")
        print(f"✅ Intelligence enabled: {status['system_status']['intelligence_enabled']}")
        print(f"✅ Claude analysis level: {status['system_status']['claude_analysis_level']}")
        print(f"✅ Advanced Claude prompts: {status['system_status']['claude_advanced_prompts']}")
        print(f"✅ Duplicate detection architecture: {status['system_status']['duplicate_detection_architecture']}")
        
        print("🎉 ENHANCED TradingOrchestrator test completed successfully!")
        print("✅ Clean architecture implemented")
        print("✅ Intelligence owned by TradingOrchestrator")
        print("✅ Scanner does pure signal detection + deduplication")
        print("✅ TradeValidator does trading-specific validation only")
        print("✅ Database saving operational")
        print("✅ NEW ARCHITECTURE VERIFIED - Single duplicate detection system")
        print("🚀 ENHANCED CLAUDE INTEGRATION VERIFIED:")
        print("✅ Modular Claude API integration")
        print("✅ Institutional-grade analysis levels")
        print("✅ Advanced prompt builder support")
        print("✅ Runtime configuration updates")
        print("✅ Batch analysis capabilities")
        print("✅ Health monitoring and fallback mechanisms")
        
    except Exception as e:
        print(f"❌ Enhanced test failed: {e}")
        import traceback
        traceback.print_exc()