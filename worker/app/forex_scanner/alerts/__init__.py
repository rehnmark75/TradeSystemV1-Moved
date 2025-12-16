# alerts/__init__.py
"""
Complete Alerts Module Interface
Provides access to all alert components with new modular Claude API
Maintains backward compatibility for all existing imports
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# ===== CLAUDE API COMPONENTS =====
# Import order: Try new modular first, fallback to old
try:
    # Try to import new modular components
    from .claude_analyzer import ClaudeAnalyzer as ModularClaudeAnalyzer
    from .api.client import APIClient
    from .validation.signal_validator import SignalValidator
    from .analysis.prompt_builder import PromptBuilder
    from .storage.result_formatter import ResultFormatter
    MODULAR_AVAILABLE = True
    logger.info("✅ Modular Claude API components loaded")
except ImportError as e:
    # Fallback to old implementation
    try:
        from .claude_api import ClaudeAnalyzer as LegacyClaudeAnalyzer
        MODULAR_AVAILABLE = False
        logger.warning(f"⚠️ Using legacy Claude API: {e}")
    except ImportError:
        logger.error("❌ No Claude API implementation available")
        LegacyClaudeAnalyzer = None
        MODULAR_AVAILABLE = False

# ===== OTHER ALERT COMPONENTS =====
# Import existing alert components with error handling
try:
    from .alert_history import AlertHistoryManager
    ALERT_HISTORY_AVAILABLE = True
    logger.debug("✅ AlertHistoryManager loaded")
except ImportError as e:
    AlertHistoryManager = None
    ALERT_HISTORY_AVAILABLE = False
    logger.warning(f"⚠️ AlertHistoryManager not available: {e}")

try:
    from .notifications import NotificationManager
    NOTIFICATIONS_AVAILABLE = True
    logger.debug("✅ NotificationManager loaded")
except ImportError as e:
    NotificationManager = None
    NOTIFICATIONS_AVAILABLE = False
    logger.warning(f"⚠️ NotificationManager not available: {e}")

try:
    from .order_executor import OrderExecutor
    ORDER_EXECUTOR_AVAILABLE = True
    logger.debug("✅ OrderExecutor loaded")
except ImportError as e:
    OrderExecutor = None
    ORDER_EXECUTOR_AVAILABLE = False
    logger.warning(f"⚠️ OrderExecutor not available: {e}")

# ===== UNIFIED CLAUDE ANALYZER =====
class ClaudeAnalyzer:
    """
    Unified Claude Analyzer Interface
    Automatically uses modular implementation if available, falls back to legacy
    """

    def __init__(self, api_key: str = None, auto_save: bool = True, save_directory: str = "claude_analysis", data_fetcher=None):
        self.api_key = api_key
        self.auto_save = auto_save
        self.save_directory = save_directory
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)

        # Initialize the appropriate implementation
        if MODULAR_AVAILABLE:
            self._init_modular_implementation()
        else:
            self._init_legacy_implementation()

    def _init_modular_implementation(self):
        """Initialize new modular implementation"""
        try:
            self.analyzer = ModularClaudeAnalyzer(
                api_key=self.api_key,
                auto_save=self.auto_save,
                save_directory=self.save_directory,
                data_fetcher=self.data_fetcher
            )
            self.implementation = "modular"
            self.logger.info("🚀 Using modular Claude API implementation")
        except Exception as e:
            self.logger.error(f"❌ Modular initialization failed: {e}")
            self._init_legacy_implementation()

    def _init_legacy_implementation(self):
        """Initialize legacy implementation"""
        try:
            if LegacyClaudeAnalyzer:
                self.analyzer = LegacyClaudeAnalyzer(
                    api_key=self.api_key,
                    auto_save=self.auto_save,
                    save_directory=self.save_directory
                )
                self.implementation = "legacy"
                self.logger.info("⚠️ Using legacy Claude API implementation")
            else:
                raise Exception("No Claude implementation available")
        except Exception as e:
            self.logger.error(f"❌ Legacy initialization failed: {e}")
            self.analyzer = None
            self.implementation = "none"
    
    # Delegate all methods to the underlying implementation
    def analyze_signal_minimal(self, signal: Dict, save_to_file: bool = True) -> Optional[Dict]:
        """Analyze signal with minimal validation"""
        if self.analyzer:
            return self.analyzer.analyze_signal_minimal(signal, save_to_file)
        return None
    
    def analyze_signal(self, signal: Dict) -> Optional[str]:
        """Analyze signal (legacy method)"""
        if self.analyzer:
            return self.analyzer.analyze_signal(signal)
        return None
    
    def test_connection(self) -> bool:
        """Test Claude API connection"""
        if self.analyzer:
            return self.analyzer.test_connection()
        return False
    
    def batch_analyze_signals_minimal(self, signals: list, save_to_file: bool = True) -> list:
        """Batch analyze signals"""
        if self.analyzer:
            return self.analyzer.batch_analyze_signals_minimal(signals, save_to_file)
        return []
    
    # Pass through any other method calls to the underlying analyzer
    def __getattr__(self, name):
        """Pass through any other method calls to the underlying analyzer"""
        if self.analyzer and hasattr(self.analyzer, name):
            return getattr(self.analyzer, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # New modular-specific methods (only available if modular is loaded)
    def get_health_status(self) -> Dict[str, Any]:
        """Get API health status (modular only)"""
        if self.implementation == "modular" and hasattr(self.analyzer, 'get_health_status'):
            return self.analyzer.get_health_status()
        return {"status": "unknown", "implementation": self.implementation}
    
    def validate_signal_structure(self, signal: Dict) -> Dict[str, Any]:
        """Validate signal structure (modular only)"""
        if self.implementation == "modular" and hasattr(self.analyzer, 'validate_signal_structure'):
            return self.analyzer.validate_signal_structure(signal)
        return {"valid": True, "validation_type": "none"}
    
    def format_analysis_result(self, signal: Dict, analysis: Dict, output_format: str = "TEXT") -> str:
        """Format analysis result (modular only)"""
        if self.implementation == "modular" and hasattr(self.analyzer, 'format_analysis_result'):
            return self.analyzer.format_analysis_result(signal, analysis, output_format)
        return str(analysis)


# ===== FACTORY FUNCTIONS =====
# Factory functions for easier instantiation
def create_claude_analyzer(api_key: str = None, auto_save: bool = True) -> ClaudeAnalyzer:
    """Create Claude analyzer (backward compatible)"""
    return ClaudeAnalyzer(api_key=api_key, auto_save=auto_save)

def create_minimal_claude_analyzer(api_key: str = None, auto_save: bool = True) -> ClaudeAnalyzer:
    """Create minimal Claude analyzer (backward compatible)"""
    return ClaudeAnalyzer(api_key=api_key, auto_save=auto_save)

def create_alert_history_manager(db_manager):
    """Create alert history manager if available"""
    if ALERT_HISTORY_AVAILABLE:
        return AlertHistoryManager(db_manager)
    return None

def create_notification_manager():
    """Create notification manager if available"""
    if NOTIFICATIONS_AVAILABLE:
        return NotificationManager()
    return None

def create_order_executor():
    """Create order executor if available"""
    if ORDER_EXECUTOR_AVAILABLE:
        return OrderExecutor()
    return None

# ===== UTILITY FUNCTIONS =====
# Quick utility functions (backward compatible)
def quick_signal_check(signal: Dict, api_key: str = None) -> bool:
    """Quick function with technical validation"""
    try:
        analyzer = create_claude_analyzer(api_key, auto_save=False)
        result = analyzer.analyze_signal_minimal(signal, save_to_file=False)
        return result.get('approved', False) if result else False
    except Exception as e:
        logging.getLogger(__name__).error(f"Quick signal check failed: {e}")
        return False

def batch_check_signals(signals: list, api_key: str = None) -> list:
    """Batch check with technical validation"""
    try:
        analyzer = create_claude_analyzer(api_key, auto_save=False)
        results = analyzer.batch_analyze_signals_minimal(signals, save_to_file=False)
        return [result.get('approved', False) for result in results]
    except Exception as e:
        logging.getLogger(__name__).error(f"Batch signal check failed: {e}")
        return [False] * len(signals)

# ===== STATUS FUNCTIONS =====
def get_alerts_module_status() -> Dict[str, Any]:
    """Get status of all alerts module components"""
    return {
        "claude_api": {
            "modular_available": MODULAR_AVAILABLE,
            "legacy_available": LegacyClaudeAnalyzer is not None,
            "implementation": "modular" if MODULAR_AVAILABLE else "legacy" if LegacyClaudeAnalyzer else "none"
        },
        "alert_history": {
            "available": ALERT_HISTORY_AVAILABLE,
            "class": AlertHistoryManager.__name__ if ALERT_HISTORY_AVAILABLE else None
        },
        "notifications": {
            "available": NOTIFICATIONS_AVAILABLE,
            "class": NotificationManager.__name__ if NOTIFICATIONS_AVAILABLE else None
        },
        "order_executor": {
            "available": ORDER_EXECUTOR_AVAILABLE,
            "class": OrderExecutor.__name__ if ORDER_EXECUTOR_AVAILABLE else None
        }
    }

def print_alerts_status():
    """Print status of alerts module components"""
    status = get_alerts_module_status()
    print("\n🚨 Alerts Module Status:")
    print("=" * 50)
    
    # Claude API status
    claude_status = status["claude_api"]
    impl = claude_status["implementation"]
    if impl == "modular":
        print("✅ Claude API: Modular (Latest)")
    elif impl == "legacy":
        print("⚠️ Claude API: Legacy (Functional)")
    else:
        print("❌ Claude API: Not Available")
    
    # Other components
    for component, info in status.items():
        if component != "claude_api":
            symbol = "✅" if info["available"] else "❌"
            name = component.replace("_", " ").title()
            print(f"{symbol} {name}: {'Available' if info['available'] else 'Not Available'}")
    
    print("=" * 50)

# ===== MODULE INFORMATION =====
__version__ = "2.0.0"
__implementation__ = "modular" if MODULAR_AVAILABLE else "legacy"

# Export main classes and functions
__all__ = [
    # Core Classes
    'ClaudeAnalyzer',
    'AlertHistoryManager', 
    'NotificationManager',
    'OrderExecutor',
    
    # Factory Functions
    'create_claude_analyzer', 
    'create_minimal_claude_analyzer',
    'create_alert_history_manager',
    'create_notification_manager',
    'create_order_executor',
    
    # Utility Functions
    'quick_signal_check',
    'batch_check_signals',
    
    # Status Functions
    'get_alerts_module_status',
    'print_alerts_status'
]