# core/__init__.py
"""
Core functionality for Forex Scanner

NOTE: After January 2026 cleanup, only SMC Simple strategy is active.
Legacy strategies have been archived to forex_scanner/archive/disabled_strategies/
"""

# Core modules
from .database import DatabaseManager
from .data_fetcher import DataFetcher
from .scanner import ForexScanner, IntelligentForexScanner
from .signal_detector import SignalDetector

# Strategy modules - Only active strategies
from .strategies import (
    BaseStrategy,
    SMCSimpleStrategy,
    create_smc_simple_strategy,
)

# Detection utilities
from .detection.price_adjuster import PriceAdjuster
from .detection.market_conditions import MarketConditionsAnalyzer

# Backtesting modules (BacktestEngine archived - use BacktestTradingOrchestrator)
from .backtest.performance_analyzer import PerformanceAnalyzer
from .backtest.signal_analyzer import SignalAnalyzer

# Alert and Trading modules (lazy imports for optional components)
def get_claude_analyzer():
    """Lazy load Claude analyzer if available"""
    try:
        from alerts import ClaudeAnalyzer
        return ClaudeAnalyzer
    except ImportError as e:
        print(f"⚠️ Claude analyzer not available: {e}")
        return None

def get_notification_manager():
    """Lazy load notification manager if available"""
    try:
        from alerts import NotificationManager
        return NotificationManager
    except ImportError as e:
        print(f"⚠️ Notification manager not available: {e}")
        return None

def get_alert_history_manager():
    """Lazy load alert history manager if available"""
    try:
        from alerts import AlertHistoryManager
        return AlertHistoryManager
    except ImportError as e:
        print(f"⚠️ Alert history manager not available: {e}")
        return None

def get_order_executor():
    """Lazy load order executor if available"""
    try:
        from alerts import OrderExecutor
        return OrderExecutor
    except ImportError as e:
        print(f"⚠️ Order executor not available: {e}")
        return None

def get_timezone_manager():
    """Lazy load timezone manager if available"""
    try:
        try:
            from utils.timezone_utils import TimezoneManager
            return TimezoneManager
        except ImportError:
            from core.utils.timezone_utils import TimezoneManager
            return TimezoneManager
    except ImportError as e:
        print(f"⚠️ Timezone manager not available: {e}")
        return None

# Version info
__version__ = "3.0.0"  # Updated for January 2026 cleanup
__author__ = "Forex Scanner Team"

# Define what gets imported with "from core import *"
__all__ = [
    # Core classes
    'DatabaseManager',
    'DataFetcher',
    'ForexScanner',
    'IntelligentForexScanner',
    'SignalDetector',

    # Strategy classes - Only active
    'BaseStrategy',
    'SMCSimpleStrategy',
    'create_smc_simple_strategy',

    # Detection utilities
    'PriceAdjuster',
    'MarketConditionsAnalyzer',

    # Backtesting classes (BacktestEngine archived)
    'PerformanceAnalyzer',
    'SignalAnalyzer',

    # Lazy loader functions
    'get_claude_analyzer',
    'get_notification_manager',
    'get_alert_history_manager',
    'get_order_executor',
    'get_timezone_manager',

    # Utility functions
    'get_available_strategies',
    'validate_core_imports',
    'print_core_status',
    'create_enhanced_scanner',

    # Module info
    '__version__',
    '__author__'
]

# Module-level convenience functions
def get_available_strategies():
    """Get list of available trading strategies"""
    return ['SMCSimpleStrategy']

def create_enhanced_scanner(intelligence_mode='backtest_consistent', auto_trading=False, **kwargs):
    """
    Create an enhanced scanner with all components properly initialized

    Args:
        intelligence_mode: Intelligence filtering mode
        auto_trading: Enable auto-trading functionality
        **kwargs: Additional scanner configuration

    Returns:
        Configured IntelligentForexScanner instance
    """
    try:
        import config

        # Initialize database manager
        db_manager = DatabaseManager(config.DATABASE_URL)

        # Override config for auto-trading if requested
        if auto_trading:
            config.ENABLE_ORDER_EXECUTION = True

        # Prepare scanner parameters
        scanner_params = {
            'db_manager': db_manager,
            'epic_list': getattr(config, 'EPIC_LIST', ['CS.D.EURUSD.CEEM.IP']),
            'intelligence_mode': intelligence_mode,
            'scan_interval': getattr(config, 'SCAN_INTERVAL', 60),
            'min_confidence': getattr(config, 'MIN_CONFIDENCE', 0.6),
            'spread_pips': getattr(config, 'SPREAD_PIPS', 1.5),
            'use_bid_adjustment': getattr(config, 'USE_BID_ADJUSTMENT', True),
            'user_timezone': getattr(config, 'USER_TIMEZONE', 'Europe/Stockholm')
        }

        # Add any additional valid kwargs
        valid_kwargs = ['scan_interval', 'min_confidence', 'spread_pips', 'use_bid_adjustment', 'user_timezone']
        for key, value in kwargs.items():
            if key in valid_kwargs:
                scanner_params[key] = value

        # Create scanner
        scanner = IntelligentForexScanner(**scanner_params)

        print(f"✅ Enhanced scanner created successfully")
        print(f"   Intelligence mode: {intelligence_mode}")
        print(f"   Auto-trading: {auto_trading}")
        print(f"   Epic count: {len(scanner_params['epic_list'])}")

        return scanner

    except Exception as e:
        print(f"❌ Failed to create enhanced scanner: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

def validate_core_imports():
    """Validate that all core imports are working"""
    import_status = {}

    # Test core imports
    try:
        from .database import DatabaseManager
        import_status['DatabaseManager'] = True
    except ImportError as e:
        import_status['DatabaseManager'] = str(e)

    try:
        from .data_fetcher import DataFetcher
        import_status['DataFetcher'] = True
    except ImportError as e:
        import_status['DataFetcher'] = str(e)

    try:
        from .scanner import ForexScanner, IntelligentForexScanner
        import_status['ForexScanner'] = True
        import_status['IntelligentForexScanner'] = True
    except ImportError as e:
        import_status['ForexScanner'] = str(e)
        import_status['IntelligentForexScanner'] = str(e)

    try:
        from .signal_detector import SignalDetector
        import_status['SignalDetector'] = True
    except ImportError as e:
        import_status['SignalDetector'] = str(e)

    # Test SMC Simple strategy
    try:
        from .strategies import SMCSimpleStrategy
        import_status['SMCSimpleStrategy'] = True
    except ImportError as e:
        import_status['SMCSimpleStrategy'] = str(e)

    # Test detection utilities
    try:
        from .detection.price_adjuster import PriceAdjuster
        import_status['PriceAdjuster'] = True
    except ImportError as e:
        import_status['PriceAdjuster'] = str(e)

    # Test backtest imports
    try:
        from .backtest.performance_analyzer import PerformanceAnalyzer
        import_status['PerformanceAnalyzer'] = True
    except ImportError as e:
        import_status['PerformanceAnalyzer'] = str(e)

    # Test optional components
    import_status['ClaudeAnalyzer'] = get_claude_analyzer() is not None
    import_status['NotificationManager'] = get_notification_manager() is not None
    import_status['AlertHistoryManager'] = get_alert_history_manager() is not None
    import_status['OrderExecutor'] = get_order_executor() is not None

    return import_status

def print_core_status():
    """Print status of all core components"""
    print("\n" + "=" * 60)
    print("FOREX SCANNER CORE STATUS (v3.0.0 - January 2026 Cleanup)")
    print("=" * 60)

    import_status = validate_core_imports()

    print("\n📦 Core Components:")
    for component, status in import_status.items():
        if status is True:
            print(f"   ✅ {component}")
        elif status is False:
            print(f"   ❌ {component} (not available)")
        else:
            print(f"   ⚠️ {component}: {status}")

    print("\n📈 Available Strategies:")
    for strategy in get_available_strategies():
        print(f"   • {strategy}")

    print("\n" + "=" * 60)
