# core/__init__.py
"""
Core functionality for Forex Scanner
Exposes all core modules and classes for easy importing
Enhanced with auto-trading and alert system support
NOW WITH SMART MONEY STRATEGIES (Phase 1 & 2)
"""

# Core modules
from .database import DatabaseManager
from .data_fetcher import DataFetcher
from .scanner import ForexScanner, IntelligentForexScanner
from .signal_detector import SignalDetector

# Strategy modules - Original strategies
from .strategies.base_strategy import BaseStrategy
from .strategies.ema_strategy import EMAStrategy
from .strategies.macd_strategy import MACDStrategy
from .strategies.combined_strategy import CombinedStrategy
from .strategies.scalping_strategy import ScalpingStrategy

# NEW: Smart Money Strategy modules (Phase 1 & 2)
def get_smart_money_ema_strategy():
    """Lazy load smart money EMA strategy if available"""
    try:
        from .strategies.smart_money_ema_strategy import SmartMoneyEMAStrategy
        return SmartMoneyEMAStrategy
    except ImportError as e:
        print(f"⚠️ Smart Money EMA Strategy not available: {e}")
        return None

def get_smart_money_macd_strategy():
    """Lazy load smart money MACD strategy if available"""
    try:
        from .strategies.smart_money_macd_strategy import SmartMoneyMACDStrategy
        return SmartMoneyMACDStrategy
    except ImportError as e:
        print(f"⚠️ Smart Money MACD Strategy not available: {e}")
        return None

# NEW: Smart Money Intelligence modules
def get_market_structure_analyzer():
    """Lazy load market structure analyzer if available"""
    try:
        from .intelligence.market_structure_analyzer import MarketStructureAnalyzer
        return MarketStructureAnalyzer
    except ImportError as e:
        print(f"⚠️ Market Structure Analyzer not available: {e}")
        return None

def get_order_flow_analyzer():
    """Lazy load order flow analyzer if available"""
    try:
        from .intelligence.order_flow_analyzer import OrderFlowAnalyzer
        return OrderFlowAnalyzer
    except ImportError as e:
        print(f"⚠️ Order Flow Analyzer not available: {e}")
        return None

# Detection utilities
from .detection.price_adjuster import PriceAdjuster
from .detection.market_conditions import MarketConditionsAnalyzer

# Backtesting modules
from .backtest.backtest_engine import BacktestEngine
from .backtest.performance_analyzer import PerformanceAnalyzer
from .backtest.signal_analyzer import SignalAnalyzer

# Alert and Trading modules (lazy imports for optional components)
def get_claude_analyzer():
    """Lazy load Claude analyzer if available"""
    try:
        # Use the new unified interface from alerts/__init__.py
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

# Utility modules
def get_timezone_manager():
    """Lazy load timezone manager if available"""
    try:
        # Try different possible import paths
        try:
            from utils.timezone_utils import TimezoneManager
            return TimezoneManager
        except ImportError:
            from core.utils.timezone_utils import TimezoneManager
            return TimezoneManager
    except ImportError as e:
        print(f"⚠️ Timezone manager not available: {e}")
        return None

# Version info - Updated for smart money integration
__version__ = "2.0.0"  # Updated for Smart Money Phase 1 & 2
__author__ = "Forex Scanner Team"

# Define what gets imported with "from core import *"
__all__ = [
    # Core classes
    'DatabaseManager',
    'DataFetcher', 
    'ForexScanner',
    'IntelligentForexScanner',
    'SignalDetector',
    
    # Strategy classes - Original
    'BaseStrategy',
    'EMAStrategy',
    'MACDStrategy', 
    'CombinedStrategy',
    'ScalpingStrategy',
    
    # Detection utilities
    'PriceAdjuster',
    'MarketConditionsAnalyzer',
    
    # Backtesting classes
    'BacktestEngine',
    'PerformanceAnalyzer',
    'SignalAnalyzer',
    
    # Lazy loader functions - Original
    'get_claude_analyzer',
    'get_notification_manager',
    'get_alert_history_manager',
    'get_order_executor',
    'get_timezone_manager',
    
    # NEW: Smart Money lazy loaders
    'get_smart_money_ema_strategy',
    'get_smart_money_macd_strategy',
    'get_market_structure_analyzer',
    'get_order_flow_analyzer',
    
    # Utility functions - Enhanced
    'get_available_strategies',
    'get_smart_money_strategies',  # NEW
    'get_scalping_modes',
    'validate_core_imports',
    'validate_smart_money_imports',  # NEW
    'print_core_status',
    'print_smart_money_status',  # NEW
    'create_enhanced_scanner',
    'create_smart_money_scanner',  # NEW
    
    # Module info
    '__version__',
    '__author__'
]

# Module-level convenience functions - Enhanced
def get_available_strategies():
    """Get list of available trading strategies"""
    strategies = [
        'EMAStrategy',
        'MACDStrategy', 
        'CombinedStrategy',
        'ScalpingStrategy'
    ]
    
    # Add smart money strategies if available
    if get_smart_money_ema_strategy():
        strategies.append('SmartMoneyEMAStrategy')
    if get_smart_money_macd_strategy():
        strategies.append('SmartMoneyMACDStrategy')
    
    return strategies

# NEW: Get smart money strategies
def get_smart_money_strategies():
    """Get list of available smart money strategies"""
    smart_strategies = []
    
    if get_smart_money_ema_strategy():
        smart_strategies.append('SmartMoneyEMAStrategy')
    if get_smart_money_macd_strategy():
        smart_strategies.append('SmartMoneyMACDStrategy')
    
    return smart_strategies

def get_scalping_modes():
    """Get available scalping modes"""
    try:
        import config
        if hasattr(config, 'SCALPING_STRATEGY_CONFIG'):
            return list(config.SCALPING_STRATEGY_CONFIG.keys())
        else:
            return ['ultra_fast', 'aggressive', 'conservative', 'dual_ma']
    except ImportError:
        return ['ultra_fast', 'aggressive', 'conservative', 'dual_ma']

# NEW: Create smart money enhanced scanner
def create_smart_money_scanner(intelligence_mode='backtest_consistent', enable_structure=True, enable_order_flow=True, **kwargs):
    """
    Create a scanner with smart money strategies enabled
    
    Args:
        intelligence_mode: Intelligence filtering mode
        enable_structure: Enable market structure analysis
        enable_order_flow: Enable order flow analysis  
        **kwargs: Additional scanner configuration
    
    Returns:
        Configured scanner with smart money strategies
    """
    try:
        import config
        
        # Set smart money configuration
        if enable_structure:
            config.USE_SMART_MONEY_EMA = True
            config.SMART_MONEY_STRUCTURE_VALIDATION = True
        
        if enable_order_flow:
            config.USE_SMART_MONEY_MACD = True
            config.SMART_MONEY_ORDER_FLOW_VALIDATION = True
        
        # Create enhanced scanner with smart money
        scanner = create_enhanced_scanner(
            intelligence_mode=intelligence_mode,
            **kwargs
        )
        
        print(f"🧠 Smart money scanner created successfully")
        print(f"   Market structure analysis: {'✅' if enable_structure else '❌'}")
        print(f"   Order flow analysis: {'✅' if enable_order_flow else '❌'}")
        
        return scanner
        
    except Exception as e:
        print(f"❌ Failed to create smart money scanner: {e}")
        # Fallback to regular enhanced scanner
        print("🔄 Falling back to regular enhanced scanner...")
        return create_enhanced_scanner(intelligence_mode=intelligence_mode, **kwargs)

def create_enhanced_scanner(intelligence_mode='backtest_consistent', auto_trading=False, **kwargs):
    """
    Create an enhanced scanner with all components properly initialized
    
    Args:
        intelligence_mode: Intelligence filtering mode ('disabled', 'backtest_consistent', 'live_only', 'enhanced')
        auto_trading: Enable auto-trading functionality (sets ENABLE_ORDER_EXECUTION)
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
        
        # Prepare scanner parameters (only use parameters that the scanner actually accepts)
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
        
        # Add any additional kwargs that are valid for the scanner
        valid_kwargs = ['scan_interval', 'min_confidence', 'spread_pips', 'use_bid_adjustment', 'user_timezone']
        for key, value in kwargs.items():
            if key in valid_kwargs:
                scanner_params[key] = value
        
        # Create scanner with enhanced functionality
        scanner = IntelligentForexScanner(**scanner_params)
        
        print(f"✅ Enhanced scanner created successfully")
        print(f"   Intelligence mode: {intelligence_mode}")
        print(f"   Auto-trading: {auto_trading}")
        print(f"   Epic count: {len(scanner_params['epic_list'])}")
        print(f"   Min confidence: {scanner_params['min_confidence']:.1%}")
        
        return scanner
        
    except Exception as e:
        print(f"❌ Failed to create enhanced scanner: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

# NEW: Validate smart money imports
def validate_smart_money_imports():
    """Validate that smart money components are working"""
    import_status = {}
    
    # Test smart money strategies
    SmartMoneyEMAStrategy = get_smart_money_ema_strategy()
    import_status['SmartMoneyEMAStrategy'] = SmartMoneyEMAStrategy is not None
    
    SmartMoneyMACDStrategy = get_smart_money_macd_strategy()
    import_status['SmartMoneyMACDStrategy'] = SmartMoneyMACDStrategy is not None
    
    # Test smart money analyzers
    MarketStructureAnalyzer = get_market_structure_analyzer()
    import_status['MarketStructureAnalyzer'] = MarketStructureAnalyzer is not None
    
    OrderFlowAnalyzer = get_order_flow_analyzer()
    import_status['OrderFlowAnalyzer'] = OrderFlowAnalyzer is not None
    
    # Test smart money integration
    try:
        from .intelligence import is_smart_money_available
        import_status['SmartMoneyIntegration'] = is_smart_money_available()
    except ImportError:
        import_status['SmartMoneyIntegration'] = False
    
    return import_status

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
        from .scanner import ForexScanner
        import_status['ForexScanner'] = True
    except ImportError as e:
        import_status['ForexScanner'] = str(e)
    
    try:
        from .scanner import IntelligentForexScanner
        import_status['IntelligentForexScanner'] = True
    except ImportError as e:
        import_status['IntelligentForexScanner'] = str(e)
    
    try:
        from .signal_detector import SignalDetector
        import_status['SignalDetector'] = True
    except ImportError as e:
        import_status['SignalDetector'] = str(e)
    
    # Test strategy imports
    try:
        from .strategies.ema_strategy import EMAStrategy
        import_status['EMAStrategy'] = True
    except ImportError as e:
        import_status['EMAStrategy'] = str(e)
    
    try:
        from .strategies.macd_strategy import MACDStrategy
        import_status['MACDStrategy'] = True
    except ImportError as e:
        import_status['MACDStrategy'] = str(e)
    
    try:
        from .strategies.combined_strategy import CombinedStrategy
        import_status['CombinedStrategy'] = True
    except ImportError as e:
        import_status['CombinedStrategy'] = str(e)
    
    try:
        from .strategies.scalping_strategy import ScalpingStrategy
        import_status['ScalpingStrategy'] = True
    except ImportError as e:
        import_status['ScalpingStrategy'] = str(e)
    
    # Test detection utilities
    try:
        from .detection.price_adjuster import PriceAdjuster
        import_status['PriceAdjuster'] = True
    except ImportError as e:
        import_status['PriceAdjuster'] = str(e)
    
    try:
        from .detection.market_conditions import MarketConditionsAnalyzer
        import_status['MarketConditionsAnalyzer'] = True
    except ImportError as e:
        import_status['MarketConditionsAnalyzer'] = str(e)
    
    # Test backtest imports
    try:
        from .backtest.backtest_engine import BacktestEngine
        import_status['BacktestEngine'] = True
    except ImportError as e:
        import_status['BacktestEngine'] = str(e)
    
    try:
        from .backtest.performance_analyzer import PerformanceAnalyzer
        import_status['PerformanceAnalyzer'] = True
    except ImportError as e:
        import_status['PerformanceAnalyzer'] = str(e)
    
    try:
        from .backtest.signal_analyzer import SignalAnalyzer
        import_status['SignalAnalyzer'] = True
    except ImportError as e:
        import_status['SignalAnalyzer'] = str(e)
    
    # Test optional alert/trading components
    ClaudeAnalyzer = get_claude_analyzer()
    import_status['ClaudeAnalyzer'] = ClaudeAnalyzer is not None
    
    NotificationManager = get_notification_manager()
    import_status['NotificationManager'] = NotificationManager is not None
    
    AlertHistoryManager = get_alert_history_manager()
    import_status['AlertHistoryManager'] = AlertHistoryManager is not None
    
    OrderExecutor = get_order_executor()
    import_status['OrderExecutor'] = OrderExecutor is not None
    
    TimezoneManager = get_timezone_manager()
    import_status['TimezoneManager'] = TimezoneManager is not None
    
    return import_status

def validate_trading_components():
    """Validate that auto-trading components are available"""
    trading_status = {}
    
    # Check order executor
    OrderExecutor = get_order_executor()
    trading_status['OrderExecutor'] = OrderExecutor is not None
    
    # Check notification manager
    NotificationManager = get_notification_manager()
    trading_status['NotificationManager'] = NotificationManager is not None
    
    # Check alert history
    AlertHistoryManager = get_alert_history_manager()
    trading_status['AlertHistoryManager'] = AlertHistoryManager is not None
    
    # Check Claude integration
    ClaudeAnalyzer = get_claude_analyzer()
    trading_status['ClaudeAnalyzer'] = ClaudeAnalyzer is not None
    
    # Check configuration
    try:
        import config
        trading_status['ENABLE_ORDER_EXECUTION'] = getattr(config, 'ENABLE_ORDER_EXECUTION', False)
        trading_status['CLAUDE_API_KEY'] = hasattr(config, 'CLAUDE_API_KEY') and getattr(config, 'CLAUDE_API_KEY', None) is not None
        trading_status['DATABASE_URL'] = hasattr(config, 'DATABASE_URL') and getattr(config, 'DATABASE_URL', None) is not None
        trading_status['CONFIG_AVAILABLE'] = True
    except ImportError:
        trading_status['CONFIG_AVAILABLE'] = False
    
    return trading_status

# NEW: Print smart money status
def print_smart_money_status():
    """Print status of smart money components"""
    print("\n🧠 Smart Money Component Status:")
    print("=" * 50)
    
    status = validate_smart_money_imports()
    
    for component, result in status.items():
        if result is True:
            print(f"✅ {component}")
        else:
            print(f"❌ {component}")
    
    successful_imports = sum(1 for result in status.values() if result is True)
    total_imports = len(status)
    
    print("=" * 50)
    print(f"📊 Smart Money Success Rate: {successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)")
    
    if successful_imports == total_imports:
        print("🎉 All smart money components available!")
        print("💡 You can now use:")
        print("   - SmartMoneyEMAStrategy (Market Structure + Order Flow)")
        print("   - SmartMoneyMACDStrategy (Order Flow Confluence)")
        print("   - MarketStructureAnalyzer (BOS, ChoCh, Swing Points)")
        print("   - OrderFlowAnalyzer (Order Blocks, FVGs, Supply/Demand)")
    elif successful_imports > 0:
        print("⚠️  Some smart money components available - partial functionality")
    else:
        print("❌ Smart money components not available - using regular strategies only")

def print_core_status():
    """Print status of all core module imports"""
    print("🔍 Core Module Import Status:")
    print("=" * 50)
    
    status = validate_core_imports()
    
    for module, result in status.items():
        if result is True:
            print(f"✅ {module}")
        else:
            print(f"❌ {module}: {result}")
    
    successful_imports = sum(1 for result in status.values() if result is True)
    total_imports = len(status)
    
    print("=" * 50)
    print(f"📊 Success Rate: {successful_imports}/{total_imports} ({successful_imports/total_imports*100:.1f}%)")
    
    if successful_imports == total_imports:
        print("🎉 All core modules imported successfully!")
    else:
        print("⚠️  Some modules failed to import. Check file structure and dependencies.")

def print_trading_status():
    """Print status of auto-trading components"""
    print("\n💰 Auto-Trading Component Status:")
    print("=" * 50)
    
    status = validate_trading_components()
    
    for component, result in status.items():
        if result is True:
            print(f"✅ {component}")
        elif result is False:
            print(f"❌ {component}")
        else:
            print(f"⚠️  {component}: {result}")
    
    # Overall trading readiness
    required_components = ['AlertHistoryManager']  # Minimum requirement
    optional_components = ['OrderExecutor', 'NotificationManager', 'ClaudeAnalyzer']
    
    core_ready = all(status.get(comp, False) for comp in required_components)
    
    print("=" * 50)
    if core_ready:
        available_optional = sum(1 for comp in optional_components if status.get(comp, False))
        print(f"🚀 Core components ready! ({available_optional}/{len(optional_components)} optional components available)")
    else:
        print("⚠️  Core trading setup incomplete - check missing components")

def test_enhanced_scanner():
    """Test enhanced scanner creation and functionality"""
    print("\n🧪 Testing Enhanced Scanner:")
    print("=" * 50)
    
    try:
        # Test scanner creation
        print("📝 Creating enhanced scanner...")
        scanner = create_enhanced_scanner(intelligence_mode='disabled')
        print("✅ Enhanced scanner created successfully")
        
        # Test basic functionality
        print("📊 Getting scanner status...")
        status = scanner.get_scanner_status()
        print(f"✅ Scanner status: {len(status)} parameters")
        
        # Print key status info
        print(f"   Running: {status.get('running', 'Unknown')}")
        print(f"   Epic count: {status.get('epic_count', 'Unknown')}")
        print(f"   Intelligence mode: {status.get('intelligence_mode', 'Unknown')}")
        print(f"   Min confidence: {status.get('min_confidence', 'Unknown')}")
        
        # Test single scan
        print("🔍 Testing single scan...")
        signals = scanner.scan_once()
        print(f"✅ Scan completed: {len(signals)} signals found")
        
        if signals:
            for i, signal in enumerate(signals[:3]):  # Show first 3 signals
                epic = signal.get('epic', 'Unknown')
                signal_type = signal.get('signal_type', 'Unknown')
                confidence = signal.get('confidence_score', 0)
                print(f"   Signal {i+1}: {signal_type} {epic} ({confidence:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced scanner test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

# NEW: Test smart money scanner
def test_smart_money_scanner():
    """Test smart money scanner creation and functionality"""
    print("\n🧠 Testing Smart Money Scanner:")
    print("=" * 50)
    
    try:
        # Check if smart money components are available
        smart_status = validate_smart_money_imports()
        available_components = sum(1 for result in smart_status.values() if result is True)
        
        if available_components == 0:
            print("❌ No smart money components available - skipping test")
            return False
        
        print(f"📝 Creating smart money scanner ({available_components}/4 components available)...")
        scanner = create_smart_money_scanner(intelligence_mode='disabled')
        print("✅ Smart money scanner created successfully")
        
        # Test smart money specific functionality
        print("🧠 Testing smart money features...")
        
        # This would test smart money specific methods if available
        # For now, just test that it works as a regular scanner
        signals = scanner.scan_once()
        print(f"✅ Smart money scan completed: {len(signals)} signals found")
        
        # Check if any signals have smart money data
        smart_signals = [s for s in signals if s.get('smart_money_validated', False)]
        print(f"🧠 Smart money validated signals: {len(smart_signals)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart money scanner test failed: {e}")
        print("🔄 Smart money components may not be fully integrated yet")
        return False

def create_scanner_factory():
    """
    Create a scanner factory function for different use cases
    """
    def scanner_factory(mode='basic', **kwargs):
        """
        Factory function to create different types of scanners
        
        Args:
            mode: 'basic', 'intelligent', 'auto_trading', 'backtest', 'smart_money'
            **kwargs: Additional configuration
        """
        if mode == 'basic':
            return create_enhanced_scanner(intelligence_mode='disabled', **kwargs)
        elif mode == 'intelligent':
            return create_enhanced_scanner(intelligence_mode='backtest_consistent', **kwargs)
        elif mode == 'auto_trading':
            return create_enhanced_scanner(intelligence_mode='live_only', auto_trading=True, **kwargs)
        elif mode == 'backtest':
            return create_enhanced_scanner(intelligence_mode='disabled', **kwargs)
        elif mode == 'smart_money':  # NEW
            return create_smart_money_scanner(**kwargs)
        else:
            raise ValueError(f"Unknown scanner mode: {mode}")
    
    return scanner_factory

# Create the factory function
create_scanner = create_scanner_factory()

# Module initialization check
if __name__ == "__main__":
    print("🚀 FOREX SCANNER CORE MODULE STATUS v2.0.0")
    print("=" * 80)
    
    # Test core imports
    print_core_status()
    
    # Test smart money components
    print_smart_money_status()
    
    # Test trading components
    print_trading_status()
    
    # Test enhanced scanner
    test_enhanced_scanner()
    
    # Test smart money scanner
    test_smart_money_scanner()
    
    print("\n" + "=" * 80)
    print("🎯 USAGE EXAMPLES:")
    print("=" * 80)
    print()
    print("# Basic scanner creation")
    print("from core import create_enhanced_scanner")
    print("scanner = create_enhanced_scanner()")
    print()
    print("# Smart money scanner")
    print("from core import create_smart_money_scanner")
    print("smart_scanner = create_smart_money_scanner()")
    print()
    print("# Auto-trading enabled")
    print("scanner = create_enhanced_scanner(auto_trading=True)")
    print()
    print("# Using the factory function")
    print("from core import create_scanner")
    print("basic_scanner = create_scanner('basic')")
    print("smart_scanner = create_scanner('smart_money')")
    print("trading_scanner = create_scanner('auto_trading')")
    print()
    print("# Direct smart money component access")
    print("from core import get_smart_money_ema_strategy, get_market_structure_analyzer")
    print("SmartEMA = get_smart_money_ema_strategy()")
    print("StructureAnalyzer = get_market_structure_analyzer()")
    print()
    print("# Check what's available")
    print("from core import get_available_strategies, get_smart_money_strategies")
    print("all_strategies = get_available_strategies()")
    print("smart_strategies = get_smart_money_strategies()")
    
    print("\n" + "=" * 80)
    print("🎉 CORE MODULE INITIALIZATION COMPLETE v2.0.0")
    print("🧠 Smart Money Phase 1 & 2 Integration Ready!")
    print("=" * 80)