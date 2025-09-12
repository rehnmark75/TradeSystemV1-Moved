# core/trading/__init__.py
"""
Trading module for order execution, session management, and orchestration
Refactored modular architecture with focused components
"""

from .trading_orchestrator import TradingOrchestrator, create_trading_orchestrator
from .order_manager import OrderManager
from .session_manager import SessionManager
from .risk_manager import RiskManager
from .trade_validator import TradeValidator
from .performance_tracker import PerformanceTracker
from .market_monitor import MarketMonitor
from .integration_manager import IntegrationManager

__all__ = [
    'TradingOrchestrator',
    'create_trading_orchestrator',
    'OrderManager',
    'SessionManager', 
    'RiskManager',
    'TradeValidator',
    'PerformanceTracker',
    'MarketMonitor',
    'IntegrationManager'
]

# Version info
__version__ = '2.0.0'
__description__ = 'Modular trading system with comprehensive risk management and monitoring'

# Module status
MODULES_STATUS = {
    'order_manager': 'Available - Order execution and position management',
    'session_manager': 'Available - Session lifecycle and statistics',
    'risk_manager': 'Available - Risk calculations and validation',
    'trade_validator': 'Available - Signal validation and filtering',
    'performance_tracker': 'Available - Performance monitoring and analytics',
    'market_monitor': 'Available - Market conditions and trading environment',
    'integration_manager': 'Available - External integrations and APIs',
    'trading_orchestrator': 'Available - Main coordination and orchestration'
}

def get_module_status():
    """Get status of all trading modules"""
    return MODULES_STATUS.copy()

def validate_trading_modules():
    """Validate that all trading modules can be imported"""
    import_status = {}
    
    try:
        from .order_manager import OrderManager
        import_status['order_manager'] = True
    except ImportError as e:
        import_status['order_manager'] = f"Failed: {e}"
    
    try:
        from .session_manager import SessionManager
        import_status['session_manager'] = True
    except ImportError as e:
        import_status['session_manager'] = f"Failed: {e}"
    
    try:
        from .risk_manager import RiskManager
        import_status['risk_manager'] = True
    except ImportError as e:
        import_status['risk_manager'] = f"Failed: {e}"
    
    try:
        from .trade_validator import TradeValidator
        import_status['trade_validator'] = True
    except ImportError as e:
        import_status['trade_validator'] = f"Failed: {e}"
    
    try:
        from .performance_tracker import PerformanceTracker
        import_status['performance_tracker'] = True
    except ImportError as e:
        import_status['performance_tracker'] = f"Failed: {e}"
    
    try:
        from .market_monitor import MarketMonitor
        import_status['market_monitor'] = True
    except ImportError as e:
        import_status['market_monitor'] = f"Failed: {e}"
    
    try:
        from .integration_manager import IntegrationManager
        import_status['integration_manager'] = True
    except ImportError as e:
        import_status['integration_manager'] = f"Failed: {e}"
    
    try:
        from .trading_orchestrator import TradingOrchestrator, create_trading_orchestrator
        import_status['trading_orchestrator'] = True
    except ImportError as e:
        import_status['trading_orchestrator'] = f"Failed: {e}"
    
    return import_status

def create_complete_trading_system(db_manager=None, **kwargs):
    """
    Create a complete trading system with all components
    
    Args:
        db_manager: Database manager instance
        **kwargs: Additional configuration parameters
        
    Returns:
        TradingOrchestrator instance with all components initialized
    """
    try:
        # Create orchestrator with all components
        orchestrator = create_trading_orchestrator(
            db_manager=db_manager,
            **kwargs
        )
        
        return orchestrator
        
    except Exception as e:
        raise RuntimeError(f"Failed to create complete trading system: {e}")

# Quick access functions for individual components
def create_order_manager(**kwargs):
    """Create standalone OrderManager"""
    return OrderManager(**kwargs)

def create_session_manager(**kwargs):
    """Create standalone SessionManager"""
    return SessionManager(**kwargs)

def create_risk_manager(**kwargs):
    """Create standalone RiskManager"""
    return RiskManager(**kwargs)

def create_trade_validator(**kwargs):
    """Create standalone TradeValidator"""
    return TradeValidator(**kwargs)

def create_performance_tracker(**kwargs):
    """Create standalone PerformanceTracker"""
    return PerformanceTracker(**kwargs)

def create_market_monitor(**kwargs):
    """Create standalone MarketMonitor"""
    return MarketMonitor(**kwargs)

def create_integration_manager(**kwargs):
    """Create standalone IntegrationManager"""
    return IntegrationManager(**kwargs)

# Configuration helpers
def get_default_trading_config():
    """Get default configuration for trading system"""
    return {
        'enable_trading': False,
        'enable_claude': False,
        'enable_notifications': True,
        'scan_interval': 60,
        'max_daily_loss': 1000.0,
        'max_positions': 5,
        'default_risk_percent': 2.0,
        'min_confidence_threshold': 70,
        'cooldown_minutes': 15,
        'validate_market_hours': True
    }

def get_trading_system_info():
    """Get comprehensive trading system information"""
    import datetime
    
    module_status = validate_trading_modules()
    successful_imports = sum(1 for status in module_status.values() if status is True)
    total_modules = len(module_status)
    
    return {
        'version': __version__,
        'description': __description__,
        'total_modules': total_modules,
        'successful_imports': successful_imports,
        'import_success_rate': f"{successful_imports/total_modules*100:.1f}%",
        'module_status': module_status,
        'modules_available': MODULES_STATUS,
        'system_ready': successful_imports == total_modules,
        'timestamp': datetime.datetime.now().isoformat()
    }

# Module documentation
def print_trading_system_help():
    """Print help information for the trading system"""
    print("🏭 Trading System - Modular Architecture")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print()
    
    print("📦 Available Modules:")
    for module, description in MODULES_STATUS.items():
        print(f"  • {module}: {description}")
    print()
    
    print("🚀 Quick Start:")
    print("  from core.trading import create_complete_trading_system")
    print("  trading_system = create_complete_trading_system()")
    print("  trading_system.scan_once()")
    print()
    
    print("🔧 Individual Components:")
    print("  from core.trading import OrderManager, RiskManager")
    print("  order_manager = OrderManager()")
    print("  risk_manager = RiskManager()")
    print()
    
    print("📊 System Information:")
    print("  from core.trading import get_trading_system_info")
    print("  info = get_trading_system_info()")
    print("  print(info)")

if __name__ == "__main__":
    print_trading_system_help()