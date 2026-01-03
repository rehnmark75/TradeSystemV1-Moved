# core/intelligence/__init__.py
"""
Market Intelligence Package
Advanced market analysis and adaptive strategy selection
NOW WITH SMART MONEY ANALYSIS (Phase 1 & 2)
"""

from .market_intelligence import MarketIntelligenceEngine

# NEW: Add configurable intelligence to solve zero-alert problem
try:
    from .intelligence_config import IntelligenceConfigManager
    CONFIGURABLE_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONFIGURABLE_INTELLIGENCE_AVAILABLE = False
    # Create placeholder class if config not available
    class IntelligenceConfigManager:
        def __init__(self, *args, **kwargs):
            pass

# NEW: Smart Money Components (Phase 1 & 2)
try:
    from .market_structure_analyzer import MarketStructureAnalyzer
    from .order_flow_analyzer import OrderFlowAnalyzer
    SMART_MONEY_AVAILABLE = True
except ImportError:
    SMART_MONEY_AVAILABLE = False
    # Create placeholder classes if smart money not available
    class MarketStructureAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze_market_structure(self, *args, **kwargs):
            return {'error': 'Smart money analysis not available'}
        def validate_signal_against_structure(self, *args, **kwargs):
            return {'structure_aligned': True, 'structure_score': 0.5}
    
    class OrderFlowAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
        def analyze_order_flow(self, *args, **kwargs):
            return {'error': 'Order flow analysis not available'}
        def validate_signal_against_order_flow(self, *args, **kwargs):
            return {'order_flow_aligned': True, 'order_flow_score': 0.5}

# Package metadata
__version__ = "2.0.0"  # Updated for smart money integration
__author__ = "Forex Scanner Team"
__description__ = "Advanced market intelligence and regime analysis system with smart money concepts"

# Export main classes for easy importing
__all__ = [
    'MarketIntelligenceEngine',
    'IntelligenceConfigManager',
    'MarketStructureAnalyzer',  # NEW: Phase 1
    'OrderFlowAnalyzer',        # NEW: Phase 2
]

# Package-level logging setup
import logging

logger = logging.getLogger(__name__)
logger.info("📊 Market Intelligence package loaded v2.0.0")

if CONFIGURABLE_INTELLIGENCE_AVAILABLE:
    logger.info("✅ Configurable intelligence thresholds available")
else:
    logger.warning("⚠️ Configurable intelligence not available - using fixed thresholds")

if SMART_MONEY_AVAILABLE:
    logger.info("🧠 Smart money analysis available (Market Structure + Order Flow)")
else:
    logger.warning("⚠️ Smart money analysis not available - using placeholder classes")

# Validation function to check if intelligence is available
def is_intelligence_available() -> bool:
    """Check if market intelligence functionality is available"""
    try:
        # Test if required dependencies are available
        import pandas as pd
        import numpy as np
        from scipy import stats
        return True
    except ImportError as e:
        logger.warning(f"⚠️ Market intelligence dependencies missing: {e}")
        return False

# Check if configurable intelligence is available
def is_configurable_intelligence_available() -> bool:
    """Check if configurable intelligence is available"""
    return CONFIGURABLE_INTELLIGENCE_AVAILABLE

# NEW: Check if smart money analysis is available
def is_smart_money_available() -> bool:
    """Check if smart money analysis is available"""
    return SMART_MONEY_AVAILABLE

# Quick factory function
def create_intelligence_engine(data_fetcher=None):
    """
    Factory function to create a MarketIntelligenceEngine
    
    Args:
        data_fetcher: DataFetcher instance, if None will try to create one
        
    Returns:
        MarketIntelligenceEngine instance
    """
    if data_fetcher is None:
        try:
            from ..data_fetcher import DataFetcher
            from ..database import DatabaseManager
            import config
            
            db_manager = DatabaseManager(config.DATABASE_URL)
            data_fetcher = DataFetcher(db_manager)
            logger.info("✅ Created DataFetcher for intelligence engine")
            
        except Exception as e:
            logger.error(f"❌ Failed to create DataFetcher: {e}")
            raise
    
    return MarketIntelligenceEngine(data_fetcher)

# Factory function for configurable intelligence
def create_configurable_intelligence(preset: str = 'minimal'):
    """
    Factory function to create configurable intelligence manager
    
    Args:
        preset: Intelligence preset ('disabled', 'minimal', 'balanced', 'advanced', 'testing')
        
    Returns:
        IntelligenceConfigManager instance
    """
    if not CONFIGURABLE_INTELLIGENCE_AVAILABLE:
        logger.error("❌ Configurable intelligence not available")
        return None
    
    return IntelligenceConfigManager(preset)

# NEW: Factory functions for smart money components
def create_market_structure_analyzer():
    """
    Factory function to create MarketStructureAnalyzer
    
    Returns:
        MarketStructureAnalyzer instance
    """
    if not SMART_MONEY_AVAILABLE:
        logger.warning("⚠️ Smart money analysis not available - returning placeholder")
        return MarketStructureAnalyzer()  # Placeholder that returns safe defaults
    
    try:
        analyzer = MarketStructureAnalyzer()
        logger.info("✅ Market Structure Analyzer created")
        return analyzer
    except Exception as e:
        logger.error(f"❌ Failed to create Market Structure Analyzer: {e}")
        return MarketStructureAnalyzer()  # Return placeholder on error

def create_order_flow_analyzer():
    """
    Factory function to create OrderFlowAnalyzer
    
    Returns:
        OrderFlowAnalyzer instance
    """
    if not SMART_MONEY_AVAILABLE:
        logger.warning("⚠️ Order flow analysis not available - returning placeholder")
        return OrderFlowAnalyzer()  # Placeholder that returns safe defaults
    
    try:
        analyzer = OrderFlowAnalyzer()
        logger.info("✅ Order Flow Analyzer created")
        return analyzer
    except Exception as e:
        logger.error(f"❌ Failed to create Order Flow Analyzer: {e}")
        return OrderFlowAnalyzer()  # Return placeholder on error

def create_smart_money_suite():
    """
    Factory function to create complete smart money analysis suite
    
    Returns:
        Dictionary containing both analyzers
    """
    return {
        'market_structure': create_market_structure_analyzer(),
        'order_flow': create_order_flow_analyzer(),
        'available': SMART_MONEY_AVAILABLE
    }

# Configuration validation - now uses database-driven config
def validate_intelligence_config():
    """Validate that required configuration is available"""
    try:
        # Try database config first
        from forex_scanner.services.intelligence_config_service import get_intelligence_config
        config = get_intelligence_config()
        if config:
            logger.info("✅ Intelligence configuration validated (database)")
            return True
    except ImportError:
        pass

    # Fallback to system config for basic requirements
    try:
        from forex_scanner import config as system_config
        required_attrs = ['EPIC_LIST', 'DATABASE_URL']
        missing_attrs = [attr for attr in required_attrs if not hasattr(system_config, attr)]

        if missing_attrs:
            logger.warning(f"⚠️ Missing config attributes for intelligence: {missing_attrs}")
            return False

        logger.info("✅ Intelligence configuration validated (fallback)")
        return True
    except ImportError:
        logger.debug("Config module not found - using database config only")
        return True  # Database config is available

# Validate smart money configuration - now uses database config
def validate_smart_money_config():
    """Validate smart money configuration"""
    try:
        # Database config is the source of truth
        from forex_scanner.services.intelligence_config_service import get_intelligence_config
        config = get_intelligence_config()
        if config:
            smart_money_enabled = getattr(config, 'enable_smart_money_collection', True)
            logger.info(f"✅ Smart money config validated (enabled: {smart_money_enabled})")
            return True
    except ImportError:
        pass

    # Fallback - smart money works with defaults
    logger.info("ℹ️ Smart money using default configuration")
    return True

# Quick setup function for solving zero-alert problem
def setup_minimal_intelligence():
    """
    Quick setup function to solve zero-alert problem
    Returns intelligence config manager with minimal preset
    """
    try:
        if CONFIGURABLE_INTELLIGENCE_AVAILABLE:
            config_mgr = IntelligenceConfigManager('minimal')
            logger.info("✅ Minimal intelligence setup complete - should generate signals")
            return config_mgr
        else:
            logger.error("❌ Configurable intelligence not available")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to setup minimal intelligence: {e}")
        return None

# NEW: Quick setup for smart money analysis
def setup_smart_money_analysis():
    """
    Quick setup function for smart money analysis
    Returns dictionary with smart money components
    """
    try:
        if not SMART_MONEY_AVAILABLE:
            logger.warning("⚠️ Smart money analysis not available - using placeholders")
            return {
                'available': False,
                'market_structure': MarketStructureAnalyzer(),
                'order_flow': OrderFlowAnalyzer()
            }
        
        smart_money_suite = create_smart_money_suite()
        logger.info("✅ Smart money analysis setup complete")
        return smart_money_suite
        
    except Exception as e:
        logger.error(f"❌ Failed to setup smart money analysis: {e}")
        return {
            'available': False,
            'error': str(e),
            'market_structure': MarketStructureAnalyzer(),
            'order_flow': OrderFlowAnalyzer()
        }

# NEW: Comprehensive system status check
def get_intelligence_system_status():
    """Get comprehensive status of all intelligence components"""
    return {
        'version': __version__,
        'market_intelligence_available': is_intelligence_available(),
        'configurable_intelligence_available': CONFIGURABLE_INTELLIGENCE_AVAILABLE,
        'smart_money_available': SMART_MONEY_AVAILABLE,
        'components': {
            'market_intelligence_engine': True,
            'intelligence_config_manager': CONFIGURABLE_INTELLIGENCE_AVAILABLE,
            'market_structure_analyzer': SMART_MONEY_AVAILABLE,
            'order_flow_analyzer': SMART_MONEY_AVAILABLE
        },
        'smart_money_phase': 'Phase 1 & 2 (Market Structure + Order Flow)' if SMART_MONEY_AVAILABLE else 'Not Available'
    }

# Run validation on import (optional)
if __name__ != "__main__":
    validate_intelligence_config()
    if SMART_MONEY_AVAILABLE:
        validate_smart_money_config()

# NEW: Quick access functions for common use cases
def get_market_structure_analyzer():
    """Quick access to market structure analyzer"""
    return create_market_structure_analyzer()

def get_order_flow_analyzer():
    """Quick access to order flow analyzer"""
    return create_order_flow_analyzer()

# NEW: Smart money validation helper
def validate_signal_with_smart_money(signal_type: str, price: float, epic: str):
    """
    Quick validation of a signal using smart money analysis
    
    Args:
        signal_type: 'BUY'/'SELL'/'BULL'/'BEAR'
        price: Signal price
        epic: Currency pair
        
    Returns:
        Dictionary with validation results
    """
    try:
        if not SMART_MONEY_AVAILABLE:
            return {
                'smart_money_available': False,
                'structure_aligned': True,
                'order_flow_aligned': True,
                'overall_aligned': True,
                'note': 'Smart money analysis not available - allowing signal'
            }
        
        # Get analyzers
        structure_analyzer = create_market_structure_analyzer()
        order_flow_analyzer = create_order_flow_analyzer()
        
        # Validate against both
        structure_result = structure_analyzer.validate_signal_against_structure(
            signal_type, price, epic
        )
        order_flow_result = order_flow_analyzer.validate_signal_against_order_flow(
            signal_type, price, epic
        )
        
        # Combine results
        overall_aligned = (
            structure_result.get('structure_aligned', True) and 
            order_flow_result.get('order_flow_aligned', True)
        )
        
        return {
            'smart_money_available': True,
            'structure_aligned': structure_result.get('structure_aligned', True),
            'structure_score': structure_result.get('structure_score', 0.5),
            'order_flow_aligned': order_flow_result.get('order_flow_aligned', True),
            'order_flow_score': order_flow_result.get('order_flow_score', 0.5),
            'overall_aligned': overall_aligned,
            'structure_reason': structure_result.get('validation_reason', ''),
            'order_flow_reason': order_flow_result.get('validation_reason', '')
        }
        
    except Exception as e:
        logger.error(f"❌ Smart money validation failed: {e}")
        return {
            'smart_money_available': False,
            'error': str(e),
            'structure_aligned': True,  # Default to allow on error
            'order_flow_aligned': True,
            'overall_aligned': True
        }