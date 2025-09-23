# core/intelligence/intelligence_config.py
"""
Configuration Wrapper for Market Intelligence Engine
Runtime configuration manager that uses the new configdata system as its source.
This provides backward compatibility while leveraging the centralized configuration approach.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import from the new configdata system
try:
    from forex_scanner.configdata import config as configdata_config
    HAS_CONFIGDATA = True
except ImportError:
    HAS_CONFIGDATA = False
    print("⚠️ ConfigData system not available, falling back to standalone mode")


class IntelligenceConfigManager:
    """
    Runtime configuration manager for Market Intelligence Engine
    Uses the new configdata system as its source for configuration values.
    Provides backward compatibility and runtime preset switching capabilities.
    """

    def __init__(self, preset: str = None):
        self.logger = logging.getLogger(__name__)

        # Use configdata system if available, otherwise fall back to standalone mode
        if HAS_CONFIGDATA:
            self.config_source = configdata_config
            # Use the preset from configdata if not specified
            if preset is None:
                preset = self.config_source.get_intelligence_preset()
            self.preset = preset
            self.logger.info(f"🧠 Intelligence Config Manager: Using configdata system with preset '{preset}'")
        else:
            self.config_source = None
            self.preset = preset or 'balanced'
            self.config = {}
            self.load_preset_fallback(self.preset)
            self.logger.warning(f"⚠️ Intelligence Config Manager: Using fallback mode with preset '{self.preset}'")

        # Try to import existing intelligence engine
        try:
            from .market_intelligence import MarketIntelligenceEngine
            self.intelligence_engine_class = MarketIntelligenceEngine
            self.has_intelligence_engine = True
            self.logger.info(f"✅ MarketIntelligenceEngine available")
        except ImportError:
            self.intelligence_engine_class = None
            self.has_intelligence_engine = False
            self.logger.warning("⚠️ MarketIntelligenceEngine not available")
    
    def load_preset(self, preset_name: str):
        """Load a configuration preset using the configdata system"""

        if HAS_CONFIGDATA and self.config_source:
            # Use configdata system to switch preset
            self.config_source.set_intelligence_preset(preset_name)
            self.preset = preset_name
            self.logger.info(f"✅ Loaded preset '{preset_name}' via configdata system")
            return
        else:
            # Fall back to standalone mode
            self.load_preset_fallback(preset_name)

    def load_preset_fallback(self, preset_name: str):
        """Fallback preset loading when configdata system is not available"""

        # Define fallback presets (only used when configdata is not available)
        presets = {
            'disabled': {
                'mode': 'disabled',
                'threshold': 0.0,
                'use_intelligence_engine': False,
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': False,
                    'volume_filter': False,
                    'time_filter': False,
                    'confidence_filter': True,  # Keep basic confidence only
                },
                'description': 'No intelligence filtering - signals based on strategy only'
            },
            
            'minimal': {
                'mode': 'live_only',
                'threshold': 0.3,  # Much lower than original 70%
                'use_intelligence_engine': False,  # Start simple
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': False,
                    'volume_filter': True,   # Only volume confirmation
                    'time_filter': False,
                    'confidence_filter': True,
                },
                'weights': {
                    'volume': 0.5,
                    'confidence': 0.5
                },
                'description': 'Minimal filtering - volume + confidence only'
            },
            
            'balanced': {
                'mode': 'live_only',
                'threshold': 0.5,  # Still lower than original 70%
                'use_intelligence_engine': True,  # Use your MarketIntelligenceEngine
                'intelligence_weight': 0.4,  # Weight of intelligence vs other factors
                'components_enabled': {
                    'market_regime_filter': True,  # Use your engine
                    'volatility_filter': True,
                    'volume_filter': True,
                    'time_filter': False,  # Time filter often too restrictive
                    'confidence_filter': True,
                },
                'weights': {
                    'market_regime': 0.3,
                    'volatility': 0.2,
                    'volume': 0.2,
                    'confidence': 0.3
                },
                'description': 'Balanced filtering with MarketIntelligenceEngine'
            },
            
            'advanced': {
                'mode': 'enhanced',
                'threshold': 0.6,  # Still lower than original 80%
                'use_intelligence_engine': True,
                'intelligence_weight': 0.6,  # Higher weight for intelligence
                'market_regime_threshold': 0.5,  # Custom threshold for regime analysis
                'components_enabled': {
                    'market_regime_filter': True,
                    'volatility_filter': True,
                    'volume_filter': True,
                    'time_filter': True,
                    'confidence_filter': True,
                },
                'weights': {
                    'market_regime': 0.4,  # Higher weight for your intelligence engine
                    'volatility': 0.2,
                    'volume': 0.2,
                    'confidence': 0.2
                },
                'description': 'Advanced filtering with full MarketIntelligenceEngine'
            },
            
            'testing': {
                'mode': 'backtest_consistent',
                'threshold': 0.4,
                'use_intelligence_engine': False,  # Keep simple for testing
                'components_enabled': {
                    'market_regime_filter': False,
                    'volatility_filter': True,
                    'volume_filter': True,
                    'time_filter': False,
                    'confidence_filter': True,
                },
                'weights': {
                    'volatility': 0.3,
                    'volume': 0.3,
                    'confidence': 0.4
                },
                'description': 'Testing mode - consistent with backtesting'
            }
        }
        
        if preset_name not in presets:
            self.logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
            preset_name = 'balanced'
        
        self.config = presets[preset_name].copy()
        self.preset = preset_name
        
        self.logger.info(f"✅ Loaded '{preset_name}' preset: {self.config['description']}")
    
    def get_effective_threshold(self) -> float:
        """Get the current intelligence threshold"""
        if HAS_CONFIGDATA and self.config_source:
            return self.config_source.get_intelligence_threshold()
        else:
            return self.config.get('threshold', 0.5)
    
    def should_use_intelligence_engine(self) -> bool:
        """Check if we should use your MarketIntelligenceEngine"""
        if HAS_CONFIGDATA and self.config_source:
            # Check if intelligence is enabled in configdata
            preset_info = getattr(self.config_source.intelligence, 'INTELLIGENCE_PRESETS', {}).get(self.preset, {})
            use_engine = preset_info.get('use_intelligence_engine', True)
            return use_engine and self.has_intelligence_engine
        else:
            return (
                self.config.get('use_intelligence_engine', False) and
                self.has_intelligence_engine
            )
    
    def get_intelligence_weight(self) -> float:
        """Get weight for intelligence engine in overall scoring"""
        if HAS_CONFIGDATA and self.config_source:
            preset_info = getattr(self.config_source.intelligence, 'INTELLIGENCE_PRESETS', {}).get(self.preset, {})
            return preset_info.get('intelligence_weight', 0.4)
        else:
            return self.config.get('intelligence_weight', 0.4)
    
    def is_component_enabled(self, component_name: str) -> bool:
        """Check if a specific intelligence component is enabled"""
        if HAS_CONFIGDATA and self.config_source:
            components = getattr(self.config_source.intelligence, 'INTELLIGENCE_COMPONENTS_ENABLED', {})
            return components.get(component_name, True)
        else:
            return self.config.get('components_enabled', {}).get(component_name, True)
    
    def get_component_weight(self, component_name: str) -> float:
        """Get weight for a component"""
        # Remove '_filter' suffix if present
        clean_name = component_name.replace('_filter', '')

        if HAS_CONFIGDATA and self.config_source:
            weights = getattr(self.config_source.intelligence, 'INTELLIGENCE_WEIGHTS', {})
            return weights.get(clean_name, 0.2)
        else:
            return self.config.get('weights', {}).get(clean_name, 0.2)
    
    def calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """
        Calculate final intelligence score using configured weights
        Only includes enabled components
        """
        # Check if intelligence is disabled
        if HAS_CONFIGDATA and self.config_source:
            mode = getattr(self.config_source.intelligence, 'INTELLIGENCE_MODE', 'live_only')
            if mode == 'disabled':
                return 1.0  # Always pass when disabled
        elif self.config.get('mode') == 'disabled':
            return 1.0  # Always pass when disabled
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            component_filter = f"{component}_filter"
            if self.is_component_enabled(component_filter):
                weight = self.get_component_weight(component)
                total_score += score * weight
                total_weight += weight
        
        # Normalize by total weight used
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        return min(1.0, max(0.0, final_score))
    
    def should_filter_signal(self, intelligence_score: float) -> bool:
        """Determine if signal should be filtered out"""
        # Check if intelligence is disabled
        if HAS_CONFIGDATA and self.config_source:
            mode = getattr(self.config_source.intelligence, 'INTELLIGENCE_MODE', 'live_only')
            if mode == 'disabled':
                return False  # Never filter when disabled
        elif self.config.get('mode') == 'disabled':
            return False  # Never filter when disabled
        
        threshold = self.get_effective_threshold()
        return intelligence_score < threshold
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current configuration status"""
        if HAS_CONFIGDATA and self.config_source:
            # Use configdata system
            return self.config_source.get_intelligence_summary()
        else:
            # Fall back to local config
            enabled_components = [
                comp.replace('_filter', '')
                for comp, enabled in self.config.get('components_enabled', {}).items()
                if enabled
            ]

            return {
                'preset': self.preset,
                'mode': self.config.get('mode', 'unknown'),
                'threshold': self.get_effective_threshold(),
                'enabled_components': enabled_components,
                'component_count': len(enabled_components),
                'description': self.config.get('description', 'No description'),
                'is_disabled': self.config.get('mode') == 'disabled',
                'uses_intelligence_engine': self.should_use_intelligence_engine(),
                'intelligence_weight': self.get_intelligence_weight()
            }
    
    def log_signal_decision(self, epic: str, intelligence_score: float, passed: bool):
        """Log intelligence filtering decision"""
        threshold = self.get_effective_threshold()
        status = "✅ PASSED" if passed else "🚫 FILTERED"
        
        engine_status = " (w/ Engine)" if self.should_use_intelligence_engine() else ""
        
        self.logger.info(
            f"{status} {epic}: Intelligence {intelligence_score:.1%} "
            f"{'≥' if passed else '<'} {threshold:.1%} "
            f"(preset: {self.preset}{engine_status})"
        )
    
    def get_market_regime_threshold(self) -> float:
        """Get threshold for market regime analysis from your intelligence engine"""
        if HAS_CONFIGDATA and self.config_source:
            return getattr(self.config_source.intelligence, 'MARKET_INTELLIGENCE_MIN_CONFIDENCE', 0.5)
        else:
            return self.config.get('market_regime_threshold', 0.5)


# =====================================================================================
# INTEGRATION STATUS
# =====================================================================================
"""
✅ INTEGRATION COMPLETE: This intelligence config manager now uses the new configdata system!

The IntelligenceConfigManager has been updated to:
1. Use configdata system as primary source when available
2. Fall back to standalone mode when configdata is not available
3. Provide backward compatibility for existing code
4. Support runtime preset switching via configdata

Usage:
    # Create manager (automatically uses configdata if available)
    manager = IntelligenceConfigManager()

    # Switch presets (delegates to configdata system)
    manager.load_preset('minimal')

    # Get configuration values (from configdata)
    threshold = manager.get_effective_threshold()
    enabled = manager.is_component_enabled('volatility_filter')

Integration Benefits:
- Centralized configuration in configdata system
- Runtime preset switching capabilities
- Backward compatibility for existing intelligence code
- Automatic fallback when configdata not available
"""

# Update your existing __init__.py to export the new config manager
def update_intelligence_init():
    """
    Add this to your existing core/intelligence/__init__.py:

    from .intelligence_config import IntelligenceConfigManager

    __all__ = [
        'MarketIntelligenceEngine',
        'IntelligenceConfigManager',  # Add this line
    ]
    """
    pass


if __name__ == "__main__":
    # Test the configuration manager with existing intelligence engine
    print("🧠 Testing Intelligence Configuration Manager")
    print("=" * 60)
    
    # Test all presets
    presets = ['disabled', 'minimal', 'balanced', 'advanced', 'testing']
    
    for preset in presets:
        print(f"\n📊 Testing preset: {preset}")
        config_mgr = IntelligenceConfigManager(preset)
        status = config_mgr.get_status_summary()
        
        print(f"   Mode: {status['mode']}")
        print(f"   Threshold: {status['threshold']:.1%}")
        print(f"   Components: {', '.join(status['enabled_components'])}")
        print(f"   Uses Intelligence Engine: {status['uses_intelligence_engine']}")
        print(f"   Description: {status['description']}")
        
        # Test score calculation
        test_scores = {
            'market_regime': 0.6,
            'volatility': 0.5,
            'volume': 0.7,
            'confidence': 0.8
        }
        
        final_score = config_mgr.calculate_weighted_score(test_scores)
        would_pass = not config_mgr.should_filter_signal(final_score)
        
        print(f"   Test signal: {final_score:.1%} → {'PASS' if would_pass else 'FILTER'}")
    
    print(f"\n✅ Configuration manager ready for integration with existing MarketIntelligenceEngine!")