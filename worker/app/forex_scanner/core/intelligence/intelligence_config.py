# core/intelligence/intelligence_config.py
"""
Configuration Wrapper for Existing Market Intelligence Engine
Works with your existing MarketIntelligenceEngine to make thresholds configurable
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime


class IntelligenceConfigManager:
    """
    Configuration manager that wraps your existing MarketIntelligenceEngine
    Makes the intelligence thresholds configurable to solve zero-alert problem
    """
    
    def __init__(self, preset: str = 'balanced'):
        self.logger = logging.getLogger(__name__)
        self.preset = preset
        self.config = {}
        self.load_preset(preset)
        
        # Try to import existing intelligence engine
        try:
            from .market_intelligence import MarketIntelligenceEngine
            self.intelligence_engine_class = MarketIntelligenceEngine
            self.has_intelligence_engine = True
            self.logger.info(f"🧠 Intelligence Config: {preset} mode with MarketIntelligenceEngine")
        except ImportError:
            self.intelligence_engine_class = None
            self.has_intelligence_engine = False
            self.logger.warning("⚠️ MarketIntelligenceEngine not available")
    
    def load_preset(self, preset_name: str):
        """Load a configuration preset that solves the zero-alert problem"""
        
        # Define presets with configurable thresholds (much lower than original 70-80%)
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
        return self.config.get('threshold', 0.5)
    
    def should_use_intelligence_engine(self) -> bool:
        """Check if we should use your MarketIntelligenceEngine"""
        return (
            self.config.get('use_intelligence_engine', False) and 
            self.has_intelligence_engine
        )
    
    def get_intelligence_weight(self) -> float:
        """Get weight for intelligence engine in overall scoring"""
        return self.config.get('intelligence_weight', 0.4)
    
    def is_component_enabled(self, component_name: str) -> bool:
        """Check if a specific intelligence component is enabled"""
        return self.config.get('components_enabled', {}).get(component_name, True)
    
    def get_component_weight(self, component_name: str) -> float:
        """Get weight for a component"""
        # Remove '_filter' suffix if present
        clean_name = component_name.replace('_filter', '')
        return self.config.get('weights', {}).get(clean_name, 0.2)
    
    def calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """
        Calculate final intelligence score using configured weights
        Only includes enabled components
        """
        if self.config['mode'] == 'disabled':
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
        if self.config['mode'] == 'disabled':
            return False  # Never filter when disabled
        
        threshold = self.get_effective_threshold()
        return intelligence_score < threshold
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current configuration status"""
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
            'is_disabled': self.config['mode'] == 'disabled',
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
        return self.config.get('market_regime_threshold', 0.5)


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