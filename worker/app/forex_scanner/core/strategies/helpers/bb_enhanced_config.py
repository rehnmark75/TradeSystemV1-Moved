# core/strategies/helpers/bb_enhanced_config.py
"""
Enhanced BB Strategy Configuration Module
🔧 CONFIG: Centralized configuration for enhanced BB features
⚙️ PURPOSE: Manage squeeze→expansion and band walk settings
🎯 INTEGRATION: Easy integration with existing config system

FEATURES:
- Squeeze/Expansion thresholds
- Band walk confirmation settings
- Trend/Range mode detection parameters
- Exit strategy configuration
- Session filtering options
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging


@dataclass
class BBEnhancedConfig:
    """
    Enhanced BB Strategy Configuration
    Based on the provided example code with forex-optimized defaults
    """
    
    # Core BB and SuperTrend indicators
    bb_length: int = 20
    bb_std: float = 2.0
    st_atr_length: int = 10
    st_multiplier: float = 3.0
    atr_length: int = 14
    
    # Squeeze → Expansion detection
    squeeze_ema_length: int = 20
    squeeze_factor: float = 0.80     # BB width < EMA * factor = squeeze
    expansion_factor: float = 1.05   # BB width > EMA * factor = expansion (was 1.07 in example)
    
    # Band walk confirmation
    band_walk_n: int = 2             # Number of consecutive closes needed
    band_buffer_atr_mult: float = 0.10  # ATR buffer multiplier for band detection
    
    # Mode detection (trend vs range)
    trend_detection: str = "both"    # 'supertrend' | 'bandwalk' | 'both'
    require_supertrend_agreement: bool = True
    
    # Execution filters
    max_spread_pips: float = 1.5
    min_atr_pips: float = 2.0
    pip_size: Optional[float] = None
    symbol: Optional[str] = None
    
    # Exit strategies
    range_sl_atr_mult: float = 0.60  # SL distance in range mode (was 0.6 in example)
    trend_trail_atr_mult: float = 2.0  # Chandelier trailing multiplier for trend mode
    range_tp_offset_pips: float = 0.0  # TP offset from opposite band
    
    # Session filtering (optional)
    session_start_hour: Optional[int] = 7   # London session start (GMT)
    session_end_hour: Optional[int] = 22    # NY session end (GMT)
    
    # Execution timing
    enter_on_close: bool = True      # Enter on signal bar close vs next bar open
    
    # Enhanced features flags
    enable_squeeze_expansion: bool = True
    enable_band_walk_confirmation: bool = True
    enable_multi_timeframe: bool = False  # Can be enabled later
    enable_volume_confirmation: bool = False  # Optional volume filter
    
    # Confidence and quality thresholds
    min_confidence_threshold: float = 0.60
    min_bb_width_threshold: float = 0.005  # Minimum 0.5% of price
    max_bb_width_threshold: float = 0.08   # Maximum 8% of price
    
    # Performance and caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for easy integration"""
        return {
            'bb_length': self.bb_length,
            'bb_std': self.bb_std,
            'st_atr_length': self.st_atr_length,
            'st_multiplier': self.st_multiplier,
            'atr_length': self.atr_length,
            'squeeze_ema_length': self.squeeze_ema_length,
            'squeeze_factor': self.squeeze_factor,
            'expansion_factor': self.expansion_factor,
            'band_walk_n': self.band_walk_n,
            'band_buffer_atr_mult': self.band_buffer_atr_mult,
            'trend_detection': self.trend_detection,
            'require_supertrend_agreement': self.require_supertrend_agreement,
            'max_spread_pips': self.max_spread_pips,
            'min_atr_pips': self.min_atr_pips,
            'range_sl_atr_mult': self.range_sl_atr_mult,
            'trend_trail_atr_mult': self.trend_trail_atr_mult,
            'range_tp_offset_pips': self.range_tp_offset_pips,
            'session_start_hour': self.session_start_hour,
            'session_end_hour': self.session_end_hour,
            'enter_on_close': self.enter_on_close,
            'enable_squeeze_expansion': self.enable_squeeze_expansion,
            'enable_band_walk_confirmation': self.enable_band_walk_confirmation,
            'min_confidence_threshold': self.min_confidence_threshold,
            'min_bb_width_threshold': self.min_bb_width_threshold,
            'max_bb_width_threshold': self.max_bb_width_threshold,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BBEnhancedConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Validate basic parameters
        if self.bb_length < 5 or self.bb_length > 50:
            errors.append("bb_length must be between 5 and 50")
        
        if self.bb_std < 1.0 or self.bb_std > 3.0:
            errors.append("bb_std must be between 1.0 and 3.0")
        
        if self.squeeze_factor <= 0 or self.squeeze_factor >= 1.0:
            errors.append("squeeze_factor must be between 0 and 1.0")
        
        if self.expansion_factor <= 1.0 or self.expansion_factor > 2.0:
            errors.append("expansion_factor must be between 1.0 and 2.0")
        
        if self.band_walk_n < 1 or self.band_walk_n > 5:
            errors.append("band_walk_n must be between 1 and 5")
        
        if self.trend_detection not in ['supertrend', 'bandwalk', 'both']:
            errors.append("trend_detection must be 'supertrend', 'bandwalk', or 'both'")
        
        # Log validation errors
        if errors:
            logger = logging.getLogger(__name__)
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
        
        return True


class BBConfigFactory:
    """
    Factory for creating pre-configured BB strategy setups
    """
    
    @staticmethod
    def fx15m_conservative(symbol: Optional[str] = None) -> BBEnhancedConfig:
        """Conservative 15M forex setup - fewer signals, higher quality"""
        return BBEnhancedConfig(
            bb_length=20,
            bb_std=2.2,  # Wider bands
            squeeze_factor=0.75,  # Tighter squeeze definition
            expansion_factor=1.10,  # Higher expansion threshold
            band_walk_n=3,  # Require more confirmation
            max_spread_pips=1.0,  # Tighter spread requirement
            min_atr_pips=3.0,  # Higher volatility requirement
            min_confidence_threshold=0.70,  # Higher confidence needed
            symbol=symbol,
            range_sl_atr_mult=0.50,  # Tighter stops
            trend_trail_atr_mult=2.5,  # Wider trailing in trends
        )
    
    @staticmethod
    def fx15m_default(symbol: Optional[str] = None) -> BBEnhancedConfig:
        """Default 15M forex setup - balanced approach"""
        return BBEnhancedConfig(
            symbol=symbol,
            # Uses class defaults which are already optimized for 15M forex
        )
    
    @staticmethod
    def fx15m_aggressive(symbol: Optional[str] = None) -> BBEnhancedConfig:
        """Aggressive 15M forex setup - more signals, lower threshold"""
        return BBEnhancedConfig(
            bb_length=18,  # Shorter period
            bb_std=1.8,    # Tighter bands
            squeeze_factor=0.85,  # Looser squeeze definition
            expansion_factor=1.03,  # Lower expansion threshold
            band_walk_n=1,  # Less confirmation required
            max_spread_pips=2.0,  # Allow wider spreads
            min_atr_pips=1.5,  # Lower volatility requirement
            min_confidence_threshold=0.55,  # Lower confidence threshold
            symbol=symbol,
            range_sl_atr_mult=0.75,  # Wider stops
            trend_trail_atr_mult=1.5,  # Tighter trailing
        )
    
    @staticmethod
    def fx5m_scalping(symbol: Optional[str] = None) -> BBEnhancedConfig:
        """5M scalping setup - quick entries and exits"""
        return BBEnhancedConfig(
            bb_length=14,  # Faster response
            bb_std=1.5,    # Tighter bands for more signals
            st_atr_length=7,  # Faster SuperTrend
            squeeze_factor=0.90,  # Looser squeeze
            expansion_factor=1.02,  # Very sensitive expansion
            band_walk_n=1,  # Single bar confirmation
            max_spread_pips=0.8,  # Very tight spreads for scalping
            min_atr_pips=1.0,  # Allow lower volatility
            symbol=symbol,
            range_sl_atr_mult=0.40,  # Very tight stops
            trend_trail_atr_mult=1.0,  # Tight trailing
            session_start_hour=8,   # Only trade major sessions
            session_end_hour=17,    # London session focus
        )
    
    @staticmethod
    def fx1h_swing(symbol: Optional[str] = None) -> BBEnhancedConfig:
        """1H swing trading setup - longer holds, wider targets"""
        return BBEnhancedConfig(
            bb_length=24,  # Longer lookback
            bb_std=2.5,    # Wider bands
            st_atr_length=14,  # Standard SuperTrend
            squeeze_factor=0.70,  # Tighter squeeze for quality
            expansion_factor=1.15,  # Significant expansion required
            band_walk_n=3,  # More confirmation
            max_spread_pips=2.5,  # Allow wider spreads
            min_atr_pips=5.0,  # Higher volatility for swings
            symbol=symbol,
            range_sl_atr_mult=1.0,  # Wider stops for swings
            trend_trail_atr_mult=3.0,  # Wide trailing for trends
            min_confidence_threshold=0.75,  # Higher quality signals
        )
    
    @staticmethod
    def get_config_for_timeframe(timeframe: str, mode: str = 'default', symbol: Optional[str] = None) -> BBEnhancedConfig:
        """Get appropriate config based on timeframe and trading mode"""
        timeframe = timeframe.upper()
        mode = mode.lower()
        
        if timeframe in ['5M', '5']:
            if mode == 'conservative':
                return BBConfigFactory.fx15m_conservative(symbol)  # Use 15M conservative for 5M conservative
            elif mode == 'aggressive':
                return BBConfigFactory.fx5m_scalping(symbol)
            else:
                return BBConfigFactory.fx5m_scalping(symbol)
        
        elif timeframe in ['15M', '15']:
            if mode == 'conservative':
                return BBConfigFactory.fx15m_conservative(symbol)
            elif mode == 'aggressive':
                return BBConfigFactory.fx15m_aggressive(symbol)
            else:
                return BBConfigFactory.fx15m_default(symbol)
        
        elif timeframe in ['1H', '60M', '60']:
            return BBConfigFactory.fx1h_swing(symbol)
        
        else:
            # Default to 15M setup for unknown timeframes
            return BBConfigFactory.fx15m_default(symbol)


def integrate_with_main_config(main_config_module, bb_config: BBEnhancedConfig):
    """
    Integrate enhanced BB config with main config.py
    This function helps merge the enhanced settings into the existing config system
    """
    try:
        # Map enhanced config to main config variables
        config_mapping = {
            'BB_PERIOD': bb_config.bb_length,
            'BB_STD_DEV': bb_config.bb_std,
            'SUPERTREND_PERIOD': bb_config.st_atr_length,
            'SUPERTREND_MULTIPLIER': bb_config.st_multiplier,
            'ATR_PERIOD': bb_config.atr_length,
            
            # Enhanced features
            'BB_SQUEEZE_EMA_LENGTH': bb_config.squeeze_ema_length,
            'BB_SQUEEZE_FACTOR': bb_config.squeeze_factor,
            'BB_EXPANSION_FACTOR': bb_config.expansion_factor,
            'BB_BAND_WALK_N': bb_config.band_walk_n,
            'BB_BAND_BUFFER_ATR_MULT': bb_config.band_buffer_atr_mult,
            
            # Mode and filters
            'BB_TREND_DETECTION': bb_config.trend_detection,
            'BB_REQUIRE_ST_AGREEMENT': bb_config.require_supertrend_agreement,
            'BB_MAX_SPREAD_PIPS': bb_config.max_spread_pips,
            'BB_MIN_ATR_PIPS': bb_config.min_atr_pips,
            
            # Exit strategies
            'BB_RANGE_SL_ATR_MULT': bb_config.range_sl_atr_mult,
            'BB_TREND_TRAIL_ATR_MULT': bb_config.trend_trail_atr_mult,
            'BB_RANGE_TP_OFFSET_PIPS': bb_config.range_tp_offset_pips,
            
            # Feature flags
            'BB_ENABLE_SQUEEZE_EXPANSION': bb_config.enable_squeeze_expansion,
            'BB_ENABLE_BAND_WALK': bb_config.enable_band_walk_confirmation,
            'BB_MIN_CONFIDENCE': bb_config.min_confidence_threshold,
        }
        
        # Apply to main config
        for config_key, config_value in config_mapping.items():
            setattr(main_config_module, config_key, config_value)
        
        # Set main strategy flag if not already set
        if not hasattr(main_config_module, 'BOLLINGER_SUPERTREND_STRATEGY'):
            setattr(main_config_module, 'BOLLINGER_SUPERTREND_STRATEGY', True)
        
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to integrate enhanced BB config: {e}")
        return False


# Example usage:
if __name__ == "__main__":
    # Test different configurations
    configs = {
        'conservative_15m': BBConfigFactory.fx15m_conservative('EURUSD'),
        'default_15m': BBConfigFactory.fx15m_default('EURUSD'),
        'aggressive_15m': BBConfigFactory.fx15m_aggressive('EURUSD'),
        'scalping_5m': BBConfigFactory.fx5m_scalping('EURUSD'),
        'swing_1h': BBConfigFactory.fx1h_swing('EURUSD'),
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  BB Period: {config.bb_length}, Std: {config.bb_std}")
        print(f"  Squeeze Factor: {config.squeeze_factor}, Expansion Factor: {config.expansion_factor}")
        print(f"  Band Walk: {config.band_walk_n} bars, Buffer: {config.band_buffer_atr_mult}")
        print(f"  SL Multiplier: {config.range_sl_atr_mult}, Trail Multiplier: {config.trend_trail_atr_mult}")
        print(f"  Min Confidence: {config.min_confidence_threshold}")
        print(f"  Valid: {config.validate()}")