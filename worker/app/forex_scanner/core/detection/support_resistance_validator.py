# core/detection/support_resistance_validator.py
"""
Support and Resistance Validator - Based on LuxAlgo TradingView Script
Validates trades against major support/resistance levels to prevent wrong direction entries

This implementation converts the TradingView Pine Script logic to Python:
- Detects pivot highs/lows using left/right bars
- Identifies support and resistance levels
- Validates volume confirmation for level breaks
- Prevents trades in wrong direction near major levels
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SupportResistanceValidator:
    """
    Support and Resistance Level Validator
    
    Based on LuxAlgo TradingView script logic:
    - Uses pivot points with configurable left/right bars
    - Volume confirmation for level breaks
    - Prevents wrong direction trades near major levels
    """
    
    def __init__(self, 
                 left_bars: int = 15,
                 right_bars: int = 15,
                 volume_threshold: float = 20.0,
                 level_tolerance_pips: float = 5.0,
                 min_level_distance_pips: float = 10.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Support/Resistance Validator
        
        Args:
            left_bars: Left bars for pivot detection (default: 15)
            right_bars: Right bars for pivot detection (default: 15)
            volume_threshold: Volume threshold percentage for breaks (default: 20%)
            level_tolerance_pips: Tolerance around levels in pips (default: 5)
            min_level_distance_pips: Minimum distance to consider level significant (default: 10)
            logger: Optional logger instance
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.volume_threshold = volume_threshold
        self.level_tolerance_pips = level_tolerance_pips
        self.min_level_distance_pips = min_level_distance_pips
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache for performance
        self._level_cache = {}
        self._cache_expiry = {}
        self.cache_duration_minutes = 15
        
        self.logger.info(f"✅ SupportResistanceValidator initialized with left_bars={left_bars}, "
                        f"right_bars={right_bars}, volume_threshold={volume_threshold}%")
    
    def validate_trade_direction(self, 
                                signal: Dict, 
                                df: pd.DataFrame,
                                epic: str) -> Tuple[bool, str, Dict]:
        """
        Main validation method - prevents trades in wrong direction near major levels
        
        Args:
            signal: Trading signal dictionary with signal_type, current_price, etc.
            df: Price data DataFrame with OHLC + volume
            epic: Trading instrument identifier
            
        Returns:
            Tuple of (is_valid, reason, validation_details)
        """
        try:
            signal_type = signal.get('signal_type', '').upper()
            current_price = self._get_current_price(signal)
            
            if not current_price or signal_type not in ['BUY', 'SELL', 'BULL', 'BEAR']:
                return True, "No validation needed - invalid signal format", {}
            
            # Get or calculate support/resistance levels
            levels = self._get_support_resistance_levels(df, epic)
            
            if not levels['support_levels'] and not levels['resistance_levels']:
                return True, "No significant levels found - trade allowed", levels
            
            # Check proximity to major levels
            validation_result = self._check_level_proximity(
                current_price=current_price,
                signal_type=signal_type,
                levels=levels,
                df=df
            )
            
            # Enhanced validation details
            pip_size = self._get_pip_size(epic)
            validation_details = {
                'support_levels': levels['support_levels'],
                'resistance_levels': levels['resistance_levels'],
                'nearest_support': levels.get('nearest_support'),
                'nearest_resistance': levels.get('nearest_resistance'),
                'current_price': current_price,
                'signal_type': signal_type,
                'pip_size': pip_size,
                'level_tolerance_pips': self.level_tolerance_pips,
                'validation_timestamp': datetime.now().isoformat()
            }
            
            if validation_result['is_valid']:
                return True, validation_result['reason'], validation_details
            else:
                return False, validation_result['reason'], validation_details
                
        except Exception as e:
            self.logger.error(f"❌ Error validating trade direction: {e}")
            return True, f"Validation error - allowing trade: {str(e)}", {}
    
    def _get_support_resistance_levels(self, df: pd.DataFrame, epic: str) -> Dict:
        """
        Get support/resistance levels using caching for performance
        
        Args:
            df: Price data DataFrame
            epic: Trading instrument
            
        Returns:
            Dictionary with support/resistance levels
        """
        cache_key = f"{epic}_{len(df)}"
        
        # Check cache
        if (cache_key in self._level_cache and 
            cache_key in self._cache_expiry and
            datetime.now() < self._cache_expiry[cache_key]):
            return self._level_cache[cache_key]
        
        # Calculate new levels
        levels = self._calculate_pivot_levels(df)
        
        # Cache results
        self._level_cache[cache_key] = levels
        self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=self.cache_duration_minutes)
        
        return levels
    
    def _calculate_pivot_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate pivot-based support and resistance levels
        Based on TradingView pivothigh/pivotlow logic
        
        Args:
            df: Price data DataFrame with OHLC columns
            
        Returns:
            Dictionary with support/resistance data
        """
        try:
            if len(df) < (self.left_bars + self.right_bars + 10):
                return {
                    'support_levels': [],
                    'resistance_levels': [],
                    'nearest_support': None,
                    'nearest_resistance': None,
                    'pivot_highs': [],
                    'pivot_lows': []
                }
            
            df_work = df.copy().reset_index(drop=True)
            
            # Calculate pivot highs and lows
            pivot_highs = self._find_pivot_highs(df_work)
            pivot_lows = self._find_pivot_lows(df_work)
            
            # Filter significant levels
            resistance_levels = self._filter_significant_levels(pivot_highs, 'resistance')
            support_levels = self._filter_significant_levels(pivot_lows, 'support')
            
            # Find nearest levels to current price
            current_price = float(df_work['close'].iloc[-1])
            nearest_support = self._find_nearest_support(support_levels, current_price)
            nearest_resistance = self._find_nearest_resistance(resistance_levels, current_price)
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'pivot_highs': pivot_highs,
                'pivot_lows': pivot_lows,
                'calculation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating pivot levels: {e}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': None,
                'nearest_resistance': None,
                'pivot_highs': [],
                'pivot_lows': []
            }
    
    def _find_pivot_highs(self, df: pd.DataFrame) -> List[float]:
        """
        Find pivot highs using left_bars and right_bars logic
        Equivalent to TradingView's pivothigh(left_bars, right_bars)
        """
        pivot_highs = []
        
        for i in range(self.left_bars, len(df) - self.right_bars):
            current_high = df.iloc[i]['high']
            
            # Check if current high is highest in the window
            is_pivot_high = True
            
            # Check left bars
            for j in range(i - self.left_bars, i):
                if df.iloc[j]['high'] >= current_high:
                    is_pivot_high = False
                    break
            
            # Check right bars
            if is_pivot_high:
                for j in range(i + 1, i + self.right_bars + 1):
                    if df.iloc[j]['high'] >= current_high:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                pivot_highs.append(float(current_high))
        
        return sorted(list(set(pivot_highs)), reverse=True)
    
    def _find_pivot_lows(self, df: pd.DataFrame) -> List[float]:
        """
        Find pivot lows using left_bars and right_bars logic
        Equivalent to TradingView's pivotlow(left_bars, right_bars)
        """
        pivot_lows = []
        
        for i in range(self.left_bars, len(df) - self.right_bars):
            current_low = df.iloc[i]['low']
            
            # Check if current low is lowest in the window
            is_pivot_low = True
            
            # Check left bars
            for j in range(i - self.left_bars, i):
                if df.iloc[j]['low'] <= current_low:
                    is_pivot_low = False
                    break
            
            # Check right bars
            if is_pivot_low:
                for j in range(i + 1, i + self.right_bars + 1):
                    if df.iloc[j]['low'] <= current_low:
                        is_pivot_low = False
                        break
            
            if is_pivot_low:
                pivot_lows.append(float(current_low))
        
        return sorted(list(set(pivot_lows)))
    
    def _filter_significant_levels(self, levels: List[float], level_type: str) -> List[float]:
        """
        Filter levels to keep only significant ones
        Removes levels that are too close together
        """
        if not levels:
            return []
        
        filtered_levels = []
        
        for level in sorted(levels, reverse=(level_type == 'resistance')):
            is_significant = True
            
            for existing_level in filtered_levels:
                pip_distance = abs(level - existing_level) * 10000  # Convert to pips
                if pip_distance < self.min_level_distance_pips:
                    is_significant = False
                    break
            
            if is_significant:
                filtered_levels.append(level)
                
                # Limit to top 5 levels to avoid clutter
                if len(filtered_levels) >= 5:
                    break
        
        return filtered_levels
    
    def _find_nearest_support(self, support_levels: List[float], current_price: float) -> Optional[float]:
        """Find nearest support level below current price"""
        supports_below = [level for level in support_levels if level < current_price]
        return max(supports_below) if supports_below else None
    
    def _find_nearest_resistance(self, resistance_levels: List[float], current_price: float) -> Optional[float]:
        """Find nearest resistance level above current price"""
        resistances_above = [level for level in resistance_levels if level > current_price]
        return min(resistances_above) if resistances_above else None
    
    def _check_level_proximity(self, 
                              current_price: float, 
                              signal_type: str, 
                              levels: Dict,
                              df: pd.DataFrame) -> Dict:
        """
        Check if trade direction conflicts with nearby major levels
        
        Key Logic:
        - For BUY signals: Check if we're too close to resistance
        - For SELL signals: Check if we're too close to support
        - Consider volume confirmation for recent level breaks
        """
        pip_size = 0.0001  # Default for most pairs (except JPY)
        
        nearest_support = levels.get('nearest_support')
        nearest_resistance = levels.get('nearest_resistance')
        
        # BUY/BULL signal validation
        if signal_type in ['BUY', 'BULL']:
            if nearest_resistance:
                distance_to_resistance = (nearest_resistance - current_price) / pip_size
                
                if distance_to_resistance <= self.level_tolerance_pips:
                    # Check if resistance was recently broken with volume
                    volume_break = self._check_volume_break(df, nearest_resistance, 'resistance')
                    
                    if not volume_break:
                        return {
                            'is_valid': False,
                            'reason': f"BUY signal too close to resistance at {nearest_resistance:.5f} "
                                    f"({distance_to_resistance:.1f} pips away, minimum: {self.level_tolerance_pips})"
                        }
                    else:
                        return {
                            'is_valid': True,
                            'reason': f"BUY signal allowed - resistance at {nearest_resistance:.5f} "
                                    f"recently broken with volume confirmation"
                        }
        
        # SELL/BEAR signal validation  
        elif signal_type in ['SELL', 'BEAR']:
            if nearest_support:
                distance_to_support = (current_price - nearest_support) / pip_size
                
                if distance_to_support <= self.level_tolerance_pips:
                    # Check if support was recently broken with volume
                    volume_break = self._check_volume_break(df, nearest_support, 'support')
                    
                    if not volume_break:
                        return {
                            'is_valid': False,
                            'reason': f"SELL signal too close to support at {nearest_support:.5f} "
                                    f"({distance_to_support:.1f} pips away, minimum: {self.level_tolerance_pips})"
                        }
                    else:
                        return {
                            'is_valid': True,
                            'reason': f"SELL signal allowed - support at {nearest_support:.5f} "
                                    f"recently broken with volume confirmation"
                        }
        
        # If we get here, trade is allowed
        return {
            'is_valid': True,
            'reason': f"{signal_type} signal allowed - no conflicting levels nearby"
        }
    
    def _check_volume_break(self, df: pd.DataFrame, level_price: float, level_type: str) -> bool:
        """
        Check if a level was recently broken with volume confirmation
        Based on TradingView script's volume threshold logic
        """
        try:
            if len(df) < 20:
                return False
            
            # Calculate volume oscillator (like in TradingView script)
            if 'volume' not in df.columns:
                return False
            
            # Calculate short and long EMAs of volume
            volume_short_ema = df['volume'].ewm(span=5).mean()
            volume_long_ema = df['volume'].ewm(span=10).mean()
            
            # Volume oscillator: 100 * (short - long) / long
            volume_osc = 100 * (volume_short_ema - volume_long_ema) / volume_long_ema
            
            # Check recent bars for level break with volume
            recent_bars = min(10, len(df))
            
            for i in range(len(df) - recent_bars, len(df)):
                current_close = df.iloc[i]['close']
                current_volume_osc = volume_osc.iloc[i]
                
                # Check for level break with volume confirmation
                if level_type == 'resistance':
                    # Check if price broke above resistance with volume
                    if (current_close > level_price and 
                        current_volume_osc > self.volume_threshold):
                        return True
                        
                elif level_type == 'support':
                    # Check if price broke below support with volume  
                    if (current_close < level_price and 
                        current_volume_osc > self.volume_threshold):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Volume break check failed: {e}")
            return False
    
    def _get_current_price(self, signal: Dict) -> Optional[float]:
        """Extract current price from signal data"""
        price_fields = ['current_price', 'entry_price', 'price', 'signal_price', 'close_price']
        
        for field in price_fields:
            if field in signal and signal[field] is not None:
                try:
                    return float(signal[field])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _get_pip_size(self, epic: str) -> float:
        """Get pip size for the given instrument"""
        if 'JPY' in epic.upper():
            return 0.01  # For JPY pairs
        else:
            return 0.0001  # For most other pairs
    
    def get_validation_summary(self) -> str:
        """Get human-readable configuration summary"""
        return (f"S/R Validator: {self.left_bars}/{self.right_bars} bars, "
               f"{self.volume_threshold}% volume threshold, "
               f"{self.level_tolerance_pips} pip tolerance")
    
    def update_configuration(self, **kwargs) -> bool:
        """Update validator configuration"""
        updated = []
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                updated.append(f"{key}={value}")
                self.logger.info(f"⚙️ Updated {key} to {value}")
            else:
                self.logger.warning(f"⚠️ Unknown configuration key: {key}")
        
        if updated:
            # Clear cache when configuration changes
            self._level_cache.clear()
            self._cache_expiry.clear()
            self.logger.info(f"✅ Updated S/R Validator: {', '.join(updated)}")
        
        return len(updated) > 0


# Integration function for trade_validator.py
def create_support_resistance_validator(logger=None, **kwargs):
    """Factory function to create SupportResistanceValidator"""
    return SupportResistanceValidator(logger=logger, **kwargs)


if __name__ == "__main__":
    # Test the validator
    print("🧪 Testing Support/Resistance Validator...")
    
    # Create test data
    import pandas as pd
    np.random.seed(42)
    
    # Generate realistic price data
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    base_price = 1.1200
    
    # Create trending price data with support/resistance levels
    price_data = []
    for i in range(200):
        # Add some trend and noise
        trend = 0.0001 * i  # Slight uptrend
        noise = np.random.normal(0, 0.0005)
        price = base_price + trend + noise
        
        # Add support/resistance bounces
        if i > 50 and price < 1.1180:  # Support level
            price = 1.1180 + abs(noise) * 0.5
        if i > 100 and price > 1.1250:  # Resistance level  
            price = 1.1250 - abs(noise) * 0.5
            
        price_data.append(price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'open': price_data,
        'high': [p + abs(np.random.normal(0, 0.0002)) for p in price_data],
        'low': [p - abs(np.random.normal(0, 0.0002)) for p in price_data],
        'close': price_data,
        'volume': np.random.randint(1000, 5000, 200)
    })
    
    # Create validator
    validator = SupportResistanceValidator()
    
    # Test different signal types
    test_signals = [
        {
            'signal_type': 'BUY',
            'current_price': 1.1245,  # Near resistance
            'epic': 'CS.D.EURUSD.CEEM.IP'
        },
        {
            'signal_type': 'SELL', 
            'current_price': 1.1185,  # Near support
            'epic': 'CS.D.EURUSD.CEEM.IP'
        },
        {
            'signal_type': 'BUY',
            'current_price': 1.1220,  # Safe distance
            'epic': 'CS.D.EURUSD.CEEM.IP'
        }
    ]
    
    print(f"✅ Created validator: {validator.get_validation_summary()}")
    
    for i, signal in enumerate(test_signals, 1):
        is_valid, reason, details = validator.validate_trade_direction(
            signal, df, signal['epic']
        )
        
        print(f"✅ Test {i} ({signal['signal_type']} @ {signal['current_price']:.5f}): "
              f"{'VALID' if is_valid else 'INVALID'} - {reason}")
        
        if details.get('nearest_support'):
            print(f"   📍 Nearest support: {details['nearest_support']:.5f}")
        if details.get('nearest_resistance'):
            print(f"   📍 Nearest resistance: {details['nearest_resistance']:.5f}")
    
    print("🎉 Support/Resistance Validator test completed!")