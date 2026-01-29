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
            
            # v2.35.1: Check for per-signal tolerance override
            tolerance_override = signal.get('sr_tolerance_pips')

            # Check proximity to major levels
            validation_result = self._check_level_proximity(
                current_price=current_price,
                signal_type=signal_type,
                levels=levels,
                df=df,
                epic=epic,
                tolerance_override=tolerance_override
            )
            
            # Enhanced validation details
            pip_size = self._get_pip_size(epic)
            effective_tolerance = tolerance_override if tolerance_override is not None else self.level_tolerance_pips
            validation_details = {
                'support_levels': levels['support_levels'],
                'resistance_levels': levels['resistance_levels'],
                'nearest_support': levels.get('nearest_support'),
                'nearest_resistance': levels.get('nearest_resistance'),
                'current_price': current_price,
                'signal_type': signal_type,
                'pip_size': pip_size,
                'level_tolerance_pips': effective_tolerance,
                'tolerance_override_applied': tolerance_override is not None,
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
        
        # Calculate new levels (pair-aware)
        levels = self._calculate_pivot_levels(df, epic)
        
        # Cache results
        self._level_cache[cache_key] = levels
        self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=self.cache_duration_minutes)
        
        return levels
    
    def _calculate_pivot_levels(self, df: pd.DataFrame, epic: str = None) -> Dict:
        """
        Calculate pivot-based support and resistance levels
        Based on TradingView pivothigh/pivotlow logic

        Args:
            df: Price data DataFrame with OHLC columns
            epic: Trading instrument for pair-aware pip calculation

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

            # Filter significant levels (pair-aware)
            resistance_levels = self._filter_significant_levels(pivot_highs, 'resistance', epic)
            support_levels = self._filter_significant_levels(pivot_lows, 'support', epic)

            # Find nearest levels to current price (pair-aware)
            current_price = float(df_work['close'].iloc[-1])
            nearest_support = self._find_nearest_support(support_levels, current_price, epic)
            nearest_resistance = self._find_nearest_resistance(resistance_levels, current_price, epic)
            
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
    
    def _filter_significant_levels(self, levels: List[float], level_type: str, epic: str = None) -> List[float]:
        """
        Filter levels to keep only significant ones
        Removes levels that are too close together

        Args:
            levels: List of price levels
            level_type: 'support' or 'resistance'
            epic: Trading instrument for pair-aware pip calculation
        """
        if not levels:
            return []

        pip_size = self._get_pip_size(epic) if epic else 0.0001
        filtered_levels = []

        for level in sorted(levels, reverse=(level_type == 'resistance')):
            is_significant = True

            for existing_level in filtered_levels:
                pip_distance = abs(level - existing_level) / pip_size  # Pair-aware pip calculation
                if pip_distance < self.min_level_distance_pips:
                    is_significant = False
                    break

            if is_significant:
                filtered_levels.append(level)

                # Limit to top 5 levels to avoid clutter
                if len(filtered_levels) >= 5:
                    break

        return filtered_levels

    def _find_nearest_support(self, support_levels: List[float], current_price: float, epic: str = None) -> Optional[float]:
        """
        Find nearest support level below current price
        ENHANCED: Also considers levels AT current price (within small tolerance)

        Args:
            support_levels: List of support price levels
            current_price: Current market price
            epic: Trading instrument for pair-aware pip calculation
        """
        pip_size = self._get_pip_size(epic) if epic else 0.0001
        pip_tolerance = pip_size * 2  # 2 pips tolerance for "AT" level detection
        supports_below = [level for level in support_levels if level <= current_price + pip_tolerance]
        return max(supports_below) if supports_below else None

    def _find_nearest_resistance(self, resistance_levels: List[float], current_price: float, epic: str = None) -> Optional[float]:
        """
        Find nearest resistance level above current price
        ENHANCED: Also considers levels AT current price (within small tolerance)

        Args:
            resistance_levels: List of resistance price levels
            current_price: Current market price
            epic: Trading instrument for pair-aware pip calculation
        """
        pip_size = self._get_pip_size(epic) if epic else 0.0001
        pip_tolerance = pip_size * 2  # 2 pips tolerance for "AT" level detection
        resistances_above = [level for level in resistance_levels if level >= current_price - pip_tolerance]
        return min(resistances_above) if resistances_above else None
    
    def _check_level_proximity(self,
                              current_price: float,
                              signal_type: str,
                              levels: Dict,
                              df: pd.DataFrame,
                              epic: str = None,
                              tolerance_override: float = None) -> Dict:
        """
        Check if trade direction conflicts with nearby major levels

        ENHANCED Key Logic:
        - For BUY signals: Check ALL support levels if we're AT one (within tolerance)
        - For SELL signals: Check ALL resistance levels if we're AT one (within tolerance)
        - Also check nearest resistance (BUY) or support (SELL) in the trade direction
        - Consider volume confirmation for recent level breaks

        Args:
            current_price: Current market price
            signal_type: 'BUY', 'SELL', 'BULL', or 'BEAR'
            levels: Dictionary with support/resistance levels
            df: Price dataframe for volume analysis
            epic: Trading instrument epic (for pair-aware pip size)
            tolerance_override: Optional per-signal tolerance override in pips (v2.35.1)
        """
        pip_size = self._get_pip_size(epic) if epic else 0.0001

        # v2.35.1: Use tolerance override if provided, otherwise use default
        effective_tolerance = tolerance_override if tolerance_override is not None else self.level_tolerance_pips

        all_support_levels = levels.get('support_levels', [])
        all_resistance_levels = levels.get('resistance_levels', [])
        nearest_support = levels.get('nearest_support')
        nearest_resistance = levels.get('nearest_resistance')

        # BUY/BULL signal validation
        if signal_type in ['BUY', 'BULL']:
            # FIRST: Check if we're AT any resistance level (price sitting on resistance)
            for resistance in all_resistance_levels:
                distance_to_resistance = abs(resistance - current_price) / pip_size

                # If we're AT or very close to a resistance level
                if distance_to_resistance <= effective_tolerance:
                    # Check if resistance was recently broken with volume
                    volume_break = self._check_volume_break(df, resistance, 'resistance')

                    if not volume_break:
                        return {
                            'is_valid': False,
                            'reason': f"BUY signal rejected - price AT resistance level {resistance:.5f} "
                                    f"({distance_to_resistance:.1f} pips away, minimum: {effective_tolerance})"
                        }
                    else:
                        # Volume break confirmed, allow trade
                        return {
                            'is_valid': True,
                            'reason': f"BUY signal allowed - resistance at {resistance:.5f} "
                                    f"recently broken with volume confirmation"
                        }

            # SECOND: Check nearest resistance above (if price is below resistance)
            if nearest_resistance and nearest_resistance > current_price:
                distance_to_resistance = (nearest_resistance - current_price) / pip_size

                if distance_to_resistance <= effective_tolerance:
                    # Check if resistance was recently broken with volume
                    volume_break = self._check_volume_break(df, nearest_resistance, 'resistance')

                    if not volume_break:
                        return {
                            'is_valid': False,
                            'reason': f"BUY signal too close to resistance at {nearest_resistance:.5f} "
                                    f"({distance_to_resistance:.1f} pips away, minimum: {effective_tolerance})"
                        }

        # SELL/BEAR signal validation
        elif signal_type in ['SELL', 'BEAR']:
            # FIRST: Check if we're AT any support level (price sitting on support)
            for support in all_support_levels:
                distance_to_support = abs(support - current_price) / pip_size

                # If we're AT or very close to a support level
                if distance_to_support <= effective_tolerance:
                    # Check if support was recently broken with volume
                    volume_break = self._check_volume_break(df, support, 'support')

                    if not volume_break:
                        return {
                            'is_valid': False,
                            'reason': f"SELL signal rejected - price AT support level {support:.5f} "
                                    f"({distance_to_support:.1f} pips away, minimum: {effective_tolerance})"
                        }
                    else:
                        # Volume break confirmed, allow trade
                        return {
                            'is_valid': True,
                            'reason': f"SELL signal allowed - support at {support:.5f} "
                                    f"recently broken with volume confirmation"
                        }

            # SECOND: Check nearest support below (if price is above support)
            if nearest_support and nearest_support < current_price:
                distance_to_support = (current_price - nearest_support) / pip_size

                if distance_to_support <= effective_tolerance:
                    # Check if support was recently broken with volume
                    volume_break = self._check_volume_break(df, nearest_support, 'support')

                    if not volume_break:
                        return {
                            'is_valid': False,
                            'reason': f"SELL signal too close to support at {nearest_support:.5f} "
                                    f"({distance_to_support:.1f} pips away, minimum: {effective_tolerance})"
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

    # =========================================================================
    # PATH-TO-TARGET BLOCKING CHECK (NEW)
    # =========================================================================

    # Pair-specific blocking thresholds
    PAIR_BLOCKING_THRESHOLDS = {
        'EURUSD': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 5},
        'GBPUSD': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 8},
        'USDJPY': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 5},
        'GBPJPY': {'critical_block_pct': 20, 'warning_block_pct': 45, 'min_sr_distance_pips': 10},
        'AUDUSD': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 4},
        'USDCHF': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 4},
        'USDCAD': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 5},
        'NZDUSD': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 4},
        'AUDJPY': {'critical_block_pct': 25, 'warning_block_pct': 50, 'min_sr_distance_pips': 7},
        'EURJPY': {'critical_block_pct': 22, 'warning_block_pct': 48, 'min_sr_distance_pips': 8},
    }

    DEFAULT_BLOCKING_THRESHOLDS = {
        'critical_block_pct': 25,  # Block if S/R within 25% of path (75%+ blocked)
        'warning_block_pct': 50,   # Warning if S/R within 50% of path
        'min_sr_distance_pips': 5
    }

    def _get_pair_blocking_thresholds(self, epic: str) -> Dict:
        """Get pair-specific thresholds for S/R blocking validation"""
        # Extract pair from epic (e.g., 'CS.D.EURUSD.CEEM.IP' -> 'EURUSD')
        pair = epic.split('.')[2] if '.' in epic and len(epic.split('.')) > 2 else epic
        return self.PAIR_BLOCKING_THRESHOLDS.get(pair, self.DEFAULT_BLOCKING_THRESHOLDS)

    def check_path_to_target_blocking(
        self,
        entry_price: float,
        take_profit: float,
        signal_type: str,
        levels: Dict,
        epic: str
    ) -> Tuple[bool, str, Dict]:
        """
        Check if S/R level blocks the path from entry to take profit target.

        This is the KEY validation that prevents trades where S/R is blocking
        most of the profit potential.

        Args:
            entry_price: Trade entry price
            take_profit: Take profit target price
            signal_type: 'BUY', 'SELL', 'BULL', or 'BEAR'
            levels: Dictionary with support/resistance levels
            epic: Trading instrument for pair-aware calculations

        Returns:
            Tuple of (is_blocked, reason, details)
            - is_blocked: True if S/R is critically blocking the path
            - reason: Human-readable explanation
            - details: Dict with analysis data for logging/storage
        """
        try:
            pip_size = self._get_pip_size(epic)
            target_distance_pips = abs(take_profit - entry_price) / pip_size

            # Skip if target distance is too small
            if target_distance_pips < 5:
                return False, "Target distance too small for path analysis", {
                    'target_distance_pips': target_distance_pips,
                    'analysis_skipped': True
                }

            # Get pair-specific blocking thresholds
            thresholds = self._get_pair_blocking_thresholds(epic)
            critical_pct = thresholds['critical_block_pct']
            warning_pct = thresholds['warning_block_pct']
            min_sr_dist = thresholds['min_sr_distance_pips']

            blocking_sr = None
            blocking_type = None
            distance_to_sr_pips = None
            path_blocked_pct = None

            if signal_type.upper() in ['BUY', 'BULL']:
                # For BUY: check resistance levels BETWEEN entry and TP
                for resistance in levels.get('resistance_levels', []):
                    if entry_price < resistance < take_profit:
                        dist_to_sr = (resistance - entry_price) / pip_size

                        # Skip if S/R is very close (might be noise)
                        if dist_to_sr < min_sr_dist:
                            continue

                        # Calculate how much of the path is blocked
                        blocked_pct = (1 - (dist_to_sr / target_distance_pips)) * 100

                        # Track the most problematic S/R level (closest to entry)
                        if blocking_sr is None or dist_to_sr < distance_to_sr_pips:
                            blocking_sr = resistance
                            blocking_type = 'resistance'
                            distance_to_sr_pips = dist_to_sr
                            path_blocked_pct = blocked_pct

            elif signal_type.upper() in ['SELL', 'BEAR']:
                # For SELL: check support levels BETWEEN entry and TP
                for support in levels.get('support_levels', []):
                    if take_profit < support < entry_price:
                        dist_to_sr = (entry_price - support) / pip_size

                        # Skip if S/R is very close (might be noise)
                        if dist_to_sr < min_sr_dist:
                            continue

                        # Calculate how much of the path is blocked
                        blocked_pct = (1 - (dist_to_sr / target_distance_pips)) * 100

                        # Track the most problematic S/R level (closest to entry)
                        if blocking_sr is None or dist_to_sr < distance_to_sr_pips:
                            blocking_sr = support
                            blocking_type = 'support'
                            distance_to_sr_pips = dist_to_sr
                            path_blocked_pct = blocked_pct

            # Build result details
            details = {
                'entry_price': entry_price,
                'take_profit': take_profit,
                'target_distance_pips': round(target_distance_pips, 1),
                'signal_type': signal_type,
                'epic': epic,
                'blocking_sr_level': blocking_sr,
                'blocking_sr_type': blocking_type,
                'distance_to_sr_pips': round(distance_to_sr_pips, 1) if distance_to_sr_pips else None,
                'path_blocked_pct': round(path_blocked_pct, 1) if path_blocked_pct else 0,
                'thresholds': thresholds,
                'is_critical': False,
                'is_warning': False
            }

            # No blocking S/R found
            if blocking_sr is None:
                return False, "Path to target clear - no S/R blocking", details

            # Check if blocking is critical (should reject trade)
            if path_blocked_pct >= (100 - critical_pct):
                details['is_critical'] = True
                reason = (
                    f"CRITICAL: {blocking_type.upper()} at {blocking_sr:.5f} blocks "
                    f"{path_blocked_pct:.0f}% of path to TP "
                    f"(only {distance_to_sr_pips:.1f} pips before S/R, target: {target_distance_pips:.1f} pips)"
                )
                self.logger.warning(f"🚧 {epic}: {reason}")
                return True, reason, details

            # Check if blocking is warning level (allow but flag)
            if path_blocked_pct >= (100 - warning_pct):
                details['is_warning'] = True
                reason = (
                    f"WARNING: {blocking_type.upper()} at {blocking_sr:.5f} blocks "
                    f"{path_blocked_pct:.0f}% of path to TP "
                    f"({distance_to_sr_pips:.1f} pips to S/R)"
                )
                self.logger.info(f"⚠️ {epic}: {reason}")
                return False, reason, details

            # S/R exists but not critically blocking
            reason = (
                f"S/R at {blocking_sr:.5f} in path but not blocking significantly "
                f"({path_blocked_pct:.0f}% blocked, threshold: {100-critical_pct}%)"
            )
            return False, reason, details

        except Exception as e:
            self.logger.error(f"❌ Error checking path-to-target blocking: {e}")
            return False, f"Path blocking check error: {str(e)}", {'error': str(e)}

    def validate_with_path_blocking(
        self,
        signal: Dict,
        df: pd.DataFrame,
        epic: str
    ) -> Tuple[bool, str, Dict]:
        """
        Complete S/R validation including path-to-target blocking check.

        This combines the existing proximity check with the new path blocking check.

        Args:
            signal: Trading signal with entry_price, take_profit, signal_type
            df: Price data DataFrame
            epic: Trading instrument

        Returns:
            Tuple of (is_valid, reason, details)
        """
        # First, run standard proximity validation
        is_valid, reason, details = self.validate_trade_direction(signal, df, epic)

        if not is_valid:
            return is_valid, reason, details

        # Extract prices for path blocking check
        entry_price = self._get_current_price(signal)
        signal_type = signal.get('signal_type', '').upper()

        # ================================================================
        # FIX: Use per-pair fixed TP from config if available for path blocking
        # This prevents overly aggressive path blocking when dynamic TPs are large
        # (Dynamic TPs from swing targets can be 45+ pips, causing 80%+ blocking)
        # ================================================================
        take_profit = None
        try:
            from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
            config = get_smc_simple_config()
            fixed_tp_pips = config.get_pair_fixed_take_profit(epic)

            if fixed_tp_pips and entry_price:
                pip_size = self._get_pip_size(epic)
                # Calculate TP price from fixed pips
                if signal_type in ['BUY', 'BULL']:
                    take_profit = entry_price + (fixed_tp_pips * pip_size)
                else:
                    take_profit = entry_price - (fixed_tp_pips * pip_size)
                self.logger.debug(f"🎯 Using fixed TP for path blocking: {fixed_tp_pips} pips = {take_profit:.5f}")
        except Exception as e:
            self.logger.debug(f"Could not get fixed TP from config: {e}")

        # Fallback to signal's dynamic TP if no fixed TP available
        if take_profit is None:
            take_profit = signal.get('take_profit') or signal.get('tp_price')

        if not entry_price or not take_profit:
            details['path_blocking_skipped'] = True
            details['path_blocking_reason'] = "Missing entry or TP price"
            return is_valid, reason, details

        # Get levels for path blocking check
        levels = self._get_support_resistance_levels(df, epic)

        # Check path-to-target blocking
        is_blocked, block_reason, block_details = self.check_path_to_target_blocking(
            entry_price=entry_price,
            take_profit=take_profit,
            signal_type=signal_type,
            levels=levels,
            epic=epic
        )

        # Merge blocking details into main details
        details['path_blocking'] = block_details

        if is_blocked:
            return False, block_reason, details

        # Add path blocking info to reason if there was a warning
        if block_details.get('is_warning'):
            reason = f"{reason} | {block_reason}"

        return True, reason, details


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