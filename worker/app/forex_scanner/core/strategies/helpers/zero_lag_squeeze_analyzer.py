# core/strategies/helpers/zero_lag_squeeze_analyzer.py
"""
Zero Lag Squeeze Momentum Analyzer Module
Implements LazyBear's Squeeze Momentum Indicator for Zero Lag strategy validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple
try:
    import config
except ImportError:
    from forex_scanner import config


class ZeroLagSqueezeAnalyzer:
    """Handles Squeeze Momentum calculations and validation for Zero Lag strategy"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = 1e-8  # Epsilon for stability
    
    def calculate_squeeze_momentum(self, df: pd.DataFrame, 
                                   bb_length: int = 20, bb_mult: float = 2.0,
                                   kc_length: int = 20, kc_mult: float = 1.5,
                                   use_true_range: bool = True) -> pd.DataFrame:
        """
        Calculate Squeeze Momentum Indicator based on LazyBear's PineScript
        
        Original PineScript Logic:
        - Bollinger Bands squeeze detection
        - Keltner Channels comparison  
        - Linear regression momentum calculation
        - Color-coded histogram output
        
        Args:
            df: DataFrame with OHLC data
            bb_length: Bollinger Bands period (default: 20)
            bb_mult: Bollinger Bands multiplier (default: 2.0)
            kc_length: Keltner Channels period (default: 20)
            kc_mult: Keltner Channels multiplier (default: 1.5)
            use_true_range: Use True Range for KC calculation
            
        Returns:
            DataFrame with squeeze momentum columns added
        """
        try:
            if df is None or df.empty or len(df) < max(bb_length, kc_length):
                self.logger.debug("Insufficient data for squeeze momentum calculation")
                return df
            
            df = df.copy()
            source = df['close']
            high = df['high']
            low = df['low']
            
            # === BOLLINGER BANDS CALCULATION ===
            basis = source.rolling(window=bb_length).mean()
            std_dev = source.rolling(window=bb_length).std()
            dev = bb_mult * std_dev
            upper_bb = basis + dev
            lower_bb = basis - dev
            
            # === KELTNER CHANNELS CALCULATION ===
            ma = source.rolling(window=kc_length).mean()
            
            if use_true_range:
                # True Range calculation
                tr1 = high - low
                tr2 = (high - source.shift(1)).abs()
                tr3 = (low - source.shift(1)).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                range_ma = true_range.rolling(window=kc_length).mean()
            else:
                # Simple range
                range_ma = (high - low).rolling(window=kc_length).mean()
            
            upper_kc = ma + range_ma * kc_mult
            lower_kc = ma - range_ma * kc_mult
            
            # === SQUEEZE DETECTION ===
            # Squeeze ON: BB inside KC (low volatility)
            sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
            
            # Squeeze OFF: BB outside KC (high volatility breakout)
            sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
            
            # No Squeeze: Neither condition
            no_sqz = ~sqz_on & ~sqz_off
            
            # === MOMENTUM CALCULATION ===
            # Linear regression of price relative to midpoint
            highest_high = high.rolling(window=kc_length).max()
            lowest_low = low.rolling(window=kc_length).min()
            midpoint = (highest_high + lowest_low) / 2
            sma_close = source.rolling(window=kc_length).mean()
            avg_midpoint = (midpoint + sma_close) / 2
            
            # Calculate momentum value using linear regression
            momentum_input = source - avg_midpoint
            
            # Linear regression calculation (simplified)
            def calculate_linreg(series, period):
                """Calculate linear regression value"""
                if len(series) < period:
                    return pd.Series([0] * len(series), index=series.index)
                
                result = []
                for i in range(len(series)):
                    if i < period - 1:
                        result.append(0)
                    else:
                        y_vals = series.iloc[i-period+1:i+1].values
                        x_vals = np.arange(period)
                        
                        if len(y_vals) == period and not np.isnan(y_vals).all():
                            # Linear regression slope * period gives the trend value
                            slope = np.polyfit(x_vals, y_vals, 1)[0]
                            result.append(slope * period)
                        else:
                            result.append(0)
                
                return pd.Series(result, index=series.index)
            
            momentum_val = calculate_linreg(momentum_input, kc_length)
            
            # === COLOR DETERMINATION (HISTOGRAM) ===
            # Green/Lime: Positive and increasing
            # Red/Maroon: Negative and decreasing
            momentum_prev = momentum_val.shift(1).fillna(0)
            
            is_positive = momentum_val > 0
            is_increasing = momentum_val > momentum_prev
            
            # Determine colors based on LazyBear's logic
            is_lime = is_positive & is_increasing
            is_green = is_positive & ~is_increasing  
            is_red = ~is_positive & ~is_increasing
            is_maroon = ~is_positive & is_increasing
            
            # === ADD TO DATAFRAME ===
            df['squeeze_bb_upper'] = upper_bb
            df['squeeze_bb_lower'] = lower_bb
            df['squeeze_kc_upper'] = upper_kc
            df['squeeze_kc_lower'] = lower_kc
            
            df['squeeze_on'] = sqz_on
            df['squeeze_off'] = sqz_off
            df['squeeze_none'] = no_sqz
            
            df['squeeze_momentum'] = momentum_val
            df['squeeze_momentum_prev'] = momentum_prev
            
            # Color states
            df['squeeze_is_lime'] = is_lime      # Positive and increasing (strongest bull)
            df['squeeze_is_green'] = is_green    # Positive but decreasing
            df['squeeze_is_red'] = is_red        # Negative and decreasing (strongest bear)  
            df['squeeze_is_maroon'] = is_maroon  # Negative but increasing
            
            # Convenience flags
            df['squeeze_bullish'] = is_lime | is_green  # Any positive momentum
            df['squeeze_bearish'] = is_red | is_maroon  # Any negative momentum
            df['squeeze_strong_bull'] = is_lime         # Strong bullish (lime)
            df['squeeze_strong_bear'] = is_red          # Strong bearish (red)
            
            # Squeeze state for easy access
            df['squeeze_state'] = 'none'
            df.loc[sqz_on, 'squeeze_state'] = 'on'
            df.loc[sqz_off, 'squeeze_state'] = 'off'
            
            self.logger.debug(f"Squeeze momentum calculated for {len(df)} bars")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating squeeze momentum: {e}")
            return df
    
    def validate_squeeze_momentum(self, latest_row: pd.Series, signal_type: str) -> bool:
        """
        Validate signal against Squeeze Momentum direction
        
        Args:
            latest_row: DataFrame row with squeeze momentum data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if momentum supports signal direction
        """
        try:
            if not getattr(config, 'SQUEEZE_MOMENTUM_ENABLED', True):
                return True
            
            momentum = latest_row.get('squeeze_momentum', 0)
            is_lime = latest_row.get('squeeze_is_lime', False)
            is_green = latest_row.get('squeeze_is_green', False)
            is_red = latest_row.get('squeeze_is_red', False)
            is_maroon = latest_row.get('squeeze_is_maroon', False)
            
            if signal_type == 'BULL':
                # RELAXED: For bullish signals, allow positive momentum OR neutral
                if is_lime:
                    self.logger.debug("✅ Squeeze Momentum LIME confirms BULL signal")
                    return True
                elif is_green:
                    self.logger.debug("🟢 Squeeze Momentum GREEN allows BULL signal")
                    return True
                elif momentum >= 0:  # Allow any non-negative momentum
                    self.logger.debug(f"⚪ Squeeze Momentum {momentum:.4f} neutral/positive for BULL")
                    return True
                else:
                    self.logger.debug(f"❌ Squeeze Momentum {momentum:.4f} negative - conflicts with BULL signal")
                    return False
            
            elif signal_type == 'BEAR':
                # RELAXED: For bearish signals, allow negative momentum OR neutral
                if is_red:
                    self.logger.debug("✅ Squeeze Momentum RED confirms BEAR signal")
                    return True
                elif is_maroon:
                    self.logger.debug("🟤 Squeeze Momentum MAROON allows BEAR signal") 
                    return True
                elif momentum <= 0:  # Allow any non-positive momentum
                    self.logger.debug(f"⚪ Squeeze Momentum {momentum:.4f} neutral/negative for BEAR")
                    return True
                else:
                    self.logger.debug(f"❌ Squeeze Momentum {momentum:.4f} positive - conflicts with BEAR signal")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating squeeze momentum: {e}")
            return True  # Allow signal on error
    
    def get_squeeze_confidence_boost(self, latest_row: pd.Series, signal_type: str) -> float:
        """
        Calculate confidence boost based on squeeze momentum conditions
        
        Args:
            latest_row: DataFrame row with squeeze momentum data  
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Confidence boost value (0.0 to 0.2)
        """
        try:
            if not getattr(config, 'SQUEEZE_MOMENTUM_ENABLED', True):
                return 0.0
            
            momentum = latest_row.get('squeeze_momentum', 0)
            squeeze_state = latest_row.get('squeeze_state', 'none')
            is_lime = latest_row.get('squeeze_is_lime', False)
            is_green = latest_row.get('squeeze_is_green', False)
            is_red = latest_row.get('squeeze_is_red', False)
            is_maroon = latest_row.get('squeeze_is_maroon', False)
            
            boost = 0.0
            max_boost = getattr(config, 'SQUEEZE_MOMENTUM_MAX_BOOST', 0.2)
            
            # === MOMENTUM DIRECTION BONUS ===
            if signal_type == 'BULL':
                if is_lime:
                    boost += 0.15  # Strongest bullish momentum
                    self.logger.debug("Squeeze boost: +15% for LIME momentum")
                elif is_green:
                    boost += 0.08  # Moderate bullish momentum
                    self.logger.debug("Squeeze boost: +8% for GREEN momentum")
                else:
                    boost -= 0.1  # Penalty for wrong momentum direction
                    self.logger.debug("Squeeze penalty: -10% for bearish momentum")
            
            elif signal_type == 'BEAR':
                if is_red:
                    boost += 0.15  # Strongest bearish momentum
                    self.logger.debug("Squeeze boost: +15% for RED momentum")
                elif is_maroon:
                    boost += 0.08  # Moderate bearish momentum
                    self.logger.debug("Squeeze boost: +8% for MAROON momentum")
                else:
                    boost -= 0.1  # Penalty for wrong momentum direction
                    self.logger.debug("Squeeze penalty: -10% for bullish momentum")
            
            # === SQUEEZE STATE BONUS ===
            if squeeze_state == 'off':
                # Recent squeeze release - high probability breakout
                boost += 0.05
                self.logger.debug("Squeeze boost: +5% for recent squeeze release")
            elif squeeze_state == 'on':
                # Currently in squeeze - reduce confidence
                boost -= 0.03
                self.logger.debug("Squeeze penalty: -3% for active squeeze")
            
            # === MOMENTUM STRENGTH BONUS ===
            momentum_strength = abs(momentum)
            if momentum_strength > 0.001:  # Strong momentum
                strength_boost = min(0.05, momentum_strength * 50)  # Cap at 5%
                boost += strength_boost
                self.logger.debug(f"Squeeze boost: +{strength_boost:.1%} for momentum strength {momentum_strength:.4f}")
            
            # Cap the total boost
            final_boost = max(-0.15, min(max_boost, boost))
            
            self.logger.debug(f"Total squeeze momentum boost: {final_boost:+.1%}")
            return final_boost
            
        except Exception as e:
            self.logger.error(f"Error calculating squeeze confidence boost: {e}")
            return 0.0
    
    def detect_squeeze_release(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        Detect if a squeeze was recently released (high probability setup)
        
        Args:
            df: DataFrame with squeeze data
            current_idx: Current bar index
            
        Returns:
            True if squeeze was recently released
        """
        try:
            lookback = getattr(config, 'SQUEEZE_RELEASE_LOOKBACK', 5)
            
            if current_idx < lookback:
                return False
            
            # Check if squeeze was ON in recent past and is now OFF
            recent_data = df.iloc[current_idx-lookback:current_idx+1]
            
            # Was squeeze on recently?
            had_squeeze = recent_data['squeeze_on'].any()
            
            # Is squeeze off now?
            current_squeeze_off = df.iloc[current_idx]['squeeze_off']
            
            if had_squeeze and current_squeeze_off:
                self.logger.debug("🎯 Squeeze release detected - high probability setup")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting squeeze release: {e}")
            return False
    
    def get_squeeze_summary(self, latest_row: pd.Series) -> Dict:
        """
        Get comprehensive squeeze momentum summary for logging/analysis
        
        Args:
            latest_row: Current market data row
            
        Returns:
            Dictionary with squeeze momentum summary
        """
        try:
            return {
                'momentum_value': latest_row.get('squeeze_momentum', 0),
                'squeeze_state': latest_row.get('squeeze_state', 'unknown'),
                'is_lime': latest_row.get('squeeze_is_lime', False),
                'is_green': latest_row.get('squeeze_is_green', False), 
                'is_red': latest_row.get('squeeze_is_red', False),
                'is_maroon': latest_row.get('squeeze_is_maroon', False),
                'bullish': latest_row.get('squeeze_bullish', False),
                'bearish': latest_row.get('squeeze_bearish', False),
                'strong_bull': latest_row.get('squeeze_strong_bull', False),
                'strong_bear': latest_row.get('squeeze_strong_bear', False),
                'bb_upper': latest_row.get('squeeze_bb_upper', 0),
                'bb_lower': latest_row.get('squeeze_bb_lower', 0),
                'kc_upper': latest_row.get('squeeze_kc_upper', 0),
                'kc_lower': latest_row.get('squeeze_kc_lower', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating squeeze summary: {e}")
            return {}