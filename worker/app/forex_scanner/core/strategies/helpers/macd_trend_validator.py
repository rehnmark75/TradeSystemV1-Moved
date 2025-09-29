# core/strategies/helpers/macd_trend_validator.py
"""
MACD Trend Validator Module
Validates MACD signals against trend indicators and momentum filters
"""

import pandas as pd
import logging
from typing import Optional, Dict
try:
    from configdata import config
    from configdata.strategies.config_macd_strategy import MACD_DISABLE_EMA200_FILTER
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies.config_macd_strategy import MACD_DISABLE_EMA200_FILTER
    except ImportError:
        MACD_DISABLE_EMA200_FILTER = False  # Fallback


class MACDTrendValidator:
    """Handles all trend and momentum validation for MACD signals"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = 1e-8  # Epsilon for stability
    
    def validate_ema_200_trend(self, row: pd.Series, signal_type: str) -> bool:
        """
        EMA 200 TREND FILTER: Ensure signals align with major trend direction
        
        Critical trend filter:
        - BUY signals: Price must be ABOVE EMA 200 (uptrend)
        - SELL signals: Price must be BELOW EMA 200 (downtrend)
        
        Args:
            row: DataFrame row with price and EMA data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if trend is correct, False if against major trend
        """
        try:
            # Check MACD-specific EMA200 filter setting first
            if MACD_DISABLE_EMA200_FILTER:
                return True

            # Fallback to global setting if MACD-specific setting not available
            if not getattr(config, 'EMA_200_TREND_FILTER_ENABLED', True):
                return True
            
            buffer_pips = getattr(config, 'EMA_200_BUFFER_PIPS', 1.0)
            
            # Get EMA 200 and price data
            ema_200 = row.get('ema_200', 0)
            if ema_200 == 0:
                self.logger.debug("EMA 200 not available for trend validation")
                return True  # Allow signal if EMA 200 not available
                
            close_price = row.get('close', 0)
            if close_price == 0:
                self.logger.debug("Close price not available for trend validation")
                return True
            
            # Calculate buffer (convert pips to price units)
            # For most forex pairs, 1 pip = 0.0001, for JPY pairs 1 pip = 0.01
            epic = row.get('epic', '')
            if 'JPY' in epic:
                pip_value = 0.01
            else:
                pip_value = 0.0001
                
            buffer = buffer_pips * pip_value
            
            # Validate trend alignment with buffer
            if signal_type == 'BULL':
                # Bullish signals need price above EMA 200 (minus buffer)
                trend_valid = close_price > (ema_200 - buffer)
                if not trend_valid:
                    self.logger.debug(f"BULL signal rejected: price {close_price:.5f} not above EMA200 {ema_200:.5f}")
                return trend_valid
            
            elif signal_type == 'BEAR':
                # Bearish signals need price below EMA 200 (plus buffer)
                trend_valid = close_price < (ema_200 + buffer)
                if not trend_valid:
                    self.logger.debug(f"BEAR signal rejected: price {close_price:.5f} not below EMA200 {ema_200:.5f}")
                return trend_valid
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating EMA 200 trend: {e}")
            return True  # Allow on error to avoid blocking signals
    
    def validate_macd_histogram_direction(self, row: pd.Series, signal_type: str) -> bool:
        """
        MACD HISTOGRAM VALIDATION: Ensure histogram supports signal direction
        
        Args:
            row: DataFrame row with MACD data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if histogram supports signal direction
        """
        try:
            histogram = row.get('macd_histogram', 0)
            
            if signal_type == 'BULL':
                # Bull signals need positive histogram
                valid = histogram > 0
                if not valid:
                    self.logger.debug(f"BULL signal rejected: negative histogram {histogram:.6f}")
                return valid
                
            elif signal_type == 'BEAR':
                # Bear signals need negative histogram
                valid = histogram < 0
                if not valid:
                    self.logger.debug(f"BEAR signal rejected: positive histogram {histogram:.6f}")
                return valid
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating MACD histogram direction: {e}")
            return True
    
    def validate_macd_line_signal_alignment(self, row: pd.Series, signal_type: str) -> bool:
        """
        MACD LINE vs SIGNAL LINE VALIDATION: Ensure proper alignment
        
        Args:
            row: DataFrame row with MACD data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if MACD line and signal line are properly aligned
        """
        try:
            macd_line = row.get('macd_line', 0)
            macd_signal = row.get('macd_signal', 0)
            
            if macd_line == 0 or macd_signal == 0:
                self.logger.debug("MACD line or signal not available")
                return True  # Allow if data not available
            
            if signal_type == 'BULL':
                # Bull signals prefer MACD line above signal line
                aligned = macd_line > macd_signal
                if not aligned:
                    self.logger.debug(f"BULL signal: MACD line {macd_line:.6f} below signal {macd_signal:.6f}")
                return aligned
                
            elif signal_type == 'BEAR':
                # Bear signals prefer MACD line below signal line
                aligned = macd_line < macd_signal
                if not aligned:
                    self.logger.debug(f"BEAR signal: MACD line {macd_line:.6f} above signal {macd_signal:.6f}")
                return aligned
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating MACD line-signal alignment: {e}")
            return True
    
    def validate_rsi_confluence(self, row: pd.Series, signal_type: str) -> bool:
        """
        RSI CONFLUENCE VALIDATION: Ensure RSI supports signal direction

        Args:
            row: DataFrame row with RSI data
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if RSI supports signal direction
        """
        try:
            rsi = row.get('rsi', 50)

            if signal_type == 'BULL':
                # Bull signals prefer RSI not overbought (< 75)
                # Ideal range is 30-70, acceptable up to 75
                valid = rsi < 75
                if not valid:
                    self.logger.debug(f"BULL signal rejected: RSI overbought at {rsi:.1f}")
                return valid

            elif signal_type == 'BEAR':
                # Bear signals prefer RSI not oversold (> 25)
                # Ideal range is 30-70, acceptable down to 25
                valid = rsi > 25
                if not valid:
                    self.logger.debug(f"BEAR signal rejected: RSI oversold at {rsi:.1f}")
                return valid

            return False

        except Exception as e:
            self.logger.error(f"Error validating RSI confluence: {e}")
            return True  # Allow on error to avoid blocking signals

    def validate_macd_momentum(self, row: pd.Series, signal_type: str) -> bool:
        """
        MACD MOMENTUM VALIDATION: Check if MACD is gaining momentum in signal direction
        
        Args:
            row: DataFrame row with MACD data and previous values
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            True if momentum supports signal
        """
        try:
            macd_line = row.get('macd_line', 0)
            macd_prev = row.get('macd_line_prev', macd_line)
            
            histogram = row.get('macd_histogram', 0)
            histogram_prev = row.get('macd_histogram_prev', histogram)
            
            if signal_type == 'BULL':
                # Bull signals prefer rising MACD and growing positive histogram
                macd_rising = macd_line >= macd_prev
                histogram_growing = histogram >= histogram_prev
                momentum_good = macd_rising and histogram_growing
                
                if not momentum_good:
                    self.logger.debug(f"BULL momentum weak: MACD rising={macd_rising}, hist growing={histogram_growing}")
                return momentum_good
                
            elif signal_type == 'BEAR':
                # Bear signals prefer falling MACD and growing negative histogram
                macd_falling = macd_line <= macd_prev
                histogram_growing = histogram <= histogram_prev  # More negative = growing for bears
                momentum_good = macd_falling and histogram_growing
                
                if not momentum_good:
                    self.logger.debug(f"BEAR momentum weak: MACD falling={macd_falling}, hist growing={histogram_growing}")
                return momentum_good
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating MACD momentum: {e}")
            return True
    
    def validate_all_trend_filters(self, row: pd.Series, signal_type: str) -> Dict:
        """
        Run all trend validation filters and return detailed results
        
        Args:
            row: DataFrame row with all indicator data
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Dictionary with validation results for each filter
        """
        try:
            results = {
                'ema_200_trend': self.validate_ema_200_trend(row, signal_type),
                'histogram_direction': self.validate_macd_histogram_direction(row, signal_type),
                'line_signal_alignment': self.validate_macd_line_signal_alignment(row, signal_type),
                'momentum_check': self.validate_macd_momentum(row, signal_type),
                'rsi_confluence': self.validate_rsi_confluence(row, signal_type)
            }
            
            # Calculate overall pass rate
            passed = sum(1 for result in results.values() if result)
            total = len(results)
            results['overall_pass_rate'] = passed / total if total > 0 else 0.0
            results['all_passed'] = all(results.values())
            
            # Log summary
            if not results['all_passed']:
                failed_filters = [name for name, result in results.items() 
                                if not result and name not in ['overall_pass_rate', 'all_passed']]
                self.logger.debug(f"{signal_type} signal failed filters: {failed_filters}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running trend validation: {e}")
            return {'error': str(e), 'all_passed': False}
    
    def apply_lenient_validation(self, validation_results: Dict, min_pass_rate: float = 0.75) -> bool:
        """
        Apply lenient validation - allow signals if most filters pass
        
        Args:
            validation_results: Results from validate_all_trend_filters
            min_pass_rate: Minimum pass rate required (0.75 = 75%)
            
        Returns:
            True if signal should be allowed under lenient rules
        """
        try:
            pass_rate = validation_results.get('overall_pass_rate', 0.0)
            
            # Always require EMA 200 trend filter and histogram direction
            critical_filters_pass = (
                validation_results.get('ema_200_trend', False) and
                validation_results.get('histogram_direction', False)
            )
            
            lenient_pass = critical_filters_pass and (pass_rate >= min_pass_rate)
            
            if lenient_pass and not validation_results.get('all_passed', False):
                self.logger.debug(f"Signal allowed under lenient validation: {pass_rate:.2%} pass rate")
            
            return lenient_pass
            
        except Exception as e:
            self.logger.error(f"Error applying lenient validation: {e}")
            return False