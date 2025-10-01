# core/strategies/base_strategy.py
"""
Base Strategy Class - ENHANCED WITH PROPER SIGNAL VALIDATION
Abstract base class for all trading strategies with enhanced confidence scoring
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd
import logging


class BaseStrategy(ABC):
    """Abstract base class for trading strategies with enhanced validation"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # ✅ NEW: Initialize enhanced signal validator
        self._validator = None
        self._initialize_validator()
    
    def _initialize_validator(self):
        """Initialize the enhanced signal validator"""
        try:
            # ✅ IMPORT: Enhanced Signal Validator with fallback pattern
            try:
                from ..detection.enhanced_signal_validator import EnhancedSignalValidator
            except ImportError:
                from forex_scanner.core.detection.enhanced_signal_validator import EnhancedSignalValidator
            self._validator = EnhancedSignalValidator(logger=self.logger)
            self.logger.info(f"[{self.name}] ✅ Enhanced signal validator initialized")
        except ImportError as e:
            self.logger.warning(f"[{self.name}] ❌ Enhanced validator not available: {e}")
            self.logger.warning(f"[{self.name}] ⚠️  Falling back to legacy confidence calculation")
            self.logger.warning(f"[{self.name}] 📝 Create: core/detection/enhanced_signal_validator.py")
            self._validator = None
    
    @abstractmethod
    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '5m'
    ) -> Optional[Dict]:
        """
        Detect trading signal from DataFrame
        
        Args:
            df: Enhanced DataFrame with technical indicators
            epic: Epic code
            spread_pips: Spread in pips for execution price calculation
            timeframe: Timeframe being analyzed
            
        Returns:
            Signal dictionary or None
        """
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for this strategy
        
        Returns:
            List of indicator names needed
        """
        pass
    
    def calculate_confidence(self, signal_data: Dict) -> float:
        """
        ENHANCED: Calculate confidence score using proper market validation
        
        Args:
            signal_data: Signal data dictionary with ALL indicator information
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if self._validator:
                # ✅ USE ENHANCED VALIDATOR: Proper confidence calculation
                should_trade, confidence, reason, analysis = self._validator.validate_signal_enhanced(signal_data)
                
                # Log detailed analysis for debugging
                self.logger.debug(f"[{self.name}] Enhanced validation: {confidence:.1%}")
                self.logger.debug(f"[{self.name}] Reason: {reason}")
                
                # Log component breakdown for analysis
                if 'components' in analysis:
                    components = analysis['components']
                    self.logger.debug(f"[{self.name}] Components: "
                                    f"efficiency={components.get('market_efficiency', 0):.1%}, "
                                    f"ema={components.get('ema_alignment', 0):.1%}, "
                                    f"macd={components.get('macd_strength', 0):.1%}, "
                                    f"trend={components.get('trend_clarity', 0):.1%}")
                
                return confidence
            else:
                # ✅ FALLBACK: Legacy confidence calculation
                return self._calculate_confidence_legacy(signal_data)
                
        except Exception as e:
            self.logger.error(f"[{self.name}] Confidence calculation error: {e}")
            # Return conservative confidence on error
            return 0.3
    
    def _calculate_confidence_legacy(self, signal_data: Dict) -> float:
        """
        Legacy confidence calculation (fallback when enhanced validator unavailable)
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            Confidence score between 0 and 1
        """
        # Basic confidence calculation as fallback
        base_confidence = 0.6
        
        try:
            # Check for some basic indicators
            ema_data = signal_data.get('ema_data', {})
            macd_data = signal_data.get('macd_data', {})
            
            # Basic EMA alignment check
            if ema_data:
                ema_9 = ema_data.get('ema_9', 0)
                ema_21 = ema_data.get('ema_21', 0)
                ema_200 = ema_data.get('ema_200', 0)
                
                if ema_9 and ema_21 and ema_200:
                    # Check for alignment
                    if (ema_9 > ema_21 > ema_200) or (ema_9 < ema_21 < ema_200):
                        base_confidence += 0.1
            
            # Basic MACD check
            if macd_data:
                macd_histogram = macd_data.get('macd_histogram', 0)
                if abs(macd_histogram) > 0.00005:  # Some momentum
                    base_confidence += 0.05
            
            return min(0.95, base_confidence)
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Legacy confidence calculation error: {e}")
            return 0.5
    
    # ✅ NEW: Enhanced signal creation with complete indicator data
    def create_enhanced_signal_data(self, latest_row: pd.Series, signal_type: str) -> Dict:
        """
        Create comprehensive signal data dictionary with ALL indicators for validation
        
        Args:
            latest_row: Latest row from DataFrame with all indicators
            signal_type: 'BULL' or 'BEAR'
            
        Returns:
            Complete signal data dictionary for validation
        """
        try:
            # ✅ CRITICAL: Include ALL indicator data for proper validation
            signal_data = {
                'ema_data': {
                    'ema_short': latest_row.get('ema_short', 0),   # Works for any periods
                    'ema_long': latest_row.get('ema_long', 0),     # Works for any periods  
                    'ema_trend': latest_row.get('ema_trend', 0)    # Works for any periods
                },
                'macd_data': {
                    'macd_line': latest_row.get('macd_line', 0),
                    'macd_signal': latest_row.get('macd_signal', 0),
                    'macd_histogram': latest_row.get('macd_histogram', 0)
                },
                'kama_data': {
                    'kama_value': latest_row.get('kama', latest_row.get('close', 0)),
                    'efficiency_ratio': latest_row.get('kama_efficiency', latest_row.get('efficiency_ratio', 0.5)),  # ← CRITICAL!
                    'kama_trend': 1.0 if signal_type == 'BULL' else -1.0
                },
                'other_indicators': {
                    'atr': latest_row.get('atr', 0),
                    'bb_upper': latest_row.get('bb_upper', 0),
                    'bb_middle': latest_row.get('bb_middle', latest_row.get('close', 0)),
                    'bb_lower': latest_row.get('bb_lower', 0),
                    'rsi': latest_row.get('rsi', 50),
                    'volume': latest_row.get('volume', 0),
                    'volume_ratio': latest_row.get('volume_ratio', 1.0)
                },
                'indicator_count': self._count_available_indicators(latest_row),
                'data_source': 'complete_dataframe_analysis',
                'signal_type': signal_type,
                'price': latest_row.get('close', 0)
            }
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error creating enhanced signal data: {e}")
            # Return minimal data structure on error
            return {
                'ema_data': {},
                'macd_data': {},
                'kama_data': {'efficiency_ratio': 0.1},  # Low efficiency = skip
                'other_indicators': {},
                'signal_type': signal_type,
                'price': latest_row.get('close', 0) if hasattr(latest_row, 'get') else 0
            }
    
    def _count_available_indicators(self, latest_row: pd.Series) -> int:
        """Count how many indicators are available in the data"""
        try:
            indicator_fields = [
                'ema_9', 'ema_21', 'ema_200', 'macd_line', 'macd_signal', 'macd_histogram',
                'kama', 'efficiency_ratio', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
                'rsi', 'volume', 'volume_ratio'
            ]
            
            available_count = 0
            for field in indicator_fields:
                if hasattr(latest_row, 'get'):
                    if latest_row.get(field) is not None and not pd.isna(latest_row.get(field)):
                        available_count += 1
                        
            return available_count
            
        except Exception:
            return 0
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required indicators
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required = self.get_required_indicators()
        available = df.columns.tolist()
        
        missing = [indicator for indicator in required if indicator not in available]
        
        if missing:
            self.logger.warning(f"[{self.name}] Missing indicators: {missing}")
            return False
        
        return True
    
    def create_base_signal(
        self, 
        signal_type: str, 
        epic: str, 
        timeframe: str,
        latest_row: pd.Series
    ) -> Dict:
        """
        Create base signal dictionary with common fields
        
        Args:
            signal_type: 'BULL' or 'BEAR'
            epic: Epic code
            timeframe: Timeframe
            latest_row: Latest row from DataFrame
            
        Returns:
            Base signal dictionary
        """
        return {
            'signal_type': signal_type,
            'epic': epic,
            'timeframe': timeframe,
            'timestamp': latest_row.get('start_time', pd.Timestamp.now()),
            'price': latest_row.get('close', 0),
            'strategy': self.name
        }
    
    def add_execution_prices(self, signal: Dict, spread_pips: float) -> Dict:
        """
        Add execution prices for BID-adjusted signals
        
        Args:
            signal: Signal dictionary
            spread_pips: Spread in pips
            
        Returns:
            Updated signal with execution prices
        """
        spread = spread_pips / 10000  # Convert to decimal
        current_price_mid = signal['price']
        
        if signal['signal_type'] == 'BULL':
            execution_price = current_price_mid + (spread / 2)  # ASK price for buying
        else:  # BEAR
            execution_price = current_price_mid - (spread / 2)  # BID price for selling
        
        signal.update({
            'price_mid': current_price_mid,
            'execution_price': execution_price,
            'spread_pips': spread_pips
        })

        return signal

    def calculate_optimal_sl_tp(
        self,
        signal: Dict,
        epic: str,
        latest_row: pd.Series,
        spread_pips: float
    ) -> Dict[str, int]:
        """
        Calculate optimized SL/TP in POINTS (not price levels) for order API.

        Uses optimal parameters from database if available, otherwise calculates
        based on ATR with conservative multipliers.

        Args:
            signal: The trading signal dictionary
            epic: Trading pair epic code
            latest_row: Latest candle data with indicators
            spread_pips: Current spread in pips

        Returns:
            {
                'stop_distance': int,  # Stop loss distance in points
                'limit_distance': int  # Take profit distance in points
            }
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get ATR for the pair
        atr = latest_row.get('atr', 0)
        if not atr or atr <= 0:
            # Fallback: estimate from current volatility (high-low range)
            atr = abs(latest_row.get('high', 0) - latest_row.get('low', 0))
            logger.warning(f"No ATR indicator, using high-low range: {atr}")

        # Convert ATR to pips/points
        if 'JPY' in epic:
            atr_pips = atr * 100  # JPY pairs: 0.01 = 1 pip
        else:
            atr_pips = atr * 10000  # Standard pairs: 0.0001 = 1 pip

        # Check for optimal parameters from database
        if self.optimal_params:
            stop_distance = int(self.optimal_params.stop_loss_pips)
            limit_distance = int(self.optimal_params.take_profit_pips)
            logger.info(f"{self.name}: Using optimal params from DB - SL={stop_distance}, TP={limit_distance}")
        else:
            # Calculate based on ATR with 2.0x multiplier (more conservative than dev-app's 1.5x)
            raw_stop = atr_pips * 2.0

            # Apply minimum safe distances
            if 'JPY' in epic:
                min_sl = 20  # Minimum 20 pips for JPY
            else:
                min_sl = 15  # Minimum 15 pips for others

            stop_distance = max(int(raw_stop), min_sl)

            # Take profit: 2.0-2.5× risk/reward based on pair volatility
            # Major pairs get higher RR targets
            if epic in ['CS.D.EURUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP']:
                risk_reward = 2.5
            else:
                risk_reward = 2.0

            limit_distance = int(stop_distance * risk_reward)

            logger.info(f"{self.name}: Calculated ATR-based SL/TP - ATR={atr_pips:.1f}, SL={stop_distance}, TP={limit_distance}, RR={risk_reward}")

        # Apply reasonable maximums to prevent excessive risk
        if 'JPY' in epic:
            max_sl = 55
        elif 'GBP' in epic:
            max_sl = 60  # GBP pairs are more volatile
        else:
            max_sl = 45

        if stop_distance > max_sl:
            logger.warning(f"Stop distance {stop_distance} exceeds max {max_sl}, capping to maximum")
            stop_distance = max_sl
            limit_distance = int(stop_distance * 2.0)  # Maintain reasonable RR

        return {
            'stop_distance': stop_distance,
            'limit_distance': limit_distance
        }

    # ✅ NEW: Validation helper methods
    def get_validation_status(self) -> Dict:
        """Get status of the enhanced validator"""
        return {
            'validator_available': self._validator is not None,
            'validator_type': 'enhanced' if self._validator else 'legacy',
            'strategy_name': self.name
        }