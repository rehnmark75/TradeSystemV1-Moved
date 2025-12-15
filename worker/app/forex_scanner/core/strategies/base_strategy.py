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
        epic = signal.get('epic', '')
        pip_value = 0.01 if epic and 'JPY' in epic.upper() else 0.0001
        spread = spread_pips * pip_value  # Convert pips to price units
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
        Calculate Stop Loss / Take Profit with reliable fallback chain

        Calculation Priority:
        1. Database optimal params (only if USE_OPTIMIZED_DATABASE_PARAMS = True)
        2. ATR-based dynamic calculation
        3. Static config fallback values

        Always enforces:
        - Pair-specific minimum stops (15-20 pips)
        - Maximum caps for risk management

        Returns:
            {
                'stop_distance': int,  # Stop loss in pips
                'limit_distance': int  # Take profit in pips
            }
        """
        import logging
        logger = logging.getLogger(__name__)

        # Import config
        try:
            from configdata import config
        except:
            from forex_scanner.configdata import config

        # Extract pair name from epic (CS.D.GBPUSD.MINI.IP → GBPUSD)
        pair = self._extract_pair_from_epic(epic)

        # Get pair-specific minimum stop
        min_stop = config.PAIR_MINIMUM_STOPS.get(pair, config.PAIR_MINIMUM_STOPS['DEFAULT'])

        logger.info(f"🎯 [{self.name}] Calculating SL/TP for {pair} (min_stop={min_stop} pips)")

        # =================================================================
        # STEP 1: Check if database parameters are enabled
        # =================================================================
        use_db_params = getattr(config, 'USE_OPTIMIZED_DATABASE_PARAMS', False)

        if use_db_params and self.optimal_params and hasattr(self.optimal_params, 'stop_loss_pips'):
            # Database params enabled - use them
            from datetime import datetime, timedelta

            param_age_days = (datetime.utcnow() - self.optimal_params.last_optimized).days if self.optimal_params.last_optimized else 999
            freshness_days = getattr(config, 'DB_PARAM_FRESHNESS_DAYS', 30)

            if param_age_days <= freshness_days:
                db_stop = int(self.optimal_params.stop_loss_pips)
                db_limit = int(self.optimal_params.take_profit_pips)

                # Enforce minimum even for DB params
                if db_stop < min_stop:
                    logger.warning(f"⚠️ DB stop {db_stop} < minimum {min_stop}, using minimum")
                    db_stop = min_stop
                    db_limit = int(db_stop * 2.0)

                logger.info(f"✅ Using DB params: SL={db_stop}, TP={db_limit} (age={param_age_days}d)")
                return self._apply_maximum_caps(epic, pair, db_stop, db_limit, logger)
            else:
                logger.info(f"⚠️ DB params too old ({param_age_days}d > {freshness_days}d), falling back to ATR")
        else:
            logger.info(f"📊 DB params DISABLED or unavailable, using ATR/static calculation")

        # =================================================================
        # STEP 2: ATR-Based Dynamic Calculation (Primary Method)
        # =================================================================
        atr = latest_row.get('atr', 0)

        if atr and atr > 0:
            # Get ATR multipliers from config
            stop_multiplier = getattr(config, 'DEFAULT_STOP_ATR_MULTIPLIER', 2.0)
            target_multiplier = getattr(config, 'DEFAULT_TARGET_ATR_MULTIPLIER', 4.0)

            # Convert ATR to pips
            if 'JPY' in pair:
                atr_pips = atr * 100  # JPY: 0.01 = 1 pip
            else:
                atr_pips = atr * 10000  # Others: 0.0001 = 1 pip

            # Calculate raw values
            raw_stop = atr_pips * stop_multiplier
            raw_limit = atr_pips * target_multiplier

            # Apply minimum
            stop_distance = max(int(raw_stop), min_stop)
            limit_distance = int(raw_limit)

            # Ensure TP is at least 1.5x SL (minimum risk/reward)
            min_rr = getattr(config, 'MINIMUM_RISK_REWARD_RATIO', 1.5)
            if limit_distance < stop_distance * min_rr:
                limit_distance = int(stop_distance * 2.0)

            logger.info(
                f"✅ ATR calculation: ATR={atr_pips:.1f}p, "
                f"raw_SL={raw_stop:.1f}p, final_SL={stop_distance}p, "
                f"TP={limit_distance}p (RR={limit_distance/stop_distance:.1f})"
            )

            return self._apply_maximum_caps(epic, pair, stop_distance, limit_distance, logger)

        # =================================================================
        # STEP 3: Static Config Fallback (ATR unavailable)
        # =================================================================
        logger.warning(f"⚠️ ATR not available for {pair}, using static config fallback")

        fallback = config.STATIC_FALLBACK_SL_TP.get(pair, config.STATIC_FALLBACK_SL_TP['DEFAULT'])
        stop_distance = max(fallback['sl'], min_stop)  # Enforce minimum
        limit_distance = fallback['tp']

        logger.info(f"📋 Static fallback: SL={stop_distance}p, TP={limit_distance}p")

        return self._apply_maximum_caps(epic, pair, stop_distance, limit_distance, logger)

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract pair name from epic code"""
        # CS.D.GBPUSD.MINI.IP → GBPUSD
        parts = epic.upper().split('.')
        for part in parts:
            # Check if part matches a currency pair pattern (6-7 chars, contains USD/EUR/GBP/etc)
            if len(part) in [6, 7] and any(curr in part for curr in ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']):
                return part

        # Fallback: check if any known pair is in epic
        try:
            from configdata import config
            for pair in config.PAIR_MINIMUM_STOPS.keys():
                if pair != 'DEFAULT' and pair in epic.upper():
                    return pair
        except:
            pass

        return 'DEFAULT'

    def _apply_maximum_caps(self, epic: str, pair: str, stop_distance: int, limit_distance: int, logger) -> Dict[str, int]:
        """Apply maximum stop loss caps for risk management"""
        try:
            from configdata import config

            # Determine maximum based on pair type
            if 'JPY' in pair:
                max_stop = config.PAIR_MAXIMUM_STOPS['JPY_PAIRS']
            elif 'GBP' in pair:
                max_stop = config.PAIR_MAXIMUM_STOPS['GBP_PAIRS']
            else:
                max_stop = config.PAIR_MAXIMUM_STOPS['DEFAULT']

            # Apply cap if exceeded
            if stop_distance > max_stop:
                logger.warning(f"⚠️ Stop {stop_distance}p exceeds max {max_stop}p, capping")
                stop_distance = max_stop
                limit_distance = int(stop_distance * 2.0)  # Maintain 2:1 RR

            logger.info(f"✅ Final SL/TP: {stop_distance}p / {limit_distance}p (max_allowed={max_stop}p)")

            return {
                'stop_distance': stop_distance,
                'limit_distance': limit_distance
            }
        except Exception as e:
            logger.error(f"Error applying caps: {e}, returning uncapped values")
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