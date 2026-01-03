# core/strategies/helpers/kama_forex_optimizer.py
"""
KAMA Forex Optimizer Module - Extracted from KAMA Strategy
🔥 FOREX SPECIFIC: All forex market optimizations and calculations for KAMA
🎯 FOCUSED: Single responsibility for KAMA forex-specific logic
📊 COMPREHENSIVE: Forex confidence adjustments, pair classification, market regime detection

This module contains all the forex-specific logic for KAMA strategy, 
extracted for better maintainability and testability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from datetime import datetime
try:
    from configdata import config
    from configdata.strategies import config_kama_strategy
except ImportError:
    from forex_scanner.configdata import config
    try:
        from forex_scanner.configdata.strategies import config_kama_strategy
    except ImportError:
        from forex_scanner.configdata.strategies import config_kama_strategy as config_kama_strategy


class KAMAForexOptimizer:
    """
    🔥 FOREX: Comprehensive forex market optimizations for KAMA strategy
    
    Responsibilities:
    - Forex pair classification and analysis
    - Currency-specific confidence adjustments
    - Market regime detection for KAMA
    - Efficiency ratio calculations optimized for forex
    - Volatility assessment and adjustments
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # KAMA Parameters from config (TIGHTENED - Phase 1 Optimization)
        self.er_period = getattr(config_kama_strategy, 'KAMA_ER_PERIOD', 14)
        self.fast_sc = getattr(config_kama_strategy, 'KAMA_FAST_SC', 2)
        self.slow_sc = getattr(config_kama_strategy, 'KAMA_SLOW_SC', 30)
        self.min_efficiency = getattr(config_kama_strategy, 'KAMA_MIN_EFFICIENCY', 0.20)  # Increased from 0.1
        self.trend_threshold = getattr(config_kama_strategy, 'KAMA_TREND_THRESHOLD', 0.05)
        self.min_bars = getattr(config_kama_strategy, 'KAMA_MIN_BARS', 50)
        self.base_confidence = getattr(config_kama_strategy, 'KAMA_BASE_CONFIDENCE', 0.75)
        
        # 🔥 Forex-specific KAMA thresholds (TIGHTENED - Phase 1 Optimization)
        # CHANGED: Increased min_efficiency thresholds for higher quality signals
        self.forex_kama_thresholds = {
            # === MAJOR USD PAIRS ===
            'CS.D.EURUSD.CEEM.IP': {
                'min_efficiency': 0.20,  # Increased from 0.12 - EUR pairs trend well
                'trend_threshold': 0.05,  # Slightly increased from 0.04
                'volatility_multiplier': 1.0,
                'confidence_bonus': 0.05
            },
            'CS.D.GBPUSD.MINI.IP': {
                'min_efficiency': 0.25,  # Increased from 0.15 - GBP more volatile, needs higher bar
                'trend_threshold': 0.06,
                'volatility_multiplier': 1.2,
                'confidence_bonus': 0.03
            },
            'CS.D.AUDUSD.MINI.IP': {
                'min_efficiency': 0.22,  # Increased from 0.13 - Commodity currency
                'trend_threshold': 0.05,
                'volatility_multiplier': 1.1,
                'confidence_bonus': 0.02
            },
            'CS.D.NZDUSD.MINI.IP': {
                'min_efficiency': 0.22,  # Increased from 0.14 - Similar to AUD
                'trend_threshold': 0.055,
                'volatility_multiplier': 1.15,
                'confidence_bonus': 0.02
            },
            'CS.D.USDCAD.MINI.IP': {
                'min_efficiency': 0.22,  # Increased from 0.11 - Commodity pair
                'trend_threshold': 0.05,
                'volatility_multiplier': 0.9,
                'confidence_bonus': 0.03
            },
            'CS.D.USDCHF.MINI.IP': {
                'min_efficiency': 0.18,  # Increased from 0.10 - Safe haven, stable trends
                'trend_threshold': 0.04,
                'volatility_multiplier': 0.8,
                'confidence_bonus': 0.04
            },

            # === JPY PAIRS (Different pip structure) ===
            'CS.D.USDJPY.MINI.IP': {
                'min_efficiency': 0.18,  # Increased from 0.10 - JPY pairs stable but need quality
                'trend_threshold': 0.08,  # Different pip structure (100 vs 10000)
                'volatility_multiplier': 0.8,
                'confidence_bonus': 0.04
            },
            'CS.D.EURJPY.MINI.IP': {
                'min_efficiency': 0.20,  # Increased from 0.12 - EUR volatility + JPY stability
                'trend_threshold': 0.09,
                'volatility_multiplier': 0.95,
                'confidence_bonus': 0.03
            },
            'CS.D.AUDJPY.MINI.IP': {
                'min_efficiency': 0.22,  # Increased from 0.13 - Commodity + JPY cross
                'trend_threshold': 0.10,
                'volatility_multiplier': 1.0,
                'confidence_bonus': 0.025
            }
        }
        
        # Cache for performance optimization
        self._forex_cache = {}
        self._cache_timeout = 300  # 5 minutes
        
        self.logger.info("🔥 KAMA Forex Optimizer initialized with 9 supported pairs")

    def get_forex_pair_type(self, epic: str) -> str:
        """
        🌍 FOREX: Classify forex pair type for KAMA optimization
        """
        try:
            if 'EUR' in epic and 'JPY' not in epic:
                return 'EUR_major'
            elif 'GBP' in epic and 'JPY' not in epic:
                return 'GBP_volatile'
            elif 'JPY' in epic:
                if 'USD' in epic:
                    return 'USDJPY_stable'
                elif 'EUR' in epic:
                    return 'EURJPY_cross'
                elif 'AUD' in epic:
                    return 'AUDJPY_commodity'
                elif 'GBP' in epic:
                    return 'GBPJPY_volatile'
                else:
                    return 'JPY_cross'
            elif 'AUD' in epic:
                return 'AUD_commodity'
            elif 'NZD' in epic:
                return 'NZD_commodity'
            elif 'CAD' in epic:
                return 'CAD_commodity'
            elif 'CHF' in epic:
                return 'CHF_safe_haven'
            else:
                return 'exotic_pair'
        except Exception as e:
            self.logger.debug(f"Error classifying forex pair {epic}: {e}")
            return 'unknown_pair'

    def get_kama_thresholds_for_pair(self, epic: str) -> Dict:
        """
        🎯 FOREX: Get KAMA-specific thresholds optimized for each forex pair
        """
        return self.forex_kama_thresholds.get(epic, {
            'min_efficiency': self.min_efficiency,
            'trend_threshold': self.trend_threshold,
            'volatility_multiplier': 1.0,
            'confidence_bonus': 0.0
        })

    def apply_forex_confidence_adjustments(
        self, 
        base_confidence: float, 
        epic: str, 
        signal_data: Dict
    ) -> float:
        """
        🔥 KAMA-FOREX: Apply KAMA-specific forex confidence adjustments
        
        This method is specifically designed for KAMA signals in forex markets,
        taking into account KAMA's unique characteristics and how they interact
        with different forex pairs.
        """
        try:
            adjusted_confidence = base_confidence
            pair_type = self.get_forex_pair_type(epic)
            thresholds = self.get_kama_thresholds_for_pair(epic)
            
            # Get KAMA-specific data
            efficiency_ratio = signal_data.get('efficiency_ratio', 0.1)
            kama_trend = signal_data.get('kama_trend', 0)
            kama_distance = signal_data.get('kama_distance', 0)
            signal_strength = signal_data.get('signal_strength', 0)
            signal_type = signal_data.get('signal_type', '')
            
            self.logger.debug(f"[KAMA FOREX] Adjusting {pair_type} confidence: base={base_confidence:.1%}, ER={efficiency_ratio:.3f}")
            
            # 🎯 KAMA-EFFICIENCY FOREX ADJUSTMENTS
            # Different forex pairs respond differently to KAMA efficiency
            
            if pair_type == 'EUR_major':
                # EUR pairs: KAMA works well due to good liquidity and trends
                if efficiency_ratio > 0.4:
                    adjusted_confidence += 0.04  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.25:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.20:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                else:
                    adjusted_confidence -= 0.01  # REDUCED penalty
                    
            elif pair_type == 'GBP_volatile':
                # GBP pairs: More volatile, KAMA needs higher efficiency to be reliable
                if efficiency_ratio > 0.5:
                    adjusted_confidence += 0.05  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.35:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.25:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                elif efficiency_ratio < 0.25:
                    adjusted_confidence -= 0.02  # REDUCED penalty
                    
            elif pair_type in ['USDJPY_stable', 'JPY_cross']:
                # USD/JPY and other JPY pairs: More stable, KAMA can work with lower efficiency
                if efficiency_ratio > 0.3:
                    adjusted_confidence += 0.04  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.2:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.18:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                # Less penalty for low efficiency due to JPY stability
                
            elif pair_type == 'EURJPY_cross':
                # EUR/JPY: Combination of EUR volatility and JPY stability
                if efficiency_ratio > 0.4:
                    adjusted_confidence += 0.04  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.25:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.20:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                elif efficiency_ratio < 0.20:
                    adjusted_confidence -= 0.01  # REDUCED penalty
                    
            elif pair_type in ['AUDJPY_commodity', 'GBPJPY_volatile']:
                # AUD/JPY, GBP/JPY: More volatile JPY crosses
                if efficiency_ratio > 0.45:
                    adjusted_confidence += 0.05  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.3:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.22:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                elif efficiency_ratio < 0.22:
                    adjusted_confidence -= 0.02  # REDUCED penalty
                    
            elif pair_type in ['AUD_commodity', 'NZD_commodity', 'CAD_commodity']:
                # Commodity currencies: Trend-following, good for KAMA
                if efficiency_ratio > 0.35:
                    adjusted_confidence += 0.04  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.25:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.22:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                elif efficiency_ratio < 0.20:
                    adjusted_confidence -= 0.01  # REDUCED penalty
                    
            elif pair_type == 'CHF_safe_haven':
                # CHF pairs: Flight-to-safety affects KAMA reliability
                if efficiency_ratio > 0.4:
                    adjusted_confidence += 0.03  # REDUCED - multiplier already applied
                elif efficiency_ratio > 0.25:
                    adjusted_confidence += 0.02  # REDUCED - multiplier already applied
                elif efficiency_ratio >= 0.18:
                    adjusted_confidence += 0.01  # REDUCED - multiplier already applied
                else:
                    adjusted_confidence -= 0.01  # REDUCED penalty
                    
            else:  # exotic_pair or unknown
                # Exotic pairs: More unpredictable, require higher standards
                if efficiency_ratio > 0.5:
                    adjusted_confidence += 0.03  # High bar for exotics
                else:
                    adjusted_confidence -= 0.06  # Penalty for low efficiency exotics
            
            # 📈 KAMA TREND STRENGTH FOREX ADJUSTMENTS
            trend_strength = abs(kama_trend)
            
            if pair_type == 'GBP_volatile':
                # GBP needs stronger trends to be reliable
                if trend_strength > thresholds['trend_threshold'] * 2:
                    adjusted_confidence += 0.06
                elif trend_strength < thresholds['trend_threshold'] * 0.5:
                    adjusted_confidence -= 0.08
                    
            elif pair_type == 'JPY_stable':
                # JPY can work with weaker trends due to stability
                if trend_strength > thresholds['trend_threshold'] * 1.5:
                    adjusted_confidence += 0.05
                elif trend_strength > thresholds['trend_threshold'] * 0.5:
                    adjusted_confidence += 0.02
                    
            else:
                # Standard trend strength adjustments
                if trend_strength > thresholds['trend_threshold'] * 2:
                    adjusted_confidence += 0.05
                elif trend_strength < thresholds['trend_threshold'] * 0.5:
                    adjusted_confidence -= 0.05
            
            # 📏 PRICE-KAMA DISTANCE FOREX ADJUSTMENTS
            # Different pairs have different acceptable distances
            
            if pair_type == 'JPY_stable':
                # JPY pairs can handle larger distances due to different pip structure
                max_distance = 0.008  # 0.8%
            elif pair_type == 'GBP_volatile':
                # GBP pairs need tighter distance due to volatility
                max_distance = 0.004  # 0.4%
            else:
                # Standard distance tolerance
                max_distance = 0.006  # 0.6%
            
            if kama_distance > max_distance:
                adjusted_confidence -= 0.06  # Too far from KAMA
            elif kama_distance > max_distance * 0.6:
                adjusted_confidence -= 0.03  # Somewhat far from KAMA
            else:
                adjusted_confidence += 0.02  # Good alignment with KAMA
            
            # 🌍 TRADING SESSION FOREX ADJUSTMENTS
            trading_session = self._determine_trading_session()
            
            if trading_session == 'london':
                # London session: High volume, good for all pairs
                if pair_type in ['EUR_major', 'GBP_volatile']:
                    adjusted_confidence += 0.04  # EUR/GBP very active in London
                else:
                    adjusted_confidence += 0.02
                    
            elif trading_session == 'new_york':
                # NY session: Good for USD pairs
                if 'USD' in epic:
                    adjusted_confidence += 0.03
                else:
                    adjusted_confidence += 0.01
                    
            elif trading_session == 'tokyo':
                # Tokyo session: Best for JPY pairs
                if pair_type == 'JPY_stable':
                    adjusted_confidence += 0.04
                else:
                    adjusted_confidence -= 0.02  # Lower volume for non-JPY
                    
            elif trading_session == 'sydney':
                # Sydney session: Lower volume, less reliable
                adjusted_confidence -= 0.03
            
            # 🎯 KAMA SIGNAL TYPE FOREX ADJUSTMENTS
            # Some forex pairs trend better in certain directions
            
            if signal_type in ['BULL', 'BUY']:
                if pair_type == 'commodity_currency' and efficiency_ratio > 0.3:
                    adjusted_confidence += 0.03  # Commodity currencies often in uptrends
                elif pair_type == 'safe_haven' and efficiency_ratio < 0.3:
                    adjusted_confidence -= 0.04  # Safe havens resist uptrends in risk-off
                    
            elif signal_type in ['BEAR', 'SELL']:
                if pair_type == 'safe_haven' and efficiency_ratio > 0.3:
                    adjusted_confidence += 0.03  # Safe havens strengthen in risk-off
                elif pair_type == 'commodity_currency' and efficiency_ratio < 0.3:
                    adjusted_confidence -= 0.03  # Commodity currencies resist downtrends in risk-on
            
            # 🔧 PAIR-SPECIFIC CONFIDENCE BONUS
            adjusted_confidence += thresholds.get('confidence_bonus', 0.0)
            
            # 🔒 FINAL BOUNDS: Apply pair-specific volatility multiplier and bounds
            volatility_multiplier = thresholds.get('volatility_multiplier', 1.0)
            if volatility_multiplier != 1.0:
                # Adjust confidence based on pair volatility characteristics
                confidence_adjustment = (volatility_multiplier - 1.0) * 0.05
                adjusted_confidence += confidence_adjustment
            
            # Ensure confidence stays within reasonable bounds
            final_confidence = max(0.05, min(0.95, adjusted_confidence))
            
            self.logger.debug(f"[KAMA FOREX] {epic} ({pair_type}): {base_confidence:.1%} → {final_confidence:.1%} "
                           f"(ER: {efficiency_ratio:.3f}, Trend: {trend_strength:.4f}, Session: {trading_session})")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"KAMA forex confidence adjustment error: {e}")
            return base_confidence

    def calculate_forex_efficiency_ratio(self, df: pd.DataFrame, epic: str) -> float:
        """
        📊 FOREX: Calculate efficiency ratio optimized for forex markets
        """
        try:
            # Use KAMA ER period for consistency
            period = min(self.er_period, len(df) - 1)
            if period < 3:
                return 0.25  # Safe default above rejection threshold
            
            # Get price series
            close_prices = df['close'].tail(period + 1)
            
            if len(close_prices) < 2:
                return 0.25
            
            # Calculate directional change (net movement)
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            direction_change = abs(end_price - start_price)
            
            # Calculate total movement (sum of all price changes)
            price_changes = close_prices.diff().dropna()
            total_movement = price_changes.abs().sum()
            
            # Handle edge cases
            if total_movement == 0 or pd.isna(total_movement) or total_movement < 1e-8:
                return 0.25
            
            # For forex, apply minimum movement threshold
            forex_thresholds = self.get_kama_thresholds_for_pair(epic)
            min_movement = close_prices.mean() * 0.0001  # 0.01% of price
            
            if total_movement < min_movement:
                return 0.30  # Above rejection threshold
            
            # Calculate efficiency with forex-specific scaling
            efficiency = direction_change / total_movement
            
            # Apply forex pair-specific adjustments
            pair_type = self.get_forex_pair_type(epic)
            if pair_type == 'JPY_stable':
                # JPY pairs often have different price scales
                efficiency = min(efficiency * 1.2, 1.0)
            elif pair_type == 'GBP_volatile':
                # GBP pairs need slight dampening
                efficiency = efficiency * 0.9
            
            # Ensure minimum efficiency above rejection threshold
            final_efficiency = max(0.21, min(1.0, efficiency))
            
            return final_efficiency
            
        except Exception as e:
            self.logger.error(f"❌ Forex efficiency ratio calculation failed: {e}")
            return 0.25

    def detect_forex_market_regime(self, current: pd.Series, df: pd.DataFrame, epic: str, timeframe: str) -> str:
        """
        🌡️ FOREX: Detect market regime optimized for KAMA in forex markets
        """
        try:
            kama_value = current.get('kama', 0)
            current_price = current.get('close', 0)
            efficiency_ratio = current.get('efficiency_ratio', 0.1)
            kama_slope = current.get('kama_slope', current.get('kama_trend', 0))
            
            if current_price <= 0 or kama_value <= 0:
                return 'ranging'
            
            # Forex-specific regime assessment
            price_kama_distance = abs(current_price - kama_value) / current_price
            pair_type = self.get_forex_pair_type(epic)
            
            # Get forex-specific thresholds
            thresholds = self.get_kama_thresholds_for_pair(epic)
            
            # Adjust thresholds based on forex pair type
            if pair_type == 'GBP_volatile':
                volatility_threshold = 0.003
                trending_threshold = 0.001
            elif pair_type == 'JPY_stable':
                volatility_threshold = 0.002
                trending_threshold = 0.0008
            else:
                volatility_threshold = 0.0025
                trending_threshold = 0.001
            
            # Combined regime assessment for forex KAMA
            if efficiency_ratio > 0.6 or abs(kama_slope) > volatility_threshold or price_kama_distance > volatility_threshold:
                return 'volatile'
            elif efficiency_ratio > 0.3 or abs(kama_slope) > trending_threshold:
                return 'trending'
            elif efficiency_ratio < 0.15 and abs(kama_slope) < trending_threshold * 0.5:
                return 'consolidating'
            else:
                return 'ranging'
            
        except Exception as e:
            self.logger.debug(f"Forex market regime detection failed: {e}")
            return 'ranging'

    def calculate_kama_pip_distance(self, current_price: float, kama_value: float, epic: str) -> float:
        """
        📏 FOREX: Calculate pip distance between price and KAMA for forex pairs
        """
        try:
            # Determine pip size based on forex pair
            if 'JPY' in epic:
                pip_size = 0.01  # JPY pairs
            else:
                pip_size = 0.0001  # Most forex pairs
            
            pip_distance = abs(current_price - kama_value) / pip_size
            return pip_distance
            
        except Exception as e:
            self.logger.debug(f"Pip distance calculation error: {e}")
            return 0.0

    def _determine_trading_session(self) -> str:
        """Determine current forex trading session"""
        try:
            import pytz
            london_tz = pytz.timezone('Europe/London')
            london_time = datetime.now(london_tz)
            hour = london_time.hour
            
            if 8 <= hour < 17:
                return 'london'
            elif 13 <= hour < 22:
                return 'new_york'
            elif 0 <= hour < 9:
                return 'sydney'
            else:
                return 'tokyo'
        except:
            return 'unknown'

    def get_performance_stats(self) -> Dict:
        """📊 Get forex optimizer performance statistics"""
        try:
            return {
                'module': 'kama_forex_optimizer',
                'parameters': {
                    'er_period': self.er_period,
                    'fast_sc': self.fast_sc,
                    'slow_sc': self.slow_sc,
                    'min_efficiency': self.min_efficiency,
                    'trend_threshold': self.trend_threshold
                },
                'forex_pairs_supported': len(self.forex_kama_thresholds),
                'cache_entries': len(self._forex_cache),
                'cache_timeout': self._cache_timeout,
                'error': None
            }
        except Exception as e:
            return {'error': str(e)}

    def clear_cache(self):
        """🧹 Clear forex optimizer cache"""
        self._forex_cache.clear()
        self.logger.debug("🧹 KAMA Forex Optimizer cache cleared")

    def debug_forex_analysis(self, df: pd.DataFrame, epic: str, timeframe: str) -> Dict:
        """
        🔍 DEBUG: Comprehensive forex analysis debugging for KAMA
        """
        try:
            debug_info = {
                'epic': epic,
                'pair_type': self.get_forex_pair_type(epic),
                'kama_thresholds': self.get_kama_thresholds_for_pair(epic),
                'trading_session': self._determine_trading_session(),
                'data_length': len(df)
            }
            
            if len(df) > 0:
                current = df.iloc[-1]
                debug_info.update({
                    'current_price': current.get('close', 0),
                    'kama_value': current.get('kama', 0),
                    'efficiency_ratio': current.get('efficiency_ratio', 0),
                    'market_regime': self.detect_forex_market_regime(current, df, epic, timeframe)
                })
                
                if current.get('close', 0) > 0 and current.get('kama', 0) > 0:
                    debug_info['pip_distance'] = self.calculate_kama_pip_distance(
                        current['close'], current['kama'], epic
                    )
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}