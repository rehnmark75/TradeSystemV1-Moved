#!/usr/bin/env python3
"""
Optimal Parameter Service
Dynamic parameter retrieval system that uses optimization results instead of static config
"""

import sys
import os
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.database import DatabaseManager
try:
    import config
except ImportError:
    from forex_scanner import config


@dataclass
class MarketConditions:
    """Market condition context for parameter selection"""
    volatility_level: str = 'medium'  # low, medium, high
    market_regime: str = 'trending'   # trending, ranging, breakout
    session: str = 'london'           # asian, london, new_york, overlap
    news_impact: str = 'normal'       # low, normal, high
    

@dataclass 
class OptimalParameters:
    """Optimal trading parameters for an epic"""
    epic: str
    ema_config: str
    confidence_threshold: float
    timeframe: str
    smart_money_enabled: bool
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    performance_score: float
    last_optimized: datetime
    market_conditions: Optional[MarketConditions] = None


@dataclass
class ZeroLagOptimalParameters:
    """Optimal Zero-Lag trading parameters for an epic"""
    epic: str
    zl_length: int
    band_multiplier: float
    confidence_threshold: float
    timeframe: str
    bb_length: int
    bb_mult: float
    kc_length: int
    kc_mult: float
    smart_money_enabled: bool
    mtf_validation_enabled: bool
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    performance_score: float
    last_optimized: datetime
    market_conditions: Optional[MarketConditions] = None


@dataclass
class MACDOptimalParameters:
    """Optimal MACD trading parameters for an epic"""
    epic: str
    fast_ema: int
    slow_ema: int
    signal_ema: int
    confidence_threshold: float
    timeframe: str
    histogram_threshold: float
    zero_line_filter: bool
    rsi_filter_enabled: bool
    momentum_confirmation: bool
    mtf_enabled: bool
    mtf_timeframes: Optional[str]
    smart_money_enabled: bool
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    performance_score: float
    win_rate: float
    last_optimized: datetime
    market_conditions: Optional[MarketConditions] = None


@dataclass
class SMCOptimalParameters:
    """Optimal Smart Money Concepts trading parameters for an epic"""
    epic: str
    smc_config: str
    confidence_threshold: float
    timeframe: str
    use_smart_money: bool
    
    # Market Structure Parameters
    swing_length: int
    structure_confirmation: int
    bos_threshold: float
    choch_threshold: float
    
    # Order Block Parameters
    order_block_length: int
    order_block_volume_factor: float
    order_block_buffer: float
    max_order_blocks: int
    
    # Fair Value Gap Parameters
    fvg_min_size: float
    fvg_max_age: int
    fvg_fill_threshold: float
    
    # Supply/Demand Zone Parameters
    zone_min_touches: int
    zone_max_age: int
    zone_strength_factor: float
    
    # Signal Generation Parameters
    confluence_required: float
    min_risk_reward: float
    max_distance_to_zone: float
    min_signal_confidence: float
    
    # Multi-timeframe Parameters
    use_higher_tf: bool
    higher_tf_multiplier: int
    mtf_confluence_weight: float
    
    # Risk Management
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    
    # Performance Metrics
    performance_score: float
    win_rate: float
    confluence_accuracy: float
    structure_break_accuracy: float
    order_block_success_rate: float
    fvg_success_rate: float
    
    last_optimized: datetime
    market_conditions: Optional[MarketConditions] = None


@dataclass
class IchimokuOptimalParameters:
    """Optimal Ichimoku Cloud trading parameters for an epic"""
    epic: str
    tenkan_period: int
    kijun_period: int
    senkou_b_period: int
    chikou_shift: int
    cloud_shift: int
    confidence_threshold: float
    timeframe: str

    # Validation thresholds
    cloud_thickness_threshold: float
    tk_cross_strength_threshold: float
    chikou_clear_threshold: float
    cloud_filter_enabled: bool
    chikou_filter_enabled: bool
    tk_filter_enabled: bool

    # Multi-timeframe settings
    mtf_enabled: bool
    mtf_timeframes: Optional[str]
    mtf_min_alignment: float
    mtf_cloud_weight: float
    mtf_tk_weight: float
    mtf_chikou_weight: float

    # Enhancement options
    momentum_confluence_enabled: bool
    smart_money_enabled: bool
    ema_200_trend_filter: bool
    contradiction_filter_enabled: bool

    # Risk management
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float

    # Performance metrics
    performance_score: float
    win_rate: float
    tk_cross_accuracy: float
    cloud_breakout_accuracy: float
    chikou_confirmation_rate: float
    perfect_alignment_rate: float
    mtf_alignment_avg: float

    last_optimized: datetime
    market_conditions: Optional[MarketConditions] = None


@dataclass
class MeanReversionOptimalParameters:
    """Optimal Mean Reversion trading parameters for an epic"""
    epic: str
    timeframe: str
    confidence_threshold: float

    # LuxAlgo Premium Oscillator Parameters
    luxalgo_length: int
    luxalgo_smoothing: int
    luxalgo_overbought_threshold: float
    luxalgo_oversold_threshold: float
    luxalgo_extreme_ob_threshold: float
    luxalgo_extreme_os_threshold: float

    # Multi-timeframe RSI Parameters
    mtf_rsi_period: int
    mtf_rsi_timeframes: Optional[str]
    mtf_min_alignment: float
    mtf_rsi_overbought: float
    mtf_rsi_oversold: float

    # RSI-EMA Divergence Parameters
    rsi_ema_period: int
    rsi_ema_rsi_period: int
    rsi_ema_divergence_sensitivity: float
    rsi_ema_min_divergence_strength: float

    # Squeeze Momentum Parameters
    squeeze_bb_length: int
    squeeze_bb_mult: float
    squeeze_kc_length: int
    squeeze_kc_mult: float
    squeeze_momentum_length: int
    squeeze_require_release: bool
    squeeze_momentum_threshold: float

    # Oscillator Confluence Parameters
    bull_confluence_threshold: float
    bear_confluence_threshold: float
    luxalgo_weight: float
    mtf_rsi_weight: float
    divergence_weight: float
    squeeze_weight: float

    # Mean Reversion Zone Parameters
    zone_validation_enabled: bool
    zone_lookback_periods: int
    zone_multiplier: float
    require_zone_touch: bool
    min_zone_distance: float
    max_zone_age: int
    zone_confidence_boost: float

    # Market Regime Parameters
    market_regime_enabled: bool
    disable_in_strong_trend: bool
    trend_strength_threshold: float
    volatility_period: int
    trend_period: int
    ranging_threshold: float

    # Multi-timeframe Analysis Parameters
    mtf_analysis_enabled: bool
    mtf_timeframes_list: Optional[str]
    mtf_min_alignment_score: float
    require_higher_tf_confluence: bool
    mtf_confidence_boost_full_alignment: float

    # Signal Quality Parameters
    min_confidence: float
    min_risk_reward: float
    max_signals_per_day: int
    min_signal_spacing_hours: int

    # Risk Management Parameters
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    position_size_multiplier: float
    max_drawdown_threshold: float
    trail_stop_enabled: bool

    # Performance Metrics
    performance_score: float
    win_rate: float
    avg_profit_pips: float
    max_consecutive_losses: int
    profit_factor: float
    sharpe_ratio: float
    confluence_accuracy: float
    divergence_success_rate: float
    zone_touch_accuracy: float
    regime_filter_effectiveness: float

    last_optimized: datetime
    market_conditions: Optional[MarketConditions] = None


class OptimalParameterService:
    """
    Service for retrieving optimal trading parameters from optimization results
    """
    
    def __init__(self, cache_duration_minutes: int = 30):
        self.logger = logging.getLogger('optimal_parameter_service')
        self.db_manager = DatabaseManager(config.DATABASE_URL)
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._parameter_cache = {}
        self._cache_timestamps = {}
        
    def get_epic_parameters(self, 
                          epic: str, 
                          market_conditions: Optional[MarketConditions] = None,
                          force_refresh: bool = False) -> OptimalParameters:
        """
        Get optimal parameters for specific epic
        
        Args:
            epic: Trading pair epic (e.g. 'CS.D.EURUSD.CEEM.IP')
            market_conditions: Current market conditions for context-aware selection
            force_refresh: Force refresh from database even if cached
            
        Returns:
            OptimalParameters object with all trading settings
        """
        cache_key = f"{epic}_{hash(str(market_conditions)) if market_conditions else 'default'}"
        
        # Check cache first (unless force refresh)
        if not force_refresh and self._is_cache_valid(cache_key):
            self.logger.debug(f"📋 Using cached parameters for {epic}")
            return self._parameter_cache[cache_key]
        
        # Get from database
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Primary query: Get best parameters for this epic
                cursor.execute("""
                    SELECT 
                        epic, best_ema_config, best_confidence_threshold, best_timeframe,
                        optimal_stop_loss_pips, optimal_take_profit_pips,
                        ROUND(optimal_take_profit_pips / optimal_stop_loss_pips, 2) as risk_reward,
                        best_win_rate, best_profit_factor, best_net_pips,
                        (best_win_rate * best_profit_factor * (best_net_pips / 100.0)) as composite_score,
                        last_updated
                    FROM ema_best_parameters 
                    WHERE epic = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, (epic,))
                
                result = cursor.fetchone()
                
                if result:
                    # Create optimal parameters from database result
                    optimal_params = OptimalParameters(
                        epic=result[0],
                        ema_config=result[1],
                        confidence_threshold=result[2],
                        timeframe=result[3],
                        smart_money_enabled=self._should_enable_smart_money(epic, result[1]),
                        stop_loss_pips=float(result[4]),
                        take_profit_pips=float(result[5]),
                        risk_reward_ratio=float(result[6]),
                        performance_score=float(result[10]),
                        last_optimized=result[11],
                        market_conditions=market_conditions
                    )
                    
                    self.logger.info(f"✅ Retrieved optimal parameters for {epic}: "
                                   f"{result[1]} config, {result[2]:.0%} confidence, "
                                   f"{result[4]:.0f}/{result[5]:.0f} SL/TP")
                    
                else:
                    # Fallback to default parameters
                    self.logger.warning(f"⚠️ No optimization data found for {epic}, using fallbacks")
                    optimal_params = self._get_fallback_parameters(epic, market_conditions)
                
                # Cache the result
                self._parameter_cache[cache_key] = optimal_params
                self._cache_timestamps[cache_key] = datetime.now()
                
                return optimal_params
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get parameters for {epic}: {e}")
            return self._get_fallback_parameters(epic, market_conditions)
    
    def get_all_epic_parameters(self, 
                               market_conditions: Optional[MarketConditions] = None) -> Dict[str, OptimalParameters]:
        """Get optimal parameters for all epics with optimization data"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT epic FROM ema_best_parameters 
                    ORDER BY best_net_pips DESC
                """)
                
                epics = [row[0] for row in cursor.fetchall()]
                
                # Get parameters for each epic
                all_parameters = {}
                for epic in epics:
                    all_parameters[epic] = self.get_epic_parameters(epic, market_conditions)
                
                self.logger.info(f"✅ Retrieved parameters for {len(all_parameters)} optimized epics")
                return all_parameters
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get all epic parameters: {e}")
            return {}
    
    def get_parameter_performance_history(self, epic: str, days: int = 30) -> pd.DataFrame:
        """Get parameter performance history for analysis"""
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT 
                        run_id, epic, ema_config, confidence_threshold, timeframe,
                        smart_money_enabled, stop_loss_pips, take_profit_pips,
                        risk_reward_ratio, total_signals, win_rate, profit_factor,
                        net_pips, composite_score, created_at
                    FROM ema_optimization_results
                    WHERE epic = %s 
                        AND created_at >= NOW() - INTERVAL '%s days'
                        AND total_signals >= 5
                    ORDER BY composite_score DESC NULLS LAST
                    LIMIT 100
                """
                
                df = pd.read_sql_query(query, conn, params=[epic, days])
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance history for {epic}: {e}")
            return pd.DataFrame()
    
    def suggest_parameter_updates(self, epic: str) -> Dict[str, any]:
        """Analyze if parameters need updating based on recent performance"""
        try:
            history_df = self.get_parameter_performance_history(epic, days=7)
            current_params = self.get_epic_parameters(epic)
            
            if history_df.empty:
                return {'needs_update': False, 'reason': 'No recent optimization data'}
            
            # Compare current vs recent best
            recent_best = history_df.iloc[0]
            performance_gap = recent_best['composite_score'] - current_params.performance_score
            
            suggestions = {
                'needs_update': performance_gap > 0.1,
                'performance_improvement': performance_gap,
                'suggested_config': recent_best['ema_config'],
                'suggested_confidence': recent_best['confidence_threshold'],
                'suggested_sl_tp': f"{recent_best['stop_loss_pips']:.0f}/{recent_best['take_profit_pips']:.0f}",
                'reason': f"Recent optimization shows {performance_gap:.3f} improvement potential"
            }
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate suggestions for {epic}: {e}")
            return {'needs_update': False, 'error': str(e)}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached parameters are still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[cache_key] 
        return cache_age < self.cache_duration
    
    def _should_enable_smart_money(self, epic: str, ema_config: str) -> bool:
        """Determine if smart money should be enabled based on epic and config"""
        # Smart money generally works better with longer timeframes and conservative configs
        smart_money_configs = ['conservative', 'swing', 'news_safe']
        return ema_config in smart_money_configs
    
    def _get_fallback_parameters(self, epic: str, market_conditions: Optional[MarketConditions] = None) -> OptimalParameters:
        """Get fallback parameters when no optimization data exists"""
        
        # Epic-specific fallbacks based on currency pair characteristics
        if 'JPY' in epic:
            # JPY pairs typically need wider stops due to higher volatility
            fallback_sl, fallback_tp = 15.0, 30.0
        elif 'GBP' in epic:
            # GBP pairs are volatile, need wider stops
            fallback_sl, fallback_tp = 12.0, 25.0
        elif 'EUR' in epic or 'USD' in epic:
            # Major pairs, standard parameters
            fallback_sl, fallback_tp = 10.0, 20.0
        else:
            # Other pairs, conservative approach
            fallback_sl, fallback_tp = 15.0, 30.0
        
        # Market condition adjustments
        if market_conditions:
            if market_conditions.volatility_level == 'high':
                fallback_sl *= 1.2
                fallback_tp *= 1.2
            elif market_conditions.volatility_level == 'low':
                fallback_sl *= 0.8
                fallback_tp *= 0.8
        
        return OptimalParameters(
            epic=epic,
            ema_config='default',
            confidence_threshold=0.55,
            timeframe='15m',
            smart_money_enabled=False,
            stop_loss_pips=fallback_sl,
            take_profit_pips=fallback_tp,
            risk_reward_ratio=fallback_tp / fallback_sl,
            performance_score=0.0,  # No optimization data
            last_optimized=datetime.now() - timedelta(days=999),  # Very old
            market_conditions=market_conditions
        )
    
    def clear_cache(self):
        """Clear parameter cache to force fresh database lookups"""
        self._parameter_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("🗑️ Parameter cache cleared")
    
    # ==================== ZERO-LAG STRATEGY METHODS ====================
    
    def get_zerolag_parameters(self, 
                              epic: str, 
                              market_conditions: Optional[MarketConditions] = None,
                              force_refresh: bool = False) -> ZeroLagOptimalParameters:
        """
        Get optimal zero-lag parameters for specific epic
        
        Args:
            epic: Trading pair epic (e.g. 'CS.D.EURUSD.CEEM.IP')
            market_conditions: Current market conditions for context-aware selection
            force_refresh: Force refresh from database even if cached
            
        Returns:
            ZeroLagOptimalParameters object with all trading settings
        """
        cache_key = f"zerolag_{epic}_{hash(str(market_conditions)) if market_conditions else 'default'}"
        
        # Check cache first (unless force refresh)
        if not force_refresh and self._is_cache_valid(cache_key):
            self.logger.debug(f"📋 Using cached zero-lag parameters for {epic}")
            return self._parameter_cache[cache_key]
        
        # Get from database
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Primary query: Get best zero-lag parameters for this epic
                cursor.execute("""
                    SELECT 
                        epic, best_zl_length, best_band_multiplier, best_confidence_threshold, best_timeframe,
                        best_bb_length, best_bb_mult, best_kc_length, best_kc_mult,
                        best_smart_money_enabled, best_mtf_validation_enabled,
                        optimal_stop_loss_pips, optimal_take_profit_pips,
                        ROUND(optimal_take_profit_pips / optimal_stop_loss_pips, 2) as risk_reward,
                        best_win_rate, best_profit_factor, best_net_pips, best_composite_score,
                        last_updated
                    FROM zerolag_best_parameters 
                    WHERE epic = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, (epic,))
                
                result = cursor.fetchone()
                
                if result:
                    # Create optimal parameters from database result
                    optimal_params = ZeroLagOptimalParameters(
                        epic=result[0],
                        zl_length=int(result[1]),
                        band_multiplier=float(result[2]),
                        confidence_threshold=float(result[3]),
                        timeframe=result[4],
                        bb_length=int(result[5]),
                        bb_mult=float(result[6]),
                        kc_length=int(result[7]),
                        kc_mult=float(result[8]),
                        smart_money_enabled=bool(result[9]),
                        mtf_validation_enabled=bool(result[10]),
                        stop_loss_pips=float(result[11]),
                        take_profit_pips=float(result[12]),
                        risk_reward_ratio=float(result[13]),
                        performance_score=float(result[17]),
                        last_optimized=result[18],
                        market_conditions=market_conditions
                    )
                    
                    self.logger.info(f"✅ Retrieved optimal zero-lag parameters for {epic}: "
                                   f"zl_len={result[1]}, band_mult={result[2]:.2f}, {result[3]:.0%} confidence, "
                                   f"squeeze={result[5]}/{result[6]:.1f}-{result[7]}/{result[8]:.1f}, "
                                   f"{result[11]:.0f}/{result[12]:.0f} SL/TP")
                    
                else:
                    # Fallback to default parameters
                    self.logger.warning(f"⚠️ No zero-lag optimization data found for {epic}, using fallbacks")
                    optimal_params = self._get_zerolag_fallback_parameters(epic, market_conditions)
                
                # Cache the result
                self._parameter_cache[cache_key] = optimal_params
                self._cache_timestamps[cache_key] = datetime.now()
                
                return optimal_params
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get zero-lag parameters for {epic}: {e}")
            return self._get_zerolag_fallback_parameters(epic, market_conditions)
    
    def get_all_zerolag_parameters(self, 
                                  market_conditions: Optional[MarketConditions] = None) -> Dict[str, ZeroLagOptimalParameters]:
        """Get optimal zero-lag parameters for all epics with optimization data"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT epic FROM zerolag_best_parameters 
                    ORDER BY best_composite_score DESC
                """)
                
                epics = [row[0] for row in cursor.fetchall()]
                
                # Get parameters for each epic
                all_parameters = {}
                for epic in epics:
                    all_parameters[epic] = self.get_zerolag_parameters(epic, market_conditions)
                
                self.logger.info(f"✅ Retrieved zero-lag parameters for {len(all_parameters)} optimized epics")
                return all_parameters
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get all zero-lag epic parameters: {e}")
            return {}
    
    def get_zerolag_performance_history(self, epic: str, days: int = 30) -> pd.DataFrame:
        """Get zero-lag parameter performance history for analysis"""
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT 
                        run_id, epic, zl_length, band_multiplier, confidence_threshold, timeframe,
                        bb_length, bb_mult, kc_length, kc_mult,
                        smart_money_enabled, mtf_validation_enabled,
                        stop_loss_pips, take_profit_pips, risk_reward_ratio, 
                        total_signals, win_rate, profit_factor, net_pips, composite_score, created_at
                    FROM zerolag_optimization_results
                    WHERE epic = %s 
                        AND created_at >= NOW() - INTERVAL '%s days'
                        AND total_signals >= 3
                    ORDER BY composite_score DESC NULLS LAST
                    LIMIT 100
                """
                
                df = pd.read_sql_query(query, conn, params=[epic, days])
                return df
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get zero-lag performance history for {epic}: {e}")
            return pd.DataFrame()
    
    def suggest_zerolag_parameter_updates(self, epic: str) -> Dict[str, any]:
        """Analyze if zero-lag parameters need updating based on recent performance"""
        try:
            history_df = self.get_zerolag_performance_history(epic, days=7)
            current_params = self.get_zerolag_parameters(epic)
            
            if history_df.empty:
                return {'needs_update': False, 'reason': 'No recent zero-lag optimization data'}
            
            # Compare current vs recent best
            recent_best = history_df.iloc[0]
            performance_gap = recent_best['composite_score'] - current_params.performance_score
            
            suggestions = {
                'needs_update': performance_gap > 0.1,
                'performance_improvement': performance_gap,
                'suggested_zl_length': int(recent_best['zl_length']),
                'suggested_band_multiplier': float(recent_best['band_multiplier']),
                'suggested_confidence': float(recent_best['confidence_threshold']),
                'suggested_squeeze_bb': f"{int(recent_best['bb_length'])}/{recent_best['bb_mult']:.1f}",
                'suggested_squeeze_kc': f"{int(recent_best['kc_length'])}/{recent_best['kc_mult']:.1f}",
                'suggested_sl_tp': f"{recent_best['stop_loss_pips']:.0f}/{recent_best['take_profit_pips']:.0f}",
                'reason': f"Recent zero-lag optimization shows {performance_gap:.3f} improvement potential"
            }
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate zero-lag suggestions for {epic}: {e}")
            return {'needs_update': False, 'error': str(e)}
    
    def _get_zerolag_fallback_parameters(self, epic: str, market_conditions: Optional[MarketConditions] = None) -> ZeroLagOptimalParameters:
        """Get fallback zero-lag parameters when no optimization data exists"""
        
        # Epic-specific fallbacks based on currency pair characteristics
        if 'JPY' in epic:
            # JPY pairs typically need wider stops due to higher volatility
            fallback_sl, fallback_tp = 15.0, 30.0
            fallback_zl_length = 89  # Longer period for more stability
            fallback_band_mult = 2.5  # Wider bands for volatility
        elif 'GBP' in epic:
            # GBP pairs are volatile, need wider stops
            fallback_sl, fallback_tp = 12.0, 25.0
            fallback_zl_length = 70
            fallback_band_mult = 2.0
        elif 'EUR' in epic or 'USD' in epic:
            # Major pairs, standard parameters
            fallback_sl, fallback_tp = 10.0, 20.0
            fallback_zl_length = 50
            fallback_band_mult = 1.5
        else:
            # Other pairs, conservative approach
            fallback_sl, fallback_tp = 15.0, 30.0
            fallback_zl_length = 89
            fallback_band_mult = 2.0
        
        # Market condition adjustments
        if market_conditions:
            if market_conditions.volatility_level == 'high':
                fallback_sl *= 1.2
                fallback_tp *= 1.2
                fallback_band_mult *= 1.2
            elif market_conditions.volatility_level == 'low':
                fallback_sl *= 0.8
                fallback_tp *= 0.8
                fallback_band_mult *= 0.8
        
        return ZeroLagOptimalParameters(
            epic=epic,
            zl_length=fallback_zl_length,
            band_multiplier=fallback_band_mult,
            confidence_threshold=0.65,  # Standard zero-lag confidence
            timeframe='15m',
            bb_length=20,  # Standard squeeze parameters
            bb_mult=2.0,
            kc_length=20,
            kc_mult=1.5,
            smart_money_enabled=False,
            mtf_validation_enabled=False,
            stop_loss_pips=fallback_sl,
            take_profit_pips=fallback_tp,
            risk_reward_ratio=fallback_tp / fallback_sl,
            performance_score=0.0,  # No optimization data
            last_optimized=datetime.now() - timedelta(days=999),  # Very old
            market_conditions=market_conditions
        )
    
    def get_macd_epic_parameters(self, 
                               epic: str, 
                               timeframe: str = '15m',
                               market_conditions: Optional[MarketConditions] = None,
                               force_refresh: bool = False) -> MACDOptimalParameters:
        """
        Get optimal MACD parameters for specific epic and timeframe
        
        Args:
            epic: Trading pair epic (e.g. 'CS.D.EURUSD.CEEM.IP')
            timeframe: Trading timeframe (e.g. '15m', '1h', '4h')
            market_conditions: Current market conditions for context-aware selection
            force_refresh: Force refresh from database even if cached
            
        Returns:
            MACDOptimalParameters object with all MACD trading settings
        """
        cache_key = f"macd_{epic}_{timeframe}_{hash(str(market_conditions)) if market_conditions else 'default'}"
        
        # Check cache first (unless force refresh)
        if not force_refresh and self._is_cache_valid(cache_key):
            self.logger.debug(f"📋 Using cached MACD parameters for {epic} ({timeframe})")
            return self._parameter_cache[cache_key]
        
        # Get from database
        try:
            self.logger.debug(f"🔍 Querying MACD parameters for {epic} ({timeframe})")
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Primary query: Get best MACD parameters for this epic and timeframe
                self.logger.debug("📊 Executing primary MACD query...")
                cursor.execute("""
                    SELECT 
                        epic, best_fast_ema, best_slow_ema, best_signal_ema,
                        best_confidence_threshold, best_timeframe, best_histogram_threshold,
                        best_zero_line_filter, best_rsi_filter_enabled, best_momentum_confirmation,
                        best_mtf_enabled, best_mtf_timeframes, best_smart_money_enabled,
                        optimal_stop_loss_pips, optimal_take_profit_pips,
                        ROUND(optimal_take_profit_pips / optimal_stop_loss_pips, 2) as risk_reward,
                        best_win_rate, best_composite_score, last_updated
                    FROM macd_best_parameters 
                    WHERE epic = %s AND best_timeframe = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, (epic, timeframe))
                
                result = cursor.fetchone()
                self.logger.debug(f"🔍 Primary query result: {result}")
                self.logger.debug(f"🔍 Result is None: {result is None}")
                self.logger.debug(f"🔍 Result truthiness: {bool(result)}")
                
                # If no timeframe-specific data, try to get any available data for this epic
                if not result:
                    self.logger.debug(f"⚠️ ENTERING FALLBACK: No {timeframe} data for {epic}, trying fallback query")
                    cursor.execute("""
                        SELECT 
                            epic, best_fast_ema, best_slow_ema, best_signal_ema,
                            best_confidence_threshold, best_timeframe, best_histogram_threshold,
                            best_zero_line_filter, best_rsi_filter_enabled, best_momentum_confirmation,
                            best_mtf_enabled, best_mtf_timeframes, best_smart_money_enabled,
                            optimal_stop_loss_pips, optimal_take_profit_pips,
                            ROUND(optimal_take_profit_pips / optimal_stop_loss_pips, 2) as risk_reward,
                            best_win_rate, best_composite_score, last_updated
                        FROM macd_best_parameters 
                        WHERE epic = %s
                        ORDER BY best_composite_score DESC NULLS LAST
                        LIMIT 1
                    """, (epic,))
                    
                    result = cursor.fetchone()
                    self.logger.debug(f"🔍 Fallback query result: {result}")
                
                if result:
                    # Create optimal MACD parameters from database result
                    optimal_params = MACDOptimalParameters(
                        epic=result[0],
                        fast_ema=int(result[1]),
                        slow_ema=int(result[2]),
                        signal_ema=int(result[3]),
                        confidence_threshold=float(result[4]),
                        timeframe=result[5],
                        histogram_threshold=float(result[6]),
                        zero_line_filter=result[7],
                        rsi_filter_enabled=result[8],
                        momentum_confirmation=result[9],
                        mtf_enabled=result[10],
                        mtf_timeframes=result[11],
                        smart_money_enabled=result[12],
                        stop_loss_pips=float(result[13]),
                        take_profit_pips=float(result[14]),
                        risk_reward_ratio=float(result[15]),
                        win_rate=float(result[16]) if result[16] else 0.0,
                        performance_score=float(result[17]) if result[17] else 0.0,
                        last_optimized=result[18],
                        market_conditions=market_conditions
                    )
                    
                    self.logger.info(f"✅ Retrieved optimal MACD parameters for {epic} ({timeframe}): "
                                   f"{result[1]}/{result[2]}/{result[3]} periods, {result[4]:.0%} confidence, "
                                   f"{result[13]:.0f}/{result[14]:.0f} SL/TP")
                    
                else:
                    # Fallback to default MACD parameters
                    self.logger.warning(f"⚠️ No MACD optimization data found for {epic} ({timeframe}), using fallbacks")
                    optimal_params = self._get_macd_fallback_parameters(epic, timeframe, market_conditions)
                
                # Cache the result
                self._parameter_cache[cache_key] = optimal_params
                self._cache_timestamps[cache_key] = datetime.now()
                
                return optimal_params
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get MACD parameters for {epic} ({timeframe}): {e}")
            return self._get_macd_fallback_parameters(epic, timeframe, market_conditions)
    
    def _get_session_adjustments(self, market_conditions: Optional[MarketConditions] = None) -> Dict[str, float]:
        """Get session-based trading adjustments for MACD parameters"""

        # Default adjustments (neutral)
        adjustments = {
            'confidence_multiplier': 1.0,
            'sensitivity_multiplier': 1.0,
            'volatility_multiplier': 1.0
        }

        if not market_conditions or not market_conditions.session:
            return adjustments

        # PHASE 2 ENHANCEMENT: Session-based parameter optimization
        # Based on TradingView community research and trading session analysis
        session = market_conditions.session.lower()

        if session == 'asian':
            # Asian session: Lower volatility, ranging markets
            adjustments.update({
                'confidence_multiplier': 1.1,   # Higher confidence needed (less volatile)
                'sensitivity_multiplier': 0.8,  # Less sensitive (lower threshold)
                'volatility_multiplier': 0.8    # Tighter stops (lower volatility)
            })
        elif session == 'london':
            # London session: High volatility, strong trends
            adjustments.update({
                'confidence_multiplier': 0.9,   # Lower confidence acceptable (more opportunities)
                'sensitivity_multiplier': 1.2,  # More sensitive (higher threshold)
                'volatility_multiplier': 1.3    # Wider stops (higher volatility)
            })
        elif session == 'new_york':
            # New York session: High volatility, continuation patterns
            adjustments.update({
                'confidence_multiplier': 0.95,  # Slightly lower confidence
                'sensitivity_multiplier': 1.1,  # Moderately sensitive
                'volatility_multiplier': 1.2    # Moderately wider stops
            })
        elif session in ['overlap', 'london_new_york']:
            # Overlap sessions: Highest volatility and opportunity
            adjustments.update({
                'confidence_multiplier': 0.85,  # Lower confidence (more aggressive)
                'sensitivity_multiplier': 1.3,  # Most sensitive
                'volatility_multiplier': 1.4    # Widest stops (highest volatility)
            })

        return adjustments

    def _get_macd_fallback_parameters(self, epic: str, timeframe: str = '15m', market_conditions: Optional[MarketConditions] = None) -> MACDOptimalParameters:
        """Get fallback MACD parameters with ENHANCED TradingView-derived adaptive selection"""

        # Import MACD config for fallback values
        from configdata.strategies.config_macd_strategy import MACD_PERIODS

        # PHASE 2 ENHANCEMENT: TradingView Community-Validated Parameter Sets
        # Research-based novel parameter combinations for different market conditions
        novel_parameters = {
            'scalping': (5, 13, 3),      # Ultra-fast for scalping (TradingView community favorite)
            'trend_following': (8, 34, 13),  # Enhanced trend detection (Zeiierman-inspired)
            'swing_trading': (21, 55, 9),    # Longer-term swings (ChartPrime methodology)
            'standard': (12, 26, 9),         # Classic Appel parameters
            'fast_response': (8, 17, 9),     # Faster response (proven 8-17-9 from database)
            'smooth_trend': (19, 39, 9)      # Smoother for trending markets
        }

        # ADAPTIVE PARAMETER SELECTION based on timeframe and market conditions
        if timeframe == '5m':
            # Use scalping parameters for very fast timeframe
            fast_ema, slow_ema, signal_ema = novel_parameters['scalping']
            confidence_threshold = 0.60  # Higher confidence for noisy timeframe
        elif timeframe == '15m':
            # Market regime adaptive selection for 15m
            if market_conditions and market_conditions.market_regime == 'trending':
                fast_ema, slow_ema, signal_ema = novel_parameters['trend_following']
                confidence_threshold = 0.50  # Lower for trending markets
            elif market_conditions and market_conditions.market_regime == 'ranging':
                fast_ema, slow_ema, signal_ema = novel_parameters['scalping']  # Faster for ranging
                confidence_threshold = 0.60  # Higher for ranging markets
            else:
                fast_ema, slow_ema, signal_ema = novel_parameters['fast_response']  # Proven 8-17-9
                confidence_threshold = 0.55  # Standard confidence
        elif timeframe == '1h':
            # Use swing parameters for hourly timeframe
            fast_ema, slow_ema, signal_ema = novel_parameters['swing_trading']
            confidence_threshold = 0.50  # Lower confidence for stable timeframe
        elif timeframe in ['4h', '1d']:
            # Smooth trend parameters for higher timeframes
            fast_ema, slow_ema, signal_ema = novel_parameters['smooth_trend']
            confidence_threshold = 0.45  # Even lower for very stable timeframes
        else:
            # Default fallback to proven fast response
            fast_ema, slow_ema, signal_ema = novel_parameters['fast_response']
            confidence_threshold = 0.55
        
        # PHASE 2 ENHANCEMENT: Session-based parameter adaptation
        session_adjustments = self._get_session_adjustments(market_conditions)
        confidence_threshold *= session_adjustments['confidence_multiplier']

        # Apply session-based histogram threshold adjustments
        histogram_threshold = 0.00003 * session_adjustments['sensitivity_multiplier']

        # JPY pairs typically need different pip values
        if 'JPY' in epic.upper():
            fallback_sl = 15.0 * session_adjustments['volatility_multiplier']
            fallback_tp = 30.0 * session_adjustments['volatility_multiplier']
        else:
            fallback_sl = 10.0 * session_adjustments['volatility_multiplier']
            fallback_tp = 20.0 * session_adjustments['volatility_multiplier']
        
        return MACDOptimalParameters(
            epic=epic,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            signal_ema=signal_ema,
            confidence_threshold=confidence_threshold,
            timeframe=timeframe,
            histogram_threshold=histogram_threshold,  # Session-adjusted threshold
            zero_line_filter=False,
            rsi_filter_enabled=True,  # Enable RSI filter by default
            momentum_confirmation=True,  # Enable momentum confirmation
            mtf_enabled=False,
            mtf_timeframes=None,
            smart_money_enabled=False,
            stop_loss_pips=fallback_sl,
            take_profit_pips=fallback_tp,
            risk_reward_ratio=fallback_tp / fallback_sl,
            win_rate=0.5,  # Assume neutral fallback
            performance_score=0.0,  # No optimization data
            last_optimized=datetime.now() - timedelta(days=999),  # Very old
            market_conditions=market_conditions
        )

    # =============================================================================
    # Ichimoku Strategy Parameter Methods
    # =============================================================================

    def get_ichimoku_epic_parameters(self,
                                   epic: str,
                                   market_conditions: Optional[MarketConditions] = None,
                                   force_refresh: bool = False) -> IchimokuOptimalParameters:
        """
        Get optimal Ichimoku parameters for specific epic

        Args:
            epic: Trading pair epic (e.g. 'CS.D.EURUSD.CEEM.IP')
            market_conditions: Current market conditions for context-aware selection
            force_refresh: Force refresh from database even if cached

        Returns:
            IchimokuOptimalParameters object with all Ichimoku trading settings
        """
        cache_key = f"ichimoku_{epic}_{hash(str(market_conditions)) if market_conditions else 'default'}"

        # Check cache first (unless force refresh)
        if not force_refresh and self._is_cache_valid(cache_key):
            self.logger.debug(f"📋 Using cached Ichimoku parameters for {epic}")
            return self._parameter_cache[cache_key]

        # Get from database
        try:
            self.logger.debug(f"🔍 Querying Ichimoku parameters for {epic}")
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Primary query: Get best Ichimoku parameters for this epic
                self.logger.debug("📊 Executing primary Ichimoku query...")
                cursor.execute("""
                    SELECT
                        epic, best_tenkan_period, best_kijun_period, best_senkou_b_period,
                        best_chikou_shift, best_cloud_shift, best_confidence_threshold, best_timeframe,
                        best_cloud_thickness_threshold, best_tk_cross_strength_threshold, best_chikou_clear_threshold,
                        best_cloud_filter_enabled, best_chikou_filter_enabled, best_tk_filter_enabled,
                        best_mtf_enabled, best_mtf_timeframes, best_mtf_min_alignment,
                        best_mtf_cloud_weight, best_mtf_tk_weight, best_mtf_chikou_weight,
                        best_momentum_confluence_enabled, best_smart_money_enabled, best_ema_200_trend_filter,
                        best_contradiction_filter_enabled, optimal_stop_loss_pips, optimal_take_profit_pips,
                        ROUND(optimal_take_profit_pips / optimal_stop_loss_pips, 2) as risk_reward,
                        best_win_rate, best_composite_score, best_tk_cross_accuracy, best_cloud_breakout_accuracy,
                        best_chikou_confirmation_rate, best_perfect_alignment_rate, best_mtf_alignment_avg, last_updated
                    FROM ichimoku_best_parameters
                    WHERE epic = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, (epic,))

                result = cursor.fetchone()
                self.logger.debug(f"🔍 Primary Ichimoku query result: {bool(result)}")

                if result:
                    # Create optimal Ichimoku parameters from database result
                    optimal_params = IchimokuOptimalParameters(
                        epic=result[0],
                        tenkan_period=int(result[1]),
                        kijun_period=int(result[2]),
                        senkou_b_period=int(result[3]),
                        chikou_shift=int(result[4]),
                        cloud_shift=int(result[5]),
                        confidence_threshold=float(result[6]),
                        timeframe=result[7],
                        cloud_thickness_threshold=float(result[8]),
                        tk_cross_strength_threshold=float(result[9]),
                        chikou_clear_threshold=float(result[10]),
                        cloud_filter_enabled=bool(result[11]),
                        chikou_filter_enabled=bool(result[12]),
                        tk_filter_enabled=bool(result[13]),
                        mtf_enabled=bool(result[14]),
                        mtf_timeframes=result[15],
                        mtf_min_alignment=float(result[16]),
                        mtf_cloud_weight=float(result[17]),
                        mtf_tk_weight=float(result[18]),
                        mtf_chikou_weight=float(result[19]),
                        momentum_confluence_enabled=bool(result[20]),
                        smart_money_enabled=bool(result[21]),
                        ema_200_trend_filter=bool(result[22]),
                        contradiction_filter_enabled=bool(result[23]),
                        stop_loss_pips=float(result[24]),
                        take_profit_pips=float(result[25]),
                        risk_reward_ratio=float(result[26]),
                        win_rate=float(result[27]),
                        performance_score=float(result[28]),
                        tk_cross_accuracy=float(result[29]),
                        cloud_breakout_accuracy=float(result[30]),
                        chikou_confirmation_rate=float(result[31]),
                        perfect_alignment_rate=float(result[32]),
                        mtf_alignment_avg=float(result[33]),
                        last_optimized=result[34],
                        market_conditions=market_conditions
                    )

                    # Cache the result
                    self._parameter_cache[cache_key] = optimal_params
                    self._cache_timestamps[cache_key] = datetime.now()

                    self.logger.info(f"✅ Retrieved optimized Ichimoku parameters for {epic}: {optimal_params.tenkan_period}-{optimal_params.kijun_period}-{optimal_params.senkou_b_period}")
                    return optimal_params

                else:
                    # No optimization data available - use traditional defaults
                    self.logger.info(f"⚠️ No Ichimoku optimization data for {epic}, using traditional defaults (9-26-52-26)")

        except Exception as e:
            self.logger.error(f"❌ Error retrieving Ichimoku parameters for {epic}: {e}")

        # Fallback to traditional Ichimoku defaults (9-26-52-26)
        return IchimokuOptimalParameters(
            epic=epic,
            tenkan_period=9,
            kijun_period=26,
            senkou_b_period=52,
            chikou_shift=26,
            cloud_shift=26,
            confidence_threshold=0.55,
            timeframe='15m',
            cloud_thickness_threshold=0.0001,
            tk_cross_strength_threshold=0.5,
            chikou_clear_threshold=0.0002,
            cloud_filter_enabled=True,
            chikou_filter_enabled=True,
            tk_filter_enabled=True,
            mtf_enabled=False,
            mtf_timeframes=None,
            mtf_min_alignment=0.6,
            mtf_cloud_weight=0.4,
            mtf_tk_weight=0.3,
            mtf_chikou_weight=0.3,
            momentum_confluence_enabled=False,
            smart_money_enabled=False,
            ema_200_trend_filter=False,
            contradiction_filter_enabled=True,
            stop_loss_pips=15.0,
            take_profit_pips=30.0,
            risk_reward_ratio=2.0,
            win_rate=0.5,
            performance_score=0.0,
            tk_cross_accuracy=0.5,
            cloud_breakout_accuracy=0.5,
            chikou_confirmation_rate=0.5,
            perfect_alignment_rate=0.5,
            mtf_alignment_avg=0.5,
            last_optimized=datetime.now() - timedelta(days=999),  # Very old
            market_conditions=market_conditions
        )

    # =============================================================================
    # SMC Strategy Parameter Methods
    # =============================================================================
    
    def get_smc_epic_parameters(self, epic: str, market_conditions: Optional[MarketConditions] = None) -> SMCOptimalParameters:
        """
        Get optimal SMC parameters for an epic from optimization results
        
        Args:
            epic: Epic code (e.g., 'CS.D.EURUSD.MINI.IP')
            market_conditions: Optional market conditions for parameter adjustment
            
        Returns:
            SMCOptimalParameters with optimal settings
        """
        
        cache_key = f"smc_{epic}_{hash(str(market_conditions))}"
        
        # Check cache first
        if cache_key in self._parameter_cache:
            cached_params, cache_time = self._parameter_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=self.cache_duration):
                self.logger.debug(f"🔄 Using cached SMC parameters for {epic}")
                return cached_params
        
        try:
            with self.db_manager.get_db_connection() as conn:
                query = """
                SELECT 
                    best_smc_config, best_confidence_level, best_timeframe, use_smart_money,
                    optimal_swing_length, optimal_structure_confirmation, 
                    optimal_bos_threshold, optimal_choch_threshold,
                    optimal_order_block_length, optimal_order_block_volume_factor,
                    optimal_order_block_buffer, optimal_max_order_blocks,
                    optimal_fvg_min_size, optimal_fvg_max_age, optimal_fvg_fill_threshold,
                    optimal_zone_min_touches, optimal_zone_max_age, optimal_zone_strength_factor,
                    optimal_confluence_required, optimal_min_risk_reward,
                    optimal_max_distance_to_zone, optimal_min_signal_confidence,
                    optimal_use_higher_tf, optimal_higher_tf_multiplier, optimal_mtf_confluence_weight,
                    optimal_stop_loss_pips, optimal_take_profit_pips, optimal_risk_reward_ratio,
                    best_win_rate, best_performance_score, confluence_accuracy, 
                    structure_break_accuracy, order_block_success_rate, fvg_success_rate,
                    last_optimized, market_regime, volatility_regime, session_preference
                FROM smc_best_parameters
                WHERE epic = %s
                """
                
                with conn.cursor() as cursor:
                    cursor.execute(query, (epic,))
                    result = cursor.fetchone()
                    
                    if result:
                        # Create optimal parameters from database result
                        optimal_params = SMCOptimalParameters(
                            epic=epic,
                            smc_config=result[0],
                            confidence_threshold=float(result[1]),
                            timeframe=result[2],
                            use_smart_money=result[3],
                            
                            # Market Structure Parameters
                            swing_length=result[4],
                            structure_confirmation=result[5],
                            bos_threshold=float(result[6]),
                            choch_threshold=float(result[7]),
                            
                            # Order Block Parameters
                            order_block_length=result[8],
                            order_block_volume_factor=float(result[9]),
                            order_block_buffer=float(result[10]),
                            max_order_blocks=result[11],
                            
                            # Fair Value Gap Parameters
                            fvg_min_size=float(result[12]),
                            fvg_max_age=result[13],
                            fvg_fill_threshold=float(result[14]),
                            
                            # Supply/Demand Zone Parameters
                            zone_min_touches=result[15],
                            zone_max_age=result[16],
                            zone_strength_factor=float(result[17]),
                            
                            # Signal Generation Parameters
                            confluence_required=float(result[18]),
                            min_risk_reward=float(result[19]),
                            max_distance_to_zone=float(result[20]),
                            min_signal_confidence=float(result[21]),
                            
                            # Multi-timeframe Parameters
                            use_higher_tf=result[22],
                            higher_tf_multiplier=result[23],
                            mtf_confluence_weight=float(result[24]),
                            
                            # Risk Management
                            stop_loss_pips=float(result[25]),
                            take_profit_pips=float(result[26]),
                            risk_reward_ratio=float(result[27]),
                            
                            # Performance Metrics
                            win_rate=float(result[28]),
                            performance_score=float(result[29]),
                            confluence_accuracy=float(result[30]),
                            structure_break_accuracy=float(result[31] or 0),
                            order_block_success_rate=float(result[32] or 0),
                            fvg_success_rate=float(result[33] or 0),
                            
                            last_optimized=result[34],
                            market_conditions=market_conditions
                        )
                        
                        # Apply market condition adjustments
                        if market_conditions:
                            optimal_params = self._apply_smc_market_adjustments(optimal_params, market_conditions)
                        
                        # Cache the result
                        self._parameter_cache[cache_key] = (optimal_params, datetime.now())
                        
                        self.logger.info(f"📊 Retrieved SMC optimal parameters for {epic}: "
                                       f"{optimal_params.smc_config} config, "
                                       f"{optimal_params.confidence_threshold:.0%} confidence, "
                                       f"{optimal_params.timeframe} timeframe, "
                                       f"Score: {optimal_params.performance_score:.6f}")
                        
                        return optimal_params
                    else:
                        # No optimization data found, use fallback
                        self.logger.info(f"⚠️ No SMC optimization data for {epic}, using fallback parameters")
                        return self._get_smc_fallback_parameters(epic, market_conditions)
                        
        except Exception as e:
            self.logger.error(f"❌ Failed to get SMC parameters for {epic}: {e}")
            return self._get_smc_fallback_parameters(epic, market_conditions)
    
    def _apply_smc_market_adjustments(self, params: SMCOptimalParameters, conditions: MarketConditions) -> SMCOptimalParameters:
        """Apply market condition adjustments to SMC parameters"""
        
        # Create a copy of the parameters
        import copy
        adjusted_params = copy.deepcopy(params)
        
        # Volatility adjustments
        if conditions.volatility_level == 'high':
            # Increase stop losses and take profits for high volatility
            adjusted_params.stop_loss_pips *= 1.3
            adjusted_params.take_profit_pips *= 1.3
            # Wider order block buffers
            adjusted_params.order_block_buffer *= 1.2
            # Larger FVG minimum size
            adjusted_params.fvg_min_size *= 1.2
            # More distance to zones allowed
            adjusted_params.max_distance_to_zone *= 1.2
            
        elif conditions.volatility_level == 'low':
            # Decrease stop losses and take profits for low volatility
            adjusted_params.stop_loss_pips *= 0.8
            adjusted_params.take_profit_pips *= 0.8
            # Tighter order block buffers
            adjusted_params.order_block_buffer *= 0.8
            # Smaller FVG minimum size
            adjusted_params.fvg_min_size *= 0.8
            # Less distance to zones
            adjusted_params.max_distance_to_zone *= 0.8
        
        # Session adjustments
        if conditions.session == 'asian':
            # Asian session: more conservative parameters
            adjusted_params.confluence_required *= 1.2  # Require more confluence
            adjusted_params.min_signal_confidence *= 1.1  # Higher confidence threshold
            
        elif conditions.session in ['london', 'new_york']:
            # Major sessions: can be slightly more aggressive
            adjusted_params.confluence_required *= 0.9
            
        # Market regime adjustments
        if conditions.market_regime == 'ranging':
            # Ranging markets: focus on mean reversion
            adjusted_params.zone_strength_factor *= 1.2  # Stronger zones needed
            adjusted_params.fvg_fill_threshold *= 0.8  # Partial fills more acceptable
            
        elif conditions.market_regime == 'trending':
            # Trending markets: allow more aggressive entries
            adjusted_params.structure_confirmation = max(1, adjusted_params.structure_confirmation - 1)
            adjusted_params.max_distance_to_zone *= 1.3  # Allow entries further from zones
        
        return adjusted_params
    
    def _get_smc_fallback_parameters(self, epic: str, market_conditions: Optional[MarketConditions] = None) -> SMCOptimalParameters:
        """Get fallback SMC parameters when optimization data is not available"""
        
        # Import SMC config for fallback values
        from configdata.strategies.config_smc_strategy import SMC_STRATEGY_CONFIG, ACTIVE_SMC_CONFIG
        
        # Get default SMC configuration
        default_config = SMC_STRATEGY_CONFIG.get(ACTIVE_SMC_CONFIG, SMC_STRATEGY_CONFIG.get('moderate', {}))
        
        # Epic-specific adjustments
        if 'JPY' in epic.upper():
            fallback_sl = 15.0
            fallback_tp = 30.0
            # JPY pairs need different thresholds
            bos_threshold = default_config.get('bos_threshold', 0.00015) * 100
            choch_threshold = default_config.get('choch_threshold', 0.00015) * 100
        else:
            fallback_sl = 10.0
            fallback_tp = 20.0
            bos_threshold = default_config.get('bos_threshold', 0.00015)
            choch_threshold = default_config.get('choch_threshold', 0.00015)
        
        return SMCOptimalParameters(
            epic=epic,
            smc_config=ACTIVE_SMC_CONFIG,
            confidence_threshold=default_config.get('min_confidence', 0.55),
            timeframe='15m',  # Standard timeframe
            use_smart_money=True,
            
            # Market Structure Parameters
            swing_length=default_config.get('swing_length', 5),
            structure_confirmation=default_config.get('structure_confirmation', 3),
            bos_threshold=bos_threshold,
            choch_threshold=choch_threshold,
            
            # Order Block Parameters
            order_block_length=default_config.get('order_block_length', 3),
            order_block_volume_factor=default_config.get('order_block_volume_factor', 1.8),
            order_block_buffer=default_config.get('order_block_buffer', 2.0),
            max_order_blocks=default_config.get('max_order_blocks', 5),
            
            # Fair Value Gap Parameters
            fvg_min_size=default_config.get('fvg_min_size', 3.0),
            fvg_max_age=default_config.get('fvg_max_age', 25),
            fvg_fill_threshold=default_config.get('fvg_fill_threshold', 0.5),
            
            # Supply/Demand Zone Parameters
            zone_min_touches=default_config.get('zone_min_touches', 2),
            zone_max_age=default_config.get('zone_max_age', 50),
            zone_strength_factor=default_config.get('zone_strength_factor', 1.4),
            
            # Signal Generation Parameters
            confluence_required=default_config.get('confluence_required', 2.0),
            min_risk_reward=default_config.get('min_risk_reward', 1.5),
            max_distance_to_zone=default_config.get('max_distance_to_zone', 10.0),
            min_signal_confidence=default_config.get('min_confidence', 0.55),
            
            # Multi-timeframe Parameters
            use_higher_tf=default_config.get('use_higher_tf', True),
            higher_tf_multiplier=default_config.get('higher_tf_multiplier', 4),
            mtf_confluence_weight=default_config.get('mtf_confluence_weight', 0.8),
            
            # Risk Management
            stop_loss_pips=fallback_sl,
            take_profit_pips=fallback_tp,
            risk_reward_ratio=fallback_tp / fallback_sl,
            
            # Performance Metrics (fallback values)
            win_rate=0.5,  # Neutral assumption
            performance_score=0.0,  # No optimization data
            confluence_accuracy=0.7,  # Conservative estimate
            structure_break_accuracy=0.6,
            order_block_success_rate=0.5,
            fvg_success_rate=0.4,
            
            last_optimized=datetime.now() - timedelta(days=999),  # Very old
            market_conditions=market_conditions
        )

    def get_mean_reversion_epic_parameters(self,
                                         epic: str,
                                         timeframe: str = '15m',
                                         market_conditions: Optional[MarketConditions] = None,
                                         force_refresh: bool = False) -> MeanReversionOptimalParameters:
        """
        Get optimal Mean Reversion parameters for specific epic and timeframe

        Args:
            epic: Trading pair epic (e.g. 'CS.D.EURUSD.CEEM.IP')
            timeframe: Trading timeframe (e.g. '15m', '1h', '4h')
            market_conditions: Current market conditions for context-aware selection
            force_refresh: Force refresh from database even if cached

        Returns:
            MeanReversionOptimalParameters object with all mean reversion trading settings
        """
        cache_key = f"mean_reversion_{epic}_{timeframe}_{hash(str(market_conditions)) if market_conditions else 'default'}"

        # Check cache first (unless force refresh)
        if not force_refresh and self._is_cache_valid(cache_key):
            self.logger.debug(f"📋 Using cached Mean Reversion parameters for {epic} ({timeframe})")
            return self._parameter_cache[cache_key]

        # Get from database
        try:
            self.logger.debug(f"🔍 Querying Mean Reversion parameters for {epic} ({timeframe})")
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Primary query: Get best mean reversion parameters for this epic and timeframe
                self.logger.debug("📊 Executing primary Mean Reversion query...")
                cursor.execute("""
                    SELECT
                        epic, timeframe, confidence_threshold,
                        luxalgo_length, luxalgo_smoothing,
                        luxalgo_overbought_threshold, luxalgo_oversold_threshold,
                        luxalgo_extreme_ob_threshold, luxalgo_extreme_os_threshold,
                        mtf_rsi_period, mtf_rsi_timeframes, mtf_min_alignment,
                        mtf_rsi_overbought, mtf_rsi_oversold,
                        rsi_ema_period, rsi_ema_rsi_period,
                        rsi_ema_divergence_sensitivity, rsi_ema_min_divergence_strength,
                        squeeze_bb_length, squeeze_bb_stddev,
                        squeeze_kc_length, squeeze_kc_multiplier, squeeze_momentum_length,
                        stop_loss_pips, take_profit_pips, risk_reward_ratio,
                        performance_score, win_rate, profit_factor, net_pips,
                        last_updated
                    FROM mean_reversion_best_parameters
                    WHERE epic = %s AND timeframe = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """, (epic, timeframe))

                result = cursor.fetchone()
                self.logger.debug(f"🔍 Primary query result: {result}")

                # If no timeframe-specific data, try to get any available data for this epic
                if not result:
                    self.logger.debug(f"⚠️ ENTERING FALLBACK: No {timeframe} data for {epic}, trying fallback query")
                    cursor.execute("""
                        SELECT
                            epic, timeframe, confidence_threshold,
                            luxalgo_length, luxalgo_smoothing,
                            luxalgo_overbought_threshold, luxalgo_oversold_threshold,
                            luxalgo_extreme_ob_threshold, luxalgo_extreme_os_threshold,
                            mtf_rsi_period, mtf_rsi_timeframes, mtf_min_alignment,
                            mtf_rsi_overbought, mtf_rsi_oversold,
                            rsi_ema_period, rsi_ema_rsi_period,
                            rsi_ema_divergence_sensitivity, rsi_ema_min_divergence_strength,
                            squeeze_bb_length, squeeze_bb_stddev,
                            squeeze_kc_length, squeeze_kc_multiplier, squeeze_momentum_length,
                            stop_loss_pips, take_profit_pips, risk_reward_ratio,
                            performance_score, win_rate, profit_factor, net_pips,
                            last_updated
                        FROM mean_reversion_best_parameters
                        WHERE epic = %s
                        ORDER BY performance_score DESC NULLS LAST
                        LIMIT 1
                    """, (epic,))

                    result = cursor.fetchone()
                    self.logger.debug(f"🔍 Fallback query result: {result}")

                if result:
                    # Create optimal mean reversion parameters from database result
                    optimal_params = MeanReversionOptimalParameters(
                        epic=result[0],
                        timeframe=result[1],
                        confidence_threshold=float(result[2]),

                        # LuxAlgo Oscillator parameters
                        luxalgo_length=int(result[3]),
                        luxalgo_smoothing=int(result[4]),
                        luxalgo_overbought_threshold=float(result[5]),
                        luxalgo_oversold_threshold=float(result[6]),
                        luxalgo_extreme_ob_threshold=float(result[7]),
                        luxalgo_extreme_os_threshold=float(result[8]),

                        # MTF RSI parameters
                        mtf_rsi_period=int(result[9]),
                        mtf_rsi_timeframes=result[10],
                        mtf_min_alignment=float(result[11]),
                        mtf_rsi_overbought=float(result[12]),
                        mtf_rsi_oversold=float(result[13]),

                        # RSI-EMA Divergence parameters
                        rsi_ema_period=int(result[14]),
                        rsi_ema_rsi_period=int(result[15]),
                        rsi_ema_divergence_sensitivity=float(result[16]),
                        rsi_ema_min_divergence_strength=float(result[17]),

                        # Squeeze Momentum parameters
                        squeeze_bb_length=int(result[18]),
                        squeeze_bb_mult=float(result[19]),
                        squeeze_kc_length=int(result[20]),
                        squeeze_kc_mult=float(result[21]),
                        squeeze_momentum_length=int(result[22]),
                        squeeze_require_release=bool(result[23]),
                        squeeze_momentum_threshold=float(result[24]),

                        # Oscillator confluence parameters
                        bull_confluence_threshold=float(result[25]),
                        bear_confluence_threshold=float(result[26]),
                        luxalgo_weight=float(result[27]),
                        mtf_rsi_weight=float(result[28]),
                        divergence_weight=float(result[29]),
                        squeeze_weight=float(result[30]),

                        # Mean reversion zone parameters
                        zone_validation_enabled=bool(result[31]),
                        zone_lookback_periods=int(result[32]),
                        zone_multiplier=float(result[33]),
                        require_zone_touch=bool(result[34]),
                        min_zone_distance=float(result[35]),
                        max_zone_age=int(result[36]),
                        zone_confidence_boost=float(result[37]),

                        # Market regime parameters
                        market_regime_enabled=bool(result[38]),
                        disable_in_strong_trend=bool(result[39]),
                        trend_strength_threshold=float(result[40]),
                        volatility_period=int(result[41]),
                        trend_period=int(result[42]),
                        ranging_threshold=float(result[43]),

                        # MTF analysis parameters
                        mtf_analysis_enabled=bool(result[44]),
                        mtf_timeframes_list=result[45],
                        mtf_min_alignment_score=float(result[46]),
                        require_higher_tf_confluence=bool(result[47]),
                        mtf_confidence_boost_full_alignment=float(result[48]),

                        # Signal quality parameters
                        min_confidence=float(result[49]),
                        min_risk_reward=float(result[50]),
                        max_signals_per_day=int(result[51]),
                        min_signal_spacing_hours=int(result[52]),

                        # Risk management parameters
                        stop_loss_pips=float(result[53]),
                        take_profit_pips=float(result[54]),
                        risk_reward_ratio=float(result[55]),
                        position_size_multiplier=float(result[56]),
                        max_drawdown_threshold=float(result[57]),
                        trail_stop_enabled=bool(result[58]),

                        # Performance metrics
                        performance_score=float(result[60]),
                        win_rate=float(result[59]),
                        avg_profit_pips=float(result[61]),
                        max_consecutive_losses=int(result[62]),
                        profit_factor=float(result[63]),
                        sharpe_ratio=float(result[64]),
                        confluence_accuracy=float(result[65]),
                        divergence_success_rate=float(result[66]),
                        zone_touch_accuracy=float(result[67]),
                        regime_filter_effectiveness=float(result[68]),

                        last_optimized=result[69],
                        market_conditions=market_conditions
                    )

                    # Apply market condition adjustments if provided
                    if market_conditions:
                        optimal_params = self._apply_mean_reversion_market_adjustments(optimal_params, market_conditions)

                    # Cache the result
                    self._parameter_cache[cache_key] = optimal_params
                    self._cache_timestamps[cache_key] = datetime.now()

                    self.logger.info(f"✅ Using optimized Mean Reversion parameters for {epic} ({timeframe}): "
                                   f"Confidence: {optimal_params.confidence_threshold:.1%}, "
                                   f"Performance: {optimal_params.performance_score:.3f}")

                    return optimal_params
                else:
                    # No optimization data found, use fallback
                    self.logger.info(f"⚠️ No Mean Reversion optimization data for {epic}, using fallback parameters")
                    return self._get_mean_reversion_fallback_parameters(epic, timeframe, market_conditions)

        except Exception as e:
            self.logger.error(f"❌ Failed to get Mean Reversion parameters for {epic} ({timeframe}): {e}")
            return self._get_mean_reversion_fallback_parameters(epic, timeframe, market_conditions)

    def _apply_mean_reversion_market_adjustments(self, params: MeanReversionOptimalParameters, conditions: MarketConditions) -> MeanReversionOptimalParameters:
        """Apply market condition adjustments to mean reversion parameters"""

        # Create a copy of the parameters
        import copy
        adjusted_params = copy.deepcopy(params)

        # Volatility adjustments
        if conditions.volatility_level == 'high':
            # Increase stop losses and take profits for high volatility
            adjusted_params.stop_loss_pips *= 1.2
            adjusted_params.take_profit_pips *= 1.2
            # Widen oscillator thresholds
            adjusted_params.luxalgo_overbought_threshold *= 0.95  # Make less sensitive
            adjusted_params.luxalgo_oversold_threshold *= 1.05
            # Increase zone distance
            adjusted_params.min_zone_distance *= 1.3

        elif conditions.volatility_level == 'low':
            # Decrease stop losses and take profits for low volatility
            adjusted_params.stop_loss_pips *= 0.8
            adjusted_params.take_profit_pips *= 0.8
            # Tighten oscillator thresholds
            adjusted_params.luxalgo_overbought_threshold *= 1.05  # Make more sensitive
            adjusted_params.luxalgo_oversold_threshold *= 0.95
            # Decrease zone distance
            adjusted_params.min_zone_distance *= 0.8

        # Session adjustments
        if conditions.session == 'asian':
            # Asian session: more conservative parameters
            adjusted_params.bull_confluence_threshold *= 1.1
            adjusted_params.bear_confluence_threshold *= 1.1
            adjusted_params.min_confidence *= 1.05

        elif conditions.session in ['london', 'new_york']:
            # Major sessions: can be slightly more aggressive
            adjusted_params.bull_confluence_threshold *= 0.95
            adjusted_params.bear_confluence_threshold *= 0.95

        # Market regime adjustments
        if conditions.market_regime == 'ranging':
            # Ranging markets: optimal for mean reversion
            adjusted_params.zone_confidence_boost *= 1.2
            adjusted_params.require_zone_touch = True

        elif conditions.market_regime == 'trending':
            # Trending markets: more cautious mean reversion
            adjusted_params.trend_strength_threshold *= 0.8  # Lower threshold = more cautious
            adjusted_params.disable_in_strong_trend = True

        return adjusted_params

    def _get_mean_reversion_fallback_parameters(self, epic: str, timeframe: str = '15m', market_conditions: Optional[MarketConditions] = None) -> MeanReversionOptimalParameters:
        """Get fallback mean reversion parameters when optimization data is not available"""

        # Epic-specific adjustments
        if 'JPY' in epic.upper():
            fallback_sl = 20.0
            fallback_tp = 35.0
            luxalgo_overbought = 85
            luxalgo_oversold = 15
            min_zone_distance = 15.0
        else:
            fallback_sl = 15.0
            fallback_tp = 25.0
            luxalgo_overbought = 80
            luxalgo_oversold = 20
            min_zone_distance = 10.0

        return MeanReversionOptimalParameters(
            epic=epic,
            timeframe=timeframe,
            confidence_threshold=0.6,

            # LuxAlgo Oscillator parameters
            luxalgo_length=14,
            luxalgo_smoothing=3,
            luxalgo_overbought_threshold=luxalgo_overbought,
            luxalgo_oversold_threshold=luxalgo_oversold,
            luxalgo_extreme_ob_threshold=90,
            luxalgo_extreme_os_threshold=10,

            # MTF RSI parameters
            mtf_rsi_period=14,
            mtf_rsi_timeframes='15m,1h,4h',
            mtf_min_alignment=0.6,
            mtf_rsi_overbought=70,
            mtf_rsi_oversold=30,

            # RSI-EMA Divergence parameters
            rsi_ema_period=21,
            rsi_ema_rsi_period=14,
            rsi_ema_divergence_sensitivity=0.6,
            rsi_ema_min_divergence_strength=0.7,

            # Squeeze Momentum parameters
            squeeze_bb_length=20,
            squeeze_bb_mult=2.0,
            squeeze_kc_length=20,
            squeeze_kc_mult=1.5,
            squeeze_momentum_length=12,
            squeeze_require_release=True,
            squeeze_momentum_threshold=0.1,

            # Oscillator confluence parameters
            bull_confluence_threshold=0.65,
            bear_confluence_threshold=0.65,
            luxalgo_weight=0.4,
            mtf_rsi_weight=0.3,
            divergence_weight=0.2,
            squeeze_weight=0.1,

            # Mean reversion zone parameters
            zone_validation_enabled=True,
            zone_lookback_periods=50,
            zone_multiplier=1.5,
            require_zone_touch=True,
            min_zone_distance=min_zone_distance,
            max_zone_age=100,
            zone_confidence_boost=0.1,

            # Market regime parameters
            market_regime_enabled=True,
            disable_in_strong_trend=True,
            trend_strength_threshold=0.7,
            volatility_period=20,
            trend_period=50,
            ranging_threshold=0.3,

            # MTF analysis parameters
            mtf_analysis_enabled=True,
            mtf_timeframes_list='15m,1h,4h',
            mtf_min_alignment_score=0.6,
            require_higher_tf_confluence=True,
            mtf_confidence_boost_full_alignment=0.2,

            # Signal quality parameters
            min_confidence=0.6,
            min_risk_reward=1.5,
            max_signals_per_day=5,
            min_signal_spacing_hours=4,

            # Risk management parameters
            stop_loss_pips=fallback_sl,
            take_profit_pips=fallback_tp,
            risk_reward_ratio=fallback_tp / fallback_sl,
            position_size_multiplier=0.8,
            max_drawdown_threshold=0.05,
            trail_stop_enabled=True,

            # Performance metrics (defaults for fallback)
            performance_score=0.0,
            win_rate=0.65,
            avg_profit_pips=15.0,
            max_consecutive_losses=3,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            confluence_accuracy=0.7,
            divergence_success_rate=0.6,
            zone_touch_accuracy=0.8,
            regime_filter_effectiveness=0.75,

            last_optimized=datetime.now() - timedelta(days=999),  # Very old
            market_conditions=market_conditions
        )


def get_optimal_parameter_service() -> OptimalParameterService:
    """Get singleton instance of OptimalParameterService"""
    if not hasattr(get_optimal_parameter_service, '_instance'):
        get_optimal_parameter_service._instance = OptimalParameterService()
    return get_optimal_parameter_service._instance


# Convenience functions for easy integration
def get_epic_optimal_parameters(epic: str, market_conditions: Optional[MarketConditions] = None) -> OptimalParameters:
    """Convenience function to get optimal parameters for an epic"""
    service = get_optimal_parameter_service()
    return service.get_epic_parameters(epic, market_conditions)


def get_epic_ema_config(epic: str) -> Dict[str, int]:
    """Get EMA periods for epic in format compatible with existing strategy"""
    params = get_epic_optimal_parameters(epic)
    
    # Convert ema_config name to actual periods
    # This integrates with existing configdata system
    try:
        from configdata import config as configdata_config
        ema_configs = getattr(configdata_config.strategies, 'EMA_STRATEGY_CONFIG', {})
        
        if params.ema_config in ema_configs:
            config_data = ema_configs[params.ema_config]
            return {
                'short': config_data.get('short', 21),
                'long': config_data.get('long', 50), 
                'trend': config_data.get('trend', 200)
            }
    except ImportError:
        pass
    
    # Fallback to default periods
    return {'short': 21, 'long': 50, 'trend': 200}


# Zero-Lag Strategy Convenience Functions
def get_zerolag_optimal_parameters(epic: str, market_conditions: Optional[MarketConditions] = None) -> ZeroLagOptimalParameters:
    """Convenience function to get optimal zero-lag parameters for an epic"""
    service = get_optimal_parameter_service()
    return service.get_zerolag_parameters(epic, market_conditions)


def get_epic_zerolag_config(epic: str) -> Dict[str, any]:
    """Get zero-lag configuration for epic in format compatible with existing strategy"""
    params = get_zerolag_optimal_parameters(epic)
    
    return {
        'zl_length': params.zl_length,
        'band_multiplier': params.band_multiplier,
        'confidence_threshold': params.confidence_threshold,
        'bb_length': params.bb_length,
        'bb_mult': params.bb_mult,
        'kc_length': params.kc_length,
        'kc_mult': params.kc_mult,
        'smart_money_enabled': params.smart_money_enabled,
        'mtf_validation_enabled': params.mtf_validation_enabled,
        'stop_loss_pips': params.stop_loss_pips,
        'take_profit_pips': params.take_profit_pips
    }


def get_all_optimized_zerolag_epics() -> List[str]:
    """Get list of all epics that have zero-lag optimization data"""
    service = get_optimal_parameter_service()
    all_params = service.get_all_zerolag_parameters()
    return list(all_params.keys())


def is_epic_zerolag_optimized(epic: str) -> bool:
    """Check if an epic has zero-lag optimization data available"""
    try:
        params = get_zerolag_optimal_parameters(epic)
        # Check if parameters are from optimization (not fallback)
        return params.performance_score > 0 and params.last_optimized > datetime.now() - timedelta(days=365)
    except Exception:
        return False


# =============================================================================
# MACD Convenience Functions
# =============================================================================

def get_macd_optimal_parameters(epic: str, timeframe: str = '15m', market_conditions: Optional[MarketConditions] = None) -> MACDOptimalParameters:
    """Get optimal MACD parameters for epic (convenience function)"""
    service = get_optimal_parameter_service()
    return service.get_macd_epic_parameters(epic, timeframe, market_conditions)


def get_epic_macd_config(epic: str, timeframe: str = '15m') -> Dict[str, any]:
    """Get MACD configuration for epic in format compatible with existing strategy"""
    params = get_macd_optimal_parameters(epic, timeframe)
    
    return {
        'fast_ema': params.fast_ema,
        'slow_ema': params.slow_ema,
        'signal_ema': params.signal_ema,
        'confidence_threshold': params.confidence_threshold,
        'timeframe': params.timeframe,
        'histogram_threshold': params.histogram_threshold,
        'zero_line_filter': params.zero_line_filter,
        'rsi_filter_enabled': params.rsi_filter_enabled,
        'momentum_confirmation': params.momentum_confirmation,
        'mtf_enabled': params.mtf_enabled,
        'mtf_timeframes': params.mtf_timeframes,
        'smart_money_enabled': params.smart_money_enabled,
        'stop_loss_pips': params.stop_loss_pips,
        'take_profit_pips': params.take_profit_pips,
        'risk_reward_ratio': params.risk_reward_ratio,
        'win_rate': params.win_rate,
        'performance_score': params.performance_score
    }


def get_all_optimized_macd_epics() -> List[str]:
    """Get list of all epics that have MACD optimization data"""
    service = get_optimal_parameter_service()
    all_params = service.get_all_macd_epic_parameters()
    return list(all_params.keys())


def is_epic_macd_optimized(epic: str, timeframe: str = '15m') -> bool:
    """Check if an epic has MACD optimization data available for specific timeframe"""
    try:
        params = get_macd_optimal_parameters(epic, timeframe)
        # Check if parameters are from optimization (not fallback)
        return params.performance_score > 0 and params.last_optimized > datetime.now() - timedelta(days=365)
    except Exception:
        return False


def get_macd_optimization_status() -> Dict[str, any]:
    """Get comprehensive MACD optimization status across all epics"""
    try:
        from forex_scanner import config
        configured_epics = set(config.EPIC_LIST)
        optimized_epics = set(get_all_optimized_macd_epics())
        
        missing_epics = configured_epics - optimized_epics
        extra_epics = optimized_epics - configured_epics
        
        return {
            'total_configured': len(configured_epics),
            'total_optimized': len(optimized_epics),
            'optimization_coverage': len(optimized_epics) / len(configured_epics) * 100 if configured_epics else 0,
            'missing_epics': list(missing_epics),
            'extra_epics': list(extra_epics),
            'ready_for_production': len(missing_epics) == 0,
            'system_type': 'MACD'
        }
    except Exception as e:
        return {
            'error': str(e),
            'system_type': 'MACD'
        }


# =============================================================================
# SMC Convenience Functions
# =============================================================================

def get_smc_optimal_parameters(epic: str, market_conditions: Optional[MarketConditions] = None) -> SMCOptimalParameters:
    """Get optimal SMC parameters for epic (convenience function)"""
    service = get_optimal_parameter_service()
    return service.get_smc_epic_parameters(epic, market_conditions)


def get_epic_smc_config(epic: str) -> Dict[str, any]:
    """Get SMC configuration for epic in format compatible with existing strategy"""
    params = get_smc_optimal_parameters(epic)
    
    return {
        'smc_config': params.smc_config,
        'confidence_threshold': params.confidence_threshold,
        'timeframe': params.timeframe,
        'use_smart_money': params.use_smart_money,
        
        # Market Structure
        'swing_length': params.swing_length,
        'structure_confirmation': params.structure_confirmation,
        'bos_threshold': params.bos_threshold,
        'choch_threshold': params.choch_threshold,
        
        # Order Blocks
        'order_block_length': params.order_block_length,
        'order_block_volume_factor': params.order_block_volume_factor,
        'order_block_buffer': params.order_block_buffer,
        'max_order_blocks': params.max_order_blocks,
        
        # Fair Value Gaps
        'fvg_min_size': params.fvg_min_size,
        'fvg_max_age': params.fvg_max_age,
        'fvg_fill_threshold': params.fvg_fill_threshold,
        
        # Supply/Demand Zones
        'zone_min_touches': params.zone_min_touches,
        'zone_max_age': params.zone_max_age,
        'zone_strength_factor': params.zone_strength_factor,
        
        # Signal Generation
        'confluence_required': params.confluence_required,
        'min_risk_reward': params.min_risk_reward,
        'max_distance_to_zone': params.max_distance_to_zone,
        'min_signal_confidence': params.min_signal_confidence,
        
        # Multi-timeframe
        'use_higher_tf': params.use_higher_tf,
        'higher_tf_multiplier': params.higher_tf_multiplier,
        'mtf_confluence_weight': params.mtf_confluence_weight,
        
        # Risk Management
        'stop_loss_pips': params.stop_loss_pips,
        'take_profit_pips': params.take_profit_pips,
        'risk_reward_ratio': params.risk_reward_ratio
    }


def get_all_optimized_smc_epics() -> List[str]:
    """Get list of all epics that have SMC optimization data"""
    try:
        db_manager = DatabaseManager()
        with db_manager.get_db_connection() as conn:
            query = "SELECT DISTINCT epic FROM smc_best_parameters ORDER BY epic"
            with conn.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                return [row[0] for row in results] if results else []
    except Exception:
        return []


def is_epic_smc_optimized(epic: str) -> bool:
    """Check if an epic has SMC optimization data available"""
    try:
        params = get_smc_optimal_parameters(epic)
        # Check if parameters are from optimization (not fallback)
        return params.performance_score > 0 and params.last_optimized > datetime.now() - timedelta(days=365)
    except Exception:
        return False


def get_smc_system_readiness() -> Dict[str, any]:
    """Check if SMC optimization system is ready for production"""
    try:
        configured_epics = set(config.EPIC_LIST)
        optimized_epics = set(get_all_optimized_smc_epics())
        
        missing_epics = configured_epics - optimized_epics
        extra_epics = optimized_epics - configured_epics
        
        return {
            'total_configured': len(configured_epics),
            'total_optimized': len(optimized_epics),
            'optimization_coverage': len(optimized_epics) / len(configured_epics) * 100 if configured_epics else 0,
            'missing_epics': list(missing_epics),
            'extra_epics': list(extra_epics),
            'ready_for_production': len(missing_epics) == 0,
            'system_type': 'SMC'
        }
    except Exception as e:
        return {
            'error': str(e),
            'system_type': 'SMC'
        }


# =============================================================================
# Mean Reversion Strategy Functions
# =============================================================================
def get_mean_reversion_optimal_parameters(epic: str, timeframe: str = '15m', market_conditions: Optional[MarketConditions] = None) -> MeanReversionOptimalParameters:
    """Get optimal Mean Reversion parameters for epic (convenience function)"""
    service = get_optimal_parameter_service()
    return service.get_mean_reversion_epic_parameters(epic, timeframe, market_conditions)


def get_epic_mean_reversion_config(epic: str, timeframe: str = '15m') -> Dict[str, any]:
    """Get Mean Reversion configuration for epic in format compatible with existing strategy"""
    params = get_mean_reversion_optimal_parameters(epic, timeframe)

    return {
        # Core parameters
        'confidence_threshold': params.confidence_threshold,
        'timeframe': params.timeframe,

        # LuxAlgo Oscillator parameters
        'luxalgo_length': params.luxalgo_length,
        'luxalgo_smoothing': params.luxalgo_smoothing,
        'luxalgo_overbought_threshold': params.luxalgo_overbought_threshold,
        'luxalgo_oversold_threshold': params.luxalgo_oversold_threshold,
        'luxalgo_extreme_ob_threshold': params.luxalgo_extreme_ob_threshold,
        'luxalgo_extreme_os_threshold': params.luxalgo_extreme_os_threshold,

        # Multi-timeframe RSI parameters
        'mtf_rsi_period': params.mtf_rsi_period,
        'mtf_rsi_timeframes': params.mtf_rsi_timeframes,
        'mtf_min_alignment': params.mtf_min_alignment,
        'mtf_rsi_overbought': params.mtf_rsi_overbought,
        'mtf_rsi_oversold': params.mtf_rsi_oversold,

        # RSI-EMA Divergence parameters
        'rsi_ema_period': params.rsi_ema_period,
        'rsi_ema_rsi_period': params.rsi_ema_rsi_period,
        'rsi_ema_divergence_sensitivity': params.rsi_ema_divergence_sensitivity,
        'rsi_ema_min_divergence_strength': params.rsi_ema_min_divergence_strength,

        # Squeeze Momentum parameters
        'squeeze_bb_length': params.squeeze_bb_length,
        'squeeze_bb_mult': params.squeeze_bb_mult,
        'squeeze_kc_length': params.squeeze_kc_length,
        'squeeze_kc_mult': params.squeeze_kc_mult,
        'squeeze_momentum_length': params.squeeze_momentum_length,
        'squeeze_require_release': params.squeeze_require_release,
        'squeeze_momentum_threshold': params.squeeze_momentum_threshold,

        # Oscillator confluence parameters
        'bull_confluence_threshold': params.bull_confluence_threshold,
        'bear_confluence_threshold': params.bear_confluence_threshold,
        'luxalgo_weight': params.luxalgo_weight,
        'mtf_rsi_weight': params.mtf_rsi_weight,
        'divergence_weight': params.divergence_weight,
        'squeeze_weight': params.squeeze_weight,

        # Mean reversion zone parameters
        'zone_validation_enabled': params.zone_validation_enabled,
        'zone_lookback_periods': params.zone_lookback_periods,
        'zone_multiplier': params.zone_multiplier,
        'require_zone_touch': params.require_zone_touch,
        'min_zone_distance': params.min_zone_distance,
        'max_zone_age': params.max_zone_age,
        'zone_confidence_boost': params.zone_confidence_boost,

        # Market regime parameters
        'market_regime_enabled': params.market_regime_enabled,
        'disable_in_strong_trend': params.disable_in_strong_trend,
        'trend_strength_threshold': params.trend_strength_threshold,
        'volatility_period': params.volatility_period,
        'trend_period': params.trend_period,
        'ranging_threshold': params.ranging_threshold,

        # Multi-timeframe analysis parameters
        'mtf_analysis_enabled': params.mtf_analysis_enabled,
        'mtf_timeframes_list': params.mtf_timeframes_list,
        'mtf_min_alignment_score': params.mtf_min_alignment_score,
        'require_higher_tf_confluence': params.require_higher_tf_confluence,
        'mtf_confidence_boost_full_alignment': params.mtf_confidence_boost_full_alignment,

        # Signal quality parameters
        'min_confidence': params.min_confidence,
        'min_risk_reward': params.min_risk_reward,
        'max_signals_per_day': params.max_signals_per_day,
        'min_signal_spacing_hours': params.min_signal_spacing_hours,

        # Risk management
        'stop_loss_pips': params.stop_loss_pips,
        'take_profit_pips': params.take_profit_pips,
        'risk_reward_ratio': params.risk_reward_ratio,
        'position_size_multiplier': params.position_size_multiplier,
        'max_drawdown_threshold': params.max_drawdown_threshold,
        'trail_stop_enabled': params.trail_stop_enabled,

        # Performance metrics
        'performance_score': params.performance_score,
        'win_rate': params.win_rate,
        'avg_profit_pips': params.avg_profit_pips,
        'max_consecutive_losses': params.max_consecutive_losses,
        'profit_factor': params.profit_factor,
        'sharpe_ratio': params.sharpe_ratio,
        'confluence_accuracy': params.confluence_accuracy,
        'divergence_success_rate': params.divergence_success_rate,
        'zone_touch_accuracy': params.zone_touch_accuracy,
        'regime_filter_effectiveness': params.regime_filter_effectiveness
    }


def get_all_optimized_mean_reversion_epics() -> List[str]:
    """Get list of all epics that have Mean Reversion optimization data"""
    try:
        service = get_optimal_parameter_service()
        with service.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT epic
                    FROM mean_reversion_best_parameters
                    WHERE best_composite_score > 0
                    ORDER BY epic
                """)
                results = cursor.fetchall()
                return [row[0] for row in results] if results else []
    except Exception:
        return []


def is_epic_mean_reversion_optimized(epic: str, timeframe: str = '15m') -> bool:
    """Check if an epic has Mean Reversion optimization data available for specific timeframe"""
    try:
        params = get_mean_reversion_optimal_parameters(epic, timeframe)
        # Check if parameters are from optimization (not fallback)
        return params.performance_score > 0 and params.last_optimized > datetime.now() - timedelta(days=365)
    except Exception:
        return False


def get_mean_reversion_system_readiness() -> Dict[str, any]:
    """Check if Mean Reversion optimization system is ready for production"""
    try:
        configured_epics = set(config.EPIC_LIST)
        optimized_epics = set(get_all_optimized_mean_reversion_epics())

        missing_epics = configured_epics - optimized_epics
        coverage_ratio = len(optimized_epics & configured_epics) / len(configured_epics) if configured_epics else 0

        return {
            'configured_epics': len(configured_epics),
            'optimized_epics': len(optimized_epics),
            'coverage_ratio': coverage_ratio,
            'missing_epics': list(missing_epics),
            'ready_for_production': len(missing_epics) == 0,
            'system_type': 'Mean Reversion'
        }
    except Exception as e:
        return {
            'error': str(e),
            'system_type': 'Mean Reversion'
        }


# =============================================================================
# Ichimoku Strategy Functions
# =============================================================================
def get_ichimoku_optimal_parameters(epic: str, market_conditions: Optional[MarketConditions] = None) -> IchimokuOptimalParameters:
    """Get optimal Ichimoku parameters for epic (convenience function)"""
    service = get_optimal_parameter_service()
    return service.get_ichimoku_epic_parameters(epic, market_conditions)


def get_epic_ichimoku_config(epic: str) -> Dict[str, any]:
    """Get Ichimoku configuration for epic in format compatible with existing strategy"""
    params = get_ichimoku_optimal_parameters(epic)

    return {
        'tenkan_period': params.tenkan_period,
        'kijun_period': params.kijun_period,
        'senkou_b_period': params.senkou_b_period,
        'chikou_shift': params.chikou_shift,
        'cloud_shift': params.cloud_shift,
        'confidence_threshold': params.confidence_threshold,
        'timeframe': params.timeframe,

        # Validation thresholds
        'cloud_thickness_threshold': params.cloud_thickness_threshold,
        'tk_cross_strength_threshold': params.tk_cross_strength_threshold,
        'chikou_clear_threshold': params.chikou_clear_threshold,
        'cloud_filter_enabled': params.cloud_filter_enabled,
        'chikou_filter_enabled': params.chikou_filter_enabled,
        'tk_filter_enabled': params.tk_filter_enabled,

        # Multi-timeframe
        'mtf_enabled': params.mtf_enabled,
        'mtf_timeframes': params.mtf_timeframes,
        'mtf_min_alignment': params.mtf_min_alignment,
        'mtf_cloud_weight': params.mtf_cloud_weight,
        'mtf_tk_weight': params.mtf_tk_weight,
        'mtf_chikou_weight': params.mtf_chikou_weight,

        # Enhancement options
        'momentum_confluence_enabled': params.momentum_confluence_enabled,
        'smart_money_enabled': params.smart_money_enabled,
        'ema_200_trend_filter': params.ema_200_trend_filter,
        'contradiction_filter_enabled': params.contradiction_filter_enabled,

        # Risk management
        'stop_loss_pips': params.stop_loss_pips,
        'take_profit_pips': params.take_profit_pips,
        'risk_reward_ratio': params.risk_reward_ratio,
    }


def is_epic_ichimoku_optimized(epic: str) -> bool:
    """Check if an epic has Ichimoku optimization data available"""
    try:
        params = get_ichimoku_optimal_parameters(epic)
        # Check if parameters are from optimization (not fallback)
        return params.performance_score > 0 and params.last_optimized > datetime.now() - timedelta(days=365)
    except Exception:
        return False


def get_all_optimized_ichimoku_epics() -> List[str]:
    """Get list of all epics that have Ichimoku optimization data"""
    try:
        service = get_optimal_parameter_service()
        with service.db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT epic
                    FROM ichimoku_best_parameters
                    WHERE best_composite_score > 0
                    ORDER BY epic
                """)
                results = cursor.fetchall()
                return [row[0] for row in results] if results else []
    except Exception:
        return []


def get_ichimoku_system_readiness() -> Dict[str, any]:
    """Check if Ichimoku optimization system is ready for production"""
    try:
        configured_epics = set(config.EPIC_LIST)
        optimized_epics = set(get_all_optimized_ichimoku_epics())

        missing_epics = configured_epics - optimized_epics
        coverage_ratio = len(optimized_epics & configured_epics) / len(configured_epics) if configured_epics else 0

        return {
            'configured_epics': len(configured_epics),
            'optimized_epics': len(optimized_epics),
            'coverage_ratio': coverage_ratio,
            'missing_epics': list(missing_epics),
            'ready_for_production': len(missing_epics) == 0,
            'system_type': 'Ichimoku'
        }
    except Exception as e:
        return {
            'error': str(e),
            'system_type': 'Ichimoku'
        }


if __name__ == "__main__":
    # Test the service
    service = OptimalParameterService()
    
    # Test getting parameters for EURUSD
    eurusd_params = service.get_epic_parameters('CS.D.EURUSD.CEEM.IP')
    print(f"✅ EURUSD Optimal Parameters:")
    print(f"   EMA Config: {eurusd_params.ema_config}")
    print(f"   Confidence: {eurusd_params.confidence_threshold:.0%}")
    print(f"   Timeframe: {eurusd_params.timeframe}")
    print(f"   SL/TP: {eurusd_params.stop_loss_pips:.0f}/{eurusd_params.take_profit_pips:.0f}")
    print(f"   Risk:Reward: 1:{eurusd_params.risk_reward_ratio:.1f}")
    print(f"   Performance Score: {eurusd_params.performance_score:.3f}")
    
    # Test with market conditions
    conditions = MarketConditions(
        volatility_level='high',
        market_regime='trending', 
        session='london'
    )
    
    eurusd_params_conditional = service.get_epic_parameters('CS.D.EURUSD.CEEM.IP', conditions)
    print(f"\n✅ EURUSD with High Volatility Conditions:")
    print(f"   SL/TP: {eurusd_params_conditional.stop_loss_pips:.0f}/{eurusd_params_conditional.take_profit_pips:.0f}")