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