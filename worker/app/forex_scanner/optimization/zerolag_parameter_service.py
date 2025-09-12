#!/usr/bin/env python3
"""
Zero-Lag Optimal Parameter Service
Retrieves and applies optimized parameters from database
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ZeroLagOptimalConfig:
    """Optimal Zero-Lag configuration for an epic"""
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
    win_rate: float
    net_pips: float
    composite_score: float
    last_updated: datetime

class ZeroLagParameterService:
    """Service for retrieving optimal Zero-Lag parameters"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger('zerolag_param_service')
        self._cache = {}
        self._cache_duration = timedelta(minutes=30)
        self._last_cache_update = None
    
    def get_optimal_parameters(self, epic: str) -> Optional[ZeroLagOptimalConfig]:
        """
        Get optimal parameters for specific epic
        Returns None if no optimization data available
        """
        try:
            # Check cache first
            cache_key = f"zl_optimal_{epic}"
            if self._is_cached(cache_key):
                self.logger.debug(f"🚀 Using cached optimal parameters for {epic}")
                return self._cache[cache_key]
            
            # Query database
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT epic, best_zl_length, best_band_multiplier, best_confidence_threshold,
                       best_timeframe, best_bb_length, best_bb_mult, best_kc_length, best_kc_mult,
                       best_smart_money_enabled, best_mtf_validation_enabled,
                       optimal_stop_loss_pips, optimal_take_profit_pips,
                       best_win_rate, best_net_pips, best_composite_score, last_updated
                FROM zerolag_best_parameters 
                WHERE epic = %s
            """, (epic,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                self.logger.debug(f"📊 No optimal parameters found for {epic}")
                return None
            
            # Create config object
            config = ZeroLagOptimalConfig(
                epic=result[0],
                zl_length=result[1],
                band_multiplier=float(result[2]),
                confidence_threshold=float(result[3]),
                timeframe=result[4],
                bb_length=result[5],
                bb_mult=float(result[6]),
                kc_length=result[7],
                kc_mult=float(result[8]),
                smart_money_enabled=result[9] or False,
                mtf_validation_enabled=result[10] or False,
                stop_loss_pips=float(result[11]),
                take_profit_pips=float(result[12]),
                win_rate=float(result[13] or 0),
                net_pips=float(result[14] or 0),
                composite_score=float(result[15] or 0),
                last_updated=result[16]
            )
            
            # Cache the result
            self._cache[cache_key] = config
            self._last_cache_update = datetime.now()
            
            self.logger.info(f"✅ Loaded optimal parameters for {epic}")
            self.logger.info(f"   ZL Length: {config.zl_length}, Band: {config.band_multiplier}")
            self.logger.info(f"   Confidence: {config.confidence_threshold*100:.1f}%, Win Rate: {config.win_rate*100:.1f}%")
            self.logger.info(f"   Score: {config.composite_score}, Net Pips: {config.net_pips}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error getting optimal parameters for {epic}: {e}")
            return None
    
    def get_all_optimized_epics(self) -> Dict[str, ZeroLagOptimalConfig]:
        """Get all epics that have optimal parameters"""
        try:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT epic, best_zl_length, best_band_multiplier, best_confidence_threshold,
                       best_timeframe, best_bb_length, best_bb_mult, best_kc_length, best_kc_mult,
                       best_smart_money_enabled, best_mtf_validation_enabled,
                       optimal_stop_loss_pips, optimal_take_profit_pips,
                       best_win_rate, best_net_pips, best_composite_score, last_updated
                FROM zerolag_best_parameters 
                ORDER BY best_composite_score DESC
            """)
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            optimized_epics = {}
            for result in results:
                config = ZeroLagOptimalConfig(
                    epic=result[0],
                    zl_length=result[1],
                    band_multiplier=float(result[2]),
                    confidence_threshold=float(result[3]),
                    timeframe=result[4],
                    bb_length=result[5],
                    bb_mult=float(result[6]),
                    kc_length=result[7],
                    kc_mult=float(result[8]),
                    smart_money_enabled=result[9] or False,
                    mtf_validation_enabled=result[10] or False,
                    stop_loss_pips=float(result[11]),
                    take_profit_pips=float(result[12]),
                    win_rate=float(result[13] or 0),
                    net_pips=float(result[14] or 0),
                    composite_score=float(result[15] or 0),
                    last_updated=result[16]
                )
                optimized_epics[result[0]] = config
            
            self.logger.info(f"✅ Loaded {len(optimized_epics)} optimized epics")
            return optimized_epics
            
        except Exception as e:
            self.logger.error(f"❌ Error getting all optimized epics: {e}")
            return {}
    
    def has_optimal_parameters(self, epic: str) -> bool:
        """Check if epic has optimal parameters available"""
        return self.get_optimal_parameters(epic) is not None
    
    def get_fallback_config(self, epic: str) -> Dict:
        """Get fallback configuration when no optimal parameters available"""
        return {
            'zl_length': 50,
            'band_multiplier': 1.5,
            'confidence_threshold': 0.65,
            'timeframe': '15m',
            'bb_length': 20,
            'bb_mult': 2.0,
            'kc_length': 20,
            'kc_mult': 1.5,
            'smart_money_enabled': False,
            'mtf_validation_enabled': False,
            'stop_loss_pips': 20.0,
            'take_profit_pips': 40.0
        }
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self._cache:
            return False
        
        if not self._last_cache_update:
            return False
        
        age = datetime.now() - self._last_cache_update
        return age < self._cache_duration
    
    def clear_cache(self):
        """Clear parameter cache"""
        self._cache.clear()
        self._last_cache_update = None
        self.logger.info("🗑️ Parameter cache cleared")
    
    def print_optimization_status(self):
        """Print status of Zero-Lag optimization"""
        optimized_epics = self.get_all_optimized_epics()
        
        print("\n" + "="*80)
        print("🚀 ZERO-LAG OPTIMIZATION STATUS")
        print("="*80)
        
        if not optimized_epics:
            print("❌ No optimized parameters available")
            print("💡 Run optimization first:")
            print("   docker exec task-worker python forex_scanner/optimization/optimize_zerolag_parameters.py --epic CS.D.USDJPY.MINI.IP --smart-presets")
            return
        
        print(f"✅ {len(optimized_epics)} epics optimized")
        print("\n📊 PERFORMANCE RANKING:")
        
        for i, (epic, config) in enumerate(optimized_epics.items(), 1):
            print(f"{i:2d}. {epic:<25} | Score: {config.composite_score:8.2f} | Win: {config.win_rate*100:5.1f}% | Pips: {config.net_pips:7.1f}")
            print(f"    ZL:{config.zl_length:2d} Band:{config.band_multiplier:4.2f} Conf:{config.confidence_threshold*100:4.1f}% | SL:{config.stop_loss_pips:4.1f} TP:{config.take_profit_pips:4.1f}")
        
        # Show which epics need optimization
        common_epics = [
            'CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.USDJPY.MINI.IP',
            'CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.NZDUSD.MINI.IP',
            'CS.D.EURJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP', 'CS.D.USDCHF.MINI.IP'
        ]
        
        unoptimized = [epic for epic in common_epics if epic not in optimized_epics]
        if unoptimized:
            print(f"\n⚠️  {len(unoptimized)} epics need optimization:")
            for epic in unoptimized:
                print(f"   {epic}")
            
            print("\n💡 Optimization commands:")
            print(f"   # Quick test (12 combinations, 3 days)")
            print(f"   docker exec task-worker python forex_scanner/optimization/optimize_zerolag_parameters.py --epic {unoptimized[0]} --smart-presets")
            print(f"   # Production optimization (144 combinations, 14+ days)")  
            print(f"   docker exec task-worker python forex_scanner/optimization/optimize_zerolag_parameters.py --epic {unoptimized[0]} --super-fast --days 21")
        
        print("="*80)


def get_zerolag_parameter_service():
    """Get Zero-Lag parameter service singleton"""
    try:
        from core.database import DatabaseManager
        import config
        db_manager = DatabaseManager(config.DATABASE_URL)
        return ZeroLagParameterService(db_manager)
    except Exception as e:
        logger.error(f"❌ Failed to create Zero-Lag parameter service: {e}")
        return None


if __name__ == "__main__":
    # CLI interface for testing
    service = get_zerolag_parameter_service()
    if service:
        service.print_optimization_status()
        
        # Test parameter retrieval
        test_epics = ['CS.D.USDJPY.MINI.IP', 'CS.D.AUDJPY.MINI.IP', 'CS.D.EURUSD.CEEM.IP']
        for epic in test_epics:
            config = service.get_optimal_parameters(epic)
            if config:
                print(f"\n✅ {epic} optimal config:")
                print(f"   ZL Length: {config.zl_length}, Band: {config.band_multiplier}")
                print(f"   Confidence: {config.confidence_threshold*100:.1f}%, Score: {config.composite_score}")