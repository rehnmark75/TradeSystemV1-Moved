# core/intelligence/dynamic_ema_config_manager.py
"""
Dynamic EMA Configuration Manager
Automatically selects optimal EMA configurations based on market conditions and performance
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    import config
except ImportError:
    from forex_scanner import config


@dataclass
class DynamicEMAConfig:
    """Data class for dynamic EMA configuration"""
    name: str
    short: int
    long: int
    trend: int
    best_volatility_regime: str  # 'low', 'medium', 'high'
    best_trend_strength: str    # 'weak', 'medium', 'strong'
    best_market_regime: str     # 'ranging', 'trending', 'breakout'
    best_session: List[str]     # ['asian', 'london', 'new_york', 'overlap']
    preferred_pairs: List[str]  # Most effective currency pairs
    min_pip_volatility: float   # Minimum daily pip volatility
    max_pip_volatility: float   # Maximum daily pip volatility
    performance_score: float = 0.0  # Historical performance score


class MarketCondition(Enum):
    """Market condition types"""
    RANGING = "ranging"
    TRENDING = "trending"
    BREAKOUT = "breakout"
    VOLATILE = "volatile"
    QUIET = "quiet"


class DynamicEMAConfigManager:
    """
    Manages dynamic EMA configuration selection based on:
    1. Current market conditions (volatility, trend strength, regime)
    2. Historical performance data
    3. Currency pair characteristics
    4. Trading session
    5. Real-time adaptation
    """
    
    def __init__(self, market_intelligence_engine, data_fetcher):
        self.market_intelligence = market_intelligence_engine
        self.data_fetcher = data_fetcher
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced configurations
        self.ema_configurations = self._initialize_enhanced_configs()
        
        # Market condition cache (15-minute expiry)
        self.market_conditions_cache = {}
        self.cache_expiry = timedelta(minutes=15)
        
        # Performance tracking
        self.performance_history = {}
        
        # Configuration selection history
        self.selection_history = {}
        
        self.logger.info("🧠 Dynamic EMA Configuration Manager initialized")
    
    def _initialize_enhanced_configs(self) -> Dict[str, DynamicEMAConfig]:
        """Initialize enhanced EMA configurations with market preferences"""
        
        configs = {
            'scalping': DynamicEMAConfig(
                name='scalping',
                short=5, long=13, trend=50,
                best_volatility_regime='high',
                best_trend_strength='weak',
                best_market_regime='ranging',
                best_session=['london', 'new_york', 'overlap'],
                preferred_pairs=['CS.D.EURUSD.CEEM.IP', 'CS.D.GBPUSD.MINI.IP'],
                min_pip_volatility=15.0,
                max_pip_volatility=80.0
            ),
            
            'default': DynamicEMAConfig(
                name='default',
                short=9, long=21, trend=200,
                best_volatility_regime='medium',
                best_trend_strength='medium',
                best_market_regime='trending',
                best_session=['london', 'new_york'],
                preferred_pairs=['CS.D.EURUSD.CEEM.IP', 'CS.D.USDJPY.MINI.IP'],
                min_pip_volatility=8.0,
                max_pip_volatility=50.0
            ),
            
            'conservative': DynamicEMAConfig(
                name='conservative',
                short=20, long=50, trend=200,
                best_volatility_regime='low',
                best_trend_strength='strong',
                best_market_regime='trending',
                best_session=['london'],
                preferred_pairs=['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
                min_pip_volatility=3.0,
                max_pip_volatility=25.0
            ),
            
            'aggressive': DynamicEMAConfig(
                name='aggressive',
                short=5, long=13, trend=50,
                best_volatility_regime='high',
                best_trend_strength='weak',
                best_market_regime='breakout',
                best_session=['overlap', 'new_york'],
                preferred_pairs=['CS.D.GBPUSD.MINI.IP', 'CS.D.EURUSD.CEEM.IP'],
                min_pip_volatility=20.0,
                max_pip_volatility=100.0
            ),
            
            'news_safe': DynamicEMAConfig(
                name='news_safe',
                short=15, long=30, trend=200,
                best_volatility_regime='high',
                best_trend_strength='strong',
                best_market_regime='breakout',
                best_session=['asian', 'overlap'],
                preferred_pairs=['CS.D.USDJPY.MINI.IP'],
                min_pip_volatility=20.0,
                max_pip_volatility=100.0
            ),
            
            'swing': DynamicEMAConfig(
                name='swing',
                short=25, long=55, trend=200,
                best_volatility_regime='low',
                best_trend_strength='strong',
                best_market_regime='trending',
                best_session=['london', 'new_york'],
                preferred_pairs=['CS.D.AUDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP'],
                min_pip_volatility=5.0,
                max_pip_volatility=30.0
            )
        }
        
        return configs
    
    def select_optimal_ema_config(
        self, 
        epic: str, 
        market_conditions: Dict = None,
        performance_weight: float = 0.3
    ) -> DynamicEMAConfig:
        """
        Select optimal EMA configuration based on conditions and performance
        
        Args:
            epic: Currency pair epic
            market_conditions: Current market conditions (auto-detected if None)
            performance_weight: Weight of historical performance in selection (0.0-1.0)
        """
        
        if market_conditions is None:
            market_conditions = self.analyze_market_conditions(epic)
        
        self.logger.info(f"🎯 Selecting optimal EMA config for {epic}")
        self.logger.info(f"   Market conditions: {market_conditions}")
        
        # Score each configuration
        config_scores = {}
        
        for config_name, config in self.ema_configurations.items():
            score = self._score_configuration(config, epic, market_conditions, performance_weight)
            config_scores[config_name] = score
            
            self.logger.debug(f"   {config_name}: {score:.3f}")
        
        # Select best configuration
        best_config_name = max(config_scores, key=config_scores.get)
        best_config = self.ema_configurations[best_config_name]
        best_score = config_scores[best_config_name]
        
        # Update selection history
        self._update_selection_history(epic, best_config_name, market_conditions)
        
        self.logger.info(f"✅ Selected '{best_config_name}' config (score: {best_score:.3f})")
        self.logger.info(f"   EMAs: {best_config.short}/{best_config.long}/{best_config.trend}")
        
        return best_config
    
    def _score_configuration(
        self, 
        config: DynamicEMAConfig, 
        epic: str, 
        market_conditions: Dict,
        performance_weight: float
    ) -> float:
        """Score a configuration based on market fit and performance"""
        
        score = 0.0
        
        # 1. Volatility regime match (25% weight)
        if market_conditions['volatility_regime'] == config.best_volatility_regime:
            score += 0.25
        elif market_conditions['volatility_regime'] in ['high', 'low'] and config.best_volatility_regime == 'medium':
            score += 0.15  # Medium config is somewhat adaptable
        
        # 2. Trend strength match (20% weight)
        if market_conditions['trend_strength'] == config.best_trend_strength:
            score += 0.20
        elif abs(['weak', 'medium', 'strong'].index(market_conditions['trend_strength']) - 
                ['weak', 'medium', 'strong'].index(config.best_trend_strength)) == 1:
            score += 0.10  # Adjacent strength levels get partial credit
        
        # 3. Market regime match (20% weight)
        if market_conditions['market_regime'] == config.best_market_regime:
            score += 0.20
        
        # 4. Currency pair preference (15% weight)
        if epic in config.preferred_pairs:
            score += 0.15
        elif len(config.preferred_pairs) == 0:  # Universal configuration
            score += 0.10
        
        # 5. Trading session match (10% weight)
        current_session = self._get_current_trading_session()
        if current_session in config.best_session:
            score += 0.10
        
        # 6. Volatility range match (10% weight)
        current_volatility = market_conditions.get('daily_pip_volatility', 20.0)
        if config.min_pip_volatility <= current_volatility <= config.max_pip_volatility:
            score += 0.10
        elif current_volatility < config.min_pip_volatility:
            # Penalty for being below minimum
            score -= 0.05
        
        # 7. Historical performance (weight specified by parameter)
        if epic in self.performance_history and config.name in self.performance_history[epic]:
            perf_data = self.performance_history[epic][config.name]
            performance_score = self._calculate_performance_score(perf_data)
            score += performance_score * performance_weight
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def analyze_market_conditions(self, epic: str) -> Dict:
        """Analyze current market conditions for an epic"""
        
        # Check cache first
        cache_key = f"{epic}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.market_conditions_cache:
            cached_time, conditions = self.market_conditions_cache[cache_key]
            if datetime.now() - cached_time < self.cache_expiry:
                return conditions
        
        try:
            # Get recent data for analysis
            pair = epic.split('.')[-2] if '.' in epic else epic
            df = self.data_fetcher.get_enhanced_data(epic, pair, timeframe='15m')
            
            if df is None or len(df) < 200:
                return self._get_default_market_conditions()
            
            conditions = {
                'volatility_regime': self._analyze_volatility_regime(df),
                'trend_strength': self._analyze_trend_strength(df),
                'market_regime': self._analyze_market_regime(df),
                'daily_pip_volatility': self._calculate_daily_pip_volatility(df),
                'current_session': self._get_current_trading_session(),
                'timestamp': datetime.now()
            }
            
            # Cache the result
            self.market_conditions_cache[cache_key] = (datetime.now(), conditions)
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions for {epic}: {e}")
            return self._get_default_market_conditions()
    
    def _analyze_volatility_regime(self, df: pd.DataFrame) -> str:
        """Analyze volatility regime (low/medium/high)"""
        
        if 'atr_14' in df.columns:
            current_atr = df['atr_14'].iloc[-1]
            atr_sma = df['atr_14'].rolling(50).mean().iloc[-1]
            
            atr_ratio = current_atr / atr_sma if atr_sma > 0 else 1.0
            
            if atr_ratio > 1.3:
                return 'high'
            elif atr_ratio < 0.7:
                return 'low'
            else:
                return 'medium'
        
        # Fallback: use price range analysis
        recent_highs = df['high'].rolling(24).max()
        recent_lows = df['low'].rolling(24).min()
        daily_ranges = (recent_highs - recent_lows) / df['close'] * 10000  # in pips
        
        current_range = daily_ranges.iloc[-1]
        avg_range = daily_ranges.rolling(50).mean().iloc[-1]
        
        if current_range > avg_range * 1.3:
            return 'high'
        elif current_range < avg_range * 0.7:
            return 'low'
        else:
            return 'medium'
    
    def _analyze_trend_strength(self, df: pd.DataFrame) -> str:
        """Analyze trend strength (weak/medium/strong)"""
        
        # Use EMA alignment for trend strength
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_200']):
            ema_9 = df['ema_9'].iloc[-1]
            ema_21 = df['ema_21'].iloc[-1]
            ema_200 = df['ema_200'].iloc[-1]
            
            # Check alignment
            if ema_9 > ema_21 > ema_200:  # Strong uptrend
                alignment_strength = (ema_9 - ema_200) / ema_200 * 10000  # in pips
                if alignment_strength > 100:
                    return 'strong'
                elif alignment_strength > 30:
                    return 'medium'
                else:
                    return 'weak'
            elif ema_9 < ema_21 < ema_200:  # Strong downtrend
                alignment_strength = (ema_200 - ema_9) / ema_200 * 10000  # in pips
                if alignment_strength > 100:
                    return 'strong'
                elif alignment_strength > 30:
                    return 'medium'
                else:
                    return 'weak'
            else:
                return 'weak'  # No clear alignment
        
        # Fallback: use ADX if available
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
            if adx > 25:
                return 'strong'
            elif adx > 15:
                return 'medium'
            else:
                return 'weak'
        
        return 'medium'  # Default
    
    def _analyze_market_regime(self, df: pd.DataFrame) -> str:
        """Analyze market regime (ranging/trending/breakout)"""
        
        # Check for recent breakout
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            current_price = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            # Breakout detection
            if current_price > bb_upper or current_price < bb_lower:
                return 'breakout'
            
            # Range vs trend analysis
            bb_width = (bb_upper - bb_lower) / current_price * 10000  # in pips
            avg_bb_width = ((df['bb_upper'] - df['bb_lower']) / df['close'] * 10000).rolling(50).mean().iloc[-1]
            
            if bb_width < avg_bb_width * 0.8:
                return 'ranging'
            else:
                return 'trending'
        
        # Fallback: use price action analysis
        recent_highs = df['high'].rolling(20).max()
        recent_lows = df['low'].rolling(20).min()
        price_range_ratio = (recent_highs.iloc[-1] - recent_lows.iloc[-1]) / df['close'].iloc[-1]
        
        if price_range_ratio < 0.02:  # Less than 2% range
            return 'ranging'
        else:
            return 'trending'
    
    def _calculate_daily_pip_volatility(self, df: pd.DataFrame) -> float:
        """Calculate average daily pip volatility"""
        
        try:
            # Calculate daily ranges in pips
            daily_ranges = (df['high'] - df['low']) / df['close'] * 10000
            return daily_ranges.rolling(20).mean().iloc[-1]
        except:
            return 20.0  # Default fallback
    
    def _get_current_trading_session(self) -> str:
        """Determine current trading session"""
        
        current_hour_utc = datetime.utcnow().hour
        
        # Trading session hours (UTC)
        if 0 <= current_hour_utc < 6:
            return 'asian'
        elif 6 <= current_hour_utc < 8:
            return 'overlap'  # Asian-London overlap
        elif 8 <= current_hour_utc < 13:
            return 'london'
        elif 13 <= current_hour_utc < 15:
            return 'overlap'  # London-NY overlap
        elif 15 <= current_hour_utc < 22:
            return 'new_york'
        else:
            return 'asian'
    
    def _calculate_performance_score(self, perf_data: Dict) -> float:
        """Calculate performance score from historical data"""
        
        if not perf_data or 'signals' not in perf_data or perf_data['signals'] == 0:
            return 0.0
        
        win_rate = perf_data.get('win_rate', 0.0)
        avg_profit = perf_data.get('avg_profit', 0.0)
        total_signals = perf_data.get('signals', 0)
        
        # Weighted performance score
        score = (win_rate * 0.6 + min(avg_profit / 50.0, 1.0) * 0.3 + min(total_signals / 100.0, 1.0) * 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _get_default_market_conditions(self) -> Dict:
        """Return default market conditions when analysis fails"""
        
        return {
            'volatility_regime': 'medium',
            'trend_strength': 'medium',
            'market_regime': 'trending',
            'daily_pip_volatility': 20.0,
            'current_session': self._get_current_trading_session(),
            'timestamp': datetime.now()
        }
    
    def _update_selection_history(self, epic: str, config_name: str, market_conditions: Dict):
        """Update configuration selection history"""
        
        if epic not in self.selection_history:
            self.selection_history[epic] = []
        
        self.selection_history[epic].append({
            'timestamp': datetime.now(),
            'config_name': config_name,
            'market_conditions': market_conditions.copy()
        })
        
        # Keep only last 100 selections per epic
        if len(self.selection_history[epic]) > 100:
            self.selection_history[epic] = self.selection_history[epic][-100:]
    
    def update_performance_data(self, epic: str, config_name: str, signal_result: Dict):
        """Update performance tracking data"""
        
        # Update in-memory tracking (existing code)
        if epic not in self.performance_history:
            self.performance_history[epic] = {}
        
        if config_name not in self.performance_history[epic]:
            self.performance_history[epic][config_name] = {
                'signals': 0, 'wins': 0, 'total_profit': 0.0,
                'win_rate': 0.0, 'avg_profit': 0.0
            }
        
        perf = self.performance_history[epic][config_name]
        perf['signals'] += 1
        
        if signal_result.get('profitable', False):
            perf['wins'] += 1
        
        profit = signal_result.get('profit_pips', 0.0)
        perf['total_profit'] += profit
        
        # Recalculate metrics
        perf['win_rate'] = perf['wins'] / perf['signals']
        perf['avg_profit'] = perf['total_profit'] / perf['signals']
        
        # ADD DATABASE STORAGE:
        try:
            from core.database import DatabaseManager
            db = DatabaseManager(config.DATABASE_URL)
            
            # Insert into database table
            insert_query = """
            INSERT INTO ema_config_performance 
            (epic, config_name, signal_time, direction, entry_price, exit_price, 
            profit_pips, profitable, confidence, market_conditions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (epic, config_name, signal_time) DO NOTHING
            """
            
            db.execute_query(insert_query, (
                epic,
                config_name,
                signal_result.get('signal_time', datetime.now()),
                signal_result.get('direction', 'UNKNOWN'),
                signal_result.get('entry_price', 0.0),
                signal_result.get('exit_price', 0.0),
                profit,
                signal_result.get('profitable', False),
                signal_result.get('confidence', 0.0),
                json.dumps(signal_result.get('market_conditions', {}))
            ))
            
            self.logger.debug(f"💾 Saved performance data to database for {epic} {config_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save performance data to database: {e}")
    
    def get_configuration_summary(self, epic: str = None) -> Dict:
        """Get summary of configuration usage and performance"""
        
        summary = {
            'total_configurations': len(self.ema_configurations),
            'configurations': {}
        }
        
        for config_name, config in self.ema_configurations.items():
            config_summary = {
                'name': config_name,
                'ema_periods': f"{config.short}/{config.long}/{config.trend}",
                'best_conditions': {
                    'volatility': config.best_volatility_regime,
                    'trend_strength': config.best_trend_strength,
                    'market_regime': config.best_market_regime
                },
                'preferred_pairs': config.preferred_pairs,
                'performance': {}
            }
            
            # Add performance data if available
            if epic and epic in self.performance_history and config_name in self.performance_history[epic]:
                config_summary['performance'] = self.performance_history[epic][config_name].copy()
            
            summary['configurations'][config_name] = config_summary
        
        return summary
    
    def force_config_refresh(self, epic: str):
        """Force refresh of cached market conditions"""
        
        # Clear cache for this epic
        keys_to_remove = [key for key in self.market_conditions_cache.keys() if key.startswith(epic)]
        for key in keys_to_remove:
            del self.market_conditions_cache[key]
        
        self.logger.info(f"🔄 Forced refresh of market conditions cache for {epic}")


# Helper function to integrate with existing EMA strategy
def get_dynamic_ema_config_manager(data_fetcher, market_intelligence_engine=None):
    """Factory function to create dynamic EMA configuration manager"""
    
    if market_intelligence_engine is None:
        # Import here to avoid circular imports
        from ..intelligence.market_intelligence import MarketIntelligenceEngine
        market_intelligence_engine = MarketIntelligenceEngine(data_fetcher)
    
    return DynamicEMAConfigManager(market_intelligence_engine, data_fetcher)