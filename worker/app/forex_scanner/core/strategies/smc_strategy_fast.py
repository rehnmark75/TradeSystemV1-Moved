# core/strategies/smc_strategy_fast.py
"""
Fast Smart Money Concepts (SMC) Strategy Implementation
Optimized for performance while maintaining SMC principles

Key Features:
- Vectorized calculations using pandas/numpy
- Minimal loops and complex analysis
- Focus on core SMC signals: structure breaks and confluence
- Compatible with existing strategy framework
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class SMCStrategyFast(BaseStrategy):
    """
    Fast SMC Strategy - Optimized Implementation
    
    Focuses on key SMC concepts with vectorized calculations:
    - Market structure breaks (BOS/ChoCH)
    - Order block zones (support/resistance)
    - Fair value gaps (price imbalances)
    - Confluence-based signals
    """
    
    def __init__(self, smc_config_name: str = None, data_fetcher=None, backtest_mode: bool = False, pipeline_mode: bool = True):
        # Initialize parent
        super().__init__('smc_fast')
        
        # Basic initialization
        self.backtest_mode = backtest_mode
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        
        # SMC configuration - simplified
        self.smc_config = self._get_smc_config(smc_config_name)
        
        # Strategy parameters
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)
        self.min_bars = 50  # Minimum bars for analysis
        
        # SMC settings - optimized for speed
        self.swing_length = self.smc_config.get('swing_length', 5)
        self.confluence_required = self.smc_config.get('confluence_required', 2)
        self.min_risk_reward = self.smc_config.get('min_risk_reward', 1.5)
        self.fvg_min_size = self.smc_config.get('fvg_min_size', 3) / 10000  # Convert to price

        # Enable/disable expensive features based on pipeline mode
        self.enhanced_validation = pipeline_mode and getattr(config, 'SMC_ENHANCED_VALIDATION', True)
        if not self.enhanced_validation:
            # In basic mode, reduce analysis complexity
            self.confluence_required = max(1, self.confluence_required - 1)  # Lower confluence requirement
            self.swing_length = max(3, self.swing_length - 1)  # Shorter swing detection
        
        self.logger.info(f"🧠 Fast SMC Strategy initialized")
        self.logger.info(f"🔧 Config: {smc_config_name or 'default'}")
        self.logger.info(f"🎯 Confluence required: {self.confluence_required}")
        self.logger.info(f"📊 Min R:R ratio: {self.min_risk_reward}")

        if self.enhanced_validation:
            self.logger.info(f"🔍 Enhanced validation ENABLED - Full SMC analysis")
        else:
            self.logger.info(f"🔧 Enhanced validation DISABLED - Fast SMC testing mode")

        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions disabled")
    
    def _get_smc_config(self, config_name: str = None) -> Dict:
        """Get SMC configuration - simplified"""
        try:
            from configdata.strategies.config_smc_strategy import SMC_STRATEGY_CONFIG, ACTIVE_SMC_CONFIG
            
            active_config = config_name or ACTIVE_SMC_CONFIG
            
            if active_config in SMC_STRATEGY_CONFIG:
                return SMC_STRATEGY_CONFIG[active_config]
            
            # Fallback to default
            return SMC_STRATEGY_CONFIG.get('default', {
                'swing_length': 5,
                'confluence_required': 2,
                'min_risk_reward': 1.5,
                'fvg_min_size': 3
            })
            
        except Exception as e:
            self.logger.warning(f"Could not load SMC config: {e}, using defaults")
            return {
                'swing_length': 5,
                'confluence_required': 2,
                'min_risk_reward': 1.5,
                'fvg_min_size': 3
            }
    
    def get_required_indicators(self) -> List[str]:
        """Required indicators for fast SMC strategy"""
        return [
            'open', 'high', 'low', 'close',  # Basic OHLC
            'volume', 'ltv'                  # Volume data
        ]
    
    def detect_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        spread_pips: float = 1.5,
        timeframe: str = '5m',
        evaluation_time: str = None
    ) -> Optional[Dict]:
        """
        Fast SMC signal detection using vectorized calculations
        """
        try:
            # Validate data requirements
            if len(df) < self.min_bars:
                return None
            
            # Fast SMC analysis
            df_smc = self._fast_smc_analysis(df.copy())

            # Check latest row for signals
            latest_row = df_smc.iloc[-1]

            # Look for structure break signals
            if not latest_row.get('structure_break', False):
                return None

            # Fast confluence calculation (skip expensive analysis in basic mode)
            confluence_score = self._fast_confluence_calculation(df_smc, len(df_smc) - 1)
            
            if confluence_score < self.confluence_required:
                return None
            
            # Determine signal type
            signal_type = 'BULL' if latest_row.get('break_direction') == 'bullish' else 'BEAR'
            
            # Calculate confidence
            confidence = self._calculate_fast_confidence(latest_row, confluence_score)
            
            if confidence < self.min_confidence:
                return None
            
            # Create signal
            signal = self._create_fast_signal(
                signal_type=signal_type,
                epic=epic,
                timeframe=timeframe,
                latest_row=latest_row,
                spread_pips=spread_pips,
                confluence_score=confluence_score,
                confidence=confidence
            )
            
            if signal:
                self.logger.info(f"🧠 Fast SMC {signal_type} signal: {confidence:.1%} confidence, confluence: {confluence_score:.1f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Fast SMC signal detection error: {e}")
            return None
    
    def _fast_smc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fast SMC analysis using vectorized operations
        """
        try:
            # 1. Fast swing point detection using rolling windows
            df = self._detect_swing_points_fast(df)

            # 2. Fast structure break detection
            df = self._detect_structure_breaks_fast(df)

            if self.enhanced_validation:
                # 3. Enhanced analysis: Fair value gaps (expensive in basic mode)
                df = self._detect_fvgs_fast(df)

                # 4. Enhanced analysis: Order block approximation (expensive in basic mode)
                df = self._detect_order_blocks_fast(df)
            else:
                # Basic mode: Skip expensive FVG and order block analysis
                df['fvg_bullish'] = False
                df['fvg_bearish'] = False
                df['order_block_bullish'] = False
                df['order_block_bearish'] = False

            return df
            
        except Exception as e:
            self.logger.error(f"Fast SMC analysis failed: {e}")
            return df
    
    def _detect_swing_points_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast swing point detection using rolling windows"""
        try:
            # Use rolling windows for fast pivot detection
            window = self.swing_length
            
            # Pivot highs: current high > surrounding highs
            df['pivot_high'] = (
                (df['high'] == df['high'].rolling(window*2+1, center=True).max()) &
                (df['high'] > df['high'].shift(1)) &
                (df['high'] > df['high'].shift(-1))
            )
            
            # Pivot lows: current low < surrounding lows
            df['pivot_low'] = (
                (df['low'] == df['low'].rolling(window*2+1, center=True).min()) &
                (df['low'] < df['low'].shift(1)) &
                (df['low'] < df['low'].shift(-1))
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fast swing detection failed: {e}")
            return df
    
    def _detect_structure_breaks_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast structure break detection"""
        try:
            # Initialize columns
            df['structure_break'] = False
            df['break_direction'] = ''
            df['break_type'] = ''
            
            # Get pivot high and low indices
            pivot_highs = df[df['pivot_high']].copy()
            pivot_lows = df[df['pivot_low']].copy()
            
            if len(pivot_highs) >= 2:
                # Check for breaks above recent pivot highs (bullish break)
                recent_high = pivot_highs['high'].iloc[-2:-1].values[0] if len(pivot_highs) >= 2 else 0
                if recent_high > 0:
                    bullish_break = df['high'] > recent_high
                    first_break_idx = bullish_break.idxmax() if bullish_break.any() else None
                    
                    if first_break_idx is not None and first_break_idx == len(df) - 1:
                        df.loc[first_break_idx, 'structure_break'] = True
                        df.loc[first_break_idx, 'break_direction'] = 'bullish'
                        df.loc[first_break_idx, 'break_type'] = 'BOS'
            
            if len(pivot_lows) >= 2:
                # Check for breaks below recent pivot lows (bearish break)
                recent_low = pivot_lows['low'].iloc[-2:-1].values[0] if len(pivot_lows) >= 2 else float('inf')
                if recent_low < float('inf'):
                    bearish_break = df['low'] < recent_low
                    first_break_idx = bearish_break.idxmax() if bearish_break.any() else None
                    
                    if first_break_idx is not None and first_break_idx == len(df) - 1:
                        df.loc[first_break_idx, 'structure_break'] = True
                        df.loc[first_break_idx, 'break_direction'] = 'bearish'
                        df.loc[first_break_idx, 'break_type'] = 'BOS'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fast structure break detection failed: {e}")
            return df
    
    def _detect_fvgs_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast Fair Value Gap detection"""
        try:
            # Initialize columns
            df['fvg_bullish'] = False
            df['fvg_bearish'] = False
            
            # Vectorized FVG detection
            # Bullish FVG: current low > 2 candles ago high
            bullish_fvg = (df['low'] > df['high'].shift(2)) & ((df['low'] - df['high'].shift(2)) >= self.fvg_min_size)
            df['fvg_bullish'] = bullish_fvg
            
            # Bearish FVG: current high < 2 candles ago low  
            bearish_fvg = (df['high'] < df['low'].shift(2)) & ((df['low'].shift(2) - df['high']) >= self.fvg_min_size)
            df['fvg_bearish'] = bearish_fvg
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fast FVG detection failed: {e}")
            return df
    
    def _detect_order_blocks_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast order block approximation using volume and price action"""
        try:
            # Initialize columns
            df['order_block_bullish'] = False
            df['order_block_bearish'] = False
            
            # Volume threshold (simple approach)
            volume_col = 'volume' if 'volume' in df.columns else 'ltv'
            if volume_col in df.columns:
                avg_volume = df[volume_col].rolling(20).mean()
                high_volume = df[volume_col] > (avg_volume * 1.5)
                
                # Strong upward moves with high volume (bullish order blocks)
                strong_up_move = (df['close'] > df['open']) & ((df['close'] - df['open']) / df['open'] > 0.002)  # 0.2% move
                df['order_block_bullish'] = high_volume & strong_up_move
                
                # Strong downward moves with high volume (bearish order blocks)
                strong_down_move = (df['close'] < df['open']) & ((df['open'] - df['close']) / df['open'] > 0.002)  # 0.2% move
                df['order_block_bearish'] = high_volume & strong_down_move
            
            return df
            
        except Exception as e:
            self.logger.error(f"Fast order block detection failed: {e}")
            return df
    
    def _fast_confluence_calculation(self, df: pd.DataFrame, current_index: int) -> float:
        """Fast confluence calculation"""
        try:
            confluence_score = 0.0
            current_row = df.iloc[current_index]
            
            # 1. Structure break (base requirement)
            if current_row.get('structure_break', False):
                confluence_score += 1.0
            
            # 2. FVG confluence (only in enhanced mode)
            if self.enhanced_validation and (current_row.get('fvg_bullish', False) or current_row.get('fvg_bearish', False)):
                confluence_score += 0.6

            # 3. Order block confluence (only in enhanced mode)
            if self.enhanced_validation:
                recent_obs = df.iloc[max(0, current_index-5):current_index+1]
                if recent_obs['order_block_bullish'].any() or recent_obs['order_block_bearish'].any():
                    confluence_score += 0.8
            
            # 4. Volume confluence
            volume_col = 'volume' if 'volume' in df.columns else 'ltv'
            if volume_col in df.columns:
                current_volume = current_row.get(volume_col, 0)
                avg_volume = df[volume_col].iloc[max(0, current_index-20):current_index].mean()
                if current_volume > (avg_volume * 1.5):
                    confluence_score += 0.4
            
            # 5. Multiple timeframe proxy (only in enhanced mode, use EMA alignment if available)
            if self.enhanced_validation and all(col in df.columns for col in ['ema_21', 'ema_50', 'ema_200']):
                ema_21 = current_row.get('ema_21', 0)
                ema_50 = current_row.get('ema_50', 0)
                ema_200 = current_row.get('ema_200', 0)

                # Check EMA alignment
                if (ema_21 > ema_50 > ema_200) or (ema_21 < ema_50 < ema_200):
                    confluence_score += 0.5
            
            return confluence_score
            
        except Exception as e:
            self.logger.error(f"Fast confluence calculation failed: {e}")
            return 0.0
    
    def _calculate_fast_confidence(self, latest_row: pd.Series, confluence_score: float) -> float:
        """Fast confidence calculation"""
        try:
            base_confidence = 0.5
            
            # Confluence factor (main component)
            confluence_factor = min(confluence_score / self.confluence_required, 1.0) * 0.4
            
            # Volume factor
            volume_col = 'volume' if 'volume' in latest_row and latest_row['volume'] > 0 else 'ltv'
            volume_factor = 0.1  # Default
            if volume_col in latest_row:
                # Simple volume boost
                volume_factor = 0.2
            
            # Price action factor
            body_size = abs(latest_row.get('close', 0) - latest_row.get('open', 0))
            range_size = latest_row.get('high', 0) - latest_row.get('low', 0)
            if range_size > 0:
                body_ratio = body_size / range_size
                price_action_factor = min(body_ratio * 0.2, 0.2)
            else:
                price_action_factor = 0.1
            
            total_confidence = base_confidence + confluence_factor + volume_factor + price_action_factor
            
            return min(max(total_confidence, 0.1), 0.95)
            
        except Exception as e:
            self.logger.error(f"Fast confidence calculation failed: {e}")
            return 0.5
    
    def _create_fast_signal(
        self,
        signal_type: str,
        epic: str,
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float,
        confluence_score: float,
        confidence: float
    ) -> Optional[Dict]:
        """Create SMC signal with fast risk management"""
        try:
            # Create base signal
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)
            
            # Add SMC-specific data
            signal.update({
                'confidence': confidence,
                'confidence_score': confidence,
                'break_type': latest_row.get('break_type', 'BOS'),
                'confluence_score': confluence_score,
                'confluence_factors': self._get_confluence_factors(latest_row),
                'strategy_type': 'smart_money_concepts_fast',
                'entry_reason': f"SMC_Fast_{signal_type}_confluence_{confluence_score:.1f}"
            })
            
            # Fast risk management levels
            signal = self._add_fast_risk_management(signal, latest_row, signal_type)
            
            # Add execution prices
            signal = self.add_execution_prices(signal, spread_pips)
            
            # Validate risk-reward ratio
            if signal.get('risk_reward_ratio', 0) < self.min_risk_reward:
                return None
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Fast signal creation failed: {e}")
            return None
    
    def _get_confluence_factors(self, latest_row: pd.Series) -> List[str]:
        """Get confluence factors for the signal"""
        factors = []
        
        if latest_row.get('structure_break', False):
            factors.append('structure_break')
        
        if latest_row.get('fvg_bullish', False) or latest_row.get('fvg_bearish', False):
            factors.append('fair_value_gap')
        
        if latest_row.get('order_block_bullish', False) or latest_row.get('order_block_bearish', False):
            factors.append('order_block')
        
        return factors
    
    def _add_fast_risk_management(self, signal: Dict, latest_row: pd.Series, signal_type: str) -> Dict:
        """Add fast risk management levels"""
        try:
            current_price = signal['price']
            
            # Simple risk management based on recent range
            atr_proxy = latest_row.get('high', current_price) - latest_row.get('low', current_price)
            if atr_proxy <= 0:
                atr_proxy = current_price * 0.001  # 0.1% default
            
            # Stop loss: 1.5 x recent range
            stop_distance = atr_proxy * 1.5
            
            # Take profit: 2x stop distance (for 2:1 R:R minimum)
            target_distance = stop_distance * self.min_risk_reward
            
            if signal_type == 'BULL':
                stop_loss = current_price - stop_distance
                take_profit = current_price + target_distance
            else:  # BEAR
                stop_loss = current_price + stop_distance
                take_profit = current_price - target_distance
            
            signal.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'stop_distance_pips': stop_distance * 10000,
                'target_distance_pips': target_distance * 10000,
                'risk_reward_ratio': self.min_risk_reward
            })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Fast risk management failed: {e}")
            # Add default levels
            signal.update({
                'stop_loss': signal['price'] * (0.999 if signal_type == 'BULL' else 1.001),
                'take_profit': signal['price'] * (1.002 if signal_type == 'BULL' else 0.998),
                'risk_reward_ratio': 2.0
            })
            return signal
    
    def get_smc_analysis_summary(self) -> Dict:
        """Get summary of current SMC analysis for fast strategy"""
        try:
            return {
                'strategy_type': 'smc_fast',
                'confluence_required': self.confluence_required,
                'min_risk_reward': self.min_risk_reward,
                'swing_length': self.swing_length,
                'config_active': self.smc_config.get('description', 'Fast SMC Strategy'),
                'performance_optimized': True
            }
        except Exception as e:
            self.logger.error(f"SMC analysis summary failed: {e}")
            return {'strategy_type': 'smc_fast', 'error': str(e)}


def create_smc_strategy_fast(data_fetcher=None, **kwargs) -> SMCStrategyFast:
    """
    Factory function to create fast SMC strategy instance
    """
    return SMCStrategyFast(data_fetcher=data_fetcher, **kwargs)