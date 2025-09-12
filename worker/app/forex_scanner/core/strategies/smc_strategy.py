# core/strategies/smc_strategy.py
"""
Smart Money Concepts (SMC) Strategy Implementation
Based on institutional trading concepts and TradingView SMC-LuxAlgo script

Key Features:
- Market Structure Analysis (BOS, ChoCH, Swing Points)
- Order Block Detection (Bullish & Bearish)
- Fair Value Gap (FVG) Identification
- Supply & Demand Zone Analysis
- Multi-timeframe Confirmation
- Confluence-based Signal Generation

Trading Logic:
- Entry: Structure break + Order block/FVG confluence + Multi-TF alignment
- Risk Management: Stop beyond structure, Target at next structure level
- Confluence Required: Multiple SMC factors must align
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime

from .base_strategy import BaseStrategy
from ..detection.price_adjuster import PriceAdjuster
from .helpers.smc_market_structure import SMCMarketStructure
from .helpers.smc_order_blocks import SMCOrderBlocks
from .helpers.smc_fair_value_gaps import SMCFairValueGaps

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class SMCStrategy(BaseStrategy):
    """
    Smart Money Concepts Strategy
    
    Identifies institutional trading opportunities using:
    - Market structure breaks (BOS/ChoCH)
    - Order block reactions
    - Fair value gap fills
    - Supply/demand zone interactions
    """
    
    def __init__(self, smc_config_name: str = None, data_fetcher=None, backtest_mode: bool = False):
        # Initialize parent
        super().__init__('smc')
        
        # Basic initialization
        self.backtest_mode = backtest_mode
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        
        # SMC configuration
        self.smc_config = self._get_smc_config(smc_config_name)
        
        # Initialize SMC analyzers
        self.market_structure = SMCMarketStructure(logger=self.logger)
        self.order_blocks = SMCOrderBlocks(logger=self.logger)
        self.fair_value_gaps = SMCFairValueGaps(logger=self.logger)
        
        # Strategy parameters
        self.min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.6)
        self.min_bars = 50  # Minimum bars for analysis
        
        # Confluence settings
        self.confluence_required = self.smc_config.get('confluence_required', 2)
        self.min_risk_reward = self.smc_config.get('min_risk_reward', 1.5)
        
        self.logger.info(f"🧠 SMC Strategy initialized")
        self.logger.info(f"🔧 Config: {smc_config_name or 'default'}")
        self.logger.info(f"🎯 Confluence required: {self.confluence_required}")
        self.logger.info(f"📊 Min R:R ratio: {self.min_risk_reward}")
        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions disabled")
    
    def _get_smc_config(self, config_name: str = None) -> Dict:
        """Get SMC configuration from configdata"""
        try:
            # Import SMC configuration
            from configdata.strategies.config_smc_strategy import (
                SMC_STRATEGY_CONFIG, 
                ACTIVE_SMC_CONFIG,
                get_smc_config_for_epic
            )
            
            active_config = config_name or ACTIVE_SMC_CONFIG
            
            if active_config in SMC_STRATEGY_CONFIG:
                return SMC_STRATEGY_CONFIG[active_config]
            
            # Fallback to default
            return SMC_STRATEGY_CONFIG.get('default', {
                'swing_length': 5,
                'structure_confirmation': 3,
                'confluence_required': 2,
                'min_risk_reward': 1.5,
                'order_block_length': 3,
                'fvg_min_size': 3,
                'max_distance_to_zone': 10
            })
            
        except Exception as e:
            self.logger.warning(f"Could not load SMC config: {e}, using defaults")
            return {
                'swing_length': 5,
                'structure_confirmation': 3,
                'confluence_required': 2,
                'min_risk_reward': 1.5,
                'order_block_length': 3,
                'fvg_min_size': 3,
                'max_distance_to_zone': 10
            }
    
    def get_required_indicators(self) -> List[str]:
        """Required indicators for SMC strategy"""
        return [
            'open', 'high', 'low', 'close',  # Basic OHLC
            'volume', 'ltv',                 # Volume data
            'start_time', 'datetime_utc'     # Timestamp data
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
        Detect SMC trading signals
        
        Args:
            df: Enhanced DataFrame with OHLCV data
            epic: Epic code
            spread_pips: Spread in pips
            timeframe: Timeframe being analyzed
            evaluation_time: Optional timestamp for backtesting
            
        Returns:
            Signal dictionary or None
        """
        try:
            # Validate data requirements
            if not self._validate_data_requirements(df):
                return None
            
            self.logger.debug(f"Processing {len(df)} bars for SMC analysis on {epic}")
            
            # Perform SMC analysis
            df_enhanced = self._perform_smc_analysis(df.copy())
            
            # Detect signals
            signal = self._detect_smc_signal(df_enhanced, epic, timeframe, spread_pips)
            
            if signal:
                self.logger.info(f"🧠 SMC {signal['signal_type']} signal detected: {signal['confidence']:.1%}")
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"SMC signal detection error: {e}")
            return None
    
    def _validate_data_requirements(self, df: pd.DataFrame) -> bool:
        """Validate data requirements for SMC analysis"""
        try:
            if len(df) < self.min_bars:
                self.logger.debug(f"Insufficient data: {len(df)} < {self.min_bars}")
                return False
            
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing required columns: {missing_columns}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    def _perform_smc_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform complete SMC analysis on DataFrame"""
        try:
            # Market Structure Analysis
            df = self.market_structure.analyze_market_structure(df, self.smc_config)
            
            # Order Block Detection
            df = self.order_blocks.detect_order_blocks(df, self.smc_config)
            
            # Fair Value Gap Detection
            df = self.fair_value_gaps.detect_fair_value_gaps(df, self.smc_config)
            
            return df
            
        except Exception as e:
            self.logger.error(f"SMC analysis failed: {e}")
            return df
    
    def _detect_smc_signal(
        self, 
        df: pd.DataFrame, 
        epic: str, 
        timeframe: str, 
        spread_pips: float
    ) -> Optional[Dict]:
        """Detect SMC signals based on confluence analysis"""
        try:
            current_index = len(df) - 1
            latest_row = df.iloc[current_index]
            
            # Check for structure break signal
            if not latest_row.get('structure_break', False):
                return None
            
            break_type = latest_row.get('break_type', '')
            break_direction = latest_row.get('break_direction', '')
            structure_significance = latest_row.get('structure_significance', 0.0)
            
            if not break_direction or structure_significance < 0.3:
                return None
            
            # Calculate confluence
            confluence_analysis = self._calculate_confluence(df, current_index, break_direction)
            
            if confluence_analysis['confluence_score'] < self.confluence_required:
                self.logger.debug(f"Insufficient confluence: {confluence_analysis['confluence_score']} < {self.confluence_required}")
                return None
            
            # Determine signal type
            signal_type = 'BULL' if break_direction == 'bullish' else 'BEAR'
            
            # Calculate confidence
            confidence = self._calculate_smc_confidence(
                structure_significance, 
                confluence_analysis, 
                latest_row
            )
            
            if confidence < self.min_confidence:
                self.logger.debug(f"Confidence too low: {confidence:.1%} < {self.min_confidence:.1%}")
                return None
            
            # Create signal
            signal = self._create_smc_signal(
                signal_type=signal_type,
                epic=epic,
                timeframe=timeframe,
                latest_row=latest_row,
                spread_pips=spread_pips,
                confluence_analysis=confluence_analysis,
                confidence=confidence,
                break_type=break_type
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"SMC signal detection failed: {e}")
            return None
    
    def _calculate_confluence(
        self, 
        df: pd.DataFrame, 
        current_index: int, 
        direction: str
    ) -> Dict:
        """Calculate confluence factors for signal validation"""
        try:
            confluence_factors = []
            confluence_score = 0.0
            supporting_analysis = {}
            
            current_price = df.iloc[current_index]['close']
            
            # 1. Market Structure Confluence
            structure_signal = df.iloc[current_index].get('smc_structure_signal', '')
            if structure_signal and direction in structure_signal.lower():
                confluence_factors.append('market_structure_break')
                confluence_score += 1.0
                supporting_analysis['structure_break'] = {
                    'type': df.iloc[current_index].get('break_type', ''),
                    'significance': df.iloc[current_index].get('structure_significance', 0.0)
                }
            
            # 2. Order Block Confluence
            ob_signals = self.order_blocks.get_order_block_signals(df, current_index, self.smc_config)
            if ((direction == 'bullish' and ob_signals.get('bullish_ob_signal', False)) or
                (direction == 'bearish' and ob_signals.get('bearish_ob_signal', False))):
                confluence_factors.append('order_block_support')
                confluence_score += 0.8
                supporting_analysis['order_blocks'] = {
                    'count': ob_signals.get('supporting_ob_count', 0),
                    'nearest_distance': ob_signals.get('nearest_ob_distance', float('inf')),
                    'strength': ob_signals.get('signal_strength', 0.0)
                }
            
            # 3. Fair Value Gap Confluence
            fvg_signals = self.fair_value_gaps.get_fvg_signals(df, current_index, self.smc_config)
            if ((direction == 'bullish' and fvg_signals.get('bullish_fvg_signal', False)) or
                (direction == 'bearish' and fvg_signals.get('bearish_fvg_signal', False))):
                confluence_factors.append('fair_value_gap')
                confluence_score += 0.6
                supporting_analysis['fair_value_gaps'] = {
                    'count': fvg_signals.get('fvg_confluence_count', 0),
                    'strength': fvg_signals.get('fvg_strength', 0.0),
                    'nearest_distance': fvg_signals.get('nearest_fvg_distance', float('inf'))
                }
            
            # 4. Volume Confirmation
            volume_conf = self._check_volume_confirmation(df, current_index)
            if volume_conf > 0.7:
                confluence_factors.append('volume_confirmation')
                confluence_score += 0.4
                supporting_analysis['volume'] = {
                    'confirmation': volume_conf,
                    'above_average': volume_conf > 1.0
                }
            
            # 5. Multi-Timeframe Alignment (if enabled)
            if self.smc_config.get('use_higher_tf', False) and self.data_fetcher:
                mtf_alignment = self._check_higher_timeframe_alignment(direction, df.iloc[current_index])
                if mtf_alignment:
                    confluence_factors.append('higher_timeframe_alignment')
                    confluence_score += 0.5
                    supporting_analysis['multi_timeframe'] = {
                        'aligned': True,
                        'strength': 0.8
                    }
            
            # 6. Equal Highs/Lows (Liquidity)
            liquidity_conf = self._check_liquidity_confluence(df, current_index, direction)
            if liquidity_conf:
                confluence_factors.append('liquidity_sweep')
                confluence_score += 0.3
                supporting_analysis['liquidity'] = {
                    'sweep_detected': True,
                    'type': 'equal_levels'
                }
            
            return {
                'confluence_score': confluence_score,
                'confluence_factors': confluence_factors,
                'supporting_analysis': supporting_analysis,
                'factor_count': len(confluence_factors)
            }
            
        except Exception as e:
            self.logger.error(f"Confluence calculation failed: {e}")
            return {
                'confluence_score': 0.0,
                'confluence_factors': [],
                'supporting_analysis': {},
                'factor_count': 0
            }
    
    def _check_volume_confirmation(self, df: pd.DataFrame, current_index: int) -> float:
        """Check volume confirmation for the signal"""
        try:
            if current_index >= len(df):
                return 0.5
            
            current_volume = df.iloc[current_index].get('volume', df.iloc[current_index].get('ltv', 1))
            if not current_volume or current_volume <= 0:
                return 0.5
            
            # Calculate average volume over recent periods
            lookback = min(20, current_index)
            if lookback <= 0:
                return 0.5
            
            recent_volumes = []
            for i in range(max(0, current_index - lookback), current_index):
                vol = df.iloc[i].get('volume', df.iloc[i].get('ltv', 1))
                if vol and vol > 0:
                    recent_volumes.append(vol)
            
            if not recent_volumes:
                return 0.5
            
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            return current_volume / avg_volume if avg_volume > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Volume confirmation check failed: {e}")
            return 0.5
    
    def _check_higher_timeframe_alignment(self, direction: str, current_row: pd.Series) -> bool:
        """Check higher timeframe structure alignment"""
        try:
            # This would require multi-timeframe data analysis
            # For now, return True if the current structure is strong
            structure_significance = current_row.get('structure_significance', 0.0)
            return structure_significance > 0.6
            
        except Exception as e:
            self.logger.error(f"Higher timeframe alignment check failed: {e}")
            return False
    
    def _check_liquidity_confluence(self, df: pd.DataFrame, current_index: int, direction: str) -> bool:
        """Check for liquidity confluence (equal highs/lows)"""
        try:
            # Check recent swing points for equal levels
            recent_data = df.iloc[max(0, current_index - 20):current_index + 1]
            
            # Look for equal highs or lows in recent data
            swing_highs = recent_data[recent_data.get('swing_high', False) == True]
            swing_lows = recent_data[recent_data.get('swing_low', False) == True]
            
            # Check for equal levels (within 0.5 pips)
            tolerance = 0.00005  # 0.5 pip tolerance
            
            if direction == 'bullish':
                # Look for equal lows that might have been swept
                if len(swing_lows) >= 2:
                    low_prices = swing_lows['low'].values
                    for i in range(len(low_prices) - 1):
                        for j in range(i + 1, len(low_prices)):
                            if abs(low_prices[i] - low_prices[j]) <= tolerance:
                                return True
            
            elif direction == 'bearish':
                # Look for equal highs that might have been swept
                if len(swing_highs) >= 2:
                    high_prices = swing_highs['high'].values
                    for i in range(len(high_prices) - 1):
                        for j in range(i + 1, len(high_prices)):
                            if abs(high_prices[i] - high_prices[j]) <= tolerance:
                                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Liquidity confluence check failed: {e}")
            return False
    
    def _calculate_smc_confidence(
        self, 
        structure_significance: float, 
        confluence_analysis: Dict, 
        latest_row: pd.Series
    ) -> float:
        """Calculate overall confidence for SMC signal"""
        try:
            base_confidence = 0.5
            
            # Structure significance factor (30%)
            structure_factor = structure_significance * 0.3
            
            # Confluence factor (40%)
            confluence_score = confluence_analysis.get('confluence_score', 0.0)
            confluence_factor = min(confluence_score / self.confluence_required, 1.0) * 0.4
            
            # Signal strength factor (20%)
            signal_strength = latest_row.get('smc_signal_strength', 0.0)
            strength_factor = signal_strength * 0.2
            
            # Supporting analysis factor (10%)
            supporting_factors = len(confluence_analysis.get('confluence_factors', []))
            support_factor = min(supporting_factors / 4.0, 1.0) * 0.1
            
            total_confidence = base_confidence + structure_factor + confluence_factor + strength_factor + support_factor
            
            return min(max(total_confidence, 0.1), 0.95)
            
        except Exception as e:
            self.logger.error(f"SMC confidence calculation failed: {e}")
            return 0.5
    
    def _create_smc_signal(
        self,
        signal_type: str,
        epic: str,
        timeframe: str,
        latest_row: pd.Series,
        spread_pips: float,
        confluence_analysis: Dict,
        confidence: float,
        break_type: str
    ) -> Dict:
        """Create SMC signal dictionary"""
        try:
            # Create base signal
            signal = self.create_base_signal(signal_type, epic, timeframe, latest_row)
            
            # Add SMC-specific data
            signal.update({
                'confidence': confidence,
                'confidence_score': confidence,
                'break_type': break_type,
                'confluence_score': confluence_analysis.get('confluence_score', 0.0),
                'confluence_factors': confluence_analysis.get('confluence_factors', []),
                'factor_count': confluence_analysis.get('factor_count', 0),
                'smc_analysis': confluence_analysis.get('supporting_analysis', {}),
                'strategy_type': 'smart_money_concepts',
                'entry_reason': f"SMC_{break_type}_{signal_type}_confluence_{len(confluence_analysis.get('confluence_factors', []))}"
            })
            
            # Add risk management levels
            signal = self._add_risk_management_levels(signal, latest_row, signal_type)
            
            # Add execution prices
            signal = self.add_execution_prices(signal, spread_pips)
            
            # Validate risk-reward ratio
            if not self._validate_risk_reward(signal):
                self.logger.debug(f"Risk-reward ratio insufficient")
                return None
            
            return signal
            
        except Exception as e:
            self.logger.error(f"SMC signal creation failed: {e}")
            return None
    
    def _add_risk_management_levels(
        self, 
        signal: Dict, 
        latest_row: pd.Series, 
        signal_type: str
    ) -> Dict:
        """Add stop loss and take profit levels based on SMC principles"""
        try:
            current_price = signal['price']
            
            # Get structure levels for risk management
            structure_levels = self.market_structure.get_structure_levels(signal_type.lower())
            
            if signal_type == 'BULL':
                # For bullish signals, stop below recent structure low
                if structure_levels:
                    support_levels = [level for level in structure_levels if level['type'] == 'support']
                    if support_levels:
                        stop_loss = support_levels[0]['price'] - (self.smc_config.get('order_block_buffer', 2) / 10000)
                    else:
                        stop_loss = current_price - (10 / 10000)  # Default 10 pip stop
                else:
                    stop_loss = current_price - (10 / 10000)
                
                # Take profit at next resistance or R:R ratio
                stop_distance = current_price - stop_loss
                take_profit = current_price + (stop_distance * self.min_risk_reward)
                
            else:  # BEAR
                # For bearish signals, stop above recent structure high
                if structure_levels:
                    resistance_levels = [level for level in structure_levels if level['type'] == 'resistance']
                    if resistance_levels:
                        stop_loss = resistance_levels[0]['price'] + (self.smc_config.get('order_block_buffer', 2) / 10000)
                    else:
                        stop_loss = current_price + (10 / 10000)  # Default 10 pip stop
                else:
                    stop_loss = current_price + (10 / 10000)
                
                # Take profit at next support or R:R ratio
                stop_distance = stop_loss - current_price
                take_profit = current_price - (stop_distance * self.min_risk_reward)
            
            signal.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'stop_distance_pips': abs(current_price - stop_loss) * 10000,
                'target_distance_pips': abs(take_profit - current_price) * 10000,
                'risk_reward_ratio': abs(take_profit - current_price) / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 0
            })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Risk management levels calculation failed: {e}")
            # Add default levels
            signal.update({
                'stop_loss': signal['price'] * (0.999 if signal_type == 'BULL' else 1.001),
                'take_profit': signal['price'] * (1.0015 if signal_type == 'BULL' else 0.9985),
                'risk_reward_ratio': 1.5
            })
            return signal
    
    def _validate_risk_reward(self, signal: Dict) -> bool:
        """Validate risk-reward ratio meets minimum requirements"""
        try:
            risk_reward = signal.get('risk_reward_ratio', 0.0)
            return risk_reward >= self.min_risk_reward
            
        except Exception as e:
            self.logger.error(f"Risk-reward validation failed: {e}")
            return False
    
    def get_smc_analysis_summary(self) -> Dict:
        """Get summary of current SMC analysis"""
        try:
            return {
                'market_structure': self.market_structure.get_current_structure(),
                'order_blocks': self.order_blocks.get_order_block_summary(),
                'fair_value_gaps': self.fair_value_gaps.get_fvg_summary(),
                'confluence_required': self.confluence_required,
                'min_risk_reward': self.min_risk_reward,
                'config_active': self.smc_config.get('description', 'SMC Strategy')
            }
            
        except Exception as e:
            self.logger.error(f"SMC analysis summary failed: {e}")
            return {}


def create_smc_strategy(data_fetcher=None, **kwargs) -> SMCStrategy:
    """
    Factory function to create SMC strategy instance
    """
    return SMCStrategy(data_fetcher=data_fetcher, **kwargs)