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
from .helpers.smc_premium_discount import SMCPremiumDiscount

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
    
    def __init__(self, smc_config_name: str = None, data_fetcher=None, backtest_mode: bool = False, 
                 epic: str = None, use_optimized_parameters: bool = True):
        # Initialize parent
        super().__init__('smc')
        
        # Basic initialization
        self.backtest_mode = backtest_mode
        self.price_adjuster = PriceAdjuster()
        self.data_fetcher = data_fetcher
        self.epic = epic
        self.use_optimized_parameters = use_optimized_parameters
        
        # Check optimization availability
        self._optimization_available = self._check_optimization_availability()
        
        # SMC configuration - prioritize optimization results if available
        self.smc_config = self._get_smc_config(smc_config_name)
        
        # Initialize SMC analyzers with data_fetcher for multi-timeframe analysis
        self.market_structure = SMCMarketStructure(logger=self.logger, data_fetcher=data_fetcher)
        self.order_blocks = SMCOrderBlocks(logger=self.logger)
        self.fair_value_gaps = SMCFairValueGaps(logger=self.logger)
        self.premium_discount = SMCPremiumDiscount(logger=self.logger, data_fetcher=data_fetcher)
        
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
        if epic:
            self.logger.info(f"📍 Epic: {epic} | Optimized Parameters: {use_optimized_parameters}")
        if backtest_mode:
            self.logger.info("🔥 BACKTEST MODE: Time restrictions disabled")
    
    def _check_optimization_availability(self) -> bool:
        """Check if SMC optimization system is available"""
        try:
            # Check for our actual SMC database parameter service
            from optimization.smc_database_parameter_service import get_smc_optimal_parameters
            return True
        except ImportError:
            self.logger.debug("SMC optimization system not available")
            return False
    
    def _get_smc_config(self, config_name: str = None) -> Dict:
        """Get SMC configuration - prioritize optimization results if available"""
        
        # Check if we should use optimized parameters
        if (self.use_optimized_parameters and self.epic and 
            hasattr(self, '_optimization_available') and self._optimization_available):
            
            try:
                # Try to get optimized SMC parameters from our database service
                from optimization.smc_database_parameter_service import get_smc_optimal_parameters
                
                optimal_params = get_smc_optimal_parameters(self.epic)
                
                # Get the base configuration and enhance with optimization results
                base_config = self._get_base_smc_config(optimal_params['smc_config'])
                
                # Apply optimized parameters where available
                optimized_config = base_config.copy()
                optimized_config.update({
                    # Core optimization parameters
                    'confidence_level': optimal_params['confidence_level'],
                    'min_confidence': optimal_params['confidence_level'],
                    'min_risk_reward': optimal_params['risk_reward_ratio'],
                    'timeframe': optimal_params['timeframe'],
                    
                    # Risk management from optimization
                    'stop_loss_pips': optimal_params['stop_loss_pips'],
                    'take_profit_pips': optimal_params['take_profit_pips'],
                    'risk_reward_ratio': optimal_params['risk_reward_ratio'],
                    
                    # Performance metadata
                    'expected_win_rate': optimal_params['expected_win_rate'],
                    'expected_profit_factor': optimal_params['expected_profit_factor'],
                    'performance_score': optimal_params['performance_score'],
                    'optimization_source': optimal_params['optimization_source'],
                    'last_optimized': optimal_params['last_optimized']
                })
                
                self.logger.info(f"🎯 Using optimized SMC parameters for {self.epic}")
                self.logger.info(f"   Config: {optimal_params['smc_config']} | Confidence: {optimal_params['confidence_level']}")
                self.logger.info(f"   Expected Win Rate: {optimal_params['expected_win_rate']:.1f}%")
                self.logger.info(f"   Performance Score: {optimal_params['performance_score']:.1f}")
                
                return optimized_config
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get optimized SMC parameters for {self.epic}: {e}")
                # Fall through to static configuration
        
        # Use static configuration from configdata
        try:
            # Import SMC configuration
            from configdata.strategies.config_smc_strategy import (
                SMC_STRATEGY_CONFIG, 
                ACTIVE_SMC_CONFIG,
                get_smc_config_for_epic
            )
            
            active_config = config_name or ACTIVE_SMC_CONFIG
            
            if active_config in SMC_STRATEGY_CONFIG:
                static_config = SMC_STRATEGY_CONFIG[active_config].copy()
                static_config['_optimized'] = False  # Mark as static
                self.logger.info(f"📋 Using STATIC SMC parameters: {active_config} config")
                return static_config
            
            # Fallback to default
            return SMC_STRATEGY_CONFIG.get('default', {
                'swing_length': 5,
                'structure_confirmation': 3,
                'confluence_required': 2,
                'min_risk_reward': 1.5,
                'order_block_length': 3,
                'fvg_min_size': 3,
                'max_distance_to_zone': 10,
                '_optimized': False
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
                'max_distance_to_zone': 10,
                '_optimized': False
            }
    
    def _get_base_smc_config(self, config_name: str = 'default') -> Dict:
        """Get base SMC configuration from the configdata system."""
        try:
            # Import SMC configuration from configdata
            from configdata.strategies.config_smc_strategy import SMC_STRATEGY_CONFIG
            
            # Get the specific configuration
            if config_name in SMC_STRATEGY_CONFIG:
                return SMC_STRATEGY_CONFIG[config_name].copy()
            else:
                # Fallback to default if config name not found
                return SMC_STRATEGY_CONFIG.get('default', self._get_fallback_smc_config())
                
        except ImportError:
            # If configdata.strategies.config_smc_strategy doesn't exist, use fallback
            self.logger.warning("SMC strategy configuration not found in configdata, using fallback")
            return self._get_fallback_smc_config()
    
    def _get_fallback_smc_config(self) -> Dict:
        """Fallback SMC configuration when configdata is unavailable."""
        return {
            # Market Structure Parameters
            'swing_length': 5,
            'structure_confirmation': 3,
            'bos_threshold': 0.5,  # Break of Structure threshold
            'choch_threshold': 0.3,  # Change of Character threshold
            
            # Order Block Parameters
            'order_block_length': 3,
            'order_block_volume_factor': 1.5,
            'order_block_buffer': 2,  # pips
            'max_order_blocks': 5,
            
            # Fair Value Gap Parameters
            'fvg_min_size': 3,  # minimum pip size
            'fvg_max_age': 50,  # maximum age in bars
            'fvg_fill_threshold': 0.7,  # percentage fill required
            
            # Supply/Demand Zone Parameters
            'zone_min_touches': 2,
            'zone_max_age': 100,  # bars
            'zone_strength_factor': 1.2,
            
            # Signal Generation Parameters
            'confluence_required': 2,  # minimum confluence factors
            'min_risk_reward': 1.5,
            'max_distance_to_zone': 10,  # pips
            'min_confidence': 0.6,
            
            # Multi-timeframe Parameters
            'use_higher_tf': True,
            'higher_tf_multiplier': 4,  # 5m -> 20m, 15m -> 1h
            'mtf_confluence_weight': 0.3,
            
            # Risk Management
            'stop_loss_pips': 10,
            'take_profit_pips': 20,
            'risk_reward_ratio': 2.0,
            
            # Metadata
            'description': 'Fallback SMC configuration',
            '_optimized': False
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
            
            # Perform SMC analysis with multi-timeframe validation
            df_enhanced = self._perform_smc_analysis(df.copy(), epic, timeframe)
            
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
    
    def _perform_smc_analysis(self, df: pd.DataFrame, epic: str = None, timeframe: str = None) -> pd.DataFrame:
        """Perform complete SMC analysis on DataFrame"""
        try:
            # Market Structure Analysis with multi-timeframe validation
            df = self.market_structure.analyze_market_structure(df, self.smc_config, epic, timeframe)
            
            # Order Block Detection
            df = self.order_blocks.detect_order_blocks(df, self.smc_config)
            
            # Fair Value Gap Detection
            df = self.fair_value_gaps.detect_fair_value_gaps(df, self.smc_config)
            
            # Premium/Discount Analysis (add as metadata, not DataFrame columns)
            current_price = df.iloc[-1]['close'] if len(df) > 0 else None
            premium_discount_analysis = self.premium_discount.analyze_premium_discount(
                df, self.smc_config, epic, current_price
            )
            
            # Store premium/discount analysis for later use in signal detection
            df.attrs['_premium_discount_analysis'] = premium_discount_analysis
            
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
        """Calculate robust weighted confluence factors for signal validation"""
        try:
            confluence_factors = {}  # Changed to dict for weighted analysis
            supporting_analysis = {}
            current_price = df.iloc[current_index]['close']
            
            # Define confluence factor weights based on institutional significance
            confluence_weights = {
                'market_structure_break': {
                    'base_weight': 0.35,      # Highest weight - core SMC concept
                    'multipliers': {
                        'BOS_LiquiditySweep': 1.3,
                        'ChoCH_LiquiditySweep': 1.2,
                        'BOS': 1.0,
                        'ChoCH': 0.9
                    }
                },
                'premium_discount_alignment': {
                    'base_weight': 0.25,      # High weight - market maker model
                    'multipliers': {
                        'optimal_entry_zones': 1.2,
                        'weekly_daily_alignment': 1.1
                    }
                },
                'order_block_support': {
                    'base_weight': 0.20,      # Medium-high weight
                    'multipliers': {
                        'very_strong': 1.3,
                        'strong': 1.1,
                        'medium': 1.0,
                        'weak': 0.7
                    }
                },
                'higher_timeframe_alignment': {
                    'base_weight': 0.20,      # Medium-high weight
                    'multipliers': {
                        'strong_trend': 1.2,
                        'medium_trend': 1.0,
                        'weak_trend': 0.8
                    }
                },
                'volume_confirmation': {
                    'base_weight': 0.15,      # Medium weight
                    'multipliers': {
                        'institutional_volume': 1.3,  # 2.0x+ average
                        'high_volume': 1.1,           # 1.5x+ average
                        'normal_volume': 1.0          # 1.2x+ average
                    }
                },
                'fair_value_gap': {
                    'base_weight': 0.15,      # Medium weight
                    'multipliers': {
                        'large_gap': 1.2,
                        'medium_gap': 1.0,
                        'small_gap': 0.8
                    }
                },
                'liquidity_sweep': {
                    'base_weight': 0.12,      # Medium-low weight
                    'multipliers': {
                        'equal_highs_lows': 1.2,
                        'swing_levels': 1.0
                    }
                }
            }
            
            # Session-based multipliers
            session_multipliers = self._get_session_multipliers()
            
            # 1. Market Structure Confluence (Enhanced)
            structure_confluence = self._analyze_structure_confluence(df, current_index, direction)
            if structure_confluence['present']:
                factor_key = 'market_structure_break'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply structure-specific multiplier
                break_type = structure_confluence.get('break_type', 'BOS')
                structure_multiplier = confluence_weights[factor_key]['multipliers'].get(break_type, 1.0)
                
                # Apply session multiplier
                session_multiplier = session_multipliers.get('structure_breaks', 1.0)
                
                final_weight = base_weight * structure_multiplier * session_multiplier
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': structure_confluence.get('significance', 0.5),
                    'details': structure_confluence
                }
                supporting_analysis['structure_break'] = structure_confluence
            
            # 2. Premium/Discount Confluence (Enhanced) 
            pd_confluence = self._check_premium_discount_confluence(df, direction)
            if pd_confluence:
                factor_key = 'premium_discount_alignment'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply premium/discount specific multipliers
                pd_multiplier = 1.0
                if pd_confluence.get('optimal_entry_zones', 0) > 1:
                    pd_multiplier = confluence_weights[factor_key]['multipliers']['optimal_entry_zones']
                elif pd_confluence.get('weekly_zone') and pd_confluence.get('daily_zone'):
                    pd_multiplier = confluence_weights[factor_key]['multipliers']['weekly_daily_alignment']
                
                session_multiplier = session_multipliers.get('premium_discount', 1.0)
                final_weight = base_weight * pd_multiplier * session_multiplier
                
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': pd_confluence.get('bias_confidence', 0.5),
                    'details': pd_confluence
                }
                supporting_analysis['premium_discount'] = pd_confluence
            
            # 3. Order Block Confluence (Enhanced)
            ob_confluence = self._analyze_order_block_confluence(df, current_index, direction)
            if ob_confluence['present']:
                factor_key = 'order_block_support'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply order block strength multiplier
                ob_strength = ob_confluence.get('strength', 'medium')
                ob_multiplier = confluence_weights[factor_key]['multipliers'].get(ob_strength, 1.0)
                
                session_multiplier = session_multipliers.get('order_blocks', 1.0)
                final_weight = base_weight * ob_multiplier * session_multiplier
                
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': ob_confluence.get('confidence', 0.5),
                    'details': ob_confluence
                }
                supporting_analysis['order_blocks'] = ob_confluence
            
            # 4. Higher Timeframe Confluence (Enhanced)
            htf_confluence = self._analyze_htf_confluence(df, current_index, direction)
            if htf_confluence['present']:
                factor_key = 'higher_timeframe_alignment'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply trend strength multiplier
                trend_strength = htf_confluence.get('trend_strength', 'medium_trend')
                htf_multiplier = confluence_weights[factor_key]['multipliers'].get(trend_strength, 1.0)
                
                # HTF alignment is less session-dependent
                session_multiplier = 1.0
                final_weight = base_weight * htf_multiplier * session_multiplier
                
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': htf_confluence.get('alignment_confidence', 0.5),
                    'details': htf_confluence
                }
                supporting_analysis['multi_timeframe'] = htf_confluence
            
            # 5. Volume Confluence (Enhanced)
            volume_confluence = self._analyze_volume_confluence(df, current_index)
            if volume_confluence['present']:
                factor_key = 'volume_confirmation'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply volume level multiplier
                volume_level = volume_confluence.get('level', 'normal_volume')
                volume_multiplier = confluence_weights[factor_key]['multipliers'].get(volume_level, 1.0)
                
                session_multiplier = session_multipliers.get('volume', 1.0)
                final_weight = base_weight * volume_multiplier * session_multiplier
                
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': volume_confluence.get('ratio', 1.0) / 3.0,  # Normalize
                    'details': volume_confluence
                }
                supporting_analysis['volume'] = volume_confluence
            
            # 6. Fair Value Gap Confluence (Enhanced)
            fvg_confluence = self._analyze_fvg_confluence(df, current_index, direction)
            if fvg_confluence['present']:
                factor_key = 'fair_value_gap'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply FVG size multiplier
                fvg_size = fvg_confluence.get('size', 'medium_gap')
                fvg_multiplier = confluence_weights[factor_key]['multipliers'].get(fvg_size, 1.0)
                
                session_multiplier = session_multipliers.get('fair_value_gaps', 1.0)
                final_weight = base_weight * fvg_multiplier * session_multiplier
                
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': fvg_confluence.get('strength', 0.5),
                    'details': fvg_confluence
                }
                supporting_analysis['fair_value_gaps'] = fvg_confluence
            
            # 7. Liquidity Confluence (Enhanced)
            liquidity_confluence = self._analyze_liquidity_confluence(df, current_index, direction)
            if liquidity_confluence['present']:
                factor_key = 'liquidity_sweep'
                base_weight = confluence_weights[factor_key]['base_weight']
                
                # Apply liquidity type multiplier
                liquidity_type = liquidity_confluence.get('type', 'swing_levels')
                liquidity_multiplier = confluence_weights[factor_key]['multipliers'].get(liquidity_type, 1.0)
                
                session_multiplier = session_multipliers.get('liquidity', 1.0)
                final_weight = base_weight * liquidity_multiplier * session_multiplier
                
                confluence_factors[factor_key] = {
                    'weight': final_weight,
                    'confidence': liquidity_confluence.get('confidence', 0.5),
                    'details': liquidity_confluence
                }
                supporting_analysis['liquidity'] = liquidity_confluence
            
            # Calculate weighted confluence score
            total_weight = sum(factor['weight'] * factor['confidence'] for factor in confluence_factors.values())
            max_possible_weight = sum(weights['base_weight'] for weights in confluence_weights.values())
            
            # Normalize score (0-1 range, then scale for compatibility)
            normalized_score = total_weight / max_possible_weight if max_possible_weight > 0 else 0.0
            final_confluence_score = normalized_score * 4.0  # Scale to match expected range
            
            # Apply confluence factor synergy bonuses
            synergy_bonus = self._calculate_confluence_synergy(confluence_factors)
            final_confluence_score += synergy_bonus
            
            return {
                'confluence_score': final_confluence_score,
                'confluence_factors': list(confluence_factors.keys()),
                'weighted_factors': confluence_factors,
                'supporting_analysis': supporting_analysis,
                'factor_count': len(confluence_factors),
                'session_multipliers': session_multipliers,
                'synergy_bonus': synergy_bonus,
                'normalization_info': {
                    'raw_score': total_weight,
                    'max_possible': max_possible_weight,
                    'normalized': normalized_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Robust confluence calculation failed: {e}")
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
    
    def _check_premium_discount_confluence(self, df: pd.DataFrame, direction: str) -> Optional[Dict]:
        """Check premium/discount confluence for signal validation"""
        try:
            # Get premium/discount analysis from DataFrame metadata
            pd_analysis = df.attrs.get('_premium_discount_analysis', None)
            
            if not pd_analysis or pd_analysis.get('error'):
                return None
            
            # Get premium/discount signal validation
            pd_signal = self.premium_discount.get_premium_discount_signal(
                pd_analysis, direction, self.smc_config
            )
            
            if not pd_signal:
                return None
            
            # Return confluence information
            return {
                'confluence_score': pd_signal.get('confluence_score', 0.0) * 0.8,  # Scale down for overall confluence
                'market_maker_bias': pd_signal.get('market_maker_bias'),
                'confluence_grade': pd_signal.get('confluence_grade'),
                'optimal_entry_zones': pd_signal.get('optimal_entry_zones', 0),
                'bias_confidence': pd_signal.get('bias_confidence', 0.0),
                'daily_zone': pd_signal.get('daily_zone'),
                'weekly_zone': pd_signal.get('weekly_zone'),
                'session_zone': pd_signal.get('session_zone'),
                'pd_factors': [factor['factor'] for factor in pd_signal.get('premium_discount_factors', [])]
            }
            
        except Exception as e:
            self.logger.error(f"Premium/discount confluence check failed: {e}")
            return None
    
    def _calculate_institutional_risk_levels(
        self, 
        entry_price: float, 
        direction: str, 
        current_row: pd.Series, 
        confluence_info: Dict
    ) -> Optional[Dict]:
        """Calculate risk levels using institutional SMC methodology"""
        try:
            # Get premium/discount analysis for advanced exit strategies
            pd_analysis = df.attrs.get('_premium_discount_analysis', None)
            
            # Get structure levels and order blocks for stop placement
            structure_levels = self.market_structure.get_structure_levels(direction)
            
            if direction.lower() in ['bull', 'bullish', 'buy']:
                return self._calculate_bullish_risk_levels(
                    entry_price, structure_levels, pd_analysis, confluence_info
                )
            else:
                return self._calculate_bearish_risk_levels(
                    entry_price, structure_levels, pd_analysis, confluence_info
                )
                
        except Exception as e:
            self.logger.error(f"Institutional risk calculation failed: {e}")
            return None
    
    def _calculate_bullish_risk_levels(
        self, 
        entry_price: float, 
        structure_levels: List[Dict], 
        pd_analysis: Dict,
        confluence_info: Dict
    ) -> Dict:
        """Calculate bullish trade risk levels with institutional methodology"""
        try:
            # 1. Institutional Stop Loss Placement
            stop_loss = self._calculate_institutional_stop_loss(
                entry_price, 'bullish', structure_levels, confluence_info
            )
            
            # 2. Take Profit Levels (Multiple targets)
            take_profit_levels = self._calculate_institutional_take_profits(
                entry_price, 'bullish', structure_levels, pd_analysis, confluence_info
            )
            primary_take_profit = take_profit_levels[0] if take_profit_levels else entry_price * 1.002
            
            # 3. Calculate metrics
            stop_distance_pips = (entry_price - stop_loss) * 10000
            target_distance_pips = (primary_take_profit - entry_price) * 10000
            risk_reward_ratio = target_distance_pips / stop_distance_pips if stop_distance_pips > 0 else 0
            
            # 4. Trailing Stop Configuration
            trailing_config = self._configure_institutional_trailing_stop('bullish', confluence_info)
            
            # 5. Structure Invalidation Level
            structure_invalidation = self._get_structure_invalidation_level(
                entry_price, 'bullish', structure_levels
            )
            
            # 6. Premium/Discount Exit Strategy
            pd_exit_levels = self._calculate_pd_exit_levels('bullish', pd_analysis) if pd_analysis else {}
            
            return {
                'stop_loss': stop_loss,
                'take_profit': primary_take_profit,
                'take_profit_levels': take_profit_levels,
                'stop_distance_pips': stop_distance_pips,
                'target_distance_pips': target_distance_pips,
                'risk_reward_ratio': risk_reward_ratio,
                'breakeven_level': entry_price + (stop_distance_pips * 0.5 / 10000),  # Move to BE at 50% of risk
                'trailing_config': trailing_config,
                'structure_invalidation': structure_invalidation,
                'pd_exit_levels': pd_exit_levels,
                'rm_type': 'institutional_bullish',
                'confidence': confluence_info.get('confluence_score', 0.5) / 4.0  # Normalize
            }
            
        except Exception as e:
            self.logger.error(f"Bullish risk calculation failed: {e}")
            return self._get_default_risk_levels(entry_price, 'bullish')
    
    def _calculate_bearish_risk_levels(
        self, 
        entry_price: float, 
        structure_levels: List[Dict], 
        pd_analysis: Dict,
        confluence_info: Dict
    ) -> Dict:
        """Calculate bearish trade risk levels with institutional methodology"""
        try:
            # 1. Institutional Stop Loss Placement
            stop_loss = self._calculate_institutional_stop_loss(
                entry_price, 'bearish', structure_levels, confluence_info
            )
            
            # 2. Take Profit Levels (Multiple targets)
            take_profit_levels = self._calculate_institutional_take_profits(
                entry_price, 'bearish', structure_levels, pd_analysis, confluence_info
            )
            primary_take_profit = take_profit_levels[0] if take_profit_levels else entry_price * 0.998
            
            # 3. Calculate metrics
            stop_distance_pips = (stop_loss - entry_price) * 10000
            target_distance_pips = (entry_price - primary_take_profit) * 10000
            risk_reward_ratio = target_distance_pips / stop_distance_pips if stop_distance_pips > 0 else 0
            
            # 4. Trailing Stop Configuration
            trailing_config = self._configure_institutional_trailing_stop('bearish', confluence_info)
            
            # 5. Structure Invalidation Level
            structure_invalidation = self._get_structure_invalidation_level(
                entry_price, 'bearish', structure_levels
            )
            
            # 6. Premium/Discount Exit Strategy
            pd_exit_levels = self._calculate_pd_exit_levels('bearish', pd_analysis) if pd_analysis else {}
            
            return {
                'stop_loss': stop_loss,
                'take_profit': primary_take_profit,
                'take_profit_levels': take_profit_levels,
                'stop_distance_pips': stop_distance_pips,
                'target_distance_pips': target_distance_pips,
                'risk_reward_ratio': risk_reward_ratio,
                'breakeven_level': entry_price - (stop_distance_pips * 0.5 / 10000),  # Move to BE at 50% of risk
                'trailing_config': trailing_config,
                'structure_invalidation': structure_invalidation,
                'pd_exit_levels': pd_exit_levels,
                'rm_type': 'institutional_bearish',
                'confidence': confluence_info.get('confluence_score', 0.5) / 4.0  # Normalize
            }
            
        except Exception as e:
            self.logger.error(f"Bearish risk calculation failed: {e}")
            return self._get_default_risk_levels(entry_price, 'bearish')
    
    def _calculate_institutional_stop_loss(
        self, 
        entry_price: float, 
        direction: str, 
        structure_levels: List[Dict],
        confluence_info: Dict
    ) -> float:
        """Calculate stop loss beyond liquidity levels where institutions hunt retail stops"""
        try:
            confluence_score = confluence_info.get('confluence_score', 0.0)
            
            # Base stop distance based on confluence quality
            if confluence_score >= 3.0:  # High confluence
                base_stop_pips = 8   # Tighter stop for high probability setups
            elif confluence_score >= 2.0:  # Medium confluence
                base_stop_pips = 12  # Standard stop
            else:  # Lower confluence
                base_stop_pips = 15  # Wider stop for lower probability
            
            # Find the nearest structure level in stop direction
            if direction == 'bullish':
                support_levels = [level for level in structure_levels 
                                if level['type'] == 'support' and level['price'] < entry_price]
                
                if support_levels:
                    # Sort by distance from entry
                    support_levels.sort(key=lambda x: entry_price - x['price'])
                    nearest_support = support_levels[0]
                    
                    # Place stop beyond the support level where retail stops congregate
                    institutional_buffer = 3 / 10000  # 3 pips beyond retail levels
                    stop_loss = nearest_support['price'] - institutional_buffer
                    
                    # Ensure minimum distance
                    min_stop = entry_price - (base_stop_pips / 10000)
                    stop_loss = min(stop_loss, min_stop)
                    
                else:
                    # No structure levels, use confluence-based stop
                    stop_loss = entry_price - (base_stop_pips / 10000)
                    
            else:  # bearish
                resistance_levels = [level for level in structure_levels 
                                   if level['type'] == 'resistance' and level['price'] > entry_price]
                
                if resistance_levels:
                    # Sort by distance from entry
                    resistance_levels.sort(key=lambda x: x['price'] - entry_price)
                    nearest_resistance = resistance_levels[0]
                    
                    # Place stop beyond the resistance level where retail stops congregate
                    institutional_buffer = 3 / 10000  # 3 pips beyond retail levels
                    stop_loss = nearest_resistance['price'] + institutional_buffer
                    
                    # Ensure minimum distance
                    max_stop = entry_price + (base_stop_pips / 10000)
                    stop_loss = max(stop_loss, max_stop)
                    
                else:
                    # No structure levels, use confluence-based stop
                    stop_loss = entry_price + (base_stop_pips / 10000)
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Institutional stop loss calculation failed: {e}")
            # Fallback
            multiplier = 0.9985 if direction == 'bullish' else 1.0015
            return entry_price * multiplier
    
    def _calculate_institutional_take_profits(
        self,
        entry_price: float,
        direction: str,
        structure_levels: List[Dict],
        pd_analysis: Dict,
        confluence_info: Dict
    ) -> List[float]:
        """Calculate multiple take profit levels using institutional methodology"""
        try:
            take_profit_levels = []
            
            # 1. Structure-based targets
            structure_targets = self._get_structure_based_targets(entry_price, direction, structure_levels)
            take_profit_levels.extend(structure_targets)
            
            # 2. Premium/Discount targets
            if pd_analysis:
                pd_targets = self._get_premium_discount_targets(entry_price, direction, pd_analysis)
                take_profit_levels.extend(pd_targets)
            
            # 3. R:R based targets
            confluence_score = confluence_info.get('confluence_score', 0.0)
            rr_targets = self._get_risk_reward_targets(entry_price, direction, confluence_score)
            take_profit_levels.extend(rr_targets)
            
            # Remove duplicates and sort
            unique_targets = list(set(take_profit_levels))
            
            if direction == 'bullish':
                unique_targets = [tp for tp in unique_targets if tp > entry_price]
                unique_targets.sort()
            else:
                unique_targets = [tp for tp in unique_targets if tp < entry_price]
                unique_targets.sort(reverse=True)
            
            # Limit to top 3 targets
            return unique_targets[:3] if unique_targets else [self._get_default_target(entry_price, direction)]
            
        except Exception as e:
            self.logger.error(f"Take profit calculation failed: {e}")
            return [self._get_default_target(entry_price, direction)]
    
    def _get_structure_based_targets(self, entry_price: float, direction: str, structure_levels: List[Dict]) -> List[float]:
        """Get take profit targets based on structure levels"""
        try:
            targets = []
            
            if direction == 'bullish':
                resistance_levels = [level for level in structure_levels 
                                   if level['type'] == 'resistance' and level['price'] > entry_price]
                for level in resistance_levels[:2]:  # Use first 2 resistance levels
                    # Target slightly before resistance where institutions distribute
                    target = level['price'] - (2 / 10000)  # 2 pips before resistance
                    targets.append(target)
                    
            else:  # bearish
                support_levels = [level for level in structure_levels 
                                if level['type'] == 'support' and level['price'] < entry_price]
                for level in support_levels[:2]:  # Use first 2 support levels
                    # Target slightly before support where institutions distribute
                    target = level['price'] + (2 / 10000)  # 2 pips before support
                    targets.append(target)
            
            return targets
            
        except Exception:
            return []
    
    def _get_premium_discount_targets(self, entry_price: float, direction: str, pd_analysis: Dict) -> List[float]:
        """Get take profit targets based on premium/discount levels"""
        try:
            targets = []
            
            daily_analysis = pd_analysis.get('daily_analysis', {})
            weekly_analysis = pd_analysis.get('weekly_analysis', {})
            
            # Daily range targets
            if daily_analysis and daily_analysis.get('golden_ratio_levels'):
                golden_levels = daily_analysis['golden_ratio_levels']
                
                if direction == 'bullish':
                    # Target premium levels for sells
                    if 'premium_entry' in golden_levels:
                        targets.append(golden_levels['premium_entry'])
                    if 'fibonacci_618' in golden_levels:
                        targets.append(golden_levels['fibonacci_618'])
                else:
                    # Target discount levels for buys
                    if 'discount_entry' in golden_levels:
                        targets.append(golden_levels['discount_entry'])
                    if 'fibonacci_382' in golden_levels:
                        targets.append(golden_levels['fibonacci_382'])
            
            # Weekly range targets (higher priority)
            if weekly_analysis and weekly_analysis.get('golden_ratio_levels'):
                golden_levels = weekly_analysis['golden_ratio_levels']
                
                if direction == 'bullish':
                    if 'fibonacci_618' in golden_levels:
                        targets.append(golden_levels['fibonacci_618'])
                else:
                    if 'fibonacci_382' in golden_levels:
                        targets.append(golden_levels['fibonacci_382'])
                        
            return targets
            
        except Exception:
            return []
    
    def _get_risk_reward_targets(self, entry_price: float, direction: str, confluence_score: float) -> List[float]:
        """Get R:R based targets adjusted for confluence quality"""
        try:
            targets = []
            
            # Adjust R:R based on confluence quality
            if confluence_score >= 3.0:
                risk_rewards = [2.0, 3.0]  # Conservative targets for high probability
            elif confluence_score >= 2.0:
                risk_rewards = [1.8, 2.5]  # Standard targets
            else:
                risk_rewards = [1.5, 2.0]  # Modest targets for lower probability
            
            # Calculate stop distance (simplified)
            base_stop_distance = 12 / 10000  # 12 pips
            
            for rr in risk_rewards:
                if direction == 'bullish':
                    target = entry_price + (base_stop_distance * rr)
                else:
                    target = entry_price - (base_stop_distance * rr)
                targets.append(target)
            
            return targets
            
        except Exception:
            return []
    
    def _get_default_target(self, entry_price: float, direction: str) -> float:
        """Get default take profit target"""
        multiplier = 1.002 if direction == 'bullish' else 0.998
        return entry_price * multiplier
    
    def _configure_institutional_trailing_stop(self, direction: str, confluence_info: Dict) -> Dict:
        """Configure institutional-style trailing stop management"""
        try:
            confluence_score = confluence_info.get('confluence_score', 0.0)
            
            # Trailing configuration based on confluence quality
            if confluence_score >= 3.0:  # High confluence - let it run
                return {
                    'enabled': True,
                    'breakeven_trigger': 8,    # Move to BE at 8 pips profit
                    'profit_protection_trigger': 15,  # Start protecting profit at 15 pips
                    'trailing_distance': 8,     # Trail 8 pips behind peak
                    'profit_lock_ratio': 0.6   # Lock in 60% of max profit
                }
            elif confluence_score >= 2.0:  # Medium confluence
                return {
                    'enabled': True,
                    'breakeven_trigger': 10,
                    'profit_protection_trigger': 18,
                    'trailing_distance': 10,
                    'profit_lock_ratio': 0.5
                }
            else:  # Lower confluence - take profit quickly
                return {
                    'enabled': True,
                    'breakeven_trigger': 12,
                    'profit_protection_trigger': 20,
                    'trailing_distance': 12,
                    'profit_lock_ratio': 0.4
                }
                
        except Exception:
            return {'enabled': False}
    
    def _get_structure_invalidation_level(self, entry_price: float, direction: str, structure_levels: List[Dict]) -> Optional[float]:
        """Get level where market structure would be invalidated"""
        try:
            if direction == 'bullish':
                support_levels = [level for level in structure_levels if level['type'] == 'support']
                if support_levels:
                    # Structure invalidated if price closes below major support
                    major_support = min(support_levels, key=lambda x: x['price'])
                    return major_support['price'] - (5 / 10000)  # 5 pips below major support
            else:
                resistance_levels = [level for level in structure_levels if level['type'] == 'resistance']
                if resistance_levels:
                    # Structure invalidated if price closes above major resistance
                    major_resistance = max(resistance_levels, key=lambda x: x['price'])
                    return major_resistance['price'] + (5 / 10000)  # 5 pips above major resistance
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pd_exit_levels(self, direction: str, pd_analysis: Dict) -> Dict:
        """Calculate premium/discount exit levels for partial profits"""
        try:
            exit_levels = {}
            
            daily_analysis = pd_analysis.get('daily_analysis', {})
            
            if daily_analysis:
                position_pct = daily_analysis.get('position_percentage', 0.5)
                
                if direction == 'bullish':
                    # Exit partial positions as we move into premium
                    if position_pct > 0.7:  # In premium zone
                        exit_levels['premium_partial_exit'] = True
                        exit_levels['exit_percentage'] = 50  # Take 50% profit
                        exit_levels['exit_reason'] = 'premium_zone_reached'
                        
                else:  # bearish
                    # Exit partial positions as we move into discount
                    if position_pct < 0.3:  # In discount zone
                        exit_levels['discount_partial_exit'] = True
                        exit_levels['exit_percentage'] = 50  # Take 50% profit
                        exit_levels['exit_reason'] = 'discount_zone_reached'
            
            return exit_levels
            
        except Exception:
            return {}
    
    def _calculate_position_sizing(self, confluence_info: Dict, risk_analysis: Dict) -> Dict:
        """Calculate position sizing based on confluence quality and risk"""
        try:
            confluence_score = confluence_info.get('confluence_score', 0.0)
            risk_reward_ratio = risk_analysis.get('risk_reward_ratio', 1.5)
            
            # Base risk percentage
            base_risk_percent = 1.0  # 1% of account
            
            # Confluence multiplier
            if confluence_score >= 3.5:  # Exceptional setup
                confluence_multiplier = 1.5   # Risk 1.5%
            elif confluence_score >= 3.0:  # High confluence
                confluence_multiplier = 1.3   # Risk 1.3%
            elif confluence_score >= 2.0:  # Medium confluence  
                confluence_multiplier = 1.0   # Risk 1.0%
            else:  # Lower confluence
                confluence_multiplier = 0.7   # Risk 0.7%
            
            # R:R adjustment (better R:R allows for slightly larger size)
            rr_multiplier = 1.0
            if risk_reward_ratio >= 2.5:
                rr_multiplier = 1.1
            elif risk_reward_ratio >= 2.0:
                rr_multiplier = 1.05
            
            final_risk_percent = min(base_risk_percent * confluence_multiplier * rr_multiplier, 2.0)  # Cap at 2%
            
            return {
                'max_risk_percent': final_risk_percent,
                'confluence_multiplier': confluence_multiplier,
                'rr_multiplier': rr_multiplier,
                'base_risk': base_risk_percent,
                'reasoning': f'Confluence: {confluence_score:.1f}, R:R: {risk_reward_ratio:.1f}'
            }
            
        except Exception:
            return {'max_risk_percent': 1.0, 'confluence_multiplier': 1.0}
    
    def _get_default_risk_levels(self, entry_price: float, direction: str) -> Dict:
        """Get default risk levels as fallback"""
        stop_multiplier = 0.9985 if direction == 'bullish' else 1.0015
        target_multiplier = 1.003 if direction == 'bullish' else 0.997
        
        stop_loss = entry_price * stop_multiplier
        take_profit = entry_price * target_multiplier
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance_pips': abs(entry_price - stop_loss) * 10000,
            'target_distance_pips': abs(take_profit - entry_price) * 10000,
            'risk_reward_ratio': 2.0,
            'rm_type': 'default',
            'confidence': 0.3
        }
    
    def _apply_fallback_risk_management(self, signal: Dict, signal_type: str) -> Dict:
        """Apply fallback risk management when advanced calculation fails"""
        entry_price = signal['price']
        
        fallback_data = {
            'stop_loss': entry_price * (0.999 if signal_type == 'BULL' else 1.001),
            'take_profit': entry_price * (1.002 if signal_type == 'BULL' else 0.998),
            'risk_reward_ratio': 2.0,
            'stop_distance_pips': 10,
            'target_distance_pips': 20,
            'rm_type': 'fallback'
        }
        
        signal.update(fallback_data)
        return signal
    
    def _passes_signal_filters(
        self, 
        signal: Dict, 
        latest_row: pd.Series, 
        epic: str, 
        timeframe: str
    ) -> bool:
        """Comprehensive signal filtering system - final validation before signal generation"""
        try:
            # Get filter configuration
            filter_config = self.smc_config.get('signal_filters', {})
            current_time = datetime.utcnow()
            
            # 1. Session-based filtering (already handled in confluence, but double-check)
            if not self._passes_session_filter(current_time, filter_config):
                return False
            
            # 2. Market condition filtering
            if not self._passes_market_condition_filter(signal, latest_row, filter_config):
                return False
            
            # 3. Spread condition filtering
            if not self._passes_spread_filter(epic, filter_config):
                return False
            
            # 4. Signal quality filtering
            if not self._passes_signal_quality_filter(signal, filter_config):
                return False
            
            # 5. Anti-overtrading filtering
            if not self._passes_overtrading_filter(epic, current_time, filter_config):
                return False
            
            # 6. Market volatility filtering
            if not self._passes_volatility_filter(latest_row, filter_config):
                return False
            
            # 7. News/event filtering (simplified)
            if not self._passes_news_filter(current_time, filter_config):
                return False
            
            # 8. Multi-timeframe coherence check
            if not self._passes_mtf_coherence_filter(signal, epic, timeframe, filter_config):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Signal filtering failed: {e}")
            # Default to allowing signal if filtering fails
            return True
    
    def _passes_session_filter(self, current_time: datetime, filter_config: Dict) -> bool:
        """Filter signals based on trading session rules"""
        try:
            utc_hour = current_time.hour
            
            # Check if Asian session filtering is enabled
            if filter_config.get('avoid_asian_session', True):
                # Asian session: 22:00-08:00 UTC
                if 22 <= utc_hour or utc_hour <= 8:
                    return False
            
            # Weekend filtering
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            if weekday >= 5:  # Saturday or Sunday
                return False
            
            # Friday late session filtering (avoid holding over weekend)
            if weekday == 4 and utc_hour >= 20:  # Friday after 8 PM UTC
                return False
            
            return True
            
        except Exception:
            return True
    
    def _passes_market_condition_filter(self, signal: Dict, latest_row: pd.Series, filter_config: Dict) -> bool:
        """Filter signals based on market conditions"""
        try:
            # Check for choppy/ranging conditions
            if filter_config.get('avoid_choppy_markets', True):
                # Simple choppiness detection using recent price action
                try:
                    # Look at recent high/low range vs average
                    recent_high = latest_row.get('recent_high', latest_row.get('high', 0))
                    recent_low = latest_row.get('recent_low', latest_row.get('low', 0))
                    current_price = latest_row.get('close', signal.get('price', 0))
                    
                    if recent_high > 0 and recent_low > 0:
                        range_size = recent_high - recent_low
                        price_position = (current_price - recent_low) / range_size if range_size > 0 else 0.5
                        
                        # Avoid signals in middle of range (choppy zone)
                        if filter_config.get('avoid_range_middle', True) and 0.4 <= price_position <= 0.6:
                            return False
                except Exception:
                    pass
            
            # Check confluence quality threshold
            min_confluence_for_filtering = filter_config.get('min_confluence_threshold', 1.5)
            confluence_score = signal.get('confluence_info', {}).get('confluence_score', 0.0)
            
            if confluence_score < min_confluence_for_filtering:
                return False
            
            return True
            
        except Exception:
            return True
    
    def _passes_spread_filter(self, epic: str, filter_config: Dict) -> bool:
        """Filter signals based on spread conditions"""
        try:
            # Maximum allowed spread (in pips)
            max_spread_pips = filter_config.get('max_spread_pips', 3.0)
            
            # This would require real-time spread data from the broker
            # For now, implement basic logic
            
            # Major pairs should have tight spreads
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
            pair_name = epic.upper()
            
            is_major_pair = any(major in pair_name for major in major_pairs)
            
            if is_major_pair:
                # Major pairs should have spreads < 2 pips during active sessions
                current_time = datetime.utcnow()
                utc_hour = current_time.hour
                
                # During London/NY sessions, expect tight spreads
                if 8 <= utc_hour <= 22:  # Active hours
                    # Would normally check actual spread here
                    # For now, assume spreads are acceptable during active hours
                    return True
                else:
                    # Outside active hours, spreads might be wider
                    # More conservative approach
                    return True
            
            return True
            
        except Exception:
            return True
    
    def _passes_signal_quality_filter(self, signal: Dict, filter_config: Dict) -> bool:
        """Filter signals based on signal quality metrics"""
        try:
            # Minimum R:R ratio filter
            min_rr_for_filtering = filter_config.get('min_rr_ratio', 1.2)
            risk_reward_ratio = signal.get('risk_reward_ratio', 0.0)
            
            if risk_reward_ratio < min_rr_for_filtering:
                return False
            
            # Maximum stop distance filter
            max_stop_distance_pips = filter_config.get('max_stop_distance_pips', 25)
            stop_distance_pips = signal.get('stop_distance_pips', 0)
            
            if stop_distance_pips > max_stop_distance_pips:
                return False
            
            # Signal confidence filter
            min_signal_confidence = filter_config.get('min_signal_confidence', 0.4)
            signal_confidence = signal.get('confidence', signal.get('confidence_score', 0.5))
            
            if signal_confidence < min_signal_confidence:
                return False
            
            # Structure significance filter
            min_structure_significance = filter_config.get('min_structure_significance', 0.3)
            structure_significance = signal.get('smc_analysis', {}).get('structure_break', {}).get('significance', 0.5)
            
            if structure_significance < min_structure_significance:
                return False
            
            return True
            
        except Exception:
            return True
    
    def _passes_overtrading_filter(self, epic: str, current_time: datetime, filter_config: Dict) -> bool:
        """Prevent overtrading by limiting signal frequency"""
        try:
            # Maximum signals per hour per epic
            max_signals_per_hour = filter_config.get('max_signals_per_hour', 2)
            
            # This would require tracking recent signals
            # For now, implement basic time-based filtering
            
            # Minimum time between signals (in minutes)
            min_signal_interval_minutes = filter_config.get('min_signal_interval_minutes', 30)
            
            # In a full implementation, you would:
            # 1. Store recent signal timestamps per epic
            # 2. Check if enough time has passed since last signal
            # 3. Implement exponential backoff for failed signals
            
            # For now, basic implementation
            current_minute = current_time.minute
            
            # Avoid signals at exact hour/half-hour boundaries (common retail trap times)
            if filter_config.get('avoid_round_times', True):
                if current_minute in [0, 30]:
                    return False
            
            return True
            
        except Exception:
            return True
    
    def _passes_volatility_filter(self, latest_row: pd.Series, filter_config: Dict) -> bool:
        """Filter signals based on market volatility conditions"""
        try:
            # Get volatility indicators if available
            atr = latest_row.get('atr', 0)
            
            if atr > 0:
                # Minimum volatility required
                min_atr_pips = filter_config.get('min_volatility_pips', 5) / 10000
                if atr < min_atr_pips:
                    return False
                
                # Maximum volatility allowed (avoid news spikes)
                max_atr_pips = filter_config.get('max_volatility_pips', 50) / 10000
                if atr > max_atr_pips:
                    return False
            
            # Check for price gaps (potential news events)
            current_open = latest_row.get('open', 0)
            previous_close = latest_row.get('previous_close', current_open)
            
            if current_open > 0 and previous_close > 0:
                gap_size_pips = abs(current_open - previous_close) * 10000
                max_gap_pips = filter_config.get('max_gap_size_pips', 10)
                
                if gap_size_pips > max_gap_pips:
                    return False
            
            return True
            
        except Exception:
            return True
    
    def _passes_news_filter(self, current_time: datetime, filter_config: Dict) -> bool:
        """Filter signals around major news events"""
        try:
            # Major news times (simplified - in production would use economic calendar)
            utc_hour = current_time.hour
            utc_minute = current_time.minute
            weekday = current_time.weekday()
            
            # Avoid major news times
            if filter_config.get('avoid_news_times', True):
                # US Non-Farm Payrolls (First Friday of month, 8:30 AM EST = 13:30 UTC)
                if weekday == 4 and utc_hour == 13 and 25 <= utc_minute <= 35:
                    return False
                
                # FOMC meetings (typically 2:00 PM EST = 19:00 UTC)
                if utc_hour == 19 and 0 <= utc_minute <= 30:
                    return False
                
                # ECB meetings (typically 7:45 AM EST = 12:45 UTC)
                if utc_hour == 12 and 40 <= utc_minute <= 50:
                    return False
                
                # London/NY market opens (avoid first 15 minutes)
                if (utc_hour == 8 and utc_minute <= 15) or (utc_hour == 13 and utc_minute <= 15):
                    return False
            
            return True
            
        except Exception:
            return True
    
    def _passes_mtf_coherence_filter(
        self, 
        signal: Dict, 
        epic: str, 
        timeframe: str, 
        filter_config: Dict
    ) -> bool:
        """Check multi-timeframe coherence to avoid conflicting signals"""
        try:
            # This is a simplified version
            # In full implementation, would check actual higher timeframe trends
            
            confluence_info = signal.get('confluence_info', {})
            
            # Check if higher timeframe alignment is available and positive
            supporting_analysis = confluence_info.get('supporting_analysis', {})
            htf_analysis = supporting_analysis.get('multi_timeframe', {})
            
            if htf_analysis:
                htf_aligned = htf_analysis.get('aligned', False)
                htf_strength = htf_analysis.get('strength', 0)
                
                # Require minimum HTF alignment for signals
                min_htf_strength = filter_config.get('min_htf_alignment_strength', 0.3)
                
                if not htf_aligned or htf_strength < min_htf_strength:
                    return False
            
            # Check premium/discount alignment
            pd_analysis = supporting_analysis.get('premium_discount', {})
            if pd_analysis:
                market_maker_bias = pd_analysis.get('market_maker_bias', 'neutral')
                signal_direction = signal.get('signal_type', '').lower()
                
                # Ensure signal direction aligns with market maker bias
                if market_maker_bias == 'bullish' and 'bull' not in signal_direction:
                    return False
                elif market_maker_bias == 'bearish' and 'bear' not in signal_direction:
                    return False
            
            return True
            
        except Exception:
            return True
    
    def _get_session_multipliers(self) -> Dict:
        """Get session-based confluence multipliers"""
        try:
            current_time = datetime.utcnow()
            utc_hour = current_time.hour
            
            # Base multipliers for different sessions
            if 13 <= utc_hour <= 16:  # London/NY Overlap
                return {
                    'structure_breaks': self.smc_config.get('overlap_session_boost', 1.5),
                    'premium_discount': 1.3,
                    'order_blocks': 1.2,
                    'volume': 1.4,
                    'fair_value_gaps': 1.2,
                    'liquidity': 1.3
                }
            elif 8 <= utc_hour <= 16:  # London Session
                return {
                    'structure_breaks': self.smc_config.get('london_session_boost', 1.2),
                    'premium_discount': 1.1,
                    'order_blocks': 1.1,
                    'volume': 1.2,
                    'fair_value_gaps': 1.0,
                    'liquidity': 1.1
                }
            elif 13 <= utc_hour <= 22:  # New York Session
                return {
                    'structure_breaks': self.smc_config.get('ny_session_boost', 1.3),
                    'premium_discount': 1.2,
                    'order_blocks': 1.0,
                    'volume': 1.1,
                    'fair_value_gaps': 1.0,
                    'liquidity': 1.2
                }
            else:  # Asian or Off-Hours
                multiplier = 0.7 if self.smc_config.get('avoid_asian_session', True) else 0.8
                return {
                    'structure_breaks': multiplier,
                    'premium_discount': multiplier,
                    'order_blocks': multiplier,
                    'volume': multiplier,
                    'fair_value_gaps': multiplier,
                    'liquidity': multiplier
                }
                
        except Exception:
            return {
                'structure_breaks': 1.0,
                'premium_discount': 1.0,
                'order_blocks': 1.0,
                'volume': 1.0,
                'fair_value_gaps': 1.0,
                'liquidity': 1.0
            }
    
    def _analyze_structure_confluence(self, df: pd.DataFrame, current_index: int, direction: str) -> Dict:
        """Analyze market structure confluence with enhanced details"""
        try:
            current_row = df.iloc[current_index]
            structure_signal = current_row.get('smc_structure_signal', '')
            
            if not structure_signal or direction not in structure_signal.lower():
                return {'present': False}
            
            break_type = current_row.get('break_type', 'BOS')
            significance = current_row.get('structure_significance', 0.5)
            break_direction = current_row.get('break_direction', '')
            
            return {
                'present': True,
                'break_type': break_type,
                'direction': break_direction,
                'significance': significance,
                'liquidity_sweep': 'LiquiditySweep' in break_type,
                'confidence_level': 'high' if significance > 0.7 else 'medium' if significance > 0.5 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Structure confluence analysis failed: {e}")
            return {'present': False}
    
    def _analyze_order_block_confluence(self, df: pd.DataFrame, current_index: int, direction: str) -> Dict:
        """Analyze order block confluence with enhanced details"""
        try:
            ob_signals = self.order_blocks.get_order_block_signals(df, current_index, self.smc_config)
            
            has_signal = ((direction == 'bullish' and ob_signals.get('bullish_ob_signal', False)) or
                         (direction == 'bearish' and ob_signals.get('bearish_ob_signal', False)))
            
            if not has_signal:
                return {'present': False}
            
            signal_strength = ob_signals.get('signal_strength', 0.5)
            supporting_count = ob_signals.get('supporting_ob_count', 0)
            nearest_distance = ob_signals.get('nearest_ob_distance', float('inf'))
            
            # Determine strength category
            if signal_strength > 0.8 and supporting_count > 2:
                strength = 'very_strong'
            elif signal_strength > 0.6 and supporting_count > 1:
                strength = 'strong'
            elif signal_strength > 0.4:
                strength = 'medium'
            else:
                strength = 'weak'
            
            return {
                'present': True,
                'strength': strength,
                'confidence': signal_strength,
                'supporting_count': supporting_count,
                'nearest_distance': nearest_distance,
                'distance_pips': nearest_distance * 10000 if nearest_distance != float('inf') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Order block confluence analysis failed: {e}")
            return {'present': False}
    
    def _analyze_htf_confluence(self, df: pd.DataFrame, current_index: int, direction: str) -> Dict:
        """Analyze higher timeframe confluence with enhanced details"""
        try:
            if not self.smc_config.get('use_higher_tf', False) or not self.data_fetcher:
                return {'present': False}
            
            current_row = df.iloc[current_index]
            mtf_alignment = self._check_higher_timeframe_alignment(direction, current_row)
            
            if not mtf_alignment:
                return {'present': False}
            
            # Try to get additional HTF strength info if available
            # This is a simplified version - in full implementation would analyze actual HTF data
            htf_strength = current_row.get('htf_trend_strength', 0.6)
            
            if htf_strength > 0.8:
                trend_strength = 'strong_trend'
            elif htf_strength > 0.6:
                trend_strength = 'medium_trend'
            else:
                trend_strength = 'weak_trend'
            
            return {
                'present': True,
                'trend_strength': trend_strength,
                'alignment_confidence': htf_strength,
                'direction': direction,
                'trend_score': htf_strength
            }
            
        except Exception as e:
            self.logger.error(f"HTF confluence analysis failed: {e}")
            return {'present': False}
    
    def _analyze_volume_confluence(self, df: pd.DataFrame, current_index: int) -> Dict:
        """Analyze volume confluence with enhanced details"""
        try:
            volume_conf = self._check_volume_confirmation(df, current_index)
            
            if volume_conf <= 1.2:  # Must be at least 1.2x average
                return {'present': False}
            
            # Categorize volume level
            if volume_conf >= 2.0:
                level = 'institutional_volume'
            elif volume_conf >= 1.5:
                level = 'high_volume'
            else:
                level = 'normal_volume'
            
            return {
                'present': True,
                'level': level,
                'ratio': volume_conf,
                'above_average': volume_conf > 1.0,
                'institutional_level': volume_conf >= 2.0
            }
            
        except Exception as e:
            self.logger.error(f"Volume confluence analysis failed: {e}")
            return {'present': False}
    
    def _analyze_fvg_confluence(self, df: pd.DataFrame, current_index: int, direction: str) -> Dict:
        """Analyze Fair Value Gap confluence with enhanced details"""
        try:
            fvg_signals = self.fair_value_gaps.get_fvg_signals(df, current_index, self.smc_config)
            
            has_signal = ((direction == 'bullish' and fvg_signals.get('bullish_fvg_signal', False)) or
                         (direction == 'bearish' and fvg_signals.get('bearish_fvg_signal', False)))
            
            if not has_signal:
                return {'present': False}
            
            fvg_count = fvg_signals.get('fvg_confluence_count', 0)
            fvg_strength = fvg_signals.get('fvg_strength', 0.5)
            nearest_distance = fvg_signals.get('nearest_fvg_distance', float('inf'))
            
            # Determine FVG size category based on strength and distance
            if fvg_strength > 0.7 and nearest_distance < 0.0005:  # Within 5 pips
                size = 'large_gap'
            elif fvg_strength > 0.5 and nearest_distance < 0.001:   # Within 10 pips
                size = 'medium_gap'
            else:
                size = 'small_gap'
            
            return {
                'present': True,
                'size': size,
                'strength': fvg_strength,
                'count': fvg_count,
                'nearest_distance': nearest_distance,
                'distance_pips': nearest_distance * 10000 if nearest_distance != float('inf') else 0
            }
            
        except Exception as e:
            self.logger.error(f"FVG confluence analysis failed: {e}")
            return {'present': False}
    
    def _analyze_liquidity_confluence(self, df: pd.DataFrame, current_index: int, direction: str) -> Dict:
        """Analyze liquidity confluence with enhanced details"""
        try:
            liquidity_conf = self._check_liquidity_confluence(df, current_index, direction)
            
            if not liquidity_conf:
                return {'present': False}
            
            # Analyze type of liquidity - check for equal highs/lows vs general swing levels
            recent_data = df.iloc[max(0, current_index - 20):current_index + 1]
            
            # Check for equal levels (more significant)
            swing_highs = recent_data[recent_data.get('swing_high', False) == True]
            swing_lows = recent_data[recent_data.get('swing_low', False) == True]
            
            has_equal_levels = False
            tolerance = 0.00005  # 0.5 pip tolerance
            
            if direction == 'bullish' and len(swing_lows) >= 2:
                low_prices = swing_lows['low'].values
                for i in range(len(low_prices) - 1):
                    for j in range(i + 1, len(low_prices)):
                        if abs(low_prices[i] - low_prices[j]) <= tolerance:
                            has_equal_levels = True
                            break
                            
            elif direction == 'bearish' and len(swing_highs) >= 2:
                high_prices = swing_highs['high'].values
                for i in range(len(high_prices) - 1):
                    for j in range(i + 1, len(high_prices)):
                        if abs(high_prices[i] - high_prices[j]) <= tolerance:
                            has_equal_levels = True
                            break
            
            liquidity_type = 'equal_highs_lows' if has_equal_levels else 'swing_levels'
            
            return {
                'present': True,
                'type': liquidity_type,
                'confidence': 0.8 if has_equal_levels else 0.6,
                'equal_levels_detected': has_equal_levels,
                'direction': direction
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity confluence analysis failed: {e}")
            return {'present': False}
    
    def _calculate_confluence_synergy(self, confluence_factors: Dict) -> float:
        """Calculate synergy bonuses for complementary confluence factors"""
        try:
            synergy_bonus = 0.0
            factor_names = list(confluence_factors.keys())
            
            # High-impact synergies
            synergies = {
                # Market structure + Premium/Discount = Very strong combination
                ('market_structure_break', 'premium_discount_alignment'): 0.3,
                
                # Market structure + Higher timeframe = Strong trend confirmation
                ('market_structure_break', 'higher_timeframe_alignment'): 0.2,
                
                # Premium/Discount + Order blocks = Institutional confluence
                ('premium_discount_alignment', 'order_block_support'): 0.25,
                
                # Liquidity sweep + Structure break = Manipulation then move
                ('liquidity_sweep', 'market_structure_break'): 0.2,
                
                # Volume + Any other factor = Confirmation strength
                ('volume_confirmation', 'market_structure_break'): 0.15,
                ('volume_confirmation', 'order_block_support'): 0.15,
                
                # Order blocks + FVG = Zone confluence
                ('order_block_support', 'fair_value_gap'): 0.1
            }
            
            # Check for synergy combinations
            for (factor1, factor2), bonus in synergies.items():
                if factor1 in factor_names and factor2 in factor_names:
                    # Weight synergy bonus by confidence of both factors
                    conf1 = confluence_factors[factor1].get('confidence', 0.5)
                    conf2 = confluence_factors[factor2].get('confidence', 0.5)
                    weighted_bonus = bonus * conf1 * conf2
                    synergy_bonus += weighted_bonus
            
            # Triple factor bonuses (rare but very strong)
            if len(factor_names) >= 3:
                if all(f in factor_names for f in ['market_structure_break', 'premium_discount_alignment', 'higher_timeframe_alignment']):
                    synergy_bonus += 0.4  # Perfect SMC confluence
                
                if all(f in factor_names for f in ['market_structure_break', 'order_block_support', 'volume_confirmation']):
                    synergy_bonus += 0.3  # Strong institutional confirmation
            
            # Cap synergy bonus
            return min(synergy_bonus, 1.0)
            
        except Exception as e:
            self.logger.error(f"Confluence synergy calculation failed: {e}")
            return 0.0
    
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
            
            # Apply comprehensive signal filtering (final validation)
            if not self._passes_signal_filters(signal, latest_row, epic, timeframe):
                self.logger.debug(f"Signal filtered out by comprehensive filtering system")
                return None
            
            return signal
            
        except Exception as e:
            self.logger.error(f"SMC signal creation failed: {e}")
            return None
    
    def _add_risk_management_levels(
        self, 
        signal: Dict, 
        current_row: pd.Series, 
        signal_type: str
    ) -> Dict:
        """Add institutional-style risk management levels based on advanced SMC principles"""
        try:
            current_price = signal['price']
            confluence_info = signal.get('confluence_info', {})
            
            # Advanced institutional risk management
            risk_analysis = self._calculate_institutional_risk_levels(
                current_price, signal_type, current_row, confluence_info
            )
            
            if not risk_analysis:
                return self._apply_fallback_risk_management(signal, signal_type)
            
            # Calculate position sizing based on confluence quality
            position_sizing = self._calculate_position_sizing(confluence_info, risk_analysis)
            
            # Add comprehensive risk management data
            risk_data = {
                # Core levels
                'stop_loss': risk_analysis['stop_loss'],
                'take_profit': risk_analysis['take_profit'],
                'breakeven_level': risk_analysis.get('breakeven_level'),
                
                # Distance metrics
                'stop_distance_pips': risk_analysis['stop_distance_pips'],
                'target_distance_pips': risk_analysis['target_distance_pips'],
                'risk_reward_ratio': risk_analysis['risk_reward_ratio'],
                
                # Institutional features
                'liquidity_stop': risk_analysis.get('liquidity_stop'),
                'structure_invalidation': risk_analysis.get('structure_invalidation'),
                'trailing_stop_config': risk_analysis.get('trailing_config'),
                
                # Position management
                'position_sizing': position_sizing,
                'max_risk_percent': position_sizing.get('max_risk_percent', 1.0),
                'confluence_multiplier': position_sizing.get('confluence_multiplier', 1.0),
                
                # Premium/Discount exits
                'pd_exit_levels': risk_analysis.get('pd_exit_levels', {}),
                
                # Risk management type
                'rm_type': risk_analysis.get('rm_type', 'institutional'),
                'rm_confidence': risk_analysis.get('confidence', 0.5)
            }
            
            signal.update(risk_data)
            return signal
            
        except Exception as e:
            self.logger.error(f"Institutional risk management calculation failed: {e}")
            return self._apply_fallback_risk_management(signal, signal_type)
    
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