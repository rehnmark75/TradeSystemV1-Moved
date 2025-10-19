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

        # Initialize Multi-Timeframe analyzer
        self.mtf_analyzer = None
        self.mtf_enabled = self.smc_config.get('mtf_enabled', True)

        if self.mtf_enabled and data_fetcher:
            try:
                from .helpers.smc_mtf_analyzer import SMCMultiTimeframeAnalyzer
                self.mtf_analyzer = SMCMultiTimeframeAnalyzer(
                    logger=self.logger,
                    data_fetcher=data_fetcher
                )
                self.logger.info(f"🔄 Multi-Timeframe analysis ENABLED: {self.mtf_analyzer.check_timeframes}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MTF analyzer: {e}")
                self.mtf_enabled = False
        else:
            self.logger.info(f"⚠️ Multi-Timeframe analysis DISABLED")

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

    def reinitialize_mtf_analyzer(self, data_fetcher):
        """
        Reinitialize MTF analyzer with a new data_fetcher

        This is called by backtest_scanner after setting data_fetcher post-initialization
        """
        if not self.mtf_enabled:
            return

        if data_fetcher:
            try:
                from .helpers.smc_mtf_analyzer import SMCMultiTimeframeAnalyzer
                self.mtf_analyzer = SMCMultiTimeframeAnalyzer(
                    logger=self.logger,
                    data_fetcher=data_fetcher
                )
                self.logger.info(f"🔄 MTF Analyzer re-initialized with backtest data_fetcher: {self.mtf_analyzer.check_timeframes}")
            except Exception as e:
                self.logger.warning(f"Failed to reinitialize MTF analyzer: {e}")
                self.mtf_enabled = False
                self.mtf_analyzer = None

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

            # Multi-Timeframe Validation (if enabled)
            mtf_result = None
            mtf_confidence_boost = 0.0
            htf_ltf_validation = None

            if self.mtf_enabled and self.mtf_analyzer and self.mtf_analyzer.is_mtf_enabled():
                try:
                    # Get structure break info for MTF context
                    structure_info = {
                        'break_type': latest_row.get('break_type', 'BOS'),
                        'break_direction': latest_row.get('break_direction', 'bullish'),
                        'significance': latest_row.get('structure_significance', 0.5)
                    }

                    # Get evaluation time (use df index if evaluation_time not provided)
                    eval_time = pd.Timestamp(evaluation_time) if evaluation_time else df.index[-1]

                    # ENHANCED: Validate LTF signal against HTF structure bias (4h)
                    htf_ltf_validation = self.mtf_analyzer.validate_ltf_entry_against_htf(
                        epic=epic,
                        current_time=eval_time,
                        ltf_signal_type=signal_type,
                        ltf_structure_info=structure_info,
                        htf_timeframe='4h'
                    )

                    # Check if HTF-LTF alignment is valid
                    if not htf_ltf_validation.get('htf_ltf_aligned', True):
                        # REJECT signal if HTF and LTF are conflicting
                        self.logger.warning(
                            f"❌ SIGNAL REJECTED: HTF-LTF conflict detected - "
                            f"4h {htf_ltf_validation.get('htf_bias', 'unknown')} vs {signal_type} signal"
                        )
                        return None

                    # Get additional confidence boost from HTF-LTF alignment
                    htf_ltf_boost = htf_ltf_validation.get('additional_boost', 0.0)
                    alignment_quality = htf_ltf_validation.get('alignment_quality', 'weak')

                    # Legacy MTF validation (15m + 4h alignment check)
                    mtf_result = self.mtf_analyzer.validate_higher_timeframe_smc(
                        epic=epic,
                        current_time=eval_time,
                        signal_type=signal_type,
                        structure_info=structure_info
                    )

                    # Combine both MTF boosts
                    legacy_mtf_boost = mtf_result.get('confidence_boost', 0.0)
                    mtf_confidence_boost = max(htf_ltf_boost, legacy_mtf_boost)  # Take best boost

                    # Log MTF results
                    if htf_ltf_validation.get('htf_ltf_aligned'):
                        confluence_factors = htf_ltf_validation.get('confluence_factors', [])
                        self.logger.info(
                            f"✅ HTF-LTF Validation PASSED: {alignment_quality.upper()} alignment "
                            f"(boost: {mtf_confidence_boost:+.3f}, factors: {len(confluence_factors)})"
                        )
                    else:
                        self.logger.debug(f"⚠️ MTF Validation WEAK: Limited HTF alignment")

                except Exception as e:
                    self.logger.warning(f"MTF validation failed: {e}, continuing without MTF")
                    mtf_confidence_boost = 0.0
                    htf_ltf_validation = None

            # Calculate confidence with MTF boost
            base_confidence = self._calculate_fast_confidence(latest_row, confluence_score)
            confidence = min(base_confidence + mtf_confidence_boost, 0.95)

            # ENHANCED: Require MTF alignment for high-confidence signals (>75%)
            # This prevents weak signals from passing just on confluence alone
            if self.mtf_enabled and self.mtf_analyzer and self.mtf_analyzer.is_mtf_enabled():
                # For high-confidence signals, require at least neutral MTF (no penalty)
                if base_confidence > 0.75 and mtf_confidence_boost < 0:
                    self.logger.debug(
                        f"⚠️ High-confidence signal REJECTED: {base_confidence:.1%} base confidence "
                        f"but MTF opposing ({mtf_confidence_boost:+.2f})"
                    )
                    return None

                # For medium-confidence signals, require at least one timeframe aligned
                if 0.65 <= base_confidence <= 0.75:
                    if not mtf_result or not mtf_result.get('timeframes_aligned', []):
                        self.logger.debug(
                            f"⚠️ Medium-confidence signal REJECTED: {base_confidence:.1%} base confidence "
                            f"but no MTF alignment"
                        )
                        return None

            if confidence < self.min_confidence:
                return None

            # Create signal with MTF data
            signal = self._create_fast_signal(
                signal_type=signal_type,
                epic=epic,
                timeframe=timeframe,
                latest_row=latest_row,
                spread_pips=spread_pips,
                confluence_score=confluence_score,
                confidence=confidence,
                mtf_result=mtf_result,
                htf_ltf_validation=htf_ltf_validation
            )

            if signal:
                mtf_status = self.mtf_analyzer.get_mtf_summary(mtf_result) if mtf_result else "MTF disabled"
                self.logger.info(
                    f"🧠 Fast SMC {signal_type} signal: {confidence:.1%} confidence, "
                    f"confluence: {confluence_score:.1f}, MTF: {mtf_status}"
                )

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
        """
        Enhanced confluence calculation with stronger requirements

        Now requires multiple strong confirmations beyond just structure break:
        - Structure break is BASE requirement (0.5 points, not enough alone)
        - Need at least 2-3 additional confirmations to reach threshold
        - Premium/discount zone awareness
        - Liquidity sweep detection
        - Multi-factor validation
        """
        try:
            confluence_score = 0.0
            current_row = df.iloc[current_index]
            break_direction = current_row.get('break_direction', '')

            # 1. Structure break (REDUCED - base requirement only)
            # Not enough to trade on its own anymore
            if current_row.get('structure_break', False):
                confluence_score += 0.5  # Reduced from 1.0

            # 2. FVG confluence - STRONG signal
            has_aligned_fvg = False
            if break_direction == 'bullish' and current_row.get('fvg_bullish', False):
                confluence_score += 0.8  # Increased from 0.6
                has_aligned_fvg = True
            elif break_direction == 'bearish' and current_row.get('fvg_bearish', False):
                confluence_score += 0.8  # Increased from 0.6
                has_aligned_fvg = True

            # 3. Order block confluence - Check RECENT and ALIGNED
            has_aligned_ob = False
            if self.enhanced_validation:
                recent_window = df.iloc[max(0, current_index-5):current_index+1]
                if break_direction == 'bullish' and recent_window['order_block_bullish'].any():
                    confluence_score += 0.9  # Increased from 0.8
                    has_aligned_ob = True
                elif break_direction == 'bearish' and recent_window['order_block_bearish'].any():
                    confluence_score += 0.9  # Increased from 0.8
                    has_aligned_ob = True

            # 4. Volume confluence - Must be SIGNIFICANT
            volume_col = 'volume' if 'volume' in df.columns else 'ltv'
            high_volume = False
            if volume_col in df.columns:
                current_volume = current_row.get(volume_col, 0)
                avg_volume = df[volume_col].iloc[max(0, current_index-20):current_index].mean()

                # Require 1.8x volume for confluence (increased from 1.5x)
                if current_volume > (avg_volume * 1.8):
                    confluence_score += 0.6  # Increased from 0.4
                    high_volume = True
                # Give partial credit for moderate volume (1.5x)
                elif current_volume > (avg_volume * 1.5):
                    confluence_score += 0.3

            # 5. EMA trend alignment (only in enhanced mode)
            ema_aligned = False
            if self.enhanced_validation and all(col in df.columns for col in ['ema_21', 'ema_50', 'ema_200']):
                ema_21 = current_row.get('ema_21', 0)
                ema_50 = current_row.get('ema_50', 0)
                ema_200 = current_row.get('ema_200', 0)
                price = current_row.get('close', 0)

                # Check EMA alignment with direction
                if break_direction == 'bullish':
                    if (ema_21 > ema_50 > ema_200) and (price > ema_21):
                        confluence_score += 0.7  # Increased from 0.5
                        ema_aligned = True
                elif break_direction == 'bearish':
                    if (ema_21 < ema_50 < ema_200) and (price < ema_21):
                        confluence_score += 0.7  # Increased from 0.5
                        ema_aligned = True

            # 6. Premium/Discount Zone Analysis (ENHANCED)
            # Check if price is in favorable zone for the signal direction
            recent_window = df.iloc[max(0, current_index-50):current_index+1]
            swing_high = recent_window['high'].max()
            swing_low = recent_window['low'].min()
            price = current_row.get('close', 0)

            in_favorable_zone = False
            pd_zone_type = 'equilibrium'

            if swing_high > swing_low:
                price_position = (price - swing_low) / (swing_high - swing_low)

                # Premium zone: upper 30% (70-100%)
                if price_position >= 0.7:
                    pd_zone_type = 'premium'
                    # Bearish signal from premium zone gets bonus
                    if break_direction == 'bearish':
                        confluence_score += 0.8  # Increased from 0.6
                        in_favorable_zone = True
                    # Bullish signal from premium zone gets penalty
                    elif break_direction == 'bullish':
                        confluence_score -= 0.5  # NEW: Penalty for buying high

                # Discount zone: lower 30% (0-30%)
                elif price_position <= 0.3:
                    pd_zone_type = 'discount'
                    # Bullish signal from discount zone gets bonus
                    if break_direction == 'bullish':
                        confluence_score += 0.8  # Increased from 0.6
                        in_favorable_zone = True
                    # Bearish signal from discount zone gets penalty
                    elif break_direction == 'bearish':
                        confluence_score -= 0.5  # NEW: Penalty for selling low

                # Equilibrium zone: 30-70%
                else:
                    pd_zone_type = 'equilibrium'
                    # Neutral zone - small bonus for any direction
                    confluence_score += 0.2

            # Log premium/discount zone for debugging
            if self.logger and pd_zone_type != 'equilibrium':
                self.logger.debug(
                    f"P/D Zone: {pd_zone_type}, favorable={in_favorable_zone}, "
                    f"direction={break_direction}, position={price_position:.1%}"
                )

            # 7. Liquidity Sweep Detection (ENHANCED)
            # Check if recent price action shows liquidity grab before reversal
            liquidity_sweep_detected = False
            if current_index >= 5:
                # Extended lookback for better liquidity sweep detection
                recent_bars = df.iloc[max(0, current_index-5):current_index+1]

                # Bullish: Look for sweep below recent lows then recovery
                if break_direction == 'bullish':
                    # Get recent lows (excluding current bar)
                    prev_lows = recent_bars['low'].iloc[:-1]
                    current_low = current_row.get('low', 0)
                    current_close = current_row.get('close', 0)
                    current_high = current_row.get('high', 0)

                    if len(prev_lows) > 0:
                        prev_low = prev_lows.min()

                        # Classic liquidity sweep: swept below then closed back above
                        if current_low < prev_low and current_close > prev_low:
                            # Strong sweep with good recovery (closed in upper 50% of range)
                            range_size = current_high - current_low
                            close_position = (current_close - current_low) / range_size if range_size > 0 else 0

                            if close_position >= 0.5:
                                confluence_score += 0.9  # Strong liquidity grab
                                liquidity_sweep_detected = True
                            else:
                                confluence_score += 0.6  # Weak recovery
                                liquidity_sweep_detected = True

                        # Equal lows liquidity sweep: current low near previous low
                        elif abs(current_low - prev_low) / prev_low < 0.0002:  # Within 2 pips
                            confluence_score += 0.4  # Equal lows = liquidity
                            liquidity_sweep_detected = True

                # Bearish: Look for sweep above recent highs then rejection
                elif break_direction == 'bearish':
                    # Get recent highs (excluding current bar)
                    prev_highs = recent_bars['high'].iloc[:-1]
                    current_high = current_row.get('high', 0)
                    current_close = current_row.get('close', 0)
                    current_low = current_row.get('low', 0)

                    if len(prev_highs) > 0:
                        prev_high = prev_highs.max()

                        # Classic liquidity sweep: swept above then closed back below
                        if current_high > prev_high and current_close < prev_high:
                            # Strong sweep with good rejection (closed in lower 50% of range)
                            range_size = current_high - current_low
                            close_position = (current_close - current_low) / range_size if range_size > 0 else 0

                            if close_position <= 0.5:
                                confluence_score += 0.9  # Strong liquidity grab
                                liquidity_sweep_detected = True
                            else:
                                confluence_score += 0.6  # Weak rejection
                                liquidity_sweep_detected = True

                        # Equal highs liquidity sweep: current high near previous high
                        elif abs(current_high - prev_high) / prev_high < 0.0002:  # Within 2 pips
                            confluence_score += 0.4  # Equal highs = liquidity
                            liquidity_sweep_detected = True

            if liquidity_sweep_detected:
                self.logger.debug(f"💧 Liquidity sweep detected: {break_direction} direction")

            # 8. Strong Candle Pattern (NEW)
            # Reward strong directional candles that show conviction
            body_size = abs(current_row.get('close', 0) - current_row.get('open', 0))
            range_size = current_row.get('high', 0) - current_row.get('low', 0)

            if range_size > 0:
                body_ratio = body_size / range_size

                # Strong directional candle (>70% body)
                if body_ratio > 0.7:
                    # Verify direction matches signal
                    is_bullish_candle = current_row.get('close', 0) > current_row.get('open', 0)
                    if (break_direction == 'bullish' and is_bullish_candle) or \
                       (break_direction == 'bearish' and not is_bullish_candle):
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
        confidence: float,
        mtf_result: Optional[Dict] = None,
        htf_ltf_validation: Optional[Dict] = None
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

            # Add Multi-Timeframe validation details if available
            if mtf_result:
                signal['mtf_validation'] = {
                    'enabled': True,
                    'timeframes_checked': mtf_result.get('timeframes_checked', []),
                    'timeframes_aligned': mtf_result.get('timeframes_aligned', []),
                    'alignment_ratio': mtf_result.get('alignment_ratio', 0.0),
                    'confidence_boost': mtf_result.get('confidence_boost', 0.0),
                    'validation_passed': mtf_result.get('validation_passed', False),
                    'htf_details': mtf_result.get('details', {})
                }

                # Update entry reason to include MTF status
                aligned_tfs = mtf_result.get('timeframes_aligned', [])
                if aligned_tfs:
                    signal['entry_reason'] += f"_MTF_{'_'.join(aligned_tfs)}"
            else:
                signal['mtf_validation'] = {'enabled': False}

            # Add HTF-LTF validation details (enhanced MTF strategy)
            if htf_ltf_validation:
                signal['htf_ltf_validation'] = {
                    'enabled': True,
                    'htf_bias': htf_ltf_validation.get('htf_bias', 'unknown'),
                    'htf_structure_type': htf_ltf_validation.get('htf_structure_type'),
                    'htf_significance': htf_ltf_validation.get('htf_significance', 0.0),
                    'alignment_quality': htf_ltf_validation.get('alignment_quality', 'weak'),
                    'aligned': htf_ltf_validation.get('htf_ltf_aligned', False),
                    'confluence_factors': htf_ltf_validation.get('confluence_factors', []),
                    'additional_boost': htf_ltf_validation.get('additional_boost', 0.0),
                    'reason': htf_ltf_validation.get('reason', '')
                }

                # Update entry reason to include HTF-LTF status
                if htf_ltf_validation.get('htf_ltf_aligned'):
                    htf_bias = htf_ltf_validation.get('htf_bias', 'unknown')
                    alignment_quality = htf_ltf_validation.get('alignment_quality', 'weak')
                    signal['entry_reason'] += f"_HTF_{htf_bias.upper()}_{alignment_quality.upper()}"
            else:
                signal['htf_ltf_validation'] = {'enabled': False}

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
            
            # Convert distances to points/pips for order API
            if 'JPY' in epic:
                stop_distance_points = int(stop_distance * 100)
                target_distance_points = int(target_distance * 100)
            else:
                stop_distance_points = int(stop_distance * 10000)
                target_distance_points = int(target_distance * 10000)

            signal.update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'stop_distance': stop_distance_points,  # For order API
                'limit_distance': target_distance_points,  # For order API
                'stop_distance_pips': stop_distance * 10000,  # Keep for compatibility
                'target_distance_pips': target_distance * 10000,  # Keep for compatibility
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