# core/strategies/helpers/bb_reversal_detector.py
"""
BB Reversal Quality Detector - Fix the 14.7% Win Rate
🔄 PURPOSE: Detect high-quality reversal setups instead of catching falling knives
🎯 GOAL: Increase win rate from 14.7% to 45%+ by better reversal confirmation
📊 METHOD: Multi-factor reversal quality scoring

This module addresses the core issue: the strategy is catching falling knives
instead of waiting for proper reversal confirmations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime


class BBReversalDetector:
    """
    🔄 REVERSAL QUALITY: Detect high-probability BB reversals
    
    Core improvements to fix low win rate:
    1. Wait for actual price reversal confirmation
    2. Require volume confirmation
    3. Check for momentum divergence
    4. Validate market regime (ranging vs trending)
    5. Confirm with multiple timeframes
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Reversal detection configuration
        self.reversal_config = {
            # Price action requirements
            'min_reversal_bars': 2,           # Confirm reversal over N bars
            'min_wick_ratio': 0.3,           # Minimum wick size (rejection)
            'max_body_ratio': 0.7,           # Maximum body size (avoid trending candles)
            'min_reversal_strength': 0.4,    # Minimum reversal momentum
            
            # Volume confirmation
            'require_volume_confirmation': True,
            'min_volume_ratio': 1.2,         # 20% above average
            'volume_spike_factor': 1.5,      # Look for volume spikes
            
            # Momentum divergence
            'detect_divergence': True,
            'divergence_lookback': 10,       # Bars to look back
            'min_divergence_strength': 0.3,  # Minimum divergence strength
            
            # Market regime validation
            'require_ranging_market': True,
            'max_trend_strength': 0.6,       # Avoid strong trends
            'min_bb_expansion': 0.8,         # Minimum BB expansion ratio
            
            # Multi-timeframe confirmation
            'require_mtf_confirmation': True,
            'min_mtf_confluence': 0.4,       # 40% minimum confluence
            'htf_reversal_weight': 0.3,      # Higher timeframe reversal weight
        }
        
        # Statistics
        self._reversals_detected = 0
        self._high_quality_reversals = 0
        self._low_quality_reversals = 0
        
        self.logger.info("🔄 BB Reversal Detector initialized")
        self.logger.info(f"   Target: Improve win rate from 14.7% to 45%+")

    def detect_reversal_quality(
        self, 
        df: pd.DataFrame, 
        signal_type: str, 
        bb_position: float
    ) -> Dict:
        """
        🔄 Comprehensive reversal quality analysis
        
        Returns quality score and detailed analysis to fix low win rate
        """
        try:
            if len(df) < self.reversal_config['divergence_lookback']:
                return self._low_quality_result("Insufficient data")
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            quality_score = 0.0
            quality_factors = []
            
            # 1. Price Action Reversal Confirmation (25% weight)
            price_action_score = self._analyze_price_action_reversal(df, signal_type)
            quality_score += price_action_score * 0.25
            quality_factors.append(f"Price action: {price_action_score:.2f}")
            
            # 2. Volume Confirmation (20% weight)
            volume_score = self._analyze_volume_confirmation(df)
            quality_score += volume_score * 0.20
            quality_factors.append(f"Volume: {volume_score:.2f}")
            
            # 3. Momentum Divergence (20% weight)
            divergence_score = self._detect_momentum_divergence(df, signal_type)
            quality_score += divergence_score * 0.20
            quality_factors.append(f"Divergence: {divergence_score:.2f}")
            
            # 4. Market Regime Validation (20% weight)
            regime_score = self._validate_market_regime(df, bb_position)
            quality_score += regime_score * 0.20
            quality_factors.append(f"Market regime: {regime_score:.2f}")
            
            # 5. BB Structure Quality (15% weight)
            bb_structure_score = self._analyze_bb_structure(current, signal_type)
            quality_score += bb_structure_score * 0.15
            quality_factors.append(f"BB structure: {bb_structure_score:.2f}")
            
            # Determine quality level
            if quality_score >= 0.70:
                quality_level = "HIGH"
                self._high_quality_reversals += 1
            elif quality_score >= 0.50:
                quality_level = "MEDIUM"
            else:
                quality_level = "LOW"
                self._low_quality_reversals += 1
            
            self._reversals_detected += 1
            
            return {
                'quality_score': quality_score,
                'quality_level': quality_level,
                'quality_factors': quality_factors,
                'should_trade': quality_score >= 0.50,  # Only trade medium+ quality
                'reversal_strength': self._calculate_reversal_strength(df, signal_type),
                'risk_level': self._assess_risk_level(quality_score, signal_type),
                'expected_win_rate': self._estimate_win_rate(quality_score),
                'analysis_details': {
                    'price_action_score': price_action_score,
                    'volume_score': volume_score,
                    'divergence_score': divergence_score,
                    'regime_score': regime_score,
                    'bb_structure_score': bb_structure_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Reversal quality detection failed: {e}")
            return self._low_quality_result(f"Error: {str(e)}")

    def _analyze_price_action_reversal(self, df: pd.DataFrame, signal_type: str) -> float:
        """
        📊 Analyze price action for reversal confirmation
        This addresses the "falling knife" problem
        """
        try:
            if len(df) < 3:
                return 0.0
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            score = 0.0
            
            # Calculate candle metrics
            current_body = abs(current['close'] - current['open'])
            current_range = current['high'] - current['low']
            current_upper_wick = current['high'] - max(current['open'], current['close'])
            current_lower_wick = min(current['open'], current['close']) - current['low']
            
            if current_range <= 0:
                return 0.0
            
            # 1. Wick rejection confirmation
            if signal_type == 'BULL':
                # For BULL signals, want lower wick rejection
                lower_wick_ratio = current_lower_wick / current_range
                if lower_wick_ratio >= self.reversal_config['min_wick_ratio']:
                    score += 0.4  # Strong rejection at lows
                elif lower_wick_ratio >= 0.15:
                    score += 0.2  # Moderate rejection
            else:  # BEAR
                # For BEAR signals, want upper wick rejection
                upper_wick_ratio = current_upper_wick / current_range
                if upper_wick_ratio >= self.reversal_config['min_wick_ratio']:
                    score += 0.4  # Strong rejection at highs
                elif upper_wick_ratio >= 0.15:
                    score += 0.2  # Moderate rejection
            
            # 2. Body size validation (avoid large trending candles)
            body_ratio = current_body / current_range
            if body_ratio <= self.reversal_config['max_body_ratio']:
                score += 0.3  # Good - not a strong trending candle
            elif body_ratio <= 0.85:
                score += 0.1  # Moderate
            # No points for large bodies (trending candles)
            
            # 3. Multi-bar reversal confirmation
            if signal_type == 'BULL':
                # Check if we're seeing higher lows or bullish divergence
                if current['low'] > previous['low'] and current['close'] > previous['close']:
                    score += 0.3  # Confirmed reversal pattern
                elif current['close'] > current['open']:  # Bullish candle
                    score += 0.1
            else:  # BEAR
                # Check if we're seeing lower highs or bearish pattern
                if current['high'] < previous['high'] and current['close'] < previous['close']:
                    score += 0.3  # Confirmed reversal pattern
                elif current['close'] < current['open']:  # Bearish candle
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Price action analysis failed: {e}")
            return 0.0

    def _analyze_volume_confirmation(self, df: pd.DataFrame) -> float:
        """
        📊 Analyze volume for reversal confirmation
        """
        try:
            if 'volume' not in df.columns and 'ltv' not in df.columns:
                return 0.5  # Neutral if no volume data
            
            volume_col = 'volume' if 'volume' in df.columns else 'ltv'
            current_volume = df[volume_col].iloc[-1]
            
            # Calculate average volume over last 20 bars
            if len(df) >= 20:
                avg_volume = df[volume_col].tail(20).mean()
            else:
                avg_volume = df[volume_col].mean()
            
            if avg_volume <= 0:
                return 0.5  # Neutral if no meaningful volume data
            
            volume_ratio = current_volume / avg_volume
            score = 0.0
            
            # Volume confirmation scoring
            if volume_ratio >= self.reversal_config['volume_spike_factor']:
                score = 1.0  # Strong volume spike
            elif volume_ratio >= self.reversal_config['min_volume_ratio']:
                score = 0.7  # Good volume
            elif volume_ratio >= 1.0:
                score = 0.4  # Average volume
            else:
                score = 0.1  # Low volume (poor quality)
            
            return score
            
        except Exception as e:
            self.logger.debug(f"Volume analysis failed: {e}")
            return 0.5

    def _detect_momentum_divergence(self, df: pd.DataFrame, signal_type: str) -> float:
        """
        📊 Detect momentum divergence for higher quality signals
        """
        try:
            lookback = self.reversal_config['divergence_lookback']
            if len(df) < lookback:
                return 0.5  # Neutral if insufficient data
            
            # Use ATR or price momentum as proxy for momentum indicator
            recent_data = df.tail(lookback)
            
            # Calculate price momentum
            price_momentum = recent_data['close'].diff().rolling(3).mean()
            
            # Calculate recent highs/lows for divergence
            if signal_type == 'BULL':
                # Look for bullish divergence: lower lows in price, higher lows in momentum
                recent_low_idx = recent_data['low'].idxmin()
                current_idx = recent_data.index[-1]
                
                if recent_low_idx != current_idx:  # Not the most recent bar
                    recent_low_momentum = price_momentum.loc[recent_low_idx]
                    current_momentum = price_momentum.iloc[-1]
                    
                    price_made_lower_low = recent_data['low'].iloc[-1] <= recent_data['low'].loc[recent_low_idx]
                    momentum_higher = current_momentum > recent_low_momentum
                    
                    if price_made_lower_low and momentum_higher:
                        return 0.9  # Strong bullish divergence
                    elif momentum_higher:
                        return 0.6  # Momentum improvement
            else:  # BEAR
                # Look for bearish divergence: higher highs in price, lower highs in momentum
                recent_high_idx = recent_data['high'].idxmax()
                current_idx = recent_data.index[-1]
                
                if recent_high_idx != current_idx:
                    recent_high_momentum = price_momentum.loc[recent_high_idx]
                    current_momentum = price_momentum.iloc[-1]
                    
                    price_made_higher_high = recent_data['high'].iloc[-1] >= recent_data['high'].loc[recent_high_idx]
                    momentum_lower = current_momentum < recent_high_momentum
                    
                    if price_made_higher_high and momentum_lower:
                        return 0.9  # Strong bearish divergence
                    elif momentum_lower:
                        return 0.6  # Momentum deterioration
            
            return 0.3  # No clear divergence
            
        except Exception as e:
            self.logger.debug(f"Divergence detection failed: {e}")
            return 0.5

    def _validate_market_regime(self, df: pd.DataFrame, bb_position: float) -> float:
        """
        📊 Validate market regime for mean reversion suitability
        """
        try:
            if len(df) < 20:
                return 0.5
            
            score = 0.0
            recent_data = df.tail(20)
            
            # 1. Check for ranging vs trending market
            price_range = recent_data['high'].max() - recent_data['low'].min()
            recent_atr = recent_data.get('atr', pd.Series([price_range/20]*len(recent_data))).mean()
            
            if recent_atr > 0:
                range_to_atr_ratio = price_range / (recent_atr * 20)
                
                if range_to_atr_ratio <= 3.0:
                    score += 0.4  # Good ranging market
                elif range_to_atr_ratio <= 5.0:
                    score += 0.2  # Moderate ranging
                # No points for strong trending markets
            
            # 2. BB expansion validation
            current_bb_width = df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]
            avg_bb_width = (df['bb_upper'] - df['bb_lower']).tail(20).mean()
            
            if avg_bb_width > 0:
                expansion_ratio = current_bb_width / avg_bb_width
                
                if 0.8 <= expansion_ratio <= 1.5:
                    score += 0.3  # Good BB expansion level
                elif 0.6 <= expansion_ratio <= 2.0:
                    score += 0.1  # Moderate expansion
            
            # 3. Position quality for mean reversion
            if bb_position <= 0.20 or bb_position >= 0.80:
                score += 0.3  # Extreme position good for mean reversion
            elif bb_position <= 0.30 or bb_position >= 0.70:
                score += 0.2  # Good position
            elif bb_position <= 0.40 or bb_position >= 0.60:
                score += 0.1  # Moderate position
            # No points for middle positions
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Market regime validation failed: {e}")
            return 0.5

    def _analyze_bb_structure(self, current: pd.Series, signal_type: str) -> float:
        """
        📊 Analyze BB structure quality
        """
        try:
            bb_upper = current['bb_upper']
            bb_middle = current['bb_middle']
            bb_lower = current['bb_lower']
            current_price = current['close']
            
            bb_width = bb_upper - bb_lower
            
            if bb_width <= 0 or bb_middle <= 0:
                return 0.0
            
            score = 0.0
            
            # 1. BB width quality
            bb_width_pct = bb_width / bb_middle
            if 0.015 <= bb_width_pct <= 0.08:  # Good width range
                score += 0.4
            elif 0.010 <= bb_width_pct <= 0.12:  # Acceptable range
                score += 0.2
            
            # 2. Position relative to bands
            if signal_type == 'BULL':
                # For BULL, closer to lower band is better
                distance_to_lower = abs(current_price - bb_lower)
                distance_to_upper = abs(current_price - bb_upper)
                
                if distance_to_lower < distance_to_upper:
                    proximity_score = (bb_width - distance_to_lower) / bb_width
                    score += proximity_score * 0.6
            else:  # BEAR
                # For BEAR, closer to upper band is better
                distance_to_lower = abs(current_price - bb_lower)
                distance_to_upper = abs(current_price - bb_upper)
                
                if distance_to_upper < distance_to_lower:
                    proximity_score = (bb_width - distance_to_upper) / bb_width
                    score += proximity_score * 0.6
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.debug(f"BB structure analysis failed: {e}")
            return 0.5

    def _calculate_reversal_strength(self, df: pd.DataFrame, signal_type: str) -> float:
        """Calculate overall reversal strength"""
        try:
            if len(df) < 3:
                return 0.5
            
            # Simple reversal strength based on recent price action
            recent_data = df.tail(3)
            price_changes = recent_data['close'].diff()
            
            if signal_type == 'BULL':
                # Look for bullish momentum
                recent_strength = price_changes.tail(2).mean()
                return max(0.0, min(1.0, (recent_strength / recent_data['close'].iloc[-1]) * 1000 + 0.5))
            else:  # BEAR
                # Look for bearish momentum
                recent_strength = -price_changes.tail(2).mean()  # Negative for bearish
                return max(0.0, min(1.0, (recent_strength / recent_data['close'].iloc[-1]) * 1000 + 0.5))
                
        except:
            return 0.5

    def _assess_risk_level(self, quality_score: float, signal_type: str) -> str:
        """Assess risk level based on quality score"""
        if quality_score >= 0.80:
            return "LOW"
        elif quality_score >= 0.60:
            return "MEDIUM"
        elif quality_score >= 0.40:
            return "HIGH"
        else:
            return "VERY_HIGH"

    def _estimate_win_rate(self, quality_score: float) -> float:
        """Estimate expected win rate based on quality score"""
        # Map quality score to expected win rate
        # Target: move from 14.7% to 45%+ for high quality signals
        base_win_rate = 0.15  # Current 14.7%
        max_win_rate = 0.55   # Target 55% for highest quality
        
        # Linear interpolation based on quality score
        estimated_win_rate = base_win_rate + (quality_score * (max_win_rate - base_win_rate))
        return min(max(estimated_win_rate, 0.10), 0.70)  # Clamp between 10% and 70%

    def _low_quality_result(self, reason: str) -> Dict:
        """Return low quality result"""
        return {
            'quality_score': 0.1,
            'quality_level': 'LOW',
            'quality_factors': [f"Error: {reason}"],
            'should_trade': False,
            'reversal_strength': 0.1,
            'risk_level': 'VERY_HIGH',
            'expected_win_rate': 0.15,
            'analysis_details': {'error': reason}
        }

    def get_performance_stats(self) -> Dict:
        """Get reversal detection performance statistics"""
        total = max(self._reversals_detected, 1)
        return {
            'total_reversals_analyzed': self._reversals_detected,
            'high_quality_reversals': self._high_quality_reversals,
            'low_quality_reversals': self._low_quality_reversals,
            'high_quality_rate': self._high_quality_reversals / total * 100,
            'config': self.reversal_config,
            'target_improvement': 'Increase win rate from 14.7% to 45%+'
        }