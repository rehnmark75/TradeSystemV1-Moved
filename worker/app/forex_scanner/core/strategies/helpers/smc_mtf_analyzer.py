# core/strategies/helpers/smc_mtf_analyzer.py
"""
Smart Money Concepts Multi-Timeframe Analyzer Module
Handles multi-timeframe SMC validation for signal confirmation

Validates structure breaks, order blocks, and FVGs across higher timeframes:
- 15m: Near-term institutional structure (session-based)
- 4h: Macro structure and daily trend context

Based on established MTF patterns from MACD/Ichimoku strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta

try:
    from configdata import config
except ImportError:
    from forex_scanner.configdata import config


class SMCMultiTimeframeAnalyzer:
    """
    Multi-Timeframe validator for Smart Money Concepts strategy

    Validates SMC signals against higher timeframe structure:
    - Structure breaks (BOS/ChoCH) alignment
    - Order block presence and alignment
    - Fair value gap direction
    - Premium/discount zone context
    """

    def __init__(self, logger: logging.Logger = None, data_fetcher=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher

        # MTF configuration - OPTION B: 15m + 4h
        self.mtf_enabled = getattr(config, 'SMC_MTF_ENABLED', True)
        self.check_timeframes = ['15m', '4h']  # Near-term + Macro structure

        # Timeframe weights (near-term favored slightly)
        self.timeframe_weights = {
            '15m': 0.6,  # Intraday structure weight
            '4h': 0.4    # Macro structure weight
        }

        # Validation thresholds
        self.min_alignment_ratio = 0.5  # At least 1 of 2 timeframes must align
        self.require_both_for_high_confidence = True

        # Confidence boost values
        self.both_aligned_boost = 0.15      # Both 15m + 4h aligned
        self.htf_15m_only_boost = 0.05      # Only 15m aligned (weak)
        self.htf_4h_only_boost = 0.08       # Only 4h aligned (moderate)
        self.both_opposing_penalty = -0.20  # Conflicting timeframes

        # Caching system (5-minute cache for HTF data)
        self.cache_duration_minutes = 5
        self.mtf_cache = {}
        self.cache_timestamps = {}

        # HTF structure bias cache (for strict HTF-LTF alignment)
        self.htf_structure_bias_cache = {}  # Stores current 4h structure direction
        self.htf_structure_cache_time = {}

        self.logger.info(f"🔄 SMC MTF Analyzer initialized: {self.check_timeframes}")

    def is_mtf_enabled(self) -> bool:
        """Check if MTF analysis is enabled and data fetcher available"""
        return self.mtf_enabled and self.data_fetcher is not None

    def validate_higher_timeframe_smc(
        self,
        epic: str,
        current_time: pd.Timestamp,
        signal_type: str,
        structure_info: Dict
    ) -> Dict:
        """
        Validate SMC signal against higher timeframes

        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            structure_info: Dict with structure break details
                {
                    'break_type': 'BOS' or 'ChoCH',
                    'break_direction': 'bullish' or 'bearish',
                    'significance': float (0-1)
                }

        Returns:
            Dictionary with MTF validation results:
            {
                'mtf_enabled': bool,
                'validation_passed': bool,
                'confidence_boost': float,
                'timeframes_checked': List[str],
                'timeframes_aligned': List[str],
                'alignment_ratio': float,
                'details': Dict[str, Dict]  # Per-timeframe results
            }
        """
        if not self.is_mtf_enabled():
            return {
                'mtf_enabled': False,
                'validation_passed': True,  # Allow signal if MTF disabled
                'confidence_boost': 0.0,
                'message': 'MTF analysis disabled or unavailable'
            }

        try:
            results = {
                'mtf_enabled': True,
                'timeframes_checked': [],
                'timeframes_aligned': [],
                'validation_passed': False,
                'confidence_boost': 0.0,
                'alignment_ratio': 0.0,
                'details': {}
            }

            aligned_count = 0
            total_checked = 0
            alignment_scores = []

            # Check each higher timeframe
            for timeframe in self.check_timeframes:
                tf_result = self._check_timeframe_smc(
                    epic=epic,
                    timeframe=timeframe,
                    current_time=current_time,
                    signal_type=signal_type,
                    structure_info=structure_info
                )

                results['timeframes_checked'].append(timeframe)
                results['details'][timeframe] = tf_result
                total_checked += 1

                if tf_result.get('aligned', False):
                    results['timeframes_aligned'].append(timeframe)
                    aligned_count += 1
                    alignment_scores.append(tf_result.get('alignment_score', 0.5))

            # Calculate alignment ratio and validation result
            if total_checked > 0:
                results['alignment_ratio'] = aligned_count / total_checked

                # Validation passes if minimum alignment met
                results['validation_passed'] = results['alignment_ratio'] >= self.min_alignment_ratio

                # Calculate confidence boost based on alignment pattern
                results['confidence_boost'] = self._calculate_mtf_confidence_boost(
                    results['timeframes_aligned'],
                    alignment_scores
                )

            # Log MTF validation results
            if results['validation_passed']:
                self.logger.debug(
                    f"✅ MTF validation PASSED for {epic} {signal_type}: "
                    f"{aligned_count}/{total_checked} aligned "
                    f"(boost: {results['confidence_boost']:+.2f})"
                )
            else:
                self.logger.debug(
                    f"⚠️ MTF validation WEAK for {epic} {signal_type}: "
                    f"{aligned_count}/{total_checked} aligned"
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in SMC MTF validation: {e}")
            return {
                'mtf_enabled': True,
                'validation_passed': True,  # Allow signal on error
                'confidence_boost': 0.0,
                'error': str(e)
            }

    def _check_timeframe_smc(
        self,
        epic: str,
        timeframe: str,
        current_time: pd.Timestamp,
        signal_type: str,
        structure_info: Dict
    ) -> Dict:
        """
        Check SMC alignment for a specific higher timeframe

        Args:
            epic: Trading pair epic
            timeframe: Timeframe to check ('15m' or '4h')
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'
            structure_info: Structure break details from lower TF

        Returns:
            Dictionary with timeframe-specific results
        """
        try:
            # Check cache first
            cache_key = f"{epic}_{timeframe}_{signal_type}"
            if self._is_cache_valid(cache_key, current_time):
                cached_result = self.mtf_cache.get(cache_key)
                if cached_result:
                    self.logger.debug(f"Using cached MTF data for {cache_key}")
                    return cached_result

            # Fetch higher timeframe data
            htf_data = self._fetch_and_cache_htf_data(epic, timeframe, current_time)

            if htf_data is None or len(htf_data) < 20:
                return {
                    'aligned': False,
                    'reason': 'insufficient_data',
                    'data_length': len(htf_data) if htf_data is not None else 0,
                    'timeframe': timeframe
                }

            # Analyze HTF structure based on timeframe type
            if timeframe == '15m':
                result = self._analyze_15m_structure(
                    htf_data,
                    signal_type,
                    structure_info
                )
            elif timeframe == '4h':
                result = self._analyze_4h_structure(
                    htf_data,
                    signal_type,
                    structure_info
                )
            else:
                result = {'aligned': False, 'reason': 'unsupported_timeframe'}

            # Add metadata
            result['timeframe'] = timeframe
            result['data_length'] = len(htf_data)
            result['weight'] = self.timeframe_weights.get(timeframe, 0.5)

            # Cache the result
            self.mtf_cache[cache_key] = result
            self.cache_timestamps[cache_key] = current_time

            return result

        except Exception as e:
            self.logger.error(f"Error checking {timeframe} SMC for {epic}: {e}")
            return {
                'aligned': False,
                'error': str(e),
                'timeframe': timeframe
            }

    def _analyze_15m_structure(
        self,
        df: pd.DataFrame,
        signal_type: str,
        structure_info: Dict
    ) -> Dict:
        """
        Analyze 15m timeframe for near-term institutional structure

        Validates:
        - Recent structure breaks (last 4-8 hours)
        - Order blocks in current session
        - FVG alignment for intraday moves

        Args:
            df: 15m OHLCV DataFrame
            signal_type: 'BULL' or 'BEAR'
            structure_info: Lower TF structure break details

        Returns:
            Dict with 15m alignment analysis
        """
        try:
            # Get recent data (last 32 bars = 8 hours)
            recent_df = df.tail(32).copy()

            if len(recent_df) < 10:
                return {'aligned': False, 'reason': 'insufficient_recent_data'}

            alignment_factors = []
            alignment_score = 0.0

            # 1. Check for recent structure breaks alignment
            structure_aligned = self._check_htf_structure_breaks(
                recent_df,
                signal_type
            )
            if structure_aligned:
                alignment_factors.append('15m_structure_break')
                alignment_score += 0.4

            # 2. Check for supporting order blocks
            order_blocks = self._detect_htf_order_blocks(recent_df, signal_type)
            if order_blocks.get('supporting_ob_present', False):
                alignment_factors.append('15m_order_block')
                alignment_score += 0.3

            # 3. Check FVG alignment
            fvg_aligned = self._check_htf_fvg_alignment(recent_df, signal_type)
            if fvg_aligned:
                alignment_factors.append('15m_fvg')
                alignment_score += 0.2

            # 4. Volume profile check
            volume_confirmed = self._check_htf_volume_profile(recent_df, signal_type)
            if volume_confirmed:
                alignment_factors.append('15m_volume')
                alignment_score += 0.1

            # Determine if 15m is aligned (need at least 0.4 score)
            is_aligned = alignment_score >= 0.4

            return {
                'aligned': is_aligned,
                'alignment_score': alignment_score,
                'factors': alignment_factors,
                'structure_breaks': structure_aligned,
                'order_blocks': order_blocks,
                'fvg_aligned': fvg_aligned,
                'volume_confirmed': volume_confirmed
            }

        except Exception as e:
            self.logger.error(f"Error analyzing 15m structure: {e}")
            return {'aligned': False, 'error': str(e)}

    def _analyze_4h_structure(
        self,
        df: pd.DataFrame,
        signal_type: str,
        structure_info: Dict
    ) -> Dict:
        """
        Analyze 4h timeframe for macro structure and trend context

        Validates:
        - Macro trend structure (last 2-5 days = 12-30 bars)
        - Major order block zones
        - Premium/discount context
        - Daily range positioning

        Args:
            df: 4h OHLCV DataFrame
            signal_type: 'BULL' or 'BEAR'
            structure_info: Lower TF structure break details

        Returns:
            Dict with 4h alignment analysis
        """
        try:
            # Get recent data (last 30 bars = 5 days)
            recent_df = df.tail(30).copy()

            if len(recent_df) < 10:
                return {'aligned': False, 'reason': 'insufficient_recent_data'}

            alignment_factors = []
            alignment_score = 0.0

            # 1. Check macro trend direction
            trend_aligned = self._check_macro_trend(recent_df, signal_type)
            if trend_aligned:
                alignment_factors.append('4h_trend')
                alignment_score += 0.35

            # 2. Check for major structure breaks
            major_structure = self._check_htf_structure_breaks(recent_df, signal_type)
            if major_structure:
                alignment_factors.append('4h_major_structure')
                alignment_score += 0.25

            # 3. Check premium/discount zone context
            pd_context = self._check_premium_discount_context(recent_df, signal_type)
            if pd_context.get('favorable', False):
                alignment_factors.append('4h_premium_discount')
                alignment_score += 0.25

            # 4. Check for major order block zones
            major_obs = self._detect_htf_order_blocks(recent_df, signal_type)
            if major_obs.get('supporting_ob_present', False):
                alignment_factors.append('4h_major_ob')
                alignment_score += 0.15

            # Determine if 4h is aligned (need at least 0.4 score)
            is_aligned = alignment_score >= 0.4

            return {
                'aligned': is_aligned,
                'alignment_score': alignment_score,
                'factors': alignment_factors,
                'trend_aligned': trend_aligned,
                'major_structure': major_structure,
                'premium_discount': pd_context,
                'major_order_blocks': major_obs
            }

        except Exception as e:
            self.logger.error(f"Error analyzing 4h structure: {e}")
            return {'aligned': False, 'error': str(e)}

    def _check_htf_structure_breaks(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> bool:
        """
        Check for structure breaks on higher timeframe that align with signal

        Uses simple swing high/low detection to identify BOS/ChoCH patterns
        """
        try:
            if len(df) < 10:
                return False

            # Simple swing detection using rolling windows
            window = 3
            df_copy = df.copy()

            # Detect swing highs
            df_copy['swing_high'] = (
                (df_copy['high'] > df_copy['high'].shift(1)) &
                (df_copy['high'] > df_copy['high'].shift(-1)) &
                (df_copy['high'] == df_copy['high'].rolling(window*2+1, center=True).max())
            )

            # Detect swing lows
            df_copy['swing_low'] = (
                (df_copy['low'] < df_copy['low'].shift(1)) &
                (df_copy['low'] < df_copy['low'].shift(-1)) &
                (df_copy['low'] == df_copy['low'].rolling(window*2+1, center=True).min())
            )

            # Get recent swing highs and lows
            recent_highs = df_copy[df_copy['swing_high']]['high'].tail(3).tolist()
            recent_lows = df_copy[df_copy['swing_low']]['low'].tail(3).tolist()

            # Check for bullish structure (higher highs/lows)
            if signal_type == 'BULL':
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    higher_highs = recent_highs[-1] > recent_highs[-2]
                    higher_lows = recent_lows[-1] > recent_lows[-2]
                    return higher_highs or higher_lows  # Either indicates bullish structure

            # Check for bearish structure (lower highs/lows)
            elif signal_type == 'BEAR':
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    lower_highs = recent_highs[-1] < recent_highs[-2]
                    lower_lows = recent_lows[-1] < recent_lows[-2]
                    return lower_highs or lower_lows  # Either indicates bearish structure

            return False

        except Exception as e:
            self.logger.error(f"Error checking HTF structure breaks: {e}")
            return False

    def _detect_htf_order_blocks(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> Dict:
        """
        Detect order blocks on higher timeframe

        Order blocks are strong moves with high volume that create support/resistance
        """
        try:
            if len(df) < 10:
                return {'supporting_ob_present': False}

            # Get volume data
            volume_col = 'volume' if 'volume' in df.columns else 'ltv'
            if volume_col not in df.columns:
                return {'supporting_ob_present': False}

            # Calculate average volume
            avg_volume = df[volume_col].rolling(10).mean()
            high_volume = df[volume_col] > (avg_volume * 1.5)

            # Detect strong bullish moves (potential bullish OBs)
            strong_bullish = (
                (df['close'] > df['open']) &
                ((df['close'] - df['open']) / df['open'] > 0.003) &
                high_volume
            )

            # Detect strong bearish moves (potential bearish OBs)
            strong_bearish = (
                (df['close'] < df['open']) &
                ((df['open'] - df['close']) / df['open'] > 0.003) &
                high_volume
            )

            # Check for supporting order blocks
            recent_bullish_obs = strong_bullish.tail(5).sum()
            recent_bearish_obs = strong_bearish.tail(5).sum()

            if signal_type == 'BULL':
                supporting_ob = recent_bullish_obs > 0
            else:
                supporting_ob = recent_bearish_obs > 0

            return {
                'supporting_ob_present': supporting_ob,
                'bullish_obs': int(recent_bullish_obs),
                'bearish_obs': int(recent_bearish_obs)
            }

        except Exception as e:
            self.logger.error(f"Error detecting HTF order blocks: {e}")
            return {'supporting_ob_present': False}

    def _check_htf_fvg_alignment(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> bool:
        """
        Check if Fair Value Gaps on HTF align with signal direction

        FVG = gap between candle highs/lows indicating price imbalance
        """
        try:
            if len(df) < 5:
                return False

            # Detect bullish FVGs (current low > 2 candles ago high)
            bullish_fvg = (
                (df['low'] > df['high'].shift(2)) &
                ((df['low'] - df['high'].shift(2)) / df['high'].shift(2) > 0.001)
            )

            # Detect bearish FVGs (current high < 2 candles ago low)
            bearish_fvg = (
                (df['high'] < df['low'].shift(2)) &
                ((df['low'].shift(2) - df['high']) / df['low'].shift(2) > 0.001)
            )

            # Check recent FVGs (last 5 bars)
            recent_bullish_fvg = bullish_fvg.tail(5).any()
            recent_bearish_fvg = bearish_fvg.tail(5).any()

            if signal_type == 'BULL':
                return recent_bullish_fvg and not recent_bearish_fvg
            else:
                return recent_bearish_fvg and not recent_bullish_fvg

        except Exception as e:
            self.logger.error(f"Error checking HTF FVG alignment: {e}")
            return False

    def _check_htf_volume_profile(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> bool:
        """Check if volume profile supports the signal direction"""
        try:
            if len(df) < 5:
                return False

            volume_col = 'volume' if 'volume' in df.columns else 'ltv'
            if volume_col not in df.columns:
                return False

            recent_data = df.tail(5)

            # Calculate volume-weighted price movement
            recent_data_copy = recent_data.copy()
            recent_data_copy['price_change'] = recent_data_copy['close'] - recent_data_copy['open']
            recent_data_copy['weighted_change'] = recent_data_copy['price_change'] * recent_data_copy[volume_col]

            total_weighted_change = recent_data_copy['weighted_change'].sum()

            if signal_type == 'BULL':
                return total_weighted_change > 0
            else:
                return total_weighted_change < 0

        except Exception as e:
            self.logger.error(f"Error checking HTF volume profile: {e}")
            return False

    def _check_macro_trend(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> bool:
        """
        Check macro trend direction on 4h timeframe

        Uses simple price action analysis
        """
        try:
            if len(df) < 10:
                return False

            recent_data = df.tail(10)

            # Calculate trend using price progression
            closes = recent_data['close'].values

            # Count higher/lower closes
            up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
            down_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])

            # Simple trend determination
            if signal_type == 'BULL':
                return up_moves > down_moves
            else:
                return down_moves > up_moves

        except Exception as e:
            self.logger.error(f"Error checking macro trend: {e}")
            return False

    def _check_premium_discount_context(
        self,
        df: pd.DataFrame,
        signal_type: str
    ) -> Dict:
        """
        Check if current price is in premium or discount zone

        Premium zone: upper 30% of recent range (sell from here)
        Discount zone: lower 30% of recent range (buy from here)
        """
        try:
            if len(df) < 10:
                return {'favorable': False}

            recent_data = df.tail(20)

            # Get recent high and low
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            current_price = df['close'].iloc[-1]

            # Calculate range and current position
            price_range = recent_high - recent_low
            if price_range == 0:
                return {'favorable': False}

            position = (current_price - recent_low) / price_range

            # Determine zone
            if position >= 0.7:
                zone = 'premium'
                favorable = (signal_type == 'BEAR')  # Sell from premium
            elif position <= 0.3:
                zone = 'discount'
                favorable = (signal_type == 'BULL')  # Buy from discount
            else:
                zone = 'equilibrium'
                favorable = True  # Neutral zone, allow both

            return {
                'favorable': favorable,
                'zone': zone,
                'position': position,
                'recent_high': recent_high,
                'recent_low': recent_low
            }

        except Exception as e:
            self.logger.error(f"Error checking premium/discount context: {e}")
            return {'favorable': False}

    def _calculate_mtf_confidence_boost(
        self,
        aligned_timeframes: List[str],
        alignment_scores: List[float]
    ) -> float:
        """
        Calculate confidence boost based on MTF alignment pattern

        Args:
            aligned_timeframes: List of aligned timeframe strings
            alignment_scores: List of alignment scores (0-1) for each aligned TF

        Returns:
            Confidence boost value (-0.20 to +0.15)
        """
        try:
            num_aligned = len(aligned_timeframes)

            # Both 15m and 4h aligned = strong confluence
            if num_aligned == 2 and '15m' in aligned_timeframes and '4h' in aligned_timeframes:
                # Base boost plus bonus for high alignment scores
                avg_score = sum(alignment_scores) / len(alignment_scores)
                return self.both_aligned_boost + (avg_score * 0.05)  # Up to 0.20 total

            # Only 15m aligned = weak (intraday only)
            elif num_aligned == 1 and '15m' in aligned_timeframes:
                return self.htf_15m_only_boost

            # Only 4h aligned = moderate (macro only)
            elif num_aligned == 1 and '4h' in aligned_timeframes:
                return self.htf_4h_only_boost

            # No alignment = penalty
            elif num_aligned == 0:
                return self.both_opposing_penalty

            # Fallback
            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating MTF confidence boost: {e}")
            return 0.0

    def _fetch_and_cache_htf_data(
        self,
        epic: str,
        timeframe: str,
        current_time: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """
        Fetch higher timeframe data with caching

        Args:
            epic: Trading pair epic
            timeframe: Timeframe to fetch ('15m', '4h', etc.)
            current_time: Current timestamp

        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            if not self.data_fetcher:
                return None

            # Calculate lookback hours based on timeframe
            if timeframe == '15m':
                lookback_hours = 10   # 40 bars * 15min = 10 hours
            elif timeframe == '4h':
                lookback_hours = 144  # 36 bars * 4h = 144 hours (~6 days)
            elif timeframe == '1h':
                lookback_hours = 60   # 60 bars * 1h = 60 hours (~2.5 days)
            elif timeframe == '1d':
                lookback_hours = 720  # 30 bars * 24h = 720 hours (~30 days)
            else:
                lookback_hours = 48   # Default fallback

            # Extract pair name from epic (e.g., CS.D.EURUSD.CEEM.IP -> EURUSD)
            pair_name = epic.split('.')[2] if '.' in epic else epic

            # Fetch data using get_enhanced_data (central method)
            # Note: 4h and other timeframes are automatically resampled from 5m by data_fetcher
            data = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair_name,
                timeframe=timeframe,
                lookback_hours=lookback_hours,
                required_indicators=['open', 'high', 'low', 'close', 'volume', 'ltv']
            )

            if data is not None and len(data) > 0:
                self.logger.debug(f"Fetched {len(data)} bars for {epic} {timeframe}")
                return data
            else:
                self.logger.warning(f"No HTF data returned for {epic} {timeframe}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching HTF data for {epic} {timeframe}: {e}")
            return None

    def _is_cache_valid(self, cache_key: str, current_time: pd.Timestamp) -> bool:
        """Check if cached MTF data is still valid"""
        if cache_key not in self.cache_timestamps:
            return False

        try:
            cached_time = self.cache_timestamps[cache_key]
            age_minutes = (current_time - cached_time).total_seconds() / 60
            return age_minutes < self.cache_duration_minutes
        except Exception:
            return False

    def clear_cache(self):
        """Clear MTF cache (useful for testing or manual refresh)"""
        self.mtf_cache.clear()
        self.cache_timestamps.clear()
        self.htf_structure_bias_cache.clear()
        self.htf_structure_cache_time.clear()
        self.logger.debug("SMC MTF cache cleared")

    def get_htf_structure_bias(
        self,
        epic: str,
        current_time: pd.Timestamp,
        timeframe: str = '4h'
    ) -> Optional[Dict]:
        """
        Get Higher Timeframe structure bias for directional filtering

        This is the core of the enhanced MTF strategy:
        - Analyzes 4h for BOS/ChoCH to establish directional bias
        - Returns current market structure direction
        - Caches result for performance

        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            timeframe: Higher timeframe to check (default '4h')

        Returns:
            Dictionary with HTF structure bias:
            {
                'bias_direction': 'bullish' | 'bearish' | 'neutral',
                'structure_type': 'BOS' | 'ChoCH' | None,
                'significance': float (0-1),
                'last_break_price': float,
                'order_blocks_present': bool,
                'fvg_zones': List[Dict],  # FVG zones for confluence
                'premium_discount_zone': str  # 'premium', 'discount', 'equilibrium'
            }
        """
        if not self.is_mtf_enabled():
            return None

        try:
            # Check cache first
            cache_key = f"{epic}_{timeframe}_bias"
            if self._is_cache_valid(cache_key, current_time):
                cached_bias = self.htf_structure_bias_cache.get(cache_key)
                if cached_bias:
                    self.logger.debug(f"Using cached HTF structure bias for {epic} {timeframe}")
                    return cached_bias

            # Fetch HTF data
            htf_data = self._fetch_and_cache_htf_data(epic, timeframe, current_time)

            if htf_data is None or len(htf_data) < 30:
                return None

            # Analyze HTF structure for bias
            bias_result = self._analyze_htf_structure_bias(htf_data, epic, timeframe)

            # Cache the result
            self.htf_structure_bias_cache[cache_key] = bias_result
            self.htf_structure_cache_time[cache_key] = current_time

            if bias_result and bias_result.get('bias_direction') != 'neutral':
                self.logger.info(
                    f"📊 HTF Structure Bias ({timeframe}): {bias_result['bias_direction'].upper()} "
                    f"({bias_result.get('structure_type', 'N/A')}, "
                    f"significance: {bias_result.get('significance', 0):.2f})"
                )

            return bias_result

        except Exception as e:
            self.logger.error(f"Error getting HTF structure bias: {e}")
            return None

    def _analyze_htf_structure_bias(
        self,
        df: pd.DataFrame,
        epic: str,
        timeframe: str
    ) -> Dict:
        """
        Analyze HTF data to determine current structure bias

        Looks for recent BOS/ChoCH patterns to establish directional bias
        """
        try:
            if len(df) < 20:
                return {'bias_direction': 'neutral'}

            # Use rolling window to detect swing points
            window = 5 if timeframe == '4h' else 3
            df_copy = df.copy()

            # Detect swing highs
            df_copy['swing_high'] = (
                (df_copy['high'] > df_copy['high'].shift(1)) &
                (df_copy['high'] > df_copy['high'].shift(-1)) &
                (df_copy['high'] == df_copy['high'].rolling(window*2+1, center=True).max())
            )

            # Detect swing lows
            df_copy['swing_low'] = (
                (df_copy['low'] < df_copy['low'].shift(1)) &
                (df_copy['low'] < df_copy['low'].shift(-1)) &
                (df_copy['low'] == df_copy['low'].rolling(window*2+1, center=True).min())
            )

            # Get recent swing highs and lows
            swing_highs = df_copy[df_copy['swing_high']]['high'].tail(5).tolist()
            swing_lows = df_copy[df_copy['swing_low']]['low'].tail(5).tolist()

            # Determine structure bias
            bias_direction = 'neutral'
            structure_type = None
            significance = 0.0
            last_break_price = None

            # Check for bullish structure (Higher Highs + Higher Lows)
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Recent structure analysis
                higher_highs = swing_highs[-1] > swing_highs[-2]
                higher_lows = swing_lows[-1] > swing_lows[-2]
                lower_highs = swing_highs[-1] < swing_highs[-2]
                lower_lows = swing_lows[-1] < swing_lows[-2]

                # Bullish BOS: Both HH and HL
                if higher_highs and higher_lows:
                    bias_direction = 'bullish'
                    structure_type = 'BOS'
                    significance = 0.8
                    last_break_price = swing_highs[-1]

                # Bullish ChoCH: HL after previous downtrend
                elif higher_lows and not higher_highs:
                    # Check if previous structure was bearish
                    if len(swing_highs) >= 3 and swing_highs[-2] < swing_highs[-3]:
                        bias_direction = 'bullish'
                        structure_type = 'ChoCH'
                        significance = 0.7
                        last_break_price = swing_lows[-1]

                # Bearish BOS: Both LH and LL
                elif lower_highs and lower_lows:
                    bias_direction = 'bearish'
                    structure_type = 'BOS'
                    significance = 0.8
                    last_break_price = swing_lows[-1]

                # Bearish ChoCH: LH after previous uptrend
                elif lower_highs and not lower_lows:
                    # Check if previous structure was bullish
                    if len(swing_lows) >= 3 and swing_lows[-2] > swing_lows[-3]:
                        bias_direction = 'bearish'
                        structure_type = 'ChoCH'
                        significance = 0.7
                        last_break_price = swing_highs[-1]

            # Detect order blocks on HTF
            order_blocks_present = self._detect_htf_order_blocks(df_copy, bias_direction).get('supporting_ob_present', False)

            # Detect FVG zones
            fvg_zones = self._detect_htf_fvg_zones(df_copy, bias_direction)

            # Get premium/discount context
            pd_context = self._check_premium_discount_context(df_copy, bias_direction)

            return {
                'bias_direction': bias_direction,
                'structure_type': structure_type,
                'significance': significance,
                'last_break_price': last_break_price,
                'order_blocks_present': order_blocks_present,
                'fvg_zones': fvg_zones,
                'premium_discount_zone': pd_context.get('zone', 'equilibrium'),
                'premium_discount_favorable': pd_context.get('favorable', False),
                'swing_highs': swing_highs,
                'swing_lows': swing_lows
            }

        except Exception as e:
            self.logger.error(f"HTF structure bias analysis failed: {e}")
            return {'bias_direction': 'neutral'}

    def _detect_htf_fvg_zones(self, df: pd.DataFrame, bias_direction: str) -> List[Dict]:
        """
        Detect Fair Value Gap zones on higher timeframe

        Returns list of FVG zones that align with bias direction
        """
        try:
            if len(df) < 5:
                return []

            fvg_zones = []

            # Detect bullish FVGs (current low > 2 candles ago high)
            if bias_direction == 'bullish':
                for i in range(2, len(df)):
                    current_low = df.iloc[i]['low']
                    prev_high = df.iloc[i-2]['high']

                    if current_low > prev_high:
                        gap_size = current_low - prev_high
                        if gap_size / prev_high > 0.001:  # 0.1% minimum gap
                            fvg_zones.append({
                                'type': 'bullish',
                                'high': current_low,
                                'low': prev_high,
                                'size': gap_size,
                                'index': i
                            })

            # Detect bearish FVGs (current high < 2 candles ago low)
            elif bias_direction == 'bearish':
                for i in range(2, len(df)):
                    current_high = df.iloc[i]['high']
                    prev_low = df.iloc[i-2]['low']

                    if current_high < prev_low:
                        gap_size = prev_low - current_high
                        if gap_size / current_high > 0.001:  # 0.1% minimum gap
                            fvg_zones.append({
                                'type': 'bearish',
                                'high': prev_low,
                                'low': current_high,
                                'size': gap_size,
                                'index': i
                            })

            # Return only recent FVGs (last 10 bars)
            recent_fvgs = [fvg for fvg in fvg_zones if fvg['index'] >= len(df) - 10]

            return recent_fvgs

        except Exception as e:
            self.logger.error(f"HTF FVG zone detection failed: {e}")
            return []

    def validate_ltf_entry_against_htf(
        self,
        epic: str,
        current_time: pd.Timestamp,
        ltf_signal_type: str,
        ltf_structure_info: Dict,
        htf_timeframe: str = '4h'
    ) -> Dict:
        """
        Validate LTF (15m) entry signal against HTF (4h) structure bias

        This implements the core enhancement:
        - Check if 4h has bullish/bearish structure
        - Validate that 15m signal aligns with 4h bias
        - Reject signals that go against HTF structure

        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            ltf_signal_type: 'BULL' or 'BEAR' from 15m
            ltf_structure_info: Structure break info from 15m
            htf_timeframe: Higher timeframe to validate against (default '4h')

        Returns:
            Dictionary with validation results:
            {
                'htf_ltf_aligned': bool,
                'htf_bias': str,
                'alignment_quality': str,  # 'strong', 'moderate', 'weak', 'conflicting'
                'confluence_factors': List[str],
                'additional_boost': float  # Extra confidence boost for alignment
            }
        """
        try:
            # Get HTF structure bias
            htf_bias_result = self.get_htf_structure_bias(epic, current_time, htf_timeframe)

            if not htf_bias_result:
                # No HTF data available - allow signal but no boost
                return {
                    'htf_ltf_aligned': True,
                    'htf_bias': 'unknown',
                    'alignment_quality': 'weak',
                    'confluence_factors': [],
                    'additional_boost': 0.0,
                    'reason': 'HTF data unavailable'
                }

            htf_bias = htf_bias_result['bias_direction']
            htf_significance = htf_bias_result.get('significance', 0.5)

            # Determine LTF direction
            ltf_direction = 'bullish' if ltf_signal_type == 'BULL' else 'bearish'

            # Check alignment
            aligned = False
            alignment_quality = 'conflicting'
            additional_boost = 0.0
            confluence_factors = []

            if htf_bias == ltf_direction:
                # Perfect alignment!
                aligned = True
                alignment_quality = 'strong' if htf_significance >= 0.7 else 'moderate'

                # Base boost for alignment
                additional_boost = 0.10 if htf_significance >= 0.7 else 0.05
                confluence_factors.append(f'{htf_timeframe}_structure_aligned')

                # Check additional confluence factors

                # 1. Order blocks present on HTF
                if htf_bias_result.get('order_blocks_present', False):
                    additional_boost += 0.05
                    confluence_factors.append(f'{htf_timeframe}_order_block')

                # 2. FVG zones present
                if htf_bias_result.get('fvg_zones', []):
                    additional_boost += 0.04
                    confluence_factors.append(f'{htf_timeframe}_fvg')

                # 3. Premium/Discount favorable
                if htf_bias_result.get('premium_discount_favorable', False):
                    additional_boost += 0.06
                    confluence_factors.append(f'{htf_timeframe}_premium_discount_zone')

                # 4. Strong HTF structure (BOS vs ChoCH)
                if htf_bias_result.get('structure_type') == 'BOS':
                    additional_boost += 0.03
                    confluence_factors.append(f'{htf_timeframe}_bos_continuation')
                elif htf_bias_result.get('structure_type') == 'ChoCH':
                    additional_boost += 0.02
                    confluence_factors.append(f'{htf_timeframe}_choch_reversal')

            elif htf_bias == 'neutral':
                # Neutral HTF - allow but no boost
                aligned = True
                alignment_quality = 'weak'
                additional_boost = 0.0
                confluence_factors.append('htf_neutral')

            else:
                # Conflicting direction - REJECT
                aligned = False
                alignment_quality = 'conflicting'
                additional_boost = -0.15  # Penalty for going against HTF
                confluence_factors.append('htf_conflicting')

            result = {
                'htf_ltf_aligned': aligned,
                'htf_bias': htf_bias,
                'htf_structure_type': htf_bias_result.get('structure_type'),
                'htf_significance': htf_significance,
                'alignment_quality': alignment_quality,
                'confluence_factors': confluence_factors,
                'additional_boost': additional_boost,
                'reason': f'{htf_timeframe} {htf_bias} vs {ltf_direction} signal'
            }

            # Log the validation result
            if aligned and alignment_quality == 'strong':
                self.logger.info(
                    f"✅ HTF-LTF ALIGNED: {htf_timeframe} {htf_bias} + 15m {ltf_direction} "
                    f"(boost: {additional_boost:+.2f}, factors: {len(confluence_factors)})"
                )
            elif not aligned:
                self.logger.warning(
                    f"❌ HTF-LTF CONFLICT: {htf_timeframe} {htf_bias} vs 15m {ltf_direction} "
                    f"(penalty: {additional_boost:+.2f})"
                )

            return result

        except Exception as e:
            self.logger.error(f"Error validating LTF against HTF: {e}")
            return {
                'htf_ltf_aligned': True,
                'htf_bias': 'error',
                'alignment_quality': 'weak',
                'confluence_factors': [],
                'additional_boost': 0.0,
                'reason': f'Validation error: {str(e)}'
            }

    def get_mtf_summary(self, validation_result: Dict) -> str:
        """
        Get human-readable summary of MTF validation results

        Args:
            validation_result: Result from validate_higher_timeframe_smc()

        Returns:
            String summary for logging/display
        """
        try:
            if not validation_result.get('mtf_enabled', False):
                return "MTF disabled"

            aligned_tfs = validation_result.get('timeframes_aligned', [])
            checked_tfs = validation_result.get('timeframes_checked', [])
            boost = validation_result.get('confidence_boost', 0.0)

            if len(aligned_tfs) == len(checked_tfs) and len(aligned_tfs) > 0:
                return f"✅ All TFs aligned ({', '.join(aligned_tfs)}) +{boost:.2f}"
            elif len(aligned_tfs) > 0:
                return f"⚠️ Partial alignment ({', '.join(aligned_tfs)}) {boost:+.2f}"
            else:
                return f"❌ No alignment {boost:+.2f}"

        except Exception:
            return "MTF summary unavailable"


def create_smc_mtf_analyzer(data_fetcher=None, logger=None) -> SMCMultiTimeframeAnalyzer:
    """
    Factory function to create SMC MTF analyzer instance

    Args:
        data_fetcher: Data fetcher instance for getting HTF data
        logger: Logger instance

    Returns:
        Configured SMCMultiTimeframeAnalyzer instance
    """
    return SMCMultiTimeframeAnalyzer(logger=logger, data_fetcher=data_fetcher)
