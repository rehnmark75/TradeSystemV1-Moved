"""
Master Pattern HTF (Higher Timeframe) Trend Validator
=====================================================
Validates that Master Pattern signals align with the higher timeframe trend direction.

This is a CRITICAL filter that prevents counter-trend entries which are the main
cause of poor win rates in the Master Pattern strategy.

Logic:
- BULL signals require 4H showing bullish structure (HH/HL pattern)
- BEAR signals require 4H showing bearish structure (LH/LL pattern)
- Uses existing SMCTrendStructure helper for structure analysis
"""

import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class MasterPatternHTFValidator:
    """
    Validates Master Pattern signals against higher timeframe (4H) trend direction.

    This filter is modeled after the successful HTF alignment validation used in
    the SMC Structure Strategy which achieved 41.4% win rate and 1.32 profit factor.
    """

    def __init__(self, data_fetcher=None, logger=None):
        """
        Initialize the HTF validator.

        Args:
            data_fetcher: Data fetcher for retrieving HTF data
            logger: Logger instance
        """
        self.data_fetcher = data_fetcher
        self.logger = logger or logging.getLogger(__name__)

        # Import SMC trend structure helper
        try:
            from .smc_trend_structure import SMCTrendStructure
            self.trend_analyzer = SMCTrendStructure()
            self.logger.info("[HTF_VALIDATOR] SMCTrendStructure initialized for HTF analysis")
        except ImportError:
            self.trend_analyzer = None
            self.logger.warning("[HTF_VALIDATOR] SMCTrendStructure not available, using EMA-based fallback")

    def validate_htf_alignment(
        self,
        signal_direction: str,
        df_4h: pd.DataFrame,
        epic: str = '',
        min_strength: float = 0.40
    ) -> Tuple[bool, Dict]:
        """
        Validate that signal direction aligns with 4H trend structure.

        Args:
            signal_direction: 'BULL' or 'BEAR'
            df_4h: 4H timeframe DataFrame with OHLC data
            epic: Epic identifier for logging
            min_strength: Minimum trend strength to consider (0.0-1.0)

        Returns:
            Tuple of (is_aligned: bool, details: Dict)
        """
        pair = self._extract_pair_from_epic(epic)

        # Default response for invalid data
        default_response = {
            'htf_trend': 'NEUTRAL',
            'htf_strength': 0.0,
            'swing_pattern': 'UNKNOWN',
            'reason': 'Insufficient data'
        }

        if df_4h is None or len(df_4h) < 20:
            self.logger.warning(f"[{pair}] HTF validation: Insufficient 4H data ({len(df_4h) if df_4h is not None else 0} bars)")
            return True, default_response  # Allow signal if no HTF data

        try:
            # Analyze 4H trend using swing structure
            htf_trend, htf_strength, swing_pattern = self._analyze_htf_trend(df_4h, pair)

            details = {
                'htf_trend': htf_trend,
                'htf_strength': htf_strength,
                'swing_pattern': swing_pattern,
                'signal_direction': signal_direction,
                'reason': ''
            }

            # Check alignment
            if htf_strength < min_strength:
                # Weak trend - allow entry but note it
                details['reason'] = f'Weak HTF trend ({htf_strength:.0%}), entry allowed'
                self.logger.info(f"[{pair}] HTF: Weak trend ({htf_strength:.0%}), allowing {signal_direction} entry")
                return True, details

            # Strong trend - check alignment
            if signal_direction == 'BULL':
                is_aligned = htf_trend in ['BULLISH', 'NEUTRAL']
                if not is_aligned:
                    details['reason'] = f'BULL signal against BEARISH 4H trend ({htf_strength:.0%})'
                else:
                    details['reason'] = f'BULL signal aligned with {htf_trend} 4H trend'
            else:  # BEAR
                is_aligned = htf_trend in ['BEARISH', 'NEUTRAL']
                if not is_aligned:
                    details['reason'] = f'BEAR signal against BULLISH 4H trend ({htf_strength:.0%})'
                else:
                    details['reason'] = f'BEAR signal aligned with {htf_trend} 4H trend'

            if is_aligned:
                self.logger.info(f"[{pair}] HTF alignment PASSED: {signal_direction} with {htf_trend} ({htf_strength:.0%})")
            else:
                self.logger.info(f"[{pair}] HTF alignment REJECTED: {signal_direction} vs {htf_trend} ({htf_strength:.0%})")

            return is_aligned, details

        except Exception as e:
            self.logger.error(f"[{pair}] HTF validation error: {e}")
            return True, {**default_response, 'reason': f'Error: {str(e)}'}

    def _analyze_htf_trend(self, df: pd.DataFrame, pair: str) -> Tuple[str, float, str]:
        """
        Analyze 4H trend using swing structure analysis.

        Returns:
            Tuple of (trend_direction, trend_strength, swing_pattern)
        """
        # Try swing structure analysis first
        if self.trend_analyzer:
            trend, strength, pattern = self._analyze_with_smc_structure(df, pair)
            # If swing analysis is inconclusive, fall back to EMA
            if strength < 0.30 or pattern in ['INSUFFICIENT_SWINGS', 'NO_PATTERN']:
                self.logger.info(f"[{pair}] Swing analysis inconclusive ({pattern}), using EMA fallback")
                return self._analyze_with_ema_fallback(df, pair)
            return trend, strength, pattern
        else:
            return self._analyze_with_ema_fallback(df, pair)

    def _analyze_with_smc_structure(self, df: pd.DataFrame, pair: str) -> Tuple[str, float, str]:
        """
        Analyze trend using SMC swing structure (HH/HL/LH/LL patterns).
        """
        try:
            # Get recent swing points
            swings = self._identify_swings(df, lookback=20)

            if len(swings['highs']) < 2 or len(swings['lows']) < 2:
                return 'NEUTRAL', 0.0, 'INSUFFICIENT_SWINGS'

            # Analyze swing pattern
            recent_highs = swings['highs'][-3:]
            recent_lows = swings['lows'][-3:]

            # Check for HH/HL (bullish) or LH/LL (bearish)
            hh_count = 0
            hl_count = 0
            lh_count = 0
            ll_count = 0

            for i in range(1, len(recent_highs)):
                if recent_highs[i]['price'] > recent_highs[i-1]['price']:
                    hh_count += 1
                else:
                    lh_count += 1

            for i in range(1, len(recent_lows)):
                if recent_lows[i]['price'] > recent_lows[i-1]['price']:
                    hl_count += 1
                else:
                    ll_count += 1

            # Determine trend
            bullish_score = hh_count + hl_count
            bearish_score = lh_count + ll_count
            total_swings = bullish_score + bearish_score

            if total_swings == 0:
                return 'NEUTRAL', 0.0, 'NO_PATTERN'

            if bullish_score > bearish_score:
                strength = bullish_score / total_swings
                pattern = f'HH:{hh_count}/HL:{hl_count}'
                return 'BULLISH', strength, pattern
            elif bearish_score > bullish_score:
                strength = bearish_score / total_swings
                pattern = f'LH:{lh_count}/LL:{ll_count}'
                return 'BEARISH', strength, pattern
            else:
                return 'NEUTRAL', 0.5, 'MIXED'

        except Exception as e:
            self.logger.warning(f"[{pair}] SMC structure analysis error: {e}")
            return self._analyze_with_ema_fallback(df, pair)

    def _analyze_with_ema_fallback(self, df: pd.DataFrame, pair: str) -> Tuple[str, float, str]:
        """
        Fallback trend analysis using EMA alignment.
        """
        try:
            # Calculate EMAs
            ema_20 = df['close'].ewm(span=20, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()

            current_price = df['close'].iloc[-1]
            current_ema_20 = ema_20.iloc[-1]
            current_ema_50 = ema_50.iloc[-1]

            # Determine trend
            above_ema20 = current_price > current_ema_20
            above_ema50 = current_price > current_ema_50
            ema20_above_ema50 = current_ema_20 > current_ema_50

            bullish_signals = sum([above_ema20, above_ema50, ema20_above_ema50])

            if bullish_signals >= 3:
                return 'BULLISH', 0.80, 'EMA_ALIGNED_BULL'
            elif bullish_signals <= 0:
                return 'BEARISH', 0.80, 'EMA_ALIGNED_BEAR'
            elif bullish_signals == 2:
                return 'BULLISH', 0.60, 'EMA_PARTIAL_BULL'
            else:
                return 'BEARISH', 0.60, 'EMA_PARTIAL_BEAR'

        except Exception as e:
            self.logger.warning(f"[{pair}] EMA fallback analysis error: {e}")
            return 'NEUTRAL', 0.0, 'ERROR'

    def _identify_swings(self, df: pd.DataFrame, lookback: int = 20, left_bars: int = 3, right_bars: int = 3) -> Dict:
        """
        Identify swing highs and lows in the price data.
        """
        highs = []
        lows = []

        # Only analyze recent data
        data = df.tail(lookback * 2)

        for i in range(left_bars, len(data) - right_bars):
            # Check for swing high
            is_high = True
            for j in range(1, left_bars + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j]:
                    is_high = False
                    break
            if is_high:
                for j in range(1, right_bars + 1):
                    if data['high'].iloc[i] <= data['high'].iloc[i + j]:
                        is_high = False
                        break

            if is_high:
                highs.append({
                    'index': i,
                    'price': data['high'].iloc[i],
                    'timestamp': data.index[i] if hasattr(data.index, '__iter__') else i
                })

            # Check for swing low
            is_low = True
            for j in range(1, left_bars + 1):
                if data['low'].iloc[i] >= data['low'].iloc[i - j]:
                    is_low = False
                    break
            if is_low:
                for j in range(1, right_bars + 1):
                    if data['low'].iloc[i] >= data['low'].iloc[i + j]:
                        is_low = False
                        break

            if is_low:
                lows.append({
                    'index': i,
                    'price': data['low'].iloc[i],
                    'timestamp': data.index[i] if hasattr(data.index, '__iter__') else i
                })

        return {'highs': highs, 'lows': lows}

    def _extract_pair_from_epic(self, epic: str) -> str:
        """Extract pair name from epic string."""
        if not epic:
            return 'UNKNOWN'

        # Map common epic formats to pairs
        epic_map = {
            'CS.D.EURUSD': 'EURUSD',
            'CS.D.GBPUSD': 'GBPUSD',
            'CS.D.USDJPY': 'USDJPY',
            'CS.D.AUDUSD': 'AUDUSD',
            'CS.D.USDCAD': 'USDCAD',
            'CS.D.USDCHF': 'USDCHF',
            'CS.D.NZDUSD': 'NZDUSD',
            'CS.D.EURJPY': 'EURJPY',
            'CS.D.GBPJPY': 'GBPJPY',
        }

        for prefix, pair in epic_map.items():
            if epic.startswith(prefix):
                return pair

        return epic.split('.')[2] if '.' in epic else epic
