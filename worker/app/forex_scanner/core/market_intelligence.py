# ============================================================================
# core/market_intelligence.py - Market Intelligence Engine
# ============================================================================

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from backtests.backtest_base import MarketRegime, TradingSession
except ImportError:
    try:
        from forex_scanner.backtests.backtest_base import MarketRegime, TradingSession
    except ImportError:
        # Fallback definitions if backtest_base not available
        from enum import Enum

        class MarketRegime(Enum):
            TRENDING_UP = "trending_up"
            TRENDING_DOWN = "trending_down"
            RANGING = "ranging"
            HIGH_VOLATILITY = "high_volatility"
            LOW_VOLATILITY = "low_volatility"
            BREAKOUT = "breakout"
            REVERSAL = "reversal"
            UNKNOWN = "unknown"

        class TradingSession(Enum):
            ASIAN = "asian"
            LONDON = "london"
            NEW_YORK = "new_york"
            OVERLAP_LONDON_NY = "london_ny_overlap"
            OVERNIGHT = "overnight"


@dataclass
class MarketContext:
    """Extended market context with intelligence metrics"""
    regime: MarketRegime
    regime_strength: float  # 0.0-1.0
    regime_duration: int    # bars since regime started
    volatility_percentile: float
    trend_quality: float   # 0.0-1.0
    momentum_strength: float
    support_resistance_quality: float
    session: TradingSession
    volume_profile: Dict[str, float]
    correlation_strength: float  # with major pairs
    news_impact_score: float = 0.0


class MarketIntelligenceEngine:
    """Advanced market analysis and regime detection engine"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Regime detection parameters
        self.trend_lookback = 50
        self.volatility_lookback = 100
        self.regime_confirmation_bars = 5

        # Caching for performance
        self._regime_cache = {}
        self._volatility_cache = {}

        self.logger.info("🧠 MarketIntelligenceEngine initialized")

    def detect_regime(self, data: pd.DataFrame, epic: str = "UNKNOWN") -> MarketRegime:
        """Detect current market regime using multi-factor analysis"""
        try:
            if len(data) < self.trend_lookback:
                return MarketRegime.UNKNOWN

            # Generate cache key
            cache_key = f"{epic}_{len(data)}_{data.index[-1] if len(data) > 0 else 'empty'}"

            if cache_key in self._regime_cache:
                return self._regime_cache[cache_key]

            # Multi-factor regime analysis
            trend_analysis = self._analyze_trend_regime(data)
            volatility_analysis = self._analyze_volatility_regime(data)
            momentum_analysis = self._analyze_momentum_regime(data)
            structure_analysis = self._analyze_structure_regime(data)

            # Combine factors with weights
            regime_scores = {
                MarketRegime.TRENDING_UP: 0.0,
                MarketRegime.TRENDING_DOWN: 0.0,
                MarketRegime.RANGING: 0.0,
                MarketRegime.HIGH_VOLATILITY: 0.0,
                MarketRegime.LOW_VOLATILITY: 0.0,
                MarketRegime.BREAKOUT: 0.0,
                MarketRegime.REVERSAL: 0.0
            }

            # Weight the different analyses
            weights = {
                'trend': 0.35,
                'volatility': 0.25,
                'momentum': 0.25,
                'structure': 0.15
            }

            # Apply trend analysis
            for regime, score in trend_analysis.items():
                regime_scores[regime] += score * weights['trend']

            # Apply volatility analysis
            for regime, score in volatility_analysis.items():
                regime_scores[regime] += score * weights['volatility']

            # Apply momentum analysis
            for regime, score in momentum_analysis.items():
                regime_scores[regime] += score * weights['momentum']

            # Apply structure analysis
            for regime, score in structure_analysis.items():
                regime_scores[regime] += score * weights['structure']

            # Select regime with highest score
            detected_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[detected_regime]

            # Require minimum confidence for non-UNKNOWN regime
            if confidence < 0.3:
                detected_regime = MarketRegime.UNKNOWN

            # Cache result
            self._regime_cache[cache_key] = detected_regime

            self.logger.debug(f"🎯 Regime detected for {epic}: {detected_regime.value} (confidence: {confidence:.2f})")

            return detected_regime

        except Exception as e:
            self.logger.warning(f"⚠️ Regime detection failed: {e}")
            return MarketRegime.UNKNOWN

    def _analyze_trend_regime(self, data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Analyze trend-based regime characteristics"""
        scores = {regime: 0.0 for regime in MarketRegime}

        try:
            if len(data) < 20:
                return scores

            # Calculate trend indicators
            close = data['close']

            # EMA-based trend analysis
            ema_fast = close.ewm(span=10).mean()
            ema_medium = close.ewm(span=20).mean()
            ema_slow = close.ewm(span=50).mean()

            # Current trend direction
            current_price = close.iloc[-1]
            fast_vs_medium = (ema_fast.iloc[-1] - ema_medium.iloc[-1]) / ema_medium.iloc[-1]
            medium_vs_slow = (ema_medium.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
            price_vs_slow = (current_price - ema_slow.iloc[-1]) / ema_slow.iloc[-1]

            # Trend strength calculation
            trend_alignment = 0
            if ema_fast.iloc[-1] > ema_medium.iloc[-1] > ema_slow.iloc[-1]:
                trend_alignment = 1  # Uptrend
            elif ema_fast.iloc[-1] < ema_medium.iloc[-1] < ema_slow.iloc[-1]:
                trend_alignment = -1  # Downtrend

            # ADX for trend strength
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()

            # Simplified ADX calculation
            dm_plus = (data['high'] - data['high'].shift(1)).clip(lower=0)
            dm_minus = (data['low'].shift(1) - data['low']).clip(lower=0)
            dm_plus[dm_plus < dm_minus] = 0
            dm_minus[dm_minus < dm_plus] = 0

            di_plus = (dm_plus.rolling(14).mean() / atr) * 100
            di_minus = (dm_minus.rolling(14).mean() / atr) * 100
            adx_raw = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx = adx_raw.rolling(14).mean()

            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20

            # Score regimes based on trend analysis
            if trend_alignment > 0 and current_adx > 25:
                scores[MarketRegime.TRENDING_UP] = min(current_adx / 50.0, 1.0)
            elif trend_alignment < 0 and current_adx > 25:
                scores[MarketRegime.TRENDING_DOWN] = min(current_adx / 50.0, 1.0)
            elif current_adx < 20:
                scores[MarketRegime.RANGING] = 1.0 - (current_adx / 20.0)

            # Detect potential reversals
            if abs(fast_vs_medium) > 0.005 and abs(medium_vs_slow) < 0.002:
                scores[MarketRegime.REVERSAL] = 0.6

            return scores

        except Exception as e:
            self.logger.warning(f"⚠️ Trend analysis failed: {e}")
            return scores

    def _analyze_volatility_regime(self, data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Analyze volatility-based regime characteristics"""
        scores = {regime: 0.0 for regime in MarketRegime}

        try:
            if len(data) < 20:
                return scores

            # Calculate volatility metrics
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()

            # Volatility percentile
            atr_window = min(self.volatility_lookback, len(atr))
            current_atr = atr.iloc[-1]
            atr_percentile = (atr.tail(atr_window) < current_atr).sum() / atr_window

            # Bollinger Bands for volatility context
            bb_period = 20
            bb_std = 2
            sma = data['close'].rolling(bb_period).mean()
            std = data['close'].rolling(bb_period).std()
            bb_upper = sma + (std * bb_std)
            bb_lower = sma - (std * bb_std)
            bb_width = (bb_upper - bb_lower) / sma
            bb_width_percentile = (bb_width.tail(50) < bb_width.iloc[-1]).sum() / min(50, len(bb_width))

            # Score volatility regimes
            if atr_percentile > 0.8 or bb_width_percentile > 0.8:
                scores[MarketRegime.HIGH_VOLATILITY] = min(atr_percentile, bb_width_percentile)
            elif atr_percentile < 0.2 or bb_width_percentile < 0.2:
                scores[MarketRegime.LOW_VOLATILITY] = 1.0 - max(atr_percentile, bb_width_percentile)

            # Breakout detection based on volatility expansion
            if len(bb_width) > 5:
                recent_expansion = bb_width.iloc[-1] / bb_width.iloc[-5]
                if recent_expansion > 1.5:
                    scores[MarketRegime.BREAKOUT] = min(recent_expansion / 2.0, 1.0)

            return scores

        except Exception as e:
            self.logger.warning(f"⚠️ Volatility analysis failed: {e}")
            return scores

    def _analyze_momentum_regime(self, data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Analyze momentum-based regime characteristics"""
        scores = {regime: 0.0 for regime in MarketRegime}

        try:
            if len(data) < 30:
                return scores

            close = data['close']

            # RSI for momentum
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # MACD for momentum
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            macd_histogram = macd_line - macd_signal

            # Rate of Change
            roc = ((close.iloc[-1] / close.iloc[-10]) - 1) * 100 if len(close) > 10 else 0

            # Score momentum regimes
            if current_rsi > 70 and macd_histogram.iloc[-1] > 0 and roc > 2:
                scores[MarketRegime.TRENDING_UP] += 0.7
            elif current_rsi < 30 and macd_histogram.iloc[-1] < 0 and roc < -2:
                scores[MarketRegime.TRENDING_DOWN] += 0.7

            # Momentum divergence signals reversal
            if len(macd_histogram) > 5:
                momentum_change = macd_histogram.iloc[-1] - macd_histogram.iloc[-5]
                price_change = close.iloc[-1] - close.iloc[-5]

                if (momentum_change > 0 and price_change < 0) or (momentum_change < 0 and price_change > 0):
                    scores[MarketRegime.REVERSAL] += 0.5

            return scores

        except Exception as e:
            self.logger.warning(f"⚠️ Momentum analysis failed: {e}")
            return scores

    def _analyze_structure_regime(self, data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Analyze market structure regime characteristics"""
        scores = {regime: 0.0 for regime in MarketRegime}

        try:
            if len(data) < 50:
                return scores

            close = data['close']
            high = data['high']
            low = data['low']

            # Support and resistance levels
            recent_highs = high.rolling(window=10).max()
            recent_lows = low.rolling(window=10).min()

            # Price position relative to recent range
            range_size = recent_highs.iloc[-1] - recent_lows.iloc[-1]
            current_position = (close.iloc[-1] - recent_lows.iloc[-1]) / range_size if range_size > 0 else 0.5

            # Higher highs and lower lows analysis
            swing_highs = high.rolling(window=5).max()
            swing_lows = low.rolling(window=5).min()

            # Count recent higher highs/lower lows
            recent_bars = min(20, len(data))
            higher_highs = 0
            lower_lows = 0

            for i in range(-recent_bars, -1):
                if i < -1:
                    if swing_highs.iloc[i] < swing_highs.iloc[i+1]:
                        higher_highs += 1
                    if swing_lows.iloc[i] > swing_lows.iloc[i+1]:
                        lower_lows += 1

            # Structure-based scoring
            if higher_highs > lower_lows and current_position > 0.6:
                scores[MarketRegime.TRENDING_UP] += 0.6
            elif lower_lows > higher_highs and current_position < 0.4:
                scores[MarketRegime.TRENDING_DOWN] += 0.6
            elif abs(higher_highs - lower_lows) <= 1:
                scores[MarketRegime.RANGING] += 0.8

            # Breakout structure detection
            if current_position > 0.9 or current_position < 0.1:
                scores[MarketRegime.BREAKOUT] += 0.5

            return scores

        except Exception as e:
            self.logger.warning(f"⚠️ Structure analysis failed: {e}")
            return scores

    def get_market_context(self, data: pd.DataFrame, epic: str = "UNKNOWN") -> MarketContext:
        """Get comprehensive market context analysis"""
        try:
            regime = self.detect_regime(data, epic)

            # Calculate additional context metrics
            volatility_percentile = self._calculate_volatility_percentile(data)
            trend_quality = self._calculate_trend_quality(data)
            momentum_strength = self._calculate_momentum_strength(data)
            sr_quality = self._calculate_support_resistance_quality(data)
            session = self._get_current_session()
            volume_profile = self._analyze_volume_profile(data)
            correlation = self._estimate_correlation_strength()

            return MarketContext(
                regime=regime,
                regime_strength=0.7,  # Placeholder - could be enhanced
                regime_duration=10,   # Placeholder - could be enhanced
                volatility_percentile=volatility_percentile,
                trend_quality=trend_quality,
                momentum_strength=momentum_strength,
                support_resistance_quality=sr_quality,
                session=session,
                volume_profile=volume_profile,
                correlation_strength=correlation
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Market context analysis failed: {e}")
            return MarketContext(
                regime=MarketRegime.UNKNOWN,
                regime_strength=0.0,
                regime_duration=0,
                volatility_percentile=0.5,
                trend_quality=0.5,
                momentum_strength=0.5,
                support_resistance_quality=0.5,
                session=TradingSession.OVERNIGHT,
                volume_profile={},
                correlation_strength=0.5
            )

    def _calculate_volatility_percentile(self, data: pd.DataFrame, window: int = 100) -> float:
        """Calculate volatility percentile"""
        try:
            if len(data) < window:
                return 0.5

            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()

            if len(atr) < window:
                return 0.5

            current_atr = atr.iloc[-1]
            atr_window = atr.tail(window)
            percentile = (atr_window < current_atr).sum() / len(atr_window)

            return min(max(percentile, 0.0), 1.0)
        except:
            return 0.5

    def _calculate_trend_quality(self, data: pd.DataFrame) -> float:
        """Calculate trend quality score"""
        try:
            if len(data) < 20:
                return 0.5

            close = data['close']
            ema_10 = close.ewm(span=10).mean()
            ema_20 = close.ewm(span=20).mean()

            # Trend consistency
            trend_consistency = 0
            for i in range(-10, 0):
                if ema_10.iloc[i] > ema_20.iloc[i]:
                    trend_consistency += 1

            return trend_consistency / 10.0
        except:
            return 0.5

    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength"""
        try:
            if len(data) < 14:
                return 0.5

            close = data['close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            # Convert RSI to momentum strength (0.5 = neutral, 1.0 = strong momentum)
            if current_rsi > 50:
                return 0.5 + (current_rsi - 50) / 100.0
            else:
                return 0.5 - (50 - current_rsi) / 100.0
        except:
            return 0.5

    def _calculate_support_resistance_quality(self, data: pd.DataFrame) -> float:
        """Calculate support/resistance quality"""
        try:
            if len(data) < 20:
                return 0.5

            # Simple implementation - could be enhanced
            high = data['high']
            low = data['low']

            # Look for level touches
            recent_high = high.tail(20).max()
            recent_low = low.tail(20).min()

            high_touches = (high.tail(20) >= recent_high * 0.999).sum()
            low_touches = (low.tail(20) <= recent_low * 1.001).sum()

            return min((high_touches + low_touches) / 10.0, 1.0)
        except:
            return 0.5

    def _get_current_session(self) -> TradingSession:
        """Get current trading session"""
        try:
            current_hour = datetime.now().hour

            if 22 <= current_hour or current_hour < 8:
                return TradingSession.ASIAN
            elif 8 <= current_hour < 13:
                return TradingSession.LONDON
            elif 13 <= current_hour < 16:
                return TradingSession.OVERLAP_LONDON_NY
            elif 16 <= current_hour < 22:
                return TradingSession.NEW_YORK
            else:
                return TradingSession.OVERNIGHT
        except:
            return TradingSession.OVERNIGHT

    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume profile (placeholder implementation)"""
        try:
            # Placeholder - real implementation would need volume data
            return {
                'volume_trend': 0.5,
                'volume_breakout': 0.0,
                'average_volume': 1000.0
            }
        except:
            return {}

    def _estimate_correlation_strength(self) -> float:
        """Estimate correlation with major pairs (placeholder)"""
        try:
            # Placeholder - real implementation would compare with major pairs
            return 0.7
        except:
            return 0.5

    def clear_cache(self):
        """Clear analysis cache"""
        self._regime_cache.clear()
        self._volatility_cache.clear()
        self.logger.info("🗑️ Market intelligence cache cleared")