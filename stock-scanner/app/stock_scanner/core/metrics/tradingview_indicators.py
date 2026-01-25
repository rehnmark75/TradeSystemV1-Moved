"""
TradingView-Style Technical Indicators

Calculates oscillators and moving averages matching TradingView's methodology.
Includes signal classification (BUY/SELL/NEUTRAL) and summary aggregation.

Oscillators (11):
- RSI, Stochastic, CCI, ADX, Awesome Oscillator, Momentum
- MACD, Stochastic RSI, Williams %R, Bull Bear Power, Ultimate Oscillator

Moving Averages (14):
- EMA/SMA at 10, 20, 30, 50, 100, 200 periods
- Ichimoku Base Line, VWMA
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class TradingViewIndicators:
    """Calculate TradingView-style indicators and signals."""

    def __init__(self):
        self.oscillator_count = 11
        self.ma_count = 14

    # ===========================
    # OSCILLATOR CALCULATIONS
    # ===========================

    def calculate_stochastic(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate Stochastic Oscillator %K and %D.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K over d_period

        Returns:
            (stoch_k, stoch_d) or None if insufficient data
        """
        if len(closes) < k_period:
            return None

        try:
            # Calculate %K
            lowest_low = np.min(lows[-k_period:])
            highest_high = np.max(highs[-k_period:])

            if highest_high == lowest_low:
                return None

            k_value = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100

            # Calculate %D (SMA of %K - need at least d_period %K values)
            if len(closes) < k_period + d_period - 1:
                return (float(k_value), float(k_value))

            # Calculate multiple %K values for %D
            k_values = []
            for i in range(d_period):
                idx = -(d_period - i)
                if idx == 0:
                    idx = len(closes)
                low = np.min(lows[max(0, idx-k_period):idx or None])
                high = np.max(highs[max(0, idx-k_period):idx or None])
                if high != low:
                    k = ((closes[idx-1 if idx > 0 else -1] - low) / (high - low)) * 100
                    k_values.append(k)

            d_value = np.mean(k_values) if k_values else k_value

            return (float(k_value), float(d_value))

        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}")
            return None

    def calculate_cci(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 20
    ) -> Optional[float]:
        """
        Calculate Commodity Channel Index (CCI).

        Typical Price = (High + Low + Close) / 3
        CCI = (Typical Price - SMA of Typical Price) / (0.015 * Mean Deviation)

        Returns:
            CCI value or None if insufficient data
        """
        if len(closes) < period:
            return None

        try:
            # Calculate Typical Price
            typical_prices = (highs + lows + closes) / 3

            # SMA of typical price
            sma_tp = np.mean(typical_prices[-period:])

            # Mean Deviation
            mean_dev = np.mean(np.abs(typical_prices[-period:] - sma_tp))

            if mean_dev == 0:
                return None

            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_dev)

            return float(cci)

        except Exception as e:
            logger.warning(f"CCI calculation failed: {e}")
            return None

    def calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> Optional[Tuple[float, float, float]]:
        """
        Calculate Average Directional Index (ADX), +DI, and -DI.

        Returns:
            (adx, plus_di, minus_di) or None if insufficient data
        """
        if len(closes) < period + 1:
            return None

        try:
            # Calculate True Range and Directional Movement
            tr = []
            plus_dm = []
            minus_dm = []

            for i in range(1, len(closes)):
                high_diff = highs[i] - highs[i-1]
                low_diff = lows[i-1] - lows[i]

                # True Range
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr.append(max(tr1, tr2, tr3))

                # Directional Movement
                if high_diff > low_diff and high_diff > 0:
                    plus_dm.append(high_diff)
                else:
                    plus_dm.append(0)

                if low_diff > high_diff and low_diff > 0:
                    minus_dm.append(low_diff)
                else:
                    minus_dm.append(0)

            # Smooth with Wilder's method
            tr_smooth = np.mean(tr[-period:])
            plus_dm_smooth = np.mean(plus_dm[-period:])
            minus_dm_smooth = np.mean(minus_dm[-period:])

            # Calculate Directional Indicators
            plus_di = (plus_dm_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0
            minus_di = (minus_dm_smooth / tr_smooth) * 100 if tr_smooth > 0 else 0

            # Calculate ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0

            # ADX is smoothed DX
            if len(tr) >= period * 2:
                dx_values = []
                for i in range(period, len(tr)):
                    tr_s = np.mean(tr[i-period:i])
                    pdm_s = np.mean(plus_dm[i-period:i])
                    mdm_s = np.mean(minus_dm[i-period:i])

                    pdi = (pdm_s / tr_s) * 100 if tr_s > 0 else 0
                    mdi = (mdm_s / tr_s) * 100 if tr_s > 0 else 0

                    dx_val = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0
                    dx_values.append(dx_val)

                adx = np.mean(dx_values[-period:]) if dx_values else dx
            else:
                adx = dx

            return (float(adx), float(plus_di), float(minus_di))

        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")
            return None

    def calculate_awesome_oscillator(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        fast: int = 5,
        slow: int = 34
    ) -> Optional[float]:
        """
        Calculate Awesome Oscillator (AO).

        AO = SMA(Median Price, 5) - SMA(Median Price, 34)
        Median Price = (High + Low) / 2

        Returns:
            AO value or None if insufficient data
        """
        if len(highs) < slow:
            return None

        try:
            median_prices = (highs + lows) / 2

            sma_fast = np.mean(median_prices[-fast:])
            sma_slow = np.mean(median_prices[-slow:])

            ao = sma_fast - sma_slow

            return float(ao)

        except Exception as e:
            logger.warning(f"Awesome Oscillator calculation failed: {e}")
            return None

    def calculate_momentum(
        self,
        closes: np.ndarray,
        period: int = 10
    ) -> Optional[float]:
        """
        Calculate Momentum indicator.

        Momentum = Close - Close[period ago]

        Returns:
            Momentum value or None if insufficient data
        """
        if len(closes) <= period:
            return None

        try:
            momentum = closes[-1] - closes[-(period + 1)]
            return float(momentum)

        except Exception as e:
            logger.warning(f"Momentum calculation failed: {e}")
            return None

    def calculate_stochastic_rsi(
        self,
        closes: np.ndarray,
        rsi_period: int = 14,
        k_period: int = 3,
        d_period: int = 3
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate Stochastic RSI (StochRSI).

        1. Calculate RSI
        2. Apply Stochastic formula to RSI values
        3. Smooth with %K and %D periods

        Returns:
            (stoch_rsi_k, stoch_rsi_d) or None if insufficient data
        """
        if len(closes) < rsi_period + k_period + d_period:
            return None

        try:
            # Calculate RSI values
            rsi_values = []
            for i in range(len(closes) - rsi_period):
                deltas = np.diff(closes[i:i+rsi_period+1])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)

                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)

                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                rsi_values.append(rsi)

            if len(rsi_values) < k_period:
                return None

            # Apply Stochastic to RSI
            rsi_array = np.array(rsi_values)
            lowest_rsi = np.min(rsi_array[-k_period:])
            highest_rsi = np.max(rsi_array[-k_period:])

            if highest_rsi == lowest_rsi:
                return None

            stoch_rsi_k = ((rsi_array[-1] - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100

            # Calculate %D (SMA of %K)
            if len(rsi_array) >= k_period + d_period - 1:
                k_values = []
                for i in range(d_period):
                    idx = -(d_period - i)
                    if idx == 0:
                        idx = len(rsi_array)
                    low_rsi = np.min(rsi_array[max(0, idx-k_period):idx or None])
                    high_rsi = np.max(rsi_array[max(0, idx-k_period):idx or None])
                    if high_rsi != low_rsi:
                        k = ((rsi_array[idx-1 if idx > 0 else -1] - low_rsi) / (high_rsi - low_rsi)) * 100
                        k_values.append(k)

                stoch_rsi_d = np.mean(k_values) if k_values else stoch_rsi_k
            else:
                stoch_rsi_d = stoch_rsi_k

            return (float(stoch_rsi_k), float(stoch_rsi_d))

        except Exception as e:
            logger.warning(f"Stochastic RSI calculation failed: {e}")
            return None

    def calculate_williams_r(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14
    ) -> Optional[float]:
        """
        Calculate Williams %R.

        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

        Returns:
            Williams %R value (-100 to 0) or None if insufficient data
        """
        if len(closes) < period:
            return None

        try:
            highest_high = np.max(highs[-period:])
            lowest_low = np.min(lows[-period:])

            if highest_high == lowest_low:
                return None

            williams = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100

            return float(williams)

        except Exception as e:
            logger.warning(f"Williams %R calculation failed: {e}")
            return None

    def calculate_bull_bear_power(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        ema_period: int = 13
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate Bull Power and Bear Power.

        Bull Power = High - EMA(Close, 13)
        Bear Power = Low - EMA(Close, 13)

        Returns:
            (bull_power, bear_power) or None if insufficient data
        """
        if len(closes) < ema_period:
            return None

        try:
            # Calculate EMA of closes
            alpha = 2 / (ema_period + 1)
            ema = closes[0]
            for price in closes[1:]:
                ema = alpha * price + (1 - alpha) * ema

            bull_power = highs[-1] - ema
            bear_power = lows[-1] - ema

            return (float(bull_power), float(bear_power))

        except Exception as e:
            logger.warning(f"Bull Bear Power calculation failed: {e}")
            return None

    def calculate_ultimate_oscillator(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28
    ) -> Optional[float]:
        """
        Calculate Ultimate Oscillator.

        Multi-timeframe momentum oscillator using 3 periods (7, 14, 28).
        Weighted average: 4*Avg7 + 2*Avg14 + Avg28

        Returns:
            Ultimate Oscillator value (0-100) or None if insufficient data
        """
        if len(closes) < period3 + 1:
            return None

        try:
            # Calculate Buying Pressure and True Range
            bp_values = []
            tr_values = []

            for i in range(1, len(closes)):
                # Buying Pressure = Close - min(Low, Previous Close)
                bp = closes[i] - min(lows[i], closes[i-1])
                bp_values.append(bp)

                # True Range
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)

            # Calculate averages for each period
            def calc_avg(period):
                if len(bp_values) < period:
                    return None
                bp_sum = sum(bp_values[-period:])
                tr_sum = sum(tr_values[-period:])
                return bp_sum / tr_sum if tr_sum > 0 else 0

            avg1 = calc_avg(period1)
            avg2 = calc_avg(period2)
            avg3 = calc_avg(period3)

            if avg1 is None or avg2 is None or avg3 is None:
                return None

            # Weighted average
            uo = ((4 * avg1 + 2 * avg2 + avg3) / 7) * 100

            return float(uo)

        except Exception as e:
            logger.warning(f"Ultimate Oscillator calculation failed: {e}")
            return None

    # ===========================
    # MOVING AVERAGE CALCULATIONS
    # ===========================

    def calculate_ema(
        self,
        prices: np.ndarray,
        period: int
    ) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None

        try:
            alpha = 2 / (period + 1)
            ema = prices[0]
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            return float(ema)

        except Exception as e:
            logger.warning(f"EMA({period}) calculation failed: {e}")
            return None

    def calculate_sma(
        self,
        prices: np.ndarray,
        period: int
    ) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None

        try:
            return float(np.mean(prices[-period:]))
        except Exception as e:
            logger.warning(f"SMA({period}) calculation failed: {e}")
            return None

    def calculate_ichimoku_base(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        period: int = 26
    ) -> Optional[float]:
        """
        Calculate Ichimoku Base Line (Kijun-sen).

        Base Line = (Highest High + Lowest Low) / 2 over period

        Returns:
            Ichimoku Base Line value or None if insufficient data
        """
        if len(highs) < period:
            return None

        try:
            highest = np.max(highs[-period:])
            lowest = np.min(lows[-period:])
            base_line = (highest + lowest) / 2
            return float(base_line)

        except Exception as e:
            logger.warning(f"Ichimoku Base Line calculation failed: {e}")
            return None

    def calculate_vwma(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        period: int = 20
    ) -> Optional[float]:
        """
        Calculate Volume Weighted Moving Average.

        VWMA = Sum(Close * Volume) / Sum(Volume) over period

        Returns:
            VWMA value or None if insufficient data
        """
        if len(closes) < period or len(volumes) < period:
            return None

        try:
            pv = closes[-period:] * volumes[-period:]
            vwma = np.sum(pv) / np.sum(volumes[-period:]) if np.sum(volumes[-period:]) > 0 else None
            return float(vwma) if vwma is not None else None

        except Exception as e:
            logger.warning(f"VWMA calculation failed: {e}")
            return None

    # ===========================
    # SIGNAL CLASSIFICATION
    # ===========================

    def classify_rsi(self, rsi: Optional[float]) -> str:
        """Classify RSI signal: <30 Buy, >70 Sell, else Neutral."""
        if rsi is None:
            return 'NEUTRAL'
        if rsi < 30:
            return 'BUY'
        elif rsi > 70:
            return 'SELL'
        return 'NEUTRAL'

    def classify_stochastic(self, stoch_k: Optional[float]) -> str:
        """Classify Stochastic signal: <20 Buy, >80 Sell, else Neutral."""
        if stoch_k is None:
            return 'NEUTRAL'
        if stoch_k < 20:
            return 'BUY'
        elif stoch_k > 80:
            return 'SELL'
        return 'NEUTRAL'

    def classify_cci(self, cci: Optional[float]) -> str:
        """Classify CCI signal: <-100 Buy, >100 Sell, else Neutral."""
        if cci is None:
            return 'NEUTRAL'
        if cci < -100:
            return 'BUY'
        elif cci > 100:
            return 'SELL'
        return 'NEUTRAL'

    def classify_adx(
        self,
        adx: Optional[float],
        plus_di: Optional[float],
        minus_di: Optional[float]
    ) -> str:
        """Classify ADX signal: +DI > -DI with strong ADX = Buy, vice versa = Sell."""
        if adx is None or plus_di is None or minus_di is None:
            return 'NEUTRAL'

        # ADX must be strong enough (>25) to give a signal
        if adx < 25:
            return 'NEUTRAL'

        if plus_di > minus_di:
            return 'BUY'
        elif minus_di > plus_di:
            return 'SELL'
        return 'NEUTRAL'

    def classify_awesome_oscillator(self, ao: Optional[float], prev_ao: Optional[float] = None) -> str:
        """Classify AO signal: >0 Buy, <0 Sell (with rising/falling consideration)."""
        if ao is None:
            return 'NEUTRAL'

        if ao > 0:
            return 'BUY'
        elif ao < 0:
            return 'SELL'
        return 'NEUTRAL'

    def classify_momentum(self, momentum: Optional[float]) -> str:
        """Classify Momentum signal: >0 Buy, <0 Sell."""
        if momentum is None:
            return 'NEUTRAL'
        if momentum > 0:
            return 'BUY'
        elif momentum < 0:
            return 'SELL'
        return 'NEUTRAL'

    def classify_macd(
        self,
        macd: Optional[float],
        macd_signal: Optional[float]
    ) -> str:
        """Classify MACD signal: MACD > Signal = Buy, MACD < Signal = Sell."""
        if macd is None or macd_signal is None:
            return 'NEUTRAL'

        if macd > macd_signal:
            return 'BUY'
        elif macd < macd_signal:
            return 'SELL'
        return 'NEUTRAL'

    def classify_stochastic_rsi(self, stoch_rsi_k: Optional[float]) -> str:
        """Classify StochRSI signal: <20 Buy, >80 Sell."""
        if stoch_rsi_k is None:
            return 'NEUTRAL'
        if stoch_rsi_k < 20:
            return 'BUY'
        elif stoch_rsi_k > 80:
            return 'SELL'
        return 'NEUTRAL'

    def classify_williams_r(self, williams: Optional[float]) -> str:
        """Classify Williams %R signal: <-80 Buy, >-20 Sell."""
        if williams is None:
            return 'NEUTRAL'
        if williams < -80:
            return 'BUY'
        elif williams > -20:
            return 'SELL'
        return 'NEUTRAL'

    def classify_bull_bear_power(
        self,
        bull_power: Optional[float],
        bear_power: Optional[float]
    ) -> str:
        """Classify Bull Bear Power: Bull > 0 = Buy, Bear < 0 = Sell."""
        if bull_power is None or bear_power is None:
            return 'NEUTRAL'

        # Combine signals
        if bull_power > 0 and abs(bull_power) > abs(bear_power):
            return 'BUY'
        elif bear_power < 0 and abs(bear_power) > abs(bull_power):
            return 'SELL'
        return 'NEUTRAL'

    def classify_ultimate_oscillator(self, uo: Optional[float]) -> str:
        """Classify Ultimate Oscillator: <30 Buy, >70 Sell."""
        if uo is None:
            return 'NEUTRAL'
        if uo < 30:
            return 'BUY'
        elif uo > 70:
            return 'SELL'
        return 'NEUTRAL'

    def classify_ma(self, price: float, ma: Optional[float]) -> str:
        """Classify MA signal: Price > MA = Buy, Price < MA = Sell."""
        if ma is None:
            return 'NEUTRAL'

        # Consider within 0.1% as neutral
        threshold = ma * 0.001
        if price > ma + threshold:
            return 'BUY'
        elif price < ma - threshold:
            return 'SELL'
        return 'NEUTRAL'

    # ===========================
    # AGGREGATION
    # ===========================

    def aggregate_signals(self, signals: List[str]) -> Tuple[int, int, int, str]:
        """
        Aggregate a list of signals into counts and overall summary.

        Args:
            signals: List of 'BUY', 'SELL', 'NEUTRAL' strings

        Returns:
            (buy_count, sell_count, neutral_count, summary)
        """
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        neutral_count = signals.count('NEUTRAL')
        total = len(signals)

        if total == 0:
            return (0, 0, 0, 'NEUTRAL')

        buy_pct = buy_count / total
        sell_pct = sell_count / total

        # Summary classification
        if buy_pct > 0.7:
            summary = 'STRONG BUY'
        elif sell_pct > 0.7:
            summary = 'STRONG SELL'
        elif buy_count > sell_count and buy_pct > 0.4:
            summary = 'BUY'
        elif sell_count > buy_count and sell_pct > 0.4:
            summary = 'SELL'
        else:
            summary = 'NEUTRAL'

        return (buy_count, sell_count, neutral_count, summary)

    def calculate_overall_score(
        self,
        osc_buy: int,
        osc_sell: int,
        osc_neutral: int,
        ma_buy: int,
        ma_sell: int,
        ma_neutral: int
    ) -> float:
        """
        Calculate overall score from -100 to +100.

        Combines oscillators and moving averages with equal weight.

        Returns:
            Score from -100 (strong sell) to +100 (strong buy)
        """
        total_osc = osc_buy + osc_sell + osc_neutral
        total_ma = ma_buy + ma_sell + ma_neutral

        osc_score = 0
        ma_score = 0

        if total_osc > 0:
            osc_score = ((osc_buy - osc_sell) / total_osc) * 100

        if total_ma > 0:
            ma_score = ((ma_buy - ma_sell) / total_ma) * 100

        # Equal weight to oscillators and MAs
        overall = (osc_score + ma_score) / 2

        return round(overall, 2)

    # ===========================
    # MAIN CALCULATION METHOD
    # ===========================

    def calculate_all(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        current_price: float,
        existing_rsi: Optional[float] = None,
        existing_macd: Optional[float] = None,
        existing_macd_signal: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate all TradingView indicators and signals.

        Args:
            highs, lows, closes, volumes: Price data arrays
            current_price: Current stock price
            existing_rsi: Pre-calculated RSI (reuse if available)
            existing_macd: Pre-calculated MACD (reuse if available)
            existing_macd_signal: Pre-calculated MACD signal (reuse if available)

        Returns:
            Dictionary with all indicator values and summary counts
        """
        metrics = {}

        # ===== OSCILLATORS =====
        oscillator_signals = []

        # 1. RSI (reuse existing if available)
        rsi_signal = self.classify_rsi(existing_rsi)
        oscillator_signals.append(rsi_signal)

        # 2. Stochastic
        stoch = self.calculate_stochastic(highs, lows, closes)
        if stoch:
            metrics['stoch_k'], metrics['stoch_d'] = stoch
            oscillator_signals.append(self.classify_stochastic(stoch[0]))
        else:
            oscillator_signals.append('NEUTRAL')

        # 3. CCI
        cci = self.calculate_cci(highs, lows, closes)
        metrics['cci_20'] = cci
        oscillator_signals.append(self.classify_cci(cci))

        # 4. ADX
        adx_result = self.calculate_adx(highs, lows, closes)
        if adx_result:
            metrics['adx_14'], metrics['plus_di'], metrics['minus_di'] = adx_result
            oscillator_signals.append(self.classify_adx(*adx_result))
        else:
            oscillator_signals.append('NEUTRAL')

        # 5. Awesome Oscillator
        ao = self.calculate_awesome_oscillator(highs, lows)
        metrics['ao_value'] = ao
        oscillator_signals.append(self.classify_awesome_oscillator(ao))

        # 6. Momentum
        momentum = self.calculate_momentum(closes)
        metrics['momentum_10'] = momentum
        oscillator_signals.append(self.classify_momentum(momentum))

        # 7. MACD (reuse existing if available)
        macd_signal = self.classify_macd(existing_macd, existing_macd_signal)
        oscillator_signals.append(macd_signal)

        # 8. Stochastic RSI
        stoch_rsi = self.calculate_stochastic_rsi(closes)
        if stoch_rsi:
            metrics['stoch_rsi_k'], metrics['stoch_rsi_d'] = stoch_rsi
            oscillator_signals.append(self.classify_stochastic_rsi(stoch_rsi[0]))
        else:
            oscillator_signals.append('NEUTRAL')

        # 9. Williams %R
        williams = self.calculate_williams_r(highs, lows, closes)
        metrics['williams_r'] = williams
        oscillator_signals.append(self.classify_williams_r(williams))

        # 10. Bull Bear Power
        bbp = self.calculate_bull_bear_power(highs, lows, closes)
        if bbp:
            metrics['bull_power'], metrics['bear_power'] = bbp
            oscillator_signals.append(self.classify_bull_bear_power(*bbp))
        else:
            oscillator_signals.append('NEUTRAL')

        # 11. Ultimate Oscillator
        uo = self.calculate_ultimate_oscillator(highs, lows, closes)
        metrics['ultimate_osc'] = uo
        oscillator_signals.append(self.classify_ultimate_oscillator(uo))

        # ===== MOVING AVERAGES =====
        ma_signals = []

        # Calculate all MAs
        ma_configs = [
            ('ema_10', 10, True),
            ('sma_10', 10, False),
            ('ema_20', 20, True),  # May exist
            ('sma_20', 20, False),  # May exist
            ('ema_30', 30, True),
            ('sma_30', 30, False),
            ('ema_50', 50, True),
            ('sma_50', 50, False),  # May exist
            ('ema_100', 100, True),
            ('sma_100', 100, False),
            ('ema_200', 200, True),
            ('sma_200', 200, False),  # May exist
        ]

        for ma_name, period, is_ema in ma_configs:
            if is_ema:
                ma_value = self.calculate_ema(closes, period)
            else:
                ma_value = self.calculate_sma(closes, period)

            metrics[ma_name] = ma_value
            ma_signals.append(self.classify_ma(current_price, ma_value))

        # Ichimoku Base Line
        ichimoku = self.calculate_ichimoku_base(highs, lows)
        metrics['ichimoku_base'] = ichimoku
        ma_signals.append(self.classify_ma(current_price, ichimoku))

        # VWMA
        vwma = self.calculate_vwma(closes, volumes)
        metrics['vwma_20'] = vwma
        ma_signals.append(self.classify_ma(current_price, vwma))

        # ===== AGGREGATE SUMMARIES =====
        osc_buy, osc_sell, osc_neutral, osc_summary = self.aggregate_signals(oscillator_signals)
        ma_buy, ma_sell, ma_neutral, ma_summary = self.aggregate_signals(ma_signals)

        metrics['tv_osc_buy'] = osc_buy
        metrics['tv_osc_sell'] = osc_sell
        metrics['tv_osc_neutral'] = osc_neutral

        metrics['tv_ma_buy'] = ma_buy
        metrics['tv_ma_sell'] = ma_sell
        metrics['tv_ma_neutral'] = ma_neutral

        # Overall summary
        overall_signals = oscillator_signals + ma_signals
        _, _, _, overall_summary = self.aggregate_signals(overall_signals)
        overall_score = self.calculate_overall_score(
            osc_buy, osc_sell, osc_neutral,
            ma_buy, ma_sell, ma_neutral
        )

        metrics['tv_overall_signal'] = overall_summary
        metrics['tv_overall_score'] = overall_score

        return metrics
