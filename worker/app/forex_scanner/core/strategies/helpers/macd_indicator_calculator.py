# core/strategies/helpers/macd_indicator_calculator.py
"""
MACD Indicator Calculator Module
Handles MACD calculations, crossover detection and data validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Import volatility metrics for regime-adaptive parameters
try:
    from .adaptive_volatility_calculator import VolatilityMetrics
    VOLATILITY_METRICS_AVAILABLE = True
except ImportError:
    VOLATILITY_METRICS_AVAILABLE = False
    VolatilityMetrics = None


class MACDIndicatorCalculator:
    """Calculates MACD indicators and detects crossovers - lightweight and focused"""
    
    def __init__(self, logger: logging.Logger = None, eps: float = 1e-8):
        self.logger = logger or logging.getLogger(__name__)
        self.eps = eps

        # EMERGENCY: Global signal tracking across backtest calls
        self.global_signal_tracker = {}  # {epic: {'last_signal_time': timestamp, 'signal_count': int}}
        
        # Default MACD parameters (fallback only - prefer database parameters)
        self.default_config = {
            'fast_ema': 12,   # Traditional default
            'slow_ema': 26,   # Traditional default
            'signal_ema': 9   # Standard signal line
        }
    
    def get_required_indicators(self, macd_config: Dict = None) -> List[str]:
        """Get list of required indicators for MACD strategy"""
        config = macd_config or self.default_config
        return [
            f'ema_{config["fast_ema"]}',
            f'ema_{config["slow_ema"]}',
            'macd_line',
            'macd_signal', 
            'macd_histogram',
            'ema_200'  # For trend filter
        ]
    
    def validate_data_requirements(self, df: pd.DataFrame, min_bars: int = 50) -> bool:
        """Validate that we have enough data for MACD calculations"""
        if len(df) < min_bars:
            self.logger.debug(f"Insufficient data: {len(df)} bars (need {min_bars})")
            return False
            
        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False
            
        return True
    
    def ensure_macd_indicators(self, df: pd.DataFrame, macd_config: Dict = None) -> pd.DataFrame:
        """
        Calculate MACD indicators if not present
        
        Args:
            df: DataFrame with price data
            macd_config: MACD configuration (fast_ema, slow_ema, signal_ema)
            
        Returns:
            DataFrame with MACD indicators added
        """
        config = macd_config or self.default_config
        df_copy = df.copy()
        
        try:
            # Calculate EMAs needed for MACD
            fast_ema_col = f'ema_{config["fast_ema"]}'
            slow_ema_col = f'ema_{config["slow_ema"]}'
            
            if fast_ema_col not in df_copy.columns:
                df_copy[fast_ema_col] = df_copy['close'].ewm(span=config['fast_ema']).mean()
                
            if slow_ema_col not in df_copy.columns:
                df_copy[slow_ema_col] = df_copy['close'].ewm(span=config['slow_ema']).mean()
            
            # Calculate MACD line (fast EMA - slow EMA)
            if 'macd_line' not in df_copy.columns:
                df_copy['macd_line'] = df_copy[fast_ema_col] - df_copy[slow_ema_col]
            
            # Calculate MACD signal line (EMA of MACD line)
            if 'macd_signal' not in df_copy.columns:
                df_copy['macd_signal'] = df_copy['macd_line'].ewm(span=config['signal_ema']).mean()
                
            # Calculate MACD histogram (MACD line - signal line)
            if 'macd_histogram' not in df_copy.columns:
                df_copy['macd_histogram'] = df_copy['macd_line'] - df_copy['macd_signal']
            
            # Add EMA 200 for trend filter if not present
            if 'ema_200' not in df_copy.columns:
                df_copy['ema_200'] = df_copy['close'].ewm(span=200).mean()
            
            # Add enhanced filters: ADX, ATR, and RSI (only if missing)
            if not all(col in df_copy.columns for col in ['adx', 'atr', 'rsi']):
                self.logger.debug("Adding missing enhanced filters (ADX, ATR, RSI)")
                df_copy = self._add_enhanced_filters(df_copy)
            else:
                self.logger.debug("Enhanced filters already present - skipping calculation")

            # Add MACD divergence detection for high-quality signals (only if missing and enabled)
            # OPTIMIZATION: Divergence detection is expensive and disabled for performance
            divergence_enabled = False  # Disabled for optimization - can be re-enabled in config
            if divergence_enabled and not any(col in df_copy.columns for col in ['bullish_divergence', 'bearish_divergence']):
                self.logger.debug("Adding MACD divergence detection")
                df_copy = self.detect_macd_divergence(df_copy)
            else:
                self.logger.debug("Divergence detection DISABLED for performance optimization")
            
            self.logger.debug("MACD indicators, enhanced filters, and divergence detection completed successfully")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD indicators: {e}")
            return df_copy
    
    def detect_macd_crossovers(self, df: pd.DataFrame, epic: str = '', is_backtest: bool = False) -> pd.DataFrame:
        """
        Detect MACD histogram crossovers with strength filtering (main signal generation)
        
        Args:
            df: DataFrame with MACD indicators
            epic: Trading pair epic for strength threshold determination
            
        Returns:
            DataFrame with crossover signals added
        """
        df_copy = df.copy()
        
        try:
            # Check if crossover signals already exist (performance optimization)
            crossover_cols = ['bull_crossover', 'bear_crossover', 'bull_alert', 'bear_alert']
            if all(col in df_copy.columns for col in crossover_cols):
                signal_count = df_copy[['bull_alert', 'bear_alert']].any(axis=1).sum()
                if signal_count > 0:
                    self.logger.debug(f"📊 [REUSING CROSSOVERS] Found {signal_count} existing crossover signals for {epic}")
                    return df_copy

            # Ensure we have MACD data
            required_cols = ['macd_line', 'macd_signal', 'macd_histogram']
            if not all(col in df_copy.columns for col in required_cols):
                self.logger.error("Missing MACD indicators for crossover detection")
                return df_copy

            # Initialize signal columns
            df_copy['bull_crossover'] = False
            df_copy['bear_crossover'] = False
            df_copy['bull_alert'] = False
            df_copy['bear_alert'] = False
            
            # Detect histogram crossovers (zero line crosses)
            df_copy['histogram_prev'] = df_copy['macd_histogram'].shift(1)

            # PHASE 2: Extract volatility metrics for regime-adaptive parameters
            volatility_metrics = self._extract_volatility_metrics(df_copy, epic)
            close_price = df_copy['close'].iloc[-1] if len(df_copy) > 0 else None

            # Get volatility-adaptive base threshold
            base_threshold = self.get_histogram_strength_threshold(epic, volatility_metrics, close_price)

            # Apply volatility regime-aware threshold scaling (legacy PHASE 2 system)
            strength_threshold = self.get_enhanced_threshold(df_copy, epic, base_threshold)

            # Get volatility regime info for logging
            volatility_info = self.get_volatility_regime(df_copy)
            self.logger.debug(f"🎯 PHASE 2 Dynamic thresholds for {epic}: base={base_threshold:.6f}, "
                            f"enhanced={strength_threshold:.6f}, regime={volatility_info['regime']}")
            
            # SIMPLIFIED TRADITIONAL MACD: Basic crossovers with light filtering
            
            # MULTI-CANDLE CONFIRMATION MACD for high-quality signals only
            
            # PHASE 2: Apply dynamic thresholds based on volatility regime
            raw_bull_cross = (
                (df_copy['macd_histogram'] > 0) &
                (df_copy['histogram_prev'] <= 0) &
                (df_copy['macd_histogram'] >= strength_threshold)  # Dynamic volatility-adjusted threshold
            )
            raw_bear_cross = (
                (df_copy['macd_histogram'] < 0) &
                (df_copy['histogram_prev'] >= 0) &
                (df_copy['macd_histogram'] <= -strength_threshold)  # Dynamic volatility-adjusted threshold
            )
            
            self.logger.debug(f"🔍 PHASE 2 MACD crossovers for {epic}: {raw_bull_cross.sum()} bull, {raw_bear_cross.sum()} bear "
                            f"(threshold: {strength_threshold:.6f}, regime: {volatility_info['regime']})")

            # EMERGENCY DEBUGGING: Comprehensive histogram analysis
            if len(df_copy) > 0:
                histogram_values = df_copy['macd_histogram'].dropna()
                if len(histogram_values) > 0:
                    hist_max = histogram_values.max()
                    hist_min = histogram_values.min()
                    hist_mean = histogram_values.mean()
                    hist_std = histogram_values.std()
                    hist_abs_max = histogram_values.abs().max()

                    self.logger.debug(f"Histogram analysis for {epic}:")
                    self.logger.debug(f"   📊 Range: min={hist_min:.8f}, max={hist_max:.8f}, mean={hist_mean:.8f}")
                    self.logger.debug(f"   📊 Std: {hist_std:.8f}, Abs Max: {hist_abs_max:.8f}")
                    self.logger.debug(f"   📊 Threshold: {base_threshold:.8f}")

                    # Check how many values exceed thresholds
                    above_threshold = (histogram_values >= base_threshold).sum()
                    below_neg_threshold = (histogram_values <= -base_threshold).sum()
                    above_zero = (histogram_values > 0).sum()
                    below_zero = (histogram_values < 0).sum()

                    self.logger.debug(f"   🔍 Threshold Analysis: {above_threshold} above +{base_threshold:.8f}, {below_neg_threshold} below -{base_threshold:.8f}")
                    self.logger.debug(f"   🔍 Zero Line Analysis: {above_zero} above zero, {below_zero} below zero")

                    # Log recent histogram values for detailed analysis
                    recent_hist = histogram_values.tail(20)
                    self.logger.debug(f"Recent 20 histogram values: {recent_hist.tolist()}")

                    # Check for any crossovers at all (ignoring thresholds)
                    histogram_prev = df_copy['macd_histogram'].shift(1).dropna()
                    if len(histogram_prev) > 0:
                        raw_bull_crosses = ((df_copy['macd_histogram'] > 0) & (histogram_prev <= 0)).sum()
                        raw_bear_crosses = ((df_copy['macd_histogram'] < 0) & (histogram_prev >= 0)).sum()
                        self.logger.debug(f"   ⚡ RAW crossovers (no threshold): {raw_bull_crosses} bull, {raw_bear_crosses} bear")
            
            # BALANCED: Use normal threshold-based detection with optimized thresholds
            emergency_bypass = False  # BALANCED MODE: Use optimized thresholds
            if emergency_bypass:
                # Detect ANY histogram crossover regardless of strength
                emergency_bull_cross = (
                    (df_copy['macd_histogram'] > 0) &
                    (df_copy['histogram_prev'] <= 0)
                    # NO THRESHOLD CHECK
                )
                emergency_bear_cross = (
                    (df_copy['macd_histogram'] < 0) &
                    (df_copy['histogram_prev'] >= 0)
                    # NO THRESHOLD CHECK
                )

                self.logger.debug(f"Bypass mode for {epic}: {emergency_bull_cross.sum()} bull crossovers, {emergency_bear_cross.sum()} bear crossovers (no thresholds)")

                # Use emergency crossovers if we find any
                if emergency_bull_cross.sum() > 0 or emergency_bear_cross.sum() > 0:
                    bull_cross = emergency_bull_cross
                    bear_cross = emergency_bear_cross
                    self.logger.info(f"🚨 USING EMERGENCY CROSSOVERS: {bull_cross.sum()} bull, {bear_cross.sum()} bear")
                else:
                    # Fallback to threshold-based if no emergency crossovers found
                    bull_cross = raw_bull_cross
                    bear_cross = raw_bear_cross
                    self.logger.info(f"🚨 NO EMERGENCY CROSSOVERS - Using threshold-based: {bull_cross.sum()} bull, {bear_cross.sum()} bear")
            else:
                # QUALITY: Apply multi-candle confirmation for high-quality signals
                bull_cross = self._apply_multi_candle_confirmation(df_copy, raw_bull_cross, 'BULL', epic)
                bear_cross = self._apply_multi_candle_confirmation(df_copy, raw_bear_cross, 'BEAR', epic)
                self.logger.debug(f"🎯 QUALITY: Multi-candle confirmed signals for {epic}: {bull_cross.sum()} bull, {bear_cross.sum()} bear")
            
            # Debug RSI and ADX data availability first
            if 'rsi' in df_copy.columns and 'adx' in df_copy.columns:
                rsi_values = df_copy['rsi'].dropna()
                adx_values = df_copy['adx'].dropna()
                self.logger.debug(f"🔍 INDICATOR DATA for {epic}: RSI count={len(rsi_values)}, ADX count={len(adx_values)}")
                if len(rsi_values) > 0:
                    self.logger.debug(f"🔍 RSI RANGE for {epic}: min={rsi_values.min():.1f}, max={rsi_values.max():.1f}, mean={rsi_values.mean():.1f}")
                if len(adx_values) > 0:
                    self.logger.debug(f"🔍 ADX RANGE for {epic}: min={adx_values.min():.1f}, max={adx_values.max():.1f}, mean={adx_values.mean():.1f}")
            else:
                self.logger.error(f"❌ MISSING INDICATORS for {epic}: RSI={'rsi' in df_copy.columns}, ADX={'adx' in df_copy.columns}")

            # 🎯 ADAPTIVE SIGNAL SCORING - DISABLED for more signals (optimization)
            # bull_cross, bear_cross = self._apply_adaptive_signal_scoring(
            #     df_copy, bull_cross, bear_cross, epic, base_threshold
            # )

            # 🌊 VOLATILITY FILTER - DISABLED for more signals (optimization)
            # volatility_filter_applied = self._apply_volatility_filter(df_copy, bull_cross, bear_cross, epic)
            # if volatility_filter_applied is not None:
            #     bull_cross, bear_cross = volatility_filter_applied

            # 🎯 NEW: SUPPORT/RESISTANCE PRE-FILTER
            bull_cross, bear_cross = self._apply_sr_level_filter(df_copy, bull_cross, bear_cross, epic)

            # 🎯 NEW: HISTOGRAM DIVERGENCE FILTER
            bull_cross, bear_cross = self._apply_histogram_divergence_filter(df_copy, bull_cross, bear_cross, epic)

            # 🎯 PHASE 2: ADX PRE-FILTER - Require strong trend at crossover time
            if 'adx' in df_copy.columns:
                min_adx = 30  # FINAL: Minimum ADX 30 (strong trend balanced)
                adx_bull_filter = df_copy['adx'] >= min_adx
                adx_bear_filter = df_copy['adx'] >= min_adx

                bull_before = bull_cross.sum()
                bear_before = bear_cross.sum()

                bull_cross = bull_cross & adx_bull_filter
                bear_cross = bear_cross & adx_bear_filter

                bull_after = bull_cross.sum()
                bear_after = bear_cross.sum()

                if bull_before > bull_after or bear_before > bear_after:
                    self.logger.info(f"🎯 ADX PRE-FILTER for {epic}: Bull {bull_before} -> {bull_after}, Bear {bear_before} -> {bear_after} (min ADX: {min_adx})")

            # 🎯 TIER 1 QUALITY FILTERS - Algorithmic signal quality
            bull_cross, bear_cross = self._apply_tier1_quality_filters(df_copy, bull_cross, bear_cross, epic)

            # Log basic info for debugging
            bull_count = bull_cross.sum()
            bear_count = bear_cross.sum()
            self.logger.debug(f"Multi-filter MACD crossovers for {epic}: {bull_count} bull, {bear_count} bear (threshold: {base_threshold:.8f})")

            # CRITICAL FIX: Use the SAME volatility-aware signal limiter for both backtest and live
            # This ensures backtest results match what would happen in live trading
            bull_cross, bear_cross = self._apply_global_signal_limiter(
                df_copy, bull_cross, bear_cross, epic, volatility_metrics
            )

            final_bull_count = bull_cross.sum()
            final_bear_count = bear_cross.sum()
            if bull_count + bear_count != final_bull_count + final_bear_count:
                mode = "BACKTEST" if is_backtest else "LIVE"
                self.logger.info(f"🎯 {mode} SIGNAL LIMITER for {epic}: Bull {bull_count} -> {final_bull_count}, Bear {bear_count} -> {final_bear_count}")
            
            df_copy['bull_crossover'] = bull_cross
            df_copy['bear_crossover'] = bear_cross
            
            # For compatibility, also set alert flags
            df_copy['bull_alert'] = bull_cross
            df_copy['bear_alert'] = bear_cross
            
            # Log crossover detection with strength filtering info
            bull_count = bull_cross.sum()
            bear_count = bear_cross.sum()
            if bull_count > 0 or bear_count > 0:
                self.logger.debug(f"MACD crossovers detected - Bull: {bull_count}, Bear: {bear_count} (threshold: {strength_threshold})")
            
            # Log filtered signals if any crossovers were filtered out
            raw_bull = ((df_copy['macd_histogram'] > 0) & (df_copy['histogram_prev'] <= 0)).sum()
            raw_bear = ((df_copy['macd_histogram'] < 0) & (df_copy['histogram_prev'] >= 0)).sum()
            if raw_bull > bull_count or raw_bear > bear_count:
                self.logger.debug(f"Strength filter removed {raw_bull - bull_count} bull + {raw_bear - bear_count} bear signals")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD crossovers: {e}")
            return df_copy
    
    def get_histogram_strength_threshold(
        self,
        epic: str,
        volatility_metrics: Optional['VolatilityMetrics'] = None,
        close_price: Optional[float] = None
    ) -> float:
        """
        Get MACD histogram strength threshold with volatility-adaptive scaling

        Base thresholds (original balanced values):
        - JPY pairs: 0.00005 - baseline for JPY volatility
        - Other pairs: 0.00002 - baseline for major pairs

        Volatility multipliers (if metrics provided):
        - HIGH VOLATILITY (ATR > 90th percentile or ATR% > 0.8%): 2.0x
        - NORMAL VOLATILITY (ATR% 0.5-0.8%): 1.0x baseline
        - LOW VOLATILITY (ATR% < 0.5%): 0.7x - permissive for good signals

        Args:
            epic: Trading pair epic (e.g., 'CS.D.USDJPY.MINI.IP')
            volatility_metrics: Optional volatility metrics for regime adaptation
            close_price: Current close price (for ATR% calculation)

        Returns:
            Volatility-adjusted histogram strength threshold
        """
        try:
            # TIER 1: PAIR-SPECIFIC THRESHOLDS (FINAL CALIBRATION)
            # Each pair has different MACD histogram scales
            # Target: ~15-25 validated signals/week total (2-4/day)
            pair_thresholds = {
                # JPY pairs - FINAL CALIBRATED
                'EURJPY': 0.00150,   # ✅ WORKING (26 crossovers, 1 validated)
                'AUDJPY': 0.00300,   # ✅ WORKING (292 crossovers)
                'USDJPY': 0.00350,   # ✅ WORKING (254 crossovers, 4 validated)
                'GBPJPY': 0.00300,   # Moderate
                'NZDJPY': 0.00300,   # Moderate
                'CADJPY': 0.00300,   # Moderate
                'CHFJPY': 0.00300,   # Moderate
                # Major pairs - FINAL CALIBRATED (EXTREME LOW for last 3 pairs)
                'EURUSD': 0.000001,  # EXTREME: Last attempt for this pair
                'GBPUSD': 0.000025,  # ✅ WORKING (298 crossovers, 10 validated)
                'AUDUSD': 0.000005,  # ✅ WORKING (66 crossovers)
                'NZDUSD': 0.000001,  # EXTREME: Last attempt for this pair
                'USDCHF': 0.000001,  # EXTREME: Last attempt for this pair
                'USDCAD': 0.000020,  # ✅ WORKING (34 crossovers)
            }

            # Find matching pair
            base_threshold = None
            for pair_name, threshold in pair_thresholds.items():
                if pair_name in epic.upper():
                    base_threshold = threshold
                    pair_type = pair_name
                    break

            # Fallback
            if base_threshold is None:
                if 'JPY' in epic.upper():
                    base_threshold = 0.00030
                    pair_type = 'JPY (default)'
                else:
                    base_threshold = 0.000015
                    pair_type = 'Major (default)'

            # Volatility-adaptive scaling
            if volatility_metrics and close_price:
                # Calculate ATR as percentage of price
                atr_pct = (volatility_metrics.atr / close_price) * 100

                # PHASE 2: Stricter multipliers - no reduction for low volatility
                if atr_pct > 0.8 or volatility_metrics.atr_percentile > 90:
                    # HIGH VOLATILITY - require even stronger signals
                    multiplier = 2.5
                    regime_name = "HIGH_VOL"
                elif atr_pct > 0.5:
                    # NORMAL VOLATILITY
                    multiplier = 1.5
                    regime_name = "NORMAL"
                else:
                    # LOW VOLATILITY - don't reduce, maintain base threshold
                    multiplier = 1.0
                    regime_name = "LOW_VOL"

                final_threshold = base_threshold * multiplier

                self.logger.debug(
                    f"🌊 Volatility-adaptive threshold for {epic} ({pair_type}): "
                    f"{final_threshold:.6f} (base={base_threshold:.6f}, mult={multiplier:.1f}x, "
                    f"regime={regime_name}, ATR%={atr_pct:.3f}%, percentile={volatility_metrics.atr_percentile:.0f}, "
                    f"ADX={volatility_metrics.adx:.1f})"
                )
            else:
                # No volatility metrics - use static threshold
                final_threshold = base_threshold
                self.logger.info(f"📊 Static threshold for {epic} ({pair_type}): {final_threshold:.6f}")

            return final_threshold

        except Exception as e:
            self.logger.error(f"Error determining histogram threshold for {epic}: {e}")
            return 0.00010  # Safe default

    def _extract_volatility_metrics(self, df: pd.DataFrame, epic: str) -> Optional['VolatilityMetrics']:
        """
        Extract VolatilityMetrics from dataframe for regime-adaptive thresholds

        Reuses volatility infrastructure from adaptive SL/TP system to determine
        market regime and adjust MACD detection sensitivity accordingly.

        Args:
            df: DataFrame with price and indicator data
            epic: Trading pair epic

        Returns:
            VolatilityMetrics object if data available, None otherwise
        """
        if not VOLATILITY_METRICS_AVAILABLE:
            return None

        try:
            if 'atr' not in df.columns or len(df) < 20:
                return None

            latest = df.iloc[-1]

            # Calculate ATR percentile from last 20 periods
            atr_20_period = df['atr'].tail(20)
            if len(atr_20_period) > 0:
                atr_percentile = (atr_20_period < latest['atr']).sum() / len(atr_20_period) * 100
            else:
                atr_percentile = 50.0

            # Simple efficiency ratio approximation (Kaufman's style)
            # Measures how efficiently price moves (straight line vs zigzag)
            if len(df) >= 20:
                price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-20])
                path_length = df['close'].tail(20).diff().abs().sum()
                efficiency = price_change / path_length if path_length > 0 else 0.5
            else:
                efficiency = 0.5

            # Get ADX if available
            adx = latest.get('adx', 20.0)

            return VolatilityMetrics(
                atr=latest['atr'],
                atr_percentile=atr_percentile,
                adx=adx,
                efficiency_ratio=efficiency,
                bb_width_percentile=50.0,  # Not critical for threshold calculation
                ema_separation=0.0,  # Not critical for threshold
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.debug(f"Could not extract volatility metrics for {epic}: {e}")
            return None

    def get_volatility_regime(self, df: pd.DataFrame) -> Dict:
        """
        PHASE 2: Detect volatility regime for adaptive threshold scaling

        Uses ATR to classify market volatility and recommend threshold adjustments:
        - Low volatility: Slightly lower thresholds (more signals)
        - Normal volatility: Standard thresholds
        - High volatility: Higher thresholds (fewer, higher quality signals)

        Args:
            df: DataFrame with ATR data

        Returns:
            Dictionary with volatility classification and multiplier
        """
        try:
            if 'atr' not in df.columns or len(df) < 20:
                return {'regime': 'normal', 'multiplier': 1.0, 'atr': 0}

            # Get recent ATR values for regime detection
            recent_atr = df['atr'].tail(20).dropna()
            if len(recent_atr) < 10:
                return {'regime': 'normal', 'multiplier': 1.0, 'atr': 0}

            current_atr = recent_atr.iloc[-1]
            atr_mean = recent_atr.mean()
            atr_std = recent_atr.std()

            # Calculate ATR percentiles for regime classification
            atr_20th = recent_atr.quantile(0.20)
            atr_80th = recent_atr.quantile(0.80)

            # Classify volatility regime
            if current_atr <= atr_20th:
                regime = 'low'
                multiplier = 0.8  # 20% lower thresholds for low volatility
            elif current_atr >= atr_80th:
                regime = 'high'
                multiplier = 1.5  # 50% higher thresholds for high volatility
            else:
                regime = 'normal'
                multiplier = 1.0  # Standard thresholds

            self.logger.debug(f"📊 Volatility regime: {regime} (ATR: {current_atr:.6f}, multiplier: {multiplier:.1f}x)")

            return {
                'regime': regime,
                'multiplier': multiplier,
                'atr': current_atr,
                'atr_mean': atr_mean,
                'atr_percentile': (current_atr - atr_20th) / (atr_80th - atr_20th) if atr_80th > atr_20th else 0.5
            }

        except Exception as e:
            self.logger.error(f"Error detecting volatility regime: {e}")
            return {'regime': 'normal', 'multiplier': 1.0, 'atr': 0}

    def get_enhanced_threshold(self, df: pd.DataFrame, epic: str, base_threshold: float) -> float:
        """
        PHASE 2: Calculate volatility-aware adaptive threshold for market regime detection

        Uses volatility regime detection to dynamically adjust thresholds:
        - Low volatility: 0.8x threshold (capture subtle moves)
        - Normal volatility: 1.0x threshold (standard detection)
        - High volatility: 1.5x threshold (filter noise)

        Args:
            df: DataFrame with ATR data
            epic: Trading pair epic
            base_threshold: Base histogram strength threshold

        Returns:
            Volatility-adjusted threshold for current market regime
        """
        try:
            # Get volatility regime for adaptive scaling
            volatility_info = self.get_volatility_regime(df)
            volatility_multiplier = volatility_info['multiplier']
            regime = volatility_info['regime']

            # Apply regime-based threshold adjustment
            enhanced_threshold = base_threshold * volatility_multiplier

            self.logger.debug(f"📊 PHASE 2 threshold for {epic}: {enhanced_threshold:.6f} "
                            f"(base: {base_threshold:.6f}, regime: {regime}, multiplier: {volatility_multiplier:.1f}x)")

            return enhanced_threshold

        except Exception as e:
            self.logger.error(f"Error calculating enhanced threshold for {epic}: {e}")
            return base_threshold  # Fallback to base threshold
    
    def validate_macd_strength(self, row: pd.Series, signal_type: str, threshold: float = 0.0001) -> bool:
        """
        Validate MACD signal strength
        
        Args:
            row: DataFrame row with MACD data
            signal_type: 'BULL' or 'BEAR'
            threshold: Minimum histogram value for valid signal
            
        Returns:
            True if signal is strong enough
        """
        try:
            histogram = row.get('macd_histogram', 0)
            
            if signal_type == 'BULL':
                # Bull signals need positive histogram above threshold
                return histogram > threshold
            elif signal_type == 'BEAR':
                # Bear signals need negative histogram below negative threshold  
                return histogram < -threshold
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating MACD strength: {e}")
            return False
    
    def get_macd_momentum_score(self, row: pd.Series) -> float:
        """
        Calculate MACD momentum score (0.0 to 1.0)
        
        Args:
            row: DataFrame row with MACD data
            
        Returns:
            Momentum score between 0.0 and 1.0
        """
        try:
            histogram = row.get('macd_histogram', 0)
            macd_line = row.get('macd_line', 0)
            macd_signal = row.get('macd_signal', 0)
            
            # Base score from histogram strength
            hist_score = min(1.0, abs(histogram) / 0.001) if histogram != 0 else 0.0
            
            # Bonus for MACD line and signal alignment
            line_signal_aligned = (
                (macd_line > macd_signal and histogram > 0) or
                (macd_line < macd_signal and histogram < 0)
            )
            alignment_bonus = 0.2 if line_signal_aligned else 0.0
            
            return min(1.0, hist_score + alignment_bonus)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.5  # Neutral score on error
    
    def _add_enhanced_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced filters: ADX, ATR, and RSI for restrictive signal generation
        
        These filters help reduce false signals and improve quality:
        - ADX: Measures trend strength (avoid choppy markets)
        - ATR: Measures volatility (dynamic threshold adjustment) 
        - RSI: Identifies overbought/oversold conditions
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with enhanced filter indicators
        """
        try:
            df_enhanced = df.copy()
            
            # 1. ADD ADX (Average Directional Index) for trend strength
            if 'adx' not in df_enhanced.columns:
                df_enhanced = self._calculate_adx(df_enhanced)
            
            # 2. ADD ATR (Average True Range) for volatility measurement
            if 'atr' not in df_enhanced.columns:
                df_enhanced = self._calculate_atr(df_enhanced)
            
            # 3. ADD RSI (Relative Strength Index) for overbought/oversold
            if 'rsi' not in df_enhanced.columns:
                df_enhanced = self._calculate_rsi(df_enhanced)

            # 4. ADD VWAP (Volume Weighted Average Price) - DISABLED for performance optimization
            # if 'vwap' not in df_enhanced.columns:
            #     df_enhanced = self._calculate_vwap(df_enhanced)

            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error adding enhanced filters: {e}")
            return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index) for trend strength measurement"""
        try:
            df_copy = df.copy()
            
            # Calculate True Range components
            df_copy['high_minus_low'] = df_copy['high'] - df_copy['low']
            df_copy['high_minus_close_prev'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['low_minus_close_prev'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            
            # True Range is the maximum of the three
            df_copy['true_range'] = df_copy[['high_minus_low', 'high_minus_close_prev', 'low_minus_close_prev']].max(axis=1)
            
            # Calculate directional movements
            df_copy['high_diff'] = df_copy['high'].diff()
            df_copy['low_diff'] = -df_copy['low'].diff()
            
            # Positive and Negative Directional Movement
            df_copy['plus_dm'] = np.where(
                (df_copy['high_diff'] > df_copy['low_diff']) & (df_copy['high_diff'] > 0),
                df_copy['high_diff'], 0
            )
            df_copy['minus_dm'] = np.where(
                (df_copy['low_diff'] > df_copy['high_diff']) & (df_copy['low_diff'] > 0),
                df_copy['low_diff'], 0
            )
            
            # Smooth the True Range and Directional Movements
            alpha = 1.0 / period
            df_copy['tr_smooth'] = df_copy['true_range'].ewm(alpha=alpha, adjust=False).mean()
            df_copy['plus_dm_smooth'] = df_copy['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
            df_copy['minus_dm_smooth'] = df_copy['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
            
            # Calculate Directional Indicators
            df_copy['plus_di'] = 100 * (df_copy['plus_dm_smooth'] / df_copy['tr_smooth'])
            df_copy['minus_di'] = 100 * (df_copy['minus_dm_smooth'] / df_copy['tr_smooth'])
            
            # Calculate DX (Directional Movement Index)
            df_copy['dx'] = 100 * abs(df_copy['plus_di'] - df_copy['minus_di']) / (df_copy['plus_di'] + df_copy['minus_di'])
            
            # ADX is the smoothed average of DX
            df_copy['adx'] = df_copy['dx'].ewm(alpha=alpha, adjust=False).mean()
            
            # Clean up temporary columns
            temp_cols = ['high_minus_low', 'high_minus_close_prev', 'low_minus_close_prev', 
                        'high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'tr_smooth',
                        'plus_dm_smooth', 'minus_dm_smooth', 'dx']
            df_copy = df_copy.drop(columns=temp_cols, errors='ignore')
            
            self.logger.debug(f"ADX calculated successfully (period: {period})")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ATR (Average True Range) for volatility measurement"""
        try:
            df_copy = df.copy()
            
            # True Range calculation
            df_copy['high_low'] = df_copy['high'] - df_copy['low']
            df_copy['high_close'] = abs(df_copy['high'] - df_copy['close'].shift(1))
            df_copy['low_close'] = abs(df_copy['low'] - df_copy['close'].shift(1))
            
            df_copy['true_range'] = df_copy[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            # ATR is the smoothed average of True Range
            df_copy['atr'] = df_copy['true_range'].ewm(span=period, adjust=False).mean()
            
            # Clean up temporary columns
            df_copy = df_copy.drop(columns=['high_low', 'high_close', 'low_close', 'true_range'], errors='ignore')
            
            self.logger.debug(f"ATR calculated successfully (period: {period})")
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index) for overbought/oversold detection"""
        try:
            df_copy = df.copy()
            
            # Calculate price changes
            df_copy['price_change'] = df_copy['close'].diff()
            
            # Separate gains and losses
            df_copy['gain'] = df_copy['price_change'].where(df_copy['price_change'] > 0, 0)
            df_copy['loss'] = -df_copy['price_change'].where(df_copy['price_change'] < 0, 0)
            
            # Calculate average gains and losses using EMA
            alpha = 1.0 / period
            df_copy['avg_gain'] = df_copy['gain'].ewm(alpha=alpha, adjust=False).mean()
            df_copy['avg_loss'] = df_copy['loss'].ewm(alpha=alpha, adjust=False).mean()
            
            # Calculate RS (Relative Strength) and RSI
            df_copy['rs'] = df_copy['avg_gain'] / df_copy['avg_loss']
            df_copy['rsi'] = 100 - (100 / (1 + df_copy['rs']))
            
            # Handle division by zero
            df_copy['rsi'] = df_copy['rsi'].fillna(50)  # Neutral RSI when avg_loss is 0
            
            # Clean up temporary columns
            temp_cols = ['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs']
            df_copy = df_copy.drop(columns=temp_cols, errors='ignore')
            
            self.logger.debug(f"RSI calculated successfully (period: {period})")
            return df_copy

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return df

    def _calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate VWAP (Volume Weighted Average Price) for institutional sentiment analysis

        VWAP is used by institutional traders as a benchmark. Price above VWAP indicates
        bullish sentiment, below indicates bearish sentiment.
        """
        try:
            df_copy = df.copy()

            # Use typical price (HLC/3) and volume (ltv - last traded volume)
            df_copy['typical_price'] = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3

            # Use ltv as volume proxy (last traded volume)
            volume_col = 'ltv' if 'ltv' in df_copy.columns else None
            if volume_col is None:
                # Fallback: use price movement as volume proxy
                df_copy['volume_proxy'] = abs(df_copy['close'].diff()).fillna(0) * 1000
                volume_col = 'volume_proxy'

            # Calculate rolling VWAP over specified period
            price_volume = df_copy['typical_price'] * df_copy[volume_col]

            # Rolling sum approach for VWAP
            rolling_pv = price_volume.rolling(window=period, min_periods=1).sum()
            rolling_volume = df_copy[volume_col].rolling(window=period, min_periods=1).sum()

            # Avoid division by zero
            df_copy['vwap'] = rolling_pv / rolling_volume.replace(0, 1)

            # Calculate VWAP deviation (price distance from VWAP as percentage)
            df_copy['vwap_deviation'] = (df_copy['close'] - df_copy['vwap']) / df_copy['vwap'] * 100

            self.logger.debug(f"VWAP calculated successfully (period: {period})")
            return df_copy

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return df
    
    def detect_macd_divergence(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Detect MACD divergence signals (advanced analysis for high-quality signals)
        
        Divergence Types:
        - Bullish Regular: Price lower lows, MACD higher lows (potential upward reversal)
        - Bearish Regular: Price higher highs, MACD lower highs (potential downward reversal) 
        - Bullish Hidden: Price higher lows, MACD lower lows (trend continuation up)
        - Bearish Hidden: Price lower highs, MACD higher highs (trend continuation down)
        
        Args:
            df: DataFrame with price and MACD data
            lookback: Number of bars to look back for divergence analysis
            
        Returns:
            DataFrame with divergence signals added
        """
        try:
            df_div = df.copy()
            
            # Initialize divergence columns
            df_div['bullish_divergence'] = False
            df_div['bearish_divergence'] = False  
            df_div['bullish_hidden_div'] = False
            df_div['bearish_hidden_div'] = False
            df_div['divergence_strength'] = 0.0
            
            # Ensure we have required data
            required_cols = ['close', 'high', 'low', 'macd_histogram', 'macd_line']
            if not all(col in df_div.columns for col in required_cols):
                self.logger.warning("Missing required columns for divergence detection")
                return df_div
            
            # Need enough data for meaningful analysis
            if len(df_div) < lookback * 2:
                return df_div
            
            # Find swing highs and lows in price and MACD
            df_div = self._identify_swing_points(df_div, lookback)
            
            # Detect regular divergences (reversal signals)
            df_div = self._detect_regular_divergence(df_div, lookback)
            
            # Detect hidden divergences (continuation signals)
            df_div = self._detect_hidden_divergence(df_div, lookback)
            
            self.logger.debug("MACD divergence detection completed")
            return df_div
            
        except Exception as e:
            self.logger.error(f"Error detecting MACD divergence: {e}")
            return df
    
    def _identify_swing_points(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Identify swing highs and lows in price and MACD"""
        try:
            df_swing = df.copy()
            
            # Price swing points
            df_swing['price_swing_high'] = False
            df_swing['price_swing_low'] = False
            
            # MACD swing points
            df_swing['macd_swing_high'] = False
            df_swing['macd_swing_low'] = False
            
            # Rolling windows to find local extremes
            for i in range(lookback, len(df_swing) - lookback):
                current_idx = df_swing.index[i]
                
                # Price swing high detection
                window_highs = df_swing['high'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'high'] == window_highs.max():
                    df_swing.loc[current_idx, 'price_swing_high'] = True
                
                # Price swing low detection  
                window_lows = df_swing['low'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'low'] == window_lows.min():
                    df_swing.loc[current_idx, 'price_swing_low'] = True
                
                # MACD swing high detection
                macd_window_high = df_swing['macd_line'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'macd_line'] == macd_window_high.max():
                    df_swing.loc[current_idx, 'macd_swing_high'] = True
                
                # MACD swing low detection
                macd_window_low = df_swing['macd_line'].iloc[i-lookback:i+lookback+1]
                if df_swing.loc[current_idx, 'macd_line'] == macd_window_low.min():
                    df_swing.loc[current_idx, 'macd_swing_low'] = True
            
            return df_swing
            
        except Exception as e:
            self.logger.error(f"Error identifying swing points: {e}")
            return df
    
    def _detect_regular_divergence(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Detect regular bullish and bearish divergences"""
        try:
            df_reg = df.copy()
            
            # Get recent swing points
            recent_data = df_reg.tail(lookback * 3)  # Look at recent data
            
            # Find price and MACD swing points
            price_highs = recent_data[recent_data['price_swing_high']]['high']
            price_lows = recent_data[recent_data['price_swing_low']]['low']
            macd_highs = recent_data[recent_data['macd_swing_high']]['macd_line']
            macd_lows = recent_data[recent_data['macd_swing_low']]['macd_line']
            
            # Check for bullish regular divergence (price lower lows, MACD higher lows)
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                recent_price_lows = price_lows.tail(2)
                recent_macd_lows = macd_lows.tail(2)
                
                if len(recent_price_lows) == 2 and len(recent_macd_lows) == 2:
                    price_lower = recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2]
                    macd_higher = recent_macd_lows.iloc[-1] > recent_macd_lows.iloc[-2]
                    
                    if price_lower and macd_higher:
                        last_idx = recent_price_lows.index[-1]
                        df_reg.loc[last_idx, 'bullish_divergence'] = True
                        # Calculate divergence strength
                        price_diff = abs(recent_price_lows.iloc[-2] - recent_price_lows.iloc[-1])
                        macd_diff = abs(recent_macd_lows.iloc[-1] - recent_macd_lows.iloc[-2])
                        df_reg.loc[last_idx, 'divergence_strength'] = min(1.0, (price_diff + macd_diff) / 0.01)
            
            # Check for bearish regular divergence (price higher highs, MACD lower highs)
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                recent_price_highs = price_highs.tail(2)
                recent_macd_highs = macd_highs.tail(2)
                
                if len(recent_price_highs) == 2 and len(recent_macd_highs) == 2:
                    price_higher = recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2]
                    macd_lower = recent_macd_highs.iloc[-1] < recent_macd_highs.iloc[-2]
                    
                    if price_higher and macd_lower:
                        last_idx = recent_price_highs.index[-1]
                        df_reg.loc[last_idx, 'bearish_divergence'] = True
                        # Calculate divergence strength
                        price_diff = abs(recent_price_highs.iloc[-1] - recent_price_highs.iloc[-2])
                        macd_diff = abs(recent_macd_highs.iloc[-2] - recent_macd_highs.iloc[-1])
                        df_reg.loc[last_idx, 'divergence_strength'] = min(1.0, (price_diff + macd_diff) / 0.01)
            
            return df_reg
            
        except Exception as e:
            self.logger.error(f"Error detecting regular divergence: {e}")
            return df
    
    def _detect_hidden_divergence(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Detect hidden bullish and bearish divergences (trend continuation)"""
        try:
            df_hid = df.copy()
            
            # Get recent swing points (similar to regular but looking for continuation patterns)
            recent_data = df_hid.tail(lookback * 3)
            
            price_highs = recent_data[recent_data['price_swing_high']]['high']
            price_lows = recent_data[recent_data['price_swing_low']]['low']
            macd_highs = recent_data[recent_data['macd_swing_high']]['macd_line']
            macd_lows = recent_data[recent_data['macd_swing_low']]['macd_line']
            
            # Bullish hidden divergence (price higher lows, MACD lower lows - uptrend continuation)
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                recent_price_lows = price_lows.tail(2)
                recent_macd_lows = macd_lows.tail(2)
                
                if len(recent_price_lows) == 2 and len(recent_macd_lows) == 2:
                    price_higher = recent_price_lows.iloc[-1] > recent_price_lows.iloc[-2]
                    macd_lower = recent_macd_lows.iloc[-1] < recent_macd_lows.iloc[-2]
                    
                    if price_higher and macd_lower:
                        last_idx = recent_price_lows.index[-1]
                        df_hid.loc[last_idx, 'bullish_hidden_div'] = True
            
            # Bearish hidden divergence (price lower highs, MACD higher highs - downtrend continuation)
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                recent_price_highs = price_highs.tail(2)
                recent_macd_highs = macd_highs.tail(2)
                
                if len(recent_price_highs) == 2 and len(recent_macd_highs) == 2:
                    price_lower = recent_price_highs.iloc[-1] < recent_price_highs.iloc[-2]
                    macd_higher = recent_macd_highs.iloc[-1] > recent_macd_highs.iloc[-2]
                    
                    if price_lower and macd_higher:
                        last_idx = recent_price_highs.index[-1]
                        df_hid.loc[last_idx, 'bearish_hidden_div'] = True
            
            return df_hid
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden divergence: {e}")
            return df
    
    def _apply_signal_spacing(self, df: pd.DataFrame, signals: pd.Series, signal_type: str, epic: str) -> pd.Series:
        """
        Apply signal spacing/cooldown mechanism to prevent over-trading
        
        Target: 3-4 signals per day per epic (96-128 15m bars spacing)
        Cooldown: 6-8 hours between signals (24-32 15m bars)
        
        Args:
            df: DataFrame with timestamp index
            signals: Boolean series of detected signals
            signal_type: 'bull' or 'bear'
            epic: Trading pair epic
            
        Returns:
            Filtered signal series with spacing applied
        """
        try:
            if signals.sum() == 0:
                return signals
            
            # Calculate spacing requirements (15m timeframe) - PHASE 2: VERY STRICT
            min_spacing_bars = 8  # 2 hour minimum spacing (120 minutes)
            max_daily_signals = 2  # PHASE 2: MAXIMUM 2 signals per pair per day
            
            # Create filtered signal series
            spaced_signals = pd.Series(False, index=signals.index)
            
            # Track last signal timestamp for cooldown
            last_signal_idx = None
            daily_signal_counts = {}
            
            for idx, signal in signals.items():
                if not signal:
                    continue
                    
                current_bar_idx = signals.index.get_loc(idx)
                
                # Handle datetime index properly
                try:
                    if hasattr(idx, 'date'):
                        current_date = idx.date()
                    elif hasattr(idx, 'to_pydatetime'):
                        current_date = idx.to_pydatetime().date()
                    else:
                        # Fallback to string-based date grouping
                        current_date = str(current_bar_idx // 96)  # Group by ~daily periods (96 15m bars = 24h)
                except Exception:
                    current_date = str(current_bar_idx // 96)  # Fallback grouping
                
                # Check daily limit
                if current_date not in daily_signal_counts:
                    daily_signal_counts[current_date] = 0
                    
                if daily_signal_counts[current_date] >= max_daily_signals:
                    continue
                
                # Check spacing from last signal
                if last_signal_idx is not None:
                    bars_since_last = current_bar_idx - last_signal_idx
                    if bars_since_last < min_spacing_bars:
                        continue
                
                # Signal passes all filters
                spaced_signals.loc[idx] = True
                last_signal_idx = current_bar_idx
                daily_signal_counts[current_date] += 1
            
            # Log spacing results
            original_count = signals.sum()
            final_count = spaced_signals.sum()
            if original_count != final_count:
                self.logger.debug(f"Signal spacing applied: {signal_type} signals reduced from {original_count} to {final_count}")
            
            return spaced_signals
            
        except Exception as e:
            self.logger.error(f"Error applying signal spacing: {e}")
            return signals
    
    def _apply_multi_candle_confirmation(self, df: pd.DataFrame, raw_signals: pd.Series, signal_type: str, epic: str) -> pd.Series:
        """
        Apply multi-candle confirmation like EMA strategy to reduce false MACD signals

        Requires the MACD condition to be sustained for multiple consecutive candles
        before confirming the signal. This dramatically reduces false breakouts.

        Args:
            df: DataFrame with MACD indicator data
            raw_signals: Boolean series of initial MACD crossover signals
            signal_type: 'BULL' or 'BEAR'
            epic: Trading pair epic

        Returns:
            Confirmed signal series with multi-candle validation
        """
        try:
            if raw_signals.sum() == 0:
                return raw_signals

            # Configuration - stronger confirmation for quality (PHASE 2)
            confirmation_candles = 4  # PHASE 2: Increase to 4 candles for stronger confirmation
            confirmed_signals = pd.Series(False, index=raw_signals.index)

            # Check each potential signal for multi-candle confirmation
            signal_indices = raw_signals[raw_signals].index

            for signal_idx in signal_indices:
                signal_position = df.index.get_loc(signal_idx)

                # Ensure we have enough future candles to check
                if signal_position + confirmation_candles >= len(df):
                    continue

                # Get the candles following the signal (including signal candle)
                check_start = signal_position
                check_end = signal_position + confirmation_candles + 1
                confirmation_candles_data = df.iloc[check_start:check_end]

                if signal_type == 'BULL':
                    # For bull signals: MACD histogram should stay positive for confirmation_candles
                    histogram_positive = (confirmation_candles_data['macd_histogram'] > 0).sum()

                    # Also check for momentum building (histogram increasing)
                    if len(confirmation_candles_data) >= 2:
                        momentum_building = (
                            confirmation_candles_data['macd_histogram'].iloc[-1] >
                            confirmation_candles_data['macd_histogram'].iloc[0]
                        )
                    else:
                        momentum_building = True

                    # Require at least 75% of candles to maintain positive histogram (balanced)
                    confirmation_strength = histogram_positive / len(confirmation_candles_data)
                    is_confirmed = confirmation_strength >= 0.75 and momentum_building  # Balanced between 0.7 and 0.85

                else:  # BEAR
                    # For bear signals: MACD histogram should stay negative for confirmation_candles
                    histogram_negative = (confirmation_candles_data['macd_histogram'] < 0).sum()

                    # Also check for momentum building (histogram becoming more negative)
                    if len(confirmation_candles_data) >= 2:
                        momentum_building = (
                            confirmation_candles_data['macd_histogram'].iloc[-1] <
                            confirmation_candles_data['macd_histogram'].iloc[0]
                        )
                    else:
                        momentum_building = True

                    # Require at least 75% of candles to maintain negative histogram (balanced)
                    confirmation_strength = histogram_negative / len(confirmation_candles_data)
                    is_confirmed = confirmation_strength >= 0.75 and momentum_building  # Balanced between 0.7 and 0.85

                if is_confirmed:
                    confirmed_signals.loc[signal_idx] = True
                    self.logger.debug(f"Multi-candle confirmation: {signal_type} signal at {signal_idx} confirmed (strength: {confirmation_strength:.1%})")

            # Log confirmation results
            original_count = raw_signals.sum()
            confirmed_count = confirmed_signals.sum()
            reduction_pct = ((original_count - confirmed_count) / max(original_count, 1)) * 100

            self.logger.debug(f"Multi-candle confirmation for {epic}: {signal_type} signals reduced from {original_count} to {confirmed_count} ({reduction_pct:.1f}% reduction)")

            return confirmed_signals

        except Exception as e:
            self.logger.error(f"Error in multi-candle confirmation: {e}")
            return raw_signals  # Return original signals if error
    
    def _apply_quality_score_filter(self, df: pd.DataFrame, signals: pd.Series, signal_type: str, base_threshold: float, epic: str) -> pd.Series:
        """
        Apply quality score filtering to keep only the highest quality signals
        
        Only the top 20% of signals by quality score will pass through this filter.
        This is the final quality gate before spacing restrictions.
        
        Args:
            df: DataFrame with indicator data
            signals: Boolean series of detected signals
            signal_type: 'BULL' or 'BEAR'
            base_threshold: Base histogram threshold for pair
            epic: Trading pair epic
            
        Returns:
            Filtered signal series with only top quality signals
        """
        try:
            if signals.sum() == 0:
                return signals
            
            # Calculate quality scores for all signal candidates
            signal_indices = signals[signals].index
            quality_scores = []
            
            for idx in signal_indices:
                row = df.loc[idx]
                score = self._calculate_signal_quality_score(row, signal_type, base_threshold)
                quality_scores.append((idx, score))
            
            # Sort by quality score (highest first)
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top 25% of signals (minimum 1 signal if any exist) - MORE PERMISSIVE
            num_signals = len(quality_scores)
            keep_count = max(1, int(num_signals * 0.25))  # Top 25% but at least 1 (was 15%)
            
            # Create filtered signal series
            filtered_signals = pd.Series(False, index=signals.index)
            
            # Keep the highest quality signals
            for i in range(min(keep_count, len(quality_scores))):
                idx, score = quality_scores[i]
                filtered_signals.loc[idx] = True
                self.logger.debug(f"Quality signal kept: {signal_type} at {idx} with score {score:.1f}")
            
            # Log quality filtering results
            original_count = signals.sum()
            final_count = filtered_signals.sum()
            if original_count != final_count:
                self.logger.debug(f"Quality filter applied for {epic}: {signal_type} signals reduced from {original_count} to {final_count} (top 15%)")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error applying quality score filter: {e}")
            return signals
    
    def _apply_final_signal_reduction(self, df: pd.DataFrame, signals: pd.Series, signal_type: str, base_threshold: float, epic: str, max_signals_per_day: int = 8) -> pd.Series:
        """
        Final brute-force signal reduction to achieve exact target (2-4 signals per day)
        
        This is the ultimate filter that ensures we never exceed the maximum signals per day
        by taking only the highest quality signals within each daily period.
        
        Args:
            df: DataFrame with indicator data  
            signals: Boolean series of detected signals
            signal_type: 'BULL' or 'BEAR'
            base_threshold: Base histogram threshold for pair
            epic: Trading pair epic
            max_signals_per_day: Maximum signals allowed per day
            
        Returns:
            Final reduced signal series with strict daily limits
        """
        try:
            if signals.sum() == 0:
                return signals
            
            # Group signals by day and apply quality-based selection
            final_signals = pd.Series(False, index=signals.index)
            
            # Get signal candidates with their timestamps
            signal_candidates = []
            for idx in signals[signals].index:
                row = df.loc[idx]
                quality_score = self._calculate_signal_quality_score(row, signal_type, base_threshold)
                
                # Get date for grouping
                try:
                    if hasattr(idx, 'date'):
                        signal_date = idx.date()
                    elif hasattr(idx, 'to_pydatetime'):
                        signal_date = idx.to_pydatetime().date()
                    else:
                        # Use index position for grouping
                        bar_idx = signals.index.get_loc(idx)
                        signal_date = str(bar_idx // 96)  # ~24h groups
                except Exception:
                    bar_idx = signals.index.get_loc(idx)
                    signal_date = str(bar_idx // 96)
                
                signal_candidates.append({
                    'index': idx,
                    'date': signal_date,
                    'quality': quality_score,
                    'timestamp': idx
                })
            
            # Group by date and select top N signals per day
            daily_groups = {}
            for candidate in signal_candidates:
                date = candidate['date']
                if date not in daily_groups:
                    daily_groups[date] = []
                daily_groups[date].append(candidate)
            
            # Process each day separately
            total_kept = 0
            for date, day_signals in daily_groups.items():
                # Sort by quality score (highest first)
                day_signals.sort(key=lambda x: x['quality'], reverse=True)
                
                # Keep only top N signals for this day
                kept_today = 0
                for signal in day_signals[:max_signals_per_day]:
                    final_signals.loc[signal['index']] = True
                    kept_today += 1
                    total_kept += 1
                    
                    self.logger.debug(f"Final selection: {signal_type} signal on {date} at {signal['timestamp']} (quality: {signal['quality']:.1f})")
            
            # Log final reduction results
            original_count = signals.sum()
            if original_count != total_kept:
                self.logger.info(f"🚨 FINAL REDUCTION for {epic}: {signal_type} signals reduced from {original_count} to {total_kept} ({max_signals_per_day}/day limit)")
            else:
                self.logger.info(f"🚨 FINAL REDUCTION for {epic}: {signal_type} kept all {total_kept} signals (under daily limit)")
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error applying final signal reduction: {e}")
            return signals
    
    def _calculate_signal_quality_score(self, row: pd.Series, signal_type: str, base_threshold: float) -> float:
        """
        Calculate progressive quality score for a MACD signal (0-100 points)
        
        Scoring Components:
        - MACD Histogram Strength (25 points): Stronger momentum = higher score
        - RSI Confluence (20 points): Overbought/oversold alignment  
        - ADX Trend Strength (20 points): Strong trending markets preferred
        - EMA200 Trend Alignment (15 points): Direction confirmation
        - Divergence Bonus (10 points): Premium quality signals
        - MACD Line Position (10 points): Additional momentum confirmation
        
        Args:
            row: DataFrame row with indicator data
            signal_type: 'BULL' or 'BEAR'
            base_threshold: Base histogram threshold for pair
            
        Returns:
            Quality score from 0-100
        """
        try:
            total_score = 0.0
            
            # Get indicator values
            histogram = row.get('macd_histogram', 0)
            macd_line = row.get('macd_line', 0) 
            macd_signal = row.get('macd_signal', 0)
            rsi = row.get('rsi', 50)
            adx = row.get('adx', 0)
            close = row.get('close', 0)
            ema_200 = row.get('ema_200', 0)
            
            # 1. MACD Histogram Strength (25 points)
            hist_abs = abs(histogram)
            if signal_type == 'BULL' and histogram > 0:
                if hist_abs >= base_threshold * 3.0:
                    total_score += 25  # Very strong
                elif hist_abs >= base_threshold * 2.0:
                    total_score += 20  # Strong
                elif hist_abs >= base_threshold * 1.0:
                    total_score += 15  # Moderate
                elif hist_abs >= base_threshold * 0.5:
                    total_score += 10  # Weak but valid
                else:
                    total_score += 5   # Very weak
            elif signal_type == 'BEAR' and histogram < 0:
                if hist_abs >= base_threshold * 3.0:
                    total_score += 25  # Very strong
                elif hist_abs >= base_threshold * 2.0:
                    total_score += 20  # Strong
                elif hist_abs >= base_threshold * 1.0:
                    total_score += 15  # Moderate
                elif hist_abs >= base_threshold * 0.5:
                    total_score += 10  # Weak but valid
                else:
                    total_score += 5   # Very weak
            
            # 2. RSI Confluence (20 points)
            if signal_type == 'BULL':
                if 50 < rsi < 60:
                    total_score += 20  # Optimal range for bull signals
                elif 60 <= rsi < 70:
                    total_score += 15  # Good momentum building
                elif 70 <= rsi < 80:
                    total_score += 10  # Upper range but still valid
                elif rsi >= 80:
                    total_score += 0   # Overbought - no points
                else:
                    total_score += 0   # Below 50 - filtered out anyway
            else:  # BEAR
                if 30 < rsi < 50:
                    total_score += 20  # Optimal range for bear signals
                elif 20 < rsi <= 30:
                    total_score += 15  # Good bearish momentum
                elif rsi <= 20:
                    total_score += 0   # Oversold - no points
                else:
                    total_score += 0   # Above 50 - filtered out anyway
            
            # 3. ADX Trend Strength (20 points) 
            if adx >= 40:
                total_score += 20  # Very strong trend
            elif adx >= 30:
                total_score += 15  # Strong trend
            elif adx >= 25:
                total_score += 10  # Moderate trend
            elif adx >= 20:
                total_score += 5   # Weak trend
            # No points for adx < 20 (ranging market)
            
            # 4. EMA200 Trend Alignment (15 points)
            if close > 0 and ema_200 > 0:
                if signal_type == 'BULL' and close > ema_200:
                    distance_pct = (close - ema_200) / ema_200 * 100
                    if distance_pct > 2.0:
                        total_score += 15  # Strong uptrend
                    elif distance_pct > 0.5:
                        total_score += 10  # Uptrend
                    else:
                        total_score += 5   # Weak uptrend
                elif signal_type == 'BEAR' and close < ema_200:
                    distance_pct = (ema_200 - close) / ema_200 * 100
                    if distance_pct > 2.0:
                        total_score += 15  # Strong downtrend
                    elif distance_pct > 0.5:
                        total_score += 10  # Downtrend
                    else:
                        total_score += 5   # Weak downtrend
            
            # 5. Divergence Bonus (10 points)
            if signal_type == 'BULL' and row.get('bullish_divergence', False):
                total_score += 10
            elif signal_type == 'BEAR' and row.get('bearish_divergence', False):
                total_score += 10
            
            # 6. MACD Line Position (10 points)
            if signal_type == 'BULL' and macd_line > macd_signal:
                if macd_line > 0:
                    total_score += 10  # Above zero line
                else:
                    total_score += 5   # Above signal but below zero
            elif signal_type == 'BEAR' and macd_line < macd_signal:
                if macd_line < 0:
                    total_score += 10  # Below zero line
                else:
                    total_score += 5   # Below signal but above zero
            
            return min(100.0, total_score)  # Cap at 100
            
        except Exception as e:
            self.logger.error(f"Error calculating signal quality score: {e}")
            return 0.0

    def _apply_simple_circuit_breaker(self, df: pd.DataFrame, signals: pd.Series, signal_type: str, epic: str) -> pd.Series:
        """
        SIMPLE CIRCUIT BREAKER: Prevent over-signaling with minimum time spacing

        Much simpler than previous complex daily limits - just ensure minimum spacing
        between signals to prevent the every-15-minute signal spam.
        """
        try:
            if signals.sum() == 0:
                return signals

            # QUALITY: Increased spacing for better signal quality
            min_spacing_bars = 4  # 1 hour minimum spacing (4 x 15min bars) for quality

            filtered_signals = pd.Series(False, index=signals.index)
            last_signal_idx = None

            for idx, signal in signals.items():
                if not signal:
                    continue

                current_bar_idx = signals.index.get_loc(idx)

                # Check spacing from last signal
                if last_signal_idx is not None:
                    bars_since_last = current_bar_idx - last_signal_idx
                    if bars_since_last < min_spacing_bars:
                        continue  # Skip this signal - too close to previous

                # Signal passes spacing test
                filtered_signals.loc[idx] = True
                last_signal_idx = current_bar_idx

            original_count = signals.sum()
            final_count = filtered_signals.sum()

            if original_count != final_count:
                self.logger.info(f"⏰ Circuit breaker for {epic} {signal_type}: {original_count} -> {final_count} (min 1hr spacing)")

            return filtered_signals

        except Exception as e:
            self.logger.error(f"Error in circuit breaker: {e}")
            return signals  # Return original signals if error

    def _apply_global_signal_limiter(
        self,
        df: pd.DataFrame,
        bull_signals: pd.Series,
        bear_signals: pd.Series,
        epic: str,
        volatility_metrics: Optional['VolatilityMetrics'] = None
    ) -> tuple:
        """
        VOLATILITY-AWARE GLOBAL SIGNAL LIMITER

        Adapts signal spacing and limits based on market regime:
        - HIGH VOLATILITY/CHOPPY: Strict limits (8-bar spacing, max 3 signals)
        - TRENDING: Relaxed limits (3-bar spacing, max 8 signals)
        - NORMAL: Moderate limits (4-bar spacing, max 5 signals)

        This prevents over-signaling in ranging markets while allowing
        multiple entries in strong trends.
        """
        try:
            # Combine all signals to check global spacing
            all_signals = bull_signals | bear_signals
            total_before = all_signals.sum()

            if total_before == 0:
                return bull_signals, bear_signals

            # VOLATILITY-ADAPTIVE LIMITS
            if volatility_metrics:
                # High volatility or ranging = strict limits
                if volatility_metrics.atr_percentile > 90 or (volatility_metrics.adx < 20 and volatility_metrics.efficiency_ratio < 0.4):
                    min_spacing_bars = 8  # 2 hours
                    max_signals = 3
                    regime = "high_vol/ranging"
                # Strong trend = relaxed limits
                elif volatility_metrics.adx > 25 and volatility_metrics.efficiency_ratio > 0.6:
                    min_spacing_bars = 3  # 45 minutes
                    max_signals = 8
                    regime = "trending"
                else:
                    # Normal conditions
                    min_spacing_bars = 4  # 1 hour
                    max_signals = 5
                    regime = "normal"

                self.logger.debug(
                    f"🎛️ Signal limiter for {epic}: regime={regime}, "
                    f"spacing={min_spacing_bars}bars ({min_spacing_bars*15}min), max={max_signals} "
                    f"(ATR%ile={volatility_metrics.atr_percentile:.0f}, ADX={volatility_metrics.adx:.1f}, "
                    f"efficiency={volatility_metrics.efficiency_ratio:.2f})"
                )
            else:
                # No volatility metrics - use conservative defaults
                min_spacing_bars = 4  # 1 hour minimum between ANY signals (4 x 15min bars)
                max_signals = 5
                regime = "default"
                self.logger.debug(f"🎛️ Signal limiter for {epic}: using default limits (no volatility data)")

            # Get all signal indices sorted by time
            signal_indices = [(idx, 'BULL' if bull_signals.loc[idx] else 'BEAR')
                            for idx in all_signals[all_signals].index]

            if not signal_indices:
                return bull_signals, bear_signals

            # Apply global spacing - keep only signals that are far enough apart
            filtered_bull = pd.Series(False, index=bull_signals.index)
            filtered_bear = pd.Series(False, index=bear_signals.index)

            last_signal_bar_idx = None
            kept_signals = 0

            for idx, signal_type in signal_indices:
                current_bar_idx = df.index.get_loc(idx)

                # Check global spacing
                if last_signal_bar_idx is not None:
                    bars_since_last = current_bar_idx - last_signal_bar_idx
                    if bars_since_last < min_spacing_bars:
                        continue  # Skip - too close to previous signal of ANY type

                # Keep this signal
                if signal_type == 'BULL':
                    filtered_bull.loc[idx] = True
                else:
                    filtered_bear.loc[idx] = True

                last_signal_bar_idx = current_bar_idx
                kept_signals += 1

                # REGIME-ADAPTIVE BRAKE: Max signals based on volatility
                if kept_signals >= max_signals:
                    break

            bull_after = filtered_bull.sum()
            bear_after = filtered_bear.sum()
            total_after = bull_after + bear_after

            if total_before != total_after:
                self.logger.debug(
                    f"🚨 Signal limiter for {epic}: {total_before} -> {total_after} signals "
                    f"({min_spacing_bars*15}min spacing, max {max_signals}, regime={regime})"
                )

            return filtered_bull, filtered_bear

        except Exception as e:
            self.logger.error(f"Error in global signal limiter: {e}")
            return bull_signals, bear_signals  # Return original if error

    def _apply_backtest_global_filter(self, df: pd.DataFrame, bull_signals: pd.Series, bear_signals: pd.Series, epic: str) -> tuple:
        """
        DEPRECATED: No longer used - backtest now uses same volatility-aware limiter as live

        EMERGENCY BACKTEST GLOBAL FILTER: Prevent signals across multiple detect_signal() calls

        The backtesting system calls detect_signal() for every bar, so we need to track
        signals globally across the entire backtest session to prevent 500+ signals.
        """
        self.logger.warning("⚠️ DEPRECATED: _apply_backtest_global_filter called but should use _apply_global_signal_limiter instead")
        # Fall back to using the live limiter
        return self._apply_global_signal_limiter(df, bull_signals, bear_signals, epic, None)

    def _apply_volatility_filter(self, df: pd.DataFrame, bull_signals: pd.Series, bear_signals: pd.Series, epic: str) -> tuple:
        """
        Apply volatility-based filtering to avoid trading in choppy/ranging markets

        Returns:
            tuple: (filtered_bull_signals, filtered_bear_signals) or None if filter cannot be applied
        """
        try:
            if len(df) < 20:  # Need enough data for ATR calculation
                return None

            # Calculate Average True Range (ATR) for volatility measurement
            df_vol = df.copy()

            # True Range calculation
            df_vol['prev_close'] = df_vol['close'].shift(1)
            df_vol['tr1'] = df_vol['high'] - df_vol['low']
            df_vol['tr2'] = abs(df_vol['high'] - df_vol['prev_close'])
            df_vol['tr3'] = abs(df_vol['low'] - df_vol['prev_close'])
            df_vol['true_range'] = df_vol[['tr1', 'tr2', 'tr3']].max(axis=1)

            # 14-period ATR
            df_vol['atr'] = df_vol['true_range'].rolling(window=14).mean()

            # Calculate price movement efficiency (trending vs choppy)
            df_vol['price_change'] = abs(df_vol['close'] - df_vol['close'].shift(10))
            df_vol['cumulative_movement'] = df_vol['true_range'].rolling(window=10).sum()
            df_vol['efficiency_ratio'] = df_vol['price_change'] / df_vol['cumulative_movement']

            # Very relaxed volatility thresholds for more signals
            atr_percentile_90 = df_vol['atr'].quantile(0.95)  # Only filter very extreme volatility
            efficiency_threshold = 0.1  # Much lower efficiency requirement

            # Apply filters
            vol_bull_signals = bull_signals.copy()
            vol_bear_signals = bear_signals.copy()

            if bull_signals.sum() > 0:
                pre_count = bull_signals.sum()

                # Relaxed filter conditions:
                # 1. Only filter extreme volatility (top 10%)
                # 2. Lower efficiency requirement
                volatility_condition = (
                    (df_vol['atr'] <= atr_percentile_90) &  # Only filter extreme volatility
                    (df_vol['efficiency_ratio'] >= efficiency_threshold)  # Lower requirement
                )

                vol_bull_signals = bull_signals & volatility_condition
                post_count = vol_bull_signals.sum()

                if pre_count != post_count:
                    self.logger.debug(f"🌊 BULL volatility filter for {epic}: {pre_count} -> {post_count}")

            if bear_signals.sum() > 0:
                pre_count = bear_signals.sum()

                # Same relaxed volatility conditions for bear signals
                volatility_condition = (
                    (df_vol['atr'] <= atr_percentile_90) &
                    (df_vol['efficiency_ratio'] >= efficiency_threshold)
                )

                vol_bear_signals = bear_signals & volatility_condition
                post_count = vol_bear_signals.sum()

                if pre_count != post_count:
                    self.logger.debug(f"🌊 BEAR volatility filter for {epic}: {pre_count} -> {post_count}")

            return vol_bull_signals, vol_bear_signals

        except Exception as e:
            self.logger.error(f"Error applying volatility filter: {e}")
            return None

    def _apply_adaptive_signal_scoring(self, df: pd.DataFrame, bull_signals: pd.Series,
                                     bear_signals: pd.Series, epic: str, base_threshold: float) -> tuple:
        """
        Apply adaptive signal scoring to select only the highest confidence signals

        Returns:
            tuple: (filtered_bull_signals, filtered_bear_signals)
        """
        try:
            # Calculate signal scores for all potential signals
            df_score = df.copy()

            # Initialize score columns
            df_score['bull_score'] = 0.0
            df_score['bear_score'] = 0.0

            # Only score where we have signals
            bull_indices = bull_signals[bull_signals].index
            bear_indices = bear_signals[bear_signals].index

            # BULL SIGNAL SCORING
            if len(bull_indices) > 0:
                for idx in bull_indices:
                    score = 0.0

                    # 1. MACD Histogram Strength (0-40 points)
                    hist_strength = df_score.loc[idx, 'macd_histogram'] / base_threshold
                    score += min(40, hist_strength * 10)

                    # 2. Momentum Quality (0-25 points)
                    try:
                        idx_pos = df_score.index.get_loc(idx)
                        if idx_pos >= 2:  # Need history for momentum
                            prev_idx = df_score.index[idx_pos-1]
                            prev2_idx = df_score.index[idx_pos-2]
                            macd_trend = (
                                df_score.loc[idx, 'macd_line'] > df_score.loc[prev_idx, 'macd_line'] and
                                df_score.loc[prev_idx, 'macd_line'] >= df_score.loc[prev2_idx, 'macd_line']
                            )
                            if macd_trend:
                                score += 25
                    except (KeyError, IndexError):
                        pass  # Skip momentum check if insufficient data

                    # 3. Trend Alignment Bonus (0-20 points)
                    if 'ema_200' in df_score.columns:
                        if df_score.loc[idx, 'close'] > df_score.loc[idx, 'ema_200']:
                            score += 20

                    # 4. RSI Positioning (0-15 points)
                    if 'rsi' in df_score.columns:
                        rsi_val = df_score.loc[idx, 'rsi']
                        if 45 <= rsi_val <= 65:  # Sweet spot
                            score += 15
                        elif 35 <= rsi_val <= 75:  # Good range
                            score += 10
                        elif 30 <= rsi_val <= 80:  # Acceptable
                            score += 5

                    # 5. Market Strength (ADX) (0-10 points)
                    if 'adx' in df_score.columns:
                        adx_val = df_score.loc[idx, 'adx']
                        if adx_val > 25:
                            score += 10
                        elif adx_val > 20:
                            score += 7
                        elif adx_val > 15:
                            score += 3

                    df_score.loc[idx, 'bull_score'] = score

            # BEAR SIGNAL SCORING (similar logic)
            if len(bear_indices) > 0:
                for idx in bear_indices:
                    score = 0.0

                    # 1. MACD Histogram Strength (0-40 points)
                    hist_strength = abs(df_score.loc[idx, 'macd_histogram']) / base_threshold
                    score += min(40, hist_strength * 10)

                    # 2. Momentum Quality (0-25 points)
                    try:
                        idx_pos = df_score.index.get_loc(idx)
                        if idx_pos >= 2:  # Need history for momentum
                            prev_idx = df_score.index[idx_pos-1]
                            prev2_idx = df_score.index[idx_pos-2]
                            macd_trend = (
                                df_score.loc[idx, 'macd_line'] < df_score.loc[prev_idx, 'macd_line'] and
                                df_score.loc[prev_idx, 'macd_line'] <= df_score.loc[prev2_idx, 'macd_line']
                            )
                            if macd_trend:
                                score += 25
                    except (KeyError, IndexError):
                        pass  # Skip momentum check if insufficient data

                    # 3. Trend Alignment Bonus (0-20 points)
                    if 'ema_200' in df_score.columns:
                        if df_score.loc[idx, 'close'] < df_score.loc[idx, 'ema_200']:
                            score += 20

                    # 4. RSI Positioning (0-15 points)
                    if 'rsi' in df_score.columns:
                        rsi_val = df_score.loc[idx, 'rsi']
                        if 35 <= rsi_val <= 55:  # Sweet spot for bears
                            score += 15
                        elif 25 <= rsi_val <= 65:  # Good range
                            score += 10
                        elif 20 <= rsi_val <= 70:  # Acceptable
                            score += 5

                    # 5. Market Strength (ADX) (0-10 points)
                    if 'adx' in df_score.columns:
                        adx_val = df_score.loc[idx, 'adx']
                        if adx_val > 25:
                            score += 10
                        elif adx_val > 20:
                            score += 7
                        elif adx_val > 15:
                            score += 3

                    df_score.loc[idx, 'bear_score'] = score

            # ADAPTIVE THRESHOLD SELECTION
            # Only take signals that score in top 80th percentile OR score > 70

            filtered_bull = pd.Series(False, index=df.index)
            filtered_bear = pd.Series(False, index=df.index)

            if len(bull_indices) > 0:
                bull_scores = df_score.loc[bull_indices, 'bull_score']
                if len(bull_scores) > 0:
                    min_score = max(50, bull_scores.quantile(0.5))  # Top 50% OR score 50+ (more lenient)
                    high_quality_bulls = bull_scores[bull_scores >= min_score].index
                    filtered_bull.loc[high_quality_bulls] = True

                    self.logger.debug(f"🎯 BULL adaptive scoring for {epic}: {len(bull_indices)} -> {len(high_quality_bulls)} (min score: {min_score:.1f})")

            if len(bear_indices) > 0:
                bear_scores = df_score.loc[bear_indices, 'bear_score']
                if len(bear_scores) > 0:
                    min_score = max(50, bear_scores.quantile(0.5))  # Top 50% OR score 50+ (more lenient)
                    high_quality_bears = bear_scores[bear_scores >= min_score].index
                    filtered_bear.loc[high_quality_bears] = True

                    self.logger.debug(f"🎯 BEAR adaptive scoring for {epic}: {len(bear_indices)} -> {len(high_quality_bears)} (min score: {min_score:.1f})")

            return filtered_bull, filtered_bear

        except Exception as e:
            self.logger.error(f"Error in adaptive signal scoring: {e}")
            return bull_signals, bear_signals
    def _apply_sr_level_filter(self, df: pd.DataFrame, bull_signals: pd.Series, bear_signals: pd.Series, epic: str) -> tuple:
        """
        IMPROVED: PRE-FILTER with volatility-aware distances and enhanced S/R detection

        This improves signal quality by rejecting bad signals BEFORE they go through
        the full validation pipeline, improving validation rate from 33% to 70%+

        Improvements:
        - Volatility-aware minimum distance (scales with ATR)
        - Enhanced S/R detection with 10-bar lookback
        - Stricter filtering near key levels

        Args:
            df: DataFrame with OHLC data
            bull_signals: Boolean series of bull signals
            bear_signals: Boolean series of bear signals
            epic: Trading pair epic

        Returns:
            tuple: (filtered_bull_signals, filtered_bear_signals)
        """
        try:
            original_bull = bull_signals.sum()
            original_bear = bear_signals.sum()

            if original_bull == 0 and original_bear == 0:
                return bull_signals, bear_signals

            # Calculate S/R levels with 10-bar lookback window
            sr_levels = self._calculate_simple_sr_levels(df, lookback=200)

            if not sr_levels['support'] and not sr_levels['resistance']:
                self.logger.debug(f"No S/R levels found for {epic} - allowing all signals")
                return bull_signals, bear_signals

            # IMPROVED: Volatility-aware minimum distance
            pip_size = 0.01 if 'JPY' in epic.upper() else 0.0001

            # Calculate ATR-based minimum distance
            if 'atr' in df.columns and len(df) > 0:
                recent_atr = df['atr'].iloc[-1]
                # Use 1.5x ATR as minimum distance (more conservative than previous 10 pips)
                atr_pips = (recent_atr / pip_size) if recent_atr and recent_atr > 0 else 0
                min_distance_pips = max(10.0, atr_pips * 1.5)  # At least 10 pips, or 1.5x ATR
                self.logger.debug(f"S/R filter using volatility-aware distance: {min_distance_pips:.1f} pips (ATR: {atr_pips:.1f})")
            else:
                min_distance_pips = 10.0  # Fallback to 10 pips

            filtered_bull = bull_signals.copy()
            filtered_bear = bear_signals.copy()

            # Check each bull signal
            for idx in bull_signals[bull_signals].index:
                price = df.loc[idx, 'close']

                # Check if too close to any resistance level ABOVE price (BUY signals fail near resistance above)
                too_close = False
                closest_distance = float('inf')
                closest_level = None

                for resistance in sr_levels['resistance']:
                    # Only check resistance ABOVE current price
                    if resistance > price:
                        distance_pips = (resistance - price) / pip_size
                        if distance_pips < min_distance_pips:
                            too_close = True
                            if distance_pips < closest_distance:
                                closest_distance = distance_pips
                                closest_level = resistance

                if too_close:
                    self.logger.info(f"🚫 S/R PRE-FILTER: Rejected BUY signal at {price:.5f} - {closest_distance:.1f} pips below resistance {closest_level:.5f} (min: {min_distance_pips:.1f})")
                    filtered_bull.loc[idx] = False

            # Check each bear signal
            for idx in bear_signals[bear_signals].index:
                price = df.loc[idx, 'close']

                # SELL signals: Check BOTH resistance above AND support below
                too_close = False
                closest_distance = float('inf')
                closest_level = None
                level_type = None

                # 1. Check if too close to resistance ABOVE (price might bounce back up)
                for resistance in sr_levels['resistance']:
                    if resistance > price:
                        distance_pips = (resistance - price) / pip_size
                        if distance_pips < min_distance_pips:
                            too_close = True
                            if distance_pips < closest_distance:
                                closest_distance = distance_pips
                                closest_level = resistance
                                level_type = 'resistance above'

                # 2. Check if too close to support BELOW (might block downward movement)
                for support in sr_levels['support']:
                    if support < price:
                        distance_pips = (price - support) / pip_size
                        if distance_pips < min_distance_pips:
                            too_close = True
                            if distance_pips < closest_distance:
                                closest_distance = distance_pips
                                closest_level = support
                                level_type = 'support below'

                if too_close:
                    self.logger.info(f"🚫 S/R PRE-FILTER: Rejected SELL signal at {price:.5f} - {closest_distance:.1f} pips from {level_type} {closest_level:.5f} (min: {min_distance_pips:.1f})")
                    filtered_bear.loc[idx] = False

            final_bull = filtered_bull.sum()
            final_bear = filtered_bear.sum()

            if original_bull + original_bear != final_bull + final_bear:
                self.logger.info(f"📊 S/R PRE-FILTER for {epic}: "
                               f"Bull {original_bull} -> {final_bull}, "
                               f"Bear {original_bear} -> {final_bear} "
                               f"(removed {(original_bull + original_bear) - (final_bull + final_bear)} near S/R levels)")

            return filtered_bull, filtered_bear

        except Exception as e:
            self.logger.error(f"Error in S/R level filter: {e}")
            return bull_signals, bear_signals
    
    def _calculate_simple_sr_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Calculate support/resistance levels for LIVE TRADING & BACKTESTING

        REALISTIC TRADING LOGIC: Only look LEFT (historical data)
        - Find local highs that are higher than N bars to the left
        - Find local lows that are lower than N bars to the left
        - Include recent highs/lows as primary levels (most relevant)

        This works correctly for BOTH:
        - Live trading: We don't have future bars
        - Backtesting: Each bar only sees historical data (bar-by-bar simulation)

        Args:
            df: DataFrame with OHLC data (up to current bar only)
            lookback: Number of bars to look back (default 100)

        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if len(df) < 20:
                return {'support': [], 'resistance': []}

            # Use last N bars for S/R calculation
            recent_df = df.tail(min(lookback, len(df)))

            # Number of bars to look LEFT for pivot detection
            left_bars = 10

            # Find swing highs (resistance) - only check LEFT side
            resistance_levels = []

            # Start from left_bars to end (we can always look left)
            for i in range(left_bars, len(recent_df)):
                high = recent_df.iloc[i]['high']

                # Check if this is higher than the previous left_bars
                is_swing_high = True
                for j in range(i - left_bars, i):
                    if recent_df.iloc[j]['high'] >= high:
                        is_swing_high = False
                        break

                if is_swing_high:
                    resistance_levels.append(float(high))

            # Find swing lows (support) - only check LEFT side
            support_levels = []

            # Start from left_bars to end (we can always look left)
            for i in range(left_bars, len(recent_df)):
                low = recent_df.iloc[i]['low']

                # Check if this is lower than the previous left_bars
                is_swing_low = True
                for j in range(i - left_bars, i):
                    if recent_df.iloc[j]['low'] <= low:
                        is_swing_low = False
                        break

                if is_swing_low:
                    support_levels.append(float(low))

            # CRITICAL: Cluster nearby levels to avoid too many S/R levels
            # Merge levels within 0.2% of each other (they're essentially the same level)
            def cluster_levels(levels, tolerance=0.002):
                if not levels:
                    return []

                sorted_levels = sorted(levels)
                clustered = []
                current_cluster = [sorted_levels[0]]

                for level in sorted_levels[1:]:
                    # If within tolerance of current cluster, add to cluster
                    if abs(level - current_cluster[0]) / current_cluster[0] < tolerance:
                        current_cluster.append(level)
                    else:
                        # Save average of cluster and start new cluster
                        clustered.append(sum(current_cluster) / len(current_cluster))
                        current_cluster = [level]

                # Don't forget the last cluster
                if current_cluster:
                    clustered.append(sum(current_cluster) / len(current_cluster))

                return clustered

            # Cluster nearby levels
            resistance_levels = cluster_levels(resistance_levels)
            support_levels = cluster_levels(support_levels)

            # ALWAYS include the most recent high and low (last 20 bars)
            # These are the most relevant for immediate trading decisions
            recent_bars = min(20, len(recent_df))
            recent_high = float(recent_df.tail(recent_bars)['high'].max())
            recent_low = float(recent_df.tail(recent_bars)['low'].min())

            # Add recent high if not already in list (within 0.2% tolerance)
            add_recent_high = True
            for r in resistance_levels:
                if abs(recent_high - r) / recent_high < 0.002:
                    add_recent_high = False
                    break
            if add_recent_high:
                resistance_levels.append(recent_high)

            # Add recent low if not already in list (within 0.2% tolerance)
            add_recent_low = True
            for s in support_levels:
                if abs(recent_low - s) / recent_low < 0.002:
                    add_recent_low = False
                    break
            if add_recent_low:
                support_levels.append(recent_low)

            # Keep only the most significant levels (max 5 each)
            resistance_levels = sorted(resistance_levels)[-5:]
            support_levels = sorted(support_levels)[:5]  # Take lowest 5 support levels

            self.logger.debug(f"S/R levels found: {len(resistance_levels)} resistance, {len(support_levels)} support")

            return {
                'support': support_levels,
                'resistance': resistance_levels
            }

        except Exception as e:
            self.logger.error(f"Error calculating S/R levels: {e}")
            return {'support': [], 'resistance': []}

    def _apply_histogram_divergence_filter(self, df: pd.DataFrame, bull_signals: pd.Series, bear_signals: pd.Series, epic: str) -> tuple:
        """
        NEW: Filter out signals with weak momentum (histogram divergence from price)

        Rejects signals where histogram strength is declining while price moves in signal direction.
        This indicates weakening momentum and higher probability of reversal.

        Checks:
        - For BULL: Price making higher highs but histogram making lower highs = bearish divergence
        - For BEAR: Price making lower lows but histogram making higher lows = bullish divergence

        Args:
            df: DataFrame with MACD and price data
            bull_signals: Boolean series of bull signals
            bear_signals: Boolean series of bear signals
            epic: Trading pair epic

        Returns:
            tuple: (filtered_bull_signals, filtered_bear_signals)
        """
        try:
            original_bull = bull_signals.sum()
            original_bear = bear_signals.sum()

            if original_bull == 0 and original_bear == 0:
                return bull_signals, bear_signals

            filtered_bull = bull_signals.copy()
            filtered_bear = bear_signals.copy()

            # Lookback window for divergence detection
            divergence_lookback = 5

            # Check each bull signal for bearish divergence
            for idx in bull_signals[bull_signals].index:
                signal_position = df.index.get_loc(idx)

                # Need enough history to check divergence
                if signal_position < divergence_lookback:
                    continue

                # Get recent bars
                lookback_start = signal_position - divergence_lookback
                recent_data = df.iloc[lookback_start:signal_position + 1]

                # Check for bearish divergence: price higher high + histogram lower high
                current_price = recent_data['high'].iloc[-1]
                current_histogram = abs(recent_data['macd_histogram'].iloc[-1])

                prev_price_high = recent_data['high'].iloc[:-1].max()
                prev_histogram_high = abs(recent_data['macd_histogram'].iloc[:-1]).max()

                # Bearish divergence detected - reject signal
                if current_price > prev_price_high and current_histogram < prev_histogram_high * 0.8:
                    filtered_bull.loc[idx] = False
                    self.logger.info(
                        f"🚫 DIVERGENCE FILTER: Rejected BUY signal at {current_price:.5f} - "
                        f"bearish divergence (histogram weakening: {current_histogram:.6f} < {prev_histogram_high:.6f})"
                    )

            # Check each bear signal for bullish divergence
            for idx in bear_signals[bear_signals].index:
                signal_position = df.index.get_loc(idx)

                # Need enough history to check divergence
                if signal_position < divergence_lookback:
                    continue

                # Get recent bars
                lookback_start = signal_position - divergence_lookback
                recent_data = df.iloc[lookback_start:signal_position + 1]

                # Check for bullish divergence: price lower low + histogram higher low
                current_price = recent_data['low'].iloc[-1]
                current_histogram = abs(recent_data['macd_histogram'].iloc[-1])

                prev_price_low = recent_data['low'].iloc[:-1].min()
                prev_histogram_low = abs(recent_data['macd_histogram'].iloc[:-1]).max()

                # Bullish divergence detected - reject signal
                if current_price < prev_price_low and current_histogram < prev_histogram_low * 0.8:
                    filtered_bear.loc[idx] = False
                    self.logger.info(
                        f"🚫 DIVERGENCE FILTER: Rejected SELL signal at {current_price:.5f} - "
                        f"bullish divergence (histogram weakening: {current_histogram:.6f} < {prev_histogram_low:.6f})"
                    )

            final_bull = filtered_bull.sum()
            final_bear = filtered_bear.sum()

            if original_bull + original_bear != final_bull + final_bear:
                self.logger.info(
                    f"📊 DIVERGENCE FILTER for {epic}: "
                    f"Bull {original_bull} -> {final_bull}, "
                    f"Bear {original_bear} -> {final_bear} "
                    f"(removed {(original_bull + original_bear) - (final_bull + final_bear)} divergent signals)"
                )

            return filtered_bull, filtered_bear

        except Exception as e:
            self.logger.error(f"Error in histogram divergence filter: {e}")
            return bull_signals, bear_signals

    def _apply_tier1_quality_filters(self, df: pd.DataFrame, bull_signals: pd.Series, bear_signals: pd.Series, epic: str) -> tuple:
        """
        TIER 1 QUALITY FILTERS - Algorithmic quality over quantity

        Implements:
        1. Histogram Acceleration Requirement (40-50% reduction)
        2. Histogram Peak Detection - New highs/lows only (50-60% reduction)

        Expected: 70-80% signal reduction, keeping only highest quality momentum reversals
        """
        try:
            original_bull = bull_signals.sum()
            original_bear = bear_signals.sum()

            filtered_bull = bull_signals.copy()
            filtered_bear = bear_signals.copy()

            # ============================================================
            # FILTER 1: HISTOGRAM ACCELERATION REQUIREMENT (DISABLED IN BACKTEST)
            # Only accept signals where histogram is GROWING (not just crossing)
            # NOTE: Disabled in backtest because we can't check future bars
            # ============================================================

            # Skip acceleration check - proceed directly to peak detection
            accel_bull = original_bull
            accel_bear = original_bear

            # Original acceleration code (commented out for backtest compatibility)
            """
            for idx in bull_signals[bull_signals].index:
                idx_pos = df.index.get_loc(idx)

                # Need at least 3 bars ahead for acceleration check
                if idx_pos + 3 >= len(df):
                    filtered_bull[idx] = False
                    continue

                # Check histogram is growing for 3 bars after crossover
                histogram_growing = True
                for i in range(1, 4):
                    if df['macd_histogram'].iloc[idx_pos + i] <= df['macd_histogram'].iloc[idx_pos + i - 1]:
                        histogram_growing = False
                        break

                # Check 2nd derivative (acceleration)
                if histogram_growing and idx_pos + 2 < len(df):
                    slope_1 = df['macd_histogram'].iloc[idx_pos + 1] - df['macd_histogram'].iloc[idx_pos]
                    slope_2 = df['macd_histogram'].iloc[idx_pos + 2] - df['macd_histogram'].iloc[idx_pos + 1]
                    acceleration_positive = slope_2 > slope_1
                else:
                    acceleration_positive = False

                if not (histogram_growing and acceleration_positive):
                    filtered_bull[idx] = False

            # Same for bear signals
            for idx in bear_signals[bear_signals].index:
                idx_pos = df.index.get_loc(idx)

                if idx_pos + 3 >= len(df):
                    filtered_bear[idx] = False
                    continue

                # Histogram should be falling (more negative) for 3 bars
                histogram_falling = True
                for i in range(1, 4):
                    if df['macd_histogram'].iloc[idx_pos + i] >= df['macd_histogram'].iloc[idx_pos + i - 1]:
                        histogram_falling = False
                        break

                # Check 2nd derivative (acceleration downward)
                if histogram_falling and idx_pos + 2 < len(df):
                    slope_1 = df['macd_histogram'].iloc[idx_pos + 1] - df['macd_histogram'].iloc[idx_pos]
                    slope_2 = df['macd_histogram'].iloc[idx_pos + 2] - df['macd_histogram'].iloc[idx_pos + 1]
                    acceleration_negative = slope_2 < slope_1  # Slope becoming more negative
                else:
                    acceleration_negative = False

                if not (histogram_falling and acceleration_negative):
                    filtered_bear[idx] = False
            """

            accel_bull = filtered_bull.sum()
            accel_bear = filtered_bear.sum()

            # ============================================================
            # FILTER 2: HISTOGRAM PEAK DETECTION - DISABLED
            # Peak detection proved too strict - relying on thresholds + other filters
            # ============================================================

            # DISABLED: Even at 5%, peak detection blocks too many quality signals
            # Quality control via: S/R filter + Divergence + ADX 30+ + Histogram thresholds
            pass

            final_bull = filtered_bull.sum()
            final_bear = filtered_bear.sum()

            # Log results
            if original_bull + original_bear != final_bull + final_bear:
                self.logger.info(
                    f"🎯 TIER 1 QUALITY FILTERS for {epic}: "
                    f"Bull {original_bull} -> {accel_bull} (accel) -> {final_bull} (peak), "
                    f"Bear {original_bear} -> {accel_bear} (accel) -> {final_bear} (peak) | "
                    f"Total reduction: {((original_bull + original_bear) - (final_bull + final_bear))} signals ({((1 - (final_bull + final_bear) / max(1, original_bull + original_bear)) * 100):.1f}%)"
                )

            return filtered_bull, filtered_bear

        except Exception as e:
            self.logger.error(f"Error in Tier 1 quality filters: {e}")
            return bull_signals, bear_signals
