# core/strategies/helpers/ichimoku_mtf_analyzer.py
"""
Ichimoku Multi-Timeframe Analyzer Module
Handles multi-timeframe Ichimoku Cloud analysis and validation

MTF Analysis Features:
- Higher timeframe cloud alignment validation
- TK line trend consistency across timeframes
- Cloud color (Senkou A vs B) consistency
- Multi-timeframe Chikou span confirmation
- Confidence boosts for perfect MTF alignment
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


class IchimokuMultiTimeframeAnalyzer:
    """Handles multi-timeframe Ichimoku analysis for signal validation"""

    def __init__(self, logger: logging.Logger = None, data_fetcher=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_fetcher = data_fetcher

        # MTF configuration
        self.mtf_enabled = getattr(config, 'ICHIMOKU_MTF_ENABLED', True)
        self.check_timeframes = getattr(config, 'ICHIMOKU_MTF_TIMEFRAMES', ['15m', '1h', '4h'])
        self.cache_duration_minutes = 5  # Cache MTF data for 5 minutes

        # Simple cache for MTF data
        self.mtf_cache = {}
        self.cache_timestamps = {}

    def is_mtf_enabled(self) -> bool:
        """Check if MTF analysis is enabled and available"""
        return self.mtf_enabled and self.data_fetcher is not None

    def validate_mtf_ichimoku(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> bool:
        """
        Validate Ichimoku signal against higher timeframes

        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'

        Returns:
            True if MTF validation passes
        """
        if not self.is_mtf_enabled():
            self.logger.debug("MTF analysis disabled or data fetcher not available")
            return True  # Allow signal if MTF disabled

        try:
            mtf_results = self.get_mtf_analysis(epic, current_time, signal_type)

            if not mtf_results['mtf_enabled']:
                return True

            validation_passed = mtf_results.get('validation_passed', False)

            if validation_passed:
                aligned_count = len(mtf_results.get('timeframes_aligned', []))
                total_count = len(mtf_results.get('timeframes_checked', []))
                self.logger.info(f"✅ MTF Ichimoku validation PASSED: {aligned_count}/{total_count} timeframes aligned")
            else:
                self.logger.warning(f"❌ MTF Ichimoku validation FAILED: insufficient alignment")

            return validation_passed

        except Exception as e:
            self.logger.error(f"Error in MTF Ichimoku validation: {e}")
            return True  # Allow signal on error

    def get_mtf_analysis(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> Dict:
        """
        Comprehensive multi-timeframe Ichimoku analysis

        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Dictionary with complete MTF analysis results
        """
        if not self.is_mtf_enabled():
            return {
                'mtf_enabled': False,
                'validation_passed': True,
                'message': 'MTF analysis disabled'
            }

        try:
            results = {
                'mtf_enabled': True,
                'timeframes_checked': [],
                'timeframes_aligned': [],
                'validation_passed': False,
                'confidence_boost': 0.0,
                'cloud_alignment': {},
                'tk_alignment': {},
                'chikou_alignment': {},
                'overall_strength': 0.0,
                'details': {}
            }

            aligned_count = 0
            total_checked = 0

            # Check each higher timeframe
            for timeframe in self.check_timeframes:
                tf_result = self._analyze_timeframe_ichimoku(epic, current_time, timeframe, signal_type)
                results['timeframes_checked'].append(timeframe)
                results['details'][timeframe] = tf_result

                total_checked += 1

                if tf_result.get('aligned', False):
                    results['timeframes_aligned'].append(timeframe)
                    aligned_count += 1

                # Store specific alignments
                results['cloud_alignment'][timeframe] = tf_result.get('cloud_aligned', False)
                results['tk_alignment'][timeframe] = tf_result.get('tk_aligned', False)
                results['chikou_alignment'][timeframe] = tf_result.get('chikou_aligned', False)

            # Calculate validation result
            if total_checked > 0:
                alignment_ratio = aligned_count / total_checked
                results['alignment_ratio'] = alignment_ratio

                # Require at least 60% of timeframes to be aligned for Ichimoku
                min_alignment = getattr(config, 'ICHIMOKU_MTF_MIN_ALIGNMENT', 0.6)
                results['validation_passed'] = alignment_ratio >= min_alignment

                # Calculate confidence boost based on alignment quality
                results['confidence_boost'] = self._calculate_mtf_confidence_boost(
                    alignment_ratio, results['cloud_alignment'], results['tk_alignment']
                )

                # Calculate overall strength
                results['overall_strength'] = self._calculate_mtf_strength(results['details'])

            self.logger.debug(f"MTF Ichimoku analysis for {epic} {signal_type}: {aligned_count}/{total_checked} aligned")
            return results

        except Exception as e:
            self.logger.error(f"Error in MTF Ichimoku analysis: {e}")
            return {
                'mtf_enabled': True,
                'validation_passed': True,  # Allow signal on error
                'error': str(e)
            }

    def _analyze_timeframe_ichimoku(self, epic: str, current_time: pd.Timestamp,
                                   timeframe: str, signal_type: str) -> Dict:
        """
        Analyze Ichimoku alignment for a specific timeframe

        Args:
            epic: Trading pair epic
            current_time: Current timestamp
            timeframe: Timeframe to analyze ('15m', '1h', '4h', etc.)
            signal_type: 'BULL' or 'BEAR'

        Returns:
            Dictionary with timeframe analysis results
        """
        try:
            # Check cache first
            cache_key = f"{epic}_{timeframe}_{signal_type}_{current_time.strftime('%Y%m%d_%H%M')}"
            if self._is_cache_valid(cache_key):
                return self.mtf_cache[cache_key]

            # Extract pair from epic
            parts = epic.split('.')
            if len(parts) >= 3:
                pair = parts[2].replace('MINI', '').replace('CFD', '')
            else:
                self.logger.debug(f"Cannot extract pair from epic: {epic}")
                return {'aligned': False, 'error': 'Invalid epic format'}

            # Determine lookback based on timeframe
            lookback_hours = self._get_lookback_hours(timeframe)

            # Fetch higher timeframe data
            df_htf = self.data_fetcher.get_enhanced_data(
                epic=epic,
                pair=pair,
                timeframe=timeframe,
                lookback_hours=lookback_hours
            )

            if df_htf is None or df_htf.empty:
                self.logger.debug(f"No {timeframe} data available for {epic}")
                return {'aligned': False, 'error': 'No data available'}

            # Ensure Ichimoku indicators are calculated
            df_htf = self._ensure_htf_ichimoku(df_htf)

            # Find the most relevant candle
            latest_htf = self._get_latest_candle(df_htf, current_time)
            if latest_htf is None:
                return {'aligned': False, 'error': 'No valid candle found'}

            # Analyze Ichimoku alignment
            result = {
                'timeframe': timeframe,
                'aligned': False,
                'cloud_aligned': False,
                'tk_aligned': False,
                'chikou_aligned': False,
                'price_cloud_relation': None,
                'tk_relation': None,
                'cloud_color': None,
                'strength_score': 0.0
            }

            # Analyze cloud position
            cloud_analysis = self._analyze_cloud_position(latest_htf, signal_type)
            result.update(cloud_analysis)

            # Analyze TK lines
            tk_analysis = self._analyze_tk_lines(latest_htf, signal_type)
            result.update(tk_analysis)

            # Analyze Chikou span if available
            if len(df_htf) > 30:  # Need enough data for Chikou
                chikou_analysis = self._analyze_chikou_span(df_htf, signal_type)
                result.update(chikou_analysis)

            # Overall alignment
            alignment_components = [
                result['cloud_aligned'],
                result['tk_aligned'],
                result.get('chikou_aligned', True)  # Default True if not checked
            ]
            aligned_count = sum(alignment_components)
            result['aligned'] = aligned_count >= 2  # At least 2 out of 3 components

            # Calculate strength score
            result['strength_score'] = aligned_count / len(alignment_components)

            # Cache result
            self.mtf_cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing {timeframe} Ichimoku: {e}")
            return {'aligned': False, 'error': str(e)}

    def _ensure_htf_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Ichimoku indicators are calculated for HTF data"""
        try:
            # Import the calculator
            from .ichimoku_indicator_calculator import IchimokuIndicatorCalculator
            calculator = IchimokuIndicatorCalculator(logger=self.logger)

            # Use default Ichimoku config for HTF analysis
            ichimoku_config = {
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_b_period': 52,
                'chikou_shift': 26,
                'cloud_shift': 26
            }

            # Calculate Ichimoku components
            df_enhanced = calculator.ensure_ichimoku_indicators(df, ichimoku_config)

            return df_enhanced

        except Exception as e:
            self.logger.error(f"Error calculating HTF Ichimoku: {e}")
            return df

    def _analyze_cloud_position(self, row: pd.Series, signal_type: str) -> Dict:
        """Analyze cloud position alignment"""
        try:
            cloud_top = row.get('cloud_top', 0)
            cloud_bottom = row.get('cloud_bottom', 0)
            close = row.get('close', 0)
            senkou_a = row.get('senkou_span_a', 0)
            senkou_b = row.get('senkou_span_b', 0)

            if any(val == 0 for val in [cloud_top, cloud_bottom, close]):
                return {'cloud_aligned': False, 'price_cloud_relation': 'unknown'}

            # Determine price position
            if close > cloud_top:
                price_relation = 'above_cloud'
            elif close < cloud_bottom:
                price_relation = 'below_cloud'
            else:
                price_relation = 'in_cloud'

            # Determine cloud color
            cloud_color = 'green' if senkou_a > senkou_b else 'red'

            # Check alignment
            if signal_type == 'BULL':
                cloud_aligned = price_relation in ['above_cloud', 'in_cloud'] and \
                               (cloud_color == 'green' or price_relation == 'above_cloud')
            else:  # BEAR
                cloud_aligned = price_relation in ['below_cloud', 'in_cloud'] and \
                               (cloud_color == 'red' or price_relation == 'below_cloud')

            return {
                'cloud_aligned': cloud_aligned,
                'price_cloud_relation': price_relation,
                'cloud_color': cloud_color
            }

        except Exception as e:
            self.logger.error(f"Error analyzing cloud position: {e}")
            return {'cloud_aligned': False, 'price_cloud_relation': 'error'}

    def _analyze_tk_lines(self, row: pd.Series, signal_type: str) -> Dict:
        """Analyze Tenkan-Kijun line alignment"""
        try:
            tenkan = row.get('tenkan_sen', 0)
            kijun = row.get('kijun_sen', 0)

            if tenkan == 0 or kijun == 0:
                return {'tk_aligned': False, 'tk_relation': 'unknown'}

            # Determine TK relationship
            if tenkan > kijun:
                tk_relation = 'tenkan_above'
            elif tenkan < kijun:
                tk_relation = 'tenkan_below'
            else:
                tk_relation = 'tenkan_equal'

            # Check alignment
            if signal_type == 'BULL':
                tk_aligned = tk_relation in ['tenkan_above', 'tenkan_equal']
            else:  # BEAR
                tk_aligned = tk_relation in ['tenkan_below', 'tenkan_equal']

            return {
                'tk_aligned': tk_aligned,
                'tk_relation': tk_relation
            }

        except Exception as e:
            self.logger.error(f"Error analyzing TK lines: {e}")
            return {'tk_aligned': False, 'tk_relation': 'error'}

    def _analyze_chikou_span(self, df: pd.DataFrame, signal_type: str) -> Dict:
        """Analyze Chikou span alignment"""
        try:
            if len(df) < 30:
                return {'chikou_aligned': True}  # Skip if insufficient data

            latest_row = df.iloc[-1]
            chikou_value = latest_row.get('chikou_span', 0)

            if chikou_value == 0:
                return {'chikou_aligned': True}  # Skip if no Chikou data

            # Get historical price range (simplified version)
            historical_data = df.iloc[-30:-20]  # Look back 20-30 periods
            if len(historical_data) == 0:
                return {'chikou_aligned': True}

            historical_high = historical_data['high'].max()
            historical_low = historical_data['low'].min()

            # Check Chikou alignment
            if signal_type == 'BULL':
                chikou_aligned = chikou_value > historical_high
            else:  # BEAR
                chikou_aligned = chikou_value < historical_low

            return {'chikou_aligned': chikou_aligned}

        except Exception as e:
            self.logger.error(f"Error analyzing Chikou span: {e}")
            return {'chikou_aligned': True}  # Default to True on error

    def _calculate_mtf_confidence_boost(self, alignment_ratio: float,
                                      cloud_alignment: Dict, tk_alignment: Dict) -> float:
        """Calculate confidence boost from MTF alignment"""
        try:
            base_boost = 0.0

            # Base boost from overall alignment ratio
            if alignment_ratio >= 0.8:
                base_boost = 0.20  # Very strong MTF alignment
            elif alignment_ratio >= 0.6:
                base_boost = 0.12  # Strong MTF alignment
            elif alignment_ratio >= 0.4:
                base_boost = 0.06  # Moderate MTF alignment

            # Additional boost for perfect cloud alignment
            cloud_aligned_count = sum(cloud_alignment.values())
            if cloud_aligned_count == len(cloud_alignment):
                base_boost += 0.05  # Perfect cloud alignment bonus

            # Additional boost for perfect TK alignment
            tk_aligned_count = sum(tk_alignment.values())
            if tk_aligned_count == len(tk_alignment):
                base_boost += 0.05  # Perfect TK alignment bonus

            return min(0.25, base_boost)  # Cap at 25%

        except Exception:
            return 0.0

    def _calculate_mtf_strength(self, details: Dict) -> float:
        """Calculate overall MTF strength score"""
        try:
            if not details:
                return 0.0

            total_strength = 0.0
            timeframe_count = 0

            for timeframe, data in details.items():
                strength = data.get('strength_score', 0.0)
                total_strength += strength
                timeframe_count += 1

            return total_strength / timeframe_count if timeframe_count > 0 else 0.0

        except Exception:
            return 0.0

    def _get_latest_candle(self, df: pd.DataFrame, current_time: pd.Timestamp) -> Optional[pd.Series]:
        """Get the most relevant candle for the given time"""
        try:
            if 'start_time' in df.columns:
                # Find the candle closest to current_time
                valid_candles = df[df['start_time'] <= current_time]
                if valid_candles.empty:
                    return df.iloc[-1]  # Use last available if all are future
                else:
                    return valid_candles.iloc[-1]
            else:
                return df.iloc[-1]

        except Exception:
            return df.iloc[-1] if not df.empty else None

    def _get_lookback_hours(self, timeframe: str) -> int:
        """Get appropriate lookback hours for timeframe"""
        timeframe_lookbacks = {
            '15m': 100,    # ~4 days of 15m data
            '1h': 200,     # ~8 days of 1h data
            '4h': 400,     # ~66 days of 4h data
            '1d': 100      # ~3 months of daily data
        }
        return timeframe_lookbacks.get(timeframe, 200)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        try:
            if cache_key not in self.cache_timestamps:
                return False

            cache_time = self.cache_timestamps[cache_key]
            age_minutes = (datetime.now() - cache_time).total_seconds() / 60

            return age_minutes < self.cache_duration_minutes

        except Exception:
            return False

    def get_mtf_summary(self, epic: str, current_time: pd.Timestamp, signal_type: str) -> str:
        """Get human-readable MTF analysis summary"""
        try:
            if not self.is_mtf_enabled():
                return "MTF analysis disabled"

            analysis = self.get_mtf_analysis(epic, current_time, signal_type)

            if not analysis.get('mtf_enabled', False):
                return "MTF analysis not available"

            aligned = len(analysis.get('timeframes_aligned', []))
            total = len(analysis.get('timeframes_checked', []))
            boost = analysis.get('confidence_boost', 0)

            summary = f"MTF: {aligned}/{total} aligned"
            if boost > 0:
                summary += f" (+{boost:.1%} confidence)"

            return summary

        except Exception as e:
            return f"MTF analysis error: {str(e)}"